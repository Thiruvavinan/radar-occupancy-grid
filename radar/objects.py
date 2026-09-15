"""Turn cell clusters into detections in the nuScenes submission format.

Boxes are emitted in the **global frame**, which is what `DetectionEval`
expects.

Where each field comes from, and what is honestly known versus assumed:

    translation   x, y from the cluster's cells; **z is fixed at 0.0**.
                  Automotive radar does not resolve height usefully, so this is
                  a placeholder, not an estimate.
    size          width and length from the cluster's spatial extent;
                  **height is hardcoded to 1.5 m** for the same reason.
    rotation      yaw from the velocity vector when the object is moving;
                  identity when it is not, since the yaw of a ~zero vector is
                  meaningless. Radar gives no orientation of its own.
    velocity      mean compensated velocity of the CURRENT cycle's returns in
                  the cluster -- not the grid's running mean, which lags and
                  would report a stopped car as still moving.
    score         the cluster's mean occupancy probability.
    class         a velocity rule only. See CLASSIFICATION_NOTE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import cKDTree

from radar.occupancy_grid import MOVING_SPEED

#: The pipeline has **no semantic information whatsoever**. RCS and Doppler
#: velocity cannot separate a car from a truck from a pedestrian without a
#: learned classifier. The most that is honestly justifiable is a velocity
#: split, so moving clusters are emitted as `car` and stationary ones as
#: `barrier` -- the two official nuScenes classes that come closest to
#: "generic mover" and "generic roadside object".
#:
#: Consequence, stated before any results are shown: scored against the full
#: 10-class nuScenes benchmark, the eight classes this pipeline never emits
#: score exactly zero by construction. That is a scope limitation, not a
#: detector failure.
#: Two rules are implemented, and which one is used changes the official mAP
#: enormously -- for reasons that are about the label mapping, not about
#: detection quality:
#:
#:   "velocity"  moving -> car, stationary -> barrier. The intuitive reading of
#:               "generic mover / generic static object", and **empirically the
#:               wrong mapping**: mini_val contains zero `barrier` ground-truth
#:               boxes, so every stationary detection is unmatchable by
#:               construction, and only the handful of moving detections can
#:               score at all.
#:
#:   "all_car"   everything -> car (the default). With no semantic information,
#:               the maximum-likelihood class is the most common one, and cars
#:               are 2056 of the 3793 eval-eligible ground-truth boxes in
#:               mini_val. A parked car is still a car, so labelling a
#:               stationary vehicle `barrier` is not conservative, it is
#:               incorrect.
#:
#: Neither rule is classification in any real sense. Both are stated here so
#: the mAP difference between them is readable as what it is: a property of the
#: label mapping.
CLASS_RULES = ("all_car", "velocity")

CLASSIFICATION_NOTE = (
    "no semantic information: radar RCS and Doppler cannot separate car from "
    "truck from pedestrian. Default rule labels every detection 'car'; "
    "8 of 10 nuScenes classes are never predicted and score 0 by construction"
)

MOVING_CLASS = "car"
STATIC_CLASS = "barrier"

#: Radar cannot measure height. Both of these are declared constants rather
#: than estimates, so nothing downstream mistakes them for measurements.
FIXED_Z = 0.0
FIXED_HEIGHT = 1.5

#: Floor on a box's footprint. A one-cell cluster would otherwise produce a
#: 0.5 x 0.5 m box, far below any real vehicle and guaranteed to fail scale
#: matching.
MIN_EXTENT = 1.5


@dataclass
class RadarDetection:
    """One detection, ready to serialise into the submission JSON."""

    sample_token: str
    translation: list                 # [x, y, z] global
    size: list                        # [width, length, height]
    rotation: list                    # [w, x, y, z] quaternion, global
    velocity: list                    # [vx, vy] global
    detection_name: str
    detection_score: float
    attribute_name: str = ""

    # Kept for plotting and analysis; not part of the submission schema.
    num_cells: int = 0
    position_ego: tuple = (0.0, 0.0)
    velocity_ego: tuple = (0.0, 0.0)

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    @property
    def is_moving(self) -> bool:
        return self.speed > MOVING_SPEED

    def to_submission(self) -> dict:
        """Exactly the keys DetectionBox.deserialize reads, nothing else."""
        return {
            "sample_token": self.sample_token,
            "translation": [float(v) for v in self.translation],
            "size": [float(v) for v in self.size],
            "rotation": [float(v) for v in self.rotation],
            "velocity": [float(v) for v in self.velocity],
            "detection_name": self.detection_name,
            "detection_score": float(self.detection_score),
            "attribute_name": self.attribute_name,
        }


def _yaw_to_quaternion(yaw: float) -> list:
    """Rotation about +z (up), as [w, x, y, z]."""
    return list(Quaternion(axis=[0.0, 0.0, 1.0], angle=float(yaw)).elements)


def _current_cycle_velocity(cycle, cell_centers, radius, cluster_is_dynamic):
    """Mean velocity of THIS cycle's returns lying on a cluster's cells.

    Returns must also agree with the cluster's motion class, or a moving
    vehicle beside a wall picks up the wall's stationary returns and averages
    to a standstill.
    """
    if cycle.num_points == 0 or cell_centers.shape[0] == 0:
        return None

    distances, _ = cKDTree(cell_centers).query(
        cycle.points_global[:, :2], k=1, distance_upper_bound=radius)
    nearby = np.isfinite(distances)

    moving = cycle.speed >= MOVING_SPEED
    consistent = moving if cluster_is_dynamic else ~moving
    selected = np.flatnonzero(nearby & consistent)
    if selected.size == 0:
        selected = np.flatnonzero(nearby)
    return selected if selected.size else None


def _detection_name(moving: bool, class_rule: str) -> str:
    if class_rule == "all_car":
        return MOVING_CLASS
    return MOVING_CLASS if moving else STATIC_CLASS


def build_detections(cycle, snapshot, labels, resolution=0.5,
                     class_rule="all_car") -> list:
    """One RadarDetection per cluster, most confident first."""
    if class_rule not in CLASS_RULES:
        raise ValueError(f"class_rule must be one of {CLASS_RULES}")
    detections = []
    radius = resolution * 3.0

    for label in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == label)
        centers = snapshot.centers_global[members]
        confidence = snapshot.p_exist[members]
        weights = confidence / confidence.sum()

        cell_speeds = np.linalg.norm(snapshot.mean_velocity[members], axis=1)
        is_dynamic = bool(np.median(cell_speeds) >= MOVING_SPEED)

        selected = _current_cycle_velocity(cycle, centers, radius, is_dynamic)
        if selected is not None:
            velocity = cycle.velocity_global[selected].mean(axis=0)
        else:
            velocity = (snapshot.mean_velocity[members] * weights[:, None]).sum(0)

        centre = (centers * weights[:, None]).sum(axis=0)
        span = centers.max(axis=0) - centers.min(axis=0) + resolution
        width, length = (max(float(span[1]), MIN_EXTENT),
                         max(float(span[0]), MIN_EXTENT))

        speed = float(np.hypot(*velocity))
        moving = speed > MOVING_SPEED
        # Yaw from the velocity vector only when there is a velocity to speak
        # of; otherwise identity, because atan2 of a ~zero vector is noise.
        rotation = (_yaw_to_quaternion(np.arctan2(velocity[1], velocity[0]))
                    if moving else _yaw_to_quaternion(0.0))

        ego_xy = cycle.global_to_ego[:2, :2] @ centre + cycle.global_to_ego[:2, 3]
        ego_v = cycle.global_to_ego[:2, :2] @ velocity

        detections.append(RadarDetection(
            sample_token=cycle.sample_token,
            translation=[float(centre[0]), float(centre[1]), FIXED_Z],
            size=[width, length, FIXED_HEIGHT],
            rotation=rotation,
            velocity=[float(velocity[0]), float(velocity[1])],
            detection_name=_detection_name(moving, class_rule),
            detection_score=float((confidence * weights).sum()),
            attribute_name="vehicle.moving" if moving else "",
            num_cells=int(members.size),
            position_ego=(float(ego_xy[0]), float(ego_xy[1])),
            velocity_ego=(float(ego_v[0]), float(ego_v[1])),
        ))

    detections.sort(key=lambda d: d.detection_score, reverse=True)
    return detections
