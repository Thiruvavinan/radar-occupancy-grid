"""Steps 6-7: cluster confident cells, then build objects from the clusters.

Note what gets clustered: **grid cells that survived the confidence filter**,
not raw radar returns. That is the whole reason the grid exists. By the time
DBSCAN runs, each input has been seen across frames, quantised to the lattice
and confidence-screened, so it is far better conditioned than a raw sweep.

The split of sources when building an object is deliberate and is the crux of
the design:

    position, extent, RCS, confidence   from the ACCUMULATED grid
    velocity                            from the CURRENT frame only

A running-mean velocity lags reality and would keep reporting a vehicle as
moving several frames after it stops -- and braking is exactly when a
downstream consumer most needs the truth. Current-frame velocity is noisier and
is still the right choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from radar.grid import MOVING_SPEED


@dataclass
class DetectConfig:
    eps: float = 2.0          # metres; two cells at 1 m resolution
    min_samples: int = 1      # a single confident cell may stand alone

    # Cluster stationary and moving cells as two separate populations. A car
    # passing a row of parked cars is spatially adjacent to them, so plain 2D
    # DBSCAN merges the two and the merged velocity averages to near zero,
    # destroying the static/dynamic distinction the pipeline exists to make.
    separate_by_motion: bool = True


@dataclass
class RadarObject:
    """One object candidate built from a cluster of confident cells."""

    position_ego: np.ndarray          # (2,) confidence-weighted centroid
    position_global: np.ndarray       # (2,)
    extent_ego: np.ndarray            # (4,) xmin, ymin, xmax, ymax
    num_cells: int
    total_hits: int
    existence_confidence: float
    rcs_mean: float
    mean_z: float
    velocity_ego: np.ndarray          # (2,) m/s
    velocity_global: np.ndarray       # (2,) m/s
    velocity_source: str              # "current_frame" or "accumulated"
    frame_index: int = -1
    sample_token: str = ""

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity_global))

    @property
    def is_moving(self) -> bool:
        return self.speed > MOVING_SPEED

    @property
    def range_from_ego(self) -> float:
        return float(np.linalg.norm(self.position_ego))

    def as_dict(self) -> dict:
        r3 = lambda v: [round(float(x), 3) for x in v]
        return {
            "frame_index": int(self.frame_index),
            "sample_token": self.sample_token,
            "position_ego": r3(self.position_ego),
            "position_global": r3(self.position_global),
            "extent_ego": r3(self.extent_ego),
            "range_from_ego": round(self.range_from_ego, 3),
            "num_cells": int(self.num_cells),
            "total_hits": int(self.total_hits),
            "existence_confidence": round(self.existence_confidence, 4),
            "rcs_mean": round(self.rcs_mean, 3),
            "mean_z": round(self.mean_z, 3),
            "velocity_ego": r3(self.velocity_ego),
            "velocity_global": r3(self.velocity_global),
            "speed": round(self.speed, 3),
            "is_moving": bool(self.is_moving),
            "velocity_source": self.velocity_source,
        }


def cluster_cells(snapshot, config: DetectConfig) -> np.ndarray:
    """(M,) cluster label per cell, -1 for noise.

    ``min_samples=1`` is unusual and deliberate: with 2, DBSCAN discards
    isolated confident cells as noise, which costs ~20 points of coverage.
    Radar is sparse enough that a cell which already cleared both the hit-count
    and probability bars is real evidence, and throwing it away for lacking a
    neighbour discards the grid's own conclusion. Noise rejection belongs in
    the confidence filter, not here.
    """
    if snapshot.is_empty:
        return np.zeros(0, dtype=int)

    if not config.separate_by_motion:
        return _dbscan(snapshot.centers_global, snapshot.p_exist, config)

    is_dynamic = np.linalg.norm(snapshot.mean_velocity, axis=1) >= MOVING_SPEED
    labels = np.full(len(snapshot), -1, dtype=int)
    next_label = 0

    for mask in (~is_dynamic, is_dynamic):
        members = np.flatnonzero(mask)
        if members.size == 0:
            continue
        sub = _dbscan(snapshot.centers_global[members],
                      snapshot.p_exist[members], config)
        found = sub >= 0
        if not found.any():
            continue
        labels[members[found]] = sub[found] + next_label
        next_label += int(sub[found].max()) + 1

    return labels


def _dbscan(centers, confidence, config) -> np.ndarray:
    """DBSCAN on one population, weighting each cell by its confidence."""
    if centers.shape[0] < config.min_samples:
        return np.full(centers.shape[0], -1, dtype=int)
    return DBSCAN(eps=config.eps, min_samples=config.min_samples).fit_predict(
        centers, sample_weight=confidence).astype(int)


def _current_frame_velocity(frame, cell_centers, radius, cluster_is_dynamic):
    """Mean velocity of this frame's returns lying on a cluster's cells.

    Returns must also agree with the cluster's motion class. Without that, a
    moving vehicle beside a wall picks up the wall's stationary returns and
    averages to a stop -- the exact failure this pipeline must not make.
    """
    if frame.num_points == 0 or cell_centers.shape[0] == 0:
        return None

    distances, _ = cKDTree(cell_centers).query(
        frame.points_global[:, :2], k=1, distance_upper_bound=radius)
    nearby = np.isfinite(distances)

    moving = frame.speed >= MOVING_SPEED
    consistent = moving if cluster_is_dynamic else ~moving
    selected = np.flatnonzero(nearby & consistent)
    if selected.size == 0:
        # Motion agreement left nothing; fall back to plain proximity rather
        # than losing the velocity entirely.
        selected = np.flatnonzero(nearby)
    return selected if selected.size else None


def build_objects(frame, snapshot, labels, config: DetectConfig,
                  resolution=1.0) -> list:
    """One RadarObject per cluster, most confident first."""
    objects = []
    radius = resolution * 1.5      # a return just off the cluster still counts

    for label in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == label)
        centers_global = snapshot.centers_global[members]
        centers_ego = snapshot.centers_ego[members]
        confidence = snapshot.p_exist[members]

        # Confidence-weighted centroid: a cell we are sure about should pull
        # the position harder than one that barely cleared threshold.
        weights = confidence / confidence.sum()
        cell_speeds = np.linalg.norm(snapshot.mean_velocity[members], axis=1)
        is_dynamic = bool(np.median(cell_speeds) >= MOVING_SPEED)

        selected = _current_frame_velocity(frame, centers_global, radius, is_dynamic)
        if selected is not None:
            velocity_global = frame.velocity_global[selected].mean(axis=0)
            velocity_ego = frame.velocity_ego[selected].mean(axis=0)
            source = "current_frame"
        else:
            velocity_global = (snapshot.mean_velocity[members] * weights[:, None]).sum(0)
            velocity_ego = frame.global_to_ego[:2, :2] @ velocity_global
            source = "accumulated"

        half = resolution / 2
        objects.append(RadarObject(
            position_ego=(centers_ego * weights[:, None]).sum(axis=0),
            position_global=(centers_global * weights[:, None]).sum(axis=0),
            extent_ego=np.array([centers_ego[:, 0].min() - half,
                                 centers_ego[:, 1].min() - half,
                                 centers_ego[:, 0].max() + half,
                                 centers_ego[:, 1].max() + half]),
            num_cells=int(members.size),
            total_hits=int(snapshot.hit_count[members].sum()),
            existence_confidence=float((confidence * weights).sum()),
            rcs_mean=float(snapshot.mean_rcs[members].mean()),
            mean_z=float(snapshot.mean_z[members].mean()),
            velocity_ego=np.asarray(velocity_ego),
            velocity_global=np.asarray(velocity_global),
            velocity_source=source,
            frame_index=frame.frame_index,
            sample_token=frame.sample_token,
        ))

    objects.sort(key=lambda o: o.existence_confidence, reverse=True)
    return objects


def detect(frame, grid, config: DetectConfig):
    """Steps 5-7 in one call: filter, cluster, build. Returns (cells, objects)."""
    confident = grid.snapshot(frame, filtered=True)
    labels = cluster_cells(confident, config)
    objects = build_objects(frame, confident, labels, config,
                            resolution=grid.config.resolution)
    return confident, objects
