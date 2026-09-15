"""Step 1-3: load nuScenes radar sweeps and put them in a common frame.

Three frames matter:

    sensor  each radar's own frame, which is how the .pcd file stores points
    ego     the vehicle frame, used for output and plotting
    global  the map frame, which is what the grid indexes cells by

Positions go sensor -> ego -> global through two rigid transforms, both read
from the devkit's calibration tables rather than hand-built.

Velocities are only *rotated*, never translated. nuScenes' vx_comp/vy_comp are
already ego-motion compensated, so they are speeds over ground that merely
happen to be written in sensor-frame axes; rotating them into the global frame
gives true velocity over ground. (A velocity has no origin, so translating one
would be meaningless.)
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion

#: nuScenes ships five radars. All are fused into one point set per keyframe.
RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)

# Row indices into the 18-dimensional nuScenes radar point record.
IDX_X, IDX_Y, IDX_Z = 0, 1, 2
IDX_RCS = 5
IDX_VX_COMP, IDX_VY_COMP = 8, 9


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------

def pose_to_matrix(translation, rotation) -> np.ndarray:
    """4x4 transform from a nuScenes pose record (translation + wxyz quaternion)."""
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) positions."""
    if points.size == 0:
        return points.reshape(0, 3)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (matrix @ homogeneous.T).T[:, :3]


def invert(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    out = np.eye(4)
    rotation = matrix[:3, :3]
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ matrix[:3, 3]
    return out


# --------------------------------------------------------------------------
# one keyframe of radar
# --------------------------------------------------------------------------

@dataclass
class RadarFrame:
    """Every radar return for one keyframe, in both frames we need."""

    sample_token: str
    scene_name: str
    frame_index: int
    timestamp: float                  # seconds
    points_ego: np.ndarray            # (N, 3)
    points_global: np.ndarray         # (N, 3)
    velocity_ego: np.ndarray          # (N, 2) compensated, ego axes
    velocity_global: np.ndarray       # (N, 2) compensated, global axes
    rcs: np.ndarray                   # (N,) dBm2
    ego_to_global: np.ndarray         # (4, 4)
    global_to_ego: np.ndarray         # (4, 4)
    ego_translation: np.ndarray       # (3,) ego origin in the global frame

    @property
    def num_points(self) -> int:
        return int(self.points_ego.shape[0])

    @property
    def speed(self) -> np.ndarray:
        """(N,) compensated ground speed of each return, m/s."""
        return np.linalg.norm(self.velocity_global, axis=1)


def _load_channel(nusc, sample, channel):
    """Load one radar's returns for one sample, transformed into ego + global."""
    if channel not in sample["data"]:
        return None

    sd_token = sample["data"][channel]
    sd = nusc.get("sample_data", sd_token)

    # RadarPointCloud filtering is class-level global state in the devkit, so
    # set it explicitly rather than trusting whatever ran last. The defaults
    # drop returns nuScenes marks invalid or ambiguous.
    RadarPointCloud.default_filters()
    raw = RadarPointCloud.from_file(osp.join(nusc.dataroot, sd["filename"])).points
    if raw.shape[1] == 0:
        return None

    calibration = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    sensor_to_ego = pose_to_matrix(calibration["translation"], calibration["rotation"])

    # The ego pose recorded at this sweep's own timestamp. nuScenes keyframes
    # are synchronised across sensors, so this IS the right pose and no
    # interpolation between odometry samples is needed -- see the README.
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_to_global = pose_to_matrix(ego["translation"], ego["rotation"])

    points_ego = transform_points(raw[[IDX_X, IDX_Y, IDX_Z], :].T, sensor_to_ego)

    # Lift the planar velocity to 3D only so the same rotation applies, then
    # drop z again.
    velocity = np.zeros((raw.shape[1], 3))
    velocity[:, 0] = raw[IDX_VX_COMP, :]
    velocity[:, 1] = raw[IDX_VY_COMP, :]
    velocity_ego = (sensor_to_ego[:3, :3] @ velocity.T).T

    return {
        "points_ego": points_ego,
        "points_global": transform_points(points_ego, ego_to_global),
        "velocity_ego": velocity_ego[:, :2],
        "velocity_global": (ego_to_global[:3, :3] @ velocity_ego.T).T[:, :2],
        "rcs": raw[IDX_RCS, :],
        "ego_to_global": ego_to_global,
        "timestamp": sd["timestamp"] / 1e6,
    }


def load_frame(nusc, sample, scene_name, frame_index) -> RadarFrame:
    """Fuse all five radars of one keyframe into a single RadarFrame."""
    parts = [p for p in (_load_channel(nusc, sample, c) for c in RADAR_CHANNELS)
             if p is not None]
    if not parts:
        raise ValueError(f"sample {sample['token']} has no usable radar returns")

    def stack(key):
        return np.concatenate([p[key] for p in parts], axis=0)

    ego_to_global = parts[0]["ego_to_global"]   # shared by all channels
    return RadarFrame(
        sample_token=sample["token"],
        scene_name=scene_name,
        frame_index=frame_index,
        timestamp=float(np.mean([p["timestamp"] for p in parts])),
        points_ego=stack("points_ego"),
        points_global=stack("points_global"),
        velocity_ego=stack("velocity_ego"),
        velocity_global=stack("velocity_global"),
        rcs=stack("rcs"),
        ego_to_global=ego_to_global,
        global_to_ego=invert(ego_to_global),
        ego_translation=ego_to_global[:3, 3].copy(),
    )


def iter_scene_frames(nusc, scene_name, max_frames=None):
    """Yield RadarFrames in order for a named scene, e.g. "scene-0061"."""
    scene = next((s for s in nusc.scene
                  if scene_name in (s["name"], s["token"])), None)
    if scene is None:
        available = ", ".join(s["name"] for s in nusc.scene)
        raise ValueError(f"scene {scene_name!r} not found. Available: {available}")

    token, index = scene["first_sample_token"], 0
    while token and (max_frames is None or index < max_frames):
        sample = nusc.get("sample", token)
        yield load_frame(nusc, sample, scene["name"], index)
        token, index = sample["next"], index + 1
