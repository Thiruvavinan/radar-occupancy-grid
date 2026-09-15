"""Rigid transforms between sensor, ego and global frames.

Coordinate convention is **ISO 8855** (x forward, y left, z up), which is what
nuScenes already uses for the ego frame. That was verified against the data
rather than assumed -- see `verify_convention` below and the README.
"""

from __future__ import annotations

import numpy as np
from pyquaternion import Quaternion


def pose_to_matrix(translation, rotation) -> np.ndarray:
    """4x4 transform from a nuScenes pose record (translation + wxyz quaternion).

    Maps points from the child frame into the parent frame.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) positions."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 3)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (matrix @ homogeneous.T).T[:, :3]


def rotate_vectors(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) vectors by a transform's rotation part only.

    Velocities are rotated, never translated: nuScenes' ``vx_comp``/``vy_comp``
    are already ego-motion compensated, so they are speeds over ground that
    merely happen to be written in sensor-frame axes.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.size == 0:
        return vectors.reshape(0, 3)
    return (matrix[:3, :3] @ vectors.T).T


def invert(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    out = np.eye(4)
    rotation = matrix[:3, :3]
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ matrix[:3, 3]
    return out


def sensor_to_ego(nusc, sample_data_token: str) -> np.ndarray:
    """Calibrated extrinsics: this sensor's frame into the ego frame."""
    sd = nusc.get("sample_data", sample_data_token)
    calibration = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return pose_to_matrix(calibration["translation"], calibration["rotation"])


def verify_convention(nusc, scene_name="scene-0061") -> dict:
    """Check empirically that the ego frame really is x-forward / y-left.

    Two independent checks, because assuming this and being wrong would put a
    silent rotation error through the whole pipeline:

      1. the ego frame's +x axis should point along the direction of travel
      2. the front radar should sit at positive x, and the left radar at
         positive y, in their calibrated extrinsics

    Returns the measured numbers so the README can quote them instead of
    asserting the convention.
    """
    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    poses, token = [], scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        poses.append((np.array(ego["translation"]), Quaternion(ego["rotation"])))
        token = sample["next"]

    alignment = []
    for (position, rotation), (next_position, _) in zip(poses[:-1], poses[1:]):
        step = (next_position - position)[:2]
        if np.linalg.norm(step) < 0.5:          # ignore near-stationary frames
            continue
        forward = (rotation.rotation_matrix @ np.array([1.0, 0.0, 0.0]))[:2]
        alignment.append(float(np.dot(step / np.linalg.norm(step),
                                      forward / np.linalg.norm(forward))))

    sample = nusc.sample[0]
    mounts = {}
    for channel in ("RADAR_FRONT", "RADAR_FRONT_LEFT"):
        sd = nusc.get("sample_data", sample["data"][channel])
        calibration = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        mounts[channel] = calibration["translation"]

    alignment = np.array(alignment)
    return {
        "frames_tested": int(alignment.size),
        "mean_cos_x_vs_travel": float(alignment.mean()),
        "min_cos_x_vs_travel": float(alignment.min()),
        "radar_front_x": float(mounts["RADAR_FRONT"][0]),
        "radar_front_left_y": float(mounts["RADAR_FRONT_LEFT"][1]),
        # A mean cosine of ~1.0 means +x IS forward; a front sensor at positive
        # x and a left sensor at positive y confirm it independently.
        "rotation_correction_needed": bool(alignment.mean() < 0.9),
    }
