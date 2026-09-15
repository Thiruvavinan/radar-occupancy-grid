"""Load all five radars for one keyframe and put them in one common frame.

Per processing cycle (one nuScenes keyframe):

  1. load each radar's sweep with the devkit's ``RadarPointCloud``
  2. transform sensor -> ego using that radar's calibrated extrinsics
  3. transform ego -> global using the ego pose **at that radar's own
     timestamp**, from the SLERP interpolator
  4. transform global -> ego at the shared reference time

Step 3 is the one that matters: the radars fire up to ~34 ms apart, so using a
single keyframe pose for all five would misplace them relative to each other.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

from radar.transform import (
    invert,
    rotate_vectors,
    sensor_to_ego,
    transform_points,
)

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
#: Ego-motion-compensated velocity. Used in preference to the raw radial
#: velocity (rows 6-7), which is uncompensated and mixes in the ego's own
#: motion. Documented in the README as the field actually used.
IDX_VX_COMP, IDX_VY_COMP = 8, 9


@dataclass
class RadarCycle:
    """All five radars for one keyframe, pose-compensated to a common time."""

    sample_token: str
    scene_name: str
    frame_index: int
    reference_timestamp: int          # microseconds; the keyframe time
    points_ego: np.ndarray            # (N, 3) at the reference time
    points_global: np.ndarray         # (N, 3)
    velocity_ego: np.ndarray          # (N, 2) compensated, ego axes
    velocity_global: np.ndarray       # (N, 2) compensated, global axes
    rcs: np.ndarray                   # (N,) dBm2
    channel_index: np.ndarray         # (N,) which radar each return came from
    ego_to_global: np.ndarray         # (4, 4) at the reference time
    global_to_ego: np.ndarray         # (4, 4)
    ego_translation: np.ndarray       # (3,)
    timestamp_spread_ms: float        # how far apart the five sweeps were

    @property
    def num_points(self) -> int:
        return int(self.points_ego.shape[0])

    @property
    def timestamp(self) -> float:
        """Reference time in seconds, for the grid's aging arithmetic."""
        return self.reference_timestamp / 1e6

    @property
    def speed(self) -> np.ndarray:
        """(N,) compensated ground speed of each return, m/s."""
        return np.linalg.norm(self.velocity_global, axis=1)


def _load_channel(nusc, sample, channel, interpolator):
    """One radar's returns, transformed into the global frame.

    The global frame is the meeting point: each radar reaches it through its
    own pose at its own timestamp, after which they are directly comparable.
    """
    if channel not in sample["data"]:
        return None

    sd_token = sample["data"][channel]
    sd = nusc.get("sample_data", sd_token)

    # Devkit filtering is class-level global state, so set it every load rather
    # than trusting whatever ran last. The defaults drop returns nuScenes marks
    # invalid or ambiguous.
    RadarPointCloud.default_filters()
    raw = RadarPointCloud.from_file(osp.join(nusc.dataroot, sd["filename"])).points
    if raw.shape[1] == 0:
        return None

    to_ego = sensor_to_ego(nusc, sd_token)
    # The pose at THIS radar's capture time, not the keyframe's.
    to_global = interpolator.matrix_at(sd["timestamp"])

    points_ego = transform_points(raw[[IDX_X, IDX_Y, IDX_Z], :].T, to_ego)

    velocity = np.zeros((raw.shape[1], 3))
    velocity[:, 0] = raw[IDX_VX_COMP, :]
    velocity[:, 1] = raw[IDX_VY_COMP, :]
    velocity_ego = rotate_vectors(velocity, to_ego)

    return {
        "points_global": transform_points(points_ego, to_global),
        "velocity_global": rotate_vectors(velocity_ego, to_global)[:, :2],
        "rcs": raw[IDX_RCS, :],
        "timestamp": sd["timestamp"],
    }


def load_cycle(nusc, sample, scene_name, frame_index, interpolator,
               channels=RADAR_CHANNELS) -> RadarCycle:
    """Fuse all five radars of one keyframe into a single pose-compensated set."""
    parts, channel_ids, timestamps = [], [], []
    for index, channel in enumerate(channels):
        loaded = _load_channel(nusc, sample, channel, interpolator)
        if loaded is None:
            continue
        parts.append(loaded)
        channel_ids.append(np.full(loaded["rcs"].shape[0], index, dtype=int))
        timestamps.append(loaded["timestamp"])

    if not parts:
        raise ValueError(f"sample {sample['token']} has no usable radar returns")

    points_global = np.concatenate([p["points_global"] for p in parts])
    velocity_global = np.concatenate([p["velocity_global"] for p in parts])

    # The shared reference time: the keyframe's own timestamp, which is also
    # LIDAR_TOP's, so a pose record exists there exactly.
    reference = int(sample["timestamp"])
    ego_to_global = interpolator.matrix_at(reference)
    global_to_ego = invert(ego_to_global)

    velocity_ego = rotate_vectors(
        np.hstack([velocity_global, np.zeros((velocity_global.shape[0], 1))]),
        global_to_ego)[:, :2]

    return RadarCycle(
        sample_token=sample["token"],
        scene_name=scene_name,
        frame_index=frame_index,
        reference_timestamp=reference,
        points_ego=transform_points(points_global, global_to_ego),
        points_global=points_global,
        velocity_ego=velocity_ego,
        velocity_global=velocity_global,
        rcs=np.concatenate([p["rcs"] for p in parts]),
        channel_index=np.concatenate(channel_ids),
        ego_to_global=ego_to_global,
        global_to_ego=global_to_ego,
        ego_translation=ego_to_global[:3, 3].copy(),
        timestamp_spread_ms=(max(timestamps) - min(timestamps)) / 1000.0,
    )


def iter_scene(nusc, scene_name, max_frames=None, channels=RADAR_CHANNELS):
    """Yield (RadarCycle, interpolator) for each keyframe of a scene."""
    from radar.sync import EgoPoseInterpolator

    scene = find_scene(nusc, scene_name)
    interpolator = EgoPoseInterpolator(nusc, scene["token"])

    token, index = scene["first_sample_token"], 0
    while token and (max_frames is None or index < max_frames):
        sample = nusc.get("sample", token)
        yield load_cycle(nusc, sample, scene["name"], index, interpolator, channels)
        token, index = sample["next"], index + 1


def find_scene(nusc, scene_name):
    scene = next((s for s in nusc.scene
                  if scene_name in (s["name"], s["token"])), None)
    if scene is None:
        available = ", ".join(s["name"] for s in nusc.scene)
        raise ValueError(f"scene {scene_name!r} not found. Available: {available}")
    return scene
