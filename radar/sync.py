"""SLERP ego-pose interpolation, for aligning five asynchronous radars.

The five nuScenes radars do **not** fire together. Measured on the mini split,
their sweeps land between -31 ms and +3 ms around the keyframe time -- a spread
of ~34 ms, which at 15 m/s is half a metre of ego travel. Combining them into
one point set without compensating for that would smear every static object
across cells and undermine the accumulation the grid depends on.

So each radar is pose-compensated at **its own capture time**:

    p_global = T_ego->global(t_radar) . T_sensor->ego . p_sensor
    p_ref    = T_global->ego(t_ref)  . p_global

Getting ``T_ego->global(t)`` at an arbitrary ``t`` is what this module does:
linear interpolation on translation, **spherical linear interpolation (SLERP)**
on the rotation quaternion. SLERP is the right tool because rotations live on
the unit quaternion sphere -- componentwise lerp of two quaternions leaves the
unit sphere and produces a non-uniform angular rate, whereas SLERP traces the
shortest arc at constant angular velocity.

Worth being precise about what this buys on nuScenes specifically: the ego_pose
table is dense (~5.5 ms median spacing) and **every sample_data record already
points at a pose stamped with its own exact timestamp**, so for radar sweeps the
lookup is usually exact and interpolation is a no-op. The interpolation still
matters for resolving an arbitrary reference time, and it is what a production
system with genuinely asynchronous sensors would need. The README states this
plainly rather than overselling it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion

from radar.transform import pose_to_matrix


@dataclass
class Pose:
    timestamp: float                  # microseconds, as nuScenes stores it
    translation: np.ndarray           # (3,)
    rotation: Quaternion

    def as_matrix(self) -> np.ndarray:
        return pose_to_matrix(self.translation, self.rotation.elements)


class EgoPoseInterpolator:
    """Ego poses for one scene, queryable at any timestamp.

    Built once per scene; the poses are sorted so a query is a binary search.
    """

    def __init__(self, nusc, scene_token: str):
        poses = _scene_pose_records(nusc, scene_token)
        if not poses:
            raise ValueError("no ego poses found for this scene")

        self.timestamps = np.array([p["timestamp"] for p in poses], dtype=np.int64)
        self.translations = np.array([p["translation"] for p in poses], dtype=float)
        self.rotations = [Quaternion(p["rotation"]) for p in poses]

        # How often an exact record exists, reported by the README.
        self.exact_hits = 0
        self.interpolated = 0
        self.clamped = 0

    def __len__(self) -> int:
        return int(self.timestamps.size)

    def pose_at(self, timestamp) -> Pose:
        """Ego pose at ``timestamp``, interpolating between bracketing records.

        Exact matches are returned untouched -- interpolating onto a timestamp
        that already has a record would only add floating-point noise.
        """
        timestamp = int(timestamp)
        index = int(np.searchsorted(self.timestamps, timestamp))

        # Exact hit, either side of the insertion point.
        if index < len(self) and self.timestamps[index] == timestamp:
            self.exact_hits += 1
            return Pose(timestamp, self.translations[index], self.rotations[index])

        # Outside the recorded range: clamp to the nearest end rather than
        # extrapolating, which would invent motion we have no evidence for.
        if index == 0 or index >= len(self):
            self.clamped += 1
            edge = 0 if index == 0 else len(self) - 1
            return Pose(timestamp, self.translations[edge], self.rotations[edge])

        before, after = index - 1, index
        span = self.timestamps[after] - self.timestamps[before]
        ratio = 0.0 if span == 0 else (timestamp - self.timestamps[before]) / span

        translation = ((1.0 - ratio) * self.translations[before]
                       + ratio * self.translations[after])
        rotation = Quaternion.slerp(self.rotations[before], self.rotations[after],
                                    amount=float(ratio))

        self.interpolated += 1
        return Pose(timestamp, translation, rotation)

    def matrix_at(self, timestamp) -> np.ndarray:
        """4x4 ego -> global transform at ``timestamp``."""
        return self.pose_at(timestamp).as_matrix()

    def stats(self) -> dict:
        total = self.exact_hits + self.interpolated + self.clamped
        return {
            "pose_records": len(self),
            "median_gap_ms": float(np.median(np.diff(self.timestamps)) / 1000.0),
            "queries": total,
            "exact": self.exact_hits,
            "interpolated": self.interpolated,
            "clamped": self.clamped,
        }


def _scene_pose_records(nusc, scene_token: str):
    """Every ego_pose belonging to one scene, sorted by timestamp.

    Poses are reached through sample_data rather than the global ego_pose table,
    because that table spans every scene in the split and poses from a different
    log must never be interpolated against.
    """
    seen, poses = set(), []
    for sd in nusc.sample_data:
        if sd["ego_pose_token"] in seen:
            continue
        sample = nusc.get("sample", sd["sample_token"])
        if sample["scene_token"] != scene_token:
            continue
        seen.add(sd["ego_pose_token"])
        poses.append(nusc.get("ego_pose", sd["ego_pose_token"]))

    return sorted(poses, key=lambda p: p["timestamp"])


def measure_radar_async(nusc, sample, channels) -> dict:
    """How far each radar's sweep sits from the keyframe time, in ms.

    This is the measurement that justifies the whole module; the README quotes
    it rather than asserting that the radars are asynchronous.
    """
    offsets = {}
    for channel in channels:
        if channel not in sample["data"]:
            continue
        sd = nusc.get("sample_data", sample["data"][channel])
        offsets[channel] = (sd["timestamp"] - sample["timestamp"]) / 1000.0
    values = list(offsets.values())
    return {
        "offsets_ms": offsets,
        "spread_ms": float(max(values) - min(values)) if values else 0.0,
    }
