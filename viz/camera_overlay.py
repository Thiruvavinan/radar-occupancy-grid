"""Project radar detections onto the camera image. Visualisation only.

**This is not fusion.** Nothing here associates radar with image content,
combines detections, or produces a fused output -- it is a sanity check that
radar objects land in plausible image locations. No metric is attached to it.
Real multi-sensor fusion is Stage 2.

The projection chain is global -> ego -> camera -> image:

    p_ego    = T_global->ego(t_cam) . p_global
    p_cam    = T_ego->cam           . p_ego          (inverse of the extrinsics)
    u, v     = K . p_cam / z_cam                     (pinhole, z_cam > 0 only)

Note ``t_cam``, not the keyframe time. The camera has its own timestamp and its
own ego pose -- CAM_FRONT fires ~34 ms before the keyframe -- so projecting
through the keyframe pose puts every box slightly wrong. Using the keyframe pose
here was a real bug, worth up to 61 px on a near object even in a frame where
the ego had moved only 6 cm; in a fast scene it would be far worse. This is the
same asynchrony handled for the radars in radar/sync.py.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from radar.occupancy_grid import MOVING_SPEED
from radar.transform import invert, pose_to_matrix

MOVING_COLOR = "#d1495b"
STATIC_COLOR = "#2a9d8f"
GT_COLOR = "#e9c46a"


def _unit_corners(wlh):
    """(3, 8) corners of a box of size (width, length, height) at the origin."""
    width, length, height = wlh
    return np.vstack([
        np.array([1, 1, 1, 1, -1, -1, -1, -1]) * length / 2,
        np.array([1, -1, -1, 1, 1, -1, -1, 1]) * width / 2,
        np.array([1, 1, -1, -1, 1, 1, -1, -1]) * height / 2,
    ])


def _box_corners_global(detection):
    """The 8 corners of a detection's 3D box, in the global frame."""
    width, length, height = detection.size
    x = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * length / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * width / 2
    z = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * height / 2
    corners = np.vstack([x, y, z])

    rotation = Quaternion(detection.rotation).rotation_matrix
    centre = np.asarray(detection.translation).reshape(3, 1)
    # The box sits on the ground: translation z is 0, so lift by half a height.
    centre = centre + np.array([[0.0], [0.0], [height / 2]])
    return rotation @ corners + centre


def project_to_image(points_global, global_to_ego, camera_to_ego, intrinsic):
    """(N,3) global -> (2,N) pixels plus a mask of points in front of the camera.

    ``global_to_ego`` must be the pose at the CAMERA's timestamp, not the
    keyframe's.
    """
    ego = (global_to_ego @ np.vstack(
        [points_global, np.ones((1, points_global.shape[1]))]))[:3]
    camera = (invert(camera_to_ego) @ np.vstack(
        [ego, np.ones((1, ego.shape[1]))]))[:3]

    in_front = camera[2, :] > 0.1          # behind the lens projects nonsense
    pixels = intrinsic @ camera
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = pixels[:2] / pixels[2]
    return pixels, in_front


def _draw_box(ax, pixels, colour, linewidth):
    """Twelve edges of a projected 3D box: front face, back face, connectors."""
    for start, end in ([(i, (i + 1) % 4) for i in range(4)]
                       + [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
                       + [(i, i + 4) for i in range(4)]):
        ax.plot([pixels[0, start], pixels[0, end]],
                [pixels[1, start], pixels[1, end]],
                color=colour, linewidth=linewidth, alpha=0.9)


def _visible(pixels, in_front, width, height):
    return (in_front.all()
            and pixels[0].max() > 0 and pixels[0].min() < width
            and pixels[1].max() > 0 and pixels[1].min() < height)


def overlay(nusc, cycle, detections, camera="CAM_FRONT", out_path=None,
            max_range=50.0, show_gt=True):
    """Draw detections, ground truth and radar returns onto the camera image."""
    sample = nusc.get("sample", cycle.sample_token)
    if camera not in sample["data"]:
        return None

    sd = nusc.get("sample_data", sample["data"][camera])
    calibration = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    camera_to_ego = pose_to_matrix(calibration["translation"], calibration["rotation"])
    intrinsic = np.array(calibration["camera_intrinsic"])

    # The camera's OWN ego pose, at its own capture time.
    camera_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    global_to_ego = invert(pose_to_matrix(camera_pose["translation"],
                                          camera_pose["rotation"]))

    image = Image.open(osp.join(nusc.dataroot, sd["filename"]))
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.imshow(image)
    ax.axis("off")
    width, height = image.size

    # Radar returns as points, for context behind the boxes.
    if cycle.num_points:
        pixels, in_front = project_to_image(
            cycle.points_global[:, :3].T, global_to_ego, camera_to_ego, intrinsic)
        visible = (in_front & (pixels[0] > 0) & (pixels[0] < width)
                   & (pixels[1] > 0) & (pixels[1] < height))
        moving = cycle.speed >= MOVING_SPEED
        ax.scatter(pixels[0, visible & ~moving], pixels[1, visible & ~moving],
                   s=8, c="white", edgecolors="0.2", linewidths=0.4, alpha=0.7)
        ax.scatter(pixels[0, visible & moving], pixels[1, visible & moving],
                   s=16, c=MOVING_COLOR, marker="^", edgecolors="none")

    # Ground truth, for comparison. Only annotations the radar actually hit --
    # the rest could not be recovered by a radar-only pipeline anyway.
    n_gt = 0
    if show_gt:
        for token in sample["anns"]:
            ann = nusc.get("sample_annotation", token)
            if ann["num_radar_pts"] == 0:
                continue
            box = nusc.get_box(token)
            corners = (Quaternion(box.orientation).rotation_matrix
                       @ _unit_corners(box.wlh)
                       + np.asarray(box.center).reshape(3, 1))
            pixels, in_front = project_to_image(corners, global_to_ego,
                                                camera_to_ego, intrinsic)
            if not _visible(pixels, in_front, width, height):
                continue
            _draw_box(ax, pixels, GT_COLOR, 1.4)
            n_gt += 1

    drawn = 0
    for detection in detections:
        if np.hypot(*detection.position_ego) > max_range:
            continue
        corners = _box_corners_global(detection)
        pixels, in_front = project_to_image(corners, global_to_ego, camera_to_ego,
                                            intrinsic)
        if not in_front.all():
            continue
        if pixels[0].max() < 0 or pixels[0].min() > width:
            continue
        if pixels[1].max() < 0 or pixels[1].min() > height:
            continue

        colour = MOVING_COLOR if detection.is_moving else STATIC_COLOR
        _draw_box(ax, pixels, colour, 2.0)
        label = f"{detection.speed:.1f} m/s" if detection.is_moving else "static"
        ax.text(float(pixels[0].mean()), float(pixels[1].min()) - 6, label,
                color=colour, fontsize=7.5, ha="center", fontweight="bold")
        drawn += 1

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.legend(handles=[
        plt.Line2D([], [], color=GT_COLOR, linewidth=1.6,
                   label="nuScenes ground truth"),
        plt.Line2D([], [], color=STATIC_COLOR, linewidth=2.2,
                   label="radar detection (static)"),
        plt.Line2D([], [], color=MOVING_COLOR, linewidth=2.2,
                   label="radar detection (moving)"),
    ], loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(f"{cycle.scene_name}  frame {cycle.frame_index:03d}  {camera}  |  "
                 f"{drawn} radar detections vs {n_gt} radar-visible annotations  |  "
                 f"visualisation only, not fusion", fontsize=9)

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
