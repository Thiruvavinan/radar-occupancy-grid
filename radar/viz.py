"""One bird's-eye-view figure per keyframe, everything on a single set of axes.

Layers, bottom to top: confident grid cells, nuScenes annotation boxes, this
frame's raw radar returns, and the extracted objects with velocity arrows.
Seeing them together is what makes the output judgeable at a glance -- both
where it works and, just as usefully, which annotations have nothing on them.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")           # headless: we only ever save files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from radar.grid import MOVING_SPEED

MOVING_COLOR = "#d1495b"
STATIC_COLOR = "#2a9d8f"
GT_COLOR = "#e9c46a"
EGO_COLOR = "#264653"


def _short(name):
    """'vehicle.car' -> 'car', 'human.pedestrian.adult' -> 'pedestrian'."""
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def plot_frame(frame, grid, cells, objects, gt_boxes=None, plot_range=60.0,
               out_path=None):
    """Render one keyframe. ``gt_boxes`` come from run.py, or None."""
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_aspect("equal")
    ax.set_xlabel("ego x (m), forward")
    ax.set_ylabel("ego y (m), left")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # 1. the cells that cleared the confidence filter
    if not cells.is_empty:
        ax.scatter(cells.centers_ego[:, 0], cells.centers_ego[:, 1],
                   c="0.80", s=14, marker="s", edgecolors="none", zorder=1)

    # 2. nuScenes annotations
    for box in gt_boxes or []:
        ax.add_patch(Polygon(box["corners"], closed=True, fill=False,
                             edgecolor=GT_COLOR, linewidth=1.6,
                             linestyle="--", zorder=2))
        cx, cy = box["center"]
        if abs(cx) < plot_range - 6 and abs(cy) < plot_range - 4:
            ax.text(cx, cy + 1.6, _short(box["name"]), fontsize=6,
                    color="#b08900", ha="center", zorder=3)

    # 3. this frame's raw returns, split by whether they are moving
    if frame.num_points:
        moving = frame.speed >= MOVING_SPEED
        ax.scatter(frame.points_ego[~moving, 0], frame.points_ego[~moving, 1],
                   facecolors="none", edgecolors="0.35", s=17, linewidths=0.7,
                   zorder=4)
        ax.scatter(frame.points_ego[moving, 0], frame.points_ego[moving, 1],
                   c=MOVING_COLOR, s=19, marker="^", edgecolors="none", zorder=5)

    # 4. the objects
    for obj in objects:
        color = MOVING_COLOR if obj.is_moving else STATIC_COLOR
        x0, y0, x1, y1 = obj.extent_ego
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=color, linewidth=2.0, zorder=6))
        ax.plot(*obj.position_ego, marker="o", color=color, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=8)
        if obj.is_moving:
            # Arrow length = one second of travel, so it reads as a speed.
            ax.arrow(obj.position_ego[0], obj.position_ego[1],
                     obj.velocity_ego[0], obj.velocity_ego[1],
                     head_width=1.3, head_length=1.7, fc=color, ec=color,
                     length_includes_head=True, zorder=7)
            ax.text(obj.position_ego[0] + 1.4, obj.position_ego[1] + 1.4,
                    f"{obj.speed:.1f} m/s", fontsize=7, color=color,
                    fontweight="bold", zorder=9)

    # the ego vehicle: a 4.7 x 1.9 m box at the origin, nose toward +x
    ax.add_patch(Rectangle((-1.4, -0.95), 4.7, 1.9, facecolor=EGO_COLOR,
                           edgecolor="none", zorder=10))

    ax.legend(handles=[
        plt.Line2D([], [], color="0.35", marker="o", linestyle="none",
                   markerfacecolor="none", markersize=5, label="radar return (static)"),
        plt.Line2D([], [], color=MOVING_COLOR, marker="^", linestyle="none",
                   markersize=5, label="radar return (moving)"),
        plt.Line2D([], [], color="0.80", marker="s", linestyle="none",
                   markersize=6, label="confident grid cell"),
        plt.Line2D([], [], color=GT_COLOR, linewidth=1.6, linestyle="--",
                   label="nuScenes annotation"),
        plt.Line2D([], [], color=STATIC_COLOR, linewidth=2, label="object: static"),
        plt.Line2D([], [], color=MOVING_COLOR, linewidth=2,
                   label="object: moving (arrow = 1 s)"),
    ], loc="upper right", fontsize=7.5, framealpha=0.95)

    n_moving = sum(1 for o in objects if o.is_moving)
    ax.set_title(
        f"{frame.scene_name}  frame {frame.frame_index:03d}   |   "
        f"{frame.num_points} returns -> {len(cells)} confident cells -> "
        f"{len(objects)} objects ({n_moving} moving)   |   "
        f"{len(gt_boxes or [])} radar-visible annotations", fontsize=10.5, pad=12)

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return None
    return fig


def plot_static_vs_moving(records, out_path, title=None):
    """Speed vs range and a speed histogram, pooled over a whole scene.

    This is the figure that shows the static/dynamic separation the pipeline
    exists to produce: static objects in a tight band near zero, movers well
    clear of the threshold, and a gap in between.
    """
    speeds = np.array([r["speed"] for r in records])
    ranges = np.array([r["range_from_ego"] for r in records])
    moving = speeds > MOVING_SPEED

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.scatter(ranges[~moving], speeds[~moving], s=10, c=STATIC_COLOR,
               label=f"static (n={int((~moving).sum())})", edgecolors="none")
    ax.scatter(ranges[moving], speeds[moving], s=10, c=MOVING_COLOR,
               label=f"moving (n={int(moving.sum())})", edgecolors="none")
    ax.axhline(MOVING_SPEED, color="0.4", linestyle="--", linewidth=1,
               label=f"threshold {MOVING_SPEED} m/s")
    ax.set_xlabel("range from ego (m)")
    ax.set_ylabel("object speed (m/s)")
    ax.set_title("Object speed vs range", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.hist(speeds, bins=60, color="0.6")
    ax.axvline(MOVING_SPEED, color=MOVING_COLOR, linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("object speed (m/s)")
    ax.set_ylabel("object count (log)")
    ax.set_title("Speed distribution", fontsize=10)
    ax.grid(True, alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
