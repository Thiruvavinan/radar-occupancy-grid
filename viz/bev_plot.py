"""BEV figure: annotations, raw returns, grid cells and detections together."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from radar.occupancy_grid import MOVING_SPEED

MOVING_COLOR = "#d1495b"
STATIC_COLOR = "#2a9d8f"
GT_COLOR = "#e9c46a"
EGO_COLOR = "#264653"


def _short(name):
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def plot_cycle(cycle, grid, cells, detections, gt_boxes=None, plot_range=50.0,
               out_path=None, masked_out=0):
    """One keyframe, everything overlaid, in the ego frame (x forward, y left)."""
    fig, ax = plt.subplots(figsize=(10.5, 10))
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_aspect("equal")
    ax.set_xlabel("ego x (m), forward")
    ax.set_ylabel("ego y (m), left")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    if not cells.is_empty:
        ax.scatter(cells.centers_ego[:, 0], cells.centers_ego[:, 1],
                   c="0.78", s=8, marker="s", edgecolors="none", zorder=1)

    for box in gt_boxes or []:
        ax.add_patch(Polygon(box["corners"], closed=True, fill=False,
                             edgecolor=GT_COLOR, linewidth=1.5,
                             linestyle="--", zorder=2))
        cx, cy = box["center"]
        if abs(cx) < plot_range - 5 and abs(cy) < plot_range - 3:
            ax.text(cx, cy + 1.4, _short(box["name"]), fontsize=6,
                    color="#b08900", ha="center", zorder=3)

    if cycle.num_points:
        moving = cycle.speed >= MOVING_SPEED
        ax.scatter(cycle.points_ego[~moving, 0], cycle.points_ego[~moving, 1],
                   facecolors="none", edgecolors="0.35", s=16, linewidths=0.7,
                   zorder=4)
        ax.scatter(cycle.points_ego[moving, 0], cycle.points_ego[moving, 1],
                   c=MOVING_COLOR, s=18, marker="^", edgecolors="none", zorder=5)

    for detection in detections:
        color = MOVING_COLOR if detection.is_moving else STATIC_COLOR
        x, y = detection.position_ego
        width, length = detection.size[0], detection.size[1]
        ax.add_patch(Rectangle((x - length / 2, y - width / 2), length, width,
                               fill=False, edgecolor=color, linewidth=1.8, zorder=6))
        ax.plot(x, y, marker="o", color=color, markersize=4,
                markeredgecolor="white", markeredgewidth=0.6, zorder=8)
        if detection.is_moving:
            vx, vy = detection.velocity_ego
            ax.arrow(x, y, vx, vy, head_width=1.1, head_length=1.4,
                     fc=color, ec=color, length_includes_head=True, zorder=7)
            ax.text(x + 1.2, y + 1.2, f"{detection.speed:.1f}", fontsize=6.5,
                    color=color, fontweight="bold", zorder=9)

    ax.add_patch(Rectangle((-1.4, -0.95), 4.7, 1.9, facecolor=EGO_COLOR,
                           edgecolor="none", zorder=10))

    ax.legend(handles=[
        plt.Line2D([], [], color="0.35", marker="o", linestyle="none",
                   markerfacecolor="none", markersize=5, label="radar return (static)"),
        plt.Line2D([], [], color=MOVING_COLOR, marker="^", linestyle="none",
                   markersize=5, label="radar return (moving)"),
        plt.Line2D([], [], color="0.78", marker="s", linestyle="none",
                   markersize=6, label="grid cell (filtered)"),
        plt.Line2D([], [], color=GT_COLOR, linewidth=1.5, linestyle="--",
                   label="nuScenes annotation"),
        plt.Line2D([], [], color=STATIC_COLOR, linewidth=1.8, label="detection: static"),
        plt.Line2D([], [], color=MOVING_COLOR, linewidth=1.8,
                   label="detection: moving (arrow = 1 s)"),
    ], loc="upper right", fontsize=7, framealpha=0.95)

    moving_count = sum(1 for d in detections if d.is_moving)
    mask_note = f"  |  road mask dropped {masked_out}" if masked_out else ""
    ax.set_title(
        f"{cycle.scene_name}  frame {cycle.frame_index:03d}  |  "
        f"{cycle.num_points} returns (5 radars, {cycle.timestamp_spread_ms:.0f} ms spread)"
        f"  ->  {len(cells)} cells  ->  {len(detections)} detections "
        f"({moving_count} moving){mask_note}", fontsize=9.5, pad=10)

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return None
    return fig
