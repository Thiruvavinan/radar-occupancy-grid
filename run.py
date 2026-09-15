"""Stage 1 end to end: a nuScenes scene in, radar objects and figures out.

    python run.py --list-scenes
    python run.py --scenes scene-0061 scene-0757 scene-1077
    python run.py --scenes scene-1077 --anchor ego

Per scene, writes to outputs/<scene>/:

    frame_XXX.png          BEV view: annotations, returns, cells, objects
    static_vs_moving.png   the static/dynamic separation over the scene
    objects.json           every object, every frame
    summary.json           aggregates and the annotation comparison

The comparison against nuScenes annotations at the bottom of this file is a
**sanity check, not a detection metric**: nearest-centroid proximity with a 4 m
gate, no IoU, no confidence sweep, no official eval protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from nuscenes.nuscenes import NuScenes

from radar.detect import DetectConfig, detect
from radar.grid import MOVING_SPEED, GridConfig, RadarOccupancyGrid
from radar.ingest import iter_scene_frames, transform_points
from radar.viz import plot_frame, plot_static_vs_moving

ROOT = os.path.dirname(os.path.abspath(__file__))
MATCH_DISTANCE = 4.0        # metres; how close an object must be to count


# --------------------------------------------------------------------------
# nuScenes annotations, for the qualitative check only
# --------------------------------------------------------------------------

def annotation_boxes(nusc, frame, max_range=60.0):
    """Radar-visible annotations for this keyframe, in the ego frame.

    Only annotations with ``num_radar_pts > 0`` are kept. An annotation the
    radar never hit cannot be recovered by a radar-only pipeline, so including
    it would measure radar physics rather than this code.
    """
    boxes = []
    for token in nusc.get("sample", frame.sample_token)["anns"]:
        ann = nusc.get("sample_annotation", token)
        if ann["num_radar_pts"] == 0:
            continue

        box = nusc.get_box(token)
        center = transform_points(np.asarray(box.center).reshape(1, 3),
                                  frame.global_to_ego)[0, :2]
        if np.linalg.norm(center) > max_range:
            continue

        velocity = nusc.box_velocity(token)
        speed = (float(np.linalg.norm(velocity[:2]))
                 if np.all(np.isfinite(velocity[:2])) else float("nan"))
        boxes.append({
            "name": ann["category_name"],
            "corners": transform_points(box.bottom_corners().T,
                                        frame.global_to_ego)[:, :2],
            "center": center,
            "speed": speed,
        })
    return boxes


def compare_to_annotations(objects, boxes):
    """For each annotation, is there an object on it, and is the motion right?

    This direction (annotation -> object) is the informative one. The reverse
    is dragged down by radar legitimately seeing plenty the dataset never
    annotates: walls, fences, barriers, vegetation.
    """
    records = []
    positions = np.array([o.position_ego for o in objects]) if objects else None

    for box in boxes:
        covered, motion_right = False, None
        if positions is not None:
            distances = np.linalg.norm(positions - box["center"], axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] < MATCH_DISTANCE:
                covered = True
                if np.isfinite(box["speed"]):
                    motion_right = (objects[nearest].is_moving
                                    == (box["speed"] > MOVING_SPEED))
        records.append({"name": box["name"], "covered": covered,
                        "motion_right": motion_right})
    return records


def summarise(records):
    """Aggregate the per-annotation comparison over a scene."""
    if not records:
        return {}
    judged = [r["motion_right"] for r in records if r["motion_right"] is not None]

    by_category = {}
    for record in records:
        bucket = by_category.setdefault(record["name"], {"n": 0, "covered": 0})
        bucket["n"] += 1
        bucket["covered"] += int(record["covered"])
    for bucket in by_category.values():
        bucket["rate"] = round(bucket["covered"] / bucket["n"], 3)

    return {
        "annotations": len(records),
        "covered": sum(r["covered"] for r in records),
        "coverage_rate": round(np.mean([r["covered"] for r in records]), 4),
        "motion_agreement": round(float(np.mean(judged)), 4) if judged else None,
        "motion_compared": len(judged),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1]["n"])),
    }


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

def run_scene(nusc, scene_name, args, grid_config, detect_config):
    scene_dir = os.path.join(args.out, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    grid = RadarOccupancyGrid(grid_config)
    all_objects, comparisons, per_frame = [], [], []
    started = time.time()

    for frame in iter_scene_frames(nusc, scene_name, max_frames=args.max_frames):
        grid.update(frame)                              # steps 1-4
        cells, objects = detect(frame, grid, detect_config)   # steps 5-7

        all_objects.extend(o.as_dict() for o in objects)
        boxes = annotation_boxes(nusc, frame, args.plot_range) if not args.no_gt else []
        comparisons.extend(compare_to_annotations(objects, boxes))

        per_frame.append({
            "frame_index": frame.frame_index,
            "radar_points": frame.num_points,
            "live_cells": grid.num_active_cells,
            "confident_cells": len(cells),
            "objects": len(objects),
            "moving": sum(1 for o in objects if o.is_moving),
        })

        if args.viz_every and frame.frame_index % args.viz_every == 0:
            plot_frame(frame, grid, cells, objects, boxes, args.plot_range,
                       os.path.join(scene_dir, f"frame_{frame.frame_index:03d}.png"))

        print(f"  frame {frame.frame_index:3d}  pts {frame.num_points:4d}  "
              f"cells {grid.num_active_cells:5d}  confident {len(cells):4d}  "
              f"objects {len(objects):3d} "
              f"({sum(1 for o in objects if o.is_moving)} moving)")

    speeds = np.array([o["speed"] for o in all_objects]) if all_objects else np.zeros(0)
    moving = speeds > MOVING_SPEED
    summary = {
        "scene": scene_name,
        "frames": len(per_frame),
        "seconds_per_frame": round((time.time() - started) / max(len(per_frame), 1), 3),
        "grid_config": vars(grid_config),
        "detect_config": vars(detect_config),
        "objects_per_frame": round(len(all_objects) / max(len(per_frame), 1), 2),
        "mean_speed_moving": round(float(speeds[moving].mean()), 3) if moving.any() else 0.0,
        "mean_speed_static": round(float(speeds[~moving].mean()), 3) if (~moving).any() else 0.0,
        "annotation_check": summarise(comparisons),
        "per_frame": per_frame,
    }

    with open(os.path.join(scene_dir, "objects.json"), "w") as handle:
        json.dump(all_objects, handle, indent=1)
    with open(os.path.join(scene_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    if all_objects:
        plot_static_vs_moving(
            all_objects, os.path.join(scene_dir, "static_vs_moving.png"),
            title=f"{scene_name}: static vs moving "
                  f"({len(all_objects)} objects over {len(per_frame)} frames)")

    check = summary["annotation_check"]
    print(f"\n  --- {scene_name} ---")
    print(f"  objects / frame            {summary['objects_per_frame']}")
    print(f"  mean speed, moving         {summary['mean_speed_moving']} m/s")
    print(f"  mean speed, static         {summary['mean_speed_static']} m/s")
    if check:
        print(f"  annotations covered        {check['covered']}/{check['annotations']}"
              f"  ({check['coverage_rate']:.1%})")
        print(f"  static/dynamic agreement   {check['motion_agreement']:.1%}"
              f" over {check['motion_compared']} comparisons")
    print()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", default=os.environ.get(
        "NUSCENES_DATAROOT", os.path.join(ROOT, "v1.0-mini")))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scenes", nargs="+",
                        default=["scene-0061", "scene-0757", "scene-1077"])
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--list-scenes", action="store_true")

    parser.add_argument("--anchor", choices=["world", "ego"], default="world",
                        help="key cells to the map (exact) or to the vehicle")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--min-existence-prob", type=float, default=0.60)
    parser.add_argument("--no-advection", action="store_true",
                        help="disable object-motion compensation (ablation)")
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=1)

    parser.add_argument("--viz-every", type=int, default=4,
                        help="render every Nth keyframe; 0 disables figures")
    parser.add_argument("--plot-range", type=float, default=60.0)
    parser.add_argument("--no-gt", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.dataroot):
        raise SystemExit(f"nuScenes root not found: {args.dataroot}\n"
                         f"Pass --dataroot or set NUSCENES_DATAROOT.")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if args.list_scenes:
        for scene in nusc.scene:
            print(f"{scene['name']}  {scene['nbr_samples']:3d} samples  "
                  f"{scene['description']}")
        return

    grid_config = GridConfig(
        anchor=args.anchor,
        resolution=args.resolution,
        min_hits=args.min_hits,
        min_existence_prob=args.min_existence_prob,
        advect_dynamic_cells=not args.no_advection,
    )
    detect_config = DetectConfig(eps=args.eps, min_samples=args.min_samples)

    for scene in args.scenes:
        print(f"\n=== {scene} ===")
        run_scene(nusc, scene, args, grid_config, detect_config)
    print(f"wrote results to {args.out}")


if __name__ == "__main__":
    main()
