"""Stage 1 end to end: 5 radars in, nuScenes-format detections and eval out.

    python scripts/run_stage1.py --verify            # coordinate + sync checks
    python scripts/run_stage1.py                     # mini_val + official eval
    python scripts/run_stage1.py --scenes scene-0061 --no-eval

Writes to outputs/stage1/:
    results_nuscenes.json     official submission format
    eval/metrics_summary.json DetectionEval output
    <scene>/frame_XXX.png     BEV figures
    <scene>/camera_XXX.png    camera overlay (visualisation only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuscenes.nuscenes import NuScenes

from eval.export_nuscenes_json import (
    sample_tokens_for_split,
    validate,
    write_results,
)
from eval.run_official_eval import format_official, run_official
from radar import road_mask
from radar.cluster import ClusterConfig, cluster_cells
from radar.ingest import RADAR_CHANNELS, find_scene, load_cycle
from radar.objects import CLASSIFICATION_NOTE, build_detections
from radar.occupancy_grid import GridConfig, RadarOccupancyGrid
from radar.sync import EgoPoseInterpolator, measure_radar_async
from radar.transform import transform_points, verify_convention
from viz.bev_plot import plot_cycle
from viz.camera_overlay import overlay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def annotation_boxes(nusc, cycle, max_range=50.0):
    """Radar-visible annotations for this keyframe, in the ego frame (plots only)."""
    boxes = []
    for token in nusc.get("sample", cycle.sample_token)["anns"]:
        ann = nusc.get("sample_annotation", token)
        if ann["num_radar_pts"] == 0:
            continue
        box = nusc.get_box(token)
        centre = transform_points(np.asarray(box.center).reshape(1, 3),
                                  cycle.global_to_ego)[0, :2]
        if np.linalg.norm(centre) > max_range:
            continue
        boxes.append({
            "name": ann["category_name"],
            "corners": transform_points(box.bottom_corners().T,
                                        cycle.global_to_ego)[:, :2],
            "center": centre,
        })
    return boxes


def run_verification(nusc):
    """The two checks the design rests on, measured rather than assumed."""
    print("\n=== coordinate convention (ISO 8855: x fwd, y left, z up) ===")
    result = verify_convention(nusc)
    print(f"  ego +x vs direction of travel : cos mean "
          f"{result['mean_cos_x_vs_travel']:+.4f}, min "
          f"{result['min_cos_x_vs_travel']:+.4f}  ({result['frames_tested']} frames)")
    print(f"  RADAR_FRONT mounted at x      : {result['radar_front_x']:+.2f} m")
    print(f"  RADAR_FRONT_LEFT mounted at y : {result['radar_front_left_y']:+.2f} m")
    print(f"  rotation correction needed    : {result['rotation_correction_needed']}")
    print("  -> nuScenes ego frame already matches ISO 8855; no rotation applied.")

    print("\n=== multi-radar asynchrony ===")
    spreads = []
    for scene in nusc.scene[:5]:
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            spreads.append(measure_radar_async(nusc, sample, RADAR_CHANNELS)["spread_ms"])
            token = sample["next"]
    spreads = np.array(spreads)
    print(f"  sweep spread across the 5 radars over {len(spreads)} keyframes:")
    print(f"    median {np.median(spreads):.1f} ms   p90 {np.percentile(spreads, 90):.1f} ms"
          f"   max {spreads.max():.1f} ms")
    print(f"  at 15 m/s that is up to {spreads.max() / 1000 * 15:.2f} m of ego travel")

    interpolator = EgoPoseInterpolator(nusc, nusc.scene[0]["token"])
    for sample in [nusc.get("sample", nusc.scene[0]["first_sample_token"])]:
        for channel in RADAR_CHANNELS:
            sd = nusc.get("sample_data", sample["data"][channel])
            interpolator.pose_at(sd["timestamp"])
    stats = interpolator.stats()
    print(f"\n  ego_pose records in scene {stats['pose_records']}, "
          f"median gap {stats['median_gap_ms']:.1f} ms")
    print(f"  pose lookups: {stats['exact']} exact, {stats['interpolated']} "
          f"interpolated, {stats['clamped']} clamped")
    print("  -> nuScenes stores a pose at each sweep's own timestamp, so SLERP is")
    print("     usually exact here; it is required for an arbitrary reference time")
    print("     and for genuinely asynchronous sensors.")
    return result


def run_scene(nusc, scene_name, args, grid_config, cluster_config):
    scene = find_scene(nusc, scene_name)
    interpolator = EgoPoseInterpolator(nusc, scene["token"])
    mask = road_mask.for_scene(nusc, scene["token"], enabled=not args.no_road_mask)
    if mask is not None and not mask.available:
        print(f"  road mask OFF: {mask.reason}")

    grid = RadarOccupancyGrid(grid_config)
    scene_dir = os.path.join(args.out, scene_name)
    detections_by_sample, per_frame = {}, []
    masked_total = 0
    started = time.time()

    token, index = scene["first_sample_token"], 0
    while token and (args.max_frames is None or index < args.max_frames):
        sample = nusc.get("sample", token)
        cycle = load_cycle(nusc, sample, scene["name"], index, interpolator)

        grid.update(cycle)
        cells = grid.snapshot(cycle, filtered=True)

        dropped = 0
        if mask is not None and mask.available:
            cells, dropped = mask.apply(cells)
            masked_total += dropped

        labels = cluster_cells(cells, cluster_config)
        detections = build_detections(cycle, cells, labels,
                                      resolution=grid_config.resolution,
                                      class_rule=args.class_rule)
        detections_by_sample[cycle.sample_token] = detections

        per_frame.append({
            "frame_index": index,
            "radar_points": cycle.num_points,
            "timestamp_spread_ms": round(cycle.timestamp_spread_ms, 1),
            "live_cells": grid.num_active_cells,
            "filtered_cells": len(cells),
            "road_masked_out": dropped,
            "detections": len(detections),
            "moving": sum(1 for d in detections if d.is_moving),
        })

        if args.viz_every and index % args.viz_every == 0:
            boxes = annotation_boxes(nusc, cycle, args.plot_range)
            plot_cycle(cycle, grid, cells, detections, boxes, args.plot_range,
                       os.path.join(scene_dir, f"frame_{index:03d}.png"), dropped)
            if not args.no_camera:
                overlay(nusc, cycle, detections,
                        out_path=os.path.join(scene_dir, f"camera_{index:03d}.png"))

        print(f"  frame {index:3d}  pts {cycle.num_points:4d}  "
              f"spread {cycle.timestamp_spread_ms:5.1f}ms  "
              f"cells {grid.num_active_cells:5d}  filtered {len(cells):4d}"
              f"{f' (-{dropped} off-road)' if dropped else ''}  "
              f"det {len(detections):3d} "
              f"({sum(1 for d in detections if d.is_moving)} moving)")

        token, index = sample["next"], index + 1

    elapsed = time.time() - started
    print(f"  {index} frames in {elapsed:.1f}s ({elapsed / max(index, 1):.2f} s/frame)"
          f"{f', road mask dropped {masked_total} cells total' if masked_total else ''}")
    return detections_by_sample, per_frame


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", default=os.environ.get(
        "NUSCENES_DATAROOT", os.path.join(ROOT, "v1.0-mini")))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--eval-set", default="mini_val")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="override the scenes; default is the eval split")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "stage1"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--verify", action="store_true",
                        help="run the coordinate and sync checks, then exit")

    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--active-radius", type=float, default=50.0)
    parser.add_argument("--min-log-odds", type=float, default=0.4)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--no-road-mask", action="store_true")
    parser.add_argument("--no-advection", action="store_true")
    parser.add_argument("--class-rule", choices=["all_car", "velocity"],
                        default="all_car",
                        help="all_car: every detection labelled 'car' "
                             "(maximum-likelihood class under no semantics). "
                             "velocity: moving->car, static->barrier")

    parser.add_argument("--viz-every", type=int, default=5)
    parser.add_argument("--plot-range", type=float, default=50.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.dataroot):
        raise SystemExit(f"nuScenes root not found: {args.dataroot}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if args.verify:
        run_verification(nusc)
        return

    grid_config = GridConfig(
        resolution=args.resolution,
        active_radius=args.active_radius,
        min_log_odds=args.min_log_odds,
        min_hits=args.min_hits,
        advect_dynamic_cells=not args.no_advection,
    )
    cluster_config = ClusterConfig(eps=args.eps, min_samples=args.min_samples)

    if args.scenes:
        scene_names = args.scenes
    else:
        from nuscenes.utils import splits
        wanted = set(getattr(splits, args.eval_set))
        scene_names = [s["name"] for s in nusc.scene if s["name"] in wanted]

    print(f"scenes: {', '.join(scene_names)}")
    print(f"classification: {CLASSIFICATION_NOTE}")

    all_detections, all_frames = {}, {}
    for scene_name in scene_names:
        print(f"\n=== {scene_name} ===")
        detections, frames = run_scene(nusc, scene_name, args,
                                       grid_config, cluster_config)
        all_detections.update(detections)
        all_frames[scene_name] = frames

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_frame.json"), "w") as handle:
        json.dump(all_frames, handle, indent=1)

    if args.no_eval:
        print("\nskipping evaluation (--no-eval)")
        return

    result_path = os.path.join(args.out, "results_nuscenes.json")
    tokens = sample_tokens_for_split(nusc, args.eval_set)
    info = write_results(result_path, all_detections, tokens,
                         use_map=not args.no_road_mask)
    print(f"\nwrote {info['boxes']} boxes over {info['samples']} samples "
          f"({info['samples_with_no_detections']} empty) -> {result_path}")
    validate(result_path, nusc, args.eval_set)
    print("submission schema validated against the installed devkit loader")

    print(f"\n=== official nuScenes DetectionEval ({args.eval_set}) ===")
    print(f"NOTE: {CLASSIFICATION_NOTE}")
    metrics = run_official(nusc, result_path, args.eval_set,
                           output_dir=os.path.join(args.out, "eval"))
    emitted = ("car",) if args.class_rule == "all_car" else ("car", "barrier")
    print(format_official(metrics, predicted_classes=emitted))

    print("\nThat mAP is dominated by the classification limitation above.")
    print("To judge detection on its own, with class collapsed to one label:")
    print(f"  python eval/run_detection_only_eval.py {result_path}")
    print(f"\nresults in {args.out}")


if __name__ == "__main__":
    main()
