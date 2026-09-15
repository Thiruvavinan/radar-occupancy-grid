"""Evaluate detection alone, with classification taken out of the question.

The official nuScenes mAP is a *detection and classification* metric. This
pipeline cannot classify at all, so nine of ten classes score zero by
construction and the headline number says almost nothing about how well the
radar actually finds things.

This script removes that confound in the least hand-wavy way available: it
relabels **every ground-truth box and every detection to a single class** and
then runs the devkit's own `accumulate` / `calc_ap` / `calc_tp` on the result.
So the matching rule, the score-sorted PR curve, the recall and precision
floors, and the true-positive error definitions are all exactly nuScenes' --
only the class dimension is collapsed.

    python eval/run_detection_only_eval.py outputs/stage1/results_nuscenes.json

What it is: a clean measure of "did it find an object, and where". What it is
NOT: an official number. Do not compare it to a published mAP, which is a
harder 10-class problem. Reported alongside the official table, never instead
of it.

Two ground-truth sets are reported, because they answer different questions:

  all           every eval-eligible annotation. The honest upper bound, and it
                includes pedestrians and cyclists that radar barely sees.
  radar-visible only annotations with num_radar_pts > 0. What a radar-only
                pipeline could in principle recover; anything else is measuring
                the sensor's physics, not this code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.nuscenes import NuScenes

#: Every box is relabelled to this. Any of the 10 official names works; `car`
#: is used so the devkit's per-class range (50 m) and TP settings apply.
SINGLE_CLASS = "car"


def _radar_visible_translations(nusc, sample_token):
    """Rounded centres of this sample's annotations that radar actually hit.

    `DetectionBox` only carries `num_pts`, which is lidar+radar combined, so
    "radar-visible" cannot be read off the box itself -- it has to come from the
    annotation table. Centres come from the same source on both sides, so
    rounding to millimetres is an exact key, not a fuzzy match.
    """
    visible = set()
    for token in nusc.get("sample", sample_token)["anns"]:
        ann = nusc.get("sample_annotation", token)
        if ann["num_radar_pts"] > 0:
            visible.add(tuple(round(v, 3) for v in ann["translation"]))
    return visible


def _relabel(eval_boxes, radar_visible_only=False, nusc=None):
    """Collapse every box onto one class, optionally keeping radar-visible GT."""
    kept = 0
    for token in eval_boxes.sample_tokens:
        visible = (_radar_visible_translations(nusc, token)
                   if radar_visible_only else None)
        boxes = []
        for box in eval_boxes[token]:
            if visible is not None:
                if tuple(round(v, 3) for v in box.translation) not in visible:
                    continue
            box.detection_name = SINGLE_CLASS
            box.detection_score = getattr(box, "detection_score", -1.0)
            boxes.append(box)
            kept += 1
        eval_boxes.boxes[token] = boxes
    return kept


def _gt_is_moving(nusc, sample_token, threshold=0.5):
    """Rounded centres of this sample's annotations that are actually moving.

    Motion is read from `nusc.box_velocity`, i.e. how the annotation itself
    moves between keyframes. It is NOT inferred from the category: a parked car
    is stationary and a walking pedestrian is not, so any mapping that calls
    "car" dynamic and "pedestrian" static gets both cases backwards.
    """
    moving = set()
    for token in nusc.get("sample", sample_token)["anns"]:
        velocity = nusc.box_velocity(token)
        if not np.all(np.isfinite(velocity[:2])):
            continue
        if float(np.linalg.norm(velocity[:2])) > threshold:
            ann = nusc.get("sample_annotation", token)
            moving.add(tuple(round(v, 3) for v in ann["translation"]))
    return moving


def _split_by_motion(nusc, ground_truth, predictions, config, threshold=0.5):
    """Precision/recall separately for genuinely moving and stationary objects.

    Ground truth is split by its own measured velocity; a detection is counted
    against whichever group its matched ground truth belongs to. Detections that
    match nothing are false positives for the group their own velocity claims,
    which is what makes this a fair test of the static/dynamic call.
    """
    distance = config.dist_th_tp          # 2.0 m, nuScenes' TP distance
    groups = {"moving": dict(tp=0, fp=0, fn=0, right=0),
              "static": dict(tp=0, fp=0, fn=0, right=0)}

    for token in ground_truth.sample_tokens:
        truth = list(ground_truth[token])
        detections = sorted(predictions[token],
                            key=lambda b: b.detection_score, reverse=True)
        moving_gt = _gt_is_moving(nusc, token, threshold)
        labels = [tuple(round(v, 3) for v in b.translation) in moving_gt
                  for b in truth]

        if not truth:
            for detection in detections:
                speed = float(np.hypot(*detection.velocity[:2]))
                groups["moving" if speed > threshold else "static"]["fp"] += 1
            continue

        centres = np.array([b.translation[:2] for b in truth])
        taken = np.zeros(len(truth), dtype=bool)
        for detection in detections:
            speed = float(np.hypot(*detection.velocity[:2]))
            said_moving = speed > threshold
            distances = np.linalg.norm(centres - np.array(detection.translation[:2]),
                                       axis=1)
            distances[taken] = np.inf
            best = int(np.argmin(distances))
            if distances[best] <= distance:
                taken[best] = True
                group = "moving" if labels[best] else "static"
                groups[group]["tp"] += 1
                groups[group]["right"] += int(said_moving == labels[best])
            else:
                groups["moving" if said_moving else "static"]["fp"] += 1

        for index, claimed in enumerate(taken):
            if not claimed:
                groups["moving" if labels[index] else "static"]["fn"] += 1

    out = {}
    for name, counts in groups.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        out[name] = {
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "motion_call_correct": counts["right"] / tp if tp else float("nan"),
            "tp": tp, "fp": fp, "fn": fn,
        }
    out["match_distance_m"] = distance
    out["speed_threshold_ms"] = threshold
    return out


def _precision_recall(ground_truth, predictions, config):
    """Plain TP / FP / FN counts at each matching distance.

    AP summarises the whole precision-recall curve into one number, which is the
    right thing for a benchmark and the wrong thing for answering "how often is
    it right?". These are the raw counts, using the same matching rule nuScenes
    uses: detections sorted by score, each taking the nearest unclaimed ground
    truth within the threshold.
    """
    results = {}
    for threshold in config.dist_ths:
        tp = fp = fn = 0
        errors = []
        for token in ground_truth.sample_tokens:
            truth = list(ground_truth[token])
            detections = sorted(predictions[token],
                                key=lambda b: b.detection_score, reverse=True)
            if not truth:
                fp += len(detections)
                continue

            centres = np.array([b.translation[:2] for b in truth])
            taken = np.zeros(len(truth), dtype=bool)
            for detection in detections:
                position = np.array(detection.translation[:2])
                distances = np.linalg.norm(centres - position, axis=1)
                distances[taken] = np.inf
                best = int(np.argmin(distances))
                if distances[best] <= threshold:
                    taken[best] = True
                    tp += 1
                    errors.append(float(distances[best]))
                else:
                    fp += 1
            fn += int((~taken).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        results[threshold] = {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0),
            "tp": tp, "fp": fp, "fn": fn,
            "mean_position_error_m": float(np.mean(errors)) if errors else float("nan"),
        }
    return results


def evaluate(nusc, result_path, eval_set="mini_val", radar_visible_only=False,
             config_name="detection_cvpr_2019", verbose=False):
    """Single-class AP and TP errors, using the devkit's own machinery."""
    config = config_factory(config_name)

    predictions, _ = load_prediction(result_path, config.max_boxes_per_sample,
                                     DetectionBox, verbose=verbose)
    ground_truth = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)

    predictions = add_center_dist(nusc, predictions)
    ground_truth = add_center_dist(nusc, ground_truth)

    # Relabel BEFORE filtering: filter_eval_boxes applies a per-class range, and
    # collapsing first means every box is judged by the same 50 m rule.
    n_pred = _relabel(predictions)
    n_gt = _relabel(ground_truth, radar_visible_only=radar_visible_only, nusc=nusc)

    predictions = filter_eval_boxes(nusc, predictions, config.class_range,
                                    verbose=verbose)
    ground_truth = filter_eval_boxes(nusc, ground_truth, config.class_range,
                                     verbose=verbose)

    # Straight counts, matched exactly the way nuScenes matches: detections
    # sorted by score, greedy nearest ground truth, one GT per detection.
    counts = _precision_recall(ground_truth, predictions, config)
    by_motion = _split_by_motion(nusc, ground_truth, predictions, config)

    rows, aps = {}, []
    for threshold in config.dist_ths:
        metric_data = accumulate(ground_truth, predictions, SINGLE_CLASS,
                                 config.dist_fcn_callable, threshold)
        ap = calc_ap(metric_data, config.min_recall, config.min_precision)
        aps.append(ap)

        # Highest recall at which any true positive survives -- the number the
        # min_recall floor is applied to, and the one worth knowing.
        nonzero = np.flatnonzero(metric_data.precision > 0)
        max_recall = float(metric_data.recall[nonzero[-1]]) if nonzero.size else 0.0

        rows[threshold] = {
            "ap": float(ap),
            "max_recall": max_recall,
            "precision_at_max_recall": float(metric_data.precision[nonzero[-1]])
            if nonzero.size else 0.0,
        }

    # TP errors are defined at a single matching distance in nuScenes.
    metric_data = accumulate(ground_truth, predictions, SINGLE_CLASS,
                             config.dist_fcn_callable, config.dist_th_tp)
    tp_errors = {name: float(calc_tp(metric_data, config.min_recall, name))
                 for name in ["trans_err", "scale_err", "orient_err", "vel_err"]}

    return {
        "eval_set": eval_set,
        "gt_set": "radar-visible only" if radar_visible_only else "all annotations",
        "detections": n_pred,
        "ground_truth": n_gt,
        "precision_recall": counts,
        "by_motion": by_motion,
        "ap_by_distance": rows,
        "mean_ap": float(np.mean(aps)),
        "tp_errors": tp_errors,
        "tp_distance_m": config.dist_th_tp,
        "min_recall": config.min_recall,
        "min_precision": config.min_precision,
    }


def format_report(result: dict) -> str:
    lines = [
        f"  ground truth : {result['gt_set']}  ({result['ground_truth']} boxes)",
        f"  detections   : {result['detections']}",
        "",
        f"  {'match dist':>11} {'precision':>10} {'recall':>8} {'F1':>7} "
        f"{'TP':>6} {'FP':>6} {'FN':>6} {'pos err':>8}",
        "  " + "-" * 68,
    ]
    for threshold, row in sorted(result["precision_recall"].items()):
        lines.append(
            f"  {threshold:>10.1f}m {row['precision']:>9.1%} {row['recall']:>7.1%} "
            f"{row['f1']:>6.1%} {row['tp']:>6d} {row['fp']:>6d} {row['fn']:>6d} "
            f"{row['mean_position_error_m']:>7.2f}m")

    motion = result.get("by_motion")
    if motion:
        lines += [
            "",
            f"  split by the ground truth's OWN measured velocity "
            f"(> {motion['speed_threshold_ms']} m/s), at "
            f"{motion['match_distance_m']} m match:",
            f"  {'':>11} {'precision':>10} {'recall':>8} {'motion call':>12} "
            f"{'TP':>6} {'FP':>6} {'FN':>6}",
            "  " + "-" * 64,
        ]
        for name in ("moving", "static"):
            row = motion[name]
            lines.append(
                f"  {name:>11} {row['precision']:>9.1%} {row['recall']:>7.1%} "
                f"{row['motion_call_correct']:>11.1%} {row['tp']:>6d} "
                f"{row['fp']:>6d} {row['fn']:>6d}")

    lines += [
        "",
        f"  {'match dist':>11} {'AP':>8} {'max recall':>11} {'precision there':>16}",
        "  " + "-" * 50,
    ]
    for threshold, row in sorted(result["ap_by_distance"].items()):
        lines.append(f"  {threshold:>10.1f}m {row['ap']:>8.4f} "
                     f"{row['max_recall']:>10.1%} "
                     f"{row['precision_at_max_recall']:>15.1%}")
    lines += [
        "  " + "-" * 50,
        f"  {'single-class AP (mean over distances)':<38} {result['mean_ap']:.4f}",
        "",
        f"  true-positive errors at {result['tp_distance_m']} m match:",
    ]
    units = {"trans_err": "m", "scale_err": "1-IoU", "orient_err": "rad",
             "vel_err": "m/s"}
    for name, value in result["tp_errors"].items():
        lines.append(f"    {name:<12} {value:>7.3f} {units[name]}")
    lines += [
        "",
        f"  (AP is 0 wherever max recall <= {result['min_recall']}: nuScenes clips",
        f"   recall below {result['min_recall']} and precision below "
        f"{result['min_precision']}.)",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_path", nargs="?",
                        default=os.path.join("outputs", "stage1",
                                             "results_nuscenes.json"))
    parser.add_argument("--dataroot", default=os.environ.get(
        "NUSCENES_DATAROOT", "v1.0-mini"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--eval-set", default="mini_val")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print("=" * 64)
    print("DETECTION-ONLY EVALUATION -- classification collapsed to one class")
    print("Uses nuScenes' own matching, PR curve and TP definitions.")
    print("NOT an official number; not comparable to a published mAP.")
    print("=" * 64)

    results = {}
    for radar_only in (False, True):
        key = "radar_visible" if radar_only else "all"
        results[key] = evaluate(nusc, args.result_path, args.eval_set, radar_only)
        label = ("vs RADAR-VISIBLE ground truth" if radar_only
                 else "vs ALL eval-eligible ground truth")
        print(f"\n--- {label} ---")
        print(format_report(results[key]))

    out = args.out or os.path.join(os.path.dirname(args.result_path),
                                   "detection_only_metrics.json")
    with open(out, "w") as handle:
        json.dump({k: {**v,
                       "ap_by_distance": {str(t): r for t, r in v["ap_by_distance"].items()},
                       "precision_recall": {str(t): r for t, r in v["precision_recall"].items()},
                       "by_motion": v["by_motion"]}
                   for k, v in results.items()}, handle, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
