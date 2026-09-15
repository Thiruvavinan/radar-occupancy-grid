"""Run the official nuScenes DetectionEval over all 10 detection classes.

This is the real benchmark. The pipeline emits only one class (see
radar/objects.py CLASSIFICATION_NOTE), so nine score exactly zero by
construction and drag the mean down -- a headline mAP read without that context
is misleading.

To judge detection on its own, with classification taken out of the question,
use `eval/run_detection_only_eval.py`.
"""

from __future__ import annotations

import json
import os

import numpy as np
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval


def run_official(nusc, result_path, eval_set="mini_val", output_dir=None,
                 config_name="detection_cvpr_2019", verbose=False):
    """Run DetectionEval and return its metrics summary as a dict."""
    output_dir = output_dir or os.path.join(os.path.dirname(result_path), "eval")
    os.makedirs(output_dir, exist_ok=True)

    evaluator = DetectionEval(
        nusc,
        config=config_factory(config_name),
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
    )
    metrics_summary = evaluator.main(render_curves=False, plot_examples=0)

    with open(os.path.join(output_dir, "metrics_summary.json")) as handle:
        return json.load(handle)


def format_official(metrics: dict, predicted_classes=("car", "barrier")) -> str:
    """Readable table, with the never-predicted classes called out as such."""
    lines = [
        f"  mAP  {metrics['mean_ap']:.4f}      NDS  {metrics['nd_score']:.4f}",
        "",
        f"  {'class':22} {'AP':>8}  {'ATE':>6} {'ASE':>6} {'AOE':>6} "
        f"{'AVE':>6} {'AAE':>6}",
        "  " + "-" * 66,
    ]
    for name, ap in metrics["mean_dist_aps"].items():
        errors = metrics["label_tp_errors"][name]
        note = "" if name in predicted_classes else "   <- never predicted"
        lines.append(
            f"  {name:22} {ap:>8.4f}  {errors['trans_err']:>6.3f} "
            f"{errors['scale_err']:>6.3f} {errors['orient_err']:>6.3f} "
            f"{errors['vel_err']:>6.3f} {errors['attr_err']:>6.3f}{note}")

    predicted = [metrics["mean_dist_aps"][c] for c in predicted_classes
                 if c in metrics["mean_dist_aps"]]
    if predicted:
        lines += ["",
                  f"  mAP over the {len(predicted)} classes this pipeline can "
                  f"emit: {np.mean(predicted):.4f}"]
    return "\n".join(lines)
