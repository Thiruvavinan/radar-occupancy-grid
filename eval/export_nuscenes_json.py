"""Write detections in the exact JSON schema the devkit's DetectionEval reads.

The schema here was verified against the **installed** nuscenes-devkit source
(`DetectionBox.deserialize` and `eval.common.loaders.load_prediction`), not
recalled from memory:

    {
      "meta":    {"use_camera", "use_lidar", "use_radar", "use_map",
                  "use_external"},          # all five keys required
      "results": {sample_token: [box, ...]}
    }

with each box carrying `sample_token`, `translation`, `size`, `rotation`,
`velocity`, `detection_name`, `detection_score`, `attribute_name`.

Two constraints that are easy to miss and fail loudly at eval time:

  * every sample in the evaluation split must appear in `results`, even with
    an empty list -- a missing token aborts the run
  * at most `max_boxes_per_sample` (500) boxes per sample
"""

from __future__ import annotations

import json
import os

from nuscenes.eval.detection.config import config_factory
from nuscenes.utils import splits

#: What this pipeline actually consumed. `use_map` is set at write time because
#: it depends on whether road masking was active.
BASE_META = {
    "use_camera": False,
    "use_lidar": False,
    "use_radar": True,
    "use_map": False,
    "use_external": False,
}


def sample_tokens_for_split(nusc, eval_set: str) -> list:
    """Every sample token in an evaluation split, in scene order."""
    scene_names = getattr(splits, eval_set, None)
    if scene_names is None:
        raise ValueError(f"unknown split {eval_set!r}; "
                         f"try 'mini_val', 'mini_train' or 'val'")

    tokens = []
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue
        token = scene["first_sample_token"]
        while token:
            tokens.append(token)
            token = nusc.get("sample", token)["next"]
    return tokens


def write_results(path, detections_by_sample, required_tokens,
                  use_map=False, max_boxes=None):
    """Serialise detections, padding any sample with no detections.

    ``detections_by_sample`` maps sample_token -> list[RadarDetection].
    ``required_tokens`` is every token the split expects.
    """
    if max_boxes is None:
        max_boxes = config_factory("detection_cvpr_2019").max_boxes_per_sample

    results, empty, truncated = {}, 0, 0
    for token in required_tokens:
        boxes = detections_by_sample.get(token, [])
        if not boxes:
            empty += 1
        if len(boxes) > max_boxes:
            # Keep the highest-scoring ones; the eval would reject the file
            # outright otherwise.
            boxes = sorted(boxes, key=lambda d: d.detection_score,
                           reverse=True)[:max_boxes]
            truncated += 1
        results[token] = [box.to_submission() for box in boxes]

    meta = dict(BASE_META, use_map=bool(use_map))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"meta": meta, "results": results}, handle)

    return {
        "path": path,
        "samples": len(results),
        "boxes": sum(len(v) for v in results.values()),
        "samples_with_no_detections": empty,
        "samples_truncated": truncated,
    }


def validate(path, nusc, eval_set: str) -> dict:
    """Re-read the file through the devkit's own loader before trusting it.

    Cheaper to fail here, with a readable message, than inside DetectionEval.
    """
    from nuscenes.eval.common.loaders import load_prediction
    from nuscenes.eval.detection.data_classes import DetectionBox

    config = config_factory("detection_cvpr_2019")
    boxes, meta = load_prediction(path, config.max_boxes_per_sample,
                                  DetectionBox, verbose=False)

    expected = set(sample_tokens_for_split(nusc, eval_set))
    got = set(boxes.sample_tokens)
    missing = expected - got
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(expected)} sample tokens are missing from "
            f"{path}; DetectionEval requires an entry for every sample in "
            f"'{eval_set}' (an empty list is fine)")

    return {"loaded_samples": len(got), "meta": meta,
            "extra_samples": len(got - expected)}
