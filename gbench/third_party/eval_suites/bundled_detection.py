# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zero-shot COCO object detection by a chat VLM, scored by the official COCO evaluation.

Data: the official MS-COCO 2017 release (images.cocodataset.org): all 5,000 val2017 images
(val2017.zip) and their ground truth (annotations/instances_val2017.json of
annotations_trainval2017.zip: 36,781 boxes of the 80 categories, 446 of them crowd regions; 48
images have no box), both sha256-pinned. The prompt gives the fixed COCO-80 vocabulary and asks
for a JSON list of `{label, box_2d, confidence}`.

Scoring: pycocotools COCOeval (iouType "bbox"), the COCO detection challenge's own evaluator, on
every requested image: each detection becomes a COCO result (category id from its label, the
box in pixels, the model's confidence as the score, 1.0 when absent); crowd regions, the area
ranges and the 100-detections-per-image cap apply as COCOeval applies them. The headline is AP
@[.50:.95] (area all, 100 detections), the challenge's primary metric; the other eleven COCOeval
summary numbers are reported too. An image whose request failed or ran over the context window
has no detections, and its ground truth still counts against recall.

The prompt is gbench's (COCO defines no prompt for a chat model) and the scores are the model's
own confidences, so the result is always `leaderboard_comparable: false` (see
`run_bundled_detection`).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BUNDLED_DETECTION_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import gbench_cache_dir, is_scored, run_eval_suite
from .metrics import record_budget_cutoffs
from .sampling import parse_shard, shard_select, stratified_indices

logger = logging.getLogger(__name__)

EVAL_NAME = "bundled_detection"
DOCS_URL = f"docs/evals/{EVAL_NAME}.md"

#: The official COCO 2017 files (images.cocodataset.org), with their sha256.
ANNOTATIONS_ZIP = ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                   "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268")
INSTANCES_MEMBER = ("annotations/instances_val2017.json",
                    "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f")
IMAGES_ZIP = ("http://images.cocodataset.org/zips/val2017.zip",
              "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05")
N_IMAGES = 5000
N_ANNOTATIONS = 36781

#: The 80 COCO categories in id order (instances_val2017.json `categories`, checked at load).
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
#: Their COCO category ids (not contiguous), same order.
COCO_CATEGORY_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                     23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                     65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
                     89, 90]
CATEGORY_ID = dict(zip(COCO_CATEGORIES, COCO_CATEGORY_IDS))
#: COCOeval.summarize() stats, in order.
COCO_STATS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")

PROMPT = (
    "Detect every object in this image that belongs to the COCO-80 categories:\n"
    f"{', '.join(COCO_CATEGORIES)}.\n\n"
    "Output ONLY a JSON array; each element has 'label' (one of the categories), "
    "'box_2d' ([ymin, xmin, ymax, xmax] normalized to [0, 1000]), and 'confidence' "
    "(0.0-1.0). Emit one element per detected object."
)


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union between two boxes in [ymin, xmin, ymax, xmax] format."""
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, xmax1 - xmin1) * max(0.0, ymax1 - ymin1)
    area2 = max(0.0, xmax2 - xmin2) * max(0.0, ymax2 - ymin2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coco_dir() -> str:
    """$GBENCH_CACHE_DIR/coco (default ~/.cache/gbench/coco)."""
    return os.path.join(gbench_cache_dir(), "coco")


def _fetch(url: str, sha256: str) -> str:
    """A pinned official COCO file in the cache, downloaded once and sha256-checked each run."""
    import urllib.request
    from .swebench_common import infra_required
    os.makedirs(_coco_dir(), exist_ok=True)
    path = os.path.join(_coco_dir(), os.path.basename(url))
    if not os.path.exists(path):
        tmp = path + ".part"
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    out.write(chunk)
        except Exception as e:                                          # noqa: BLE001
            raise infra_required(EVAL_NAME, f"{url} could not be downloaded "
                                 f"({type(e).__name__}: {e})", DOCS_URL)
        os.replace(tmp, path)
    digest = _sha256_file(path)
    if digest != sha256:
        raise RuntimeError(f"{EVAL_NAME}: {path} has sha256 {digest}, expected {sha256} (delete "
                           "it to download it again)")
    return path


def load_ground_truth() -> Dict[str, Any]:
    """instances_val2017.json, from the pinned annotations zip (extracted once, sha256-checked)."""
    target = os.path.join(_coco_dir(), os.path.basename(INSTANCES_MEMBER[0]))
    if not (os.path.exists(target) and _sha256_file(target) == INSTANCES_MEMBER[1]):
        with zipfile.ZipFile(_fetch(*ANNOTATIONS_ZIP)) as zf:
            data = zf.read(INSTANCES_MEMBER[0])
        if hashlib.sha256(data).hexdigest() != INSTANCES_MEMBER[1]:
            raise RuntimeError(f"{EVAL_NAME}: {INSTANCES_MEMBER[0]} has the wrong sha256")
        with open(target + ".part", "wb") as fh:
            fh.write(data)
        os.replace(target + ".part", target)
    with open(target, encoding="utf-8") as fh:
        gt = json.load(fh)
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    if (len(gt["images"]) != N_IMAGES or len(gt["annotations"]) != N_ANNOTATIONS
            or [c["name"] for c in cats] != COCO_CATEGORIES
            or [c["id"] for c in cats] != COCO_CATEGORY_IDS):
        raise RuntimeError(f"{EVAL_NAME}: {INSTANCES_MEMBER[0]} is not the COCO 2017 val "
                           "ground truth this suite expects")
    return gt


def _require_pycocotools() -> None:
    try:
        from pycocotools.coco import COCO  # noqa: F401
        from pycocotools.cocoeval import COCOeval  # noqa: F401
    except ImportError:
        from .swebench_common import infra_required
        raise infra_required(EVAL_NAME, "scoring is pycocotools COCOeval, the COCO evaluator; "
                             "install pycocotools (pip install -e '.[evals]')", DOCS_URL)


def _load_bundled_detection_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """One request per val2017 image, with that image's ground truth as the gold.

    `limit` (--eval-limit) takes a seeded subset spread over the images' most frequent category
    ("none" for the 48 images without a box).
    """
    gt = load_ground_truth()
    cat_name = {c["id"]: c["name"] for c in gt["categories"]}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in gt["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    images = gt["images"]

    def _key(img: Dict[str, Any]) -> str:
        counts: Dict[str, int] = {}
        for ann in anns_by_img.get(img["id"], []):
            counts[cat_name[ann["category_id"]]] = counts.get(cat_name[ann["category_id"]], 0) + 1
        return max(sorted(counts), key=lambda k: counts[k]) if counts else "none"

    keys = [_key(img) for img in images]
    keep = stratified_indices(keys, limit, seed=EVAL_NAME)
    zip_path = _fetch(*IMAGES_ZIP)
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        for i in keep:
            img = images[i]
            try:
                data = zf.read(f"val2017/{img['file_name']}")
            except KeyError:
                raise RuntimeError(f"{EVAL_NAME}: val2017.zip has no image {img['file_name']!r}")
            width, height = int(img["width"]), int(img["height"])
            objects = []
            for ann in anns_by_img.get(img["id"], []):
                x, y, w, h = (float(v) for v in ann["bbox"])
                objects.append({
                    "id": ann["id"], "category_id": ann["category_id"],
                    "label": cat_name[ann["category_id"]], "bbox": [x, y, w, h],
                    "area": float(ann["area"]), "iscrowd": int(ann.get("iscrowd", 0)),
                    # [ymin, xmin, ymax, xmax] in [0, 1000], for the per-image hit rate
                    "box_2d": [y / height * 1000.0, x / width * 1000.0,
                               (y + h) / height * 1000.0, (x + w) / width * 1000.0]})
            gold = {"image_id": img["id"], "width": width, "height": height, "objects": objects}
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            samples.append((messages, gold, {"category": keys[i], "image_id": img["id"]}))

    logger.info("Loaded %d of %d COCO val2017 images for bundled_detection.", len(samples),
                len(images))
    return samples


def _parse_detections(response_text: str) -> List[Dict[str, Any]]:
    """Parse the model's detections into [{label, box:[ymin,xmin,ymax,xmax] in [0,1000], conf}]."""
    if not response_text:
        return []
    raw: Any = None
    m = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
    try:
        raw = json.loads(m.group(0) if m else response_text)
    except Exception:
        raw = None
    dets: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for pred in raw:
            if not isinstance(pred, dict):
                continue
            box = pred.get("box_2d") or pred.get("bbox") or pred.get("box")
            if not box or len(box) != 4:
                continue
            try:
                box = [float(x) for x in box]
            except (ValueError, TypeError):
                continue
            if max(box) <= 1.0:                       # [0,1] -> [0,1000]
                box = [x * 1000.0 for x in box]
            conf = pred.get("confidence", pred.get("score", 1.0))
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 1.0
            dets.append({"label": str(pred.get("label", "")).strip().lower(),
                         "box": box, "conf": max(0.0, min(1.0, conf))})
    return dets


def _gold_objects(gold: Any) -> List[Dict[str, Any]]:
    """The gold boxes of one image: the loader's {"objects": [...]} or a bare list of
    {"label", "box_2d"} (box_2d = [ymin, xmin, ymax, xmax] in [0, 1000])."""
    if isinstance(gold, dict):
        return list(gold.get("objects") or [])
    return list(gold or [])


def _eval_bundled_detection(response_text: str, gold_objects: Any) -> bool:
    """Secondary per-image flag: a correctly labelled box at IoU >= 0.5 with a non-crowd gold box.
    The headline is COCOeval AP (run_bundled_detection)."""
    dets = _parse_detections(response_text)
    golds = [g for g in _gold_objects(gold_objects) if not g.get("iscrowd")]
    for d in dets:
        for gold in golds:
            if d["label"] and d["label"] == str(gold.get("label", "")).strip().lower():
                if _compute_iou(d["box"], gold.get("box_2d")) >= 0.5:
                    return True
    return False


def _image_ids(traces: List[Dict[str, Any]]) -> List[int]:
    """One COCO image id per trace, numbered in the order of the COCO image ids.

    Repeated attempts of one image are scored as separate images, so each trace gets its own
    id; numbering them in COCO id order keeps COCOeval's order (it sorts imgIds), which breaks
    ties between equal scores, the same as on the official ground truth.
    """
    def _key(i: int) -> Tuple[int, int]:
        gold = traces[i].get("gold_answer")
        return (int(gold["image_id"]) if isinstance(gold, dict) else i, i)
    order = sorted(range(len(traces)), key=_key)
    ids = [0] * len(traces)
    for rank, i in enumerate(order):
        ids[i] = rank
    return ids


def coco_ground_truth(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A COCO ground-truth dataset for the requested images, one image per trace (_image_ids).

    A trace whose gold is the loader's dict carries the official boxes (pixels, segmentation
    area, iscrowd); a bare {"label", "box_2d"} list is read as boxes on a 1000 x 1000 image.
    """
    images, annotations = [], []
    for image_id, t in zip(_image_ids(traces), traces):
        gold = t.get("gold_answer")
        if isinstance(gold, dict):
            width, height = int(gold["width"]), int(gold["height"])
            objs = [(o["category_id"], [float(v) for v in o["bbox"]], float(o["area"]),
                     int(o.get("iscrowd", 0))) for o in gold.get("objects") or []]
        else:
            width = height = 1000
            objs = []
            for o in gold or []:
                ymin, xmin, ymax, xmax = (float(v) for v in o["box_2d"])
                cat = CATEGORY_ID.get(str(o.get("label", "")).strip().lower())
                if cat is not None:
                    objs.append((cat, [xmin, ymin, xmax - xmin, ymax - ymin],
                                 (xmax - xmin) * (ymax - ymin), 0))
        images.append({"id": image_id, "width": width, "height": height})
        for cat, bbox, area, crowd in objs:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": cat, "bbox": bbox, "area": area,
                                "iscrowd": crowd})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n} for n, i in CATEGORY_ID.items()]}


def coco_detections(traces: List[Dict[str, Any]], dataset: Dict[str, Any]
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Each trace's parsed detections as COCO results ({image_id, category_id, bbox, score}).

    A label outside the 80 categories has no category id and a box without positive width and
    height has no area: neither can be a COCO result, so both are dropped and counted.
    """
    sizes = {img["id"]: (img["width"], img["height"]) for img in dataset["images"]}
    results: List[Dict[str, Any]] = []
    counts = {"unknown_labels": 0, "invalid_boxes": 0}
    for image_id, t in zip(_image_ids(traces), traces):
        width, height = sizes[image_id]
        for d in _parse_detections(t.get("response_text") or ""):
            cat = CATEGORY_ID.get(d["label"])
            if cat is None:
                counts["unknown_labels"] += 1
                continue
            ymin, xmin, ymax, xmax = d["box"]
            x, y = xmin / 1000.0 * width, ymin / 1000.0 * height
            w, h = (xmax - xmin) / 1000.0 * width, (ymax - ymin) / 1000.0 * height
            if w <= 0 or h <= 0:
                counts["invalid_boxes"] += 1
                continue
            results.append({"image_id": image_id, "category_id": cat, "bbox": [x, y, w, h],
                            "score": d["conf"]})
    return results, counts


def _compute_map(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """pycocotools COCOeval (bbox) over the traces: the 12 summary stats, AP as `map`."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset = coco_ground_truth(traces)
    results, counts = coco_detections(traces, dataset)
    stats: Dict[str, Optional[float]] = {name: 0.0 for name in COCO_STATS}
    quiet = io.StringIO()
    if results and dataset["annotations"]:
        with contextlib.redirect_stdout(quiet):
            coco_gt = COCO()
            coco_gt.dataset = dataset
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(results)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = sorted(img["id"] for img in dataset["images"])
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        # COCOeval reports -1 for a number with no ground truth (e.g. no small boxes).
        stats = {name: (None if float(v) < 0 else float(v))
                 for name, v in zip(COCO_STATS, evaluator.stats)}
    elif not dataset["annotations"]:
        stats = {name: None for name in COCO_STATS}
    return {"stats": stats, "map": stats["AP"], "map_50": stats["AP50"],
            "detections": len(results), "images": len(dataset["images"]), **counts}


def run_bundled_detection(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run zero-shot COCO val2017 detection; headline = COCOeval AP@[.50:.95] x100."""
    _require_pycocotools()
    samples = _load_bundled_detection_samples(limit=kwargs.get("limit"))
    # The plan counts the images base runs on this node (GBENCH_SHARD's shard of the drawn list).
    on_shard = shard_select(samples, parse_shard(os.environ.get("GBENCH_SHARD")))
    eval_plan = eval_limit_plan(EVAL_NAME, kwargs.get("limit"), len(on_shard),
                                attempts=base_attempts())
    result = run_eval_suite(
        eval_name=EVAL_NAME,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_bundled_detection,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,        # the loader already drew the --eval-limit subset
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    result["eval_limit_plan"] = eval_plan
    # Every scored trace goes in: COCO counts ground truth on every image, so a reply with no
    # usable box is an image with zero detections, not an image with no objects. A request that
    # got no reply from the endpoint is a harness error (base marks it scoring_excluded): its
    # image is left out of the evaluation, as it is out of base's accuracy, never scored as an
    # image with no detection. With no image scored the score is withheld.
    traces = result.get("sample_traces") or []
    scored = [t for t in traces if is_scored(t)]
    coco = _compute_map(scored)

    def _x100(v: Optional[float]) -> Optional[float]:
        return None if v is None else round(v * 100.0, 2)

    result["hit_rate_any_box"] = result.get("accuracy")
    # base's per-image flags (any box returned) are a secondary diagnostic: their count and their
    # tally by each image's most frequent category move to their own fields. COCOeval's AP is a
    # corpus metric, so no category row or correct count stands next to it.
    result["images_with_any_box"] = result.get("correct_answers")
    result["category_hit_rate"] = result.get("category_accuracy") or {}
    result["category_accuracy"] = {}
    result["correct_answers"] = None
    result["correct_answers_note"] = ("not applicable: the headline is COCOeval AP over the "
                                      "scored images; images_with_any_box counts the images with "
                                      "at least one usable box (hit_rate_any_box)")
    result["coco_stats"] = {name: _x100(v) for name, v in coco["stats"].items()}
    result["map_50_95"] = _x100(coco["map"])
    result["map_50"] = _x100(coco["map_50"])
    result["coco_detections"] = coco["detections"]
    result["coco_unknown_labels"] = coco["unknown_labels"]
    result["coco_invalid_boxes"] = coco["invalid_boxes"]
    result["metric"] = ("COCO AP@[.50:.95] (pycocotools COCOeval, bbox, area all, 100 detections); "
                        "the other COCOeval summary numbers in coco_stats")
    result["accuracy"] = result["map_50_95"] if scored else None
    if not scored and traces:
        result["accuracy_withheld_reason"] = result.get("accuracy_withheld_reason") or (
            "no image was scored: " + str(result.get("status_reason")
                                          or "every request left out of the score"))
    result["dataset"] = ("COCO 2017 val (images.cocodataset.org val2017.zip + "
                         "annotations_trainval2017.zip instances_val2017.json)")
    # The images scored; repeated attempts of one image count once.
    n_images = len({t.get("source_sample_idx", i) for i, t in enumerate(traces) if is_scored(t)})
    scope = f"on {n_images:,} of the {N_IMAGES:,} val2017 images"
    if kwargs.get("limit"):
        scope += f" (--eval-limit {kwargs['limit']} subset)"
    if os.environ.get("GBENCH_SHARD"):
        scope += f" (GBENCH_SHARD {os.environ['GBENCH_SHARD']} shard)"
    reasons = ["zero-shot chat-VLM detection with gbench's prompt (COCO defines none for a chat "
               "model), ranked by the model's self-reported confidences (1.0 when absent), "
               f"scored by COCOeval {scope}"]
    from .vqa_common import lost_requests_reason
    lost = lost_requests_reason(traces)
    if lost:
        reasons.append(lost)
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
