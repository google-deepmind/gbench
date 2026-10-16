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

"""Native ScreenSpot GUI grounding vision evaluation suite (a mouse-click tool).

Data: the official ScreenSpot-v2 release of its authors (OS-Copilot/ScreenSpot-v2, OS-Atlas),
pinned and sha256-checked, read exactly as screenspot_pointing reads it
(screenspot_pointing.screenspot_v2_targets): screenspot_{mobile,desktop,web}_v2.json (501 + 334 +
437 = 1,272 targets) and the screenshots of screenspotv2_image.zip, each sent as its own bytes
with the MIME type of those bytes, and each target box converted as ScreenSpot's reference script
converts it (njucckevin/SeeClick pretrain/screenspot_test.py @0ef37ac4: [x, y, w, h] pixels ->
[x0, y0, x1, y1] divided by the image size).

Prompt: gbench's. ScreenSpot defines no model-agnostic prompt (its reference script asks with
SeeClick's own training prompt and output format), so this suite asks the model to click the
element with a `mouse_click` tool whose x and y are in [0, 1000]; the result is always
`leaderboard_comparable: false`. screenspot_pointing asks for a `Point: (x, y)` in text instead.

Scoring: ScreenSpot's click accuracy, as the reference script scores it: the clicked point must
fall inside the target box (edges included). The headline is the accuracy over all 1,272 targets
(the "Avg" column of the ScreenSpot-v2 table in the OS-Atlas paper); the six Mobile / Desktop /
Web x Text / Icon columns are in `screenspot_v2_cells` and `category_accuracy`.

A local ScreenSpot-Pro data dir is scored instead only when GBENCH_SCREENSPOT_DATA_DIR (legacy
alias SCREENSPOT_PRO_DATA_DIR) names one; a named dir without `annotations/` and `images/` is a
hard error, never a silent switch to ScreenSpot-v2. The result records the dataset scored.

Sampling: the reference script decodes with the model's own generation config (no temperature or
token limit of its own), so no max_tokens is sent (screenspot_pointing.OFFICIAL_SAMPLING); the
temperature follows gbench's rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0
without;
`--max-output-tokens` is not applied in either mode (`max_output_tokens_note`). Override the
temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_SCREENSPOT_TEMPERATURE`, which takes precedence over both; the result records
`official_sampling`.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import apply_shard, finalize_shard_fields, current_shard, run_eval_suite
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base
from .screenspot_pointing import (DATASET, DATASET_REVISION, N_TARGETS, OFFICIAL_SAMPLING,
                                  cell_accuracies, image_data_url, screenspot_v2_targets)

logger = logging.getLogger(__name__)

EVAL_NAME = "screenspot"
DOCS_URL = f"docs/evals/{EVAL_NAME}.md"

MOUSE_CLICK_TOOL = {
    "type": "function",
    "function": {
        "name": "mouse_click",
        "description": "Performs a mouse click.",
        "parameters": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "description": 'The button to click. Either "left", "middle" or "right".',
                },
                "repeats": {
                    "type": "integer",
                    "description": "The number of times to click. Default is 1.",
                    "default": 1,
                },
                "x": {
                    "type": "integer",
                    "description": "The normalized x coordinate within the [0, 1000] range of the image.",
                },
                "y": {
                    "type": "integer",
                    "description": "The normalized y coordinate within the [0, 1000] range of the image.",
                },
            },
            "required": ["button", "y", "x"],
        },
    },
}

PROMPT = 'Click on the UI element: "{instruction}".\nUse the computer tool at your disposal.'


def _click_sample(instruction: str, image_url: str, box_1000: Tuple[float, float, float, float],
                  meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]:
    """One request: the prompt and the screenshot in one user turn, with the mouse-click tool.

    No per-sample max_tokens or temperature: run_eval_suite resolves both (the token-limit policy
    and gbench's temperature rule)."""
    content = [
        {"type": "text", "text": PROMPT.format(instruction=instruction)},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    return ([{"role": "user", "content": content}], box_1000,
            {"tools": [MOUSE_CLICK_TOOL], "tool_choice": "auto", **meta})


def _screenspot_pro_dir() -> Optional[str]:
    """The ScreenSpot-Pro data dir named by GBENCH_SCREENSPOT_DATA_DIR (or the legacy
    SCREENSPOT_PRO_DATA_DIR), or None when neither is set."""
    return (os.environ.get("GBENCH_SCREENSPOT_DATA_DIR", "").strip()
            or os.environ.get("SCREENSPOT_PRO_DATA_DIR", "").strip() or None)


def _load_screenspot_pro_local(data_dir: str,
                               limit: Optional[int] = None) -> List[Tuple[Any, Any, Dict[str, Any]]]:
    """Every target of a local ScreenSpot-Pro dir (annotations/*.json + images/), stratified by
    `data_type` when limited. A dir without both subdirectories, or one that yields no target, is
    a hard error."""
    from .swebench_common import infra_required
    annotations_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(annotations_dir) or not os.path.isdir(images_dir):
        raise infra_required(EVAL_NAME, f"the ScreenSpot-Pro data dir {data_dir!r} (set by "
                             "GBENCH_SCREENSPOT_DATA_DIR / SCREENSPOT_PRO_DATA_DIR) has no "
                             "annotations/ and images/ subdirectories; unset the variable to "
                             "score the official ScreenSpot-v2 release", DOCS_URL)
    samples = []
    for ann_file in sorted(os.listdir(annotations_dir)):
        if not ann_file.endswith(".json"):
            continue
        with open(os.path.join(annotations_dir, ann_file), "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            instruction = entry.get("instruction")
            img_filename = entry.get("img_filename")
            bbox = entry.get("bbox")
            if not instruction or not img_filename or not bbox:
                raise RuntimeError(f"{EVAL_NAME}: {ann_file} has an entry without instruction, "
                                   f"img_filename or bbox ({entry!r:.200})")
            img_path = os.path.join(images_dir, img_filename)
            if not os.path.exists(img_path):
                raise RuntimeError(f"{EVAL_NAME}: {ann_file} names {img_filename!r}, which is not "
                                   f"in {images_dir}")
            with open(img_path, "rb") as f:
                image_url, w, h = image_data_url(f.read())
            box = (bbox[0] * 1000 / w, bbox[1] * 1000 / h, bbox[2] * 1000 / w, bbox[3] * 1000 / h)
            data_type = entry.get("ui_type") or entry.get("data_type") or "unknown"
            samples.append(_click_sample(instruction, image_url, box, {
                "category": str(data_type), "data_type": str(data_type),
                "img_filename": img_filename}))
    if not samples:
        raise infra_required(EVAL_NAME, f"the ScreenSpot-Pro data dir {data_dir!r} holds no "
                             "annotated target", DOCS_URL)
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    samples, _ = apply_shard(samples, EVAL_NAME)
    return stratified_sample(samples, limit, lambda s: s[2].get("data_type"), seed="screenspot")


def _load_screenspot_samples(max_soft_tokens: Optional[int] = None, limit: Optional[int] = None
                             ) -> Tuple[List[Tuple[Any, Any, Dict[str, Any]]], str]:
    """(samples, dataset label): the official ScreenSpot-v2 release, or the ScreenSpot-Pro dir a
    variable names (see the module docstring)."""
    pro_dir = _screenspot_pro_dir()
    if pro_dir:
        samples = _load_screenspot_pro_local(pro_dir, limit=limit)
        dataset_label = f"ScreenSpot-Pro (local data dir {pro_dir})"
        logger.info("Loaded %d ScreenSpot-Pro targets from %s.", len(samples), pro_dir)
    else:
        # --shard first, over the full ordered target list, then the --eval-limit draw within the
        # shard (stratified by cell with the same seed screenspot_v2_targets draws a limit with).
        # Unsharded, screenspot_v2_targets draws the limit itself and reads only those images.
        sharded = current_shard() is not None
        targets = screenspot_v2_targets(None if sharded else limit, seed="screenspot",
                                        eval_name=EVAL_NAME, docs_url=DOCS_URL)
        if sharded:
            targets, _ = apply_shard(targets, EVAL_NAME)
            targets = stratified_sample(targets, limit,
                                        lambda t: f"{t['platform']}/{t['data_type']}",
                                        seed="screenspot")
        samples = []
        for target in targets:
            box = tuple(v * 1000.0 for v in target["box"])
            samples.append(_click_sample(target["instruction"], target["image_url"], box, {
                "category": f"{target['platform']}/{target['data_type']}",
                "platform": target["platform"], "data_type": target["data_type"],
                "data_source": target["data_source"], "img_filename": target["img_filename"]}))
        dataset_label = f"{DATASET}@{DATASET_REVISION} (mobile + desktop + web, {N_TARGETS:,})"
    if max_soft_tokens is not None:
        for _msg, _target, extra in samples:
            extra["mm_processor_kwargs"] = {"max_soft_tokens": max_soft_tokens}
    return samples, dataset_label


def parse_click_coordinates(text: str) -> tuple[int | None, int | None]:
    """Extract (x, y) coordinates from model tool call response."""
    patterns = [
        r"mouse_click\{[^}]*x[:\s]*(\d+)[^}]*y[:\s]*(\d+)",
        r"mouse_click\{[^}]*y[:\s]*(\d+)[^}]*x[:\s]*(\d+)",
        r'"x"\s*:\s*(\d+).*?"y"\s*:\s*(\d+)',
        r'"y"\s*:\s*(\d+).*?"x"\s*:\s*(\d+)',
        r"mouse_click\([^)]*x\s*=\s*(\d+)[^)]*y\s*=\s*(\d+)",
        r"mouse_click\([^)]*y\s*=\s*(\d+)[^)]*x\s*=\s*(\d+)",
    ]
    for i, pattern in enumerate(patterns):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            if i in (1, 3, 5):
                return int(m.group(2)), int(m.group(1))
            return int(m.group(1)), int(m.group(2))
    return None, None


def _eval_screenspot(response_text: str, gold_bbox: Any) -> bool:
    """Check if predicted click point (x, y) falls inside the target bbox in [0, 1000] scale."""
    pred_x, pred_y = parse_click_coordinates(response_text)
    if pred_x is None or pred_y is None:
        return False
    x1, y1, x2, y2 = gold_bbox[0], gold_bbox[1], gold_bbox[2], gold_bbox[3]
    return x1 <= pred_x <= x2 and y1 <= pred_y <= y2


def run_screenspot(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    eval_max_soft_tokens: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run the screenspot GUI-grounding suite (the official ScreenSpot-v2 release; a named local
    ScreenSpot-Pro dir instead). The dataset actually scored is recorded on the result."""
    limit = kwargs.get("limit")
    samples, dataset_label = _load_screenspot_samples(eval_max_soft_tokens, limit=limit)
    eval_plan = eval_limit_plan(EVAL_NAME, limit, len(samples), attempts=base_attempts())
    result = run_eval_suite(
        eval_name=EVAL_NAME,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's targets, kept as is
        eval_fn=_eval_screenspot,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,        # already stratified in the loader
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # no max_tokens (temperature: gbench's rule)
    )
    result["eval_limit_plan"] = eval_plan
    traces = result.get("sample_traces") or []
    pro = _screenspot_pro_dir() is not None
    result["dataset"] = dataset_label
    result["metric"] = ("ScreenSpot click accuracy over all targets (the mouse_click point inside "
                        "the target box" + ("" if pro else "; the OS-Atlas ScreenSpot-v2 table's "
                                            "Avg, per cell in screenspot_v2_cells") + ")")
    if not pro:
        result["screenspot_v2_cells"] = cell_accuracies(traces)
    reasons = ["gbench mouse-click tool prompt: ScreenSpot defines no model-agnostic prompt (its "
               "reference script uses SeeClick's own prompt and output format)"]
    if pro:
        reasons.append("a local ScreenSpot-Pro data dir was scored, not the pinned ScreenSpot-v2 "
                       "release")
    if limit:
        total = "all" if pro else f"{N_TARGETS:,}"
        reasons.append(f"--eval-limit {limit} subset of the {total} targets")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (the reference script uses the "
                       "model's own generation config)")
    from .vqa_common import lost_requests_reason
    lost = lost_requests_reason(traces)
    if lost:
        reasons.append(lost)
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields,
    # which `gbench --merge` drops once the shards' union is the full set).
    return finalize_shard_fields(result)
