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

"""ScreenSpot-v2 with a text pointing prompt (registered as screenspot_pointing).

Renamed from `semantic_keypoint`: there is no upstream "semantic keypoint" benchmark. The old
name still runs this suite, with a deprecation warning.

Data: the official ScreenSpot-v2 release of its authors (OS-Copilot/ScreenSpot-v2, OS-Atlas,
pinned below): screenspot_{mobile,desktop,web}_v2.json (501 + 334 + 437 = 1,272 targets) and the
screenshots of screenspotv2_image.zip, every file sha256-checked. Each screenshot is sent with the
MIME type of its bytes (109 of the 730 are JPEG, 78 of them named .png). The target box is
converted as ScreenSpot's reference script does (njucckevin/SeeClick pretrain/screenspot_test.py
@0ef37ac4: [x, y, w, h] pixels -> [x0, y0, x1, y1] divided by the image size).

Prompt: gbench's. ScreenSpot defines no model-agnostic prompt: its reference script asks with
SeeClick's own training prompt and reads SeeClick's own output format (a [0, 1] point or a Qwen-VL
<box>), and every published model is scored with its own grounding prompt. This suite asks for the
element's centre as `Point: (x, y)` in [0, 1000] (a step-by-step variant under `--thinking`), so the
result is always `leaderboard_comparable: false`; `screenspot` asks with a mouse-click tool.

Scoring: ScreenSpot's click accuracy, as the reference script scores it: the predicted point must
fall inside the target box (edges included). The headline is the accuracy over all 1,272 targets,
the "Avg" column of the ScreenSpot-v2 table in the OS-Atlas paper (a micro average); the six
Mobile / Desktop / Web x Text / Icon columns are reported too.

Sampling: the reference script decodes with the model's own generation config
(`model.generation_config = GenerationConfig.from_pretrained(...)`, no temperature or token limit
of its own), so no max_tokens is sent (base.OfficialSampling); the temperature follows gbench's
rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0 without; `--max-output-tokens` is not applied in either mode
(`max_output_tokens_note`). Override the temperature for a whole run with `--temperature`, or for
this suite alone with
`GBENCH_SCREENSPOT_POINTING_TEMPERATURE` (the pre-rename `GBENCH_SEMANTIC_KEYPOINT_TEMPERATURE` is
still read as a deprecated alias), which takes precedence over both; the result records
`official_sampling`.
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import OfficialSampling, run_eval_suite
from .metrics import record_budget_cutoffs
from .sampling import parse_shard, shard_select, stratified_indices

logger = logging.getLogger(__name__)

#: The registry name (renamed from `semantic_keypoint`).
EVAL_NAME = "screenspot_pointing"
DOCS_URL = f"docs/evals/{EVAL_NAME}.md"

#: The ScreenSpot-v2 release (OS-Copilot/ScreenSpot-v2 on the HF Hub, Apache-2.0), pinned.
DATASET = "OS-Copilot/ScreenSpot-v2"
DATASET_REVISION = "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c"
#: Platform -> (annotation file, its sha256), in the reference script's task order.
SPLITS = {
    "mobile": ("screenspot_mobile_v2.json",
               "fdd1595b64179e31407eed5929fa12e80adc6ba2ee372a3802d36da379ea1825"),
    "desktop": ("screenspot_desktop_v2.json",
                "96f494e0b28eead9187c67794b02bde63350b9cb753acc5fc0384fa77c7182a4"),
    "web": ("screenspot_web_v2.json",
            "3e89e245d69f56a8e8f430917dc02f95b6fc36f8c8f2836c2cafe4dad10e823d"),
}
SPLIT_SIZES = {"mobile": 501, "desktop": 334, "web": 437}
#: The screenshots: screenspotv2_image/<img_filename> inside this zip (the HF LFS oid).
IMAGE_ZIP = ("screenspotv2_image.zip",
             "750555b8b06e3898634c610cd904600feb7ec7ca530e625a00c1416a73d829ee")
IMAGE_DIR_IN_ZIP = "screenspotv2_image"
N_TARGETS = 1272
DATA_TYPES = ("text", "icon")
#: The six columns of the ScreenSpot-v2 table (OS-Atlas paper), as category keys.
CELLS = tuple(f"{platform}/{data_type}" for platform in SPLITS for data_type in DATA_TYPES)

OFFICIAL_SAMPLING = OfficialSampling(
    reference="njucckevin/SeeClick pretrain/screenspot_test.py @0ef37ac4 (the model's own "
              "GenerationConfig; no temperature or max_new_tokens of its own)",
    send_temperature=False, send_max_tokens=False)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(filename: str, sha256: str, eval_name: str = EVAL_NAME, docs_url: str = DOCS_URL) -> str:
    """One pinned file of the release, sha256-checked (infra_required when it cannot be fetched,
    naming `eval_name`, the suite that needs it)."""
    from huggingface_hub import hf_hub_download
    from .swebench_common import infra_required
    try:
        path = hf_hub_download(DATASET, filename, repo_type="dataset", revision=DATASET_REVISION)
    except Exception as e:                                              # noqa: BLE001
        raise infra_required(eval_name, f"{DATASET}@{DATASET_REVISION} {filename} could not be "
                             f"downloaded ({type(e).__name__}: {e})", docs_url)
    digest = _sha256(path)
    if digest != sha256:
        raise RuntimeError(f"{eval_name}: {DATASET}@{DATASET_REVISION} {filename} has sha256 "
                           f"{digest}, expected {sha256}")
    return path


def image_data_url(data: bytes) -> Tuple[str, int, int]:
    """(data URL, width, height) of an image, its MIME type read from the bytes (PIL's format),
    not from the file name: some of the release's .png files hold JPEG data."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        mime = Image.MIME.get(img.format or "")
        width, height = img.size
    if not mime:
        raise RuntimeError(f"{EVAL_NAME}: a screenshot has an image format PIL gives no MIME type "
                           f"for ({img.format!r})")
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}", width, height


def normalized_box(bbox: List[float], width: int, height: int) -> List[float]:
    """screenspot_test.py: [x, y, w, h] pixels -> [x0, y0, x1, y1] / image size."""
    x, y, w, h = (float(v) for v in bbox[:4])
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def screenspot_v2_targets(limit: Optional[int] = None, seed: str = "semantic_keypoint",
                          eval_name: str = EVAL_NAME,
                          docs_url: str = DOCS_URL) -> List[Dict[str, Any]]:
    """Every target of the pinned ScreenSpot-v2 release (limit: stratified by cell with `seed`).

    Each target is {platform, data_type, instruction, img_filename, data_source, image_url (the
    screenshot's bytes as a data URL with the MIME type of those bytes), box ([x0, y0, x1, y1] as
    fractions of the image size, converted as screenspot_test.py converts it)}. Shared by
    screenspot_pointing and screenspot; `eval_name` / `docs_url` name the suite in errors."""
    names = {} if eval_name == EVAL_NAME else {"eval_name": eval_name, "docs_url": docs_url}
    items: List[Tuple[str, Dict[str, Any]]] = []
    for platform, (filename, sha256) in SPLITS.items():
        with open(_fetch(filename, sha256, **names), encoding="utf-8") as fh:
            rows = json.load(fh)
        if len(rows) != SPLIT_SIZES[platform]:
            raise RuntimeError(f"{eval_name}: {filename} has {len(rows)} targets, expected "
                               f"{SPLIT_SIZES[platform]}")
        items.extend((platform, row) for row in rows)
    keep = stratified_indices([f"{p}/{r.get('data_type')}" for p, r in items], limit, seed=seed)
    zip_path = _fetch(*IMAGE_ZIP, **names)
    logger.info("Loading %d of %d ScreenSpot-v2 targets from %s@%s.", len(keep), len(items),
                DATASET, DATASET_REVISION[:8])

    targets: List[Dict[str, Any]] = []
    images: Dict[str, Tuple[str, int, int]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for i in keep:
            platform, item = items[i]
            instruction = item.get("instruction", "")
            name, bbox = item.get("img_filename"), item.get("bbox")
            if not name or not bbox or len(bbox) < 4:
                raise RuntimeError(f"{eval_name}: a {platform} target has no image or bbox "
                                   f"(instruction {instruction!r})")
            if name not in images:
                try:
                    data = zf.read(f"{IMAGE_DIR_IN_ZIP}/{name}")
                except KeyError:
                    raise RuntimeError(f"{eval_name}: {IMAGE_ZIP[0]} has no image {name!r} "
                                       f"(instruction {instruction!r})")
                images[name] = image_data_url(data)
            data_url, width, height = images[name]
            targets.append({"platform": platform, "data_type": item.get("data_type", "text"),
                            "instruction": instruction, "img_filename": name,
                            "data_source": item.get("data_source"), "image_url": data_url,
                            "box": normalized_box(bbox, width, height)})
    return targets


def _load_screenspot_pointing_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Every ScreenSpot-v2 target with the gbench pointing prompt (limit: stratified by cell)."""
    samples = []
    for target in screenspot_v2_targets(limit, seed="semantic_keypoint"):
        instruction, box = target["instruction"], target["box"]
        # The box is a fraction of the image size (an edge may sit a hair past 1.0).
        gold_point = {"x": (box[0] + box[2]) / 2.0 * 1000.0,
                      "y": (box[1] + box[3]) / 2.0 * 1000.0, "bbox": box,
                      "bbox_normalized": True}

        if enable_thinking:
            prompt = (
                f"Target Element: {instruction}\n"
                "Locate the precise semantic keypoint/center coordinate for the target element in the image. "
                "Analyze the visual boundaries step by step, then output the final point in the format: Point: (x, y) where x and y are in [0, 1000]."
            )
        else:
            prompt = (
                f"Target Element: {instruction}\n"
                "Output the normalized keypoint coordinate [x, y] within [0, 1000] corresponding to the center of the target element. "
                "Format: Point: (x, y)"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": target["image_url"]}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, gold_point, {
            "category": f"{target['platform']}/{target['data_type']}",
            "platform": target["platform"], "data_type": target["data_type"],
            "data_source": target["data_source"], "img_filename": target["img_filename"]}))
    return samples


def _eval_screenspot_pointing(response_text: str, gold_point: Any) -> bool:
    """ScreenSpot-v2 click-accuracy rule: the predicted point falls INSIDE the target bbox.

    The old check accepted any point within a fixed 50-unit radius of the box CENTRE, which
    both over-accepts tiny boxes (a point well outside a 20-unit-wide element still passes)
    and under-accepts large ones (a valid click in a wide banner beyond 50 units fails).
    Point-in-bbox is the published metric and needs no arbitrary radius.
    """
    if not response_text or not isinstance(gold_point, dict):
        return False
    bbox = gold_point.get("bbox")
    if not bbox or len(bbox) < 4:
        return False

    # Pattern matching for (x, y), [x, y], or {"x": x, "y": y}
    pred_x, pred_y = None, None
    pt_matches = re.findall(r"\(?\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)?", response_text)
    if pt_matches:
        pred_x, pred_y = float(pt_matches[-1][0]), float(pt_matches[-1][1])
    else:
        x_match = re.search(r'"x"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response_text)
        y_match = re.search(r'"y"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response_text)
        if x_match and y_match:
            pred_x = float(x_match.group(1))
            pred_y = float(y_match.group(1))
    if pred_x is None or pred_y is None:
        return False

    # Put the gold bbox in the same [0, 1000] space the gold centre used.
    x0, y0, x1, y1 = (float(c) for c in bbox[:4])
    if gold_point.get("bbox_normalized") or all(0.0 <= c <= 1.0 for c in (x0, y0, x1, y1)):
        x0, y0, x1, y1 = x0 * 1000.0, y0 * 1000.0, x1 * 1000.0, y1 * 1000.0
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    # Normalize a [0, 1]-scale prediction to [0, 1000] to match the box.
    if pred_x <= 1.0 and pred_y <= 1.0 and x1 > 1.0:
        pred_x *= 1000.0
        pred_y *= 1000.0

    return x0 <= pred_x <= x1 and y0 <= pred_y <= y1


def cell_accuracies(traces: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Accuracy x100 per Mobile / Desktop / Web x Text / Icon cell (None for an empty cell), over
    the scored targets as base's accuracy (a request with no reply from the endpoint is a harness
    error, left out)."""
    from .base import is_scored
    out: Dict[str, Optional[float]] = {}
    for cell in CELLS:
        hits = [bool(t.get("is_correct")) for t in traces
                if t.get("category") == cell and is_scored(t)]
        out[cell] = round(100.0 * sum(hits) / len(hits), 2) if hits else None
    return out


def run_screenspot_pointing(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    eval_name: str = EVAL_NAME,
    **kwargs,
) -> Dict[str, Any]:
    """Run ScreenSpot-v2 (official release) with the gbench pointing prompt.

    `eval_name` selects the `GBENCH_<NAME>_TEMPERATURE` override (and the result's name).
    """
    samples = _load_screenspot_pointing_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    # The plan counts the targets base runs on this node (GBENCH_SHARD's shard of the drawn list).
    on_shard = shard_select(samples, parse_shard(os.environ.get("GBENCH_SHARD")))
    eval_plan = eval_limit_plan(eval_name, kwargs.get("limit"), len(on_shard),
                                attempts=base_attempts())
    result = run_eval_suite(
        eval_name=eval_name,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_screenspot_pointing,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,        # already stratified by cell in the loader
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,
    )
    result["eval_limit_plan"] = eval_plan
    traces = result.get("sample_traces") or []
    result["metric"] = ("ScreenSpot-v2 click accuracy over all targets (predicted point inside the "
                        "target box; the OS-Atlas table's Avg) under a gbench 'Point: (x, y)' "
                        "prompt; per Mobile / Desktop / Web x Text / Icon cell in "
                        "screenspot_v2_cells")
    result["screenspot_v2_cells"] = cell_accuracies(traces)
    result["dataset"] = f"{DATASET}@{DATASET_REVISION} (mobile + desktop + web, {N_TARGETS:,})"
    reasons = ["gbench pointing prompt: ScreenSpot defines no model-agnostic prompt (its reference "
               "script uses SeeClick's own prompt and output format); the screenspot suite asks "
               "with a mouse-click tool"]
    if kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs['limit']} subset of the {N_TARGETS:,} targets")
    if os.environ.get("GBENCH_SHARD"):
        reasons.append(f"GBENCH_SHARD {os.environ['GBENCH_SHARD']} shard")
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
    return result
