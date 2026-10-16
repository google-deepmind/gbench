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

"""Generic JSONL Evaluation Suite runner for zero-code custom benchmark evaluation."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from .base import run_eval_suite
from .fc_common import base_suite_plan, run_attempts

logger = logging.getLogger(__name__)


def _load_custom_jsonl_samples(
    jsonl_path: Union[str, Path],
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load evaluation samples from a user-provided JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom JSONL evaluation dataset not found: {path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # 1. Format messages
            if "messages" in item and isinstance(item["messages"], list):
                messages = item["messages"]
            elif "prompt" in item:
                messages = [{"role": "user", "content": item["prompt"]}]
            elif "question" in item:
                messages = [{"role": "user", "content": item["question"]}]
            elif "input" in item:
                messages = [{"role": "user", "content": item["input"]}]
            else:
                # A row with nothing to send would silently shrink the denominator.
                raise ValueError(
                    f"custom_jsonl: {path.name} line {idx + 1} has no prompt (one of "
                    "`messages`, `prompt`, `question`, `input`); fix or remove the row")

            # 2. Extract gold answer & eval_type.
            # First PRESENT key wins, by presence not truthiness: an `or` chain dropped a
            # legitimately falsy gold (0, 0.0, "", False) through to the next key and left
            # the row with gold=None, i.e. unscorable (always wrong).
            gold = None
            for _gk in ("gold", "answer", "gold_answer", "target"):
                if _gk in item:
                    gold = item[_gk]
                    break
            eval_type = str(item.get("eval_type") or "contains").lower().strip()
            if eval_type not in EVAL_TYPES + _MC_ALIASES:
                # A misspelled strict type must not fall back to the lenient `contains`.
                raise ValueError(
                    f"custom_jsonl: {path.name} line {idx + 1} has eval_type {eval_type!r}; "
                    f"allowed: {', '.join(EVAL_TYPES + _MC_ALIASES)}")
            category = str(item.get("category") or item.get("domain") or "default")

            extra = {
                "eval_type": eval_type,
                "category": category,
            }
            # The scorer only receives (response, gold), so a per-row `eval_type` recorded
            # in the sample metadata never reached it: every row was graded with every
            # strategy at once, and `"eval_type": "exact"` still passed on containment.
            samples.append((messages, {_ET_KEY: eval_type, "gold": gold}, extra))
            if limit is not None and len(samples) >= limit:
                break

    logger.info(f"Loaded {len(samples)} custom evaluation samples from {path.name}")
    return samples


_ET_KEY = "__gbench_eval_type"

#: `eval_type` values a row may declare. `contains` stays the default so existing files
#: score exactly as before; the others narrow the check to what the row asked for.
EVAL_TYPES = ("contains", "exact", "numeric", "multiple_choice")
#: Accepted spellings of `multiple_choice`.
_MC_ALIASES = ("mc", "letter")


#: A number in a response: comma-grouped thousands (299,792,458 or 1,234.5) read as one number,
#: else a plain integer or decimal.
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.?\d+)")


def _numeric_match(resp: str, gold_str: str) -> bool:
    """Some number in `resp` equals the gold within 1e-4. A match with commas is read both ways:
    as one comma-grouped number (299,792,458) and as a list of numbers (85,90,100), and passes
    when either reading holds the gold."""
    try:
        gold_num = float(re.sub(r"[^\d.eE+-]", "", gold_str))
    except (ValueError, TypeError):
        return False
    for found in _NUMBER_RE.findall(resp):
        readings = [found.replace(",", "")]
        if "," in found:
            readings += found.split(",")
        for value in readings:
            try:
                if abs(float(value) - gold_num) < 1e-4:
                    return True
            except ValueError:
                continue
    return False


def _contains_match(resp: str, gold_str: str) -> bool:
    """The gold as a whole token of the response, case-insensitively: no letter, digit or
    underscore right before or after it (so `Au` is not found in `because`, nor `4` in `14`)."""
    return re.search(r"(?<!\w)" + re.escape(gold_str) + r"(?!\w)", resp,
                     re.IGNORECASE) is not None


def _mc_match(resp: str, gold_str: str) -> bool:
    if not (len(gold_str) == 1 and gold_str.isalpha()):
        return False
    from .extraction_common import last_mc_letter
    picked = last_mc_letter(resp, letters="ABCDEFGHIJ")
    return bool(picked) and picked.lower() == gold_str.lower()


def _eval_custom_jsonl(response_text: str, gold_answer: Any) -> bool:
    """Score one row with the strategy the row declares in `eval_type`.

    Every strategy used to be tried on every row and any hit passed, so a row marked
    `"eval_type": "exact"` still passed on a substring. `contains` matches the gold as a whole
    token (_contains_match), so a short gold such as `4` or `Au` is not matched inside `14` or
    `because`.
    """
    eval_type = "contains"
    if isinstance(gold_answer, dict) and _ET_KEY in gold_answer:
        eval_type = str(gold_answer.get(_ET_KEY) or "contains").lower().strip()
        gold_answer = gold_answer.get("gold")

    if not response_text or gold_answer is None:
        return False
    resp = response_text.strip()
    gold_str = str(gold_answer).strip()
    if not gold_str:
        return False

    if eval_type == "exact":
        return resp.lower() == gold_str.lower()
    if eval_type == "numeric":
        return _numeric_match(resp, gold_str)
    if eval_type in ("multiple_choice",) + _MC_ALIASES:
        return _mc_match(resp, gold_str)

    # Default `contains`: exact, then the gold as a whole token, then the type-appropriate check.
    if resp.lower() == gold_str.lower() or _contains_match(resp, gold_str):
        return True
    if _mc_match(resp, gold_str):
        return True
    return _numeric_match(resp, gold_str)


def run_custom_jsonl(
    jsonl_path: Union[str, Path],
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native evaluation on a user-provided JSONL benchmark file."""
    path = Path(jsonl_path)
    eval_name = path.stem.lower()
    samples = _load_custom_jsonl_samples(path, limit=kwargs.get("limit"))

    limit_plan = base_suite_plan(eval_name, samples, kwargs.get("limit"), run_attempts(kwargs))
    result = run_eval_suite(
        eval_name=eval_name,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_custom_jsonl,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    result["eval_limit_plan"] = limit_plan
    return result
