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

# Evaluation Suite: bfcl_live
# Description: BFCL Live single-turn function calling + irrelevance/relevance detection

"""gbench native built-in runner for bfcl_live (Tool Use & Function Calling).

Renamed from `bfcl_v3_live`: the data is the Live set of the current BFCL v4 files, and "live" is
BFCL's own name for this category group (bfcl_eval TEST_COLLECTION_MAPPING["live"]); the old
name still runs this suite, with a deprecation warning.

The BFCL **Live** categories (user-contributed, first released with BFCL v2/v3): live_simple,
live_multiple, live_parallel, live_parallel_multiple, live_irrelevance and live_relevance, as
shipped in the current BFCL_v4 data files at the pinned bfcl_eval commit (see bfcl.py). It was
previously mis-named `bfcl_v4_agentic`, which implied the v4 agentic track (web search /
memory); that track is now a separate suite, `bfcl_v4_agentic`.

Protocol, scoring and aggregation are the bfcl suite's: BFCL's FC request, and bfcl_eval's own
eval_runner (decode_ast, ast_checker and the relevance rule) scoring the replies in the local
harness image gbench-bfcl:2026.3.23 (docker/bfcl.Dockerfile; no Docker or no image is a hard
error before anything is generated). Reply text is never a call, as in BFCL's FC handler. The
headline is BFCL's "Live Overall Acc" column: the entry-weighted accuracy over the four live AST
categories; irrelevance and relevance detection are reported next to it, as on the leaderboard.

The checker is bounded as the bfcl suite's (GBENCH_BFCL_TIMEOUT_S, unset by default): a checker it
stops leaves the replies unscored (HARNESS_ERROR, out of every accuracy, `harness_errors`).

Sampling: gbench's rule for every suite (base.resolve_temperature): temperature 1.0 with
`--thinking`, 0.0 without (BFCL's own 0.001 is recorded under `official_sampling`:
fc_common.BFCL_OFFICIAL_SAMPLING). No max_tokens, as BFCL's handler sends none
(--max-output-tokens is not applied; `max_output_tokens_note`). Override the temperature for a
whole run with `--temperature`, or for this suite alone with `GBENCH_BFCL_LIVE_TEMPERATURE` (the
pre-rename `GBENCH_BFCL_V3_LIVE_TEMPERATURE` is still read as a deprecated alias), which takes
precedence over both. No LLM judge.
"""

import logging
from typing import Any, Dict, List, Tuple

from .base import run_eval_suite
from . import bfcl as _bfcl
from .bfcl import (DATASET_REVISION, LIVE_CATEGORIES, _eval_bfcl, bfcl_batch_scorer, bfcl_gold,
                   bfcl_request, bfcl_summary, load_bfcl_entries, run_notes)
from .bfcl_v4_agentic import BFCL_DOCKERFILE, bfcl_image
from .fc_common import BFCL_OFFICIAL_SAMPLING, base_suite_plan, run_attempts

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
#: The registry name (renamed from `bfcl_v3_live`).
EVAL_NAME = "bfcl_live"


def _load_bfcl_live_samples() -> Tuple[List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
                                          Dict[str, List[Dict[str, Any]]]]:
    """Samples for the six BFCL Live categories plus the original function docs by entry id.

    `--eval-limit` is applied by run_eval_suite, stratified over the categories.
    """
    samples = []
    functions_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for entry in load_bfcl_entries(LIVE_CATEGORIES, EVAL_NAME):
        messages, tools = bfcl_request(entry)
        # The tools go on the REQUEST: with them only in the prompt, a tool-call parser lifts
        # the model's call out of `content` and discards it (measured: 12/20 empty replies).
        meta: Dict[str, Any] = {"category": entry["category"]}
        if tools:
            meta["tools"] = tools
        samples.append((messages, bfcl_gold(entry), meta))
        functions_by_id[entry["id"]] = entry["function"]
    logger.info("Loaded %d BFCL Live samples.", len(samples))
    return samples, functions_by_id


def _eval_bfcl_live(response_text: str, gold_ground_truth: Any,
                       tool_calls: Any = None) -> bool:
    """The ported rule's verdict for one reply (see bfcl._eval_bfcl; the suite scores with
    bfcl_eval itself, bfcl.bfcl_batch_scorer).

    Parameter names match exactly; reply text is never a call (BFCL's FC rule); for irrelevance
    the reply is correct iff it makes no call (an empty reply included).
    """
    return _eval_bfcl(response_text, gold_ground_truth, tool_calls=tool_calls)


def run_bfcl_live(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute the bfcl_live native built-in evaluation suite."""
    # The checker runs in the harness image: checked before anything is generated.
    harness_version, _ = _bfcl._harness_probe(EVAL_NAME, [], True)
    samples, functions_by_id = _load_bfcl_live_samples()
    scorer = bfcl_batch_scorer(functions_by_id, EVAL_NAME)
    limit_plan = base_suite_plan(EVAL_NAME, samples, kwargs.get("limit"), run_attempts(kwargs))
    result = run_eval_suite(
        eval_name=EVAL_NAME,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=scorer,
        declared_scoring_mode="deterministic",
        supports_attempts=True,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        # the pre-rename name keeps --eval-limit drawing the same entries
        limit_seed="bfcl_v3_live",
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=BFCL_OFFICIAL_SAMPLING,
    )
    result["eval_limit_plan"] = limit_plan
    live = bfcl_summary(result.get("category_accuracy") or {})["live"]
    result["bfcl_summary"] = {"live": live}
    result["dataset_revision"] = DATASET_REVISION
    result["single_turn_checker"] = (f"bfcl-eval {harness_version} eval_runner in the local "
                                     f"image {bfcl_image()} ({BFCL_DOCKERFILE})")
    result["pooled_accuracy"] = result.get("accuracy")
    if live["Live Overall Acc"] is not None:
        result["accuracy"] = live["Live Overall Acc"]
        result["metric"] = ("BFCL Live Overall Acc: entry-weighted accuracy over live_simple, "
                            "live_multiple, live_parallel and live_parallel_multiple; irrelevance "
                            "and relevance detection in bfcl_summary")
    else:
        result["metric"] = ("pooled accuracy over the categories run (Live Overall Acc needs all "
                            "four live AST categories); per-column scores in bfcl_summary")
    # A checker stopped by GBENCH_BFCL_TIMEOUT_S leaves the replies unscored (HARNESS_ERROR).
    result["harness_errors"] = list(scorer.harness_errors)
    reasons = ["BFCL Live slice only: the headline is the leaderboard's Live Overall Acc column, a "
               "category subscore, not BFCL Overall", *run_notes(kwargs.get("limit"))]
    if result["harness_errors"]:
        unscored = _bfcl.harness_unscored(result.get("sample_traces") or [])
        total = int(result.get("total_questions") or 0)
        result["status"] = ("failed" if unscored + int(result.get("failed_requests") or 0) >= total
                            else "completed_with_errors")
        reasons += [f"harness error: {e['error']}" for e in result["harness_errors"]]
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (BFCL generates at 0.001)")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
