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

"""Native AMC and AIME high-school math competition evaluation suite.

A gbench composite, NOT a canonical benchmark: AMC 12 2022-2023 (`AI-MO/aimo-validation-amc`,
83 problems, multiple choice converted to numeric answers) plus AIME 2022-2024
(`AI-MO/aimo-validation-aime`, 90 problems), both pinned below, with a gbench prompt. Both sets
predate current training cutoffs and ship their AoPS worked solutions, so the score is a
contaminated reference. `leaderboard_comparable` is always False; `accuracy_amc` /
`accuracy_aime` report the two competitions separately. Either dataset failing to load is a
hard error: a score over one competition under the combined name would be a partial run.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_AMC_AIME_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from .base import apply_shard, finalize_shard_fields, run_eval_suite
from .extraction_common import final_number
from .metrics import record_budget_cutoffs
from .limit_plan import base_attempts, eval_limit_plan
from .sampling import stratified_sample, presharded_for_base

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/amc_aime.md"

#: (category, dataset, pinned revision). Snapshots reconciled on 2026-09-24.
_SOURCES = (
    ("AMC", "AI-MO/aimo-validation-amc", "69d78a4a2c840e82d69af6bc742bda09005f6316"),
    ("AIME", "AI-MO/aimo-validation-aime", "13f9e12f613e720c2a2b2f345dd04b998a29494d"),
)
#: Problems in the full set (83 AMC + 90 AIME).
_FULL_SET = 173


def _prompt(q_text: str, enable_thinking: bool) -> str:
    """gbench's prompt for both competitions (there is no upstream protocol for the blend)."""
    if enable_thinking:
        return (f"Problem: {q_text}\n\n"
                "Solve the math competition problem step by step. Write your final answer as an integer "
                "at the end in the format: 'The final answer is X'.")
    return (f"Problem: {q_text}\n\n"
            "Solve the math competition problem. Write your final answer as an integer "
            "at the end in the format: 'The final answer is X'.")


def _load_amc_aime_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load the pinned AMC and AIME sets; raise if either cannot be loaded."""
    from datasets import load_dataset

    samples = []
    failed: Dict[str, str] = {}
    for category, dataset, revision in _SOURCES:
        try:
            rows = list(load_dataset(dataset, split="train", revision=revision))
        except Exception as e:
            failed[category] = f"{dataset}@{revision[:8]}: {e}"
            continue
        for item in rows:
            raw_ans = str(item.get("answer", "")).strip()
            try:
                gold_ans = str(int(float(raw_ans)))
            except ValueError:
                gold_ans = raw_ans
            messages = [{"role": "user", "content": _prompt(item.get("problem", ""), enable_thinking)}]
            samples.append((messages, gold_ans, {"category": category}))

    # No partial run under the combined name: both competitions or a hard error.
    if len(failed) == len(_SOURCES):
        raise RuntimeError(
            "amc_aime: neither the AMC nor the AIME dataset could be loaded; refusing to report "
            f"a result over zero samples ({'; '.join(failed.values())}). See {DOCS_URL}")
    if failed:
        raise RuntimeError(
            f"amc_aime: the {' and '.join(failed)} dataset could not be loaded "
            f"({'; '.join(failed.values())}); refusing to report a score over the other "
            f"competition alone under the amc_aime name. See {DOCS_URL}")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard,
    # stratified by competition, not a contiguous head of each set.
    samples, _ = apply_shard(samples, "amc_aime")
    samples = stratified_sample(samples, limit, lambda s: s[2]["category"], seed="amc_aime")
    logger.info(f"Loaded {len(samples)} AMC/AIME math competition samples from HF Hub.")
    return samples


def _eval_amc_aime(response_text: str, gold_answer: str) -> bool:
    """Check the model's stated number against the gold AMC/AIME answer.

    Shared extraction: \\boxed{} and every answer anchor are honoured before
    the last-number fallback, so a boxed answer is no longer skipped in favour of whatever
    number happened to appear last in the working.
    """
    if not response_text:
        return False
    pred = final_number(response_text)
    if pred is None:
        return False
    try:
        return int(float(pred)) == int(float(gold_answer))
    except ValueError:
        return pred.strip().lower() == str(gold_answer).strip().lower()


def run_amc_aime(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native AMC & AIME competition math evaluation suite."""
    samples = _load_amc_aime_samples(enable_thinking, limit=kwargs.get("limit"))
    # --eval-limit N is N problems for the whole suite, spread over AMC and AIME (the loader's
    # stratified draw), never N per competition.
    per_competition = {category: 0 for category, _d, _r in _SOURCES}
    for _m, _g, meta in samples:
        category = str((meta or {}).get("category"))
        per_competition[category] = per_competition.get(category, 0) + 1
    limit_plan = eval_limit_plan("amc_aime", kwargs.get("limit"), len(samples), per_competition,
                                 setting_noun="competition", attempts=base_attempts())
    result = run_eval_suite(
        eval_name="amc_aime",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's problems, kept as is
        eval_fn=_eval_amc_aime,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    _components(result)
    result["eval_limit_plan"] = limit_plan
    result["dataset"] = " + ".join(f"{d}@{r}" for _, d, r in _SOURCES)

    # Both sources are PRE-CUTOFF and ship full AoPS worked solutions, so the headline reads
    # high from recall. Unlike the `aime` suite there is no post-cutoff set here to form a
    # clean headline, so the number cannot be de-contaminated - only labelled honestly.
    result["metric"] = ("blended AMC+AIME exact-match pass@1 over PRE-CUTOFF problems - a "
                        "gbench composite (NOT a canonical metric) AND a CONTAMINATED "
                        "reference; see accuracy_amc / accuracy_aime for the components")
    result["contamination_note"] = (
        "Two caveats. (1) NON-CANONICAL BLEND: AMC and AIME are distinct competitions "
        "normally reported separately; the headline `accuracy` averages them - use "
        "`accuracy_amc` / `accuracy_aime`. (2) CONTAMINATION: every problem is from "
        "AI-MO/aimo-validation-amc / -aime, whose problems AND worked solutions predate the "
        "training cutoff (artofproblemsolving.com), so a high score is indistinguishable "
        "from recall. For a clean AIME number use the `aime` suite (MathArena AIME 2026).")
    _comparability(result, kwargs.get("limit"))
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)


def _components(result: Dict[str, Any]) -> None:
    """AMC and AIME are DIFFERENT competitions (different difficulty and, canonically, reported
    separately) - the single blended headline is a gbench construction, not a canonical metric.
    Each component comes from category_accuracy so the blend is never the only number visible."""
    cat = result.get("category_accuracy", {}) or {}
    for comp in ("AMC", "AIME"):
        result.pop(f"accuracy_{comp.lower()}", None)
        result.pop(f"{comp.lower()}_n", None)
        if comp in cat and cat[comp].get("total"):
            result[f"accuracy_{comp.lower()}"] = cat[comp]["accuracy"]
            result[f"{comp.lower()}_n"] = cat[comp]["total"]


def _comparability(result: Dict[str, Any], limit: Optional[int]) -> None:
    """leaderboard_comparable is always False (a blend with a gbench prompt over a contaminated
    set); the reason also names an --eval-limit subset, replies the run's token budget cut off
    and requests that got no reply."""
    # A blended AMC+AIME headline is a gbench composite with no published leaderboard; the
    # reference set is also pre-cutoff/contaminated. Not directly comparable to either
    # competition's numbers.
    reasons = ["non-canonical AMC+AIME blend (no combined leaderboard) with a gbench prompt, over "
               "a pre-cutoff/contaminated reference set; see accuracy_amc / accuracy_aime for the "
               "components"]
    if limit:
        reasons.append(f"subset run (--eval-limit {limit}: {result.get('total_questions')} of "
                       f"{_FULL_SET} problems)")
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    no_reply = sum(1 for t in result.get("sample_traces") or []
                   if t.get("harness_error") or t.get("response_text", "") is None)
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint (harness "
                       "errors, left out of the score)")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run (`gbench --merge`): the generic merge (the union of
    the shards' traces re-counted; the headline is correct over scored, as each shard's is) with
    the AMC and AIME components and the comparability reasons recomputed over the union."""
    from ...merge import generic_merge
    merged = generic_merge("amc_aime", results)
    _components(merged)
    limit = merged.get("eval_limit") or (merged.get("run_shape") or {}).get("eval_limit")
    _comparability(merged, limit)
    unreconciled = merged.get("merge_unreconciled_fields")
    if isinstance(unreconciled, dict):
        for field in ("accuracy_amc", "accuracy_aime", "amc_n", "aime_n",
                      "replies_cut_off_by_budget", "replies_cut_off_in_reasoning"):
            unreconciled.pop(field, None)
        if not unreconciled:
            merged.pop("merge_unreconciled_fields")
    return merged
