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

# Evaluation Suite: frames
# Description: Google FRAMES (Factuality, Retrieval, And reasoning MEasurement Set - 824 questions)

"""gbench native built-in runner for frames (Factuality & RAG).

Protocol: the paper's "Naive Prompt" single-step setting (arXiv 2409.12941, Section 4.1): the
question alone is the user turn, with no retrieval and no tools. Each reply is graded by the
paper's auto-rating prompt (appendix figure, transcribed verbatim with its system turn) and
its `Decision: TRUE|FALSE` line; the headline is accuracy = TRUE / graded.

The autorater model is gbench's Gemini cascade (the paper used Gemini-Pro-1.5-0514), so the
result is flagged leaderboard_comparable=False and names the models that graded it; each trace
keeps the autorater's reply (`judge_reply`) and the model that gave it (`judge_model`). An
autorater reply without a Decision line is an autorater failure, not the model's answer: it is
marked JUDGE_FAILED, left out of the accuracy, counted in `judge_failures` (the run is
`completed_with_errors`) and named in `leaderboard_comparable_reason` (the paper publishes no
parser that would decide it otherwise).

Sampling: gbench's rule (base.resolve_temperature): temperature 1.0 with `--thinking`, 0.0
without. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_FRAMES_TEMPERATURE`,
which takes precedence over both. The paper publishes no autorater sampling settings, so the
autorater runs at gbench's deterministic judge temperature 0.0 with the paper's system turn
(recorded in `judge_sampling`), which none of this affects.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_required_skip,
                   judge_cascade, judge_config, judge_generate_cascade, JUDGE_TEMPERATURE)
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base
from .fc_common import base_suite_plan
from .limit_plan import base_attempts
from .simpleqa import withhold_if_nothing_scored

logger = logging.getLogger(__name__)

PILLAR = "Factuality & RAG"

_DATASET = "google/frames-benchmark"
_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
_FULL_SIZE = 824

#: System turn of the paper's autorater.
AUTORATER_SYSTEM = "You are a helpful assistant."

# The paper's auto-rating prompt (arXiv 2409.12941 appendix; Figure 6 in v3, Figure 7 in v1),
# transcribed from the LaTeX source (auto_rate_prompt.tex): one line per `\\` break, blank
# lines where the rendered figure has them. The source's typographic apostrophe (U+2019) and
# en dash (U+2013) are kept as escapes, so the text sent to the autorater is the source's.
AUTORATER_TEMPLATE = (
    "===Task===\n"
    "\n"
    "I need your help in evaluating an answer provided by an LLM against a ground truth "
    "answer. Your task is to determine if the ground truth answer is present in the LLM\u2019s "
    "response. Please analyze the provided data and make a decision.\n"
    "===Instructions===\n"
    "1. Carefully compare the \"Predicted Answer\" with the \"Ground Truth Answer\".\n"
    "2. Consider the substance of the answers \u2013 look for equivalent information or correct "
    "answers. Do not focus on exact wording unless the exact wording is crucial to the "
    "meaning.\n"
    "3. Your final decision should be based on whether the meaning and the vital facts of the "
    "\"Ground Truth Answer\" are present in the \"Predicted Answer:\"\n"
    "\n"
    "===Input Data===\n"
    "- Question: {question}\n"
    "- Predicted Answer: {llm_response}\n"
    "- Ground Truth Answer: {ground_truth_answer}\n"
    "\n"
    "===Output Format===\n"
    "Provide your final evaluation in the following format:\n"
    "\"Explanation:\" (How you made the decision?)\n"
    "\"Decision:\" (\"TRUE\" or \"FALSE\" )\n"
    "\n"
    "Please proceed with the evaluation."
)

# The last `Decision` label wins; quotes, markdown bold and the colon around it are tolerated
# because the format line itself shows the label in quotes.
_DECISION_RE = re.compile(r"Decision\W*(TRUE|FALSE)\b", re.IGNORECASE)


def parse_decision(text: Optional[str]) -> Optional[bool]:
    """The autorater's verdict: True for TRUE, False for FALSE, None when there is none."""
    hits = _DECISION_RE.findall(text or "")
    if not hits:
        return None
    return hits[-1].upper() == "TRUE"


def _load_frames_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load FRAMES from HF Hub (google/frames-benchmark, pinned); raises on load/schema failure (no fabricated fallback)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET, split='test', revision=_REVISION)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for frames: {e}")
        raise RuntimeError(f"Could not load dataset for frames: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for frames returned empty rows")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "frames")
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("reasoning_types"), seed="frames")

    samples = []
    for item in rows:
        # google/frames-benchmark columns are capitalised: 'Prompt' / 'Answer'.
        prompt = item.get("Prompt")
        gold = item.get("Answer")
        if not str(prompt or "").strip() or gold is None:
            raise RuntimeError(
                "frames: unexpected dataset schema (missing 'Prompt'/'Answer'); "
                "refusing to fabricate sample data"
            )
        cat = str(item.get("reasoning_types") or "multi_hop")
        # The question as the dataset holds it (67 of the 824 prompts end in whitespace): sent,
        # and shown to the autorater, verbatim; so is the gold.
        messages = [{"role": "user", "content": str(prompt)}]
        samples.append((messages, str(gold), {"category": cat}))

    logger.info(f"Loaded {len(samples)} frames samples.")
    return samples


async def _async_judge_frames(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 64,
) -> None:
    """FRAMES paper autorater (verbatim prompt + Decision line) over the Gemini cascade."""
    from tqdm import tqdm
    import asyncio

    semaphore = asyncio.Semaphore(judge_concurrency)
    cfg = judge_config(system_instruction=AUTORATER_SYSTEM)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        if resp_text is None or trace.get("harness_error"):
            # The request got no reply from the served endpoint: a harness error (counted in
            # failed_requests; base leaves it out of the score). The judge is not called.
            trace["is_correct"] = False
            trace["judge_grade"] = "not_judged_failed_request"
            trace["scoring_excluded"] = True
            pbar.update(1)
            return
        if not resp_text:
            # An empty reply the server returned (the model's outcome, counted in
            # empty_responses): nothing to grade, the judge is not called, and it must not also
            # count as a judge failure.
            trace["is_correct"] = False
            trace["judge_grade"] = "not_judged_empty_response"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        question = messages[0].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")
        prompt = AUTORATER_TEMPLATE.format(
            question=question, llm_response=resp_text, ground_truth_answer=gold)

        async with semaphore:
            text, judge_used = await judge_generate_cascade(prompt, config=cfg)
        if text is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy,
            # NOT scored INCORRECT. base.run_eval_suite drops it from pass/fail.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return

        decision = parse_decision(text)
        trace["judge_reply"] = text.strip()
        trace["judge_model"] = judge_used
        trace["is_correct"] = bool(decision)
        if decision is None:
            # The autorater answered without a Decision line: an autorater failure, not the
            # model's answer. Counted in judge_failures and left out of the accuracy
            # (scoring_excluded) instead of guessing a verdict or scoring it wrong.
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["scoring_excluded"] = True
            trace["status"] = "FAILED"
        else:
            trace["judge_grade"] = "TRUE" if decision else "FALSE"
            trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [FRAMES]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_frames(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute frames native built-in evaluation suite."""
    skip = gemini_required_skip("frames", model_name)
    if skip is not None:
        return skip
    limit = kwargs.get("limit")
    samples = _load_frames_samples(limit=limit)
    # One generation per sample: base runs a batch-scored suite (async_eval_fn, no
    # supports_attempts) at 1 attempt whatever --attempt-count asks.
    limit_plan = base_suite_plan("frames", presharded_for_base(samples), limit,
                                 base_attempts(batch_scored=True))
    result = run_eval_suite(
        eval_name="frames",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        async_eval_fn=_async_judge_frames,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    result["eval_limit_plan"] = limit_plan
    traces = result.get("sample_traces", []) or []
    result["metric"] = ("FRAMES accuracy = autorater Decision TRUE / graded questions "
                        "(paper's Naive Prompt setting: closed book, single step)")
    result["judge"] = "gbench-gemini-cascade"
    # One autorater call per graded question, so this is also the answered calls per model.
    result["judge_models"] = dict(Counter(t["judge_model"] for t in traces
                                          if t.get("judge_model")))
    result["judge_cascade"] = judge_cascade()
    # The paper gives the autorater prompt and model only; no sampling settings are published.
    result["judge_sampling"] = {
        "reference": ("arXiv 2409.12941 appendix auto-rating prompt with its system turn, rated "
                      "by Gemini-Pro-1.5-0514; the paper publishes no autorater sampling settings"),
        "system_instruction": AUTORATER_SYSTEM, "temperature": JUDGE_TEMPERATURE,
        "temperature_source": "gbench's deterministic judge default (base.judge_config)"}
    reasons = ["autorater is gbench's standard Gemini cascade instead of the paper's "
               "Gemini-Pro-1.5-0514 (a gbench convention)"]
    if limit:
        reasons.append(f"--eval-limit subset ({len(samples)} of {_FULL_SIZE} questions)")
    failed = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_FAILED")
    if failed:
        reasons.append(f"{failed} autorater repl(ies) without a Decision line left out of the "
                       "accuracy as autorater failures")
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    withhold_if_nothing_scored(result)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
