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

# Evaluation Suite: humanitys_last_exam
# Description: Humanity's Last Exam (HLE - 2,500 frontier expert-level STEM & scientific reasoning questions)

"""gbench native built-in runner for humanitys_last_exam (STEM & Scientific Reasoning).

Protocol, as in centerforaisafety/hle `hle_eval/`: run_model_predictions.py sends the official
SYSTEM_PROMPT as a system turn and the question (plus its image, when it has one) as the user
turn; run_judge_results.py grades every reply with the official JUDGE_PROMPT, as structured
output (extracted_final_answer / reasoning / correct / confidence), and reports accuracy plus
the RMS calibration error from its `calib_err(beta=100)`, both ported here verbatim.

The judge model is gbench's Gemini cascade (the official script uses o3-mini-2025-01-31), so
the result is flagged leaderboard_comparable=False and names the models that graded it.

Sampling: the temperature follows gbench's rule (base.resolve_temperature): 1.0 with
`--thinking`, 0.0 without. HLE's own setting per mode is recorded (base.OfficialSampling, see
official_sampling()): without `--thinking`, temperature 0, as the HLE README states
("temperature is defaulted to 0 for our evaluation") and as run_model_predictions.py sent
`--temperature 0.0` to non-reasoning models until commit 136e12e; with `--thinking`, no
temperature at all (the script sent none to reasoning models ("o1") before 136e12e, and since
then its request line is commented out, so it sends none to any model), which the result records
as not matching the 1.0 gbench sends. In both modes no max_tokens of its own: the README
says
"Our evaluation uses the maximum number of tokens available" and run_model_predictions.py's
--max_completion_tokens defaults to None, so each request asks for the rest of the context window
and `--max-output-tokens` is not applied (`max_output_tokens_note`). A reply that runs to the end
of the server's context window is therefore the protocol's outcome ("we observed some model
collapse", README): it is judged as it arrived and counted in `replies_at_context_window`, not as
a reply cut off by the run's budget. Override the temperature for a whole run with
`--temperature`, or for this suite alone with `GBENCH_HUMANITYS_LAST_EXAM_TEMPERATURE`, which
takes precedence over both (and over the rule's 1.0 / 0.0); the result records `official_sampling`.

The judge call sends what run_judge_results.py's extract_answer sends: the JUDGE_PROMPT as the
only (user) message, structured output (its ExtractedAnswer model as the response schema),
max_completion_tokens=4096 (as max_output_tokens: o3-mini counts its reasoning against that
limit, as Gemini counts its thinking) and no temperature. The result records it under
`judge_sampling`; none of the settings above changes it. The question and the gold answer enter
the JUDGE_PROMPT exactly as the dataset stores them (run_judge_results.py passes question["answer"]
unchanged). A judge reply without a yes/no verdict, for example one stopped before its JSON object
closed (the 4096 budget ran out), is a judge failure: marked JUDGE_FAILED, left out of the accuracy
and the calibration error, counted in `judge_failures` (the run is `completed_with_errors`) and
named in `leaderboard_comparable_reason`; the incomplete ones are also counted in
`judge_replies_incomplete_json`. Upstream prints "Missing judge response ..., you should rerun the
judge" for such a question and then divides by every question, so an unjudged question counts as
wrong there; gbench keeps a judge-side failure out of the model's score.
"""

import json
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .limit_plan import base_attempts, eval_limit_plan
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_required_skip,
                   judge_cascade, judge_config, judge_generate_cascade, OfficialSampling)
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base
from .simpleqa import withhold_if_nothing_scored

logger = logging.getLogger(__name__)

PILLAR = "STEM & Scientific Reasoning"

#: cais/hle, the original 2,500-question HLE. Its last data change is 021a3d7 (2025-05-20);
#: this commit only added eval.yaml. HLE-Rolling and HLE-Diamond are separate datasets.
_DATASET = "cais/hle"
_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
_FULL_SIZE = 2500

#: Both modes: README @22ed3074b1e7 "Our evaluation uses the maximum number of tokens available";
#: run_model_predictions.py `--max_completion_tokens` default None.
_MAX_TOKENS_REFERENCE = ("; no max_tokens: 'Our evaluation uses the maximum number of tokens "
                         "available' (README), --max_completion_tokens default None")


def official_sampling(thinking: bool) -> OfficialSampling:
    """HLE's sampling for the run's mode (see the module docstring)."""
    if thinking:
        return OfficialSampling(
            reference=("centerforaisafety/hle hle_eval/run_model_predictions.py @22ed3074b1e7: "
                       "no temperature is sent (the line is commented out since 136e12e; before "
                       "it, reasoning ('o1') models were sent none)" + _MAX_TOKENS_REFERENCE),
            send_temperature=False, send_max_tokens=False)
    return OfficialSampling(
        reference=("centerforaisafety/hle README @22ed3074b1e7 ('temperature is defaulted to 0 "
                   "for our evaluation'; run_model_predictions.py --temperature 0.0, sent to "
                   "non-reasoning models before 136e12e)" + _MAX_TOKENS_REFERENCE),
        temperature=0.0, send_max_tokens=False)

# Copied verbatim from centerforaisafety/hle (commit 22ed3074b1e7) hle_eval/run_model_predictions.py.
SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

# Copied verbatim from the same commit's hle_eval/run_judge_results.py (upstream writes `\%` in
# a plain string, which Python keeps as backslash-percent; `\\%` here is the same text).
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""

#: run_judge_results.py extract_answer @22ed3074b1e7: client.beta.chat.completions.parse(
#: model=args.judge, max_completion_tokens=4096, messages=[user prompt],
#: response_format=ExtractedAnswer), with no temperature.
JUDGE_MAX_COMPLETION_TOKENS = 4096
JUDGE_REFERENCE = ("centerforaisafety/hle hle_eval/run_judge_results.py @22ed3074b1e7 "
                   "extract_answer: o3-mini-2025-01-31, the JUDGE_PROMPT as the only (user) "
                   "message, response_format=ExtractedAnswer, max_completion_tokens=4096, no "
                   "temperature")

#: calib_err's target bin size in the official script (`calib_err(..., p='2', beta=100)`).
_CALIB_BETA = 100
_JUDGE_FIELDS = ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"]


def _judge_schema() -> Any:
    """Gemini response schema mirroring upstream's `ExtractedAnswer` structured output.

    Field order is pinned (reasoning before the verdict, as upstream's model declares it);
    `strict` is part of upstream's schema and is ignored when parsing.
    """
    try:
        from google.genai import types
    except ImportError:
        return None
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "extracted_final_answer": types.Schema(type=types.Type.STRING),
            "reasoning": types.Schema(type=types.Type.STRING),
            "correct": types.Schema(type=types.Type.STRING, enum=["yes", "no"]),
            "confidence": types.Schema(type=types.Type.INTEGER),
            "strict": types.Schema(type=types.Type.BOOLEAN),
        },
        required=list(_JUDGE_FIELDS),
        property_ordering=list(_JUDGE_FIELDS),
    )


def hle_judge_config() -> Any:
    """The judge config of run_judge_results.py's call: the ExtractedAnswer schema as structured
    output, max_output_tokens 4096 (its max_completion_tokens) and no temperature."""
    schema = _judge_schema()
    if schema is None:
        return None
    cfg = judge_config(response_mime_type="application/json", response_schema=schema,
                       max_output_tokens=JUDGE_MAX_COMPLETION_TOKENS)
    return cfg.model_copy(update={"temperature": None}) if cfg is not None else None


def judge_sampling_record() -> Dict[str, Any]:
    """The `judge_sampling` block of an HLE result."""
    return {"reference": JUDGE_REFERENCE, "temperature": "not sent (upstream sends none)",
            "max_output_tokens": JUDGE_MAX_COMPLETION_TOKENS,
            "max_output_tokens_note": (
                "Gemini counts its thinking against max_output_tokens, as o3-mini counts its "
                "reasoning against max_completion_tokens; a structured reply stopped before its "
                "JSON object closed is a judge failure, counted in judge_replies_incomplete_json"),
            "response_schema": "ExtractedAnswer (extracted_final_answer, reasoning, correct, "
                               "confidence, strict)"}


def judge_reply_is_complete_json(text: Optional[str]) -> bool:
    """Whether a judge reply is one complete JSON object (a ```json fence around it allowed).

    Under the ExtractedAnswer response schema a finished reply always is one, so a reply that is
    not was stopped before the object closed; the budget that stops it is max_output_tokens
    4096, which Gemini's thinking also counts against. Upstream's parse raises on such a reply
    and extract_answer returns None (the question is left unjudged and counted wrong)."""
    raw = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    try:
        return isinstance(json.loads(raw), dict)
    except (ValueError, TypeError):
        return False


def _load_humanitys_last_exam_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load humanitys_last_exam benchmark dataset directly from HF Hub (cais/hle, pinned)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET, split='test', revision=_REVISION)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for humanitys_last_exam: {e}")
        raise RuntimeError(f"Could not load dataset for humanitys_last_exam: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for humanitys_last_exam returned empty rows")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "humanitys_last_exam")
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="humanitys_last_exam")

    samples = []
    for item in rows:
        prompt = item.get("question")
        gold = item.get("answer")
        if not prompt or gold is None:
            raise RuntimeError(
                "humanitys_last_exam: unexpected dataset schema (missing 'question'/'answer'); "
                "refusing to fabricate sample data"
            )
        # As the dataset stores it: run_judge_results.py passes question["answer"] unchanged
        # (27 of the 2,500 answers carry surrounding whitespace).
        gold = str(gold)
        cat = item.get("category", "Mathematics")

        # As upstream format_message(): the text part first, then the image data-URI for the
        # multimodal subset ("" when the question is text-only).
        image = item.get("image")
        if image:
            content: Any = [{"type": "text", "text": str(prompt)},
                            {"type": "image_url", "image_url": {"url": image}}]
        else:
            content = str(prompt)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} humanitys_last_exam samples.")
    return samples


def _question_text(messages: Any) -> str:
    """The question as upstream's judge sees it: the user turn's text, without the image."""
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content
                           if isinstance(p, dict) and p.get("type") == "text")
        return str(content or "")
    return ""


def parse_judge_reply(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The judge's structured verdict, or None when it carries no yes/no `correct`.

    The reply is the schema's JSON object. A reply in the prompt's own `field: value` layout is
    read too, since that is the same verdict. A missing confidence is 100, the prompt's rule.
    """
    raw = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    data: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            data = parsed
    except (ValueError, TypeError):
        for key in ("extracted_final_answer", "correct", "confidence"):
            m = re.search(rf"^\W*{key}\W*:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
            if m:
                data[key] = m.group(1).strip()
    verdict = re.match(r"\W*(yes|no)\b", str(data.get("correct") or ""), re.IGNORECASE)
    if not verdict:
        return None
    correct = verdict.group(1).lower()
    confidence = 100.0
    m = re.search(r"-?\d+(?:\.\d+)?", str(data.get("confidence", "")))
    if m:
        confidence = float(m.group(0))
    return {"model_answer": str(data.get("extracted_final_answer", "")),
            "reasoning": str(data.get("reasoning", "")),
            "correct": correct,
            "confidence": confidence}


# Ported verbatim from centerforaisafety/hle hle_eval/run_judge_results.py (upstream source:
# hendrycks/outlier-exposure utils/calibration_tools.py). The loop runs to len(bins) - 1, so
# the last (highest-confidence) bin never contributes; kept so the number matches the official
# script. Needs len(confidence) >= beta (fewer leaves `bins` empty and upstream raises).
def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def hle_calibration(sample_traces: List[Dict[str, Any]]) -> Tuple[Optional[float], int]:
    """Official HLE calibration error (percent) over the judged traces, and how many fed it.

    As dump_metrics(): correct = the judge's yes/no, confidence = the judge-extracted confidence
    / 100, then `100 * round(calib_err(confidence, correct, p='2', beta=100), 2)`. Returns None
    below 2 * beta judged samples, where upstream's estimator has no bin left to measure.
    """
    correct, confidence = [], []
    for t in sample_traces:
        if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE" or "hle_confidence" not in t:
            continue
        correct.append(bool(t.get("is_correct")))
        confidence.append(t["hle_confidence"])
    n = len(correct)
    if n < 2 * _CALIB_BETA:
        return None, n
    cerr = calib_err(np.array(confidence) / 100, np.array(correct), p='2', beta=_CALIB_BETA)
    return round(100 * round(float(cerr), 2), 2), n


async def _async_judge_humanitys_last_exam(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 64,
) -> None:
    """Official HLE judge (JUDGE_PROMPT, structured output) over the Gemini cascade."""
    from tqdm import tqdm
    import asyncio

    semaphore = asyncio.Semaphore(judge_concurrency)
    cfg = hle_judge_config()

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

        question = _question_text(trace.get("messages", []))
        gold = trace.get("gold_answer")
        gold = "" if gold is None else str(gold)
        prompt = JUDGE_PROMPT.format(question=question, correct_answer=gold, response=resp_text)

        async with semaphore:
            text, judge_used = await judge_generate_cascade(prompt, config=cfg)
        if text is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy,
            # NOT scored INCORRECT. base.run_eval_suite drops it from pass/fail.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return

        trace["judge_model"] = judge_used
        trace["judge_reply"] = text.strip()
        verdict = parse_judge_reply(text)
        if verdict is None:
            # A reply without a yes/no verdict is a judge failure, not the model's answer:
            # counted in judge_failures and left out of the accuracy (scoring_excluded) and the
            # calibration error.
            trace["is_correct"] = False
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["scoring_excluded"] = True
            trace["judge_reply_complete_json"] = judge_reply_is_complete_json(text)
            trace["status"] = "FAILED"
            pbar.update(1)
            return
        # upstream: `"yes" in judge_response["correct"]`
        trace["is_correct"] = "yes" in verdict["correct"]
        trace["judge_grade"] = verdict["correct"]
        trace["judge_response"] = verdict
        trace["hle_confidence"] = verdict["confidence"]
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [HLE]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_humanitys_last_exam(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute humanitys_last_exam native built-in evaluation suite."""
    skip = gemini_required_skip("humanitys_last_exam", model_name)
    if skip is not None:
        return skip
    limit = kwargs.get("limit")
    samples = _load_humanitys_last_exam_samples(limit=limit)
    eval_plan = eval_limit_plan(
        "humanitys_last_exam", limit, len(samples), attempts=base_attempts(batch_scored=True))
    result = run_eval_suite(
        eval_name="humanitys_last_exam",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        async_eval_fn=_async_judge_humanitys_last_exam,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=official_sampling(enable_thinking),
    )
    result["eval_limit_plan"] = eval_plan
    traces = result.get("sample_traces", []) or []
    # Accuracy stays the headline, with upstream's Wald 95% half-width beside it.
    n_scored = sum(1 for t in traces if str(t.get("judge_grade") or "") != "JUDGE_OUTAGE"
                   and not t.get("scoring_excluded"))
    acc = result.get("accuracy")
    if n_scored and isinstance(acc, (int, float)):
        result["accuracy_ci95_half_width"] = round(1.96 * math.sqrt(acc * (100 - acc) / n_scored), 2)
    cerr, n_conf = hle_calibration(traces)
    result["calibration_error"] = cerr
    result["calibration_samples"] = n_conf
    if cerr is None:
        result["calibration_note"] = (
            f"not reported: {n_conf} judged sample(s); the official calib_err (beta="
            f"{_CALIB_BETA}) needs at least {2 * _CALIB_BETA}, since it skips its last bin")
    result["metric"] = ("HLE accuracy = judge 'correct: yes' / graded questions (headline) + "
                        "calibration_error (official RMS calib_err, beta=100, judge-extracted "
                        "confidence)")
    result["judge"] = "gbench-gemini-cascade"
    # One judge call per graded question, so this is also the answered calls per model.
    result["judge_models"] = dict(Counter(t["judge_model"] for t in traces
                                          if t.get("judge_model")))
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = judge_sampling_record()
    # Judge replies stopped before their JSON object closed (see judge_reply_is_complete_json).
    incomplete = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_FAILED"
                     and t.get("judge_reply_complete_json") is False)
    result["judge_replies_incomplete_json"] = incomplete
    reasons = ["graded by gbench's standard Gemini cascade instead of the official "
               "o3-mini-2025-01-31 judge (a gbench convention)"]
    if limit:
        reasons.append(f"--eval-limit subset ({len(samples)} of {_FULL_SIZE} questions)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        if enable_thinking:
            reasons.append(f"temperature {result.get('temperature')} (HLE sends reasoning "
                           "models no temperature)")
        else:
            reasons.append(f"temperature {result.get('temperature')} (HLE evaluates at 0)")
    failed = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_FAILED")
    if failed:
        reasons.append(f"{failed} judge repl(ies) without a yes/no verdict left out of the score "
                       f"as judge failures ({incomplete} of them stopped before the "
                       f"ExtractedAnswer JSON closed: max_output_tokens "
                       f"{JUDGE_MAX_COMPLETION_TOKENS}, which the judge's thinking counts "
                       "against); upstream counts an unjudged question as wrong")
    # A reply that reached the end of the context window with no max_tokens sent is HLE's
    # protocol outcome (no cap, 'the maximum number of tokens available'), counted by base in
    # replies_at_context_window; only a reply cut off by a max_tokens that was sent can be a
    # budget cut-off, and HLE sends none.
    cut = record_budget_cutoffs(result, parts=[{
        "sample_traces": [t for t in traces if t.get("max_tokens_sent") is not None],
        "official_max_tokens_sent": result.get("official_max_tokens_sent")}])
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    withhold_if_nothing_scored(result)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
