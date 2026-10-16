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

# Evaluation Suite: healthbench
# Description: HealthBench (OpenAI) - clinical conversations graded against physician rubrics

"""gbench native built-in runner for healthbench (Medical & Health).

HealthBench (openai/healthbench, the 5,000-conversation oss_eval set, pinned to the HF revision
whose file is byte-identical to simple-evals' INPUT_PATH). Grading is simple-evals'
healthbench_eval.py, ported:
  * every rubric item is graded in its own call with GRADER_TEMPLATE (verbatim), the
    conversation plus the response rendered as "role: content" blocks and the item rendered
    as "[points] criterion", so the grader sees the point value and applies the template's
    rule for negative-point (undesirable) criteria;
  * a reply without a boolean `criteria_met` is asked again (upstream retries until it gets
    one; gbench stops after a few tries and marks the conversation JUDGE_FAILED);
  * the per-conversation score is achieved points / total positive points, NOT clipped;
  * the headline is the mean over conversations, clipped to [0, 1] once (upstream
    _compute_clipped_stats), with the same clipped means per example tag (themes) and per
    rubric tag (axes);
  * a request that got no reply from the served endpoint (a timeout, an over-context
    rejection, a server error) is a harness error, never the model's outcome: it is not graded
    and stays out of the mean, the 0.5 pass rate and the per-tag means (base marks it
    `scoring_excluded`, counts it in failed_requests and names it in status_reason). simple-evals'
    sampler instead returns "No response (bad request)." for a request its API rejects and grades
    that text; the reason names the requests left out. An empty reply (the endpoint answered with
    no text) is the model's answer and is graded as it stands.
The grader is the gbench Gemini judge cascade instead of gpt-4.1-2025-04-14, so the result carries
leaderboard_comparable=False. The grader call otherwise follows simple-evals' default HealthBench
grader (grading_sampler, ChatCompletionSampler(system_message=OPENAI_SYSTEM_MESSAGE_API,
max_tokens=2048)): the system message "You are a helpful assistant.", temperature 0.5, its 2048
tokens for the reply and no response format, so the grader's JSON reply is read with simple-evals'
parse_json_to_dict (simpleqa.grader_config). The Gemini judge counts its thinking against
max_output_tokens (gpt-4.1 does not think), so it is sent 2048 +
gemini_budget.GEMINI_THINKING_ALLOWANCE (8192); a rubric verdict that still stops there is parsed
as it arrived (judge_replies_at_max_output_tokens), a call whose every answer stopped there with no
text is a judge outage (judge_outages_after_max_output_tokens), and either is named in
leaderboard_comparable_reason. Every rubric verdict records the Gemini model that gave it
(`judge_model`) and the attempt it took (`grader_attempts`). Requires GEMINI_API_KEY or
GEMINI_API_KEYS (hard error without a key).

Sampling: as simple-evals samples the model under test (simpleqa.simple_evals_samples for the
messages, gpqa.SIMPLE_EVALS_SAMPLING for the chat sampler, as gpqa and simpleqa pass it): in both
modes the chat sampler of simple_evals.py's `gpt-4o` entry,
ChatCompletionSampler(system_message=OPENAI_SYSTEM_MESSAGE_API, max_tokens=2048), so the
conversation is sent after the system message "You are a helpful assistant.", at gbench's
temperature (base.resolve_temperature: 1.0 with `--thinking`, 0.0 without) instead of the
sampler's default 0.5, which is recorded (the grader sees that system turn too, since healthbench_eval grades the
sampler's actual_queried_message_list). Without `--thinking` every request carries max_tokens
2048 (OfficialSampling.max_tokens under base's token-limit policy; `--max-output-tokens` is not
applied); with `--thinking` it carries the run's budget (`--max-output-tokens`), the 2048 being
written for non-reasoning generation (recorded as not applied to reasoning). simple-evals'
o-series samplers (ResponsesSampler with `reasoning_model=True`, OChatCompletionSampler) are
keyed on OpenAI model names and are not reasoning settings for a served model (gpqa.py).
Override the temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_HEALTHBENCH_TEMPERATURE`, which takes precedence over both (and over the rule's 1.0 / 0.0);
the result records
`official_sampling` and `sampler_max_tokens`, and a reply cut off by any budget other than the
2048 of a no-think request (the run's budget, or a max_tokens the context window lowered) is
counted in `replies_cut_off_by_budget`.
"""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_required_skip,
                   is_scored, judge_cascade, judge_generate_cascade_detailed, strip_thinking_tags)
from .metrics import record_budget_cutoffs
from .gpqa import SIMPLE_EVALS_SAMPLING, simple_evals_messages  # noqa: F401 (shared with gpqa)
from .sampling import stratified_sample, presharded_for_base
from .simpleqa import (GRADER_TEMPERATURE, grader_config, grader_cutoff_reason,
                       grader_sampling_record, merge_grader_shards,
                       record_sampler_max_tokens, simple_evals_samples)
from .simpleqa import GRADER_SYSTEM_MESSAGE  # noqa: F401  (the grader system message, re-exported)

logger = logging.getLogger(__name__)

PILLAR = "Medical & Health"
DOCS_URL = "docs/evals/healthbench.md"
_PASS_THRESHOLD = 0.5  # per-conversation score counted as a "pass" for the secondary pass rate

_DATASET = "openai/healthbench"
#: File sha256 e99dd3c6..., identical to simple-evals' INPUT_PATH blob.
_REVISION = "40ee1968852fc57f625934251ac22be47077a8fb"
_FILENAME = "2025-05-07-06-14-12_oss_eval.jsonl"
OFFICIAL_GRADER = "gpt-4.1-2025-04-14"
#: Total grader calls per rubric item before giving up on a well-formed reply.
_GRADER_ATTEMPTS = 3
#: simple-evals ChatCompletionSampler returns this text when the API rejects a request
#: (openai.BadRequestError), and healthbench_eval grades it like any other response. gbench does
#: not: a request that got no reply from the served endpoint is a harness error, left out of the
#: score and named in leaderboard_comparable_reason with this text.
NO_RESPONSE_TEXT = "No response (bad request)."

# simple-evals healthbench_eval.py GRADER_TEMPLATE, verbatim.
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def _load_healthbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load HealthBench conversations + weighted rubrics; raises on load/schema failure."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_DATASET, filename=_FILENAME, repo_type="dataset",
                               revision=_REVISION)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load dataset for healthbench: {e}")
        raise RuntimeError(f"Could not load dataset for healthbench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for healthbench returned empty rows")
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "healthbench")
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("example_tags"), seed="healthbench")

    samples = []
    for item in rows:
        raw_prompt = item.get("prompt")
        if isinstance(raw_prompt, list):
            messages = raw_prompt
        elif raw_prompt:
            messages = [{"role": "user", "content": str(raw_prompt)}]
        else:
            raise RuntimeError("healthbench: unexpected schema (missing 'prompt')")
        rubrics = item.get("rubrics") or []
        parsed = [{"criterion": r.get("criterion", ""), "points": r.get("points", 0),
                   "tags": list(r.get("tags") or [])}
                  for r in rubrics if isinstance(r, dict) and r.get("criterion")]
        if not parsed or not any(r["points"] > 0 for r in parsed):
            raise RuntimeError(
                "healthbench: unexpected schema (no rubric with positive points); refusing to fabricate"
            )
        tags = list(item.get("example_tags") or [])
        samples.append((messages, json.dumps(parsed), {
            "category": tags[0] if tags else "clinical",   # the theme tag
            "example_tags": tags,
            "prompt_id": item.get("prompt_id", ""),
        }))

    logger.info(f"Loaded {len(samples)} healthbench samples.")
    return samples


def rubric_item_str(item: Dict[str, Any]) -> str:
    """simple-evals RubricItem.__str__."""
    return f"[{item['points']}] {item['criterion']}"


def parse_json_to_dict(json_string: Optional[str]) -> Dict[str, Any]:
    """simple-evals parse_json_to_dict: drop ```json fences, parse, {} on failure."""
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", str(json_string or "").strip())
    try:
        parsed = json.loads(json_cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def calculate_score(rubric_items: List[Dict[str, Any]], grades: List[Dict[str, Any]]) -> Optional[float]:
    """simple-evals calculate_score: achieved points / total positive points (unclipped)."""
    total_possible_points = sum(r["points"] for r in rubric_items if r["points"] > 0)
    if total_possible_points == 0:
        return None
    achieved_points = sum(r["points"] for r, g in zip(rubric_items, grades, strict=True)
                          if g["criteria_met"])
    return achieved_points / total_possible_points


def clipped_mean(values: List[float]) -> float:
    """simple-evals _compute_clipped_stats(values, "mean"): clip the MEAN, not each value."""
    return min(1.0, max(0.0, sum(values) / len(values)))


def _make_scorer(judge_usage: Counter, judge_concurrency: int = 64):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        from tqdm import tqdm

        semaphore = asyncio.Semaphore(judge_concurrency)
        config = grader_config()      # grading_sampler sends no response format

        async def _grade_item(convo_str: str, item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            """(outcome, grading): the grader's parsed reply plus the Gemini model that gave it
            (`judge_model`), the attempt it took (`grader_attempts`) and how that reply ended
            (`judge_finish_reason`; MAX_TOKENS: it stopped at simpleqa.GRADER_OUTPUT_LIMIT and is
            parsed as it arrived); for JUDGE_FAILED the last reply that had no boolean (`judge_reply`);
            for JUDGE_OUTAGE the cascade's replies that stopped at the cap with no text
            (`judge_max_tokens_without_text`, when any). Every outcome carries
            `judge_replies_at_max_output_tokens` (moved onto the trace by _judge): the grader
            replies of every attempt (a retried one included) that stopped at MAX_TOKENS."""
            grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
                "<<rubric_item>>", rubric_item_str(item))
            last: Dict[str, Any] = {}
            cut = 0         # replies of this item's attempts that stopped at MAX_TOKENS
            for attempt in range(1, _GRADER_ATTEMPTS + 1):
                async with semaphore:
                    text, model, info = await judge_generate_cascade_detailed(grader_prompt,
                                                                              config=config)
                if text is None:
                    empty_at_cap = int(info.get("max_tokens_without_text") or 0)
                    return "JUDGE_OUTAGE", {"judge_replies_at_max_output_tokens": cut,
                                            **({"judge_max_tokens_without_text": empty_at_cap}
                                               if empty_at_cap else {})}
                judge_usage[model] += 1
                if info.get("finish_reason") == "MAX_TOKENS":
                    cut += 1
                grading = parse_json_to_dict(text)
                label = grading.get("criteria_met")
                if label is True or label is False:
                    return "OK", {**grading, "judge_model": model, "grader_attempts": attempt,
                                  "judge_finish_reason": info.get("finish_reason"),
                                  "judge_replies_at_max_output_tokens": cut}
                last = {"judge_model": model, "grader_attempts": attempt,
                        "judge_finish_reason": info.get("finish_reason"),
                        "judge_reply": text.strip(),
                        "judge_replies_at_max_output_tokens": cut}
            return "JUDGE_FAILED", last

        async def _judge(trace: Dict[str, Any], pbar) -> None:
            raw = trace.get("response_text")
            rubric = json.loads(trace.get("gold_answer") or "[]")
            if raw is None or trace.get("harness_error"):
                # No reply from the served endpoint: a harness error, never the model's outcome.
                # Not graded and left out of the mean (base counts it once in failed_requests and
                # names it in status_reason).
                trace["judge_grade"] = "not_judged_failed_request"
                trace["scoring_excluded"] = True
                trace["is_correct"] = False
                pbar.update(1)
                return
            response = strip_thinking_tags(raw)
            convo = [m for m in (trace.get("messages") or []) if isinstance(m, dict)]
            convo_str = "\n\n".join(
                f"{m.get('role')}: {m.get('content')}"
                for m in convo + [{"role": "assistant", "content": response}])
            graded = await asyncio.gather(*[_grade_item(convo_str, item) for item in rubric])
            if trace.get("status") == "PENDING_JUDGE":   # keep EMPTY_RESPONSE etc.
                trace["status"] = "OK"
            outcomes = {o for o, _ in graded}
            # every grader reply's end, per rubric item (the result counts them)
            trace["judge_finish_reasons"] = [g.get("judge_finish_reason") for _, g in graded]
            # grader replies that stopped at the limit, over every attempt (retried ones too);
            # taken out of each item's grading, which keeps the grader's own fields
            trace["judge_replies_at_max_output_tokens"] = sum(
                int(g.pop("judge_replies_at_max_output_tokens", 0) or 0) for _, g in graded)
            empty_at_cap = sum(int(g.get("judge_max_tokens_without_text") or 0)
                               for _, g in graded)
            if empty_at_cap:
                trace["judge_max_tokens_without_text"] = empty_at_cap
            if "JUDGE_OUTAGE" in outcomes:
                # Judge cascade exhausted (infra outage): excluded from the mean, NOT scored 0.
                trace["judge_grade"] = "JUDGE_OUTAGE"
                pbar.update(1)
                return
            if "JUDGE_FAILED" in outcomes:
                # The grader answered but never with a boolean: a grading failure, reported
                # as one and kept out of the mean, the 0.5 pass rate and the per-theme counts
                # (scoring_excluded) instead of silently scoring 0. The replies that never
                # carried a boolean are kept for audit.
                trace["judge_failed_items"] = [
                    {"criterion": item.get("criterion"), **g}
                    for item, (o, g) in zip(rubric, graded) if o == "JUDGE_FAILED"]
                trace["judge_grade"] = "JUDGE_FAILED"
                trace["scoring_excluded"] = True
                trace["is_correct"] = False
                pbar.update(1)
                return
            grades = [g for _, g in graded]
            score = calculate_score(rubric, grades)
            trace["healthbench_score"] = round(score, 6)
            trace["rubric_grades"] = [
                {**item, "criteria_met": g["criteria_met"],
                 "explanation": g.get("explanation", "No explanation provided"),
                 "judge_model": g.get("judge_model"), "grader_attempts": g.get("grader_attempts"),
                 "judge_finish_reason": g.get("judge_finish_reason")}
                for item, g in zip(rubric, grades)]
            trace["is_correct"] = bool(score >= _PASS_THRESHOLD)
            trace["judge_grade"] = f"SCORE_{score:.2f}"
            pbar.update(1)

        with tqdm(total=len(sample_traces), desc="Judging [HEALTHBENCH]") as pbar:
            await asyncio.gather(*[_judge(t, pbar) for t in sample_traces])
    return _score


def healthbench_summary(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall and per-tag clipped means, as simple-evals _aggregate_get_clipped_mean reports them.

    Example tags (theme:..., physician_agreed_category:...) take the conversation's overall
    score; rubric tags (axis:..., level:..., cluster:...) take calculate_score over that
    conversation's items carrying the tag, when any of them has positive points. A trace
    base.is_scored rejects (a judge outage, or `scoring_excluded`: a request that got no reply
    from the served endpoint, a grading failure) is left out.
    """
    overall: List[float] = []
    tag_values: Dict[str, List[float]] = defaultdict(list)
    for t in traces:
        if ("healthbench_score" not in t or not is_scored(t)
                or t.get("judge_grade") == "JUDGE_FAILED"):
            continue
        score = float(t["healthbench_score"])
        overall.append(score)
        for tag in ((t.get("extra_payload") or {}).get("example_tags") or []):
            tag_values[tag].append(score)
        by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in t.get("rubric_grades") or []:
            for tag in item.get("tags") or []:
                by_tag[tag].append(item)
        for tag, items in by_tag.items():
            tag_score = calculate_score(items, items)
            if tag_score is not None:
                tag_values[tag].append(tag_score)
    if not overall:
        return {}
    return {
        "overall_score": clipped_mean(overall),
        "n_samples": len(overall),
        "tag_scores": {tag: round(clipped_mean(v), 4) for tag, v in sorted(tag_values.items())},
    }


def run_healthbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run HealthBench rubric grading (hard-errors without a Gemini key)."""
    skip = gemini_required_skip("healthbench", model_name)
    if skip is not None:
        return skip
    limit = kwargs.get("limit")
    samples = simple_evals_samples(_load_healthbench_samples(limit=limit), enable_thinking)
    judge_usage: Counter = Counter()
    eval_plan = eval_limit_plan(
        "healthbench", limit, len(samples),
        attempts=base_attempts(batch_scored=True, supports_attempts=True), unit="conversations")
    result = run_eval_suite(
        eval_name="healthbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's conversations, kept as is
        async_eval_fn=_make_scorer(judge_usage),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        supports_attempts=True,   # every trace is graded on its own
        official_sampling=SIMPLE_EVALS_SAMPLING,   # per mode, cap included (token-limit policy)
    )
    result["eval_limit_plan"] = eval_plan
    max_tokens_reasons = record_sampler_max_tokens(result, enable_thinking)
    # HealthBench's score is the clipped mean rubric score, not the share of conversations
    # clearing 0.5 (kept as pass_rate_at_0.5). Outages and grader failures carry no score and
    # stay out of the mean; they are reported as judge_outages / judge_failures. A request that
    # got no reply from the served endpoint is a harness error: not graded, left out of the mean,
    # counted in failed_requests and named in status_reason.
    traces = result.get("sample_traces", [])
    summary = healthbench_summary(traces)
    result["pass_rate_at_0.5"] = result.get("accuracy")
    result["metric"] = ("HealthBench overall score: clip(mean weighted rubric score, 0, 1) over "
                        "conversations (simple-evals)")
    if summary:
        result["healthbench_mean_rubric_score"] = round(summary["overall_score"], 4)
        result["healthbench_scored_conversations"] = summary["n_samples"]
        result["healthbench_tag_scores"] = summary["tag_scores"]
        result["accuracy"] = round(summary["overall_score"] * 100.0, 2)
        # The per-theme table shows the theme's clipped mean, not base's 0.5 pass rate.
        for theme, cell in (result.get("category_accuracy") or {}).items():
            if isinstance(cell, dict) and theme in summary["tag_scores"]:
                cell["pass_rate_at_0.5"] = cell.get("accuracy")
                cell["accuracy"] = round(summary["tag_scores"][theme] * 100.0, 2)
    no_reply_kinds = Counter(str((t.get("harness_error") or {}).get("kind") or "no_reply")
                             for t in traces if t.get("harness_error"))
    no_reply = sum(no_reply_kinds.values())
    if not summary and traces:
        # No conversation received a score (every one a request with no reply from the served
        # endpoint, a judge outage or a grading failure): the score is withheld, never 0.
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(no_reply_kinds.items()))
        why = (f"no conversation received a HealthBench score: "
               f"{no_reply} request(s) that got no reply from the served endpoint (harness "
               f"errors{': ' + kinds if kinds else ''}), "
               f"{result.get('judge_outages', 0)} judge outage(s), "
               f"{result.get('judge_failures', 0)} grader failure(s)")
        result["accuracy"] = None
        result["status"] = "failed"
        result["status_reason"] = why
        result["accuracy_withheld_reason"] = why
    result["dataset"] = f"{_DATASET}@{_REVISION} ({_FILENAME})"
    result["judge"] = "gbench-gemini-cascade"
    result["judge_models"] = dict(judge_usage)   # answered grader calls per model
    # grader replies that stopped at simpleqa.GRADER_OUTPUT_LIMIT (parsed as they arrived), and
    # judge outages whose cascade saw replies stop at the limit with no text
    # (every attempt's: a cut reply retried for want of a boolean counts too)
    result["judge_replies_at_max_output_tokens"] = sum(
        int(t.get("judge_replies_at_max_output_tokens") or 0) for t in traces)
    result["judge_outages_after_max_output_tokens"] = sum(
        1 for t in traces if t.get("judge_max_tokens_without_text")
        and t.get("judge_grade") == "JUDGE_OUTAGE")
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = grader_sampling_record()
    reasons = [f"graded by the gbench Gemini judge cascade "
               f"({', '.join(sorted(judge_usage)) or 'no judge call answered'}) at simple-evals' "
               f"grader temperature {GRADER_TEMPERATURE}, not simple-evals' {OFFICIAL_GRADER}"]
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint and are left "
                       f"out of the mean as harness errors (simple-evals grades a request its "
                       f"API rejects as {NO_RESPONSE_TEXT!r} and retries other errors until a "
                       "response arrives)")
    if limit:
        reasons.append(f"--eval-limit {limit} runs a subset of the 5,000 conversations")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (simple-evals' chat sampler "
                       "samples at 0.5)")
    reasons.extend(max_tokens_reasons)
    grader_cut = grader_cutoff_reason(result)
    if grader_cut:
        reasons.append(grader_cut)
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """gbench --merge: simpleqa.merge_grader_shards("healthbench", ...) (the grader cut-off counts
    summed and named once over the union)."""
    return merge_grader_shards("healthbench", results)
