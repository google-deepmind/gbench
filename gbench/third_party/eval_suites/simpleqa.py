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

# Evaluation Suite: simpleqa
# Description: OpenAI SimpleQA (Factuality & Hallucination Benchmark - 4,326 short questions)

"""gbench native built-in runner for simpleqa (Factuality & Knowledge).

Protocol, as in openai/simple-evals `simpleqa_eval.py`: the question alone is the user turn;
the reply is graded by an LLM with simple-evals' GRADER_TEMPLATE (verbatim) into CORRECT /
INCORRECT / NOT_ATTEMPTED, parsed with simple-evals' own letter regex. The headline is
simple-evals' `score` = is_correct over all graded questions; F-score and
accuracy_given_attempted are reported alongside, as simple-evals prints them.

The grader model is gbench's Gemini cascade (simple-evals uses gpt-4.1-2025-04-14), so the
result is flagged leaderboard_comparable=False and names the models that graded it. The grader
call otherwise follows simple-evals' grading_sampler (simple_evals.py @652c89d0,
ChatCompletionSampler(model="gpt-4.1-2025-04-14", system_message=OPENAI_SYSTEM_MESSAGE_API,
max_tokens=2048)): the system message "You are a helpful assistant.", the sampler's default
temperature 0.5, its max_tokens 2048 for the reply and no response format (grader_config).
Gemini counts its thinking against max_output_tokens (gpt-4.1 does not think), so the grader is
sent 2048 + gemini_budget.GEMINI_THINKING_ALLOWANCE (8192) as max_output_tokens: its reply keeps
upstream's 2048 tokens. A grader reply that still stops there is parsed as it arrived, as
simple-evals parses a reply cut at its max_tokens (counted in judge_replies_at_max_output_tokens),
and a call whose every answer stopped there with no text ends as a judge outage (excluded,
counted in judge_outages and judge_outages_after_max_output_tokens); either makes the run name
them in leaderboard_comparable_reason.
The gold target enters the grader prompt exactly as the CSV stores it (simple-evals passes
row["answer"] unchanged), surrounding whitespace included. An empty reply is the model's answer:
it is not sent to the grader and counts as NOT_ATTEMPTED. A request that got no reply from the
served endpoint is a harness error: it is not graded and stays out of every figure (base marks
it `scoring_excluded` and names it in `status_reason`), and a run with nothing graded withholds
its score (accuracy and the split null) instead of reporting 0.

Sampling: simple-evals' chat sampler, shared with gpqa: gpqa.SIMPLE_EVALS_SAMPLING, the one
declaration that base's token-limit policy applies per mode, and gpqa.simple_evals_messages.
Every request, with and without `--thinking`, is shaped as simple_evals.py's `gpt-4o` entry sends
it, a
ChatCompletionSampler (sampler/chat_completion_sampler.py) with
`system_message=OPENAI_SYSTEM_MESSAGE_API` and `max_tokens=2048`: the system message "You are
a helpful assistant." before the question. The temperature follows gbench's rule
(base.resolve_temperature), not the sampler's default 0.5: 1.0 with `--thinking`, 0.0 without;
the sampler's 0.5 is recorded under `official_sampling`. A no-think request
carries max_tokens 2048 (OfficialSampling.max_tokens), so
`--max-output-tokens` is not applied (base's `max_output_tokens_note`) and a longer reply is cut
there and graded as it arrived, as simple-evals does (`replies_at_sampler_max_tokens` counts
them, as base's `replies_at_official_max_tokens`). A `--thinking` request carries the run's
budget (`--max-output-tokens`) instead: the 2048 was written for non-reasoning generation, and
`official_sampling` records it as not applied to reasoning, with `max_tokens_sent`.
simple_evals.py's o-series entries (ResponsesSampler with `reasoning_model=True`,
OChatCompletionSampler), which send no system message, temperature or max_tokens, are keyed on
OpenAI model names and are not reasoning settings for a served model (gpqa.py). The result
records `official_sampling` (with `max_tokens` 2048, and `max_tokens_applied` true on a no-think
run and false with `--thinking`) and `sampler_max_tokens`; a reply cut off by any budget other
than the 2048 of a no-think request (the run's budget, or a max_tokens the context window
lowered) is counted in `replies_cut_off_by_budget`. Override the temperature for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SIMPLEQA_TEMPERATURE`, which takes
precedence over both (and over the rule's 1.0 / 0.0); a temperature other than 0.5 is named in
`leaderboard_comparable_reason`.
"""

import ast
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_required_skip,
                   is_scored, judge_cascade, judge_config, judge_generate_cascade_detailed)
from .gemini_budget import (cutoff_reason, gemini_output_limit, merge_with_cutoff_reason,
                            output_limit_record)
from .metrics import record_budget_cutoffs
from .gpqa import (SIMPLE_EVALS_CHAT_MAX_TOKENS, SIMPLE_EVALS_SAMPLING,
                   SIMPLE_EVALS_SYSTEM_MESSAGE, simple_evals_messages)
from .sampling import stratified_sample, presharded_for_base

logger = logging.getLogger(__name__)

PILLAR = "Factuality & Knowledge"

#: The HF mirror at this commit is byte-identical to simple-evals' canonical
#: simple_qa_test_set.csv (sha256 feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032,
#: 4,326 rows); pinned so the question set cannot drift under the suite.
_DATASET = "basicv8vc/SimpleQA"
_REVISION = "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94"
_FULL_SIZE = 4326

# Copied verbatim from openai/simple-evals simpleqa_eval.py (commit 652c89d0ca9d), including
# the trailing spaces on four lines. Do not edit: the few-shot rubric is the grading rule.
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

#: simple_evals.py (@652c89d0) grading_sampler = ChatCompletionSampler(model="gpt-4.1-2025-04-14",
#: system_message=OPENAI_SYSTEM_MESSAGE_API, max_tokens=2048), at the sampler's default
#: temperature 0.5. HealthBench's default grader and BrowseComp's grader are the same sampler.
GRADER_SYSTEM_MESSAGE = SIMPLE_EVALS_SYSTEM_MESSAGE
GRADER_TEMPERATURE = 0.5
GRADER_MAX_TOKENS = 2048
GRADER_REFERENCE = ("openai/simple-evals@652c89d0 simple_evals.py grading_sampler: "
                    "ChatCompletionSampler(model='gpt-4.1-2025-04-14', "
                    "system_message=OPENAI_SYSTEM_MESSAGE_API, max_tokens=2048), default "
                    "temperature 0.5")


#: What the grader calls send as max_output_tokens: grading_sampler's 2048 for the reply plus the
#: thinking allowance (the Gemini grader counts its thinking against max_output_tokens).
GRADER_OUTPUT_LIMIT = gemini_output_limit(GRADER_MAX_TOKENS)


def grader_config(**overrides: Any) -> Any:
    """The judge config for simple-evals' grading sampler: its system message, its temperature
    0.5 and its max_tokens=2048 for the reply, sent as max_output_tokens GRADER_OUTPUT_LIMIT
    (2048 + gemini_budget.GEMINI_THINKING_ALLOWANCE).

    The model is the gbench Gemini cascade; the call follows grading_sampler otherwise, which
    sends no response format. The Gemini judge counts its thinking against max_output_tokens
    (gpt-4.1 does not think), so the allowance keeps the reply's 2048. Shared with healthbench
    and browsecomp."""
    cfg = judge_config(system_instruction=GRADER_SYSTEM_MESSAGE,
                       max_output_tokens=GRADER_OUTPUT_LIMIT, **overrides)
    if cfg is not None:
        cfg = cfg.model_copy(update={"temperature": GRADER_TEMPERATURE})
    return cfg


def grader_sampling_record() -> Dict[str, Any]:
    """The `judge_sampling` block of a result graded through grader_config()."""
    return {"reference": GRADER_REFERENCE, "system_instruction": GRADER_SYSTEM_MESSAGE,
            "temperature": GRADER_TEMPERATURE,
            "response_mime_type": "not sent (grading_sampler sends no response_format)",
            **output_limit_record(GRADER_MAX_TOKENS, "grading_sampler's max_tokens",
                                  "gpt-4.1")}


def grader_cutoff_reason(result: Dict[str, Any]) -> str:
    """The leaderboard_comparable reason for grader replies still cut at GRADER_OUTPUT_LIMIT
    (result's judge_replies_at_max_output_tokens / judge_outages_after_max_output_tokens), or
    "" when there are none. Shared with healthbench and browsecomp."""
    return cutoff_reason("grader", int(result.get("judge_replies_at_max_output_tokens") or 0),
                         int(result.get("judge_outages_after_max_output_tokens") or 0),
                         GRADER_MAX_TOKENS)


def merge_grader_shards(suite: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """gbench --merge for a simple-evals suite (simpleqa, healthbench, browsecomp): the generic
    re-count of the union (merge.SUMMED_COUNTS sums the grader cut-off counts), with
    grader_cutoff_reason stated once over the summed counts instead of each shard's."""
    return merge_with_cutoff_reason(suite, results, grader_cutoff_reason)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """gbench --merge: merge_grader_shards("simpleqa", ...)."""
    return merge_grader_shards("simpleqa", results)


def simple_evals_samples(samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
                         thinking: bool) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """The samples with the messages simple-evals' chat sampler sends, the same in both modes
    (gpqa.simple_evals_messages; shared with healthbench and browsecomp). The per-request
    max_tokens is not in the samples: it is declared once, in gpqa.SIMPLE_EVALS_SAMPLING, which
    these suites pass to run_eval_suite (2048 on a no-think run, the run's budget with
    --thinking)."""
    return [(simple_evals_messages(messages, thinking), gold, dict(meta))
            for messages, gold, meta in samples]


def record_sampler_max_tokens(result: Dict[str, Any], thinking: bool) -> List[str]:
    """`sampler_max_tokens` and `replies_at_sampler_max_tokens` (gpqa's field names) on `result`,
    read from base's official-cap fields; returns the comparability reasons they add.

    `sampler_max_tokens` is 2048 on a no-think run and None with --thinking (the 2048 is not
    applied to reasoning; the run's budget is sent);
    `replies_at_sampler_max_tokens` is base's `replies_at_official_max_tokens`, the replies that
    stopped at 2048 (simple-evals cuts them the same way). A no-think request whose prompt left
    less than 2048 tokens of the context window got a smaller max_tokens, which simple-evals would
    not have sent: those are named."""
    cap = None if thinking else SIMPLE_EVALS_CHAT_MAX_TOKENS
    traces = result.get("sample_traces") or []
    result["sampler_max_tokens"] = cap
    result["replies_at_sampler_max_tokens"] = (
        int(result.get("replies_at_official_max_tokens") or 0) if cap else 0)
    lowered = sum(1 for t in traces
                  if cap and isinstance(t.get("max_tokens_sent"), int)
                  and 0 < t["max_tokens_sent"] < cap)
    if lowered:
        return [f"{lowered} request(s) got less than simple-evals' max_tokens={cap} because the "
                "prompt left too little of the context window"]
    return []


def withhold_if_nothing_scored(result: Dict[str, Any], what: str = "question") -> bool:
    """When no trace was scored (every one a judge outage, a grader failure or otherwise left
    out of the score) but the run is not already `failed`, mark it `failed` and withhold its
    score (accuracy None, `accuracy_withheld_reason`), so a run of nothing but harness errors
    never reports 0. Returns whether it did. Shared by the judged suites of this family."""
    traces = result.get("sample_traces") or []
    scored = sum(1 for t in traces if str(t.get("judge_grade") or "") != "JUDGE_OUTAGE"
                 and not t.get("scoring_excluded"))
    if not traces or scored or result.get("status") in ("error", "failed"):
        return False
    why = (f"no {what} was scored: {result.get('judge_outages', 0)} judge outage(s), "
           f"{result.get('judge_failures', 0)} judge failure(s), "
           f"{result.get('scoring_excluded', 0)} left out of the score, of {len(traces)}")
    result["accuracy"] = None
    result["status"] = "failed"
    result["status_reason"] = why
    result["accuracy_withheld_reason"] = why
    return True


CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


def grade_letter(grading_response: Optional[str]) -> str:
    """simple-evals' parse: the first A/B/C in the grader reply; "C" (NOT_ATTEMPTED) if none."""
    match = re.search(r"(A|B|C)", grading_response or "")
    return match.group(0) if match else "C"


def _simpleqa_topic(metadata: Any) -> Optional[str]:
    """Topic out of SimpleQA's `metadata` column, which is a stringified Python dict."""
    if isinstance(metadata, dict):
        topic = metadata.get("topic")
        return str(topic).strip() if topic else None
    text = str(metadata or "").strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, dict) and value.get("topic"):
            return str(value["topic"]).strip()
    m = re.search(r"['\"]topic['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    return m.group(1).strip() if m else None


def _load_simpleqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load SimpleQA (pinned HF mirror of simple-evals' CSV); raises on load/schema failure."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET, split='test', revision=_REVISION)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for simpleqa: {e}")
        raise RuntimeError(f"Could not load dataset for simpleqa: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for simpleqa returned empty rows")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "simpleqa")
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("metadata"), seed="simpleqa")

    samples = []
    for item in rows:
        prompt = item.get("problem")
        gold = item.get("answer")
        if not prompt or gold is None:
            raise RuntimeError(
                "simpleqa: unexpected dataset schema (missing 'problem'/'answer'); "
                "refusing to fabricate sample data"
            )
        # The gold target goes into GRADER_TEMPLATE as the CSV stores it: simple-evals passes
        # row["answer"] unchanged, and 211 of the 4,326 answers carry surrounding whitespace.
        gold = str(gold)
        # SimpleQA has no top-level `category`; the topic lives inside `metadata`, which
        # the row stores as a stringified dict. The default therefore applied to every
        # row and the per-category table reported one bucket, "geography", for the whole
        # benchmark.
        cat = item.get("category") or _simpleqa_topic(item.get("metadata")) or "uncategorized"

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} simpleqa samples.")
    return samples


async def _async_judge_simpleqa(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 64,
) -> None:
    """simple-evals SimpleQA grading (GRADER_TEMPLATE + letter parse) over the Gemini cascade."""
    from tqdm import tqdm
    import asyncio

    semaphore = asyncio.Semaphore(judge_concurrency)
    cfg = grader_config()

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        if resp_text is None or trace.get("harness_error"):
            # No reply from the served endpoint: a harness error, never the model's outcome. Not
            # graded and left out of every figure (base marks it scoring_excluded and names it).
            trace["is_correct"] = False
            trace["scoring_excluded"] = True
            trace["judge_grade"] = "not_judged_failed_request"
            pbar.update(1)
            return
        if not resp_text:
            # Nothing to grade, so the judge is not called. An empty reply gives no answer and
            # contradicts nothing, which is the rubric's NOT_ATTEMPTED; base already counts
            # the sample under failed_requests / empty_responses.
            trace["is_correct"] = False
            trace["simpleqa_grade"] = "NOT_ATTEMPTED"
            trace["judge_grade"] = "not_judged_empty_response"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        # The question is the user turn (simple-evals' system message comes first).
        question = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        gold = trace.get("gold_answer")
        gold = "" if gold is None else str(gold)

        prompt = GRADER_TEMPLATE.format(question=question, target=gold, predicted_answer=resp_text)

        async with semaphore:
            text, judge_used, info = await judge_generate_cascade_detailed(prompt, config=cfg)
        if text is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy,
            # NOT scored 0. base.run_eval_suite drops it from pass/fail accuracy.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            if info.get("max_tokens_without_text"):
                # answers of the cascade that stopped at max_output_tokens with no text
                trace["judge_max_tokens_without_text"] = int(info["max_tokens_without_text"])
            trace["status"] = "OK"
            pbar.update(1)
            return
        # how the grader's reply ended (MAX_TOKENS: it stopped at GRADER_OUTPUT_LIMIT and is
        # parsed as it arrived)
        trace["judge_finish_reason"] = info.get("finish_reason")

        grade = CHOICE_LETTER_TO_STRING[grade_letter(text)]
        trace["is_correct"] = grade == "CORRECT"
        trace["simpleqa_grade"] = grade
        trace["judge_grade"] = grade
        trace["judge_reply"] = text.strip()
        trace["judge_model"] = judge_used
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [SIMPLEQA]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def simpleqa_aggregate(sample_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """simple-evals' SimpleQA aggregate over the graded traces.

    Mirrors SimpleQAEval.__call__: score = is_correct; is_given_attempted = is_correct +
    is_incorrect; accuracy_given_attempted = is_correct / is_given_attempted; f1 = harmonic
    mean of accuracy_given_attempted and is_correct. Fractions in [0, 1]. A trace base.is_scored
    rejects (a judge outage, or `scoring_excluded`: a request that got no reply from the served
    endpoint) is not counted; `n` is 0 when nothing was graded.
    """
    counts = Counter(t.get("simpleqa_grade") for t in sample_traces if is_scored(t))
    n = sum(counts[g] for g in CHOICE_STRINGS)
    is_correct = counts["CORRECT"] / n if n else 0.0
    is_incorrect = counts["INCORRECT"] / n if n else 0.0
    is_not_attempted = counts["NOT_ATTEMPTED"] / n if n else 0.0
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = (is_correct / is_given_attempted
                                if is_given_attempted > 0 else 0)
    f1 = (2 * accuracy_given_attempted * is_correct / (accuracy_given_attempted + is_correct)
          if (accuracy_given_attempted + is_correct) > 0 else 0)
    return {"n": n, "is_correct": is_correct, "is_incorrect": is_incorrect,
            "is_not_attempted": is_not_attempted, "is_given_attempted": is_given_attempted,
            "accuracy_given_attempted": accuracy_given_attempted, "f1": f1}


def run_simpleqa(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute simpleqa native built-in evaluation suite."""
    skip = gemini_required_skip("simpleqa", model_name)
    if skip is not None:
        return skip
    limit = kwargs.get("limit")
    samples = simple_evals_samples(_load_simpleqa_samples(limit=limit), enable_thinking)
    eval_plan = eval_limit_plan(
        "simpleqa", limit, len(samples), attempts=base_attempts(batch_scored=True))
    result = run_eval_suite(
        eval_name="simpleqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        async_eval_fn=_async_judge_simpleqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=SIMPLE_EVALS_SAMPLING,   # per mode, cap included (token-limit policy)
    )
    result["eval_limit_plan"] = eval_plan
    max_tokens_reasons = record_sampler_max_tokens(result, enable_thinking)
    traces = result.get("sample_traces", []) or []
    agg = simpleqa_aggregate(traces)
    graded = bool(agg["n"])
    if graded:
        # Headline = simple-evals' `score` (is_correct over every graded question).
        result["accuracy"] = round(agg["is_correct"] * 100.0, 2)
    # With nothing graded (every question a harness error or a judge outage) the split is
    # withheld (null) along with the headline, never reported as 0.
    for key, value in (("correct", agg["is_correct"]), ("incorrect", agg["is_incorrect"]),
                       ("not_attempted", agg["is_not_attempted"]),
                       ("accuracy_given_attempted", agg["accuracy_given_attempted"]),
                       ("f_score", agg["f1"])):
        result[key] = round(value * 100.0, 2) if graded else None
    result["metric"] = ("SimpleQA score = is_correct over all graded questions (simple-evals "
                        "headline); f_score (harmonic mean of correct and "
                        "accuracy_given_attempted), accuracy_given_attempted and the "
                        "correct / incorrect / not_attempted split are reported alongside")
    result["judge"] = "gbench-gemini-cascade"
    # One judge call per graded question, so this is also the answered calls per model.
    result["judge_models"] = dict(Counter(t["judge_model"] for t in traces
                                          if t.get("judge_model")))
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = grader_sampling_record()
    # Upstream's parse takes the first A/B/C anywhere, so a grader that adds prose ("Based on
    # ...") is read by its first capital letter. Count those replies so a run can be audited.
    result["grader_replies_not_bare_letter"] = sum(
        1 for t in traces
        if "judge_reply" in t and t["judge_reply"] not in CHOICE_LETTERS)
    # Grader replies that stopped at GRADER_OUTPUT_LIMIT (parsed as they arrived), and judge
    # outages whose cascade answers stopped there with no text.
    result["judge_replies_at_max_output_tokens"] = sum(
        1 for t in traces if t.get("judge_finish_reason") == "MAX_TOKENS")
    result["judge_outages_after_max_output_tokens"] = sum(
        1 for t in traces if t.get("judge_max_tokens_without_text"))
    reasons = ["graded by gbench's standard Gemini cascade instead of simple-evals' "
               "gpt-4.1-2025-04-14 grader (a gbench convention)"]
    if limit:
        reasons.append(f"--eval-limit subset ({len(samples)} of {_FULL_SIZE} questions)")
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
    withhold_if_nothing_scored(result)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
