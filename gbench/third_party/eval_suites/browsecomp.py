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

# Evaluation Suite: browsecomp
# Description: OpenAI BrowseComp (Agentic Web Search & Deep Research Fact Retrieval)

r"""gbench native built-in runner for browsecomp (Agentic & Web Research).

The smolagents/browse_comp dataset stores the 'problem' and 'answer' fields
XOR-encrypted with a per-row 'canary' password (the canonical OpenAI
simple-evals scheme). We decrypt them at load time and grade with the canonical
BrowseComp LLM grader.

What is canonical here, verified against openai/simple-evals@652c89d0
(`browsecomp_eval.py`, `simple_evals.py`, `sampler/chat_completion_sampler.py`):

* **Data:** `smolagents/browse_comp` pinned at revision f975f60a, row for row identical to
  simple-evals' `browse_comp_test_set.csv` (sha256 7b24471c..., 1,266 rows), decrypted with
  simple-evals' `derive_key` / `decrypt`.
* **Prompt:** `QUERY_TEMPLATE` renders byte-identical user turns to upstream's (whose
  placeholder is `{Question}`; gbench's is `{question}`): the question followed by the
  Explanation / Exact Answer / Confidence format block, as the user message. There is no extra
  header or footer.
* **Sampling:** simple-evals samples every eval through the `gpt-4o` entry of simple_evals.py,
  a `ChatCompletionSampler(system_message=OPENAI_SYSTEM_MESSAGE_API, max_tokens=2048)` at the
  sampler's default temperature 0.5, which prepends the system message "You are a helpful
  assistant.". gbench sends the same (simpleqa.simple_evals_samples for the messages,
  gpqa.SIMPLE_EVALS_SAMPLING for the sampler, as simpleqa and healthbench do): the system message
  and max_tokens, but not the temperature: the model samples at gbench's rule for every suite
  (base.resolve_temperature), 0.0 without `--thinking` and 1.0 with it (the sampler's 0.5 is
  recorded under `official_sampling`); max_tokens 2048 on every no-think request (a reply that
  stops there is graded as it arrived, as simple-evals grades it; `--max-output-tokens` is not
  applied) and the run's `--max-output-tokens` with `--thinking` (the 2048 was written for
  non-reasoning generation). simple-evals' o-series samplers are keyed on OpenAI model names and
  are not reasoning settings for a served model (gpqa.py). Override the temperature for a whole
  run with `--temperature`, or for this suite alone with `GBENCH_BROWSECOMP_TEMPERATURE`, which
  takes precedence over both; a value other than 0.5 is named in `leaderboard_comparable_reason`.
* **Grader:** `GRADER_TEMPLATE` is upstream's text exactly, including its duplicated "if there
  if there", the two blank lines before "confidence:" and the literal `0|\%|` (upstream writes
  `\%` in a plain string, which Python keeps as backslash-percent). `[question]` receives the raw
  decrypted problem, as `grade_sample(problem, answer, response_text)` does upstream, not the
  format-wrapped user turn. The grader call follows simple-evals' `grading_sampler`
  (`ChatCompletionSampler(model="gpt-4.1-2025-04-14", system_message=OPENAI_SYSTEM_MESSAGE_API,
  max_tokens=2048)`, temperature 0.5): the system message, temperature 0.5, the 2048 tokens for
  the reply, the prompt as the only user message (simpleqa.grader_config); the model is gbench's
  Gemini cascade. Gemini counts its thinking against max_output_tokens (gpt-4.1 does not think),
  so the grader is sent 2048 + gemini_budget.GEMINI_THINKING_ALLOWANCE (8192). A grader reply
  that still stops there is counted in `judge_replies_at_max_output_tokens`, a call whose every
  answer stopped there with no text in `judge_outages_after_max_output_tokens`, and either is
  named in `leaderboard_comparable_reason`.
* `n_repeats` is 1 upstream, and 1 here.

* **Tools / protocol:** Upstream declares NO tools and runs a single turn - browsing-capable
  models brought their own browser, others answered closed-book from parametric memory. gbench
  matches this exactly: no tools, single turn, no injected search. (An earlier gbench build
  injected a Gemini-grounded `web_search` + tool loop because a non-browsing model scored 0/20;
  that was a deliberate deviation and made the suite non-comparable. It has been REMOVED - the
  canonical protocol is closed-book, and a low closed-book score is the correct result for a
  non-browsing model, not a bug. Upstream itself reports 0.6% for GPT-4o without browsing.)

What DEVIATES:

* **Grader model:** Upstream uses `gpt-4.1-2025-04-14`; gbench grades with its standard Gemini
  cascade (a gbench convention). Each trace records the model that answered (`judge_model`) and
  its reply (`judge_reply`).
* **Verdict parse:** Upstream's grade_sample returns `re.search(r"correct: (yes|no)", reply)
  .group(0)` ("correct: yes") and __call__ compares that with "yes", so upstream's code as
  written never scores a sample correct. gbench reads the captured yes/no instead (the evident
  intent), and its pattern `correct:\s*(yes|no)` also accepts other capitalisation and spacing;
  `judge_verdicts_read_only_by_relaxed_parse` counts the verdicts upstream's pattern would miss.
* **A grader reply without a verdict:** Upstream defaults it to "no". It is a grader failure,
  not the model's answer, so gbench marks it JUDGE_FAILED, leaves it out of the score, counts it
  in `judge_failures` (the run is `completed_with_errors`) and names it in
  `leaderboard_comparable_reason`.

A run here is therefore reported with `leaderboard_comparable=False` and a
`leaderboard_comparable_reason` that names every difference that applies to it.

A note on the expected score: this is closed-book (no browsing), and BrowseComp is built to
defeat exactly that - upstream reports 0.6% for GPT-4o without browsing against 51.5% for Deep
Research. A low number here is the model's genuine closed-book fact-retrieval score, not a
harness bug.
"""

import base64
import hashlib
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_required_skip,
                   judge_cascade, judge_generate_cascade_detailed)
from .gpqa import SIMPLE_EVALS_SAMPLING
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base
from .fc_common import base_suite_plan
from .limit_plan import base_attempts
from .simpleqa import (grader_config, grader_cutoff_reason, grader_sampling_record,
                       merge_grader_shards,
                       record_sampler_max_tokens, simple_evals_samples,
                       withhold_if_nothing_scored)

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Web Research"

#: The HF mirror at this revision is row for row identical to simple-evals'
#: browse_comp_test_set.csv (sha256 _UPSTREAM_CSV_SHA256, 1,266 rows); pinned so the question
#: set cannot drift under the suite.
_DATASET = "smolagents/browse_comp"
_REVISION = "f975f60afc3c811de82951165f5a9a921d024e14"
_UPSTREAM_CSV_SHA256 = "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
_FULL_SIZE = 1266
UPSTREAM = "openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47"
#: The raw decrypted problem, kept in the sample meta for the grader's [question].
PROBLEM_KEY = "browsecomp_problem"

# Canonical OpenAI simple-evals BrowseComp templates (browsecomp_eval.py @652c89d0).
QUERY_TEMPLATE = """{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

# Upstream's text exactly: "if there if there", two blank lines before "confidence:", and
# `0|\%|` (upstream writes `\%` in a plain string, which Python keeps as backslash-percent;
# `\\%` here is the same text).
GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""

#: simple-evals' verdict pattern (browsecomp_eval.py grade_sample), and gbench's reading of it:
#: the captured yes/no, with any capitalisation and spacing after the colon.
UPSTREAM_VERDICT_RE = re.compile(r"correct: (yes|no)")
_VERDICT_RE = re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)


def parse_verdict(reply: Optional[str]) -> Optional[str]:
    """The grader's verdict, "yes" or "no", or None when the reply has none (see the module
    docstring for how this reads upstream's pattern)."""
    match = _VERDICT_RE.search(reply or "")
    return match.group(1).lower() if match else None


def _derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the canary password (SHA-256 keystream)."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64 XOR-ciphertext using the canary-derived keystream."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _load_browsecomp_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load BrowseComp from HF Hub (smolagents/browse_comp, pinned) and canary-decrypt each row.

    Raises on load/schema/decrypt failure (no fabricated fallback).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET, split="test", revision=_REVISION)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for browsecomp: {e}")
        raise RuntimeError(f"Could not load dataset for browsecomp: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for browsecomp returned empty rows")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "browsecomp")
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("problem_topic"), seed="browsecomp")

    samples = []
    for item in rows:
        enc_problem = item.get("problem")
        enc_answer = item.get("answer")
        canary = item.get("canary")
        if not enc_problem or not enc_answer or not canary:
            raise RuntimeError(
                "browsecomp: unexpected dataset schema "
                "(missing 'problem'/'answer'/'canary'); refusing to fabricate sample data"
            )
        try:
            question = _decrypt(enc_problem, canary)
            gold = _decrypt(enc_answer, canary)
        except Exception as e:
            raise RuntimeError(
                f"browsecomp: failed to canary-decrypt a row: {e}"
            ) from e
        topic = str(item.get("problem_topic") or "General")
        messages = [{"role": "user", "content": QUERY_TEMPLATE.format(question=question)}]
        # The grader's [question] is the raw problem (upstream grade_sample(problem, ...)), not
        # the user turn above; base sends only the request fields it knows from the meta.
        samples.append((messages, gold, {"category": topic, PROBLEM_KEY: question}))

    logger.info(f"Loaded {len(samples)} browsecomp samples (canary-decrypted).")
    return samples


def grader_prompt(trace: Dict[str, Any]) -> str:
    """The grader prompt for a trace: GRADER_TEMPLATE with the raw problem, the reply and the
    decrypted gold answer."""
    problem = (trace.get("extra_payload") or {}).get(PROBLEM_KEY)
    if problem is None:
        raise RuntimeError("browsecomp: a trace has no raw problem for the grader "
                           f"({PROBLEM_KEY} missing from its sample meta)")
    return GRADER_TEMPLATE.format(question=problem, response=trace.get("response_text") or "",
                                  correct_answer=str(trace.get("gold_answer") or ""))


async def _async_judge_browsecomp(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 64,
) -> None:
    """Canonical BrowseComp LLM grader (correct: yes/no). Requires GEMINI_API_KEY."""
    from tqdm import tqdm
    import asyncio

    semaphore = asyncio.Semaphore(judge_concurrency)
    cfg = grader_config()

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

        prompt = grader_prompt(trace)
        async with semaphore:
            grade_str, judge_used, info = await judge_generate_cascade_detailed(prompt, config=cfg)
        if grade_str is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy, NOT scored
            # wrong. base.run_eval_suite drops JUDGE_OUTAGE from the pass/fail denominator
            # and reports it separately, so an outage can never masquerade as a low score.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            if info.get("max_tokens_without_text"):
                # answers of the cascade that stopped at max_output_tokens with no text
                trace["judge_max_tokens_without_text"] = int(info["max_tokens_without_text"])
            trace["status"] = "OK"
            pbar.update(1)
            return

        trace["judge_model"] = judge_used
        trace["judge_reply"] = grade_str.strip()
        trace["judge_finish_reason"] = info.get("finish_reason")
        verdict = parse_verdict(grade_str)
        if verdict is None:
            # The grader answered without a `correct:` verdict: a grader failure, not the
            # model's answer. Counted in judge_failures and left out of the score
            # (scoring_excluded); upstream would default it to "no".
            trace["is_correct"] = False
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["scoring_excluded"] = True
            trace["status"] = "FAILED"
            pbar.update(1)
            return
        trace["is_correct"] = verdict == "yes"
        trace["judge_grade"] = f"correct: {verdict}"
        trace["judge_verdict_upstream_regex"] = bool(UPSTREAM_VERDICT_RE.search(grade_str))
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [BROWSECOMP]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_browsecomp(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute browsecomp native built-in evaluation suite."""
    skip = gemini_required_skip("browsecomp", model_name)
    if skip:
        return skip
    limit = kwargs.get("limit")
    samples = simple_evals_samples(_load_browsecomp_samples(limit=limit), enable_thinking)
    # Canonical BrowseComp is CLOSED-BOOK: upstream (openai/simple-evals) declares no tools and
    # ran a single turn per question - browsing-capable models brought their own browser, others
    # answered from parametric memory. gbench matches that protocol exactly (no injected search,
    # no tool loop). BrowseComp is BUILT to defeat a non-browsing model - upstream reports 0.6%
    # for GPT-4o without browsing vs 51.5% for Deep Research - so a low number here is a faithful
    # result, not a harness bug: it is the closed-book score of the model under test.

    # One generation per sample: base runs a batch-scored suite (async_eval_fn, no
    # supports_attempts) at 1 attempt whatever --attempt-count asks.
    limit_plan = base_suite_plan("browsecomp", presharded_for_base(samples), limit,
                                 base_attempts(batch_scored=True))
    result = run_eval_suite(
        eval_name="browsecomp",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        async_eval_fn=_async_judge_browsecomp,
        thinking=enable_thinking,
        extra_payload=dict(kwargs.get("extra_payload") or {}),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=SIMPLE_EVALS_SAMPLING,   # per mode, cap included (token-limit policy)
    )
    result["eval_limit_plan"] = limit_plan
    max_tokens_reasons = record_sampler_max_tokens(result, enable_thinking)
    traces = result.get("sample_traces", []) or []
    result["metric"] = ("BrowseComp accuracy = grader 'correct: yes' / graded questions "
                        "(simple-evals is_correct; closed book, single turn)")
    result["dataset"] = f"{_DATASET}@{_REVISION} (test; = simple-evals browse_comp_test_set.csv)"
    result["judge"] = "gbench-gemini-cascade"
    # One grader call per graded question, so this is also the answered calls per model.
    result["judge_models"] = dict(Counter(t["judge_model"] for t in traces
                                          if t.get("judge_model")))
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = grader_sampling_record()
    grader_failures = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_FAILED")
    relaxed_only = sum(1 for t in traces if t.get("judge_verdict_upstream_regex") is False)
    at_cap = sum(1 for t in traces if t.get("judge_finish_reason") == "MAX_TOKENS")
    outage_at_cap = sum(1 for t in traces if t.get("judge_max_tokens_without_text"))
    result["judge_verdicts_read_only_by_relaxed_parse"] = relaxed_only
    result["judge_replies_at_max_output_tokens"] = at_cap
    result["judge_outages_after_max_output_tokens"] = outage_at_cap
    reasons = ["graded by gbench's standard Gemini cascade instead of simple-evals' "
               "gpt-4.1-2025-04-14 grader (a gbench convention)"]
    if limit:
        reasons.append(f"--eval-limit subset ({len(samples)} of {_FULL_SIZE} questions)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (simple-evals' chat sampler "
                       "samples at 0.5)")
    reasons.extend(max_tokens_reasons)
    if grader_failures:
        reasons.append(f"{grader_failures} grader repl(ies) without a 'correct: yes|no' verdict "
                       "left out of the score as grader failures (simple-evals scores such a "
                       "reply 'no')")
    if relaxed_only:
        reasons.append(f"{relaxed_only} verdict(s) read only by gbench's case- and "
                       "space-tolerant parse (simple-evals' pattern 'correct: (yes|no)' misses "
                       "them and scores 'no')")
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


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """gbench --merge: simpleqa.merge_grader_shards("browsecomp", ...) (the grader cut-off counts
    summed and named once over the union)."""
    return merge_grader_shards("browsecomp", results)
