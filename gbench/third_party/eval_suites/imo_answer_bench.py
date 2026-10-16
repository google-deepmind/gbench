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

"""Native IMO-AnswerBench (Google DeepMind IMO-Bench) evaluation suite.

Dataset: IMO-AnswerBench v2, `imobench/answerbench_v2.csv` from google-deepmind/superhuman at a
pinned commit (CC-BY-4.0): 400 olympiad problems with a short final answer, 100 each of
Algebra / Combinatorics / Geometry / Number theory. v2 (2026-02-12) corrected gold answers and
problem statements; v1 (`answerbench.csv`, also the `OpenEvals/IMO-AnswerBench` HF mirror) is
deprecated upstream.

Scoring follows the paper (Luong et al., EMNLP 2025, arXiv 2511.01846): every response goes to
the AnswerAutoGrader prompt of Appendix A.5, which reads the full model solution, extracts the
final answer itself and replies \\boxed{Correct} or \\boxed{Incorrect}; there is no partial credit.
Each graded trace keeps the grader's last reply (`judge_raw`), its model and the call count.
A grader that never gives a boxed verdict in 3 calls is JUDGE_FAILED: a grading failure, left
out of the accuracy (scoring_excluded) and counted, never scored as the model's answer.
The paper averages each model over 8 runs, so the suite runs at least 8 attempts per problem and
the headline is avg@k. The paper's grader is Gemini 2.5 Pro; gbench uses its Gemini judge
cascade, so the result carries leaderboard_comparable=False.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_IMO_ANSWER_BENCH_TEMPERATURE`,
which takes precedence over both. The paper publishes no grader sampling settings, so grading
runs at gbench's deterministic judge temperature 0.0 (recorded in `judge_sampling`), which none
of this affects.
"""

import asyncio
import csv
import functools
import hashlib
import io
import logging
import os
import re
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_api_keys,
                   gemini_required_skip, judge_cascade, judge_generate_cascade, strip_thinking_tags,
                   JUDGE_TEMPERATURE)
from .metrics import record_budget_cutoffs
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample, presharded_for_base

logger = logging.getLogger(__name__)

PILLAR = "Mathematics & Proofs"
DOCS_URL = "docs/evals/imo_answer_bench.md"

UPSTREAM = "github.com/google-deepmind/superhuman"
#: Last commit touching imobench/answerbench_v2.csv (2026-02-13).
REVISION = "955558ad67ef24dccd27545c326988bdad1b4471"
_CSV_URL = (f"https://raw.githubusercontent.com/google-deepmind/superhuman/{REVISION}/"
            "imobench/answerbench_v2.csv")
_CSV_SHA256 = "275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9"
_DATASET = f"{UPSTREAM}@{REVISION[:8]}:imobench/answerbench_v2.csv"
#: The paper's Table 4: "Results are averaged over 8 runs".
OFFICIAL_RUNS = 8
OFFICIAL_GRADER = "Gemini 2.5 Pro"
#: Total grader calls per response before a reply without a boxed verdict is JUDGE_FAILED.
_GRADER_ATTEMPTS = 3

#: answerbench_v2.csv at REVISION has one malformed row: imo-bench-algebra-036 lost the quote
#: that closes its Problem field, so a CSV reader folds the answer into the problem and shifts
#: every later column. This restores the quote (the v1 row is quoted this way).
_ROW_REPAIR = ('\\]\n,"$Y(x)=A+\\frac{B}{x}-x$",Algebra,Functional Equation,Iran 2002',
               '\\]\n","$Y(x)=A+\\frac{B}{x}-x$",Algebra,Functional Equation,Iran 2002')
_CATEGORIES = ("Algebra", "Combinatorics", "Geometry", "Number theory")

# Appendix A.5 "Query prompt for AnswerAutoGrader" (arXiv 2511.01846, appendix.tex), with the
# LaTeX typesetting removed and the text, numbering and the "exmaple" typo kept as printed.
# Placeholders are filled with str.replace: the text has literal braces.
GRADER_PROMPT = r"""# System Role: Deterministic Mathematical Autograder

You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the Model Solution is mathematically equivalent to the Golden Answer. You must NOT grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)

Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.

* **Algebraic Equivalence:** e.g., `n(n+1)/2` is equivalent to `n^2/2 + n/2`. You must verify the algebra.
* **Numerical Equivalence:** e.g., `1/2` is equivalent to `0.5`; `sqrt(2)/2` is equivalent to `1/sqrt(2)`.
* **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {1, 2} is equivalent to {2, 1}).
* **Partial Credit:** No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
* **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.

# 3. Output Protocol (Strict Compliance Required)

You must execute the task using a two-part structure. Failure to follow this structure will result in task failure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. Make your thinking concise. This section details your reasoning process and must follow these steps sequentially:

1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer based on the Extraction Protocol. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. Detail the steps taken to verify mathematical equivalence (e.g., simplification, algebraic manipulation). You must actively try to prove they are the same before concluding they are different.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").

**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY** the final grade.

* If Correct: \boxed{Correct}
* If Incorrect: \boxed{Incorrect}

**CRITICAL CONSTRAINT: Do not add any text, explanations, or formatting outside the <thinking> tags or the final \boxed{} output.**

Output exmaple:

<thinking>
1. **Golden Answer:** $(-\infty, -4) \cup (-4, \infty)$
2. **Extracted Model Answer:** $\emptyset$ (the empty set)
3. **Equivalence Analysis:**
The Golden Answer is a non-empty set of real numbers.
The Model Answer is the empty set.
These two sets are not equivalent. The empty set contains no elements, while the Golden Answer contains an infinite number of elements.
4. **Conclusion:** Incorrect
</thinking>
\boxed{Incorrect}

# 4. Input Data
Here is the problem, model solution, and golden answer to grade:

Problem: {Problem_Statement}
Model Solution: {Model_Solution}
Golden Answer: {Golden_Answer}"""

_VERDICT_RE = re.compile(r"\\boxed\{\s*(Correct|Incorrect)\s*\}", re.IGNORECASE)


def _cache_path() -> Path:
    root = Path(os.environ.get("GBENCH_CACHE_DIR", Path.home() / ".cache" / "gbench"))
    return root / "imo_answer_bench" / f"answerbench_v2-{REVISION[:8]}.csv"


def _fetch_csv() -> str:
    """answerbench_v2.csv at REVISION, from the local cache or GitHub, sha256-verified."""
    path = _cache_path()
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(_CSV_URL, timeout=60) as resp:
            data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != _CSV_SHA256:
        raise RuntimeError(f"answerbench_v2.csv sha256 {digest} does not match the pinned "
                           f"{_CSV_SHA256} ({'cached at ' + str(path) if path.exists() else _CSV_URL})")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    return data.decode("utf-8")


def parse_answerbench_csv(text: str) -> List[Dict[str, str]]:
    """Rows of answerbench_v2.csv with the one malformed row repaired; raises on any other damage."""
    if text.count(_ROW_REPAIR[0]) != 1:
        raise RuntimeError("answerbench_v2.csv: the known malformed row imo-bench-algebra-036 was "
                           "not found exactly once; the pinned file has changed")
    rows = list(csv.DictReader(io.StringIO(text.replace(*_ROW_REPAIR), newline="")))
    ids = [r.get("Problem ID") for r in rows]
    broken = [r.get("Problem ID") for r in rows
              if None in r or any(v is None for v in r.values()) or not r.get("Short Answer")]
    counts = Counter(r.get("Category") for r in rows)
    if broken or len(set(ids)) != len(rows) or any(counts[c] != 100 for c in _CATEGORIES) \
            or len(rows) != 400:
        raise RuntimeError(f"answerbench_v2.csv did not parse into 400 well-formed rows "
                           f"(rows={len(rows)}, categories={dict(counts)}, broken={broken[:5]})")
    return rows


def _load_imo_answer_bench_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load IMO-AnswerBench v2 from the pinned upstream commit; raises rather than substituting."""
    try:
        rows = parse_answerbench_csv(_fetch_csv())
    except Exception as e:
        logger.error("Failed to load IMO-AnswerBench v2: %s", e)
        raise RuntimeError(f"Could not load dataset for imo_answer_bench: {e}") from e

    if not rows:
        raise RuntimeError("imo_answer_bench: dataset returned no rows")
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "imo_answer_bench")
    # Stratified across the four categories, not a contiguous head:
    # the 400 problems are stored grouped, so a head was 20 Algebra questions.
    rows = stratified_sample(rows, limit, lambda r: r.get("Category"), seed="imo_answer_bench")
    logger.info("Loaded %d IMO-AnswerBench v2 problems from %s.", len(rows), _DATASET)

    samples = []
    for item in rows:
        problem = str(item.get("Problem") or "").strip()
        gold = str(item.get("Short Answer") or "").strip()
        # The paper does not publish its solver prompt; the grader extracts the answer from
        # the whole solution, so the instruction only asks for a clearly marked final answer.
        prompt = f"Solve the following olympiad problem:\n\n{problem}\n\n"
        if enable_thinking:
            prompt += ("Reason step by step, then present the final exact answer in the "
                       "format: 'Final Answer: \\boxed{answer}'.")
        else:
            prompt += ("Present the final exact answer in the format: "
                       "'Final Answer: \\boxed{answer}'.")
        samples.append(([{"role": "user", "content": prompt}], gold, {
            "category": str(item.get("Category") or "unknown"),
            "subcategory": str(item.get("Subcategory") or ""),
            "problem_id": str(item.get("Problem ID") or ""),
            # The grader sees the dataset problem, not gbench's wrapper around it.
            "problem": problem,
        }))
    return samples


def grader_prompt(problem: str, solution: str, gold: str) -> str:
    return (GRADER_PROMPT.replace("{Problem_Statement}", problem)
            .replace("{Model_Solution}", solution).replace("{Golden_Answer}", gold))


def parse_grader_verdict(text: Optional[str]) -> Optional[bool]:
    """The last \\boxed{Correct|Incorrect} in the grader's reply, or None if there is none."""
    found = _VERDICT_RE.findall(str(text or ""))
    return found[-1].lower() == "correct" if found else None


async def _async_grade_imo(sample_traces: List[Dict[str, Any]],
                           judge_concurrency: int = 32,
                           judge_usage: Optional[Counter] = None) -> None:
    """AnswerAutoGrader over the full response of every sample.

    `judge_usage`, when given, counts the answered grader calls per model (retries included).
    """
    from tqdm import tqdm

    if not gemini_api_keys():
        # run_imo_answer_bench's gemini_required_skip gate already hard-errors on an absent or
        # invalid key; never downgrade to textual matching.
        from .swebench_common import infra_required
        raise infra_required(
            "imo_answer_bench",
            "requires GEMINI_API_KEY (or GEMINI_API_KEYS) for AnswerAutoGrader grading",
            DOCS_URL)
    usage = judge_usage if judge_usage is not None else Counter()

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _grade(trace: Dict[str, Any], pbar: tqdm) -> None:
        solution = strip_thinking_tags(trace.get("response_text"))
        # Settled before grading, so a judge outage below does not leave PENDING_JUDGE behind.
        if trace.get("status") == "PENDING_JUDGE":   # keep EMPTY_RESPONSE etc.
            trace["status"] = "OK"
        if not solution:
            trace["is_correct"] = False
            trace["judge_grade"] = "no_response"
            pbar.update(1)
            return
        meta = trace.get("extra_payload") or {}
        prompt = grader_prompt(str(meta.get("problem") or ""), solution,
                               str(trace.get("gold_answer") or ""))
        verdict, model, reply, calls = None, None, None, 0
        for _ in range(_GRADER_ATTEMPTS):
            async with semaphore:
                text, used = await judge_generate_cascade(prompt)
            calls += 1
            if text is None:
                # Judge cascade exhausted (infra outage): excluded from accuracy by
                # base.run_eval_suite, NOT scored wrong. An earlier reply without a verdict
                # is kept for the record.
                trace["judge_grade"] = "JUDGE_OUTAGE"
                trace["judge_calls"] = calls
                if reply is not None:
                    trace["judge_model"], trace["judge_raw"] = model, reply
                pbar.update(1)
                return
            model, reply = used, text
            usage[model] += 1
            verdict = parse_grader_verdict(text)
            if verdict is not None:
                break
        trace["judge_model"] = model
        # The grader's last reply (its <thinking> analysis and boxed verdict), so the verdict
        # can be audited and parsed again from the stored result.
        trace["judge_raw"] = reply
        trace["judge_calls"] = calls
        if verdict is None:
            # The grader answered but never with a boxed verdict: a grading failure, not the
            # model's outcome. Counted by base in judge_failures (the run reports
            # completed_with_errors) and left out of the accuracy's numerator and denominator.
            trace["is_correct"] = False
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["scoring_excluded"] = True
        else:
            trace["is_correct"] = verdict
            trace["judge_grade"] = "Correct" if verdict else "Incorrect"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Grading [IMO_ANSWER_BENCH]") as pbar:
        await asyncio.gather(*[_grade(t, pbar) for t in sample_traces])


def run_imo_answer_bench(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run IMO-AnswerBench v2 (400 problems, avg@8) under the AnswerAutoGrader prompt."""
    # Fail-fast: the grader is load-bearing. A missing/invalid key must hard-error BEFORE the
    # generation run, never downgrade to textual-only scoring.
    gemini_required_skip("imo_answer_bench", model_name)
    limit = kwargs.get("limit")
    samples = _load_imo_answer_bench_samples(enable_thinking=enable_thinking, limit=limit)
    # The paper averages 8 runs; a larger explicit --attempt-count is honoured.
    attempts = max(OFFICIAL_RUNS, int(kwargs.get("attempt_count") or 1))
    # --eval-limit N is N problems for the whole suite (stratified over the four categories);
    # the paper's 8 runs per problem are attempts, not more problems.
    limit_plan = eval_limit_plan("imo_answer_bench", limit, len(samples), attempts=attempts)
    used: Counter = Counter()
    result = run_eval_suite(
        eval_name="imo_answer_bench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's problems, kept as is
        async_eval_fn=functools.partial(_async_grade_imo, judge_usage=used),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        attempt_count=attempts,
        supports_attempts=True,   # every attempt is graded on its own
    )
    result["eval_limit_plan"] = limit_plan
    result["dataset"] = _DATASET
    result["metric"] = (f"accuracy averaged over {attempts} runs, graded with the AnswerAutoGrader "
                        "prompt (no partial credit)")
    result["judge"] = "gbench-gemini-cascade"
    result["judge_models"] = dict(used)   # answered grader calls per model
    result["judge_cascade"] = judge_cascade()
    # The paper names the grader model and prompt only; no sampling settings are published.
    result["judge_sampling"] = {
        "reference": ("arXiv 2511.01846 Appendix A.5 AnswerAutoGrader prompt, graded by "
                      f"{OFFICIAL_GRADER}; the paper publishes no grader sampling settings"),
        "temperature": JUDGE_TEMPERATURE,
        "temperature_source": "gbench's deterministic judge default (base.judge_config)"}
    reasons = [f"graded by the gbench Gemini judge cascade "
               f"({', '.join(sorted(used)) or 'no judge call answered'}) with the AnswerAutoGrader "
               f"prompt, not the paper's {OFFICIAL_GRADER}"]
    if result.get("temperature") == 0.0:
        reasons.append(f"the {attempts} runs are greedy (temperature 0.0), so they differ only "
                       "through server nondeterminism")
    if limit:
        reasons.append(f"--eval-limit {limit} runs a subset of the 400 problems")
    failed = sum(1 for t in (result.get("sample_traces") or [])
                 if t.get("judge_grade") == "JUDGE_FAILED")
    if failed:
        reasons.append(f"{failed} responses got no grader verdict in {_GRADER_ATTEMPTS} calls "
                       "(JUDGE_FAILED; excluded)")
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
