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

"""PutnamBench informal track: natural-language answers and proofs, graded against the reference.

Dataset: informal/putnam.json of trishullab/PutnamBench at the commit putnam_formal pins
(putnam_formal.REVISION, verified against its git blob id): 673 problems, 345 with a stated
answer (`informal_solution`), the rest proofs ("None."). PutnamBench's leaderboard is formal
only, so this track is a gbench protocol:

* a gold that states a single closed-form answer (170 problems) is graded on the answer: a
  reply whose stated answer matches it after normalization is correct; any other reply goes to
  the AnswerAutoGrader prompt of IMO-AnswerBench (imo_answer_bench.GRADER_PROMPT, arXiv
  2511.01846 App. A.5), which decides whether the final answer is mathematically equivalent to
  the reference, so an equal answer written differently is never scored wrong by a string
  comparison;
* every other problem goes to the Gemini judge with the dataset's problem statement and the
  reference (when there is one), which grades the proof.

The judges run on the gbench Gemini judge cascade at gbench's deterministic judge temperature
0.0 (no upstream grader exists). Markdown around a proof verdict (`**Grade:** CORRECT`) is
removed before it is read. A judge reply without a verdict is asked again (3 calls in total)
and then JUDGE_FAILED: a grading failure, counted in judge_failures and left out of the
accuracy (scoring_excluded), never scored as the model's answer; an exhausted cascade is
JUDGE_OUTAGE, also excluded from the accuracy. Every judged trace records `judge_model`,
`judge_raw` and `judge_calls`; the result records `judge_models` and `judge_cascade`.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_PUTNAM_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .base import (run_eval_suite, judge_cascade, judge_generate_cascade, gemini_key_live_valid,
                   gemini_api_keys, strip_thinking_tags, JUDGE_TEMPERATURE)
from .imo_answer_bench import grader_prompt, parse_grader_verdict
from .metrics import record_budget_cutoffs
from .putnam_formal import DATASET, INFORMAL_BLOB, grade_line_verdict, load_putnambench
from .fc_common import base_suite_plan
from .limit_plan import base_attempts
from .sampling import stratified_indices
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

#: Total judge calls per sample before a reply without a verdict is JUDGE_FAILED.
_JUDGE_ATTEMPTS = 3
#: Problems in the pinned informal/putnam.json.
_N_PROBLEMS = 673
#: The model prompt: this head, the dataset's informal statement, then the mode's instruction.
_PROMPT_HEAD = "Putnam Competition Problem:\n"
_INSTRUCTIONS = {
    True: ("\n\nThink step by step to construct a complete mathematical proof, and state your "
           "final answer on the last line in the format: 'Final Answer: <answer>'."),
    False: ("\n\nSolve the mathematical problem and state your final answer on the last line "
            "in the format: 'Final Answer: <answer>'."),
}


def check_putnam_prerequisites() -> Tuple[bool, str]:
    """Validate prerequisites for Putnam evaluation with LLM proof judge. The key is read only
    through base.gemini_api_keys() (GEMINI_API_KEYS, else GEMINI_API_KEY), the keys the judge
    itself uses, so the gate never validates a key the judge would not send."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False, "Package 'google-genai' is not installed. Install via: pip install google-genai"

    key = ",".join(gemini_api_keys())
    if not key:
        return False, ("GEMINI_API_KEY (or GEMINI_API_KEYS) is not set. "
                       "export GEMINI_API_KEY='...'")
    _ok, _why = gemini_key_live_valid(key)
    if not _ok:
        return False, (f"The Gemini judge is not usable ({_why}); a Gemini key and judge models "
                       "the API serves are required for the Putnam judge (fail-fast, not after a "
                       "full run).")
    return True, ""


def _informal_rows() -> List[Dict[str, Any]]:
    """informal/putnam.json of the pinned PutnamBench release."""
    return load_putnambench()["informal"]


def _load_putnam_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Every problem of informal/putnam.json at the pinned PutnamBench commit."""
    raw_samples = list(_informal_rows())
    logger.info(f"Loaded {len(raw_samples)} Putnam problems from {DATASET} informal/putnam.json "
                f"(git blob {INFORMAL_BLOB[:8]}).")

    samples = []
    for item in raw_samples:
        q_text = item.get("informal_statement") or item.get("question", "")
        gold_ans = str(item.get("informal_solution") or item.get("answer", "") or "").strip()
        if gold_ans == "None.":
            gold_ans = ""        # the release marks a proof-only problem this way
        tags = item.get("tags")
        if isinstance(tags, str) and tags.startswith("["):
            import ast
            try:
                tags = ast.literal_eval(tags)
            except Exception:
                pass
        if isinstance(tags, list) and len(tags) > 0:
            category = str(tags[0]).strip("'\"")
        elif isinstance(tags, str):
            category = tags.strip("'\"[] ")
        else:
            category = str(item.get("category", "proof"))

        prompt = f"{_PROMPT_HEAD}{q_text}{_INSTRUCTIONS[bool(enable_thinking)]}"
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_ans, {"category": category}))

    if limit and 0 < limit < len(samples):
        # Stratified by first tag, not a contiguous head.
        keep = stratified_indices([m["category"] for _, _, m in samples], limit, "putnam")
        samples = [samples[i] for i in keep]
    return samples


def _problem_statement(messages: Any) -> str:
    """The dataset's informal statement inside the model prompt (without gbench's head and
    instruction), which is what the judge is shown."""
    content = messages[0].get("content", "") if messages else ""
    text = str(content or "")
    if text.startswith(_PROMPT_HEAD):
        text = text[len(_PROMPT_HEAD):]
    for instruction in _INSTRUCTIONS.values():
        if text.endswith(instruction):
            return text[:-len(instruction)]
    return text


def _balanced_groups(text: str, command: str) -> List[str]:
    """The brace-balanced argument of every `command{...}` in `text` (e.g. `\\boxed`), in order;
    an argument whose braces never close is skipped."""
    out = []
    start = 0
    needle = command + "{"
    while True:
        i = text.find(needle, start)
        if i < 0:
            return out
        depth, j = 0, i + len(command)
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[i + len(needle):j])
                    break
            j += 1
        start = i + len(needle)


def _wrap(part: str) -> str:
    """`part` as one term: parenthesized unless it is a single token."""
    return part if re.fullmatch(r"[\w.]+", part) else f"({part})"


def _normalize_putnam_math(s: str) -> str:
    """Normalize LaTeX fractions, roots, and delimiters for comparison. Nested braces are
    resolved innermost first, so `\\frac{4\\sqrt{2}-5}{3}` becomes `(4sqrt(2)-5)/3`, the form the
    release writes as `(4\\sqrt{2}-5)/3`."""
    text = s.replace("$", "").strip().lower()
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\\[dt]frac\b", r"\\frac", text)
    text = re.sub(r"\\sqrt\s*(\d+|[a-z])\b", r"sqrt(\1)", text)       # \sqrt2
    for _ in range(64):                                                # innermost first
        new = re.sub(r"\\?sqrt\{([^{}]*)\}", r"sqrt(\1)", text)
        new = re.sub(r"\\?frac\{([^{}]*)\}\{([^{}]*)\}",
                     lambda m: f"{_wrap(m.group(1))}/{_wrap(m.group(2))}", new)
        if new == text:
            break
        text = new
    text = text.replace("\\", "").strip()
    # Collapse (but keep) whitespace: word boundaries are needed to tell the `2` of an
    # answer from the `2` inside `12`. A trailing sentence period is not part of the value.
    text = re.sub(r"\s+", " ", text).strip().rstrip(".,;").strip()
    return text


def _spaceless(s: str) -> str:
    """Layout must not decide correctness: `12 - 8sqrt(2)` == `12-8sqrt(2)`."""
    return re.sub(r"\s+", "", s or "")


# Alphabetic tokens allowed inside a bare math answer; anything else means prose.
_MATH_WORDS = {"sqrt", "frac", "pi", "log", "ln", "exp", "sin", "cos", "tan", "inf",
               "infty", "mod", "max", "min", "sum", "prod", "lim", "cdot", "times", "e"}


def _gold_closed_form(gold_answer: str) -> Optional[str]:
    """The single math expression a PutnamBench `informal_solution` asserts, if it has one.

    The golds are prose that embeds the answer ("The minimum is $12 - 8\\sqrt{2}$."), not a
    bare value. Of the 345 non-empty golds in the pinned release, 168 contain exactly one
    `$...$` span, which is unambiguously the answer, and 2 more are bare values (170
    deterministic in all). The other 175 either contain no span ("Prove that there are no
    such functions.") or several (a case split, a parameterised family) and cannot be
    reduced to one value - those are for the judge, not for string comparison.

    A gold that is already a bare value (`100`, `1/2`) is returned as-is.
    """
    text = str(gold_answer or "").strip()
    if not text:
        return None
    spans = re.findall(r"\$([^$]+)\$", text)
    if len(spans) == 1:
        return spans[0].strip()
    if spans:
        return None                      # several values: a case split, not one answer
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", text)]
    if any(w not in _MATH_WORDS for w in words):
        return None                      # prose ("The limit does not exist.")
    return text


def _pred_closed_form(response_text: str) -> Tuple[str, bool]:
    """The model's stated answer and whether it was explicitly anchored.

    `\\boxed{}` (braces balanced, so a nested `\\frac{..}{..}` is kept whole) or "Final
    Answer:" means the model committed to a value, so it is held to it exactly. With no anchor
    there is only a concluding sentence to search.
    """
    text = str(response_text or "").strip()
    boxed = _balanced_groups(text, "\\boxed")
    if boxed:
        return boxed[-1].strip(), True
    anchored = re.findall(r"Final Answer\s*:?\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if anchored:
        return anchored[-1].strip(), True
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return (lines[-1] if lines else ""), False


def _states_answer(pred: str, target: str) -> bool:
    """Is `target` the answer `pred` states? Equality, or a boundary-delimited occurrence.

    The occurrence check exists only for unanchored responses. It is boundary-delimited so
    a gold of `2` is not found inside `12`; the reverse direction (the prediction being a
    substring of the reference) is never allowed, because that is what credited a bare `2`
    against "The minimum is $12 - 8\\sqrt{2}$.".
    """
    if not target:
        return False
    if _spaceless(pred) == _spaceless(target):
        return True
    for m in re.finditer(re.escape(target), pred):
        before = pred[m.start() - 1] if m.start() else " "
        after = pred[m.end()] if m.end() < len(pred) else " "
        if not (before.isalnum() or before in "._") and not (after.isalnum() or after in "._"):
            return True
    return False


def _eval_putnam(response_text: str, gold_answer: str) -> bool:
    """Does the reply state the gold's single closed-form answer (after normalization)?

    True is final. False is not a verdict for a closed-form gold: the batch scorer sends the
    reply to the answer grader, which decides equivalence (`(-5+4\\sqrt2)/3`, "The sum is 3").
    The previous comparison was bidirectional containment of the whole gold *sentence*, which
    passed any short answer that happened to be a substring of the reference prose (a `2`
    against "The minimum is $12 - 8\\sqrt{2}$."); both sides are reduced to the answer
    expression and compared for equality.
    """
    if not response_text:
        return False

    gold = str(gold_answer).strip()
    if not gold:
        # A pure-proof problem: nothing to match against, so this is not a scoring
        # authority. The judge decides.
        return False

    target = _gold_closed_form(gold)
    if target is None:
        return False        # not reducible to one value -> judge territory, fail closed

    pred, anchored = _pred_closed_form(response_text)
    # The model may wrap its value in prose ("The limit equals $1/8$."). Reduce the
    # prediction the same way the gold was reduced, so both sides are comparable.
    pred = _gold_closed_form(pred) or pred
    norm_pred = _normalize_putnam_math(pred)
    norm_gold = _normalize_putnam_math(target)
    if not norm_gold:
        return False
    if anchored:
        return _spaceless(norm_pred) == _spaceless(norm_gold)
    return _states_answer(norm_pred, norm_gold)


def parse_proof_verdict(text: Optional[str]) -> Optional[bool]:
    """The proof judge's verdict: the last `Grade: CORRECT|INCORRECT` line, or a reply that is
    only the word CORRECT or INCORRECT; None when the reply states no verdict.

    Markdown and LaTeX around the verdict (`**Grade:** CORRECT`, `Grade: **CORRECT**`,
    `**Grade**: INCORRECT`) are removed first; a line restating the format
    (`Grade: CORRECT / INCORRECT`) is not a verdict."""
    verdict = grade_line_verdict(text)
    if verdict is not None:
        return verdict
    bare = re.sub(r"[\s*_.`'\"$]+", "", str(text or "")).upper()
    if bare in ("CORRECT", "INCORRECT"):
        return bare == "CORRECT"
    return None


def _proof_prompt(problem_statement: str, gold: str, response: str) -> str:
    reference = (f"Reference Solution (the official answer):\n{gold}\n\n"
                 "The candidate need not match the reference's wording, but its "
                 "conclusion must agree with it.\n\n") if gold else ""
    return (
        "You are an expert mathematician and Putnam Competition grader.\n\n"
        f"Problem Statement:\n{problem_statement}\n\n"
        f"{reference}"
        f"Candidate Mathematical Proof:\n{response}\n\n"
        "Evaluate if the proof is mathematically rigorous, sound, and fully completes the proof.\n"
        "Respond strictly in the format:\n"
        "Grade: CORRECT / INCORRECT"
    )


async def _async_judge_putnam(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 64,
    judge_usage: Optional[Counter] = None,
) -> None:
    """Grade every sample after generation: closed-form answers by match or the answer grader,
    everything else by the proof judge. `judge_usage`, when given, counts the answered judge
    calls per model (retries included)."""
    from tqdm import tqdm
    import asyncio

    # run_putnam hard-errors up front when the judge is unavailable, so the key is guaranteed here.
    # No keyless fallback: scoring judge-territory (proof/case-split) problems deterministically
    # would fabricate is_correct=False for a missing judge (the forbidden missing-prereq->0% path).
    semaphore = asyncio.Semaphore(judge_concurrency)
    usage = judge_usage if judge_usage is not None else Counter()

    async def _ask(trace: Dict[str, Any], prompt: str, parse) -> None:
        """Up to _JUDGE_ATTEMPTS judge calls until `parse` reads a verdict; records it."""
        verdict, model, text, calls = None, None, None, 0
        for _ in range(_JUDGE_ATTEMPTS):
            async with semaphore:
                reply, used = await judge_generate_cascade(prompt)
            calls += 1
            if reply is None:
                # Judge cascade exhausted (infra outage): excluded from pass/fail accuracy
                # by base.run_eval_suite, NOT scored INCORRECT.
                trace["judge_grade"] = "JUDGE_OUTAGE"
                trace["judge_calls"] = calls
                if text is not None:
                    trace["judge_raw"], trace["judge_model"] = text, model
                return
            usage[used] += 1
            text, model = reply, used
            verdict = parse(reply)
            if verdict is not None:
                break
        trace["judge_model"] = model
        trace["judge_raw"] = text
        trace["judge_calls"] = calls
        if verdict is None:
            # The judge answered but never with a verdict: a grading failure, not the model's
            # outcome. Counted by base in judge_failures (the run reports completed_with_errors)
            # and left out of the accuracy's numerator and denominator.
            trace["is_correct"] = False
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["scoring_excluded"] = True
        else:
            trace["is_correct"] = verdict
            trace["judge_grade"] = "CORRECT" if verdict else "INCORRECT"

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        gold = str(trace.get("gold_answer") or "").strip()
        trace["status"] = "OK"

        if not resp_text:
            trace["is_correct"] = False
            if not gold or _gold_closed_form(gold) is None:
                trace["status"] = "FAILED"
            pbar.update(1)
            return

        solution = strip_thinking_tags(str(resp_text))
        problem = _problem_statement(trace.get("messages", []))
        if gold and _gold_closed_form(gold) is not None:
            # A single closed-form answer: graded on the answer. A match is final; anything
            # else is decided by the answer grader, never by the string comparison alone.
            if _eval_putnam(str(resp_text), gold):
                trace["is_correct"] = True
                trace["grading"] = "answer_match"
            else:
                trace["grading"] = "answer_grader"
                await _ask(trace, grader_prompt(problem, solution, gold), parse_grader_verdict)
            pbar.update(1)
            return

        trace["grading"] = "proof_judge"
        await _ask(trace, _proof_prompt(problem, gold, solution), parse_proof_verdict)
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [PUTNAM]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_putnam(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Putnam Mathematical Competition evaluation suite with requirements check."""
    # No-skip: the LLM judge is REQUIRED (a closed-form answer that does not match is decided
    # by the answer grader, and 503 of 673 problems are judge territory), so a missing
    # google-genai / Gemini key must HARD-ERROR, never skip and never fall back to scoring proofs
    # as wrong.
    ok, reason = check_putnam_prerequisites()
    if not ok:
        raise infra_required("putnam", reason, "docs/evals/putnam.md")

    limit = kwargs.get("limit")
    samples = _load_putnam_samples(enable_thinking=enable_thinking, limit=limit)
    # --eval-limit N is N problems for the whole suite (one generation each: a batch-judged
    # suite that does not declare supports_attempts runs 1 attempt). The plan counts the problems
    # base runs on this node (GBENCH_SHARD's shard of the drawn list), so merged shards add up.
    limit_plan = base_suite_plan("putnam", samples, limit, base_attempts(batch_scored=True))
    used: Counter = Counter()

    async def _judge(sample_traces: List[Dict[str, Any]]) -> None:
        await _async_judge_putnam(sample_traces, judge_usage=used)

    result = run_eval_suite(
        eval_name="putnam",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_putnam,
        async_eval_fn=_judge,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
    )
    result["eval_limit_plan"] = limit_plan
    traces = result.get("sample_traces") or []
    # gbench runs the INFORMAL PutnamBench track (answer match + answer grader + Gemini proof
    # judge). That is a gbench protocol, not PutnamBench's canonical metric (formal Lean 4 proof
    # verification - see the separate putnam_formal suite), so it is never comparable to the
    # published leaderboard.
    result["dataset"] = f"{DATASET} informal/putnam.json"
    result["judge"] = "gbench-gemini-cascade"
    result["judge_models"] = dict(used)   # answered judge calls per model, retries included
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = {
        "temperature": JUDGE_TEMPERATURE,
        "temperature_source": ("gbench's deterministic judge default (base.judge_config); the "
                               "informal track has no upstream grader"),
        "answer_grader": ("IMO-AnswerBench AnswerAutoGrader prompt (arXiv 2511.01846 App. A.5), "
                          "for closed-form answers that do not match the reference"),
        "proof_judge": "gbench's proof-grading prompt with the reference",
        "attempts_per_sample": _JUDGE_ATTEMPTS}
    result["grading_counts"] = dict(Counter(t.get("grading") for t in traces if t.get("grading")))
    reasons = ["informal answer check + LLM proof judge (a gbench protocol); PutnamBench's "
               "canonical metric is formal (Lean 4) proof verification - see the putnam_formal "
               "suite"]
    if limit and 0 < int(limit) < _N_PROBLEMS:
        reasons.append(f"--eval-limit {limit} runs a subset of the {_N_PROBLEMS} problems")
    if os.environ.get("GBENCH_SHARD"):
        reasons.append(f"GBENCH_SHARD {os.environ['GBENCH_SHARD']} shard")
    failed = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_FAILED")
    if failed:
        reasons.append(f"{failed} samples got no judge verdict in {_JUDGE_ATTEMPTS} calls "
                       "(JUDGE_FAILED; excluded)")
    outages = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_OUTAGE")
    if outages:
        reasons.append(f"{outages} samples were not graded (judge cascade outage; excluded)")
    # PutnamBench's informal track has no official cap: every request carries the run's
    # --max-output-tokens, and the replies it cut off are counted and named.
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
