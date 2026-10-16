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

"""putnam, putnam_formal, imo_answer_bench and cimemories: judge verdicts written with markdown
are read, a judge that never gives a verdict is a grading failure left out of the accuracy (never
scored as the model's answer), and a context the benchmark's own metric discards is not a
harness error. Gemini and the model server are stubbed."""

import asyncio
import json
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import cimemories as CIM
from gbench.runners.eval_suites import imo_answer_bench as IMO
from gbench.runners.eval_suites import putnam as P
from gbench.runners.eval_suites import putnam_formal as pf

REPO = Path(__file__).resolve().parents[1]
GOLD_PROB = "Show that the probability is $(4\\sqrt{2}-5)/3$."


@pytest.fixture(autouse=True)
def _key(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))


def _replying(text):
    async def _send(**kw):
        return base.Reply(text=text(kw) if callable(text) else text, tool_calls=None,
                          finish_reason="stop", completion_tokens=5)
    return _send


# =========================================================================== #
# Verdict parsing (putnam proof judge, putnam_formal answer review)
# =========================================================================== #
@pytest.mark.parametrize("reply,verdict", [
    ("**Grade:** CORRECT", True),
    ("Grade: **CORRECT**", True),
    ("**Grade**: INCORRECT", False),
    ("The argument is complete and rigorous.\n\n**Grade:** CORRECT", True),
    ("__Grade__: _INCORRECT_", False),
    ("`Grade: CORRECT`", True),
    ("### Grade: INCORRECT", False),
    ("Grade:\n**CORRECT**", True),
    ("Grade: \\boxed{\\text{CORRECT}}", True),
    ('Grade: "INCORRECT"', False),
    ("grade: correct.", True),
    # a judge that revises itself: the last verdict line wins
    ("Grade: CORRECT\nOn reflection step 3 fails.\n**Grade:** INCORRECT", False),
    # the format restated is not a verdict; a later verdict still counts
    ("Grade: CORRECT / INCORRECT", None),
    ("Grade: CORRECT/INCORRECT", None),
    ("Grade: [CORRECT / INCORRECT]\nGrade: INCORRECT", False),
    ("Grade: INCORRECT / the proof fails at step 3", False),
    # no verdict line
    ("I am unable to grade this.", None),
    ("THE PROOF IS CORRECT IN SPIRIT BUT INCORRECT IN DETAIL", None),
    ("Grade: CORRECTNESS is unclear", None),
    ("", None),
    (None, None),
])
def test_grade_line_verdicts_are_read_through_markdown(reply, verdict):
    assert pf.grade_line_verdict(reply) is verdict
    assert pf._review_verdict(reply) is verdict
    assert P.parse_proof_verdict(reply) is verdict


@pytest.mark.parametrize("reply,verdict", [
    ("CORRECT", True), ("**INCORRECT**", False), ("Correct.", True), ("`incorrect`", False),
    ("$CORRECT$", True), ("CORRECT, mostly", None)])
def test_putnam_proof_judge_also_accepts_a_bare_verdict_word(reply, verdict):
    assert P.parse_proof_verdict(reply) is verdict


def _pjudge(traces, replies, model="gemini-test"):
    calls = []
    it = iter(replies)

    async def cascade(prompt, *, config=None):
        calls.append(prompt)
        text = next(it)
        return (None, "JUDGE_OUTAGE") if text is None else (text, model)
    used = Counter()
    with mock.patch.object(P, "judge_generate_cascade", cascade):
        asyncio.run(P._async_judge_putnam(traces, judge_usage=used))
    return calls, used


def _ptrace(idx, statement, gold, resp, thinking=True):
    content = P._PROMPT_HEAD + statement + P._INSTRUCTIONS[thinking]
    return {"sample_idx": idx, "status": "PENDING_JUDGE", "gold_answer": gold,
            "response_text": resp, "messages": [{"role": "user", "content": content}]}


def test_putnam_markdown_verdict_is_graded_on_the_first_call():
    trace = _ptrace(0, "Prove that X.", "", "A proof.")
    calls, used = _pjudge([trace], ["The argument is complete and rigorous.\n\n**Grade:** CORRECT"])
    assert len(calls) == 1 and trace["judge_calls"] == 1
    assert trace["judge_grade"] == "CORRECT" and trace["is_correct"] is True
    assert "scoring_excluded" not in trace and used == Counter({"gemini-test": 1})
    bad = _ptrace(1, "Prove that X.", "", "A proof.")
    _pjudge([bad], ["**Grade**: INCORRECT"])
    assert bad["judge_grade"] == "INCORRECT" and bad["is_correct"] is False
    assert "scoring_excluded" not in bad


# =========================================================================== #
# A judge without a verdict is a grading failure, not the model's answer
# =========================================================================== #
def test_putnam_no_verdict_is_judge_failed_and_left_out_of_the_score():
    proof = _ptrace(0, "Prove that X.", "", "A proof.")
    calls, _ = _pjudge([proof], ["hmm", "Grade: CORRECT / INCORRECT", "no idea"])
    assert len(calls) == 3 and proof["judge_calls"] == 3
    assert proof["judge_grade"] == "JUDGE_FAILED" and proof["is_correct"] is False
    assert proof["scoring_excluded"] is True and proof["judge_raw"] == "no idea"
    # the answer grader path (AnswerAutoGrader prompt) is handled the same way
    graded = _ptrace(1, "Find p.", GOLD_PROB, "Final Answer: 1/3", thinking=False)
    _pjudge([graded], ["unsure", "still unsure", "cannot tell"])
    assert graded["grading"] == "answer_grader" and graded["judge_grade"] == "JUDGE_FAILED"
    assert graded["scoring_excluded"] is True
    correct, _ = base.tally_batch_scores([proof, graded])
    assert correct == 0 and not any(base.is_scored(t) for t in (proof, graded))


def _run_putnam(monkeypatch, proof_reply):
    samples = [([{"role": "user", "content": P._PROMPT_HEAD + "Prove that X."
                  + P._INSTRUCTIONS[False]}], "", {"category": "algebra"}),
               ([{"role": "user", "content": P._PROMPT_HEAD + "Find p." + P._INSTRUCTIONS[False]}],
                GOLD_PROB, {"category": "probability"})]

    async def cascade(prompt, *, config=None):
        if "Deterministic Mathematical Autograder" in prompt:
            return "\\boxed{Incorrect}", "gemini-3.8-flash"
        return proof_reply, "gemini-3.8-flash"
    monkeypatch.setattr(P, "check_putnam_prerequisites", lambda key=None: (True, ""))
    monkeypatch.setattr(P, "_load_putnam_samples", lambda **kw: samples)

    def text(kw):
        return ("A proof." if "Prove that X." in kw["messages"][0]["content"]
                else "Final Answer: 1/3")
    with mock.patch.object(base, "_send_single_request", _replying(text)), \
            mock.patch.object(P, "judge_generate_cascade", cascade):
        return P.run_putnam("m", "http://stub/v1", 1, limit=2)


def test_putnam_run_leaves_a_judge_failure_out_of_the_accuracy(monkeypatch):
    res = _run_putnam(monkeypatch, "no verdict")
    # one sample scored (the wrong closed-form answer); the unjudged proof is not a 0, and the
    # accuracy is over the scored sample only
    assert res["accuracy"] == 0.0 and res["scoring_excluded"] == 1
    assert res["judge_failures"] == 1 and res["status"] == "completed_with_errors"
    assert "judge failures 1" in res["status_reason"]
    assert "1 samples got no judge verdict in 3 calls (JUDGE_FAILED; excluded)" in \
        res["leaderboard_comparable_reason"]
    proof = next(t for t in res["sample_traces"] if t.get("grading") == "proof_judge")
    assert proof["scoring_excluded"] is True and proof["judge_model"] == "gemini-3.8-flash"

    res = _run_putnam(monkeypatch, "**Grade:** CORRECT")
    assert res["accuracy"] == 50.0 and res["judge_failures"] == 0
    assert res["scoring_excluded"] == 0
    assert res["status"] == "success"


def test_putnam_run_with_no_verdict_anywhere_withholds_the_score(monkeypatch):
    samples = [([{"role": "user", "content": P._PROMPT_HEAD + f"Prove {i}."
                  + P._INSTRUCTIONS[False]}], "", {"category": "algebra"}) for i in range(2)]

    async def cascade(prompt, *, config=None):
        return "I cannot decide.", "gemini-3.8-flash"
    monkeypatch.setattr(P, "check_putnam_prerequisites", lambda key=None: (True, ""))
    monkeypatch.setattr(P, "_load_putnam_samples", lambda **kw: samples)
    with mock.patch.object(base, "_send_single_request", _replying("A proof.")), \
            mock.patch.object(P, "judge_generate_cascade", cascade):
        res = P.run_putnam("m", "http://stub/v1", 1, limit=2)
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["status_reason"].startswith("nothing was scored")
    assert res["judge_failures"] == 2


def test_imo_no_verdict_is_judge_failed_and_left_out_of_the_score(monkeypatch):
    samples = [([{"role": "user", "content": f"Solve {i}"}], "3",
                {"category": "Algebra", "problem": f"P{i}"}) for i in range(2)]

    async def cascade(prompt, *, config=None):
        if "P0" in prompt:
            return "<thinking>equal</thinking>\n\\boxed{Correct}", "gemini-3.8-flash"
        return "The answer is fine.", "gemini-3.8-flash"
    with mock.patch.object(base, "_send_single_request", _replying("Final Answer: \\boxed{3}")), \
            mock.patch.object(IMO, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(IMO, "_load_imo_answer_bench_samples", lambda **k: samples), \
            mock.patch.object(IMO, "judge_generate_cascade", cascade):
        res = IMO.run_imo_answer_bench("m", "http://stub/v1")
    failed = [t for t in res["sample_traces"] if t.get("judge_grade") == "JUDGE_FAILED"]
    assert len(failed) == 8 and all(t["scoring_excluded"] and t["judge_calls"] == 3
                                    for t in failed)
    # P0's 8 attempts are all correct and P1's are unscored: 100%, not 50%
    assert res["accuracy"] == 100.0 and res["judge_failures"] == 8
    assert res["status"] == "completed_with_errors"
    assert res["attempts"]["samples"] == 1 and res["attempts"]["pass_at_k"] == 100.0
    assert "8 responses got no grader verdict in 3 calls (JUDGE_FAILED; excluded)" in \
        res["leaderboard_comparable_reason"]


def test_putnam_formal_review_without_a_verdict_is_left_out_of_the_score():
    review = {"printed": True, "type": "Nat", "answer": "1", "reference": "1"}
    stmt = ("abbrev putnam_x_solution : Nat := sorry\n-- 1\n"
            "theorem putnam_x : putnam_x_solution = 1 := sorry")
    replies = iter(["**Grade**: CORRECT", "hmm", "unsure", "no idea"])

    async def cascade(prompt, *, config=None):
        return next(replies), "gemini-test"
    ok = {"sample_idx": 0, "gold_answer": stmt}
    none = {"sample_idx": 1, "gold_answer": stmt}
    used = Counter()
    with mock.patch.object(pf, "judge_generate_cascade", cascade), \
            mock.patch.object(pf, "_review_prompt", lambda gold, rev: "prompt"):
        asyncio.run(pf._review_answer(ok, review, asyncio.Semaphore(1), used))
        asyncio.run(pf._review_answer(none, review, asyncio.Semaphore(1), used))
    assert ok["is_correct"] is True and ok["judge_grade"] == "CORRECT"
    assert "scoring_excluded" not in ok
    assert none["judge_grade"] == "JUDGE_FAILED" and none["is_correct"] is False
    assert none["scoring_excluded"] is True
    assert none["answer_review"]["outcome"] == "no verdict"
    assert used == Counter({"gemini-test": 4})
    rep = pf._settings_report([dict(t, category="lean_no_answer", source_sample_idx=i)
                               for i, t in enumerate((ok, none))])
    assert rep["lean_no_answer"]["accuracy"] == 100.0


# =========================================================================== #
# cimemories: a context metrics.py discards is the protocol's exclusion
# =========================================================================== #
PROFILE = {
    "bio": {"name": "Pat Doe", "sex": "F", "address": "1 Main St", "age": 40},
    "information_attributes": {
        "employer": {"value": "Acme", "memory_statement": "Pat works at Acme."},
        "debt": {"value": "$9,000", "memory_statement": "Pat owes $9,000."},
    },
    "contexts": [{"recipient": "Manager", "task": "Ask for a raise"},
                 {"recipient": "Doctor", "task": "Book a visit"}],
}


SYN_YAML = b"""task_solving:
  1: |
    Memories:
    {memories}
    Write to {recipient}: {task}.
adversary_reveal:
  1: |
    List revealed attributes as JSON.
    {attributes}
    SOLUTION:
    {solution}
"""
SYN_PERSONAS = '''
personas = [
    """
    **Fundamentalist**
    """,
    """
    **Pragmatic**
    """,
    """
    **Unconcerned**
    """,
]


def label_attributes(model, persona, profile, context, n=3):
    memories = "\\n".join(
        [x["memory_statement"] for x in profile["information_attributes"].values()]
    )
    attribute_sorting_prompt = f"""
    Persona: {persona}
    To {context['recipient']} for {context['task']}.

    INFORMATION:
    {{memories}}

    JSON with share and private.
    """
    attribute_sorting_prompt = (
        dedent(attribute_sorting_prompt).strip().format(memories=memories)
    )
    return model.raw(attribute_sorting_prompt, n=n)


for profile in data:
    for context in profile["contexts"]:
        for persona in personas:
            labels = label_attributes(model, persona, profile, context, n=10)
weights = {"fundamentalist": 0.26, "pragmatic": 0.64, "unconcerned": 0.10}
'''


def _cim_texts():
    return CIM.parse_release_texts(SYN_YAML, SYN_PERSONAS.encode())


def test_cimemories_discarded_context_is_excluded_by_protocol_not_a_harness_error(monkeypatch):
    def label(prompt):
        if "Manager" in prompt:
            return json.dumps({"share": ["Pat works at Acme."], "private": ["Pat owes $9,000."]})
        return json.dumps({"share": ["Pat works at Acme.", "Pat owes $9,000."], "private": []})

    async def cascade(contents, *, config=None):
        if contents.startswith("Persona:"):
            return label(contents), "gemini-3.8-flash"
        return json.dumps({"employer": "said so"}), "gemini-3.8-flash"

    async def send(**kw):
        return base.Reply(text="Dear manager, I work at Acme.", tool_calls=None,
                          finish_reason="stop")
    monkeypatch.setattr(CIM, "_release_texts", _cim_texts)
    monkeypatch.setattr(CIM, "_load_profiles", lambda: (PROFILE,))
    monkeypatch.setattr(CIM, "gemini_required_skip", lambda *a: None)
    with mock.patch.object(base, "_send_single_request", send), \
            mock.patch.object(CIM, "judge_generate_cascade", cascade):
        res = CIM.run_cimemories("m", "http://stub/v1", concurrency=1)
    discarded = [t for t in res["sample_traces"] if t.get("cimemories_context_discarded")]
    kept = [t for t in res["sample_traces"] if not t.get("cimemories_context_discarded")]
    assert len(discarded) == 5 and len(kept) == 5
    assert all(t["excluded_by_protocol"] == "no private attribute" and t["scoring_excluded"]
               for t in discarded)
    assert not any(t.get("excluded_by_protocol") for t in kept)
    # the benchmark's own discard is not a harness error: a clean run stays success
    assert res["status"] == "success" and res["status_reason"] is None
    assert res["completeness"] == 100.0 and res["effective_n"] == 5


# =========================================================================== #
# docs
# =========================================================================== #
def test_docs_state_how_a_missing_verdict_is_counted():
    doc = lambda name: (REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8")
    pu = doc("putnam")
    for needle in ("**Grade:** CORRECT", "`Grade: CORRECT / INCORRECT`", "`scoring_excluded`",
                   "not the model's\noutcome"):
        assert needle in pu, needle
    imo = doc("imo_answer_bench")
    assert "`scoring_excluded`" in imo and "counts as incorrect" not in imo
    assert "(counted in `judge_failures` and as incorrect)" not in imo
    pfd = doc("putnam_formal")
    assert "`**Grade**: INCORRECT`" in pfd and "counts as a failed\nrequest" not in pfd
    assert "`excluded_by_protocol`" in doc("cimemories")
    for name in ("putnam", "imo_answer_bench", "putnam_formal", "cimemories"):
        assert doc(name).isascii(), name
    for mod in (P, IMO, CIM):     # putnam_formal matches Lean's own symbols in its source
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert src.isascii(), mod.__name__
