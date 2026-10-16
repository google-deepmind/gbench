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

"""Judged suites cimemories, charxiv, putnam, putnam_formal and imo_answer_bench: upstream call
parameters, batching and retries, recorded judge models and replies, withheld scores, and the
headline counts. Docker, Gemini and the model server are stubbed."""

import asyncio
import base64
import json
import re
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import charxiv as C
from gbench.runners.eval_suites import cimemories as CIM
from gbench.runners.eval_suites import imo_answer_bench as IMO
from gbench.runners.eval_suites import putnam as P
from gbench.runners.eval_suites import putnam_formal as pf

REPO = Path(__file__).resolve().parents[1]
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@pytest.fixture(autouse=True)
def _key(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))


def _replying(seen, text):
    async def _send(**kw):
        seen.append(kw)
        return base.Reply(text=text(kw) if callable(text) else text, tool_calls=None,
                          finish_reason="stop", completion_tokens=5)
    return _send


# =========================================================================== #
# charxiv
# =========================================================================== #
def _dtrace(idx, qid, resp="resp", answer="ans"):
    return {"sample_idx": idx, "category": "descriptive", "response_text": f"{resp}{idx}",
            "gold_answer": json.dumps({"track": "descriptive", "qid": qid,
                                       "answer": f"{answer}{idx}"})}


def _rtrace(idx, question="Q"):
    return {"sample_idx": idx, "category": "reasoning", "response_text": f"r{idx}",
            "gold_answer": json.dumps({"track": "reasoning", "inst_category": 2,
                                       "raw_question": question, "answer": "B"})}


def _grade_all(prompt):
    """A judge that scores every T{i} of a descriptive prompt 1 and a reasoning prompt 1."""
    found = re.search(r"You will be given (\d+) pairs", prompt)
    n = int(found.group(1)) if found else 0
    if n:
        out = {}
        for i in range(1, n + 1):
            out[f"extract_answer_T{i}"] = f"x{i}"
            out[f"score_T{i}"] = 1 if i % 2 else 0
        return json.dumps(out)
    return json.dumps({"extracted_answer": "y", "score": 1})


def _judge(traces, reply=_grade_all, model="gemini-test"):
    calls = []

    async def cascade(prompt, *, config=None):
        calls.append(prompt)
        text = reply(prompt) if callable(reply) else reply
        return (None, "JUDGE_OUTAGE") if text is None else (text, model)
    used = Counter()
    with mock.patch.object(C, "judge_generate_cascade", cascade):
        asyncio.run(C._async_judge_charxiv(traces, judge_usage=used))
    return calls, used


def test_charxiv_descriptive_answers_are_graded_five_per_prompt_grouped_by_template():
    # 7 answers to template 3 and 2 to template 1, interleaved as charts are in the dataset
    qids = [3, 1, 3, 3, 3, 1, 3, 3, 3]
    traces = [_dtrace(i, q) for i, q in enumerate(qids)]
    calls, used = _judge(list(reversed(traces)))          # the order they arrive does not matter
    # upstream's order: template 1 (2 answers) first, then template 3 in chunks of 5 and 2
    assert len(calls) == 3
    sizes = [int(re.search(r"You will be given (\d+) pairs", c).group(1)) for c in calls]
    assert sizes == [2, 5, 2]
    first, second = calls[0], calls[1]
    assert C.DESCRIPTIVE_GRADING_QMAP[1] in first and C.DESCRIPTIVE_GRADING_QMAP[3] in second
    assert str(["extract_answer_T1", "score_T1", "extract_answer_T2", "score_T2"]) in first
    assert first.endswith("T1:\nResponse 1: resp1\nGround Truth 1: ans1\n\n"
                          "T2:\nResponse 2: resp5\nGround Truth 2: ans5\n\n")
    assert "Response 5: resp6\n" in second and "Response 1: resp0\n" in second
    # score_T{i} goes back to the i-th answer of its prompt
    by_idx = {t["sample_idx"]: t for t in traces}
    assert by_idx[1]["is_correct"] is True and by_idx[5]["is_correct"] is False
    assert by_idx[5]["judge_batch"] == {"template": 1, "position": 2, "size": 2, "calls": 1}
    assert by_idx[0]["extracted_answer"] == "x1" and by_idx[2]["extracted_answer"] == "x2"
    assert all(t["judge_model"] == "gemini-test" for t in traces)
    assert all(json.loads(t["judge_raw"]) for t in traces)
    assert used == Counter({"gemini-test": 3})


def test_charxiv_prompt_for_one_answer_is_the_single_triplet_render():
    prompt = C._build_descriptive_grading_prompt(8, [("k", "R", "G")])
    assert "You will be given 1 pairs" in prompt
    assert str(["extract_answer_T1", "score_T1"]) in prompt
    assert prompt.endswith("T1:\nResponse 1: R\nGround Truth 1: G\n\n")
    assert C.build_json_keys(2) == str(["extract_answer_T1", "score_T1", "extract_answer_T2",
                                        "score_T2"])


def test_charxiv_retries_an_invalid_reply_up_to_ten_times_like_upstream():
    replies = iter(["not json", '{"extract_answer_T1": "a"}',      # missing score_T1
                    '{"extract_answer_T1": "a", "score_T1": "1"}',  # score not in [0, 1]
                    '{"extract_answer_T1": "a", "score_T1": 1}'])
    trace = _dtrace(0, 2)
    calls, _ = _judge([trace], reply=lambda p: next(replies))
    assert len(calls) == 4 and trace["judge_grade"] == "CORRECT"
    assert trace["judge_batch"]["calls"] == 4

    trace = _dtrace(0, 2)
    calls, _ = _judge([trace], reply="```json\n{}\n```")
    assert len(calls) == C.JUDGE_MAX_ATTEMPTS == 10
    assert trace["judge_grade"] == "JUDGE_FAILED" and trace["is_correct"] is False
    assert trace["judge_score"] == -1 and trace["judge_raw"] == "```json\n{}\n```"


def test_charxiv_reasoning_invalid_score_is_not_retried_and_outage_is_excluded():
    bad = _rtrace(0)
    calls, _ = _judge([bad], reply='{"extracted_answer": "z", "score": 2}')
    # upstream accepts the reply (only json.loads and the keys are checked) and get_stats scores
    # the out-of-range score 0
    assert len(calls) == 1 and bad["judge_grade"] == "JUDGE_FAILED" and bad["judge_score"] == 2
    down = [_dtrace(0, 4), _dtrace(1, 4)]
    _judge(down, reply=None)
    assert all(t["judge_grade"] == "JUDGE_OUTAGE" and "is_correct" not in t for t in down)


def test_charxiv_request_is_text_then_image_with_the_mime_of_the_bytes():
    for raw, mime in ((JPEG, "image/jpeg"), (PNG, "image/png")):
        b64 = base64.b64encode(raw).decode()
        (messages, gold, meta) = C._mk_sample(b64, "What is the title?", {"track": "x"}, "x")
        content = messages[0]["content"]
        assert [p["type"] for p in content] == ["text", "image_url"]
        assert content[0]["text"] == "What is the title?"
        assert content[1]["image_url"]["url"] == f"data:{mime};base64,{b64}"
        assert meta == {"category": "x"}


def test_charxiv_run_records_the_judge_and_reports_the_headline_track_counts(monkeypatch):
    samples = []
    for i, q in enumerate((1, 1, 18)):
        samples.append(([{"role": "user", "content": [{"type": "text", "text": f"d{i}"}]}],
                        json.dumps({"track": "descriptive", "qid": q, "answer": "a"}),
                        {"category": "descriptive"}))
    for i in range(2):
        samples.append(([{"role": "user", "content": [{"type": "text", "text": f"r{i}"}]}],
                        json.dumps({"track": "reasoning", "inst_category": 1,
                                    "raw_question": f"q{i}", "answer": "3"}),
                        {"category": "reasoning"}))

    def reply(prompt):
        if "q1" in prompt:
            return json.dumps({"extracted_answer": "4", "score": 0})
        return _grade_all(prompt)
    seen = []

    async def cascade(prompt, *, config=None):
        return reply(prompt), "gemini-3.8-flash"
    with mock.patch.object(base, "_send_single_request", _replying(seen, "answer")), \
            mock.patch.object(C, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(C, "_load_charxiv_samples", lambda limit=None: samples), \
            mock.patch.object(C, "judge_generate_cascade", cascade):
        res = C.run_charxiv("m", "http://stub/v1", concurrency=1)
    assert res["charxiv_reasoning_accuracy"] == 50.0
    assert res["charxiv_descriptive_accuracy"] == pytest.approx(66.67)
    assert res["accuracy"] == 50.0 and res["headline_track"] == "reasoning"
    assert (res["total_questions"], res["correct_answers"], res["effective_n"]) == (2, 1, 2)
    assert (res["questions_all_tracks"], res["correct_all_tracks"]) == (5, 3)
    assert res["judge"] == "gbench-gemini-cascade" and res["judge_cascade"] == base.judge_cascade()
    # 2 descriptive prompts (template 1 with two answers, template 18) + 2 reasoning prompts
    assert res["judge_models"] == {"gemini-3.8-flash": 4}
    assert all(t["judge_model"] == "gemini-3.8-flash" and t["judge_raw"]
               for t in res["sample_traces"])
    assert "max_tokens (256" in res["leaderboard_comparable_reason"]
    assert res["judge_sampling"]["descriptive_answers_per_prompt"] == 5
    assert res["judge_sampling"]["max_attempts_per_prompt"] == 10


def test_charxiv_source_has_no_second_default_judge():
    src = Path(C.__file__).read_text(encoding="utf-8")
    assert "DEFAULT_JUDGE_MODEL" not in src and src.isascii()


# =========================================================================== #
# putnam (informal)
# =========================================================================== #
GOLD_PROB = "Show that the probability is $(4\\sqrt{2}-5)/3$."


@pytest.mark.parametrize("reply", [
    "Final Answer: $\\frac{4\\sqrt{2}-5}{3}$",
    "Final Answer: \\boxed{\\frac{4\\sqrt{2}-5}{3}}",
    "Final Answer: (4\\sqrt2-5)/3",
    "Final Answer: \\frac{4\\sqrt{2}-5}{3}",
    "Final Answer: \\dfrac{4\\sqrt{2}-5}{3}",
])
def test_putnam_nested_fractions_and_boxed_answers_match(reply):
    assert P._eval_putnam(reply, GOLD_PROB) is True


def test_putnam_existing_string_rules_still_hold():
    gold = r"The minimum is $12 - 8\sqrt{2}$."
    assert P._eval_putnam("Final Answer: 2", gold) is False
    assert P._eval_putnam(r"Final Answer: 12-8\sqrt{2}", gold) is True
    assert P._eval_putnam("Final Answer: 1/8", r"The limit equals $\frac{1}{8}$.") is True
    assert P._eval_putnam("Final Answer: 1/9", r"The limit equals $\frac{1}{8}$.") is False


def _ptrace(idx, content, gold, resp):
    return {"sample_idx": idx, "status": "PENDING_JUDGE", "gold_answer": gold,
            "response_text": resp, "messages": [{"role": "user", "content": content}]}


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


def test_putnam_an_equal_answer_written_differently_goes_to_the_answer_grader():
    content = P._PROMPT_HEAD + "Find the probability." + P._INSTRUCTIONS[False]
    trace = _ptrace(0, content, GOLD_PROB, "work...\nFinal Answer: (-5+4\\sqrt{2})/3")
    calls, used = _pjudge([trace], ["<thinking>equal</thinking>\n\\boxed{Correct}"])
    assert calls == [IMO.grader_prompt("Find the probability.",
                                       "work...\nFinal Answer: (-5+4\\sqrt{2})/3", GOLD_PROB)]
    assert trace["is_correct"] is True and trace["grading"] == "answer_grader"
    assert trace["judge_model"] == "gemini-test" and "boxed{Correct}" in trace["judge_raw"]
    assert used == Counter({"gemini-test": 1})

    match = _ptrace(1, content, "$3$", "Final Answer: 3")
    calls, _ = _pjudge([match], [])
    assert calls == [] and match["is_correct"] is True and match["grading"] == "answer_match"


def test_putnam_proof_judge_sees_the_dataset_statement_only_and_no_verdict_is_judge_failed():
    content = P._PROMPT_HEAD + "Prove that X." + P._INSTRUCTIONS[True]
    trace = _ptrace(0, content, "", "A proof.")
    calls, _ = _pjudge([trace], ["I am unable to grade this.", "", "THE PROOF IS CORRECT IN "
                                 "SPIRIT BUT INCORRECT IN DETAIL"])
    assert len(calls) == 3
    assert "Problem Statement:\nProve that X.\n\n" in calls[0]
    assert "Think step by step" not in calls[0] and "Putnam Competition Problem" not in calls[0]
    assert trace["judge_grade"] == "JUDGE_FAILED" and trace["is_correct"] is False
    assert trace["judge_calls"] == 3

    ok = _ptrace(1, content, "", "A proof.")
    _pjudge([ok], ["analysis\nGrade: CORRECT"])
    assert ok["judge_grade"] == "CORRECT" and ok["is_correct"] is True
    out = _ptrace(2, content, "", "A proof.")
    _pjudge([out], [None])
    assert out["judge_grade"] == "JUDGE_OUTAGE" and "is_correct" not in out


def test_putnam_run_counts_judge_failures_and_records_the_judge(monkeypatch):
    content0 = "Prove that X."
    samples = [([{"role": "user", "content": P._PROMPT_HEAD + content0 + P._INSTRUCTIONS[False]}],
                "", {"category": "algebra"}),
               ([{"role": "user", "content": P._PROMPT_HEAD + "Find p." + P._INSTRUCTIONS[False]}],
                GOLD_PROB, {"category": "probability"})]
    async def cascade(prompt, *, config=None):
        # keyed on the prompt, not on call order: the two samples are judged concurrently
        if "Deterministic Mathematical Autograder" in prompt:
            return "\\boxed{Incorrect}", "gemini-3.8-flash"
        return "no verdict", "gemini-3.8-flash"
    monkeypatch.setattr(P, "check_putnam_prerequisites", lambda key=None: (True, ""))
    monkeypatch.setattr(P, "_load_putnam_samples", lambda **kw: samples)
    seen = []

    def text(kw):
        return "A proof." if content0 in kw["messages"][0]["content"] else "Final Answer: 1/3"
    with mock.patch.object(base, "_send_single_request", _replying(seen, text)), \
            mock.patch.object(P, "judge_generate_cascade", cascade):
        res = P.run_putnam("m", "http://stub/v1", 1, limit=2)
    assert res["judge_failures"] == 1 and res["status"] == "completed_with_errors"
    assert res["judge"] == "gbench-gemini-cascade" and res["judge_cascade"] == base.judge_cascade()
    assert res["judge_models"] == {"gemini-3.8-flash": 4}
    assert res["grading_counts"] == {"proof_judge": 1, "answer_grader": 1}
    assert all(t["judge_model"] == "gemini-3.8-flash" for t in res["sample_traces"])
    reason = res["leaderboard_comparable_reason"]
    assert "--eval-limit 2 runs a subset of the 673 problems" in reason
    assert "1 samples got no judge verdict" in reason
    assert res["judge_sampling"]["temperature"] == base.JUDGE_TEMPERATURE


def test_putnam_source_has_no_second_default_judge():
    src = Path(P.__file__).read_text(encoding="utf-8")
    assert "DEFAULT_JUDGE_MODEL" not in src and "judge_model: str" not in src and src.isascii()


# =========================================================================== #
# imo_answer_bench
# =========================================================================== #
def test_imo_keeps_the_graders_reply():
    trace = {"status": "PENDING_JUDGE", "response_text": "So 3.\nFinal Answer: \\boxed{3}",
             "gold_answer": "3", "extra_payload": {"problem": "Find n."}}
    replies = iter(["no verdict here", "<thinking>1. ... Correct</thinking>\n\\boxed{Correct}"])

    async def cascade(prompt, *, config=None):
        return next(replies), "gemini-3.8-flash"
    with mock.patch.object(IMO, "judge_generate_cascade", cascade):
        asyncio.run(IMO._async_grade_imo([trace]))
    assert trace["judge_grade"] == "Correct" and trace["judge_calls"] == 2
    assert trace["judge_raw"].endswith("\\boxed{Correct}") and "<thinking>" in trace["judge_raw"]
    assert IMO.parse_grader_verdict(trace["judge_raw"]) is True
    assert trace["judge_model"] == "gemini-3.8-flash"


# =========================================================================== #
# putnam_formal
# =========================================================================== #
T = "`" * 3


def test_lean_fences_are_paired_when_a_block_is_abandoned():
    final = ("abbrev putnam_1990_a4_solution : Nat := 2\n"
             "theorem putnam_1990_a4 : True := sorry")
    reply = (f"{T}lean\n/-- a long analysis that never closes its comment\n" + "prose\n" * 50
             + f"{T}lean\n{final}\n{T}\n")
    assert pf.lean_blocks(reply) == [final]
    assert pf.lean_blocks(f"{T}lean\nfirst\n{T}\ntext\n{T}lean4\nsecond\n{T}") == ["first",
                                                                                 "second"]
    assert pf.lean_blocks(f"{T}python\nx = 1\n{T}\n{T}lean\ny\n{T}") == ["y"]
    assert pf.lean_blocks(f"Here: {T}lean\ninline{T}") == ["inline"]          # inline fences
    assert pf.lean_blocks(f"{T}lean\nno closing fence") == []
    # a longer opening fence is closed only by a fence at least as long
    assert pf.lean_blocks(f"````lean\ns := \"x\"\n{T}\nmore\n````") == [f's := "x"\n{T}\nmore']


def test_verify_compiles_the_final_block(monkeypatch, tmp_path):
    seen = {}
    ref = tmp_path / "Reference.olean"
    ref.write_bytes(b"x")
    monkeypatch.setattr(pf, "_split_statement", lambda stmt: ("putnam_x", "", "", None))
    monkeypatch.setattr(pf, "_reference_olean", lambda stmt: ref)

    def capture(code, stmt):
        seen["code"] = code
        raise pf._HarnessError("stop here")
    monkeypatch.setattr(pf, "_solution_source", capture)
    reply = f"{T}lean\n/-- unclosed\nprose\n{T}lean\ntheorem putnam_x : True := trivial\n{T}"
    ok, why = pf._verify(reply, "theorem putnam_x : True := sorry")
    assert (ok, why) == (False, "harness error: stop here")
    assert seen["code"] == "theorem putnam_x : True := trivial"


def test_harness_errors_are_left_out_of_the_accuracy(monkeypatch):
    verdicts = iter([(False, "harness error: docker"), (True, "ok"), (False, "harness error: x")])
    monkeypatch.setattr(pf, "_verify", lambda r, s: next(verdicts))
    traces = [{"sample_idx": i, "status": "PENDING_JUDGE", "response_text": "r",
               "gold_answer": "theorem t : True := sorry", "category": "lean_with_answer"}
              for i in range(3)]
    stats = {}
    asyncio.run(pf._make_scorer(1, stats)(traces))
    assert stats["verification_harness_errors"] == 2
    assert [t.get("scoring_excluded", False) for t in traces] == [True, False, True]
    assert traces[0]["judge_grade"] == "JUDGE_FAILED"
    correct, _ = base.tally_batch_scores(traces)
    assert correct == 1


def test_putnam_formal_headline_counts_describe_the_with_answer_setting(monkeypatch):
    traces = ([{"category": "lean_with_answer", "source_sample_idx": i, "is_correct": i == 0,
                "gold_answer": "theorem t : True := sorry", "verification": "ok"}
               for i in range(2)]
              + [{"category": "lean_no_answer", "source_sample_idx": i, "is_correct": False,
                  "gold_answer": "theorem t : True := sorry", "verification": "x"}
                 for i in range(2)])
    monkeypatch.setattr(pf, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(pf, "_load_putnam_formal_samples", lambda limit=None: [])
    monkeypatch.setattr(pf, "_lean_image_provenance", lambda: {})
    monkeypatch.setattr(pf, "run_eval_suite", lambda **kw: {
        "sample_traces": traces, "total_questions": 4, "correct_answers": 1})
    out = pf.run_putnam_formal("m", "http://stub/v1", 1)
    assert out["accuracy"] == 50.0
    assert (out["total_questions"], out["correct_answers"], out["effective_n"]) == (2, 1, 2)
    assert (out["total_questions_all_settings"], out["correct_answers_all_settings"]) == (4, 1)


def test_putnam_formal_setting_without_a_scored_attempt_has_no_accuracy():
    rep = pf._settings_report([{"category": "lean_with_answer", "source_sample_idx": 0,
                                "is_correct": False, "scoring_excluded": True,
                                "gold_answer": "theorem t : True := sorry"}])
    assert rep["lean_with_answer"]["accuracy"] is None
    assert rep["lean_no_answer"]["accuracy"] is None


def test_putnam_formal_says_why_it_keeps_the_house_budget():
    src = Path(pf.__file__).read_text(encoding="utf-8")
    doc = (REPO / "docs" / "evals" / "putnam_formal.md").read_text(encoding="utf-8")
    assert "PutnamBench sets no max_tokens" not in src and "sets no `max_tokens`" not in doc
    assert "publishes no generation harness" in src and "publishes no generation harness" in doc


# =========================================================================== #
# cimemories
# =========================================================================== #
PROFILE = {
    "bio": {"name": "Pat Doe", "sex": "F", "address": "1 Main St", "age": 40},
    "information_attributes": {
        "employer": {"value": "Acme", "memory_statement": "Pat works at Acme."},
        "debt": {"value": "$9,000", "memory_statement": "Pat owes $9,000."},
        "clinic_visit": {"value": "May 2", "memory_statement": "Pat visited a clinic on May 2."},
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
SYN = CIM.parse_release_texts(SYN_YAML, SYN_PERSONAS.encode())


class _Labeler:
    """Labeler and REVEAL judge stand-in. `label(prompt, i)` scripts the labeler."""

    def __init__(self, label, model=lambda i: "gemini-3.8-flash"):
        self.label, self.model, self.configs, self.counts = label, model, [], Counter()

    async def __call__(self, contents, *, config=None):
        if contents.startswith("Persona:"):
            i = self.counts[contents]
            self.counts[contents] += 1
            self.configs.append(config)
            return self.label(contents, i), self.model(i)
        return json.dumps({"employer": "said so"}), "gemini-3.8-flash"


def _run_cim(monkeypatch, labeler):
    counter = Counter()

    async def fake_send(**kw):
        counter[kw["messages"][0]["content"]] += 1
        return base.Reply(text="Dear manager, I work at Acme.", tool_calls=None,
                          finish_reason="stop")
    monkeypatch.setattr(CIM, "_release_texts", lambda: SYN)
    monkeypatch.setattr(CIM, "_load_profiles", lambda: (PROFILE,))
    monkeypatch.setattr(CIM, "gemini_required_skip", lambda *a: None)
    with mock.patch.object(base, "_send_single_request", fake_send), \
            mock.patch.object(CIM, "judge_generate_cascade", labeler):
        return CIM.run_cimemories("m", "http://stub/v1", concurrency=1)


def test_cimemories_labeler_sends_upstreams_parameters():
    cfg = CIM._labeler_config()
    assert cfg.temperature is None
    assert cfg.thinking_config.thinking_level.value == "LOW"
    ident = CIM._labeler_identity()
    assert ident["temperature"] is None and ident["thinking_level"] == "low"
    assert ident["upstream_reasoning_effort"] == "minimal" and ident["cascade"] == \
        base.judge_cascade()
    assert not hasattr(CIM, "_LABELER_TEMPERATURE")


def test_cimemories_every_context_discarded_withholds_the_score(monkeypatch):
    # the labels never agree on a private attribute: upstream discards every context
    def label(prompt, i):
        return json.dumps({"share": ["Pat works at Acme."],
                           "private": ["Pat owes $9,000."] if i else []})
    labeler = _Labeler(label)
    res = _run_cim(monkeypatch, labeler)
    assert res["status"] == "failed"
    assert res["accuracy"] is None and res["completeness"] is None
    assert res["violation_at_5"] is None and res["effective_n"] == 0
    why = res["accuracy_withheld_reason"]
    assert "every context (2 of 2) was discarded" in why and "labeler outcome" in why
    assert res["status_reason"].startswith("nothing was scored: every context")
    assert res["attempts"]["avg_at_k"] is None and res["attempts"]["samples"] == 0
    # upstream's parameters on every labeler call
    assert {c.temperature for c in labeler.configs} == {None}
    reason = res["leaderboard_comparable_reason"]
    assert "keep 0 of 2 contexts" in reason and "45.7 of 49" in reason


def test_cimemories_labeler_diagnostics_and_mixed_models_are_reported(monkeypatch):
    def label(prompt, i):
        if i == 9:      # reworded statements (period dropped): parses, matches nothing
            return json.dumps({"share": ["Pat works at Acme"], "private": ["Pat owes $9,000"]})
        return json.dumps({"share": ["Pat works at Acme."],
                           "private": ["Pat owes $9,000.", "Pat visited a clinic on May 2."]})
    labeler = _Labeler(label, model=lambda i: "gemini-3.6-flash" if i == 0 else "gemini-3.8-flash")
    res = _run_cim(monkeypatch, labeler)
    rep = res["cimemories_report"]
    assert rep["labeler_replies_matching_no_statement"] == 6        # 2 contexts x 3 personas
    diag = res["cimemories_labels"]["p0c0"]["labeler_diagnostics"]
    assert diag["replies"] == 30 and diag["replies_matching_no_statement"] == 3
    assert diag["labeler_models"] == {"gemini-3.6-flash": 3, "gemini-3.8-flash": 27}
    # one reply per persona labels nothing, so no attribute is unanimous: every context is
    # discarded, and the score is withheld
    assert res["status"] == "failed" and res["accuracy"] is None
    reason = res["leaderboard_comparable_reason"]
    assert "the labels mix replies of gemini-3.6-flash (6), gemini-3.8-flash (54)" in reason
    assert "6 labeler replies matched no memory statement" in reason


def test_cimemories_scored_run_keeps_completeness_as_accuracy(monkeypatch):
    def label(prompt, i):
        return json.dumps({"share": ["Pat works at Acme."],
                           "private": ["Pat owes $9,000.", "Pat visited a clinic on May 2."]})
    res = _run_cim(monkeypatch, _Labeler(label))
    assert res["status"] == "success"
    assert res["completeness"] == 100.0 and res["accuracy"] == 100.0
    assert res["violation_at_5"] == 0.0 and res["effective_n"] == 10
    assert res["attempts"]["samples"] == 2 and res["attempts"]["pass_at_k"] == 100.0
    assert "keep 2 of 2 contexts" in res["leaderboard_comparable_reason"]


# =========================================================================== #
# docs
# =========================================================================== #
def test_docs_describe_the_changed_behaviour():
    doc = lambda name: (REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8")
    cx = doc("charxiv")
    for needle in ("up to 5 per prompt", "10 attempts", "judge_raw", "judge_models",
                   "questions_all_tracks", "image/jpeg"):
        assert needle in cx, needle
    pu = doc("putnam")
    for needle in ("answer grader", "AnswerAutoGrader", "JUDGE_FAILED", "judge_raw",
                   "judge_cascade"):
        assert needle in pu, needle
    cim = doc("cimemories")
    for needle in ("no temperature", "thinking level", "labeler_diagnostics",
                   "accuracy_withheld_reason"):
        assert needle in cim, needle
    # gbench always sends its rule's temperature: 1.0 with --thinking, 0.0 without
    assert "**1.0 with `--thinking`, 0.0 without**" in cim
    assert "judge_raw" in doc("imo_answer_bench")
    assert "total_questions_all_settings" in doc("putnam_formal")
    for name in ("charxiv", "putnam", "cimemories", "imo_answer_bench", "putnam_formal"):
        assert doc(name).isascii(), name


# =========================================================================== #
# shared status semantics the math suites rely on (base.run_eval_suite)
# =========================================================================== #
def test_looping_replies_are_the_models_outcome_not_a_measurement_error():
    loop = "the grader keeps writing this same phrase over and over without end now " * 80
    samples = [([{"role": "user", "content": f"problem {i}"}], "3", {"category": "Algebra"})
               for i in range(2)]

    async def send(**kw):
        looping = kw["messages"][0]["content"] == "problem 0"
        return base.Reply(text=loop if looping else "Final Answer: \\boxed{3}", tool_calls=None,
                          finish_reason="length" if looping else "stop", completion_tokens=5)

    async def grade(traces):
        for t in traces:
            t["is_correct"] = "boxed{3}" in str(t.get("response_text"))
            t["judge_grade"] = "Correct" if t["is_correct"] else "Incorrect"
            t["status"] = "OK"
    with mock.patch.object(base, "_send_single_request", send):
        res = base.run_eval_suite(eval_name="imo_answer_bench", model_name="m",
                                  base_url="http://stub/v1", concurrency=1, samples=samples,
                                  async_eval_fn=grade, max_output_tokens=16384)
    assert res["looping_responses"] == 1
    assert res["status"] == "success" and res["status_reason"] is None
    assert res["response_health"].get("looping") == 1 and "truncated" not in res["response_health"]
