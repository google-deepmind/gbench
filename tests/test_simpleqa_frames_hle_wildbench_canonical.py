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

"""Canonical fidelity of the judged suites simpleqa, frames, humanitys_last_exam and
wildbench: official judge prompts verbatim, official parsing and scoring, pinned
datasets, the judge model recorded on the result, and no no-key fallback.

The template hashes were taken from the upstream files at the commits cited in each module;
a change to a prompt must be a deliberate re-sync, not an edit.
"""

import hashlib
import sys
import types
from unittest import mock

import numpy as np
import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import frames
from gbench.runners.eval_suites import humanitys_last_exam as hle
from gbench.runners.eval_suites import wildbench as lmsys
from gbench.runners.eval_suites import simpleqa


def _as_detailed(judge):
    """A (text, model) judge stub as the cascade's detailed call, (text, model, info), which
    wildbench uses to tell a refusal from an outage."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _user_text(messages):
    """The user turn's text, whether the content is a string or a list of parts."""
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return "".join(p.get("text", "") for p in c if p.get("type") == "text")
            return c
    return ""


def _run_suite(module, run_fn, loader_name, samples, replies, judge, **kw):
    """Drive a suite's run_* end to end: canned model replies, a fake judge cascade.

    `replies` maps the user text to the model reply; `judge(prompt, config)` returns the judge
    text (or None for an outage). Returns (result, judge_calls).
    """
    import contextlib
    calls = []
    extra = []
    if module is lmsys:
        # wildbench also judges each reply against the three WB-Reward references; those
        # pairwise calls are answered with a tie and kept out of `calls`.
        samples = [(m, g, {"session_id": f"s{i}", "wb_pairwise_target_is_a": True, **meta})
                   for i, (m, g, meta) in enumerate(samples)]
        refs = {p: {meta["session_id"]: "reference reply" for _, _, meta in samples}
                for p in lmsys.REFERENCE_MODELS}
        extra.append(mock.patch.object(lmsys, "_load_reference_outputs", lambda: refs))
        # WB-Elo: no pool models, so no pool grading calls (the headline is WB_score.task_macro)
        extra.append(mock.patch.object(lmsys, "_load_wb_elo_pool_outputs", lambda: {}))

    async def _fake_send(*a, **k):
        text = replies[_user_text(k["messages"])]
        return base.Reply(text=text, tool_calls=None, finish_reason="stop")

    async def _fake_judge(prompt, config=None):
        if module is lmsys and "<|begin_of_response_A|>" in prompt:
            return '{"choice": "A=B"}', "gemini-test-judge"
        calls.append((prompt, config))
        out = judge(prompt, config)
        return (out, "gemini-test-judge") if out is not None else (None, "JUDGE_OUTAGE")

    with contextlib.ExitStack() as stack:
        for patch in [mock.patch.object(base, "_send_single_request", _fake_send),
                      (mock.patch.object(module, "judge_generate_cascade_detailed",
                                         _as_detailed(_fake_judge)) if module in (lmsys, simpleqa) else
                       mock.patch.object(module, "judge_generate_cascade", _fake_judge)),
                      mock.patch.object(module, "gemini_required_skip", lambda *a, **k: None),
                      mock.patch.object(module, loader_name, lambda limit=None: samples)] + extra:
            stack.enter_context(patch)
        result = run_fn("m", "http://x", 1, **kw)
    return result, calls


# --------------------------------------------------------------------------- #
# no fallback: every suite hard-errors without a judge key, no heuristic path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module,run_name", [
    (simpleqa, "run_simpleqa"),
    (frames, "run_frames"),
    (hle, "run_humanitys_last_exam"),
    (lmsys, "run_wildbench"),
])
def test_missing_judge_key_is_a_hard_error_not_a_fallback(module, run_name, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        getattr(module, run_name)("m", "http://x", 1)
    src = open(module.__file__, encoding="utf-8").read()
    assert "judge_fallback" not in src, "the substring/heuristic no-key path must stay removed"


@pytest.mark.parametrize("module,revision", [
    (simpleqa, "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94"),
    (frames, "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"),
    (hle, "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"),
    (lmsys, "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"),
])
def test_dataset_revision_is_pinned_and_passed_to_the_loader(module, revision, monkeypatch):
    assert module._REVISION == revision
    seen = {}

    def _fake_load_dataset(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after recording the call")

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=_fake_load_dataset))
    loader = {simpleqa: simpleqa._load_simpleqa_samples,
              frames: frames._load_frames_samples,
              hle: hle._load_humanitys_last_exam_samples,
              lmsys: lmsys._load_wildbench_samples}[module]
    with pytest.raises(RuntimeError):
        loader()
    assert seen.get("revision") == revision


# --------------------------------------------------------------------------- #
# simpleqa (#62, #63)
# --------------------------------------------------------------------------- #
def test_simpleqa_grader_template_is_simple_evals_verbatim():
    # sha256 of openai/simple-evals simpleqa_eval.py GRADER_TEMPLATE @ 652c89d0ca9d
    assert _sha(simpleqa.GRADER_TEMPLATE) == \
        "063c04ea798f91c4706c050b66a721afc77a458b12d8f0d72286d4b0eee8a3ee"
    assert simpleqa.GRADER_TEMPLATE.endswith(
        'Just return the letters "A", "B", or "C", with no text around it.')
    assert simpleqa.GRADER_SYSTEM_MESSAGE == "You are a helpful assistant."


@pytest.mark.parametrize("reply,letter", [
    ("A", "A"), ("B", "B"), ("C", "C"),
    ("", "C"),                       # no letter -> NOT_ATTEMPTED, as upstream defaults
    ("B\n", "B"),
    ("The grade is CORRECT", "C"),   # upstream's regex takes the first A/B/C character
])
def test_simpleqa_grade_letter_is_upstreams_regex(reply, letter):
    assert simpleqa.grade_letter(reply) == letter


def test_simpleqa_aggregate_matches_simple_evals():
    traces = ([{"simpleqa_grade": "CORRECT", "judge_grade": "CORRECT"}] * 3
              + [{"simpleqa_grade": "INCORRECT", "judge_grade": "INCORRECT"}] * 1
              + [{"simpleqa_grade": "NOT_ATTEMPTED", "judge_grade": "NOT_ATTEMPTED"}] * 1
              + [{"judge_grade": "JUDGE_OUTAGE"}])
    agg = simpleqa.simpleqa_aggregate(traces)
    assert agg["n"] == 5
    assert agg["is_correct"] == pytest.approx(0.6)
    assert agg["accuracy_given_attempted"] == pytest.approx(0.75)
    assert agg["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_simpleqa_end_to_end_headline_is_is_correct():
    qs = ["q1", "q2", "q3", "q4"]
    samples = [([{"role": "user", "content": q}], f"gold-{q}", {"category": "Music"}) for q in qs]
    replies = {"q1": "right", "q2": "right", "q3": "wrong", "q4": "I don't know"}
    verdict = {"right": "A", "wrong": "B", "I don't know": "C"}

    def judge(prompt, config):
        return verdict[prompt.split("Predicted answer: ")[-1].split("\n")[0]]

    res, calls = _run_suite(simpleqa, simpleqa.run_simpleqa, "_load_simpleqa_samples",
                            samples, replies, judge)
    assert res["accuracy"] == 50.0                 # 2 correct / 4 graded (simple-evals score)
    assert res["correct"] == 50.0
    assert res["incorrect"] == 25.0
    assert res["not_attempted"] == 25.0
    assert res["accuracy_given_attempted"] == pytest.approx(66.67)
    assert res["f_score"] == pytest.approx(round(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5) * 100, 2))
    assert res["leaderboard_comparable"] is False
    assert "gpt-4.1-2025-04-14" in res["leaderboard_comparable_reason"]
    assert res["judge"] == "gbench-gemini-cascade"
    assert res["judge_models"] == {"gemini-test-judge": 4}
    assert res["grader_replies_not_bare_letter"] == 0
    prompt, cfg = calls[0]
    assert prompt.startswith(simpleqa.GRADER_TEMPLATE.split("{question}")[0])
    if cfg is not None:                             # google-genai installed
        assert cfg.system_instruction == "You are a helpful assistant."


def test_simpleqa_empty_reply_is_not_attempted_without_a_judge_call():
    samples = [([{"role": "user", "content": "q1"}], "g", {}),
               ([{"role": "user", "content": "q2"}], "g", {})]
    res, calls = _run_suite(simpleqa, simpleqa.run_simpleqa, "_load_simpleqa_samples",
                            samples, {"q1": "", "q2": "x"}, lambda p, c: "A")
    assert len(calls) == 1
    assert res["not_attempted"] == 50.0
    assert res["accuracy"] == 50.0


def test_simpleqa_prose_grader_reply_is_parsed_as_upstream_and_counted():
    samples = [([{"role": "user", "content": "q1"}], "g", {})]
    res, _ = _run_suite(simpleqa, simpleqa.run_simpleqa, "_load_simpleqa_samples",
                        samples, {"q1": "x"}, lambda p, c: "Based on the rubric: A")
    assert res["incorrect"] == 100.0                  # first capital A/B/C is the B of "Based"
    assert res["grader_replies_not_bare_letter"] == 1


def test_simpleqa_limit_is_flagged():
    samples = [([{"role": "user", "content": "q1"}], "g", {})]
    res, _ = _run_suite(simpleqa, simpleqa.run_simpleqa, "_load_simpleqa_samples",
                        samples, {"q1": "x"}, lambda p, c: "A", limit=1)
    assert "--eval-limit subset (1 of 4326" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# frames (#83)
# --------------------------------------------------------------------------- #
def test_frames_autorater_is_the_paper_prompt():
    t = frames.AUTORATER_TEMPLATE
    # U+2019 and U+2013 are the LaTeX source's own characters (auto_rate_prompt.tex).
    assert _sha(t) == "19be51708eda955fd476f7dfa67ded558c626ccd30c0eed2aae51e0f3c6d394e"
    assert "the LLM\u2019s response" in t and "answers \u2013 look" in t
    assert t.startswith("===Task===\n\nI need your help in evaluating an answer provided by an "
                        "LLM against a ground truth answer.")
    assert ('Provide your final evaluation in the following format:\n'
            '"Explanation:" (How you made the decision?)\n'
            '"Decision:" ("TRUE" or "FALSE" )\n\nPlease proceed with the evaluation.') in t
    assert frames.AUTORATER_SYSTEM == "You are a helpful assistant."
    assert "Grade:" not in t


@pytest.mark.parametrize("reply,expected", [
    ("Explanation: matches.\nDecision: TRUE", True),
    ('"Explanation:" differs\n"Decision:" "FALSE"', False),
    ("**Explanation:** ok\n**Decision:** True", True),
    ("Decision: FALSE\nOn reflection...\nDecision: TRUE", True),   # the last decision wins
    ("Explanation: the decision is hard", None),
    ("", None),
])
def test_frames_parse_decision(reply, expected):
    assert frames.parse_decision(reply) is expected


def test_frames_end_to_end():
    samples = [([{"role": "user", "content": q}], "Paris", {"category": "Temporal"})
               for q in ("q1", "q2", "q3", "q4")]
    replies = {"q1": "Paris", "q2": "Lyon", "q3": "Paris?", "q4": "Marseille"}
    verdict = {"Paris": "Explanation: same.\nDecision: TRUE",
               "Lyon": "Explanation: no.\nDecision: FALSE",
               "Paris?": "Explanation: likely the same.\n\"Decision:\" \"TRUE\"",
               "Marseille": "I cannot decide."}

    def judge(prompt, config):
        return verdict[prompt.split("- Predicted Answer: ")[1].split("\n")[0]]

    res, calls = _run_suite(frames, frames.run_frames, "_load_frames_samples",
                            samples, replies, judge)
    # no Decision line -> JUDGE_FAILED: an autorater failure, left out of the accuracy
    assert res["accuracy"] == 66.67                 # 2 TRUE / 3 graded
    assert res["judge_failures"] == 1 and res["scoring_excluded"] == 1
    grades = sorted(t["judge_grade"] for t in res["sample_traces"])
    assert grades == ["FALSE", "JUDGE_FAILED", "TRUE", "TRUE"]
    assert res["leaderboard_comparable"] is False
    assert "Gemini-Pro-1.5-0514" in res["leaderboard_comparable_reason"]
    assert res["judge_models"] == {"gemini-test-judge": 4}
    prompt, cfg = calls[0]
    assert "- Ground Truth Answer: Paris\n" in prompt
    if cfg is not None:
        assert cfg.system_instruction == "You are a helpful assistant."


# --------------------------------------------------------------------------- #
# humanitys_last_exam (#47)
# --------------------------------------------------------------------------- #
def test_hle_prompts_are_the_official_ones():
    # sha256 of centerforaisafety/hle @ 22ed3074b1e7 SYSTEM_PROMPT / JUDGE_PROMPT
    assert _sha(hle.SYSTEM_PROMPT) == \
        "40f399fe8c277e2cb98ad8ca663461141d1abbf01a18da3c4129eccc706b4326"
    assert _sha(hle.JUDGE_PROMPT) == \
        "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835"
    assert "0|\\%| and 100|\\%|" in hle.JUDGE_PROMPT


def test_hle_loader_sends_system_prompt_and_text_before_image(monkeypatch):
    rows = [{"id": "1", "question": "What is shown?", "answer": "a cat", "category": "Other",
             "image": "data:image/png;base64,AAAA"},
            {"id": "2", "question": "2+2?", "answer": "4", "category": "Math", "image": ""}]
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *a, **k: rows))
    samples = hle._load_humanitys_last_exam_samples()
    by_gold = {s[1]: s[0] for s in samples}
    img = by_gold["a cat"]
    assert img[0] == {"role": "system", "content": hle.SYSTEM_PROMPT}
    assert img[1]["content"] == [
        {"type": "text", "text": "What is shown?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert by_gold["4"][1] == {"role": "user", "content": "2+2?"}
    assert hle._question_text(img) == "What is shown?"


def test_hle_parse_judge_reply():
    js = ('{"extracted_final_answer": "17", "reasoning": "same", "correct": "yes", '
          '"confidence": 85, "strict": true}')
    assert hle.parse_judge_reply(js) == {"model_answer": "17", "reasoning": "same",
                                         "correct": "yes", "confidence": 85.0}
    assert hle.parse_judge_reply("```json\n" + js + "\n```")["correct"] == "yes"
    lines = ("extracted_final_answer: 3\n\nreasoning: differs\n\ncorrect: no\n\n"
             "confidence: 40%")
    v = hle.parse_judge_reply(lines)
    assert (v["correct"], v["confidence"], v["model_answer"]) == ("no", 40.0, "3")
    assert hle.parse_judge_reply("**correct:** yes\n**confidence:** 70")["correct"] == "yes"
    assert hle.parse_judge_reply("correct: no, the values differ")["correct"] == "no"
    # the prompt's rule: no confidence available -> 100
    assert hle.parse_judge_reply('{"correct": "no"}')["confidence"] == 100.0
    assert hle.parse_judge_reply("the value 17 shows up but the answer is 3") is None
    assert hle.parse_judge_reply('{"correct": "maybe"}') is None


def test_hle_calib_err_is_the_official_estimator():
    # 250 samples, beta=100: bins [0,100) and [100,250); upstream's loop stops before the
    # last bin, so only the 100 lowest-confidence samples count.
    conf = np.concatenate([np.linspace(0.10, 0.199, 100), np.linspace(0.5, 0.9, 150)])
    correct = np.concatenate([np.ones(100), np.zeros(150)])
    expected = abs(conf[:100].mean() - 1.0) * np.sqrt(100 / 250)
    assert hle.calib_err(conf, correct, p='2', beta=100) == pytest.approx(expected)


def test_hle_calibration_uses_judge_confidence_and_needs_two_bins():
    traces = [{"is_correct": True, "hle_confidence": 10.0 + i * 0.099, "judge_grade": "yes"}
              for i in range(100)]
    traces += [{"is_correct": False, "hle_confidence": 90.0, "judge_grade": "no"}
               for _ in range(100)]
    traces += [{"judge_grade": "JUDGE_OUTAGE"}, {"judge_grade": "JUDGE_FAILED"}]
    cerr, n = hle.hle_calibration(traces)
    assert n == 200
    mean_conf = np.mean([10.0 + i * 0.099 for i in range(100)]) / 100
    assert cerr == round(100 * round(float(abs(mean_conf - 1.0) * np.sqrt(0.5)), 2), 2)
    assert hle.hle_calibration(traces[:150]) == (None, 150)


def test_hle_end_to_end():
    n = 220
    samples = [([{"role": "system", "content": hle.SYSTEM_PROMPT},
                 {"role": "user", "content": f"q{i}"}], f"a{i}", {"category": "Math"})
               for i in range(n)]
    replies = {f"q{i}": f"Explanation: x\nAnswer: a{i if i % 2 else 'x'}\nConfidence: 80%"
               for i in range(n)}

    def judge(prompt, config):
        q = prompt.split("[question]: ")[1].split("\n")[0]
        i = int(q[1:])
        if i == 0:
            return None                              # one judge outage
        ok = "yes" if i % 2 else "no"
        return ('{"extracted_final_answer": "x", "reasoning": "r", "correct": "%s", '
                '"confidence": 80, "strict": true}' % ok)

    res, calls = _run_suite(hle, hle.run_humanitys_last_exam,
                            "_load_humanitys_last_exam_samples", samples, replies, judge)
    assert res["judge_outages"] == 1
    assert res["accuracy"] == round(110 / 219 * 100, 2)
    assert res["calibration_samples"] == 219
    assert res["calibration_error"] is not None
    assert res["accuracy_ci95_half_width"] > 0
    assert res["leaderboard_comparable"] is False
    assert "o3-mini-2025-01-31" in res["leaderboard_comparable_reason"]
    assert res["judge_models"] == {"gemini-test-judge": n - 1}
    prompt, cfg = calls[0]
    assert hle.SYSTEM_PROMPT not in prompt           # the judge sees the question only
    if cfg is not None:
        assert cfg.response_mime_type == "application/json"
        assert list(cfg.response_schema.property_ordering) == [
            "extracted_final_answer", "reasoning", "correct", "confidence", "strict"]


def test_hle_has_no_deterministic_prepass():
    assert not hasattr(hle, "_eval_humanitys_last_exam")


# --------------------------------------------------------------------------- #
# wildbench (#50, #27)
# --------------------------------------------------------------------------- #
def test_wildbench_template_is_score_v2_verbatim():
    # sha256 of allenai/WildBench evaluation/eval_template.score.v2.md @ d6b8dcaf377d
    assert _sha(lmsys.EVAL_TEMPLATE) == \
        "c5984ae77b009f8dad0e8e37093348345f4e8d3099ce812bd7799ef7b73f005f"


def test_wildbench_prompt_splits_history_from_current_query():
    msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "now list them"}]
    long_out = "word " * 3000
    p = lmsys.build_wb_judge_prompt(msgs, ["Is it numbered?", "Is it sorted?"], long_out)
    assert "<|begin_of_history|>\n\nUSER: first\n\nASSISTANT: reply\n\n\n\n<|end_of_history|>" in p
    assert "<|begin_of_query|>\n\nnow list them\n\n<|end_of_query|>" in p
    assert "<|begin_of_checklist|>\n\n- Is it numbered?\n- Is it sorted?\n\n\n<|end_of_checklist|>" in p
    assert long_out in p                              # --max_words_to_eval -1: no truncation
    assert "{$" not in p


@pytest.mark.parametrize("reply,score", [
    ('{"strengths": "clear", "weaknesses": "none", "score": "8"}', 8.0),
    # batch_results_format.py drops a reply json.loads cannot read (no regex fallback)
    ('```\n{"strengths": "a", "weaknesses": "b", "score": "6"}\n```', None),
    ('{"strengths": "a", "weaknesses": "b", "score": 9}', 9.0),
    ("x" * 500, None),                                # no length-based credit
    ('{"strengths": "a", "weaknesses": "b"}', None),
])
def test_wildbench_score_parse(reply, score):
    assert lmsys.parse_wb_score(reply) == score


def test_lmsys_end_to_end_wb_score():
    samples = [([{"role": "user", "content": q}], ["c1"], {"category": "Math"})
               for q in ("q1", "q2", "q3", "q4")]
    replies = {"q1": "good", "q2": "fair", "q3": "", "q4": "odd"}
    verdict = {"good": '{"strengths": "s", "weaknesses": "w", "score": "9"}',
               "fair": '{"strengths": "s", "weaknesses": "w", "score": "5"}',
               "": '{"strengths": "N/A", "weaknesses": "empty", "score": "1"}',
               "odd": "no score here"}

    def judge(prompt, config):
        out = prompt.split("<|begin_of_response|>\n\n")[1].split("\n\n<|end_of_response|>")[0]
        return verdict[out]

    res, calls = _run_suite(lmsys, lmsys.run_wildbench,
                            "_load_wildbench_samples", samples, replies, judge)
    # raw 9, 5, empty judged 1, unparseable -> excluded: adjusted (8 + 0 - 8) / 3 * 10
    assert res["wb_score"] == 0.0
    assert res["scored_samples"] == 3
    assert len(calls) == 4                            # the empty reply is judged too
    assert res["judge_failures"] == 1 and res["judge_outages"] == 0   # a grading failure
    assert res["leaderboard_comparable"] is False
    assert "gpt-4o-2024-05-13" in res["leaderboard_comparable_reason"]
    assert res["judge_models"] == {"gemini-test-judge": 4}
    if calls[0][1] is not None:
        assert calls[0][1].response_mime_type == "application/json"


def test_lmsys_headline_is_mean_raw_minus_5_times_20():
    samples = [([{"role": "user", "content": q}], ["c"], {}) for q in ("q1", "q2")]
    replies = {"q1": "a", "q2": "b"}
    scores = {"a": "10", "b": "7"}

    def judge(prompt, config):
        out = prompt.split("<|begin_of_response|>\n\n")[1].split("\n\n<|end_of_response|>")[0]
        return '{"strengths": "", "weaknesses": "", "score": "%s"}' % scores[out]

    res, _ = _run_suite(lmsys, lmsys.run_wildbench,
                        "_load_wildbench_samples", samples, replies, judge)
    assert res["wb_score"] == pytest.approx((8.5 - 5) * 20)


def test_lmsys_has_no_heuristic_fallback():
    assert not hasattr(lmsys, "_eval_lmsys_noncoding_hard")


# --------------------------------------------------------------------------- #
# lmsys grading failures, judge record schema, frames header, docs
# --------------------------------------------------------------------------- #
def test_lmsys_unparseable_reply_is_judge_failed_excluded_and_statuses_are_ok():
    import asyncio

    def _trace(text):
        return {"messages": [{"role": "user", "content": "q"}], "gold_answer": ["c"],
                "response_text": text, "status": "PENDING_JUDGE"}
    bad, outage, good = _trace("odd"), _trace("gone"), _trace("fine")
    replies = {"odd": '{"strengths": "s", "weaknesses": "w", "score": "[1~10]"}',
               "gone": None, "fine": '{"score": "8"}'}

    async def _fake_judge(prompt, config=None):
        out = replies[prompt.split("<|begin_of_response|>\n\n")[1].split("\n\n<|end_of_response|>")[0]]
        return (out, "gemini-test-judge") if out is not None else (None, "JUDGE_OUTAGE")

    with mock.patch.object(lmsys, "judge_generate_cascade_detailed", _as_detailed(_fake_judge)):
        asyncio.run(lmsys._async_judge_wildbench([bad, outage, good]))
    assert bad["judge_grade"] == "JUDGE_FAILED" and bad["scoring_excluded"] is True
    assert "wb_score" not in bad and bad["status"] == "OK"
    assert outage["judge_grade"] == "JUDGE_OUTAGE" and outage["status"] == "OK"
    assert good["wb_score"] == 6.0 and good["status"] == "OK"


def test_lmsys_grading_failure_is_not_scored_at_the_failure_value():
    samples = [([{"role": "user", "content": q}], ["c"], {"category": "Math"}) for q in ("q1", "q2")]
    replies = {"q1": "good", "q2": "odd"}
    verdict = {"good": '{"score": "10"}', "odd": "no score here"}

    def judge(prompt, config):
        return verdict[prompt.split("<|begin_of_response|>\n\n")[1].split("\n\n<|end_of_response|>")[0]]

    res, _ = _run_suite(lmsys, lmsys.run_wildbench,
                        "_load_wildbench_samples", samples, replies, judge)
    # only the raw-10 task is in the mean: (10 - 5) * 20; the failure is not counted at -8
    assert res["wb_score"] == 100.0 and res["scored_samples"] == 1
    assert res["judge_failures"] == 1 and res["scoring_excluded"] == 1
    assert len(res["sample_traces"]) == 2                  # every trace is kept in the result


@pytest.mark.parametrize("module,run_name,loader,reply", [
    (simpleqa, "run_simpleqa", "_load_simpleqa_samples", "A"),
    (frames, "run_frames", "_load_frames_samples", "Explanation: same.\nDecision: TRUE"),
    (hle, "run_humanitys_last_exam", "_load_humanitys_last_exam_samples",
     '{"extracted_final_answer": "x", "reasoning": "r", "correct": "yes", "confidence": 90, '
     '"strict": true}'),
    (lmsys, "run_wildbench", "_load_wildbench_samples", '{"score": "7"}'),
])
def test_results_record_the_shared_judge_schema(module, run_name, loader, reply):
    gold = ["c"] if module is lmsys else "g"
    samples = [([{"role": "user", "content": q}], gold, {"category": "x"}) for q in ("q1", "q2")]
    res, calls = _run_suite(module, getattr(module, run_name), loader, samples,
                            {"q1": "a", "q2": "b"}, lambda p, c: reply)
    assert res["judge"] == "gbench-gemini-cascade"
    assert res["judge_cascade"] == base.judge_cascade()
    # judge_models counts answered judge calls; these suites make one call per graded sample
    assert res["judge_models"] == {"gemini-test-judge": len(calls)} and len(calls) == 2


def test_frames_module_header_uses_the_paper_name():
    # the module's "# Description:" header line (after the license header)
    head = next(ln for ln in open(frames.__file__, encoding="utf-8").read().splitlines()
                if ln.startswith("# Description:"))
    assert "Factuality, Retrieval, And reasoning MEasurement Set" in head
    assert "Multi-hop Evaluation Suite" not in open(frames.__file__, encoding="utf-8").read()


@pytest.mark.parametrize("suite", ["simpleqa", "frames", "humanitys_last_exam",
                                   "wildbench"])
def test_docs_match_the_key_gate_judge_schema_and_cli(suite):
    from pathlib import Path
    doc = (Path(__file__).resolve().parent.parent / "docs" / "evals" / f"{suite}.md").read_text(
        encoding="utf-8")
    run = doc.split("## Run", 1)[1].split("```")[1]
    assert "--max-output-tokens" in run                  # required by the CLI
    assert "`GEMINI_API_KEYS`" in doc                    # the gate reads it too
    assert "`judge_cascade`" in doc and "answered" in doc
    assert "unless `--max-output-tokens` is set" not in doc
    if suite == "wildbench":
        assert "`JUDGE_FAILED`" in doc and "`scoring_excluded`" in doc
    assert all(ord(ch) < 128 for ch in doc)
