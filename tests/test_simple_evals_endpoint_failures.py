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

"""simpleqa, healthbench and wildbench: a request that got no reply from the served endpoint is a
harness error. It is not graded, it stays out of every figure (headline, split, per-tag means,
WB-Reward totals), it is counted and named, and a run of nothing but harness errors withholds its
score (null + reason) instead of reporting 0. The endpoint and Gemini are stubbed."""

import asyncio
import json
from pathlib import Path
from unittest import mock

from gbench.runners.eval_suites import base, healthbench, simpleqa, wildbench

def _patch_detailed(module, judge):
    """Patch a suite's judge_generate_cascade_detailed with a (text, model) judge stub; the
    detailed call returns (text, model, info) with nothing cut off at the grader's cap."""
    async def detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {"finish_reason": None, "max_tokens_without_text": 0}
    return mock.patch.object(module, "judge_generate_cascade_detailed", detailed)


REPO = Path(__file__).resolve().parent.parent


def _simpleqa_detailed(judge):
    """A (text, model) judge stub as simpleqa's detailed cascade call (text, model, info)."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed


def _serve(replies):
    """A fake _send_single_request answering by the last user text (None = HTTP 500)."""
    async def _send(**kw):
        user = next(m["content"] for m in reversed(kw["messages"]) if m.get("role") == "user")
        reply = replies[user]
        if reply is None:
            return base.Reply(text=None, tool_calls=None, error="HTTP 500 server error")
        return base.Reply(text=reply, tool_calls=None, finish_reason="stop", completion_tokens=5)
    return _send


def _doc(name):
    return " ".join((REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8").split())


# --------------------------------------------------------------------------- #
# simpleqa
# --------------------------------------------------------------------------- #
def _simpleqa(replies):
    samples = [([{"role": "user", "content": q}], "Paris", {"category": "T"}) for q in replies]
    prompts = []

    async def _judge(prompt, config=None):
        prompts.append(prompt)
        return "A", "gemini-3.8-flash"
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies)), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples", lambda limit=None: samples), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed",
                              _simpleqa_detailed(_judge)):
        return simpleqa.run_simpleqa("m", "http://x/v1", 1, max_output_tokens=512), prompts


def test_simpleqa_endpoint_failure_is_left_out_of_every_figure():
    res, prompts = _simpleqa({"q0": "Paris", "q1": None})
    assert len(prompts) == 1                                   # the failure is not graded
    assert res["accuracy"] == 100.0 and res["correct"] == 100.0
    assert res["not_attempted"] == 0.0 and res["incorrect"] == 0.0
    assert res["accuracy_given_attempted"] == 100.0 and res["f_score"] == 100.0
    assert res["status"] == "completed_with_errors"
    assert res["failed_requests"] == 1 and res["scoring_excluded"] == 1
    assert "harness errors left out of the score" in res["status_reason"]
    failed = res["sample_traces"][1]
    assert failed["judge_grade"] == "not_judged_failed_request"
    assert "simpleqa_grade" not in failed
    assert failed["scoring_excluded"] is True
    assert failed["harness_error"]["kind"] == "endpoint_failure"


def test_simpleqa_all_endpoint_failures_withhold_the_score_and_the_split():
    res, prompts = _simpleqa({"q0": None, "q1": None})
    assert prompts == []
    assert res["status"] == "failed"
    for key in ("accuracy", "correct", "incorrect", "not_attempted", "accuracy_given_attempted",
                "f_score"):
        assert res[key] is None, key
    assert "nothing was scored" in res["accuracy_withheld_reason"]
    assert "endpoint_failure 2" in res["status_reason"]


def test_simpleqa_empty_reply_is_still_the_models_not_attempted():
    res, prompts = _simpleqa({"q0": "Paris", "q1": ""})
    assert len(prompts) == 1
    assert res["accuracy"] == 50.0 and res["not_attempted"] == 50.0
    empty = res["sample_traces"][1]
    assert empty["simpleqa_grade"] == "NOT_ATTEMPTED" and not empty.get("scoring_excluded")


def test_simpleqa_aggregate_skips_what_base_leaves_out():
    traces = [{"simpleqa_grade": "CORRECT", "judge_grade": "CORRECT"},
              {"simpleqa_grade": "NOT_ATTEMPTED", "judge_grade": "x", "scoring_excluded": True},
              {"judge_grade": "JUDGE_OUTAGE"}]
    agg = simpleqa.simpleqa_aggregate(traces)
    assert agg["n"] == 1 and agg["is_correct"] == 1.0 and agg["is_not_attempted"] == 0.0
    assert simpleqa.simpleqa_aggregate(traces[1:])["n"] == 0


# --------------------------------------------------------------------------- #
# healthbench
# --------------------------------------------------------------------------- #
def _healthbench(replies):
    rubric = json.dumps([{"criterion": "Says hi", "points": 5, "tags": ["axis:x"]},
                         {"criterion": "Is rude", "points": -2, "tags": []}])
    samples = [([{"role": "user", "content": q}], rubric,
                {"category": "theme:x", "example_tags": ["theme:x"]}) for q in replies]
    prompts = []

    async def _judge(prompt, config=None):
        prompts.append(prompt)
        met = "true" if "Says hi" in prompt else "false"
        return '{"explanation": "e", "criteria_met": %s}' % met, "gemini-3.8-flash"
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies)), \
            mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(healthbench, "_load_healthbench_samples",
                              lambda limit=None: samples), \
            _patch_detailed(healthbench, _judge):
        return healthbench.run_healthbench("m", "http://x/v1", 1, max_output_tokens=512), prompts


def test_healthbench_endpoint_failure_is_not_graded_and_left_out_of_the_mean():
    res, prompts = _healthbench({"h1": "hi", "h2": None})
    assert len(prompts) == 2                                   # h1's two rubric items only
    assert not any(healthbench.NO_RESPONSE_TEXT in p for p in prompts)
    assert res["accuracy"] == 100.0 and res["healthbench_scored_conversations"] == 1
    assert res["healthbench_tag_scores"] == {"axis:x": 1.0, "theme:x": 1.0}
    assert res["status"] == "completed_with_errors"
    assert res["failed_requests"] == 1 and res["scoring_excluded"] == 1
    assert "healthbench_no_response_graded" not in res
    failed = res["sample_traces"][1]
    assert failed["judge_grade"] == "not_judged_failed_request"
    assert failed["scoring_excluded"] is True and "healthbench_score" not in failed
    assert "1 request(s) got no reply from the served endpoint" in \
        res["leaderboard_comparable_reason"]


def test_healthbench_all_endpoint_failures_withhold_the_score():
    res, prompts = _healthbench({"h1": None, "h2": None})
    assert prompts == []
    assert res["accuracy"] is None and res["status"] == "failed"
    assert "healthbench_scored_conversations" not in res
    why = res["accuracy_withheld_reason"]
    assert "no conversation received a HealthBench score" in why
    assert "2 request(s) that got no reply from the served endpoint" in why
    assert "endpoint_failure 2" in why and res["status_reason"] == why


def test_healthbench_summary_skips_a_trace_base_left_out():
    kept = {"healthbench_score": 0.2, "judge_grade": "SCORE_0.20", "rubric_grades": []}
    left_out = {"healthbench_score": 1.0, "judge_grade": "SCORE_1.00", "rubric_grades": [],
                "scoring_excluded": True}
    s = healthbench.healthbench_summary([kept, left_out])
    assert s["n_samples"] == 1 and s["overall_score"] == 0.2


# --------------------------------------------------------------------------- #
# wildbench
# --------------------------------------------------------------------------- #
def _wb(replies):
    samples = [([{"role": "user", "content": q}], ["c"],
                {"category": cat, "wb_task_groups": ["Creative Tasks"], "session_id": q,
                 "wb_pairwise_target_is_a": True})
               for q, cat in (("s1", "Brainstorming"), ("s2", "Editing"), ("s3", "Editing"))]
    refs = {p: {"s1": "r", "s2": "r", "s3": "r"} for p in wildbench.REFERENCE_MODELS}
    prompts = []

    async def _judge(prompt, config=None):
        prompts.append(prompt)
        if "<|begin_of_response_A|>" in prompt:
            return '{"choice": "A+"}', "gemini-3.7-flash", {"finish_reason": "STOP"}
        return '{"score": "8"}', "gemini-3.8-flash", {"finish_reason": "STOP"}
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies)), \
            mock.patch.object(wildbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(wildbench, "_load_wildbench_samples", lambda **k: samples), \
            mock.patch.object(wildbench, "_load_reference_outputs", lambda: refs), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_outputs", lambda: iter([])), \
            mock.patch.object(wildbench, "judge_generate_cascade_detailed", _judge):
        return wildbench.run_wildbench("m", "http://x/v1", 1, max_output_tokens=512), prompts


def test_wildbench_endpoint_failure_is_in_no_reward_total_and_not_judged():
    res, prompts = _wb({"s1": "reply-one", "s2": None, "s3": "reply-three"})
    # 2 score calls + 2 x 3 pairwise calls; nothing for s2
    assert len(prompts) == 8
    assert not any(wildbench.EMPTY_RESPONSE_PLACEHOLDER in p for p in prompts)
    for prefix in wildbench.REFERENCE_MODELS:
        ref = res["wb_reward"]["-1"]["references"][prefix]
        assert (ref["win"], ref["total"]) == (2, 2), prefix
        assert res["wb_reward_by_reference"][prefix] == 50.0
    assert res["wb_score"] == 60.0 and res["scored_samples"] == 2
    assert res["category_accuracy"]["Editing"]["accuracy"] == 60.0     # s3 only, not -8
    assert res["status"] == "completed_with_errors"
    assert res["failed_requests"] == 1 and res["scoring_excluded"] == 1
    failed = res["sample_traces"][1]
    assert failed["judge_grade"] == "not_judged_failed_request"
    assert failed["scoring_excluded"] is True
    assert all("outcome" not in r and r["judge_grade"] == "not_judged_failed_request"
               for r in failed["wb_pairwise"].values())
    assert res["wb_reward_judge_failures"] == {p: 0 for p in wildbench.REFERENCE_MODELS}


def test_wildbench_all_endpoint_failures_withhold_every_score():
    res, prompts = _wb({"s1": None, "s2": None, "s3": None})
    assert prompts == []
    assert res["status"] == "failed" and res["accuracy"] is None and res["wb_score"] is None
    assert res["wb_reward_mix"] is None and res["wb_reward_task_macro"] is None
    assert "nothing was scored" in res["accuracy_withheld_reason"]


def test_wb_reward_summary_leaves_out_a_harness_error_but_not_a_score_mode_failure():
    outcome = wildbench.pairwise_outcome("A+", True, "x", "y")
    pairwise = {p: {"judge_grade": "CHOICE_A+", "outcome": dict(outcome)}
                for p in wildbench.REFERENCE_MODELS}
    no_reply = {"response_text": None, "scoring_excluded": True,
                "harness_error": {"kind": "endpoint_failure", "detail": "500"},
                "wb_pairwise": pairwise}
    # a WB-Score grading failure on a task keeps its comparisons (judged on their own)
    score_failed = {"response_text": "x", "scoring_excluded": True, "judge_grade": "JUDGE_FAILED",
                    "wb_pairwise": pairwise}
    summary = wildbench.wb_reward_summary([no_reply, score_failed], None)
    for prefix in wildbench.REFERENCE_MODELS:
        assert summary["by_k"]["-1"]["references"][prefix]["total"] == 1


def test_wildbench_has_no_worst_score_for_a_failed_request():
    assert not hasattr(wildbench, "FAILED_REQUEST_WB_SCORE")
    assert wildbench.wb_score_by_category([
        {"category": "c", "response_text": None, "scoring_excluded": True},
        {"category": "c", "response_text": "x", "wb_score": 4.0}]) == {"c": 40.0}


def test_direct_scorer_call_on_a_failed_request_marks_it_left_out():
    failed = {"messages": [{"role": "user", "content": "q"}], "gold_answer": ["c"],
              "response_text": None, "status": "FAILED",
              "extra_payload": {"session_id": "s", "wb_pairwise_target_is_a": True}}
    calls = []

    async def _judge(prompt, config=None):
        calls.append(prompt)
        return '{"score": "8"}', "g", {}
    with mock.patch.object(wildbench, "judge_generate_cascade_detailed", _judge):
        asyncio.run(wildbench._async_judge_wildbench(
            [failed], references={p: {"s": "r"} for p in wildbench.REFERENCE_MODELS}))
    assert calls == [] and failed["scoring_excluded"] is True
    assert set(failed["wb_pairwise"]) == set(wildbench.REFERENCE_MODELS)


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #
def test_docs_state_the_harness_error_rule():
    sq = _doc("simpleqa")
    assert "is a **harness error**, never the model's outcome" in sq
    assert "`judge_grade: not_judged_failed_request`" in sq and "are null" in sq
    hb = _doc("healthbench")
    assert "are **harness errors**, never the model's outcome" in hb
    assert "healthbench_no_response_graded" not in hb and "NO_RESPONSE_GRADING_FAILED" not in hb
    assert "`No response (bad request).`" in hb           # the upstream difference is named
    wb = _doc("wildbench")
    assert "counts at -8" not in wb and "at the worst score (raw 1)" not in wb
    assert "compared as an empty reply" not in wb
    assert "in no reference's `total`" in wb
    for doc in (sq, hb, wb):
        assert all(ord(ch) < 128 for ch in doc)
