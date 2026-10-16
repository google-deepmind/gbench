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

"""base's ANSWER_IN_THINKING_POLICY outside base.run_eval_suite: a reply that stopped with its
answer inside the thinking channel (reasoning, no content, no tool call) is the model's outcome,
scored, never a failed request, and counted per run in `answer_in_thinking` /
`answer_in_thinking_of` / `answer_in_thinking_rate` by every suite that counts replies itself,
and by `gbench --merge`."""

import pathlib

from gbench import merge as M
from gbench.runners.eval_suites import agent_dojo, base, bfcl, t_eval

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_record_and_its_rate():
    assert base.answer_in_thinking_fields(1, 4) == {
        "answer_in_thinking": 1, "answer_in_thinking_of": 4, "answer_in_thinking_rate": 0.25}
    assert base.answer_in_thinking_fields(0, 0)["answer_in_thinking_rate"] is None


def test_classify_reply_names_the_answer_in_thinking_shape():
    # stopped on its own with reasoning only: empty_reasoned; cut off at the budget: truncated
    assert base.classify_reply(base.Reply("", None, "stop", "the answer is 4")) == \
        "empty_reasoned"
    assert base.classify_reply(base.Reply("", None, "length", "the answer is")) == "truncated"
    assert base.classify_reply(base.Reply("", None, "stop", None)) == "empty"


def test_bfcl_counts_a_stopped_checker_on_an_answer_in_thinking_reply():
    # base no longer counts an answer-in-thinking reply as a failed request, so a checker that
    # stopped on one must be counted by bfcl (a truly empty reply is already counted by base)
    traces = [{"status": "HARNESS_ERROR", "status_before_harness_error": "EMPTY_RESPONSE",
               "health": "empty_reasoned"},
              {"status": "HARNESS_ERROR", "status_before_harness_error": "EMPTY_RESPONSE",
               "health": "empty"},
              {"status": "HARNESS_ERROR", "status_before_harness_error": "OK", "health": "ok"}]
    assert bfcl.harness_unscored(traces) == 2


def _shard(label, traces, **kw):
    i, n = label.split("/")
    correct = sum(1 for t in traces if t.get("is_correct"))
    return {"eval_name": "unit", "shard": label, "shard_index": int(i), "shard_count": int(n),
            "sample_traces": traces, "total_questions": len(traces), "correct_answers": correct,
            "accuracy": round(100.0 * correct / len(traces), 2), "category_accuracy": {},
            "status": "success", "status_reason": None, **kw}


def _trace(key, gidx, in_thinking=False):
    return {"sample_key": key, "global_idx": gidx, "attempt_idx": 0, "sample_idx": gidx,
            "is_correct": not in_thinking, "category": "c",
            "health": "empty_reasoned" if in_thinking else "ok",
            **({"answer_in_thinking": True, "status": "EMPTY_RESPONSE"} if in_thinking else {})}


def test_merge_sums_the_counts_and_recomputes_the_rate_over_the_union():
    a = _shard("1/2", [_trace("a", 0, True), _trace("b", 1)],
               **base.answer_in_thinking_fields(1, 2))
    b = _shard("2/2", [_trace("c", 2), _trace("d", 3), _trace("e", 4), _trace("f", 5)],
               **base.answer_in_thinking_fields(0, 4))
    merged = M.generic_merge("unit", [a, b])
    assert (merged["answer_in_thinking"], merged["answer_in_thinking_of"],
            merged["answer_in_thinking_rate"]) == (1, 6, round(1 / 6, 4))
    assert "merge_unreconciled_fields" not in merged
    assert merged["status"] == "success" and merged["accuracy"] == round(500 / 6, 2)


def test_agent_dojo_reports_the_rate_over_every_request():
    result = {"sample_traces": [{"requests": 3}, {"requests": 5}], "accuracy": 50.0,
              "category_accuracy": {}}
    counters = dict.fromkeys(agent_dojo._COUNTERS, 0)
    counters["answer_in_thinking"] = 2
    items = {"benign": {"workspace/user_task_0": True, "workspace/user_task_1": False},
             "under_attack": {}, "attack_success": {}}
    agent_dojo._apply_summary(result, items, {}, counters, [])
    assert (result["answer_in_thinking"], result["answer_in_thinking_of"],
            result["answer_in_thinking_rate"]) == (2, 8, 0.25)
    assert result["status"] == "success"
    assert "thinking" not in (result.get("leaderboard_comparable_reason") or "")


def test_t_eval_sums_the_count_over_its_files():
    files = [{"total_questions": 3, "answer_in_thinking": 1, "scoring_excluded": 0,
              "judge_outages": 0, "failed_requests": 0, "correct_answers": 1},
             {"total_questions": 5, "answer_in_thinking": 2, "scoring_excluded": 0,
              "judge_outages": 0, "failed_requests": 0, "correct_answers": 2}]
    merged = t_eval._merge_file_results(files)
    assert (merged["answer_in_thinking"], merged["answer_in_thinking_of"],
            merged["answer_in_thinking_rate"]) == (3, 8, 0.375)


def test_the_docs_state_the_policy():
    text = (ROOT / "docs" / "evals" / "answer_in_thinking.md").read_text(encoding="utf-8")
    for needed in ("answer_in_thinking_rate", "empty_reasoned", "Not a failed request",
                   "Gemma 4", "replies_answer_in_thinking", "empty_after_reasoning_turns"):
        assert needed in text, needed
    for suite in ("multi_swe_bench", "lcb", "mmlu_redux", "arc_agi"):
        doc = (ROOT / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")
        assert "answer_in_thinking" in doc, suite


def test_arc_agi_counts_answer_in_thinking_apart_from_empty_replies(monkeypatch):
    from gbench.runners.eval_suites import arc_agi as A
    samples = [([{"role": "user", "content": "q"}], "[[1]]", {"task_id": f"t{i}",
                                                              "category": "arc"})
               for i in range(3)]

    def _t(i, health, status, ok):
        return {"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0, "health": health,
                "status": status, "is_correct": ok, "response_text": "" if not ok else "[[1]]",
                "finish_reason": "stop", "extra_payload": {"task_id": f"t{i}"},
                "category": "arc"}
    traces = [_t(0, "ok", "OK", True), _t(1, "empty_reasoned", "EMPTY_RESPONSE", False),
              _t(2, "empty", "EMPTY_RESPONSE", False)]
    monkeypatch.setattr(A, "_load_arc_agi_samples", lambda **k: samples)
    monkeypatch.setattr(A, "run_eval_suite", lambda **kw: {
        "sample_traces": [dict(t) for t in traces], "status": "success",
        "attempts": {"attempts_per_sample": 1}})
    monkeypatch.setattr(A, "_retry_failed_tries", lambda traces, shown, kw: {
        "retried_attempts": 0, "recovered_by_retry": 0})
    res = A.run_arc_agi("m", "http://x/v1", 1)
    assert res["empty_responses"] == 1 and res["failed_requests"] == 1
    assert (res["answer_in_thinking"], res["answer_in_thinking_of"]) == (1, 3)
    assert res["status"] == "completed_with_errors"
    assert "empty_responses 1" in res["status_reason"]
    # both empty replies are scored (wrong): one task of three solved
    assert res["arc_agi_report"]["tasks"] == 3


def test_swe_bench_live_keeps_each_tabs_count_and_adds_them_at_the_top():
    from gbench.runners.eval_suites import swe_bench_live as LV
    tabs = list(LV.HEADLINE_SETTING and [LV.HEADLINE_SETTING]) + ["other_tab"]
    results = {tabs[0]: {"status": "success", "accuracy": 50.0, "total_questions": 2,
                         "correct_answers": 1, "sample_traces": [],
                         **base.answer_in_thinking_fields(1, 10)},
               tabs[1]: {"status": "success", "accuracy": 0.0, "total_questions": 1,
                         "correct_answers": 0, "sample_traces": [],
                         **base.answer_in_thinking_fields(3, 30)}}
    combined = LV.combine_settings(results)
    assert (combined["answer_in_thinking"], combined["answer_in_thinking_of"],
            combined["answer_in_thinking_rate"]) == (4, 40, 0.1)
    assert combined["settings"][tabs[1]]["answer_in_thinking_rate"] == 0.1
    assert combined["settings"][tabs[0]]["answer_in_thinking"] == 1
