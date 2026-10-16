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

"""BFCL's agentic track: what was drawn, what each verdict was, and what the headline covers.

* An --eval-limit / --shard draw comes before the Web Search setting's key check: a draw with no
  web_search entry checks no key, and its undrawn categories are not reported as run.
* A group's summary needs every one of its categories, the Agentic score both groups (rounded
  once); a subset's headline is the pooled accuracy, BFCL's zero-filled value kept beside it.
* Every scored entry is a trace (correct ones included); error types count in-score rows only.
* bfcl's @k block is null, never 0, when every attempt is a harness error, and a full run with an
  unscored category withholds Overall Acc.

No network, no docker: the harness CLI and the image are faked.
"""

import json
from pathlib import Path

import pytest

from gbench.runners.eval_suites import bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG

ALL = list(AG.WEB_SEARCH_CATEGORIES) + list(AG.MEMORY_CATEGORIES)


def test_a_group_needs_every_category_and_the_means_are_rounded_once():
    rows = {"web_search_base": (50, 100), "web_search_no_snippet": (50, 100),
            "memory_kv": (52, 155), "memory_vector": (52, 155), "memory_rec_sum": (51, 155)}
    per_cat = {c: {"correct": a, "total": b, "accuracy": round(100.0 * a / b, 2)}
               for c, (a, b) in rows.items()}
    s = AG.agentic_scores(per_cat)
    # exact: web 50.0, memory 33.1183 -> overall 41.559...; BFCL's own value is 41.56 too
    assert s["overall_accuracy_agentic"] == round((50.0 + 100.0 * 155 / 465) / 2, 2)
    assert AG.bfcl_zero_filled_overall(per_cat) == s["overall_accuracy_agentic"]
    partial = {c: v for c, v in per_cat.items() if c != "memory_rec_sum"}
    s = AG.agentic_scores(partial)
    assert s["summary_memory"] is None and s["overall_accuracy_agentic"] is None
    assert s["summary_web_search"] == 50.0
    assert AG.bfcl_zero_filled_overall(partial) == round(
        (50.0 + (100.0 * 52 / 155 * 2 + 0.0) / 3) / 2, 2)


def _result_file(root: Path, model: str, category: str, rows):
    d = root / model.replace("/", "_") / "agentic"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"BFCL_v4_{category}_result.json").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _score_file(root: Path, model: str, category: str, wrong):
    d = root / model.replace("/", "_") / "agentic"
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"accuracy": 0.5})] + [json.dumps(w) for w in wrong]
    (d / f"BFCL_v4_{category}_score.json").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_every_scored_entry_is_a_trace_with_its_verdict(tmp_path):
    res_root, score_root = tmp_path / "result", tmp_path / "score"
    _result_file(res_root, "m", "memory_kv", [
        {"id": "memory_kv_prereq_0", "result": "x"},
        {"id": "memory_kv_0", "result": [["remembered"]]},
        {"id": "memory_kv_1", "result": [["forgot"]]}])
    _score_file(score_root, "m", "memory_kv", [
        {"id": "memory_kv_1", "valid": False, "test_category": "memory_kv",
         "error": {"error_type": "agentic:answer_not_found", "error_message": "no match",
                   "details": {"model_response": "forgot", "possible_answers": ["x"]}}}])
    # a category without a score file has no verdict to report
    _result_file(res_root, "m", "memory_vector", [{"id": "memory_vector_0", "result": "y"}])
    traces = AG._parse_score_traces(score_root, "m", ["memory_kv", "memory_vector"], res_root)
    assert [(t["question_id"], t["is_correct"]) for t in traces] == [
        ("memory_kv_0", True), ("memory_kv_1", False)]
    assert traces[0]["response_text"] == "[['remembered']]"
    assert traces[1]["error_type"] == "agentic:answer_not_found"


def test_error_types_count_the_entries_in_the_score(tmp_path):
    score_root = tmp_path / "score"
    _score_file(score_root, "m", "memory_kv", [
        {"id": f"memory_kv_{i}", "valid": False,
         "error": {"error_type": "agentic:no_last_message"}} for i in range(3)])
    assert AG._score_error_types(score_root, "m", ["memory_kv"]) == {
        "agentic:no_last_message": 3}
    left_out = {("memory_kv", "memory_kv_0"), ("memory_kv", "memory_kv_1")}
    assert AG._score_error_types(score_root, "m", ["memory_kv"], left_out) == {
        "agentic:no_last_message": 1}


@pytest.fixture
def drawn_run(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic over all five categories with an --eval-limit draw that holds only
    memory entries; the harness CLI, the image and the key check are faked."""
    root = tmp_path / "bfcl"
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GBENCH_SHARD", "BFCL_PROJECT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites", lambda url: (True, "", list(ALL)))
    state = {"key_checks": 0}

    def search_backend():
        state["key_checks"] += 1
        return None, "GEMINI_API_KEY is not set"
    monkeypatch.setattr(AG, "search_backend", search_backend)
    draw = {"memory_kv": ["memory_kv_0"], "memory_vector": ["memory_vector_0"]}
    monkeypatch.setattr(AG, "run_in_image", lambda suite, mod, fn, args, timeout=0: (
        draw if fn == "_select_limited_ids" else {}))

    def cli(args, env, timeout, model="", backend=None):
        if args[0] == "generate":
            rdir = args[args.index("--result-dir") + 1]
            for cat, ids in draw.items():
                _result_file(root / rdir, model, cat, [{"id": i, "result": [["ok"]]} for i in ids])
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda r, m, cats: {
        c: {"correct": 1, "total": 1, "accuracy": 100.0} for c in cats})
    return state


def test_a_draw_without_web_search_entries_needs_no_key(drawn_run):
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1, limit=2)
    assert drawn_run["key_checks"] == 0
    report = res["bfcl_report"]
    assert report["categories_run"] == ["memory_kv", "memory_vector"]
    assert report["categories_not_drawn"] == ["web_search_base", "web_search_no_snippet",
                                              "memory_rec_sum"]
    assert report["web_search_categories_run"] == [] and "web_search" not in res["settings"]
    assert res["settings"]["memory"]["categories"] == ["memory_kv", "memory_vector"]
    reason = res["leaderboard_comparable_reason"]
    assert AG.SEARCH_SUBSTITUTION_REASON not in reason
    assert "no entry of web_search_base, web_search_no_snippet, memory_rec_sum was drawn" in reason
    # two of the five categories: no Agentic score; the pooled accuracy is the headline
    assert res["overall_accuracy_agentic"] is None and res["summary_memory"] is None
    assert res["accuracy"] == res["accuracy_pooled"] == 100.0
    assert res["metric"].startswith("accuracy_pooled")
    assert res["overall_accuracy_agentic_bfcl_missing_as_zero"] == round(
        (0.0 + 200.0 / 3) / 2, 2)
    assert res["status"] == "success"


def test_bfcl_attempts_are_null_when_every_attempt_is_a_harness_error():
    traces = [{"sample_idx": i, "source_sample_idx": 0, "is_correct": False,
               "scoring_excluded": True} for i in range(3)]
    rep = bfcl._attempts_report(traces, 3)
    assert rep["samples"] == 0 and rep["generations"] == 0
    assert rep["avg_at_k"] is None and rep["pass_at_k"] is None and rep["pass_hat_k"] is None
