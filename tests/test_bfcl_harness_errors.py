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

"""bfcl / bfcl_v4_agentic: harness errors are never the model's outcome, in every part alike.

An entry whose request got no reply (BFCL's "Error during inference", which its checker scores
wrong) is left out of its category's correct and total, counted under failed_requests and named:
the single-turn part gets this from base, the multi-turn part marks its traces, the agentic part
takes such entries out of `bfcl evaluate`'s counts (read back from the result files, with the
score file telling which of the entries it leaves out the checker scored correct). The memory
entries of a scenario whose prerequisite entry failed its request ran against an incomplete
memory and are left out the same way. A category with nothing left in the score is unscored
(null, never 0), and a run with nothing left withholds its score.

Upstream runs each memory scenario's prerequisite conversations (`*_prereq` entries, written to
BFCL_v4_<category>_prereq_result.json) before the scenario's questions, records a request that
raised as "Error during inference: ..." and goes on; such an entry never saves its memory
snapshot, so the questions run against an incomplete memory. They are not scored entries: they
stay out of failed_requests and are counted apart (memory_prereq_failed_requests,
bfcl_report.prereq_inference_errors), make the result and the Memory setting
completed_with_errors and add a named non-comparability reason; the scored entries of their
scenario are harness errors, left out of the score (memory_prereq_failed). bfcl carries them
from its agentic part.

docker, the harness CLI, the endpoint and the network checks are stubbed; nothing contacts a
model or Gemini, and every file is written under tmp_path.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG

REPO = Path(__file__).resolve().parents[1]
ERR_CONN = "Error during inference: Connection error."
ERR_502 = "Error during inference: Error code: 502 - bad gateway"
ERR_TIMEOUT = "Error during inference: Request timed out."
ERR_CONTEXT = ("Error during inference: Error code: 400 - {'message': \"This model's maximum "
               "context length is 8192 tokens\"}")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                 "GBENCH_GEMINI_RELAY_URL", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
                 "https_proxy", "NO_PROXY", "no_proxy", "GBENCH_SEARCH_MODELS",
                 "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL",
                 "GBENCH_BFCL_TIMEOUT_S", "BFCL_TIMEOUT_S", "GBENCH_SHARD", "BFCL_PROJECT_ROOT",
                 "GBENCH_BFCL_PROJECT_ROOT", "GBENCH_BFCL_TEMPERATURE",
                 "GBENCH_BFCL_REUSE_GENERATIONS", "BFCL_REUSE_GENERATIONS",
                 AG.SEARCH_LOG_ENV, AG.GUARD_LOG_ENV):
        monkeypatch.delenv(name, raising=False)
    AG._RELAY_HOSTS_OK.clear()
    yield
    AG._RELAY_HOSTS_OK.clear()


# --------------------------------------------------------------------------- #
# the harness' files, where upstream writes them
# --------------------------------------------------------------------------- #
def _memory_dir(root: Path, model: str, category: str) -> Path:
    d = root / model.replace("/", "_") / "agentic" / "memory" / category[len("memory_"):]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_results(root: Path, model: str, category: str, rows, prereq: bool = False) -> Path:
    path = _memory_dir(root, model, category) / (
        f"BFCL_v4_{category}{'_prereq' if prereq else ''}_result.json")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_scores(root: Path, model: str, category: str, rows, wrong) -> Path:
    """What upstream's agentic_runner + save_eval_results write: a header with the counts, then
    one line per entry the checker scored wrong (an inference error always is: its result is not
    a list)."""
    invalid = [r["id"] for r in rows
               if not isinstance(r.get("result"), list) or r["id"] in wrong]
    total = len(rows)
    correct = total - len(invalid)
    lines = [{"accuracy": correct / total if total else 0, "correct_count": correct,
              "total_count": total}]
    lines += [{"id": i, "model_name": model, "test_category": category, "valid": False,
               "error": {"error_type": "agentic:answer_not_found", "error_message": ["no"]}}
              for i in invalid]
    path = _memory_dir(root, model, category) / f"BFCL_v4_{category}_score.json"
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    return path


def _q(category: str, i: int, scenario: str = "customer") -> str:
    # upstream: "memory_<i>-<scenario>-<j>" with "memory" replaced by the category
    return f"{category}_{i}-{scenario}-{i}"


def _p(category: str, i: int, scenario: str = "customer") -> str:
    return f"{category}_prereq_{i}-{scenario}-{i}"


def _ok(entry_id: str):
    return {"id": entry_id, "result": [["answer"]]}


def _err(entry_id: str, text: str = ERR_CONN):
    return {"id": entry_id, "result": text}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic with the harness CLI stubbed at its files: `bfcl generate` writes
    state["rows"] / state["prereq"] ({category: rows}) as upstream's result files, `bfcl
    evaluate` writes each evaluated category's score file from them, scoring wrong an inference
    error and every id in state["wrong"]. The score files are then read by gbench's own code."""
    root = tmp_path / "bfcl"
    state = {"rows": {}, "prereq": {}, "wrong": set()}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.MEMORY_CATEGORIES)))

    def cli(args, env, timeout, model="", backend=None):
        if args[0] == "generate":
            for cat, rows in state["rows"].items():
                _write_results(root / "result", model, cat, rows)
            for cat, rows in state["prereq"].items():
                _write_results(root / "result", model, cat, rows, prereq=True)
        elif args[0] == "evaluate":
            cats = args[args.index("--test-category") + 1].split(",")
            for cat in cats:
                _write_scores(root / "score", model, cat, state["rows"].get(cat) or [],
                              state["wrong"])
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    state["root"] = root
    return state


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_the_memory_scenario_is_read_from_upstreams_ids():
    assert AG.memory_scenario("memory_kv_7-healthcare-3") == "healthcare"
    assert AG.memory_scenario("memory_rec_sum_prereq_10-healthcare-0") == "healthcare"
    assert AG.memory_scenario("web_search_base_4") is None


def test_the_harness_error_kinds_are_bases():
    assert AG.INFERENCE_ERROR_KINDS == {"failed": "endpoint_failure",
                                        "timed_out": "endpoint_timeout",
                                        "over_context": "over_context"}
    assert set(AG.INFERENCE_ERROR_KINDS.values()) == set(base.ENDPOINT_FAILURE_KINDS.values())


def test_harness_error_entries_read_the_result_and_score_files(tmp_path):
    cat = "memory_kv"
    rows = [_ok(_q(cat, 0)), _ok(_q(cat, 1)),                       # customer: prereq failed
            _ok(_q(cat, 2, "finance")), _err(_q(cat, 3, "finance"), ERR_TIMEOUT)]
    _write_results(tmp_path / "result", "org/m", cat, rows)
    _write_results(tmp_path / "result", "org/m", cat,
                   [_err(_p(cat, 0), ERR_502), _ok(_p(cat, 1)), _ok(_p(cat, 2, "finance"))],
                   prereq=True)
    # the checker scored customer-1 wrong and customer-0 correct
    _write_scores(tmp_path / "score", "org/m", cat, rows, wrong={_q(cat, 1)})
    got = AG.harness_error_entries(tmp_path / "result", tmp_path / "score", "org/m", [cat])
    assert got == {cat: {"entries": {_q(cat, 3, "finance"): "endpoint_timeout",
                                     _q(cat, 0): "memory_prereq_failed",
                                     _q(cat, 1): "memory_prereq_failed"},
                         "correct": 1}}
    per_cat = AG._parse_scores(tmp_path / "score", "org/m", [cat])
    assert per_cat[cat]["correct"] == 2 and per_cat[cat]["total"] == 4
    left, unscored = AG._leave_out(per_cat, got, "bfcl_v4_agentic")
    # one entry is left in the score (finance-2, correct)
    assert left[cat] == {"correct": 1, "total": 1, "accuracy": 100.0} and unscored == []


def test_an_entry_whose_own_request_failed_is_not_also_a_dependent(tmp_path):
    cat = "memory_vector"
    rows = [_err(_q(cat, 0), ERR_CONTEXT), _ok(_q(cat, 1))]
    _write_results(tmp_path / "result", "m", cat, rows)
    _write_results(tmp_path / "result", "m", cat, [_err(_p(cat, 0))], prereq=True)
    _write_scores(tmp_path / "score", "m", cat, rows, wrong=set())
    got = AG.harness_error_entries(tmp_path / "result", tmp_path / "score", "m", [cat])
    assert got[cat]["entries"] == {_q(cat, 0): "over_context", _q(cat, 1): "memory_prereq_failed"}
    assert got[cat]["correct"] == 1


def test_files_that_disagree_are_a_hard_error_never_a_guessed_score():
    per_cat = {"memory_kv": {"correct": 1, "total": 1, "accuracy": 100.0}}
    left_out = {"memory_kv": {"entries": {"a": "endpoint_failure", "b": "endpoint_failure"},
                              "correct": 0}}
    with pytest.raises(RuntimeError, match="disagree"):
        AG._leave_out(per_cat, left_out, "bfcl_v4_agentic")


def test_a_group_with_an_unscored_category_has_no_mean():
    per_cat = {"memory_kv": {"correct": 1, "total": 2}, "memory_vector": {"correct": 2,
                                                                          "total": 2},
               "web_search_base": {"correct": 1, "total": 1},
               "web_search_no_snippet": {"correct": 0, "total": 1}}
    scores = AG.agentic_scores(per_cat, unscored=["memory_rec_sum"])
    assert scores["summary_memory"] is None and scores["overall_accuracy_agentic"] is None
    assert scores["summary_web_search"] == 50.0
    # a group missing a category that did not run has no mean either (as the bfcl suite's
    # columns); BFCL's own value counts the missing category as 0
    assert AG.agentic_scores(per_cat)["summary_memory"] is None
    assert AG.bfcl_zero_filled_overall(per_cat) == round(((50 + 100 + 0) / 3 + 50) / 2, 2)


# --------------------------------------------------------------------------- #
# the multi-turn part
# --------------------------------------------------------------------------- #
def _mt(i, status="OK", correct=True, cat="multi_turn_base", **extra):
    return {"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0, "category": cat,
            "question_id": f"{cat}_{i}", "is_correct": correct, "status": status,
            "error": None if status == "OK" else f"{status} error",
            "health": "ok" if status == "OK" else "request_failed", "steps": 1, **extra}


def test_multi_turn_harness_errors_are_left_out_of_the_score():
    traces = [_mt(0), _mt(1, correct=False), _mt(2, "FAILED", False),
              _mt(3, "TIMEOUT", False), _mt(4, "OVER_CONTEXT", False),
              _mt(5, "TIMEOUT", False, harness_timeout=True),
              _mt(6, "FAILED", False, cat="multi_turn_miss_func")]
    counts = bfcl._multi_turn_counts(traces)
    assert counts["category_accuracy"] == {"multi_turn_base": {"correct": 1, "total": 2,
                                                               "accuracy": 50.0}}
    assert counts["categories_unscored"] == ["multi_turn_miss_func"]
    assert counts["failed_requests"] == 5 and counts["timed_out_requests"] == 2
    assert counts["over_context_prompts"] == 1 and counts["scoring_excluded"] == 5
    assert counts["request_harness_errors"] == {"endpoint_failure": 2, "endpoint_timeout": 1,
                                                "over_context": 1, "unfinished": 1}
    assert counts["harness_errors_left_out"] == {
        "multi_turn_base": {"endpoint_timeout": 1, "over_context": 1, "unfinished": 1,
                            "endpoint_failure": 1},
        "multi_turn_miss_func": {"endpoint_failure": 1}}
    kinds = {t["sample_idx"]: t["harness_error"]["kind"] for t in traces
             if t.get("scoring_excluded")}
    assert kinds == {2: "endpoint_failure", 3: "endpoint_timeout", 4: "over_context",
                     5: "unfinished", 6: "endpoint_failure"}
    assert not any(t.get("scoring_excluded") for t in traces[:2])


# --------------------------------------------------------------------------- #
# the agentic part, scored from the harness' files
# --------------------------------------------------------------------------- #
def test_agentic_entries_whose_request_failed_leave_the_score(harness):
    cat = "memory_kv"
    harness["rows"] = {cat: [_ok(_q(cat, 0)), _ok(_q(cat, 1)), _err(_q(cat, 2)),
                             _err(_q(cat, 3), ERR_TIMEOUT)]}
    harness["wrong"] = {_q(cat, 1)}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1,
                                 eval_categories=cat)
    # upstream would report 1/4; the two failed requests are harness errors, left out
    assert res["category_accuracy"][cat] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert res["accuracy"] == 50.0 and res["accuracy_pooled"] == 50.0
    assert res["total_questions"] == 4 and res["effective_n"] == 2
    assert res["failed_requests"] == 2 and res["timed_out_requests"] == 1
    assert res["scoring_excluded"] == 2
    assert res["request_harness_errors"] == {"endpoint_failure": 1, "endpoint_timeout": 1}
    assert res["bfcl_report"]["harness_errors_left_out"] == {
        cat: {"endpoint_failure": 1, "endpoint_timeout": 1}}
    assert res["status"] == "completed_with_errors"
    assert ("2 scored entries whose request failed (endpoint_failure: 1, endpoint_timeout: 1; "
            "bfcl_report.inference_errors) left out of the score as harness errors (upstream "
            "scores them wrong)") in res["leaderboard_comparable_reason"]
    memory = res["settings"]["memory"]
    assert memory["total_questions"] == 4 and memory["correct_answers"] == 1
    assert memory["scoring_excluded"] == 2 and memory["failed_requests"] == 2
    # the score-file traces of those entries say why they are out of the score
    marked = {t["question_id"]: t for t in res["sample_traces"]}
    assert marked[_q(cat, 2)]["scoring_excluded"] and marked[_q(cat, 2)]["status"] == "FAILED"
    assert marked[_q(cat, 3)]["harness_error"]["kind"] == "endpoint_timeout"
    assert marked[_q(cat, 3)]["status"] == "TIMEOUT"
    assert not marked[_q(cat, 1)].get("scoring_excluded")          # the model's own wrong answer


def test_memory_entries_of_a_scenario_whose_prerequisite_failed_leave_the_score(harness):
    cat = "memory_kv"
    harness["rows"] = {cat: [_ok(_q(cat, 0)), _ok(_q(cat, 1)),
                             _ok(_q(cat, 2, "finance")), _ok(_q(cat, 3, "finance"))]}
    harness["prereq"] = {cat: [_err(_p(cat, 0), ERR_502), _ok(_p(cat, 1)),
                               _ok(_p(cat, 2, "finance"))]}
    harness["wrong"] = {_q(cat, 1), _q(cat, 3, "finance")}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1,
                                 eval_categories=cat)
    # the customer entries ran against an incomplete memory: out, whatever the checker said
    # (customer-0 scored correct comes off the correct count too)
    assert res["category_accuracy"][cat] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert res["request_harness_errors"] == {"memory_prereq_failed": 2}
    assert res["memory_prereq_failed_requests"] == 1 and res["failed_requests"] == 0
    assert res["status"] == "completed_with_errors"
    assert res["settings"]["memory"]["status"] == "completed_with_errors"
    assert "the memory entries of those scenarios ran against an incomplete memory and are " \
           "left out of the score (harness errors)" in res["leaderboard_comparable_reason"]
    marked = {t["question_id"]: t for t in res["sample_traces"]}
    assert marked[_q(cat, 1)]["harness_error"]["kind"] == "memory_prereq_failed"
    assert marked[_q(cat, 1)]["status"] == "OK"       # its own request did not fail


def test_an_agentic_run_with_nothing_left_in_the_score_withholds_it(harness):
    cat = "memory_kv"
    harness["rows"] = {cat: [_ok(_q(cat, 0)), _err(_q(cat, 1))]}
    harness["prereq"] = {cat: [_err(_p(cat, 0))]}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1,
                                 eval_categories=cat)
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["accuracy_pooled"] is None and res["category_accuracy"] == {}
    assert res["categories_unscored"] == [cat]
    assert res["accuracy_withheld_reason"].startswith(
        "every one of the 2 scored entries failed its request, was not finished or ran against "
        "an incomplete memory")
    assert "score withheld" in res["leaderboard_comparable_reason"]


def test_an_unscored_category_leaves_bfcls_means_null_and_the_pooled_rate_headlines(harness):
    harness["rows"] = {"memory_kv": [_err(_q("memory_kv", 0)), _err(_q("memory_kv", 1))],
                       "memory_vector": [_ok(_q("memory_vector", 0)),
                                         _ok(_q("memory_vector", 1))],
                       "memory_rec_sum": [_ok(_q("memory_rec_sum", 0)),
                                          _ok(_q("memory_rec_sum", 1))]}
    harness["wrong"] = {_q("memory_rec_sum", 1)}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1,
                                 eval_categories="memory_kv,memory_vector,memory_rec_sum")
    assert "memory_kv" not in res["category_accuracy"]
    assert res["categories_unscored"] == ["memory_kv"]
    assert res["summary_memory"] is None and res["overall_accuracy_agentic"] is None
    assert res["accuracy"] == res["accuracy_pooled"] == 75.0
    assert res["metric"].startswith("accuracy_pooled") and "memory_kv left unscored" in \
        res["metric"]
    assert res["settings"]["memory"]["categories_unscored"] == ["memory_kv"]
    assert "memory_kv left unscored" in res["leaderboard_comparable_reason"]
    assert res["status"] == "completed_with_errors"


# --------------------------------------------------------------------------- #
# bfcl: every part applies the same rule
# --------------------------------------------------------------------------- #
def _bfcl_parts(monkeypatch, mt_traces):
    """run_bfcl's single-turn and multi-turn parts stubbed at the harness boundary (the agentic
    part runs through the `harness` fixture)."""
    single = [("simple_python", "q-ok-0"), ("simple_python", "q-fail-1"),
              ("simple_python", "q-ok-2"), ("irrelevance", "q-fail-3")]
    samples = [([{"role": "user", "content": text}],
                {"id": f"{cat}_{i}", "category": cat, "ground_truth": []}, {"category": cat})
               for i, (cat, text) in enumerate(single)]
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single_: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": t["question_id"], "category": t["category"]}
                                 for t in mt_traces]))
    monkeypatch.setattr(bfcl, "_load_bfcl_samples", lambda cats: (
        [s for s in samples if s[2]["category"] in cats], {s[1]["id"]: [] for s in samples}))
    monkeypatch.setattr(bfcl, "check_single_turn", lambda items, eval_name="bfcl": [
        {"valid": True, "error_type": None, "error": []} for _ in items])
    monkeypatch.setattr(bfcl, "endpoint_network", lambda url: None)
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", lambda *a, **k: {
        "traces": [dict(t) for t in mt_traces], "guard_refusals": [], "harness_error": None,
        "endpoint_relay": None})
    monkeypatch.setattr(bfcl, "_check_agentic", lambda url, cats: None)

    async def send(**kw):
        if "q-fail" in json.dumps(kw.get("messages") or []):
            return base.Reply(None, None, None, error="HTTP 500")
        return base.Reply("", [{"id": "c", "type": "function",
                                "function": {"name": "f", "arguments": "{}"}}], "tool_calls",
                          completion_tokens=3)
    return send


def test_bfcl_leaves_harness_errors_out_of_every_part(harness, monkeypatch):
    cat = "memory_kv"
    harness["rows"] = {cat: [_ok(_q(cat, 0)), _ok(_q(cat, 1, "finance")),
                             _err(_q(cat, 2, "finance"))]}
    harness["prereq"] = {cat: [_err(_p(cat, 0))]}
    send = _bfcl_parts(monkeypatch, [_mt(0), _mt(1, correct=False), _mt(2, "FAILED", False),
                                     _mt(3, "TIMEOUT", False, cat="multi_turn_miss_func")])
    with mock.patch.object(base, "_send_single_request", send):
        res = bfcl.run_bfcl("m", "http://127.0.0.1:9/v1", 1, eval_categories=(
            "simple_python,irrelevance,multi_turn_base,multi_turn_miss_func,memory_kv"))
    ca = res["category_accuracy"]
    # single-turn: the failed request is out (base); irrelevance had only a failed one
    assert ca["simple_python"] == {"correct": 2, "total": 2, "accuracy": 100.0}
    # multi-turn: the FAILED trace is out; miss_func had only a timed-out one
    assert ca["multi_turn_base"] == {"correct": 1, "total": 2, "accuracy": 50.0}
    # agentic: the failed entry and the customer entry (its prerequisite failed) are out
    assert ca[cat]["correct"] == 1 and ca[cat]["total"] == 1
    assert res["categories_unscored"] == ["irrelevance", "multi_turn_miss_func"]
    assert not any(c in ca for c in res["categories_unscored"])
    assert res["bfcl_summary"]["non_live"]["Irrelevance Detection"] is None
    assert res["request_harness_errors"] == {"endpoint_failure": 4, "endpoint_timeout": 1,
                                             "memory_prereq_failed": 1}
    assert res["scoring_excluded"] == 6 and res["failed_requests"] == 5
    assert res["effective_n"] == 5 and res["total_questions"] == 11
    assert res["memory_prereq_failed_requests"] == 1
    assert res["status"] == "completed_with_errors" and res["accuracy"] == res["pooled_accuracy"]
    assert res["pooled_accuracy"] == 80.0
    reason = res["leaderboard_comparable_reason"]
    assert ("4 single-turn and multi-turn entries left out of the score as harness errors "
            "(endpoint_failure: 3, endpoint_timeout: 1; BFCL scores a failed request wrong)"
            ) in reason
    assert "categories left unscored, every entry a harness error: irrelevance, " \
           "multi_turn_miss_func" in reason
    assert "agentic: " in reason and "1 scored entry whose request failed" in reason
    assert "harness errors left out of the score: 6 entries" in res["status_reason"]
    mt = {t["question_id"]: t for t in res["sample_traces"]
          if t.get("category", "").startswith("multi_turn")}
    assert mt["multi_turn_base_2"]["scoring_excluded"]
    assert mt["multi_turn_base_2"]["harness_error"]["kind"] == "endpoint_failure"
    assert res["multi_turn_report"]["harness_errors_left_out"] == {
        "multi_turn_base": {"endpoint_failure": 1},
        "multi_turn_miss_func": {"endpoint_timeout": 1}}
    assert res["agentic_report"]["scoring_excluded"] == 2


def test_bfcl_status_is_failed_when_every_entry_of_every_part_is_a_harness_error(
        harness, monkeypatch):
    cat = "memory_kv"
    harness["rows"] = {cat: [_err(_q(cat, 0))]}
    send = _bfcl_parts(monkeypatch, [_mt(0, "FAILED", False)])
    with mock.patch.object(base, "_send_single_request", send):
        res = bfcl.run_bfcl("m", "http://127.0.0.1:9/v1", 1,
                            eval_categories="irrelevance,multi_turn_base,memory_kv")
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["category_accuracy"] == {}
    assert res["accuracy_withheld_reason"].startswith("nothing measured the model")
    assert res["categories_unscored"] == ["irrelevance", cat, "multi_turn_base"]


# --------------------------------------------------------------------------- #
# the guides say what the code does
# --------------------------------------------------------------------------- #
def test_the_guides_describe_the_harness_error_rule():
    guide = (REPO / "docs" / "evals" / "bfcl.md").read_text(encoding="utf-8")
    agentic = (REPO / "docs" / "evals" / "bfcl_v4_agentic.md").read_text(encoding="utf-8")
    flat = " ".join(guide.split())
    flat_ag = " ".join(agentic.split())
    assert "(scored wrong) while the run goes on" not in flat
    assert "and scores wrong, as BFCL records it as an inference error" not in flat
    assert "Harness errors are never the model's outcome" in flat
    assert "`request_harness_errors`" in flat and "`categories_unscored`" in flat
    assert "the dependent memory entries were scored against an incomplete memory" not in flat_ag
    assert "the unfinished ones count as wrong" not in flat_ag
    assert "`memory_prereq_failed`" in flat_ag and "`harness_errors_left_out`" in flat_ag
    assert "as the model's own outcome" not in flat_ag and "as the model's own outcome" not in flat


# --------------------------------------------------------------------------- #
# memory prerequisite entries whose request failed
# --------------------------------------------------------------------------- #
def _prereq_id(category: str, i: int, scenario: str = "customer") -> str:
    # upstream: "memory_prereq_<i>-<scenario>-<i>" with "memory" replaced by the category
    return f"{category}_prereq_{i}-{scenario}-{i}"


# --------------------------------------------------------------------------- #
# reading the prerequisite entries back
# --------------------------------------------------------------------------- #
def test_prereq_errors_are_read_from_upstreams_prereq_file(tmp_path):
    _write_results(tmp_path, "org/m", "memory_kv", [
        {"id": _prereq_id("memory_kv", 0), "result": ERR_CONN},
        {"id": _prereq_id("memory_kv", 1), "result": [["ok"]]},
        {"id": _prereq_id("memory_kv", 2), "result": ERR_TIMEOUT},
        {"id": _prereq_id("memory_kv", 3), "result": ERR_CONTEXT},
        {"id": _prereq_id("memory_kv", 4), "result": ERR_502}], prereq=True)
    _write_results(tmp_path, "org/m", "memory_kv", [
        {"id": "memory_kv_0-customer-0", "result": ERR_CONN},
        {"id": "memory_kv_1-customer-1", "result": [["ok"]]}])
    _write_results(tmp_path, "org/m", "memory_vector", [
        {"id": _prereq_id("memory_vector", 0), "result": [["ok"]]}], prereq=True)
    cats = ["memory_kv", "memory_vector"]
    assert AG.prereq_inference_errors(tmp_path, "org/m", cats) == {
        "memory_kv": {"failed": 2, "timed_out": 1, "over_context": 1}}
    # the scored entries' count is unchanged: the prerequisite file is not read for it
    assert AG.inference_errors(tmp_path, "org/m", cats) == {
        "memory_kv": {"failed": 1, "timed_out": 0, "over_context": 0}}


def test_a_prereq_row_in_the_main_file_is_counted_once(tmp_path):
    row = {"id": _prereq_id("memory_rec_sum", 0), "result": ERR_CONN}
    _write_results(tmp_path, "m", "memory_rec_sum", [row], prereq=True)
    _write_results(tmp_path, "m", "memory_rec_sum",
                   [row, {"id": _prereq_id("memory_rec_sum", 1), "result": ERR_502},
                    {"id": "memory_rec_sum_0-customer-0", "result": [["ok"]]}])
    assert AG.prereq_inference_errors(tmp_path, "m", ["memory_rec_sum"]) == {
        "memory_rec_sum": {"failed": 2, "timed_out": 0, "over_context": 0}}
    assert AG.inference_errors(tmp_path, "m", ["memory_rec_sum"]) == {}


def test_only_memory_categories_have_prerequisites(tmp_path):
    d = tmp_path / "m" / "agentic"
    d.mkdir(parents=True)
    (d / "BFCL_v4_web_search_base_prereq_result.json").write_text(
        json.dumps({"id": "web_search_base_prereq_0", "result": ERR_CONN}) + "\n",
        encoding="utf-8")
    assert AG.prereq_inference_errors(tmp_path, "m", ["web_search_base"]) == {}


def test_the_reason_names_the_count_and_the_consequence():
    one = AG.prereq_failure_reason({"memory_kv": {"failed": 1, "timed_out": 0,
                                                  "over_context": 0}})
    assert one.startswith("1 memory prerequisite entry failed its request (memory_kv: 1; "
                          "bfcl_report.prereq_inference_errors)")
    many = AG.prereq_failure_reason({
        "memory_vector": {"failed": 0, "timed_out": 2, "over_context": 0},
        "memory_kv": {"failed": 1, "timed_out": 0, "over_context": 1}})
    assert many.startswith("4 memory prerequisite entries failed their request (memory_kv: 2, "
                           "memory_vector: 2; ")
    assert many.endswith("the memory entries of those scenarios ran against an incomplete "
                         "memory and are left out of the score (harness errors)")


# --------------------------------------------------------------------------- #
# run_bfcl_v4_agentic with the harness CLI stubbed
# --------------------------------------------------------------------------- #
@pytest.fixture
def agentic(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic with the harness CLI stubbed. Each `bfcl generate` writes
    `state["rows"]` ({category: rows}, scored entries) and `state["prereq"]` ({category: rows},
    the prerequisite entries, in their own file) where upstream writes them; `state["relay"]`
    is the endpoint relay record it appends; `state["scores"]` the per-category scores
    evaluate reports."""
    root = tmp_path / "bfcl"
    state = {"rows": {}, "prereq": {}, "relay": None, "scores": None}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.MEMORY_CATEGORIES)))

    def cli(args, env, timeout, model="", backend=None):
        if args[0] == "generate":
            for cat, rows in state["rows"].items():
                _write_results(root / "result", model, cat, rows)
            for cat, rows in state["prereq"].items():
                _write_results(root / "result", model, cat, rows, prereq=True)
            ctx = AG._suite()
            if ctx.relays is not None and state["relay"] is not None:
                ctx.relays.append({"command": "generate", "endpoint_relay": state["relay"],
                                   "gemini_relay": None})
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda r, m, cats: state["scores"] or {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    return state


def _ok_rows(cat, n=2):
    return [{"id": f"{cat}_{i}-customer-{i}", "result": [["ok"]]} for i in range(n)]


def test_failed_prereq_entries_without_a_relay_are_harness_errors(agentic):
    # a local endpoint without a real OPENAI_API_KEY: no relay, so no endpoint_failures
    agentic["rows"] = {"memory_kv": _ok_rows("memory_kv")}
    agentic["scores"] = {"memory_kv": {"correct": 2, "total": 2, "accuracy": 100.0},
                         "memory_vector": {"correct": 1, "total": 2, "accuracy": 50.0},
                         "memory_rec_sum": {"correct": 1, "total": 2, "accuracy": 50.0}}
    agentic["prereq"] = {"memory_kv": [
        {"id": _prereq_id("memory_kv", 0), "result": ERR_CONN},
        {"id": _prereq_id("memory_kv", 1), "result": ERR_502},
        {"id": _prereq_id("memory_kv", 2), "result": [["ok"]]}]}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["status"] == "completed_with_errors"
    assert res["failed_requests"] == 0 and res["timed_out_requests"] == 0
    assert res["memory_prereq_failed_requests"] == 2
    assert res["harness_errors"] == [] and "endpoint_failures" not in res
    report = res["bfcl_report"]
    assert report["inference_errors"] == {}
    assert report["prereq_inference_errors"] == {
        "memory_kv": {"failed": 2, "timed_out": 0, "over_context": 0}}
    assert res["leaderboard_comparable"] is False
    assert ("2 memory prerequisite entries failed their request (memory_kv: 2; "
            "bfcl_report.prereq_inference_errors); the memory entries of those scenarios ran "
            "against an incomplete memory and are left out of the score (harness errors)") in \
        res["leaderboard_comparable_reason"]
    memory = res["settings"]["memory"]
    assert memory["status"] == "completed_with_errors"
    assert memory["prereq_failed_requests"] == 2 and memory["failed_requests"] == 0
    # both scored memory_kv entries are of the scenario whose prerequisites failed: harness
    # errors, left out of the score, so memory_kv is unscored (never 0) and the summaries that
    # weigh it are null; the headline is the pooled rate over the entries left in the score
    assert report["harness_errors_left_out"] == {"memory_kv": {"memory_prereq_failed": 2}}
    assert res["scoring_excluded"] == 2 and memory["scoring_excluded"] == 2
    assert "memory_kv" not in res["category_accuracy"]
    assert res["categories_unscored"] == ["memory_kv"]
    assert memory["categories_unscored"] == ["memory_kv"] and memory["summary"] is None
    assert res["summary_memory"] is None and res["overall_accuracy_agentic"] is None
    assert res["accuracy"] == res["accuracy_pooled"] == 50.0
    assert res["metric"].startswith("accuracy_pooled") and "memory_kv left unscored" in \
        res["metric"]
    assert res["status"] == "completed_with_errors" and "accuracy_withheld_reason" not in res


def test_a_clean_run_has_no_prereq_failures(agentic):
    agentic["rows"] = {"memory_kv": _ok_rows("memory_kv")}
    agentic["prereq"] = {"memory_kv": [{"id": _prereq_id("memory_kv", 0), "result": [["ok"]]}]}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["status"] == "success"
    assert res["memory_prereq_failed_requests"] == 0
    assert res["bfcl_report"]["prereq_inference_errors"] == {}
    assert res["settings"]["memory"]["status"] == "success"
    assert res["settings"]["memory"]["prereq_failed_requests"] == 0
    assert "memory prerequisite" not in (res["leaderboard_comparable_reason"] or "")


def test_prereq_failures_stay_out_of_the_withheld_check(agentic):
    # every scored entry failed its request: the score is withheld over the scored entries
    # alone, and the prerequisite failures are counted apart
    agentic["rows"] = {c: [{"id": f"{c}_{i}-customer-{i}", "result": ERR_CONN}
                           for i in range(2)] for c in AG.MEMORY_CATEGORIES}
    agentic["prereq"] = {"memory_vector": [{"id": _prereq_id("memory_vector", 0),
                                            "result": ERR_TIMEOUT}]}
    agentic["scores"] = {c: {"correct": 0, "total": 2, "accuracy": 0.0}
                         for c in AG.MEMORY_CATEGORIES}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["failed_requests"] == 6 and res["timed_out_requests"] == 0
    assert res["accuracy_withheld_reason"].startswith("every one of the 6 scored entries")
    assert res["memory_prereq_failed_requests"] == 1
    assert "1 memory prerequisite entry failed its request (memory_vector: 1; " in \
        res["leaderboard_comparable_reason"]


def test_prereq_failures_and_relay_failures_are_both_named(agentic):
    agentic["relay"] = {"upstream_errors": {"http_503": 3}, "requests_refused": 0}
    agentic["rows"] = {"memory_kv": _ok_rows("memory_kv")}
    agentic["scores"] = {"memory_kv": {"correct": 2, "total": 2, "accuracy": 100.0},
                         "memory_vector": {"correct": 1, "total": 2, "accuracy": 50.0},
                         "memory_rec_sum": {"correct": 1, "total": 2, "accuracy": 50.0}}
    agentic["prereq"] = {"memory_kv": [{"id": _prereq_id("memory_kv", 0), "result": ERR_502}]}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    reason = res["leaderboard_comparable_reason"]
    assert "the served endpoint failed 3 relayed model request(s) (http_503: 3)" in reason
    assert "1 memory prerequisite entry failed its request" in reason
    # the scored entries of that scenario are left out of the score (memory_kv unscored)
    assert res["status"] == "completed_with_errors" and res["scoring_excluded"] == 2
    assert "memory_kv left unscored" in reason


# --------------------------------------------------------------------------- #
# bfcl carries its agentic part's prerequisite failures
# --------------------------------------------------------------------------- #
def test_bfcl_status_counts_prereq_failures():
    base = {"total_questions": 4, "failed_requests": 0, "truncated_responses": 0,
            "sample_traces": []}
    assert bfcl._status(dict(base)) == "success"
    assert bfcl._status({**base, "memory_prereq_failed_requests": 0}) == "success"
    assert bfcl._status({**base, "memory_prereq_failed_requests": 3}) == "completed_with_errors"


def test_bfcl_reports_its_agentic_parts_prereq_failures(agentic, monkeypatch):
    monkeypatch.setattr(bfcl, "_check_agentic", lambda url, cats: None)
    agentic["rows"] = {c: _ok_rows(c) for c in AG.MEMORY_CATEGORIES}
    agentic["scores"] = {c: {"correct": 2, "total": 2, "accuracy": 100.0}
                         for c in AG.MEMORY_CATEGORIES}
    agentic["prereq"] = {"memory_rec_sum": [
        {"id": _prereq_id("memory_rec_sum", 0), "result": ERR_CONN},
        {"id": _prereq_id("memory_rec_sum", 1), "result": ERR_CONTEXT}]}
    res = bfcl.run_bfcl("m", "http://127.0.0.1:9/v1", 1, eval_categories="memory")
    assert res["status"] == "completed_with_errors"
    assert res["failed_requests"] == 0 and res["memory_prereq_failed_requests"] == 2
    assert res["agentic_report"]["prereq_inference_errors"] == {
        "memory_rec_sum": {"failed": 1, "timed_out": 0, "over_context": 1}}
    assert res["agentic_report"]["settings"]["memory"]["status"] == "completed_with_errors"
    assert ("agentic: " in res["leaderboard_comparable_reason"]
            and "2 memory prerequisite entries failed their request (memory_rec_sum: 2; "
            in res["leaderboard_comparable_reason"])
    # memory_rec_sum's entries are all of that scenario: left unscored, never 0; the other
    # memory categories are scored
    assert "memory_rec_sum" not in res["category_accuracy"]
    assert res["categories_unscored"] == ["memory_rec_sum"]
    assert res["scoring_excluded"] == 2
    assert res["request_harness_errors"] == {"memory_prereq_failed": 2}
    assert res["accuracy"] is not None


def test_bfcl_without_prereq_failures_reports_zero(agentic, monkeypatch):
    monkeypatch.setattr(bfcl, "_check_agentic", lambda url, cats: None)
    agentic["rows"] = {c: _ok_rows(c) for c in AG.MEMORY_CATEGORIES}
    res = bfcl.run_bfcl("m", "http://127.0.0.1:9/v1", 1, eval_categories="memory")
    assert res["memory_prereq_failed_requests"] == 0
    assert res["agentic_report"]["prereq_inference_errors"] == {}
    assert "memory prerequisite" not in (res["leaderboard_comparable_reason"] or "")


# --------------------------------------------------------------------------- #
# the guides say so
# --------------------------------------------------------------------------- #
def test_the_guides_document_prereq_failures():
    agentic_doc = (REPO / "docs/evals/bfcl_v4_agentic.md").read_text(encoding="utf-8")
    bfcl_doc = (REPO / "docs/evals/bfcl.md").read_text(encoding="utf-8")
    for doc in (agentic_doc, bfcl_doc):
        assert "memory_prereq_failed_requests" in doc and "prereq_inference_errors" in doc
    assert "prereq_failed_requests" in agentic_doc.replace("memory_prereq_failed_requests", "")
    assert "BFCL_v4_<category>_prereq_result.json" in agentic_doc


def test_the_bfcl_guide_lists_every_kind_of_harness_errors_record():
    doc = (REPO / "docs/evals/bfcl.md").read_text(encoding="utf-8")
    start = doc.index("- `harness_errors`")
    bullet = doc[start:doc.index("\n- ", start + 1)]
    assert "`timeout_s`" in bullet and "`entries_failed`" in bullet
    assert "`categories_unscored`" in bullet and "no wall clock" in bullet
    assert "before any of its entries finished leaves its categories unscored" in \
        " ".join(bullet.split())
