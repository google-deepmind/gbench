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

"""LiveCodeBench and Codeforces keep each problem's tests out of the result.

A full LiveCodeBench release carries about 9.4 GB of tests and the Codeforces generated tests
about 20 GB; stored in every trace's gold they made the result files tens of GB. The tests stay in
a store the run owns and grades from; each trace's gold keeps the counts and sha256 of the tests.
The verdicts and everything a sharded run's merge recomputes stay in the traces.
"""

import hashlib
import json
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import codeforces as CF
from gbench.runners.eval_suites import lcb as L

_MARK_IN = "4242 1717"             # a test input no other field holds
_MARK_OUT = "5959"


def _lcb_rows():
    return [{"question_id": "abc100_a", "question_title": "Sum", "difficulty": "easy",
             "question_content": "Read two integers and print their sum.", "starter_code": "",
             "contest_date": "2025-03-01T00:00:00",
             "public_test_cases": json.dumps([{"input": "2 3", "output": "5",
                                               "testtype": "stdin"}]),
             "private_test_cases": json.dumps([{"input": _MARK_IN, "output": _MARK_OUT,
                                                "testtype": "stdin"}]),
             "metadata": "{}"}]


def test_the_loader_puts_the_tests_in_the_store_and_a_digest_in_the_gold():
    rows = _lcb_rows()
    store = {}
    with mock.patch.object(L, "_load_lcb_lite", return_value=rows), \
            mock.patch("datasets.load_dataset", return_value=[]):
        samples = L._load_lcb_samples(limit=1, tests_store=store)
    (messages, gold, meta), = samples
    assert "tests" not in gold and gold["tests_ref"] == "abc100_a"
    assert store["abc100_a"][1]["input"] == _MARK_IN
    assert gold["test_counts"] == {"public_test_cases": 1, "private_test_cases": 1}
    assert gold["tests_sha256"]["private_test_cases"] == hashlib.sha256(
        rows[0]["private_test_cases"].encode("utf-8")).hexdigest()
    assert gold["contest_date"].startswith("2025-03")
    # the meta goes out with the request: only the category and the sample_key's id
    assert meta == {"category": "code_gen_easy", "question_id": "abc100_a"}
    assert _MARK_IN not in json.dumps([messages, gold, meta])


def test_a_gold_whose_tests_were_not_resolved_is_never_graded_as_failed():
    gold = {"task": "generation", "question_id": "q", "tests_ref": "q", "fn_name": None}
    with pytest.raises(RuntimeError, match="not in this run's test store"):
        L._resolve_gold(gold, {})
    with pytest.raises(RuntimeError, match="were not resolved"):
        L._verify_single_lcb_sample("```python\nprint(5)\n```", gold)
    resolved = L._resolve_gold(gold, {"q": [{"input": "", "output": "5"}]})
    assert resolved["tests"] == [{"input": "", "output": "5"}] and "tests" not in gold


def _lcb_offline_run(monkeypatch, shard=None, n_problems=2):
    """run_lcb on stdin problems through the store: problem 0 answered wrongly first and
    repaired, the others right. Returns (result, the store the loader was given)."""
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    monkeypatch.delenv("GBENCH_LCB_TEMPERATURE", raising=False)
    if shard:
        monkeypatch.setenv("GBENCH_SHARD", shard)
    else:
        monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    seen = {}

    def loader(enable_thinking=False, limit=None, tests_store=None):
        seen["store"] = tests_store
        rows = []
        for i in range(n_problems):
            qid = f"p{i}"
            tests_store[qid] = [{"input": "2 3", "output": "5"},
                                {"input": _MARK_IN, "output": _MARK_OUT}]
            q = f"Problem {i}: print the sum of two integers."
            rows.append(([{"role": "system", "content": L._SYSTEM_MESSAGE_GENERIC},
                          {"role": "user", "content": L._generation_prompt(q, "")}],
                         {"task": "generation", "question_id": qid, "fn_name": None,
                          "question_content": q, "contest_date": f"2025-0{i + 1}-01",
                          "tests_ref": qid, "test_counts": {"public_test_cases": 1,
                                                            "private_test_cases": 1}},
                         {"category": "code_gen_easy", "question_id": qid}))
        return L._shard_rows(rows, 0)

    monkeypatch.setattr(L, "_load_lcb_samples", loader)
    good = "```python\nprint(sum(map(int, input().split())))\n```"

    async def fake_send(**k):
        msgs = k["messages"]
        if msgs[0]["content"] == L._SR_SYSTEM_MESSAGE_GENERIC:
            return base.Reply("fixed\n" + good, None, "stop", completion_tokens=5)
        wrong = "Problem 0:" in msgs[1]["content"]
        return base.Reply("```python\nprint(6)\n```" if wrong else good, None, "stop",
                          completion_tokens=5)

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    return L.run_lcb("m", "http://x/v1", 4), seen["store"]


def test_lcb_grades_from_the_store_and_the_result_holds_no_test(monkeypatch):
    res, store = _lcb_offline_run(monkeypatch)
    assert res["scenarios"]["code_generation"] == {"correct": 10, "total": 20, "accuracy": 50.0}
    assert res["self_repair"]["accuracy"] == 100.0 and res["self_repair"]["repair_requests"] == 10
    text = json.dumps(res)
    assert _MARK_IN not in text and _MARK_OUT not in text
    assert all("tests" not in t["gold_answer"] for t in res["sample_traces"])
    assert store == {}                                  # emptied when the run ended
    assert res["contest_date_range"] == ["2025-01", "2025-02"]
    assert "not stored" in res["tests_in_traces"]


def test_lcb_merge_of_two_shards_equals_the_unsharded_run(monkeypatch):
    from gbench import merge
    full, _ = _lcb_offline_run(monkeypatch, n_problems=3)
    shards = [_lcb_offline_run(monkeypatch, shard=f"{i}/2", n_problems=3)[0] for i in (1, 2)]
    assert [s["shard"] for s in shards] == ["1/2", "2/2"]
    merged = merge.merge_suite("lcb", shards)
    for field in ("accuracy", "correct_answers", "total_questions", "effective_n",
                  "blended_all_scenarios_accuracy", "lcb_scenarios", "contest_date_range",
                  "replies_cut_off_by_budget"):
        assert merged[field] == full[field], field
    assert merged["scenarios"] == full["scenarios"]
    assert merged["self_repair"]["accuracy"] == full["self_repair"]["accuracy"]
    assert merged["self_repair"]["repair_requests"] == full["self_repair"]["repair_requests"]
    assert merged["attempts"]["pass_at_k"] == full["attempts"]["pass_at_k"]
    assert merged["attempts"]["samples"] == 3
    assert merged["lcb_scenarios_not_run"] == full["lcb_scenarios_not_run"]
    assert merged["leaderboard_comparable_reason"] == full["leaderboard_comparable_reason"]
    assert "sharded run" not in merged["leaderboard_comparable_reason"]
    for field in ("scenarios", "self_repair", "lcb_scenarios"):
        assert field not in (merged.get("merge_unreconciled_fields") or {})


def test_lcb_scenarios_list_what_ran_and_name_why_the_others_did_not(monkeypatch):
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_TEST_GEN", raising=False)
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    traces = [{"category": "code_gen_easy"}]
    assert L._scenarios_not_run(traces, True, 4) == {
        "code_execution": "dropped by --eval-limit 4",
        "test_output_prediction": "dropped by --eval-limit 4"}
    reason = L._scenarios_reason(L._scenarios_not_run(traces, True, 4))
    assert reason == ("scenario(s) not run: code_execution, test_output_prediction (dropped by "
                      "--eval-limit 4) (LiveCodeBench publishes all four)")
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", "0")
    assert L._scenarios_not_run(traces, False, None)["self_repair"] == (
        "left out with GBENCH_LCB_INCLUDE_SELF_REPAIR=0")


# --------------------------------------------------------------------------- #
# Codeforces                                                                    #
# --------------------------------------------------------------------------- #
def _cf_rows():
    return [{"id": f"1/{c}", "contest_id": "1", "contest_type": "CF",
             "official_tests": [{"input": "1", "output": "2"}],
             "generated_checker": "print(1)" if c == "B" else None,
             "time_limit": 2.0, "memory_limit": 256.0, "testset_size": 4,
             "official_tests_complete": False, "generated_tests": 0, "language": "python",
             "prompt": f"Solve 1/{c}."} for c in ("A", "B")]


def test_codeforces_loader_stores_tests_and_checker_outside_the_gold(monkeypatch):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _cf_rows())
    store = {}
    samples = CF._load_codeforces_samples(tests_store=store)
    golds = [g for _, g, _ in samples]
    assert all("official_tests" not in g and "checker" not in g for g in golds)
    assert {g["tests_ref"] for g in golds} == {"1/A", "1/B"}
    assert store["1/B"]["checker"] == "print(1)"
    g = next(g for g in golds if g["problem_id"] == "1/B")
    assert g["official_tests_count"] == 1 and g["generated_tests_count"] == 0
    assert g["has_checker"] is True
    assert g["checker_sha256"] == hashlib.sha256(b"print(1)").hexdigest()
    assert g["tests_sha256"]["official_tests"] == CF._tests_sha256([{"input": "1",
                                                                     "output": "2"}])
    # without a store the gold holds them, as before
    inline = CF._load_codeforces_samples()
    assert inline[0][1]["official_tests"] == [{"input": "1", "output": "2"}]


def test_codeforces_judges_from_the_store_and_records_coverage_from_counts(monkeypatch):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _cf_rows())
    monkeypatch.setattr(CF, "_require_generated_tests_mirror", lambda: None)
    monkeypatch.setattr(CF, "sandbox_mode", lambda: "none", raising=False)
    from gbench.runners.eval_suites import sandbox
    monkeypatch.setattr(sandbox, "sandbox_mode", lambda: "none")
    judged = []
    monkeypatch.setattr(CF, "_run_case", lambda code, test, timeout, checker, mode="stdio":
                        judged.append((test["output"], checker)) or True)

    def fake_suite(**kw):
        traces = []
        for i, (_msgs, gold, meta) in enumerate(kw["samples"]):
            ok = kw["eval_fn"]("```python\nprint(2)\n```", gold)
            traces.append({"sample_idx": i, "sample_key": f"k{i}", "global_idx": i,
                           "gold_answer": gold, "response_text": "```python\nprint(2)\n```",
                           "is_correct": ok, "category": meta["category"]})
        return {"accuracy": 100.0, "sample_traces": traces}

    monkeypatch.setattr(CF, "run_eval_suite", fake_suite)
    res = CF.run_codeforces("m", "http://x/v1", 1)
    assert sorted(judged) == [("2", ""), ("2", "print(1)")]
    assert all(t["is_correct"] for t in res["sample_traces"])
    assert all("official_tests" not in t["gold_answer"] and "checker" not in t["gold_answer"]
               for t in res["sample_traces"])
    assert res["problems_with_checker"] == 1 and res["tests_per_problem_median"] == 1
    assert res["test_coverage_pct"] == 25.0

    # the merge recomputes the coverage over the union from the stored counts
    one, two = dict(res), dict(res)
    one["sample_traces"], two["sample_traces"] = res["sample_traces"][:1], res["sample_traces"][1:]
    one.update(shard="1/2", shard_index=1, shard_count=2, problems_with_checker=0)
    two.update(shard="2/2", shard_index=2, shard_count=2, problems_with_checker=1)
    merged = CF.merge_shards([one, two])
    assert merged["problems_with_checker"] == 1 and merged["test_coverage_pct"] == 25.0
    assert "problems_with_checker" not in (merged.get("merge_unreconciled_fields") or {})
