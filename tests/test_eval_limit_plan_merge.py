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

"""--eval-limit plans across shards and runs: merge_plans robustness, merge_suite summing the plan
on a suite's own merge_shards path, the empty-shard plan, base's per-call plan, the fc_common
full-run plan registration and evals.py clearing a stale plan between runs."""

import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, fc_common, limit_plan


def _r(plan):
    return {"eval_limit_plan": plan}


# --------------------------------------------------------------------------- #
# merge_plans
# --------------------------------------------------------------------------- #

def test_merge_plans_keeps_an_equal_note_and_drops_a_differing_one():
    a = {"limit": 6, "samples": 3, "per_setting": None, "attempts_per_sample": 1,
         "generations": 3, "note": "plus 2 repair calls"}
    merged = limit_plan.merge_plans([_r(a), _r(dict(a))])
    assert merged["note"] == "plus 2 repair calls" and merged["samples"] == 6
    merged = limit_plan.merge_plans([_r(a), _r(dict(a, note="plus 1 repair call"))])
    assert "note" not in merged and merged["unmerged_fields"] == ["note"]
    assert merged["generations"] == 6


def test_merge_plans_keeps_unknown_counts_unknown():
    full = {"limit": None, "samples": None, "per_setting": {"a": None, "b": None},
            "attempts_per_sample": 1, "rollouts": None}
    merged = limit_plan.merge_plans([_r(full), _r(dict(full, per_setting={"a": None, "b": None}))])
    assert merged["samples"] is None and merged["rollouts"] is None
    assert merged["per_setting"] == {"a": None, "b": None}
    # one shard knows, the other does not: still unknown
    known = {"limit": None, "samples": 4, "per_setting": {"a": 4, "b": 0},
             "attempts_per_sample": 1, "rollouts": 4}
    merged = limit_plan.merge_plans([_r(known), _r(full)])
    assert merged["samples"] is None and merged["rollouts"] is None
    assert merged["per_setting"] == {"a": None, "b": None}


def test_merge_plans_does_not_sum_per_sample_constants():
    p = {"limit": 4, "samples": 2, "per_setting": None, "attempts_per_sample": 3,
         "rollouts": 6, "tries_per_rollout_at_most": 2, "phase_runs_per_mini_scenario": 2,
         "self_repair_requests_at_most": 6}
    merged = limit_plan.merge_plans([_r(p), _r(dict(p))])
    assert merged["limit"] == 4 and merged["attempts_per_sample"] == 3
    assert merged["tries_per_rollout_at_most"] == 2
    assert merged["phase_runs_per_mini_scenario"] == 2
    assert merged["samples"] == 4 and merged["rollouts"] == 12
    assert merged["self_repair_requests_at_most"] == 12


def test_merge_plans_an_empty_shard_adds_zero_to_counts_it_did_not_record():
    p = {"limit": 3, "samples": 3, "per_setting": {"a": 2, "b": 1}, "attempts_per_sample": 1,
         "episodes": 9}
    empty = {"limit": 3, "samples": 0, "per_setting": None, "attempts_per_sample": 1,
             "generations": 0}
    merged = limit_plan.merge_plans([_r(p), _r(empty)])
    assert merged["samples"] == 3 and merged["episodes"] == 9 and "generations" not in merged
    assert merged["per_setting"] == {"a": 2, "b": 1}
    assert "shards_without_plan" not in merged
    assert limit_plan.merge_plans([_r(p), {"accuracy": 1.0}])["shards_without_plan"] == 1


# --------------------------------------------------------------------------- #
# merge_suite: a suite's own merge_shards records the summed plan
# --------------------------------------------------------------------------- #

def test_merge_suite_sums_the_plan_on_a_suites_own_merge_shards_path():
    from gbench import merge
    fake = types.ModuleType("fake_suite")
    fake.merge_shards = lambda results: dict(results[0])          # keeps shard 1's plan
    plans = [{"limit": 5, "samples": 3, "per_setting": {"a": 2, "b": 1},
              "attempts_per_sample": 2, "episodes": 6},
             {"limit": 5, "samples": 2, "per_setting": {"a": 1, "b": 1},
              "attempts_per_sample": 2, "episodes": 4}]
    with mock.patch.object(merge.importlib, "import_module", return_value=fake):
        merged = merge.merge_suite("fake_suite", [_r(p) for p in plans])
    plan = merged["eval_limit_plan"]
    assert plan["samples"] == 5 and plan["episodes"] == 10
    assert plan["per_setting"] == {"a": 3, "b": 2} and plan["attempts_per_sample"] == 2


# --------------------------------------------------------------------------- #
# base: the empty-shard plan, and a plan per call
# --------------------------------------------------------------------------- #

def _run_base(eval_name, n, limit=None, **kw):
    async def _fake_send(**k):
        return base.Reply(text="x", tool_calls=None, finish_reason="stop", reasoning=None,
                          error=None, completion_tokens=1, stop_reason=None)
    with mock.patch.object(base, "_send_single_request", _fake_send):
        return base.run_eval_suite(
            eval_name=eval_name, model_name="m", base_url="http://x", concurrency=1,
            samples=[([{"role": "user", "content": f"q{i}"}], "x", {}) for i in range(n)],
            eval_fn=lambda r, g: True, limit=limit, **kw)


def test_an_empty_shard_records_a_zero_sample_plan(monkeypatch, caplog):
    limit_plan.clear_plan("plan_empty_x")
    monkeypatch.setenv("GBENCH_SHARD", "3/3")
    with caplog.at_level("INFO"):
        res = _run_base("plan_empty_x", 2, limit=2, attempt_count=2, supports_attempts=True)
    assert res["empty_shard"] is True
    plan = res["eval_limit_plan"]
    assert plan["samples"] == 0 and plan["generations"] == 0
    assert plan["attempts_per_sample"] == 2 and plan["limit"] == 2
    assert any("[plan_empty_x] --eval-limit 2: 0 sample(s)" in r.message for r in caplog.records)


def test_an_empty_shard_keeps_the_suites_own_plan(monkeypatch):
    own = limit_plan.eval_limit_plan("plan_empty_y", 4, 0, {"a": 0, "b": 0})
    monkeypatch.setenv("GBENCH_SHARD", "3/3")
    assert _run_base("plan_empty_y", 2, limit=4)["eval_limit_plan"] == own
    limit_plan.clear_plan("plan_empty_y")


def test_a_second_base_call_does_not_reuse_the_first_calls_plan(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    limit_plan.clear_plan("plan_two_pass")
    first = _run_base("plan_two_pass", 7)["eval_limit_plan"]
    second = _run_base("plan_two_pass", 3)["eval_limit_plan"]
    assert first["samples"] == 7 and second["samples"] == 3
    assert limit_plan.planned_for("plan_two_pass") is None


# --------------------------------------------------------------------------- #
# fc_common: a full run's plan is registered, so base neither logs nor records a second one
# --------------------------------------------------------------------------- #

def test_fc_common_full_run_plan_is_the_one_base_records(monkeypatch, caplog):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    limit_plan.clear_plan("plan_fc_full")
    own = fc_common.eval_limit_plan("plan_fc_full", None, None, {"single": None, "multi": None},
                                    note="plus prerequisite entries")
    assert limit_plan.planned_for("plan_fc_full") is own
    with caplog.at_level("INFO"):
        res = _run_base("plan_fc_full", 4)
    assert res["eval_limit_plan"] is own
    assert not any("[plan_fc_full] full run (no --eval-limit): 4 sample(s)" in r.message
                   for r in caplog.records)
    limit_plan.clear_plan("plan_fc_full")


# --------------------------------------------------------------------------- #
# evals.py clears a plan an earlier run of the suite registered
# --------------------------------------------------------------------------- #

def test_evals_clears_a_stale_plan_between_runs(monkeypatch):
    from gbench.runners import evals
    from gbench.runners import eval_suites
    seen = []

    def fake_runner(**kw):
        seen.append(limit_plan.planned_for("plan_evals_x"))
        if kw.get("limit"):
            limit_plan.eval_limit_plan("plan_evals_x", kw["limit"], kw["limit"])
        return {"status": "success"}

    monkeypatch.setitem(eval_suites.SUITES, "plan_evals_x", fake_runner)
    monkeypatch.setattr(evals, "_server_max_model_len", lambda url: None)
    runner = object.__new__(evals.EvalsBenchmarkRunner)
    runner.config = types.SimpleNamespace(eval_limit=5)
    runner._run_single_eval("plan_evals_x", "m", "http://x/v1", 1)
    runner.config = types.SimpleNamespace(eval_limit=None)
    runner._run_single_eval("plan_evals_x", "m", "http://x/v1", 1)
    assert seen == [None, None]
    limit_plan.clear_plan("plan_evals_x")
