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

"""`--eval-limit N` is N samples for the whole suite, never N per setting.

On 2026-10-08 `--eval-limit 5` ran 5 instances in each of swe_bench_live's 9 runnable tabs (45
agent rollouts, 4-5 hours more than asked for). sampling.allocate_limit deals N over a suite's
settings (tabs, splits, languages, domains) so the suite runs N in total.
"""

import pytest

from gbench.runners.eval_suites.sampling import allocate_limit

TABS = ["lite", "full", "verified", "ccpp", "csharp", "java", "tsjs", "go", "rust"]


@pytest.mark.parametrize("limit", [1, 2, 5, 9, 13, 50])
def test_the_suite_runs_exactly_n_whatever_the_number_of_settings(limit):
    counts = allocate_limit(limit, TABS, seed="swe_bench_live")
    assert sum(counts.values()) == limit
    assert set(counts) == set(TABS)
    # each setting gets one before any gets two
    assert max(counts.values()) - min(counts.values()) <= 1


def test_it_is_deterministic_and_seeded():
    assert allocate_limit(5, TABS, seed="s") == allocate_limit(5, TABS, seed="s")


def test_no_limit_means_every_sample_of_every_setting():
    assert allocate_limit(None, ["a", "b"]) == {"a": None, "b": None}
    assert allocate_limit(0, ["a", "b"]) == {"a": None, "b": None}


def test_a_small_setting_gives_its_unused_share_to_the_others():
    counts = allocate_limit(10, ["a", "b", "c"], available={"a": 1})
    assert counts["a"] == 1 and sum(counts.values()) == 10
    # fewer samples than asked for in total: every sample runs
    assert sum(allocate_limit(10, ["a", "b"], available={"a": 2, "b": 3}).values()) == 5


def test_the_headline_setting_gets_the_first_sample():
    # bigcodebench reports Instruct, lcb code generation, swe_bench_live Lite as the headline
    assert allocate_limit(1, ["instruct", "complete"], seed="bigcodebench",
                          first=("instruct",)) == {"instruct": 1, "complete": 0}
    counts = allocate_limit(2, ["code_generation", "code_execution", "test_output_prediction"],
                            seed="lcb", first=("code_generation",))
    assert counts["code_generation"] >= 1 and sum(counts.values()) == 2
    for n in (1, 3, 9):
        c = allocate_limit(n, TABS, seed="swe_bench_live", first=("lite",))
        assert c["lite"] >= 1 and sum(c.values()) == n
    # without `first` the deal is what it was
    assert allocate_limit(5, TABS, seed="swe_bench_live", first=()) == \
        allocate_limit(5, TABS, seed="swe_bench_live")


# --------------------------------------------------------------------------- #
# the plan: logged once at the start, recorded as eval_limit_plan
# --------------------------------------------------------------------------- #
from unittest import mock                                               # noqa: E402

from gbench.runners.eval_suites import base, limit_plan                 # noqa: E402


def _run_base(eval_name, n, limit=None, attempts=1):
    async def _fake_send(**k):
        return base.Reply(text="x", tool_calls=None, finish_reason="stop", reasoning=None,
                          error=None, completion_tokens=1, stop_reason=None)
    with mock.patch.object(base, "_send_single_request", _fake_send):
        return base.run_eval_suite(
            eval_name=eval_name, model_name="m", base_url="http://x", concurrency=1,
            samples=[([{"role": "user", "content": f"q{i}"}], "x", {"category": f"c{i % 2}"})
                     for i in range(n)],
            eval_fn=lambda r, g: True, limit=limit, attempt_count=attempts,
            supports_attempts=True)


def test_a_base_suite_records_its_plan(caplog):
    limit_plan.clear_plan("gsm8k")
    with caplog.at_level("INFO"):
        res = _run_base("gsm8k", 12, limit=5, attempts=2)
    plan = res["eval_limit_plan"]
    assert plan["limit"] == 5 and plan["samples"] == 5 and plan["attempts_per_sample"] == 2
    assert plan["generations"] == 10
    assert any("[gsm8k] --eval-limit 5: 5 sample(s) x 2 attempts = 10 generations" in r.message
               for r in caplog.records)
    limit_plan.clear_plan("gsm8k")
    full = _run_base("gsm8k", 12)["eval_limit_plan"]
    assert full["limit"] is None and full["samples"] == 12


def test_a_suites_own_plan_is_kept_and_a_stale_one_is_cleared():
    own = limit_plan.eval_limit_plan("bigcodebench", 5, 5, {"instruct": 3, "complete": 2},
                                     setting_noun="split")
    assert _run_base("bigcodebench", 3, limit=5)["eval_limit_plan"] == own
    limit_plan.clear_plan("bigcodebench")
    assert limit_plan.planned_for("bigcodebench") is None
    assert _run_base("bigcodebench", 3, limit=5)["eval_limit_plan"]["samples"] == 3


def test_a_merged_sharded_run_sums_the_shards_plans():
    from gbench.merge import generic_merge
    shards = []
    for i, (n, setting) in enumerate([(3, {"a": 2, "b": 1}), (2, {"a": 1, "b": 1})]):
        traces = [{"sample_key": f"k{i}{j}", "category": "c", "is_correct": True,
                   "status": "OK"} for j in range(n)]
        shards.append({"eval_name": "x", "sample_traces": traces, "accuracy": 100.0,
                       "total_questions": n, "correct_answers": n,
                       "shard_index": i + 1, "shard_count": 2,
                       "eval_limit_plan": {"limit": 5, "samples": n, "per_setting": setting,
                                           "attempts_per_sample": 1, "generations": n}})
    merged = generic_merge("x", shards)
    plan = merged["eval_limit_plan"]
    assert plan["samples"] == 5 and plan["per_setting"] == {"a": 3, "b": 2}
    assert plan["generations"] == 5
    assert "eval_limit_plan" not in merged.get("merge_unreconciled_fields", {})


def test_gaia2_eval_limit_wins_over_its_per_config_knob(monkeypatch):
    from gbench.runners.eval_suites import gaia2
    src = open(gaia2.__file__, encoding="utf-8").read()
    # the operator's GBENCH_GAIA2_LIMIT is dropped when --eval-limit is set
    assert "GBENCH_GAIA2_LIMIT=%s is ignored because --eval-limit %s is set" in src
    shares = gaia2.config_limit_shares(5, gaia2.run_configs("run", ""))
    assert sum(shares.values()) == 5


@pytest.mark.parametrize("n", range(1, 12))
def test_first_never_makes_the_deal_uneven(n):
    # 2026-10-09 swarm: first= used to give the headline an extra sample, so at lcb N=3 a
    # scenario got 0 while code generation got 2
    for settings, head in ((["code_generation", "code_execution", "test_output_prediction"],
                            "code_generation"), (["instruct", "complete"], "instruct"),
                           (TABS, "lite")):
        c = allocate_limit(n, settings, seed="s", first=(head,))
        assert sum(c.values()) == n and c[head] >= 1
        assert max(c.values()) - min(c.values()) <= 1
