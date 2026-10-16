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

"""The agentic suites' --eval-limit plans are true and their operator knobs never beat it.

agent_dojo's plan line counts the injection-task-as-user checks (3N conversations, not 2N);
GBENCH_SKILLSBENCH_TASKS and GBENCH_OSWORLD_LIMIT are ignored with a warning when --eval-limit is
set (as GBENCH_GAIA2_LIMIT is); gaia2's line counts the mini scenarios' second phase, gaia2-run
always runs ARE's 3 runs, and a GBENCH_GAIA2_LIMIT-only run is not logged as a full run;
mcp_bench / wildclawbench / ui_control_osworld cap the plan at the selectable set and correct it
from the launcher's count; each suite's own merge_shards records the summed plan.
"""

import logging
import sys
from pathlib import Path

import pytest

from gbench.runners.eval_suites import agent_dojo as AD
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import mcp_bench as M
from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import ui_control_osworld as U
from gbench.runners.eval_suites import wildclawbench as W

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("GBENCH_SHARD", "GBENCH_SKILLSBENCH_TASKS", "GBENCH_OSWORLD_LIMIT",
              "GBENCH_GAIA2_LIMIT", "GBENCH_GAIA2_NUM_RUNS", "GBENCH_AGENTDOJO_ATTACK"):
        monkeypatch.delenv(k, raising=False)


def _lines(caplog, tag):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(tag)]


# --------------------------------------------------------------------------------------------
# agent_dojo: N benign + N attacked + K injection-task-as-user checks, all in the one line
# --------------------------------------------------------------------------------------------
def test_agent_dojo_plan_line_counts_the_injection_task_checks(caplog):
    by_suite = {"banking": ["u0", "u1"], "slack": ["s0"]}
    inj = {"banking": {"u0": ["i0"], "u1": ["i1"]}, "slack": {"s0": ["i0"]}}
    with caplog.at_level(logging.INFO):
        plan = AD.agent_dojo_plan(3, 3, {"banking": 2, "slack": 1}, by_suite, inj,
                                  attack="important_instructions")
    (line,) = _lines(caplog, "[agent_dojo]")
    assert "6 episodes" in line and "3 injection-task-as-user check(s)" in line
    assert line.endswith("= 9 conversations")
    assert (plan["samples"], plan["episodes"], plan["injection_task_checks"],
            plan["conversations"]) == (3, 6, 3, 9)
    assert not any(" check(s)" in k for k in plan)       # the record keeps plain keys


def test_agent_dojo_dos_attack_has_no_injection_checks():
    plan = AD.agent_dojo_plan(None, 2, {"banking": 2}, {"banking": ["u0", "u1"]},
                              {"banking": {"u0": ["i0", "i1"], "u1": ["i0", "i1"]}},
                              attack="captcha_dos")
    assert plan["episodes"] == 4 and plan["injection_task_checks"] == 0


def test_agent_dojo_run_records_the_conversations_it_ran(monkeypatch, caplog):
    from test_agent_dojo_episodes_and_harness_errors import _fake
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake())
    with caplog.at_level(logging.INFO):
        r = AD.run_agent_dojo("m", "http://h/v1", limit=2, concurrency=1)
    plan = r["eval_limit_plan"]
    kinds = [t.get("kind") or t.get("episode_kind") for t in r["sample_traces"]]
    assert plan["conversations"] == len(r["sample_traces"]) == len(kinds)
    assert plan["conversations"] == plan["episodes"] + plan["injection_task_checks"] > 4
    (line,) = _lines(caplog, "[agent_dojo]")
    assert f"= {plan['conversations']} conversations" in line


def test_agent_dojo_merge_sums_the_plan():
    def shard(n):
        return {"item_results": {"benign": {}, "under_attack": {}, "attack_success": {}},
                "eval_limit_plan": {"limit": 4, "samples": n, "per_setting": {"banking": n},
                                    "attempts_per_sample": 2, "episodes": 2 * n,
                                    "injection_task_checks": n, "conversations": 3 * n}}
    m = AD.merge_shards([shard(2), shard(1)])
    assert m["eval_limit_plan"]["samples"] == 3 and m["eval_limit_plan"]["conversations"] == 9


# --------------------------------------------------------------------------------------------
# skillsbench: --eval-limit wins over GBENCH_SKILLSBENCH_TASKS
# --------------------------------------------------------------------------------------------
def test_skillsbench_task_list_is_ignored_under_eval_limit(monkeypatch, caplog):
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASKS", ",".join(f"t{i}" for i in range(20)))
    with caplog.at_level(logging.WARNING):
        assert SB.explicit_tasks_env(5) is None
    assert any("GBENCH_SKILLSBENCH_TASKS" in m and "ignored" in m
               for m in (r.getMessage() for r in caplog.records))
    assert SB.explicit_tasks_env(None) == ",".join(f"t{i}" for i in range(20))


def test_skillsbench_run_with_a_task_list_and_a_limit_draws_n(monkeypatch):
    from test_shard_harness_suites import _sb_run, _summary
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASKS", ",".join(f"t{i}" for i in range(20)))
    monkeypatch.setenv("GBENCH_SKILLSBENCH_CONDITIONS", "with-skills,without-skills")
    summary = _summary(["t1", "t2", "t3"])
    summary["selection"] = {"limit_draw": True,
                            "condition_limits": SB.condition_limit_shares(
                                5, ["with-skills", "without-skills"])}
    res, seen = _sb_run(monkeypatch, summary, limit=5)
    envs = [seen["cmd"][i + 1] for i, x in enumerate(seen["cmd"]) if x == "-e"]
    assert not any(e.startswith("SKILLSBENCH_TASKS=") for e in envs)
    assert any(e.startswith("SKILLSBENCH_CONDITION_LIMITS=") for e in envs)
    assert res["eval_limit_plan"]["samples"] == 5 and res["eval_limit_plan"]["rollouts"] == 5


def test_skillsbench_plan_records_the_distinct_tasks_of_nested_draws():
    shares = {"with-skills": 3, "without-skills": 2}
    plan = SB.limit_plan(5, list(shares), shares, [])
    assert plan["samples"] == 5 and plan["distinct_tasks"] == 3


# --------------------------------------------------------------------------------------------
# ui_control_osworld: --eval-limit wins over GBENCH_OSWORLD_LIMIT; the plan is the selection
# --------------------------------------------------------------------------------------------
def test_osworld_eval_limit_wins_over_the_env_knob(monkeypatch, caplog):
    monkeypatch.setenv("GBENCH_OSWORLD_LIMIT", "50")
    with caplog.at_level(logging.WARNING):
        assert U.effective_limit(5) == (5, False)
    assert any("GBENCH_OSWORLD_LIMIT" in r.getMessage() and "ignored" in r.getMessage()
               for r in caplog.records)
    assert U.effective_limit(None) == (50, True)
    monkeypatch.delenv("GBENCH_OSWORLD_LIMIT")
    assert U.effective_limit(None) == (None, False)


def test_osworld_knob_only_plan_does_not_claim_an_eval_limit(caplog):
    with caplog.at_level(logging.INFO):
        plan = U.osworld_plan(50, "", None, knob=True)
    (line,) = _lines(caplog, "[ui_control_osworld]")
    assert "--eval-limit 50" not in line and "GBENCH_OSWORLD_LIMIT 50" in line
    assert plan["limit"] is None and plan["samples"] == 50


def test_osworld_plan_is_capped_and_corrected_from_the_selection():
    plan = U.osworld_plan(1000, "", None)
    assert plan["samples"] == U._FULL_TASK_COUNT
    fixed = U.plan_from_summary(U.osworld_plan(10, "chrome", None), 4)
    assert fixed["samples"] == fixed["episodes"] == 4 and fixed["samples_logged"] == 10


def test_osworld_merge_sums_the_plan():
    def shard(n):
        return {"sample_traces": [], "status": "success",
                "eval_limit_plan": {"limit": 4, "samples": n, "per_setting": None,
                                    "attempts_per_sample": 1, "episodes": n}}
    m = U.merge_shards([shard(2), shard(2)])
    assert m["eval_limit_plan"]["samples"] == 4 and m["eval_limit_plan"]["episodes"] == 4


# --------------------------------------------------------------------------------------------
# gaia2
# --------------------------------------------------------------------------------------------
def test_gaia2_run_mode_line_counts_the_mini_second_phase(caplog):
    shares = G.config_limit_shares(6, G.run_configs("gaia2-run", ""))
    assert shares["mini"] == 1
    with caplog.at_level(logging.INFO):
        plan = G.limit_plan(6, shares, "gaia2-run", "", 3)
    (line,) = _lines(caplog, "[gaia2]")
    assert line.endswith("= 21 scenario_runs") and "+ 3 (the mini" in line
    assert plan["scenario_runs"] == 21 and plan["phase_runs_per_mini_scenario"] == 2
    assert not any("second phase" in k for k in plan)


def test_gaia2_knob_only_run_is_not_logged_as_a_full_run(caplog):
    with caplog.at_level(logging.INFO):
        plan = G.limit_plan(None, None, "run", "", 3, knob_limit=2)
    (line,) = _lines(caplog, "[gaia2]")
    assert "full run" not in line and "GBENCH_GAIA2_LIMIT 2 per config" in line
    assert plan["samples"] == 2 * len(G.run_configs("run", ""))
    assert plan["scenario_runs"] == 3 * plan["samples"]


def test_gaia2_run_mode_always_runs_ares_three_runs(monkeypatch, caplog):
    monkeypatch.setenv("GBENCH_GAIA2_NUM_RUNS", "5")
    with caplog.at_level(logging.WARNING):
        assert G.resolve_num_runs("gaia2-run", 1)[0] == 3
    assert any("ignored" in r.getMessage() for r in caplog.records)
    assert G.resolve_num_runs("run", 1) == (5, "GBENCH_GAIA2_NUM_RUNS")
    monkeypatch.delenv("GBENCH_GAIA2_NUM_RUNS")
    assert G.resolve_num_runs("gaia2-run", 7)[0] == 3
    assert G.resolve_num_runs("run", 7)[0] == 7


def test_gaia2_run_mode_does_not_forward_another_num_runs(monkeypatch, tmp_path):
    from test_gaia2_toolbench_judge_protocol_and_credentials import _gaia2_run
    res, [(cmd, _)], _ = _gaia2_run(monkeypatch, tmp_path,
                                    env={"GBENCH_GAIA2_MODE": "gaia2-run",
                                         "GBENCH_GAIA2_NUM_RUNS": "5"})
    envs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
    assert not any(e.startswith("GAIA2_NUM_RUNS=") for e in envs)
    assert res["eval_limit_plan"]["attempts_per_sample"] == 3


def test_gaia2_merge_sums_the_plan():
    def shard(n):
        return {"status": "success", "sample_traces": [],
                "eval_limit_plan": {"limit": 6, "samples": n, "per_setting": {"time": n},
                                    "attempts_per_sample": 3, "scenario_runs": 3 * n}}
    m = G.merge_shards([shard(2), shard(1)])
    assert m["eval_limit_plan"]["samples"] == 3 and m["eval_limit_plan"]["scenario_runs"] == 9


# --------------------------------------------------------------------------------------------
# mcp_bench / wildclawbench: capped at the selectable set, corrected from the launcher
# --------------------------------------------------------------------------------------------
def test_mcp_bench_plan_is_capped_and_corrected_from_the_manifest():
    plan = M.mcp_bench_plan(500)
    assert plan["samples"] == plan["task_runs"] == M.UPSTREAM_TASK_COUNT == 104
    fixed = M.plan_from_kept(M.mcp_bench_plan(10), 7)
    assert fixed["samples"] == fixed["task_runs"] == 7 and fixed["samples_logged"] == 10
    assert M.plan_from_kept(plan, None) is plan


def test_wildclaw_plan_is_capped_and_corrected_from_the_summary():
    plan = W.wildclaw_plan(500, "")
    assert plan["samples"] == W._FULL_TASK_COUNT == 60
    fixed = W.plan_from_summary(W.wildclaw_plan(10, ""), 3)
    assert fixed["samples"] == fixed["task_runs"] == 3 and fixed["samples_logged"] == 10
    assert W.wildclaw_plan(5, "some-task")["limit"] == 5


def test_wildclaw_merge_sums_the_plan():
    def shard(n):
        return {"status": "success", "raw_summary": {"per_task": []},
                "eval_limit_plan": {"limit": 4, "samples": n, "per_setting": None,
                                    "attempts_per_sample": 1, "task_runs": n}}
    m = W.merge_shards([shard(2), shard(2)])
    assert m["eval_limit_plan"]["samples"] == 4 and m["eval_limit_plan"]["task_runs"] == 4


def test_skillsbench_merge_sums_the_plan():
    def shard(n, task):
        return {"status": "success", "sample_traces": [
            {"sample_key": f"task:without-skills/{task}", "condition": "without-skills",
             "task": task, "reward": 1.0}],
            "eval_limit_plan": {"limit": 4, "samples": n, "per_setting": {"without-skills": n},
                                "attempts_per_sample": 1, "rollouts": n, "distinct_tasks": n}}
    m = SB.merge_shards([shard(2, "a"), shard(2, "b")])
    assert m["eval_limit_plan"]["samples"] == 4 and m["eval_limit_plan"]["rollouts"] == 4
