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

"""`--eval-limit N` is N samples for the whole suite in the agentic / environment suites.

On 2026-10-08 `--eval-limit 5` ran 5 swe_bench_live instances in each of its 9 runnable tabs (45
agent rollouts, 4-5 hours over). Each suite that runs settings (tabs, task types, domains,
capability configs, suites, conditions) now deals the N over them; the suites without settings
draw N from one flattened list. Each test checks that --eval-limit 5 selects exactly 5 samples
(and the per-setting counts) and that no limit selects everything.
"""

import os
import sys
from pathlib import Path

import pytest

from gbench.runners.eval_suites import agent_dojo as AD
from gbench.runners.eval_suites import gaia2 as G2
from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import swe_agentic as SA
from gbench.runners.eval_suites import swe_bench_live as SBL
from gbench.runners.eval_suites import swe_lancer as SL
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import terminal_bench as TB
from gbench.runners.eval_suites import toolbench as T

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gbench" / "docker"))
import gaia2_run as GRUN  # noqa: E402
import skillsbench_run as SRUN  # noqa: E402


@pytest.fixture(autouse=True)
def _no_shard(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)


# --------------------------------------------------------------------------------------------
# swe_bench_live: N instances over the tabs that run
# --------------------------------------------------------------------------------------------
LINUX_TABS = [s for s in SBL.SETTINGS if s != "windows"]


def test_swe_bench_live_shares_are_n_for_the_suite_not_n_per_tab():
    shares, plan = SBL.limit_shares(5, LINUX_TABS)
    assert sum(shares.values()) == 5 and set(shares.values()) == {0, 1}
    assert plan["samples"] == 5 and plan["rollouts"] == 5 and plan["limit"] == 5
    assert plan["per_setting"] == {t: shares[t] for t in LINUX_TABS}
    shares20, _ = SBL.limit_shares(20, LINUX_TABS)
    assert sum(shares20.values()) == 20 and max(shares20.values()) - min(shares20.values()) <= 1
    full, plan = SBL.limit_shares(None, LINUX_TABS)
    assert set(full.values()) == {None} and plan["limit"] is None
    assert plan["samples"] == sum(SBL.setting_rows(t) for t in LINUX_TABS)


def _live_run(monkeypatch, limit):
    seen = {}
    monkeypatch.setattr(SBL.swe_agentic, "resolve_mode", lambda name: (SA.AGENTIC, "default"))
    monkeypatch.setattr(SBL, "missing_prerequisites",
                        lambda settings, mode: {"windows": "not a Windows host"})
    monkeypatch.setattr(SBL.swebench_common, "require_task_network_blocked",
                        lambda *a, **k: None)
    monkeypatch.setattr(SBL, "_raise_if_stop_requested", lambda where: None)

    def run_setting(setting, mode, model, url, conc, kwargs):
        seen[setting] = (kwargs.get("limit"), kwargs.get("limit_note"))
        n = kwargs.get("limit") or SBL.setting_rows(setting)
        return {"status": "success", "accuracy": 0.0, "total_questions": n,
                "correct_answers": 0, "leaderboard_comparable": False,
                "leaderboard_comparable_reason": kwargs.get("limit_note") or "full"}
    monkeypatch.setattr(SBL, "_run_setting", run_setting)
    return SBL.run_swe_bench_live("m", "http://h/v1", 1, limit=limit), seen


def test_swe_bench_live_eval_limit_5_runs_5_instances_in_total(monkeypatch):
    res, seen = _live_run(monkeypatch, 5)
    assert sum(lim for lim, _ in seen.values()) == 5 and len(seen) == 5
    assert all(lim == 1 for lim, _ in seen.values())
    assert all("--eval-limit 5 for the suite" in note for _, note in seen.values())
    assert set(res["settings_not_run_for_limit"]) == set(LINUX_TABS) - set(seen)
    assert "no instance (not run)" in res["leaderboard_comparable_reason"]
    assert res["eval_limit_plan"]["samples"] == 5
    # the Windows tab stays reported as infra_required, the others that got 0 are not run
    assert res["settings"]["windows"]["status"] == "infra_required"


def test_swe_bench_live_without_a_limit_runs_every_tab_completely(monkeypatch):
    res, seen = _live_run(monkeypatch, None)
    assert set(seen) == set(LINUX_TABS) and all(lim is None for lim, _ in seen.values())
    assert "settings_not_run_for_limit" not in res


# --------------------------------------------------------------------------------------------
# swe_lancer: N tasks over the task types
# --------------------------------------------------------------------------------------------
def _lancer_tasks():
    return ([{"question_id": f"ic{i}", "variant": "ic_swe", "set": SL.SPLIT} for i in range(30)]
            + [{"question_id": f"m{i}", "variant": "swe_manager", "set": SL.SPLIT}
               for i in range(30)])


def test_swe_lancer_eval_limit_5_is_5_tasks_over_both_types():
    tasks = _lancer_tasks()
    shares = SL.variant_shares(tasks, SL.VARIANTS, 5)
    assert sum(shares.values()) == 5 and sorted(shares.values()) == [2, 3]
    picked = {v: SL.select_tasks(tasks, v, shares[v])[0] for v in SL.VARIANTS}
    assert sum(len(ids) for ids in picked.values()) == 5
    one = SL.variant_shares(tasks, SL.VARIANTS, 1)
    zero = next(v for v, n in one.items() if n == 0)
    assert SL.select_tasks(tasks, zero, 0)[0] == []          # a share of 0 runs no task
    full = SL.variant_shares(tasks, SL.VARIANTS, None)
    assert full == {"ic_swe": None, "swe_manager": None}
    assert len(SL.select_tasks(tasks, "ic_swe", None)[0]) == 30


# --------------------------------------------------------------------------------------------
# tau2: N tasks over the domains
# --------------------------------------------------------------------------------------------
def _tau_domains(monkeypatch, tmp_path, limit, sizes):
    ran = []
    monkeypatch.setattr(TC, "_shard_task_ids", lambda d, s, lim, **kw: (
        [f"{d}-{i}" for i in range(sizes[d])], sizes[d]))

    def one(domain, *a, task_ids=None, **k):
        ran.append((domain, None if task_ids is None else list(task_ids)))
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": 0.5}
        TC._DOMAIN_TASK_OUTCOMES[domain] = {t: [1, 1] for t in (task_ids or ["x"])}
        return len(task_ids or ["x"]), 1.0, 1, 0
    monkeypatch.setattr(TC, "_run_one_domain", one)
    res = TC._run_domains("tau2", list(sizes), "m", "http://h/v1", 1, limit, "d", False, 4,
                          str(tmp_path), {}, "h", False, shard=None)
    return res, ran


def test_tau2_eval_limit_5_is_5_tasks_over_the_3_domains(monkeypatch, tmp_path):
    sizes = {"retail": 114, "airline": 50, "telecom": 114}
    res, ran = _tau_domains(monkeypatch, tmp_path, 5, sizes)
    assert sum(len(ids) for _, ids in ran) == 5
    assert sorted(len(ids) for _, ids in ran) == [1, 2, 2]
    plan = res["eval_limit_plan"]
    assert plan["samples"] == 5 and plan["attempts_per_sample"] == 4
    assert plan["conversations"] == 20 and sum(plan["per_setting"].values()) == 5


def test_tau2_eval_limit_2_leaves_a_domain_out_and_says_so(monkeypatch, tmp_path):
    sizes = {"retail": 114, "airline": 50, "telecom": 114}
    res, ran = _tau_domains(monkeypatch, tmp_path, 2, sizes)
    assert len(ran) == 2 and sum(len(ids) for _, ids in ran) == 2
    (skipped,) = res["limit_skipped_domains"]
    assert skipped not in [d for d, _ in ran]
    assert f"gives {skipped} no task" in res["leaderboard_comparable_reason"]


def test_tau2_without_a_limit_runs_every_task_of_every_domain(monkeypatch, tmp_path):
    sizes = {"retail": 3, "airline": 3, "telecom": 3}
    res, ran = _tau_domains(monkeypatch, tmp_path, None, sizes)
    assert ran == [("retail", None), ("airline", None), ("telecom", None)]   # tau2's own list
    assert res["eval_limit_plan"]["limit"] is None


# --------------------------------------------------------------------------------------------
# gaia2: N scenarios over the capability configs
# --------------------------------------------------------------------------------------------
def test_gaia2_eval_limit_5_is_one_scenario_of_each_capability():
    shares = G2.config_limit_shares(5, G2.run_configs("run", ""))
    assert shares == {c: 1 for c in G2._CAPABILITY_CONFIGS}
    three = G2.config_limit_shares(3, G2.run_configs("run", ""))
    assert sum(three.values()) == 3 and list(three.values()).count(0) == 2
    assert sorted(G2.limit_skipped_capabilities(three)) == sorted(
        c for c, n in three.items() if n == 0)
    plan = G2.limit_plan(5, shares, "run", "", 3)
    assert plan["samples"] == 5 and plan["scenario_runs"] == 15
    assert G2.config_limit_shares(None, G2.run_configs("run", "")) is None


def test_gaia2_run_mode_deals_the_limit_over_the_mini_config_too():
    shares = G2.config_limit_shares(6, G2.run_configs("gaia2-run", ""))
    assert shares == {**{c: 1 for c in G2._CAPABILITY_CONFIGS}, "mini": 1}
    plan = G2.limit_plan(6, shares, "gaia2-run", "", 3)
    # each mini scenario runs in the Agent2Agent and the Noise phase
    assert plan["samples"] == 6 and plan["scenario_runs"] == 6 * 3 + 3
    skipped = G2.limit_skipped_capabilities({"mini": 0, "time": 1})
    assert skipped == list(G2._GAIA2_RUN_EXTRA_CAPABILITIES)


def test_gaia2_launcher_draws_each_configs_share_and_runs_none_for_a_zero():
    ids = [f"s{i}" for i in range(40)]
    assert len(GRUN.select_scenario_ids(ids, None, 2)) == 2
    assert GRUN.select_scenario_ids(ids, None, 0) == []
    assert len(GRUN.select_scenario_ids(ids, None, None)) == 40
    assert GRUN.parse_config_limits('{"time": 1, "search": 0}') == {"time": 1, "search": 0}
    assert GRUN.parse_config_limits("") is None
    for bad in ('{"time": -1}', "[1]", "x", '{"time": true}'):
        with pytest.raises(ValueError):
            GRUN.parse_config_limits(bad)


def test_gaia2_launcher_skips_a_config_whose_share_is_zero(monkeypatch):
    import types
    calls = []
    cli = types.SimpleNamespace(run_dataset=lambda *a, **k: calls.append(k["config"]) or "ok")
    sub = types.SimpleNamespace(run_dataset=lambda *a, **k: calls.append(k["config"]) or "ok")
    pkg = types.ModuleType("are.simulation.benchmark")
    pkg.cli, pkg.gaia2_submission = cli, sub
    for name in ("are", "are.simulation"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "are.simulation.benchmark", pkg)
    GRUN.install_limit_skip({"time": 1, "search": 0})
    assert cli.run_dataset(config="time") == "ok" and calls == ["time"]
    with pytest.raises(GRUN.LimitSkipped):
        cli.run_dataset(config="search")
    with pytest.raises(GRUN.LimitSkipped):
        sub.run_dataset(config="mini")            # not in the deal: 0
    assert calls == ["time"]


# --------------------------------------------------------------------------------------------
# agent_dojo: N user tasks over the suites, one injection task each
# --------------------------------------------------------------------------------------------
def test_agent_dojo_eval_limit_5_is_5_user_tasks_over_the_suites():
    ordered = [(s, f"{s}-u{i}") for s, n in (("workspace", 40), ("travel", 20),
                                             ("banking", 16), ("slack", 21)) for i in range(n)]
    picked = AD.select_user_tasks(ordered, 5)
    assert len(picked) == 5
    per_suite = {s: sum(1 for p, _ in picked if p == s) for s, _ in ordered}
    assert sorted(per_suite.values()) == [1, 1, 1, 2]
    assert picked == [p for p in ordered if p in picked]       # the package's order kept
    assert AD.select_user_tasks(ordered, None) == ordered
    pairs = AD.injection_pairs(["i0", "i1", "i2"], ["u0", "u1", "u2", "u3"], 5, "slack")
    assert all(len(v) == 1 for v in pairs.values())
    assert len({v[0] for v in pairs.values()}) == 3            # dealt in turn, then wraps
    assert AD.injection_pairs(["i0", "i1"], ["u0"], None, "slack") == {"u0": ["i0", "i1"]}


def test_agent_dojo_run_with_a_limit_runs_one_attack_per_user_task(monkeypatch):
    sys.path.insert(0, str(REPO / "tests"))
    from test_agent_dojo_episodes_and_harness_errors import _fake
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake())
    r = AD.run_agent_dojo("m", "http://h/v1", limit=2, concurrency=1)
    kinds = [t.get("kind") or t.get("episode_kind") for t in r["sample_traces"]]
    assert r["eval_limit_plan"]["samples"] == 2 and r["eval_limit_plan"]["episodes"] == 4
    assert len(r["item_results"]["benign"]) == 2
    assert len(r["item_results"]["under_attack"]) == 2
    assert kinds   # every episode traced


# --------------------------------------------------------------------------------------------
# skillsbench: N rollouts over the conditions
# --------------------------------------------------------------------------------------------
def test_skillsbench_eval_limit_5_over_both_conditions_is_5_rollouts():
    conds = ["with-skills", "without-skills"]
    shares = SB.condition_limit_shares(5, conds)
    assert sum(shares.values()) == 5 and sorted(shares.values()) == [2, 3]
    assert SB.condition_limit_shares(5, ["without-skills"]) is None   # TASK_LIMIT 5 of it
    assert SB.condition_limit_shares(None, conds) is None
    plan = SB.limit_plan(5, conds, shares, [])
    assert plan["samples"] == 5 and plan["per_setting"] == shares
    assert SB.limit_plan(5, ["without-skills"], None, [])["samples"] == 5


def test_skillsbench_launcher_draws_nested_task_sets(monkeypatch):
    tasks = [f"t{i:02d}" for i in range(30)]
    monkeypatch.setattr(SRUN, "_task_category", lambda t: int(t[1:]) % 4)
    got = SRUN.condition_tasks(tasks, {"with-skills": 3, "without-skills": 2})
    assert len(got["with-skills"]) == 3 and len(got["without-skills"]) == 2
    assert set(got["without-skills"]) <= set(got["with-skills"])
    assert SRUN.condition_tasks(tasks, {"with-skills": 0})["with-skills"] == []


# --------------------------------------------------------------------------------------------
# one flattened list: the stratified draw is N in total
# --------------------------------------------------------------------------------------------
def test_multi_language_and_group_draws_are_n_in_total():
    tasks = [SA.AgentTask(instance_id=f"{lang}-{i}", problem_statement="", image="",
                          category=lang)
             for lang in ("c", "cpp", "go", "java", "js", "rust", "ts", "python")
             for i in range(10)]
    picked, _ = SA.select_tasks(tasks, 5, "multi_swe_bench")
    assert len(picked) == 5 and len({t.category for t in picked}) == 5
    assert len(SA.select_tasks(tasks, None, "multi_swe_bench")[0]) == 80
    groups = {g: [str(i) for i in range(20)] for g in ("G1_instruction", "G1_tool", "G2_category",
                                                        "G3_instruction")}
    sel, _, n_full = T.select_query_ids(groups, 5)
    assert sum(len(v) for v in sel.values()) == 5 and n_full == 80
    assert sum(len(v) for v in T.select_query_ids(groups, None)[0].values()) == 80
    names = [f"task-{i}" for i in range(89)]
    assert len(TB.select_task_names(names, None, 5)) == 5
    assert len(TB.select_task_names(names, None, None)) == 89


def test_the_agentic_runner_logs_and_records_its_plan(caplog):
    from gbench.runners.eval_suites.limit_plan import eval_limit_plan
    with caplog.at_level("INFO"):
        plan = eval_limit_plan("swe_bench_live", 5, 5, {"lite": 1, "full": 0},
                               setting_noun="tab", attempts=1, unit="rollouts")
    assert plan == {"limit": 5, "samples": 5, "per_setting": {"lite": 1, "full": 0},
                    "attempts_per_sample": 1, "rollouts": 5}
    assert "[swe_bench_live] --eval-limit 5: 5 sample(s) over 1 tab(s)" in caplog.text
    src = (REPO / "gbench" / "runners" / "eval_suites" / "swe_agentic.py").read_text()
    assert '"eval_limit_plan": plan' in src


def test_swe_lancer_eval_limit_1_runs_one_task_of_one_type(monkeypatch):
    sys.path.insert(0, str(REPO / "tests"))
    from test_swe_endpoint_relay_live_tabs_and_lancer import _lancer_stubs, _row, _tasks
    rows = {"ic_swe": [_row(f"ic{i}", "ic_swe", 100.0, True) for i in range(3)],
            "swe_manager": [_row(f"m{i}", "swe_manager", 1000.0, True) for i in range(2)]}
    runs = _lancer_stubs(monkeypatch, rows, _tasks())
    res = SL.run_swe_lancer("m", "http://127.0.0.1:1/v1", 1, limit=1)
    assert len(runs) == 1                                   # the other type is not run
    (taskset,) = [a.split("=", 1)[1] for a in runs[0] if a.startswith("swelancer.taskset=")]
    assert len(taskset.strip("[]").split(",")) == 1
    assert res["eval_limit_plan"]["samples"] == 1 and len(res["variants"]) == 1
