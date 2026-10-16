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

"""Shards, subsets and completeness in the container suites.

terminal_bench
  * a run without a shard or limit expects the pinned dataset's 89 tasks x k trials: a Harbor
    that stops early leaves the missing trials as harness errors (completed_with_errors, partial,
    not comparable), and a non-zero Harbor exit is never a clean run;
  * merge_shards re-counts the union of the shards' trials (harness errors, selections, attempts);
ui_control_osworld
  * the shard is validated on the host, forwarded normalized and recorded (shard_applied), an
    empty shard withholds its score, an image that does not report the shard or the limit draw is
    refused, every run gets its own directory, and merge_shards re-scores the union;
  * the launcher stops on a malformed shard and draws --eval-limit across the domains;
the container launchers (osworld, wildclawbench, skillsbench, mcp_bench, gaia2)
  * draw --eval-limit with a copy of sampling.stratified_indices that stays equal to it, and each
    host refuses a limited run whose launcher does not report the draw;
tau2 / tau3
  * --eval-limit is a seeded draw passed to tau2 as task_ids, not tau2's head of the file;
docs
  * every documented eval command passes --max-output-tokens (fences parsed as CommonMark does).

Docker, Harbor, Gemini and the network are stubbed; nothing is written outside tmp.
"""

import json
import os
import random
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gbench.merge import merge_suite
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import terminal_bench as TB
from gbench.runners.eval_suites import ui_control_osworld as U
from gbench.runners.eval_suites import wildclawbench as W
from gbench.runners.eval_suites.sampling import stratified_indices, stratified_sample

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gbench" / "docker"))
sys.path.insert(0, str(REPO / "tests"))
import gaia2_run as GRUN  # noqa: E402
import mcp_bench_run as MRUN  # noqa: E402
import mobilegym_run as MGRUN  # noqa: E402
import skillsbench_run as SRUN  # noqa: E402
import ui_control_osworld_run as ORUN  # noqa: E402
import wildclawbench_run as WRUN  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("GBENCH_SHARD", "VLLM_API_KEY", "OPENAI_API_KEY", "GBENCH_OSWORLD_LIMIT",
                 "GBENCH_OSWORLD_DOMAIN", "GBENCH_OSWORLD_MODEL_API_KEY", "OSWORLD_LIMIT",
                 "OSWORLD_DOMAIN", "OSWORLD_RUN_DIR", "WILDCLAW_LIMIT", "WILDCLAW_TASK",
                 "WILDCLAW_CATEGORIES", "SKILLSBENCH_TASK_LIMIT", "SKILLSBENCH_TASKS",
                 "GBENCH_SKILLSBENCH_TASKS", "GAIA2_LIMIT", "GAIA2_MODE", "GAIA2_SHARD",
                 "GBENCH_TB_TURN_MAX_TOKENS", "GBENCH_TB_TIMEOUT_MULTIPLIER", "GBENCH_TB_PARSER",
                 "GBENCH_TB_CONTAINER_THREADS", "GBENCH_TERMINAL_BENCH_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    base._RUN_KNOBS.clear()
    yield
    base._RUN_KNOBS.clear()


# =========================================================================== #
# terminal_bench: every expected trial is accounted for
# =========================================================================== #
def _tb_trial(root, name, reward, task, exc=None):
    trial = os.path.join(root, "job", name)
    os.makedirs(os.path.join(trial, "agent"), exist_ok=True)
    data = {"trial_name": name, "task_name": task,
            "verifier_result": {"rewards": {"reward": reward}},
            "agent_result": {"n_output_tokens": 10, "metadata": {"n_episodes": 1}}}
    if exc:
        # raised before the agent started (the environment did not come up): a harness error
        data["exception_info"] = {"exception_type": exc, "exception_message": "down"}
        data["verifier_result"] = None
    with open(os.path.join(trial, "result.json"), "w") as f:
        json.dump(data, f)


def _tb_run(monkeypatch, tmp_path, trials, *, returncode=0, full=None):
    # Harbor's jobs dirs go under the test's tmp_path (and are kept there)
    monkeypatch.setenv("GBENCH_TB_KEEP_JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))
    monkeypatch.setattr(TB, "dns_recheck", lambda task_names, hosts: list(hosts))
    if full is not None:
        monkeypatch.setattr(TB, "_dataset_task_names", lambda dataset: list(full))
    seen = {}

    class _Proc:
        stdout = iter(())

        def __init__(self):
            self.returncode = returncode

        def poll(self):
            return returncode

        def wait(self):
            return returncode

    def popen(cmd, **kw):
        seen["cmd"] = cmd
        jobs = cmd[cmd.index("--jobs-dir") + 1]
        for args in trials:
            _tb_trial(jobs, *args)
        return _Proc()
    monkeypatch.setattr(TB.subprocess, "Popen", popen)
    # --thinking: the run whose rule temperature (1.0) is the TB2.1 leaderboard's
    return TB.run_terminal_bench("m", "http://h:1/v1", 1, enable_thinking=True), seen


def _five(task, rewards=(1.0, 1.0, 0.0, 0.0, 1.0), exc_at=None):
    name = task.split("/")[-1]
    return [(f"{name}__{k}", r, task, "EnvironmentStartTimeoutError" if k == exc_at else None)
            for k, r in enumerate(rewards)]


def test_a_full_run_that_harbor_ended_early_is_never_a_clean_score(monkeypatch, tmp_path):
    # Harbor killed (137) after writing 3 tasks x 5 trials of the pinned 89-task dataset
    assert TB.FULL_TASK_COUNT == 89 and TB.TERMINAL_BENCH_TRIALS == 5
    trials = [t for i in range(3) for t in _five(f"terminal-bench/t{i}")]
    res, seen = _tb_run(monkeypatch, tmp_path, trials, returncode=137)
    assert "--include-task-name" not in seen["cmd"]              # a full run
    assert res["expected_trials"] == 445
    assert res["harness_errors"] == {"missing": 430}
    assert res["failed_requests"] == 430 and res["partial"] is True
    assert res["status"] == "completed_with_errors"
    assert res["leaderboard_comparable"] is False
    reason = res["leaderboard_comparable_reason"]
    assert "430 trial(s) of the selected tasks without a result (89 task(s) x 5 trials" in reason
    assert "Harbor exited with status 137" in reason
    assert res["harbor_returncode"] == 137
    assert "430 of the 445 expected trial(s) have no result" in res["status_reason"]
    assert "Harbor exited with status 137" in res["status_reason"]
    assert res["setting_deviations"] == []
    # the score is over the trials Harbor wrote (never the missing ones as 0)
    assert res["total_questions"] == 15 and res["accuracy"] == 60.0


def test_a_complete_full_run_is_canonical_and_a_nonzero_exit_is_not(monkeypatch, tmp_path):
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", 2)
    trials = _five("terminal-bench/a") + _five("terminal-bench/b")
    res, _ = _tb_run(monkeypatch, tmp_path, trials)
    assert res["status"] == "success" and res["status_reason"] is None
    assert res["harness_errors"] == {} and res["expected_trials"] == 10
    assert res["leaderboard_comparable"] is True
    assert res["leaderboard_comparable_reason"] == "full canonical run"
    res, _ = _tb_run(monkeypatch, tmp_path, trials, returncode=1)
    assert res["status"] == "completed_with_errors" and res["partial"] is False
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith("Harbor exited with status 1")


FULL4 = [f"terminal-bench/t{i}" for i in range(4)]


def _tb_shard(monkeypatch, tmp_path, label, trials_for, returncode=0):
    monkeypatch.setenv("GBENCH_SHARD", label)
    names = TB.select_task_names(FULL4, tuple(int(x) for x in label.split("/")), None)
    trials = [t for n in names for t in trials_for(n)]
    res, _ = _tb_run(monkeypatch, tmp_path, trials, full=FULL4, returncode=returncode)
    return res


def test_merge_recounts_the_union_of_the_shards_trials(monkeypatch, tmp_path):
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", 4)
    a = _tb_shard(monkeypatch, tmp_path, "1/2", _five)
    b = _tb_shard(monkeypatch, tmp_path, "2/2",
                  lambda n: _five(n, exc_at=0 if n.endswith("t1") else None))
    assert a["selected_tasks"] == ["terminal-bench/t0", "terminal-bench/t2"]
    assert b["harness_errors"] == {"exception": 1}
    merged = merge_suite("terminal_bench", [a, b])            # the suite's own hook
    assert merged["selected_tasks"] == FULL4 and merged["is_full"] is True
    assert merged["harness_errors"] == {"exception": 1}
    assert merged["harness_error_trials"] == ["t1__0"]
    assert merged["total_questions"] == merged["effective_n"] == 19
    assert merged["correct_answers"] == 11 and merged["accuracy"] == round(1100 / 19, 2)
    assert merged["failed_requests"] == 1 and merged["expected_trials"] == 20
    assert merged["attempts"]["samples"] == 4 and merged["attempts"]["generations"] == 19
    assert merged["status"] == "completed_with_errors"
    assert merged["leaderboard_comparable"] is False
    reason = merged["leaderboard_comparable_reason"]
    assert "1 trial(s) lost to harness errors (exceptions" in reason
    assert "sharded run" not in reason
    assert set(merged["duration_s_by_shard"]) == {"1/2", "2/2"}


def test_a_clean_complete_merge_is_canonical_and_a_missing_task_is_named(monkeypatch, tmp_path):
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", 4)
    a = _tb_shard(monkeypatch, tmp_path, "1/2", _five)
    b = _tb_shard(monkeypatch, tmp_path, "2/2", _five)
    merged = merge_suite("terminal_bench", [a, b])
    assert merged["status"] == "success" and merged["leaderboard_comparable"] is True
    assert merged["leaderboard_comparable_reason"] == "full canonical run"
    # shard 2's Harbor wrote only t1's trials and exited 137
    b = _tb_shard(monkeypatch, tmp_path, "2/2", lambda n: _five(n) if n.endswith("t1") else [],
                  returncode=137)
    merged = merge_suite("terminal_bench", [a, b])
    assert merged["harness_errors"] == {"missing": 5} and merged["partial"] is True
    reason = merged["leaderboard_comparable_reason"]
    assert "5 trial(s) of the selected tasks without a result (4 task(s) x 5 trials" in reason
    assert "shard 2/2: Harbor exited with status 137" in reason


def test_merge_keeps_setting_deviations_and_names_an_incomplete_union(monkeypatch, tmp_path):
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", 5)                 # a task no shard selected
    monkeypatch.setenv("GBENCH_TB_PARSER", "xml")
    a = _tb_shard(monkeypatch, tmp_path, "1/2", _five)
    b = _tb_shard(monkeypatch, tmp_path, "2/2", _five)
    merged = TB.merge_shards([a, b])
    reason = merged["leaderboard_comparable_reason"]
    assert "parser='xml'" in reason and reason.count("parser='xml'") == 1
    assert "the merged shards hold 4 tasks, not the full 5" in reason
    assert merged["is_full"] is False


# =========================================================================== #
# ui_control_osworld: host side
# =========================================================================== #
def _osw(monkeypatch, tmp_path, summary, **kwargs):
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(U.swebench_common, "require_task_network_blocked", lambda *a, **k: None)
    seen = {"runs": []}

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            seen["runs"].append(list(cmd))
            run_dir = next(c.split("=", 1)[1] for c in cmd if c.startswith("OSWORLD_RUN_DIR="))
            if summary is not None:
                (Path(run_dir) / "ui_control_osworld_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(U.subprocess, "run", fake_run)
    return U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:1/v1",
                                    **kwargs), seen


def _osw_summary(tasks, shard=None, limit=None, n_full=6):
    scored = [v for _t, v in tasks if v is not None]
    return {"success_rate": (sum(scored) / len(tasks)) if tasks else None,
            "n_selected": len(tasks), "n_scored": len(scored),
            "n_missing": len(tasks) - len(scored), "per_domain": {},
            "tasks": [{"task": t, "domain": t.split("/")[0], "example_id": t.split("/")[1],
                       "result": v} for t, v in tasks],
            "selection": {"n_full": n_full, "limit": limit, "n_selected": len(tasks),
                          "limit_draw": ORUN.LIMIT_DRAW if limit else None},
            "shard": ({"spec": shard, "n_full": n_full, "n_selected": len(tasks)}
                      if shard else None)}


def test_osworld_records_the_shard_it_forwarded_normalized(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_SHARD", " 2/3 ")
    res, seen = _osw(monkeypatch, tmp_path,
                     _osw_summary([("os/a", 1.0), ("calc/b", None)], shard="2/3"))
    [cmd] = seen["runs"]
    assert "GBENCH_SHARD=2/3" in cmd
    assert res["shard"] == "2/3" and res["accuracy"] == 100.0
    assert "sharded run (--shard 2/3)" in res["leaderboard_comparable_reason"]
    assert [t["sample_key"] for t in res["sample_traces"]] == ["task:os/a", "task:calc/b"]
    assert res["sample_traces"][1]["scoring_excluded"] is True
    stamped = base.finalize_shard_fields(dict(res))
    assert stamped["shard_applied"] is True and stamped["shard_index"] == 2


def test_osworld_validates_the_shard_before_anything_starts(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_SHARD", "4/3")
    with pytest.raises(ValueError):
        _osw(monkeypatch, tmp_path, None)
    monkeypatch.setenv("GBENCH_SHARD", "1/1")                     # the whole set: not forwarded
    res, seen = _osw(monkeypatch, tmp_path, _osw_summary([("os/a", 1.0)]))
    assert not any(c.startswith("GBENCH_SHARD=") for c in seen["runs"][0])
    assert "shard" not in res


def test_osworld_empty_shard_withholds_the_score(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_SHARD", "9/9")
    summary = _osw_summary([], shard="9/9", n_full=6)
    summary["empty_selection"] = True
    res, _ = _osw(monkeypatch, tmp_path, summary)
    assert res["status"] == "success" and res["accuracy"] is None and res["empty_shard"]
    assert res["accuracy_withheld_reason"] == (
        "empty shard 9/9: no items on this node (6 in the full set)")
    assert res["shard"] == "9/9" and res["full_set_items"] == 6


def test_osworld_refuses_an_image_that_does_not_report_the_shard_or_the_draw(monkeypatch,
                                                                             tmp_path):
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    stale = _osw_summary([("os/a", 1.0)])                         # no shard record
    with pytest.raises(RuntimeError, match="did not report applying it"):
        _osw(monkeypatch, tmp_path, stale)
    monkeypatch.delenv("GBENCH_SHARD")
    with pytest.raises(RuntimeError, match="how it drew the --eval-limit subset"):
        _osw(monkeypatch, tmp_path, _osw_summary([("os/a", 1.0)]), limit=2)
    res, _ = _osw(monkeypatch, tmp_path, _osw_summary([("os/a", 1.0), ("calc/b", 0.0)],
                                                      limit=2), limit=2)
    assert res["selection"]["limit_draw"] == U.LIMIT_DRAW == ORUN.LIMIT_DRAW


def test_osworld_runs_in_a_fresh_directory_and_never_reads_an_old_summary(monkeypatch, tmp_path):
    # a summary left in the VM dir by an earlier run (or an older image) is never read
    (tmp_path / "ui_control_osworld_summary.json").write_text(json.dumps(
        _osw_summary([("os/a", 1.0)])), encoding="utf-8")
    with pytest.raises(RuntimeError, match="produced no summary"):
        _osw(monkeypatch, tmp_path, None)
    res1, s1 = _osw(monkeypatch, tmp_path, _osw_summary([("os/a", 1.0)]))
    res2, s2 = _osw(monkeypatch, tmp_path, _osw_summary([("os/a", 0.0)]))
    assert res1["run_dir"] != res2["run_dir"]
    for res in (res1, res2):
        assert os.path.dirname(res["run_dir"]) == str(tmp_path / "gbench_runs")
    assert res2["accuracy"] == 0.0


def test_osworld_merge_rescores_the_union(monkeypatch, tmp_path):
    rows = {"1/2": [("os/a", 1.0), ("calc/c", None)], "2/2": [("os/b", 0.0), ("calc/d", 0.5)]}
    shards = []
    for label, tasks in rows.items():
        monkeypatch.setenv("GBENCH_SHARD", label)
        res, _ = _osw(monkeypatch, tmp_path, _osw_summary(tasks, shard=label))
        shards.append(res)
    merged = merge_suite("ui_control_osworld", shards)
    assert merged["n_selected"] == 4 and merged["n_scored"] == 3 and merged["n_missing"] == 1
    assert merged["accuracy"] == 50.0
    assert merged["per_domain"]["calc"]["success_rate"] == 0.5
    assert merged["status"] == "completed_with_errors"
    assert "sharded run" not in merged["leaderboard_comparable_reason"]
    assert merged["leaderboard_comparable"] is False


#: The fingerprint of the osworld launcher files at each launcher label value.
_OSWORLD_LAUNCHER_SHA256 = {
    "2": "39ed327414bedef2ca4526f3b1efee87baa03d35a49826072d6c961c5dac69be",
    "3": "2c566e80ca22f35a0fc2dc872d4ee5bd33a98de8a0f979858c7dc299454b37f9",
    "4": "6729bb0f034f0bae30b0b61b57fba7685d84bb1a0b88e115f69fe5ff9d0d45bc",
}


def test_the_osworld_label_follows_the_launcher():
    import hashlib
    df = (REPO / "gbench" / "docker" / "ui_control_osworld.Dockerfile").read_text()
    assert f'LABEL {U.LAUNCHER_LABEL}="{U.LAUNCHER_VERSION}"' in df
    digest = hashlib.sha256()
    for name in ("ui_control_osworld_run.py", "ui_control_osworld_entrypoint.sh"):
        digest.update((REPO / "gbench" / "docker" / name).read_bytes())
    assert _OSWORLD_LAUNCHER_SHA256.get(U.LAUNCHER_VERSION) == digest.hexdigest(), (
        "the osworld launcher changed: bump ui_control_osworld.LAUNCHER_VERSION and the "
        "Dockerfile label, and record the new fingerprint here")
    doc = (REPO / "docs" / "evals" / "ui_control_osworld.md").read_text()
    assert f"`{U.LAUNCHER_LABEL}={U.LAUNCHER_VERSION}`" in doc


def test_the_osworld_gate_refuses_an_image_that_predates_the_launcher(monkeypatch, tmp_path):
    monkeypatch.setattr(U.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(U, "_docker_image_exists", lambda _i: True)
    real_exists = os.path.exists
    monkeypatch.setattr(U.os.path, "exists", lambda p: p == "/dev/kvm" or real_exists(p))
    (tmp_path / "docker_vm_data").mkdir()
    (tmp_path / "docker_vm_data" / "Ubuntu.qcow2").write_text("x")
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    for label in ("", "1"):
        monkeypatch.setattr(U.subprocess, "run", lambda *a, _l=label, **k: types.SimpleNamespace(
            returncode=0, stderr="", stdout=_l if "--format" in a[0] else ""))
        ok, why = U.check_ui_control_osworld_prerequisites()
        assert not ok and f"{U.LAUNCHER_LABEL}={U.LAUNCHER_VERSION}" in why
        assert "Rebuild it" in why and "docker build" in why
    monkeypatch.setattr(U.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stderr="", stdout=U.LAUNCHER_VERSION if "--format" in a[0] else ""))
    assert U.check_ui_control_osworld_prerequisites() == (True, "")


# =========================================================================== #
# ui_control_osworld: the launcher
# =========================================================================== #
def _osworld_tree(tmp_path, meta):
    osw = tmp_path / "OSWorld"
    (osw / "evaluation_examples").mkdir(parents=True)
    (osw / "evaluation_examples" / "test_all.json").write_text(json.dumps(meta))
    return str(osw)


@pytest.mark.parametrize("spec", ["x", "0/3", "4/3", "1/0", "a/b"])
def test_the_osworld_launcher_stops_on_a_malformed_shard(monkeypatch, spec):
    monkeypatch.setenv("GBENCH_SHARD", spec)
    with pytest.raises(SystemExit) as ei:
        ORUN._shard()
    assert ei.value.code not in (0, None)


def test_the_osworld_launcher_draws_the_limit_across_the_domains(monkeypatch, tmp_path):
    meta = {"chrome": [f"c{i}" for i in range(5)], "os": [f"o{i}" for i in range(5)],
            "vlc": [f"v{i}" for i in range(5)]}
    monkeypatch.setattr(ORUN, "OSWORLD_DIR", _osworld_tree(tmp_path, meta))
    monkeypatch.setattr(ORUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("OSWORLD_LIMIT", "3")
    _p, sel, selection = ORUN._build_subset_meta()
    assert sorted(sel) == ["chrome", "os", "vlc"]                 # the head would be 3 chrome
    assert selection["limit_draw"] == ORUN.LIMIT_DRAW and selection["n_full"] == 15
    flat = sorted((d, e) for d, ids in meta.items() for e in ids)
    want = [flat[k] for k in stratified_indices([d for d, _ in flat], 3, "ui_control_osworld")]
    assert sorted((d, e) for d, ids in sel.items() for e in ids) == sorted(want)
    monkeypatch.setenv("OSWORLD_LIMIT", "two")
    with pytest.raises(SystemExit):
        ORUN._build_subset_meta()


def test_the_osworld_launcher_does_not_start_osworld_for_an_empty_shard(monkeypatch, tmp_path):
    monkeypatch.setattr(ORUN, "OSWORLD_DIR", _osworld_tree(tmp_path, {"os": ["a", "b"]}))
    monkeypatch.setattr(ORUN, "WORKDIR", str(tmp_path))
    run_dir = tmp_path / "gbench_runs" / "r1"
    monkeypatch.setenv("OSWORLD_RUN_DIR", str(run_dir))
    monkeypatch.setenv("GBENCH_SHARD", "3/3")
    monkeypatch.setattr(ORUN, "_install_model_rewrite", lambda served: None)
    monkeypatch.setattr(ORUN, "_run_osworld", lambda p: pytest.fail("OSWorld started"))
    monkeypatch.chdir(tmp_path)
    assert ORUN.main() == 0
    summary = json.loads((run_dir / "ui_control_osworld_summary.json").read_text())
    assert summary["n_selected"] == 0 and summary["empty_selection"] is True
    assert summary["shard"] == {"spec": "3/3", "n_full": 2, "n_selected": 0}
    assert not (tmp_path / "ui_control_osworld_summary.json").exists()


# =========================================================================== #
# The launchers' copies of sampling.stratified_indices, and their draws
# =========================================================================== #
@pytest.mark.parametrize("launcher", [ORUN, WRUN, SRUN, MRUN, GRUN, MGRUN],
                         ids=["osworld", "wildclawbench", "skillsbench", "mcp_bench", "gaia2",
                              "mobilegym"])
def test_every_launcher_copy_of_stratified_indices_equals_samplings(launcher):
    rnd = random.Random(7)
    for _ in range(200):
        n = rnd.randint(0, 40)
        keys = [rnd.choice(["a", "b", "c", None, ("t", 1), "d"]) for _ in range(n)]
        limit = rnd.choice([None, 0, -1, 1, 2, 5, n, n + 3, rnd.randint(0, n + 1)])
        seed = rnd.choice(["wildclawbench", "gaia2", "mcp_bench", None, "x"])
        assert launcher.stratified_indices(keys, limit, seed) == stratified_indices(
            keys, limit, seed)


def test_the_wildclawbench_launcher_draws_the_limit_across_the_categories(monkeypatch):
    index = {f"{c}_task_{i}": {"category": c, "modality": "pure-text", "path": f"/t/{c}/{i}.md"}
             for c in WRUN.ALL_CATEGORIES for i in range(1, 11)}
    monkeypatch.setenv("WILDCLAW_LIMIT", "6")
    got = WRUN._selected_task_ids(index)
    assert len(got) == 6 and {index[t]["category"] for t in got} == set(WRUN.ALL_CATEGORIES)
    monkeypatch.setenv("WILDCLAW_LIMIT", "six")
    with pytest.raises(SystemExit):
        WRUN._selected_task_ids(index)


def test_the_skillsbench_launcher_draws_the_limit_across_the_categories(monkeypatch, tmp_path):
    cats = {"a1": "physics", "a2": "physics", "a3": "physics", "b1": "finance", "c1": "security"}
    for task, cat in cats.items():
        (tmp_path / "tasks" / task).mkdir(parents=True)
        (tmp_path / "tasks" / task / "task.md").write_text(
            f"---\nmetadata:\n  category: {cat}\n  secondary_category: other\n---\nbody\n")
    monkeypatch.setattr(SRUN, "SKILLSBENCH_DIR", str(tmp_path))
    monkeypatch.setenv("SKILLSBENCH_TASK_LIMIT", "3")
    got = SRUN._task_list()
    assert len(got) == 3 and {cats[t] for t in got} == {"physics", "finance", "security"}
    assert SRUN._task_category("b1") == "finance"
    monkeypatch.setenv("SKILLSBENCH_TASK_LIMIT", "x")
    with pytest.raises(SystemExit):
        SRUN._task_list()


def test_the_mcp_bench_launcher_draws_the_limit_across_the_server_groups(tmp_path):
    groups = [{"servers": [f"S{g}"], "tasks": [{"task_id": f"g{g}_t{i}"} for i in range(4)]}
              for g in range(3)]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"server_tasks": groups}))
    got = MRUN._limited_task_ids([str(path)], None, 3)
    assert {t.split("_")[0] for t in got} == {"g0", "g1", "g2"}   # file order: all of g0
    shard = MRUN._sharded_task_ids([str(path)], (1, 2))
    assert MRUN._limited_task_ids([str(path)], shard, 2) <= shard


def test_the_gaia2_hook_draws_the_limit_from_the_stream(monkeypatch, tmp_path):
    rows = [{"scenario_id": f"s{i:02d}", "data": "{}"} for i in range(12)]

    class Split(list):
        def filter(self, fn):
            return Split(r for r in self if fn(r))

    class IDD(dict):
        pass
    fake = types.ModuleType("datasets")
    fake.IterableDatasetDict = IDD
    fake.load_dataset = lambda path, *a, **k: IDD(validation=Split(rows))
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setattr(GRUN, "DATASET", "/opt/gaia2/gaia2")
    GRUN.install_shard_hook(None, 3)
    got = [r["scenario_id"] for r in fake.load_dataset("/opt/gaia2/gaia2", name="time")[
        "validation"]]
    ids = sorted(r["scenario_id"] for r in rows)
    assert got == [ids[k] for k in stratified_indices([None] * 12, 3, "gaia2")]
    assert got != ids[:3] or got == sorted(got)
    assert GRUN.shard_summary(str(tmp_path)) is None              # no shard in this run
    sel = GRUN.selection_summary(str(tmp_path))
    assert sel["limit"] == 3 and sel["limit_draw"] == GRUN.LIMIT_DRAW
    assert sel["per_config"][0]["scenario_ids"] == got
    assert GRUN.select_scenario_ids([r["scenario_id"] for r in rows], (2, 3), 2) == [
        GRUN.shard_scenario_ids(ids, (2, 3))[k]
        for k in stratified_indices([None] * 4, 2, "gaia2")]
    with pytest.raises(ValueError):
        GRUN.parse_limit("many")


# =========================================================================== #
# The hosts refuse a limited run whose launcher does not report its draw
# =========================================================================== #
def test_skillsbench_refuses_a_limited_run_without_the_draw(monkeypatch):
    from test_shard_harness_suites import _sb_run, _summary
    with pytest.raises(RuntimeError, match="how it drew the --eval-limit subset"):
        _sb_run(monkeypatch, _summary(["a", "b"]), limit=2)
    summary = dict(_summary(["a", "b"]), selection={"limit": 2, "limit_draw": SRUN.LIMIT_DRAW})
    res, seen = _sb_run(monkeypatch, summary, limit=2)
    assert "SKILLSBENCH_TASK_LIMIT=2" in seen["cmd"]
    assert res["selection"]["limit_draw"] == SRUN.LIMIT_DRAW
    assert SB.HARNESS_VERSION == "7"
    assert f'LABEL {SB.HARNESS_LABEL}="7"' in (
        REPO / "gbench" / "docker" / "skillsbench.Dockerfile").read_text()


def _wildclaw(monkeypatch, tmp_path, limit, summary_extra):
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.container_guard, "require_metadata_blocked", lambda *a, **k: None)
    wd = tmp_path / "wc_wd"
    wd.mkdir(exist_ok=True)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(wd))
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTfixtureKEY000000000000000000000")
    seen = {}

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
        if cmd[:3] == ["docker", "run", "--rm"]:
            seen["cmd"] = list(cmd)
            summary = {"n_tasks": 6, "n_scored": 6, "n_missing": 0, "global_mean": 0.5,
                       "judge_fallback_tasks": 0, "n_search_setup_failures": 0,
                       "n_harness_errors": 0, "harness_error_kinds": {}, "harness_events": {},
                       "n_unattributed_harness_events": 0}
            summary.update(summary_extra)
            with open(wd / "wildclawbench_summary.json", "w") as f:
                json.dump(summary, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    return W.run_wildclawbench("m", "http://127.0.0.1:9123/v1", limit=limit), seen


def test_wildclawbench_refuses_a_limited_run_without_the_draw(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="how it drew the --eval-limit subset"):
        _wildclaw(monkeypatch, tmp_path, 6, {})
    res, seen = _wildclaw(monkeypatch, tmp_path, 6, {"selection": {
        "limit": 6, "limit_draw": WRUN.LIMIT_DRAW}})
    assert "WILDCLAW_LIMIT=6" in seen["cmd"]
    assert res["selection"]["limit_draw"] == WRUN.LIMIT_DRAW


def test_gaia2_refuses_a_limited_run_without_the_draw(monkeypatch, tmp_path):
    from test_gaia2_toolbench_judge_protocol_and_credentials import _gaia2_run
    with pytest.raises(RuntimeError, match="how it drew the --eval-limit scenarios"):
        _gaia2_run(monkeypatch, tmp_path, limit=5)
    sel = {"limit": 1, "limit_draw": GRUN.LIMIT_DRAW, "per_config": []}
    res, [(cmd, _kw)], _ = _gaia2_run(monkeypatch, tmp_path, summary={"selection": sel},
                                      limit=5)
    # --eval-limit 5 is 5 scenarios for the run: 1 of each of the 5 configs, never 5 per config
    limits = json.loads(next(c for c in cmd if c.startswith("GAIA2_CONFIG_LIMITS="))
                        .split("=", 1)[1])
    assert sum(limits.values()) == 5 and set(limits.values()) == {1}
    assert not any(c.startswith("GAIA2_LIMIT=") for c in cmd)
    assert res["selection"]["limit_draw"] == GRUN.LIMIT_DRAW
    assert res["eval_limit_plan"]["samples"] == 5


def test_mcp_bench_names_the_seeded_draw():
    src = (REPO / "gbench" / "runners" / "eval_suites" / "mcp_bench.py").read_text()
    assert 'not (manifest or {}).get("limit_draw")' in src
    assert MRUN.LIMIT_DRAW == "stratified_indices(key=server group, seed=mcp_bench)"


# =========================================================================== #
# tau2 / tau3: the limit is a seeded draw passed as task_ids
# =========================================================================== #
def _fake_tau2(monkeypatch, ids_by_domain):
    sim = types.ModuleType("tau2.data_model.simulation")
    sim.TextRunConfig = types.SimpleNamespace(
        model_fields={"task_split_name": types.SimpleNamespace(default="base")})
    helpers = types.ModuleType("tau2.runner.helpers")
    helpers.get_tasks = lambda task_set_name, task_split_name: [
        types.SimpleNamespace(id=i) for i in ids_by_domain[task_set_name]]
    for name in ("tau2", "tau2.data_model", "tau2.runner"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "tau2.data_model.simulation", sim)
    monkeypatch.setitem(sys.modules, "tau2.runner.helpers", helpers)


def test_tau_limit_is_a_seeded_draw_not_tau2s_head(monkeypatch, tmp_path):
    ids = [str(i) for i in range(20)]
    _fake_tau2(monkeypatch, {"retail": ids, "airline": ids})
    got, n = TC._shard_task_ids("retail", None, 4, seed="tau2")
    assert n == 20 and got == stratified_sample(ids, 4, seed="tau2") and got != ids[:4]
    ran = []

    def one(domain, *a, task_ids=None, **k):
        ran.append((domain, list(task_ids)))
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": 0.5}
        TC._DOMAIN_TASK_OUTCOMES[domain] = {t: [1, 1] for t in task_ids}
        return len(task_ids), 1.0, 1, 0
    monkeypatch.setattr(TC, "_run_one_domain", one)
    res = TC._run_domains("tau2", ["retail", "airline"], "m", "http://h/v1", 1, 2, "d",
                          False, 4, str(tmp_path), {}, "h", False, shard=None)
    # --eval-limit 2 is 2 tasks for the run (one per domain here), never 2 per domain
    assert ran == [("retail", stratified_sample(ids, 1, seed="tau2")),
                   ("airline", stratified_sample(ids, 1, seed="tau2"))]
    assert res["selected_task_ids"] == dict(ran) and "shard" not in res
    assert res["domain_task_counts"] == {"retail": 20, "airline": 20}
    assert res["eval_limit_plan"]["samples"] == 2


def test_tau_never_passes_the_limit_as_num_tasks():
    src = (REPO / "gbench" / "runners" / "eval_suites" / "tau_common.py").read_text()
    body = src[src.index("def _run_one_domain("):src.index("def _run_one_domain(") + 6000]
    assert "num_tasks=None," in body


# =========================================================================== #
# docs
# =========================================================================== #
def test_the_cyberseceval_run_command_passes_max_output_tokens():
    text = (REPO / "docs" / "evals" / "cyberseceval.md").read_text(encoding="utf-8")
    run = text[text.index("## Run"):text.index("## Notes")]
    block = re.search(r"```bash\n(.*?)\n```", run, re.S).group(1)
    assert "--evals cyberseceval" in block and "--max-output-tokens" in block
    # the prose line that quotes a fence is not one (its info string holds backticks)
    prose = [ln for ln in text.splitlines() if ln.strip().startswith("```<lang> hint line")]
    assert prose and all("`" in ln.strip()[3:] for ln in prose)


def test_the_doc_lint_fence_parser_follows_commonmark():
    from test_shard_harness_suites import _doc_commands
    text = ("- extracted like upstream: the first block including its\n"
            "  ```<lang> hint line (upstream `re.findall(r\"```(.*?)```\")[0]`), or the whole\n"
            "\n```bash\ngbench --evals-only --evals x \\\n   --eval-limit 2\n```\n")
    assert _doc_commands(text) == [(5, "gbench --evals-only --evals x  --eval-limit 2")]
