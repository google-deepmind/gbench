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

"""swe_lancer's task-table cache and swe_bench_live's bounded SWE-agent batches.

swe_lancer: a cache file of another shape (the list of issues an older gbench kept under the
same name) made load_tasks raise AttributeError ('list' object has no attribute 'get') before
any task ran; such a file is now read again from the image and replaced.

swe_bench_live: a SWE-agent-for-eval batch had no bound in gbench and nothing removed its
containers when gbench was stopped. The watchdog (SweagentWatch) names a stalled rollout
(harness error agent_stalled) and stops a batch that makes no progress, and every exit path
removes the batch's containers (label org.gbench.sweagent_batch). Docker, the model endpoint and
the network are stubbed; nothing is written outside tmp.
"""

import json
import os
import signal
import threading
import time
import types

import pytest

from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_lancer as SL
from gbench.runners.eval_suites import swebench_common as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UUID = "2f07fb32-3e8d-49dc-9395-c6328231d7db"


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TRAJ_DIR", str(tmp_path / "traj_live"))
    monkeypatch.setenv("GBENCH_SWE_LANCER_TRAJ_DIR", str(tmp_path / "traj_lancer"))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


# ============================================================================================
# swe_lancer: the task-table cache
# ============================================================================================
TABLE = {"sha256": SL.TASKS_SHA256, "tasks": [
    {"question_id": "1_1", "variant": "ic_swe", "set": "diamond", "price": 500.0},
    {"question_id": "2-manager-0", "variant": "swe_manager", "set": "diamond", "price": 1000.0}]}
#: The shape of the file the old SWELancer-Benchmark runner cached under the same name.
OLD_MANIFEST = [{"id": "1", "title": "[$250] Expense - Hold option is missing",
                 "issue_repo_steps": "..."}]


def _cache_path(tmp_path):
    return tmp_path / "cache" / "swe_lancer" / "issues_manifest.json"


def _table_runner(calls, table=TABLE, stdout=None):
    def _run(cmd, **k):
        calls.append(list(cmd))
        if SL._TASKS_SCRIPT in cmd:
            return _proc(stdout=(json.dumps(table) + "\n") if stdout is None else stdout)
        return _proc()
    return _run


@pytest.mark.parametrize("stale", [OLD_MANIFEST, {"sha256": "0" * 64, "tasks": TABLE["tasks"]},
                                   {"sha256": SL.TASKS_SHA256, "tasks": {"a": 1}},
                                   {"sha256": SL.TASKS_SHA256, "tasks": ["x"]}, "text", None])
def test_a_cache_file_of_another_shape_is_read_again_and_replaced(monkeypatch, tmp_path, stale):
    path = _cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stale), encoding="utf-8")
    calls = []
    monkeypatch.setattr(SL.subprocess, "run", _table_runner(calls))
    assert SL.load_tasks() == TABLE["tasks"]
    assert len(calls) == 1 and "--network" in calls[0] and "none" in calls[0]
    assert json.loads(path.read_text(encoding="utf-8")) == TABLE
    assert SL.load_tasks() == TABLE["tasks"] and len(calls) == 1      # now served from cache


def test_a_table_script_that_printed_nothing_is_a_named_error(monkeypatch):
    monkeypatch.setattr(SL.subprocess, "run", _table_runner([], stdout=""))
    with pytest.raises(RuntimeError, match="printed no table"):
        SL.load_tasks()
    monkeypatch.setattr(SL.subprocess, "run", _table_runner([], stdout="not json\n"))
    with pytest.raises(RuntimeError, match="printed no table"):
        SL.load_tasks()
    monkeypatch.setattr(SL.subprocess, "run", _table_runner(
        [], table={"sha256": SL.TASKS_SHA256, "tasks": []}))
    with pytest.raises(RuntimeError, match="no task rows"):
        SL.load_tasks()


def test_swe_lancer_runs_past_the_cache_file_an_older_gbench_wrote(monkeypatch, tmp_path):
    """The sweep's crash: `Eval 'swe_lancer' failed: 'list' object has no attribute 'get'`."""
    path = _cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(OLD_MANIFEST), encoding="utf-8")
    monkeypatch.setattr(SL, "check_swe_lancer_prerequisites", lambda: (True, ""))
    harness = []

    def _run(cmd, **k):
        if SL._TASKS_SCRIPT in cmd:
            return _proc(stdout=json.dumps(TABLE) + "\n")
        if cmd[:2] == ["docker", "run"]:
            harness.append(list(cmd))
            report = [a.split("=", 1)[1] for a in cmd if a.startswith("swelancer.report_path=")][0]
            with open(report, "w", encoding="utf-8") as f:
                json.dump({"summary": {}, "tasks": [
                    {"question_id": "1_1", "variant": "ic_swe", "price": 500.0, "correct": True,
                     "earned": 500.0, "available": 500.0, "system_error": None}]}, f)
        return _proc()
    monkeypatch.setattr(SL.subprocess, "run", _run)
    res = SL.run_swe_lancer("m", "http://127.0.0.1:9/v1", 1, variant="ic_swe")
    assert res["status"] == "success" and res["accuracy"] == 100.0
    assert len(harness) == 1 and 'swelancer.taskset=["1_1"]' in harness[0]
    assert json.loads(path.read_text(encoding="utf-8")) == TABLE


# ============================================================================================
# swe_bench_live: the SWE-agent command
# ============================================================================================
def test_the_batch_container_gets_an_init_and_the_task_containers_a_label(tmp_path):
    out = str(tmp_path)
    cmd = LV.sweagent_command(f"{out}/instances.jsonl", out, "org/model",
                              "http://127.0.0.1:4242/v1", 2, "gbench-sweagent-abc")
    image_at = cmd.index(LV.sweagent_image())
    assert "--init" in cmd[:image_at]
    assert ("--instances.deployment.docker_args=--label=org.gbench.sweagent_batch="
            "gbench-sweagent-abc") in cmd[image_at:]
    # the fork's own config, unmodified, is still the only config file
    assert cmd.count("--config") == 1
    assert cmd[cmd.index("--config") + 1] == "/opt/SWE-agent-for-eval/config/default.yaml"


def test_the_upstream_limits_and_the_watchdog_bounds_are_the_forks():
    assert LV.SWEAGENT_COMMAND_TIMEOUT_S == 90 * 60
    assert LV.SWEAGENT_MAX_CONSECUTIVE_TIMEOUTS == 3
    assert LV.SWEAGENT_TOTAL_EXECUTION_S == 5 * 60 * 60
    assert LV.SWEAGENT_CALL_LIMIT == 100
    # longer than any single silent step: a command, or a model request through the relay
    assert LV.SWEAGENT_STALL_S > LV.SWEAGENT_COMMAND_TIMEOUT_S
    assert LV.SWEAGENT_STALL_S > C.EndpointRelay.TIMEOUT_S
    assert (LV.SWEAGENT_STALL_S, LV.SWEAGENT_STALL_GRACE_S) == (6300, 600)


def test_the_task_container_names_follow_swe_rex():
    image = "starryzhang/sweb.eval.x86_64.mikedh_1776_trimesh-2354"
    pat = LV.task_container_name_re(image)
    # the name the sweep's run gave it (rex-deploy "Starting container ...")
    assert pat.fullmatch(f"starryzhangsweb.eval.x86_64.mikedh_1776_trimesh-2354-{UUID}")
    assert not pat.fullmatch(f"starryzhangsweb.eval.x86_64.mikedh_1776_trimesh-23541-{UUID}")
    assert not pat.fullmatch("starryzhangsweb.eval.x86_64.mikedh_1776_trimesh-2354-x")


# ============================================================================================
# swe_bench_live: the watchdog
# ============================================================================================
class FakeDocker:
    """`docker ps -a --filter label=...` / `docker rm -f ...` over a set of task containers."""

    def __init__(self, batch, names):
        self.batch = batch
        self.containers = {f"c{i}": n for i, n in enumerate(names)}
        self.calls = []

    def __call__(self, cmd, **k):
        self.calls.append(list(cmd))
        if cmd[:3] == ["docker", "ps", "-a"]:
            assert cmd[cmd.index("--filter") + 1] == f"label=org.gbench.sweagent_batch={self.batch}"
            return _proc(stdout="".join(f"{c}\t{n}\n" for c, n in self.containers.items()))
        if cmd[:3] == ["docker", "rm", "-f"]:
            for c in cmd[3:]:
                self.containers.pop(c, None)
        return _proc()

    def removed(self):
        return [c[3:] for c in self.calls if c[:3] == ["docker", "rm", "-f"]]


def _touch(path, when):
    """Create `path` with mtime `when`, and give its directory that mtime too (creating the file
    moved it to the real clock)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        pass
    os.utime(path, (when, when))
    os.utime(os.path.dirname(path), (when, when))


def _images(*iids):
    return {i: LV.python_instance_image(i) for i in iids}


def _task_name(iid):
    return LV.python_instance_image(iid).replace("/", "") + "-" + UUID


def test_a_stalled_rollout_loses_its_task_container_and_is_named(monkeypatch, tmp_path):
    out = str(tmp_path)
    now = 1_000_000.0
    docker = FakeDocker("b1", [_task_name("o__r-1"), _task_name("o__r-2")])
    monkeypatch.setattr(LV.subprocess, "run", docker)
    watch = LV.SweagentWatch(out, _images("o__r-1", "o__r-2", "o__r-3", "o__r-4"), "b1",
                             clock=lambda: now)
    stall = LV.SWEAGENT_STALL_S
    _touch(os.path.join(out, "o__r-1", "o__r-1.info.log"), now - stall - 1)     # stalled
    _touch(os.path.join(out, "o__r-2", "o__r-2.info.log"), now - stall + 60)    # alive
    _touch(os.path.join(out, "o__r-3", "o__r-3.info.log"), now - 10 * stall)    # finished
    _touch(os.path.join(out, "o__r-3", "o__r-3.pred"), now - 10 * stall)
    _touch(os.path.join(out, "sweagent_run.log"), now)                          # batch alive
    # o__r-4 has not started (no directory): never stalled
    watch.check(now)
    assert sorted(watch.stalled) == ["o__r-1"]
    assert "no file of this rollout changed" in watch.stalled["o__r-1"]
    assert docker.removed() == [["c0"]] and list(docker.containers) == ["c1"]
    assert watch.stopped is None
    watch.check(now + 5)                                        # flagged once
    assert docker.removed() == [["c0"]]
    rec = watch.record()
    assert rec["stalled_instances"] == ["o__r-1"] and rec["batch_stopped"] is None
    assert rec["removed_task_containers"] == {"o__r-1": [_task_name("o__r-1")]}
    assert rec["stall_limit_s"] == 6300 and rec["batch_stop_limit_s"] == 6900


def test_a_batch_with_no_progress_is_stopped(monkeypatch, tmp_path):
    out = str(tmp_path)
    start = 2_000_000.0
    docker = FakeDocker("b2", [_task_name("o__r-1")])
    monkeypatch.setattr(LV.subprocess, "run", docker)
    watch = LV.SweagentWatch(out, _images("o__r-1"), "b2", clock=lambda: start)
    _touch(os.path.join(out, "sweagent_run.log"), start)
    _touch(os.path.join(out, "o__r-1", "o__r-1.info.log"), start)
    for p in (os.path.join(out, "o__r-1"), out):
        os.utime(p, (start, start))
    limit = LV.SWEAGENT_STALL_S + LV.SWEAGENT_STALL_GRACE_S
    watch.check(start + limit - 1)
    assert watch.stopped is None and sorted(watch.stalled) == ["o__r-1"]
    watch.check(start + limit + 1)
    assert watch.stopped and "no file of the SWE-agent batch changed" in watch.stopped
    assert ["b2"] in docker.removed()                       # the SWE-agent container


def test_read_sweagent_run_names_a_stalled_rollout_whatever_swe_agent_recorded(tmp_path):
    out = str(tmp_path)
    tasks = [A.AgentTask(f"o__r-{n}", "p", "img", "o/r") for n in range(3)]
    for iid in ("o__r-0", "o__r-1"):
        d = os.path.join(out, iid)
        os.makedirs(d)
        with open(os.path.join(d, f"{iid}.traj"), "w", encoding="utf-8") as f:
            json.dump({"info": {"exit_status": "submitted (exit_error)",
                                "model_stats": {"api_calls": 7}},
                       "trajectory": [{"thought": "Exit due to unknown error: aiohttp",
                                       "response": "", "observation": ""}]}, f)
        with open(os.path.join(d, f"{iid}.pred"), "w", encoding="utf-8") as f:
            json.dump({"instance_id": iid, "model_patch": ""}, f)
    traces = LV.read_sweagent_run(tasks, out, time.time(), relay_failures=0,
                                  stalled={"o__r-0": "stalled: no file changed"},
                                  stopped="no file of the SWE-agent batch changed for 7000 s")
    by = {t["extra_payload"]["instance_id"]: t for t in traces}
    assert by["o__r-0"]["agent_infra_error"] is True
    assert by["o__r-0"]["agent_harness_error"] == LV.AGENT_STALLED
    assert by["o__r-0"]["agent_error"].startswith("stalled: no file changed")
    # the same ending without a stall stays the model's outcome, as upstream counts it
    assert by["o__r-1"]["agent_infra_error"] is False and "agent_harness_error" not in by["o__r-1"]
    assert by["o__r-2"]["agent_infra_error"] is True
    assert "gbench stopped the batch, no file of the SWE-agent batch" in by["o__r-2"]["agent_error"]
    assert "agent_harness_error" not in by["o__r-2"]


def test_a_stalled_rollout_is_counted_as_agent_stalled_in_the_tab(monkeypatch, tmp_path):
    rows = [{"instance_id": f"o__r-{n}", "problem_statement": "p", "base_commit": "c",
             "repo": "o/r"} for n in range(3)]

    def _rollouts(tasks, rows, model_name, relay_url, concurrency, out_dir, relay=None):
        with open(os.path.join(out_dir, "watchdog.json"), "w", encoding="utf-8") as f:
            json.dump({"stall_limit_s": 6300, "stalled_instances": ["o__r-1"],
                       "batch_stopped": None}, f)
        traces = []
        for i, t in enumerate(tasks):
            tr = {"sample_idx": i, "source_sample_idx": i, "category": t.category,
                  "agent_patch": "diff --git a/x b/x\n", "response_text": "diff --git a/x b/x\n",
                  "extra_payload": {"instance_id": t.instance_id, **t.meta},
                  "agent_exit_status": "submitted", "agent_infra_error": False,
                  "agent_error": None, "agent_steps": 3, "truncated_turns": None,
                  "is_correct": False, "status": "OK"}
            if t.instance_id == "o__r-1":
                tr.update(agent_infra_error=True, agent_harness_error=LV.AGENT_STALLED,
                          agent_error="stalled: no file of this rollout changed for 6400 s",
                          status="AGENT_INFRA_ERROR")
            traces.append(tr)
        return traces
    monkeypatch.setattr(LV, "run_sweagent", _rollouts)

    async def _scorer(traces):
        for t in traces:
            t["is_correct"] = t["extra_payload"]["instance_id"] == "o__r-0"
    res = LV._sweagent_tab("lite", rows, _scorer, {}, lambda r, m: [], "m",
                           "http://127.0.0.1:9/v1", 1, {})
    assert res["harness_errors"] == {LV.AGENT_STALLED: 1}
    assert res["harness_error_ids"] and "o__r-1" in json.dumps(res["harness_error_ids"])
    assert (res["correct_answers"], res["scored_instances"], res["accuracy"]) == (1, 2, 50.0)
    assert res["status"] == "completed_with_errors" and res["leaderboard_comparable"] is False
    assert res["agent_watchdog"]["stalled_instances"] == ["o__r-1"]
    assert res["agent"]["command_timeout_s"] == 5400
    assert res["agent"]["max_consecutive_command_timeouts"] == 3
    assert res["agent"]["total_command_time_limit_s"] == 18000
    assert "agent_watchdog" in LV._PER_SETTING_FIELDS


def test_a_hung_batch_ends_in_bounded_time_as_a_named_harness_error(monkeypatch, tmp_path):
    """The sweep's hang, end to end with docker stubbed: the rollout started and then nothing
    moved. The watchdog removes the stalled rollout's task container, stops the batch, and the
    instance comes back as an agent_stalled harness error."""
    out = str(tmp_path / "out")
    os.makedirs(out)
    monkeypatch.setattr(LV, "SWEAGENT_STALL_S", 0.4)
    monkeypatch.setattr(LV, "SWEAGENT_STALL_GRACE_S", 0.4)
    monkeypatch.setattr(LV, "SWEAGENT_WATCH_POLL_S", 0.05)
    monkeypatch.setattr(LV, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})
    iid = "mikedh__trimesh-2354"
    rows = [{"instance_id": iid, "problem_statement": "p", "base_commit": "c",
             "repo": "mikedh/trimesh"}]
    tasks = [A.AgentTask(iid, "p", LV.python_instance_image(iid), "mikedh/trimesh")]
    stopped = threading.Event()
    state = {"batch": None, "task": {"t1": _task_name(iid)}, "calls": []}

    def _run(cmd, **k):
        state["calls"].append(list(cmd))
        if "run-batch" in cmd:
            state["batch"] = cmd[cmd.index("--name") + 1]
            state["run_batch"] = list(cmd)
            d = os.path.join(out, iid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{iid}.info.log"), "w", encoding="utf-8") as f:
                f.write("STEP 6\ncat <<EOF > reproduce_issue.py\n")
            assert stopped.wait(20), "the watchdog never stopped the batch"
            return _proc(137)
        if cmd[:3] == ["docker", "ps", "-a"]:
            want = f"label=org.gbench.sweagent_batch={state['batch']}"
            assert cmd[cmd.index("--filter") + 1] == want
            return _proc(stdout="".join(f"{c}\t{n}\n" for c, n in state["task"].items()))
        if cmd[:3] == ["docker", "rm", "-f"]:
            for c in cmd[3:]:
                state["task"].pop(c, None)
                if c == state["batch"]:
                    stopped.set()
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    t0 = time.time()
    traces = LV.run_sweagent(tasks, rows, "m", "http://127.0.0.1:9/v1", 1, out)
    assert time.time() - t0 < 20
    assert "--init" in state["run_batch"]
    (tr,) = traces
    assert tr["agent_infra_error"] is True and tr["agent_harness_error"] == LV.AGENT_STALLED
    assert f"gbench removed its task container {_task_name(iid)}" in tr["agent_error"]
    rms = [c[3:] for c in state["calls"] if c[:3] == ["docker", "rm", "-f"]]
    assert ["t1"] in rms and [state["batch"]] in rms and not state["task"]
    rec = json.load(open(os.path.join(out, "watchdog.json"), encoding="utf-8"))
    assert rec["stalled_instances"] == [iid] and rec["batch_stopped"]


def test_a_finished_batch_removes_the_task_containers_swe_agent_left(monkeypatch, tmp_path):
    out = str(tmp_path / "out")
    os.makedirs(out)
    monkeypatch.setattr(LV, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})
    iid = "o__r-1"
    rows = [{"instance_id": iid, "problem_statement": "p", "base_commit": "c", "repo": "o/r"}]
    tasks = [A.AgentTask(iid, "p", LV.python_instance_image(iid), "o/r")]
    state = {"batch": None, "task": {"t9": _task_name(iid)}, "calls": []}

    def _run(cmd, **k):
        state["calls"].append(list(cmd))
        if "run-batch" in cmd:
            state["batch"] = cmd[cmd.index("--name") + 1]
            d = os.path.join(out, iid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{iid}.traj"), "w", encoding="utf-8") as f:
                json.dump({"info": {"exit_status": "submitted", "model_stats": {"api_calls": 2}},
                           "trajectory": []}, f)
            with open(os.path.join(d, f"{iid}.pred"), "w", encoding="utf-8") as f:
                json.dump({"instance_id": iid, "model_patch": "diff --git a/x b/x\n"}, f)
            return _proc()
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _proc(stdout="".join(f"{c}\t{n}\n" for c, n in state["task"].items()))
        if cmd[:3] == ["docker", "rm", "-f"]:
            for c in cmd[3:]:
                state["task"].pop(c, None)
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    (tr,) = LV.run_sweagent(tasks, rows, "m", "http://127.0.0.1:9/v1", 1, out)
    assert tr["agent_infra_error"] is False and tr["agent_patch"] == "diff --git a/x b/x\n"
    rms = [c[3:] for c in state["calls"] if c[:3] == ["docker", "rm", "-f"]]
    assert [state["batch"]] in rms and ["t9"] in rms and not state["task"]
    rec = json.load(open(os.path.join(out, "watchdog.json"), encoding="utf-8"))
    assert rec["stalled_instances"] == [] and rec["batch_stopped"] is None


# ============================================================================================
# swe_bench_live: cleanup when gbench is told to stop
# ============================================================================================
def test_a_sigterm_to_a_gbench_run_removes_the_batch_containers(monkeypatch, tmp_path):
    """The CLI turns SIGTERM into SystemExit while a run executes (cli._termination_handlers);
    run_sweagent's `finally` then removes the SWE-agent container and the batch's task
    containers before the exit goes on."""
    from gbench import cli
    out = str(tmp_path / "out")
    os.makedirs(out)
    monkeypatch.setattr(LV, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})
    iid = "o__r-1"
    rows = [{"instance_id": iid, "problem_statement": "p", "base_commit": "c", "repo": "o/r"}]
    tasks = [A.AgentTask(iid, "p", LV.python_instance_image(iid), "o/r")]
    state = {"batch": None, "task": {"t5": _task_name(iid)}, "calls": []}

    def _run(cmd, **k):
        state["calls"].append(list(cmd))
        if "run-batch" in cmd:
            state["batch"] = cmd[cmd.index("--name") + 1]
            # never send a SIGTERM this process would die of
            assert threading.current_thread() is threading.main_thread()
            assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN, None)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
            pytest.fail("the SIGTERM did not end the batch")
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _proc(stdout="".join(f"{c}\t{n}\n" for c, n in state["task"].items()))
        if cmd[:3] == ["docker", "rm", "-f"]:
            for c in cmd[3:]:
                state["task"].pop(c, None)
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as exc:
        with cli._termination_handlers():
            LV.run_sweagent(tasks, rows, "m", "http://127.0.0.1:9/v1", 1, out)
    assert exc.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == before
    rms = [c[3:] for c in state["calls"] if c[:3] == ["docker", "rm", "-f"]]
    assert [state["batch"]] in rms and ["t5"] in rms and not state["task"]


# ============================================================================================
# the docs say what the code does
# ============================================================================================
def _doc(name):
    with open(os.path.join(REPO, "docs", "evals", f"{name}.md"), encoding="utf-8") as f:
        return " ".join(f.read().split())


def test_the_docs_describe_the_limits_the_watchdog_and_the_cleanup():
    doc = _doc("swe_bench_live")
    for needle in ("agent_stalled", "agent_watchdog", "watchdog.json",
                   "org.gbench.sweagent_batch", "90 minutes", "6300 s", "6900 s",
                   "exit_command_timeout", "exit_total_execution_time", "--init",
                   "bashlex 0.18", "SIGTERM", "SIGKILL"):
        assert needle in doc, needle
    lancer = _doc("swe_lancer")
    assert "swe_lancer/issues_manifest.json" in lancer and "read again from the image" in lancer
