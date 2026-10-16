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

"""swe_bench_live: a heredoc is no longer a hang, the watchdog names a stall, a signal stops it.

Until 2026-10-08 the task containers' runtime (swe-rex 1.4.0 with bashlex 0.18) split each command
with bashlex and joined the pieces with " ; ", so a heredoc such as `cat <<EOF > file` lost its
body and the shell waited until SWE-agent-for-eval's 90-minute command limit; gbench's watchdog
ended such an instance after 5 minutes as a harness error (agent_heredoc_dropped). The runtime is
now fixed (gbench/docker/sweagent_for_eval_swerex_heredoc.patch, value-tested in
test_swe_bench_live_task_runtime.py), so that rule is retired: a long command that holds a heredoc
is an ordinary long command, and only the stall rule ends an instance. The command a stalled
instance was running is read from its trace log, whose logger is named per instance in a batch.

A SIGTERM/SIGHUP to the run is recorded (utils.process_groups.request_stop): a batch running in
a worker thread (--suite-timeout) is stopped, no further tab starts, and the suite the signal
cut short gets a failed record naming the signal. Docker, the endpoint and the network are
stubbed; nothing is written outside tmp.
"""

import json
import os
import threading
import time
import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swebench_common as C
from gbench.utils import process_groups

UUID = "2f07fb32-3e8d-49dc-9395-c6328231d7db"


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TRAJ_DIR", str(tmp_path / "traj_live"))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)
    process_groups.clear_stop()
    yield
    process_groups.clear_stop()


def test_the_heredoc_rule_is_retired():
    """The runtime runs heredocs (SWEREX_RUNTIME_SHA256), so nothing of the old rule is left to
    end an instance early."""
    for gone in ("SWEAGENT_HEREDOC_GRACE_S", "AGENT_HEREDOC_DROPPED", "sweagent_heredoc_dropped",
                 "_heredoc_rule_drops", "_swerex_splitter", "SWEREX_RUNTIME_DEFECT"):
        assert not hasattr(LV, gone), gone
    assert "heredoc_grace_s" not in LV.SweagentWatch.__init__.__code__.co_varnames


def _trace_log(path, records):
    """A SWE-agent trace log: (level, logger, message) records."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (level, name, msg) in enumerate(records):
            f.write(f"2026-09-29 00:12:{i:02d},750 - {level} - {name} - {msg}\n")


HEREDOC = ("cat <<EOF > reproduce_issue.py\nimport trimesh\n\nprint('x')\nEOF\n\n"
           "python3 reproduce_issue.py")


def test_the_inflight_command_is_the_last_input_without_output(tmp_path):
    p = str(tmp_path / "i" / "i.trace.log")
    _trace_log(p, [("TRACE", "swea-env", "Input:\nls"), ("TRACE", "swea-env", "Output:\na b"),
                   ("INFO", "swea-agent", "STEP 6"), ("TRACE", "swea-env", "Input:\n" + HEREDOC)])
    assert LV.sweagent_inflight_command(p) == HEREDOC
    _trace_log(p, [("TRACE", "swea-env", "Input:\n" + HEREDOC),
                   ("TRACE", "swea-env", "Output:\n")])
    assert LV.sweagent_inflight_command(p) is None           # it returned
    assert LV.sweagent_inflight_command(str(tmp_path / "missing.log")) is None


def test_a_batch_runs_per_instance_logger_names_are_read(tmp_path):
    """In a batch SWE-agent names the logger per instance ("swea-env-<instance id>"), as on the
    2026-10-08 smoke run, where a pattern for the bare name matched no record, so the watchdog
    never saw the command an instance was running."""
    env = "swea-env-flexget__flexget-4306"
    p = str(tmp_path / "i" / "i.trace.log")
    _trace_log(p, [("TRACE", env, "Input:\nls"), ("TRACE", env, "Output:\na b"),
                   ("INFO", "swea-agent-flexget__flexget-4306", "STEP 6"),
                   ("TRACE", env, "Input:\n" + HEREDOC)])
    assert LV.sweagent_inflight_command(p) == HEREDOC
    _trace_log(p, [("TRACE", env, "Input:\n" + HEREDOC), ("TRACE", env, "Output:\n")])
    assert LV.sweagent_inflight_command(p) is None           # it returned


# ============================================================================================
# the watchdog ends the hang early and names it
# ============================================================================================
class FakeDocker:
    def __init__(self, batch, names):
        self.batch = batch
        self.containers = {f"c{i}": n for i, n in enumerate(names)}
        self.calls = []

    def __call__(self, cmd, **k):
        self.calls.append(list(cmd))
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _proc(stdout="".join(f"{c}\t{n}\n" for c, n in self.containers.items()))
        if cmd[:3] == ["docker", "rm", "-f"]:
            for c in cmd[3:]:
                self.containers.pop(c, None)
        return _proc()

    def removed(self):
        return [c[3:] for c in self.calls if c[:3] == ["docker", "rm", "-f"]]


def _task_name(iid):
    return LV.python_instance_image(iid).replace("/", "") + "-" + UUID


def _age(root, when):
    for dirpath, dirs, files in os.walk(root):
        for name in files + dirs:
            os.utime(os.path.join(dirpath, name), (when, when))
    os.utime(root, (when, when))


def test_a_long_command_with_a_heredoc_is_an_ordinary_long_command(monkeypatch, tmp_path):
    """With the runtime fixed, a command that holds a heredoc and runs long (silent for far more
    than the retired rule's 300 s) is left alone until the stall limit; a stall then names the
    command it was running, read with the batch's per-instance logger name."""
    out = str(tmp_path)
    now = 1_000_000.0
    iid, other = "mikedh__trimesh-2354", "o__r-2"
    docker = FakeDocker("b1", [_task_name(iid), _task_name(other)])
    monkeypatch.setattr(LV.subprocess, "run", docker)
    watch = LV.SweagentWatch(out, {i: LV.python_instance_image(i) for i in (iid, other)}, "b1",
                             clock=lambda: now)
    long_heredoc = "cat <<EOF > t.py\nimport time; time.sleep(3000)\nEOF\npython3 t.py"
    _trace_log(os.path.join(out, iid, f"{iid}.trace.log"),
               [("TRACE", f"swea-env-{iid}", "Input:\n" + long_heredoc)])
    _trace_log(os.path.join(out, other, f"{other}.trace.log"),
               [("TRACE", f"swea-env-{other}", "Input:\n" + HEREDOC)])
    _age(os.path.join(out, iid), now - 3000)
    _age(os.path.join(out, other), now - LV.SWEAGENT_STALL_S + 60)
    watch.check(now)
    assert watch.stalled == {} and docker.removed() == []
    _age(os.path.join(out, iid), now - LV.SWEAGENT_STALL_S - 1)
    watch.check(now)
    assert sorted(watch.stalled) == [iid] and watch.kinds[iid] == LV.AGENT_STALLED
    why = watch.stalled[iid]
    assert why.startswith("stalled: no file of this rollout changed for")
    assert "first line: 'cat <<EOF > t.py'" in why
    assert f"gbench removed its task container {_task_name(iid)}" in why
    assert docker.removed() == [["c0"]]
    rec = watch.record()
    assert rec["stalled_instances"] == [iid]
    assert "heredoc_dropped_instances" not in rec and "heredoc_grace_s" not in rec


def test_an_instance_that_already_ended_is_never_flagged(monkeypatch, tmp_path):
    """SWE-agent lists an instance whose run raised as 'Uncaught <Exception>' in
    run_batch_exit_statuses.yaml, with no .pred: it ended, it is not stalled."""
    out = str(tmp_path)
    now = 3_000_000.0
    iid = "o__r-1"
    docker = FakeDocker("b3", [_task_name(iid)])
    monkeypatch.setattr(LV.subprocess, "run", docker)
    watch = LV.SweagentWatch(out, {iid: LV.python_instance_image(iid)}, "b3", clock=lambda: now)
    _trace_log(os.path.join(out, iid, f"{iid}.trace.log"),
               [("TRACE", "swea-env", "Input:\n" + HEREDOC)])
    _age(os.path.join(out, iid), now - LV.SWEAGENT_STALL_S - 10)
    with open(os.path.join(out, "run_batch_exit_statuses.yaml"), "w", encoding="utf-8") as f:
        f.write(f"instances_by_exit_status:\n  Uncaught RuntimeError:\n  - {iid}\n")
    watch.check(now)
    assert watch.stalled == {} and docker.removed() == []


def test_halt_waits_for_a_check_in_progress(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Slow(LV.SweagentWatch):
        def check(self, now=None):
            entered.set()
            release.wait(5)
            time.sleep(0.2)
            self.stalled["x"] = "set by the check"
    watch = Slow(str(tmp_path), {}, "b4", poll_s=0.01).start()
    assert entered.wait(5)
    release.set()
    watch.halt()
    assert watch.stalled == {"x": "set by the check"}


def _rows_tasks(iid):
    rows = [{"instance_id": iid, "problem_statement": "p", "base_commit": "c", "repo": "o/r"}]
    return rows, [A.AgentTask(iid, "p", LV.python_instance_image(iid), "o/r")]


def test_a_stall_ends_the_rollout_as_a_named_harness_error(monkeypatch, tmp_path):
    """End to end with docker stubbed: the rollout's command never returns and nothing moves;
    the watchdog removes the task container at the stall limit, and the instance is an
    agent_stalled harness error naming the command."""
    out = str(tmp_path / "out")
    os.makedirs(out)
    monkeypatch.setattr(LV, "SWEAGENT_STALL_S", 0.3)
    monkeypatch.setattr(LV, "SWEAGENT_WATCH_POLL_S", 0.05)
    monkeypatch.setattr(LV, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})
    iid = "mikedh__trimesh-2354"
    rows, tasks = _rows_tasks(iid)
    removed_task = threading.Event()
    state = {"task": {"t1": _task_name(iid)}}

    def _run(cmd, **k):
        if "run-batch" in cmd:
            _trace_log(os.path.join(out, iid, f"{iid}.trace.log"),
                       [("TRACE", f"swea-env-{iid}", "Input:\nsleep 100000")])
            assert removed_task.wait(20), "the watchdog never ended the stall"
            return _proc(0)
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _proc(stdout="".join(f"{c}\t{n}\n" for c, n in state["task"].items()))
        if cmd[:3] == ["docker", "rm", "-f"]:
            for c in cmd[3:]:
                if state["task"].pop(c, None):
                    removed_task.set()
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    t0 = time.time()
    (tr,) = LV.run_sweagent(tasks, rows, "m", "http://127.0.0.1:9/v1", 1, out)
    assert time.time() - t0 < 20
    assert tr["agent_infra_error"] is True
    assert tr["agent_harness_error"] == LV.AGENT_STALLED
    assert tr["agent_error"].startswith("stalled:") and "'sleep 100000'" in tr["agent_error"]
    rec = json.load(open(os.path.join(out, "watchdog.json"), encoding="utf-8"))
    assert rec["stalled_instances"] == [iid] and rec["batch_stopped"] is None


def test_the_tab_records_the_task_runtime(monkeypatch, tmp_path):
    rows = [{"instance_id": "o__r-0", "problem_statement": "p", "base_commit": "c",
             "repo": "o/r"}]

    def _rollouts(tasks, rows, model_name, relay_url, concurrency, out_dir, relay=None):
        return [{"sample_idx": 0, "source_sample_idx": 0, "category": tasks[0].category,
                 "agent_patch": "diff --git a/x b/x\n", "response_text": "diff --git a/x b/x\n",
                 "extra_payload": {"instance_id": tasks[0].instance_id, **tasks[0].meta},
                 "agent_exit_status": "submitted", "agent_infra_error": False,
                 "agent_error": None, "agent_steps": 3, "truncated_turns": None,
                 "is_correct": False, "status": "OK"}]
    monkeypatch.setattr(LV, "run_sweagent", _rollouts)

    async def _scorer(traces):
        for t in traces:
            t["is_correct"] = True
    res = LV._sweagent_tab("lite", rows, _scorer, {}, lambda r, m: [], "m",
                           "http://127.0.0.1:9/v1", 1, {})
    assert res["agent"]["command_timeout_s"] == LV.SWEAGENT_COMMAND_TIMEOUT_S
    assert res["agent"]["task_runtime"] == {
        "swe_rex": "1.4.0", "patch": "gbench/docker/sweagent_for_eval_swerex_heredoc.patch",
        "session_code_sha256": LV.SWEREX_RUNTIME_SHA256, "pins": list(LV.SWEREX_RUNTIME_PINS)}
    assert "heredoc" not in (res.get("leaderboard_comparable_reason") or "")


# ============================================================================================
# a signal stops the suite, not only its batch
# ============================================================================================
def test_a_signal_stops_a_batch_running_in_a_worker_thread(monkeypatch, tmp_path):
    out = str(tmp_path / "out")
    os.makedirs(out)
    monkeypatch.setattr(LV, "SWEAGENT_WATCH_POLL_S", 0.05)
    monkeypatch.setattr(LV, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})
    iid = "o__r-1"
    rows, tasks = _rows_tasks(iid)
    batch_gone = threading.Event()
    state = {"batch": None, "calls": []}

    def _run(cmd, **k):
        state["calls"].append(list(cmd))
        if "run-batch" in cmd:
            state["batch"] = cmd[cmd.index("--name") + 1]
            # what evals._run_with_timeout hands this worker thread when the main thread gets
            # the CLI's SystemExit
            process_groups.request_stop(15, thread=threading.get_ident())
            assert batch_gone.wait(20), "the watchdog never stopped the batch"
            return _proc(137)
        if cmd[:3] == ["docker", "rm", "-f"] and state["batch"] in cmd[3:]:
            batch_gone.set()
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _proc(stdout="")
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    box = {}

    def _worker():
        try:
            LV.run_sweagent(tasks, rows, "m", "http://127.0.0.1:9/v1", 1, out)
        except SystemExit as e:
            box["exit"] = e.code
    t = threading.Thread(target=_worker)
    t.start()
    t.join(30)
    assert not t.is_alive() and box.get("exit") == 128 + 15
    rec = json.load(open(os.path.join(out, "watchdog.json"), encoding="utf-8"))
    assert rec["batch_stopped"] == "the run was stopped by signal 15"


def test_no_tab_starts_after_a_signal(monkeypatch):
    started = []
    monkeypatch.setattr(LV, "missing_prerequisites", lambda settings, mode: {})
    monkeypatch.setattr(LV, "_run_setting",
                        lambda setting, *a, **k: started.append(setting) or {})
    process_groups.request_stop(1)
    with pytest.raises(SystemExit) as exc:
        LV.run_swe_bench_live("m", "http://127.0.0.1:9/v1", 1, eval_categories="java")
    assert exc.value.code == 128 + 1 and started == []


def test_the_suite_a_signal_cut_short_gets_a_failed_record(tmp_path):
    from gbench.core.config import BenchmarkConfig
    from gbench.core.models import ModelConfig, ModelFormat
    from gbench.runners import evals as E
    from gbench.runners.eval_suites import SUITES

    config = BenchmarkConfig()
    config.evals = ["gsm8k", "mmlu"]
    config.batch_sizes = [1]
    config.skip_existing = False
    config.eval_max_output_tokens = 256
    config.results_dir = tmp_path / "results"
    config.remote_endpoint = "http://127.0.0.1:9"
    config.initialize()
    model = ModelConfig(short_name="m", hf_model_id="org/m", total_params_b=1.0, is_moe=False)

    def _stopped(**kw):
        process_groups.request_stop(15)
        raise SystemExit(128 + 15)
    runner = E.EvalsBenchmarkRunner(config)
    with mock.patch.dict(SUITES, {"gsm8k": _stopped, "mmlu": lambda **k: {}}), \
            mock.patch("gbench.runners.eval_suites.discover_and_register_plugins",
                       return_value={}), \
            mock.patch("gbench.utils.endpoint.list_served_models",
                       lambda base_url, timeout=5.0, headers=None: [{"id": "org/m"}]):
        with pytest.raises(SystemExit):
            runner.run(model, ModelFormat.HF)
    (rec,) = runner.partial_results
    assert rec["eval_name"] == "gsm8k" and rec["failed"] is True and rec["accuracy"] is None
    assert rec["error"].startswith("terminated by signal 15 (SIGTERM) after ")
    saved = list((config.log_manager.results_dir).glob("eval_gsm8k_*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["error"].startswith("terminated by signal 15")


def test_a_signal_is_handed_to_the_suites_worker_thread_and_the_run_record_ends(monkeypatch):
    """Under --suite-timeout the suite runs in a worker thread: the CLI's SIGTERM handler raises
    SystemExit in the main thread only. _run_with_timeout hands the signal to the worker, which
    sees it (stop_requested) after the main thread's exit; the run-wide record ends with the
    CLI's context, so nothing leaks into a later run in the same process."""
    import signal
    from gbench import cli
    from gbench.runners import evals as E
    started, seen = threading.Event(), {}
    done = threading.Event()

    def _suite():
        started.set()
        for _ in range(400):
            stop = process_groups.stop_requested()
            if stop:
                seen["stop"] = stop[0]
                break
            time.sleep(0.05)
        done.set()
        return {}

    def _kill():
        assert started.wait(10)
        os.kill(os.getpid(), signal.SIGTERM)
    killer = threading.Thread(target=_kill)
    with pytest.raises(SystemExit) as exc:
        with cli._termination_handlers():
            killer.start()
            E._run_with_timeout("swe_bench_live", _suite, {}, 3600, "m")
    killer.join(10)
    assert exc.value.code == 128 + signal.SIGTERM
    assert process_groups.stop_requested() is None          # the run-wide record ended
    assert done.wait(30) and seen.get("stop") == signal.SIGTERM
