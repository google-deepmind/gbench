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

"""wildclawbench: grading isolation, grade() cap hits, and DNS / egress of the task network.

* The processes the agent left in a task container are stopped when its grading starts (the ones
  that ran before the OpenClaw gateway started are kept), so nothing the agent started can read
  the grader key or send judge requests with it while grade() runs; a judge failure counts only
  for a request that arrived during such a grade() exec.
* Every grade() cap hit of such an exec is a harness error, with or without a judge call.
* The metadata guard's probe also resolves the host names the tasks need on the task network; the
  orchestrator's network monitor watches DNS and web egress for the whole run, and a task during
  which the network failed is a harness error.

Docker is stubbed everywhere: no container starts, the only processes the embedded scripts touch
are read from /proc (the stop script never runs here).
"""

import json
import os
import subprocess
import sys
import time
import types

import pytest

from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "gbench", "docker"))
import wildclawbench_cascade_judge as J  # noqa: E402
import wildclawbench_run as RUN  # noqa: E402


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


def _jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture(autouse=True)
def _fresh_launcher_state():
    RUN._BASELINES.clear()
    RUN._ISOLATION.clear()
    yield
    RUN._BASELINES.clear()
    RUN._ISOLATION.clear()


# --------------------------------------------------------------------------- #
# The runner: the guard resolves the task hosts; the monitor's host lists are forwarded
# --------------------------------------------------------------------------- #
class _Guard:
    def __init__(self):
        self.calls = []

    def __call__(self, eval_name, image, network, docs_url=None, *, resolve=()):
        self.calls.append({"eval": eval_name, "image": image, "network": network,
                           "docs": docs_url, "resolve": list(resolve)})
        return CG.UNREACHABLE


def _runner(monkeypatch, tmp_path, summary=None, env=None):
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    wd = tmp_path / "wd"
    wd.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(wd))
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_JUDGE_PORT", "GBENCH_WILDCLAWBENCH_TASK_HOST", "GBENCH_SHARD",
              "GBENCH_WILDCLAWBENCH_PARALLEL", "GEMINI_OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    cap = {"runs": []}
    full = {"n_tasks": 60, "n_scored": 60, "global_mean": 0.5, "missing_inputs": {}}
    full.update(summary or {})

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"]:
            cap["runs"].append(list(cmd))
            with open(os.path.join(str(wd), "wildclawbench_summary.json"), "w") as f:
                json.dump(full, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    return W.run_wildclawbench("m", "http://127.0.0.1:9123/v1", concurrency=4), cap


def test_the_guard_resolves_the_task_hosts_on_the_task_network(monkeypatch, tmp_path):
    guard = _Guard()
    monkeypatch.setattr(CG, "require_metadata_blocked", guard)
    result, cap = _runner(monkeypatch, tmp_path)
    assert guard.calls == [{"eval": "wildclawbench", "image": W._task_image(),
                            "network": "bridge", "docs": W.DOCS_URL,
                            "resolve": list(W.TASK_HOSTS)}]
    [cmd] = cap["runs"]
    assert f"WILDCLAW_DNS_HOSTS={','.join(W.TASK_HOSTS)}" in cmd
    assert f"WILDCLAW_EGRESS_HOSTS={','.join(W.EGRESS_HOSTS)}" in cmd
    assert result["task_hosts"] == list(W.TASK_HOSTS)
    # the launcher's own defaults are the same lists
    assert tuple(RUN.DNS_HOSTS_DEFAULT) == W.TASK_HOSTS
    assert tuple(RUN.EGRESS_HOSTS_DEFAULT) == W.EGRESS_HOSTS
    assert set(W.EGRESS_HOSTS) <= set(W.TASK_HOSTS)
    assert RUN.FIREWALL_COMMAND == CG.FIREWALL_COMMAND


def test_a_task_host_that_does_not_resolve_starts_no_orchestrator(monkeypatch, tmp_path):
    # the real guard over a fake docker: the metadata server is blocked, but DNS fails
    seen = []

    def fake(cmd, *a, **k):
        seen.append(list(cmd))
        if CG.PROBE_LABEL in cmd:
            lines = ["GBENCH_METADATA_PROBE=unreachable timeout ipv6:101"]
            lines += [f"GBENCH_DNS_PROBE=fail {h} gaierror:-3" for h in W.TASK_HOSTS]
            return subprocess.CompletedProcess(cmd, 0, "\n".join(lines) + "\n", "")
        raise AssertionError(f"no other docker call expected: {cmd}")
    monkeypatch.setattr(subprocess, "run", fake)
    harness = tmp_path / "h"
    harness.mkdir()
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_NETWORK", raising=False)
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_HOST", raising=False)
    with pytest.raises(RuntimeError) as ei:
        W.run_wildclawbench("m", "http://127.0.0.1:9123/v1")
    msg = str(ei.value)
    assert "cannot resolve host names" in msg and "pypi.org: gaierror:-3" in msg
    assert CG.FIREWALL_COMMAND in msg and CG.PLAIN_RULE_DELETE_COMMAND in msg
    [probe] = seen                                  # one probe container, no orchestrator
    script = probe[-1]
    assert all(repr(h)[1:-1] in script for h in W.TASK_HOSTS)
    assert "--network" in probe and probe[probe.index("--network") + 1] == "bridge"


def test_the_result_names_the_new_counters(monkeypatch, tmp_path):
    monkeypatch.setattr(CG, "require_metadata_blocked", _Guard())
    summary = {
        "n_harness_errors": 3, "harness_error_kinds": {"grading_infra": 2, "network": 1},
        "grading_timeouts": {"with_judge_call": 1, "without_judge_call": 1, "unrecorded": 0,
                             "not_isolated": 1},
        "grading_isolation": {"tasks_isolated": 57, "tasks_not_isolated": 1,
                              "processes_stopped": 140},
        "grader_events_not_counted": 2,
        "network_monitor": {"interval_s": 15, "rounds": 900, "failed_rounds": 3,
                            "down_rounds": 1, "host_failures": {"dns": {}, "tcp": {}},
                            "gaps": [[10.0, 100.0]]},
        "n_tasks_network_unmonitored": 2,
        "name_resolution_errors_unconfirmed": {"01_Productivity_Flow_task_2": {"chat.jsonl": 1}},
    }
    result, _cap = _runner(monkeypatch, tmp_path, summary=summary)
    assert result["status"] == "completed_with_errors"
    assert result["grading_timeouts"]["without_judge_call"] == 1
    assert result["grading_isolation"]["tasks_not_isolated"] == 1
    assert result["grader_events_not_counted"] == 2
    assert result["network_monitor"]["down_rounds"] == 1
    assert result["name_resolution_errors_unconfirmed"] == {
        "01_Productivity_Flow_task_2": {"chat.jsonl": 1}}
    reason = result["leaderboard_comparable_reason"]
    assert ("2 task(s) whose grade() hit the 120 s cap are harness errors, left out of the "
            "headline (1 with a judge call in flight, 1 without one, 0 without an exec record, "
            "0 after a stop of the agent's processes that failed for a reason the agent cannot "
            "cause)") in reason
    assert ("1 task(s) whose leftover processes could not all be stopped before grading (cause "
            "not recorded): an unverified grading environment; after a stop the agent can have "
            "failed, graded as upstream grades them with their judge failures and grade() cap "
            "hits (1) not counted as harness errors") in reason
    assert "2 judge failure event(s) of a request with the grader key" in reason
    assert "has 1 gap(s) longer than 60 s; 2 task(s) ran partly unmonitored" in reason
    assert ("1 task(s) whose logs show a failed name lookup while the network monitor saw no "
            "resolver failure (at least half of the names not resolving) in their run window "
            "(scored as run)") in reason
    assert "1 task network lost DNS or web egress during the task" in reason


def test_the_labels_follow_the_launcher_and_the_image():
    assert [k for k, _ in W.HARNESS_ERROR_LABELS] == list(RUN.HARNESS_ERROR_KINDS)
    assert "network" in RUN.HARNESS_ERROR_KINDS
    assert W.HARNESS_ERRORS_VERSION == "9"
    assert f'LABEL {W.HARNESS_ERRORS_LABEL}="{W.HARNESS_ERRORS_VERSION}"' in \
        _read("gbench/docker/wildclawbench.Dockerfile")


# --------------------------------------------------------------------------- #
# The launcher: the process list before the gateway, the stop when grading starts
# --------------------------------------------------------------------------- #
class _Docker:
    """A fake `docker` for the launcher's patched subprocess.run."""

    def __init__(self, snapshot="[[1, 100], [13, 150]]", stop=None, running="true"):
        self.cmds = []
        self.snapshot = snapshot
        self.stop = stop if stop is not None else (0, '{"stopped": 4, "left": []}', "")
        self.running = running

    def __call__(self, cmd, *a, **k):
        self.cmds.append(list(cmd))
        if cmd[:2] == ["docker", "exec"] and RUN._PROCESS_SNAPSHOT_SCRIPT in cmd:
            if isinstance(self.snapshot, Exception):
                raise self.snapshot
            return subprocess.CompletedProcess(cmd, 0, self.snapshot + "\n", "")
        if cmd[:2] == ["docker", "exec"] and RUN._STOP_LEFTOVER_SCRIPT in cmd:
            rc, out, err = self.stop
            return subprocess.CompletedProcess(cmd, rc, out, err)
        if cmd[:4] == ["docker", "inspect", "-f", "{{.State.Running}}"]:
            if self.running is None:
                return subprocess.CompletedProcess(cmd, 1, "", "Error: No such object: x")
            return subprocess.CompletedProcess(cmd, 0, self.running + "\n", "")
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, "172.17.0.5 \n", "")
        if cmd[:2] == ["docker", "exec"] and RUN.GRADE_RUNNER in cmd:
            if k.get("timeout") == "cap":
                raise subprocess.TimeoutExpired(cmd, 120)
            return subprocess.CompletedProcess(
                cmd, 0, '{"overall_score": 1.0}\n',
                "urllib3 NameResolutionError: Temporary failure in name resolution\n")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    def kinds(self):
        out = []
        for c in self.cmds:
            if RUN._PROCESS_SNAPSHOT_SCRIPT in c:
                out.append("snapshot")
            elif RUN._STOP_LEFTOVER_SCRIPT in c:
                out.append("stop")
            elif c[:2] == ["docker", "cp"]:
                out.append("cp " + c[-1].split(":", 1)[1])
            elif c[:2] == ["docker", "exec"] and RUN.GRADE_RUNNER in c:
                out.append("grade")
            elif c[:3] == ["docker", "run", "-d"]:
                out.append("run")
        return out


def _patched(monkeypatch, tmp_path, docker):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://172.17.0.1:18795")
    monkeypatch.setenv(RUN.GRADER_KEY_ENV, "gbench-grader-xyz")
    monkeypatch.delenv("WILDCLAW_TASK_NETWORK", raising=False)
    background = []

    def run_background(task_id, bash_cmd, log_path):
        background.append((task_id, bash_cmd, str(log_path)))
        return types.SimpleNamespace(pid=1)
    du = types.SimpleNamespace(subprocess=types.SimpleNamespace(run=docker),
                               run_background=run_background, TMP_WORKSPACE="/tmp_workspace")
    RUN._patch_task_containers(du)
    return du, background


def _upstream_task(du, out_dir, container="t1", grade_timeout=None):
    """What run_batch.run_single_task does for one OpenClaw task, as docker calls."""
    du.subprocess.run(["docker", "run", "-d", "--name", container, "img"])
    du.run_background(container, bash_cmd="openclaw gateway --port 18789",
                      log_path=os.path.join(out_dir, "gateway.log"))
    du.run_background(container, bash_cmd="openclaw agent --message x",
                      log_path=os.path.join(out_dir, "agent.log"))
    du.subprocess.run(["docker", "cp", "/h/ws/gt", f"{container}:/tmp_workspace/gt"])
    du.subprocess.run(["docker", "cp", "/x/transcript_loader.py",
                       f"{container}:/tmp/_transcript_loader.py"])
    du.subprocess.run(["docker", "cp", "/tmp/r.py", f"{container}:/tmp/_grade_runner.py"])
    return du.subprocess.run(["docker", "exec", "-e", "OPENROUTER_API_KEY=gbench-cascade",
                              container, "python3", RUN.GRADE_RUNNER], timeout=grade_timeout)


def test_leftover_processes_are_stopped_before_the_ground_truth_is_copied_in(monkeypatch,
                                                                            tmp_path):
    docker = _Docker()
    du, background = _patched(monkeypatch, tmp_path, docker)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "gateway.log").write_text("getaddrinfo EAI_AGAIN arxiv.org\nok\n")
    (out_dir / "agent.log").write_text("done\n")
    (out_dir / "chat.jsonl").write_text('{"x": "curl: (6) Could not resolve host: a.b"}\n')

    def run_single_task(task, *a, **k):
        _upstream_task(du, str(out_dir))
        return {"task_id": "t1", "error": None, "scores": {"overall_score": 1.0}}
    rb = types.SimpleNamespace(run_single_task=run_single_task)
    RUN._track_task_windows(rb)
    rb.run_single_task({"task_id": "03_Social_Interaction_task_1"})
    # the process list once, before the gateway; the stop before the first grading copy
    kinds = [k for k in docker.kinds() if k != "run"]
    assert kinds == ["snapshot", "stop", "cp /tmp_workspace/gt", "cp /tmp/_transcript_loader.py",
                     "cp /tmp/_grade_runner.py", "grade"]
    assert [b[1] for b in background] == ["openclaw gateway --port 18789",
                                          "openclaw agent --message x"]
    stop = next(c for c in docker.cmds if RUN._STOP_LEFTOVER_SCRIPT in c)
    assert stop[:3] == ["docker", "exec", "t1"] and json.loads(stop[-1]) == [[1, 100], [13, 150]]
    grade = next(c for c in docker.cmds if RUN.GRADE_RUNNER in c and c[:2] == ["docker", "exec"])
    assert "OPENROUTER_API_KEY=gbench-grader-xyz" in grade
    [rec] = [json.loads(x) for x in (tmp_path / RUN.TASK_WINDOWS).read_text().splitlines()]
    assert rec["process_baseline"] == [[1, 100], [13, 150]]
    assert rec["isolation"]["ok"] is True and rec["isolation"]["stopped"] == 4
    [g] = rec["grading"]
    assert g["isolated"] is True and g["timed_out"] is False
    assert g["name_resolution_errors"] == 1
    assert rec["name_resolution_errors"] == {"gateway.log": 1, "chat.jsonl": 1, "grade": 1}
    assert "_logs" not in rec


def test_a_process_list_that_cannot_be_taken_fails_the_start(monkeypatch, tmp_path):
    docker = _Docker(snapshot="garbage")
    du, background = _patched(monkeypatch, tmp_path, docker)
    with pytest.raises(RuntimeError, match="process list of t1 is unreadable"):
        du.run_background("t1", bash_cmd="openclaw gateway", log_path=str(tmp_path / "g.log"))
    assert background == []                         # the gateway never started
    docker = _Docker(snapshot=subprocess.TimeoutExpired(["docker"], 120))
    du, background = _patched(monkeypatch, tmp_path, docker)
    with pytest.raises(RuntimeError, match="could not list the processes of t2"):
        du.run_background("t2", bash_cmd="openclaw gateway", log_path=str(tmp_path / "g.log"))


@pytest.mark.parametrize("stop,running,down", [
    ((3, '{"stopped": 7, "left": [44]}', ""), "true", False),       # a process survived
    ((127, "", "python3: not found"), "true", False),               # the agent removed python3
    ((1, "", "Error response from daemon: No such container"), None, True),   # container gone
])
def test_a_stop_that_fails_leaves_the_grade_unisolated(monkeypatch, tmp_path, stop, running,
                                                       down):
    docker = _Docker(stop=stop, running=running)
    du, _bg = _patched(monkeypatch, tmp_path, docker)

    def run_single_task(task, *a, **k):
        _upstream_task(du, str(tmp_path))
        return {"task_id": "t1", "error": None}
    rb = types.SimpleNamespace(run_single_task=run_single_task)
    RUN._track_task_windows(rb)
    rb.run_single_task({"task_id": "03_Social_Interaction_task_1"})
    [rec] = [json.loads(x) for x in (tmp_path / RUN.TASK_WINDOWS).read_text().splitlines()]
    assert rec["isolation"]["ok"] is False and rec["isolation"]["error"]
    assert bool(rec["isolation"].get("container_down")) is down
    assert [g["isolated"] for g in rec["grading"]] == [False]
    # the grade still runs, as upstream runs it
    assert any(RUN.GRADE_RUNNER in c and c[:2] == ["docker", "exec"] for c in docker.cmds)


def test_grading_without_a_process_list_stops_nothing(monkeypatch, tmp_path):
    docker = _Docker()
    du, _bg = _patched(monkeypatch, tmp_path, docker)
    res = RUN.stop_leftover_processes("t9", None, docker)
    assert res["ok"] is False and "no process list" in res["error"]
    assert docker.cmds == []


def test_a_failed_grade_exec_on_a_gone_container_is_recorded(monkeypatch, tmp_path):
    class Gone(_Docker):
        def __call__(self, cmd, *a, **k):
            if cmd[:2] == ["docker", "exec"] and RUN.GRADE_RUNNER in cmd:
                self.cmds.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 1, "", "Error response from daemon")
            return super().__call__(cmd, *a, **k)
    docker = Gone(running=None)
    du, _bg = _patched(monkeypatch, tmp_path, docker)

    def run_single_task(task, *a, **k):
        _upstream_task(du, str(tmp_path))
        return {"task_id": "t1", "error": None}
    rb = types.SimpleNamespace(run_single_task=run_single_task)
    RUN._track_task_windows(rb)
    rb.run_single_task({"task_id": "03_Social_Interaction_task_1"})
    [rec] = [json.loads(x) for x in (tmp_path / RUN.TASK_WINDOWS).read_text().splitlines()]
    assert rec["grading"][0]["container_down"] is True


def test_the_process_scan_reads_proc_only():
    ns = {}
    exec(RUN._PROC_SCAN, ns)                         # defines scan(); reads /proc, sends nothing
    procs = ns["scan"]({os.getpid()})
    assert os.getpid() not in procs and all(isinstance(v, int) for v in procs.values())
    # the stop freezes before it kills, keeps PID 1, itself and the listed processes
    stop = RUN._STOP_LEFTOVER_SCRIPT
    assert "(signal.SIGSTOP, signal.SIGKILL)" in stop and "skip = {1, os.getpid()}" in stop
    assert "not in keep" in stop and 'fields[0] in ("Z", "X")' in RUN._PROC_SCAN
    for script in (RUN._PROCESS_SNAPSHOT_SCRIPT, RUN._STOP_LEFTOVER_SCRIPT,
                   RUN._NETWORK_MONITOR_SCRIPT):
        compile(script, "<script>", "exec")


# --------------------------------------------------------------------------- #
# The summary: grader-key events, cap hits, the network
# --------------------------------------------------------------------------- #
def _harness(tmp_path, tids):
    root = tmp_path / "h"
    for tid in tids:
        cat = tid.split("_task_")[0]
        d = root / "tasks" / cat
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}_x.md").write_text(
            f"---\nid: {tid}\nname: X\ncategory: {cat}\ntimeout_seconds: 60\n"
            f"modality: pure-text\n---\n\n## Prompt\n\nDo it.\n", encoding="utf-8")
    return str(root)


def _score(out_root, tid, score):
    cat = tid.split("_task_")[0]
    d = os.path.join(out_root, "openclaw", cat, tid, "m_20260101_0000_abc123")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "score.json"), "w") as f:
        json.dump(score, f)


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    (wd / RUN.NETWORK_MONITOR_DIR).mkdir(parents=True)
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))
    for k in ("WILDCLAW_TASK", "WILDCLAW_CATEGORIES", "WILDCLAW_LIMIT", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    return wd


def _monitor(wd, rounds):
    _jsonl(wd / RUN.NETWORK_MONITOR_DIR / RUN.NETWORK_MONITOR, rounds)


def _ok_round(t):
    return {"t": t, "dns": {h: "ok" for h in W.TASK_HOSTS},
            "tcp": {h: "ok" for h in W.EGRESS_HOSTS}}


def test_grader_key_events_count_only_in_an_isolated_grade_exec(launcher, monkeypatch, tmp_path):
    tids = ["04_Search_Retrieval_task_1", "04_Search_Retrieval_task_2",
            "04_Search_Retrieval_task_3"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 0.0})
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 0.0, "end": 400.0, "ips": ["172.17.0.2"],
         "grading": [{"start": 300.0, "end": 350.0, "timed_out": False, "isolated": True}]},
        {"task_id": tids[1], "start": 0.0, "end": 400.0, "ips": ["172.17.0.3"],
         "grading": [{"start": 300.0, "end": 350.0, "timed_out": False, "isolated": False}]},
        {"task_id": tids[2], "start": 0.0, "end": 400.0, "ips": ["172.17.0.4"],
         "grading": [{"start": 300.0, "end": 320.0, "timed_out": True, "isolated": True}]}])
    _monitor(launcher, [_ok_round(t) for t in range(0, 420, 15)])
    _jsonl(launcher / RUN.HARNESS_EVENTS, [
        # a forged grader request during the agent's run: not counted
        {"time": 100.0, "kind": "judge", "client": "172.17.0.2", "start": 99.0},
        # a grader request during task_2's grade() while a leftover process ran: not counted
        {"time": 330.0, "kind": "judge", "client": "172.17.0.3", "start": 310.0},
        # task_3's grade() hit the cap; its judge request arrived in the exec, failed after it
        {"time": 360.0, "kind": "judge_web_search", "client": "172.17.0.4", "start": 315.0},
        # an agent-side service failure is attributed as before
        {"time": 150.0, "kind": "agent_helper", "client": "172.17.0.3"}])
    summary = RUN._summarize(out_root, RUN._task_index())
    kinds = {r["task_id"]: r["harness_errors"] for r in summary["per_task"]}
    assert kinds == {tids[0]: [], tids[1]: ["agent_helper"], tids[2]: ["judge_web_search"]}
    assert summary["grader_events_not_counted"] == 2
    assert summary["n_unattributed_harness_events"] == 0
    assert summary["harness_events"] == {"judge": 2, "judge_web_search": 1, "agent_helper": 1}


def test_cap_hits_by_kind(launcher, monkeypatch, tmp_path):
    tids = ["05_Creative_Synthesis_task_1", "05_Creative_Synthesis_task_2",
            "05_Creative_Synthesis_task_3"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    msg = "Command '['docker', 'exec']' timed out after 120 seconds"
    for t in tids:
        _score(out_root, t, {"overall_score": 0.0, "error": msg})
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 0.0, "end": 400.0, "ips": ["172.17.0.2"],
         "grading": [{"start": 200.0, "end": 320.0, "timed_out": True, "isolated": True}]},
        {"task_id": tids[1], "start": 0.0, "end": 400.0, "ips": ["172.17.0.3"],
         "grading": [{"start": 200.0, "end": 320.0, "timed_out": True, "isolated": False}]},
        {"task_id": tids[2], "start": 0.0, "end": 400.0, "ips": ["172.17.0.4"],
         "grading": []}])
    _monitor(launcher, [_ok_round(t) for t in range(0, 420, 15)])
    summary = RUN._summarize(out_root, RUN._task_index())
    rows = {r["task_id"]: r for r in summary["per_task"]}
    assert rows[tids[0]]["harness_errors"] == ["grading_infra"]
    assert rows[tids[0]]["grading_timeout"] == "without_judge_call"
    assert rows[tids[1]]["harness_errors"] == []                  # scored as upstream scores it
    assert rows[tids[1]]["grading_timeout"] == "not_isolated"
    assert rows[tids[2]]["harness_errors"] == ["grading_infra"]
    assert rows[tids[2]]["grading_timeout"] == "unrecorded"
    assert summary["grading_timeouts"] == {"with_judge_call": 0, "without_judge_call": 1,
                                           "unrecorded": 1, "not_isolated_infra": 0,
                                           "not_isolated": 1}
    assert summary["global_mean"] == pytest.approx(0.0) and summary["n_headline_tasks"] == 1


def test_isolation_is_summarized(launcher, monkeypatch, tmp_path):
    tids = ["01_Productivity_Flow_task_1", "01_Productivity_Flow_task_2"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 1.0})
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 0.0, "end": 100.0,
         "isolation": {"ok": True, "stopped": 5, "left": [], "error": None}},
        {"task_id": tids[1], "start": 0.0, "end": 100.0,
         "isolation": {"ok": False, "stopped": 2, "left": [9], "error": "exit 3",
                       "cause": "model"}}])
    _monitor(launcher, [_ok_round(t) for t in range(0, 120, 15)])
    summary = RUN._summarize(out_root, RUN._task_index())
    assert summary["grading_isolation"] == {"tasks_isolated": 1, "tasks_not_isolated": 1,
                                            "stop_failure_causes": {"model": 1},
                                            "processes_stopped": 7}
    row = next(r for r in summary["per_task"] if r["task_id"] == tids[1])
    assert row["grading_isolated"] is False and row["grading_isolation_error"] == "exit 3"
    assert row["grading_isolation_cause"] == "model"
    assert row["harness_errors"] == []
    assert summary["global_mean"] == pytest.approx(1.0)


def test_network_failures_during_a_task(launcher, monkeypatch, tmp_path):
    tids = ["01_Productivity_Flow_task_%d" % i for i in (1, 2, 3, 4)]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 0.0})
    down = {"t": 150.0, "dns": {h: "fail gaierror:-3" for h in W.TASK_HOSTS},
            "tcp": {h: "fail no-address" for h in W.EGRESS_HOSTS}}
    partial = _ok_round(450.0)
    partial["dns"]["arxiv.org"] = "fail gaierror:-3"
    rounds = [_ok_round(t) for t in range(0, 800, 15) if t not in (150, 450)] + [down, partial]
    _monitor(launcher, rounds)
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        # the resolver was down while task_1 ran
        {"task_id": tids[0], "start": 100.0, "end": 200.0},
        # one name failed while task_2 ran, and its logs show a failed lookup: one name failing
        # says nothing about the resolver the task used, so it is not confirmed
        {"task_id": tids[1], "start": 400.0, "end": 500.0,
         "name_resolution_errors": {"chat.jsonl": 2}},
        # the same partial failure, but task_3's logs show no failed lookup
        {"task_id": tids[2], "start": 420.0, "end": 480.0},
        # task_4's logs show a failed lookup the monitor never saw
        {"task_id": tids[3], "start": 600.0, "end": 700.0,
         "name_resolution_errors": {"agent.log": 1}}])
    summary = RUN._summarize(out_root, RUN._task_index())
    rows = {r["task_id"]: r for r in summary["per_task"]}
    assert rows[tids[0]]["harness_errors"] == ["network"]
    assert rows[tids[1]]["harness_errors"] == []
    assert rows[tids[2]]["harness_errors"] == []
    assert rows[tids[3]]["harness_errors"] == []
    assert rows[tids[3]]["name_resolution_errors_unconfirmed"] == {"agent.log": 1}
    assert summary["name_resolution_errors_unconfirmed"] == {tids[1]: {"chat.jsonl": 2},
                                                             tids[3]: {"agent.log": 1}}
    mon = summary["network_monitor"]
    assert mon["down_rounds"] == 1 and mon["failed_rounds"] == 2 and mon["gaps"] == []
    assert mon["host_failures"]["dns"]["arxiv.org"] == 2
    assert mon["host_failures"]["tcp"]["github.com"] == 1
    assert summary["harness_error_kinds"] == {"network": 1}
    assert summary["n_tasks_network_unmonitored"] == 0


def test_gaps_in_the_monitor_are_named(launcher, monkeypatch, tmp_path):
    tids = ["01_Productivity_Flow_task_1", "01_Productivity_Flow_task_2"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 1.0})
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 10.0, "end": 100.0},
        {"task_id": tids[1], "start": 100.0, "end": 400.0}])
    _monitor(launcher, [_ok_round(t) for t in (5.0, 20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0)])
    summary = RUN._summarize(out_root, RUN._task_index())
    assert summary["network_monitor"]["gaps"] == [[110.0, 400.0]]    # the monitor died at 110
    rows = {r["task_id"]: r for r in summary["per_task"]}
    assert "network_unmonitored" not in rows[tids[0]]
    assert rows[tids[1]]["network_unmonitored"] is True
    assert summary["n_tasks_network_unmonitored"] == 1
    assert RUN.monitor_gaps([], (0.0, 30.0)) == []
    assert RUN.monitor_gaps([], (0.0, 300.0)) == [[0.0, 300.0]]


def test_name_lookup_failures_in_logs(tmp_path):
    (tmp_path / "gateway.log").write_text(
        "fetch failed: getaddrinfo EAI_AGAIN arxiv.org\nall good\n"
        "socket.gaierror: [Errno -3] Temporary failure in name resolution\n")
    (tmp_path / "agent.log").write_text("no problem here\n")
    got = RUN.scan_name_resolution_errors([str(tmp_path / "gateway.log"),
                                           str(tmp_path / "agent.log"),
                                           str(tmp_path / "missing.log")])
    assert got == {"gateway.log": 2}
    assert RUN.count_name_resolution_errors("curl: (6) Could not resolve host: x\nok") == 1


# --------------------------------------------------------------------------- #
# The network monitor container and the preflight
# --------------------------------------------------------------------------- #
def test_the_monitor_runs_gbench_code_on_the_task_network(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("WILDCLAW_TASK_NETWORK", "bridge")
    monkeypatch.setenv("WILDCLAW_TASK_LABEL", "gbench_wildclaw_run=abc")
    monkeypatch.setenv("WILDCLAW_TASK_IMAGE", "wildclawbench-ubuntu:v1.3")
    monkeypatch.setenv("WILDCLAW_DNS_HOSTS", ",".join(W.TASK_HOSTS))
    monkeypatch.setenv("WILDCLAW_EGRESS_HOSTS", ",".join(W.EGRESS_HOSTS))
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "SECRETKEY0000000000000000000000000000")
    seen = []

    def run(cmd, *a, **k):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "cid\n", "")
    name, why = RUN.start_network_monitor(run)
    assert why is None and name.startswith("gbench-wildclaw-netmon-")
    [cmd] = seen
    out_dir = os.path.join(str(tmp_path), RUN.NETWORK_MONITOR_DIR)
    assert cmd[:4] == ["docker", "run", "-d", "--pull"] and cmd[4] == "never"
    for flag, value in (("--network", "bridge"), ("--label", "gbench_wildclaw_run=abc"),
                        ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"),
                        ("-v", f"{out_dir}:{out_dir}:rw"), ("--entrypoint", "python3")):
        assert cmd[cmd.index(flag) + 1] == value, flag
    assert "--read-only" in cmd and "-e" not in cmd and cmd.count("-v") == 1
    assert "wildclawbench-ubuntu:v1.3" in cmd and RUN._NETWORK_MONITOR_SCRIPT in cmd
    assert cmd[-3:] == [str(RUN.NETWORK_MONITOR_INTERVAL_S), ",".join(W.TASK_HOSTS),
                        ",".join(W.EGRESS_HOSTS)]
    assert not any("AIzaSECRET" in str(x) for x in cmd)


def test_the_monitor_script_writes_rounds(tmp_path):
    # run the script itself for one round: localhost, and a name whose 64-character label no
    # resolver is asked about (Python refuses to encode it), so nothing leaves the host
    out = tmp_path / "rounds.jsonl"
    bad = "a" * 64 + ".example"
    proc = subprocess.Popen([sys.executable, "-c", RUN._NETWORK_MONITOR_SCRIPT, str(out), "60",
                             f"localhost,{bad}", "localhost"])
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not (out.exists() and out.read_text().strip()):
            time.sleep(0.2)
    finally:
        proc.kill()
        proc.wait()
    rd = json.loads(out.read_text().splitlines()[0])
    assert rd["dns"]["localhost"] == "ok"
    assert rd["dns"][bad] in ("fail UnicodeError", "fail UnicodeEncodeError")
    assert rd["tcp"]["localhost"].startswith(("ok", "fail errno:"))
    c = RUN.classify_round({"t": rd["t"], "dns": rd["dns"], "tcp": rd["tcp"]})
    assert c["failed"] is True and c["dns_failed"] == [bad]
    # one egress host: down exactly when its connection failed (nothing may listen on 443 here)
    assert c["down"] is rd["tcp"]["localhost"].startswith("fail")


def _main(monkeypatch, tmp_path, first=None, start=("netmon", None), batch=None):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18795")
    monkeypatch.setenv("WILDCLAW_MODEL_RELAY_URL", "http://127.0.0.1:18795/model/v1")
    monkeypatch.setenv("WILDCLAW_TASK_NETWORK", "bridge")
    monkeypatch.setattr(RUN, "preflight_search_setup", lambda: (True, "{}"))
    stopped = []
    monkeypatch.setattr(RUN, "start_network_monitor", lambda: start)
    monkeypatch.setattr(RUN, "wait_first_round", lambda name: (first, None if first else "none"))
    monkeypatch.setattr(RUN, "stop_network_monitor", lambda name: stopped.append(name))
    ran = []
    monkeypatch.setattr(RUN, "_run_batch", batch or (lambda cfg, model: ran.append(1) or
                                                     str(tmp_path / "output")))
    monkeypatch.setattr(RUN, "_task_index", lambda: {})
    return stopped, ran


def test_no_dns_on_the_task_network_stops_the_run_before_any_task(monkeypatch, tmp_path, capsys):
    first = {"t": 1.0, "dns": {h: "fail gaierror:-3" for h in W.TASK_HOSTS},
             "tcp": {h: "fail no-address" for h in W.EGRESS_HOSTS}}
    stopped, ran = _main(monkeypatch, tmp_path, first=first)
    assert RUN.main() == 5
    err = capsys.readouterr().err
    assert "PREFLIGHT FAILED" in err and "cannot resolve host names" in err
    assert CG.FIREWALL_COMMAND in err
    assert ran == [] and stopped == ["netmon"]


def test_no_egress_stops_the_run(monkeypatch, tmp_path, capsys):
    first = {"t": 1.0, "dns": {h: "ok" for h in W.TASK_HOSTS},
             "tcp": {h: "fail timeout" for h in W.EGRESS_HOSTS}}
    stopped, ran = _main(monkeypatch, tmp_path, first=first)
    assert RUN.main() == 5
    assert "cannot open a connection to the web" in capsys.readouterr().err
    assert ran == [] and stopped == ["netmon"]


def test_a_monitor_that_does_not_start_stops_the_run(monkeypatch, tmp_path, capsys):
    stopped, ran = _main(monkeypatch, tmp_path, start=(None, "docker run failed"))
    assert RUN.main() == 5
    assert "the network monitor of the task network could not start" in capsys.readouterr().err
    assert ran == [] and stopped == []
    stopped, ran = _main(monkeypatch, tmp_path, first=None)
    assert RUN.main() == 5 and ran == [] and stopped == ["netmon"]


def test_the_monitor_is_removed_when_the_run_raises(monkeypatch, tmp_path):
    def boom(cfg, model):
        raise RuntimeError("run_batch crashed")
    stopped, _ran = _main(monkeypatch, tmp_path, first=_ok_round(1.0), batch=boom)
    with pytest.raises(RuntimeError, match="run_batch crashed"):
        RUN.main()
    assert stopped == ["netmon"]


def test_wait_first_round_reports_a_stopped_monitor(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))

    def run(cmd, *a, **k):
        if cmd[:4] == ["docker", "inspect", "-f", "{{.State.Running}}"]:
            return subprocess.CompletedProcess(cmd, 0, "false\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "Traceback: boom")
    first, why = RUN.wait_first_round("netmon", run, timeout_s=5, sleep=lambda s: None)
    assert first is None and "the monitor container stopped" in why and "boom" in why
    clock = iter([0.0, 0.0, 10.0])

    def running(cmd, *a, **k):
        return subprocess.CompletedProcess(cmd, 0, "true\n", "")
    first, why = RUN.wait_first_round("netmon", running, timeout_s=5, sleep=lambda s: None,
                                      clock=lambda: next(clock))
    assert first is None and "no result within 5 s" in why


# --------------------------------------------------------------------------- #
# The proxy: a chat request's event records when it arrived
# --------------------------------------------------------------------------- #
def test_proxy_events_record_when_the_request_arrived(monkeypatch, tmp_path):
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    J.record_harness_event("judge", "x", "172.17.0.2", start=12.5)
    J.record_harness_event("model_endpoint", "HTTP 503", "172.17.0.2")
    lines = [json.loads(x) for x in (tmp_path / J.HARNESS_EVENTS).read_text().splitlines()]
    assert lines[0]["start"] == 12.5 and lines[0]["kind"] == "judge"
    assert "start" not in lines[1]
    src = _read("gbench/docker/wildclawbench_cascade_judge.py")
    assert "self._chat_completion(client, grader, started)" in src


# --------------------------------------------------------------------------- #
# Docs follow the code
# --------------------------------------------------------------------------- #
def test_the_guide_describes_isolation_cap_hits_and_the_network():
    doc = _read("docs/evals/wildclawbench.md")
    flat = _flat(doc)
    for kind, _label in W.HARNESS_ERROR_LABELS:
        assert f"`{kind}`" in doc, kind
    for key in ("grader_events_not_counted", "grading_timeouts", "grading_isolation",
                "network_monitor", "name_resolution_errors_unconfirmed", "task_hosts",
                "n_tasks_network_unmonitored", "process_baseline", "with_judge_call",
                "without_judge_call", "not_isolated", "unrecorded"):
        assert f"`{key}" in doc, key
    for host in W.TASK_HOSTS:
        assert f"`{host}`" in doc, host
    assert "### Grading isolation" in doc and "### Network: DNS and web egress" in doc
    assert f"`org.gbench.wildclawbench-harness-errors={W.HARNESS_ERRORS_VERSION}`" in doc
    assert CG.FIREWALL_COMMAND in doc and CG.PLAIN_RULE_DELETE_COMMAND in doc
    assert f"Every {RUN.NETWORK_MONITOR_INTERVAL_S} s it resolves those names" in flat
    assert f"gaps longer than {RUN.NETWORK_MONITOR_GAP_S} s" in flat
    assert "whether a judge call of that `grade()` was in flight or not" in flat
    assert "A `grade()` that hits the cap without any judge call" not in flat
    assert "it scores 0" not in doc and "counted as 0" not in doc


def test_the_changed_files_are_plain_ascii_without_internal_labels():
    import re
    labels = re.compile("|".join([r"\b[Ww]" + r"ave ?\d", "the " + "reviewer", r"\bW" + r"2O\b"]))
    for rel in ("gbench/runners/eval_suites/wildclawbench.py", "gbench/docker/wildclawbench_run.py",
                "gbench/docker/wildclawbench_cascade_judge.py",
                "gbench/docker/wildclawbench.Dockerfile", "docs/evals/wildclawbench.md",
                "tests/test_wildclawbench_grading_isolation_and_network.py"):
        text = _read(rel)
        assert all(ord(c) < 128 for c in text), rel
        assert not labels.search(text), rel


def test_container_addresses_keep_only_addresses():
    def run(cmd, *a, **k):
        return subprocess.CompletedProcess(cmd, 0, "172.17.0.8 invalid IP <no value> fd00::5 \n", "")
    assert RUN.container_ips("t1", run) == ["172.17.0.8", "fd00::5"]
