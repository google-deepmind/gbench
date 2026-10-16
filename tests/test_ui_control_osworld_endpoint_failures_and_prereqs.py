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

"""ui_control_osworld: a task during which a model request failed on the endpoint's side is a
harness error (the launcher's per-example record); the prerequisite check names every missing
prerequisite and refuses a task-image override the provider cannot follow; GBENCH_OSWORLD_LIMIT
is checked before anything starts; the run directory is the invoking user's; a merge keeps each
shard's selection. Docker, OSWorld and the network are stubbed; nothing is written outside tmp."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import ui_control_osworld as U

_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import ui_control_osworld_run as RUN  # noqa: E402


# --------------------------------------------------------------------------- #
# The launcher: the per-example record of failed model requests.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status, exc, harness", [
    (None, ConnectionError("x"), True), (500, None, True), (503, None, True), (429, None, True),
    (401, None, True), (404, None, True), (400, None, False), (413, None, False),
    (422, None, False)])
def test_what_counts_as_the_endpoints_failure(status, exc, harness):
    assert RUN.endpoint_failure(status, exc) is harness


def test_the_request_wrapper_records_failures_in_the_running_examples_dir(monkeypatch, tmp_path):
    replies = iter([types.SimpleNamespace(status_code=200), types.SimpleNamespace(status_code=503),
                    types.SimpleNamespace(status_code=400), types.SimpleNamespace(status_code=500)])

    def post(url, *a, **k):
        if k.get("json", {}).get("boom"):
            raise ConnectionError("refused")
        return next(replies)
    agent = types.SimpleNamespace(requests=types.SimpleNamespace(post=post))
    mm = types.ModuleType("mm_agents")
    mm.agent = agent
    monkeypatch.setitem(sys.modules, "mm_agents", mm)
    monkeypatch.setitem(sys.modules, "mm_agents.agent", agent)
    RUN._install_model_rewrite("served")
    ex = tmp_path / "ex1"
    ex.mkdir()
    monkeypatch.setitem(RUN._CURRENT, "dir", str(ex))
    url = "http://127.0.0.1:1/v1/chat/completions"
    for _ in range(3):
        agent.requests.post(url, json={"model": "gpt-4o"})
    with pytest.raises(ConnectionError):
        agent.requests.post(url, json={"model": "gpt-4o", "boom": 1})
    agent.requests.post("http://x/other", json={})       # not a chat request: never recorded
    lines = [json.loads(x) for x in (ex / RUN.ENDPOINT_FAILURES).read_text().splitlines()]
    assert [(ln["status"], ln["error"], ln["harness"]) for ln in lines] == [
        (503, None, True), (400, None, False), (None, "ConnectionError", True)]
    assert RUN.endpoint_failures(str(ex)) == 2
    assert RUN.endpoint_failures(str(tmp_path / "none")) == 0


def test_nothing_is_recorded_outside_an_example(monkeypatch, tmp_path):
    monkeypatch.setitem(RUN._CURRENT, "dir", None)
    RUN.record_endpoint_failure(status=503)
    assert not list(tmp_path.iterdir())


def test_the_example_tracking_wraps_run_single_example(monkeypatch, tmp_path):
    seen = []

    def run_single_example(agent, env, example, max_steps, instruction, args, example_dir,
                           scores):
        seen.append(RUN._CURRENT["dir"])
    lib = types.ModuleType("lib_run_single")
    lib.run_single_example = run_single_example
    monkeypatch.setitem(sys.modules, "lib_run_single", lib)
    RUN._install_example_tracking()
    lib.run_single_example(None, None, {}, 1, "", None, str(tmp_path / "ex"), [])
    assert seen == [str(tmp_path / "ex")] and RUN._CURRENT["dir"] is None


def test_the_summary_counts_each_tasks_endpoint_failures(tmp_path):
    base = tmp_path / "results" / "pyautogui" / "screenshot" / "gpt-4o" / "os"
    for ex, result, failed in (("a", "1.0", 0), ("b", "0.0", 2)):
        (base / ex).mkdir(parents=True)
        (base / ex / "result.txt").write_text(result)
        for _ in range(failed):
            with open(base / ex / RUN.ENDPOINT_FAILURES, "a") as f:
                f.write(json.dumps({"status": 503, "harness": True}) + "\n")
    summary = RUN._summarize("pyautogui", "gpt-4o", "screenshot", str(tmp_path / "results"),
                             {"os": ["a", "b"]})
    assert [t["endpoint_failures"] for t in summary["tasks"]] == [0, 2]
    assert summary["n_endpoint_failed"] == 1


def test_the_run_directory_is_handed_to_the_owner(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "x").write_text("1")
    owned = []
    monkeypatch.setattr(RUN.os, "lchown", lambda p, u, g: owned.append((os.path.relpath(
        p, tmp_path), u, g)))
    monkeypatch.setenv("OSWORLD_HOST_OWNER", "1234:5678")
    RUN.hand_over(str(tmp_path))
    assert sorted(owned) == [(".", 1234, 5678), ("results", 1234, 5678),
                             (os.path.join("results", "x"), 1234, 5678)]
    owned.clear()
    monkeypatch.setenv("OSWORLD_HOST_OWNER", "not-an-owner")
    RUN.hand_over(str(tmp_path))
    assert owned == []


# --------------------------------------------------------------------------- #
# gbench: endpoint-failed tasks left out of the score.
# --------------------------------------------------------------------------- #
_SUMMARY = {"success_rate": 0.5, "n_selected": 3, "n_scored": 3, "n_missing": 1,
            "per_domain": {"os": {"success_rate": 0.5, "n": 3, "n_scored": 2}},
            "tasks": [{"task": "os/a", "domain": "os", "result": 1.0, "endpoint_failures": 0},
                      {"task": "os/b", "domain": "os", "result": 0.0, "endpoint_failures": 1},
                      {"task": "os/c", "domain": "os", "result": None, "endpoint_failures": 0}]}


def test_the_score_leaves_out_endpoint_failed_tasks():
    scored = U.compute_ui_control_osworld_score(_SUMMARY)
    assert scored["success_rate"] == 1.0 and scored["success_rate_missing_as_zero"] == 0.5
    assert scored["per_domain"]["os"]["n_scored"] == 1
    assert scored["per_domain"]["os"]["n_endpoint_failed"] == 1
    traces = {t["task"]: t for t in U.task_traces(_SUMMARY)}
    assert traces["os/b"]["harness_error"] == "endpoint_failed" and traces["os/b"][
        "scoring_excluded"]
    assert traces["os/c"]["harness_error"] == "no_result"
    assert "harness_error" not in traces["os/a"]


def _run(monkeypatch, tmp_path, summary):
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.delenv("GBENCH_OSWORLD_LIMIT", raising=False)
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites", lambda: (True, ""))
    seen = {}

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            seen["cmd"] = list(cmd)
            run_dir = next(c.split("=", 1)[1] for c in cmd if c.startswith("OSWORLD_RUN_DIR="))
            seen["mode"] = os.stat(run_dir).st_mode & 0o777
            (Path(run_dir) / "ui_control_osworld_summary.json").write_text(json.dumps(summary))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(U.subprocess, "run", fake_run)
    return U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:8000/v1"), seen


def test_a_run_names_endpoint_failed_tasks_and_keeps_its_directory_private(monkeypatch,
                                                                           tmp_path):
    res, seen = _run(monkeypatch, tmp_path, _SUMMARY)
    assert res["accuracy"] == 100.0 and res["n_endpoint_failed"] == 1 and res["n_scored"] == 1
    assert res["scoring_excluded"] == 2 and res["status"] == "completed_with_errors"
    assert "1 of 3 task(s) had a model request that got no reply" in res["status_reason"]
    assert "1 task(s) had a model request fail on the endpoint's side" in \
        res["leaderboard_comparable_reason"]
    assert seen["mode"] == 0o700
    assert f"OSWORLD_HOST_OWNER={os.getuid()}:{os.getgid()}" in seen["cmd"]
    assert res["task_image"]["image"] == "happysixd/osworld-docker"


# --------------------------------------------------------------------------- #
# gbench: the temperature (1.0 with --thinking, 0.0 without, unless the operator set one) reaches
# run.py's --temperature and every request through the relay.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thinking,env,want,source", [
    (False, None, 0.0, "default"), (True, None, 1.0, "thinking"),
    (False, "0.4", 0.4, "env:GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE")])
def test_the_temperature_reaches_run_py_and_every_request(monkeypatch, tmp_path, thinking, env,
                                                          want, source):
    from gbench.runners.eval_suites import base as B
    monkeypatch.delitem(B._RUN_KNOBS, "temperature", raising=False)
    if env is None:
        monkeypatch.delenv("GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE", env)
    relays = []
    real_relay = U.model_endpoint_relay

    def relay(endpoint, key, temperature=None):
        relays.append(real_relay(endpoint, key, temperature))
        return relays[-1]
    monkeypatch.setattr(U, "model_endpoint_relay", relay)
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.delenv("GBENCH_OSWORLD_LIMIT", raising=False)
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites", lambda: (True, ""))
    seen = {}

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            seen["cmd"] = list(cmd)
            run_dir = next(c.split("=", 1)[1] for c in cmd if c.startswith("OSWORLD_RUN_DIR="))
            (Path(run_dir) / "ui_control_osworld_summary.json").write_text(json.dumps(_SUMMARY))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(U.subprocess, "run", fake_run)
    res = U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:8000/v1",
                                   enable_thinking=thinking)
    assert f"OSWORLD_TEMPERATURE={want}" in seen["cmd"]
    assert (res["temperature"], res["temperature_source"]) == (want, source)
    # the relay sets it on the PromptAgent's request body, over run.py's own value
    body = json.dumps({"model": "m", "temperature": 1.0, "top_p": 0.9}).encode()
    sent = json.loads(relays[0].patch_body("/v1/chat/completions", body, "application/json"))
    assert sent["temperature"] == want and sent["top_p"] == 0.9

    # the launcher hands it to run.py as --temperature
    argv = {}
    fake = types.ModuleType("run")
    fake.config = lambda: (argv.setdefault("argv", list(sys.argv)), types.SimpleNamespace(
        action_space="pyautogui", model="gpt-4o", observation_type="screenshot",
        result_dir=str(tmp_path)))[1]
    fake.get_unfinished = lambda *a: []
    fake.test = lambda *a: None
    monkeypatch.setitem(sys.modules, "run", fake)
    monkeypatch.setattr(RUN, "_install_example_tracking", lambda: None)
    monkeypatch.setattr(RUN, "_run_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setenv("OSWORLD_TEMPERATURE", str(want))
    subset = tmp_path / "subset.json"
    subset.write_text("{}")
    RUN._run_osworld(str(subset))
    a = argv["argv"]
    assert a[a.index("--temperature") + 1] == str(want)


# --------------------------------------------------------------------------- #
# gbench: the prerequisite check, the task image, the limit.
# --------------------------------------------------------------------------- #
def test_the_check_names_every_missing_prerequisite(monkeypatch):
    monkeypatch.setattr(U.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(U.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(U, "_docker_image_exists", lambda image: image == U._image())
    monkeypatch.setattr(U, "image_launcher_version", lambda image: U.LAUNCHER_VERSION)
    real_exists = os.path.exists
    monkeypatch.setattr(U.os.path, "exists",
                        lambda p: False if p == "/dev/kvm" else real_exists(p))
    monkeypatch.delenv("GBENCH_OSWORLD_VM_DIR", raising=False)
    monkeypatch.setenv("GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE", "other/vm:1")
    ok, why = U.check_ui_control_osworld_prerequisites()
    assert not ok
    for needle in ("(1) GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE='other/vm:1' cannot change",
                   "(2) /dev/kvm is not present", "(3) GBENCH_OSWORLD_VM_DIR is unset",
                   "(4) the task VM image 'happysixd/osworld-docker' is not on the host daemon"):
        assert needle in why, needle


def test_the_task_image_is_always_the_providers_name(monkeypatch):
    monkeypatch.setenv("GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE", "happysixd/osworld-docker")
    assert U.task_image_override_refusal() is None
    monkeypatch.setenv("GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE", "evil/vm@sha256:" + "0" * 64)
    assert "starts 'happysixd/osworld-docker' by name" in U.task_image_override_refusal()
    assert U._task_image() == "happysixd/osworld-docker"


@pytest.mark.parametrize("value", ["0", "-1", "abc", "2.5"])
def test_a_bad_limit_is_refused_before_anything_starts(monkeypatch, value):
    monkeypatch.setenv("GBENCH_OSWORLD_LIMIT", value)
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites",
                        lambda: pytest.fail("the prerequisite check ran"))
    monkeypatch.setattr(U.subprocess, "run", lambda *a, **k: pytest.fail("a process started"))
    with pytest.raises(RuntimeError, match="must be a positive integer"):
        U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:8000/v1")


def test_a_good_limit_is_forwarded(monkeypatch):
    assert U.osworld_limit("3") == 3 and U.osworld_limit("") is None and U.osworld_limit(None) is None


def test_a_merge_keeps_each_shards_selection_and_its_harness_errors():
    a = {"shard": "1/2", "selection": {"n_full": 4, "limit": None},
         "sample_traces": U.task_traces({"tasks": _SUMMARY["tasks"][:2]}), "status": "success"}
    b = {"shard": "2/2", "selection": {"n_full": 4, "limit": None},
         "sample_traces": U.task_traces({"tasks": _SUMMARY["tasks"][2:]}), "status": "success"}
    m = U.merge_shards([a, b])
    assert "selection" not in m and m["selection_by_shard"] == {"1/2": a["selection"],
                                                                "2/2": b["selection"]}
    assert m["n_scored"] == 1 and m["n_missing"] == 1 and m["n_endpoint_failed"] == 1
    assert m["accuracy"] == 100.0 and m["status"] == "completed_with_errors"


def test_the_guide_describes_the_new_rules():
    doc = " ".join(open("docs/evals/ui_control_osworld.md", encoding="utf-8").read().split())
    for needle in ("`gbench_endpoint_failures.jsonl`", "`n_endpoint_failed`",
                   "`selection_by_shard`", "`OSWORLD_HOST_OWNER`", "mode `0700`",
                   "must be a positive integer", "every missing one in the same message"):
        assert needle in doc, needle
