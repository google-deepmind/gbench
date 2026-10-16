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

"""mcp_bench, toolbench, mcp_atlas and cimemories: credentials, the pinned MCP client, the judge
requests and the SoWR reference.

  * mcp_bench / toolbench - every credential reaches the container by name, never as a value on
    the docker command line, so neither a TimeoutExpired (whose message is the whole command) nor
    a quoted container tail can carry one.
  * mcp_bench image      - the MCP client is pinned inside upstream's declared range, every
    installer step runs, every `python` server shares the one interpreter, uv prefers the image's
    own Pythons.
  * mcp_bench judge      - the request carries upstream's fields only (no temperature) and rotates
    the configured keys.
  * mcp_bench run        - no variable bypasses the pre-run check; protocol overrides are named.
  * toolbench            - ToolEval's max_tokens 1000 reaches the judge and cut replies are
    counted; SoWR needs a reference answer for every test id the run covers; the cached server's
    cache-miss path is upstream's (real-API proxy, then the simulator) and the simulator's model is
    the judge cascade.
  * mcp_atlas / cimemories - no max_tokens is sent, as their harnesses send none.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import traceback

import pytest

from gbench.runners.eval_suites import cimemories as CIM
from gbench.runners.eval_suites import mcp_atlas as ATLAS
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCKER_DIR = os.path.join(_HERE, "..", "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import mcp_bench_cascade_judge as CJ  # noqa: E402

GEMINI = "AIzaSyTESTgeminiKEY0000001111"
GEMINI_LIST = GEMINI + ",AIzaSyTESTsecondKEY222233334444"
SERVER = {"GOOGLE_MAPS_API_KEY": "maps-secret-55556666", "NCI_API_KEY": "nci-secret-77778888",
          "HF_TOKEN": "hf_secret_99990000aaaa", "NPS_API_KEY": "nps-secret-bbbbcccc",
          "NASA_API_KEY": "nasa-secret-ddddeeee"}
SIM_KEY = "simulator-secret-ffff0000"
SECRETS = [GEMINI, GEMINI_LIST, SIM_KEY] + list(SERVER.values())


def _read(*parts):
    with open(os.path.join(_HERE, "..", *parts), encoding="utf-8") as fh:
        return fh.read()


def _doc(name):
    return " ".join(_read("docs", "evals", name).split())


def _no_secret(text):
    return not any(s in text for s in SECRETS) and "AIzaSyTEST" not in text


def _exc_text(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


# --------------------------------------------------------------------------- #
# mcp_bench: credentials by name
# --------------------------------------------------------------------------- #
_CONNECTED = {f"S{i}": {"status": "success", "tools": 3, "error": ""} for i in range(28)}
_PROBE_OK = {"servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
             "connections": _CONNECTED}


@pytest.fixture
def mb_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setenv("GEMINI_API_KEYS", GEMINI_LIST)
    for k, v in SERVER.items():
        monkeypatch.setenv(k, v)
    for k in ("GBENCH_MCP_BENCH_DISTRACTION", "GBENCH_MCP_BENCH_DISABLE_STABILITY",
              "GBENCH_MCP_BENCH_ASSUME_NETWORK", "GBENCH_SHARD", "GBENCH_MCP_BENCH_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: _PROBE_OK)
    # the judge's Gemini relay (tests/test_mcp_credential_isolation_and_dns.py covers it): a
    # stand-in, docker is stubbed
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: _RelayStandIn())
    # the credentialed-servers container (tests/test_mcp_wildclawbench_credentials_and_cascade.py
    # covers it): a stand-in
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: _CredStandIn())


class _CredStandIn:
    def client_env(self):
        return {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10", "MCP_CREDENTIALED_TOKEN": "t0k"}

    def stop(self):
        pass

    def record(self):
        return {}


#: The relay placeholder the stand-in hands out (the shape of gemini_relay's).
RELAY_PLACEHOLDER = "gbench-gemini-relay-" + "0" * 48


class _RelayStandIn:
    def client_env(self):
        return {"GEMINI_API_KEY": RELAY_PLACEHOLDER, "GEMINI_API_KEYS": RELAY_PLACEHOLDER,
                "GEMINI_OPENAI_BASE_URL": "http://172.17.0.1:9/v1beta/openai",
                "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:9"}

    def stop(self):
        pass

    def record(self):
        return {}


def _mb_container(monkeypatch, behaviour):
    """Stub subprocess.run: `docker run` calls `behaviour(cmd, kw)`; other docker calls succeed."""
    runs = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            runs.append((cmd, kw))
            return behaviour(cmd, kw)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    return runs


def _mb_results(cmd, kw):
    out = cmd[cmd.index("-v") + 1].split(":")[0]
    metrics = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
               "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
               "tool_selection_score": 5.0, "planning_effectiveness_and_efficiency_score": 5.0}
    with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
        json.dump(metrics, fh)
    limited = any(str(a).startswith("MCP_TASK_LIMIT=") for a in cmd)
    with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
        # a limited run's launcher reports its seeded draw (limit_draw)
        json.dump({"filtered": True, "unavailable_servers": [], "tasks_kept": 104,
                   "limit_draw": "stratified_indices(key=server group, seed=mcp_bench)"
                   if limited else None}, fh)
    with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
        json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": [
            {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": []}]}, fh)
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_mcp_bench_run_passes_every_credential_by_name(mb_env, monkeypatch):
    runs = _mb_container(monkeypatch, _mb_results)
    res = MB.run_mcp_bench("m", "http://x/v1")
    [(cmd, kw)] = runs
    assert _no_secret(" ".join(cmd))
    names = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
    for key in ["GEMINI_API_KEY", "GEMINI_API_KEYS", "MCP_ENDPOINT", "MCP_CREDENTIALED_TOKEN"]:
        assert key in names, key                  # by name: docker reads the value from env
    env = kw.get("env") or os.environ
    # the judge's Gemini key stays in gbench: the container gets the relay's placeholder
    assert env["GEMINI_API_KEY"] == env["GEMINI_API_KEYS"] == RELAY_PLACEHOLDER
    assert not any(GEMINI in str(v) for v in env.values() if v)
    # the server keys go only to the credentialed-servers container, never to this one
    for key in SERVER:
        assert key not in names and key not in env, key
    # the one override: gbench's no-think temperature 0.0 (upstream sends none)
    assert res["status"] == "success" and res["protocol_overrides"] == ["temperature 0.0 set by default (upstream sends none)"]


def test_mcp_bench_timeout_message_carries_no_credential(mb_env, monkeypatch):
    def timeout(cmd, kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    _mb_container(monkeypatch, timeout)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*exceeded") as info:
        MB.run_mcp_bench("m", "http://x/v1")
    text = _exc_text(info.value)
    assert "TimeoutExpired" in text and _no_secret(text)


def test_mcp_bench_container_tail_is_redacted(mb_env, monkeypatch):
    def failed(cmd, kw):
        return subprocess.CompletedProcess(
            cmd, 1, "", f"boom key={GEMINI} nasa={SERVER['NASA_API_KEY']}")

    _mb_container(monkeypatch, failed)
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*rc=1.*\[REDACTED\]") as info:
        MB.run_mcp_bench("m", "http://x/v1")
    assert _no_secret(_exc_text(info.value))


def test_mcp_bench_probe_tail_is_redacted(monkeypatch):
    monkeypatch.setenv("NASA_API_KEY", SERVER["NASA_API_KEY"])
    monkeypatch.setattr(MB.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "api_key = " + SERVER["NASA_API_KEY"]))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED") as info:
        MB.provisioning_probe("img")
    assert SERVER["NASA_API_KEY"] not in str(info.value) and "[REDACTED]" in str(info.value)


# --------------------------------------------------------------------------- #
# mcp_bench: the gate has no bypass; overrides are named
# --------------------------------------------------------------------------- #
def test_mcp_bench_gap_is_a_hard_error_whatever_the_environment(mb_env, monkeypatch):
    gap = dict(_PROBE_OK, connections=dict(_CONNECTED, S0={"status": "success_no_tools",
                                                            "tools": 0, "error": ""}))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: gap)
    monkeypatch.setenv("GBENCH_MCP_BENCH_NO_SUBSET", "1")
    runs = _mb_container(monkeypatch, _mb_results)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*S0 \\(connected, but listed no tools"):
        MB.run_mcp_bench("m", "http://x/v1")
    assert runs == []


def test_mcp_bench_forwards_every_judge_cascade_knob(mb_env, monkeypatch):
    knobs = {"GBENCH_JUDGE_MODELS": "g1,g2", "GBENCH_JUDGE_CASCADE_ROUNDS": "4",
             "GBENCH_JUDGE_BACKOFF": "0.5"}
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)
    runs = _mb_container(monkeypatch, _mb_results)
    MB.run_mcp_bench("m", "http://x/v1")
    cmd = runs[0][0]
    for k, v in knobs.items():
        assert f"{k}={v}" in cmd, k
    assert "`GBENCH_JUDGE_CASCADE_ROUNDS` / `GBENCH_JUDGE_BACKOFF` | passed through" in (
        _doc("mcp_bench.md"))


def test_mcp_bench_names_protocol_overrides_and_the_task_limit(mb_env, monkeypatch):
    runs = _mb_container(monkeypatch, _mb_results)
    monkeypatch.setenv("GBENCH_MCP_BENCH_DISTRACTION", "0")
    monkeypatch.setenv("GBENCH_MCP_BENCH_DISABLE_STABILITY", "1")
    res = MB.run_mcp_bench("m", "http://x/v1", limit=5)
    cmd = runs[0][0]
    assert "MCP_DISTRACTION_COUNT=0" in cmd and "MCP_DISABLE_STABILITY=1" in cmd
    assert not any("NO_SUBSET" in a for a in cmd)
    reason = res["leaderboard_comparable_reason"]
    assert "0 distraction servers per task instead of upstream's 10" in reason
    assert "judge stability off: one judge pass per task instead of upstream's 5" in reason
    assert "--eval-limit ran only 5 tasks (a seeded draw" in reason and res["is_full"] is False
    # the two knobs and gbench's no-think temperature 0.0
    assert len(res["protocol_overrides"]) == 3
    # upstream's own value is not an override
    monkeypatch.setenv("GBENCH_MCP_BENCH_DISTRACTION", "10")
    monkeypatch.delenv("GBENCH_MCP_BENCH_DISABLE_STABILITY")
    assert MB.run_mcp_bench("m", "http://x/v1")["protocol_overrides"] == ["temperature 0.0 set by default (upstream sends none)"]


# --------------------------------------------------------------------------- #
# mcp_bench image: the pinned client and the one environment
# --------------------------------------------------------------------------- #
def test_mcp_bench_client_pin_is_inside_upstreams_declared_range():
    from packaging.requirements import Requirement
    from packaging.version import Version
    text = _read("gbench", "docker", "mcp_bench_constraints.txt")
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    # the client pin, among the pins of the rest of the checked image's set
    assert [ln for ln in lines if ln.lower().startswith("mcp==")] == ["mcp==1.16.0"]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", ln) for ln in lines), lines
    # Accenture/mcp-bench@7a8eaea mcp_servers/requirements.txt: mcp[cli]>=1.9.0 and mcp>=1.0.0
    for declared in ("mcp[cli]>=1.9.0", "mcp>=1.0.0"):
        assert Version("1.16.0") in Requirement(declared).specifier


def test_mcp_bench_dockerfile_resolves_the_system_python_against_the_pin():
    df = _read("gbench", "docker", "mcp_bench.Dockerfile")
    assert "COPY mcp_bench_constraints.txt /app/constraints.txt" in df
    run_install = df[df.index("RUN cd /app/mcp-bench/mcp_servers \\"):]
    run_install = run_install[:run_install.index("true\n")]
    assert "PIP_CONSTRAINT=/app/constraints.txt UV_CONSTRAINT=/app/constraints.txt" in run_install
    # every installer step runs: only its `set -e` line changes, and $0 stays install.sh's path
    assert "sed 's/^set -e.*/set +e/' install.sh" in run_install
    assert run_install.rstrip().endswith("/app/mcp-bench/mcp_servers/install.sh; \\")
    # the drift guard is a step of its own that fails the build (no trailing `; true` after it)
    guard = df[:df.index("RUN cd /app/mcp-bench/mcp_servers \\")]
    guard = guard[guard.rindex("RUN "):]
    assert "grep -c '^set -e' /app/mcp-bench/mcp_servers/install.sh" in guard
    assert "exit 1" in guard and "true" not in guard
    assert "export PIP_CONSTRAINT=/app/constraints.txt" in df
    assert "\"mcp[cli]==1.16.0\"" in df and "assert v == '1.16.0'" in df
    assert "mcp<2" not in df
    # no server gets a venv of its own beyond the `uv run` servers upstream's commands.json has
    assert "uv venv" not in df and "fastmcp venv" not in df
    assert "ENV UV_PYTHON_PREFERENCE=system" in df
    assert df.index("ENV UV_PYTHON_PREFERENCE=system") < df.index("bash -c \"$(sed")


def test_mcp_bench_install_step_transform_keeps_the_script_directory(tmp_path):
    """The installer copy runs with `set +e` and still finds its own directory through $0."""
    srv = tmp_path / "mcp_servers"
    srv.mkdir()
    (srv / "install.sh").write_text(
        "#!/bin/bash\nset -e  # Exit on any error\n"
        "main() { cd \"$(dirname \"$0\")\"; false; pwd > ran_in.txt; }\nmain \"$@\"\n")
    script = subprocess.run(["sed", "s/^set -e.*/set +e/", str(srv / "install.sh")],
                            capture_output=True, text=True, check=True).stdout
    subprocess.run(["bash", "-c", script, str(srv / "install.sh")], cwd=str(tmp_path), check=True)
    assert (srv / "ran_in.txt").read_text().strip() == str(srv)


# --------------------------------------------------------------------------- #
# mcp_bench in-image judge: upstream's request, key rotation
# --------------------------------------------------------------------------- #
class _Completions:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, []

    async def create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError(f"401 bad key {GEMINI}")
        msg = type("M", (), {"content": '{"score": 7}'})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()], "usage": None})()


class _Client:
    def __init__(self, name, fail=False):
        self.chat = type("Chat", (), {"completions": _Completions(name, fail)})()


def test_mcp_bench_judge_request_carries_upstreams_fields_only(monkeypatch):
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "g1")
    j = CJ.CascadeGeminiJudge()
    client = _Client("a")
    j._async_clients = [client]
    import itertools
    j._client_cycle = itertools.cycle(j._async_clients)
    assert asyncio.run(j.get_completion("sys", "user", 4000)) == '{"score": 7}'
    [kw] = client.chat.completions.calls
    # llm/provider.py get_completion: {model, messages, max_tokens}; no temperature
    assert set(kw) == {"model", "messages", "max_tokens"}
    assert kw["max_tokens"] == 4000 and kw["messages"][0] == {"role": "system", "content": "sys"}
    src = _read("gbench", "docker", "mcp_bench_cascade_judge.py")
    assert "temperature=" not in src


def test_mcp_bench_judge_rotates_keys_per_round_and_hides_them(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", GEMINI_LIST)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "g1,g2")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "2")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "0")
    assert CJ._api_keys() == GEMINI_LIST.split(",")
    built = []

    class FakeAsyncOpenAI:
        def __init__(self, api_key, base_url, max_retries=None):
            assert max_retries == 0         # one attempt: the cascade moves on itself
            built.append(api_key)
            self.chat = _Client(api_key, fail=True).chat

    import types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    j = CJ.CascadeGeminiJudge()
    with pytest.raises(RuntimeError, match="JUDGE_OUTAGE") as info:
        asyncio.run(j.get_completion("s", "u", 10))
    assert built == GEMINI_LIST.split(",")                  # one client per key
    per_client = [len(c.chat.completions.calls) for c in j._async_clients]
    assert per_client == [2, 2]                             # round 1 key 1, round 2 key 2
    assert GEMINI not in str(info.value) and "[REDACTED]" in str(info.value)


def test_mcp_bench_doc_states_the_pin_the_judge_request_and_the_credentials():
    doc = _doc("mcp_bench.md")
    for needle in ("`mcp==1.16.0`", "`mcp[cli]>=1.9.0`", "(2025-10-07)", "28/28",
                   "**No temperature** is sent", "`docker run -e NAME`", "`[REDACTED]`",
                   "`set +e`", "`protocol_overrides`", "no variable runs the tasks anyway"):
        assert needle in doc, needle
    assert "temperature 0.0" not in doc and "GBENCH_MCP_BENCH_NO_SUBSET" not in doc
    src = _read("gbench", "runners", "eval_suites", "mcp_bench.py")
    assert "pinned at 0.0" not in src and "no temperature" in src


# --------------------------------------------------------------------------- #
# toolbench: credentials by name
# --------------------------------------------------------------------------- #
def _tb_answer(solved):
    return {"query": "q", "answer": {"final_answer": "S" if solved else "N",
                                     "answer_details": []}}


@pytest.fixture
def tb_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_KEY", SIM_KEY)
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR",
              "GBENCH_TOOLBENCH_REFERENCE", "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_served_model_id", lambda endpoint, fallback: fallback)
    # the bridge network's gateway (after the metadata guard) and the image's records of its
    # tool calls: stubbed, no docker (the relays listen on loopback)
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: "127.0.0.1")
    monkeypatch.setattr(T, "read_server_calls", lambda path: [])
    monkeypatch.setattr(T, "read_observations", lambda path: {})

    async def status(query, answer, sem):
        return "Solved"

    async def pref(query, cand, ref, sem):
        return "cand"

    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    ref_root = tmp_path / "ref"
    (ref_root / T._REFERENCE_DEFAULT).mkdir(parents=True)
    for g in T._GROUPS:
        (ref_root / T._REFERENCE_DEFAULT / f"{g}.json").write_text(
            json.dumps({"1": _tb_answer(False), "2": _tb_answer(False)}))
    monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: str(ref_root))
    monkeypatch.setattr(T, "_group_test_ids",
                        lambda image, groups, limit: {g: ["1", "2"] for g in groups})
    return ref_root


def _tb_container(monkeypatch, behaviour):
    runs = []

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["docker", "run"]:
            runs.append((cmd, kw))
            return behaviour(cmd, kw)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    return runs


def _tb_answers(cmd, kw):
    out = cmd[cmd.index("-v") + 1].split(":")[0]
    os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
    for g in T._GROUPS:
        with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
            json.dump({"1": _tb_answer(True), "2": _tb_answer(True)}, fh)
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_toolbench_run_passes_every_credential_by_name(tb_env, monkeypatch):
    runs = _tb_container(monkeypatch, _tb_answers)
    res = T.run_toolbench("m", "http://x:8000/v1", 2)
    [(cmd, kw)] = runs
    assert _no_secret(" ".join(cmd))
    names = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
    # The container runs the model's tool calls: it gets the Gemini relay's placeholder by name,
    # never the key; GBENCH_TOOLBENCH_SIMULATOR_KEY (set by the fixture) is ignored.
    assert "GEMINI_API_KEY" in names and "GEMINI_API_KEYS" in names
    assert kw["env"]["GEMINI_API_KEY"].startswith("gbench-gemini-relay-")
    assert kw["env"]["GEMINI_API_KEYS"] == kw["env"]["GEMINI_API_KEY"]
    assert GEMINI not in json.dumps(kw["env"])
    assert not any(n.startswith(("TB_SIMULATOR_KEY", "GBENCH_TOOLBENCH_SIMULATOR")) for n in names)
    assert "TB_SIMULATOR_KEY" not in kw["env"] and "TB_SIMULATOR_KEY" not in os.environ
    assert SIM_KEY not in json.dumps(res)
    assert res["sowr"] == 100.0 and res["judge_max_output_tokens"] == 1000 + 8192


def test_toolbench_timeout_and_tail_carry_no_credential(tb_env, monkeypatch):
    def timeout(cmd, kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    _tb_container(monkeypatch, timeout)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*16h timeout") as info:
        T.run_toolbench("m", "http://x:8000/v1", 2)
    assert _no_secret(_exc_text(info.value))

    _tb_container(monkeypatch, lambda cmd, kw: subprocess.CompletedProcess(
        cmd, 1, "", f"simulator api_key: {SIM_KEY}\ngemini {GEMINI}"))
    with pytest.raises(RuntimeError, match=r"rc=1.*\[REDACTED\]") as info:
        T.run_toolbench("m", "http://x:8000/v1", 2)
    assert _no_secret(_exc_text(info.value))


# --------------------------------------------------------------------------- #
# toolbench: SoWR needs a reference answer for every test id the run covers
# --------------------------------------------------------------------------- #
def test_toolbench_reference_dir_missing_a_group_is_a_hard_error_before_the_run(
        tb_env, monkeypatch, tmp_path):
    own = tmp_path / "own"
    (own / T._REFERENCE_DEFAULT).mkdir(parents=True)
    (own / T._REFERENCE_DEFAULT / "G1_instruction.json").write_text(
        json.dumps({"1": _tb_answer(False), "2": _tb_answer(False)}))
    (own / T._REFERENCE_DEFAULT / "G1_tool.json").write_text(json.dumps({"1": _tb_answer(False)}))
    (own / T._REFERENCE_DEFAULT / "G2_category.json").write_text("{}")
    monkeypatch.setenv("GBENCH_TOOLBENCH_REFERENCE_DIR", str(own))
    runs = _tb_container(monkeypatch, _tb_answers)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*do not cover the run") as info:
        T.run_toolbench("m", "http://x:8000/v1", 2)
    msg = str(info.value)
    assert runs == [], "the container never runs without the reference"
    assert str(own / T._REFERENCE_DEFAULT) in msg
    assert "G1_category (no G1_category.json, or one with no answers)" in msg
    assert "G2_category (no G2_category.json, or one with no answers)" in msg
    assert "G1_tool (1 of the 2 test ids the run covers have no reference answer, e.g. 2)" in msg
    assert "G1_instruction" not in msg.split("do not cover the run:")[1]


def test_toolbench_no_sowr_is_a_hard_error_not_a_note(tb_env, monkeypatch):
    async def scores(cand, ref, times, conc, test_ids=None, excluded=None):
        by_group = {g: {"sopr": 1.0, "sowr": None if g == "G3_instruction" else 1.0,
                        "n_candidate": 2} for g in cand}
        return {"by_group": by_group, "sopr": 1.0, "sowr": None, "n_candidate": 12}

    monkeypatch.setattr(T, "_score_groups", scores)
    _tb_container(monkeypatch, _tb_answers)
    # no score is published without SoWR: withheld (null + the reason), never a note beside one
    res = T.run_toolbench("m", "http://x:8000/v1", 2)
    assert res["accuracy"] is None and res["status"] == "failed"
    assert "no SoWR for group(s) ['G3_instruction']" in res["accuracy_withheld_reason"]
    assert "sowr_note" not in _read("gbench", "runners", "eval_suites", "toolbench.py")


def _baselines_zip(path, groups, ids=("11", "12")):
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        for g in groups:
            for qid in ids:
                zf.writestr(f"data_baselines/chatgpt_cot/{g}/{qid}_CoT@1.json", "{}")
    return str(path)


def test_toolbench_conversion_must_cover_every_test_id_before_it_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    zip_path = _baselines_zip(tmp_path / "b.zip", ["G1_tool"])
    monkeypatch.setattr(T, "_download_baselines", lambda: zip_path)
    monkeypatch.setattr(T, "_group_test_ids",
                        lambda image, groups, limit: {g: ["11", "12", "13"] for g in groups})
    converted = []

    def fake_run(cmd, **kw):
        mount = cmd[cmd.index("-v") + 1].split(":")[0]
        g = cmd[cmd.index("--answer_dir") + 1].rsplit("/", 1)[1]
        files = sorted(os.listdir(os.path.join(mount, "in", g)))
        converted.append(g)
        with open(os.path.join(mount, f"{g}.json"), "w") as fh:
            json.dump({f.split("_")[0]: {"query": f, "answer": {}} for f in files}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*left 1 of the group's 3 test ids "
                                           "without an answer \\(e.g. 13\\); nothing was cached"):
        T._baseline_reference_root("chatgpt_cot", ["G1_tool"], "img")
    cache = tmp_path / "cache" / "toolbench_baselines" / T.BASELINES_REVISION[:12] / "converted"
    assert not (cache / "chatgpt_cot" / "G1_tool.json").exists()


def test_toolbench_short_cached_reference_is_converted_again(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    cache = tmp_path / "cache" / "toolbench_baselines" / T.BASELINES_REVISION[:12] / "converted"
    (cache / "chatgpt_cot").mkdir(parents=True)
    (cache / "chatgpt_cot" / "G1_tool.json").write_text("{}")          # an empty entry
    (cache / "chatgpt_cot" / "G3_instruction.json").write_text(json.dumps(
        {"11": {"query": "q", "answer": {}}, "12": {"query": "q", "answer": {}}}))
    zip_path = _baselines_zip(tmp_path / "b.zip", ["G1_tool", "G3_instruction"])
    monkeypatch.setattr(T, "_download_baselines", lambda: zip_path)
    monkeypatch.setattr(T, "_group_test_ids",
                        lambda image, groups, limit: {g: ["11", "12"] for g in groups})
    converted = []

    def fake_run(cmd, **kw):
        mount = cmd[cmd.index("-v") + 1].split(":")[0]
        g = cmd[cmd.index("--answer_dir") + 1].rsplit("/", 1)[1]
        converted.append(g)
        with open(os.path.join(mount, f"{g}.json"), "w") as fh:
            json.dump({q: {"query": "q", "answer": {}} for q in ("11", "12")}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    root = T._baseline_reference_root("chatgpt_cot", ["G1_tool", "G3_instruction"], "img")
    assert converted == ["G1_tool"], "only the short entry is converted again"
    ref = T._load_groups(os.path.join(root, "chatgpt_cot"), ["G1_tool"])
    assert sorted(ref["G1_tool"]) == ["11", "12"]


# --------------------------------------------------------------------------- #
# toolbench: ToolEval's max_tokens reaches the judge; cut replies are counted
# --------------------------------------------------------------------------- #
def test_toolbench_judge_config_sends_tooleval_max_tokens():
    for func in ("check_answer_status", "parse_answer_status", "select_better_answer"):
        cfg = T._judge_generation_config(func, True)
        assert T._JUDGE_MAX_TOKENS == 1000
        # ToolEval's 1000 for the reply plus the Gemini thinking allowance
        assert cfg.max_output_tokens == 1000 + 8192 == T._JUDGE_OUTPUT_LIMIT
        assert cfg.temperature == 0.2


def test_toolbench_cascade_counts_replies_cut_at_max_tokens(monkeypatch):
    from google.genai import types as gt

    def reply(finish, call=None):
        parts = [gt.Part(function_call=gt.FunctionCall(name=call, args={"answer_status": "Solved",
                                                                          "reason": "r"}))] \
            if call else [gt.Part(text="thinking...")]
        return gt.GenerateContentResponse(candidates=[gt.Candidate(
            content=gt.Content(role="model", parts=parts), finish_reason=finish)])

    script = {"m1": reply(gt.FinishReason.MAX_TOKENS),
              "m2": reply(gt.FinishReason.STOP, call="check_answer_status")}

    class Models:
        async def generate_content(self, model, contents, config):
            assert config.max_output_tokens == 1000 + 8192
            return script[model]

    client = type("C", (), {"aio": type("A", (), {"models": Models()})()})()
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setattr(T._base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(T._base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(T._base, "judge_cascade", lambda: ["m1", "m2"])
    monkeypatch.setattr(T._base, "_judge_cascade_rounds", lambda: 1)

    async def one():
        verdicts = T._Verdicts()
        T._VERDICTS.set(verdicts)
        cfg = T._judge_generation_config("check_answer_status", True)
        out = await T._tooleval_cascade("prompt", config=cfg)
        return out, verdicts

    (args, model), verdicts = asyncio.run(one())
    assert model == "m2" and json.loads(args)["answer_status"] == "Solved"
    assert verdicts.max_tokens_cut == 1 and verdicts.outages == 0


def test_toolbench_result_reports_the_cut_counts(tb_env, monkeypatch):
    _tb_container(monkeypatch, _tb_answers)
    res = T.run_toolbench("m", "http://x:8000/v1", 2)
    assert res["judge_replies_cut_at_max_tokens_by_group"] == {g: 0 for g in T._GROUPS}
    assert "max_tokens 1000" in res["metric"]


def test_toolbench_doc_states_the_judge_cap_the_reference_rule_and_the_credentials():
    doc = _doc("toolbench.md")
    for needle in ("`max_output_tokens=9192`", "`judge_replies_cut_at_max_tokens_by_group`",
                   "`judge_max_output_tokens`", "missing ids, hard-errors",
                   "a conversion that leaves any test id without an answer is `infra_required`",
                   "`docker run -e NAME`", "`[REDACTED]`"):
        assert needle in doc, needle
    assert "sowr_note" not in doc and "the cascade's own limit applies" not in doc
    src = _read("gbench", "runners", "eval_suites", "toolbench.py")
    assert "the cascade's own limit applies" not in src


# --------------------------------------------------------------------------- #
# mcp_atlas / cimemories: no max_tokens, as their harnesses send none
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sampling", [ATLAS.MCP_ATLAS_SAMPLING, CIM.CIMEMORIES_SAMPLING])
def test_harnesses_without_max_tokens_opt_out_of_the_cap(sampling):
    assert sampling.send_max_tokens is False and sampling.send_temperature is False
    rec = sampling.record(None, "official:server_default")
    assert rec["send_max_tokens"] is False
    assert rec["max_tokens"].startswith("not sent")
    assert "no max_tokens" in sampling.reference or "{model, messages, tools}" in sampling.reference


def test_cimemories_reason_names_window_cuts_not_a_budget(monkeypatch):
    monkeypatch.setattr(CIM, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(CIM, "_load_profiles", lambda: ())
    monkeypatch.setattr(CIM, "_load_cimemories_samples", lambda limit=None: [])
    got = {}

    def fake_suite(**kw):
        got.update(kw)
        return {"sample_traces": [], "temperature": None,
                "temperature_source": "official:server_default"}

    def fake_score(result, profiles, state):
        result["cimemories_report"] = {
            "contexts_generated": 490, "trials_without_response": 0,
            "trials_reveal_judge_outage": 0, "trials_label_outage": 0,
            "reveal_judge_unparsed": 0, "trials_budget_exhausted": 1, "trials_truncated": 0}

    monkeypatch.setattr(CIM, "run_eval_suite", fake_suite)
    monkeypatch.setattr(CIM, "_score", fake_score)
    res = CIM.run_cimemories("m", "http://x", max_output_tokens=512)
    assert got["official_sampling"].send_max_tokens is False
    reason = res["leaderboard_comparable_reason"]
    assert "1 generations that reached the end of the context window" in reason
    assert "--max-output-tokens" not in reason
    doc = _doc("cimemories.md")
    assert "no `max_tokens`" in doc and "`max_output_tokens_note`" in doc
    assert not re.search(r"hit `--max-output-tokens`", doc)


# --------------------------------------------------------------------------- #
# toolbench: the cached server's cache-miss path is upstream's; the simulator is the cascade
# --------------------------------------------------------------------------- #
def test_toolbench_cached_server_runs_upstreams_config():
    ep = _read("docker", "toolbench_entrypoint.sh")
    assert f"toolbench_url: {T.UPSTREAM_REAL_API_PROXY}" in ep
    assert "is_save: true" in ep and "127.0.0.1:0" not in ep
    assert "temperature: 0" in ep and 'api_key: "${SIMULATOR_KEYS}"' in ep
    # the simulator's API is the Gemini API's OpenAI-compatible endpoint behind gbench's Gemini
    # relay (the relay's URL; the key stays with the relay), with no override
    assert 'SIMULATOR_BASE="${RELAY}/v1beta/openai/"' in ep and "api_base: ${SIMULATOR_BASE}" in ep
    assert 'SIMULATOR_KEYS="${GEMINI_API_KEYS:-${GEMINI_API_KEY:-}}"' in ep
    assert "gbench-gemini-relay-[0-9a-f]{48}" in ep
    assert "TB_SIMULATOR_KEY" not in ep and "TB_SIMULATOR_BASE" not in ep
    # the readiness probe waits for an HTTP reply and fails the container when none comes
    assert '[ -n "$code" ] && [ "$code" != "000" ]' in ep and "exit 3" in ep
    assert "|| echo 000" not in ep
    df = _read("docker", "toolbench.Dockerfile")
    assert "RUN python /tmp/toolbench_patch_simulator.py" in df


def test_toolbench_simulator_uses_the_judge_cascade(tb_env, monkeypatch):
    from gbench.runners.eval_suites import base as B
    monkeypatch.delenv("GBENCH_TOOLBENCH_SIMULATOR_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", GEMINI_LIST)
    monkeypatch.delenv("GBENCH_TOOLBENCH_SIMULATOR_KEY")
    runs = _tb_container(monkeypatch, _tb_answers)
    res = T.run_toolbench("m", "http://x:8000/v1", 2)
    cmd = runs[0][0]
    assert f"GBENCH_JUDGE_MODELS={','.join(B.judge_cascade())}" in cmd
    assert f"TB_SIMULATOR_ROUNDS={B._judge_cascade_rounds()}" in cmd
    assert "GEMINI_API_KEYS" in cmd and _no_secret(" ".join(cmd))
    sim = res["simulator"]
    assert sim["models"] == B.judge_cascade() and sim["upstream_model"] == "gpt-4-turbo"
    assert sim["source"] == "gbench judge cascade" and sim["is_save"] is True
    assert sim["real_api_proxy"] == T.UPSTREAM_REAL_API_PROXY
    assert "simulated by the gbench Gemini cascade instead of gpt-4-turbo" in (
        res["leaderboard_comparable_reason"])
    # No knob names another simulator model: GBENCH_TOOLBENCH_SIMULATOR_MODEL is ignored.
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_MODEL", "gemini-x")
    runs = _tb_container(monkeypatch, _tb_answers)
    res = T.run_toolbench("m", "http://x:8000/v1", 2)
    assert f"GBENCH_JUDGE_MODELS={','.join(B.judge_cascade())}" in runs[0][0]
    assert res["simulator"]["source"] == "gbench judge cascade"
    assert "gemini-x" not in res["leaderboard_comparable_reason"]


_UPSTREAM_SIM_BLOCK = (
    "def write_log(request, response, type):\n"
    "    pass\n"
    "@app.post('/virtual')\n"
    "# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))\n"
    "def get_virtual_response(request: Request, info: Info):\n"
    "    real_response = requests.post(CONFIG['toolbench_url'], headers=headers, "
    "data=json.dumps(data))\n"
    "def fake_response_function_chat(api_example, tool_input, api_doc):\n"
    "    system_prompt = {'role': 'system', 'content': 'sim'}\n"
    "    user_prompt = {'role': 'user', 'content': str(tool_input)}\n"
    "    client = OpenAI(\n"
    "        api_key = OPENAI_API_KEY,\n"
    "        base_url = OPENAI_API_BASE,\n"
    "    )\n"
    "    max_retries = 3 \n"
    "    flag = False\n"
    "    for attempt in range(max_retries):\n"
    "        response = client.chat.completions.create(\n"
    "            model = CONFIG['model'],\n"
    "            messages=[system_prompt, user_prompt],\n"
    "            max_tokens = 1024,\n"
    "            temperature=CONFIG['temperature'],\n"
    "            response_format={\"type\": \"json_object\"},\n"
    "        )\n"
    "        result = response.choices[0].message.content\n"
    "        if is_valid_json(result):\n"
    "            flag = True\n"
    "            break\n"
    "    return result if flag else json.dumps({'error': 'Failed to generate fake response', "
    "'response': ''})\n"
    "if __name__ == \"__main__\":\n"
    "    pass\n"
)


def test_toolbench_simulator_patch_cascades_with_upstreams_request(tmp_path, monkeypatch):
    main = tmp_path / "main.py"
    main.write_text(_UPSTREAM_SIM_BLOCK)
    patch = _read("docker", "toolbench_patch_simulator.py").replace("/stb/server/main.py",
                                                                    str(main))
    exec(compile(patch, "toolbench_patch_simulator.py", "exec"), {"__name__": "patch"})
    patched = main.read_text()
    assert "model = CONFIG['model']" not in patched and "_gbench_simulate(" in patched
    calls = []

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, max_retries=None):
            self.key = api_key
            outer = self

            class Completions:
                def create(self, **kw):
                    calls.append((outer.key, kw))
                    if kw["model"] == "m1":
                        raise RuntimeError("429")
                    msg = type("M", (), {"content": '{"error": "", "response": "ok"}'})()
                    return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m1,m2")
    monkeypatch.setenv("TB_SIMULATOR_ROUNDS", "1")
    monkeypatch.setenv("TB_SIMULATOR_BACKOFF", "0")
    import time as _time
    app = type("App", (), {"post": lambda self, path: (lambda fn: fn)})()
    namespace = {"OpenAI": FakeOpenAI, "OPENAI_API_KEY": "key-a;key-b", "app": app,
                 "Request": object, "Info": object, "__name__": "main",
                 "OPENAI_API_BASE": "http://sim.invalid/v1", "CONFIG": {"model": "m0",
                                                                       "temperature": 0},
                 "os": os, "time": _time, "json": json,
                 "is_valid_json": lambda r: bool(r) and r.startswith("{")}
    exec(compile(patched, "main.py", "exec"), namespace)
    out = namespace["fake_response_function_chat"]([], {"a": 1}, {})
    assert json.loads(out)["response"] == "ok"
    assert [(k, kw["model"]) for k, kw in calls] == [("key-a", "m1"), ("key-a", "m2")]
    assert set(calls[1][1]) == {"model", "messages", "max_tokens", "temperature",
                                "response_format"}
    # upstream's 1024 for the JSON plus the Gemini thinking allowance (gemini_budget)
    assert calls[1][1]["max_tokens"] == 1024 + 8192 and calls[1][1]["temperature"] == 0
    assert calls[1][1]["response_format"] == {"type": "json_object"}


def test_toolbench_prerequisites_require_the_real_api_proxy(monkeypatch):
    import urllib.error
    import urllib.request

    def refused(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    ok, why = T.real_api_proxy_reachable(timeout_s=1)
    assert ok is False and T.UPSTREAM_REAL_API_PROXY in why and "server error" in why

    def answered(req, timeout=None):
        assert req.get_method() == "POST" and req.full_url == T.UPSTREAM_REAL_API_PROXY
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", answered)
    assert T.real_api_proxy_reachable(timeout_s=1) == (True, "")


def test_toolbench_doc_states_the_cache_miss_path():
    doc = _doc("toolbench.md")
    for needle in ("`toolbench_url: http://8.130.32.149:8080/rapidapi`", "`is_save: true`",
                   "the gbench Gemini judge cascade in place of the model",
                   "`docker/toolbench_patch_simulator.py`", "The proxy must be reachable"):
        assert needle in doc, needle
    assert "`gemini-2.5-flash` | cache-miss simulator model" not in doc


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
