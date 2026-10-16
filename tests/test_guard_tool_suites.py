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

"""The cloud metadata guard in the tool suites, and their harness-error accounting.

* mcp_atlas, mcp_bench and wildclawbench call container_guard.require_metadata_blocked before they
  start a container that runs model-controlled code with network access, on the network that
  container uses (wildclawbench and mcp_bench now default to the docker bridge); a guard that
  refuses stops the run before any such container starts. gaia2 and toolbench run no
  model-controlled code and do not call it.
* mcp_atlas: a URL the model wrote for exa's or oxylabs' page tools (malformed, non-http or
  non-public included) goes unchanged to the release's server in the credentialed-servers
  container, whose provider fetches pages from its own infrastructure; its reply is the model's
  outcome, never a Gemini lookup and never a harness error.
* wildclawbench: proxy events carry the requesting container, the grader's judge calls are told
  apart from the agent's (the agent's own bad requests are no event), a model endpoint failure the
  agent's retry got past is no harness error, a grade() cap hit without a judge call is the model's
  outcome, and missing provisioned inputs are harness errors.
* gaia2: judge failures, endpoint failures and raised runs are left out of the success rates.

Docker, the model, the Gemini API and ARE are stubbed: no container starts, nothing contacts
169.254.169.254, and every file is written under tmp_path.
"""

import asyncio
import http.client
import json
import os
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import gaia2_run as GRUN  # noqa: E402
import wildclawbench_cascade_judge as J  # noqa: E402
import wildclawbench_run as RUN  # noqa: E402


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


class GuardRecorder:
    """Stands in for container_guard.require_metadata_blocked: records each call; refuses when
    `refuse` is set (as the real guard does when the metadata server answers)."""

    def __init__(self, refuse=False):
        self.calls = []
        self.resolve = []
        self.refuse = refuse

    def __call__(self, eval_name, image, network, docs_url=None, *, resolve=()):
        self.calls.append((eval_name, image, network, docs_url))
        self.resolve.append(list(resolve))
        if self.refuse:
            raise RuntimeError(f"[INFRA REQUIRED] {eval_name}: metadata server reachable "
                               f"({CG.FIREWALL_COMMAND})")
        return CG.UNREACHABLE


@pytest.fixture(autouse=True)
def _fresh_guard_cache():
    CG.reset_verdicts()
    yield
    CG.reset_verdicts()


def _guard(monkeypatch, refuse=False):
    rec = GuardRecorder(refuse)
    monkeypatch.setattr(CG, "require_metadata_blocked", rec)
    return rec


# --------------------------------------------------------------------------- #
# mcp_atlas: the guard before the environment container
# --------------------------------------------------------------------------- #
class _Docker:
    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, "1\n", "")
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        return subprocess.CompletedProcess(cmd, 0, "id\n", "")


def _atlas_credentials(monkeypatch):
    """Every credentialed server's variables set (mcp_atlas starts nothing without them)."""
    for server in M.CREDENTIALED_SERVERS:
        for var in M.ENVIRONMENT_SERVERS[server]:
            if var != "MONGODB_CONNECTION_STRING":
                monkeypatch.setenv(var, f"{var.lower()}-value-1234")


def test_mcp_atlas_checks_its_own_network_before_mongodb_and_the_environment(monkeypatch):
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    _atlas_credentials(monkeypatch)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    order = []
    rec = GuardRecorder()

    def guard(*a, **k):
        order.append(len(docker.cmds))
        return rec(*a, **k)
    monkeypatch.setattr(CG, "require_metadata_blocked", guard)
    sb = M._start_container("gbench-mcp-atlas:f24ba3fb")
    # the environment's network and the credentialed servers' network, each checked right
    # after `docker network create`, before the MongoDB and the two agent-environment containers
    assert rec.calls == [("mcp_atlas", "gbench-mcp-atlas:f24ba3fb", sb.network, M.DOCS_URL),
                         ("mcp_atlas", "gbench-mcp-atlas:f24ba3fb", sb.cred_network, M.DOCS_URL)]
    assert order == [1, 2]
    assert docker.cmds[0][:4] == ["docker", "network", "create", sb.network]
    assert docker.cmds[1][:4] == ["docker", "network", "create", sb.cred_network]
    assert docker.cmds[2][:3] == ["docker", "run", "-d"]


def test_mcp_atlas_operator_mongodb_checks_both_run_networks(monkeypatch):
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://db.example:27017/")
    _atlas_credentials(monkeypatch)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    rec = _guard(monkeypatch)
    sb = M._start_container("gbench-mcp-atlas:f24ba3fb")
    assert rec.calls == [("mcp_atlas", "gbench-mcp-atlas:f24ba3fb", sb.network, M.DOCS_URL),
                         ("mcp_atlas", "gbench-mcp-atlas:f24ba3fb", sb.cred_network, M.DOCS_URL)]
    assert sb.mongo_container is None and None not in (sb.network, sb.cred_network)


def test_mcp_atlas_refused_guard_starts_nothing_and_removes_the_network(monkeypatch):
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    _atlas_credentials(monkeypatch)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    _guard(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match="metadata server reachable"):
        M._start_container("gbench-mcp-atlas:f24ba3fb")
    assert not any(c[:2] == ["docker", "run"] for c in docker.cmds)
    created = docker.cmds[0]
    assert created[:3] == ["docker", "network", "create"]
    assert docker.cmds[-1] == ["docker", "network", "rm", created[3]]


def test_mcp_atlas_refused_guard_with_operator_mongodb_starts_nothing(monkeypatch):
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://db.example:27017/")
    _atlas_credentials(monkeypatch)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    _guard(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match="metadata server reachable"):
        M._start_container("gbench-mcp-atlas:f24ba3fb")
    assert not any(c[:2] == ["docker", "run"] for c in docker.cmds)
    assert [c[:3] for c in docker.cmds] == [["docker", "network", "create"],
                                            ["docker", "network", "rm"]]


def test_mcp_atlas_real_guard_refuses_a_reachable_metadata_server(monkeypatch):
    """The real guard, docker stubbed: the probe reports the metadata server reachable, so the
    run stops naming the firewall rule and the environment never starts."""
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://db.example:27017/")
    _atlas_credentials(monkeypatch)
    cmds = []

    def fake(cmd, **kw):
        cmds.append(list(cmd))
        if CG.PROBE_LABEL in cmd:
            return subprocess.CompletedProcess(cmd, 0, "GBENCH_METADATA_PROBE=reachable\n", "")
        return subprocess.CompletedProcess(cmd, 0, "id\n", "")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as ei:
        M._start_container("gbench-mcp-atlas:f24ba3fb")
    assert CG.FIREWALL_COMMAND in str(ei.value)
    network = cmds[0][3]
    [probe] = [c for c in cmds if CG.PROBE_LABEL in c]
    assert probe[probe.index("--network") + 1] == network and "--pull" in probe
    assert "gbench-mcp-atlas:f24ba3fb" in probe
    assert not any(c[:3] == ["docker", "run", "-d"] for c in cmds)


# --------------------------------------------------------------------------- #
# mcp_atlas: exa / oxylabs page tools run on the release's servers (the model's outcome)
# --------------------------------------------------------------------------- #
@pytest.fixture
def cred_server(monkeypatch):
    """The credentialed-servers container's /call-tool, stood in: records each body and answers
    as the release's server would for a URL its provider cannot fetch."""
    sent = []

    async def exec_http(container, method, path, timeout, body=None):
        sent.append((container, method, path, body))
        return 200, json.dumps([{"type": "text", "text": "provider says: cannot fetch"}])

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "_ground", lambda *a, **k: pytest.fail("a page tool reached Gemini"))
    return sent


@pytest.mark.parametrize("tool,args", [
    ("exa_web_fetch_exa", {"urls": ["not a url", "http://127.0.0.1:8080/admin"]}),
    ("exa_web_fetch_exa", {"urls": ["file:///etc/passwd", "https://ok.example/"]}),
    ("oxylabs_universal_scraper", {"url": "not a url"}),
    ("oxylabs_universal_scraper", {"url": "http://169.254.169.254/computeMetadata/v1/"}),
    ("oxylabs_amazon_product_scraper", {"query": "B0TEST1234", "domain": "attacker.example/p?"}),
])
def test_page_tool_urls_go_unchanged_to_the_releases_server(cred_server, tool, args):
    sb = M._Sandbox("http://127.0.0.1:9", cred_container="cred",
                    cred_servers=M.CREDENTIALED_SERVERS)
    token = M._ACTIVE_SANDBOX.set(sb)
    try:
        out = asyncio.run(M.execute_mcp_tool(tool, args, 3))
    finally:
        M._ACTIVE_SANDBOX.reset(token)
    assert cred_server == [("cred", "POST", "/call-tool", {"tool_name": tool, "tool_args": args})]
    assert out == "provider says: cannot fetch"
    assert sb.harness_error_samples == {} and sb.gemini_outages == 0 and sb.search_calls == 0
    assert sb.page_tool_calls == {tool: 1}


def test_the_gemini_url_check_and_lookup_are_gone():
    for name in ("_public_url_problem", "URL_INVALID", "URL_NOT_PUBLIC", "url_context_page",
                 "_url_context_request", "GEMINI_PAGE_TOOLS"):
        assert not hasattr(M, name), name
    src = _read("gbench/runners/eval_suites/mcp_atlas.py")
    assert "refused_by_url_check" not in src and "url_context" not in src


# --------------------------------------------------------------------------- #
# mcp_bench: bridge network, endpoint at the gateway, the guard first
# --------------------------------------------------------------------------- #
_CONNECTED = {f"S{i}": {"status": "success", "tools": 3, "error": ""} for i in range(28)}
_PROBE_OK = {"servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
             "connections": _CONNECTED}


@pytest.fixture
def mb(monkeypatch):
    for k in ("GBENCH_MCP_BENCH_NETWORK", "GBENCH_MCP_BENCH_DISTRACTION", "GBENCH_SHARD",
              "GBENCH_MCP_BENCH_DISABLE_STABILITY", "GBENCH_MCP_BENCH_ASSUME_NETWORK"):
        monkeypatch.delenv(k, raising=False)
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTKEY0000000000000000000000000000")
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "server-key-value-1234")
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    events = []
    monkeypatch.setattr(MB, "provisioning_probe",
                        lambda image, relay_env=None: events.append(("probe", MB._network()))
                        or _PROBE_OK)
    state = {"runs": [], "events": events}
    # the credentialed-servers container (tests/test_mcp_wildclawbench_credentials_and_cascade.py
    # covers it): a stand-in
    cred = types.SimpleNamespace(
        client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                            "MCP_CREDENTIALED_TOKEN": "t0k"},
        stop=lambda: None, record=lambda: {})
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: events.append(("cred", network)) or cred)
    # the judge's Gemini relay (tests/test_mcp_credential_isolation_and_dns.py covers it): a
    # stand-in, docker is stubbed
    relay = types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48,
                            "GEMINI_API_KEYS": "gbench-gemini-relay-" + "0" * 48,
                            "GEMINI_OPENAI_BASE_URL": "http://172.17.0.1:9/v1beta/openai",
                            "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:9"},
        stop=lambda: None, record=lambda: {})
    monkeypatch.setattr(MB, "_start_judge_relay",
                        lambda network: events.append(("relay", network)) or relay)

    def fake_run(cmd, **kw):
        if cmd[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(state.get("ipam", [])), "")
        if cmd[:2] == ["docker", "run"]:
            state["runs"].append((list(cmd), kw))
            events.append(("run", cmd[cmd.index("--network") + 1]))
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            metrics = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
                       "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
                       "tool_selection_score": 5.0,
                       "planning_effectiveness_and_efficiency_score": 5.0}
            with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                json.dump(metrics, fh)
            with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                json.dump({"filtered": True, "unavailable_servers": [], "tasks_kept": 104}, fh)
            with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": [
                    {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": []}]}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    return state


def test_mcp_bench_runs_on_the_bridge_after_the_guard(mb, monkeypatch):
    rec = GuardRecorder()

    def guard(*a, **k):
        mb["events"].append(("guard", a[2]))
        return rec(*a, **k)
    monkeypatch.setattr(CG, "require_metadata_blocked", guard)
    res = MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    assert rec.calls == [("mcp_bench", MB._image(), "bridge", MB.DOCS_URL)]
    assert rec.resolve == [MB.server_dns_names()]      # the servers' API hosts resolve there
    assert mb["events"] == [("guard", "bridge"), ("cred", "bridge"), ("probe", "bridge"),
                            ("relay", "bridge"), ("run", "bridge")]
    [(cmd, kw)] = mb["runs"]
    assert "host" not in cmd
    assert kw["env"]["MCP_ENDPOINT"] == "http://172.17.0.1:8123/v1"
    assert res["docker_network"] == "bridge"


def test_mcp_bench_refused_guard_starts_no_container(mb, monkeypatch):
    _guard(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match="metadata server reachable"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    assert mb["runs"] == [] and not any(e[0] in ("probe", "cred") for e in mb["events"])


def test_mcp_bench_host_network_is_checked_as_host(mb, monkeypatch):
    monkeypatch.setenv("GBENCH_MCP_BENCH_NETWORK", "host")
    rec = _guard(monkeypatch)
    MB.run_mcp_bench("m", "http://127.0.0.1:8123")
    assert rec.calls[0][2] == "host"
    [(cmd, kw)] = mb["runs"]
    assert cmd[cmd.index("--network") + 1] == "host"
    assert kw["env"]["MCP_ENDPOINT"] == "http://127.0.0.1:8123/v1"


def test_mcp_bench_probe_uses_the_runs_network(monkeypatch):
    monkeypatch.setenv("GBENCH_MCP_BENCH_NETWORK", "evalnet")
    seen = []

    def fake(cmd, **kw):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "MCP_BENCH_PROBE " + json.dumps(_PROBE_OK), "")
    monkeypatch.setattr(MB.subprocess, "run", fake)
    MB.provisioning_probe("img")
    assert seen[0][seen[0].index("--network") + 1] == "evalnet"


def _inspect(monkeypatch, ipam):
    monkeypatch.setattr(MB.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0 if ipam is not None else 1, json.dumps(ipam or []), ""))


def test_container_endpoint(monkeypatch):
    _inspect(monkeypatch, [{"Subnet": "10.9.0.0/16", "Gateway": "10.9.0.1"}])
    assert MB.container_endpoint("http://localhost:8000", "evalnet") == "http://10.9.0.1:8000/v1"
    assert MB.container_endpoint("http://u:p@127.0.0.1:8000/v1", "evalnet") == \
        "http://u:p@10.9.0.1:8000/v1"
    assert MB.container_endpoint("https://api.example.com/v1", "evalnet") == \
        "https://api.example.com/v1"
    assert MB.container_endpoint("http://127.0.0.1:8000", "host") == "http://127.0.0.1:8000/v1"
    _inspect(monkeypatch, [{"Gateway": "fd00::1"}])
    assert MB.container_endpoint("http://[::1]:8000/v1", "v6") == "http://[fd00::1]:8000/v1"
    _inspect(monkeypatch, None)
    assert MB.container_endpoint("http://127.0.0.1:8000", "bridge") == "http://172.17.0.1:8000/v1"
    with pytest.raises(RuntimeError, match="reports no gateway address"):
        MB.container_endpoint("http://127.0.0.1:8000", "evalnet")


# --------------------------------------------------------------------------- #
# wildclawbench: bridge by default, the guard on the task network with the task image
# --------------------------------------------------------------------------- #
def _run_wildclaw(monkeypatch, tmp_path, env=None, summary_text=None, stderr=""):
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wd"))
    os.makedirs(tmp_path / "wd", exist_ok=True)
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_JUDGE_PORT", "GBENCH_WILDCLAWBENCH_TASK_HOST", "GBENCH_SHARD",
              "GBENCH_WILDCLAWBENCH_PARALLEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    cap = {"runs": []}

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"]:
            cap["runs"].append(list(cmd))
            wd = next(x.split("=", 1)[1] for x in cmd if str(x).startswith("WILDCLAW_WORKDIR="))
            with open(os.path.join(wd, "wildclawbench_summary.json"), "w") as f:
                if summary_text is not None:
                    f.write(summary_text)
                else:
                    json.dump({"n_tasks": 60, "n_scored": 60, "global_mean": 0.5,
                               "recovered_model_endpoint_events": 2,
                               "missing_inputs": {}}, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    result = W.run_wildclawbench("m", "http://127.0.0.1:9123/v1", concurrency=4)
    return result, cap


def test_wildclaw_default_bridge_is_checked_with_the_task_image(monkeypatch, tmp_path):
    rec = _guard(monkeypatch)
    result, cap = _run_wildclaw(monkeypatch, tmp_path)
    assert rec.calls == [("wildclawbench", W._task_image(), "bridge", W.DOCS_URL)]
    [cmd] = cap["runs"]
    assert "WILDCLAW_TASK_NETWORK=bridge" in cmd
    assert "WILDCLAW_PARALLEL=4" in cmd                     # bridge runs tasks in parallel
    # not rewritten: the temperature relay on 127.0.0.1 relays to it
    url = next(x.split("=", 1)[1] for x in cmd if x.startswith("GBENCH_MODEL_BASE_URL="))
    assert url.startswith("http://127.0.0.1:")
    assert result["model_temperature_relay"]["relays_to"] == "http://127.0.0.1:9123/v1"
    assert "WILDCLAW_JUDGE_HOST=172.17.0.1" in cmd
    assert result["task_network"] == "bridge"
    assert result["recovered_model_endpoint_events"] == 2
    assert "2 model endpoint failure(s) a later relay call of the same task got past" in \
        result["leaderboard_comparable_reason"]


def test_wildclaw_unreadable_summary_is_a_harness_error_with_the_orchestrator_tail(
        monkeypatch, tmp_path):
    """2026-10-10: a full disk left the summary empty and the suite failed with a bare
    'Expecting value: line 1 column 1 (char 0)'."""
    _guard(monkeypatch)
    with pytest.raises(RuntimeError) as info:
        _run_wildclaw(monkeypatch, tmp_path, summary_text="",
                      stderr="OSError: [Errno 28] No space left on device")
    msg = str(info.value)
    assert "[INFRA REQUIRED]" in msg and "summary is not readable JSON" in msg
    assert "not a 0%" in msg and "No space left on device" in msg
    with pytest.raises(RuntimeError, match="summary is not readable JSON.*not an object"):
        _run_wildclaw(monkeypatch, tmp_path, summary_text="[]")


def test_wildclaw_refused_guard_starts_no_orchestrator(monkeypatch, tmp_path):
    _guard(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match="metadata server reachable"):
        _run_wildclaw(monkeypatch, tmp_path)


def test_wildclaw_host_network_is_checked_as_host(monkeypatch, tmp_path):
    rec = _guard(monkeypatch)
    _result, cap = _run_wildclaw(monkeypatch, tmp_path,
                                 env={"GBENCH_WILDCLAWBENCH_TASK_NETWORK": "host"})
    assert rec.calls[0][2] == "host"
    assert "WILDCLAW_PARALLEL=1" in cap["runs"][0]


def test_wildclaw_real_guard_host_message_names_the_bridge(monkeypatch, tmp_path):
    def fake(cmd, *a, **k):
        if CG.PROBE_LABEL in cmd:
            return subprocess.CompletedProcess(cmd, 0, "GBENCH_METADATA_PROBE=reachable\n", "")
        raise AssertionError(f"no other docker call expected: {cmd}")
    monkeypatch.setattr(subprocess, "run", fake)
    harness = tmp_path / "h"
    harness.mkdir()
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "host")
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_JUDGE_PORT", raising=False)
    with pytest.raises(RuntimeError) as ei:
        W.run_wildclawbench("m", "http://127.0.0.1:9123/v1")
    msg = str(ei.value)
    assert "DOCKER-USER chain does not filter the host network" in msg
    assert CG.FIREWALL_COMMAND in msg and "bridge or user-defined docker network" in msg


# --------------------------------------------------------------------------- #
# gaia2 and toolbench: their containers run the model's tool calls, so they are guarded
# --------------------------------------------------------------------------- #
def test_gaia2_and_toolbench_call_the_guard_before_their_bridge_containers():
    for suite in ("gaia2", "toolbench"):
        src = _read(f"gbench/runners/eval_suites/{suite}.py")
        assert "container_guard.require_metadata_blocked(" in src
        assert 'NETWORK = "bridge"' in src and '"--network", "host"' not in src
        doc = _flat(_read(f"docs/evals/{suite}.md"))
        assert "`container_guard.require_metadata_blocked`" in doc
        assert "**`--network bridge`**" in doc and "No model-controlled code runs in" not in doc


# --------------------------------------------------------------------------- #
# wildclawbench proxy: grader vs agent, client addresses, recovery
# --------------------------------------------------------------------------- #
@pytest.fixture
def proxy(monkeypatch, tmp_path):
    wd = tmp_path / "proxy_wd"
    wd.mkdir()
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(wd))
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTKEY0000000000000000000000000000")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-test")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "2")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "0")
    monkeypatch.setenv(J.GRADER_KEY_ENV, "gbench-grader-abc123")
    J.reset_usage()
    J._RELAY_FAILED_CLIENTS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), J._Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield types.SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}", wd=wd)
    server.shutdown()
    server.server_close()


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _chat(url, key):
    req = urllib.request.Request(url + "/v1/chat/completions", method="POST",
                                 data=json.dumps({"messages": [{"role": "user",
                                                                "content": "q"}]}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _gemini_answers(monkeypatch, code=None, content="ok"):
    def post(url, payload, headers, timeout):
        if code is not None:
            raise urllib.error.HTTPError(url, code, "x", {}, None)
        return {"choices": [{"message": {"content": content}}]}
    monkeypatch.setattr(J, "_post_json", post)


def test_agent_request_refused_by_gemini_is_its_own_outcome(proxy, monkeypatch):
    _gemini_answers(monkeypatch, code=400)
    status, body = _chat(proxy.url, "gbench-cascade")
    assert status == 400 and body["error"]["type"] == "invalid_request_error"
    assert _lines(proxy.wd / J.HARNESS_EVENTS) == []
    assert _lines(proxy.wd / J.GRADER_CALLS) == []


def test_grader_request_refused_is_a_judge_event_with_its_client(proxy, monkeypatch):
    _gemini_answers(monkeypatch, code=400)
    status, _body = _chat(proxy.url, "gbench-grader-abc123")
    assert status == 503
    [ev] = _lines(proxy.wd / J.HARNESS_EVENTS)
    assert ev["kind"] == "judge" and ev["client"] == "127.0.0.1"
    [call] = _lines(proxy.wd / J.GRADER_CALLS)
    assert call["client"] == "127.0.0.1" and call["ok"] is False
    assert call["end"] >= call["start"]


def test_agent_service_failure_is_an_agent_helper_event(proxy, monkeypatch):
    _gemini_answers(monkeypatch, code=503)
    status, _body = _chat(proxy.url, "gbench-cascade")
    assert status == 503
    [ev] = _lines(proxy.wd / J.HARNESS_EVENTS)
    assert ev["kind"] == "agent_helper" and ev["client"] == "127.0.0.1"


def test_agent_gets_an_empty_reply_the_grader_does_not(proxy, monkeypatch):
    _gemini_answers(monkeypatch, content="")
    status, body = _chat(proxy.url, "gbench-cascade")
    assert status == 200 and body["choices"][0]["message"]["content"] == ""
    assert _lines(proxy.wd / J.HARNESS_EVENTS) == []
    status, _body = _chat(proxy.url, "gbench-grader-abc123")
    assert status == 503
    assert [e["kind"] for e in _lines(proxy.wd / J.HARNESS_EVENTS)] == ["judge"]
    [call] = _lines(proxy.wd / J.GRADER_CALLS)
    assert call["ok"] is False


def test_grader_key_must_match_the_runs_key(monkeypatch):
    monkeypatch.setenv(J.GRADER_KEY_ENV, "gbench-grader-abc123")
    assert J.is_grader_key("Bearer gbench-grader-abc123")
    assert not J.is_grader_key("Bearer gbench-cascade")
    assert not J.is_grader_key("Bearer gbench-grader-other")
    monkeypatch.delenv(J.GRADER_KEY_ENV)
    assert not J.is_grader_key("Bearer ")


class _Endpoint(BaseHTTPRequestHandler):
    codes = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        code = _Endpoint.codes.pop(0)
        body = json.dumps({"ok": code}).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_relay_records_a_recovery_after_a_failure(proxy, monkeypatch):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("GBENCH_MODEL_BASE_URL",
                           f"http://127.0.0.1:{upstream.server_address[1]}/v1")
        _Endpoint.codes = [503, 200, 200]
        for _ in range(3):
            conn = http.client.HTTPConnection("127.0.0.1", int(proxy.url.rsplit(":", 1)[1]),
                                              timeout=15)
            conn.request("POST", "/model/v1/chat/completions", body=b"{}",
                         headers={"Content-Type": "application/json"})
            conn.getresponse().read()
            conn.close()
    finally:
        upstream.shutdown()
        upstream.server_close()
    events = _lines(proxy.wd / J.HARNESS_EVENTS)
    assert [e["kind"] for e in events] == ["model_endpoint", "model_endpoint_recovered"]
    assert all(e["client"] == "127.0.0.1" for e in events)


# --------------------------------------------------------------------------- #
# wildclawbench launcher: addresses, the grader key, grading windows
# --------------------------------------------------------------------------- #
def test_task_containers_default_to_the_bridge_and_record_their_address(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.delenv("WILDCLAW_TASK_NETWORK", raising=False)
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://172.17.0.1:18795")
    monkeypatch.setenv(RUN.GRADER_KEY_ENV, "gbench-grader-xyz")
    seen = []

    def fake(cmd, *a, **k):
        seen.append(list(cmd))
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, "172.17.0.5  \n", "")
        if cmd[:2] == ["docker", "exec"] and RUN.GRADE_RUNNER in cmd:
            if k.get("timeout"):
                raise subprocess.TimeoutExpired(cmd, k["timeout"])
            return subprocess.CompletedProcess(cmd, 0, "{}", "")
        return subprocess.CompletedProcess(cmd, 0, "cid\n", "")
    docker_utils = types.SimpleNamespace(subprocess=types.SimpleNamespace(run=fake))
    RUN._patch_task_containers(docker_utils)

    def run_single_task(task, *a, **k):
        docker_utils.subprocess.run(["docker", "run", "-d", "--name", "t1", "img"])
        docker_utils.subprocess.run(["docker", "exec", "-e", "OPENROUTER_API_KEY=gbench-cascade",
                                     "t1", "python3", RUN.GRADE_RUNNER])
        with pytest.raises(subprocess.TimeoutExpired):
            docker_utils.subprocess.run(["docker", "exec", "-e",
                                         "OPENROUTER_API_KEY=gbench-cascade", "t1", "python3",
                                         RUN.GRADE_RUNNER], timeout=120)
        return {"task_id": "t1", "error": None}
    rb = types.SimpleNamespace(run_single_task=run_single_task)
    RUN._track_task_windows(rb)
    rb.run_single_task({"task_id": "01_Productivity_Flow_task_1"})
    start = next(c for c in seen if c[:3] == ["docker", "run", "-d"])
    assert start[3:5] == ["--network", "bridge"]
    grades = [c for c in seen if c[:2] == ["docker", "exec"] and RUN.GRADE_RUNNER in c]
    assert all("OPENROUTER_API_KEY=gbench-grader-xyz" in c for c in grades)
    assert not any("OPENROUTER_API_KEY=gbench-cascade" in c for c in grades)
    [rec] = [json.loads(x) for x in (tmp_path / RUN.TASK_WINDOWS).read_text().splitlines()]
    assert rec["ips"] == ["172.17.0.5"]
    assert [g["timed_out"] for g in rec["grading"]] == [False, True]


def _harness(tmp_path, tasks):
    root = tmp_path / "h"
    for tid, (workspace, checks) in tasks.items():
        cat = tid.split("_task_")[0]
        d = root / "tasks" / cat
        d.mkdir(parents=True, exist_ok=True)
        body = (f"---\nid: {tid}\nname: X\ncategory: {cat}\ntimeout_seconds: 60\n"
                f"modality: pure-text\n---\n\n## Prompt\n\nDo it.\n")
        if workspace is not None:
            body += f"\n## Workspace Path\n\n```\n{workspace}\n```\n"
        if checks is not None:
            body += f"\n## Automated Checks\n\n```python\n{checks}\n```\n"
        (d / f"{tid}_x.md").write_text(body, encoding="utf-8")
    return str(root)


_GT_CHECKS = ('def grade(transcript, workspace_path):\n'
              '    gt_file = Path("/tmp_workspace") / "gt" / "ground_truth.json"\n'
              '    img = base / "gt" / "gt.png"\n'
              '    return {"overall_score": 1.0}')


def test_missing_inputs(monkeypatch, tmp_path):
    root = _harness(tmp_path, {
        "01_Productivity_Flow_task_1": ("workspace/01/task_1", _GT_CHECKS),
        "01_Productivity_Flow_task_2": ("workspace/01/task_2", _GT_CHECKS),
        "01_Productivity_Flow_task_3": ("workspace/01/task_3", _GT_CHECKS),
        "01_Productivity_Flow_task_4": ("workspace/01/task_4", "def grade(**k):\n    return {}"),
        "01_Productivity_Flow_task_5": (None, _GT_CHECKS),
    })
    monkeypatch.setattr(RUN, "HARNESS_DIR", root)
    ws = os.path.join(root, "workspace", "01")
    os.makedirs(os.path.join(ws, "task_2", "gt"))                    # gt/ empty
    os.makedirs(os.path.join(ws, "task_3", "gt"))
    open(os.path.join(ws, "task_3", "gt", "ground_truth.json"), "w").close()   # gt.png missing
    os.makedirs(os.path.join(ws, "task_4"))                           # no gt use
    index = RUN._task_index()
    got = RUN.find_missing_inputs(index, sorted(index))
    assert got["01_Productivity_Flow_task_1"] == [
        f"workspace directory {os.path.join(ws, 'task_1')}"]
    assert got["01_Productivity_Flow_task_2"] == [
        f"ground-truth directory {os.path.join(ws, 'task_2', 'gt')} (missing or empty)"]
    assert got["01_Productivity_Flow_task_3"] == [
        f"ground-truth file {os.path.join(ws, 'task_3', 'gt', 'gt.png')}"]
    assert "01_Productivity_Flow_task_4" not in got and "01_Productivity_Flow_task_5" not in got
    open(os.path.join(ws, "task_3", "gt", "gt.png"), "w").close()
    assert RUN.missing_inputs(index["01_Productivity_Flow_task_3"]["path"]) == []


def _score(out_root, tid, score):
    cat = tid.split("_task_")[0]
    d = os.path.join(out_root, "openclaw", cat, tid, "m_20260101_0000_abc123")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "score.json"), "w") as f:
        json.dump(score, f)


def _jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    monkeypatch.delenv("WILDCLAW_LIMIT", raising=False)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    return wd


def test_events_go_to_the_task_whose_container_made_the_request(launcher, monkeypatch, tmp_path):
    tids = ["03_Social_Interaction_task_1", "03_Social_Interaction_task_2"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {t: (None, None) for t in tids}))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 1.0})
    # two tasks in parallel, both windows hold every event
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 0.0, "end": 100.0, "ips": ["172.17.0.2"]},
        {"task_id": tids[1], "start": 0.0, "end": 100.0, "ips": ["172.17.0.3"]}])
    _jsonl(launcher / RUN.HARNESS_EVENTS, [
        {"time": 10.0, "kind": "agent_web_search", "client": "172.17.0.3"},
        {"time": 20.0, "kind": "model_endpoint", "client": "172.17.0.2"},
        {"time": 30.0, "kind": "model_endpoint_recovered", "client": "172.17.0.2"},
        {"time": 40.0, "kind": "judge", "client": "172.17.0.9"}])
    summary = RUN._summarize(out_root, RUN._task_index())
    kinds = {r["task_id"]: r["harness_errors"] for r in summary["per_task"]}
    assert kinds == {tids[0]: [], tids[1]: ["agent_web_search"]}
    assert summary["recovered_model_endpoint_events"] == 1
    assert summary["harness_events"] == {"agent_web_search": 1, "model_endpoint": 1, "judge": 1}
    assert summary["n_unattributed_harness_events"] == 1          # 172.17.0.9 ran no task
    assert summary["global_mean"] == pytest.approx(1.0) and summary["n_headline_tasks"] == 1


def test_an_unrecovered_endpoint_failure_is_a_harness_error(launcher, monkeypatch, tmp_path):
    tid = "03_Social_Interaction_task_1"
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {tid: (None, None)}))
    _jsonl(launcher / RUN.TASK_WINDOWS, [{"task_id": tid, "start": 0.0, "end": 100.0}])
    _jsonl(launcher / RUN.HARNESS_EVENTS, [
        {"time": 5.0, "kind": "model_endpoint_recovered", "client": "127.0.0.1"},
        {"time": 20.0, "kind": "model_endpoint", "client": "127.0.0.1"}])
    summary = RUN._summarize(str(tmp_path / "out"), RUN._task_index())
    # a recovery before the failure does not count; the task ran (a run record) but has no score
    assert summary["per_task"][0]["harness_errors"] == ["model_endpoint"]
    assert summary["recovered_model_endpoint_events"] == 0


def test_every_isolated_grade_cap_hit_is_a_harness_error(launcher, monkeypatch, tmp_path):
    # (was: a cap hit counted only with a judge call; now every cap hit of a grade() exec run with
    # the leftover processes stopped is a harness error, recorded by whether a judge call ran)
    tids = ["03_Social_Interaction_task_1", "03_Social_Interaction_task_2"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {t: (None, None) for t in tids}))
    out_root = str(tmp_path / "out")
    msg = "Command '['docker', 'exec']' timed out after 120 seconds"
    for t in tids:
        _score(out_root, t, {"overall_score": 0.0, "error": msg})
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 0.0, "end": 400.0, "ips": ["172.17.0.2"],
         "grading": [{"start": 200.0, "end": 320.0, "timed_out": True, "isolated": True}]},
        {"task_id": tids[1], "start": 0.0, "end": 400.0, "ips": ["172.17.0.3"],
         "grading": [{"start": 210.0, "end": 330.0, "timed_out": True, "isolated": True}]}])
    # a judge call of task_1's grade(); task_2's grade() made none
    _jsonl(launcher / RUN.GRADER_CALLS, [{"start": 250.0, "end": 290.0,
                                          "client": "172.17.0.2", "ok": True}])
    summary = RUN._summarize(out_root, RUN._task_index())
    kinds = {r["task_id"]: r["harness_errors"] for r in summary["per_task"]}
    assert kinds == {tids[0]: ["grading_infra"], tids[1]: ["grading_infra"]}
    timeouts = {r["task_id"]: r["grading_timeout"] for r in summary["per_task"]}
    assert timeouts == {tids[0]: "with_judge_call", tids[1]: "without_judge_call"}
    assert summary["grading_timeouts"] == {"with_judge_call": 1, "without_judge_call": 1,
                                           "unrecorded": 0, "not_isolated_infra": 0,
                                           "not_isolated": 0}
    assert summary["global_mean"] is None and summary["n_headline_tasks"] == 0


def test_missing_inputs_make_a_harness_error(launcher, monkeypatch, tmp_path):
    tids = ["01_Productivity_Flow_task_1", "01_Productivity_Flow_task_2"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {t: (None, None) for t in tids}))
    out_root = str(tmp_path / "out")
    _score(out_root, tids[0], {"overall_score": 0.0, "error": "missing gt file: /x"})
    _score(out_root, tids[1], {"overall_score": 1.0})
    (launcher / RUN.MISSING_INPUTS).write_text(json.dumps(
        {tids[0]: ["ground-truth file /h/workspace/gt/ground_truth.json"]}))
    summary = RUN._summarize(out_root, RUN._task_index())
    row = next(r for r in summary["per_task"] if r["task_id"] == tids[0])
    assert row["harness_errors"] == ["missing_input"]
    assert row["missing_inputs"] == ["ground-truth file /h/workspace/gt/ground_truth.json"]
    assert summary["harness_error_kinds"] == {"missing_input": 1}
    assert summary["missing_inputs"] == {tids[0]: [
        "ground-truth file /h/workspace/gt/ground_truth.json"]}
    assert summary["global_mean"] == pytest.approx(1.0)


def test_runner_names_the_new_kinds(monkeypatch):
    assert W.describe_harness_errors({"missing_input": 2, "agent_helper": 1}) == (
        "2 provisioned input data (workspace or ground truth) missing, 1 Gemini helper-model "
        "service failure")
    assert [k for k, _ in W.HARNESS_ERROR_LABELS] == list(RUN.HARNESS_ERROR_KINDS)
    assert W.HARNESS_ERRORS_VERSION == "9"
    assert f'LABEL {W.HARNESS_ERRORS_LABEL}="{W.HARNESS_ERRORS_VERSION}"' in \
        _read("gbench/docker/wildclawbench.Dockerfile")
    entry = _read("gbench/docker/wildclawbench_entrypoint.sh")
    assert "export WILDCLAW_GRADER_KEY" in entry and "secrets.token_hex" in entry


# --------------------------------------------------------------------------- #
# gaia2: runs with a harness error are left out of the rates
# --------------------------------------------------------------------------- #
class _Exc(Exception):
    def __init__(self, status=None):
        super().__init__("x")
        if status is not None:
            self.status_code = status


class BadRequestError(Exception):
    pass


class ContextWindowExceededError(Exception):
    status_code = 400


def test_endpoint_failure_classification():
    for status in (500, 502, 503, 504, 408, 429, 401, 403, 404):
        assert GRUN.endpoint_failure(_Exc(status)), status
    for status in (400, 422):
        assert not GRUN.endpoint_failure(_Exc(status)), status
    assert GRUN.endpoint_failure(ConnectionRefusedError())            # no HTTP status
    assert not GRUN.endpoint_failure(BadRequestError())
    assert not GRUN.endpoint_failure(ContextWindowExceededError())


@pytest.mark.parametrize("row,failures,kinds", [
    ({"status": "failed"}, [{"role": "agent", "endpoint_failure": True}], ["model_endpoint"]),
    ({"status": "failed"}, [{"role": "agent", "endpoint_failure": False}], []),
    ({"status": "failed", "has_exception": True, "exception_type": "ServiceUnavailableError",
      "rationale": "Exception"}, [{"role": "judge"}], ["judge"]),
    ({"status": "failed", "has_exception": True, "exception_type": "KeyError",
      "rationale": "Exception"}, [], ["validation_exception"]),
    ({"status": "failed", "has_exception": True, "exception_type": "RuntimeError"}, [],
     ["run_exception"]),
    ({"status": "failed", "has_exception": True, "exception_type": "ScenarioTimeoutError"}, [], []),
    ({"status": "no_validation"}, [], []),
    ({"status": "success"}, [], []),
])
def test_classify_run(row, failures, kinds):
    assert GRUN.classify_run(row, failures) == kinds


def test_gbench_statistics_leaves_harness_runs_out():
    rows = [
        {"base_scenario_id": "s1", "run_number": None, "config": "time", "phase_name": "standard",
         "status": "success"},
        {"base_scenario_id": "s2", "run_number": None, "config": "time", "phase_name": "standard",
         "status": "failed"},
        {"base_scenario_id": "s3", "run_number": None, "config": "time", "phase_name": "standard",
         "status": "failed", "has_exception": True, "exception_type": "InternalServerError",
         "rationale": "Exception"},
    ]
    by_run = {GRUN._run_key("time", "standard", "s2", None): [
        {"role": "agent", "endpoint_failure": True, "error": "APIConnectionError"}],
        GRUN._run_key("time", "standard", "s9", None): [{"role": "judge"}]}
    loose = [{"role": "agent", "scenario_id": None}]
    kept = []
    stats = GRUN.gbench_statistics(rows, by_run, loose,
                                   lambda keep: kept.extend(keep) or {"global": {"n": len(keep)}})
    assert [r["base_scenario_id"] for r in kept] == ["s1"]
    assert stats["n_harness_error_runs"] == 2
    assert stats["harness_error_kinds"] == {"model_endpoint": 1, "validation_exception": 1}
    assert stats["unattributed_call_failures"] == 2          # the loose one + s9 (no such run)
    assert stats["statistics"] == {"global": {"n": 1}}


def test_gaia2_launcher_reads_the_recomputed_statistics(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    are = {"statistics": {"global": {"macro_success_rate": 20.0, "validated_runs": 10}}}
    (tmp_path / "benchmark_stats.json").write_text(json.dumps(are))
    (tmp_path / GRUN.GBENCH_STATS).write_text(json.dumps({
        "statistics": {"global": {"macro_success_rate": 25.0, "validated_runs": 8},
                       "per_capability": {}},
        "n_runs": 10, "n_harness_error_runs": 2, "harness_error_kinds": {"judge": 2},
        "unattributed_call_failures": 0, "runs": [
            {"scenario_id": "s1", "harness_errors": ["judge"]},
            {"scenario_id": "s2", "harness_errors": []}]}))
    s = GRUN._summarize()
    assert s["harness_accounting"] is True and s["macro_success_rate"] == 25.0
    assert s["validated_runs"] == 8 and s["are_reported"]["macro_success_rate"] == 20.0
    assert s["harness_error_kinds"] == {"judge": 2} and s["n_harness_error_runs"] == 2
    assert [r["scenario_id"] for r in s["harness_error_runs"]] == ["s1"]


def _gaia2(monkeypatch, tmp_path, **summary):
    monkeypatch.setattr(G, "check_gaia2_prerequisites", lambda: (True, ""))
    # the bridge network's gateway, where the relays listen on loopback here; no docker
    monkeypatch.setattr(G.container_guard, "require_metadata_blocked",
                        lambda *a, **k: "unreachable")
    monkeypatch.setattr(G.gemini_relay, "network_gateway", lambda *a, **k: "127.0.0.1")
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTKEY0000000000000000000000000000")
    for k in ("GBENCH_SHARD", "GBENCH_GAIA2_CONFIG", "GBENCH_GAIA2_LIMIT", "GBENCH_GAIA2_MODE",
              "GEMINI_OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    wd = tmp_path / "gaia2_wd"
    wd.mkdir(exist_ok=True)
    monkeypatch.setattr(G.tempfile, "mkdtemp", lambda prefix="": str(wd))
    full = {"harness_accounting": True, "macro_success_rate": 40.0, "micro_success_rate": 38.0,
            "total_scenarios": 10, "validated_runs": 8, "no_validation_runs": 0,
            "agent_invoked": True, "per_capability": {}, "harness_events": {}}
    full.update(summary)

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"]:
            (wd / "gaia2_summary.json").write_text(json.dumps(full))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(G.subprocess, "run", fake_run)
    return G.run_gaia2("m", "http://127.0.0.1:9123/v1")


def test_gaia2_harness_runs_are_named_and_set_the_status(monkeypatch, tmp_path):
    res = _gaia2(monkeypatch, tmp_path, n_harness_error_runs=3,
                 harness_error_kinds={"judge": 2, "model_endpoint": 1},
                 are_reported={"macro_success_rate": 30.0})
    assert res["status"] == "completed_with_errors" and res["accuracy"] == pytest.approx(40.0)
    assert res["n_harness_error_runs"] == 3 and res["are_reported"]["macro_success_rate"] == 30.0
    reason = res["leaderboard_comparable_reason"]
    assert ("3 scenario run(s) hit a harness error and are left out of the success rates (1 a "
            "model endpoint call of the agent failed on the endpoint's side, 2 a judge call of "
            "the validation failed)") in reason
    assert "count as failures" not in reason


def test_gaia2_withholds_a_rate_it_cannot_account_for(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="no harness-error accounting"):
        _gaia2(monkeypatch, tmp_path, harness_accounting=False)
    with pytest.raises(RuntimeError, match=r"2 failed LLM call\(s\) \(agent\) could not be tied"):
        _gaia2(monkeypatch, tmp_path, unattributed_call_failures=2,
               unattributed_call_failure_roles=["agent"])
    with pytest.raises(RuntimeError, match="no scenario run produced a validated score"):
        _gaia2(monkeypatch, tmp_path, validated_runs=0, n_harness_error_runs=10,
               harness_error_kinds={"model_endpoint": 10})


def test_gaia2_gate_refuses_an_image_without_the_label(monkeypatch):
    monkeypatch.setattr(G.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="", stderr=""))
    monkeypatch.setattr(G, "_docker_image_exists", lambda _i: True)
    ok, why = G.check_gaia2_prerequisites()
    assert not ok and "predates the current launcher" in why
    assert f"{G.HARNESS_ERRORS_LABEL}={G.HARNESS_ERRORS_VERSION}" in why
    assert f'LABEL {G.HARNESS_ERRORS_LABEL}="{G.HARNESS_ERRORS_VERSION}"' in \
        _read("gbench/docker/gaia2.Dockerfile")


def test_gaia2_launcher_runs_are_through_the_hooks():
    src = _read("gbench/docker/gaia2_run.py")
    assert '[sys.executable, os.path.abspath(__file__), "--are-hooked", *cmd[1:]]' in src
    assert "litellm_engine.completion = completion" in src
    assert "cli.generate_and_save_reports = generate_and_save_reports" in src
    assert [k for k, _ in G.HARNESS_ERROR_LABELS] == list(GRUN.HARNESS_ERROR_KINDS)


# --------------------------------------------------------------------------- #
# Docs follow the code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite", ["mcp_atlas", "mcp_bench", "wildclawbench"])
def test_guarded_suites_document_the_rule(suite):
    doc = _read(f"docs/evals/{suite}.md")
    assert CG.FIREWALL_COMMAND in doc and "169.254.169.254:80" in doc
    assert "iptables-persistent" in doc and "gbench never runs sudo" in doc
    assert "`container_guard.require_metadata_blocked`" in doc


def test_suite_docs_name_the_new_behavior():
    atlas = _flat(_read("docs/evals/mcp_atlas.md"))
    # the page tools run on the release's servers: no gbench URL check any more
    assert "`INVALID_URL`" not in atlas and "refused_by_url_check" not in atlas
    assert "`search_substitution.upstream_tool_calls`" in atlas
    bench = _flat(_read("docs/evals/mcp_bench.md"))
    assert "`GBENCH_MCP_BENCH_NETWORK`" in bench and "`docker_network`" in bench
    wc = _flat(_read("docs/evals/wildclawbench.md"))
    assert "| `GBENCH_WILDCLAWBENCH_TASK_NETWORK` | `bridge` |" in wc
    assert "`recovered_model_endpoint_events`" in wc and "`missing_inputs`" in wc
    assert f"`org.gbench.wildclawbench-harness-errors={W.HARNESS_ERRORS_VERSION}`" in wc
    assert "a grade() that fails on the missing data is a harness error" not in wc
    gaia2 = _flat(_read("docs/evals/gaia2.md"))
    for kind, _label in G.HARNESS_ERROR_LABELS:
        assert f"`{kind}`" in gaia2, kind
    assert "`are_reported`" in gaia2
    assert f"`{G.HARNESS_ERRORS_LABEL}={G.HARNESS_ERRORS_VERSION}`" in gaia2
    assert "so the affected scenario runs count as failures" not in gaia2


# --------------------------------------------------------------------------- #
# mcp_bench / toolbench: a run where no model request got a reply publishes no score
# --------------------------------------------------------------------------- #
def test_every_model_request_failed():
    assert MB.every_model_request_failed(None) == 0
    assert MB.every_model_request_failed({"counts": {"replies": 3, "failed_requests": 2}}) == 0
    assert MB.every_model_request_failed({"counts": {"replies": 0, "failed_requests": 5}}) == 5
    assert MB.every_model_request_failed({"counts": {"replies": 0, "failed_requests": 0}}) == 0


def test_mcp_bench_all_failed_requests_withhold_the_score(mb, monkeypatch):
    _guard(monkeypatch)
    monkeypatch.setattr(MB, "check_request_log", lambda *a, **k: {
        "counts": {"replies": 0, "failed_requests": 7}, "failed_by_error": {}})
    with pytest.raises(RuntimeError, match=r"none of the run's 7 model-under-test request\(s\) "
                                           r"got a reply"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")


def test_toolbench_uses_the_same_rule():
    src = _read("gbench/runners/eval_suites/toolbench.py")
    assert "all_failed = every_model_request_failed(replies)" in src
    for doc in ("docs/evals/toolbench.md", "docs/evals/mcp_bench.md"):
        assert "When not one request of the run got a reply, every outcome is a harness error" in \
            _flat(_read(doc))
