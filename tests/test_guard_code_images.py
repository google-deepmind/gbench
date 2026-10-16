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

"""The containers that run model-controlled code or tool calls in bfcl, bfcl_v4_agentic, tau2,
tau3, livebench, bigcodebench, multipl_e, ojbench and aider_polyglot, and their network.

Docker is stubbed in every test here: no container starts and nothing contacts the metadata
server or a model endpoint. The endpoint probe's script runs against sockets on 127.0.0.1 only.

* A container that needs no network runs with `--network none`: the BFCL image probe, single-turn
  checker, id selection, entry counts and `bfcl evaluate`, the OJBench judge, the MultiPL-E
  evaluator, the LiveBench image checks.
* A container that runs the model's code or calls with network access runs on docker's bridge
  network, never the host network (the host's DOCKER-USER rule covers a bridge network only), and
  only after the checks of tau_common.require_endpoint_network (the metadata guard, then, for an
  endpoint on this host, a TCP connection to host.docker.internal from that network), or, for
  bigcodebench, which never talks to the endpoint, the metadata guard alone.
* An endpoint on this host is reached at host.docker.internal; a server bound to 127.0.0.1 only
  is a hard error before the run, never a run of failed requests.
"""

import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import bfcl as B
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites import bigcodebench as BCB
from gbench.runners.eval_suites import container_guard as cg
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import multipl_e as ME
from gbench.runners.eval_suites import ojbench as OJ
from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "gbench" / "runners" / "eval_suites"
DOCS = REPO / "docs" / "evals"
ALIAS = "host.docker.internal"

METADATA_UNREACHABLE = (0, "GBENCH_METADATA_PROBE=unreachable timeout\n", "")


def unreachable_resolving(*names):
    """The probe's answer: metadata server unreachable, each of `names` resolved."""
    return (0, "GBENCH_METADATA_PROBE=unreachable timeout\n"
            + "".join(f"GBENCH_DNS_PROBE=ok {n}\n" for n in names), "")


METADATA_REACHABLE = (0, "GBENCH_METADATA_PROBE=reachable\n", "")
ENDPOINT_OPEN = (0, "GBENCH_ENDPOINT_PROBE=open\n", "")
ENDPOINT_CLOSED = (0, "GBENCH_ENDPOINT_PROBE=closed ConnectionRefusedError 111\n", "")


class FakeDocker:
    """Records every docker argv; answers `docker run` from a list of (rc, stdout, stderr)."""

    def __init__(self, answers=(), raise_on_run=None):
        self.calls, self.answers, self.raise_on_run = [], list(answers), raise_on_run

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if self.raise_on_run is not None:
            raise self.raise_on_run
        rc, out, err = self.answers.pop(0)
        return subprocess.CompletedProcess(argv, rc, out, err)

    @property
    def runs(self):
        return [c for c in self.calls if c[:2] == ["docker", "run"]]


@pytest.fixture(autouse=True)
def _fresh_caches():
    cg.reset_verdicts()
    TC._ENDPOINT_REACHABLE.clear()
    yield
    cg.reset_verdicts()
    TC._ENDPOINT_REACHABLE.clear()


def _gate_recorder(monkeypatch, fail=None):
    """Replace the shared gate: record its calls and answer as the real one does."""
    calls = []

    def gate(eval_name, image, url, docs_url):
        calls.append((eval_name, image, url, docs_url))
        if fail is not None:
            raise fail
        return TC.container_url(url)
    monkeypatch.setattr(TC, "require_endpoint_network", gate)
    return calls


def _network(cmd):
    return cmd[cmd.index("--network") + 1]


# --------------------------------------------------------------------------- #
# the shared helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url, expected", [
    ("http://localhost:8000/v1", f"http://{ALIAS}:8000/v1"),
    ("http://127.0.0.1:9/v1", f"http://{ALIAS}:9/v1"),
    ("http://127.4.5.6:8000", f"http://{ALIAS}:8000"),
    ("http://0.0.0.0:8000/v1", f"http://{ALIAS}:8000/v1"),
    ("http://[::1]:8000/v1", f"http://{ALIAS}:8000/v1"),
    ("HTTP://LOCALHOST:8000/v1", f"HTTP://{ALIAS}:8000/v1"),
    ("http://u:p@localhost:8000/v1", f"http://u:p@{ALIAS}:8000/v1"),
    ("http://127.0.0.1", f"http://{ALIAS}"),
    ("https://api.example.com/v1", "https://api.example.com/v1"),
    ("http://localhostx:8000/v1", "http://localhostx:8000/v1"),
    ("http://10.0.0.5:8000/v1", "http://10.0.0.5:8000/v1"),
    ("", ""),
])
def test_only_a_loopback_host_becomes_host_docker_internal(url, expected):
    assert TC.container_url(url) == expected


def test_the_bridge_network_arguments_never_name_the_host_network():
    assert TC.ENDPOINT_NETWORK == "bridge"
    assert TC.ENDPOINT_NETWORK_ARGS == ("--network", "bridge",
                                        "--add-host", f"{ALIAS}:host-gateway")


def test_proxies_are_rewritten_for_the_bridge_and_the_alias_goes_direct():
    out = TC.container_proxy_env({"HTTP_PROXY": "http://127.0.0.1:3128",
                                  "https_proxy": "http://proxy.corp:8080",
                                  "NO_PROXY": "localhost,127.0.0.1"})
    assert out["HTTP_PROXY"] == f"http://{ALIAS}:3128"
    assert out["https_proxy"] == "http://proxy.corp:8080"
    assert out["NO_PROXY"] == f"localhost,127.0.0.1,{ALIAS}" and out["no_proxy"] == ALIAS
    assert TC.container_proxy_env({"NO_PROXY": "x"}) == {}
    assert TC.container_proxy_env({"HTTP_PROXY": "http://p:1", "NO_PROXY": ALIAS})["NO_PROXY"] \
        == ALIAS


def test_the_gate_probes_the_bridge_then_the_endpoint_and_returns_its_url(monkeypatch):
    fake = FakeDocker([METADATA_UNREACHABLE, ENDPOINT_OPEN])
    monkeypatch.setattr(subprocess, "run", fake)
    url = TC.require_endpoint_network("tau2", "img:1", "http://127.0.0.1:8000/v1",
                                      "docs/evals/tau2.md")
    assert url == f"http://{ALIAS}:8000/v1"
    metadata, endpoint = fake.runs
    assert _network(metadata) == "bridge" and "169.254.169.254" in metadata[-1]
    # the tau container resolves no Gemini host: its Gemini calls go to gbench's Gemini relay at
    # the bridge gateway's address
    assert repr(TC.GEMINI_API_HOST) not in metadata[-1]
    assert endpoint[:6] == ["docker", "run", "--rm", "--pull", "never", "--name"]
    for pair in (["--network", "bridge"], ["--add-host", f"{ALIAS}:host-gateway"],
                 ["--cap-drop", "ALL"], ["--security-opt", "no-new-privileges"],
                 ["--entrypoint", "python3"]):
        i = endpoint.index(pair[0])
        assert endpoint[i:i + 2] == pair
    assert "--read-only" in endpoint and "img:1" in endpoint
    assert "-e" not in endpoint and "-v" not in endpoint and "--env-file" not in endpoint
    assert f"(({ALIAS!r}, 8000)" in endpoint[-1]
    # both answers are cached: a second suite container probes nothing
    assert TC.require_endpoint_network("tau2", "img:1", "http://localhost:8000/v1",
                                       "docs/evals/tau2.md") == f"http://{ALIAS}:8000/v1"
    assert len(fake.runs) == 2


def test_a_remote_endpoint_needs_only_the_metadata_probe(monkeypatch):
    # the metadata probe also resolves the endpoint's host name and Hugging Face (livebench)
    fake = FakeDocker([unreachable_resolving("api.example.com", "huggingface.co")])
    monkeypatch.setattr(subprocess, "run", fake)
    assert TC.require_endpoint_network("livebench", "img", "https://api.example.com/v1",
                                       "d.md") == "https://api.example.com/v1"
    assert len(fake.runs) == 1
    assert "'api.example.com'" in fake.runs[0][-1] and "'huggingface.co'" in fake.runs[0][-1]


def test_a_reachable_metadata_server_stops_before_the_endpoint_probe(monkeypatch):
    fake = FakeDocker([METADATA_REACHABLE])
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as ei:
        TC.require_endpoint_network("aider_polyglot", "img", "http://127.0.0.1:8000/v1",
                                    "docs/evals/aider_polyglot.md")
    msg = str(ei.value)
    assert "[INFRA REQUIRED] aider_polyglot" in msg and cg.FIREWALL_COMMAND in msg
    assert cg.PERSIST_COMMAND in msg and len(fake.runs) == 1


def test_an_endpoint_bound_to_loopback_only_is_infra_required_and_not_cached(monkeypatch):
    fake = FakeDocker([METADATA_UNREACHABLE, ENDPOINT_CLOSED, ENDPOINT_CLOSED])
    monkeypatch.setattr(subprocess, "run", fake)
    for _ in range(2):
        with pytest.raises(RuntimeError) as ei:
            TC.require_endpoint_network("bfcl", "img", "http://127.0.0.1:8000/v1",
                                        "docs/evals/bfcl.md")
        msg = str(ei.value)
        assert "[INFRA REQUIRED] bfcl" in msg and f"{ALIAS}:8000" in msg
        assert "http://127.0.0.1:8000" in msg and "127.0.0.1 only" in msg
        assert "ConnectionRefusedError 111" in msg and "docs/evals/bfcl.md" in msg
    assert len(fake.runs) == 3          # the metadata verdict is cached, a failure is not


@pytest.mark.parametrize("answer", [(125, "", "docker: No such image: img"),
                                    (127, "", "exec: python3: not found"), (0, "", "")])
def test_an_endpoint_probe_without_a_verdict_is_infra_required(monkeypatch, answer):
    fake = FakeDocker([unreachable_resolving(TC.GEMINI_API_HOST), answer])
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] tau3 .*gave no verdict"):
        TC.require_endpoint_network("tau3", "img", "http://localhost:8000/v1", "d.md")


def test_an_endpoint_probe_timeout_removes_the_probe_container(monkeypatch):
    fake = FakeDocker([unreachable_resolving(TC.GEMINI_API_HOST)])
    monkeypatch.setattr(subprocess, "run", fake)
    cg.require_metadata_blocked("tau2", "img", "bridge", resolve=[TC.GEMINI_API_HOST])
    fake.raise_on_run = subprocess.TimeoutExpired(["docker"], 120)
    with pytest.raises(RuntimeError, match="could not check that the model endpoint"):
        TC.require_endpoint_network("tau2", "img", "http://localhost:8000/v1", "d.md")
    removed = [c for c in fake.calls if c[:3] == ["docker", "rm", "-f"]]
    assert removed and removed[0][3].startswith("gbench-endpoint-probe-")


def test_the_endpoint_probe_script_opens_and_closes_a_connection_and_sends_nothing(tmp_path):
    script = TC._endpoint_probe_script("127.0.0.1", 1)
    assert "send" not in script and "GET" not in script and "Metadata" not in script
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()
    try:
        out = subprocess.run([sys.executable, "-c", TC._endpoint_probe_script("127.0.0.1", port)],
                             capture_output=True, text=True, timeout=60).stdout
        assert out.strip() == "GBENCH_ENDPOINT_PROBE=open"
        conn, _ = server.accept()
        conn.settimeout(5)
        assert conn.recv(16) == b""            # closed without a byte sent
        conn.close()
        out = subprocess.run(
            [sys.executable, "-c", TC._endpoint_probe_script("127.0.0.1", closed_port)],
            capture_output=True, text=True, timeout=60).stdout
        assert out.startswith("GBENCH_ENDPOINT_PROBE=closed ConnectionRefusedError")
    finally:
        server.close()


# --------------------------------------------------------------------------- #
# tau2 / tau3
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("domains", [["airline"], ["banking_knowledge"]])
def test_the_tau_container_runs_on_the_bridge(domains):
    cmd = TC._docker_run_command("n", "gbench-tau2:x", domains, "/io", "/cache", None, [],
                                 suites_dir="/s")
    assert cmd[cmd.index("--network"):cmd.index("--network") + 4] == list(TC.ENDPOINT_NETWORK_ARGS)
    assert "host" not in cmd and cmd.count("--network") == 1


@pytest.fixture
def tau_launch(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(TC, "check_role_providers", lambda: (True, ""))
    monkeypatch.setattr(TC, "check_tau_image", lambda: (True, ""))
    monkeypatch.setattr(TC, "_image_id", lambda image: "sha256:x")
    monkeypatch.setattr(TC.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))

    @contextlib.contextmanager
    def bridge(eval_name, image, base_url, agent_base_url, domains, docs_url):
        # the Gemini gate and the relays on the bridge gateway
        # (tests/test_tau_gemini_cascade_and_relays.py)
        yield TC._TauBridge(agent_base_url or base_url)
    monkeypatch.setattr(TC, "_bridge_services", bridge)
    state = {"launched": []}

    def launch(cmd, env=None):
        state["launched"].append(cmd)
        io_dir = next(a.split(":")[0] for a in cmd if a.endswith(":/gbench_io:rw"))
        with open(os.path.join(io_dir, "job.json"), encoding="utf-8") as fh:
            state["job"] = json.load(fh)
        with open(os.path.join(io_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump({"ok": True, "result": {"tau2_report": {}}}, fh)
        return 0, ""
    monkeypatch.setattr(TC, "_launch_image", launch)
    return state


def test_tau_checks_the_network_before_its_container_and_the_agent_uses_its_url(
        monkeypatch, tau_launch):
    gate = _gate_recorder(monkeypatch)
    TC.run_tau_suite("tau2", ["airline"], "m", "http://127.0.0.1:9/v1", 1, None,
                     "docs/evals/tau2.md")
    assert gate == [("tau2", TC.tau2_image(), "http://127.0.0.1:9/v1", "docs/evals/tau2.md")]
    assert tau_launch["job"]["base_url"] == f"http://{ALIAS}:9/v1"
    assert "bridge" in tau_launch["launched"][0]


def test_tau3_starts_no_container_when_the_gate_fails(monkeypatch, tau_launch):
    _gate_recorder(monkeypatch, fail=RuntimeError("[INFRA REQUIRED] tau3 ... DOCKER-USER"))
    with pytest.raises(RuntimeError, match="DOCKER-USER"):
        TC.run_tau_suite("tau3", ["banking_knowledge"], "m", "http://127.0.0.1:9/v1", 1, None,
                         "docs/evals/tau3.md")
    assert tau_launch["launched"] == []


def test_tau_forwards_the_proxies_as_the_bridge_needs_them():
    host = {"HTTPS_PROXY": "http://127.0.0.1:3128", "OPENAI_API_KEY": "sk-endpoint-0123456789"}
    env = TC._launch_env(host)
    assert env["HTTPS_PROXY"] == f"http://{ALIAS}:3128" and env["NO_PROXY"] == ALIAS
    names = TC._forwarded_env(host)
    assert {"HTTPS_PROXY", "NO_PROXY", "no_proxy"} <= set(names)
    assert "NO_PROXY" not in TC._forwarded_env({"OPENAI_API_KEY": "x"})


# --------------------------------------------------------------------------- #
# BFCL
# --------------------------------------------------------------------------- #
def test_bfcl_jobs_run_with_no_network_unless_they_ask_for_the_bridge():
    none = AG._docker_run_command("n", [], [], "pass")
    assert _network(none) == "none" and "--add-host" not in none
    bridge = AG._docker_run_command("n", [], [], "pass", network="bridge")
    i = bridge.index("--network")
    assert bridge[i:i + 4] == list(TC.ENDPOINT_NETWORK_ARGS)
    for bad in ("host", "default", "mynet"):
        with pytest.raises(ValueError):
            AG._docker_run_command("n", [], [], "pass", network=bad)


def _capture_launch(monkeypatch, write_result=None):
    seen = {}

    def launch(cmd, env, timeout, name, stream=False):
        seen.update(cmd=cmd, env=env)
        io = next((a.split(":")[0] for a in cmd if a.endswith(":/gbench_io:rw")), None)
        if io:
            with open(os.path.join(io, "job.json"), encoding="utf-8") as fh:
                seen["job"] = json.load(fh)
            with open(os.path.join(io, "result.json"), "w", encoding="utf-8") as fh:
                json.dump({"ok": True, "value": write_result}, fh)
        return 0, ""
    monkeypatch.setattr(AG, "_launch", launch)
    return seen


def test_bfcl_evaluate_runs_with_no_network_and_no_gate(monkeypatch, tmp_path):
    _gate_recorder(monkeypatch, fail=AssertionError("evaluate needs no network check"))
    seen = _capture_launch(monkeypatch)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}
    AG._run_cli(["evaluate", "--model", "m"], env, None, model="m")
    assert _network(seen["cmd"]) == "none" and "--add-host" not in seen["cmd"]
    assert seen["env"]["OPENAI_BASE_URL"] == "http://127.0.0.1:9/v1"


def test_bfcl_generate_runs_on_the_bridge_after_the_gate(monkeypatch, tmp_path):
    gate = _gate_recorder(monkeypatch)
    seen = _capture_launch(monkeypatch)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
           "REMOTE_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
           "HTTP_PROXY": "http://localhost:3128"}
    AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    assert gate == [("bfcl_v4_agentic", AG.bfcl_image(), "http://127.0.0.1:8000/v1",
                     AG.DOCS_URL)]
    cmd = seen["cmd"]
    assert cmd[cmd.index("--network"):cmd.index("--network") + 4] == \
        list(TC.ENDPOINT_NETWORK_ARGS)
    for var in ("OPENAI_BASE_URL", "REMOTE_OPENAI_BASE_URL"):
        assert seen["env"][var] == f"http://{ALIAS}:8000/v1"
    assert seen["env"]["HTTP_PROXY"] == f"http://{ALIAS}:3128"
    for var in ("NO_PROXY", "no_proxy"):
        assert seen["env"][var] == ALIAS and ["-e", var] == cmd[cmd.index(var) - 1:
                                                                cmd.index(var) + 1]
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1", "the caller's env is untouched"


def test_bfcl_run_in_image_on_the_bridge_hands_the_job_the_container_url(monkeypatch):
    gate = _gate_recorder(monkeypatch)
    seen = _capture_launch(monkeypatch, write_result={"traces": []})
    AG.run_in_image("bfcl", "bfcl", "multi_turn_job", {"base_url": "http://localhost:8000/v1",
                                                        "model_name": "m"}, network="bridge")
    assert gate == [("bfcl", AG.bfcl_image(), "http://localhost:8000/v1", "docs/evals/bfcl.md")]
    assert seen["job"]["kwargs"]["base_url"] == f"http://{ALIAS}:8000/v1"
    assert "--add-host" in seen["cmd"] and _network(seen["cmd"]) == "bridge"


def test_bfcl_run_in_image_defaults_to_no_network_and_no_gate(monkeypatch):
    _gate_recorder(monkeypatch, fail=AssertionError("no network, no check"))
    seen = _capture_launch(monkeypatch, write_result={"release": "x"})
    AG.run_in_image("bfcl", "bfcl", "image_probe", {"single_turn": True})
    assert _network(seen["cmd"]) == "none"


def test_the_bfcl_multi_turn_part_asks_for_the_bridge(monkeypatch):
    seen = {}

    def run_in_image(eval_name, module, func, kwargs, **kw):
        seen.update(func=func, kw=kw, kwargs=kwargs)
        return {"traces": [], "guard_refusals": []}
    monkeypatch.setattr(B._agentic, "run_in_image", run_in_image)
    B.run_multi_turn_in_image("m", "http://127.0.0.1:8000/v1", 1,
                              [{"id": "multi_turn_base_0", "category": "multi_turn_base"}],
                              temperature=0.001, thinking=False)
    assert seen["func"] == "multi_turn_job" and seen["kw"]["network"] == "bridge"


class _Stop(Exception):
    pass


def test_run_bfcl_checks_the_network_before_the_first_multi_turn_generation(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setattr(B, "_harness_probe", lambda name, multi, single: (B.BFCL_EVAL_RELEASE, []))
    checked = []

    def endpoint_network(url):
        checked.append(url)
        raise _Stop()
    monkeypatch.setattr(B, "endpoint_network", endpoint_network)
    monkeypatch.setattr(B, "run_eval_suite", lambda **kw: pytest.fail("generated before the check"))
    monkeypatch.setattr(B, "run_multi_turn_in_image", lambda *a, **k: pytest.fail("multi-turn ran"))
    with pytest.raises(_Stop):
        B.run_bfcl("m", "http://127.0.0.1:8000/v1", 1, eval_categories="multi_turn_base")
    assert checked == ["http://127.0.0.1:8000/v1"]


def test_a_single_turn_bfcl_run_needs_no_network_check(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setattr(B, "_harness_probe", lambda name, multi, single: (B.BFCL_EVAL_RELEASE, []))
    monkeypatch.setattr(B, "endpoint_network", lambda url: pytest.fail("checked a network"))

    def load(cats):
        raise _Stop()
    monkeypatch.setattr(B, "_load_bfcl_samples", load)
    with pytest.raises(_Stop):
        B.run_bfcl("m", "http://127.0.0.1:8000/v1", 1, eval_categories="simple_python")


def test_bfcl_endpoint_network_is_the_shared_gate_for_the_bfcl_suite(monkeypatch):
    gate = _gate_recorder(monkeypatch)
    assert B.endpoint_network("http://127.0.0.1:8000/v1") == f"http://{ALIAS}:8000/v1"
    assert gate == [("bfcl", AG.bfcl_image(), "http://127.0.0.1:8000/v1", B.DOCS_URL)]


def test_the_agentic_prerequisites_end_with_the_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path))

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(AG.urllib.request, "urlopen", lambda url, timeout=None: _Resp())
    monkeypatch.setattr(AG, "image_problem", lambda: None)
    monkeypatch.setattr(AG, "run_in_image", lambda *a, **k: {
        "release": AG.BFCL_EVAL_RELEASE, "agentic_categories": list(AG.AGENTIC_CATEGORIES),
        "agentic_mismatch": None})
    gate = _gate_recorder(monkeypatch)
    ok, why, cats = AG.check_bfcl_prerequisites("http://127.0.0.1:8000/v1")
    assert ok and why == "" and set(cats) == set(AG.AGENTIC_CATEGORIES)
    assert gate == [("bfcl_v4_agentic", AG.bfcl_image(), "http://127.0.0.1:8000/v1",
                     AG.DOCS_URL)]
    _gate_recorder(monkeypatch, fail=RuntimeError("[INFRA REQUIRED] bfcl_v4_agentic ..."))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        AG.check_bfcl_prerequisites("http://127.0.0.1:8000/v1")


# --------------------------------------------------------------------------- #
# livebench
# --------------------------------------------------------------------------- #
def test_the_livebench_run_container_is_on_the_bridge():
    cmd = LB._build_command("img", "/out", "m", "m", f"http://{ALIAS}:8000/v1", 4, None, "", "")
    assert cmd[3:7] == list(TC.ENDPOINT_NETWORK_ARGS) and "host" not in cmd
    assert f"--api-base http://{ALIAS}:8000/v1 " in cmd[-1]


def _livebench(monkeypatch, tmp_path):
    for name in ("GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS",
                 "GBENCH_LIVEBENCH_RELEASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(LB, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None, "commit": None})
    monkeypatch.setattr(LB, "server_aborted_requests", lambda url, ids: None)
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    runs = []

    def fake_run(cmd, **_kw):
        runs.append(cmd)
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\nm,50.0," + ",".join("50.0" for _ in cols)
            + "\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return runs


def test_livebench_checks_the_network_and_passes_the_container_url(monkeypatch, tmp_path):
    runs = _livebench(monkeypatch, tmp_path)
    gate = _gate_recorder(monkeypatch)
    LB.run_livebench("m", "http://127.0.0.1:8000/v1", 4, max_output_tokens=4096)
    assert gate == [("livebench", LB._image(), "http://127.0.0.1:8000/v1", LB.DOCS_URL)]
    (run,) = runs
    assert _network(run) == "bridge" and f"--api-base http://{ALIAS}:8000/v1 " in run[-1]


def test_livebench_starts_no_run_when_the_gate_fails(monkeypatch, tmp_path):
    runs = _livebench(monkeypatch, tmp_path)
    _gate_recorder(monkeypatch, fail=RuntimeError("[INFRA REQUIRED] livebench ... DOCKER-USER"))
    with pytest.raises(RuntimeError, match="DOCKER-USER"):
        LB.run_livebench("m", "http://127.0.0.1:8000/v1", 4, max_output_tokens=4096)
    assert runs == []


def test_the_livebench_image_checks_have_no_network():
    src = (SUITES / "livebench.py").read_text(encoding="utf-8")
    checks = re.findall(r'\["docker", "run", "--rm", "--network", "(\w+)", image, "python"', src)
    assert checks == ["none", "none"]


# --------------------------------------------------------------------------- #
# aider_polyglot
# --------------------------------------------------------------------------- #
def test_the_aider_container_is_on_the_bridge_as_upstream_docker_sh():
    cmd = A._build_command("aider-benchmark:5dc9490b", "/bench", "gbench", "openai/m",
                           f"http://{ALIAS}:8000/v1", 1, A._LANGUAGES, "", None, "")
    assert cmd[3:7] == list(TC.ENDPOINT_NETWORK_ARGS) and "host" not in cmd
    assert f"OPENAI_API_BASE=http://{ALIAS}:8000/v1" in cmd


def _aider(monkeypatch, tmp_path):
    (tmp_path / "polyglot-benchmark").mkdir()
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench_dir: None)
    monkeypatch.setattr(A, "_served_model_id", lambda url, fallback: "m")
    runs = []

    def fake_run(cmd, **_kw):
        runs.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    return runs


def test_aider_checks_the_network_and_hands_aider_the_container_url(monkeypatch, tmp_path):
    runs = _aider(monkeypatch, tmp_path)
    gate = _gate_recorder(monkeypatch)
    with pytest.raises(RuntimeError, match="did not confirm"):   # the fake runs no aider
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert gate == [("aider_polyglot", A._image(), "http://127.0.0.1:8000/v1", A.DOCS_URL)]
    (run,) = runs
    assert _network(run) == "bridge" and f"OPENAI_API_BASE=http://{ALIAS}:8000/v1" in run


def test_aider_starts_no_container_when_the_gate_fails(monkeypatch, tmp_path):
    runs = _aider(monkeypatch, tmp_path)
    _gate_recorder(monkeypatch, fail=RuntimeError("[INFRA REQUIRED] aider_polyglot ..."))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert runs == []


# --------------------------------------------------------------------------- #
# bigcodebench
# --------------------------------------------------------------------------- #
def _bcb_docker(monkeypatch, image_present=True):
    monkeypatch.setattr(BCB.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(BCB.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=1 if cmd[:3] == ["docker", "image", "inspect"] and not image_present else 0,
        stdout=b"", stderr=b""))
    calls = []
    monkeypatch.setattr(cg, "require_metadata_blocked",
                        lambda *a, **k: calls.append((a, k)) or cg.UNREACHABLE)
    return calls


def test_bigcodebench_guards_the_bridge_before_generation(monkeypatch):
    calls = _bcb_docker(monkeypatch)
    assert BCB.check_bigcodebench_prerequisites() == (True, "")
    assert calls == [(("bigcodebench", BCB._image(), "bridge", BCB.DOCS_URL),
                      {"resolve": ("huggingface.co",)})]
    calls = _bcb_docker(monkeypatch, image_present=False)
    ok, _why = BCB.check_bigcodebench_prerequisites()
    assert not ok and calls == [], "no probe of an image that is not there"


def test_bigcodebench_guard_failure_is_a_hard_error(monkeypatch):
    _bcb_docker(monkeypatch)

    def reachable(*a, **k):
        raise RuntimeError("[INFRA REQUIRED] bigcodebench ... " + cg.FIREWALL_COMMAND)
    monkeypatch.setattr(cg, "require_metadata_blocked", reachable)
    with pytest.raises(RuntimeError, match="DOCKER-USER"):
        BCB.check_bigcodebench_prerequisites()


def test_bigcodebench_sanitize_and_evaluate_run_on_the_bridge(monkeypatch):
    import asyncio
    runs = []
    monkeypatch.setattr(BCB.subprocess, "run", lambda cmd, **k: runs.append(cmd) or
                        types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    metrics = {}
    score = BCB._make_scorer("full", 2, metrics, "gbench-bigcodebench:09dd993f")
    trace = {"response_text": "def f():\n    return 1\n",
             "extra_payload": {"task_id": "BigCodeBench/0", "split": "instruct"}}
    asyncio.run(score([trace]))
    assert [r[r.index("--entrypoint") + 1] if "--entrypoint" in r else "evaluate"
            for r in runs] == ["bigcodebench.sanitize", "evaluate"]
    assert all(_network(r) == BCB.HARNESS_NETWORK == "bridge" for r in runs)


# --------------------------------------------------------------------------- #
# ojbench, multipl_e
# --------------------------------------------------------------------------- #
def test_the_ojbench_judge_has_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_OJBENCH_TESTDATA", str(tmp_path))
    seen = []

    def fake_run(cmd, **_kw):
        seen.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            work = next(a.split(":")[0] for a in cmd if a.endswith(":/work"))
            Path(work, "results.jsonl").write_text(
                json.dumps({"gbench_uid": 0, "is_passed": True}) + "\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(OJ.subprocess, "run", fake_run)
    out = OJ._judge_in_container([{"id": 1, "language": "cpp", "content": "x", "gbench_uid": 0}],
                                 1)
    assert out == [{"gbench_uid": 0, "is_passed": True}]
    (run,) = [c for c in seen if c[:2] == ["docker", "run"]]
    assert _network(run) == "none" and run.count("--network") == 1


def test_the_multipl_e_evaluator_has_no_network():
    src = (SUITES / "multipl_e.py").read_text(encoding="utf-8")
    runs = re.findall(r'\["docker", "run", "--rm"([^\]]*)', src)
    assert runs and all(r.startswith(', "--network", "none"') for r in runs), runs
    assert ME.DOCS_URL == "docs/evals/multipl_e.md"


# --------------------------------------------------------------------------- #
# no host network anywhere; the docs say which container runs where
# --------------------------------------------------------------------------- #
OWNED = ("bfcl.py", "bfcl_v4_agentic.py", "tau2.py", "tau3.py", "tau_common.py", "livebench.py",
         "bigcodebench.py", "multipl_e.py", "ojbench.py", "aider_polyglot.py")


@pytest.mark.parametrize("module", OWNED)
def test_no_suite_here_uses_the_host_network(module):
    src = (SUITES / module).read_text(encoding="utf-8")
    assert not re.search(r'"--network",\s*"host"', src) and "--network host" not in src


BRIDGE_DOCS = {"bfcl.md": "Containers and network", "bfcl_v4_agentic.md": "Containers and network",
               "tau2.md": "Container and network", "tau3.md": "Container and network",
               "livebench.md": "Containers and network",
               "bigcodebench.md": "Containers and network",
               "aider_polyglot.md": "Container and network"}
NONE_DOCS = {"multipl_e.md": "Containers and network", "ojbench.md": "Container and network",
             "bfcl_live.md": None}


def _flat(text):
    return " ".join(text.split())


@pytest.mark.parametrize("doc, heading", sorted(BRIDGE_DOCS.items()))
def test_bridge_docs_name_the_host_rule_and_the_guard(doc, heading):
    text = (DOCS / doc).read_text(encoding="utf-8")
    assert f"## {heading}" in text or f"### {heading}" in text
    flat = _flat(text)
    assert cg.FIREWALL_COMMAND in flat and cg.PERSIST_COMMAND in flat
    assert "require_metadata_blocked" in flat
    assert re.search(r"never (on |uses )?the host network|host network is never used", flat)
    assert "169.254.169.254:80" in flat
    if doc != "bigcodebench.md":        # bigcodebench's containers never talk to the endpoint
        assert ALIAS in flat and "127.0.0.1` only" in flat
    assert "--network host" not in flat


@pytest.mark.parametrize("doc, heading", sorted(NONE_DOCS.items()))
def test_no_network_docs_say_so(doc, heading):
    text = (DOCS / doc).read_text(encoding="utf-8")
    if heading:
        assert f"## {heading}" in text
    flat = _flat(text)
    assert "`--network none`" in flat and "needs no host firewall rule" in flat
    assert "--network host" not in flat


def test_the_bfcl_docs_list_every_job_and_its_network():
    bfcl = _flat((DOCS / "bfcl.md").read_text(encoding="utf-8"))
    for row in ("| image probe (`bfcl.image_probe`) | the release and pinned-file checks | "
                "`--network none` |",
                "| single-turn checker (`bfcl.single_turn_verdicts`) |",
                "| multi-turn part (`bfcl.multi_turn_job`) |"):
        assert row in bfcl
    agentic = _flat((DOCS / "bfcl_v4_agentic.md").read_text(encoding="utf-8"))
    assert "| `bfcl evaluate` | BFCL's checker over the result files | `--network none` |" \
        in agentic
    assert AG.NETWORK_JOBS == ("generate",)
