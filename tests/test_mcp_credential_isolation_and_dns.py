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

"""mcp_bench and mcp_atlas: no credential where model-controlled code runs, name resolution
checked before the run, harness errors kept out of the score, and a metadata-server verdict for
an environment gbench did not start.

* mcp_bench: the judge's Gemini key stays in gbench (the container gets gemini_relay's
  placeholder, and the relay forwards only the judge's chat completion); the metadata guard also
  resolves the servers' API host names; the launcher's network check has no IP-only test; each
  task a harness error fell in (endpoint, judge, DNS, failed) is left out of upstream's averages,
  and a run with none left publishes no score.
* mcp_atlas: the servers that need credentials run in a container of their own, reached through
  docker exec, and the environment container the model's shell and code tools run in gets no
  credential; a lara-translate_import_tmx call that would inject a shell command is answered by
  gbench; GBENCH_MCP_ATLAS_SANDBOX_URL is refused when it holds credentialed servers, and its
  metadata-server verdict comes from the probe run through its own code tool.

Docker, the model and the Gemini API are stubbed (local stub servers on OS-assigned ports): no
container starts, nothing contacts 169.254.169.254, and every file is written under tmp_path.
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import gemini_relay
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import mcp_bench as MB

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import mcp_bench_cascade_judge as CJ  # noqa: E402
import mcp_bench_run as RUN  # noqa: E402

#: The real metadata guard, kept before any test stubs it.
_REAL_GUARD = CG.require_metadata_blocked
REAL_KEY = "AIzaSyREALjudgeKEY000011112222"
PLACEHOLDER = gemini_relay.PLACEHOLDER_PREFIX + "ab" * 24


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, key, body=None):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


# =========================================================================================== #
# mcp_bench: the judge's key behind gbench's Gemini relay
# =========================================================================================== #
@pytest.fixture
def gemini_stub(monkeypatch):
    """A stand-in Gemini OpenAI-compatible API on loopback (base.gemini_openai_base_url accepts a
    loopback address for a local stub); records what reaches it."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                         "body": body})
            data = json.dumps({"id": "c1", "object": "chat.completion", "created": 0,
                               "model": body.get("model"),
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant",
                                                        "content": '{"score": 7}'}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                         "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv, url = _serve(H)
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", url + "/v1beta/openai")
    yield seen
    srv.shutdown()
    srv.server_close()


def test_the_judge_reaches_gemini_only_through_the_relay(gemini_stub, monkeypatch):
    gateways = []
    monkeypatch.setattr(gemini_relay, "network_gateway",
                        lambda eval_name, network, docs_url=None:
                        gateways.append((eval_name, network)) or "127.0.0.1")
    relay = MB._start_judge_relay("bridge")
    try:
        assert gateways == [("mcp_bench", "bridge")]        # it listens on the gateway
        env = relay.client_env()
        assert gemini_relay.is_placeholder(env["GEMINI_API_KEY"])
        assert env["GEMINI_API_KEYS"] == env["GEMINI_API_KEY"]
        assert env["GEMINI_OPENAI_BASE_URL"] == relay.origin + "/v1beta/openai"
        # the in-container judge, with the container's environment only
        for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                     "GBENCH_GEMINI_RELAY_URL"):
            monkeypatch.setenv(name, env[name])
        monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-test")
        out = asyncio.run(CJ.CascadeGeminiJudge().get_completion("sys", "user", 50))
        assert out == '{"score": 7}'
        [call] = gemini_stub
        assert call["path"] == "/v1beta/openai/chat/completions"
        assert call["auth"] == f"Bearer {REAL_KEY}"          # added by the relay, in gbench
        assert call["body"]["model"] == "gemini-test" and call["body"]["max_tokens"] == 50
        assert "temperature" not in call["body"]
        # nothing but the judge's chat completion is forwarded, and only with the placeholder
        for path in ("/v1beta/openai/embeddings", "/v1beta/openai/models",
                     "/v1beta/models/gemini-test:generateContent"):
            assert _post(relay.origin + path, env["GEMINI_API_KEY"]) == 403, path
        assert _post(relay.origin + "/v1beta/openai/chat/completions", "wrong") == 401
        assert _post(relay.origin + "/v1beta/openai/chat/completions", REAL_KEY) == 401
    finally:
        relay.stop()
    rec = relay.record()
    assert rec["requests_relayed"] == 1 and rec["openai_calls"] == [
        "POST /v1beta/openai/chat/completions"] and rec["native_methods"] == []
    assert REAL_KEY not in json.dumps(rec)
    assert len(gemini_stub) == 1


def test_the_relay_listens_on_loopback_with_the_host_network(gemini_stub, monkeypatch):
    monkeypatch.setattr(gemini_relay, "network_gateway",
                        lambda *a, **k: pytest.fail("the host network has no gateway to ask"))
    relay = MB._start_judge_relay("host")
    try:
        assert relay.bind_host == "127.0.0.1"
    finally:
        relay.stop()


def _judge_env(monkeypatch, keys, relay=None, base=None):
    monkeypatch.setenv("GEMINI_API_KEYS", ",".join(keys))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for name, value in (("GBENCH_GEMINI_RELAY_URL", relay), ("GEMINI_OPENAI_BASE_URL", base)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_the_judge_sends_a_placeholder_only_to_the_relay_and_a_real_key_never_there(monkeypatch):
    relay = "http://172.17.0.1:40001"
    _judge_env(monkeypatch, [PLACEHOLDER], relay, relay + "/v1beta/openai")
    assert CJ.gemini_openai_base() == relay + "/v1beta/openai/"
    _judge_env(monkeypatch, [PLACEHOLDER], relay)
    assert CJ.gemini_openai_base() == relay + "/v1beta/openai/"
    refusals = [
        ([PLACEHOLDER], None, None, "GBENCH_GEMINI_RELAY_URL is unset"),
        ([PLACEHOLDER], relay, "https://generativelanguage.googleapis.com/v1beta/openai",
         "is not the relay's URL"),
        ([PLACEHOLDER], relay + "/path", None, "is not an http(s) origin"),
        ([PLACEHOLDER], "http://u:p@172.17.0.1:1", None, "is not an http(s) origin"),
        # a real key never goes to the relay's http address
        ([REAL_KEY], relay, relay + "/v1beta/openai", "is not the Gemini API"),
        ([PLACEHOLDER, REAL_KEY], relay, relay + "/v1beta/openai", "is not the Gemini API"),
    ]
    for keys, rel, base, why in refusals:
        _judge_env(monkeypatch, keys, rel, base)
        with pytest.raises(RuntimeError, match=re.escape(why)):
            CJ.gemini_openai_base()
    _judge_env(monkeypatch, [REAL_KEY])
    assert CJ.gemini_openai_base() == CJ.GEMINI_OPENAI_BASE_URL


def test_the_judge_copy_of_the_placeholder_rule_is_the_relays():
    assert CJ.RELAY_PLACEHOLDER_PREFIX == gemini_relay.PLACEHOLDER_PREFIX
    assert CJ.RELAY_URL_ENV == gemini_relay.RELAY_URL_ENV
    assert CJ.RELAY_OPENAI_PREFIX == gemini_relay.OPENAI_PREFIX
    for key in (PLACEHOLDER, PLACEHOLDER.upper(), PLACEHOLDER + "0", REAL_KEY, "", None,
                gemini_relay.PLACEHOLDER_PREFIX + "g" * 48):
        assert CJ.is_placeholder(key) == gemini_relay.is_placeholder(key), key


def test_the_judge_reports_each_outage_to_the_launcher(monkeypatch):
    hits = []
    monkeypatch.setattr(CJ.CascadeGeminiJudge, "outage_hook", lambda: hits.append(1))
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "g1")
    judge = CJ.CascadeGeminiJudge()

    class _Completions:
        async def create(self, **kw):
            raise RuntimeError("503")
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    judge._async_clients = [client]
    judge._client_cycle = iter([client])
    with pytest.raises(RuntimeError, match="JUDGE_OUTAGE"):
        asyncio.run(judge.get_completion("s", "u", 10))
    assert hits == [1]


# --------------------------------------------------------------------------- #
# mcp_bench run: the placeholder in the container, names checked, harness errors apart
# --------------------------------------------------------------------------- #
_CONNECTED = {f"S{i}": {"status": "success", "tools": 3, "error": ""} for i in range(28)}
_PROBE_OK = {"servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
             "connections": _CONNECTED}
_METRICS = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
            "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
            "tool_selection_score": 5.0, "planning_effectiveness_and_efficiency_score": 5.0}


class _Relay:
    """Stands in for a started gemini_relay.GeminiRelay."""

    def __init__(self):
        self.stopped = False

    def client_env(self):
        return {"GEMINI_API_KEY": PLACEHOLDER, "GEMINI_API_KEYS": PLACEHOLDER,
                "GEMINI_OPENAI_BASE_URL": "http://172.17.0.1:40001/v1beta/openai",
                "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:40001"}

    def stop(self):
        self.stopped = True

    def record(self):
        return {"requests_relayed": 3, "clients_hold": "a per-run placeholder"}


class _Cred:
    """Stands in for a started mcp_bench.CredentialedServers."""

    def __init__(self, image, base, network):
        self.image, self.base, self.network = image, base, network
        self.stopped = False

    def client_env(self):
        return {"MCP_CREDENTIALED_RELAY": "172.17.0.1:40002", "MCP_CREDENTIALED_TOKEN": "t0k"}

    def record(self):
        return {"container": self.base + "-cred", "relay": {"sessions": {}}}

    def stop(self):
        self.stopped = True


def _tasks(*specs):
    return [{"task_id": t, "file": f, "status": st, "kinds": list(k), "details": [],
             "judge_calls_failed": j} for t, f, st, k, j in specs]


@pytest.fixture
def bench(monkeypatch):
    for k in ("GBENCH_MCP_BENCH_NETWORK", "GBENCH_MCP_BENCH_DISTRACTION", "GBENCH_SHARD",
              "GBENCH_MCP_BENCH_DISABLE_STABILITY", "GBENCH_MCP_BENCH_ASSUME_NETWORK",
              "GBENCH_MCP_BENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setenv("GEMINI_API_KEYS", REAL_KEY)
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "server-key-value-1234")
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: _PROBE_OK)
    state = {"runs": [], "guard": [], "relays": [], "creds": [],
             "results": dict(_METRICS), "harness": {"version": MB.HARNESS_LOG_VERSION,
                                                    "tasks": _tasks(("t1", "f.json",
                                                                     "completed", (), 0))}}

    def guard(eval_name, image, network, docs_url=None, *, resolve=()):
        state["guard"].append((network, list(resolve)))
        return CG.UNREACHABLE
    monkeypatch.setattr(CG, "require_metadata_blocked", guard)

    def relay(network):
        r = _Relay()
        state["relays"].append((network, r))
        return r
    monkeypatch.setattr(MB, "_start_judge_relay", relay)

    def cred(image, base, network):
        c = _Cred(image, base, network)
        state["creds"].append(c)
        return c
    monkeypatch.setattr(MB, "start_credentialed_servers", cred)

    def fake_run(cmd, **kw):
        if cmd[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "[]", "")
        if cmd[:2] == ["docker", "run"]:
            state["runs"].append((list(cmd), kw))
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                json.dump(state["results"], fh)
            with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                json.dump({"unavailable_servers": [], "tasks_kept": 104}, fh)
            if state["harness"] is not None:
                with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                    json.dump(state["harness"], fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    return state


def test_the_container_gets_the_relay_placeholder_never_the_key(bench):
    res = MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    [(cmd, kw)] = bench["runs"]
    [(network, relay)] = bench["relays"]
    assert network == "bridge" and relay.stopped
    names = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                 "GBENCH_GEMINI_RELAY_URL"):
        assert name in names, name                 # by name: the value is the docker CLI's env
    env = kw["env"]
    assert env["GEMINI_API_KEY"] == env["GEMINI_API_KEYS"] == PLACEHOLDER
    assert env["GEMINI_OPENAI_BASE_URL"] == "http://172.17.0.1:40001/v1beta/openai"
    assert REAL_KEY not in " ".join(cmd)
    assert not any(REAL_KEY in str(v) for v in env.values())
    assert res["gemini_relay"] == relay.record() and res["status"] == "success"
    assert "MCP_HARNESS_LOG=/out/harness_log.json" in names
    assert "MCP_DNS_NAMES=" + ",".join(MB.server_dns_names()) in names
    # no server key reaches the model container: the credentialed container gets them
    assert not any(n.split("=")[0] in MB._SERVER_KEYS for n in names)
    assert not any(k in env for k in MB._SERVER_KEYS)
    assert "server-key-value-1234" not in " ".join(cmd) + " ".join(map(str, env.values()))
    assert "MCP_CREDENTIALED_RELAY" in names and "MCP_CREDENTIALED_TOKEN" in names
    [c] = bench["creds"]
    assert c.stopped and res["credentialed_servers"] == c.record()


def test_the_relay_is_stopped_when_the_container_fails(bench, monkeypatch):
    def failing(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(MB.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*rc=1"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    [(_, relay)] = bench["relays"]
    assert relay.stopped


def test_the_guard_resolves_the_servers_and_a_named_endpoint(bench):
    MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    MB.run_mcp_bench("m", "http://vllm.internal:8000/v1")
    MB.run_mcp_bench("m", "http://10.1.2.3:8000/v1")
    (_, local), (_, named), (_, by_ip) = bench["guard"]
    assert local == by_ip == MB.server_dns_names()
    assert named == MB.server_dns_names() + ["vllm.internal"]
    assert MB.server_dns_names() == sorted(set(MB.SERVER_API_HOSTS.values()))
    assert len(MB.SERVER_API_HOSTS) == 21 and "OSINT Intelligence" not in MB.SERVER_API_HOSTS


def test_a_dns_outage_stops_the_run_before_the_container(monkeypatch, bench):
    """The real guard over a stubbed docker: the metadata server is unreachable, but a server's
    API host does not resolve on the network, so nothing but the probe starts."""
    monkeypatch.setattr(CG, "require_metadata_blocked", _REAL_GUARD)
    names = MB.server_dns_names()
    probes = []

    def fake_run(cmd, **kw):
        if CG.PROBE_LABEL in cmd:
            probes.append(list(cmd))
            lines = ["GBENCH_METADATA_PROBE=unreachable timeout"]
            lines += [f"GBENCH_DNS_PROBE=ok {n}" for n in names if n != "api.nasa.gov"]
            lines.append("GBENCH_DNS_PROBE=fail api.nasa.gov gaierror:-3")
            return subprocess.CompletedProcess(cmd, 0, "\n".join(lines) + "\n", "")
        if cmd[:2] == ["docker", "run"]:
            bench["runs"].append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "[]", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    assert "cannot resolve host names" in str(ei.value) and "api.nasa.gov" in str(ei.value)
    assert CG.FIREWALL_COMMAND in str(ei.value)
    assert len(probes) == 1 and bench["runs"] == [] and bench["relays"] == []


def test_tasks_a_harness_error_fell_in_are_left_out_and_named(bench):
    bench["harness"]["tasks"] = _tasks(
        ("t1", "f.json", "completed", (), 1), ("t2", "f.json", "completed", ("judge_outage",), 0),
        ("t3", "f.json", "failed", ("task_failed", "dns_failure"), 0),
        ("t4", "f.json", "completed", ("model_endpoint",), 0))
    res = MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    assert res["status"] == "completed_with_errors"
    assert res["tasks_scored"] == 1 and res["judge_calls_failed"] == 1
    un = res["tasks_unscored"]
    assert un["count"] == 3 and un["task_ids"] == ["t2", "t3", "t4"]
    assert un["by_kind"] == {"dns_failure": 1, "judge_outage": 1, "model_endpoint": 1,
                             "task_failed": 1}
    [entry] = [e for e in res["harness_errors"] if e.get("tasks")]
    assert entry["tasks"] == 3
    reason = res["leaderboard_comparable_reason"]
    assert "3 of 4 task(s) left out of the score because a harness error fell in them" in reason
    assert "1 judge call(s) of 1 scored task(s) failed" in reason
    [partial] = [e for e in res["harness_errors"] if e.get("kind") == "judge_outage_partial"]
    assert partial["judge_calls_failed"] == 1 and partial["task_ids"] == ["t1"]


def test_a_run_with_no_task_left_to_score_publishes_nothing(bench):
    bench["harness"]["tasks"] = _tasks(("t1", "f.json", "completed", ("dns_outage",), 0),
                                       ("t2", "f.json", "failed", ("task_failed",), 0))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*no task could be scored: of the 2 "
                                           r"task\(s\) run, 2 had a harness error"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")


def test_a_missing_or_foreign_harness_record_is_infra_required(bench):
    bench["harness"] = None
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*no per-task harness record"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    bench["harness"] = {"version": 99, "tasks": []}
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*no per-task harness record"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    bench["harness"] = {"version": MB.HARNESS_LOG_VERSION, "tasks": []}
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*harness record lists no task"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")


def test_files_are_weighted_by_their_scored_tasks_and_an_empty_file_is_left_out(bench):
    low = dict(_METRICS, task_completion_score=0.0, tool_selection_score=0.0,
               planning_effectiveness_and_efficiency_score=0.0)
    bench["results"] = {"/app/mcp-bench/tasks/a.json": dict(_METRICS),
                        "/app/mcp-bench/tasks/b.json": low}
    bench["harness"]["tasks"] = _tasks(("t1", "a.json", "completed", (), 0),
                                       ("t2", "a.json", "completed", (), 0),
                                       ("t3", "b.json", "completed", ("judge_outage",), 0))
    res = MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")
    # b.json kept no scored task (upstream then repeats the previous file's metrics): left out
    assert res["overall_score"] == round(MB.compute_overall_score(_METRICS)["overall"], 4)
    assert list(res["per_file"]) == ["a.json"]


def test_missing_judged_dimensions_publish_no_score(bench):
    bench["results"] = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0}
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*lacks scored dimensions"):
        MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")


def test_the_image_must_carry_the_harness_label(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setattr(MB, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(MB.shutil, "which", lambda x: "/usr/bin/docker")
    labels = {"org.gbench.model-request-policy": "2"}

    def fake_run(cmd, **kw):
        out = json.dumps(labels) if "--format" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    ok, why = MB.check_mcp_bench_prerequisites()
    assert not ok and f"{MB.HARNESS_LABEL}={MB.HARNESS_VERSION}" in why and "Rebuild" in why
    labels[MB.HARNESS_LABEL] = MB.HARNESS_VERSION
    assert MB.check_mcp_bench_prerequisites() == (True, "")
    df = _read("gbench/docker/mcp_bench.Dockerfile")
    assert f'LABEL {MB.HARNESS_LABEL}="{MB.HARNESS_VERSION}"' in df
    assert MB.HARNESS_LOG_VERSION == RUN.HARNESS_LOG_VERSION
    assert set(MB.HARNESS_ERROR_KINDS) == set(RUN.HARNESS_KINDS)


def test_provisioning_gaps_name_the_names_that_do_not_resolve():
    keys = {k: "set" for k in MB._REQUIRED_KEYS}
    probe = dict(_PROBE_OK, servers=sorted(_CONNECTED), unavailable=["Wikipedia"],
                 network_up=False,
                 network_check={"up": False, "unresolved": {"en.wikipedia.org": "gaierror:-3"}})
    gaps = MB.provisioning_gaps(probe, env_keys=keys)
    assert gaps[0] == ("the container cannot resolve the servers' API host names: "
                       "en.wikipedia.org (gaierror:-3)")
    assert gaps[1] == ("servers the launcher cannot serve (their API host names do not resolve "
                       "in the container): Wikipedia")


def test_the_entrypoint_needs_the_relay_url_with_a_placeholder(tmp_path):
    bench_dir = tmp_path / "bench"
    (bench_dir / "mcp_servers").mkdir(parents=True)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "mcp_bench_run.py").write_text("")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_python = bindir / "python"
    fake_python.write_text("#!/bin/sh\necho launched\n")
    fake_python.chmod(0o755)
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "MCPBENCH_DIR": str(bench_dir),
           "ADAPTER_DIR": str(adapter), "MCP_ENDPOINT": "http://x/v1", "MCP_MODEL": "m",
           "GEMINI_API_KEY": PLACEHOLDER}
    script = os.path.join(_DOCKER_DIR, "mcp_bench_entrypoint.sh")
    proc = subprocess.run(["bash", script], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0 and "GBENCH_GEMINI_RELAY_URL is required" in proc.stderr
    env["GBENCH_GEMINI_RELAY_URL"] = "http://172.17.0.1:40001"
    proc = subprocess.run(["bash", script], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0 and "launched" in proc.stdout


# --------------------------------------------------------------------------- #
# mcp_bench launcher: the network check and the per-task harness record
# --------------------------------------------------------------------------- #
class _Conn:
    def close(self):
        pass


def test_the_network_check_has_no_ip_only_test(monkeypatch):
    monkeypatch.delenv("MCP_ASSUME_NETWORK", raising=False)
    monkeypatch.setenv("MCP_DNS_NAMES", "api.nasa.gov,en.wikipedia.org")
    connects = []

    def no_dns(name, *a, **k):
        raise socket.gaierror(-3, "Temporary failure in name resolution")
    monkeypatch.setattr(RUN.socket, "getaddrinfo", no_dns)
    monkeypatch.setattr(RUN.socket, "create_connection",
                        lambda addr, timeout=None: connects.append(addr) or _Conn())
    check = RUN.network_check()
    assert check["up"] is False and check["egress"] is False
    assert check["unresolved"] == {"api.nasa.gov": "gaierror:-3",
                                   "en.wikipedia.org": "gaierror:-3"}
    assert connects == []            # no connection to an address stands in for the names
    assert RUN._network_up() is False and RUN.LAST_NETWORK_CHECK["up"] is False


def test_the_network_check_needs_every_name_and_one_connection(monkeypatch):
    monkeypatch.delenv("MCP_ASSUME_NETWORK", raising=False)
    monkeypatch.setenv("MCP_DNS_NAMES", "a.example.org, b.example.org")
    monkeypatch.setattr(RUN.socket, "getaddrinfo", lambda name, *a, **k: [("ok",)])
    monkeypatch.setattr(RUN.socket, "create_connection", lambda addr, timeout=None: _Conn())
    assert RUN.network_check() == {"up": True, "assumed": False, "names": 2, "unresolved": {},
                                   "egress": True}

    def refused(addr, timeout=None):
        raise ConnectionRefusedError(111, "refused")
    monkeypatch.setattr(RUN.socket, "create_connection", refused)
    assert RUN.network_check()["up"] is False
    monkeypatch.setenv("MCP_ASSUME_NETWORK", "1")
    monkeypatch.setattr(RUN.socket, "getaddrinfo",
                        lambda *a, **k: pytest.fail("nothing is looked up when assumed"))
    assert RUN.network_check()["up"] is True and RUN.LAST_NETWORK_CHECK["assumed"] is True


def test_the_probe_reports_the_network_check():
    assert "check = dict(R.LAST_NETWORK_CHECK)" in MB._PROBE
    assert "'network_check': check" in MB._PROBE


def _completed(task_id, **extra):
    return dict({"task_id": task_id, "status": "completed", "evaluation": {"x": 1},
                 "execution_results": []}, **extra)


@pytest.mark.parametrize("result,kinds", [
    (_completed("t"), []),
    (_completed("t", evaluation=None), ["judge_outage"]),
    ({"task_id": "t", "status": "failed", "error": "No tools discovered from any server"},
     ["task_failed"]),
    (None, ["task_raised"]),
    # a resolver that did not answer counts when the text names a server API host or the
    # resolver; a model-chosen name (SERVFAIL maps to EAI_AGAIN) or a successful tool result
    # quoting such text is the model's outcome
    (_completed("t", execution_results=[{"error": "getaddrinfo EAI_AGAIN api.example.com"}]), []),
    (_completed("t", execution_results=[{"error": "getaddrinfo EAI_AGAIN api.nasa.gov"}]),
     ["dns_failure"]),
    (_completed("t", execution_results=[{"result": "connection timed out; no servers could be "
                                                   "reached"}]), []),
    (_completed("t", execution_results=[{"error": "connection timed out; no servers could be "
                                                  "reached (10.9.9.9#53)"}]), ["dns_failure"]),
    (_completed("t", execution_results=[{"error": "getaddrinfo ENOTFOUND api.nasa.gov"}]),
     ["dns_failure"]),
    # a name the model chose that does not exist is the model's outcome
    (_completed("t", execution_results=[{"error": "getaddrinfo ENOTFOUND made-up.example"}]), []),
    (_completed("t", execution_results=[{"result": 'Failed to resolve "bogus.example".'}]), []),
])
def test_the_harness_flags_each_kind(result, kinds, tmp_path):
    harness = RUN.HarnessAccounting(["api.nasa.gov", "en.wikipedia.org"], resolvers=["10.9.9.9"],
                                    bridge_failures=str(tmp_path / "none.jsonl"))
    rec = harness.begin("t", "/app/mcp-bench/tasks/single.json")
    harness.end(rec, result)
    assert rec["kinds"] == kinds and rec["file"] == "single.json"
    assert harness.scored([{"task_id": "t"}]) == ([] if kinds else [{"task_id": "t"}])


def test_the_watchdog_flags_an_outage_on_the_running_task(monkeypatch):
    names = ["a.org", "b.org", "c.org", "d.org"]
    harness = RUN.HarnessAccounting(names)
    rec = harness.begin("t1")
    monkeypatch.setattr(RUN, "resolve_all", lambda ns, timeout_s=None: {
        n: ("gaierror:-2" if n == "a.org" else None) for n in ns})
    harness.watch_once()                    # one name of four: recorded, no outage
    assert rec["kinds"] == [] and harness.dns_name_failures == {"a.org": 1}
    monkeypatch.setattr(RUN, "resolve_all", lambda ns, timeout_s=None: {
        n: ("gaierror:-3" if n in ("a.org", "b.org") else None) for n in ns})
    harness.watch_once()                    # half of them: an outage while t1 runs
    harness.end(rec, _completed("t1"))
    assert rec["kinds"] == ["dns_outage"] and harness.dns_rounds == 2
    [outage] = harness.dns_outages
    assert outage["failed"] == 2 and outage["of"] == 4
    other = harness.begin("t2")
    harness.end(other, _completed("t2"))
    assert other["kinds"] == []
    log = harness.record()
    assert log["version"] == RUN.HARNESS_LOG_VERSION and [t["task_id"] for t in log["tasks"]] == [
        "t1", "t2"]
    assert MB.harness_accounting(log)["tasks_scored"] == 1


def test_the_hooks_keep_flagged_tasks_out_of_upstreams_aggregation():
    seen = []

    class Runner:
        tasks_file = "/app/mcp-bench/tasks/a.json"

        async def execute_single_task_with_model(self, task_info, *a, **k):
            tid = task_info["task"]["task_id"]
            if tid == "t3":
                Judge.outage_hook()          # one stability run's judge call failed
            return _completed(tid, evaluation=None if tid == "t2" else {"x": 1})

    class Aggregator:
        def aggregate_current_metrics(self, results):
            seen.append([r["task_id"] for r in results])
            return {}

    class Judge:
        outage_hook = None

    harness = RUN.install_harness_accounting(Runner, Aggregator, Judge,
                                             RUN.HarnessAccounting(["x.org"]))
    runner, results = Runner(), []

    async def go():
        for tid in ("t1", "t2", "t3"):
            results.append(await runner.execute_single_task_with_model({"task": {"task_id": tid}}))
            Aggregator().aggregate_current_metrics(results)
    asyncio.run(go())
    assert seen == [["t1"], ["t1"], ["t1", "t3"]]
    assert [t["kinds"] for t in harness.tasks] == [[], ["judge_outage"], []]
    assert [t["judge_calls_failed"] for t in harness.tasks] == [0, 0, 1]
    assert all(t["file"] == "a.json" for t in harness.tasks)


def test_an_unrecovered_endpoint_failure_flags_its_task(monkeypatch, tmp_path):
    class Unavailable(Exception):
        status_code = 503

    class Completions:
        def __init__(self, script):
            self.script = list(script)

        async def create(self, **kw):
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                finish_reason="stop", message=types.SimpleNamespace(content=item,
                                                                    tool_calls=None))])

    def provider(script):
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=Completions(script)), api_key="dummy")

        async def get_completion(system, user, max_tokens):
            last = None
            for _ in range(3):                 # upstream's 3 attempts
                try:
                    resp = await client.chat.completions.create(model="m", messages=[],
                                                                max_tokens=max_tokens)
                    content = resp.choices[0].message.content
                    if not content:
                        raise ValueError("Empty content received from LLM")
                    return content
                except Exception as e:
                    last = e
            raise last
        return types.SimpleNamespace(client=client, get_completion=get_completion)

    scripts = {"fails": [Unavailable("503")] * 3, "recovers": [Unavailable("503"), "ok"],
               "empty": ["", "", ""]}

    class LLMFactory:
        @staticmethod
        async def create_llm_provider(model_config):
            return provider(scripts[model_config.script])

    factory = types.ModuleType("llm.factory")
    factory.LLMFactory = LLMFactory
    monkeypatch.setitem(sys.modules, "llm", types.ModuleType("llm"))
    monkeypatch.setitem(sys.modules, "llm.factory", factory)
    monkeypatch.setenv("MCP_ENABLE_THINKING", "false")
    monkeypatch.setenv("MCP_REQUEST_LOG", str(tmp_path / "req.json"))
    for k in ("MCP_MAX_TOKENS", "MCP_TEMPERATURE", "MCP_REQUEST_TIMEOUT_S",
              "LLAMA_3_1_8B_BASE_URL", "MCP_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MCP_MODEL_CONFIG_NAME", "slot")
    harness = RUN.HarnessAccounting([])
    RUN.install_request_policy(harness)

    async def run(kind):
        rec = harness.begin(kind)
        p = await LLMFactory.create_llm_provider(types.SimpleNamespace(name="slot", script=kind))
        try:
            await p.get_completion("s", "u", 10)
        except Exception:
            pass
        harness.end(rec, _completed(kind))
        return rec["kinds"]
    assert asyncio.run(run("fails")) == ["model_endpoint"]
    assert asyncio.run(run("recovers")) == []        # upstream's retry got past it
    assert asyncio.run(run("empty")) == []           # the model's own empty reply


# =========================================================================================== #
# mcp_atlas: the credentials in a container of their own
# =========================================================================================== #
_SHELL_CODE_FILE = ("cli-mcp-server", "mcp-code-executor", "mcp-server-code-runner",
                    "desktop-commander", "filesystem", "git", "memory", "fetch")


def test_the_servers_are_split_by_whether_they_need_a_credential():
    assert set(M.CREDENTIALED_SERVERS) | set(M.KEYLESS_SERVERS) == set(M.ENVIRONMENT_SERVERS)
    assert not set(M.CREDENTIALED_SERVERS) & set(M.KEYLESS_SERVERS)
    assert all(M.ENVIRONMENT_SERVERS[s] for s in M.CREDENTIALED_SERVERS)
    assert not any(M.ENVIRONMENT_SERVERS[s] for s in M.KEYLESS_SERVERS)
    assert set(_SHELL_CODE_FILE) <= set(M.KEYLESS_SERVERS)       # the model's shell and code
    # 20 keyless: ddg-search runs in the environment container for its fetch_content; 15 with a
    # credential: exa and oxylabs run there for their page tools
    assert len(M.CREDENTIALED_SERVERS) == 15 and len(M.KEYLESS_SERVERS) == 20
    assert {"exa", "oxylabs"} <= set(M.CREDENTIALED_SERVERS)
    order = list(M.RELEASE_SERVERS)
    for group in (M.CREDENTIALED_SERVERS, M.KEYLESS_SERVERS):
        assert list(group) == sorted(group, key=order.index)
    assert M.CREDENTIALED_COMMAND[-4:] == ("--host", "127.0.0.1", "--port", "1984")


def _atlas_credentials(monkeypatch, value="cred-secret-value-9876"):
    for server in M.CREDENTIALED_SERVERS:
        for var in M.ENVIRONMENT_SERVERS[server]:
            if var != "MONGODB_CONNECTION_STRING":
                monkeypatch.setenv(var, f"{var}-{value}")


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


def test_only_the_credentialed_container_gets_the_credentials(monkeypatch):
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    _atlas_credentials(monkeypatch)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    monkeypatch.setattr(CG, "require_metadata_blocked", lambda *a, **k: CG.UNREACHABLE)
    sb = M._start_container("gbench-mcp-atlas:f24ba3fb")
    runs = [c for c in docker.cmds if c[:3] == ["docker", "run", "-d"]]
    [cred] = [c for c in runs if sb.cred_container in c]
    [env] = [c for c in runs if c[c.index("--name") + 1] == sb.container]
    variables = sorted({v for s in M.CREDENTIALED_SERVERS for v in M.ENVIRONMENT_SERVERS[s]
                        if v != "MONGODB_CONNECTION_STRING"})
    env_names = [cred[i + 1] for i, a in enumerate(cred) if a == "-e"]
    for var in variables:
        assert var in env_names, var                           # by name
        assert var not in env and not any(a.startswith(var) for a in env)
    assert f"ENABLED_SERVERS={','.join(M.CREDENTIALED_SERVERS)}" in cred
    assert f"ENABLED_SERVERS={','.join(M.KEYLESS_SERVERS)}" in env
    assert "cred-secret-value-9876" not in " ".join(cred + env)
    # the credentialed servers: no published port, the API on the container's loopback
    assert "-p" not in cred and cred[-len(M.CREDENTIALED_COMMAND):] == list(M.CREDENTIALED_COMMAND)
    assert cred[cred.index("--network") + 1] == sb.cred_network != sb.network
    assert env[env.index("-p") + 1].startswith("127.0.0.1:")
    assert env[env.index("--network") + 1] == sb.network
    assert sb.cred_servers == M.CREDENTIALED_SERVERS
    assert set(sb.metadata_guard) == {"environment", "credentialed_servers"}


def test_an_unset_credential_stops_the_run_before_anything_starts(monkeypatch):
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    _atlas_credentials(monkeypatch)
    monkeypatch.delenv("AIRTABLE_API_KEY")
    monkeypatch.delenv("SLACK_MCP_XOXD_TOKEN")
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    with pytest.raises(RuntimeError) as ei:
        M._start_container("gbench-mcp-atlas:f24ba3fb")
    msg = str(ei.value)
    assert "2 of the 15 MCP servers that need credentials" in msg
    assert "airtable: AIRTABLE_API_KEY unset (it also needs the release's Airtable base" in msg
    assert "slack: SLACK_MCP_XOXD_TOKEN unset (" in msg and "mongodb" not in msg
    assert docker.cmds == []
    assert M.credential_gaps() == {"airtable": ["AIRTABLE_API_KEY"],
                                   "slack": ["SLACK_MCP_XOXD_TOKEN"]}


def test_the_catalogs_merge_in_template_order():
    env = [{"name": "arxiv_search"}, {"name": "calculator_calculate"},
           {"name": "weather_find"}, {"name": "wikipedia_search"}]
    cred = [{"name": "airtable_list"}, {"name": "github_a"}, {"name": "github_b"},
            {"name": "weather-data_now"}]
    merged = [t["name"] for t in M.merge_catalogs(env, cred)]
    assert merged == ["airtable_list", "arxiv_search", "calculator_calculate", "github_a",
                      "github_b", "weather_find", "weather-data_now", "wikipedia_search"]


def test_the_exec_reply_is_read_like_an_http_reply():
    assert M._exec_reply(0, '[{"type":"text","text":"x"}]\n200', "", "c") == (
        200, '[{"type":"text","text":"x"}]')
    assert M._exec_reply(0, '{"detail":"no"}\n500', "", "c") == (500, '{"detail":"no"}')
    with pytest.raises(TimeoutError):
        M._exec_reply(28, "\n000", "curl: (28) Operation timed out", "c")
    with pytest.raises(OSError, match="could not be reached"):
        M._exec_reply(1, "", "Error response from daemon: container c is not running", "c")
    with pytest.raises(OSError):
        M._exec_reply(7, "\n000", "curl: (7) Failed to connect", "c")
    argv = M._exec_argv("c", "POST", "/call-tool", 60, True)
    assert argv[:4] == ["docker", "exec", "-i", "c"] and argv[-1] == "http://127.0.0.1:1984/call-tool"
    assert "--data-binary" in argv and "@-" in argv


class _Exec:
    """Stands in for asyncio.create_subprocess_exec of `docker exec ... curl`."""

    def __init__(self, reply=('[{"type": "text", "text": "from cred"}]', 200), rc=0, err=""):
        self.calls = []
        self.reply, self.rc, self.err = reply, rc, err

    async def __call__(self, *argv, **kw):
        outer = self

        class P:
            returncode = outer.rc

            async def communicate(self, data=None):
                outer.calls.append((list(argv), json.loads(data) if data else None))
                body, status = outer.reply
                return f"{body}\n{status}".encode(), outer.err.encode()

            def kill(self):
                pass
        return P()


@pytest.fixture
def env_server():
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            calls.append(json.loads(self.rfile.read(n) or b"{}"))
            data = json.dumps([{"type": "text", "text": "from env"}]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv, url = _serve(H)
    yield calls, url
    srv.shutdown()
    srv.server_close()


def _call(sandbox, name, args):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(M.execute_mcp_tool(name, args, 0))
    finally:
        M._ACTIVE_SANDBOX.reset(token)


def test_calls_go_to_the_container_that_runs_their_server(monkeypatch, env_server):
    calls, url = env_server
    exe = _Exec()
    monkeypatch.setattr(M.asyncio, "create_subprocess_exec", exe)
    sb = M._Sandbox(url, container="env", cred_container="cred",
                    cred_servers=M.CREDENTIALED_SERVERS)
    assert _call(sb, "github_list_commits", {"owner": "o"}) == "from cred"
    assert _call(sb, "MongoDB_find", {"collection": "c"}) == "from cred"   # main.py's mapping
    assert _call(sb, "calculator_calculate", {"expression": "1+1"}) == "from env"
    assert [c[1]["tool_name"] for c in exe.calls] == ["github_list_commits", "MongoDB_find"]
    assert exe.calls[0][0][:4] == ["docker", "exec", "-i", "cred"]
    assert calls == [{"tool_name": "calculator_calculate", "tool_args": {"expression": "1+1"}}]


def test_a_lost_credentialed_container_is_a_harness_error(monkeypatch):
    monkeypatch.setattr(M.asyncio, "create_subprocess_exec",
                        _Exec(reply=("", ""), rc=1, err="Error: container cred is not running"))
    monkeypatch.setattr(M, "_container_running", lambda name: False)
    sb = M._Sandbox("http://127.0.0.1:9", container="env", cred_container="cred",
                    cred_servers=M.CREDENTIALED_SERVERS)
    out = _call(sb, "github_list_commits", {})
    assert out.startswith("Error: Failed to call tool github_list_commits:")
    assert sb.transport_errors == 1 and sb.harness_error_samples == {0: 1}
    assert "credentialed-servers container cred is no longer running" in sb.lost


def test_a_shell_injecting_lara_import_is_answered_not_run(monkeypatch):
    exe = _Exec()
    monkeypatch.setattr(M.asyncio, "create_subprocess_exec", exe)
    sb = M._Sandbox("http://127.0.0.1:9", container="env", cred_container="cred",
                    cred_servers=M.CREDENTIALED_SERVERS)
    for url in ('http://x/a.tmx"; env; "', "https://x/$(cat /proc/1/environ)",
                "-K/proc/self/environ", "https://x/a b", "https://x/`id`", "file:///etc/passwd"):
        out = _call(sb, "lara-translate_import_tmx", {"id": "mem_1", "tmx_url": url})
        assert json.loads(out) == {"detail": "Failed to call tool 'lara-translate_import_tmx': "
                                             + M.LARA_IMPORT_TMX_REFUSAL}, url
    assert exe.calls == [] and sb.refused_calls == {"lara-translate_import_tmx": 6}
    # a plain URL, or content (the server raises before its curl), is passed on as upstream
    _call(sb, "lara-translate_import_tmx", {"id": "mem_1", "tmx_url": "https://x.org/a.tmx?b=1"})
    _call(sb, "lara-translate_import_tmx", {"id": "m", "tmx_url": '"; id', "tmx_content": "<tmx/>"})
    _call(sb, "lara-translate_translate", {"text": [{"text": "$(id)"}], "target": "fr"})
    assert len(exe.calls) == 3


def test_the_preflight_covers_both_containers(monkeypatch):
    env_servers = [[s, "OK"] for s in M.KEYLESS_SERVERS]
    cred_servers = [[s, "OK"] for s in M.CREDENTIALED_SERVERS]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            data = json.dumps({"servers": env_servers, "total": 19, "online": 19}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv, url = _serve(H)
    execs = []

    def exec_http(container, method, path, timeout, body=None):
        execs.append((container, method, path))
        return 200, json.dumps({"servers": cred_servers, "total": 13, "online": 13})

    health = {}

    async def health_calls(u, cred_container=None, cred_servers=()):
        health.update(url=u, cred=cred_container, servers=cred_servers)
        return {s: (True, "ok") for s in M.RELEASE_SERVERS}
    monkeypatch.setattr(M, "_exec_http", exec_http)
    monkeypatch.setattr(M, "_run_health_calls", health_calls)
    try:
        sb = M._Sandbox(url, container="env", cred_container="cred",
                        cred_servers=M.CREDENTIALED_SERVERS)
        got = M._preflight(sb)
    finally:
        srv.shutdown()
        srv.server_close()
    assert execs == [("cred", "GET", "/enabled-servers")]
    assert got["servers_configured"] == 32 and got["servers_online"] == 32
    assert health == {"url": url, "cred": "cred", "servers": M.CREDENTIALED_SERVERS}


def test_the_health_calls_of_credentialed_servers_go_through_exec(monkeypatch, env_server):
    calls, url = env_server
    execs = []

    async def exec_async(container, method, path, timeout, body=None):
        execs.append(body["tool_name"])
        return 200, json.dumps([{"type": "text", "text": "fine"}])
    monkeypatch.setattr(M, "_exec_http_async", exec_async)
    monkeypatch.setattr(M, "gemini_search_reply", lambda *a, **k: (
        200, json.dumps([{"type": "text", "text": "Title: t"}]), ["gemini:x"]))
    asyncio.run(M._run_health_calls(url, "cred", M.CREDENTIALED_SERVERS))
    assert sorted(execs) == sorted(M.HEALTH_CHECK_CALLS[s][0] for s in M.CREDENTIALED_SERVERS)
    # ddg-search's check call is its search tool: gbench's grounding answers it
    assert sorted(c["tool_name"] for c in calls) == sorted(
        M.HEALTH_CHECK_CALLS[s][0] for s in M.KEYLESS_SERVERS
        if s not in M.GEMINI_SEARCH_SERVERS)


def test_a_stopped_credentialed_container_means_a_lost_environment(monkeypatch):
    monkeypatch.setattr(M, "_container_running", lambda name: name != "cred")
    sb = M._Sandbox("http://127.0.0.1:9", container="env", cred_container="cred")
    alive, why = M.environment_alive(sb)
    assert not alive and "credentialed-servers container cred is no longer running" in why


# --------------------------------------------------------------------------- #
# mcp_atlas: an environment gbench did not start
# --------------------------------------------------------------------------- #
def _external(servers, probe_replies):
    """A stand-in external environment: /enabled-servers lists `servers`; each /call-tool
    answers the next of `probe_replies` ((status, text))."""
    calls = []
    replies = list(probe_replies)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, text):
            data = text.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, '{"message": "ok"}')
            return self._send(200, json.dumps({"servers": [[s, "OK"] for s in servers],
                                               "total": len(servers), "online": len(servers)}))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            calls.append(json.loads(self.rfile.read(n) or b"{}"))
            return self._send(*replies.pop(0))
    srv, url = _serve(H)
    return srv, url, calls


def _code_reply(stdout):
    return 200, json.dumps([{"type": "text", "text": json.dumps(
        {"status": "success", "output": stdout, "file_path": "/data/x.py"})}])


def test_an_external_environment_with_credentialed_servers_is_refused():
    srv, url, calls = _external(list(M.KEYLESS_SERVERS) + ["github", "exa"], [])
    try:
        with pytest.raises(RuntimeError) as ei:
            M.check_external_environment(M._Sandbox(url))
    finally:
        srv.shutdown()
        srv.server_close()
    msg = str(ei.value)
    assert "configured with servers that need credentials (github, exa)" in msg
    assert "ENABLED_SERVERS=" + ",".join(M.KEYLESS_SERVERS) in msg
    assert calls == []                              # no probe, nothing run there


def test_the_external_metadata_verdict_comes_from_its_code_tool():
    creds = _code_reply("GBENCH_CREDENTIAL_PROBE=none proc1:read\n")
    reply = _code_reply("GBENCH_METADATA_PROBE=unreachable timeout ipv6:101\n")
    srv, url, calls = _external(list(M.KEYLESS_SERVERS) + ["ddg-search"], [creds, reply])
    try:
        sb = M._Sandbox(url)
        M.check_external_environment(sb)
    finally:
        srv.shutdown()
        srv.server_close()
    # first the credential check, then the metadata probe, both through the code tool, uncached
    cred_call, call = calls
    assert cred_call["tool_args"] == {"code": M.credential_probe_script()}
    assert call["tool_name"] == "mcp-code-executor_execute_code" and call["use_cache"] is False
    assert call["tool_args"] == {"code": CG.python_probe_script()}
    assert "mcp-code-executor_execute_code" in sb.metadata_guard["environment"]
    assert "/proc/1/environ read" in sb.credential_check


def test_a_reachable_metadata_server_in_the_external_environment_stops_the_run():
    for line, rule in (("GBENCH_METADATA_PROBE=reachable ipv4", CG.FIREWALL_COMMAND),
                       ("GBENCH_METADATA_PROBE=reachable ipv6", CG.FIREWALL_COMMAND6)):
        srv, url, _ = _external(list(M.KEYLESS_SERVERS), [_code_reply(line + "\n")])
        try:
            with pytest.raises(RuntimeError) as ei:
                M.probe_external_metadata(M._Sandbox(url))
        finally:
            srv.shutdown()
            srv.server_close()
        msg = str(ei.value)
        assert rule in msg and "GBENCH_MCP_ATLAS_SANDBOX_URL" in msg
        assert "does not filter the host network" in msg and "never runs sudo" in msg


def test_no_verdict_from_either_code_tool_fails_closed():
    srv, url, calls = _external(list(M.KEYLESS_SERVERS), [
        (500, '{"detail": "Unknown tool"}'),
        _code_reply("GBENCH_METADATA_PROBE=error 13\n")])
    try:
        with pytest.raises(RuntimeError, match="could not establish.*no verdict"):
            M.probe_external_metadata(M._Sandbox(url))
    finally:
        srv.shutdown()
        srv.server_close()
    assert [c["tool_name"] for c in calls] == list(M.EXTERNAL_PROBE_TOOLS)
    assert calls[1]["tool_args"] == {"languageId": "python", "code": CG.python_probe_script()}


def test_the_second_code_tool_can_give_the_verdict():
    srv, url, _ = _external(list(M.KEYLESS_SERVERS), [
        (200, json.dumps([{"type": "text", "text": "no python here"}])),
        (200, json.dumps([{"type": "text", "text": "GBENCH_METADATA_PROBE=unreachable 111"}]))])
    try:
        assert M.probe_external_metadata(M._Sandbox(url)) == "mcp-server-code-runner_run-code"
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_external_setup_order_and_its_cleanup(monkeypatch):
    order = []
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(M, "_require_credentials", lambda: order.append("credentials"))
    monkeypatch.setattr(M, "_wait_ready", lambda sb, t: order.append(("ready", t)))
    monkeypatch.setattr(M, "check_external_environment", lambda sb: order.append("checked"))
    monkeypatch.setattr(M, "_start_credentialed_side", lambda image, base: order.append(
        "credentialed") or {"cred_container": base + "-cred", "cred_network": base + "-cred-net",
                            "mongo_container": None, "cred_servers": M.CREDENTIALED_SERVERS})
    sb = M._open_sandbox()
    assert order == ["credentials", ("ready", M.HEALTH_TIMEOUT_S), "checked", "credentialed"]
    assert sb.container is None and sb.cred_container.endswith("-cred")
    assert sb.cred_servers == M.CREDENTIALED_SERVERS and "credentialed_servers" in sb.metadata_guard
    # a failed check starts nothing and leaves nothing behind
    removed = []

    def refuse(sb):
        raise RuntimeError("[INFRA REQUIRED] mcp_atlas: refused")
    monkeypatch.setattr(M, "check_external_environment", refuse)
    monkeypatch.setattr(M, "_start_credentialed_side",
                        lambda *a: pytest.fail("nothing starts after a refused check"))
    monkeypatch.setattr(M, "_stop_container", lambda sb: removed.append(sb.cred_container))
    with pytest.raises(RuntimeError, match="refused"):
        M._open_sandbox()
    assert removed == [None]


# =========================================================================================== #
# Docs follow the code
# =========================================================================================== #
def test_the_mcp_bench_guide_states_the_relay_the_names_and_the_harness_record():
    doc = _flat(_read("docs/evals/mcp_bench.md"))
    for needle in ("The judge's Gemini key never enters the container",
                   "`POST /v1beta/openai/chat/completions`", "`GBENCH_GEMINI_RELAY_URL`",
                   "There is no test by IP address alone",
                   f"`{MB.HARNESS_LABEL}={MB.HARNESS_VERSION}`",
                   "`/out/harness_log.json`", "A DNS outage is found here, before the run",
                   "`dns_names_checked`", "`gemini_relay`", "`tasks_scored`", "`tasks_unscored`",
                   "When no task is left to score, the run hard-errors",
                   "`model_endpoint`", "`judge_outage`", "`dns_failure`", "`dns_outage`",
                   "`task_failed`", "`task_raised`", "every 30 s", "within 15 s"):
        assert needle in doc, needle
    assert "8.8.8.8" not in doc
    assert "gbench does not pass `GEMINI_OPENAI_BASE_URL` into the container" not in doc
    assert f"{len(MB.SERVER_API_HOSTS) - 5} more" in doc
    assert RUN.DNS_WATCH_INTERVAL_S == 30 and RUN.DNS_LOOKUP_TIMEOUT_S == 15


def test_the_mcp_atlas_guide_states_the_split_and_the_external_checks():
    raw = _read("docs/evals/mcp_atlas.md")
    doc = _flat(raw)
    for needle in ("Two containers, and the credentials in only one of them",
                   "`KEYLESS_SERVERS`", "`CREDENTIALED_SERVERS`", "`docker exec`",
                   "`use_cache: false`", "`tool_calls.refused_by_gbench`",
                   "`agent_environment.metadata_guard`", "`credentialed_servers`",
                   "must list no server that needs a credential",
                   "The check never depends on a knob", M.LARA_IMPORT_TMX_REFUSAL):
        assert needle in doc, needle
    assert "ENABLED_SERVERS=" + ",".join(M.KEYLESS_SERVERS) in raw
    for server in M.CREDENTIALED_SERVERS + M.KEYLESS_SERVERS:
        assert server in doc, server
    assert "as in the release's own setup" not in doc
    assert "the check does not run" not in doc and "gbench then starts no container" not in doc
    assert CG.FIREWALL_COMMAND in raw


def test_the_changed_files_are_plain_ascii_without_internal_labels():
    rels = ("gbench/runners/eval_suites/mcp_bench.py", "gbench/runners/eval_suites/mcp_atlas.py",
            "gbench/docker/mcp_bench_run.py", "gbench/docker/mcp_bench_cascade_judge.py",
            "gbench/docker/mcp_bench_entrypoint.sh", "gbench/docker/mcp_bench.Dockerfile",
            "docker/mcp_atlas.Dockerfile", "docs/evals/mcp_bench.md", "docs/evals/mcp_atlas.md",
            "tests/test_mcp_credential_isolation_and_dns.py")
    for rel in rels:
        text = _read(rel)
        assert text.isascii(), rel
        low = text.lower()
        # built from parts, so this file does not hold them
        for label in ("wave" + " 2", "wave" + "2", "the " + "reviewer", "coordi" + "nator",
                      "integ" + "rator"):
            assert label not in low, (rel, label)
