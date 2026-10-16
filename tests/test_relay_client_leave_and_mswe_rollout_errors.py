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

"""A client that leaves stops the endpoint's reply; an MSWE-agent rollout that its parser ends on
a reasoning-only reply is the model's outcome.

* EndpointRelay: a client that closes its connection while the relay waits for a non-streamed
  reply (the endpoint answers only when it has generated all of it, as vLLM does), or while the
  relay reads a reply's body, makes the relay close the endpoint's connection at once (the
  endpoint stops generating) and counts it in client_closed_streams, not as an endpoint failure;
  normal replies are relayed unchanged and a client that closes after reading a whole stream is
  not counted.
* multi_swe_bench: the driver records the exception an instance's rollout raised (type, message,
  model turns, last reply kind); mswe_agent_trace keeps a rollout ended by a reply with no content
  (content None) as the model's outcome (crashed_empty_reply, unresolved, no patch, counted in
  empty_reply_crashes), and calls an instance a setup failure only when its setup failed.
* MSWE-agent's OpenAI client gets a timeout above the relay's per-turn limit (instead of the
  SDK's 600 s), passed in the spec, reported by the driver and recorded in the result.

Hermetic: loopback stubs only; docker and the endpoint are stubbed; files under tmp_path.
"""

import http.client
import http.server
import importlib.util
import json
import os
import select
import socket
import socketserver
import sys
import threading
import time
import types

import pytest

from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swebench_common as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================================
# EndpointRelay: the client leaves while the relay waits or reads
# ============================================================================================
def _client_gone(sock) -> bool:
    """What an ASGI server sees: the connection's end of file (or reset) while it generates."""
    r, _, _ = select.select([sock], [], [], 0)
    if not r:
        return False
    try:
        return not sock.recv(1, socket.MSG_PEEK)
    except OSError:
        return True


def _slow_upstream(state, mode, gen_s=8.0):
    """mode "buffered": no headers until the whole reply is generated (vLLM non-streamed);
    "trickle": headers with a Content-Length, then the body a few bytes at a time."""
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state["started"] = time.time()
            body = json.dumps({"choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "x" * 200}}]}).encode()
            if mode == "trickle":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
            deadline = time.time() + gen_s
            sent = 0
            while time.time() < deadline:
                if _client_gone(self.connection):
                    state["disconnected"] = time.time()
                    return
                if mode == "trickle" and sent < len(body) - 10:
                    try:
                        self.wfile.write(body[sent:sent + 1])
                        self.wfile.flush()
                    except OSError:
                        state["disconnected"] = time.time()
                        return
                    sent += 1
                time.sleep(0.02)
            state["finished"] = True
            try:
                if mode == "buffered":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.wfile.write(body[sent:])
                self.wfile.flush()
            except OSError:
                pass

        def log_message(self, *a):
            pass
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _relay_conn(relay, timeout):
    host, port = relay.url.split("//")[1].split("/")[0].split(":")
    return http.client.HTTPConnection(host, int(port), timeout=timeout)


def _wait(pred, limit=10.0):
    deadline = time.time() + limit
    while not pred() and time.time() < deadline:
        time.sleep(0.02)
    return pred()


@pytest.mark.parametrize("mode", ["buffered", "trickle"])
def test_a_client_that_gives_up_on_a_non_streamed_reply_stops_the_endpoint(mode):
    state = {}
    srv = _slow_upstream(state, mode)
    relay = C.EndpointRelay(f"http://127.0.0.1:{srv.server_address[1]}/v1", "127.0.0.1",
                            name="leave-relay").start()
    try:
        conn = _relay_conn(relay, timeout=1.0)     # an SDK read timeout, scaled down
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps({"model": "m", "messages": [{"role": "user",
                                                                  "content": "q"}]}),
                     headers={"Content-Type": "application/json"})
        with pytest.raises((socket.timeout, TimeoutError)):
            resp = conn.getresponse()
            resp.read()                            # trickle: headers came, the body does not
        conn.close()
        closed_at = time.time()
        assert _wait(lambda: "disconnected" in state)
        assert _wait(lambda: relay.record()["client_closed_streams"] == 1)
    finally:
        relay.stop()
        srv.shutdown()
        srv.server_close()
    # the endpoint saw its client leave right after the agent's client did, mid-generation
    assert state["disconnected"] - closed_at < 1.5
    assert "finished" not in state
    rec = relay.record()
    assert rec["client_closed_streams"] == 1 and rec["upstream_errors"] == {}
    assert rec["completion_replies"] == 0 and C.relay_failure_count(rec) == 0


def _quick_upstream(stream=False):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i in range(3):
                    data = ("data: " + json.dumps({"choices": [{"index": 0, "delta": {
                        "content": f"t{i}"}, "finish_reason": "stop" if i == 2 else None}]})
                        + "\n\n").encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                done = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(done), done))
                self.wfile.flush()
                return
            body = json.dumps({"choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "hello"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("stream", [False, True])
def test_normal_replies_are_relayed_whole_and_not_counted_as_left(stream):
    srv = _quick_upstream(stream)
    relay = C.EndpointRelay(f"http://127.0.0.1:{srv.server_address[1]}/v1", "127.0.0.1",
                            name="ok-relay").start()
    try:
        for _ in range(3):
            conn = _relay_conn(relay, timeout=10)
            conn.request("POST", "/v1/chat/completions",
                         body=json.dumps({"model": "m", "stream": stream, "messages": []}),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            conn.close()                           # closes right after the whole reply
            assert resp.status == 200
            assert (b"[DONE]" in data) if stream else json.loads(data)["choices"][0][
                "message"]["content"] == "hello"
        time.sleep(3 * C._ClientWatch.POLL_S)
    finally:
        relay.stop()
        srv.shutdown()
        srv.server_close()
    rec = relay.record()
    assert rec["client_closed_streams"] == 0 and rec["upstream_errors"] == {}
    assert rec["completion_replies"] == 3 and rec["requests_relayed"] == 3


def test_an_endpoint_that_does_not_answer_is_still_a_connection_failure():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", name="down-relay").start()
    try:
        conn = _relay_conn(relay, timeout=10)
        conn.request("POST", "/v1/chat/completions", body=b"{}",
                     headers={"Content-Type": "application/json"})
        assert conn.getresponse().status == 502
        conn.close()
    finally:
        relay.stop()
    rec = relay.record()
    assert rec["upstream_errors"] == {"connection": 1} and rec["client_closed_streams"] == 0


# ============================================================================================
# multi_swe_bench: the driver records a rollout's exception
# ============================================================================================
def _driver():
    spec = importlib.util.spec_from_file_location(
        "mswe_agent_run_mswefix", os.path.join(REPO, "gbench", "docker", "mswe_agent_run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_sweagent(monkeypatch, reply):
    """sweagent.agent.{agents,models} stand-ins: Agent.run queries the model once and parses the
    reply the way MSWE-agent's parser does (re.finditer, TypeError on None)."""
    import re
    agents = types.ModuleType("sweagent.agent.agents")
    models = types.ModuleType("sweagent.agent.models")

    class OpenAIModel:
        def query(self, history):
            return reply

    class Agent:
        def __init__(self):
            self.model = OpenAIModel()

        def run(self, setup_args, env, *a, **k):
            self.forward_model("obs", "state", "c")
            out = self.model.query([])
            list(re.compile("```").finditer(out))
            return {}, []

        def forward_model(self, observation, state, lang):
            return ""
    agents.Agent, models.OpenAIModel = Agent, OpenAIModel
    pkg = types.ModuleType("sweagent")
    sub = types.ModuleType("sweagent.agent")
    sub.agents, sub.models, pkg.agent = agents, models, sub
    for name, mod in (("sweagent", pkg), ("sweagent.agent", sub),
                      ("sweagent.agent.agents", agents), ("sweagent.agent.models", models)):
        monkeypatch.setitem(sys.modules, name, mod)
    return Agent


def _env(iid):
    return types.SimpleNamespace(record=types.SimpleNamespace(data={"instance_id": iid}))


def test_the_driver_records_a_rollout_ended_by_a_reply_without_content(monkeypatch, tmp_path):
    d = _driver()
    Agent = _fake_sweagent(monkeypatch, None)
    d.install_step_limit()
    d.install_rollout_error_record()
    with pytest.raises(TypeError):
        Agent().run(None, _env("o__r-1"))
    rec = d.ROLLOUT_ERRORS["o__r-1"]
    assert rec["type"] == "TypeError" and rec["last_reply"] == "none" and rec["step"] == 1
    assert rec["error"].startswith("TypeError: ")
    out = d.collect(str(tmp_path), ["o__r-1"])["o__r-1"]
    assert out["completed"] is False and out["rollout_error"] == rec
    assert (d.reply_kind(None), d.reply_kind("  \n"), d.reply_kind("ls")) == (
        "none", "empty", "text")


def test_the_driver_qualifies_sdk_exceptions(monkeypatch):
    d = _driver()
    Agent = _fake_sweagent(monkeypatch, "ok")
    import openai
    err = openai.APITimeoutError(request=__import__("httpx").Request("POST", "http://x"))
    monkeypatch.setattr(Agent, "forward_model", lambda *a: (_ for _ in ()).throw(err))
    d.install_rollout_error_record()
    with pytest.raises(openai.APITimeoutError):
        Agent().run(None, _env("o__r-2"))
    assert d.ROLLOUT_ERRORS["o__r-2"]["error"].startswith("openai.APITimeoutError: ")
    assert d.ROLLOUT_ERRORS["o__r-2"]["last_reply"] == "unknown"


def test_the_driver_gives_the_client_the_spec_timeout(monkeypatch):
    d = _driver()
    _fake_sweagent(monkeypatch, "ok")
    import openai
    sys.modules["sweagent.agent.models"].OpenAI = openai.OpenAI
    rec = d.install_client_timeout(M.mswe_agent_timeouts().client_s)
    client = sys.modules["sweagent.agent.models"].OpenAI(api_key="EMPTY",
                                                         base_url="http://127.0.0.1:1/v1")
    assert client.timeout.read == rec["read_timeout_s"] == M.mswe_agent_timeouts().client_s
    assert rec["max_retries"] == client.max_retries     # the SDK's retries are unchanged
    # the driver's own default is only a fallback for a spec without a timeout; the run's spec
    # carries mswe_agent_timeouts().client_s (test_the_spec_carries_the_client_timeout)
    assert d.CLIENT_TIMEOUT_S >= C.EndpointRelay.TIMEOUT_S
    main = open(os.path.join(REPO, "gbench", "docker", "mswe_agent_run.py")).read()
    main = main.split("def main(")[1]
    assert "install_rollout_error_record()" in main and "install_client_timeout(" in main


# ============================================================================================
# multi_swe_bench: mswe_agent_trace
# ============================================================================================
def _task():
    return A.AgentTask("o__r-1", "task", "mswebench/o_m_r:pr-1", "c",
                       meta={"org": "o", "repo": "r", "number": 1})


def _crash_summary(**inst):
    return {"images": {"o__r-1": {"error": None}}, "instances": {"o__r-1": {
        "completed": False, "patch": "", "exit_status": None, "api_calls": 2, "steps": 2,
        "step_limit_reached": False, "trajectory_file": None, **inst}}}


_EMPTY = {"type": "TypeError", "error": "TypeError: expected string or bytes-like object",
          "message": "expected string or bytes-like object", "step": 3, "last_reply": "none"}


def test_a_reasoning_only_reply_crash_is_the_models_outcome(tmp_path):
    tr = M.mswe_agent_trace(0, _task(), _crash_summary(rollout_error=dict(_EMPTY)),
                            str(tmp_path), time.time())
    assert tr["agent_infra_error"] is False and tr["status"] == "OK"
    assert tr["agent_exit_status"] == "crashed_empty_reply" and tr["agent_patch"] == ""
    assert "setup" not in tr["agent_exit_note"] and "no content" in tr["agent_exit_note"]
    assert tr["agent_error"] is None


@pytest.mark.parametrize("crash,model", [
    ({"type": "KeyError", "error": "KeyError: 'x'", "message": "'x'", "step": 4,
      "last_reply": "text"}, True),
    ({"type": "APITimeoutError", "error": "openai.APITimeoutError: Request timed out.",
      "message": "Request timed out.", "step": 4, "last_reply": "pending"}, False),
    ({"type": "DockerException", "error": "docker.DockerException: x",
      "message": "Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
      "step": 4, "last_reply": "text"}, False),
    ({"type": "RuntimeError", "error": "RuntimeError: x", "message": "x", "step": 0,
      "last_reply": "unknown"}, False),
    # raised inside the model request: an endpoint answering 200 with an HTML page makes the
    # SDK return a str, and models.py fails on it - not the model's outcome
    ({"type": "AttributeError", "error": "AttributeError: 'str' object has no attribute 'usage'",
      "message": "'str' object has no attribute 'usage'", "step": 3,
      "last_reply": "pending"}, False),
    ({"type": "IndexError", "error": "IndexError: list index out of range",
      "message": "list index out of range", "step": 3, "last_reply": "pending"}, False),
    # MSWE-agent's own model-side limits stay the model's outcome
    ({"type": "ContextWindowExceededError", "error": "ContextWindowExceededError: x",
      "message": "x", "step": 9, "last_reply": "pending"}, True),
])
def test_other_rollout_crashes_follow_the_exit_error_rule(tmp_path, crash, model):
    tr = M.mswe_agent_trace(0, _task(), _crash_summary(rollout_error=crash), str(tmp_path),
                            time.time())
    assert tr["agent_infra_error"] is (not model)
    if model:
        assert tr["agent_exit_status"] == "crashed" and tr["agent_exit_note"]
    else:
        assert "its rollout raised" in tr["agent_error"]
        assert "setup failed" not in tr["agent_error"]


def test_setup_failures_and_unrecorded_ends_stay_infrastructure(tmp_path):
    setup = M.mswe_agent_trace(0, _task(), _crash_summary(
        setup_error={"step": "repo_reset", "error": "git clone failed"},
        rollout_error=dict(_EMPTY)), str(tmp_path), time.time())
    assert setup["agent_infra_error"] is True
    assert "its setup failed at repo_reset" in setup["agent_error"]
    bare = M.mswe_agent_trace(0, _task(), _crash_summary(), str(tmp_path), time.time())
    assert bare["agent_infra_error"] is True
    assert "recorded no setup or rollout error" in bare["agent_error"]
    assert "container or repository setup failed" not in bare["agent_error"]


def test_the_run_counts_empty_reply_crashes_and_scores_them(monkeypatch, tmp_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_swe_reference_scaffolds as H
    H._agentic_setup(monkeypatch, tmp_path, H._report((), (1, 2)))
    rollouts = M.run_mswe_agent

    def _with_crash(*a, **k):
        traces = rollouts(*a, **k)
        traces[0].update(agent_patch="", agent_exit_status="crashed_empty_reply",
                         agent_exit_note="no content")
        return traces
    monkeypatch.setattr(M, "run_mswe_agent", _with_crash)
    res = M.run_multi_swe_bench("m", "http://127.0.0.1:1/v1", 1, enable_thinking=True)
    assert res["empty_reply_crashes"] == 1 and len(res["empty_reply_crash_ids"]) == 1
    assert res["total_questions"] == 2 and res["scored_instances"] == 2
    assert res["agent"]["openai_client_timeout_s"] == M.mswe_agent_timeouts().client_s


def test_the_spec_carries_the_client_timeout(monkeypatch, tmp_path):
    rows = {"o__r-1": {"org": "o", "repo": "r", "number": 1, "instance_id": "o__r-1",
                       "resolved_issues": [{"title": "t", "body": "b"}]}}
    tasks = [A.AgentTask("o__r-1", "t", M.published_image("o", "r", 1), "go",
                         meta={"org": "o", "repo": "r", "number": 1})]
    monkeypatch.setattr(M, "host_pull_images", lambda images, workers=1: {
        i: {"state": "present"} for i in images})
    monkeypatch.setattr(M.swebench_common, "require_task_network_blocked", lambda *a, **k: None)
    specs = []

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            work = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            specs.append(json.load(open(os.path.join(work, "spec.json"))))
            json.dump({"images": {"o__r-1": {"error": None}}, "instances": {"o__r-1": {
                "completed": False, "api_calls": 2, "rollout_error": dict(_EMPTY)}},
                "openai_client": {"read_timeout_s": float(M.MSWE_AGENT_CLIENT_TIMEOUT_S)}},
                open(os.path.join(work, "summary.json"), "w"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(M.subprocess, "run", _run)
    out = tmp_path / "out"
    traces = M.run_mswe_agent(tasks, rows, "m", "http://127.0.0.1:5/v1", 1, str(out), 1.0, 0.95)
    assert specs[0]["client_timeout_s"] == M.mswe_agent_timeouts().client_s
    assert traces[0]["agent_exit_status"] == "crashed_empty_reply"
    assert not traces[0]["agent_infra_error"]
    assert M._client_records(str(out)) == {"go": {
        "read_timeout_s": float(M.MSWE_AGENT_CLIENT_TIMEOUT_S)}}


def test_the_image_carries_driver_4_and_the_docs_describe_it():
    dockerfile = open(os.path.join(REPO, "gbench", "docker", "mswe_agent.Dockerfile")).read()
    assert M.MSWE_AGENT_DRIVER_VERSION == "4"
    assert 'org.gbench.mswe_agent_driver="4"' in dockerfile
    assert 'grep -q "^def install_rollout_error_record"' in dockerfile
    assert "install_client_timeout(d.CLIENT_TIMEOUT_S)" in dockerfile
    doc = " ".join(open(os.path.join(REPO, "docs", "evals", "multi_swe_bench.md")).read().split())
    for needle in ("`crashed_empty_reply`", "`empty_reply_crashes`", "`rollout_error`",
                   "`openai_client_timeout_s`", "client_closed_streams"):
        assert needle in doc, needle


def test_old_driver_images_are_refused_with_the_rebuild_command(monkeypatch):
    monkeypatch.setattr(M.os.path, "exists", lambda p: True)
    labels = {"org.gbench.mswe_agent": M.MSWE_AGENT_COMMIT,
              "org.gbench.mswe_agent_config_sha256": M.MSWE_AGENT_CONFIG_SHA256,
              "org.gbench.mswe_agent_java_repos":
                  f"multi-swe-bench@{M.MSWE_AGENT_JAVA_REF}: {M.MSWE_AGENT_JAVA_ORGS}",
              M.MSWE_AGENT_DRIVER_LABEL: "3"}
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps(labels), stderr=""))
    ok, why = M.check_mswe_agent_prerequisites()
    assert not ok and "org.gbench.mswe_agent_driver='3'" in why
    assert "docker build -t gbench-mswe-agent:88217624 -f gbench/docker/mswe_agent.Dockerfile" \
        in why


def test_the_client_watch_works_past_fd_1024():
    """select() refuses fds >= 1024 and the watch used to stop silently there; a long gbench
    process can hold that many fds."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 1200:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))
    held = [socket.socket() for _ in range(1100)]
    try:
        client, relay_side = socket.socketpair()
        endpoint, endpoint_peer = socket.socketpair()
        assert relay_side.fileno() >= 1024
        watch = C._ClientWatch(relay_side)
        watch.attach(endpoint)
        client.close()                                  # the client leaves
        assert _wait(lambda: watch.aborted, 5.0)
        endpoint_peer.settimeout(5.0)
        assert endpoint_peer.recv(1) == b""             # the endpoint connection was shut down
        watch.stop()
        for s_ in (relay_side, endpoint, endpoint_peer):
            s_.close()
    finally:
        for s_ in held:
            s_.close()
