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

"""wildclawbench: a judge call that turns thinking off is sent at the lowest Gemini thinking level
the cascade accepts, and the mapping is recorded. mcp_bench: every credentialed-server session
that fails outside the model is recorded (the relay answers only once the credentialed side is
up, the stream ends with an end frame, the bridge records a failed start and a lost session), and
the relay's failures during the run and a stopped credentialed-servers container are a harness
error of the run. Docker, Gemini and the network are stubbed; files go to tmp only."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import mcp_bench_bridge as BRIDGE  # noqa: E402
import mcp_bench_credentialed as CRED  # noqa: E402
import mcp_bench_run as MRUN  # noqa: E402
import wildclawbench_cascade_judge as J  # noqa: E402

_REAL_POPEN = subprocess.Popen
_UPSTREAM_TASKS = os.path.join(os.path.dirname(_REPO), "gbench-prereqs", "WildClawBench", "tasks")


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


def _lines(path):
    return [json.loads(x) for x in open(path).read().splitlines()] if os.path.exists(path) else []


# =========================================================================== #
# wildclawbench: judge calls that turn thinking off
# =========================================================================== #
@pytest.mark.parametrize("body,off", [
    ({"thinking": {"type": "disabled"}}, True),
    ({"thinking": {"type": "DISABLED"}}, True),
    ({"reasoning_effort": "none"}, True),
    ({"reasoning": {"effort": "none"}}, True),
    ({"reasoning": {"enabled": False}}, True),
    ({"thinking": {"type": "enabled", "budget_tokens": 100}}, False),
    ({"reasoning_effort": "high"}, False),
    ({"reasoning": {"effort": "low"}}, False),
    ({"messages": []}, False),
    ("not a dict", False),
])
def test_the_ways_a_request_turns_thinking_off(body, off):
    assert J.thinking_off_requested(body) is off


def test_the_level_is_the_lowest_every_default_cascade_model_accepts():
    # gemini-3.8-flash / 3.7-flash / 3.1-pro-preview offer low, medium, high; 3.6 / 3.5-flash also
    # minimal: the one level all five accept is "low", and the host records the same value.
    assert J.THINKING_OFF_LEVEL == W.THINKING_OFF_LEVEL == "low"
    assert list(J._DEFAULT_JUDGE_CASCADE) == list(B.GEMINI_CASCADE)


@pytest.mark.skipif(not os.path.isdir(_UPSTREAM_TASKS), reason="needs the provisioned checkout")
def test_the_thinking_off_tasks_are_the_pinned_tasks_that_send_it():
    found = []
    for root, _dirs, files in os.walk(_UPSTREAM_TASKS):
        for name in files:
            if not name.endswith(".md"):
                continue
            with open(os.path.join(root, name), encoding="utf-8") as fh:
                text = fh.read()
            if 'extra_body={"thinking": {"type": "disabled"}}' in text:
                found.append(next(ln.split(":", 1)[1].strip() for ln in text.splitlines()
                                  if ln.startswith("id:")))
    assert sorted(found) == sorted(W.THINKING_OFF_TASKS)


@pytest.fixture
def gemini(monkeypatch, tmp_path):
    """The proxy's Gemini calls captured: {url, payload} per attempt; a model in `fail` answers
    with that HTTP status."""
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTKEY0000000000000000000000000000")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", ",".join(B.GEMINI_CASCADE))
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "0")
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    J.reset_usage()
    state = types.SimpleNamespace(sent=[], fail={}, wd=tmp_path)

    def post(url, payload, headers, timeout):
        state.sent.append({"url": url, "payload": json.loads(json.dumps(payload))})
        model = payload.get("model") or url.rsplit("/models/", 1)[-1].split(":")[0]
        if model in state.fail:
            raise urllib.error.HTTPError(url, state.fail[model], "x", {}, None)
        if url.endswith(":generateContent"):
            return {"candidates": [{"content": {"parts": [{"text": "grounded"}]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": '{"ok": 1}'}}]}
    monkeypatch.setattr(J, "_post_json", post)
    return state


_SAFETY_JUDGE = {"model": "openai/gpt-5.4", "temperature": 0, "max_tokens": 1000,
                 "response_format": {"type": "json_object"},
                 "thinking": {"type": "disabled"},
                 "messages": [{"role": "system", "content": "You are a deterministic grader."},
                              {"role": "user", "content": "grade"}]}


def test_a_thinking_off_judge_call_is_sent_at_level_low_to_every_model(gemini):
    gemini.fail = {B.GEMINI_CASCADE[0]: 503, B.GEMINI_CASCADE[1]: 404}
    J.cascade_chat_completion(dict(_SAFETY_JUDGE))
    sent = [s["payload"] for s in gemini.sent]
    assert [p["model"] for p in sent] == list(B.GEMINI_CASCADE[:3])
    for p in sent:
        assert p["reasoning_effort"] == "low" and "thinking" not in p
        # every other field of the task's request as it sent it
        assert p["temperature"] == 0
        # the task's cap plus the Gemini thinking allowance
        assert p["max_tokens"] == 1000 + J.GEMINI_THINKING_ALLOWANCE
        assert p["response_format"] == {"type": "json_object"}
        assert p["messages"] == _SAFETY_JUDGE["messages"]
    # the same request for every model, only the model differs
    assert len({json.dumps({k: v for k, v in p.items() if k != "model"}, sort_keys=True)
                for p in sent}) == 1


def test_a_request_that_keeps_thinking_gets_no_thinking_setting(gemini):
    J.cascade_chat_completion({"model": "x", "reasoning_effort": "high",
                               "thinking": {"type": "enabled", "budget_tokens": 64},
                               "messages": [{"role": "user", "content": "grade"}]})
    [p] = [s["payload"] for s in gemini.sent]
    assert "reasoning_effort" not in p and "thinking" not in p and "reasoning" not in p


def test_the_grounded_judge_gets_the_same_level():
    req = J.gemini_grounded_request({"thinking": {"type": "disabled"}, "temperature": 0,
                                     "messages": [{"role": "user", "content": "verify"}]})
    assert req["generationConfig"] == {"temperature": 0,
                                       "thinkingConfig": {"thinkingLevel": "low"}}
    plain = J.gemini_grounded_request({"messages": [{"role": "user", "content": "verify"}]})
    assert "generationConfig" not in plain


@pytest.fixture
def proxy(gemini, monkeypatch):
    monkeypatch.setenv(J.GRADER_KEY_ENV, "gbench-grader-abc123")
    server = ThreadingHTTPServer(("127.0.0.1", 0), J._Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gemini.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield gemini
    server.shutdown()
    server.server_close()


def _chat(url, key, body):
    req = urllib.request.Request(url + "/v1/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=15) as r:
        return r.status, json.loads(r.read())


def test_the_proxy_counts_thinking_off_requests_by_kind(proxy):
    assert _chat(proxy.url, "gbench-grader-abc123", _SAFETY_JUDGE)[0] == 200
    assert _chat(proxy.url, "gbench-cascade", dict(_SAFETY_JUDGE))[0] == 200
    assert _chat(proxy.url, "gbench-grader-abc123",
                 {"messages": [{"role": "user", "content": "q"}]})[0] == 200
    assert J.usage_snapshot()["thinking_off"] == {"judge": 1, "agent_helper": 1}
    usage = json.loads((proxy.wd / "gemini_usage.json").read_text())
    assert usage["thinking_off"] == {"judge": 1, "agent_helper": 1}
    efforts = [s["payload"].get("reasoning_effort") for s in proxy.sent]
    assert efforts == ["low", "low", None]
    # and the result records the mapping with those counts
    rec = W.thinking_off_mapping(usage)
    assert rec["requests"] == {"judge": 1, "agent_helper": 1}
    assert rec["sent_as"]["reasoning_effort"] == "low"
    assert rec["sent_as"]["grounded_generation_config"] == {"thinkingConfig": {
        "thinkingLevel": "low"}}
    assert rec["tasks"] == list(W.THINKING_OFF_TASKS)


def test_the_result_records_the_mapping():
    [judge] = [c for c in W.gemini_substitutions() if c["component"].startswith("LLM judge")]
    assert "reasoning_effort 'low'" in judge["served_by"]
    assert "thinking/reasoning/reasoning_effort settings dropped" not in judge["served_by"]
    assert judge["thinking_off"]["tasks"] == list(W.THINKING_OFF_TASKS)
    assert "requests" not in judge["thinking_off"]
    assert any("thinking level low" in r and "3 tasks" in r for r in W._SUBSTITUTION_REASONS)
    assert W.thinking_off_mapping({})["requests"] == {}


def test_the_wildclawbench_image_and_guide_follow_the_mapping():
    assert W.HARNESS_ERRORS_VERSION == "9"
    assert f'LABEL {W.HARNESS_ERRORS_LABEL}="{W.HARNESS_ERRORS_VERSION}"' in \
        _read("gbench/docker/wildclawbench.Dockerfile")
    doc = _flat(_read("docs/evals/wildclawbench.md"))
    assert f"`{W.HARNESS_ERRORS_LABEL}={W.HARNESS_ERRORS_VERSION}`" in doc
    assert "### Judge calls that turn thinking off" in _read("docs/evals/wildclawbench.md")
    for needle in ("`reasoning_effort: \"low\"`", "`generationConfig.thinkingConfig.thinkingLevel: "
                   "\"low\"`", "`gemini_usage.thinking_off`", "`judge_thinking_off`",
                   "`THINKING_OFF_LEVEL`", "`THINKING_OFF_TASKS`"):
        assert needle in doc, needle
    for task in W.THINKING_OFF_TASKS:
        assert f"`{task}`" in doc, task
    assert "`thinking`/`reasoning`/`reasoning_effort` settings are dropped |" not in doc


# =========================================================================== #
# mcp_bench: the credentialed script says when a session is up
# =========================================================================== #
@pytest.fixture
def cred_table(tmp_path, monkeypatch):
    base = tmp_path / "mcp-bench" / "mcp_servers"
    (base / "echo").mkdir(parents=True)
    (base / "echo" / "echo.py").write_text(
        "import sys\nsys.stdout.write(sys.stdin.readline())\nsys.stdout.flush()\n")
    table = {
        "NASA Data": {"cmd": f"{sys.executable} echo.py", "env": ["NASA_API_KEY"],
                      "cwd": "../echo"},
        "BioMCP": {"cmd": "gbench-no-such-command run", "env": ["NCI_API_KEY"], "cwd": "../echo"},
        "Math MCP": {"cmd": "node x.js", "env": [], "cwd": "../echo"}}
    (base / "commands.json").write_text(json.dumps(table))
    monkeypatch.setattr(CRED, "MCPBENCH_DIR", str(tmp_path / "mcp-bench"))
    return tmp_path


def _script(args, cwd, stdin=b""):
    env = {"PATH": os.environ.get("PATH", ""), "MCPBENCH_DIR": str(cwd / "mcp-bench"),
           "PYTHONDONTWRITEBYTECODE": "1", "NASA_API_KEY": "nasa-key-value-1234"}
    return subprocess.run([sys.executable, os.path.join(_DOCKER_DIR, "mcp_bench_credentialed.py"),
                           *args], input=stdin, capture_output=True, env=env, timeout=60)


def test_every_mode_first_says_whether_the_session_is_up(cred_table):
    ok = _script(["stdio", "nasa-data"], cred_table, stdin=b'{"id": 1}\n')
    first, rest = ok.stdout.split(b"\n", 1)
    assert json.loads(first) == {"ready": True} and rest == b'{"id": 1}\n'
    assert ok.returncode == 0
    for args, code, needle in ((["stdio", "biomcp"], 1, "could not start"),
                               (["stdio", "math-mcp"], 2, "not a credentialed server"),
                               (["http", "biomcp"], 2, "not a credentialed http server"),
                               (["connect", "80"], 2, "not a port"),
                               (["connect", "65000"], 1, "cannot connect")):
        proc = _script(args, cred_table)
        status = json.loads(proc.stdout.split(b"\n", 1)[0])
        assert status["ready"] is False and needle in status["error"], args
        assert proc.returncode == code, args


# =========================================================================== #
# mcp_bench: the relay answers only once the session is up, and frames what comes back
# =========================================================================== #
class _FakeExec:
    """Stands in for `docker exec -i <container> python3 CREDENTIALED_SCRIPT <mode> <arg>`:
    runs the script body the test chose for the mode."""
    scripts = {}
    calls = []

    def __new__(cls, argv, **kw):
        cls.calls.append(list(argv))
        return _REAL_POPEN([sys.executable, "-c", cls.scripts[argv[-2]]], **kw)


_READY = "import sys, json\nprint(json.dumps({'ready': True}), flush=True)\n"
_ECHO = _READY + ("while True:\n    b = sys.stdin.buffer.read1(65536)\n    if not b:\n"
                  "        break\n    sys.stdout.buffer.write(b)\n    sys.stdout.buffer.flush()\n")


@pytest.fixture
def relay(monkeypatch):
    _FakeExec.calls = []
    _FakeExec.scripts = {"stdio": _ECHO, "connect": _ECHO,
                         "http": ("import sys, json; print(json.dumps({'ready': True, "
                                  "'port': 45678}), flush=True); sys.stdin.read()")}
    monkeypatch.setattr(MB.subprocess, "Popen", _FakeExec)
    monkeypatch.setattr(MB.CredentialedServersRelay, "READY_TIMEOUT_S",
                        {"stdio": 5, "http-start": 5, "http-connect": 5})
    r = MB.CredentialedServersRelay("gbench_mcp_bench_x-cred", "127.0.0.1").start()
    yield r
    r.stop()


@pytest.fixture
def bridge_files(tmp_path, monkeypatch):
    def configure(relay_address, token):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"relay": relay_address, "token": token}))
        monkeypatch.setattr(BRIDGE, "BRIDGE_CONFIG", str(cfg))
        monkeypatch.setattr(BRIDGE, "BRIDGE_FAILURES", str(tmp_path / "fails.jsonl"))
        return types.SimpleNamespace(cfg=str(cfg), fails=str(tmp_path / "fails.jsonl"))
    return configure


def _raw_session(address, header, timeout=10):
    host, port = address.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=timeout)
    s.sendall(json.dumps(header).encode() + b"\n")
    return s


def _answer(sock):
    buf = b""
    while not buf.endswith(b"\n"):
        c = sock.recv(1)
        if not c:
            break
        buf += c
    return buf.decode().strip()


@pytest.mark.parametrize("script,needle", [
    # docker exec into a stopped or removed container: an error, exit 1, no line
    ("import sys; sys.stderr.write('Error response from daemon: container x is not running');"
     "sys.exit(1)", "docker exec ended without an answer"),
    # the server's command could not start
    ("import sys, json; print(json.dumps({'ready': False, 'error': 'the server could not "
     "start (FileNotFoundError)'})); sys.exit(1)", "could not start"),
])
def test_a_session_that_did_not_come_up_is_refused_and_recorded(relay, script, needle):
    _FakeExec.scripts["stdio"] = script
    t0 = time.time()
    s = _raw_session(relay.address, {"token": relay.token, "op": "stdio", "server": "biomcp"})
    answer = _answer(s)
    s.close()
    assert answer.startswith("ERR not started: ") and needle in answer
    rec = relay.record()
    assert rec["failed"] == {"stdio not started": 1} and "sessions" not in rec or \
        rec["sessions"] == {}
    [f] = rec["failures"]
    assert f["op"] == "stdio" and f["server"] == "biomcp" and needle in f["why"]
    assert f["t"] >= t0 - 1 and rec["failure_times"] == [f["t"]]
    assert "(exit 1" in f["why"]


def test_a_session_that_never_answers_is_refused_after_the_timeout(relay, monkeypatch):
    monkeypatch.setattr(MB.CredentialedServersRelay, "READY_TIMEOUT_S",
                        {"stdio": 1, "http-start": 1, "http-connect": 1})
    _FakeExec.scripts["stdio"] = "import time; time.sleep(30)"
    s = _raw_session(relay.address, {"token": relay.token, "op": "stdio", "server": "biomcp"})
    answer = _answer(s)
    s.close()
    assert answer == "ERR not started: no answer within 1 s"
    assert relay.record()["failed"] == {"stdio not started": 1}


def test_the_bridge_records_a_session_the_relay_did_not_open(relay, bridge_files):
    _FakeExec.scripts["stdio"] = "import sys; sys.exit(1)"
    files = bridge_files(relay.address, relay.token)
    assert BRIDGE.run_stdio("nasa-data") == 1
    [rec] = MRUN.read_bridge_failures(files.fails)
    assert rec["op"] == "stdio" and "refused the session (ERR not started" in rec["why"]
    _FakeExec.scripts["http"] = ("import json; print(json.dumps({'ready': False, 'error': "
                                 "'the server exited (1)'}))")
    assert BRIDGE.run_http("google-maps", _free_port()) == 1
    rec = MRUN.read_bridge_failures(files.fails)[-1]
    assert rec["op"] == "http-start" and "the server exited (1)" in rec["why"]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_what_comes_back_is_framed_and_ends_with_the_exit_code(relay):
    s = _raw_session(relay.address, {"token": relay.token, "op": "stdio", "server": "nasa-data"})
    assert _answer(s) == "OK"
    s.sendall(b'{"jsonrpc": "2.0"}\n')
    s.shutdown(socket.SHUT_WR)
    got = []
    with s.makefile("rb") as stream:
        assert BRIDGE.read_frames(stream, got.append) == ("end", 0, True)
    s.close()
    assert b"".join(got) == b'{"jsonrpc": "2.0"}\n'
    assert relay.record()["failed"] == {}


def test_a_docker_exec_failure_while_attached_is_recorded_and_sent_in_the_end_frame(relay):
    _FakeExec.scripts["stdio"] = _READY + ("sys.stdout.write(sys.stdin.readline()); "
                                           "sys.stdout.flush(); sys.exit(137)")
    t0 = time.time()
    s = _raw_session(relay.address, {"token": relay.token, "op": "stdio", "server": "nasa-data"})
    assert _answer(s) == "OK"
    s.sendall(b"ping\n")                       # the bridge stays attached: no half-close
    got = []
    with s.makefile("rb") as stream:
        assert BRIDGE.read_frames(stream, got.append) == ("end", 137, True)
    s.close()
    assert b"".join(got) == b"ping\n"
    rec = relay.record()
    assert rec["failed"] == {"stdio ended (exit 137)": 1}
    assert [f["t"] >= t0 - 1 for f in rec["failures"]] == [True]
    assert "docker exec exit 137" in rec["failures"][0]["why"]


def test_a_session_the_bridge_ended_is_no_failure(relay):
    _FakeExec.scripts["stdio"] = _READY + "sys.stdin.read(); sys.exit(1)"
    s = _raw_session(relay.address, {"token": relay.token, "op": "stdio", "server": "nasa-data"})
    assert _answer(s) == "OK"
    s.shutdown(socket.SHUT_WR)                 # the client ended the session
    with s.makefile("rb") as stream:
        how, code, _answered = BRIDGE.read_frames(stream, lambda d: None)
    s.close()
    assert (how, code) == ("end", 1)
    time.sleep(0.2)
    assert relay.record()["failed"] == {}


def test_a_session_cut_by_the_relays_stop_is_no_failure(relay):
    _FakeExec.scripts["stdio"] = _READY + "import time; time.sleep(30)"
    s = _raw_session(relay.address, {"token": relay.token, "op": "stdio", "server": "nasa-data"})
    assert _answer(s) == "OK"
    relay.stop()
    s.close()
    assert relay.record()["failed"] == {}


# =========================================================================== #
# mcp_bench: the bridge records a failed start and a lost session while its client uses it
# =========================================================================== #
def _run_bridge_stdio(files, keep_stdin_open=True, send=b'{"jsonrpc": "2.0", "id": 0}\n'):
    """The bridge's stdio mode in its own process (its stdin and stdout are the MCP client's),
    with its config and failure file in tmp. Returns (exit code, stdout)."""
    code = ("import sys; sys.path.insert(0, %r)\nimport mcp_bench_bridge as B\n"
            "B.BRIDGE_CONFIG = %r\nB.BRIDGE_FAILURES = %r\n"
            "sys.exit(B.run_stdio('nasa-data'))\n") % (_DOCKER_DIR, files.cfg, files.fails)
    proc = _REAL_POPEN([sys.executable, "-c", code], stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(send)
    proc.stdin.flush()
    if not keep_stdin_open:
        proc.stdin.close()
    try:
        out = b""
        deadline = time.time() + 30
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        out = proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.stdin.close()
        except OSError:
            pass
    return proc.wait(), out


def test_a_server_that_exits_before_it_answers_is_recorded(relay, bridge_files):
    # `uv run` of BioMCP / NASA Data failing at start: the process started, then exited silently
    _FakeExec.scripts["stdio"] = _READY + "sys.exit(0)"
    files = bridge_files(relay.address, relay.token)
    rc, out = _run_bridge_stdio(files)
    assert rc == 1 and out == b""
    [rec] = MRUN.read_bridge_failures(files.fails)
    assert rec["op"] == "stdio" and rec["why"] == "the server exited before it answered"


def test_a_session_lost_mid_task_is_recorded(relay, bridge_files):
    _FakeExec.scripts["stdio"] = _READY + ("sys.stdout.write(sys.stdin.readline()); "
                                           "sys.stdout.flush(); sys.exit(137)")
    files = bridge_files(relay.address, relay.token)
    rc, out = _run_bridge_stdio(files)
    assert rc == 1 and out == b'{"jsonrpc": "2.0", "id": 0}\n'
    [rec] = MRUN.read_bridge_failures(files.fails)
    assert rec["why"] == "the session ended on the credentialed side (exit 137)"


def test_a_server_that_exits_on_its_own_after_it_answered_is_its_outcome(relay, bridge_files):
    _FakeExec.scripts["stdio"] = _READY + ("sys.stdout.write(sys.stdin.readline()); "
                                           "sys.stdout.flush()")
    files = bridge_files(relay.address, relay.token)
    rc, out = _run_bridge_stdio(files)
    assert rc == 0 and out == b'{"jsonrpc": "2.0", "id": 0}\n'
    assert MRUN.read_bridge_failures(files.fails) == []


def test_a_session_the_client_ended_is_no_failure(relay, bridge_files):
    files = bridge_files(relay.address, relay.token)
    rc, out = _run_bridge_stdio(files, keep_stdin_open=False)
    assert rc == 0 and out == b'{"jsonrpc": "2.0", "id": 0}\n'
    assert MRUN.read_bridge_failures(files.fails) == []


class _CutRelay:
    """A relay that answers OK and then closes the stream (after `data` in one frame, when
    given) without an end frame: the session is lost."""

    def __init__(self, data=b"", http_line=None):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.address = f"127.0.0.1:{self.srv.getsockname()[1]}"
        self.data, self.http_line = data, http_line
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            BRIDGE._readline(c)
            c.sendall(b"OK\n" + (self.http_line or b""))
            if self.data:
                c.sendall(b"D%x\n" % len(self.data) + self.data)
            c.close()

    def close(self):
        self.srv.close()


def test_a_stream_without_the_end_frame_is_a_lost_session(bridge_files):
    cut = _CutRelay(data=b'{"partial": true}\n')
    try:
        files = bridge_files(cut.address, "t")
        rc, out = _run_bridge_stdio(files)
    finally:
        cut.close()
    assert rc == 1 and out == b'{"partial": true}\n'
    [rec] = MRUN.read_bridge_failures(files.fails)
    assert rec["why"].startswith("the session was lost (the relay's stream ended")


def test_an_http_start_with_no_ready_line_is_recorded(bridge_files):
    cut = _CutRelay()
    try:
        files = bridge_files(cut.address, "t")
        assert BRIDGE.run_http("google-maps", _free_port()) == 1
    finally:
        cut.close()
    [rec] = MRUN.read_bridge_failures(files.fails)
    assert rec["op"] == "http-start" and "did not start (no answer)" in rec["why"]


def test_an_http_server_session_lost_while_running_is_recorded(bridge_files):
    cut = _CutRelay(http_line=b'{"ready": true, "port": 45678}\n')
    try:
        files = bridge_files(cut.address, "t")
        assert BRIDGE.run_http("google-maps", _free_port()) == 1
    finally:
        cut.close()
    [rec] = MRUN.read_bridge_failures(files.fails)
    assert rec["op"] == "http-start" and rec["why"].startswith("the session was lost")


def test_an_http_server_that_exits_on_its_own_ends_the_bridge(relay, bridge_files):
    _FakeExec.scripts["http"] = ("import json; print(json.dumps({'ready': True, 'port': 45678}),"
                                 " flush=True)")
    files = bridge_files(relay.address, relay.token)
    assert BRIDGE.run_http("google-maps", _free_port()) == 0
    assert MRUN.read_bridge_failures(files.fails) == []


def test_an_http_connection_carried_and_lost(relay, bridge_files):
    _FakeExec.scripts["http"] = ("import sys, json; print(json.dumps({'ready': True, "
                                 "'port': 45678}), flush=True); sys.stdin.read()")
    files = bridge_files(relay.address, relay.token)
    port = _free_port()
    t = threading.Thread(target=BRIDGE.run_http, args=("google-maps", port), daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline and not relay._ports:
        time.sleep(0.05)
    # a request answered and closed by the server: no failure
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    c.sendall(b"GET /mcp HTTP/1.1\r\n\r\n")
    c.shutdown(socket.SHUT_WR)
    got = b""
    while True:
        chunk = c.recv(65536)
        if not chunk:
            break
        got += chunk
    c.close()
    assert got == b"GET /mcp HTTP/1.1\r\n\r\n"
    assert MRUN.read_bridge_failures(files.fails) == []
    # docker exec failing under an open connection: recorded
    _FakeExec.scripts["connect"] = _READY + "import time; time.sleep(0.3); sys.exit(125)"
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    c.sendall(b"GET /mcp HTTP/1.1\r\n\r\n")
    assert c.recv(65536) == b""
    c.close()
    deadline = time.time() + 10
    while time.time() < deadline and not MRUN.read_bridge_failures(files.fails):
        time.sleep(0.05)
    [rec] = MRUN.read_bridge_failures(files.fails)
    assert rec["op"] == "http-connect"
    assert rec["why"] == "the session ended on the credentialed side (exit 125)"


def test_a_bridge_failure_in_a_tasks_window_makes_it_a_harness_error(tmp_path):
    fails = tmp_path / "fails.jsonl"
    acct = MRUN.HarnessAccounting(names=[], resolvers=[], bridge_failures=str(fails))
    rec = acct.begin("t1", "f.json")
    BRIDGE.BRIDGE_FAILURES, saved = str(fails), BRIDGE.BRIDGE_FAILURES
    try:
        BRIDGE.record_failure("nasa-data", "stdio", "the server exited before it answered")
    finally:
        BRIDGE.BRIDGE_FAILURES = saved
    acct.end(rec, {"task_id": "t1", "status": "completed", "evaluation": {"x": 1}})
    [t] = acct.record()["tasks"]
    assert t["kinds"] == ["credentialed_relay"]
    assert "the server exited before it answered" in t["details"][0]
    assert acct.scored([{"task_id": "t1"}]) == []


# =========================================================================== #
# mcp_bench: the run's credentialed side on gbench's side
# =========================================================================== #
def test_the_container_state_is_taken_when_gbench_stops_it(monkeypatch):
    cred = MB.CredentialedServers("img", "gbench_mcp_bench_x", "bridge")
    cred._created = ["network", "container"]
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "false\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    cred.stop()
    assert calls[0] == ["docker", "inspect", "-f", "{{.State.Running}}", "gbench_mcp_bench_x-cred"]
    assert calls[1][:3] == ["docker", "rm", "-f"]
    assert cred.record()["running_at_stop"] == {"running": False,
                                                "state": "State.Running=false"}


def test_a_container_docker_no_longer_knows_is_not_running(monkeypatch):
    cred = MB.CredentialedServers("img", "b", "bridge")
    monkeypatch.setattr(MB.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "Error: No such object: b-cred"))
    assert cred.running() == (False, "Error: No such object: b-cred")


def test_the_run_record_counts_only_the_runs_failures():
    rec = {"relay": {"failures": [{"t": 5.0, "op": "stdio", "server": "biomcp", "why": "probe"},
                                  {"t": 20.0, "op": "stdio", "server": "biomcp", "why": "run"}],
                     "failure_times": [5.0, 20.0, 21.0]},
           "running_at_stop": {"running": True, "state": "State.Running=true"}}
    run = MB.credentialed_run_record(rec, 10.0)
    assert run["sessions_failed"] == 2 and [f["why"] for f in run["failures"]] == ["run"]
    assert run["container_running_at_end"] is True
    assert MB.credentialed_run_record({}, 0.0) == {
        "sessions_failed": 0, "failures": [], "container_running_at_end": None,
        "container_state": "not recorded"}


_PROBE_OK = {"servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
             "connections": {f"S{i}": {"status": "success", "tools": 3, "error": ""}
                             for i in range(28)}}


def _run_mcp_bench(monkeypatch, cred_record):
    for k in ("GBENCH_MCP_BENCH_NETWORK", "GBENCH_MCP_BENCH_DISTRACTION", "GBENCH_SHARD",
              "GBENCH_MCP_BENCH_DISABLE_STABILITY", "GBENCH_MCP_BENCH_ASSUME_NETWORK",
              "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTKEY0000000000000000000000000000")
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "server-key-value-1234")
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: _PROBE_OK)
    monkeypatch.setattr(MB.container_guard, "require_metadata_blocked",
                        lambda *a, **k: CG.UNREACHABLE)
    relay = types.SimpleNamespace(client_env=lambda: {}, stop=lambda: None, record=lambda: {})
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: relay)
    cred = types.SimpleNamespace(client_env=lambda: {}, stop=lambda: None,
                                 record=lambda: cred_record(time.time()))
    monkeypatch.setattr(MB, "start_credentialed_servers", lambda image, base, network: cred)

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
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
                    {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": [],
                     "judge_calls_failed": 0, "judge_models": {"gemini-3.8-flash": 5}}]}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    return MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1")


def test_relay_failures_during_the_run_make_it_completed_with_errors(monkeypatch):
    res = _run_mcp_bench(monkeypatch, lambda now: {
        "relay": {"failures": [{"t": now - 3600, "op": "stdio", "server": "biomcp",
                                "why": "before the run"},
                               {"t": now, "op": "stdio", "server": "nasa-data",
                                "why": "the session ended on the credentialed side (docker exec "
                                       "exit 137)"}],
                  "failure_times": [now - 3600, now]},
        "running_at_stop": {"running": True, "state": "State.Running=true"}})
    assert res["status"] == "completed_with_errors"
    [entry] = [e for e in res["harness_errors"] if e.get("kind") == "credentialed_relay"]
    assert entry["sessions_failed"] == 1 and entry["container_running_at_end"] is True
    assert entry["failures"][0]["server"] == "nasa-data"
    assert res["credentialed_servers_run"]["sessions_failed"] == 1
    assert "1 credentialed-server session(s) failed on gbench's relay during the run" in \
        res["leaderboard_comparable_reason"]


def test_a_stopped_credentialed_container_makes_the_run_completed_with_errors(monkeypatch):
    res = _run_mcp_bench(monkeypatch, lambda now: {
        "relay": {"failures": [], "failure_times": []},
        "running_at_stop": {"running": False, "state": "State.Running=false"}})
    assert res["status"] == "completed_with_errors"
    [entry] = [e for e in res["harness_errors"] if e.get("kind") == "credentialed_relay"]
    assert entry["sessions_failed"] == 0 and entry["container_running_at_end"] is False
    assert "the credentialed-servers container was not running at the end of the run" in \
        res["leaderboard_comparable_reason"]


def test_a_clean_credentialed_side_leaves_the_run_a_success(monkeypatch):
    res = _run_mcp_bench(monkeypatch, lambda now: {
        "relay": {"failures": [], "failure_times": []},
        "running_at_stop": {"running": True, "state": "State.Running=true"}})
    assert res["status"] == "success"
    assert not [e for e in res["harness_errors"] if e.get("kind") == "credentialed_relay"]


def test_an_image_with_the_old_session_protocol_is_refused(monkeypatch):
    monkeypatch.setattr(MB, "_image_labels", lambda image: {
        MB.REQUEST_POLICY_LABEL: MB.REQUEST_POLICY_VERSION, MB.HARNESS_LABEL: "2"})
    monkeypatch.setattr(MB.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(MB.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    ok, why = MB.check_mcp_bench_prerequisites()
    assert not ok and f"{MB.HARNESS_LABEL}={MB.HARNESS_VERSION}" in why
    assert "session protocol" in why and MB.HARNESS_VERSION == "3"


def test_the_mcp_bench_image_and_guide_follow_the_code():
    df = _read("gbench/docker/mcp_bench.Dockerfile")
    assert f'LABEL {MB.HARNESS_LABEL}="{MB.HARNESS_VERSION}"' in df
    doc = _flat(_read("docs/evals/mcp_bench.md"))
    for needle in (f"`{MB.HARNESS_LABEL}={MB.HARNESS_VERSION}`", "`ERR not started: <why>`",
                   "the stdio server exited before it sent a byte while the client still used "
                   "the session", "`credentialed_servers_run`", "`container_running_at_end`",
                   "an end frame that carries docker exec's exit code",
                   "`harness_errors` kind `credentialed_relay`"):
        assert needle in doc, needle
    assert "A session the relay did not open (no answer, refused, or lost before the server " \
           "answered)" not in doc
    header = _read("gbench/docker/mcp_bench_bridge.py")
    assert "lost before the server answered" not in header
    assert "the server exited before it answered" in header or \
        "exited before it sent a byte" in header
    assert MB.CredentialedServersRelay.READY_TIMEOUT_S["http-start"] < \
        BRIDGE.ANSWER_TIMEOUT_S["http-start"]
    assert MB.CredentialedServersRelay.READY_TIMEOUT_S["http-start"] > CRED.HTTP_READY_TIMEOUT_S


@pytest.mark.parametrize("rel", [
    "gbench/runners/eval_suites/mcp_bench.py", "gbench/runners/eval_suites/wildclawbench.py",
    "gbench/docker/mcp_bench_run.py", "gbench/docker/mcp_bench_bridge.py",
    "gbench/docker/mcp_bench_credentialed.py", "gbench/docker/mcp_bench.Dockerfile",
    "gbench/docker/wildclawbench_cascade_judge.py", "gbench/docker/wildclawbench.Dockerfile",
    "docs/evals/mcp_bench.md", "docs/evals/wildclawbench.md",
    "tests/test_wildclawbench_thinking_level_and_mcp_bench_sessions.py"])
def test_the_changed_files_are_plain_ascii_without_internal_labels(rel):
    text = _read(rel)
    assert all(ord(c) < 128 for c in text), rel
    low = text.lower()
    # built from parts, so this file does not hold them
    for label in ("wave" + " 2", "wave" + "2", "the " + "reviewer", "skep" + "tic"):
        assert label not in low, (rel, label)
