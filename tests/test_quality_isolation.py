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
"""The GemmaClaw QA suite runs in gbench's own container, isolated from the host and from every
credential, and harness errors never count as the model under test's outcome.

* The container: no network (cloud metadata unreachable, proved from inside), a read-only root
  filesystem, the suite and the model's tools as a user without capabilities, only two host paths
  (both read-only), no host environment; the relay token reaches it on standard input only and
  stays with the forwarder, which runs as another user. Proved with the runner's own docker
  command line, entrypoint and forwarder in a small image built from a local base.
* The host never runs anything the model under test can write: git runs with hooks and fsmonitor
  disabled on a checkout no container sees, the image is built from `git archive` of the commit,
  and the container's outputs are read from the docker cp stream in memory (regular files only).
* The relay forwards only the Gemini API calls the suite makes and the endpoint's chat calls;
  refused and unauthenticated requests are counted apart, not as harness errors. Gemini
  non-answers and truncated replies are harness errors.
* Harness errors are tied to the scenarios whose windows hold them: those scenarios are listed
  apart and not scored; an error no window holds withholds the pass rate.

Model-free: stub Gemini and model endpoints, no GemmaClaw build, no Gemini call. Files are written
under tmp only.
"""

import http.client
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qsl, urlsplit

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import quality as Q

ROOT = Path(__file__).resolve().parents[1]

GEM_A = "AIzaSyISOtestGemA00000000000000000001"
GEM_B = "AIzaSyISOtestGemB00000000000000000002"
ENDPOINT_KEY = "vllm-ISOtest-endpoint-key-000001"
PROXY_PASSWORD = "hunter2ISOpass"
HOSTILE_ENV = {
    "GEMINI_API_KEY": GEM_A, "GEMINI_API_KEYS": f"{GEM_A},{GEM_B}", "GEMINI_KEY": GEM_A,
    "GENAI_KEY": GEM_A, "LLM_KEY": GEM_B, "GEMINI_API_KEY_BACKUP": GEM_B,
    "HTTPS_PROXY": f"http://u:{PROXY_PASSWORD}@proxy.invalid:3128",
    "KAGGLE_KEY": "kaggle-ISOtest-0001", "SERPAPI_KEY": "serp-ISOtest-0001",
    "VLLM_API_KEY": ENDPOINT_KEY, "OPENAI_API_KEY": "sk-ISOtest-openai-0001"}
SECRETS = (GEM_A, GEM_B, ENDPOINT_KEY, PROXY_PASSWORD, "kaggle-ISOtest-0001", "serp-ISOtest-0001",
           "sk-ISOtest-openai-0001")

GOOD_SSE = (b'data: {"candidates": [{"content": {"parts": [{"text": "OK"}]}, '
            b'"finishReason": "STOP"}]}\r\n\r\n')
GOOD_JSON = b'{"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}'
GOOD_CHAT = b'{"choices": [{"message": {"role": "assistant", "content": "hi"}}]}'


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class _Upstream:
    """A stub HTTP upstream (Gemini API or model endpoint): records every request and answers from
    `script` [(status, body)], then `default`."""

    def __init__(self, default=(200, GOOD_JSON)):
        self.requests, self.script, self.default = [], [], default
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                return

            def _any(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                split = urlsplit(self.path)
                outer.requests.append({"method": self.command, "path": split.path,
                                       "query": dict(parse_qsl(split.query)),
                                       "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body})
                status, data = outer.script.pop(0) if outer.script else outer.default
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

            do_GET = do_POST = do_DELETE = _any

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def port(self):
        return self.server.server_address[1]

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _UnixConn(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("relay", timeout=30)
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        s.connect(self._path)
        self.sock = s


def _relay_call(relay, path, *, token=True, method="POST", body=b"{}", headers=None):
    """(status, body) of one request to the relay's unix socket, as the forwarder sends it."""
    hdr = dict(headers or {})
    if token:
        hdr[Q.RELAY_TOKEN_HEADER] = relay.token if token is True else token
    conn = _UnixConn(relay.socket_path)
    try:
        conn.request(method, path, body=body if method not in ("GET", "HEAD") else None, headers=hdr)
    except (BrokenPipeError, ConnectionResetError):
        pass
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


@pytest.fixture
def gemini():
    up = _Upstream()
    yield up
    up.close()


@pytest.fixture
def model_up():
    up = _Upstream(default=(200, GOOD_CHAT))
    yield up
    up.close()


@pytest.fixture
def relay_to(gemini, monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "GEMINI_API_ORIGIN", gemini.url)
    relays = []

    def make(**kw):
        d = Path(os.environ.get("TMPDIR", "/tmp"))
        sock_dir = Path(__import__("tempfile").mkdtemp(prefix="gbq_t_", dir=str(d)))
        relay = Q.KeyRelay([GEM_A, GEM_B], **kw).start(str(sock_dir / Q.RELAY_SOCKET_NAME))
        relays.append((relay, sock_dir))
        return relay
    yield make
    for relay, sock_dir in relays:
        relay.stop()
        shutil.rmtree(sock_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# the relay: what it forwards, what it refuses, what it counts
# --------------------------------------------------------------------------- #
def test_the_relay_forwards_only_the_gemini_calls_the_suite_makes(relay_to, gemini):
    relay = relay_to()
    allowed = [f"/v1beta/models/{m}:{meth}" for m in sorted(Q.ALLOWED_GEMINI_MODELS)
               for meth in ("generateContent", "embedContent")]
    gemini.default = (200, b'{"embedding": {"values": [0.5]}, "candidates": [{"content": {"parts"'
                           b': [{"text": "x"}, {"inlineData": {"data": "AA=="}}]}, '
                           b'"finishReason": "STOP"}]}')
    for path in allowed:
        assert _relay_call(relay, path)[0] == 200, path
    refused = [("GET", "/v1beta/files"), ("DELETE", "/v1beta/files/abc"),
               ("DELETE", "/v1beta/cachedContents/abc"), ("DELETE", "/v1beta/tunedModels/t1"),
               ("GET", "/v1beta/models"), ("POST", "/upload/v1beta/files"),
               ("POST", "/v1beta/models/gemini-2.5-pro:generateContent"),
               ("POST", "/v1beta/models/gemini-3.1-pro-preview:countTokens"),
               ("GET", "/v1beta/models/gemini-3.1-pro-preview:generateContent"),
               ("POST", "/v1beta/models/gemini-3.1-pro-preview:generateContent/../x"),
               ("POST", "/model/chat/completions")]            # no model endpoint configured
    n_before = len(gemini.requests)
    for method, path in refused:
        status, data = _relay_call(relay, path, method=method)
        assert status == 403 and b"not a" in data, (method, path)
    assert len(gemini.requests) == n_before == len(allowed)          # never forwarded
    rep = relay.report()
    assert rep["harness_errors"] == {} and rep["rejected"] == 0
    assert sum(r["requests"] for r in rep["refused"]) == len(refused)
    assert {(r["route"], r["method"], r["path"]) for r in rep["refused"]} >= {
        ("gemini", "GET", "/v1beta/files"), ("gemini", "DELETE", "/v1beta/cachedContents/abc")}


def test_the_relay_forwards_only_the_endpoints_chat_calls(relay_to, model_up):
    relay = relay_to(model_upstream=model_up.url + "/v1", model_key=ENDPOINT_KEY)
    assert _relay_call(relay, "/model/chat/completions")[0] == 200
    assert _relay_call(relay, "/model/models", method="GET")[0] == 200
    for method, path in (("GET", "/model/metrics"), ("POST", "/model/../admin"),
                         ("POST", "/model/v1/load_lora_adapter"), ("DELETE", "/model/models")):
        assert _relay_call(relay, path, method=method)[0] == 403, path
    assert [(r["method"], r["path"]) for r in model_up.requests] == [
        ("POST", "/v1/chat/completions"), ("GET", "/v1/models")]
    assert all(r["headers"]["authorization"] == f"Bearer {ENDPOINT_KEY}" for r in model_up.requests)
    assert relay.report()["harness_errors"] == {}


def test_only_the_forwarders_token_header_opens_the_relay_and_it_is_never_forwarded(relay_to,
                                                                                  gemini):
    relay = relay_to()
    path = "/v1beta/models/gemini-3.1-pro-preview:generateContent"
    # the token anywhere else (as the suite's key, a Bearer token, key=) opens nothing
    for headers, suffix in (({"x-goog-api-key": relay.token}, ""),
                            ({"Authorization": f"Bearer {relay.token}"}, ""),
                            ({}, f"?key={relay.token}")):
        assert _relay_call(relay, path + suffix, token=False, headers=headers)[0] == 401
    assert _relay_call(relay, path, token="gbench-relay-" + "0" * 48)[0] == 401
    assert gemini.requests == []
    status, _ = _relay_call(relay, path + "?alt=sse&key=junk&x=1",
                            headers={"x-goog-api-key": Q.KEY_PLACEHOLDER,
                                     "Authorization": "Bearer junk",
                                     "x-goog-user-project": "someone-elses",
                                     "Content-Type": "application/json"})
    assert status == 200
    [req] = gemini.requests
    assert req["headers"]["x-goog-api-key"] in (GEM_A, GEM_B)
    assert "authorization" not in req["headers"] and "x-goog-user-project" not in req["headers"]
    assert Q.RELAY_TOKEN_HEADER not in req["headers"] and req["query"] == {"alt": "sse"}
    assert relay.token not in json.dumps(req["headers"]) + json.dumps(req["query"])
    rep = relay.report()
    # unauthenticated requests are counted apart: they are not harness errors
    assert rep["rejected"] == 4 and rep["harness_errors"] == {}


def test_an_oversized_request_is_refused_unread(relay_to, gemini, monkeypatch):
    monkeypatch.setattr(Q, "_MAX_REQUEST_BODY", 1024)
    relay = relay_to()
    path = "/v1beta/models/gemini-3.1-pro-preview:generateContent"
    status, data = _relay_call(relay, path, body=b"x" * 2048)
    assert status == 413 and b"request body over 1024 bytes" in data
    assert gemini.requests == []
    rep = relay.report()
    assert rep["harness_errors"] == {} and rep["refused"][0]["path"] == path
    assert _relay_call(relay, path, body=b"{}")[0] == 200          # a normal one still goes


@pytest.mark.parametrize("cap,method,body,outcome", [
    ("alternate model", "generateContent", GOOD_JSON, "answered"),
    ("alternate model", "streamGenerateContent", GOOD_SSE, "answered"),
    ("alternate model", "streamGenerateContent",
     b'data: {"candidates": [{"content": {"parts": [{"text": "par"}]}}]}\r\n\r\n', "truncated"),
    ("alternate model", "generateContent",
     b'{"candidates": [{"content": {"parts": [{"text": "cut"}]}, "finishReason": "MAX_TOKENS"}]}',
     "truncated"),
    ("alternate model", "generateContent", b'{"candidates": []}', "non_answer"),
    ("alternate model", "generateContent",
     b'{"promptFeedback": {"blockReason": "SAFETY"}}', "non_answer"),
    ("alternate model", "generateContent",
     b'{"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "SAFETY"}]}',
     "non_answer"),
    ("alternate model", "generateContent",
     b'{"candidates": [{"content": {"parts": [{"text": "hm", "thought": true}]}, '
     b'"finishReason": "STOP"}]}', "non_answer"),
    ("alternate model", "generateContent",
     b'{"candidates": [{"content": {"parts": [{"functionCall": {"name": "f"}}]}, '
     b'"finishReason": "STOP"}]}', "answered"),
    ("image generation", "generateContent", GOOD_JSON, "non_answer"),
    ("image generation", "generateContent",
     b'{"candidates": [{"content": {"parts": [{"inlineData": {"data": "AA=="}}]}, '
     b'"finishReason": "STOP"}]}', "answered"),
    ("memory embeddings", "embedContent", b'{"embedding": {"values": [0.1]}}', "answered"),
    ("memory embeddings", "embedContent", b'{"embedding": {}}', "non_answer"),
    ("memory embeddings", "batchEmbedContents",
     b'{"embeddings": [{"values": [0.1]}, {"values": []}]}', "non_answer"),
    ("alternate model", "generateContent", b"<html>oops</html>", "non_answer"),
])
def test_gemini_non_answers_and_truncated_replies_are_named(cap, method, body, outcome):
    assert Q.gemini_reply_outcome(cap, method, body) == outcome


def test_a_gemini_non_answer_or_truncated_reply_is_a_harness_error(relay_to, gemini):
    relay = relay_to()
    gemini.script = [(200, b'{"candidates": []}'),
                     (200, b'{"candidates": [{"content": {"parts": [{"text": "c"}]}, '
                           b'"finishReason": "MAX_TOKENS"}]}'),
                     (200, GOOD_JSON), (400, b'{"error": {"code": 400}}')]
    path = "/v1beta/models/gemini-3.1-pro-preview:generateContent"
    for _ in range(4):
        _relay_call(relay, path)
    rep = relay.report()
    assert rep["harness_errors"] == {"gemini_non_answer": 1, "gemini_truncated": 1,
                                     "gemini_http_400": 1}
    assert [kind for _t, kind in rep["harness_error_events"]] == [
        "gemini_non_answer", "gemini_truncated", "gemini_http_400"]
    [row] = rep["gemini_calls"]
    assert row["statuses"] == {"200": 3, "400": 1}
    assert row["non_answers"] == 1 and row["truncated"] == 1


def test_endpoint_failures_are_harness_errors_and_the_endpoints_own_400_is_not(relay_to,
                                                                              model_up):
    relay = relay_to(model_upstream=model_up.url + "/v1")
    model_up.script = [(503, b"{}"), (400, b'{"error": "context length"}')]
    assert _relay_call(relay, "/model/chat/completions")[0] == 503
    assert _relay_call(relay, "/model/chat/completions")[0] == 400
    assert relay.report()["harness_errors"] == {"model_http_503": 1}
    assert "authorization" not in model_up.requests[0]["headers"]     # gbench's key-less server


# --------------------------------------------------------------------------- #
# harness errors and the scenarios they fall in
# --------------------------------------------------------------------------- #
def _iso(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _scen_summary(title, start, end, status="pass"):
    return {"scenarios": [{"name": title, "status": status, "steps": []}],
            "counts": {"total": 1, "passed": int(status == "pass"),
                       "failed": int(status != "pass")},
            "run": {"startedAt": _iso(start), "finishedAt": _iso(end)}}


def test_errors_are_tied_to_the_scenarios_running_when_they_happened():
    summaries = {"a": _scen_summary("A", 1000, 1010), "b": _scen_summary("B", 1005, 1030),
                 "c": _scen_summary("C", 2000, 2010)}
    windows = Q.scenario_windows({"scenarios": []}, summaries)
    assert set(windows) == {"A", "B", "C"}
    by_title, unattributed = Q.attribute_events(
        [(1007.0, "gemini_http_503"), (2010.5, "model_unreachable"), (1500.0, "gemini_http_429"),
         (None, "forwarder_exited")], windows)
    assert by_title == {"A": {"gemini_http_503": 1}, "B": {"gemini_http_503": 1},
                        "C": {"model_unreachable": 1}}
    assert unattributed == {"gemini_http_429": 1, "forwarder_exited": 1}
    # a run of one scenario: the run's own window
    one = {"scenarios": [{"name": "Only", "status": "pass"}],
           "run": {"startedAt": _iso(10), "finishedAt": _iso(20)}}
    assert Q.scenario_windows(one, {}) == {"Only": [(10.0, 20.0)]}


def _write_meta(dest: Path):
    for pid in ("openai", "google", "vllm", "qa-lab"):
        d = dest / "extensions" / pid
        d.mkdir(parents=True, exist_ok=True)
        (d / "openclaw.plugin.json").write_text(json.dumps({"id": pid, "providers": [pid]}))
    scen = dest / "qa" / "scenarios" / "models"
    scen.mkdir(parents=True, exist_ok=True)
    (scen / "gpt54.md").write_text(
        "# E\n\n```yaml qa-scenario\nid: gpt54-thinking\ntitle: E\nexecution:\n  kind: flow\n"
        "  config:\n    requiredLiveProvider: openai\n    requiredLiveModel: gpt-5.4\n```\n")


def _cfg(tmp_path, **kw):
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "results"
    config.results_dir.mkdir(exist_ok=True)
    config.gemmaclaw_commit = "0123456789abcdef0123456789abcdef01234567"
    config.gemmaclaw_path = str(tmp_path / "gemmaclaw")
    for k, v in kw.items():
        setattr(config, k, v)
    config.initialize()
    return config


class _Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now


def _volume_source(argv, dest):
    for i, a in enumerate(argv):
        if a == "--volume" and argv[i + 1].split(":")[1:2] == [dest]:
            return argv[i + 1].split(":")[0]
    raise AssertionError(dest)


def _run_on_stubs(tmp_path, monkeypatch, gemini, model_up, script, summary, scenario_summaries,
                  remote=False, forwarder_errors=(), entrypoint=None):
    """A whole run with docker stubbed: the fake `docker run` reads the relay token from its
    standard input and plays the forwarder (every `script` step is (clock time, path, method))."""
    monkeypatch.setattr(Q, "GEMINI_API_ORIGIN", gemini.url)
    clock = _Clock()
    monkeypatch.setattr(Q, "time", types.SimpleNamespace(time=clock.time, sleep=lambda s: None,
                                                          monotonic=clock.time))
    for k, v in HOSTILE_ENV.items():
        monkeypatch.setenv(k, v)
    if not remote:
        monkeypatch.delenv("VLLM_API_KEY")
    runner = Q.QualityBenchmarkRunner(_cfg(
        tmp_path, **({"remote_endpoint": model_up.url + "/v1"} if remote else {})))
    runner.server_port = model_up.port                  # gbench's "own" server: the stub
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._start_server = mock.MagicMock(return_value=True)
    runner._cleanup_server = mock.MagicMock()
    runner._image = "gbench-quality:test"
    patches = [{"file": Q.GEMINI_IMAGE_PATCH_FILE, "change": "c", "state": "applied",
                "sha256": "0" * 64}]
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="c0ffee" * 6 + "abcd")

    def meta(image, dest):
        _write_meta(dest)
        return {"patches": patches, "commit": "c0ffee"}

    runner._image_metadata = meta
    runner._remove_container = mock.MagicMock()
    seen = {}

    class FakeDocker:
        def __init__(self, argv, stdin=None, stdout=None, stderr=None, env=None, **kw):
            seen["argv"], seen["docker_env"] = list(argv), dict(env or {})
            self.stdin = io.BytesIO()
            self.stdin.close = lambda: None
            self.returncode = None
            self._stdout = stdout

        def poll(self):
            if self.returncode is None:
                token = self.stdin.getvalue().decode().strip()
                seen["token"] = token
                argv = seen["argv"]
                seen["config"] = Path(_volume_source(argv, Q.CONTAINER_CONFIG_PATH)).read_text()
                sock = os.path.join(_volume_source(argv, Q.CONTAINER_RELAY_DIR),
                                    Q.RELAY_SOCKET_NAME)
                relay = types.SimpleNamespace(token=token, socket_path=sock)
                for t, path, method in script:
                    clock.now = t
                    seen.setdefault("statuses", []).append(
                        _relay_call(relay, path, method=method)[0])
                self._stdout.write("suite output line\n")
                self._stdout.flush()
                self.returncode = 0
            return self.returncode

        def wait(self, timeout=None):
            return self.poll()

        def kill(self):
            pass

    monkeypatch.setattr(Q.subprocess, "Popen", FakeDocker)
    runner._collect_container_outputs = lambda name: {
        "summary": summary, "scenario_summaries": scenario_summaries,
        "entrypoint": entrypoint or {"command_exit": 0, "forwarder_alive_at_end": True},
        "forwarder_errors": list(forwarder_errors)}
    model = ModelConfig(short_name="gemma-4-E2B-it", hf_model_id="google/gemma-4-E2B-it",
                        total_params_b=5.2, is_moe=False)
    # The endpoint's /v1/models (the served id check) answered without a request to the stub, so
    # the stub sees only the suite's own calls.
    monkeypatch.setattr("gbench.utils.endpoint.list_served_models",
                        lambda base_url, timeout=5.0, headers=None: [{"id": model.hf_model_id}])
    with mock.patch("gbench.runners.evals._server_max_model_len", return_value=32768):
        res = runner.run(model, ModelFormat.HF)
    saved = Path(runner.config.log_manager.get_quality_result_path(
        model.short_name, ModelFormat.HF.value)).read_text()
    log = Path(runner.config.log_manager.get_quality_log_path(
        model.short_name, ModelFormat.HF.value)).read_text()
    return runner, res, seen, saved, log


ALT = "/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse"
IMG = "/v1beta/models/gemini-3.1-flash-image-preview:generateContent"
CHAT = "/model/chat/completions"


def test_a_run_scores_only_scenarios_no_harness_error_fell_in(tmp_path, monkeypatch, gemini,
                                                             model_up):
    gemini.script = [(200, GOOD_SSE), (503, b"{}")]
    model_up.script = [(200, GOOD_CHAT), (200, GOOD_CHAT), (503, b"{}"), (200, GOOD_CHAT)]
    script = [(1001, ALT, "POST"), (1002, CHAT, "POST"),        # A: fine
              (2001, IMG, "POST"), (2002, CHAT, "POST"),        # B: Gemini 503
              (3001, CHAT, "POST"),                             # C: endpoint 503
              (4001, CHAT, "POST"),                             # D: fine
              (4002, "/v1beta/files", "GET")]                   # D: refused, not an error
    summary = {"counts": {"total": 5, "passed": 3, "failed": 2},
               "scenarios": [{"name": "A", "status": "pass"}, {"name": "B", "status": "pass"},
                             {"name": "C", "status": "fail"}, {"name": "D", "status": "fail"},
                             {"name": "E", "status": "pass"}],
               "run": {"startedAt": _iso(900), "finishedAt": _iso(5000)}}
    scen = {"a": _scen_summary("A", 1000, 1010), "b": _scen_summary("B", 2000, 2010),
            "c": _scen_summary("C", 3000, 3010, "fail"), "d": _scen_summary("D", 4000, 4010,
                                                                            "fail"),
            "gpt54-thinking": _scen_summary("E", 4500, 4510)}
    runner, res, seen, saved, log = _run_on_stubs(tmp_path, monkeypatch, gemini, model_up,
                                                  script, summary, scen)
    assert seen["statuses"] == [200, 200, 503, 200, 503, 200, 403]
    # gbench's own (key-less) server is behind the relay too: its failure is counted
    assert [r["path"] for r in model_up.requests] == ["/v1/chat/completions"] * 4
    assert res["harness_errors"] == {"gemini_http_503": 1, "model_http_503": 1}
    assert res["status"] == "completed_with_errors"
    hes = {r["title"]: r for r in res["harness_error_scenarios"]}
    assert hes["B"] == {"id": "b", "title": "B", "suite_status": "pass",
                        "harness_errors": {"gemini_http_503": 1}}
    assert hes["C"]["harness_errors"] == {"model_http_503": 1} and hes["C"]["id"] == "c"
    # scored: A (pass) and D (fail); B passed anyway and is still not scored; E needs openai
    assert (res["total_scenarios"], res["passed_scenarios"], res["failed_scenarios"]) == (2, 1, 1)
    assert res["pass_rate"] == 50.0 and res["pass_rate_withheld_reason"] is None
    assert res["unattributed_harness_errors"] == {}
    assert [r["id"] for r in res["infra_required_scenarios"]] == ["gpt54-thinking"]
    assert res["relay_refused_requests"] == [{"route": "gemini", "method": "GET",
                                              "path": "/v1beta/files", "requests": 1,
                                              "scenarios": ["D"]}]
    assert res["relay_rejected_requests"] == 0
    reason = res["leaderboard_comparable_reason"]
    assert "gemini_http_503 x1" in reason and "not scored: b, c" in reason
    assert "through the container's relay forwarder" in reason
    assert res["model_endpoint_relayed"] is True and res["model_endpoint_key_at_relay"] is False
    # the relay token went on standard input only: not on the command line, in the environment,
    # the configuration, the log or the result
    token = seen["token"]
    assert token.startswith("gbench-relay-")
    blob = json.dumps([seen["argv"], seen["docker_env"], seen["config"], log, saved, res])
    for secret in (token, *SECRETS):
        assert secret not in blob, secret
    cfg = json.loads(seen["config"])["models"]["providers"]
    assert cfg["vllm"]["baseUrl"] == Q.CONTAINER_MODEL_BASE_URL
    assert cfg["google"]["baseUrl"] == Q.CONTAINER_GEMINI_BASE_URL
    assert cfg["vllm"]["apiKey"] == cfg["google"]["apiKey"] == Q.KEY_PLACEHOLDER
    assert res["gemmaclaw_patches"][0]["file"] == Q.GEMINI_IMAGE_PATCH_FILE
    assert res["sandbox"]["tool"] == "docker" and res["sandbox"]["image"] == "gbench-quality:test"
    assert res["sandbox"]["writable"] == [
        "/tmp (tmpfs)", f"{Q.CONTAINER_HOME} (tmpfs)",
        *(f"{d} (a volume removed with the container)" for d in (
            Q.CONTAINER_OUTPUT_DIR, Q.CONTAINER_ARTIFACTS_DIR, Q.CONTAINER_NPM_CACHE))]


def test_an_error_no_scenario_window_holds_withholds_the_pass_rate(tmp_path, monkeypatch,
                                                                  gemini, model_up):
    gemini.script = [(429, b"{}"), (429, b"{}")]
    summary = {"counts": {"total": 1, "passed": 1, "failed": 0},
               "scenarios": [{"name": "A", "status": "pass"}, {"name": "B", "status": "fail"}]}
    scen = {"a": _scen_summary("A", 1000, 1010), "b": _scen_summary("B", 2000, 2010, "fail")}
    _r, res, _seen, _saved, _log = _run_on_stubs(
        tmp_path, monkeypatch, gemini, model_up, [(1500, ALT, "POST")], summary, scen,
        forwarder_errors=[{"t": 2005.0, "kind": "relay_unreachable", "route": "model"}],
        entrypoint={"command_exit": 1, "forwarder_alive_at_end": False})
    assert res["harness_errors"] == {"forwarder_exited": 1, "forwarder_relay_unreachable": 1,
                                     "gemini_http_429": 1}
    assert res["unattributed_harness_errors"] == {"forwarder_exited": 1, "gemini_http_429": 1}
    assert [r["title"] for r in res["harness_error_scenarios"]] == ["B"]
    assert res["pass_rate"] is None and "gemini_http_429 x1" in res["pass_rate_withheld_reason"]
    assert "pass_rate withheld" in res["leaderboard_comparable_reason"]
    assert res["status"] == "completed_with_errors"


def test_a_scenario_whose_worker_the_suite_could_not_run_is_a_harness_error(tmp_path,
                                                                         monkeypatch, gemini,
                                                                         model_up):
    worker = [{"name": "isolated scenario worker", "status": "fail",
               "details": "gateway failed to start"}]
    summary = {"counts": {"total": 3, "passed": 1, "failed": 2},
               "scenarios": [{"name": "A", "status": "pass", "steps": []},
                             {"name": "B", "status": "fail", "steps": worker},
                             {"name": "C", "status": "fail",
                              "steps": [{"name": "reply check", "status": "fail"}]}]}
    scen = {"a": _scen_summary("A", 1000, 1010), "c": _scen_summary("C", 3000, 3010, "fail")}
    _r, res, _seen, _saved, _log = _run_on_stubs(tmp_path, monkeypatch, gemini, model_up, [],
                                                 summary, scen)
    assert res["harness_errors"] == {"suite_worker_failed": 1}
    assert res["harness_error_scenarios"] == [{"id": None, "title": "B", "suite_status": "fail",
                                               "harness_errors": {"suite_worker_failed": 1}}]
    assert (res["total_scenarios"], res["passed_scenarios"], res["failed_scenarios"]) == (2, 1, 1)
    assert res["pass_rate"] == 50.0 and res["status"] == "completed_with_errors"
    assert "scenario workers the suite could not run" in res["leaderboard_comparable_reason"]


def test_a_suite_that_lingers_after_its_summary_is_stopped(tmp_path, monkeypatch, gemini,
                                                         model_up):
    """A suite whose isolated worker failed can leave servers open and never exit: once it has
    printed its summary line, gbench waits SUITE_EXIT_GRACE_S, then stops its container."""
    monkeypatch.setattr(Q, "GEMINI_API_ORIGIN", gemini.url)
    clock = _Clock()

    def sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(Q, "time", types.SimpleNamespace(time=clock.time, sleep=sleep,
                                                          monotonic=clock.time))
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    # the endpoint's /v1/models (the served id check), answered without a request to the stub
    monkeypatch.setattr("gbench.utils.endpoint.list_served_models",
                        lambda base_url, timeout=5.0, headers=None: [{"id": "org/m"}])
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, remote_endpoint=model_up.url + "/v1"))
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="c0ffee")
    runner._cleanup_server = mock.MagicMock()
    runner._remove_container = mock.MagicMock()
    runner._image = "gbench-quality:test"
    runner._image_metadata = lambda image, dest: (_write_meta(dest), {"patches": [],
                                                                      "commit": "c"})[1]
    state = {"stopped_at": None}

    class Lingering:
        def __init__(self, argv, stdin=None, stdout=None, **kw):
            self.argv, self.stdout = argv, stdout
            self.stdin = io.BytesIO()
            self.stdin.close = lambda: None
            self.returncode = None
            stdout.write("QA suite summary: /opt/gemmaclaw/gbench_qa_out/qa-suite-summary"
                         ".json\n")
            stdout.flush()

        def poll(self):
            if state["stopped_at"] is not None:
                self.returncode = 143
            return self.returncode

        def wait(self, timeout=None):
            return self.poll()

        def kill(self):
            pass

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "stop"]:
            state["stopped_at"] = clock.now
            state["name"] = cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(Q.subprocess, "Popen", Lingering)
    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    runner._collect_container_outputs = lambda name: {
        "summary": {"counts": {"total": 1, "passed": 1, "failed": 0},
                    "scenarios": [{"name": "A", "status": "pass"}],
                    "run": {"startedAt": _iso(0), "finishedAt": _iso(10)}},
        "scenario_summaries": {}, "entrypoint": None, "forwarder_errors": []}
    model = ModelConfig(short_name="m", hf_model_id="org/m", total_params_b=1.0, is_moe=False)
    # a window above GemmaClaw's declared 8192-token output cap (one at the cap is refused)
    with mock.patch("gbench.runners.evals._server_max_model_len", return_value=16384):
        res = runner.run(model, ModelFormat.HF)
    assert state["stopped_at"] is not None and state["stopped_at"] > Q.SUITE_EXIT_GRACE_S
    assert state["name"].startswith("gbench-quality-")
    assert res["suite_stopped_after_summary"] is True and res["suite_exit_code"] == 143
    assert res["pass_rate"] == 100.0 and res["status"] == "completed"


def test_the_container_gets_no_host_environment_and_no_writable_host_path(tmp_path, monkeypatch,
                                                                          gemini, model_up):
    _r, res, seen, _saved, _log = _run_on_stubs(
        tmp_path, monkeypatch, gemini, model_up, [], {"counts": {"total": 1, "passed": 1,
                                                               "failed": 0}}, {}, remote=True)
    argv = seen["argv"]
    image_at = argv.index("gbench-quality:test")
    flags, command = argv[:image_at], argv[image_at + 1:]
    assert command[:4] == ["pnpm", "openclaw", "qa", "suite"]
    assert command[command.index("--alt-model") + 1] == Q.GEMINI_ALT_MODEL
    assert command[command.index("--output-dir") + 1] == Q.CONTAINER_OUTPUT_DIRNAME
    assert flags[flags.index("--network") + 1] == "none" and "--read-only" in flags
    assert flags[flags.index("--cap-drop") + 1] == "ALL"
    assert [flags[i + 1] for i, a in enumerate(flags) if a == "--cap-add"] == ["SETUID", "SETGID"]
    assert flags[flags.index("--security-opt") + 1] == "no-new-privileges"
    assert "--privileged" not in flags and "--net" not in flags and "-p" not in flags
    binds = [flags[i + 1] for i, a in enumerate(flags) if a == "--volume" and ":" in flags[i + 1]]
    assert [b.split(":")[1:] for b in binds] == [[Q.CONTAINER_RELAY_DIR, "ro"],
                                                 [Q.CONTAINER_CONFIG_PATH, "ro"]]
    anon = [flags[i + 1] for i, a in enumerate(flags) if a == "--volume" and ":" not in flags[i + 1]]
    assert anon == [Q.CONTAINER_OUTPUT_DIR, Q.CONTAINER_ARTIFACTS_DIR, Q.CONTAINER_STATUS_DIR,
                    Q.CONTAINER_NPM_CACHE]
    env = dict(flags[i + 1].split("=", 1) for i, a in enumerate(flags) if a == "--env")
    assert env["npm_config_cache"] == Q.CONTAINER_NPM_CACHE and env["npm_config_offline"] == "true"
    assert set(env) == set(res["suite_env"])
    assert env["VLLM_API_KEY"] == Q.KEY_PLACEHOLDER and "PATH" not in env
    for name in HOSTILE_ENV:
        assert name not in env or name == "VLLM_API_KEY", name
    assert not any(s in json.dumps(argv) for s in SECRETS)
    # the docker CLI itself runs without the Gemini keys or any provider variable
    assert not any(s in json.dumps(seen["docker_env"]) for s in (GEM_A, GEM_B, "sk-ISOtest"))
    assert res["model_endpoint_key_at_relay"] is True
    removed = res["non_gemini_env_removed"]
    assert "OPENAI_API_KEY" in removed and all(Q._is_provider_env(n) for n in removed)


# --------------------------------------------------------------------------- #
# the host reads the container's outputs as data only
# --------------------------------------------------------------------------- #
def _tar_stream(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, kind, payload in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, payload
                tf.addfile(info)
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
    return buf.getvalue()


def test_container_outputs_are_read_in_memory_regular_files_only(tmp_path, monkeypatch):
    host_secret = tmp_path / "host-credentials.json"
    host_secret.write_text(GEM_A)
    stream = _tar_stream([
        ("gbench_qa_out", "dir", None),
        ("gbench_qa_out/qa-suite-summary.json", "symlink", str(host_secret)),
        ("gbench_qa_out/scenarios/a/qa-suite-summary.json", "file", b'{"scenarios": []}'),
        ("gbench_qa_out/scenarios/b/qa-suite-summary.json", "file", b"x" * 2048),
        ("gbench_qa_out/run.sh", "file", b"#!/bin/sh\ntouch /tmp/pwned\n"),
        ("gbench_qa_out/scenarios/../../evil/qa-suite-summary.json", "file", b"{}")])
    calls = []

    class FakeCp:
        def __init__(self, argv, stdout=None, stderr=None, env=None):
            calls.append(argv)
            self.stdout, self.stderr = io.BytesIO(stream), io.BytesIO(b"")

        def wait(self):
            return 0

    monkeypatch.setattr(Q.subprocess, "Popen", FakeCp)
    before = sorted(p.name for p in tmp_path.iterdir())
    got = Q.read_container_files("ctr", Q.CONTAINER_OUTPUT_DIR, lambda n: n.endswith(".json"),
                                 env={}, max_file_bytes=1024)
    assert calls == [["docker", "cp", f"ctr:{Q.CONTAINER_OUTPUT_DIR}", "-"]]
    assert set(got) == {"gbench_qa_out/scenarios/a/qa-suite-summary.json",
                        "gbench_qa_out/scenarios/../../evil/qa-suite-summary.json"}
    assert all(GEM_A.encode() not in v for v in got.values())       # the symlink is not followed
    assert sorted(p.name for p in tmp_path.iterdir()) == before     # nothing written to the host
    # the runner only keeps the summary names it expects
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path))
    monkeypatch.setattr(Q, "read_container_files", lambda c, path, wanted, env, **kw: {
        n: b"{}" for n in ("gbench_qa_out/qa-suite-summary.json",
                           "gbench_qa_out/scenarios/ok-1/qa-suite-summary.json",
                           "gbench_qa_out/scenarios/../../evil/qa-suite-summary.json",
                           "gbench_qa_out/run.sh",
                           "gbench-status/entrypoint.json") if wanted(n)})
    out = runner._collect_container_outputs("ctr")
    assert set(out["scenario_summaries"]) == {"ok-1"} and out["summary"] == {}


# --------------------------------------------------------------------------- #
# setup: git on the host never runs a hook, and never writes the checkout
# --------------------------------------------------------------------------- #
_UPSTREAM_API = (Q._RELAY_PATCH_OLD_1 + "  let url: URL;\n  return normalized;\n}\n\n"
                 "export function f(params) {\n  return g({\n" + Q._RELAY_PATCH_OLD_2 + "  });\n}\n")
_UPSTREAM_INDEX = "export const x = {\n" + Q._IMAGE_PATCH_OLD + "};\n"


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          cwd=str(repo), check=True, capture_output=True, text=True).stdout.strip()


def _git_repo(path: Path, extra=None) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    for rel, text in {Q.GEMINI_RELAY_PATCH_FILE: _UPSTREAM_API,
                      Q.GEMINI_IMAGE_PATCH_FILE: _UPSTREAM_INDEX,
                      "package.json": '{"name": "gemmaclaw"}\n', **(extra or {})}.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "c")
    return _git(path, "rev-parse", "HEAD")


def _plant_hooks(repo: Path, marker: Path):
    """Hooks and an fsmonitor that would record the environment they ran with."""
    script = f"#!/bin/sh\nenv >> {marker}\n"
    hooks = repo / ".git" / "hooks"
    for name in ("post-checkout", "post-merge", "post-rewrite", "reference-transaction",
                 "pre-auto-gc", "fsmonitor-watchman", "post-index-change"):
        (hooks / name).write_text(script)
        (hooks / name).chmod(0o755)
    fsmon = repo / "fsmonitor.sh"
    fsmon.write_text(script)
    fsmon.chmod(0o755)
    _git(repo, "config", "core.fsmonitor", str(fsmon))


def _fake_docker(monkeypatch, built):
    """docker image inspect / build faked; git runs for real."""
    real_run = Q.subprocess.run
    state = {"labels": {}}

    def fake_run(cmd, *a, **kw):
        if cmd[0] != "docker":
            return real_run(cmd, *a, **kw)
        built.setdefault("envs", []).append(dict(kw.get("env") or {}))
        if cmd[1:3] == ["image", "inspect"]:
            label = state["labels"].get(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0 if label else 1, (label or "") + "\n", "")
        if cmd[1] == "build":
            ctx = Path(cmd[-1])
            built["files"] = {str(p.relative_to(ctx)): p.read_bytes()
                              for p in ctx.rglob("*") if p.is_file()}
            built["argv"] = list(cmd)
            commit = [a for a in cmd if a.startswith("GEMMACLAW_COMMIT=")][0].split("=", 1)[1]
            state["labels"][cmd[cmd.index("-t") + 1]] = commit
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(cmd)
    monkeypatch.setattr(Q.subprocess, "run", fake_run)


def test_setup_builds_from_git_archive_and_never_runs_a_hook_or_writes_the_checkout(tmp_path,
                                                                                    monkeypatch):
    repo = tmp_path / "gemmaclaw"
    sha = _git_repo(repo)
    marker = tmp_path / "hook-ran.txt"
    _plant_hooks(repo, marker)
    (repo / "untracked-planted.ts").write_text("export const planted = 1;\n")   # not committed
    for k, v in HOSTILE_ENV.items():
        monkeypatch.setenv(k, v)
    snapshot = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts}
    built = {}
    _fake_docker(monkeypatch, built)
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, gemmaclaw_commit=sha))
    assert runner._setup_gemmaclaw_repo() == sha, runner._setup_failure
    assert runner._image == Q.quality_image_tag(sha)
    assert not marker.exists()                                   # no hook, no fsmonitor ran
    assert {p: p.read_bytes() for p in repo.rglob("*")
            if p.is_file() and ".git" not in p.parts} == snapshot
    files = built["files"]
    assert "src/untracked-planted.ts" not in files and "src/.git/config" not in files
    assert Q._IMAGE_PATCH_NEW.encode() in files[f"src/{Q.GEMINI_IMAGE_PATCH_FILE}"]
    assert Q._RELAY_PATCH_NEW_1.encode() in files[f"src/{Q.GEMINI_RELAY_PATCH_FILE}"]
    assert json.loads(files["gemmaclaw_patches.json"])[1]["state"] == "applied"
    for name in (Q.QUALITY_DOCKERFILE, *Q._IMAGE_HELPERS):
        assert files[name] == (ROOT / "docker" / name).read_bytes()
    assert f"GEMMACLAW_COMMIT={sha}" in built["argv"]
    for env in built["envs"]:
        assert not any(s in json.dumps(env) for s in (GEM_A, GEM_B, "sk-ISOtest"))
    # a second setup finds the image and builds nothing
    built.pop("argv")
    assert runner._setup_gemmaclaw_repo() == sha and "argv" not in built
    assert not marker.exists()


def test_gbenchs_own_clone_has_no_working_tree_and_follows_the_remote(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    _git_repo(upstream)
    marker = tmp_path / "hook-ran.txt"
    monkeypatch.setattr(Q, "GEMMACLAW_REMOTE", str(upstream))
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    built = {}
    _fake_docker(monkeypatch, built)
    config = _cfg(tmp_path, gemmaclaw_commit="HEAD")
    config.gemmaclaw_path = None
    runner = Q.QualityBenchmarkRunner(config)
    first = runner._setup_gemmaclaw_repo()
    clone = tmp_path / "cache" / "gemmaclaw"
    assert first == _git(upstream, "rev-parse", "HEAD")
    assert sorted(p.name for p in clone.iterdir()) == [".git"]     # --no-checkout
    _plant_hooks(clone, marker)
    (upstream / "package.json").write_text('{"name": "gemmaclaw", "v": 2}\n')
    _git(upstream, "commit", "-qam", "c2")
    branch = _git(upstream, "rev-parse", "--abbrev-ref", "HEAD")
    runner.config.gemmaclaw_commit = branch
    second = runner._setup_gemmaclaw_repo()
    assert second == _git(upstream, "rev-parse", "HEAD") != first   # fetched, origin/<branch>
    assert not marker.exists()
    assert sorted(p.name for p in clone.iterdir()) == [".git", "fsmonitor.sh"]


def test_the_pinned_commit_wins_even_with_gbench_env_prod(tmp_path, monkeypatch):
    """GBENCH_ENV=prod used to make setup take the checkout's HEAD without fetching. That path
    is gone: the pinned commit is resolved and built whatever GBENCH_ENV says."""
    repo = tmp_path / "gemmaclaw"
    pinned = _git_repo(repo)
    (repo / "package.json").write_text('{"name": "gemmaclaw", "v": 2}\n')
    _git(repo, "commit", "-qam", "c2")
    assert _git(repo, "rev-parse", "HEAD") != pinned
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    monkeypatch.setenv("GBENCH_ENV", "prod")
    built = {}
    _fake_docker(monkeypatch, built)
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, gemmaclaw_commit=pinned))
    assert runner._setup_gemmaclaw_repo() == pinned, runner._setup_failure
    assert f"GEMMACLAW_COMMIT={pinned}" in built["argv"]


def test_a_committed_file_holding_a_key_stops_the_image_build(tmp_path, monkeypatch):
    repo = tmp_path / "gemmaclaw"
    sha = _git_repo(repo, extra={"config/.env": f"GEMINI_API_KEY={GEM_B}\n"})
    monkeypatch.setenv("GEMINI_API_KEYS", f"{GEM_A},{GEM_B}")
    built = {}
    _fake_docker(monkeypatch, built)
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, gemmaclaw_commit=sha))
    assert runner._setup_gemmaclaw_repo() is None
    assert runner._setup_failure.startswith("infra_required: config/.env at GemmaClaw commit")
    assert GEM_B not in runner._setup_failure and "argv" not in built   # never built
    model = ModelConfig(short_name="m", hf_model_id="org/m", total_params_b=1.0, is_moe=False)
    runner._preflight_check = mock.MagicMock(return_value=True)
    with pytest.raises(RuntimeError, match="Failed to set up gemmaclaw repository: "
                                           "infra_required: config/.env"):
        runner.run(model, ModelFormat.HF)


# --------------------------------------------------------------------------- #
# preflight: Docker, not bubblewrap
# --------------------------------------------------------------------------- #
def test_the_preflight_names_docker_when_it_is_missing_or_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path))
    which = {"git": "/usr/bin/git"}
    monkeypatch.setattr(Q.shutil, "which", lambda x: which.get(x))
    outcome = {"docker": subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker "
                                                                "daemon at unix:///var/run/x")}

    def fake_run(cmd, **kw):
        if cmd[0] == "docker":
            return outcome["docker"]
        return subprocess.CompletedProcess(cmd, 0, "git version 2.43.0\n", "")
    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    assert runner._preflight_check() is False
    assert runner._preflight_failure.startswith("infra_required: Docker (the docker CLI) is not "
                                                "on PATH")
    which["docker"] = "/usr/bin/docker"
    assert runner._preflight_check() is False
    assert "the Docker daemon is not reachable: Cannot connect" in runner._preflight_failure
    outcome["docker"] = subprocess.CompletedProcess([], 0, "29.1.3\n", "")
    assert runner._preflight_check() is True and runner._preflight_failure is None
    monkeypatch.setattr(Q, "_DOCKER_DIR", tmp_path / "no-docker-dir")
    assert runner._preflight_check() is False
    assert "quality.Dockerfile is not in this gbench installation" in runner._preflight_failure


# --------------------------------------------------------------------------- #
# the container itself (real Docker): a small image with the real entrypoint and forwarder
# --------------------------------------------------------------------------- #
_PROBE_DOCKERFILE = """FROM python:3.12-slim
RUN printf 'qa:x:10001:10001::/home/qa:/bin/sh\\ngbrelay:x:10002:10002::/nonexistent:/bin/false\\n' \\
      >> /etc/passwd \\
 && printf 'qa:x:10001:\\ngbrelay:x:10002:\\n' >> /etc/group \\
 && install -d -o 10001 -g 10001 -m 0700 /opt/gemmaclaw/gbench_qa_out /opt/gemmaclaw/.artifacts \\
      /home/qa /opt/gbench/npm-cache \\
 && install -d -o 0 -g 10002 -m 0770 /gbench-status \\
 && install -d -m 0755 /opt/gbench /run/gbench-relay /etc/gbench \\
 && touch /etc/gbench/openclaw.json
COPY quality_entrypoint.sh quality_relay_forwarder.py /opt/gbench/
RUN chmod 0755 /opt/gbench/quality_entrypoint.sh && chmod 0644 /opt/gbench/quality_relay_forwarder.py
ENTRYPOINT ["/opt/gbench/quality_entrypoint.sh"]
"""

#: Run in the container as the suite's user: everything the model's exec tool could try.
_PROBE = r'''
import errno, glob, http.client, json, os, re, socket
rep = {"uid": os.getuid()}
rep["capeff"] = [l.split()[1] for l in open("/proc/self/status") if l.startswith("CapEff")][0]
rep["env"] = dict(os.environ)
rep["proc_environ"] = {}
for p in glob.glob("/proc/[0-9]*"):
    try:
        rep["proc_environ"][p] = open(p + "/environ", "rb").read().decode("latin-1")
    except OSError as e:
        rep["proc_environ"][p] = type(e).__name__
fwd = [p for p in glob.glob("/proc/[0-9]*")
       if b"quality_relay_forwarder" in open(p + "/cmdline", "rb").read()]
rep["forwarder_seen"] = bool(fwd)
for what in ("environ", "mem", "fd/0", "maps"):
    try:
        with open(fwd[0] + "/" + what, "rb") as f:
            f.read(16)
        rep["forwarder_" + what] = "readable"
    except OSError as e:
        rep["forwarder_" + what] = type(e).__name__
pat = re.compile(rb"AIza[0-9A-Za-z_\-]{20,}|gbench-relay-[0-9a-f]{48}|ISOtest|hunter2")
rep["file_hits"] = []
for root, dirs, files in os.walk("/"):
    dirs[:] = [d for d in dirs if os.path.join(root, d) not in ("/proc", "/sys", "/dev")]
    for n in files:
        path = os.path.join(root, n)
        try:
            if os.path.islink(path) or os.path.getsize(path) > (32 << 20):
                continue
            with open(path, "rb") as f:
                m = pat.search(f.read())
        except OSError:
            continue
        if m:
            rep["file_hits"].append(path)
def tcp(host, port):
    try:
        socket.create_connection((host, port), timeout=5).close()
        return "connected"
    except OSError as e:
        return errno.errorcode.get(e.errno, type(e).__name__)
rep["metadata_ip"] = tcp("169.254.169.254", 80)
rep["metadata_name"] = tcp("metadata.google.internal", 80)
rep["outside"] = tcp("1.1.1.1", 443)
rep["interfaces"] = sorted(os.listdir("/sys/class/net"))
class U(http.client.HTTPConnection):
    def connect(self):
        s = socket.socket(socket.AF_UNIX); s.settimeout(10)
        s.connect("/run/gbench-relay/relay.sock"); self.sock = s
def call(conn, method, path, body=b"{}"):
    try:
        conn.request(method, path, body=body if method == "POST" else None,
                     headers={"x-goog-api-key": "gbench-relay", "Content-Type": "application/json"})
        r = conn.getresponse(); r.read(); return r.status
    except OSError as e:
        return type(e).__name__
rep["relay_direct"] = call(U("x"), "POST", "/v1beta/models/gemini-3.1-pro-preview:generateContent")
fw = lambda: http.client.HTTPConnection("127.0.0.1", 18431, timeout=30)
rep["via_forwarder_alt"] = call(fw(), "POST", "/v1beta/models/gemini-3.1-pro-preview:generateContent")
rep["via_forwarder_files"] = call(fw(), "GET", "/v1beta/files")
rep["via_forwarder_chat"] = call(fw(), "POST", "/model/chat/completions")
def write(path):
    try:
        with open(path, "a") as f:
            f.write("x")
        return "written"
    except OSError as e:
        return errno.errorcode.get(e.errno, type(e).__name__)
rep["writes"] = {p: write(p) for p in ("/opt/gemmaclaw/planted.js", "/run/gbench-relay/x",
                                         "/etc/gbench/openclaw.json", "/gbench-status/x",
                                         "/opt/gbench/quality_entrypoint.sh", "/usr/bin/x",
                                         "/opt/gemmaclaw/gbench_qa_out/ok", "/tmp/ok",
                                         "/home/qa/ok", "/opt/gemmaclaw/.artifacts/ok")}
rep["host_home_visible"] = os.path.exists(HOST_HOME)
rep["docker_socket_visible"] = os.path.exists("/var/run/docker.sock")
print("PROBE" + json.dumps(rep))
'''


def _docker_ready() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "info"], capture_output=True).returncode == 0


def _probe_image(tmp_path) -> str:
    ctx = tmp_path / "probe-ctx"
    ctx.mkdir()
    (ctx / "Dockerfile").write_text(_PROBE_DOCKERFILE)
    for name in Q._IMAGE_HELPERS:
        shutil.copy2(ROOT / "docker" / name, ctx / name)
    tag = "gbench-quality-probe:test"
    build_env = {k: v for k, v in os.environ.items() if k not in HOSTILE_ENV}
    res = subprocess.run(["docker", "build", "-q", "-t", tag, str(ctx)], capture_output=True,
                         text=True, timeout=900, env=build_env)
    assert res.returncode == 0, res.stderr
    return tag


def test_inside_the_container_no_key_no_token_no_network_no_writable_host_path(tmp_path,
                                                                             monkeypatch,
                                                                             gemini, model_up):
    """The proof, with the runner's own docker command line, entrypoint and forwarder: the suite's
    user finds no key or token in its environment, in any process it can read, or in any file;
    cannot read the forwarder; reaches no network (the cloud metadata server included), only the
    forwarder; the relay refuses it without the token; the checkout and the host paths are
    read-only. Where Docker is not usable, the runner refuses to start, naming Docker."""
    for k, v in HOSTILE_ENV.items():
        monkeypatch.setenv(k, v)
    if not _docker_ready():
        runner = Q.QualityBenchmarkRunner(_cfg(tmp_path))
        assert runner._preflight_check() is False
        assert "Docker" in runner._preflight_failure
        return
    image = _probe_image(tmp_path)
    monkeypatch.setattr(Q, "GEMINI_API_ORIGIN", gemini.url)
    relay_dir = Path(__import__("tempfile").mkdtemp(prefix="gbq_p_", dir="/tmp")) / "relay"
    relay_dir.mkdir()
    relay_dir.chmod(0o711)
    relay = Q.KeyRelay([GEM_A], model_upstream=model_up.url + "/v1",
                       model_key=ENDPOINT_KEY).start(str(relay_dir / Q.RELAY_SOCKET_NAME))
    config = tmp_path / "openclaw.json"
    config.write_text('{"models": {}}')
    config.chmod(0o644)
    env, _ = Q.sandbox_suite_env(os.environ, home=Q.CONTAINER_HOME,
                                 npm_cache=f"{Q.CONTAINER_HOME}/.npm",
                                 config_path=Q.CONTAINER_CONFIG_PATH,
                                 relay_base_url=Q.CONTAINER_GEMINI_BASE_URL,
                                 model_api_key=Q.KEY_PLACEHOLDER)
    name = f"gbench-quality-probe-{os.getpid()}"
    probe = _PROBE.replace("HOST_HOME", repr(os.path.expanduser("~")))
    argv = Q.container_run_argv(image, name, relay_dir=str(relay_dir), config_file=str(config),
                                env=env, command=["python3", "-c", probe])
    decoy = subprocess.Popen(["sleep", "120"], env=dict(os.environ))    # holds the keys
    try:
        out = subprocess.run(argv, input=relay.token + "\n", capture_output=True, text=True,
                             timeout=600, env=Q.provider_free_env(os.environ)[0])
        inspect = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    finally:
        subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True)
        decoy.kill()
        decoy.wait()
        relay.stop()
        shutil.rmtree(relay_dir.parent, ignore_errors=True)
    assert out.returncode == 0, out.stdout + out.stderr
    rep = json.loads(out.stdout.split("PROBE", 1)[1])
    assert rep["uid"] == Q.SUITE_UID and int(rep["capeff"], 16) == 0
    text = out.stdout + out.stderr + inspect.stdout
    for secret in (relay.token, *SECRETS):
        assert secret not in text, secret
    assert rep["file_hits"] == []
    assert rep["forwarder_seen"] is True
    for what in ("environ", "mem", "fd/0", "maps"):
        assert rep["forwarder_" + what] == "PermissionError", what
    assert len(rep["proc_environ"]) < 10                        # the container's processes only
    assert rep["metadata_ip"] in ("ENETUNREACH", "EHOSTUNREACH")
    assert rep["metadata_name"] == "gaierror" and rep["outside"] in ("ENETUNREACH",
                                                                     "EHOSTUNREACH")
    assert rep["interfaces"] == ["lo"]
    assert rep["relay_direct"] == 401                           # the socket, without the token
    assert rep["via_forwarder_alt"] == 200 and rep["via_forwarder_chat"] == 200
    assert rep["via_forwarder_files"] == 403
    writes = rep["writes"]
    # a new file in the checkout, the relay's directory or /usr: the file system is read-only
    for path in ("/opt/gemmaclaw/planted.js", "/run/gbench-relay/x", "/usr/bin/x"):
        assert writes[path] == "EROFS", (path, writes[path])
    # existing files (the entrypoint, gbench's openclaw.json): refused, whichever check comes first
    for path in ("/etc/gbench/openclaw.json", "/opt/gbench/quality_entrypoint.sh"):
        assert writes[path] in ("EROFS", "EACCES"), (path, writes[path])
    assert writes["/gbench-status/x"] == "EACCES"                # gbench's status: not the suite's
    for path in ("/opt/gemmaclaw/gbench_qa_out/ok", "/tmp/ok", "/home/qa/ok",
                 "/opt/gemmaclaw/.artifacts/ok"):
        assert writes[path] == "written", path
    assert rep["host_home_visible"] is False and rep["docker_socket_visible"] is False
    # upstream saw the real keys, never the token or the placeholder as a credential
    assert [r["headers"]["x-goog-api-key"] for r in gemini.requests] == [GEM_A]
    assert model_up.requests[0]["headers"]["authorization"] == f"Bearer {ENDPOINT_KEY}"
    rep_relay = relay.report()
    assert rep_relay["rejected"] == 1 and rep_relay["refused"][0]["path"] == "/v1beta/files"
    assert rep_relay["harness_errors"] == {}


def test_the_container_refuses_to_start_without_the_token(tmp_path):
    if not _docker_ready():
        return
    image = _probe_image(tmp_path)
    name = f"gbench-quality-probe-nt-{os.getpid()}"
    relay_dir = Path(__import__("tempfile").mkdtemp(prefix="gbq_n_", dir="/tmp"))
    config = tmp_path / "c.json"
    config.write_text("{}")
    try:
        out = subprocess.run(Q.container_run_argv(image, name, relay_dir=str(relay_dir),
                                                  config_file=str(config), env={},
                                                  command=["true"]),
                             input="", capture_output=True, text=True, timeout=300)
    finally:
        subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True)
        shutil.rmtree(relay_dir, ignore_errors=True)
    assert out.returncode == 97 and "no relay token" in out.stderr


# --------------------------------------------------------------------------- #
# the image files and the docs in step with the code
# --------------------------------------------------------------------------- #
def test_the_image_files_agree_with_the_runner():
    fwd = (ROOT / "docker" / "quality_relay_forwarder.py").read_text()
    for needle in (f"FORWARD_PORT = {Q.FORWARDER_PORT}",
                   f'RELAY_SOCKET = "{Q.CONTAINER_RELAY_DIR}/{Q.RELAY_SOCKET_NAME}"',
                   f'STATUS_DIR = "{Q.CONTAINER_STATUS_DIR}"',
                   f'RELAY_TOKEN_HEADER = "{Q.RELAY_TOKEN_HEADER}"',
                   f'MODEL_PREFIX = "{Q.KeyRelay.MODEL_PREFIX}"'):
        assert needle in fwd, needle
    entry = (ROOT / "docker" / "quality_entrypoint.sh").read_text()
    assert f"STATUS_DIR={Q.CONTAINER_STATUS_DIR}" in entry and "cd /opt/gemmaclaw" in entry
    for code in (94, 95, 96, 97, 98):
        assert f'" {code}' in entry or f"{code}\n" in entry, code
    dockerfile = (ROOT / "docker" / Q.QUALITY_DOCKERFILE).read_text()
    for needle in (f"useradd -u {Q.SUITE_UID}", "useradd -u 10002",
                   f"install -d -o qa -g qa -m 0700 {Q.CONTAINER_OUTPUT_DIR} "
                   f"{Q.CONTAINER_ARTIFACTS_DIR}",
                   f"install -d -o root -g gbrelay -m 0770 {Q.CONTAINER_STATUS_DIR}",
                   Q.CONTAINER_RELAY_DIR, Q.CONTAINER_CONFIG_PATH.rsplit("/", 1)[0],
                   "gbench.quality.gemmaclaw_commit", "metadata.tar", "gemmaclaw_patches.json",
                   "node:24-bookworm@sha256:"):
        assert needle in dockerfile, needle
    assert Q.quality_image_tag("a" * 40).startswith("gbench-quality:aaaaaaaaaaaa-")


def test_the_quality_doc_describes_the_container():
    doc = (ROOT / "docs" / "quality.md").read_text(encoding="utf-8")
    flat = " ".join(doc.split())
    for needle in ("docker/quality.Dockerfile", "`--network none`", "`--read-only`",
                   "no-new-privileges", "169.254.169.254", "metadata.google.internal",
                   "`harness_error_scenarios`", "`unattributed_harness_errors`",
                   "`pass_rate_withheld_reason`", "`relay_refused_requests`",
                   "`relay_rejected_requests`", "`model_endpoint_key_at_relay`", "`suite_env`",
                   "`gemini_non_answer`", "`gemini_truncated`", "git archive",
                   "core.hooksPath=/dev/null", f"127.0.0.1:{Q.FORWARDER_PORT}",
                   Q.RELAY_TOKEN_HEADER, "tests/test_quality_isolation.py"):
        assert needle in doc, needle
    for model in sorted(Q.ALLOWED_GEMINI_MODELS):
        assert f"`{model}`" in doc, model
    for method in sorted(Q.ALLOWED_GEMINI_METHODS):
        assert f"`{method}`" in doc, method
    for method, path in sorted(Q.ALLOWED_MODEL_CALLS):
        assert f"`{method} {path}`" in doc, path
    assert "bubblewrap sandbox" not in flat and "`bwrap`" not in doc
    assert "Node.js, pnpm and bubblewrap are not needed on the host" in flat
    assert "host network is shared" not in flat
