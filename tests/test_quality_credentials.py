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

"""No credential reaches the model under test through the GemmaClaw QA suite, and every Gemini
call the suite makes is recorded.

* The suite runs in gbench's local container: its environment is a fixed set of variables, the
  processes it can read and the files it can read hold no Gemini key, no key of another provider,
  no endpoint key and no relay token, and it has no network (checked with a real container run
  from the runner's own docker command line against a host process and a host file that do hold
  them; tests/test_quality_isolation.py checks the network, the forwarder and the read-only file
  system).
* The google plugin and the model endpoint get the in-container forwarder's base URLs and a
  placeholder, never a key; the forwarder adds the relay token, and the relay forwards to the
  Gemini API only, with the real key, and counts every call by capability, model, method and
  status. Failed calls are harness errors (status completed_with_errors).
* A remote endpoint's own key stays with the relay too.
* gbench's two recorded edits of the checkout apply exactly once, revert, and refuse a drifted file;
  they are made in the image's build context, never in a checkout.
* Scenarios that need a model provider gbench does not run are reported as infra_required, apart
  from the scored scenarios.

Everything runs on stubs: no GemmaClaw build, no vLLM, no Gemini call; files are written under tmp
only (the container proof needs Docker, a prerequisite of the quality runner).
"""

import http.client
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit, parse_qsl

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import quality as Q

ROOT = Path(__file__).resolve().parents[1]

GEM_A = "AIzaSyCREDtestGemA000000000000001"
GEM_B = "AIzaSyCREDtestGemB000000000000002"
ENDPOINT_KEY = "vllm-CREDtest-endpoint-key-0001"
OPENAI = "sk-CREDtest-openai-0000000000001"
SECRETS = (GEM_A, GEM_B, ENDPOINT_KEY, OPENAI)
#: A Gemini reply the relay counts as an answer.
ANSWER = b'{"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}'


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class _Upstream:
    """A stub HTTP upstream: records every request and answers from `script` (status, body,
    chunked), then 200 with a JSON body."""

    def __init__(self):
        self.requests = []
        self.script = []
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
                status, data, chunked = (outer.script.pop(0) if outer.script
                                         else (200, b'{"ok": true}', False))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    half = len(data) // 2
                    for part in (data[:half], data[half:]):
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                        self.wfile.flush()
                        if chunked == "broken":          # the upstream drops mid-stream
                            self.close_connection = True
                            return
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                self.wfile.flush()

            do_GET = do_POST = _any

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()




@pytest.fixture
def upstream():
    up = _Upstream()
    yield up
    up.close()


class _UnixConn(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("relay", timeout=30)
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        s.connect(self._path)
        self.sock = s


def _call(relay, path, headers=None, body=b"{}", method="POST", token=True):
    """(status, body) of one request to the relay's unix socket, as the forwarder sends it (with
    the relay token unless `token` is False); the body read to the end, chunked or not."""
    hdr = dict(headers or {})
    if token:
        hdr[Q.RELAY_TOKEN_HEADER] = relay.token
    conn = _UnixConn(relay.socket_path)
    conn.request(method, path, body=body if method != "GET" else None, headers=hdr)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def _start(relay):
    sock_dir = tempfile.mkdtemp(prefix="gbq_c_", dir="/tmp")
    return relay.start(os.path.join(sock_dir, Q.RELAY_SOCKET_NAME))


def _stop(relay):
    relay.stop()
    shutil.rmtree(os.path.dirname(relay.socket_path), ignore_errors=True)


@pytest.fixture
def relay_to(upstream, monkeypatch):
    monkeypatch.setattr(Q, "GEMINI_API_ORIGIN", upstream.url)
    relays = []

    def make(**kw):
        relay = _start(Q.KeyRelay([GEM_A, GEM_B], **kw))
        relays.append(relay)
        return relay
    yield make
    for relay in relays:
        _stop(relay)


# --------------------------------------------------------------------------- #
# the relay
# --------------------------------------------------------------------------- #
def test_the_relay_upstream_is_the_gemini_api_and_no_variable_changes_it(monkeypatch):
    assert Q.GEMINI_API_ORIGIN == "https://generativelanguage.googleapis.com"
    for name in ("GEMINI_OPENAI_BASE_URL", "GBENCH_GEMINI_RELAY_UPSTREAM", Q.GEMINI_RELAY_ENV,
                 "HTTPS_PROXY", "GOOGLE_GEMINI_BASE_URL"):
        monkeypatch.setenv(name, "https://evil.example.test/v1beta")
    relay = Q.KeyRelay([GEM_A])
    assert relay._gemini.geturl() == "https://generativelanguage.googleapis.com"
    assert relay.token.startswith("gbench-relay-") and len(relay.token) > 40
    assert Q.KeyRelay([GEM_A]).token != relay.token
    with pytest.raises(ValueError):
        Q.KeyRelay([])


def test_the_relay_forwards_with_the_real_key_and_never_the_token(relay_to, upstream):
    relay = relay_to()
    sse = (b'data: {"candidates": [{"content": {"parts": [{"text": "OK"}]}, '
           b'"finishReason": "STOP"}]}\r\n\r\n')
    upstream.script = [(200, sse, True), (200, b'{"embedding": {"values": [0.1]}}', False)]
    status, data = _call(relay, "/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent"
                                "?alt=sse",
                         {"x-goog-api-key": Q.KEY_PLACEHOLDER, "Content-Type": "application/json"},
                         b'{"contents": []}')
    assert (status, data) == (200, sse)                     # streamed back as it came
    # a stray key= and Authorization are dropped as well
    status, _ = _call(relay, "/v1beta/models/gemini-embedding-001:embedContent?key=junk",
                      {"Content-Type": "application/json", "Authorization": "Bearer junk"}, b'{}')
    assert status == 200
    first, second = upstream.requests
    assert first["path"] == "/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent"
    assert first["query"] == {"alt": "sse"} and second["query"] == {}
    assert {first["headers"]["x-goog-api-key"], second["headers"]["x-goog-api-key"]} == {GEM_A,
                                                                                          GEM_B}
    for req in upstream.requests:
        assert "authorization" not in req["headers"]
        assert Q.RELAY_TOKEN_HEADER not in req["headers"]
        assert relay.token not in json.dumps(req["headers"]) + json.dumps(req["query"])
    rep = relay.report()
    assert rep["harness_errors"] == {}
    assert {(c["capability"], c["model"], c["method"], c["requests"], c["statuses"]["200"])
            for c in rep["gemini_calls"]} == {
        ("alternate model", "gemini-3.1-pro-preview", "streamGenerateContent", 1, 1),
        ("memory embeddings", "gemini-embedding-001", "embedContent", 1, 1)}


def test_the_relay_refuses_a_request_without_the_token(relay_to, upstream):
    relay = relay_to()
    path = "/v1beta/models/gemini-3.1-pro-preview:generateContent"
    assert _call(relay, path, {}, token=False)[0] == 401
    assert _call(relay, path, {"x-goog-api-key": GEM_A}, token=False)[0] == 401   # a real key
    assert _call(relay, path, {"Authorization": f"Bearer {relay.token}"}, token=False)[0] == 401
    assert upstream.requests == []
    rep = relay.report()
    # counted apart: an unauthenticated request is not a harness error
    assert rep["rejected"] == 3 and rep["harness_errors"] == {}


def test_the_relay_tries_the_next_key_and_counts_failed_calls_as_harness_errors(relay_to,
                                                                                  upstream):
    relay = relay_to()
    path = "/v1beta/models/gemini-3.1-pro-preview:generateContent"
    upstream.script = [(429, b'{"error": "quota"}', False), (200, ANSWER, False)]
    assert _call(relay, path) == (200, ANSWER)
    keys = [r["headers"]["x-goog-api-key"] for r in upstream.requests]
    assert sorted(keys) == [GEM_A, GEM_B]                            # the other key on 429
    upstream.script = [(503, b'{"error": "down"}', False), (503, b'{"error": "down"}', False)]
    assert _call(relay, path)[0] == 503
    rep = relay.report()
    assert rep["harness_errors"] == {"gemini_http_503": 1}
    [row] = rep["gemini_calls"]
    assert row["statuses"] == {"200": 1, "503": 1} and row["requests"] == 2


def test_a_response_that_breaks_off_is_a_harness_error(relay_to, upstream):
    relay = relay_to()
    upstream.script = [(200, b'data: {"candidates": []}\r\n\r\ndata: {"more": 1}\r\n\r\n',
                        "broken")]
    with pytest.raises(http.client.HTTPException):
        _call(relay, "/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent")
    rep = relay.report()
    assert rep["harness_errors"] == {"gemini_stream_broken": 1}
    assert rep["gemini_calls"][0]["statuses"] == {"200": 1}


def test_an_unreachable_gemini_api_is_a_harness_error(monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                                        # nothing listens there
    monkeypatch.setattr(Q, "GEMINI_API_ORIGIN", f"http://127.0.0.1:{port}")
    relay = _start(Q.KeyRelay([GEM_A]))
    try:
        status, data = _call(relay, "/v1beta/models/gemini-3.1-pro-preview:generateContent")
        assert status == 502 and GEM_A.encode() not in data
        assert relay.report()["harness_errors"] == {"gemini_unreachable": 1}
    finally:
        _stop(relay)


def test_the_endpoint_key_stays_with_the_relay(relay_to):
    model = _Upstream()
    try:
        relay = relay_to(model_upstream=model.url + "/v1", model_key=ENDPOINT_KEY)
        status, _ = _call(relay, "/model/chat/completions",
                          {"Authorization": f"Bearer {Q.KEY_PLACEHOLDER}"}, b'{"stream": true}')
        assert status == 200
        [req] = model.requests
        assert req["path"] == "/v1/chat/completions"
        assert req["headers"]["authorization"] == f"Bearer {ENDPOINT_KEY}"
        assert _call(relay, "/model/chat/completions", {}, token=False)[0] == 401
        rep = relay.report()
        assert rep["model_endpoint_calls"] == [{"capability": "model under test", "model": "",
                                                "method": "/chat/completions", "requests": 1,
                                                "statuses": {"200": 1}}]
        assert rep["harness_errors"] == {} and rep["rejected"] == 1
    finally:
        model.close()


def test_every_gemini_capability_is_named():
    c = Q.KeyRelay.classify
    img = json.dumps({"contents": [{"parts": [{"inlineData": {"mimeType": "image/png",
                                                              "data": "x"}}]}]}).encode()
    search = json.dumps({"contents": [], "tools": [{"google_search": {}}]}).encode()
    assert c("/v1beta/models/gemini-embedding-001:batchEmbedContents", b"{}")[0] == \
        "memory embeddings"
    assert c("/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent", b"{}")[0] == \
        "alternate model"
    assert c("/v1beta/models/gemini-3.1-flash-image-preview:generateContent", img)[0] == \
        "image generation"
    assert c("/v1beta/models/gemini-3-flash-preview:generateContent", img) == \
        ("media understanding", "gemini-3-flash-preview", "generateContent")
    assert c("/v1beta/models/gemini-2.5-flash:generateContent", search)[0] == \
        "web search (Google Search grounding)"
    assert c("/v1beta/models/gemini-2.5-flash-preview-tts:generateContent", b"{}")[0] == "speech"
    assert c("/v1beta/models/lyria-3-clip-preview:generateContent", b"{}")[0] == \
        "music generation"
    assert c("/v1beta/models/veo-3.1-generate-preview:predictLongRunning", b"{}")[0] == \
        "video generation"
    assert c("/v1beta/models/gemini-2.5-flash:generateContent", b'{"contents": []}')[0] == \
        "text generation"




# --------------------------------------------------------------------------- #
# the container
# --------------------------------------------------------------------------- #
def test_the_suite_env_holds_no_credential():
    environ = {"GEMINI_API_KEY": GEM_A, "GEMINI_API_KEYS": f"{GEM_A},{GEM_B}",
               "VLLM_API_KEY": ENDPOINT_KEY, "OPENAI_API_KEY": OPENAI, "NPM_TOKEN": "npm_TEST1",
               "SLACK_BOT_TOKEN": "xoxb-TEST1", "DB_PASSWORD": "pw-TEST1",
               "SOME_SERVICE_SECRET": "sec-TEST1", "GOOGLE_APPLICATION_CREDENTIALS": "/k.json",
               # the Gemini key under names no name filter knows, a proxy URL with a password,
               # and keys of other services named *_KEY
               "GEMINI_KEY": GEM_A, "GENAI_KEY": GEM_A, "LLM_KEY": GEM_B,
               "GEMINI_API_KEY_BACKUP": GEM_B,
               "HTTPS_PROXY": "http://u:hunter2TEST@proxy:3128", "KAGGLE_KEY": "kaggle-TEST1",
               "SERPAPI_KEY": "serp-TEST1",
               "SSH_AUTH_SOCK": "/run/user/1/ssh", "DBUS_SESSION_BUS_ADDRESS": "unix:x",
               "XDG_CONFIG_HOME": "/h/.config", "KRB5CCNAME": "/tmp/krb5cc_1",
               "CLOUDSDK_CONFIG": "/h/.config/gcloud", "HOME": "/h", "PATH": "/usr/bin",
               "LANG": "C.UTF-8"}
    env, removed = Q.sandbox_suite_env(environ, home=Q.CONTAINER_HOME,
                                       npm_cache=Q.CONTAINER_NPM_CACHE,
                                       config_path=Q.CONTAINER_CONFIG_PATH,
                                       relay_base_url=Q.CONTAINER_GEMINI_BASE_URL,
                                       model_api_key=Q.KEY_PLACEHOLDER)
    for secret in (GEM_A, GEM_B, ENDPOINT_KEY, OPENAI, "npm_TEST1", "xoxb-TEST1", "pw-TEST1",
                   "sec-TEST1", "/k.json", "hunter2TEST", "kaggle-TEST1", "serp-TEST1"):
        assert all(secret not in v for v in env.values()), secret
    # nothing of the host environment is taken over, whatever its name
    assert not set(env) & set(environ) - {"HOME", "LANG", "VLLM_API_KEY"}
    assert env["HOME"] == Q.CONTAINER_HOME and env["LANG"] == "C.UTF-8" and "PATH" not in env
    assert env["npm_config_cache"] == Q.CONTAINER_NPM_CACHE
    assert env[Q.GEMINI_RELAY_ENV] == Q.CONTAINER_GEMINI_BASE_URL
    assert env["VLLM_API_KEY"] == Q.KEY_PLACEHOLDER and env["OPENCLAW_CONFIG_PATH"] == \
        Q.CONTAINER_CONFIG_PATH
    assert {"OPENAI_API_KEY", "NPM_TOKEN", "SLACK_BOT_TOKEN", "DB_PASSWORD",
            "SOME_SERVICE_SECRET", "GOOGLE_APPLICATION_CREDENTIALS"} <= set(removed)
    # the host-side git and docker commands do not get them either
    host = Q.host_env(environ)
    for secret in (GEM_A, GEM_B, ENDPOINT_KEY, OPENAI, "hunter2TEST", "kaggle-TEST1",
                   "serp-TEST1"):
        assert all(secret not in v for v in host.values()), secret
    assert host["PATH"] == "/usr/bin" and host["HOME"] == "/h"


def test_the_sandbox_command_isolates_processes_and_shows_only_what_it_binds():
    argv = Q.container_run_argv("gbench-quality:x", "gbench-quality-t",
                                relay_dir="/tmp/r/relay", config_file="/tmp/r/openclaw.json",
                                env={"HOME": Q.CONTAINER_HOME},
                                command=["pnpm", "openclaw", "qa", "suite"])
    assert argv[:2] == ["docker", "run"]
    image_at = argv.index("gbench-quality:x")
    flags = argv[:image_at]
    assert argv[image_at + 1:] == ["pnpm", "openclaw", "qa", "suite"]
    for flag in ("--read-only", "--interactive", "--init"):
        assert flag in flags
    assert flags[flags.index("--network") + 1] == "none"
    assert flags[flags.index("--cap-drop") + 1] == "ALL"
    assert flags[flags.index("--security-opt") + 1] == "no-new-privileges"
    binds = [flags[i + 1] for i, a in enumerate(flags) if a == "--volume" and ":" in flags[i + 1]]
    assert binds == [f"/tmp/r/relay:{Q.CONTAINER_RELAY_DIR}:ro",
                     f"/tmp/r/openclaw.json:{Q.CONTAINER_CONFIG_PATH}:ro"]
    tmpfs = [flags[i + 1].split(":")[0] for i, a in enumerate(flags) if a == "--tmpfs"]
    assert tmpfs == ["/tmp", Q.CONTAINER_HOME]
    home = os.path.expanduser("~")
    assert not any(a == home or a.startswith(home + os.sep) for a in flags)
    assert "/var/run/docker.sock" not in json.dumps(argv) and "--privileged" not in flags
    assert [flags[i + 1] for i, a in enumerate(flags) if a == "--env"] == [
        f"HOME={Q.CONTAINER_HOME}"]


def _cfg(tmp_path, repo, **kw):
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "results"
    config.results_dir.mkdir(exist_ok=True)
    config.gemmaclaw_commit = "0123456789abcdef0123456789abcdef01234567"
    config.gemmaclaw_path = str(repo)
    for k, v in kw.items():
        setattr(config, k, v)
    config.initialize()
    return config


def _docker_ready() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "info"], capture_output=True).returncode == 0


_PROBE_DOCKERFILE = """FROM python:3.12-slim
RUN printf 'qa:x:10001:10001::/home/qa:/bin/sh\\ngbrelay:x:10002:10002::/nonexistent:/bin/false\\n' \\
      >> /etc/passwd \\
 && printf 'qa:x:10001:\\ngbrelay:x:10002:\\n' >> /etc/group \\
 && install -d -o 10001 -g 10001 -m 0700 /opt/gemmaclaw/gbench_qa_out /opt/gemmaclaw/.artifacts \\
      /home/qa \\
 && install -d -o 0 -g 10002 -m 0770 /gbench-status \\
 && install -d -o 10001 -g 10001 -m 0755 /opt/gbench/npm-cache \\
 && install -d -m 0755 /run/gbench-relay /etc/gbench \\
 && touch /etc/gbench/openclaw.json
COPY quality_entrypoint.sh quality_relay_forwarder.py /opt/gbench/
RUN chmod 0755 /opt/gbench/quality_entrypoint.sh && chmod 0644 /opt/gbench/quality_relay_forwarder.py
ENTRYPOINT ["/opt/gbench/quality_entrypoint.sh"]
"""


def test_the_sandboxed_suite_sees_no_key_in_its_env_its_processes_or_its_files(tmp_path,
                                                                                  monkeypatch):
    """The proof: the runner's own container command line, entrypoint and forwarder, around a
    probe that prints the environment, every /proc/*/environ it can see, and every file holding a
    key. A host process started with the keys in its environment (as gbench itself is) and a host
    file holding them are invisible; the same probe outside the container finds the keys, so the
    probe works. Where Docker is not usable, the runner never starts the suite: its preflight
    check refuses, naming Docker."""
    for name, value in (("GEMINI_API_KEY", GEM_A), ("GEMINI_API_KEYS", f"{GEM_A},{GEM_B}"),
                        ("VLLM_API_KEY", ENDPOINT_KEY), ("OPENAI_API_KEY", OPENAI),
                        ("GBENCH_CACHE_DIR", str(tmp_path / "cache"))):
        monkeypatch.setenv(name, value)
    if not _docker_ready():
        runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, tmp_path))
        assert runner._preflight_check() is False
        assert "Docker" in runner._preflight_failure
        return
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Dockerfile").write_text(_PROBE_DOCKERFILE)
    for name in ("quality_entrypoint.sh", "quality_relay_forwarder.py"):
        shutil.copy2(ROOT / "docker" / name, ctx / name)
    image = "gbench-quality-probe:cred"
    built = subprocess.run(["docker", "build", "-q", "-t", image, str(ctx)], capture_output=True,
                           text=True, timeout=900)
    assert built.returncode == 0, built.stderr
    decoy = subprocess.Popen(["sleep", "120"], env=dict(os.environ))   # holds the keys
    host_file = tmp_path / "hosthome" / ".bashrc"
    host_file.parent.mkdir()
    host_file.write_text(f"export GEMINI_API_KEY={GEM_A}\nexport VLLM_API_KEY={ENDPOINT_KEY}\n")
    relay = _start(Q.KeyRelay([GEM_A, GEM_B]))
    name = f"gbench-quality-cred-{os.getpid()}"
    try:
        config = tmp_path / "openclaw.json"
        config.write_text('{"models": {}}')
        config.chmod(0o644)
        os.chmod(os.path.dirname(relay.socket_path), 0o711)
        env, _removed = Q.sandbox_suite_env(
            os.environ, home=Q.CONTAINER_HOME, npm_cache=Q.CONTAINER_NPM_CACHE,
            config_path=Q.CONTAINER_CONFIG_PATH, relay_base_url=Q.CONTAINER_GEMINI_BASE_URL,
            model_api_key=Q.KEY_PLACEHOLDER)
        probe = (
            "echo ::ENV::; env; echo ::PROC::; "
            "for f in /proc/[0-9]*/environ; do tr '\\0' '\\n' < \"$f\"; done 2>/dev/null; "
            "echo ::PIDS:: $(ls -d /proc/[0-9]* | wc -l); "
            f"test -e {host_file} && echo HOST-FILE-VISIBLE; "
            f"test -e {os.path.expanduser('~')}/.bashrc && echo HOST-HOME-VISIBLE; "
            "echo ::GREP::; grep -rIlE 'AIza|CREDtest|gbench-relay-[0-9a-f]{48}' "
            "/tmp /etc /home /root /var /opt /run /gbench-status 2>/dev/null; echo ::END::")
        argv = Q.container_run_argv(image, name, relay_dir=os.path.dirname(relay.socket_path),
                                    config_file=str(config), env=env,
                                    command=["sh", "-c", probe])
        out = subprocess.run(argv, input=relay.token + "\n", capture_output=True, text=True,
                             timeout=300, env=Q.host_env(os.environ))
        assert out.returncode == 0, out.stderr
        text = out.stdout
        assert "::END::" in text and "::GREP::\n::END::" in text        # no file holds a key
        for secret in (*SECRETS, relay.token):
            assert secret not in text + out.stderr, secret
        assert "HOST-FILE-VISIBLE" not in text and "HOST-HOME-VISIBLE" not in text
        assert int(text.split("::PIDS::")[1].split()[0]) < 10        # only the container's own
        assert f"HOME={Q.CONTAINER_HOME}" in text and f"VLLM_API_KEY={Q.KEY_PLACEHOLDER}" in text
        control = subprocess.run(["sh", "-c", f"tr '\\0' '\\n' < /proc/{decoy.pid}/environ; "
                                              f"cat {host_file}"],
                                 capture_output=True, text=True, timeout=30)
        assert GEM_A in control.stdout and ENDPOINT_KEY in control.stdout
    finally:
        subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True)
        _stop(relay)
        decoy.kill()
        decoy.wait()


def test_the_preflight_needs_docker_and_not_bubblewrap(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, tmp_path))
    # no bubblewrap, no node, no npx on this host: the suite runs in the container
    which = {"git": "/usr/bin/git"}
    monkeypatch.setattr(Q.shutil, "which", lambda x: which.get(x))
    outcome = {"docker": subprocess.CompletedProcess([], 1, "", "permission denied while trying "
                                                                "to connect to the Docker daemon")}

    def fake_run(cmd, **kw):
        name = os.path.basename(cmd[0])
        assert name in ("git", "docker"), cmd                 # nothing else is probed
        if name == "docker":
            return outcome["docker"]
        return subprocess.CompletedProcess(cmd, 0, "git version 2.43.0\n", "")

    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    assert runner._preflight_check() is False
    assert "infra_required: Docker (the docker CLI) is not on PATH" in runner._preflight_failure
    which["docker"] = "/usr/bin/docker"
    assert runner._preflight_check() is False
    assert "permission denied while trying to connect" in runner._preflight_failure
    assert runner._preflight_failure.startswith("infra_required: the Docker daemon is not")
    outcome["docker"] = subprocess.CompletedProcess([], 0, "29.1.3\n", "")
    assert runner._preflight_check() is True and runner._preflight_failure is None


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          cwd=str(repo), check=True, capture_output=True, text=True).stdout.strip()


def test_a_file_the_sandbox_shows_that_holds_a_key_stops_the_run(tmp_path, monkeypatch):
    """The image shows the model under test every file of the commit: a committed file holding a
    key stops the build before docker runs (the path only, never the value)."""
    repo = tmp_path / "gemmaclaw"
    repo.mkdir()
    (repo / ".env").write_text(f"GEMINI_API_KEY={GEM_B}\n")
    assert Q.find_key_exposure([GEM_B], roots=[], files=[str(repo / ".env")]) == str(repo / ".env")
    assert Q.find_key_exposure([GEM_A], roots=[str(repo)]) is None
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    sha = _git(repo, "rev-parse", "HEAD")
    ctx = tmp_path / "ctx"
    with pytest.raises(RuntimeError) as err:
        Q.assemble_build_context(repo, sha, ctx, git_env=Q.host_env(os.environ),
                                 secret_values=[GEM_A, GEM_B])
    msg = str(err.value)
    assert msg.startswith("infra_required: .env at GemmaClaw commit")
    assert GEM_B not in msg and GEM_A not in msg
    assert not (ctx / "gemmaclaw_patches.json").exists()           # stopped before the edits
    assert (repo / ".env").read_text() == f"GEMINI_API_KEY={GEM_B}\n"   # the checkout untouched


# --------------------------------------------------------------------------- #
# a whole run on stubs
# --------------------------------------------------------------------------- #
_REMAP = """# Thinking slash model remap

```yaml qa-scenario
id: thinking-slash-model-remap
title: Thinking slash model remap
execution:
  kind: flow
  config:
    requiredProviderMode: live-frontier
    anthropicModelRef: anthropic/claude-sonnet-4-6
    openAiXhighModelRef: openai/gpt-5.4
    docsRef: repo/qa/scenarios/index.md
```
"""
_GPT54 = """# GPT-5.4 thinking visibility switch

```yaml qa-scenario
id: gpt54-thinking-visibility-switch
title: GPT-5.4 thinking visibility switch
execution:
  kind: flow
  config:
    requiredLiveProvider: openai
    requiredLiveModel: gpt-5.4
```
"""
_PLAIN = """# DM baseline conversation

```yaml qa-scenario
id: dm-chat-baseline
title: DM baseline conversation
execution:
  kind: flow
  config:
    prompt: "say hi; see docs/help/testing.md and site:openai.com/news"
```
"""


def _fake_checkout(repo: Path) -> None:
    for pid in ("openai", "anthropic", "google", "vllm"):
        d = repo / "extensions" / pid
        d.mkdir(parents=True)
        (d / "openclaw.plugin.json").write_text(json.dumps({"id": pid, "providers": [pid]}))
    scen = repo / "qa" / "scenarios"
    (scen / "models").mkdir(parents=True)
    (scen / "channels").mkdir()
    (scen / "models" / "thinking-slash-model-remap.md").write_text(_REMAP)
    (scen / "models" / "gpt54-thinking-visibility-switch.md").write_text(_GPT54)
    (scen / "channels" / "dm-chat-baseline.md").write_text(_PLAIN)
    (scen / "index.md").write_text("# index\n")


def test_non_gemini_scenarios_name_the_provider_they_need(tmp_path):
    _fake_checkout(tmp_path)
    found = Q.non_gemini_scenarios(tmp_path)
    assert set(found) == {"thinking-slash-model-remap", "gpt54-thinking-visibility-switch"}
    remap = found["thinking-slash-model-remap"]
    assert remap["providers"] == ["anthropic", "openai"]
    assert remap["needs"] == ["model anthropic/claude-sonnet-4-6", "model openai/gpt-5.4"]
    assert found["gpt54-thinking-visibility-switch"] == {
        "title": "GPT-5.4 thinking visibility switch", "providers": ["openai"],
        "needs": ["primary model openai/gpt-5.4"]}


def _meta_from(repo: Path):
    """_image_metadata stubbed: the fake checkout's scenarios and manifests."""
    def meta(image, dest):
        shutil.copytree(repo / "extensions", dest / "extensions", dirs_exist_ok=True)
        shutil.copytree(repo / "qa", dest / "qa", dirs_exist_ok=True)
        return {"patches": [], "commit": "c0ffee"}
    return meta


def _volume_source(argv, dest):
    for i, a in enumerate(argv):
        if a == "--volume" and argv[i + 1].split(":")[1:2] == [dest]:
            return argv[i + 1].split(":")[0]
    raise AssertionError(dest)


class _FakeDocker:
    """A stubbed `docker run`: takes the relay token from its standard input and, on its first
    poll, runs `act(token, relay socket, container env, config)` as the container would."""

    def __init__(self, act):
        self.act = act
        self.seen = {}

    def __call__(self, argv, stdin=None, stdout=None, stderr=None, env=None, **kw):
        outer = self
        self.seen["argv"], self.seen["docker_env"] = list(argv), dict(env or {})

        class Proc:
            def __init__(self):
                self.returncode = None
                self._buf = bytearray()
                self.stdin = types.SimpleNamespace(write=self._buf.extend, close=lambda: None)

            def poll(self):
                if self.returncode is None:
                    token = self._buf.decode().strip()
                    cenv = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv)
                                if a == "--env")
                    cfg_text = Path(_volume_source(argv, Q.CONTAINER_CONFIG_PATH)).read_text()
                    sock = os.path.join(_volume_source(argv, Q.CONTAINER_RELAY_DIR),
                                        Q.RELAY_SOCKET_NAME)
                    outer.seen.update(token=token, env=cenv, config=cfg_text)
                    outer.act(types.SimpleNamespace(token=token, socket_path=sock), cenv,
                              json.loads(cfg_text))
                    self.returncode = 0
                return self.returncode

            def wait(self, timeout=None):
                return self.poll()

            def kill(self):
                pass
        return Proc()


def test_a_run_gives_the_suite_the_relay_and_records_every_gemini_call(tmp_path, monkeypatch,
                                                                        upstream):
    monkeypatch.setattr(Q, "GEMINI_API_ORIGIN", upstream.url)
    model_up = _Upstream()
    for name, value in (("GEMINI_API_KEYS", f"{GEM_A},{GEM_B}"), ("OPENAI_API_KEY", OPENAI),
                        ("VLLM_API_KEY", ENDPOINT_KEY),
                        ("GBENCH_CACHE_DIR", str(tmp_path / "cache"))):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # the endpoint's /v1/models (the served id check), answered without a request to the stub
    monkeypatch.setattr("gbench.utils.endpoint.list_served_models",
                        lambda base_url, timeout=5.0, headers=None:
                        [{"id": "google/gemma-4-E2B-it"}])
    repo = tmp_path / "gemmaclaw"
    _fake_checkout(repo)
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, repo,
                                           remote_endpoint=model_up.url + "/v1"))
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._image = "gbench-quality:test"
    records = [{"file": Q.GEMINI_IMAGE_PATCH_FILE, "change": "c", "state": "applied",
                "sha256": "0" * 64},
               {"file": Q.GEMINI_RELAY_PATCH_FILE, "change": "r", "state": "applied",
                "sha256": "1" * 64}]
    meta = _meta_from(repo)

    def image_meta(image, dest):
        out = meta(image, dest)
        return {**out, "patches": records}

    runner._image_metadata = image_meta
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="c0ffee")
    runner._cleanup_server = mock.MagicMock()
    runner._remove_container = mock.MagicMock()
    upstream.script = [(200, b'data: {"candidates": [{"content": {"parts": [{"text": "hi"}]}, '
                             b'"finishReason": "STOP"}]}\r\n\r\n', True),
                       (503, b'{"error": {"code": 503}}', False)]
    seen = {}

    def act(relay, env, cfg):
        google, vllm = cfg["models"]["providers"]["google"], cfg["models"]["providers"]["vllm"]
        for secret in SECRETS:
            assert all(secret not in v for v in env.values()) and secret not in json.dumps(cfg)
        assert env["VLLM_API_KEY"] == Q.KEY_PLACEHOLDER
        assert google["baseUrl"] == env[Q.GEMINI_RELAY_ENV] == Q.CONTAINER_GEMINI_BASE_URL
        assert google["apiKey"] == vllm["apiKey"] == Q.KEY_PLACEHOLDER
        assert google["models"][0]["id"] == "gemini-3.1-pro-preview"
        assert google["request"] == {"allowPrivateNetwork": True}
        assert vllm["baseUrl"] == Q.CONTAINER_MODEL_BASE_URL
        assert vllm["models"][0]["input"] == ["text"]            # a text-only model under test
        # the suite's Gemini and endpoint calls, as the forwarder hands them to the relay
        hdr = {"x-goog-api-key": google["apiKey"], "Content-Type": "application/json"}
        seen["alt"] = _call(relay, "/v1beta/models/gemini-3.1-pro-preview:"
                                   "streamGenerateContent?alt=sse", hdr)
        seen["image"] = _call(relay, "/v1beta/models/gemini-3.1-flash-image-preview:"
                                     "generateContent", hdr)
        seen["chat"] = _call(relay, "/model/chat/completions",
                             {"Authorization": f"Bearer {vllm['apiKey']}"})

    docker = _FakeDocker(act)
    monkeypatch.setattr(Q.subprocess, "Popen", docker)
    runner._collect_container_outputs = lambda name: {
        "summary": {"counts": {"total": 4, "passed": 2, "failed": 2},
                    "scenarios": [{"name": "DM baseline conversation", "status": "pass"},
                                  {"name": "Some scenario", "status": "fail"},
                                  {"name": "Thinking slash model remap", "status": "pass"},
                                  {"name": "GPT-5.4 thinking visibility switch",
                                   "status": "fail"}],
                    "run": {"startedAt": "2020-01-01T00:00:00.000Z",
                            "finishedAt": "2099-01-01T00:00:00.000Z"}},
        "scenario_summaries": {}, "entrypoint": {"command_exit": 0,
                                                 "forwarder_alive_at_end": True},
        "forwarder_errors": []}
    model = ModelConfig(short_name="gemma-4-E2B-it", hf_model_id="google/gemma-4-E2B-it",
                        total_params_b=5.2, is_moe=False)
    try:
        with mock.patch("gbench.runners.evals._server_max_model_len", return_value=32768):
            res = runner.run(model, ModelFormat.HF)
    finally:
        model_up.close()
    assert seen["alt"][0] == 200 and seen["image"][0] == 503 and seen["chat"][0] == 200
    # upstream got the real keys, never the token
    assert [r["headers"]["x-goog-api-key"] in (GEM_A, GEM_B) for r in upstream.requests] == \
        [True, True]
    assert model_up.requests[0]["headers"]["authorization"] == f"Bearer {ENDPOINT_KEY}"
    token = docker.seen["token"]
    for req in upstream.requests + model_up.requests:
        assert token not in json.dumps(req["headers"])
    assert token not in json.dumps(docker.seen["argv"]) + docker.seen["config"]
    # every Gemini call recorded; a failed one is a harness error. The summary has four scenarios
    # and no per-scenario windows, so the error cannot be tied to one: the pass rate is withheld.
    calls = {c["capability"]: c for c in res["gemini_calls"]}
    assert calls["alternate model"]["statuses"] == {"200": 1}
    assert calls["image generation"]["statuses"] == {"503": 1}
    assert res["harness_errors"] == {"gemini_http_503": 1}
    assert res["status"] == "completed_with_errors"
    assert "gemini_http_503 x1" in res["leaderboard_comparable_reason"]
    assert res["unattributed_harness_errors"] == {"gemini_http_503": 1}
    assert res["pass_rate"] is None and res["pass_rate_withheld_reason"]
    assert res["model_endpoint_relayed"] is True and res["model_endpoint_key_at_relay"] is True
    assert res["model_endpoint_calls"][0]["statuses"] == {"200": 1}
    # scenarios that need another provider: apart, not scored
    infra = {r["id"]: r for r in res["infra_required_scenarios"]}
    assert set(infra) == {"thinking-slash-model-remap", "gpt54-thinking-visibility-switch"}
    assert infra["thinking-slash-model-remap"]["providers"] == ["anthropic", "openai"]
    assert infra["thinking-slash-model-remap"]["suite_status"] == "pass"
    assert (res["total_scenarios"], res["passed_scenarios"], res["failed_scenarios"]) == (2, 1, 1)
    assert res["suite_counts"] == {"total": 4, "passed": 2, "failed": 2}
    assert "thinking-slash-model-remap (anthropic, openai)" in res["leaderboard_comparable_reason"]
    # substitutions: a text-only model under test gets Gemini image understanding
    subs = {s["role"]: s for s in res["gemini_substitutions"]}
    assert res["image_input_declared"] is False
    assert subs["memory embeddings"]["gemini_model"] == Q.GEMINI_EMBEDDING_MODEL
    assert subs["memory embeddings"]["upstream"] == "openai/text-embedding-3-small"
    assert subs["image understanding (image tool)"]["gemini_model"] == \
        Q.GEMINI_IMAGE_UNDERSTANDING_MODEL
    for m in (Q.UPSTREAM_EMBEDDING_MODEL, Q.GEMINI_EMBEDDING_MODEL,
              Q.UPSTREAM_IMAGE_UNDERSTANDING_MODEL, Q.GEMINI_IMAGE_UNDERSTANDING_MODEL):
        assert m in res["leaderboard_comparable_reason"]
    assert res["gemmaclaw_patches"] == records
    assert res["sandbox"]["tool"] == "docker" and res["sandbox"]["image"] == "gbench-quality:test"
    saved = Path(runner.config.log_manager.get_quality_result_path(
        model.short_name, ModelFormat.HF.value)).read_text()
    for secret in (*SECRETS, token):
        assert secret not in saved and secret not in json.dumps(res)


def test_a_multimodal_model_under_test_is_declared_with_image_input(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    repo = tmp_path / "gemmaclaw"
    repo.mkdir()
    (repo / "extensions").mkdir()
    (repo / "qa").mkdir()
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, repo,
                                           remote_endpoint="http://127.0.0.1:9/v1"))
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="c0ffee")
    runner._cleanup_server = mock.MagicMock()
    runner._remove_container = mock.MagicMock()
    runner._image = "gbench-quality:test"
    runner._image_metadata = _meta_from(repo)
    seen = {}
    docker = _FakeDocker(lambda relay, env, cfg: seen.update(cfg=cfg, env=env))
    monkeypatch.setattr(Q.subprocess, "Popen", docker)
    runner._collect_container_outputs = lambda name: {
        "summary": {"counts": {"total": 1, "passed": 1, "failed": 0}},
        "scenario_summaries": {}, "entrypoint": {"command_exit": 0,
                                                 "forwarder_alive_at_end": True},
        "forwarder_errors": []}
    model = ModelConfig(short_name="gemma-4-E4B-it", hf_model_id="google/gemma-4-E4B-it",
                        total_params_b=8.0, is_moe=False)
    model.supports_multimodal = True
    with mock.patch("gbench.runners.evals._server_max_model_len", return_value=32768):
        res = runner.run(model, ModelFormat.HF)
    vllm = seen["cfg"]["models"]["providers"]["vllm"]
    assert vllm["models"][0]["input"] == ["text", "image"]
    # a key-less endpoint is behind the relay too, so its failures are counted
    assert vllm["baseUrl"] == Q.CONTAINER_MODEL_BASE_URL and vllm["apiKey"] == Q.KEY_PLACEHOLDER
    assert seen["env"]["VLLM_API_KEY"] == Q.KEY_PLACEHOLDER
    assert res["image_input_declared"] is True and res["model_endpoint_relayed"] is True
    assert res["model_endpoint_key_at_relay"] is False
    assert "image understanding (image tool)" not in {
        s["role"] for s in res["gemini_substitutions"]}
    assert res["status"] == "completed" and res["harness_errors"] == {}
    assert res["gemini_calls"] == [] and res["infra_required_scenarios"] == []
    assert res["pass_rate"] == 100.0


# --------------------------------------------------------------------------- #
# gbench's edits of the checkout
# --------------------------------------------------------------------------- #
#: extensions/google/api.ts lines 45-88 at the pinned GemmaClaw commit, verbatim.
_UPSTREAM_API_TS = '''import { parseGeminiAuth } from "./gemini-auth.js";

function resolveTrustedGoogleGenerativeAiBaseUrl(baseUrl?: string): string {
  const normalized =
    normalizeGoogleGenerativeAiBaseUrl(baseUrl ?? DEFAULT_GOOGLE_API_BASE_URL) ??
    DEFAULT_GOOGLE_API_BASE_URL;
  let url: URL;
  try {
    url = new URL(normalized);
  } catch {
    throw new Error(
      "Google Generative AI baseUrl must be a valid https URL on generativelanguage.googleapis.com",
    );
  }
  if (
    url.protocol !== "https:" ||
    url.hostname.toLowerCase() !== "generativelanguage.googleapis.com"
  ) {
    throw new Error(
      "Google Generative AI baseUrl must use https://generativelanguage.googleapis.com",
    );
  }
  return normalized;
}

export function resolveGoogleGenerativeAiHttpRequestConfig(params: {
  apiKey: string;
  baseUrl?: string;
  headers?: Record<string, string>;
  request?: ProviderRequestTransportOverrides;
  capability: "image" | "audio" | "video";
  transport: "http" | "media-understanding";
}) {
  return resolveProviderHttpRequestConfig({
    baseUrl: resolveTrustedGoogleGenerativeAiBaseUrl(params.baseUrl),
    defaultBaseUrl: DEFAULT_GOOGLE_API_BASE_URL,
    allowPrivateNetwork: false,
    headers: params.headers,
    request: params.request,
    defaultHeaders: parseGeminiAuth(params.apiKey).headers,
    provider: "google",
    api: "google-generative-ai",
    capability: params.capability,
    transport: params.transport,
  });
}
'''


def _upstream_files(repo: Path) -> Path:
    api = repo / Q.GEMINI_RELAY_PATCH_FILE
    api.parent.mkdir(parents=True, exist_ok=True)
    api.write_text(_UPSTREAM_API_TS, encoding="utf-8")
    index = repo / Q.GEMINI_IMAGE_PATCH_FILE
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("export const x = {\n" + Q._IMAGE_PATCH_OLD + "};\n", encoding="utf-8")
    return api


def test_the_relay_edit_applies_once_reverts_and_refuses_drift(tmp_path):
    api = _upstream_files(tmp_path)
    recs = Q.apply_gemmaclaw_patches(tmp_path)
    assert [(r["file"], r["state"]) for r in recs] == [(Q.GEMINI_IMAGE_PATCH_FILE, "applied"),
                                                       (Q.GEMINI_RELAY_PATCH_FILE, "applied")]
    patched = api.read_text()
    assert patched.count("function isGbenchGeminiRelayBaseUrl") == 1
    assert "process.env.GBENCH_GEMINI_RELAY_BASE_URL?.trim()" in patched
    assert "allowPrivateNetwork: isGbenchGeminiRelayBaseUrl(params.baseUrl)," in patched
    assert "allowPrivateNetwork: false," not in patched
    # the https-only check stays for every other base URL
    assert 'url.hostname.toLowerCase() !== "generativelanguage.googleapis.com"' in patched
    import hashlib
    assert recs[1]["sha256"] == hashlib.sha256(patched.encode()).hexdigest()
    assert [r["state"] for r in Q.apply_gemmaclaw_patches(tmp_path)] == ["already applied"] * 2
    assert api.read_text() == patched
    Q.revert_gemmaclaw_patches(tmp_path)
    assert api.read_text() == _UPSTREAM_API_TS
    assert Q._IMAGE_PATCH_OLD in (tmp_path / Q.GEMINI_IMAGE_PATCH_FILE).read_text()
    api.write_text(_UPSTREAM_API_TS.replace("allowPrivateNetwork: false,",
                                            "allowPrivateNetwork: true,"))
    with pytest.raises(RuntimeError, match="infra_required: extensions/google/api.ts does not"):
        Q.apply_gemmaclaw_patches(tmp_path)
    assert api.read_text().count("isGbenchGeminiRelayBaseUrl") == 0   # all or none




def test_setup_makes_both_edits_in_the_build_context_never_in_the_checkout(tmp_path,
                                                                             monkeypatch):
    repo = tmp_path / "gemmaclaw"
    api = _upstream_files(repo)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "upstream")
    sha = _git(repo, "rev-parse", "HEAD")
    before = {p: p.read_bytes() for p in repo.rglob("*")
              if p.is_file() and ".git" not in p.parts}
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    real_run = Q.subprocess.run
    seen = []

    def fake_run(cmd, *a, env=None, **kw):
        seen.append((list(cmd), dict(env or {})))
        if cmd[0] != "docker":
            return real_run(cmd, *a, env=env, **kw)
        if cmd[1:3] == ["image", "inspect"]:
            built = any(c[1] == "build" for c, _e in seen if c[0] == "docker")
            return subprocess.CompletedProcess(cmd, 0 if built else 1,
                                               sha + "\n" if built else "", "")
        ctx = Path(cmd[-1])
        seen.append((["context"], {
            "api": (ctx / "src" / Q.GEMINI_RELAY_PATCH_FILE).read_text(),
            "index": (ctx / "src" / Q.GEMINI_IMAGE_PATCH_FILE).read_text(),
            "records": (ctx / "gemmaclaw_patches.json").read_text()}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    runner = Q.QualityBenchmarkRunner(_cfg(tmp_path, repo, gemmaclaw_commit=sha))
    assert runner._setup_gemmaclaw_repo() == sha, runner._setup_failure
    [ctx] = [e for c, e in seen if c == ["context"]]
    assert Q._RELAY_PATCH_NEW_1 in ctx["api"] and Q._RELAY_PATCH_NEW_2 in ctx["api"]
    assert Q._IMAGE_PATCH_NEW in ctx["index"]
    assert [(r["file"], r["state"]) for r in json.loads(ctx["records"])] == [
        (Q.GEMINI_IMAGE_PATCH_FILE, "applied"), (Q.GEMINI_RELAY_PATCH_FILE, "applied")]
    # the checkout is only read: no checkout, no edit, no install or build in it
    assert {p: p.read_bytes() for p in repo.rglob("*")
            if p.is_file() and ".git" not in p.parts} == before
    assert api.read_text() == _UPSTREAM_API_TS
    git_cmds = [c for c, _e in seen if c[0] == "git"]
    assert not any("checkout" in c for c in git_cmds) and not any(c[0] == "npx" for c, _ in seen)
    for c in git_cmds:
        assert c[1:5] == ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"]
    for _c, env in seen:
        assert GEM_A not in json.dumps(env)
    assert runner._image == Q.quality_image_tag(sha)


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_the_quality_doc_describes_the_sandbox_the_relay_and_the_new_fields():
    doc = (ROOT / "docs" / "quality.md").read_text(encoding="utf-8")
    for needle in ("Docker", "docker/quality.Dockerfile", Q.GEMINI_RELAY_ENV,
                   Q.GEMINI_RELAY_PATCH_FILE,
                   "https://generativelanguage.googleapis.com", "`gemini_calls`",
                   "`harness_errors`", "`completed_with_errors`", "`infra_required_scenarios`",
                   "`suite_counts`", "`image_input_declared`", "`model_endpoint_relayed`",
                   "`model_endpoint_calls`", "`sandbox`", "`status`",
                   Q.UPSTREAM_EMBEDDING_MODEL, Q.GEMINI_EMBEDDING_MODEL,
                   Q.UPSTREAM_IMAGE_UNDERSTANDING_MODEL, Q.GEMINI_IMAGE_UNDERSTANDING_MODEL,
                   Q.CONTAINER_NPM_CACHE, "web search", "/proc", Q.KEY_PLACEHOLDER):
        assert needle in doc, needle
    for line in (Q._RELAY_PATCH_NEW_1 + Q._RELAY_PATCH_NEW_2).splitlines():
        assert line in doc, line
    assert "Besides the model under test, the live-frontier lane calls two models" not in doc
