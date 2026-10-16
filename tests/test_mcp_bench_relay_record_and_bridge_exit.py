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

"""mcp_bench: the credentialed relay's record keeps the port it listened on after it stopped; a
connection that fails while the credentialed HTTP server exits on its own is not a failure; tasks
left out on the bridge's record alone, with nothing on gbench's relay, are counted; the HTTP
server's environment deviation is documented. The relay is a local stub; nothing is written
outside tmp."""

import json
import os
import socket
import socketserver
import sys
import threading
import time

import pytest

from gbench.runners.eval_suites import mcp_bench as MB

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "gbench", "docker"))
import mcp_bench_bridge as BRIDGE  # noqa: E402


def test_the_relay_record_keeps_its_port_after_stop():
    relay = MB.CredentialedServersRelay("cred", "127.0.0.1", servers={}).start()
    port = int(relay.address.rsplit(":", 1)[1])
    assert port > 0
    relay.stop()
    assert relay.record()["listens"] == f"127.0.0.1:{port}"


@pytest.mark.parametrize("by_kind, run, expected", [
    ({"credentialed_relay": 2}, {"sessions_failed": 0, "container_running_at_end": True}, 2),
    ({"credentialed_relay": 2}, {"sessions_failed": 1, "container_running_at_end": True}, 0),
    ({"credentialed_relay": 2}, {"sessions_failed": 0, "container_running_at_end": False}, 0),
    ({"model_endpoint": 1}, {"sessions_failed": 0, "container_running_at_end": True}, 0),
])
def test_bridge_records_nothing_on_the_relay_corroborates_are_counted(by_kind, run, expected):
    assert MB.unconfirmed_bridge_failures({"by_kind": by_kind}, run) == expected


class _FakeRelay:
    """A relay stub: http-start answers OK and the server's ready line, then (when `exit_server`)
    the end frame of a server that exited on its own once a connection was asked for;
    http-connect is refused, after that end frame went out."""

    def __init__(self, exit_server):
        self.exit_server = exit_server
        self.connect_seen = threading.Event()
        relay = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                header = json.loads(self.rfile.readline())
                if header["op"] == "http-start":
                    self.wfile.write(b'OK\n{"ready": true, "port": 4321}\n')
                    self.wfile.flush()
                    relay.connect_seen.wait(5)
                    if relay.exit_server:
                        self.wfile.write(b'E{"exit": 0}\n')
                        self.wfile.flush()
                    else:
                        time.sleep(3)
                elif header["op"] == "http-connect":
                    relay.connect_seen.set()
                    time.sleep(0.3)               # the server's exit reaches the bridge first
                    self.wfile.write(b"ERR not started: connection refused\n")
                    self.wfile.flush()

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bridge_with(monkeypatch, tmp_path, exit_server):
    relay = _FakeRelay(exit_server)
    cfg = tmp_path / "bridge.json"
    cfg.write_text(json.dumps({"relay": f"127.0.0.1:{relay.server.server_address[1]}",
                               "token": "t"}))
    failures = tmp_path / "failures.jsonl"
    monkeypatch.setattr(BRIDGE, "BRIDGE_CONFIG", str(cfg))
    monkeypatch.setattr(BRIDGE, "BRIDGE_FAILURES", str(failures))
    monkeypatch.setattr(BRIDGE, "EXIT_SETTLE_S", 1.0)
    port = _free_port()
    out = {}
    worker = threading.Thread(target=lambda: out.update(rc=BRIDGE.run_http("google-maps", port)),
                              daemon=True)
    worker.start()
    for _ in range(100):
        try:
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            break
        except OSError:
            time.sleep(0.05)
    client.sendall(b"GET / HTTP/1.1\r\n\r\n")
    return relay, client, worker, out, failures


def test_a_connection_failing_while_the_server_exits_is_not_a_failure(monkeypatch, tmp_path):
    relay, client, worker, out, failures = _bridge_with(monkeypatch, tmp_path, exit_server=True)
    try:
        worker.join(10)
        time.sleep(1.5)                           # the connection's session has settled
    finally:
        client.close()
        relay.close()
    assert out.get("rc") == 0
    assert not failures.exists() or failures.read_text() == ""


def test_a_connection_failing_while_the_server_runs_is_recorded(monkeypatch, tmp_path):
    relay, client, worker, out, failures = _bridge_with(monkeypatch, tmp_path, exit_server=False)
    try:
        deadline = time.time() + 6
        while time.time() < deadline and not (failures.exists() and failures.read_text()):
            time.sleep(0.1)
    finally:
        client.close()
        relay.close()
    [rec] = [json.loads(x) for x in failures.read_text().splitlines()]
    assert rec["server"] == "google-maps" and rec["op"] == "http-connect"


def test_the_http_server_environment_deviation_is_documented():
    doc = " ".join(open(os.path.join(_REPO, "docs", "evals", "mcp_bench.md"),
                        encoding="utf-8").read().split())
    assert "starts it with its whole environment (`os.environ.copy()`" in doc
    assert "`unconfirmed_bridge_failures`" in doc
    head = open(os.path.join(_REPO, "gbench", "docker", "mcp_bench_credentialed.py"),
                encoding="utf-8").read()
    assert "One deviation: upstream starts it with its whole" in head
