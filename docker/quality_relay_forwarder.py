#!/usr/bin/env python3
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
"""The relay forwarder of gbench's quality container (docker/quality.Dockerfile).

The container has no network. This process, started by the entrypoint as user `gbrelay`, listens
on 127.0.0.1:FORWARD_PORT inside it and forwards every HTTP request to gbench's key relay over the
unix socket RELAY_SOCKET (the one host path the container reaches), adding the per-run relay token
in RELAY_TOKEN_HEADER. It reads the token from the first line of its standard input and keeps it
in memory only: the QA suite and every tool the model under test runs are user `qa`, hold no
capability, and cannot read this process's memory, environment or file descriptors.

Every request the forwarder cannot hand to the relay, and every relay response that breaks off,
is appended to STATUS_DIR/forwarder-errors.jsonl ({"t": epoch seconds, "kind", "route"}), a
directory user `qa` cannot write; gbench counts those as harness errors. STATUS_DIR/forwarder-ready
is created once the port listens. Standard library only.
"""

import http.client
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Kept in step with gbench/runners/quality.py (FORWARDER_PORT, CONTAINER_RELAY_SOCKET,
#: CONTAINER_STATUS_DIR, RELAY_TOKEN_HEADER, MODEL_PREFIX).
FORWARD_PORT = 18431
RELAY_SOCKET = "/run/gbench-relay/relay.sock"
STATUS_DIR = "/gbench-status"
RELAY_TOKEN_HEADER = "x-gbench-relay-token"
MODEL_PREFIX = "/model"
#: Longer than the relay's own upstream timeout (3600 s for the model endpoint), so the relay, not
#: the forwarder, decides when an upstream took too long.
RELAY_TIMEOUT_S = 3700.0

_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                         "proxy-connection", "te", "trailer", "trailers", "transfer-encoding",
                         "upgrade", "content-length"})

_TOKEN = ""
_LOG_LOCK = threading.Lock()


def _record(kind: str, route: str) -> None:
    line = json.dumps({"t": time.time(), "kind": kind, "route": route}) + "\n"
    with _LOG_LOCK:
        fd = os.open(os.path.join(STATUS_DIR, "forwarder-errors.jsonl"),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)


class _RelayConnection(http.client.HTTPConnection):
    """HTTP over the relay's unix socket."""

    def __init__(self, timeout: float):
        super().__init__("gbench-relay", timeout=timeout)

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(RELAY_SOCKET)
        self.sock = sock


class _BrokeOff(Exception):
    """The relay's response broke off while it was being streamed back."""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gbench-forwarder"

    def log_message(self, format, *args):     # noqa: A002 - no request lines on stderr
        return

    def _read_body(self) -> bytes:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";", 1)[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _send_error_json(self, status: int, message: str) -> None:
        data = json.dumps({"error": {"code": status, "message": message}}).encode("utf-8")
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        self.wfile.flush()

    def _any(self):
        route = ("model" if self.path == MODEL_PREFIX or self.path.startswith(MODEL_PREFIX + "/")
                 or self.path.startswith(MODEL_PREFIX + "?") else "gemini")
        try:
            body = self._read_body()
        except (OSError, ValueError):
            self.close_connection = True
            return
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in _HOP_BY_HOP and k.lower() != RELAY_TOKEN_HEADER}
        headers[RELAY_TOKEN_HEADER] = _TOKEN
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        conn = _RelayConnection(RELAY_TIMEOUT_S)
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException):
            conn.close()
            _record("relay_unreachable", route)
            self._send_error_json(502, "gbench forwarder: the gbench relay is unreachable")
            return
        try:
            self._stream_back(resp)
        except _BrokeOff:
            _record("relay_stream_broken", route)
            self.close_connection = True
        except OSError:                          # the client went away
            self.close_connection = True
        finally:
            conn.close()

    def _stream_back(self, resp: http.client.HTTPResponse) -> None:
        def _pull(read) -> bytes:
            try:
                return read(65536)
            except (OSError, http.client.HTTPException) as e:
                raise _BrokeOff(type(e).__name__) from e

        self.send_response_only(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in _HOP_BY_HOP and k.lower() not in ("date", "server"):
                self.send_header(k, v)
        length = resp.getheader("Content-Length")
        if self.command == "HEAD" or resp.status in (204, 304) or resp.status < 200:
            if length is not None:
                self.send_header("Content-Length", length)
            self.end_headers()
            self.wfile.flush()
            return
        if length is not None:
            self.send_header("Content-Length", length)
            self.end_headers()
            remaining = int(length)
            while remaining > 0:
                chunk = _pull(resp.read)
                if not chunk:
                    raise _BrokeOff("short body")
                remaining -= len(chunk)
                self.wfile.write(chunk)
            self.wfile.flush()
            return
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            chunk = _pull(resp.read1)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _any


def main() -> int:
    global _TOKEN
    _TOKEN = sys.stdin.readline().strip()
    sys.stdin.close()
    if not _TOKEN:
        sys.stderr.write("gbench forwarder: no relay token on standard input\n")
        return 2
    server = ThreadingHTTPServer(("127.0.0.1", FORWARD_PORT), _Handler)
    server.daemon_threads = True
    with open(os.path.join(STATUS_DIR, "forwarder-ready"), "w", encoding="utf-8"):
        pass
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
