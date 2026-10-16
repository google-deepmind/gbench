#!/usr/bin/env python3
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
#
# The bridge to a credentialed MCP server, INSIDE the gbench mcp_bench model container. The five
# servers that need one of the keys MCP-Bench's README requires (Google Maps, BioMCP, Hugging Face,
# National Parks, NASA Data) do not run in this container, where model-driven servers run model
# text (the Scientific Computing server's sympy parse_expr, the OSINT server's commands) and no key
# is present. They run in gbench's credentialed-servers container (mcp_bench_credentialed.py). The
# launcher (mcp_bench_run.install_credentialed_bridges) points their commands.json entries at this
# script, so upstream's runner and connection check start it exactly where they would start the
# server, and it carries the session to the real server:
#
#   mcp_bench_bridge.py stdio <slug>              a stdio server: this process's stdin and stdout
#                                                 are passed to and from the server, byte for byte
#   mcp_bench_bridge.py http <slug> --port <N>    an HTTP server (Google Maps): listens on
#                                                 localhost:<N> right away (upstream checks it 3 s
#                                                 after the start) and carries every connection to
#                                                 the server once it is ready; the server runs as
#                                                 long as this process does, and this process
#                                                 ends when the server does
#
# Each session is a TCP connection to gbench's relay (mcp_bench.CredentialedServersRelay, in the
# gbench process, on this container network's gateway): one JSON line {"token", "op", "server",
# "port"}, then "OK" or "ERR <reason>" from the relay. The relay answers OK only once the
# credentialed side said the session is up (the stdio server's process started, the HTTP server
# accepts connections, the connection to it is open; for http-start the OK is followed by the
# server's {"ready": true, "port"} line). After OK this process's bytes go to the session as they
# are, and what comes back arrives in frames: "D<hex length>\n" and the bytes, then, when the
# credentialed side ends, "E{"exit": <code>}\n" (0 when the session ended normally there; another
# code is a failure of docker exec or of the credentialed container). The relay's address and the
# per-run token are in BRIDGE_CONFIG, which the launcher writes (a stdio server gets only the MCP
# SDK's default variables, so they cannot come from the environment).
#
# A session that failed for a reason outside the model is written to BRIDGE_FAILURES with its
# time: the relay did not answer or refused it (its credentialed side did not come up, e.g. the
# credentialed container is gone or the server's command could not start); the stdio server
# exited before it sent a byte while the client still used the session (a failed start, e.g.
# `uv run` could not fetch the server's requirements); the HTTP server did not report ready; or
# the session ended without the end frame, or with a nonzero exit, while the client still used it
# (the session was lost). The launcher's harness accounting makes the task running then a harness
# error (credentialed_relay), never the model's outcome. A server that exits on its own after it
# answered is its own outcome, as it would be upstream (for an HTTP server this process then
# closes its listeners and ends, as the server's port would close, and a connection that failed
# while the server was exiting, EXIT_SETTLE_S, is not recorded); a session the client ended is
# no failure.
# Standard library only.

import json
import os
import socket
import sys
import threading
import time

#: Written by the launcher: {"relay": "<host>:<port>", "token": "<per-run token>"}.
BRIDGE_CONFIG = "/tmp/gbench-mcp-bench-credentialed.json"
#: One JSON line per session that failed outside the model: {"t", "server", "op", "why"}.
BRIDGE_FAILURES = "/tmp/gbench-mcp-bench-bridge-failures.jsonl"
#: Seconds to connect to the relay.
RELAY_TIMEOUT_S = 30
#: Seconds to wait for the relay's answer, per session kind: longer than the relay's own wait for
#: the session to come up (mcp_bench.CredentialedServersRelay.READY_TIMEOUT_S), which for an HTTP
#: server includes its wait for the server to accept connections.
ANSWER_TIMEOUT_S = {"stdio": 60, "http-start": 180, "http-connect": 60}
#: How long an HTTP connection accepted before the server is ready waits for it.
HTTP_READY_WAIT_S = 120
#: The longest frame header line.
FRAME_HEADER_LIMIT = 4096
#: How long a connection whose session failed waits to learn whether the server exited on its
#: own meanwhile (the control session's end), before the failure is recorded.
EXIT_SETTLE_S = 2.0


def record_failure(server, op, why):
    try:
        with open(BRIDGE_FAILURES, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), "server": server, "op": op,
                                 "why": str(why)[:200]}) + "\n")
    except OSError:
        pass
    sys.stderr.write(f"mcp_bench_bridge: {server} {op}: {why}\n")


def _config():
    with open(BRIDGE_CONFIG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    host, _, port = str(cfg["relay"]).rpartition(":")
    return host.strip("[]"), int(port), str(cfg["token"])


def _readline(sock, limit=4096):
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        buf += chunk
        if len(buf) > limit:
            break
    return buf


class SessionFailed(Exception):
    pass


def open_session(server, op, port=None):
    """A socket carrying one session to the credentialed server `server` (`op`: stdio,
    http-start or http-connect), after the relay's OK. Raises SessionFailed otherwise."""
    try:
        host, relay_port, token = _config()
    except (OSError, ValueError, KeyError) as e:
        raise SessionFailed(f"no relay configuration ({type(e).__name__})")
    try:
        sock = socket.create_connection((host, relay_port), timeout=RELAY_TIMEOUT_S)
        sock.settimeout(ANSWER_TIMEOUT_S.get(op, RELAY_TIMEOUT_S))
        header = {"token": token, "op": op, "server": server}
        if port is not None:
            header["port"] = int(port)
        sock.sendall(json.dumps(header).encode("utf-8") + b"\n")
        answer = _readline(sock).decode("utf-8", "replace").strip()
    except OSError as e:
        raise SessionFailed(f"the relay did not answer ({type(e).__name__})")
    if answer != "OK":
        sock.close()
        raise SessionFailed(f"the relay refused the session ({answer or 'closed'})")
    sock.settimeout(None)
    return sock


def read_frames(stream, write):
    """Pass the data frames of the relay's `stream` (a binary file) to `write` until the end
    frame. Returns (how, detail, answered): ("end", exit code, ...) at the end frame, ("lost",
    why, ...) when the stream ended or broke without one, ("client", None, ...) when `write`
    failed (the client went away); `answered` is whether any byte of the session arrived."""
    answered = False
    while True:
        try:
            head = stream.readline(FRAME_HEADER_LIMIT)
        except OSError as e:
            return "lost", f"the relay's stream broke ({type(e).__name__})", answered
        if not head.endswith(b"\n"):
            return "lost", "the relay's stream ended without the session's end", answered
        if head.startswith(b"D"):
            try:
                n = int(head[1:].strip(), 16)
            except ValueError:
                return "lost", "an unreadable frame from the relay", answered
            try:
                data = stream.read(n)
            except OSError as e:
                return "lost", f"the relay's stream broke ({type(e).__name__})", answered
            if len(data) < n:
                return "lost", "the relay's stream ended inside a frame", answered
            if data:
                answered = True
                try:
                    write(data)
                except OSError:
                    return "client", None, answered
            continue
        if head.startswith(b"E"):
            try:
                code = int(json.loads(head[1:].decode("utf-8"))["exit"])
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                return "lost", "an unreadable end frame from the relay", answered
            return "end", code, answered
        return "lost", "an unreadable frame from the relay", answered


def session_failure(how, detail, answered, client_open):
    """Why a session that ended so failed outside the model (None when it did not): lost or ended
    by a nonzero exit while the client still used it, or a server that exited before it answered.
    A session the client ended, or a server that exited on its own after it answered, is none."""
    if how == "client" or not client_open:
        return None
    if how == "lost":
        return f"the session was lost ({detail})"
    if detail != 0:
        return f"the session ended on the credentialed side (exit {detail})"
    if not answered:
        return "the server exited before it answered"
    return None


def _forward(read, sock, client_open):
    """Copy `read` (the client's side) to the session; at its end clear `client_open` and close
    the session's input. A failed send (the session's side is gone) leaves `client_open` set:
    the frame reader reports that end."""
    while True:
        try:
            data = read(65536)
        except OSError:
            data = b""
        if not data:
            client_open.clear()
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            return
        try:
            sock.sendall(data)
        except OSError:
            return


def run_stdio(server):
    try:
        sock = open_session(server, "stdio")
    except SessionFailed as e:
        record_failure(server, "stdio", e)
        return 1
    client_open = threading.Event()
    client_open.set()
    threading.Thread(target=_forward, args=(lambda n: os.read(0, n), sock, client_open),
                     daemon=True).start()

    def write_out(data):
        while data:
            data = data[os.write(1, data):]
    with sock.makefile("rb") as stream:
        how, detail, answered = read_frames(stream, write_out)
        still_open = client_open.is_set()
    sock.close()
    why = session_failure(how, detail, answered, still_open)
    if why:
        record_failure(server, "stdio", why)
        return 1
    if how == "end" and still_open:
        sys.stderr.write(f"mcp_bench_bridge: {server} stdio: the server exited\n")
    return 0


def _listeners(port):
    """Listening sockets on localhost:<port> (127.0.0.1, and ::1 when the host has it). Exits with
    upstream's EADDRINUSE text when the port is taken, so upstream retries another port."""
    socks = []
    for family, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue
        try:
            s.bind((addr, port))
            s.listen(64)
            socks.append(s)
        except OSError as e:
            s.close()
            if getattr(e, "errno", None) == 98 and family == socket.AF_INET:
                sys.stderr.write(f"Error: listen EADDRINUSE: address already in use {addr}:{port}\n")
                sys.exit(1)
    if not socks:
        sys.stderr.write(f"mcp_bench_bridge: cannot listen on localhost:{port}\n")
        sys.exit(1)
    return socks


def run_http(server, port):
    listeners = _listeners(port)
    ready = threading.Event()
    # Set when the control session ended normally: the server exited on its own. A connection
    # that fails then (accepted in the moment before this process closes its listeners) is the
    # server's own exit, not a failure outside the model.
    exited = threading.Event()
    state = {"port": None}

    def carry(conn):
        with conn:
            if not ready.wait(HTTP_READY_WAIT_S) or state["port"] is None:
                return
            if exited.is_set():
                return
            try:
                up = open_session(server, "http-connect", state["port"])
            except SessionFailed as e:
                if not exited.wait(EXIT_SETTLE_S):
                    record_failure(server, "http-connect", e)
                return
            client_open = threading.Event()
            client_open.set()
            with up:
                threading.Thread(target=_forward, args=(conn.recv, up, client_open),
                                 daemon=True).start()
                with up.makefile("rb") as stream:
                    how, detail, _answered = read_frames(stream, conn.sendall)
                    still_open = client_open.is_set()
                try:
                    conn.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
            # A server that closed the connection (end frame, exit 0) did so on its own.
            if how == "end" and detail == 0:
                return
            why = session_failure(how, detail, True, still_open)
            if why and not exited.wait(EXIT_SETTLE_S):
                record_failure(server, "http-connect", why)

    def accept(listener):
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=carry, args=(conn,), daemon=True).start()

    for listener in listeners:
        threading.Thread(target=accept, args=(listener,), daemon=True).start()
    try:
        ctrl = open_session(server, "http-start")
    except SessionFailed as e:
        record_failure(server, "http-start", e)
        return 1
    with ctrl, ctrl.makefile("rb") as stream:
        try:
            line = stream.readline(FRAME_HEADER_LIMIT).decode("utf-8", "replace").strip()
        except OSError:
            line = ""
        try:
            status = json.loads(line)
        except ValueError:
            status = {}
        status = status if isinstance(status, dict) else {}
        port_ok = isinstance(status.get("port"), int) and not isinstance(status.get("port"), bool)
        if status.get("ready") is not True or not port_ok:
            record_failure(server, "http-start", "the server did not start ("
                           f"{status.get('error') or line or 'no answer'})")
            return 1
        state["port"] = int(status["port"])
        ready.set()
        # The server runs while this session is open; the session ends when the server exits.
        how, detail, _answered = read_frames(stream, lambda data: None)
    why = session_failure(how, detail, True, True)
    if why:
        record_failure(server, "http-start", why)
        return 1
    # The server exited on its own: connections from here on are its outcome, not failures, and
    # the port closes, as the server's would.
    exited.set()
    for listener in listeners:
        try:
            listener.close()
        except OSError:
            pass
    sys.stderr.write(f"mcp_bench_bridge: {server} http: the server exited\n")
    return 0


def main(argv):
    if len(argv) == 3 and argv[1] == "stdio":
        return run_stdio(argv[2])
    if len(argv) == 5 and argv[1] == "http" and argv[3] == "--port":
        return run_http(argv[2], int(argv[4]))
    sys.stderr.write("usage: mcp_bench_bridge.py stdio <server> | http <server> --port <N>\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
