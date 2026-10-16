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
# The credentialed-servers side of the gbench mcp_bench run, INSIDE the credentialed-servers
# container: a container of the gbench-mcp-bench image that gbench starts with the five server
# keys MCP-Bench's README requires (GOOGLE_MAPS_API_KEY, NCI_API_KEY, HF_TOKEN, NPS_API_KEY,
# NASA_API_KEY) and nothing else of the run, on a docker network of its own, with no published port
# and no listening service. It runs only the five servers whose commands.json entry lists one of
# those keys (Google Maps, BioMCP, Hugging Face, National Parks, NASA Data), and none of the servers
# that run model text (the Scientific Computing server's sympy parse_expr, the OSINT server's
# commands), so no process that runs model-written code can read a key.
#
# gbench reaches this script only through `docker exec -i <container> python3 <this file> ...`,
# issued by its relay in the gbench process (mcp_bench.CredentialedServersRelay) for each session
# the model container's bridge (mcp_bench_bridge.py) opens. Each mode is one session, and its
# first line on stdout says whether the session is up: {"ready": true, ...} once it is, or
# {"ready": false, "error": ...} (and a nonzero exit) when it could not start. The relay answers
# the bridge only after that line: OK for a ready session, ERR otherwise (and ERR when docker exec
# ends without the line, e.g. the container is gone), and the bridge records every session that
# is not opened. After the ready line this script exits 0 however the session ends; any other
# exit code the relay sees then is docker exec's own (the exec or the container failed).
#
#   stdio <slug>     start the server's upstream command (commands.json: `cmd` split on spaces,
#                    `cwd` under mcp_servers/) with the environment upstream's MCP client gives a
#                    stdio server (the MCP SDK's default variables HOME, LOGNAME, PATH, SHELL, TERM,
#                    USER, plus the server's own `env` keys), print {"ready": true} once the
#                    process started, pass stdin to it and its stdout lines (newline-delimited
#                    JSON-RPC, as the MCP stdio transport writes them) back with every key value
#                    masked. It ends when either side closes.
#   http <slug>      start the HTTP server (Google Maps: `transport: http`) the way upstream's
#                    MCPConnector.start_http_server does (its `--port` replaced by a free port,
#                    MCP_SERVER_PORT set). One deviation: upstream starts it with its whole
#                    environment (os.environ.copy() plus the server's env and MCP_SERVER_PORT);
#                    here it gets only the MCP SDK's default variables above, its own key and
#                    MCP_SERVER_PORT, so the other servers' keys never reach it. It prints one
#                    line {"ready": true, "port": <port>} once it accepts connections on the
#                    container's loopback (or {"ready": false, "error": ...}), and keep it running
#                    until stdin closes or the server exits.
#   connect <port>   connect to 127.0.0.1:<port> (a server of an http session), print
#                    {"ready": true} once connected, pass stdin to it and its replies back, every
#                    key value masked (same-length masks, so an HTTP Content-Length or chunk size
#                    stays right).
#
# The tools, their descriptions and schemas, and every request and reply are the servers' own; the
# only change on the way back is a key value (as is, JSON-escaped or URL-encoded) replaced by
# asterisks. The servers' own stderr stays in this container. Standard library only.

import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse

MCPBENCH_DIR = os.environ.get("MCPBENCH_DIR", "/app/mcp-bench")
#: The five keys MCP-Bench's README requires; a server whose commands.json `env` lists one of them
#: runs here, never in the model container.
CREDENTIAL_KEYS = ("GOOGLE_MAPS_API_KEY", "NCI_API_KEY", "HF_TOKEN", "NPS_API_KEY", "NASA_API_KEY")
#: The variables the MCP SDK's stdio client passes to a server (mcp.client.stdio
#: DEFAULT_INHERITED_ENV_VARS on POSIX).
DEFAULT_ENV_VARS = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")
#: How long an http session waits for its server to accept connections.
HTTP_READY_TIMEOUT_S = 90
#: How long a session waits for its server to exit after its input closed, before stopping it.
EXIT_GRACE_S = 5
#: A connect session sends the bytes it held back (a possible start of a key) after this long
#: without new data.
IDLE_FLUSH_S = 0.5


def slug(name):
    """The session name of a server: its commands.json name lowercased, runs of other characters
    as one dash ("Google Maps" -> "google-maps")."""
    out, dash = [], False
    for c in str(name).lower():
        if c.isalnum():
            out.append(c)
            dash = False
        elif not dash and out:
            out.append("-")
            dash = True
    return "".join(out).strip("-")


def commands_path():
    """The server table: the pristine copy the model container's launcher keeps
    (commands.upstream.json) when it exists, else commands.json."""
    base = os.path.join(MCPBENCH_DIR, "mcp_servers")
    pristine = os.path.join(base, "commands.upstream.json")
    return pristine if os.path.isfile(pristine) else os.path.join(base, "commands.json")


def credentialed_servers(table=None):
    """{slug: (name, spec)} for every commands.json server whose `env` lists a CREDENTIAL_KEYS
    variable."""
    if table is None:
        with open(commands_path(), encoding="utf-8") as fh:
            table = json.load(fh)
    return {slug(n): (n, s) for n, s in table.items()
            if isinstance(s, dict) and any(k in CREDENTIAL_KEYS for k in s.get("env") or [])}


def server_env(spec, extra=None):
    """The environment of a server process: DEFAULT_ENV_VARS from this process, the server's own
    `env` keys that are set, and `extra`."""
    env = {k: os.environ[k] for k in DEFAULT_ENV_VARS if k in os.environ}
    for k in spec.get("env") or []:
        if os.environ.get(k):
            env[k] = os.environ[k]
    env.update(extra or {})
    return env


def server_cwd(spec):
    """The server's working directory: commands.json `cwd` ("../<dir>") under mcp_servers/, as
    upstream's runner resolves it."""
    cwd = str(spec.get("cwd") or "")
    if cwd.startswith("../"):
        cwd = cwd[3:]
    return os.path.join(MCPBENCH_DIR, "mcp_servers", cwd)


def with_port(argv, port):
    """`argv` with the value after --port (or in --port=N) replaced by `port`, and --port <port>
    appended when there is none (upstream's MCPConnector._update_command_port)."""
    out, i, found = [], 0, False
    while i < len(argv):
        a = argv[i]
        if a == "--port" and i + 1 < len(argv):
            out += [a, str(port)]
            i += 2
            found = True
            continue
        if a.startswith("--port="):
            out.append(f"--port={port}")
            found = True
        else:
            out.append(a)
        i += 1
    return out if found else out + ["--port", str(port)]


# --------------------------------------------------------------------------- #
# Masking: no key value leaves this container.
# --------------------------------------------------------------------------- #
def key_forms():
    """Every form of every set key value (as is, JSON-escaped, URL-encoded), longest first, as
    bytes."""
    forms = set()
    for k in CREDENTIAL_KEYS:
        v = (os.environ.get(k) or "").strip()
        if len(v) < 4:
            continue
        for f in (v, json.dumps(v)[1:-1], urllib.parse.quote(v, safe=""),
                  urllib.parse.quote_plus(v, safe=""), urllib.parse.quote(v)):
            if f:
                forms.add(f.encode("utf-8"))
    return sorted(forms, key=len, reverse=True)


def mask(data, forms):
    """`data` with every key form replaced by as many asterisks."""
    for f in forms:
        if f in data:
            data = data.replace(f, b"*" * len(f))
    return data


class StreamMasker:
    """Masks key forms in a byte stream read in arbitrary pieces: feed() returns what can be sent
    now and holds back the longest tail that could be the start of a key form; flush() sends the
    rest."""

    def __init__(self, forms):
        self.forms = forms
        self.longest = max((len(f) for f in forms), default=1)
        self.pending = b""

    def _held(self, data):
        for n in range(min(len(data), self.longest - 1), 0, -1):
            tail = data[-n:]
            if any(f.startswith(tail) for f in self.forms):
                return n
        return 0

    def feed(self, data):
        data = mask(self.pending + data, self.forms)
        n = self._held(data) if self.forms else 0
        self.pending = data[len(data) - n:] if n else b""
        return data[:len(data) - n] if n else data

    def flush(self):
        out, self.pending = mask(self.pending, self.forms), b""
        return out


# --------------------------------------------------------------------------- #
# Sessions.
# --------------------------------------------------------------------------- #
def _stop(proc):
    """Stop `proc` and its process group: wait EXIT_GRACE_S, then SIGTERM, then SIGKILL."""
    if proc.poll() is None:
        try:
            proc.wait(timeout=EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue
    try:                                    # anything the server started and left in its group
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def _stdin_to(write, close):
    """Copy this process's stdin to `write` until EOF, then call `close`."""
    try:
        while True:
            data = os.read(0, 65536)
            if not data:
                break
            write(data)
    except OSError:
        pass
    finally:
        try:
            close()
        except OSError:
            pass


def _out(data):
    while data:
        n = os.write(1, data)
        data = data[n:]


def run_stdio(name_slug):
    servers = credentialed_servers()
    if name_slug not in servers:
        _status(False, error="not a credentialed server")
        return 2
    _name, spec = servers[name_slug]
    if spec.get("transport") == "http":
        _status(False, error="an http server")
        return 2
    try:
        proc = subprocess.Popen(str(spec.get("cmd") or "").split(), cwd=server_cwd(spec),
                                env=server_env(spec), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                start_new_session=True)
    except OSError as e:
        _status(False, error=f"the server's command could not start ({type(e).__name__})")
        return 1
    forms = key_forms()
    _status(True)

    def write(data):
        proc.stdin.write(data)
        proc.stdin.flush()
    threading.Thread(target=_stdin_to, args=(write, proc.stdin.close), daemon=True).start()
    try:
        for line in iter(proc.stdout.readline, b""):
            _out(mask(line, forms))
    except OSError:
        pass
    finally:
        _stop(proc)
    return 0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _accepts(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


def _status(ready, **fields):
    """The session's first line on stdout: whether it is up (see the header)."""
    _out(json.dumps(dict(ready=ready, **fields)).encode("utf-8") + b"\n")


def run_http(name_slug):
    servers = credentialed_servers()
    if name_slug not in servers or servers[name_slug][1].get("transport") != "http":
        _status(False, error="not a credentialed http server")
        return 2
    _name, spec = servers[name_slug]
    port = _free_port()
    argv = with_port(str(spec.get("cmd") or "").split(), port)
    try:
        proc = subprocess.Popen(argv, cwd=server_cwd(spec),
                                env=server_env(spec, {"MCP_SERVER_PORT": str(port)}),
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        _status(False, error=f"the server's command could not start ({type(e).__name__})")
        return 1
    try:
        deadline = time.monotonic() + HTTP_READY_TIMEOUT_S
        while True:
            if proc.poll() is not None:
                _status(False, error=f"the server exited ({proc.returncode})")
                return 1
            if _accepts(port):
                break
            if time.monotonic() > deadline:
                _status(False, error="the server did not accept connections within "
                                     f"{HTTP_READY_TIMEOUT_S} s")
                return 1
            time.sleep(0.2)
        _status(True, port=port)
        while proc.poll() is None:          # until the session's input closes or the server exits
            ready, _, _ = select.select([0], [], [], 1.0)
            if ready and not os.read(0, 65536):
                break
    except OSError:
        pass
    finally:
        _stop(proc)
    return 0


def run_connect(port):
    try:
        port = int(port)
    except ValueError:
        port = 0
    if not 1024 <= port <= 65535:
        _status(False, error="not a port")
        return 2
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    except OSError as e:
        _status(False, error=f"cannot connect to the server's port ({type(e).__name__})")
        return 1
    sock.settimeout(None)
    _status(True)

    def half_close():
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    threading.Thread(target=_stdin_to, args=(sock.sendall, half_close), daemon=True).start()
    masker = StreamMasker(key_forms())
    try:
        while True:
            ready, _, _ = select.select([sock], [], [], IDLE_FLUSH_S)
            if not ready:
                _out(masker.flush())
                continue
            data = sock.recv(65536)
            if not data:
                break
            _out(masker.feed(data))
    except OSError:
        pass
    finally:
        try:
            _out(masker.flush())
        except OSError:
            pass
        sock.close()
    return 0


def main(argv):
    if len(argv) != 3 or argv[1] not in ("stdio", "http", "connect"):
        _status(False, error="usage: mcp_bench_credentialed.py stdio|http <server> | connect <port>")
        return 2
    mode, arg = argv[1], argv[2]
    if mode == "stdio":
        return run_stdio(arg)
    if mode == "http":
        return run_http(arg)
    return run_connect(arg)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
