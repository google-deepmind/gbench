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

"""Round-robin reverse proxy in front of several OpenAI-compatible endpoints.

scripts/eval_cluster.sh tunnels every vLLM pod of a cluster to a local port and puts this proxy
in front of them, so one gbench client (`--remote-endpoint http://127.0.0.1:<port>/v1`) spreads
its requests over every pod. Any method and path is forwarded unchanged (headers, body, query);
responses are streamed back as they arrive, so server-sent-event streams keep their chunking.

* Backends are given as ports on 127.0.0.1 or as base URLs: --backends 8001,8002 or
  --backends http://10.0.0.5:8000,http://10.0.0.6:8000.
* A backend that refuses the connection (for example a tunnel that dropped) is skipped for
  --down-seconds and the request goes to the next one; nothing is retried once a backend has
  accepted the request, so a request is never sent twice. When every backend refuses, the client
  gets 502 with the reasons.
* --expect-model ID: before listening, every backend's /v1/models must list ID, else the proxy
  exits 1 naming the backend and what it serves (so a foreign server on a tunnel port is never
  benchmarked). The check sends VLLM_API_KEY as a bearer token when it is set; client requests
  are forwarded with their own Authorization header.
* It binds 127.0.0.1 by default. A busy --port is an error (exit 1), never another port;
  --port 0 picks a free port. The line "async_proxy listening on http://HOST:PORT" is printed
  (flushed) once it accepts connections.
* Requests in flight to the backends are not capped (--max-connections 0, the default; aiohttp's
  own default would hold every request past its 100th in a queue), so the client's concurrency
  (eval_cluster.sh runs gbench with --batch-sizes 2048) reaches the pods. Every request holds two
  sockets, so the process raises its open-file soft limit to the hard limit at start.

Requires aiohttp (a gbench dependency). Stdlib otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import socket
import sys
import time
from typing import Dict, List, Optional, Sequence

import aiohttp
from aiohttp import web

#: Errors raised before a connection to the backend exists (refused, unreachable, or the
#: connect timeout), after which the request is safe to send to another backend.
_CONNECT_ERRORS = (aiohttp.ClientConnectorError,
                   getattr(aiohttp, "ConnectionTimeoutError", aiohttp.ServerTimeoutError))

#: Hop-by-hop headers (RFC 9110 7.6.1) and the ones aiohttp recomputes: never forwarded.
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
               "trailer", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


def parse_backends(raw: str) -> List[str]:
    """'8001,8002' or 'http://h:1,http://h:2' -> base URLs without a trailing '/'."""
    out = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            port = int(item)
            if not 1 <= port <= 65535:
                raise ValueError(f"backend port {item} is not a port number")
            out.append(f"http://127.0.0.1:{port}")
        elif item.startswith(("http://", "https://")):
            out.append(item.rstrip("/"))
        else:
            raise ValueError(f"backend {item!r} is neither a port nor an http(s) URL")
    if not out:
        raise ValueError("no backends given")
    return out


class RoundRobin:
    """The backend rotation over the backends that are up; a backend marked down is tried last
    until its time is up."""

    def __init__(self, backends: Sequence[str], down_seconds: float):
        self.backends = list(backends)
        self.down_seconds = down_seconds
        self._turn = 0
        self._down_until: Dict[str, float] = {}

    def order(self) -> List[str]:
        """Every backend once: the backends that are up, starting at the next one in rotation,
        then the ones marked down."""
        now = time.monotonic()
        up = [b for b in self.backends if self._down_until.get(b, 0.0) <= now]
        down = [b for b in self.backends if self._down_until.get(b, 0.0) > now]
        if up:
            k = self._turn % len(up)
            up = up[k:] + up[:k]
        self._turn += 1
        return up + down

    def mark_down(self, backend: str) -> None:
        self._down_until[backend] = time.monotonic() + self.down_seconds


async def check_models(session: aiohttp.ClientSession, backends: Sequence[str], model: str
                       ) -> List[str]:
    """The problems found when every backend's /v1/models must list `model` ([] when all do)."""
    problems = []
    key = os.environ.get("VLLM_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for base in backends:
        try:
            async with session.get(f"{base}/v1/models", headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    problems.append(f"{base}/v1/models answered {resp.status}: {body[:200]}")
                    continue
                data = json.loads(body)
        except Exception as e:                                               # noqa: BLE001
            problems.append(f"{base}/v1/models failed: {type(e).__name__}: {e}")
            continue
        served = [str(m.get("id")) for m in (data.get("data") or []) if isinstance(m, dict)]
        if model not in served:
            problems.append(f"{base} serves {served or 'no model'}, not {model!r}")
    return problems


_SESSION = web.AppKey("session", aiohttp.ClientSession)


def make_app(backends: Sequence[str], down_seconds: float = 30.0,
             connect_timeout: float = 10.0, max_connections: int = 0) -> web.Application:
    """The proxy application. `max_connections` caps the connections open to the backends at
    once (0: no cap; a request past the cap waits for a free connection)."""
    rotation = RoundRobin(backends, down_seconds)

    async def _startup(app: web.Application) -> None:
        app[_SESSION] = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=max_connections, limit_per_host=0),
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=connect_timeout))

    async def _cleanup(app: web.Application) -> None:
        await app[_SESSION].close()

    async def forward(request: web.Request) -> web.StreamResponse:
        session = request.app[_SESSION]
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        refused: List[str] = []
        for base in rotation.order():
            url = f"{base}{request.rel_url}"
            try:
                upstream = await session.request(request.method, url, headers=headers,
                                                 data=body if body else None,
                                                 allow_redirects=False)
            except _CONNECT_ERRORS as e:
                # The connection was never made, so nothing reached the backend: try the next.
                rotation.mark_down(base)
                refused.append(f"{base}: {type(e).__name__}: {e}")
                continue
            except aiohttp.ClientError as e:
                # The backend accepted the connection and then failed: the request may have
                # reached it, so it is not re-sent elsewhere.
                rotation.mark_down(base)
                return web.json_response(
                    {"error": {"message": f"backend {base} failed: {type(e).__name__}: {e}",
                               "type": "proxy_error"}}, status=502)
            try:
                response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
                for k, v in upstream.headers.items():
                    if k.lower() not in _HOP_BY_HOP:
                        response.headers.add(k, v)
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
            finally:
                upstream.release()
        return web.json_response(
            {"error": {"message": "every backend refused the connection", "backends": refused,
                       "type": "proxy_error"}}, status=502)

    app = web.Application(client_max_size=1024 ** 3)
    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    app.router.add_route("*", "/{tail:.*}", forward)
    return app


async def serve(args: argparse.Namespace) -> int:
    backends = parse_backends(args.backends)
    if args.expect_model:
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=args.check_timeout)) as session:
            problems = await check_models(session, backends, args.expect_model)
        if problems:
            for p in problems:
                print(f"async_proxy: {p}", file=sys.stderr, flush=True)
            return 1
    app = make_app(backends, args.down_seconds, args.connect_timeout, args.max_connections)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
        sock.listen(1024)
    except OSError as e:
        sock.close()
        print(f"async_proxy: cannot listen on {args.host}:{args.port} ({e}); the port is in use "
              "or not allowed", file=sys.stderr, flush=True)
        return 1
    port = sock.getsockname()[1]
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.SockSite(runner, sock).start()
    print(f"async_proxy listening on http://{args.host}:{port} -> {', '.join(backends)}",
          flush=True)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--backends", required=True,
                   help="Comma-separated ports on 127.0.0.1 or base URLs of the endpoints.")
    p.add_argument("--port", type=int, required=True,
                   help="Port to listen on (0 = a free port). A busy port is an error.")
    p.add_argument("--host", default="127.0.0.1", help="Address to bind (default 127.0.0.1).")
    p.add_argument("--expect-model", default=None,
                   help="Refuse to start unless every backend's /v1/models lists this id.")
    p.add_argument("--down-seconds", type=float, default=30.0,
                   help="How long a backend that refused a connection is tried last.")
    p.add_argument("--connect-timeout", type=float, default=10.0,
                   help="Seconds to wait for a backend connection before trying the next.")
    p.add_argument("--check-timeout", type=float, default=30.0,
                   help="Seconds for each --expect-model check.")
    p.add_argument("--max-connections", type=int, default=0,
                   help="Cap on the connections open to the backends at once; a request past it "
                        "waits for a free one (default 0: no cap).")
    return p


def raise_open_file_limit() -> int:
    """Raise the open-file soft limit to the hard limit (each request in flight holds a client
    and a backend socket); returns the soft limit in effect."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or soft < hard:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            soft = hard
        except (ValueError, OSError):
            pass
    return soft


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        parse_backends(args.backends)
    except ValueError as e:
        print(f"async_proxy: {e}", file=sys.stderr)
        return 2
    if args.max_connections < 0:
        print("async_proxy: --max-connections must be 0 (no cap) or a positive number",
              file=sys.stderr)
        return 2
    raise_open_file_limit()
    try:
        return asyncio.run(serve(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
