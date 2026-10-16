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

"""Host-side Gemini key relay for containers that run model-controlled code.

A suite whose container runs code the model under test controls (its tool calls, code it wrote,
or a process that shares the container, user or environment with them) must not hand that
container a real Gemini key. When the same container also needs the Gemini API (a judge, web
search grounding, embeddings), it gets a per-run placeholder instead and sends its Gemini calls
to this relay: a thread of the gbench process, listening on one address the container reaches
(a docker network's gateway, or 127.0.0.1 for a container on the host network), which adds the
real key and forwards the call to the Gemini API.

Interface
---------
    from gbench.runners.eval_suites import gemini_relay

    bind = gemini_relay.network_gateway("mcp_bench", "bridge")   # e.g. "172.17.0.1"
    relay = gemini_relay.GeminiRelay("mcp_bench", bind).start()   # infra_required on a bad setup
    try:
        env = relay.client_env()
        # {"GEMINI_API_KEY": <placeholder>, "GEMINI_API_KEYS": <placeholder>,
        #  "GEMINI_OPENAI_BASE_URL": "http://172.17.0.1:<port>/v1beta/openai",
        #  "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:<port>"}
        # docker run ... -e GEMINI_API_KEY -e GEMINI_API_KEYS -e GEMINI_OPENAI_BASE_URL
        #     -e GBENCH_GEMINI_RELAY_URL ...   (names only; the values in the docker CLI's env)
        ...
    finally:
        relay.stop()
    rec = relay.record()          # into the result; never holds a key
    kinds = rec["harness_errors"]  # {kind: n}; rec["harness_error_events"] has their times

GeminiRelay(eval_name, bind_host, *, docs_url=None, port=0, openai_calls=OPENAI_CALLS,
            native_methods=NATIVE_METHODS, native_models=None,
            rejections_are_harness_errors=True, timeout_s=600.0, name="gemini-relay")

* `bind_host` is an IP address (or "localhost"); an unspecified address (0.0.0.0, ::) is
  refused, so the relay never listens on every interface. `port` 0 picks a free port.
* The constructor raises infra_required (swebench_common.infra_required) when no Gemini key is
  configured (base.gemini_api_keys: GEMINI_API_KEYS, else GEMINI_API_KEY, a list is rotated)
  or when GEMINI_OPENAI_BASE_URL is refused (base.gemini_openai_base_url: the Gemini API over
  https, or a loopback address for a local stub); start() raises it when the address cannot be
  bound. The upstream is fixed then: the checked OpenAI-compatible base URL, and its origin for
  the native paths.
* `placeholder` is the per-run token (PLACEHOLDER_PREFIX + 48 hex digits). A client presents it
  as its Gemini key: `Authorization: Bearer <placeholder>` (the OpenAI SDK), `x-goog-api-key`
  (google-genai, search_tool) or `?key=`. Anything else is answered 401 and counted in
  `requests_rejected`, never forwarded. Container code that checks GEMINI_OPENAI_BASE_URL the way
  base.check_gemini_openai_base_url does should accept the relay's URL only while every key it
  holds is a placeholder (is_placeholder), so a real key is never sent to it.
* Forwarded, and nothing else (another method or path is answered 403 and counted in
  `requests_refused`, never forwarded):
    - the OpenAI-compatible surface under OPENAI_PREFIX (/v1beta/openai): `openai_calls`, by
      default POST /chat/completions, POST /embeddings and GET /models, to
      <GEMINI_OPENAI_BASE_URL><path> with `Authorization: Bearer <key>`;
    - the native API: POST /v1beta/models/<model>:<method> for a method in `native_methods`
      (by default generateContent, streamGenerateContent, embedContent, batchEmbedContents) and,
      when `native_models` is given, a model in it, to <origin of GEMINI_OPENAI_BASE_URL><path>
      with `x-goog-api-key: <key>`. Only the `alt` query parameter is passed on.
  Only Content-Type and Accept are passed on from the client; every credential it sends is
  dropped. A request body over MAX_REQUEST_BODY is answered 413 unread.
* The keys are taken in turn, one per request; on 401, 403 or 429 the next key is tried. A key
  appears in no reply, log line, error message, record or command line: an error reply's body
  has every key masked, and the relay's own errors name only the exception type.
* Redirects are not followed: a 3xx goes back with its status and body, without its Location.
  Replies stream back as they arrive (status and Content-Type kept).
* Limits on what a client can make the relay hold: a chunk-size or trailer line over
  MAX_CHUNK_LINE bytes (or more than MAX_TRAILER_LINES trailer lines) is answered 400, and a body
  that would take the request bodies held at once over MAX_INFLIGHT_BODY_BYTES is answered 503.
  Both are counted in `requests_over_limits` {kind: n} (`chunk_line_too_long`,
  `too_many_trailers`, `relay_busy`), never as harness errors.
* Counted (record()): calls by method, surface and status; `harness_errors` {kind: n} with
  `harness_error_events` [{"t", "kind", "client"}] (the request's arrival time and the client
  address, for attribution by time window and container): `gemini_unreachable` (no answer,
  a timeout), `gemini_stream_broken` (a reply that broke off), `gemini_http_<status>` for a 3xx,
  401, 403, 408, 429 or 5xx after every key was tried. A 4xx answer to the request's content
  (400, 404, 413, 422, ...; a body over MAX_REQUEST_BODY is `request_too_large`) is a harness
  error too with `rejections_are_harness_errors` (the default: the suite's harness builds the
  requests, as a judge does); with False it goes to `rejections` / `rejection_events`, for a
  client whose request content is the model's. Unauthenticated and refused requests are never
  harness errors.

network_gateway(eval_name, network, docs_url=None) is the IPv4 gateway of a docker network
(`docker network inspect`), the address a container on it reaches the host at; infra_required
when docker cannot name one. For the host network use "127.0.0.1".

search_tool honours GBENCH_GEMINI_RELAY_URL: with it set and every configured key a
placeholder, its grounding requests go to <relay>/v1beta/models/<model>:generateContent.

Standard library only at import (base and swebench_common are imported where needed), so code
running in a container that mounts the gbench tree can import PLACEHOLDER_PREFIX and
is_placeholder from here.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Every placeholder starts with this (search_tool keeps the same constant).
PLACEHOLDER_PREFIX = "gbench-gemini-relay-"
#: The variable that tells a container client (search_tool) the relay's origin.
RELAY_URL_ENV = "GBENCH_GEMINI_RELAY_URL"
#: The client path prefix of the OpenAI-compatible surface.
OPENAI_PREFIX = "/v1beta/openai"
#: The OpenAI-compatible calls forwarded by default: (method, path below OPENAI_PREFIX).
OPENAI_CALLS: FrozenSet[Tuple[str, str]] = frozenset({
    ("POST", "/chat/completions"), ("POST", "/embeddings"), ("GET", "/models")})
#: The native methods forwarded by default (POST /v1beta/models/<model>:<method>).
NATIVE_METHODS: FrozenSet[str] = frozenset({
    "generateContent", "streamGenerateContent", "embedContent", "batchEmbedContents"})
#: The largest request body read (a Gemini request is capped far below it).
MAX_REQUEST_BODY = 64 << 20
#: How many requests are handled at once (more wait).
MAX_CONCURRENT_REQUESTS = 128
#: The most request-body bytes held in memory at once over all requests; a request whose body
#: would go over it is answered 503 (the relay is busy), so whoever holds the placeholder cannot
#: make the relay buffer without limit.
MAX_INFLIGHT_BODY_BYTES = 256 << 20
#: The longest chunk-size or trailer line of a chunked body read (a longer one is answered 400).
MAX_CHUNK_LINE = 1024
#: The most trailer lines of a chunked body read (more is answered 400).
MAX_TRAILER_LINES = 64
#: Statuses on which the next key is tried.
ROTATE_STATUSES = (401, 403, 429)
#: Statuses that are the service's failure whatever the request (with every 3xx and 5xx).
SERVICE_FAILURE_STATUSES = (401, 403, 408, 429)
#: The most of an error reply read (to mask the keys in it) before it is sent on.
_ERROR_BODY_CAP = 4 << 20

_NATIVE_PATH = re.compile(r"^/v1beta/models/([A-Za-z0-9][A-Za-z0-9._-]{0,127}):([A-Za-z]+)$")
_TOKEN_SHAPE = re.compile(r"^" + re.escape(PLACEHOLDER_PREFIX) + r"[0-9a-f]{48}$")


def is_placeholder(key: Optional[str]) -> bool:
    """True when `key` has the shape of a relay placeholder (never a real Gemini key)."""
    return bool(_TOKEN_SHAPE.match((key or "").strip()))


def _check_bind_host(bind_host: str) -> str:
    host = (bind_host or "").strip()
    if host == "localhost":
        return host
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError("the Gemini relay listens on an IP address (a docker network gateway or "
                         "a loopback address), not on a host name") from None
    if addr.is_unspecified:
        raise ValueError("the Gemini relay never listens on every interface (0.0.0.0 or ::); "
                         "give it a docker network gateway or a loopback address")
    return host


def network_gateway(eval_name: str, network: str, docs_url: Optional[str] = None) -> str:
    """The IPv4 gateway of the docker network `network` (the host's address on it), or
    infra_required when docker cannot name one."""
    from .swebench_common import infra_required
    docs_url = docs_url or f"docs/evals/{eval_name}.md"
    net = (network or "").strip() or "bridge"
    try:
        proc = subprocess.run(
            ["docker", "network", "inspect", "-f",
             "{{range .IPAM.Config}}{{.Gateway}} {{end}}", net],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise infra_required(eval_name, f"gbench could not inspect the docker network {net!r} "
                             f"({type(exc).__name__}) to find the address its containers reach "
                             "the Gemini relay at.", docs_url) from exc
    gateways = []
    for word in (proc.stdout or "").split() if proc.returncode == 0 else []:
        try:
            if ipaddress.ip_address(word).version == 4:
                gateways.append(word)
        except ValueError:
            continue
    if not gateways:
        raise infra_required(eval_name, f"the docker network {net!r} has no IPv4 gateway "
                             "(`docker network inspect` names none), so its containers cannot "
                             "reach the Gemini relay gbench runs on this host.", docs_url)
    return gateways[0]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Follows nothing: a 3xx is raised as an HTTPError and relayed as it is (without its
    Location), so the key is never re-sent to another URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


class _OverLimit(Exception):
    """A request over one of the relay's limits: answered `status` with `message`, counted in
    requests_over_limits under `kind`."""

    def __init__(self, kind: str, status: int, message: str):
        super().__init__(message)
        self.kind, self.status, self.message = kind, status, message


class _Handler(BaseHTTPRequestHandler):
    server_version = "gbench-gemini-relay"

    def log_message(self, format, *args):     # noqa: A002 - no request lines on stderr
        return

    def _dispatch(self):
        self.server.relay._handle(self)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _dispatch


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def _shape(text: str) -> str:
    return re.sub(r"[^\x21-\x7e]", "?", text or "")[:160]


class GeminiRelay:
    """See the module docstring."""

    def __init__(self, eval_name: str, bind_host: str, *, docs_url: Optional[str] = None,
                 port: int = 0, openai_calls: Iterable[Tuple[str, str]] = OPENAI_CALLS,
                 native_methods: Iterable[str] = NATIVE_METHODS,
                 native_models: Optional[Iterable[str]] = None,
                 rejections_are_harness_errors: bool = True, timeout_s: float = 600.0,
                 name: str = "gemini-relay"):
        from . import base
        from .swebench_common import infra_required
        self.eval_name = eval_name
        self._docs_url = docs_url or f"docs/evals/{eval_name}.md"
        self.bind_host = _check_bind_host(bind_host)
        keys = base.gemini_api_keys()
        if not keys:
            raise infra_required(eval_name, "GEMINI_API_KEY (or GEMINI_API_KEYS) is unset, and "
                                 "the Gemini relay the suite's containers use needs it.",
                                 self._docs_url)
        upstream, why = base.gemini_openai_base_url()
        if upstream is None:
            raise infra_required(eval_name, why + ".", self._docs_url)
        parts = urllib.parse.urlsplit(upstream)
        self._openai_base = upstream.rstrip("/")
        self._origin = urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        self._keys = list(keys)
        self._port = int(port)
        self._openai_calls = frozenset((m.upper(), p) for m, p in openai_calls)
        self._native_methods = frozenset(native_methods)
        self._native_models = frozenset(native_models) if native_models is not None else None
        self._rejections_are_harness_errors = bool(rejections_are_harness_errors)
        self._timeout_s = float(timeout_s)
        self._name = name
        self.placeholder = PLACEHOLDER_PREFIX + secrets.token_hex(24)
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
        self._next_key = 0
        self._calls: Dict[Tuple[str, str, str], int] = {}
        self._errors: Dict[str, int] = {}
        self._events: List[Dict[str, Any]] = []
        self._rejections: Dict[str, int] = {}
        self._rejection_events: List[Dict[str, Any]] = []
        self._refused: Dict[Tuple[str, str], int] = {}
        self.requests_relayed = 0
        self.requests_rejected = 0
        self._over_limits: Dict[str, int] = {}
        self._inflight = 0
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    # -- addresses ---------------------------------------------------------------------------- #
    @property
    def origin(self) -> str:
        """http://<bind_host>:<port> (the port it listens on once started)."""
        port = self._server.server_address[1] if self._server else self._port
        host = f"[{self.bind_host}]" if ":" in self.bind_host else self.bind_host
        return f"http://{host}:{port}"

    @property
    def openai_base_url(self) -> str:
        """The OpenAI-compatible base URL a client uses in place of GEMINI_OPENAI_BASE_URL."""
        return self.origin + OPENAI_PREFIX

    @property
    def native_generate_url(self) -> str:
        """The native generateContent URL template ("{model}" left in), search_tool's shape."""
        return self.origin + "/v1beta/models/{model}:generateContent"

    def client_env(self) -> Dict[str, str]:
        """The variables a container client needs: the placeholder as its Gemini key, and the
        relay's addresses. Pass them to docker by name."""
        return {"GEMINI_API_KEY": self.placeholder, "GEMINI_API_KEYS": self.placeholder,
                "GEMINI_OPENAI_BASE_URL": self.openai_base_url, RELAY_URL_ENV: self.origin}

    # -- lifecycle ---------------------------------------------------------------------------- #
    def start(self) -> "GeminiRelay":
        from .swebench_common import infra_required
        try:
            server = _Server((self.bind_host, self._port), _Handler)
        except OSError as exc:
            raise infra_required(self.eval_name, f"the Gemini relay could not listen on "
                                 f"{self.bind_host}:{self._port or 'a free port'} "
                                 f"({type(exc).__name__}: {exc.strerror or exc}).",
                                 self._docs_url) from exc
        server.relay = self
        self._server = server
        self._port = server.server_address[1]
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1},
                                        name=self._name, daemon=True)
        self._thread.start()
        logger.info("[%s] Gemini relay listening on %s", self.eval_name, self.origin)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def __enter__(self) -> "GeminiRelay":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- accounting --------------------------------------------------------------------------- #
    def _event(self, kind: str, when: float, client: str, rejection: bool = False) -> None:
        rec = {"t": when, "kind": kind, "client": client}
        with self._lock:
            if rejection and not self._rejections_are_harness_errors:
                self._rejections[kind] = self._rejections.get(kind, 0) + 1
                self._rejection_events.append(rec)
            else:
                self._errors[kind] = self._errors.get(kind, 0) + 1
                self._events.append(rec)
        logger.warning("[%s] Gemini relay: %s", self.eval_name, kind)

    def _count_call(self, method: str, surface: str, status: str) -> None:
        with self._lock:
            key = (method, surface, status)
            self._calls[key] = self._calls.get(key, 0) + 1

    def record(self) -> Dict[str, Any]:
        """What a result records about the relay (never a key)."""
        with self._lock:
            return {
                "listens": self.origin,
                "relays_to": {"openai_compatible": self._openai_base,
                              "native": self._origin + "/v1beta/models"},
                "adds": "the Gemini API key (GEMINI_API_KEYS, else GEMINI_API_KEY; "
                        f"{len(self._keys)} key(s) in turn)",
                "clients_hold": "a per-run placeholder",
                "openai_calls": sorted(f"{m} {OPENAI_PREFIX}{p}" for m, p in self._openai_calls),
                "native_methods": sorted(self._native_methods),
                "native_models": (sorted(self._native_models) if self._native_models is not None
                                  else None),
                "calls": [{"method": m, "surface": s, "status": st, "requests": n}
                          for (m, s, st), n in sorted(self._calls.items())],
                "requests_relayed": self.requests_relayed,
                "requests_rejected": self.requests_rejected,
                "requests_over_limits": dict(sorted(self._over_limits.items())),
                "requests_refused": [{"method": m, "path": p, "requests": n}
                                     for (m, p), n in sorted(self._refused.items())],
                "harness_errors": dict(sorted(self._errors.items())),
                "harness_error_events": list(self._events),
                "rejections": dict(sorted(self._rejections.items())),
                "rejection_events": list(self._rejection_events),
            }

    def harness_error_count(self) -> int:
        with self._lock:
            return sum(self._errors.values())

    # -- request handling --------------------------------------------------------------------- #
    def _mask(self, data: bytes) -> bytes:
        for key in sorted(self._keys, key=len, reverse=True):
            for form in {key, urllib.parse.quote(key, safe=""), urllib.parse.quote_plus(key)}:
                data = data.replace(form.encode("utf-8"), b"[REDACTED]")
        return data

    def _rotation(self) -> List[str]:
        with self._lock:
            start = self._next_key % len(self._keys)
            self._next_key += 1
        return self._keys[start:] + self._keys[:start]

    def _presented(self, h: BaseHTTPRequestHandler, query: List[Tuple[str, str]]) -> str:
        auth = (h.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        goog = (h.headers.get("x-goog-api-key") or "").strip()
        if goog:
            return goog
        for k, v in query:
            if k == "key":
                return v.strip()
        return ""

    def _route(self, method: str, path: str) -> Optional[Tuple[str, str, str]]:
        """(target URL without query, surface label, key header) for an allowed call."""
        if path.startswith(OPENAI_PREFIX + "/"):
            sub = path[len(OPENAI_PREFIX):]
            if (method, sub) in self._openai_calls:
                return self._openai_base + sub, f"openai {sub}", "authorization"
            return None
        m = _NATIVE_PATH.match(path)
        if (m and method == "POST" and m.group(2) in self._native_methods
                and (self._native_models is None or m.group(1) in self._native_models)):
            return self._origin + path, f"native {m.group(1)}:{m.group(2)}", "x-goog-api-key"
        return None

    def _take_body_bytes(self, n: int) -> bool:
        """Reserve `n` bytes of MAX_INFLIGHT_BODY_BYTES for a body being read; False (nothing
        reserved) when that would go over it."""
        with self._lock:
            if self._inflight + n > MAX_INFLIGHT_BODY_BYTES:
                return False
            self._inflight += n
            return True

    def _give_body_bytes(self, n: int) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - n)

    def _over_limit(self, kind: str) -> None:
        with self._lock:
            self._over_limits[kind] = self._over_limits.get(kind, 0) + 1
        logger.warning("[%s] Gemini relay: request over a limit (%s)", self.eval_name, kind)

    @staticmethod
    def _chunk_line(h: BaseHTTPRequestHandler) -> bytes:
        """One chunk-size or trailer line, read up to MAX_CHUNK_LINE bytes; _OverLimit
        ("chunk_line_too_long") for a longer line."""
        line = h.rfile.readline(MAX_CHUNK_LINE + 1)
        if len(line) > MAX_CHUNK_LINE:
            raise _OverLimit("chunk_line_too_long", 400, f"a chunked-body line over "
                                                         f"{MAX_CHUNK_LINE} bytes")
        return line

    def _read_body(self, h: BaseHTTPRequestHandler) -> Tuple[Optional[bytes], int]:
        """(the request body, b"" for none, or None when it is over MAX_REQUEST_BODY (not read
        further); the bytes reserved for it in MAX_INFLIGHT_BODY_BYTES, which the caller gives
        back). ValueError for a framing it cannot read; _OverLimit for a line over
        MAX_CHUNK_LINE, too many trailers, or a body the in-flight budget cannot hold."""
        reserved = 0
        try:
            if "chunked" in (h.headers.get("Transfer-Encoding") or "").lower():
                chunks, total = [], 0
                while True:
                    size = int(self._chunk_line(h).split(b";", 1)[0].strip() or b"0", 16)
                    if size < 0:
                        raise ValueError("negative chunk size")
                    if size == 0:
                        for _ in range(MAX_TRAILER_LINES):
                            if self._chunk_line(h) in (b"\r\n", b"\n", b""):
                                return b"".join(chunks), reserved
                        raise _OverLimit("too_many_trailers", 400,
                                         f"more than {MAX_TRAILER_LINES} trailer lines")
                    total += size
                    if total > MAX_REQUEST_BODY:
                        return None, reserved
                    if not self._take_body_bytes(size):
                        raise _OverLimit("relay_busy", 503, "the relay holds too many request "
                                                            "bodies at once; retry")
                    reserved += size
                    chunks.append(h.rfile.read(size))
                    self._chunk_line(h)
            length = int(h.headers.get("Content-Length") or 0)
            if length < 0:
                raise ValueError("negative Content-Length")
            if length > MAX_REQUEST_BODY:
                return None, reserved
            if not length:
                return b"", reserved
            if not self._take_body_bytes(length):
                raise _OverLimit("relay_busy", 503, "the relay holds too many request bodies at "
                                                    "once; retry")
            reserved = length
            return h.rfile.read(length), reserved
        except BaseException:
            self._give_body_bytes(reserved)
            raise

    @staticmethod
    def _send_json(h: BaseHTTPRequestHandler, status: int, message: str) -> None:
        data = json.dumps({"error": {"code": status, "message": f"gbench Gemini relay: "
                                                               f"{message}"}}).encode("utf-8")
        h.send_response_only(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.send_header("Connection", "close")
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(data)
        h.wfile.flush()
        h.close_connection = True

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        with self._slots:
            try:
                self._handle_request(h)
            except OSError:              # the client went away
                h.close_connection = True

    def _handle_request(self, h: BaseHTTPRequestHandler) -> None:
        arrived = time.time()
        client = h.client_address[0] if h.client_address else ""
        split = urllib.parse.urlsplit(h.path)
        path = split.path or "/"
        query = urllib.parse.parse_qsl(split.query, keep_blank_values=True)
        presented = self._presented(h, query).encode("utf-8")
        if not hmac.compare_digest(presented, self.placeholder.encode("utf-8")):
            with self._lock:
                self.requests_rejected += 1
            self._send_json(h, 401, "missing or wrong placeholder key")
            return
        route = None if split.fragment or "/../" in path + "/" else self._route(h.command, path)
        if route is None:
            with self._lock:
                key = (h.command, _shape(path))
                self._refused[key] = self._refused.get(key, 0) + 1
            logger.warning("[%s] Gemini relay refused a %s request", self.eval_name, h.command)
            self._send_json(h, 403, f"{h.command} {_shape(path)} is not a Gemini API call this "
                                    "relay forwards")
            return
        target, surface, key_header = route
        try:
            body, reserved = self._read_body(h)
        except _OverLimit as e:
            self._over_limit(e.kind)
            self._send_json(h, e.status, e.message)
            return
        except ValueError:
            with self._lock:
                key = (h.command, _shape(path))
                self._refused[key] = self._refused.get(key, 0) + 1
            self._send_json(h, 400, "unreadable request body framing")
            return
        try:
            self._forward(h, arrived, client, target, surface, key_header, query, body)
        finally:
            self._give_body_bytes(reserved)

    def _forward(self, h: BaseHTTPRequestHandler, arrived: float, client: str, target: str,
                 surface: str, key_header: str, query: List[Tuple[str, str]],
                 body: Optional[bytes]) -> None:
        if body is None:
            self._count_call(h.command, surface, "413 (relay)")
            self._event("request_too_large", arrived, client, rejection=True)
            self._send_json(h, 413, f"request body over {MAX_REQUEST_BODY} bytes")
            return
        body = body or None
        alt = [(k, v) for k, v in query if k == "alt"]
        url = target + ("?" + urllib.parse.urlencode(alt) if alt else "")
        resp = None
        status = 0
        for i, key in enumerate(self._rotation()):
            req = urllib.request.Request(url, data=body, method=h.command)
            for name in ("Content-Type", "Accept"):
                if h.headers.get(name):
                    req.add_header(name, h.headers[name])
            req.add_header("User-Agent", "gbench-gemini-relay")
            if key_header == "authorization":
                req.add_header("Authorization", f"Bearer {key}")
            else:
                req.add_header("x-goog-api-key", key)
            try:
                resp = self._opener.open(req, timeout=self._timeout_s)
            except urllib.error.HTTPError as e:
                resp = e
            except Exception as e:           # no answer: never the exception's text
                self._count_call(h.command, surface, "unreachable")
                self._event("gemini_unreachable", arrived, client)
                self._send_json(h, 502, f"the Gemini API did not answer ({type(e).__name__})")
                return
            status = int(getattr(resp, "status", None) or resp.code)
            if status in ROTATE_STATUSES and i + 1 < len(self._keys):
                try:
                    resp.read()
                    resp.close()
                except Exception:
                    pass
                continue
            break
        with self._lock:
            self.requests_relayed += 1
        self._count_call(h.command, surface, str(status))
        ok = 200 <= status < 300
        ctype = resp.headers.get("Content-Type") if resp.headers is not None else None
        try:
            if not ok:
                try:
                    data = resp.read(_ERROR_BODY_CAP)
                except Exception:
                    data = b""
                self._reply(h, status, ctype, self._mask(data or b""))
                if status >= 500 or status < 200 or 300 <= status < 400 \
                        or status in SERVICE_FAILURE_STATUSES:
                    self._event(f"gemini_http_{status}", arrived, client)
                else:
                    self._event(f"gemini_http_{status}", arrived, client, rejection=True)
                return
            h.send_response_only(status)
            if ctype:
                h.send_header("Content-Type", ctype)
            h.send_header("Connection", "close")
            h.end_headers()
            while True:
                try:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                except Exception:
                    self._event("gemini_stream_broken", arrived, client)
                    h.close_connection = True
                    return
                if not chunk:
                    break
                h.wfile.write(chunk)
                h.wfile.flush()
            h.close_connection = True
        finally:
            try:
                resp.close()
            except Exception:
                pass

    @staticmethod
    def _reply(h: BaseHTTPRequestHandler, status: int, ctype: Optional[str], data: bytes) -> None:
        h.send_response_only(status)
        if ctype:
            h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(data)))
        h.send_header("Connection", "close")
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(data)
        h.wfile.flush()
        h.close_connection = True
