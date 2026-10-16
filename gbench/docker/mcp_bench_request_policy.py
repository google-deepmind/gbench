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
# The model-under-test request settings gbench applies inside a delegated harness image. The
# gbench-mcp-bench image (gbench/docker/mcp_bench_request_policy.py) and the gbench-toolbench
# image (docker/toolbench_request_policy.py) carry identical copies; each harness names its own
# environment prefix (MCP_ or TB_). The harness builds its chat requests as upstream does; this
# module only:
#
#   * sets the chat template's thinking switch to the run's mode:
#     extra_body.chat_template_kwargs.enable_thinking = <prefix>ENABLE_THINKING (true|false);
#   * sends the run's temperature (<prefix>TEMPERATURE: gbench sets it on every run, 1.0 with
#     --thinking and 0.0 without unless the operator sets another); both harnesses send none of
#     their own;
#   * with --thinking (<prefix>MAX_TOKENS set), sends the run's budget in place of the harness's
#     own max_tokens, which upstream wrote for non-reasoning generation, cut to the room the
#     served context window leaves after the prompt (the server's /tokenize count; without that
#     endpoint, a rejection that states the prompt's token count, or a max_tokens larger than the
#     window, is read and the request re-sent with less). Without --thinking the harness's
#     max_tokens is sent as it is;
#   * with <prefix>REQUEST_TIMEOUT_S set (gbench sets it with the --thinking budget, sized to it
#     as for its own requests), sends it as each request's timeout: that many seconds to read the
#     reply, and 5 s to connect as in the openai client's default. Otherwise the harness client's
#     own timeout applies (openai's default: 600 s to read the reply);
#   * counts the replies, those that stopped at max_tokens (finish_reason "length") and those
#     that stopped there with no answer text and no tool call (the budget spent in the reasoning
#     channel), and the requests that ended in an error the harness received:
#       - failed_requests: an error of the served endpoint or of the connection (timed_out counts
#         the timeouts among them; failed_by_error counts them by exception class and HTTP
#         status), and a context-window rejection of a request sent the --thinking budget that
#         the refits above did not resolve (the budget could not be fitted to the window),
#         unless /tokenize reported that the prompt alone leaves less than MIN_ROOM;
#       - rejected_over_context: the served window rejected prompt + max_tokens where the
#         harness's own max_tokens was sent, or where /tokenize reported that the prompt alone
#         leaves less than MIN_ROOM (the harness's own handling of such a rejection applies, as it
#         would upstream);
#       - cancelled: requests the harness abandoned in flight (an asyncio cancellation);
#     and appends the counts as one JSON line to <prefix>REQUEST_LOG when the process exits. The
#     line carries no URL, key or message text.

import asyncio
import atexit
import json
import os
import re
import threading
import urllib.request
from urllib.parse import urlsplit, urlunsplit

#: vLLM's rejection of a prompt + max_tokens longer than the context window.
_WINDOW_RE = re.compile(r"maximum context length is (\d+) tokens")
_PROMPT_RE = re.compile(r"prompt contains (at least )?(\d+) input tokens")
#: vLLM's rejection of a max_tokens larger than the whole window.
_OVER_WINDOW_RE = re.compile(r"cannot be greater than [\w.=]*?=(\d+)")
#: The least answer room a fitted request keeps; a prompt that leaves less is sent as it is and
#: the server's rejection reaches the harness, whose own handling applies.
MIN_ROOM = 256
#: Re-sends after a context rejection before the error is passed on.
MAX_REFITS = 8
#: Seconds to connect when <prefix>REQUEST_TIMEOUT_S is sent: the openai client's default
#: (DEFAULT_TIMEOUT connect=5.0), which only the time to read the reply replaces.
CONNECT_TIMEOUT_S = 5.0
_PLACEHOLDER_KEYS = {"", "EMPTY", "dummy", "none", "None"}


def _tokenize_url(base_url):
    """`<server root>/tokenize` for an OpenAI-compatible `.../v1` base, without user info, query
    or fragment; None when the base is not a URL."""
    try:
        parts = urlsplit(str(base_url or "").strip())
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    host = parts.hostname
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return None
    if port:
        host += f":{port}"
    path = re.sub(r"/v1/?$", "", parts.path.rstrip("/"))
    return urlunsplit((parts.scheme, host, path + "/tokenize", "", ""))


#: Set (True) on an exception `RequestPolicy.failed` counted as a failed request (an error of the
#: served endpoint or of the connection), so the launcher attributes it to the call it ended.
ENDPOINT_FAILURE_ATTR = "gbench_endpoint_failure"


class RequestPolicy:
    """The run's thinking switch, temperature, --thinking budget and request timeout for one
    harness, and the counts of its requests."""

    def __init__(self, prefix, base_url=None, env=None):
        env = os.environ if env is None else env
        self.prefix = prefix
        switch = (env.get(prefix + "ENABLE_THINKING") or "").strip().lower()
        self.enable_thinking = {"true": True, "false": False}.get(switch)
        raw = (env.get(prefix + "MAX_TOKENS") or "").strip()
        self.max_tokens = int(raw) if raw else None
        raw = (env.get(prefix + "TEMPERATURE") or "").strip()
        self.temperature = float(raw) if raw else None
        raw = (env.get(prefix + "REQUEST_TIMEOUT_S") or "").strip()
        self.request_timeout_s = int(raw) if raw else None
        self.log_path = (env.get(prefix + "REQUEST_LOG") or "").strip() or None
        self.tokenize_url = _tokenize_url(base_url)
        self._lock = threading.Lock()
        self._dumped = False
        self.counts = {"replies": 0, "cut_at_max_tokens": 0, "cut_in_reasoning": 0,
                       "fitted_to_context": 0, "refitted_after_rejection": 0,
                       "failed_requests": 0, "timed_out": 0, "rejected_over_context": 0,
                       "cancelled": 0}
        self.failed_by_error = {}
        self.max_tokens_sent = set()

    def settings(self):
        return {"enable_thinking": self.enable_thinking, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "request_timeout_s": self.request_timeout_s}

    def timeout(self):
        """The request's timeout for <prefix>REQUEST_TIMEOUT_S: read, write and pool that many
        seconds, connect CONNECT_TIMEOUT_S (an httpx.Timeout, which both clients accept per
        request; the plain number where httpx is absent). None when it is not set."""
        if self.request_timeout_s is None:
            return None
        try:
            import httpx
        except ImportError:
            return float(self.request_timeout_s)
        return httpx.Timeout(float(self.request_timeout_s), connect=CONNECT_TIMEOUT_S)

    def _count(self, key, n=1):
        with self._lock:
            self.counts[key] += n

    def room(self, kw, api_key=None):
        """Tokens the served window leaves after this request's prompt, from the server's
        /tokenize (chat template, tools and template switch included); None when unknown."""
        if not self.tokenize_url:
            return None
        body = {"model": kw.get("model"), "messages": kw.get("messages") or [],
                "add_generation_prompt": True}
        if kw.get("tools"):
            body["tools"] = kw["tools"]
        ctk = (kw.get("extra_body") or {}).get("chat_template_kwargs")
        if ctk:
            body["chat_template_kwargs"] = ctk
        headers = {"Content-Type": "application/json"}
        if api_key and str(api_key) not in _PLACEHOLDER_KEYS:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            data = json.dumps(body, default=str).encode("utf-8")
            req = urllib.request.Request(self.tokenize_url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                got = json.loads(resp.read())
            return int(got["max_model_len"]) - int(got["count"])
        except Exception:                                   # noqa: BLE001 - no such endpoint
            return None

    def prepare(self, kwargs, api_key=None):
        """The request the harness built, with the run's switch, temperature, budget and
        timeout."""
        return self._prepare(kwargs, api_key)[0]

    def _prepare(self, kwargs, api_key=None):
        """(prepare's request, the room /tokenize reported for it or None)."""
        kw = dict(kwargs)
        room = None
        if self.enable_thinking is not None:
            extra = dict(kw.get("extra_body") or {})
            ctk = dict(extra.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = self.enable_thinking
            extra["chat_template_kwargs"] = ctk
            kw["extra_body"] = extra
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.max_tokens is not None:
            kw.pop("max_completion_tokens", None)
            kw["max_tokens"] = self.max_tokens
            room = self.room(kw, api_key)
            if room is not None and MIN_ROOM <= room < self.max_tokens:
                kw["max_tokens"] = room
                self._count("fitted_to_context")
        if self.request_timeout_s is not None:
            kw["timeout"] = self.timeout()
        return kw, room

    def refit(self, error, current):
        """A smaller max_tokens after the server rejected prompt + max_tokens as longer than its
        context window; None when that is not the error, no budget is applied, or no smaller
        request keeps MIN_ROOM for the answer."""
        if self.max_tokens is None or not current:
            return None
        text = str(error)
        window, prompt = _WINDOW_RE.search(text), _PROMPT_RE.search(text)
        if window and prompt:
            limit, count = int(window.group(1)), int(prompt.group(2))
            if prompt.group(1) or count == limit - current + 1:
                new = current // 4          # a lower bound: the server stopped counting there
            else:
                new = limit - count - 16
        else:
            over = _OVER_WINDOW_RE.search(text)
            if not over:
                return None
            new = int(over.group(1)) - MIN_ROOM
        if new < MIN_ROOM or new >= current:
            return None
        return new

    def record(self, response, kw):
        """Count one reply of the model under test."""
        try:
            choice = response.choices[0]
            finish = getattr(choice, "finish_reason", None)
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) or ""
            tool_calls = getattr(message, "tool_calls", None)
        except (AttributeError, IndexError, TypeError):
            return
        with self._lock:
            self.counts["replies"] += 1
            if isinstance(kw.get("max_tokens"), int):
                self.max_tokens_sent.add(kw["max_tokens"])
            if finish == "length":
                self.counts["cut_at_max_tokens"] += 1
                if not str(content).strip() and not tool_calls:
                    self.counts["cut_in_reasoning"] += 1

    def failed(self, error, room=None):
        """Count one request that ended in `error`, which the harness receives: a context-window
        rejection where the harness's own max_tokens was sent (no --thinking budget), or where
        /tokenize reported that the prompt alone leaves less than MIN_ROOM, is
        rejected_over_context; anything else is a failed request, and `error` is marked with
        ENDPOINT_FAILURE_ATTR so the caller can tell which attempt of its own ended that way."""
        text = str(error)
        over_context = bool(_WINDOW_RE.search(text) or _OVER_WINDOW_RE.search(text))
        with self._lock:
            if over_context and (self.max_tokens is None
                                 or (room is not None and room < MIN_ROOM)):
                self.counts["rejected_over_context"] += 1
                return
            self.counts["failed_requests"] += 1
            try:
                setattr(error, ENDPOINT_FAILURE_ATTR, True)
            except Exception:                               # noqa: BLE001 - a record only
                pass
            if any("Timeout" in c.__name__ for c in type(error).__mro__):
                self.counts["timed_out"] += 1
            status = getattr(error, "status_code", None)
            key = type(error).__name__ + (f" {status}" if isinstance(status, int) else "")
            self.failed_by_error[key] = self.failed_by_error.get(key, 0) + 1

    def call(self, create, kwargs, api_key=None):
        """`create(**kwargs)` (a synchronous OpenAI client) under this policy."""
        kw, room = self._prepare(kwargs, api_key)
        for attempt in range(MAX_REFITS + 1):
            try:
                response = create(**kw)
            except Exception as exc:                        # noqa: BLE001 - re-raised below
                new = self.refit(exc, kw.get("max_tokens")) if attempt < MAX_REFITS else None
                if new is None:
                    self.failed(exc, room)
                    raise
                kw["max_tokens"] = new
                self._count("refitted_after_rejection")
                continue
            self.record(response, kw)
            return response

    async def acall(self, create, kwargs, api_key=None):
        """`await create(**kwargs)` (an asynchronous OpenAI client) under this policy."""
        kw, room = await asyncio.to_thread(self._prepare, kwargs, api_key)
        for attempt in range(MAX_REFITS + 1):
            try:
                response = await create(**kw)
            except asyncio.CancelledError:
                self._count("cancelled")
                raise
            except Exception as exc:                        # noqa: BLE001 - re-raised below
                new = self.refit(exc, kw.get("max_tokens")) if attempt < MAX_REFITS else None
                if new is None:
                    self.failed(exc, room)
                    raise
                kw["max_tokens"] = new
                self._count("refitted_after_rejection")
                continue
            self.record(response, kw)
            return response

    def dump(self):
        """Append this process's settings and counts to the request log (once)."""
        with self._lock:
            if self._dumped or not self.log_path:
                return
            self._dumped = True
            line = json.dumps({"settings": self.settings(), "counts": dict(self.counts),
                               "failed_by_error": dict(self.failed_by_error),
                               "max_tokens_sent": sorted(self.max_tokens_sent)})
        fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)

    def install_dump_at_exit(self):
        atexit.register(self.dump)
        return self
