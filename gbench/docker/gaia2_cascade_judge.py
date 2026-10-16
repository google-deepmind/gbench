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
# Gemini judge proxy for the gbench gaia2 orchestrator. It runs INSIDE the gbench-gaia2 image, on
# 127.0.0.1:$GAIA2_JUDGE_PORT only, and is the only thing in a gaia2 run that talks to an external
# model: ARE's LLM judge (`--judge_provider local --judge_endpoint http://127.0.0.1:<port>/v1
# --judge_model openai/gbench-cascade`) calls it through LiteLLM's OpenAI-compatible client, and it
# answers every judge request with gbench's Gemini judge cascade in place of ARE's default
# meta-llama/Meta-Llama-3.3-70B-Instruct. ARE's judge prompts are untouched.
#
# The cascade loop (rounds, backoff, the usage tally and the harness events) is the one
# wildclawbench's proxy uses: wildclawbench_cascade_judge.py, copied next to this file in the
# image. The models are the ones gbench passes at run time in GBENCH_JUDGE_MODELS (always
# base.judge_cascade(); without it the proxy does not start, so no default list baked into the
# image is ever used). This server serves only:
#   POST <base>/chat/completions  -> judge_chat_completion (HTTP 400 for a request it does not
#                                    serve, HTTP 503 when no cascade model answered)
#   GET  <base>/models            -> a one-model list, for probes
#   GET  <anything else>          -> {"status": "ok"}, for the entrypoint's health check
# There is no model relay and no web search here.
#
# The judge request: ARE sends the model tag and its messages only; they go to Gemini's
# OpenAI-compatible chat endpoint verbatim with the cascade's model and temperature 0 (the ARE
# paper, arXiv:2509.17158 section 4.1: the verifier runs "with temperature set to 0"; the current
# upstream judge engine, gaia2-cli gaia2_cli/judge/engine.py, sends temperature 0). No max_tokens is
# sent (ARE sends none). The reply's `model` is set to the Gemini model that answered, which the
# launcher records with every soft check.
#
# Credentials: the ARE agent's tool calls run in this container, so it never holds a Gemini key.
# GEMINI_API_KEYS / GEMINI_API_KEY hold a per-run placeholder of gbench's Gemini relay (a thread of
# the gbench process on the docker network's gateway, gemini_relay.GeminiRelay), which adds the key;
# every judge call goes to the relay's URL, GBENCH_GEMINI_RELAY_URL + /v1beta/openai (and
# GEMINI_OPENAI_BASE_URL, when set, must be that URL). A key that is not a relay placeholder, or a
# missing relay URL, stops the proxy before it listens.
#
# Every Gemini call is counted in $WILDCLAW_WORKDIR/gemini_usage.json, every judge call no cascade
# model served is written to $WILDCLAW_WORKDIR/harness_events.jsonl (the entrypoint points
# WILDCLAW_WORKDIR at $GAIA2_WORKDIR), and every judge call to $WILDCLAW_WORKDIR/judge_calls.jsonl
# (time, the request's sha256, the model that answered or the outage, the first characters of the
# reply); the launcher and the gbench runner copy them into the run's records.

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wildclawbench_cascade_judge as cascade  # noqa: E402

logger = logging.getLogger("gaia2.cascade_judge")

#: The one address the proxy listens on: ARE runs in the same container.
LISTEN_HOST = "127.0.0.1"
#: The judge's temperature (paper section 4.1; gaia2-cli create_litellm_engine).
JUDGE_TEMPERATURE = 0.0
#: Seconds one Gemini call of a judge request may take before the cascade asks its next model.
REQUEST_TIMEOUT_S = 120.0
#: Request fields not forwarded: the model (the cascade picks it), the temperature (pinned above),
#: streaming, and provider-specific fields Gemini's OpenAI-compatible endpoint rejects.
_DROP = ("model", "temperature", "stream", "provider", "transforms", "route", "models",
         "thinking", "reasoning", "reasoning_effort")
#: Every judge call, one JSON line each (in $WILDCLAW_WORKDIR).
JUDGE_CALLS = "judge_calls.jsonl"
#: Characters of a reply kept in JUDGE_CALLS.
REPLY_CHARS = 600
#: The prefix of every gbench Gemini relay placeholder (gemini_relay.PLACEHOLDER_PREFIX), then 48
#: hex digits.
RELAY_PLACEHOLDER_PREFIX = "gbench-gemini-relay-"
#: The relay's origin, http://<address>:<port> (gemini_relay.RELAY_URL_ENV).
RELAY_URL_ENV = "GBENCH_GEMINI_RELAY_URL"
#: The relay's OpenAI-compatible path (gemini_relay.OPENAI_PREFIX).
RELAY_OPENAI_PREFIX = "/v1beta/openai"
_PLACEHOLDER = re.compile(r"^" + re.escape(RELAY_PLACEHOLDER_PREFIX) + r"[0-9a-f]{48}$")
_LOG_LOCK = threading.Lock()


class RelayRefused(RuntimeError):
    """The proxy's credentials are not a relay placeholder with the relay's URL: no call is made."""


def is_placeholder(key):
    """True when `key` has the shape of a relay placeholder (gemini_relay.is_placeholder)."""
    return bool(_PLACEHOLDER.match((key or "").strip()))


def relay_openai_base():
    """The relay's OpenAI-compatible base URL (no trailing slash). Raises RelayRefused, naming the
    variables only, unless every configured key is a relay placeholder, GBENCH_GEMINI_RELAY_URL is
    an http(s) origin, and GEMINI_OPENAI_BASE_URL is unset or that same URL."""
    keys = cascade.gemini_keys()
    if not keys or not all(is_placeholder(k) for k in keys):
        raise RelayRefused("the gaia2 container runs the agent's tool calls, so it holds only a "
                           "placeholder of gbench's Gemini relay (GEMINI_API_KEYS / "
                           "GEMINI_API_KEY); a real key is refused")
    relay = (os.environ.get(RELAY_URL_ENV) or "").strip()
    try:
        parts = urlsplit(relay)
        parts.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        parts = None
    if (not relay or parts is None or parts.scheme not in ("http", "https") or not parts.hostname
            or "@" in parts.netloc or parts.path not in ("", "/") or parts.query
            or parts.fragment or any(c.isspace() or ord(c) < 32 for c in relay)):
        raise RelayRefused(f"{RELAY_URL_ENV} is not an http(s) origin (scheme://host:port)")
    base = f"{parts.scheme}://{parts.netloc}{RELAY_OPENAI_PREFIX}"
    given = (os.environ.get("GEMINI_OPENAI_BASE_URL") or "").strip().rstrip("/")
    if given and given != base:
        raise RelayRefused(f"GEMINI_OPENAI_BASE_URL is not the relay's URL ({RELAY_URL_ENV} + "
                           f"{RELAY_OPENAI_PREFIX})")
    return base


def _log_call(record):
    workdir = os.environ.get("WILDCLAW_WORKDIR", "").strip()
    if not workdir:
        return
    with _LOG_LOCK:
        try:
            with open(os.path.join(workdir, JUDGE_CALLS), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:  # the record is kept for audit, never a reason to fail a judge call
            logger.warning("gaia2: could not write %s: %s", JUDGE_CALLS, e)


def judge_request(body):
    """The payload sent to Gemini for ARE's judge request `body` (the cascade adds the model)."""
    if not isinstance(body, dict):
        raise cascade.UnsupportedRequest("the judge request must be a JSON object")
    if body.get("tools") is not None or body.get("tool_choice") is not None:
        raise cascade.UnsupportedRequest("ARE's judge sends no tools; a request with tools is "
                                         "not served")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise cascade.UnsupportedRequest("the judge request needs messages")
    payload = {k: v for k, v in body.items() if k not in _DROP}
    payload["temperature"] = JUDGE_TEMPERATURE
    return payload


def judge_chat_completion(body):
    """One judge request through the Gemini cascade: the OpenAI ChatCompletion dict of the first
    non-empty reply, its `model` set to the Gemini model that answered. Raises (after the cascade
    recorded the harness event) when no model answered."""
    payload = judge_request(body)
    digest = hashlib.sha256(json.dumps(payload.get("messages"), sort_keys=True,
                                       default=str).encode("utf-8")).hexdigest()
    started = time.time()

    def _one(model):
        parsed = cascade._post_json(relay_openai_base() + "/chat/completions",
                                    dict(payload, model=model),
                                    {"Authorization": f"Bearer {cascade.next_gemini_key()}"},
                                    REQUEST_TIMEOUT_S)
        choices = parsed.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if content is None or str(content).strip() == "":
            raise RuntimeError("empty content")
        parsed["model"] = model
        return parsed

    try:
        # Every Gemini call goes to the relay: the cascade checks the relay's URL (not the
        # module's own GEMINI_OPENAI_BASE_URL rule) before any placeholder is sent.
        reply = cascade._run_cascade(_one, "judge", base_check=relay_openai_base)
    except Exception as e:
        _log_call({"time": started, "request_sha256": digest, "served_model": None, "ok": False,
                   "error": cascade.redact(e)[:300]})
        raise
    content = str(reply["choices"][0]["message"].get("content") or "")
    _log_call({"time": started, "request_sha256": digest, "served_model": reply.get("model"),
               "ok": True, "reply": content[:REPLY_CHARS]})
    return reply


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default access logging
        pass

    def _send_json(self, code, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.partition("?")[0]
        if path.rstrip("/").endswith("/models"):
            self._send_json(200, {"object": "list", "data": [
                {"id": "gbench-cascade", "object": "model", "owned_by": "gbench"}]})
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self):
        if not self.path.partition("?")[0].rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as e:
            self._send_json(400, {"error": {"message": f"bad request: {cascade.redact(e)}"}})
            return
        try:
            self._send_json(200, judge_chat_completion(body))
        except cascade.UnsupportedRequest as e:
            self._send_json(400, {"error": {"message": str(e), "type": "unsupported_request"}})
        except Exception as e:   # recorded as a harness event by the cascade
            self._send_json(503, {"error": {"message": cascade.redact(e), "type": "judge_outage"}})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not cascade.gemini_keys():
        logger.error("GEMINI_API_KEYS / GEMINI_API_KEY is not set; the judge proxy does not start.")
        return 2
    try:
        relay_openai_base()
    except RelayRefused as e:
        logger.error("%s; the judge proxy does not start.", e)
        return 7
    if not (os.environ.get("GBENCH_JUDGE_MODELS") or "").strip():
        logger.error("GBENCH_JUDGE_MODELS is not set (gbench passes its judge cascade); the judge "
                     "proxy does not start.")
        return 9
    port = int(os.environ.get("GAIA2_JUDGE_PORT", "18790"))
    server = ThreadingHTTPServer((LISTEN_HOST, port), _Handler)
    logger.info("gaia2 Gemini judge proxy listening on %s:%d (judge cascade=%s, rounds=%d)",
                LISTEN_HOST, port, ",".join(cascade._cascade()), cascade._rounds())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
