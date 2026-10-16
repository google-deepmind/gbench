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
# Cascading Gemini judge for the gbench mcp_bench container. Runs INSIDE the
# gbench-mcp-bench image; drop-in for MCP-Bench's own `LLMProvider` (duck-typed).
#
# It preserves upstream MCP-Bench's judge PROCEDURE completely - the same 6 sub-dimension
# prompts, the same 5x randomized-order stability averaging (that all lives in the upstream
# evaluator and is untouched) - and swaps ONLY the model call underneath: instead of a single
# Azure o4-mini, it uses gbench's ESTABLISHED Gemini cascade (the same model list, key rotation,
# rounds and backoff as base.judge_generate_cascade; gbench passes the run's list as
# GBENCH_JUDGE_MODELS), reached through Gemini's OpenAI-compatible endpoint so no google-genai
# dependency is needed in the container. Each request carries what upstream's
# LLMProvider.get_completion sends (llm/provider.py): the model, the system and user messages, and
# the evaluator's max_tokens (upstream's switch to max_completion_tokens is keyed on Azure OpenAI
# model names such as o4-mini and does not apply). No temperature is sent, as upstream sends none.
# The OpenAI client makes one attempt per request (max_retries=0), so a failure moves the cascade
# to its next model at once, as base.judge_generate_cascade does; a model the API answers HTTP 404
# for is not asked again in that call. Each reply is reported to the launcher with the model that
# gave it (CascadeGeminiJudge.answer_hook), so the result records the answering models per task.
#
# Because the judge model is Gemini (not the canonical o4-mini the MCP-Bench leaderboard
# requires), a run scored this way is never leaderboard_comparable - by design, for
# consistency with every other gbench judged suite.
#
# The key and the endpoint. gbench runs model-driven MCP servers in this container (the
# Scientific Computing server parses expressions the model writes, the OSINT server runs commands
# on targets it names), so it never gives the container a real Gemini key: GEMINI_API_KEY and
# GEMINI_API_KEYS hold a per-run placeholder of gbench's Gemini relay (gemini_relay.py, a thread of
# the gbench process that adds the real key), and GEMINI_OPENAI_BASE_URL is the relay's
# OpenAI-compatible URL, <GBENCH_GEMINI_RELAY_URL>/v1beta/openai. While every key is such a
# placeholder, the judge sends its requests only to that URL (relay_openai_base). With a real key,
# GEMINI_OPENAI_BASE_URL (default the Gemini API) must be the Gemini API over https or a loopback
# address (check_gemini_openai_base_url, a copy of gbench's base.check_gemini_openai_base_url), so
# a real key is never sent to the relay's address or anywhere else. A refused endpoint raises
# before a client is built.

import asyncio
import itertools
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mcp_bench.cascade_judge")

# The Gemini model cascade. gbench passes the run's list as GBENCH_JUDGE_MODELS
# (base.gemini_cascade_env()), so the judge calls exactly the models gbench records; this default,
# used only when neither GBENCH_JUDGE_MODELS nor GBENCH_JUDGE_MODEL is set, equals
# base.GEMINI_CASCADE (tests/test_gemini_cascade.py checks every such copy).
_DEFAULT_JUDGE_CASCADE = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                          "gemini-3.5-flash", "gemini-3.1-pro-preview"]
#: The HTTP status of a model the Gemini API does not serve: not asked again in the same call.
NOT_SERVED_STATUS = 404
#: The Gemini API's OpenAI-compatible endpoint: the default of GEMINI_OPENAI_BASE_URL.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_API_HOST = "generativelanguage.googleapis.com"


def check_gemini_openai_base_url(raw: Optional[str]) -> Tuple[Optional[str], str]:
    """(base URL without its trailing slash, "") when `raw` may receive a Gemini key, else
    (None, reason). An empty `raw` is the Gemini API itself (GEMINI_OPENAI_BASE_URL).

    A verbatim copy of gbench's base.check_gemini_openai_base_url (this image cannot import
    gbench): only the Gemini API over https (host generativelanguage.googleapis.com, any port and
    path) or a loopback address over http or https (localhost, 127.0.0.0/8 or ::1) is accepted. A
    URL with a user name or password, whitespace or control characters, a bad port or any other
    host or scheme is refused. The reason never repeats the value, which may hold a credential."""
    import ipaddress
    from urllib.parse import urlsplit
    raw = (raw or "").strip()
    if not raw:
        return GEMINI_OPENAI_BASE_URL.rstrip("/"), ""
    refused = (None, "GEMINI_OPENAI_BASE_URL is not the Gemini API (https://"
                     f"{_GEMINI_API_HOST}/...) or a loopback address, so the Gemini key is not "
                     "sent to it; unset it")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in raw):
        return refused
    try:
        u = urlsplit(raw)
        host = (u.hostname or "").lower()
        u.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        return refused
    if "@" in u.netloc or not host:
        return refused
    if u.scheme == "https" and host == _GEMINI_API_HOST:
        return raw.rstrip("/"), ""
    loopback = host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if loopback and u.scheme in ("http", "https"):
        return raw.rstrip("/"), ""
    return refused


#: The prefix of every gbench Gemini relay placeholder (gemini_relay.PLACEHOLDER_PREFIX), then 48
#: hex digits.
RELAY_PLACEHOLDER_PREFIX = "gbench-gemini-relay-"
#: The relay's origin, http://<address>:<port> (gemini_relay.RELAY_URL_ENV).
RELAY_URL_ENV = "GBENCH_GEMINI_RELAY_URL"
#: The relay's OpenAI-compatible path (gemini_relay.OPENAI_PREFIX).
RELAY_OPENAI_PREFIX = "/v1beta/openai"
_PLACEHOLDER = re.compile(r"^" + re.escape(RELAY_PLACEHOLDER_PREFIX) + r"[0-9a-f]{48}$")


def is_placeholder(key: Optional[str]) -> bool:
    """True when `key` has the shape of a relay placeholder (gemini_relay.is_placeholder)."""
    return bool(_PLACEHOLDER.match((key or "").strip()))


def relay_openai_base(relay: Optional[str], raw: Optional[str]) -> Tuple[Optional[str], str]:
    """(the relay's OpenAI-compatible base URL without a trailing slash, "") when `relay`
    (GBENCH_GEMINI_RELAY_URL) is an http(s) origin and `raw` (GEMINI_OPENAI_BASE_URL) is unset or
    that same URL; else (None, reason). The reason never repeats a value."""
    from urllib.parse import urlsplit
    relay = (relay or "").strip()
    if not relay:
        return None, (f"the Gemini key is a relay placeholder but {RELAY_URL_ENV} is unset, so "
                      "the judge has nowhere to send it")
    try:
        parts = urlsplit(relay)
        parts.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        parts = None
    if (parts is None or parts.scheme not in ("http", "https") or not parts.hostname
            or "@" in parts.netloc or parts.path not in ("", "/") or parts.query
            or parts.fragment or any(c.isspace() or ord(c) < 32 for c in relay)):
        return None, f"{RELAY_URL_ENV} is not an http(s) origin (scheme://host:port)"
    base = f"{parts.scheme}://{parts.netloc}{RELAY_OPENAI_PREFIX}"
    given = (raw or "").strip().rstrip("/")
    if given and given != base:
        return None, (f"GEMINI_OPENAI_BASE_URL is not the relay's URL ({RELAY_URL_ENV} + "
                      f"{RELAY_OPENAI_PREFIX}), so the placeholder key is not sent to it")
    return base, ""


def gemini_openai_base(keys: Optional[List[str]] = None) -> str:
    """The judge's endpoint, with a trailing slash. While every key (`keys`, default _api_keys())
    is a relay placeholder, the relay's URL (relay_openai_base); otherwise GEMINI_OPENAI_BASE_URL
    checked by check_gemini_openai_base_url, which refuses the relay's http address. Raises
    RuntimeError, naming the variables only, when the endpoint is refused."""
    keys = _api_keys() if keys is None else keys
    raw = os.environ.get("GEMINI_OPENAI_BASE_URL")
    if keys and all(is_placeholder(k) for k in keys):
        base, why = relay_openai_base(os.environ.get(RELAY_URL_ENV), raw)
    else:
        base, why = check_gemini_openai_base_url(raw)
    if base is None:
        raise RuntimeError(why + " - the mcp_bench Gemini cascade judge cannot run.")
    return base + "/"


def _cascade() -> List[str]:
    """Mirror base.judge_cascade(): GBENCH_JUDGE_MODELS (csv) > GBENCH_JUDGE_MODEL (single) > default."""
    raw = (os.environ.get("GBENCH_JUDGE_MODELS") or "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    single = (os.environ.get("GBENCH_JUDGE_MODEL") or "").strip()
    if single:
        return [single]
    return list(_DEFAULT_JUDGE_CASCADE)


def _api_keys() -> List[str]:
    """Mirror base.gemini_api_keys(): GEMINI_API_KEYS (comma/semicolon list) > GEMINI_API_KEY."""
    def _split(raw: Optional[str]) -> List[str]:
        return [k.strip() for k in re.split(r"[;,]", raw or "") if k.strip()]

    keys = _split(os.environ.get("GEMINI_API_KEYS")) or _split(os.environ.get("GEMINI_API_KEY"))
    return list(dict.fromkeys(keys))


def _without_keys(text: str, keys: List[str]) -> str:
    """`text` with every API key value replaced, so an error message never carries one."""
    for key in sorted(keys, key=len, reverse=True):
        text = text.replace(key, "[REDACTED]")
    return text


def _rounds() -> int:
    return max(1, int(os.environ.get("GBENCH_JUDGE_CASCADE_ROUNDS", "3")))


def _backoff() -> float:
    return float(os.environ.get("GBENCH_JUDGE_BACKOFF", "1.0"))


class CascadeGeminiJudge:
    """Duck-typed replacement for MCP-Bench's LLMProvider used only as the judge.

    Contract the upstream evaluator depends on (verified against llm/provider.py):
      * ``async get_completion(system_prompt, user_prompt, max_tokens, return_usage=False)``
        returning a ``str`` (or ``(str, usage_dict)`` when ``return_usage=True``);
      * attributes ``deployment_name`` / ``provider_type``;
      * ``clean_and_parse_json(raw)`` (the evaluator parses the judge JSON through it).
    Raises on total outage - matching LLMProvider, which raises; the evaluator's stability
    loop already catches per-iteration failures and skips them. Before it raises, it calls
    ``CascadeGeminiJudge.outage_hook`` when the launcher set one (mcp_bench_run.py counts the
    failed judge calls of each task); after each reply it calls ``answer_hook(model)`` with the
    model that gave it.
    """

    #: Called (no arguments) each time a judge call fails on every model of the cascade.
    outage_hook = None
    #: Called with the answering model's name each time a judge call gets a reply.
    answer_hook = None

    def __init__(self, deployment_name: str = "gemini-cascade") -> None:
        self.deployment_name = deployment_name
        self.provider_type = "openai_compatible"
        self.client = None  # unused: get_completion is overridden
        self._async_clients: Optional[List[Any]] = None
        self._client_cycle = None
        self._keys: List[str] = []
        self._cascade = _cascade()
        self._rounds = _rounds()
        self._backoff = _backoff()

    def _next_client(self):
        """The next client in round-robin order, one per configured key (as
        base._next_judge_client rotates its clients)."""
        if self._async_clients is None:
            from openai import AsyncOpenAI  # container ships `openai`
            self._keys = _api_keys()
            if not self._keys:
                raise RuntimeError("GEMINI_API_KEY is not set - the mcp_bench Gemini cascade judge "
                                   "cannot run.")
            base_url = gemini_openai_base(self._keys)   # refused: raises, no key sent
            # One attempt per request: the cascade moves on to the next model itself.
            self._async_clients = [AsyncOpenAI(api_key=k, base_url=base_url, max_retries=0)
                                   for k in self._keys]
            self._client_cycle = itertools.cycle(self._async_clients)
        return next(self._client_cycle)

    async def get_completion(self, system_prompt: str, user_prompt: str, max_tokens: int,
                             return_usage: bool = False) -> Any:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        last_err: Optional[Exception] = None
        last_not_served: Optional[Exception] = None
        not_served: set = set()
        # Structurally identical to base.judge_generate_cascade: a client (key) per round, every
        # model each round (one attempt each, max_retries=0), a model answered HTTP 404 skipped for
        # the rest of the call; only back off (burst) BETWEEN rounds after a whole-cascade pass
        # fails. The request is upstream's: model, messages, max_tokens (no temperature).
        for rnd in range(self._rounds):
            client = self._next_client()
            for model in self._cascade:
                if model in not_served:
                    continue
                try:
                    resp = await client.chat.completions.create(
                        model=model, messages=messages, max_tokens=max_tokens)
                    content = resp.choices[0].message.content
                    if content is not None and content.strip() != "":
                        hook = CascadeGeminiJudge.answer_hook
                        if hook is not None:
                            hook(model)
                        if return_usage:
                            usage = getattr(resp, "usage", None)
                            usage_dict = {
                                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                            }
                            return content.strip(), usage_dict
                        return content.strip()
                except Exception as e:  # 429/network/timeout/model-missing -> next model
                    if getattr(e, "status_code", None) == NOT_SERVED_STATUS:
                        not_served.add(model)     # not served: not asked again in this call
                        last_not_served = e
                    else:
                        last_err = e
                    logger.debug("mcp_bench judge model %s failed: %s", model,
                                 _without_keys(str(e), self._keys))
            if len(not_served) >= len(self._cascade):
                break
            if rnd < self._rounds - 1:
                await asyncio.sleep(self._backoff * (2 ** rnd) + random.uniform(0, self._backoff))
        hook = CascadeGeminiJudge.outage_hook
        if hook is not None:
            hook()
        served_note = (f"; not served (HTTP 404): {', '.join(sorted(not_served))}"
                       if not_served else "")
        raise RuntimeError("JUDGE_OUTAGE: all Gemini cascade models failed ("
                           + _without_keys(str(last_err or last_not_served), self._keys) + ")"
                           + served_note)

    # Ported verbatim in spirit from LLMProvider.clean_and_parse_json (provider.py:210): strip
    # ```json fences, then json / json_repair. The evaluator calls this on the judge output.
    def clean_and_parse_json(self, raw_json: str) -> Any:
        if raw_json is None:
            return None
        s = raw_json.strip()
        if s.startswith("```"):
            s = s.split("```", 2)
            s = s[1] if len(s) > 1 else raw_json
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
            s = s.strip().rstrip("`").strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            from json_repair import loads as _repair_loads
            return _repair_loads(s)
        except Exception:
            # last resort: extract the outermost object
            import re
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None
