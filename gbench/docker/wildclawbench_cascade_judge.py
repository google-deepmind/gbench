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
# Gemini proxy for the gbench wildclawbench orchestrator. It runs INSIDE the gbench-wildclawbench
# orchestrator image, on one HTTP port the sibling task containers can reach, and is the only thing
# in a wildclawbench run that talks to an external model or search service. Every such call goes to
# the Gemini API (GEMINI_API_KEYS, else GEMINI_API_KEY; a key list is rotated). It also relays the
# agent's calls to the model under test, adding the model endpoint's key (GBENCH_MODEL_API_KEY). The
# keys stay in this process: none is ever written to a task container, a reply, a log line or an
# error message.
#
# It listens on WILDCLAW_JUDGE_HOST (default 127.0.0.1; the runner passes the host the task
# containers use to reach it: 127.0.0.1 with the host network, the docker bridge gateway otherwise),
# never on every interface.
#
# It serves four request kinds:
#
# 1. POST <base>/chat/completions - the task judges. WildClawBench's per-task grading code
#    (`automated_checks` -> `grade()`) builds its OWN judge prompts and calls, VERBATIM, through the
#    stock OpenAI SDK:
#
#        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=os.environ["OPENROUTER_BASE_URL"])
#        client.chat.completions.create(model=os.environ.get("JUDGE_MODEL", "openai/gpt-5.4"), ...)
#
#    The launcher points OPENROUTER_BASE_URL at this proxy. The judge procedure and prompts are
#    untouched; only the model call underneath changes: instead of OpenRouter's `openai/gpt-5.4` it
#    runs gbench's Gemini judge cascade (the model list, rounds and backoff of
#    base.judge_generate_cascade; gbench passes the run's list as GBENCH_JUDGE_MODELS) through
#    Gemini's OpenAI-compatible endpoint. The client's `model` is ignored; every other field
#    (messages, temperature as the task sends it or none when it sends none, max_tokens,
#    response_format, ...) is forwarded unchanged, so the judge call's parameters, structured
#    output and multimodal content pass through. OpenRouter routing fields are dropped. A request
#    that turns thinking off (`thinking: {"type": "disabled"}`, which the judges of
#    06_Safety_Alignment tasks 2, 3 and 6 send through extra_body; also `reasoning_effort: "none"`
#    and OpenRouter's `reasoning: {"effort": "none"}` or `{"enabled": false}`) is sent at the
#    nearest Gemini setting: every cascade model thinks and none can turn it off, so it gets
#    `reasoning_effort: "low"`, Gemini thinking level low, the lowest level every model of the
#    default cascade accepts (THINKING_OFF_LEVEL; generationConfig.thinkingConfig on the grounded
#    path), and the proxy counts it (gemini_usage `thinking_off`). Any other `thinking` /
#    `reasoning` / `reasoning_effort` setting is dropped (the pinned tasks send none). A model the
#    Gemini API answers HTTP 404 for (not served) is not asked again in that request and never
#    makes the request a content refusal.
#
#    The output budget. The tasks' max_tokens (128 to 16384) size the judge's reply (a 128-token
#    cap leaves no room for thinking); every Gemini cascade model thinks and counts its thinking
#    against max_output_tokens, so a judge sent the task's cap unchanged would have less room for
#    its verdict than the task gives it. A chat
#    request that carries max_tokens or max_completion_tokens is sent that value plus
#    GEMINI_THINKING_ALLOWANCE (the same allowance as gbench's gemini_budget module; the grounded
#    path's maxOutputTokens likewise), so the reply keeps the task's room. A reply that still
#    stops there (finish_reason "length") is returned as it arrived and counted; a grader reply
#    that stops there with no text is counted and the cascade asks its next model; a grader
#    request whose cascade ended in an outage after such a reply is counted too
#    (gemini_usage `max_tokens_cut`, `max_tokens_empty`, `outage_after_max_tokens`, by kind:
#    judge, judge_web_search, agent_helper; `thinking_allowance` counts the requests whose cap
#    was raised). The grounded path returns Gemini's own finishReason, mapped to the OpenAI value
#    (MAX_TOKENS -> "length", STOP -> "stop", a block reason -> "content_filter") and kept as
#    `native_finish_reason`.
#
#    A judge request that carries OpenRouter's server-side web search tool
#    ({"type": "openrouter:web_search", ...}; 04_Search_Retrieval_task_1's path-verification judge)
#    is answered by Gemini with Google Search grounding instead: generateContent with the
#    google_search tool over the same cascade, the reply returned as an OpenAI chat completion (the
#    grounding sources as url_citation annotations). Any other non-function tool type, or web search
#    mixed with function tools, is rejected with HTTP 400 (the pinned tasks send neither).
#
#    The same path serves the OpenRouter helper model 25 task prompts offer the agent. A request
#    whose Authorization carries the run's grader placeholder (WILDCLAW_GRADER_KEY, which the
#    launcher gives each task's grade()) is the grader's: any failure to serve it is a harness
#    event of the task ("judge" / "judge_web_search", with the time the request arrived and
#    whether the Gemini service failed it or the request itself was refused), and its start and
#    end go to grader_calls.jsonl. Any other request is the agent's: it gets the first reply a
#    cascade model gives (an empty one included), a refusal of its own content by the Gemini API
#    (HTTP 400, 413 or 422 from every model the API serves, on every attempt) is answered with
#    HTTP 400 and is no event, and any other failure of every attempt (HTTP 5xx, 429, 401, 403, a
#    timeout, a network error, an unreadable reply, or no model of the cascade served) is an
#    "agent_helper" harness event. Every answered chat request is written to gemini_answers.jsonl
#    with the model that answered, the time it arrived and the requesting container's address, so
#    the launcher records the answering models per task.
#
# 2. GET /res/v1/web/search - the OpenClaw agent's `web_search` tool. The image's OpenClaw is
#    configured with its Brave web-search provider; the launcher rewrites the Brave API origin in the
#    task container's OpenClaw bundle to this server, so the tool keeps the name, description,
#    parameters and result shape the image ships while every query is answered here from Gemini
#    Google Search grounding (gbench's search_tool.search_with_fallback: the grounding model cascade,
#    its global rate limiter and cascade rounds). The reply is the Brave response shape OpenClaw
#    parses ({"web": {"results": [{title, url, description}]}}). Each lookup is a search-engine
#    lookup (search_tool's lookup with a deadline, 2026-10-12): the results are the cited sources'
#    titles and real URLs, every description is empty, and no text a Gemini model wrote reaches
#    the agent; an answer citing no source is not an answer. What each lookup did (models asked,
#    links resolved) is summed in gemini_usage.json (agent_web_search_engine) and kept with its
#    gemini_answers.jsonl record (`search`). A query no grounding model could
#    serve returns HTTP 502, so OpenClaw reports a failed search to the agent (never "no results").
#    OpenClaw aborts a web_search after AGENT_WEB_SEARCH_TIMEOUT_S (60 s, which the launcher sets;
#    OpenClaw's own default is 30 s), so a lookup gets a deadline of WEB_SEARCH_DEADLINE_S (57 s)
#    from when the request arrived: one no grounding model
#    answered by then also returns HTTP 502 and is an "agent_web_search" harness event, and an
#    answer whose reply could not be written or came after OpenClaw's timeout is one too (the
#    agent never saw it); only a reply delivered in time is recorded as answered.
#    The Brave Search API is never contacted.
#
# 3. GET/POST /model/v1/<path> - the OpenClaw agent's calls to the model under test. The task
#    container's my_api.json points OpenClaw at <this server>/model/v1 with a placeholder apiKey;
#    the request is forwarded to GBENCH_MODEL_BASE_URL/<path> with `Authorization: Bearer
#    <GBENCH_MODEL_API_KEY>` in place of the placeholder, and the reply (streamed or not) is passed
#    back unchanged, status included. A redirect is never followed (it is passed back as is), so
#    the key never goes to another address. No relay-side timeout: OpenClaw's own request timeout
#    and the task timeout apply, as with a direct connection. An unreachable model endpoint is an
#    HTTP 502. Each chat request relayed is written to model_requests.jsonl (MODEL_REQUESTS: the
#    per-turn max_tokens OpenClaw sent, how the reply ended, whether answer or reasoning text came,
#    the token usage when the endpoint reported it), read from the bytes as they pass unchanged.
#
# 4. GET <anything else> - a health reply ({"status": "ok"}) and a /models list, for probes.
#
# Every Gemini call goes to GEMINI_OPENAI_BASE_URL (default the Gemini API), which must be the
# Gemini API over https or a loopback address (check_gemini_openai_base_url, a copy of gbench's
# base.check_gemini_openai_base_url): otherwise the proxy does not start and no key is sent.
#
# Every Gemini call is counted by kind and serving model in $WILDCLAW_WORKDIR/gemini_usage.json,
# which the launcher copies into the run summary. Every failure of a service a task depends on (a
# grader judge call not served, an agent helper-model call failed on the service side, an agent
# web search no grounding model served in time or whose answer did not reach the agent in time,
# the model endpoint unreachable, an HTTP 5xx or 429 from it, or its reply cut off) is written to $WILDCLAW_WORKDIR/harness_events.jsonl with its time and the
# requesting container's address (a chat request's event also with the time it arrived). The
# launcher attributes an event to the task whose run window holds its time and whose container
# made the request; a grader event counts only when its request arrived during a grade() exec of
# that task (see the launcher's grading isolation). After a model endpoint failure, the first
# relay success for the same container is written as model_endpoint_recovered: a failure the
# agent's retry got past is not a harness error.
#
# Dependency-light on purpose: stdlib only (http.server + urllib) plus gbench's stdlib-only
# search_tool.py, which the gbench runner mounts read-only at $WILDCLAW_SEARCH_TOOL_DIR.

import importlib.util
import itertools
import json
import logging
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("wildclawbench.cascade_judge")

# The Gemini model cascade. gbench passes the run's list as GBENCH_JUDGE_MODELS
# (base.gemini_cascade_env()), so the proxy calls exactly the models gbench records; this default,
# used only when neither GBENCH_JUDGE_MODELS nor GBENCH_JUDGE_MODEL is set, equals
# base.GEMINI_CASCADE (tests/test_gemini_cascade.py checks every such copy).
_DEFAULT_JUDGE_CASCADE = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                          "gemini-3.5-flash", "gemini-3.1-pro-preview"]

#: OpenRouter's server-side web search tool, the one judge tool the pinned tasks send
#: (04_Search_Retrieval_task_1). Served by Gemini Google Search grounding.
WEB_SEARCH_TOOL_TYPES = frozenset({"openrouter:web_search"})

#: Where the Brave web search request of the task container's OpenClaw arrives.
BRAVE_WEB_SEARCH_PATH = "/res/v1/web/search"

#: The model relay's path prefix: <prefix>/<path> is forwarded to GBENCH_MODEL_BASE_URL/<path>.
MODEL_RELAY_PATH = "/model/v1"

REDACTED = "[REDACTED]"

#: The Gemini thinking level a request that turns thinking off is sent at. OpenRouter's
#: `thinking: {"type": "disabled"}` (06_Safety_Alignment tasks 2, 3 and 6) has no Gemini
#: counterpart: gbench's cascade models all think and none can turn it off. gemini-3.8-flash,
#: gemini-3.7-flash and gemini-3.1-pro-preview offer low / medium / high, gemini-3.6-flash and
#: gemini-3.5-flash also minimal, so "low" is the lowest level every model of the default cascade
#: accepts and the same request goes to each of them (the level gbench's cimemories labeler sends
#: for upstream's reasoning_effort "minimal"). On the OpenAI-compatible endpoint it is
#: `reasoning_effort: "low"`, which Gemini maps to thinking level low.
THINKING_OFF_LEVEL = "low"
#: The `reasoning_effort` / OpenRouter `reasoning.effort` value that turns thinking off.
_THINKING_OFF_EFFORTS = frozenset({"none"})

#: Room added to a chat request's max_tokens / max_completion_tokens for the Gemini model's
#: thinking, which counts against max_output_tokens (the tasks' caps were written for a reply
#: with no thinking in it). A copy of gbench's gemini_budget.GEMINI_THINKING_ALLOWANCE (this image
#: cannot import gbench); tests/test_gemini_thinking_allowance.py checks they are equal.
GEMINI_THINKING_ALLOWANCE = 8192
#: The request fields that carry an output cap (OpenAI chat completions).
_CAP_FIELDS = ("max_tokens", "max_completion_tokens")


def with_thinking_allowance(body):
    """(a copy of `body` with GEMINI_THINKING_ALLOWANCE added to each output cap it sends,
    whether any cap was raised). A cap that is not a positive integer is left as it is."""
    out, raised = dict(body), False
    for field in _CAP_FIELDS:
        value = out.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        out[field] = value + GEMINI_THINKING_ALLOWANCE
        raised = True
    return out, raised


#: Gemini finishReason -> OpenAI finish_reason for the grounded path's reply.
_FINISH_REASONS = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
                   "RECITATION": "content_filter", "BLOCKLIST": "content_filter",
                   "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter",
                   "IMAGE_SAFETY": "content_filter"}


def openai_finish_reason(native):
    """The OpenAI finish_reason for a Gemini finishReason ("stop" when Gemini gave none; an
    unknown reason lower-cased)."""
    if not native:
        return "stop"
    return _FINISH_REASONS.get(str(native), str(native).lower())


#: What the output limit did to this thread's last chat request: `raised` (a cap was raised by
#: GEMINI_THINKING_ALLOWANCE), `cut` (the reply returned stopped at it), `empty` (replies of the
#: cascade that stopped at it with no text). Each request of the proxy runs in its own thread.
_LIMIT = threading.local()


def _reset_limit_state(raised=False):
    _LIMIT.state = {"raised": bool(raised), "cut": False, "empty": 0}


def last_limit_state():
    """The output-limit record of this thread's last chat request (see _LIMIT)."""
    return dict(getattr(_LIMIT, "state", None) or {"raised": False, "cut": False, "empty": 0})


def _note_reply_finish(finish_reason, has_text):
    """Record a cascade reply that stopped at the output limit (finish_reason "length")."""
    if finish_reason != "length":
        return
    state = getattr(_LIMIT, "state", None)
    if state is None:
        _reset_limit_state()
        state = _LIMIT.state
    if has_text:
        state["cut"] = True
    else:
        state["empty"] += 1


def thinking_off_requested(body):
    """True when a chat request turns thinking off: `thinking: {"type": "disabled"}`
    (OpenRouter/Anthropic), `reasoning_effort: "none"` (OpenAI) or `reasoning: {"effort": "none"}`
    / `{"enabled": false}` (OpenRouter)."""
    if not isinstance(body, dict):
        return False
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and str(thinking.get("type") or "").lower() == "disabled":
        return True
    if str(body.get("reasoning_effort") or "").lower() in _THINKING_OFF_EFFORTS:
        return True
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            return True
        if str(reasoning.get("effort") or "").lower() in _THINKING_OFF_EFFORTS:
            return True
    return False


class UnsupportedRequest(ValueError):
    """A judge request this proxy does not serve (answered with HTTP 400)."""


class RequestRejected(RuntimeError):
    """Every attempt on a model the Gemini API serves was refused for the request's own content
    (HTTP 400, 413 or 422): not a service failure. A model answered HTTP 404 (not served) is left
    out of that rule."""


#: Gemini API statuses that refuse a request for its own content.
REJECTION_STATUSES = frozenset({400, 413, 422})
#: The Gemini API status of a model it does not serve (base.judge_generate_cascade's rule): that
#: model is not asked again in the same request and says nothing about the request's content.
NOT_SERVED_STATUS = 404
_NOT_SERVED_WARNED = set()


def _cascade():
    """Mirror base.judge_cascade(): GBENCH_JUDGE_MODELS (csv) > GBENCH_JUDGE_MODEL (single) > default."""
    raw = (os.environ.get("GBENCH_JUDGE_MODELS") or "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    single = (os.environ.get("GBENCH_JUDGE_MODEL") or "").strip()
    if single:
        return [single]
    return list(_DEFAULT_JUDGE_CASCADE)


def _rounds():
    return max(1, int(os.environ.get("GBENCH_JUDGE_CASCADE_ROUNDS", "3")))


def _backoff():
    return float(os.environ.get("GBENCH_JUDGE_BACKOFF", "1.0"))


#: The Gemini API's OpenAI-compatible endpoint: the default of GEMINI_OPENAI_BASE_URL.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_API_HOST = "generativelanguage.googleapis.com"


class GeminiEndpointRefused(RuntimeError):
    """GEMINI_OPENAI_BASE_URL is neither the Gemini API nor a loopback address: no key is sent."""


def check_gemini_openai_base_url(raw):
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


def _gemini_base():
    """The Gemini OpenAI-compatible base every Gemini call of this proxy goes to. Raises
    GeminiEndpointRefused (before any key is sent) when GEMINI_OPENAI_BASE_URL is not accepted by
    check_gemini_openai_base_url."""
    base, why = check_gemini_openai_base_url(os.environ.get("GEMINI_OPENAI_BASE_URL"))
    if base is None:
        raise GeminiEndpointRefused(why)
    return base


def _gemini_native_base():
    """The generateContent base: GEMINI_OPENAI_BASE_URL without its /openai suffix."""
    base = _gemini_base()
    return base[:-len("/openai")] if base.endswith("/openai") else base


# --------------------------------------------------------------------------- #
# Gemini keys: GEMINI_API_KEYS wins over GEMINI_API_KEY; either may be a comma/semicolon list.
# --------------------------------------------------------------------------- #
_KEY_LOCK = threading.Lock()
_KEY_STATE = {"raw": None, "cycle": None}


def gemini_keys():
    """The configured Gemini keys, in order, de-duplicated (base.gemini_api_keys' rule)."""
    def _split(raw):
        return [k.strip() for k in re.split(r"[;,]", raw or "") if k.strip()]
    keys = _split(os.environ.get("GEMINI_API_KEYS")) or _split(os.environ.get("GEMINI_API_KEY"))
    return list(dict.fromkeys(keys))


def next_gemini_key():
    """The next key of the configured list (round-robin), or "" when none is set."""
    keys = gemini_keys()
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    with _KEY_LOCK:
        raw = ",".join(keys)
        if _KEY_STATE["raw"] != raw:
            _KEY_STATE["raw"], _KEY_STATE["cycle"] = raw, itertools.cycle(keys)
        return next(_KEY_STATE["cycle"])


def redact(text):
    """`text` with every configured Gemini key, the model endpoint's key (and any `key=` URL
    parameter) replaced."""
    text = str(text)
    raw_values = [os.environ.get("GEMINI_API_KEYS", ""), os.environ.get("GEMINI_API_KEY", ""),
                  os.environ.get("GBENCH_MODEL_API_KEY", "")]
    for secret in sorted({*gemini_keys(), *(v.strip() for v in raw_values if v.strip())},
                         key=len, reverse=True):
        for form in (secret, urllib.parse.quote(secret, safe="")):
            if form:
                text = text.replace(form, REDACTED)
    return re.sub(r"(?i)\bkey=[^&\s\"']+", "key=" + REDACTED, text)


# --------------------------------------------------------------------------- #
# Usage tally (-> $WILDCLAW_WORKDIR/gemini_usage.json, copied into the run summary).
# --------------------------------------------------------------------------- #
_USAGE_LOCK = threading.Lock()
#: Calls per section and serving model ("failed" when none served it); `thinking_off` counts the
#: chat requests that turned thinking off, each sent at THINKING_OFF_LEVEL, by kind (judge,
#: judge_web_search for a grader's request, agent_helper for an agent's); `thinking_allowance`
#: the chat requests whose output cap was raised by GEMINI_THINKING_ALLOWANCE, `max_tokens_cut`
#: the replies returned that still stopped at the raised cap, `max_tokens_empty` the replies that
#: stopped there with no text (the cascade asked its next model), `outage_after_max_tokens` the
#: requests no model served after such a reply, each by the same kinds.
#: `agent_web_search_engine` sums what the agent's web searches did (search_tool.last_lookup:
#: each is a search-engine lookup, sources only): lookups, answered, models_asked,
#: answers_without_source, links_resolved, links_unresolved, links_direct.
_USAGE = {"judge": {}, "judge_web_search": {}, "agent_web_search": {}, "thinking_off": {},
          "thinking_allowance": {}, "max_tokens_cut": {}, "max_tokens_empty": {},
          "outage_after_max_tokens": {}, "agent_web_search_engine": {}}


def usage_snapshot():
    with _USAGE_LOCK:
        return json.loads(json.dumps(_USAGE))


def reset_usage():
    with _USAGE_LOCK:
        for section in _USAGE.values():
            section.clear()


def _count(section, key, n=1):
    with _USAGE_LOCK:
        bucket = _USAGE[section]
        bucket[key] = bucket.get(key, 0) + n
        snapshot = json.dumps(_USAGE, indent=2, sort_keys=True)
    workdir = os.environ.get("WILDCLAW_WORKDIR", "").strip()
    if not workdir:
        return
    path = os.path.join(workdir, "gemini_usage.json")
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(snapshot)
        os.replace(tmp, path)
    except OSError as e:  # the tally is a record, never a reason to fail a judge call
        logger.warning("wildclawbench: could not write %s: %s", path, e)


# --------------------------------------------------------------------------- #
# Harness events (-> $WILDCLAW_WORKDIR/harness_events.jsonl, read by the launcher).
# --------------------------------------------------------------------------- #
#: One JSON line per failure of a service a task depends on, never of the model under test, with
#: the address of the client that made the request: the model endpoint behind the relay
#: ("model_endpoint": unreachable, an HTTP 5xx or 429, or a reply cut off mid-stream), a judge
#: request of a task's grade() no cascade model served or the proxy could not serve ("judge",
#: "judge_web_search"), an agent's helper-model request that failed on the service side
#: ("agent_helper"), or an agent web search no grounding model served within its deadline, or
#: whose answer could not be written or came after OpenClaw's web_search timeout
#: ("agent_web_search"). A chat request's event (judge, judge_web_search, agent_helper) and a web
#: search's delivery event also carry the time the request arrived (`start`) and whether the Gemini service failed it (`service_failure`: every model
#: failed or none was served) or the request was refused (a request the proxy does not serve, a bad
#: request, the Gemini API refusing its content). After a model_endpoint failure, the first relay
#: success for the same client is written as "model_endpoint_recovered". The launcher attributes
#: an event to the task whose run window holds its time and whose container made the request, and
#: a grader event only when its request arrived during a grade() exec of that task that ran after
#: the processes the agent left had been stopped (or whose stop failed for a reason the agent
#: cannot cause); those tasks are harness errors.
HARNESS_EVENTS = "harness_events.jsonl"
HARNESS_EVENT_KINDS = ("model_endpoint", "judge", "judge_web_search", "agent_helper",
                       "agent_web_search", "model_endpoint_recovered")
#: One JSON line per judge request of a task's grade() ({start, end, client, ok}): the launcher
#: uses them only to say whether a grade() that hit its 120 s cap had a judge call in flight
#: (grading_timeouts with_judge_call / without_judge_call); both are harness errors.
GRADER_CALLS = "grader_calls.jsonl"
#: One JSON line per chat request the Gemini API answered, and per agent web search whose answer
#: reached OpenClaw within its web_search timeout ({time, start, kind, model, client, grader}): the
#: launcher records the answering models per task.
GEMINI_ANSWERS = "gemini_answers.jsonl"
#: The environment variable holding the placeholder key a task's grade() sends (the launcher sets
#: OPENROUTER_API_KEY to it in the grading `docker exec`); requests with any other key are the
#: agent's. The entrypoint draws a new random value per run.
GRADER_KEY_ENV = "WILDCLAW_GRADER_KEY"
_EVENT_LOCK = threading.Lock()


def _append_workdir_line(name, record):
    workdir = os.environ.get("WILDCLAW_WORKDIR", "").strip()
    if not workdir:
        return
    path = os.path.join(workdir, name)
    with _EVENT_LOCK:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:  # logged; the run's summary then lacks this record
            logger.warning("wildclawbench: could not write %s: %s", path, e)


def record_harness_event(kind, detail="", client=None, start=None, service_failure=None):
    """Append one harness event (time, kind, a short redacted detail, the client's address, and
    for a chat request the time it arrived, `start`, and `service_failure`: True when the Gemini
    service failed it, False when the request was refused) to HARNESS_EVENTS. The launcher counts
    a grader request's event only when `start` falls in a grade() exec of the task."""
    record = {"time": time.time(), "kind": kind, "detail": redact(detail)[:300], "client": client}
    if start is not None:
        record["start"] = start
    if service_failure is not None:
        record["service_failure"] = bool(service_failure)
    _append_workdir_line(HARNESS_EVENTS, record)


def record_answer(kind, model, client=None, start=None, grader=False, search=None):
    """Append one answered request (its kind, the Gemini model that answered, the client's
    address, when it arrived, whether it carried the grader key) to GEMINI_ANSWERS; an agent web
    search's record also carries `search`, what its lookup did (search_tool.last_lookup)."""
    record = {"time": time.time(), "start": start, "kind": kind, "model": model,
              "client": client, "grader": bool(grader)}
    if search is not None:
        record["search"] = search
    _append_workdir_line(GEMINI_ANSWERS, record)


#: The model that answered the last cascade call of this thread (each request of the proxy is
#: served in its own thread).
_ANSWERED = threading.local()


def last_answering_model():
    """The Gemini model that answered this thread's last successful cascade call, or None."""
    return getattr(_ANSWERED, "model", None)


def record_grader_call(start, end, client, ok):
    """Append one judge request of a task's grade() to GRADER_CALLS."""
    _append_workdir_line(GRADER_CALLS, {"start": start, "end": end, "client": client,
                                        "ok": bool(ok)})


def is_grader_key(authorization):
    """True when an Authorization header carries the run's grader placeholder key."""
    key = os.environ.get(GRADER_KEY_ENV, "").strip()
    value = str(authorization or "").strip()
    return bool(key) and value in (f"Bearer {key}", key)


# --------------------------------------------------------------------------- #
# Judge cascade.
# --------------------------------------------------------------------------- #
def _post_json(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_cascade(call_one, section, record=True, base_check=None):
    """Try every cascade model each round; back off (burst) only BETWEEN rounds after a whole pass
    fails. `call_one(model)` returns the reply or raises. Raises RuntimeError (JUDGE_OUTAGE) when
    every model failed every round, matching base.judge_generate_cascade, or RequestRejected when
    every attempt on a model the API serves was refused for the request's content. A model the
    API answers HTTP 404 for is not asked again in this call. With `record`, any failure (no key,
    a refused GEMINI_OPENAI_BASE_URL, the outage, a rejection) is recorded as a harness event of
    `section`; without it the caller decides. `base_check` (default _gemini_base) runs before any
    key is sent and raises when the endpoint is refused; a server with another endpoint (gaia2's
    relay) passes its own check here instead of replacing this module's."""
    try:
        return _cascade_attempts(call_one, section, base_check)
    except Exception as e:
        if record:
            record_harness_event(section, str(e),
                                 service_failure=not isinstance(e, RequestRejected))
        raise


def _warn_not_served(model):
    if model not in _NOT_SERVED_WARNED:
        _NOT_SERVED_WARNED.add(model)
        logger.warning("wildclawbench: the Gemini API answered HTTP 404 for model %s (not served "
                       "for this key); the cascade skips it. Check GBENCH_JUDGE_MODELS.", model)


def _cascade_attempts(call_one, section, base_check=None):
    if not gemini_keys():
        raise RuntimeError("GEMINI_API_KEYS / GEMINI_API_KEY is not set - the wildclawbench Gemini "
                           "cascade judge cannot run.")
    (base_check or _gemini_base)()   # a refused endpoint fails here, before any key is sent
    cascade, rounds, backoff = _cascade(), _rounds(), _backoff()
    _ANSWERED.model = None
    not_served = set()
    last_err = None          # the last failure of a model the API serves
    last_not_served = None
    rejected = False         # a model the API serves refused the request's content
    other_failure = False    # any other failure of a model the API serves
    for rnd in range(rounds):
        for model in cascade:
            if model in not_served:
                continue
            try:
                reply = call_one(model)
                _count(section, model)
                _ANSWERED.model = model
                return reply
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                err = RuntimeError(f"{model} HTTP {e.code}: {redact(detail)}")
                if e.code == NOT_SERVED_STATUS:
                    not_served.add(model)
                    last_not_served = err
                    _warn_not_served(model)
                else:
                    last_err = err
                    if e.code in REJECTION_STATUSES:
                        rejected = True
                    else:
                        other_failure = True
            except Exception as e:  # network/timeout/parse/empty -> next model
                last_err = RuntimeError(f"{model}: {redact(e)}")
                other_failure = True
            logger.debug("wildclawbench judge model %s failed: %s", model,
                         last_err if model not in not_served else last_not_served)
        if rejected and not other_failure:
            break   # every model the API serves refused the request: another round cannot help
        if len(not_served) >= len(cascade):
            break   # no model of the cascade is served
        if rnd < rounds - 1:
            time.sleep(backoff * (2 ** rnd) + random.uniform(0, backoff))
    _count(section, "failed")
    if rejected and not other_failure:
        raise RequestRejected(f"the Gemini API refused the request ({last_err})")
    served_note = (f"; not served (HTTP 404): {', '.join(sorted(not_served))}"
                   if not_served else "")
    raise RuntimeError(f"JUDGE_OUTAGE: all Gemini cascade models failed "
                       f"({last_err or last_not_served}){served_note}")


def _request_timeout():
    # Per-request timeout must stay well under upstream's 120s docker-exec grading cap
    # (src/utils/grading.py wraps the WHOLE grade() - all its judge calls - in timeout=120), or a
    # single hung request kills grade() and scores the task 0 with no fallback recorded.
    return float(os.environ.get("WILDCLAW_JUDGE_REQUEST_TIMEOUT_S", "40"))


def _split_tools(body):
    """(function tools, web search requested) for a judge request; rejects what is not served."""
    tools = body.get("tools")
    if tools is None:
        return [], False
    if not isinstance(tools, list):
        raise UnsupportedRequest("`tools` must be a list")
    functions, web, other = [], False, []
    for t in tools:
        kind = t.get("type") if isinstance(t, dict) else None
        if kind == "function":
            functions.append(t)
        elif kind in WEB_SEARCH_TOOL_TYPES:
            web = True
        else:
            other.append(kind)
    if other:
        raise UnsupportedRequest(f"unsupported judge tool type(s) {other}; the proxy serves "
                                 f"function tools and {sorted(WEB_SEARCH_TOOL_TYPES)}")
    if web and functions:
        raise UnsupportedRequest("a web search judge request with function tools is not served")
    return functions, web


def cascade_chat_completion(body, record=True, require_content=True):
    """Run one judge request through the Gemini cascade. `body` is the client's parsed request.

    Returns the OpenAI ChatCompletion dict of the first non-empty reply (with `require_content`
    False, of the first reply: an agent's helper-model request gets what the model answered, an
    empty reply included). A request with OpenRouter's web search tool goes to Gemini Google Search
    grounding (grounded_chat_completion); every other request goes to Gemini's OpenAI-compatible
    chat endpoint. `record`: see _run_cascade.
    """
    functions, web = _split_tools(body)
    if web:
        return grounded_chat_completion(body, record=record, require_content=require_content)

    # Forward the client's request verbatim (temperature as it sends it, none when it sends none),
    # overriding only the model (per cascade attempt). Drop provider-specific / OpenRouter-only
    # fields Gemini's OpenAI-compat endpoint rejects; a request that turns thinking off (the
    # 06_Safety_Alignment judges' extra_body `thinking: {"type": "disabled"}`) is sent at the
    # lowest thinking level every default cascade model accepts (THINKING_OFF_LEVEL), any other
    # thinking/reasoning setting is dropped.
    _DROP = ("model", "stream", "provider", "transforms", "route", "models",
             "thinking", "reasoning", "reasoning_effort")
    base_payload, raised = with_thinking_allowance(
        {k: v for k, v in body.items() if k not in _DROP})
    _reset_limit_state(raised)
    if functions:
        base_payload["tools"] = functions
    if thinking_off_requested(body):
        base_payload["reasoning_effort"] = THINKING_OFF_LEVEL
    timeout = _request_timeout()

    def _one(model):
        parsed = _post_json(_gemini_base() + "/chat/completions", dict(base_payload, model=model),
                            {"Authorization": f"Bearer {next_gemini_key()}"}, timeout)
        choices = parsed.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        has_text = content is not None and str(content).strip() != ""
        if choices:
            _note_reply_finish(choices[0].get("finish_reason"), has_text)
        if not require_content and choices:
            return parsed
        if not has_text:
            raise RuntimeError("empty content")
        return parsed

    return _run_cascade(_one, "judge", record=record)


def _gemini_parts(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append({"text": str(p.get("text") or "")})
            else:
                raise UnsupportedRequest("a web search judge request may carry text content only")
        return parts
    raise UnsupportedRequest("unsupported message content")


def gemini_grounded_request(body):
    """The generateContent request for an OpenAI chat request with web search: system messages ->
    systemInstruction, user/assistant -> user/model turns, the google_search tool, the request's
    temperature (none when it sends none), max_tokens / max_completion_tokens plus
    GEMINI_THINKING_ALLOWANCE -> maxOutputTokens, and a request that turns thinking off -> thinkingConfig.thinkingLevel
    THINKING_OFF_LEVEL."""
    system, contents = [], []
    for m in body.get("messages") or []:
        role = (m or {}).get("role")
        parts = _gemini_parts((m or {}).get("content"))
        if role in ("system", "developer"):
            system.extend(parts)
        elif role in ("user", "assistant"):
            contents.append({"role": "user" if role == "user" else "model", "parts": parts})
        else:
            raise UnsupportedRequest(f"message role {role!r} is not served with web search")
    if not contents:
        raise UnsupportedRequest("a web search judge request needs a user message")
    request = {"contents": contents, "tools": [{"google_search": {}}]}
    if system:
        request["systemInstruction"] = {"parts": system}
    generation = {}
    if body.get("temperature") is not None:
        generation["temperature"] = body["temperature"]
    capped, _raised = with_thinking_allowance(body)
    max_tokens = capped.get("max_completion_tokens") or capped.get("max_tokens")
    if max_tokens:
        generation["maxOutputTokens"] = int(max_tokens)
    if thinking_off_requested(body):
        generation["thinkingConfig"] = {"thinkingLevel": THINKING_OFF_LEVEL}
    if generation:
        request["generationConfig"] = generation
    return request


def openai_completion_from_gemini(data, model):
    """An OpenAI ChatCompletion dict from a grounded generateContent reply (text, the grounding
    sources as url_citation annotations, and Gemini's finishReason as the OpenAI finish_reason,
    kept as native_finish_reason)."""
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or []
                   if isinstance(p, dict) and not p.get("thought"))
    annotations = []
    for chunk in (cand.get("groundingMetadata") or {}).get("groundingChunks") or []:
        web = (chunk or {}).get("web") or {}
        if web.get("uri"):
            annotations.append({"type": "url_citation",
                                "url_citation": {"url": web["uri"], "title": web.get("title") or ""}})
    usage = data.get("usageMetadata") or {}
    native = cand.get("finishReason")
    return {
        "id": "gbench-grounded-" + format(random.getrandbits(48), "012x"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": openai_finish_reason(native),
                     "native_finish_reason": native,
                     "message": {"role": "assistant", "content": text,
                                 "annotations": annotations}}],
        "usage": {"prompt_tokens": usage.get("promptTokenCount", 0),
                  "completion_tokens": usage.get("candidatesTokenCount", 0),
                  "total_tokens": usage.get("totalTokenCount", 0)},
    }


def grounded_chat_completion(body, record=True, require_content=True):
    """A judge request with web search, answered by Gemini with Google Search grounding over the
    judge cascade (generateContent + the google_search tool)."""
    request = gemini_grounded_request(body)
    _reset_limit_state(with_thinking_allowance(body)[1])
    timeout = _request_timeout()

    def _one(model):
        data = _post_json(f"{_gemini_native_base()}/models/{model}:generateContent", request,
                          {"x-goog-api-key": next_gemini_key()}, timeout)
        reply = openai_completion_from_gemini(data, model)
        choice = reply["choices"][0]
        has_text = bool(choice["message"]["content"].strip())
        _note_reply_finish(choice["finish_reason"], has_text)
        if require_content and not has_text:
            raise RuntimeError("empty content")
        return reply

    return _run_cascade(_one, "judge_web_search", record=record)


def wants_web_search(body):
    """True when a judge request asks for OpenRouter's web search tool (without raising)."""
    tools = body.get("tools") if isinstance(body, dict) else None
    return isinstance(tools, list) and any(
        isinstance(t, dict) and t.get("type") in WEB_SEARCH_TOOL_TYPES for t in tools)


# --------------------------------------------------------------------------- #
# The OpenClaw agent's web_search: Brave response shape, answered by Gemini grounding.
# --------------------------------------------------------------------------- #
_SEARCH_TOOL = {"module": None}


def search_tool():
    """gbench's search_tool.py, loaded from $WILDCLAW_SEARCH_TOOL_DIR (mounted by the runner)."""
    if _SEARCH_TOOL["module"] is None:
        path = os.path.join(os.environ.get("WILDCLAW_SEARCH_TOOL_DIR", "/app/gbench"),
                            "search_tool.py")
        if not os.path.isfile(path):
            raise RuntimeError(f"gbench search_tool.py not found at {path} (the gbench runner "
                               f"mounts it read-only; set WILDCLAW_SEARCH_TOOL_DIR)")
        spec = importlib.util.spec_from_file_location("search_tool", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SEARCH_TOOL["module"] = module
    return _SEARCH_TOOL["module"]


#: Brave query parameters Gemini grounding has no counterpart for; accepted, not applied.
UNAPPLIED_BRAVE_FILTERS = ("country", "search_lang", "ui_lang", "freshness")

#: OpenClaw's own web_search timeout (upstream): the OpenClaw 2026.3.11 bundle of the pinned task
#: image resolves tools.web.search.timeoutSeconds with a default of 30 s (createWebSearchTool:
#: resolveTimeoutSeconds(search?.timeoutSeconds, 30)), and the image's openclaw.json sets none.
#: Upstream's search provider is Brave, which answers in about a second.
UPSTREAM_WEB_SEARCH_TIMEOUT_S = 30
#: How long the task container's OpenClaw waits for one web_search reply in a gbench run: the
#: launcher writes this to tools.web.search.timeoutSeconds in every task container
#: (wildclawbench_run imports it from here, so the timeout and WEB_SEARCH_DEADLINE_S cannot drift).
#: A recorded deviation from upstream's 30 s: Gemini grounding needs longer than Brave. 2026-10-13
#: A/B (70 recorded agent queries x 3 repeats, search_tool's search-engine mode): answered in time
#: 65.2% at width 2 with a 27 s deadline, 61.9% at width 3, 60.5% at width 4, and 94.8% (Wilson
#: 95% 90.9-97.1%) at width 4 with a 57 s deadline (answered: median 20.2 s, p90 38.4 s). A lookup
#: still running at this timeout is aborted on the agent's side (the tool result reads "This
#: operation was aborted"), whatever this proxy answers later. An integer (OpenClaw's config
#: schema takes a positive integer).
AGENT_WEB_SEARCH_TIMEOUT_S = 60
#: The deadline of one agent web search here (search_with_fallback's `deadline`, from when the
#: request arrived): short enough that the reply, the results or HTTP 502 with an agent_web_search
#: harness event when no grounding model answered in time, reaches OpenClaw before its own timeout.
WEB_SEARCH_DEADLINE_S = AGENT_WEB_SEARCH_TIMEOUT_S - 3.0


def brave_search_reply(query_string, client=None, started=None):
    """(HTTP status, JSON body) for one Brave web search request (its query string); `client` is
    the requesting container's address, recorded with a harness event; `started` is when the
    request arrived (time.monotonic(); default now). The lookup gets WEB_SEARCH_DEADLINE_S from
    `started`: a lookup no grounding model answered by then is a failed search (HTTP 502 and an
    agent_web_search harness event). The model that answered a successful lookup is left in
    last_search_model() for the caller, which records the answer once the reply is delivered."""
    _ANSWERED.search_model = None
    _ANSWERED.search_lookup = None
    started = time.monotonic() if started is None else started
    params = urllib.parse.parse_qs(query_string or "", keep_blank_values=False)
    query = (params.get("q") or [""])[0].strip()
    if not query:
        return 422, {"type": "ErrorResponse",
                     "error": {"code": "VALIDATION", "status": 422,
                               "detail": "web search needs a non-empty query (q)"}}
    try:
        count = max(1, min(20, int((params.get("count") or ["5"])[0])))
    except ValueError:
        count = 5
    if any(params.get(k) for k in UNAPPLIED_BRAVE_FILTERS):
        _count("agent_web_search", "filters_not_applied")
    try:
        results, backend = search_tool().search_with_fallback(
            query, max_results=count, deadline=started + WEB_SEARCH_DEADLINE_S)
    except Exception as e:  # a broken search backend is a failed search, reported as such
        results, backend = [{"title": "error", "url": "", "snippet": redact(e)}], "error"
    _count("agent_web_search", backend)
    lookup = _count_search_lookup()
    if str(backend).startswith("gemini:"):
        _ANSWERED.search_model = str(backend).split(":", 1)[1]
    else:
        detail = "; ".join(redact(r.get("snippet", "")) for r in results[:1]) or backend
        asked = ""
        if lookup and lookup.get("models_asked"):
            # In the harness event only: the agent's 502 body never lists gbench's models.
            asked = f" (models asked: {', '.join(map(str, lookup['models_asked']))})"
        record_harness_event("agent_web_search", f"{backend}: {detail}{asked}", client,
                             service_failure=True)
        return 502, {"type": "ErrorResponse",
                     "error": {"code": "SEARCH_FAILED", "status": 502,
                               "detail": f"web search failed: Gemini Google Search grounding "
                                         f"could not serve this query ({backend}: {detail})"}}
    items = [{"title": r.get("title") or "", "url": r.get("url") or "",
              "description": r.get("snippet") or ""}
             for r in results if r.get("title") != "no results"]
    return 200, {"type": "search", "query": {"original": query},
                 "web": {"type": "search", "results": items}}


def last_search_model():
    """The grounding model that answered this thread's last brave_search_reply, or None."""
    return getattr(_ANSWERED, "search_model", None)


def _count_search_lookup():
    """What this thread's last agent web search lookup did (search_tool.last_lookup(): a
    search-engine lookup, sources only, no generated text), counted in the usage tally's
    agent_web_search_engine section and kept for its GEMINI_ANSWERS record; None when the
    search backend reports nothing."""
    try:
        last_lookup = getattr(search_tool(), "last_lookup", None)
        lookup = last_lookup() if callable(last_lookup) else None
    except Exception:  # the record is never a reason to fail a search
        lookup = None
    if not isinstance(lookup, dict):
        return None
    _ANSWERED.search_lookup = lookup
    _count("agent_web_search_engine", "lookups")
    if lookup.get("answered_by"):
        _count("agent_web_search_engine", "answered")
    for key, n in (("models_asked", len(lookup.get("models_asked") or [])),
                   ("answers_without_source", lookup.get("answers_without_source")),
                   ("links_resolved", lookup.get("links_resolved")),
                   ("links_unresolved", lookup.get("links_unresolved")),
                   ("links_direct", lookup.get("links_direct"))):
        if isinstance(n, int) and n > 0:
            _count("agent_web_search_engine", key, n)
    return lookup


def settle_web_search(client, arrived, started, code, delivered, now=None):
    """Record how one agent web search ended once its reply was written (or the write failed):
    an answered lookup (`code` 200) whose reply reached OpenClaw within its web_search timeout is
    an answer (GEMINI_ANSWERS, with the model that answered); one whose reply could not be written
    (the connection was gone) or came after AGENT_WEB_SEARCH_TIMEOUT_S (OpenClaw had already
    aborted the call) is an agent_web_search harness event: the agent never got the results.
    `arrived` is the wall-clock time the request arrived, `started` its time.monotonic()."""
    elapsed = (time.monotonic() if now is None else now) - started
    model = last_search_model()
    if code != 200:
        return
    if delivered and elapsed < AGENT_WEB_SEARCH_TIMEOUT_S:
        record_answer("agent_web_search", model, client, start=arrived,
                      search=getattr(_ANSWERED, "search_lookup", None))
        return
    why = ("the reply could not be written (the client had closed the connection)"
           if not delivered else
           f"answered after {elapsed:.1f} s, past OpenClaw's {AGENT_WEB_SEARCH_TIMEOUT_S:.0f} s "
           f"web_search timeout")
    record_harness_event("agent_web_search", f"gemini:{model}: {why}; the agent's call was "
                         f"aborted", client, start=arrived, service_failure=True)


# --------------------------------------------------------------------------- #
# Model relay: the agent's calls to the model under test, with the endpoint's key added here.
# --------------------------------------------------------------------------- #
#: Request headers not forwarded to the model endpoint: hop-by-hop, recomputed by urllib, or
#: replaced (Authorization carries the task container's placeholder).
_RELAY_DROP_REQUEST = frozenset({"host", "authorization", "proxy-authorization", "content-length",
                                 "transfer-encoding", "connection", "keep-alive",
                                 "proxy-connection", "te", "upgrade", "accept-encoding"})
#: Response headers not passed back: hop-by-hop, or set by this server itself.
_RELAY_DROP_RESPONSE = frozenset({"transfer-encoding", "connection", "keep-alive", "server",
                                  "date"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: urllib's handler would copy the Authorization header (the model
    endpoint's key) onto the request to the redirect target, another origin included. The 3xx
    reply reaches the relay as an HTTPError and is passed back unchanged."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


#: The relay connects to the model endpoint directly, never through an http(s)_proxy from the
#: environment (the task containers' own model calls did not use one either), and never follows
#: a redirect (_NoRedirect).
_RELAY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

#: Clients (task container addresses) whose last model relay request failed; the first success
#: after that is recorded as a model_endpoint_recovered event.
_RELAY_FAILED_CLIENTS = set()
_RELAY_LOCK = threading.Lock()


def _relay_failure(client, detail):
    record_harness_event("model_endpoint", detail, client)
    with _RELAY_LOCK:
        _RELAY_FAILED_CLIENTS.add(client)


def _relay_success(client):
    with _RELAY_LOCK:
        recovered = client in _RELAY_FAILED_CLIENTS
        _RELAY_FAILED_CLIENTS.discard(client)
    if recovered:
        record_harness_event("model_endpoint_recovered", "", client)


def _endpoint_failure_status(code):
    """True for a model endpoint reply that is a failure of the endpoint, not of the model: every
    5xx, and 429."""
    return code == 429 or 500 <= int(code) <= 599


#: One JSON line per chat request the model relay passed on ({time, start, client, status,
#: stream, max_tokens, finish_reason, content, reasoning, usage}): the per-turn max_tokens the
#: agent's OpenClaw sent (max_tokens, else max_completion_tokens; None when it sent neither), how
#: the reply ended (the last finish_reason), whether any answer text (content) and any reasoning
#: text arrived, and the token usage when the endpoint reported it (a streamed reply reports it
#: only when the request asked for it). The launcher records them per task.
MODEL_REQUESTS = "model_requests.jsonl"
#: The most bytes of a non-streamed reply the relay keeps to read its finish_reason and usage.
_SCAN_JSON_MAX_BYTES = 8 * 1024 * 1024
#: The longest server-sent-event line the relay reads (longer lines are passed on unread).
_SCAN_LINE_MAX_BYTES = 1024 * 1024


def request_shape(body):
    """(per-turn output cap, streamed) of a chat request body (bytes or None): the cap is
    max_tokens, else max_completion_tokens, None when it sends neither; (None, False) for a body
    that is not a JSON object."""
    try:
        parsed = json.loads(body.decode("utf-8")) if body else None
    except (ValueError, UnicodeDecodeError):
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    cap = parsed.get("max_tokens")
    if cap is None:
        cap = parsed.get("max_completion_tokens")
    return (cap if isinstance(cap, int) and not isinstance(cap, bool) else None,
            bool(parsed.get("stream")))


class ReplyScan:
    """Reads a chat completion reply as the relay passes it on (feed() each chunk, then done()):
    the last finish_reason, whether answer text or reasoning text arrived, and the usage when the
    reply carries it. Server-sent events are read line by line; a JSON reply is read at the end
    (up to _SCAN_JSON_MAX_BYTES). Never changes what is passed on."""

    def __init__(self, streamed):
        self.streamed = bool(streamed)
        self._buf = b""
        self._size = 0
        self._overflow = False
        self.finish_reason = None
        self.content = False
        self.reasoning = False
        self.usage = None

    def _choice(self, choice):
        if not isinstance(choice, dict):
            return
        if choice.get("finish_reason"):
            self.finish_reason = str(choice["finish_reason"])
        for part in (choice.get("delta"), choice.get("message")):
            if not isinstance(part, dict):
                continue
            if part.get("content"):
                self.content = True
            if part.get("reasoning_content") or part.get("reasoning"):
                self.reasoning = True

    def _event(self, obj):
        if not isinstance(obj, dict):
            return
        for choice in obj.get("choices") or []:
            self._choice(choice)
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens",
                                                     "total_tokens") if k in usage}

    def _line(self, line):
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            self._event(json.loads(data.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            pass

    def feed(self, chunk):
        try:
            if self.streamed:
                self._buf += chunk
                *lines, self._buf = self._buf.split(b"\n")
                for line in lines:
                    self._line(line)
                if len(self._buf) > _SCAN_LINE_MAX_BYTES:
                    self._buf = b""
            elif not self._overflow:
                self._size += len(chunk)
                if self._size > _SCAN_JSON_MAX_BYTES:
                    self._overflow, self._buf = True, b""
                else:
                    self._buf += chunk
        except Exception:   # the scan is a record, never a reason to fail the relay
            self._buf = b""

    def done(self):
        try:
            if self.streamed:
                self._line(self._buf)
            elif self._buf and not self._overflow:
                self._event(json.loads(self._buf.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            pass
        self._buf = b""
        return {"finish_reason": self.finish_reason, "content": self.content,
                "reasoning": self.reasoning, "usage": self.usage}


def record_model_request(record):
    """Append one relayed chat request's record to MODEL_REQUESTS."""
    _append_workdir_line(MODEL_REQUESTS, record)


def model_relay_target(path):
    """GBENCH_MODEL_BASE_URL/<rest> for a request path <MODEL_RELAY_PATH>/<rest> (query kept), or
    None when `path` is not under MODEL_RELAY_PATH."""
    prefix = MODEL_RELAY_PATH
    if not (path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?")):
        return None
    return os.environ.get("GBENCH_MODEL_BASE_URL", "").strip().rstrip("/") + path[len(prefix):]


class _Answered(Exception):
    """A chat request was answered with an error reply."""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default access logging
        pass

    def _request_body(self):
        """The request body: Content-Length bytes, or a chunked body decoded."""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else None

    def _relay_model(self, target):
        """Forward this request to the model endpoint with its key; stream the reply back."""
        if not os.environ.get("GBENCH_MODEL_BASE_URL", "").strip():
            self._send_json(502, {"error": {"message": "model relay: GBENCH_MODEL_BASE_URL is not "
                                                       "set", "type": "model_relay_error"}})
            return
        body = self._request_body()
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _RELAY_DROP_REQUEST}
        key = os.environ.get("GBENCH_MODEL_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(target, data=body, method=self.command, headers=headers)
        arrived = time.time()
        chat = self.command == "POST" and target.split("?", 1)[0].rstrip("/").endswith(
            "/completions")
        max_tokens, streamed = request_shape(body) if chat else (None, False)
        try:
            upstream = _RELAY_OPENER.open(req)   # no relay-side timeout (see the header)
        except urllib.error.HTTPError as e:
            upstream = e                         # an error status (or a 3xx) is passed back as is
        except Exception as e:
            _relay_failure(self.client_address[0], f"unreachable ({type(e).__name__})")
            self._send_json(502, {"error": {
                "message": f"model relay: the model endpoint could not be reached ({redact(e)})",
                "type": "model_relay_error"}})
            return
        client = self.client_address[0]
        with upstream:
            code = upstream.getcode()
            failed = _endpoint_failure_status(code)
            if failed:
                _relay_failure(client, f"HTTP {code}")
            self.send_response(code)
            for k, v in upstream.headers.items():
                if k.lower() not in _RELAY_DROP_RESPONSE:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            event_stream = "text/event-stream" in str(upstream.headers.get("Content-Type") or "")
            scan = ReplyScan(streamed or event_stream) if chat else None
            read = getattr(upstream, "read1", None) or upstream.read
            while True:
                try:
                    chunk = read(65536)
                except Exception as e:   # the endpoint cut the reply off: a truncated reply
                    _relay_failure(client, f"reply cut off ({type(e).__name__})")
                    failed = True
                    break
                if not chunk:
                    # A reply shorter than its Content-Length ends early without an error.
                    left = getattr(upstream, "length", None)
                    if isinstance(left, int) and left > 0:
                        _relay_failure(client, f"reply cut off ({left} bytes missing)")
                        failed = True
                    break
                if scan is not None:
                    scan.feed(chunk)
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    failed = True
                    break   # the client went away; closing the upstream reply ends the request
            if not failed and 200 <= code < 300:
                _relay_success(client)
            if scan is not None:
                record_model_request({"time": time.time(), "start": arrived, "client": client,
                                      "status": code, "stream": scan.streamed,
                                      "max_tokens": max_tokens, **scan.done()})

    def _send_json(self, code, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _web_search(self, query):
        """Serve one agent web search (brave_search_reply) and record how it ended
        (settle_web_search): an answer only when the reply was written within OpenClaw's
        web_search timeout, else an agent_web_search harness event."""
        client, arrived, started = self.client_address[0], time.time(), time.monotonic()
        code, body = brave_search_reply(query, client, started=started)
        delivered = True
        try:
            self._send_json(code, body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            delivered = False
        settle_web_search(client, arrived, started, code, delivered)

    def do_GET(self):
        target = model_relay_target(self.path)
        if target is not None:
            self._relay_model(target)
            return
        path, _, query = self.path.partition("?")
        if path.rstrip("/") == BRAVE_WEB_SEARCH_PATH:
            self._web_search(query)
            return
        if path.startswith("/res/"):
            self._send_json(404, {"type": "ErrorResponse", "error": {
                "code": "NOT_SERVED", "status": 404,
                "detail": f"{path} is not served; only {BRAVE_WEB_SEARCH_PATH} is answered "
                          f"(from Gemini Google Search grounding)"}})
            return
        # Lightweight health / models endpoints so clients that probe them succeed.
        if path.rstrip("/").endswith("/models"):
            self._send_json(200, {"object": "list", "data": [
                {"id": "gbench-cascade", "object": "model", "owned_by": "gbench"}]})
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self):
        target = model_relay_target(self.path)
        if target is not None:
            self._relay_model(target)
            return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        client = self.client_address[0]
        grader = is_grader_key(self.headers.get("Authorization"))
        started, ok = time.time(), False
        try:
            self._chat_completion(client, grader, started)
            ok = True
        except _Answered:
            pass
        finally:
            if grader:
                record_grader_call(started, time.time(), client, ok)

    def _chat_completion(self, client, grader, started=None):
        """Serve one chat request: a task grade()'s judge call (`grader`: every failure is a
        harness event of the task) or an agent's helper-model call (only a service-side failure
        is an event, "agent_helper"; the Gemini API refusing the agent's own request is answered
        with HTTP 400 and is the agent's outcome). `started` is when the request arrived; it is
        recorded with the event and with the answer (GEMINI_ANSWERS). Raises _Answered after an
        error reply."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(body, dict):
                raise ValueError("the request body is not a JSON object")
        except Exception as e:
            if grader:
                record_harness_event("judge", f"bad request: {e}", client, started,
                                     service_failure=False)
            self._send_json(400, {"error": {"message": f"bad request: {redact(e)}"}})
            raise _Answered()
        kind = "judge_web_search" if wants_web_search(body) else "judge"
        if thinking_off_requested(body):
            # Sent at THINKING_OFF_LEVEL (see cascade_chat_completion), counted per kind.
            _count("thinking_off", kind if grader else "agent_helper")
        label = kind if grader else "agent_helper"
        try:
            parsed = cascade_chat_completion(body, record=False, require_content=grader)
        except UnsupportedRequest as e:
            if grader:
                record_harness_event(kind, f"not served: {e}", client, started,
                                     service_failure=False)
            self._send_json(400, {"error": {"message": str(e), "type": "unsupported_request"}})
            raise _Answered()
        except RequestRejected as e:
            if grader:
                record_harness_event(kind, str(e), client, started, service_failure=False)
                self._send_json(503, {"error": {"message": redact(e), "type": "judge_outage"}})
            else:
                self._send_json(400, {"error": {"message": redact(e),
                                                "type": "invalid_request_error"}})
            raise _Answered()
        except Exception as e:
            # Surface as 503 so the client's create() raises; the upstream grade() then retries and
            # (per its own design) may fall back. The launcher records that as a judge fallback.
            _count_limit_state(label, outage=True)
            record_harness_event(kind if grader else "agent_helper", str(e), client, started,
                                 service_failure=True)
            self._send_json(503, {"error": {"message": redact(e), "type": "judge_outage"}})
            raise _Answered()
        _count_limit_state(label, outage=False)
        record_answer(kind if grader else "agent_helper", last_answering_model(), client,
                      started, grader)
        self._send_json(200, parsed)


def _count_limit_state(label, outage):
    """Count this thread's last chat request in the output-limit sections of the usage tally
    (thinking_allowance, max_tokens_cut, max_tokens_empty, outage_after_max_tokens) under `label`
    (judge, judge_web_search or agent_helper)."""
    state = last_limit_state()
    if state["raised"]:
        _count("thinking_allowance", label)
    if state["cut"] and not outage:
        _count("max_tokens_cut", label)
    for _ in range(int(state["empty"])):
        _count("max_tokens_empty", label)
    if outage and state["empty"]:
        _count("outage_after_max_tokens", label)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not gemini_keys():
        logger.error("GEMINI_API_KEYS / GEMINI_API_KEY is not set; the proxy does not start.")
        return 2
    try:
        search_tool()
    except Exception as e:
        logger.error("cannot load gbench search_tool: %s", redact(e))
        return 3
    if not os.environ.get("GBENCH_MODEL_BASE_URL", "").strip():
        logger.error("GBENCH_MODEL_BASE_URL is not set; the model relay has no endpoint and the "
                     "proxy does not start.")
        return 4
    try:
        _gemini_base()
    except GeminiEndpointRefused as e:
        logger.error("%s; the proxy does not start.", e)
        return 7
    host = os.environ.get("WILDCLAW_JUDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("WILDCLAW_JUDGE_PORT", "18795"))
    server = ThreadingHTTPServer((host, port), _Handler)
    logger.info("wildclawbench Gemini proxy listening on %s:%d (judge cascade=%s, rounds=%d; "
                "agent web_search on %s from Gemini Google Search grounding; model relay on %s)",
                host, port, ",".join(_cascade()), _rounds(), BRAVE_WEB_SEARCH_PATH,
                MODEL_RELAY_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
