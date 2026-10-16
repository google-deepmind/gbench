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

"""A `web_search` tool the agentic suites can actually call, backed by Gemini grounding.

`gaia` and `deepsearch_qa` are web-research benchmarks: their questions are answerable only
by looking things up. Run without any tool they score a structural 0 - on the 2026-08-15
sweep GAIA replied "I do not have access to the internet. Therefore, I cannot watch the
video" and both suites returned 0/20. That is not a measurement of the model.

This offers Gemini Google-Search grounding (`GEMINI_API_KEYS`, else `GEMINI_API_KEY`), exposed
as an OpenAI-style tool so the evaluated model drives the search itself. `bfcl_v4_agentic` keeps
BFCL's own web_search tool contract and serves it with this grounding cascade
(search_with_fallback, installed in its harness child); `mcp_atlas` serves the search tools of
the release's four search engines (brave-search, ddg-search, exa, oxylabs) through this module's
grounding cascade, key rotation, rate limiter and retry shape, under their own names and argument
schemas (mcp_atlas.gemini_search_reply); their page tools run as upstream on the release's
servers, with no Gemini call; wildclawbench's Gemini proxy serves OpenClaw's
web_search with it. The older mcp_atlas figures below are from its previous generic web_search
wiring.

Comparability: canonical GAIA leaderboard entries use a full browsing agent (page fetch,
file download, code execution). This is search-only, so a gbench GAIA number is **not**
leaderboard-comparable; the result records the backend it used.

Two backend facts, both measured live on 2026-08-20. Read them before quoting any number
from a search-dependent suite.

**Google Search grounding has its own quota, separate per-model from the RPM/TPM a project
dashboard reports.** The identical request returns HTTP 200 without `google_search` and can
be HTTP 429 with it, and the 429 is per-model - so one model being quota'd says nothing about
the next. A grounding 429 cannot be diagnosed from the model's RPM/TPM graph. This is exactly
why the search backend is a MODEL CASCADE (below), not a single model.

**Grounding uses a MODEL CASCADE, not a scraper fallback.** Grounding quota is per-model, so
a per-model 429 (which a concurrent burst hits) is recovered by trying the next model, each
with its own quota. The models are gbench's one Gemini cascade (base.GEMINI_CASCADE:
gemini-3.8-flash -> gemini-3.7-flash -> gemini-3.6-flash -> gemini-3.5-flash ->
gemini-3.1-pro-preview), which gbench hands to a container as GBENCH_JUDGE_MODELS;
GBENCH_SEARCH_MODELS replaces it for search only. A model the API answers 404 for (not served)
is not retried in place and not asked again in that lookup. This replaced the
DuckDuckGo fallback, which on this box is captcha-blocked (HTTP 202 anomaly page, 0/12 gold
recall) and returned blind snippets that also masked grounding failures - a second real
grounding model is strictly better. `search_backend_calls` records which model served each
call; check it before believing a number.

**High --sandboxes is throttle-safe.** The 2026-08-21 sweep ran the search suites at
--sandboxes 32; the request burst 429'd every cascade model at once and browsecomp / gaia /
mcp_atlas each recorded 100%% all-grounding-models-failed (the grounding path itself was fine
- verified live at concurrency 1 and 16 the same day). Two guards make any fan-out safe:
(1) a GLOBAL token-bucket rate limiter (`GBENCH_SEARCH_MAX_RPS`, default 8) that every
grounding request passes through, so the total request rate self-limits no matter how many
sandboxes call at once; and (2) when the whole cascade fails one pass, a bounded
backoff-with-jitter retry of the entire cascade (`GBENCH_SEARCH_CASCADE_ROUNDS`, default 3)
that lets a transient burst drain. Together they turn what was a fake zero into a completed
lookup. Tune the RPS down if you still see all-fail in the tally, up for a paid key.

**A lookup with a deadline hedges across the cascade.** A caller that must answer within a
deadline (wildclawbench's proxy: 57 s inside the 60 s web_search timeout gbench sets, OpenClaw's default 30 s; mcp_atlas: the
harness's 60 s tool-call timeout less its grace) used to give the FIRST model's request all of
the time left, so one slow request used up the whole deadline and no other model was ever asked:
the 2026-10-10 smoke run lost a wildclawbench task (04_Search_Retrieval_task_4) to 12
consecutive "deadline-passed" lookups, 12 different queries each failing at exactly 27 s, with
only gemini-3.8-flash ever tried, while its other searches answered in 3.7-14.6 s. Such a lookup
now sends a hedged request ("The Tail at Scale", Dean and Barroso, 2013): when the running
request has not answered within `GBENCH_SEARCH_HEDGE_DELAY_S` (default 12 s), the next cascade
model is asked in parallel, and the first successful answer is the lookup's. A failure (429, 404,
retries used up) still moves to the next model at once. Every hedge takes its token from the
global rate limiter, and each request's timeout stays capped at the deadline. A lookup without a
deadline (gaia, deepsearch_qa, bfcl_v4_agentic) asks the models one after another, as before.

**A lookup with a deadline is a search-engine lookup: sources only, no generated text.** Its
callers (wildclawbench's agent web_search behind the Brave API shape, mcp_atlas' search tools)
stand in for a search engine, so since 2026-10-12 such a lookup returns what a search engine
returns and nothing Gemini wrote:
- the request asks "Search the web and list the most relevant sources for: <query>"
  (SEARCH_ENGINE_PROMPT_PREFIX, the query unchanged inside it);
- the first `GBENCH_SEARCH_INITIAL_WIDTH` (default 2) cascade models are asked at once, and the
  hedge delay, failure hand-over, rate limiter and rounds above apply to the rest;
- an answer counts only when it cites at least one grounding source; an answer citing none is a
  failure of that model (the next model is asked; a later round does not ask it again), never
  an answer;
- each result is {title, url, snippet}: the title Google Search grounding reports for the source,
  the source's real URL (Google's grounding redirect link resolved by one HEAD request to the
  redirect service, reading its Location header; the source page itself is not contacted; a link
  that does not resolve within the time left is returned as the redirect link and counted), and an
  EMPTY snippet. No groundingSupports segment, no part of the answer and no grounded-answer
  fallback reaches the caller. Why (reported 2026-10-12 from a wildclawbench smoke run): such
  Gemini-written snippets handed the agent a fabricated pull-request number (#30340).
- duplicates (the same real URL) are dropped, the order kept, at most max_results returned;
- last_lookup() gives the caller what the lookup did (models asked, the model that answered,
  answers without a source, links resolved or not), for its result record.
Measured 2026-10-12 (5 recorded wildclawbench agent queries x gemini-3.8-flash and gemini-3.7-flash,
one try each): the query as the prompt answered 2/10 within 27 s (median 21.7 s), thinkingLevel
low 2/10 (no effect, so it is not set), the "list the most relevant sources" prompt 5/10, and that
prompt sent to both models at once 4/5 queries within 27 s against 1/5 for the query alone.
Queries with `site:` or a quoted phrase missed 27 s in every variant: Gemini does not treat them as
search operators. A lookup without a deadline keeps the behaviour described below unchanged.

**No page is fetched - NO `url_context` (anti-contamination).** Each result of a lookup without
a deadline is the title and URL of a source the grounding cited, with the segment of the Gemini
model's grounded answer that cites it (or the start of that answer): text the Gemini model wrote,
not text quoted from the page, and no page body is ever fetched (a lookup with a deadline returns
no such text, see above). The Gemini API's
`url_context` tool would let this fetch full page bodies (deeper retrieval), but it was
evaluated 2026-08-21 and **rejected**: these are public benchmarks whose question->answer
pairs live on the web (HF dataset viewers, GitHub, papers, "solved" write-ups), and fetching
a page that hosts the answer key is contamination - cheating, not capability. url_context
amplifies that over grounded answer segments (it ingests the whole page). If ever reconsidered
it must ship WITH a fetch-domain blocklist, a canary/exact-answer check, and per-URL logging -
never bare.
No suite fetches pages through Gemini: mcp_atlas' page tools (ddg-search_fetch_content,
exa_web_fetch_exa, oxylabs_universal_scraper and oxylabs_amazon_product_scraper) run on the
release's own servers, as upstream, with the release's credentials.

**The key never travels in a URL.** Every grounding request sends the Gemini key in the
`x-goog-api-key` header, so no request URL, and no exception text that echoes one, carries it;
a failed lookup's error text also has every configured key masked before it is returned (it
reaches the model under test as the tool result). The keys are read as base.gemini_api_keys
reads them: GEMINI_API_KEYS when it holds a key, else GEMINI_API_KEY, each one key or a comma or
semicolon list. The module stays standard-library only and imports nothing from gbench, because
wildclawbench's proxy loads it on its own.

**Through gbench's Gemini relay.** In a container that runs model-controlled code, gbench gives
the client a per-run placeholder as its Gemini key and the relay's origin in
`GBENCH_GEMINI_RELAY_URL` (gemini_relay.GeminiRelay.client_env). With that variable set, grounding
requests go to `<relay>/v1beta/models/<model>:generateContent`, where the relay adds the real
key; they are sent only while the key is a relay placeholder, so a real key is never sent to the
relay's address (the lookup fails with that reason instead).
"""

import json
import logging
import os
import queue
import random
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: How many grounding chunks to hand back per query.
MAX_RESULTS = int(os.environ.get("GBENCH_SEARCH_MAX_RESULTS", "8"))

#: The tool as the evaluated model sees it.
WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        # What the results are: the cited sources' titles and URLs with segments of a
        # Gemini-written grounded answer (search_with_fallback); no page is fetched.
        "description": (
            "Search the web through a Gemini model grounded in Google Search. Each result gives "
            "the title and URL of a source the Gemini model cited, with the part of that model's "
            "written answer that cites the source, or the start of its answer when it cited none "
            "(text written by the Gemini model, not quoted from the page). Use it whenever the "
            "answer depends on information you do not already know, and call it repeatedly to "
            "refine or cross-check. Prefer specific queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}


#: Grounding-model cascade, tried in order until one succeeds. Grounding quota is PER MODEL
#: (measured 2026-08-21 on one key: one model 429 while the others all 200 for the same
#: grounded request), so a per-model 429 - which a concurrent burst hits (mcp_atlas: 33%% of
#: searches got no data) - is recovered by the next model's separate quota. This REPLACED the
#: DuckDuckGo fallback: DDG is captcha-blocked on this box and returns blind snippets, whereas a
#: second grounding model is a real, equivalent backend.
#: This module imports nothing from gbench (wildclawbench's proxy loads it on its own), so this
#: is a copy of base.GEMINI_CASCADE, used only when no knob names the models; gbench hands the
#: list to a container as GBENCH_JUDGE_MODELS / GBENCH_SEARCH_MODELS (base.gemini_cascade_env),
#: and tests/test_gemini_cascade.py checks that the copy equals base.GEMINI_CASCADE.
_DEFAULT_SEARCH_CASCADE = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                           "gemini-3.5-flash", "gemini-3.1-pro-preview"]

#: Attempts per grounding call for TRANSIENT errors, and the base backoff. 429 is not
#: retried on the same model - the cascade moves to the next model instead (its own quota),
#: and if the WHOLE cascade fails, search_with_fallback backs off and retries the cascade
#: (see _SEARCH_CASCADE_ROUNDS). That is where a burst-429 is recovered; retrying the same
#: throttled model in place would only add latency.
_SEARCH_ATTEMPTS = int(os.environ.get("GBENCH_SEARCH_ATTEMPTS", "3"))
_SEARCH_BACKOFF = float(os.environ.get("GBENCH_SEARCH_BACKOFF", "1.0"))


def _cascade_rounds() -> int:
    """How many times to retry the WHOLE cascade when every model fails one pass.

    A synchronized all-model failure is the signature of a rate burst (the 2026-08-21 sweep
    ran the search suites at --sandboxes 32 and 429'd every cascade model at once, recording
    100%% all-grounding-models-failed). Backing off and re-running the cascade lets the burst
    drain and the lookup complete. Read at call time so a sweep/test can tune it live.
    """
    return max(1, int(os.environ.get("GBENCH_SEARCH_CASCADE_ROUNDS", "3")))


#: How long a grounding request of a lookup with a deadline may run unanswered before the next
#: cascade model is asked in parallel (search_with_fallback's hedged request). Measured
#: 2026-10-10 (wildclawbench smoke run, gemini_answers.jsonl): agent searches answered in 3.7,
#: 10.2, 11.5 and 14.6 s, and 12 others got no answer from gemini-3.8-flash within the proxy's
#: 27 s deadline (57 s since 2026-10-13). 12 s left the hedge 15 s of the 27 s deadline, more than
#: the slowest answer measured, while a request that answers in the usual time sends no hedge.
_DEFAULT_HEDGE_DELAY_S = 12.0


def _float_knob(name: str, default: float) -> float:
    """The number in environment variable `name` (`default` when unset). Raises ValueError
    naming the variable when it is set to something that is not a number."""
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None


def _hedge_delay() -> Optional[float]:
    """The hedge delay in seconds (GBENCH_SEARCH_HEDGE_DELAY_S, default _DEFAULT_HEDGE_DELAY_S),
    or None when it is 0 or less: then no timed hedge is sent, and a further model of a lookup
    with a deadline is asked only once a running request failed. The first _initial_width()
    models are still asked at once at the start, so the models are asked strictly one after
    another only with GBENCH_SEARCH_INITIAL_WIDTH=1 as well. Read at call time, like
    _cascade_rounds."""
    delay = _float_knob("GBENCH_SEARCH_HEDGE_DELAY_S", _DEFAULT_HEDGE_DELAY_S)
    return delay if delay > 0 else None


#: The prompt of a search-engine lookup (a lookup with a deadline), before the caller's query,
#: which goes in unchanged. Measured 2026-10-12 on 5 recorded wildclawbench agent queries x 2
#: models: 5/10 answers within 27 s with this prompt, 2/10 with the query alone as the prompt.
SEARCH_ENGINE_PROMPT_PREFIX = "Search the web and list the most relevant sources for: "
#: What last_lookup() reports as `search_mode` for a lookup with a deadline.
SEARCH_ENGINE_MODE = "search-engine"
#: How many cascade models a search-engine lookup asks at once when it starts. Measured
#: 2026-10-12: with the sources prompt, the first answer of gemini-3.8-flash and gemini-3.7-flash
#: asked together came within 27 s for 4 of 5 recorded queries.
_DEFAULT_INITIAL_WIDTH = 2
#: The longest one grounding redirect link may take to resolve (a HEAD request, the Location
#: header read); the time left before the lookup's deadline lowers it. Measured 2026-10-12: a
#: HEAD to vertexaisearch.cloud.google.com/grounding-api-redirect/<token> answered 302 with the
#: source URL in 0.08-0.13 s.
_DEFAULT_REDIRECT_TIMEOUT_S = 3.0
#: Google's grounding redirect links (web.uri of a grounding chunk); only these are resolved.
GROUNDING_REDIRECT_PREFIX = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
#: gemini_search's reply in a search-engine lookup when the answer cited no grounding source.
_NO_SOURCES = "search backend failed: the grounded answer cited no web source"


def _initial_width() -> int:
    """How many cascade models a search-engine lookup asks at once at its start
    (GBENCH_SEARCH_INITIAL_WIDTH, default _DEFAULT_INITIAL_WIDTH, at least 1). Read at call
    time, like _cascade_rounds."""
    raw = os.environ.get("GBENCH_SEARCH_INITIAL_WIDTH")
    if raw is None:
        return _DEFAULT_INITIAL_WIDTH
    try:
        return max(1, int(raw))
    except ValueError:
        raise ValueError(f"GBENCH_SEARCH_INITIAL_WIDTH={raw!r} is not a whole number") from None


def _redirect_timeout() -> float:
    """The per-link timeout of resolving a grounding redirect (GBENCH_SEARCH_REDIRECT_TIMEOUT_S,
    default _DEFAULT_REDIRECT_TIMEOUT_S; 0 or less: links are not resolved)."""
    return _float_knob("GBENCH_SEARCH_REDIRECT_TIMEOUT_S", _DEFAULT_REDIRECT_TIMEOUT_S)


class _RateLimiter:
    """Thread-safe token bucket that caps the GLOBAL grounding request rate.

    The grounding rate limit is per-model, but the burst that trips it is global: a high
    --sandboxes fan-out fires many concurrent `search_with_fallback` calls, and when their
    combined request rate exceeds the limit every cascade model 429s at once (browsecomp /
    gaia / mcp_atlas each recorded 100%% failure on 2026-08-21). Capping the total request
    rate here makes ANY --sandboxes safe: excess callers block until a token frees, so the
    request rate self-limits regardless of fan-out. `rps <= 0` disables it. clock/sleep are
    injectable so the behaviour is testable without real waits.
    """

    def __init__(self, rps: float, burst: float, *,
                 clock=time.monotonic, sleep=time.sleep):
        self.rps = float(rps)
        self.capacity = max(1.0, float(burst))
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = threading.Lock()

    def acquire(self, deadline: Optional[float] = None,
                cancel: Optional[threading.Event] = None) -> bool:
        """Take one token, waiting until one frees; True once taken. With `deadline` (a value of
        this limiter's clock, time.monotonic() by default) it returns False, without a token,
        instead of waiting until or past the deadline. With `cancel` it returns False, without a
        token, once that event is set (checked before every take, so after at most one wait)."""
        if self.rps <= 0:
            return True
        while True:
            with self._lock:
                if cancel is not None and cancel.is_set():
                    return False
                now = self._clock()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self._updated) * self.rps)
                self._updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                wait = (1.0 - self.tokens) / self.rps
            if deadline is not None and deadline - now <= wait:
                return False
            self._sleep(wait)


def _build_rate_limiter() -> _RateLimiter:
    rps = float(os.environ.get("GBENCH_SEARCH_MAX_RPS", "8"))
    burst = float(os.environ.get("GBENCH_SEARCH_MAX_BURST", str(max(1, int(rps)))))
    return _RateLimiter(rps, burst)


#: The global limiter every grounding HTTP request passes through. Built from the environment
#: at import; call reset_rate_limiter() after changing GBENCH_SEARCH_MAX_RPS/_BURST.
_RATE_LIMITER = _build_rate_limiter()


def reset_rate_limiter() -> None:
    """Rebuild the global limiter from the current environment (after an env change / tests)."""
    global _RATE_LIMITER
    _RATE_LIMITER = _build_rate_limiter()


def _model_list(raw: Optional[str]) -> List[str]:
    return [m.strip() for m in (raw or "").split(",") if m.strip()]


def _search_cascade() -> List[str]:
    """The ordered list of grounding models to try: base.search_cascade's rule without importing
    gbench. GBENCH_SEARCH_MODELS (comma separated), else GBENCH_SEARCH_MODEL (one model), else
    the Gemini cascade every other role uses: GBENCH_JUDGE_MODELS, else GBENCH_JUDGE_MODEL, else
    _DEFAULT_SEARCH_CASCADE (= base.GEMINI_CASCADE)."""
    for plural, single in (("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL"),
                           ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL")):
        models = _model_list(os.environ.get(plural))
        if models:
            return models
        one = (os.environ.get(single) or "").strip()
        if one:
            return [one]
    return list(_DEFAULT_SEARCH_CASCADE)


def _split_keys(raw: Optional[str]) -> List[str]:
    return [k.strip() for k in re.split(r"[;,]", raw or "") if k.strip()]


def _gemini_keys() -> List[str]:
    """The configured Gemini keys, in order, de-duplicated: base.gemini_api_keys' rule
    (GEMINI_API_KEYS when it holds a key, else GEMINI_API_KEY; either one key or a comma or
    semicolon list). Copied here because this module is loaded on its own (no gbench import)."""
    keys = _split_keys(os.environ.get("GEMINI_API_KEYS")) or \
        _split_keys(os.environ.get("GEMINI_API_KEY"))
    return list(dict.fromkeys(keys))


def _gemini_key_set() -> bool:
    """A Gemini key is configured (_gemini_keys is not empty)."""
    return bool(_gemini_keys())


def search_available() -> bool:
    return _gemini_key_set()


def unavailable_reason() -> str:
    return ("web search requires GEMINI_API_KEYS or GEMINI_API_KEY (Google-Search grounding "
            "across the model "
            "cascade). Without it, every question that needs a lookup is unanswerable, so the "
            "suite hard-errors rather than report a structural 0%")


def _is_failure(results: List[Dict[str, str]]) -> bool:
    """True when grounding reported a failure (429/network/etc) rather than an answer.

    "no results" is a real answer and must NOT advance the cascade - re-running a genuinely
    empty query on another model just burns quota and can invent a match. Grounding is
    trustworthy when it says empty.
    """
    return bool(results) and results[0].get("title") == "error"


import itertools

_KEY_LOCK = threading.Lock()
_KEY_CYCLE: Optional[Any] = None
#: The key list _KEY_CYCLE rotates over; a different list (the environment changed) rebuilds it.
_KEY_CYCLE_KEYS: Tuple[str, ...] = ()


def get_gemini_api_key() -> str:
    """The next Gemini key in round-robin order over _gemini_keys(), or "" when none is set."""
    global _KEY_CYCLE, _KEY_CYCLE_KEYS
    keys = tuple(_gemini_keys())
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    with _KEY_LOCK:
        if _KEY_CYCLE is None or _KEY_CYCLE_KEYS != keys:
            _KEY_CYCLE, _KEY_CYCLE_KEYS = itertools.cycle(keys), keys
        return next(_KEY_CYCLE)


#: What a configured key is replaced with in an error text (base.REDACTED's value).
_REDACTED = "[REDACTED]"


def _mask_keys(text: str) -> str:
    """`text` with every configured Gemini key (as is, and URL-encoded) replaced by _REDACTED,
    longest first, so an error text returned to the caller never carries one."""
    import urllib.parse
    forms = set()
    for key in _gemini_keys():
        forms.update({key, urllib.parse.quote(key, safe=""), urllib.parse.quote_plus(key)})
    for form in sorted((f for f in forms if f), key=len, reverse=True):
        text = text.replace(form, _REDACTED)
    return text


def _remaining(deadline: Optional[float]) -> Optional[float]:
    """Seconds left before `deadline` (a time.monotonic() value), or None without one."""
    return None if deadline is None else deadline - time.monotonic()


#: The Gemini API's generateContent endpoint. The key goes in the x-goog-api-key header.
GEMINI_GENERATE_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
                       "{model}:generateContent")
#: Set by gbench in a container whose Gemini calls go through its host-side key relay
#: (gemini_relay.RELAY_URL_ENV): the relay's origin, http://<address>:<port>. Used only while the
#: key the request would carry is the relay's per-run placeholder, so a real key never goes there.
RELAY_URL_ENV = "GBENCH_GEMINI_RELAY_URL"
#: The prefix of every relay placeholder (gemini_relay.PLACEHOLDER_PREFIX), then 48 hex digits.
RELAY_PLACEHOLDER_PREFIX = "gbench-gemini-relay-"


def _generate_url(model: str, key: str) -> Tuple[Optional[str], str]:
    """(generateContent URL for `model`, "") or (None, why the request must not be sent).

    Without RELAY_URL_ENV this is GEMINI_GENERATE_URL. With it, the relay's URL, and only when
    `key` is a relay placeholder: a real key is never sent to the relay's address."""
    import urllib.parse
    relay = os.environ.get(RELAY_URL_ENV, "").strip()
    if not relay:
        return GEMINI_GENERATE_URL.format(model=model), ""
    if not re.fullmatch(re.escape(RELAY_PLACEHOLDER_PREFIX) + r"[0-9a-f]{48}", key or ""):
        return None, (f"{RELAY_URL_ENV} is set but the Gemini key is not the relay's "
                      f"placeholder, so the key is not sent to it; unset {RELAY_URL_ENV}")
    try:
        parts = urllib.parse.urlsplit(relay)
        parts.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        parts = None
    if (parts is None or parts.scheme not in ("http", "https") or not parts.hostname
            or "@" in parts.netloc or parts.path not in ("", "/") or parts.query
            or parts.fragment):
        return None, f"{RELAY_URL_ENV} is not an http(s) origin (scheme://host:port)"
    return (f"{parts.scheme}://{parts.netloc}/v1beta/models/"
            f"{urllib.parse.quote(str(model), safe='')}:generateContent"), ""


#: Ceiling of one grounding request; a caller's deadline lowers it.
_REQUEST_TIMEOUT_S = 90.0
#: gemini_search's reply when its deadline stopped it before a request was answered.
_NO_ANSWER_BEFORE_DEADLINE = "search backend failed: no answer before the deadline"
#: gemini_search's reply in a hedged request whose lookup was answered by another model.
_HEDGE_ABANDONED = "search backend failed: the lookup was answered by another model"


#: HTTP statuses that say the model itself is not available (not served for this key), so a
#: retry on the same model cannot help: the cascade moves on and does not ask it again.
_MODEL_NOT_SERVED_STATUSES = (404,)


#: Per thread: whether the last gemini_search call's model answered a status of
#: _MODEL_NOT_SERVED_STATUSES (search_with_fallback then does not ask it again in that lookup),
#: and, in a hedged request's thread, the event set once its lookup is answered (`abandoned`).
_LAST_CALL = threading.local()


def _last_model_not_served() -> bool:
    return bool(getattr(_LAST_CALL, "not_served", False))


def gemini_search(query: str, max_results: int = MAX_RESULTS,
                  model: Optional[str] = None,
                  deadline: Optional[float] = None,
                  search_engine: bool = False) -> List[Dict[str, str]]:
    """Google-Search-grounded results as [{title, url, snippet}], from one grounding model.

    `deadline` (optional, a time.monotonic() value) caps each request's timeout at the time left
    and stops the retries once it has passed (the lookup is then a failure). A model the API
    answers 404 for (not served) is not retried in place.

    `search_engine` (search_with_fallback sets it for a lookup with a deadline): the prompt is
    SEARCH_ENGINE_PROMPT_PREFIX + query, and the results are every grounding source that has a
    URL, as {title, url (as grounding gives it, a redirect link), snippet ""}, uncapped (the
    caller resolves, de-duplicates and caps them); an answer that cites no source is the failure
    _NO_SOURCES. No text the Gemini model wrote is returned."""
    results, not_served = _gemini_search(query, max_results, model, deadline, search_engine)
    _LAST_CALL.not_served = not_served
    return results


def _gemini_search(query: str, max_results: int = MAX_RESULTS, model: Optional[str] = None,
                   deadline: Optional[float] = None,
                   search_engine: bool = False) -> Tuple[List[Dict[str, str]], bool]:
    """gemini_search's results, and whether the model answered HTTP 404 (not served)."""
    import urllib.request

    key = get_gemini_api_key()
    if not key:
        return [{"title": "error", "url": "", "snippet": "GEMINI_API_KEY is not set"}], False
    if not model:
        model = _search_cascade()[0]
    url, why = _generate_url(model, key)
    if url is None:
        return [{"title": "error", "url": "", "snippet": f"search backend failed: {why}"}], False
    text = SEARCH_ENGINE_PROMPT_PREFIX + str(query) if search_engine else str(query)
    payload = {"contents": [{"parts": [{"text": text}]}],
               "tools": [{"google_search": {}}]}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    # Retry transient errors, but NOT 429 on the SAME model. Measured 2026-08-20: 2 of 12
    # grounding calls failed with a transient error and both succeeded on re-issue; without a
    # retry a one-off blip became a permanent "the web has no answer" for that sample. 429 is
    # excluded here on purpose - a throttled model does not clear in-place, so the cascade
    # moves to the next model (its own quota) and, if every model is throttled at once,
    # search_with_fallback backs off and retries the whole cascade. Every attempt first takes
    # a token from the global rate limiter, so a high --sandboxes fan-out cannot burst past
    # the grounding rate limit in the first place.
    data = None
    abandoned = getattr(_LAST_CALL, "abandoned", None)
    for attempt in range(_SEARCH_ATTEMPTS):
        left = _remaining(deadline)
        if left is not None and left <= 0:
            return [{"title": "error", "url": "", "snippet": _NO_ANSWER_BEFORE_DEADLINE}], False
        if abandoned is not None and abandoned.is_set():
            # A hedged request whose lookup another model already answered: no further attempt.
            return [{"title": "error", "url": "", "snippet": _HEDGE_ABANDONED}], False
        # With a deadline, the wait for a token stops at the deadline too; in a hedged request it
        # also stops once the lookup is answered. Reviewed 2026-10-10: a request that lost the race
        # while it waited for a token (a busy limiter) went on to take one and send its request
        # after the lookup had been answered, spending a token and the model's grounding quota.
        if deadline is None:
            got = _RATE_LIMITER.acquire()
        elif abandoned is None:
            got = _RATE_LIMITER.acquire(deadline)
        else:
            got = _RATE_LIMITER.acquire(deadline, cancel=abandoned)
        if abandoned is not None and abandoned.is_set():
            return [{"title": "error", "url": "", "snippet": _HEDGE_ABANDONED}], False
        left = _remaining(deadline)
        if got is False or (left is not None and left <= 0):
            return [{"title": "error", "url": "", "snippet": _NO_ANSWER_BEFORE_DEADLINE}], False
        timeout = _REQUEST_TIMEOUT_S if left is None else max(0.1, min(_REQUEST_TIMEOUT_S, left))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 429 or code in _MODEL_NOT_SERVED_STATUSES \
                    or attempt == _SEARCH_ATTEMPTS - 1:
                # A failed lookup is reported to the model as a failed lookup, not as "no
                # results" - otherwise the model concludes the fact does not exist. The text
                # reaches the model: every configured key is masked in it. A model that is not
                # served (404) is not retried in place: no retry can make it answer.
                return [{"title": "error", "url": "",
                         "snippet": _mask_keys(f"search backend failed: {e}")}], \
                    code in _MODEL_NOT_SERVED_STATUSES
            # transient (network / 5xx): exponential backoff with jitter, then retry.
            pause = _SEARCH_BACKOFF * (2 ** attempt) + random.uniform(0, _SEARCH_BACKOFF)
            left = _remaining(deadline)
            time.sleep(pause if left is None else max(0.0, min(pause, left)))

    cand = (data.get("candidates") or [{}])[0]
    gm = cand.get("groundingMetadata") or {}
    chunks = gm.get("groundingChunks") or []

    if search_engine:
        # Sources only: the title and URL grounding reports for each cited source, never text
        # the Gemini model wrote (no groundingSupports segment, no answer text, no fallback).
        sources = []
        for ch in chunks:
            web = (ch or {}).get("web") or {}
            if web.get("uri"):
                sources.append({"title": web.get("title") or web.get("domain") or "result",
                                "url": web["uri"], "snippet": ""})
        if not sources:
            return [{"title": "error", "url": "", "snippet": _NO_SOURCES}], False
        return sources, False

    bodies: Dict[int, List[str]] = {}
    for sup in gm.get("groundingSupports") or []:
        text = ((sup.get("segment") or {}).get("text") or "").strip()
        for idx in sup.get("groundingChunkIndices") or []:
            if text:
                bodies.setdefault(int(idx), []).append(text)

    answer = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
    results: List[Dict[str, str]] = []
    for i, ch in enumerate(chunks[:max_results]):
        web = ch.get("web") or {}
        results.append({
            "title": web.get("title") or web.get("domain") or "result",
            "url": web.get("uri") or "",
            "snippet": " ".join(bodies.get(i, [])) or answer[:500],
        })
    if not results and answer:
        results = [{"title": "grounded-answer", "url": "", "snippet": answer[:1500]}]
    if not results:
        results = [{"title": "no results", "url": "", "snippet": "The search returned nothing."}]
    return results, False


async def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Tool executor for `run_eval_suite(tool_executor=...)`. Returns the tool message."""
    import asyncio
    if name != "web_search":
        return json.dumps({"error": f"unknown tool '{name}'"})
    query = str((args or {}).get("query") or "").strip()
    if not query:
        return json.dumps({"error": "web_search requires a non-empty 'query'"})
    results, backend = await asyncio.to_thread(search_with_fallback, query)
    return json.dumps({"query": query, "results": results, "backend": backend},
                      ensure_ascii=False)


def search_with_fallback(query: str, max_results: int = MAX_RESULTS,
                         deadline: Optional[float] = None) -> Tuple[List[Dict[str, str]], str]:
    """Ground the query, cascading across gemini models until one succeeds.

    `deadline` (optional, a time.monotonic() value) bounds the whole lookup: each request's
    timeout is capped at the time left, the wait for a rate-limit token stops at it, and the
    cascade, its rounds and its backoff stop once it has passed; the lookup is then a failure
    with backend "deadline-passed" (also when the time ran out during the last attempt). With a
    deadline, the lookup is a search-engine lookup (see the module docstring): the first
    GBENCH_SEARCH_INITIAL_WIDTH models are asked at once, a request that has not answered within
    GBENCH_SEARCH_HEDGE_DELAY_S is hedged with the next model (_hedged_pass), so one slow model
    cannot use up the whole deadline, only an answer citing a source counts, and the results are
    the sources' titles and real URLs with empty snippets (_search_engine_results); last_lookup()
    then describes the lookup.

    Google-Search grounding quota is PER MODEL (measured 2026-08-20/21: on one key, one model
    returned 429 while the other models all returned 200 for the identical grounded request).
    So a per-model 429 - which a concurrent burst hits easily (mcp_atlas: 299 429s, 33%% of its
    searches got no data) - is recovered by trying the NEXT model, each with its own quota. This
    replaced the DuckDuckGo fallback, which on this box is captcha-blocked and returns blind
    snippets; a second real grounding model is strictly better than a broken scraper. The backend that actually
    served (the model id) is returned and tallied so a run never silently claims grounding it
    did not get.
    """
    def _tallied(results, backend):
        _BACKEND_TALLY[backend] = _BACKEND_TALLY.get(backend, 0) + 1
        return results, backend

    def _past_deadline():
        left = _remaining(deadline)
        return left is not None and left <= 0

    def _deadline_failure():
        return _tallied([{"title": "error", "url": "",
                          "snippet": "search backend failed: no grounding model answered before "
                                     "the deadline"}], "deadline-passed")

    _LAST_LOOKUP.info = None
    if _gemini_key_set():
        cascade = _search_cascade()
        rounds = _cascade_rounds()
        last = None
        not_served: set = set()
        lookup: Dict[str, Any] = {}
        if deadline is not None:
            # A malformed knob of the search-engine path raises here, before any request is
            # sent, never after a model answered (the 2026-10-12 review: a bad
            # GBENCH_SEARCH_REDIRECT_TIMEOUT_S raised in _search_engine_results while
            # last_lookup() already named the model that answered).
            _initial_width(), _hedge_delay(), _redirect_timeout()
            lookup = {"search_mode": SEARCH_ENGINE_MODE, "models_asked": [], "answered_by": None,
                      "answers_without_source": 0, "links_resolved": 0,
                      "links_unresolved": 0, "links_direct": 0, "unresolved_reasons": []}
            _LAST_LOOKUP.info = lookup
        for rnd in range(rounds):
            if deadline is not None:
                # One pass with hedged requests (_hedged_pass); the rounds, their backoff and
                # the not-served set are as without a deadline, except that the set also holds
                # the models that answered without a source (not asked again in a later round).
                if _past_deadline():
                    return _deadline_failure()
                model, results, stopped = _hedged_pass(query, max_results, cascade, deadline,
                                                       not_served, lookup)
                if model is not None:
                    found = _search_engine_results(results, max_results, deadline, lookup)
                    lookup["answered_by"] = model    # only once its results are built
                    return _tallied(found, f"gemini:{model}")
                if stopped:
                    return _deadline_failure()   # the deadline stopped it (a token wait included)
                last = results or last
                if len(not_served) >= len(cascade):
                    break
                if rnd < rounds - 1:
                    pause = _SEARCH_BACKOFF * (2 ** rnd) + random.uniform(0, _SEARCH_BACKOFF)
                    time.sleep(max(0.0, min(pause, _remaining(deadline))))
                continue
            # Without a deadline: the models in turn, on the caller's thread.
            for model in cascade:
                if model in not_served:
                    continue
                _LAST_CALL.not_served = False
                results = gemini_search(query, max_results, model=model)
                missing = _last_model_not_served()
                if not _is_failure(results):
                    return _tallied(results, f"gemini:{model}")
                if missing:
                    not_served.add(model)
                last = results
            if len(not_served) >= len(cascade):
                break                    # no model left that could answer
            # Every model failed this pass. A single model's 429 is handled above by moving to
            # the next model, so reaching here means they were ALL failing at once - the
            # signature of a rate burst (or a transient network-wide fault). Back off with
            # jitter and retry the whole cascade: the burst drains while we wait, which is what
            # turns a --sandboxes-32 all-fail into a completed lookup instead of a fake zero.
            if rnd < rounds - 1:
                time.sleep(_SEARCH_BACKOFF * (2 ** rnd) + random.uniform(0, _SEARCH_BACKOFF))
        # Still failing after every round: quotas genuinely out, or a real outage; or the time
        # ran out during the last attempt, which is a deadline failure like any other.
        if _past_deadline():
            return _deadline_failure()
        return _tallied(last or [{"title": "error", "url": "",
                                  "snippet": "all grounding models failed"}],
                        "all-grounding-models-failed")
    # No key: nothing to ground with. The search suites hard-error before this is reachable.
    return _tallied([{"title": "error", "url": "", "snippet": unavailable_reason()}],
                    "none")


def _hedged_pass(query: str, max_results: int, cascade: List[str], deadline: float,
                 not_served: set, lookup: Optional[Dict[str, Any]] = None
                 ) -> Tuple[Optional[str], Optional[List[Dict[str, str]]], bool]:
    """One pass of search_with_fallback over `cascade` for a lookup with a deadline:
    (the model that answered, its results, False), else (None, the last failure, whether the
    deadline stopped the pass). Every request is a search-engine request (gemini_search with
    search_engine=True: an answer citing no source is a failure). `lookup` (optional) gets each
    model asked appended to "models_asked" and each answer without a source counted in
    "answers_without_source".

    The first _initial_width() models not in `not_served` are asked at once. When the newest
    running request has not
    answered within _hedge_delay(), the next model is asked in parallel (a hedged request), as
    long as at least that delay is left before the deadline; when a request fails, the next model
    is asked at once. The first successful answer ends the pass. Each request runs gemini_search
    on a thread of its own, so a model's 404 (not served) is read from that thread's _LAST_CALL
    and added to `not_served` here. A model whose answer cited no source is added to
    `not_served` too: the next model is asked at once, but a later round of search_with_fallback
    does not ask it again, so rounds retry only transient failures (429, 5xx, network). Measured
    by the 2026-10-12 review: a query no model cited a source for cost 15 grounding requests
    (5 models x 3 rounds) before this, len(cascade) now. A request still running when the pass
    ends is abandoned: its result is never tallied, it sends no further attempt (also when it was
    still waiting for a rate-limiter token: it stops waiting without one), and its timeout is
    capped at the deadline like every request's."""
    answers: "queue.Queue[Tuple[str, List[Dict[str, str]], bool]]" = queue.Queue()
    abandoned = threading.Event()
    delay = _hedge_delay()
    waiting = [m for m in cascade if m not in not_served]
    running = 0
    next_hedge: Optional[float] = None
    last: Optional[List[Dict[str, str]]] = None
    stopped = False

    def ask(model: str) -> None:
        _LAST_CALL.not_served = False
        _LAST_CALL.abandoned = abandoned
        try:
            results = gemini_search(query, max_results, model=model, deadline=deadline,
                                    search_engine=True)
        except Exception as e:  # gemini_search reports its failures as results; this is a guard
            results = [{"title": "error", "url": "",
                        "snippet": _mask_keys(f"search backend failed: {e}")}]
        answers.put((model, results, _last_model_not_served()))

    def launch() -> None:
        nonlocal running, next_hedge
        model = waiting.pop(0)
        if lookup is not None:
            lookup.setdefault("models_asked", []).append(model)
        threading.Thread(target=ask, args=(model,), name=f"gbench-search-{model}",
                         daemon=True).start()
        running += 1
        next_hedge = None if delay is None else time.monotonic() + delay

    try:
        for _ in range(min(_initial_width(), len(waiting))):
            launch()
        while running:
            left = _remaining(deadline)
            if left <= 0:
                return None, last, True
            wait = left
            if waiting and next_hedge is not None:
                wait = min(wait, max(0.0, next_hedge - time.monotonic()))
            try:
                model, results, missing = answers.get(timeout=wait)
            except queue.Empty:
                if waiting and next_hedge is not None and time.monotonic() >= next_hedge:
                    if _remaining(deadline) >= delay:
                        logger.info("grounding search: no answer within %gs, asking %s too",
                                    delay, waiting[0])
                        launch()
                    else:
                        next_hedge = None    # too little time left for a hedge to answer
                continue
            running -= 1
            if not _is_failure(results):
                return model, results, False
            if results and results[0].get("snippet") == _NO_ANSWER_BEFORE_DEADLINE:
                stopped = True
            elif results and results[0].get("snippet") == _NO_SOURCES:
                if lookup is not None:
                    lookup["answers_without_source"] = lookup.get("answers_without_source", 0) + 1
                not_served.add(model)    # not asked again in a later round
            elif missing:
                not_served.add(model)
            last = results
            if waiting and _remaining(deadline) > 0:
                launch()
        return None, last, stopped
    finally:
        abandoned.set()


#: Per thread: what the last search_with_fallback call with a deadline did (last_lookup).
_LAST_LOOKUP = threading.local()


def last_lookup() -> Optional[Dict[str, Any]]:
    """What this thread's last search_with_fallback call did, when it had a deadline (a
    search-engine lookup), else None: {search_mode, models_asked (in the order asked, every
    round), answered_by (None when no model answered), answers_without_source, links_resolved
    (grounding redirect links resolved to the source URL), links_unresolved (returned as the
    redirect link), links_direct (a source URL that was not a redirect link), unresolved_reasons}.
    The link counts are of the results returned."""
    info = getattr(_LAST_LOOKUP, "info", None)
    if info is None:
        return None
    return {k: list(v) if isinstance(v, list) else v for k, v in info.items()}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the 3xx reply reaches _redirect_location as an HTTPError, whose
    Location header is the source URL. The source page itself is never requested."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


#: The opener grounding redirect links are resolved with (no redirect followed; the
#: environment's proxy settings apply, as for the grounding requests).
_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect())
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
#: The longest Location a resolved link may have; a longer one keeps the redirect link (the
#: URL goes into the agent's context unchanged).
_MAX_LOCATION_CHARS = 2048


def _redirect_location(url: str, timeout: float) -> str:
    """The http(s) URL a grounding redirect link points to: one HEAD request, no redirect
    followed, the Location header read. Raises ValueError (or the request's error) otherwise.
    No key or other header of the grounding request is sent."""
    import urllib.error
    import urllib.parse
    req = urllib.request.Request(url, method="HEAD")
    try:
        with _REDIRECT_OPENER.open(req, timeout=timeout) as r:
            raise ValueError(f"HTTP {getattr(r, 'status', '?')} without a redirect")
    except urllib.error.HTTPError as e:
        try:
            location = (e.headers.get("Location") or "").strip() if e.headers else ""
        finally:
            e.close()
        if e.code not in _REDIRECT_STATUSES:
            raise ValueError(f"HTTP {e.code}") from None
        if urllib.parse.urlsplit(location).scheme not in ("http", "https"):
            raise ValueError(f"HTTP {e.code} without an http(s) Location") from None
        if len(location) > _MAX_LOCATION_CHARS:
            raise ValueError(f"HTTP {e.code} with a Location of {len(location)} characters "
                             f"(more than {_MAX_LOCATION_CHARS})") from None
        return location


def _resolve_redirects(links: List[str], deadline: float) -> Tuple[Dict[str, str], Dict[str, str]]:
    """({redirect link: source URL}, {redirect link: why it did not resolve}) for `links`, all
    resolved in parallel (_redirect_location), each within _redirect_timeout() and all of them
    within the time left before `deadline`. A link still unresolved then is given up (its thread
    ends at its own timeout; its result is not used)."""
    per_link = _redirect_timeout()
    if per_link <= 0:
        return {}, {u: "not resolved (GBENCH_SEARCH_REDIRECT_TIMEOUT_S is 0)" for u in links}
    budget = min(per_link, _remaining(deadline))
    if budget <= 0:
        return {}, {u: "no time left before the deadline" for u in links}
    done: Dict[str, Tuple[bool, str]] = {}

    def one(u: str) -> None:
        try:
            done[u] = (True, _redirect_location(u, budget))
        except Exception as e:  # any failure keeps the redirect link
            done[u] = (False, _mask_keys(f"{type(e).__name__}: {e}")[:200])

    threads = [threading.Thread(target=one, args=(u,), name="gbench-search-redirect",
                                daemon=True) for u in links]
    end = time.monotonic() + budget
    for t in threads:
        t.start()
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    got = dict(done)
    resolved = {u: v for u, (ok, v) in got.items() if ok}
    failed = {u: (got[u][1] if u in got else f"no reply within {budget:.1f} s")
              for u in links if u not in resolved}
    return resolved, failed


def _search_engine_results(sources: List[Dict[str, str]], max_results: int, deadline: float,
                           lookup: Dict[str, Any]) -> List[Dict[str, str]]:
    """The results of a search-engine lookup from the answering model's sources
    (gemini_search(search_engine=True)): grounding redirect links resolved to the source URL
    (_resolve_redirects; one that does not resolve is kept as the redirect link), duplicates of a
    URL dropped in order, at most max_results, each {title, url, snippet: ""}. The link counts
    and the reasons links did not resolve go into `lookup`."""
    links = list(dict.fromkeys(r["url"] for r in sources
                               if r.get("url", "").startswith(GROUNDING_REDIRECT_PREFIX)))
    resolved, failed = _resolve_redirects(links, deadline) if links else ({}, {})
    out: List[Dict[str, str]] = []
    seen: set = set()
    for r in sources:
        url = resolved.get(r["url"], r["url"])
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": r.get("title") or "result", "url": url, "snippet": ""})
        if r["url"] in resolved:
            lookup["links_resolved"] += 1
        elif r["url"] in failed:
            lookup["links_unresolved"] += 1
            lookup["unresolved_reasons"].append(failed[r["url"]])
        else:
            lookup["links_direct"] += 1
        if len(out) >= max_results:
            break
    return out


#: Which backend actually served each call this process. `search_backend_name()` reports
#: what was CONFIGURED; on 2026-08-20 that read `gemini-google-search-grounding` on a run
#: where every single lookup had 429'd. Only a tally of what answered can tell those apart.
_BACKEND_TALLY: Dict[str, int] = {}


def backend_tally() -> Dict[str, int]:
    """Calls served per backend since the last reset, e.g. `{"gemini:gemini-3.8-flash": 812}`."""
    return dict(_BACKEND_TALLY)


def reset_backend_tally() -> None:
    _BACKEND_TALLY.clear()


def search_backend_name() -> Optional[str]:
    if _gemini_key_set():
        return "gemini-grounding-cascade[" + ",".join(_search_cascade()) + "]"
    return None
