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

"""A grounding lookup with a deadline is a search-engine lookup: sources only, no generated text.

Incident (reported 2026-10-12): wildclawbench's agent web_search and mcp_atlas' search tools stand
in for a search engine, but search_tool returned, as each result's snippet, the segment of the
Gemini model's grounded answer that cites the source (or the start of that answer, or the answer
itself as a "grounded-answer" result): text Gemini wrote. A fabricated pull-request number
(#30340) reached the agent that way. The same day an A/B on 5 recorded agent queries x 2 models
measured 2/10 answers within the proxy's 27 s deadline with the query as the prompt and 5/10 with
"Search the web and list the most relevant sources for: <query>", and 4/5 queries answered in time
when both models were asked at once.

A lookup with a deadline now: asks the first two cascade models at once with that prompt, counts
only an answer citing a source, returns {title, real URL (the grounding redirect resolved by its
Location header), snippet ""}, de-duplicated and capped, and reports what it did (last_lookup).
A lookup without a deadline (gaia, deepsearch_qa, bfcl_v4_agentic) is unchanged.

No network: the Gemini API (urllib.request.urlopen) and the redirect service (search_tool's
_REDIRECT_OPENER) are stubs; the waits are real but short.
"""

import email.message
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import search_tool as ST

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "gbench", "docker"))
import wildclawbench_cascade_judge as J  # noqa: E402

# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
KEY = "AIza" + "EngineFakeKey00000000000000000000001"
REDIRECT = ST.GROUNDING_REDIRECT_PREFIX
GENERATED = "Gemini wrote this: the fix landed in PR #30340"
SEGMENT = "a segment Gemini wrote that cites the source"


def _chunk(uri, title="site.example"):
    return {"web": {"uri": uri, "title": title}}


def _body(chunks, answer=GENERATED):
    """A generateContent reply: answer text, groundingSupports segments citing every chunk, and
    the chunks."""
    gm = {"groundingChunks": chunks,
          "groundingSupports": [{"segment": {"text": SEGMENT},
                                 "groundingChunkIndices": list(range(len(chunks)))}]}
    return {"candidates": [{"content": {"parts": [{"text": answer}]},
                            "groundingMetadata": gm}]}


class _Reply:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _CountingLimiter:
    """The global rate limiter, counting the tokens taken (no waits)."""

    def __init__(self):
        self.taken = 0
        self._lock = threading.Lock()

    def acquire(self, deadline=None, cancel=None):
        with self._lock:
            self.taken += 1
        return True


class _RedirectService:
    """The grounding redirect service: `targets[url]` is a URL (302 with that Location), an int
    (that HTTP status, no Location), "hang" (no reply before the request's timeout) or "200"."""

    def __init__(self):
        self.targets = {}
        self.requests = []
        self.release = threading.Event()

    def open(self, req, timeout=None):
        self.requests.append({"url": req.full_url, "method": req.get_method(),
                              "headers": dict(req.header_items()), "timeout": timeout})
        target = self.targets.get(req.full_url, "hang")
        if target == "hang":
            self.release.wait(timeout)
            raise socket.timeout("timed out")
        if target == "200":
            return _Reply({})
        headers = email.message.Message()
        if isinstance(target, str):
            headers["Location"] = target
            raise urllib.error.HTTPError(req.full_url, 302, "Found", headers, None)
        raise urllib.error.HTTPError(req.full_url, target, "status", headers, None)


@pytest.fixture
def stub(monkeypatch):
    """urlopen by model: `plan[model]` is (latency s, outcome): outcome a dict is the reply body,
    an int raises that HTTP status. Each call is recorded with its model, time, thread and the
    request payload."""
    state = {"plan": {}, "calls": [], "limiter": _CountingLimiter(),
             "redirects": _RedirectService()}
    release = threading.Event()

    def urlopen(req, timeout=None):
        model = re.search(r"models/([^:]+):", req.full_url).group(1)
        state["calls"].append({"model": model, "at": time.monotonic(),
                               "thread": threading.current_thread().name,
                               "payload": json.loads(req.data.decode()),
                               "headers": dict(req.header_items())})
        latency, outcome = state["plan"].get(model, (0.0, _body([])))
        if release.wait(min(latency, timeout)) or latency > timeout:
            raise socket.timeout("timed out")
        if isinstance(outcome, int):
            raise urllib.error.HTTPError(req.full_url, outcome,
                                         f"status {outcome} for key {KEY}", {}, None)
        return _Reply(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ST, "_RATE_LIMITER", state["limiter"])
    monkeypatch.setattr(ST, "_REDIRECT_OPENER", state["redirects"])
    monkeypatch.setattr(ST, "_KEY_CYCLE", None)
    monkeypatch.setattr(ST, "_SEARCH_ATTEMPTS", 1)
    monkeypatch.setattr(ST, "_SEARCH_BACKOFF", 0.01)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "m1,m2,m3")
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "5")
    monkeypatch.delenv("GBENCH_SEARCH_INITIAL_WIDTH", raising=False)
    monkeypatch.delenv("GBENCH_SEARCH_REDIRECT_TIMEOUT_S", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GBENCH_GEMINI_RELAY_URL", raising=False)
    ST.reset_backend_tally()
    yield state
    release.set()
    state["redirects"].release.set()
    for t in threading.enumerate():
        if t.name.startswith("gbench-search-"):
            t.join(timeout=5)
    ST.reset_backend_tally()


def _models(state):
    return [c["model"] for c in state["calls"]]


def _settle(seconds=0.5):
    """Let abandoned requests run on, to show they send nothing more and tally nothing."""
    time.sleep(seconds)


def _no_generated_text(results):
    text = json.dumps(results)
    return GENERATED not in text and SEGMENT not in text and "grounded-answer" not in text


# --------------------------------------------------------------------------- #
# the request: the sources prompt, two models at once
# --------------------------------------------------------------------------- #
def test_both_first_models_are_asked_at_once_with_the_sources_prompt(stub):
    stub["plan"] = {"m1": (0.3, _body([_chunk("https://a.example/1")])),
                    "m2": (0.3, _body([_chunk("https://b.example/1")]))}
    started = time.monotonic()
    _results, backend = ST.search_with_fallback('who wrote "pathlib"', deadline=started + 3.0)
    assert backend in ("gemini:m1", "gemini:m2")
    assert sorted(_models(stub)) == ["m1", "m2"]
    assert all(c["at"] - started < 0.15 for c in stub["calls"])        # both at t=0
    assert len({c["thread"] for c in stub["calls"]}) == 2
    for call in stub["calls"]:
        assert call["payload"] == {
            "contents": [{"parts": [{"text": ST.SEARCH_ENGINE_PROMPT_PREFIX
                                     + 'who wrote "pathlib"'}]}],
            "tools": [{"google_search": {}}]}                          # no thinkingConfig
    assert ST.SEARCH_ENGINE_PROMPT_PREFIX == \
        "Search the web and list the most relevant sources for: "
    assert stub["limiter"].taken == 2                                  # one token per request


def test_the_initial_width_knob(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_INITIAL_WIDTH", "1")
    stub["plan"] = {"m1": (0.1, _body([_chunk("https://a.example/1")]))}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    _settle(0.2)
    assert backend == "gemini:m1" and _models(stub) == ["m1"]
    monkeypatch.setenv("GBENCH_SEARCH_INITIAL_WIDTH", "0")
    assert ST._initial_width() == 1
    monkeypatch.delenv("GBENCH_SEARCH_INITIAL_WIDTH")
    assert ST._initial_width() == ST._DEFAULT_INITIAL_WIDTH == 2


def test_a_third_model_is_hedged_after_the_delay(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "0.3")
    stub["plan"] = {"m1": (60.0, _body([_chunk("https://a.example/")])),
                    "m2": (60.0, _body([_chunk("https://b.example/")])),
                    "m3": (0.05, _body([_chunk("https://c.example/")]))}
    started = time.monotonic()
    results, backend = ST.search_with_fallback("q", deadline=started + 3.0)
    assert backend == "gemini:m3" and results[0]["url"] == "https://c.example/"
    [m3] = [c for c in stub["calls"] if c["model"] == "m3"]
    assert 0.3 <= m3["at"] - started < 0.6


# --------------------------------------------------------------------------- #
# which answer counts
# --------------------------------------------------------------------------- #
def test_the_first_answer_with_a_source_wins_and_a_chunkless_answer_moves_on(stub):
    """m1 answers first but cites no source: a failure of m1, so m3 is asked at once (no wait
    for the 5 s hedge delay); m2's answer with a source is the lookup's."""
    stub["plan"] = {"m1": (0.05, _body([])),
                    "m2": (0.3, _body([_chunk("https://b.example/")])),
                    "m3": (2.0, _body([_chunk("https://c.example/")]))}
    started = time.monotonic()
    results, backend = ST.search_with_fallback("q", deadline=started + 3.0)
    assert backend == "gemini:m2" and time.monotonic() - started < 1.0
    assert results == [{"title": "site.example", "url": "https://b.example/", "snippet": ""}]
    assert _models(stub)[:2] in (["m1", "m2"], ["m2", "m1"]) and _models(stub)[2] == "m3"
    lookup = ST.last_lookup()
    assert lookup["search_mode"] == "search-engine"
    assert lookup["answered_by"] == "m2" and lookup["answers_without_source"] == 1
    assert sorted(lookup["models_asked"]) == ["m1", "m2", "m3"]
    _settle()
    assert ST.backend_tally() == {"gemini:m2": 1}                    # the answering model


def test_chunkless_answers_from_every_model_are_a_failed_lookup_not_an_answer(stub):
    stub["plan"] = {m: (0.0, _body([])) for m in ("m1", "m2", "m3")}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "all-grounding-models-failed"
    assert results == [{"title": "error", "url": "", "snippet": ST._NO_SOURCES}]
    assert _no_generated_text(results)
    lookup = ST.last_lookup()
    assert lookup["answered_by"] is None and lookup["answers_without_source"] == 3
    assert ST.backend_tally() == {"all-grounding-models-failed": 1}


def test_a_429_hands_over_at_once_and_the_key_is_masked(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "m1,m2")
    stub["plan"] = {"m1": (0.0, 429), "m2": (0.0, 500)}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "all-grounding-models-failed"
    assert KEY not in json.dumps(results) and "[REDACTED]" in results[0]["snippet"]
    assert KEY not in json.dumps(ST.last_lookup())


# --------------------------------------------------------------------------- #
# the results: real URLs, no generated text, de-duplicated, capped
# --------------------------------------------------------------------------- #
def test_no_generated_text_reaches_the_results(stub):
    stub["plan"] = {"m1": (0.0, _body([_chunk("https://a.example/", "a.example"),
                                       _chunk("https://b.example/", "b.example")])),
                    "m2": (1.0, _body([]))}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "gemini:m1"
    assert results == [{"title": "a.example", "url": "https://a.example/", "snippet": ""},
                       {"title": "b.example", "url": "https://b.example/", "snippet": ""}]
    assert _no_generated_text(results)
    assert ST.last_lookup()["links_direct"] == 2


def test_redirect_links_resolve_to_the_source_url_by_the_location_header(stub):
    stub["redirects"].targets = {REDIRECT + "t1": "https://docs.python.org/3/library/pathlib.html",
                                 REDIRECT + "t2": "https://treyhunner.com/2018/12/pathlib/"}
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + "t1", "python.org"),
                                       _chunk(REDIRECT + "t2", "treyhunner.com")])),
                    "m2": (1.0, _body([]))}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "gemini:m1"
    assert [r["url"] for r in results] == ["https://docs.python.org/3/library/pathlib.html",
                                           "https://treyhunner.com/2018/12/pathlib/"]
    assert all(r["snippet"] == "" for r in results)
    reqs = stub["redirects"].requests
    assert sorted(r["url"] for r in reqs) == [REDIRECT + "t1", REDIRECT + "t2"]
    assert all(r["method"] == "HEAD" for r in reqs)
    # no key, and no other header of the grounding request, goes to the redirect service
    assert all(not any(h.lower() == "x-goog-api-key" for h in r["headers"]) for r in reqs)
    assert KEY not in json.dumps(reqs)
    lookup = ST.last_lookup()
    assert (lookup["links_resolved"], lookup["links_unresolved"]) == (2, 0)
    assert stub["limiter"].taken == 2          # the two grounding requests; HEADs take no token


def test_a_link_that_does_not_resolve_is_kept_as_the_redirect_link_and_recorded(stub):
    stub["redirects"].targets = {REDIRECT + "ok": "https://a.example/",
                                 REDIRECT + "gone": 404, REDIRECT + "plain": "200",
                                 REDIRECT + "ftp": "ftp://x.example/"}
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + n) for n in
                                       ("ok", "gone", "plain", "ftp")])),
                    "m2": (1.0, _body([]))}
    results, _backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert [r["url"] for r in results] == ["https://a.example/", REDIRECT + "gone",
                                           REDIRECT + "plain", REDIRECT + "ftp"]
    lookup = ST.last_lookup()
    assert (lookup["links_resolved"], lookup["links_unresolved"]) == (1, 3)
    reasons = " | ".join(lookup["unresolved_reasons"])
    assert "HTTP 404" in reasons and "without a redirect" in reasons and "Location" in reasons


def test_redirect_resolution_stops_at_the_deadline(stub):
    """The redirect service hangs: the lookup still ends at its deadline, with the redirect
    link, not after the 3 s per-link timeout."""
    stub["plan"] = {"m1": (0.1, _body([_chunk(REDIRECT + "slow")])), "m2": (5.0, _body([]))}
    started = time.monotonic()
    results, backend = ST.search_with_fallback("q", deadline=started + 0.6)
    elapsed = time.monotonic() - started
    assert backend == "gemini:m1" and results[0]["url"] == REDIRECT + "slow"
    assert 0.5 <= elapsed < 0.85, elapsed
    assert stub["redirects"].requests[0]["timeout"] <= 0.5
    lookup = ST.last_lookup()
    assert lookup["links_unresolved"] == 1 and "within" in lookup["unresolved_reasons"][0]


def test_redirect_resolution_stops_at_the_per_link_timeout(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_REDIRECT_TIMEOUT_S", "0.3")
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + "slow")])), "m2": (5.0, _body([]))}
    started = time.monotonic()
    results, _backend = ST.search_with_fallback("q", deadline=started + 5.0)
    assert results[0]["url"] == REDIRECT + "slow"
    assert time.monotonic() - started < 0.6


def test_results_are_deduplicated_in_order_and_capped(stub):
    stub["redirects"].targets = {REDIRECT + "a1": "https://a.example/",
                                 REDIRECT + "a2": "https://a.example/",   # same page
                                 REDIRECT + "b": "https://b.example/",
                                 REDIRECT + "c": "https://c.example/"}
    chunks = [_chunk(REDIRECT + "a1", "A"), _chunk(REDIRECT + "a1", "A again"),
              _chunk(REDIRECT + "a2", "A too"), _chunk(REDIRECT + "b", "B"),
              _chunk(REDIRECT + "c", "C")]
    stub["plan"] = {"m1": (0.0, _body(chunks)), "m2": (1.0, _body([]))}
    results, _ = ST.search_with_fallback("q", max_results=2, deadline=time.monotonic() + 3.0)
    assert results == [{"title": "A", "url": "https://a.example/", "snippet": ""},
                       {"title": "B", "url": "https://b.example/", "snippet": ""}]
    assert len(stub["redirects"].requests) == 4        # each distinct redirect link once


# --------------------------------------------------------------------------- #
# abandoned requests
# --------------------------------------------------------------------------- #
def test_abandoned_requests_send_nothing_after_the_lookup_is_answered(stub, monkeypatch):
    """m2 fails with a transient 503 after m1 has answered: it would retry in place, but its
    lookup is over, so it sends nothing more, and m3 is never asked."""
    monkeypatch.setattr(ST, "_SEARCH_ATTEMPTS", 3)
    stub["plan"] = {"m1": (0.05, _body([_chunk("https://a.example/")])), "m2": (0.3, 503)}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "gemini:m1"
    _settle(1.0)
    assert sorted(_models(stub)) == ["m1", "m2"]
    assert stub["limiter"].taken == 2
    assert ST.backend_tally() == {"gemini:m1": 1}


# --------------------------------------------------------------------------- #
# a lookup without a deadline is unchanged
# --------------------------------------------------------------------------- #
def test_a_lookup_without_a_deadline_is_unchanged(stub):
    """gaia, deepsearch_qa, bfcl_v4_agentic: the query as the prompt, the models in turn on the
    caller's thread, the Gemini-written segment as each snippet, the grounded-answer fallback,
    no redirect resolved, and last_lookup() None. The expected values are what search_tool
    returned before 2026-10-12 for the same replies."""
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + "t1", "a.example"),
                                       _chunk(REDIRECT + "t2", "b.example")]))}
    results, backend = ST.search_with_fallback("q")
    assert backend == "gemini:m1"
    assert results == [{"title": "a.example", "url": REDIRECT + "t1", "snippet": SEGMENT},
                       {"title": "b.example", "url": REDIRECT + "t2", "snippet": SEGMENT}]
    assert stub["calls"][0]["payload"] == {"contents": [{"parts": [{"text": "q"}]}],
                                           "tools": [{"google_search": {}}]}
    assert stub["calls"][0]["thread"] == threading.current_thread().name
    assert stub["redirects"].requests == []
    assert ST.last_lookup() is None
    stub["plan"] = {"m1": (0.0, _body([], answer="grounded prose"))}
    results, backend = ST.search_with_fallback("q")
    assert results == [{"title": "grounded-answer", "url": "", "snippet": "grounded prose"}]
    stub["plan"] = {"m1": (0.0, {"candidates": [{"content": {"parts": [{"text": "A" * 600}]},
                                                 "groundingMetadata": {"groundingChunks": [
                                                     _chunk("https://a.example/")]}}]})}
    results, _ = ST.search_with_fallback("q")
    assert results == [{"title": "site.example", "url": "https://a.example/",
                        "snippet": "A" * 500}]
    assert _models(stub) == ["m1", "m1", "m1"]


def test_last_lookup_is_cleared_by_a_lookup_without_a_deadline(stub):
    stub["plan"] = {"m1": (0.0, _body([_chunk("https://a.example/")])), "m2": (1.0, _body([]))}
    ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert ST.last_lookup()["answered_by"] == "m1"
    ST.search_with_fallback("q")
    assert ST.last_lookup() is None


# --------------------------------------------------------------------------- #
# the callers record it
# --------------------------------------------------------------------------- #
def test_wildclawbench_proxy_returns_sources_only_and_records_the_lookup(stub, monkeypatch,
                                                                          tmp_path):
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    J.reset_usage()
    monkeypatch.setitem(J._SEARCH_TOOL, "module", ST)
    stub["redirects"].targets = {REDIRECT + "t1": "https://a.example/page"}
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + "t1", "a.example"),
                                       _chunk(REDIRECT + "t2", "b.example")])),
                    "m2": (2.0, _body([]))}
    started = time.monotonic()
    code, body = J.brave_search_reply("q=pathlib&count=5", "172.17.0.4", started=started)
    assert code == 200
    assert body["web"]["results"] == [
        {"title": "a.example", "url": "https://a.example/page", "description": ""},
        {"title": "b.example", "url": REDIRECT + "t2", "description": ""}]
    assert GENERATED not in json.dumps(body) and SEGMENT not in json.dumps(body)
    assert stub["calls"][0]["payload"]["contents"][0]["parts"][0]["text"] == \
        ST.SEARCH_ENGINE_PROMPT_PREFIX + "pathlib"
    usage = J.usage_snapshot()["agent_web_search_engine"]
    assert usage["lookups"] == usage["answered"] == 1 and usage["models_asked"] == 2
    assert usage["links_resolved"] == 1 and usage["links_unresolved"] == 1
    J.settle_web_search("172.17.0.4", 5000.0, started, code, True, now=started + 1.0)
    [answer] = [json.loads(x) for x in (tmp_path / J.GEMINI_ANSWERS).read_text().splitlines()]
    assert answer["model"] == "m1" and answer["search"]["search_mode"] == "search-engine"
    assert answer["search"]["links_resolved"] == 1
    J.reset_usage()


def test_wildclawbench_failed_lookup_names_the_models_asked(stub, monkeypatch, tmp_path):
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "m1,m2")
    J.reset_usage()
    monkeypatch.setitem(J._SEARCH_TOOL, "module", ST)
    stub["plan"] = {"m1": (0.0, _body([])), "m2": (0.0, _body([]))}
    code, body = J.brave_search_reply("q=x", "172.17.0.4")
    assert code == 502
    [ev] = [json.loads(x) for x in (tmp_path / J.HARNESS_EVENTS).read_text().splitlines()]
    assert ev["kind"] == "agent_web_search" and "models asked: m" in ev["detail"]
    assert "cited no web source" in ev["detail"]
    # The agent sees the failure, never gbench's model list (2026-10-12 review).
    assert "cited no web source" in body["error"]["detail"]
    assert "models asked" not in json.dumps(body) and "m1" not in json.dumps(body)
    usage = J.usage_snapshot()["agent_web_search_engine"]
    assert usage["lookups"] == 1 and "answered" not in usage
    assert usage["answers_without_source"] == 2
    J.reset_usage()


def test_mcp_atlas_serves_sources_only_and_records_the_lookup(stub):
    stub["redirects"].targets = {REDIRECT + "t1": "https://a.example/page"}
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + "t1", "a.example")])),
                    "m2": (2.0, _body([]))}
    info = {}
    status, body, backends = M.gemini_search_reply(
        "exa_web_search_exa", {"query": "pathlib"}, deadline=time.monotonic() + 3.0, info=info)
    assert status == 200 and backends == ["gemini:m1"]
    text = json.loads(body)[0]["text"]
    assert text.startswith("Title: a.example\nURL: https://a.example/page\n")
    assert GENERATED not in text and SEGMENT not in text
    [lookup] = info["lookups"]
    assert lookup["search_mode"] == "search-engine" and lookup["links_resolved"] == 1
    counts = dict.fromkeys(M.SEARCH_ENGINE_COUNTS, 0)
    M._count_search_lookup(counts, lookup)
    assert counts == {"lookups": 1, "answered": 1, "models_asked": 2,
                      "answers_without_source": 0, "links_resolved": 1,
                      "links_unresolved": 0, "links_direct": 0}
    assert "Gemini grounding search returning sources only, no generated text" in \
        M.SEARCH_ENGINE_RULE


def test_a_redirect_timeout_of_zero_resolves_no_link(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_REDIRECT_TIMEOUT_S", "0")
    stub["redirects"].targets = {REDIRECT + "t1": "https://a.example/"}
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + "t1")])), "m2": (1.0, _body([]))}
    results, _ = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert results[0]["url"] == REDIRECT + "t1" and stub["redirects"].requests == []
    assert "GBENCH_SEARCH_REDIRECT_TIMEOUT_S" in ST.last_lookup()["unresolved_reasons"][0]


# --------------------------------------------------------------------------- #
# 2026-10-12 devil's-advocate review
# --------------------------------------------------------------------------- #
def test_a_query_no_model_cites_a_source_for_asks_each_model_once_across_rounds(stub,
                                                                                monkeypatch):
    """Review finding: a sourceless answer was retried like a 429 burst in every cascade round,
    15 requests (5 models x 3 rounds) for one query that then failed anyway. A model that
    answered without a source is now not asked again in a later round."""
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "3")
    stub["plan"] = {m: (0.0, _body([])) for m in ("m1", "m2", "m3")}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "all-grounding-models-failed"
    assert results == [{"title": "error", "url": "", "snippet": ST._NO_SOURCES}]
    assert sorted(_models(stub)) == ["m1", "m2", "m3"]          # len(cascade), not x rounds
    assert stub["limiter"].taken == 3
    assert ST.last_lookup()["answers_without_source"] == 3


def test_later_rounds_retry_only_the_transient_failures(stub, monkeypatch):
    """m1 and m3 answer without a source, m2 is throttled (429): round 2 asks only m2 again."""
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "2")
    stub["plan"] = {"m1": (0.0, _body([])), "m2": (0.0, 429), "m3": (0.0, _body([]))}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "all-grounding-models-failed"
    assert sorted(_models(stub)) == ["m1", "m2", "m2", "m3"]
    assert ST.last_lookup()["models_asked"].count("m2") == 2


@pytest.mark.parametrize("knob,value", [("GBENCH_SEARCH_REDIRECT_TIMEOUT_S", "3s"),
                                        ("GBENCH_SEARCH_INITIAL_WIDTH", "two"),
                                        ("GBENCH_SEARCH_HEDGE_DELAY_S", "12s")])
def test_a_malformed_knob_fails_before_any_request(stub, monkeypatch, knob, value):
    """Review finding: GBENCH_SEARCH_REDIRECT_TIMEOUT_S=3s raised in _search_engine_results after
    a model answered, so last_lookup() named an answering model for a search the proxy returned
    HTTP 502 for. The knobs are now read before any request, and the error names the knob."""
    monkeypatch.setenv(knob, value)
    stub["plan"] = {"m1": (0.0, _body([_chunk("https://a.example/")]))}
    with pytest.raises(ValueError, match=knob):
        ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert stub["calls"] == [] and ST.last_lookup() is None and ST.backend_tally() == {}


def test_answered_by_is_set_only_once_the_results_are_built(stub, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("results could not be built")

    monkeypatch.setattr(ST, "_search_engine_results", broken)
    stub["plan"] = {"m1": (0.0, _body([_chunk("https://a.example/")])), "m2": (1.0, _body([]))}
    with pytest.raises(RuntimeError):
        ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert ST.last_lookup()["answered_by"] is None and ST.backend_tally() == {}


def test_a_hedge_delay_of_zero_still_asks_the_initial_width_at_once(stub, monkeypatch):
    """As env-knobs.md says: delay 0 disables only the timed hedge; strictly one model after
    another needs GBENCH_SEARCH_INITIAL_WIDTH=1 as well."""
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "0")
    stub["plan"] = {"m1": (0.3, _body([_chunk("https://a.example/")])),
                    "m2": (0.3, _body([_chunk("https://b.example/")]))}
    started = time.monotonic()
    ST.search_with_fallback("q", deadline=started + 3.0)
    assert sorted(_models(stub)) == ["m1", "m2"]
    assert all(c["at"] - started < 0.15 for c in stub["calls"])
    stub["calls"].clear()
    monkeypatch.setenv("GBENCH_SEARCH_INITIAL_WIDTH", "1")
    stub["plan"] = {"m1": (0.0, _body([])), "m2": (0.0, _body([_chunk("https://b.example/")]))}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "gemini:m2" and _models(stub) == ["m1", "m2"]


def test_an_overlong_location_keeps_the_redirect_link(stub):
    stub["redirects"].targets = {REDIRECT + "t1": "https://a.example/" + "x" * 2100}
    stub["plan"] = {"m1": (0.0, _body([_chunk(REDIRECT + "t1")])), "m2": (1.0, _body([]))}
    results, _ = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert results[0]["url"] == REDIRECT + "t1"
    assert "2048" in ST.last_lookup()["unresolved_reasons"][0]
