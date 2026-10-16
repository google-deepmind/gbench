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

"""A grounding lookup with a deadline hedges across the model cascade.

2026-10-10 smoke run (wildclawbench, run_61x1yysu): 04_Search_Retrieval_task_4 lost its score to 12
consecutive agent web searches that failed "deadline-passed", 12 different queries each failing at
27 s (the proxy's WEB_SEARCH_DEADLINE_S), while gemini-3.8-flash answered the task's other searches
in 3.7-14.6 s; gemini_usage.json showed no other model of the cascade ever asked. search_tool gave
the first model's request a timeout of all the time left, so one slow request used up the deadline
and the cascade never moved on. A lookup with a deadline now sends a hedged request: when the
running request has not answered within GBENCH_SEARCH_HEDGE_DELAY_S, the next model is asked in
parallel, and the first successful answer is the lookup's.

Review of the fix (2026-10-10): a request that lost the race while it was waiting for a rate-limiter
token (a busy limiter) still took one and sent its request after the lookup had been answered. The
wait for a token now stops, without one, once the lookup is answered.

Since 2026-10-12 a lookup with a deadline is a search-engine lookup (tests/test_search_engine_mode.py):
it asks two models at once and counts only an answer citing a source. These tests keep one model
at the start (GBENCH_SEARCH_INITIAL_WIDTH=1) to check the hedge itself, and the stub's answers cite
one source each.

No network: urlopen is a stub whose latency is set per model; the waits are real but short.
"""

import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import search_tool as ST
from gbench.runners.eval_suites import wildclawbench as W

# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
KEY = "AIza" + "HedgeFakeKey000000000000000000000001"


class _Grounded:
    def __init__(self, model):
        self._body = json.dumps({"candidates": [{"content": {"parts": [
            {"text": f"answer from {model}"}]}, "groundingMetadata": {"groundingChunks": [
                {"web": {"uri": f"https://{model}.example/", "title": model}}]}}]}).encode()

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


@pytest.fixture
def stub(monkeypatch):
    """urlopen by model: `plan[model]` is (latency s, outcome): outcome 200 answers, an int raises
    that HTTP status, "timeout" raises socket.timeout. A request waits min(latency, its timeout),
    so a hang ends at the deadline as a real urlopen does. Returns the state; the hedge threads
    are released and joined at teardown."""
    state = {"plan": {}, "calls": [], "threads": [], "limiter": _CountingLimiter()}
    release = threading.Event()

    def urlopen(req, timeout=None):
        model = re.search(r"models/([^:]+):", req.full_url).group(1)
        state["calls"].append(model)
        state["threads"].append(threading.current_thread().name)
        latency, outcome = state["plan"].get(model, (0.0, 200))
        if release.wait(min(latency, timeout)) or latency > timeout:
            raise socket.timeout("timed out")
        if outcome == "timeout":
            raise socket.timeout("timed out")
        if outcome != 200:
            raise urllib.error.HTTPError(req.full_url, outcome,
                                         f"status {outcome} for key {KEY}", {}, None)
        return _Grounded(model)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ST, "_RATE_LIMITER", state["limiter"])
    monkeypatch.setattr(ST, "_KEY_CYCLE", None)
    monkeypatch.setattr(ST, "_SEARCH_ATTEMPTS", 1)
    monkeypatch.setattr(ST, "_SEARCH_BACKOFF", 0.01)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "m1,m2,m3")
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "0.3")
    monkeypatch.setenv("GBENCH_SEARCH_INITIAL_WIDTH", "1")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GBENCH_GEMINI_RELAY_URL", raising=False)
    ST.reset_backend_tally()
    yield state
    release.set()
    for t in threading.enumerate():
        if t.name.startswith("gbench-search-"):
            t.join(timeout=5)
    ST.reset_backend_tally()


def _settle(seconds=0.5):
    """Let abandoned hedge threads run on, to show they send nothing more and tally nothing."""
    time.sleep(seconds)


def test_a_hanging_first_model_is_hedged_and_the_second_model_answers(stub):
    stub["plan"] = {"m1": (60.0, 200), "m2": (0.1, 200)}
    started = time.monotonic()
    results, backend = ST.search_with_fallback("q", deadline=started + 2.0)
    elapsed = time.monotonic() - started
    assert backend == "gemini:m2"
    assert results[0]["url"] == "https://m2.example/" and results[0]["snippet"] == ""
    assert 0.3 <= elapsed < 1.0, elapsed               # the hedge delay, then m2's latency
    assert stub["calls"] == ["m1", "m2"]               # m2 answered before m3's hedge was due
    assert stub["limiter"].taken == 2                  # one token per request, hedges included
    _settle()
    assert ST.backend_tally() == {"gemini:m2": 1}      # the abandoned m1 request is not tallied


def test_a_request_that_answers_before_the_hedge_delay_sends_no_hedge(stub):
    stub["plan"] = {"m1": (0.05, 200)}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 2.0)
    assert backend == "gemini:m1" and results[0]["url"] == "https://m1.example/"
    _settle()
    assert stub["calls"] == ["m1"] and stub["limiter"].taken == 1
    assert ST.backend_tally() == {"gemini:m1": 1}


def test_every_model_slow_is_a_deadline_failure_at_the_deadline(stub, monkeypatch):
    """Hedges at 0 and 0.4 s; at 0.8 s only 0.2 s is left, less than the hedge delay, so m3 is
    not asked. The lookup ends at the deadline, not after the hung requests."""
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "0.4")
    stub["plan"] = {m: (60.0, 200) for m in ("m1", "m2", "m3")}
    started = time.monotonic()
    results, backend = ST.search_with_fallback("q", deadline=started + 1.0)
    elapsed = time.monotonic() - started
    assert backend == "deadline-passed" and results[0]["title"] == "error"
    assert 1.0 <= elapsed < 1.25, elapsed
    assert stub["calls"] == ["m1", "m2"]
    assert ST.backend_tally() == {"deadline-passed": 1}


def test_a_429_moves_to_the_next_model_at_once(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "5")
    stub["plan"] = {"m1": (0.0, 429), "m2": (0.05, 200)}
    started = time.monotonic()
    _results, backend = ST.search_with_fallback("q", deadline=started + 3.0)
    assert backend == "gemini:m2"
    assert time.monotonic() - started < 1.0            # no wait for the 5 s hedge delay
    assert stub["calls"] == ["m1", "m2"]
    assert ST.backend_tally() == {"gemini:m2": 1}


def test_a_not_served_model_found_by_a_hedge_thread_is_not_asked_again(stub, monkeypatch):
    """The 404 is read from the hedge thread's _LAST_CALL and kept for the later rounds."""
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "2")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "m1,m2")
    stub["plan"] = {"m1": (0.0, 404), "m2": (0.0, 503)}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "all-grounding-models-failed"
    assert stub["calls"] == ["m1", "m2", "m2"]
    assert "503" in results[0]["snippet"]


def test_a_failed_hedged_lookup_masks_the_key(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "m1")
    stub["plan"] = {"m1": (0.0, 500)}
    results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "all-grounding-models-failed"
    assert KEY not in json.dumps(results) and "[REDACTED]" in results[0]["snippet"]


def test_an_abandoned_request_sends_no_retry(stub, monkeypatch):
    """m1 fails with a transient 503 after m2 has answered: it would retry in place, but its
    lookup is over, so it sends nothing more."""
    monkeypatch.setattr(ST, "_SEARCH_ATTEMPTS", 3)
    stub["plan"] = {"m1": (0.5, 503), "m2": (0.05, 200)}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "gemini:m2"
    _settle(1.0)
    assert stub["calls"].count("m1") == 1
    assert ST.backend_tally() == {"gemini:m2": 1}


class _TakeCountingLimiter(ST._RateLimiter):
    """The real token bucket (real waits), counting the tokens actually taken."""

    def __init__(self, rps, burst):
        super().__init__(rps, burst)
        self.taken = 0

    def acquire(self, deadline=None, cancel=None):
        got = super().acquire(deadline) if cancel is None \
            else super().acquire(deadline, cancel=cancel)
        self.taken += bool(got)
        return got


def test_a_hedge_still_waiting_for_a_token_when_the_lookup_is_answered_sends_nothing(
        stub, monkeypatch):
    """Busy limiter (2 rps, burst 1): the m2 hedge launched at 0.2 s waits for a token until
    0.5 s, and m1 answers at 0.35 s. The hedge stops waiting without a token and sends nothing
    (before the review it took the token and sent its request at 0.5 s)."""
    rl = _TakeCountingLimiter(2.0, 1.0)
    monkeypatch.setattr(ST, "_RATE_LIMITER", rl)
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "0.2")
    stub["plan"] = {"m1": (0.35, 200), "m2": (0.0, 200)}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "gemini:m1"
    _settle(1.0)
    assert stub["calls"] == ["m1"]
    assert rl.taken == 1
    assert ST.backend_tally() == {"gemini:m1": 1}


def test_an_abandoned_retry_waiting_for_a_token_sends_nothing(stub, monkeypatch):
    """Busy limiter (1 rps, burst 2): m1 fails with a transient 503 at 0.25 s and its in-place
    retry waits for a token until about 1.0 s; the m2 hedge answers at 0.3 s. The retry is not
    sent (before the review it was, at about 1.0 s)."""
    rl = _TakeCountingLimiter(1.0, 2.0)
    monkeypatch.setattr(ST, "_RATE_LIMITER", rl)
    monkeypatch.setattr(ST, "_SEARCH_ATTEMPTS", 3)
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "0.2")
    stub["plan"] = {"m1": (0.25, 503), "m2": (0.1, 200)}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 3.0)
    assert backend == "gemini:m2"
    _settle(1.5)
    assert stub["calls"] == ["m1", "m2"]
    assert rl.taken == 2
    assert ST.backend_tally() == {"gemini:m2": 1}


def test_the_rate_limiter_stops_waiting_without_a_token_once_cancelled():
    clock = [0.0]
    cancel = threading.Event()

    def sleep(s):
        clock[0] += s
        cancel.set()                   # the lookup is answered while this caller waits

    rl = ST._RateLimiter(1.0, 1.0, clock=lambda: clock[0], sleep=sleep)
    rl.tokens = 0.0
    assert rl.acquire(deadline=10.0, cancel=cancel) is False
    assert rl.tokens == pytest.approx(0.0)          # no token taken
    assert rl.acquire(deadline=10.0) is True        # the token is still there for others


def test_a_hedge_delay_of_zero_asks_the_models_one_after_another(stub, monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "0")
    assert ST._hedge_delay() is None
    stub["plan"] = {"m1": (60.0, 200), "m2": (0.0, 200)}
    _results, backend = ST.search_with_fallback("q", deadline=time.monotonic() + 0.6)
    assert backend == "deadline-passed" and stub["calls"] == ["m1"]


def test_a_lookup_without_a_deadline_never_hedges(stub):
    """gaia, deepsearch_qa and bfcl_v4_agentic: the models are asked in turn, on the caller's
    thread, as before."""
    stub["plan"] = {"m1": (0.6, 200)}
    results, backend = ST.search_with_fallback("q")
    assert backend == "gemini:m1"
    assert stub["calls"] == ["m1"]
    assert stub["threads"] == [threading.current_thread().name]


def test_the_hedge_delay_default_and_its_knob_are_wired(monkeypatch):
    monkeypatch.delenv("GBENCH_SEARCH_HEDGE_DELAY_S", raising=False)
    assert ST._hedge_delay() == ST._DEFAULT_HEDGE_DELAY_S == 12.0
    monkeypatch.setenv("GBENCH_SEARCH_HEDGE_DELAY_S", "7.5")
    assert ST._hedge_delay() == 7.5
    assert base.NON_RUN_SHAPING_ENV["GBENCH_SEARCH_HEDGE_DELAY_S"] == "infra"
    assert "GBENCH_SEARCH_HEDGE_DELAY_S" in W._FORWARDED_KNOBS
