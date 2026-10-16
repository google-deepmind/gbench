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

"""The web-research suites must be able to actually search.

`gaia` and `deepsearch_qa` scored 0/20 on the 2026-08-15 sweep because they were run with
no tool at all - GAIA literally replied "I do not have access to the internet". These tests
cover the `web_search` tool, the agentic loop that drives it, and the refusal to report a
structural zero when no search backend is configured. Nothing here touches the network.
"""

import asyncio
import json
import os
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, search_tool


# --------------------------------------------------------------------------- #
# the tool itself
# --------------------------------------------------------------------------- #
def test_tool_schema_is_a_valid_openai_function():
    fn = search_tool.WEB_SEARCH_TOOL
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "web_search"
    params = fn["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["query"]["type"] == "string"
    assert params["required"] == ["query"]


def _grounding_payload():
    return {"candidates": [{
        "content": {"parts": [{"text": "Three species appear on camera."}]},
        "groundingMetadata": {
            "groundingChunks": [{"web": {"title": "birds.org", "uri": "https://birds.org/a"}},
                                {"web": {"domain": "audubon.org", "uri": "https://a.org/b"}}],
            "groundingSupports": [
                {"segment": {"text": "Three species appear"}, "groundingChunkIndices": [0]}],
        }}]}


def _with_urlopen(payload):
    class _Resp:
        def read(self): return json.dumps(payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return mock.patch("urllib.request.urlopen", return_value=_Resp())


def test_search_maps_grounding_chunks_to_results():
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), _with_urlopen(_grounding_payload()):
        got = search_tool.gemini_search("how many bird species")
    assert [r["title"] for r in got] == ["birds.org", "audubon.org"]
    assert got[0]["snippet"] == "Three species appear"      # from groundingSupports
    assert got[1]["snippet"].startswith("Three species")     # falls back to the answer text
    assert got[0]["url"] == "https://birds.org/a"


def test_search_with_no_chunks_falls_back_to_the_grounded_answer():
    payload = {"candidates": [{"content": {"parts": [{"text": "The answer is 42."}]}}]}
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), _with_urlopen(payload):
        got = search_tool.gemini_search("q")
    assert len(got) == 1 and "42" in got[0]["snippet"]


def test_backend_failure_is_reported_as_a_failure_not_as_no_results():
    """"No results" tells the model the fact does not exist; that is a different claim."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), \
         mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
        got = search_tool.gemini_search("q")
    assert got[0]["title"] == "error" and "failed" in got[0]["snippet"]


def test_executor_rejects_unknown_tools_and_empty_queries():
    assert "unknown tool" in asyncio.run(search_tool.execute_tool("rm_rf", {"query": "x"}))
    assert "non-empty" in asyncio.run(search_tool.execute_tool("web_search", {"query": "  "}))


def test_executor_returns_json_the_model_can_read(monkeypatch):
    # The key must be set: `execute_tool` routes through `search_with_fallback`, which
    # picks the DDG branch when there is no key - and would then hit the real network.
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    for knob in ("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS",
                 "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(knob, raising=False)
    with mock.patch.object(search_tool, "gemini_search",
                           return_value=[{"title": "t", "url": "u", "snippet": "s"}]):
        out = json.loads(asyncio.run(search_tool.execute_tool("web_search", {"query": "q"})))
    assert out["query"] == "q" and out["results"][0]["title"] == "t"
    # The model is told which grounding model served it, so a cascade hop is never invisible.
    assert out["backend"] == f"gemini:{base.GEMINI_CASCADE[0]}"


# --------------------------------------------------------------------------- #
# the agentic loop
# --------------------------------------------------------------------------- #
def _run_loop(replies, executor, **kw):
    seen = {"n": 0, "convos": []}

    async def _fake_send(**k):
        seen["convos"].append(list(k["messages"]))
        r = replies[min(seen["n"], len(replies) - 1)]
        seen["n"] += 1
        return r

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="gaia", model_name="m", base_url="http://x", concurrency=1,
            samples=[([{"role": "user", "content": "q"}], "3", {})],
            eval_fn=lambda resp, g: g in (resp or ""), tool_executor=executor, **kw)
    return res, seen


_ASK = base.Reply(text="", tool_calls=[{"id": "c1", "function": {
    "name": "web_search", "arguments": '{"query":"birds"}'}}], finish_reason="tool_calls")
_ANSWER = base.Reply(text="FINAL ANSWER: 3", tool_calls=None, finish_reason="stop")


def test_tool_call_is_executed_and_fed_back():
    ran = []

    async def ex(name, args):
        ran.append((name, args))
        return '{"results":[{"snippet":"three"}]}'

    res, seen = _run_loop([_ASK, _ANSWER], ex)
    assert ran == [("web_search", {"query": "birds"})]
    assert res["correct_answers"] == 1
    assert res["tool_rounds"]["samples_using_tools"] == 1
    assert res["tool_rounds"]["total_rounds"] == 1
    assert res["tool_rounds"]["forced_final_answers"] == 0   # it answered on its own
    # the follow-up request carried the assistant turn AND the tool result
    roles = [m["role"] for m in seen["convos"][1]]
    assert roles == ["user", "assistant", "tool"]


def test_loop_is_bounded():
    """A model that only ever asks to search must not spin forever.

    Since 2026-08-18 the last turn is a FORCED FINAL: exhausting the round budget used to
    return a reply whose only content was a tool call, which was flattened to text and
    graded as the answer (browsecomp scored a 183-char `web_search({...})` against the
    gold). One extra request, with the tools withdrawn, makes the model commit."""
    async def ex(name, args):
        return "{}"
    res, seen = _run_loop([_ASK], ex, max_tool_rounds=3)
    assert seen["n"] == 5                       # initial + 3 tool rounds + forced final
    assert res["tool_rounds"]["total_rounds"] == 3
    assert res["tool_rounds"]["forced_final_answers"] == 1
    # the forced turn must not offer tools again
    assert seen["convos"][-1][-1]["role"] == "user"
    assert "No further tool calls" in seen["convos"][-1][-1]["content"]


def test_a_raising_tool_is_reported_to_the_model_not_crashed_on():
    async def ex(name, args):
        raise RuntimeError("network down")
    res, seen = _run_loop([_ASK, _ANSWER], ex)
    tool_msg = [m for m in seen["convos"][1] if m["role"] == "tool"][0]
    assert "RuntimeError" in tool_msg["content"]
    assert res["status"] != "error"


def test_single_turn_suites_are_untouched():
    """No executor -> exactly one request, no tool bookkeeping."""
    res, seen = _run_loop([_ANSWER], None)
    assert seen["n"] == 1
    assert res["tool_rounds"] is None


# --------------------------------------------------------------------------- #
# no search backend -> skip, never a structural zero
# --------------------------------------------------------------------------- #
def test_gaia_hard_errors_without_a_backend(monkeypatch):
    """No-skip policy: gaia REQUIRES the GEMINI-backed search backend (grounding is the only
    backend now - the DDG fallback was removed 2026-08-21), so without the key it hard-errors
    (infra_required) up front, never a silent skip or structural zero. HF prereqs are stubbed so
    the test isolates the search-backend gate."""
    import pytest, importlib
    mod = importlib.import_module("gbench.runners.eval_suites.gaia")
    monkeypatch.setattr(mod, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        mod.run_gaia("m", "http://x", concurrency=1, limit=2)


def test_deepsearch_qa_hard_errors_without_a_backend(monkeypatch):
    """No-skip policy: deepsearch_qa REQUIRES the GEMINI-backed search+judge, so without the key
    it hard-errors (infra_required) up front - never a silent skip or structural zero."""
    import importlib
    mod = importlib.import_module("gbench.runners.eval_suites.deepsearch_qa")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="docs/evals/deepsearch_qa.md"):
        mod.run_deepsearch_qa("m", "http://x", concurrency=1, limit=2)


def test_search_backend_name_is_the_model_cascade(monkeypatch):
    """The backend name lists the grounding cascade, so a result records exactly which models
    could have served it."""
    from gbench.runners.eval_suites import search_tool
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    for knob in ("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS",
                 "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(knob, raising=False)
    assert search_tool.search_available() is True
    name = search_tool.search_backend_name()
    assert name == "gemini-grounding-cascade[" + ",".join(base.GEMINI_CASCADE) + "]"
    assert "gemini-2.5-flash" not in name
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert search_tool.search_available() is False
    assert search_tool.search_backend_name() is None


@pytest.mark.parametrize("suite", ["gaia", "deepsearch_qa"])
def test_web_research_suites_offer_the_tool_and_disclaim_comparability(suite):
    import inspect, importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    src = inspect.getsource(getattr(mod, f"run_{suite}"))
    # search_tool's web_search tool, described as what it returns (Gemini-written grounded
    # answers), and its executor with each call recorded on the sample's trace
    assert "GROUNDED_SEARCH_TOOL" in src and "tool_executor=record_web_search" in src
    from gbench.runners.eval_suites import gaia as G
    from gbench.runners.eval_suites.search_tool import WEB_SEARCH_TOOL
    assert G.GROUNDED_SEARCH_TOOL["function"]["name"] == WEB_SEARCH_TOOL["function"]["name"]
    assert G.GROUNDED_SEARCH_TOOL["function"]["parameters"] == \
        WEB_SEARCH_TOOL["function"]["parameters"]
    # search-only is not the browsing agent the public leaderboard uses
    assert 'result["leaderboard_comparable"] = False' in src


@pytest.mark.parametrize("suite", ["browsecomp"])
def test_browsecomp_is_closed_book_no_tools(suite):
    """Canonical BrowseComp (openai/simple-evals) declares NO tools and runs a single turn.
    gbench matches that: no injected web_search, no tool loop. (An earlier build injected a
    Gemini-grounded search because a non-browsing model scored 0/20; that deviated from the
    protocol and was removed - a low closed-book score is the correct result, not a bug.)"""
    import importlib, inspect
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    src = inspect.getsource(getattr(mod, f"run_{suite}"))
    assert "WEB_SEARCH_TOOL" not in src, "no injected search tool - canonical is closed-book"
    assert "tool_executor" not in src, "no tool loop - canonical BrowseComp is single-turn"
    # Only the Gemini grader (vs upstream gpt-4.1) still deviates, so it stays non-comparable.
    assert 'result["leaderboard_comparable"] = False' in src


# --------------------------------------------------------------------------- #
# mcp_atlas: only the release's search-engine servers are served by Gemini grounding
# --------------------------------------------------------------------------- #
def test_mcp_atlas_grounds_only_the_release_search_engine_tools():
    """The suite once answered any search-like MCP tool with gbench's web search and every
    other tool with an error. Now the release's MCP servers answer their own tools, and only
    the four search-engine servers (brave-search, ddg-search, exa, oxylabs) are served by
    Gemini Google Search grounding, under their own tool names; the generic web_search tool
    is never offered."""
    from gbench.runners.eval_suites import mcp_atlas as M
    assert set(M.GEMINI_SEARCH_SERVERS) == {"brave-search", "ddg-search", "exa", "oxylabs"}
    assert {t["name"].split("_", 1)[0] for t in M.GEMINI_SEARCH_TOOLS} == set(
        M.GEMINI_SEARCH_SERVERS)
    assert M.is_gemini_search_tool("exa_web_search_exa")
    for name in ("fetch_fetch", "wikipedia_search_wikipedia", "web_search"):
        assert not M.is_gemini_search_tool(name), name
    assert not hasattr(M, "is_search_like")


def test_mcp_atlas_tool_call_needs_an_active_agent_environment():
    from gbench.runners.eval_suites.mcp_atlas import execute_mcp_tool
    with pytest.raises(RuntimeError, match="no agent environment"):
        asyncio.run(execute_mcp_tool("filesystem_read_file", {"path": "/data/x"}))


def test_mcp_atlas_failed_tool_call_is_fed_back_as_the_error_body():
    """upstream feeds a failed tool back as its error only; no advice on phrasing the answer"""
    from gbench.runners.eval_suites.mcp_atlas import harness_tool_output
    body = '{"detail":"Failed to call tool \'notion_API-post-search\': Unknown tool"}'
    assert harness_tool_output(500, body, "notion_API-post-search") == (body, "error_reply")


# --------------------------------------------------------------------------- #
# browsecomp prompting must stay canonical
# --------------------------------------------------------------------------- #
def test_browsecomp_query_template_matches_upstream_simple_evals():
    """Verified against openai/simple-evals@main/browsecomp_eval.py on 2026-08-18: the
    question plus the Explanation / Exact Answer / Confidence block, as ONE user message,
    with no system prompt, header or footer. Drift here silently changes the benchmark."""
    from gbench.runners.eval_suites import browsecomp as B
    q = B.QUERY_TEMPLATE
    assert q.startswith("{question}"), "the question must lead; upstream adds no preamble"
    for line in ("Your response should be in the following format:",
                 "Explanation: {your explanation for your final answer}",
                 "Exact Answer: {your succinct, final answer}",
                 "Confidence: {your confidence score between 0% and 100% for your answer}"):
        assert line.replace("{", "{{").replace("}", "}}") in q or line in q.replace("{{", "{").replace("}}", "}")
    assert "system" not in q.lower()


def test_browsecomp_docstring_states_closed_book_protocol():
    """The suite is now canonical closed-book (no tools). The docstring must say so and must
    NOT claim a search tool is wired - the code and the docs used to disagree, so a 0% read as
    expected rather than as a finding, and now the reverse mistake (doc still promising search)
    would be just as misleading."""
    from gbench.runners.eval_suites import browsecomp as B
    doc = (B.__doc__ or "").lower()
    assert "closed-book" in doc, "the docstring must state the protocol is closed-book"
    assert "no tools" in doc or "no injected search" in doc
    # It must NOT claim search is wired now (past tense describing the removal is fine).
    assert "gbench offers a single `web_search`" not in doc


# --- gemini model cascade (replaced the DuckDuckGo fallback 2026-08-21) -------------------
def test_cascade_advances_on_backend_failure_not_on_empty(monkeypatch):
    """A per-model 429 advances to the next grounding model; a genuine "no results" does NOT
    (grounding is trustworthy when it says empty, and advancing would burn another quota)."""
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    for knob in ("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS",
                 "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(knob, raising=False)
    first, second = base.GEMINI_CASCADE[:2]

    # model 1 429s, model 2 serves -> backend is model 2
    def two(q, n=8, model=None):
        if model == first:
            return [{"title": "error", "url": "", "snippet": "HTTP Error 429"}]
        return [{"title": "t", "url": "", "snippet": "s"}]
    monkeypatch.setattr(ST, "gemini_search", two)
    assert ST.search_with_fallback("q")[1] == f"gemini:{second}"

    # a genuine empty from the first model is a real answer - do NOT advance
    calls = []
    def empty(q, n=8, model=None):
        calls.append(model)
        return [{"title": "no results", "url": "", "snippet": "The search returned nothing."}]
    monkeypatch.setattr(ST, "gemini_search", empty)
    assert ST.search_with_fallback("q")[1] == f"gemini:{first}"
    assert calls == [first], "must not try further models on a genuine empty"


def test_backend_actually_used_is_reported(monkeypatch):
    """A run must record the model that actually served, never claim one it did not use."""
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(ST, "gemini_search",
                        lambda q, n=8, model=None: [{"title": "t", "url": "", "snippet": "s"}])
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.5-flash,gemini-2.5-flash")
    assert ST.search_with_fallback("q")[1] == "gemini:gemini-3.5-flash"


def test_all_models_failing_reports_failure_not_emptiness(monkeypatch):
    """When every grounding model 429s, the tool returns a FAILURE (not a fake empty), so the
    model is never told 'the fact does not exist' off an exhausted quota."""
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("GBENCH_SEARCH_MODELS", raising=False)
    monkeypatch.delenv("GBENCH_SEARCH_MODEL", raising=False)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "1")  # one pass -> every model once
    tried = []
    def allfail(q, n=8, model=None):
        tried.append(model)
        return [{"title": "error", "url": "", "snippet": "429"}]
    monkeypatch.setattr(ST, "gemini_search", allfail)
    rows, backend = ST.search_with_fallback("q")
    assert backend == "all-grounding-models-failed" and ST._is_failure(rows)
    assert len(tried) == 5, "every model in the default cascade must be attempted"


def test_cascade_is_overridable_and_defaults_to_five_models(monkeypatch):
    """GBENCH_SEARCH_MODELS overrides the order; the legacy singular GBENCH_SEARCH_MODEL is a
    one-model cascade; the default is gbench's one 5-model Gemini cascade (base.GEMINI_CASCADE)."""
    from gbench.runners.eval_suites import search_tool as ST
    for knob in ("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS",
                 "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(knob, raising=False)
    assert ST._search_cascade() == list(base.GEMINI_CASCADE) == [
        "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
        "gemini-3.1-pro-preview"]
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "a, b ,c")
    assert ST._search_cascade() == ["a", "b", "c"]
    monkeypatch.delenv("GBENCH_SEARCH_MODELS")
    monkeypatch.setenv("GBENCH_SEARCH_MODEL", "gemini-3.5-flash")
    assert ST._search_cascade() == ["gemini-3.5-flash"]


def test_ddg_is_fully_removed():
    """The DuckDuckGo fallback was removed - no keyless scraper path remains."""
    from gbench.runners.eval_suites import search_tool as ST
    assert not hasattr(ST, "ddg_search")
    assert not hasattr(ST, "ddg_available")


# --- grounding retry ----------------------------------------------------------------------
def _urlopen_failing(times, exc, ok_payload=None):
    state = {"n": 0}
    class _R:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(ok_payload or {
            "candidates": [{"content": {"parts": [{"text": "ans"}]},
                            "groundingMetadata": {"groundingChunks": [
                                {"web": {"title": "t", "uri": "u"}}]}}]}).encode()
    def _open(req, timeout=None):
        state["n"] += 1
        if state["n"] <= times:
            raise exc
        return _R()
    return _open, state


def test_transient_grounding_error_is_retried(monkeypatch):
    """2 of 12 grounding calls failed transiently on 2026-08-20 and both succeeded on
    re-issue. Without a retry each fell through to the blocked DDG fallback, turning a blip
    into a permanent "the web has no answer" for that sample."""
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(ST, "_SEARCH_BACKOFF", 0.0)
    opener, state = _urlopen_failing(2, OSError("connection reset"))
    monkeypatch.setattr("urllib.request.urlopen", opener)
    rows = ST.gemini_search("q")
    assert not ST._is_failure(rows) and state["n"] == 3


def test_429_is_NOT_retried_on_the_same_model(monkeypatch):
    """A 429 does not clear in-place, so gemini_search does not retry the SAME model - it
    fails fast so the cascade can move to the next model (its own quota), and if every model
    is throttled at once search_with_fallback backs off and retries the whole cascade. This
    asserts only the in-model behaviour: a 429 costs exactly one attempt."""
    import urllib.error
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(ST, "_SEARCH_BACKOFF", 0.0)
    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
    opener, state = _urlopen_failing(99, err)
    monkeypatch.setattr("urllib.request.urlopen", opener)
    rows = ST.gemini_search("q")
    assert ST._is_failure(rows) and state["n"] == 1, "429 must cost exactly one attempt"


# --- rate limiter + cascade-round backoff (--sandboxes 32 throttle-safety, 2026-08-21) ----
def test_rate_limiter_smooths_a_burst_with_a_fake_clock():
    """A token bucket at R rps with burst B lets the first B requests through instantly, then
    paces the rest at ~1/R each - so N concurrent callers cannot burst past the grounding
    rate limit. Driven by a fake clock so it asserts the pacing without real waits."""
    from gbench.runners.eval_suites import search_tool as ST
    now = {"t": 0.0}
    slept = {"total": 0.0}
    def clock():
        return now["t"]
    def sleep(s):
        slept["total"] += s
        now["t"] += s          # time advances while we wait -> tokens refill
    rl = ST._RateLimiter(rps=5.0, burst=5.0, clock=clock, sleep=sleep)
    for _ in range(5):         # first `burst` are free
        rl.acquire()
    assert slept["total"] == 0.0
    for _ in range(5):         # next 5 are paced at 1/5s each
        rl.acquire()
    assert abs(slept["total"] - 1.0) < 1e-6   # 5 requests / 5 rps = 1.0s of pacing


def test_rate_limiter_is_disabled_when_rps_non_positive():
    from gbench.runners.eval_suites import search_tool as ST
    slept = {"n": 0}
    rl = ST._RateLimiter(rps=0.0, burst=1.0, clock=lambda: 0.0,
                         sleep=lambda s: slept.__setitem__("n", slept["n"] + 1))
    for _ in range(1000):
        rl.acquire()
    assert slept["n"] == 0, "rps<=0 must never block"


def test_rate_limiter_rebuilds_from_env():
    from gbench.runners.eval_suites import search_tool as ST
    import os
    from unittest import mock
    with mock.patch.dict(os.environ, {"GBENCH_SEARCH_MAX_RPS": "3", "GBENCH_SEARCH_MAX_BURST": "9"}):
        ST.reset_rate_limiter()
        assert ST._RATE_LIMITER.rps == 3.0 and ST._RATE_LIMITER.capacity == 9.0
    ST.reset_rate_limiter()   # restore the default for the rest of the suite


def test_every_grounding_request_takes_a_rate_limit_token(monkeypatch):
    """The limiter is only useful if the request path actually goes through it."""
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    calls = {"n": 0}
    monkeypatch.setattr(ST._RATE_LIMITER, "acquire",
                        lambda: calls.__setitem__("n", calls["n"] + 1))
    with _with_urlopen(_grounding_payload()):
        ST.gemini_search("q")
    assert calls["n"] == 1, "gemini_search must take exactly one token per HTTP attempt"


def test_whole_cascade_failure_is_retried_in_rounds(monkeypatch):
    """A synchronized all-model 429 (a rate burst) is recovered by backing off and re-running
    the cascade - the burst drains and the lookup completes instead of returning a fake zero.
    This is the fix for the --sandboxes-32 100%-all-failed result on 2026-08-21."""
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    for knob in ("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS",
                 "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(knob, raising=False)
    # limiter off for a deterministic test (auto-restored by monkeypatch, no leak)
    monkeypatch.setattr(ST, "_RATE_LIMITER", ST._RateLimiter(0.0, 1.0))
    monkeypatch.setattr(ST, "_SEARCH_BACKOFF", 0.0)   # no real sleeps
    sleeps = {"n": 0}
    monkeypatch.setattr(ST.time, "sleep", lambda s: sleeps.__setitem__("n", sleeps["n"] + 1))
    state = {"round": 0}
    def burst_then_ok(q, n=8, model=None):
        # first full pass (all 5 models) fails; second pass, first model serves
        if state["round"] < 5:
            state["round"] += 1
            return [{"title": "error", "url": "", "snippet": "429"}]
        return [{"title": "t", "url": "", "snippet": "s"}]
    monkeypatch.setattr(ST, "gemini_search", burst_then_ok)
    rows, backend = ST.search_with_fallback("q")
    assert backend == f"gemini:{base.GEMINI_CASCADE[0]}", "recovers on the retry round"
    assert sleeps["n"] == 1, "backed off once between the failed pass and the retry"


def test_persistent_all_fail_gives_up_after_configured_rounds(monkeypatch):
    """A genuine outage (not a burst) must still terminate: after GBENCH_SEARCH_CASCADE_ROUNDS
    passes it reports all-grounding-models-failed rather than retrying forever."""
    from gbench.runners.eval_suites import search_tool as ST
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("GBENCH_SEARCH_MODELS", raising=False)
    monkeypatch.delenv("GBENCH_SEARCH_MODEL", raising=False)
    monkeypatch.setattr(ST, "_RATE_LIMITER", ST._RateLimiter(0.0, 1.0))
    monkeypatch.setattr(ST, "_SEARCH_BACKOFF", 0.0)
    monkeypatch.setattr(ST.time, "sleep", lambda s: None)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "2")
    tried = []
    monkeypatch.setattr(ST, "gemini_search",
                        lambda q, n=8, model=None: (tried.append(model),
                                                    [{"title": "error", "url": "", "snippet": "429"}])[1])
    rows, backend = ST.search_with_fallback("q")
    assert backend == "all-grounding-models-failed"
    assert len(tried) == 10, "2 rounds x 5 models before giving up"


# --- mcp_atlas ----------------------------------------------------------------------------
def test_mcp_atlas_unwraps_dict_shaped_tool_entries():
    """5 of the 500 MCP-Atlas rows ship tools as objects, not bare names. `str(x)` made the
    whole dict repr the function NAME - illegal under ^[a-zA-Z0-9_-]{1,64}$ - so those tools
    could never be called."""
    from gbench.runners.eval_suites.mcp_atlas import _parse_list_field
    raw = ("[{'name': 'mongodb_list-collections', 'optionalParams': [], "
           "'requiredParams': ['database']}, 'fetch_fetch']")
    assert _parse_list_field(raw) == ["mongodb_list-collections", "fetch_fetch"]


def test_mcp_atlas_claims_parsing_is_unaffected_by_the_dict_unwrap():
    from gbench.runners.eval_suites.mcp_atlas import _parse_list_field
    assert _parse_list_field('["claim one.", "claim two."]') == ["claim one.", "claim two."]


# --- mcp_atlas scoring must match scaleapi/mcp-atlas score_claims.py ----------------------
def test_mcp_atlas_coverage_mapping_is_ternary_not_binary():
    """`coverage_to_score` in the official CoverageEvaluator. The middle rung is the point:
    a binary supported/not judgement throws away every partially-covered claim."""
    from gbench.runners.eval_suites.mcp_atlas import _COVERAGE_TO_SCORE
    assert _COVERAGE_TO_SCORE == {"fulfilled": 1.0,
                                  "partially_fulfilled": 0.5,
                                  "not_fulfilled": 0.0}


def test_mcp_atlas_judge_prompt_grades_exactly_one_claim():
    """The official evaluator calls the judge once PER CLAIM. Batching all claims into one
    prompt is a different measurement, not an optimisation."""
    from gbench.runners.eval_suites.mcp_atlas import _CLAIM_JUDGE_PROMPT
    assert "{claim}" in _CLAIM_JUDGE_PROMPT and "{response}" in _CLAIM_JUDGE_PROMPT
    assert "CLAIM TO EVALUATE" in _CLAIM_JUDGE_PROMPT
    # the numeric-tolerance guidance is part of the rubric, not decoration
    assert "within 5% of the claimed number" in _CLAIM_JUDGE_PROMPT
    # a batched prompt would have to enumerate; the official one never does
    assert "claim numbers" not in _CLAIM_JUDGE_PROMPT


@pytest.mark.asyncio
async def test_mcp_atlas_coverage_is_the_mean_of_per_claim_scores(monkeypatch):
    """coverage_score = round(total_score / len(claims), 3), including partial credit."""
    from gbench.runners.eval_suites import mcp_atlas as M
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    outcomes = iter(["fulfilled", "partially_fulfilled", "not_fulfilled"])

    class _Models:
        async def generate_content(self, **kw):
            return type("R", (), {"text": '{"coverage_outcome": "%s"}' % next(outcomes)})()

    class _Client:
        def __init__(self, **kw): self.aio = type("A", (), {"models": _Models()})()

    monkeypatch.setattr("google.genai.Client", _Client)
    # upstream extract_claims drops claims of 3 characters or fewer, so use real-length ones
    tr = [{"gold_answer": json.dumps(["claim a", "claim b", "claim c"]), "response_text": "x"}]
    await M._async_judge_mcp_atlas(tr)
    assert tr[0]["claim_recall"] == 0.5          # (1.0 + 0.5 + 0.0) / 3
    assert tr[0]["fully_covered_claims"] == 1
    assert tr[0]["partially_covered_claims"] == 1
    assert tr[0]["is_correct"] is False          # below the 0.75 leaderboard pass bar


@pytest.mark.asyncio
async def test_mcp_atlas_empty_response_is_scored_zero_and_KEPT(monkeypatch):
    """"All rows are kept (including errors) for a holistic picture" - score_claims.py.
    Dropping them inflates the mean: on 2026-08-20 excluding 40 empties read 12.56% where
    keeping them read 11.56%."""
    from gbench.runners.eval_suites import mcp_atlas as M
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr("google.genai.Client", lambda **kw: type("C", (), {"aio": None})())
    tr = [{"gold_answer": json.dumps(["a", "b"]), "response_text": ""}]
    await M._async_judge_mcp_atlas(tr)
    assert tr[0]["claim_recall"] == 0.0 and tr[0]["is_correct"] is False
    assert "no response" in tr[0]["judge_grade"]


def test_mcp_atlas_reports_the_official_pass_rate_thresholds():
    from gbench.runners.eval_suites.mcp_atlas import _PASS_THRESHOLDS
    assert _PASS_THRESHOLDS == (0.50, 0.75)
