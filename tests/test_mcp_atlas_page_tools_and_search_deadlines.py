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

"""mcp_atlas page tools and search deadlines: the page tools run on the release's servers (no
Gemini URL context tool), and gbench's Gemini-served search tools stay on the harness's clock.

* The page tools (exa_web_fetch_exa, oxylabs_universal_scraper, oxylabs_amazon_product_scraper,
  ddg-search_fetch_content) run on the release's servers with the harness's 60 s tool-call
  timeout, like every other tool: no URL context lookup, no page deadline of gbench's own, no
  page cache of gbench's own (the environment's cache applies as upstream).
* A search call gets the harness's 60 s timeout, its grounding lookup is sent with the call's
  deadline, a failure in gbench's own serving or a wall-clock stop during a Gemini call is a
  harness error, and a task the wall clock stopped after waiting on Gemini is named.

Offline: the grounding backend and the credentialed-servers container are stubs; the real
Gemini request fails the test if reached.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import search_tool

REPO = Path(__file__).resolve().parents[1]
DOC = " ".join((REPO / "docs" / "evals" / "mcp_atlas.md").read_text(encoding="utf-8").split())
FAKE_KEY = "AIzaFAKE-robust-search-key-0123456789"
HITS = [{"title": "T", "url": "https://t.example/", "snippet": "S"}]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test reached a real Gemini request")

    monkeypatch.setattr(search_tool, "gemini_search", refuse)
    monkeypatch.setattr(search_tool, "search_with_fallback", refuse)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)


def _call(sandbox, name, args, idx=None):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(M.execute_mcp_tool(name, args, idx))
    finally:
        M._ACTIVE_SANDBOX.reset(token)


# --------------------------------------------------------------------------- #
# the page tools: no URL context lookup, the harness's clock
# --------------------------------------------------------------------------- #
def test_no_url_context_machinery_is_left():
    for name in ("url_context_page", "_url_context_request", "GEMINI_PAGE_TOOLS",
                 "URL_CONTEXT_LOOKUP_TIMEOUT_S", "URL_CONTEXT_REQUEST_TIMEOUT_S",
                 "gemini_call_timeout", "_page_prompt", "_seconds_summary",
                 "_public_url_problem", "GEMINI_GENERATE_URL"):
        assert not hasattr(M, name), name
    sb = M._Sandbox("http://127.0.0.1:9")
    assert not hasattr(sb, "pages") and not hasattr(sb, "url_context_lookups")


def test_page_calls_get_the_harness_timeout_and_no_cache_of_gbenchs(monkeypatch):
    sent = []

    async def exec_http(container, method, path, timeout, body=None):
        sent.append((timeout, body["tool_name"]))
        return 200, json.dumps([{"type": "text", "text": "page"}])

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "_ground", lambda *a, **k: pytest.fail("a page tool reached Gemini"))
    sb = M._Sandbox("http://127.0.0.1:9", cred_container="cred",
                    cred_servers=M.CREDENTIALED_SERVERS)
    for _ in range(2):
        assert _call(sb, "oxylabs_universal_scraper", {"url": "https://a.example/"}, 0) == "page"
    # both calls reach the server: the environment's own cache (main.py) answers repeats, as
    # upstream; gbench keeps no page cache of its own
    assert sent == [(M.TOOL_CALL_TIMEOUT_S, "oxylabs_universal_scraper")] * 2
    assert sb.search_calls == 0 and sb.search_cache == {} and sb.gemini_wait_s == {}


# --------------------------------------------------------------------------- #
# the search tools: deadline, own failures, wall clock
# --------------------------------------------------------------------------- #
def test_search_calls_get_the_harness_timeout(monkeypatch):
    seen = {}

    def ground(query, wanted):
        seen["ground"] = M._GROUND_DEADLINE.get() - time.monotonic()
        return HITS, "gemini:g"

    monkeypatch.setattr(M, "_ground", ground)
    sb = M._Sandbox("http://127.0.0.1:9")
    _call(sb, "ddg-search_search", {"query": "q"}, 0)
    grace = M._lookup_grace(M.TOOL_CALL_TIMEOUT_S)
    assert M.TOOL_CALL_TIMEOUT_S - grace - 1 < seen["ground"] <= M.TOOL_CALL_TIMEOUT_S - grace
    assert M._GROUND_DEADLINE.get() is None, "the deadline does not leak past the call"


def test_the_grounding_lookup_is_sent_with_the_calls_deadline(monkeypatch):
    got = {}

    def swf(query, max_results=10, deadline=None):
        got["deadline"] = deadline
        return [{"title": "T", "url": "https://t.example/", "snippet": "S"}], "gemini:g"

    monkeypatch.setattr(search_tool, "search_with_fallback", swf)
    dl = time.monotonic() + 12.0
    M.gemini_search_reply("brave-search_brave_web_search", {"query": "q"}, deadline=dl)
    assert got["deadline"] == dl
    M.gemini_search_reply("brave-search_brave_web_search", {"query": "q"})
    assert got["deadline"] is None


def test_a_failure_in_gbenchs_own_serving_is_a_harness_error(monkeypatch):
    def broken(query, wanted):
        raise KeyError(f"bug near {FAKE_KEY}")

    monkeypatch.setattr(M, "_ground", broken)
    sb = M._Sandbox("http://127.0.0.1:9")
    out = _call(sb, "exa_web_search_exa", {"query": "q"}, 4)
    assert out.startswith('{"detail":"Failed to call tool \'exa_web_search_exa\': Error: '
                          'gbench could not serve the call (KeyError')
    assert FAKE_KEY not in out and "tool raised" not in out
    assert sb.harness_error_samples == {4: 1} and sb.gemini_outages == 1


def test_a_wall_clock_stop_during_a_gemini_call_is_a_harness_error(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(M, "_ground", lambda *a, **k: (release.wait(5), (HITS, "g"))[1])
    sb = M._Sandbox("http://127.0.0.1:9")

    async def run():
        token = M._ACTIVE_SANDBOX.set(sb)
        try:
            task = asyncio.ensure_future(M.execute_mcp_tool(
                "oxylabs_google_search_scraper", {"query": "q"}, 9))
            await asyncio.sleep(0.2)
            task.cancel()                       # what base's wall clock does to the sample
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            M._ACTIVE_SANDBOX.reset(token)

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert sb.gemini_wall_clock_cuts == 1 and sb.harness_error_samples == {9: 1}
    assert sb.gemini_wait_s[9] >= 0.15


# --------------------------------------------------------------------------- #
# a whole run
# --------------------------------------------------------------------------- #
def _environment():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/enabled-servers":
                servers = [[s, "OK"] for s in M.ENVIRONMENT_SERVERS]
                return self._send({"servers": servers, "total": len(servers),
                                   "online": len(servers), "offline": 0})
            return self._send({"message": "MCP Agent Environment API"})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/list-tools":
                return self._send([{"name": "fetch_fetch", "description": "Fetches.",
                                    "inputSchema": {"type": "object"}}])
            return self._send([{"type": "text", "text": "ok"}])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def atlas_run(monkeypatch):
    srv, url = _environment()
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", url)
    # the external environment's checks, gbench's credentialed-servers container and the
    # image checks they need (tests/test_mcp_credential_isolation_and_dns.py covers them): no
    # docker here
    monkeypatch.setattr(M, "_external_environment_setup", lambda sandbox: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_MCP_ATLAS_TEMPERATURE", raising=False)
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    rows = [{"TASK": f"t{i}", "PROMPT": f"Search q{i}",
             "ENABLED_TOOLS": '["oxylabs_google_search_scraper", "fetch_fetch"]',
             "GTFA_CLAIMS": '["The page says hello."]'} for i in range(2)]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)

    async def judge(prompt, config=None):
        return json.dumps({"claim_text": "c", "coverage_outcome": "fulfilled",
                           "justification": "j", "confidence_level": 1.0}), "gemini-test"

    monkeypatch.setattr(M, "judge_generate_cascade", judge)

    async def fake_send(**k):
        async with k["semaphore"]:          # base's request slot, which starts the task's clock
            pass
        last = k["messages"][-1]
        if last["role"] == "user":
            query = last["content"].split()[-1]
            return base.Reply(text="", tool_calls=[{"id": "c1", "type": "function", "function": {
                "name": "oxylabs_google_search_scraper",
                "arguments": json.dumps({"query": query})}}], finish_reason="tool_calls")
        return base.Reply(text="The page says hello.", tool_calls=None, finish_reason="stop")

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    yield
    srv.shutdown()
    srv.server_close()


def test_a_run_whose_wall_clock_stops_a_search_lookup_does_not_score_that_task(atlas_run,
                                                                               monkeypatch):
    release = threading.Event()

    def ground(query, wanted):                    # holds only task t1's lookup
        if query == "q1":
            release.wait(5)
        return HITS, "gemini:g"

    monkeypatch.setattr(M, "_ground", ground)
    monkeypatch.setattr(M, "MCP_ATLAS_TOOL_LOOP",
                        M.MCP_ATLAS_TOOL_LOOP._replace(max_sample_wall_s=1.5))
    try:
        res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=2,
                              max_output_tokens=256)
    finally:
        release.set()
    by_task = {t["extra_payload"]["task_id"]: t for t in res["sample_traces"]}
    assert by_task["t0"]["claim_recall"] == 1.0
    t1 = by_task["t1"]
    assert t1["judge_grade"] == M.HARNESS_ERROR_GRADE and t1["scoring_excluded"] is True
    assert res["harness_errors"][0]["gemini_calls_cut_by_wall_clock"] == 1
    assert "still waiting on the Gemini API when the task's" in \
        res["leaderboard_comparable_reason"]
    assert res["status"] == "completed_with_errors"
    assert set(res["search_substitution"]["gemini_wait_s"]) == {
        "total", "max_task", "wall_clock_limited_tasks", "wall_clock_limited_tasks_wait_s"}
    assert all("gemini_wait_s" in t for t in res["sample_traces"])


def test_a_wall_clock_stop_after_waiting_on_gemini_is_named(monkeypatch):
    sb = M._Sandbox("http://env")
    sb.gemini_wait_s = {0: 3.0, 1: 42.25}
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_open_sandbox", lambda: sb)
    monkeypatch.setattr(M, "_wait_ready", lambda s, t: None)
    monkeypatch.setattr(M, "_preflight", lambda s: {"health_check": "passed"})
    monkeypatch.setattr(M, "_list_tools", lambda s: [{"name": "fetch_fetch"}])
    monkeypatch.setattr(M, "_load_mcp_atlas_samples", lambda limit=None, catalog=None: [])
    monkeypatch.setattr(M, "run_eval_suite", lambda **kw: {
        "status": "success", "sample_traces": [
            {"sample_idx": 0, "category": "exa", "claim_recall": 1.0, "is_correct": True,
             "claim_scores": [1.0], "judge_models": ["g"]},
            {"sample_idx": 1, "category": "exa", "claim_recall": 0.0, "is_correct": False,
             "status": "WALL_CLOCK_LIMIT"}]})
    res = M.run_mcp_atlas("m", "http://x", concurrency=1)
    assert [t["gemini_wait_s"] for t in res["sample_traces"]] == [3.0, 42.2]
    wait = res["search_substitution"]["gemini_wait_s"]
    assert wait == {"total": 45.2, "max_task": 42.2, "wall_clock_limited_tasks": 1,
                    "wall_clock_limited_tasks_wait_s": 42.2}
    assert ("1 task(s) stopped by the 1800 s wall clock (scored 0) had waited 42.2 s in all "
            "on gbench's Gemini-served search tools") in res["leaderboard_comparable_reason"]


def test_doc_states_the_clock_and_the_records():
    for needle in ("60 s tool-call timeout", "2 s before it", "`gemini_wait_s`",
                   "`gemini_calls_cut_by_wall_clock`", "`upstream_tool_calls`",
                   "`claim_judge_models`"):
        assert needle in DOC, needle
    for gone in ("finishReason", "RECITATION", "300 s", "a relative URL stays relative",
                 "is not an Amazon domain"):
        assert gone not in DOC, gone
