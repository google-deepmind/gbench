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

"""mcp_atlas: the search-engine servers' page tools (ddg-search_fetch_content, exa_web_fetch_exa,
oxylabs_universal_scraper, oxylabs_amazon_product_scraper) run on the release's servers as
upstream, the search tools' inputSchema check, Gemini grounding failures as harness errors, and
lookups that stop on time.

Offline: the grounding backend is a stub (`mcp_atlas._ground`), the credentialed-servers
container's API a stub (`mcp_atlas._exec_http_async`), and the environment a local HTTP server;
the real Gemini request fails the test if reached.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema
import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import search_tool

REPO = Path(__file__).resolve().parents[1]
DOC = " ".join((REPO / "docs" / "evals" / "mcp_atlas.md").read_text(encoding="utf-8").split())
DOCKERFILE = (REPO / "docker" / "mcp_atlas.Dockerfile").read_text(encoding="utf-8")
FAKE_KEY = "AIzaFAKE-url-tools-key-0123456789"
HITS = [{"title": "T", "url": "https://t.example/", "snippet": "S"}]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test reached a real Gemini request")

    monkeypatch.setattr(search_tool, "gemini_search", refuse)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)


@pytest.fixture
def cred(monkeypatch):
    """The credentialed-servers container's /call-tool, stood in: `state["reply"]` is the
    (status, body) it answers; `state["sent"]` records each request."""
    state = {"sent": [], "reply": (200, json.dumps([{"type": "text", "text": "page text"}]))}

    async def exec_http(container, method, path, timeout, body=None):
        state["sent"].append({"container": container, "path": path, "timeout": timeout,
                              "body": body})
        reply = state["reply"]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "_ground", lambda *a, **k: pytest.fail("a page tool reached Gemini"))
    return state


def _call(sandbox, name, args, idx=None):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(M.execute_mcp_tool(name, args, idx))
    finally:
        M._ACTIVE_SANDBOX.reset(token)


def _cred_sandbox():
    return M._Sandbox("http://127.0.0.1:9", cred_container="cred",
                      cred_servers=M.CREDENTIALED_SERVERS)


# --------------------------------------------------------------------------- #
# the page tools: the release's servers, as upstream
# --------------------------------------------------------------------------- #
def test_the_page_tools_are_the_search_servers_non_search_tools():
    assert M.UPSTREAM_SEARCH_TOOLS == ("ddg-search_fetch_content", "exa_web_fetch_exa",
                                       "oxylabs_universal_scraper",
                                       "oxylabs_amazon_product_scraper")
    assert set(M.UPSTREAM_SEARCH_TOOLS) < {t["name"] for t in M.GEMINI_SEARCH_TOOLS}
    assert not set(M.UPSTREAM_SEARCH_TOOLS) & set(M.GEMINI_SERVED_TOOLS)
    assert set(M.UPSTREAM_SEARCH_TOOLS) | set(M.GEMINI_SERVED_TOOLS) == {
        t["name"] for t in M.GEMINI_SEARCH_TOOLS}
    for tool in M.UPSTREAM_SEARCH_TOOLS:
        assert not M.is_gemini_search_tool(tool)
        assert M.tool_server(tool) in M.ENVIRONMENT_SEARCH_SERVERS
        with pytest.raises(ValueError, match="not served by gbench"):
            M.gemini_search_reply(tool, {"url": "https://a.example/"})


@pytest.mark.parametrize("tool,args", [
    ("exa_web_fetch_exa", {"urls": ["https://a.example/", "https://b.example/"],
                           "maxCharacters": 5}),
    ("oxylabs_universal_scraper", {"url": "https://a.example/", "output_format": "links"}),
    ("oxylabs_amazon_product_scraper", {"query": "B0TEST1234", "domain": "co.uk"}),
])
def test_exa_and_oxylabs_page_tools_go_to_the_credentialed_servers(cred, tool, args):
    sb = _cred_sandbox()
    out = _call(sb, tool, args, 0)
    assert out == "page text", "the server's reply, unchanged"
    assert cred["sent"] == [{"container": "cred", "path": "/call-tool",
                             "timeout": M.TOOL_CALL_TIMEOUT_S,
                             "body": {"tool_name": tool, "tool_args": args}}]
    assert (sb.search_calls, sb.gemini_outages, sb.harness_error_samples) == (0, 0, {})
    assert sb.page_tool_calls == {tool: 1}


def test_a_page_tool_error_is_the_servers_reply_and_the_models_outcome(cred):
    body = json.dumps({"detail": "Failed to call tool 'exa_web_fetch_exa': Tool "
                                 "'exa_web_fetch_exa' execution failed: Error fetching URL(s): "
                                 "https://u.example/: CRAWL_NOT_FOUND"})
    cred["reply"] = (500, body)
    sb = _cred_sandbox()
    out = _call(sb, "exa_web_fetch_exa", {"urls": ["https://u.example/"]}, 4)
    assert out == body and sb.error_replies == 1
    assert sb.harness_error_samples == {}, "a failed fetch is the model's outcome, as upstream"
    cred["reply"] = asyncio.TimeoutError()
    out = _call(sb, "oxylabs_universal_scraper", {"url": "https://slow.example/"}, 4)
    assert out == f"Tool call timed out after {M.TOOL_CALL_TIMEOUT_S:g}s"
    assert sb.timeouts == 1 and sb.harness_error_samples == {}


def test_a_credential_in_a_credentialed_reply_is_masked(cred, monkeypatch):
    secret = "exa-real-key-value-0123456789"
    monkeypatch.setenv("EXA_API_KEY", secret)
    cred["reply"] = (500, json.dumps({"detail": f"web_fetch_exa error (401): bad key {secret}"}))
    out = _call(_cred_sandbox(), "exa_web_fetch_exa", {"urls": ["https://a.example/"]}, 0)
    assert secret not in out and base.REDACTED in out


def test_ddg_fetch_content_goes_to_the_environment_container(cred):
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps([{"type": "text", "text": "the page"}]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        sb = M._Sandbox(f"http://127.0.0.1:{srv.server_address[1]}", cred_container="cred",
                        cred_servers=M.CREDENTIALED_SERVERS)
        out = _call(sb, "ddg-search_fetch_content", {"url": "https://a.example/"}, 0)
    finally:
        srv.shutdown()
        srv.server_close()
    assert out == "the page" and cred["sent"] == []
    assert calls == [{"tool_name": "ddg-search_fetch_content",
                      "tool_args": {"url": "https://a.example/"}}]


def test_ddg_fetch_content_errors_are_the_servers_text():
    out = "Error: Could not access the webpage (HTTP 404)"
    # test_servers.py's pass rule treats such a text as a failure, as upstream
    assert not M.health_call_ok("ddg-search", 200, json.dumps([{"type": "text", "text": out}]))


def test_search_tools_still_ground_and_page_tools_never_do(monkeypatch):
    asked = []
    monkeypatch.setattr(M, "_ground", lambda q, n: (asked.append((q, n)) or (HITS, "gemini:g")))
    for tool in M.UPSTREAM_SEARCH_TOOLS:
        with pytest.raises(ValueError):
            M.gemini_search_reply(tool, {"url": "https://a.example/"})
    assert asked == []
    M.gemini_search_reply("oxylabs_amazon_search_scraper", {"query": "hub"})
    assert asked == [("hub site:amazon.com", 10)]


# --------------------------------------------------------------------------- #
# arguments of the search tools: the environment's inputSchema check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tool,args", [
    ("brave-search_brave_web_search", {}),
    ("brave-search_brave_web_search", {"query": "q", "count": "abc"}),
    ("brave-search_brave_web_search", {"query": "q", "count": "3"}),
    ("exa_web_search_exa", {"query": "q", "extra": 1}),
    ("exa_web_search_exa", {"query": "q", "numResults": 0}),
    ("oxylabs_google_search_scraper", {"query": "q", "parse": "false"}),
    ("oxylabs_amazon_search_scraper", {"query": "q", "render": "js"}),
])
def test_invalid_arguments_get_the_mcp_servers_validation_error(tool, args, monkeypatch):
    monkeypatch.setattr(M, "_ground", lambda *a: pytest.fail("looked up an invalid call"))
    schema = next(t["inputSchema"] for t in M.GEMINI_SEARCH_TOOLS if t["name"] == tool)
    with pytest.raises(jsonschema.ValidationError) as ref:
        jsonschema.validate(instance=args, schema=schema)
    status, body, backends = M.gemini_search_reply(tool, args)
    assert status == 500 and backends == []
    assert json.loads(body) == {"detail": f"Failed to call tool '{tool}': Input validation "
                                          f"error: {ref.value.message}"}


def test_valid_arguments_pass_the_schema(monkeypatch):
    monkeypatch.setattr(M, "_ground", lambda q, n: ([], "gemini:g"))
    for tool, args in [("brave-search_brave_web_search", {"query": "q", "count": 2.0}),
                       ("ddg-search_search", {"query": "q", "max_results": 3, "region": "de-de"}),
                       ("exa_web_search_exa", {"query": "q", "numResults": 10}),
                       ("oxylabs_google_search_scraper", {"query": "q", "parse": False})]:
        status, body, _ = M.gemini_search_reply(tool, args)
        assert "Input validation error" not in body, (tool, body)


# --------------------------------------------------------------------------- #
# search calls in a run: statistics and harness errors
# --------------------------------------------------------------------------- #
def test_search_calls_record_backends_and_harness_errors(monkeypatch):
    def ground(query, wanted):
        if query == "down":
            return [{"title": "error", "url": "", "snippet": "all grounding models failed"}], \
                "all-grounding-models-failed"
        return HITS, "gemini:gemini-a"

    monkeypatch.setattr(M, "_ground", ground)
    sb = M._Sandbox("http://127.0.0.1:9")
    out = _call(sb, "exa_web_search_exa", {"query": "up"}, 0)
    assert out.startswith("Title: T")
    out = _call(sb, "exa_web_search_exa", {"query": "down"}, 3)
    assert out.startswith('{"detail":"Failed to call tool \'exa_web_search_exa\': Error: Gemini '
                          'Google Search grounding failed')
    again = _call(sb, "exa_web_search_exa", {"query": "down"}, 3)
    assert again == out, "a failed reply is not cached, and fails again"
    assert sb.search_backends == {"gemini:gemini-a": 1, "all-grounding-models-failed": 2}
    assert sb.gemini_outages == 2 and sb.harness_error_samples == {3: 2}
    assert sb.search_calls == 3 and sb.error_replies == 2


def test_a_lookup_past_the_tool_call_timeout_is_a_harness_error(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(M, "_ground", lambda *a: (release.wait(5), (HITS, "x"))[1])
    monkeypatch.setattr(M, "TOOL_CALL_TIMEOUT_S", 0.2)
    sb = M._Sandbox("http://127.0.0.1:9")
    t0 = time.monotonic()
    out = _call(sb, "oxylabs_google_search_scraper", {"query": "q"}, 7)
    took = time.monotonic() - t0
    release.set()
    assert out == "Tool call timed out after 0.2s"
    assert took < 2, "asyncio.run did not wait for the lookup thread"
    assert (sb.timeouts, sb.search_timeouts, sb.gemini_outages) == (1, 1, 1)
    assert sb.harness_error_samples == {7: 1}


def test_calls_that_cannot_reach_the_environment_are_harness_errors():
    sb = M._Sandbox("http://127.0.0.1:9")        # nothing listens on port 9
    out = _call(sb, "fetch_fetch", {"url": "https://a.example/"}, 2)
    assert out.startswith("Error: Failed to call tool fetch_fetch: ")
    assert sb.transport_errors == 1 and sb.harness_error_samples == {2: 1}


def test_calls_that_cannot_reach_the_credentialed_servers_are_harness_errors(cred, monkeypatch):
    cred["reply"] = OSError("the credentialed-servers container cred could not be reached")
    monkeypatch.setattr(M, "_container_running", lambda name: True)
    sb = _cred_sandbox()
    out = _call(sb, "oxylabs_universal_scraper", {"url": "https://a.example/"}, 5)
    assert out.startswith("Error: Failed to call tool oxylabs_universal_scraper: ")
    assert sb.transport_errors == 1 and sb.harness_error_samples == {5: 1}


def test_the_executor_takes_the_sample_index():
    assert base._executor_wants_session(M.execute_mcp_tool)


def test_preflight_search_checks_do_not_wait_for_a_hung_lookup(monkeypatch):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            data = json.dumps([{"type": "text", "text": "ok"}]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    release = threading.Event()
    monkeypatch.setattr(M, "_ground", lambda q, n: (release.wait(5), ([], "x"))[1])
    monkeypatch.setattr(M, "HEALTH_TIMEOUT_S", 0.5)
    try:
        t0 = time.monotonic()
        results = asyncio.run(M._run_health_calls(f"http://127.0.0.1:{srv.server_address[1]}"))
        took = time.monotonic() - t0
    finally:
        release.set()
        srv.shutdown()
        srv.server_close()
    assert took < 3, took
    # only brave-search's and ddg-search's checks are answered by Gemini grounding
    assert {s for s, (ok, _) in results.items() if not ok} == set(M.GEMINI_PREFLIGHT_SERVERS)
    assert all(results[s][1] == "Timed out after 0.5s" for s in M.GEMINI_PREFLIGHT_SERVERS)


# --------------------------------------------------------------------------- #
# a whole run: a task with a harness error is not scored
# --------------------------------------------------------------------------- #
def _environment():
    calls = []

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
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))
                              or b"{}")
            if self.path == "/list-tools":
                return self._send([{"name": "fetch_fetch", "description": "Fetches.",
                                    "inputSchema": {"type": "object"}}])
            calls.append(body)
            return self._send([{"type": "text", "text": "ok"}])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", calls


@pytest.fixture
def atlas_run(monkeypatch):
    srv, url, calls = _environment()
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", url)
    # the external environment's checks, gbench's credentialed-servers container and the
    # image checks they need (tests/test_mcp_credential_isolation_and_dns.py covers them): no
    # docker here
    monkeypatch.setattr(M, "_external_environment_setup", lambda sandbox: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_MCP_ATLAS_TEMPERATURE", raising=False)
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    state = {"down": set()}

    def ground(query, wanted):
        if query in state["down"]:
            return [{"title": "error", "url": "", "snippet": "no model answered"}], "none"
        return HITS, "gemini:g"

    monkeypatch.setattr(M, "_ground", ground)
    rows = [{"TASK": f"t{i}", "PROMPT": f"Search q{i}",
             "ENABLED_TOOLS": '["exa_web_search_exa", "fetch_fetch"]',
             "GTFA_CLAIMS": '["The page says hello."]'} for i in range(2)]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)

    async def judge(prompt, config=None):
        return json.dumps({"claim_text": "c", "coverage_outcome": "fulfilled",
                           "justification": "j", "confidence_level": 1.0}), "gemini-test"

    monkeypatch.setattr(M, "judge_generate_cascade", judge)

    async def fake_send(**k):
        last = k["messages"][-1]
        if last["role"] == "user":
            query = last["content"].split()[-1]
            return base.Reply(text="", tool_calls=[{"id": "c1", "type": "function", "function": {
                "name": "exa_web_search_exa", "arguments": json.dumps({"query": query})}}],
                finish_reason="tool_calls")
        return base.Reply(text="The page says hello.", tool_calls=None, finish_reason="stop")

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    yield {"calls": calls, "state": state}
    srv.shutdown()
    srv.server_close()


def test_a_task_hit_by_a_gemini_outage_is_not_scored(atlas_run):
    atlas_run["state"]["down"] = {"q1"}
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    by_task = {t["extra_payload"]["task_id"]: t for t in res["sample_traces"]}
    assert by_task["t0"]["claim_recall"] == 1.0 and by_task["t0"]["is_correct"] is True
    assert by_task["t0"]["claim_judge_models"] == ["gemini-test"]
    t1 = by_task["t1"]
    assert t1["judge_grade"] == M.HARNESS_ERROR_GRADE and t1["scoring_excluded"] is True
    assert "claim_recall" not in t1 and t1["harness_error_calls"] == 1
    assert res["accuracy"] == res["pass_rate_0.75"] == 100.0, "over the scored task only"
    assert res["status"] == "completed_with_errors"
    assert res["harness_errors"][0]["tasks_unscored"] == 1
    assert res["harness_errors"][0]["task_ids"] == ["t1"]
    assert res["harness_errors"][0]["gemini_calls"] == 1
    reason = res["leaderboard_comparable_reason"]
    assert "1 search tool call(s) the Gemini API failed to serve" in reason
    assert "1 task(s) with such a harness error not scored" in reason
    assert "their page tools (ddg-search fetch_content, exa web_fetch_exa, oxylabs " \
           "universal_scraper and amazon_product_scraper) run as upstream" in reason
    sub = res["search_substitution"]
    assert sub["tools"] == sub["search_tools"] == list(M.GEMINI_SERVED_TOOLS)
    assert sub["upstream_tools"] == list(M.UPSTREAM_SEARCH_TOOLS)
    assert set(sub["upstream_tools_backend"]) == set(M.UPSTREAM_SEARCH_TOOLS)
    assert "EXA_API_KEY" in sub["upstream_tools_backend"]["exa_web_fetch_exa"]
    assert sub["servers_not_run"] == ["brave-search"] and sub["not_read"] == ["BRAVE_API_KEY"]
    assert "jsonschema" in sub["argument_validation"] and sub["gemini_outages"] == 1
    assert not any(k.startswith(("page_", "url_context")) for k in sub)
    # the model's search calls never reach the environment (only the pre-flight's exa check)
    assert [c["tool_args"] for c in atlas_run["calls"]
            if c["tool_name"] == "exa_web_search_exa"] == [{"query": "python programming"}]


def test_a_run_without_harness_errors_keeps_its_status(atlas_run):
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    assert res["status"] == "success" and "harness_errors" not in res
    assert "failed to serve" not in res["leaderboard_comparable_reason"]
    assert res["search_substitution"]["grounding_backends"] == {"gemini:g": 2}


def test_only_harness_errors_leave_nothing_scored(atlas_run):
    atlas_run["state"]["down"] = {"q0", "q1"}
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    assert res["status"] == "failed"
    assert res["harness_errors"][0]["tasks_unscored"] == 2
    # withheld (null with a reason), never a 0 the model did not earn
    assert res["accuracy"] is None and res["accuracy_withheld_reason"]
    assert "pass_rate_0.75" not in res


def test_a_wall_clock_stop_does_not_score_a_task_with_a_harness_error(monkeypatch):
    sb = M._Sandbox("http://env")
    sb.harness_error_samples[1] = 1
    sb.gemini_outages = 1
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_open_sandbox", lambda: sb)
    monkeypatch.setattr(M, "_wait_ready", lambda s, t: None)
    monkeypatch.setattr(M, "_preflight", lambda s: {"health_check": "passed"})
    monkeypatch.setattr(M, "_list_tools", lambda s: [{"name": "fetch_fetch"}])
    monkeypatch.setattr(M, "_load_mcp_atlas_samples", lambda limit=None, catalog=None: [])
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"status": "success", "scoring_excluded": 0, "correct_answers": 1,
                "sample_traces": [
                    {"sample_idx": 0, "category": "ddg-search", "claim_recall": 1.0,
                     "is_correct": True, "claim_scores": [1.0], "judge_models": ["g"]},
                    # base's wall-clock rule undid the judge's HARNESS_ERROR and scored it 0
                    {"sample_idx": 1, "category": "ddg-search", "claim_recall": 0.0,
                     "is_correct": False, "status": "WALL_CLOCK_LIMIT",
                     "scorer_judge_grade": M.HARNESS_ERROR_GRADE}]}

    monkeypatch.setattr(M, "run_eval_suite", fake_run)
    res = M.run_mcp_atlas("m", "http://x", concurrency=1)
    assert seen["tool_executor"] is M.execute_mcp_tool
    assert seen["async_eval_fn"].keywords["harness_error_samples"] is sb.harness_error_samples
    t1 = res["sample_traces"][1]
    assert t1["judge_grade"] == M.HARNESS_ERROR_GRADE and t1["scoring_excluded"] is True
    assert res["accuracy"] == 100.0 and res["scoring_excluded"] == 1
    assert res["category_accuracy"] == {"ddg-search": {"correct": 1, "total": 1,
                                                       "accuracy": 100.0}}
    assert res["status"] == "completed_with_errors"


def test_the_judge_sets_harness_error_tasks_aside_without_a_judge_call():
    calls = []

    async def judge(prompt, config=None):
        calls.append(prompt)
        return json.dumps({"coverage_outcome": "fulfilled"}), "g"

    tr = [{"sample_idx": 4, "gold_answer": '["The answer is 42."]', "response_text": "42"}]
    orig = M.judge_generate_cascade
    M.judge_generate_cascade = judge
    try:
        asyncio.run(M._async_judge_mcp_atlas(tr, harness_error_samples={4: 3}))
    finally:
        M.judge_generate_cascade = orig
    assert calls == [] and tr[0]["judge_grade"] == "HARNESS_ERROR"
    assert tr[0]["scoring_excluded"] is True and tr[0]["harness_error_calls"] == 3


# --------------------------------------------------------------------------- #
# prerequisites, image and doc
# --------------------------------------------------------------------------- #
def test_missing_jsonschema_is_infra_required(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "jsonschema":
            raise ImportError("no jsonschema")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, reason = M.check_mcp_atlas_prerequisites()
    assert not ok and "pip install jsonschema" in reason


def test_dockerfile_header_counts_the_servers_it_runs():
    assert "the 36 MCP servers of the benchmark" not in DOCKERFILE
    assert "gbench runs 35 of the 36, see addition 4" in DOCKERFILE
    assert "URL context" not in DOCKERFILE
    assert "runs the 20" in DOCKERFILE and "runs the 15 that need" in DOCKERFILE


def test_doc_states_the_page_tools_and_harness_errors():
    for needle in ("`UPSTREAM_SEARCH_TOOLS`", "`GEMINI_SERVED_TOOLS`", "`HARNESS_ERROR`",
                   "`harness_errors`", "`completed_with_errors`", "Input validation error",
                   "jsonschema", "default 10 (5 for `brave-search_brave_local_search`)",
                   "`upstream_tools`", "`upstream_tool_calls`", "`catalog_drift`",
                   "`oxylabs_amazon_product_scraper`", "`EXA_API_KEY`", "`OXYLABS_USERNAME`",
                   "`OXYLABS_PASSWORD`"):
        assert needle in DOC, needle
    for gone in ("URL context", "url_context", "GEMINI_PAGE_TOOLS", "(open question)",
                 "`page_backend`", "300 s"):
        assert gone not in DOC, gone
    for tool in M.UPSTREAM_SEARCH_TOOLS:
        assert f"`{tool}`" in DOC, tool
