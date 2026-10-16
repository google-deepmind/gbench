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

"""mcp_atlas: the search tools of the release's search-engine servers (brave-search,
ddg-search, exa, oxylabs) served by Gemini Google Search grounding behind the same tool names and
argument schemas; their page tools run on the release's servers (never on Gemini).

Offline: the grounding backend is replaced at `mcp_atlas._ground`; the real request
(`search_tool.gemini_search`) fails the test if anything reaches it.
"""

import asyncio
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import search_tool

REPO = Path(__file__).resolve().parents[1]
DOC = " ".join((REPO / "docs" / "evals" / "mcp_atlas.md").read_text(encoding="utf-8").split())
DOCKERFILE = (REPO / "docker" / "mcp_atlas.Dockerfile").read_text(encoding="utf-8")

#: The search-server tool names the 500 tasks' ENABLED_TOOLS use at the pinned dataset
#: revision (tasks listing each: oxylabs 94, exa 77, brave-search 63, ddg-search 56).
DATASET_SEARCH_TOOLS = (
    "brave-search_brave_web_search", "ddg-search_search", "ddg-search_fetch_content",
    "exa_web_search_exa", "oxylabs_google_search_scraper", "oxylabs_universal_scraper",
    "oxylabs_amazon_search_scraper",
)
HITS = [{"title": "Alpha", "url": "https://a.example/", "snippet": "alpha text"},
        {"title": "Beta", "url": "https://b.example/", "snippet": "beta text"}]
NO_RESULTS = [{"title": "no results", "url": "", "snippet": "The search returned nothing."}]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test reached the real Gemini grounding request")

    monkeypatch.setattr(search_tool, "gemini_search", refuse)


@pytest.fixture
def grounding(monkeypatch):
    """Replace the grounding backend; returns the list of (query, wanted) it was asked."""
    state = {"asked": [], "results": HITS, "backend": "gemini:gemini-test"}

    def ground(query, wanted):
        state["asked"].append((query, wanted))
        return [dict(r) for r in state["results"]][:wanted], state["backend"]

    monkeypatch.setattr(M, "_ground", ground)
    return state


def _text(status_body_backends):
    status, body, _ = status_body_backends
    assert status == 200, body
    blocks = json.loads(body)
    assert [b["type"] for b in blocks] == ["text"]
    return blocks[0]["text"]


# --------------------------------------------------------------------------- #
# the tool catalog: same names and argument schemas as the release's servers
# --------------------------------------------------------------------------- #
def test_the_ten_tool_definitions_are_the_pinned_servers_listing():
    digest = hashlib.sha256(json.dumps(M.GEMINI_SEARCH_TOOLS, sort_keys=True).encode())
    assert digest.hexdigest() == M.GEMINI_SEARCH_TOOLS_SHA256
    assert [t["name"] for t in M.GEMINI_SEARCH_TOOLS] == [
        "brave-search_brave_web_search", "brave-search_brave_local_search",
        "ddg-search_search", "ddg-search_fetch_content",
        "exa_web_search_exa", "exa_web_fetch_exa",
        "oxylabs_universal_scraper", "oxylabs_google_search_scraper",
        "oxylabs_amazon_search_scraper", "oxylabs_amazon_product_scraper"]
    assert set(DATASET_SEARCH_TOOLS) <= {t["name"] for t in M.GEMINI_SEARCH_TOOLS}
    for tool in M.GEMINI_SEARCH_TOOLS:
        assert set(tool) == {"name", "description", "inputSchema"}
        assert tool["inputSchema"]["required"] in (["query"], ["url"], ["urls"])


def test_the_environment_runs_the_release_servers_minus_brave_search():
    # minus brave-search, whose tools are all search tools: ddg-search, exa and oxylabs run,
    # for their page tools (exa and oxylabs with their credentials)
    assert M.REMOVED_SEARCH_SERVERS == ("brave-search",)
    assert set(M.ENVIRONMENT_SERVERS) == set(M.RELEASE_SERVERS) - set(M.REMOVED_SEARCH_SERVERS)
    assert len(M.ENVIRONMENT_SERVERS) == 35 and "ddg-search" in M.KEYLESS_SERVERS
    assert {"exa", "oxylabs"} <= set(M.CREDENTIALED_SERVERS)
    assert set(M.REPLACED_SEARCH_SERVICES) == set(M.GEMINI_SEARCH_SERVERS)


def test_merged_catalog_places_the_search_tools_where_the_release_lists_them():
    env = [{"name": "arxiv_search_papers"}, {"name": "calculator_calculate"},
           {"name": "desktop-commander_list_directory"}, {"name": "fetch_fetch"},
           {"name": "wikipedia_search_wikipedia"}]
    names = [t["name"] for t in M.merged_catalog(env)]
    assert names == ["arxiv_search_papers",
                     "brave-search_brave_web_search", "brave-search_brave_local_search",
                     "calculator_calculate",
                     "ddg-search_search", "ddg-search_fetch_content",
                     "desktop-commander_list_directory",
                     "exa_web_search_exa", "exa_web_fetch_exa",
                     "fetch_fetch",
                     "oxylabs_universal_scraper", "oxylabs_google_search_scraper",
                     "oxylabs_amazon_search_scraper", "oxylabs_amazon_product_scraper",
                     "wikipedia_search_wikipedia"]
    # a running server's own entry gives way to the pinned listing: each name is listed once
    other = [{"name": "exa_web_search_exa", "inputSchema": {"x": 1}}, {"name": "fetch_fetch"}]
    merged = M.merged_catalog(other)
    assert [t["name"] for t in merged].count("exa_web_search_exa") == 1
    exa = next(t for t in merged if t["name"] == "exa_web_search_exa")
    assert exa["inputSchema"] == M.GEMINI_SEARCH_TOOLS[4]["inputSchema"]
    # a copy: a caller cannot change the module's definitions
    exa["inputSchema"]["properties"].clear()
    assert M.GEMINI_SEARCH_TOOLS[4]["inputSchema"]["properties"]


def test_a_task_declares_the_verbatim_search_schema():
    catalog = M.merged_catalog([{"name": "fetch_fetch", "description": "F", "inputSchema": {}}])
    tools = M.declared_tools(["exa_web_search_exa", "fetch_fetch"], catalog)
    exa = tools[0]["function"]
    assert exa["name"] == "exa_web_search_exa" and exa["strict"] is False
    assert exa["parameters"] == M.GEMINI_SEARCH_TOOLS[4]["inputSchema"]
    assert exa["description"].startswith("Search the web for any topic")


# --------------------------------------------------------------------------- #
# replies, in each replaced server's text layout
# --------------------------------------------------------------------------- #
def test_brave_web_search_layout_and_count(grounding):
    text = _text(M.gemini_search_reply("brave-search_brave_web_search",
                                       {"query": "q", "count": 1}))
    assert text == "Title: Alpha\nDescription: alpha text\nURL: https://a.example/"
    assert grounding["asked"] == [("q", 1)]
    # the old invalid name main.py maps to this tool is served the same way
    M.gemini_search_reply("brave_brave_web_search", {"query": "q"})
    assert grounding["asked"][-1] == ("q", 10)
    assert M.is_gemini_search_tool("brave_brave_web_search")


def test_exa_search_layout_and_its_page_tool_is_not_served_by_gbench(grounding):
    text = _text(M.gemini_search_reply("exa_web_search_exa", {"query": "q"}))
    assert text.split("\n\n---\n\n")[0] == ("Title: Alpha\nURL: https://a.example/\n"
                                             "Published: N/A\nAuthor: N/A\n"
                                             "Highlights:\nalpha text")
    # web_fetch_exa runs on the release's exa server (EXA_API_KEY), never on Gemini
    assert not M.is_gemini_search_tool("exa_web_fetch_exa")
    with pytest.raises(ValueError, match="not served by gbench"):
        M.gemini_search_reply("exa_web_fetch_exa", {"urls": ["https://a.example/"]})
    assert len(grounding["asked"]) == 1, "no grounding for a page tool"
    grounding["results"] = NO_RESULTS
    assert _text(M.gemini_search_reply("exa_web_search_exa", {"query": "q"})) == (
        "No search results found. Please try a different query.")


def test_ddg_search_and_fetch_layouts(grounding):
    text = _text(M.gemini_search_reply("ddg-search_search", {"query": "q", "region": "de-de"}))
    assert text == ("Found 2 search results:\n\n1. Alpha\n   URL: https://a.example/\n"
                    "   Summary: alpha text\n\n2. Beta\n   URL: https://b.example/\n"
                    "   Summary: beta text\n")
    # fetch_content is not served by gbench: the release's server answers it in the environment
    with pytest.raises(ValueError, match="not served by gbench"):
        M.gemini_search_reply("ddg-search_fetch_content",
                              {"url": "https://a.example/", "max_length": 10})
    grounding["results"] = NO_RESULTS
    empty = _text(M.gemini_search_reply("ddg-search_search", {"query": "q"}))
    assert empty.startswith("No results were found for your search query.")
    # test_servers.py's pass rule fails an empty search, as it does upstream
    assert not M.health_call_ok("ddg-search", 200, json.dumps([{"type": "text",
                                                                "text": empty}]))


def test_oxylabs_layouts(grounding):
    parsed = json.loads(_text(M.gemini_search_reply("oxylabs_google_search_scraper",
                                                    {"query": "q", "limit": 1})))
    assert parsed == {"results": [{"pos": 1, "url": "https://a.example/", "title": "Alpha",
                                   "desc": "alpha text"}]}
    links = _text(M.gemini_search_reply("oxylabs_google_search_scraper",
                                        {"query": "q", "parse": False, "output_format": "links"}))
    assert links == "[Alpha] https://a.example/\n[Beta] https://b.example/"
    M.gemini_search_reply("oxylabs_amazon_search_scraper", {"query": "usb hub"})
    assert grounding["asked"][-1] == ("usb hub site:amazon.com", 10)
    # the page tools run on the release's oxylabs server (the Oxylabs credentials)
    for tool, args in (("oxylabs_universal_scraper", {"url": "https://a.example/"}),
                       ("oxylabs_amazon_product_scraper", {"query": "B0TEST1234"})):
        assert not M.is_gemini_search_tool(tool)
        with pytest.raises(ValueError, match="not served by gbench"):
            M.gemini_search_reply(tool, args)
    assert len(grounding["asked"]) == 3


def test_grounding_failure_is_the_environments_error_reply(grounding):
    grounding["results"] = [{"title": "error", "url": "", "snippet": "all grounding models failed"}]
    status, body, backends = M.gemini_search_reply("exa_web_search_exa", {"query": "q"})
    assert status == 500 and backends == ["gemini:gemini-test"]
    assert json.loads(body) == {"detail": "Failed to call tool 'exa_web_search_exa': Error: "
                                          "Gemini Google Search grounding failed: all grounding "
                                          "models failed"}
    assert M.harness_tool_output(status, body, "exa_web_search_exa") == (body, "error_reply")


def test_bad_arguments(grounding):
    # the inputSchema check the environment's MCP server runs (jsonschema.validate)
    status, body, _ = M.gemini_search_reply("ddg-search_search", {"max_results": 3})
    assert status == 500 and json.loads(body)["detail"] == (
        "Failed to call tool 'ddg-search_search': Input validation error: 'query' is a "
        "required property")
    status, body, _ = M.gemini_search_reply("exa_web_search_exa", {"query": "q", "numResults": "x"})
    assert status == 500 and "Input validation error: 'x' is not of type 'number'" in body
    # gbench's own check: nothing to look up
    status, body, _ = M.gemini_search_reply("ddg-search_search", {"query": "  "})
    assert status == 500 and "Invalid arguments for search: 'query' must be a non-empty" in body
    status, body, _ = M.gemini_search_reply("ddg-search_search", [])
    assert status == 422 and "Input should be a valid dictionary" in body
    assert grounding["asked"] == [], "no lookup for an invalid call"


def test_a_key_in_an_error_never_reaches_the_model(grounding, monkeypatch):
    key = "AIzaFAKE-gemini-key-0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    grounding["results"] = [{"title": "error", "url": "", "snippet": f"failed for key={key}"}]
    status, body, _ = M.gemini_search_reply("exa_web_search_exa", {"query": "q"})
    assert key not in body and base.REDACTED in body


# --------------------------------------------------------------------------- #
# tool calls: served by gbench, never by the environment
# --------------------------------------------------------------------------- #
def _serve(calls):
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
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def environment():
    calls = []
    srv, url = _serve(calls)
    yield calls, url
    srv.shutdown()
    srv.server_close()


def _call(sandbox, name, args):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(M.execute_mcp_tool(name, args))
    finally:
        M._ACTIVE_SANDBOX.reset(token)


def test_search_calls_are_served_by_gbench_with_the_environment_cache(grounding, environment):
    calls, url = environment
    sb = M._Sandbox(url)
    first = _call(sb, "brave-search_brave_web_search", {"query": "q"})
    again = _call(sb, "brave-search_brave_web_search", {"query": "q"})
    other = _call(sb, "fetch_fetch", {"url": "https://a.example/"})
    assert first == again and first.startswith("Title: Alpha")
    assert len(grounding["asked"]) == 1, "main.py caches these servers' successful replies"
    assert other == "ok" and [c["tool_name"] for c in calls] == ["fetch_fetch"]
    assert (sb.calls, sb.search_calls, sb.search_cache_hits) == (3, 2, 1)
    assert sb.search_backends == {"gemini:gemini-test": 1}


def test_search_call_errors_and_timeouts(grounding, environment, monkeypatch):
    _, url = environment
    sb = M._Sandbox(url)
    grounding["results"] = [{"title": "error", "url": "", "snippet": "quota"}]
    out = _call(sb, "ddg-search_search", {"query": "q"})
    assert out.startswith('{"detail":"Failed to call tool \'ddg-search_search\': Error: Gemini')
    assert (sb.error_replies, sb.search_error_replies) == (1, 1)
    out = _call(sb, "ddg-search_search", {"query": "q"})
    assert len(grounding["asked"]) == 2, "a failed reply is not cached"

    import time as _time
    monkeypatch.setattr(M, "_ground", lambda q, n: (_time.sleep(0.5), (HITS, "x"))[1])
    monkeypatch.setattr(M, "TOOL_CALL_TIMEOUT_S", 0.1)
    assert _call(sb, "exa_web_search_exa", {"query": "slow"}) == "Tool call timed out after 0.1s"
    assert (sb.timeouts, sb.search_timeouts) == (1, 1)


# --------------------------------------------------------------------------- #
# the pre-flight: the environment's own servers, then the search engines through Gemini
# --------------------------------------------------------------------------- #
def test_preflight_checks_the_search_engines_through_gemini(grounding, environment):
    calls, url = environment
    got = M._preflight(M._Sandbox(url))
    assert got["health_check"] == "passed (36/36 servers)"
    assert got["servers_served_by_gbench"] == ["brave-search"]
    assert got["search_tools_served_by_gbench"] == list(M.GEMINI_SERVED_TOOLS)
    assert got["tools_run_in_environment"] == list(M.UPSTREAM_SEARCH_TOOLS)
    assert got["health_check_through_gemini"] == ["brave-search", "ddg-search"]
    assert got["servers_configured"] == 35
    # brave-search's and ddg-search's check calls are search tools gbench answers with
    # grounding; exa's and oxylabs' reach their own servers (here the one environment)
    assert {c["tool_name"] for c in calls} == {
        t for s, (t, _) in M.HEALTH_CHECK_CALLS.items()
        if s in M.ENVIRONMENT_SERVERS and s not in M.GEMINI_PREFLIGHT_SERVERS}
    assert {"exa_web_search_exa", "oxylabs_google_search_scraper"} <= {
        c["tool_name"] for c in calls}
    assert len(grounding["asked"]) == 2


def test_failing_grounding_fails_the_preflight_and_names_gemini(grounding, environment):
    _, url = environment
    grounding["results"] = [{"title": "error", "url": "", "snippet": "no key"}]
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*2 of 36") as e:
        M._preflight(M._Sandbox(url, container="c"))
    msg = str(e.value)
    assert ("brave-search (not run: its search tools are served by gbench with Gemini Google "
            "Search grounding)") in msg
    assert "ddg-search (its fetch_content runs on the release's server" in msg
    assert "EXA_API_KEY" not in msg and "BRAVE_API_KEY" not in msg
    assert "Not running in the agent environment" not in msg
    assert "through the Gemini API (Google Search grounding" in msg


def test_an_external_environment_that_still_runs_a_search_engine_is_not_called(grounding):
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            servers = ([[s, "OK"] for s in M.ENVIRONMENT_SERVERS]
                       + [["brave-search", "ERROR_NOT_ONLINE"]])
            return self._send({"servers": servers, "total": 36, "online": 35, "offline": 1})

        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            return self._send([{"type": "text", "text": "ok"}])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        got = M._preflight(M._Sandbox(f"http://127.0.0.1:{srv.server_address[1]}"))
    finally:
        srv.shutdown()
        srv.server_close()
    assert "36/36" in got["health_check"]
    assert not any(c["tool_name"].startswith("brave-search_") for c in calls)


# --------------------------------------------------------------------------- #
# provisioning and credentials
# --------------------------------------------------------------------------- #
def test_prerequisites_refuse_an_image_without_this_server_template(monkeypatch):
    monkeypatch.delenv("GBENCH_MCP_ATLAS_SANDBOX_URL", raising=False)
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://db.example:27017/")
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/docker")
    search = {"value": ""}

    class _P:
        def __init__(self, out=""):
            self.returncode, self.stdout = 0, out

    def run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _P((search["value"] if M.IMAGE_SEARCH_LABEL in cmd[4]
                       else M.UPSTREAM_REVISION) + "\n")
        return _P()

    monkeypatch.setattr(M.subprocess, "run", run)
    ok, reason = M.check_mcp_atlas_prerequisites()
    assert not ok and "server template" in reason and "Rebuild" in reason
    search["value"] = "removed:brave-search,exa,oxylabs"          # built by an earlier gbench
    ok, reason = M.check_mcp_atlas_prerequisites()
    assert not ok and "'removed:brave-search'" in reason
    assert "docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker" in reason
    search["value"] = M.IMAGE_SEARCH_LABEL_VALUE
    assert M.check_mcp_atlas_prerequisites() == (True, "")


def test_no_brave_or_gemini_key_is_passed_to_the_container(monkeypatch):
    for v in {v for vs in M.RELEASE_SERVERS.values() for v in vs}:
        monkeypatch.delenv(v, raising=False)
    for name in ("BRAVE_API_KEY", "EXA_API_KEY", "OXYLABS_USERNAME", "OXYLABS_PASSWORD",
                 "GITHUB_TOKEN", "GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.setenv(name, f"value-of-{name.lower()}")
    # exa and oxylabs run for their page tools with their credentials; brave-search does not run
    assert M._container_env_args() == ["-e", "EXA_API_KEY", "-e", "GITHUB_TOKEN",
                                       "-e", "OXYLABS_PASSWORD", "-e", "OXYLABS_USERNAME"]


def test_dockerfile_removes_brave_search_and_labels_it():
    # the image removes brave-search; ddg-search, exa and oxylabs stay in the template
    assert f"ARG REMOVED_SEARCH_SERVERS={','.join(M.REMOVED_SEARCH_SERVERS)}\n" in DOCKERFILE
    assert (f'LABEL {M.IMAGE_SEARCH_LABEL}="removed:${{REMOVED_SEARCH_SERVERS}}"'
            in DOCKERFILE)
    assert M.IMAGE_SEARCH_LABEL_VALUE == "removed:" + ",".join(M.REMOVED_SEARCH_SERVERS)
    assert "mcp_server_template.json" in DOCKERFILE and "t['mcpServers'].pop(s)" in DOCKERFILE
    assert "ARG GEMINI_SEARCH_SERVERS" not in DOCKERFILE


# --------------------------------------------------------------------------- #
# a whole run records the substitution
# --------------------------------------------------------------------------- #
def test_a_run_records_the_substitution(grounding, environment, monkeypatch):
    calls, url = environment
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", url)
    # the external environment's checks, gbench's credentialed-servers container and the
    # image checks they need (tests/test_mcp_credential_isolation_and_dns.py covers them): no
    # docker here
    monkeypatch.setattr(M, "_external_environment_setup", lambda sandbox: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_MCP_ATLAS_TEMPERATURE", raising=False)
    rows = [{"TASK": "t1", "PROMPT": "Who is Alpha?",
             "ENABLED_TOOLS": '["exa_web_search_exa", "fetch_fetch"]',
             "GTFA_CLAIMS": '["Alpha is alpha text."]'}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    seen = []

    async def fake_send(**k):
        seen.append(dict(k, messages=list(k["messages"])))
        if k["messages"][-1]["role"] == "user":
            return base.Reply(text="", tool_calls=[{"id": "c1", "type": "function", "function": {
                "name": "exa_web_search_exa", "arguments": '{"query": "alpha"}'}}],
                finish_reason="tool_calls")
        return base.Reply(text="Alpha is alpha text.", tool_calls=None, finish_reason="stop")

    async def judge(prompt, config=None):
        return json.dumps({"claim_text": "c", "coverage_outcome": "fulfilled",
                           "justification": "j", "confidence_level": 1.0}), "gemini-test"

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    monkeypatch.setattr(M, "judge_generate_cascade", judge)
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)

    declared = [t["function"]["name"] for t in seen[0]["extra_payload"]["tools"]]
    assert declared == ["exa_web_search_exa", "fetch_fetch"]
    assert seen[1]["messages"][-1]["content"].startswith("Title: Alpha\nURL: https://a.example/")
    # the model's search call is served by gbench; only the pre-flight's exa check reaches the
    # exa server (here the one environment)
    exa = [c for c in calls if c["tool_name"].startswith("exa_")]
    assert exa == [{"tool_name": "exa_web_search_exa",
                    "tool_args": {"query": "python programming"}}]
    sub = res["search_substitution"]
    assert sub["servers"] == list(M.GEMINI_SEARCH_SERVERS)
    assert sub["calls"] == 1 and sub["grounding_backends"] == {"gemini:gemini-test": 1}
    assert sub["not_read"] == ["BRAVE_API_KEY"]
    assert sub["upstream_tools"] == list(M.UPSTREAM_SEARCH_TOOLS)
    assert sub["tools"] == sub["search_tools"] == list(M.GEMINI_SERVED_TOOLS)
    assert set(sub["release_servers"]) == set(M.GEMINI_SEARCH_SERVERS)
    assert res["leaderboard_comparable"] is False
    assert "replaced by Gemini Google Search grounding" in res["leaderboard_comparable_reason"]
    assert res["agent_environment"]["catalog_tools"] == 1 + len(M.GEMINI_SEARCH_TOOLS)
    assert res["agent_environment"]["enabled_tools_not_in_environment"] == []


# --------------------------------------------------------------------------- #
# the doc states it
# --------------------------------------------------------------------------- #
def test_doc_states_the_substitution():
    for needle in ("Gemini Google Search grounding", "`search_substitution`",
                   M.IMAGE_SEARCH_LABEL, "not read", "`GEMINI_API_KEYS`",
                   "same tool names and argument schemas", "site:amazon.com"):
        assert needle in DOC, needle
    for server in M.GEMINI_SEARCH_SERVERS:
        assert server in DOC, server
    for tool in M.GEMINI_SEARCH_TOOLS:
        assert f"`{tool['name']}`" in DOC, tool["name"]
    assert "There is no reduced-server, stub or web-search substitute mode" not in DOC
