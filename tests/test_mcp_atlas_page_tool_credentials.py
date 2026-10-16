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

"""mcp_atlas: Exa and Oxylabs page tools on the release's servers, every credential required.

* exa's web_fetch_exa and oxylabs' universal_scraper / amazon_product_scraper run on the
  release's exa-mcp-server 3.2.1 and oxylabs-mcp 0.4.1 in the credentialed-servers container,
  with EXA_API_KEY and OXYLABS_USERNAME / OXYLABS_PASSWORD; only the search tools (brave-search's
  two, ddg-search's, exa's and oxylabs' two) are served by Gemini Google Search grounding, and
  the run records that as not comparable. The image removes only brave-search.
* A credentialed server with any variable unset stops the suite with infra_required before any
  container starts and before generation, naming exactly the unset variables per server and the
  release sample data the data servers need.
* The model's declarations for the search-engine servers are the pinned listing, and a running
  server whose listing differs stops the run (catalog_drift).
* A page-tool reply that says the server could not reach the Exa or Oxylabs API is a harness
  error; every other page-tool reply is the model's outcome.
* docs/evals/mcp_atlas.md lists every credential with where to get it, and the run command.

Docker, Gemini and the network are stubbed; nothing is written outside tmp.
"""

import asyncio
import json
import re
import subprocess
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import search_tool

REPO = Path(__file__).resolve().parents[1]
DOC_RAW = (REPO / "docs" / "evals" / "mcp_atlas.md").read_text(encoding="utf-8")
DOC = " ".join(DOC_RAW.split())
DOCKERFILE = (REPO / "docker" / "mcp_atlas.Dockerfile").read_text(encoding="utf-8")
SECRET = "cred-secret-value-4242"

#: Every variable a credentialed server of the environment needs.
CRED_VARS = sorted({v for s in M.CREDENTIALED_SERVERS for v in M.ENVIRONMENT_SERVERS[s]})


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test reached a real Gemini request")

    monkeypatch.setattr(search_tool, "gemini_search", refuse)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKE-mcp-atlas-0123456789")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GBENCH_MCP_ATLAS_SANDBOX_URL", raising=False)


def _all_credentials(monkeypatch):
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    for var in CRED_VARS:
        if var != "MONGODB_CONNECTION_STRING":
            monkeypatch.setenv(var, f"{var}-{SECRET}")


class _Docker:
    """subprocess.run stand-in: records every command, answers as a healthy docker would."""

    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, "1\n", "")
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        return subprocess.CompletedProcess(cmd, 0, "id\n", "")


def _call(sandbox, name, args, idx=None):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(M.execute_mcp_tool(name, args, idx))
    finally:
        M._ACTIVE_SANDBOX.reset(token)


# --------------------------------------------------------------------------- #
# the server split: exa and oxylabs run, with their credentials
# --------------------------------------------------------------------------- #
def test_exa_and_oxylabs_run_in_the_credentialed_servers_container():
    assert M.REMOVED_SEARCH_SERVERS == ("brave-search",)
    assert M.ENVIRONMENT_SEARCH_SERVERS == ("ddg-search", "exa", "oxylabs")
    assert M.ENVIRONMENT_SERVERS["exa"] == ("EXA_API_KEY",)
    assert set(M.ENVIRONMENT_SERVERS["oxylabs"]) == {"OXYLABS_USERNAME", "OXYLABS_PASSWORD"}
    assert "exa" in M.CREDENTIALED_SERVERS and "oxylabs" in M.CREDENTIALED_SERVERS
    assert "brave-search" not in M.ENVIRONMENT_SERVERS
    assert (len(M.ENVIRONMENT_SERVERS), len(M.CREDENTIALED_SERVERS), len(M.KEYLESS_SERVERS)) == (
        35, 15, 20)
    # their providers' API hosts are resolved before the credentialed container starts
    assert {"api.exa.ai", "realtime.oxylabs.io"} <= set(M.CREDENTIALED_HOSTS)
    assert not {"api.exa.ai", "realtime.oxylabs.io"} & set(M.ENVIRONMENT_HOSTS)


def test_only_search_tools_are_served_by_gemini():
    assert M.GEMINI_SERVED_TOOLS == (
        "brave-search_brave_web_search", "brave-search_brave_local_search", "ddg-search_search",
        "exa_web_search_exa", "oxylabs_google_search_scraper", "oxylabs_amazon_search_scraper")
    assert M.UPSTREAM_SEARCH_TOOLS == ("ddg-search_fetch_content", "exa_web_fetch_exa",
                                       "oxylabs_universal_scraper",
                                       "oxylabs_amazon_product_scraper")
    for tool in M.GEMINI_SERVED_TOOLS:
        assert M.is_gemini_search_tool(tool)
    for tool in M.UPSTREAM_SEARCH_TOOLS:
        assert not M.is_gemini_search_tool(tool)


def test_the_dockerfile_removes_only_brave_search():
    assert "ARG REMOVED_SEARCH_SERVERS=brave-search\n" in DOCKERFILE
    assert 'LABEL gbench.mcp_atlas.search_servers="removed:${REMOVED_SEARCH_SERVERS}"' in DOCKERFILE
    assert M.IMAGE_SEARCH_LABEL_VALUE == "removed:brave-search"
    assert "gbench runs 35 of the 36" in DOCKERFILE


def test_the_container_env_carries_exa_and_oxylabs_credentials_by_name_only(monkeypatch):
    _all_credentials(monkeypatch)
    monkeypatch.setenv("BRAVE_API_KEY", f"brave-{SECRET}")
    monkeypatch.setenv("OXYLABS_AI_STUDIO_API_KEY", f"ai-studio-{SECRET}")
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    monkeypatch.setattr(CG, "require_metadata_blocked", lambda *a, **k: CG.UNREACHABLE)
    sb = M._start_container("gbench-mcp-atlas:f24ba3fb")
    runs = [c for c in docker.cmds if c[:3] == ["docker", "run", "-d"]]
    [cred] = [c for c in runs if c[c.index("--name") + 1] == sb.cred_container]
    [env] = [c for c in runs if c[c.index("--name") + 1] == sb.container]
    names = [cred[i + 1] for i, a in enumerate(cred) if a == "-e"]
    for var in ("EXA_API_KEY", "OXYLABS_USERNAME", "OXYLABS_PASSWORD"):
        assert var in names, var
        assert not any(var in a for a in env), var
    assert "BRAVE_API_KEY" not in " ".join(cred + env)
    # the release's template passes only the username and password to oxylabs
    assert "OXYLABS_AI_STUDIO_API_KEY" not in " ".join(cred + env)
    assert SECRET not in " ".join(cred + env), "values stay off argv"
    enabled = next(a for a in cred if a.startswith("ENABLED_SERVERS=")).split("=", 1)[1]
    assert {"exa", "oxylabs"} <= set(enabled.split(","))
    env_enabled = next(a for a in env if a.startswith("ENABLED_SERVERS=")).split("=", 1)[1]
    assert not {"exa", "oxylabs", "brave-search"} & set(env_enabled.split(","))


# --------------------------------------------------------------------------- #
# every credential required, named per server, before anything starts
# --------------------------------------------------------------------------- #
_CASES = [(s, v) for s in M.CREDENTIALED_SERVERS for v in M.ENVIRONMENT_SERVERS[s]
          if v != "MONGODB_CONNECTION_STRING"]


@pytest.mark.parametrize("server,var", _CASES, ids=[f"{s}:{v}" for s, v in _CASES])
def test_each_unset_variable_stops_the_run_naming_it_and_its_server(server, var, monkeypatch):
    _all_credentials(monkeypatch)
    monkeypatch.delenv(var)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    with pytest.raises(RuntimeError) as ei:
        M._start_container("gbench-mcp-atlas:f24ba3fb")
    msg = str(ei.value)
    assert "INFRA REQUIRED" in msg
    assert f"1 of the {len(M.CREDENTIALED_SERVERS)} MCP servers that need credentials" in msg
    assert f"{server}: {var} unset" in msg
    for other in M.ENVIRONMENT_SERVERS[server]:
        if other != var:
            assert other not in msg, "only the unset variable is named"
    if server in M.DATA_SETUP:
        assert f"(it also needs {M.DATA_SETUP[server]})" in msg
    assert docker.cmds == [], "no container (or network) was started"
    assert M.credential_gaps() == {server: [var]}


def test_several_unset_variables_are_named_per_server(monkeypatch):
    _all_credentials(monkeypatch)
    for var in ("EXA_API_KEY", "OXYLABS_PASSWORD", "AIRTABLE_API_KEY"):
        monkeypatch.delenv(var)
    with pytest.raises(RuntimeError) as ei:
        M._require_credentials()
    msg = str(ei.value)
    assert "3 of the 15 MCP servers that need credentials" in msg
    assert "exa: EXA_API_KEY unset;" in msg
    assert "oxylabs: OXYLABS_PASSWORD unset" in msg and "OXYLABS_USERNAME" not in msg
    assert ("airtable: AIRTABLE_API_KEY unset (it also needs the release's Airtable base copied "
            "into that account") in msg
    assert "appIF9byLfQwdHqE2/shr1KTZOgPl0qQmA8" in msg
    assert M.DOCS_URL in msg


def test_data_setup_names_the_release_data_exports():
    assert M.DATA_SETUP_SERVERS == ("airtable", "google-workspace", "mongodb", "notion",
                                    "slack")
    assert "calendar_mcp_eval_export.zip" in M.DATA_SETUP["google-workspace"]
    assert "mcp-atlas-notion-data.zip" in M.DATA_SETUP["notion"]
    assert "slack_mcp_eval_export.zip" in M.DATA_SETUP["slack"]
    assert "mongo_dump_video_game_store-UNZIP-FIRST.zip" in M.DATA_SETUP["mongodb"]


def test_an_external_environment_needs_every_credential_too(monkeypatch):
    _all_credentials(monkeypatch)
    monkeypatch.delenv("OXYLABS_USERNAME")
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", "http://127.0.0.1:9")
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    monkeypatch.setattr(M, "_http", lambda *a, **k: pytest.fail("contacted the environment"))
    with pytest.raises(RuntimeError, match=r"oxylabs: OXYLABS_USERNAME unset"):
        M._open_sandbox()
    assert docker.cmds == []


def test_a_run_with_an_unset_credential_never_generates(monkeypatch):
    _all_credentials(monkeypatch)
    monkeypatch.delenv("EXA_API_KEY")
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    monkeypatch.setattr(M, "run_eval_suite", lambda **kw: pytest.fail("generation started"))
    monkeypatch.setattr(M, "_load_mcp_atlas_samples",
                        lambda **kw: pytest.fail("the samples were loaded"))
    with pytest.raises(RuntimeError, match=r"exa: EXA_API_KEY unset"):
        M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    assert not any(c[:3] == ["docker", "run", "-d"] for c in docker.cmds)


# --------------------------------------------------------------------------- #
# the pre-flight: exa and oxylabs checked on their own servers
# --------------------------------------------------------------------------- #
def test_the_preflight_sends_exa_and_oxylabs_checks_to_their_servers(monkeypatch):
    cred_calls, env_calls, grounded = [], [], []

    async def exec_http(container, method, path, timeout, body=None):
        cred_calls.append(body["tool_name"])
        return 200, json.dumps([{"type": "text", "text": f"ok {SECRET}"}])

    class _Resp:
        status = 200

        async def text(self):
            return json.dumps([{"type": "text", "text": "ok"}])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, json=None, timeout=None):
            env_calls.append(json["tool_name"])
            return _Resp()

    import aiohttp
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: _Session())
    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "_ground", lambda q, n: (grounded.append(q) or (
        [{"title": "T", "url": "https://t.example/", "snippet": "S"}], "gemini:g")))
    results = asyncio.run(M._run_health_calls("http://env", "cred", M.CREDENTIALED_SERVERS))
    assert all(ok for ok, _ in results.values()) and len(results) == 36
    assert "exa_web_search_exa" in cred_calls and "oxylabs_google_search_scraper" in cred_calls
    assert sorted(grounded) == sorted([M.HEALTH_CHECK_CALLS["brave-search"][1]["query"],
                                       M.HEALTH_CHECK_CALLS["ddg-search"][1]["query"]])
    assert set(env_calls) == {t for s, (t, _) in M.HEALTH_CHECK_CALLS.items()
                              if s in M.KEYLESS_SERVERS and s != "ddg-search"}


def test_a_failing_exa_check_names_its_key_not_gemini(monkeypatch):
    results = {s: (True, "") for s in M.RELEASE_SERVERS}
    results["exa"] = (False, '{"detail":"Failed to call tool \'exa_web_search_exa\': 401"}')
    msg = M.preflight_failure_reason(results, local_env=True,
                                     enabled=list(M.ENVIRONMENT_SERVERS))
    assert "Running but failing their check: exa (needs EXA_API_KEY" in msg
    assert "its web_fetch_exa runs on the release's server in the credentialed-servers " \
           "container" in msg
    assert "through the Gemini API" not in msg


def test_a_credential_in_a_preflight_detail_is_masked(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", f"exa-{SECRET}")

    async def exec_http(container, method, path, timeout, body=None):
        return 200, json.dumps({"error": f"bad key exa-{SECRET}"})

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "HEALTH_CHECK_CALLS", {"exa": M.HEALTH_CHECK_CALLS["exa"],
                                                  "oxylabs": M.HEALTH_CHECK_CALLS["oxylabs"]})
    monkeypatch.setattr(M, "RELEASE_SERVERS", {"exa": ("EXA_API_KEY",),
                                               "oxylabs": M.RELEASE_SERVERS["oxylabs"]})
    results = asyncio.run(M._run_health_calls("http://env", "cred", ("exa", "oxylabs")))
    ok, detail = results["exa"]
    assert not ok and SECRET not in detail and base.REDACTED in detail


# --------------------------------------------------------------------------- #
# the catalog: the pinned listing, and the running servers must agree with it
# --------------------------------------------------------------------------- #
def _live(tool, **extra):
    entry = dict(json.loads(json.dumps(tool)))
    entry.update({"title": None, "annotations": {"readOnlyHint": True}, "_meta": None}, **extra)
    return entry


def test_catalog_drift_accepts_the_pinned_listing():
    live = [_live(t) for t in M.GEMINI_SEARCH_TOOLS if t["name"].split("_", 1)[0]
            in M.ENVIRONMENT_SEARCH_SERVERS] + [{"name": "fetch_fetch", "description": "x"}]
    assert M.catalog_drift(live) == []


def test_catalog_drift_names_a_changed_or_unknown_tool():
    fetch = next(t for t in M.GEMINI_SEARCH_TOOLS if t["name"] == "exa_web_fetch_exa")
    changed = _live(fetch)
    changed["inputSchema"] = dict(changed["inputSchema"], required=["urls", "maxCharacters"])
    extra = {"name": "oxylabs_ai_scraper", "description": "AI Studio", "inputSchema": {}}
    problems = M.catalog_drift([changed, extra, {"name": "github_list_commits"}])
    # exa and oxylabs run (they list an entry), so their pinned tools they do not list are named
    missing = [f"{t['name']}: in the pinned listing but not listed by the running server"
               for t in M.GEMINI_SEARCH_TOOLS
               if t["name"].split("_", 1)[0] in ("exa", "oxylabs")
               and t["name"] != "exa_web_fetch_exa"]
    assert problems == [
        "exa_web_fetch_exa: its description or inputSchema differs from the pinned listing",
        "oxylabs_ai_scraper: not in the pinned listing"] + missing


def test_a_run_whose_servers_drift_stops_before_the_samples_load(monkeypatch):
    sb = M._Sandbox("http://env")
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_open_sandbox", lambda: sb)
    monkeypatch.setattr(M, "_wait_ready", lambda s, t: None)
    monkeypatch.setattr(M, "_preflight", lambda s: {"health_check": "passed"})
    monkeypatch.setattr(M, "_stop_container", lambda s: None)
    monkeypatch.setattr(M, "_list_tools", lambda s: [
        {"name": "exa_web_fetch_exa", "description": "changed", "inputSchema": {}}])
    monkeypatch.setattr(M, "_load_mcp_atlas_samples",
                        lambda **kw: pytest.fail("the samples were loaded"))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*exa_web_fetch_exa: its "
                                           r"description or inputSchema differs"):
        M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)


def test_the_declared_page_tools_are_the_pinned_listing():
    catalog = M.merged_catalog([{"name": "oxylabs_universal_scraper", "description": "live",
                                 "inputSchema": {"type": "object"}},
                                {"name": "fetch_fetch", "description": "F", "inputSchema": {}}])
    [fn] = [t["function"] for t in M.declared_tools(["oxylabs_universal_scraper"], catalog)]
    pinned = next(t for t in M.GEMINI_SEARCH_TOOLS if t["name"] == "oxylabs_universal_scraper")
    assert fn["parameters"] == pinned["inputSchema"] and fn["description"] == pinned["description"]


# --------------------------------------------------------------------------- #
# a page tool whose server cannot reach its provider: a harness error
# --------------------------------------------------------------------------- #
#: What the rebuilt image's servers answered with no network (docker run --network none,
#: placeholder credentials): exa as an error reply, oxylabs as a text reply.
_EXA_UNREACHABLE = (500, json.dumps(
    {"detail": "Failed to call tool 'exa_web_fetch_exa': web_fetch_exa error: fetch failed"}))
_OXY_UNREACHABLE = (200, json.dumps([{"type": "text", "text": (
    "Request error during POST request: [Errno -3] Temporary failure in name resolution")}]))


@pytest.mark.parametrize("tool,args,reply", [
    ("exa_web_fetch_exa", {"urls": ["https://a.example/"]}, _EXA_UNREACHABLE),
    ("oxylabs_universal_scraper", {"url": "https://a.example/"}, _OXY_UNREACHABLE),
    ("oxylabs_amazon_product_scraper", {"query": "B0TEST1234"}, _OXY_UNREACHABLE),
])
def test_a_provider_the_server_cannot_reach_is_a_harness_error(tool, args, reply, monkeypatch):
    async def exec_http(container, method, path, timeout, body=None):
        return reply

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    sb = M._Sandbox("http://127.0.0.1:9", cred_container="cred",
                    cred_servers=M.CREDENTIALED_SERVERS)
    out = _call(sb, tool, args, 6)
    assert out, "the model still gets the server's reply, as upstream"
    assert sb.harness_error_samples == {6: 1} and sb.provider_unreachable == {tool: 1}


@pytest.mark.parametrize("tool,text", [
    ("exa_web_fetch_exa", '{"detail":"Failed to call tool \'exa_web_fetch_exa\': Error fetching '
                          'URL(s): https://u.example/: CRAWL_NOT_FOUND"}'),
    ("exa_web_fetch_exa", "No content found for the provided URL(s)."),
    ("oxylabs_universal_scraper", "Error: 'job'"),
    ("oxylabs_universal_scraper", "# A page that mentions Request error during POST request: x"),
    ("fetch_fetch", "Request error during POST request: [Errno -3]"),
])
def test_other_page_replies_are_not_unreachable_replies(tool, text):
    assert not M.provider_unreachable(tool, text)
    if (tool, text) == ("oxylabs_universal_scraper", "Error: 'job'"):
        # a provider account error (a harness error of its own kind), not an unreachable provider
        assert M.provider_account_error(tool, text)
    else:
        assert not M.provider_account_error(tool, text), "the model's outcome"


def test_a_run_names_provider_unreachable_calls(monkeypatch):
    sb = M._Sandbox("http://env", cred_container="cred", cred_servers=M.CREDENTIALED_SERVERS)
    sb.provider_unreachable = {"oxylabs_universal_scraper": 1}
    sb.harness_error_samples = {1: 1}
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_open_sandbox", lambda: sb)
    monkeypatch.setattr(M, "_wait_ready", lambda s, t: None)
    monkeypatch.setattr(M, "_preflight", lambda s: {"health_check": "passed"})
    monkeypatch.setattr(M, "_list_tools", lambda s: [{"name": "fetch_fetch"}])
    monkeypatch.setattr(M, "_stop_container", lambda s: None)
    monkeypatch.setattr(M, "environment_alive", lambda s: (True, ""))
    monkeypatch.setattr(M, "_load_mcp_atlas_samples", lambda limit=None, catalog=None: [])
    monkeypatch.setattr(M, "run_eval_suite", lambda **kw: {
        "status": "success", "sample_traces": [
            {"sample_idx": 0, "category": "oxylabs", "claim_recall": 1.0, "is_correct": True,
             "claim_scores": [1.0], "judge_models": ["g"]},
            {"sample_idx": 1, "category": "oxylabs", "judge_grade": M.HARNESS_ERROR_GRADE,
             "scoring_excluded": True, "is_correct": False}]})
    res = M.run_mcp_atlas("m", "http://x", concurrency=1)
    assert res["status"] == "completed_with_errors"
    assert res["harness_errors"][0]["provider_unreachable_calls"] == {
        "oxylabs_universal_scraper": 1}
    assert res["search_substitution"]["upstream_provider_unreachable"] == {
        "oxylabs_universal_scraper": 1}
    assert "1 page tool call(s) whose server could not reach its provider's API" in (
        res["leaderboard_comparable_reason"])


# --------------------------------------------------------------------------- #
# the per-claim judge model
# --------------------------------------------------------------------------- #
def test_each_claims_judge_model_is_recorded(monkeypatch):
    models = iter(["gemini-a", "gemini-b"])

    async def judge(prompt, config=None):
        return json.dumps({"coverage_outcome": "fulfilled"}), next(models)

    monkeypatch.setattr(M, "judge_generate_cascade", judge)
    tr = [{"sample_idx": 0, "gold_answer": '["The answer is 42.", "It is even."]',
           "response_text": "42, which is even."}]
    asyncio.run(M._async_judge_mcp_atlas(tr, judge_concurrency=1))
    assert tr[0]["claim_judge_models"] == ["gemini-a", "gemini-b"]
    assert tr[0]["judge_models"] == ["gemini-a", "gemini-b"]
    assert len(tr[0]["claim_scores"]) == 2


# --------------------------------------------------------------------------- #
# the doc: every credential, where to get it, and the run command
# --------------------------------------------------------------------------- #
def _requirements_rows():
    section = DOC_RAW[DOC_RAW.index("## Requirements"):DOC_RAW.index("## Env knobs")]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.strip().startswith("|") and len(cells) == 5 and cells[0] not in ("Server", "---"):
            rows[cells[0]] = cells
    return rows


def test_the_doc_lists_every_credential_with_where_to_get_it():
    rows = _requirements_rows()
    assert set(rows) == set(M.CREDENTIALED_SERVERS)
    for server, (_, variables, runs_with, where, data) in rows.items():
        for var in M.ENVIRONMENT_SERVERS[server]:
            assert f"`{var}`" in variables, (server, var)
        assert runs_with, server
        assert re.search(r"\((https?://[^)]+)\)", where) or server == "mongodb", server
        assert re.search(r"\bfree\b|\bpaid\b", where), (server, "free or paid")
        if server in M.DATA_SETUP and server != "mongodb":
            assert data, (server, "sample data")
    assert "paid, with a free trial" in rows["oxylabs"][3]
    assert "free tier" in rows["exa"][3]
    assert "`web_fetch_exa`" in rows["exa"][2]
    assert "`universal_scraper`" in rows["oxylabs"][2]
    assert "`amazon_product_scraper`" in rows["oxylabs"][2]


def test_the_doc_states_the_rules_and_the_run_command():
    assert "`BRAVE_API_KEY` is not read" in DOC
    assert "if any is unset, the suite stops with `infra_required` before any container starts" \
        in DOC
    assert "no server is dropped and there is no partial run" in DOC
    run = DOC_RAW[DOC_RAW.index("## Run"):]
    for var in CRED_VARS:
        assert var in run, var
    assert "--evals mcp_atlas" in run and "--max-output-tokens" in run
    assert "docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker" in run
    for host in M.CREDENTIALED_HOSTS + M.ENVIRONMENT_HOSTS:
        assert f"`{host}`" in DOC, host
    for server in M.KEYLESS_SERVERS + M.CREDENTIALED_SERVERS:
        assert server in DOC, server
    assert "the 20 servers that need no credential" in DOC
    assert "runs the 15 that need one" in DOC
    assert "35 of them, all but brave-search" in DOC
