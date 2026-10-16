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

"""mcp_atlas: a provider-side HTTP 5xx of the exa and oxylabs page tools is a harness error;
the other credentialed servers' account-error replies are counted; identifier-style credential
values are masked; a failed prerequisite check names the unset credentials too; the catalog check
names a pinned tool a running search server does not list. Docker, Gemini and the network are
stubbed; nothing is written outside tmp."""

import asyncio
import json

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import mcp_atlas as M


def _exa_reply(error, tool="exa_web_fetch_exa"):
    return 500, json.dumps({"detail": f"Failed to call tool '{tool}': {error}"})


def _oxy_reply(text):
    return 200, json.dumps([{"type": "text", "text": text}])


def _sandbox():
    return M._Sandbox("http://127.0.0.1:9", cred_container="cred",
                      cred_servers=M.CREDENTIALED_SERVERS)


def _call(sandbox, name, args, idx=None):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(M.execute_mcp_tool(name, args, idx))
    finally:
        M._ACTIVE_SANDBOX.reset(token)


def _replying(monkeypatch, reply):
    async def exec_http(container, method, path, timeout, body=None):
        return reply
    monkeypatch.setattr(M, "_exec_http_async", exec_http)


@pytest.mark.parametrize("tool, reply, status", [
    ("exa_web_fetch_exa", _exa_reply("web_fetch_exa error (503): SERVICE_OVERLOADED"), "HTTP 503"),
    ("exa_web_fetch_exa", _exa_reply("web_fetch_exa error (500): Internal error"), "HTTP 500"),
    ("exa_web_fetch_exa", _exa_reply("web_fetch_exa error (504): Gateway timeout"), "HTTP 504"),
    ("oxylabs_universal_scraper", _oxy_reply("HTTP error during POST request: 502 - Bad Gateway"),
     "HTTP 502"),
])
def test_a_provider_5xx_is_a_harness_error(monkeypatch, tool, reply, status):
    _replying(monkeypatch, reply)
    sb = _sandbox()
    _call(sb, tool, {"x": 1}, 3)
    assert sb.provider_unavailable == {f"{tool}: {status}": 1}
    assert sb.harness_error_samples == {3: 1} and sb.provider_account_errors == {}


def test_the_5xx_signature_is_the_servers_own_error_only():
    # a page that quotes it is the page
    page = M.harness_tool_output(*_oxy_reply("Users see 'web_fetch_exa error (503): x'"),
                                 "exa_web_fetch_exa")[0]
    assert M.provider_unavailable("exa_web_fetch_exa", page) == ""
    assert M.provider_unavailable("exa_web_fetch_exa",
                                  _exa_reply("web_fetch_exa error (400): bad")[1]) == ""
    assert M.provider_unavailable("fetch_fetch", "HTTP error during POST request: 503 - x") == ""


def test_the_preflight_fails_a_check_whose_provider_is_down(monkeypatch):
    async def exec_http(container, method, path, timeout, body=None):
        if body["tool_name"].startswith("exa_"):
            return 200, '[{"type":"text","text":"Title: r"}]'
        return _oxy_reply("HTTP error during POST request: 503 - Service Unavailable")
    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "gemini_search_reply",
                        lambda *a, **k: (200, '[{"type":"text","text":"r"}]', []))
    res = asyncio.run(M._run_health_calls("http://127.0.0.1:1", "cred", ("exa", "oxylabs")))
    ok, detail = res["oxylabs"]
    assert not ok and detail.startswith("the provider's API failed on its side (HTTP 503): ")


@pytest.mark.parametrize("tool, reply, counted", [
    ("github_list_commits", (500, json.dumps({"detail": "Failed to call tool "
                                              "'github_list_commits': GitHub API error: 401 "
                                              "Bad credentials"})), True),
    ("twelvedata_GetPrice", _oxy_reply('Error: {"code": 429, "message": "You have run out of API '
                                       'credits for the current minute."}'), True),
    ("slack_conversations_history", _oxy_reply("Error: invalid_auth"), True),
    # a successful reply that mentions a rate limit is the content, not an error
    ("notion_API-post-search", _oxy_reply("Page: API rate limit policy for the team"), False),
    # the model's own failure
    ("github_list_commits", (500, json.dumps({"detail": "Failed to call tool "
                                              "'github_list_commits': Not Found"})), False),
])
def test_other_credentialed_servers_account_errors_are_counted_not_excluded(
        monkeypatch, tool, reply, counted):
    _replying(monkeypatch, reply)
    sb = _sandbox()
    _call(sb, tool, {"x": 1}, 2)
    assert sb.harness_error_samples == {}
    assert sb.account_error_suspects == ({M.tool_server(tool): 1} if counted else {})


def test_identifier_style_credentials_are_masked(monkeypatch):
    user, client_id = "oxy_user_7c1e9d4b", "1234abcd-client.apps.googleusercontent.com"
    monkeypatch.setattr(base, "_REGISTERED_SECRETS", set())     # this test's registrations only
    for var in M.RELEASE_SERVERS["oxylabs"] + M.RELEASE_SERVERS["google-workspace"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OXYLABS_USERNAME", user)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", client_id)
    assert not base.is_secret_env_name("OXYLABS_USERNAME")
    M.register_credential_values()
    _replying(monkeypatch, _oxy_reply(f"Job for user {user} failed; client {client_id}"))
    out = _call(_sandbox(), "oxylabs_universal_scraper", {"url": "https://a.example/"}, 1)
    assert user not in out and client_id not in out and base.REDACTED in out
    assert user not in base.redact_text(f"log: {user}")


def test_a_failed_prerequisite_check_also_names_unset_credentials(monkeypatch):
    for var in {v for vs in M.RELEASE_SERVERS.values() for v in vs}:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites",
                        lambda: (False, "image gbench-mcp-atlas:f24ba3fb not found."))
    monkeypatch.setattr(M, "_open_sandbox", lambda: pytest.fail("a sandbox was started"))
    with pytest.raises(RuntimeError) as ei:
        M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1)
    msg = str(ei.value)
    assert "image gbench-mcp-atlas:f24ba3fb not found." in msg
    assert "Also," in msg and "exa: EXA_API_KEY" in msg and "slack: SLACK_MCP_XOXC_TOKEN" in msg


def _live(tool):
    return dict(json.loads(json.dumps(tool)))


def test_a_running_search_server_that_lists_fewer_tools_is_drift():
    exa = [_live(t) for t in M.GEMINI_SEARCH_TOOLS if t["name"] == "exa_web_fetch_exa"]
    ddg = [_live(t) for t in M.GEMINI_SEARCH_TOOLS if t["name"].startswith("ddg-search_")]
    oxy = [_live(t) for t in M.GEMINI_SEARCH_TOOLS if t["name"].startswith("oxylabs_")]
    problems = M.catalog_drift(exa + ddg + oxy)
    assert problems == ["exa_web_search_exa: in the pinned listing but not listed by the "
                        "running server"]
    # a server that lists nothing is offline (the pre-flight's business), not drift
    assert M.catalog_drift(ddg + oxy) == []


def test_the_guide_describes_the_new_rules():
    doc = " ".join(open("docs/evals/mcp_atlas.md", encoding="utf-8").read().split())
    for needle in ("`provider_unavailable_calls`", "`search_substitution.upstream_provider_"
                   "unavailable`", "`tool_calls.account_error_suspects`",
                   "Known gap: the other credentialed servers' account errors",
                   "a pinned tool of a running server", "a provider-side error page"):
        assert needle in doc, needle
