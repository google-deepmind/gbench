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

"""Provider account errors and pre-run key checks of mcp_atlas, mcp_bench and complexfuncbench.

* mcp_atlas: a reply of the exa or oxylabs page tools that says the provider refused the run's
  account (rejected credentials, exhausted credit or plan, the rate limit) is a harness error
  (counted per tool and reason, the task not scored), like a provider the server cannot reach, and
  so is oxylabs' error reply with no job record ("Error: 'job'"); a reply that could be the
  model's own (Exa's 403, a 400 with its status) stays the model's outcome; the exa signatures
  only match the environment's error reply, never page text;
  the pre-flight fails an exa or oxylabs check whose reply says either; a run whose every task
  with an outcome is a harness error withholds its score.
* mcp_bench: every set server key is sent once to the service its server calls before any
  container starts; a refused key, or one with no answer of the service's own after the retries,
  stops the suite with infra_required naming the key, never its value.
* complexfuncbench: the documented local-server command carries the --num-gpus the CLI requires.

Docker, Gemini, the providers and the network are stubbed; nothing is written outside tmp. The
replies used below are the ones the rebuilt image's servers and the live services gave to a
placeholder key.
"""

import asyncio
import json
import os
import re
import shlex
import subprocess
import types

import pytest

from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import mcp_bench as MB

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def _flat(text):
    return " ".join(text.split())


# =========================================================================== #
# mcp_atlas: the page tools' provider account errors
# =========================================================================== #
#: The agent environment's reply to exa_web_fetch_exa with a rejected EXA_API_KEY (the image's
#: exa-mcp-server 3.2.1 against the live Exa API), verbatim.
_EXA_401 = (500, '{"detail":"Failed to call tool \'exa_web_fetch_exa\': web_fetch_exa error '
                 '(401): Invalid API key. Provide a valid key using \'Authorization: Bearer '
                 '<key>\' or \'x-api-key: <key>\'. Create a key at '
                 'https://dashboard.exa.ai/api-keys\\nTimestamp: 2026-09-28T13:41:06.739Z"}')
#: The environment's reply to oxylabs_universal_scraper with rejected Oxylabs credentials (the
#: live API answers HTTP 401 with an empty body), verbatim.
_OXY_401 = (200, '[{"type":"text","text":"Error: Expecting value: line 1 column 1 (char 0)",'
                 '"annotations":null,"_meta":null}]')


def _exa_reply(error, tool="exa_web_fetch_exa"):
    return 500, json.dumps({"detail": f"Failed to call tool '{tool}': {error}"})


def _oxy_reply(text):
    return 200, json.dumps([{"type": "text", "text": text}])


def _call(sandbox, name, args, idx=None):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(M.execute_mcp_tool(name, args, idx))
    finally:
        M._ACTIVE_SANDBOX.reset(token)


def _sandbox():
    return M._Sandbox("http://127.0.0.1:9", cred_container="cred",
                      cred_servers=M.CREDENTIALED_SERVERS)


@pytest.mark.parametrize("tool,args,reply,reason", [
    ("exa_web_fetch_exa", {"urls": ["https://a.example/"]}, _EXA_401, "HTTP 401"),
    ("exa_web_fetch_exa", {"urls": ["https://a.example/"]},
     _exa_reply("web_fetch_exa error (402): You have run out of credits.\nTimestamp: t"),
     "HTTP 402"),
    ("exa_web_fetch_exa", {"urls": ["https://a.example/"]},
     _exa_reply("web_fetch_exa error (429): Rate limit exceeded"), "HTTP 429"),
    ("exa_web_fetch_exa", {"urls": ["https://a.example/"]},
     _exa_reply("You've hit Exa's free MCP rate limit. To continue using without limits, create "
                "your own Exa API key.\n\nFix: Create API key at "
                "https://dashboard.exa.ai/api-keys"),
     "HTTP 429 (the server's rate-limit reply)"),
    ("oxylabs_universal_scraper", {"url": "https://a.example/"}, _OXY_401,
     "no JSON body (the API's refusal of the credentials, HTTP 401, or a provider-side error "
     "page)"),
    ("oxylabs_amazon_product_scraper", {"query": "B0TEST1234"}, _OXY_401,
     "no JSON body (the API's refusal of the credentials, HTTP 401, or a provider-side error "
     "page)"),
    ("oxylabs_universal_scraper", {"url": "https://a.example/"},
     _oxy_reply('HTTP error during POST request: 429 - {"message": "Too many requests"}'),
     "HTTP 429"),
    ("oxylabs_universal_scraper", {"url": "https://a.example/"},
     _oxy_reply("HTTP error during POST request: 403 - Forbidden"), "HTTP 403"),
    ("oxylabs_amazon_product_scraper", {"query": "B0TEST1234"},
     _oxy_reply("HTTP error during POST request: 401 - "), "HTTP 401"),
    ("oxylabs_universal_scraper", {"url": "https://a.example/"},
     _oxy_reply("HTTP error during POST request: 407 - "), "HTTP 407"),
    # an error reply in JSON without a job record (an account refusal whose body has no job)
    ("oxylabs_universal_scraper", {"url": "https://a.example/"}, _oxy_reply("Error: 'job'"),
     "no job record in the error reply (Error: 'job')"),
])
def test_a_provider_account_error_is_a_harness_error(tool, args, reply, reason, monkeypatch):
    async def exec_http(container, method, path, timeout, body=None):
        return reply

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    sb = _sandbox()
    out = _call(sb, tool, args, 3)
    # the model still gets the server's reply, as upstream
    assert out == M.harness_tool_output(reply[0], reply[1], tool)[0]
    assert sb.harness_error_samples == {3: 1}
    assert sb.provider_account_errors == {f"{tool}: {reason}": 1}
    assert sb.provider_unreachable == {}


@pytest.mark.parametrize("tool,reply", [
    # Exa's 403 also covers content-safety refusals of the request, and the reply has no tag
    ("exa_web_fetch_exa", _exa_reply("web_fetch_exa error (403): Request rejected by content "
                                     "safety moderation")),
    ("exa_web_fetch_exa", _exa_reply("web_fetch_exa error (400): Invalid request body")),
    ("exa_web_fetch_exa", _exa_reply("Error fetching URL(s): https://u.example/: CRAWL_NOT_FOUND")),
    ("oxylabs_universal_scraper", _oxy_reply("HTTP error during POST request: 400 - bad value")),
    ("oxylabs_universal_scraper", _oxy_reply("HTTP error during POST request: 613 - faulted")),
    # a page that quotes the signatures is the page
    ("exa_web_fetch_exa", _oxy_reply("# Issue\nURL: https://g.example/\nUsers see "
                                     "'web_fetch_exa error (401): Invalid API key'")),
    ("oxylabs_universal_scraper", _oxy_reply("# Page\nError: Expecting value: line 1 column 1 "
                                             "(char 0)")),
    # another server's reply is never an Exa or Oxylabs account error
    ("twelvedata_GetPrice", _oxy_reply("Error: Expecting value: line 1 column 1 (char 0)")),
    ("github_list_commits", _exa_reply("web_fetch_exa error (401): Invalid API key",
                                       tool="github_list_commits")),
])
def test_replies_the_model_can_cause_stay_its_outcome(tool, reply, monkeypatch):
    async def exec_http(container, method, path, timeout, body=None):
        return reply

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    sb = _sandbox()
    _call(sb, tool, {"x": 1}, 4)
    assert sb.harness_error_samples == {} and sb.provider_account_errors == {}
    assert sb.provider_unreachable == {}


def test_the_exa_unreachable_signature_matches_only_the_environments_error_reply():
    page = M.harness_tool_output(*_oxy_reply(
        "# Issue\nURL: https://g.example/\nUsers report 'web_fetch_exa error: fetch failed'"),
        "exa_web_fetch_exa")[0]
    assert not M.provider_unreachable("exa_web_fetch_exa", page)
    assert M.provider_unreachable("exa_web_fetch_exa",
                                  _exa_reply("web_fetch_exa error: fetch failed")[1])
    # a detail that only contains it is not the server's own error
    assert not M.provider_unreachable(
        "exa_web_fetch_exa", _exa_reply("x web_fetch_exa error: fetch failed")[1])
    # the error names the call's own tool
    assert not M.provider_unreachable(
        "exa_web_fetch_exa", _exa_reply("web_search_exa error: fetch failed")[1])
    assert M.provider_unreachable(
        "exa_web_search_exa",
        _exa_reply("web_search_exa error: fetch failed", tool="exa_web_search_exa")[1])


def test_an_unreachable_provider_is_not_also_counted_as_an_account_error(monkeypatch):
    async def exec_http(container, method, path, timeout, body=None):
        return _oxy_reply("Request error during POST request: [Errno -3] Temporary failure in "
                          "name resolution")

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    sb = _sandbox()
    _call(sb, "oxylabs_universal_scraper", {"url": "https://a.example/"}, 1)
    assert sb.provider_unreachable == {"oxylabs_universal_scraper": 1}
    assert sb.provider_account_errors == {} and sb.harness_error_samples == {1: 1}


def test_the_rules_follow_the_pinned_servers_error_formats():
    assert M.PROVIDER_SERVERS == ("exa", "oxylabs")
    assert M.EXA_ACCOUNT_STATUSES == (401, 402, 429) and 403 not in M.EXA_ACCOUNT_STATUSES
    assert M.OXYLABS_ACCOUNT_STATUSES == (401, 403, 407, 429)
    assert 400 not in M.OXYLABS_ACCOUNT_STATUSES
    for tool in M.UPSTREAM_SEARCH_TOOLS:
        if M.tool_server(tool) in M.PROVIDER_SERVERS:
            assert M._provider_tool(tool)[0] == M.tool_server(tool)
    assert M._provider_tool("ddg-search_fetch_content") == ("", "")
    assert not M.provider_account_error("fetch_fetch", M.OXYLABS_NO_JSON_REPLY)


# ---- the pre-flight ---------------------------------------------------------
def _health(monkeypatch, oxy_reply, exa_reply=(200, '[{"type":"text","text":"Title: r"}]')):
    async def exec_http(container, method, path, timeout, body=None):
        return exa_reply if body["tool_name"].startswith("exa_") else oxy_reply

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "gemini_search_reply",
                        lambda *a, **k: (200, '[{"type":"text","text":"r"}]', []))
    return asyncio.run(M._run_health_calls("http://127.0.0.1:1", "cred", ("exa", "oxylabs")))


def test_the_preflight_fails_an_oxylabs_check_whose_provider_is_unreachable(monkeypatch):
    res = _health(monkeypatch, _oxy_reply("Request error during POST request: [Errno -3] "
                                          "Temporary failure in name resolution"))
    ok, detail = res["oxylabs"]
    assert not ok and detail.startswith("provider API unreachable: ")
    assert res["exa"][0]
    reason = M.preflight_failure_reason(res, local_env=True, enabled=list(M.RELEASE_SERVERS))
    assert "oxylabs" in reason and "provider API unreachable" in reason


def test_the_preflight_fails_an_oxylabs_check_whose_provider_refuses_the_account(monkeypatch):
    res = _health(monkeypatch, _oxy_reply('HTTP error during POST request: 401 - {"job": {}}'))
    ok, detail = res["oxylabs"]
    assert not ok and detail.startswith("the provider refused the account (HTTP 401): ")
    # the rejected-credentials reply already fails the release's own pass rule
    res = _health(monkeypatch, _OXY_401)
    assert res["oxylabs"][0] is False


def test_the_preflight_passes_working_exa_and_oxylabs_checks(monkeypatch):
    res = _health(monkeypatch, _oxy_reply("# Results\n1. python.org"))
    assert res["oxylabs"] == (True, res["oxylabs"][1]) and res["exa"][0]
    # exa's error replies are non-2xx, which the release's pass rule fails already
    res = _health(monkeypatch, _oxy_reply("# Results"), exa_reply=_exa_reply(
        "web_search_exa error (401): Invalid API key", tool="exa_web_search_exa"))
    assert res["exa"][0] is False


# ---- the run's record -------------------------------------------------------
def _run(monkeypatch, sb, traces):
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
        "status": "success", "accuracy": 0.0, "correct_answers": 0, "sample_traces": traces})
    return M.run_mcp_atlas("m", "http://x", concurrency=1)


def test_a_run_names_provider_account_errors(monkeypatch):
    sb = _sandbox()
    sb.provider_account_errors = {"exa_web_fetch_exa: HTTP 402": 2}
    sb.harness_error_samples = {1: 2}
    res = _run(monkeypatch, sb, [
        {"sample_idx": 0, "category": "exa", "claim_recall": 1.0, "is_correct": True,
         "claim_scores": [1.0], "judge_models": ["g"]},
        {"sample_idx": 1, "category": "exa", "judge_grade": M.HARNESS_ERROR_GRADE,
         "scoring_excluded": True, "is_correct": False}])
    assert res["status"] == "completed_with_errors" and res["accuracy"] == 100.0
    assert res["harness_errors"][0]["provider_account_error_calls"] == {
        "exa_web_fetch_exa: HTTP 402": 2}
    assert res["search_substitution"]["upstream_provider_account_errors"] == {
        "exa_web_fetch_exa: HTTP 402": 2}
    assert ("2 page tool call(s) whose provider refused the run's account (Exa or Oxylabs: "
            "rejected credentials, exhausted credit or plan, or the rate limit; "
            "exa_web_fetch_exa: HTTP 402 x2)") in res["leaderboard_comparable_reason"]


def test_a_run_whose_every_task_is_a_harness_error_withholds_its_score(monkeypatch):
    sb = _sandbox()
    sb.provider_account_errors = {"oxylabs_universal_scraper: HTTP 429": 2}
    sb.harness_error_samples = {0: 1, 1: 1}
    # marked by the fallback after the judge ran (the judge's own marking goes through base)
    res = _run(monkeypatch, sb, [
        {"sample_idx": 0, "category": "o", "claim_recall": 0.0, "is_correct": False,
         "judge_grade": "0/2 claims"},
        {"sample_idx": 1, "category": "o", "claim_recall": 0.0, "is_correct": False,
         "judge_grade": "0/2 claims"}])
    assert res["status"] == "failed"
    assert res["accuracy"] is None and "pass_rate_0.75" not in res
    assert res["accuracy_withheld_reason"] == (
        "nothing was scored: 2 task(s) had a harness error (HARNESS_ERROR) and no other task was "
        "scored")
    assert res["correct_answers"] == 0
    assert all(t["judge_grade"] == M.HARNESS_ERROR_GRADE for t in res["sample_traces"])


def test_the_mcp_atlas_doc_describes_the_account_errors():
    doc = _flat(_read("docs/evals/mcp_atlas.md"))
    assert doc.isascii()
    for needle in ("`upstream_provider_account_errors`", "`provider_account_error_calls`",
                   "`web_fetch_exa error (<status>): <Exa's error>`", "`NO_MORE_CREDITS`",
                   "`INVALID_API_KEY`", "`PROHIBITED_CONTENT`", "`CONTENT_FILTER_ERROR`",
                   "`Error: Expecting value: line 1 column 1 (char 0)`",
                   "`HTTP error during POST request: <status> - ...`", "`Error: 'job'`",
                   "`provider API unreachable`", "`accuracy_withheld_reason`"):
        assert needle in doc, needle
    assert "an account error the provider returns during the run" not in doc
    for status in M.OXYLABS_ACCOUNT_STATUSES:
        assert str(status) in doc


# =========================================================================== #
# mcp_bench: the live check of the server keys before the run
# =========================================================================== #
_ACCEPTED = {"developer.nps.gov": (200, '{"data": []}'), "api.nasa.gov": (200, "{}"),
             "huggingface.co": (200, '{"name": "u"}'),
             "clinicaltrialsapi.cancer.gov": (200, '{"data": []}'),
             "maps.googleapis.com": (200, '{"status": "OK", "results": [{}]}')}
_KEYS = {"NPS_API_KEY": "nps-value-0000001", "NASA_API_KEY": "nasa-value-000001",
         "HF_TOKEN": "hf_value000000001", "NCI_API_KEY": "nci-value-0000001",
         "GOOGLE_MAPS_API_KEY": "maps-value-000001"}


def _stub_services(monkeypatch, replies=None):
    """A fake of the five services: {host: (status, body) or an exception, or a list of them
    consumed in turn}. Records every request."""
    replies = dict(_ACCEPTED, **(replies or {}))
    seen = []

    def get(url, headers, timeout):
        host = re.match(r"https://([^/]+)/", url).group(1)
        seen.append((url, dict(headers)))
        reply = replies[host]
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(MB, "_key_check_get", get)
    monkeypatch.setattr(MB.time, "sleep", lambda s: None)
    return seen


def test_every_accepted_key_passes_and_is_sent_as_its_server_sends_it(monkeypatch):
    seen = _stub_services(monkeypatch)
    assert MB.server_key_problems(dict(_KEYS)) == {}
    by_host = {}
    for url, headers in seen:
        by_host.setdefault(re.match(r"https://([^/]+)/", url).group(1), []).append((url, headers))
    assert by_host["developer.nps.gov"] == [
        ("https://developer.nps.gov/api/v1/parks?limit=1", {"X-Api-Key": _KEYS["NPS_API_KEY"]})]
    assert by_host["api.nasa.gov"] == [
        (f"https://api.nasa.gov/neo/rest/v1/stats?api_key={_KEYS['NASA_API_KEY']}", {})]
    assert by_host["huggingface.co"][0][1] == {"Authorization": f"Bearer {_KEYS['HF_TOKEN']}"}
    assert by_host["clinicaltrialsapi.cancer.gov"][0][1] == {"x-api-key": _KEYS["NCI_API_KEY"]}
    maps = [u for u, _ in by_host["maps.googleapis.com"]]
    assert [u.split("/maps/api/", 1)[1].split("&key=")[0] for u in maps] == [
        q for _, q in MB.GOOGLE_MAPS_KEY_CHECKS]
    assert all(u.endswith("&key=" + _KEYS["GOOGLE_MAPS_API_KEY"]) for u in maps)


def test_a_refused_key_is_named_with_the_services_answer_never_its_value(monkeypatch):
    for k, v in _KEYS.items():
        monkeypatch.setenv(k, v)
    _stub_services(monkeypatch, {
        "developer.nps.gov": (403, '{"error": {"code": "API_KEY_INVALID", "message": "Your API '
                                   'key is not valid."}}'),
        "huggingface.co": (401, '{"error":"Invalid username or password."}'),
        "maps.googleapis.com": [(200, '{"status": "OK"}'), (200, '{"status": "OK"}'),
                                (200, '{"error_message": "This API project is not authorized '
                                      'to use this API.", "status": "REQUEST_DENIED"}'),
                                (200, '{"status": "ZERO_RESULTS"}')]})
    problems = MB.server_key_problems()
    assert set(problems) == {"NPS_API_KEY", "HF_TOKEN", "GOOGLE_MAPS_API_KEY"}
    assert problems["NPS_API_KEY"].startswith("refused: HTTP 403") and \
        "API_KEY_INVALID" in problems["NPS_API_KEY"]
    assert problems["HF_TOKEN"].startswith("refused: HTTP 401")
    assert problems["GOOGLE_MAPS_API_KEY"] == (
        "refused: Places API: HTTP 200 REQUEST_DENIED (This API project is not authorized to use "
        "this API.)")
    text = json.dumps(problems)
    assert not any(v in text for v in _KEYS.values())


def test_a_service_that_does_not_answer_is_retried_then_stops_the_suite(monkeypatch):
    seen = _stub_services(monkeypatch, {
        "api.nasa.gov": [OSError("timed out"), (503, "busy"), (200, "{}")],
        "clinicaltrialsapi.cancer.gov": [(502, "bad gateway")],
        "maps.googleapis.com": [(200, '{"status": "OVER_QUERY_LIMIT"}')]})
    problems = MB.server_key_problems(dict(_KEYS))
    # NASA answered on the third attempt; NCI and Maps never did
    assert set(problems) == {"NCI_API_KEY", "GOOGLE_MAPS_API_KEY"}
    assert problems["NCI_API_KEY"] == (
        f"no answer of the service's own after {MB.KEY_CHECK_ATTEMPTS} attempts: HTTP 502 bad "
        "gateway")
    assert "OVER_QUERY_LIMIT" in problems["GOOGLE_MAPS_API_KEY"]
    assert sum(1 for u, _ in seen if "api.nasa.gov" in u) == 3
    assert sum(1 for u, _ in seen if "cancer.gov" in u) == MB.KEY_CHECK_ATTEMPTS


def test_unset_keys_are_not_sent(monkeypatch):
    seen = _stub_services(monkeypatch)
    assert MB.server_key_problems({"NPS_API_KEY": " ", "HF_TOKEN": ""}) == {}
    assert seen == []
    assert MB.server_key_problems({"NASA_API_KEY": _KEYS["NASA_API_KEY"]}) == {}
    assert [u for u, _ in seen] == [
        f"https://api.nasa.gov/neo/rest/v1/stats?api_key={_KEYS['NASA_API_KEY']}"]


def _prereqs_pass_up_to_the_keys(monkeypatch):
    monkeypatch.setattr(MB.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(MB.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(MB, "_image_labels", lambda image: {
        MB.REQUEST_POLICY_LABEL: MB.REQUEST_POLICY_VERSION,
        MB.HARNESS_LABEL: MB.HARNESS_VERSION})
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "x" * 35)
    monkeypatch.setattr(MB, "gemini_key_live_valid", lambda key: (True, ""))


def test_the_prerequisites_stop_on_a_refused_key_and_name_the_unset_ones(monkeypatch):
    _prereqs_pass_up_to_the_keys(monkeypatch)
    for k, v in _KEYS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("NCI_API_KEY")
    _stub_services(monkeypatch, {"api.nasa.gov": (403, '{"error": {"code": "API_KEY_INVALID"}}')})
    ok, why = MB.check_mcp_bench_prerequisites()
    assert not ok
    assert why.startswith("the live check of the server keys failed before the run")
    assert "NASA_API_KEY (NASA Data): refused: HTTP 403" in why
    assert "Also unset: NCI_API_KEY (BioMCP)" in why
    assert MB.DOCS_URL in why and not any(v in why for v in _KEYS.values())


def test_the_prerequisites_pass_when_every_set_key_is_accepted(monkeypatch):
    _prereqs_pass_up_to_the_keys(monkeypatch)
    for k, v in _KEYS.items():
        monkeypatch.setenv(k, v)
    _stub_services(monkeypatch)
    assert MB.check_mcp_bench_prerequisites() == (True, "")


def test_a_refused_key_stops_the_run_before_any_container(monkeypatch):
    _prereqs_pass_up_to_the_keys(monkeypatch)
    for k, v in _KEYS.items():
        monkeypatch.setenv(k, v)
    _stub_services(monkeypatch, {"developer.nps.gov": (403, "{}")})
    ran = []
    monkeypatch.setattr(MB.subprocess, "run", lambda cmd, **kw: (
        ran.append(cmd), subprocess.CompletedProcess(cmd, 0, "", ""))[1])
    monkeypatch.setattr(MB.container_guard, "require_metadata_blocked",
                        lambda *a, **k: pytest.fail("the metadata check ran"))
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda *a, **k: pytest.fail("a container started"))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*NPS_API_KEY \(National Parks\): "
                                           r"refused: HTTP 403"):
        MB.run_mcp_bench("m", "http://x/v1")
    assert not any(c[:2] == ["docker", "run"] for c in ran)


def test_the_result_records_the_key_check(monkeypatch):
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    for k, v in _KEYS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(MB.container_guard, "require_metadata_blocked", lambda *a, **k: None)
    connected = {f"S{i}": {"status": "success", "tools": 3, "error": ""} for i in range(28)}
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: {
        "servers": sorted(connected), "unavailable": [], "unbuilt": [], "network_up": True,
        "connections": connected})
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: types.SimpleNamespace(
                            client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                                                "MCP_CREDENTIALED_TOKEN": "t0k"},
                            stop=lambda: None, record=lambda: {}))
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48},
        stop=lambda: None, record=lambda: {}))

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                json.dump({"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
                           "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
                           "tool_selection_score": 5.0,
                           "planning_effectiveness_and_efficiency_score": 5.0}, fh)
            with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                json.dump({"filtered": True, "unavailable_servers": [], "tasks_kept": 104}, fh)
            with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": [
                    {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": []}]}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    res = MB.run_mcp_bench("m", "http://x/v1")
    check = res["server_key_check"]
    assert set(check) == set(MB._REQUIRED_KEYS)
    assert check["NPS_API_KEY"] == "https://developer.nps.gov/api/v1/parks: accepted"
    assert check["GOOGLE_MAPS_API_KEY"].startswith("the Maps Platform APIs its server calls: "
                                                   "Geocoding API, Elevation API, Places API")
    assert not any(v in json.dumps(check) for v in _KEYS.values())


def test_the_mcp_bench_doc_lists_the_checks_the_code_sends():
    raw = _read("docs/evals/mcp_bench.md")
    doc = _flat(raw)
    assert raw.isascii()
    section = raw.split("### Checking the keys", 1)[1].split("\n## ", 1)[0]
    for key, (url, how) in MB.SERVER_KEY_CHECKS.items():
        row = next(line for line in section.splitlines() if line.startswith(f"| `{key}`"))
        base = url.split("?", 1)[0]
        assert base in row, key
        if how.startswith("query:"):
            assert f"{how[6:]}=<key>" in row, key
        elif how == "bearer":
            assert "Authorization: Bearer" in row, key
        else:
            assert f"`{how}`" in row, key
    maps_row = next(line for line in section.splitlines()
                    if line.startswith("| `GOOGLE_MAPS_API_KEY`"))
    for _, query in MB.GOOGLE_MAPS_KEY_CHECKS:
        assert f"`{query.split('?', 1)[0].rsplit('/', 1)[0]}`" in maps_row, query
    assert f"up to {MB.KEY_CHECK_ATTEMPTS} attempts" in doc
    assert "`server_key_check`" in doc
    assert "the pinned server calls the Hugging Face Hub API (`huggingface.co`) unauthenticated " \
           "and never sends it" in doc
    assert "Movie Recommender's its own TMDB" in doc
    # the manual snippet survives a network error and a reply that is not JSON
    assert "except (urllib.error.URLError, OSError) as e:" in section
    assert "except ValueError:" in section


# =========================================================================== #
# complexfuncbench: the documented commands pass the CLI's own checks
# =========================================================================== #
def test_the_documented_local_command_carries_num_gpus():
    from gbench import cli
    parser = cli.create_parser()
    run = _read("docs/evals/complexfuncbench.md").split("## Run", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", run, re.S)
    commands = [" ".join(line.rstrip("\\").strip() for line in b.splitlines()
                         if not line.startswith("export ")) for b in blocks]
    commands = [c for c in commands if c.strip().startswith("gbench ")]
    local = [c for c in commands if "--remote-endpoint" not in c]
    assert local and len(commands) == 2
    for command in commands:
        args = parser.parse_args(shlex.split(command)[1:])
        assert args.evals == ["complexfuncbench"] and args.max_output_tokens
        # cli._run refuses a local run without --num-gpus (it launches a vLLM server)
        assert args.remote_endpoint or (args.models and args.num_gpus and args.tool_call_parser)
