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

"""BFCL V4 agentic web_search served by Gemini Google Search grounding, and nothing else.

  * No SerpAPI code path, key probe, knob or message is left in the BFCL suites, their image or
    their docs; SerpAPI is named only as the official service the substitution replaces.
  * The Web Search setting runs whenever a Gemini key is set (GEMINI_API_KEYS, else
    GEMINI_API_KEY) and the Gemini API does not reject it, the judged suites' gate; without one
    it is infra_required naming GEMINI_API_KEY and the other categories still run.
  * The tool the model sees is BFCL's own: WebSearchAPI.search_engine_query's arguments and its
    result shape ({title, href, body}, or {title, href} without snippets, cut to max_results).
  * The result records the backend and the substitution, and is not leaderboard-comparable
    with the reason "web search served by Gemini Google Search grounding (BFCL queries
    DuckDuckGo via SerpAPI)".
  * The key reaches only the web_search `bfcl generate` container, through its environment;
    it never reaches the model (search results, tool results), the log, the result or argv.

Everything runs offline: the bfcl_eval modules the search is installed into are stand-ins with
upstream's shapes, the grounding endpoint is a stubbed urlopen, the harness CLI is stubbed. The
test marked `image` runs the real pinned harness in the local image (no network); set
GBENCH_TEST_BFCL_IMAGE=1 to run it."""

import inspect
import io
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base, bfcl, search_tool
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites import tau_common as TC_GUARD

REPO = Path(__file__).resolve().parents[1]
KEY = "gemKEY-bfcl-search-0123456789abcdef"
REASON = ("web search served by Gemini Google Search grounding (BFCL queries DuckDuckGo via "
          "SerpAPI)")
MEMORY = list(AG.MEMORY_CATEGORIES)
WEB = list(AG.WEB_SEARCH_CATEGORIES)

image = pytest.mark.skipif(not os.environ.get("GBENCH_TEST_BFCL_IMAGE"),
                           reason="set GBENCH_TEST_BFCL_IMAGE=1 to run the BFCL harness image")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GBENCH_BFCL_TIMEOUT_S", "BFCL_TIMEOUT_S",
                 "GBENCH_BFCL_V4_AGENTIC_TEMPERATURE", "GBENCH_BFCL_REUSE_GENERATIONS",
                 "BFCL_REUSE_GENERATIONS", AG.SEARCH_LOG_ENV, *AG.SEARCH_KNOB_NAMES):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# no SerpAPI path is left
# --------------------------------------------------------------------------- #

OWNED = ["gbench/runners/eval_suites/bfcl_v4_agentic.py", "gbench/runners/eval_suites/bfcl.py",
         "gbench/docker/bfcl.Dockerfile", "docs/evals/bfcl_v4_agentic.md", "docs/evals/bfcl.md"]


@pytest.mark.parametrize("rel", OWNED)
def test_serpapi_is_named_only_as_the_official_service(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    for token in ("SERPAPI_API_KEY", "serpapi.com", "serpapi_", "SerpApiClient",
                  "GBENCH_BFCL_SERPAPI", "SERPAPI_", "account.json", "searches left"):
        assert token not in text, (rel, token)
    for m in re.finditer(r"(?i)serp", text):
        assert re.search(r"(?i)via[\s-]+$", text[max(0, m.start() - 8):m.start()]), (
            rel, text[max(0, m.start() - 60):m.end() + 20])
    assert text.isascii()


def test_the_module_has_no_search_backend_but_gemini():
    for name in ("serpapi_account", "_observe_serpapi_calls", "serpapi_log_summary",
                 "_serpapi_watch", "Unconfirmed", "_same_error", "_mask_key",
                 "SERPAPI_ACCOUNT_URL", "SERPAPI_LOG_NAME", "_LAST_ACCOUNT"):
        assert not hasattr(AG, name), name
    assert AG.SEARCH_BACKEND == "gemini-google-search-grounding"
    assert AG.SEARCH_KEY_NAME == "GEMINI_API_KEY"
    assert AG.SEARCH_SUBSTITUTION_REASON == REASON
    assert not [n for n in AG._CHILD_ENV_NAMES if "SERP" in n]
    assert {"GEMINI_API_KEY", "GEMINI_API_KEYS", AG.SEARCH_LOG_ENV} <= set(AG._CHILD_ENV_NAMES)
    assert set(AG.SEARCH_KNOB_NAMES) <= set(AG._CHILD_ENV_NAMES)


# --------------------------------------------------------------------------- #
# the gate: a Gemini key the Gemini API does not reject
# --------------------------------------------------------------------------- #

@pytest.fixture
def live(monkeypatch):
    """The live key check, stubbed: `state["answer"]` is what it returns, `state["keys"]` what
    it was asked about."""
    state = {"answer": (True, ""), "keys": []}
    monkeypatch.setattr(AG, "gemini_key_live_valid",
                        lambda key: state["keys"].append(key) or state["answer"])
    return state


def test_no_key_no_search_whatever_else_is_set(live, monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key-0123456789")
    backend, why = AG.search_backend()
    assert backend is None and why.startswith("GEMINI_API_KEY (or GEMINI_API_KEYS) is unset")
    assert live["keys"] == []


@pytest.mark.parametrize("env,checked", [
    ({"GEMINI_API_KEY": KEY}, KEY),
    ({"GEMINI_API_KEYS": "gemA-0123456789;gemB-0123456789"}, "gemA-0123456789,gemB-0123456789"),
    ({"GEMINI_API_KEYS": "gemA-0123456789", "GEMINI_API_KEY": KEY}, "gemA-0123456789"),
])
def test_the_key_is_read_as_the_judged_suites_read_it(live, monkeypatch, env, checked):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert AG.search_backend() == (AG.SEARCH_BACKEND, "")
    assert live["keys"] == [checked]


def test_a_rejected_key_blocks_the_setting_without_echoing_it(live, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    live["answer"] = (False, f"HTTP 400 API key not valid {KEY}")
    backend, why = AG.search_backend()
    assert backend is None and "rejected the key" in why and "HTTP 400" in why
    assert KEY not in why


def test_the_bfcl_suite_names_the_gemini_key(live, monkeypatch):
    monkeypatch.setattr(AG, "check_bfcl_prerequisites", lambda url: (True, "", list(
        AG.AGENTIC_CATEGORIES)))
    why = bfcl._check_agentic("http://x/v1", WEB)
    assert "GEMINI_API_KEY (or GEMINI_API_KEYS) is unset" in why
    assert "Set GEMINI_API_KEY (or GEMINI_API_KEYS)" in why
    assert bfcl._check_agentic("http://x/v1", MEMORY) is None


# --------------------------------------------------------------------------- #
# a run: the substitution recorded, the key only where it searches
# --------------------------------------------------------------------------- #

@pytest.fixture
def harness(monkeypatch, tmp_path, live):
    """run_bfcl_v4_agentic with the harness CLI stubbed; `state["log"]` rows land in the web
    search log during the generate that searches."""
    root = tmp_path / "bfcl"
    state = {"cli": [], "log": [], "live": live}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites", lambda url: (True, "", list(
        AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "_register_model", lambda m: m)

    def cli(args, env, timeout, model="", backend=None):
        state["cli"].append({"args": list(args), "env": dict(env), "backend": backend})
        if args[0] == "generate" and AG.SEARCH_LOG_ENV in env:
            with open(env[AG.SEARCH_LOG_ENV], "a", encoding="utf-8") as f:
                for row in state["log"]:
                    f.write(json.dumps(row) + "\n")
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda r, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    state["root"] = root
    return state


def test_a_run_with_a_key_records_the_substitution_and_is_not_comparable(harness, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash,gemini-3.6-flash")
    harness["log"] = [{"backend": "gemini:gemini-3.7-flash", "results": 4, "error": None},
                      {"backend": "gemini:gemini-3.6-flash", "results": 2, "error": None}]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert res["status"] == "success" and res["settings"]["web_search"]["status"] == "success"
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == REASON + "; temperature 0.0 set by default (BFCL generates at 0.001)"
    report = res["bfcl_report"]
    assert report["search_backend"].startswith(
        "gemini-google-search-grounding (BFCL's web_search tool served by Gemini Google Search "
        "grounding, model cascade gemini-3.7-flash, gemini-3.6-flash;")
    sub = report["search_substitution"]
    assert sub["models"] == ["gemini-3.7-flash", "gemini-3.6-flash"]
    assert sub["leaderboard_comparable"] is False and sub["leaderboard_comparable_reason"] == REASON
    assert "DuckDuckGo via SerpAPI" in sub["official"] and "google_search" in sub["served_by"]
    assert "region is not applied" in sub["differences"]
    assert report["web_search_calls"]["by_backend"] == {"gemini:gemini-3.7-flash": 1,
                                                        "gemini:gemini-3.6-flash": 1}
    assert KEY not in json.dumps(res)


def test_only_the_generate_that_searches_gets_the_key_and_the_knobs(harness, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", f"{KEY};gemSECOND-0123456789")
    monkeypatch.setenv("GBENCH_SEARCH_MAX_RPS", "2")
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    gen, ev = harness["cli"]
    assert gen["args"][0] == "generate" and ev["args"][0] == "evaluate"
    assert gen["env"]["GEMINI_API_KEYS"] == f"{KEY},gemSECOND-0123456789"
    assert gen["env"]["GEMINI_API_KEY"] == KEY and gen["env"]["GBENCH_SEARCH_MAX_RPS"] == "2"
    assert gen["env"][AG.SEARCH_LOG_ENV].endswith(AG.SEARCH_LOG_NAME)
    assert gen["backend"] == AG.SEARCH_BACKEND
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", AG.SEARCH_LOG_ENV):
        assert name not in ev["env"], name


def test_without_a_key_web_search_is_infra_required_and_memory_runs(harness):
    # The suite's own run hard-errors (its score weighs the Web Search setting); bfcl's agentic
    # part reports the setting as infra_required and runs Memory.
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert harness["cli"] == []
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    [gen, ev] = harness["cli"]
    assert gen["args"][gen["args"].index("--test-category") + 1] == ",".join(MEMORY)
    web = res["settings"]["web_search"]
    assert web["status"] == "infra_required" and web["prerequisite"] == "GEMINI_API_KEY"
    assert "GEMINI_API_KEY" in web["error"] and web["when"] == "before generation"
    assert set(res["category_accuracy"]) == set(MEMORY)
    assert res["bfcl_report"]["search_substitution"] is None
    assert REASON not in res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False and res["status"] == "completed_with_errors"


def test_a_failed_search_is_infra_required_not_scored(harness, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    harness["log"] = [{"backend": "all-grounding-models-failed", "results": 0,
                       "error": f"search backend failed: HTTP Error 429 [{base.REDACTED}]"}]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    web = res["settings"]["web_search"]
    assert web["status"] == "infra_required" and web["when"] == "during generation"
    assert "for 1 of the 1 web searches" in web["infra_required"]
    assert "GBENCH_SEARCH_MAX_RPS" in web["infra_required"]
    assert set(res["category_accuracy"]) == set(MEMORY)
    # the Web Search setting was not scored: its infra_required entry is the reason, not the
    # substitution
    assert REASON not in res["leaderboard_comparable_reason"]


def test_no_credential_reaches_the_command_line(monkeypatch, tmp_path):
    seen = {}

    def launch(cmd, env, timeout, name, stream=False):
        seen.update(cmd=cmd, env=env)
        return 0, ""
    monkeypatch.setattr(AG, "_launch", launch)
    # `bfcl generate` runs on the bridge network after the network checks, docker calls of their
    # own (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    # the Gemini key stays in this process: gbench's Gemini relay (here on 127.0.0.1, its docker
    # and metadata checks stubbed: tests/test_bfcl_execution_guard_and_relays.py) adds it
    monkeypatch.setattr(AG, "_metadata_gate", lambda *a, **k: None)
    monkeypatch.setattr(AG, "_bridge_gateway", lambda *a, **k: "127.0.0.1")
    monkeypatch.setattr(AG, "_require_relay_reachable", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEYS", f"{KEY},gemSECOND-0123456789")
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "GEMINI_API_KEY": KEY,
           "GEMINI_API_KEYS": f"{KEY},gemSECOND-0123456789", AG.SEARCH_LOG_ENV: "/x/log",
           "GBENCH_SEARCH_MODELS": "gemini-3.7-flash"}
    AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    cmd = seen["cmd"]
    assert KEY not in " ".join(cmd) and "gemSECOND" not in " ".join(cmd)
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", AG.SEARCH_LOG_ENV, "GBENCH_SEARCH_MODELS",
                 AG.GEMINI_RELAY_URL_ENV):
        assert ["-e", name] == cmd[cmd.index(name) - 1:cmd.index(name) + 1]
    # the container holds the relay's placeholder, never the key
    assert KEY not in json.dumps(seen["env"]) and "gemSECOND" not in json.dumps(seen["env"])
    assert seen["env"]["GEMINI_API_KEY"].startswith("gbench-gemini-relay-")


# --------------------------------------------------------------------------- #
# the tool the model sees: BFCL's own contract
# --------------------------------------------------------------------------- #

def _fake_bfcl(monkeypatch):
    """Stand-ins for the two bfcl_eval modules the search is installed into, with upstream's
    shapes: WebSearchAPI (show_snippet, search_engine_query, fetch_url_content) and
    base_handler.execute_multi_turn_func_call."""
    class WebSearchAPI:
        def __init__(self):
            self.show_snippet = True

        def _load_scenario(self, initial_config, long_context=False):
            self.show_snippet = initial_config["show_snippet"]

        def search_engine_query(self, keywords, max_results=10, region="wt-wt"):
            """upstream docstring"""
            raise AssertionError("the harness' own search must not run")

        def fetch_url_content(self, url, mode="raw"):
            return {"content": url}
    ws = types.ModuleType("bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search")
    ws.WebSearchAPI = WebSearchAPI
    bh = types.ModuleType("bfcl_eval.model_handler.base_handler")

    def execute(func_call_list, initial_config, involved_classes, model_name, test_entry_id,
                long_context=False, is_evaL_run=False):
        # upstream: a list result is handed to the model as str(), a dict as json.dumps()
        inst = WebSearchAPI()
        inst._load_scenario(initial_config.get("WebSearchAPI", {"show_snippet": True}))
        out = []
        for call in func_call_list:
            try:
                value = eval("inst." + call, {"inst": inst})
                out.append(json.dumps(value) if isinstance(value, dict) else str(value))
            except Exception as e:
                out.append(f"Error during execution: {e}")
        return out, {"WebSearchAPI": inst}
    bh.execute_multi_turn_func_call = execute
    fsc = types.ModuleType("bfcl_eval.eval_checker.multi_turn_eval.func_source_code")
    fsc.web_search = ws
    mh = types.ModuleType("bfcl_eval.model_handler")
    mh.base_handler = bh
    for name, mod in (("bfcl_eval.eval_checker.multi_turn_eval.func_source_code", fsc),
                      (ws.__name__, ws), ("bfcl_eval.model_handler", mh), (bh.__name__, bh)):
        monkeypatch.setitem(sys.modules, name, mod)
    return WebSearchAPI, bh


@pytest.fixture
def tool(monkeypatch, tmp_path):
    """The search installed over the stand-ins, search_with_fallback stubbed: `state["reply"]`
    is what it returns (or raises), `state["calls"]` what it was asked."""
    WebSearchAPI, bh = _fake_bfcl(monkeypatch)
    state = {"calls": [], "reply": None, "cls": WebSearchAPI, "bh": bh,
             "log": tmp_path / AG.SEARCH_LOG_NAME}

    def search(query, max_results=8):
        state["calls"].append((query, max_results))
        if isinstance(state["reply"], Exception):
            raise state["reply"]
        return state["reply"]
    monkeypatch.setattr(search_tool, "search_with_fallback", search)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    AG._serve_web_search_with_gemini(str(state["log"]))
    return state


def _rows(n, backend="gemini:gemini-3.7-flash"):
    return [{"title": f"t{i}", "url": f"https://r/{i}", "snippet": f"s{i}"} for i in range(n)], \
        backend


def _log(state):
    return [json.loads(line) for line in state["log"].read_text().splitlines()]


def test_the_method_keeps_upstreams_shape(tool):
    inst = tool["cls"]()
    assert inspect.ismethod(inst.search_engine_query)
    names = [n for n, _ in inspect.getmembers(inst, inspect.ismethod) if not n.startswith("_")]
    assert names == ["fetch_url_content", "search_engine_query"]
    params = inspect.signature(inst.search_engine_query).parameters
    assert list(params) == ["keywords", "max_results", "region"]
    assert (params["max_results"].default, params["region"].default) == (10, "wt-wt")
    assert tool["cls"].search_engine_query.__doc__ == "upstream docstring"


def test_results_have_bfcls_shape_and_are_cut_to_max_results(tool):
    tool["reply"] = _rows(12)
    got = tool["cls"]().search_engine_query("capital of France", region="fr-fr")
    assert len(got) == 10 and got[0] == {"title": "t0", "href": "https://r/0", "body": "s0"}
    assert [list(r) for r in got] == [["title", "href", "body"]] * 10
    assert tool["calls"] == [("capital of France", None)]        # region is not applied
    assert tool["cls"]().search_engine_query("q", max_results=3) == got[:3]
    assert len(tool["cls"]().search_engine_query("q", max_results=None)) == 12
    assert [r["results"] for r in _log(tool)] == [10, 3, 12]


def test_without_snippets_there_is_no_body(tool):
    tool["reply"] = _rows(2)
    res, _ = tool["bh"].execute_multi_turn_func_call(
        ["search_engine_query(keywords='tea')"], {"WebSearchAPI": {"show_snippet": False}},
        ["WebSearchAPI"], "m", "web_search_no_snippet_0")
    assert res == [str([{"title": "t0", "href": "https://r/0"},
                        {"title": "t1", "href": "https://r/1"}])]


def test_an_answer_with_no_source_and_an_empty_answer(tool):
    tool["reply"] = ([{"title": "no results", "url": "", "snippet": "The search returned nothing."}],
                     "gemini:gemini-3.7-flash")
    assert tool["cls"]().search_engine_query("zzz") == []
    tool["reply"] = ([{"title": "grounded-answer", "url": "", "snippet": "Paris."}],
                     "gemini:gemini-3.7-flash")
    assert tool["cls"]().search_engine_query("q") == [
        {"title": "grounded-answer", "href": "", "body": "Paris."}]
    assert [(r["results"], r["error"]) for r in _log(tool)] == [(0, None), (1, None)]


def test_a_max_results_the_slice_cannot_take_spends_no_search(tool):
    tool["reply"] = _rows(3)
    res, _ = tool["bh"].execute_multi_turn_func_call(
        ["search_engine_query(keywords='tea', max_results='x')"], {}, ["WebSearchAPI"], "m", "w")
    assert res[0].startswith("Error during execution: slice indices must be integers")
    assert tool["calls"] == [] and not tool["log"].exists()


@pytest.mark.parametrize("reply,backend,needle", [
    (([{"title": "error", "url": "", "snippet": f"search backend failed: key={KEY}"}],
      "all-grounding-models-failed"), "all-grounding-models-failed", "search backend failed"),
    (([{"title": "error", "url": "", "snippet": "web search requires GEMINI_API_KEY"}], "none"),
     "none", "requires GEMINI_API_KEY"),
    (RuntimeError(f"boom {KEY}"), "error", "RuntimeError: boom"),
])
def test_a_search_no_model_served_gets_bfcls_error_text(tool, reply, backend, needle):
    tool["reply"] = reply
    res, _ = tool["bh"].execute_multi_turn_func_call(
        ["search_engine_query(keywords='tea')"], {}, ["WebSearchAPI"], "m", "w")
    assert res == [json.dumps({"error": AG.SEARCH_FAILED_TEXT})]
    [row] = _log(tool)
    assert row["backend"] == backend and needle in row["error"] and row["results"] == 0
    assert KEY not in tool["log"].read_text()
    assert AG.search_log_summary([row])["failed"] == 1


def test_the_key_never_reaches_the_model(tool):
    tool["reply"] = ([{"title": f"t {KEY}", "url": f"https://r/?k={KEY}", "snippet": KEY}],
                     "gemini:gemini-3.7-flash")
    got = tool["cls"]().search_engine_query("q")
    assert KEY not in json.dumps(got) and base.REDACTED in got[0]["body"]
    # every tool result the harness hands the model is masked, whatever tool produced it
    res, _ = tool["bh"].execute_multi_turn_func_call(
        [f"fetch_url_content(url='https://x/{KEY}')"], {}, ["WebSearchAPI"], "m", "w")
    assert KEY not in res[0] and base.REDACTED in res[0]


def test_the_real_grounding_request_is_turned_into_bfcls_shape(monkeypatch, tmp_path):
    """search_tool's own request and parsing (urlopen stubbed): generateContent with the
    google_search tool on the cascade's first model; chunks become {title, href, body}."""
    WebSearchAPI, _bh = _fake_bfcl(monkeypatch)
    sent = []
    reply = {"candidates": [{
        "content": {"parts": [{"text": "Paris is the capital of France."}]},
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": "https://vertexaisearch.cloud.google.com/r/1",
                                         "title": "wikipedia.org"}},
                                {"web": {"uri": "https://vertexaisearch.cloud.google.com/r/2",
                                         "domain": "britannica.com"}}],
            "groundingSupports": [{"segment": {"text": "Paris is the capital of France."},
                                   "groundingChunkIndices": [0]}]}}]}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        sent.append((req.full_url, json.loads(req.data)))
        return _Resp(json.dumps(reply).encode())
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(search_tool, "_RATE_LIMITER", search_tool._RateLimiter(0, 1))
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash,gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    log = tmp_path / "log.jsonl"
    AG._serve_web_search_with_gemini(str(log))
    got = WebSearchAPI().search_engine_query("capital of France")
    [(url, payload)] = sent
    assert url.startswith("https://generativelanguage.googleapis.com/v1beta/models/"
                          "gemini-3.7-flash:generateContent")
    assert payload == {"contents": [{"parts": [{"text": "capital of France"}]}],
                       "tools": [{"google_search": {}}]}
    assert got == [{"title": "wikipedia.org", "href": "https://vertexaisearch.cloud.google.com/r/1",
                    "body": "Paris is the capital of France."},
                   {"title": "britannica.com", "href": "https://vertexaisearch.cloud.google.com/r/2",
                    "body": "Paris is the capital of France."}]
    assert json.loads(log.read_text()) == {"backend": "gemini:gemini-3.7-flash", "results": 2,
                                           "error": None}


# --------------------------------------------------------------------------- #
# docs and the image
# --------------------------------------------------------------------------- #

def test_the_docs_describe_the_gemini_search_as_the_code_does():
    doc = (REPO / "docs" / "evals" / "bfcl_v4_agentic.md").read_text(encoding="utf-8")
    flat = " ".join(doc.split())
    for needle in (REASON, AG.SEARCH_FAILED_TEXT, AG.SEARCH_LOG_NAME, AG.SEARCH_LOG_ENV,
                   "prerequisite: GEMINI_API_KEY", "search_substitution", "web_search_calls",
                   "GEMINI_API_KEYS", "`gemini-google-search-grounding`", "empty_results"):
        assert needle in flat, needle
    for knob in AG.SEARCH_KNOB_NAMES:
        assert f"`{knob}`" in doc, knob
    for model in search_tool._DEFAULT_SEARCH_CASCADE:
        assert f"`{model}`" in doc, model
    bfcl_doc = " ".join((REPO / "docs" / "evals" / "bfcl.md").read_text(encoding="utf-8").split())
    assert REASON in bfcl_doc and "prerequisite: GEMINI_API_KEY" in bfcl_doc
    dockerfile = (REPO / "gbench" / "docker" / "bfcl.Dockerfile").read_text(encoding="utf-8")
    assert "Gemini Google Search grounding" in dockerfile


@image
def test_image_the_pinned_harness_runs_the_gemini_search(tmp_path):
    """The pinned release's own executor over its own WebSearchAPI, with the Gemini search
    installed (search_with_fallback stubbed, the container has no network): the method is mapped,
    the model gets str() of BFCL's result list, a failure gets BFCL's error text, and no key
    leaves the container."""
    program = (AG._package_preamble() + "\n".join([
        "import json, os",
        "from gbench.runners.eval_suites import search_tool",
        "KEY = os.environ['GEMINI_API_KEY']",
        "def fake(query, max_results=8):",
        "    if query == 'fail':",
        "        return [{'title': 'error', 'url': '', 'snippet': 'down ' + KEY}], "
        "'all-grounding-models-failed'",
        "    return [{'title': 't%d' % i, 'url': 'https://r/%d' % i, 'snippet': 's%d ' % i + KEY}"
        " for i in range(3)], 'gemini:gemini-3.7-flash'",
        "search_tool.search_with_fallback = fake",
        "from gbench.runners.eval_suites.bfcl_v4_agentic import _serve_web_search_with_gemini",
        "_serve_web_search_with_gemini(os.environ['GBENCH_BFCL_SEARCH_LOG'])",
        "from bfcl_eval.model_handler import base_handler",
        "res, _ = base_handler.execute_multi_turn_func_call(["
        "\"search_engine_query(keywords='tea', max_results=2)\", "
        "\"search_engine_query(keywords='fail')\"], {'WebSearchAPI': {'show_snippet': False}}, "
        "['WebSearchAPI'], 'm', 'web_search_no_snippet_0')",
        "print('RESULT', json.dumps(res))",
    ]) + "\n")
    root = str(tmp_path)
    log = tmp_path / "search.jsonl"
    env = dict(os.environ, GEMINI_API_KEY=KEY, BFCL_PROJECT_ROOT=root,
               GBENCH_BFCL_SEARCH_LOG=str(log))
    name = AG._container_name("gemini-search")
    cmd = AG._docker_run_command(name, ["GEMINI_API_KEY", "BFCL_PROJECT_ROOT",
                                        "GBENCH_BFCL_SEARCH_LOG"], [(root, root, "rw")], program)
    cmd[cmd.index("--network") + 1] = "none"
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(proc.stdout.split("RESULT", 1)[1])
    assert res == [str([{"title": "t0", "href": "https://r/0"},
                        {"title": "t1", "href": "https://r/1"}]),
                   json.dumps({"error": AG.SEARCH_FAILED_TEXT})]
    assert KEY not in proc.stdout + proc.stderr + log.read_text()
    summary = AG.search_log_summary(AG._read_search_log(log))
    assert (summary["requests"], summary["failed"]) == (2, 1)
