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

"""Per-setting prerequisites in BFCL, and ACEBench's context window as its pinned vLLM applies it.

  * BFCL V4's Web Search setting is served by Gemini Google Search grounding. Without a Gemini
    key, with one the Gemini API rejects, or when a search no grounding model served reached the
    model, that setting alone is reported infra_required in its own entry (never scored, never
    dropped, Overall Acc and the Agentic score null, status not success); single-turn,
    multi-turn and memory still run. A generate stopped for the web search is followed by the
    harness' own resume of the memory categories. Only a run that holds nothing but web_search
    hard-errors. A run that scores it is not leaderboard-comparable (the search substitution).
  * The harness launcher keeps output that may still be the start of a credential (a shorter
    credential that begins a longer one included) unmasked until it is final, and never cuts
    inside a whole credential, so what it streams equals one-shot masking.
  * ACEBench's LLMInfer serves the model with max_model_len=8192 on vllm==0.6.1.post1: a prompt
    longer than 8192 tokens is ignored by the scheduler and gets an empty reply (no request is
    sent; a row is scored on "", a conversation goes on with it), and a prompt of P <= 8192
    tokens gets at most 8193 - P tokens. Prompts are counted as that vLLM counts them.

Everything runs offline: the harness CLI, the model endpoint, the Gemini key check and the user
simulator are stubbed; the /tokenize test uses a local server on a free port."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import acebench as AB
from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"
KEY = "gemKEY-settings-0123456789abcdef"
MEMORY = list(AG.MEMORY_CATEGORIES)
WEB = list(AG.WEB_SEARCH_CATEGORIES)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("GBENCH_BFCL_TIMEOUT_S", "BFCL_TIMEOUT_S", "GBENCH_SHARD",
                 "GBENCH_BFCL_V4_AGENTIC_TEMPERATURE", "GBENCH_BFCL_TEMPERATURE",
                 "GBENCH_BFCL_REUSE_GENERATIONS", "BFCL_REUSE_GENERATIONS"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# bfcl_v4_agentic: the Web Search setting on its own
# --------------------------------------------------------------------------- #

@pytest.fixture
def harness(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic with the harness stubbed. `state["backend"]` is what the key check
    answers; `state["stop"]` makes the generate that searches raise that exception (the log
    watch's SearchUnavailable, or a wall-clock TimeoutExpired); `state["log"]` rows land in the
    web search log during that generate; every generate writes a result line per memory
    category it runs, and the searching one a cut-off line as a stopped container leaves."""
    root = tmp_path / "bfcl"
    state = {"cli": [], "backend": (AG.SEARCH_BACKEND, ""), "stop": None, "log": [], "counts": {},
             "resume_stop": False}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(AG, "check_bfcl_prerequisites", lambda url: (True, "", list(
        AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: state["backend"])
    monkeypatch.setattr(AG, "_register_model", lambda m: m)

    def cli(args, env, timeout, model="", backend=None):
        ids_file = root / "test_case_ids_to_generate.json"
        state["cli"].append({"args": list(args), "env": dict(env), "timeout": timeout,
                             "ids": (json.loads(ids_file.read_text()) if ids_file.exists()
                                     else None)})
        if args[0] != "generate":
            return 0, ""
        cats = args[args.index("--test-category") + 1].split(",")
        rdir = args[args.index("--result-dir") + 1] if "--result-dir" in args else "result"
        for cat in cats:
            if cat in MEMORY:
                path = root / rdir / "m" / "agentic" / f"BFCL_v4_{cat}_result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": f"{cat}_{len(state['cli'])}", "result": "x"}) + "\n")
        searching = AG.SEARCH_LOG_ENV in env
        if searching:
            with open(env[AG.SEARCH_LOG_ENV], "a", encoding="utf-8") as f:
                for row in state["log"]:
                    f.write(json.dumps(row) + "\n")
            cut = root / rdir / "m" / "agentic" / f"BFCL_v4_{MEMORY[0]}_result.json"
            if state["stop"] is not None and cut.exists():
                with cut.open("a", encoding="utf-8") as f:
                    f.write('{"id": "memory_kv_cut", "resu')
            if state["stop"] is not None:
                raise state["stop"]
        elif state["resume_stop"]:
            raise subprocess.TimeoutExpired(cmd="bfcl generate", timeout=timeout)
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    def run_in_image(name, module, func, kwargs, **k):
        if func == "scored_entry_counts":
            return {c: 4 for c in kwargs["categories"]}
        assert func == "_select_limited_ids"
        return {c: [f"{c}_{i}" for i in range(2)] for c in kwargs["categories"]}
    monkeypatch.setattr(AG, "run_in_image", run_in_image)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda root_, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    state["root"] = root
    return state


def _generates(state):
    return [c for c in state["cli"] if c["args"][0] == "generate"]


def _categories(call):
    args = call["args"]
    return args[args.index("--test-category") + 1].split(",")


def test_a_missing_key_leaves_web_search_infra_required_and_memory_runs(harness):
    harness["backend"] = (None, "GEMINI_API_KEY (or GEMINI_API_KEYS) is unset: BFCL's web_search "
                                "tool is served by Gemini Google Search grounding")
    # The suite's own run: its score weighs the Web Search setting, so a missing key is a hard
    # error before anything is generated.
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*bfcl_v4_agentic.*eval-categories"):
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert _generates(harness) == []
    # bfcl's agentic part (the same harness, run for bfcl) reports the setting and runs Memory.
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    [gen] = _generates(harness)
    assert _categories(gen) == MEMORY
    assert "GEMINI_API_KEY" not in gen["env"] and AG.SEARCH_LOG_ENV not in gen["env"]
    web = res["settings"]["web_search"]
    assert web["status"] == "infra_required" and web["when"] == "before generation"
    assert web["prerequisite"] == "GEMINI_API_KEY" and web["categories"] == WEB
    assert web["error"].startswith("[INFRA REQUIRED] bfcl")
    assert (web["total_questions"], web["accuracy"]) == (0, None)
    assert res["settings_infra_required"] == ["web_search"]
    assert res["settings"]["memory"]["status"] == "success"
    assert set(res["category_accuracy"]) == set(MEMORY)
    assert res["summary_web_search"] is None and res["overall_accuracy_agentic"] is None
    assert res["accuracy"] == res["summary_memory"] == 50.0
    assert "summary_memory" in res["metric"] and "null" in res["metric"]
    assert res["status"] == "completed_with_errors" and res["leaderboard_comparable"] is False
    assert "Web Search setting" in res["leaderboard_comparable_reason"]
    assert res["bfcl_report"]["search_backend"].startswith(
        "none: GEMINI_API_KEY (or GEMINI_API_KEYS) is unset")
    assert res["bfcl_report"]["search_substitution"] is None
    assert KEY not in json.dumps(res)


def test_web_search_alone_without_its_key_is_a_hard_error(harness):
    harness["backend"] = (None, "the Gemini API rejected the key (GEMINI_API_KEYS / "
                                "GEMINI_API_KEY: HTTP 401)")
    with pytest.raises(AG.SearchUnavailable) as e:
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2, eval_categories=",".join(WEB))
    assert str(e.value).startswith("[INFRA REQUIRED]") and e.value.when == "before generation"
    assert "rejected" in e.value.reason and harness["cli"] == []


def test_a_search_that_fails_during_generate_resumes_memory_and_scores_it(harness):
    harness["stop"] = AG.SearchUnavailable("Gemini Google Search grounding served none of the "
                                           "models of its cascade for 3 of the 9 web searches")
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    first, resume = _generates(harness)
    assert _categories(first) == list(AG.AGENTIC_CATEGORIES)
    assert first["env"]["GEMINI_API_KEY"] == KEY and first["env"]["GEMINI_API_KEYS"] == KEY
    # the harness' resume: memory only, no --allow-overwrite, no key
    assert _categories(resume) == MEMORY and "--allow-overwrite" not in resume["args"]
    assert "GEMINI_API_KEY" not in resume["env"] and "GEMINI_API_KEYS" not in resume["env"]
    assert AG.SEARCH_LOG_ENV not in resume["env"]
    # the line the stop cut off was dropped before the resume read the result file
    kv = harness["root"] / "result" / "m" / "agentic" / f"BFCL_v4_{MEMORY[0]}_result.json"
    assert all(json.loads(line) for line in kv.read_text().splitlines())
    [ev] = [c for c in harness["cli"] if c["args"][0] == "evaluate"]
    assert _categories(ev) == MEMORY and "GEMINI_API_KEY" not in ev["env"]
    web = res["settings"]["web_search"]
    assert web["when"] == "during generation" and "3 of the 9" in web["infra_required"]
    assert res["bfcl_report"]["memory_resumed"] is True and res["harness_errors"] == []
    assert set(res["category_accuracy"]) == set(MEMORY)
    # the suite's own run: the Memory numbers stay in settings, the suite's score is withheld
    # (a harness failure: the search service, not the model)
    assert res["status"] == "error" and res["overall_accuracy_agentic"] is None
    assert res["accuracy"] is None and res["failure_kind"] == "infra_required"
    assert "Web Search setting" in res["accuracy_withheld_reason"]
    assert res["settings"]["memory"]["summary"] is not None


def test_a_limited_run_resumes_with_an_id_file_of_the_memory_ids(harness):
    harness["stop"] = AG.SearchUnavailable("a search was refused")
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2, limit=10)
    first, resume = _generates(harness)
    assert set(first["ids"]) == set(AG.AGENTIC_CATEGORIES)
    assert set(resume["ids"]) == set(MEMORY) and resume["ids"] == {c: first["ids"][c]
                                                                   for c in MEMORY}
    assert "--run-ids" in resume["args"] and "--allow-overwrite" not in resume["args"]
    assert res["settings_infra_required"] == ["web_search"]
    assert res["leaderboard_comparable"] is False


def test_a_failure_found_after_generate_needs_no_resume(harness):
    harness["log"] = [{"backend": "gemini:gemini-3.7-flash", "results": 5, "error": None},
                      {"backend": "all-grounding-models-failed", "results": 0,
                       "error": "search backend failed: HTTP Error 429: Too Many Requests"}]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert len(_generates(harness)) == 1 and res["bfcl_report"]["memory_resumed"] is False
    web = res["settings"]["web_search"]
    assert "for 1 of the 2 web searches" in web["infra_required"]
    assert set(res["category_accuracy"]) == set(MEMORY)


def test_a_wall_clock_stop_caused_by_the_search_resumes_memory(harness, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    harness["log"] = [{"backend": "all-grounding-models-failed", "results": 0,
                       "error": "all grounding models failed"}]
    harness["stop"] = subprocess.TimeoutExpired(cmd="bfcl generate", timeout=600)
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    web = res["settings"]["web_search"]
    assert "GBENCH_BFCL_TIMEOUT_S=600s" in web["infra_required"]
    assert "for 1 of the 1 web searches" in web["infra_required"]
    first, resume = _generates(harness)
    assert first["timeout"] == 600.0 and resume["timeout"] == 600.0      # a part of its own
    assert res["harness_errors"] == [] and set(res["category_accuracy"]) == set(MEMORY)


def test_a_resume_the_wall_clock_stops_counts_its_unfinished_entries(harness, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    harness["stop"] = AG.SearchUnavailable("a search failed in transport")
    harness["resume_stop"] = True
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    [err] = res["harness_errors"]
    assert err["part"] == "`bfcl generate` (the memory categories, resumed)"
    assert err["entries_planned"] == 12          # 3 memory categories x 4 planned entries
    assert res["settings_infra_required"] == ["web_search"]
    assert set(res["category_accuracy"]) == set(MEMORY)


def test_a_plain_wall_clock_stop_is_not_the_search(harness, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    harness["stop"] = subprocess.TimeoutExpired(cmd="bfcl generate", timeout=600)
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert "settings_infra_required" not in res and len(_generates(harness)) == 1
    [err] = res["harness_errors"]
    assert err["part"] == "`bfcl generate`" and err["entries_planned"] == 20
    assert set(res["category_accuracy"]) == set(AG.AGENTIC_CATEGORIES)


def test_web_search_alone_losing_its_key_mid_run_is_a_hard_error(harness):
    harness["stop"] = AG.SearchUnavailable("a search was refused")
    with pytest.raises(AG.SearchUnavailable) as e:
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2, eval_categories=",".join(WEB))
    assert e.value.when == "during generation" and "refused" in e.value.reason
    assert not any(c["args"][0] == "evaluate" for c in harness["cli"])


def test_a_full_run_with_a_working_key_is_one_generate_as_upstream(harness):
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    [gen] = _generates(harness)
    assert _categories(gen) == list(AG.AGENTIC_CATEGORIES) and "--allow-overwrite" in gen["args"]
    assert res["status"] == "success" and "settings_infra_required" not in res
    assert set(res["settings"]) == {"memory", "web_search"}
    assert res["settings"]["web_search"]["status"] == "success"
    # the deviations: the web search is served by Gemini grounding, and the no-think 0.0
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == AG.SEARCH_SUBSTITUTION_REASON + "; temperature 0.0 set by default (BFCL generates at 0.001)"
    assert res["bfcl_report"]["search_backend"].startswith(AG.SEARCH_BACKEND)


def test_the_log_watch_stops_generate_with_a_search_unavailable(tmp_path, monkeypatch):
    log = tmp_path / AG.SEARCH_LOG_NAME
    log.write_text(json.dumps({"backend": "all-grounding-models-failed", "results": 0,
                               "error": f"search backend failed: key={base.REDACTED}"}) + "\n")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    problem = AG._search_watch(log)()
    assert isinstance(problem, AG.SearchUnavailable)
    assert "for 1 of the 1 web searches" in problem.reason and KEY not in str(problem)


# --------------------------------------------------------------------------- #
# the launcher: a credential that begins a longer one
# --------------------------------------------------------------------------- #

def _launch_pieces(pieces, env):
    prog = ("import sys, time\n"
            f"pieces = {pieces!r}\n"
            "for i, p in enumerate(pieces):\n"
            "    if i: time.sleep(0.3)\n"
            "    sys.stdout.write(p); sys.stdout.flush()\n")
    return AG._launch([sys.executable, "-c", prog], env, 60, "gbench-test-unused")


def test_a_credential_that_begins_a_longer_one_waits_for_the_rest():
    short = "tokA1b2C3d4E5"
    longer = short + "SECRETtail9876zyx"
    env = dict(os.environ, A_SERVICE_TOKEN=short, B_SERVICE_TOKEN=longer)
    rc, out = _launch_pieces(["x" * 10 + short, longer[len(short):] + "\n"], env)
    assert rc == 0 and out == "x" * 10 + base.REDACTED + "\n"
    assert "SECRETtail" not in out
    # the shorter one alone is masked once the stream shows it is not the longer one
    rc, out = _launch_pieces(["x" * 10 + short, " done\n"], env)
    assert out == "x" * 10 + base.REDACTED + " done\n"


def test_a_line_break_inside_a_multi_line_credential_is_not_a_cut():
    value = "abc1234\nxyz5678"
    env = dict(os.environ, MY_SERVICE_SECRET=value, OTHER_SERVICE_TOKEN="xyz5678QRSTU")
    rc, out = _launch_pieces(["a " + value + " b\n", "c xyz5678QRSTU d\n"], env)
    assert rc == 0 and out == f"a {base.REDACTED} b\nc {base.REDACTED} d\n"
    assert out == base.redact_text("a " + value + " b\nc xyz5678QRSTU d\n",
                                   base.secret_values(env))


# --------------------------------------------------------------------------- #
# the bfcl suite
# --------------------------------------------------------------------------- #

def _single_result(categories):
    ca = {c: {"correct": 3, "total": 4, "accuracy": 75.0} for c in categories}
    return {"benchmark_type": "eval", "eval_name": "bfcl", "model_name": "m", "thinking": False,
            "total_questions": 4 * len(ca), "correct_answers": 3 * len(ca), "failed_requests": 0,
            "empty_responses": 0, "truncated_responses": 0, "over_context_prompts": 0,
            "timed_out_requests": 0, "judge_outages": 0, "response_health": {"ok": 4 * len(ca)},
            "category_accuracy": ca, "accuracy": 75.0, "status": "success",
            "sample_traces": []}


@pytest.fixture
def suite(monkeypatch, harness):
    """run_bfcl with the single-turn and multi-turn parts stubbed; the agentic part is the real
    run_bfcl_v4_agentic over the stubbed harness."""
    seen = {"single": None, "multi": None, "agentic": 0}
    monkeypatch.setattr(bfcl, "_load_bfcl_samples", lambda cats: ([], {}))
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": f"{c}_0", "category": c} for c in multi]))

    def single(**kw):
        seen["single"] = kw
        return _single_result(bfcl.SINGLE_TURN_CATEGORIES)
    monkeypatch.setattr(bfcl, "run_eval_suite", single)

    def multi(model, url, conc, items, **kw):
        seen["multi"] = items
        return {"traces": [{"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0,
                            "category": it["category"], "question_id": it["id"],
                            "is_correct": True, "status": "OK", "health": "ok", "steps": 1}
                           for i, it in enumerate(items)], "guard_refusals": []}
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", multi)
    # the multi-turn part is stubbed, so the network checks before its container
    # (tests/test_guard_code_images.py) do not apply
    monkeypatch.setattr(bfcl, "endpoint_network", lambda base_url: base_url)
    real = AG.run_bfcl_v4_agentic

    def agentic(*a, **k):
        seen["agentic"] += 1
        return real(*a, **k)
    monkeypatch.setattr(AG, "run_bfcl_v4_agentic", agentic)
    return seen


def test_bfcl_without_the_key_scores_everything_but_web_search(suite, harness):
    harness["backend"] = (None, "the Gemini API rejected the key (GEMINI_API_KEYS / "
                                "GEMINI_API_KEY: HTTP 401)")
    res = bfcl.run_bfcl("m", "http://x/v1", 4)
    assert suite["single"] is not None and suite["multi"] and suite["agentic"] == 1
    assert res["settings_infra_required"] == ["web_search"]
    web = res["web_search_setting"]
    assert web["status"] == "infra_required" and "rejected the key" in web["infra_required"]
    assert not set(WEB) & set(res["category_accuracy"]) and not set(WEB) & set(
        res["categories_run"])
    assert set(MEMORY) <= set(res["categories_run"])
    overall = res["bfcl_summary"]["overall"]
    assert overall["Overall Acc"] is None and overall["Web Search Acc"] is None
    assert overall["Memory Acc"] == 50.0
    assert res["bfcl_summary"]["agentic"]["Agentic Overall Acc"] is None
    # a full run whose Web Search setting could not run: Overall Acc is withheld (a harness
    # error), not replaced by the pooled figure, which stays on the result
    assert res["accuracy"] is None and res["pooled_accuracy"] is not None
    assert "Web Search setting" in res["accuracy_withheld_reason"]
    assert res["status"] == "completed_with_errors" and res["leaderboard_comparable"] is False
    assert "Web Search setting" in res["leaderboard_comparable_reason"]
    assert res["agentic_report"]["settings"]["web_search"]["status"] == "infra_required"
    assert KEY not in json.dumps(res)


def test_bfcl_without_memory_does_not_start_the_agentic_part(suite, harness):
    harness["backend"] = (None, "GEMINI_API_KEY (or GEMINI_API_KEYS) is unset")
    res = bfcl.run_bfcl("m", "http://x/v1", 4,
                        eval_categories="single_turn,multi_turn,web_search")
    assert suite["agentic"] == 0 and harness["cli"] == []
    assert res["web_search_setting"]["when"] == "before generation"
    assert res["web_search_setting"]["prerequisite"] == "GEMINI_API_KEY"
    assert res["settings_infra_required"] == ["web_search"]
    assert res["status"] == "completed_with_errors"
    assert (f"prerequisite ({AG.SEARCH_PREREQUISITE}) was not available before generation"
            in res["leaderboard_comparable_reason"])


def test_bfcl_of_web_search_alone_without_the_key_is_a_hard_error(suite, harness):
    harness["backend"] = (None, "GEMINI_API_KEY (or GEMINI_API_KEYS) is unset")
    with pytest.raises(RuntimeError,
                       match=r"INFRA REQUIRED.*bfcl.*GEMINI_API_KEY \(or GEMINI_API_KEYS\) is unset"):
        bfcl.run_bfcl("m", "http://x/v1", 4, eval_categories="web_search")
    assert suite["single"] is None and suite["multi"] is None and harness["cli"] == []


def test_bfcl_keeps_its_parts_when_the_key_dies_mid_run(suite, harness):
    harness["stop"] = AG.SearchUnavailable("Gemini Google Search grounding served none of the "
                                           "models of its cascade for 2 of the 5 web searches")
    res = bfcl.run_bfcl("m", "http://x/v1", 4)
    assert res["web_search_setting"]["when"] == "during generation"
    assert set(bfcl.SINGLE_TURN_CATEGORIES + bfcl.MULTI_TURN_CATEGORIES + MEMORY) <= set(
        res["category_accuracy"])
    assert res["settings_infra_required"] == ["web_search"]


def test_bfcl_keeps_its_parts_when_web_search_alone_loses_its_key(suite, harness):
    harness["stop"] = AG.SearchUnavailable("a search failed in transport")
    res = bfcl.run_bfcl("m", "http://x/v1", 4,
                        eval_categories="single_turn,multi_turn,web_search")
    assert suite["agentic"] == 1
    assert res["web_search_setting"]["when"] == "during generation"
    assert "transport" in res["web_search_setting"]["infra_required"]
    assert set(bfcl.SINGLE_TURN_CATEGORIES) <= set(res["category_accuracy"])
    assert "Web Search setting" in res["leaderboard_comparable_reason"]


def test_bfcl_still_stops_on_another_missing_prerequisite(suite, monkeypatch):
    def run(*a, **k):
        raise AG.infra_required("bfcl_v4_agentic", "the harness image vanished", AG.DOCS_URL)
    monkeypatch.setattr(AG, "run_bfcl_v4_agentic", run)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*harness image vanished"):
        bfcl.run_bfcl("m", "http://x/v1", 4)


# --------------------------------------------------------------------------- #
# ACEBench: LLMInfer's window as vllm==0.6.1.post1 applies it
# --------------------------------------------------------------------------- #

def _msgs(n_chars, user="u"):
    return [{"role": "system", "content": "s" * n_chars}, {"role": "user", "content": user}]


@pytest.mark.parametrize("tokens,want", [
    (7169, (1024, 7169, "server")),        # 8193 - 7169 = 1024: the cap still fits
    (7170, (1023, 7170, "server")),
    (8192, (1, 8192, "server")),           # scheduled: the stop check fires at length 8193
    (8193, (0, 8193, "server")),           # FINISHED_IGNORED: not sent, the reply is ""
    (20000, (0, 20000, "server")),
])
def test_the_window_matches_the_pinned_vllm(monkeypatch, tokens, want):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (tokens, tokens, "server"))
    assert AB.upstream_window_budget(_msgs(9000), 1024) == want


def test_a_conservative_count_decides_nothing_near_the_window(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (7000, 7000, "tokenizer"))
    assert AB.upstream_window_budget(_msgs(9000), 1024) == (1024, 7000, "tokenizer")
    for tokens in (7170, 8192, 9000):
        monkeypatch.setattr(base, "_prompt_token_bounds",
                            lambda *a, _t=tokens, **k: (_t, _t, "tokenizer"))
        assert AB.upstream_window_budget(_msgs(9000), 1024) == (1024, None, "uncounted")


def test_the_uncounted_bound_reaches_the_last_token(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        lambda *a, **k: pytest.fail("a prompt that cannot reach it is not counted"))
    n = AB._UPSTREAM_SEQUENCE_CAP - AB._TEMPLATE_BOUND_TOKENS - 1024 - 1        # "u" is 1 byte
    assert AB.upstream_window_budget(_msgs(n), 1024) == (1024, None, "short")


def test_the_prompt_is_counted_as_upstreams_vllm_encodes_it(monkeypatch):
    """LLMInfer renders the chat template to text and vLLM 0.6.1 encodes it with special
    tokens: /tokenize is asked with add_special_tokens=true."""
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"count": 8193}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setitem(base._RUN_KNOBS, "tokenize_url",
                            f"http://127.0.0.1:{srv.server_address[1]}")
        monkeypatch.setitem(base._RUN_KNOBS, "served_model", "m")
        monkeypatch.setattr(base, "_COUNT_CACHE", {})
        got = AB.upstream_window_budget(_msgs(9000, "tokenize-me"), 1024,
                                        {"enable_thinking": False})
    finally:
        srv.shutdown()
    assert got == (0, 8193, "server")
    [body] = bodies
    assert body["add_special_tokens"] is True and body["add_generation_prompt"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][1]["content"] == "tokenize-me"


def _gold(row_id):
    return json.dumps({"id": row_id, "sub": "normal_atom_bool", "language": "en",
                       "ground_truth": {"Api": {}}, "functions": [], "question": ""})


def _ace_run(monkeypatch, sizes, reply, attempts=1, limit=None):
    """run_acebench over English normal_atom_bool rows whose prompt sizes (server counts) are
    `sizes` (row id -> tokens); returns (result, max_tokens of every request sent)."""
    base.set_run_knobs(max_output_tokens=4096)
    monkeypatch.setitem(base._RUN_KNOBS, "attempt_count", attempts)
    monkeypatch.setattr(AB, "_load_acebench_samples", lambda **k: [
        (_msgs(9000, rid), _gold(rid), {"category": "normal_atom_bool"})
        for rid in sizes] if k.get("language") == "en" else [])
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [])
    monkeypatch.setattr(AB, "check_binop_evaluator", lambda: None)
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda m, *a, **k: (
        sizes[m[1]["content"]],) * 2 + ("server",))
    scored = []
    real_eval = AB._eval_acebench

    def eval_(text, gold):
        scored.append((text, json.loads(gold)["id"]))
        return real_eval(text, gold)
    monkeypatch.setattr(AB, "_eval_acebench", eval_)
    sent = []

    async def send(**k):
        sent.append((k["messages"][1]["content"], k["max_output_tokens"]))
        return reply
    with mock.patch.object(base, "_send_single_request", send):
        res = AB.run_acebench("m", "http://127.0.0.1:9/v1", 1, enable_thinking=False,
                              limit=limit)
    return res, sent, scored


def test_a_row_longer_than_the_window_is_scored_on_an_empty_reply(monkeypatch):
    res, sent, scored = _ace_run(
        monkeypatch, {"normal_atom_bool_1": 3000, "normal_atom_bool_2": 9000},
        base.Reply("[Api()]", None, "stop", completion_tokens=3))
    assert [rid for rid, _m in sent] == ["normal_atom_bool_1"]         # the other is not sent
    assert ("", "normal_atom_bool_2") in scored
    over = [t for t in res["sample_traces"] if t.get("over_upstream_window")]
    [t] = over
    assert (t["response_text"], t["request_sent"], t["finish_reason"]) == ("", False, "length")
    assert t["status"] == "OK" and t["prompt_tokens"] == 9000 and t["max_tokens_sent"] is None
    win = res["acebench_report"]["upstream_context_window"]
    assert win["single_turn_rows_over_window"] == 1 and win["single_turn_rows_limited"] == 0
    assert res["total_questions"] == 2 and res["response_health"]["over_upstream_window"] == 1
    assert res["leaderboard_comparable"] is True, res["leaderboard_comparable_reason"]
    assert res["status"] == "success"
    assert "abort" not in json.dumps(res["acebench_report"])


def test_rows_longer_than_the_window_get_every_attempt(monkeypatch):
    res, sent, _scored = _ace_run(
        monkeypatch, {"normal_atom_bool_1": 3000, "normal_atom_bool_2": 9000},
        base.Reply("[Api()]", None, "stop", completion_tokens=3), attempts=2)
    assert len(sent) == 2
    over = [t for t in res["sample_traces"] if t.get("over_upstream_window")]
    assert sorted(t["attempt_idx"] for t in over) == [0, 1]
    assert len({t["source_sample_idx"] for t in over}) == 1
    sources = {t["source_sample_idx"] for t in res["sample_traces"]}
    assert len(sources) == 2 and res["attempts"]["samples"] == 2
    assert res["attempts"]["generations"] == 4


def test_only_the_room_is_a_per_request_cap(monkeypatch):
    rows = [(_msgs(9000, k), f"g-{k}", {"category": "normal_atom_bool"})
            for k in ("a", "b", "c", "d")]
    sizes = {"a": 100, "b": 7600, "c": 8192, "d": 8193}
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda m, *a, **k: (
        sizes[m[1]["content"]],) * 2 + ("server",))
    send, over, verdicts = AB._single_turn_window(rows)
    assert [(g, (meta or {}).get("max_tokens")) for _m, g, meta in send] == [
        ("g-a", None), ("g-b", 593), ("g-c", 1)]
    assert [row[1] for row, _t in over] == ["g-d"] and verdicts["g-d"][0] == 0


def test_the_limit_and_the_shard_cover_the_rows_not_sent(monkeypatch):
    assert AB._window_quotas(None, 10, 2) == (None, None)
    assert AB._window_quotas(10, 10, 0) == (10, None)
    assert AB._window_quotas(12, 10, 2) == (10, 2)
    assert AB._window_quotas(1, 5, 5) == (1, 0)
    rows = [((_msgs(1, str(i)), f"g{i}", {"category": "normal_atom_bool"}), 9000)
            for i in range(7)]
    shards = []
    for i in (1, 2, 3):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/3")
        shards.append([r[0][1] for r in AB._select_unsent_rows(rows, None)])
    assert sorted(sum(shards, [])) == sorted(r[0][1] for r in rows)
    monkeypatch.delenv("GBENCH_SHARD")
    assert len(AB._select_unsent_rows(rows, 3)) == 3


def _conversation_shared():
    return {"session": None, "api_url": "u", "model_name": "m", "payload": {}, "semaphore": None,
            "budget": 1024, "official_cap": 1024, "temperature": 0.0, "thinking": False,
            "ctk": None, "upstream_window": True}


def test_an_agent_step_longer_than_the_window_hands_an_empty_reply_to_the_user(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (8500, 8500, "server"))
    seen = []
    replies = iter(["I want to send a message.", "Are you there?", "finish conversation"])

    async def user_sim(messages):
        seen.append([dict(m) for m in messages])
        return next(replies), "g1"

    async def send(**k):
        raise AssertionError("a prompt longer than the window is not sent")
    monkeypatch.setattr(AB, "_user_sim_generate", user_sim)
    monkeypatch.setattr(AB, "_Environment", lambda *a, **k: types.SimpleNamespace(refusals=[]))
    # an agent prompt long enough to be counted (the history of a long conversation)
    monkeypatch.setattr(AB, "_multi_turn_agent_prompts", lambda *a: ("s" * 9000, "u"))
    item = {"language": "en", "category": "agent_multi_turn", "key": "agent_multi_turn",
            "row": {"id": "agent_multi_turn_0", "function": [], "initial_config": {},
                    "involved_classes": ["BaseApi"], "question": "send a message"},
            "answer": {"ground_truth": [], "mile_stone": []}}
    with mock.patch.object(base, "_send_single_request", send):
        trace = asyncio.run(AB._run_conversation(_conversation_shared(), item))
    assert trace["status"] == "OK" and trace["model_requests"] == 0
    assert trace["window_over_steps"] == 2
    # the user simulator got the agent's empty reply, as APIUSER gets LLMInfer's ""
    assert seen[1][-1] == {"role": "user", "content": ""}
    assert trace["agent_dialogue"][1] == {"sender": "agent", "recipient": "user", "message": ""}


def test_agent_steps_over_the_window_are_reported_not_flagged(monkeypatch):
    trace = {"status": "OK", "is_correct": True, "process_accuracy": 1.0,
             "acebench_language": "en", "category": "agent_multi_step",
             "gold_answer": json.dumps({"sub": "agent_multi_step", "id": "agent_multi_step_0"}),
             "execution_guard_refusals": [], "truncated_replies": 0,
             "replies_at_official_max_tokens": 0, "model_requests": 2, "user_sim_requests": 0,
             "user_sim_models": {}, "user_sim_replies_cut_at_max_tokens": 0,
             "user_sim_empty_at_max_tokens": 0, "window_limited_requests": 1,
             "window_over_steps": 3, "window_uncounted_requests": 0}
    base.set_run_knobs(max_output_tokens=4096)
    monkeypatch.setitem(base._RUN_KNOBS, "attempt_count", 1)
    monkeypatch.setattr(AB, "_load_acebench_samples", lambda **k: [
        (_msgs(10, "normal_atom_bool_1"), _gold("normal_atom_bool_1"),
         {"category": "normal_atom_bool"})] if k.get("language") == "en" else [])
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [
        {"category": "agent_multi_step", "key": "agent_multi_step", "language": language,
         "row": {"id": "agent_multi_step_0"}, "answer": {}}] if language == "en" else [])
    monkeypatch.setattr(AB, "check_binop_evaluator", lambda: None)

    async def run_agents(*a, **k):
        assert k["upstream_window"] is True
        return [trace]
    monkeypatch.setattr(AB, "_run_agent_items", run_agents)
    monkeypatch.setattr(AB, "check_user_simulator", lambda: None)

    async def send(**k):
        return base.Reply("[Api()]", None, "stop", completion_tokens=3)
    with mock.patch.object(base, "_send_single_request", send):
        res = AB.run_acebench("m", "http://127.0.0.1:9/v1", 1, enable_thinking=False)
    win = res["acebench_report"]["upstream_context_window"]
    assert win["agent_steps_over_window"] == 3 and win["agent_conversations_over_window"] == 1
    assert win["agent_requests_limited"] == 1
    assert res["leaderboard_comparable"] is True, res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #

def test_the_docs_describe_the_settings_and_the_window():
    ace = " ".join((DOCS / "acebench.md").read_text(encoding="utf-8").split())
    for needle in ("vllm==0.6.1.post1", "FINISHED_IGNORED", "8193 - P",
                   "add_special_tokens: true", "single_turn_rows_over_window",
                   "agent_steps_over_window", "request_sent: false"):
        assert needle in ace, needle
    assert "aborts the run" not in ace and "only so it has a trace" not in ace
    bf = " ".join((DOCS / "bfcl.md").read_text(encoding="utf-8").split())
    for needle in ("web_search_setting", "settings_infra_required", "harness' own resume"):
        assert needle in bf, needle
    assert "is a hard error like a missing one" not in bf
    ag = " ".join((DOCS / "bfcl_v4_agentic.md").read_text(encoding="utf-8").split())
    for needle in ("settings.web_search", "settings_infra_required", "memory_resumed",
                   "without `--allow-overwrite`", "clean_up_memory_prereq_entries"):
        assert needle in ag, needle
    assert "never falls back to a memory-only run" not in ag
    for name in ("acebench.md", "bfcl.md", "bfcl_v4_agentic.md"):
        assert (DOCS / name).read_text(encoding="utf-8").isascii(), name


# --------------------------------------------------------------------------- #
# the real harness image (GBENCH_TEST_BFCL_IMAGE=1): the memory resume
# --------------------------------------------------------------------------- #

image = pytest.mark.skipif(not os.environ.get("GBENCH_TEST_BFCL_IMAGE"),
                           reason="set GBENCH_TEST_BFCL_IMAGE=1 to run the BFCL harness image")


def _stub_model(log):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"object": "list", "data": [{"id": "stub", "object": "model"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log.append(req)
            self._send({"id": "x", "object": "chat.completion", "created": 0,
                        "model": req["model"],
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant", "content": "I do not know."}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                  "total_tokens": 15}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


@image
def test_image_a_web_search_stop_leaves_memory_to_the_harness_resume(tmp_path, monkeypatch):
    """The pinned harness' own resume finishes the memory entries a stopped generate left: the
    first generate runs for real, then an entry it wrote is removed and a cut-off line added (as
    a stop leaves them) and the web search is declared lost; the resume regenerates only what is
    missing and `bfcl evaluate` scores every planned memory entry."""
    log = []
    srv, url = _stub_model(log)
    root = tmp_path / "bfcl"
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))
    real_cli = AG._run_cli
    seen = {"calls": [], "removed": None}

    def cli(args, env, timeout, model="", backend=None):
        seen["calls"].append(list(args))
        rc, out = real_cli(args, env, timeout, model=model, backend=backend)
        if args[0] == "generate" and AG.SEARCH_LOG_ENV in env:
            [path] = list(root.glob("result_limit3/stub/**/BFCL_v4_memory_kv_result.json"))
            lines = path.read_text(encoding="utf-8").splitlines()
            seen["removed"] = json.loads(lines[-1])["id"]
            path.write_text("\n".join(lines[:-1]) + '\n{"id": "memory_kv_cut", "res',
                            encoding="utf-8")
            raise AG.SearchUnavailable("simulated: a search no grounding model served")
        return rc, out
    monkeypatch.setattr(AG, "_run_cli", cli)
    try:
        res = AG.run_bfcl_v4_agentic("stub", url, 2,
                                     eval_categories="memory_kv,web_search_base", limit=3)
    finally:
        srv.shutdown()
    assert [c[0] for c in seen["calls"]] == ["generate", "generate", "evaluate"]
    assert "--allow-overwrite" not in seen["calls"][1]
    [path] = list(root.glob("result_limit3/stub/**/BFCL_v4_memory_kv_result.json"))
    ids = [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert seen["removed"] in ids and "memory_kv_cut" not in ids
    assert res["settings_infra_required"] == ["web_search"]
    assert set(res["category_accuracy"]) == {"memory_kv"}
    assert res["category_accuracy"]["memory_kv"]["total"] == 2
    assert res["bfcl_report"]["memory_resumed"] is True and res["harness_errors"] == []
    assert KEY not in json.dumps(res)
