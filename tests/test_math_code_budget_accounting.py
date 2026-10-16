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

"""Token-budget accounting of livebench and aider_polyglot, and MathArena client fields.

* livebench: `max_output_tokens_note` never names a value as sent when LiveBench was given
  another one (a budget lowered to fit the served window, in every token-limit policy), and replies
  cut off after the answer began are counted from LiveBench's answer rows with and without
  --thinking (read_answer_rows); rows that cannot be checked, and LiveBench's api_error rows other
  than a reply the model finished empty (counted apart as `empty`), make the run not comparable.
* aider_polyglot: aider's own num_exhausted_context_windows (a leaderboard column) is summed,
  counted as replies cut off by the served window, and makes the run not comparable; the other
  leaderboard columns aider's summarize_results computes are reported.
* hmmt/aime: MathArena's batch_processing is an APIClient constructor argument, not a request
  field, so it is no gap.
* A reply cut off by a limit other than the suite's official cap (the run's budget, a
  max_tokens the context window lowered, or the window's end where no max_tokens is sent) is
  counted (`replies_cut_off_by_budget`, `replies_cut_off_in_reasoning`) and makes the result
  not comparable: bigcodebench, lcb, multipl_e, ojbench, scicode, hmmt and aime. A reply that
  stopped at the official cap is the protocol's outcome.
* livebench's one fixed max_tokens always fits the served window (lowered as base lowers a
  request, recorded; a lowered official cap is not comparable), and an image whose LiveBench
  commit cannot be read is not comparable.
* hmmt and aime reproduce MathArena's own config for a served model it ships one for
  (hmmt.MATHARENA_MODEL_CONFIGS), max_tokens included; Gemma keeps generation_config.json.
* ojbench's unused mismatch helper is gone.

The endpoint of the second half is a local stub HTTP server on an ephemeral port, so the tests
see what base actually sends. Docker and the graders are stubbed.
"""

import json
import re
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import aime as AI
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bigcodebench as B
from gbench.runners.eval_suites import hmmt
from gbench.runners.eval_suites import hmmt as H
from gbench.runners.eval_suites import lcb as L
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import matharena_common as MA
from gbench.runners.eval_suites import multipl_e as M
from gbench.runners.eval_suites import ojbench as OJ
from gbench.runners.eval_suites import scicode as SC
from gbench.runners.eval_suites import tau_common as TC_GUARD


# --------------------------------------------------------------------------- #
# livebench                                                                    #
# --------------------------------------------------------------------------- #
#: LiveBench's config for gemma-4-31b-it at 0baa4aa7 (livebench/model/model_configs/google.yml).
GEMMA4_31B = {
    "display_name": "gemma-4-31b-it",
    "api_name": {"google": "gemma-4-31b-it",
                 "https://openrouter.ai/api/v1": "google/gemma-4-31b-it"},
    "default_provider": "https://openrouter.ai/api/v1",
    "api_keys": {"https://openrouter.ai/api/v1": "OPENROUTER_API_KEY"},
    "aliases": ["gemma-4-31b"],
    "api_kwargs": {"default": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 65536},
                   "https://openrouter.ai/api/v1": {
                       "extra_body": {"reasoning": {"enabled": True}}}},
    "prompt_prefix": None, "preserve_reasoning": None, "cost_per_million": None, "api_base": None,
    "agentic_native_tools": None,
}

OK_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
          "total_output_tokens": 10}


def _row(answer, total, **extra):
    return {"question_id": "q", "choices": [{"index": 0, "turns": [answer]}],
            "total_output_tokens": total, **extra}


def _run(monkeypatch, tmp_path, config=None, rows=(OK_ROW,), extra=None, **kwargs):
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "2025-04-02")
    monkeypatch.delenv("GBENCH_LIVEBENCH_CATEGORIES", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_MODEL", raising=False)
    if extra is None:
        monkeypatch.delenv("GBENCH_LIVEBENCH_EXTRA_ARGS", raising=False)
    else:
        monkeypatch.setenv("GBENCH_LIVEBENCH_EXTRA_ARGS", extra)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config", lambda image, names: {
        "name": names[-1] if config else None, "config": config,
        "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    seen = {}

    def fake_run(cmd, **_kw):
        seen["inner"] = cmd[-1]
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\ngoogle-gemma-4-31b-it,50.0,"
            + ",".join("50.0" for _ in cols) + "\n", encoding="utf-8")
        d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "math" / "x" / "model_answer"
        d.mkdir(parents=True, exist_ok=True)
        (d / "google-gemma-4-31b-it.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        if rows:   # instruction_following is scored: its answer rows are copied out too
            d = (tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "instruction_following"
                 / "paraphrase" / "model_answer")
            d.mkdir(parents=True, exist_ok=True)
            (d / "google-gemma-4-31b-it.jsonl").write_text(json.dumps(OK_ROW) + "\n",
                                                           encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    r = LB.run_livebench("google/gemma-4-31B-it", "http://h/v1", 4, **kwargs)
    return r, seen["inner"]


def test_lowered_run_budget_note_names_only_what_was_sent(monkeypatch, tmp_path):
    # --thinking, no LiveBench config, 32768-token window, --max-output-tokens 30000.
    base.set_run_knobs(max_output_tokens=30000, max_model_len=32768)
    r, inner = _run(monkeypatch, tmp_path, enable_thinking=True, max_output_tokens=30000)
    assert "--max-tokens 24576 " in inner and r["max_output_tokens"] == 24576
    note = r["max_output_tokens_note"]
    assert "was sent this run's budget" not in note and "was sent instead" not in note
    assert note == (
        f"{base.REASONING_CAP_NOTE} (gen_api_answer.py's --max-tokens default is 4096); this "
        "run's budget is --max-output-tokens 30000; lowered to 24576 to fit the served context "
        "window (32768 tokens, 8192 left for the prompt); each request was sent max_tokens=24576")
    assert r["official_sampling"]["max_tokens_sent"] == 24576


def test_lowered_config_budget_note_names_only_what_was_sent(monkeypatch, tmp_path):
    # gemma-4-31b-it's LiveBench config, --thinking, 98304-token window, --max-output-tokens 96000.
    base.set_run_knobs(max_output_tokens=96000, max_model_len=98304)
    r, _ = _run(monkeypatch, tmp_path, config=GEMMA4_31B, enable_thinking=True,
                max_output_tokens=96000)
    assert r["max_output_tokens"] == 90112
    note = r["max_output_tokens_note"]
    assert "was sent instead" not in note
    assert note == (
        "LiveBench's config for gemma-4-31b-it sets max_tokens=65536; --max-output-tokens 96000 "
        "is larger and is the budget a thinking run gets (a thinking run is never capped below "
        "--max-output-tokens); lowered to 90112 to fit the served context window (98304 tokens, "
        "8192 left for the prompt); each request was sent max_tokens=90112")
    rec = r["official_sampling"]
    assert rec["max_tokens"] == 65536 and rec["max_tokens_sent"] == 90112
    assert rec["max_tokens_applied"] is False
    assert "was sent instead" not in rec["max_tokens_note"]
    assert "max_tokens 90112 was sent (fitted to the served context window)" in \
        rec["max_tokens_note"]


def test_lowered_official_cap_note_names_only_what_was_sent(monkeypatch, tmp_path):
    base.set_run_knobs(max_model_len=6000)
    r, inner = _run(monkeypatch, tmp_path, rows=(_row("long", 100),))
    assert "--max-tokens 3000 " in inner
    assert r["max_output_tokens_note"] == (
        "gen_api_answer.py's --max-tokens default is 4096, the cap a no-think run gets; lowered "
        "to 3000 to fit the served context window (6000 tokens, 3000 left for the prompt); each "
        "request was sent max_tokens=3000")


def test_unlowered_budget_keeps_the_policy_note(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    r, _ = _run(monkeypatch, tmp_path, enable_thinking=True, max_output_tokens=32768)
    plan = base.resolve_token_budget("livebench", LB.OFFICIAL_SAMPLING, thinking=True,
                                     max_output_tokens=32768)
    assert r["max_output_tokens_note"] == plan.note
    assert "each request was sent this run's budget, --max-output-tokens 32768" in plan.note


def test_extra_args_budget_note_says_what_replaced_the_policy(monkeypatch, tmp_path):
    r, inner = _run(monkeypatch, tmp_path, extra="--max-tokens 8000")
    assert inner.count("--max-tokens") == 1 and r["max_output_tokens"] == 8000
    assert r["max_output_tokens_note"] == (
        "gen_api_answer.py's --max-tokens default is 4096, the cap a no-think run gets; "
        "GBENCH_LIVEBENCH_EXTRA_ARGS asked for --max-tokens 8000 instead; each request was sent "
        "max_tokens=8000")


def _write_rows(tmp_path, rows):
    d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "math" / "t" / "model_answer"
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n",
                               encoding="utf-8")


def test_read_answer_rows_counts_a_reply_cut_off_after_the_answer_began(monkeypatch, tmp_path):
    """The review's probe: vLLM completion_tokens C = 32768 = max_tokens, reasoning_tokens
    R = 30000, which LiveBench adds again: total 62768. The answer is K = 2768 tokens, so
    C = (total + K) / 2 = 32768."""
    cut_text = "The answer is 4 because ... (cut"
    finished_text = "The answer is 4."
    counts = {cut_text: 2768, finished_text: 100}
    monkeypatch.setattr(LB, "_token_count", lambda text: counts[text])
    _write_rows(tmp_path, [
        _row(cut_text, 62768),                       # cut after the answer began
        _row(finished_text, 40000),                  # C = 20050: finished well inside the budget
        _row("x", 32768),                            # total at max_tokens: cut
        _row("y", 1000),                             # below max_tokens: C is below it too
        _row("", 32768, eval_status="token_exhaustion"),
    ])
    rows = LB.read_answer_rows(str(tmp_path), 32768)
    assert rows == {"rows": 5, "cut": 3, "in_reasoning": 1, "uncounted": 0, "empty": 0,
                    "failed": 0}
    assert LB.answer_cutoffs(str(tmp_path), 32768, True) == (3, 1, 5)
    assert LB.answer_cutoffs(str(tmp_path), 32768, False) == (3, 1, 5)


def test_read_answer_rows_tolerates_the_uncounted_delimiters(monkeypatch, tmp_path):
    # C = 32768 but K was retokenized 20 tokens short (delimiters and retokenization): the
    # estimate (62768 + 2748) / 2 = 32758 is within the tolerance, so still a cut-off.
    monkeypatch.setattr(LB, "_token_count", lambda text: 2748)
    _write_rows(tmp_path, [_row("answer", 62768)])
    assert LB.read_answer_rows(str(tmp_path), 32768)["cut"] == 1
    # far below the budget it is not
    monkeypatch.setattr(LB, "_token_count", lambda text: 2768 - 2 * 40)
    assert LB.read_answer_rows(str(tmp_path), 32768)["cut"] == 0


def test_read_answer_rows_uses_the_runs_tokenizer(monkeypatch, tmp_path):
    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    monkeypatch.setattr(base, "_run_tokenizer", lambda: FakeTokenizer())
    _write_rows(tmp_path, [_row("a b c d", 100 + 96)])    # (196 + 4) / 2 = 100 = max_tokens
    assert LB.read_answer_rows(str(tmp_path), 100)["cut"] == 1


def test_read_answer_rows_names_what_it_cannot_check(monkeypatch, tmp_path):
    monkeypatch.setattr(LB, "_token_count", lambda text: None)       # no --tokenizer
    _write_rows(tmp_path, [
        _row("a", 50000),                                            # above max_tokens
        _row("b", -1),                                               # no usage from the server
        {"question_id": "q", "choices": [{"index": 0, "turns": ["c", "d"]}],
         "total_output_tokens": 40000},                              # several turns
        {"question_id": "q", "choices": [{"index": 0, "turns": ["e", "f"]}],
         "total_output_tokens": 100},                                # several turns, below
        _row("$ERROR$", 0, eval_status="api_error"),
    ])
    rows = LB.read_answer_rows(str(tmp_path), 32768)
    # the api_error row carries no "No message returned from litellm" error: an endpoint failure,
    # not an empty reply the model finished
    assert rows == {"rows": 5, "cut": 0, "in_reasoning": 0, "uncounted": 3, "empty": 0,
                    "failed": 1}


def test_run_livebench_thinking_mid_answer_cutoff_is_counted(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    monkeypatch.setattr(LB, "_token_count", lambda text: 2768)
    r, _ = _run(monkeypatch, tmp_path, rows=(OK_ROW, _row("The answer is 4 ... (cut", 62768)),
                enable_thinking=True, max_output_tokens=32768)
    assert r["replies_cut_off_by_budget"] == 1 and r["replies_cut_off_in_reasoning"] == 0
    assert r["replies_cut_off_uncounted"] == 0
    assert r["leaderboard_comparable"] is False
    assert ("1 repl(ies) cut off by this run's max_tokens 32768, not by an official cap (0 of "
            "them inside the reasoning channel") in r["leaderboard_comparable_reason"]
    assert "(total + the answer's tokens under --tokenizer) / 2" in r["replies_cut_off_note"]


def test_run_livebench_rows_it_cannot_check_are_not_comparable(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    monkeypatch.setattr(LB, "_token_count", lambda text: None)
    r, _ = _run(monkeypatch, tmp_path, rows=(OK_ROW, _row("answer", 50000)),
                enable_thinking=True, max_output_tokens=32768)
    assert r["replies_cut_off_by_budget"] == 0 and r["replies_cut_off_uncounted"] == 1
    assert r["leaderboard_comparable"] is False
    assert ("1 answer row(s) could not be checked for a cut-off at this run's max_tokens 32768"
            in r["leaderboard_comparable_reason"])
    assert "pass --tokenizer" in r["leaderboard_comparable_reason"]


def test_run_livebench_endpoint_failures_are_harness_errors(monkeypatch, tmp_path):
    # at the official cap no cut-off is counted, but a failed request is still a harness error
    r, _ = _run(monkeypatch, tmp_path,
                rows=(OK_ROW, _row("$ERROR$", 0, eval_status="api_error")))
    assert r["max_output_tokens"] == 4096 and r["replies_cut_off_by_budget"] == 0
    assert r["failed_requests"] == 1 and r["status"] == "completed_with_errors"
    assert r["leaderboard_comparable"] is False
    assert "1 request(s) failed at the served endpoint" in r["leaderboard_comparable_reason"]
    ok, _ = _run(monkeypatch, tmp_path, rows=(OK_ROW,))
    assert ok["failed_requests"] == 0 and ok["status"] == "success"
    assert ok["leaderboard_comparable"] is True, ok.get("leaderboard_comparable_reason")


# --------------------------------------------------------------------------- #
# aider_polyglot                                                               #
# --------------------------------------------------------------------------- #
def _aider_result(lang, outcomes, **counters):
    return {"testcase": "x", "language": lang, "model": "openai/m", "edit_format": "whole",
            "commit_hash": "abc1234", "tests_outcomes": outcomes, "cost": 0.0,
            "duration": 10.0, "test_timeouts": 0, "num_error_outputs": 0, "num_user_asks": 0,
            "num_exhausted_context_windows": 0, "num_malformed_responses": 0,
            "syntax_errors": 0, "indentation_errors": 0, "lazy_comments": 0,
            "prompt_tokens": 100, "completion_tokens": 50, **counters}


def _write_aider(run_dir, name, lang, data):
    d = run_dir / lang / "exercises" / "practice" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / ".aider.results.json").write_text(json.dumps(data), encoding="utf-8")


def test_parse_results_computes_aiders_leaderboard_columns(tmp_path):
    _write_aider(tmp_path, "p1", "python", _aider_result("python", [True]))
    _write_aider(tmp_path, "p2", "python", _aider_result(
        "python", [False, True], num_exhausted_context_windows=2, num_malformed_responses=1,
        num_error_outputs=3, test_timeouts=1, duration=30.0, cost=0.5))
    _write_aider(tmp_path, "r1", "rust", _aider_result(
        "rust", [False, False], num_exhausted_context_windows=1, syntax_errors=2,
        num_user_asks=1, lazy_comments=1, indentation_errors=1))
    (tmp_path / "go" / "exercises" / "practice" / "g1").mkdir(parents=True)
    d = tmp_path / "go" / "exercises" / "practice" / "g1"
    (d / ".aider.results.json").write_text(json.dumps({"exception": "Traceback ..."}))
    agg = A._parse_results(str(tmp_path))
    assert agg["n"] == 3 and agg["pass1"] == 1 and agg["pass2"] == 2
    assert agg["columns"] == {
        "test_cases": 3, "total_tests": 4, "pass_rate_1": 33.3, "pass_rate_2": 66.7,
        "pass_num_1": 1, "pass_num_2": 2, "percent_cases_well_formed": 66.7,
        "num_with_malformed_responses": 1, "error_outputs": 3, "num_malformed_responses": 1,
        "user_asks": 1, "lazy_comments": 1, "syntax_errors": 2, "indentation_errors": 1,
        "exhausted_context_windows": 3, "test_timeouts": 1, "prompt_tokens": 300,
        "completion_tokens": 150, "seconds_per_case": 16.7, "total_cost": 0.5,
        "model": "openai/m", "edit_format": "whole", "commit_hash": "abc1234"}


def _aider_run(monkeypatch, tmp_path, exhausted_per_case, window=None, thinking=True):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    for var in ("GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_EDIT_FORMAT", "GBENCH_AIDER_EXTRA_ARGS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    if window:
        base.set_run_knobs(max_model_len=window)
    langs = ["cpp", "go", "java", "javascript", "python", "rust"]

    def fake_run(cmd, **_kw):
        run_dir = tmp_path / "2026-09-25-00-00-00--gbench"
        for i in range(A._EXPECTED_FULL):
            lang = langs[i % len(langs)]
            _write_aider(run_dir, f"x{i}", lang, _aider_result(
                lang, [True], num_exhausted_context_windows=exhausted_per_case if i < 2 else 0))
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            # the run's temperature (base.resolve_temperature): 1.0 with thinking, 0.0 without
            "temperature": 1.0 if thinking else 0.0}}
        # the container prints its OOM-kill count last (none here)
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n"
                                     + A._MEMORY_MARKER + "12884901888\n" + A._OOM_MARKER + "0\n")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    return A.run_aider_polyglot("m", "http://h/v1", concurrency=1, enable_thinking=thinking)


def test_aider_exhausted_context_windows_are_counted_and_not_comparable(monkeypatch, tmp_path):
    r = _aider_run(monkeypatch, tmp_path, exhausted_per_case=1, window=32768)
    assert r["exhausted_context_windows"] == 2 and r["replies_cut_off_by_budget"] == 2
    assert r["aider_leaderboard_columns"]["exhausted_context_windows"] == 2
    assert r["aider_leaderboard_columns"]["test_cases"] == A._EXPECTED_FULL
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "2 exhausted context window(s) (aider's num_exhausted_context_windows" in reason
    assert "served 32768-token context window" in reason
    assert "serve a larger --max-model-len" in reason
    assert r["status"] == "success" and r["accuracy"] == 100.0


def test_aider_run_without_exhausted_windows_stays_comparable(monkeypatch, tmp_path):
    r = _aider_run(monkeypatch, tmp_path, exhausted_per_case=0, thinking=False)
    assert r["exhausted_context_windows"] == 0 and r["replies_cut_off_by_budget"] == 0
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")
    assert r["aider_leaderboard_columns"]["percent_cases_well_formed"] == 100.0


# --------------------------------------------------------------------------- #
# hmmt / aime: MathArena client fields                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ["stepfun/3.5-flash", "stepfun/3.7-flash", "tiiuae/7b-reasoning"])
@pytest.mark.parametrize("thinking", [True, False])
def test_matharena_batch_processing_is_not_a_request_field(path, thinking):
    blob, _hf, fields = hmmt.MATHARENA_MODEL_CONFIGS[path]
    assert fields["batch_processing"] is False
    _sampling, _extra, not_applicable, gaps = hmmt.translate_matharena_config(path, blob, fields,
                                                                              thinking)
    assert gaps == [] and not_applicable == []
    assert "batch_processing" in hmmt._MA_CLIENT_FIELDS


def test_extra_args_budget_above_the_official_cap_is_not_the_cap_applied(monkeypatch, tmp_path):
    r, _ = _run(monkeypatch, tmp_path, config=GEMMA4_31B, enable_thinking=True,
                extra="--max-tokens 80000")
    rec = r["official_sampling"]
    assert r["max_output_tokens"] == 80000 and rec["max_tokens_sent"] == 80000
    assert rec["max_tokens"] == 65536 and rec["max_tokens_applied"] is False


# --------------------------------------------------------------------------- #
# Token limits and served-model settings of the other maths and code suites
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def endpoint():
    """A stub OpenAI-compatible endpoint that records every request body."""
    records = []
    state = {"reply": lambda body: ("ok", None, "stop")}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):                                      # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            records.append(body)
            content, reasoning, finish = state["reply"](body)
            msg = {"role": "assistant", "content": content}
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
            data = json.dumps({"id": "x", "object": "chat.completion", "choices": [
                {"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 7,
                          "total_tokens": 17}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield types.SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                                    records=records, state=state)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _cut_in_reasoning(body):
    """A thinking model that ran out of room while reasoning: no content, finish length."""
    return "", "Let me think about this step by step ...", "length"


def _cut_mid_answer(body):
    """A reply that stopped at the limit part-way through its answer."""
    return "```python\ndef task_func():\n    x = [", "reasoning", "length"


# --------------------------------------------------------------------------- #
# bigcodebench                                                                 #
# --------------------------------------------------------------------------- #
def _bcb_setup(monkeypatch):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_TEMPERATURE", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [
        ([{"role": "user", "content": B.make_raw_chat_prompt("Do X.", split)}], "BigCodeBench/0",
         {"category": split, "task_id": "BigCodeBench/0", "split": split, "hard": False})
        for split in B.SPLITS])

    def empty_programs_fail(subset, parallel, metrics, image):
        async def _score(traces):
            for t in traces:
                t["is_correct"] = bool(str(t.get("response_text") or "").strip())
                t["status"] = "OK"
        return _score

    monkeypatch.setattr(B, "_make_scorer", empty_programs_fail)


def test_bigcodebench_thinking_reply_cut_off_in_reasoning_is_not_comparable(monkeypatch,
                                                                           endpoint):
    # The review's probe: every reply runs out of the run's budget inside the reasoning channel.
    # A --thinking request is sent the run's budget in place of run_codegen's 1280 (the
    # token-limit policy), so a reply that stops there measures that budget, not the protocol.
    _bcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384, max_model_len=32768)
    endpoint.state["reply"] = _cut_in_reasoning
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert [b["max_tokens"] for b in endpoint.records] == [16384, 16384]
    assert r["accuracy"] == 0.0
    assert r["replies_cut_off_by_budget"] == 2 and r["replies_cut_off_in_reasoning"] == 2
    assert r["leaderboard_comparable"] is False
    assert ("2 repl(ies) stopped at this run's max_tokens (16384), 2 of them inside the reasoning "
            "channel with no answer (run_codegen's 1280 is not applied to reasoning; raise "
            "--max-output-tokens") in r["leaderboard_comparable_reason"]


def test_bigcodebench_thinking_reply_cut_off_mid_answer_is_counted_too(monkeypatch, endpoint):
    _bcb_setup(monkeypatch)
    endpoint.state["reply"] = _cut_mid_answer
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    # no --max-output-tokens: every request is sent the thinking floor
    floor = base.THINKING_MIN_OUTPUT_TOKENS
    assert [b["max_tokens"] for b in endpoint.records] == [floor, floor]
    assert r["replies_cut_off_by_budget"] == 2 and r["replies_cut_off_in_reasoning"] == 0
    assert f"2 repl(ies) stopped at this run's max_tokens ({floor}), 0 of them" in \
        r["leaderboard_comparable_reason"]


def test_bigcodebench_no_think_reply_at_run_codegens_1280_is_the_protocols_outcome(monkeypatch,
                                                                                 endpoint):
    _bcb_setup(monkeypatch)
    endpoint.state["reply"] = lambda body: ("\n    return 1", None, "length")
    r = B.run_bigcodebench("m", endpoint.url, 2)
    assert {b["max_tokens"] for b in endpoint.records} == {1280}
    assert r["replies_at_official_max_tokens"] == 2
    assert r["replies_cut_off_by_budget"] == 0 and r["replies_cut_off_in_reasoning"] == 0
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


# --------------------------------------------------------------------------- #
# lcb and multipl_e: every reply stopped off the official cap                  #
# --------------------------------------------------------------------------- #
_STDIN_Q = "Read two integers and print their sum."


def _lcb_setup(monkeypatch):
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", "1")
    monkeypatch.delenv("GBENCH_LCB_TEMPERATURE", raising=False)
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    samples = [([{"role": "system", "content": L._SYSTEM_MESSAGE_GENERIC},
                 {"role": "user", "content": L._generation_prompt(_STDIN_Q, "")}],
                {"task": "generation", "fn_name": None, "question_content": _STDIN_Q,
                 "tests": [{"input": "2 3", "output": "5"}]},
                {"category": "code_gen_easy", "contest_date": "2025-02-01"})]
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: samples)


def test_lcb_thinking_reply_cut_off_mid_answer_is_not_comparable(monkeypatch, endpoint):
    _lcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=32768)
    endpoint.state["reply"] = _cut_mid_answer
    r = L.run_lcb("m", endpoint.url, 4, enable_thinking=True)
    assert r["replies_cut_off_by_budget"] == 20          # 10 generations + 10 self-repairs
    assert r["replies_cut_off_in_reasoning"] == 0
    assert r["self_repair"]["cut_off_by_budget"] == 10
    assert all(t["lcb_self_repair"].get("max_tokens_sent") == 32768
               for t in r["sample_traces"] if not t["lcb_self_repair"]["kept"])
    assert ("20 repl(ies) (10 generation, 10 self-repair) stopped at a max_tokens other than "
            "lcb_runner's 2000 before finishing (this run's budget") in \
        r["leaderboard_comparable_reason"]


def test_lcb_no_think_reply_at_lcb_runners_2000_is_not_a_cutoff(monkeypatch, endpoint):
    _lcb_setup(monkeypatch)
    endpoint.state["reply"] = _cut_mid_answer
    r = L.run_lcb("m", endpoint.url, 4)
    assert {b["max_tokens"] for b in endpoint.records} == {2000}
    assert r["replies_cut_off_by_budget"] == 0 and r["replies_cut_off_in_reasoning"] == 0
    assert r["self_repair"]["cut_off_by_budget"] == 0
    assert "stopped at a max_tokens" not in (r.get("leaderboard_comparable_reason") or "")


def _mpe_setup(monkeypatch):
    monkeypatch.setenv("GBENCH_MULTIPL_E_SAMPLES", "2")
    monkeypatch.delenv("GBENCH_MULTIPL_E_LANGS", raising=False)
    monkeypatch.delenv("GBENCH_MULTIPL_E_TEMPERATURE", raising=False)
    monkeypatch.setattr(M, "check_multipl_e_prerequisites", lambda: (True, ""))
    gold = json.dumps({"name": "HumanEval_0", "language": "jl", "prompt": "p", "tests": "t",
                       "stop_tokens": []})
    monkeypatch.setattr(M, "_load_multipl_e_samples", lambda **k: [
        ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], gold,
         {"category": "jl", "name": "HumanEval_0"})])

    def fake_scorer(image, workers, metrics):
        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = False, "OK"
            metrics.update(chat_adapter_parse_failures=0, chat_adapter_empty_responses=0,
                           multipl_e_no_response=0)
        return _score

    monkeypatch.setattr(M, "_make_scorer", fake_scorer)


def test_multipl_e_cutoffs_per_mode(monkeypatch, endpoint):
    _mpe_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    endpoint.state["reply"] = _cut_mid_answer
    r = M.run_multipl_e("m", endpoint.url, 2)
    assert r["replies_cut_off_by_budget"] == 0                   # at the script's 2048
    assert "stopped at a max_tokens" not in (r.get("leaderboard_comparable_reason") or "")
    r = M.run_multipl_e("m", endpoint.url, 2, enable_thinking=True)
    assert r["replies_cut_off_by_budget"] == 2 and r["replies_cut_off_in_reasoning"] == 0
    assert ("2 repl(ies) stopped at a max_tokens other than the chat script's 2048 before "
            "finishing (this run's budget") in r["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# ojbench                                                                      #
# --------------------------------------------------------------------------- #
def _oj_setup(monkeypatch):
    monkeypatch.setenv("GBENCH_OJBENCH_SAMPLES", "1")
    monkeypatch.delenv("GBENCH_OJBENCH_TEMPERATURE", raising=False)
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(OJ, "_load_ojbench_samples", lambda limit=None: [
        ([{"role": "user", "content": "solve"}], 1,
         {"category": lang, "row": {"id": 1, "dataset": "NOI", "language": lang,
                                    "difficulty": "easy"}}) for lang in ("cpp", "python")])

    def fake_scorer(workers):
        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = False, "OK"
        return _score

    monkeypatch.setattr(OJ, "_make_scorer", fake_scorer)


def test_ojbench_uses_the_papers_settings_per_mode_through_the_policy(monkeypatch, endpoint):
    _oj_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=32768)
    r = OJ.run_ojbench("m", endpoint.url, 2)
    assert all("max_tokens" not in b for b in endpoint.records)
    assert r["official_sampling"]["send_max_tokens"] is False
    assert r["max_output_tokens_note"].startswith("not sent: the official harness sends no "
                                                  "max_tokens")
    endpoint.records.clear()
    r = OJ.run_ojbench("m", endpoint.url, 2, enable_thinking=True)
    assert {b["max_tokens"] for b in endpoint.records} == {65536}
    assert r["official_sampling"]["max_tokens_sent"] == 65536
    assert "the official reasoning-model sampler sends max_tokens=65536" in \
        r["max_output_tokens_note"]


def test_ojbench_reply_at_the_papers_64k_is_its_outcome_but_the_window_is_not(monkeypatch,
                                                                            endpoint):
    _oj_setup(monkeypatch)
    endpoint.state["reply"] = _cut_mid_answer
    r = OJ.run_ojbench("m", endpoint.url, 2, enable_thinking=True)
    assert r["replies_at_official_max_tokens"] == 2 and r["replies_cut_off_by_budget"] == 0
    assert "repl(ies)" not in r["leaderboard_comparable_reason"]     # only the 1-candidate note
    base.set_run_knobs(max_model_len=262144)
    r = OJ.run_ojbench("m", endpoint.url, 2)                     # no max_tokens: the window
    assert r["replies_cut_off_by_budget"] == 2
    assert ("2 repl(ies) stopped at the end of the served 262144-token context window (no "
            "max_tokens is sent) before finishing") in r["leaderboard_comparable_reason"]


def test_ojbench_unused_mismatch_helper_is_gone():
    assert not hasattr(OJ, "recommended_sampling_mismatch")


# --------------------------------------------------------------------------- #
# scicode                                                                      #
# --------------------------------------------------------------------------- #
def test_scicode_substep_reply_at_the_window_is_not_comparable(monkeypatch):
    monkeypatch.setattr(SC, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(SC, "resolve_test_data", lambda: ("/nonexistent.h5", "test"))
    monkeypatch.setattr(SC, "_digest", lambda path: SC._TEST_DATA_SHA256)
    monkeypatch.setattr(SC, "_load_scicode_problems", lambda limit=None: [
        {"problem_id": "1", "problem_name": "p1", "required_dependencies": "",
         "sub_steps": [{"step_number": "1.1", "test_cases": ["assert True"]}]}])
    monkeypatch.delenv("GBENCH_SCICODE_TEMPERATURE", raising=False)

    def fake_one(client, model, prob, **k):
        return {"problem_id": "1", "problem_name": "p1", "steps": [
            {"step_number": "1.1", "special": False, "gen_failed": False, "correct": False,
             "response_text": "", "finish_reason": "length", "error": None}]}

    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **k: object())
    monkeypatch.setattr(SC, "_process_one_problem", fake_one)
    base.set_run_knobs(max_model_len=131072)
    r = SC.run_scicode("m", "http://h/v1", limit=1, concurrency=1, enable_thinking=True)
    assert r["replies_cut_off_by_budget"] == 1 and r["replies_cut_off_in_reasoning"] == 1
    assert ("1 sub-step repl(ies) reached the end of the served 131072-token context window"
            in r["leaderboard_comparable_reason"])


# --------------------------------------------------------------------------- #
# livebench                                                                    #
# --------------------------------------------------------------------------- #
# GEMMA4_31B (LiveBench's config for gemma-4-31b-it) is defined at the top of this file.
_OK_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
           "total_output_tokens": 10}


def _livebench(monkeypatch, tmp_path, config=None, commit=LB.LIVEBENCH_COMMIT, extra=None,
               answer_rows=(_OK_ROW,), **kwargs):
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "2025-04-02")
    monkeypatch.delenv("GBENCH_LIVEBENCH_CATEGORIES", raising=False)
    if extra is None:
        monkeypatch.delenv("GBENCH_LIVEBENCH_EXTRA_ARGS", raising=False)
    else:
        monkeypatch.setenv("GBENCH_LIVEBENCH_EXTRA_ARGS", extra)
    monkeypatch.delenv("GBENCH_LIVEBENCH_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_MODEL", raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config", lambda image, names: {
        "name": names[-1] if config else None, "config": config, "commit": commit})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    seen = {}

    def fake_run(cmd, **k):
        seen["inner"] = cmd[-1]
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\ngoogle-gemma-4-31b-it,50.0,"
            + ",".join("50.0" for _ in cols) + "\n", encoding="utf-8")
        answers = tmp_path / "model_answer" / "data" / "live_bench" / "math" / "x" / "model_answer"
        answers.mkdir(parents=True, exist_ok=True)
        (answers / "google-gemma-4-31b-it.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in answer_rows), encoding="utf-8")
        if answer_rows:   # instruction_following is scored: its answer rows are copied out too
            if_dir = (tmp_path / "model_answer" / "data" / "live_bench" / "instruction_following"
                      / "paraphrase" / "model_answer")
            if_dir.mkdir(parents=True, exist_ok=True)
            (if_dir / "google-gemma-4-31b-it.jsonl").write_text(json.dumps(_OK_ROW) + "\n",
                                                                encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    r = LB.run_livebench("google/gemma-4-31B-it", "http://h/v1", 4, **kwargs)
    cfg = json.load(open(tmp_path / LB._MODEL_CONFIG_FILE))
    return r, seen["inner"], cfg


def test_livebench_config_cap_is_fitted_to_the_served_window_and_named(monkeypatch, tmp_path):
    # The review's probe: gemma-4-31b-it's 65536 on a 32768-token window, --thinking.
    base.set_run_knobs(max_output_tokens=16384, max_model_len=32768)
    r, inner, cfg = _livebench(monkeypatch, tmp_path, config=GEMMA4_31B, enable_thinking=True,
                               max_output_tokens=16384)
    fitted = 32768 - LB._PROMPT_RESERVE_TOKENS
    assert cfg["api_kwargs"]["default"]["max_tokens"] == fitted
    assert "--max-tokens" not in inner
    assert r["max_output_tokens"] == fitted
    assert r["max_output_tokens_source"] == "official:fit-to-window"
    assert f"lowered to {fitted} to fit the served context window (32768 tokens" in \
        r["max_output_tokens_note"]
    rec = r["official_sampling"]
    assert rec["max_tokens"] == 65536 and rec["max_tokens_sent"] == fitted
    assert rec["max_tokens_applied"] is False
    assert r["leaderboard_comparable"] is False
    assert ("LiveBench's config for gemma-4-31b-it sends 65536, which does not fit the served "
            "32768-token context window with room for the prompt (serve with --max-model-len "
            ">= 73728)") in r["leaderboard_comparable_reason"]


def test_livebench_default_cap_that_does_not_fit_a_small_window_is_named(monkeypatch, tmp_path):
    base.set_run_knobs(max_model_len=6000)
    r, inner, _ = _livebench(monkeypatch, tmp_path)
    assert "--max-tokens 3000 " in inner
    assert r["max_output_tokens_source"] == "official:fit-to-window"
    assert "gen_api_answer.py's --max-tokens default sends 4096" in \
        r["leaderboard_comparable_reason"]


def test_livebench_explicit_budget_is_fitted_but_stays_comparable(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32000, max_model_len=32768)
    r, inner, _ = _livebench(monkeypatch, tmp_path, enable_thinking=True, max_output_tokens=32000)
    assert "--max-tokens 24576 " in inner
    assert r["max_output_tokens_source"] == "cli:--max-output-tokens:fit-to-window"
    assert r["official_sampling"]["max_tokens_sent"] == 24576
    # the fitted budget is no reason; the thinking run's temperature 1.0 is the only one
    assert r["leaderboard_comparable"] is False
    assert r["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking; sent as the config's temperature) (LiveBench samples at "
        "temperature 0, or the question's required temperature)")


def test_livebench_extra_args_budget_is_rewritten_to_fit(monkeypatch, tmp_path):
    base.set_run_knobs(max_model_len=32768)
    r, inner, _ = _livebench(monkeypatch, tmp_path, extra="--question-end 2 --max-tokens 40000")
    assert inner.count("--max-tokens") == 1 and "--max-tokens 24576" in inner
    assert r["max_output_tokens"] == 24576
    assert r["max_output_tokens_source"] == "env:GBENCH_LIVEBENCH_EXTRA_ARGS:fit-to-window"
    assert ("GBENCH_LIVEBENCH_EXTRA_ARGS asked for --max-tokens 40000 instead; lowered to 24576 "
            "to fit the served context window") in r["max_output_tokens_note"]
    assert "was sent instead" not in r["max_output_tokens_note"]
    assert "max_tokens=40000 from GBENCH_LIVEBENCH_EXTRA_ARGS" in r["leaderboard_comparable_reason"]


def test_livebench_fit_never_raises_a_budget_and_needs_a_window():
    base.set_run_knobs(max_model_len=12000)
    assert LB.fit_to_window(5000) == (5000, None)            # fits: 12000 // 2 = 6000 of room
    assert LB.fit_to_window(7000) == (6000, 12000)
    base._RUN_KNOBS["max_model_len"] = None
    assert LB.fit_to_window(10 ** 6) == (10 ** 6, None)


def test_livebench_unreadable_image_commit_is_not_comparable(monkeypatch, tmp_path):
    r, _, _ = _livebench(monkeypatch, tmp_path, commit=None)
    assert r["livebench_image_commit"] is None
    assert r["leaderboard_comparable"] is False
    assert "LiveBench commit could not be read" in r["leaderboard_comparable_reason"]
    assert ".git directory in the build context" in r["leaderboard_comparable_reason"]


def test_livebench_counts_replies_cut_off_by_the_runs_budget(monkeypatch, tmp_path):
    exhausted = {"question_id": "q1", "choices": [{"index": 0, "turns": [""]}],
                 "total_output_tokens": 32768, "eval_status": "token_exhaustion"}
    base.set_run_knobs(max_output_tokens=32768)
    r, inner, _ = _livebench(monkeypatch, tmp_path, enable_thinking=True, max_output_tokens=32768,
                             answer_rows=(_OK_ROW, exhausted))
    assert "mkdir -p /out/model_answer && find data -path '*/model_answer/*'" in inner
    assert "-exec cp --parents {} /out/model_answer/" in inner
    assert "-name google-gemma-4-31b-it.jsonl" in inner
    assert r["replies_cut_off_by_budget"] == 1 and r["replies_cut_off_in_reasoning"] == 1
    assert "reasoning_tokens" in r["replies_cut_off_note"]
    assert ("1 repl(ies) cut off by this run's max_tokens 32768, not by an official cap (1 of them "
            "inside the reasoning channel") in r["leaderboard_comparable_reason"]


def test_livebench_reply_at_the_official_cap_is_not_a_cutoff(monkeypatch, tmp_path):
    at_cap = {"question_id": "q1", "choices": [{"index": 0, "turns": ["long"]}],
              "total_output_tokens": 4096}
    r, _, _ = _livebench(monkeypatch, tmp_path, answer_rows=(at_cap,))
    assert r["max_output_tokens"] == 4096
    assert r["replies_cut_off_by_budget"] == 0 and "replies_cut_off_note" not in r
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")
    # the same reply at a max_tokens lowered to fit a small window is a cut-off
    base.set_run_knobs(max_model_len=6000)
    lowered = dict(at_cap, total_output_tokens=3000)
    r, _, _ = _livebench(monkeypatch, tmp_path, answer_rows=(lowered,))
    assert r["replies_cut_off_by_budget"] == 1 and r["replies_cut_off_in_reasoning"] == 0


def test_livebench_answer_rows_that_cannot_be_read_are_named(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    r, _, _ = _livebench(monkeypatch, tmp_path, enable_thinking=True, max_output_tokens=32768,
                         answer_rows=())
    assert "LiveBench's answer rows were not found" in r["leaderboard_comparable_reason"]


def test_livebench_answer_cutoffs_reads_the_rows(tmp_path):
    d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "math" / "t" / "model_answer"
    d.mkdir(parents=True)
    rows = [{"choices": [{"turns": ["a"]}], "total_output_tokens": 100},
            {"choices": [{"turns": ["a"]}], "total_output_tokens": 4096},
            {"choices": [{"turns": [""]}], "total_output_tokens": 4096,
             "eval_status": "token_exhaustion"}]
    (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")
    assert LB.answer_cutoffs(str(tmp_path), 4096, False) == (2, 1, 3)
    # the rule is the same with --thinking: a total exactly at max_tokens is a cut-off (vLLM's
    # completion_tokens never exceed max_tokens, and the reasoning tokens LiveBench adds are >= 0)
    assert LB.answer_cutoffs(str(tmp_path), 4096, True) == (2, 1, 3)


# --------------------------------------------------------------------------- #
# hmmt and aime: MathArena's own config for the served model                   #
# --------------------------------------------------------------------------- #
def _matharena_setup(monkeypatch, mod, generation_config=None):
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MA, "self_check", lambda config: None)
    monkeypatch.delenv("GBENCH_HMMT_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_AIME_TEMPERATURE", raising=False)
    if generation_config is None:
        monkeypatch.setattr(OJ, "served_model_recommended_sampling", lambda model_name=None: (
            pytest.fail("MathArena's config decides; generation_config.json is not read")))
    else:
        monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                            lambda model_name=None: (dict(generation_config), "gc/stub"))
    loader = "_load_hmmt_samples" if mod is H else "_load_aime_samples"
    monkeypatch.setattr(mod, loader, lambda *a, **k: [
        ([{"role": "user", "content": "1+1?"}], "2", {"category": "x", "problem_idx": 1})])

    def fake_scorer(comp, config, stats, **kw):
        stats.update(MA._new_stats())

        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = True, "OK"
        return _score

    monkeypatch.setattr(mod.MA, "make_scorer", fake_scorer)
    return H.run_hmmt if mod is H else AI.run_aime


QWEN35_27B = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "top_k": 20,
              "min_p": 0.0, "repetition_penalty": 1.0}


@pytest.mark.parametrize("mod", [H, AI], ids=["hmmt", "aime"])
@pytest.mark.parametrize("thinking,knob,sent", [(False, 65536, 192000), (True, 32768, 192000),
                                                (True, 262144, 262144)])
def test_matharena_reproduces_its_config_for_a_model_it_ships_one_for(monkeypatch, endpoint, mod,
                                                                      thinking, knob, sent):
    run = _matharena_setup(monkeypatch, mod)
    base.set_run_knobs(max_output_tokens=knob, tokenizer="Qwen/Qwen3.5-27B")
    endpoint.state["reply"] = lambda body: ("\\boxed{2}", None, "stop")
    r = run("m", endpoint.url, 2, enable_thinking=thinking)
    assert len(endpoint.records) == 4
    # the temperature is gbench's (1.0 with thinking, 0.0 without); the rest is the config's
    want = dict(QWEN35_27B, temperature=1.0 if thinking else 0.0)
    for body in endpoint.records:
        assert {k: body[k] for k in QWEN35_27B} == want
        assert body["max_tokens"] == sent
        assert "reasoning_effort" not in body
    cfg = r["matharena_model_config"]
    assert cfg["path"] == "configs/models/qwen/qwen3.5_27b.yaml"
    assert cfg["huggingface_id"] == "Qwen/Qwen3.5-27B" and cfg["matched_name"] == "Qwen/Qwen3.5-27B"
    assert cfg["not_reproduced"] == [] and cfg["not_applicable"] == []
    assert r["recommended_sampling"]["source"].endswith("configs/models/qwen/qwen3.5_27b.yaml")
    assert r["recommended_sampling"]["max_tokens"] == 192000
    assert r["official_sampling"]["max_tokens"] == 192000
    blob = H.MATHARENA_MODEL_CONFIGS["qwen/qwen3.5_27b"][0]
    assert f"qwen3.5_27b.yaml (blob {blob[:12]})" in r["official_sampling"]["reference"]
    if thinking:
        assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")
    else:
        assert r["leaderboard_comparable_reason"].startswith("temperature=0.0 (default); ")


@pytest.mark.parametrize("mod", [H, AI], ids=["hmmt", "aime"])
def test_matharena_config_cap_clamped_by_the_window_is_named(monkeypatch, endpoint, mod):
    run = _matharena_setup(monkeypatch, mod)
    base.set_run_knobs(max_output_tokens=65536, tokenizer="Qwen/Qwen3.5-27B",
                       max_model_len=131072)
    endpoint.state["reply"] = lambda body: ("\\boxed{2}", None, "stop")
    r = run("m", endpoint.url, 2)
    assert max(b["max_tokens"] for b in endpoint.records) < 192000
    assert "MathArena's config configs/models/qwen/qwen3.5_27b.yaml sends 192000" in \
        r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("mod", [H, AI], ids=["hmmt", "aime"])
def test_matharena_config_reasoning_switch_and_reasoning_effort(monkeypatch, endpoint, mod):
    run = _matharena_setup(monkeypatch, mod)
    endpoint.state["reply"] = lambda body: ("\\boxed{2}", None, "stop")
    base.set_run_knobs(max_output_tokens=32768, tokenizer="openai/gpt-oss-20b")
    r = run("m", endpoint.url, 2, enable_thinking=True)
    for body in endpoint.records:
        assert body["reasoning_effort"] == "high" and body["max_tokens"] == 128000
        # the config sends no temperature, a thinking run sends 1.0 (base.resolve_temperature)
        assert body["temperature"] == 1.0 and "top_p" not in body
    assert r["recommended_sampling"]["reasoning_effort"] == "high"
    assert r["leaderboard_comparable"] is False
    assert r["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking); MathArena samples each model at its provider-recommended "
        "settings, which set no temperature for the served model")
    # Qwen3.5-2B's config turns thinking on: a no-think run is not its MathArena row
    endpoint.records.clear()
    base.set_run_knobs(tokenizer="Qwen/Qwen3.5-2B")
    r = run("m", endpoint.url, 2, enable_thinking=False)
    assert "MathArena's config qwen/qwen3.5_2b sets enable_thinking=True; this run is no-think" \
        in r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("mod", [H, AI], ids=["hmmt", "aime"])
def test_matharena_gemma_keeps_generation_config_and_counts_cutoffs(monkeypatch, endpoint, mod):
    run = _matharena_setup(monkeypatch, mod, generation_config=OJ.GEMMA4_RECOMMENDED)
    base.set_run_knobs(max_output_tokens=32768, tokenizer="google/gemma-4-E4B-it")
    endpoint.state["reply"] = _cut_in_reasoning
    r = run("m", endpoint.url, 2, enable_thinking=True)
    for body in endpoint.records:
        assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 0.95, 64)
        assert body["max_tokens"] == 32768
    assert r["matharena_model_config"] is None
    assert r["recommended_sampling"] == {"source": "gc/stub", **OJ.GEMMA4_RECOMMENDED}
    assert r["replies_cut_off_by_budget"] == 4 and r["replies_cut_off_in_reasoning"] == 4
    assert ("4 repl(ies) (4 first, 0 last-chance) cut off by --max-output-tokens (MathArena "
            "ships no config for this model, so it defines no max_tokens for it; gbench sends "
            "--max-output-tokens, its house budget: raise the flag), 4 of them inside "
            "the reasoning channel") in r["leaderboard_comparable_reason"]


def test_matharena_counts_last_chance_replies_cut_off_too():
    sampling = OJ.recommended_official_sampling(dict(OJ.GEMMA4_RECOMMENDED), "gc")
    gemma = H.ServedModelSampling(sampling, {}, {"source": "gc"}, None, [])
    result = {"max_tokens_sent": [32768], "sample_traces": [
        {"finish_reason": "stop", "response_text": "no box", "max_tokens_sent": 32768,
         "matharena_last_chance": {"finish_reason": "length", "response_text": ""}},
        {"finish_reason": "stop", "response_text": "\\boxed{1}", "max_tokens_sent": 32768}]}
    reasons = H.served_model_reasons("hmmt", result, gemma, {}, True)
    assert result["replies_cut_off_by_budget"] == 1
    assert result["replies_cut_off_in_reasoning"] == 1
    assert any("(0 first, 1 last-chance)" in x for x in reasons)


def test_matharena_config_table_is_consistent():
    configs = H.MATHARENA_MODEL_CONFIGS
    assert len(configs) == 59
    for path, (blob, hf, fields) in configs.items():
        assert re.fullmatch(r"[0-9a-f]{40}", blob), path
        assert "/" in hf and "model" in fields and "api" in fields, path
        if fields["api"] == "vllm":
            assert "max_tokens" in fields, path      # SamplingParams would default to 16
        for thinking in (False, True):
            sampling, _, _, _ = H.translate_matharena_config(path, blob, fields, thinking)
            base.check_official_sampling_params("hmmt", sampling)
        # every config is found under its own huggingface_id, and ties send one request
        assert H.matharena_model_config(hf) is not None, hf
    assert H.matharena_model_config("moonshotai/Kimi-K2.5")[0] == "moonshot/k25"
    assert H.matharena_model_config("/models/Qwen3.5-27B")[0] == "qwen/qwen3.5_27b"
    for gemma in ("google/gemma-4-31B-it", "google/gemma-4-E4B-it", "gemma-4-26B-A4B-it"):
        assert H.matharena_model_config(gemma) is None
    assert MA.MATHARENA_COMMIT.startswith("b89f2f0a")


def test_matharena_config_translation_follows_apiclient():
    def tr(path, thinking=False):
        blob, _, fields = H.MATHARENA_MODEL_CONFIGS[path]
        return H.translate_matharena_config(path, blob, fields, thinking)

    s, extra, na, gaps = tr("mistral/leanstral_1_5")          # every field null: nothing sent
    assert s.send_temperature is False and s.send_max_tokens is False and not s.params
    s, extra, na, gaps = tr("llama4/maverick")                # OpenRouter routing: not applicable
    assert (s.temperature, s.params, s.max_tokens) == (0.6, {"top_p": 0.9}, 32000)
    assert na and "provider routing" in na[0] and not gaps
    s, extra, na, gaps = tr("nvidia/nemotron-3-super", True)  # offline vLLM path
    assert (s.temperature, s.params, s.max_tokens) == (1.0, {"top_p": 0.95}, 192000)
    assert na and "offline vLLM path" in na[0]
    s, extra, na, gaps = tr("glm/glm-53-flash", True)          # harness ignored; clear_thinking
    assert extra == {"reasoning_effort": "max"}
    assert any("allow_harness" in x for x in na)
    assert gaps and "clear_thinking" in gaps[0]
    s, extra, na, gaps = tr("glm/glm-52", False)               # GLM's thinking switch
    assert gaps == ["MathArena's config glm/glm-52 runs with thinking enabled "
                    "(extra_body.thinking); this run is no-think"]
    assert tr("glm/glm-52", True)[3] == []
    s = tr("qwen/qwen3.5_27b")[0]
    assert s.reasoning is not None and s.reasoning.max_tokens == 192000
    assert base.resolve_token_budget("hmmt", s, thinking=False).max_tokens == 192000
    assert base.resolve_token_budget("hmmt", s, thinking=True,
                                     max_output_tokens=262144).max_tokens == 262144


# --------------------------------------------------------------------------- #
# docs                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path,needles", [
    ("docs/evals/bigcodebench.md", ("replies_cut_off_by_budget", "replies_cut_off_in_reasoning")),
    ("docs/evals/livebench.md", ("fit-to-window", "cap + 8192", ".git")),
    ("docs/evals/hmmt.md", ("matharena_model_config", "MATHARENA_MODEL_CONFIGS",
                            "replies_cut_off_by_budget")),
    ("docs/evals/aime.md", ("matharena_model_config", "MATHARENA_MODEL_CONFIGS",
                            "replies_cut_off_by_budget")),
    ("docs/evals/lcb.md", ("replies_cut_off_by_budget", "cut_off_by_budget")),
    ("docs/evals/multipl_e.md", ("replies_cut_off_by_budget",)),
    ("docs/evals/ojbench.md", ("replies_cut_off_by_budget",)),
    ("docs/evals/scicode.md", ("replies_cut_off_by_budget",)),
    ("docker/livebench.Dockerfile", (".git",)),
])
def test_docs_state_the_rules(path, needles):
    text = (REPO / path).read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for needle in needles:
        assert needle in flat, (path, needle)
