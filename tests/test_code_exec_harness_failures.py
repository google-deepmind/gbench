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

"""Harness failures of the code-execution suites are never scored as the model's outcome.

* aider_polyglot: the benchmark container keeps the image's HOME (/root), where rustup installed
  Rust, as upstream's benchmark/docker.sh does; a test run that shows a toolchain in the image
  could not run is a harness error (`toolchain`). Only the test runs' outputs are read for
  toolchain failures, never the model's replies. A model request aider gave up on is a harness
  error (`model_endpoint`) only when no reply followed and the endpoint failed (connection, DNS,
  5xx, 429, or litellm.Timeout: gbench sizes the timeout for a reply to the end of the served
  window, so one that still times out means a slow endpoint); a request whose ninth attempt
  succeeded is scored, and one aider gave up on for any other cause is the model's outcome,
  scored and counted. GBENCH_AIDER_LANGUAGES is lowercased as benchmark.py lowercases --languages.
* livebench: requests that failed at the endpoint (api_error rows other than empty replies) are
  scored 0 inside LiveBench's averages, so the headline and the scores of their categories are
  withheld; requests the server aborted withhold every score.
* bigcodebench: the image's real output when the Hub is unreachable (datasets' "Couldn't reach
  '<repo>' on the Hub", with no resolver text) is named as `hub_unreachable`.
* omnidocbench: the evaluator, which compiles the model's LaTeX, runs with --network none.

Docker, the endpoint and the network are stubbed everywhere; nothing is contacted.
"""

import asyncio
import json
import subprocess
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import bigcodebench as BCB
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import omnidocbench as ODB
from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"


def _flat(text):
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_MODEL", "GBENCH_AIDER_EDIT_FORMAT",
                 "GBENCH_AIDER_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# aider_polyglot: histories as aider's io and benchmark.py write them
# --------------------------------------------------------------------------- #
#: The chat history's header and instructions (io.py: tool lines are "> ...  ", user lines
#: "#### ...  ").
HEADER = ("\n# aider chat started at 2026-09-27 00:00:00\n\n"
          "> Aider v0.86.3.dev+less  \n"
          "> Model: openai/google/gemma-4-26B-A4B-it with diff edit format  \n"
          "> Added src/lib.rs to the chat.  \n"
          "\n#### # Instructions  \n#### Implement the function.  \n")


def _reply(text):
    """io.ai_output: a blank line, the reply, a blank line."""
    return "\n" + text.strip() + "\n\n"


def _tool(*lines):
    return "".join(f"> {ln}  \n" for ln in lines)


def _tests(output):
    """benchmark.py run_unit_tests: the output as a bare fenced block, with no newline after."""
    return f"```\n{output}\n```"


def _retries(error, description=None, upto=8):
    """aider's send_message retry log: the error (and litellm's description) before each retry."""
    text = ""
    for n in range(upto):
        delay = 0.125 * 2 ** (n + 1)
        text += _tool(error) + (_tool(description) if description else "")
        text += _tool(f"Retrying in {delay:.1f} seconds...")
    return text


#: `cargo test` with HOME other than /root in the aider-benchmark:5dc9490b image (captured 2026-09-27:
#: docker run --network none -e HOME=/benchmarks aider-benchmark:5dc9490b ... cargo test).
RUSTUP_NO_DEFAULT = (
    "error: rustup could not choose a version of cargo to run, because one wasn't specified "
    "explicitly, and no default is configured.\n"
    "help: run 'rustup default stable' to download the latest stable release of Rust and set it "
    "as your default toolchain.")
CARGO_PASS = "running 10 tests\ntest result: ok. 10 passed; 0 failed; 0 ignored"
CARGO_FAIL = "running 10 tests\ntest result: FAILED. 7 passed; 3 failed; 0 ignored"
CONN = "litellm.APIConnectionError: APIConnectionError: OpenAIException - Connection error."
TIMEOUT = ("litellm.Timeout: Timeout Error: OpenAIException - Request timed out. "
           "Error_str: Request timed out.")
TIMEOUT_DESC = ("The API provider timed out without returning a response. They may be down or "
                "overloaded.")
TOKENS = "Tokens: 1.6k sent, 2.5k received."
#: a retried failure that is neither an endpoint failure nor a timeout
APIERR = "litellm.APIError: APIError: OpenAIException - Unexpected response body."


def test_the_container_keeps_the_images_home_as_upstream_does():
    cmd = A._build_command("aider-benchmark:5dc9490b", "/b", "gbench", "openai/m",
                           "http://host.docker.internal:8000/v1", 4, ["rust"], "", None, "",
                           settings_dir="/s")
    env = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-e"]
    assert env == ["AIDER_DOCKER=1", "AIDER_BENCHMARK_DIR=/benchmarks",
                   "OPENAI_API_BASE=http://host.docker.internal:8000/v1", "OPENAI_API_KEY=EMPTY"]
    assert not any(arg.startswith(("HOME=", "RUSTUP_HOME=", "CARGO_HOME=")) for arg in cmd)
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "bridge"


def test_a_toolchain_that_could_not_run_is_a_harness_error():
    history = (HEADER + _reply("src/lib.rs\n```rust\npub fn f() {}\n```")
               + _tool(TOKENS, "Applied edit to src/lib.rs") + _tests(RUSTUP_NO_DEFAULT))
    found = A.exercise_harness_error(history)
    assert found == {"kind": "toolchain", "detail": RUSTUP_NO_DEFAULT.splitlines()[0]}
    assert A.exercise_harness_error(_tests("error: no default toolchain configured"))["kind"] \
        == "toolchain"


def test_toolchain_text_the_model_wrote_is_not_read():
    # the model's reply quoting a toolchain or network failure, in prose and in a fence
    reply = _reply("You may see:\n```\n" + RUSTUP_NO_DEFAULT + "\n```\nand\n```\n"
                   "Downloading from https://github.com/foo/bar\nConnection refused\n```\n")
    history = (HEADER + reply + _tool(TOKENS, "Applied edit to src/lib.rs")
               + _tests(CARGO_FAIL))
    assert A.exercise_test_outputs(history) == [CARGO_FAIL]
    assert A.exercise_harness_error(history) is None
    # the instructions are not read either
    assert A.exercise_harness_error("#### Connection refused by github.com  \n"
                                    + _tool("Applied edit to x") + _tests(CARGO_FAIL)) is None


def test_a_toolchain_host_failure_in_a_later_test_run_still_counts():
    gradle = ("Downloading https://services.gradle.org/distributions/gradle-8.7-bin.zip\n"
              'Exception in thread "main" java.net.UnknownHostException: services.gradle.org')
    history = (HEADER + _reply("x") + _tool(TOKENS, "Applied edit to A.java") + _tests(CARGO_FAIL)
               + "\n#### fix it  \n" + _reply("y") + _tool(TOKENS, "Applied edit to A.java")
               + _tests(gradle))
    assert len(A.exercise_test_outputs(history)) == 2
    assert A.exercise_harness_error(history)["kind"] == "toolchain_network"


def test_a_request_whose_ninth_attempt_succeeded_is_scored():
    history = (HEADER + _retries(TIMEOUT, TIMEOUT_DESC) + _reply("src/lib.rs\n```rust\n```")
               + _tool(TOKENS, "Applied edit to src/lib.rs") + _tests(CARGO_PASS))
    assert A.request_give_ups(history) == []
    assert A.exercise_harness_error(history) is None
    history = (HEADER + _retries(CONN) + _reply("done") + _tool(TOKENS) + _tests(CARGO_PASS))
    assert A.request_give_ups(history) == [] and A.exercise_harness_error(history) is None
    # a reply with no text still logs aider's usage line
    history = HEADER + _retries(CONN) + _tool(TOKENS) + _tests(CARGO_FAIL)
    assert A.request_give_ups(history) == []


def test_a_request_the_endpoint_kept_failing_is_a_harness_error():
    # the ninth failure, as check_and_open_urls logs it after the last retry
    history = HEADER + _retries(CONN) + _tool(CONN) + _tests(CARGO_FAIL)
    (give_up,) = A.request_give_ups(history)
    assert give_up["cause"] == "APIConnectionError" and give_up["endpoint"] is True
    found = A.exercise_harness_error(history)
    assert found["kind"] == "model_endpoint"
    assert found["detail"] == f"{CONN} (Retrying in 32.0 seconds...)"
    for name in ("InternalServerError", "BadGatewayError", "ServiceUnavailableError",
                 "RateLimitError"):
        err = f"litellm.{name}: {name}: OpenAIException - boom"
        history = HEADER + _retries(err, "down") + _tool(err, "down") + _tests(CARGO_FAIL)
        assert A.exercise_harness_error(history)["kind"] == "model_endpoint", name
    # a failed lookup logged as the cause, whatever litellm called it
    dns = ("litellm.APIError: APIError: OpenAIException - [Errno -3] Temporary failure in name "
           "resolution")
    history = HEADER + _retries(dns) + _tool(dns) + _tests(CARGO_FAIL)
    assert A.exercise_harness_error(history)["kind"] == "model_endpoint"


def test_a_request_that_kept_timing_out_is_an_endpoint_failure():
    history = (HEADER + _retries(TIMEOUT, TIMEOUT_DESC) + _tool(TIMEOUT, TIMEOUT_DESC)
               + _tests(CARGO_FAIL))
    (give_up,) = A.request_give_ups(history)
    assert give_up["cause"] == "Timeout" and give_up["endpoint"] is True
    assert give_up["detail"].startswith("litellm.Timeout: Timeout Error")
    assert A.exercise_harness_error(history)["kind"] == "model_endpoint"


def test_a_request_given_up_for_another_cause_is_the_models_outcome():
    history = HEADER + _retries(APIERR) + _tool(APIERR) + _tests(CARGO_FAIL)
    (give_up,) = A.request_give_ups(history)
    assert give_up["cause"] == "APIError" and give_up["endpoint"] is False
    assert A.exercise_harness_error(history) is None


def _exercise(root, lang, name, outcomes, history):
    d = root / lang / "exercises" / "practice" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / ".aider.results.json").write_text(json.dumps(
        {"language": lang, "testcase": name, "tests_outcomes": outcomes}))
    (d / ".aider.chat.history.md").write_text(history)


def test_parse_results_scores_other_give_ups_and_leaves_harness_errors_out(tmp_path):
    ok = HEADER + _reply("x") + _tool(TOKENS, "Applied edit to src/lib.rs") + _tests(CARGO_PASS)
    ninth_ok = HEADER + _retries(CONN) + _reply("x") + _tool(TOKENS) + _tests(CARGO_PASS)
    timed_out = (HEADER + _retries(TIMEOUT, TIMEOUT_DESC) + _tool(TIMEOUT, TIMEOUT_DESC)
                 + _tests(CARGO_FAIL))
    endpoint = HEADER + _retries(CONN) + _tool(CONN) + _tests(CARGO_FAIL)
    gave_up = HEADER + _retries(APIERR) + _tool(APIERR) + _tests(CARGO_FAIL)
    rustup = HEADER + _reply("x") + _tool(TOKENS, "Applied edit") + _tests(RUSTUP_NO_DEFAULT)
    _exercise(tmp_path, "rust", "a", [True], ok)
    _exercise(tmp_path, "rust", "b", [True], ninth_ok)
    _exercise(tmp_path, "rust", "c", [False, False], timed_out)
    _exercise(tmp_path, "python", "d", [False, False], endpoint)
    _exercise(tmp_path, "rust", "e", [False, False], rustup)
    _exercise(tmp_path, "rust", "f", [False, False], gave_up)
    agg = A._parse_results(str(tmp_path))
    assert agg["n"] == 3 and agg["pass2"] == 2
    assert agg["per_lang"] == {"rust": {"total": 3, "pass1": 2, "pass2": 2}}
    assert sorted((h["exercise"], h["kind"]) for h in agg["harness_errors"]) == [
        ("python/exercises/practice/d", "model_endpoint"),
        ("rust/exercises/practice/c", "model_endpoint"),
        ("rust/exercises/practice/e", "toolchain")]
    assert [(g["exercise"], g["language"], g["cause"]) for g in agg["request_give_ups"]] == [
        ("rust/exercises/practice/f", "rust", "APIError")]


def test_a_run_reports_the_scored_give_ups(monkeypatch, tmp_path):
    langs = A._LANGUAGES
    gave_up = HEADER + _retries(APIERR) + _tool(APIERR) + _tests(CARGO_FAIL)
    ok = HEADER + _reply("x") + _tool(TOKENS, "Applied edit") + _tests(CARGO_PASS)

    def fake_run(cmd, **_kw):
        base = tmp_path / "2026-run1--gbench"
        for i in range(A._EXPECTED_FULL):
            _exercise(base, langs[i % len(langs)], f"x{i}", [False, False] if i < 3 else [True],
                      gave_up if i < 3 else ok)
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        # the container prints its OOM-kill count last (none here)
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n"
                                     + A._MEMORY_MARKER + "12884901888\n" + A._OOM_MARKER + "0\n")
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert r["status"] == "success" and r["harness_errors"] == 0
    assert r["total_questions"] == A._EXPECTED_FULL
    assert r["correct_answers"] == A._EXPECTED_FULL - 3
    assert r["scored_request_give_ups"] == 3
    assert {g["cause"] for g in r["scored_request_give_up_exercises"]} == {"APIError"}
    assert r["leaderboard_comparable"] is True


@pytest.mark.parametrize("value, langs, hosts", [
    ("Java,Rust", ["java", "rust"],
     ["services.gradle.org", "repo.maven.apache.org", "index.crates.io", "static.crates.io"]),
    (" RUST  rust ", ["rust"], ["index.crates.io", "static.crates.io"]),
    ("CPP,Go,Java,JavaScript,Python,Rust", A._LANGUAGES,
     ["services.gradle.org", "repo.maven.apache.org", "index.crates.io", "static.crates.io"]),
])
def test_languages_are_lowercased_as_benchmark_py_does(monkeypatch, value, langs, hosts):
    monkeypatch.setenv("GBENCH_AIDER_LANGUAGES", value)
    assert A._languages() == langs
    assert A.container_hosts() == hosts


def test_the_aider_guide_documents_home_and_the_new_accounting():
    flat = _flat((DOCS / "aider_polyglot.md").read_text(encoding="utf-8"))
    for text in ("stays the image's `/root`", "`RUSTUP_HOME`", "`toolchain`",
                 "`scored_request_give_ups`", "`scored_request_give_up_exercises`",
                 "`litellm.Timeout`", "`litellm.APIConnectionError`", "`litellm.RateLimitError`",
                 "Retrying in 32.0 seconds...", "any case"):
        assert text in flat, text
    assert "gives up once the next delay" not in flat
    assert "HOME=/benchmarks" not in A.__doc__


# --------------------------------------------------------------------------- #
# livebench: failed requests inside LiveBench's averages
# --------------------------------------------------------------------------- #
FAILED = {"question_id": "q", "eval_status": "api_error", "error": "APIConnectionError",
          "error_msg": "Connection error.", "choices": [{"turns": ["$ERROR$"]}]}
EMPTY = {"question_id": "q", "eval_status": "api_error", "error": "Exception",
         "error_msg": "No message returned from litellm (finish_reason=stop)",
         "choices": [{"turns": ["$ERROR$"]}], "total_output_tokens": 0}
OK = {"question_id": "q", "eval_status": "ok", "total_output_tokens": 10,
      "choices": [{"turns": ["42"]}]}


def _livebench(monkeypatch, tmp_path, rows_by_cat, aborted=(None, None)):
    for name in ("GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS",
                 "GBENCH_LIVEBENCH_RELEASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(LB, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None,
                                              "commit": LB.LIVEBENCH_COMMIT})
    counts = list(aborted)
    monkeypatch.setattr(LB, "server_aborted_requests", lambda url, ids: counts.pop(0))
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))

    def fake_run(cmd, **_kw):
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\nm,40.0," + ",".join("40.0" for _ in cols)
            + "\n", encoding="utf-8")
        for cat, rows in rows_by_cat.items():
            d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / cat / "t" / "model_answer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return LB.run_livebench("m", "http://127.0.0.1:8000/v1", 4, max_output_tokens=4096)


def test_failed_requests_are_counted_per_category(tmp_path):
    for cat, rows in {"math": [FAILED, OK, EMPTY], "coding": [FAILED, FAILED],
                      "language": [OK, EMPTY]}.items():
        d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / cat / "t" / "model_answer"
        d.mkdir(parents=True)
        (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert LB.failed_requests_by_category(str(tmp_path)) == {"coding": 2, "math": 1}
    assert LB.read_answer_rows(str(tmp_path), 4096)["failed"] == 3


def test_failed_requests_withhold_the_average_and_their_categories(monkeypatch, tmp_path):
    rows = {c: [OK] for c in LB._CATEGORIES}
    rows["math"] = [OK, FAILED]
    rows["coding"] = [FAILED, OK, EMPTY]
    r = _livebench(monkeypatch, tmp_path, rows)
    assert r["accuracy"] is None and r["status"] == "completed_with_errors"
    reason = r["accuracy_withheld_reason"]
    assert reason.startswith("2 request(s) failed at the served endpoint (coding 1, math 1)")
    assert "LiveBench scored them 0 inside its averages" in reason
    assert set(r["dimension_scores_withheld"]) == {"coding", "math"}
    assert set(r["dimension_scores"]) == set(LB._CATEGORIES) - {"coding", "math"}
    # the questions judged with a score (none in this fixture's output), not the categories
    assert r["total_questions"] == sum(1 for t in r["sample_traces"]
                                       if not t.get("scoring_excluded"))
    assert r["correct_answers"] is None
    assert r["failed_requests"] == 2
    assert r["failed_requests_by_category"] == {"coding": 1, "math": 1}
    assert r["empty_replies"] == 1 and r["leaderboard_comparable"] is False


def test_empty_replies_alone_keep_the_score(monkeypatch, tmp_path):
    rows = {c: [OK] for c in LB._CATEGORIES}
    rows["math"] = [OK, EMPTY]
    r = _livebench(monkeypatch, tmp_path, rows)
    assert r["accuracy"] == 40.0 and "accuracy_withheld_reason" not in r
    assert r["dimension_scores_withheld"] == {} and len(r["dimension_scores"]) == 6
    assert r["failed_requests"] == 0 and r["failed_requests_by_category"] == {}


def test_requests_the_server_aborted_withhold_every_score(monkeypatch, tmp_path):
    rows = {c: [OK] for c in LB._CATEGORIES}
    rows["math"] = [OK, EMPTY]
    r = _livebench(monkeypatch, tmp_path, rows, aborted=(5, 7))
    assert r["server_aborted_requests"] == 2
    assert r["accuracy"] is None and r["dimension_scores"] == {}
    assert set(r["dimension_scores_withheld"]) == set(LB._CATEGORIES)
    assert "2 request(s) were aborted or failed inside the served vLLM server" in \
        r["accuracy_withheld_reason"]


def test_the_livebench_guide_documents_the_withheld_scores():
    flat = _flat((DOCS / "livebench.md").read_text(encoding="utf-8"))
    for text in ("`accuracy_withheld_reason`", "`dimension_scores_withheld`",
                 "`failed_requests_by_category`", "`--retry-failures`"):
        assert text in flat, text


# --------------------------------------------------------------------------- #
# bigcodebench: the Hub unreachable, as the image reports it
# --------------------------------------------------------------------------- #
#: stderr of `bigcodebench.sanitize` in gbench-bigcodebench:09dd993f with --network none (captured
#: 2026-09-27), from the datasets frames on.
BCB_NO_NETWORK = (
    "Traceback (most recent call last):\n"
    '  File "/usr/local/lib/python3.10/site-packages/datasets/load.py", line 1871, in '
    "dataset_module_factory\n"
    "    raise e1 from None\n"
    '  File "/usr/local/lib/python3.10/site-packages/datasets/load.py", line 1805, in '
    "dataset_module_factory\n"
    "    raise ConnectionError(f\"Couldn't reach '{path}' on the Hub ({type(e).__name__})\")\n"
    "ConnectionError: Couldn't reach 'bigcode/bigcodebench' on the Hub (ConnectionError)\n")


@pytest.mark.parametrize("output, expected", [
    (BCB_NO_NETWORK, {"kind": "hub_unreachable",
                      "detail": "ConnectionError: Couldn't reach 'bigcode/bigcodebench' on the "
                                "Hub (ConnectionError)"}),
    ("huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate "
     "the file on the Hub and we cannot find the requested files in the local cache.",
     {"kind": "hub_unreachable",
      "detail": "huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying "
                "to locate the file on the Hub and we cannot find the requested files in the "
                "local cache."}),
    ("x\nhuggingface.co: Temporary failure in name resolution\n",
     {"kind": "dns", "detail": "huggingface.co: Temporary failure in name resolution"}),
    ("Traceback\nKeyError: 'task_id'\n", None),
])
def test_bigcodebench_names_why_its_harness_produced_nothing(output, expected):
    assert BCB._harness_network_failure(output) == expected


def test_a_split_whose_harness_could_not_reach_the_hub_is_named(monkeypatch):
    def fake_run(cmd, **_kw):
        if "bigcodebench.sanitize" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr=BCB_NO_NETWORK)
        raise AssertionError("evaluate must not run after a failed sanitize")
    monkeypatch.setattr(BCB.subprocess, "run", fake_run)
    traces = [{"extra_payload": {"task_id": "BigCodeBench/0", "split": "instruct"},
               "response_text": "x"}]
    metrics = {}
    asyncio.run(BCB._make_scorer("full", 1, metrics, "img")(traces))
    assert metrics["bigcodebench_harness_errors"]["instruct"]["kind"] == "hub_unreachable"
    assert ("could not reach the Hugging Face Hub (ConnectionError: Couldn't reach "
            "'bigcode/bigcodebench' on the Hub (ConnectionError))") in \
        metrics["bigcodebench_report"]["error"]
    assert traces[0]["scoring_excluded"] is True and traces[0]["status"] == "HARNESS_ERROR"


def test_the_bigcodebench_guide_names_the_hub_kind():
    flat = _flat((DOCS / "bigcodebench.md").read_text(encoding="utf-8"))
    assert "`hub_unreachable`" in flat
    assert "Couldn't reach 'bigcode/bigcodebench' on the Hub" in flat


# --------------------------------------------------------------------------- #
# omnidocbench: the evaluator compiles the model's LaTeX with no network
# --------------------------------------------------------------------------- #
def test_the_omnidocbench_evaluator_runs_with_no_network():
    cmd = ODB._build_command("gbench-omnidocbench:193627ae", "/w/gt.json", "/w/pred", "/w/out", 2)
    image_at = cmd.index("gbench-omnidocbench:193627ae")
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--network" in cmd[:image_at] and cmd[cmd.index("--network") + 1] == "none"


def test_the_omnidocbench_revision_check_runs_with_no_network(monkeypatch):
    seen = []

    def fake_run(argv, **_kw):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, ODB._EVALUATOR_REVISION + "\n", "")
    monkeypatch.setattr(ODB.subprocess, "run", fake_run)
    ODB._verify_evaluator_revision("gbench-omnidocbench:193627ae", "build")
    (argv,) = seen
    assert argv[argv.index("--network") + 1] == "none"
    assert argv.index("--network") < argv.index("gbench-omnidocbench:193627ae")


def test_the_omnidocbench_guide_says_it_runs_with_no_network():
    flat = _flat((DOCS / "omnidocbench.md").read_text(encoding="utf-8"))
    assert "`--network none`" in flat and "`pdflatex`" in flat
    assert "--network none" in ODB.__doc__


# --------------------------------------------------------------------------- #
# aider_polyglot: the image's toolchains are checked in the container first
# --------------------------------------------------------------------------- #
def test_the_toolchain_check_runs_before_anything_else():
    cmd = A._build_command("aider-benchmark:5dc9490b", "/b", "gbench", "openai/m", "http://h:8000/v1", 1,
                           A._LANGUAGES, "", None, "", settings_dir="/s")
    script = cmd[-1]
    assert script.startswith(A._TOOLCHAIN_CHECK.strip())
    assert script.index("/npm-install/node_modules/jest/package.json") \
        < script.index("check_settings.py") < script.index("./benchmark/benchmark.py")
    assert "29.*) ;;" in script and "cargo --version" in script and "exit 4" in script


def test_a_stale_image_is_infra_required_before_any_score(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    # what the check printed in an image whose unpinned jest resolved to 30.5.1 (2026-09-27)
    monkeypatch.setattr(A.subprocess, "run", lambda cmd, **_kw: types.SimpleNamespace(
        returncode=4, stderr="", stdout=A._TOOLCHAIN_MARKER + "jest=30.5.1\n"))
    with pytest.raises(RuntimeError) as ei:
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    msg = str(ei.value)
    assert "[INFRA REQUIRED] aider_polyglot" in msg and "jest=30.5.1" in msg
    assert "docker/aider_polyglot.Dockerfile" in msg
    assert A._toolchain_problem("x\n" + A._TOOLCHAIN_MARKER + "cargo: error: rustup could not "
                                "choose a version of cargo to run\n") \
        == "cargo: error: rustup could not choose a version of cargo to run"
    assert A._toolchain_problem("GBENCH_AIDER_SETTINGS {}\n") is None


def test_the_image_pins_the_exercises_jest():
    dockerfile = (REPO / "docker" / "aider_polyglot.Dockerfile").read_text(encoding="utf-8")
    assert "    jest@29.7.0 \\\n" in dockerfile
    assert "    jest \\\n" not in dockerfile
    exercises = REPO.parent / "gbench-prereqs" / "aider-bench" / "polyglot-benchmark"
    if exercises.is_dir():      # the pin is the exercises' own devDependency
        for pkg in (exercises / "javascript" / "exercises" / "practice").glob("*/package.json"):
            assert json.loads(pkg.read_text())["devDependencies"]["jest"] == "^29.7.0", pkg
