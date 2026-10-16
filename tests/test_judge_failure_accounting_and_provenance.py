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

"""Shared accounting, provenance and docs that follow the suites' latest behavior.

* base.run_eval_suite names a judge failure the suite leaves out of the score once, under the
  judge failures, and still names every other sample the suite leaves out.
* The provenance registry describes the suites' current harness-error handling (nestful's
  sandboxed Win Rate, aider_polyglot's memory limit, terminal_bench's phase-aware exception
  attribution, the judged suites' JUDGE_FAILED exclusion, wildclawbench's thinking-off judge
  calls, toolbench's failed model requests, gaia2's configs and stop handling).
* The toolchains guide lists only what each suite checks for on the host.

Nothing here touches the network, docker or Gemini.
"""

import re
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import aider_polyglot, base, gaia2, mcp_bench, nestful
from gbench.runners.eval_suites import toolbench, wildclawbench
from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _ok(text):
    return base.Reply(text=text, tool_calls=None, finish_reason="stop", reasoning=None,
                      error=None, completion_tokens=5, stop_reason=None)


def _run(n, marks):
    """run_eval_suite over n samples answered "A", with a batch scorer that applies `marks`."""
    async def _fake_send(**k):
        return _ok("A")

    async def scorer(traces):
        for t in traces:
            t["status"] = "OK"
            t["is_correct"] = True
            t.update(marks.get(t["sample_idx"], {}))

    samples = [([{"role": "user", "content": f"q{i}"}], "A", {}) for i in range(n)]
    with mock.patch.object(base, "_send_single_request", _fake_send):
        return base.run_eval_suite(eval_name="unit_w2r", model_name="m", base_url="http://x",
                                   concurrency=2, samples=samples, max_output_tokens=64,
                                   async_eval_fn=scorer)


# --------------------------------------------------------------------------- #
# base: a judge failure the suite leaves out is named once
# --------------------------------------------------------------------------- #
def test_an_excluded_judge_failure_is_named_once():
    res = _run(2, {1: {"judge_grade": "JUDGE_FAILED", "scoring_excluded": True,
                       "is_correct": False}})
    assert res["accuracy"] == 100.0 and res["status"] == "completed_with_errors"
    why = res["status_reason"]
    assert "failed_requests 1 (judge failures 1, of which 1 left out of the score)" in why
    assert "scoring_excluded" not in why
    assert res["scoring_excluded"] == 1 and res["judge_failures"] == 1


def test_a_judge_failure_scored_as_the_suite_scores_it_is_not_called_left_out():
    res = _run(2, {1: {"judge_grade": "unparsed", "is_correct": False}})
    assert res["accuracy"] == 50.0
    why = res["status_reason"]
    assert "judge failures 1" in why and "left out of the score" not in why


def test_other_suite_exclusions_are_still_named_beside_a_judge_failure():
    res = _run(3, {1: {"judge_grade": "JUDGE_FAILED", "scoring_excluded": True},
                   2: {"scoring_excluded": True}})
    why = res["status_reason"]
    assert "judge failures 1, of which 1 left out of the score" in why
    assert "scoring_excluded 1 (left out of the score by the suite)" in why
    assert res["scoring_excluded"] == 2 and res["accuracy"] == 100.0


def test_a_run_of_only_excluded_judge_failures_withholds_its_score():
    res = _run(2, {i: {"judge_grade": "JUDGE_FAILED", "scoring_excluded": True}
                   for i in range(2)})
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["accuracy_withheld_reason"] == (
        "nothing was scored: failed_requests 2 (judge failures 2, of which 2 left out of the "
        "score)")


def test_the_env_knobs_guide_says_how_a_judge_failure_is_named():
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert ("is named once, under `judge failures N, of which M left out of the score`"
            in knobs)


# --------------------------------------------------------------------------- #
# provenance follows the suites' current behavior
# --------------------------------------------------------------------------- #
def test_the_provenance_describes_the_nestful_sandbox_and_harness_errors():
    m = CS["nestful"]["method"]
    assert CS["nestful"]["synced"] >= "2026-09-27"
    assert "inside a bubblewrap container of its own whatever GBENCH_SANDBOX says" in m
    assert "environment PATH and HOME=/tmp only" in m
    assert "no working bubblewrap -> the Win Rate alone infra_required" in m
    assert "requests with no reply are harness errors left out of every metric" in m
    assert nestful._SANDBOX_ENV == {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"}


def test_the_provenance_describes_aider_polyglots_memory_limit():
    m = CS["aider_polyglot"]["method"]
    assert "benchmark/docker.sh --memory=12g --memory-swap=12g" in m
    assert aider_polyglot.CONTAINER_MEMORY_ARGS == ("--memory", "12g", "--memory-swap", "12g")
    # the harness-error sentence reads as one list again, the model's give-ups after it
    assert ("GBENCH_AIDER_LANGUAGES is lowercased as benchmark.py does: left out"
            not in m)
    assert m.index("left out of every figure") < m.index("a request aider gave up on")


def test_the_provenance_describes_the_judged_suites_and_terminal_bench():
    assert "JUDGE_FAILED, a grading failure left out of the accuracy (scoring_excluded)" in \
        CS["putnam"]["method"]
    assert "is JUDGE_FAILED, left out of the accuracy (scoring_excluded)" in \
        CS["putnam_formal"]["method"]
    assert "then JUDGE_FAILED, a grading failure left out of the accuracy (scoring_excluded)" in \
        CS["imo_answer_bench"]["method"]
    assert "excluded_by_protocol, metrics.py's own exclusion" in CS["cimemories"]["method"]
    tb = CS["terminal_bench"]["method"]
    assert "terminal_bench.trial_exception_verdict" in tb
    assert "exceptions_kept_as_model_outcome" in tb


def test_the_provenance_describes_the_container_suites_labels_and_changes():
    wc = CS["wildclawbench"]["method"]
    assert "Gemini thinking level low" in wc and "judge_thinking_off" in wc
    assert (f"{wildclawbench.HARNESS_ERRORS_LABEL}={wildclawbench.HARNESS_ERRORS_VERSION};"
            in wc)
    mb = CS["mcp_bench"]["method"]
    assert f"{mcp_bench.HARNESS_LABEL}={mcp_bench.HARNESS_VERSION}, else infra_required" in mb
    assert "a credentialed-server session that failed outside the model;" in mb
    tlb = CS["toolbench"]["method"]
    assert "model_request_failures.jsonl" in tlb
    assert f"{toolbench.HARNESS_LABEL}={toolbench.HARNESS_VERSION}, else infra_required" in tlb
    g2 = CS["gaia2"]["method"]
    assert f"{gaia2.HARNESS_ERRORS_LABEL}={gaia2.HARNESS_ERRORS_VERSION} is infra_required" in g2
    assert "stop_sent_to_server" in g2 and "failed_configs" in g2
    assert "missing_capabilities" in g2


def test_the_provenance_names_no_local_checkout():
    for suite, entry in CS.items():
        assert "my-vllm" not in entry.get("method", ""), suite


# --------------------------------------------------------------------------- #
# toolchains lists what each suite checks for
# --------------------------------------------------------------------------- #
def test_toolchains_lists_the_aider_and_ojbench_images_not_host_toolchains():
    doc = _read("docs/evals/toolchains.md")
    for gone in ("apt install golang-go", "apt install rustc cargo", "apt install nodejs npm",
                 "apt install openjdk-21-jdk", "| `pypy3` |", "apt install cmake build-essential",
                 "pip install -e .` + DMOJ", "sudo apt-get install -y golang-go"):
        assert gone not in doc, gone
    rows = [l for l in doc.splitlines() if l.startswith("| ")]
    aider = next(l for l in rows if l.startswith("| `docker build -t aider-benchmark:5dc9490b"))
    assert "GBENCH_AIDER_BENCHMARK_DIR" in aider and "--memory 12g --memory-swap 12g" in aider
    assert aider_polyglot._IMAGE_DEFAULT == "aider-benchmark:5dc9490b"
    oj = next(l for l in rows if l.startswith("| `docker build -t gbench-ojbench:5e94480b"))
    assert "GBENCH_OJBENCH_TESTDATA" in oj and "PyPy3 and g++ in the image" in oj
    assert ("`aider_polyglot`, `multipl_e` and `ojbench` need none of the language toolchains "
            "on the host") in _flat(doc)


def test_toolchains_readme_and_env_knobs_name_the_nestful_sandbox():
    doc = _read("docs/evals/toolchains.md")
    bwrap_row = next(l for l in doc.splitlines() if l.startswith("| `bubblewrap` (`bwrap`)"))
    assert "`nestful`, whose Win Rate runs the model's calls" in bwrap_row
    sandbox = _flat(doc.split("## Sandboxing", 1)[1].split("\n\n")[1])
    assert "`nestful`'s Win Rate worker also always runs in a bubblewrap container" in sandbox
    assert "Without a working bubblewrap the Win Rate alone is `infra_required`" in sandbox
    readme_row = next(l for l in _read("docs/environment-variables.md").splitlines()
                      if l.startswith("| `GBENCH_SANDBOX`"))
    assert "`nestful`'s Win Rate worker also always runs in a bubblewrap container" in readme_row
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "acebench's evaluator and nestful's Win Rate worker run in containers of their own" \
        in knobs
    assert ("the Win Rate runs the model's calls in a bubblewrap sandbox of its own whatever "
            "GBENCH_SANDBOX says") in knobs


@pytest.mark.parametrize("rel", ["docs/evals/toolchains.md", "docs/evals/env-knobs.md"])
def test_shared_docs_stay_ascii(rel):
    text = _read(rel)
    assert all(ord(ch) < 128 for ch in text), rel


# --------------------------------------------------------------------------- #
# the run summary's Effective N is the headline's denominator
# --------------------------------------------------------------------------- #
def _summary_rows(tmp_path, result):
    import csv
    from gbench.cli import _save_eval_summary_csv
    path = _save_eval_summary_csv(tmp_path, [result], [], [], {})
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_effective_n_leaves_out_an_endpoint_failure(tmp_path):
    from gbench.cli import _effective_n
    res = _run(3, {})
    assert _effective_n(res, res["total_questions"]) == 3
    traces = [{"sample_idx": 0, "is_correct": True}, {"sample_idx": 1, "is_correct": False},
              {"sample_idx": 2, "scoring_excluded": True,
               "harness_error": {"kind": "endpoint_failure", "detail": "HTTP 500"}},
              {"sample_idx": 3, "judge_grade": "JUDGE_OUTAGE"}]
    result = {"eval_name": "unit_w2r", "model_short": "m", "format": "vllm",
              "total_questions": 4, "correct_answers": 1, "accuracy": 50.0,
              "status": "completed_with_errors", "sample_traces": traces,
              "category_accuracy": {}}
    rows = _summary_rows(tmp_path, result)
    assert rows[0]["Questions"] == "4" and rows[0]["Effective N"] == "2"


def test_effective_n_prefers_the_suites_own_and_falls_back_to_the_questions():
    from gbench.cli import _effective_n
    assert _effective_n({"effective_n": 7, "sample_traces": [{}] * 9}, 9) == 7
    # traces that are not one per question (e.g. per trial) say nothing about the denominator
    assert _effective_n({"sample_traces": [{"scoring_excluded": True}] * 3}, 9) == 9
    assert _effective_n({}, 5) == 5
