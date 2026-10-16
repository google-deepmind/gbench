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
"""Regression locks for the heavy-suite 'fake 0%%' artifacts.

A judge/harness failure must never masquerade as a clean 0%% model score:
* ojbench - an UNGRADED submission (judge returned no `is_passed`) must not be scored False; a
  wholly-ungraded run must hard-error, a partial one must be scoring_excluded.
* gaia2   - an agent-less `are-benchmark run` (model never invoked) must hard-error, not publish
  the real-but-meaningless macro=0.0 it returns.
* swebench - `error_ids` must reach _classify_swebench_errors so the patch-apply-vs-infra split
  (and the all-infra-error guard) actually works.
"""

import asyncio
import inspect

import pytest


# --------------------------------------------------------------------------- #
# ojbench: an ungraded submission is a harness failure, not a model 0          #
# --------------------------------------------------------------------------- #
import gbench.runners.eval_suites.ojbench as OJ


def _trace(id_, lang, resp="print(1)"):
    return {"extra_payload": {"row": {"id": id_, "language": lang}}, "response_text": resp}


def test_ojbench_all_ungraded_hard_errors(monkeypatch):
    # The judge returns the submissions WITHOUT an is_passed verdict (the wrong-DMOJ symptom:
    # results.jsonl byte-identical to input). This must raise, never report a clean 0%.
    monkeypatch.setattr(OJ, "_judge_in_container",
                        lambda records, n: [{"id": r["id"], "language": r["language"]} for r in records])
    score = OJ._make_scorer(4)
    traces = [_trace(1, "py"), _trace(2, "cpp")]
    with pytest.raises(RuntimeError, match="no graded verdicts|never a 0|not a 0"):
        asyncio.run(score(traces))


def test_ojbench_partial_ungraded_is_excluded_not_wrong(monkeypatch):
    # id 1 graded (failed), id 2 came back with no verdict -> id 2 must be scoring_excluded, not False.
    def fake_judge(records, n):
        return [{"id": 1, "language": "py", "is_passed": False},
                {"id": 2, "language": "cpp"}]  # no is_passed for id 2
    monkeypatch.setattr(OJ, "_judge_in_container", fake_judge)
    traces = [_trace(1, "py"), _trace(2, "cpp")]
    asyncio.run(OJ._make_scorer(4)(traces))
    assert traces[0]["is_correct"] is False and not traces[0].get("scoring_excluded")
    assert traces[1].get("scoring_excluded") is True     # excluded, not a model 0


def test_ojbench_graded_pass_is_scored(monkeypatch):
    monkeypatch.setattr(OJ, "_judge_in_container",
                        lambda records, n: [{"id": 1, "language": "py", "is_passed": True}])
    traces = [_trace(1, "py")]
    asyncio.run(OJ._make_scorer(4)(traces))
    assert traces[0]["is_correct"] is True
    assert not traces[0].get("scoring_excluded")


# --------------------------------------------------------------------------- #
# gaia2: an agent-less no-op must hard-error, not publish macro=0.0            #
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites.gaia2 import _agent_never_invoked


def test_gaia2_agent_never_invoked_flags_the_noop():
    # are-benchmark ran agent-less: validated runs exist but the container saw no agent LLM usage.
    assert _agent_never_invoked({"validated_runs": 15, "agent_invoked": False}) is True


def test_gaia2_real_run_is_not_flagged():
    assert _agent_never_invoked({"validated_runs": 15, "agent_invoked": True}) is False


def test_gaia2_older_summary_without_key_never_false_fires():
    # A summary from before the guard omits agent_invoked -> must stay silent, not hard-error.
    assert _agent_never_invoked({"validated_runs": 15}) is False


def test_gaia2_no_validated_runs_not_flagged():
    assert _agent_never_invoked({"validated_runs": 0, "agent_invoked": False}) is False


# --------------------------------------------------------------------------- #
# swebench: error_ids must be propagated into the report the classifier reads  #
# --------------------------------------------------------------------------- #
import gbench.runners.eval_suites.swebench_common as SWE


def test_swebench_report_propagates_error_ids():
    # The scorer copies a subset of the harness report into metrics["swebench_report"]; error_ids
    # MUST be in that subset or _classify_swebench_errors always sees [] and the infra guard is dead
    # code (an all-infra-errored run would publish a fake "success, 0%").
    src = inspect.getsource(SWE.make_swebench_scorer)
    assert '"error_ids"' in src, "make_swebench_scorer must carry error_ids into swebench_report"
