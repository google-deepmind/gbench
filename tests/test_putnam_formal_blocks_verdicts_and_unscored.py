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

"""putnam_formal: a code line with the closing fence glued to it closes its block, and a later
inline block wins over an earlier fenced one; the judge's verdict line is read through HTML and
LaTeX wrappers while a restatement of the format is no verdict (putnam reads it the same way);
a problem whose every attempt was a harness error or a judge outage is unscored, not unsolved.
Pure functions; no Lean, no judge."""

from gbench.runners.eval_suites import putnam_formal as PF


def test_a_glued_closing_fence_closes_the_block():
    reply = "Proof:\n```lean4\ntheorem t : True := by\n  trivial```\nDone."
    assert PF.lean_blocks(reply) == ["theorem t : True := by\n  trivial"]


def test_a_later_inline_block_wins_over_an_earlier_fenced_one():
    reply = "```lean\nLEMMA\n```\nActually, use this: ```lean FINAL```"
    assert PF.lean_blocks(reply)[-1].strip() == "FINAL"
    assert PF.lean_blocks(reply)[0] == "LEMMA"


def test_an_abandoned_block_and_an_unclosed_block():
    assert PF.lean_blocks("```lean\nOLD\n```lean\nNEW\n```") == ["NEW"]
    assert PF.lean_blocks("```lean\nnever closed") == []


def test_wrapped_verdicts_are_read_and_format_echoes_are_not():
    for text, want in (("Grade: CORRECT", True), ("**Grade:** INCORRECT", False),
                       ("<b>Grade:</b> CORRECT", True),
                       ("Grade: \\boxed{\\text{CORRECT}}", True),
                       ("Grade: \\textsc{Incorrect}", False),
                       ("Answer with Grade: CORRECT / INCORRECT.\nGrade: INCORRECT", False),
                       ("Grade: CORRECT or INCORRECT", None),
                       ("Grade: CORRECT | INCORRECT", None),
                       ("Grade: CORRECT/INCORRECT", None),
                       ("no verdict here", None)):
        assert PF.grade_line_verdict(text) is want, text


def _t(setting, idx, ok, **kw):
    return {"category": setting, "source_sample_idx": idx, "is_correct": ok, **kw}


def test_a_problem_with_no_scored_attempt_is_unscored_not_unsolved():
    traces = [_t("lean_with_answer", 1, True), _t("lean_with_answer", 1, False),
              _t("lean_with_answer", 2, False),
              _t("lean_with_answer", 3, False, scoring_excluded=True),
              _t("lean_with_answer", 3, False, judge_grade="JUDGE_OUTAGE")]
    entry = PF._settings_report(traces)["lean_with_answer"]
    assert entry["problems"] == 2 and entry["problems_unscored"] == 1
    assert entry["solved"] == 1 and entry["solved_pct"] == 50.0
    only_errors = [_t("lean_no_answer", 9, False, scoring_excluded=True)]
    entry = PF._settings_report(only_errors)["lean_no_answer"]
    assert entry["problems"] == 0 and entry["problems_unscored"] == 1
    assert entry["solved_pct"] is None and entry["accuracy"] is None


def test_a_full_run_with_an_unscored_problem_is_not_a_subset(monkeypatch):
    monkeypatch.setattr(PF, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(PF, "_load_putnam_formal_samples", lambda limit=None: [])
    monkeypatch.setattr(PF, "_lean_image_provenance", lambda: {"reproducible": True})
    monkeypatch.setattr(PF, "N_PROBLEMS", 2)
    traces = [_t(s, 0, True, gold_answer="theorem t : True", verification="ok")
              for s in PF.SETTINGS]
    traces += [_t(s, 1, False, gold_answer="theorem u : True", scoring_excluded=True,
                  verification="harness error: docker")
               for s in PF.SETTINGS]
    monkeypatch.setattr(PF, "run_eval_suite", lambda **kw: {"sample_traces": traces})
    out = PF.run_putnam_formal("m", "http://srv/v1", 1)
    reason = out["leaderboard_comparable_reason"]
    assert "--eval-limit subset" not in reason
    assert '1 problem(s) of "Lean with answer" had no scored attempt' in reason
