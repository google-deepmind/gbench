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

"""putnam_formal verifier: the DATASET statement is what gets proved.

Regression tests. Any reply containing "theorem" used to be compiled
as-is and judged from its own compiler output, so `theorem putnam_x : True := trivial`, an
added `axiom`, a commented-out statement, redefined helpers or a forged `#print axioms`
report all scored as solved. The verdict now comes from a trusted checker that never runs
reply code (see putnam_formal._CHECKER_LEAN).

The pure tests run everywhere. The kernel tests replay every known attack against the real
Lean image, plus the answer-faithfulness cases of the "Lean no answer" setting, and run only
when GBENCH_TEST_LEAN_IMAGE names a local Mathlib image, e.g.
`GBENCH_TEST_LEAN_IMAGE=gbench-putnam-formal:97b5e945` (several minutes; each case compiles a
reference, a solution and runs the checker in fresh containers).
"""

import json
import os

import pytest

from gbench.runners.eval_suites import putnam_formal as pf

PLAIN = "theorem putnam_test_a\n(n : ℕ)\n: n + 0 = n :=\nsorry"
ANSWER = ("abbrev putnam_test_b_solution : ℕ := sorry\n-- 2\n"
          "theorem putnam_test_b\n: 1 + 1 = putnam_test_b_solution :=\nsorry")
FIXED = ("abbrev putnam_test_f_solution : Prop := True\n"
         "theorem putnam_test_f\n: putnam_test_f_solution ↔ (1 = 1) :=\nsorry")
HELPER = ("def putnam_test_g_good (n : ℕ) : Prop := n = n\n"
          "theorem putnam_test_g\n: ∀ n, putnam_test_g_good n :=\nsorry")


def test_split_statement():
    name, sig, pre, slot = pf._split_statement(ANSWER)
    assert name == "putnam_test_b"
    assert sig.startswith("theorem putnam_test_b") and not sig.endswith("sorry")
    assert slot == ("putnam_test_b_solution", "ℕ")
    assert pf._split_statement(PLAIN)[3] is None
    # A FIXED answer (no `sorry`) is not a slot to fill: it is a dataset helper.
    assert pf._split_statement(FIXED)[3] is None


def test_identifier_led_bodies_are_tactics():
    """Review finding: one-line tactics with arguments were treated as term proofs."""
    assert pf._as_proof_term("nlinarith [sq_nonneg (a - b)]").startswith(" by\n")
    assert pf._as_proof_term("exact two_mul_le_add_sq a b").startswith(" by\n")
    assert pf._as_proof_term("by simp") == " by simp"
    assert pf._as_proof_term("fun n => rfl") == " fun n => rfl"


def test_solution_source_keeps_full_replies_and_places_bodies():
    src = pf._solution_source("import Mathlib\nopen Real\n"
                              "theorem putnam_test_a (n : ℕ) : n + 0 = n := by simp", PLAIN)
    assert src.startswith(pf._LEAN_HEADER) and src.count("import Mathlib") == 1
    assert "open Real" in src and "theorem putnam_test_a (n : ℕ)" in src
    body = pf._solution_source("simp", PLAIN)
    assert "theorem putnam_test_a\n(n : ℕ)\n: n + 0 = n := by\n  simp" in body


def test_verdict_comes_only_from_the_checker_line():
    n = "c0ffee"
    assert pf._parse_verdict(f"noise\nGBENCH_VERDICT {n} ok {{}}\n", n, 0) == (True, "ok", None)
    assert pf._parse_verdict(
        f'GBENCH_VERDICT {n} fail {{"reason": "non-standard axioms: [cheat]"}}', n, 1)[0] is False
    assert pf._parse_verdict("'x' depends on axioms: [propext]", n, 0) == (
        False, "checker produced no verdict", None)
    # A line without this check's nonce is not a verdict.
    assert pf._parse_verdict("GBENCH_VERDICT ok\n", n, 0)[0] is False


# --------------------------------------------------------------------------- #
# Real kernel: every attack from the pre-release reviews, plus honest proofs.
# --------------------------------------------------------------------------- #
_IMAGE = os.environ.get("GBENCH_TEST_LEAN_IMAGE")
kernel = pytest.mark.skipif(not _IMAGE, reason="set GBENCH_TEST_LEAN_IMAGE to a local "
                                                "Mathlib image to run the kernel tests")

_ATTACKS = [
    ("comment_wrap", PLAIN,
     "/-\ntheorem putnam_test_a\n(n : ℕ)\n: n + 0 = n := -/\ntheorem putnam_test_a : True := trivial"),
    ("variable_include_false", PLAIN,
     "variable (cheat : False)\ninclude cheat\n\n"
     "theorem putnam_test_a\n(n : ℕ)\n: n + 0 = n :=\n  cheat.elim"),
    ("namespace_shadow", PLAIN,
     "namespace X\naxiom cheat : False\ntheorem putnam_test_a\n(n : ℕ)\n: n + 0 = n :=\n"
     "  cheat.elim\nend X\ntheorem putnam_test_a : True := trivial"),
    ("added_axiom_with_fake_report", PLAIN,
     "axiom cheat : False\ntheorem putnam_test_a\n(n : ℕ)\n: n + 0 = n :=\n  cheat.elim\n"
     "#check \"'putnam_test_a' depends on axioms: [propext]\""),
    ("override_print_axioms", PLAIN,
     "axiom cheat : False\nopen Lean Elab Command in\nelab_rules : command\n"
     "  | `(#print axioms $id:ident) => logInfo m!\"'{id.getId}' depends on axioms: [propext]\"\n\n"
     "theorem putnam_test_a\n(n : ℕ)\n: n + 0 = n :=\n  cheat.elim"),
    ("exit_zero_before_the_theorem", PLAIN,
     "#eval (IO.Process.exit 0 : IO Unit)\n\ntheorem putnam_test_a\n(n : ℕ)\n: n + 0 = n := by simp"),
    ("silenced_sorry", PLAIN,
     "set_option warn.sorry false\ntheorem putnam_test_a\n(n : ℕ)\n: n + 0 = n :=\n  sorry"),
    ("native_decide", PLAIN,
     "theorem putnam_test_a\n(n : ℕ)\n: n + 0 = n := by\n  have : (2:ℕ) + 2 = 4 := by native_decide\n"
     "  simp"),
    ("macro_rules_rewrite", PLAIN,
     "macro_rules | `($a = $b) => `(True)\ntheorem putnam_test_a\n(n : ℕ)\n: n + 0 = n := trivial"),
    ("redefined_helper", HELPER,
     "def putnam_test_g_good (n : ℕ) : Prop := True\n"
     "theorem putnam_test_g\n: ∀ n, putnam_test_g_good n := fun _ => trivial"),
    ("flipped_fixed_answer", FIXED,
     "abbrev putnam_test_f_solution : Prop := (1 = 1)\n"
     "theorem putnam_test_f\n: putnam_test_f_solution ↔ (1 = 1) := Iff.rfl"),
    ("wrong_answer", ANSWER,
     "abbrev putnam_test_b_solution : ℕ := 3\ntheorem putnam_test_b\n"
     ": 1 + 1 = putnam_test_b_solution := by rfl"),
    ("answer_slot_type_changed", ANSWER,
     "abbrev putnam_test_b_solution : Prop := True\ntheorem putnam_test_b\n"
     ": 1 + 1 = putnam_test_b_solution := by rfl"),
]
_HONEST = [
    ("full_theorem", PLAIN, "theorem putnam_test_a (n : ℕ) : n + 0 = n := by simp"),
    ("body_only_tactic", PLAIN, "simp"),
    ("helper_lemma_and_imports", PLAIN,
     "import Mathlib\nopen Real\nlemma aux (m : ℕ) : m + 0 = m := Nat.add_zero m\n"
     "theorem putnam_test_a (n : ℕ) : n + 0 = n := aux n"),
    ("answer_filled", ANSWER,
     "abbrev putnam_test_b_solution : ℕ := 2\ntheorem putnam_test_b\n"
     ": 1 + 1 = putnam_test_b_solution := by rfl"),
    ("answer_as_reducible_def", ANSWER,
     "@[reducible] def putnam_test_b_solution : ℕ := 2\ntheorem putnam_test_b\n"
     ": 1 + 1 = putnam_test_b_solution := by rfl"),
    ("fixed_answer_kept", FIXED,
     "abbrev putnam_test_f_solution : Prop := True\ntheorem putnam_test_f\n"
     ": putnam_test_f_solution ↔ (1 = 1) := by simp [putnam_test_f_solution]"),
    ("helper_kept", HELPER,
     "def putnam_test_g_good (n : ℕ) : Prop := n = n\n"
     "theorem putnam_test_g\n: ∀ n, putnam_test_g_good n := fun _ => rfl"),
]


# "Lean no answer": statements in the release's own format (import, placeholder, answer
# comment). The reply supplies the placeholder's value; the checker must prove it equal to the
# reference answer, so an answer that restates the problem, or a wrong answer that still makes
# the theorem provable, fails even though its proof is valid.
NA_LEAST = ("import Mathlib\n\nabbrev putnam_test_r_solution : Nat := sorry\n-- 101\n"
            "/-- The least prime above 100. -/\n"
            "theorem putnam_test_r : IsLeast {p : Nat | p.Prime /\\ 100 < p} putnam_test_r_solution :=\n"
            "sorry\n")
NA_PROP = ("import Mathlib\n\nabbrev putnam_test_p_solution : Prop := sorry\n-- True\n"
           "theorem putnam_test_p : putnam_test_p_solution <-> forall n : Nat, n % 2 = 0 \\/ n % 2 = 1 :=\n"
           "sorry\n")
NA_FUN = ("import Mathlib\n\nabbrev putnam_test_m_solution : Prod Int Int -> Int := sorry\n"
          "-- fun (a, b) => a - b\n"
          "theorem putnam_test_m : putnam_test_m_solution (5, 2) = 3 :=\nsorry\n")
_LEAST_TYPE = "IsLeast {p : Nat | p.Prime /\\ 100 < p} putnam_test_r_solution"
#: Tactic proof of the least-prime statement, usable as a body or under a full theorem.
_LEAST_TACTICS = ("  refine And.intro (show Nat.Prime 101 /\\ 100 < 101 by norm_num) ?_\n"
                  "  intro p hp\n  exact hp.2")
_SINF = "sInf {p : Nat | p.Prime /\\ 100 < p}"
_SINF_TACTICS = ("  refine And.intro (Nat.sInf_mem (Exists.intro 101 "
                 "(show Nat.Prime 101 /\\ 100 < 101 by norm_num))) ?_\n"
                 "  intro p hp\n  exact Nat.sInf_le hp")
#: (label, statement, reply, expected outcome). "review": the checker proves the answer neither
#: equal nor different, so it goes to the judge stage (never accepted by the checker alone);
#: otherwise the exact failure reason.
_NO_ANSWER_ATTACKS = [
    ("restated_as_sInf", NA_LEAST,
     f"noncomputable abbrev putnam_test_r_solution : Nat := {_SINF}\n"
     f"theorem putnam_test_r : {_LEAST_TYPE} := by\n{_SINF_TACTICS}", "review"),
    ("restated_via_choose", NA_LEAST,
     f"theorem ex : exists p, IsLeast {{p : Nat | p.Prime /\\ 100 < p}} p := by\n"
     f"  refine Exists.intro ({_SINF}) ?_\n{_SINF_TACTICS}\n"
     "noncomputable abbrev putnam_test_r_solution : Nat := Classical.choose ex\n"
     f"theorem putnam_test_r : {_LEAST_TYPE} :=\n  Classical.choose_spec ex", "review"),
    # The placeholder stays `sorry`, so the proof of the concrete answer does not type-check.
    ("answer_left_as_sorry", NA_LEAST, _LEAST_TACTICS, "solution did not compile"),
    ("answer_and_proof_left_as_sorry", NA_LEAST,
     f"abbrev putnam_test_r_solution : Nat := sorry\ntheorem putnam_test_r : {_LEAST_TYPE} := sorry",
     "fail non-standard axioms: [sorryAx]"),
    ("restated_prop", NA_PROP,
     "abbrev putnam_test_p_solution : Prop := forall n : Nat, n % 2 = 0 \\/ n % 2 = 1\n"
     "theorem putnam_test_p : putnam_test_p_solution <-> forall n : Nat, n % 2 = 0 \\/ n % 2 = 1 :=\n"
     "  Iff.rfl", "review"),
    ("provable_but_wrong_answer", NA_FUN,
     "abbrev putnam_test_m_solution : Prod Int Int -> Int := fun (a, b) => 3\n"
     "theorem putnam_test_m : putnam_test_m_solution (5, 2) = 3 := by decide", "review"),
]
_NO_ANSWER_HONEST = [
    ("reference_answer", NA_LEAST,
     f"abbrev putnam_test_r_solution : Nat := 101\ntheorem putnam_test_r : {_LEAST_TYPE} := by\n"
     f"{_LEAST_TACTICS}"),
    ("equal_answer_other_form", NA_LEAST,
     f"abbrev putnam_test_r_solution : Nat := 100 + 1\ntheorem putnam_test_r : {_LEAST_TYPE} := by\n"
     f"{_LEAST_TACTICS}"),
    ("prop_answer", NA_PROP,
     "abbrev putnam_test_p_solution : Prop := True\n"
     "theorem putnam_test_p : putnam_test_p_solution <-> forall n : Nat, n % 2 = 0 \\/ n % 2 = 1 := by\n"
     "  simp only [putnam_test_p_solution, true_iff]\n  intro n\n  omega"),
    ("function_answer_same_pattern", NA_FUN,
     "abbrev putnam_test_m_solution : Prod Int Int -> Int := fun (a, b) => a - b\n"
     "theorem putnam_test_m : putnam_test_m_solution (5, 2) = 3 := by decide"),
    ("function_answer_projections", NA_FUN,
     "abbrev putnam_test_m_solution : Prod Int Int -> Int := fun p => p.1 - p.2\n"
     "theorem putnam_test_m : putnam_test_m_solution (5, 2) = 3 := by decide"),
]
# "Lean with answer": the same statements with the answer inlined by the release's
# rewrite_solutions.py transformation (pf.with_answer_statement).
_WITH_ANSWER_HONEST = [
    ("body_only", pf.with_answer_statement(NA_LEAST), _LEAST_TACTICS),
    ("inlined_pattern", pf.with_answer_statement(NA_FUN),
     "theorem putnam_test_m : ((fun (a, b) => a - b) : Prod Int Int -> Int ) (5, 2) = 3 := by decide"),
    # A helper with the same pattern first makes Lean reuse its matcher in the theorem's type;
    # the checker compares types with local definitions unfolded, so this is still accepted.
    ("reused_matcher", pf.with_answer_statement(NA_FUN),
     "theorem helper : ((fun (a, b) => a - b) : Prod Int Int -> Int) (1, 1) = 0 := by decide\n"
     "theorem putnam_test_m : ((fun (a, b) => a - b) : Prod Int Int -> Int ) (5, 2) = 3 := by decide"),
]


@kernel
@pytest.mark.parametrize("label,stmt,reply,expected", _NO_ANSWER_ATTACKS,
                         ids=[a[0] for a in _NO_ANSWER_ATTACKS])
def test_no_answer_unfaithful_answers_are_not_accepted_by_the_checker(monkeypatch, tmp_path, label,
                                                                     stmt, reply, expected):
    """Each unfaithful answer ends exactly as expected: sent to the judge stage (the judge then
    decides; see test_a_restated_answer_the_judge_rejects_is_scored_wrong in the payload tests),
    or failed with the given reason. None is accepted by the checker."""
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    ok, why = pf._verify(f"```lean\n{reply}\n```", stmt)
    assert ok is False, why
    if expected == "review":
        assert why.startswith(pf.REVIEW_PREFIX), why
        review = json.loads(why[len(pf.REVIEW_PREFIX):])
        assert review["different"] == "" and review["answer"] != review["reference"], review
    else:
        assert why == expected, why


@kernel
@pytest.mark.parametrize("label,stmt,reply", _NO_ANSWER_HONEST + _WITH_ANSWER_HONEST,
                         ids=[h[0] for h in _NO_ANSWER_HONEST + _WITH_ANSWER_HONEST])
def test_honest_answers_are_accepted_in_both_settings(monkeypatch, tmp_path, label, stmt, reply):
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    assert pf._verify(f"```lean\n{reply}\n```", stmt) == (True, "ok")


@kernel
def test_restated_answer_goes_to_the_answer_review(monkeypatch, tmp_path):
    """The tactics can neither prove `sInf {...}` equal to 101 nor prove it different, so the
    checker hands both elaborated answers to the judge stage instead of accepting it."""
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    ok, why = pf._verify(f"```lean\n{_NO_ANSWER_ATTACKS[0][2]}\n```", NA_LEAST)
    assert ok is False and why.startswith(pf.REVIEW_PREFIX), why
    review = json.loads(why[len(pf.REVIEW_PREFIX):])
    assert "sInf" in review["answer"] and review["reference"] == "101", review


@kernel
def test_checker_compiles_in_the_image(monkeypatch):
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    assert pf._checker_compiles() == (True, "")


@kernel
@pytest.mark.parametrize("label,stmt,reply", _ATTACKS, ids=[a[0] for a in _ATTACKS])
def test_attacks_are_rejected_by_the_kernel_checker(monkeypatch, tmp_path, label, stmt, reply):
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    assert pf._eval_putnam_formal(f"```lean\n{reply}\n```", stmt) is False


@kernel
@pytest.mark.parametrize("label,stmt,reply", _HONEST, ids=[h[0] for h in _HONEST])
def test_honest_proofs_are_accepted(monkeypatch, tmp_path, label, stmt, reply):
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    assert pf._eval_putnam_formal(f"```lean\n{reply}\n```", stmt) is True


# --------------------------------------------------------------------------- #
# Scoring harness: parallel, contained, and honest about what was not judged.
# --------------------------------------------------------------------------- #
import asyncio
import subprocess
import threading


def _proc(rc, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def test_docker_failure_is_a_harness_error_not_a_wrong_proof(monkeypatch, tmp_path):
    """A down daemon or missing image (docker run exit 125) used to look like a statement
    that does not compile, i.e. an "unverifiable" problem scored 0 for every model."""
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(pf, "_image_id", lambda: "img")
    monkeypatch.setattr(pf, "_lean_container", lambda *a, **k: _proc(125, err="no daemon"))
    ok, why = pf._verify("simp", PLAIN)
    assert ok is False and why.startswith("harness error") and "no daemon" in why


def test_unexpected_errors_are_contained(monkeypatch):
    monkeypatch.setattr(pf, "_reference_olean", lambda s: (_ for _ in ()).throw(OSError("disk")))
    ok, why = pf._verify("simp", PLAIN)
    assert ok is False and why == "harness error: OSError: disk"


def test_container_is_killed_on_timeout(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 1)
        return _proc(0)
    monkeypatch.setattr(pf.subprocess, "run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        pf._lean_container([], "true", 1)
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "kill", name]


def test_project_dir_is_configurable(monkeypatch):
    seen = []
    monkeypatch.setattr(pf, "LEAN_PROJECT", "/srv/lake")
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or _proc(0))
    pf._lean_container([], "true", 1)
    assert seen[0][-1] == "cd /srv/lake && true"


def _traces(n, health="ok"):
    return [{"status": "PENDING_JUDGE", "response_text": "simp", "gold_answer": f"stmt{i % 2}",
             "health": health} for i in range(n)]


def test_scorer_verifies_in_parallel_and_keeps_truncation(monkeypatch):
    barrier = threading.Barrier(4, timeout=10)

    def fake_verify(reply, stmt):
        barrier.wait()      # deadlocks (BrokenBarrierError) unless 4 checks run at once
        return True, "ok"
    monkeypatch.setattr(pf, "_verify", fake_verify)
    traces = _traces(4, health="truncated")
    asyncio.run(pf._make_scorer(4, {})(traces))
    assert all(t["is_correct"] and t["status"] == "TRUNCATED" for t in traces)


def test_scorer_discloses_unverifiable_and_flags_harness_errors(monkeypatch):
    verdicts = iter([(False, pf.UNVERIFIABLE), (False, "harness error: docker"),
                     (False, pf.UNVERIFIABLE), (True, "ok")])
    monkeypatch.setattr(pf, "_verify", lambda r, s: next(verdicts))
    traces, stats = _traces(4), {}
    asyncio.run(pf._make_scorer(1, stats)(traces))
    assert stats == {"unverifiable_statements": 1, "verification_harness_errors": 1,
                     "judge_models": {}}
    assert traces[1]["judge_grade"] == "JUDGE_FAILED"
    assert "judge_grade" not in traces[0] and traces[3]["is_correct"] is True


def test_run_marks_unverifiable_runs_not_leaderboard_comparable(monkeypatch):
    monkeypatch.setattr(pf, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(pf, "_load_putnam_formal_samples", lambda limit=None: [])
    monkeypatch.setattr(pf, "_lean_image_provenance", lambda: {})
    monkeypatch.setattr(pf, "_verify", lambda r, s: (False, pf.UNVERIFIABLE))

    def fake_suite(**kw):
        assert kw["supports_attempts"] is True
        asyncio.run(kw["async_eval_fn"](_traces(2)))
        return {"accuracy": 0.0}
    monkeypatch.setattr(pf, "run_eval_suite", fake_suite)
    out = pf.run_putnam_formal("m", "http://srv/v1", 1, sandboxes=2)
    assert out["unverifiable_statements"] == 2 and out["leaderboard_comparable"] is False
