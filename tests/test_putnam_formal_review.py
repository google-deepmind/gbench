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

"""putnam_formal: the checker's verdict channel and the two-stage answer-faithfulness check.

The verdict must be one the program under test cannot write: the harness draws a nonce after
the solution is compiled and gives it only to the checker, the checker prints exactly one
nonce-tagged verdict line with every line break in it escaped, and `ok` counts only with exit
code 0. In the "Lean no answer" setting the checker decides the answer by a kernel proof
(equal or different) when it can, and otherwise hands it to the gbench Gemini judge cascade.

The pure tests use fake containers and a fake judge. The kernel tests run only when
GBENCH_TEST_LEAN_IMAGE names the local image, e.g.
`GBENCH_TEST_LEAN_IMAGE=gbench-putnam-formal:97b5e945`.
"""

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from gbench.runners.eval_suites import putnam as P
from gbench.runners.eval_suites import putnam_formal as pf

REPO = Path(__file__).resolve().parents[1]
PLAIN = "theorem putnam_test_a\n(n : \u2115)\n: n + 0 = n :=\nsorry"
LEAST = ("import Mathlib\n\nabbrev putnam_test_r_solution : Nat := sorry\n-- 101\n"
         "theorem putnam_test_r : IsLeast {p : Nat | p.Prime /\\ 100 < p} putnam_test_r_solution :=\n"
         "sorry\n")
REVIEW = {"answer": "1 / \u221a2", "reference": "\u221a2 / 2", "type": "\u211d",
          "different": "", "printed": True, "print_error": ""}


def _proc(rc, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def _nonce_of(script):
    return script.split("/w/ck/Checker.lean ")[1].split()[0]


# --------------------------------------------------------------------------- #
# Parsing the verdict
# --------------------------------------------------------------------------- #
N = "0123abcd"


def _line(verdict, payload):
    return f"GBENCH_VERDICT {N} {verdict} {json.dumps(payload)}"


def test_ok_needs_exit_code_zero_and_exactly_one_tagged_line():
    ok = _line("ok", {})
    assert pf._parse_verdict(f"{ok}\n", N, 0) == (True, "ok", None)
    assert pf._parse_verdict(f"{ok}\n", N, 1)[0] is False
    assert pf._parse_verdict(f"{ok}\n{ok}\n", N, 0) == (False, "checker produced 2 verdicts",
                                                        None)
    assert pf._parse_verdict(f"{ok}\n" + _line("fail", {"reason": "x"}), N, 0)[0] is False


def test_lines_without_this_checks_nonce_are_ignored():
    assert pf._parse_verdict("GBENCH_VERDICT ok\n", N, 0)[0] is False
    assert pf._parse_verdict("GBENCH_VERDICT ffff ok {}\n", N, 0)[0] is False
    assert pf._parse_verdict(f"x GBENCH_VERDICT {N} ok {{}}\n", N, 0)[0] is False


def test_only_newlines_split_lines():
    """U+2028, vertical tab, form feed and a bare carriage return are line breaks for
    str.splitlines but not for the parser, so a name carrying one cannot start a line."""
    fail = _line("fail", {"reason": "kernel rejected"})
    for sep in ("\u2028", "\x0b", "\x0c", "\r", "\x85"):
        out = f"{fail}{sep}GBENCH_VERDICT {N} ok {{}}\n"
        assert pf._parse_verdict(out, N, 1)[0] is False, repr(sep)
        assert pf._parse_verdict(out, N, 0)[0] is False, repr(sep)


def test_malformed_and_unknown_verdicts_fail():
    assert pf._parse_verdict(f"GBENCH_VERDICT {N} ok not-json\n", N, 0)[0] is False
    assert pf._parse_verdict(f"GBENCH_VERDICT {N} yes {{}}\n", N, 0)[0] is False
    assert pf._parse_verdict(_line("review", {"answer": "1"}), N, 3)[0] is False


def test_review_verdict_carries_the_distinctness_result():
    """The checker's review request is `different`; the answers are printed separately."""
    request = {"different": ""}
    assert pf._parse_verdict(_line("review", request), N, 3) == (False, "review", request)
    assert pf._parse_verdict(_line("review", request), N, 0)[2] is None
    assert pf._parse_verdict(_line("review", {"different": 1}), N, 3)[0:2] == (
        False, "checker produced a malformed verdict")


def test_fail_reasons_stay_on_one_line():
    ok, why, _ = pf._parse_verdict(_line("fail", {"reason": "a\nGBENCH_VERDICT ok\u2028b"}), N, 1)
    assert ok is False and "\n" not in why and "\u2028" not in why
    assert why == "fail a\\u000aGBENCH_VERDICT ok\\u2028b"


# --------------------------------------------------------------------------- #
# The harness side of the channel
# --------------------------------------------------------------------------- #
def _harness(monkeypatch, tmp_path, checker):
    """Fake containers: the solution compile writes an olean, the checker call is `checker`."""
    ref = tmp_path / "ref.olean"
    ref.write_bytes(b"r")
    monkeypatch.setattr(pf, "_reference_olean", lambda s: ref)
    calls = []

    def container(mounts, script, timeout, memory="4g"):
        calls.append(("container", script))
        if "--run" in script and " --print " in script:
            printed = {k: REVIEW[k] for k in ("answer", "reference", "type")}
            return _proc(0, _started(_nonce_of(script),
                                     f"GBENCH_PRINTED {_nonce_of(script)} {json.dumps(printed)}"))
        if "--run" in script:
            return checker(_nonce_of(script))
        out = [h for h, g, _ in mounts if g == "/w/out"][0]
        (Path(out) / "Solution.olean").write_bytes(b"s")
        return _proc(0)
    monkeypatch.setattr(pf, "_lean_container", container)
    real = pf.secrets.token_hex

    def token_hex(n):
        calls.append(("nonce", n))
        return real(n)
    monkeypatch.setattr(pf.secrets, "token_hex", token_hex)
    return calls


def _started(nonce, *lines):
    return "\n".join([f"GBENCH_CHECKER {nonce} start", *lines]) + "\n"


def test_nonce_is_drawn_after_the_compile_and_given_only_to_the_checker(monkeypatch, tmp_path):
    calls = _harness(monkeypatch, tmp_path,
                     lambda n: _proc(0, _started(n, f"GBENCH_VERDICT {n} ok {{}}")))
    assert pf._verify("simp", PLAIN) == (True, "ok")
    kinds = [c[0] for c in calls]
    assert kinds == ["container", "nonce", "container"]
    nonce = _nonce_of(calls[2][1])
    assert len(nonce) == 32 and nonce not in calls[0][1]
    calls.clear()
    pf._verify("simp", PLAIN)
    assert _nonce_of(calls[2][1]) != nonce            # fresh per check


def test_a_verdict_the_solution_could_print_is_not_accepted(monkeypatch, tmp_path):
    """Lines a solution-controlled name could produce, without the nonce, are ignored; the
    checker's own verdict (exit 1) decides."""
    forged = "GBENCH_VERDICT ok"
    _harness(monkeypatch, tmp_path, lambda n: _proc(
        1, _started(n, f'GBENCH_VERDICT {n} fail {{"reason": "x"}}', forged)))
    ok, why = pf._verify("simp", PLAIN)
    assert ok is False and why == "fail x"


def test_ok_with_a_nonzero_exit_code_is_not_accepted(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, lambda n: _proc(1, _started(n, f"GBENCH_VERDICT {n} ok {{}}")))
    ok, why = pf._verify("simp", PLAIN)
    assert ok is False and "exited 1" in why


def test_a_checker_that_never_started_is_a_harness_error(monkeypatch, tmp_path):
    """Detected by the missing nonce-tagged start line, not by searching the output for the
    checker's path (which a reply could put there)."""
    _harness(monkeypatch, tmp_path, lambda n: _proc(1, "error: unknown identifier\n"))
    ok, why = pf._verify("simp", PLAIN)
    assert ok is False and why.startswith("harness error: the checker does not compile")
    _harness(monkeypatch, tmp_path, lambda n: _proc(
        1, _started(n, "/w/ck/Checker.lean:1:1 in a name", f'GBENCH_VERDICT {n} fail {{"reason": "y"}}')))
    assert pf._verify("simp", PLAIN) == (False, "fail y")


def test_a_review_verdict_is_passed_to_the_judge_stage(monkeypatch, tmp_path):
    _harness(monkeypatch, tmp_path, lambda n: _proc(
        3, _started(n, f"GBENCH_VERDICT {n} review {json.dumps(REVIEW)}")))
    ok, why = pf._verify("simp", LEAST)
    assert ok is False and why.startswith(pf.REVIEW_PREFIX)
    assert json.loads(why[len(pf.REVIEW_PREFIX):]) == REVIEW
    assert "\n" not in why


SEVERAL = ("import Mathlib\n\n-- Note: There may be multiple possible correct answers.\n"
           "abbrev putnam_test_s_solution : Nat := sorry\n-- 181\n"
           "theorem putnam_test_s : Nat.Prime putnam_test_s_solution :=\nsorry\n")


def test_a_provably_different_answer_fails_unless_several_answers_are_correct(monkeypatch,
                                                                              tmp_path):
    diff = dict(REVIEW, different="norm_num")
    _harness(monkeypatch, tmp_path, lambda n: _proc(
        3, _started(n, f"GBENCH_VERDICT {n} review {json.dumps(diff)}")))
    ok, why = pf._verify("simp", LEAST)
    assert ok is False and why == (f"fail {pf.UNFAITHFUL}: putnam_test_r_solution is provably "
                                   "different from the reference answer (norm_num)")
    ok, why = pf._verify("simp", SEVERAL)
    assert ok is False and why.startswith(pf.REVIEW_PREFIX)
    assert pf._REVIEW_SEVERAL in pf._review_prompt(SEVERAL, diff)
    assert pf._REVIEW_SEVERAL not in pf._review_prompt(LEAST, diff)


def test_the_several_answers_note_is_recognised_in_its_release_wordings():
    for note in ("-- Note: This problem may have multiple correct answers.",
                 "-- Note: There might be multiple possible correct answers.",
                 "-- Note: this problem admits several possible correct solutions; this is the one",
                 "-- Note: Boosted domain to R, which is fine. There may be multiple correct answers."):
        assert pf.SEVERAL_ANSWERS.search(f"import Mathlib\n{note}\nabbrev x : Nat := sorry\n")
    assert not pf.SEVERAL_ANSWERS.search("-- Note: uses (N -> R) instead of (Fin n -> R)\n")
    assert not pf.SEVERAL_ANSWERS.search("theorem t : A = c * id -- a scalar multiple\n")


def test_checker_startup_check_uses_the_nonce(monkeypatch):
    scripts = []

    def container(mounts, script, timeout, memory="4g"):
        scripts.append(script)
        n = script.split()[-1]
        return _proc(2, _started(n, f"GBENCH_VERDICT {n} usage {{}}"))
    monkeypatch.setattr(pf, "_lean_container", container)
    assert pf._checker_compiles() == (True, "")
    monkeypatch.setattr(pf, "_lean_container", lambda *a, **k: _proc(
        1, "/w/ck/Checker.lean:113:62: error: Invalid field `replay`\n"))
    ok, why = pf._checker_compiles()
    assert ok is False and "Invalid field" in why and "Rebuild" not in why


def test_checker_source_escapes_and_tags_every_line():
    src = pf._CHECKER_LEAN
    assert "__DISTINCT__" not in src and "__TACTICS__" not in src
    for tac in pf.DISTINCTNESS_TACTICS:
        assert f'"{tac}"' in src
    assert 'IO.println s!"GBENCH_VERDICT {nonce} {verdict} {oneLine payload.compress}"' in src
    assert 'IO.println s!"GBENCH_CHECKER {nonce} start"' in src
    assert "v == 0x2028 || v == 0x2029" in src
    assert "base.replay solC" in src and "toKernelEnv.replay" not in src
    # Every other println is the nonce-less usage line for a missing or malformed nonce.
    others = [ln for ln in src.splitlines() if "IO.println" in ln
              and "{nonce}" not in ln]
    assert others and all('"GBENCH_VERDICT - usage {}"' in ln for ln in others)


# --------------------------------------------------------------------------- #
# Stage two: the judge review
# --------------------------------------------------------------------------- #
def _review_trace(gold=LEAST):
    return {"status": "PENDING_JUDGE", "response_text": "reply", "gold_answer": gold,
            "category": "lean_no_answer", "health": "ok"}


def _fake_judge(monkeypatch, replies):
    seen = []
    replies = iter(replies)

    async def judge(prompt, config=None):
        seen.append(prompt)
        text = next(replies)
        return (None, "JUDGE_OUTAGE") if text is None else (text, "gemini-test")
    monkeypatch.setattr(pf, "judge_generate_cascade", judge)
    return seen


def _score(monkeypatch, verdict, replies, trace=None):
    monkeypatch.setattr(pf, "_verify", lambda r, s: verdict)
    seen = _fake_judge(monkeypatch, replies)
    trace = trace or _review_trace()
    stats = {}
    asyncio.run(pf._make_scorer(1, stats)([trace]))
    return trace, stats, seen


_REVIEW_WHY = pf.REVIEW_PREFIX + json.dumps(REVIEW)


def test_judge_accepts_an_equal_answer_the_tactics_could_not_bridge(monkeypatch):
    t, stats, seen = _score(monkeypatch, (False, _REVIEW_WHY), ["Same number.\nGrade: CORRECT"])
    assert t["is_correct"] is True and t["verification"] == "ok"
    assert t["answer_review"] == {"stage": "judge", "faithful": True,
                                  "judge_model": "gemini-test", "outcome": "faithful"}
    assert t["judge_model"] == "gemini-test" and stats["judge_models"] == {"gemini-test": 1}
    prompt = seen[0]
    assert "1 / \u221a2" in prompt and "\u221a2 / 2" in prompt and "putnam_test_r_solution" in prompt
    assert "```lean\n101\n```" in prompt                       # the release's answer comment
    assert "-- 101" not in prompt                              # the statement is the prompt's


def test_judge_rejection_is_an_unfaithful_answer(monkeypatch):
    t, _, _ = _score(monkeypatch, (False, _REVIEW_WHY),
                     ["The answer restates the condition. Grade: CORRECT?\nGrade: INCORRECT"])
    assert t["is_correct"] is False and t["judge_grade"] == "INCORRECT"
    assert t["verification"].startswith(f"fail {pf.UNFAITHFUL}: the judge (gemini-test)")
    assert t["answer_review"]["faithful"] is False


def test_judge_outage_is_not_scored_and_no_verdict_is_a_failed_request(monkeypatch):
    t, _, _ = _score(monkeypatch, (False, _REVIEW_WHY), [None])
    assert t["judge_grade"] == "JUDGE_OUTAGE" and t["is_correct"] is False
    assert t["answer_review"]["outcome"] == "judge outage"
    t, _, seen = _score(monkeypatch, (False, _REVIEW_WHY), ["hmm", "unsure", "no idea"])
    assert len(seen) == pf._REVIEW_ATTEMPTS
    assert t["judge_grade"] == "JUDGE_FAILED" and t["is_correct"] is False
    assert t["answer_review"]["outcome"] == "no verdict"


def test_a_long_answer_is_sent_to_the_judge(monkeypatch):
    """No size limit decides a score: a long elaborated answer is reviewed like any other."""
    big = dict(REVIEW, answer="x" * 200001)
    t, _, seen = _score(monkeypatch, (False, pf.REVIEW_PREFIX + json.dumps(big)),
                        ["Same object.\nGrade: CORRECT"])
    assert len(seen) == 1 and big["answer"] in seen[0]
    assert t["is_correct"] is True and t["answer_review"]["outcome"] == "faithful"


def test_kernel_decided_answers_record_the_kernel_stage(monkeypatch):
    t, stats, seen = _score(monkeypatch, (True, "ok"), [])
    assert t["answer_review"] == {"stage": "kernel", "faithful": True} and seen == []
    assert stats["judge_models"] == {}
    t, _, _ = _score(monkeypatch, (False, f"fail {pf.UNFAITHFUL}: x is provably different"), [])
    assert t["answer_review"] == {"stage": "kernel", "faithful": False}
    t, _, _ = _score(monkeypatch, (False, "solution did not compile"), [])
    assert "answer_review" not in t                        # the answer was never compared


def test_review_prompt_includes_the_informal_statement(monkeypatch):
    monkeypatch.setattr(pf, "_RELEASE", {"informal": [
        {"problem_name": "putnam_test_r", "informal_statement": "Find the least prime above 100."}]})
    prompt = pf._review_prompt(LEAST, REVIEW)
    assert "Find the least prime above 100." in prompt and "Grade: CORRECT" in prompt


def test_report_counts_the_review_stages_and_leaves_outages_out_of_accuracy():
    na = "lean_no_answer"
    traces = [
        {"category": na, "source_sample_idx": 0, "is_correct": True, "gold_answer": LEAST,
         "verification": "ok", "answer_review": {"stage": "kernel", "faithful": True}},
        {"category": na, "source_sample_idx": 1, "is_correct": True, "gold_answer": LEAST,
         "verification": "ok", "answer_review": {"stage": "judge", "faithful": True,
                                                 "outcome": "faithful"}},
        {"category": na, "source_sample_idx": 2, "is_correct": False, "gold_answer": LEAST,
         "verification": f"fail {pf.UNFAITHFUL}: judge", "judge_grade": "INCORRECT",
         "answer_review": {"stage": "judge", "faithful": False, "outcome": "not faithful"}},
        {"category": na, "source_sample_idx": 3, "is_correct": False, "gold_answer": LEAST,
         "verification": "answer review not done: judge outage", "judge_grade": "JUDGE_OUTAGE",
         "answer_review": {"stage": "judge", "faithful": None, "outcome": "judge outage"}},
    ]
    rep = pf._settings_report(traces)[na]
    assert rep["answer_review"] == {"kernel_faithful": 1, "kernel_not_faithful": 0,
                                    "judge_faithful": 1, "judge_not_faithful": 1,
                                    "judge_outage": 1, "judge_no_verdict": 0,
                                    "not_printed": 0}
    assert rep["judge_outages"] == 1 and rep["accuracy"] == round(2 / 3 * 100, 2)
    # problem 3's only attempt is a judge outage: unscored, not unsolved
    assert rep["solved"] == 2 and rep["problems"] == 3 and rep["answer_not_faithful"] == 1
    assert rep["problems_unscored"] == 1


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def _run(monkeypatch, traces, stats_update=None):
    monkeypatch.setattr(pf, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(pf, "_load_putnam_formal_samples", lambda limit=None: [])
    monkeypatch.setattr(pf, "_lean_image_provenance", lambda: {"reproducible": True})
    monkeypatch.setattr(pf, "N_PROBLEMS", 1)

    def fake_suite(**kw):
        return {"sample_traces": traces}
    monkeypatch.setattr(pf, "run_eval_suite", fake_suite)
    real = pf._make_scorer

    def scorer(workers, stats, judge_concurrency=32):
        stats.update(stats_update or {"verification_harness_errors": 0, "judge_models": {}})
        return real(workers, stats, judge_concurrency)
    monkeypatch.setattr(pf, "_make_scorer", scorer)
    return pf.run_putnam_formal("m", "http://srv/v1", 1)


def _t(cat, ok=True, **kw):
    return {"category": cat, "source_sample_idx": 0, "is_correct": ok, "gold_answer": LEAST,
            "verification": "ok", **kw}


def test_run_records_the_judge_and_both_stages(monkeypatch):
    out = _run(monkeypatch, [_t("lean_with_answer"), _t("lean_no_answer")],
               {"verification_harness_errors": 0, "judge_models": {"gemini-test": 2}})
    assert out["leaderboard_comparable"] is True
    assert out["judge"] == "gbench-gemini-cascade" and out["judge_models"] == {"gemini-test": 2}
    assert out["judge_cascade"]
    for tac in pf.FAITHFULNESS_TACTICS + pf.DISTINCTNESS_TACTICS:
        assert tac in out["answer_faithfulness"]
    assert "Gemini judge cascade" in out["answer_faithfulness"]


def test_harness_errors_make_the_run_not_leaderboard_comparable(monkeypatch):
    out = _run(monkeypatch, [_t("lean_with_answer"), _t("lean_no_answer")],
               {"verification_harness_errors": 3, "judge_models": {}})
    assert out["leaderboard_comparable"] is False
    assert "3 replies were not judged (harness errors)" in out["leaderboard_comparable_reason"]


def test_unreviewed_answers_make_the_run_not_leaderboard_comparable(monkeypatch):
    outage = _t("lean_no_answer", ok=False, judge_grade="JUDGE_OUTAGE",
                answer_review={"stage": "judge", "faithful": None, "outcome": "judge outage"})
    out = _run(monkeypatch, [_t("lean_with_answer"), outage])
    assert out["leaderboard_comparable"] is False
    assert "did not get an answer review" in out["leaderboard_comparable_reason"]


def test_a_missing_judge_key_is_a_startup_error(monkeypatch):
    monkeypatch.setattr(pf.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd, **kw: _proc(0))
    monkeypatch.setattr(pf, "_probe_image", lambda: ({
        "toolchain": pf.LEAN_TOOLCHAIN, "mathlib_rev": pf.MATHLIB_REV,
        "putnambench_revision": pf.REVISION}, ""))
    monkeypatch.setattr(pf, "_checker_compiles", lambda: (True, ""))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    ok, why = pf.check_putnam_formal_prerequisites()
    assert ok is False and "GEMINI_API_KEY" in why
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(pf, "gemini_key_live_valid", lambda k: (False, "HTTP 403"))
    ok, why = pf.check_putnam_formal_prerequisites()
    assert ok is False and "HTTP 403" in why
    monkeypatch.setattr(pf, "gemini_key_live_valid", lambda k: (True, ""))
    monkeypatch.setattr(pf, "_IMAGE_FACTS", {})
    assert pf.check_putnam_formal_prerequisites() == (True, "")


# --------------------------------------------------------------------------- #
# Source and docs hygiene
# --------------------------------------------------------------------------- #
def test_no_internal_labels_in_the_suite_sources():
    for mod in (P, pf):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        assert not re.search(r"\((?:audit|wave)\b[^)]*\)", text, re.IGNORECASE), mod.__name__


def test_docs_describe_the_verdict_channel_and_both_review_stages():
    doc = (REPO / "docs" / "evals" / "putnam_formal.md").read_text(encoding="utf-8")
    for item in (*pf.FAITHFULNESS_TACTICS, *pf.DISTINCTNESS_TACTICS, "GEMINI_API_KEY",
                 "answer_review", "judge_models", "nonce", "Lean.Environment.replay",
                 "harness errors", "--max-output-tokens"):
        assert item in doc, item
    assert doc.isascii()


# --------------------------------------------------------------------------- #
# Real kernel
# --------------------------------------------------------------------------- #
_IMAGE = os.environ.get("GBENCH_TEST_LEAN_IMAGE")
kernel = pytest.mark.skipif(not _IMAGE, reason="set GBENCH_TEST_LEAN_IMAGE to the local "
                                                "putnam_formal image to run the kernel tests")

NA_PROP = ("import Mathlib\n\nabbrev putnam_test_p_solution : Prop := sorry\n-- True\n"
           "theorem putnam_test_p : putnam_test_p_solution <-> forall n : Nat, n % 2 = 0 \\/ n % 2 = 1 :=\n"
           "sorry\n")


def _crafted(name_lean):
    """Two replies that make the checker print a solution-chosen name: a declaration the kernel
    rejects (added with kernel checking switched off at compile time), and, in the no-answer
    setting, an answer that is a local inductive type."""
    replay = (
        "open Lean Elab Command in\n"
        "#eval show CommandElabM Unit from liftCoreM <|\n"
        "  withOptions (fun o => o.setBool `debug.skipKernelTC true) <|\n"
        f"    addDecl (.thmDecl {{ name := Name.mkSimple {name_lean}, levelParams := [],\n"
        "                        type := mkConst ``False, value := mkConst ``True.intro })\n\n"
        "theorem putnam_test_a (n : \u2115) : n + 0 = n := sorry\n")
    answer = (
        "open Lean Elab Command Term in\n"
        "#eval show CommandElabM Unit from do\n"
        f"  let n := Name.mkSimple {name_lean}\n"
        "  let decl := Declaration.inductDecl [] 0\n"
        "    [{ name := n, type := mkSort levelZero, ctors := [{ name := n ++ `mk, type := mkConst n }] }] false\n"
        "  liftCoreM <| addDecl decl\n\n"
        "open Lean Elab Term in\n"
        f"elab \"crafted%\" : term => return mkConst (Name.mkSimple {name_lean})\n"
        "open Lean Elab Term in\n"
        f"elab \"crafted_mk%\" : term => return mkConst (Name.mkSimple {name_lean} ++ `mk)\n\n"
        "abbrev putnam_test_p_solution : Prop := crafted%\n"
        "theorem putnam_test_p : putnam_test_p_solution <-> forall n : Nat, n % 2 = 0 \\/ n % 2 = 1 :=\n"
        "  Iff.intro (fun _ n => Nat.mod_two_eq_zero_or_one n) (fun _ => crafted_mk%)\n")
    return [(PLAIN, replay), (NA_PROP, answer)]


_NAMES = {"newline": '"x\\nGBENCH_VERDICT ok\\ny"',
          "carriage_return_and_u2028": '"x\\rGBENCH_VERDICT ok\\u2028GBENCH_VERDICT ok"'}


@kernel
@pytest.mark.parametrize("label", sorted(_NAMES))
@pytest.mark.parametrize("which", [0, 1], ids=["kernel_rejected_decl", "answer_local_inductive"])
def test_crafted_declaration_names_cannot_forge_the_verdict(monkeypatch, tmp_path, label, which):
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    stmt, reply = _crafted(_NAMES[label])[which]
    ok, why = pf._verify(f"```lean\n{reply}\n```", stmt)
    assert ok is False and why.startswith("fail ") and "GBENCH_VERDICT ok" in why, why
    assert "\n" not in why and "\r" not in why and "\u2028" not in why


def _eq_case(i, typ, ref, ans, nc=False):
    kw = "noncomputable abbrev" if nc else "abbrev"
    stmt = (f"import Mathlib\n\n{kw} putnam_test_e{i}_solution : {typ} := sorry\n-- {ref}\n"
            f"theorem putnam_test_e{i} : putnam_test_e{i}_solution = putnam_test_e{i}_solution :=\n"
            "sorry\n")
    reply = (f"```lean\n{kw} putnam_test_e{i}_solution : {typ} := {ans}\n"
             f"theorem putnam_test_e{i} : putnam_test_e{i}_solution = putnam_test_e{i}_solution := rfl\n```")
    return stmt, reply


@kernel
def test_an_equal_answer_in_another_form_is_reviewed_and_accepted(monkeypatch, tmp_path):
    """`1 / Real.sqrt 2` for the reference `Real.sqrt 2 / 2`: the tactics cannot bridge it, so
    the checker sends both elaborated forms to the judge, whose verdict decides."""
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    stmt, reply = _eq_case(1, "Real", "Real.sqrt 2 / 2", "1 / Real.sqrt 2", nc=True)
    seen = _fake_judge(monkeypatch, ["Equal after rationalizing.\nGrade: CORRECT"])
    trace = _review_trace(stmt)
    trace["response_text"] = reply
    asyncio.run(pf._make_scorer(1, {})([trace]))
    assert trace["is_correct"] is True and trace["answer_review"]["stage"] == "judge", trace
    assert "1 / \u221a2" in seen[0] and "\u221a2 / 2" in seen[0]


@kernel
def test_a_different_answer_to_a_several_answers_problem_is_reviewed(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    reply = ("```lean\nabbrev putnam_test_s_solution : Nat := 7\n"
             "theorem putnam_test_s : Nat.Prime putnam_test_s_solution := by norm_num\n```")
    ok, why = pf._verify(reply, SEVERAL)
    assert ok is False and why.startswith(pf.REVIEW_PREFIX), why
    review = json.loads(why[len(pf.REVIEW_PREFIX):])
    assert review["answer"] == "7" and review["reference"] == "181" and review["different"]


@kernel
def test_a_provably_different_answer_is_rejected_by_the_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "LEAN_IMAGE", _IMAGE)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    stmt, reply = _eq_case(2, "Nat", "5", "6")
    ok, why = pf._verify(reply, stmt)
    assert ok is False and why.startswith(f"fail {pf.UNFAITHFUL}: ") and "provably different" in why
