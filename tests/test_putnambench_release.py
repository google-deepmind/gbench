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

"""putnam / putnam_formal on the pinned PutnamBench release, both Lean leaderboard settings.

Pure tests: no network, no Docker. The kernel cases (answer faithfulness in the "Lean no
answer" setting) are in tests/test_putnam_formal_verifier.py.
"""

import asyncio
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from gbench.runners.eval_suites import putnam as P
from gbench.runners.eval_suites import putnam_formal as pf

REPO = Path(__file__).resolve().parents[1]

# Release-format statements (import, open, placeholder, answer comment, docstring).
LEAST = ("import Mathlib\n\nabbrev putnam_test_r_solution : Nat := sorry\n-- 101\n"
         "/-- The least prime above 100. -/\n"
         "theorem putnam_test_r : IsLeast {p : Nat | p.Prime /\\ 100 < p} putnam_test_r_solution :=\n"
         "sorry\n")
MULTI = ("import Mathlib\n\nopen Nat\n\nabbrev putnam_test_c_solution : Prod Nat Nat := sorry\n"
         "-- (3, 1444)\ntheorem putnam_test_c\n: putnam_test_c_solution.1 = 3 /\\\n"
         "  putnam_test_c_solution.2 = 1444 :=\nsorry\n")
REAL = ("import Mathlib\n\nnoncomputable abbrev putnam_test_q_solution : Real := sorry\n-- 1/8\n"
        "theorem putnam_test_q : putnam_test_q_solution > 0 :=\n  sorry\n")
PROOF = ("import Mathlib\n\nopen Set\n\n/-- Proof only. -/\n"
         "theorem putnam_test_z (n : Nat) : n + 0 = n := by\n  sorry\n")


# --------------------------------------------------------------------------- #
# The two settings
# --------------------------------------------------------------------------- #
def test_with_answer_matches_the_release_rewrite_script():
    """Expected strings are the output of lean4/scripts/rewrite_solutions.py at the pinned
    commit on these inputs (including its trailing space after the type and its stripping of
    one pair of outer parentheses on first use)."""
    assert pf.with_answer_statement(MULTI) == (
        "import Mathlib\n\nopen Nat\n\n-- (3, 1444)\ntheorem putnam_test_c\n"
        ": ((3, 1444) : Prod Nat Nat ).1 = 3 /\\\n  ((3, 1444) : Prod Nat Nat ).2 = 1444 :=\n"
        "sorry\n")
    assert pf.with_answer_statement(REAL) == (
        "import Mathlib\n\n-- 1/8\ntheorem putnam_test_q : ((1/8) : Real ) > 0 :=\n  sorry\n")
    assert pf.with_answer_statement(LEAST) == (
        "import Mathlib\n\n-- 101\n/-- The least prime above 100. -/\ntheorem putnam_test_r : "
        "IsLeast {p : Nat | p.Prime /\\ 100 < p} ((101) : Nat ) :=\nsorry\n")
    assert pf.with_answer_statement(PROOF) == PROOF


def test_no_answer_prompt_drops_only_the_answer_comment():
    shown = pf.no_answer_statement(LEAST)
    assert "-- 101" not in shown and "101" not in shown.replace("100", "")
    assert shown == LEAST.replace("-- 101\n", "")
    assert pf.no_answer_statement(PROOF) == PROOF


def test_answer_slot_reads_the_placeholder_like_the_release_script():
    assert pf._answer_slot(REAL)[:3] == ("putnam_test_q_solution", "Real ", "1/8")
    assert pf._answer_slot(MULTI)[:3] == ("putnam_test_c_solution", "Prod Nat Nat ", "(3, 1444)")
    assert pf._answer_slot(PROOF) is None
    with pytest.raises(ValueError):
        pf._answer_slot("abbrev putnam_x_solution : Nat := sorry\ntheorem putnam_x : True :=\nsorry\n")


def test_reference_compiles_the_answer_next_to_the_placeholder():
    src = pf._reference_source(MULTI)
    lines = src.splitlines()
    i = lines.index("-- (3, 1444)")
    assert lines[i + 1] == ("noncomputable def putnam_test_c_solution.gbench_reference : "
                            "Prod Nat Nat  :=")
    assert lines[i + 2] == "  ((3, 1444) : Prod Nat Nat )"
    assert src.index("gbench_reference") < src.index("theorem putnam_test_c")
    assert pf._reference_source(PROOF) == PROOF


def test_split_statement_handles_the_release_forms():
    name, sig, pre, slot = pf._split_statement(REAL)
    assert (name, slot) == ("putnam_test_q", ("putnam_test_q_solution", "Real"))
    assert sig == "theorem putnam_test_q : putnam_test_q_solution > 0"
    # `theorem putnam_1993_b5:` (no space) and a `:= by sorry` ending both occur upstream.
    name, sig, _, _ = pf._split_statement("import Mathlib\ntheorem putnam_1993_b5:\n1 = 1 := by\n  sorry")
    assert name == "putnam_1993_b5" and sig == "theorem putnam_1993_b5:\n1 = 1"
    assert pf._split_statement(pf.with_answer_statement(REAL))[3] is None


def test_solution_source_uses_the_statement_preamble():
    full = pf._solution_source("import Mathlib\ntheorem putnam_test_c : True := trivial", MULTI)
    assert full.startswith("import Mathlib\nnoncomputable section\nopen Nat\n")
    assert full.count("import") == 1
    body = pf._solution_source("simp", PROOF)
    assert body.count("import Mathlib") == 1 and "open Set" in body
    assert "theorem putnam_test_z (n : Nat) : n + 0 = n := by\n  simp" in body


# --------------------------------------------------------------------------- #
# The pinned release
# --------------------------------------------------------------------------- #
def test_git_ids_match_git():
    """Values from `git hash-object` / `git write-tree` on the same two files."""
    files = {"a.lean": b"import Mathlib\n", "b.lean": b"theorem x : True := trivial\n"}
    assert pf._git_tree_sha1(files) == "81589d471535d7fe6f8d77137615ecc869a7a1f7"
    assert pf._git_blob_sha1(b"import Mathlib\n").hex() == "0b4be76b1d496db7a9ddbe3aba98a49478d54dfe"


def _tarball(src, informal):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        top = f"PutnamBench-{pf.REVISION}"
        entries = {f"{top}/lean4/src/{n}": d for n, d in src.items()}
        entries[f"{top}/informal/putnam.json"] = informal
        entries[f"{top}/lean4/lakefile.lean"] = b"-- not a statement\n"
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_release(monkeypatch, tmp_path):
    src = {"putnam_test_r.lean": LEAST.encode(), "putnam_test_z.lean": PROOF.encode()}
    informal = json.dumps([
        {"problem_name": "putnam_test_r", "informal_statement": "Least prime above 100?",
         "informal_solution": "Show that the answer is $101$.", "tags": ["number_theory"]},
        {"problem_name": "putnam_test_z", "informal_statement": "Show n + 0 = n.",
         "informal_solution": "None.", "tags": ["algebra"]},
    ]).encode()
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(pf, "LEAN_SRC_TREE", pf._git_tree_sha1(src))
    monkeypatch.setattr(pf, "INFORMAL_BLOB", pf._git_blob_sha1(informal).hex())
    monkeypatch.setattr(pf, "_RELEASE", {})
    urls = []

    def urlopen(url, timeout=None):
        urls.append(url)
        return _Resp(_tarball(src, informal))
    monkeypatch.setattr(pf.urllib.request, "urlopen", urlopen)
    return urls, tmp_path


def test_release_is_downloaded_at_the_pinned_commit_verified_and_cached(fake_release, monkeypatch):
    urls, root = fake_release
    rel = pf.load_putnambench()
    assert urls == [f"https://codeload.github.com/trishullab/PutnamBench/tar.gz/{pf.REVISION}"]
    assert set(rel["lean"]) == {"putnam_test_r", "putnam_test_z"}
    cached = root / "putnambench" / pf.REVISION
    assert (cached / "lean4" / "src" / "putnam_test_r.lean").read_text() == LEAST
    # Second process: served from the cache, re-verified, no network.
    monkeypatch.setattr(pf, "_RELEASE", {})
    monkeypatch.setattr(pf.urllib.request, "urlopen", lambda *a, **k: 1 / 0)
    assert pf.load_putnambench()["lean"]["putnam_test_r"] == LEAST
    # A modified cached statement is refused.
    (cached / "lean4" / "src" / "putnam_test_r.lean").write_text(LEAST.replace("101", "103"))
    monkeypatch.setattr(pf, "_RELEASE", {})
    with pytest.raises(RuntimeError, match="git tree"):
        pf.load_putnambench()


def test_release_with_other_content_is_refused(fake_release, monkeypatch):
    monkeypatch.setattr(pf, "LEAN_SRC_TREE", "0" * 40)
    with pytest.raises(RuntimeError, match="not the pinned"):
        pf.load_putnambench()


def test_pins_are_the_release_values():
    assert pf.REVISION == "97b5e9458b04e3d1044e8f8afe1724ff133f5a53"
    assert pf.LEAN_SRC_TREE == "f10ab3f77620e127ca58cae55924f6449da403ce"
    assert pf.INFORMAL_BLOB == "77675840ebadc2bc9b5a93f390bb0a678f33dbbc"
    assert pf.LEAN_TOOLCHAIN == "leanprover/lean4:v4.33.1"
    assert pf.MATHLIB_REV == "0df444a360eaa60ab8c11dca51a86af692955474"
    assert (pf.N_PROBLEMS, pf.N_ANSWER_PROBLEMS) == (672, 346)
    assert pf.DEFAULT_LEAN_IMAGE == "gbench-putnam-formal:97b5e945"


# --------------------------------------------------------------------------- #
# Samples and the report
# --------------------------------------------------------------------------- #
def test_loader_builds_both_settings_for_every_problem(fake_release, monkeypatch):
    monkeypatch.setattr(pf, "N_PROBLEMS", 2)
    samples = pf._load_putnam_formal_samples()
    assert [m["category"] for _, _, m in samples] == ["lean_with_answer"] * 2 + ["lean_no_answer"] * 2
    by = {(m["category"], pf._split_statement(g)[0]): (msgs[0]["content"], g)
          for msgs, g, m in samples}
    prompt, gold = by[("lean_with_answer", "putnam_test_r")]
    assert gold == pf.with_answer_statement(LEAST) and "abbrev" not in gold
    assert "((101) : Nat )" in prompt
    prompt, gold = by[("lean_no_answer", "putnam_test_r")]
    assert gold == LEAST                                   # the checker needs the answer
    assert "-- 101" not in prompt and "abbrev putnam_test_r_solution : Nat := sorry" in prompt
    assert "closed-form" in prompt and "putnam_test_r_solution" in prompt
    # A proof-only problem is the same task in both settings.
    assert by[("lean_with_answer", "putnam_test_z")] == by[("lean_no_answer", "putnam_test_z")]
    assert all(set(m) == {"category"} for _, _, m in samples)   # nothing leaks into the request


def test_limit_is_n_samples_over_both_settings_from_one_problem_order(fake_release, monkeypatch):
    """--eval-limit N is N samples for the suite (sampling.allocate_limit over the two settings),
    not N problems in each; the setting with fewer runs a subset of the other's problems."""
    monkeypatch.setattr(pf, "N_PROBLEMS", 2)
    samples = pf._load_putnam_formal_samples(limit=1)
    assert len(samples) == 1
    samples = pf._load_putnam_formal_samples(limit=2)
    names = [pf._split_statement(g)[0] for _, g, _ in samples]
    assert len(samples) == 2 and names[0] == names[1]
    assert {m["category"] for _, _, m in samples} == set(pf.SETTINGS)


def test_loader_refuses_a_release_with_another_problem_count(fake_release):
    with pytest.raises(RuntimeError, match="expected 672"):
        pf._load_putnam_formal_samples()


def _trace(cat, src, ok, gold=LEAST, why="ok"):
    return {"category": cat, "source_sample_idx": src, "is_correct": ok, "gold_answer": gold,
            "verification": why}


def test_report_counts_solved_problems_at_k_and_unfaithful_answers():
    wa, na = "lean_with_answer", "lean_no_answer"
    traces = [_trace(wa, 0, False), _trace(wa, 0, True), _trace(wa, 1, False), _trace(wa, 1, False),
              _trace(na, 2, False, why=f"fail {pf.UNFAITHFUL}: x is not provably equal"),
              _trace(na, 2, False, why="solution did not compile"),
              _trace(na, 3, True, gold=PROOF), _trace(na, 3, True, gold=PROOF)]
    rep = pf._settings_report(traces)
    assert rep[wa] == {"leaderboard_column": "Lean with answer", "results_key": "lean-wsolution",
                       "problems": 2, "problems_unscored": 0, "attempts_per_problem": 2,
                       "solved": 1, "solved_pct": 50.0, "accuracy": 25.0}
    assert rep[na]["results_key"] == "lean-nosolution" and rep[na]["solved"] == 1
    assert rep[na]["answer_placeholder_attempts"] == 2 and rep[na]["answer_not_faithful"] == 1


def _fake_suite(traces):
    def run(**kw):
        assert kw["supports_attempts"] is True and kw["limit"] is None
        return {"accuracy": 99.0, "sample_traces": traces}
    return run


def test_run_reports_both_settings_with_the_with_answer_headline(monkeypatch):
    monkeypatch.setattr(pf, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(pf, "_load_putnam_formal_samples", lambda limit=None: [])
    monkeypatch.setattr(pf, "_lean_image_provenance", lambda: {"reproducible": True})
    monkeypatch.setattr(pf, "N_PROBLEMS", 2)
    wa, na = "lean_with_answer", "lean_no_answer"
    traces = [_trace(wa, 0, True), _trace(wa, 1, False), _trace(na, 2, False), _trace(na, 3, False)]
    monkeypatch.setattr(pf, "run_eval_suite", _fake_suite(traces))
    out = pf.run_putnam_formal("m", "http://srv/v1", 1)
    assert out["accuracy"] == 50.0 and out["accuracy_lean_no_answer"] == 0.0
    assert out["putnambench_report"][wa]["solved"] == 1
    assert out["leaderboard_comparable"] is True and out["compute_budget"] == "pass@1"
    assert "with_reducible rfl" in out["answer_faithfulness"]
    assert out["dataset"].startswith("github.com/trishullab/PutnamBench@97b5e945")


def test_a_limited_run_is_not_leaderboard_comparable(monkeypatch):
    monkeypatch.setattr(pf, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(pf, "_load_putnam_formal_samples", lambda limit=None: [])
    monkeypatch.setattr(pf, "_lean_image_provenance", lambda: {})
    traces = [_trace("lean_with_answer", 0, True), _trace("lean_no_answer", 1, True)]
    monkeypatch.setattr(pf, "run_eval_suite", _fake_suite(traces))
    out = pf.run_putnam_formal("m", "http://srv/v1", 1, limit=1)
    assert out["leaderboard_comparable"] is False
    assert "1 of 672" in out["leaderboard_comparable_reason"]


def test_missing_prerequisites_hard_error(monkeypatch):
    monkeypatch.setattr(pf, "check_putnam_formal_prerequisites", lambda: (False, "no image"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*no image"):
        pf.run_putnam_formal("m", "http://srv/v1", 1)


# --------------------------------------------------------------------------- #
# Checker invocation and verdicts
# --------------------------------------------------------------------------- #
def _proc(rc, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def _capture_checker(monkeypatch, tmp_path, checker_result):
    ref = tmp_path / "ref.olean"
    ref.write_bytes(b"x")
    monkeypatch.setattr(pf, "_reference_olean", lambda s: ref)
    scripts = []

    def container(mounts, script, timeout, memory="4g"):
        scripts.append(script)
        if "--run" in script:
            return checker_result(script) if callable(checker_result) else checker_result
        (Path([h for h, g, _ in mounts if g == "/w/out"][0]) / "Solution.olean").write_bytes(b"s")
        return _proc(0)
    monkeypatch.setattr(pf, "_lean_container", container)
    return scripts


def _checker_ok(script):
    nonce = script.split("/w/ck/Checker.lean ")[1].split()[0]
    return _proc(0, f"GBENCH_CHECKER {nonce} start\nGBENCH_VERDICT {nonce} ok {{}}\n")


def test_no_answer_statements_send_the_placeholder_and_reference_answer(monkeypatch, tmp_path):
    scripts = _capture_checker(monkeypatch, tmp_path, _checker_ok)
    assert pf._verify("abbrev putnam_test_r_solution : Nat := 101\nsimp", LEAST) == (True, "ok")
    assert scripts[0].startswith("lake env lean -DautoImplicit=false ")
    assert scripts[1].endswith("putnam_test_r putnam_test_r_solution "
                               "putnam_test_r_solution.gbench_reference")
    scripts.clear()
    pf._verify("simp", pf.with_answer_statement(LEAST))
    assert scripts[1].endswith("/w/ref/Reference.olean putnam_test_r")


def test_an_indented_tactic_reply_keeps_its_layout(monkeypatch, tmp_path):
    """A body-only reply whose lines are all indented used to lose the first line's indent
    (the block was stripped before it was re-indented), so an honest proof did not parse."""
    ref = tmp_path / "ref.olean"
    ref.write_bytes(b"x")
    monkeypatch.setattr(pf, "_reference_olean", lambda s: ref)
    seen = []

    def container(mounts, script, timeout, memory="4g"):
        if "--run" not in script:
            src = [h for h, g, _ in mounts if g == "/w/in"][0]
            seen.append((Path(src) / "Solution.lean").read_text())
        return _proc(1)
    monkeypatch.setattr(pf, "_lean_container", container)
    reply = "```lean\n  intro p hp\n  exact hp.2\n```"
    pf._verify(reply, PROOF)
    assert seen[0].endswith(":= by\n  intro p hp\n  exact hp.2\n")


def test_a_checker_that_does_not_compile_is_a_harness_error(monkeypatch, tmp_path):
    _capture_checker(monkeypatch, tmp_path,
                     _proc(1, "/w/ck/Checker.lean:12:3: error: unknown identifier 'foo'\n"))
    ok, why = pf._verify("simp", PROOF)
    assert ok is False and why.startswith("harness error: the checker does not compile")


def test_a_checker_killed_for_memory_is_a_failed_proof(monkeypatch, tmp_path):
    _capture_checker(monkeypatch, tmp_path, _proc(137))
    assert pf._verify("simp", PROOF) == (False, "checker ran out of memory on this reply")


def test_checker_source_embeds_the_faithfulness_procedure():
    src = pf._CHECKER_LEAN
    assert "__TACTICS__" not in src and "__UNFAITHFUL__" not in src
    for tac in pf.FAITHFULNESS_TACTICS:
        assert f'"{tac}"' in src
    assert f"{pf.UNFAITHFUL}: " in src


# --------------------------------------------------------------------------- #
# Prerequisites: local image only, pinned toolchain
# --------------------------------------------------------------------------- #
def test_missing_image_names_the_build_command_and_is_never_pulled(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return _proc(1 if cmd[:3] == ["docker", "image", "inspect"] else 0)
    monkeypatch.setattr(pf.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(pf.subprocess, "run", run)
    ok, why = pf.check_putnam_formal_prerequisites()
    assert ok is False and pf.BUILD_COMMAND in why
    assert not any("pull" in c for c in calls)


def test_an_image_with_another_toolchain_is_refused(monkeypatch):
    monkeypatch.setattr(pf.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd, **kw: _proc(0))
    monkeypatch.setattr(pf, "_probe_image", lambda: ({
        "toolchain": "leanprover/lean4:v4.34.0-rc2", "mathlib_rev": "67874d61",
        "putnambench_revision": None}, ""))
    ok, why = pf.check_putnam_formal_prerequisites()
    assert ok is False and "v4.34.0-rc2" in why and "Rebuild" in why


def test_the_probe_reads_the_image_pins(monkeypatch):
    manifest = json.dumps({"packages": [{"name": "plausible", "rev": "b7"},
                                        {"name": "mathlib", "rev": pf.MATHLIB_REV}]})
    out = (f"TOOLCHAIN={pf.LEAN_TOOLCHAIN}\nREVISION={pf.REVISION}\nMANIFEST_BEGIN\n{manifest}\n"
           "MANIFEST_END\nMATHLIB_OK\n")
    monkeypatch.setattr(pf, "_lean_container", lambda m, s, t, memory="4g": _proc(0, out))
    facts, why = pf._probe_image()
    assert facts == {"toolchain": pf.LEAN_TOOLCHAIN, "putnambench_revision": pf.REVISION,
                     "mathlib_rev": pf.MATHLIB_REV} and why == ""


def test_dockerfile_builds_the_pinned_project():
    text = (REPO / "docker" / "putnam_formal.Dockerfile").read_text()
    assert "FROM leanprovercommunity/lean4@sha256:" in text
    for pin in (pf.REVISION, pf.LEAN_SRC_TREE, pf.LEAN_TOOLCHAIN, pf.MATHLIB_REV,
                "lake exe cache get", "lake build putnam_with_solutions",
                "scripts/rewrite_solutions.py", ".gbench-putnambench-revision"):
        assert pin in text, pin
    assert text.isascii()


# --------------------------------------------------------------------------- #
# The informal suite reads the same release
# --------------------------------------------------------------------------- #
def test_informal_loader_uses_the_pinned_release(fake_release):
    samples = P._load_putnam_samples()
    assert [g for _, g, _ in samples] == ["Show that the answer is $101$.", ""]
    assert [m for _, _, m in samples] == [{"category": "number_theory"}, {"category": "algebra"}]
    assert "Least prime above 100?" in samples[0][0][0]["content"]
    assert len(P._load_putnam_samples(limit=1)) == 1


def test_informal_prerequisites_accept_a_key_list(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    seen = []
    monkeypatch.setattr(P, "gemini_key_live_valid", lambda k: seen.append(k) or (True, ""))
    assert P.check_putnam_prerequisites() == (True, "")
    assert seen == ["k1,k2"]


# --------------------------------------------------------------------------- #
# Docs agree with the code
# --------------------------------------------------------------------------- #
def test_docs_state_the_pins_the_settings_and_the_run_command():
    doc = (REPO / "docs" / "evals" / "putnam_formal.md").read_text()
    for item in (pf.REVISION, pf.LEAN_TOOLCHAIN, pf.MATHLIB_REV, pf.BUILD_COMMAND,
                 "Lean with answer", "Lean no answer", "--max-output-tokens",
                 *pf.FAITHFULNESS_TACTICS):
        assert item in doc, item
    informal = (REPO / "docs" / "evals" / "putnam.md").read_text()
    assert pf.REVISION[:8] in informal and "--max-output-tokens" in informal
    assert doc.isascii() and informal.isascii()
