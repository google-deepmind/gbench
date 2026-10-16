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

"""Every run-shaping knob moves the evals scaffold_id, and a refresh cannot hide a code change.

Two halves. The first drives build_evals_scaffold end to end: each declared run-shaping variable
(and each deprecated alias) set for the suite it belongs to gives another scaffold id, the Gemini
cascade variables move it only through the cascade they resolve to, and every GBENCH_* variable a
suite's own code, vendored code or container reads is classified. The second checks the
fingerprint refresh on a scratch copy of canonical_sync.py: a moved fingerprint with no decision
is refused and nothing is written; an anchor bumped since the last refresh or a
VERDICT_PRESERVING declaration of the new fingerprint lets it through.

No model, no network, no docker: the scaffold reads package metadata only, and the refresh
writes a copy under tmp_path.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

from gbench.core import scaffold
from gbench.core.scaffold import ContractField, build_evals_scaffold
from gbench.runners import eval_suites as E
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import canonical_sync as cs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scaffold import _Config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SUITES_DIR = REPO / "gbench" / "runners" / "eval_suites"
SCORING = ContractField.SCORING_VERSION.value

#: Maintainer-owned plugin modules: never read.
_PLUGIN_FILES = {"swe_bench_pro_agentic.py", "jetski_agent.py", "slashbin_agent.py"}

#: The variables that choose the Gemini cascades: pinned as the cascade they resolve to.
_CASCADE_ENV = sorted(scaffold._CASCADE_ENV)


def _all_knob_names():
    names = set(base.RUN_SHAPING_GLOBAL_ENV)
    for knobs in base.RUN_SHAPING_SUITE_ENV.values():
        names |= set(knobs)
    for aliases in base.RUN_SHAPING_ALIASES.values():
        names |= set(aliases)
    for suite in set(E.SUITES) | set(base.RENAMED_SUITES):
        names.add(base.temperature_env_var(suite))
        names |= set(base.legacy_temperature_env_vars(suite))
    return sorted(names | {"GBENCH_SHARD", "GBENCH_EVAL_PLUGINS_PATH"})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _all_knob_names():
        monkeypatch.delenv(name, raising=False)
    scaffold._gpu_models.cache_clear()
    yield
    scaffold._gpu_models.cache_clear()


def _sid(suite):
    return build_evals_scaffold(_Config(evals=[suite])).scaffold_id


# --------------------------------------------------------------------------- #
# Run-shaping variables, end to end through the scaffold
# --------------------------------------------------------------------------- #
_SUITE_KNOBS = sorted((suite, knob) for suite, knobs in base.RUN_SHAPING_SUITE_ENV.items()
                      for knob in knobs)
_GLOBAL_KNOBS = sorted(k for k in base.RUN_SHAPING_GLOBAL_ENV if k not in scaffold._CASCADE_ENV)


@pytest.mark.parametrize("suite,knob", _SUITE_KNOBS)
def test_every_declared_suite_knob_moves_the_evals_scaffold_id(monkeypatch, suite, knob):
    before = _sid(suite)
    monkeypatch.setenv(knob, "probe-value")
    assert _sid(suite) != before, (suite, knob)


@pytest.mark.parametrize("knob", _GLOBAL_KNOBS)
def test_every_declared_global_knob_moves_the_evals_scaffold_id(monkeypatch, knob):
    before = _sid("gsm8k")
    monkeypatch.setenv(knob, "probe-value")
    assert _sid("gsm8k") != before, knob


def test_a_suite_temperature_moves_the_id():
    suite = "gsm8k"
    with pytest.MonkeyPatch.context() as mp:
        before = _sid(suite)
        mp.setenv(base.temperature_env_var(suite), "0.7")
        assert _sid(suite) != before


@pytest.mark.parametrize("canonical,alias", sorted(
    (c, a) for c, aliases in base.RUN_SHAPING_ALIASES.items() for a in aliases))
def test_a_deprecated_alias_moves_the_id_as_its_canonical_name_does(monkeypatch, canonical, alias):
    suite = next(s for s, knobs in sorted(base.RUN_SHAPING_SUITE_ENV.items()) if canonical in knobs)
    before = _sid(suite)
    monkeypatch.setenv(alias, "probe-value")
    via_alias = _sid(suite)
    assert via_alias != before
    monkeypatch.delenv(alias)
    monkeypatch.setenv(canonical, "probe-value")
    assert _sid(suite) == via_alias


@pytest.mark.parametrize("knob", _CASCADE_ENV)
def test_a_cascade_variable_moves_the_id_through_the_cascade_it_resolves(monkeypatch, knob):
    before = build_evals_scaffold(_Config(evals=["gsm8k"]))
    monkeypatch.setenv(knob, "gemini-probe")
    after = build_evals_scaffold(_Config(evals=["gsm8k"]))
    assert after.scaffold_id != before.scaffold_id
    assert knob not in after.pinned[SCORING]["env"]
    judge = after.pinned[SCORING]["judge"]
    assert "gemini-probe" in judge["search_cascade"]


def test_naming_the_default_cascade_is_the_same_grading(monkeypatch):
    before = _sid("gsm8k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", ",".join(base.GEMINI_CASCADE))
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", ",".join(base.GEMINI_CASCADE))
    assert _sid("gsm8k") == before


#: The measurement-changing knobs found missing from the scaffold: each stays declared for its
#: suite and moves that suite's evals scaffold id.
_NAMED = {
    "terminal_bench": ["GBENCH_TB_PARSER", "GBENCH_TB_TIMEOUT_MULTIPLIER",
                       "GBENCH_TB_TURN_MAX_TOKENS"],
    "tau2": ["GBENCH_TAU2_TOP_P", "GBENCH_TAU2_TOP_K", "GBENCH_TAU2_SEED",
             "GBENCH_TAU2_MAX_ERRORS", "GBENCH_TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM"],
    "tau3": ["GBENCH_TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM"],
    "toolbench": ["GBENCH_TOOLBENCH_METHOD", "GBENCH_TOOLBENCH_REFERENCE",
                  "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_TOOLBENCH_GROUPS",
                  "GBENCH_TOOLBENCH_MAX_QUERY_COUNT"],
    "wildclawbench": ["GBENCH_WILDCLAWBENCH_THINKING", "GBENCH_WILDCLAWBENCH_TASK"],
    "skillsbench": ["GBENCH_SKILLSBENCH_AGENT", "GBENCH_SKILLSBENCH_CONDITIONS"],
    "scicode": ["GBENCH_SCICODE_WITH_BACKGROUND"],
    "ui_control_osworld": ["GBENCH_OSWORLD_OBS_TYPE", "GBENCH_OSWORLD_LIMIT"],
    "lcb": ["GBENCH_LCB_MIN_DATE", "GBENCH_LCB_INCLUDE_TEST_GEN",
            "GBENCH_LCB_INCLUDE_SELF_REPAIR"],
    "ruler": ["GBENCH_RULER_LENGTHS"],
    "mcp_bench": ["GBENCH_MCP_BENCH_DISTRACTION"],
    "gaia2": ["GBENCH_GAIA2_NUM_RUNS"],
    "agent_dojo": ["GBENCH_AGENTDOJO_ATTACK"],
    "gaia": ["GBENCH_SEARCH_MAX_RESULTS"],
    "bfcl_v4_agentic": ["GBENCH_BFCL_ENABLE_THINKING"],
    "bfcl": ["GBENCH_BFCL_ENABLE_THINKING"],
}


@pytest.mark.parametrize("suite,knob", sorted((s, k) for s, ks in _NAMED.items() for k in ks))
def test_every_named_run_shaping_knob_moves_the_scaffold_id(monkeypatch, suite, knob):
    declared = set(base.RUN_SHAPING_GLOBAL_ENV) | set(base.RUN_SHAPING_SUITE_ENV.get(suite, {}))
    assert knob in declared, (suite, knob)
    before = _sid(suite)
    monkeypatch.setenv(knob, "probe-value")
    assert _sid(suite) != before


def test_a_scoring_class_knob_lands_in_the_scoring_version():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GBENCH_TAU2_EVAL_LLM", "gemini-probe")
        sc = build_evals_scaffold(_Config(evals=["tau2"]))
    assert sc.pinned[SCORING]["env"] == {"GBENCH_TAU2_EVAL_LLM": "gemini-probe"}


# --------------------------------------------------------------------------- #
# Every variable a suite's code or container reads is classified
# --------------------------------------------------------------------------- #
_FULL_NAME = re.compile(r"^GBENCH_[A-Z0-9_]*[A-Z0-9]$")
_SHELL_READ = re.compile(r"\$\{?(GBENCH_[A-Z0-9_]*[A-Z0-9])\b")
_DOCKER_DECL = re.compile(r"^\s*(?:ENV|ARG)\s+(GBENCH_[A-Z0-9_]*[A-Z0-9])", re.M)


def _skipped(path):
    return (".bak" in path.name or "__pycache__" in path.parts or path.name in _PLUGIN_FILES
            or path.name == "canonical_sync.py" or path.suffix in (".pyc", ".orig", ".rej"))


def _env_names_read(path):
    """GBENCH_* names a file reads: a Python string constant that is exactly the name (an env
    key, not prose), a shell expansion, or a Dockerfile ENV / ARG."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return set()
    if path.suffix == ".py":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                return set()
        return {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and _FULL_NAME.match(n.value)}
    return set(_SHELL_READ.findall(text)) | set(_DOCKER_DECL.findall(text))


def test_every_variable_the_suites_code_and_containers_read_is_classified():
    """eval_suites (vendored subpackages included), gbench/docker and docker/: every GBENCH_*
    name read there is run-shaping (declared, an alias, or a suite temperature) or listed in
    base.NON_RUN_SHAPING_ENV with its reason."""
    classified = set(_all_knob_names()) | set(base.NON_RUN_SHAPING_ENV)
    roots = [SUITES_DIR, REPO / "gbench" / "docker", REPO / "docker"]
    unclassified = {}
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and not _skipped(p)):
            for name in _env_names_read(path) - classified:
                unclassified.setdefault(name, []).append(path.relative_to(REPO).as_posix())
    assert not unclassified, unclassified


# --------------------------------------------------------------------------- #
# The refresh records a moved fingerprint only after its decision
# --------------------------------------------------------------------------- #
def _scratch(tmp_path, code_fp, fp_anchors, declared=None):
    """A copy of canonical_sync.py whose generated block records `code_fp` / `fp_anchors`, and
    whose VERDICT_PRESERVING is `declared`."""
    text = Path(cs.__file__).read_text(encoding="utf-8")
    start = text.index("\n" + cs._BEGIN + "\n") + 1
    end = text.index("\n" + cs._END + "\n", start) + 1 + len(cs._END)
    text = text[:start] + cs._generated_block(code_fp, {}, fp_anchors) + text[end:]
    head = "VERDICT_PRESERVING: Dict[str, Dict[str, str]] = {"
    open_at = text.index(head)
    close_at = text.index("\n}\n", open_at) + 1
    body = "".join(f"    {k!r}: {v!r},\n" for k, v in (declared or {}).items())
    text = text[:open_at] + head + "\n" + body + text[close_at:]
    path = tmp_path / "canonical_sync.py"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def now_fp(monkeypatch):
    """The fingerprints the refresh computes, set per test."""
    current = {}
    monkeypatch.setattr(cs, "code_fingerprints", lambda pkg=None: dict(current))
    monkeypatch.setattr(cs, "dependency_table", lambda pkg=None: {})
    return current


_GSM = cs.anchor_digest(cs.CANONICAL_SYNC["gsm8k"])


def test_a_moved_fingerprint_with_no_decision_is_refused_and_nothing_is_written(tmp_path, now_fp):
    path = _scratch(tmp_path, {"gsm8k": "a" * 16}, {"gsm8k": _GSM})
    before = path.read_text(encoding="utf-8")
    now_fp["gsm8k"] = "b" * 16
    with pytest.raises(cs.UndecidedCodeChange) as err:
        cs.refresh_code_fingerprints(path)
    assert err.value.names == ["gsm8k"] and "gsm8k" in str(err.value)
    assert path.read_text(encoding="utf-8") == before


def test_an_anchor_bumped_since_the_last_refresh_is_the_decision(tmp_path, now_fp):
    path = _scratch(tmp_path, {"gsm8k": "a" * 16}, {"gsm8k": "0" * 16})
    now_fp["gsm8k"] = "b" * 16
    assert cs.refresh_code_fingerprints(path) == {"gsm8k": ("a" * 16, "b" * 16)}
    ns = cs._load_version(path.read_text(encoding="utf-8"))
    assert ns["CODE_FP"] == {"gsm8k": "b" * 16}
    # the new digest is recorded: the next change needs a decision of its own
    assert ns["FP_ANCHORS"] == {"gsm8k": _GSM}
    now_fp["gsm8k"] = "c" * 16
    with pytest.raises(cs.UndecidedCodeChange):
        cs.refresh_code_fingerprints(path)


def test_a_declaration_of_exactly_the_new_fingerprint_is_the_decision(tmp_path, now_fp):
    ok = {"gsm8k": {"code_fp": "b" * 16, "reason": "renamed a local variable"}}
    old = {"gsm8k": {"code_fp": "c" * 16, "reason": "an older refactor"}}
    blank = {"gsm8k": {"code_fp": "b" * 16, "reason": " "}}
    now_fp["gsm8k"] = "b" * 16
    for declared, passes in ((ok, True), (old, False), (blank, False)):
        path = _scratch(tmp_path, {"gsm8k": "a" * 16}, {"gsm8k": _GSM}, declared)
        if passes:
            cs.refresh_code_fingerprints(path)
        else:
            with pytest.raises(cs.UndecidedCodeChange):
                cs.refresh_code_fingerprints(path)


def test_a_new_entry_is_adopted_but_a_moved_fingerprint_without_a_digest_is_undecided(
        tmp_path, now_fp):
    # every recorded fingerprint carries an FP_ANCHORS digest, so a missing one is never
    # evidence of a bump: the moved gsm8k fingerprint needs a decision; a new entry does not
    now_fp.update({"gsm8k": "b" * 16, "mmlu": "d" * 16})
    path = _scratch(tmp_path, {"gsm8k": "a" * 16}, {})
    with pytest.raises(cs.UndecidedCodeChange) as err:
        cs.refresh_code_fingerprints(path)
    assert err.value.names == ["gsm8k"]
    now_fp.update({"gsm8k": "a" * 16})
    cs.refresh_code_fingerprints(path)
    ns = cs._load_version(path.read_text(encoding="utf-8"))
    assert set(ns["FP_ANCHORS"]) == {"gsm8k", "mmlu"}


def test_an_unmoved_fingerprint_refreshes_its_digest_without_a_decision(tmp_path, now_fp):
    now_fp["gsm8k"] = "a" * 16
    path = _scratch(tmp_path, {"gsm8k": "a" * 16}, {"gsm8k": "0" * 16})
    assert cs.refresh_code_fingerprints(path) == {}
    assert cs._load_version(path.read_text(encoding="utf-8"))["FP_ANCHORS"] == {"gsm8k": _GSM}


def test_the_cli_refresh_reports_a_refusal_and_fails(monkeypatch, capsys):
    def refuse(path=None):
        raise cs.UndecidedCodeChange(["gsm8k"], {"gsm8k": ("a" * 16, "b" * 16)})
    monkeypatch.setattr(cs, "refresh_code_fingerprints", refuse)
    assert cs.main(["--refresh-code-fp"]) == 1
    out = capsys.readouterr().out
    assert "gsm8k" in out and "Nothing was written" in out and "VERDICT_PRESERVING" in out


def test_the_check_fails_while_an_anchor_moved_after_the_last_refresh(monkeypatch, capsys):
    monkeypatch.setattr(cs, "stale_code_fingerprints", lambda pkg=None: {})
    monkeypatch.setitem(cs.FP_ANCHORS, "gsm8k", "0" * 16)
    assert cs.main(["--check"]) == 1
    assert "gsm8k" in capsys.readouterr().out


def test_every_recorded_fingerprint_carries_the_digest_of_its_current_anchors():
    assert set(cs.FP_ANCHORS) == set(cs.CODE_FP)
    stale = cs.stale_anchor_digests()
    assert not stale, (
        "anchors changed after the fingerprints were recorded: " + ", ".join(sorted(stale))
        + "; run python gbench/runners/eval_suites/canonical_sync.py --refresh-code-fp")


def test_every_synced_anchor_is_a_date_with_an_optional_same_day_serial():
    """A second verdict-relevant reconcile on the day of the last bump appends .2, .3, ...:
    a date alone could not move the anchor, the refresh would refuse and the id would hold."""
    for name, entry in {**cs.CANONICAL_SYNC, **cs.SHARED_SYNC}.items():
        assert cs.SYNCED_RE.match(str(entry.get("synced", ""))), name
    assert cs.SYNCED_RE.match("2026-09-28.2") and not cs.SYNCED_RE.match("2026-09-28.0")
    before = cs.anchor_digest({"synced": "2026-09-28"})
    assert cs.anchor_digest({"synced": "2026-09-28.2"}) != before
