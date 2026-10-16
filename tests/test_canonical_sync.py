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

"""Tests for the canonical-sync provenance registry and its code-fingerprint gate.

The registry is what moves the evals scaffold_id when a suite's scorer, prompt or dataset changes
(scaffold.build_evals_scaffold folds in each suite's anchors). A registry nobody has to touch when
the code changes is a registry that drifts, so the gate below fails whenever a suite's code (or a
module several suites share) no longer matches the fingerprint recorded for it, and names what to
do about it.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

from gbench.runners.eval_suites import SUITES
from gbench.runners.eval_suites import canonical_sync as cs
from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC, canonical_sync_for

PKG = Path(cs.__file__).resolve().parent
ROOT = PKG.parents[2]


def test_known_suite_is_reconciled_with_required_fields():
    rec = canonical_sync_for("gaia2")
    assert rec["status"] == "reconciled"
    assert rec["synced"] == CANONICAL_SYNC["gaia2"]["synced"]
    assert rec["upstream"] and rec["method"]


def test_unknown_suite_is_honestly_unverified():
    rec = canonical_sync_for("definitely_not_a_suite")
    assert rec["status"] == "unverified"
    assert "not yet" in rec["note"]


def test_every_entry_has_a_date_and_method():
    for name, entry in CANONICAL_SYNC.items():
        assert entry.get("synced"), f"{name} missing synced date"
        assert entry.get("method"), f"{name} missing method"


def test_no_stale_registry_keys():
    # Every provenance entry must name a currently-registered suite (catch typos / removed suites).
    unknown = sorted(k for k in CANONICAL_SYNC if k not in SUITES)
    assert not unknown, f"canonical_sync.py names suites not in SUITES: {unknown}"


def test_every_registered_suite_is_reconciled():
    # Every built-in suite carries a provenance stamp. A newly added suite must get a
    # canonical_sync entry (do not invent a date/commit; reconcile it first) before it ships.
    missing = sorted(s for s in SUITES if s not in CANONICAL_SYNC and not _is_plugin(s))
    assert not missing, f"suites with no canonical_sync entry (reconcile then stamp): {missing}"


def _is_plugin(name):
    module = getattr(SUITES.get(name), "__module__", "") or ""
    return not module.startswith("gbench.runners.eval_suites.")


def test_stamp_shape_is_json_safe():
    # The stamp is attached to every result and serialized, so it must be plain dict/str.
    rec = canonical_sync_for("wildclawbench")
    assert isinstance(rec, dict) and all(isinstance(v, str) for v in rec.values())


def test_the_stamp_names_the_code_state_its_entry_describes():
    rec = canonical_sync_for("gsm8k")
    assert rec["code_fp"] == cs.CODE_FP["gsm8k"]
    assert re.fullmatch(r"[0-9a-f]{16}", rec["code_fp"])


# --------------------------------------------------------------------------- #
# Structured fields
# --------------------------------------------------------------------------- #
def test_every_entry_is_version_pinned_or_exempt_with_a_reason():
    unpinned = sorted(n for n, e in CANONICAL_SYNC.items()
                      if not cs.entry_pinned(e) and n not in cs.DATE_ONLY_EXEMPT)
    assert not unpinned, (f"entries with neither a pinned revision nor a checksum: {unpinned} "
                          "(pin the upstream commit / package version, or record a checksum)")
    for name, reason in cs.DATE_ONLY_EXEMPT.items():
        assert name in CANONICAL_SYNC and reason.strip()


def test_every_revision_and_checksum_leads_with_its_pin():
    bad = {n: cs.malformed_anchor_fields(e) for n, e in CANONICAL_SYNC.items()
           if cs.malformed_anchor_fields(e)}
    assert not bad, (f"structured fields that do not lead with a pin: {bad} "
                     "(revision: <pin>[ (qualifier)][, <pin> ...]; checksum: sha256:/git-blob:/"
                     "git-tree: first)")


@pytest.mark.parametrize("value,pinned", [
    ("1c3a" * 10, True),
    ("sha256:" + "ab" * 32, True),
    ("1.2.0", True),
    ("mini-swe-agent==2.4.2, swebench==5.0.2", True),
    ("pycocotools 2.0.11 (as verified)", False),
    ("main", False),
    ("2026-09-27", False),
])
def test_the_pin_predicate(value, pinned):
    assert cs.entry_pinned({"revision": value}) is pinned


def test_the_cli_reports_pins_with_the_same_predicate():
    from gbench.cli import provenance_pinned

    differ = sorted(n for n, e in CANONICAL_SYNC.items()
                    if provenance_pinned(e) != cs.entry_pinned(e))
    assert not differ, differ


def _module_pins(path):
    """Module-level 40-hex constants whose name says they pin a revision."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)
                and re.fullmatch(r"[0-9a-f]{40}", value.value)):
            continue
        for name in names:
            if re.search(r"(REVISION|COMMIT|_REF|_SHA)$", name):
                yield name, value.value


def test_every_pin_a_suite_holds_is_in_its_entry():
    """A suite that pins a commit in code records the full commit in its entry's anchors:
    revision, dataset or checksum (the fields the scaffold reads), so repinning any of them,
    a secondary harness's included, moves the suite's scaffold id."""
    missing = []
    for name, entry in sorted(CANONICAL_SYNC.items()):
        module = PKG / f"{name}.py"
        if not module.exists():
            continue
        recorded = " ".join(str(entry.get(f, "")) for f in ("revision", "dataset", "checksum"))
        missing += [f"{name}.{const}={sha}" for const, sha in _module_pins(module)
                    if sha not in recorded]
    assert not missing, f"pins held in code but not recorded in the entry: {missing}"


def test_browsecomp_records_the_dataset_revision_its_loader_reads():
    from gbench.runners.eval_suites import browsecomp

    assert CANONICAL_SYNC["browsecomp"]["revision"] == browsecomp._REVISION
    assert CANONICAL_SYNC["browsecomp"]["checksum"].startswith("sha256:7b24471c")


@pytest.mark.parametrize("suite", ["browsecomp", "simpleqa", "healthbench", "frames", "gdpval"])
def test_the_simple_evals_suites_are_stamped_after_their_protocol_changes(suite):
    """Their sampling, judge budget and exclusions changed on 2026-09-27; the stamps say so."""
    assert CANONICAL_SYNC[suite]["synced"] >= "2026-09-27"


# --------------------------------------------------------------------------- #
# The code-fingerprint gate
# --------------------------------------------------------------------------- #
def test_code_fingerprints_current():
    stale = cs.stale_code_fingerprints()
    assert not stale, (
        "Code fingerprints no longer match the code: "
        + ", ".join(f"{n} (recorded {old}, now {new or 'removed'})"
                    for n, (old, new) in sorted(stale.items()))
        + ". " + cs.GATE_STEPS)


def test_the_recorded_dependency_table_is_current():
    assert cs.DEPENDS_ON == cs.dependency_table(), (
        "DEPENDS_ON is stale: run python gbench/runners/eval_suites/canonical_sync.py "
        "--refresh-code-fp")


def test_every_module_several_suites_share_has_its_own_entry():
    """Otherwise one edit would have to be re-stamped once per importing suite."""
    candidates = cs._Package().shared_candidates()
    assert not candidates, (f"modules owned by several suites, add a SHARED_SYNC entry: "
                            f"{candidates}")


def test_shared_entries_name_modules_and_carry_a_date_and_method():
    for name, entry in cs.SHARED_SYNC.items():
        assert (PKG / f"{name}.py").exists(), name
        assert name not in CANONICAL_SYNC, name
        assert cs.SYNCED_RE.match(entry["synced"]) and entry["method"], name


def test_verdict_preserving_declarations_are_well_formed():
    for name, decl in cs.VERDICT_PRESERVING.items():
        assert cs.registry_entry(name) is not None, name
        assert re.fullmatch(r"[0-9a-f]{16}", decl.get("code_fp", "")), name
        assert str(decl.get("reason", "")).strip(), name


def test_every_suite_depends_on_the_engine_it_runs_through():
    """Nearly every suite runs through base; its entry must reach their scaffold signal."""
    through_base = [n for n in CANONICAL_SYNC if "base" in cs.DEPENDS_ON.get(n, [])]
    assert len(through_base) >= 70
    signal = cs.scoring_signal(["gsm8k"])
    assert signal["depends_on"]["base"] == cs.anchors(cs.SHARED_SYNC["base"])


def test_the_maintainer_plugin_module_is_never_read():
    pkg = cs._Package()
    for name in pkg.boundaries:
        assert not any(p.stem == "swe_bench_pro_agentic" for p in pkg.files(name)), name


def test_the_gate_runs_as_a_script_with_nothing_but_the_stdlib():
    done = subprocess.run(
        ["python3", "-S", "-c",
         "import runpy, sys; sys.argv = ['canonical_sync.py', '--check']; "
         f"runpy.run_path({str(PKG / 'canonical_sync.py')!r}, run_name='__main__')"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "current" in done.stdout


def test_the_bump_check_accepts_the_gate_adoption_against_the_last_commit():
    """HEAD's registry records no fingerprints, so every fingerprint here is a baseline."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:gbench/runners/eval_suites/canonical_sync.py"],
        capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip("the registry is not committed at HEAD")
    base = cs._load_version(shown.stdout)
    here = cs._load_version((PKG / "canonical_sync.py").read_text(encoding="utf-8"))
    if base.get("CODE_FP"):
        pytest.skip("HEAD already records fingerprints; the scratch-tree tests cover the rule")
    assert cs.bump_violations(base, here) == []
