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

"""Tests for the Golden Set changelog gate.

A policy gate has two failure modes and they are not symmetric. A missed
violation lets an unexplained dataset change through, which is what the
gate exists to stop. A false positive is worse in practice, because a
gate that nags on reformatting gets marked non-required and then stops
catching anything at all. Both directions are tested here, and the
false-positive cases are deliberately the fiddly ones: reindentation,
key reordering, changelog-only edits, and the sixteen-case backfill.
"""

import json

import pytest

from gbench.changelog_gate import (
    DATASET_ROOT,
    Advisory,
    ChangeStatus,
    FileChange,
    GateReport,
    Rule,
    check_changes,
)


def case(version="1.0", changelog=None, prompt="What is 2 + 2?", **extra):
    """A minimal but realistically shaped golden case."""
    meta = {"version": version, "description": "a case"}
    if changelog is not None:
        meta["changelog"] = changelog
    body = {
        "id": "math_canonical",
        "title": "Math",
        "category": "reasoning",
        "conversation": {"messages": [{"role": "user", "content": prompt}]},
        "golden_truth": {"match_type": "contains_all", "expected_outputs": ["4"]},
        "meta": meta,
    }
    body.update(extra)
    return json.dumps(body, indent=2)


def entry(version, reason="because"):
    return {"version": version, "reason": reason}


def path_for(name="math_canonical.json"):
    return f"{DATASET_ROOT}/{name}"


def modified(before, after, name="math_canonical.json"):
    return FileChange(path_for(name), ChangeStatus.MODIFIED, before=before, after=after)


def added(after, name="math_canonical.json"):
    return FileChange(path_for(name), ChangeStatus.ADDED, after=after)


def rules(report):
    return [v.rule for v in report.violations]


# ----------------------------------------------------------------------
# The gate must not fire on changes the snapshot id would ignore
# ----------------------------------------------------------------------

def test_reindenting_a_case_is_not_a_change():
    """The single most important false positive to avoid.

    A gate that fires on whitespace is a gate someone switches off, and
    a switched-off gate enforces nothing. This is also the property that
    keeps the gate consistent with the hash, which ignores formatting.
    """
    body = json.loads(case())
    report = check_changes([modified(json.dumps(body, indent=2), json.dumps(body, indent=8))])
    assert report.ok
    assert report.unchanged == 1


def test_reordering_keys_is_not_a_change():
    body = json.loads(case())
    reordered = dict(reversed(list(body.items())))
    assert check_changes([modified(json.dumps(body), json.dumps(reordered))]).ok


def test_appending_a_changelog_entry_alone_is_not_a_change():
    """The bootstrap paradox. Writing down why must not itself demand a why."""
    before = case(version="1.1", changelog=[entry("1.0")])
    after = case(version="1.1", changelog=[entry("1.0"), entry("1.1")])
    assert check_changes([modified(before, after)]).ok


def test_the_sixteen_case_backfill_passes_untouched():
    """Seeding 1.0 onto every existing case alters no case content.

    If this failed, the contract could not be adopted without sixteen
    meaningless version bumps on its very first commit.
    """
    changes = [
        modified(
            case(version="1.0"),
            case(version="1.0", changelog=[entry("1.0", "seeded at contract adoption")]),
            name=f"case_{i}.json",
        )
        for i in range(16)
    ]
    report = check_changes(changes)
    assert report.ok
    assert report.unchanged == 16


def test_an_untouched_repo_passes():
    assert check_changes([]).ok


def test_changes_outside_the_dataset_are_ignored():
    report = check_changes([FileChange("gbench/cli.py", ChangeStatus.MODIFIED,
                                       before="a", after="b")])
    assert report.ok
    assert report.inspected == 0


def test_a_json_file_in_a_subdirectory_is_not_a_case():
    """Only files directly in the dataset root are cases."""
    change = FileChange(f"{DATASET_ROOT}/assets/manifest.json", ChangeStatus.MODIFIED,
                        before="{}", after='{"a": 1}')
    assert check_changes([change]).inspected == 0


# ----------------------------------------------------------------------
# The gate must fire when content actually moved
# ----------------------------------------------------------------------

def test_changing_a_prompt_without_a_bump_fails():
    report = check_changes([modified(case(), case(prompt="What is 3 + 3?"))])
    assert not report.ok
    assert rules(report) == [Rule.MISSING_BUMP]


def test_changing_the_expected_answer_without_a_bump_fails():
    before = case()
    after = json.loads(before)
    after["golden_truth"]["expected_outputs"] = ["5"]
    report = check_changes([modified(before, json.dumps(after))])
    assert rules(report) == [Rule.MISSING_BUMP]


def test_bumping_without_a_changelog_entry_fails():
    report = check_changes([modified(
        case(version="1.0", changelog=[entry("1.0")]),
        case(version="1.1", changelog=[entry("1.0")], prompt="different"),
    )])
    assert rules(report) == [Rule.MISSING_CHANGELOG_ENTRY]


def test_a_changelog_entry_with_an_empty_reason_fails():
    """An entry with no reason records that something changed, which the id already did."""
    report = check_changes([modified(
        case(version="1.0", changelog=[entry("1.0")]),
        case(version="1.1", changelog=[entry("1.0"), entry("1.1", "   ")], prompt="different"),
    )])
    assert rules(report) == [Rule.EMPTY_REASON]


def test_a_proper_bump_with_a_reason_passes():
    report = check_changes([modified(
        case(version="1.0", changelog=[entry("1.0")]),
        case(version="1.1",
             changelog=[entry("1.0"), entry("1.1", "answer_pattern now ignores working")],
             prompt="different"),
    )])
    assert report.ok
    assert report.inspected == 1
    assert report.unchanged == 0


def test_a_version_that_goes_backwards_fails():
    report = check_changes([modified(
        case(version="1.2", changelog=[entry("1.2")]),
        case(version="1.1", changelog=[entry("1.2"), entry("1.1")], prompt="different"),
    )])
    assert rules(report) == [Rule.VERSION_REGRESSED]


def test_a_non_numeric_version_still_has_to_move():
    """The gate insists a version changed. It does not legislate a scheme."""
    ok = check_changes([modified(
        case(version="2026-01-01", changelog=[entry("2026-01-01")]),
        case(version="2026-02-01",
             changelog=[entry("2026-01-01"), entry("2026-02-01")], prompt="different"),
    )])
    assert ok.ok

    bad = check_changes([modified(
        case(version="2026-01-01", changelog=[entry("2026-01-01")]),
        case(version="2026-01-01", changelog=[entry("2026-01-01")], prompt="different"),
    )])
    assert rules(bad) == [Rule.MISSING_BUMP]


def test_a_renamed_case_is_treated_as_modified():
    change = FileChange(path_for("renamed.json"), ChangeStatus.RENAMED,
                        before=case(), after=case(prompt="different"))
    assert rules(check_changes([change])) == [Rule.MISSING_BUMP]


def test_a_pure_rename_needs_no_bump():
    change = FileChange(path_for("renamed.json"), ChangeStatus.RENAMED,
                        before=case(), after=case())
    assert check_changes([change]).ok


# ----------------------------------------------------------------------
# Append-only history
# ----------------------------------------------------------------------

def test_editing_a_past_changelog_entry_fails():
    """History cannot be rewritten to make a past change look like it never happened."""
    report = check_changes([modified(
        case(version="1.1", changelog=[entry("1.0", "original"), entry("1.1")]),
        case(version="1.1", changelog=[entry("1.0", "rewritten"), entry("1.1")]),
    )])
    assert rules(report) == [Rule.REWRITTEN_HISTORY]


def test_deleting_a_past_changelog_entry_fails():
    report = check_changes([modified(
        case(version="1.1", changelog=[entry("1.0"), entry("1.1")]),
        case(version="1.1", changelog=[entry("1.1")]),
    )])
    assert rules(report) == [Rule.REWRITTEN_HISTORY]


def test_reformatting_a_past_changelog_entry_is_allowed():
    """Append-only is about content, not bytes, for the same reason the gate is."""
    report = check_changes([modified(
        case(version="1.0", changelog=[{"version": "1.0", "reason": "x"}]),
        case(version="1.0", changelog=[{"reason": "x", "version": "1.0"}]),
    )])
    assert report.ok


def test_history_rewriting_is_caught_even_when_content_did_not_move():
    """Otherwise the check would only run on PRs that changed a case anyway."""
    report = check_changes([modified(
        case(changelog=[entry("1.0", "original")]),
        case(changelog=[entry("1.0", "quietly reworded")]),
    )])
    assert not report.ok


# ----------------------------------------------------------------------
# Added and deleted cases
# ----------------------------------------------------------------------

def test_a_new_case_must_start_at_the_seed_version():
    report = check_changes([added(case(version="2.0", changelog=[entry("2.0")]))])
    assert rules(report) == [Rule.SEED_VERSION]


def test_a_new_case_at_the_seed_version_with_an_entry_passes():
    assert check_changes([added(case(version="1.0", changelog=[entry("1.0", "new case")]))]).ok


def test_a_new_case_without_a_changelog_entry_fails():
    assert rules(check_changes([added(case(version="1.0"))])) == [Rule.MISSING_CHANGELOG_ENTRY]


def test_a_new_case_without_a_meta_block_fails():
    body = json.loads(case())
    del body["meta"]
    assert rules(check_changes([added(json.dumps(body))])) == [Rule.MISSING_META]


def test_a_deleted_case_is_advised_not_failed():
    """There is no file left to carry an entry, so the rule is unenforceable.

    A gate that cannot be satisfied is a gate people learn to override,
    and an override habit costs more than this one unenforced case.
    """
    report = check_changes([FileChange(path_for(), ChangeStatus.DELETED, before=case())])
    assert report.ok
    assert len(report.advisories) == 1
    assert "deleted" in report.advisories[0].detail


# ----------------------------------------------------------------------
# Malformed input
# ----------------------------------------------------------------------

def test_a_case_that_stops_parsing_fails():
    assert rules(check_changes([modified(case(), "{not json")])) == [Rule.MALFORMED_JSON]


def test_a_case_that_was_already_broken_at_base_does_not_block_the_fix():
    """Failing here would block the very PR that repairs the file."""
    assert check_changes([modified("{not json", case())]).ok


def test_a_new_case_that_does_not_parse_fails():
    assert rules(check_changes([added("[]")])) == [Rule.MALFORMED_JSON]


# ----------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------

def test_swapping_an_asset_requires_the_referencing_case_to_bump():
    """The rule most likely to catch a real silent regression.

    Replacing a JPEG changes what the model is asked while leaving every
    case file byte-identical, so nothing else in the diff would show it.
    """
    asset = FileChange(f"{DATASET_ROOT}/assets/chart.png", ChangeStatus.MODIFIED,
                       before="old-bytes", after="new-bytes")
    dataset = {"vision_chart.json": case(prompt="Describe assets/chart.png")}
    report = check_changes([asset], dataset)
    assert rules(report) == [Rule.UNBUMPED_ASSET_CONSUMER]


def test_swapping_an_asset_passes_when_the_referencing_case_bumped():
    asset = FileChange(f"{DATASET_ROOT}/assets/chart.png", ChangeStatus.MODIFIED,
                       before="old-bytes", after="new-bytes")
    bumped = modified(
        case(version="1.0", changelog=[entry("1.0")], prompt="Describe assets/chart.png"),
        case(version="1.1", changelog=[entry("1.0"), entry("1.1", "new chart")],
             prompt="Describe assets/chart.png v2"),
        name="vision_chart.json",
    )
    dataset = {"vision_chart.json": bumped.after}
    assert check_changes([asset, bumped], dataset).ok


def test_an_asset_swap_does_not_implicate_cases_that_do_not_use_it():
    asset = FileChange(f"{DATASET_ROOT}/assets/chart.png", ChangeStatus.MODIFIED,
                       before="old", after="new")
    dataset = {"math_canonical.json": case(prompt="What is 2 + 2?")}
    assert check_changes([asset], dataset).ok


def test_a_case_may_reference_an_asset_by_bare_file_name():
    asset = FileChange(f"{DATASET_ROOT}/assets/audio/clip.wav", ChangeStatus.MODIFIED,
                       before="old", after="new")
    dataset = {"audio_case.json": case(prompt="Transcribe clip.wav")}
    assert rules(check_changes([asset], dataset)) == [Rule.UNBUMPED_ASSET_CONSUMER]


def test_the_asset_rule_is_skipped_rather_than_faked_without_a_dataset():
    asset = FileChange(f"{DATASET_ROOT}/assets/chart.png", ChangeStatus.MODIFIED,
                       before="old", after="new")
    assert check_changes([asset]).ok


def test_deleting_a_case_and_its_asset_together_does_not_fail():
    asset = FileChange(f"{DATASET_ROOT}/assets/chart.png", ChangeStatus.DELETED, before="old")
    gone = FileChange(path_for("vision_chart.json"), ChangeStatus.DELETED,
                      before=case(prompt="Describe assets/chart.png"))
    dataset = {"vision_chart.json": case(prompt="Describe assets/chart.png")}
    report = check_changes([asset, gone], dataset)
    assert report.ok


# ----------------------------------------------------------------------
# Report rendering, which is the only thing a failing contributor reads
# ----------------------------------------------------------------------

def test_a_failure_message_names_the_file_the_rule_and_the_fix():
    report = check_changes([modified(case(), case(prompt="different"))])
    rendered = report.render()
    assert "math_canonical.json" in rendered
    assert Rule.MISSING_BUMP.value in rendered
    assert "meta.changelog" in rendered
    assert "Reformatting alone" in rendered


def test_a_passing_message_says_how_much_was_actually_checked():
    """`passed` with nothing inspected reads the same as `passed` with real coverage."""
    body = json.loads(case())
    report = check_changes([modified(json.dumps(body), json.dumps(body, indent=8))])
    assert "1 changed case(s) checked" in report.render()
    assert "1 reformat-only" in report.render()


def test_advisories_are_rendered_even_when_the_gate_passes():
    report = check_changes([FileChange(path_for(), ChangeStatus.DELETED, before=case())])
    assert "For human review" in report.render()


def test_multiple_violations_are_all_reported_not_just_the_first():
    """Fixing one violation per CI round trip is how a gate becomes hated."""
    report = check_changes([
        modified(case(), case(prompt="a"), name="one.json"),
        modified(case(), case(prompt="b"), name="two.json"),
    ])
    assert len(report.violations) == 2


def test_an_empty_report_is_ok_by_construction():
    assert GateReport().ok
