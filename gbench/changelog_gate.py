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

"""Enforcement for the Golden Set changelog rule.

The scaffold contract in :mod:`gbench.core.scaffold` makes a dataset
change *visible*, because the snapshot id moves. It does not make the
change *explained*. Six months later a reviewer looking at a score
discontinuity can see that `gd_661165ef` became `gd_9c1e04b2` and still
have no idea why, which is most of the way to no information at all.

So a change to a case must come with two things: a bump to
``meta.version``, and an appended ``meta.changelog`` entry saying why.
This module decides whether a diff satisfies that, and it is separate
from the CLI in ``scripts/check_golden_changelog.py`` so the rules can
be unit tested without a git repository.

Three properties are worth stating up front, because each one exists to
stop a specific way this kind of gate dies.

**The gate compares canonical content, not bytes.** It reads
:func:`gbench.canonical.canonical_json`, the same function the snapshot
hash reads. That is what makes the gate and the contract provably
consistent: if a change would not move the id, it does not trip the
gate. A gate that fires on reindentation gets switched off within a
month, and then it is enforcing nothing.

**The changelog itself is excluded from the comparison.** Without that
the rule is circular, because appending the entry that explains a bump
is itself a content change, which would demand another bump, which would
demand another entry. It also makes the initial backfill possible:
seeding a ``1.0`` entry onto all sixteen existing cases alters no case
content and so passes untouched.

**The changelog is append-only.** Existing entries may not be edited or
removed, so history cannot be quietly rewritten to make a past change
look like it never happened.

Deletions are reported rather than failed. There is no file left to
carry a changelog entry, so the rule is unenforceable by construction,
and a gate that cannot be satisfied is a gate people learn to override.

This module sits at the top of the package rather than under
``gbench.core`` because ``gbench/core/__init__.py`` imports the model
registry, which reaches Hugging Face at import time. The gate reads JSON
text out of a git diff and has no business touching the network, so it
takes nothing heavier than :mod:`gbench.canonical`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .canonical import CHANGELOG_KEY, JSONValue, canonical_json, strip_changelog

# The version every newly added case starts at. Fixed rather than free
# so that "when did this case enter the set" is answerable by looking
# for a 1.0 entry, in every case, without exception.
SEED_VERSION = "1.0"

DATASET_ROOT = "gbench/golden_dataset"
ASSETS_SUBDIR = "assets"


class ChangeStatus(str, Enum):
    """Git's diff statuses, narrowed to the four that reach the rules."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class Rule(str, Enum):
    """Named so a failure message says which rule fired, and tests can assert on it."""

    MALFORMED_JSON = "malformed-json"
    MISSING_META = "missing-meta"
    MISSING_VERSION = "missing-version"
    SEED_VERSION = "seed-version"
    MISSING_BUMP = "missing-bump"
    VERSION_REGRESSED = "version-regressed"
    MISSING_CHANGELOG_ENTRY = "missing-changelog-entry"
    EMPTY_REASON = "empty-reason"
    REWRITTEN_HISTORY = "rewritten-history"
    UNBUMPED_ASSET_CONSUMER = "unbumped-asset-consumer"


@dataclass(frozen=True)
class FileChange:
    """One entry from ``git diff --name-status``, with both sides resolved.

    ``before`` is None for an addition and ``after`` is None for a
    deletion. Both are the raw file text rather than parsed JSON,
    because a case that stopped parsing is itself something the gate has
    to be able to report.
    """

    path: str
    status: ChangeStatus
    before: Optional[str] = None
    after: Optional[str] = None

    @property
    def name(self) -> str:
        """Trailing path component, which is how cases are named in reports."""
        return self.path.rsplit("/", 1)[-1]

    @property
    def is_case(self) -> bool:
        """A case is a JSON file directly in the dataset root, not under assets/."""
        return (
            self.path.startswith(f"{DATASET_ROOT}/")
            and self.path.endswith(".json")
            and "/" not in self.path[len(DATASET_ROOT) + 1:]
        )

    @property
    def is_asset(self) -> bool:
        return self.path.startswith(f"{DATASET_ROOT}/{ASSETS_SUBDIR}/")


@dataclass(frozen=True)
class Violation:
    """A rule the diff broke. Any violation fails the gate."""

    path: str
    rule: Rule
    detail: str

    def render(self) -> str:
        return f"  {self.path}\n    [{self.rule.value}] {self.detail}"


@dataclass(frozen=True)
class Advisory:
    """Something a human should look at, which the gate cannot decide."""

    path: str
    detail: str

    def render(self) -> str:
        return f"  {self.path}\n    {self.detail}"


@dataclass(frozen=True)
class GateReport:
    violations: Tuple[Violation, ...] = ()
    advisories: Tuple[Advisory, ...] = ()
    inspected: int = 0
    unchanged: int = 0

    @property
    def ok(self) -> bool:
        """Advisories never fail the gate. Only violations do."""
        return not self.violations

    def render(self) -> str:
        lines: List[str] = []
        if self.violations:
            lines.append(f"Golden changelog gate FAILED with {len(self.violations)} violation(s):")
            lines.extend(v.render() for v in self.violations)
            lines.append("")
            lines.append(
                "A case whose content changed needs a `meta.version` bump and an appended\n"
                "`meta.changelog` entry for the new version with a non-empty `reason`.\n"
                "Reformatting alone does not count as a content change and needs neither."
            )
        else:
            lines.append(
                f"Golden changelog gate passed. {self.inspected} changed case(s) checked, "
                f"{self.unchanged} reformat-only."
            )
        if self.advisories:
            lines.append("")
            lines.append("For human review, not blocking:")
            lines.extend(a.render() for a in self.advisories)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Version and changelog helpers
# ----------------------------------------------------------------------

def _parse_version(value: object) -> Optional[Tuple[int, ...]]:
    """Parse a dotted numeric version, or None if it is not one.

    Non-numeric versions are allowed rather than rejected, because the
    gate's job is to insist a version *moved*, not to legislate a
    numbering scheme. They simply lose the regression check.
    """
    if not isinstance(value, str):
        return None
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return None


def _is_forward(before: object, after: object) -> bool:
    """True when the version moved, and moved forward if that is decidable."""
    if before == after:
        return False
    parsed_before, parsed_after = _parse_version(before), _parse_version(after)
    if parsed_before is None or parsed_after is None:
        return True
    return parsed_after > parsed_before


def _meta(case: Mapping[str, JSONValue]) -> Optional[Mapping[str, JSONValue]]:
    meta = case.get("meta")
    return meta if isinstance(meta, dict) else None


def _changelog(case: Mapping[str, JSONValue]) -> List[JSONValue]:
    meta = _meta(case)
    if meta is None:
        return []
    entries = meta.get(CHANGELOG_KEY)
    return list(entries) if isinstance(entries, list) else []


def _entry_for(case: Mapping[str, JSONValue], version: object) -> Optional[Mapping[str, JSONValue]]:
    for entry in _changelog(case):
        if isinstance(entry, dict) and entry.get("version") == version:
            return entry
    return None


def _content_changed(before: Mapping[str, JSONValue], after: Mapping[str, JSONValue]) -> bool:
    """Canonical comparison with the changelog stripped, matching the hash exactly."""
    return canonical_json(strip_changelog(dict(before))) != canonical_json(
        strip_changelog(dict(after))
    )


# ----------------------------------------------------------------------
# Per-case rules
# ----------------------------------------------------------------------

def _require_logged_version(
    path: str,
    case: Mapping[str, JSONValue],
    version: object,
) -> List[Violation]:
    """A version is only logged if an entry names it and gives a real reason."""
    entry = _entry_for(case, version)
    if entry is None:
        return [Violation(
            path,
            Rule.MISSING_CHANGELOG_ENTRY,
            f"meta.version is {version!r} but meta.changelog has no entry for it. "
            f"Append {{\"version\": {version!r}, \"reason\": \"...\"}}.",
        )]
    reason = entry.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return [Violation(
            path,
            Rule.EMPTY_REASON,
            f"the changelog entry for {version!r} has no reason. An entry without a "
            "reason records that something changed, which the snapshot id already did.",
        )]
    return []


def _check_added(change: FileChange) -> List[Violation]:
    case, broken = _load(change.path, change.after)
    if broken is not None:
        return [broken]

    meta = _meta(case)
    if meta is None:
        return [Violation(change.path, Rule.MISSING_META,
                          "new case has no meta block, so it can carry no version.")]

    version = meta.get("version")
    if version is None:
        return [Violation(change.path, Rule.MISSING_VERSION,
                          "new case has no meta.version.")]
    if version != SEED_VERSION:
        return [Violation(
            change.path,
            Rule.SEED_VERSION,
            f"new case starts at {version!r}, but every case enters the set at "
            f"{SEED_VERSION!r} so that its first entry is findable.",
        )]
    return _require_logged_version(change.path, case, version)


def _check_modified(change: FileChange) -> Tuple[List[Violation], bool]:
    """Returns the violations and whether the case's content actually moved."""
    before, broken_before = _load(change.path, change.before)
    if broken_before is not None:
        # The base revision was already broken. Not this PR's fault, and
        # failing on it would block the PR that fixes it.
        return [], False

    after, broken_after = _load(change.path, change.after)
    if broken_after is not None:
        return [broken_after], True

    violations = _check_append_only(change.path, before, after)

    if not _content_changed(before, after):
        return violations, False

    meta_after = _meta(after)
    if meta_after is None:
        violations.append(Violation(change.path, Rule.MISSING_META,
                                    "case content changed but the case has no meta block."))
        return violations, True

    old_version = (_meta(before) or {}).get("version")
    new_version = meta_after.get("version")

    if new_version is None:
        violations.append(Violation(change.path, Rule.MISSING_VERSION,
                                    "case content changed but there is no meta.version to bump."))
        return violations, True

    if new_version == old_version:
        violations.append(Violation(
            change.path,
            Rule.MISSING_BUMP,
            f"case content changed but meta.version is still {new_version!r}. "
            "The snapshot id moved, so the version must too.",
        ))
        return violations, True

    if not _is_forward(old_version, new_version):
        violations.append(Violation(
            change.path,
            Rule.VERSION_REGRESSED,
            f"meta.version went backwards, {old_version!r} to {new_version!r}.",
        ))
        return violations, True

    violations.extend(_require_logged_version(change.path, after, new_version))
    return violations, True


def _check_append_only(
    path: str,
    before: Mapping[str, JSONValue],
    after: Mapping[str, JSONValue],
) -> List[Violation]:
    """Existing entries must survive verbatim, in order, as a prefix of the new list."""
    old_entries = _changelog(before)
    new_entries = _changelog(after)

    if len(new_entries) < len(old_entries):
        return [Violation(
            path,
            Rule.REWRITTEN_HISTORY,
            f"meta.changelog lost {len(old_entries) - len(new_entries)} entry(ies). "
            "The changelog is append-only so a past change cannot be hidden.",
        )]

    for index, old in enumerate(old_entries):
        if canonical_json(old) != canonical_json(new_entries[index]):
            return [Violation(
                path,
                Rule.REWRITTEN_HISTORY,
                f"meta.changelog entry {index} was edited. The changelog is append-only, "
                "so correct a past entry by appending a new one that says so.",
            )]
    return []


def _load(path: str, text: Optional[str]) -> Tuple[Dict[str, JSONValue], Optional[Violation]]:
    if text is None:
        return {}, Violation(path, Rule.MALFORMED_JSON, "file content was not available.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        return {}, Violation(path, Rule.MALFORMED_JSON, f"case does not parse: {err}")
    if not isinstance(parsed, dict):
        return {}, Violation(path, Rule.MALFORMED_JSON,
                             f"case is a {type(parsed).__name__}, expected an object.")
    return parsed, None


# ----------------------------------------------------------------------
# Asset rule
# ----------------------------------------------------------------------

def _asset_references(asset_path: str) -> Tuple[str, ...]:
    """The strings a case would plausibly use to point at this asset.

    Both the dataset-relative path and the bare file name, because cases
    are free to reference either and over-matching here is the safe
    direction: a spurious break costs a glance, a missed one costs a
    silent prompt change.
    """
    relative = asset_path[len(DATASET_ROOT) + 1:]
    return (relative, relative.rsplit("/", 1)[-1])


def _check_assets(
    changes: Sequence[FileChange],
    dataset_after: Mapping[str, str],
    bumped: Set[str],
    deleted: Set[str],
) -> List[Violation]:
    """An image swap is a prompt change, so every case that uses it must bump.

    Hashing the assets is what makes the change visible. This is what
    makes it explained, and it is the rule most likely to catch a real
    silent regression, because swapping a JPEG leaves every case file
    untouched.
    """
    violations: List[Violation] = []
    for change in changes:
        if not change.is_asset:
            continue
        needles = _asset_references(change.path)
        for case_name, text in sorted(dataset_after.items()):
            if case_name in deleted or case_name in bumped:
                continue
            if not any(needle in text for needle in needles):
                continue
            violations.append(Violation(
                f"{DATASET_ROOT}/{case_name}",
                Rule.UNBUMPED_ASSET_CONSUMER,
                f"asset {change.path} changed and this case references it, but the case "
                "was not bumped. Swapping an asset changes the prompt without touching "
                "the case file.",
            ))
    return violations


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def check_changes(
    changes: Sequence[FileChange],
    dataset_after: Optional[Mapping[str, str]] = None,
) -> GateReport:
    """Apply the changelog rules to a diff.

    Args:
        changes: Every changed path, with both sides of the diff
            resolved. Paths outside the golden dataset are ignored, so
            the caller may pass the whole diff.
        dataset_after: Case file name to raw text, at the head revision.
            Only needed to resolve which cases reference a changed
            asset. Omitting it skips the asset rule rather than silently
            passing it, which the report makes visible by not counting
            those cases as inspected.

    Returns:
        A :class:`GateReport`. ``ok`` is False if any rule was broken.
    """
    dataset_after = dataset_after or {}

    violations: List[Violation] = []
    advisories: List[Advisory] = []
    bumped: Set[str] = set()
    deleted: Set[str] = set()
    inspected = 0
    unchanged = 0

    for change in changes:
        if not change.is_case:
            continue

        if change.status is ChangeStatus.DELETED:
            deleted.add(change.name)
            advisories.append(Advisory(
                change.path,
                "case was deleted. There is no file left to carry a changelog entry, so "
                "confirm by hand that removing it was intended.",
            ))
            continue

        if change.status is ChangeStatus.ADDED:
            inspected += 1
            found = _check_added(change)
            violations.extend(found)
            if not found:
                bumped.add(change.name)
            continue

        # RENAMED is treated as MODIFIED. A rename with an edit is an
        # edit, and a rename alone leaves canonical content identical so
        # it costs nothing.
        inspected += 1
        found, content_moved = _check_modified(change)
        violations.extend(found)
        if not content_moved:
            unchanged += 1
        elif not found:
            bumped.add(change.name)

    violations.extend(_check_assets(changes, dataset_after, bumped, deleted))

    return GateReport(
        violations=tuple(violations),
        advisories=tuple(advisories),
        inspected=inspected,
        unchanged=unchanged,
    )
