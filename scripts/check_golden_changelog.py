#!/usr/bin/env python3
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

"""CI gate: a Golden Set case that changed must say why it changed.

This is the git plumbing only. Every rule lives in
``gbench/changelog_gate.py`` so it can be unit tested without a
repository, and so the gate reads the same canonicaliser as the scaffold
hash. Keeping those two on one function is what stops the gate firing on
changes the id would not have noticed.

Usage:
    python scripts/check_golden_changelog.py                  # vs origin/main
    python scripts/check_golden_changelog.py --base <sha>
    python scripts/check_golden_changelog.py --base <sha> --head <sha>

With no ``--head`` the working tree is compared, so the same command
works locally on uncommitted edits and in CI on a merge base.

Exit codes:
    0  no violations
    1  at least one violation
    2  the diff could not be read
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gbench.changelog_gate import (  # noqa: E402
    DATASET_ROOT,
    ChangeStatus,
    FileChange,
    check_changes,
)

# git's single-letter statuses. C (copied) is treated as an addition
# because the copy is a new case that needs its own 1.0 entry, and T
# (type change) as a modification.
_STATUS = {
    "A": ChangeStatus.ADDED,
    "C": ChangeStatus.ADDED,
    "D": ChangeStatus.DELETED,
    "M": ChangeStatus.MODIFIED,
    "R": ChangeStatus.RENAMED,
    "T": ChangeStatus.MODIFIED,
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _show(ref: str, path: str) -> Optional[str]:
    """File content at a revision, or None if it did not exist there."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _read_head(head: Optional[str], path: str) -> Optional[str]:
    """File content at the head revision, or from the working tree."""
    if head is not None:
        return _show(head, path)
    full = REPO_ROOT / path
    return full.read_text(encoding="utf-8") if full.exists() else None


def collect_changes(base: str, head: Optional[str]) -> List[FileChange]:
    """Read the diff and resolve both sides of every golden dataset path."""
    args = ["diff", "--name-status", "-M", "-z", base]
    if head is not None:
        args.append(head)
    fields = [f for f in _git(*args).split("\0") if f]

    changes: List[FileChange] = []
    index = 0
    while index < len(fields):
        raw_status = fields[index]
        letter = raw_status[0]
        # A rename or copy is emitted as three fields: status, old path,
        # new path. Everything else is two.
        if letter in ("R", "C"):
            old_path, new_path = fields[index + 1], fields[index + 2]
            index += 3
        else:
            old_path = new_path = fields[index + 1]
            index += 2

        if not new_path.startswith(f"{DATASET_ROOT}/"):
            continue

        status = _STATUS.get(letter)
        if status is None:
            continue

        changes.append(FileChange(
            path=new_path,
            status=status,
            before=None if status is ChangeStatus.ADDED else _show(base, old_path),
            after=None if status is ChangeStatus.DELETED else _read_head(head, new_path),
        ))

    return changes


def read_dataset(head: Optional[str]) -> Dict[str, str]:
    """Every case at the head revision, for resolving asset references."""
    cases: Dict[str, str] = {}
    if head is None:
        for path in sorted((REPO_ROOT / DATASET_ROOT).glob("*.json")):
            cases[path.name] = path.read_text(encoding="utf-8")
        return cases

    listing = _git("ls-tree", "--name-only", "-z", head, f"{DATASET_ROOT}/")
    for path in sorted(f for f in listing.split("\0") if f.endswith(".json")):
        content = _show(head, path)
        if content is not None:
            cases[path.rsplit("/", 1)[-1]] = content
    return cases


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main",
                        help="revision to diff against (default: origin/main)")
    parser.add_argument("--head", default=None,
                        help="revision to diff (default: the working tree)")
    args = parser.parse_args(argv)

    try:
        changes = collect_changes(args.base, args.head)
        dataset = read_dataset(args.head)
    except subprocess.CalledProcessError as err:
        print(f"Could not read the diff against {args.base!r}: {err.stderr.strip()}",
              file=sys.stderr)
        print("In CI this usually means the checkout was shallow. "
              "Set `fetch-depth: 0`.", file=sys.stderr)
        return 2

    report = check_changes(changes, dataset)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
