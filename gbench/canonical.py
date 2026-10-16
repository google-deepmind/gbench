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

"""Canonical JSON, shared by the scaffold contract and the changelog gate.

This is a leaf module on purpose. It has no imports beyond the standard
library, and it deliberately sits outside ``gbench.core`` because that
package's ``__init__`` builds the whole model registry, which contacts
Hugging Face. The changelog gate runs in CI on a diff of JSON text files
and has nothing to do with models, so it must not pay that cost or take
that dependency. A policy gate that fails when Hugging Face is slow is a
policy gate that gets marked non-required.

The more important reason both callers read this one module: the gate
and the contract must agree on what counts as a change. The scaffold
snapshot hashes canonical form, so if the gate compared raw bytes it
would fire on reindentation that the hash ignores, and if it compared
something looser it would miss a change the hash caught. One function,
two callers, no drift possible.
"""

from __future__ import annotations

import json
from typing import Dict, List, Union

# Anything that survives a json.dump. Used instead of `Any` so that a
# value which cannot be serialised is a type error here rather than a
# TypeError at run start.
JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]

# Kept out of the hash and out of the gate comparison. The changelog is
# metadata *about* a change, not part of the case, and including it
# would be circular: appending the entry that explains a bump would
# itself be a content change, demanding another bump.
CHANGELOG_KEY = "changelog"


def _canonicalise(obj: JSONValue) -> JSONValue:
    """Normalise a JSON value so equivalent inputs hash equally.

    Integers become floats because a case that writes ``"temperature": 0``
    and one that writes ``0.0`` describe the same request, and an id that
    disagreed with itself over that would break a series for no reason.
    Booleans are excluded explicitly: ``bool`` subclasses ``int`` in
    Python, so ``True`` would silently become ``1.0`` without this guard.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _canonicalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v) for v in obj]
    return obj


def canonical_json(obj: JSONValue) -> str:
    """Return the canonical string form used for every hash in gbench.

    Sorted keys and no incidental whitespace, so reformatting a file
    cannot move an id. The changelog gate reads this same function, which
    is what stops the gate and the contract from drifting apart: if the
    hash would not move, the gate must not fire.
    """
    return json.dumps(
        _canonicalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def strip_changelog(case: Dict[str, JSONValue]) -> Dict[str, JSONValue]:
    """Return a copy of a case with ``meta.changelog`` removed.

    ``meta.version`` is deliberately kept. A version bump should move the
    id even in the rare case where it accompanies no content change, for
    example a revert.
    """
    if not isinstance(case, dict) or "meta" not in case:
        return case
    stripped = dict(case)
    meta = stripped.get("meta")
    if isinstance(meta, dict) and CHANGELOG_KEY in meta:
        meta = dict(meta)
        meta.pop(CHANGELOG_KEY)
        stripped["meta"] = meta
    return stripped
