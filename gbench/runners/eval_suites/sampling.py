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

"""Representative subsetting for `--eval-limit`.

Every loader used to reduce its dataset with a contiguous head - `rows[:limit]`,
`ds.select(range(limit))`, `islice`, or a break-on-count. Benchmark datasets are almost
always **stored sorted by category**, so the head is one category:

    cais/mmlu test = 14,042 rows / 57 subjects, sorted by subject
    -> rows[:20] is 20 `abstract_algebra` questions, published as "mmlu 85%"

Measured on the 2026-08-15 full sweep: **56 of 93 scored suites** put all 20 sampled rows
in a single category (`mmlu`->abstract_algebra, `mmlu_pro`->business, `ruler`->only the 4k
band, `multipl_e`/`aider_polyglot`->cpp, `copilot_bench_swe`->one repo, ...).

`stratified_indices` / `stratified_sample` replace that with a deterministic round-robin
across the category key, so a limited run is a *sample of the benchmark* rather than a
sample of whichever category sorts first.

Determinism: selection is seeded on the suite name, so the same `--eval-limit` picks the
same rows on every machine and every run. Two runs are comparable; a run and a
pre-stratification run are NOT.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["stratified_indices", "stratified_sample", "category_counts", "limit_dataset"]


def _rng(seed: Optional[str]) -> random.Random:
    """Deterministic RNG derived from a stable string (usually the suite name)."""
    digest = hashlib.sha256((seed or "gbench").encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _stable_key(key: Any) -> Any:
    """A hashable, deterministic stand-in for a category key.

    Category fields are not always scalars: healthbench's `example_tags` is a list, which
    is unhashable and raised `TypeError: unhashable type: 'list'` here - killing the whole
    suite rather than degrading. `repr` is used rather than `tuple(...)` because nested
    lists would still be unhashable, and it is stable for the JSON-derived values these
    rows hold.
    """
    if key is None:
        return "<none>"
    try:
        hash(key)
    except TypeError:
        return repr(key)
    return key


def _group(keys: Sequence[Any]) -> "OrderedDict[Any, List[int]]":
    groups: "OrderedDict[Any, List[int]]" = OrderedDict()
    for index, key in enumerate(keys):
        groups.setdefault(_stable_key(key), []).append(index)
    return groups


def stratified_indices(keys: Sequence[Any], limit: Optional[int],
                       seed: Optional[str] = None) -> List[int]:
    """Indices of a representative subset of `limit` items, given each item's category.

    Round-robins across categories (largest first on ties broken deterministically) so
    every category is represented before any category is sampled twice. Within a category
    the order is shuffled with the seeded RNG, so we do not simply take that category's
    head either - benchmark rows are often ordered by difficulty inside a subject.

    Returns indices in ascending order so callers can slice a dataset with them directly.
    """
    total = len(keys)
    if limit is None or limit <= 0 or limit >= total:
        return list(range(total))

    rng = _rng(seed)
    groups = _group(keys)
    for indices in groups.values():
        rng.shuffle(indices)

    # Visit categories in a stable, seeded order so no category is systematically favoured
    # when `limit` does not divide evenly.
    order = list(groups)
    rng.shuffle(order)

    chosen: List[int] = []
    while len(chosen) < limit:
        progressed = False
        for key in order:
            bucket = groups[key]
            if not bucket:
                continue
            chosen.append(bucket.pop())
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:                     # every bucket drained
            break
    return sorted(chosen)


def allocate_limit(limit: Optional[int], settings: Sequence[Any],
                   available: Optional[Dict[Any, int]] = None,
                   seed: Optional[str] = None,
                   first: Sequence[Any] = ()) -> Dict[Any, Optional[int]]:
    """How many of `--eval-limit`'s `limit` samples each of a suite's `settings` (tabs, splits,
    languages, domains, scenarios: parts the suite runs as separate passes) gets, so the suite
    as a whole runs `limit` samples, never `limit` per setting.

    The rule for every suite: `--eval-limit N` is N samples in total (distinct dataset items,
    counted once whatever the number of attempts a protocol makes per item). The samples are
    dealt round-robin over the settings in a seeded order (stratified_indices), so each setting
    gets one before any gets two; a setting can get 0 when N is smaller than the number of
    settings, and a suite then does not run it. `available` caps a setting at its own size (its
    unused share goes to the others). `first`: settings that get one sample each before the deal
    (in that order; e.g. the suite's headline setting, so a limit smaller than the number of
    settings still measures what the result reports as its headline). `limit` None or <= 0
    means no limit: every setting gets None (all of its samples)."""
    names = list(dict.fromkeys(settings))
    if limit is None or limit <= 0:
        return {name: None for name in names}
    room = {name: (int(limit) if (available or {}).get(name) is None
                   else min(int(available[name]), int(limit))) for name in names}
    keys: List[Any] = []
    for name in names:
        keys.extend([name] * room[name])
    counts: Dict[Any, Optional[int]] = {name: 0 for name in names}
    for i in stratified_indices(keys, int(limit), seed):
        counts[keys[i]] += 1
    # A `first` setting the deal left at 0 takes one sample from the setting holding the most
    # (the last such in `settings` order), so the deal stays as even as it was.
    for name in first:
        if name in counts and counts[name] == 0 and room.get(name, 0) > 0:
            donors = [n for n in names if n != name and n not in first and counts[n]]
            if donors:
                most = max(counts[n] for n in donors)
                donor = [n for n in donors if counts[n] == most][-1]
                counts[donor] -= 1
                counts[name] += 1
    return counts


def stratified_sample(rows: Sequence[Any], limit: Optional[int],
                      key_fn: Optional[Callable[[Any], Any]] = None,
                      seed: Optional[str] = None) -> List[Any]:
    """`rows` reduced to `limit`, spread across the categories `key_fn` reports.

    With no `key_fn` (or when every row reports the same category) this degrades to a
    seeded random subset rather than a contiguous head - still better than the head,
    because dataset order correlates with difficulty and source.
    """
    if limit is None or limit <= 0 or limit >= len(rows):
        return list(rows)
    keys = [key_fn(r) if key_fn else None for r in rows]
    return [rows[i] for i in stratified_indices(keys, limit, seed)]


def parse_shard(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a `--shard` spec `"I/N"` (1-indexed shard I of N) into `(I, N)`. Returns None for an
    empty/None spec. Raises ValueError on a malformed spec or out-of-range I (so a typo fails fast
    instead of silently running the whole set)."""
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    if "/" not in s:
        raise ValueError(f"--shard must be I/N (e.g. 1/8), got {spec!r}")
    i_str, n_str = s.split("/", 1)
    try:
        i, n = int(i_str), int(n_str)
    except ValueError:
        raise ValueError(f"--shard must be I/N with integers, got {spec!r}")
    if n < 1 or i < 1 or i > n:
        raise ValueError(f"--shard I/N needs 1 <= I <= N and N >= 1, got {spec!r}")
    return (i, n)


def shard_select(items: Sequence[Any], shard: Optional[Tuple[int, int]]) -> List[Any]:
    """Select shard `I` of `N` from an ALREADY-ORDERED sequence by round-robin (`items[I-1::N]`).

    Round-robin over a deterministic (for example stratified) order gives shards that are
    non-overlapping and whose union is exactly the full set, and keeps each shard category-balanced.
    Callers must pass a stably-ordered sequence (sort/stratify first) so the partition is reproducible
    across machines. `shard=None` returns the full list unchanged.
    """
    if shard is None:
        return list(items)
    i, n = shard
    return list(items)[i - 1::n]


def presharded_for_base(samples: Sequence[Any],
                        environ: Optional[Dict[str, str]] = None) -> List[Any]:
    """`samples`, already this node's shard (a suite that draws its own --eval-limit subset
    selects its shard of the full ordered set first), laid out so that base.run_eval_suite, which
    applies GBENCH_SHARD "I/N" to the list it is given (shard_select: items[I-1::N]), keeps exactly
    them, in order: each sample sits at a position shard I selects and the positions between them
    are empty (None). Without a shard, or with 1/1, the list is returned as it is."""
    env = os.environ if environ is None else environ
    shard = parse_shard(env.get("GBENCH_SHARD"))
    if not shard or shard[1] == 1:
        return list(samples)
    index, count = shard
    laid_out: List[Any] = [None] * (len(samples) * count)
    laid_out[index - 1::count] = list(samples)
    return laid_out


def category_counts(rows: Iterable[Any],
                    key_fn: Optional[Callable[[Any], Any]] = None) -> Dict[Any, int]:
    """Category histogram - for asserting coverage in tests and for run diagnostics."""
    counts: Dict[Any, int] = {}
    for row in rows:
        key = key_fn(row) if key_fn else None
        counts[key] = counts.get(key, 0) + 1
    return counts


def limit_dataset(ds: Any, limit: Optional[int], key_col: Optional[str] = None,
                  seed: Optional[str] = None) -> Any:
    """Reduce a HuggingFace `Dataset` (or a plain list of dicts) to `limit`, stratified.

    Drop-in replacement for `ds.select(range(min(limit, len(ds))))`. Reads only the
    category column, so it stays cheap on image and long-context datasets where
    materialising every row to build samples would not be.

    `key_col` missing from the dataset is not an error - the selection degrades to a
    seeded random subset, which is still preferable to a contiguous head.
    """
    if limit is None or limit <= 0:
        return ds
    try:
        total = len(ds)
    except TypeError:
        return ds
    if total <= limit:
        return ds

    keys: Sequence[Any] = [None] * total
    if key_col:
        try:
            columns = getattr(ds, "column_names", None)
            if columns is not None and key_col in columns:
                keys = list(ds[key_col])                       # single columnar read
            elif columns is None:
                keys = [(row or {}).get(key_col) for row in ds]
        except Exception:
            keys = [None] * total

    indices = stratified_indices(keys, limit, seed)
    if hasattr(ds, "select"):
        return ds.select(indices)
    return [ds[i] for i in indices]
