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

"""Contract: `--eval-limit` must return a representative subset, not a contiguous head.

Datasets are stored grouped by category, so `rows[:limit]` returns one category. The
2026-08-15 full sweep measured 56 of 93 scored suites collapsing to a single category at
`--eval-limit 20` (`mmlu`->abstract_algebra, `mmlu_pro`->business, `ruler`->the 4k band,
`multipl_e`/`aider_polyglot`->cpp, ...). These tests lock the fix in.
"""

import re
from pathlib import Path

import pytest

from gbench.runners.eval_suites.sampling import (
    category_counts,
    limit_dataset,
    stratified_indices,
    stratified_sample,
)

SUITE_DIR = Path(__file__).resolve().parent.parent / "gbench" / "runners" / "eval_suites"


# --------------------------------------------------------------------------- #
# the primitive
# --------------------------------------------------------------------------- #

def _sorted_by_category(n_cats=57, per_cat=250):
    """A dataset shaped like cais/mmlu: many categories, stored sorted by category."""
    return [{"subject": f"s{i // per_cat:02d}", "idx": i} for i in range(n_cats * per_cat)]


def test_contiguous_head_is_one_category_but_stratified_is_not():
    rows = _sorted_by_category()
    key = lambda r: r["subject"]
    assert len(category_counts(rows[:20], key)) == 1          # the bug, reproduced
    assert len(category_counts(stratified_sample(rows, 20, key, seed="mmlu"), key)) == 20


def test_selection_is_deterministic_and_seed_dependent():
    rows = _sorted_by_category()
    key = lambda r: r["subject"]
    a = [r["idx"] for r in stratified_sample(rows, 15, key, seed="x")]
    b = [r["idx"] for r in stratified_sample(rows, 15, key, seed="x")]
    c = [r["idx"] for r in stratified_sample(rows, 15, key, seed="y")]
    assert a == b, "same seed must select the same rows on every run and machine"
    assert a != c, "different suites must not all sample the same offsets"


def test_rare_categories_are_not_crowded_out():
    rows = [{"c": "rare1"}, {"c": "rare2"}] + [{"c": "big"}] * 100
    got = category_counts(stratified_sample(rows, 5, lambda r: r["c"], seed="u"), lambda r: r["c"])
    assert set(got) == {"rare1", "rare2", "big"}
    assert got["rare1"] == 1 and got["rare2"] == 1


def test_degenerate_and_boundary_inputs():
    rows = _sorted_by_category(3, 5)
    key = lambda r: r["subject"]
    assert len(stratified_sample(rows, None, key)) == len(rows)     # no limit
    assert len(stratified_sample(rows, 0, key)) == len(rows)        # 0 == "no limit"
    assert len(stratified_sample(rows, 10_000, key)) == len(rows)   # limit > n
    assert stratified_sample([], 5, key) == []                      # empty
    assert len(stratified_sample(rows, 4, None, seed="s")) == 4     # no key_fn
    # a single-category source is not an error, it just cannot be spread
    one = [{"subject": "only"}] * 20
    assert len(stratified_sample(one, 5, key, seed="s")) == 5


def test_indices_are_ascending_and_unique():
    keys = [f"c{i % 7}" for i in range(500)]
    idx = stratified_indices(keys, 30, seed="s")
    assert len(idx) == 30 and len(set(idx)) == 30
    assert idx == sorted(idx), "callers slice datasets with these; they must be ordered"


def test_limit_dataset_handles_missing_column_and_list_input():
    rows = _sorted_by_category(5, 10)
    assert len(limit_dataset(rows, 7, "subject", seed="s")) == 7
    assert len(limit_dataset(rows, 7, "does_not_exist", seed="s")) == 7   # degrades, no raise
    assert len(limit_dataset(rows, None, "subject")) == len(rows)


# --------------------------------------------------------------------------- #
# the guard: no loader may reintroduce a contiguous head
# --------------------------------------------------------------------------- #

#: Suites whose slice is legitimately positional and carries no category to spread over.
_ALLOWED = {
    "codeforces",     # explicit CF/ICPC/IOI quotas
    "bfcl",           # per-category loop over the requested categories
    "ruler",          # length-banded loop
    "mmmu_pro",       # per-subject loop
}

_CONTIGUOUS = re.compile(
    r'\.select\(\s*range\(\s*min\(\s*limit|'          # ds.select(range(min(limit, ...)))
    r'=\s*\w+\[\s*:\s*limit\s*\]',                    # rows = rows[:limit]
)


def test_no_loader_slices_a_contiguous_head():
    offenders = []
    for path in sorted(SUITE_DIR.glob("*.py")):
        if path.stem in _ALLOWED or path.stem in {"sampling", "base"}:
            continue
        src = path.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _CONTIGUOUS.search(line):
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "these loaders take a contiguous head under --eval-limit, which returns one "
        "category. Use sampling.limit_dataset / stratified_sample:\n  "
        + "\n  ".join(offenders))


def test_base_truncation_is_stratified():
    """The central `samples[:limit]` backstop must also stratify."""
    src = (SUITE_DIR / "base.py").read_text(encoding="utf-8")
    assert "stratified_sample(" in src
    assert "samples[:limit]" not in src


# --------------------------------------------------------------------------- #
# end-to-end: drive real loaders over a synthetic sorted dataset
# --------------------------------------------------------------------------- #
from unittest import mock  # noqa: E402


def _sorted_dataset(n_categories=10, per_category=50, **columns):
    """A `datasets.Dataset` grouped by category, the layout where a contiguous head is one category.

    Real HF splits are stored sorted by subject (`cais/mmlu`: 14 042 rows, 57 subjects),
    so a contiguous head returns one category. Anything less than a real Dataset would
    not exercise `.select()`, which is what the loaders actually call.
    """
    from datasets import Dataset
    rows = []
    for c in range(n_categories):
        for i in range(per_category):
            row = {"category": f"cat_{c:02d}"}
            for key, make in columns.items():
                row[key] = make(c, i)
            rows.append(row)
    return Dataset.from_list(rows)


def _categories_of(samples):
    out = set()
    for s in samples:
        meta = s[2] if len(s) > 2 else None
        if isinstance(meta, dict) and meta.get("category"):
            out.add(str(meta["category"]))
    return out


def test_mmlu_pro_loader_spans_categories_at_limit_20():
    """The measured failure: `--eval-limit 20` returned only `business`."""
    from gbench.runners.eval_suites.mmlu_pro import _load_mmlu_pro_samples
    ds = _sorted_dataset(
        question=lambda c, i: f"q{c}-{i}",
        options=lambda c, i: ["a", "b", "c", "d"],
        answer_index=lambda c, i: i % 4,
    )
    with mock.patch("datasets.load_dataset", return_value=ds):
        samples = _load_mmlu_pro_samples(limit=20)
    cats = _categories_of(samples)
    assert len(samples) == 20
    assert len(cats) >= 8, f"collapsed to {sorted(cats)}"


# NOTE: RULER no longer has an HF loader; it generates per-tokenizer data via prepare.py and
# applies --eval-limit via stratified_sample over the generated (task@band) samples.


def test_limit_larger_than_the_dataset_returns_everything():
    from gbench.runners.eval_suites.mmlu_pro import _load_mmlu_pro_samples
    ds = _sorted_dataset(n_categories=3, per_category=4,
                         question=lambda c, i: f"q{c}-{i}",
                         options=lambda c, i: ["a", "b"],
                         answer_index=lambda c, i: 0)
    with mock.patch("datasets.load_dataset", return_value=ds):
        samples = _load_mmlu_pro_samples(limit=1000)
    assert len(samples) == 12


def test_no_limit_returns_everything_untouched():
    from gbench.runners.eval_suites.mmlu_pro import _load_mmlu_pro_samples
    ds = _sorted_dataset(n_categories=3, per_category=4,
                         question=lambda c, i: f"q{c}-{i}",
                         options=lambda c, i: ["a", "b"],
                         answer_index=lambda c, i: 0)
    with mock.patch("datasets.load_dataset", return_value=ds):
        samples = _load_mmlu_pro_samples(limit=None)
    assert len(samples) == 12


def test_the_same_limit_selects_the_same_rows_across_runs():
    """Two runs of the same suite must be comparable."""
    from gbench.runners.eval_suites.mmlu_pro import _load_mmlu_pro_samples
    ds = _sorted_dataset(question=lambda c, i: f"q{c}-{i}",
                         options=lambda c, i: ["a", "b"],
                         answer_index=lambda c, i: 0)
    with mock.patch("datasets.load_dataset", return_value=ds):
        first = [s[0][-1]["content"] for s in _load_mmlu_pro_samples(limit=20)]
    with mock.patch("datasets.load_dataset", return_value=ds):
        second = [s[0][-1]["content"] for s in _load_mmlu_pro_samples(limit=20)]
    assert first == second


# --------------------------------------------------------------------------- #
# crash regressions from the live 2026-08-15 sweep
# --------------------------------------------------------------------------- #
def test_unhashable_category_key_does_not_crash():
    """healthbench's `example_tags` is a LIST; it raised TypeError and killed the suite."""
    rows = [{"id": i, "tags": ["a", "b"] if i % 2 else ["c"]} for i in range(50)]
    got = stratified_sample(rows, 10, key_fn=lambda r: r["tags"], seed="t")
    assert len(got) == 10
    assert len({tuple(r["tags"]) for r in got}) == 2, "both tag groups must be represented"


def test_nested_unhashable_key_does_not_crash():
    rows = [{"k": [["deep"], {"d": 1}]} for _ in range(30)]
    assert len(stratified_sample(rows, 5, key_fn=lambda r: r["k"], seed="t")) == 5


def test_dict_category_key_does_not_crash():
    rows = [{"k": {"a": i % 3}} for i in range(30)]
    got = stratified_sample(rows, 6, key_fn=lambda r: r["k"], seed="t")
    assert len(got) == 6


def test_no_loader_keys_a_tuple_sample_as_a_dict():
    """`stratified_sample(samples, ..., lambda r: r.get(...))` where `samples` holds built
    (messages, gold, meta) tuples raised "'tuple' object has no attribute 'get'" and killed
    beam_128k outright. Catch the shape mismatch statically."""
    offenders = []
    for path in sorted(SUITE_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(
                r"stratified_sample\(\s*(\w+)\s*,[^\n]*lambda (\w+): \(\2 or \{\}\)\.get\(", src):
            var = m.group(1)
            if re.search(rf"{var}\.append\(\s*\(", src):
                offenders.append(f"{path.name}:{src[:m.start()].count(chr(10)) + 1}: "
                                 f"{var} holds tuples, but the key function calls .get() on it")
    assert not offenders, "\n  ".join([""] + offenders)


# --------------------------------------------------------------------------- #
# the thinking budget must not be swallowed by the floor
# --------------------------------------------------------------------------- #
def _budget(thinking, **kw):
    """Return the max_tokens actually sent for a one-sample run."""
    from unittest import mock
    from gbench.runners.eval_suites import base as B
    sent = []

    async def _fake(**k):
        sent.append(k)
        return B.Reply(text="x", tool_calls=None, finish_reason="stop",
                       reasoning=None, error=None, completion_tokens=1, stop_reason=None)

    with mock.patch.object(B, "_send_single_request", _fake):
        B.run_eval_suite(eval_name="aime", model_name="m", base_url="http://x",
                         concurrency=1, samples=[([{"role": "user", "content": "q"}], "x", {})],
                         eval_fn=lambda r, g: True, thinking=thinking, **kw)
    return sent[0]["max_output_tokens"]


def test_thinking_gets_a_bigger_budget_than_no_thinking():
    """REGRESSION 2026-08-18. `_run_suite_async` had a `32768 if thinking` default that
    could never fire: run_eval_suite applies the output floor FIRST, so max_output_tokens
    was never None by the time it was reached. Every --thinking run was silently capped at
    16384, and `aime` lost 3/3 responses to the cap and reported 0%."""
    from gbench.runners.eval_suites import base as B
    assert _budget(thinking=False) == B.DEFAULT_MIN_OUTPUT_TOKENS
    assert _budget(thinking=True) == B.THINKING_MIN_OUTPUT_TOKENS
    assert B.THINKING_MIN_OUTPUT_TOKENS > B.DEFAULT_MIN_OUTPUT_TOKENS


def test_an_explicit_operator_budget_still_wins_over_the_thinking_floor():
    from gbench.runners.eval_suites import base as B
    B.set_run_knobs(max_output_tokens=4096)
    try:
        assert _budget(thinking=True) == 4096
    finally:
        B._RUN_KNOBS.pop("max_output_tokens", None)
