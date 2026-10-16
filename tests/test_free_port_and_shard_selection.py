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

"""Concurrency helpers: free-port allocation and deterministic shard selection."""

import socket

import pytest

from gbench.runners.eval_suites.base import free_port
from gbench.runners.eval_suites.sampling import parse_shard, shard_select


# ── free_port ───────────────────────────────────────────────────────────────

def test_free_port_returns_preferred_when_free():
    # Pick a high port unlikely to be in use, then confirm free_port hands it back.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        candidate = s.getsockname()[1]
    # candidate is now released; free_port(candidate) should return it (it is free).
    assert free_port(candidate) == candidate


def test_free_port_falls_back_when_preferred_is_taken():
    # Hold the preferred port, so free_port must pick a different, open one.
    with socket.socket() as held:
        held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        got = free_port(taken)
        assert got != taken
        # The returned port must itself be bindable.
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", got))


def test_free_port_no_preference_is_bindable():
    got = free_port()
    assert isinstance(got, int) and got > 0
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", got))


# ── parse_shard ─────────────────────────────────────────────────────────────

def test_parse_shard_valid_and_empty():
    assert parse_shard("3/8") == (3, 8)
    assert parse_shard("1/1") == (1, 1)
    assert parse_shard(None) is None
    assert parse_shard("") is None
    assert parse_shard("  ") is None


@pytest.mark.parametrize("bad", ["8", "3/0", "0/8", "9/8", "a/8", "3/b", "-1/8", "3/8/2"])
def test_parse_shard_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)


# ── shard_select (the acceptance property) ──────────────────────────────────

def test_shard_select_none_returns_full():
    items = list(range(37))
    assert shard_select(items, None) == items


@pytest.mark.parametrize("total,n", [(100, 8), (37, 5), (8, 8), (5, 8), (1000, 13)])
def test_shards_are_non_overlapping_and_union_to_full(total, n):
    items = list(range(total))
    shards = [shard_select(items, (i, n)) for i in range(1, n + 1)]
    flat = [x for sh in shards for x in sh]
    # Union is exactly the full set, and no element appears in two shards.
    assert sorted(flat) == items
    assert len(flat) == len(set(flat)) == total
    # Round-robin keeps shard sizes within 1 of each other.
    sizes = [len(sh) for sh in shards]
    assert max(sizes) - min(sizes) <= 1


def test_shard_select_is_deterministic():
    items = list(range(50))
    assert shard_select(items, (2, 7)) == shard_select(list(range(50)), (2, 7))


# ── run_eval_suite honors GBENCH_SHARD end to end (native path) ──────────────

def _run_sharded(n_samples, shard_spec, monkeypatch):
    from unittest import mock
    from gbench.runners.eval_suites import base
    if shard_spec is None:
        monkeypatch.delenv("GBENCH_SHARD", raising=False)
    else:
        monkeypatch.setenv("GBENCH_SHARD", shard_spec)
    samples = [([{"role": "user", "content": f"q{i}"}], str(i), {}) for i in range(n_samples)]

    async def _fake_send(*a, **k):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="unit_shard_suite", model_name="m", base_url="http://x",
            concurrency=1, samples=samples, eval_fn=lambda resp, gold: True)
    return res


def test_run_eval_suite_applies_shard_and_partitions(monkeypatch):
    # Full run scores all 10; shard 1/2 and 2/2 each score 5 and together cover all 10.
    assert _run_sharded(10, None, monkeypatch)["total_questions"] == 10
    a = _run_sharded(10, "1/2", monkeypatch)["total_questions"]
    b = _run_sharded(10, "2/2", monkeypatch)["total_questions"]
    assert a == 5 and b == 5 and a + b == 10
