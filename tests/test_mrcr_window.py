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

"""MRCR: rows over the served window are skipped (as upstream does), results per bin.

Regression: all 2,400 rows (8 length bins up to 1M tokens)
were sent and every over-window row scored 0, capping the headline at ~62.5% on a 128K model
and ~75% on a 256K one before the model answered anything.
"""

import json

import pytest

from gbench.runners.eval_suites import base, mrcr


def test_bin_labels_follow_the_upstream_edges():
    assert mrcr._bin_label(4096) == "4k-8k"          # [4096, 8192]
    assert mrcr._bin_label(8192) == "4k-8k"
    assert mrcr._bin_label(8193) == "8k-16k"
    assert mrcr._bin_label(131072) == "64k-128k"
    assert mrcr._bin_label(1048576) == "512k-1024k"
    assert mrcr._bin_label(2_000_000) == ">1024k"


def test_bins_count_prompt_plus_answer_tokens():
    class _Enc:
        def encode(self, text, disallowed_special=()):
            return str(text).split()
    msgs = [{"role": "user", "content": "a b c"}, {"role": "assistant", "content": "d e"}]
    assert mrcr._bin_tokens(_Enc(), msgs, "x y", n_chars=0) == 7


def _prompt(n_words, n_msgs=1):
    per = max(1, n_words // n_msgs)
    return json.dumps([{"role": "user", "content": "w " * per} for _ in range(n_msgs)])


def test_cheap_bounds_include_the_chat_template_allowance():
    """Review finding: MRCR rows near the edge have 630-1350 messages at ~4 template tokens
    each, which the content-only count ignored, so over-window rows were still sent."""
    window = 131072
    budget = window - mrcr._ANSWER_ROOM
    n_msgs = 1000
    template = base._TEMPLATE_TOKENS_PER_REQUEST + n_msgs * base._TEMPLATE_TOKENS_PER_MESSAGE
    fits_only_without_template = int((budget - template // 2) * 3)
    called = []
    ok, how = mrcr._fits_window(lambda: called.append(1) or _prompt(10), fits_only_without_template,
                                n_msgs, window)
    assert called, "a row that fits only if the template is ignored must be counted, not assumed"


def test_exact_count_decides_borderline_rows(monkeypatch):
    window = 131072
    budget = window - mrcr._ANSWER_ROOM
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        lambda msgs, tools=None: (budget - 5, budget - 5, "server"))
    assert mrcr._fits_window(lambda: _prompt(10), budget * 4, 10, window) == (True, "server")
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        lambda msgs, tools=None: (budget + 5, budget + 5, "server"))
    assert mrcr._fits_window(lambda: _prompt(10), budget * 4, 10, window) == (False, "server")


def test_estimate_only_decisions_are_conservative(monkeypatch):
    """Without an exact count, a borderline row is skipped unless even the HIGH estimate fits:
    a row that reaches the server over the window scores 0."""
    window = 131072
    budget = window - mrcr._ANSWER_ROOM
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        lambda msgs, tools=None: (budget - 1000, budget + 1000, "estimate"))
    assert mrcr._fits_window(lambda: _prompt(10), budget * 4, 10, window) == (False, "estimate")


def test_rows_clearly_over_or_under_are_decided_without_tokenizing():
    window = 131072
    budget = window - mrcr._ANSWER_ROOM
    boom = lambda: (_ for _ in ()).throw(AssertionError("must not tokenize"))
    assert mrcr._fits_window(boom, budget * 5 + 10, 10, window) == (False, "bounds")
    assert mrcr._fits_window(boom, 1000, 10, window) == (True, "bounds")


def test_unknown_window_is_a_hard_error_not_a_blind_run(monkeypatch):
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", None)
    monkeypatch.setattr(mrcr, "_get_server_max_model_len", lambda url, model_name=None: None)
    monkeypatch.delenv("GBENCH_MRCR_WINDOW", raising=False)
    monkeypatch.setattr(mrcr, "_load_mrcr_samples",
                        lambda **k: pytest.fail("must not load or send rows without a window"))
    with pytest.raises(RuntimeError) as exc:
        mrcr.run_mrcr("m", "http://srv/v1", 1)
    assert "GBENCH_MRCR_WINDOW" in str(exc.value)


def test_breakdown_is_per_needles_and_bin_and_counts_failures_as_zero():
    traces = [
        {"category": "2needle/4k-8k", "mrcr_grade": 1.0, "response_text": "a"},
        {"category": "2needle/4k-8k", "mrcr_grade": 0.5, "response_text": "b"},
        {"category": "2needle/64k-128k", "mrcr_grade": None, "response_text": None},  # failed
        {"category": "8needle/8k-16k", "mrcr_grade": 0.25, "response_text": "c"},
    ]
    out = mrcr._mrcr_breakdown(traces)
    assert out["mrcr_bins"] == {"2needle": {"4k-8k": 75.0, "64k-128k": 0.0},
                                "8needle": {"8k-16k": 25.0}}
    assert out["mrcr_bin_counts"]["2needle"] == {"4k-8k": 2, "64k-128k": 1}
    # Bins are ordered by length, not alphabetically.
    assert list(out["mrcr_bins"]["2needle"]) == ["4k-8k", "64k-128k"]


def test_category_table_shows_the_mean_grade(monkeypatch):
    """The per-category table showed base.py's exact-match pass rate next to mrcr_bins."""
    samples = [([{"role": "user", "content": "q"}], {"answer": "ans", "canary": ""},
                {"category": "2needle/4k-8k", "canary": ""})]
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", 131072)
    monkeypatch.setattr(mrcr, "_load_mrcr_samples", lambda **k: samples)
    fake = {"sample_traces": [{"category": "2needle/4k-8k", "response_text": "an",
                               "gold_answer": {"answer": "ans", "canary": ""}}],
            "category_accuracy": {"2needle/4k-8k": {"correct": 0, "total": 1, "accuracy": 0.0}},
            "accuracy": 0.0}
    monkeypatch.setattr(mrcr, "run_eval_suite", lambda **k: fake)
    out = mrcr.run_mrcr("m", "http://srv/v1", 1)
    cell = out["category_accuracy"]["2needle/4k-8k"]
    assert cell["accuracy"] == out["mrcr_bins"]["2needle"]["4k-8k"] > 0
    assert cell["exact_match_accuracy"] == 0.0
    assert out["leaderboard_comparable"] is False


class _FakeDS:
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, k):
        return [r[k] for r in self.rows] if isinstance(k, str) else self.rows[k]


class _WordEnc:
    def encode(self, text, disallowed_special=()):
        return str(text).split()


def _row(words, needles):
    content = "w " * words
    return {"prompt": json.dumps([{"role": "user", "content": content}]), "answer": "a b",
            "n_chars": len(content), "total_messages": 1, "n_needles": needles,
            "random_string_to_prepend": "zz"}


def _load(monkeypatch, rows, window, limit=None, enc=_WordEnc()):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeDS(rows))
    monkeypatch.setattr(mrcr, "_o200k_encoder", lambda: enc)
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        lambda msgs, tools=None: (len(str(msgs[0]["content"]).split()),) * 2
                        + ("server",))
    monkeypatch.setattr(mrcr, "_CELL_ROWS", {"2needle/4k-8k": 3, "2needle/8k-16k": 3,
                                             "8needle/4k-8k": 3, "8needle/8k-16k": 3})
    stats = {}
    return mrcr._load_mrcr_samples(limit=limit, window=window, stats=stats), stats


def test_loader_drops_over_window_rows_before_the_limit_and_stratifies(monkeypatch):
    rows = ([_row(6000, 2)] * 3 + [_row(12000, 2)] * 3 + [_row(6000, 8)] * 3
            + [_row(12000, 8)] * 2 + [_row(40000, 8)] * 5)            # 5 rows over the window
    samples, stats = _load(monkeypatch, rows, window=20000, limit=4)
    assert stats["rows_in_window"] == 11 and stats["rows_skipped_over_window"] == 5
    cats = sorted(s[2]["category"] for s in samples)
    assert len(samples) == 4 and len(set(cats)) == 4, cats           # one from each cell
    assert stats["partial_bins"] == {"8needle/8k-16k": "2/3"}


def test_estimated_bins_make_no_partial_bin_claim(monkeypatch):
    samples, stats = _load(monkeypatch, [_row(6000, 2)] * 2, window=20000, enc=None)
    assert stats["bins_tokenizer"] == "chars/4 estimate" and stats["partial_bins"] is None


def test_answer_room_scales_with_the_gold_answer():
    assert mrcr._answer_room(0) == 512 >= mrcr._ANSWER_ROOM
    assert mrcr._answer_room(2000) == 2 * 2000 + 512


def test_window_override_wins_and_bad_values_are_explained(monkeypatch):
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", 262144)
    monkeypatch.setenv("GBENCH_MRCR_WINDOW", "not-a-number")
    with pytest.raises(RuntimeError) as exc:
        mrcr.run_mrcr("m", "http://srv/v1", 1)
    assert "GBENCH_MRCR_WINDOW" in str(exc.value)
    seen = {}
    monkeypatch.setenv("GBENCH_MRCR_WINDOW", "65536")
    monkeypatch.setattr(mrcr, "_load_mrcr_samples",
                        lambda **k: seen.update(window=k["window"]) or [])
    with pytest.raises(RuntimeError):
        mrcr.run_mrcr("m", "http://srv/v1", 1)
    assert seen["window"] == 65536
