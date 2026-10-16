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

"""`--eval-limit N` is N samples for the whole suite: knowledge, long-context and vision suites.

Before 2026-10-09 a limited run of a suite with settings multiplied N: mmmu_pro ran N questions x
2 settings x 2 prompt modes (4N requests), charxiv N charts x 5 questions (5N), and
longbench_v2_code N questions in each of its two settings (2N, the CoT one two queries each);
ruler generated every task@band cell before drawing N. Each now runs N in total, dealt over its
settings with sampling.allocate_limit, and every suite logs its planned workload at its start
and records it as `eval_limit_plan`.
"""

import io
import json
import logging
import re
import zipfile
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from gbench.runners.eval_suites import base, limit_plan
from gbench.runners.eval_suites.sampling import allocate_limit

SUITE_DIR = Path(__file__).resolve().parent.parent / "gbench" / "runners" / "eval_suites"


class _FakeDS:
    """Enough of `datasets.Dataset` for the loaders: column reads, row reads, select, cast."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0].keys()) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        return self.rows[key]

    def select(self, indices):
        return _FakeDS([self.rows[i] for i in indices])

    def cast_column(self, *_a, **_k):
        return self


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _no_shard_no_knobs(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setattr(base, "_RUN_KNOBS", {})


# --------------------------------------------------------------------------- #
# the plan line and record
# --------------------------------------------------------------------------- #
def test_the_plan_is_logged_once_and_recorded(caplog):
    with caplog.at_level(logging.INFO, logger=limit_plan.__name__):
        plan = limit_plan.eval_limit_plan("x", 5, 5, {"a": 3, "b": 2, "c": 0},
                                          setting_noun="track", attempts=2)
    assert plan == {"limit": 5, "samples": 5, "per_setting": {"a": 3, "b": 2, "c": 0},
                    "attempts_per_sample": 2, "generations": 10}
    lines = [r.getMessage() for r in caplog.records if r.name == limit_plan.__name__]
    assert lines == ["[x] --eval-limit 5: 5 sample(s) over 2 track(s) (a 3, b 2) "
                     "(c: 0, not run) x 2 attempts = 10 generations"]
    assert limit_plan.planned_for("x") == plan


def test_a_full_run_plan_has_no_limit():
    plan = limit_plan.eval_limit_plan("y", None, 14042, unit="requests")
    assert plan == {"limit": None, "samples": 14042, "per_setting": None,
                    "attempts_per_sample": 1, "requests": 14042}


OWNED = ["gpqa", "mmlu", "mmlu_pro", "mmlu_redux", "multilingual_mmlu", "medxpertqa",
         "simpleqa", "healthbench", "humanitys_last_exam", "wmdp", "ruler", "loft_scifact", "mrcr",
         "aa_lcr", "beam_128k", "longbench_v2_code", "chartqa", "charxiv", "coco_caption",
         "docvqa", "infographicvqa", "mmmu_pro", "omnidocbench", "screenspot",
         "screenspot_pointing", "textvqa", "bundled_detection"]


@pytest.mark.parametrize("suite", OWNED)
def test_every_suite_logs_and_records_its_plan(suite):
    """Each runner (gpqa_diamond runs through gpqa's) plans its workload and records it."""
    src = (SUITE_DIR / f"{suite}.py").read_text(encoding="utf-8")
    assert "eval_limit_plan(" in src
    assert re.search(r"""["']eval_limit_plan["']\]? *[:=]""", src), suite


# --------------------------------------------------------------------------- #
# mmmu_pro: N requests over the setting x mode units, not 4N
# --------------------------------------------------------------------------- #
def _mmmu_fake(n=12):
    from gbench.runners.eval_suites import mmmu_pro
    red = Image.new("RGB", (2, 2), "red")
    std = [{"id": f"q{i}", "question": "Which? <image 1>", "options": "['a', 'b']",
            "answer": "A", "subject": f"s{i % 3}", "image_1": red} for i in range(n)]
    vis = [{"id": f"q{i}", "image": red, "options": "['a', 'b']", "answer": "A",
            "subject": f"s{i % 3}"} for i in range(n)]

    def load(path, name=None, **kw):
        return _FakeDS(std if name == mmmu_pro.STANDARD else vis)
    return load


@pytest.mark.parametrize("limit", [1, 3, 5, 9])
def test_mmmu_pro_limit_is_n_requests_over_the_units(limit):
    from gbench.runners.eval_suites import mmmu_pro
    with mock.patch("datasets.load_dataset", side_effect=_mmmu_fake()):
        samples = mmmu_pro._load_mmmu_pro_samples(limit=limit, modes=("cot", "direct"))
    assert len(samples) == limit                              # not 4 x limit
    shares = mmmu_pro.mmmu_pro_limit_shares(limit, ("cot", "direct"))
    got = {u: sum(1 for _m, _g, meta in samples if meta["category"] == u) for u in shares}
    assert got == shares and sum(shares.values()) == limit
    # a smaller share's questions are among a larger share's (one seeded draw)
    ids = {u: {meta["mmmu_pro_id"] for _m, _g, meta in samples if meta["category"] == u}
           for u in shares}
    small, large = sorted(ids.values(), key=len)[0], sorted(ids.values(), key=len)[-1]
    assert small <= large


def test_mmmu_pro_no_limit_runs_every_question_in_every_unit():
    from gbench.runners.eval_suites import mmmu_pro
    with mock.patch("datasets.load_dataset", side_effect=_mmmu_fake()):
        samples = mmmu_pro._load_mmmu_pro_samples(limit=None, modes=("cot", "direct"))
    assert len(samples) == 12 * 4


def test_mmmu_pro_records_the_plan(monkeypatch):
    from gbench.runners.eval_suites import mmmu_pro

    def fake_run(**kw):
        return {"sample_traces": [], "accuracy": None, "total_questions": 0}
    monkeypatch.setattr(mmmu_pro, "run_eval_suite", fake_run)
    with mock.patch("datasets.load_dataset", side_effect=_mmmu_fake()):
        res = mmmu_pro.run_mmmu_pro("m", "http://x/v1", 1, limit=5)
    plan = res["eval_limit_plan"]
    assert plan["limit"] == 5 and plan["samples"] == 5 and plan["generations"] == 5
    assert sum(plan["per_setting"].values()) == 5 and len(plan["per_setting"]) == 4


# --------------------------------------------------------------------------- #
# charxiv: N questions over the two tracks, not N charts x 5
# --------------------------------------------------------------------------- #
def _charxiv_rows(n=6):
    rows = []
    for i in range(n):
        rows.append({
            "image": None, "category": f"c{i % 3}", "figure_path": f"images/{i}.jpg",
            **{f"descriptive_q{k}": 1 for k in range(1, 5)},
            **{f"descriptive_a{k}": f"a{k}" for k in range(1, 5)},
            "reasoning_q": "What?", "reasoning_a": "x", "reasoning_a_type": 1,
            "subplot_loc": None, "subplot_row": 0, "subplot_col": 0})
    return rows


def _charxiv_zip(monkeypatch, tmp_path, n=6):
    from gbench.runners.eval_suites import charxiv as C
    path = tmp_path / "images.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            zf.writestr(f"{i}.jpg", b"\xff\xd8\xff\xe0chart" + bytes([i]))
    monkeypatch.setattr(C, "chart_images_zip", lambda: str(path))


@pytest.mark.parametrize("limit", [1, 2, 5, 7])
def test_charxiv_limit_is_n_questions_over_the_tracks(monkeypatch, tmp_path, limit):
    from gbench.runners.eval_suites import charxiv as C
    _charxiv_zip(monkeypatch, tmp_path)
    with mock.patch("datasets.load_dataset", return_value=_charxiv_rows()):
        samples = C._load_charxiv_samples(limit=limit)
    assert len(samples) == limit                              # not 5 x limit
    shares = allocate_limit(limit, C.TRACKS, available={"descriptive": 24, "reasoning": 6},
                            seed="charxiv")
    got = {t: sum(1 for _m, _g, meta in samples if meta["category"] == t) for t in C.TRACKS}
    assert got == shares


def test_charxiv_no_limit_runs_every_question(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import charxiv as C
    _charxiv_zip(monkeypatch, tmp_path)
    with mock.patch("datasets.load_dataset", return_value=_charxiv_rows()):
        samples = C._load_charxiv_samples(limit=None)
    assert len(samples) == 6 * 5
    # a full run keeps the order it always had: each chart's 4 descriptive, then its reasoning
    assert [meta["category"] for _m, _g, meta in samples[:5]] == ["descriptive"] * 4 + ["reasoning"]


def test_charxiv_records_the_plan(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import charxiv as C
    _charxiv_zip(monkeypatch, tmp_path)
    monkeypatch.setattr(C, "gemini_required_skip", lambda *a, **k: None)
    monkeypatch.setattr(C, "run_eval_suite", lambda **kw: {"sample_traces": []})
    with mock.patch("datasets.load_dataset", return_value=_charxiv_rows()):
        res = C.run_charxiv("m", "http://x/v1", limit=5)
    plan = res["eval_limit_plan"]
    assert plan["samples"] == 5 and sum(plan["per_setting"].values()) == 5
    assert "5 of the 5,000 questions" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# longbench_v2_code: N (setting, question) samples, not N per setting
# --------------------------------------------------------------------------- #
def _longbench_run(monkeypatch, limit):
    import test_longbench_v2_cot_setting as T
    return T._run(monkeypatch, {"non_cot": T._RIGHT, "cot": T._COT, "answer": T._RIGHT},
                  rows=T._rows(8), **({"limit": limit} if limit else {}))


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_longbench_limit_is_n_samples_over_the_settings(monkeypatch, limit):
    from gbench.runners.eval_suites import longbench_v2_code as L
    res, sent = _longbench_run(monkeypatch, limit)
    shares = allocate_limit(limit, list(L.SETTINGS), seed=L.EVAL_NAME)
    kinds = [kind for kind, _kw in sent]
    assert kinds.count("non_cot") == shares["non_cot"]
    assert kinds.count("cot") == shares["cot"]                # each CoT then its answer query
    assert kinds.count("answer") == shares["cot"]
    assert shares["non_cot"] + shares["cot"] == limit
    plan = res["eval_limit_plan"]
    assert plan["samples"] == limit and plan["per_setting"] == shares
    assert plan["rollouts"] == limit
    # a setting dealt nothing is not run, and the headline is the setting that ran
    assert set(res["settings"]) == {s for s, n in shares.items() if n}
    assert res["headline_setting"] in res["settings"]


def test_longbench_no_limit_runs_every_question_in_both_settings(monkeypatch):
    res, sent = _longbench_run(monkeypatch, None)
    kinds = [kind for kind, _kw in sent]
    assert kinds.count("non_cot") == 8 and kinds.count("cot") == 8
    assert res["eval_limit_plan"]["limit"] is None and res["eval_limit_plan"]["samples"] == 16


# --------------------------------------------------------------------------- #
# ruler: N samples over the task@band cells; only those cells are generated
# --------------------------------------------------------------------------- #
def _ruler(monkeypatch, limit):
    from gbench.runners.eval_suites import ruler as R
    monkeypatch.setattr(R, "_ruler_dir", lambda: "/ruler")
    monkeypatch.setattr(R, "_verify_revision", lambda d: R._RULER_REVISION)
    monkeypatch.setattr(R, "_require_context", lambda u, b: None)
    monkeypatch.setattr(R, "_served_chat_template", lambda tok, think: ("<t>", 7))
    monkeypatch.delenv("GBENCH_RULER_LENGTHS", raising=False)
    monkeypatch.setenv("GBENCH_RULER_NUM_SAMPLES", "4")
    prepared = []

    def fake_prepare(ruler_dir, tokenizer, task, band, num_samples, chat_template=None,
                     revision=None):
        prepared.append((task, band, num_samples))
        return [{"input": f"{task} {band} {i}", "outputs": ["42"]} for i in range(num_samples)]
    monkeypatch.setattr(R, "_prepare_task", fake_prepare)
    captured = {}

    def fake_suite(**kw):
        captured.update(kw)
        return {"sample_traces": [{"gold_answer": g, "response_text": "42",
                                   "category": m["category"], "status": "OK"}
                                  for _msgs, g, m in kw["samples"]], "temperature": 0.0}
    monkeypatch.setattr(R, "run_eval_suite", fake_suite)
    kwargs = {"tokenizer": "org/tok", **({"limit": limit} if limit else {})}
    return R.run_ruler("m", "http://x/v1", 4, **kwargs), captured, prepared


def test_ruler_limit_is_n_samples_and_generates_only_their_cells(monkeypatch):
    from gbench.runners.eval_suites import ruler as R
    res, captured, prepared = _ruler(monkeypatch, 5)
    assert len(captured["samples"]) == 5
    cells = {m["category"] for _msgs, _g, m in captured["samples"]}
    assert len(cells) == 5 and len(prepared) == 5             # 5 cells generated, not 78
    assert all(n == 4 for _t, _b, n in prepared)              # each at the full count (the cache)
    plan = res["eval_limit_plan"]
    assert plan["samples"] == 5 and sum(plan["per_setting"].values()) == 5
    assert len(plan["per_setting"]) == len(R._TASKS) * len(R._lengths())


def test_ruler_no_limit_runs_the_full_matrix(monkeypatch):
    from gbench.runners.eval_suites import ruler as R
    res, captured, prepared = _ruler(monkeypatch, None)
    n_cells = len(R._TASKS) * len(R._lengths())
    assert len(prepared) == n_cells and len(captured["samples"]) == n_cells * 4
    assert res["eval_limit_plan"]["limit"] is None


# --------------------------------------------------------------------------- #
# single-unit loaders: --eval-limit 5 is 5 samples, no limit is everything
# --------------------------------------------------------------------------- #
def _hf(rows):
    return mock.patch("datasets.load_dataset", side_effect=lambda *a, **k: _FakeDS(rows))


def _simpleqa():
    from gbench.runners.eval_suites import simpleqa as M
    rows = [{"problem": f"q{i}", "answer": "a", "metadata": f"m{i % 4}"} for i in range(23)]
    return M._load_simpleqa_samples, _hf(rows), 23


def _hle():
    from gbench.runners.eval_suites import humanitys_last_exam as M
    rows = [{"question": f"q{i}", "answer": "a", "category": f"c{i % 4}", "image": ""}
            for i in range(23)]
    return M._load_humanitys_last_exam_samples, _hf(rows), 23


def _wmdp():
    from gbench.runners.eval_suites import wmdp as M
    rows = [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": 1}
            for i in range(7)]
    return M._load_wmdp_samples, _hf(rows), 7 * len(M._SUBSETS)


def _multilingual():
    from gbench.runners.eval_suites import multilingual_mmlu as M
    rows = [{"Question": f"q{i}", "A": "a", "B": "b", "C": "c", "D": "d", "Answer": "A"}
            for i in range(3)]
    return M._load_multilingual_mmlu_samples, _hf(rows), 3 * len(M.MMMLU_CONFIGS)


def _medxpertqa():
    from gbench.runners.eval_suites import medxpertqa as M
    rows = [{"id": f"q{i}", "question": "Q?", "options": {"A": "x", "B": "y"}, "label": "A",
             "question_type": f"t{i % 2}", "images": []} for i in range(9)]
    return M._load_medxpertqa_samples, _hf(rows), 9 * len(M.SUBSETS)


def _chartqa():
    from gbench.runners.eval_suites import chartqa as M
    rows = [{"query": f"q{i}", "label": ["1"], "type": "human" if i % 2 else "augmented",
             "image": {"bytes": _png(), "path": None}} for i in range(13)]
    return M._load_chartqa_samples, _hf(rows), 13


LOADERS = {"simpleqa": _simpleqa, "humanitys_last_exam": _hle, "wmdp": _wmdp,
           "multilingual_mmlu": _multilingual, "medxpertqa": _medxpertqa, "chartqa": _chartqa}


@pytest.mark.parametrize("suite", sorted(LOADERS))
def test_single_unit_loaders_select_exactly_n_or_everything(suite):
    loader, patch, total = LOADERS[suite]()
    with patch:
        assert len(loader(limit=5)) == 5
    with patch:
        assert len(loader(limit=None)) == total


# --------------------------------------------------------------------------- #
# the runners record what they loaded
# --------------------------------------------------------------------------- #
def _fake_suite(**kw):
    traces = [{"sample_idx": i, "category": (m or {}).get("category"), "gold_answer": g,
               "response_text": "x", "status": "OK", "is_correct": False}
              for i, (_msgs, g, m) in enumerate(s for s in kw["samples"] if s is not None)]
    return {"sample_traces": traces, "total_questions": len(traces), "temperature": 0.0,
            "status": "success"}


@pytest.mark.parametrize("suite,runner,loader", [
    ("chartqa", "run_chartqa", "_load_chartqa_samples"),
    ("docvqa", "run_docvqa", "_load_docvqa_samples"),
    ("textvqa", "run_textvqa", "_load_textvqa_samples"),
    ("infographicvqa", "run_infographicvqa", "_load_infographicvqa_samples"),
    ("loft_scifact", "run_loft_scifact", "_load_loft_scifact_samples"),
])
def test_runners_record_the_plan_of_what_they_loaded(monkeypatch, suite, runner, loader):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    seen = {}

    def fake_loader(*a, limit=None, **k):
        seen["limit"] = limit
        return [([{"role": "user", "content": "q"}], "a", {"category": "c"})] * 5
    monkeypatch.setattr(mod, loader, fake_loader)
    monkeypatch.setattr(mod, "run_eval_suite", _fake_suite)
    res = getattr(mod, runner)("m", "http://x/v1", 1, limit=5)
    assert seen["limit"] == 5
    assert res["eval_limit_plan"] == {"limit": 5, "samples": 5, "per_setting": None,
                                      "attempts_per_sample": 1, "generations": 5}
