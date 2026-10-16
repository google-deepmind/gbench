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

"""gdpval: an --eval-limit subset is drawn only from tasks whose reference can be scored.

2026-10-11 smoke run (`--eval-limit 1`, smoke-10-11-031629/gdpval): the draw picked task
e222075d, whose only expert deliverable is `GreenEnergy_v1.mp4`. All 3 generations finished
(finish_reason stop, ~2.6k tokens), every comparison was NO_REFERENCE (no_text_rendering), and
the run failed with "nothing was scored: all 3 sample(s) left out of the score". At the pinned
revision 39 of the 220 tasks have no text-renderable reference (35 ship no deliverable, 2 .mp4,
1 PDF without a text layer, 1 .zip of .wav stems), which is known before any generation.

The fix: the --eval-limit draw comes from the tasks whose reference renders to text (a
file-name check, then a render of the drawn references, a task that renders no text replaced);
the full run is unchanged and keeps every task. Nothing here touches the network.
"""

import json
from unittest import mock

import pytest

from gbench.runners.eval_suites import gdpval as G
from gbench.runners.eval_suites.sampling import stratified_sample

_OCCUPATIONS = ("Accountants", "Editors", "Film and Video Editors", "Lawyers", "Pharmacists")
#: task deliverables by kind: text renders; the rest never render to text
_KINDS = {
    "xlsx": ["deliverable_files/a/Model.xlsx"],
    "none": [],
    "mp4": ["deliverable_files/b/Reel.mp4"],
    "png": ["deliverable_files/c/chart.png"],
    "wav": ["deliverable_files/d/mix.wav"],
    "scan": ["deliverable_files/e/Scan.pdf"],        # renderable format, no text layer
    "mixed": ["deliverable_files/f/chart.png", "deliverable_files/f/Memo.docx"],
}


def _rows(kinds):
    rows = []
    for i, kind in enumerate(kinds):
        rows.append({
            "task_id": f"t{i:02d}-{kind}", "prompt": f"Task {i}.",
            "occupation": _OCCUPATIONS[i % len(_OCCUPATIONS)], "sector": "S",
            "reference_files": [], "deliverable_files": list(_KINDS[kind]),
            "rubric_json": json.dumps([{"score": 1, "criterion": "c"}]),
        })
    return rows


def _render(local, name):
    """render_local_file stand-in (the real render_attachment runs, so a format with no converter
    never reaches it): a scanned PDF and audio have no text (render_local_file summarises audio
    by its duration), every other file does."""
    if name == "Scan.pdf":
        return None, "[attachment 'Scan.pdf' is a PDF with no text layer]"
    if name.rsplit(".", 1)[-1] in G._AUDIO_EXTS:
        return None, f"[attachment '{name}' is 30s of audio]"
    return f"content of {name}", None


def _patched():
    """The dataset stand-ins; `fetch` (the per-file download) counts the files fetched."""
    fetch = mock.patch.object(G, "_fetch", side_effect=lambda rel: rel)
    return fetch, mock.patch.object(G, "render_local_file", side_effect=_render)


def _load(rows, limit):
    fetch, local = _patched()
    with mock.patch("datasets.load_dataset", return_value=rows), fetch as fetched, local:
        samples, references = G._load_gdpval_samples(limit=limit)
    return samples, references, G._DRAW.get(), fetched


def _ids(samples):
    return [s[2]["task_id"] for s in samples]


_MIX = ["xlsx", "none", "mp4", "scan", "xlsx", "png", "wav", "xlsx", "mixed", "none",
        "xlsx", "scan", "mp4", "xlsx", "xlsx", "none", "mixed", "xlsx", "wav", "xlsx"]


@pytest.mark.parametrize("limit", range(1, len(_MIX)))
def test_every_limited_draw_holds_only_tasks_whose_reference_renders(limit):
    samples, references, draw, _ = _load(_rows(_MIX), limit)
    scorable = sum(k in ("xlsx", "mixed") for k in _MIX)
    assert len(samples) == min(limit, scorable)
    for task_id in _ids(samples):
        assert task_id.split("-")[1] in ("xlsx", "mixed"), task_id
        assert references[task_id]["text"]
    assert set(references) == set(_ids(samples))
    assert draw["tasks_in_pool"] == len(_MIX) and draw["tasks_drawn"] == len(samples)
    assert draw["tasks_with_renderable_reference_format"] == scorable + _MIX.count("scan")
    assert draw["tasks_without_reference_by_format"] == {
        "no_deliverable_shipped": _MIX.count("none"),
        "no_text_rendering": _MIX.count("mp4") + _MIX.count("png") + _MIX.count("wav")}


def test_the_smoke_case_a_draw_that_hit_an_mp4_task_now_takes_a_scorable_one():
    """The pre-fix draw (stratified_sample over every task) picks one task for --eval-limit 1;
    make that task's only deliverable an .mp4, as e222075d's was."""
    rows = _rows(["xlsx"] * 12)
    old_pick = stratified_sample(rows, 1, lambda r: r["occupation"], seed="gdpval")[0]
    old_pick["deliverable_files"] = list(_KINDS["mp4"])
    samples, references, draw, _ = _load(rows, 1)
    assert len(samples) == 1 and _ids(samples)[0] != old_pick["task_id"]
    assert references[_ids(samples)[0]]["text"]
    assert draw["tasks_without_reference_by_format"] == {"no_text_rendering": 1}


def test_a_drawn_reference_that_renders_no_text_is_replaced_and_recorded():
    rows = _rows(["xlsx"] * 12)
    first = stratified_sample(rows, 1, lambda r: r["occupation"], seed="gdpval")[0]
    first["deliverable_files"] = list(_KINDS["scan"])
    samples, _, draw, _ = _load(rows, 1)
    assert _ids(samples) != [first["task_id"]] and len(samples) == 1
    assert draw["drawn_without_text_rendering"] == [
        {"task_id": first["task_id"],
         "note": "[attachment 'Scan.pdf' is a PDF with no text layer]"}]
    assert draw["tasks_with_renderable_reference_format"] == 12


@pytest.mark.parametrize("limit", [None, 0, len(_MIX), len(_MIX) + 5])
def test_the_full_run_keeps_every_task_including_the_unscorable_ones(limit):
    samples, references, draw, _ = _load(_rows(_MIX), limit)
    assert len(samples) == len(_MIX) and draw is None
    assert sum(1 for r in references.values() if not r["text"]) == (
        len(_MIX) - sum(k in ("xlsx", "mixed") for k in _MIX))


def test_the_draw_is_deterministic():
    a = _ids(_load(_rows(_MIX), 4)[0])
    b = _ids(_load(_rows(_MIX), 4)[0])
    assert a == b


def test_the_limited_draw_renders_references_only_for_the_tasks_it_considers():
    """A limited run must not download every reference of the set to pick a few."""
    _, _, _, fetched = _load(_rows(["xlsx"] * 30), 2)
    assert fetched.call_count == 2


@pytest.mark.parametrize("ext", sorted(set(G._RENDERABLE_EXTS) | {"mp4", "step", "psd", ""}))
def test_the_format_check_agrees_with_render_reference(ext):
    """reference_format_reason says "no_text_rendering" exactly when render_reference returns no
    text for the file whatever its bytes (and without downloading it)."""
    rel = f"deliverable_files/x/file.{ext}" if ext else "deliverable_files/x/Makefile"
    with mock.patch.object(G, "_fetch", side_effect=AssertionError("downloaded")), \
            mock.patch.object(G, "render_local_file", return_value=("text", None)):
        try:
            text, _ = G.render_reference([rel])
        except AssertionError:
            text = "would download"
    if ext in G._AUDIO_EXTS:
        # audio is downloaded and summarised (duration) but never rendered to text
        assert G.reference_format_reason([rel]) == "no_text_rendering"
    elif G.reference_format_reason([rel]) == "no_text_rendering":
        assert text is None
    else:
        assert text is not None


def test_run_gdpval_records_the_draw_in_the_plan_the_result_and_the_reason(monkeypatch):
    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    rows = _rows(_MIX)

    def fake_suite(**kw):
        return {"total_questions": 3, "temperature": 1.0, "temperature_source": "thinking"}

    monkeypatch.setattr(G, "run_eval_suite", fake_suite)
    fetch, local = _patched()
    with mock.patch("datasets.load_dataset", return_value=rows), fetch, local:
        r = G.run_gdpval("m", "http://x", 1, enable_thinking=True, limit=1)
    plan, draw = r["eval_limit_plan"], r["eval_limit_draw"]
    assert plan["samples"] == 1 and plan["tasks_in_pool"] == len(_MIX)
    assert plan["tasks_with_renderable_reference_format"] == draw[
        "tasks_with_renderable_reference_format"]
    assert "drawn only from tasks whose expert reference renders to text" in (
        r["leaderboard_comparable_reason"])


def test_a_full_run_records_no_draw(monkeypatch):
    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    G._DRAW.set({"stale": True})
    monkeypatch.setattr(G, "_load_gdpval_samples", lambda limit=None: ([], {}))
    monkeypatch.setattr(G, "run_eval_suite", lambda **kw: {"total_questions": 0})
    r = G.run_gdpval("m", "http://x", 1)
    assert "eval_limit_draw" not in r and "tasks_in_pool" not in r["eval_limit_plan"]


# 2026-10-11 review: the draw was decided per shard (limit < this shard's size). Shards differ in
# size (220 tasks over 8 shards: 28 x4, 27 x4), so `--eval-limit 27` drew from the scorable pool
# on the 28-task shards and ran the 27-task shards whole, unscorable tasks included, and the
# merged eval_limit_plan named a draw rule half the shards had not followed. The decision is now
# taken against the full set.
def _load_shard(rows, limit, shard, monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", shard)
    return _load(rows, limit)


@pytest.mark.parametrize("shard", ["1/3", "2/3", "3/3"])
def test_a_sharded_limit_at_a_shards_size_still_draws_from_scorable_tasks(shard, monkeypatch):
    """20 tasks over 3 shards are 7, 7, 6: `--eval-limit 6` used to run shard 3/3 whole."""
    samples, references, draw, _ = _load_shard(_rows(_MIX), 6, shard, monkeypatch)
    assert draw is not None
    assert samples and all(references[t]["text"] for t in _ids(samples))
    assert draw["scorable_short_of_limit"] == 6 - len(samples)


def test_a_sharded_limit_covering_the_full_set_is_a_full_run(monkeypatch):
    samples, _, draw, _ = _load_shard(_rows(_MIX), len(_MIX), "3/3", monkeypatch)
    assert draw is None and len(samples) == 6


def test_every_shard_of_a_limited_run_states_the_same_reason_and_the_plans_merge(monkeypatch):
    from gbench.runners.eval_suites.limit_plan import merge_plans
    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(G, "run_eval_suite", lambda **kw: {"total_questions": 1})
    results = []
    for shard in ("1/3", "2/3", "3/3"):
        monkeypatch.setenv("GBENCH_SHARD", shard)
        fetch, local = _patched()
        with mock.patch("datasets.load_dataset", return_value=_rows(_MIX)), fetch, local:
            results.append(G.run_gdpval("m", "http://x", 1, limit=6))
    assert len({r["leaderboard_comparable_reason"] for r in results}) == 1
    merged = merge_plans(results)
    assert merged["tasks_in_pool"] == len(_MIX)
    assert merged["tasks_with_renderable_reference_format"] == sum(
        k in ("xlsx", "mixed", "scan") for k in _MIX)
    assert merged["draw_rule"] and merged["scorable_short_of_limit"] == 18 - merged["samples"]


def test_a_limit_above_the_scorable_tasks_records_the_shortfall(monkeypatch):
    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(G, "run_eval_suite", lambda **kw: {"total_questions": 1})
    fetch, local = _patched()
    with mock.patch("datasets.load_dataset", return_value=_rows(_MIX)), fetch, local:
        r = G.run_gdpval("m", "http://x", 1, limit=len(_MIX) - 1)
    scorable = sum(k in ("xlsx", "mixed") for k in _MIX)
    assert r["eval_limit_plan"]["samples"] == scorable
    assert r["eval_limit_plan"]["scorable_short_of_limit"] == len(_MIX) - 1 - scorable
