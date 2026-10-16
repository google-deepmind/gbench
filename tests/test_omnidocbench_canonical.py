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

"""Canonical OmniDocBench: gbench generates markdown; scoring is delegated to OmniDocBench's own
evaluator image (per-modality CDM/TEDS/edit-distance composite)."""
import json
import types
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.omnidocbench as O


def test_omnidocbench_hard_errors_without_docker(monkeypatch):
    monkeypatch.setattr(O.shutil, "which", lambda _x: None)
    with pytest.raises(RuntimeError, match="docs/evals/omnidocbench.md"):
        O._check_prereqs(O._SCORER_IMAGE_DEFAULT)


def test_omnidocbench_hard_errors_without_image(monkeypatch):
    monkeypatch.setattr(O.shutil, "which", lambda _x: "/usr/bin/docker")

    def fake_run(cmd, **_kw):
        rc = 0 if cmd[:2] == ["docker", "info"] else 1     # daemon ok, image inspect fails
        return types.SimpleNamespace(returncode=rc, stdout=b"", stderr=b"")

    monkeypatch.setattr(O.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not found"):
        O._check_prereqs(O._SCORER_IMAGE_DEFAULT)


def test_build_command_mounts_gt_preds_out_and_runs_the_evaluator():
    cmd = O._build_command("img", "/w/gt.json", "/w/predictions", "/w/result", 4)
    joined = " ".join(cmd)
    assert cmd[0] == "docker" and "--rm" in cmd
    assert "/w/gt.json:/workspace/gt/gt.json:ro" in joined
    assert "/w/predictions:/workspace/data_md/predictions:ro" in joined
    assert "/w/result:/workspace/result" in joined
    assert "pdf_validation.py --config configs/gbench_end2end.yaml" in joined
    # the canonical per-modality metric set is requested
    assert "CDM" in joined and "TEDS" in joined and "Edit_dist" in joined
    assert "match_method: quick_match" in joined


def test_parse_summary_reads_the_composite(tmp_path):
    out = tmp_path
    (out / "predictions_quick_match_run_summary.json").write_text(json.dumps({
        "notebook_metric_summary": {
            "overall_notebook": 42.5,
            "metrics": {
                "text_block_Edit_dist": {"raw": 0.30, "notebook_value": 0.30},
                "display_formula_CDM": {"raw": 0.55, "notebook_value": 55.0},
                "table_TEDS": {"raw": 0.62, "notebook_value": 62.0},
                "reading_order_Edit_dist": {"raw": 0.12, "notebook_value": 0.12},
            }}}))
    s = O._parse_summary(str(out))
    assert s["overall_notebook"] == 42.5
    assert O._nb_value(s["metrics"], "display_formula_CDM") == 55.0
    assert O._nb_value(s["metrics"], "table_TEDS") == 62.0
    assert O._raw_value(s["metrics"], "text_block_Edit_dist") == 0.30


def test_gold_carries_page_name_for_prediction_filenames():
    """Each sample's gold must carry the image basename so the runner names the .md the evaluator
    matches (GT image_path `x.jpg` -> prediction `x.md`)."""
    import inspect
    src = inspect.getsource(O._load_omnidocbench_samples)
    assert '"page_name"' in src


def test_run_omnidocbench_delegates_and_reports_composite(monkeypatch, tmp_path):
    """End-to-end shape with docker + generation mocked: composite becomes the headline accuracy,
    per-modality values are surfaced, and a --eval-limit run is not leaderboard-comparable."""
    monkeypatch.setattr(O, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(O, "hf_hub_download", lambda **k: str(_write_gt(tmp_path)))
    monkeypatch.setattr(O, "_load_omnidocbench_samples", lambda limit=None: [("m", "g", {})])

    def fake_run_eval_suite(**kw):
        return {"eval_name": "omnidocbench", "accuracy": 0.0, "sample_traces": [
            {"response_text": "# hello", "gold_answer": json.dumps({"page_name": "p1.jpg"})}]}
    monkeypatch.setattr(O, "run_eval_suite", fake_run_eval_suite)

    def fake_docker(cmd, **kw):
        # find the mounted result dir and drop a summary there (simulate the evaluator)
        out = [c.split(":")[0] for c in cmd if isinstance(c, str) and c.endswith(":/workspace/result")][0]
        with open(f"{out}/predictions_quick_match_run_summary.json", "w") as f:
            json.dump({"notebook_metric_summary": {"overall_notebook": 50.0, "metrics": {
                "text_block_Edit_dist": {"raw": 0.2, "notebook_value": 0.2},
                "display_formula_CDM": {"raw": 0.5, "notebook_value": 50.0},
                "table_TEDS": {"raw": 0.6, "notebook_value": 60.0}}}}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(O.subprocess, "run", fake_docker)

    r = O.run_omnidocbench("m", "http://x", limit=1)
    assert r["accuracy"] == 50.0 and r["omnidocbench_overall"] == 50.0
    assert r["formula_cdm"] == 50.0 and r["table_teds"] == 60.0
    assert r["pages_scored"] == 1
    assert r["leaderboard_comparable"] is False           # --eval-limit subset


def _write_gt(tmp_path):
    p = tmp_path / "OmniDocBench.json"
    p.write_text(json.dumps([{"page_info": {"image_path": "images/p1.jpg"}, "layout_dets": []}]))
    return p
