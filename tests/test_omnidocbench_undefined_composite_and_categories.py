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

"""omnidocbench: an undefined composite (a subset without tables or formulas) withholds the headline
with its reason and leaves the counts unset; base's per-subset pass flag is kept under its own name,
never as category_accuracy. Docker and generation are stubbed; files go to tmp only."""

import json
import types

from gbench.runners.eval_suites import omnidocbench as O


def _run(monkeypatch, tmp_path, overall):
    gt = tmp_path / "OmniDocBench.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "images/p1.jpg"}, "layout_dets": []}]))
    monkeypatch.setattr(O, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(O, "hf_hub_download", lambda **k: str(gt))
    monkeypatch.setattr(O, "_load_omnidocbench_samples", lambda limit=None: [("m", "g", {})])
    monkeypatch.setattr(O, "run_eval_suite", lambda **kw: {
        "eval_name": "omnidocbench", "accuracy": 100.0, "status": "success",
        "category_accuracy": {"academic": {"correct": 1, "total": 1, "accuracy": 100.0}},
        "sample_traces": [{"response_text": "# hello",
                           "gold_answer": json.dumps({"page_name": "p1.jpg"})}]})

    def fake_docker(cmd, **kw):
        out = [c.split(":")[0] for c in cmd
               if isinstance(c, str) and c.endswith(":/workspace/result")][0]
        with open(f"{out}/predictions_quick_match_run_summary.json", "w") as f:
            json.dump({"notebook_metric_summary": {"overall_notebook": overall, "metrics": {
                "text_block_Edit_dist": {"raw": 0.2, "notebook_value": 0.2}}}}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(O.subprocess, "run", fake_docker)
    return O.run_omnidocbench("m", "http://x", limit=1)


def test_an_undefined_composite_is_withheld_with_its_reason(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, None)
    assert r["accuracy"] is None and r["correct_answers"] is None and r["effective_n"] == 1
    assert r["accuracy_withheld_reason"].startswith("composite undefined")
    assert r["status_reason"].startswith("accuracy withheld: composite undefined")
    assert r["status"] == "success"                      # not a harness error


def test_the_pass_flag_is_not_category_accuracy(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, 42.0)
    assert r["accuracy"] == 42.0 and r["category_accuracy"] == {}
    assert r["category_edit_distance_pass_rate"]["academic"]["accuracy"] == 100.0
    assert "not the OmniDocBench composite" in r["category_edit_distance_pass_rate_note"]
