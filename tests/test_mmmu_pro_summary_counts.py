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

"""mmmu_pro's summary row reconciles with its headline: total_questions counts the distinct
questions asked, effective_n those with a scored reply, correct_answers is null (the headline is a
mean of two best-of-mode rates) and base's pooled counts move to pooled_*. The run is stubbed; no
model, no network."""

from gbench.runners.eval_suites import mmmu_pro

_CATS = ("standard_10/cot", "standard_10/direct", "vision/cot", "vision/direct")


def test_the_row_counts_questions_not_replies(monkeypatch):
    def fake_run(**kw):
        traces = []
        for i, (_m, gold, meta) in enumerate(kw["samples"]):
            t = {"category": meta["category"], "is_correct": i % 2 == 0, "status": "OK",
                 "gold_answer": gold}
            if gold["id"] == "q3":                      # every reply of q3 a harness error
                t.update(status="FAILED", scoring_excluded=True, is_correct=False)
            traces.append(t)
        return {"sample_traces": traces, "accuracy": 50.0, "total_questions": len(traces),
                "correct_answers": sum(t["is_correct"] for t in traces)}

    samples = [([{"role": "user", "content": [{"type": "text", "text": "t"}]}],
                {"answer": "A", "options": ["a", "b"], "id": qid}, {"category": cat})
               for qid in ("q1", "q2", "q3") for cat in _CATS]
    monkeypatch.setattr(mmmu_pro, "run_eval_suite", fake_run)
    monkeypatch.setattr(mmmu_pro, "_load_mmmu_pro_samples", lambda *a, **k: samples)
    res = mmmu_pro.run_mmmu_pro("m", "http://x/v1", 1)
    assert res["pooled_total_questions"] == 12
    assert res["total_questions"] == 3 and res["effective_n"] == 2
    assert res["correct_answers"] is None and "pooled_correct_answers" in res
    assert "not a count" in res["correct_answers_note"]
