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

"""Math suites keep the evidence behind every number and say what their headline covers.

* cruxeval keeps the pass@5 generations, so each pass@5 can be re-derived from the result.
* amc_aime names an --eval-limit subset, budget cut-offs and requests with no reply, and its
  merge recomputes the two components over the union.
* hmmt / aime test a last-chance reply against the max_tokens its own request carried, and say
  when MathArena's offline vLLM path renders a default chat template.

No network, no model server.
"""

from unittest import mock

import pytest

from gbench.runners.eval_suites import amc_aime as AA
from gbench.runners.eval_suites import cruxeval
from gbench.runners.eval_suites import hmmt as H
from gbench.runners.eval_suites import ojbench as OJ


# --------------------------------------------------------------------------- #
# cruxeval                                                                     #
# --------------------------------------------------------------------------- #
def test_cruxeval_keeps_the_pass_at_5_generations():
    item = {"task": "output", "code": "def f(x): return x", "input": "1", "output": "1"}

    def trace(i, ok):
        return {"sample_idx": i, "source_sample_idx": 0, "category": "output_prediction",
                "is_correct": ok, "response_text": "x", "gold_answer": item}
    runs = []

    def fake_run(**kw):
        n = len(runs)
        runs.append(kw)
        traces = [trace(i, n == 1 and i == 3) for i in range(10)]
        return {"accuracy": 0.0, "temperature": kw["official_sampling"].temperature,
                "sample_traces": traces, "status": "success", "max_tokens_sent": [1024],
                "response_health": {"ok": 10}, "attempts": {"attempts_per_sample": 10}}
    with mock.patch.object(cruxeval, "sandbox_skip_reason", return_value=None), \
            mock.patch.object(cruxeval, "_grader_self_check", return_value=None), \
            mock.patch.object(cruxeval, "_load_cruxeval_samples", return_value=[]), \
            mock.patch.object(cruxeval, "run_eval_suite", side_effect=fake_run):
        res = cruxeval.run_cruxeval("m", "http://x/v1", 1)
    run5 = res["pass_at_5_run"]
    assert len(run5["sample_traces"]) == 10 and run5["max_tokens_sent"] == [1024]
    assert run5["response_health"] == {"ok": 10}
    rederived = cruxeval._cruxeval_report(run5["sample_traces"])
    for label, entry in rederived.items():
        if "pass_at_5" in entry:
            assert res["cruxeval_report"][label]["pass_at_5"] == entry["pass_at_5"]


# --------------------------------------------------------------------------- #
# amc_aime                                                                     #
# --------------------------------------------------------------------------- #
def _amc_trace(i, cat, ok, **extra):
    return {"sample_idx": i, "global_idx": i, "sample_key": f"k{i}", "category": cat,
            "is_correct": ok, "response_text": "The final answer is 1", "finish_reason": "stop",
            **extra}


def _amc_result(traces, **extra):
    scored = [t for t in traces if not t.get("scoring_excluded")]
    cats = {}
    for t in scored:
        c = cats.setdefault(t["category"], {"correct": 0, "total": 0})
        c["total"] += 1
        c["correct"] += bool(t["is_correct"])
    for c in cats.values():
        c["accuracy"] = round(c["correct"] / c["total"] * 100.0, 2)
    ok = sum(1 for t in scored if t["is_correct"])
    return {"eval_name": "amc_aime", "sample_traces": traces, "total_questions": len(traces),
            "correct_answers": ok, "accuracy": round(ok / len(scored) * 100.0, 2),
            "category_accuracy": cats, "status": "success", **extra}


def test_amc_aime_names_the_subset_budget_cutoffs_and_lost_requests(monkeypatch):
    traces = [_amc_trace(0, "AMC", True),
              _amc_trace(1, "AIME", False, finish_reason="length", max_tokens_sent=16384),
              _amc_trace(2, "AIME", False, response_text=None, scoring_excluded=True,
                         harness_error={"kind": "timeout"})]
    monkeypatch.setattr(AA, "_load_amc_aime_samples", lambda *a, **k: [])
    monkeypatch.setattr(AA, "run_eval_suite", lambda **k: _amc_result(traces))
    res = AA.run_amc_aime("m", "http://x/v1", 1, limit=3)
    reason = res["leaderboard_comparable_reason"]
    assert reason.startswith("non-canonical AMC+AIME blend")
    assert "subset run (--eval-limit 3: 3 of 173 problems)" in reason
    assert "1 repl(ies) cut off by this run's token budget" in reason
    assert "1 request(s) got no reply from the served endpoint" in reason
    assert res["replies_cut_off_by_budget"] == 1
    assert res["accuracy_amc"] == 100.0 and res["accuracy_aime"] == 0.0


def test_amc_aime_merge_recomputes_the_components_over_the_union():
    one = _amc_result([_amc_trace(0, "AMC", True), _amc_trace(2, "AIME", False)],
                      shard="1/2", shard_index=1, shard_count=2, eval_limit=None)
    two = _amc_result([_amc_trace(1, "AMC", False), _amc_trace(3, "AIME", True)],
                      shard="2/2", shard_index=2, shard_count=2, eval_limit=None)
    for r in (one, two):
        AA._components(r)
        AA._comparability(r, None)
    merged = AA.merge_shards([one, two])
    assert merged["accuracy"] == 50.0
    assert merged["accuracy_amc"] == 50.0 and merged["amc_n"] == 2
    assert merged["accuracy_aime"] == 50.0 and merged["aime_n"] == 2
    assert "accuracy_amc" not in (merged.get("merge_unreconciled_fields") or {})
    assert merged["leaderboard_comparable_reason"].startswith("non-canonical AMC+AIME blend")


# --------------------------------------------------------------------------- #
# hmmt / aime                                                                  #
# --------------------------------------------------------------------------- #
def test_a_last_chance_reply_is_tested_against_its_own_max_tokens():
    sampling = H.OfficialSampling(reference="cfg", max_tokens=2000)
    served = H.ServedModelSampling(sampling, {}, {"source": "cfg"},
                                   {"path": "configs/models/x.yaml"}, [])

    def result(lc):
        return {"max_tokens_sent": [2000], "official_max_tokens_sent": [2000], "sample_traces": [
            {"finish_reason": "stop", "response_text": "no box", "max_tokens_sent": 2000,
             "matharena_last_chance": lc}]}
    # first request at the config's cap; the last-chance prompt holds the first reply, so the
    # window lowered its max_tokens: a reply stopped there was cut off by the window
    clamped = result({"finish_reason": "length", "response_text": "", "max_tokens_sent": 1500})
    H.served_model_reasons("hmmt", clamped, served, {}, thinking=False)
    assert clamped["replies_cut_off_by_budget"] == 1
    at_cap = result({"finish_reason": "length", "response_text": "x", "max_tokens_sent": 2000})
    H.served_model_reasons("hmmt", at_cap, served, {}, thinking=False)
    assert at_cap["replies_cut_off_by_budget"] == 0
    # an older record without the field: the first request's max_tokens stands in
    legacy = result({"finish_reason": "length", "response_text": "x"})
    H.served_model_reasons("hmmt", legacy, served, {}, thinking=False)
    assert legacy["replies_cut_off_by_budget"] == 0


def test_a_configless_models_house_budget_is_named_as_gbenchs_choice():
    sampling = OJ.recommended_official_sampling({"temperature": 1.0}, "gc")
    served = H.ServedModelSampling(sampling, {}, {"source": "gc"}, None, [])
    res = {"max_tokens_sent": [16384], "sample_traces": [
        {"finish_reason": "length", "response_text": "x", "max_tokens_sent": 16384}]}
    reasons = H.served_model_reasons("hmmt", res, served, {}, thinking=False)
    text = " ".join(reasons)
    assert "MathArena ships no config for this model, so it defines no max_tokens" in text
    assert "house budget" in text and "sets no max_tokens for a model" not in text


@pytest.mark.parametrize("thinking", [True, False])
def test_an_offline_vllm_config_notes_the_default_chat_template(thinking):
    served = H.served_model_sampling("hmmt", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                                     H.DOCS_URL, thinking)
    note = served.model_config["chat_template"]
    assert "default chat template" in note and f"enable_thinking={thinking}" in note
    # not a comparability gap: a template that ignores the switch renders the same
    assert not any("chat template" in g for g in served.gaps)
