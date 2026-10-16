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

"""`--eval-limit N` is N samples for the whole suite: function-calling, tool, search and judged
suites (bfcl and its agentic track, acebench, t_eval, and the single-pass suites).

Before 2026-10-09, t_eval ran ceil(N / 16) rows of each of its 16 files (16 rows at
`--eval-limit 5`), and bfcl / acebench gave every part at least one entry (3 entries at
`--eval-limit 1`). bfcl's and acebench's own run tests are in test_bfcl_v4_overall_and_multi_turn.py
and test_acebench_agent_and_bfcl_search_log.py.
"""

import json
import re
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import fc_common, ifeval, lab_bench, t_eval
from gbench.runners.eval_suites.sampling import parse_shard, presharded_for_base, shard_select

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "gbench" / "runners" / "eval_suites"


# --------------------------------------------------------------------------- #
# the shared pieces
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("limit", [1, 2, 3, 5, 8, 22, 40])
def test_part_quotas_add_up_to_the_limit(limit):
    parts = {"single": {f"s{i}": None for i in range(13)}, "multi": {f"m{i}": None for i in range(4)},
             "agentic": {f"a{i}": None for i in range(5)}}
    q = fc_common.part_quotas(limit, parts, seed="bfcl")
    assert sum(q.values()) == limit
    # one per category before any gets two: a part's share follows its categories
    if limit <= 22:
        assert all(q[p] <= len(c) for p, c in parts.items())
    assert fc_common.part_quotas(limit, parts, seed="bfcl") == q          # deterministic


def test_part_quotas_without_a_limit_run_every_part_whole():
    assert fc_common.part_quotas(None, {"a": {"x": 3}, "b": {"y": 2}, "c": {}}, "s") == {
        "a": None, "b": None}
    assert fc_common.part_quotas(0, {"a": {"x": 3}}, "s") == {"a": None}


def test_part_quotas_cap_a_category_at_its_size():
    q = fc_common.part_quotas(10, {"a": {"x": 1}, "b": {"y": 50}}, "s")
    assert q == {"a": 1, "b": 9}
    assert sum(fc_common.part_quotas(10, {"a": {"x": 1}, "b": {"y": 2}}, "s").values()) == 3


def test_eval_limit_plan_record():
    plan = fc_common.eval_limit_plan("x", 5, 5, {"a": 3, "b": 0, "c": 2}, attempts=2)
    assert plan == {"limit": 5, "samples": 5, "per_setting": {"a": 3, "b": 0, "c": 2},
                    "attempts_per_sample": 2, "generations": 10}
    full = fc_common.eval_limit_plan("x", None, None, {"a": None}, note="n")
    assert full == {"limit": None, "samples": None, "per_setting": {"a": None},
                    "attempts_per_sample": 1, "generations": None, "note": "n"}


def test_base_suite_plan_counts_what_base_runs(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    assert fc_common.base_suite_plan("x", list(range(20)), 5, 1)["samples"] == 5
    assert fc_common.base_suite_plan("x", list(range(3)), 5, 1)["samples"] == 3
    assert fc_common.base_suite_plan("x", list(range(20)), None, 1)["samples"] == 20
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    # a list base shards itself, and a suite's own (presharded) shard
    assert fc_common.base_suite_plan("x", list(range(20)), None, 1)["samples"] == len(
        shard_select(list(range(20)), parse_shard("2/3")))
    assert fc_common.base_suite_plan("x", presharded_for_base([1, 2, 3, 4]), 5, 1)["samples"] == 4


# --------------------------------------------------------------------------- #
# t_eval: N rows over the 16 files, never one (or N) per file
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("limit", [1, 5, 16, 33])
def test_t_eval_file_limits_add_up_to_the_limit(limit):
    per_file = t_eval.file_limits(limit)
    assert len(per_file) == 16 and sum(per_file.values()) == limit
    assert max(per_file.values()) - min(per_file.values()) <= 1


def test_t_eval_no_limit_reads_every_row_of_every_file():
    assert set(t_eval.file_limits(None).values()) == {None}


def _t_eval_run(monkeypatch, limit):
    loaded = {}

    def fake_load(dim, stem, limit=None):
        loaded[stem] = limit
        n = 30 if limit is None else limit
        return [([{"role": "user", "content": f"{stem}-{i}"}], json.dumps({"dimension": dim}),
                 {"category": f"t_eval::{dim}"}) for i in range(n)]

    def fake_run(**kw):
        traces = [{"sample_idx": i, "gold_answer": g, "category": m["category"],
                   "response_text": "reply", "status": "OK", "is_correct": False,
                   "finish_reason": "stop", "max_tokens_sent": 512}
                  for i, (_msgs, g, m) in enumerate(kw["samples"])]
        assert kw["limit"] is None
        return {"sample_traces": traces, "total_questions": len(traces), "failed_requests": 0,
                "scoring_excluded": 0, "status": "success", "temperature": 0.0,
                "official_sampling": {}}
    monkeypatch.setattr(t_eval, "_load_file_samples", fake_load)
    monkeypatch.setattr(t_eval, "run_eval_suite", fake_run)
    monkeypatch.setattr(t_eval, "_score_file", lambda dim, traces, language="en": {"m": 0.5})
    return t_eval.run_t_eval("m", "http://x/v1", 1, limit=limit), loaded


def test_t_eval_eval_limit_5_runs_5_rows_in_total(monkeypatch):
    res, loaded = _t_eval_run(monkeypatch, 5)
    assert sum(loaded.values()) == 5 and len(loaded) == 5        # 5 files of 1 row; 11 not run
    assert len(res["sample_traces"]) == 5
    plan = res["eval_limit_plan"]
    assert plan["limit"] == 5 and plan["samples"] == 5 and plan["generations"] == 5
    assert sum(plan["per_setting"].values()) == 5 and len(plan["per_setting"]) == 16
    assert list(res["file_status"].values()).count("not_drawn") == 11
    assert "5 samples over the 16 files in total" in res["leaderboard_comparable_reason"]


def test_t_eval_without_a_limit_runs_every_file_whole(monkeypatch):
    res, loaded = _t_eval_run(monkeypatch, None)
    assert len(loaded) == 16 and set(loaded.values()) == {None}
    assert len(res["sample_traces"]) == 16 * 30
    assert res["eval_limit_plan"]["limit"] is None
    assert "not_drawn" not in res["file_status"].values()


# --------------------------------------------------------------------------- #
# single-pass suites: N samples, recorded in eval_limit_plan
# --------------------------------------------------------------------------- #

def _ifeval_rows(n):
    return [{"key": 1000 + i, "prompt": f"Write {i} words.",
             "instruction_id_list": ["length_constraints:number_words"],
             "kwargs": [{"relation": "at least", "num_words": i + 1}]} for i in range(n)]


def test_ifeval_eval_limit_5_selects_5_prompts(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    with mock.patch("datasets.load_dataset", return_value=_ifeval_rows(40)):
        assert len(ifeval._load_ifeval_samples(limit=5)) == 5
        assert len(ifeval._load_ifeval_samples()) == 40


@pytest.mark.parametrize("module,run,loader", [
    (lab_bench, "run_lab_bench", "_load_lab_bench_samples"),
    (ifeval, "run_ifeval", "_load_ifeval_samples"),
])
@pytest.mark.parametrize("limit", [5, None])
def test_single_pass_suites_record_their_plan(monkeypatch, module, run, loader, limit):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    n = 5 if limit else 12
    samples = [([{"role": "user", "content": f"p{i}"}], "A", {"category": "c"}) for i in range(n)]
    patches = [mock.patch.object(module, loader, return_value=samples),
               mock.patch.object(module, "run_eval_suite", side_effect=lambda **kw: {
                   "sample_traces": [], "temperature": None, "attempts": {}})]
    if module is ifeval:
        patches.append(mock.patch.object(ifeval, "_scorer", return_value=None))
    if module is lab_bench:
        patches.append(mock.patch.object(lab_bench, "_require_image_input", return_value=None))
    for p in patches:
        p.start()
    try:
        res = getattr(module, run)("m", "http://x/v1", concurrency=1, limit=limit)
    finally:
        for p in reversed(patches):
            p.stop()
    assert res["eval_limit_plan"] == {"limit": limit, "samples": n, "per_setting": None,
                                      "attempts_per_sample": 1, "generations": n}


OWNED_SINGLE_PASS = ["bfcl_live", "api_bank", "complexfuncbench", "nestful", "nexusraven_api_eval",
                     "seal_tools", "gorilla_apibench", "deepsearch_qa", "browsecomp",
                     "cyberseceval", "gdpval", "cimemories", "wildbench", "i18n_translate",
                     "ifeval", "frames", "lab_bench", "custom_jsonl"]


@pytest.mark.parametrize("suite", OWNED_SINGLE_PASS + ["bfcl", "bfcl_v4_agentic", "acebench",
                                                       "t_eval"])
def test_every_suite_logs_and_records_its_plan(suite):
    src = (SUITES / f"{suite}.py").read_text(encoding="utf-8")
    assert re.search(r"limit_plan = (base_suite_plan|eval_limit_plan)\(", src), suite
    assert '"eval_limit_plan"] = limit_plan' in src or '["eval_limit_plan"] = plan' in src, suite


@pytest.mark.parametrize("suite", OWNED_SINGLE_PASS + ["bfcl", "bfcl_v4_agentic", "acebench",
                                                       "t_eval"])
def test_every_doc_names_the_plan(suite):
    assert "`eval_limit_plan`" in (ROOT / "docs" / "evals" / f"{suite}.md").read_text(
        encoding="utf-8"), suite
