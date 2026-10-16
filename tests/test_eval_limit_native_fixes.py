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

"""The eval_limit_plan tells the truth (2026-10-09 swarm findings):

* under --shard, suites that draw --eval-limit in their loader and let base shard the list record
  THIS shard's samples (they recorded the pre-shard count on every shard, so a merged plan said
  N x shards);
* batch-scored suites (async_eval_fn, no supports_attempts) plan 1 attempt, as base runs;
* nexusraven's repair pass is not sharded a second time, and the plan records its repair calls;
* bfcl's agentic part counts 1 attempt; arc_agi records its retries;
* the headline setting gets the first sample (putnam_formal, charxiv, longbench_v2_code);
* the README / CLI / tau2 plan examples are the code's real output.
"""

import json
import logging

import pytest

from gbench.runners.eval_suites import base, limit_plan
from gbench.runners.eval_suites.sampling import allocate_limit, parse_shard, shard_select


class _Stop(Exception):
    """Raised by a fake run_eval_suite: the plan is logged and registered before it."""


def _stop(**_kw):
    raise _Stop


def _samples(n, key="category", values=("a", "b")):
    return [([{"role": "user", "content": f"q{i}"}], f"g{i}", {key: values[i % len(values)],
                                                               "category": values[i % len(values)],
                                                               "medxpertqa_id": f"id{i}"})
            for i in range(n)]


def _on_shard(n, shard):
    return len(shard_select(list(range(n)), parse_shard(shard)))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    saved = dict(base._RUN_KNOBS)
    yield
    base._RUN_KNOBS.clear()
    base._RUN_KNOBS.update(saved)
    limit_plan._PLANNED.clear()


def _plan_of(eval_name, call):
    limit_plan.clear_plan(eval_name)
    with pytest.raises(_Stop):
        call()
    plan = limit_plan.planned_for(eval_name)
    assert plan is not None, eval_name
    return plan


# --------------------------------------------------------------------------- #
# pre-shard counts: the plan is this shard's samples, and the shards add up
# --------------------------------------------------------------------------- #

def _putnam(monkeypatch, samples):
    from gbench.runners.eval_suites import putnam as M
    monkeypatch.setattr(M, "check_putnam_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_load_putnam_samples", lambda **k: samples)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "putnam", lambda: M.run_putnam("m", "http://x/v1", 1, limit=7)


def _medxpertqa(monkeypatch, samples):
    from gbench.runners.eval_suites import medxpertqa as M
    monkeypatch.setattr(M, "_load_medxpertqa_samples", lambda **k: samples)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "medxpertqa", lambda: M.run_medxpertqa("m", "http://x/v1", 1, limit=7)


def _mmmu_pro(monkeypatch, samples):
    from gbench.runners.eval_suites import mmmu_pro as M
    modes = M._prompt_modes()
    units = list(M.mmmu_pro_limit_shares(None, modes))
    rows = [(m, g, {**meta, "category": units[i % len(units)]})
            for i, (m, g, meta) in enumerate(samples)]
    monkeypatch.setattr(M, "_load_mmmu_pro_samples", lambda *a, **k: rows)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "mmmu_pro", lambda: M.run_mmmu_pro("m", "http://x/v1", 1, limit=7)


def _bundled_detection(monkeypatch, samples):
    from gbench.runners.eval_suites import bundled_detection as M
    monkeypatch.setattr(M, "_require_pycocotools", lambda: None)
    monkeypatch.setattr(M, "_load_bundled_detection_samples", lambda **k: samples)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return M.EVAL_NAME, lambda: M.run_bundled_detection("m", "http://x/v1", 1, limit=7)


def _screenspot_pointing(monkeypatch, samples):
    from gbench.runners.eval_suites import screenspot_pointing as M
    monkeypatch.setattr(M, "_load_screenspot_pointing_samples", lambda **k: samples)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "screenspot_pointing", lambda: M.run_screenspot_pointing("m", "http://x/v1", 1,
                                                                    limit=7)


def _putnam_formal(monkeypatch, samples):
    from gbench.runners.eval_suites import putnam_formal as M
    rows = [(m, g, {"category": M.SETTINGS[i % 2]}) for i, (m, g, _meta) in enumerate(samples)]
    monkeypatch.setattr(M, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_load_putnam_formal_samples", lambda **k: rows)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "putnam_formal", lambda: M.run_putnam_formal("m", "http://x/v1", 1, limit=7)


PRE_SHARD = {"putnam": _putnam, "medxpertqa": _medxpertqa, "mmmu_pro": _mmmu_pro,
             "bundled_detection": _bundled_detection,
             "screenspot_pointing": _screenspot_pointing, "putnam_formal": _putnam_formal}


@pytest.mark.parametrize("suite", sorted(PRE_SHARD))
def test_a_shards_plan_counts_the_samples_base_runs_on_it(monkeypatch, suite):
    samples = _samples(7, key="subset", values=("Text", "MM"))
    total = 0
    for i in (1, 2, 3):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/3")
        name, call = PRE_SHARD[suite](monkeypatch, samples)
        plan = _plan_of(name, call)
        assert plan["samples"] == _on_shard(7, f"{i}/3"), (suite, i, plan)
        if plan.get("per_setting"):
            assert sum(plan["per_setting"].values()) == plan["samples"]
        total += plan["samples"]
    assert total == 7                       # merged shards add up to the samples drawn


@pytest.mark.parametrize("suite", sorted(PRE_SHARD))
def test_without_a_shard_the_plan_is_every_drawn_sample(monkeypatch, suite):
    name, call = PRE_SHARD[suite](monkeypatch, _samples(7, key="subset", values=("Text", "MM")))
    assert _plan_of(name, call)["samples"] == 7


def test_ruler_shard_plan_counts_this_shards_rows_per_cell(monkeypatch):
    from gbench.runners.eval_suites import ruler as R
    monkeypatch.setattr(R, "_ruler_dir", lambda: "/ruler")
    monkeypatch.setattr(R, "_verify_revision", lambda d: R._RULER_REVISION)
    monkeypatch.setattr(R, "_require_context", lambda u, b: None)
    monkeypatch.setattr(R, "_served_chat_template", lambda tok, think: ("<t>", 7))
    monkeypatch.setattr(R, "_require_prompts_fit", lambda *a, **k: None)
    monkeypatch.delenv("GBENCH_RULER_LENGTHS", raising=False)
    monkeypatch.setenv("GBENCH_RULER_NUM_SAMPLES", "4")
    monkeypatch.setattr(R, "_prepare_task", lambda d, tok, task, band, n, chat_template=None,
                        revision=None: [{"input": f"{task} {band} {i}", "outputs": ["1"]}
                                        for i in range(n)])
    monkeypatch.setattr(R, "run_eval_suite", _stop)
    total = 0
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        plan = _plan_of("ruler", lambda: R.run_ruler("m", "http://x/v1", 1, tokenizer="t",
                                                     limit=5))
        assert plan["samples"] == _on_shard(5, f"{i}/2")
        assert sum(plan["per_setting"].values()) == plan["samples"]
        total += plan["samples"]
    assert total == 5


def test_t_eval_shard_plan_counts_this_shards_rows_of_each_file(monkeypatch):
    from gbench.runners.eval_suites import t_eval as T
    monkeypatch.setattr(T, "missing_packages", lambda: [])
    monkeypatch.setattr(T, "_vendor", lambda: None)

    def stop(*a, **k):
        raise _Stop
    monkeypatch.setattr(T, "_load_file_samples", stop)
    per_file = T.file_limits(32)                      # 2 rows of each of the 16 files
    total = 0
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        plan = _plan_of("t_eval", lambda: T.run_t_eval("m", "http://x/v1", 1, limit=32))
        assert plan["per_setting"] == {k: _on_shard(n, f"{i}/2") for k, n in per_file.items()}
        assert plan["samples"] == 16
        total += plan["samples"]
    assert total == 32


# --------------------------------------------------------------------------- #
# batch-scored suites run 1 attempt, and their plan says so
# --------------------------------------------------------------------------- #

def _browsecomp(monkeypatch, samples):
    from gbench.runners.eval_suites import browsecomp as M
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a, **k: None)
    monkeypatch.setattr(M, "_load_browsecomp_samples", lambda **k: samples)
    monkeypatch.setattr(M, "simple_evals_samples", lambda s, t: s)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "browsecomp", lambda: M.run_browsecomp("m", "http://x/v1", 1, limit=5)


def _frames(monkeypatch, samples):
    from gbench.runners.eval_suites import frames as M
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a, **k: None)
    monkeypatch.setattr(M, "_load_frames_samples", lambda **k: samples)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "frames", lambda: M.run_frames("m", "http://x/v1", 1, limit=5)


def _wildbench(monkeypatch, samples):
    from gbench.runners.eval_suites import wildbench as M
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a, **k: None)
    monkeypatch.setattr(M, "_load_wildbench_samples", lambda **k: samples)
    monkeypatch.setattr(M, "_load_reference_outputs", lambda: {})
    monkeypatch.setattr(M, "_load_wb_elo_pool_outputs", lambda: {})
    monkeypatch.setattr(M, "wb_elo_pool_items", lambda *a, **k: [])
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "wildbench", lambda: M.run_wildbench("m", "http://x/v1", 1, limit=5)


def _cyberseceval(monkeypatch, samples):
    from gbench.runners.eval_suites import cyberseceval as M
    monkeypatch.setattr(M, "_warn_removed_env", lambda: None)
    monkeypatch.setattr(M, "check_icd_image", lambda: {})
    monkeypatch.setattr(M, "_selftest_semgrep_or_raise", lambda: None)
    monkeypatch.setattr(M, "instruct_v2_path", lambda: ("/x", "test"))
    monkeypatch.setattr(M, "_load_cyberseceval_samples", lambda **k: samples)
    monkeypatch.setattr(M, "run_eval_suite", _stop)
    return "cyberseceval", lambda: M.run_cyberseceval("m", "http://x/v1", 1, limit=5)


@pytest.mark.parametrize("make", [_browsecomp, _frames, _wildbench, _cyberseceval],
                         ids=["browsecomp", "frames", "wildbench", "cyberseceval"])
def test_batch_scored_suites_plan_one_attempt_whatever_attempt_count(monkeypatch, make):
    base.set_run_knobs(attempt_count=3)
    name, call = make(monkeypatch, _samples(5))
    plan = _plan_of(name, call)
    # base.run_eval_suite runs these at 1 attempt (async_eval_fn, no supports_attempts)
    assert plan["attempts_per_sample"] == 1 and plan["generations"] == 5, plan


# --------------------------------------------------------------------------- #
# nexusraven: the repair pass is this node's, kept whole and attached to its queries
# --------------------------------------------------------------------------- #

def test_nexusraven_repairs_are_not_sharded_twice(monkeypatch):
    from gbench.runners.eval_suites import nexusraven_api_eval as N
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    tasks = ["toolalpaca", "toolllm", "cve_cpe", "emailrep", "virustotal"]
    samples = [([{"role": "user", "content": f"q{i}"}],
                json.dumps({"task": tasks[i % 5], "key": i}), {"category": tasks[i % 5]})
               for i in range(10)]
    monkeypatch.setattr(N, "_load_nexusraven_api_eval_samples", lambda **k: samples)
    # every first reply is rejected by the parser; every repaired one is right
    monkeypatch.setattr(N, "score", lambda task, key, text: (text == "fixed", text != "fixed"))
    monkeypatch.setattr(N, "fix_prompt", lambda text: "fix")
    calls = []

    def fake_base(**kw):
        # base: GBENCH_SHARD's shard of the list it is given, empty slots left out
        ran = [s for s in shard_select(kw["samples"], parse_shard("1/2")) if s is not None]
        first = not calls
        calls.append(ran)
        return {"sample_traces": [
            {"sample_idx": i, "gold_answer": g, "status": "OK", "category": m["category"],
             "response_text": "bad" if first else "fixed", "is_correct": not first}
            for i, (_msgs, g, m) in enumerate(ran)], "status": "success",
            "failed_requests": 0, "total_questions": len(ran)}
    monkeypatch.setattr(N, "run_eval_suite", fake_base)
    res = N.run_nexusraven_api_eval("m", "http://x/v1", 1)
    assert len(calls) == 2 and len(calls[1]) == len(calls[0]) == 5    # every repair ran
    traces = res["sample_traces"]
    assert all(t.get("output_fixing_response") == "fixed" for t in traces)
    assert res["output_fixing"]["repaired_correct"] == 5
    plan = res["eval_limit_plan"]
    assert plan["repair_calls"] == 5 and plan["samples"] == 5


def test_nexusraven_plan_bounds_its_repair_calls(monkeypatch):
    from gbench.runners.eval_suites import nexusraven_api_eval as N
    monkeypatch.setattr(N, "_load_nexusraven_api_eval_samples", lambda **k: _samples(4))
    monkeypatch.setattr(N, "run_eval_suite", _stop)
    plan = _plan_of(N.EVAL_NAME, lambda: N.run_nexusraven_api_eval("m", "http://x/v1", 1,
                                                                   limit=4))
    assert plan["repair_calls_at_most"] == plan["generations"] == 4


# --------------------------------------------------------------------------- #
# bfcl: --attempt-count k is k attempts of the single- and multi-turn entries, 1 of agentic
# --------------------------------------------------------------------------- #

def test_bfcl_plan_counts_one_attempt_for_agentic_entries(monkeypatch, caplog):
    from gbench.runners.eval_suites import bfcl as B

    def stop(*a, **k):
        raise _Stop
    monkeypatch.setattr(B, "_harness_probe", stop)
    with caplog.at_level(logging.INFO):
        plan = _plan_of("bfcl", lambda: B.run_bfcl("m", "http://x/v1", 1, limit=30,
                                                   attempt_count=3))
    ps = plan["per_setting"]
    assert ps["agentic"] > 0 and ps["single"] > 0
    assert plan["generations"] == 3 * (ps["single"] + ps["multi"]) + ps["agentic"]
    assert plan["attempts_per_part"] == {"single": 3, "multi": 3, "agentic": 1}
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[bfcl] --eval")]
    assert len(lines) == 1 and lines[0].endswith(f"= {plan['generations']} generations")
    assert "agentic 1 attempt" in lines[0]


def test_bfcl_plan_with_one_attempt_is_unchanged(monkeypatch):
    from gbench.runners.eval_suites import bfcl as B

    def stop(*a, **k):
        raise _Stop
    monkeypatch.setattr(B, "_harness_probe", stop)
    plan = _plan_of("bfcl", lambda: B.run_bfcl("m", "http://x/v1", 1, limit=30))
    assert plan["generations"] == plan["samples"] == 30


# --------------------------------------------------------------------------- #
# arc_agi: the retries are recorded
# --------------------------------------------------------------------------- #

def test_arc_agi_plan_records_its_retries(monkeypatch):
    from gbench.runners.eval_suites import arc_agi as A
    samples = [([{"role": "user", "content": "q"}], "[[1]]", {"task_id": f"t{i // 2}",
                                                              "category": "arc"})
               for i in range(4)]
    monkeypatch.setattr(A, "_load_arc_agi_samples", lambda **k: samples)
    monkeypatch.setattr(A, "run_eval_suite", lambda **kw: {
        "sample_traces": [], "status": "success", "attempts": {"attempts_per_sample": 2}})
    monkeypatch.setattr(A, "_retry_failed_tries", lambda traces, shown, kw: {
        "retried_attempts": 3, "recovered_by_retry": 1})
    res = A.run_arc_agi("m", "http://x/v1", 1, limit=2)
    plan = res["eval_limit_plan"]
    assert plan["generations"] == 8
    assert plan["retry_generations_at_most"] == 8 * (A._RETRY_ATTEMPTS - 1)
    assert plan["retry_generations"] == 3


# --------------------------------------------------------------------------- #
# the headline setting gets the first sample
# --------------------------------------------------------------------------- #

def test_putnam_formal_limit_1_runs_the_headline_setting(monkeypatch):
    from gbench.runners.eval_suites import putnam_formal as M
    names = [f"putnam_{i:04d}" for i in range(M.N_PROBLEMS)]
    stmt = "theorem x : True := sorry"
    monkeypatch.setattr(M, "load_putnambench", lambda: {
        "lean": {n: stmt for n in names},
        "informal": [{"problem_name": n, "tags": ["algebra"]} for n in names]})
    samples = M._load_putnam_formal_samples(limit=1)
    assert [meta["category"] for _m, _g, meta in samples] == ["lean_with_answer"]


def _spy(monkeypatch, module):
    seen = []

    def spy(*a, **k):
        seen.append(k.get("first"))
        return allocate_limit(*a, **k)
    monkeypatch.setattr(module, "allocate_limit", spy)
    return seen


def test_charxiv_deals_the_headline_track_first(monkeypatch, tmp_path):
    from unittest import mock
    from gbench.runners.eval_suites import charxiv as C
    import test_eval_limit_knowledge_lc_vision as K
    K._charxiv_zip(monkeypatch, tmp_path)
    seen = _spy(monkeypatch, C)
    with mock.patch("datasets.load_dataset", return_value=K._charxiv_rows()):
        samples = C._load_charxiv_samples(limit=1)
    assert seen == [("reasoning",)]
    assert [meta["category"] for _m, _g, meta in samples] == ["reasoning"]


def test_longbench_deals_the_headline_setting_first(monkeypatch):
    from gbench.runners.eval_suites import longbench_v2_code as L
    import test_eval_limit_knowledge_lc_vision as K
    seen = _spy(monkeypatch, L)
    res, _sent = K._longbench_run(monkeypatch, 1)
    assert seen == [(L.HEADLINE_SETTING,)]
    assert res["headline_setting"] == L.HEADLINE_SETTING


# --------------------------------------------------------------------------- #
# the documented plan examples are the code's output
# --------------------------------------------------------------------------- #

def test_readme_swe_bench_live_example_is_the_real_line(caplog):
    from pathlib import Path
    from gbench.runners.eval_suites import swe_bench_live as S
    tabs = ["lite", "full", "verified", "ccpp", "csharp", "java", "tsjs", "go", "rust"]
    with caplog.at_level(logging.INFO):
        S.limit_shares(5, tabs)
    line = next(r.getMessage() for r in caplog.records
                if r.getMessage().startswith("[swe_bench_live] --eval-limit 5"))
    readme = (Path(__file__).resolve().parents[1] / "docs" / "cli-reference.md").read_text()
    assert f"`{line}`" in readme


def test_tau2_doc_example_is_the_real_deal():
    import re
    from pathlib import Path
    from gbench.runners.eval_suites.tau_common import CORE_DOMAINS, domain_limit_shares
    shares = domain_limit_shares(5, list(CORE_DOMAINS),
                                 {"retail": 114, "airline": 50, "telecom": 114}, "tau2")
    doc = re.sub(r"\s+", " ", (Path(__file__).resolve().parents[1] / "docs/evals/tau2.md")
                 .read_text())
    assert ("over 3 domain(s) (" + ", ".join(f"{d} {n}" for d, n in shares.items()) + ") x 4"
            in doc)
    assert "/".join(str(shares[d]) for d in CORE_DOMAINS) + " over retail, airline and telecom" \
        in doc
