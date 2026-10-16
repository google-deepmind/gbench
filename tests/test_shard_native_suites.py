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

"""Sharding, merging and served-model details of the native suites.

* a result that covers one shard of a --shard I/N run is never leaderboard-comparable, in the
  suites that rewrite their comparability after base (gpqa family, mmlu family, longbench,
  multipl_e, lcb, omnidocbench and the others);
* --shard is applied to the full ordered set BEFORE the --eval-limit draw in every native loader
  that draws its own limit, and base keeps exactly the loader's shard
  (sampling.presharded_for_base);
* multipl_e: a completion the evaluator returned no result for is a harness error in a real run;
  problem ids are unique across languages; its shards merge to upstream's per-problem pass@1;
* omnidocbench: the shards' pages are merged by re-scoring their union with the evaluator;
* mrcr takes the context window of the model under test from GET /v1/models;
* ojbench forwards OJBENCH_GET_TIMEOUT_S into the judge container.
No test starts a server or a container, calls Gemini or reaches the network.
"""

import ast
import json
import os
import types
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "gbench" / "runners" / "eval_suites"


@pytest.fixture(autouse=True)
def _no_shard(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)


def _text_of(content):
    return content if isinstance(content, str) else " ".join(
        p.get("text", "") for p in content or [] if isinstance(p, dict))


def _answer(text):
    """A stub for base's request call that answers every prompt with `text`."""
    async def _send(**kw):
        return base.Reply(text=text, tool_calls=None, finish_reason="stop", completion_tokens=5)
    return _send


# --------------------------------------------------------------------------- #
# A partial shard is not leaderboard-comparable                               #
# --------------------------------------------------------------------------- #
def _gpqa_result(**extra):
    return {"official_sampling": {"temperature_matches_official": True}, "failed_requests": 0,
            "truncated_responses": 0, "sample_traces": [], **extra}


def test_the_gpqa_family_marks_a_partial_shard_not_comparable(monkeypatch):
    from gbench.runners.eval_suites import gpqa
    monkeypatch.setenv("GBENCH_SHARD", "3/20")
    out = gpqa._flag_comparability(_gpqa_result(shard="3/20"), None, 10, 198, gpqa.N_REPEATS,
                                   thinking=True)
    assert out["leaderboard_comparable"] is False
    assert "sharded run (--shard 3/20)" in out["leaderboard_comparable_reason"]
    assert out["shard_index"] == 3 and out["shard_count"] == 20 and out["shard_applied"]
    monkeypatch.delenv("GBENCH_SHARD")
    clean = gpqa._flag_comparability(_gpqa_result(), None, 10, 198, gpqa.N_REPEATS, thinking=True)
    assert clean["leaderboard_comparable"] is True
    assert "leaderboard_comparable_reason" not in clean


def test_longbench_settings_keep_the_shard_reason(monkeypatch):
    from gbench.runners.eval_suites import longbench_v2_code as L
    monkeypatch.setenv("GBENCH_SHARD", "2/4")
    res = {"shard": "2/4"}
    L._set_comparable(res, ["temperature 0.5 (LongBench-v2 uses 0.1)"])
    assert res["leaderboard_comparable"] is False
    reasons = res["leaderboard_comparable_reason"].split("; ")
    assert reasons[0].startswith("temperature 0.5")
    assert any(r.startswith("sharded run (--shard 2/4)") for r in reasons)
    res = {"shard": "2/4"}
    L._set_comparable(res, [])
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith("sharded run (--shard 2/4)")


def _function_source(module: str, name: str) -> str:
    tree = ast.parse((SUITES / f"{module}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment((SUITES / f"{module}.py").read_text(encoding="utf-8"),
                                          node)
    raise AssertionError(f"{module}.{name} not found")


#: suite module -> the function whose result is returned last (after its comparability rewrite)
_FINAL = {
    "gpqa": "_flag_comparability", "mmlu": "run_mmlu", "mmlu_pro": "run_mmlu_pro",
    "mmlu_redux": "run_mmlu_redux", "wmdp": "run_wmdp", "longbench_v2_code": "_set_comparable",
    "multipl_e": "run_multipl_e", "lcb": "run_lcb", "omnidocbench": "run_omnidocbench",
    "aime": "run_aime", "hmmt": "run_hmmt", "bigcodebench": "run_bigcodebench",
    "ojbench": "run_ojbench", "loft_scifact": "run_loft_scifact", "causalbench": "run_causalbench",
    "gsm8k": "run_gsm8k", "cyberseceval": "apply_comparability", "mrcr": "run_mrcr",
    "coco_caption": "run_coco_caption", "chartqa": "run_chartqa", "screenspot": "run_screenspot",
}


@pytest.mark.parametrize("module", sorted(_FINAL))
def test_each_suite_adds_the_shard_reason_after_its_comparability(module):
    assert "finalize_shard_fields(" in _function_source(module, _FINAL[module])


def test_a_sharded_gsm8k_run_is_not_comparable(monkeypatch):
    import datasets
    from gbench.runners.eval_suites import gsm8k
    rows = [{"question": f"q{i}?", "answer": f"work\n#### {i}"} for i in range(6)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    with mock.patch.object(base, "_send_single_request", _answer("The answer is 0.")):
        res = gsm8k.run_gsm8k("m", "http://x/v1", 1)
    assert res["shard"] == "1/2" and res["total_questions"] == 3
    assert res["leaderboard_comparable"] is False
    assert "sharded run (--shard 1/2)" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# --shard before --eval-limit                                                 #
# --------------------------------------------------------------------------- #
#: suite module -> its loader (each draws its own --eval-limit subset)
_LOADERS = {
    "aime": "_load_aime_samples", "hmmt": "_load_hmmt_samples",
    "amc_aime": "_load_amc_aime_samples", "chartqa": "_load_chartqa_samples",
    "docvqa": "_load_docvqa_samples", "textvqa": "_load_textvqa_samples",
    "infographicvqa": "_load_infographicvqa_samples", "charxiv": "_load_charxiv_samples",
    "coco_caption": "_load_coco_caption_samples", "screenspot": "_load_screenspot_samples",
    "multipl_e": "_load_multipl_e_samples", "spider2": "_load_spider2_samples",
    "bigcodebench": "_load_bigcodebench_samples", "ojbench": "_load_ojbench_samples",
    "cyberseceval": "_load_cyberseceval_samples", "codeforces": "_load_codeforces_samples",
    "lcb": "_load_lcb_samples", "aa_lcr": "_load_aa_lcr_samples",
    "beam_128k": "_load_beam_128k_samples", "browsecomp": "_load_browsecomp_samples",
    "cimemories": "_load_cimemories_samples", "deepsearch_qa": "_load_deepsearch_qa_samples",
    "frames": "_load_frames_samples", "gdpval": "_load_gdpval_samples",
    "healthbench": "_load_healthbench_samples",
    "humanitys_last_exam": "_load_humanitys_last_exam_samples",
    "i18n_translate": "_load_i18n_translate_samples",
    "imo_answer_bench": "_load_imo_answer_bench_samples",
    "loft_scifact": "_load_loft_scifact_samples", "simpleqa": "_load_simpleqa_samples",
    "omnidocbench": "_load_omnidocbench_samples", "mrcr": "_load_mrcr_samples",
}
_LIMIT_CALLS = ("stratified_sample", "limit_dataset", "select_pages", "screenspot_v2_targets",
                "stratified_indices")
_SHARD_CALLS = ("apply_shard", "_shard_rows")


def _calls(src: str, names) -> list:
    """Line numbers of the calls to `names` in `src`."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in names:
                out.append(node.lineno)
    return sorted(out)


@pytest.mark.parametrize("module", sorted(_LOADERS))
def test_every_loader_shards_the_full_set_before_its_limit_draw(module):
    src = _function_source(module, _LOADERS[module])
    if module == "screenspot":       # the ScreenSpot-Pro path shards in its own loader
        src += "\n" + _function_source(module, "_load_screenspot_pro_local")
    shards, limits = _calls(src, _SHARD_CALLS), _calls(src, _LIMIT_CALLS)
    assert shards, f"{module}: the loader applies no shard"
    assert limits, f"{module}: the loader draws no limit"
    assert min(shards) < max(limits), f"{module}: the limit is drawn before the shard"


def _run_eval_suite_samples_args(module: str) -> list:
    tree = ast.parse((SUITES / f"{module}.py").read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_eval_suite":
            for kw in node.keywords:
                if kw.arg == "samples":
                    out.append(kw.value)
    return out


@pytest.mark.parametrize("module", sorted(_LOADERS))
def test_base_keeps_exactly_the_loaders_shard(module):
    args = _run_eval_suite_samples_args(module)
    assert args, f"{module}: no run_eval_suite(samples=...) call"
    for arg in args:
        name = getattr(arg.func, "id", None) if isinstance(arg, ast.Call) else None
        assert name in ("_for_base", "presharded_for_base"), \
            f"{module}: run_eval_suite would shard the loader's shard again"


def _amc_aime_rows(monkeypatch, n=6):
    import datasets

    def _load(name, split=None, revision=None):
        cat = "AMC" if "amc" in name else "AIME"
        return [{"problem": f"{cat}-problem-{i}", "answer": str(i)} for i in range(n)]
    monkeypatch.setattr(datasets, "load_dataset", _load)


def _problems(res):
    return sorted(_text_of(t["messages"][-1]["content"]).split("\n")[0]
                  for t in res["sample_traces"])


def test_a_limited_sharded_run_draws_its_limit_from_its_own_shard(monkeypatch):
    from gbench.runners.eval_suites import amc_aime
    _amc_aime_rows(monkeypatch)
    full = [f"Problem: {c}-problem-{i}" for c in ("AMC", "AIME") for i in range(6)]
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    shard = set(full[1::3])
    with mock.patch.object(base, "_send_single_request", _answer("The final answer is 1")):
        res = amc_aime.run_amc_aime("m", "http://x/v1", 1, limit=2)
    got = _problems(res)
    assert len(got) == 2 and set(got) <= shard, got
    assert res["shard"] == "2/3" and res["total_questions"] == 2
    # stratified by competition within the shard
    assert {g.split("-")[0] for g in got} == {"Problem: AMC", "Problem: AIME"}


def test_the_shards_of_an_unlimited_run_partition_the_full_set(monkeypatch):
    from gbench.runners.eval_suites import amc_aime
    _amc_aime_rows(monkeypatch)
    seen = []
    for i in (1, 2, 3):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/3")
        with mock.patch.object(base, "_send_single_request", _answer("The final answer is 1")):
            seen.append(_problems(amc_aime.run_amc_aime("m", "http://x/v1", 1)))
    union = [p for part in seen for p in part]
    assert len(union) == len(set(union)) == 12


def test_an_empty_shard_is_an_honest_empty_row(monkeypatch):
    from gbench.runners.eval_suites import amc_aime
    _amc_aime_rows(monkeypatch, n=2)
    monkeypatch.setenv("GBENCH_SHARD", "7/7")
    with mock.patch.object(base, "_send_single_request", _answer("x")):
        res = amc_aime.run_amc_aime("m", "http://x/v1", 1)
    assert res["status"] == "success" and res["accuracy"] is None
    assert res["empty_shard"] is True and res["leaderboard_comparable"] is False


def test_an_image_dataset_is_sharded_by_row_before_the_limit(monkeypatch):
    """A toy split of 12 rows with --shard 2/4 and --eval-limit 2 yields 2 rows of rows[1::4]."""
    import datasets
    from gbench.runners.eval_suites import chartqa

    class _Split(list):
        def select(self, idx):
            return _Split(self[i] for i in idx)

    rows = _Split({"query": f"q{i}", "label": [str(i)], "type": "human" if i % 2 else "augmented",
                   "imgname": f"{i}.png", "image": {"bytes": b"\x89PNG" + bytes([i])}}
                  for i in range(12))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setenv("GBENCH_SHARD", "2/4")
    samples = chartqa._load_chartqa_samples(limit=2)
    want = {r["query"] for r in rows[1::4]}
    got = [_text_of(s[0][0]["content"]) for s in samples]
    assert len(got) == 2 and all(any(w in g for w in want) for g in got), got


def test_presharded_for_base_lays_a_shard_out_for_base():
    from gbench.runners.eval_suites.sampling import presharded_for_base, shard_select
    for n in (1, 2, 3, 5):
        for i in range(1, n + 1):
            for size in (0, 1, 4, 7):
                items = list(range(size))
                laid = presharded_for_base(items, {"GBENCH_SHARD": f"{i}/{n}"})
                assert shard_select(laid, (i, n)) == items
    assert presharded_for_base([1, 2], {}) == [1, 2]


def test_lcb_shards_the_concatenated_scenarios_as_base_would():
    from gbench.runners.eval_suites import lcb
    gen, exe, test = list(range(7)), list(range(100, 105)), list(range(200, 209))
    full = gen + exe + test
    for n in (1, 2, 3, 4):
        union = []
        for i in range(1, n + 1):
            with mock.patch.dict(os.environ, {"GBENCH_SHARD": f"{i}/{n}"}):
                part = (lcb._shard_rows(gen, 0) + lcb._shard_rows(exe, len(gen))
                        + lcb._shard_rows(test, len(gen) + len(exe)))
            assert part == full[i - 1::n]
            union += part
        assert sorted(union) == sorted(full)
    assert lcb._shard_rows(gen, 3) == gen            # no shard: every row


def test_screenspot_draws_its_limit_within_the_shard(monkeypatch):
    from gbench.runners.eval_suites import screenspot as S
    targets = [{"platform": p, "data_type": d, "instruction": f"{p}-{d}-{i}", "image_url": "u",
                "box": [0.1, 0.1, 0.2, 0.2], "data_source": "s", "img_filename": f"{i}.png"}
               for i, (p, d) in enumerate([(p, d) for p in ("mobile", "web") for d in
                                           ("text", "icon")] * 3)]
    asked = []

    def _targets(limit, **k):
        asked.append(limit)
        return targets if limit is None else targets[:limit]
    monkeypatch.setattr(S, "screenspot_v2_targets", _targets)
    monkeypatch.delenv("GBENCH_SCREENSPOT_DATA_DIR", raising=False)
    monkeypatch.delenv("SCREENSPOT_PRO_DATA_DIR", raising=False)
    monkeypatch.setenv("GBENCH_SHARD", "1/3")
    samples, _label = S._load_screenspot_samples(limit=2)
    assert asked == [None]                     # the full list, then shard, then limit
    shard = {t["instruction"] for t in targets[0::3]}
    got = [s[2]["img_filename"] for s in samples]
    assert len(samples) == 2
    assert {targets[int(g.split(".")[0])]["instruction"] for g in got} <= shard
    monkeypatch.delenv("GBENCH_SHARD")
    S._load_screenspot_samples(limit=2)
    assert asked == [None, 2]                  # unsharded: screenspot_v2_targets draws it


# --------------------------------------------------------------------------- #
# multipl_e                                                                   #
# --------------------------------------------------------------------------- #
def _mpe_samples(n=2, lang="lua"):
    gold = json.dumps({"language": lang, "name": "HumanEval_0", "prompt": "", "tests": "",
                       "stop_tokens": []})
    return [([{"role": "user", "content": f"q{i}"}], gold, {"category": lang,
                                                           "problem_id": f"{lang}/p{i}"})
            for i in range(n)]


def _evaluator_answers(names):
    """A stub docker run of the MultiPL-E evaluator that returns a result for `names` only."""
    def _run(cmd, **k):
        out = next(a.split(":", 1)[0] for a in cmd if a.endswith(":/out:rw"))
        for name in names:
            with open(os.path.join(out, f"{name}.results.json"), "w") as f:
                json.dump({"name": name, "results": [{"status": "OK", "exit_code": 0}]}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return _run


def test_multipl_e_a_completion_without_an_evaluator_result_is_a_harness_error(monkeypatch):
    """Through the real base: base marks every trace is_correct=False before the scorer runs,
    so a missing result must still be found."""
    from gbench.runners.eval_suites import multipl_e as M
    metrics = {}
    reply = "[[ ## completion ## ]]\nreturn 1\n\n[[ ## completed ## ]]"
    with mock.patch.object(base, "_send_single_request", _answer(reply)), \
            mock.patch.object(M.subprocess, "run", _evaluator_answers(["p000000"])):
        res = base.run_eval_suite(eval_name="multipl_e", model_name="m", base_url="http://x/v1",
                                  concurrency=1, samples=_mpe_samples(),
                                  async_eval_fn=M._make_scorer("img", 1, metrics))
    missed = [t for t in res["sample_traces"] if t.get("harness_error")]
    assert len(missed) == 1
    missed = missed[0]
    assert missed["status"] == "HARNESS_ERROR" and missed["scoring_excluded"] is True
    assert missed["harness_error"]["kind"] == "no_harness_result"
    assert metrics["multipl_e_unscored"] == 1
    assert res["status"] == "completed_with_errors"
    assert res["accuracy"] == 100.0                 # over the one scored completion


def test_multipl_e_every_completion_with_a_result_is_scored(monkeypatch):
    from gbench.runners.eval_suites import multipl_e as M
    metrics = {}
    reply = "[[ ## completion ## ]]\nreturn 1\n\n[[ ## completed ## ]]"
    with mock.patch.object(base, "_send_single_request", _answer(reply)), \
            mock.patch.object(M.subprocess, "run", _evaluator_answers(["p000000", "p000001"])):
        res = base.run_eval_suite(eval_name="multipl_e", model_name="m", base_url="http://x/v1",
                                  concurrency=1, samples=_mpe_samples(),
                                  async_eval_fn=M._make_scorer("img", 1, metrics))
    assert res["status"] == "success" and "multipl_e_unscored" not in metrics
    assert all(not t.get("scoring_excluded") for t in res["sample_traces"])


def test_multipl_e_problem_ids_are_unique_across_languages(monkeypatch):
    import datasets
    from gbench.runners.eval_suites import multipl_e as M
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, cfg, **k: [
        {"name": "HumanEval_0_has_close_elements", "prompt": "p", "tests": "t",
         "stop_tokens": []}])
    samples = M._load_multipl_e_samples(langs=["lua", "rb"])
    keys = {base.sample_key(s) for s in samples}
    assert len(keys) == 2 and "problem_id:lua/HumanEval_0_has_close_elements" in keys


def _mpe_trace(key, lang, ok, idx, attempt=0, excluded=False):
    return {"sample_key": key, "category": lang, "is_correct": ok, "sample_idx": idx,
            "source_sample_idx": idx, "attempt_idx": attempt, "global_idx": idx,
            **({"scoring_excluded": True, "harness_error": {"kind": "no_harness_result"}}
               if excluded else {})}


def test_multipl_e_shards_merge_to_the_per_problem_pass_at_1(monkeypatch):
    from gbench.runners.eval_suites import multipl_e as M
    # problem A: 2 scored completions, 1 correct; problem B: 1 scored (1 harness error), correct
    a = [_mpe_trace("problem_id:lua/A", "lua", True, 0, 0),
         _mpe_trace("problem_id:lua/A", "lua", False, 0, 1)]
    b = [_mpe_trace("problem_id:rb/B", "rb", True, 0, 0),
         _mpe_trace("problem_id:rb/B", "rb", False, 0, 1, excluded=True)]
    shard = lambda i, tr: {"eval_name": "multipl_e", "shard": f"{i}/2", "status": "success",
                           "sample_traces": tr, "accuracy": 50.0 if i == 1 else 100.0,
                           "category_accuracy": {}, "multipl_e_unscored": 0 if i == 1 else 1,
                           "leaderboard_comparable": False,
                           "leaderboard_comparable_reason": "sharded run (--shard 1/2): x"}
    merged = M.merge_shards([shard(1, a), shard(2, b)])
    assert merged["accuracy"] == 75.0             # mean over problems of c/n: (0.5 + 1) / 2
    assert merged["multipl_e_unscored"] == 1
    assert merged["category_accuracy"]["lua"]["accuracy"] == 50.0
    assert merged["category_accuracy"]["rb"]["accuracy"] == 100.0
    # the same traces in one unsharded run give the same headline
    whole = {"sample_traces": [dict(t, source_sample_idx=i) for i, t in
                               enumerate(a[:1] + a[1:] + b)]}
    whole["sample_traces"][1]["source_sample_idx"] = 0
    whole["sample_traces"][3]["source_sample_idx"] = 2
    overall, _ = M.per_problem_pass_at_1(whole["sample_traces"])
    assert round(overall, 2) == 75.0


# --------------------------------------------------------------------------- #
# omnidocbench                                                                #
# --------------------------------------------------------------------------- #
def _odb_trace(page, key, idx, text="# t", excluded=False):
    return {"sample_key": key, "global_idx": idx, "attempt_idx": 0, "is_correct": False,
            "category": "v1.5", "gold_answer": json.dumps({"page_name": page}),
            "response_text": text, "finish_reason": "stop", "health": "ok",
            **({"harness_error": {"kind": "endpoint_failure"}, "scoring_excluded": True}
               if excluded else {})}


def _odb_stubs(monkeypatch, tmp_path, pages):
    from gbench.runners.eval_suites import omnidocbench as O
    gt = tmp_path / "OmniDocBench.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": f"images/{p}"}, "layout_dets": []}
                              for p in pages]))
    monkeypatch.setattr(O, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(O, "hf_hub_download", lambda **k: str(gt))
    seen = {"calls": 0}

    def _docker(cmd, **k):
        mounts = {c.split(":")[1]: c.split(":")[0] for c in cmd
                  if isinstance(c, str) and c.startswith("/") and ":" in c}
        seen["calls"] += 1
        seen["preds"] = sorted(os.listdir(mounts["/workspace/data_md/predictions"]))
        # a stand-in composite: 10 points per scored page
        with open(os.path.join(mounts["/workspace/result"],
                               "predictions_quick_match_run_summary.json"), "w") as f:
            json.dump({"notebook_metric_summary": {"overall_notebook": 10.0 * len(seen["preds"]),
                                                   "metrics": {}}}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(O.subprocess, "run", _docker)
    return O, seen


def test_omnidocbench_shards_merge_by_rescoring_their_union(monkeypatch, tmp_path):
    O, seen = _odb_stubs(monkeypatch, tmp_path, ["a.png", "b.png", "c.png"])
    common = {"eval_name": "omnidocbench", "status": "success", "temperature": 0.0,
              "thinking": False, "leaderboard_comparable": False}
    s1 = dict(common, shard="1/2", accuracy=10.0, sample_traces=[
        _odb_trace("a.png", "k:a", 0), _odb_trace("c.png", "k:c", 2)],
        leaderboard_comparable_reason="sharded run (--shard 1/2): shard 1 of 2, a partial set")
    s2 = dict(common, shard="2/2", accuracy=10.0, sample_traces=[_odb_trace("b.png", "k:b", 1)],
              leaderboard_comparable_reason="sharded run (--shard 2/2): shard 2 of 2, a partial set")
    merged = O.merge_shards([s1, s2])
    assert seen["calls"] == 1 and seen["preds"] == ["a.md", "b.md", "c.md"]
    assert merged["accuracy"] == 30.0 and merged["omnidocbench_overall"] == 30.0
    assert merged["pages_scored"] == 3 and merged["total_questions"] == 3
    assert merged["leaderboard_comparable"] is True        # the full union, greedy, no errors
    assert "leaderboard_comparable_reason" not in merged


def test_omnidocbench_merge_names_a_missing_evaluator(monkeypatch, tmp_path):
    from gbench.merge import MergeError
    O, _seen = _odb_stubs(monkeypatch, tmp_path, ["a.png"])

    def _missing(image):
        raise RuntimeError("[INFRA REQUIRED] omnidocbench: image 'gbench-omnidocbench:193627ae' not found")
    monkeypatch.setattr(O, "_check_prereqs", _missing)
    shard = {"eval_name": "omnidocbench", "shard": "1/1", "status": "success", "accuracy": 1.0,
             "sample_traces": [_odb_trace("a.png", "k:a", 0)]}
    with pytest.raises(MergeError, match="evaluator"):
        O.merge_shards([shard])


def test_omnidocbench_an_empty_shard_is_not_handed_to_the_evaluator(monkeypatch, tmp_path):
    O, seen = _odb_stubs(monkeypatch, tmp_path, ["a.png"])
    monkeypatch.setattr(O, "_load_omnidocbench_samples", lambda limit=None: [])
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    res = O.run_omnidocbench("m", "http://x/v1")
    assert seen["calls"] == 0
    assert res["status"] == "success" and res["accuracy"] is None and res["empty_shard"]
    assert res["leaderboard_comparable"] is False


# --------------------------------------------------------------------------- #
# own-loop suites carry a shard-independent sample id                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module,function", [("mmlu", "run_mmlu"), ("mmlu_pro", "run_mmlu_pro"),
                                             ("mmlu_redux", "run_mmlu_redux"),
                                             ("wmdp", "run_wmdp")])
def test_own_loop_traces_carry_a_sample_key(module, function):
    assert '"sample_key": sample_key(' in _function_source(module, function) or \
        't["sample_key"] = sample_key(' in _function_source(module, function)


# --------------------------------------------------------------------------- #
# mrcr: the window of the model under test                                    #
# --------------------------------------------------------------------------- #
class _Listing:
    def __init__(self, data):
        self._body = json.dumps({"data": data}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def test_mrcr_takes_the_window_of_the_model_under_test(monkeypatch):
    import urllib.request
    from gbench.runners.eval_suites import mrcr
    listing = [{"id": "other/model", "max_model_len": 4096},
               {"id": "org/served", "max_model_len": 131072}]
    sent = {}

    def _open(req, timeout=None):
        sent["url"] = req.full_url
        sent["auth"] = req.get_header("Authorization")
        return _Listing(listing)
    monkeypatch.setattr(urllib.request, "urlopen", _open)
    monkeypatch.setenv("VLLM_API_KEY", "sk-test-endpoint-key")
    assert mrcr._get_server_max_model_len("http://x/v1", "org/served") == 131072
    assert sent["url"] == "http://x/v1/models" and sent["auth"] == "Bearer sk-test-endpoint-key"
    assert mrcr._get_server_max_model_len("http://x/v1", "served") == 131072   # last component
    assert mrcr._get_server_max_model_len("http://x/v1", "absent/thing") is None
    assert mrcr._get_server_max_model_len("http://x/v1") is None     # two models, none named
    listing[:] = [{"id": "only/model", "max_model_len": 8192}]
    assert mrcr._get_server_max_model_len("http://x/v1") == 8192


# --------------------------------------------------------------------------- #
# ojbench: OJBENCH_GET_TIMEOUT_S                                              #
# --------------------------------------------------------------------------- #
def test_ojbench_forwards_the_poll_interval_into_the_judge(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import ojbench as OJ
    monkeypatch.setattr(OJ, "_testdata_dir", lambda: str(tmp_path))
    seen = {}

    def _run(cmd, **k):
        seen.setdefault("cmd", cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(OJ.subprocess, "run", _run)
    monkeypatch.setenv("OJBENCH_GET_TIMEOUT_S", "300")
    with pytest.raises(RuntimeError, match="no results"):     # the stub judge wrote nothing
        OJ._judge_in_container([{"id": 1}], 2)
    cmd = seen["cmd"]
    assert "OJBENCH_GET_TIMEOUT_S=300" in cmd
    assert cmd.index("OJBENCH_GET_TIMEOUT_S=300") < cmd.index(OJ._image())
    monkeypatch.delenv("OJBENCH_GET_TIMEOUT_S")
    assert OJ._get_timeout_env() == []
    for bad in ("abc", "0", "-5"):
        monkeypatch.setenv("OJBENCH_GET_TIMEOUT_S", bad)
        with pytest.raises(ValueError, match="OJBENCH_GET_TIMEOUT_S"):
            OJ._get_timeout_env()


def test_gpqa_shards_merge_with_the_repeats_recomputed():
    from gbench.runners.eval_suites import gpqa, gpqa_diamond

    def copy(q, r, ok):
        return {"sample_key": f"sha256:{q}-{r}", "source_sample_idx": q, "attempt_idx": r,
                "global_idx": q * 10 + r, "is_correct": ok, "category": "c"}
    s1 = [copy(0, 0, True), copy(0, 1, False)]
    s2 = [copy(1, 0, True), copy(1, 1, True)]
    shard = lambda i, tr: {"eval_name": "gpqa_diamond", "shard": f"{i}/2", "status": "success",
                           "sample_traces": tr, "n_repeats": 2,
                           "accuracy": round(100.0 * sum(t["is_correct"] for t in tr) / len(tr), 2),
                           "repeats": {"n_repeats": 2, "questions": 1, "generations": 2}}
    merged = gpqa_diamond.merge_shards([shard(1, s1), shard(2, s2)])
    assert gpqa_diamond.merge_shards is gpqa.merge_shards
    assert merged["accuracy"] == 75.0
    rep = merged["repeats"]
    assert rep["questions"] == 2 and rep["generations"] == 4
    assert rep["per_repeat_accuracy"] == [100.0, 50.0]


def test_coco_caption_shards_merge_by_scoring_the_union(monkeypatch):
    from gbench.merge import MergeError
    from gbench.runners.eval_suites import coco_caption as C
    monkeypatch.setattr(C, "check_coco_caption_prerequisites", lambda: (True, ""))
    seen = {}

    def _metrics(gts, res):
        seen["images"] = sorted(gts)
        scores = {m: 0.5 for m in C.METRICS}
        scores["CIDEr"] = 0.9
        return scores, {iid: (1.0 if iid != 3 else 0.1) for iid in gts}
    monkeypatch.setattr(C, "compute_caption_metrics", _metrics)

    def trace(iid):
        return {"sample_key": f"sha256:{iid}", "global_idx": iid, "attempt_idx": 0,
                "extra_payload": {"image_id": iid}, "gold_answer": [f"ref {iid}"],
                "response_text": "a caption", "is_correct": True, "category": "captioning",
                "status": "OK", "finish_reason": "stop"}
    common = {"eval_name": "coco_caption", "status": "success", "temperature": 0.0,
              "official_sampling": {"temperature_matches_official": True}}
    s1 = dict(common, shard="1/2", accuracy=120.0, cider=1.2, sample_traces=[trace(1), trace(3)])
    s2 = dict(common, shard="2/2", accuracy=80.0, cider=0.8, sample_traces=[trace(2)])
    merged = C.merge_shards([s1, s2])
    assert seen["images"] == [1, 2, 3]
    assert merged["accuracy"] == 90.0 and merged["cider"] == 0.9     # CIDEr over the union
    assert merged["per_image_pass_rate"] == round(200 / 3, 2)
    assert merged["leaderboard_comparable"] is True
    monkeypatch.setattr(C, "check_coco_caption_prerequisites", lambda: (False, "no java"))
    with pytest.raises(MergeError, match="no java"):
        C.merge_shards([s1, s2])
