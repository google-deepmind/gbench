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

"""`--eval-limit 5` is 5 samples for the whole suite, in every code and math suite.

Before 2026-10-09, bigcodebench ran 5 tasks in each of its two splits (10 samples) and
putnam_formal 5 problems in each of its two settings (10), and lcb filled the 5 from code
generation alone. Each test here builds the suite's selection from synthetic data and checks the
total (and the count per setting), and that no limit keeps everything.
"""

import json
import logging
from collections import Counter

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites.sampling import allocate_limit


def _count(samples, key="category"):
    return Counter(m.get(key) for _, _, m in samples)


# --------------------------------------------------------------------------- lcb
def _lcb_stubs(monkeypatch, n=20):
    from gbench.runners.eval_suites import lcb as L
    gen = [{"question_id": f"q{i}", "difficulty": ("easy", "medium", "hard")[i % 3],
            "question_content": f"problem {i}", "starter_code": "", "contest_date": "2025-03-01",
            "public_test_cases": json.dumps([{"input": "1", "output": "1", "testtype": "stdin"}]),
            "private_test_cases": "[]", "metadata": "{}"} for i in range(n)]
    exe = [{"code": f"def f(x):\n    return x + {i}", "input": "f(1)", "output": str(1 + i)}
           for i in range(n)]
    tst = [{"question_id": f"t{i}", "question_content": "c", "starter_code": "",
            "test": json.dumps([{"input": "1", "output": "1"}]), "difficulty": "easy"}
           for i in range(n)]

    def fake_load(repo, *a, **k):
        return {L._LCB_EXEC_REPO: exe, L._LCB_TESTGEN_REPO: tst}[repo]
    import datasets
    monkeypatch.setattr(L, "_load_lcb_lite", lambda tag: gen)
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    return L


def test_lcb_limit_is_five_samples_spread_over_the_three_scenarios(monkeypatch):
    L = _lcb_stubs(monkeypatch)
    samples = L._load_lcb_samples(limit=5)
    by_scenario = Counter(L._scenario(m["category"]) for _, _, m in samples)
    assert len(samples) == 5
    assert dict(by_scenario) == {k: v for k, v in allocate_limit(
        5, ["code_generation", "code_execution", "test_output_prediction"], seed="lcb").items()
        if v}
    assert set(by_scenario) == {"code_generation", "code_execution", "test_output_prediction"}
    assert len(L._load_lcb_samples(limit=None)) == 60


def test_lcb_limit_without_test_output_prediction_spreads_over_two(monkeypatch):
    L = _lcb_stubs(monkeypatch)
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_TEST_GEN", "0")
    samples = L._load_lcb_samples(limit=5)
    by_scenario = Counter(L._scenario(m["category"]) for _, _, m in samples)
    assert len(samples) == 5 and set(by_scenario) == {"code_generation", "code_execution"}
    assert len(L._load_lcb_samples(limit=None)) == 40


# --------------------------------------------------------------------------- multipl_e
def test_multipl_e_limit_is_five_problems_over_all_languages(monkeypatch):
    from gbench.runners.eval_suites import multipl_e as ME
    import datasets

    def fake_load(name, cfg, split=None, revision=None):
        lang = cfg.split("-", 1)[1]
        return [{"name": f"HumanEval_{i}", "prompt": f"{lang} {i}", "tests": "t",
                 "stop_tokens": []} for i in range(10)]
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    langs = ["cpp", "java", "py", "rs", "go", "js"]
    samples = ME._load_multipl_e_samples(langs=langs, limit=5)
    assert len(samples) == 5 and max(_count(samples).values()) == 1
    assert len(ME._load_multipl_e_samples(langs=langs, limit=None)) == 60


# --------------------------------------------------------------------------- cruxeval
def test_cruxeval_limit_is_five_items_over_both_tasks(monkeypatch):
    from gbench.runners.eval_suites import cruxeval as C
    import datasets
    rows = [{"id": f"sample_{i}", "code": f"def f(x):\n    return x + {i}", "input": "1",
             "output": str(1 + i)} for i in range(10)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = C._load_cruxeval_samples(limit=5)
    assert len(samples) == 5
    assert sorted(_count(samples).values()) == [2, 3]
    assert len(C._load_cruxeval_samples(limit=None)) == 20


# --------------------------------------------------------------------------- bigcodebench
def _bcb_rows(n=10):
    return [{"task_id": f"BigCodeBench/{i}", "instruct_prompt": f"Do {i}.\n",
             "complete_prompt": f"def task_func_{i}():\n", "entry_point": "task_func"}
            for i in range(n)]


def test_bigcodebench_limit_is_five_samples_over_the_two_splits(monkeypatch):
    from gbench.runners.eval_suites import bigcodebench as B
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _bcb_rows())
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = B._load_bigcodebench_samples(limit=5, subset="full")
    counts = _count(samples, "split")
    assert len(samples) == 5
    assert dict(counts) == allocate_limit(5, B.SPLITS, seed="bigcodebench")
    # the split with fewer tasks runs a subset of the other's tasks
    tasks = {sp: {m["task_id"] for _, _, m in samples if m["split"] == sp} for sp in B.SPLITS}
    small, large = sorted(tasks.values(), key=len)
    assert small <= large
    full = B._load_bigcodebench_samples(limit=None, subset="full")
    assert len(full) == 20 and dict(_count(full, "split")) == {"instruct": 10, "complete": 10}


def test_bigcodebench_run_logs_and_records_the_plan(monkeypatch, caplog):
    from gbench.runners.eval_suites import bigcodebench as B
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _bcb_rows())
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "harness_cpus", lambda: [0])
    seen = {}

    def fake_run(**kw):
        seen["samples"] = [s for s in kw["samples"] if s is not None]
        return {"sample_traces": []}
    monkeypatch.setattr(B, "run_eval_suite", fake_run)
    with caplog.at_level(logging.INFO):
        out = B.run_bigcodebench("m", "http://x/v1", 1, limit=5)
    assert len(seen["samples"]) == 5
    plan = out["eval_limit_plan"]
    assert plan["limit"] == 5 and plan["samples"] == 5 and sum(plan["per_setting"].values()) == 5
    assert any(r.getMessage().startswith("[bigcodebench] --eval-limit 5: 5 sample(s) over 2 split")
               for r in caplog.records)


# --------------------------------------------------------------------------- codeforces
def test_codeforces_limit_is_five_problems(monkeypatch):
    from gbench.runners.eval_suites import codeforces as CF
    import datasets
    rows = [{"id": f"{i}/A", "contest_id": i, "official_tests": [{"input": "1", "output": "1"}],
             "generated_tests": 0, "contest_type": "CF"} for i in range(12)]
    prompts = [{"id": f"{i}/A", "language": "python", "prompt": f"p{i}"} for i in range(12)]
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda repo, cfg, **k: prompts if cfg == "verifiable-prompts" else rows)
    monkeypatch.setattr(CF, "_load_generated_tests", lambda contest, pid: [])
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    assert len(CF._load_codeforces_samples(limit=5)) == 5
    assert len(CF._load_codeforces_samples(limit=None)) == 12


# --------------------------------------------------------------------------- ojbench
def test_ojbench_limit_is_five_problems_over_both_languages(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import ojbench as OJ
    import huggingface_hub
    path = tmp_path / "full.jsonl"
    path.write_text("".join(json.dumps({"id": i, "prompt": f"p{i}", "language": lang,
                                        "dataset": "NOI", "difficulty": "easy"}) + "\n"
                            for i in range(10) for lang in ("python", "cpp")))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: str(path))
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = OJ._load_ojbench_samples(limit=5)
    assert len(samples) == 5 and sorted(_count(samples).values()) == [2, 3]
    assert len(OJ._load_ojbench_samples(limit=None)) == 20


# --------------------------------------------------------------------------- scicode
def test_scicode_limit_is_five_problems_whatever_their_sub_steps(monkeypatch):
    from gbench.runners.eval_suites import scicode as S
    import datasets
    rows = [{"problem_id": str(i), "problem_name": f"P{i}", "required_dependencies": "",
             "sub_steps": [{"step_number": f"{i}.{j}"} for j in range(1, 4)]}
            for i in range(1, 13)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [dict(r) for r in rows])
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    assert len(S._load_scicode_problems(limit=5)) == 5
    assert len(S._load_scicode_problems(limit=None)) == 12


# --------------------------------------------------------------------------- aider_polyglot
def test_aider_limit_is_five_exercises_over_the_languages(monkeypatch):
    from gbench.runners.eval_suites import aider_polyglot as A
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    exercises = sorted(f"{lang}/ex{i}" for lang in ("cpp", "go", "java", "javascript",
                                                     "python", "rust") for i in range(5))
    selected, _label, _pos = A.select_exercises(exercises, 5)
    assert len(selected) == 5
    assert max(Counter(e.split("/")[0] for e in selected).values()) == 1
    assert A.select_exercises(exercises, None)[0] == exercises


# --------------------------------------------------------------------------- livebench
def test_livebench_limit_is_five_questions_over_the_categories(monkeypatch):
    from gbench.runners.eval_suites import livebench as LB
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    cats = ["math", "coding", "reasoning", "language", "data_analysis", "instruction_following"]
    rows = [(c, f"{c}_task", f"{c}-{i}") for c in cats for i in range(4)]
    selected, _l, _p, n_full = LB.select_questions(rows, cats, 5)
    assert len(selected) == 5 and n_full == 24
    assert max(Counter(q.split("-")[0] for q in selected).values()) == 1
    assert len(LB.select_questions(rows, cats, None)[0]) == 24


# --------------------------------------------------------------------------- matharena suites
@pytest.mark.parametrize("suite", ["aime", "hmmt"])
def test_matharena_suites_limit_is_five_problems(monkeypatch, suite):
    import importlib
    import datasets
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    rows = [{"problem_idx": i, "problem": f"Problem {i}", "answer": i,
             "problem_type": ["Algebra", "Geometry"][i % 2]} for i in range(1, 31)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    load = mod._load_aime_samples if suite == "aime" else mod._load_hmmt_samples
    assert len(load(limit=5)) == 5
    assert len(load(limit=None)) == 30


def test_amc_aime_limit_is_five_problems_over_both_competitions(monkeypatch):
    from gbench.runners.eval_suites import amc_aime as AA
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [
        {"problem": f"p{i}", "answer": str(i)} for i in range(10)])
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = AA._load_amc_aime_samples(limit=5)
    assert len(samples) == 5 and sorted(_count(samples).values()) == [2, 3]
    assert len(AA._load_amc_aime_samples(limit=None)) == 20


def test_imo_answer_bench_limit_is_five_problems(monkeypatch):
    from gbench.runners.eval_suites import imo_answer_bench as IMO
    rows = [{"Problem ID": f"imo-{i}", "Problem": f"p{i}", "Short Answer": str(i),
             "Category": ["Algebra", "Combinatorics", "Geometry", "Number theory"][i % 4]}
            for i in range(40)]
    monkeypatch.setattr(IMO, "_fetch_csv", lambda: "")
    monkeypatch.setattr(IMO, "parse_answerbench_csv", lambda text: [dict(r) for r in rows])
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = IMO._load_imo_answer_bench_samples(limit=5)
    assert len(samples) == 5
    assert len(IMO._load_imo_answer_bench_samples(limit=None)) == 40


def test_putnam_limit_is_five_problems(monkeypatch):
    from gbench.runners.eval_suites import putnam as P
    monkeypatch.setattr(P, "_informal_rows", lambda: [
        {"informal_statement": f"s{i}", "informal_solution": str(i),
         "tags": [["algebra", "analysis", "geometry"][i % 3]]} for i in range(30)])
    assert len(P._load_putnam_samples(limit=5)) == 5
    assert len(P._load_putnam_samples(limit=None)) == 30


# --------------------------------------------------------------------------- putnam_formal
def _putnam_formal_release(monkeypatch, n=12):
    from gbench.runners.eval_suites import putnam_formal as pf
    lean = {f"putnam_19{i:02d}_a1": f"theorem putnam_19{i:02d}_a1 : True := sorry"
            for i in range(n)}
    informal = [{"problem_name": name, "tags": [["algebra", "analysis"][i % 2]]}
                for i, name in enumerate(sorted(lean))]
    monkeypatch.setattr(pf, "load_putnambench", lambda: {"lean": lean, "informal": informal})
    monkeypatch.setattr(pf, "N_PROBLEMS", n)
    return pf


def test_putnam_formal_limit_is_five_samples_over_the_two_settings(monkeypatch):
    pf = _putnam_formal_release(monkeypatch)
    samples = pf._load_putnam_formal_samples(limit=5)
    counts = _count(samples)
    assert len(samples) == 5
    assert dict(counts) == allocate_limit(5, pf.SETTINGS, seed="putnam_formal")
    full = pf._load_putnam_formal_samples(limit=None)
    assert len(full) == 24 and dict(_count(full)) == {s: 12 for s in pf.SETTINGS}


def test_putnam_formal_run_records_the_plan_and_a_subset_reason(monkeypatch, caplog):
    pf = _putnam_formal_release(monkeypatch)
    monkeypatch.setattr(pf, "check_putnam_formal_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(pf, "_lean_image_provenance", lambda: {"reproducible": True})
    seen = {}

    def fake_run(**kw):
        seen["samples"] = kw["samples"]
        return {"sample_traces": []}
    monkeypatch.setattr(pf, "run_eval_suite", fake_run)
    with caplog.at_level(logging.INFO):
        out = pf.run_putnam_formal("m", "http://x/v1", 1, limit=5)
    assert len(seen["samples"]) == 5
    plan = out["eval_limit_plan"]
    assert plan["samples"] == 5 and plan["per_setting"] == allocate_limit(
        5, pf.SETTINGS, seed="putnam_formal")
    assert any(r.getMessage().startswith("[putnam_formal] --eval-limit 5: 5 sample(s) over 2 ")
               for r in caplog.records)
    assert "--eval-limit subset" in out["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- arc_agi
def test_arc_agi_limit_is_five_tasks_with_all_their_test_pairs(monkeypatch):
    from gbench.runners.eval_suites import arc_agi as ARC
    grid = [[1, 2], [3, 4]]
    tasks = [{"id": f"t{i}", "train": [{"input": grid, "output": grid}],
              "test": [{"input": grid, "output": grid}] * (1 + i % 2)} for i in range(12)]
    monkeypatch.setattr(ARC, "_load_arc_agi_tasks", lambda: tasks)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = ARC._load_arc_agi_samples(limit=5)
    assert len({m["task_id"] for _, _, m in samples}) == 5
    full = ARC._load_arc_agi_samples(limit=None)
    assert len({m["task_id"] for _, _, m in full}) == 12 and len(full) == 18


# --------------------------------------------------------------------------- base-routed suites
@pytest.mark.parametrize("suite", ["gsm8k", "causalbench"])
def test_base_routed_suites_send_five_requests(monkeypatch, suite):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    if suite == "gsm8k":
        samples = [([{"role": "user", "content": f"q{i}"}], str(i), {}) for i in range(20)]
        monkeypatch.setattr(mod, "_load_gsm8k_samples", lambda **k: samples)
    else:
        monkeypatch.setattr(mod, "_require_nltk", lambda: None)
        samples = [([{"role": "user", "content": f"q{i}"}], "entailment",
                    {"category": f"{2 + i % 3}_vars"}) for i in range(20)]
        monkeypatch.setattr(mod, "_load_causalbench_samples", lambda **k: samples)
    sent = []

    async def fake_send(**k):
        sent.append(k)
        return base.Reply("The answer is 1.", None, "stop", completion_tokens=5)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    getattr(mod, f"run_{suite}")("m", "http://127.0.0.1:1/v1", 2, limit=5)
    assert len(sent) == 5
    sent.clear()
    getattr(mod, f"run_{suite}")("m", "http://127.0.0.1:1/v1", 2)
    assert len(sent) == 20
