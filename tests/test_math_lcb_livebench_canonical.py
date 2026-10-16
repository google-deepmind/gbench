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

"""Canonical fidelity: hmmt, amc_aime, lcb, livebench, aime, codeforces.

Offline: datasets, the model endpoint and the LiveBench container are all stubbed. The LCB
graders run for real (unsandboxed under conftest's GBENCH_SANDBOX=none).
"""

import json
import os
import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import tau_common as TC_GUARD


def _fake_load_dataset(rows_by_path, calls):
    def _load(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        if path not in rows_by_path:
            raise RuntimeError(f"offline: {path}")
        return rows_by_path[path]
    return _load


# --- hmmt ---------------------------------------------------------------------------------

HMMT_ROWS = [
    {"problem_idx": 1, "problem": "Compute $\\{1\\}$.", "answer": "103", "problem_type": ["Number Theory"]},
    {"problem_idx": 2, "problem": "Find x.", "answer": "\\frac{1}{576}", "problem_type": ["Algebra", "Number Theory"]},
]


def test_hmmt_prompt_is_matharenas_instruction_first_verbatim():
    from gbench.runners.eval_suites import hmmt as H
    # configs/competitions/hmmt/hmmt_feb_2026.yaml renders to this; runner.py puts the
    # instruction before "\n\n{problem}" and adds no CoT trigger.
    assert H._hmmt_prompt("P?") == "Put your final answer within \\boxed{}.\n\nP?"
    # braces in the problem are left alone (MathArena substitutes into the template only)
    assert H._hmmt_prompt("{x}") == "Put your final answer within \\boxed{}.\n\n{x}"


def test_hmmt_loader_pins_revision_and_uses_problem_type():
    from gbench.runners.eval_suites import hmmt as H
    calls = []
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"MathArena/hmmt_feb_2026": HMMT_ROWS}, calls)):
        samples = H._load_hmmt_samples()
    assert calls[0][2].get("revision") == H._REVISION
    assert [m["category"] for _, _, m in samples] == ["Number Theory", "Algebra/Number Theory"]
    msgs = samples[0][0]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert msgs[0]["content"].startswith("Put your final answer within \\boxed{}.\n\n")


def test_hmmt_comparability_names_every_gap():
    from gbench.runners.eval_suites import matharena_common as MA
    single = MA.comparability_reasons({"temperature": 0.0}, {}, 5, base.DEFAULT_TEMPERATURE)
    joined = "; ".join(single)
    assert "1 run(s) per problem" in joined and "averages 4" in joined
    assert "temperature=0.0" in joined and "--eval-limit 5" in joined
    # MathArena's setup (4 runs at the recommended temperature, every response graded) is
    # comparable now that the grader and the last-chance reprompt are MathArena's own.
    best = MA.comparability_reasons(
        {"temperature": base.DEFAULT_TEMPERATURE, "attempts": {"attempts_per_sample": 4}}, {},
        None, base.DEFAULT_TEMPERATURE)
    assert best == []
    # model-agnostic wording, and grading/request failures are named
    assert "provider-recommended temperature" in joined
    bad = MA.comparability_reasons(
        {"temperature": 1.0, "attempts": {"attempts_per_sample": 4}},
        {"grading_errors": 1, "last_chance_failures": 2, "ungraded_failed_requests": 3}, None, 1.0)
    assert len(bad) == 3


def test_run_hmmt_sets_flag_and_dataset(monkeypatch):
    from gbench.runners.eval_suites import hmmt as H
    from gbench.runners.eval_suites import matharena_common as MA
    monkeypatch.setattr(H, "_load_hmmt_samples", lambda limit=None: [])
    from gbench.runners.eval_suites import ojbench as OJ
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",   # a Gemma 4 model is served
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MA, "self_check", lambda config: None)
    seen = {}

    def _fake_run(**kw):
        seen.update(kw)
        return {"accuracy": 50.0, "temperature": 1.0, "attempts": {"attempts_per_sample": 4}}
    monkeypatch.setattr(H, "run_eval_suite", _fake_run)
    res = H.run_hmmt("m", "http://x/v1", 1, enable_thinking=True)
    assert seen["attempt_count"] == 4 and seen["supports_attempts"] is True
    assert res["leaderboard_comparable"] is True
    assert res["dataset"].endswith(H._REVISION)
    assert res["competition"] == "HMMT February 2026"
    assert res["matharena"]["commit"] == MA.MATHARENA_COMMIT
    assert "matharena_status" not in res


def test_hmmt_docs_disclose_the_non_current_competition_and_drop_the_gemma_claim():
    text = open(os.path.join(os.path.dirname(__file__), "..", "docs", "evals", "hmmt.md")).read()
    assert "no longer lists as current" not in text and "hmmt_feb_2026" in text
    assert "MathArena samples Gemma" not in text
    assert "--max-output-tokens 65536" in text


# --- amc_aime -----------------------------------------------------------------------------

AMC = [{"problem": f"a{i}", "answer": f"{i}.0"} for i in range(6)]
AIME = [{"problem": f"b{i}", "answer": str(i)} for i in range(6)]


def test_amc_aime_one_dataset_failing_is_a_hard_error():
    from gbench.runners.eval_suites import amc_aime as A
    calls = []
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"AI-MO/aimo-validation-amc": AMC}, calls)):
        with pytest.raises(RuntimeError, match="AIME dataset could not be loaded"):
            A._load_amc_aime_samples()
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"AI-MO/aimo-validation-aime": AIME}, [])):
        with pytest.raises(RuntimeError, match="AMC dataset could not be loaded"):
            A._load_amc_aime_samples(limit=4)          # a limited run is no exception


def test_amc_aime_pins_revisions_and_stratifies_the_limit():
    from gbench.runners.eval_suites import amc_aime as A
    calls = []
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"AI-MO/aimo-validation-amc": AMC, "AI-MO/aimo-validation-aime": AIME}, calls)):
        full = A._load_amc_aime_samples()
        sub = A._load_amc_aime_samples(limit=4)
    revs = {c[0]: c[2].get("revision") for c in calls}
    assert revs == {d: r for _, d, r in A._SOURCES}
    assert len(full) == 12 and full[0][1] == "0"          # float gold normalised
    assert sorted(m["category"] for _, _, m in sub) == ["AIME", "AIME", "AMC", "AMC"]


# --- aime ---------------------------------------------------------------------------------

AIME26 = [{"problem_idx": 2, "problem": "q", "answer": 62},
          {"problem_idx": 1, "problem": "p", "answer": 277}]


def test_aime_pins_default_revisions_and_hard_errors_on_missing_reference():
    from gbench.runners.eval_suites import aime as A
    calls = []
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"MathArena/aime_2026": AIME26}, calls)):
        samples = A._load_aime_samples()
    assert [(c[0], c[2].get("revision")) for c in calls] == [("MathArena/aime_2026", A._REVISION)]
    assert [g for _, g, _ in samples] == ["277", "62"]         # problem_idx order, str gold
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset({}, [])):
        with pytest.raises(RuntimeError, match="MathArena/aime_2026"):
            A._load_aime_samples()


def test_aime_override_is_recorded_unpinned():
    # There is no dataset override any more: the suite always reads the pinned MathArena set.
    from gbench.runners.eval_suites import aime as A
    assert not hasattr(A, "_AIME_POSTCUTOFF_DATASET")
    assert A._DATASET == "MathArena/aime_2026"


def test_aime_reason_reflects_the_resolved_temperature(monkeypatch):
    from gbench.runners.eval_suites import aime as A
    from gbench.runners.eval_suites import matharena_common as MA
    monkeypatch.setattr(A, "_load_aime_samples", lambda *a, **k: [])
    from gbench.runners.eval_suites import ojbench as OJ
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",   # a Gemma 4 model is served
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MA, "self_check", lambda config: None)
    monkeypatch.setattr(A, "run_eval_suite", lambda **kw: {
        "accuracy": 10.0, "temperature": 0.0, "attempts": {"attempts_per_sample": 4},
        "category_accuracy": {}, "sample_traces": []})
    res = A.run_aime("m", "http://x/v1", 1, enable_thinking=False)
    assert res["leaderboard_comparable"] is False
    assert "temperature=0.0" in res["leaderboard_comparable_reason"]
    assert res["dataset"].endswith(A._REVISION)


# --- codeforces ---------------------------------------------------------------------------

def _cf_row(i, declared=0):
    return {"id": f"{1000 + i}/A", "contest_id": str(1000 + i), "contest_type": "CF",
            "official_tests": [{"input": "1", "output": "2"}], "generated_checker": None,
            "time_limit": 2.0, "memory_limit": 256.0, "testset_size": 10,
            "official_tests_complete": False, "generated_tests": str(declared),
            "language": "python", "prompt": "Solve it in Python."}


def test_codeforces_pins_revision_for_problems_and_prompts():
    from gbench.runners.eval_suites import codeforces as CF
    calls = []
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"open-r1/codeforces": [_cf_row(0)]}, calls)):
        CF._load_codeforces_samples()
    assert len(calls) == 2
    assert all(c[2].get("revision") == CF._REVISION for c in calls)


def test_codeforces_incomplete_generated_tests_mirror_hard_errors(monkeypatch):
    from gbench.runners.eval_suites import codeforces as CF
    monkeypatch.setattr(CF, "_load_generated_tests",
                        lambda cid, pid: [{"input": "1", "output": "2"}] * 3)
    ok = [_cf_row(0, declared=3)]
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"open-r1/codeforces": ok}, [])):
        assert len(CF._load_codeforces_samples()[0][1]["generated_tests"]) == 3
    short = [_cf_row(0, declared=29)]           # shard holds 3 of the 29 declared cases
    with mock.patch("datasets.load_dataset", side_effect=_fake_load_dataset(
            {"open-r1/codeforces": short}, [])):
        with pytest.raises(RuntimeError, match="missing cases for 1 problem"):
            CF._load_codeforces_samples()


# --- livebench ----------------------------------------------------------------------------

def test_livebench_command_forwards_budget_and_model_config():
    from gbench.runners.eval_suites import livebench as L
    cmd = L._build_command("img", "/out", "google/gemma", "disp", "http://h:8000/v1", 4, 1.0,
                           "2025-04-02", "", max_tokens=32768,
                           model_config="/out/gbench_model_config.yml")
    inner = cmd[-1]
    assert inner.startswith("cp -f /out/gbench_model_config.yml ")
    assert "MODEL_CONFIGS_DIR" in inner
    assert "--model gbench-served-model --model-display-name disp" in inner
    assert "--max-tokens 32768" in inner and "--force-temperature 1.0" in inner
    # the operator's extra args come after gbench's budget, so theirs wins in argparse
    inner2 = L._build_command("img", "/out", "m", "d", "http://h/v1", 1, 0.0, "",
                              "--max-tokens 999", max_tokens=16384)[-1]
    assert inner2.index("--max-tokens 16384") < inner2.index("--max-tokens 999")


def test_livebench_model_config_carries_the_reasoning_toggle(tmp_path):
    import yaml
    from gbench.runners.eval_suites import livebench as L
    for think in (True, False):
        path = L._write_model_config(str(tmp_path), "google/gemma-4-E4B-it", think)
        assert path == "/out/gbench_model_config.yml"
        cfg = list(yaml.safe_load_all(open(tmp_path / L._MODEL_CONFIG_FILE)))[0]
        assert cfg["display_name"] == "gbench-served-model"
        assert cfg["api_name"] == {"local": "google/gemma-4-E4B-it"}
        assert cfg["api_kwargs"]["default"]["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": think}
        assert "max_tokens" not in cfg["api_kwargs"]["default"]   # would override --max-tokens


def test_livebench_budget_resolution():
    # The token-limit policy: no-think sends gen_api_answer.py's --max-tokens default 4096; a
    # thinking run gets the run's budget (or the thinking floor, fitted to the window).
    from gbench.runners.eval_suites import livebench as L
    S = L.OFFICIAL_SAMPLING
    assert L._budget(S, True, None)[:2] == (base.THINKING_MIN_OUTPUT_TOKENS, "default")
    assert L._budget(S, False, None)[:2] == (L.LIVEBENCH_MAX_TOKENS, "official")
    assert L._budget(S, False, 65536)[:2] == (4096, "official")
    assert L._budget(S, True, 65536)[:2] == (65536, "cli:--max-output-tokens")
    base.set_run_knobs(max_model_len=32768)
    assert L._budget(S, True, None)[:2] == (32768 - L._PROMPT_RESERVE_TOKENS,
                                            "default:fit-to-window")
    assert L._budget(S, True, 65536)[:2] == (65536, "cli:--max-output-tokens")  # never lowered


def _run_livebench_stubbed(monkeypatch, tmp_path, csv_text, **kwargs):
    from gbench.runners.eval_suites import livebench as L
    monkeypatch.setattr(L, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(L, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None,
                                              "commit": L.LIVEBENCH_COMMIT})
    monkeypatch.setattr(L.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    seen = {}

    def fake_run(cmd, **_kw):
        seen["cmd"] = cmd
        (tmp_path / "all_groups.csv").write_text(csv_text, encoding="utf-8")
        answers = tmp_path / "model_answer" / "data" / "live_bench" / "math" / "x" / "model_answer"
        answers.mkdir(parents=True, exist_ok=True)
        (answers / "google-gemma-4-e4b-it.jsonl").write_text(json.dumps(
            {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
             "total_output_tokens": 10}) + "\n", encoding="utf-8")
        if "instruction_following" in csv_text:   # scored: its answer rows are copied out too
            if_dir = (tmp_path / "model_answer" / "data" / "live_bench" / "instruction_following"
                      / "paraphrase" / "model_answer")
            if_dir.mkdir(parents=True, exist_ok=True)
            (if_dir / "google-gemma-4-e4b-it.jsonl").write_text(json.dumps(
                {"question_id": "i1", "choices": [{"index": 0, "turns": ["ok"]}],
                 "total_output_tokens": 10}) + "\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    return L.run_livebench("google/gemma-4-E4B-it", "http://h:8000/v1", 2, **kwargs), seen


def test_run_livebench_records_settings_and_comparability(monkeypatch, tmp_path):
    csv_text = ("model,average,coding,data_analysis,instruction_following,language,math,reasoning\n"
                "google-gemma-4-e4b-it,50,40,50,60,50,60,40\n")
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "2025-04-02")
    monkeypatch.delenv("GBENCH_LIVEBENCH_CATEGORIES", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_EXTRA_ARGS", raising=False)
    res, seen = _run_livebench_stubbed(monkeypatch, tmp_path, csv_text, enable_thinking=True)
    assert "--max-tokens 32768" in seen["cmd"][-1]
    assert res["thinking"] is True and res["max_output_tokens"] == 32768
    # a thinking run samples at 1.0 (base.resolve_temperature), its only departure here
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking; sent as the config's temperature) (LiveBench samples at "
        "temperature 0, or the question's required temperature)")
    cfg = json.load(open(tmp_path / "gbench_model_config.yml"))
    assert cfg["api_kwargs"]["default"]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
    assert cfg["api_kwargs"]["default"]["temperature"] == 1.0
    assert "--force-temperature" not in seen["cmd"][-1]


def test_run_livebench_reasons_and_extra_args_budget(monkeypatch, tmp_path):
    csv_text = "model,average,math\ngoogle-gemma-4-e4b-it,60,60\n"
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "latest")   # unpinned: LiveBench's default
    monkeypatch.setenv("GBENCH_LIVEBENCH_CATEGORIES", "math")
    monkeypatch.setenv("GBENCH_LIVEBENCH_EXTRA_ARGS", "--max-tokens 8000")
    res, seen = _run_livebench_stubbed(monkeypatch, tmp_path, csv_text, enable_thinking=False)
    assert res["max_output_tokens"] == 8000
    assert res["max_output_tokens_source"] == "env:GBENCH_LIVEBENCH_EXTRA_ARGS"
    assert seen["cmd"][-1].count("--max-tokens") == 1       # gbench does not add a second one
    assert res["leaderboard_comparable"] is False
    assert "unpinned release" in res["leaderboard_comparable_reason"]
    assert "category subset" in res["leaderboard_comparable_reason"]


def test_roadmap_no_longer_claims_livebench_scores_math_only():
    text = open(os.path.join(os.path.dirname(__file__), "..", "docs", "evals_roadmap.md"),
                encoding="utf-8").read()
    assert "scores the math category only" not in text
    assert "all 6 core categories" in text


# --- lcb ----------------------------------------------------------------------------------

def test_lcb_generation_prompt_is_lcb_runners_openai_chat_prompt():
    from gbench.runners.eval_suites import lcb as L
    assert L._SYSTEM_MESSAGE_GENERIC == (
        "You are an expert Python programmer. You will be given a question (problem specification) "
        "and will generate a correct Python program that matches the specification and passes all tests.")
    stdin = L._generation_prompt("Q", "")
    assert stdin == (
        "### Question:\nQ\n\n### Format: Read the inputs from stdin solve the problem and write the "
        "answer to stdout (do not directly test on the sample inputs). Enclose your code within "
        "delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs "
        "the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n")
    starter = L._generation_prompt("Q", "class Solution:\n    def f(self):")
    assert starter == (
        "### Question:\nQ\n\n### Format: You will use the following starter code to write the "
        "solution to the problem and enclose your code within delimiters.\n```python\n"
        "class Solution:\n    def f(self):\n```\n\n### Answer: (use the provided format with backticks)\n\n")


def test_lcb_extract_code_takes_the_last_block():
    from gbench.runners.eval_suites import lcb as L
    out = "```python\nprint(1)\n```\nfixed:\n```python\nprint(2)\n```"
    assert L._extract_code(out) == "print(2)"
    assert L._extract_code("print(2)") == ""                # no fenced block -> empty program


def test_lcb_loader_pins_revision_and_reads_func_name(monkeypatch):
    from gbench.runners.eval_suites import lcb as L
    seen = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda repo, f, **kw: seen.append((repo, f, kw)) or f"/nonexistent/{f}")
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: [])
    L._load_lcb_lite("v6")
    assert seen == [(L._LCB_REPO, "test6.jsonl", {"repo_type": "dataset", "revision": L._LCB_REVISION})]
    assert L._func_name('{"func_name": "twoSum"}') == "twoSum"
    assert L._func_name("{}") is None and L._func_name({"func_name": "f"}) == "f"


def test_lcb_stdio_grader_is_line_wise_with_decimal_fallback():
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample as V
    gold = {"task": "generation", "fn_name": None,
            "tests": [{"input": "", "output": "0.50\n1", "testtype": "stdin"}]}
    assert V("```python\nprint(0.5)\nprint(1)\n```", gold) is True      # Decimal equality
    assert V("```python\nprint(0.5)\nprint(2)\n```", gold) is False
    assert V("```python\nprint('0.5 1')\n```", gold) is False             # line count differs


def test_lcb_call_based_grader_uses_func_name_and_accepts_tuples():
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample as V
    gold = {"task": "generation", "fn_name": "pair",
            "tests": [{"input": "3", "output": "[3, 4]", "testtype": "functional"}]}
    helper_first = ("```python\nclass Solution:\n    def aVeryLongHelperName(self, x):\n"
                    "        return x\n    def pair(self, x):\n        return (x, x + 1)\n```")
    assert V(helper_first, gold) is True     # entry point from metadata, not a name heuristic


def test_lcb_execution_and_test_output_scoring():
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample as V
    ex = {"task": "execution", "code": "def f(a):\n    return a * 2", "input": "f(a = 3)",
          "output": "6"}
    assert V("[ANSWER]\nassert f(a = 3) == 6\n[/ANSWER]", ex) is True
    assert V("assert f(a = 3) == f(a = 3)\n[/ANSWER]", ex) is False       # repeats the call
    tg = {"task": "test_gen", "output": "[1, 2]"}
    assert V("```python\nassert g(1) == [1, 2]\n```", tg) is True
    assert V("```python\nassert g(1) == [2, 1]\n```", tg) is False


def test_lcb_vendored_graders_are_upstream_files():
    from gbench.runners.eval_suites import lcb as L
    assert "def grade_stdio(" in L._LCB_TESTING_UTIL and "def run_test(" in L._LCB_TESTING_UTIL
    assert "def unsafe_execute(" in L._LCB_UTILS_EXECUTE
    assert L._LCB_BASE_IMPORTS.startswith("from itertools import accumulate")
    assert "def check_testcase_output(" in L._LCB_TEST_OUTPUT_PREDICTION


def _lcb_result(k, temperature):
    traces = []
    for src in range(2):
        for a in range(k):
            traces.append({"sample_idx": len(traces), "source_sample_idx": src,
                           "category": "code_gen_easy", "is_correct": src == 0,
                           "extra_payload": {"contest_date": "2025-01-04"}})
    traces.append({"sample_idx": len(traces), "source_sample_idx": 99,
                   "category": "code_execution", "is_correct": True})
    traces.append({"sample_idx": len(traces), "source_sample_idx": 98,
                   "category": "test_gen_medium", "is_correct": True})
    res = {"accuracy": 0.0, "temperature": temperature, "sample_traces": traces,
           "correct_answers": 0, "total_questions": len(traces), "max_tokens_sent": [2000]}
    if k > 1:
        res["attempts"] = {"attempts_per_sample": k, "avg_at_k": 0.0}
    return res


@pytest.mark.parametrize("k,temp,limit,comparable", [
    (10, 0.2, None, True),
    (1, 0.2, None, False),
    (10, 0.0, None, False),
    (10, 0.2, 5, False),
])
def test_lcb_comparability_follows_lcb_runner_sampling(monkeypatch, k, temp, limit, comparable):
    from gbench.runners.eval_suites import lcb as L
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    # no model to repair with here: a stand-in self-repair report (all four scenarios run)
    monkeypatch.setattr(L, "_run_self_repair", lambda traces, **kw: {
        "correct": 1, "total": 2, "accuracy": 50.0, "failed_requests": 0})
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **kw: [])
    seen = {}

    def fake_suite(**kw):
        seen.update(kw)
        return _lcb_result(k, temp)

    monkeypatch.setattr(L, "run_eval_suite", fake_suite)
    res = L.run_lcb("m", "http://x/v1", 1, limit=limit)
    assert res["leaderboard_comparable"] is comparable
    assert seen["supports_attempts"] is True and seen["extra_payload"]["top_p"] == 0.95
    assert res["accuracy"] == 50.0 and res["scenarios"]["code_execution"]["total"] == 1
    if k > 1:
        assert res["attempts"]["samples"] == 2           # headline @k is code generation only
    if not comparable:
        assert res["leaderboard_comparable_reason"]


def test_lcb_grader_self_check_passes_here_and_blocks_a_broken_grader(monkeypatch):
    from gbench.runners.eval_suites import lcb as L
    assert L._grader_self_check() is None
    monkeypatch.setattr(L, "_run_grader_raw", lambda payload, timeout: (False, "boom"))
    reason = L._grader_self_check()
    assert reason and "known-correct" in reason
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        L.run_lcb("m", "http://x/v1", 1)


# --- lcb: the verdict cannot be forged by the model's program ------------------------------

_LCB_STDIN_GOLD = {"task": "generation", "fn_name": None,
                   "tests": [{"input": "2 3", "output": "5", "testtype": "stdin"}]}


def test_lcb_verdict_cannot_be_forged_by_fd_write_and_exit():
    """A grader that trusted the last stdout line and the program could reach that fd: this
    exact program (write a pass verdict to every fd, then os._exit) scored a PASS."""
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample as V
    forge = ("```python\nimport os\nfor fd in range(3, 40):\n    try:\n"
             "        os.write(fd, b'{\"passed\": true, \"detail\": \"\"}\\n')\n"
             "    except OSError:\n        pass\nos._exit(0)\n```")
    assert V(forge, _LCB_STDIN_GOLD) is False
    # the same attack with the nonce-shaped verdict line and from fds 0-2 as well
    forge_all = ("```python\nimport os\nfor fd in range(0, 64):\n    try:\n"
                 "        os.write(fd, b'\\n{\"nonce\": \"\", \"passed\": true, \"detail\": \"\"}\\n')\n"
                 "    except OSError:\n        pass\nos._exit(0)\n```")
    assert V(forge_all, _LCB_STDIN_GOLD) is False
    assert V("```python\nimport os\nos._exit(0)\n```", _LCB_STDIN_GOLD) is False
    assert V("```python\nprint(6)\n```", _LCB_STDIN_GOLD) is False
    # the grader still passes a correct program in this environment (the asserts above are live)
    assert V("```python\nprint(sum(map(int, input().split())))\n```", _LCB_STDIN_GOLD) is True
    # code execution: the prediction is model text executed inside the grader too
    ex = {"task": "execution", "code": "def f(x):\n    return x * 2", "input": "f(2)", "output": "4"}
    assert V("assert f(2) == __import__('os').write(3, b'{\"passed\": true}\\n') and "
             "__import__('os')._exit(0)\n[/ANSWER]", ex) is False


_LCB_PROC_ATTACK = r'''```python
import os
pp = os.getppid()
nonce = b""
try:
    fd = os.open("/proc/%d/fd/0" % pp, os.O_RDONLY | os.O_NONBLOCK)
    nonce = os.read(fd, 4096).strip()
except OSError:
    pass
line = b'{"nonce": "' + nonce + b'", "passed": true, "detail": "forged"}\n'
for n in range(0, 64):
    try:
        w = os.open("/proc/%d/fd/%d" % (pp, n), os.O_WRONLY | os.O_NONBLOCK)
        os.write(w, line)
    except OSError:
        pass
os._exit(0)
```'''


def test_lcb_verdict_cannot_be_forged_through_proc(monkeypatch):
    """Stealing the nonce from the grader's stdin and writing to its verdict fd through
    /proc/<ppid>/fd is blocked because the grader process is non-dumpable. Positive control:
    the same program DOES forge a pass against a grader with that one line disabled."""
    from gbench.runners.eval_suites import lcb as L
    assert L._verify_single_lcb_sample(_LCB_PROC_ATTACK, _LCB_STDIN_GOLD) is False
    key = "if _libc.prctl(4, 0, 0, 0, 0) != 0:"
    assert key in L._GRADER_MAIN
    monkeypatch.setattr(L, "_GRADER_MAIN", L._GRADER_MAIN.replace(key, "if False:"))
    assert L._verify_single_lcb_sample(_LCB_PROC_ATTACK, _LCB_STDIN_GOLD) is True


def test_lcb_runner_accepts_only_a_verdict_carrying_its_nonce(monkeypatch):
    from gbench.runners.eval_suites import lcb as L
    sent = {}

    def fake_run(argv, input=None, **kw):
        sent["input"] = input
        nonce = input.rsplit("}", 1)[1]
        lines = ['{"passed": true, "detail": ""}',
                 '{"nonce": "not-the-nonce", "passed": true, "detail": ""}']
        if sent.get("with_nonce"):
            lines.append(json.dumps({"nonce": nonce, "passed": True, "detail": ""}))
        return types.SimpleNamespace(stdout="\n".join(lines) + "\n", stderr="")

    monkeypatch.setattr(L, "run_sandboxed", fake_run)
    payload = {"kind": "test_output", "module": "m", "source": "", "prediction": "1",
               "expected": "1"}
    assert L._run_grader_raw(payload, 5)[0] is False
    header, rest = sent["input"].split("\n", 1)
    public = json.loads(rest[:int(header)])                  # length-prefixed public part
    assert "expected" not in public and public["prediction"] == "1"
    header2, rest2 = rest[int(header):].split("\n", 1)
    assert json.loads(rest2[:int(header2)]) == {"expected": "1"}   # then the private part
    assert len(rest2) - int(header2) == 32                   # then a 128-bit hex nonce
    sent["with_nonce"] = True
    assert L._run_grader_raw(payload, 5)[0] is True


def test_lcb_func_name_malformed_metadata_is_an_error():
    from gbench.runners.eval_suites import lcb as L
    with pytest.raises(RuntimeError, match="3704.*not valid JSON"):
        L._func_name("{bad json", "3704")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        L._func_name("[1, 2]", "q1")
    assert L._func_name(None) is None and L._func_name("") is None
    assert L._func_name('{"func_name": "countPartitions"}', "3704") == "countPartitions"


def test_lcb_test_output_prediction_is_graded_on_the_output_key():
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample as V
    assert V("assert f() == 1", {"task": "test_gen", "output": "1"}) is True
    assert V("assert f() == 2", {"task": "test_gen", "output": "1"}) is False
    # the former `expected` key is not read: without `output` nothing is graded and the row fails
    assert V("assert f() == 1", {"task": "test_gen", "expected": "1"}) is False


# --- codeforces: the generated-test cap is recorded and disclosed ---------------------------

@pytest.mark.parametrize("cap,limit", [(0, None), (5, None), (5, 3)])
def test_codeforces_records_and_discloses_the_generated_test_cap(monkeypatch, cap, limit):
    from gbench.runners.eval_suites import codeforces as CF
    monkeypatch.setattr(CF, "_MAX_GENERATED_TESTS", cap)
    monkeypatch.setattr(CF, "_require_generated_tests_mirror", lambda: None)
    monkeypatch.setattr(CF, "_load_codeforces_samples", lambda **kw: [])
    monkeypatch.setattr(CF, "run_eval_suite", lambda **kw: {"accuracy": 0.0, "sample_traces": []})
    res = CF.run_codeforces("m", "http://x/v1", 1, limit=limit)
    assert res["max_generated_tests"] == cap and res["leaderboard_comparable"] is False
    reason = res["leaderboard_comparable_reason"]
    assert ("generated tests capped at 5 per problem" in reason) is (cap > 0)
    assert ("--eval-limit 3" in reason) is bool(limit)
    assert "memory limits not enforced" in reason
