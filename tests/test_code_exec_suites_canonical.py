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
"""Canonical-fidelity locks for the code-execution suites.

* bigcodebench  - upstream chat prompt (instruction_prefix + instruct_prompt), pinned dataset
  revision, LOCAL harness image (never `docker pull`), greedy-only leaderboard_comparable.
* multipl_e     - the pinned revision's chat protocol (DSPy ChatAdapter messages, ChatAdapter.parse,
  postprocess_completion), no gbench repair heuristics, parse failures disclosed.
* ojbench       - per-language Pass@1/Pass@8 over the official 8 candidates, C++ Pass@1 headline,
  verdicts keyed per generation.
* scicode       - a crashed problem is a harness error, left out; headline = main-problem resolve
  rate.
* aider_polyglot - --thinking reaches aider's requests through aider/extra_params model settings.
"""

import asyncio
import json
import os
import types

import pytest

import gbench.runners.eval_suites.aider_polyglot as A
import gbench.runners.eval_suites.bigcodebench as B
import gbench.runners.eval_suites.multipl_e as M
import gbench.runners.eval_suites.ojbench as OJ
import gbench.runners.eval_suites.scicode as SC
from gbench.runners.eval_suites import tau_common as TC_GUARD

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _proc(rc=0, out="", err=""):
    return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)


# --------------------------------------------------------------------------- #
# bigcodebench                                                                 #
# --------------------------------------------------------------------------- #
def test_bcb_prompt_is_upstream_make_raw_chat_prompt():
    # bigcodebench/generate.py default + provider/utility.py make_raw_chat_prompt (split=instruct,
    # tokenizer=None): f"{instruction_prefix}\n{task_prompt.strip()}\n"
    assert B.INSTRUCTION_PREFIX == ("Please provide a self-contained Python script that solves "
                                    "the following problem in a markdown code block:")
    task = "  Do X.\n\nYou should write self-contained code starting with:\n```\ndef task_func():\n```\n\n"
    assert B.make_raw_chat_prompt(task) == B.INSTRUCTION_PREFIX + "\n" + task.strip() + "\n"


def test_bcb_loader_pins_revision_and_prepends_prefix(monkeypatch):
    seen = {}

    def fake_load(name, split=None, revision=None):
        seen.update(name=name, split=split, revision=revision)
        return [{"task_id": "BigCodeBench/0", "instruct_prompt": "Do X.\n",
                 "complete_prompt": "def task_func():\n", "entry_point": "task_func"}]

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    samples = B._load_bigcodebench_samples(subset="hard")
    assert seen == {"name": "bigcode/bigcodebench-hard", "split": "v0.1.4",
                    "revision": B._REVISIONS["hard"]}
    assert samples[0][0] == [{"role": "user", "content": B.INSTRUCTION_PREFIX + "\nDo X.\n"}]
    # both leaderboard columns: the Complete prompt is fenced (make_raw_chat_prompt split=complete)
    assert samples[1][0] == [{"role": "user", "content": B.INSTRUCTION_PREFIX
                              + "\n```\ndef task_func():\n```\n"}]
    assert [m["split"] for _, _, m in samples] == ["instruct", "complete"]


def test_bcb_never_pulls_and_names_the_local_build(monkeypatch):
    cmds = []

    def fake_run(cmd, **k):
        cmds.append(cmd)
        return _proc(1 if cmd[:3] == ["docker", "image", "inspect"] else 0)

    monkeypatch.setattr(B.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_IMAGE", raising=False)
    ok, reason = B.check_bigcodebench_prerequisites()
    assert not ok
    assert "docker build -t gbench-bigcodebench:09dd993f -f docker/bigcodebench.Dockerfile docker" in reason
    assert not any("pull" in c for c in cmds)
    src = open(B.__file__, encoding="utf-8").read()
    assert '"pull"' not in src and ":latest" not in src


def test_bcb_dockerfile_is_pinned():
    df = open(os.path.join(REPO, "docker", "bigcodebench.Dockerfile"), encoding="utf-8").read()
    assert "09dd993f46c3fbf3a799465bb96d524edcb0b199" in df
    code = "\n".join(l for l in df.splitlines() if not l.lstrip().startswith("#"))
    assert "refs/heads/main" not in code and ":latest" not in code and "commits?per_page" not in code
    assert "FROM python:3.10.16-slim-bookworm" in df
    assert "/bigcodebench/Requirements/requirements-eval.txt" in df


def test_bcb_unknown_subset_is_an_error(monkeypatch):
    monkeypatch.setenv("GBENCH_BIGCODEBENCH_SUBSET", "tiny")
    with pytest.raises(ValueError, match="unknown subset"):
        B.run_bigcodebench("m", "http://h/v1", 1)


@pytest.mark.parametrize("temp,limit,comparable", [(0.0, None, True), (1.0, None, False),
                                                   (0.0, 20, False)])
def test_bcb_leaderboard_comparable_is_greedy_full_split(monkeypatch, temp, limit, comparable):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])
    monkeypatch.setattr(B, "run_eval_suite", lambda **k: {"accuracy": 50.0, "temperature": temp,
                                                          "status": "success",
                                                          "max_tokens_sent": [1280]})
    r = B.run_bigcodebench("m", "http://h/v1", 1, limit=limit)
    assert r["leaderboard_comparable"] is comparable
    assert r["subset"] == "full" and B._REVISIONS["full"] in r["dataset_revision"]
    if not comparable:
        reason = r["leaderboard_comparable_reason"]
        assert ("temperature=1.0" in reason) if temp else ("--eval-limit 20" in reason)


# --------------------------------------------------------------------------- #
# multipl_e                                                                    #
# --------------------------------------------------------------------------- #
def test_mpe_chat_messages_are_the_chat_adapter_messages():
    msgs = M.chat_messages("jl", "function add(x::Int64, y::Int64)::Int64\n")
    assert [m["role"] for m in msgs] == ["system", "user"]
    sysm = msgs[0]["content"]
    assert sysm.startswith("Your input fields are:\n1. `lang` (str): The programming language of "
                           "the program to complete\n")
    assert "1. `reasoning` (str): \n2. `completion` (str): The complete program including the full prefix\n" in sysm
    assert "[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        I will give you" in sysm
    assert sysm.endswith("\n        \n        You MUST return a full program with the prefix verbatim "
                         "and the completion.")
    assert msgs[1]["content"] == (
        "[[ ## lang ## ]]\njl\n\n[[ ## prefix ## ]]\nfunction add(x::Int64, y::Int64)::Int64\n\n\n"
        "Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, "
        "then `[[ ## completion ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.")


def test_mpe_parse_matches_chat_adapter_rules():
    ok = "[[ ## reasoning ## ]]\nr\n\n[[ ## completion ## ]]\n```julia\nx\n```\n\n[[ ## completed ## ]]"
    assert M.parse_chat_fields(ok) == {"reasoning": "r", "completion": "```julia\nx\n```"}
    assert M.parse_chat_fields("[[ ## completion ## ]]\nonly one field") is None
    assert M.parse_chat_fields("plain prose") is None
    # first occurrence wins; inline content after the header is kept
    two = "[[ ## reasoning ## ]] a\n[[ ## completion ## ]] b\n[[ ## completion ## ]]\nc"
    assert M.parse_chat_fields(two) == {"reasoning": "a", "completion": "b"}


def test_mpe_postprocess_is_upstream():
    assert M.postprocess_completion(["\nend"], "function f(x)\n",
                                    "```julia\nfunction f(x)\n  x\nend\n```") == "function f(x)\n  x"
    # the stop token only applies after the prompt-length head
    assert M.postprocess_completion(["\ndef"], "def f():\n", "def f():\n  return 1\ndef g(): pass") \
        == "def f():\n  return 1"


def test_mpe_scorer_writes_prompt_empty_and_counts_parse_failures(monkeypatch):
    written = []

    def fake_run(cmd, **k):
        out = next(a.split(":", 1)[0] for a in cmd if a.endswith(":/out:rw"))
        for name in sorted(os.listdir(out)):
            if name.endswith(".json"):
                data = json.load(open(os.path.join(out, name)))
                written.append(data)
                res = {"name": data["name"], "results": [{"status": "OK", "exit_code": 0}]}
                json.dump(res, open(os.path.join(out, name.replace(".json", ".results.json")), "w"))
        return _proc()

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    gold = json.dumps({"name": "HumanEval_0", "language": "jl", "prompt": "function f(x)\n",
                       "tests": "@test f(1) == 1", "stop_tokens": ["\nend"]})
    traces = [
        {"gold_answer": gold, "response_text":
            "[[ ## reasoning ## ]]\nr\n[[ ## completion ## ]]\n```julia\nfunction f(x)\n  x\nend\n```"},
        {"gold_answer": gold, "response_text": "function f(x) x end"},   # no ChatAdapter fields
    ]
    metrics = {}
    asyncio.run(M._make_scorer("img", 2, metrics)(traces))
    assert [w["prompt"] for w in written] == ["", ""]
    assert written[0]["completions"] == ["function f(x)\n  x"]
    assert written[1]["completions"] == [""]
    assert metrics["chat_adapter_parse_failures"] == 1
    assert "completion" in traces[1]["scoring_note"]


def test_mpe_no_repair_heuristics_left():
    for name in ("_assemble_code", "_balance_against_tests", "_find_decl_ws_insensitive"):
        assert not hasattr(M, name)


def test_mpe_samples_knob(monkeypatch):
    monkeypatch.delenv("GBENCH_MULTIPL_E_SAMPLES", raising=False)
    # the upstream chat recipe's 20 by default; a larger --attempt-count is honoured
    assert M._samples_per_problem(None) == 20 and M._samples_per_problem(1) == 20
    assert M._samples_per_problem(25) == 25
    monkeypatch.setenv("GBENCH_MULTIPL_E_SAMPLES", "5")
    assert M._samples_per_problem(20) == 5
    monkeypatch.setenv("GBENCH_MULTIPL_E_SAMPLES", "many")
    with pytest.raises(ValueError):
        M._samples_per_problem(1)


def test_mpe_loader_pins_dataset_revision(monkeypatch):
    seen = []

    def fake_load(name, cfg, split=None, revision=None):
        seen.append((name, cfg, split, revision))
        return [{"name": "HumanEval_0", "prompt": "p", "tests": "t", "stop_tokens": []}]

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    samples = M._load_multipl_e_samples(langs=["jl"])
    assert seen == [("nuprl/MultiPL-E", "humaneval-jl", "test", M._DATASET_REVISION)]
    assert samples[0][0] == M.chat_messages("jl", "p")


def test_mpe_parse_failures_are_scored_as_upstream_and_stay_comparable(monkeypatch):
    # DSPy 3.0.0b1's async path has no JSONAdapter fallback: upstream records completion "".
    monkeypatch.setenv("GBENCH_MULTIPL_E_SAMPLES", "20")
    monkeypatch.delenv("GBENCH_MULTIPL_E_LANGS", raising=False)
    monkeypatch.setattr(M, "check_multipl_e_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_load_multipl_e_samples", lambda langs=None, limit=None: [])

    def fake_suite(**k):
        k["async_eval_fn"]  # the scorer is built with the shared metrics dict
        return {"accuracy": 40.0, "temperature": 0.2, "status": "success",
                "max_tokens_sent": [M.UPSTREAM_MAX_TOKENS]}

    def fake_scorer(image, workers, metrics):
        metrics["chat_adapter_parse_failures"] = 3
        return None

    monkeypatch.setattr(M, "_make_scorer", fake_scorer)
    monkeypatch.setattr(M, "run_eval_suite", fake_suite)
    r = M.run_multipl_e("m", "http://h/v1", 1)
    assert r["leaderboard_comparable"] is True
    assert r["chat_adapter_parse_failures"] == 3
    assert r["samples_per_problem"] == 20


# --------------------------------------------------------------------------- #
# ojbench                                                                      #
# --------------------------------------------------------------------------- #
def _oj_trace(pid, lang, diff="easy", resp="code"):
    return {"extra_payload": {"row": {"id": pid, "language": lang, "difficulty": diff}},
            "response_text": resp}


def test_oj_verdicts_are_keyed_per_generation(monkeypatch):
    # Three attempts of the same (id, language): each must get ITS OWN verdict.
    def fake_judge(records, n):
        return [{**r, "is_passed": r["gbench_uid"] == 1} for r in records]

    monkeypatch.setattr(OJ, "_judge_in_container", fake_judge)
    traces = [_oj_trace(7, "cpp") for _ in range(3)]
    asyncio.run(OJ._make_scorer(2)(traces))
    assert [t["is_correct"] for t in traces] == [False, True, False]


def test_oj_ambiguous_verdict_without_uid_is_not_copied(monkeypatch):
    monkeypatch.setattr(OJ, "_judge_in_container",
                        lambda records, n: [{"id": 7, "language": "cpp", "is_passed": True}])
    traces = [_oj_trace(7, "cpp"), _oj_trace(7, "cpp")]
    asyncio.run(OJ._make_scorer(2)(traces))
    assert all(t.get("scoring_excluded") for t in traces)


def test_oj_per_language_report_matches_paper_columns():
    traces = []
    # cpp: problem 1 -> 2/8 pass, problem 2 -> 0/8; python: problem 1 -> 8/8, problem 2 -> 0/8
    for pid, lang, passes, diff in ((1, "cpp", 2, "easy"), (2, "cpp", 0, "hard"),
                                    (1, "python", 8, "easy"), (2, "python", 0, "hard")):
        for a in range(8):
            t = _oj_trace(pid, lang, diff)
            t["is_correct"] = a < passes
            traces.append(t)
    rep = OJ._per_language_report(traces, 8)
    assert rep["cpp"]["pass_at_1"] == 12.5 and rep["cpp"]["pass_at_8"] == 50.0
    assert rep["python"]["pass_at_1"] == 50.0 and rep["python"]["pass_at_8"] == 50.0
    assert rep["cpp"]["pass_rate_by_difficulty"] == {"easy": 25.0, "hard": 0.0}
    assert rep["cpp"]["problems"] == 2 and rep["cpp"]["generations"] == 16


def test_oj_samples_default_is_official_eight(monkeypatch):
    monkeypatch.delenv("GBENCH_OJBENCH_SAMPLES", raising=False)
    assert OJ._samples_per_problem(1) == 8 and OJ._samples_per_problem(16) == 16
    monkeypatch.setenv("GBENCH_OJBENCH_SAMPLES", "0")
    with pytest.raises(ValueError):
        OJ._samples_per_problem(1)


def test_oj_headline_is_cpp_pass_at_1(monkeypatch):
    monkeypatch.delenv("GBENCH_OJBENCH_SAMPLES", raising=False)
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",   # a Gemma 4 model is served
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(OJ, "_load_ojbench_samples", lambda limit=None: [])
    seen = {}

    def fake_suite(**k):
        seen.update(k)
        traces = []
        for lang, ok in (("cpp", 3), ("python", 6)):
            for a in range(8):
                t = _oj_trace(1, lang)
                t["is_correct"] = a < ok
                traces.append(t)
        return {"accuracy": 56.25, "temperature": 1.0, "status": "success", "scoring_excluded": 0,
                "max_tokens_sent": [65536], "sample_traces": traces}

    monkeypatch.setattr(OJ, "run_eval_suite", fake_suite)
    r = OJ.run_ojbench("m", "http://h/v1", 1, enable_thinking=True)
    assert seen["attempt_count"] == 8 and seen["supports_attempts"] is True
    assert r["accuracy"] == 37.5 and r["correct_answers"] == 3 and r["total_questions"] == 8
    assert r["per_language"]["python"]["pass_at_1"] == 75.0
    assert r["pooled_pass_at_1"] == 56.25
    assert r["leaderboard_comparable"] is True
    r2 = OJ.run_ojbench("m", "http://h/v1", 1, enable_thinking=True, limit=10)
    assert r2["leaderboard_comparable"] is False and "--eval-limit 10" in r2["leaderboard_comparable_reason"]


def test_oj_loader_pins_revision_and_categorises_by_language(monkeypatch, tmp_path):
    p = tmp_path / "full.jsonl"
    p.write_text(json.dumps({"id": 1, "prompt": "solve", "language": "python",
                             "dataset": "NOI", "difficulty": "easy"}) + "\n")
    seen = {}

    def fake_dl(**k):
        seen.update(k)
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    s = OJ._load_ojbench_samples()
    assert seen["revision"] == OJ._DATASET_REVISION
    assert s[0][2]["category"] == "python"


# --------------------------------------------------------------------------- #
# scicode                                                                      #
# --------------------------------------------------------------------------- #
def _sc_problems(sizes):
    return [{"problem_id": str(i + 1), "problem_name": f"p{i + 1}", "required_dependencies": "",
             "sub_steps": [{"step_number": f"{i + 1}.{k + 1}", "test_cases": ["assert True"]}
                           for k in range(n)]}
            for i, n in enumerate(sizes)]


def _sc_run(monkeypatch, problems, crash_ids=(), limit=None, digest=None):
    monkeypatch.setattr(SC, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(SC, "resolve_test_data", lambda: ("/nonexistent.h5", "test"))
    # the reference file's digest is checked; default to the pinned one.
    monkeypatch.setattr(SC, "_digest", lambda path: digest or SC._TEST_DATA_SHA256)
    monkeypatch.setattr(SC, "_load_scicode_problems", lambda limit=None: problems)
    monkeypatch.setenv("GBENCH_SCICODE_TEMPERATURE", "0")
    monkeypatch.delenv("GBENCH_SCICODE_WITH_BACKGROUND", raising=False)

    def fake_one(client, model, prob, **k):
        if prob["problem_id"] in crash_ids:
            raise RuntimeError("boom")
        return {"problem_id": prob["problem_id"], "problem_name": prob["problem_name"],
                "steps": [{"step_number": s["step_number"], "special": False, "gen_failed": False,
                           "correct": True, "response_text": "x", "error": None}
                          for s in prob["sub_steps"]]}

    monkeypatch.setattr(SC, "_process_one_problem", fake_one)
    return SC.run_scicode("m", "http://h/v1", limit=limit, concurrency=2)


def test_sc_crashed_problem_is_a_harness_error_left_out_of_the_score(monkeypatch):
    # A problem whose processing crashed has no verdict: a harness error, left out of both
    # accuracies (never scored as failed), counted and named.
    r = _sc_run(monkeypatch, _sc_problems([2, 3, 4]), crash_ids=("2",))
    assert r["total_problems"] == 2 and r["correct_problems"] == 2
    assert r["total_steps"] == 6 and r["correct_steps"] == 6
    assert r["accuracy"] == r["problem_accuracy"] == 100.0             # headline = resolve rate
    assert r["substep_accuracy"] == 100.0
    assert r["total_questions"] == 3 and r["effective_n"] == 2 and r["correct_answers"] == 2
    assert r["errored_problems"] == 1 and r["harness_error_problems"] == 1
    assert r["status"] == "completed_with_errors" and "harness errors left out" in r["status_reason"]
    crashed = [t for t in r["sample_traces"] if t["problem_id"] == "2"]
    assert len(crashed) == 3 and all(t["error"].startswith("problem_crashed") for t in crashed)
    assert all(t["scoring_excluded"] and t["harness_error"]["kind"] == "problem_crashed"
               for t in crashed)
    assert r["scoring_excluded"] == 3
    assert r["leaderboard_comparable"] is False
    assert "1 problem(s) crashed" in r["leaderboard_comparable_reason"]


def test_sc_full_split_greedy_run_is_comparable(monkeypatch):
    # 65 problems / 288 scored sub-steps, like upstream's fixed test-split denominators.
    r = _sc_run(monkeypatch, _sc_problems([4] * 64 + [32]))
    assert (r["total_problems"], r["total_steps"]) == (65, 288)
    assert r["leaderboard_comparable"] is True and r["accuracy"] == 100.0


def test_sc_problem_without_sub_steps_is_an_error(monkeypatch):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda *a, **k: [{"problem_id": "9", "sub_steps": []}])
    with pytest.raises(RuntimeError, match="no sub_steps"):
        SC._load_scicode_problems()


# --------------------------------------------------------------------------- #
# aider_polyglot                                                               #
# --------------------------------------------------------------------------- #
def test_aider_settings_file_carries_enable_thinking(tmp_path):
    extra = A._write_model_settings(str(tmp_path), True)
    data = json.load(open(tmp_path / A._SETTINGS_FILE))
    assert data == [{"name": "aider/extra_params", "extra_params": extra}]
    assert extra == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                     "timeout": A.model_request_timeout_s(), "max_retries": 0}
    assert oct(os.stat(tmp_path / A._SETTINGS_FILE).st_mode & 0o777) == "0o644"


def test_aider_command_mounts_and_reads_the_settings():
    cmd = A._build_command("img", "/b", "gbench", "openai/m", "http://h/v1", 1, A._LANGUAGES,
                           "", None, "", settings_dir="/tmp/s")
    joined = " ".join(cmd)
    assert "/tmp/s:/gbench-aider:ro" in cmd
    assert "--read-model-settings /gbench-aider/model-settings.yml" in joined


@pytest.mark.parametrize("thinking", [True, False])
def test_aider_thinking_reaches_the_container(monkeypatch, tmp_path, thinking):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    monkeypatch.delenv("GBENCH_AIDER_EXTRA_ARGS", raising=False)
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    seen = {}

    def fake_run(cmd, **k):
        mount = next(a for a in cmd if a.endswith(":/gbench-aider:ro"))
        host = mount.split(":", 1)[0]
        seen["settings"] = json.load(open(os.path.join(host, A._SETTINGS_FILE)))
        d = tmp_path / "2026-run--gbench" / "python" / "exercises" / "practice" / "x0"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".aider.results.json").write_text(json.dumps({"language": "python",
                                                           "tests_outcomes": [True]}))
        # the in-container pre-flight reports what aider loaded (echoed from the file).
        return types.SimpleNamespace(returncode=0, stderr="", stdout=A._CHECK_MARKER + json.dumps(
            {"files_loaded": ["/gbench-aider/model-settings.yml"],
             "extra_params": seen["settings"][0]["extra_params"]}) + "\n")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    (tmp_path / "polyglot-benchmark" / "python" / "exercises" / "practice" / "x0").mkdir(
        parents=True, exist_ok=True)   # the exercises a limited run selects from
    r = A.run_aider_polyglot("m", "http://h/v1", concurrency=1, enable_thinking=thinking, limit=1)
    ctk = seen["settings"][0]["extra_params"]["extra_body"]["chat_template_kwargs"]
    assert ctk == {"enable_thinking": thinking}
    assert r["thinking"] is thinking
    assert r["model_extra_params"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is thinking
    assert r["leaderboard_comparable"] is False and "--eval-limit 1" in r["leaderboard_comparable_reason"]


def test_aider_refuses_a_competing_settings_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    monkeypatch.setenv("GBENCH_AIDER_EXTRA_ARGS", "--read-model-settings /x.yml")
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    with pytest.raises(ValueError, match="read-model-settings"):
        A.run_aider_polyglot("m", "http://h/v1", enable_thinking=True)


# --------------------------------------------------------------------------- #
# Code-execution suites: comparability, disclosure and pre-flight checks      #
# --------------------------------------------------------------------------- #
def _bcb_run(monkeypatch, suite_result, metrics_update=None, limit=None):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])

    def fake_make_scorer(subset, parallel, metrics, image):
        metrics.update(metrics_update or {})
        return None

    monkeypatch.setattr(B, "_make_scorer", fake_make_scorer)
    monkeypatch.setattr(B, "run_eval_suite", lambda **k: dict(suite_result))
    return B.run_bigcodebench("m", "http://h/v1", 1, limit=limit)


def test_bcb_discloses_the_generation_path_it_reproduces(monkeypatch):
    # no-think runs take upstream's default vllm path (prefill), as every Gemma leaderboard row
    r = _bcb_run(monkeypatch, {"accuracy": 50.0, "temperature": 0.0, "status": "success",
                               "max_tokens_sent": [B.UPSTREAM_MAX_NEW_TOKENS]})
    assert r["leaderboard_comparable"] is True and r["prefill"] is True
    assert "--backend vllm" in r["generation_path"] and "prefill" in r["generation_path"]
    note = r["leaderboard_comparable_note"]
    assert "prefill flag is true" in note


def test_bcb_harness_error_is_never_comparable(monkeypatch):
    r = _bcb_run(monkeypatch, {"accuracy": 0.0, "temperature": 0.0, "status": "success"},
                 {"bigcodebench_report": {"error": "harness produced no parsed results"}})
    assert r["status"] == "error" and r["leaderboard_comparable"] is False
    assert "harness error: harness produced no parsed results" in r["leaderboard_comparable_reason"]
    assert "leaderboard_comparable_note" not in r


def test_bcb_tasks_without_a_harness_result_are_counted_and_gate_comparability(monkeypatch):
    def fake_run(cmd, **k):
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        if "bigcodebench.sanitize" not in cmd:
            json.dump({"eval": {"BigCodeBench/0": [{"status": "pass", "details": {}}]}},
                      open(os.path.join(mount, "samples_eval_results.json"), "w"))
        return _proc()

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    traces = [{"extra_payload": {"task_id": f"BigCodeBench/{i}"}, "response_text": "x"}
              for i in range(3)]
    metrics = {}
    asyncio.run(B._make_scorer("full", 1, metrics, "img")(traces))
    assert [t["is_correct"] for t in traces] == [True, False, False]
    assert metrics["bigcodebench_unscored"] == 2
    assert traces[1]["scoring_note"] == "no harness result for this task"
    r = _bcb_run(monkeypatch, {"accuracy": 33.3, "temperature": 0.0, "status": "success"},
                 {"bigcodebench_unscored": 2})
    assert r["leaderboard_comparable"] is False
    assert "2 task(s) got no harness result" in r["leaderboard_comparable_reason"]


def test_mpe_empty_responses_and_parse_failures_are_counted_apart(monkeypatch):
    def fake_run(cmd, **k):
        out = next(a.split(":", 1)[0] for a in cmd if a.endswith(":/out:rw"))
        for name in sorted(os.listdir(out)):
            if name.endswith(".json"):
                data = json.load(open(os.path.join(out, name)))
                json.dump({"name": data["name"], "results": [{"status": "OK", "exit_code": 0}]},
                          open(os.path.join(out, name.replace(".json", ".results.json")), "w"))
        return _proc()

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    gold = json.dumps({"name": "HumanEval_0", "language": "jl", "prompt": "function f(x)\n",
                       "tests": "@test f(1) == 1", "stop_tokens": ["\nend"]})
    traces = [{"gold_answer": gold, "response_text": ""},
              {"gold_answer": gold, "response_text": "no fields here"},
              {"gold_answer": gold, "response_text": None}]
    metrics = {}
    asyncio.run(M._make_scorer("img", 1, metrics)(traces))
    assert metrics["chat_adapter_parse_failures"] == 1
    assert metrics["chat_adapter_empty_responses"] == 1
    assert metrics["multipl_e_no_response"] == 1
    assert traces[0]["scoring_note"].startswith("empty response")
    assert all(t["assembled_program"] == "" for t in traces)


def _mpe_run(monkeypatch, suite_result, metrics_update):
    monkeypatch.delenv("GBENCH_MULTIPL_E_SAMPLES", raising=False)
    monkeypatch.delenv("GBENCH_MULTIPL_E_LANGS", raising=False)
    monkeypatch.setattr(M, "check_multipl_e_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_load_multipl_e_samples", lambda **k: [])

    def fake_make_scorer(image, workers, metrics):
        metrics.update(metrics_update)
        return None

    monkeypatch.setattr(M, "_make_scorer", fake_make_scorer)
    monkeypatch.setattr(M, "run_eval_suite", lambda **k: dict(suite_result))
    return M.run_multipl_e("m", "http://h/v1", 1, attempt_count=20)


def test_mpe_comparability_counts_empty_replies_errors_and_unscored(monkeypatch):
    ok = {"accuracy": 50.0, "temperature": 0.2, "status": "success",
          "max_tokens_sent": [M.UPSTREAM_MAX_TOKENS]}
    assert _mpe_run(monkeypatch, ok, {"chat_adapter_parse_failures": 0})["leaderboard_comparable"] is True
    r = _mpe_run(monkeypatch, ok, {"chat_adapter_parse_failures": 3,
                                   "chat_adapter_empty_responses": 2})
    assert r["leaderboard_comparable"] is False
    assert r["leaderboard_comparable_reason"].startswith("2 empty response(s) scored as empty programs")
    r = _mpe_run(monkeypatch, ok, {"multipl_e_report": {"error": "no parsed results"}})
    assert r["status"] == "error" and "harness error: no parsed results" in r["leaderboard_comparable_reason"]
    r = _mpe_run(monkeypatch, ok, {"multipl_e_unscored": 4})
    assert "4 completion(s) got no evaluator result" in r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("sent,comparable,needle", [
    (None, True, None),                      # no max_tokens: the model's default context length
    ([7692], False, "none is sent"),
])
def test_oj_no_think_budget_is_the_models_default_context_length(monkeypatch, sent, comparable,
                                                                 needle):
    # Paper section 3.1: non-reasoning models get max_tokens = the model's default context length,
    # so a no-think request sends none (base's token-limit policy, send_max_tokens=False).
    monkeypatch.delenv("GBENCH_OJBENCH_SAMPLES", raising=False)
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",   # a Gemma 4 model is served
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(OJ, "_load_ojbench_samples", lambda limit=None: [])
    seen = {}

    def fake_suite(**k):
        seen.update(k)
        traces = []
        for lang in ("cpp", "python"):
            for a in range(8):
                t = _oj_trace(1, lang)
                t["is_correct"] = a < 4
                traces.append(t)
        # 1.0: Gemma 4's recommended temperature, which OJBench's protocol samples at
        return {"accuracy": 50.0, "temperature": 1.0, "status": "success", "scoring_excluded": 0,
                "max_tokens_sent": sent, "sample_traces": traces}

    monkeypatch.setattr(OJ, "run_eval_suite", fake_suite)
    r = OJ.run_ojbench("m", "http://h/v1", 1, enable_thinking=False)
    assert seen["official_sampling"].send_max_tokens is False
    assert r["leaderboard_comparable"] is comparable
    if needle:
        assert needle in r["leaderboard_comparable_reason"]


def test_sc_default_mirror_is_read_at_the_pinned_revision(monkeypatch, tmp_path):
    f = tmp_path / "test_data.h5"
    f.write_bytes(b"x")
    seen = {}
    monkeypatch.delenv("GBENCH_SCICODE_TEST_DATA", raising=False)
    monkeypatch.delenv("SCICODE_TEST_DATA", raising=False)
    monkeypatch.setattr(SC, "_TEST_DATA_REPO", SC._TEST_DATA_DEFAULT_REPO)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: seen.update(kw) or str(f))
    path, prov = SC.resolve_test_data()
    assert seen["revision"] == SC._TEST_DATA_REVISION == "72c247d3a8410921b2e848e046d71ed63d9a0ddb"
    assert prov == f"hf:{SC._TEST_DATA_DEFAULT_REPO}@{SC._TEST_DATA_REVISION}/test_data.h5"
    assert SC._TEST_DATA_SHA256 == "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"


def test_sc_digest_mismatch_on_the_default_mirror_is_a_hard_error(monkeypatch):
    monkeypatch.setattr(SC, "_digest", lambda path: "f" * 64)
    prov = f"hf:{SC._TEST_DATA_DEFAULT_REPO}@{SC._TEST_DATA_REVISION}/test_data.h5"
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*expected " + SC._TEST_DATA_SHA256):
        SC.check_test_data_digest("/cache/test_data.h5", prov)
    monkeypatch.setattr(SC, "_digest", lambda path: SC._TEST_DATA_SHA256)
    assert SC.check_test_data_digest("/cache/test_data.h5", prov) == (SC._TEST_DATA_SHA256, None)


def test_sc_user_file_with_another_digest_is_not_comparable(monkeypatch):
    monkeypatch.setattr(SC, "_digest", lambda path: "a" * 64)
    digest, reason = SC.check_test_data_digest("/mine/test_data.h5", "GBENCH_SCICODE_TEST_DATA")
    assert digest == "a" * 64
    assert reason.startswith("unverified reference file") and "/mine/test_data.h5" in reason
    r = _sc_run(monkeypatch, _sc_problems([4] * 64 + [32]), digest="a" * 64)
    assert r["leaderboard_comparable"] is False
    assert "unverified reference file" in r["leaderboard_comparable_reason"]
    assert r["test_data"]["verified"] is False and r["test_data"]["sha256"] == "a" * 64
    ok = _sc_run(monkeypatch, _sc_problems([4] * 64 + [32]))
    assert ok["leaderboard_comparable"] is True and ok["test_data"]["verified"] is True


def test_sc_docs_state_the_pin_and_the_digest():
    doc = open(os.path.join(REPO, "docs", "evals", "scicode.md"), encoding="utf-8").read()
    assert SC._TEST_DATA_REVISION in doc and SC._TEST_DATA_SHA256 in doc


def test_aider_preflight_runs_before_benchmark_and_writes_its_script(tmp_path):
    A._write_model_settings(str(tmp_path), False)
    assert (tmp_path / A._CHECK_FILE).read_text() == A._CHECK_SRC
    assert "models.register_models" in A._CHECK_SRC and "extra_params" in A._CHECK_SRC
    cmd = A._build_command("img", "/b", "gbench", "openai/m", "http://h/v1", 1, A._LANGUAGES,
                           "", None, "", settings_dir=str(tmp_path))
    script = cmd[-1]
    pre = "python3 /gbench-aider/check_settings.py /gbench-aider/model-settings.yml openai/m && "
    assert pre in script and script.index(pre) < script.index("./benchmark/benchmark.py")


@pytest.mark.parametrize("stdout", [
    "",                                                                   # no report at all
    A._CHECK_MARKER + json.dumps({"files_loaded": [], "extra_params": None}) + "\n",
    A._CHECK_MARKER + json.dumps({"files_loaded": ["/gbench-aider/model-settings.yml"],
                                  "extra_params": {"extra_body": {"chat_template_kwargs": {
                                      "enable_thinking": False}}}}) + "\n",   # wrong value
])
def test_aider_unconfirmed_settings_are_a_hard_error(monkeypatch, tmp_path, stdout):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    monkeypatch.delenv("GBENCH_AIDER_EXTRA_ARGS", raising=False)
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))

    def fake_run(cmd, **k):
        d = tmp_path / "2026-run--gbench" / "python" / "exercises" / "practice" / "x0"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".aider.results.json").write_text(json.dumps({"language": "python",
                                                           "tests_outcomes": [True]}))
        return _proc(out=stdout)

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    (tmp_path / "polyglot-benchmark" / "python" / "exercises" / "practice" / "x0").mkdir(
        parents=True, exist_ok=True)   # the exercises a limited run selects from
    with pytest.raises(RuntimeError, match="did not confirm that it loaded"):
        A.run_aider_polyglot("m", "http://h/v1", concurrency=1, enable_thinking=True, limit=1)
