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
"""Code-execution suites follow their upstream extraction, judging and estimators.

* codeforces: open-r1's extract_code (last ```python block, none = no submission) and the piston
  codeforces `run` script's judging (yes/no folding, \\r stripped from the input, empty lines,
  checker verdict "1"/"100" with exit 0, file input mode, unlimited stack).
* bigcodebench: text the server's reasoning parser moved out of a no-think reply is scored (base's
  server-parser policy) and named; a task the harness returned no result for is a harness error.
* multipl_e: a completion the container returned no result for is a harness error.
* scicode: the test cases' `scicode.compare.cmp` import resolves (vendored upstream module).
* spider2: the prediction round-trips through CSV and is compared with upstream's
  compare_multi_pandas_table; the checkout's revision is recorded and checked.
* omnidocbench: --eval-limit draws a stratified subset, not a contiguous head.
* lcb: pass@1 is the per-problem mean of c_i/n_i over the scored generations.
Nothing here reaches a network, docker or a model server.
"""

import os
import re
import resource
import sys

import pytest

from gbench.runners.eval_suites import codeforces as CF

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gold(tests, checker="", input_mode="stdio", time_limit=5):
    return {"official_tests": tests, "generated_tests": [], "checker": checker,
            "time_limit": time_limit, "input_mode": input_mode}


# --------------------------------------------------------------------------- #
# codeforces: extraction
# --------------------------------------------------------------------------- #
def test_extract_code_is_open_r1s_verbatim():
    assert CF.extract_code("no fence at all") == ""
    assert CF.extract_code("```py\nprint(1)\n```") == ""            # the language must be python
    assert CF.extract_code("```\nprint(1)\n```") == ""
    assert CF.extract_code("```python\nA\n```\ntext\n```python\nB\n```") == "B\n"
    assert CF.extract_code("```python\nA\n```", None) == ""
    # the closing fence is the next ``` wherever it stands
    assert CF.extract_code("```python\nx = 1\n    ```\nmore") == "x = 1\n    "


def test_a_reply_without_a_python_block_is_no_submission():
    """The whole reply is never run as code: open-r1 scores '' as no submission."""
    gold = _gold([{"input": "", "output": "2"}])
    assert CF._eval_codeforces("print(2)", gold) is False
    assert CF._eval_codeforces("```python\nprint(2)\n```", gold) is True


def test_an_indented_closing_fence_does_not_shift_the_final_program():
    """The live failure: a draft block closed by an indented fence made the old regex take the
    next column-0 opener as its close, so every later pair shifted by one and the graded 'last
    block' was prose. open-r1's pattern still finds the final program."""
    final = "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n"
    reply = ("Plan first.\n```python\ndef draft(n):\n    return n\n    ```\n"
             "That draft is wrong.\n```python\n" + final + "```\n"
             "Final check on the `pair_info` index: it is right.\n```\nnote\n```\n"
             "Done.")
    old = re.findall(r"```(?:python)?\s*\n(.*?)\n```", reply, re.DOTALL)
    assert old[-1] != final.rstrip("\n")                             # what used to be graded
    assert CF.extract_code(reply) == final
    assert CF._eval_codeforces(reply, _gold([{"input": "21\n", "output": "42"}])) is True


# --------------------------------------------------------------------------- #
# codeforces: the piston run script's judging
# --------------------------------------------------------------------------- #
def test_the_diff_folds_whole_word_yes_no_and_normalizes_whitespace():
    assert CF._outputs_match("Yes\nno", "YES\nNO")
    assert CF._outputs_match("yes-no", "YES-NO")
    assert not CF._outputs_match("yesno", "YESNO")                   # whole words only
    assert CF._outputs_match(" a   b \n\n", "a b")
    assert not CF._outputs_match("a b", "ab")
    assert CF._outputs_match("1\n9\n0", "1\r\n9\r\n0  ")
    assert CF._outputs_match("\u00e9no", "\u00e9NO")                   # C locale: ASCII \b
    assert CF._outputs_match("", "")


def test_the_input_loses_carriage_returns_at_line_ends():
    code = "import sys\nprint(len(sys.stdin.buffer.read()))\n"
    assert CF._run_case(code, {"input": "1 2\r\n3\r\n", "output": "6"}, 10.0, "") is True


def test_the_file_helpers_follow_sed():
    assert CF._strip_cr(b"a\r\nb\r\nc\r") == b"a\nb\nc"
    assert CF._strip_cr(b"a\rb\n") == b"a\rb\n"
    assert CF._drop_empty_lines(b"a\n\nb\n\n") == b"a\nb\n"
    assert CF._drop_empty_lines(b"a\n\nb") == b"a\nb"
    assert CF._drop_empty_lines(b"\n\n") == b""
    assert CF._drop_empty_lines(b" \n\r\n") == b" \n\r\n"
    assert CF._expected_as_judged("Y\r\n\r\n\r\n") == b"Y\n"
    assert CF._expected_as_judged("") == b""


def test_a_nonzero_exit_fails_even_with_the_right_output():
    assert CF._run_case("print(1)\nraise SystemExit(3)\n", {"input": "", "output": "1"},
                        10.0, "") is False


def test_the_checker_verdict_is_1_or_100_on_exit_0():
    test = {"input": "", "output": "Y"}
    for body, want in (("print(1)", True), ("print(100)", True), ("print(' 1 ')\nprint()", True),
                       ("print(0)", False), ("print('1.0')", False),
                       ("print('debug')\nprint(1)", False),
                       ("import sys\nprint(1)\nsys.exit(2)", False)):
        assert CF._run_case("print('X')\n", test, 10.0, body + "\n") is want, body


def test_the_checker_reads_the_files_the_run_script_writes():
    """argv: input (\\r stripped), expected (echo'd, \\r and empty lines dropped), output (empty
    lines dropped)."""
    checker = ("import sys\n"
               "inp, ref, sub = (open(p, 'rb').read() for p in sys.argv[1:4])\n"
               "print(1 if (inp, ref, sub) == (b'a\\nb\\n', b'Y\\n', b'X\\nZ') else 0)\n")
    code = "print('X\\n\\nZ', end='')\n"
    assert CF._run_case(code, {"input": "a\r\nb\r\n", "output": "Y\r\n\r\n\r\n"}, 10.0,
                        checker) is True


def test_file_input_mode_reads_input_txt_and_writes_output_txt():
    code = ("d = open('input.txt').read().split()\n"
            "open('output.txt', 'w').write(str(sum(map(int, d))) + '\\n')\n"
            "print('stdout is discarded')\n")
    gold = _gold([{"input": "1 2\r\n", "output": "3"}], input_mode="file")
    assert CF._eval_codeforces("```python\n" + code + "```", gold) is True
    assert CF._run_case("print(3)\n", {"input": "1 2", "output": "3"}, 10.0, "", "file") is False


def test_the_program_runs_with_the_run_scripts_unlimited_stack():
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    code = "import resource\nprint(resource.getrlimit(resource.RLIMIT_STACK)[0])\n"
    assert CF._run_case(code, {"input": "", "output": str(hard)}, 10.0, "") is True


def test_the_loader_keeps_the_rows_input_mode(monkeypatch):
    row = {"id": "1/A", "contest_id": "1", "contest_type": "CF", "official_tests":
           [{"input": "1", "output": "2"}], "generated_checker": None, "time_limit": 2.0,
           "memory_limit": 256.0, "testset_size": 1, "official_tests_complete": True,
           "input_mode": "file", "generated_tests": 0, "language": "python", "prompt": "Solve."}
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [row])
    (msgs, gold, meta), = CF._load_codeforces_samples()
    assert gold["input_mode"] == "file"


def test_the_result_records_the_extraction_and_the_judge(monkeypatch):
    traces = [{"sample_idx": 0, "response_text": "print(2)", "is_correct": False,
               "gold_answer": _gold([{"input": "", "output": "2"}])},
              {"sample_idx": 1, "response_text": "```python\nprint(2)\n```", "is_correct": True,
               "gold_answer": _gold([{"input": "", "output": "2"}])}]
    monkeypatch.setattr(CF, "_require_generated_tests_mirror", lambda: None)
    monkeypatch.setattr(CF, "_load_codeforces_samples", lambda **k: [])
    monkeypatch.setattr(CF, "run_eval_suite", lambda **k: {"sample_traces": traces})
    res = CF.run_codeforces("m", "http://x", 1)
    assert res["replies_without_python_block"] == 1
    assert "1416fa0cf21595d2083b399a2a0bbddd7f6e9563" in res["extraction"]
    assert "7a38da202481bb7a85647a2c6cdc00990ad051bc" in res["judge"]


# --------------------------------------------------------------------------- #
# bigcodebench
# --------------------------------------------------------------------------- #
import asyncio
import hashlib
import json
import types
from unittest import mock

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bigcodebench as B


def _bcb_suite_stubs(monkeypatch, tasks=("BigCodeBench/0",)):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "harness_cpus", lambda: [0])
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [
        ([{"role": "user", "content": f"{split} {tid}"}], tid,
         {"category": split, "task_id": tid, "split": split, "hard": False})
        for tid in tasks for split in B.SPLITS])


class _Resp:
    status = 200

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body

    async def text(self):
        return json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_bigcodebench_scores_text_the_reasoning_parser_moved_and_names_it(monkeypatch):
    """No-think prefill path: the server's reasoning parser put the start of the continuation in
    `reasoning`. base recovers reasoning + content in generation order; that whole text is what
    the harness sanitizes, and the run names the replies (not comparable without token ids)."""
    _bcb_suite_stubs(monkeypatch)
    real_send = base._send_single_request

    class _Session:
        def post(self, url, json=None, **kw):
            return _Resp({"choices": [{"message": {"role": "assistant",
                                                   "content": "    return 1\n```",
                                                   "reasoning": "def task_func():\n"},
                                       "finish_reason": "stop", "stop_reason": None}],
                          "usage": {"completion_tokens": 9}})

    async def send(**kw):
        kw["session"] = _Session()
        return await real_send(**kw)

    written = []

    def fake_run(cmd, **k):
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        if "bigcodebench.sanitize" in cmd:
            with open(os.path.join(mount, "samples.jsonl"), encoding="utf-8") as f:
                written.extend(json.loads(line) for line in f)
        else:
            with open(os.path.join(mount, "x_eval_results.json"), "w") as f:
                json.dump({"eval": {"BigCodeBench/0": [{"status": "pass", "details": {}}]}}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    with mock.patch.object(base, "_send_single_request", send):
        r = B.run_bigcodebench("m", "http://x/v1", 1)
    # one sanitize pass per split, each over its one task
    assert [w["solution"] for w in written] == ["def task_func():\n    return 1\n```"] * 2
    assert r["replies_split_by_reasoning_parser"] == {
        "count": 2, "recovered_from": {"reasoning_then_content": 2}}
    assert r["server_parser_events"] == {base.SERVER_PARSER_MOVED_TEXT: 2}
    assert r["leaderboard_comparable"] is False
    assert "2 no-think repl(ies) the server's reasoning parser split were rebuilt as reasoning " \
           "then content" in r["leaderboard_comparable_reason"]


def test_bigcodebench_moved_text_rebuilt_from_token_ids_stays_comparable(monkeypatch):
    moved = [{"extra_payload": {"split": s, "task_id": "BigCodeBench/0"},
              "server_parser": {"event": base.SERVER_PARSER_MOVED_TEXT,
                                "recovered_from": "token_ids"}} for s in B.SPLITS]
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])
    monkeypatch.setattr(B, "_make_scorer", lambda *a: None)
    monkeypatch.setattr(B, "run_eval_suite", lambda **k: {
        "accuracy": 50.0, "temperature": 0.0, "status": "success", "sample_traces": moved,
        "max_tokens_sent": [B.UPSTREAM_MAX_NEW_TOKENS]})
    r = B.run_bigcodebench("m", "http://h/v1", 1)
    assert r["replies_split_by_reasoning_parser"] == {"count": 2,
                                                      "recovered_from": {"token_ids": 2}}
    assert "reasoning parser" not in (r.get("leaderboard_comparable_reason") or "")


def test_bigcodebench_a_task_without_a_harness_result_is_a_harness_error(monkeypatch):
    def fake_run(cmd, **k):
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        if "bigcodebench.sanitize" not in cmd:
            with open(os.path.join(mount, "s_eval_results.json"), "w") as f:
                json.dump({"eval": {"BigCodeBench/0": [{"status": "pass", "details": {}}]}}, f)
            with open(os.path.join(mount, "s_pass_at_k.json"), "w") as f:
                json.dump({"pass@1": 0.5}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    traces = [{"sample_idx": i, "extra_payload": {"task_id": f"BigCodeBench/{i}",
                                                   "split": "instruct"},
               "response_text": "x"} for i in range(2)]
    metrics = {}
    asyncio.run(B._make_scorer("full", 1, metrics, "img")(traces))
    assert traces[0]["is_correct"] is True and not traces[0].get("scoring_excluded")
    t1 = traces[1]
    assert t1["scoring_excluded"] is True and t1["status"] == "HARNESS_ERROR"
    assert t1["harness_error"]["kind"] == "no_harness_result"
    assert metrics["bigcodebench_unscored"] == 1
    assert metrics["bigcodebench_pass_at_1"] is None
    assert "it returned no result for 1 task(s)" in \
        metrics["bigcodebench_pass_at_1_withheld"]["instruct"]
    # base leaves it out of accuracy and category_accuracy and names it in the status
    assert base.is_scored(traces[0]) and not base.is_scored(t1)


def test_bigcodebench_run_names_harness_misses_as_left_out(monkeypatch):
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])

    def scorer(subset, parallel, metrics, image):
        metrics["bigcodebench_unscored"] = 2
    monkeypatch.setattr(B, "_make_scorer", scorer)
    monkeypatch.setattr(B, "run_eval_suite", lambda **k: {
        "accuracy": 50.0, "temperature": 0.0, "status": "completed_with_errors",
        "sample_traces": [], "max_tokens_sent": [B.UPSTREAM_MAX_NEW_TOKENS]})
    r = B.run_bigcodebench("m", "http://h/v1", 1)
    assert "2 task(s) got no harness result (harness errors, left out of the score; " \
           "bigcodebench_unscored)" in r["leaderboard_comparable_reason"]
    assert "scored as failed" not in r["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# multipl_e
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import multipl_e as M


def _mpe_traces(n):
    gold = json.dumps({"language": "lua", "name": "HumanEval_0", "prompt": "", "tests": "",
                       "stop_tokens": []})
    return [{"sample_idx": i, "gold_answer": gold,
             "response_text": "[[ ## completion ## ]]\nreturn 1\n\n[[ ## completed ## ]]"}
            for i in range(n)]


def test_multipl_e_a_completion_without_a_container_result_is_a_harness_error(monkeypatch):
    def fake_run(cmd, **k):
        out = next(a.split(":", 1)[0] for a in cmd if a.endswith(":/out:rw"))
        with open(os.path.join(out, "p000000.results.json"), "w") as f:
            json.dump({"name": "p000000", "results": [{"status": "OK", "exit_code": 0}]}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    traces = _mpe_traces(2)
    metrics = {}
    asyncio.run(M._make_scorer("img", 1, metrics)(traces))
    assert traces[0]["is_correct"] is True
    t1 = traces[1]
    assert t1["scoring_excluded"] is True and t1["status"] == "HARNESS_ERROR"
    assert t1["harness_error"]["kind"] == "no_harness_result"
    assert metrics["multipl_e_unscored"] == 1


def test_multipl_e_a_container_that_scored_nothing_withholds_the_score(monkeypatch):
    monkeypatch.setattr(M.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr="boom"))
    traces = _mpe_traces(2)
    metrics = {}
    asyncio.run(M._make_scorer("img", 1, metrics)(traces))
    assert all(t["scoring_excluded"] and t["harness_error"]["kind"] == "no_results"
               for t in traces)
    assert "no parsed results" in metrics["multipl_e_report"]["error"]


def test_multipl_e_pass_at_1_is_the_per_problem_mean():
    """upstream pass_k.py: c/n per problem, then the mean over problems."""
    t = []
    for i in range(10):                                   # problem 0: 10 scored, all correct
        t.append({"sample_idx": i, "source_sample_idx": 0, "category": "lua",
                  "is_correct": True})
    t.append({"sample_idx": 10, "source_sample_idx": 1, "category": "lua", "is_correct": False})
    for i in range(11, 20):                               # problem 1: 9 harness errors
        t.append({"sample_idx": i, "source_sample_idx": 1, "category": "lua",
                  "is_correct": False, "scoring_excluded": True})
    overall, per_lang = M.per_problem_pass_at_1(t)
    assert overall == 50.0 and per_lang == {"lua": 50.0}
    res = {"accuracy": 90.91, "sample_traces": t,
           "category_accuracy": {"lua": {"correct": 10, "total": 11, "accuracy": 90.91}},
           "attempts": {"avg_at_k": 90.91}}
    M.apply_per_problem_pass_at_1(res)
    assert res["accuracy"] == 50.0 and res["category_accuracy"]["lua"]["accuracy"] == 50.0
    assert res["category_accuracy"]["lua"]["correct"] == 10
    assert res["attempts"]["avg_at_k"] == 50.0


# --------------------------------------------------------------------------- #
# lcb
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import lcb as L


def test_lcb_pass_at_1_is_lcb_runners_per_problem_mean():
    t = [{"source_sample_idx": 0, "is_correct": True} for _ in range(10)]
    t.append({"source_sample_idx": 1, "is_correct": False})
    assert L.pass_at_1(t) == 50.0
    assert L._at_k(t)["avg_at_k"] == 50.0
    assert L.pass_at_1([]) is None
    even = [{"source_sample_idx": p, "is_correct": i < p} for p in range(3) for i in range(4)]
    assert round(L.pass_at_1(even), 6) == round(100.0 * 3 / 12, 6)   # equal n: pooled mean


def test_lcb_headline_with_uneven_exclusions_matches_lcb_runner(monkeypatch):
    traces = []
    for i in range(10):
        traces.append({"sample_idx": i, "source_sample_idx": 0, "category": "code_gen_easy",
                       "is_correct": True})
    traces.append({"sample_idx": 10, "source_sample_idx": 1, "category": "code_gen_easy",
                   "is_correct": False})
    for i in range(11, 20):
        traces.append({"sample_idx": i, "source_sample_idx": 1, "category": "code_gen_easy",
                       "is_correct": False, "scoring_excluded": True,
                       "harness_error": {"kind": "endpoint_failure", "detail": "500"}})
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: [])
    monkeypatch.setattr(L, "_include_self_repair", lambda: False)
    monkeypatch.setattr(L, "run_eval_suite", lambda **k: {
        "accuracy": 90.91, "correct_answers": 10, "total_questions": 20, "temperature": 0.2,
        "sample_traces": traces, "status": "completed_with_errors",
        "attempts": {"attempts_per_sample": 10, "avg_at_k": 90.91}})
    r = L.run_lcb("m", "http://x/v1", 1)
    assert r["accuracy"] == 50.0
    assert r["scenarios"]["code_generation"] == {"correct": 10, "total": 11, "accuracy": 50.0}
    assert r["attempts"]["avg_at_k"] == 50.0


# --------------------------------------------------------------------------- #
# scicode
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import scicode as S


def _body_after_header(path):
    with open(path, "rb") as f:
        data = f.read()
    return data.split(b"\n\n", 1)[1]


def test_the_vendored_scicode_modules_are_upstreams_verbatim():
    vend = os.path.join(S.SCICODE_VENDOR_DIR, "scicode")
    cmp_body = _body_after_header(os.path.join(vend, "compare", "cmp.py"))
    assert hashlib.sha256(cmp_body).hexdigest() == S.SCICODE_CMP_SHA256
    init_body = _body_after_header(os.path.join(vend, "__init__.py"))
    assert hashlib.sha256(init_body).hexdigest() == \
        "63fd5804866fc8cac57c9cdca88a14daddd4c8ce9653a37b8a5692a3629261b8"
    with open(os.path.join(vend, "compare", "__init__.py"), encoding="utf-8") as f:
        assert all(line.startswith("#") for line in f.read().splitlines())


def test_the_sandbox_imports_the_vendored_cmp():
    assert S.check_cmp_importable() is None


def _tiny_h5(tmp_path):
    import h5py
    import numpy as np
    path = str(tmp_path / "t.h5")
    with h5py.File(path, "w") as f:
        f["17.1/test1/var1"] = np.array([1.0, 2.0])
    return path


def test_a_step_whose_test_imports_cmp_is_scored_on_the_models_code(tmp_path, monkeypatch,
                                                                    request):
    """Step 17.1's shape: the test case imports scicode.compare.cmp before calling the model's
    function. It used to fail with ModuleNotFoundError whatever the model wrote.

    This covers the cmp import and the scoring, not the jail: the h5 fixture lives under pytest's
    temporary directory in /tmp, which the bubblewrap jail replaces with an empty tmpfs, so the
    step runs unsandboxed on purpose (test_the_sandbox_imports_the_vendored_cmp covers the import
    inside the jail)."""
    from gbench.runners.eval_suites import sandbox
    monkeypatch.setenv("GBENCH_SANDBOX", "none")
    sandbox.sandbox_available.cache_clear()
    request.addfinalizer(sandbox.sandbox_available.cache_clear)
    h5 = _tiny_h5(tmp_path)
    tests = ["from scicode.compare.cmp import cmp_tuple_or_list\n"
             "assert cmp_tuple_or_list(f(), target)"]
    good = S._score_step("import numpy as np\ndef f():\n    return [1.0, 2.0]\n", "17.1",
                         tests, h5, 120)
    assert good["correct"] is True, good["stderr_tail"]
    bad = S._score_step("def f():\n    return [1.0, 3.0]\n", "17.1", tests, h5, 120)
    assert bad["correct"] is False and "AssertionError" in bad["stderr_tail"]
    assert "ModuleNotFoundError" not in bad["stderr_tail"]


def test_run_scicode_refuses_to_start_when_cmp_cannot_be_imported(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(S, "resolve_test_data", lambda: (str(tmp_path / "t.h5"), "env"))
    monkeypatch.setattr(S, "check_test_data_digest", lambda h5, prov: ("d" * 64, None))
    monkeypatch.setattr(S, "check_cmp_importable", lambda: "the sandbox could not import it")
    with pytest.raises(Exception) as err:
        S.run_scicode("m", "http://x/v1", 1)
    assert "the sandbox could not import it" in str(err.value)
    assert "31 scored sub-steps" in str(err.value)


def test_lcb_category_accuracy_is_the_per_problem_mean(monkeypatch):
    traces = [{"sample_idx": i, "source_sample_idx": 0, "category": "code_gen_hard",
               "is_correct": True} for i in range(10)]
    traces.append({"sample_idx": 10, "source_sample_idx": 1, "category": "code_gen_hard",
                   "is_correct": False})
    traces += [{"sample_idx": i, "source_sample_idx": 1, "category": "code_gen_hard",
                "is_correct": False, "scoring_excluded": True,
                "harness_error": {"kind": "endpoint_failure", "detail": "500"}}
               for i in range(11, 20)]
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: [])
    monkeypatch.setattr(L, "_include_self_repair", lambda: False)
    monkeypatch.setattr(L, "run_eval_suite", lambda **k: {
        "accuracy": 90.91, "correct_answers": 10, "total_questions": 20, "temperature": 0.2,
        "sample_traces": traces, "status": "completed_with_errors",
        "category_accuracy": {"code_gen_hard": {"correct": 10, "total": 11, "accuracy": 90.91}},
        "attempts": {"attempts_per_sample": 10, "avg_at_k": 90.91}})
    r = L.run_lcb("m", "http://x/v1", 1)
    assert r["category_accuracy"]["code_gen_hard"] == {"correct": 10, "total": 11,
                                                       "accuracy": 50.0}


# --------------------------------------------------------------------------- #
# spider2
# --------------------------------------------------------------------------- #
import sqlite3

import pandas as pd

from gbench.runners.eval_suites import spider2 as SP


def _sqlite(tmp_path, rows, ddl="CREATE TABLE t (year TEXT, code TEXT, v REAL)"):
    path = str(tmp_path / "db.sqlite")
    con = sqlite3.connect(path)
    con.execute(ddl)
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


def _csv(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


def test_spider2_the_prediction_round_trips_through_csv_as_upstream(tmp_path):
    """Numbers the database stores as TEXT read back from CSV as numbers, as the gold does
    (local017's own gold SQL returns Year as '2001'); compared natively they never matched."""
    db = _sqlite(tmp_path, [("2001", "007", 19.4260563380282)])
    gold = _csv(tmp_path / "local900.csv", "year,code,v\n2001,7,19.43\n")
    reply = "```sql\nSELECT year, code, CAST(v AS TEXT) AS v FROM t\n```"
    assert SP.score_local_instance("local900", reply, db, [gold], {}) == (1, None)
    con = sqlite3.connect(db)
    native = pd.read_sql_query("SELECT year, code, CAST(v AS TEXT) AS v FROM t", con)
    con.close()
    assert SP.compare_pandas_table(native, pd.read_csv(gold)) == 0    # what used to be scored


def test_spider2_a_failed_query_is_the_models_zero_with_upstreams_error_info(tmp_path):
    db = _sqlite(tmp_path, [("2001", "1", 1.0)])
    gold = _csv(tmp_path / "local900.csv", "year\n2001\n")
    score, info = SP.score_local_instance("local900", "SELECT nope FROM missing", db, [gold], {})
    assert score == 0 and "no such table" in info
    score, info = SP.score_local_instance("local900", "SELECT code FROM t", db, [gold], {})
    assert (score, info) == (0, "Result Error")


def test_spider2_a_result_file_that_cannot_be_read_is_upstreams_evaluation_error(tmp_path,
                                                                                 monkeypatch):
    db = _sqlite(tmp_path, [("2001", "1", 1.0)])
    gold = _csv(tmp_path / "local900.csv", "year\n2001\n")
    monkeypatch.setattr(SP, "get_sqlite_result", lambda *a, **k: (True, None))
    score, info = SP.score_local_instance("local900", "SELECT year FROM t", db, [gold], {})
    assert score == 0 and info.startswith("Evaluation Error:")


def _gold_frame_dir(tmp_path, names):
    exec_dir = tmp_path / "gold" / "exec_result"
    exec_dir.mkdir(parents=True)
    for name in names:
        _csv(exec_dir / name, "a,b,c,d\n1,2,3,4\n5,6,7,8\n")
    return str(tmp_path / "gold"), str(exec_dir)


def test_spider2_single_variant_condition_cols_follow_compare_multi_pandas_table(tmp_path):
    """Only {id}_a.csv: upstream passes the flat list's FIRST entry as condition_cols (index 0 is
    falsy, so every column is required)."""
    gold_dir, exec_dir = _gold_frame_dir(tmp_path, ["local900_a.csv"])
    paths = SP._resolve_gold_paths(gold_dir, "local900")
    assert [os.path.basename(p) for p in paths] == ["local900_a.csv"]
    gold = pd.read_csv(paths[0])
    only_b, only_c = gold[["b"]], gold[["c"]]
    assert SP._score_against_golds(only_b, paths, [1, 2], False, instance_id="local900") == 1
    assert SP._score_against_golds(only_c, paths, [1, 2], False, instance_id="local900") == 0
    assert SP._score_against_golds(gold[["a"]], paths, [0], False, instance_id="local900") == 0
    assert SP._score_against_golds(gold, paths, [0], False, instance_id="local900") == 1


def test_spider2_several_variants_apply_the_flat_condition_cols_to_each(tmp_path):
    gold_dir, _ = _gold_frame_dir(tmp_path, ["local900_a.csv", "local900_b.csv"])
    paths = SP._resolve_gold_paths(gold_dir, "local900")
    gold = pd.read_csv(paths[0])
    assert SP._score_against_golds(gold[["b"]], paths, [1, 2], False, instance_id="local900") == 0
    assert SP._score_against_golds(gold[["b", "c"]], paths, [1, 2], False,
                                   instance_id="local900") == 1


def test_spider2_the_id_csv_alone_is_used_when_it_exists(tmp_path):
    gold_dir, _ = _gold_frame_dir(tmp_path, ["local900.csv", "local900_a.csv"])
    paths = SP._resolve_gold_paths(gold_dir, "local900")
    assert [os.path.basename(p) for p in paths] == ["local900.csv"]
    assert SP._is_single_gold(paths, "local900") is True
    gold = pd.read_csv(paths[0])
    assert SP._score_against_golds(gold[["b"]], paths, [1, 2], False, instance_id="local900") == 0


def test_spider2_ignore_order_defaults_to_false_as_upstream(tmp_path):
    db = _sqlite(tmp_path, [("2001", "1", 1.0), ("2002", "2", 2.0)])
    gold = _csv(tmp_path / "local900.csv", "year\n2002\n2001\n")
    reply = "```sql\nSELECT year FROM t ORDER BY year\n```"
    assert SP.score_local_instance("local900", reply, db, [gold], {})[0] == 0
    assert SP.score_local_instance("local900", reply, db, [gold], {"ignore_order": True})[0] == 1


def test_spider2_the_scorer_grades_through_score_local_instance(tmp_path):
    db = _sqlite(tmp_path, [("2001", "007", 1.0)])
    os.rename(db, str(tmp_path / "shop.sqlite"))
    gold_dir, exec_dir = str(tmp_path / "gold"), str(tmp_path / "gold" / "exec_result")
    os.makedirs(exec_dir)
    _csv(os.path.join(exec_dir, "local900.csv"), "code\n7\n")
    traces = [{"sample_idx": 0, "response_text": "```sql\nSELECT code FROM t\n```",
               "extra_payload": {"db": "shop", "instance_id": "local900",
                                 "condition_cols": None, "ignore_order": False}},
              {"sample_idx": 1, "response_text": "```sql\nSELECT year FROM t\n```",
               "extra_payload": {"db": "shop", "instance_id": "local900"}},
              {"sample_idx": 2, "response_text": "SELECT 1",
               "extra_payload": {"db": "gone", "instance_id": "local900"}}]
    asyncio.run(SP._make_scorer(str(tmp_path), gold_dir)(traces))
    assert traces[0]["is_correct"] is True and traces[0]["status"] == "OK"
    assert traces[1]["is_correct"] is False and traces[1]["error_info"] == "Result Error"
    assert traces[2]["status"] == "SCORING_ERROR" and traces[2]["scoring_excluded"] is True
    assert traces[2]["harness_error"]["kind"] == "grader_error"


def test_spider2_the_checkout_revision_is_recorded_and_checked(tmp_path, monkeypatch):
    assert SP._git_head(str(tmp_path)) is None
    assert SP._git_head(None) is None
    pinned = SP.SPIDER2_COMMIT
    assert SP.revision_gaps({"metadata": pinned, "gold": pinned}) == []
    gaps = SP.revision_gaps({"metadata": "a" * 40, "gold": None})
    assert any("gold checkout's commit could not be read" in g for g in gaps)
    assert any("metadata checkout is Spider2 aaaaaaaa, not the pinned "
               f"{pinned[:8]}" in g for g in gaps)

    monkeypatch.setattr(SP, "check_spider2_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SP, "_load_spider2_samples", lambda limit=None: [])
    monkeypatch.setattr(SP, "run_eval_suite", lambda **k: {"accuracy": 0.0, "sample_traces": []})
    monkeypatch.setattr(SP, "repo_revisions", lambda: {"metadata": pinned, "gold": "b" * 40})
    r = SP.run_spider2("m", "http://x/v1", 1)
    assert r["spider2_repo_pinned"] == pinned
    assert r["spider2_repo_revisions"] == {"metadata": pinned, "gold": "b" * 40}
    reason = r["leaderboard_comparable_reason"]
    assert "the gold checkout is Spider2 bbbbbbbb" in reason
    assert "the prompt and sampling are gbench's own" in reason
    assert r["prompt"] == SP.PROMPT_NOTE and "prescribes no generation prompt" in r["prompt"]
    assert SP.SPIDER2_COMMIT[:8] in r["scoring"]


def test_spider2_git_head_reads_a_real_checkout(tmp_path):
    import subprocess
    repo = tmp_path / "Spider2"
    sub = repo / "spider2-lite" / "evaluation_suite" / "gold"
    sub.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
           "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}
    _csv(sub / "x.csv", "a\n1\n")
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env,
                       capture_output=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], env=env,
                          capture_output=True, text=True).stdout.strip()
    assert SP._git_head(str(sub)) == head


# --------------------------------------------------------------------------- #
# omnidocbench
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import omnidocbench as O


def _page(subset, source):
    return {"page_info": {"page_attribute": {"subset": subset, "data_source": source}}}


def _split_like_upstream():
    """The split's shape: one data source at a time, the v1.5 slides first, the hard subsets
    after them."""
    pages = [_page("v1.5", "PPT2PDF") for _ in range(40)]
    pages += [_page("v1.5", src) for src in ("book", "exam_paper", "newspaper") for _ in range(20)]
    pages += [_page("equation_hard", "book") for _ in range(10)]
    pages += [_page("layout_hard", "magazine") for _ in range(10)]
    pages += [_page("table_hard", "research_report") for _ in range(10)]
    return pages


def test_omnidocbench_a_limited_run_is_stratified_not_a_head():
    pages = _split_like_upstream()
    chosen = O.select_pages(pages, 4)
    assert chosen == sorted(chosen) and len(chosen) == 4
    assert sorted(O.page_subset(pages[k]) for k in chosen) == [
        "equation_hard", "layout_hard", "table_hard", "v1.5"]
    assert chosen != list(range(4))
    many = O.select_pages(pages, 20)
    v15_sources = {O.page_source(pages[k]) for k in many if O.page_subset(pages[k]) == "v1.5"}
    assert len(v15_sources) == 4                       # the v1.5 picks spread over its sources
    assert O.select_pages(pages, 20) == many           # seeded
    assert O.select_pages(pages, None) == list(range(len(pages)))
    assert O.select_pages(pages, 10 ** 6) == list(range(len(pages)))


def test_omnidocbench_page_attributes_tolerate_missing_fields():
    assert O.page_subset({}) == "unknown" and O.page_source({"page_info": {}}) == "unknown"
    assert O.page_source({"page_info": {"page_attribute": "x"}}) == "unknown"


def test_omnidocbench_the_loader_scores_the_selected_pages(monkeypatch, tmp_path):
    from PIL import Image as PILImage

    pages = _split_like_upstream()
    names = [f"p{i:03d}.png" for i in range(len(pages))]
    annos = []
    for name, page in zip(names, pages):
        page = json.loads(json.dumps(page))
        page["page_info"]["image_path"] = name
        page["layout_dets"] = [{"category_type": "text_block", "text": name, "order": 0}]
        annos.append(page)
    img = PILImage.new("RGB", (4, 4))

    class _DS:
        def __init__(self, decoded):
            self.decoded = decoded

        def __len__(self):
            return len(names)

        def __getitem__(self, i):
            return {"image": img} if self.decoded else {"image": {"path": names[i]}}

        def cast_column(self, *a, **k):
            return _DS(False)

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _DS(True))
    anno_file = tmp_path / "OmniDocBench.json"
    anno_file.write_text(json.dumps(annos), encoding="utf-8")
    monkeypatch.setattr(O, "hf_hub_download", lambda **k: str(anno_file))
    samples = O._load_omnidocbench_samples(limit=4)
    got = sorted(json.loads(gold)["text"] for _msgs, gold, _meta in samples)
    want = sorted(names[k] for k in O.select_pages(annos, 4))
    assert got == want
    assert sorted(meta["category"] for _m, _g, meta in samples) == [
        "equation_hard", "layout_hard", "table_hard", "v1.5"]
