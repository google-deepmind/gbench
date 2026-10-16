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

"""hmmt / aime on MathArena's protocol, the LCB grader that never holds the expected outputs,
LCB / MultiPL-E sample counts, and BigCodeBench's two splits and prefill path.

Offline: datasets and the model endpoint are stubbed. MathArena's grader runs for real in this
interpreter (`run_driver(where="host")`) for answers that do not need sympy's LaTeX parser; the
grader image itself is exercised only when it has been built (last test). The LCB graders run
for real (unsandboxed under conftest's GBENCH_SANDBOX=none).
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import types

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import matharena_common as MA

REPO = os.path.join(os.path.dirname(__file__), "..")


# --------------------------------------------------------------------------- #
# MathArena: vendored sources, prompts, last chance                            #
# --------------------------------------------------------------------------- #
def test_vendored_matharena_files_are_the_pinned_blobs():
    files = MA.vendored_files()
    assert set(files) == set(MA.VENDORED_BLOBS)
    assert "def extract_and_grade(" in files["matharena/grader.py"]
    assert "def check_answers(" in files["matharena/parser.py"]
    assert "MIT License" in files["LICENSE"] and "SRI Lab, ETH Zurich" in files["LICENSE"]
    assert MA.MATHARENA_COMMIT == "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b"


def test_a_modified_vendored_file_is_refused(tmp_path, monkeypatch):
    for rel in MA.VENDORED_BLOBS:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((MA.SRC_DIR / rel).read_bytes())
    (tmp_path / "matharena" / "parser.py").write_text("# edited\n", encoding="utf-8")
    monkeypatch.setattr(MA, "SRC_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="not MathArena's"):
        MA.vendored_files()


def test_competition_configs_and_prompts_are_matharenas():
    hmmt = MA.competition_config(MA.COMPETITIONS["hmmt/hmmt_feb_2026"])
    aime = MA.competition_config(MA.COMPETITIONS["aime/aime_2026"])
    assert hmmt["strict_parsing"] is False and hmmt["n_problems"] == 33
    assert aime["n_problems"] == 30
    assert MA.build_prompt(hmmt, "P {x}") == "Put your final answer within \\boxed{}.\n\nP {x}"
    assert MA.build_prompt(aime, "Q") == ("Put your final answer within \\boxed{}.\nThe answer is "
                                          "an integer between 0 and 999 inclusive.\n\nQ")
    from gbench.runners.eval_suites import hmmt as H
    assert hmmt["instruction"].replace("{{}}", "{}") == H._ANSWER_INSTRUCTION


def test_last_chance_prompt_is_runner_verbatim():
    assert MA.LAST_CHANCE_PROMPT == (
        "\nYour last message does not provide a final answer in a way that follows the "
        "formatting instructions.\nPlease based on the conversation history, report the final "
        "answer again within \\boxed{}.\nIf you did not find the answer, please use "
        "\\boxed{None}.\nDo not reason about the problem again or use tools, simply try to "
        "extract the final answer from the previous reasoning.\nBoxed answers in "
        "thinking/reasoning stages will be ignored; only the final response message is "
        "considered.\n")
    assert "only valid answers" in MA._last_chance_prompt(["A", "B"])


def _conv(text, reasoning=None):
    return MA._prompt_messages({"messages": [{"role": "user", "content": "p"}]}) \
        + MA._assistant_turn(text, reasoning)


def test_host_driver_check_mode_is_the_runners_last_chance_test():
    config = MA.competition_config(MA.COMPETITIONS["hmmt/hmmt_feb_2026"])
    items = [{"id": 0, "conversation": _conv("the answer is 5")},
             {"id": 1, "conversation": _conv("", reasoning="\\boxed{5}")},   # box only in reasoning
             {"id": 2, "conversation": _conv("so \\boxed{5}")},
             {"id": 3, "conversation": _conv("<think>\\boxed{5}</think>The answer is five.")}]
    recs = MA.run_driver("check", items, config, where="host")
    assert [r["valid"] for r in recs] == [False, False, True, False]


def test_host_driver_grades_with_matharenas_parser():
    config = MA.competition_config(MA.COMPETITIONS["aime/aime_2026"])
    cases = [("so \\boxed{277}", "277", True), ("\\boxed{0277}", "277", True),
             ("The answer is 62.", "62", True),          # strict_parsing false: last integer
             ("\\boxed{61}", "62", False), ("", "62", False)]
    items = [{"id": i, "gold": g, "output_tokens": 0, "conversation": _conv(t)}
             for i, (t, g, _) in enumerate(cases)]
    recs = MA.run_driver("grade", items, config, where="host")
    assert [r.get("is_correct") for r in recs] == [c[2] for c in cases]
    assert recs[0]["answer"] == "277"


def test_eval_helpers_grade_one_response_on_the_host():
    from gbench.runners.eval_suites import aime as A, hmmt as H
    assert H._eval_hmmt(r"\boxed{\frac{1}{2}}", "0.5") is True
    assert H._eval_hmmt("We test 5 cases, therefore the answer is 12.", "5") is False
    assert A._eval_aime("Final: \\boxed{42}", "42") is True


# --------------------------------------------------------------------------- #
# MathArena: the scorer (last chance, then grading), end to end offline        #
# --------------------------------------------------------------------------- #
def _host_driver(monkeypatch):
    real = MA.run_driver
    monkeypatch.setattr(MA, "run_driver",
                        lambda mode, items, config, **kw: real(mode, items, config, where="host"))


def _run_hmmt_offline(monkeypatch, replies, last_chance_reply, attempts=None):
    """run_hmmt against stubs: `replies[problem]` is the first reply's content, and
    `last_chance_reply(messages)` answers the reprompt (a Reply)."""
    from gbench.runners.eval_suites import hmmt as H
    _host_driver(monkeypatch)
    from gbench.runners.eval_suites import ojbench as OJ
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",   # a Gemma 4 model is served
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MA, "self_check", lambda config: None)
    rows = [{"problem_idx": 1, "problem": "one", "answer": "5", "problem_type": ["Algebra"]},
            {"problem_idx": 2, "problem": "two", "answer": "7", "problem_type": ["Geometry"]}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    sent = []

    async def fake_send(**k):
        sent.append(k)
        msgs = k["messages"]
        if msgs[-1]["content"] == MA.LAST_CHANCE_PROMPT:
            return last_chance_reply(msgs)
        problem = msgs[-1]["content"].rsplit("\n\n", 1)[1]
        return base.Reply(text=replies[problem], tool_calls=None, finish_reason="stop",
                          reasoning="thinking about " + problem, completion_tokens=3)

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    kw = {"attempt_count": attempts} if attempts else {}
    res = H.run_hmmt("m", "http://x/v1", 2, enable_thinking=True, max_output_tokens=4096, **kw)
    return res, sent


def test_scorer_reprompts_only_unboxed_replies_and_grades_the_last_chance_answer(monkeypatch):
    res, sent = _run_hmmt_offline(
        monkeypatch, {"one": "\\boxed{5}", "two": "I think it is seven."},
        lambda msgs: base.Reply(text="\\boxed{7}", tool_calls=None, finish_reason="stop",
                                completion_tokens=2))
    firsts = [k for k in sent if k["messages"][-1]["content"] != MA.LAST_CHANCE_PROMPT]
    lasts = [k for k in sent if k["messages"][-1]["content"] == MA.LAST_CHANCE_PROMPT]
    assert len(firsts) == 8                              # 2 problems x MathArena's 4 runs
    assert len(lasts) == 4                               # only the 4 runs of problem two
    # the reprompt resends the conversation with the reasoning dropped, then the prompt
    msgs = lasts[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "I think it is seven." and "thinking about" not in json.dumps(msgs)
    assert lasts[0]["thinking"] is True and lasts[0]["temperature"] == 1.0
    assert res["accuracy"] == 100.0
    assert res["last_chance"] == {"reprompted": 4, "request_failures": 0}
    assert res["attempts"]["attempts_per_sample"] == 4
    assert res["leaderboard_comparable"] is True
    tr = next(t for t in res["sample_traces"] if t.get("matharena_last_chance"))
    assert tr["matharena_answer"] == "7" and tr["matharena_last_chance"]["response_text"] == "\\boxed{7}"


def test_a_failed_last_chance_is_counted_and_breaks_comparability(monkeypatch):
    res, _ = _run_hmmt_offline(
        monkeypatch, {"one": "\\boxed{5}", "two": "no answer"},
        lambda msgs: base.Reply(None, None, None, None, "timeout after 1s"))
    # A last-chance request with no reply from the endpoint is a harness error: left out of the
    # score (problem one's 4 correct runs remain), counted and named, never scored wrong.
    assert res["accuracy"] == 100.0
    assert res["last_chance"]["request_failures"] == 4
    assert res["leaderboard_comparable"] is False
    assert "4 last-chance request(s) got no reply from the served endpoint" in \
        res["leaderboard_comparable_reason"]
    geometry = [t for t in res["sample_traces"] if t["category"] == "Geometry"]
    assert {t["status"] for t in geometry} == {"LAST_CHANCE_FAILED"}
    assert all(t["scoring_excluded"] and t["harness_error"]["kind"] == "endpoint_timeout"
               for t in geometry)
    assert res["scoring_excluded"] == 4 and res["status"] == "completed_with_errors"
    # base counts a last-chance request with no reply as its own harness error, not as a
    # suite exclusion
    assert "4 follow-up request(s) of the suite's protocol that got no reply" in \
        res["status_reason"]
    assert "scoring_excluded 4 (left out of the score by the suite)" not in res["status_reason"]


def test_grading_errors_are_harness_failures_not_wrong_answers(monkeypatch):
    real = MA.run_driver

    def flaky(mode, items, config, **kw):
        recs = real(mode, items, config, where="host")
        if mode == "grade":
            recs[0] = {"id": recs[0]["id"], "error": "grading took longer than 300 s"}
        return recs
    res, _ = _run_hmmt_offline(monkeypatch, {"one": "\\boxed{5}", "two": "\\boxed{7}"},
                               lambda msgs: None)
    monkeypatch.setattr(MA, "run_driver", flaky)
    from gbench.runners.eval_suites import hmmt as H
    res = H.run_hmmt("m", "http://x/v1", 2, enable_thinking=True, max_output_tokens=4096)
    assert res["grading"]["errors"] == 1 and res["judge_failures"] == 1
    assert "could not be graded" in res["leaderboard_comparable_reason"]


def test_run_hmmt_and_aime_hard_error_without_the_grader_image(monkeypatch):
    from gbench.runners.eval_suites import aime as A, hmmt as H
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (False, "image 'gbench-matharena:b89f2f0a' not found"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        H.run_hmmt("m", "http://x/v1", 1)
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MA, "self_check", lambda config: "graded a known answer wrongly")
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        A.run_aime("m", "http://x/v1", 1)


def test_self_check_needs_the_latex_parser(monkeypatch):
    from importlib.metadata import version
    config = MA.competition_config(MA.COMPETITIONS["hmmt/hmmt_feb_2026"])
    good = [{"id": i, "is_correct": want} for i, (_, _, want) in enumerate(MA._CANARIES)]
    monkeypatch.setattr(MA, "run_driver", lambda *a, **k: [dict(r) for r in good])
    assert MA.self_check(config) is None
    # a grader whose sympy cannot parse LaTeX misreads the second canary and is refused
    monkeypatch.setattr(MA, "run_driver", lambda *a, **k: [
        dict(r, is_correct=False) if r["id"] == 1 else dict(r) for r in good])
    assert "graded a known answer wrongly" in MA.self_check(config)
    monkeypatch.undo()
    # this interpreter is such a grader unless its antlr runtime happens to be 4.11
    text, gold, _ = MA._CANARIES[1]
    rec = MA.run_driver("grade", [{"id": 0, "gold": gold, "output_tokens": 0,
                                   "conversation": _conv(text)}], config, where="host")[0]
    assert rec.get("is_correct") is version("antlr4-python3-runtime").startswith("4.11")


def test_matharena_dockerfile_pins_the_antlr_runtime_and_is_documented():
    df = open(os.path.join(REPO, "docker", "matharena.Dockerfile"), encoding="utf-8").read()
    assert "antlr4-python3-runtime==4.11.0" in df and "sympy==1.14.0" in df
    assert "FROM python:3.12-slim" in df and "parse_latex" in df
    for doc in ("hmmt.md", "aime.md"):
        text = open(os.path.join(REPO, "docs", "evals", doc), encoding="utf-8").read()
        assert "docker build -t gbench-matharena:b89f2f0a -f docker/matharena.Dockerfile docker" in text
        assert "--max-output-tokens 65536" in text and "b89f2f0a" in text
        assert text.isascii()


def test_hmmt_and_aime_pin_the_current_matharena_sets():
    from gbench.runners.eval_suites import aime as A, hmmt as H
    assert (H._DATASET, H._REVISION) == ("MathArena/hmmt_feb_2026",
                                         "02fba4f74d8e68e73e66a02d540fd979c05c274c")
    assert (A._DATASET, A._REVISION) == ("MathArena/aime_2026",
                                         "d2de22f3c656b4f56cf8981212186377d1e23bc3")


@pytest.mark.skipif(shutil.which("docker") is None or subprocess.run(
    ["docker", "image", "inspect", MA.image()], capture_output=True).returncode != 0,
    reason="the gbench-matharena:b89f2f0a image has not been built")
def test_grader_image_grades_every_hmmt_2026_gold_answer():
    """Needs the built image. Every gold answer, boxed, must grade as correct, including the two
    that only sympy's LaTeX parser can read."""
    config = MA.competition_config(MA.COMPETITIONS["hmmt/hmmt_feb_2026"])
    assert MA.self_check(config) is None
    golds = ["-\\frac{1}{21}", "(3+\\sqrt{6})^{-1/3}", "50(1 - \\frac{1}{2^{101} - 1})",
             "2 - \\frac{\\pi}{2}", "74^\\circ", "\\frac{\\sqrt{1740}}{3}", "3^{2025}"]
    items = [{"id": i, "gold": g, "output_tokens": 0, "conversation": _conv(f"\\boxed{{{g}}}")}
             for i, g in enumerate(golds)]
    assert all(r.get("is_correct") for r in MA.run_driver("grade", items, config))


# --------------------------------------------------------------------------- #
# LCB: the process that runs the program never holds the expected outputs      #
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import lcb as L  # noqa: E402

_SECRET_GOLD = {"task": "generation", "fn_name": None,
                "tests": [{"input": "x", "output": "8412679903"}]}


def test_lcb_program_cannot_read_the_expected_output_from_memory():
    """A grader that ran upstream's run_test in the child with the whole input_output
    payload in memory: this program found the expected output with gc and printed it, a pass."""
    steal = ("```python\nimport gc, json\nanswer = '0'\nfor o in gc.get_objects():\n"
             "    if isinstance(o, dict):\n        if 'input_output' in o:\n"
             "            try:\n                answer = json.loads(o['input_output'])['outputs'][0]\n"
             "            except Exception:\n                pass\n"
             "        elif isinstance(o.get('outputs'), list) and o['outputs']:\n"
             "            answer = str(o['outputs'][0])\nprint(answer)\n```")
    assert L._verify_single_lcb_sample(steal, _SECRET_GOLD) is False


def test_lcb_imitating_the_childs_messages_reports_only_an_output():
    forge = ("```python\nimport os, json\nfor fd in range(3, 64):\n"
             "    for obj in ({'setup': 'ok'}, {'i': 0, 'stdout': '%s'}):\n"
             "        d = json.dumps(obj).encode()\n        try:\n"
             "            os.write(fd, b'%%d\\n' %% len(d) + d)\n        except OSError:\n"
             "            pass\nos._exit(0)\n```")
    assert L._verify_single_lcb_sample(forge % "0", _SECRET_GOLD) is False
    # reporting the right output is just the right output
    assert L._verify_single_lcb_sample(forge % "8412679903", _SECRET_GOLD) is True


def test_lcb_code_execution_allows_exactly_one_comparison():
    ex = {"task": "execution", "code": "def f(x):\n    return x * 7", "input": "f(6)",
          "output": "42"}
    V = L._verify_single_lcb_sample
    assert V("assert f(6) == 42\n[/ANSWER]", ex) is True
    assert V("assert f(6) == 41\n[/ANSWER]", ex) is False
    assert V("assert f(6) == 42, 2\n[/ANSWER]", ex) is True     # an assert message, as upstream
    # probing the stand-in for the answer, or skipping the comparison, fails
    assert V("assert f(6) == 0 or any(_gbench_expected_ == i for i in range(100))\n[/ANSWER]",
             ex) is False
    assert V("assert f(6) == 1 or True\n[/ANSWER]", ex) is False
    tup = {"task": "execution", "code": "def f(x):\n    return (x, 1)", "input": "f(6)",
           "output": "(6, 1)"}
    assert V("assert f(6) == (6, 1)\n[/ANSWER]", tup) is True
    assert V("assert f(6) == 6, 1\n[/ANSWER]", tup) is False   # upstream parses this as 6


def test_lcb_test_output_prediction_cannot_read_the_gold():
    tg = {"task": "test_gen", "output": "[3, 1]"}
    V = L._verify_single_lcb_sample
    assert V("assert f(1) == [3, 1]", tg) is True
    assert V("assert f(1) == (3, 1)", tg) is False
    assert V("assert f(1) == json.loads(expected_output)", tg) is False


_REF_RUN_TEST = r'''
import builtins, json, sys
import numpy, pandas
p = json.loads(sys.stdin.read())
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(50000)
ns = {"__builtins__": builtins.__dict__, "__name__": "lcb_testing_util"}
exec(compile(p["source"], "lcb_testing_util.py", "exec"), ns)
res, meta = ns["run_test"]({"input_output": p["io"]}, test=p["code"], debug=False, timeout=6)
def pos(x):
    try:
        return bool(x > 0)
    except Exception:
        return False
flags = [pos(x) is True for x in (res or [])]
print("REF", json.dumps(len(flags) == p["n"] and all(flags)))
'''


@pytest.mark.parametrize("code,tests,fn", [
    ("print(0.5)\nprint(1)", [("", "0.50\n1")], None),
    ("print('1 2')", [("", "1\n2")], None),
    ("x=input()\nif x=='2':\n    raise ValueError()\nprint(x)", [("1", "1"), ("2", "2")], None),
    ("if __name__ == '__main__':\n    print(input()[::-1])", [("abc", "cba")], None),
    ("class Solution:\n    def f(self, a):\n        return (a, a)", [("1", "[1, 1]")], "f"),
    ("class Solution:\n    def f(self, a):\n        return [(a, a)]", [("1", "[[1, 1]]")], "f"),
    ("class Solution:\n    def f(self, a):\n        return {1: a}", [("1", '{"1": 1}')], "f"),
    ("import numpy as np\nclass Solution:\n    def f(self, a):\n        return np.int64(a)",
     [("3", "[3]")], "f"),
    ("import numpy as np\nclass Solution:\n    def f(self, a):\n        return np.array([a, a])",
     [("3", "[3, 3]")], "f"),
    ("from collections import Counter\nclass Solution:\n    def f(self, s):\n"
     "        return Counter(s)", [('"aab"', '{"a": 2, "b": 1}')], "f"),
    ("from fractions import Fraction\nclass Solution:\n    def f(self, a):\n"
     "        return Fraction(3, 2)", [("3", "1.5")], "f"),
    ("class Solution:\n    def __init__(self):\n        self.n = 0\n    def f(self, a):\n"
     "        self.n += 1\n        return self.n", [("0", "1"), ("0", "2")], "f"),
    ("class Solution:\n    def f(self, a):\n        return [1,2]",
     [("0", "[1, 2]"), ("0", "[1, 3]"), ("0", "[1, 2]")], "f"),
])
def test_lcb_verdict_matches_upstream_run_test(code, tests, fn):
    """The split grader (execute in the child, compare in the grader) gives upstream's verdict."""
    io = json.dumps({"inputs": [t[0] for t in tests], "outputs": [t[1] for t in tests],
                     "fn_name": fn})
    r = subprocess.run([sys.executable, "-c", _REF_RUN_TEST], capture_output=True, text=True,
                       timeout=120, input=json.dumps({"source": L._LCB_TESTING_UTIL, "io": io,
                                                      "code": code, "n": len(tests)}))
    ref = json.loads(next(l[4:] for l in r.stdout.splitlines()[::-1] if l.startswith("REF ")))
    gold = {"task": "generation", "fn_name": fn,
            "tests": [{"input": i, "output": o} for i, o in tests]}
    assert L._verify_single_lcb_sample("```python\n" + code + "\n```", gold) is ref


def test_lcb_defaults_to_ten_samples_and_all_three_scenarios(monkeypatch):
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **kw: [])
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_TEST_GEN", raising=False)
    monkeypatch.delenv("LCB_INCLUDE_TEST_GEN", raising=False)
    seen = {}
    monkeypatch.setattr(L, "run_eval_suite", lambda **kw: seen.update(kw) or {
        "accuracy": 0.0, "temperature": 0.2, "sample_traces": []})
    res = L.run_lcb("m", "http://x/v1", 1)
    assert seen["attempt_count"] == 10
    # all three scenarios are loaded by default; the result lists only those that ran
    assert L._include_test_gen() is True
    assert res["lcb_scenarios"] == []
    assert res["lcb_scenarios_not_run"]["test_output_prediction"] == "no sample loaded"
    L.run_lcb("m", "http://x/v1", 1, attempt_count=12)
    assert seen["attempt_count"] == 12


def test_lcb_self_check_covers_test_output_prediction(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "_run_grader_raw",
                        lambda payload, timeout: calls.append(payload["kind"]) or (True, ""))
    assert L._grader_self_check() is None
    assert calls == ["generation", "generation", "execution", "test_output"]


# --------------------------------------------------------------------------- #
# MultiPL-E: 20 completions per problem by default                             #
# --------------------------------------------------------------------------- #
def test_multipl_e_defaults_to_the_recipes_twenty_completions(monkeypatch):
    from gbench.runners.eval_suites import multipl_e as M
    monkeypatch.delenv("GBENCH_MULTIPL_E_SAMPLES", raising=False)
    assert M._samples_per_problem(None) == 20 and M._samples_per_problem(30) == 30


# --------------------------------------------------------------------------- #
# BigCodeBench: both splits, Hard from the full run, the prefill path          #
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import bigcodebench as B  # noqa: E402


def test_bcb_complete_prompt_is_upstreams_fenced_prompt():
    assert B.make_raw_chat_prompt("  def f():\n    pass\n", "complete") == (
        B.INSTRUCTION_PREFIX + "\n```\ndef f():\n    pass\n```\n")
    assert B.RESPONSE_PREFIX == ("Below is a Python script with a self-contained function that "
                                 "solves the problem and passes corresponding tests:")
    assert B.prefill_message() == {"role": "assistant",
                                   "content": B.RESPONSE_PREFIX + "\n```python\n"}


def test_bcb_full_run_reads_the_hard_ids_and_builds_both_splits(monkeypatch):
    rows = {"bigcode/bigcodebench": [
        {"task_id": f"BigCodeBench/{i}", "instruct_prompt": f"I{i}", "complete_prompt": f"C{i}",
         "entry_point": "task_func"} for i in range(3)],
        "bigcode/bigcodebench-hard": [{"task_id": "BigCodeBench/1"}]}
    monkeypatch.setattr("datasets.load_dataset", lambda name, **kw: rows[name])
    samples = B._load_bigcodebench_samples()
    assert len(samples) == 6
    assert [(m["task_id"], m["split"], m["hard"]) for _, _, m in samples][2:4] == [
        ("BigCodeBench/1", "instruct", True), ("BigCodeBench/1", "complete", True)]
    pre = B._with_prefill(samples)
    assert pre[0][0][-1] == B.prefill_message()
    assert pre[0][2]["continue_final_message"] is True and pre[0][2]["add_generation_prompt"] is False


def test_bcb_prefill_probe_follows_the_request_builder(monkeypatch):
    def builder(forward):
        async def _send(session, api_url, model_name, messages, extra_payload, semaphore, pbar,
                        max_output_tokens=0, temperature=0.0, thinking=False):
            body = {"messages": messages, **(extra_payload if forward else {})}
            async with session.post(api_url, json=body) as resp:
                await resp.json()
            return base.Reply("", None, "stop")
        return _send

    monkeypatch.setattr(base, "_send_single_request", builder(True))
    assert B._prefill_supported() is True
    monkeypatch.setattr(base, "_send_single_request", builder(False))
    assert B._prefill_supported() is False


def test_bcb_scores_each_split_with_its_own_harness_pass(monkeypatch):
    runs = []

    def fake_run(cmd, **k):
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        if "bigcodebench.sanitize" not in cmd:
            split = cmd[cmd.index("--split") + 1]
            runs.append(split)
            passed = {"instruct": ["BigCodeBench/0", "BigCodeBench/1"],
                      "complete": ["BigCodeBench/1"]}[split]
            json.dump({"eval": {t: [{"status": "pass" if t in passed else "fail", "details": {}}]
                                for t in ("BigCodeBench/0", "BigCodeBench/1")}},
                      open(os.path.join(mount, "samples_eval_results.json"), "w"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    traces = [{"extra_payload": {"task_id": f"BigCodeBench/{i}", "split": s, "hard": i == 1},
               "response_text": "\nx", "category": s}
              for s in ("instruct", "complete") for i in range(2)]
    metrics = {}
    asyncio.run(B._make_scorer("full", 1, metrics, "img")(traces))
    assert runs == ["instruct", "complete"]
    scores = B._split_scores(traces, "full")
    assert scores["instruct"]["full"]["pass_at_1"] == 100.0
    assert scores["complete"]["full"]["pass_at_1"] == 50.0
    assert scores["complete"]["hard"] == {"correct": 1, "total": 1, "pass_at_1": 100.0}


def test_bcb_thinking_runs_take_the_openai_path_and_report_both_splits(monkeypatch):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: pytest.fail("probe not needed"))
    seen = {}
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [
        ([{"role": "user", "content": "u"}], "BigCodeBench/0",
         {"category": "instruct", "task_id": "BigCodeBench/0", "split": "instruct", "hard": True})])

    def fake_suite(**k):
        seen.update(k)
        # make_request's request for a served model: greedy (temperature 0.0), and under the
        # token-limit policy the run's budget in place of the 1280
        return {"accuracy": 25.0, "temperature": 0.0, "status": "success",
                "max_tokens_sent": [32768], "sample_traces": [
            {"extra_payload": {"split": s, "hard": h}, "is_correct": c, "response_text": "x"}
            for s, h, c in (("instruct", True, True), ("instruct", False, False),
                            ("complete", True, False), ("complete", False, False))]}
    monkeypatch.setattr(B, "run_eval_suite", fake_suite)
    r = B.run_bigcodebench("m", "http://h/v1", 1, enable_thinking=True)
    assert seen["samples"][0][0] == [{"role": "user", "content": "u"}]       # no prefill
    # the openai path's own settings, with no reasoning-model branch and no reasoning_effort
    assert seen["official_sampling"] is B.SAMPLING[False] and B.SAMPLING[False].reasoning is None
    assert "reasoning_effort" not in (seen["extra_payload"] or {})
    assert r["prefill"] is False and "--backend openai" in r["generation_path"]
    assert r["accuracy"] == 50.0 and r["blended_splits_accuracy"] == 25.0
    assert r["splits"]["instruct"]["hard"]["pass_at_1"] == 100.0
    assert r["splits"]["complete"]["full"]["pass_at_1"] == 0.0
    assert "prefill flag is false" in r["leaderboard_comparable_note"]


def test_bcb_no_think_run_hard_errors_if_the_prefill_cannot_be_sent(monkeypatch):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: False)
    with pytest.raises(RuntimeError, match="continue_final_message"):
        B.run_bigcodebench("m", "http://h/v1", 1)


def test_bcb_prefill_completions_must_start_with_the_newline(monkeypatch):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])
    monkeypatch.setattr(B, "run_eval_suite", lambda **k: {
        "accuracy": 50.0, "temperature": 0.0, "status": "success", "sample_traces": [
            {"extra_payload": {"split": "instruct"}, "is_correct": True, "response_text": "\nx"},
            {"extra_payload": {"split": "instruct"}, "is_correct": False, "response_text": "y"}]})
    r = B.run_bigcodebench("m", "http://h/v1", 1)
    assert r["prefill_first_char_newline"] == {"yes": 1, "of": 2}
    assert r["leaderboard_comparable"] is False
    assert "1 prefilled completion(s) did not start with the newline" in \
        r["leaderboard_comparable_reason"]


def test_a_grader_that_does_not_run_keeps_the_traces_and_errors_the_result(monkeypatch):
    def broken(mode, items, config, **kw):
        raise RuntimeError("docker: Cannot connect to the Docker daemon")
    _host_driver(monkeypatch)
    res, _ = _run_hmmt_offline(monkeypatch, {"one": "\\boxed{5}", "two": "\\boxed{7}"},
                               lambda msgs: None)
    monkeypatch.setattr(MA, "run_driver", broken)
    from gbench.runners.eval_suites import hmmt as H
    res = H.run_hmmt("m", "http://x/v1", 2, enable_thinking=True, max_output_tokens=4096)
    assert res["status"] == "error" and "Docker daemon" in res["error"]
    assert len(res["sample_traces"]) == 8
    assert {t["status"] for t in res["sample_traces"]} == {"GRADING_ERROR"}
    assert res["leaderboard_comparable"] is False


def test_host_driver_removes_its_temp_directory(tmp_path, monkeypatch):
    """Each host run wrote MathArena's files to a new /tmp/matharena_* and never removed it;
    the unit tests left 17,090 of them in the host's /tmp (2026-10-10)."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    config = MA.competition_config(MA.COMPETITIONS["aime/aime_2026"])
    items = [{"id": 0, "gold": "7", "output_tokens": 0, "conversation": _conv("\\boxed{7}")}]
    assert MA.run_driver("grade", items, config, where="host")[0]["is_correct"] is True
    assert list(tmp_path.iterdir()) == []
