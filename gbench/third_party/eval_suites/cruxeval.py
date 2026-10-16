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

# Evaluation Suite: cruxeval
# Description: CRUXEval (MIT/Meta Code Reasoning & I/O Execution Simulation Benchmark)

"""gbench native built-in runner for cruxeval (Code Reasoning & Execution).

CRUXEval (Gu et al., 2024) has two tasks over the same 800 short Python functions: CRUXEval-O
(given f and an input, state the output) and CRUXEval-I (given f and an output, state an input
that produces it). This suite runs BOTH, following facebookresearch/cruxeval at commit
190faf16:
  * prompts: `prompts.py` verbatim - make_direct_output_prompt / make_direct_input_prompt (the
    2-shot direct prompts) for a no-think run, make_cot_output_prompt / make_cot_input_prompt
    (the +CoT prompts) with `--thinking` - sent with the system prompt of the repo's chat-model
    path (`openai/openai_prompt.py`) and, for a direct run, openai_run.py's `[/ANSWER]` stop
    sequence (a thinking run is sent no stop string, since it also matches inside the reasoning
    channel);
  * max_tokens: openai_run.py's caps, 100 for the direct prompts and 1000 for the CoT prompts,
    are the official cap of each mode (base.OfficialSampling.max_tokens) and base's token-limit
    policy applies them: a direct (no-think) request is sent `max_tokens=100` exactly
    (`--max-output-tokens` is not applied, see `max_output_tokens_note`); a thinking (CoT)
    request is sent the run's budget (`--max-output-tokens`, else the thinking floor), since
    the 1000 was written for non-reasoning CoT and the reasoning shares the budget, and the
    result records that the official cap was not applied to reasoning;
  * extraction: openai_prompt.py's extract_answer_{direct,cot}_{output,input}, applied to the
    reply up to the first `[/ANSWER]` (what the stop sequence leaves);
  * scoring: evaluation/utils_general.evaluate_score - run the function source followed by
    the line `assert <gold output> == <answer>` with upstream's executor (utils_execute
    unsafe_execute, reliability_guard, a 3 s limit); an output answer that contains
    `f(<input>)`, or an input answer without `f(`, does not count. The run happens in the gbench
    sandbox (bubblewrap), and the pass/fail verdict comes from a grader process that holds the
    gold output and performs the comparison, never from the program's exit status (see the
    grading section below);
  * metric: pass@1 per task, `c / n` averaged over problems (evaluate_generations.py), with the
    upstream rule that excluded generations leave n unless none passes.

Sampling: the published protocol (README and openai/openai_run.py, for the direct and the CoT
prompts alike), 10 samples per problem, pass@1 from a pass at temperature 0.2 and pass@5 from a
second pass at temperature 0.8 (base.OfficialSampling; the result records `official_sampling`).
Both passes run with and without `--thinking`, and both sample at gbench's temperature for every
suite (base.resolve_temperature): 1.0 with --thinking, 0.0 without, so neither column is the
published one. `--attempt-count K` above 10 raises the sample count. `--temperature`, or for this
suite alone `GBENCH_CRUXEVAL_TEMPERATURE` (which takes precedence over both), overrides the
temperature of both passes in either mode. This suite has no LLM judge.

Sharding: `--shard I/N` selects shard I of the 1,600 items (round-robin in dataset order, the
CRUXEval-O and CRUXEval-I item of each function side by side) before the `--eval-limit` draw.

The per-task pass@1 values are CRUXEval's leaderboard columns and carry their own
`leaderboard_comparable` flag in `cruxeval_report`. The headline is the mean of CRUXEval-I and
CRUXEval-O pass@1; upstream publishes the two separately, so the headline is flagged
`leaderboard_comparable=False`.
"""

import asyncio
import concurrent.futures
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
from .arc_agi import _for_base   # lays out an already-sharded sample list for run_eval_suite
from .base import OfficialSampling, apply_shard, run_eval_suite, strip_thinking_tags
from .limit_plan import eval_limit_plan
from .lcb import _GRADER_MAIN   # the grader/child process split lcb.py uses for code execution
from .sampling import stratified_sample
from .sandbox import run_sandboxed, sandbox_skip_reason
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Code Reasoning & Execution"
DOCS_URL = "docs/evals/cruxeval.md"

_DATASET_ID = "cruxeval-org/cruxeval"
#: HF dataset revision the loader reads (test split, 800 functions).
_DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"

#: openai/openai_prompt.py: the system message of the repo's chat-model path.
_SYSTEM_PROMPT = ("You are an expert at Python programming, code execution, test case "
                  "generation, and fuzzing.")
_STOP = "[/ANSWER]"
#: utils_general.evaluate_score -> check_correctness(code_to_execute, 3).
_EXEC_TIMEOUT_S = 3
#: The protocol behind published results: openai_run.py `n=10`, pass@1 at temperature 0.2 and
#: pass@5 at temperature 0.8 (README: "temperatures 0.2, 0.8 with n_sample=10").
_UPSTREAM_SAMPLES = 10
_UPSTREAM_TEMPERATURE = 0.2
_UPSTREAM_PASS5_TEMPERATURE = 0.8
#: openai_run.py `max_tokens`: 100 for the direct prompts, 1000 for the CoT prompts.
_UPSTREAM_DIRECT_MAX_TOKENS = 100
_UPSTREAM_COT_MAX_TOKENS = 1000
_TASKS = (("output", "output_prediction", "CRUXEval-O"), ("input", "input_prediction", "CRUXEval-I"))


# --- facebookresearch/cruxeval prompts.py (commit 190faf16), verbatim -------------------------
# The f-strings are split per line only to keep the source readable; the rendered text is
# identical to upstream's (tests pin it).
def make_direct_output_prompt(s):
    code, input = s
    return (
        'You are given a Python function and an assertion containing an input to the '
        'function. Complete the assertion with a literal (no unsimplified expressions, no '
        'function calls) containing the output when executing the provided code on the given '
        'input, even if the function is incorrect or incomplete. Do NOT output any extra '
        'information. Provide the full assertion with the correct output in [ANSWER] and '
        '[/ANSWER] tags, following the examples.\n'
        '\n'
        '[PYTHON]\n'
        'def f(n):\n'
        '    return n\n'
        'assert f(17) == ??\n'
        '[/PYTHON]\n'
        '[ANSWER]\n'
        'assert f(17) == 17\n'
        '[/ANSWER]\n'
        '\n'
        '[PYTHON]\n'
        'def f(s):\n'
        '    return s + "a"\n'
        'assert f("x9j") == ??\n'
        '[/PYTHON]\n'
        '[ANSWER]\n'
        'assert f("x9j") == "x9ja"\n'
        '[/ANSWER]\n'
        '\n'
        '[PYTHON]\n'
        f'{code}\n'
        f'assert f({input}) == ??\n'
        '[/PYTHON]\n'
        '[ANSWER]\n'
    )


def make_cot_output_prompt(s):
    code, input = s
    return (
        'You are given a Python function and an assertion containing an input to the '
        'function. Complete the assertion with a literal (no unsimplified expressions, no '
        'function calls) containing the output when executing the provided code on the given '
        'input, even if the function is incorrect or incomplete. Do NOT output any extra '
        'information. Execute the program step by step before arriving at an answer, and '
        'provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, '
        'following the examples.\n'
        '\n'
        '[PYTHON]\n'
        'def f(s):\n'
        '    s = s + s\n'
        '    return "b" + s + "a"\n'
        'assert f("hi") == ??\n'
        '[/PYTHON]\n'
        '[THOUGHT]\n'
        "Let's execute the code step by step:\n"
        '\n'
        '1. The function f is defined, which takes a single argument s.\n'
        '2. The function is called with the argument "hi", so within the function, s is '
        'initially "hi".\n'
        '3. Inside the function, s is concatenated with itself, so s becomes "hihi".\n'
        '4. The function then returns a new string that starts with "b", followed by the '
        'value of s (which is now "hihi"), and ends with "a".\n'
        '5. The return value of the function is therefore "bhihia".\n'
        '[/THOUGHT]\n'
        '[ANSWER]\n'
        'assert f("hi") == "bhihia"\n'
        '[/ANSWER]\n'
        '\n'
        '[PYTHON]\n'
        f'{code}\n'
        f'assert f({input}) == ??\n'
        '[/PYTHON]\n'
        '[THOUGHT]\n'
    )


def make_direct_input_prompt(s):
    code, output = s
    return (
        'You will be given a function f and an output in the form f(??) == output. Find any '
        'input such that executing f on the input leads to the given output. There may be '
        'multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, '
        'complete the assertion with one such input that will produce the output when '
        'executing the function.\n'
        '\n'
        '[PYTHON]\n'
        'def f(my_list):\n'
        '    count = 0\n'
        '    for i in my_list:\n'
        '        if len(i) % 2 == 0:\n'
        '            count += 1\n'
        '    return count\n'
        'assert f(??) == 3\n'
        '[/PYTHON]\n'
        '[ANSWER]\n'
        'assert f(["mq", "px", "zy"]) == 3\n'
        '[/ANSWER]\n'
        '\n'
        '[PYTHON]\n'
        'def f(s1, s2):\n'
        '    return s1 + s2\n'
        'assert f(??) == "banana"\n'
        '[/PYTHON]\n'
        '[ANSWER]\n'
        'assert f("ba", "nana") == "banana"\n'
        '[/ANSWER]\n'
        '\n'
        '[PYTHON]\n'
        f'{code}\n'
        f'assert f(??) == {output}\n'
        '[/PYTHON]\n'
        '[ANSWER]\n'
    )


def make_cot_input_prompt(s):
    code, output = s
    return (
        'You will be given a function f and an output in the form f(??) == output. Your task '
        'is to find any input such that executing f on the input leads to the given output. '
        'There may be multiple answers, but only output one. First, think step by step. You '
        'MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a '
        'passing assertion containing the input and the given output.\n'
        '\n'
        '[PYTHON]\n'
        'def f(x):\n'
        '    return x + 1\n'
        'assert f(??) == 17\n'
        '[/PYTHON]\n'
        '[THOUGHT]\n'
        'To find an input such that executing f on the input leads to the given output, we '
        'can work backwards from the given assertion. We know that f(??) == 17. \n'
        '\n'
        'Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? '
        'should be 16. \n'
        '[/THOUGHT]\n'
        '[ANSWER]\n'
        'assert f(16) == 17\n'
        '[/ANSWER]\n'
        '\n'
        '[PYTHON]\n'
        f'{code}\n'
        f'assert f(??) == {output}\n'
        '[/PYTHON]\n'
        '[THOUGHT]\n'
    )


# --- facebookresearch/cruxeval openai/openai_prompt.py extractors, verbatim --------------------
def extract_answer_direct_output(gen):
    if "==" in gen:
        gen = gen.split("==")[1]
    return gen.strip()


def extract_answer_direct_input(gen):
    if "==" in gen:
        gen = gen.split("==")[0].strip()
    if "assert f" in gen:
        gen = "f" + gen.split("assert f")[1].strip()
    return gen.strip()


def extract_answer_cot_input(gen):
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[0]
        if "assert f" in gen:
            gen = "f" + gen.split("assert f")[1].strip()
        return gen.strip()
    else:
        return gen.split('\n')[-1].strip()


def extract_answer_cot_output(gen):
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[1]
        return gen.strip()
    else:
        return gen.split('\n')[-1].strip()


_PROMPTS = {("output", False): make_direct_output_prompt, ("output", True): make_cot_output_prompt,
            ("input", False): make_direct_input_prompt, ("input", True): make_cot_input_prompt}
_EXTRACTORS = {("output", False): extract_answer_direct_output,
               ("output", True): extract_answer_cot_output,
               ("input", False): extract_answer_direct_input,
               ("input", True): extract_answer_cot_input}


def _load_cruxeval_samples(
    limit: Optional[int] = None,
    enable_thinking: bool = False,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]]:
    """CRUXEval-O and CRUXEval-I samples from HF Hub (pinned); raises on load/schema failure."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET_ID, split="test", revision=_DATASET_REVISION)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for cruxeval: {e}")
        raise RuntimeError(f"Could not load dataset for cruxeval: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for cruxeval returned empty rows")

    samples = []
    for item in rows:
        code, inp, out = item.get("code"), item.get("input"), item.get("output")
        if code is None or inp is None or out is None:
            raise RuntimeError(
                "cruxeval: unexpected dataset schema (missing 'code'/'input'/'output'); "
                "refusing to fabricate sample data"
            )
        for mode, category, _ in _TASKS:
            prompt = _PROMPTS[(mode, enable_thinking)]((code, inp if mode == "output" else out))
            messages = [{"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}]
            gold = {"id": item.get("id"), "code": code, "input": inp, "output": out,
                    "mode": mode, "cot": enable_thinking}
            meta: Dict[str, Any] = {"category": category}
            if not enable_thinking:
                # A stop string also matches inside the reasoning channel, which would end a
                # thinking turn early; a thinking run is cut at the tag after the fact instead.
                # The max_tokens of each mode is _official(...).max_tokens, applied by base.
                meta["stop"] = [_STOP]
            samples.append((messages, gold, meta))

    # --shard first, over the full ordered item list, then the --eval-limit draw.
    samples, _ = apply_shard(samples, "cruxeval")
    # Stratified over the two tasks, not a contiguous head.
    samples = stratified_sample(samples, limit, lambda s: s[2]["category"], seed="cruxeval")
    logger.info(f"Loaded {len(samples)} cruxeval samples (CRUXEval-O + CRUXEval-I).")
    return samples


# --- grading ----------------------------------------------------------------------------------
#
# Upstream executes `<code>\nassert <gold output> == <answer>` and counts the sample as passed
# when check_correctness's worker records "passed", i.e. when exec() returned. gbench runs the
# same program with upstream's own executor (utils_execute.unsafe_execute and reliability_guard,
# vendored below), inside the sandbox, in the grader/child process split that lcb.py uses for
# its code-execution scenario (lcb._GRADER_MAIN):
#
# * the child process runs the program; the gold output is not in it. The program runs as
#   `<code>\nassert _gbench_expected_ == <answer>`, where `_gbench_expected_` is a stand-in
#   whose __eq__ sends the value it is compared with to the grader and returns the grader's
#   answer;
# * the grader process, which never runs the program, holds the gold output and compares
#   (`gold == value`, Python equality as in upstream's assert);
# * the sample passes only when exactly one such comparison happened, the grader found it
#   true, and the program then finished (upstream's "passed"). The verdict is written by the
#   grader with a per-call nonce; the exit status of the child plays no part in it.
#
# An answer that exits the interpreter, raises SystemExit or kills its own process therefore
# fails, as it does upstream (reliability_guard sets exit/quit to None, and a worker that dies
# records no "passed"). Values cross from the child to the grader as plain data (None, bool,
# int, float, complex, str, bytes, bytearray, list, tuple, dict, set, frozenset, Decimal,
# Fraction); an object of any other type never equals the gold output. Every CRUXEval gold
# output is plain data of these types.


# facebookresearch/cruxeval evaluation/utils_execute.py at commit 190faf16, verbatim (git blob
# 9eec77cd6e6ad80d32ab4f452a4907f3f4446f80; the file has no trailing newline). Only
# unsafe_execute (and the helpers it calls) runs here; check_correctness's multiprocessing
# wrapper is replaced by the grader process.
_UTILS_EXECUTE = r'''# Copyright (c) Meta Platforms, Inc. and affiliates.

# This code is adapted from OpenAI's release
# https://github.com/openai/human-eval/blob/master/human_eval/execution.py

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile


def check_correctness(check_program, timeout=3):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result[0] == "passed"


def unsafe_execute(check_program, result, timeout):

    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Run program.
        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None'''

#: Runs in the child, in the vendored module's namespace (so unsafe_execute is in scope).
_CHILD_SOURCE = r'''
import builtins as _gb_builtins
import decimal as _gb_decimal
import fractions as _gb_fractions


def _gb_encode(x, depth=0):
    """A value, in a form the grader rebuilds as the same Python data type."""
    if depth > 200:
        return ["o", "too deep"]
    if x is None:
        return ["N"]
    if isinstance(x, bool):
        return ["b", bool.__bool__(x)]
    if isinstance(x, int):
        v = int.__int__(x)
        return ["i", ("-" if v < 0 else "") + format(abs(v), "x")]
    if isinstance(x, float):
        return ["f", float.hex(float.__float__(x))]
    if isinstance(x, complex):
        return ["c", float.hex(complex.__getattribute__(x, "real")),
                float.hex(complex.__getattribute__(x, "imag"))]
    if isinstance(x, _gb_decimal.Decimal):
        return ["d", _gb_decimal.Decimal.__str__(x)]
    if isinstance(x, _gb_fractions.Fraction):
        return ["q", _gb_encode(x.numerator, depth + 1), _gb_encode(x.denominator, depth + 1)]
    if isinstance(x, str):
        return ["s", str.__str__(x)]
    if isinstance(x, bytes):
        return ["y", bytes.hex(x)]
    if isinstance(x, bytearray):
        return ["Y", bytearray.hex(x)]
    if isinstance(x, list):
        return ["l", [_gb_encode(e, depth + 1) for e in list.__iter__(x)]]
    if isinstance(x, tuple):
        return ["t", [_gb_encode(e, depth + 1) for e in tuple.__iter__(x)]]
    if isinstance(x, dict):
        return ["m", [[_gb_encode(k, depth + 1), _gb_encode(v, depth + 1)]
                      for k, v in dict.items(x)]]
    if isinstance(x, frozenset):
        return ["F", [_gb_encode(e, depth + 1) for e in frozenset.__iter__(x)]]
    if isinstance(x, set):
        return ["S", [_gb_encode(e, depth + 1) for e in set.__iter__(x)]]
    return ["o", type(x).__name__]


class _GbExpected:
    """Stands in for the gold output in `assert <gold output> == <answer>`."""
    __hash__ = None

    def __init__(self, ask):
        self._ask = ask

    def __eq__(self, other):
        answer = self._ask(_gb_encode(other))
        if answer == b"T":
            return True
        if answer == b"F":
            return False
        raise ValueError("the comparison raised in the grader")

    def __repr__(self):
        return "<gold output>"


def gbench_child(p, send, ask):
    _gb_builtins._gbench_expected_ = _GbExpected(ask)
    result = []
    unsafe_execute(p["program"], result, p["timeout"])
    send({"completed": bool(result) and result[0] == "passed",
          "detail": str(result[:1])[:200]})
'''

#: Runs in the grader process: rebuilds the value the program compared and compares it with
#: the gold output, which only this process holds.
_GRADER_SOURCE = r'''
import decimal as _gb_decimal
import fractions as _gb_fractions


class _GbOpaque:
    """A value that is not plain data: equal only to itself."""
    __hash__ = object.__hash__

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<%s>" % self.name


def _gb_decode(v, depth=0):
    if depth > 200 or not isinstance(v, list) or not v:
        raise ValueError("bad value")
    tag = v[0]
    if tag == "N":
        return None
    if tag == "b":
        return v[1] is True
    if tag == "i":
        s = str(v[1])
        return -int(s[1:], 16) if s.startswith("-") else int(s, 16)
    if tag == "f":
        return float.fromhex(v[1])
    if tag == "c":
        return complex(float.fromhex(v[1]), float.fromhex(v[2]))
    if tag == "d":
        return _gb_decimal.Decimal(str(v[1]))
    if tag == "q":
        return _gb_fractions.Fraction(_gb_decode(v[1], depth + 1), _gb_decode(v[2], depth + 1))
    if tag == "s":
        if not isinstance(v[1], str):
            raise ValueError("bad str")
        return v[1]
    if tag == "y":
        return bytes.fromhex(v[1])
    if tag == "Y":
        return bytearray.fromhex(v[1])
    if tag == "l":
        return [_gb_decode(e, depth + 1) for e in v[1]]
    if tag == "t":
        return tuple(_gb_decode(e, depth + 1) for e in v[1])
    if tag == "m":
        return {_gb_decode(k, depth + 1): _gb_decode(x, depth + 1) for k, x in v[1]}
    if tag == "F":
        return frozenset(_gb_decode(e, depth + 1) for e in v[1])
    if tag == "S":
        return set(_gb_decode(e, depth + 1) for e in v[1])
    if tag == "o":
        return _GbOpaque(str(v[1])[:80])
    raise ValueError("bad tag")


class _GbJudge:
    def __init__(self, p, q):
        self.records = []
        self.eq_answers = []
        self.expected_value = None
        self.expected_error = None
        try:
            self.expected_value = eval(q["expected"], {})
        except BaseException as e:
            self.expected_error = repr(e)

    def record(self, msg):
        if isinstance(msg, dict):
            self.records.append(msg)

    def answer(self, value):
        # `assert <gold output> == <answer>`, evaluated here; only the first comparison counts
        if self.eq_answers or self.expected_error is not None:
            self.eq_answers.append(b"E")
            return b"E"
        try:
            ok = bool(self.expected_value == _gb_decode(value))
            ans = b"T" if ok else b"F"
        except Exception:
            ans = b"E"
        self.eq_answers.append(ans)
        return ans

    def verdict(self):
        done = next((r for r in self.records if "completed" in r), None)
        ok = (done is not None and done.get("completed") is True
              and self.eq_answers == [b"T"])
        return ok, (done.get("detail", "") if done is not None else "no report")


def gbench_judge(p, q):
    return _GbJudge(p, q)
'''

#: The grader's wall-clock limit on top of the program's own 3 s: interpreter start-up and the
#: numpy/pandas import the shared grader performs.
_GRADER_STARTUP_S = 20


def _run_program(program: str, gold_output: str,
                 timeout: float = _EXEC_TIMEOUT_S) -> Tuple[bool, str]:
    """Run `program` (which compares `_gbench_expected_` with the answer) in the sandbox:
    (passed, diagnostic). Anything but a pass verdict carrying this call's nonce is a fail."""
    import json
    import secrets
    public = {"module": "cruxeval_utils_execute", "source": _UTILS_EXECUTE,
              "child_source": _CHILD_SOURCE, "grader_source": _GRADER_SOURCE,
              "program": program, "timeout": timeout}
    # ensure_ascii (the default): a part's length in characters is its length in bytes.
    body, secret = json.dumps(public), json.dumps({"expected": gold_output})
    nonce = secrets.token_hex(16)
    # One BLAS thread: the grader imports numpy, and a BLAS pool per check costs seconds of CPU
    # on a many-core host. reliability_guard sets OMP_NUM_THREADS=1 for the program as well.
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    try:
        r = run_sandboxed([sys.executable, "-c", _GRADER_MAIN],
                          input=f"{len(body)}\n{body}{len(secret)}\n{secret}{nonce}",
                          capture_output=True, text=True, env=env,
                          timeout=timeout + 1 + _GRADER_STARTUP_S)
    except subprocess.TimeoutExpired:
        return False, f"no verdict within {timeout + 1 + _GRADER_STARTUP_S:.0f}s"
    except Exception as e:                                              # noqa: BLE001
        return False, repr(e)
    for line in reversed((r.stdout or "").splitlines()):
        try:
            verdict = json.loads(line)
        except ValueError:
            continue
        if isinstance(verdict, dict) and verdict.get("nonce") == nonce:
            return verdict.get("passed") is True, str(verdict.get("detail") or "")
    return False, (r.stderr or "")[-500:]


def _grader_self_check() -> Optional[str]:
    """Why the checks cannot run here, or None.

    Every failed check scores a sample as wrong, so a broken grading environment (sandbox,
    numpy/pandas missing for the grader) would read as a model scoring 0%. Check known answers
    first: three correct ones must pass and a wrong one must fail."""
    plus_one = {"code": "def f(x):\n    return x + 1", "input": "1", "output": "2"}
    encode = {"code": "def f(s):\n    return s.encode()", "input": "'a'", "output": "b'a'"}
    cases = [
        ({**plus_one, "mode": "output"}, "assert f(1) == 2", "passed"),
        ({**plus_one, "mode": "input"}, "assert f(1) == 2", "passed"),
        ({**encode, "mode": "output"}, "assert f('a') == b'a'", "passed"),
        ({**plus_one, "mode": "output"}, "assert f(1) == 3", "failed"),
    ]
    for gold, reply, want in cases:
        if _check(reply, {**gold, "id": "self-check", "cot": False}) != want:
            return ("the CRUXEval execution check gave the wrong verdict on a known answer, so "
                    "every sample would be mis-scored (check bubblewrap, that numpy and pandas "
                    f"import in {sys.executable}, and that the grader can call prctl and fork)")
    return None


def _check(response_text: str, gold_target: Any) -> str:
    """The verdict for one reply: passed, failed, or excluded (evaluate_score skips it).

    The suite always scores dict golds built by _load_cruxeval_samples (function, input,
    output, task). A bare string gold is a literal-check helper only, never used by a run:
    it is taken as an expected output literal and checked as a direct-mode CRUXEval-O answer
    with no function and no input, so the answer compared with the gold is all that runs."""
    if isinstance(gold_target, dict):
        code, inp, out = gold_target["code"], gold_target["input"], gold_target["output"]
        mode, cot = gold_target.get("mode", "output"), bool(gold_target.get("cot"))
    else:
        code, inp, out, mode, cot = "", None, str(gold_target).strip(), "output", False
    if not response_text:
        return "failed"
    gen = response_text.split(_STOP, 1)[0]
    answer = _EXTRACTORS[(mode, cot)](gen)
    if mode == "input" and "f(" not in answer:
        return "excluded"
    if mode == "output" and inp is not None and f"f({inp})" in answer:
        return "excluded"
    # evaluate_score runs f"{c}\nassert {o} == {g}"; the gold output `o` stays with the grader
    # and a stand-in takes its place.
    passed, _ = _run_program(f"{code}\nassert _gbench_expected_ == {answer}", out)
    return "passed" if passed else "failed"


def _eval_cruxeval(response_text: str, gold_target: Any) -> bool:
    """Execution check of one reply (evaluate_score for a single generation)."""
    return _check(response_text, gold_target) == "passed"


async def _async_score_cruxeval(sample_traces: List[Dict[str, Any]]) -> None:
    """Run the execution checks after generation, in parallel (each is a subprocess)."""
    loop = asyncio.get_running_loop()
    pending = [t for t in sample_traces if t.get("status") == "PENDING_JUDGE"]
    workers = min(32, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = await asyncio.gather(*[
            loop.run_in_executor(pool, _check, strip_thinking_tags(t.get("response_text")),
                                 t.get("gold_answer"))
            for t in pending])
    for t, verdict in zip(pending, verdicts):
        t["cruxeval_check"] = verdict
        t["is_correct"] = verdict == "passed"
        t["status"] = "TRUNCATED" if t.get("health") in ("truncated", "looping") else "OK"
    for t in sample_traces:
        t.setdefault("cruxeval_check", "failed")


def _pass_at_k(n: int, c: int, k: int) -> float:
    """evaluation/utils_general.pass_at_k."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def _cruxeval_report(sample_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-task pass@1 (and pass@5 with >= 5 samples) as evaluate_generations.py computes them.

    evaluate_score drops excluded generations from a problem's list, then replaces the list
    with all-False when nothing passed; so n shrinks only for problems with a passing sample.
    A generation whose request got no reply from the endpoint is a harness error (base marks it
    scoring_excluded): it is left out of its problem's list, and a problem with no generation
    left is not counted."""
    from .base import is_scored
    report: Dict[str, Any] = {}
    for mode, category, label in _TASKS:
        problems: Dict[Any, List[str]] = {}
        for t in sample_traces:
            if not is_scored(t):
                continue
            gold = t.get("gold_answer") or {}
            if not isinstance(gold, dict) or gold.get("mode") != mode:
                continue
            problems.setdefault(t.get("source_sample_idx", t.get("sample_idx")), []).append(
                t.get("cruxeval_check") or "failed")
        if not problems:
            continue
        p1, p5, samples = [], [], []
        for checks in problems.values():
            c = checks.count("passed")
            n = sum(1 for v in checks if v != "excluded") if c else len(checks)
            samples.append(len(checks))
            p1.append(_pass_at_k(n, c, 1))
            if len(checks) >= 5:
                p5.append(_pass_at_k(n, c, 5))
        entry = {"problems": len(problems), "samples_per_problem": max(samples),
                 "pass_at_1": round(sum(p1) / len(p1) * 100.0, 2),
                 "excluded_generations": sum(ch.count("excluded") for ch in problems.values())}
        if p5 and len(p5) == len(p1):
            entry["pass_at_5"] = round(sum(p5) / len(p5) * 100.0, 2)
        report[label] = entry
    return report


def _long_direct_replies(sample_traces: List[Dict[str, Any]]) -> int:
    """Direct-prompt replies longer than openai_run.py's 100-token cap. A direct request is sent
    that cap, so this is nonzero only for a request whose cap was changed on the way (base
    sends the context-clamped value when the prompt plus the cap does not fit the window)."""
    return sum(1 for t in sample_traces
               if not (t.get("gold_answer") or {}).get("cot")
               and int(t.get("completion_tokens") or 0) > _UPSTREAM_DIRECT_MAX_TOKENS)


def _direct_caps_changed(sample_traces: List[Dict[str, Any]]) -> List[Any]:
    """The max_tokens values other than openai_run.py's 100 that direct requests were sent."""
    return sorted({t.get("max_tokens_sent") for t in sample_traces
                   if not (t.get("gold_answer") or {}).get("cot")
                   and t.get("max_tokens_sent") not in (None, _UPSTREAM_DIRECT_MAX_TOKENS)})


def _official(temperature: float, which: str, cot: bool = False) -> OfficialSampling:
    """The published sampling of one pass (README and openai/openai_run.py at 190faf16), with
    openai_run.py's max_tokens for the prompts the run sends: 100 for the direct prompts (a
    no-think run), 1000 for the CoT prompts (a thinking run). base's token-limit policy sends
    the 100 exactly and does not apply the 1000 to a thinking run (it was written for
    non-reasoning CoT)."""
    cap = _UPSTREAM_COT_MAX_TOKENS if cot else _UPSTREAM_DIRECT_MAX_TOKENS
    return OfficialSampling(
        reference=(f"facebookresearch/cruxeval@190faf16 README + openai/openai_run.py "
                   f"(n=10, temperatures 0.2 and 0.8; max_tokens {cap} for the "
                   f"{'CoT' if cot else 'direct'} prompts): {which} at {temperature}"),
        temperature=temperature, max_tokens=cap)


def _off(value: Any, official: float) -> bool:
    return value is None or abs(float(value) - official) > 1e-9


def run_cruxeval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute cruxeval native built-in evaluation suite (CRUXEval-I + CRUXEval-O).

    The published protocol: 10 samples per problem, pass@1 at temperature 0.2 and pass@5 at
    temperature 0.8 (README; openai_run.py `n=10`, temperatures 0.2 and 0.8, for the direct and
    the CoT prompts alike). Both passes run with and without thinking, both at gbench's
    temperature (1.0 with thinking, 0.0 without) unless the operator sets one."""
    # Scoring executes the dataset function with the model's answer: require isolation.
    # Missing/blocked bubblewrap is an external prerequisite -> hard error (no-skip);
    # GBENCH_SANDBOX=none opts into unsandboxed execution.
    blocked = sandbox_skip_reason()
    if blocked:
        raise infra_required("cruxeval", blocked, DOCS_URL)
    broken = _grader_self_check()
    if broken:
        raise infra_required("cruxeval", broken, DOCS_URL)
    limit = kwargs.get("limit")
    samples = _load_cruxeval_samples(limit=limit, enable_thinking=enable_thinking)
    n = max(_UPSTREAM_SAMPLES, int(kwargs.get("attempt_count") or 1))
    # --eval-limit N is N items for the whole suite, spread over CRUXEval-O and CRUXEval-I (the
    # loader's stratified draw), never N per task. The pass@1 and pass@5 runs send the same
    # items, n generations each: 2n attempts per item, not more items.
    per_task = {label: 0 for _, _, label in _TASKS}
    by_category = {category: label for _, category, label in _TASKS}
    for sample in samples:
        meta = sample[2] if isinstance(sample, tuple) and len(sample) > 2 else None
        category = meta.get("category") if isinstance(meta, dict) else None
        label = by_category.get(category, str(category))
        per_task[label] = per_task.get(label, 0) + 1
    limit_plan = eval_limit_plan("cruxeval", limit, len(samples), per_task,
                                 setting_noun="task", attempts=2 * n)
    common = dict(
        eval_name="cruxeval",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=_for_base(samples),  # already this shard's items; base keeps exactly them
        async_eval_fn=_async_score_cruxeval,
        declared_scoring_mode="execution",
        supports_attempts=True,      # each trace is checked on its own, never keyed by id
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,                  # already drawn by the loader, after the shard
        max_output_tokens=kwargs.get("max_output_tokens"),
        attempt_count=n,
    )
    result = run_eval_suite(
        official_sampling=_official(_UPSTREAM_TEMPERATURE, "pass@1", enable_thinking), **common)
    report = _cruxeval_report(result.get("sample_traces") or [])
    temp = result.get("temperature")
    pass5_result = run_eval_suite(
        official_sampling=_official(_UPSTREAM_PASS5_TEMPERATURE, "pass@5", enable_thinking),
        **common)
    report5 = _cruxeval_report(pass5_result.get("sample_traces") or [])
    pass5_temp = pass5_result.get("temperature")
    for label, entry in report.items():
        entry.pop("pass_at_5", None)
        if label in report5 and "pass_at_5" in report5[label]:
            entry["pass_at_5"] = report5[label]["pass_at_5"]
    rep = result.get("attempts") or {}
    k = rep.get("attempts_per_sample", 1) if rep else 1
    failed5 = int(pass5_result.get("failed_requests") or 0)
    long_direct = _long_direct_replies(result.get("sample_traces") or [])
    long_direct5 = _long_direct_replies(pass5_result.get("sample_traces") or [])

    # Why a column (its pass@1 and pass@5) would differ from the published one.
    column_reasons = []
    if enable_thinking:
        column_reasons.append("--thinking run: the +CoT prompts with the model's native "
                              "reasoning on top (sent the run's budget: openai_run.py's CoT "
                              f"max_tokens={_UPSTREAM_COT_MAX_TOKENS} is not applied to "
                              "reasoning); compare with CRUXEval+CoT figures only with that "
                              "caveat")
    if _off(temp, _UPSTREAM_TEMPERATURE):
        column_reasons.append(f"pass@1 sampled at temperature {temp} "
                              f"({result.get('temperature_source')}; published pass@1 uses "
                              f"{_UPSTREAM_TEMPERATURE})")
    if _off(pass5_temp, _UPSTREAM_PASS5_TEMPERATURE):
        column_reasons.append(f"pass@5 sampled at temperature {pass5_temp} "
                              f"({pass5_result.get('temperature_source')}; published pass@5 "
                              f"uses {_UPSTREAM_PASS5_TEMPERATURE})")
    if k != _UPSTREAM_SAMPLES:
        column_reasons.append(f"{k} samples per problem (published: {_UPSTREAM_SAMPLES})")
    for which, count in (("pass@1", long_direct), ("pass@5", long_direct5)):
        if count:
            column_reasons.append(
                f"{count} direct repl(ies) in the {which} pass ran past the upstream chat "
                f"path's {_UPSTREAM_DIRECT_MAX_TOKENS}-token cap, which would have cut them off")
    for which, traces_ in (("pass@1", result.get("sample_traces") or []),
                           ("pass@5", pass5_result.get("sample_traces") or [])):
        changed = _direct_caps_changed(traces_)
        if changed:
            column_reasons.append(
                f"direct requests in the {which} pass were sent max_tokens {changed} (the prompt "
                f"plus openai_run.py's {_UPSTREAM_DIRECT_MAX_TOKENS} did not fit the context "
                "window)")
    for which, res_ in (("pass@1", result), ("pass@5", pass5_result)):
        traces_ = res_.get("sample_traces") or []
        no_reply = sum(1 for t in traces_ if t.get("harness_error"))
        empty = int(res_.get("empty_responses") or 0)
        if no_reply:
            column_reasons.append(f"{no_reply} request(s) in the {which} pass got no reply from "
                                  "the served endpoint (harness errors, left out of the score)")
        if empty:
            column_reasons.append(f"{empty} request(s) in the {which} pass came back empty and "
                                  "count as wrong")
    if limit:
        column_reasons.append(f"--eval-limit subset ({len(samples)} of 1,600 items)")
    for label, entry in report.items():
        entry["pass_at_1_temperature"] = temp
        if "pass_at_5" in entry:
            entry["pass_at_5_temperature"] = pass5_temp
        entry["leaderboard_comparable"] = not column_reasons
        if column_reasons:
            entry["leaderboard_comparable_reason"] = "; ".join(column_reasons)

    result["cruxeval_report"] = report
    result["eval_limit_plan"] = limit_plan
    # The tasks this run asked (CRUXEval-I, CRUXEval-O or both); a task whose every request got
    # no reply from the endpoint (harness errors) has no pass@1, so the mean has no value either:
    # withheld, never the other task's pass@1 under the mean's name.
    modes_run = {t["gold_answer"].get("mode") for t in result.get("sample_traces") or []
                 if isinstance(t.get("gold_answer"), dict)}
    labels_run = [label for mode, _, label in _TASKS if mode in modes_run]
    per_task = [report[label]["pass_at_1"] for label in labels_run if label in report]
    if labels_run and len(per_task) == len(labels_run):
        result["accuracy"] = round(sum(per_task) / len(per_task), 2)
    elif labels_run:
        missing = [label for label in labels_run if label not in report]
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = (
            f"no scored generation in {' and '.join(missing)} (every request got no reply from "
            "the served endpoint: harness errors), so the mean of the two pass@1 has no value: "
            + str(result.get("status_reason") or "every request left out of the score"))
    # The pass@5 run's harness errors and failures are part of the result too.
    if pass5_result.get("status") not in (None, "success"):
        cause = f"pass@5 run: {pass5_result.get('status_reason') or pass5_result.get('status')}"
        result["status_reason"] = "; ".join(filter(None, (result.get("status_reason"), cause)))
        if result.get("status") == "success":
            result["status"] = "completed_with_errors"
    result["metric"] = ("mean of CRUXEval-I and CRUXEval-O pass@1 (execution-checked); the "
                        "per-task pass@1 and pass@5 are in cruxeval_report")
    result["pass_at_5_run"] = {
        "temperature": pass5_temp,
        "temperature_source": pass5_result.get("temperature_source"),
        "official_sampling": pass5_result.get("official_sampling"),
        "status": pass5_result.get("status"),
        "failed_requests": failed5,
        "total_questions": pass5_result.get("total_questions"),
        "duration_s": pass5_result.get("duration_s"),
        "max_tokens_sent": pass5_result.get("max_tokens_sent"),
        "response_health": pass5_result.get("response_health"),
        # every T=0.8 generation with its verdict, so each pass@5 in cruxeval_report can be
        # re-derived from the artifact (_cruxeval_report over these traces)
        "sample_traces": pass5_result.get("sample_traces") or [],
    }
    reasons = ["headline is the mean of CRUXEval-I and CRUXEval-O pass@1 (upstream reports the "
               "two separately; each column's own flag is in cruxeval_report)"] + column_reasons
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
