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

"""Native LiveCodeBench algorithmic coding evaluation suite.

Follows LiveCodeBench's own runner, `lcb_runner` (github.com/LiveCodeBench/LiveCodeBench at
commit 28fef95e), for an OpenAI-compatible chat model (`LMStyle.OpenAIChat`):

* **Code generation** (the headline): `livecodebench/code_generation_lite` at a pinned
  revision and release window, the upstream system message and `### Question / ### Format /
  ### Answer` prompt, the LAST fenced block as the program (`extract_code`), and upstream's
  own grader (`testing_util.run_test`, vendored verbatim below and run inside the bubblewrap
  jail): line-wise stdout comparison with a Decimal fallback, and call-based tests through
  the `func_name` in the problem's metadata, 6 s per test.
* **Code execution** (reported separately): `livecodebench/execution-v2`, the upstream
  direct-output prompt, `extract_execution_code`, and scoring by executing
  `assert <output> == <prediction>` (utils_execute, 3 s).
* **Test output prediction** (reported separately): `livecodebench/test_generation`, the
  upstream prompt, `extract_test_output_code` and `check_testcase_output`.
* **Self-repair** (reported separately): runner/base_runner.py `run_main_repair` on this run's
  code generation samples. An accepted program is kept; every rejected one gets
  prompts/self_repair.py's prompt (the question, the program and the failure message built by
  `get_check_prompt` from the grader's failure metadata) and one new generation (`--n 1`). All
  programs are graded again with the code generation grader, and pass@1 is their mean.

All four scenarios run by default, as LiveCodeBench publishes each on its leaderboard;
`GBENCH_LCB_INCLUDE_TEST_GEN=0` leaves test output prediction out,
`GBENCH_LCB_INCLUDE_SELF_REPAIR=0` leaves self-repair out, and the result lists the scenarios
that ran (`lcb_scenarios`). Grading keeps the expected outputs away from the process that runs
the model's code (see the grading section below).

Sampling: lcb_runner's (OFFICIAL_SAMPLING): top_p 0.95 and frequency/presence penalties 0 on
every request, self-repair included, with n=10 samples per problem from which pass@1 is estimated
(a larger `--attempt-count` is honoured). The temperature is gbench's rule for every suite
(base.resolve_temperature), not lcb_runner's 0.2: 1.0 with --thinking, 0.0 without (at 0.2
Gemma's reasoning looped until the budget ended in 40 of 50 replies). `--temperature`, or
`GBENCH_LCB_TEMPERATURE` for this suite alone (it takes precedence), overrides it in either mode;
the result records lcb_runner's 0.2 under `official_sampling`, and a temperature other than 0.2
makes the run not leaderboard_comparable.

max_tokens follows base's token-limit policy (OFFICIAL_SAMPLING.max_tokens, resolved by
base.resolve_token_budget for every request, self-repair included):

* no-think: lcb_runner's max_tokens 2000 (parser.py --max_tokens default); the run's
  --max-output-tokens is not applied. Only a prompt that leaves less room in the served context
  window gets less, which the result names.
* --thinking: the 2000 cap was written for non-reasoning generation and is not applied; each
  request is sent the run's budget (--max-output-tokens, else base's thinking floor) and the
  result records that the official cap was not applied to reasoning. lcb_runner has no
  reasoning-model setting for the style gbench runs: its reasoning styles (OpenAIReason,
  OpenAIReasonPreview, Claude3Thinking, GeminiThinking) are vendor-API variants with their own
  prompts, and its open reasoning models (LMStyle.DeepSeekR1, LMStyle.QwQ) run through the vllm
  runner with --max_tokens given on the command line.

The result's `max_output_tokens_note` states the mode's choice. A reply (code generation or
self-repair) that stopped at a limit other than lcb_runner's 2000 (with --thinking, the run's
budget; without it, a max_tokens the context window lowered) is counted in
`replies_cut_off_by_budget`, and those with no answer text, cut off inside the reasoning channel,
in `replies_cut_off_in_reasoning`.
`leaderboard_comparable` is True only for lcb_runner's sampling
(no other sampling field, decoding penalties included) with all four scenarios on a full,
post-v1 release, and no reply cut off by such a limit; otherwise the reason names what differs.
Grading is by execution; no LLM judge is involved, so gbench's judge temperature (pinned at 0.0)
plays no part.
"""

import ast
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from .base import (OFFICIAL_SAMPLING_PARAM_KEYS, OfficialSampling, current_shard,
                   finalize_shard_fields, is_scored, resolve_token_budget, run_eval_suite,
                   suite_env)
from .metrics import budget_cutoffs
from .limit_plan import eval_limit_plan
from .sampling import allocate_limit, limit_dataset, stratified_sample, presharded_for_base
from .sandbox import run_sandboxed, sandbox_skip_reason
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/lcb.md"

#: Which LiveCodeBench release window to evaluate. The point of the benchmark is to test
#: on problems published AFTER the model was trained, so this SHOULD be set per model;
#: `release_v1` (2023-05..2024-03) is contaminated for anything current and is not the
#: default for that reason. Window tags like `v5_v6` select only newly-added problems.
_LCB_VERSION = os.environ.get("GBENCH_LCB_VERSION", "release_v6")

#: Optional hard floor on `contest_date` (ISO, e.g. "2024-10"). The release tags are
#: cumulative - `release_v6` still contains the 2023 problems from v1 - so a tag alone
#: cannot express "only problems published after this model was trained". Setting this is
#: the precise contamination control (upstream's --start_date); a window tag like `v5_v6`
#: is the coarse one.
_LCB_MIN_DATE = os.environ.get("GBENCH_LCB_MIN_DATE", "").strip()

#: Dataset snapshots, pinned on 2026-09-24 (HF commit shas).
_LCB_REPO = "livecodebench/code_generation_lite"
_LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
_LCB_EXEC_REPO = "livecodebench/execution-v2"
_LCB_EXEC_REVISION = "ff6ea0e2a638001006ddcc41259eff23a4283fb2"
_LCB_TESTGEN_REPO = "livecodebench/test_generation"
_LCB_TESTGEN_REVISION = "6f3ac40bbecf81eba15899139d279b077f2816fd"

#: lcb_runner commit the prompts, extraction and graders were ported from.
_LCB_RUNNER_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

#: lcb_runner/runner/parser.py defaults: --n, --temperature, --top_p, --max_tokens, --timeout.
_LCB_N = 10
_LCB_TEMPERATURE = 0.2
_LCB_TOP_P = 0.95
_LCB_MAX_TOKENS = 2000
_LCB_TIMEOUT_S = 6

#: What lcb_runner's OpenAIChat runner (runner/oai_runner.py; runner/vllm_runner.py sends the same)
#: sends with those defaults: the temperature, top_p, frequency_penalty and presence_penalty of 0,
#: and max_tokens. Self-repair uses the same settings (its input file is named after them). The
#: max_tokens is an official cap for non-reasoning generation (see the module docstring).
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"github.com/LiveCodeBench/LiveCodeBench@{_LCB_RUNNER_COMMIT[:8]} "
               "lcb_runner/runner/parser.py (--temperature 0.2, --top_p 0.95, --max_tokens 2000, "
               "--n 10) and runner/oai_runner.py (frequency_penalty 0, presence_penalty 0)"),
    temperature=_LCB_TEMPERATURE,
    params={"top_p": _LCB_TOP_P, "frequency_penalty": 0, "presence_penalty": 0},
    max_tokens=_LCB_MAX_TOKENS)
#: compute_code_execution_metrics.evaluate_score: check_correctness(code, 3).
_LCB_EXEC_TIMEOUT_S = 3


#: `version_tag` -> shard files, copied from ALLOWED_FILES in `code_generation_lite.py`
#: (the dataset's loading script, LiveCodeBench/code_generation_lite).
#: Each release is cumulative; a WINDOW tag like `v4_v5` selects only the shards added
#: between two releases, which is how you evaluate strictly past a model's cutoff.
_LCB_RELEASE_FILES = {f"release_v{i}": [f"test{j}.jsonl" if j > 1 else "test.jsonl"
                                        for j in range(1, i + 1)]
                      for i in range(1, 7)}
_LCB_RELEASE_FILES["release_latest"] = _LCB_RELEASE_FILES["release_v6"]
for _i in range(1, 7):
    _LCB_RELEASE_FILES[f"v{_i}"] = [f"test{_i}.jsonl" if _i > 1 else "test.jsonl"]
for _a in range(1, 7):
    for _b in range(_a + 1, 7):
        _LCB_RELEASE_FILES[f"v{_a}_v{_b}"] = [
            f"test{_j}.jsonl" if _j > 1 else "test.jsonl" for _j in range(_a, _b + 1)]


# --- prompts (lcb_runner/prompts, LMStyle.OpenAIChat branch, verbatim) --------------------
# The prompts, the extraction helpers and the vendored graders below are LiveCodeBench code
# (MIT License, Copyright (c) 2024 LiveCodeBench); the full notice is with the vendored
# sources at the end of this file.

#: prompts/code_generation.py PromptConstants.
_SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all "
    "tests.")
_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose "
    "your code within delimiters.")
_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT.")

#: prompts/code_execution.py format_prompt_execution_base.
_EXEC_SYSTEM_MESSAGE = ("You are an expert at Python programming, code execution, test case "
                        "generation, and fuzzing.")

#: prompts/test_output_prediction.py PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC.
_TESTGEN_SYSTEM_MESSAGE = (
    "You are a helpful programming assistant and an expert Python programmer. You are helping "
    "a user to write a test case to help to check the correctness of the function. The user has "
    "written a input for the testcase. You will calculate the output of the testcase and write "
    "the whole assertion statement in the markdown code block with the correct output.")


def _generation_prompt(question_content: str, starter_code: str) -> str:
    """get_generic_question_template_answer."""
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += (
            f"### Format: {_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def _execution_prompt(code: str, input: str) -> str:
    """make_direct_output_prompt (upstream's default; --cot_code_execution is off)."""
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def repeatNumber(number : int) -> int:
    return number
assert repeatNumber(number = 17) == ??
[/PYTHON]
[ANSWER]
assert repeatNumber(number = 17) == 17
[/ANSWER]

[PYTHON]
def addCharacterA(string : str) -> str:
    return string + "a"
assert addCharacterA(string = "x9j") == ??
[/PYTHON]
[ANSWER]
assert addCharacterA(string = "x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[ANSWER]
"""


def _format_testcase_func_name_input(function_name: str, testcase: str) -> str:
    """format_testcase_func_name_input."""
    input_str = ", ".join(testcase.split("\n"))
    return f"assert {function_name}({input_str}) == # TODO"


def _parse_function_name_from_starter_code(starter_code: str) -> Optional[str]:
    """parse_function_name_from_starter_code."""
    tree = ast.parse(starter_code)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert fn is None
            fn = node.name
    return fn


def _test_output_prompt(question_content: str, starter_code: str, testcase_input: str) -> str:
    """get_generic_question_template_test_completion."""
    prompt = f"Problem:\n{question_content}"
    prompt += f"Function:\n```\n{starter_code}\n```\n"

    # parse function name from starter_code
    func_name = _parse_function_name_from_starter_code(starter_code)
    prompt += "Please complete the following test case:\n\n"
    prompt += (
        f"```\n{_format_testcase_func_name_input(func_name, testcase_input)}\n```\n"
    )
    return prompt


# --- self-repair (lcb_runner/prompts/self_repair.py, LMStyle.OpenAIChat branch, verbatim) ----

#: self_repair.py PromptConstants.
_SR_SYSTEM_MESSAGE_GENERIC = (
    "You are a helpful programming assistant and an expert Python programmer. You are helping "
    "a user write a program to solve a problem. The user has written some code, but it has some "
    "errors and is not passing the tests. You will help the user by first giving a concise (at "
    "most 2-3 sentences) textual explanation of what is wrong with the code. After you have "
    "pointed out what is wrong with the code, you will then generate a fixed version of the "
    "program. You must put the entired fixed program within code delimiters only for once.")
_SR_FORMATTING_REPEAT = (
    "First reason about the code providing a textual explanation of what is wrong with the code "
    "and then generate a fixed of the program enclosed code delimiters.")
_SR_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows.")


def _get_check_prompt(question: str, result, metadata):
    """get_check_prompt."""
    ## assumes i/o examples are already truncated!
    ## less pressure on storing 10 MB json because on a single large input-output pair
    # result_by_test_case = result
    # assert len(metadata) == 1, f"metadata = {metadata}"
    # metadata = metadata[0]
    metadata = json.loads(metadata)
    if "error_code" not in metadata:
        return ""
    if metadata["error_code"] == -1:
        # time limit exceeded
        message = f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"
    elif metadata["error_code"] == -2:
        # wrong answer
        message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata['inputs']}\nGenerated Output: {metadata['output']}\nExpected: {metadata['expected']}"
    elif metadata["error_code"] == -3:
        # time limit exceeded
        message = f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}"
        pass
    elif metadata["error_code"] == -4:
        # runtime error
        message = f"The above code is incorrect and got a runtime error.\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n{metadata['error']}"
    else:
        raise NotImplementedError(
            f"metadata['error_code'] = {metadata['error_code']} not implemented || {metadata=}"
        )
    return message


def _get_generic_question_template_answer(question: str, code, result, metadata):
    """get_generic_question_template_answer (self_repair.py)."""
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Answer:\n```python\n{code}\n```\n\n"
    prompt += _get_check_prompt(question, result, metadata) + "\n"
    prompt += f"### Format: {_SR_FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def _self_repair_prompt(question: str, code: str, result: bool, metadata: str):
    """format_prompt_self_repair for LMStyle.OpenAIChat: "" for an accepted program, else the
    system + user chat messages."""
    if result:
        # The code is accepted, no need to change anything.
        return ""
    chat_messages = [
        {"role": "system", "content": _SR_SYSTEM_MESSAGE_GENERIC},
    ]
    chat_messages += [
        {
            "role": "user",
            "content": _get_generic_question_template_answer(
                question, code, result, metadata
            )
            + "\n\n"
            + _SR_FORMATTING_REPEAT,
        },
    ]
    return chat_messages


def _self_repair_metadata(meta: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """(metadata JSON for _get_check_prompt, which failure message it yields).

    The failure metadata is run_test's at the pinned commit. Two of its shapes have no message in
    get_check_prompt, which was written for the grader LiveCodeBench had when self-repair was
    added (commit 8eac093f): a program that fails before its first test (it does not compile, or
    its entry point is missing) comes back as -4 without the test fields the -4 message reads,
    and a grading run that returned nothing comes back as -5. The earlier grader reported the
    first as a compilation error (-1, with the error text) and the second as no metadata ({},
    no failure message), and that is what they are given here."""
    kinds = {-1: "compilation_error", -2: "wrong_answer", -3: "time_limit_exceeded",
             -4: "runtime_error"}
    code = meta.get("error_code") if isinstance(meta, dict) else None
    if code not in kinds:
        return json.dumps({}), "no_failure_message"
    if code == -4 and "inputs" not in meta:
        return (json.dumps({"error_code": -1, "error": meta.get("error_message", "")}),
                "compilation_error")
    return json.dumps(meta), kinds[code]


# --- extraction (lcb_runner/utils/extraction_utils.py, verbatim for OpenAIChat) -----------

def _extract_code(model_output: str) -> str:
    """extract_code: the LAST fenced block, or "" when there is none."""
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def _extract_execution_code(model_output: str, cot: bool = False) -> str:
    """extract_execution_code."""
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()


def _extract_test_output_code(model_output: str) -> str:
    """extract_test_output_code."""
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    # first try to extract ```python if not then try ```
    indexlines = [
        i
        for i, line in enumerate(outputlines)
        if "```python" in line or "```Python" in line
    ]
    if indexlines:
        start_index = indexlines[0]
    else:
        start_index = None
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if start_index is not None:
        indexlines = [i for i in indexlines if i > start_index]
        indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


# --- loading ------------------------------------------------------------------------------

def _load_lcb_lite(version_tag: str):
    """Rows for a LiveCodeBench release window, read straight from the repo's JSONL shards.

    `load_dataset(..., version_tag=...)` went through the repo's loading script, and
    `datasets` 5.x refuses to execute dataset scripts at all ("Dataset scripts are no longer
    supported"). The shard-per-release layout is stable and documented on the card, so the
    mapping is replicated here and the files are fetched directly, from the pinned revision.
    This keeps the one property that matters: being able to name a window past the model's
    training cutoff.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    files = _LCB_RELEASE_FILES.get(version_tag)
    if not files:
        raise RuntimeError(
            f"lcb: unknown GBENCH_LCB_VERSION {version_tag!r}. "
            f"Known: {sorted(_LCB_RELEASE_FILES)}")
    paths = [hf_hub_download(_LCB_REPO, f, repo_type="dataset", revision=_LCB_REVISION)
             for f in files]
    return load_dataset("json", data_files=paths, split="train")


def _decode_test_cases(raw: Any, field: str, qid: Any) -> List[Dict[str, Any]]:
    """Test cases from either LCB encoding, or raise.

    `code_generation` ships plain JSON; `code_generation_lite` ships
    base64 -> zlib -> pickle. The previous `except: tc_data = []` silently produced a
    problem with no private tests, which does not fail the run - it makes the problem
    EASIER and inflates the score, invisibly. An undecodable payload is a broken input, so
    it stops the run instead.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, dict)]
    text = str(raw)
    try:
        return [t for t in json.loads(text) if isinstance(t, dict)]
    except Exception:                                                   # noqa: BLE001
        pass
    try:
        import base64
        import pickle
        import zlib
        decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(
            text.encode("utf-8")))))
        return [t for t in decoded if isinstance(t, dict)]
    except Exception as e:                                              # noqa: BLE001
        raise RuntimeError(
            f"lcb: could not decode {field} for question {qid!r} as JSON or as "
            f"base64+zlib+pickle ({e}). Refusing to continue: an empty test list would "
            f"silently make this problem free rather than fail it."
        ) from e


def _func_name(metadata: Any, qid: Any = None) -> Optional[str]:
    """`metadata["func_name"]`: set for call-based (LeetCode) problems, absent for stdin ones.

    Upstream's CodeGenerationProblem json.loads the metadata and would stop on a malformed
    value. Reading it as `{}` instead would grade a call-based problem as a stdin one, i.e.
    score it wrong without saying so, so a value that is present but not a JSON object raises.
    Only an absent/empty value means "no func_name".
    """
    if metadata is None or metadata == "":
        return None
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError as e:
            raise RuntimeError(
                f"lcb: question {qid!r} has metadata that is not valid JSON ({e}): "
                f"{metadata[:200]!r}. Refusing to grade a call-based problem as stdin.") from e
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"lcb: question {qid!r} has metadata that is not a JSON object: {metadata!r:.200}")
    return metadata.get("func_name")


def _include_test_gen() -> bool:
    """Test output prediction runs unless GBENCH_LCB_INCLUDE_TEST_GEN (or the legacy
    LCB_INCLUDE_TEST_GEN) is "0"."""
    return (suite_env("GBENCH_LCB_INCLUDE_TEST_GEN", "LCB_INCLUDE_TEST_GEN") or "1").strip() != "0"


def _include_self_repair() -> bool:
    """Self-repair runs unless GBENCH_LCB_INCLUDE_SELF_REPAIR is "0"."""
    return (os.environ.get("GBENCH_LCB_INCLUDE_SELF_REPAIR") or "1").strip() != "0"


def _shard_rows(rows: List[Any], offset: int) -> List[Any]:
    """This node's rows of one scenario under --shard I/N (GBENCH_SHARD). The full ordered set is
    code generation, then code execution, then test output prediction, concatenated; shard I
    keeps the positions I-1, I-1+N, ... of that list (base's round-robin), so a scenario whose
    rows start at `offset` keeps the rows at (offset + j) % N == I-1. All rows without a shard."""
    shard = current_shard()
    if shard is None:
        return list(rows)
    return [r for j, r in enumerate(rows) if (offset + j) % shard.count == shard.index - 1]


def _tests_digest(item: Dict[str, Any], public: List[Dict[str, Any]],
                  private: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What a stored code generation gold says about the problem's tests instead of holding
    them: the count of each field, and the sha256 of each field as the dataset row stores it
    (public_test_cases as JSON text, private_test_cases base64+zlib+pickle), so the tests a
    verdict was graded on can be re-fetched from the pinned revision and checked."""
    import hashlib

    def _sha(raw: Any) -> Optional[str]:
        if raw is None or raw == "":
            return None
        text = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"test_counts": {"public_test_cases": len(public), "private_test_cases": len(private)},
            "tests_sha256": {"public_test_cases": _sha(item.get("public_test_cases")),
                             "private_test_cases": _sha(item.get("private_test_cases"))}}


def _resolve_gold(gold: Any, tests_store: Optional[Dict[str, List[Dict[str, Any]]]]) -> Any:
    """A code generation gold with its tests, for the grader. A gold the loader built with a
    `tests_store` names its tests (`tests_ref`) instead of holding them; they are looked up in
    that run's store. Any other gold comes back as it is."""
    if not (isinstance(gold, dict) and "tests" not in gold and gold.get("tests_ref") is not None):
        return gold
    ref = gold["tests_ref"]
    if tests_store is None or ref not in tests_store:
        raise RuntimeError(f"lcb: the tests of question {gold.get('question_id')!r} are not in "
                           "this run's test store, so the program cannot be graded")
    return {**gold, "tests": tests_store[ref]}


def _load_lcb_samples(enable_thinking: bool = False, limit: Optional[int] = None,
                      tests_store: Optional[Dict[str, List[Dict[str, Any]]]] = None
                      ) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load LiveCodeBench samples: code generation (public + private tests), code execution
    and test output prediction (unless GBENCH_LCB_INCLUDE_TEST_GEN=0).

    With `tests_store` (a dict the run owns), a code generation problem's public and private
    tests go into the store under the gold's `tests_ref`, and the gold keeps only their counts
    and sha256 (_tests_digest). The gold is stored in every trace (once per sample), and a full
    release holds about 9.4 GB of tests, so the result would otherwise carry them ten times over.
    Without a store the gold holds the tests (`tests`)."""
    from datasets import load_dataset

    samples = []

    # 1. Code Generation (with both public and private test cases)
    # `code_generation_lite`, NOT `code_generation`. The latter is the frozen v1 dump:
    # 400 problems dated 2023-05..2024-03, every one of them inside any current model's
    # training window - which defeats the single thing LiveCodeBench exists to do. "lite"
    # is not a reduced benchmark: verified 2026-08-20 on problem 1873_A, both ship 1 public
    # + 4 private cases; lite just stores `private_test_cases` base64+zlib+pickle'd, which
    # is why it is 1.25GB against 9.4GB. Only lite carries v2..v6 and accepts window tags
    # (`v4_v5` = problems added between those releases), so only lite can be pointed past a
    # model's cutoff. Set GBENCH_LCB_VERSION to the window you want.
    ds_gen = _load_lcb_lite(_LCB_VERSION)
    if _LCB_MIN_DATE:
        before = len(ds_gen)
        ds_gen = [r for r in ds_gen if str(r.get("contest_date") or "") >= _LCB_MIN_DATE]
        logger.info("lcb: contest_date >= %s keeps %d of %d problems.",
                    _LCB_MIN_DATE, len(ds_gen), before)
        if not ds_gen:
            raise RuntimeError(
                f"lcb: GBENCH_LCB_MIN_DATE={_LCB_MIN_DATE!r} excluded every problem in "
                f"{_LCB_VERSION!r}. Pick a later release tag or an earlier date.")
    # --shard first, over the full ordered set (the three scenarios concatenated: _shard_rows),
    # then the --eval-limit draw within the shard: N samples for the whole suite, spread over
    # the scenarios that run (sampling.allocate_limit, seeded on the suite name), never N per
    # scenario. A scenario that gets 0 is not run in that limited run.
    n_gen = len(ds_gen)
    ds_gen = _shard_rows(list(ds_gen), 0)
    # 2. Code Execution. Part of the default run, so a load failure is a hard error rather
    # than a silently smaller suite.
    try:
        ds_exec = list(load_dataset(_LCB_EXEC_REPO, split="test", revision=_LCB_EXEC_REVISION))
    except Exception as e:                                              # noqa: BLE001
        raise RuntimeError(
            f"lcb: could not load the code execution scenario ({_LCB_EXEC_REPO}@"
            f"{_LCB_EXEC_REVISION[:8]}): {e}. See {DOCS_URL}") from e
    n_exec = len(ds_exec)
    ds_exec = _shard_rows(ds_exec, n_gen)
    # 3. Test output prediction - part of the default run like the other two scenarios
    # (GBENCH_LCB_INCLUDE_TEST_GEN=0, legacy alias LCB_INCLUDE_TEST_GEN=0, leaves it out),
    # scored with upstream's check_testcase_output.
    ds_test: List[Any] = []
    if _include_test_gen():
        try:
            ds_test = list(load_dataset(_LCB_TESTGEN_REPO, split="test",
                                        revision=_LCB_TESTGEN_REVISION))
        except Exception as e:                                          # noqa: BLE001
            raise RuntimeError(
                f"lcb: could not load the test output prediction scenario "
                f"({_LCB_TESTGEN_REPO}@{_LCB_TESTGEN_REVISION[:8]}): {e}. It is part of the "
                f"default run; set GBENCH_LCB_INCLUDE_TEST_GEN=0 to leave it out on purpose. "
                f"See {DOCS_URL}") from e
        ds_test = _shard_rows(ds_test, n_gen + n_exec)
    else:
        logger.info("LCB: test output prediction left out (GBENCH_LCB_INCLUDE_TEST_GEN=0).")
    sizes = {"code_generation": len(ds_gen), "code_execution": len(ds_exec)}
    if _include_test_gen():
        sizes["test_output_prediction"] = len(ds_test)
    # code generation is the headline scenario: it gets the first sample of a limited run
    counts = allocate_limit(limit, list(sizes), available=sizes, seed="lcb",
                            first=("code_generation",))
    if limit is not None and limit > 0:
        # Stratified by difficulty, not a contiguous head: the split is ordered, so a head
        # skewed the easy/medium/hard mix. The other two: a seeded subset, not a head either.
        n = counts["code_generation"] or 0
        ds_gen = limit_dataset(ds_gen, n, "difficulty", seed="lcb") if n else []
        n = counts["code_execution"] or 0
        ds_exec = stratified_sample(ds_exec, n, None, seed="lcb_exec") if n else []
        n = counts.get("test_output_prediction") or 0
        ds_test = stratified_sample(ds_test, n, None, seed="lcb_testgen") if n else []

    raw_gen = list(ds_gen)
    for item in raw_gen:
        content = item.get("question_content") or item.get("question", "")
        starter = item.get("starter_code") or ""
        difficulty = item.get("difficulty", "coding")

        # Collect both public and private test cases
        qid = str(item.get("question_id") or "")
        public = _decode_test_cases(item.get("public_test_cases"), "public_test_cases",
                                    item.get("question_id"))
        private = _decode_test_cases(item.get("private_test_cases"), "private_test_cases",
                                     item.get("question_id"))
        all_tests = public + private

        messages = [{"role": "system", "content": _SYSTEM_MESSAGE_GENERIC},
                    {"role": "user", "content": _generation_prompt(content, starter)}]
        gold = {"task": "generation", "question_id": qid,
                "fn_name": _func_name(item.get("metadata"), item.get("question_id")),
                # restated by the self-repair prompt (never sent to the grader)
                "question_content": content,
                # the problem window the score covers (the gold is never sent to the server)
                "contest_date": str(item.get("contest_date") or ""),
                **_tests_digest(item, public, private)}
        if tests_store is None:
            gold["tests"] = all_tests
        else:
            ref = qid if qid and qid not in tests_store else f"{qid}#{len(tests_store)}"
            tests_store[ref] = all_tests
            gold["tests_ref"] = ref
        # question_id gives the trace its shard-independent sample_key
        samples.append((messages, gold, {"category": f"code_gen_{difficulty}",
                                         "question_id": qid}))
    logger.info(f"Loaded {len(raw_gen)} LCB Code Generation samples (public + private tests).")

    for item in ds_exec:
        code = str(item.get("code", ""))
        call_input = str(item.get("input", ""))
        messages = [{"role": "system", "content": _EXEC_SYSTEM_MESSAGE},
                    {"role": "user", "content": _execution_prompt(code, call_input)}]
        samples.append((messages,
                        {"task": "execution", "code": code, "input": call_input,
                         "output": str(item.get("output", ""))},
                        {"category": "code_execution"}))
    logger.info("Loaded LCB Code Execution samples.")

    for item in ds_test:
        tests = item.get("test")
        tests = json.loads(tests) if isinstance(tests, str) else (tests or [])
        if not tests:
            raise RuntimeError(f"lcb: test output prediction row {item.get('question_id')!r} "
                               f"has no test case; refusing to fabricate one.")
        prompt = _test_output_prompt(str(item.get("question_content", "")),
                                     str(item.get("starter_code", "")),
                                     str(tests[0].get("input", "")))
        messages = [{"role": "system", "content": _TESTGEN_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}]
        samples.append((messages, {"task": "test_gen", "output": str(tests[0].get("output", ""))},
                        {"category": f"test_gen_{item.get('difficulty', 'medium')}"}))
    if _include_test_gen():
        logger.info("Loaded LCB Test Output Prediction samples.")

    return samples


# --- grading ------------------------------------------------------------------------------
#
# Each grader runs upstream's own code inside the sandbox, in a fresh interpreter. The vendored
# source travels in the JSON payload on stdin (never a `-c` argument, which would hit
# MAX_ARG_STRLEN on large tests), and is exec'd in a namespace whose `__builtins__` is a dict,
# as it is for an imported module (reliability_guard assigns `__builtins__["help"]`).
#
# The process that runs the model's code never holds the expected answers, so it cannot produce
# a pass it did not earn:
#
# * stdin carries two length-prefixed parts and then a per-call nonce. The first part is what
#   the program may see (its code, the test inputs, the entry point); the second holds the
#   expected outputs. The grader process reads only the first part, keeps a private duplicate of
#   stdout as the verdict fd, points fd 1 at /dev/null, marks itself non-dumpable (so no other
#   process of this user can read its memory or open its fds through /proc/<pid>, or ptrace it)
#   and forks.
# * The child re-points fd 0 at /dev/null and closes the verdict fd, then waits until the grader
#   has read the second part, and only then runs the program through upstream's own helpers
#   (compile_code, get_function, call_method, Capturing, unsafe_execute, parse_assert_statement,
#   reliability_guard, the same per-test alarms and error codes). Instead of comparing, it
#   sends what the program produced (captured stdout, or the return value in a typed encoding
#   that keeps Python and numpy data types) to the grader, one message per test.
# * The grader, which never runs model code, compares with upstream's own comparison code
#   (grade_stdio's line-wise check with the Decimal fallback via get_stripped_lines /
#   convert_line_to_decimals, grade_call_based's `prediction == gt_out`, check_testcase_output's
#   `==` against json.loads(expected)), builds upstream's per-test result list, reads the nonce
#   once the child is gone and writes a nonce-tagged verdict. For code execution, where upstream
#   runs `assert <expected> == <prediction>`, the program runs `assert <stand-in> == <prediction>`
#   and the stand-in's __eq__ asks the grader, which answers from the expected value it alone
#   holds; the sample passes only if exactly one such comparison happened, it was true and the
#   program then finished.
#
# So writing anything to any fd, stealing the nonce, or imitating the child's messages can at
# most report an output, and an output passes only if it is the right one. The one difference
# from upstream: a returned object that is not a plain Python or numpy data value (for example
# an instance of a class with its own __eq__) cannot be sent and never equals the expected
# output, where upstream would call its __eq__. No LiveCodeBench answer is such an object.

#: Keys of a grader payload that the model's program must never see.
_PRIVATE_KEYS = ("outputs", "expected")

_GRADER_MAIN = r'''
import builtins, ctypes, json, os, sys, warnings
# Upstream grades in a process forked from its evaluator, which already has numpy and pandas
# (via datasets) imported; reliability_guard would break a first import of either here.
import numpy, pandas
warnings.filterwarnings("ignore", message=r".*multi-threaded.*fork", category=DeprecationWarning)


def _read_exact(n):
    chunks = []
    while n > 0:
        b = os.read(0, min(n, 1 << 20))
        if not b:
            raise EOFError("lcb grader: payload truncated")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def _read_part():
    hdr = b""
    while not hdr.endswith(b"\n"):
        c = os.read(0, 1)
        if not c:
            raise EOFError("lcb grader: no payload header")
        hdr += c
    return json.loads(_read_exact(int(hdr)))


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


_p = _read_part()                     # the part the program may see
if hasattr(sys, "set_int_max_str_digits"):     # Python 3.11+; 3.10 has no such limit
    sys.set_int_max_str_digits(50000)

_verdict_fd = os.dup(1)
_null_w = os.open(os.devnull, os.O_WRONLY)
os.dup2(_null_w, 1)
os.close(_null_w)
_libc = ctypes.CDLL(None, use_errno=True)
if _libc.prctl(4, 0, 0, 0, 0) != 0:            # PR_SET_DUMPABLE, 0
    raise OSError(ctypes.get_errno(), "lcb grader: prctl(PR_SET_DUMPABLE, 0) failed")
_parent = os.getpid()
_msg_r, _msg_w = os.pipe()            # child -> grader: framed messages
_ans_r, _ans_w = os.pipe()            # grader -> child: comparison answers (code execution)
_go_r, _go_w = os.pipe()              # grader -> child: the expected outputs are read
_pid = os.fork()
if _pid == 0:
    # Child: the only process that runs model code. No stdin, no verdict fd, no expected output.
    try:
        _libc.prctl(1, 9, 0, 0, 0)             # PR_SET_PDEATHSIG, SIGKILL
        if os.getppid() != _parent:
            os._exit(1)
        for _fd in (_msg_r, _ans_w, _go_w, _verdict_fd):
            os.close(_fd)
        _null_r = os.open(os.devnull, os.O_RDONLY)
        os.dup2(_null_r, 0)
        os.close(_null_r)
        if os.read(_go_r, 1) != b"1":
            os._exit(1)
        os.close(_go_r)

        def _send(obj):
            data = json.dumps(obj).encode()
            _write_all(_msg_w, b"%d\n" % len(data) + data)

        def _ask(value):
            _send({"t": "eq", "v": value})
            return os.read(_ans_r, 1)

        _ns = {"__builtins__": builtins.__dict__, "__name__": _p["module"]}
        exec(compile(_p["source"], _p["module"] + ".py", "exec"), _ns)
        exec(compile(_p["child_source"], "lcb_gbench_child.py", "exec"), _ns)
        _ns["gbench_child"](_p, _send, _ask)
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    os._exit(0)

# Grader process: never runs model code.
for _fd in (_msg_w, _ans_r, _go_r):
    os.close(_fd)
_q = _read_part()                     # the expected outputs, read only after the fork
_write_all(_go_w, b"1")
os.close(_go_w)
_gns = {"__builtins__": builtins.__dict__, "__name__": _p["module"]}
exec(compile(_p["source"], _p["module"] + ".py", "exec"), _gns)
exec(compile(_p["grader_source"], "lcb_gbench_grader.py", "exec"), _gns)
_judge = _gns["gbench_judge"](_p, _q)
_buf = bytearray()
_limit = 256 << 20
_total = 0
_bad = None
while True:
    _chunk = os.read(_msg_r, 1 << 20)
    if not _chunk:
        break
    _buf += _chunk
    _total += len(_chunk)
    if _total > _limit:
        _bad = "the program's output exceeded %d bytes" % _limit
        break
    while _bad is None:
        _nl = _buf.find(b"\n", 0, 24)
        if _nl < 0:
            if len(_buf) >= 24:
                _bad = "malformed message from the grading process"
            break
        try:
            _n = int(_buf[:_nl])
        except ValueError:
            _bad = "malformed message from the grading process"
            break
        if len(_buf) < _nl + 1 + _n:
            break
        try:
            _msg = json.loads(bytes(_buf[_nl + 1:_nl + 1 + _n]))
        except ValueError:
            _msg = None
        del _buf[:_nl + 1 + _n]
        if isinstance(_msg, dict) and _msg.get("t") == "eq":
            try:
                _write_all(_ans_w, _judge.answer(_msg.get("v")))
            except OSError:
                pass
        else:
            _judge.record(_msg)
    if _bad is not None:
        break
if _bad is not None:
    try:
        os.kill(_pid, 9)
    except OSError:
        pass
os.close(_msg_r)
os.close(_ans_w)
os.waitpid(_pid, 0)
if _bad is not None:
    _passed, _detail = False, _bad
else:
    try:
        _passed, _detail = _judge.verdict()
    except BaseException as _e:
        _passed, _detail = False, "grading error: %r" % (_e,)
_nonce = b""
while True:
    _chunk = os.read(0, 4096)
    if not _chunk:
        break
    _nonce += _chunk
_meta = getattr(_judge, "meta", None) if _bad is None else None
os.write(_verdict_fd, json.dumps({"nonce": _nonce.decode("ascii", "replace").strip(),
                                  "passed": _passed is True,
                                  "detail": str(_detail)[:300],
                                  "meta": _meta if isinstance(_meta, dict) else None},
                                 default=str).encode() + b"\n")
'''

#: Runs in the child, in the vendored module's namespace (so upstream's helpers are in scope):
#: executes the program exactly as upstream does, up to the point where upstream compares, and
#: sends what the program produced instead.
_CHILD_SOURCE = r'''
import builtins as _gb_builtins
import collections as _gb_collections
import decimal as _gb_decimal
import fractions as _gb_fractions

try:
    import numpy as _gb_np
except Exception:                       # pragma: no cover - numpy is always present here
    _gb_np = None


def _gb_encode(x, depth=0):
    """A value, in a form the grader rebuilds as the same Python/numpy data type."""
    if depth > 200:
        return ["o", "too deep"]
    if x is None:
        return ["N"]
    if isinstance(x, bool):
        return ["b", bool.__bool__(x)]
    if _gb_np is not None and isinstance(x, _gb_np.ndarray):
        if x.dtype.kind not in "biufcU":
            return ["o", "ndarray " + str(x.dtype)]
        return ["a", x.dtype.str, list(x.shape), _gb_encode(x.tolist(), depth + 1)]
    if _gb_np is not None and isinstance(x, _gb_np.generic):
        if x.dtype.kind not in "biufcU":
            return ["o", type(x).__name__]
        return ["g", x.dtype.str, _gb_encode(x.item(), depth + 1)]
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
    if isinstance(x, _gb_collections.UserString):
        return ["s", str(x.data)]
    if isinstance(x, list):
        return ["l", [_gb_encode(e, depth + 1) for e in list.__iter__(x)]]
    if isinstance(x, _gb_collections.UserList):
        return ["l", [_gb_encode(e, depth + 1) for e in x.data]]
    if isinstance(x, tuple):
        return ["t", [_gb_encode(e, depth + 1) for e in tuple.__iter__(x)]]
    if isinstance(x, dict):
        return ["m", [[_gb_encode(k, depth + 1), _gb_encode(v, depth + 1)]
                      for k, v in dict.items(x)]]
    return ["o", type(x).__name__]


def _gb_exec_call_based(code, all_inputs, fn_name, timeout, send):
    # grade_call_based, up to the comparison
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        send({"setup": "none"})
        return
    method = get_function(compiled_sol, fn_name)
    if method is None:
        send({"setup": "none"})
        return
    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]
    send({"setup": "ok"})
    for idx, gt_inp in enumerate(all_inputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            prediction = method(*gt_inp)
            signal.alarm(0)
            # don't penalize model if it produces tuples instead of lists
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            send({"i": idx, "value": _gb_encode(prediction)})
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                send({"i": idx, "code": -3, "error": repr(e)})
            else:
                send({"i": idx, "code": -4, "error": repr(e)})
            return
        finally:
            signal.alarm(0)
            faulthandler.disable()


def _gb_exec_stdio(code, all_inputs, timeout, send):
    # grade_stdio, up to the comparison
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        send({"setup": "none"})
        return
    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        send({"setup": "none"})
        return
    send({"setup": "ok"})
    for idx, gt_inp in enumerate(all_inputs):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                call_method(method, gt_inp)
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    send({"i": idx, "code": -3, "error": repr(e)})
                else:
                    send({"i": idx, "code": -4, "error": repr(e)})
                return
            finally:
                signal.alarm(0)
                faulthandler.disable()
        send({"i": idx, "stdout": captured_output[0]})


class _GbExpected:
    """Stands in for the expected output in `assert <expected> == <prediction>`."""
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
        return "<expected output>"


def gbench_child(p, send, ask):
    kind = p["kind"]
    if kind == "generation":
        # run_test's preamble
        signal.signal(signal.SIGALRM, timeout_handler)
        reliability_guard()
        signal.alarm(p["timeout"])
        try:
            if p.get("fn_name") is None:
                _gb_exec_stdio(p["code"], p["inputs"], p["timeout"], send)
            else:
                _gb_exec_call_based(p["code"], p["inputs"], p["fn_name"], p["timeout"], send)
        except Exception as e:
            # run_test: "Error during testing: {e}"
            send({"fatal": str(e)})
        finally:
            signal.alarm(0)
    elif kind == "execution":
        _gb_builtins._gbench_expected_ = _GbExpected(ask)
        result = []
        unsafe_execute(p["program"], result, p["timeout"])
        send({"completed": bool(result) and result[0] == "passed",
              "detail": str(result[:1])[:200]})
    else:
        # check_testcase_output, up to the comparison
        testcase_str = p["prediction"]
        if len(testcase_str.splitlines()) > 1:
            for line in testcase_str.splitlines():
                if line.startswith("#"):
                    continue
                if "assert" in line:
                    testcase_str = line
                    break

        testcase_str = testcase_str.strip()

        if "assert" in testcase_str:
            testcase_output_str = str(parse_assert_statement(testcase_str))

        else:
            testcase_output_str = testcase_str

        try:
            testcase_output_eval = eval(testcase_output_str)
        except:
            send({"eval_failed": True})
            return
        send({"value": _gb_encode(testcase_output_eval)})
'''

#: Runs in the grader process, in the vendored module's namespace: rebuilds what the program
#: produced and applies upstream's comparison to the expected outputs.
_GRADER_SOURCE = r'''
import decimal as _gb_decimal
import fractions as _gb_fractions

import numpy as _gb_np


class _GbOpaque:
    """A returned value that is not plain data: equal only to itself."""
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
    if tag == "l":
        return [_gb_decode(e, depth + 1) for e in v[1]]
    if tag == "t":
        return tuple(_gb_decode(e, depth + 1) for e in v[1])
    if tag == "m":
        return {_gb_decode(k, depth + 1): _gb_decode(x, depth + 1) for k, x in v[1]}
    if tag in ("g", "a"):
        dt = _gb_np.dtype(str(v[1]))
        if dt.kind not in "biufcU":
            raise ValueError("bad dtype")
        if tag == "g":
            return dt.type(_gb_decode(v[2], depth + 1))
        shape = tuple(int(n) for n in v[2])
        return _gb_np.array(_gb_decode(v[3], depth + 1), dtype=dt).reshape(shape)
    if tag == "o":
        return _GbOpaque(str(v[1])[:80])
    raise ValueError("bad tag")


def _gb_positive(x):
    try:
        return bool(x > 0)
    except Exception:
        return False


class _GbJudge:
    def __init__(self, p, q):
        self.p, self.q = p, q
        self.kind = p["kind"]
        self.records = []
        self.eq_answers = []
        self.expected_value = None
        self.expected_error = None
        if self.kind == "execution":
            env = {}
            try:
                exec(BASE_IMPORTS, env)
                try:
                    self.expected_value = eval(q["expected"], env)
                except NameError:
                    exec(p["code"], env)
                    self.expected_value = eval(q["expected"], env)
            except BaseException as e:
                self.expected_error = repr(e)

    def record(self, msg):
        if isinstance(msg, dict):
            self.records.append(msg)

    def answer(self, value):
        # code execution: `assert <expected> == <prediction>`, evaluated here
        if self.kind != "execution" or self.eq_answers or self.expected_error is not None:
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
        if self.kind == "generation":
            results, meta = self._generation()
            # The failure metadata run_test would return (LiveCodeBench's self-repair prompt is
            # built from it); it leaves this process with the verdict.
            self.meta = meta
            flags = [_gb_positive(x) is True for x in (results or [])]
            return len(flags) == len(self.p["inputs"]) and all(flags), str(meta)
        if self.kind == "execution":
            done = next((r for r in self.records if "completed" in r), None)
            ok = (done is not None and done.get("completed") is True
                  and self.eq_answers == [b"T"])
            return ok, (done.get("detail", "") if done is not None else "no report")
        return self._test_output(), ""

    def _first(self):
        setup, fatal, by_idx = None, None, {}
        for r in self.records:
            if "fatal" in r and fatal is None:
                fatal = r["fatal"]
            elif "setup" in r and setup is None:
                setup = r["setup"]
            elif isinstance(r.get("i"), int) and r["i"] not in by_idx:
                by_idx[r["i"]] = r
        return setup, fatal, by_idx

    def _gt_inp(self, idx):
        # the test input as grade_call_based / grade_stdio hold it when they build metadata
        raw = self.p["inputs"][idx]
        if self.p.get("fn_name") is None:
            return raw
        try:
            return [json.loads(line) for line in raw.split("\n")]
        except Exception:
            return raw

    def _stopped(self, idx):
        # The program ended before reporting test idx (it exited or was killed); upstream's
        # check_correctness then finds no result and reports a TestRunnerError (-5).
        return {"error": "the program stopped before reporting test %d" % idx,
                "error_code": -5, "error_message": "TestRunnerError"}

    def _code_error(self, rec, idx, gt_out):
        # grade_call_based / grade_stdio: a timeout (-3) or an exception (-4) in test idx
        return {"error": rec.get("error"), "error_code": rec["code"],
                "error_message": ("Time Limit Exceeded" if rec["code"] == -3
                                  else "Runtime Error"),
                "inputs": truncatefn(self._gt_inp(idx)), "expected": truncatefn(gt_out)}

    def _generation(self):
        # run_test: any exception while grading is [-4]
        setup, fatal, by_idx = self._first()
        if fatal is not None:
            return [-4], {"error_code": -4, "error_message": "Error during testing: %s" % fatal}
        if setup != "ok":
            # grade_* returned None (no entry point), which run_test fails to unpack
            return [-4], {"error_code": -4, "error_message": "Error during testing: cannot "
                                                             "unpack non-iterable NoneType object"}
        all_outputs = self.q["outputs"]
        n = len(self.p["inputs"])
        if self.p.get("fn_name") is None:
            return self._stdio(all_outputs, by_idx, n)
        try:
            all_outputs = [json.loads(output) for output in all_outputs]
        except Exception as e:
            return [-4], {"error_code": -4, "error_message": "Error during testing: %s" % e}
        all_results = []
        for idx, gt_out in enumerate(all_outputs[:n]):
            rec = by_idx.get(idx)
            if rec is None:
                return all_results, self._stopped(idx)
            if "code" in rec:
                all_results.append(rec["code"])
                return all_results, self._code_error(rec, idx, gt_out)
            # grade_call_based, from the comparison on
            try:
                prediction = _gb_decode(rec.get("value"))
                if isinstance(prediction, tuple):
                    prediction = list(prediction)
                tmp_result = prediction == gt_out
                all_results.append(tmp_result)
                if not tmp_result:
                    return all_results, {
                        "output": truncatefn(prediction),
                        "inputs": truncatefn(self._gt_inp(idx)),
                        "expected": truncatefn(gt_out),
                        "error_code": -2,
                        "error_message": "Wrong Answer",
                    }
            except Exception as e:
                all_results.append(-4)
                return all_results, {"error": repr(e), "error_code": -4,
                                     "error_message": "Runtime Error",
                                     "inputs": truncatefn(self._gt_inp(idx)),
                                     "expected": truncatefn(gt_out)}
        return all_results, {}

    def _stdio(self, all_outputs, by_idx, n):
        all_results = []
        for idx, gt_out in enumerate(all_outputs[:n]):
            rec = by_idx.get(idx)
            if rec is None:
                return all_results, self._stopped(idx)
            if "code" in rec:
                all_results.append(rec["code"])
                return all_results, self._code_error(rec, idx, gt_out)
            prediction = rec.get("stdout")
            if not isinstance(prediction, str):
                all_results.append(-4)
                return all_results, {"error": "no output", "error_code": -4,
                                     "error_message": "Runtime Error",
                                     "inputs": truncatefn(self._gt_inp(idx)),
                                     "expected": truncatefn(gt_out)}
            # grade_stdio, from the comparison on
            stripped_prediction_lines = get_stripped_lines(prediction)
            stripped_gt_out_lines = get_stripped_lines(gt_out)

            WA_send_args = {
                "output": truncatefn(prediction),
                "inputs": truncatefn(self._gt_inp(idx)),
                "expected": truncatefn(gt_out),
                "error_code": -2,
            }

            if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
                all_results.append(-2)
                WA_send_args["error_message"] = "Wrong answer: mismatched output length"
                return all_results, WA_send_args

            for output_line_idx, (
                stripped_prediction_line,
                stripped_gt_out_line,
            ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
                WA_send_args["error_message"] = (
                    f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
                )

                ## CASE 1: exact match
                if stripped_prediction_line == stripped_gt_out_line:
                    continue

                ## CASE 2: element-wise comparision
                success, decimal_prediction_line = convert_line_to_decimals(
                    stripped_prediction_line
                )
                if not success:
                    all_results.append(-2)
                    return all_results, WA_send_args
                success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
                if not success:
                    all_results.append(-2)
                    return all_results, WA_send_args

                if decimal_prediction_line == decimal_gtout_line:
                    continue

                all_results.append(-2)
                return all_results, WA_send_args
            all_results.append(True)
        return all_results, {}

    def _test_output(self):
        # check_testcase_output, from the comparison on
        rec = next((r for r in self.records if "value" in r or "eval_failed" in r), None)
        global_result = None
        if rec is None or rec.get("eval_failed"):
            global_result = False
        try:
            expected_output_eval = json.loads(self.q["expected"])
        except:
            global_result = False
        if global_result is None:
            try:
                global_result = _gb_decode(rec["value"]) == expected_output_eval
            except Exception:
                global_result = False
        return global_result is True


def gbench_judge(p, q):
    return _GbJudge(p, q)
'''


def _run_grader_verdict(payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Run one vendored grader in the sandbox: {"passed", "detail", "meta"}. Anything but a
    pass verdict carrying this call's nonce is a fail. `meta` is run_test's failure metadata for
    a code generation grade (None otherwise, or when no verdict came back).

    `payload` holds everything the grader needs; the keys in _PRIVATE_KEYS (the expected
    outputs) travel in the second stdin part, which only the grader process reads."""
    import secrets
    import subprocess
    public = {k: v for k, v in payload.items() if k not in _PRIVATE_KEYS}
    public["child_source"] = _CHILD_SOURCE
    public["grader_source"] = _GRADER_SOURCE
    private = {k: payload[k] for k in _PRIVATE_KEYS if k in payload}
    # ensure_ascii: a part's length in characters is its length in bytes
    body, secret = json.dumps(public), json.dumps(private)
    nonce = secrets.token_hex(16)
    try:
        r = run_sandboxed([sys.executable, "-c", _GRADER_MAIN],
                          input=f"{len(body)}\n{body}{len(secret)}\n{secret}{nonce}",
                          capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # lcb_runner's global timeout: a run that never reports counts as all tests failed.
        return {"passed": False, "detail": f"no verdict within {timeout:.0f}s", "meta": None}
    except Exception as e:                                              # noqa: BLE001
        return {"passed": False, "detail": repr(e), "meta": None}
    for line in reversed((r.stdout or "").splitlines()):
        try:
            verdict = json.loads(line)
        except ValueError:
            continue
        if isinstance(verdict, dict) and verdict.get("nonce") == nonce:
            meta = verdict.get("meta")
            return {"passed": verdict.get("passed") is True,
                    "detail": str(verdict.get("detail") or ""),
                    "meta": meta if isinstance(meta, dict) else None}
    return {"passed": False, "detail": (r.stderr or "")[-500:], "meta": None}


def _run_grader_raw(payload: Dict[str, Any], timeout: float) -> Tuple[bool, str]:
    """(passed, diagnostic) of _run_grader_verdict."""
    v = _run_grader_verdict(payload, timeout)
    return v["passed"], v["detail"]


def _run_grader(payload: Dict[str, Any], timeout: float) -> bool:
    return _run_grader_raw(payload, timeout)[0]


def _generation_payload(resp_text: str, gold_payload: Dict[str, Any]
                        ) -> Optional[Tuple[Dict[str, Any], float]]:
    """(grader payload, timeout) for a code generation grade, or None when the problem has no
    usable tests."""
    if "tests" not in gold_payload and gold_payload.get("tests_ref") is not None:
        # a stored gold that names its tests: grading it without them would fail every program
        raise RuntimeError(f"lcb: the tests of question {gold_payload.get('question_id')!r} "
                           "were not resolved before grading (_resolve_gold)")
    tests = gold_payload.get("tests") or []
    if not tests or not all(isinstance(t, dict) for t in tests):
        return None
    payload = {
        "kind": "generation", "module": "lcb_testing_util", "source": _LCB_TESTING_UTIL,
        "code": _extract_code(resp_text), "timeout": _LCB_TIMEOUT_S,
        "inputs": [str(t.get("input", "")) for t in tests],
        "fn_name": gold_payload.get("fn_name"),
        "outputs": [str(t.get("output", "")) for t in tests],
    }
    # check_correctness joins the worker after (timeout + 1) * len(inputs) + 5 seconds.
    return payload, (_LCB_TIMEOUT_S + 1) * len(tests) + 5


def _grader_self_check() -> Optional[str]:
    """Why the graders cannot run here, or None.

    Every grader failure scores a sample as wrong, so a broken grading environment (sandbox,
    missing numpy/pandas) would read as a model scoring 0%. Grade four known-correct answers
    first and stop the suite if any of them fails.
    """
    canaries = [
        ("stdin", "```python\nprint(sum(map(int, input().split())))\n```",
         {"task": "generation", "fn_name": None,
          "tests": [{"input": "2 3", "output": "5", "testtype": "stdin"}]}),
        ("call-based", "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```",
         {"task": "generation", "fn_name": "add",
          "tests": [{"input": "2\n3", "output": "5", "testtype": "functional"}]}),
        ("code execution", "assert f(2) == 4\n[/ANSWER]",
         {"task": "execution", "code": "def f(x):\n    return x * 2", "input": "f(2)",
          "output": "4"}),
        ("test output prediction", "```python\nassert f(2) == [1, 2]\n```",
         {"task": "test_gen", "output": "[1, 2]"}),
    ]
    for name, response, gold in canaries:
        if not _verify_single_lcb_sample(response, gold):
            return (f"the LiveCodeBench {name} grader failed a known-correct answer, so every "
                    f"sample would score 0 (check bubblewrap, that numpy and pandas import in "
                    f"{sys.executable}, and that the grader can call prctl and fork)")
    if _include_self_repair():
        # The self-repair prompt is built from the grader's failure metadata.
        verdict, meta = _judge_lcb_sample("```python\nprint(6)\n```", canaries[0][2])
        if verdict or not isinstance(meta, dict) or meta.get("error_code") != -2 \
                or meta.get("inputs") != "2 3" or meta.get("expected") != "5":
            return (f"the LiveCodeBench grader did not report the failure metadata of a wrong "
                    f"answer ({meta!r}), which the self-repair prompt is built from")
    return None


def _verify_single_lcb_sample(resp_text: str, gold_payload: Any) -> bool:
    """Grade one LCB response with upstream's scorer for its scenario."""
    if not resp_text:
        return False

    if isinstance(gold_payload, list):
        gold_payload = {"task": "generation", "tests": gold_payload}
    elif not isinstance(gold_payload, dict):
        # An unparseable payload means nothing was executed against the submission.
        # Returning True auto-passed those rows.
        return False

    task = gold_payload.get("task", "generation")

    # Task 1: Code Generation - codegen_metrics / check_correctness / run_test.
    if task == "generation":
        built = _generation_payload(resp_text, gold_payload)
        if built is None:
            # A problem whose tests could not be decoded is unverified, not solved, and a
            # case that cannot run is not a passed case.
            return False
        return _run_grader(*built)

    # Task 2: Code Execution - compute_code_execution_metrics.evaluate_score.
    elif task == "execution":
        code, call_input = gold_payload.get("code"), gold_payload.get("input")
        expected = gold_payload.get("output")
        if code is None or call_input is None or expected is None:
            return False
        prediction = _extract_execution_code(resp_text)
        if call_input in prediction:
            # Upstream skips a prediction that repeats the call; with nothing executed the
            # sample scores False.
            return False
        # Upstream executes f"{BASE_IMPORTS}\n{code}\nassert {expected} == {prediction}"; the
        # expected output stays with the grader and a stand-in takes its place.
        payload = {
            "kind": "execution", "module": "lcb_utils_execute", "source": _LCB_UTILS_EXECUTE,
            "timeout": _LCB_EXEC_TIMEOUT_S, "code": code,
            "program": f"{_LCB_BASE_IMPORTS}\n{code}\nassert _gbench_expected_ == {prediction}",
            "expected": str(expected),
        }
        # check_correctness joins after timeout + 1; the rest covers interpreter start-up.
        return _run_grader(payload, _LCB_EXEC_TIMEOUT_S + 1 + 10)

    # Task 3: Test Output Prediction - check_testcase_output (it evals the prediction, so it
    # runs in the sandbox too).
    elif task == "test_gen":
        expected = gold_payload.get("output")
        if expected is None:
            return False
        payload = {
            "kind": "test_output", "module": "lcb_test_output_prediction",
            "source": _LCB_TEST_OUTPUT_PREDICTION,
            "prediction": _extract_test_output_code(resp_text), "expected": str(expected),
        }
        return _run_grader(payload, 20)

    # Unrecognised task. This used to `return True`, i.e. auto-pass anything the verifier
    # did not understand. Measured 2026-08-20 it never fired (0 of 400 problems), but a
    # scorer whose default is "correct" turns a future schema change into silent inflation.
    return False


def _judge_lcb_sample(resp_text: str, gold_payload: Any) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """(verdict, failure metadata): _verify_single_lcb_sample's verdict and, for code
    generation, the metadata run_test reports for the program (the self-repair prompt is built
    from it). An empty response still fails, but its (empty) program is run for the metadata,
    as lcb_runner grades the empty extraction."""
    if isinstance(gold_payload, list):
        gold_payload = {"task": "generation", "tests": gold_payload}
    if not isinstance(gold_payload, dict) or gold_payload.get("task", "generation") != "generation":
        return _verify_single_lcb_sample(resp_text, gold_payload), None
    built = _generation_payload(resp_text or "", gold_payload)
    if built is None:
        return False, None
    v = _run_grader_verdict(*built)
    return (v["passed"] and bool(resp_text)), v["meta"]


async def _async_judge_lcb(sample_traces: List[Dict[str, Any]],
                           tests_store: Optional[Dict[str, List[Dict[str, Any]]]] = None
                           ) -> Tuple[int, int, Dict[str, Dict[str, Any]]]:
    """Execute all LCB test suites in parallel using ProcessPoolExecutor post-generation. A
    code generation gold that names its tests is graded with them from `tests_store`."""
    import asyncio
    import concurrent.futures

    loop = asyncio.get_running_loop()
    max_workers = min(32, (os.cpu_count() or 4))

    with tqdm(total=len(sample_traces), desc="Judging [LCB]") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            tasks = []
            for sample in sample_traces:
                resp = sample.get("response_text") or sample.get("response") or ""
                gold = _resolve_gold(sample.get("gold_answer"), tests_store)
                fut = loop.run_in_executor(pool, _judge_lcb_sample, resp, gold)
                tasks.append((sample, fut))

            correct_count = 0
            category_stats = {}
            for sample, fut in tasks:
                verdict, meta = await fut
                if _scenario(sample.get("category")) == "code_generation":
                    sample["lcb_grading"] = meta
                sample["is_correct"] = verdict
                sample["correct"] = verdict
                sample["status"] = "OK" if verdict else "FAILED"
                cat = sample.get("category")
                if cat:
                    if cat not in category_stats:
                        category_stats[cat] = {"correct": 0, "total": 0}
                    category_stats[cat]["total"] += 1
                    if verdict:
                        category_stats[cat]["correct"] += 1
                if verdict:
                    correct_count += 1
                pbar.update(1)

    return correct_count, len(sample_traces), category_stats


def _scenario(category: Any) -> str:
    cat = str(category or "")
    if cat == "code_execution":
        return "code_execution"
    if cat.startswith("test_gen"):
        return "test_output_prediction"
    return "code_generation"


def _problem_of(trace: Dict[str, Any]) -> Any:
    """The problem (source sample) a trace belongs to: its position in the full ordered set with
    its sample_key when base recorded them (the same on every shard, so a merged union groups
    right), else its source sample index in this run."""
    if trace.get("global_idx") is not None:
        return ("global", trace.get("global_idx"), str(trace.get("sample_key")))
    return ("source", trace.get("source_sample_idx", trace.get("sample_idx")))


def _by_problem(traces: List[Dict[str, Any]]) -> Dict[Any, List[bool]]:
    """The verdicts of `traces`, grouped by problem (source sample)."""
    by_src: Dict[Any, List[bool]] = {}
    for t in traces:
        by_src.setdefault(_problem_of(t), []).append(bool(t.get("is_correct")))
    return by_src


def pass_at_1(traces: List[Dict[str, Any]]) -> Optional[float]:
    """lcb_runner's pass@1 estimate over `traces` (evaluation/pass_k_utils.py
    compute_metrics_from_results with k=1): the mean over problems of c_i / n_i, the correct share
    of each problem's generations, not correct / total over every generation. The two differ when
    problems have different numbers of scored generations (a generation left out of the score as
    a harness error). None when there is no trace (percent otherwise)."""
    by_src = _by_problem(traces)
    if not by_src:
        return None
    return 100.0 * sum(sum(v) / len(v) for v in by_src.values()) / len(by_src)


def _at_k(traces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """avg@k / pass@k / pass^k over one scenario's traces (grouped by source sample). avg@k is
    lcb_runner's pass@1 estimate, the mean over problems of c_i / n_i (pass_at_1)."""
    by_src = _by_problem(traces)
    if not by_src:
        return None
    gens = sum(len(v) for v in by_src.values())
    n = len(by_src)
    return {
        "samples": n,
        "generations": gens,
        "avg_at_k": round(pass_at_1(traces), 2),
        "pass_at_k": round(sum(1 for v in by_src.values() if any(v)) / n * 100.0, 2),
        "pass_hat_k": round(sum(1 for v in by_src.values() if all(v)) / n * 100.0, 2),
        "unstable_samples": sum(1 for v in by_src.values() if any(v) and not all(v)),
    }


def _cut_off_in_reasoning(reply: Dict[str, Any]) -> bool:
    """A reply (trace or self-repair record) that stopped at max_tokens with no answer text: a
    --thinking generation whose budget ran out inside the reasoning channel."""
    return (reply.get("finish_reason") == "length"
            and not str(reply.get("response_text") or "").strip())


def _grade_all(items: List[Tuple[str, Any]]) -> List[Tuple[bool, Optional[Dict[str, Any]]]]:
    """_judge_lcb_sample over (response, gold) pairs, in parallel processes."""
    import concurrent.futures
    if not items:
        return []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 4))) as pool:
        return list(pool.map(_judge_lcb_sample, [r for r, _ in items], [g for _, g in items]))


async def _send_self_repair_requests(prompts: List[List[Dict[str, str]]], *, model_name: str,
                                     base_url: str, concurrency: int, thinking: bool,
                                     temperature: Optional[float], payload: Dict[str, Any],
                                     budget: Optional[int]) -> List[Tuple[Any, Optional[int]]]:
    """One chat request per self-repair prompt (lcb_runner's --n 1), sent with the code
    generation requests' sampling and max_tokens (`budget`, the token-limit policy's value for
    the mode, lowered for a prompt that leaves less room in the context window); returns
    (base.Reply, the max_tokens sent) pairs in order."""
    import asyncio
    import aiohttp
    from . import base
    payload = {**payload, **base.decoding_penalties()}
    ctk = base.effective_chat_template_kwargs(payload.get("chat_template_kwargs"), thinking)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    api_url = f"{base_url.rstrip('/')}/chat/completions"

    async def _one(session, messages):
        sent_max, oversize = await base.clamp_to_context_async(messages, budget, None, ctk)
        if oversize:
            return (base.Reply(None, None, None, None, f"{base.OVER_CONTEXT_ERROR}: {oversize}"),
                    None)
        return await base._send_single_request(
            session=session, api_url=api_url, model_name=model_name, messages=messages,
            extra_payload=payload, semaphore=semaphore, pbar=None,
            max_output_tokens=sent_max, temperature=temperature, thinking=thinking), sent_max

    connector = aiohttp.TCPConnector(limit=max(1, concurrency) + 20)
    async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=base.request_timeout_s(budget))) as session:
        return await asyncio.gather(*[_one(session, m) for m in prompts])


def _self_repair_report(gen: List[Dict[str, Any]], *, budget: Optional[int],
                        official_cap: Optional[int]) -> Dict[str, Any]:
    """The self-repair scenario's report from the code generation traces that took part (each
    carries `lcb_self_repair`): computed the same way for one run and for the union of a sharded
    run's traces (merge_shards)."""
    from collections import Counter
    scored = [t for t in gen if not t["lcb_self_repair"].get("harness_error")]
    graded = [{"source_sample_idx": t.get("source_sample_idx", t.get("sample_idx")),
               "sample_key": t.get("sample_key"), "global_idx": t.get("global_idx"),
               "is_correct": bool(t["lcb_self_repair"].get("is_correct"))} for t in scored]
    correct = sum(1 for g in graded if g["is_correct"])
    sent = [t["lcb_self_repair"] for t in gen if not t["lcb_self_repair"].get("kept")]
    messages_by_kind = Counter(str(r.get("failure_message")) for r in sent)
    # pass@1 as lcb_runner computes it: the mean over problems of c_i / n_i (pass_at_1)
    return {
        "correct": correct, "total": len(scored),
        "accuracy": round(pass_at_1(graded), 2) if scored else None,
        "at_k": _at_k(graded),
        "kept_accepted_programs": len(gen) - len(sent),
        "repair_requests": len(sent),
        "failed_requests": sum(1 for r in sent if r.get("response_text") is None),
        # repair requests with no reply: left out of the scenario's pass@1
        "harness_errors": sum(1 for r in sent if r.get("harness_error")),
        "truncated_responses": sum(1 for r in sent if r.get("finish_reason") == "length"),
        # stopped at a limit other than lcb_runner's max_tokens (this run's budget, or a
        # max_tokens the context window lowered)
        "cut_off_by_budget": sum(1 for r in sent if r.get("finish_reason") == "length"
                                 and (official_cap is None
                                      or r.get("max_tokens_sent") != official_cap)),
        # stopped at max_tokens with no answer at all: the budget ran out inside the reasoning
        "cut_off_in_reasoning": sum(1 for r in sent if _cut_off_in_reasoning(r)),
        "failure_messages": dict(sorted(messages_by_kind.items())),
        "max_output_tokens": budget,
        "prompt": (f"lcb_runner@{_LCB_RUNNER_COMMIT[:8]} prompts/self_repair.py "
                   "format_prompt_self_repair (LMStyle.OpenAIChat)"),
    }


def _run_self_repair(traces: List[Dict[str, Any]], *, model_name: str, base_url: str,
                     concurrency: int, thinking: bool, temperature: Optional[float],
                     payload: Dict[str, Any], budget: Optional[int] = None,
                     official_cap: Optional[int] = None,
                     tests_store: Optional[Dict[str, List[Dict[str, Any]]]] = None
                     ) -> Dict[str, Any]:
    """LiveCodeBench's self-repair scenario over this run's code generation samples.

    lcb_runner (runner/base_runner.py run_main_repair, --scenario selfrepair --n 1) takes each
    code generation sample with its grade and failure metadata: an accepted program is kept as it
    is, every other one gets format_prompt_self_repair's prompt (the question, the program, the
    failure message) and one new generation. All programs are then graded again with the code
    generation grader, and pass@1 is their mean. The sampling is the code generation run's, and
    so is the max_tokens (`budget`; default: the token-limit policy's value for the mode, as
    base.run_eval_suite resolves it for the code generation requests). `official_cap` is
    lcb_runner's max_tokens when this run sends it (no-think): a reply stopped there is the
    protocol's outcome; one stopped at any other limit is counted in `cut_off_by_budget`.
    Each trace gets `lcb_self_repair`; the return value is the scenario's report.

    A code generation sample whose request got no reply from the endpoint (a harness error base
    marks scoring_excluded) has no program to repair and is left out; a repair request that gets
    no reply is a harness error too, left out of the scenario's pass@1 and counted in
    `harness_errors` (never scored as a failed program). A gold that names its tests is graded
    with them from `tests_store`.
    """
    import asyncio
    from collections import Counter
    gen = [t for t in traces
           if _scenario(t.get("category")) == "code_generation" and is_scored(t)]
    if budget is None:
        budget = resolve_token_budget("lcb", OFFICIAL_SAMPLING, thinking=thinking).max_tokens
    jobs: List[Tuple[Dict[str, Any], List[Dict[str, str]], str]] = []
    messages_by_kind: Counter = Counter()
    for t in gen:
        gold = t.get("gold_answer")
        if isinstance(gold, list):
            gold = {"task": "generation", "tests": gold}
        question = (gold or {}).get("question_content") if isinstance(gold, dict) else None
        if question is None:
            raise RuntimeError("lcb: a code generation sample carries no question_content, so "
                               "its self-repair prompt cannot be built")
        meta_json, kind = _self_repair_metadata(t.get("lcb_grading"))
        code = _extract_code(t.get("response_text") or "")
        prompt = _self_repair_prompt(str(question), code, bool(t.get("is_correct")), meta_json)
        if prompt == "":
            continue
        messages_by_kind[kind] += 1
        jobs.append((t, prompt, kind))
    replies = asyncio.run(_send_self_repair_requests(
        [p for _, p, _ in jobs], model_name=model_name, base_url=base_url,
        concurrency=concurrency, thinking=thinking, temperature=temperature, payload=payload,
        budget=budget)) if jobs else []

    # Grade every program: the repaired ones and, as lcb_runner does, the kept ones again.
    to_grade: List[Tuple[Dict[str, Any], str]] = []
    repaired = {id(t): (kind, reply, sent_max)
                for (t, _, kind), (reply, sent_max) in zip(jobs, replies)}
    for t in gen:
        if id(t) in repaired:
            kind, reply, sent_max = repaired[id(t)]
            t["lcb_self_repair"] = {
                "kept": False, "failure_message": kind, "response_text": reply.text,
                "reasoning": reply.reasoning, "reasoning_chars": len(reply.reasoning or ""),
                "finish_reason": reply.finish_reason, "max_tokens_sent": sent_max,
                "completion_tokens": reply.completion_tokens, "error": reply.error}
            to_grade.append((t, reply.text or ""))
        else:
            t["lcb_self_repair"] = {"kept": True}
            to_grade.append((t, t.get("response_text") or ""))
    # A repair request that got no reply has no program to grade: a harness error.
    from . import base
    for t, _text in to_grade:
        rec = t["lcb_self_repair"]
        if not rec["kept"] and rec["response_text"] is None:
            err = str(rec.get("error") or "")
            status = ("OVER_CONTEXT" if err.startswith(base.OVER_CONTEXT_ERROR)
                      else "TIMEOUT" if err.startswith("timeout") else "FAILED")
            rec["harness_error"] = base.endpoint_harness_error(status, err or "no reply")
    graded_pairs = [(t, text) for t, text in to_grade
                    if not t["lcb_self_repair"].get("harness_error")]
    verdicts = _grade_all([(text, _resolve_gold(t.get("gold_answer"), tests_store))
                           for t, text in graded_pairs])
    for (t, text), (verdict, _meta) in zip(graded_pairs, verdicts):
        t["lcb_self_repair"]["is_correct"] = bool(verdict)
    for t, _text in to_grade:
        t["lcb_self_repair"].setdefault("is_correct", False)
    return _self_repair_report(gen, budget=budget, official_cap=official_cap)


#: The four scenarios LiveCodeBench publishes, in the order the result lists them.
_ALL_SCENARIOS = ("code_generation", "code_execution", "test_output_prediction", "self_repair")


def _apply_scenarios(result: Dict[str, Any], traces: List[Dict[str, Any]]
                     ) -> Dict[str, List[Dict[str, Any]]]:
    """Report the scenarios SEPARATELY on `result` (scenarios, per-category pass@1, the code
    generation headline and its @k) from `traces`, over the scored ones; returns them by
    scenario. `result` carries base's pooled accuracy / counts / attempts block on entry (a
    run's, or the generic merge's recount over a sharded run's union)."""
    # LiveCodeBench never combines the scenarios, and blending is not cosmetic: on the
    # 2026-08-20 run the headline read 87.37% off 73.25% generation and 99.16% execution, with
    # execution supplying 479 of 879 rows.
    # Over the scored traces: a request that got no reply from the endpoint is a harness error
    # (scoring_excluded), left out of every scenario and its @k. Each scenario's accuracy is
    # lcb_runner's pass@1 estimate, the mean over problems of c_i / n_i over the scored
    # generations (pass_at_1), not base's pooled correct / total, which differs from it once a
    # harness error leaves problems with different numbers of scored generations.
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for tr in traces:
        if is_scored(tr):
            buckets.setdefault(_scenario(tr.get("category")), []).append(tr)
    scenarios = {}
    for name, trs in buckets.items():
        ok = sum(1 for t in trs if t.get("is_correct"))
        scenarios[name] = {"correct": ok, "total": len(trs),
                           "accuracy": round(pass_at_1(trs), 2)}
    result["scenarios"] = scenarios
    # Each category (code_gen_<difficulty>, code_execution, test_gen_<difficulty>) the same way:
    # base pools its accuracy over the scored generations; lcb_runner's per-difficulty pass@1 is
    # the mean over that difficulty's problems of c_i / n_i. The counts stay base's.
    by_category: Dict[Any, List[Dict[str, Any]]] = {}
    for tr in traces:
        if is_scored(tr):
            by_category.setdefault(tr.get("category"), []).append(tr)
    for name, entry in (result.get("category_accuracy") or {}).items():
        if isinstance(entry, dict) and by_category.get(name):
            entry["accuracy"] = round(pass_at_1(by_category[name]), 2)

    # `accuracy` becomes code generation alone - the scenario "LiveCodeBench pass@1" means
    # - so the headline is a number that exists upstream. The blend is still reported, but
    # under a name that cannot be mistaken for a published figure. With k attempts per
    # problem the headline is upstream's pass@1 estimate (mean over problems of c_i / n_i).
    attempts_block = result.get("attempts")
    k = _attempts_per_problem(result)
    if "code_generation" in scenarios:
        cg = scenarios["code_generation"]
        result["blended_all_scenarios_accuracy"] = result.get("accuracy")
        result["all_scenarios_correct"] = result.get("correct_answers")
        result["all_scenarios_total"] = result.get("total_questions")
        result["accuracy"] = cg["accuracy"]
        result["correct_answers"] = cg["correct"]
        # every code generation generation run; the headline's denominator leaves out the ones
        # that got no reply (harness errors)
        result["total_questions"] = sum(1 for t in traces
                                        if _scenario(t.get("category")) == "code_generation")
        result["effective_n"] = cg["total"]
        result.pop("accuracy_withheld_reason", None)
        result["metric"] = (f"code_generation pass@1 ({result.get('lcb_version') or _LCB_VERSION}"
                            + (f", estimated from n={k} samples per problem" if k > 1 else "")
                            + "); other scenarios reported separately")
        if k > 1:
            # base's @k block pools every scenario; the headline one is code generation only.
            cg_atk = _at_k(buckets["code_generation"])
            if cg_atk and isinstance(attempts_block, dict):
                result["attempts_all_scenarios"] = attempts_block
                result["attempts"] = {**attempts_block, **cg_atk}
    elif any(_scenario(t.get("category")) == "code_generation" for t in traces):
        # Every code generation request got no reply (harness errors): the headline has no
        # value, and the other scenarios' blend must not stand in for it.
        result["blended_all_scenarios_accuracy"] = result.get("accuracy")
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = (
            "no code generation request was scored (every one got no reply from the served "
            "endpoint: harness errors), so code generation pass@1 has no value; other scenarios "
            "in `scenarios`")
    return buckets


def _attempts_per_problem(result: Dict[str, Any]) -> int:
    """The samples per problem the result's @k block records (1 without one)."""
    block = result.get("attempts_all_scenarios") or result.get("attempts")
    if isinstance(block, dict):
        return int(block.get("attempts_per_sample", 1) or 1)
    return 1


def _scenarios_not_run(traces: List[Dict[str, Any]], self_repair_ran: bool,
                       limit: Optional[int]) -> Dict[str, str]:
    """{scenario: why it did not run} for each of the four scenarios with no trace in `traces`
    (self-repair: with no repair report), from this run's settings."""
    ran = {_scenario(t.get("category")) for t in traces}
    if self_repair_ran:
        ran.add("self_repair")
    shard = current_shard()
    out: Dict[str, str] = {}
    for name in _ALL_SCENARIOS:
        if name in ran:
            continue
        if name == "test_output_prediction" and not _include_test_gen():
            out[name] = "left out with GBENCH_LCB_INCLUDE_TEST_GEN=0"
        elif name == "self_repair" and not _include_self_repair():
            out[name] = "left out with GBENCH_LCB_INCLUDE_SELF_REPAIR=0"
        elif name == "self_repair":
            out[name] = "no code generation sample was scored"
        elif limit:
            out[name] = f"dropped by --eval-limit {limit}"
        elif shard is not None and shard.partial:
            out[name] = f"no sample on shard {shard.label}"
        else:
            out[name] = "no sample loaded"
    return out


def _scenarios_reason(not_run: Dict[str, str]) -> Optional[str]:
    """The comparability reason naming the scenarios that did not run, with why, or None."""
    if not not_run:
        return None
    by_cause: Dict[str, List[str]] = {}
    for name in _ALL_SCENARIOS:
        if name in not_run:
            by_cause.setdefault(not_run[name], []).append(name)
    return ("scenario(s) not run: "
            + "; ".join(f"{', '.join(names)} ({cause})" for cause, names in by_cause.items())
            + " (LiveCodeBench publishes all four)")


def _outcome_reasons(result: Dict[str, Any], traces: List[Dict[str, Any]],
                     self_repair: Optional[Dict[str, Any]], *, thinking_run: bool,
                     budget: Optional[int]) -> List[str]:
    """The comparability reasons that come from what happened to this run's requests: replies
    stopped at a limit other than lcb_runner's 2000 (recorded as replies_cut_off_by_budget /
    replies_cut_off_in_reasoning) and requests with no reply. Computed the same way for a run and
    for a sharded run's union."""
    reasons: List[str] = []
    # A reply stopped at a limit other than lcb_runner's 2000 (with --thinking, this run's
    # budget: the 2000 cap is not applied to reasoning; without it, a max_tokens the context
    # window lowered) measures that limit, not the model.
    gen_cut, gen_empty = budget_cutoffs(result)
    repair_cut = (self_repair or {}).get("cut_off_by_budget", 0) or 0
    cut = gen_empty if thinking_run else 0
    cut_repair = ((self_repair or {}).get("cut_off_in_reasoning", 0) or 0) if thinking_run else 0
    result["replies_cut_off_by_budget"] = gen_cut + repair_cut
    result["replies_cut_off_in_reasoning"] = cut + cut_repair
    if cut or cut_repair:
        reasons.append(
            f"{cut + cut_repair} repl(ies) ({cut} generation, {cut_repair} self-repair) "
            f"reached this run's max_tokens {budget} inside the reasoning channel "
            "with no answer (lcb_runner's 2000 was defined for non-reasoning generation and "
            "is not applied with --thinking; raise --max-output-tokens)")
    partial = gen_cut + repair_cut - cut - cut_repair
    if partial:
        reasons.append(
            f"{partial} repl(ies) ({gen_cut - cut} generation, {repair_cut - cut_repair} "
            "self-repair) stopped at a max_tokens other than lcb_runner's 2000 before finishing "
            + ("(this run's budget; raise --max-output-tokens)" if thinking_run
               else "(the context window left less room)"))
    no_response = sum(1 for t in traces
                      if t.get("harness_error") or t.get("response_text", "") is None)
    if no_response:
        reasons.append(f"{no_response} generation request(s) got no reply from the served "
                       "endpoint (harness errors, left out of the score; lcb_runner retries a "
                       "failed request up to 10 times)")
    if self_repair and self_repair.get("harness_errors"):
        reasons.append(f"{self_repair['harness_errors']} self-repair request(s) got no reply "
                       "from the served endpoint (harness errors, left out of the self-repair "
                       "pass@1)")
    return reasons


def _contest_dates(traces: List[Dict[str, Any]]) -> Optional[List[str]]:
    """[first, last] contest month of the code generation problems in `traces`, or None."""
    dates = []
    for t in traces:
        gold = t.get("gold_answer") if isinstance(t.get("gold_answer"), dict) else {}
        extra = t.get("extra_payload") if isinstance(t.get("extra_payload"), dict) else {}
        date = gold.get("contest_date") or extra.get("contest_date")
        if date:
            dates.append(str(date)[:7])
    dates.sort()
    return [dates[0], dates[-1]] if dates else None


def run_lcb(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native LiveCodeBench algorithmic coding evaluation suite."""
    # Executes model-written code: require bubblewrap isolation. Missing/blocked isolation is an
    # EXTERNAL prerequisite gap -> hard-error (no-skip), even when invoked directly (bypassing the
    # central evals.py gate). GBENCH_SANDBOX=none opts into unsandboxed execution.
    _blocked = sandbox_skip_reason() or _grader_self_check()
    if _blocked:
        raise infra_required("lcb", _blocked, DOCS_URL)
    limit = kwargs.get("limit")
    # The code generation tests stay in this run's store, never in a trace (_load_lcb_samples);
    # the store is emptied when the run ends.
    tests_store: Dict[str, List[Dict[str, Any]]] = {}
    try:
        samples = _load_lcb_samples(enable_thinking, limit=limit, tests_store=tests_store)
        attempts = max(_LCB_N, int(kwargs.get("attempt_count") or 1))
        per_scenario = {name: 0 for name in _ALL_SCENARIOS[:3]
                        if name != "test_output_prediction" or _include_test_gen()}
        for _messages, _gold, _meta in samples:
            name = _scenario((_meta or {}).get("category"))
            per_scenario[name] = per_scenario.get(name, 0) + 1
        limit_plan = eval_limit_plan("lcb", limit, len(samples), per_scenario,
                                     setting_noun="scenario", attempts=attempts)
        if _include_self_repair() and per_scenario.get("code_generation"):
            # self-repair is a second pass on the same code generation samples (not more
            # samples): one repair request per rejected program, at most one per attempt
            limit_plan["self_repair_requests_at_most"] = per_scenario["code_generation"] * attempts
        # max_tokens: OFFICIAL_SAMPLING.max_tokens under base's token-limit policy (lcb_runner's
        # 2000 exactly without --thinking; the run's budget with it). The same plan sizes
        # self-repair.
        plan = resolve_token_budget("lcb", OFFICIAL_SAMPLING, thinking=enable_thinking)
        # lcb_runner sends top_p 0.95 and frequency/presence penalties 0 with every request; an
        # explicit extra_payload still wins. The same payload goes to the self-repair requests.
        extra_payload = {**(OFFICIAL_SAMPLING.params or {}), **(kwargs.get("extra_payload") or {})}

        async def _judge(traces: List[Dict[str, Any]]):
            return await _async_judge_lcb(traces, tests_store)

        result = run_eval_suite(
            eval_name="lcb",
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=presharded_for_base(samples),  # this shard's samples, kept as is
            async_eval_fn=_judge,
            declared_scoring_mode="execution",  # sandboxed test execution, not an LLM judge
            thinking=enable_thinking,
            extra_payload=extra_payload,
            limit=limit,
            max_output_tokens=kwargs.get("max_output_tokens"),
            # lcb_runner's --n: 10 samples per problem, from which pass@1 is estimated. Every
            # trace is graded on its own (no id-keyed batch), so k attempts are k real
            # generations.
            attempt_count=attempts,
            supports_attempts=True,
            official_sampling=OFFICIAL_SAMPLING,
        )
        result["lcb_version"] = _LCB_VERSION
        result["eval_limit_plan"] = limit_plan
        traces = result.get("sample_traces", []) or []
        buckets = _apply_scenarios(result, traces)
        k = _attempts_per_problem(result)
        # Self-repair: a second generation for every rejected code generation program, from
        # lcb_runner's self-repair prompt and the grader's failure metadata.
        self_repair = None
        if _include_self_repair() and buckets.get("code_generation"):
            self_repair = _run_self_repair(
                traces, model_name=model_name, base_url=base_url, concurrency=concurrency,
                thinking=enable_thinking, temperature=result.get("temperature"),
                payload=extra_payload, budget=plan.max_tokens,
                official_cap=plan.official_cap if plan.policy == "official" else None,
                tests_store=tests_store)
            result["scenarios"]["self_repair"] = {f: self_repair.get(f)
                                                  for f in ("correct", "total", "accuracy")}
            result["self_repair"] = self_repair
            if self_repair.get("harness_errors"):
                # Named in the run's status like base's own harness errors.
                cause = (f"self-repair harness errors left out of its score: "
                         f"{self_repair['harness_errors']} repair request(s) that got no reply "
                         "from the served endpoint")
                result["status_reason"] = "; ".join(filter(None, (result.get("status_reason"),
                                                                  cause)))
                if result.get("status") == "success":
                    result["status"] = "completed_with_errors"
    finally:
        tests_store.clear()
    # base's max_output_tokens_note states the mode's choice; it covers the self-repair requests,
    # which are sent the same max_tokens.
    if plan.policy == "reasoning_run_budget":
        result["max_output_tokens_note"] = "; ".join(filter(None, (
            result.get("max_output_tokens_note"),
            "lcb_runner defines no reasoning-model max_tokens for the OpenAIChat style (its "
            "reasoning styles are vendor-API variants with their own prompts; open reasoning "
            "models take --max_tokens on the command line)")))
    # The scenarios that ran: those with a trace, and self-repair when it ran. The others are
    # named with why (a toggle, the --eval-limit draw, the shard).
    not_run = _scenarios_not_run(traces, self_repair is not None, limit)
    result["lcb_scenarios"] = [s for s in _ALL_SCENARIOS if s not in not_run]
    result["lcb_scenarios_not_run"] = not_run
    result["lcb_min_contest_date"] = _LCB_MIN_DATE or None
    result["dataset"] = f"{_LCB_REPO}@{_LCB_REVISION}:{_LCB_VERSION}"
    result["lcb_runner_commit"] = _LCB_RUNNER_COMMIT
    result["tests_in_traces"] = (
        "not stored: each code generation trace's gold_answer names its problem (question_id) "
        "in `dataset` with its test counts and the sha256 of public_test_cases / "
        "private_test_cases as the dataset row stores them")
    # The problem dates the score actually covers. LiveCodeBench is only contamination-free
    # to the extent this window sits past the model's cutoff, and a release tag alone does
    # not say - the tags are cumulative, so `release_v6` still carries v1's 2023 problems.
    dates = _contest_dates(traces)
    if dates:
        result["contest_date_range"] = dates

    # Prompt, extraction and graders are upstream's; what can still differ is the sampling,
    # the scenarios and the problem set.
    _noncanon = []
    if _LCB_VERSION in ("release_v1", "v1"):
        _noncanon.append(f"contamination-prone version {_LCB_VERSION}")
    if limit:
        _noncanon.append(f"subset run (--eval-limit {limit})")
    if k < _LCB_N:
        _noncanon.append(f"{k} sample(s) per problem; lcb_runner estimates pass@1 from "
                         f"n={_LCB_N}")
    temp = result.get("temperature")
    if temp is None or float(temp) != _LCB_TEMPERATURE:
        _noncanon.append(f"temperature={temp} ({result.get('temperature_source')}); "
                         f"lcb_runner samples at {_LCB_TEMPERATURE}")
    # The request's sampling fields (decoding penalties are merged in last, as base sends them)
    # against lcb_runner's: every field it sends, and nothing else.
    official = dict(OFFICIAL_SAMPLING.params or {})
    sent = {**extra_payload, **(result.get("decoding_penalties") or {})}
    differ = sorted(f for f in set(sent) | set(official)
                    if f in OFFICIAL_SAMPLING_PARAM_KEYS and sent.get(f) != official.get(f))
    if differ:
        _noncanon.append("sampling " + ", ".join(
            f"{f}={sent.get(f, 'not sent')!r} (lcb_runner: {official.get(f, 'not sent')!r})"
            for f in differ))
    budgets = result.get("max_tokens_sent") or []
    if plan.policy == "official" and list(budgets) != [_LCB_MAX_TOKENS]:
        _noncanon.append(f"max_tokens={budgets or 'not sent'}; lcb_runner sends "
                         f"{_LCB_MAX_TOKENS} (a request whose prompt left less room in the "
                         "context window was sent less)")
    # the scenarios after the settings, in the order merge_shards rebuilds them
    if _scenarios_reason(not_run):
        _noncanon.append(_scenarios_reason(not_run))
    _noncanon.extend(_outcome_reasons(result, traces, self_repair,
                                      thinking_run=plan.policy == "reasoning_run_budget",
                                      budget=plan.max_tokens))
    result["leaderboard_comparable"] = not _noncanon
    if _noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(_noncanon)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded LiveCodeBench run (`gbench --merge`).

    The generic merge takes the union of the shards' traces (an item is its sample_key at its
    position in the full set) and sums the counts; the LiveCodeBench fields are then recomputed
    over that union from the stored verdicts, as run_lcb computes them for one run: every
    scenario's pass@1 (the mean over problems of c_i / n_i), the code generation headline and its
    @k, self-repair (from each trace's `lcb_self_repair`), the scenarios that ran, the problem
    window, the cut-off counts and the comparability reasons that come from them. The reasons
    that come from the run's settings (version, --eval-limit, samples per problem, sampling,
    max_tokens) are the shards' own, which agree. Nothing is graded again: the tests are not in
    the traces (see _load_lcb_samples)."""
    from ...merge import generic_merge, other_reasons
    merged = generic_merge("lcb", [dict(r, accuracy=None) for r in results])
    traces = merged.get("sample_traces") or []
    live = [r for r in results if not r.get("empty_shard")] or list(results)
    # base's all-scenario @k block, over the union (generic_merge rebuilt it from shard 1's
    # code generation block)
    first_all = next((r.get("attempts_all_scenarios") for r in live
                      if isinstance(r.get("attempts_all_scenarios"), dict)), None)
    if isinstance(first_all, dict):
        by_src = _by_problem([t for t in traces if is_scored(t)])
        n_src = len(by_src)
        merged["attempts"] = dict(
            first_all, samples=n_src, generations=len(traces), avg_at_k=merged.get("accuracy"),
            pass_at_k=round(sum(1 for v in by_src.values() if any(v)) / n_src * 100.0, 2)
            if n_src else None,
            pass_hat_k=round(sum(1 for v in by_src.values() if v and all(v)) / n_src * 100.0, 2)
            if n_src else None,
            unstable_samples=sum(1 for v in by_src.values() if any(v) and not all(v)))
    for field in ("scenarios", "blended_all_scenarios_accuracy", "all_scenarios_correct",
                  "all_scenarios_total", "attempts_all_scenarios", "metric", "self_repair",
                  "lcb_scenarios", "lcb_scenarios_not_run", "contest_date_range",
                  "replies_cut_off_by_budget", "replies_cut_off_in_reasoning", "effective_n"):
        merged.pop(field, None)
    buckets = _apply_scenarios(merged, traces)
    thinking_run = bool(merged.get("thinking"))
    # the per-request max_tokens of the token-limit plan (official_sampling records it)
    osamp = next((r["official_sampling"] for r in live
                  if isinstance(r.get("official_sampling"), dict)), {})
    budget = osamp.get("max_tokens_sent", osamp.get("max_tokens"))
    self_repair = None
    gen = [t for t in buckets.get("code_generation") or []
           if isinstance(t.get("lcb_self_repair"), dict)]
    if gen:
        self_repair = _self_repair_report(gen, budget=budget,
                                          official_cap=None if thinking_run else _LCB_MAX_TOKENS)
        merged["scenarios"]["self_repair"] = {f: self_repair.get(f)
                                              for f in ("correct", "total", "accuracy")}
        merged["self_repair"] = self_repair
    # Why a scenario is missing from the union: the shards' own causes (they ran one config).
    causes: Dict[str, List[str]] = {}
    for r in live:
        for name, cause in (r.get("lcb_scenarios_not_run") or {}).items():
            # no shard had a sample of it: the full set had none
            cause = "no sample loaded" if cause.startswith("no sample on shard ") else cause
            if cause not in causes.setdefault(name, []):
                causes[name].append(cause)
    ran = {_scenario(t.get("category")) for t in traces} | ({"self_repair"} if self_repair else set())
    not_run = {name: "; ".join(causes.get(name) or ["no sample on any shard"])
               for name in _ALL_SCENARIOS if name not in ran}
    merged["lcb_scenarios"] = [s for s in _ALL_SCENARIOS if s not in not_run]
    merged["lcb_scenarios_not_run"] = not_run
    dates = _contest_dates(traces)
    if dates:
        merged["contest_date_range"] = dates
    # The settings' reasons from the shards (each shard's own outcome reasons set aside), then
    # the scenarios and the outcome reasons over the union.
    settings_reasons: List[str] = []
    for r in live:
        own = _outcome_reasons(dict(r), r.get("sample_traces") or [], r.get("self_repair"),
                               thinking_run=thinking_run, budget=budget)
        own.append(_scenarios_reason(r.get("lcb_scenarios_not_run") or {}) or "")
        for reason in other_reasons(r, own):
            if reason not in settings_reasons:
                settings_reasons.append(reason)
    reasons = settings_reasons + [x for x in [_scenarios_reason(not_run)] if x]
    reasons += _outcome_reasons(merged, traces, self_repair, thinking_run=thinking_run,
                                budget=budget)
    merged["leaderboard_comparable"] = not reasons
    merged["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    if isinstance(merged.get("merge_unreconciled_fields"), dict):
        for field in ("scenarios", "self_repair", "lcb_scenarios", "lcb_scenarios_not_run",
                      "contest_date_range", "replies_cut_off_by_budget",
                      "replies_cut_off_in_reasoning", "attempts_all_scenarios", "metric",
                      "blended_all_scenarios_accuracy", "all_scenarios_correct",
                      "all_scenarios_total"):
            merged["merge_unreconciled_fields"].pop(field, None)
        if not merged["merge_unreconciled_fields"]:
            merged.pop("merge_unreconciled_fields")
    return merged


# --- vendored lcb_runner sources ----------------------------------------------------------
# Copied verbatim from github.com/LiveCodeBench/LiveCodeBench at commit 28fef95e and run only
# inside the sandbox by _GRADER_MAIN (the prompt templates and extraction helpers above are
# from the same commit). utils_execute.py carries its own Apache-2.0 header (adapted from
# openai/human-eval). Do not edit: a changed line is no longer upstream's grader.
#
# LiveCodeBench's license (LICENSE at that commit), which covers the code below and the
# lcb_runner code above:
#
#   MIT License
#
#   Copyright (c) 2024 LiveCodeBench
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

def _vendored_constant(source: str, name: str) -> Any:
    """A module-level literal from vendored source, read without executing it."""
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == name):
            return ast.literal_eval(node.value)
    raise KeyError(name)


#: lcb_runner/evaluation/testing_util.py (run_test, grade_stdio, grade_call_based, ...).
_LCB_TESTING_UTIL = r'''import ast
import json
import sys
import faulthandler
import platform

# used for debugging to time steps
from datetime import datetime

# to run the solution files we're using a timing based approach
import signal

import numpy as np

from io import StringIO

# used for testing the code that reads from input
from unittest.mock import patch, mock_open

# from pyext import RuntimeModule
from types import ModuleType

from enum import Enum
from decimal import Decimal
import time

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# Custom mock for sys.stdin that supports buffer attribute
class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        # Delegate other attributes to StringIO
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")  # Convert to bytes

    def read(self, *args):
        # Return as byte strings that can be split
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception as e:
        return code


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # Create custom stdin mock with buffer support
    mock_stdin = MockStdinWithBuffer(inputs)

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)  # Use our custom mock instead of StringIO
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    # call-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)

    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)

    if method is None:
        return

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]

    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            # handle floating point comparisons

            all_results.append(tmp_result)

            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }

        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
):
    ## runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    ## we wrap the given code inside another function
    code = make_function(code)

    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        return

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_send_args

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )

            ## CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            ## CASE 2: element-wise comparision
            ## if there are floating elements
            ## use `decimal` library for good floating point comparision
            ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            ## note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(
                stripped_prediction_line
            )
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args

            if decimal_prediction_line == decimal_gtout_line:
                continue

            all_results.append(-2)
            return all_results, WA_send_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(sample, test=None, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    signal.signal(signal.SIGALRM, timeout_handler)

    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError as e:
        raise e
        in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None

        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        assert False, "should not happen: test code is none"
        return in_outs, {"error": "no test code provided"}
    elif test is not None:
        results = []
        sol = import_string
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            signal.alarm(timeout)
            try:
                results, metadata = grade_call_based(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    fn_name=method_name,
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                signal.alarm(0)
        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it

            signal.alarm(timeout)
            try:
                results, metadata = grade_stdio(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                signal.alarm(0)


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

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    # builtins.exit = None
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
    sys.modules["tkinter"] = None
'''

#: lcb_runner/evaluation/utils_execute.py (BASE_IMPORTS, unsafe_execute, ...).
_LCB_UTILS_EXECUTE = r'''# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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


BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
# from sortedcontainers import SortedList, SortedDict, SortedSet
# import sortedcontainers
from operator import iand
import sys
"""

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

#: Prepended to every code-execution check program (compute_code_execution_metrics).
_LCB_BASE_IMPORTS = _vendored_constant(_LCB_UTILS_EXECUTE, "BASE_IMPORTS")

#: lcb_runner/evaluation/compute_test_output_prediction_metrics.py, the two scoring functions
#: (parse_assert_statement, check_testcase_output) with the module's own imports.
_LCB_TEST_OUTPUT_PREDICTION = r'''import ast
import json


def parse_assert_statement(statement):
    """
    Parse a Python assert statement and extract the expected output
    from the right side of the '==' operator as a string.

    :param statement: A string containing the assert statement.
    :return: The expected output from the assert statement as a string.
    """
    try:
        parsed = ast.parse(statement, mode="exec")
    except SyntaxError:
        return "Invalid syntax"

    if len(parsed.body) == 0:
        return "Empty statement"

    if not isinstance(parsed.body[0], ast.Assert):
        return "Not an assert statement"

    comparison = parsed.body[0].test

    if not isinstance(comparison, ast.Compare) or not isinstance(
        comparison.ops[0], ast.Eq
    ):
        return "Not an equality assertion"

    # Extract and return the right side of the '==' operator as a string
    return ast.get_source_segment(statement, comparison.comparators[0])


def check_testcase_output(testcase_str, expected_output):

    if len(testcase_str.splitlines()) > 1:
        for line in testcase_str.splitlines():
            if line.startswith("#"):
                continue
            if "assert" in line:
                testcase_str = line
                break

    testcase_str = testcase_str.strip()

    if "assert" in testcase_str:
        testcase_output_str = str(parse_assert_statement(testcase_str))

    else:
        testcase_output_str = testcase_str

    global_result = None

    try:
        testcase_output_eval = eval(testcase_output_str)
    except:
        global_result = False
        # print("Failed to eval testcase output", testcase_output_str)
        # breakpoint()

    try:
        expected_output_eval = json.loads(expected_output)
    except:
        global_result = False
        print("Failed to eval expected testcase output", expected_output)

    if global_result is None:
        global_result = testcase_output_eval == expected_output_eval

    return global_result
'''
