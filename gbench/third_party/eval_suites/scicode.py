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

# Evaluation Suite: scicode
# Description: SciCode (Scientific Python Problem Solving & Numerical Code Execution Benchmark)

"""gbench native built-in runner for scicode (Coding & Algorithmic).

Faithful port of SciCode's **sequential, per-sub-step** protocol
(scicode-bench/SciCode `eval/scripts/gencode.py` + `test_generated_code.py`):

Generation
    A problem has N sub-steps. For step k we build a prompt containing every
    PRIOR step's description and the model's OWN generated code for that step
    (separated by "------"), followed by step k's description, function header
    and return line, plus the shared dependency block. The model writes only
    step k's function; its code is fed forward as context for step k+1. This is
    N model calls per problem, not a single whole-problem shot.

Scoring
    Each generated step is executed on its own: dependencies + all prior steps'
    code + this step's code, then the canonical `target` bindings
    (`process_hdf5_to_tuple(step_number, n, test_data.h5)`) and the step's test
    cases. A step passes iff its process exits 0. A PROBLEM is correct iff every
    scored sub-step passes.

Metrics
    The headline `accuracy` is the **main-problem resolve rate** (correct problems /
    problems), the number the SciCode leaderboard ranks by; **sub-step accuracy**
    (correct sub-steps / scored sub-steps) is reported alongside, as the leaderboard's
    "Subproblem" column. A full run covers upstream's fixed test split
    (test_generated_code.py: 65 problems, 288 sub-steps). Harness errors are never scored
    as the model's outcome: a sub-step whose generation got no reply from the served
    endpoint (after the retries), the sub-steps after it (built on its code, so not
    generated), a sub-step whose test script was never started (the sandbox did not start, or
    the script, the interpreter or test_data.h5 is not there inside it: no verdict; a script is
    run from a file, as upstream runs it, so its size is no limit, in a directory bound into the
    sandbox), and every sub-step of a problem whose processing crashed are left out of
    the sub-step accuracy, and their problem out of the problem accuracy; they are counted
    (`failed_generations`, `steps_not_generated`, `scoring_failed_steps`, `errored_problems`,
    `harness_error_problems`), named in `status_reason` (status `completed_with_errors`),
    and make the run non-comparable. With no problem scored, `accuracy` is null with
    `accuracy_withheld_reason`. A context-window rejection of a sub-step prompt is not
    retried: when the prompt carries the model's own code of earlier sub-steps (a runaway reply
    fed forward), the step and the steps after it fail and the problem stays in the score, as
    upstream's gencode.py (no handling) and fixed denominators make it (`over_context_steps`,
    `steps_not_generated_after_over_context`, each trace's `over_context` attribution); a prompt
    built from the dataset alone that the window cannot hold is a harness error (kind
    `over_context`).

Sampling:
    gbench's rule for every suite (base.resolve_temperature): temperature 0.0 without
    --thinking (also SciCode's own default: eval/scripts/gencode.py `--temperature`, default
    0; OFFICIAL_SAMPLING), 1.0 with it. `--temperature`, or `GBENCH_SCICODE_TEMPERATURE` for
    this suite alone (it takes precedence), overrides either; the result records SciCode's 0
    under `official_sampling`, and only a temperature-0 run is reported as
    leaderboard_comparable. No max_tokens is sent, as SciCode sends none for a
    `litellm/*` model, the route its eval/scripts README gives for vLLM (src/scicode/gen/
    models.py generate_litellm_response; only its Anthropic route sets 4096): the server
    grants the rest of the context window, and `--max-output-tokens` is recorded as not
    applied (`max_output_tokens_note`). A sub-step reply that reaches the end of the window
    is counted in `replies_cut_off_by_budget` (those with no answer text also in
    `replies_cut_off_in_reasoning`, those repeating themselves in `replies_cut_off_looping`)
    and makes the run not leaderboard_comparable.
    `--thinking` toggles the model's native reasoning
    channel (chat-template kwarg `enable_thinking`). Set `GBENCH_SCICODE_WITH_BACKGROUND=1`
    to inject the human-written step background (canonical `--with-background`); the
    default is the without-background protocol the leaderboard uses.

Selection (`--shard I/N`, `--eval-limit`): the problems are ordered by problem_id, the shard is
taken first (base.apply_shard_indexed, round-robin over that order), then `--eval-limit` problems
are drawn from it (sampling.stratified_sample, seed "scicode"); every sub-step of a selected
problem runs (its steps build on each other). The result records the shard identity
(base.shard_identity) and a partial shard's comparability reason; each sub-step trace carries
`sample_key` (problem and step) and `global_idx` (its problem's position in the ordered list). An
empty shard runs nothing and reports `accuracy` null with `accuracy_withheld_reason`.
merge_shards (gbench --merge) re-counts the problem resolve rate and the sub-step accuracy over
the union of the shards' traces. SciCode generates each sub-step once: `--attempt-count` above 1
is recorded as not applied (`attempts.not_applied`).

The endpoint key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY) is sent by gbench's
own OpenAI client; the scored code runs in the sandbox without any credential
(sandbox.credential_free_env).

Note on validation: SciCode withholds gold (`ground_truth_code` /
`general_solution` are empty in the HF dataset), so there is no gold self-check.
Validate the wiring with a capable model - scores should be non-zero and track
difficulty; an all-zero run means the h5/keying/plumbing is wrong, not the model.
"""

import ast
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from .base import (REPETITION_THRESHOLD, SHARD_REASON, UNCAPPED_TIMEOUT_TOKENS, OfficialSampling,
                   add_comparability_reason, apply_shard_indexed, attribute_endpoint_rejection,
                   current_shard, degenerate_char_run, endpoint_api_key, endpoint_harness_error,
                   get_run_knob, is_context_window_rejection, looping_channel, repetition_run,
                   request_timeout_s, resolve_official_sampling, resolve_token_budget,
                   shard_identity, suite_env)
from .metrics import budget_cutoffs
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample
from .sandbox import run_sandboxed, sandbox_mode, sandbox_skip_reason
from .swe_thread_cap import THREAD_VARS
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/scicode.md"

#: SciCode's generation default: eval/scripts/gencode.py `--temperature` (default 0), passed to
#: every model call (scicode-bench/SciCode@e3158ea0).
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("github.com/scicode-bench/SciCode@e3158ea011d4235245a547460d3688d7ccbf9900 "
               "eval/scripts/gencode.py: --temperature default 0; src/scicode/gen/models.py "
               "generate_litellm_response sends no max_tokens"),
    temperature=0.0, send_max_tokens=False)

PILLAR = "Coding & Algorithmic"

# --------------------------------------------------------------------------- #
# Canonical prompt templates (verbatim from scicode-bench/SciCode eval/data/,  #
# Apache-2.0). NOTE the upstream naming: `--with-background` uses the          #
# multistep template (human background is injected into the step text), while  #
# the default (without background) uses the background-comment template (the   #
# model must generate its own "Background:" comment).                          #
# --------------------------------------------------------------------------- #
_TEMPLATE_MULTISTEP = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""

_TEMPLATE_BG_COMMENT = """PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
"""

#: Sub-steps SciCode supplies directly (they depend on prior human scaffolding),
#: keyed by step_number. Not generated, not scored; injected as context for the
#: later steps of their problems. Bytes vendored under scicode_data/ (see the
#: ATTRIBUTION there).
_SPECIAL_STEP_IDS = ("13.6", "62.1", "76.3")
#: HF commit of SciCode1/SciCode the problems are read from.
_DATASET_REVISION = "4510f6a6aa27c43fad7b43da2c59602a86e88480"
#: Upstream's fixed test-split denominators (test_generated_code.py: PROB_NUM - DEV_PROB_NUM,
#: STEP_NUM). 291 sub-steps minus the 3 supplied ones = 288 scored.
_TEST_PROBLEMS = 65
_TEST_STEPS = 288
_SCICODE_DATA_DIR = os.path.join(os.path.dirname(__file__), "scicode_data")


def _load_special_gold(step_number: str) -> Optional[str]:
    if step_number not in _SPECIAL_STEP_IDS:
        return None
    path = os.path.join(_SCICODE_DATA_DIR, f"{step_number}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("scicode: could not read vendored gold %s (%s)", path, e)
        return None


#: Third-party mirror of SciCode's reference-output HDF5. The canonical source is a Google
#: Drive folder linked from the SciCode README, which is not scriptable without gdown/auth,
#: and the HF dataset (`SciCode1/SciCode`) ships only the problem JSONLs. Override with
#: GBENCH_SCICODE_TEST_DATA (a local path) or GBENCH_SCICODE_TEST_DATA_REPO (a different
#: mirror). The bare SCICODE_TEST_DATA / SCICODE_TEST_DATA_REPO still work as deprecated aliases.
_TEST_DATA_DEFAULT_REPO = "Srimadh/Scicode-test-data-h5"
_TEST_DATA_REPO = suite_env("GBENCH_SCICODE_TEST_DATA_REPO", "SCICODE_TEST_DATA_REPO",
                            default=_TEST_DATA_DEFAULT_REPO)
_TEST_DATA_FILE = "test_data.h5"
#: The default mirror is read at this pinned commit, and the file must have this sha256 (its
#: LFS digest at that commit, checked 2026-09-24 against the HF API and a local download).
_TEST_DATA_REVISION = "72c247d3a8410921b2e848e046d71ed63d9a0ddb"
_TEST_DATA_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"


def resolve_test_data() -> Tuple[Optional[str], Optional[str]]:
    """Path to SciCode's `test_data.h5`, fetching it if needed. Returns (path, provenance).

    Every SciCode test binds its expected values as `target` from this file
    (`process_hdf5_to_tuple`); without it every test raises NameError and even a perfect
    solution scores 0 - a structural zero, not a model result. It is ~1 GB and is fetched
    at eval time like any other dataset, then cached by huggingface_hub.
    """
    local = suite_env("GBENCH_SCICODE_TEST_DATA", "SCICODE_TEST_DATA")
    if local:
        return (local, "GBENCH_SCICODE_TEST_DATA") if os.path.exists(local) else (None, None)
    # The default mirror is pinned to a commit (and its digest is checked in run_scicode). A
    # user-chosen mirror is read at its HEAD; its digest decides whether the run is comparable.
    revision = _TEST_DATA_REVISION if _TEST_DATA_REPO == _TEST_DATA_DEFAULT_REPO else None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_TEST_DATA_REPO, filename=_TEST_DATA_FILE,
                               repo_type="dataset", revision=revision)
    except Exception as e:
        logger.warning("scicode: could not fetch %s/%s (%s)", _TEST_DATA_REPO,
                       _TEST_DATA_FILE, e)
        return None, None
    # The canonical file lives on Google Drive; the default mirror is pinned and digest-checked,
    # but it has not been compared with the Drive original.
    logger.warning(
        "scicode: using the third-party mirror %s%s for %s. The canonical copy is the Google "
        "Drive folder linked from the SciCode README. Set GBENCH_SCICODE_TEST_DATA to use your "
        "own copy.", _TEST_DATA_REPO, f"@{revision[:8]}" if revision else " (HEAD)",
        _TEST_DATA_FILE)
    return path, (f"hf:{_TEST_DATA_REPO}@{revision}/{_TEST_DATA_FILE}" if revision
                  else f"hf:{_TEST_DATA_REPO}/{_TEST_DATA_FILE} (HEAD)")


def check_test_data_digest(path: str, provenance: str) -> Tuple[str, Optional[str]]:
    """(sha256, non-comparable reason or None) for the reference file.

    The default mirror at its pinned commit MUST have the pinned digest; anything else there is
    a hard error (a changed reference file silently changes every score). A file the user
    supplied (GBENCH_SCICODE_TEST_DATA, or another GBENCH_SCICODE_TEST_DATA_REPO) is allowed, but
    when its digest differs the run is not leaderboard-comparable and the reason names it.
    """
    digest = _digest(path)
    if digest == _TEST_DATA_SHA256:
        return digest, None
    if provenance.startswith(f"hf:{_TEST_DATA_DEFAULT_REPO}@"):
        raise infra_required(
            "scicode",
            f"{_TEST_DATA_FILE} from the pinned mirror {_TEST_DATA_DEFAULT_REPO}@"
            f"{_TEST_DATA_REVISION[:8]} has sha256 {digest}, expected {_TEST_DATA_SHA256}. The "
            f"cached copy ({path}) is corrupt or was replaced; delete it so it is fetched again, "
            "or set GBENCH_SCICODE_TEST_DATA to a verified copy",
            DOCS_URL)
    source = provenance if provenance.startswith("hf:") else f"{provenance}={path}"
    return digest, (f"unverified reference file: {_TEST_DATA_FILE} from {source} has sha256 "
                    f"{digest[:16]}..., not the pinned {_TEST_DATA_DEFAULT_REPO}@"
                    f"{_TEST_DATA_REVISION[:8]} digest {_TEST_DATA_SHA256[:16]}...")


def _digest(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# SciCode's test cases import `scicode.compare.cmp` (90 of them in the test split, over 31
# scored sub-steps of 20 problems: `from scicode.compare.cmp import cmp_tuple_or_list` or
# `are_dicts_close`), and upstream runs them where the `scicode` package is installed. That module
# (and the two package __init__ files above it) is vendored verbatim from SciCode@e3158ea0 in
# _scicode_vendor/, which every scored sub-step gets on its PYTHONPATH; run_scicode checks before
# generation that the sandbox imports it. SCICODE_CMP_SHA256 is the upstream file's sha256.
SCICODE_VENDOR_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "_scicode_vendor"))
SCICODE_CMP_UPSTREAM = ("github.com/scicode-bench/SciCode@e3158ea011d4235245a547460d3688d7ccbf9900 "
                        "src/scicode/compare/cmp.py")
SCICODE_CMP_SHA256 = "8b3e790cc29dc87ef1f0f1b9bfd68c877728c78c40b541a69d10015221e9e235"
#: What the vendored module needs (cmp.py imports numpy, scipy.sparse and sympy).
_CMP_DEPS = ("numpy", "scipy", "sympy")
#: The distribution that provides an import name, where the two differ (for the install hint).
_DEPENDENCY_DISTRIBUTIONS = {"mpl_toolkits": "matplotlib"}


def dependency_modules(problems: List[Dict[str, Any]]) -> List[str]:
    """Every module the selected problems' `required_dependencies` import (the import lines the
    dataset prepends to a problem's code), dotted names as written, sorted. A line that does not
    parse is left to the scored code, which runs it as upstream does."""
    modules: Set[str] = set()
    for prob in problems:
        text = prob.get("required_dependencies") or ""
        try:
            tree = ast.parse(text)
        except SyntaxError:
            logger.warning("scicode: problem %s: required_dependencies do not parse; not checked",
                           prob.get("problem_id"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules.add(node.module)
    return sorted(modules)


def missing_dependency_modules(modules: List[str]) -> List[str]:
    """The modules of `modules` this interpreter (the one every scored sub-step runs with)
    cannot import."""
    import importlib.util
    missing = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def scoring_env(n_threads: str = "1") -> Dict[str, str]:
    """The environment of a scored sub-step: this process's, the BLAS/OpenMP thread cap, and
    the vendored `scicode` package first on PYTHONPATH."""
    env = {**os.environ, **{v: n_threads for v in THREAD_VARS}}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (SCICODE_VENDOR_DIR, os.environ.get("PYTHONPATH")) if p)
    return env


def check_cmp_importable() -> Optional[str]:
    """Why the sandbox cannot import the test cases' `scicode.compare.cmp`, or None when it can
    (the import runs in the same sandbox and environment as a scored sub-step)."""
    probe = ("from scicode.compare.cmp import are_dicts_close, cmp_tuple_or_list\n"
             "import scicode.compare.cmp as m\n"
             f"assert m.__file__.startswith({SCICODE_VENDOR_DIR!r}), m.__file__\n")
    try:
        res = run_sandboxed([sys.executable, "-c", probe], capture_output=True, text=True,
                            timeout=120, env=scoring_env(), readable=[SCICODE_VENDOR_DIR])
    except Exception as e:                                              # noqa: BLE001
        return f"the import probe could not run ({type(e).__name__}: {e})"
    if res.returncode != 0:
        return ("the sandbox could not import the vendored scicode.compare.cmp from "
                f"{SCICODE_VENDOR_DIR}: " + ((res.stderr or "").strip().splitlines() or ["?"])[-1])
    return None


# SciCode's `target` reference outputs are read from test_data.h5 by the canonical
# `process_hdf5_to_tuple`. The `scicode` PyPI/git package pulls an invasively broad dependency
# tree (numpy 2.x, litellm, inspect-ai, aioboto3, ...) that conflicts with gbench's pinned stack,
# so its h5 loader is vendored verbatim as the module upstream's test script imports it from,
# _scicode_vendor/scicode/parse/parse.py (h5py + scipy + numpy, already in the eval env), on every
# scored sub-step's PYTHONPATH like the comparison helpers (scicode.compare.cmp).
H5_LOADER_IMPORT = "from scicode.parse.parse import process_hdf5_to_tuple"
#: The most characters of a sub-step's prompt a sample trace keeps.
_TRACE_PROMPT_CHARS = 8000


# --------------------------------------------------------------------------- #
# Generation: canonical sequential per-sub-step protocol.                      #
# --------------------------------------------------------------------------- #
def _extract_python_script(response: Optional[str]) -> str:
    """Port of scicode.gen.models.extract_python_script.

    Pull the code out of a ```python``` (or bare ```) fence, then strip any
    import lines (dependencies are supplied separately and prepended). Falls back
    to the raw text if no fence is present. Robust to malformed fences.
    """
    if not response:
        return ""
    try:
        if "```" in response:
            if "```python" in response:
                python_script = response.split("```python")[1].split("```")[0]
            else:
                python_script = response.split("```")[1].split("```")[0]
        else:
            python_script = response
    except IndexError:
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '',
                           python_script, flags=re.MULTILINE)
    return python_script


def _extract_function_name(function_header: str) -> str:
    """Vendored from scicode.parse.parse.extract_function_name (Apache-2.0).

    Canonical quirk preserved: the `def` pattern is tried BEFORE `class`, so a
    header like `class Maxwell:\\n    def __init__(...)` resolves to `__init__`
    (not `Maxwell`). Kept verbatim so the supplied-step context matches upstream.
    """
    pattern = r'\bdef\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    pattern = r'\bclass\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    raise ValueError('Function name or class name not found.')


def _get_function_from_code(code_string: Optional[str], function_name: str) -> Optional[str]:
    """Vendored from scicode.parse.parse.get_function_from_code (Apache-2.0).

    Returns the AST-unparsed source of the FIRST def/class whose name matches -
    so a class's `__init__` comes back WITHOUT its enclosing class, which is
    exactly how canonical forwards the supplied steps for problems 13 and 62.
    None if no node matches; the raw string on parse error (canonical behavior).
    """
    if code_string is None:
        return None
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception as e:  # noqa: BLE001 - mirror canonical: raw string on parse error
        logger.debug("scicode: get_function_from_code(%s) failed: %s", function_name, e)
        return code_string
    return None


def _process_problem_code(sub_steps: List[Dict[str, Any]], num_steps: int) -> str:
    header = sub_steps[num_steps - 1].get("function_header") or ""
    return_line = sub_steps[num_steps - 1].get("return_line") or ""
    return f"{header}\n\n{return_line}"


def _process_problem_steps(
    sub_steps: List[Dict[str, Any]],
    num_steps: int,
    with_background: bool,
    previous_llm_code: List[str],
) -> Tuple[str, str, str]:
    """Return (problem_steps_str, next_step_str, previous_code_str) for step `num_steps`.

    Mirrors Gencode.process_problem_steps: prior steps contribute their
    description (+ background when enabled) and generated code, separated by
    "------"; the next step contributes its description and header/return line.
    """
    output_lines: List[str] = []
    previous_code: List[str] = []
    for i in range(num_steps - 1):
        desc = sub_steps[i].get("step_description_prompt") or ""
        if with_background:
            desc = desc + "\n" + (sub_steps[i].get("step_background") or "")
        output_lines.append(desc)
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")

    next_desc = sub_steps[num_steps - 1].get("step_description_prompt") or ""
    if with_background:
        next_desc = next_desc + "\n" + (sub_steps[num_steps - 1].get("step_background") or "")
    next_step = [next_desc, _process_problem_code(sub_steps, num_steps)]

    problem_steps_str = "\n\n".join(output_lines[:-1]) if output_lines else ""
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)
    return problem_steps_str, next_step_str, previous_code_str


def _build_step_prompt(
    prob: Dict[str, Any],
    num_steps: int,
    with_background: bool,
    previous_llm_code: List[str],
) -> str:
    template = _TEMPLATE_MULTISTEP if with_background else _TEMPLATE_BG_COMMENT
    problem_steps_str, next_step_str, _ = _process_problem_steps(
        prob["sub_steps"], num_steps, with_background, previous_llm_code)
    dependencies = prob.get("required_dependencies") or ""
    # .format is safe: only the template carries {placeholders}; substituted
    # values may freely contain braces (code, f-strings) - they aren't re-parsed.
    return template.format(
        problem_steps_str=problem_steps_str,
        next_step_str=next_step_str,
        dependencies=dependencies,
    )


def _loops(text: Optional[str], reasoning: Optional[str]) -> bool:
    """True when the reply's content, or its reasoning when only that repeats, is a repetition
    loop (base.looping_channel)."""
    looped, _ = looping_channel(text, reasoning)
    return (repetition_run(looped) >= REPETITION_THRESHOLD
            or degenerate_char_run(looped) is not None)


def _complete(client, model: str, prompt: str, temperature: float,
              max_tokens: Optional[int], thinking: bool,
              retries: int = 5) -> Optional[Dict[str, Any]]:
    """One chat call (single-turn agentic pattern).

    Returns a dict {text, reasoning, finish_reason, completion_tokens}; after exhausting the
    retries, {text: None, error, timed_out} (no reply from the served endpoint: a harness
    error). `reasoning` is the model's thinking channel (vLLM's gemma4 reasoning
    parser separates it into `reasoning_content`; some builds use `reasoning`) - captured so
    the trace records BOTH the thinking content and its length, and so a truncated
    (finish_reason=="length") turn is distinguishable from a wrong answer.
    """
    import time
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if max_tokens is not None:              # None: no max_tokens, as SciCode sends none
        kwargs["max_tokens"] = max_tokens
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            msg = choice.message
            reasoning = (getattr(msg, "reasoning_content", None)
                         or getattr(msg, "reasoning", None) or "")
            usage = getattr(resp, "usage", None)
            return {
                "text": msg.content or "",
                "reasoning": reasoning,
                "finish_reason": getattr(choice, "finish_reason", None),
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
        except Exception as e:  # noqa: BLE001 - endpoint/transport errors are retried
            status = getattr(e, "status_code", None)
            if is_context_window_rejection(status, f"{getattr(e, 'body', '') or ''} {e}"):
                # The same prompt is refused the same way every time: no retry. The caller
                # decides whose outcome it is (the prompt may carry the model's own code).
                logger.warning("scicode: the endpoint refused the prompt as longer than its "
                               "context window (HTTP %s): %s", status, e)
                return {"text": None, "error": f"{type(e).__name__}: {e}",
                        "timed_out": False, "over_context": True, "status": status}
            if attempt == retries - 1:
                logger.warning("scicode: generation call failed after %d tries: %s",
                               retries, e)
                return {"text": None, "error": f"{type(e).__name__}: {e}",
                        "timed_out": "Timeout" in type(e).__name__}
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return None


#: How a sub-step's script is started. A POSIX shell checks that the interpreter, the script and
#: the data file it reads are there where the script runs (inside the sandbox's filesystem), prints
#: STEP_STARTED as the first line of stdout, then replaces itself with `python <file>` (exec: the
#: process, its argv, sys.path[0] and exit status are those of upstream's `python <file>`). A run
#: whose stdout does not begin with that line never reached the script's code: the sandbox did not
#: start, or a file the step needs is missing in it. That is no verdict (a harness error), never
#: the model's failed step.
STEP_STARTED = "gbench-scicode-step-started"
_STEP_LAUNCH = (
    'py=$1; script=$2; shift 2; '
    '[ -x "$py" ] || { echo "not found where the step runs: $py" >&2; exit 125; }; '
    'for f in "$script" "$@"; do '
    '[ -r "$f" ] || { echo "not found where the step runs: $f" >&2; exit 125; }; done; '
    f'echo {STEP_STARTED}; exec "$py" "$script"')


def _step_started(stdout: Any) -> Tuple[bool, str]:
    """(the launcher's STEP_STARTED line opens `stdout`, the program's own stdout after it)."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    stdout = stdout or ""
    head = STEP_STARTED + "\n"
    return (True, stdout[len(head):]) if stdout.startswith(head) else (False, stdout)


def _not_started(returncode: Optional[int], stdout: str, stderr: str,
                 detail: str) -> Dict[str, Any]:
    return {"correct": None, "returncode": returncode, "stdout_tail": (stdout or "")[-1200:],
            "stderr_tail": (stderr or "")[-1200:],
            "harness_error": {"kind": "scoring_failed", "detail": detail[:300]}}


def _score_step(assembled_code: str, step_number: str, test_cases: List[str],
                h5: str, timeout: int) -> Dict[str, Any]:
    """Execute one assembled sub-step against its h5-bound targets, as upstream's
    test_generated_code.py does: the assembled code, then `from scicode.parse.parse import
    process_hdf5_to_tuple`, the targets and the test cases, written to a script file that is run
    with `python <file>` (in gbench's sandbox, which binds the script's directory).

    Returns {correct, returncode, stdout_tail, stderr_tail} - the execution detail is kept
    for the trajectory so a failed step is readable (traceback / assertion) instead of a bare
    boolean. `correct` is True iff the script exited 0; a script that ran past `timeout` is a
    failed step, as upstream counts it. A script that was never started (the sandbox did not
    start, the script, the interpreter or `h5` is not there inside it, an OS error; see
    _STEP_LAUNCH) is no verdict: `correct` None with `harness_error`, left out of the score.
    """
    lines = [assembled_code, H5_LOADER_IMPORT]
    lines.append(f"targets = process_hdf5_to_tuple({step_number!r}, {len(test_cases)}, {h5!r})")
    for i, tc in enumerate(test_cases):
        lines.append(f"target = targets[{i}]")
        lines.append(str(tc))
    script = "\n".join(lines) + "\n"
    # Cap BLAS/OpenMP threads PER scoring subprocess. numpy/scipy otherwise spawn one thread
    # per core, so at high --sandboxes (e.g. 48) the box runs 48 x N-core processes and
    # thrashes on oversubscription (same pathology as swe_thread_cap). Parallelism comes from
    # running many problems at once, not from threading one step; GBENCH_SCICODE_THREADS
    # raises the per-process cap if a genuinely heavy step needs it.
    # The vendored `scicode` package (the test cases' `scicode.compare.cmp`) is on PYTHONPATH.
    n_threads = os.environ.get("GBENCH_SCICODE_THREADS", "1")
    env = scoring_env(n_threads)
    # A file, not `python -c`: a program over the kernel's single-argument limit (128 KiB)
    # cannot be passed on the command line. The jail replaces /tmp (where mkdtemp puts the
    # directory) with an empty one, so the directory is bound into it (`writable`, as codeforces
    # binds its work directory); it holds only this script and is removed afterwards.
    script_dir = tempfile.mkdtemp(prefix="gbench_scicode_step_")
    script_path = os.path.join(script_dir, f"{re.sub(r'[^0-9A-Za-z_.]', '_', step_number)}.py")
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            res = run_sandboxed(
                ["sh", "-c", _STEP_LAUNCH, "scicode-step", sys.executable, script_path, h5],
                capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
                env=env, writable=[script_dir], readable=[SCICODE_VENDOR_DIR, h5],
            )
        except subprocess.TimeoutExpired as e:
            started, out = _step_started(e.stdout)
            err = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) \
                else (e.stderr or "")
            if not started:
                return _not_started(None, out, err, f"the sandbox did not start the sub-step's "
                                                    f"script within {timeout}s")
            # the step ran past its time: a failed step
            return {"correct": False, "returncode": None, "timed_out": True,
                    "stdout_tail": out[-1200:],
                    "stderr_tail": "\n".join(
                        x for x in (err.rstrip(), f"TimeoutExpired: the sub-step ran past its "
                                                  f"{timeout}s limit") if x)[-1200:]}
        started, out = _step_started(res.stdout)
        if not started:
            why = ((res.stderr or "").strip().splitlines() or ["no output"])[-1]
            return _not_started(res.returncode, out, res.stderr or "",
                                f"the sub-step's script was not started (exit "
                                f"{res.returncode}): {why}")
        return {"correct": res.returncode == 0, "returncode": res.returncode,
                "stdout_tail": out[-1200:],
                "stderr_tail": (res.stderr or "")[-1200:]}
    except Exception as e:  # noqa: BLE001 - the script could not run at all: no verdict
        return _not_started(None, "", f"{type(e).__name__}: {e}", f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(script_dir, ignore_errors=True)


def check_step_sandbox(h5: str) -> Optional[str]:
    """Why the sandbox cannot run a sub-step's script the way a scored one runs (the script file,
    the interpreter, the vendored h5 loader and `h5` inside the jail), or None when it can: a
    trivial sub-step that binds no test and only opens `h5` must pass. Under GBENCH_SANDBOX=none
    there is no jail: the steps run on this host, where `h5` was found."""
    if sandbox_mode() == "none":
        return None
    out = _score_step("pass", "sandbox_check", [], h5, 120)
    if out.get("correct"):
        return None
    detail = ((out.get("harness_error") or {}).get("detail")
              or ((out.get("stderr_tail") or "").strip().splitlines() or ["?"])[-1])
    return f"the sandbox could not run a trivial sub-step script that only opens {h5}: {detail}"


def _process_one_problem(
    client, model: str, prob: Dict[str, Any], *,
    temperature: float, max_tokens: Optional[int], thinking: bool,
    with_background: bool, h5: str, eval_timeout: int,
) -> Dict[str, Any]:
    """Generate every sub-step sequentially, then score each generated step."""
    sub_steps = prob["sub_steps"]
    n = len(sub_steps)
    deps = prob.get("required_dependencies") or ""
    previous_llm_code: List[str] = [""] * n
    # which earlier steps' code in the prompt the model wrote (not a supplied step's)
    model_wrote: List[bool] = [False] * n
    steps_report: List[Dict[str, Any]] = []

    for k in range(1, n + 1):
        step_number = str(sub_steps[k - 1].get("step_number") or f"{prob['problem_id']}.{k}")
        test_cases = list(sub_steps[k - 1].get("test_cases") or [])

        gold = _load_special_gold(step_number)
        if gold is not None:
            # Supplied step: use as context, never generate or score it. Match
            # canonical exactly - the header-named node via get_function_from_code
            # (so problem 13/62 forward a bare __init__, not the full class).
            header = sub_steps[k - 1].get("function_header") or ""
            func_name = _extract_function_name(header)
            previous_llm_code[k - 1] = _get_function_from_code(gold, func_name) or ""
            steps_report.append({
                "step_number": step_number, "special": True,
                "gen_failed": False, "correct": False, "prompt": None,
                "response_text": None, "reasoning": None, "reasoning_chars": 0,
                "finish_reason": None, "completion_tokens": None, "error": None,
            })
            continue

        prompt = _build_step_prompt(prob, k, with_background, previous_llm_code)
        result = _complete(client, model, prompt, temperature, max_tokens, thinking)
        gen_failed = result is None or result.get("text") is None
        if gen_failed and (result or {}).get("over_context") and any(model_wrote[:k - 1]):
            # The endpoint refused the prompt as longer than its context window, and the prompt
            # carries the model's own code of the earlier sub-steps (a runaway reply fed forward,
            # as upstream's gencode.py feeds it). Upstream has no handling for it and scores
            # against fixed denominators, so the step and the steps after it fail and the
            # problem stays in the score: the model's outcome, recorded as such.
            rejection = attribute_endpoint_rejection(
                result.get("status"), result.get("error"),
                upstream_outcome=("the sub-step's prompt, built from the model's own earlier "
                                  "code, exceeds the window: no code for it or the steps after "
                                  "it, which fail against the fixed denominators"),
                upstream_reference="SciCode eval/scripts/gencode.py and test_generated_code.py")
            steps_report.append({
                "step_number": step_number, "special": False,
                "gen_failed": False, "correct": False, "prompt": prompt,
                "response_text": None, "reasoning": "", "reasoning_chars": 0,
                "finish_reason": None, "completion_tokens": None,
                "error": "over_context", "execution": None,
                "over_context": {**rejection,
                                 "detail": str(result.get("error") or "")[:300]},
            })
            for j in range(k + 1, n + 1):
                later = str(sub_steps[j - 1].get("step_number") or f"{prob['problem_id']}.{j}")
                special = _load_special_gold(later) is not None
                steps_report.append({
                    "step_number": later, "special": special, "gen_failed": False,
                    "after_over_context": not special, "correct": False, "prompt": None,
                    "response_text": None, "reasoning": None, "reasoning_chars": 0,
                    "finish_reason": None, "completion_tokens": None, "execution": None,
                    "error": None if special else (
                        f"not generated: the prompt of sub-step {step_number} exceeded the "
                        "served context window with the model's own earlier code (the model's "
                        "outcome: a failed step)"),
                })
            break
        if gen_failed:
            # No reply from the served endpoint after the retries: a harness error, never the
            # model's outcome. The steps after it would be built on its missing code, so they
            # are not generated; the problem is left out of the score (see run_scicode).
            # A prompt gbench built from the dataset alone that the window cannot hold is a
            # harness error of its own kind (over_context: serve a larger window).
            why = str((result or {}).get("error") or "no reply")
            steps_report.append({
                "step_number": step_number, "special": False,
                "gen_failed": True, "correct": False, "prompt": prompt,
                "response_text": None, "reasoning": "", "reasoning_chars": 0,
                "finish_reason": None, "completion_tokens": None,
                "error": "generation_failed", "execution": None,
                "harness_error": endpoint_harness_error(
                    "OVER_CONTEXT" if (result or {}).get("over_context")
                    else "TIMEOUT" if (result or {}).get("timed_out") else "FAILED", why),
            })
            for j in range(k + 1, n + 1):
                later = str(sub_steps[j - 1].get("step_number") or f"{prob['problem_id']}.{j}")
                special = _load_special_gold(later) is not None
                steps_report.append({
                    "step_number": later, "special": special, "gen_failed": False,
                    "not_generated": not special, "correct": False, "prompt": None,
                    "response_text": None, "reasoning": None, "reasoning_chars": 0,
                    "finish_reason": None, "completion_tokens": None, "execution": None,
                    "error": None if special else (
                        f"not generated: sub-step {step_number} got no reply from the served "
                        "endpoint (harness error)"),
                })
            break
        response = None if gen_failed else result["text"]
        reasoning = "" if gen_failed else (result.get("reasoning") or "")
        finish_reason = None if gen_failed else result.get("finish_reason")
        completion_tokens = None if gen_failed else result.get("completion_tokens")
        code = _extract_python_script(response)
        previous_llm_code[k - 1] = code
        model_wrote[k - 1] = bool(str(code or "").strip())

        # Assembled program for this step = deps + prior steps' code + this step.
        prior = "\n".join(previous_llm_code[:k - 1])
        assembled = f"{deps}\n{prior}\n{code}"

        execution = None
        if gen_failed or not test_cases:
            correct = False
        else:
            execution = _score_step(assembled, step_number, test_cases, h5, eval_timeout)
            correct = bool(execution["correct"])
        scoring_failed = bool(execution and execution.get("harness_error"))

        steps_report.append({
            "step_number": step_number, "special": False,
            "gen_failed": gen_failed, "correct": correct,
            "prompt": prompt, "response_text": response,
            "reasoning": reasoning, "reasoning_chars": len(reasoning),
            "finish_reason": finish_reason, "completion_tokens": completion_tokens,
            "error": ("generation_failed" if gen_failed else
                      "scoring_failed" if scoring_failed else None),
            "execution": execution,   # None for special/gen-failed/no-test steps
            # the step's script could not be run at all: no verdict (a harness error)
            **({"scoring_failed": True, "harness_error": execution["harness_error"]}
               if scoring_failed else {}),
        })

    return {"problem_id": str(prob["problem_id"]),
            "problem_name": prob.get("problem_name"),
            "steps": steps_report}


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def _load_scicode_problems(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load full SciCode problems (all sub-step fields) from HF (SciCode1/SciCode, test)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("SciCode1/SciCode", split="test", revision=_DATASET_REVISION)
        rows = list(ds)
    except Exception as e:
        logger.error("Failed to load dataset for scicode: %s", e)
        raise RuntimeError(f"Could not load dataset for scicode: {e}") from e
    if not rows:
        raise RuntimeError("Dataset for scicode returned empty rows")

    # The shard first, over the problems ordered by problem_id (base.apply_shard_indexed), then
    # the --eval-limit draw within it: stratified, not a contiguous head. NOTE:
    # --eval-limit caps PROBLEMS, not scored items -- each problem expands into its dependent
    # sub-steps (which build on each other, so they can't be capped mid-problem): a problem is
    # one sample, and --eval-limit N runs N problems (eval_limit_plan counts their sub-steps).
    rows = sorted(rows, key=lambda r: str(r.get("problem_id") or ""))
    rows, _label, positions = apply_shard_indexed(rows, "scicode")
    for row, pos in zip(rows, positions):
        row["_global_idx"] = pos
    rows = stratified_sample(rows, limit, None, seed="scicode")

    problems: List[Dict[str, Any]] = []
    for item in rows:
        sub_steps = item.get("sub_steps") or []
        if not isinstance(sub_steps, list) or not sub_steps:
            # Dropping it would silently shrink the fixed denominators.
            raise RuntimeError(f"scicode: problem {item.get('problem_id')!r} has no sub_steps; "
                               "refusing to score a partial benchmark")
        problems.append({
            "problem_id": str(item.get("problem_id") or "").strip(),
            "problem_name": str(item.get("problem_name") or "Scientific Problem"),
            "required_dependencies": str(item.get("required_dependencies") or ""),
            "sub_steps": sub_steps,
            "global_idx": item.get("_global_idx"),
        })
    logger.info("Loaded %d scicode problems.", len(problems))
    return problems


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def run_scicode(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run SciCode with the canonical sequential per-sub-step protocol.

    Gated up front on the reference h5, the vendored loader's deps and a trivial sub-step run
    in the sandbox exactly as a scored one (check_step_sandbox) so a real run never reports a
    structural 0%. Generation is parallel across problems and
    sequential within each problem; scoring binds `target` from the h5.
    """
    # Executes model-written code: require bubblewrap isolation. Missing/blocked isolation is an
    # EXTERNAL prerequisite gap -> hard-error (no-skip), even when invoked directly (bypassing the
    # central evals.py gate). GBENCH_SANDBOX=none opts into unsandboxed execution.
    _blocked = sandbox_skip_reason()
    if _blocked:
        raise infra_required("scicode", _blocked, DOCS_URL)
    h5, provenance = resolve_test_data()
    if not h5:
        raise infra_required(
            "scicode",
            "SciCode tests evaluate against per-problem reference outputs bound as `target` "
            f"from the benchmark's test_data HDF5, and it could not be fetched from "
            f"{_TEST_DATA_REPO}. Set GBENCH_SCICODE_TEST_DATA to a local copy (the canonical file "
            "is the Google Drive folder linked from the SciCode README). Without it every "
            "test errors and the suite would report a structural 0%",
            DOCS_URL)
    # Before any generation: a wrong reference file on the default mirror stops the run here.
    h5_sha256, h5_reason = check_test_data_digest(h5, provenance)
    import importlib.util
    _missing = [p for p in ("h5py", *_CMP_DEPS) if importlib.util.find_spec(p) is None]
    if _missing:
        raise infra_required(
            "scicode",
            "SciCode scoring binds each test's `target` from test_data.h5 with SciCode's own "
            "reader (process_hdf5_to_tuple) and runs test cases that import SciCode's "
            "scicode.compare.cmp (both vendored in gbench); they require "
            f"{', '.join(_missing)}. Install: `pip install {' '.join(_missing)}` "
            "(see docs/evals/scicode.md).",
            DOCS_URL)
    _cmp_blocked = check_cmp_importable()
    if _cmp_blocked:
        raise infra_required(
            "scicode",
            f"{_cmp_blocked}. 90 test cases of the test split import it, so 31 scored sub-steps "
            "would fail on the import whatever the model wrote", DOCS_URL)
    # A trivial sub-step run exactly as a scored one (script file, interpreter, vendored loader,
    # test_data.h5, all inside the jail): a jail that cannot run it stops the run here, before
    # any generation, instead of leaving every sub-step without a verdict.
    _step_blocked = check_step_sandbox(h5)
    if _step_blocked:
        raise infra_required(
            "scicode",
            f"{_step_blocked}. Every scored sub-step runs this way, so none could be scored. The "
            "jail replaces /tmp and /run with empty directories: keep GBENCH_SCICODE_TEST_DATA "
            "outside them.", DOCS_URL)

    temperature, temp_source = resolve_official_sampling("scicode", OFFICIAL_SAMPLING,
                                                         thinking=enable_thinking)
    with_background = str(os.environ.get("GBENCH_SCICODE_WITH_BACKGROUND", "")).strip().lower() \
        in ("1", "true", "yes")
    # base's token-limit policy: SciCode sends no max_tokens (OFFICIAL_SAMPLING.send_max_tokens is
    # False) in either mode, so the server grants the rest of the window, and a run budget is
    # recorded as not applied.
    plan = resolve_token_budget(
        "scicode", OFFICIAL_SAMPLING, thinking=enable_thinking,
        max_output_tokens=kwargs.get("max_output_tokens") or get_run_knob("max_output_tokens"))
    max_tokens = plan.max_tokens
    # Canonical uses a 1800s per-step subprocess timeout (test_generated_code.py);
    # a tighter bound would false-fail slow numerical steps. Override via env.
    eval_timeout = int(suite_env("GBENCH_SCICODE_EVAL_TIMEOUT_S", "SCICODE_EVAL_TIMEOUT_S",
                                 default="1800"))
    workers = max(1, int(concurrency))

    shard = current_shard()
    problems = _load_scicode_problems(limit=limit)
    if not problems and shard is not None and shard.partial:
        # An empty shard (more shards than problems): nothing runs here, nothing is scored.
        reason = (f"empty shard {shard.label}: no items on this node ({_TEST_PROBLEMS} in the "
                  "full set)")
        logger.info("scicode: %s", reason)
        return {
            "benchmark_type": "eval", "eval_name": "scicode", "model_name": model_name,
            "thinking": enable_thinking, "status": "success", "status_reason": reason,
            "accuracy": None, "accuracy_withheld_reason": reason, "empty_shard": True,
            "full_set_items": _TEST_PROBLEMS, "total_questions": 0, "correct_answers": 0,
            "sample_traces": [], **shard_identity(shard),
            "leaderboard_comparable": False, "leaderboard_comparable_reason": reason,
        }
    if not problems:
        raise RuntimeError("scicode: no problem was selected from the test split, so nothing can "
                           "be scored (a harness failure, not a 0%)")
    # Before any generation: every module the selected problems' required_dependencies import
    # (numpy, scipy, sympy, mpl_toolkits.mplot3d, ...) must import in the interpreter the scored
    # code runs with, else those problems' sub-steps would fail for a reason that is not the
    # model's.
    _missing_deps = missing_dependency_modules(dependency_modules(problems))
    if _missing_deps:
        _dists = sorted({_DEPENDENCY_DISTRIBUTIONS.get(m.split(".")[0], m.split(".")[0])
                         for m in _missing_deps})
        raise infra_required(
            "scicode",
            "the selected problems' required_dependencies import "
            f"{', '.join(_missing_deps)}, which this interpreter ({sys.executable}), the one the "
            "scored code runs with, cannot import. Install: `pip install "
            f"{' '.join(_dists)}` (`pip install -e .[evals]` installs every one; see "
            "docs/evals/scicode.md).",
            DOCS_URL)

    # --eval-limit N is N problems for the whole suite (a problem is the dataset item; its
    # sub-steps are generated in order, once each, and are not more samples).
    limit_plan = eval_limit_plan("scicode", limit, len(problems), attempts=1,
                                 unit="problem rollouts")
    limit_plan["sub_step_generations_at_most"] = sum(len(p["sub_steps"]) for p in problems)
    # The global "Concurrency: N (from --batch-sizes)" startup line is NOT this suite's
    # concurrency: scicode is in SANDBOX_EVALS, so --sandboxes overrides it. Log the value
    # actually in effect so the parallelism is unambiguous.
    logger.info("scicode: %d problems | concurrency=%d | sandbox=%s | thinking=%s | "
                "temp=%.2f (%s) | max_tokens=%s/step | with_background=%s",
                len(problems), workers, sandbox_mode(), enable_thinking, temperature,
                temp_source, max_tokens, with_background)

    from openai import OpenAI
    model = model_name[len("openai/"):] if model_name.startswith("openai/") else model_name
    # With no max_tokens a reply may run to the end of the window, so the default per-request
    # timeout covers that (base.request_timeout_s over the served window, or base's
    # UNCAPPED_TIMEOUT_TOKENS when the window is unknown), as base's does.
    http_timeout = suite_env("GBENCH_SCICODE_HTTP_TIMEOUT_S", "SCICODE_HTTP_TIMEOUT_S", default="")
    # The endpoint key (VLLM_API_KEY, else OPENAI_API_KEY), sent by this process only.
    client = OpenAI(base_url=base_url, api_key=endpoint_api_key() or "EMPTY",
                    timeout=float(http_timeout) if http_timeout
                    else float(request_timeout_s(
                        max_tokens or get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)))

    results: List[Dict[str, Any]] = []
    errored_problems = 0
    by_id = {prob["problem_id"]: prob for prob in problems}
    with tqdm(total=len(problems), desc="scicode") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _process_one_problem, client, model, prob,
                    temperature=temperature, max_tokens=max_tokens,
                    thinking=enable_thinking, with_background=with_background,
                    h5=h5, eval_timeout=eval_timeout,
                ): prob["problem_id"]
                for prob in problems
            }
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    errored_problems += 1
                    logger.error("scicode: problem %s failed: %s", futures[fut], e)
                    results.append(_crashed_problem_report(by_id[futures[fut]], e))
                pbar.update(1)

    # Aggregate: problem accuracy (headline) + sub-step accuracy. Special steps
    # are excluded from both numerator and denominator (canonical: not generated).
    # Harness errors are left out of both, never scored as the model's outcome: a sub-step whose
    # generation got no reply from the served endpoint, the sub-steps after it (not generated),
    # and every sub-step of a problem whose processing crashed. A problem with any of them has no
    # verdict, so it is left out of the problem accuracy too; all are counted and named.
    total_steps = correct_steps = failed_gen = not_generated = steps_run = scoring_failed = 0
    total_problems = correct_problems = harness_error_problems = 0
    over_context_steps = after_over_context = 0
    sample_traces: List[Dict[str, Any]] = []
    gidx_of = {prob["problem_id"]: prob.get("global_idx") for prob in problems}
    idx = 0
    for pr in sorted(results, key=lambda r: str(r.get("problem_id"))):
        prob_all_ok = True
        prob_scored_any = False
        prob_harness_error = False
        for st in pr["steps"]:
            if st["special"]:
                continue
            steps_run += 1
            harness = bool(st.get("gen_failed") or st.get("not_generated") or st.get("crashed")
                           or st.get("scoring_failed"))
            if st.get("gen_failed"):
                failed_gen += 1
            if st.get("scoring_failed"):
                scoring_failed += 1
            if st.get("not_generated"):
                not_generated += 1
            if st.get("over_context"):
                over_context_steps += 1
            if st.get("after_over_context"):
                after_over_context += 1
            if harness:
                prob_harness_error = True
            else:
                total_steps += 1
                prob_scored_any = True
                if st["correct"]:
                    correct_steps += 1
                else:
                    prob_all_ok = False
            sample_traces.append({
                "sample_idx": idx,
                "category": "scientific_python",
                "problem_id": pr["problem_id"],
                "step_number": st["step_number"],
                "sample_key": f"step:{pr['problem_id']}/{st['step_number']}",
                **({"global_idx": gidx_of[pr["problem_id"]]}
                   if gidx_of.get(pr["problem_id"]) is not None else {}),
                "gold_answer": st["step_number"],
                "response_text": st["response_text"],
                "reasoning": st.get("reasoning"),           # thinking content
                "reasoning_chars": st.get("reasoning_chars", 0),  # thinking length
                "finish_reason": st.get("finish_reason"),
                "completion_tokens": st.get("completion_tokens"),
                "is_correct": bool(st["correct"]) and not harness,
                "status": "error" if harness else "ok",
                "error": st["error"],
                # the step's execution (return code, stdout/stderr tails) and its prompt (capped)
                "execution": st.get("execution"),
                **({"over_context": st["over_context"]} if st.get("over_context") else {}),
                "prompt": (st.get("prompt") or "")[:_TRACE_PROMPT_CHARS] or None,
                **({"harness_error": st.get("harness_error") or {
                    "kind": "problem_crashed" if st.get("crashed") else "not_generated",
                    "detail": str(st.get("error") or "")[:300]},
                    "scoring_excluded": True} if harness else {}),
            })
            idx += 1
        if prob_harness_error:
            harness_error_problems += 1
            continue
        total_problems += 1
        if prob_scored_any and prob_all_ok:
            correct_problems += 1

    substep_accuracy = round(100.0 * correct_steps / total_steps, 2) if total_steps else None
    problem_accuracy = (round(100.0 * correct_problems / total_problems, 2) if total_problems
                        else None)

    causes = []
    if failed_gen or errored_problems or scoring_failed:
        kinds = ", ".join(f"{label} {n}" for label, n in (
            ("sub-step generations with no reply", failed_gen),
            ("sub-steps not generated after one", not_generated),
            ("sub-steps whose script could not be run", scoring_failed),
            ("problems crashed", errored_problems)) if n)
        causes.append(f"harness errors left out of the score: {harness_error_problems} "
                      f"problem(s) ({kinds})")
    if errored_problems == len(problems) or total_problems == 0:
        status = "failed"
        status_reason = "nothing was scored: " + ("; ".join(causes) or "no sub-step was scored")
    elif causes:
        status = "completed_with_errors"
        status_reason = "; ".join(causes)
    else:
        status = "success"
        status_reason = None

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "scicode",
        "model_name": model_name,
        "thinking": enable_thinking,
        "status": status,
        # what made the status other than success (None on success)
        "status_reason": status_reason,
        # Headline = main-problem resolve rate over the problems with a verdict (None, withheld,
        # when none has one); total/correct kept consistent with it.
        "accuracy": problem_accuracy,
        **({"accuracy_withheld_reason": status_reason} if problem_accuracy is None else {}),
        "total_questions": len(results),
        "effective_n": total_problems,
        "correct_answers": correct_problems,
        # Secondary metrics (persisted to JSON; not surfaced in eval_summary.csv).
        "substep_accuracy": substep_accuracy,
        "problem_accuracy": problem_accuracy,
        "correct_steps": correct_steps,
        "total_steps": total_steps,
        "correct_problems": correct_problems,
        "total_problems": total_problems,
        "failed_generations": failed_gen,
        "steps_not_generated": not_generated,
        # sub-steps whose test script could not be run at all (no verdict)
        "scoring_failed_steps": scoring_failed,
        "errored_problems": errored_problems,
        # sub-steps whose prompt, carrying the model's own earlier code, the endpoint refused as
        # longer than its window, and the steps after them: failed steps, the model's outcome
        "over_context_steps": over_context_steps,
        "steps_not_generated_after_over_context": after_over_context,
        # problems left out of the score (a harness error in one of their sub-steps)
        "harness_error_problems": harness_error_problems,
        "scoring_excluded": sum(1 for t in sample_traces if t.get("scoring_excluded")),
        "with_background": with_background,
        "metric": (f"main-problem resolve rate {correct_problems}/{total_problems} (headline); "
                   f"sub-step accuracy {correct_steps}/{total_steps}"),
        "dataset_revision": f"SciCode1/SciCode@{_DATASET_REVISION} (test)",
        "temperature": temperature,
        "temperature_source": temp_source,
        "official_sampling": {**OFFICIAL_SAMPLING.record(temperature, temp_source),
                              **plan.record()},
        "max_output_tokens_note": plan.note,
        "sample_traces": sample_traces,
    }
    # Provenance of the reference outputs decides every score, so it is part of the result.
    result["test_data"] = {"path": h5, "source": provenance, "sha256": h5_sha256,
                           "verified": h5_reason is None}
    result.update(shard_identity(shard))
    result.update(_attempts_record(kwargs))
    result["eval_limit_plan"] = limit_plan
    # The comparison helpers the test cases import (vendored verbatim, on each step's PYTHONPATH).
    result["scicode_compare_module"] = {"upstream": SCICODE_CMP_UPSTREAM,
                                        "sha256": SCICODE_CMP_SHA256}
    # The leaderboard: without background, greedy, over the whole test split.
    noncanon: List[str] = []
    if h5_reason:
        noncanon.append(h5_reason)
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    elif shard is not None and shard.partial:
        pass            # a partial shard: base.SHARD_REASON below
    elif (len(results), steps_run) != (_TEST_PROBLEMS, _TEST_STEPS):
        noncanon.append(f"{len(results)} problems / {steps_run} sub-steps run, not upstream's "
                        f"{_TEST_PROBLEMS} problems / {_TEST_STEPS} sub-steps")
    if with_background:
        noncanon.append("with background (the leaderboard is the without-background setting)")
    if temperature != 0.0:
        noncanon.append(f"temperature={temperature} ({temp_source}; SciCode's default is greedy, "
                        "temperature 0)")
    if failed_gen or errored_problems:
        noncanon.append(f"{failed_gen} sub-step generation(s) got no reply from the served "
                        f"endpoint and {errored_problems} problem(s) crashed (harness errors: "
                        f"{harness_error_problems} problem(s) left out of the score)")
    # SciCode sends no max_tokens, so only the served context window bounds a reply: a sub-step
    # reply that reached it measures the window, not the model.
    cut, in_reasoning = budget_cutoffs(result)
    result["replies_cut_off_by_budget"] = cut
    result["replies_cut_off_in_reasoning"] = in_reasoning
    # A reply that reached the window repeating itself is a loop, which a larger window only
    # makes longer: no advice to serve one for those. Its content or, for a reply cut off while
    # looping in its reasoning, its reasoning (base.looping_channel, as base's classify_reply).
    looping = sum(1 for t in sample_traces
                  if t.get("finish_reason") == "length" and t.get("response_text") is not None
                  and _loops(t["response_text"], t.get("reasoning")))
    result["replies_cut_off_looping"] = looping
    if cut:
        window = get_run_knob("max_model_len")
        advice = ("SciCode sends no max_tokens, so serve a larger --max-model-len"
                  if cut > looping else
                  "every one a repetition loop, the model's own outcome, which a larger window "
                  "would not change")
        noncanon.append(f"{cut} sub-step repl(ies) reached the end of the served "
                        f"{window or 'unknown'}-token context window before finishing "
                        f"({in_reasoning} of them inside the reasoning channel with no answer, "
                        f"{looping} repetition loop(s); {advice})")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    if shard is not None and shard.partial:
        add_comparability_reason(result, SHARD_REASON.format(
            label=shard.label, index=shard.index, count=shard.count))
    return result


def _attempts_record(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """{"attempts": {...not_applied}} when --attempt-count asked for more than 1: SciCode's
    protocol generates each sub-step once, feeding it to the next step."""
    k = int(kwargs.get("attempt_count") or get_run_knob("attempt_count") or 1)
    if k <= 1:
        return {}
    return {"attempts": {"attempts_requested": k, "attempts_per_sample": 1,
                         "not_applied": ("scicode: SciCode's sequential protocol generates each "
                                         "sub-step once (the next step builds on it); "
                                         f"--attempt-count {k} is not applied")}}


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run's results (gbench --merge): the problem resolve rate
    (the headline) and the sub-step accuracy re-counted over the union of the shards' sub-step
    traces, de-duplicated by sample_key. A problem with any excluded (harness-error) sub-step has
    no verdict, as in run_scicode."""
    from ...merge import MergeError
    union: Dict[str, Dict[str, Any]] = {}
    for r in results:
        traces = r.get("sample_traces")
        if not isinstance(traces, list) or (traces and not all(
                isinstance(t, dict) and t.get("sample_key") for t in traces)):
            raise MergeError("scicode: a shard carries no keyed sub-step traces; re-run it with "
                             "the current gbench")
        for t in traces:
            union.setdefault(str(t["sample_key"]), t)
    traces = sorted(union.values(), key=lambda t: (t.get("global_idx", 0),
                                                   str(t.get("step_number"))))
    by_problem: Dict[str, List[Dict[str, Any]]] = {}
    for t in traces:
        by_problem.setdefault(str(t.get("problem_id")), []).append(t)
    total_steps = sum(1 for t in traces if not t.get("scoring_excluded"))
    correct_steps = sum(1 for t in traces if not t.get("scoring_excluded") and t.get("is_correct"))
    scored_problems = [p for p, ts in by_problem.items()
                       if not any(t.get("scoring_excluded") for t in ts)]
    correct_problems = sum(1 for p in scored_problems
                           if all(t.get("is_correct") for t in by_problem[p]))
    merged = dict(results[0])
    merged["sample_traces"] = traces
    merged["total_questions"] = len(by_problem)
    merged["effective_n"] = merged["total_problems"] = len(scored_problems)
    merged["correct_answers"] = merged["correct_problems"] = correct_problems
    merged["total_steps"], merged["correct_steps"] = total_steps, correct_steps
    merged["problem_accuracy"] = (round(100.0 * correct_problems / len(scored_problems), 2)
                                  if scored_problems else None)
    merged["accuracy"] = merged["problem_accuracy"]
    merged["substep_accuracy"] = (round(100.0 * correct_steps / total_steps, 2)
                                  if total_steps else None)
    for field in ("failed_generations", "steps_not_generated", "scoring_failed_steps",
                  "errored_problems", "over_context_steps",
                  "steps_not_generated_after_over_context",
                  "harness_error_problems", "scoring_excluded", "replies_cut_off_by_budget",
                  "replies_cut_off_in_reasoning", "replies_cut_off_looping"):
        merged[field] = sum(int(r.get(field) or 0) for r in results)
    statuses = {r.get("status") for r in results}
    merged["status"] = "completed_with_errors" if "completed_with_errors" in statuses else "success"
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = "no problem of any shard has a verdict"
    else:
        merged.pop("accuracy_withheld_reason", None)
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if reason and not reason.startswith(shard_prefix) and not reason.startswith(
                    "empty shard ") and reason not in kept:
                kept.append(reason)
    if not results[0].get("eval_limit") and len(by_problem) != _TEST_PROBLEMS:
        kept.append(f"the merged shards hold {len(by_problem)} problems, not upstream's "
                    f"{_TEST_PROBLEMS}")
    merged["leaderboard_comparable"] = not kept
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    return merged


def _crashed_problem_report(prob: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    """A problem whose processing raised: every generated (non-supplied) step is marked
    `crashed`, a harness error that run_scicode leaves out of the score, counts and names."""
    steps = []
    for k, st in enumerate(prob["sub_steps"], start=1):
        step_number = str(st.get("step_number") or f"{prob['problem_id']}.{k}")
        special = step_number in _SPECIAL_STEP_IDS
        steps.append({
            "step_number": step_number, "special": special, "gen_failed": False,
            "crashed": not special, "correct": False, "response_text": None,
            "reasoning": None, "reasoning_chars": 0, "finish_reason": None,
            "completion_tokens": None,
            "error": None if special else f"problem_crashed: {type(exc).__name__}: {exc}"[:500],
        })
    return {"problem_id": str(prob["problem_id"]), "problem_name": prob.get("problem_name"),
            "steps": steps}
