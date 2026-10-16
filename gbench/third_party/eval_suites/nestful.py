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

# Evaluation Suite: nestful
# Description: NESTFUL (IBM Nested Output-to-Input Function Calling Benchmark)

"""gbench native built-in runner for nestful (Tool Use & Function Calling).

Runs IBM/NESTFUL exactly as its own pipeline does (src/eval.py -> src/instruct_data_prep.py ->
src/scorer.py) with the DeepSeek prompt/parser pair, the chat-model path upstream ships: data from
the pinned checkout's `data_v2/nestful_data.jsonl`, the `deepseek_prompt_input` system prompt, the
whole reply parsed with `json.loads` (`parse_deepseek_output`), label grounding
(`ground_seq_nested_repsonse`) on gold and prediction, and `calculate_scores` for F1 Intent / F1 Slot /
Partial and Full Match / Win Rate.

One data source: the IBM/NESTFUL checkout at NESTFUL_COMMIT, as upstream's run.sh uses it
(`--dataset data_v2/nestful_data.jsonl`, `--executable_func_dir data_v2/executable_functions`).
Without the checkout's `data_v2/nestful_data.jsonl` the whole suite is infra_required, naming it.
Only the Win Rate needs the checkout's reference functions (`data_v2/executable_functions`). When
they are missing, or not at the pinned revision, while the data file is there, the other metrics
are still computed and the Win Rate alone is reported infra_required in its own entry
(`win_rate_metric`, listed in `metrics_infra_required`; `win_rate` null, status
completed_with_errors, not comparable), as upstream's calculate_scores(win_rate_flag=False)
reports it "no".

The Win Rate executes the model's reply: the reply names the reference functions and supplies
their arguments, and 18 of those offered in the data eval, exec or run their arguments as a
command. The worker that runs them lives in a bubblewrap sandbox of its own (_sandbox_argv): no
network, an environment of PATH and HOME=/tmp alone, no /proc, and only the interpreter and the
reference functions on its filesystem, read-only, so no key, token, credential file or metadata
server is within its reach. The worker reports what it can see when it starts and the parent
refuses it unless that is so. Without a working sandbox the Win Rate alone is reported
infra_required; it never runs unsandboxed.

Upstream's scorer runs the reference functions with network; the sandbox has none. The worker
notices every failed use of the network (_install_network_probe: a name lookup, connection or send
that fails). On a row whose pinned gold sequence needs the network (_GOLD_NEEDS_NETWORK, one row
of the pinned data), a call that did not win and failed such a use may have got its verdict from
the sandbox rather than from the reply, and the worker cannot tell which (a host name that exists
nowhere fails with network too), so it is treated as a harness error. The row is marked
`nestful_win_blocked_by_sandbox` (what the call tried), counted in
`win_rate_network_blocked_calls`, left out of the Win Rate (never scored as a loss) and named in
status_reason and leaderboard_comparable_reason (status completed_with_errors, not comparable).
When every answered row is such a call the Win Rate is null, with `win_rate_withheld_reason`. On
every other row the gold needs no network, so a losing call that failed a use of the network went
off the gold path by the reply's own choice: it is scored as the loss it is, marked
`nestful_win_network_blocked_offpath` and counted in `win_rate_network_blocked_offpath`.

A request that got no reply from the endpoint is a harness error: base leaves it out of the
score, and every metric here is computed over the rows that were answered. When no row was
answered every metric is null (never 0) and the run is not leaderboard-comparable; when some were
not, leaderboard_comparable_reason names how many and why.

Sampling: temperature 0.0 without `--thinking`, 1.0 with it (gbench's rule for every suite,
base.resolve_temperature; NESTFUL's own is 0.0, the src/eval.py `--temperature` default). max_tokens=1000 (src/eval.py `--max_tokens` default; run.sh overrides neither)
on every request of a no-think run. A `--thinking` run does not apply that cap (it was written for
non-reasoning generation): each request gets the run's --max-output-tokens, else the thinking
floor, and the result records the cap as not applied (base.resolve_token_budget). A reply cut off
by any other budget (an empty reply after reasoning included) is counted and makes the run not
leaderboard-comparable. Override the temperature for a whole run with `--temperature`, or for this
suite alone with `GBENCH_NESTFUL_TEMPERATURE`, which takes precedence over both; the result records
upstream's value under `official_sampling`. No LLM judge.
"""

import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from .base import OfficialSampling, run_eval_suite, strip_thinking_tags
from .sampling import parse_shard, stratified_sample
from .fc_common import base_suite_plan, run_attempts
from .metrics import scorable_traces
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/nestful.md"

#: NESTFUL's executable Win Rate needs the reference function library from the IBM/NESTFUL
#: checkout: `data_v2/executable_functions/` (basic_functions.py + func_file_map.json +
#: ~4348 py_code_file_*.py). Point this at that directory; the dataset is read from the same
#: checkout (`data_v2/nestful_data.jsonl`).
_FUNC_DIR_ENV = "GBENCH_NESTFUL_FUNC_DIR"

#: IBM/NESTFUL revision this suite is pinned to (HEAD of main, 2025-05-23).
NESTFUL_COMMIT = "fc2c4123e73500a56185a5fb354f05d1c8b4890c"

#: src/eval.py: SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens) with
#: `--temperature` defaulting to 0.0 and `--max_tokens` to 1000 (run.sh passes neither), for the
#: vLLM-served model. The token-limit policy (base.resolve_token_budget) sends the 1000 exactly on a
#: no-think run and does not apply it to a --thinking run; NESTFUL defines no reasoning settings.
UPSTREAM_MAX_TOKENS = 1000
NESTFUL_SAMPLING = OfficialSampling(
    reference=(f"IBM/NESTFUL@{NESTFUL_COMMIT[:12]} src/eval.py (--temperature default 0.0, "
               "--max_tokens default 1000)"),
    temperature=0.0,
    max_tokens=UPSTREAM_MAX_TOKENS,
)

#: git blob SHA-1 (`git hash-object`) of the pinned files, relative to `data_v2/`.
_PINNED_BLOBS = {
    "nestful_data.jsonl": "ab49f52a8a11289247f7f4c1b5265c01d449c1aa",
    "executable_functions/basic_functions.py": "1c030e35bed8e99368383a73013e647a2629c101",
    "executable_functions/func_file_map.json": "0e55a03b7782017f997af6cfe3e670e84fa9a5b2",
}

# --------------------------------------------------------------------------- #
# Canonical NESTFUL prompt: src/instruct_data_prep.py `deepseek_prompt_input` + `get_icl_str`
# (generic branch) with the first 3 examples of src/icl_examples.json (run.sh: --icl_count 3).
# The specs and examples go IN THE PROMPT TEXT; NESTFUL does not use the tools API. (Offering API
# tools makes the model emit ONE structured tool_call and stop, so it never composes the sequence:
# measured 2026-09-08, 0% full/partial/win.)
# --------------------------------------------------------------------------- #
_SYSTEM_PREAMBLE = (
    "You are a helpful assistant with access to the following function calls. Your task is to "
    "produce a sequence of function calls necessary to generate response to the user utterance. "
    "Here is a list of functions in JSON format that you can invoke:\n"
)
_FORMAT_INSTRUCTION = (
    "\nDO NOT try to answer the user question, just invoke the tools needed to respond to the "
    "user, if any. The output MUST strictly adhere to the following JSON format: "
    "[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": "
    "\"value2\"}, \"label\": \"$var_1\"}, ... (more tool calls as required)]. Please make sure "
    "the parameter type is correct and follow the documentation for parameter format. If no "
    "function call is needed, please directly output an empty list.\nHere are some examples:\n"
)

#: The canonical 3 in-context examples (input + gold output sequence), vendored verbatim from
#: IBM/NESTFUL src/icl_examples.json. Only input/output are shown, as in get_icl_str's generic
#: branch.
_ICL_EXAMPLES = [
    {"input": "In objective test a correct ans score 4 marks and on a wrong ans 2 marks are deducted. a student score 480 marks from 150 question. how many ans were correct?",
     "output": [{"name": "multiply", "label": "$var_1", "arguments": {"arg_0": 150, "arg_1": 2}},
                {"name": "add", "label": "$var_2", "arguments": {"arg_0": 480, "arg_1": "$var_1.result$"}},
                {"name": "add", "label": "$var_3", "arguments": {"arg_0": 4, "arg_1": 2}},
                {"name": "divide", "label": "$var_4", "arguments": {"arg_0": "$var_2.result$", "arg_1": "$var_3.result$"}}]},
    {"input": "A student traveled 10 percent of the distance of the trip alone, continued another 30 miles with a friend, and then finished the last half of the trip alone. How many miles long was the trip?",
     "output": [{"name": "inverse", "label": "$var_1", "arguments": {"arg_0": 10}},
                {"name": "subtract", "label": "$var_2", "arguments": {"arg_0": 1, "arg_1": "$var_1.result$"}},
                {"name": "divide", "label": "$var_3", "arguments": {"arg_0": 1, "arg_1": 2}},
                {"name": "subtract", "label": "$var_4", "arguments": {"arg_0": "$var_2.result$", "arg_1": "$var_3.result$"}},
                {"name": "divide", "label": "$var_5", "arguments": {"arg_0": 30, "arg_1": "$var_4.result$"}}]},
    {"input": "Given a string \"Hello world! How are you?\", extract non-whitespace substrings and count the words from the resulting substrings.",
     "output": [{"name": "non_whitespace_substrings", "arguments": {"input_str": "Hello world! How are you?"}, "label": "$var1"},
                {"name": "count_words_from_sentences", "arguments": {"sentences": "$var1.output_0$"}, "label": "$var2"}]},
]


def _icl_str() -> str:
    """The few-shot block, formatted exactly as NESTFUL's get_icl_str (generic branch)."""
    out = ""
    for idx, ex in enumerate(_ICL_EXAMPLES, 1):
        out += f"\n#Example-{idx}\nInput: {ex['input']}\nOutput: {json.dumps(ex['output'])}\n"
    return out


def _system_prompt(raw_tools: List[Dict[str, Any]]) -> str:
    """Canonical NESTFUL system prompt, byte-for-byte as upstream builds it.

    src/eval.py json-encodes each row's `tools` before prompt construction and
    `deepseek_prompt_input` json-encodes that string again, so the specs reach the model as a
    JSON string literal. The prompt ends with the ICL block plus a newline."""
    return (_SYSTEM_PREAMBLE + json.dumps(json.dumps(raw_tools)) + _FORMAT_INSTRUCTION
            + _icl_str() + "\n")


# --------------------------------------------------------------------------- #
# Pinned checkout
# --------------------------------------------------------------------------- #
def _git_blob_sha1(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _check_pinned(data_v2: str, rel: str) -> None:
    path = os.path.join(data_v2, rel)
    if _git_blob_sha1(path) != _PINNED_BLOBS[rel]:
        raise infra_required(
            "nestful",
            f"data_v2/{rel} does not match the pinned IBM/NESTFUL revision {NESTFUL_COMMIT}. "
            f"Check out that commit (git -C <NESTFUL checkout> checkout {NESTFUL_COMMIT}).",
            DOCS_URL,
        )


#: The Win Rate's two prerequisites, as its infra_required entry names them.
_FUNCTIONS_PREREQ = (f"{_FUNC_DIR_ENV}: IBM/NESTFUL data_v2/executable_functions at "
                     f"{NESTFUL_COMMIT}")
_SANDBOX_PREREQ = ("a working bubblewrap (bwrap) sandbox for the reference functions (bubblewrap "
                   "installed, unprivileged user namespaces allowed)")


class WinRateUnavailable(RuntimeError):
    """A prerequisite of the executable Win Rate is missing: IBM/NESTFUL's reference functions
    (absent or not at the pinned revision) or the sandbox they run in. Only the Win Rate needs
    them: run_nestful computes the other metrics and reports the Win Rate infra_required in its
    own entry (win_rate_infra_required), naming `prerequisite`."""

    def __init__(self, reason: str, prerequisite: str = _FUNCTIONS_PREREQ):
        self.reason = reason
        self.prerequisite = prerequisite
        super().__init__(f"[INFRA REQUIRED] nestful: NESTFUL's Win Rate cannot be scored: {reason} "
                         f"See '{DOCS_URL}'.")


def _func_dir() -> str:
    """The IBM/NESTFUL `data_v2/executable_functions` directory at the pinned revision.

    Raises WinRateUnavailable when it is absent, incomplete or at another revision: the
    executable Win Rate is a canonical NESTFUL metric and is never faked; it is reported
    infra_required on its own while the other metrics run."""
    d = prereqs_path("NESTFUL/data_v2/executable_functions", os.environ.get(_FUNC_DIR_ENV))
    if d:
        d = os.path.normpath(os.path.abspath(os.path.expanduser(d)))
    if not d or not os.path.isfile(os.path.join(d, "basic_functions.py")) \
            or not os.path.isfile(os.path.join(d, "func_file_map.json")):
        raise WinRateUnavailable(
            f"the executable Win Rate needs IBM/NESTFUL's reference functions. Set {_FUNC_DIR_ENV} "
            "to the checkout's data_v2/executable_functions directory (contains basic_functions.py, "
            "func_file_map.json and py_code_file_*.py). Clone github.com/IBM/NESTFUL and check out "
            f"{NESTFUL_COMMIT} to obtain it.")
    data_v2 = os.path.dirname(d)
    for rel in ("executable_functions/basic_functions.py", "executable_functions/func_file_map.json"):
        if _git_blob_sha1(os.path.join(data_v2, rel)) != _PINNED_BLOBS[rel]:
            raise WinRateUnavailable(
                f"data_v2/{rel} does not match the pinned IBM/NESTFUL revision {NESTFUL_COMMIT}. "
                f"Check out that commit (git -C <NESTFUL checkout> checkout {NESTFUL_COMMIT}).")
    return d


def win_rate_infra_required(reason: str, prerequisite: str = _FUNCTIONS_PREREQ) -> Dict[str, Any]:
    """The result entry of the Win Rate when its prerequisite is missing while the other metrics
    run: not computed, not scored, the missing prerequisite named."""
    sentence = str(reason).rstrip()
    sentence += "" if sentence.endswith(".") else "."
    return {"status": "infra_required", "metric": "win_rate",
            "prerequisite": prerequisite,
            "infra_required": str(reason),
            "error": (f"[INFRA REQUIRED] nestful: NESTFUL's Win Rate was not scored: {sentence} "
                      f"See '{DOCS_URL}'."),
            "leaderboard_comparable": False,
            "leaderboard_comparable_reason": ("NESTFUL's Win Rate was not scored (missing "
                                              "prerequisite, infra_required)"),
            "value": None}


def _checkout_data_file() -> str:
    """`data_v2/nestful_data.jsonl` of the IBM/NESTFUL checkout when its reference functions are
    not usable: next to the directory GBENCH_NESTFUL_FUNC_DIR names, else under
    $GBENCH_PREREQS_DIR/NESTFUL. No such file is infra_required for the whole suite (the checkout
    is NESTFUL's only data source); a file that is there but not at the pinned revision is a hard
    error, as with the functions present."""
    explicit = (os.environ.get(_FUNC_DIR_ENV) or "").strip()
    if explicit:
        func_dir = os.path.normpath(os.path.abspath(os.path.expanduser(explicit)))
        path = os.path.join(os.path.dirname(func_dir), "nestful_data.jsonl")
        if not os.path.isfile(path):
            raise infra_required(
                "nestful",
                f"{path} is missing; {_FUNC_DIR_ENV} must point into a full IBM/NESTFUL checkout "
                f"(revision {NESTFUL_COMMIT}) whose data_v2/ holds nestful_data.jsonl, the only "
                "data source (upstream run.sh --dataset data_v2/nestful_data.jsonl).",
                DOCS_URL,
            )
    else:
        path = prereqs_path("NESTFUL/data_v2/nestful_data.jsonl") or ""
        if not path or not os.path.isfile(path):
            raise infra_required(
                "nestful",
                "no IBM/NESTFUL checkout: NESTFUL's rows are read only from the checkout's "
                "data_v2/nestful_data.jsonl (upstream run.sh --dataset), and neither "
                f"{_FUNC_DIR_ENV} nor $GBENCH_PREREQS_DIR/NESTFUL provides it. Clone "
                f"github.com/IBM/NESTFUL, check out {NESTFUL_COMMIT}, and set {_FUNC_DIR_ENV} to "
                "its data_v2/executable_functions (or GBENCH_PREREQS_DIR to the directory that "
                "holds NESTFUL/).",
                DOCS_URL,
            )
    path = os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    _check_pinned(os.path.dirname(path), "nestful_data.jsonl")
    return path


def _dataset(func_dir: Optional[str]) -> str:
    """The checkout's data_v2/nestful_data.jsonl: next to the reference functions when they are
    usable (_data_file), else from the checkout without them (_checkout_data_file)."""
    if func_dir is not None:
        return _data_file(func_dir)
    return _checkout_data_file()


def _data_file(func_dir: str) -> str:
    """`data_v2/nestful_data.jsonl` next to the functions directory (run.sh's --dataset)."""
    data_v2 = os.path.dirname(os.path.normpath(func_dir))
    path = os.path.join(data_v2, "nestful_data.jsonl")
    if not os.path.isfile(path):
        raise infra_required(
            "nestful",
            f"{path} is missing; {_FUNC_DIR_ENV} must point into a full IBM/NESTFUL checkout "
            f"(revision {NESTFUL_COMMIT}) whose data_v2/ holds nestful_data.jsonl.",
            DOCS_URL,
        )
    _check_pinned(data_v2, "nestful_data.jsonl")
    return path


# --------------------------------------------------------------------------- #
# Executable Win Rate: port of src/scorer.py calculate_ans / calculate_win_score. The predicted
# call list is executed against the benchmark's reference function library and the final output
# compared to `gold_answer`.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=8)
def _basic_func_list(func_dir: str) -> frozenset:
    """Function names defined in basic_functions.py (scorer.py: lines starting `def `)."""
    names = set()
    with open(os.path.join(func_dir, "basic_functions.py"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("def "):
                func_str = line.strip().replace("def ", "")
                names.add(func_str[:func_str.index("(")])
    return frozenset(names)


@functools.lru_cache(maxsize=8)
def _func_file_map(func_dir: str) -> Dict[str, str]:
    with open(os.path.join(func_dir, "func_file_map.json"), encoding="utf-8") as f:
        return json.load(f)


def _load_func_module(func_dir: str, file_name: str):
    """Import a reference-function file afresh on every call, as scorer.py does, so module-level
    state never leaks between calls."""
    file_path = os.path.join(func_dir, file_name)
    spec = importlib.util.spec_from_file_location(file_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[file_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _listit(t: Any) -> Any:
    return list(map(_listit, t)) if isinstance(t, (list, tuple)) else t


#: scorer.py's signal.alarm(10) around each calculate_ans call.
_WIN_ALARM_S = 10
#: How long run_nestful waits for the Win Rate worker's verdict before it kills the worker and
#: scores the example as a failure: a reference function stuck in C code (a huge integer power,
#: say) never returns to the interpreter, so the alarm cannot interrupt it.
_WIN_BACKSTOP_S = 30
#: How long the worker may take to start (a Python interpreter in its sandbox).
_WIN_START_S = 120
#: The command line of upstream's scoring process (run.sh: `python -u src/scorer.py --model_name
#: ... --result_file_path ... --executable_func_dir data_v2/executable_functions`), set as the
#: worker's sys.argv: three reference functions read it (parse_command_line_options parses it,
#: meets scorer.py's options and exits, which upstream's bare `except:` scores as a failed call).
_UPSTREAM_SCORER_ARGV = ("src/scorer.py", "--model_name", "model", "--result_file_path",
                         "results/nestful_3/model/output.jsonl", "--executable_func_dir",
                         "data_v2/executable_functions")


def _calculate_ans(func_calls: Any, spec_lib: List[Dict[str, Any]], func_dir: str,
                   alarm_s: int = _WIN_ALARM_S) -> Any:
    """Execute the nested call list and return the final output (or False on any failure).

    Port of scorer.py::calculate_ans: resolves `$label.field$` references from earlier outputs,
    dispatches each call to basic_functions.py or func_file_map.json, and returns the last call's
    single output value. The 10s alarm matches upstream's signal.alarm; it can only be set on the
    main thread, which is why run_nestful calls this in _WinRateWorker's process."""
    use_alarm = threading.current_thread() is threading.main_thread()
    if use_alarm:
        def _handler(signum, frame):
            raise TimeoutError("Time limit exceeded!")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(alarm_s)
    try:
        basic = _basic_func_list(func_dir)
        file_map = _func_file_map(func_dir)
        variable_result_map: Dict[str, Dict[str, Any]] = {}
        for f in func_calls:
            label = f["label"].replace("$", "")
            output_params = [s for s in spec_lib if s["name"] == f["name"]][0]["output_parameters"]
            output_params = list(output_params.keys())
            arg_values = []
            arg_val_list = []
            for _k, v in f["arguments"].items():
                if type(v) == str and v.startswith("$") and v.endswith("$"):
                    v = v[1:-1]
                    v_l, out_param = v.split(".", 1)[0], v.split(".", 1)[1]
                    v = variable_result_map[v_l][out_param]
                elif type(v) == str and v.startswith("$var"):
                    v = v[1:]
                    v_l, out_param = v.split(".", 1)[0], v.split(".", 1)[1]
                    v = variable_result_map[v_l][out_param]
                arg_val_list.append(v)
                # Unused, but kept: upstream builds this string, and str() of an int over
                # 4300 digits raises, which makes the call count as a failure there.
                arg_values.append(str(v))
            name = f["name"]
            file_name = "basic_functions.py" if name in basic else file_map[name]
            func = getattr(_load_func_module(func_dir, file_name), name)
            try:
                res = func(*arg_val_list)
            except Exception:
                return False
            if len(output_params) == 1:
                variable_result_map[label] = {output_params[0]: res}
            else:
                return False
        final_var = func_calls[-1]["label"].replace("$", "")
        return next(iter(variable_result_map[final_var].values()))
    except TimeoutError:
        return False
    except Exception:
        return False
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _win_score(pred_func_calls: Any, gold_ans: Any, tools: Any, func_dir: str,
               alarm_s: int = _WIN_ALARM_S) -> bool:
    """Port of scorer.py::calculate_win_score. `pred_func_calls` is the parsed reply
    (`pred_dict_list`); `gold_ans` and `tools` are the JSON texts src/eval.py stores."""
    if not pred_func_calls:
        return False
    gold_ans = json.loads(gold_ans) if isinstance(gold_ans, str) else gold_ans
    tools = json.loads(tools) if isinstance(tools, str) else tools
    pred_ans = _calculate_ans(pred_func_calls, tools, func_dir, alarm_s)
    if type(gold_ans) == float and type(pred_ans) == float:
        dec = str(gold_ans).split(".")
        # Upstream indexes [1] unconditionally and would crash on e.g. 1e-05.
        if len(dec) > 1:
            pred_ans = round(pred_ans, len(dec[1]))
    if pred_ans == gold_ans:
        return True
    if type(pred_ans) == tuple and type(gold_ans) == list:
        return _listit(pred_ans) == gold_ans
    return False


def _sandbox_facts(func_dir: Optional[str]) -> Dict[str, Any]:
    """What the Win Rate worker can see of its host: its process id, the names of its
    environment variables, the entries of its root, whether /proc and /run exist, its network
    interfaces, and whether the reference functions are there and read-only. The worker reports
    them on its ready line; the parent checks them (_isolation_problem) before sending a reply."""
    import socket
    try:
        ifaces: Optional[List[str]] = sorted(name for _index, name in socket.if_nameindex())
    except OSError:
        ifaces = None
    return {"pid": os.getpid(), "env": sorted(os.environ), "root": sorted(os.listdir("/")),
            "proc": os.path.exists("/proc"), "run": os.path.exists("/run"), "ifaces": ifaces,
            "functions": bool(func_dir) and os.path.isfile(os.path.join(func_dir,
                                                                        "basic_functions.py")),
            "functions_writable": bool(func_dir) and os.access(func_dir, os.W_OK)}


def _install_network_probe() -> List[str]:
    """Make the Win Rate worker notice a reference function's failed use of the network, which its
    sandbox does not have. Wraps the socket module's name lookups (getaddrinfo, gethostbyname,
    gethostbyname_ex, gethostbyaddr, getnameinfo), create_connection, and socket.socket's connect,
    connect_ex, sendto and bind, and returns the list that each failed use is added to (what was
    called, on what, and the error; at most 5 per call, which _win_worker_main empties before
    each). A failed name lookup counts wherever it happens; a failed connection or send counts
    for any reason but a non-blocking one still in progress; bind counts only when its address
    failed to resolve. The 10 s alarm of _calculate_ans firing inside one of these calls is the
    call's timeout (a loss), never noted. Every wrapped call returns or raises exactly as
    before."""
    import errno
    import socket
    refused: List[str] = []
    in_progress = {errno.EINPROGRESS, errno.EALREADY, errno.EWOULDBLOCK, errno.EAGAIN,
                   errno.EISCONN, errno.EINTR}
    lookup_errors = (socket.gaierror, socket.herror)

    def note(what, target, error):
        text = f"{what}({target!r}): {type(error).__name__}: {error}"[:300]
        if len(refused) < 5 and text not in refused:
            refused.append(text)

    def wrap(owner, name, method, any_failure):
        real = getattr(owner, name)

        @functools.wraps(real)
        def probe(*args, **kwargs):
            try:
                return real(*args, **kwargs)
            except OSError as e:
                # upstream's 10 s alarm (_calculate_ans raises TimeoutError("Time limit
                # exceeded!"), an OSError) firing inside the call is the call's timeout, not a
                # failed use of the network
                alarm = isinstance(e, TimeoutError) and str(e) == "Time limit exceeded!"
                if not alarm and (isinstance(e, lookup_errors) or (
                        any_failure and not isinstance(e, (BlockingIOError, InterruptedError)))):
                    rest = args[1:] if method else args
                    # sendto(data[, flags], address): the address is its last argument
                    target = rest[-1] if name == "sendto" and rest else (rest[0] if rest
                                                                         else None)
                    note(name, target, e)
                raise
        setattr(owner, name, probe)

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr",
                 "getnameinfo"):
        wrap(socket, name, False, False)
    wrap(socket, "create_connection", False, True)
    wrap(socket.socket, "connect", True, True)
    wrap(socket.socket, "sendto", True, True)
    wrap(socket.socket, "bind", True, False)
    real_connect_ex = socket.socket.connect_ex

    @functools.wraps(real_connect_ex)
    def connect_ex(self, address):
        try:
            code = real_connect_ex(self, address)
        except OSError as e:
            note("connect_ex", address, e)
            raise
        if code and code not in in_progress:
            note("connect_ex", address, OSError(code, os.strerror(code)))
        return code
    socket.socket.connect_ex = connect_ex
    return refused


def _win_worker_main(func_dir: Optional[str] = None) -> None:
    """Entry point of the _WinRateWorker process: a ready line carrying _sandbox_facts (for the
    reference functions directory `func_dir`), then one JSON request per line on stdin
    ([pred, gold_answer, tools, func_dir, alarm_s]) and one verdict line back: "1" for a win, "0"
    for a loss, and "n" followed by a JSON list of what was tried for a call that did not win and
    failed a use of the network (_install_network_probe): the sandbox, which has none, may have
    decided that verdict rather than the reply.

    sys.argv is upstream's scoring command (_UPSTREAM_SCORER_ARGV), which the reference functions
    that read it see there. The protocol keeps its own copies of stdin/stdout; the process's fd 0
    and 1 point at /dev/null, so a reference function that reads or prints cannot disturb it."""
    refused = _install_network_probe()
    sys.argv = list(_UPSTREAM_SCORER_ARGV)
    inp = os.fdopen(os.dup(0), "rb")
    out = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    sys.stdout = open(os.devnull, "w")
    facts = _sandbox_facts(func_dir)
    out.write(b"ready " + json.dumps(facts).encode("utf-8") + b"\n")
    out.flush()
    for line in inp:
        del refused[:]
        try:
            pred, gold, tools, func_dir, alarm_s = json.loads(line)
            win = _win_score(pred, gold, tools, func_dir, alarm_s)
        except BaseException:                                  # noqa: BLE001 - a failed call
            win = False
        if win:
            out.write(b"1\n")
        elif refused:
            out.write(b"n " + json.dumps(refused).encode("utf-8") + b"\n")
        else:
            out.write(b"0\n")
        out.flush()


# --------------------------------------------------------------------------- #
# The Win Rate worker's sandbox. The reply decides which reference functions run and with what
# arguments, and some of them eval, exec or run their arguments, so the worker is model-controlled
# code: it runs in a bubblewrap container of its own whatever GBENCH_SANDBOX says.
# --------------------------------------------------------------------------- #
#: The worker's environment, all of it (bubblewrap --clearenv, then these).
_SANDBOX_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"}
#: Variables the worker may show besides _SANDBOX_ENV: bubblewrap sets PWD, and Python's C
#: locale coercion sets LC_CTYPE.
_SANDBOX_ENV_EXTRA = frozenset({"PWD", "LC_CTYPE"})
#: Size of the worker's private /tmp (its working directory), where bubblewrap can bound it.
_SANDBOX_TMP_BYTES = 256 << 20
#: What the worker's root holds besides the first components of the interpreter's and the
#: reference functions' directories.
_SANDBOX_ROOT = frozenset({"usr", "bin", "sbin", "lib", "lib32", "lib64", "libx32", "etc", "dev",
                           "tmp"})
#: The worker's /etc/hosts: the loopback names every host has, so a reference function that names
#: localhost reaches the sandbox's own loopback as upstream's reaches its host's (without the file
#: `localhost` does not resolve there). The host's own /etc/hosts is not shown.
_SANDBOX_HOSTS = b"127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n"
#: What the sandbox changes in the Win Rate: the reference functions have no network. Checked on
#: the pinned data by scoring every gold call sequence with upstream's scorer.py (with network,
#: from the checkout, under its own command line) and with this worker: 1857 and 1856 of 1861
#: wins, differing on one sequence alone (_GOLD_NEEDS_NETWORK), which the worker reports as a
#: failed use of the network.
_SANDBOX_NETWORK_NOTE = ("no network (loopback alone, and an /etc/hosts that names localhost "
                         "alone where bubblewrap offers --ro-bind-data): on a row whose pinned "
                         "gold sequence needs the network (one on the pinned data: sample "
                         "fa57c28a-e9d8-44fb-a358-a9404a557d5b fetches https://www.example.com "
                         "with get_url_content, which upstream's scorer, run with network, "
                         "scores a win), a call that did not win and failed a use of the network "
                         "may have got its verdict from the sandbox, so it is a harness error, "
                         "marked nestful_win_blocked_by_sandbox, counted in "
                         "win_rate_network_blocked_calls and left out of the Win Rate (the run "
                         "is then completed_with_errors and not comparable); on every other row "
                         "the gold needs no network, so such a call went off the gold path by "
                         "the reply's own choice and is scored (a loss), marked "
                         "nestful_win_network_blocked_offpath and counted in "
                         "win_rate_network_blocked_offpath")
#: The rows of the pinned data whose gold call sequence uses the network: scoring every gold
#: sequence with upstream's scorer.py (with network) and with the sandboxed worker differs on this
#: one alone (get_url_content of https://www.example.com). Of the 17 rows that offer a function
#: that uses the network, the other 16 have gold sequences that use none. The data file is checked
#: against its pinned blob, so the set is exact for every run.
_GOLD_NEEDS_NETWORK = frozenset({"fa57c28a-e9d8-44fb-a358-a9404a557d5b"})
#: The functions the worker runs, sent to it as source (_worker_source): it imports nothing from
#: gbench, whose tree is not in its sandbox.
_WORKER_FUNCTIONS = ("_basic_func_list", "_func_file_map", "_load_func_module", "_listit",
                     "_calculate_ans", "_win_score", "_sandbox_facts", "_install_network_probe",
                     "_win_worker_main")


@functools.lru_cache(maxsize=1)
def _worker_source() -> str:
    """The worker program: this module's scoring functions, verbatim (_sandbox_argv appends the
    call of _win_worker_main)."""
    parts = ["import functools, importlib.util, json, os, signal, sys, threading",
             "from typing import Any, Dict, List, Optional, Tuple",
             f"_WIN_ALARM_S = {_WIN_ALARM_S!r}",
             f"_UPSTREAM_SCORER_ARGV = {_UPSTREAM_SCORER_ARGV!r}"]
    parts += [inspect.getsource(globals()[name]) for name in _WORKER_FUNCTIONS]
    return "\n\n".join(parts) + "\n"


@functools.lru_cache(maxsize=1)
def _bwrap_options() -> frozenset:
    """The options this bubblewrap offers (from `bwrap --help`), to use --disable-userns and
    --size where it has them."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return frozenset()
    try:
        text = subprocess.run([bwrap, "--help"], capture_output=True, text=True, timeout=10,
                              env={}).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(re.findall(r"--[a-z][a-z-]+", text))


def _interpreter_dirs() -> List[str]:
    """The directories the worker's interpreter runs from that are not under /usr: its prefix
    and base prefix (a virtual environment and the Python it was made from) and the directory of
    its executable."""
    dirs = set()
    for path in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                 os.path.dirname(os.path.realpath(sys.executable))):
        for candidate in {path, os.path.realpath(path)}:
            if candidate and candidate != "/" and os.path.isdir(candidate) \
                    and not (candidate + os.sep).startswith("/usr/"):
                dirs.add(candidate)
    return sorted(dirs)


def _sandbox_argv(func_dir: str, hosts_fd: Optional[int] = None) -> List[str]:
    """The worker's command: a bubblewrap container with new user, PID, IPC, UTS, network and
    cgroup namespaces (`--unshare-all --unshare-user`, and `--disable-userns` where bwrap has
    it), no capabilities, the environment reduced to _SANDBOX_ENV, a new session, killed with
    gbench. Its filesystem holds /usr and the interpreter's own directories read-only, the /bin,
    /sbin and /lib* links or directories, the dynamic loader's cache, a private minimal /dev, a
    private /tmp (the working directory) and the reference functions directory, read-only at its
    own path; with `hosts_fd` (a file descriptor carrying _SANDBOX_HOSTS, which the command must
    inherit) also a read-only /etc/hosts copied from it. No /proc, /run, /var, /home, gbench tree
    or cache is there, and the only network interface is loopback, so no key, token, credential
    file, Docker socket or metadata server can be reached. Raises WinRateUnavailable when
    bubblewrap is not installed."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise WinRateUnavailable(
            "the Win Rate executes the model's calls, and gbench runs them only in a bubblewrap "
            "sandbox, but bubblewrap (bwrap) is not installed (`apt-get install bubblewrap`; it "
            "must be allowed to create unprivileged user namespaces).", _SANDBOX_PREREQ)
    options = _bwrap_options()
    cmd = [bwrap, "--unshare-all", "--unshare-user"]
    if "--disable-userns" in options:
        cmd.append("--disable-userns")
    cmd += ["--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for name, value in _SANDBOX_ENV.items():
        cmd += ["--setenv", name, value]
    cmd += ["--ro-bind", "/usr", "/usr"]
    for name in ("bin", "sbin", "lib", "lib32", "lib64", "libx32"):
        path = "/" + name
        if os.path.islink(path):
            cmd += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            cmd += ["--ro-bind", path, path]
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d"):
        if os.path.exists(path):
            cmd += ["--ro-bind", path, path]
    if hosts_fd is not None:
        cmd += ["--ro-bind-data", str(hosts_fd), "/etc/hosts"]
    # An interpreter directory under /tmp is bound after the private /tmp, which would hide it.
    under_tmp = [p for p in _interpreter_dirs() if (p + os.sep).startswith("/tmp/")]
    for path in _interpreter_dirs():
        if path not in under_tmp:
            cmd += ["--ro-bind", path, path]
    cmd += ["--dev", "/dev"]
    if "--size" in options:
        cmd += ["--size", str(_SANDBOX_TMP_BYTES)]
    # After the private /tmp, so a functions directory (or interpreter) under /tmp stays visible.
    cmd += ["--tmpfs", "/tmp"]
    for path in under_tmp:
        cmd += ["--ro-bind", path, path]
    cmd += ["--ro-bind", func_dir, func_dir, "--chdir", "/tmp"]
    program = _worker_source() + f"\n_win_worker_main({func_dir!r})\n"
    return cmd + [sys.executable, "-I", "-c", program]


def _isolation_problem(facts: Any, func_dir: str) -> Optional[str]:
    """None when the worker's report (_sandbox_facts) shows the sandbox _sandbox_argv describes;
    else what is wrong."""
    if not isinstance(facts, dict):
        return "the worker did not report what it can see"
    wrong = []
    pid = facts.get("pid")
    if not isinstance(pid, int) or pid > 16:
        wrong.append(f"it is process {pid}, not one of the first of its own PID namespace")
    extra_env = sorted(set(facts.get("env") or ()) - set(_SANDBOX_ENV) - _SANDBOX_ENV_EXTRA)
    if extra_env:
        wrong.append("its environment holds " + ", ".join(extra_env))
    if facts.get("ifaces") != ["lo"]:
        wrong.append(f"its network interfaces are {facts.get('ifaces')}, not loopback alone")
    if facts.get("proc") or facts.get("run"):
        wrong.append("/proc or /run is visible in it")
    allowed = set(_SANDBOX_ROOT) | {p.strip("/").split("/")[0]
                                    for p in _interpreter_dirs() + [func_dir]}
    extra_root = sorted(set(facts.get("root") or ()) - allowed)
    if extra_root:
        wrong.append(f"its root holds {', '.join(extra_root)}")
    if not facts.get("functions"):
        wrong.append("the reference functions are not visible in it")
    if facts.get("functions_writable"):
        wrong.append("the reference functions are writable in it")
    if wrong:
        return "the Win Rate worker's sandbox is not isolated as it must be: " + "; ".join(wrong)
    return None


def _blocked_uses(payload: bytes) -> List[str]:
    """What a call tried of the network, from the worker's "n" verdict line (at most 5 entries of
    300 characters)."""
    try:
        uses = json.loads(payload)
    except ValueError:
        uses = None
    if not isinstance(uses, list) or not uses:
        return ["a use of the network (the worker named none)"]
    return [str(u)[:300] for u in uses[:5]]


class _WinRateWorker:
    """Runs the executable Win Rate (scorer.py calculate_win_score) in a sandboxed child process.

    The reply is model-controlled code, so the child runs in the bubblewrap container
    _sandbox_argv builds; it starts at construction (before any generation, when run_nestful
    creates it) and again after a kill, and each time its report of what it can see is checked
    (_isolation_problem). A child that does not start, or is not isolated, raises
    WinRateUnavailable; nothing is ever executed outside the sandbox.

    Upstream caps each calculate_ans call with signal.alarm(10), which only works on the main
    thread; with --suite-timeout the suite runs on a worker thread. The child runs every call on
    its own main thread, so the 10 s alarm always applies, and a call that ignores the alarm is
    killed after _WIN_BACKSTOP_S and scored as a failure (the child is restarted).

    A call that did not win and failed a use of the network, which the sandbox withholds, sets
    `last_blocked` to what it tried (score() returns False); run_nestful leaves it out of the Win
    Rate as a harness error."""

    def __init__(self, func_dir: str, alarm_s: int = _WIN_ALARM_S,
                 backstop_s: float = _WIN_BACKSTOP_S):
        self.func_dir, self.alarm_s, self.backstop_s = func_dir, alarm_s, backstop_s
        self.proc: Optional[subprocess.Popen] = None
        #: calls that timed out or crashed the child (each scored as a failure)
        self.kills = 0
        #: what the child reported it can see (_sandbox_facts), checked at every start
        self.facts: Optional[Dict[str, Any]] = None
        #: for the last score(): what the call tried of the network the sandbox withholds, when
        #: it did not win and such a use failed (a harness error); else None
        self.last_blocked: Optional[List[str]] = None
        #: bytes of the protocol stream read but not yet returned as a line (_readline)
        self._buf = b""
        self._start()

    def _readline(self, timeout: float) -> Optional[bytes]:
        """One line of the worker's protocol stream (ending in a newline) within `timeout`
        seconds, else None. It reads the pipe's descriptor directly into a buffer of its own, so
        a partial line (a reference function that writes to the protocol stream without a
        newline, say) never blocks past the deadline; at the end of the stream the bytes left, if
        any, are returned without a newline (never a verdict)."""
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                return None
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                rest, self._buf = self._buf, b""
                return rest or None
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line + b"\n"

    def _start(self) -> None:
        self._buf = b""
        # /etc/hosts for the container (_SANDBOX_HOSTS), handed to bubblewrap through a pipe.
        hosts_fd: Optional[int] = None
        if "--ro-bind-data" in _bwrap_options():
            hosts_fd, hosts_w = os.pipe()
            try:
                os.write(hosts_w, _SANDBOX_HOSTS)
            finally:
                os.close(hosts_w)
        try:
            argv = _sandbox_argv(self.func_dir, hosts_fd=hosts_fd)
            try:
                self.proc = subprocess.Popen(
                    argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env={}, cwd="/",
                    pass_fds=() if hosts_fd is None else (hosts_fd,))
            except OSError as e:
                raise WinRateUnavailable(f"the Win Rate worker's sandbox could not be started "
                                         f"({type(e).__name__}: {e}).", _SANDBOX_PREREQ) from None
        finally:
            if hosts_fd is not None:
                os.close(hosts_fd)
        line = self._readline(_WIN_START_S) or b""
        facts: Any = None
        if line.startswith(b"ready "):
            try:
                facts = json.loads(line[len(b"ready "):])
            except ValueError:
                facts = None
        if facts is None:
            code = self.proc.poll()
            self._stop()
            raise WinRateUnavailable(
                "the Win Rate worker's bubblewrap sandbox did not start"
                + (f" (exit code {code})" if code is not None else "")
                + f": it could not start the interpreter {sys.executable} there, or bubblewrap "
                "cannot create unprivileged user namespaces (on Ubuntu 23.10+: `sudo sysctl "
                "kernel.apparmor_restrict_unprivileged_userns=0`); its message, if any, is in "
                "the log.", _SANDBOX_PREREQ)
        why = _isolation_problem(facts, self.func_dir)
        if why:
            self._stop()
            raise WinRateUnavailable(why + ".", _SANDBOX_PREREQ)
        self.facts = facts

    def _stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:                              # noqa: BLE001
                    pass
            self.proc = None

    def score(self, pred_func_calls: Any, gold_ans: Any, tools: Any) -> bool:
        """_win_score(pred_func_calls, gold_ans, tools, func_dir), computed in the child. A call
        that did not win and failed a use of the network returns False with `last_blocked` set
        to what it tried (_blocked_uses); `last_blocked` is None after every other call."""
        self.last_blocked = None
        if not pred_func_calls:
            return False
        if self.proc is not None and self.proc.poll() is not None:
            self._stop()
        if self.proc is None:
            self._start()
        request = json.dumps([pred_func_calls, gold_ans, tools, self.func_dir, self.alarm_s])
        try:
            self.proc.stdin.write(request.encode("utf-8") + b"\n")
            self.proc.stdin.flush()
            answer = self._readline(self.backstop_s)
        except OSError:                                        # the child died mid-call
            answer = None
        if answer in (b"1\n", b"0\n"):
            return answer == b"1\n"
        if answer is not None and answer.startswith(b"n ") and answer.endswith(b"\n"):
            self.last_blocked = _blocked_uses(answer[2:])
            return False
        self._stop()
        self.kills += 1
        return False

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:                                  # noqa: BLE001
                pass
        self._stop()


# --------------------------------------------------------------------------- #
# Reply parsing: src/output_parsers.py parse_deepseek_output + ground_seq_nested_repsonse
# --------------------------------------------------------------------------- #
def _ibm_ground(api_list: Any) -> List[Dict[str, Any]]:
    """Verbatim port of ground_seq_nested_repsonse: replace each `$label.field` reference with the
    NAME of the call that produced the label and drop the labels, so the match metrics compare
    references by producer rather than by the label spelling ('$var_1' vs '$var1')."""
    def check_label_in_slot(label, slot_v):
        if slot_v.startswith("$var"):
            if "." in slot_v:
                lbl_slot = slot_v.split(".", 1)[0].replace("$", "")
                if lbl_slot == label:
                    return True
        return False

    label_api_map = {}
    for api in api_list:
        if api["name"] == "varResult":
            continue
        if api["name"] == "var_result":
            continue
        if "label" in api:
            lbl = api["label"].replace("$", "")
            label_api_map[lbl] = api["name"]

    grounded_api_list = []
    for api in api_list:
        if api["name"] == "var_result":
            continue
        temp_arguments = {}
        if "arguments" in api:
            arg_dict = api["arguments"]
        elif "parameters" in api:
            arg_dict = api["parameters"]
        else:
            arg_dict = {}
        for s_n, s_v in arg_dict.items():
            for l, a in label_api_map.items():
                if type(s_v) == str and check_label_in_slot(l, s_v):
                    s_v = s_v.replace(l, a)
                elif type(s_v) == list:
                    new_s_v = []
                    for v in s_v:
                        if type(v) == str and check_label_in_slot(l, v):
                            v = v.replace(l, a)
                        elif type(v) == dict and check_label_in_slot(l, json.dumps(v)):
                            v = json.loads(json.dumps(v).replace(l, a))
                        new_s_v.append(v)
                    s_v = new_s_v
            temp_arguments[s_n] = s_v
        grounded_api_list.append({"name": api["name"], "arguments": temp_arguments})
    return grounded_api_list


def _ibm_parse(generated_text: str, gold_calls: List[Dict[str, Any]]) -> Tuple[List[str], List[str], Any, bool]:
    """Port of parse_deepseek_output (skip_grounding=False).

    Returns (pred_func_calls, gold_func_calls, pred_dict_list, pred_has_parsing_errors). The whole
    reply must be JSON; anything else (prose, a code fence) is a parsing error with no calls."""
    gold_func_calls = [json.dumps(func) for func in _ibm_ground(gold_calls)]
    pred_func_calls: Any = []
    pred_dict_list: Any = []
    has_err = False
    try:
        pred_dict_list = json.loads(generated_text)
        pred_func_calls = (_ibm_ground(pred_dict_list) if "label" in generated_text
                           else pred_dict_list)
        pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except Exception:
        has_err = True
        # A reply that decodes to a bare number/bool/null leaves the undecoded value here, and
        # upstream's calculate_scores then raises on it and aborts the run; score it as no calls.
        if not isinstance(pred_func_calls, list):
            pred_func_calls = []
    return pred_func_calls, gold_func_calls, pred_dict_list, has_err


# --------------------------------------------------------------------------- #
# Match metrics: port of src/scorer.py calculate_scores + src/utils.py (compute_score_sklearn,
# post_process_api_with_args). Do NOT "improve" this to micro/pooled averaging: that was the
# pre-2026-09 gbench behaviour and it does not match the published NESTFUL numbers.
# --------------------------------------------------------------------------- #
_DUMMY_CALL = '{"name": "dummy", "arguments": {}}'


def _ibm_dollar_fix(val: Any) -> Any:
    """scorer.py repair: a pred arg like `$var1.x` (missing its closing `$`) becomes `$var1.x$`."""
    if type(val) == str and val.startswith("$") and not val.endswith("$"):
        return val + "$"
    return val


def _ibm_post_process(api_with_args_gold: List[str],
                      api_with_args_pred: List[str]) -> Tuple[List[str], List[str]]:
    """Verbatim port of utils.post_process_api_with_args: name-align unequal-length call lists."""
    def align_lists(list1, list2):
        aligned1, aligned2, i, j = [], [], 0, 0
        while i < len(list1) or j < len(list2):
            if i < len(list1) and j < len(list2) and list1[i] == list2[j]:
                aligned1.append(list1[i]); aligned2.append(list2[j]); i += 1; j += 1
            elif i < len(list1):
                aligned1.append(list1[i]); aligned2.append(""); i += 1
            else:
                aligned1.append(""); aligned2.append(list2[j]); j += 1
        return aligned1, aligned2

    names_gold = [api.split("(", 1)[0] for api in api_with_args_gold]
    names_pred = [api.split("(", 1)[0] for api in api_with_args_pred]
    if len(names_gold) == len(names_pred):
        return api_with_args_gold, api_with_args_pred
    try:
        names_gold, names_pred = align_lists(names_gold, names_pred)
        g, p = list(api_with_args_gold), list(api_with_args_pred)
        upd_gold = ["" if n == "" else g.pop(0) for n in names_gold]
        upd_pred = ["" if n == "" else p.pop(0) for n in names_pred]
        return upd_gold, upd_pred
    except Exception:                                          # noqa: BLE001
        return api_with_args_gold, api_with_args_pred


def _ibm_macro_prf(gold_output: List[List[str]], pred_output: List[List[str]]) -> Tuple[float, float, float]:
    """Port of utils.compute_score_sklearn: macro P/R/F1, binarizer fit on gold."""
    if not gold_output:
        return 0.0, 0.0, 0.0
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.preprocessing import MultiLabelBinarizer
    b = MultiLabelBinarizer()
    b.fit(gold_output)
    g = b.transform(gold_output)
    p = b.transform(pred_output)
    return (float(precision_score(g, p, average="macro", zero_division=0)),
            float(recall_score(g, p, average="macro", zero_division=0)),
            float(f1_score(g, p, average="macro", zero_division=0)))


def _ibm_macro_f1(gold_output: List[List[str]], pred_output: List[List[str]]) -> float:
    return _ibm_macro_prf(gold_output, pred_output)[2]


def _as_call_strings(calls: Any) -> List[str]:
    return [c if isinstance(c, str) else json.dumps(c) for c in (calls or [])]


def _ibm_score_example(pred_func_calls: List[str], gold_func_calls: List[str]) -> Dict[str, Any]:
    """One iteration of calculate_scores' loop: intent names, slot maps, Partial Match."""
    has_err = False
    gold_names, pred_names = [], []
    for f in pred_func_calls:
        if not f:
            continue
        try:
            if f.strip() == _DUMMY_CALL:
                continue
            f = json.loads(f.replace("<|endoftext|>", "").strip())
            pred_names.append(str(f["name"]))
        except Exception:
            has_err = True
    for f in gold_func_calls:
        if not f:
            continue
        try:
            f = json.loads(f.replace("<|endoftext|>", "").strip())
            gold_names.append(str(f["name"]))
        except Exception:
            pass

    pred_map: Dict[Any, List[str]] = {}
    gold_map: Dict[Any, List[str]] = {}
    for f in pred_func_calls:
        if f.strip() == _DUMMY_CALL:
            continue
        try:
            if not f:
                continue
            f = json.loads(f.replace("<|endoftext|>", "").strip())
            if type(f) != dict or "name" not in f:
                raise ValueError
            api_name = f["name"]
            pred_map[api_name] = []   # set before the args loop, as upstream: survives a bad args
            for arg, val in f["arguments"].items():
                pred_map[f["name"]].append(f"{arg} = {_ibm_dollar_fix(val)}")
        except Exception:
            has_err = True
    for f in gold_func_calls:
        if not f:
            continue
        try:
            f = json.loads(f.replace("<|endoftext|>", "").strip())
            gold_map[f["name"]] = []
            for arg, val in f["arguments"].items():
                gold_map[f["name"]].append(f"{arg} = {val}")
        except Exception:
            pass
    slot_pairs = [(gold_map.get(k, []), pred_map.get(k, []))
                  for k in set(pred_map.keys()).union(gold_map.keys())]

    api_with_args_gold = []
    for f in gold_func_calls:
        f = json.loads(f.replace("<|endoftext|>", "").strip())
        args = ", ".join(sorted([f"{key} = {val}" for key, val in f["arguments"].items()]))
        api_with_args_gold.append(f"{str(f['name'])}({args})")
    api_with_args_pred = []
    for f in pred_func_calls:
        try:
            f = json.loads(f.replace("<|endoftext|>", "").strip())
            f_name = str(f["name"])
            try:
                args = ", ".join(sorted([f"{key} = {_ibm_dollar_fix(val)}"
                                         for key, val in f["arguments"].items()]))
            except Exception:
                args = {}
            api_with_args_pred.append(f"{f_name}({args})")
        except Exception:
            continue
    api_with_args_gold, api_with_args_pred = _ibm_post_process(api_with_args_gold, api_with_args_pred)
    from sklearn.metrics import accuracy_score
    try:
        acc = float(accuracy_score(api_with_args_gold, api_with_args_pred))
    except Exception:                                          # noqa: BLE001 - upstream: 0.0
        acc = 0.0
    return {"gold_intent": gold_names, "pred_intent": pred_names, "slot_pairs": slot_pairs,
            "accuracy_combined": acc, "has_parsing_errors": has_err}


def _ibm_corpus_scores(pred_per_ex: List[List[Any]], gold_per_ex: List[List[Any]],
                       parse_errors: Optional[List[bool]] = None) -> Dict[str, float]:
    """calculate_scores' corpus metrics over parsed, grounded per-example call lists.

    F1 Intent / F1 Slot are sklearn macro scores; Partial Match is the mean per-example
    `accuracy_combined`; Full Match the fraction of examples at exactly 1.0."""
    gold_intent, pred_intent, gold_slot, pred_slot = [], [], [], []
    accs, full, n_err = [], 0, 0
    for i, (pred_calls, gold_calls) in enumerate(zip(pred_per_ex, gold_per_ex)):
        ex = _ibm_score_example(_as_call_strings(pred_calls), _as_call_strings(gold_calls))
        gold_intent.append(ex["gold_intent"])
        pred_intent.append(ex["pred_intent"])
        for g, p in ex["slot_pairs"]:
            gold_slot.append(g)
            pred_slot.append(p)
        accs.append(ex["accuracy_combined"])
        if ex["accuracy_combined"] == 1:
            full += 1
        if ex["has_parsing_errors"] or (parse_errors and parse_errors[i]):
            n_err += 1
    p_i, r_i, f1_i = _ibm_macro_prf(gold_intent, pred_intent)
    p_s, r_s, f1_s = _ibm_macro_prf(gold_slot, pred_slot)
    n = len(accs)
    return {
        "p_intent": p_i, "r_intent": r_i, "f1_intent": f1_i,
        "p_slot": p_s, "r_slot": r_s, "f1_slot": f1_s,
        "partial_match_accuracy": (sum(accs) / n) if n else 0.0,
        "full_match_accuracy": (full / n) if n else 0.0,
        "num_pred_examples_w_parsing_errors": n_err,
    }


# --------------------------------------------------------------------------- #
# Dataset + per-sample verdict
# --------------------------------------------------------------------------- #
def _load_nestful_samples(
    limit: Optional[int] = None,
    data_path: Optional[str] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load NESTFUL from the pinned checkout's data_v2/nestful_data.jsonl (all 1861 rows)."""
    if data_path is None:
        try:
            func_dir: Optional[str] = _func_dir()
        except WinRateUnavailable:
            func_dir = None
        data_path = _dataset(func_dir)
    with open(data_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise RuntimeError("nestful: data_v2/nestful_data.jsonl has no rows")

    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, None, seed="nestful")

    samples = []
    for item in rows:
        if not isinstance(item.get("output"), list) or not item.get("output") \
                or not isinstance(item.get("tools"), list) or "input" not in item \
                or "gold_answer" not in item:
            raise RuntimeError(
                f"nestful: unexpected row schema (sample_id={item.get('sample_id')!r}); "
                "refusing to fabricate sample data")
        # Gold carries the scoring context: the gold call list (match metrics) plus gold_answer
        # and tools as the JSON texts src/eval.py hands to the scorer (executable Win Rate).
        gold = json.dumps({"sample_id": item.get("sample_id"),
                           "output": item["output"],
                           "gold_answer": json.dumps(item["gold_answer"]),
                           "tools": item["tools"]})
        messages = [{"role": "system", "content": _system_prompt(item["tools"])},
                    {"role": "user", "content": item["input"]}]
        samples.append((messages, gold, {"category": "nestful"}))

    logger.info(f"Loaded {len(samples)} nestful samples.")
    return samples


def _gold_sequence(gold_target: str) -> List[Dict[str, Any]]:
    """The raw gold call list from a sample's gold (a legacy bare list is accepted too)."""
    try:
        obj = json.loads(gold_target)
    except Exception:
        return []
    if isinstance(obj, dict) and "output" in obj:
        obj = obj["output"]
    if isinstance(obj, dict):
        obj = [obj]
    return obj if isinstance(obj, list) else []


def _eval_nestful(response_text: str, gold_target: str) -> bool:
    """Per-sample verdict = scorer.py's Full Match for this example (accuracy_combined == 1)."""
    if not response_text or not str(gold_target).strip():
        return False
    gold = _gold_sequence(str(gold_target).strip())
    if not gold:
        return False
    try:
        pred_calls, gold_calls, _, _ = _ibm_parse(response_text.strip(), gold)
        return _ibm_score_example(pred_calls, gold_calls)["accuracy_combined"] == 1
    except Exception:                                          # noqa: BLE001 - malformed gold
        return False


def replies_cut_off(traces: List[Dict[str, Any]], thinking: bool) -> int:
    """Replies stopped by a budget before they finished (finish reason length, an empty reply
    after reasoning included) other than at NESTFUL's 1000: with --thinking every such stop (the
    run's budget), without it a stop below the 1000 (a served context window that left less)."""
    return sum(1 for t in traces if t.get("finish_reason") == "length"
               and (thinking or t.get("max_tokens_sent") != UPSTREAM_MAX_TOKENS))


def _noncanonical_reasons(result: Dict[str, Any], limit: Optional[int]) -> List[str]:
    reasons = []
    traces = result.get("sample_traces") or []
    left_out = len(traces) - sum(1 for _t in scorable_traces(result))
    if left_out:
        kinds = result.get("request_harness_errors") or {}
        why = ("harness errors: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
               if kinds else "no reply to score")
        if left_out == len(traces):
            reasons.append(f"nothing was scored: none of the {len(traces)} requests got a reply "
                           f"({why}); every metric is null")
        else:
            reasons.append(f"{left_out} of {len(traces)} requests got no reply and are left out "
                           f"of every metric ({why}); NESTFUL scores every row")
    win_entry = result.get("win_rate_metric")
    if win_entry:
        reasons.append(f"NESTFUL's Win Rate was not scored: its prerequisite "
                       f"({win_entry.get('prerequisite') or _FUNCTIONS_PREREQ}) was missing "
                       f"(infra_required: {str(win_entry['infra_required']).rstrip('. ')})")
    if limit:
        reasons.append(f"--eval-limit {limit} (NESTFUL scores all 1861 rows)")
    if parse_shard(os.environ.get("GBENCH_SHARD")):
        reasons.append("sharded run (a subset of the rows)")
    temp = result.get("temperature")
    official = result.get("official_sampling")
    matches = (official.get("temperature_matches_official") if official
               else temp is not None and float(temp) == 0.0)
    if not matches:
        reasons.append(f"temperature {temp} (upstream src/eval.py decodes at 0.0)")
    if result.get("thinking"):
        reasons.append("thinking enabled (upstream generates without a reasoning channel)")
    if result.get("decoding_penalties"):
        reasons.append("decoding penalties set")
    attempts = result.get("attempts") or {}
    if (attempts.get("attempts_per_sample") or 1) > 1:
        reasons.append("--attempt-count > 1 (metrics pool every attempt)")
    blocked = int(result.get("win_rate_network_blocked_calls") or 0)
    if blocked:
        ids = [str(s) for s in result.get("win_rate_network_blocked_samples") or []]
        named = ", ".join(ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")
        reasons.append(f"{blocked} Win Rate call(s) failed a use of the network, which the "
                       "worker's sandbox withholds (harness errors, left out of the Win Rate"
                       + (f": {named}" if named else "") + "); upstream's scorer runs them with "
                       "network")
    killed = int(result.get("win_rate_killed_calls") or 0)
    if killed:
        reasons.append(f"{killed} Win Rate call(s) killed by the 30 s backstop and scored as "
                       "failures (upstream would still be running them)")
    cut = int(result.get("replies_cut_off") or 0)
    if cut:
        reasons.append(f"{cut} replies cut off by the token budget before they finished (not at "
                       f"NESTFUL's max_tokens {UPSTREAM_MAX_TOKENS})")
    return reasons


def run_nestful(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute nestful; headline = Full Match Accuracy, with Partial Match, F1 Intent / F1 Slot and
    the executable Win Rate, all computed as IBM/NESTFUL src/scorer.py computes them, over the rows
    that got a reply (none: every metric null). Without the reference functions or the worker's
    sandbox the Win Rate alone is reported infra_required (win_rate_metric) and the other metrics
    are computed. A Win Rate call that did not win and failed a use of the network the sandbox
    withholds is a harness error, left out of the Win Rate (win_rate_network_blocked_calls), on a
    row whose gold needs the network (_GOLD_NEEDS_NETWORK); on any other row it is scored and
    counted (win_rate_network_blocked_offpath)."""
    # Fail fast, before the rollout, on anything the scorer needs.
    win_problem: Optional[str] = None
    win_prereq = _FUNCTIONS_PREREQ
    try:
        func_dir: Optional[str] = _func_dir()
    except WinRateUnavailable as e:
        func_dir, win_problem = None, e.reason
    data_path = _dataset(func_dir)
    if importlib.util.find_spec("sklearn") is None:
        raise infra_required("nestful", "scikit-learn is required by scorer.py's metrics "
                             "(pip install scikit-learn).", DOCS_URL)
    # The sandboxed Win Rate worker starts (and its isolation is checked) before any request is
    # sent. Without the reference functions, or without a sandbox, the Win Rate is not computed
    # (scorer.py calculate_scores with win_rate_flag=False); the other metrics are.
    worker: Optional[_WinRateWorker] = None
    if func_dir is not None:
        try:
            worker = _WinRateWorker(func_dir)
        except WinRateUnavailable as e:
            win_problem, win_prereq = e.reason, e.prerequisite
    if win_problem is not None:
        logger.error("nestful: NESTFUL's Win Rate is infra_required and not scored (%s); F1 "
                     "Intent, F1 Slot, Partial and Full Match are computed",
                     win_problem.rstrip(". "))
    pred_per_ex: List[List[str]] = []
    gold_per_ex: List[List[str]] = []
    parse_errors: List[bool] = []
    win_ok = 0
    #: rows whose Win Rate was scored, and those left out as harness errors (their sample ids)
    win_scored = 0
    blocked_ids: List[Any] = []
    #: scored rows whose losing call failed a use of the network off the gold path
    offpath_ids: List[Any] = []
    sandbox_facts: Optional[Dict[str, Any]] = None
    try:
        samples = _load_nestful_samples(limit=kwargs.get("limit"), data_path=data_path)
        limit_plan = base_suite_plan("nestful", samples, kwargs.get("limit"), run_attempts(kwargs))
        result = run_eval_suite(
            eval_name="nestful",
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=samples,
            eval_fn=_eval_nestful,
            thinking=enable_thinking,
            extra_payload=kwargs.get("extra_payload"),
            limit=kwargs.get("limit"),
            max_output_tokens=kwargs.get("max_output_tokens"),
            official_sampling=NESTFUL_SAMPLING,
        )
        result["eval_limit_plan"] = limit_plan
        for t in scorable_traces(result):
            gold_str = str(t.get("gold_answer") or "").strip()
            gold_calls = _gold_sequence(gold_str)
            if not gold_calls:
                continue
            gspec = json.loads(gold_str)
            # src/eval.py strips the generation before scoring; reasoning tags are removed first,
            # as for the per-sample verdict.
            resp = strip_thinking_tags(t.get("response_text") or "").strip()
            pred_calls, gold_func_calls, pred_dicts, has_err = _ibm_parse(resp, gold_calls)
            pred_per_ex.append(pred_calls)
            gold_per_ex.append(gold_func_calls)
            parse_errors.append(has_err)
            if worker is None:
                continue
            # The Win Rate executes the reply as parsed, with the model's own labels (ungrounded).
            try:
                win = worker.score(pred_dicts, gspec.get("gold_answer"), gspec.get("tools") or [])
            except WinRateUnavailable as e:
                # The worker was restarted after a kill and its sandbox did not come back: the
                # Win Rate is withheld as a whole, never scored in part or outside the sandbox.
                worker.close()
                worker, win_problem, win_prereq, win_ok = None, e.reason, e.prerequisite, 0
                win_scored, blocked_ids, offpath_ids = 0, [], []
                for other in result.get("sample_traces") or []:
                    other.pop("nestful_win", None)
                    other.pop("nestful_win_blocked_by_sandbox", None)
                    other.pop("nestful_win_network_blocked_offpath", None)
                logger.error("nestful: NESTFUL's Win Rate is not scored (%s)",
                             win_problem.rstrip(". "))
                continue
            blocked = getattr(worker, "last_blocked", None)
            if blocked and gspec.get("sample_id") in _GOLD_NEEDS_NETWORK:
                # The row's gold needs the network the sandbox withholds, so the verdict may have
                # come from the sandbox rather than the reply: a harness error, left out of the
                # Win Rate and named.
                t["nestful_win_blocked_by_sandbox"] = list(blocked)
                blocked_ids.append(gspec.get("sample_id"))
                continue
            if blocked:
                # The row's gold needs no network: the reply chose a call off the gold path that
                # uses it. Its verdict is the reply's (a loss), counted.
                t["nestful_win_network_blocked_offpath"] = list(blocked)
                offpath_ids.append(gspec.get("sample_id"))
            t["nestful_win"] = bool(win)
            win_scored += 1
            win_ok += 1 if win else 0
    finally:
        if worker is not None:
            sandbox_facts = getattr(worker, "facts", None)
            worker.close()
    result["win_rate_killed_calls"] = worker.kills if worker is not None else None
    result["win_rate_network_blocked_calls"] = len(blocked_ids) if worker is not None else None
    if worker is not None and blocked_ids:
        result["win_rate_network_blocked_samples"] = blocked_ids
    result["win_rate_network_blocked_offpath"] = len(offpath_ids) if worker is not None else None
    if worker is not None and offpath_ids:
        result["win_rate_network_blocked_offpath_samples"] = offpath_ids
    result["replies_cut_off"] = replies_cut_off(result.get("sample_traces") or [],
                                                bool(enable_thinking))

    n = len(pred_per_ex)
    metric_keys = ("f1_intent", "f1_slot", "p_intent", "r_intent", "p_slot", "r_slot",
                   "partial_match_accuracy", "full_match_accuracy")
    if n:
        scores = _ibm_corpus_scores(pred_per_ex, gold_per_ex, parse_errors)
        for key in metric_keys:
            result[key] = round(scores[key] * 100.0, 2)
        result["parsing_errors"] = scores["num_pred_examples_w_parsing_errors"]
    else:
        # No row was answered (every request a harness error): nothing to score, so every metric
        # is withheld (null), never reported as 0.
        for key in metric_keys:
            result[key] = None
        result["parsing_errors"] = None
    if worker is not None:
        result["win_rate"] = round(win_ok / win_scored * 100.0, 2) if win_scored else None
        if n and not win_scored:
            result["win_rate_withheld_reason"] = (
                f"no Win Rate call was scored: the calls of all {len(blocked_ids)} answered rows "
                "failed a use of the network, which the worker's sandbox withholds (harness "
                "errors)")
        result["win_rate_sandbox"] = {"sandbox": "bubblewrap", "network": "none (loopback only)",
                                      "worker_reported": sandbox_facts,
                                      "note": _SANDBOX_NETWORK_NOTE}
        result["metric"] = ("Full Match Accuracy (IBM/NESTFUL scorer.py "
                            "'percentage_times_full_score'); Partial Match Accuracy, F1 Intent / "
                            "F1 Slot (sklearn macro) and the executable Win Rate also reported.")
    else:
        # Not scored, never dropped: the Win Rate's own entry names the missing prerequisite.
        result["win_rate"] = None
        result["win_rate_metric"] = win_rate_infra_required(win_problem or "", win_prereq)
        result["metrics_infra_required"] = ["win_rate"]
        result["metric"] = ("Full Match Accuracy (IBM/NESTFUL scorer.py "
                            "'percentage_times_full_score'); Partial Match Accuracy and F1 "
                            "Intent / F1 Slot (sklearn macro) also reported; the executable Win "
                            "Rate was not scored (infra_required, see win_rate_metric).")
    result["dataset_source"] = "checkout"
    result["dataset_revision"] = f"IBM/NESTFUL@{NESTFUL_COMMIT} data_v2/nestful_data.jsonl"
    if n:
        result["accuracy"] = result["full_match_accuracy"]
    else:
        result["accuracy"] = None
        if not result.get("accuracy_withheld_reason"):
            result["accuracy_withheld_reason"] = (result.get("status_reason")
                                                  or "nothing was scored: no row was answered")
        if result.get("status") in ("success", "completed_with_errors"):
            result["status"] = "failed"
            result["status_reason"] = result["accuracy_withheld_reason"]
    if worker is None and result.get("status") == "success":
        result["status"] = "completed_with_errors"
    if worker is not None and blocked_ids:
        why = (f"{len(blocked_ids)} Win Rate call(s) failed a use of the network, which the "
               "worker's sandbox withholds (harness errors, left out of the Win Rate)")
        if result.get("status") == "success":
            result["status"] = "completed_with_errors"
        result["status_reason"] = "; ".join(x for x in (result.get("status_reason"), why) if x)
    reasons = _noncanonical_reasons(result, kwargs.get("limit"))
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) if reasons else None
    return result
