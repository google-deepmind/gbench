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

# Evaluation Suite: acebench
# Description: ACEBench (arXiv:2501.12851) - tool-use function calling: Normal, Special and Agent, English and Chinese

"""gbench native built-in runner for acebench (Tool Use & Function Calling).

Canonical ACEBench (chenchen0103/ACEBench) as the repo's generate.py + eval_main.py run it, in both
published settings (its English and Chinese leaderboards, `--language en|zh`), with every category:
Normal (12 sub-categories), Special (3) and Agent (agent_multi_turn, agent_multi_step).

Normal and Special: every row of data_all/data_<lang>/ at the pinned commit is sent with the
official system prompt (model_inference/prompt_en.py or prompt_zh.py, vendored in
_acebench_vendor/; the functions rendered by str.format, i.e. as a Python list) and
USER_PROMPT_<LANG>, and scored by the official model_eval code vendored below: normal rows by
decode_ast + normal_checker (single-turn replies through extract_outermost_bracket_content,
multi-turn replies with all whitespace removed), special rows by eval_main.special_eval's
substring rules.

Agent: common_inference.CommonInference.multi_turn_inference / multi_step_inference replayed turn
by turn. The model under test answers the agent prompts (model_inference/multi_turn/common_agent.py
with TRAVEL_PROMPT / BASE_PROMPT, multi_step/common_agent_step.py), its calls run against the
upstream scenario classes (message, reminder, food platform, travel; vendored in _acebench_vendor/)
through a replica of execute_agent_func_call, and in agent_multi_turn a user simulator
(multi_turn/APIModel_user.py APIUSER, upstream gpt-4o) plays the user. The user simulator is the
one replaced component: gbench's Gemini judge cascade answers its prompts, at APIUSER's own
sampling (temperature 0.001, top_p 1, max_tokens 1000 for the reply; recorded under
`user_simulator`). gpt-4o does not think and the Gemini simulator counts its thinking against
max_output_tokens, so it is sent 1000 + gemini_budget.GEMINI_THINKING_ALLOWANCE (8192): its reply
keeps upstream's 1000 tokens. Replies that still stop there (used as they are), replies that stop
there with no text (the cascade asks its next model) and simulator outages whose cascade saw such
replies are counted and make the run not leaderboard-comparable. The final scenario state is scored
by eval_main.agent_eval (agent_checker against the possible answer, end to end) and
agent_eval_process (milestones).
Only agent_multi_turn needs the user simulator: without a Gemini key (unset, or rejected by a live
check) that category is reported infra_required in its own entry (`agent_multi_turn_category`,
listed in `categories_infra_required`; never run or scored) while the other 16 categories run;
agent_summary and the Summary, which weigh it, are then null, and the status is
completed_with_errors (a hard error only when agent_multi_turn is all the run selected).

Aggregation follows evaluation_helper.generate_result_csv per language: accuracy per sub-category
(multi-turn Normal scored per conversation), normal / special / agent summaries as unweighted
means, and Summary = 0.2676 x special + 0.578 x normal + 0.1545 x agent, the leaderboard's
"overall". The headline is the English Summary; the Chinese one is `overall_zh`.

Model calls are never run as code in gbench's process: upstream eval()s the call strings it builds;
here each call is dispatched to the scenario method directly, and anything else upstream's eval()
of a call string would run (a nested call of a builtin, for instance) is refused and counted
(`execution_guard_refusals`; a run with any is not leaderboard-comparable). A BinOp argument
(`'code ' + str(42)`, `[1] + [2]`, `1 << 3`), which upstream resolves with
eval(ast.unparse(node)), is evaluated exactly that way: arithmetic over literals in-process, any
other BinOp by Python's own eval in a bubblewrap container of its own (whatever GBENCH_SANDBOX
says): new user, PID, IPC, UTS, network and cgroup namespaces, no capabilities, an empty
environment, a filesystem holding only the interpreter (/usr and its own directories read-only, a
private /dev and /tmp; no /proc, no /run and so no Docker socket, no home or cache directory), and
CPU, memory, file-size and time limits. The eval sees the globals upstream's eval sees there (the
module's stdlib imports, its name). The value comes back as a literal; a builtin exception as the
same exception, any other as an Exception subclass of the same name and text (upstream's callers
catch Exception). A value that depends on the evaluating process (its working directory, files,
environment or clock) comes from the container, not from upstream's process. Only what the
isolated eval cannot give as upstream would (a name only upstream's process has, a result that is
not a literal, an evaluation past the limits, a literal product or power too large to compute) is
refused and counted the same way. The container is checked before anything is generated (no
bubblewrap, or a container that is not isolated so, is a hard error).

Sampling: ACEBench's settings for a locally served open-weight model (model_inference/model_infer.py
LLMInfer, the class generate.py builds for such a model): temperature 0.0 (gbench's rule for every
suite, base.resolve_temperature: 0.0 without `--thinking`, 1.0 with it), top_p 0.9 in both modes, and max_tokens=1024 on every request of the model under test in a no-think
run, within LLMInfer's 8192-token window (max_model_len) as its pinned vLLM 0.6.1.post1 applies
it: a request whose prompt leaves less room gets that room, and a prompt longer than the window is
not sent, its reply is the empty text that vLLM returns for it, and it is counted
(upstream_window_budget). A `--thinking` run applies neither (both were written
for non-reasoning generation): each request gets the run's --max-output-tokens, else the thinking
floor, and the result records the cap as not applied (base.resolve_token_budget). Replies cut off
by any other budget, an empty reply after reasoning included, are counted and make the run not
leaderboard-comparable. Override the temperature for a whole run with `--temperature`, or for
this suite alone with `GBENCH_ACEBENCH_TEMPERATURE`, which takes precedence over both; the result
records upstream's value under `official_sampling`.

The single-turn rows are scored after generation in worker threads (_row_scorer), so an evaluator
container never stalls the event loop; the agent conversations decode their messages in worker
threads too.
"""

import ast
import asyncio
import builtins
import contextvars
import copy
import functools
import hashlib
import importlib
import inspect
import json
import logging
import operator
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import base
from .base import (OVER_CONTEXT_ERROR, OfficialSampling, apply_shard, clamp_to_context_async,
                   decoding_penalties, effective_chat_template_kwargs, gbench_cache_dir,
                   gemini_api_keys, gemini_key_live_valid, get_run_knob, judge_cascade,
                   request_timeout_s, run_eval_suite, strip_thinking_tags)
from gbench.third_party._acebench_vendor import multi_step_prompts as _MS
from gbench.third_party._acebench_vendor import multi_turn_prompts as _MT
from gbench.third_party._acebench_vendor import prompt_en as _PROMPT_EN
from gbench.third_party._acebench_vendor import prompt_zh as _PROMPT_ZH
from .gemini_budget import (GEMINI_THINKING_ALLOWANCE, gemini_output_limit,
                            merge_with_cutoff_reason, output_limit_record)
from .metrics import scorable_traces
from .fc_common import eval_limit_plan, part_quotas
from .sampling import parse_shard, stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/acebench.md"

#: chenchen0103/ACEBench revision (HEAD of main, 2025-10-29). It carries 5 ground-truth, 1 time and
#: 1 question fix that the older HF reformatting (oliveirabruno01/acebench) lacks.
ACEBENCH_COMMIT = "56dd66cf6439b0d9655ee1b353e4cd745c6f664e"
_RAW_URL = ("https://raw.githubusercontent.com/chenchen0103/ACEBench/{commit}/data_all/"
            "data_{language}/{rel}")

#: The two published settings (generate.py / eval_main.py --language); English is the headline.
LANGUAGES = ("en", "zh")

#: LLMInfer's max_tokens, the cap every request of the model under test carries upstream.
UPSTREAM_MAX_TOKENS = 1024
#: model_inference/model_infer.py LLMInfer.__init__: SamplingParams(temperature=0.0, max_tokens=1024,
#: top_p=0.9): get_model builds it (directly, or as its Llama subclass) for a model served from
#: local weights (--model-path); the hosted-API classes send generate.py's --temperature 0.7 and
#: --top-p 1 instead. The token-limit policy (base.resolve_token_budget) sends the 1024 on a no-think
#: run (less where LLMInfer's window leaves less, UPSTREAM_MAX_MODEL_LEN) and does not apply it to a
#: --thinking run (a cap written for non-reasoning generation), which gets the run's own budget;
#: ACEBench defines no reasoning-model settings.
ACEBENCH_SAMPLING = OfficialSampling(
    reference=(f"chenchen0103/ACEBench@{ACEBENCH_COMMIT[:12]} model_inference/model_infer.py "
               "LLMInfer (SamplingParams temperature=0.0, max_tokens=1024, top_p=0.9)"),
    temperature=0.0,
    params={"top_p": 0.9},
    max_tokens=UPSTREAM_MAX_TOKENS,
)

#: model_inference/model_infer.py LLMInfer.__init__: LLM(..., max_model_len=8192), the context
#: window of ACEBench's own vLLM (requirements.txt: vllm==0.6.1.post1, V0 engine) for a model
#: served from local weights. What that engine does with it (vllm-project/vllm@v0.6.1.post1,
#: commit acda0b35d00e): chunked prefill stays off below 32K (engine/arg_utils.py), so the
#: scheduler's prompt limit is min(8192, max_num_batched_tokens = max(8192, 2048)) = 8192
#: (config.py SchedulerConfig, core/scheduler.py _get_prompt_limit). A text prompt of more than
#: 8192 tokens is not rejected (engine/llm_engine.py _validate_model_inputs checks the length
#: only for multimodal models): the scheduler marks it FINISHED_IGNORED with a warning
#: (core/scheduler.py _schedule_prefills) and the engine still returns its output, empty text
#: with finish reason "length" (llm_engine.py _process_model_outputs, sequence.py
#: get_finished_reason). LLMInfer.inference returns that "" and upstream's run goes on: a
#: single-turn row is scored on "", an agent's step hands "" to the user or the next step. A
#: prompt of P <= 8192 tokens is generated until its length exceeds 8192
#: (engine/output_processor/stop_checker.py), so it gets at most 8193 - P tokens. A no-think run
#: emulates that window (upstream_window_budget); a --thinking run does not apply it either, as
#: it does not apply the 1024 cap.
UPSTREAM_MAX_MODEL_LEN = 8192
#: The longest a sequence can grow under that window: the stop check fires once the length is
#: greater than max_model_len, on the token that makes it 8193.
_UPSTREAM_SEQUENCE_CAP = UPSTREAM_MAX_MODEL_LEN + 1
#: A bound on the tokens a chat template (and a BOS the tokenizer adds) puts around a two-message
#: prompt, used only to skip counting a prompt that cannot come near the window (a token is at
#: least one byte of text).
_TEMPLATE_BOUND_TOKENS = 512
#: The /tokenize fields that count a prompt as upstream's vLLM does: LLMInfer renders the chat
#: template to text (apply_chat_template(tokenize=False, add_generation_prompt=True)) and vLLM
#: 0.6.1 encodes that text with tokenizer.encode(prompt), whose add_special_tokens defaults to
#: True (transformers_utils/tokenizer_group/tokenizer_group.py TokenizerGroup.encode), so a
#: tokenizer that adds a BOS adds it on top of the template's.
_UPSTREAM_TOKENIZE_FIELDS = {"add_special_tokens": True}


def upstream_window_budget(messages: List[Dict[str, Any]], cap: int,
                           chat_template_kwargs: Optional[Dict[str, Any]] = None
                           ) -> Tuple[int, Optional[int], str]:
    """(max_tokens, prompt tokens, how) of a no-think request under LLMInfer's window: `cap`, the
    room the window leaves after the prompt when that is less, or 0 for a prompt longer than the
    window, which upstream's vLLM ignores (no request is sent; the reply is "").

    `how` is "short" when the prompt cannot come near the window (its text in bytes plus
    _TEMPLATE_BOUND_TOKENS plus `cap` fits; not counted); "server" when the served model's
    /tokenize counted it exactly as upstream's vLLM does (its chat template, then the text encoded
    with special tokens: _UPSTREAM_TOKENIZE_FIELDS); "tokenizer" when the run's tokenizer plus a
    template allowance, a conservative count, puts it clearly inside the window; or "uncounted"
    when the prompt may be near the window and no exact count was possible (only a character
    estimate, or a conservative count that is near the window): `cap` is used and the run is not
    comparable."""
    text_bytes = sum(len(str(m.get("content") or "").encode("utf-8")) for m in messages)
    if text_bytes + _TEMPLATE_BOUND_TOKENS + cap <= _UPSTREAM_SEQUENCE_CAP:
        return cap, None, "short"
    try:
        bounds = base._prompt_token_bounds(messages, None, chat_template_kwargs,
                                           prefill=dict(_UPSTREAM_TOKENIZE_FIELDS))
    except Exception:                              # noqa: BLE001 - counted as not counted
        bounds = None
    if bounds is None or bounds[2] not in ("server", "tokenizer"):
        return cap, None, "uncounted"
    tokens = int(bounds[1])
    if bounds[2] == "tokenizer":
        # A conservative count decides nothing near the window: the prompt may be shorter.
        if tokens + cap <= _UPSTREAM_SEQUENCE_CAP:
            return cap, tokens, "tokenizer"
        return cap, None, "uncounted"
    if tokens > UPSTREAM_MAX_MODEL_LEN:
        return 0, tokens, "server"
    return min(cap, _UPSTREAM_SEQUENCE_CAP - tokens), tokens, "server"


#: category.py ACE_DATA_CATEGORY["normal"] / ["special"] / ["agent"], in upstream order.
NORMAL_CATEGORIES = [
    "normal_single_turn_single_function", "normal_single_turn_parallel_function",
    "normal_multi_turn_user_adjust", "normal_multi_turn_user_switch", "normal_similar_api",
    "normal_preference", "normal_atom_bool", "normal_atom_enum", "normal_atom_number",
    "normal_atom_list", "normal_atom_object_deep", "normal_atom_object_short",
]
SPECIAL_CATEGORIES = ["special_incomplete", "special_error_param", "special_irrelevant"]
AGENT_CATEGORIES = ["agent_multi_step", "agent_multi_turn"]

#: git blob SHA-1 of (data file, possible_answer file) per category at ACEBENCH_COMMIT, English
#: (data_all/data_en/) and Chinese (data_all/data_zh/).
_PINNED_BLOBS = {
    "normal_single_turn_single_function": ("ed01f8043c749a4a00d77b6bf0e8d314381eb3db", "6f88a3fe31a066b57d9ed6e14ee579444fb0a70f"),
    "normal_single_turn_parallel_function": ("d50841cf81fa58891e3ba9704bacb484c90cf235", "14ea2a7228d1fb397f88b536f9b75e4e5590ac75"),
    "normal_multi_turn_user_adjust": ("8558a6e0443def265bb288d84d44515eb0a854df", "699cd06abeb5d23a576e5d28ceb9a2438a5004db"),
    "normal_multi_turn_user_switch": ("66e2d79c29e128e3acd7628a7bb37bfda7738056", "5f62ec775408076e43b956d3117aebeb079d98ab"),
    "normal_similar_api": ("531df4c635d997224fc2636061d4c8c5bd58ad9c", "71ad71098133835f3fc2ddf25a0155eb6e105d9f"),
    "normal_preference": ("5c499fb7c23859a9b0e9c280f3fcbf45c87e538e", "29240ae5cc362ec5408031296088f9e54aabb87b"),
    "normal_atom_bool": ("c00d5d9abc01fb0745c47fa05f02b0b7e52d695d", "bb203728ceeeabc3fbabbd02970d644dc4691f8d"),
    "normal_atom_enum": ("4d2c8413b173208c43d97f364dad8e39cb412bed", "197bb4e079d3af0a3c3fef0561a8ed2f5652b376"),
    "normal_atom_number": ("47c135b7846991ed99940f5ba5b070e0e262f893", "3b4041b4f8c5a9f57f75ed069cd71489f0ccab8a"),
    "normal_atom_list": ("30fb28336370ea49a41fa4a082c849f8911bb82a", "fd2e8bc7cef1c3dff9c92f7da6e5071deeb59c06"),
    "normal_atom_object_deep": ("3eb171b80bae6bc7bdb95f3d69ac12cb3025cf13", "ea299bf3970841e858a936cc6014380132405f30"),
    "normal_atom_object_short": ("97a6310c60123fd134cfe27e2a2504507657b5cb", "76b0c1f1aefd68b3b4684d5c5e778e480ab4c9be"),
    "special_incomplete": ("8a153c7029dc4c082370a648e5954bfe83b879ae", "3ce8de3722b4852d6111e7a713efb9105b83176b"),
    "special_error_param": ("da29b2b1e72c0b8278cfae29ede448dd27751d94", "9c7d95186c02cd7b3344b739872fd51583d02790"),
    "special_irrelevant": ("c98112ad9bf9aa1b2f509854247badb424f98440", "d515bda2439b8859f369af351376d755335eaccb"),
    "agent_multi_turn": ("267f30a65ee34b5635986053bf51bf2cb7aa2dbf", "b037c9dda02308133aeb9a3f366477b7c2c8b0b5"),
    "agent_multi_step": ("ebb09a4657232801de246bef83501effe1c45e49", "157612102e5f6994412e7ca10ab109682e6d8cbf"),
}
_PINNED_BLOBS_ZH = {
    "normal_single_turn_single_function": ("9ba10a383b5f0f5ab295d0e19b37e80843d6f827", "3897327b0660f11ede9e9a5da0059b47a3a7e109"),
    "normal_single_turn_parallel_function": ("19e41d31dc6cd1924e967df91455553bc881bb6b", "61a02a648671442fd873e98859059592bfcd4dd8"),
    "normal_multi_turn_user_adjust": ("7b98bd95621f2557ec135fe17ee7539da6c05430", "65cb7ab862757808ea2fd34d3a5dfcb290c5303e"),
    "normal_multi_turn_user_switch": ("dd6495aafeb5512d7a3dac0bbbfe1906b9fbea21", "d9866598d69f733f19407d3303605c37801cdaad"),
    "normal_similar_api": ("9b49723938345040ba164c9aa72988dedeb2f4b3", "edfbed06ad40a168d5cd4dd88963b020e0f161cb"),
    "normal_preference": ("0962853152cb5aac8d0bbd868c1d8928a1da3274", "27542d62cd7366f85030b2bd8b31fb48b8e81142"),
    "normal_atom_bool": ("7cb9e3301e67b94d96d9b238273cc1218fd88b57", "9bd39dafe390afdee9a881e4be3a9a5453e03911"),
    "normal_atom_enum": ("2c8471fa0e9cabf06f8fdb9ab10a954f6d373acb", "197ee9d62882f57f801c9052adfbd6bd1b570a26"),
    "normal_atom_number": ("7e697a25c13bede91d843417dcdde5bcf9b570d6", "7fe48ab7e2ca118f853e24bf79ddd9cc05259d6a"),
    "normal_atom_list": ("cea5f7c955e296a776634164569c56f932b2fa65", "d11af08aeec8d3e409d3fe0d33a0e726d35901cf"),
    "normal_atom_object_deep": ("8a858101ed906a993aebfe84f2ee3544f09b97f0", "cfea8c25a545269ca558623ef3a361a18dc81882"),
    "normal_atom_object_short": ("dfa547973379f86965790b26ae5887ec1442e6b5", "50f3983e3813d843cef4c008366b3bad4a08ea00"),
    "special_incomplete": ("f343ddb7a5194857491b72ac8f5dbd567aa1333e", "c2b149561ee960a80cea6d130a5227ed0a71e24c"),
    "special_error_param": ("c84a6d0c97273a9a05d0b287dab8568c0e5efa3b", "632cba19d787d02e1cf502bb2bcbb53e2889687e"),
    "special_irrelevant": ("e032ba495ad54936cae2dd4abb2a838c52a2e177", "d515bda2439b8859f369af351376d755335eaccb"),
    "agent_multi_turn": ("a9e87be1d20fe29fd757d40ebe61ee15313cde31", "834d4863b72a9c7879ce9a94f25fc97d80a6dacf"),
    "agent_multi_step": ("d11d7a923e5c783104dcdd764edfb2926b743516", "aca1bd11696c1f7b4a60055cf60d0a3f10ab0b00"),
}
PINNED_BLOBS = {"en": _PINNED_BLOBS, "zh": _PINNED_BLOBS_ZH}
#: evaluation_helper.generate_result_csv Summary weights.
SUMMARY_WEIGHTS = {"special": 0.2676, "normal": 0.578, "agent": 0.1545}
#: Normal + Special rows and agent conversations per language at ACEBENCH_COMMIT.
ROW_COUNTS = {"en": 973, "zh": 967}
AGENT_COUNTS = {"en": 50, "zh": 50}

# ---- Official single-turn prompts (model_inference/prompt_en.py / prompt_zh.py, vendored) ----
_SYS_NORMAL = _PROMPT_EN.SYSTEM_PROMPT_FOR_NORMAL_DATA_EN
_SYS_PREFERENCE = _PROMPT_EN.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN
_SYS_SPECIAL = _PROMPT_EN.SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN
_USER_PROMPT = _PROMPT_EN.USER_PROMPT_EN
#: language -> (normal, preference, special, user) templates, as common_inference.single_turn_inference.
_PROMPTS = {
    "en": (_SYS_NORMAL, _SYS_PREFERENCE, _SYS_SPECIAL, _USER_PROMPT),
    "zh": (_PROMPT_ZH.SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH, _PROMPT_ZH.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH,
           _PROMPT_ZH.SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH, _PROMPT_ZH.USER_PROMPT_ZH),
}


# --------------------------------------------------------------------------- #
# Vendored from model_eval/checker.py + model_eval/utils.py (MIT, chenchen0103/ACEBench @
# ACEBENCH_COMMIT). The logic is kept line for line so the verdicts match eval_main.py, quirks
# included (for instance "enum" maps to `enumerate`, so an enum value is only type-checked against
# the answer); only the error message strings, which never affect a verdict, are shortened.
# --------------------------------------------------------------------------- #
PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
    "list": list,
    "object": dict,
    "objectArray": list,
    "list(string)": list,
    "list(enum)": list,
    "int": int,
    "enum": enumerate,
    "number": int,
}
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple", "list(string)", "list(enum)", "object", "objectArray"]


def standardize_string(input_string: str) -> str:
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def get_possible_answer_type(possible_answer):
    if possible_answer != "":  # Optional parameter
        return type(possible_answer)
    return None


def find_description(func_descriptions, name):
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] in name:
                return func_description
        return None
    else:
        return func_descriptions


def sum_key_list(data):
    key_counter = Counter()
    for dictionary in data:
        key_counter.update(dictionary.keys())
    return dict(key_counter)


def flatten_dates(d):
    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in d.items()}


def is_function_call_format_valid(decoded_output):
    if type(decoded_output) == list:
        for item in decoded_output:
            if type(item) != dict:
                return False
        return True
    return False


def type_checker(param, value, possible_answer, expected_type_description,
                 expected_type_converted, nested_type_converted, func_name):
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error"}
    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type != None:
        if possible_answer_type != expected_type_converted:
            is_variable = True
    if value == "true":
        value = True
    if value == "false":
        value = False
    if type(value) == expected_type_converted:
        if nested_type_converted == None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(param, value_item, possible_answer_item,
                                                      str(nested_type_converted),
                                                      nested_type_converted, None, func_name)
                        if not checker_result["valid"]:
                            flag = False
                            break
                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}
            result["valid"] = False
            result["error"] = ["nested type checking failed"]
            result["error_type"] = "type_error"
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type != None:
        if type(value) == possible_answer_type or possible_answer == value:
            result["is_variable"] = True
            return result
    result["valid"] = False
    result["error"] = [f"wrong type for parameter ({param}) of api ({func_name})"]
    result["error_type"] = "type_error"
    return result


def string_checker(param, model_output, possible_answer, function, question, test_category):
    standardize_model_output = standardize_string(model_output)
    standardize_possible_answer_item = standardize_string(possible_answer)
    if "agent" in test_category:
        if standardize_model_output != standardize_possible_answer_item:
            return {"valid": False, "error": ["wrong value"], "error_type": "value_error:string"}
    else:
        if standardize_possible_answer_item not in standardize_model_output:
            return {"valid": False, "error": ["wrong value"], "error_type": "value_error:string"}
    return {"valid": True, "error": []}


def list_checker(param, model_output, possible_answer, func_name):
    standardize_model_output = list(model_output)
    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) == str:
            standardize_model_output[i] = standardize_string(model_output[i])
    standardize_possible_answer = []
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(standardize_string(possible_answer[i]))
        else:
            standardize_possible_answer.append(possible_answer[i])
    if standardize_model_output != standardize_possible_answer:
        return {"valid": False, "error": ["wrong value"], "error_type": "value_error:list/tuple"}
    return {"valid": True, "error": []}


def dict_checker(param, model_output, possible_answers, func_name):
    result = {"valid": True, "error": [], "error_type": "dict_checker:unclear"}
    possible_answer = possible_answers
    if not isinstance(model_output, dict):
        result["valid"] = False
        result["error_type"] = "value_error"
        return result
    else:
        if len(list(model_output.keys())) != len(list(possible_answer.keys())):
            result["valid"] = False
            result["error_type"] = "value_error"
            return result
        for key, value in model_output.items():
            if value == "true":
                value = True
            if value == "false":
                value = False
            if key not in possible_answer:
                result["valid"] = False
                result["error_type"] = "value_error"
                return result
            expected_values = possible_answer[key]
            if isinstance(expected_values, dict):
                result = dict_checker(param, value, expected_values, func_name)
                if not result["valid"]:
                    return result
            else:
                standardize_value = value
                if type(value) == str:
                    standardize_value = standardize_string(value)
                standardize_possible_answer = []
                if type(possible_answer[key]) == str:
                    standardize_possible_answer.append(standardize_string(possible_answer[key]))
                else:
                    if type(possible_answer[key]) == dict:
                        standardize_possible_answer.append(flatten_dates(possible_answer[key]))
                    else:
                        standardize_possible_answer.append(possible_answer[key])
                if isinstance(standardize_possible_answer, list):
                    standardize_possible_answer = standardize_possible_answer[0]
                if str(standardize_possible_answer) not in str(standardize_value):
                    result["valid"] = False
                    result["error_type"] = "value_error"
                    return result
    return result


def list_dict_checker(param, model_output, possible_answers, func_name):
    result = {"valid": True, "error": [], "error_type": "list_dict_checker:unclear"}
    if len(model_output) != len(possible_answers):
        result["valid"] = False
        result["error_type"] = "value_error:list_dict_count"
        return result
    for dict_index in range(len(model_output)):
        if dict_index >= len(possible_answers):
            break
        result = dict_checker(param, model_output[dict_index], possible_answers[dict_index], func_name)
        if not result["valid"]:
            return result
    return result


def simple_function_checker(func_description, model_output, possible_answers, question, test_category):
    result = {"valid": True, "error": [], "error_type": ""}
    possible_answer = list(possible_answers.values())[0]
    if list(model_output.values())[0] == {} and func_description["parameters"] == {}:
        return result
    elif list(model_output.values())[0] == {} or func_description["parameters"] == {}:
        result["valid"] = False
        result["error_type"] = "wrong_param"
        return result
    if possible_answer == func_description["parameters"]["properties"]:
        return result
    elif possible_answer == {} or func_description["parameters"] == {}:
        result["valid"] = False
        result["error_type"] = "wrong_param"
        return result
    func_name = func_description["name"]
    if func_name not in model_output:
        result["valid"] = False
        result["error_type"] = "wrong_function_name"
        return result
    model_params = model_output[func_name]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]
    for param in required_params:
        if param not in model_params:
            return {"valid": False, "error": f"lack required_params: {param}", "error_type": "lack_args"}
    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            return {"valid": False, "error": f"addition params: {param}", "error_type": "addition_args"}
        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]
        is_variable = False
        nested_type_converted = None
        expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            try:
                nested_type = param_details[param]["items"]["type"]
            except Exception:
                if "string" in param_details[param]["type"]:
                    nested_type = "string"
                elif "enum" in param_details[param]["type"]:
                    nested_type = "enum"
                else:
                    nested_type = "dict"
            nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]
        if expected_type_description == "tuple" and type(value) == tuple:
            value = list(value)
        if expected_type_description == "float" and type(value) == int:
            value = float(value)
        type_check_result = type_checker(param, value, possible_answer[param], expected_type_description,
                                         expected_type_converted, nested_type_converted, func_name)
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return {"valid": False, "error": type_check_result["error"],
                    "error_type": type_check_result["error_type"]}
        if not is_variable:
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param], func_name)
                if not result["valid"]:
                    return {"valid": False, "error": result["error"], "error_type": result["error_type"]}
            elif expected_type_converted == list and nested_type_converted == dict:
                if expected_type_description == "objectArray":
                    if len(value) != len(possible_answer[param]):
                        return {"valid": False, "error": ["Wrong number of parameters for dictionary."],
                                "error_type": "value_error:dict_items"}
                    # Upstream calls dict_checker with 3 of its 4 arguments here, which raises on
                    # the first pair (no objectArray parameter exists in the pinned data).
                    if len(value) > 0:
                        raise TypeError("dict_checker() missing 1 required positional argument: 'func_name'")
                result = list_dict_checker(param, value, possible_answer[param], func_name)
                if not result["valid"]:
                    return {"valid": False, "error": result["error"], "error_type": result["error_type"]}
            elif expected_type_converted == str:
                result = string_checker(param, value, possible_answer[param], func_description,
                                        question, test_category)
                if not result["valid"]:
                    return {"valid": False, "error": result["error"], "error_type": result["error_type"]}
            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param], func_name)
                if not result["valid"]:
                    return {"valid": False, "error": result["error"], "error_type": result["error_type"]}
    return result


def normal_checker(func_descriptions, model_output, possible_answers, question, test_category):
    result = {"valid": True}
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["The number of functions does not match the answer."],
                "error_type": "wrong functions number"}
    func_name_list = list(possible_answers.keys())
    possible_answers_list = []
    for key, value in possible_answers.items():
        possible_answers_list.append({key: value})
    for index in range(len(possible_answers_list)):
        current_dict = possible_answers_list[index]
        keys_to_update = list(current_dict.keys())
        for key in keys_to_update:
            new_key = re.sub(r"_\d+$", "", key)
            if new_key != key:
                current_dict[new_key] = current_dict.pop(key)
    output_list = sum_key_list(model_output)
    answer_list = sum_key_list(possible_answers_list)
    for name, count in output_list.items():
        if name not in answer_list:
            return {"valid": False, "error": ["extra function"], "error_type": "function_mismatch"}
    for name, count in answer_list.items():
        if name not in output_list:
            return {"valid": False, "error": ["missing function"], "error_type": "function_mismatch"}
    for name, count in output_list.items():
        if name not in answer_list or count != answer_list[name]:
            return {"valid": False, "error": ["incorrect count"], "error_type": "function_mismatch"}
    for i in range(len(possible_answers_list)):
        func_description = find_description(func_descriptions, func_name_list[i])
        valid_found = False
        for j in range(len(model_output)):
            if list(model_output[j].keys())[0] == list(possible_answers_list[i].keys())[0]:
                result = simple_function_checker(func_description, model_output[j],
                                                 possible_answers_list[i], question, test_category)
                if result["valid"]:
                    valid_found = True
                    break
            else:
                result = {"valid": False, "error": ["wrong_function"],
                          "error_type": "simple_function_checker:unclear"}
        if valid_found:
            continue
        if not result["valid"]:
            return result
    return result


# --------------------------------------------------------------------------- #
# BinOp arguments: upstream resolves them with eval(ast.unparse(node)) (model_inference/utils.py,
# multi_turn/common_agent.py, multi_turn/execution_role.py, multi_step/execution_role_step.py).
# --------------------------------------------------------------------------- #
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
#: Size limits for _literal_binop, so one reply cannot stall the scorer (10**10**10 would never
#: finish, "a" * 10**10 would exhaust memory, and a chain of large integer products grows
#: quadratically). No answer in the data comes near them; a result past them is refused and counted.
_MAX_INT_BITS = 100_000
_MAX_SEQ_LEN = 1_000_000


class _GuardRefusal(ValueError):
    """An expression upstream's eval() would run and gbench does not."""


#: Where the refusals of the conversation or row being scored are collected (see _guard_refusal).
_REFUSALS: "contextvars.ContextVar[Optional[List[str]]]" = contextvars.ContextVar(
    "acebench_guard_refusals", default=None)


def _guard_refusal(source: str, why: str) -> _GuardRefusal:
    """A refusal of `source`, recorded in the current collector (if any)."""
    sink = _REFUSALS.get()
    if sink is not None:
        sink.append(source[:200])
    return _GuardRefusal(f"gbench does not evaluate {source[:200]!r}: {why} (upstream eval()s it)")


def _bounded(op, left, right):
    """_BINOPS[op](left, right), refused (ValueError) when the result would be huge."""
    if op is ast.Pow and isinstance(left, int) and isinstance(right, int) \
            and abs(left) > 1 and right > 0 and abs(left).bit_length() * right > _MAX_INT_BITS:
        raise ValueError("power too large to evaluate")
    if op is ast.Mult:
        if isinstance(left, int) and isinstance(right, int) \
                and abs(left).bit_length() + abs(right).bit_length() > _MAX_INT_BITS:
            raise ValueError("product too large to evaluate")
        for seq, n in ((left, right), (right, left)):
            if isinstance(seq, (str, bytes)) and isinstance(n, int) and len(seq) * n > _MAX_SEQ_LEN:
                raise ValueError("repetition too large to evaluate")
    return _BINOPS[op](left, right)


def _is_literal_arithmetic(node) -> bool:
    """Only number / string / bytes constants, unary +/- and the _BINOPS operators."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, complex, str, bytes)) \
            and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_literal_arithmetic(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _is_literal_arithmetic(node.left) and _is_literal_arithmetic(node.right)
    return False


def _literal_binop(node):
    """eval(ast.unparse(node)) for arithmetic over literals, evaluated here (the same value). An
    integer power, integer product or string repetition whose result would be huge is refused and
    counted (_GuardRefusal, a ValueError) instead of stalling the scorer."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex, str, bytes)):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else +v
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _bounded(type(n.op), ev(n.left), ev(n.right))
        raise ValueError(f"unsupported expression in argument: {ast.unparse(n)}")
    try:
        return ev(node)
    except ValueError as e:
        if "too large" in str(e):
            raise _guard_refusal(ast.unparse(node), f"its result is {e}") from None
        raise


#: The globals upstream's eval sees where each role resolves a BinOp (ACEBench@ACEBENCH_COMMIT):
#: the module's name, the stdlib names it has (given to the isolated eval, "module" or
#: "module:attribute"), and the names only upstream's process has (the module's own functions,
#: classes and constants, and the method's locals), which gbench does not rebuild. The execution
#: roles star-import their utils module (no __all__), so its public names are theirs too.
_EXECUTION_STDLIB = {"importlib": "importlib", "inspect": "inspect", "json": "json", "re": "re",
                     "copy": "copy", "List": "typing:List", "ast": "ast"}
_EXECUTION_ONLY = ("CLASS_FILE_PATH_MAPPING_ZH", "CLASS_FILE_PATH_MAPPING_EN", "STATELESS_CLASSES",
                   "execute_agent_func_call", "is_empty_execute_response", "self", "value")
_BINOP_SCOPES: Dict[str, Dict[str, Any]] = {
    # model_inference/utils.py resolve_ast_by_type(value): the Normal rows' decode_ast
    "normal": {"module": "model_inference.utils", "stdlib": {"ast": "ast", "json": "json"},
               "upstream_only": ("decode_ast", "ast_parse", "resolve_ast_call",
                                 "resolve_ast_by_type", "value")},
    # model_inference/multi_turn/common_agent.py CommonAgent.resolve_ast_by_type(self, value)
    "agent": {"module": "model_inference.multi_turn.common_agent",
              "stdlib": {"re": "re", "ast": "ast"},
              "upstream_only": ("TRAVEL_PROMPT_EN", "BASE_PROMPT_EN", "TRAVEL_PROMPT_ZH",
                                "BASE_PROMPT_ZH", "MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH",
                                "MULTI_TURN_AGENT_PROMPT_USER_ZH",
                                "MULTI_TURN_AGENT_PROMPT_SYSTEM_EN",
                                "MULTI_TURN_AGENT_PROMPT_USER_EN", "CommonAgent", "self",
                                "value")},
    # model_inference/multi_turn/execution_role.py EXECUTION.resolve_ast_by_type(self, value)
    "execution": {"module": "model_inference.multi_turn.execution_role",
                  "stdlib": _EXECUTION_STDLIB, "upstream_only": _EXECUTION_ONLY + ("EXECUTION",)},
    # model_inference/multi_step/execution_role_step.py EXECUTION_STEP.resolve_ast_by_type
    "execution_step": {"module": "model_inference.multi_step.execution_role_step",
                       "stdlib": _EXECUTION_STDLIB,
                       "upstream_only": _EXECUTION_ONLY + ("EXECUTION_STEP",)},
}
#: Module attributes upstream's globals hold that depend on its checkout (its path, loader).
_UPSTREAM_DUNDERS = ("__file__", "__spec__", "__loader__", "__cached__")
#: decode_function_list's three roles (see _agent_ast_parse) -> their scope.
_ROLE_SCOPE = {"fail": "agent", "skip": "execution", "raise": "execution_step"}

#: Limits of one isolated eval: wall time (s), CPU time (s), address space, file size and the
#: container's private /tmp (bytes).
_BINOP_TIMEOUT_S = 30
_BINOP_CPU_S = 10
_BINOP_MEMORY_BYTES = 2 << 30
_BINOP_FILE_BYTES = 16 << 20
_BINOP_TMP_BYTES = 64 << 20

#: The isolated evaluator (`python -I -c`, the request on stdin): upstream's
#: eval(ast.unparse(node)) with that module's globals, the value sent back as its repr, an
#: exception as its type, module and text.
_BINOP_EVALUATOR = r"""
import builtins, importlib, json, resource, sys
req = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_AS, (req["memory"], req["memory"]))
resource.setrlimit(resource.RLIMIT_CPU, (req["cpu"], req["cpu"]))
resource.setrlimit(resource.RLIMIT_FSIZE, (req["fsize"], req["fsize"]))
g = {"__builtins__": builtins.__dict__, "__name__": req["name"], "__package__": req["package"],
     "__doc__": None}
for name, target in req["stdlib"].items():
    module, _, attr = target.partition(":")
    obj = importlib.import_module(module)
    g[name] = getattr(obj, attr) if attr else obj
try:
    value = eval(compile(req["source"], "<string>", "eval"), g)
except BaseException as e:
    try:
        text = str(e)
    except BaseException:
        text = ""
    out = {"raised": type(e).__name__, "module": str(type(e).__module__),
           "builtin": type(e).__module__ == "builtins", "text": text}
else:
    try:
        out = {"repr": repr(value), "type": type(value).__name__}
    except BaseException as e:
        out = {"unrepresentable": type(value).__name__, "text": type(e).__name__}
sys.stdout.write(json.dumps(out))
"""

#: What the container's root holds besides the interpreter's own directories (see
#: _evaluator_argv); anything else there means the isolation is not what it should be.
_CONTAINER_ROOT = frozenset({"usr", "bin", "sbin", "lib", "lib32", "lib64", "libx32", "etc", "dev",
                             "tmp"})


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
    """The directories the evaluator's interpreter runs from that are not under /usr: its prefix
    and base prefix (a virtual environment and the Python it was made from) and the directory of
    its executable."""
    dirs = set()
    for path in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                 os.path.dirname(os.path.realpath(sys.executable))):
        real = os.path.realpath(path)
        for candidate in {path, real}:
            if candidate and candidate != "/" and os.path.isdir(candidate) \
                    and not (candidate + os.sep).startswith("/usr/"):
                dirs.add(candidate)
    return sorted(dirs)


def _evaluator_argv() -> List[str]:
    """The isolated evaluator's command: a bubblewrap container of its own, whatever
    GBENCH_SANDBOX says. New user, PID, IPC, UTS, network and cgroup namespaces
    (`--unshare-all --unshare-user`; `--disable-userns` where bwrap has it), no capabilities, an
    empty environment and a new session, killed with gbench. Its filesystem holds only what the
    interpreter needs: /usr and the interpreter's own directories read-only, the /bin, /sbin and
    /lib* links or directories, the dynamic loader's cache, a private minimal /dev and a small
    private /tmp. No /proc is mounted (the host's process table and environments stay out of
    reach), and /run, /var, /home, the gbench cache and every other host path are absent, so
    neither the Docker daemon's socket nor any credential file is there."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap (bwrap) is not installed")
    options = _bwrap_options()
    cmd = [bwrap, "--unshare-all", "--unshare-user"]
    if "--disable-userns" in options:
        cmd.append("--disable-userns")
    cmd += ["--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv",
            "--ro-bind", "/usr", "/usr"]
    for name in ("bin", "sbin", "lib", "lib32", "lib64", "libx32"):
        path = "/" + name
        if os.path.islink(path):
            cmd += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            cmd += ["--ro-bind", path, path]
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d"):
        if os.path.exists(path):
            cmd += ["--ro-bind", path, path]
    for path in _interpreter_dirs():
        cmd += ["--ro-bind", path, path]
    cmd += ["--dev", "/dev"]
    if "--size" in options:
        cmd += ["--size", str(_BINOP_TMP_BYTES)]
    cmd += ["--tmpfs", "/tmp", "--chdir", "/"]
    return cmd + [sys.executable, "-I", "-c", _BINOP_EVALUATOR]


def _upstream_exception(name: str, builtin: bool, text: str,
                        module: Optional[str] = None) -> BaseException:
    """The exception upstream's eval raised: a builtin type rebuilt with its text; any other (a
    stdlib one such as json.JSONDecodeError, whose constructor needs more than the text) as an
    Exception subclass of the same name and module with the same text. Upstream's callers catch
    Exception, so the verdict is the same, and an error message that names the type reads the
    same."""
    cls = getattr(builtins, name, None) if builtin else None
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            return cls(text)
        except Exception:
            pass
    if not builtin and name.isidentifier():
        return type(name, (Exception,), {"__module__": module or "builtins"})(text)
    return Exception(text)


def _isolated_eval(source: str, scope: str) -> Any:
    """eval(source) as upstream's `scope` module runs it, in the isolated evaluator."""
    spec = _BINOP_SCOPES[scope]
    names = {n.id for n in ast.walk(ast.parse(source, mode="eval")) if isinstance(n, ast.Name)}
    only_upstream = sorted(names & (set(spec["upstream_only"]) | set(_UPSTREAM_DUNDERS)))
    if only_upstream:
        raise _guard_refusal(source, f"it names {', '.join(only_upstream)}, which only upstream's "
                             "process has")
    request = json.dumps({"source": source, "name": spec["module"],
                          "package": spec["module"].rsplit(".", 1)[0], "stdlib": spec["stdlib"],
                          "memory": _BINOP_MEMORY_BYTES, "cpu": _BINOP_CPU_S,
                          "fsize": _BINOP_FILE_BYTES})
    try:
        proc = subprocess.run(_evaluator_argv(), input=request, capture_output=True, text=True,
                              timeout=_BINOP_TIMEOUT_S, env={}, cwd="/")
    except subprocess.TimeoutExpired:
        raise _guard_refusal(source, f"it ran past the isolated evaluator's {_BINOP_TIMEOUT_S} s "
                             "limit") from None
    try:
        out = json.loads(proc.stdout or "")
    except ValueError:
        raise _guard_refusal(source, f"the isolated evaluator stopped (exit {proc.returncode}, "
                             f"past its CPU or memory limit)") from None
    if "repr" in out:
        try:
            return ast.literal_eval(out["repr"])
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            raise _guard_refusal(source, f"its value (a {out.get('type')}) is not a literal, so "
                                 "it cannot leave the isolated evaluator") from None
    if "raised" in out:
        if out["raised"] == "MemoryError":
            raise _guard_refusal(source, "it ran past the isolated evaluator's memory limit")
        if out["raised"] in ("SystemExit", "KeyboardInterrupt", "GeneratorExit"):
            raise _guard_refusal(source, f"it raises {out['raised']}, which would stop "
                                 "upstream's run")
        raise _upstream_exception(out["raised"], bool(out.get("builtin")), out.get("text") or "",
                                  out.get("module"))
    raise _guard_refusal(source, f"its value (a {out.get('unrepresentable')}) has no repr")


def _binop_value(node, scope: str) -> Any:
    """What upstream's eval(ast.unparse(node)) gives in `scope` (see _BINOP_SCOPES)."""
    if _is_literal_arithmetic(node):
        return _literal_binop(node)
    return _isolated_eval(ast.unparse(node), scope)


#: What the container check evaluates (a BinOp, as a model's argument would be): the evaluator's
#: process id, whether /proc, /run and the Docker socket exist, and the entries of its root.
_CONTAINER_PROBE = ("[__import__('os').getpid(), __import__('os').path.exists('/proc'), "
                    "__import__('os').path.exists('/run'), "
                    "__import__('os').path.exists('/var/run/docker.sock'), "
                    "sorted(__import__('os').listdir('/'))] + []")


def container_problem() -> Optional[str]:
    """None when the evaluator's container runs and is isolated as _evaluator_argv says (its own
    PID namespace, no /proc, no /run or Docker socket, nothing on its root but the interpreter's
    directories) and evaluates as upstream's eval does; else why not."""
    if not shutil.which("bwrap"):
        return ("bubblewrap (bwrap) is not installed (`apt-get install bubblewrap`; it must be "
                "allowed to create unprivileged user namespaces)")
    try:
        pid, has_proc, has_run, has_socket, root = _isolated_eval(_CONTAINER_PROBE, "normal")
        value = _isolated_eval("[1] + [2 << 1]", "normal")
    except Exception as e:                  # noqa: BLE001
        return (f"the evaluator's bubblewrap container does not run ({type(e).__name__}: {e}); "
                "bubblewrap must be allowed to create unprivileged user namespaces (on Ubuntu "
                "23.10+: `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`)")
    allowed = set(_CONTAINER_ROOT) | {p.strip("/").split("/")[0] for p in _interpreter_dirs()}
    wrong = []
    if not isinstance(pid, int) or pid > 16:
        wrong.append(f"it is process {pid}, not one of the first of its own PID namespace")
    if has_proc or has_run or has_socket:
        wrong.append("/proc, /run or the Docker socket is visible in it")
    extra = sorted(set(root) - allowed)
    if extra:
        wrong.append(f"its root holds {', '.join(extra)}")
    if value != [1, 4]:
        wrong.append("it returned a wrong value for [1] + [2 << 1]")
    if wrong:
        return "the evaluator's container is not isolated as it must be: " + "; ".join(wrong)
    return None


def check_binop_evaluator() -> None:
    """The isolated evaluator's container must run, isolated, before anything is generated
    (hard error otherwise). GBENCH_SANDBOX does not apply: the evaluator always runs in it."""
    why = container_problem()
    if why:
        raise infra_required(
            "acebench", f"upstream eval()s the BinOp arguments of model calls, and gbench runs "
            f"that eval in a bubblewrap container of its own: {why}", DOCS_URL)


# --------------------------------------------------------------------------- #
# Vendored from model_inference/utils.py (decode_ast for a non-FC model) and eval_main.py.
# --------------------------------------------------------------------------- #
def ast_parse(input_str):
    parsed = ast.parse(input_str, mode="eval")
    extracted = []
    for elem in parsed.body.elts:
        assert isinstance(elem, ast.Call)
        extracted.append(resolve_ast_by_type(elem))
    return extracted


def resolve_ast_call(elem):
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    # The ast.NameConstant / ast.Ellipsis branches upstream are unreachable on Python 3.8+
    # (ast.Constant covers them) and are left out.
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {resolve_ast_by_type(k): resolve_ast_by_type(v) for k, v in zip(value.keys, value.values)}
    elif isinstance(value, ast.BinOp):
        output = _binop_value(value, "normal")
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            func_parts = []
            func_part = value.func
            while isinstance(func_part, ast.Attribute):
                func_parts.append(func_part.attr)
                func_part = func_part.value
            if isinstance(func_part, ast.Name):
                func_parts.append(func_part.id)
            func_name = ".".join(reversed(func_parts))
            output = {func_name: {}}
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        # Upstream: eval(ast.unparse(value.body[0].value)); a Lambda body is not a list, so this
        # always raises.
        raise TypeError("'Lambda' body is not subscriptable")
    elif isinstance(value, ast.Subscript):
        output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def extract_outermost_bracket_content(text):
    start = -1
    depth = 0
    for i, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = i
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start:i + 1]
    return None


def normal_row_valid(result_text: str, functions: Any, possible_answer: Any, question: str,
                     category: str) -> bool:
    """One row of normal_single_turn_eval / normal_multi_turn_eval."""
    try:
        if "multi_turn" in category:
            raw = "".join(result_text.split())
        else:
            raw = extract_outermost_bracket_content(result_text)
        decoded = ast_parse(raw)
    except Exception:
        return False
    if not is_function_call_format_valid(decoded):
        return False
    candidates = possible_answer if type(possible_answer) == list else [possible_answer]
    for candidate in candidates:
        try:
            if normal_checker(functions, decoded, candidate, question, category)["valid"]:
                return True
        except Exception:
            # Upstream does not catch here and the run aborts; the row cannot pass.
            return False
    return False


def special_row_valid(result_text: str, possible_answer: Any, category: str) -> bool:
    """One row of eval_main.special_eval (plain, case-sensitive substring rules)."""
    valid = True
    if "incomplete" in category:
        for name, values in possible_answer.items():
            if "Missing necessary parameters" not in result_text:
                valid = False
            elif name not in result_text:
                valid = False
            else:
                for value in values:
                    if value not in result_text:
                        valid = False
    elif "error" in category:
        for name, values in possible_answer.items():
            if "There is incorrect value" not in result_text:
                valid = False
            else:
                for value in values:
                    if value not in result_text:
                        valid = False
    elif "irrelevant" in category:
        if "the limitations of the function" not in result_text:
            valid = False
    return valid


def multiplt_turn_accuracy(score_list):
    """evaluation_helper.multiplt_turn_accuracy: (end-to-end, process) over conversations."""
    end_score_list = []
    process_score_list = []
    for score in score_list:
        end_score = 0 if False in score["valid"] else 1
        process_score = round(score["valid"].count(True) / len(score["valid"]), 3)
        end_score_list.append(end_score)
        process_score_list.append(process_score)
    return (round(sum(end_score_list) / len(end_score_list), 3),
            round(sum(process_score_list) / len(process_score_list), 3))


def calculate_unweighted_accuracy(accuracies: List[float]) -> float:
    if len(accuracies) == 0:
        return 0
    return round(sum(accuracies) / len(accuracies), 3)


# --------------------------------------------------------------------------- #
# Data, per-sample verdict
# --------------------------------------------------------------------------- #
def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _pinned_file(rel: str, blob: str, language: str = "en") -> List[Dict[str, Any]]:
    """One pinned data_all/data_<language>/ JSON-lines file (downloaded once, checked against its
    git blob id)."""
    path = os.path.join(gbench_cache_dir(), "acebench", ACEBENCH_COMMIT, f"data_{language}", rel)
    data = None
    if os.path.isfile(path):
        with open(path, "rb") as f:
            data = f.read()
        if _git_blob_sha1(data) != blob:
            data = None
    if data is None:
        url = _RAW_URL.format(commit=ACEBENCH_COMMIT, language=language, rel=rel)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except Exception as e:
            raise infra_required("acebench", f"the data file {url} could not be downloaded ({e}).",
                                 DOCS_URL) from e
        if _git_blob_sha1(data) != blob:
            raise infra_required("acebench", f"{url} does not match the pinned file (git blob "
                                 f"{blob}).", DOCS_URL)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _category_rows(sub: str, language: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(row, possible answer) pairs of one category file, in file order."""
    data_blob, answer_blob = PINNED_BLOBS[language][sub]
    data = _pinned_file(f"data_{sub}.json", data_blob, language)
    answers = {a["id"]: a for a in _pinned_file(f"possible_answer/data_{sub}.json", answer_blob,
                                                language)}
    out = []
    for row in data:
        if row["id"] not in answers:
            raise RuntimeError(f"acebench: no possible answer for {language} {row['id']}")
        out.append((row, answers[row["id"]]))
    return out


def _category_key(sub: str, language: str) -> str:
    """The result key of a sub-category: its name for English, `<name>_zh` for Chinese."""
    return sub if language == "en" else f"{sub}_{language}"


def _system_prompt(category: str, row: Dict[str, Any], language: str = "en") -> str:
    """generate.generate_singal + common_inference.single_turn_inference: which template, time
    and profile."""
    normal, preference, special, _user = _PROMPTS[language]
    functions = row["function"]
    if isinstance(functions, (dict, str)):
        functions = [functions]
    if "special" in category:
        return special.format(time=row["time"], function=functions)
    if "preference" in category:
        return preference.format(profile=row["profile"], function=functions)
    return normal.format(time=row["time"], function=functions)


def _load_acebench_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
    language: str = "en",
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """The Normal + Special rows of one language (973 English, 967 Chinese) as single-turn
    samples. The agent rows run as conversations (_load_agent_items)."""
    categories = categories or ["normal", "special"]
    if "agent" in categories:
        raise ValueError("acebench: the agent rows run as conversations; load them with "
                         "_load_agent_items")
    subs = [c for c in NORMAL_CATEGORIES + SPECIAL_CATEGORIES if c.split("_", 1)[0] in categories]
    user_template = _PROMPTS[language][3]
    samples = []
    for sub in subs:
        for row, answer in _category_rows(sub, language):
            messages = [{"role": "system", "content": _system_prompt(sub, row, language)},
                        {"role": "user", "content": user_template.format(question=row["question"])}]
            gold = json.dumps({"id": row["id"], "sub": sub, "language": language,
                               "ground_truth": answer["ground_truth"],
                               "functions": row["function"], "question": row["question"]})
            samples.append((messages, gold, {"category": _category_key(sub, language)}))

    # Stratified on the sub-category, not a contiguous head.
    samples = stratified_sample(samples, limit, lambda s: (s[2] or {}).get("category"),
                                seed="acebench")
    logger.info(f"Loaded {len(samples)} acebench {language} samples ({'+'.join(categories)}).")
    return samples


def _load_agent_items(language: str = "en") -> List[Dict[str, Any]]:
    """The agent_multi_step and agent_multi_turn conversations of one language (20 + 30)."""
    items = []
    for sub in AGENT_CATEGORIES:
        for row, answer in _category_rows(sub, language):
            items.append({"language": language, "category": sub,
                          "key": _category_key(sub, language), "row": row, "answer": answer})
    return items


def _eval_acebench(response_text: str, gold: str) -> bool:
    """Row-level verdict (for a multi-turn row: that turn; conversations are scored in run_acebench)."""
    try:
        g = json.loads(gold)
    except Exception:
        return False
    text = response_text or ""
    if g["sub"].startswith("special"):
        return special_row_valid(text, g["ground_truth"], g["sub"])
    functions = g["functions"]
    if isinstance(functions, (dict, str)):
        functions = [functions]
    return normal_row_valid(text, functions, g["ground_truth"], g["question"], g["sub"])


def _id_key(row_id: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in row_id.split("_")[-2:])


# --------------------------------------------------------------------------- #
# Agent category: the model inference side of model_inference/common_inference.py (multi_turn /
# multi_step), with its agent, execution and user roles, replayed turn by turn.
# --------------------------------------------------------------------------- #
#: generate.py --max-dialog-turns default: dialogue steps per agent conversation.
MAX_DIALOG_TURNS = 40

#: common_inference.py SAVED_CLASS: the attributes of each scenario class the checker compares.
SAVED_CLASS = {
    "BaseApi": ["wifi", "logged_in"],
    "MessageApi": ["inbox"],
    "ReminderApi": ["reminder_list"],
    "FoodPlatform": ["users", "logged_in_users", "orders"],
    "Finance": ["user_accounts", "is_logged_in", "deposit_history", "withdrawal_history",
                "loan_history", "orders", "holdings"],
    "Travel": ["users", "reservations"],
}

#: multi_turn_utils.CLASS_FILE_PATH_MAPPING_{EN,ZH}, relative to _acebench_vendor/scenarios<lang>.
_SCENARIO_MODULES = {
    "BaseApi": "phone_platform.base_api",
    "MessageApi": "phone_platform.message",
    "ReminderApi": "phone_platform.reminder",
    "FoodPlatform": "phone_platform.food_services",
    "Travel": "travel",
}

#: APIModel_user.py get_init_prompt: the first user turn the simulator answers.
_USER_OPENING = {"en": "Is there anything you need help with today?",
                 "zh": "\u4eca\u5929\u6709\u4ec0\u4e48\u9700\u8981\u5e2e\u52a9\u7684\u5417\uff1f"}

#: EXECUTION_STEP.respond's reply to a message it cannot decode as calls.
_STEP_NOT_A_CALL = ("Please do not ask me any questions, use the known conditions to solve the "
                    "problem")

#: What upstream's eval() of a call string can reach besides the scenario instances: the globals
#: of multi_turn_utils.py / multi_step_utils.py and the builtins. gbench never evaluates these.
_UPSTREAM_EVAL_NAMES = frozenset({
    "importlib", "inspect", "json", "re", "copy", "List", "execute_agent_func_call",
    "is_empty_execute_response", "_process_method_calls", "CLASS_FILE_PATH_MAPPING_ZH",
    "CLASS_FILE_PATH_MAPPING_EN", "STATELESS_CLASSES",
}) | frozenset(dir(builtins))


class _RequestFailed(Exception):
    def __init__(self, status: str, error: str):
        super().__init__(error)
        self.status = status


class _UserSimulatorOutage(Exception):
    pass


def _agent_resolve_call(elem, scope: str = "agent"):
    """common_agent.CommonAgent.resolve_ast_call (the same in EXECUTION and EXECUTION_STEP);
    `scope` is the role's module, for a BinOp argument (_BINOP_SCOPES)."""
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = _agent_resolve_value(arg.value, scope)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def _agent_resolve_value(value, scope: str = "agent"):
    """common_agent.CommonAgent.resolve_ast_by_type. A BinOp is upstream's
    eval(ast.unparse(value)) in the role's module (_binop_value). The NameConstant / Ellipsis
    branches are unreachable on Python 3.8+ and left out; a Lambda raises as upstream's
    `value.body[0]` does."""
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [_agent_resolve_value(v, scope) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            _agent_resolve_value(k, scope): _agent_resolve_value(v, scope)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.BinOp):
        output = _binop_value(value, scope)
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = _agent_resolve_call(value, scope)
    elif isinstance(value, ast.Tuple):
        output = tuple(_agent_resolve_value(v, scope) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        raise TypeError(f"'{type(value.body).__name__}' object is not subscriptable")
    elif isinstance(value, ast.Subscript):
        output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def _agent_ast_parse(input_str: str, non_call: str):
    """ast_parse of the three roles, which differ only on a list element that is not a call:
    CommonAgent returns False ("fail"), EXECUTION skips it ("skip"), EXECUTION_STEP raises
    ("raise")."""
    scope = _ROLE_SCOPE[non_call]
    cleaned_input = input_str.strip("[]'")
    parsed = ast.parse(cleaned_input, mode="eval")
    extracted = []
    if isinstance(parsed.body, ast.Call):
        extracted.append(_agent_resolve_call(parsed.body, scope))
    elif isinstance(parsed.body, (ast.Tuple, ast.List)):
        for elem in parsed.body.elts:
            if isinstance(elem, ast.Call):
                extracted.append(_agent_resolve_call(elem, scope))
            elif non_call == "fail":
                return False
            elif non_call == "raise":
                raise ValueError("Element is not a function call")
    return extracted


def _parse_nested_value(value):
    if isinstance(value, dict):
        func_name = list(value.keys())[0]
        args = value[func_name]
        args_str = ", ".join(f"{k}={_parse_nested_value(v)}" for k, v in args.items())
        return f"{func_name}({args_str})"
    return repr(value)


def _decoded_output_to_execution_list(decoded_output):
    execution_list = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(
                f"{k}={_parse_nested_value(v)}" for k, v in value.items()
            )
            execution_list.append(f"{key}({args_str})")
    return execution_list


def decode_function_list(result: str, non_call: str = "fail") -> List[str]:
    """decode_function_list of CommonAgent / EXECUTION / EXECUTION_STEP: the call strings a
    message decodes to (raises when it is not a call list)."""
    func = result
    if " " == func[0]:
        func = func[1:]
    if not func.startswith("["):
        func = "[" + func
    if not func.endswith("]"):
        func = func + "]"
    decoded_output = _agent_ast_parse(func, non_call)
    return _decoded_output_to_execution_list(decoded_output)


async def _decode_off_loop(result: str, non_call: str) -> List[str]:
    """decode_function_list in a worker thread (in this task's context, so its refusals reach the
    conversation's collector): a BinOp argument runs a blocking isolated eval, which must not
    stall the other conversations."""
    return await asyncio.to_thread(decode_function_list, result, non_call)


_METHOD_CALL = re.compile(r"\b([a-zA-Z_]\w*)\s*(?=\()")


def _process_method_calls(function_call_string: str, instance_mapping) -> List[str]:
    """multi_turn_utils._process_method_calls: the first `name(` becomes `<instance>.name(`, once
    per instance that has the method; a name no instance has gives no call."""
    match = _METHOD_CALL.search(function_call_string)
    processed_string_list = []
    if match:
        match_start, match_end = match.span()
        before_match = function_call_string[:match_start]
        after_match = function_call_string[match_end:]
        func_name = match.group(1)
        if func_name in instance_mapping:
            for name in instance_mapping[func_name]:
                processed_string_list.append(before_match + f"{name}.{func_name}" + after_match)
    return processed_string_list


class _Environment:
    """One conversation's scenario instances (execute_agent_func_call keeps them in module globals
    under a per-test-entry name and creates them on the first call)."""

    def __init__(self, initial_config: Dict[str, Any], involved_classes: List[str], language: str,
                 model_name: str = "", test_entry_id: str = ""):
        self.initial_config = initial_config
        self.involved_classes = list(involved_classes)
        self.language = language
        # execute_agent_func_call's instance names (the served model id stands in for
        # generate.py's --model name).
        self.prefix = (f"{model_name.replace('-', '_').replace('.', '_').replace('/', '_')}_"
                       f"{test_entry_id}")
        self._instances: Optional[Dict[str, Any]] = None
        self.namespace: Dict[str, Any] = {}
        self.refusals: List[str] = []

    def instance_name(self, class_name: str) -> str:
        return f"{self.prefix}_{class_name.lower()}_instance"

    def instances(self) -> Dict[str, Any]:
        if self._instances is None:
            self._instances = {}
            for class_name in self.involved_classes:
                module = importlib.import_module(
                    f"gbench.third_party._acebench_vendor.scenarios{self.language}."
                    f"{_SCENARIO_MODULES[class_name]}")
                class_instance = getattr(module, class_name)()
                class_instance._load_scenario(
                    copy.deepcopy(self.initial_config.get(class_name, {})), long_context=False)
                class_instance._load_scenario(
                    copy.deepcopy(self.initial_config.get("BaseApi", {})), long_context=False)
                self._instances[class_name] = class_instance
                self.namespace[self.instance_name(class_name)] = class_instance
        return self._instances

    # ---- the call strings upstream eval()s, dispatched without eval ----
    def _refuse(self, node) -> _GuardRefusal:
        text = ast.unparse(node)
        self.refusals.append(text)
        return _GuardRefusal(f"gbench does not evaluate {text!r} (upstream eval()s it)")

    def _name(self, node: ast.Name):
        if node.id in self.namespace:
            return self.namespace[node.id]
        if node.id in _UPSTREAM_EVAL_NAMES:
            raise self._refuse(node)
        raise NameError(f"name '{node.id}' is not defined")

    def _value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [self._value(v) for v in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._value(v) for v in node.elts)
        if isinstance(node, ast.Set):
            return {self._value(v) for v in node.elts}
        if isinstance(node, ast.Dict) and None not in node.keys:
            return {self._value(k): self._value(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp):
            v = self._value(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.Not):
                return not v
            return ~v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _bounded(type(node.op), self._value(node.left), self._value(node.right))
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Call):
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id not in self.namespace \
                    and root.id not in _UPSTREAM_EVAL_NAMES:
                raise NameError(f"name '{root.id}' is not defined")
            if isinstance(node.func, ast.Name) and node.func.id == "dict":
                return dict(*[self._value(a) for a in node.args],
                            **{kw.arg: self._value(kw.value) for kw in node.keywords})
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in self.namespace):
                return self._call(node)
            raise self._refuse(node)
        raise self._refuse(node)

    def _call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            target = getattr(self._value(func.value), func.attr)
        elif isinstance(func, ast.Name):
            target = self._name(func)
        else:
            raise self._refuse(node)
        args = [self._value(a) for a in node.args]
        kwargs: Dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                kwargs.update(self._value(kw.value))
            else:
                kwargs[kw.arg] = self._value(kw.value)
        return target(*args, **kwargs)

    def call(self, source: str):
        """What upstream's `eval(func_call)` returns for one processed call string."""
        node = ast.parse(source, filename="<string>", mode="eval").body
        if not isinstance(node, ast.Call):
            return self._value(node)
        if isinstance(node.func, ast.Attribute):
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                self._name(root)
        return self._call(node)


def execute_agent_func_call(func_call_list: List[str], env: _Environment, step: bool):
    """multi_turn_utils.execute_agent_func_call (step=False) / multi_step_utils' (step=True).

    Kept statement for statement, quirks included: a method several instances share (turn_on_wifi)
    runs on each and the last result is kept; a call no instance knows re-uses the previous call's
    result, or raises (UnboundLocalError) when there is none; on an error the multi-turn version
    returns the literal text "Error during execution: {str(e)}" (upstream's missing f-prefix) and
    the involved class names, the multi-step version records the message and goes on."""
    instances = env.instances()
    class_method_name_mapping: Dict[str, List[str]] = {}
    for class_name, class_instance in instances.items():
        instance_name = env.instance_name(class_name)
        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            if method_name in class_method_name_mapping:
                class_method_name_mapping[method_name].append(instance_name)
            else:
                class_method_name_mapping[method_name] = [instance_name]

    execution_results = []
    for func_call in func_call_list:
        func_calls = _process_method_calls(func_call, class_method_name_mapping)
        try:
            for func_call in func_calls:
                func_call_result = env.call(func_call)
            execution_results.append(func_call_result)  # noqa: F821 - unbound as upstream
        except Exception as e:
            if step:
                execution_results.append(f"Error during execution: {str(e)}")
                continue
            errors = "Error during execution: {str(e)}"
            return errors, env.involved_classes

    for index in range(len(execution_results)):
        if type(execution_results[index]) == str:
            continue
        elif type(execution_results[index]) == dict:
            try:
                execution_results[index] = json.dumps(execution_results[index])
            except Exception:
                execution_results[index] = str(execution_results[index])
        else:
            execution_results[index] = str(execution_results[index])
    return execution_results, instances


def _parse_execution_results(single_turn_execution_results) -> List[Any]:
    """EXECUTION(_STEP).respond: each result JSON-decoded where it parses (iterating the error
    text of the multi-turn version character by character, as upstream does)."""
    parsed_results = []
    for item in single_turn_execution_results:
        try:
            parsed_results.append(json.loads(item))
        except json.JSONDecodeError:
            parsed_results.append(item)
    return parsed_results


def _remove_prefix(text: str) -> str:
    """APIModel_user.remove_prefix."""
    if text.startswith("user:"):
        return text[5:]
    elif text.startswith("agent:"):
        return text[6:]
    return text


#: APIUSER's request (model_inference/multi_turn/APIModel_user.py __init__ defaults, which
#: common_inference.CommonInference uses: APIUSER(model_name=..., involved_class=..., language=...)):
#: temperature 0.001, top_p 1, max_tokens 1000 on every chat.completions.create. The model is the
#: one replaced component; these are sent to it as they are.
USER_SIM_TEMPERATURE = 0.001
USER_SIM_TOP_P = 1.0
USER_SIM_MAX_TOKENS = 1000
#: What the Gemini simulator is sent as max_output_tokens: APIUSER's 1000 for the reply plus the
#: thinking allowance (gpt-4o does not think; Gemini counts its thinking against the limit).
USER_SIM_OUTPUT_LIMIT = gemini_output_limit(USER_SIM_MAX_TOKENS)


def _user_sim_config(system_instruction: str) -> Any:
    """APIUSER's sampling as a Gemini config (its max_tokens 1000 plus the thinking allowance as
    max_output_tokens, USER_SIM_OUTPUT_LIMIT), with the instructions as the system instruction;
    None without google-genai."""
    try:
        from google.genai import types
    except ImportError:
        return None
    return types.GenerateContentConfig(system_instruction=system_instruction,
                                       temperature=USER_SIM_TEMPERATURE, top_p=USER_SIM_TOP_P,
                                       max_output_tokens=USER_SIM_OUTPUT_LIMIT)


#: The counts of the user simulator answering now (set by _UserSimulator._complete): replies
#: that stopped at max_output_tokens (USER_SIM_OUTPUT_LIMIT) with text ("cut", used as they are, as
#: upstream uses a reply its gpt-4o cut at 1000) and without text ("empty": the thinking took the
#: whole limit before answering, and the cascade asked its next model); "outage_at_cap" counts the
#: simulator outages whose cascade saw an "empty" reply.
_USER_SIM_CUTS: "contextvars.ContextVar[Optional[Dict[str, int]]]" = contextvars.ContextVar(
    "acebench_user_sim_cuts", default=None)


def _stopped_at_max_tokens(resp: Any) -> bool:
    """Did a Gemini reply stop at its output-token limit (finish reason MAX_TOKENS)?"""
    for cand in getattr(resp, "candidates", None) or []:
        reason = getattr(cand, "finish_reason", None)
        if reason is not None and str(getattr(reason, "name", reason)).endswith("MAX_TOKENS"):
            return True
    return False


async def _user_sim_cascade(contents: Any, config: Any = None) -> Tuple[Optional[str], str]:
    """One request on the gbench judge cascade: base.judge_generate_cascade's models, key
    rotation, rounds and backoff (the same loop, as toolbench's _tooleval_cascade), also counting
    in _USER_SIM_CUTS the replies that stopped at max_output_tokens. `(text, model)` from the
    first model that answers with text, or `(None, "JUDGE_OUTAGE")` when every model fails every
    round."""
    if not gemini_api_keys():
        return None, "JUDGE_OUTAGE"
    try:
        if not base._judge_clients():
            return None, "JUDGE_OUTAGE"
    except ImportError:
        return None, "JUDGE_OUTAGE"
    cfg = config if config is not None else base.judge_config()
    cuts = _USER_SIM_CUTS.get()
    rounds = base._judge_cascade_rounds()
    cascade = judge_cascade()
    not_served: set = set()
    for rnd in range(rounds):
        client = base._next_judge_client()
        for model in cascade:
            if model in not_served:
                continue
            try:
                resp = await client.aio.models.generate_content(
                    model=model, contents=contents, config=cfg)
                text = getattr(resp, "text", None)
                at_limit = _stopped_at_max_tokens(resp)
                if text is not None:
                    if at_limit and cuts is not None:
                        cuts["cut"] += 1
                    base.record_gemini_answer(model)
                    return text, model
                if at_limit and cuts is not None:
                    cuts["empty"] += 1
            except Exception as e:  # noqa: BLE001 - 429/network/timeout: move to next model
                if getattr(e, "code", None) == 404:
                    # a model the API does not serve never answers: not asked again in this call,
                    # as base.judge_generate_cascade does
                    not_served.add(model)
                logger.debug("acebench user simulator model %s failed: %s", model, e)
        if len(not_served) >= len(cascade):
            break
        if rnd < rounds - 1:
            await asyncio.sleep(base._JUDGE_BACKOFF * (2 ** rnd)
                                + random.uniform(0, base._JUDGE_BACKOFF))
    return None, "JUDGE_OUTAGE"


async def _user_sim_generate(messages: List[Dict[str, str]]) -> Tuple[Optional[str], str]:
    """One user-simulator reply from gbench's Gemini judge cascade, for APIUSER's chat.

    APIUSER sends [system: instructions, user: opening, system: its reply, user: agent, ...]
    to its OpenAI model. Gemini takes the instructions as system_instruction and no system turns
    inside a conversation, so the simulator's own earlier replies (upstream's mid-conversation
    system messages) go out as its model turns and the agent's messages as user turns.
    Sampling: APIUSER's own (temperature 0.001, top_p 1, max_tokens 1000 plus the Gemini thinking
    allowance; _user_sim_config)."""
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in messages[1:]]
    cfg = _user_sim_config(messages[0]["content"])
    return await _user_sim_cascade(contents, config=cfg)


class _UserSimulator:
    """multi_turn/APIModel_user.py APIUSER, answering through _user_sim_generate."""

    def __init__(self, involved_class: List[str], language: str):
        self.involved_class = involved_class
        self.language = language
        self.messages: List[Dict[str, str]] = []
        self.models: List[str] = []
        # replies that stopped at max_output_tokens, with and without text (_USER_SIM_CUTS)
        self.cuts: Dict[str, int] = {"cut": 0, "empty": 0, "outage_at_cap": 0}

    def _system_prompt(self) -> str:
        suffix = self.language.upper()
        if "BaseApi" in self.involved_class:
            return getattr(_MT, f"SYSTEM_PROMPT_BASE_{suffix}")
        if "Travel" in self.involved_class:
            return getattr(_MT, f"SYSTEM_PROMPT_TRAVEL_{suffix}")
        raise UnboundLocalError("local variable 'system_prompt' referenced before assignment")

    async def _complete(self) -> str:
        token = _USER_SIM_CUTS.set(self.cuts)
        empty_before = self.cuts["empty"]
        try:
            text, model = await _user_sim_generate(self.messages)
        finally:
            _USER_SIM_CUTS.reset(token)
        if text is None:
            if self.cuts["empty"] > empty_before:
                # the outage's cascade saw replies stop at the limit with no text
                self.cuts["outage_at_cap"] += 1
            raise _UserSimulatorOutage(model)
        self.models.append(model)
        return text

    async def get_init_prompt(self, question: str) -> str:
        self.messages = [{"role": "system",
                          "content": self._system_prompt().format(instruction=question)},
                         {"role": "user", "content": _USER_OPENING[self.language]}]
        response = await self._complete()
        self.messages.append({"role": "system", "content": response})
        return response

    def step(self, message: str) -> None:
        self.messages.append({"role": "user", "content": _remove_prefix(message)})

    async def respond(self) -> Dict[str, Any]:
        response = await self._complete()
        self.messages.append({"role": "system", "content": response})
        return {"sender": "user", "recipient": "agent", "message": response}


class _AgentModel:
    """The model under test as LLMInfer.inference calls it: [system, user] per step, at the
    suite's sampling, through base._send_single_request (one per conversation)."""

    def __init__(self, shared: Dict[str, Any]):
        self.shared = shared
        self.requests = 0
        self.truncated = 0
        self.at_official_cap = 0
        self.completion_tokens = 0
        # no-think requests LLMInfer's window left less than the 1024 cap, steps whose prompt
        # was longer than the window (not sent; the reply is ""), and requests whose prompt
        # could come near the window but could not be counted exactly (sent the 1024)
        self.window_limited = 0
        self.window_over = 0
        self.window_uncounted = 0

    async def inference(self, system_prompt: str, user_prompt: str) -> str:
        s = self.shared
        convo = [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}]
        limit = s["budget"]
        if s.get("upstream_window"):
            # LLMInfer's 8192-token window (a no-think run), as vLLM 0.6.1 applies it.
            limit, _tokens, how = await asyncio.to_thread(
                upstream_window_budget, convo, s["budget"], s["ctk"])
            self.window_uncounted += int(how == "uncounted")
            if limit == 0:
                # Upstream's vLLM ignores the prompt and returns an empty output: LLMInfer
                # returns "" and the conversation goes on with it (UPSTREAM_MAX_MODEL_LEN).
                self.window_over += 1
                return ""
            self.window_limited += int(limit < s["budget"])
        sent_max, oversize = await clamp_to_context_async(convo, limit, None, s["ctk"])
        if oversize:
            raise _RequestFailed("OVER_CONTEXT", oversize)
        reply = await base._send_single_request(
            session=s["session"], api_url=s["api_url"], model_name=s["model_name"],
            messages=convo, extra_payload=s["payload"], semaphore=s["semaphore"], pbar=None,
            max_output_tokens=sent_max, temperature=s["temperature"], thinking=s["thinking"])
        self.requests += 1
        self.completion_tokens += int(reply.completion_tokens or 0)
        if reply.text is None:
            err = reply.error or "request failed"
            status = ("OVER_CONTEXT" if OVER_CONTEXT_ERROR in err
                      else "TIMEOUT" if err.startswith("timeout") else "FAILED")
            raise _RequestFailed(status, err)
        if reply.finish_reason == "length":
            # A reply that stops at the official cap it was sent (1024, or the room LLMInfer's
            # window leaves) is the protocol's outcome, as upstream's LLMInfer run stops it there,
            # not a truncation of this run's budget. Any other stop at the budget (the run's
            # budget with --thinking, or a served window smaller than upstream's) is one; an empty
            # reply after reasoning included.
            if s.get("official_cap") is not None and sent_max == limit:
                self.at_official_cap += 1
            else:
                self.truncated += 1
        return strip_thinking_tags(reply.text)


def _multi_turn_agent_prompts(functions, involved_class, language: str, history: str):
    """common_agent.CommonAgent.respond's prompts."""
    suffix = language.upper()
    system_prompt = getattr(_MT, f"MULTI_TURN_AGENT_PROMPT_SYSTEM_{suffix}")
    user_prompt = getattr(_MT, f"MULTI_TURN_AGENT_PROMPT_USER_{suffix}").format(
        functions=functions, history=history)
    prompts = _PROMPT_EN if language == "en" else _PROMPT_ZH
    if "Travel" in involved_class:
        system_prompt += getattr(prompts, f"TRAVEL_PROMPT_{suffix}")
    if "BaseApi" in involved_class:
        system_prompt += getattr(prompts, f"BASE_PROMPT_{suffix}")
    return system_prompt, user_prompt


def _multi_step_agent_prompts(functions, language: str, history: str):
    """common_agent_step.CommonAgent_Step.respond's prompts."""
    suffix = language.upper()
    return (getattr(_MS, f"MULTI_TURN_AGENT_PROMPT_SYSTEM_{suffix}"),
            getattr(_MS, f"MULTI_TURN_AGENT_PROMPT_USER_{suffix}").format(
                functions=functions, history=history))


def _saved_state(result_instance_list: List[Any]) -> List[Dict[str, Any]]:
    """common_inference: the SAVED_CLASS attributes of every recorded instance set."""
    result_list = []
    for result_instance in result_instance_list:
        for name, instance in result_instance.items():
            item_dict = {}
            for item in instance.__dict__:
                if item in SAVED_CLASS[name]:
                    item_dict[item] = instance.__dict__[item]
            result_list.append({name: item_dict})
    return result_list


async def _multi_turn_inference(model: _AgentModel, row: Dict[str, Any], language: str,
                                record: Dict[str, Any]):
    """CommonInference.multi_turn_inference (agent, user simulator, execution)."""
    functions = row["function"]
    if isinstance(functions, (dict, str)):
        functions = [functions]
    initial_config, involved_classes = row["initial_config"], row["involved_classes"]
    env = _Environment(initial_config, involved_classes, language, model.shared["model_name"],
                       row["id"].split("_")[-1])
    user = _UserSimulator(involved_classes, language)
    record["env"], record["user"] = env, user
    init_message = await user.get_init_prompt(row["question"])
    message_history = [{"sender": "user", "recipient": "agent", "message": init_message}]
    record["history"] = message_history
    inference_data = ""
    result_instance_list: List[Any] = []
    mile_stone: List[Any] = []
    record["mile_stone"] = mile_stone
    for index in range(MAX_DIALOG_TURNS):
        last = message_history[-1]
        # Scene.get_inference_message, called before every role's turn
        if last["sender"] == "user":
            inference_data += "user:" + str(last["message"]) + "\n"
        elif last["sender"] == "agent":
            inference_data += "agent:" + last["message"] + "\n"
        elif last["sender"] == "execution":
            inference_data += "execution:" + str(last["message"]) + "\n"
        if last["recipient"] == "user":
            user.step(last["message"])
            current_message = await user.respond()
        elif last["recipient"] == "agent":
            response = await model.inference(*_multi_turn_agent_prompts(
                functions, involved_classes, language, inference_data))
            current_message = {"sender": "agent"}
            if re.search(r"\[.*\]", response):
                try:
                    await _decode_off_loop(response, "fail")
                    current_message["recipient"] = "execution"
                except Exception:
                    current_message["recipient"] = "user"
            else:
                current_message["recipient"] = "user"
            current_message["message"] = response
        else:
            mile_stone.append(last["message"])
            function_call_list = await _decode_off_loop(last["message"], "skip")
            results, result_instance = execute_agent_func_call(function_call_list, env, step=False)
            current_message = {"sender": "execution", "recipient": "agent",
                               "message": _parse_execution_results(results)}
            if isinstance(result_instance, dict):
                if result_instance not in result_instance_list:
                    result_instance_list.append(result_instance)
        message_history.append(current_message)
        if index > 1 and "finish conversation" in current_message["message"]:
            break
    return _saved_state(result_instance_list), mile_stone


async def _multi_step_inference(model: _AgentModel, row: Dict[str, Any], language: str,
                                record: Dict[str, Any]):
    """CommonInference.multi_step_inference (agent and execution; no user simulator)."""
    functions = row["function"]
    if isinstance(functions, (dict, str)):
        functions = [functions]
    env = _Environment(row["initial_config"], row["involved_classes"], language,
                       model.shared["model_name"], row["id"].split("_")[-1])
    record["env"] = env
    message_history = [{"sender": "user", "recipient": "agent", "message": row["question"]}]
    record["history"] = message_history
    inference_data = ""
    result_instance_list: List[Any] = []
    mile_stone: List[Any] = []
    record["mile_stone"] = mile_stone
    for index in range(MAX_DIALOG_TURNS):
        last = message_history[-1]
        if index == 0 or last["sender"] == "execution":
            # Mulit_Step_Scene.get_inference_message, called only before the agent's turn
            if last["sender"] == "user":
                inference_data += "user:" + str(last["message"]) + "\n"
            elif last["sender"] == "agent":
                inference_data += "agent:" + last["message"] + "\n"
            elif last["sender"] == "execution":
                inference_data += "execution result:" + str(last["message"]) + "\n"
            response = await model.inference(*_multi_step_agent_prompts(
                functions, language, inference_data))
            current_message = {"sender": "agent",
                               "recipient": ("execution" if re.match(r"\[.*?\]", response)
                                             else "user"),
                               "message": response}
        else:
            current_message = {"sender": "execution", "recipient": "agent"}
            try:
                function_call_list = await _decode_off_loop(last["message"], "raise")
            except Exception:
                current_message["message"] = _STEP_NOT_A_CALL
                result_instance: Any = {}
            else:
                results, result_instance = execute_agent_func_call(function_call_list, env,
                                                                   step=True)
                current_message["message"] = _parse_execution_results(results)
            mile_stone.append(last["message"])
            if result_instance not in result_instance_list:
                result_instance_list.append(result_instance)
        message_history.append(current_message)
        if index > 1 and "finish conversation" in current_message["message"]:
            break
    return _saved_state(result_instance_list), mile_stone


# ---- eval_main.agent_eval / agent_eval_process and checker.agent_checker ----
def agent_checker(model_output: Dict[str, Any], possible_answer: Dict[str, Any]) -> Dict[str, Any]:
    """model_eval/checker.py agent_checker (error texts shortened; they never affect a verdict)."""
    result = {"valid": True, "error": [], "error_type": "class attributes wrong"}
    scenario_name = list(possible_answer.keys())[0]
    possible_answer = list(possible_answer.values())[0]
    model_params = list(model_output.values())[0]
    for model_param, model_value in model_params.items():
        if model_param in possible_answer:
            possible_answer_value = possible_answer[model_param]
        else:
            result["valid"] = False
            result["error"].append(f"class({scenario_name}) attributes({model_param}) missing")
            continue
        if isinstance(possible_answer_value, dict):
            for param, value in possible_answer_value.items():
                if param not in model_value:
                    result["valid"] = False
                    result["error"].append(f"class({scenario_name}) attributes({model_param}) wrong")
                elif value != model_value[param]:
                    result["valid"] = False
                    result["error"].append(
                        f"class({scenario_name}) attributes({model_param}.{param}) wrong")
        else:
            if possible_answer_value != model_value:
                result["valid"] = False
                result["error"].append(f"class({scenario_name}) attributes({model_param}) wrong")
    return result


def agent_entry_valid(model_result_item: List[Any], possible_answer_item: Any) -> Tuple[bool, str]:
    """One entry of eval_main.agent_eval: (valid, error_type). Quirk kept: a possible-answer class
    with no class of the same attributes in the result reuses the previous class's verdict."""
    possible_answer_item_ = possible_answer_item
    if type(possible_answer_item_) != list:
        possible_answer_item_ = [possible_answer_item_]
    checker_result = {"valid": True}
    if len(possible_answer_item_) != len(model_result_item):
        return False, "wrong number of class"
    is_valid, error_type = True, ""
    for index in range(len(possible_answer_item_)):
        possible_keys = set(possible_answer_item_[index].keys())
        matched_dict = None
        for model_dict in model_result_item:
            if possible_keys == set(model_dict.keys()):
                matched_dict = model_dict
                break
        if matched_dict:
            checker_result = agent_checker(matched_dict, possible_answer_item_[index])
        if checker_result["valid"] == False:  # noqa: E712 - as upstream
            is_valid = False
            error_type = checker_result["error_type"]
    return is_valid, error_type


def agent_process_score(model_process: List[Any], call_process: List[Any]) -> float:
    """One wrong entry of eval_main.agent_eval_process: the share of milestones found in order in
    the agent's calls (several milestone lists: the best, each rounded to 3 decimals; one list:
    unrounded). A right entry scores 1.0 there without this."""
    def matched(stones: List[str]) -> float:
        result_len, milestone_len = len(model_process), len(stones)
        result_indices, current_index = [], 0
        for stone in stones:
            while current_index < result_len:
                if model_process[current_index].strip() == stone.strip():
                    result_indices.append(current_index)
                    current_index += 1
                    break
                current_index += 1
        return 1.00 if milestone_len == 0 else len(result_indices) / milestone_len

    if isinstance(call_process[0], list):
        return max(round(matched(item), 3) for item in call_process)
    return matched(call_process)


def _json_roundtrip(value: Any) -> Any:
    """What the result file holds: base_inference.write_result json.dumps(ensure_ascii=False),
    eval_main load_file json.loads."""
    return json.loads(json.dumps(value, ensure_ascii=False))


async def _run_conversation(model_shared: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """One agent conversation, scored: a trace for the result."""
    row, answer, language = item["row"], item["answer"], item["language"]
    model = _AgentModel(model_shared)
    record: Dict[str, Any] = {}
    trace: Dict[str, Any] = {
        "category": item["key"], "acebench_language": language, "question_id": row["id"],
        "attempt_idx": item.get("attempt_idx", 0),
        "gold_answer": json.dumps({"id": row["id"], "sub": item["category"], "language": language,
                                   "ground_truth": answer["ground_truth"],
                                   "mile_stone": answer["mile_stone"]}, ensure_ascii=False),
    }
    status, error = "OK", None
    result_list: Any = None
    process: List[Any] = []
    # BinOp arguments gbench refuses while decoding this conversation's messages (this task only).
    binop_refusals: List[str] = []
    token = _REFUSALS.set(binop_refusals)
    try:
        if item["category"] == "agent_multi_turn":
            result_list, process = await _multi_turn_inference(model, row, language, record)
        else:
            result_list, process = await _multi_step_inference(model, row, language, record)
        result_list, process = _json_roundtrip(result_list), _json_roundtrip(process)
    except _RequestFailed as e:
        status, error = e.status, str(e)
    except _UserSimulatorOutage as e:
        status, error = "USER_SIM_OUTAGE", f"the user simulator got no reply ({e})"
    except Exception as e:                      # noqa: BLE001 - upstream would abort the run here
        status, error = "ERROR", f"{type(e).__name__}: {e}"
    finally:
        _REFUSALS.reset(token)
    history = record.get("history") or []
    agent_msgs = [m["message"] for m in history if m.get("sender") == "agent"]
    env, user = record.get("env"), record.get("user")
    trace.update({
        "status": status,
        "error": error,
        "agent_dialogue": json.loads(json.dumps(
            [{k: m[k] for k in ("sender", "recipient", "message")} for m in history],
            ensure_ascii=False, default=str)),
        "agent_result": result_list,
        "agent_process": process or list(record.get("mile_stone") or []),
        "model_requests": model.requests,
        "truncated_replies": model.truncated,
        "replies_at_official_max_tokens": model.at_official_cap,
        "completion_tokens": model.completion_tokens,
        "user_sim_requests": len(user.models) if user else 0,
        "user_sim_models": dict(Counter(user.models)) if user else {},
        "user_sim_replies_cut_at_max_tokens": user.cuts["cut"] if user else 0,
        "user_sim_empty_at_max_tokens": user.cuts["empty"] if user else 0,
        "user_sim_outages_after_max_tokens": user.cuts["outage_at_cap"] if user else 0,
        "window_limited_requests": model.window_limited,
        # agent steps whose prompt was longer than LLMInfer's window: not sent, reply ""
        "window_over_steps": model.window_over,
        "window_uncounted_requests": model.window_uncounted,
        "execution_guard_refusals": (list(env.refusals) if env else []) + binop_refusals,
        "dialogue_steps": max(0, len(history) - 1),
    })
    if status == "OK":
        valid, error_type = agent_entry_valid(result_list, answer["ground_truth"])
        trace["is_correct"] = valid
        trace["agent_error_type"] = error_type or None
        trace["process_accuracy"] = (1.0 if valid else
                                     agent_process_score(process, answer["mile_stone"]))
        trace["response_text"] = agent_msgs[-1] if agent_msgs else ""
    else:
        trace["is_correct"] = False
        trace["process_accuracy"] = None
        # Failed requests carry no response (scorable_traces skips them); a user-simulator
        # outage is excluded like a judge outage; an upstream-crash case scores wrong.
        trace["response_text"] = (agent_msgs[-1] if agent_msgs else "") if status == "ERROR" \
            else None
        if status == "USER_SIM_OUTAGE":
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["response_text"] = agent_msgs[-1] if agent_msgs else ""
    return trace


async def _run_agent_items(items: List[Dict[str, Any]], model_name: str, base_url: str,
                           concurrency: int, thinking: bool, temperature: Optional[float],
                           payload: Dict[str, Any], max_tokens: int,
                           official_cap: Optional[int] = None,
                           upstream_window: bool = False) -> List[Dict[str, Any]]:
    """Every agent conversation, `concurrency` at a time; each request asks for `max_tokens`
    (clamped to the context window). `official_cap` is that same number when it is the official
    cap (a no-think run), so a reply stopping there is counted as the protocol's outcome.
    `upstream_window` applies LLMInfer's 8192-token window to each request as well
    (upstream_window_budget; a no-think run)."""
    import aiohttp
    ctk = effective_chat_template_kwargs(payload.get("chat_template_kwargs"), thinking)
    conversations = asyncio.Semaphore(max(1, concurrency))
    connector = aiohttp.TCPConnector(limit=max(1, concurrency) + 20)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s(max_tokens))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        shared = {"session": session, "api_url": f"{base_url.rstrip('/')}/chat/completions",
                  "model_name": model_name, "payload": payload, "ctk": ctk,
                  "semaphore": asyncio.Semaphore(max(1, concurrency)), "budget": max_tokens,
                  "official_cap": official_cap, "temperature": temperature, "thinking": thinking,
                  "upstream_window": upstream_window}

        async def one(item):
            async with conversations:
                return await _run_conversation(shared, item)
        return list(await asyncio.gather(*[one(it) for it in items]))


#: What agent_multi_turn's user simulator needs in gbench.
USER_SIM_PREREQUISITE = ("GEMINI_API_KEY (or GEMINI_API_KEYS): the gbench Gemini judge cascade "
                         "that plays agent_multi_turn's user simulator")


def user_simulator_problem() -> Optional[str]:
    """Why agent_multi_turn's user simulator (gbench's Gemini judge cascade) cannot run, or None
    when it can: no key, or every key rejected by a live check."""
    keys = gemini_api_keys()
    if not keys:
        return ("agent_multi_turn needs its user simulator: upstream's gpt-4o "
                "(model_inference/multi_turn/APIModel_user.py, generate.py --user-model), replaced "
                "by gbench's Gemini judge cascade, which needs GEMINI_API_KEY (or GEMINI_API_KEYS)")
    ok, why = gemini_key_live_valid(",".join(keys))
    if not ok:
        if str(why).startswith(base.MODELS_NOT_SERVED_PREFIX):
            return (f"{why}; every model of the Gemini cascade that answers as "
                    "agent_multi_turn's user simulator must be one the API serves")
        return (f"the Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) that answers as "
                f"agent_multi_turn's user simulator was rejected ({why})")
    return None


class UserSimulatorUnavailable(RuntimeError):
    """agent_multi_turn's user simulator cannot run (user_simulator_problem). Only that category
    needs it: run_acebench reports it infra_required in its own entry
    (agent_multi_turn_infra_required) and runs the others, and raises infra_required only when
    agent_multi_turn is all the run selected."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"[INFRA REQUIRED] acebench: agent_multi_turn cannot run: {reason}. "
                         f"See '{DOCS_URL}'.")


def check_user_simulator() -> None:
    """Raise UserSimulatorUnavailable when agent_multi_turn's user simulator cannot run."""
    problem = user_simulator_problem()
    if problem:
        raise UserSimulatorUnavailable(problem)


def agent_multi_turn_infra_required(reason: str, held_out: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The result entry of agent_multi_turn when its user simulator cannot run while the other
    categories run: not run, not scored, the missing prerequisite named. `held_out` are the
    selected conversations that were not run."""
    sentence = str(reason).rstrip()
    sentence += "" if sentence.endswith(".") else "."
    by_language = Counter(it["language"] for it in held_out)
    return {"status": "infra_required", "category": "agent_multi_turn",
            "languages": [lang for lang in LANGUAGES if lang in by_language],
            "conversations_not_run": {lang: by_language[lang] for lang in LANGUAGES
                                      if lang in by_language},
            "prerequisite": USER_SIM_PREREQUISITE,
            "infra_required": str(reason),
            "error": (f"[INFRA REQUIRED] acebench: agent_multi_turn was not run and not scored: "
                      f"{sentence} See '{DOCS_URL}'."),
            "leaderboard_comparable": False,
            "leaderboard_comparable_reason": ("agent_multi_turn was not run (missing "
                                              "prerequisite, infra_required)"),
            "total_questions": 0, "correct_answers": 0, "accuracy": None}


def _max_tokens(thinking: bool) -> "base.TokenBudget":
    """The per-request budget, by the token-limit policy run_eval_suite applies to the
    single-turn rows (base.resolve_token_budget over ACEBENCH_SAMPLING): LLMInfer's
    max_tokens=1024 on a no-think run (each request within LLMInfer's window,
    upstream_window_budget); on a --thinking run that cap is not applied and each request gets
    the run's --max-output-tokens (else the thinking floor)."""
    return base.resolve_token_budget("acebench", ACEBENCH_SAMPLING, thinking=thinking)


# --------------------------------------------------------------------------- #
# Aggregation (evaluation_helper.generate_result_csv, one language)
# --------------------------------------------------------------------------- #
def aggregate(rows: List[Tuple[str, str, bool, int]],
              agent: Optional[Dict[str, List[Tuple[bool, float]]]] = None,
              unavailable: Optional[List[str]] = None) -> Dict[str, Any]:
    """Per-sub-category accuracy plus the leaderboard aggregates of one language.

    `rows` are the single-turn (sub, id, valid, attempt) verdicts. Multi-turn rows are grouped by
    conversation (id `..._<conv>_<turn>`) as normal_multi_turn_eval does, within one attempt; a
    conversation counts only if every turn is valid. `agent` maps agent_multi_step /
    agent_multi_turn to their (valid, process score) entries (agent_eval). Rounding follows
    upstream; a sub-category that did not run counts 0, as generate_result_csv reads it.

    `unavailable` are sub-categories reported infra_required (never scored): each gets an entry
    with status infra_required and a null accuracy, and every aggregate that weighs one (its
    summary, the Summary) is null rather than computed with a 0 in its place."""
    by_sub: Dict[str, List[Tuple[str, bool, int]]] = {}
    for sub, row_id, valid, attempt in rows:
        by_sub.setdefault(sub, []).append((row_id, valid, attempt))
    per_sub: Dict[str, Dict[str, Any]] = {}
    for sub, items in by_sub.items():
        if "multi_turn" in sub:
            items = sorted(items, key=lambda it: (it[2],) + _id_key(it[0]))
            score_list: List[Dict[str, Any]] = []
            for row_id, valid, attempt in items:
                conv = (attempt, row_id.split("_")[-2])
                if score_list and score_list[-1]["turn"] == conv:
                    score_list[-1]["valid"].append(valid)
                else:
                    score_list.append({"turn": conv, "valid": [valid]})
            end, process = multiplt_turn_accuracy(score_list)
            per_sub[sub] = {"accuracy": end, "process_accuracy": process,
                            "conversations": len(score_list), "rows": len(items),
                            "correct": sum(1 for sc in score_list if False not in sc["valid"]),
                            "total": len(score_list)}
        elif sub.startswith("special"):
            n_ok = sum(1 for it in items if it[1])
            per_sub[sub] = {"accuracy": n_ok / len(items), "rows": len(items),
                            "correct": n_ok, "total": len(items)}
        else:
            n_ok = sum(1 for it in items if it[1])
            per_sub[sub] = {"accuracy": round(n_ok / len(items), 3), "rows": len(items),
                            "correct": n_ok, "total": len(items)}
    for sub, entries in (agent or {}).items():
        if not entries:
            continue
        n_ok = sum(1 for valid, _p in entries if valid)
        per_sub[sub] = {"accuracy": round(n_ok / len(entries), 3),
                        "process_accuracy": round(sum(p for _v, p in entries) / len(entries), 3),
                        "conversations": len(entries), "correct": n_ok, "total": len(entries)}

    missing = [s for s in (unavailable or []) if s not in per_sub]
    for sub in missing:
        per_sub[sub] = {"status": "infra_required", "accuracy": None, "correct": 0, "total": 0,
                        **({"process_accuracy": None, "conversations": 0}
                           if sub.startswith("agent") or "multi_turn" in sub else {"rows": 0})}

    def acc(sub: str) -> float:
        return per_sub.get(sub, {}).get("accuracy", 0)

    def mean(subs: List[str]) -> Optional[float]:
        if any(s in missing for s in subs):
            return None
        return calculate_unweighted_accuracy([acc(s) for s in subs])

    normal = mean(NORMAL_CATEGORIES)
    special = mean(SPECIAL_CATEGORIES)
    agent_summary = mean(AGENT_CATEGORIES)
    summary = (None if None in (special, normal, agent_summary) else
               round(special * SUMMARY_WEIGHTS["special"] + normal * SUMMARY_WEIGHTS["normal"]
                     + agent_summary * SUMMARY_WEIGHTS["agent"], 3))
    atom = mean([s for s in NORMAL_CATEGORIES if "_atom_" in s])
    # The leaderboard page's columns (docs/script.js: atom, single, multi, api, pref, summary,
    # special, agent, overall); generate_result_csv computes the same values.
    columns = {"atom": atom,
               "single": mean(["normal_single_turn_single_function",
                               "normal_single_turn_parallel_function"]),
               "multi": mean(["normal_multi_turn_user_switch", "normal_multi_turn_user_adjust"]),
               "api": None if "normal_similar_api" in missing else acc("normal_similar_api"),
               "pref": None if "normal_preference" in missing else acc("normal_preference"),
               "summary": normal, "special": special, "agent": agent_summary, "overall": summary}
    return {"per_sub_category": per_sub, "normal_summary": normal, "special_summary": special,
            "agent_summary": agent_summary, "summary": summary, "leaderboard_columns": columns}


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value * 100.0, 2)


def draw_ceiling(per_sub: Dict[str, Any]) -> Tuple[List[str], Optional[float]]:
    """(the sub-categories of one language this run did not run, the Summary a model right on
    every item the run did draw would get). generate_result_csv reads a sub-category with no
    result as accuracy 0, so a subset that misses sub-categories caps the Summary below 1.0; the
    ceiling says how far. None when a sub-category was reported infra_required (the Summary is
    null then)."""
    subs = NORMAL_CATEGORIES + SPECIAL_CATEGORIES + AGENT_CATEGORIES
    not_run = [sub for sub in subs if sub not in per_sub]
    if any((per_sub.get(sub) or {}).get("status") == "infra_required" for sub in subs):
        return not_run, None

    def mean(group: List[str]) -> float:
        return calculate_unweighted_accuracy([1.0 if sub in per_sub else 0.0 for sub in group])
    return not_run, round(mean(SPECIAL_CATEGORIES) * SUMMARY_WEIGHTS["special"]
                          + mean(NORMAL_CATEGORIES) * SUMMARY_WEIGHTS["normal"]
                          + mean(AGENT_CATEGORIES) * SUMMARY_WEIGHTS["agent"], 3)


def _part_quotas(limit: Optional[int], single_counts: Dict[str, int],
                 agent_counts: Dict[str, int]) -> Tuple[Optional[int], Optional[int]]:
    """--eval-limit N over the single-turn rows and the agent conversations, N in total: dealt
    over the sub-categories (per language) of both parts (fc_common.part_quotas, seeded on the
    suite), one per sub-category before any gets two. A part can get 0 (not run in that limited
    run); no limit: (None, None)."""
    q = part_quotas(limit, {"single": dict(single_counts), "agent": dict(agent_counts)},
                    seed="acebench")
    return q.get("single", 0 if limit and limit > 0 else None), \
        q.get("agent", 0 if limit and limit > 0 else None)


#: A single-turn row: (messages, gold, meta).
_Row = Tuple[List[Dict[str, Any]], str, Dict[str, Any]]


def _single_turn_window(samples: List[_Row],
                        chat_template_kwargs: Optional[Dict[str, Any]] = None
                        ) -> Tuple[List[_Row], List[Tuple[_Row, int]],
                                   Dict[str, Tuple[int, Optional[int], str]]]:
    """LLMInfer's 8192-token window over the single-turn rows of a no-think run
    (upstream_window_budget): (rows to send, rows longer than the window, verdicts).

    A row the window leaves less than the 1024 cap gets that room as its per-request max_tokens
    (base sends it as declared and counts a reply stopping there as stopping at the official
    cap). A row whose prompt is longer than the window is not sent: it is returned with its
    prompt tokens, and _unsent_row_traces scores the reply upstream's vLLM gives it, "".
    `verdicts` maps each row's gold to its (max_tokens, prompt tokens, how)."""
    from concurrent.futures import ThreadPoolExecutor

    def check(sample):
        return upstream_window_budget(sample[0], UPSTREAM_MAX_TOKENS, chat_template_kwargs)

    with ThreadPoolExecutor(max_workers=16) as pool:
        found = list(pool.map(check, samples))
    send: List[_Row] = []
    over: List[Tuple[_Row, int]] = []
    verdicts: Dict[str, Tuple[int, Optional[int], str]] = {}
    for (messages, gold, meta), verdict in zip(samples, found):
        verdicts[gold] = verdict
        limit, tokens, _how = verdict
        if limit == 0:
            over.append(((messages, gold, meta), int(tokens or 0)))
        elif limit < UPSTREAM_MAX_TOKENS:
            send.append((messages, gold, {**(meta or {}), "max_tokens": limit}))
        else:
            send.append((messages, gold, meta))
    return send, over, verdicts


def _window_quotas(limit: Optional[int], n_send: int, n_over: int
                   ) -> Tuple[Optional[int], Optional[int]]:
    """--eval-limit's single-turn quota split over the rows sent and the rows longer than
    LLMInfer's window, in proportion to their counts (the rows sent keep at least one)."""
    if not limit or limit <= 0 or not n_over:
        return limit, None
    over = min(n_over, int(round(limit * n_over / max(1, n_send + n_over))))
    send = limit - over
    if n_send and send < 1:
        send, over = 1, max(0, limit - 1)
    return send, over


def _select_unsent_rows(rows: List[Tuple[_Row, int]], quota: Optional[int]
                        ) -> List[Tuple[_Row, int]]:
    """The rows longer than LLMInfer's window that this run covers: GBENCH_SHARD and the
    --eval-limit quota applied to them as base applies them to the rows it sends (round-robin
    shard of the ordered list, then a subset stratified on the sub-category), so the shards
    still add up to every row once."""
    rows, _label = apply_shard(rows, "acebench")
    if quota is None:
        return rows
    return stratified_sample(rows, quota, lambda r: (r[0][2] or {}).get("category"),
                             seed="acebench")


def _unsent_row_traces(rows: List[Tuple[_Row, int]], eval_fn, attempts: int,
                       first_idx: int, first_source: int) -> List[Dict[str, Any]]:
    """Traces of the single-turn rows whose prompt is longer than LLMInfer's window. No request
    is sent: upstream's vLLM ignores such a prompt and returns empty text with finish reason
    "length" (UPSTREAM_MAX_MODEL_LEN), so the reply is "" and the row's checker scores it, as
    upstream's evaluation scores the "" in its result file. One trace per attempt."""
    traces: List[Dict[str, Any]] = []
    for j, ((messages, gold, meta), tokens) in enumerate(rows):
        ok = bool(eval_fn("", gold))
        for a in range(max(1, attempts)):
            traces.append({
                "sample_idx": first_idx + len(traces), "source_sample_idx": first_source + j,
                "attempt_idx": a, "category": (meta or {}).get("category"),
                "messages": messages, "gold_answer": gold, "response_text": "",
                "finish_reason": "length", "health": "ok", "max_tokens_sent": None,
                "completion_tokens": 0, "error": None, "is_correct": ok, "status": "OK",
                "over_upstream_window": True, "request_sent": False, "prompt_tokens": tokens,
            })
    return traces


def _add_unsent_rows(result: Dict[str, Any], traces: List[Dict[str, Any]]) -> None:
    """Merge _unsent_row_traces into base's single-turn result: its traces, counts, row
    accuracy and @k report."""
    if not traces:
        return
    result["sample_traces"] = list(result.get("sample_traces") or []) + traces
    result["total_questions"] = int(result.get("total_questions") or 0) + len(traces)
    result["correct_answers"] = int(result.get("correct_answers") or 0) + sum(
        1 for t in traces if t["is_correct"])
    scored = (result["total_questions"] - int(result.get("judge_outages") or 0)
              - int(result.get("scoring_excluded") or 0))
    result["accuracy"] = round(result["correct_answers"] / scored * 100.0, 2) if scored else 0.0
    health = dict(result.get("response_health") or {})
    health["over_upstream_window"] = health.get("over_upstream_window", 0) + len(traces)
    result["response_health"] = health
    report = result.get("attempts")
    if isinstance(report, dict) and (report.get("attempts_per_sample") or 1) > 1:
        by_source: Dict[Any, List[bool]] = {}
        for t in result["sample_traces"]:
            if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE" or t.get("scoring_excluded"):
                continue
            by_source.setdefault(t.get("source_sample_idx", t.get("sample_idx")), []).append(
                bool(t.get("is_correct")))
        n = len(by_source) or 1
        report.update(
            samples=len(by_source), generations=result["total_questions"],
            avg_at_k=result["accuracy"],
            pass_at_k=round(sum(1 for v in by_source.values() if any(v)) / n * 100.0, 2),
            pass_hat_k=round(sum(1 for v in by_source.values() if v and all(v)) / n * 100.0, 2),
            unstable_samples=sum(1 for v in by_source.values() if any(v) and not all(v)))


#: Worker threads of _row_scorer (a non-literal BinOp argument starts an evaluator container).
_SCORER_THREADS = 8


def _row_scorer(eval_fn):
    """async_eval_fn scoring the single-turn rows after generation, off the event loop: each
    reply base left pending goes through `eval_fn` (the row check, exactly as base's per-sample
    eval_fn path calls it: the reply with its thinking stripped, and the gold) in a worker
    thread, so an evaluator container never stalls the event loop."""
    async def score(sample_traces: List[Dict[str, Any]]) -> None:
        gate = asyncio.Semaphore(_SCORER_THREADS)

        async def one(t: Dict[str, Any]) -> None:
            gold = t.get("gold_answer")
            async with gate:
                ok = await asyncio.to_thread(
                    eval_fn, strip_thinking_tags(t.get("response_text") or ""), gold)
            t["is_correct"] = bool(ok)
            t["status"] = "TRUNCATED" if t.get("health") in ("truncated", "looping") else "OK"

        await asyncio.gather(*(one(t) for t in sample_traces
                               if t.get("status") == "PENDING_JUDGE"))
    return score


def _single_turn_cut(traces: List[Dict[str, Any]], limits: Dict[str, int], thinking: bool) -> int:
    """Single-turn replies cut off by a budget before they finished (finish reason length, an
    empty reply after reasoning included): with --thinking every such stop (the run's budget);
    without it, a stop below the row's official cap (1024, or the room LLMInfer's window leaves;
    `limits` maps gold -> the latter), i.e. a served window smaller than upstream's. Rows longer
    than the window are left out (not sent; their "" is upstream's reply)."""
    n = 0
    for t in traces:
        if t.get("finish_reason") != "length" or t.get("over_upstream_window"):
            continue
        if not thinking and t.get("max_tokens_sent") == limits.get(t.get("gold_answer"),
                                                                    UPSTREAM_MAX_TOKENS):
            continue
        n += 1
    return n


def user_sim_cutoff_reason(sim: Dict[str, Any]) -> str:
    """The leaderboard_comparable reason for user-simulator replies that stopped at
    USER_SIM_OUTPUT_LIMIT (the result's user_simulator counts), or "" when there are none. One
    entry with no "; " inside, so a sharded run's merge matches it whole."""
    cut = int((sim or {}).get("replies_cut_at_max_tokens") or 0)
    empty = int((sim or {}).get("empty_replies_at_max_tokens") or 0)
    outage = int((sim or {}).get("outages_after_max_tokens") or 0)
    if not (cut or empty or outage):
        return ""
    return (f"the user simulator's replacement stopped at max_output_tokens "
            f"{USER_SIM_OUTPUT_LIMIT} (APIUSER's max_tokens {USER_SIM_MAX_TOKENS} plus "
            f"{GEMINI_THINKING_ALLOWANCE} for the Gemini model's thinking) in {cut} replies used "
            f"as they were and {empty} with no text (the cascade asked its next model), and "
            f"{outage} simulator outage(s) followed such replies")


def _recount_user_simulator(merged: Dict[str, Any]) -> None:
    """A merged result's user_simulator record, counted again over the union's agent traces
    (each shard's record counts only its own conversations): the generic merge leaves a record
    the shards disagree on unreconciled."""
    unreconciled = merged.get("merge_unreconciled_fields") or {}
    shard_records = [v for v in (unreconciled.get("user_simulator") or {}).values()
                     if isinstance(v, dict)]
    sim = dict(merged.get("user_simulator") or (shard_records[0] if shard_records else {}))
    if not sim:
        return
    agent = [t for t in merged.get("sample_traces") or []
             if isinstance(t, dict) and "user_sim_requests" in t]
    sim["models_used"] = dict(sum((Counter(t.get("user_sim_models") or []) for t in agent),
                                  Counter()))
    sim["requests"] = sum(int(t.get("user_sim_requests") or 0) for t in agent)
    for key, field in (("replies_cut_at_max_tokens", "user_sim_replies_cut_at_max_tokens"),
                       ("empty_replies_at_max_tokens", "user_sim_empty_at_max_tokens"),
                       ("outages_after_max_tokens", "user_sim_outages_after_max_tokens")):
        sim[key] = sum(int(t.get(field) or 0) for t in agent)
    merged["user_simulator"] = sim
    if "user_simulator" in unreconciled:
        unreconciled.pop("user_simulator")
        if not unreconciled:
            merged.pop("merge_unreconciled_fields", None)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """gbench --merge: the generic re-count of the union's traces (which refuses shards whose
    headline is not correct / scored), with the user_simulator record and its cut-off reason
    counted again over the union's agent conversations, never one shard's."""
    return merge_with_cutoff_reason(
        "acebench", results,
        lambda r: user_sim_cutoff_reason(
            r.get("user_simulator") if isinstance(r.get("user_simulator"), dict) else {}),
        _recount_user_simulator)


def _status(result: Dict[str, Any]) -> str:
    """base._run_suite_async's status rule over the merged counts: single-turn replies that
    stopped at the official max_tokens (base's `replies_at_official_max_tokens`, which its
    `truncated_responses` includes) are the protocol's outcome and do not count as truncations
    (an agent conversation is counted in `truncated_responses` only for a reply cut shorter)."""
    total, failed = result.get("total_questions") or 0, result.get("failed_requests") or 0
    truncated = (result.get("truncated_responses") or 0) - (
        result.get("replies_at_official_max_tokens") or 0)
    if total == 0:
        return "error"
    if failed == total:
        return "failed"
    if failed == 0 and not result.get("judge_outages") and max(0, truncated) * 10 <= total:
        return "success"
    return "completed_with_errors"


def run_acebench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute acebench: Normal, Special and Agent in English and Chinese; headline = the English
    Summary (the leaderboard's overall), `overall_zh` the Chinese one. Without the user
    simulator's Gemini key, agent_multi_turn alone is reported infra_required
    (agent_multi_turn_category) and the Summaries are null."""
    limit = kwargs.get("limit")
    attempts = max(1, int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1))

    # Everything that can stop the run is checked before the first generation.
    singles: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    agent_items: List[Dict[str, Any]] = []
    for language in LANGUAGES:
        singles += _load_acebench_samples(enable_thinking=enable_thinking, language=language)
        agent_items += _load_agent_items(language=language)
    single_quota, agent_quota = _part_quotas(
        limit, Counter((s[2] or {}).get("category") for s in singles),
        Counter(it["key"] for it in agent_items))
    # --eval-limit N is N samples for the suite: a part none of whose samples was drawn is not run.
    if single_quota == 0:
        singles = []
    if agent_quota == 0:
        agent_items = []
    agent_items, shard_label = apply_shard(agent_items, "acebench")
    agent_items = stratified_sample(agent_items, agent_quota, lambda it: it["key"],
                                    seed="acebench:agent")
    limited = bool(limit and limit > 0)
    limit_plan = eval_limit_plan(
        "acebench", limit, (single_quota or 0) + (agent_quota or 0) if limited else None,
        per_setting={"single_turn": single_quota, "agent": agent_quota}, attempts=attempts,
        setting_word="part")
    # agent_multi_turn alone needs the user simulator: without it that category is reported
    # infra_required in its own entry and the other categories run (a hard error only when it is
    # all the run selected).
    sim_problem: Optional[str] = None
    held_out: List[Dict[str, Any]] = []
    if any(it["category"] == "agent_multi_turn" for it in agent_items):
        try:
            check_user_simulator()
        except UserSimulatorUnavailable as e:
            sim_problem = e.reason
        if sim_problem:
            held_out = [it for it in agent_items if it["category"] == "agent_multi_turn"]
            agent_items = [it for it in agent_items if it["category"] != "agent_multi_turn"]
            if not singles and not agent_items:
                raise infra_required("acebench", sim_problem, DOCS_URL)
            logger.error("acebench: agent_multi_turn is infra_required and not run (%s); the "
                         "other categories run and are scored", sim_problem.rstrip(". "))
    check_binop_evaluator()

    # BinOp arguments gbench refuses while scoring the Normal rows: (row id, expression).
    single_refusals: List[Tuple[str, str]] = []

    def eval_fn(response_text: str, gold: str) -> bool:
        sink: List[str] = []
        token = _REFUSALS.set(sink)
        try:
            return _eval_acebench(response_text, gold)
        finally:
            _REFUSALS.reset(token)
            if sink:
                try:
                    row_id = str(json.loads(gold).get("id"))
                except Exception:
                    row_id = "?"
                single_refusals.extend((row_id, text) for text in sink)

    budget = _max_tokens(enable_thinking)
    # LLMInfer's 8192-token window, emulated where its 1024 cap applies (a no-think run).
    upstream_window = budget.policy == "official"
    verdicts: Dict[str, Tuple[int, Optional[int], str]] = {}
    over_rows: List[Tuple[_Row, int]] = []
    send_quota, over_quota = single_quota, None
    if upstream_window and singles:
        request_ctk = effective_chat_template_kwargs(
            (kwargs.get("extra_payload") or {}).get("chat_template_kwargs"), False)
        singles, over_rows, verdicts = _single_turn_window(singles, request_ctk)
        send_quota, over_quota = _window_quotas(single_quota, len(singles), len(over_rows))

    result = run_eval_suite(
        eval_name="acebench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=singles,
        # Scored after generation in worker threads (_row_scorer), with eval_fn per row.
        async_eval_fn=_row_scorer(eval_fn),
        declared_scoring_mode="deterministic",
        supports_attempts=True,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=send_quota,
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=ACEBENCH_SAMPLING,
    )
    single_traces = list(result.get("sample_traces") or [])
    window_limits = {gold: v[0] for gold, v in verdicts.items() if 0 < v[0] < UPSTREAM_MAX_TOKENS}
    single_cut = _single_turn_cut(single_traces, window_limits, enable_thinking)
    sent_golds = {t.get("gold_answer") for t in single_traces}
    unsent = _select_unsent_rows(over_rows, over_quota) if over_rows else []
    if unsent:
        # The attempts base ran of each row it sent (when it sent any), for these rows too.
        row_attempts = (int((result.get("attempts") or {}).get("attempts_per_sample") or 1)
                        if single_traces else attempts)
        _add_unsent_rows(result, _unsent_row_traces(
            unsent, eval_fn, row_attempts, first_idx=len(single_traces),
            first_source=1 + max((t.get("source_sample_idx", t.get("sample_idx", 0))
                                  for t in single_traces), default=-1)))
    window_counts = {
        "limited": sum(1 for g in sent_golds if g in window_limits),
        "over_window": len(unsent),
        "uncounted": sum(1 for g in sent_golds
                         if (verdicts.get(g) or (0, None, ""))[2] == "uncounted"),
    }
    rows: Dict[str, List[Tuple[str, str, bool, int]]] = {lang: [] for lang in LANGUAGES}
    for t in scorable_traces(result):
        g = json.loads(t["gold_answer"])
        rows[g.get("language", "en")].append(
            (g["sub"], g["id"], bool(t.get("is_correct")), int(t.get("attempt_idx") or 0)))
    result["row_accuracy"] = result.get("accuracy")

    # The agent conversations, at the same sampling as the single-turn requests.
    temperature = result.get("temperature")
    payload = {**(ACEBENCH_SAMPLING.params or {}), **(kwargs.get("extra_payload") or {}),
               **decoding_penalties()}
    max_tokens = int(budget.max_tokens)
    runs = [{**it, "attempt_idx": a} for it in agent_items for a in range(attempts)]
    agent_traces = (asyncio.run(_run_agent_items(
        runs, model_name, base_url, concurrency, enable_thinking, temperature, payload,
        max_tokens, official_cap=max_tokens if budget.policy == "official" else None,
        upstream_window=upstream_window))
        if runs else [])
    agent_entries: Dict[str, Dict[str, List[Tuple[bool, float]]]] = {
        lang: {c: [] for c in AGENT_CATEGORIES} for lang in LANGUAGES}
    for t in agent_traces:
        if t["status"] == "OK" or t["status"] == "ERROR":
            p = t.get("process_accuracy")
            agent_entries[t["acebench_language"]][json.loads(t["gold_answer"])["sub"]].append(
                (bool(t["is_correct"]), float(p) if p is not None else 0.0))

    offset = len(result.get("sample_traces") or [])
    src_offset = 1 + max((t.get("source_sample_idx", t.get("sample_idx", 0))
                          for t in result.get("sample_traces") or []), default=-1)
    for i, t in enumerate(agent_traces):
        t["sample_idx"] = offset + i
        t["source_sample_idx"] = src_offset + i // attempts
    failed = [t for t in agent_traces if t["status"] in ("FAILED", "TIMEOUT", "OVER_CONTEXT")]
    outages = [t for t in agent_traces if t["status"] == "USER_SIM_OUTAGE"]
    errors = [t for t in agent_traces if t["status"] == "ERROR"]
    refusals = sum(len(t["execution_guard_refusals"]) for t in agent_traces)
    result["sample_traces"] = list(result.get("sample_traces") or []) + agent_traces
    result["total_questions"] = int(result.get("total_questions") or 0) + len(agent_traces)
    result["correct_answers"] = int(result.get("correct_answers") or 0) + sum(
        1 for t in agent_traces if t["is_correct"])
    result["failed_requests"] = int(result.get("failed_requests") or 0) + len(failed)
    result["timed_out_requests"] = int(result.get("timed_out_requests") or 0) + sum(
        1 for t in failed if t["status"] == "TIMEOUT")
    result["over_context_prompts"] = int(result.get("over_context_prompts") or 0) + sum(
        1 for t in failed if t["status"] == "OVER_CONTEXT")
    result["judge_outages"] = int(result.get("judge_outages") or 0) + len(outages)
    result["truncated_responses"] = int(result.get("truncated_responses") or 0) + sum(
        1 for t in agent_traces if t["truncated_replies"])
    # Agent replies that stopped at the official 1024 are counted per reply in the report; they
    # are not truncations (truncated_replies leaves them out).
    agent_at_cap = sum(int(t.get("replies_at_official_max_tokens") or 0) for t in agent_traces)

    per_language: Dict[str, Dict[str, Any]] = {}
    category_accuracy: Dict[str, Dict[str, Any]] = {}
    held_languages = {it["language"] for it in held_out}
    for language in LANGUAGES:
        agg = aggregate(rows[language], agent_entries[language],
                        unavailable=["agent_multi_turn"] if language in held_languages else None)
        per_sub = {s: {**v, "accuracy": _pct(v["accuracy"]),
                       **({"process_accuracy": _pct(v["process_accuracy"])}
                          if "process_accuracy" in v else {})}
                   for s, v in agg["per_sub_category"].items()}
        not_run, ceiling = draw_ceiling(agg["per_sub_category"])
        per_language[language] = {
            "overall": _pct(agg["summary"]),
            # sub-categories this run did not run (generate_result_csv reads each as 0) and the
            # Summary a model right on every drawn item would get
            "subcategories_not_run": not_run,
            "summary_ceiling": _pct(ceiling),
            "normal_summary": _pct(agg["normal_summary"]),
            "special_summary": _pct(agg["special_summary"]),
            "agent_summary": _pct(agg["agent_summary"]),
            "leaderboard_columns": {k: _pct(v) for k, v in agg["leaderboard_columns"].items()},
            "per_sub_category": per_sub,
        }
        # base's shape ({correct, total, accuracy} per category). A multi-turn Normal
        # sub-category counts conversations (correct = every turn right), an agent one
        # conversations (correct = end-to-end right). A sub-category reported infra_required
        # keeps its row, with a null accuracy and its status.
        for s, v in per_sub.items():
            category_accuracy[_category_key(s, language)] = {
                "correct": v["correct"], "total": v["total"], "accuracy": v["accuracy"],
                **({"status": v["status"]} if "status" in v else {})}
    en, zh = per_language["en"], per_language["zh"]
    result["per_language"] = per_language
    result["per_sub_category"] = en["per_sub_category"]
    result["per_sub_category_zh"] = zh["per_sub_category"]
    result["category_accuracy"] = category_accuracy
    result["normal_summary"] = en["normal_summary"]
    result["special_summary"] = en["special_summary"]
    result["agent_summary"] = en["agent_summary"]
    result["leaderboard_columns"] = en["leaderboard_columns"]
    result["overall_zh"] = zh["overall"]
    result["subcategories_not_run"] = {lang: per_language[lang]["subcategories_not_run"]
                                       for lang in LANGUAGES}
    result["summary_ceiling"] = {lang: per_language[lang]["summary_ceiling"]
                                 for lang in LANGUAGES}
    if rows["en"] or any(agent_entries["en"].values()) or "en" in held_languages:
        result["accuracy"] = en["overall"]
    result["metric"] = ("ACEBench Summary (English): 0.578 x normal_summary + 0.2676 x "
                        "special_summary + 0.1545 x agent_summary, the leaderboard's overall "
                        "(generate_result_csv); each summary is the unweighted mean of its "
                        "sub-categories. The Chinese Summary is overall_zh; per_language holds "
                        "every leaderboard column of both")
    if held_out:
        # Not run, never scored, never dropped: its own entry names the missing prerequisite,
        # and the aggregates that weigh it (agent_summary, the Summary) are null.
        result["categories_infra_required"] = ["agent_multi_turn"]
        result["agent_multi_turn_category"] = agent_multi_turn_infra_required(sim_problem or "",
                                                                              held_out)
        result["metric"] += ("; agent_multi_turn was not run (infra_required, see "
                             "agent_multi_turn_category), so agent_summary and the Summary, which "
                             "weigh it, are null (accuracy null); every other column is computed")
    result["dataset_revision"] = (f"chenchen0103/ACEBench@{ACEBENCH_COMMIT} data_all/data_en and "
                                  f"data_all/data_zh")
    sim_cut = sum(int(t.get("user_sim_replies_cut_at_max_tokens") or 0) for t in agent_traces)
    sim_empty = sum(int(t.get("user_sim_empty_at_max_tokens") or 0) for t in agent_traces)
    sim_outage_cap = sum(int(t.get("user_sim_outages_after_max_tokens") or 0)
                         for t in agent_traces)
    agent_cut = sum(int(t.get("truncated_replies") or 0) for t in agent_traces)
    agent_over_steps = sum(int(t.get("window_over_steps") or 0) for t in agent_traces)
    result["user_simulator"] = {
        "upstream": ("gpt-4o (model_inference/multi_turn/APIModel_user.py APIUSER, generate.py "
                     "--user-model default; temperature 0.001, top_p 1, max_tokens 1000)"),
        "replaced_by": "gbench Gemini judge cascade",
        "cascade": judge_cascade(),
        "sampling": (f"APIUSER's: temperature {USER_SIM_TEMPERATURE}, top_p {USER_SIM_TOP_P}, "
                     f"max_tokens {USER_SIM_MAX_TOKENS} for the reply, sent as max_output_tokens "
                     f"{USER_SIM_OUTPUT_LIMIT} with the Gemini thinking allowance"),
        **output_limit_record(USER_SIM_MAX_TOKENS, "APIUSER's max_tokens", "gpt-4o"),
        # the Gemini model that answered each user-simulator request
        "models_used": dict(sum((Counter(t["user_sim_models"]) for t in agent_traces),
                                Counter())),
        "requests": sum(t["user_sim_requests"] for t in agent_traces),
        # replies that stopped at USER_SIM_OUTPUT_LIMIT (finish reason MAX_TOKENS): with text,
        # used as they are; without text (the thinking took the whole limit), the cascade asked
        # its next model; and simulator outages whose cascade saw such a reply
        "replies_cut_at_max_tokens": sim_cut,
        "empty_replies_at_max_tokens": sim_empty,
        "outages_after_max_tokens": sim_outage_cap,
        **({"infra_required": sim_problem} if held_out else {}),
    }
    result["acebench_report"] = {
        "languages": list(LANGUAGES),
        "agent_conversations": len(agent_traces),
        # selected agent_multi_turn conversations not run (user simulator infra_required)
        "agent_conversations_infra_required": len(held_out),
        "agent_model_requests": sum(t["model_requests"] for t in agent_traces),
        "max_dialog_turns": MAX_DIALOG_TURNS,
        "max_tokens_per_request": max_tokens,
        "upstream_max_tokens": UPSTREAM_MAX_TOKENS,
        "token_policy": budget.policy,
        "agent_replies_at_official_max_tokens": agent_at_cap,
        "execution_guard_refusals": refusals,
        "single_turn_guard_refusals": len(single_refusals),
        "single_turn_guard_refused": [{"id": i, "expression": e} for i, e in single_refusals[:20]],
        "agent_failed": len(failed), "agent_user_sim_outages": len(outages),
        "agent_errors": len(errors),
        # replies cut off by a budget below the official cap (the run's budget with --thinking,
        # or a served window smaller than LLMInfer's), an empty reply after reasoning included
        "single_turn_replies_cut_off": single_cut,
        "agent_replies_cut_off": agent_cut,
        "upstream_context_window": {
            "max_model_len": UPSTREAM_MAX_MODEL_LEN,
            "applied": upstream_window,
            **({} if upstream_window else {
                "note": "not applied to reasoning: like the 1024 cap, it bounds non-reasoning "
                        "generation in ACEBench's own run"}),
            # rows / steps whose prompt was longer than the window: not sent, and scored (the
            # row) or carried on (the conversation) with the empty reply upstream's vLLM gives
            "over_window_reply": ("'' (vLLM 0.6.1.post1 ignores a prompt longer than "
                                  "max_model_len and returns empty text; not sent)"),
            "single_turn_rows_limited": window_counts["limited"],
            "single_turn_rows_over_window": window_counts["over_window"],
            "single_turn_rows_uncounted": window_counts["uncounted"],
            "agent_requests_limited": sum(int(t.get("window_limited_requests") or 0)
                                          for t in agent_traces),
            "agent_steps_over_window": agent_over_steps,
            "agent_conversations_over_window": sum(1 for t in agent_traces
                                                   if t.get("window_over_steps")),
            "agent_requests_uncounted": sum(int(t.get("window_uncounted_requests") or 0)
                                            for t in agent_traces),
        },
    }
    if shard_label:
        result["shard"] = shard_label
    result["status"] = _status(result)
    if not single_traces and agent_traces and result.get("status_reason") == "no samples were run":
        result.pop("status_reason")         # (base's, for the single-turn part not drawn)
    result["eval_limit_plan"] = limit_plan
    if held_out and result["status"] == "success":
        result["status"] = "completed_with_errors"

    reasons: List[str] = []
    if held_out:
        reasons.append(f"agent_multi_turn ({len(held_out)} conversations) was not run and not "
                       f"scored: its user simulator's prerequisite ({USER_SIM_PREREQUISITE}) was "
                       f"missing (infra_required: {str(sim_problem).rstrip('. ')}); agent_summary "
                       f"and the Summary are null")
    if limit:
        reasons.append(f"--eval-limit {limit} (ACEBench has {sum(ROW_COUNTS.values())} Normal + "
                       f"Special rows and {sum(AGENT_COUNTS.values())} agent conversations over "
                       f"the two languages)")
    en_not_run = per_language["en"]["subcategories_not_run"]
    subset = bool(limit) or bool(parse_shard(os.environ.get("GBENCH_SHARD")))
    if subset and en_not_run and result.get("accuracy") is not None:
        agent_missing = [sub for sub in en_not_run if sub in AGENT_CATEGORIES]
        reasons.append(
            f"the English Summary (the headline) counts the {len(en_not_run)} English "
            f"sub-categories this run did not draw as 0, as generate_result_csv reads a missing "
            f"category ({', '.join(en_not_run)}"
            + (f"; both agent categories among them" if len(agent_missing) == 2 else "")
            + f"), so a model right on every drawn item would score "
            f"{per_language['en']['summary_ceiling']}% (summary_ceiling)")
    if parse_shard(os.environ.get("GBENCH_SHARD")):
        reasons.append("sharded run (a subset of the rows; conversations may be split)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (ACEBench runs local models at 0.0)")
    if result.get("decoding_penalties"):
        reasons.append("decoding penalties set")
    if attempts > 1 or ((result.get("attempts") or {}).get("attempts_per_sample") or 1) > 1:
        reasons.append("--attempt-count > 1 (accuracies pool every attempt)")
    if result["failed_requests"]:
        reasons.append(f"{result['failed_requests']} failed requests or conversations (left out "
                       f"of the accuracies)")
    if outages:
        reasons.append(f"{len(outages)} agent_multi_turn conversations lost their user simulator "
                       f"(Gemini cascade outage; left out of the accuracies)")
    if errors:
        reasons.append(f"{len(errors)} agent conversations hit a case upstream's run aborts on "
                       f"(scored wrong): {errors[0]['error']}")
    if refusals:
        reasons.append(f"{refusals} model calls refused by the execution guard (upstream would "
                       f"have eval()ed them)")
    if single_refusals:
        reasons.append(f"{len(single_refusals)} BinOp arguments in Normal replies gbench did not "
                       f"evaluate (upstream eval()s them), e.g. {single_refusals[0][1]!r}")
    if single_cut or agent_cut:
        by = (f"the run's budget, max_tokens {max_tokens} (--thinking)" if enable_thinking
              else "a served context window that left less than ACEBench's cap")
        reasons.append(f"{single_cut} single-turn and {agent_cut} agent replies were cut off by "
                       f"{by} before they finished")
    uncounted = (window_counts["uncounted"]
                 + result["acebench_report"]["upstream_context_window"]["agent_requests_uncounted"])
    if uncounted:
        reasons.append(f"{uncounted} requests whose prompt could come near LLMInfer's "
                       f"{UPSTREAM_MAX_MODEL_LEN}-token window could not be counted exactly (no "
                       f"/tokenize count; the run's tokenizer gives only a conservative one) and "
                       f"were sent max_tokens {UPSTREAM_MAX_TOKENS}")
    sim_reason = user_sim_cutoff_reason(result["user_simulator"])
    if sim_reason:
        reasons.append(sim_reason)
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    return result
