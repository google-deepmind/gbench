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

# Evaluation Suite: nexusraven_api_eval
# Description: NexusRaven API Evaluation (single-call function calling; all 5 tasks, upstream SIMPLE-agent protocol)
# NOTE: Nexusflow/NexusRaven_API_evaluation is licensed CC-BY-NC-4.0 (non-commercial).

"""gbench native built-in runner for nexusraven_api_eval (Tool Use & Function Calling).

Renamed from `nexus_function_calling`: the suite runs Nexusflow's NexusRaven_API_evaluation set,
the five tasks the NexusRaven repo evaluates; the old name still runs this suite, with a
deprecation warning.

Runs nexusflowai/NexusRaven `raven/eval/evaluator.py` the way its chat-model scripts do
(`scripts/evaluate_gpt4.sh`: llm `openai_gpt4`, agent `SIMPLE`, i.e. LangChain 0.0.294's
structured-chat agent with `max_iterations=1` driving an `OpenAIChat` LLM):

  - each task's API list is exec'd into Python functions by `raven.utils.build_functions`
    (required args first, `JSON:` defaults decoded) whose body is `return (name, locals())`;
    the toolllm task uses `ToolLLMEvaluationDataHelper.build_functions` on the raw queries;
  - the prompt is LangChain's structured-chat prompt (PREFIX, one
    `name: name(signature) - docstring, args: {...}` line per context tool, FORMAT_INSTRUCTIONS,
    SUFFIX) rendered as the single user message `System: ...\\nHuman: <query>\\n\\n`, sent with
    stop `["Observation:"]`;
  - the reply goes through StructuredChatOutputParser; a malformed action blob gets LangChain's
    one OutputFixingParser repair call (NAIVE_FIX prompt) before counting as a failure;
  - a tool action runs the tool (pydantic-v1 argument validation, then the function); a final
    answer string is parsed as a Python call and applied to the function;
  - correct iff the function name and the FULL bound argument dict (defaults included) equal the
    reference call's.

The repo publishes its zero-shot results per domain (README, docs/Zero-shot_Evaluation.png):
"Generic Domain" (toolalpaca, toolllm) and "Cybersecurity Domain" (cve_cpe, emailrep,
virustotal), each the unweighted mean of its tasks' accuracies. Both are reported
(`domain_accuracy`); the headline is their unweighted mean.

Sampling: temperature 0.0 without `--thinking`, 1.0 with it, for the query and the repair call
alike (gbench's rule for every suite, base.resolve_temperature; the SIMPLE-agent scripts' own is 0,
evaluator.py build_llm: OpenAIChat with temperature 0). Override it
for a whole run with `--temperature`, or for this suite alone with
`GBENCH_NEXUSRAVEN_API_EVAL_TEMPERATURE` (the pre-rename `GBENCH_NEXUS_FUNCTION_CALLING_TEMPERATURE`
is still read as a deprecated alias), which takes precedence over both; the result records
upstream's value under `official_sampling`. No max_tokens is sent, in either mode, as OpenAIChat sends none
(LangChain 0.0.294 llms/openai.py: the request is the model plus model_kwargs): the server grants
the room left in the context window, and --max-output-tokens is not applied
(`max_output_tokens_note`). No LLM judge.
"""

import ast
import builtins
import inspect
import json
import logging
import os
import re
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from .base import OfficialSampling, run_eval_suite, strip_thinking_tags
from .metrics import scorable_traces
from .sampling import parse_shard, presharded_for_base, stratified_sample
from .fc_common import base_suite_plan, run_attempts
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
#: The registry name (renamed from `nexus_function_calling`; the stratified-sample seed keeps the
#: old string).
EVAL_NAME = "nexusraven_api_eval"
DOCS_URL = f"docs/evals/{EVAL_NAME}.md"

_REPO = "Nexusflow/NexusRaven_API_evaluation"
#: HF revision of the evaluation set (2023-09-29, the release the NexusRaven repo evaluates).
_REVISION = "85c7c326fcfb9c6587daabd9582e7cfaddcc65bd"
#: nexusflowai/NexusRaven revision the protocol is ported from.
NEXUSRAVEN_COMMIT = "f32882763bb3e220722883821919c14abdd3d156"
#: The five tasks the repo evaluates (README "Evaluation"); toolllm comes from raw_queries.
TASKS = ["cve_cpe", "emailrep", "virustotal", "toolalpaca", "toolllm"]
TASK_SIZES = {"cve_cpe": 78, "emailrep": 38, "virustotal": 151, "toolalpaca": 51, "toolllm": 21}
#: The two numbers the repo publishes for its zero-shot evaluation (README figure
#: docs/Zero-shot_Evaluation.png at NEXUSRAVEN_COMMIT), each the unweighted mean of its tasks'
#: accuracies: the figure's GPT-4 cybersecurity bar, 78.12, is the mean of its cve_cpe 64.10 and
#: virustotal 80.80 (docs/Single-Attempt_Function_Calling.png) and emailrep 34/38.
DOMAINS = {"Generic Domain": ["toolalpaca", "toolllm"],
           "Cybersecurity Domain": ["cve_cpe", "emailrep", "virustotal"]}
#: raven/eval/evaluator.py build_llm: the OpenAI chat models run as OpenAIChat(temperature=0),
#: which sends no other sampling field (and no max_tokens: LangChain 0.0.294 OpenAIChat's request
#: is {"model"} plus model_kwargs).
NEXUSRAVEN_SAMPLING = OfficialSampling(
    reference=(f"nexusflowai/NexusRaven@{NEXUSRAVEN_COMMIT[:12]} raven/eval/evaluator.py "
               "build_llm (OpenAIChat, gpt_params temperature 0)"),
    temperature=0.0,
    send_max_tokens=False,
)

# --------------------------------------------------------------------------- #
# LangChain 0.0.294 structured-chat agent text (langchain/agents/structured_chat/prompt.py,
# base.py HUMAN_MESSAGE_TEMPLATE) and output_parsers/prompts.py NAIVE_FIX, verbatim.
# --------------------------------------------------------------------------- #
PREFIX = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""
SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:"""
HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"
NAIVE_FIX = """Instructions:
--------------
{instructions}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------

Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:"""
#: StructuredChatAgent._stop
_STOP = ["Observation:"]
_ACTION_BLOB = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------- #
# Functions: raven/utils.py build_functions and toolllm_evaluation_data_utils.py, verbatim
# apart from exec'ing into an explicit namespace that holds the names the upstream module had.
# --------------------------------------------------------------------------- #
def build_functions(api_rows: List[Dict[str, Any]]) -> Dict[str, Callable]:
    namespace: Dict[str, Any] = {"Any": Any, "Callable": Callable, "Dict": Dict, "List": List,
                                 "Tuple": Tuple, "ast": ast, "json": json}
    function_names = []
    for function_dict in api_rows:
        name = function_dict["name"]
        description = function_dict["description"]
        args_dicts = function_dict["args_dicts"]

        args_dicts = sorted(args_dicts, key=lambda d: d["required"], reverse=True)

        args_strs = []
        for arg_dict in args_dicts:
            if arg_dict["required"]:
                default_str = ""
            else:
                default = arg_dict["default"]
                if isinstance(default, str) and default.startswith("JSON:"):
                    default = default.removeprefix("JSON:")
                    default = json.loads(default)
                default_str = f" = {default}"

            if arg_dict["type"] == "None":
                type_str = ""
            else:
                type_str = f": {arg_dict['type']}"

            args_str = f"{arg_dict['name']}{type_str}{default_str}"
            args_strs.append(args_str)

        function_str = f'def {name}({", ".join(args_strs)}):\n    """{description}"""\n    return ("{name}", locals())'
        exec(function_str, namespace)
        function_names.append(name)
    return {f: namespace[f] for f in function_names}


def build_toolllm_functions(raw_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Callable], List[Dict[str, Any]]]:
    """ToolLLMEvaluationDataHelper.build_functions: (functions, eval rows). Later definitions of a
    name replace earlier ones, as upstream's dict.update does."""
    namespace: Dict[str, Any] = {"Callable": Callable, "Dict": Dict, "List": List, "json": json,
                                 "datetime": datetime, "ENUM": Enum}
    functions: Dict[str, Callable] = {}
    rows = []
    for example in raw_rows:
        example = json.loads(example["query_dict"])
        function_names = []
        for function_str in example["context"]:
            # Replace braces since they cause problems for langchain's formatting
            function_str = function_str.replace("{", "")
            function_str = function_str.replace("}", "")
            function_name = function_str[: function_str.find("(")].removeprefix("def ")
            # Remove the function body to use our own for evaluation
            function_str = function_str[: function_str.find("args_dict")]
            function_str = f"""
{function_str}return ("{function_name}", {{k: int(v) if isinstance(v, float) else v for k, v in locals().items()}})
"""
            exec(function_str, namespace)
            function_names.append(function_name)
        functions.update({n: namespace[n] for n in function_names})
        rows.append({"context_functions": function_names, "prompt": example["Input"],
                     "reference_function_call": example["Output"]})
    return functions, rows


# --------------------------------------------------------------------------- #
# LangChain StructuredTool.from_function / create_schema_from_function / _parse_input
# --------------------------------------------------------------------------- #
class _Tool:
    """The parts of a LangChain 0.0.294 StructuredTool the SIMPLE agent uses."""

    def __init__(self, func: Callable):
        from pydantic.v1 import Extra, create_model, validate_arguments

        class _SchemaConfig:
            extra: Any = Extra.forbid
            arbitrary_types_allowed: bool = True

        if func.__doc__ is None:
            raise ValueError("Function must have a docstring if description not provided.")
        self.func = func
        self.name = func.__name__
        self.description = f"{self.name}{inspect.signature(func)} - {func.__doc__.strip()}"
        inferred = validate_arguments(func, config=_SchemaConfig).model
        for key in ("run_manager", "callbacks"):
            inferred.__fields__.pop(key, None)
        props = inferred.schema()["properties"]
        valid = {k: props[k] for k in inspect.signature(func).parameters
                 if k not in ("run_manager", "callbacks")}
        fields = {}
        for field_name in valid:
            field = inferred.__fields__[field_name]
            fields[field_name] = (field.outer_type_, field.field_info)
        self.args_schema = create_model(f"{self.name}SchemaSchema", **fields)
        self.args = self.args_schema.schema()["properties"]

    def run(self, tool_input: Any) -> Any:
        if isinstance(tool_input, str):
            key_ = next(iter(self.args_schema.__fields__.keys()))
            self.args_schema.validate({key_: tool_input})
            return self.func(tool_input)
        result = self.args_schema.parse_obj(tool_input)
        parsed = {k: v for k, v in result.dict().items() if k in tool_input}
        return self.func(**parsed)


def agent_prompt(tools: List[_Tool], query: str) -> str:
    """StructuredChatAgent.create_prompt rendered, then turned into OpenAIChat's single prompt
    string by get_buffer_string ("System: ...\\nHuman: ...")."""
    tool_strings = []
    for tool in tools:
        # create_prompt quadruples the braces and the template render halves them, so the
        # model sees the args schema with doubled braces, as upstream's prompts did.
        args_schema = re.sub("}", "}}}}", re.sub("{", "{{{{", str(tool.args)))
        tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")
    formatted_tools = "\n".join(tool_strings)
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
    template = "\n\n".join([PREFIX, formatted_tools, format_instructions, SUFFIX])
    system = template.format()   # SystemMessagePromptTemplate render (no variables)
    human = HUMAN_MESSAGE_TEMPLATE.format(input=query, agent_scratchpad="")
    return f"System: {system}\nHuman: {human}"


def fix_prompt(completion: str) -> str:
    """OutputFixingParser's retry prompt for a reply the structured-chat parser rejected."""
    error = repr(ValueError(f"Could not parse LLM output: {completion}")).replace(
        "ValueError(", "OutputParserException(", 1)
    return NAIVE_FIX.format(instructions=FORMAT_INSTRUCTIONS, completion=completion, error=error)


class _ParseError(Exception):
    pass


def parse_agent_output(text: str) -> Tuple[str, Any, Any]:
    """StructuredChatOutputParser.parse -> ("finish", output, None) or ("action", tool, input)."""
    try:
        action_match = _ACTION_BLOB.search(text)
        if action_match is not None:
            response = json.loads(action_match.group(1).strip(), strict=False)
            if isinstance(response, list):
                response = response[0]
            if response["action"] == "Final Answer":
                return "finish", response["action_input"], None
            return "action", response["action"], response.get("action_input", {})
        return "finish", text, None
    except Exception as e:
        raise _ParseError(f"Could not parse LLM output: {text}") from e


def parse_function_call_to_name_and_args(function: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    """raven/utils.py, verbatim (keyword floats become ints; positional args do not)."""
    function = ast.parse(function).body[0]
    function_name = function.value.func.id
    keywords = function.value.keywords
    keywords = {k.arg: ast.literal_eval(k.value) for k in keywords}
    args = [ast.literal_eval(arg) for arg in function.value.args]
    keywords = {k: int(v) if isinstance(v, float) else v for k, v in keywords.items()}
    return function_name, args, keywords


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
class _Task:
    def __init__(self, name: str, functions: Dict[str, Callable]):
        self.name = name
        self.functions = functions
        self.tools = {n: _Tool(f) for n, f in functions.items()}
        #: sample key -> (context function names, reference name, reference bound args)
        self.samples: Dict[int, Tuple[List[str], str, Dict[str, Any]]] = {}


_TASKS: Dict[str, _Task] = {}


def predict(task: _Task, context: List[str], parsed: Tuple[str, Any, Any]) -> Tuple[Any, Any]:
    """The evaluator's (predicted_function_name, predicted_args_dict) for one parsed reply."""
    kind, first, second = parsed
    if kind == "action":
        if first not in context:
            # InvalidTool, then max_iterations=1 stops the agent with a plain-text answer that
            # does not parse as a call.
            return None, None
        output = task.tools[first].run(second)
    elif isinstance(first, str):
        function_name, args, kwargs = parse_function_call_to_name_and_args(first)
        output = task.functions[function_name](*args, **kwargs)
    else:
        return None, None
    if not isinstance(output, tuple):
        return None, None
    return output


def score(task_name: str, key: int, text: str) -> Tuple[bool, bool]:
    """(correct, parser_rejected) for one reply without the repair call."""
    task = _TASKS[task_name]
    context, ref_name, ref_args = task.samples[key]
    try:
        parsed = parse_agent_output(text)
    except _ParseError:
        return False, True
    try:
        name, args = predict(task, context, parsed)
    except Exception:                                          # noqa: BLE001 - evaluator: (None, None)
        return False, False
    return name == ref_name and args == ref_args, False


def _load_nexusraven_api_eval_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """All five NexusRaven tasks (339 queries), prompts built as the SIMPLE agent builds them."""
    try:
        from datasets import load_dataset
        sq = list(load_dataset(_REPO, "standardized_queries", split="train", revision=_REVISION))
        al = list(load_dataset(_REPO, "standardized_api_list", split="train", revision=_REVISION))
        rq = list(load_dataset(_REPO, "raw_queries", split="train", revision=_REVISION))
    except Exception as e:
        raise infra_required(EVAL_NAME, f"{_REPO}@{_REVISION} could not be loaded ({e}).",
                             DOCS_URL) from e

    rows: List[Dict[str, Any]] = []
    for task_name in TASKS:
        if task_name == "toolllm":
            functions, eval_rows = build_toolllm_functions([r for r in rq if r["dataset"] == "toolllm"])
        else:
            functions = build_functions([r for r in al if r["dataset"] == task_name])
            eval_rows = [r for r in sq if r["dataset"] == task_name]
        task = _Task(task_name, functions)
        _TASKS[task_name] = task
        if len(eval_rows) != TASK_SIZES[task_name]:
            raise RuntimeError(f"{EVAL_NAME}: {task_name} has {len(eval_rows)} queries, "
                               f"expected {TASK_SIZES[task_name]}")
        for key, sample in enumerate(eval_rows):
            if "reference_function_call" in sample:
                ref_name, ref_args = eval(sample["reference_function_call"],
                                          {"__builtins__": builtins}, dict(functions))
            else:
                ref_name = sample["python_function_name"]
                _, ref_args = functions[ref_name](**json.loads(sample["python_args_dict"]))
            context = list(sample["context_functions"])
            task.samples[key] = (context, ref_name, ref_args)
            rows.append({"task": task_name, "key": key, "prompt": sample["prompt"],
                         "context": context})

    # Seeded on the pre-rename name, so --eval-limit draws the same queries as before.
    rows = stratified_sample(rows, limit, lambda r: r["task"], seed="nexus_function_calling")
    samples = []
    for r in rows:
        task = _TASKS[r["task"]]
        content = agent_prompt([task.tools[k] for k in r["context"]], r["prompt"])
        gold = json.dumps({"task": r["task"], "key": r["key"]})
        samples.append(([{"role": "user", "content": content}], gold, {"category": r["task"]}))
    logger.info(f"Loaded {len(samples)} {EVAL_NAME} samples from HF Hub.")
    return samples


def _eval_nexusraven_api_eval(response_text: str, gold: str) -> bool:
    """First-pass verdict; a parser-rejected reply is re-scored after the repair call."""
    g = json.loads(gold)
    return score(g["task"], g["key"], response_text or "")[0]


def _eval_fixed(response_text: str, gold: str) -> bool:
    """Verdict on the repaired reply: parsed once more with no further repair."""
    return _eval_nexusraven_api_eval(response_text, gold)


def domain_accuracy(per_task: Dict[str, List[bool]]) -> Dict[str, Optional[float]]:
    """NexusRaven's two domain numbers in percent: the unweighted mean of the exact accuracies of
    the domain's tasks that ran (None when none of them ran)."""
    out: Dict[str, Optional[float]] = {}
    for domain, tasks in DOMAINS.items():
        accs = [100.0 * sum(per_task[t]) / len(per_task[t]) for t in tasks if per_task.get(t)]
        out[domain] = round(sum(accs) / len(accs), 2) if accs else None
    return out


def run_nexusraven_api_eval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute nexusraven_api_eval: per-task accuracy as raven/eval/evaluator.py reports it and the
    repo's two domain accuracies; the headline is the unweighted mean of the two domains."""
    samples = _load_nexusraven_api_eval_samples(limit=kwargs.get("limit"))
    extra = dict(kwargs.get("extra_payload") or {})
    common = dict(model_name=model_name, base_url=base_url, concurrency=concurrency,
                  thinking=enable_thinking,
                  max_output_tokens=kwargs.get("max_output_tokens"))
    limit_plan = base_suite_plan(EVAL_NAME, samples, kwargs.get("limit"), run_attempts(kwargs),
                                 note="plus one OutputFixingParser repair call per reply that "
                                      "fails to parse")
    # The protocol's follow-ups: at most one repair call per generation (the count that ran is
    # recorded as `repair_calls` once the first pass is scored).
    if limit_plan.get("generations") is not None:
        limit_plan["repair_calls_at_most"] = limit_plan["generations"]
    result = run_eval_suite(eval_name=EVAL_NAME, samples=samples,
                            eval_fn=_eval_nexusraven_api_eval, limit=kwargs.get("limit"),
                            limit_seed="nexus_function_calling",
                            extra_payload={**extra, "stop": list(_STOP)},
                            official_sampling=NEXUSRAVEN_SAMPLING, **common)
    result["eval_limit_plan"] = limit_plan

    traces = list(scorable_traces(result))
    verdict: Dict[int, bool] = {}
    fix_samples, fix_of = [], []
    for i, t in enumerate(traces):
        g = json.loads(t["gold_answer"])
        # The same reasoning-stripped text the per-sample verdict saw.
        text = strip_thinking_tags(t.get("response_text") or "")
        ok, rejected = score(g["task"], g["key"], text)
        verdict[i] = ok
        if rejected:
            # OutputFixingParser: one repair call (no stop sequence), then parse once more.
            fix_samples.append(([{"role": "user", "content": fix_prompt(text)}], t["gold_answer"],
                                {"category": g["task"]}))
            fix_of.append(i)
    fixed = repair_harness_errors = 0
    limit_plan["repair_calls"] = len(fix_samples)
    if fix_samples:
        # The repairs are already this node's (they come from its traces): laid out so base's
        # GBENCH_SHARD selection keeps every one, in order, and a trace's sample_idx indexes
        # fix_samples.
        fix_result = run_eval_suite(eval_name=EVAL_NAME, samples=presharded_for_base(fix_samples),
                                    eval_fn=_eval_fixed, extra_payload=extra or None, limit=None,
                                    attempt_count=1, official_sampling=NEXUSRAVEN_SAMPLING,
                                    **common)
        for ft in fix_result.get("sample_traces", []):
            i = fix_of[ft.get("source_sample_idx", ft["sample_idx"])]
            traces[i]["output_fixing_response"] = ft.get("response_text")
            if ft.get("harness_error"):
                # The repair call got no reply from the served endpoint: a harness error, so the
                # query has no verdict. Left out of every accuracy, counted and named.
                traces[i]["harness_error"] = {
                    **ft["harness_error"],
                    "detail": f"output-fixing call: {ft['harness_error'].get('detail')}"[:300]}
                traces[i]["scoring_excluded"] = True
                repair_harness_errors += 1
                continue
            ok = bool(ft.get("is_correct")) and ft.get("response_text") is not None
            verdict[i] = ok
            fixed += int(ok)
        result["output_fixing"] = {"repair_calls": len(fix_samples), "repaired_correct": fixed,
                                   "failed_requests": fix_result.get("failed_requests"),
                                   "harness_errors": repair_harness_errors}
    per_task: Dict[str, List[bool]] = {}
    for i, t in enumerate(traces):
        t["is_correct"] = verdict[i] and not t.get("scoring_excluded")
        if t.get("scoring_excluded"):
            continue
        per_task.setdefault(json.loads(t["gold_answer"])["task"], []).append(verdict[i])
    result["per_task"] = {task: {"n": len(v), "accuracy": round(100.0 * sum(v) / len(v), 2)}
                          for task, v in per_task.items() if v}
    result["domain_accuracy"] = domain_accuracy(per_task)
    if repair_harness_errors:
        result["scoring_excluded"] = (int(result.get("scoring_excluded") or 0)
                                      + repair_harness_errors)
        cause = (f"harness errors left out of the score: {repair_harness_errors} output-fixing "
                 "call(s) that got no reply from the served endpoint")
        result["status_reason"] = "; ".join(filter(None, (result.get("status_reason"), cause)))
        if result.get("status") == "success":
            result["status"] = "completed_with_errors"
    # A domain whose queries ran but none was scored (each one's request or output-fixing call
    # got no reply from the endpoint: harness errors) has no accuracy, and so has no mean of the
    # two domains: withheld, never the other domain's accuracy under the mean's name.
    tasks_run = {json.loads(t["gold_answer"])["task"] for t in result.get("sample_traces") or []
                 if t.get("gold_answer")}
    domains_run = [d for d, tasks in DOMAINS.items() if tasks_run & set(tasks)]
    ran = [result["domain_accuracy"][d] for d in domains_run
           if result["domain_accuracy"].get(d) is not None]
    if domains_run and len(ran) == len(domains_run):
        result["accuracy"] = round(sum(ran) / len(ran), 2)
    elif domains_run:
        missing = [d for d in domains_run if result["domain_accuracy"].get(d) is None]
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = (
            f"no scored query in the {' and '.join(missing)} (every request or output-fixing "
            "call of it got no reply from the served endpoint: harness errors), so the mean of "
            "the two domains has no value: " + str(result.get("status_reason")
                                                   or "left out of the score"))
        if not per_task and result.get("status") not in ("error", "failed"):
            result["status"] = "failed"
    if result["per_task"]:
        result["correct_answers"] = sum(sum(v) for v in per_task.values())
    result["effective_n"] = sum(len(v) for v in per_task.values())
    # base's shape ({correct, total, accuracy} per category), with the repaired verdicts.
    result["category_accuracy"] = {task: {"correct": sum(v), "total": len(v),
                                          "accuracy": result["per_task"][task]["accuracy"]}
                                   for task, v in per_task.items() if v}
    result["metric"] = ("unweighted mean of NexusRaven's two zero-shot domain accuracies (Generic: "
                        "toolalpaca, toolllm; Cybersecurity: cve_cpe, emailrep, virustotal; each the "
                        "unweighted mean of evaluator.py task accuracy, function name + full bound "
                        "argument dict); domain_accuracy and per_task hold the published numbers")
    result["dataset_revision"] = f"{_REPO}@{_REVISION}; protocol nexusflowai/NexusRaven@{NEXUSRAVEN_COMMIT}"
    reasons = ["NexusRaven publishes the Generic and Cybersecurity domain accuracies; the headline "
               "is gbench's unweighted mean of the two (domain_accuracy holds the comparable "
               "numbers)"]
    if kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs.get('limit')} (NexusRaven scores all "
                       f"{sum(TASK_SIZES.values())} queries)")
    if parse_shard(os.environ.get("GBENCH_SHARD")):
        reasons.append("sharded run (a subset of the queries)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (the SIMPLE-agent scripts "
                       "sample at 0)")
    if result.get("decoding_penalties"):
        reasons.append("decoding penalties set")
    if ((result.get("attempts") or {}).get("attempts_per_sample") or 1) > 1:
        reasons.append("--attempt-count > 1 (accuracy pools every attempt)")
    no_reply = sum(1 for t in result.get("sample_traces") or [] if t.get("harness_error"))
    if no_reply:
        reasons.append(f"{no_reply} quer(ies) whose request or output-fixing call got no reply "
                       "from the served endpoint (harness errors, left out of the score)")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
