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

"""Berkeley Function Calling Leaderboard (BFCL) v4, every scoring category, BFCL's own protocol.

Categories: all 22 of bfcl_eval's ALL_SCORING_CATEGORIES at commit BFCL_EVAL_COMMIT (the current
upstream release): 13 single-turn (non-live simple_python/java/javascript, multiple, parallel,
parallel_multiple, irrelevance; live simple/multiple/parallel/parallel_multiple, irrelevance,
relevance), 4 multi-turn (multi_turn_base, _miss_func, _miss_param, _long_context) and 5 agentic
(memory_kv, memory_vector, memory_rec_sum, web_search_base, web_search_no_snippet).

Single-turn: BFCL's FC mode for an OpenAI chat-completions model. The dataset's turn-0 messages
are sent verbatim with the language-hinted, convert_to_tool'ed docs as `tools`, as
OpenAICompletionsHandler._query_FC sends them; each reply becomes the model result that handler's
_parse_query_response_FC records (its tool calls as [{name: arguments}], else its text), and
bfcl_eval's own eval_runner (_evaluate_single_ast_entry / _evaluate_single_relevance_entry, i.e.
decode_ast, ast_checker and the relevance rule) scores it in the harness image. Reply text is
never a call, as in BFCL's FC handler.

Multi-turn: run inside bfcl-eval itself, in the harness image. Each entry goes through
OpenAICompletionsHandler.inference_multi_turn_FC (BFCL's own loop: tools per turn, held-out
functions added on their turn, 20-step cap), its calls execute in bfcl_eval's executable backends,
and eval_runner's multi-turn entry check (state and execution-response checkers) scores it.

Agentic: delegated to the bfcl_v4_agentic suite, which drives the same bfcl-eval harness. Its
Web Search setting is served by Gemini Google Search grounding in place of BFCL's DuckDuckGo via
SerpAPI search, so a run that scores it is not leaderboard-comparable ("web search served by
Gemini Google Search grounding (BFCL queries DuckDuckGo via SerpAPI)"). Without a Gemini key
(GEMINI_API_KEYS / GEMINI_API_KEY), with one the Gemini API rejects, or when a search the
grounding cascade could not serve reached the model during generation, that setting alone is
reported infra_required (`web_search_setting`, `settings_infra_required`): not scored, never
dropped, Overall Acc and the Agentic column null, the status completed_with_errors. Single-turn,
multi-turn and memory run as usual. Only a run that holds nothing but web_search hard-errors.

The harness image is the local gbench-bfcl:2026.3.23 (docker/bfcl.Dockerfile, launched by
bfcl_v4_agentic.run_in_image): bfcl-eval 2026.3.23, byte-identical to gorilla at
BFCL_EVAL_COMMIT, with its own pins. gbench's own process never imports bfcl_eval. Inside the
image every bfcl_eval file the single-turn checker or the multi-turn path uses must match the
pinned commit, else the suite hard-errors (infra_required), as it does without Docker or the image.

Networks: the image probe and the single-turn checker run with `--network none`. The multi-turn
part talks to the model endpoint and executes the model's calls with eval() in its own process
(under the execution guard, _harness_patches), so it runs on docker's bridge network, never the
host network (the host's DOCKER-USER rule keeps the cloud metadata server out of reach of a bridge
network only). Before the first generation, endpoint_network checks that the metadata server is
blocked there (container_guard) and that an endpoint on this host answers at
host.docker.internal, the URL the part then uses, or, with a real endpoint key, that the bridge
gateway where gbench's endpoint relay listens is reachable; any failing is a hard error
(infra_required). No key enters the container: a real OPENAI_API_KEY is added by
swebench_common.EndpointRelay on the bridge gateway (multi_turn_report.endpoint_relay) and the
container holds the placeholder EMPTY. The agentic part's containers, relays and execution guard
follow bfcl_v4_agentic (a harness project root that holds a .env, which upstream's commands would
load into the process that runs the model's calls, is refused there before any container starts);
its checks run as this suite's, so a failure names bfcl and docs/evals/bfcl.md
(bfcl_v4_agentic.suite_context).

Bounds, as BFCL's own: each request by its client's timeout (the multi-turn and agentic clients'
are sized for the longest answer the served context window allows, base.request_timeout_s of the
window, or of base.UNCAPPED_TIMEOUT_TOKENS when the server reports none), each multi-turn entry by
BFCL's 20-step limit per turn, and an entry whose request fails is recorded as an inference error
while the run goes on; no part is bounded as a whole.
Harness errors are never the model's outcome, in every part alike (base's rule): an entry whose
request failed (FAILED, TIMEOUT, OVER_CONTEXT; BFCL records an inference error and scores it
wrong) is left out of its category's correct and total, counted under `failed_requests` (and
`timed_out_requests` / `over_context_prompts`), in `scoring_excluded` and, by kind, in
`request_harness_errors` (`endpoint_failure`, `endpoint_timeout`, `over_context`), named in
`leaderboard_comparable_reason`, the run completed_with_errors. A multi-turn entry whose checker
raised (`multi_turn:checker_exception`; upstream's evaluation aborts there) is left out the same
way, as `checker_exception`. The single-turn part gets this
from base; multi-turn traces are marked `scoring_excluded` with their `harness_error`; the
agentic part takes such entries out of BFCL evaluate's counts. The agentic part's memory
prerequisite entries whose request failed (not scored entries, so not in `failed_requests`:
`memory_prereq_failed_requests`, per category in `agentic_report.prereq_inference_errors`) leave
their scenario's memory entries running against an incomplete memory, so those entries are left
out of the score too (`memory_prereq_failed`). A category with no entry left in the score is
unscored (null columns, never 0; `categories_unscored`); a multi-turn or agentic part whose every
entry failed its request is also recorded under `harness_errors`; a run where nothing was scored
withholds its score (accuracy null, `accuracy_withheld_reason`, status failed).
GBENCH_BFCL_TIMEOUT_S, when set, bounds each harness part on its own (the single-turn checker,
the multi-turn part, `bfcl generate`, `bfcl evaluate`). A part it stops is recorded under
`harness_errors` and the rest of the run is kept: finished multi-turn and agentic entries are
scored and the unfinished ones are harness errors left out of the score (`unfinished`, counted
as timed out); replies the checker did not score are marked HARNESS_ERROR and left out of every
accuracy; a stopped `bfcl evaluate` leaves the agentic categories unscored. Such a run is
completed_with_errors and not leaderboard-comparable.
Endpoint failures gbench's endpoint relay counted (`upstream_errors`, `requests_refused`) are
harness errors: `endpoint_failures`, completed_with_errors and a named non-comparability reason.

Headline: BFCL Overall Acc as eval_runner_helper.py computes it: 10% Non-Live AST, 10% Live AST,
10% Irrelevance, 30% Multi-Turn, 40% Agentic. Every leaderboard column is in `bfcl_summary`.

Sampling: gbench's rule for every suite (base.resolve_temperature): temperature 1.0 with
`--thinking`, 0.0 without (BFCL's own 0.001, the default of `bfcl generate --temperature`, is
recorded under `official_sampling`: fc_common.BFCL_OFFICIAL_SAMPLING). The same value for all three parts: the agentic part is handed this suite's temperature. No request carries max_tokens, as
BFCL's handler sends none (the server grants the rest of the context window); --max-output-tokens
is not applied (`max_output_tokens_note`). Override the temperature for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BFCL_TEMPERATURE`, which takes precedence
over both; a value other than BFCL's 0.001 makes the result not leaderboard-comparable. No LLM judge.
"""

import ast
import builtins
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .base import (UNCAPPED_TIMEOUT_TOKENS, decoding_penalties, effective_chat_template_kwargs,
                   endpoint_harness_error, gbench_cache_dir, get_run_knob, redact_text,
                   request_timeout_s, resolve_official_sampling, run_eval_suite, suite_env)
from . import bfcl_v4_agentic as _agentic
from .bfcl_v4_agentic import (BFCL_DOCKERFILE, BFCL_EVAL_RELEASE, MEMORY_CATEGORIES,
                              WEB_SEARCH_CATEGORIES, HarnessTimeout, SearchUnavailable,
                              bfcl_image, image_build_command, max_tokens_note, part_timeout,
                              report_progress, web_search_infra_required)
from .fc_common import (BFCL_EVAL_COMMIT, BFCL_OFFICIAL_SAMPLING, bfcl_convert_to_tools,
                        bfcl_request_functions, bfcl_score, eval_limit_plan, part_quotas)
from .sampling import parse_shard, shard_select, stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/bfcl.md"

BFCL_DATA_URL = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/"
                 f"{BFCL_EVAL_COMMIT}/berkeley-function-call-leaderboard/bfcl_eval/data")

# bfcl_eval/constants/category_mapping.py. `sql`, `exec_*`, `rest`, `chatable` and
# `multi_turn_composite` are commented out upstream and no longer scored; `format_sensitivity`
# is not scored and is skipped for FC models.
NON_LIVE_CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "multiple",
                       "parallel", "parallel_multiple", "irrelevance"]
LIVE_CATEGORIES = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
                   "live_irrelevance", "live_relevance"]
SINGLE_TURN_CATEGORIES = NON_LIVE_CATEGORIES + LIVE_CATEGORIES
MULTI_TURN_CATEGORIES = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param",
                         "multi_turn_long_context"]
#: The bfcl_v4_agentic suite runs these; bfcl delegates to it.
AGENTIC_CATEGORIES = list(MEMORY_CATEGORIES) + list(WEB_SEARCH_CATEGORIES)
#: bfcl_eval ALL_SCORING_CATEGORIES, in its order.
ALL_SCORING_CATEGORIES = SINGLE_TURN_CATEGORIES + MULTI_TURN_CATEGORIES + AGENTIC_CATEGORIES
SUPPORTED_BFCL_CATEGORIES = list(ALL_SCORING_CATEGORIES)
#: bfcl_eval TEST_COLLECTION_MAPPING (its "all" also holds format_sensitivity, which FC models skip).
_CATEGORY_GROUPS = {
    "all": ALL_SCORING_CATEGORIES,
    "all_scoring": ALL_SCORING_CATEGORIES,
    "single_turn": SINGLE_TURN_CATEGORIES,
    "non_live": NON_LIVE_CATEGORIES,
    "live": LIVE_CATEGORIES,
    "multi_turn": MULTI_TURN_CATEGORIES,
    "agentic": AGENTIC_CATEGORIES,
    "memory": list(MEMORY_CATEGORIES),
    "web_search": list(WEB_SEARCH_CATEGORIES),
    "python": ["simple_python", "irrelevance", "parallel", "multiple", "parallel_multiple",
               "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
               "live_irrelevance", "live_relevance"],
    "non_python": ["simple_java", "simple_javascript"],
}
# BFCL v4 renamed these; the old names keep working.
_RENAMED_CATEGORIES = {"simple": "simple_python", "java": "simple_java",
                       "javascript": "simple_javascript"}

#: The categories each segment of BFCL Overall Acc weighs, with its weight
#: (eval_runner_helper.generate_leaderboard_csv). live_relevance is reported but not weighed.
OVERALL_SEGMENTS: List[Tuple[str, List[str], int]] = [
    ("Non-Live AST", ["simple_python", "simple_java", "simple_javascript", "multiple",
                      "parallel", "parallel_multiple"], 10),
    ("Live AST", ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"], 10),
    ("Irrelevance", ["irrelevance", "live_irrelevance"], 10),
    ("Multi-Turn", MULTI_TURN_CATEGORIES, 30),
    ("Agentic", AGENTIC_CATEGORIES, 40),
]

DATASET_REVISION = (f"ShishirPatil/gorilla@{BFCL_EVAL_COMMIT} "
                    "(berkeley-function-call-leaderboard/bfcl_eval/data, BFCL_v4_*.json)")

#: git blob SHA-1 (`git hash-object`) of every single-turn data file at BFCL_EVAL_COMMIT, relative
#: to bfcl_eval/data/. A cached or downloaded file must match before it is read, so an offline copy
#: from another bfcl-eval release can never be scored under this revision.
_PINNED_BLOBS = {
    "BFCL_v4_simple_python.json": "356a0db61e54eb1d92d8e10d014463cf2eb513cc",
    "possible_answer/BFCL_v4_simple_python.json": "268c55641a21b51fc15730ce39ce906444a935a1",
    "BFCL_v4_simple_java.json": "eff995f128c757f635154dfc4e63678d23dbe5b2",
    "possible_answer/BFCL_v4_simple_java.json": "bce120d34933ab61f868236d1cb56a088d9b0585",
    "BFCL_v4_simple_javascript.json": "0cf350ff1f6d61b05f1cb1b8105fa6b29edd4a27",
    "possible_answer/BFCL_v4_simple_javascript.json": "36375f0e31ca5598a1b807ac3af342fe9d787376",
    "BFCL_v4_multiple.json": "83643fa853e7a29a50e7303b0279284c37dd63c8",
    "possible_answer/BFCL_v4_multiple.json": "1f2f1fbfd9b7c8859ef5a92d6c18b0a6b2bc2663",
    "BFCL_v4_parallel.json": "04040475c8c77facf4f3e75e7079607c25c557d2",
    "possible_answer/BFCL_v4_parallel.json": "94c4be296d0fee3f7ae8863c4855783da563ca22",
    "BFCL_v4_parallel_multiple.json": "832039d6758e2569d65342b3fc9524924a03832a",
    "possible_answer/BFCL_v4_parallel_multiple.json": "52b010cb01334c98456cc3ec0886dd85d51ec284",
    "BFCL_v4_irrelevance.json": "0077629949f2d52734cef2f110b0d040d8c47554",
    "BFCL_v4_live_simple.json": "e8f4b8d33c07753affc77bfc4a646f94109de5af",
    "possible_answer/BFCL_v4_live_simple.json": "2c4778cee08e5ce7b236861de9a927d3ea7cde1f",
    "BFCL_v4_live_multiple.json": "b7c108f2edc86563fd3e1363e5f6fd961452d142",
    "possible_answer/BFCL_v4_live_multiple.json": "b2f4c71547d3a534e2e40231db14943bfa81cc74",
    "BFCL_v4_live_parallel.json": "fb88e7e7f12cb0e557750a043833bab87c6f98b1",
    "possible_answer/BFCL_v4_live_parallel.json": "2b62cca73e4b6877e2026934e667d0781d9978e9",
    "BFCL_v4_live_parallel_multiple.json": "54997d2e1fa4061d5701f6f42079c95102f8c09e",
    "possible_answer/BFCL_v4_live_parallel_multiple.json": "df36b6020f21ef6f38ebb4a8c6e90206915985af",
    "BFCL_v4_live_irrelevance.json": "547f60220b546ea5c864815dbb310b2f46ec0f43",
    "BFCL_v4_live_relevance.json": "1b0190995f4554b980da2dd7ab15195f4e047e62",
}


def resolve_categories(spec: Optional[str]) -> List[str]:
    """Categories named by `--eval-categories` (single names or BFCL's group names)."""
    if not spec:
        return list(SUPPORTED_BFCL_CATEGORIES)
    out: List[str] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        token = _RENAMED_CATEGORIES.get(token, token)
        if token in _CATEGORY_GROUPS:
            chosen = _CATEGORY_GROUPS[token]
        elif token in SUPPORTED_BFCL_CATEGORIES:
            chosen = [token]
        else:
            raise ValueError(
                f"bfcl: unknown category {token!r}. Supported: {', '.join(SUPPORTED_BFCL_CATEGORIES)} "
                f"(or {', '.join(_CATEGORY_GROUPS)}).")
        out.extend(c for c in chosen if c not in out)
    return out


def _cache_path(rel: str) -> Path:
    return Path(gbench_cache_dir()) / "bfcl_data" / BFCL_EVAL_COMMIT / rel


def _read_jsonl(text: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _data_file(rel: str, eval_name: str) -> List[Dict[str, Any]]:
    """One BFCL data file (JSON lines) at the pinned commit, downloaded once and cached.

    The file must match its pinned git blob id. A cached copy that does not is fetched again;
    a download that fails or does not match stops the suite (infra_required)."""
    path = _cache_path(rel)
    blob = _PINNED_BLOBS[rel]
    stale = ""
    if path.exists():
        data = path.read_bytes()
        if _git_blob_sha1(data) == blob:
            return _read_jsonl(data.decode("utf-8"))
        stale = f"; the cached copy at {path} does not match the pinned file"
    url = f"{BFCL_DATA_URL}/{rel}"
    err: Optional[Exception] = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gbench"})
            with urllib.request.urlopen(req, timeout=60) as res:
                data = res.read()
            break
        except Exception as e:
            err = e
    else:
        raise infra_required(eval_name, f"the BFCL data file {url} could not be downloaded "
                             f"({err}){stale}.", f"docs/evals/{eval_name}.md")
    if _git_blob_sha1(data) != blob:
        raise infra_required(eval_name, f"{url} does not match the pinned file (git blob {blob}).",
                             f"docs/evals/{eval_name}.md")
    rows = _read_jsonl(data.decode("utf-8"))   # parse before caching: never cache a bad file
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    os.replace(tmp.name, path)
    return rows


def load_bfcl_entries(categories: List[str], eval_name: str) -> List[Dict[str, Any]]:
    """Single-turn test entries with their ground truth: {id, category, question, function,
    ground_truth}."""
    entries: List[Dict[str, Any]] = []
    for cat in categories:
        rows = _data_file(f"BFCL_v4_{cat}.json", eval_name)
        answers: Optional[Dict[str, Any]] = None
        if "relevance" not in cat:            # (ir)relevance categories ship no possible answer
            answers = {a["id"]: a["ground_truth"]
                       for a in _data_file(f"possible_answer/BFCL_v4_{cat}.json", eval_name)}
        for row in rows:
            if answers is not None and row["id"] not in answers:
                raise RuntimeError(f"{eval_name}: BFCL entry {row['id']} has no possible answer")
            entries.append({"id": row["id"], "category": cat, "question": row["question"],
                            "function": row["function"],
                            "ground_truth": answers[row["id"]] if answers is not None else []})
    logger.info("Loaded %d BFCL entries (%s) from %s", len(entries), ", ".join(categories),
                DATASET_REVISION)
    return entries


def bfcl_request(entry: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(messages, tools) exactly as BFCL's FC handler sends a single-turn entry."""
    messages = copy.deepcopy(entry["question"][0])
    tools = bfcl_convert_to_tools(bfcl_request_functions(entry["function"], entry["category"]))
    return messages, tools


def bfcl_gold(entry: Dict[str, Any]) -> Dict[str, Any]:
    # The function docs stay out of the gold (they are large and already on the request);
    # bfcl_batch_scorer looks them up by id.
    return {"id": entry["id"], "category": entry["category"], "ground_truth": entry["ground_truth"]}


def _load_bfcl_samples(categories: List[str]) -> Tuple[List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
                                                       Dict[str, List[Dict[str, Any]]]]:
    """Samples for run_eval_suite plus the original function docs by entry id."""
    samples = []
    functions_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for entry in load_bfcl_entries(categories, "bfcl"):
        messages, tools = bfcl_request(entry)
        meta: Dict[str, Any] = {"category": entry["category"]}
        if tools:
            meta["tools"] = tools
        samples.append((messages, bfcl_gold(entry), meta))
        functions_by_id[entry["id"]] = entry["function"]
    return samples, functions_by_id


_PY_TO_BFCL_TYPE = {str: "string", bool: "boolean", int: "integer", float: "float",
                    list: "array", dict: "dict"}


def _docs_from_possible_answer(ground_truth: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Minimal function docs rebuilt from a possible answer, for golds recorded without their
    docs (older traces, unit tests): a parameter is required unless "" is among its accepted
    values, and its type is that of its first non-empty accepted value."""
    def bfcl_type(value: Any) -> str:
        return _PY_TO_BFCL_TYPE.get(type(value), "string")

    docs = []
    for call in ground_truth:
        for name, params in call.items():
            props: Dict[str, Any] = {}
            required = []
            for pname, accepted in params.items():
                accepted = accepted if isinstance(accepted, list) else [accepted]
                sample = next((a for a in accepted if a != ""), "")
                spec: Dict[str, Any] = {"type": bfcl_type(sample), "description": ""}
                if spec["type"] == "array":
                    spec["items"] = {"type": bfcl_type(sample[0]) if sample else "string"}
                props[pname] = spec
                if "" not in accepted:
                    required.append(pname)
            docs.append({"name": name, "description": "",
                         "parameters": {"type": "dict", "properties": props, "required": required}})
    return docs


def bfcl_verdict(response_text: str, gold: Any, tool_calls: Any = None) -> Dict[str, Any]:
    """fc_common.bfcl_score (the ported rule) for a sample gold.

    The suites do not score with it: bfcl_batch_scorer runs bfcl_eval's own eval_runner in the
    harness image (single_turn_verdicts). The port is kept as a reference implementation, checked
    against eval_runner in the tests.

    `gold` is {"category", "ground_truth", "function"} (the original docs). A bare
    possible_answer list is scored with docs rebuilt from it, as simple/parallel by its
    length; an empty one is an irrelevance item.
    """
    if isinstance(gold, dict) and "category" in gold:
        if gold.get("function") is None:
            raise ValueError("bfcl: scoring needs the entry's function docs under gold['function']")
        return bfcl_score(gold["category"], gold["function"], gold.get("ground_truth") or [],
                          tool_calls=tool_calls, response_text=response_text)
    ground_truth = gold if isinstance(gold, list) else ([gold] if isinstance(gold, dict) else [])
    category = ("irrelevance" if not ground_truth
                else "parallel" if len(ground_truth) > 1 else "simple")
    return bfcl_score(category, _docs_from_possible_answer(ground_truth), ground_truth,
                      tool_calls=tool_calls, response_text=response_text)


def _eval_bfcl(response_text: str, gold: Any, tool_calls: Any = None) -> bool:
    """The ported rule's verdict for one reply, scored off the STRUCTURED tool calls base.py
    captured (see bfcl_verdict for the accepted gold shapes)."""
    return bool(bfcl_verdict(response_text, gold, tool_calls)["valid"])


# --------------------------------------------------------------------------- #
# Single-turn scoring: bfcl_eval's eval_runner, in the harness image
# --------------------------------------------------------------------------- #

#: git blob ids at BFCL_EVAL_COMMIT of the bfcl_eval files the single-turn checker imports or runs,
#: relative to the installed package (checked in the image before anything is generated).
_ST_PINNED_BLOBS: Dict[str, str] = {
    "eval_checker/eval_runner.py": "2d5c41aac41cd2f1caedf741fb01dfdfd57b1990",
    "eval_checker/ast_eval/ast_checker.py": "160fd75cfc8b2b7ef9f83a244f81da0fe0244e39",
    "eval_checker/ast_eval/type_convertor/java_type_converter.py":
        "7d2b141dfcb8e527df6424893e60a833eb69f3e1",
    "eval_checker/ast_eval/type_convertor/js_type_converter.py":
        "4205b92e05af83d2e61a7bf4bd2530718b57acee",
    "model_handler/api_inference/openai_completion.py": "357584f8dbfb1161c4f5b0aa4a3b024dbe38120b",
    "model_handler/base_handler.py": "a1025e9cbf5d86bcdc62cd0564006284eaffb00a",
    "model_handler/utils.py": "7abd9a242f27c72399dbdbb0f2749b0464e1a0b8",
    "constants/enums.py": "7400315f045419d4a8a28525f3999c1844f5e9af",
    "constants/type_mappings.py": "fdbdde48402f8f2812b146f9b7341b6aae9e55b8",
    "constants/category_mapping.py": "492142b986438fe8a5a6c38931e567bcd45c0d9b",
    "constants/eval_config.py": "813d6de70333b6137f5fb62e116956b13bb339d3",
    "constants/default_prompts.py": "1f5e9e1e8119b2277bee2198214ee9d6aaf6f6e2",
    "utils.py": "552a10f36ec10840773f75e50fed01ca19b4bfc6",
}

#: The model name the single-turn checker registers: an OpenAI chat-completions FC model, as
#: bfcl_v4_agentic._register_model registers the served model (no '_', so eval_runner's
#: name escaping maps it to itself).
_CHECKER_MODEL = "gbench-openai-completions-fc"


def upstream_model_result(response_text: Optional[str], tool_calls: Any) -> Any:
    """What OpenAICompletionsHandler._parse_query_response_FC records as the model result of one
    reply: [{name: arguments}] for its tool calls, else the message content."""
    if tool_calls:
        return [{(tc.get("function") or {}).get("name"): (tc.get("function") or {}).get("arguments")}
                for tc in tool_calls]
    return response_text


def single_turn_verdicts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """bfcl_eval's verdict for each {id, category, function, ground_truth, model_result}: the
    entry check `bfcl evaluate` runs (eval_runner.relevance_file_runner / ast_file_runner per
    entry, with their language and return format), for an OpenAI chat-completions FC model.
    Runs where bfcl_eval is importable: in the harness image (check_single_turn)."""
    _set_project_root()
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.eval_runner import (_evaluate_single_ast_entry,
                                                    _evaluate_single_relevance_entry)
    from bfcl_eval.utils import is_java, is_js, is_relevance_or_irrelevance
    from .bfcl_v4_agentic import _patched_openai_handler, _register_model

    _register_model(_CHECKER_MODEL)
    handler_cls = _patched_openai_handler()

    class _Checker(handler_cls):
        def _build_client_kwargs(self):
            # decode_ast never sends a request; the client needs a key to be built.
            return {"api_key": "unused", "base_url": "http://127.0.0.1:9/v1"}

    handler = _Checker(model_name=_CHECKER_MODEL, temperature=0.001,
                       registry_name=_CHECKER_MODEL, is_fc_model=True)
    out = []
    for it in items:
        cat = it["category"]
        prompt_entry = {"id": it["id"], "function": it["function"]}
        if is_relevance_or_irrelevance(cat):
            verdict = _evaluate_single_relevance_entry(handler, it["id"], it["model_result"],
                                                       prompt_entry, _CHECKER_MODEL, cat)
        else:
            if is_java(cat):
                language, return_format = Language.JAVA, ReturnFormat.JAVA
            elif is_js(cat):
                language, return_format = Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
            else:
                language, return_format = Language.PYTHON, ReturnFormat.PYTHON
            verdict = _evaluate_single_ast_entry(handler, it["id"], it["model_result"],
                                                 it["ground_truth"], prompt_entry, _CHECKER_MODEL,
                                                 cat, language=language,
                                                 return_format=return_format,
                                                 has_tool_call_tag=False)
        error = verdict.get("error")
        out.append({"valid": bool(verdict.get("valid")), "error_type": verdict.get("error_type"),
                    "error": [str(e) for e in error] if isinstance(error, list)
                    else ([str(error)] if error else [])})
    return out


def check_single_turn(items: List[Dict[str, Any]], eval_name: str = "bfcl") -> List[Dict[str, Any]]:
    """single_turn_verdicts in the harness image, bounded by GBENCH_BFCL_TIMEOUT_S when it is set
    (bfcl_v4_agentic.part_timeout; raises HarnessTimeout past it)."""
    return _agentic.run_in_image(eval_name, "bfcl", "single_turn_verdicts", {"items": items},
                                 timeout=part_timeout(), part="the single-turn checker")


def bfcl_batch_scorer(functions_by_id: Dict[str, List[Dict[str, Any]]], eval_name: str = "bfcl",
                      checker=None):
    """async_eval_fn scoring every trace with BFCL's own checker (check_single_turn, or `checker`,
    a function with single_turn_verdicts' contract).

    Batch scoring (rather than a per-sample eval_fn) is what lets an EMPTY reply be scored:
    base marks it wrong without calling eval_fn, but for irrelevance it is the right answer.
    Such a trace is marked `empty_is_answer`, so base does not count it as a failed request.
    A failed request (no response at all) is not checked: base leaves it out of the score as a
    harness error.

    A checker stopped by GBENCH_BFCL_TIMEOUT_S (HarnessTimeout) leaves the replies unscored: each
    is marked HARNESS_ERROR and `scoring_excluded` (out of every accuracy, as it was never
    checked), and the part is recorded in the scorer's `harness_errors` list for the result.
    """
    check = checker or (lambda items: check_single_turn(items, eval_name))
    harness_errors: List[Dict[str, Any]] = []

    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        scored = [t for t in sample_traces if t.get("response_text") is not None]
        items = []
        for t in scored:
            gold = t.get("gold_answer") or {}
            items.append({"id": gold["id"], "category": gold["category"],
                          "function": functions_by_id[gold["id"]],
                          "ground_truth": gold.get("ground_truth") or [],
                          "model_result": upstream_model_result(t["response_text"],
                                                                t.get("tool_calls"))})
        try:
            verdicts = check(items) if items else []
        except HarnessTimeout as e:
            logger.error("[%s] %s: %d replies are left unscored", eval_name, e, len(scored))
            harness_errors.append(e.record(replies_unscored=len(scored)))
            for t in scored:
                t.update(is_correct=False, status="HARNESS_ERROR", scoring_excluded=True,
                         harness_error=str(e), status_before_harness_error=t.get("status"))
            return
        if len(verdicts) != len(items):
            raise RuntimeError(f"{eval_name}: the BFCL checker returned {len(verdicts)} verdicts "
                               f"for {len(items)} replies")
        for t, verdict in zip(scored, verdicts):
            t["is_correct"] = bool(verdict["valid"])
            if verdict["valid"] and not str(t["response_text"]).strip() and not t.get("tool_calls"):
                t["empty_is_answer"] = True
            if not verdict["valid"]:
                t["bfcl_error_type"] = verdict.get("error_type")
                t["bfcl_error"] = verdict.get("error")
            if t.get("status") == "PENDING_JUDGE":
                t["status"] = "TRUNCATED" if t.get("health") in ("truncated", "looping") else "OK"
    _score.harness_errors = harness_errors          # type: ignore[attr-defined]
    return _score


def harness_unscored(traces: List[Dict[str, Any]]) -> int:
    """Replies a stopped checker left unscored that base did not already count as failed (an
    empty reply is counted under failed_requests before the checker runs; one with the answer
    inside the thinking channel is not, base's ANSWER_IN_THINKING_POLICY)."""
    return sum(1 for t in traces if t.get("status") == "HARNESS_ERROR"
               and (t.get("status_before_harness_error") != "EMPTY_RESPONSE"
                    or t.get("health") == "empty_reasoned"))


# --------------------------------------------------------------------------- #
# Aggregation: eval_runner_helper.generate_leaderboard_csv
# --------------------------------------------------------------------------- #

def _cat_counts(category_accuracy: Dict[str, Any], cat: str) -> Optional[Tuple[int, int]]:
    stats = category_accuracy.get(cat)
    if not stats or not stats.get("total"):
        return None
    return int(stats["correct"]), int(stats["total"])


def _unweighted(parts: List[Optional[Tuple[float, int]]]) -> Optional[Tuple[float, int]]:
    """calculate_unweighted_accuracy; None when a component was not run."""
    if any(p is None for p in parts):
        return None
    return sum(p[0] for p in parts) / len(parts), sum(p[1] for p in parts)


def _weighted(parts: List[Optional[Tuple[float, int]]]) -> Optional[Tuple[float, int]]:
    """calculate_weighted_accuracy (by entry count); None when a component was not run."""
    if any(p is None for p in parts):
        return None
    total = sum(p[1] for p in parts)
    return sum(p[0] * p[1] for p in parts) / total, total


def _percentage_weighted(parts: List[Optional[Tuple[float, int]]],
                         weights: List[int]) -> Optional[Tuple[float, int]]:
    """calculate_percentage_weighted_accuracy; None when a component was not run."""
    if any(p is None for p in parts):
        return None
    norm = [w / sum(weights) for w in weights]
    return sum(p[0] * w for p, w in zip(parts, norm)), sum(p[1] for p in parts)


def _bfcl_columns(category_accuracy: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    """BFCL's score tables as fractions: the rows of data_non_live.csv, data_live.csv,
    data_multi_turn.csv and data_agentic.csv, and the accuracy columns of data_overall.csv.
    None marks a column whose categories did not all run (BFCL shows N/A there)."""
    def acc(cat: str) -> Optional[Tuple[float, int]]:
        counts = _cat_counts(category_accuracy, cat)
        return (counts[0] / counts[1], counts[1]) if counts else None

    a = {cat: acc(cat) for cat in ALL_SCORING_CATEGORIES}
    simple_ast = _unweighted([a["simple_python"], a["simple_java"], a["simple_javascript"]])
    non_live_ast = _unweighted([simple_ast, a["multiple"], a["parallel"], a["parallel_multiple"]])
    live_ast = _weighted([a["live_simple"], a["live_multiple"], a["live_parallel"],
                          a["live_parallel_multiple"]])
    multi_turn = _unweighted([a[c] for c in MULTI_TURN_CATEGORIES])
    web_search = _unweighted([a["web_search_base"], a["web_search_no_snippet"]])
    memory = _unweighted([a["memory_kv"], a["memory_vector"], a["memory_rec_sum"]])
    agentic = _unweighted([web_search, memory])
    total_irrelevance = _unweighted([a["irrelevance"], a["live_irrelevance"]])
    overall = _percentage_weighted([non_live_ast, live_ast, total_irrelevance, multi_turn, agentic],
                                   [w for _, _, w in OVERALL_SEGMENTS])

    def frac(value: Optional[Tuple[float, int]]) -> Optional[float]:
        return None if value is None else value[0]

    return {
        "overall": {
            "Overall Acc": frac(overall),
            "Non-Live AST Acc": frac(non_live_ast),
            "Non-Live Simple AST": frac(simple_ast),
            "Non-Live Multiple AST": frac(a["multiple"]),
            "Non-Live Parallel AST": frac(a["parallel"]),
            "Non-Live Parallel Multiple AST": frac(a["parallel_multiple"]),
            "Live Acc": frac(live_ast),
            "Live Simple AST": frac(a["live_simple"]),
            "Live Multiple AST": frac(a["live_multiple"]),
            "Live Parallel AST": frac(a["live_parallel"]),
            "Live Parallel Multiple AST": frac(a["live_parallel_multiple"]),
            "Multi Turn Acc": frac(multi_turn),
            "Multi Turn Base": frac(a["multi_turn_base"]),
            "Multi Turn Miss Func": frac(a["multi_turn_miss_func"]),
            "Multi Turn Miss Param": frac(a["multi_turn_miss_param"]),
            "Multi Turn Long Context": frac(a["multi_turn_long_context"]),
            "Web Search Acc": frac(web_search),
            "Web Search Base": frac(a["web_search_base"]),
            "Web Search No Snippet": frac(a["web_search_no_snippet"]),
            "Memory Acc": frac(memory),
            "Memory KV": frac(a["memory_kv"]),
            "Memory Vector": frac(a["memory_vector"]),
            "Memory Recursive Summarization": frac(a["memory_rec_sum"]),
            "Relevance Detection": frac(a["live_relevance"]),
            "Irrelevance Detection": frac(total_irrelevance),
        },
        "non_live": {
            "Non-Live Overall Acc": frac(non_live_ast),
            "AST Summary": frac(non_live_ast),
            "Simple AST": frac(simple_ast),
            "Python Simple AST": frac(a["simple_python"]),
            "Java Simple AST": frac(a["simple_java"]),
            "JavaScript Simple AST": frac(a["simple_javascript"]),
            "Multiple AST": frac(a["multiple"]),
            "Parallel AST": frac(a["parallel"]),
            "Parallel Multiple AST": frac(a["parallel_multiple"]),
            "Irrelevance Detection": frac(a["irrelevance"]),
        },
        "live": {
            "Live Overall Acc": frac(live_ast),
            "AST Summary": frac(live_ast),
            "Python Simple AST": frac(a["live_simple"]),
            "Python Multiple AST": frac(a["live_multiple"]),
            "Python Parallel AST": frac(a["live_parallel"]),
            "Python Parallel Multiple AST": frac(a["live_parallel_multiple"]),
            "Irrelevance Detection": frac(a["live_irrelevance"]),
            "Relevance Detection": frac(a["live_relevance"]),
        },
        "multi_turn": {
            "Multi Turn Overall Acc": frac(multi_turn),
            "Base": frac(a["multi_turn_base"]),
            "Miss Func": frac(a["multi_turn_miss_func"]),
            "Miss Param": frac(a["multi_turn_miss_param"]),
            "Long Context": frac(a["multi_turn_long_context"]),
        },
        "agentic": {
            "Agentic Overall Acc": frac(agentic),
            "Web Search Summary": frac(web_search),
            "Web Search Base": frac(a["web_search_base"]),
            "Web Search No Snippet": frac(a["web_search_no_snippet"]),
            "Memory Summary": frac(memory),
            "Memory KV": frac(a["memory_kv"]),
            "Memory Vector": frac(a["memory_vector"]),
            "Memory Recursive Summarization": frac(a["memory_rec_sum"]),
        },
    }


def bfcl_summary(category_accuracy: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    """_bfcl_columns in percent, rounded as BFCL's CSVs are (2 decimals)."""
    return {table: {col: None if v is None else round(v * 100.0, 2) for col, v in cols.items()}
            for table, cols in _bfcl_columns(category_accuracy).items()}


def overall_score(category_accuracy: Dict[str, Any]) -> Optional[float]:
    """BFCL Overall Acc in percent (10/10/10/30/40), or None unless every weighed category ran."""
    return bfcl_summary(category_accuracy)["overall"]["Overall Acc"]


def pooled(category_accuracy: Dict[str, Any]) -> Optional[float]:
    """Plain correct/total over the categories in `category_accuracy`, in percent."""
    total = sum(int(v.get("total") or 0) for v in category_accuracy.values())
    if not total:
        return None
    return round(sum(int(v.get("correct") or 0) for v in category_accuracy.values()) / total * 100.0, 2)


def run_notes(limit: Optional[int], categories_subset: bool = False) -> List[str]:
    """Run-shaping choices that make a result a subset of the benchmark."""
    notes = []
    if categories_subset:
        notes.append("category subset (--eval-categories)")
    if limit:
        notes.append(f"--eval-limit {limit} subset")
    if os.environ.get("GBENCH_SHARD"):
        notes.append(f"shard {os.environ['GBENCH_SHARD']}")
    return notes


# --------------------------------------------------------------------------- #
# Multi-turn: bfcl_eval's own inference loop, executable backends and checker
# --------------------------------------------------------------------------- #

#: git blob ids at BFCL_EVAL_COMMIT of every bfcl_eval file the multi-turn categories read or run,
#: relative to the installed package (38 files). BFCL_EVAL_RELEASE (bfcl_v4_agentic.py) matches
#: them all; any other copy that differs in one of them is refused.
_MT_PINNED_BLOBS: Dict[str, Tuple[str, ...]] = {
    "data/BFCL_v4_multi_turn_base.json": ("656fbd5e56969373353cd58b880234a6cc4edc99",),
    "data/possible_answer/BFCL_v4_multi_turn_base.json": ("3ee85fcd5e8cccca24e2347e464b58fd1b9f02a1",),
    "data/BFCL_v4_multi_turn_miss_func.json": ("28e7987841719aec4e4f3f3f4b6c2f1915c831f3",),
    "data/possible_answer/BFCL_v4_multi_turn_miss_func.json": ("ec10a3566a613ddce8e05438425d40f6ec15fbfc",),
    "data/BFCL_v4_multi_turn_miss_param.json": ("2e87caf87fb48f7807a18610311bdfa1138d0599",),
    "data/possible_answer/BFCL_v4_multi_turn_miss_param.json": ("94edb50ede0de76f35e937d4b96128857357d95b",),
    "data/BFCL_v4_multi_turn_long_context.json": ("bc6584c32771f9f14168a2802e6b454e480ace4e",),
    "data/possible_answer/BFCL_v4_multi_turn_long_context.json": ("ec7f2711df89dc3ad667351e5349906c0a732912",),
    "data/multi_turn_func_doc/gorilla_file_system.json": ("b9a6d2d314febe8ccf201887ef87cfe081ffbf4e",),
    "data/multi_turn_func_doc/math_api.json": ("52a911035a2b71390027e02af70c1abb904c71b3",),
    "data/multi_turn_func_doc/message_api.json": ("a178734ad47f269571c877e6e433b26171516024",),
    "data/multi_turn_func_doc/posting_api.json": ("2a8737a376640ba304c313d23d63049927f823eb",),
    "data/multi_turn_func_doc/ticket_api.json": ("f8281657b64c4b6410a8590cc82b9584bc39729f",),
    "data/multi_turn_func_doc/trading_bot.json": ("7dce37326156f65efc55df30db26064d68e8dc8c",),
    "data/multi_turn_func_doc/travel_booking.json": ("95008945fa5a9bc104a77647e4990d1e36ce7ec5",),
    "data/multi_turn_func_doc/vehicle_control.json": ("7139dfc3bf2646be8f71ac383760f28899b17f85",),
    "eval_checker/multi_turn_eval/func_source_code/gorilla_file_system.py": ("6d269dd5fcf809d514eb531263f1e0e14246993c",),
    "eval_checker/multi_turn_eval/func_source_code/long_context.py": ("938105bea5a07e46ee15615ab11d1c0988124b5d",),
    "eval_checker/multi_turn_eval/func_source_code/math_api.py": ("e1b43b3817a3d8cbedc011bb36d4d9253fd6a05a",),
    "eval_checker/multi_turn_eval/func_source_code/message_api.py": ("3fb461dfbf60ecb9b6cc832162c97368d69ec1ae",),
    "eval_checker/multi_turn_eval/func_source_code/posting_api.py": ("ccd7566181bc36889d235c2ad62a064a6e16ba00",),
    "eval_checker/multi_turn_eval/func_source_code/ticket_api.py": ("6ce72a1cd23a11d814841b46a39d173ccc4e47bb",),
    "eval_checker/multi_turn_eval/func_source_code/trading_bot.py": ("e745788b15d729194ef39844f5eb1a3a67a4d55a",),
    "eval_checker/multi_turn_eval/func_source_code/travel_booking.py": ("f357ccdabdedc5fed0019392101a7de1f524ef77",),
    "eval_checker/multi_turn_eval/func_source_code/vehicle_control.py": ("ab7ef900d1a4d0120638b79c95441710eb8f5282",),
    "eval_checker/multi_turn_eval/multi_turn_checker.py": ("34b36715cdfaa4464ba1be137af4d30e85da8efb",),
    "eval_checker/multi_turn_eval/multi_turn_utils.py": ("dfcf6512a2eeb592820f3a0fa0fb564a117facc4",),
    "eval_checker/eval_runner.py": ("2d5c41aac41cd2f1caedf741fb01dfdfd57b1990",),
    "model_handler/base_handler.py": ("a1025e9cbf5d86bcdc62cd0564006284eaffb00a",),
    "model_handler/api_inference/openai_completion.py": ("357584f8dbfb1161c4f5b0aa4a3b024dbe38120b",),
    "model_handler/utils.py": ("7abd9a242f27c72399dbdbb0f2749b0464e1a0b8",),
    "constants/default_prompts.py": ("1f5e9e1e8119b2277bee2198214ee9d6aaf6f6e2",),
    "constants/enums.py": ("7400315f045419d4a8a28525f3999c1844f5e9af",),
    "constants/executable_backend_config.py": ("03a8a43fd6b75dd0961504edc6704a728b74db5b",),
    "constants/type_mappings.py": ("fdbdde48402f8f2812b146f9b7341b6aae9e55b8",),
    "constants/category_mapping.py": ("492142b986438fe8a5a6c38931e567bcd45c0d9b",),
    "constants/eval_config.py": ("813d6de70333b6137f5fb62e116956b13bb339d3",),
    "utils.py": ("552a10f36ec10840773f75e50fed01ca19b4bfc6",),
}

def _default_project_root() -> str:
    """$GBENCH_CACHE_DIR/bfcl (default ~/.cache/gbench/bfcl), shared with bfcl_v4_agentic."""
    return os.path.join(gbench_cache_dir(), "bfcl")


def _files_differing(pins: Dict[str, Any]) -> Tuple[str, Path, List[str]]:
    """(installed release, package root, files that differ from `pins`) for the importable
    bfcl_eval. A pin is one blob id or a tuple of accepted ones."""
    import bfcl_eval
    try:
        version = importlib.metadata.version("bfcl-eval")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    root = Path(bfcl_eval.__file__).resolve().parent
    differ = []
    for rel, blobs in pins.items():
        accepted = (blobs,) if isinstance(blobs, str) else tuple(blobs)
        path = root / rel
        if not path.is_file() or _git_blob_sha1(path.read_bytes()) not in accepted:
            differ.append(rel)
    return version, root, differ


def _differ_text(what: str, version: str, root: Path, differ: List[str]) -> str:
    return (f"the bfcl-eval {version} in the harness image ({root}) differs from "
            f"gorilla@{BFCL_EVAL_COMMIT} in {len(differ)} file(s) {what} "
            f"({', '.join(differ[:4])}{', ...' if len(differ) > 4 else ''}); the image must hold "
            f"bfcl-eval=={BFCL_EVAL_RELEASE}: `{image_build_command()}`")


def check_multi_turn_harness(eval_name: str = "bfcl") -> str:
    """The installed bfcl-eval release, after checking every file the multi-turn path uses
    against BFCL_EVAL_COMMIT. Runs where bfcl_eval is importable (the harness image); raises
    infra_required when the package is missing or differs."""
    try:
        import bfcl_eval  # noqa: F401
    except ImportError:
        raise infra_required(
            eval_name, "the multi-turn categories run in BFCL's own harness (its executable "
            f"backends, inference loop and multi-turn checker), and bfcl-eval is not importable "
            f"here; gbench runs bfcl-eval=={BFCL_EVAL_RELEASE} in the local image "
            f"{bfcl_image()}: `{image_build_command()}`.", DOCS_URL)
    version, root, differ = _files_differing(_MT_PINNED_BLOBS)
    if differ:
        raise infra_required(eval_name, _differ_text("the multi-turn categories use", version,
                                                     root, differ) + ".", DOCS_URL)
    return version


def image_probe(multi_turn_categories: Optional[List[str]] = None,
                single_turn: bool = False) -> Dict[str, Any]:
    """Inside the harness image, before anything is generated: the installed release, whether
    the files the single-turn checker and the multi-turn path use match the pinned commit, and
    the (id, category) of every multi-turn entry of `multi_turn_categories`."""
    try:
        release: Optional[str] = importlib.metadata.version("bfcl-eval")
    except importlib.metadata.PackageNotFoundError:
        release = None
    out: Dict[str, Any] = {"release": release, "single_turn_mismatch": None,
                           "multi_turn_mismatch": None, "multi_turn_index": []}
    if single_turn:
        version, root, differ = _files_differing(_ST_PINNED_BLOBS)
        if differ:
            out["single_turn_mismatch"] = _differ_text("the single-turn checker uses", version,
                                                       root, differ)
    if multi_turn_categories:
        try:
            check_multi_turn_harness()
        except RuntimeError as e:
            out["multi_turn_mismatch"] = str(e)
            return out
        out["multi_turn_index"] = [[it["id"], it["category"]]
                                   for it in load_multi_turn_entries(list(multi_turn_categories))]
    return out


def _harness_probe(eval_name: str, multi_turn_categories: List[str],
                   single_turn: bool) -> Tuple[str, List[Dict[str, Any]]]:
    """(release, [{id, category}] of the multi-turn entries) from the harness image, after the
    image and every pinned file the run needs are checked; raises infra_required otherwise."""
    problem = _agentic.image_problem()
    if problem:
        what = ("the single-turn checker and the multi-turn categories" if single_turn and
                multi_turn_categories else
                "the multi-turn categories" if multi_turn_categories else "the single-turn checker")
        raise infra_required(eval_name, f"{what} run in BFCL's own harness: {problem}", DOCS_URL)
    found = _agentic.run_in_image(eval_name, "bfcl", "image_probe",
                         {"multi_turn_categories": list(multi_turn_categories),
                          "single_turn": bool(single_turn)}, timeout=1800)
    if found.get("release") != BFCL_EVAL_RELEASE:
        raise infra_required(eval_name, f"the harness image {bfcl_image()} holds bfcl-eval "
                             f"{found.get('release') or 'none'}, not {BFCL_EVAL_RELEASE}: "
                             f"`{image_build_command()}`.", DOCS_URL)
    for key in ("single_turn_mismatch", "multi_turn_mismatch"):
        if found.get(key):
            message = str(found[key])
            if message.startswith("[INFRA REQUIRED]"):
                raise RuntimeError(message)
            raise infra_required(eval_name, message + ".", DOCS_URL)
    return found["release"], [{"id": i, "category": c} for i, c in found.get("multi_turn_index") or []]


def _set_project_root() -> None:
    """bfcl_eval creates its result/score/lock directories when eval_config is first imported,
    under BFCL_PROJECT_ROOT (default: next to site-packages). Import it with the variable pointing
    at gbench's cache, then unset the variable again: eval_config reads it only once, and a
    leftover value would read as the deprecated alias of GBENCH_BFCL_PROJECT_ROOT elsewhere."""
    if "bfcl_eval.constants.eval_config" in sys.modules or os.environ.get("BFCL_PROJECT_ROOT"):
        return
    os.environ["BFCL_PROJECT_ROOT"] = (suite_env("GBENCH_BFCL_PROJECT_ROOT", default=None)
                                       or _default_project_root())
    try:
        import bfcl_eval.constants.eval_config  # noqa: F401
    finally:
        os.environ.pop("BFCL_PROJECT_ROOT", None)


def load_multi_turn_entries(categories: List[str]) -> List[Dict[str, Any]]:
    """{id, category, entry, eval_entry, ground_truth} per multi-turn entry, loaded by bfcl_eval.

    `entry` is what BFCL's generation step hands the model handler (load_dataset_entry with the
    language hint); `eval_entry` and `ground_truth` are what its evaluation step reads."""
    _set_project_root()
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    out = []
    for cat in categories:
        entries = load_dataset_entry(cat)
        evals = {e["id"]: e for e in load_dataset_entry(cat, include_prereq=False,
                                                        include_language_specific_hint=False)}
        truth = {g["id"]: g["ground_truth"] for g in load_ground_truth_entry(cat)}
        for e in entries:
            if e["id"] not in truth or e["id"] not in evals:
                raise RuntimeError(f"bfcl: multi-turn entry {e['id']} has no possible answer")
            out.append({"id": e["id"], "category": cat, "entry": e, "eval_entry": evals[e["id"]],
                        "ground_truth": truth[e["id"]]})
    logger.info("Loaded %d BFCL multi-turn entries (%s)", len(out), ", ".join(categories))
    return out


#: The AST nodes a call's arguments may contain: what convert_to_function_call's repr() of a JSON
#: value produces (literals, containers, a sign, and bare names such as `nan`).
_LITERAL_NODES =(ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.UnaryOp, ast.USub,
                  ast.UAdd, ast.Name, ast.Load, ast.keyword, ast.Starred)


def _refused_call(tree: ast.Expression, namespace: Dict[str, Any], local_names: set) -> Optional[str]:
    """Why an executable call string must not be evaluated, or None when it is a plain call.

    convert_to_function_call renders every structured call as `name(k=<repr of a JSON value>)`,
    and execute_multi_turn_func_call prefixes backend method names with their instance. A
    benign call is therefore `<instance>.<method>(literals)` or an unknown name (which raises
    NameError, as upstream does); anything else would run code the model wrote."""
    node = tree.body
    if not isinstance(node, ast.Call):
        return ast.unparse(node)[:80]
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in namespace or func.id in local_names or hasattr(builtins, func.id):
            return func.id
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner = func.value.id
        if (not owner.endswith("_instance") or owner not in namespace
                or func.attr.startswith("_")):
            return ast.unparse(func)
    else:
        return ast.unparse(func)[:80]
    for arg in list(node.args) + list(node.keywords):
        if not all(isinstance(sub, _LITERAL_NODES) for sub in ast.walk(arg)):
            return ast.unparse(func)
    return None


def _upstream_call_name(source: str) -> str:
    """The name execute_multi_turn_func_call puts in its "Function call X is not allowed." error:
    the text before the first '(' and, for a method call, the part after the instance name."""
    name = source.split("(")[0]
    if "." in name:
        name = name.split(".")[1]
    return name


class _ExecutionGuard:
    """Replaces `eval` inside bfcl_eval's multi_turn_utils: for the duration of the multi-turn
    part (_harness_patches), and for the whole harness child of a `bfcl` command of the agentic
    part (bfcl_v4_agentic._install_execution_guard).

    execute_multi_turn_func_call evaluates the model's call strings with `eval`. Every call a
    model can make through the tools it was given evaluates exactly as upstream (same result,
    same exception text); a call that would reach a Python builtin, a module global or injected
    code is refused with upstream's own "Function call X is not allowed." error instead, X named
    the way upstream names it (the bare method name). Each refusal is kept in `refused` and, with
    `on_refused`, handed to it (the call string, cut to 200 characters)."""

    def __init__(self, mtu: Any, on_refused: Optional[Any] = None):
        self.mtu = mtu
        self.local_names = set(mtu.execute_multi_turn_func_call.__code__.co_varnames)
        self.refused: List[str] = []
        self.on_refused = on_refused
        self._lock = threading.Lock()

    def __call__(self, source: str, *args: Any) -> Any:
        tree = compile(source, "<string>", "eval", ast.PyCF_ONLY_AST)
        namespace = vars(self.mtu)
        if _refused_call(tree, namespace, self.local_names) is not None:
            with self._lock:
                self.refused.append(source[:200])
            if self.on_refused is not None:
                try:
                    self.on_refused(source[:200])
                except Exception as e:              # noqa: BLE001 - never changes the answer
                    logger.warning("could not record an execution-guard refusal (%s: %s)",
                                   type(e).__name__, e)
            raise Exception(f"Function call {_upstream_call_name(source)} is not allowed.")
        return builtins.eval(compile(tree, "<string>", "eval"), namespace)


@contextlib.contextmanager
def _execution_guard(on_refused: Optional[Any] = None):
    """The execution guard in place of `eval` in bfcl_eval's multi_turn_utils for the block."""
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu
    guard = _ExecutionGuard(mtu, on_refused=on_refused)
    mtu.eval = guard
    try:
        yield guard
    finally:
        if "eval" in vars(mtu):
            delattr(mtu, "eval")


@contextlib.contextmanager
def _harness_patches():
    """The execution guard, and bfcl_eval's per-step progress prints silenced."""
    import bfcl_eval.model_handler.base_handler as base_handler
    with _execution_guard() as guard:
        base_handler.print = lambda *a, **k: None
        try:
            yield guard
        finally:
            if "print" in vars(base_handler):
                delattr(base_handler, "print")


def _instance_prefix(name: str) -> str:
    return re.sub(r"[-./:]", "_", name)


#: What the multi-turn checker appends to the model name for its two replays (the model's calls,
#: then the ground truth's): multi_turn_checker passes `model_name` and `model_name +
#: "_ground_truth"`, and execute_multi_turn_func_call adds "_eval" to both.
_REPLAY_SUFFIXES = ("", "_eval", "_ground_truth_eval")


def _drop_instances(session_key: str, entry_id: str) -> None:
    """Forget the backend instances one entry created.

    execute_multi_turn_func_call keeps them in its module globals for the rest of the process,
    named `<model>_<entry>_<class>_instance` with [-./:] replaced by '_', where <model> is the
    session key (generation) or the key plus a _REPLAY_SUFFIXES suffix (the checker's replays).
    Only names built from exactly this key and this entry are removed: one attempt's key can be
    the start of another's (`..._a1` and `..._a10`), and one entry id the start of another's."""
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu
    prefix, entry = _instance_prefix(session_key), _instance_prefix(entry_id)
    heads = tuple(f"{prefix}{suffix}_{entry}_" for suffix in _REPLAY_SUFFIXES)
    namespace = vars(mtu)
    for key in [k for k in list(namespace) if k.startswith(heads) and k.endswith("_instance")]:
        namespace.pop(key, None)


_CONTEXT_MAX = re.compile(r"maximum context length is (\d+) tokens")


def _fc_handler(model_name: str, base_url: str, temperature: float, thinking: bool,
                session_key: str, request_timeout: Optional[float] = None):
    """bfcl_eval's OpenAICompletionsHandler in FC mode against the gbench endpoint.

    Only the request is extended: the thinking switch in chat_template_kwargs and any decoding
    penalties. No max_tokens is sent, as upstream sends none. The inference loop, tool
    compilation, history handling and decoding are upstream's; the client gets the endpoint, its
    key and a timeout sized for the longest answer (`request_timeout`)."""
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    extra_body: Dict[str, Any] = dict(decoding_penalties())
    extra_body["chat_template_kwargs"] = effective_chat_template_kwargs(None, thinking)
    timeout = float(request_timeout or multi_turn_request_timeout())

    class _GbenchFCHandler(OpenAICompletionsHandler):
        def _build_client_kwargs(self):
            return {"api_key": os.environ.get("OPENAI_API_KEY") or "EMPTY",
                    "base_url": base_url, "timeout": timeout}

        def generate_with_backoff(self, **kwargs):
            kwargs["extra_body"] = dict(extra_body)
            api_response, latency = super().generate_with_backoff(**kwargs)
            steps = getattr(self._local, "steps", None)
            if steps is not None:
                choice = api_response.choices[0]
                msg = choice.message
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                usage = getattr(api_response, "usage", None)
                steps.append({"finish_reason": choice.finish_reason,
                              "completion_tokens": getattr(usage, "completion_tokens", None),
                              "reasoning_chars": len(reasoning or ""),
                              "latency_s": round(latency, 3)})
            return api_response, latency

    handler = _GbenchFCHandler(model_name=model_name, temperature=temperature,
                               registry_name=model_name, is_fc_model=True)
    handler.model_name_underline_replaced = session_key   # prefix of its backend instances
    handler._local = threading.local()
    return handler


def _failure_status(error: BaseException) -> str:
    from openai import APITimeoutError, BadRequestError
    if isinstance(error, APITimeoutError):
        return "TIMEOUT"
    if isinstance(error, BadRequestError) and _CONTEXT_MAX.search(str(error)):
        return "OVER_CONTEXT"
    return "FAILED"


#: The error type of a multi-turn entry whose checker (_evaluate_single_multi_turn_entry) raised.
#: Upstream's evaluation aborts there; gbench makes it a harness error of the entry
#: (multi_turn_harness_error, kind CHECKER_EXCEPTION_KIND), never the model's wrong answer.
CHECKER_EXCEPTION = "multi_turn:checker_exception"
CHECKER_EXCEPTION_KIND = "checker_exception"


def _score_multi_turn_entry(handler: Any, item: Dict[str, Any], session_key: str) -> Dict[str, Any]:
    """Generate one multi-turn entry with BFCL's loop and score it with BFCL's checker."""
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
    handler._local.steps = []
    status, error = "OK", None
    try:
        result, _metadata = handler.inference(copy.deepcopy(item["entry"]),
                                              include_input_log=False, exclude_state_log=True)
    except Exception as e:
        # _llm_response_generation.multi_threaded_inference records the same string; the
        # checker then fails the entry as multi_turn:inference_error.
        result = f"Error during inference: {e}"
        status, error = _failure_status(e), f"{type(e).__name__}: {e}"
    try:
        verdict = _evaluate_single_multi_turn_entry(
            handler, item["id"], result, item["ground_truth"], copy.deepcopy(item["eval_entry"]),
            session_key, item["category"])
    except Exception as e:
        # Upstream's evaluation would abort here. The entry is marked with CHECKER_EXCEPTION
        # instead, a harness error left out of the score (multi_turn_harness_error).
        verdict = {"valid": False, "error": {
            "error_message": [f"multi_turn_checker raised {type(e).__name__}: {e}"],
            "error_type": CHECKER_EXCEPTION}}
    finally:
        _drop_instances(session_key, item["id"])
    steps = handler._local.steps
    truncated = any(s.get("finish_reason") == "length" for s in steps)
    if status == "OK" and truncated:
        status = "TRUNCATED"
    err = verdict.get("error") or {}
    message = err.get("error_message")
    return {
        "category": item["category"],
        "question_id": item["id"],
        "gold_answer": {"id": item["id"], "category": item["category"],
                        "ground_truth": item["ground_truth"]},
        "response_text": json.dumps(result, ensure_ascii=False, default=str)[:20000],
        "multi_turn_result": result if isinstance(result, list) else None,
        "steps": len(steps),
        "finish_reasons": [s.get("finish_reason") for s in steps],
        "completion_tokens": sum(int(s.get("completion_tokens") or 0) for s in steps),
        "reasoning_chars": sum(s.get("reasoning_chars") or 0 for s in steps),
        "health": "truncated" if truncated else ("request_failed" if error else "ok"),
        "error": error,
        "is_correct": bool(verdict.get("valid")),
        "status": status,
        **({} if verdict.get("valid") else {
            "bfcl_error_type": err.get("error_type"),
            "bfcl_error": (message if isinstance(message, list) else [str(message)])[:3],
            "bfcl_error_details": json.dumps(err.get("details"), default=str)[:4000]
            if err.get("details") is not None else None}),
    }


def run_multi_turn(model_name: str, base_url: str, concurrency: int,
                   items: List[Dict[str, Any]], *, temperature: float, thinking: bool,
                   attempts: int = 1, request_timeout: Optional[float] = None) -> Dict[str, Any]:
    """Every item x `attempts` through BFCL's multi-turn loop and checker, `concurrency` at a time
    (where bfcl_eval is importable: the harness image, see multi_turn_job).

    Each attempt gets its own instance-name prefix, so no backend state is shared between
    attempts, runs or the evaluation's model and ground-truth replays; an entry's cleanup removes
    only its own attempt's instances (_drop_instances), however many attempts run at once.

    Each finished entry is also reported as it completes (bfcl_v4_agentic.report_progress:
    {"job": its index, "trace", "refused": guard refusals since the last report}), so a part
    stopped by GBENCH_BFCL_TIMEOUT_S keeps the entries it finished."""
    run_id = uuid.uuid4().hex[:12]
    jobs = [(item, a) for item in items for a in range(max(1, attempts))]
    traces: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
    handlers = {a: _fc_handler(model_name, base_url, temperature, thinking,
                               f"gbench_bfcl_{run_id}_a{a}", request_timeout)
                for a in range(max(1, attempts))}
    with _harness_patches() as guard:
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
            futures = {pool.submit(_score_multi_turn_entry, handlers[a], item,
                                   handlers[a].model_name_underline_replaced): i
                       for i, (item, a) in enumerate(jobs)}
            reported = 0
            with tqdm(total=len(jobs), desc="bfcl multi-turn", unit="entry") as pbar:
                for fut in as_completed(futures):
                    i = futures[fut]
                    traces[i] = fut.result()
                    refused = list(guard.refused[reported:])
                    reported += len(refused)
                    report_progress({"job": i, "trace": traces[i], "refused": refused})
                    pbar.update(1)
    source = {item["id"]: i for i, item in enumerate(items)}
    out = []
    for i, ((item, a), trace) in enumerate(zip(jobs, traces)):
        out.append({"sample_idx": i, "source_sample_idx": source[item["id"]], "attempt_idx": a,
                    **trace})
    return {"traces": out, "guard_refusals": list(guard.refused)}


def multi_turn_job(model_name: str, base_url: str, concurrency: int, entries: List[List[str]],
                   temperature: float, thinking: bool, attempts: int = 1,
                   request_timeout: Optional[float] = None) -> Dict[str, Any]:
    """Inside the harness image: the multi-turn entries named by `entries` ([id, category], in
    that order), loaded by bfcl_eval, through run_multi_turn."""
    categories = sorted({c for _i, c in entries})
    by_id = {it["id"]: it for it in load_multi_turn_entries(categories)}
    missing = [i for i, _c in entries if i not in by_id]
    if missing:
        raise RuntimeError(f"bfcl: multi-turn entries {missing[:4]} are not in the harness data")
    items = [by_id[i] for i, _c in entries]
    return run_multi_turn(model_name, base_url, concurrency, items, temperature=temperature,
                          thinking=thinking, attempts=attempts, request_timeout=request_timeout)


def multi_turn_request_timeout() -> int:
    """Seconds one multi-turn request may take. No max_tokens is sent, so a reply can run to the
    end of the context window: base.request_timeout_s of the served window, or of
    base.UNCAPPED_TIMEOUT_TOKENS when the server reports none (base's rule for a request that
    carries no max_tokens)."""
    return request_timeout_s(get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)


def endpoint_network(base_url: str) -> Optional[str]:
    """The multi-turn part's checks before its container starts: the cloud metadata server is
    blocked on the bridge network the part runs on, where BFCL executes the model's calls with
    eval(). Without a real endpoint key, tau_common.require_endpoint_network also checks that a
    named endpoint host resolves there and that a loopback endpoint answers there at
    host.docker.internal, and returns the URL the container uses (told, as
    bfcl_v4_agentic.endpoint_gate tells it, when the container gets a proxy). With a real key
    (OPENAI_API_KEY), which never enters the container, bfcl_v4_agentic.bridge_network_check
    checks that the bridge gateway, where gbench's endpoint relay adds the key, is reachable from
    there, and None is returned. Raises infra_required otherwise. run_bfcl calls it before the
    first generation; the answers are cached, so the same checks when the part's container starts
    cost nothing more."""
    if _agentic.endpoint_key() is None:
        from .tau_common import container_proxy_env, require_endpoint_network
        if container_proxy_env(dict(os.environ)):
            # the container gets the proxy variables (_MT_ENV_NAMES): a request it sends through
            # the proxy needs only the proxy's host to resolve there
            return require_endpoint_network("bfcl", bfcl_image(), base_url, DOCS_URL,
                                            proxied=True)
        return require_endpoint_network("bfcl", bfcl_image(), base_url, DOCS_URL)
    return _agentic.bridge_network_check("bfcl", base_url, DOCS_URL)


#: The variables of this process' environment the multi-turn part's container receives, when
#: set (the endpoint key only as a placeholder: bridge_services).
_MT_ENV_NAMES = ("OPENAI_API_KEY", "GBENCH_REPETITION_PENALTY", "GBENCH_FREQUENCY_PENALTY",
                 "GBENCH_PRESENCE_PENALTY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "no_proxy")


def run_multi_turn_in_image(model_name: str, base_url: str, concurrency: int,
                            items: List[Dict[str, Any]], *, temperature: float, thinking: bool,
                            attempts: int = 1) -> Dict[str, Any]:
    """multi_turn_job in the harness image for `items` ({id, category}), talking to `base_url`
    from docker's bridge network (reached at host.docker.internal when it is on this host; the
    host network is never used, see endpoint_network). No key enters the container, where BFCL
    executes the model's calls: with a real endpoint key (OPENAI_API_KEY), gbench's endpoint relay
    on the bridge gateway adds it (bfcl_v4_agentic.bridge_services), the container gets the
    relay's URL and the placeholder EMPTY, and the relay's record is returned under
    `endpoint_relay`; a placeholder key is passed by name as it is. Progress streams to stderr.

    Bounded as BFCL bounds it: each request by the client timeout, each entry by BFCL's step
    limit per turn (an entry whose request fails is an inference error: a harness error, left
    out of the score by _multi_turn_counts), and the part as a whole by nothing unless
    GBENCH_BFCL_TIMEOUT_S is set. A part stopped by that wall clock keeps the entries it
    finished; the others become TIMEOUT traces (`harness_timeout`: unfinished, left out of the
    score) and the stop is returned under `harness_error`."""
    from .tau_common import ENDPOINT_NETWORK
    timeout_s = multi_turn_request_timeout()
    with _agentic.bridge_services("bfcl", DOCS_URL, base_url, key=_agentic.endpoint_key(),
                                  request_timeout=timeout_s) as svc:
        env = dict(os.environ)
        env.update(svc.container_env(env))
        env_names = [n for n in _MT_ENV_NAMES if env.get(n) is not None]
        try:
            part = _agentic.run_in_image("bfcl", "bfcl", "multi_turn_job", {
                "model_name": model_name, "base_url": svc.endpoint_url,
                "concurrency": int(concurrency),
                "entries": [[it["id"], it["category"]] for it in items],
                "temperature": temperature, "thinking": bool(thinking),
                "attempts": int(attempts), "request_timeout": timeout_s},
                timeout=part_timeout(), env=env, env_names=env_names, stream=True,
                part="the multi-turn part", network=ENDPOINT_NETWORK)
            part = {**part, "harness_error": None}
        except HarnessTimeout as e:
            part = stopped_multi_turn(e, items, attempts)
    part["endpoint_relay"] = svc.record()["endpoint_relay"]
    return part


def stopped_multi_turn(stop: HarnessTimeout, items: List[Dict[str, Any]],
                       attempts: int = 1) -> Dict[str, Any]:
    """run_multi_turn's result for a part stopped by its wall clock: the traces of the entries it
    reported (stop.progress), and a TIMEOUT trace for every other entry x attempt."""
    attempts = max(1, int(attempts))
    jobs = [(item, a) for item in items for a in range(attempts)]
    source = {item["id"]: i for i, item in enumerate(items)}
    done: Dict[int, Dict[str, Any]] = {}
    refused: List[str] = []
    for row in stop.progress:
        if isinstance(row, dict) and isinstance(row.get("trace"), dict) \
                and isinstance(row.get("job"), int) and 0 <= row["job"] < len(jobs):
            done[row["job"]] = row["trace"]
            refused += [str(r) for r in row.get("refused") or []]
    traces = []
    for i, (item, a) in enumerate(jobs):
        trace = done.get(i) or {
            "category": item["category"], "question_id": item["id"],
            "gold_answer": {"id": item["id"], "category": item["category"]},
            "response_text": None, "multi_turn_result": None, "steps": 0, "finish_reasons": [],
            "completion_tokens": 0, "reasoning_chars": 0, "health": "request_failed",
            "error": f"{stop} before this entry finished", "is_correct": False,
            "status": "TIMEOUT", "harness_timeout": True}
        traces.append({"sample_idx": i, "source_sample_idx": source[item["id"]],
                       "attempt_idx": a, **trace})
    logger.error("[bfcl] %s: %d of %d multi-turn entries finished; the rest are left out of the "
                 "score (harness errors)", stop, len(done), len(jobs))
    return {"traces": traces, "guard_refusals": refused,
            "harness_error": stop.record(entries_finished=len(done),
                                         entries_unfinished=len(jobs) - len(done))}


def multi_turn_harness_error(trace: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The harness error ({"kind", "detail"}) of a multi-turn trace, or None. An entry a part
    stopped by GBENCH_BFCL_TIMEOUT_S did not finish is `unfinished`; an entry whose request
    failed (FAILED, TIMEOUT, OVER_CONTEXT: upstream's "Error during inference") is base's
    endpoint failure of that status (base.ENDPOINT_FAILURE_KINDS); an entry whose checker raised
    (CHECKER_EXCEPTION, where upstream's evaluation aborts) is `checker_exception`."""
    if trace.get("harness_timeout"):
        return {"kind": _agentic.UNFINISHED_KIND,
                "detail": redact_text(str(trace.get("error") or "not finished"))[:300]}
    endpoint = endpoint_harness_error(trace.get("status"), trace.get("error"))
    if endpoint is not None:
        return endpoint
    if trace.get("bfcl_error_type") == CHECKER_EXCEPTION:
        detail = (trace.get("bfcl_error") or ["the multi-turn checker raised"])[0]
        return {"kind": CHECKER_EXCEPTION_KIND, "detail": redact_text(str(detail))[:300]}
    return None


def _multi_turn_counts(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The multi-turn part's counts. A trace that is a harness error (multi_turn_harness_error:
    its request failed, or a stopped part did not finish it) is never the model's outcome: it is
    marked `scoring_excluded` with its `harness_error` and left out of its category's correct
    and total, as base leaves out a request that got no reply, while a failed request still
    counts under failed_requests (and timed_out_requests / over_context_prompts). An entry whose
    checker raised (`checker_exception`) is left out the same way. A category every trace of
    which is one is left out of category_accuracy (`categories_unscored`: null, never 0)."""
    by_cat: Dict[str, Dict[str, int]] = {}
    left_out: Dict[str, Dict[str, int]] = {}
    for t in traces:
        st = by_cat.setdefault(t["category"], {"correct": 0, "total": 0})
        err = multi_turn_harness_error(t)
        if err is not None:
            t.update(is_correct=False, scoring_excluded=True, harness_error=err)
            kinds = left_out.setdefault(t["category"], {})
            kinds[err["kind"]] = kinds.get(err["kind"], 0) + 1
            continue
        st["total"] += 1
        st["correct"] += int(bool(t["is_correct"]))
    by_kind: Dict[str, int] = {}
    for kinds in left_out.values():
        for kind, n in kinds.items():
            by_kind[kind] = by_kind.get(kind, 0) + n
    return {
        "category_accuracy": {c: {"correct": s["correct"], "total": s["total"],
                                  "accuracy": round(s["correct"] / s["total"] * 100.0, 2)}
                              for c, s in by_cat.items() if s["total"]},
        "categories_unscored": sorted(c for c, s in by_cat.items() if not s["total"]),
        "total_questions": len(traces),
        "correct_answers": sum(1 for t in traces if t["is_correct"]),
        "failed_requests": sum(1 for t in traces if t["status"] in _REQUEST_FAILED),
        "timed_out_requests": sum(1 for t in traces if t["status"] == "TIMEOUT"),
        "over_context_prompts": sum(1 for t in traces if t["status"] == "OVER_CONTEXT"),
        "truncated_responses": sum(1 for t in traces if t["status"] == "TRUNCATED"),
        "scoring_excluded": sum(by_kind.values()),
        "request_harness_errors": dict(sorted(by_kind.items())),
        "harness_errors_left_out": {c: dict(sorted(k.items()))
                                    for c, k in sorted(left_out.items())},
    }


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #

def _mixed_attempts_plan(limit: int, quotas: Dict[str, Optional[int]],
                         part_attempts: Dict[str, int], attempts: int,
                         note: str) -> Dict[str, Any]:
    """The limited run's eval_limit_plan (logged in one line) when the parts run different
    attempts per entry (--attempt-count above 1 with agentic entries drawn, which generate once):
    the generations are each part's entries x its attempts."""
    from . import limit_plan as _limit_plan
    counts = {p: int(n or 0) for p, n in quotas.items()}
    ran = {p: n for p, n in counts.items() if n}
    idle = [p for p, n in counts.items() if not n]
    generations = sum(n * part_attempts[p] for p, n in counts.items())
    line = (f"[bfcl] --eval-limit {int(limit)}: {sum(counts.values())} sample(s) over "
            f"{len(ran)} part(s) (" + ", ".join(f"{p} {n}" for p, n in ran.items()) + ")"
            + (f" ({', '.join(idle)}: 0, not run)" if idle else "")
            + " x " + ", ".join(f"{p} {part_attempts[p]} attempt"
                                + ("s" if part_attempts[p] != 1 else "") for p in ran)
            + f" = {generations} generations")
    logger.info(line)
    plan: Dict[str, Any] = {"limit": int(limit), "samples": sum(counts.values()),
                            "per_setting": counts, "attempts_per_sample": attempts,
                            "generations": generations}
    if note:
        plan["note"] = note
    _limit_plan._PLANNED["bfcl"] = plan
    return plan


def _part_limits(limit: Optional[int], parts: Dict[str, List[str]]) -> Dict[str, Optional[int]]:
    """`--eval-limit N` split over the parts that run, N entries in total: dealt over the
    categories of every part (fc_common.part_quotas, seeded on the suite), so a part gets one
    entry per category drawn. A part can get 0, and is then not run in that limited run."""
    return part_quotas(limit, {p: {(i, c): None for i, c in enumerate(cats)}
                               for p, cats in parts.items()}, seed="bfcl")


def _check_agentic(base_url: str, categories: List[str]) -> Optional[str]:
    """Before generating anything: fail when the agentic part cannot run canonically (the
    harness image, the endpoint, the project root, the bridge network `bfcl generate` runs on,
    and, when the Web Search setting will run, what its generate needs there: the grounding
    links' host resolving and the bridge gateway, where gbench's Gemini relay listens), and
    return why BFCL's Web Search setting cannot run (no Gemini key, or one the Gemini API
    rejects: its web_search tool is served by Gemini Google Search grounding), or None. That
    setting is then reported infra_required while every other setting runs. The checks run for
    this suite: a failure names bfcl and docs/evals/bfcl.md (bfcl_v4_agentic.suite_context)."""
    from .bfcl_v4_agentic import check_bfcl_prerequisites, search_backend, suite_context
    why: Optional[str] = None
    if any(c in WEB_SEARCH_CATEGORIES for c in categories):
        backend, reason = search_backend()
        if backend is None:
            why = (f"{reason}. Set GEMINI_API_KEY (or GEMINI_API_KEYS) to a key the Gemini API "
                   "accepts and re-run.")
    with suite_context("bfcl", DOCS_URL, web_search=bool(
            why is None and any(c in WEB_SEARCH_CATEGORIES for c in categories))):
        ok, reason, _ = check_bfcl_prerequisites(base_url)
    if not ok:
        raise infra_required("bfcl", f"the agentic categories cannot run: {reason}", DOCS_URL)
    return why


def _empty_part(model_name: str, thinking: bool) -> Dict[str, Any]:
    return {"benchmark_type": "eval", "eval_name": "bfcl", "model_name": model_name,
            "thinking": thinking, "total_questions": 0, "correct_answers": 0,
            "failed_requests": 0, "empty_responses": 0, "truncated_responses": 0,
            "over_context_prompts": 0, "timed_out_requests": 0, "judge_failures": 0,
            "judge_outages": 0, "scoring_mode": "deterministic", "response_health": {},
            "attempts": None, "category_accuracy": {}, "sample_traces": [], "duration_s": 0.0}


def _status(result: Dict[str, Any]) -> str:
    """base._run_suite_async's status rule over the merged counts; an entry left out of the
    score as a harness error (`scoring_excluded`), a stopped harness part or a part whose every
    entry failed its request (`harness_errors`), endpoint failures gbench's endpoint relay
    counted (`endpoint_failures`), memory prerequisite entries of the agentic part whose request
    failed (`memory_prereq_failed_requests`) or a setting whose prerequisite was missing
    (`settings_infra_required`) makes it completed_with_errors, or failed when nothing was
    scored."""
    total, failed = result["total_questions"], result["failed_requests"]
    unscored = harness_unscored(result.get("sample_traces") or [])
    # entries left out of the score as harness errors, in every part
    excluded = int(result.get("scoring_excluded") or 0)
    if total == 0:
        return "error"
    if failed + unscored >= total or excluded >= total:
        return "failed"
    if (failed == 0 and not excluded and not result.get("judge_outages")
            and not result.get("harness_errors")
            and not result.get("settings_infra_required") and not result.get("endpoint_failures")
            and not result.get("memory_prereq_failed_requests")
            and result["truncated_responses"] * 10 <= total):
        return "success"
    return "completed_with_errors"


def _status_reason(result: Dict[str, Any]) -> Optional[str]:
    """What made the merged status other than success (None on success), in base's words: the
    failed requests, the entries left out of the score as harness errors by kind, the harness
    parts stopped or left unscored, endpoint failures, failed memory prerequisites, a setting
    reported infra_required and replies cut off over 10%."""
    if result.get("status") == "success":
        return None
    causes: List[str] = []
    failed = int(result.get("failed_requests") or 0)
    if failed:
        parts = [f"{k} {int(result.get(key) or 0)}" for k, key in (
            ("timed out", "timed_out_requests"), ("over the context window",
                                                  "over_context_prompts"))
                 if int(result.get(key) or 0)]
        causes.append(f"failed_requests {failed}" + (f" ({', '.join(parts)})" if parts else ""))
    excluded = int(result.get("scoring_excluded") or 0)
    if excluded:
        kinds = result.get("request_harness_errors") or {}
        causes.append(f"harness errors left out of the score: {excluded} entries"
                      + (" (" + ", ".join(f"{k} {v}" for k, v in kinds.items()) + ")"
                         if kinds else ""))
    for err in result.get("harness_errors") or []:
        causes.append(f"harness error: {err.get('error')}")
    if result.get("endpoint_failures"):
        causes.append("endpoint failures the relay counted: "
                      + json.dumps(result["endpoint_failures"], sort_keys=True))
    if result.get("memory_prereq_failed_requests"):
        causes.append(f"memory_prereq_failed_requests {result['memory_prereq_failed_requests']}")
    for setting in result.get("settings_infra_required") or []:
        causes.append(f"the {setting} setting is infra_required")
    total = int(result.get("total_questions") or 0)
    if total and int(result.get("truncated_responses") or 0) * 10 > total:
        causes.append(f"truncated_responses {result['truncated_responses']} of {total} "
                      "(over 10%)")
    return "; ".join(causes) or None


#: The statuses of a multi-turn trace whose request failed (bfcl._failure_status).
_REQUEST_FAILED = ("FAILED", "TIMEOUT", "OVER_CONTEXT")


def _all_requests_failed(part: str, n: int, how: str) -> Dict[str, Any]:
    """The `harness_errors` entry of a part whose every one of `n` entries failed its request
    (`how` says how) with no wall clock stop of its own: nothing of it measured the model, so
    its categories are left unscored (null columns, not 0)."""
    return {"part": part, "entries_failed": n,
            "error": (f"{part}: every one of its {n} entries failed its request ({how}); its "
                      "categories are left unscored")}


def _add_kinds(result: Dict[str, Any], kinds: Dict[str, int]) -> None:
    """Add a part's harness errors left out of the score, by kind, to the result's
    `request_harness_errors` (base's field, which the single-turn part starts)."""
    if not kinds:
        return
    merged = dict(result.get("request_harness_errors") or {})
    for kind, n in kinds.items():
        merged[kind] = int(merged.get(kind) or 0) + int(n)
    result["request_harness_errors"] = dict(sorted(merged.items()))


def _attempts_report(traces: List[Dict[str, Any]], attempts: int) -> Optional[Dict[str, Any]]:
    """base's avg@k / pass@k / pass^k over the single-turn and multi-turn traces."""
    if attempts <= 1:
        return None
    traces = [t for t in traces if not t.get("scoring_excluded")]
    by_source: Dict[Any, List[bool]] = {}
    for t in traces:
        by_source.setdefault(t.get("source_sample_idx", t.get("sample_idx")), []).append(
            bool(t.get("is_correct")))
    n = len(by_source)
    return {"attempts_requested": attempts, "attempts_per_sample": attempts, "samples": n,
            "generations": len(traces),
            # None, never 0, when every trace is a harness error (as base's @k block)
            "avg_at_k": (round(sum(t.get("is_correct") is True for t in traces) / len(traces)
                               * 100.0, 2) if traces else None),
            "pass_at_k": (round(sum(1 for v in by_source.values() if any(v)) / n * 100.0, 2)
                          if n else None),
            "pass_hat_k": (round(sum(1 for v in by_source.values() if v and all(v)) / n * 100.0, 2)
                           if n else None),
            "unstable_samples": sum(1 for v in by_source.values() if any(v) and not all(v)),
            "not_applied_to": "the agentic categories (the bfcl_v4_agentic harness generates once)"}


def run_bfcl(
    model_name: str,
    base_url: str,
    concurrency: int,
    eval_categories: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run BFCL v4 (single-turn, multi-turn, agentic) and headline BFCL Overall Acc."""
    categories = resolve_categories(eval_categories)
    categories_subset = set(categories) != set(SUPPORTED_BFCL_CATEGORIES)
    thinking = bool(kwargs.get("enable_thinking", False))
    limit = kwargs.get("limit")
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    single = [c for c in categories if c in SINGLE_TURN_CATEGORIES]
    multi = [c for c in categories if c in MULTI_TURN_CATEGORIES]
    agentic = [c for c in categories if c in AGENTIC_CATEGORIES]
    # The agentic harness cannot be sharded: it runs whole in shard 1, so the shards still
    # union to the benchmark exactly once.
    agentic_shard_skip = bool(agentic and shard and shard[0] != 1)
    if agentic_shard_skip:
        agentic = []
    quotas = _part_limits(limit, {"single": single, "multi": multi, "agentic": agentic})
    # --eval-limit N is N entries for the suite: a part none of whose entries was drawn does not
    # run (nor need its prerequisites).
    single, multi, agentic = ([] if quotas.get(p) == 0 else cats for p, cats in
                              (("single", single), ("multi", multi), ("agentic", agentic)))
    # 1.0 with --thinking, 0.0 without, unless the operator sets another; the same value for
    # every part.
    temperature, temperature_source = resolve_official_sampling("bfcl", BFCL_OFFICIAL_SAMPLING,
                                                                thinking=thinking)
    attempts = max(1, int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1))
    # The planned workload, in one line before anything runs (a full run: every entry of the
    # parts' categories).
    limited = bool(limit and int(limit) > 0)
    note = (("plus the memory prerequisite entries the drawn agentic questions depend on"
             if agentic else "") if limited else
            f"every entry of {len(single) + len(multi) + len(agentic)} categories")
    # --attempt-count applies to the single- and multi-turn parts; the agentic harness generates
    # each entry once.
    part_attempts = {p: (1 if p == "agentic" else attempts) for p in quotas}
    if limited and attempts > 1 and quotas.get("agentic"):
        limit_plan = _mixed_attempts_plan(limit, quotas, part_attempts, attempts, note)
    else:
        limit_plan = eval_limit_plan(
            "bfcl", limit, sum(quotas.values()) if limited else None,
            per_setting=quotas, attempts=attempts, setting_word="part", note=note)
    if attempts > 1 and quotas.get("agentic", 0) != 0:
        limit_plan["attempts_per_part"] = part_attempts

    # Everything that can stop the run is checked before the first generation.
    mt_items: List[Dict[str, Any]] = []
    harness_version = None
    if single or multi:
        harness_version, mt_items = _harness_probe("bfcl", multi, bool(single))
    if multi:
        endpoint_network(base_url)
    # Why BFCL's Web Search setting cannot run (its Gemini key), or None: that setting alone is
    # then reported infra_required, unless it is all the run holds.
    web_why: Optional[str] = None
    web_cats = [c for c in agentic if c in WEB_SEARCH_CATEGORIES]
    if agentic:
        web_why = _check_agentic(base_url, agentic)
        if web_why is not None and not (single or multi
                                        or any(c in MEMORY_CATEGORIES for c in agentic)):
            raise infra_required("bfcl", web_why, DOCS_URL)
    samples: List[Any] = []
    functions_by_id: Dict[str, Any] = {}
    if single:
        samples, functions_by_id = _load_bfcl_samples(single)

    started = time.time()
    # Harness parts GBENCH_BFCL_TIMEOUT_S stopped; everything else in the run is kept.
    harness_errors: List[Dict[str, Any]] = []
    # Categories that ran and have no entry left in the score (every entry a harness error),
    # besides those of a part whose every entry failed (named in harness_errors).
    categories_unscored: List[str] = []
    if single:
        scorer = bfcl_batch_scorer(functions_by_id)
        result = run_eval_suite(
            eval_name="bfcl",
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=samples,
            async_eval_fn=scorer,
            declared_scoring_mode="deterministic",
            supports_attempts=True,
            thinking=thinking,
            extra_payload=kwargs.get("extra_payload"),
            limit=quotas.get("single"),
            max_output_tokens=kwargs.get("max_output_tokens"),
            official_sampling=BFCL_OFFICIAL_SAMPLING,
        )
        result["single_turn_checker"] = (f"bfcl-eval {harness_version} eval_runner in the local "
                                         f"image {bfcl_image()} ({BFCL_DOCKERFILE})")
        harness_errors += scorer.harness_errors
        # Base leaves the harness errors out of the score (a request that got no reply, a reply
        # a stopped checker did not score): a category that ran with none of its entries left
        # is out of category_accuracy (null columns, never 0) and named here.
        ran_single = {t.get("category") for t in result.get("sample_traces") or []}
        categories_unscored += sorted(c for c in ran_single - {None}
                                      if c not in (result.get("category_accuracy") or {}))
    else:
        result = _empty_part(model_name, thinking)
    result.update(temperature=temperature, temperature_source=temperature_source,
                  official_sampling=BFCL_OFFICIAL_SAMPLING.record(temperature, temperature_source),
                  max_output_tokens_note=max_tokens_note())

    mt_part: Optional[Dict[str, Any]] = None
    mt_over_context = 0            # multi-turn entries left out for outgrowing the window
    if multi:
        if shard:
            mt_items = shard_select(mt_items, shard)
        if quotas.get("multi"):
            mt_items = stratified_sample(mt_items, quotas["multi"], key_fn=lambda it: it["category"],
                                         seed="bfcl")
        mt_part = run_multi_turn_in_image(model_name, base_url, concurrency, mt_items,
                                          temperature=temperature, thinking=thinking,
                                          attempts=attempts)
        if mt_part.get("harness_error"):
            harness_errors.append(mt_part["harness_error"])
        # Its harness errors (a failed request, an entry a stopped part did not finish) are left
        # out of the score, never the model's outcome; a category nothing is left of is unscored.
        counts = _multi_turn_counts(mt_part["traces"])
        mt_over_context = int(counts["request_harness_errors"].get("over_context") or 0)
        # A part whose every entry failed its request measured nothing of the model: its entries
        # are counted (failed_requests) and its categories left unscored, never reported as 0.
        mt_all_failed = bool(mt_part["traces"]) and all(
            t["status"] in _REQUEST_FAILED for t in mt_part["traces"])
        if mt_all_failed:
            by_status: Dict[str, int] = {}
            for t in mt_part["traces"]:
                by_status[t["status"]] = by_status.get(t["status"], 0) + 1
            unscored_cats = list(counts["categories_unscored"])
            if mt_part.get("harness_error"):
                # a wall clock stopped the part before any entry finished: its record says so
                mt_part["harness_error"]["categories_unscored"] = unscored_cats
            else:
                harness_errors.append({**_all_requests_failed(
                    "the multi-turn part", len(mt_part["traces"]),
                    ", ".join(f"{k} {v}" for k, v in sorted(by_status.items()))),
                    "categories_unscored": unscored_cats})
                logger.error("[bfcl] %s", harness_errors[-1]["error"])
        else:
            categories_unscored += counts["categories_unscored"]
        if counts["scoring_excluded"]:
            logger.error("[bfcl] %d multi-turn entries are harness errors, left out of the score "
                         "(%s)", counts["scoring_excluded"],
                         ", ".join(f"{k} {v}" for k, v in counts["request_harness_errors"].items()))
        for key in ("total_questions", "correct_answers", "failed_requests", "timed_out_requests",
                    "over_context_prompts", "truncated_responses", "scoring_excluded"):
            result[key] = int(result.get(key) or 0) + counts[key]
        _add_kinds(result, counts["request_harness_errors"])
        result["category_accuracy"] = {**(result.get("category_accuracy") or {}),
                                       **counts["category_accuracy"]}
        offset = len(result.get("sample_traces") or [])
        src_offset = 1 + max((t.get("source_sample_idx", t.get("sample_idx", 0))
                              for t in result.get("sample_traces") or []), default=-1)
        health = dict(result.get("response_health") or {})
        for t in mt_part["traces"]:
            t["sample_idx"] += offset
            t["source_sample_idx"] += src_offset
            health[t["health"]] = health.get(t["health"], 0) + 1
        result["response_health"] = health
        result["sample_traces"] = list(result.get("sample_traces") or []) + mt_part["traces"]
        result["multi_turn_report"] = {
            "harness": (f"bfcl-eval {harness_version} in the local image {bfcl_image()} "
                        f"({BFCL_DOCKERFILE}; files checked against gorilla@{BFCL_EVAL_COMMIT})"),
            "entries": len(mt_items), "attempts": attempts,
            "budget": "no max_tokens sent (as BFCL's handler)",
            "request_timeout_s": multi_turn_request_timeout(),
            "model_requests": sum(t["steps"] for t in mt_part["traces"]),
            "force_terminated": sum(1 for t in mt_part["traces"]
                                    if t.get("bfcl_error_type") == "multi_turn:force_terminated"),
            "execution_guard_refusals": len(mt_part["guard_refusals"]),
            # gbench's relay that adds the endpoint key outside the container (None: the key is
            # unset or a placeholder, and the container reaches the endpoint directly)
            "endpoint_relay": mt_part.get("endpoint_relay"),
            # the endpoint failures that relay counted (harness errors), or None
            "endpoint_failures": _agentic.endpoint_failures([mt_part.get("endpoint_relay")]),
            # per category, the entries left out of the score as harness errors, by kind
            "harness_errors_left_out": counts["harness_errors_left_out"],
        }
        if mt_part["guard_refusals"]:
            logger.warning("[bfcl] %d model call(s) were refused by the execution guard, e.g. %r",
                           len(mt_part["guard_refusals"]), mt_part["guard_refusals"][0])

    ag_result: Optional[Dict[str, Any]] = None
    # BFCL's Web Search setting when its prerequisite is missing: its own infra_required entry.
    web_entry: Optional[Dict[str, Any]] = None
    if agentic and web_why is not None and not any(c in MEMORY_CATEGORIES for c in agentic):
        # Nothing of the agentic part can run: the other parts run and this setting is reported.
        web_entry = web_search_infra_required(web_cats, web_why, "before generation", "bfcl")
        _agentic._log_web_infra(web_entry, "bfcl")
    elif agentic:
        from .bfcl_v4_agentic import run_bfcl_v4_agentic
        # The agentic part generates at this suite's temperature, whatever its own knobs say
        # (temperature_source marks the value as already resolved). It checks the Gemini key
        # itself and reports a missing one as its Web Search setting's entry.
        try:
            # Its prerequisite and network checks name this suite and its doc.
            with _agentic.suite_context("bfcl", DOCS_URL):
                ag_result = run_bfcl_v4_agentic(
                    model_name, base_url, concurrency, enable_thinking=thinking,
                    eval_categories=(",".join(agentic) if set(agentic) != set(AGENTIC_CATEGORIES)
                                     else None),
                    limit=quotas.get("agentic"), temperature=temperature,
                    temperature_source=temperature_source)
        except SearchUnavailable as e:
            # The web search was the whole agentic part and lost its prerequisite: the
            # single-turn and multi-turn results stand; the Web Search setting is reported
            # infra_required.
            web_entry = web_search_infra_required(web_cats, e.reason, e.when, "bfcl")
            _agentic._log_web_infra(web_entry, "bfcl")
        except HarnessTimeout as e:
            # `bfcl evaluate` stopped: nothing of the agentic part is scored; the single-turn and
            # multi-turn results stand (the agentic categories stay unscored, N/A). A stopped
            # `bfcl generate` before it keeps its own record.
            logger.error("[bfcl] %s: the agentic categories are left unscored", e)
            harness_errors += list(e.earlier)
            harness_errors.append(e.record(categories_unscored=list(agentic)))
        except Exception as e:
            if str(e).startswith("[INFRA REQUIRED]"):
                # Another prerequisite of the agentic part that disappeared mid-run (the harness
                # image, the endpoint) is a hard error, like a missing one. What the other parts
                # measured is logged before the suite stops.
                logger.error("[bfcl] the agentic part hit a missing prerequisite; the parts "
                             "already run scored %s", {
                                 c: v.get("accuracy") for c, v in
                                 (result.get("category_accuracy") or {}).items()})
            raise
    if ag_result is not None:
        harness_errors += list(ag_result.get("harness_errors") or [])
        ag_ca = {c: v for c, v in (ag_result.get("category_accuracy") or {}).items() if c in agentic}
        # The agentic part leaves its harness errors out of the score (entries whose request
        # failed or that a stopped generate did not finish, memory entries of a scenario whose
        # prerequisite request failed); a category nothing is left of is unscored, and when
        # nothing is left at all it withholds its score: its categories left unscored, never 0.
        ag_withheld = ag_result.get("status") == "failed"
        ag_unscored = [c for c in (ag_result.get("categories_unscored") or []) if c in agentic]
        result["total_questions"] += int(ag_result.get("total_questions") or 0)
        if not ag_withheld:
            result["correct_answers"] += sum(int(v["correct"]) for v in ag_ca.values())
        for key in ("failed_requests", "timed_out_requests", "over_context_prompts",
                    "scoring_excluded"):
            result[key] = int(result.get(key) or 0) + int(ag_result.get(key) or 0)
        _add_kinds(result, ag_result.get("request_harness_errors") or {})
        # its memory prerequisite entries whose request failed: not scored entries, so not in
        # failed_requests, but harness errors (the agentic part's reason names them)
        result["memory_prereq_failed_requests"] = int(
            ag_result.get("memory_prereq_failed_requests") or 0)
        if ag_withheld:
            unscored_all = sorted(set(ag_ca) | set(ag_unscored))
            if ag_result.get("harness_errors"):
                # a wall clock stopped `bfcl generate` before any entry finished: its record
                # (merged above) says so
                ag_result["harness_errors"][-1]["categories_unscored"] = unscored_all
            else:
                harness_errors.append({
                    "part": "the agentic part", "categories_unscored": unscored_all,
                    "error": (f"the agentic part: {ag_result.get('accuracy_withheld_reason')}; "
                              "its categories are left unscored")})
            logger.error("[bfcl] the agentic part's score is withheld (%s); its categories are "
                         "left unscored", ag_result.get("accuracy_withheld_reason"))
        else:
            result["category_accuracy"] = {**(result.get("category_accuracy") or {}), **ag_ca}
            # (named in the agentic part's reason)
            categories_unscored += ag_unscored
        offset = len(result.get("sample_traces") or [])
        result["sample_traces"] = list(result.get("sample_traces") or []) + [
            {**t, "sample_idx": offset + i, **({"scoring_excluded": True} if ag_withheld else {})}
            for i, t in enumerate(ag_result.get("sample_traces") or [])]
        if ag_result.get("gemini_models_answered") is not None:
            # the Gemini models that answered the agentic part's web searches
            result["gemini_models_answered"] = ag_result["gemini_models_answered"]
            result["gemini_cascade"] = ag_result.get("gemini_cascade")
        if (ag_result.get("settings") or {}).get("web_search", {}).get("status") \
                == "infra_required":
            web_entry = dict(ag_result["settings"]["web_search"])
        result["agentic_report"] = {
            "delegated_to": "bfcl_v4_agentic",
            "settings": ag_result.get("settings"),
            "leaderboard_comparable": ag_result.get("leaderboard_comparable"),
            "leaderboard_comparable_reason": ag_result.get("leaderboard_comparable_reason"),
            "temperature": ag_result.get("temperature"),
            "temperature_source": ag_result.get("temperature_source"),
            **(ag_result.get("bfcl_report") or {}),
            # the endpoint failures gbench's endpoint relay counted during `bfcl generate`, or None
            "endpoint_failures": ag_result.get("endpoint_failures"),
            # its entries left out of the score as harness errors (per category and kind in
            # harness_errors_left_out)
            "scoring_excluded": int(ag_result.get("scoring_excluded") or 0),
            "accuracy_withheld_reason": ag_result.get("accuracy_withheld_reason"),
            "sample_traces_note": "per-question traces are the harness' first 40 score-file rows",
        }

    if web_entry is not None:
        # BFCL's Web Search setting: not scored, never dropped. Overall Acc and the Agentic
        # column, which weigh it, stay null.
        result["settings_infra_required"] = ["web_search"]
        result["web_search_setting"] = web_entry
    category_accuracy = result.get("category_accuracy") or {}
    result["bfcl_summary"] = bfcl_summary(category_accuracy)
    result["dataset_revision"] = DATASET_REVISION
    result["pooled_accuracy"] = pooled(category_accuracy)
    # the entries in the score (harness errors left out), for the summary's Effective N
    result["effective_n"] = sum(int(v.get("total") or 0) for v in category_accuracy.values())
    # every category that ran and has no entry in the score because of harness errors (its
    # columns null, never 0): those named in harness_errors and those left unscored in a part
    # that scored others
    result["categories_unscored"] = sorted(set(categories_unscored) | {
        c for err in harness_errors for c in (err.get("categories_unscored") or [])})
    result["duration_s"] = round(time.time() - started, 2)
    score = overall_score(category_accuracy)
    if score is not None:
        result["accuracy"] = score
        result["metric"] = ("BFCL Overall Acc: 10% Non-Live AST, 10% Live AST, 10% Irrelevance, "
                            "30% Multi-Turn, 40% Agentic (eval_runner_helper.py); every "
                            "leaderboard column in bfcl_summary")
    elif web_entry is not None and not categories_subset:
        # A full run whose Web Search setting could not run (its prerequisite missing: a harness
        # error, not the model's outcome): Overall Acc weighs that setting, so the headline is
        # withheld rather than replaced by a pooled figure over what did run.
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = (
            f"BFCL Overall Acc weighs the Web Search setting ({', '.join(web_cats)}), which was "
            f"not scored: its prerequisite ({_agentic.SEARCH_PREREQUISITE}) was not available "
            f"{web_entry['when']} (infra_required); the scored columns are in bfcl_summary and "
            "pooled_accuracy")
        result["metric"] = ("BFCL Overall Acc (withheld: the Web Search setting was not scored); "
                            "per-column scores in bfcl_summary")
    elif not categories_subset and result["categories_unscored"]:
        # A full run with a category every entry of which was a harness error: Overall Acc weighs
        # it, so the headline is withheld rather than replaced by a pooled figure over the rest.
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = (
            f"BFCL Overall Acc weighs {', '.join(result['categories_unscored'])}, which "
            f"{'has' if len(result['categories_unscored']) == 1 else 'have'} no entry in the "
            "score (every entry a harness error); the scored columns are in bfcl_summary and "
            "pooled_accuracy")
        result["metric"] = ("BFCL Overall Acc (withheld: a category it weighs has no entry in the "
                            "score); per-column scores in bfcl_summary")
    else:
        # an explicit --eval-categories subset (or a segment that did not run in full): the
        # pooled accuracy over the categories run, never a 0 for nothing scored
        result["accuracy"] = result["pooled_accuracy"]
        result["metric"] = ("pooled accuracy over the categories run (BFCL Overall Acc needs every "
                            "category it weighs); per-column scores in bfcl_summary")
    result["harness_errors"] = harness_errors
    # Endpoint failures gbench's endpoint relay counted in the multi-turn part and in each
    # `bfcl generate` of the agentic part: harness errors, never the model's outcome.
    relay_records = [mt_part.get("endpoint_relay") if mt_part else None] + [
        r.get("endpoint_relay") for r in ((ag_result or {}).get("bfcl_report") or {}).get(
            "relays") or []]
    endpoint_fail = _agentic.endpoint_failures(relay_records)
    if endpoint_fail:
        result["endpoint_failures"] = endpoint_fail
    result["status"] = _status(result)
    if result["status"] in ("failed", "error"):
        # Nothing was scored: no score, not a 0.
        result["accuracy_withheld_reason"] = (
            "nothing measured the model: every entry run failed its request, was not finished "
            "or was left out of the score as a harness error" if result["status"] == "failed"
            else "no entry of the run was scored")
        result["accuracy"] = None
    result["status_reason"] = _status_reason(result)
    if attempts > 1 and (single or multi):
        result["attempts"] = _attempts_report(
            [t for t in result["sample_traces"] if t.get("category") not in AGENTIC_CATEGORIES],
            attempts)
    result["categories_run"] = [c for c in categories if c in category_accuracy]
    result["eval_limit_plan"] = limit_plan

    reasons: List[str] = []
    missing = [name for name, cats, _ in OVERALL_SEGMENTS
               if any(c not in category_accuracy for c in cats)]
    if missing:
        reasons.append(f"BFCL Overall Acc weighs 21 categories and the {', '.join(missing)} "
                       "segment(s) did not run in full")
    reasons += run_notes(limit, categories_subset=categories_subset)
    if agentic_shard_skip:
        reasons.append("the agentic categories run in shard 1 only")
    if web_entry is not None and ag_result is None:
        # (from the agentic part, its own reason below says it)
        reasons.append(f"BFCL's Web Search setting ({', '.join(web_cats)}) was not scored: its "
                       f"prerequisite ({_agentic.SEARCH_PREREQUISITE}) was not available "
                       f"{web_entry['when']} (infra_required: {web_entry['infra_required']})")
    if not result["official_sampling"]["temperature_matches_official"]:
        reasons.append(f"temperature {temperature} (BFCL generates at 0.001)")
    if ag_result is not None:
        ag_temp = ag_result.get("temperature")
        if ag_temp is None or abs(float(ag_temp) - float(temperature)) > 1e-9:
            reasons.append(f"the agentic part generated at temperature {ag_temp}, not the "
                           f"{temperature} of the other parts")
    if attempts > 1:
        reasons.append(f"avg@{attempts} over {attempts} attempts (BFCL generates once)")
    if mt_part and mt_part["guard_refusals"]:
        reasons.append(f"{len(mt_part['guard_refusals'])} multi-turn calls refused by the "
                       "execution guard (upstream would have run them)")
    mt_endpoint_fail = (result.get("multi_turn_report") or {}).get("endpoint_failures")
    if mt_endpoint_fail:
        # (the agentic part's own endpoint failures are in its reason below)
        reasons.append(_agentic.endpoint_failure_reason(mt_endpoint_fail, "the multi-turn part"))
    # The single-turn and multi-turn entries left out of the score as harness errors (the
    # agentic part's are in its reason below): the score is over the rest.
    own_kinds = dict(result.get("request_harness_errors") or {})
    for kind, n in ((ag_result or {}).get("request_harness_errors") or {}).items():
        own_kinds[kind] = int(own_kinds.get(kind) or 0) - int(n)
    own_kinds = {k: v for k, v in own_kinds.items() if v > 0}
    if own_kinds:
        n_own = sum(own_kinds.values())
        reasons.append(f"{n_own} single-turn and multi-turn "
                       f"{'entry' if n_own == 1 else 'entries'} left out of the score as harness "
                       "errors (" + ", ".join(f"{k}: {v}" for k, v in sorted(own_kinds.items()))
                       + "; BFCL scores a failed request wrong"
                       + ("; its evaluation aborts where the multi-turn checker raises"
                          if own_kinds.get(CHECKER_EXCEPTION_KIND) else "") + ")")
    # Multi-turn and agentic entries left out because their conversation outgrew the served
    # context window: the conversation grows with the model's own tool calls and results, which
    # upstream scores as the model's outcome (gbench leaves them out as harness errors).
    grown = int(mt_over_context) + int(((ag_result or {}).get("request_harness_errors") or {})
                                       .get("over_context") or 0)
    if grown:
        reasons.append(f"{grown} multi-turn or agentic entr{'y' if grown == 1 else 'ies'} left "
                       "out as harness errors because the conversation outgrew the served context "
                       "window (it grows with the model's own tool calls and results, which "
                       "upstream scores as the model's outcome)")
    if categories_unscored and set(categories_unscored) - set(AGENTIC_CATEGORIES):
        reasons.append("categories left unscored, every entry a harness error: " + ", ".join(
            sorted(set(categories_unscored) - set(AGENTIC_CATEGORIES))))
    if result.get("accuracy_withheld_reason"):
        reasons.append(f"score withheld: {result['accuracy_withheld_reason']}")
    agentic_errors = list((ag_result or {}).get("harness_errors") or [])
    for err in harness_errors:
        if err not in agentic_errors:           # those are in the agentic reason below
            reasons.append(f"harness error: {err['error']}")
    if ag_result is not None and not ag_result.get("leaderboard_comparable"):
        reasons.append(f"agentic: {ag_result.get('leaderboard_comparable_reason')}")
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    return result
