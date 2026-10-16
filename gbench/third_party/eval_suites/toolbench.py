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

# Evaluation Suite: toolbench
# Description: ToolBench / StableToolBench - multi-turn agentic RapidAPI tool use, ToolEval SoPR/SoWR

"""gbench native built-in runner for toolbench (Tool Use & Function Calling).

Canonical ToolBench is scored by StableToolBench (THUNLP-MT/StableToolBench, arXiv:2403.07714):
the model under test runs a MULTI-TURN DFSDT agentic loop, calling RapidAPI tools against a CACHED
API server (deterministic; a call it has not cached goes to the ToolBench real-API proxy, then to
an LLM simulator, as upstream's server/config.yml sets it), finishing with a `Finish` call. The
simulator's model (gpt-4-turbo upstream) is gbench's Gemini judge cascade on the Gemini API's
OpenAI-compatible endpoint, the one path (no other provider, base URL, key or model can be set),
recorded in `simulator`, `gemini_substitutions` and leaderboard_comparable_reason. gbench
DELEGATES the inference + answer-format conversion to StableToolBench's own pipeline (subprocess,
pointing inference at the gbench-served /v1), then scores the converted answer trees with:

  * SoPR (Solvable Pass Rate) - each answer judged Solved/Unsure/Unsolved (1/0.5/0), mean over
    the group's queries, mean over evaluate_times, computed PER TEST GROUP exactly as upstream's
    toolbench/tooleval/eval_pass_rate.py scores each --test_set. HEADLINE `accuracy` = the
    unweighted mean of the per-group SoPRs, the "Average" column of the StableToolBench README
    leaderboard (I1-Inst, I1-Cat, I1-Tool, I2-Cat, I2-Inst, I3-Inst).
  * SoWR (Solvable Win Rate)  - preference vs the README tables' reference ChatGPT-CoT (the
    published baseline answers, stabletoolbench/baselines `chatgpt_cot`), also per group
    (eval_preference.py) with the unweighted mean reported as `sowr`: per pass one comparison
    per test id the run covered (the one Solved answer, else the one Unsure answer, else
    select_better_answer), a win when the candidate takes more passes than the reference.

The judge is gbench's standard Gemini cascade (base.judge_generate_cascade's models, key rotation,
rounds and backoff) in place of gpt-4-turbo; with the simulator's model, the only substitutions.
Everything else of ToolEval is kept: the prompt texts verbatim, the config.yaml function (name,
description, parameters) as the one tool with the call forced (Gemini function calling, mode ANY),
`reason` where the scripts ask for it, the grader's temperature 0.2 and max_tokens 1000, three
tries per call. gpt-4-turbo does not think and the Gemini judge counts its thinking against
max_output_tokens, so the judge is sent 1000 + gemini_budget.GEMINI_THINKING_ALLOWANCE (8192): its
reply keeps upstream's 1000 tokens. A reply that still stops there without the forced call is
counted (`judge_replies_cut_at_max_tokens_by_group`) and the cascade asks its next model, a judge
outage whose cascade saw such a reply is counted in
`judge_outages_after_max_output_tokens_by_group`, and either is named in
leaderboard_comparable_reason. A call that still has no valid verdict is left out of the pass or
comparison (counted, a harness error), never scored Unsolved or as a loss; a query no SoWR pass
could vote on is left out of the SoWR denominator, and a group with no vote publishes no SoWR.
Because the grader differs, a run here is a gbench-internal number rather than a like-for-like
leaderboard entry (leaderboard_comparable=False).

HARD-ERRORS (infra_required, never skips, never a fabricated number) if the StableToolBench
checkout, the cached /virtual server, a Gemini API key (GEMINI_API_KEYS, else GEMINI_API_KEY: the
judge and the simulator), the real-API proxy, the solvable-query data or the SoWR reference
answers are absent, including reference answers that miss any test id the run covers. Heavy
external provisioning - see docs/evals/toolbench.md.

Network and credentials: the container runs the model's tool calls (the DFSDT loop posts each one
to the in-container cached server, which passes an uncached call on to the real-API proxy and the
simulator), so it runs on docker's bridge network after container_guard.require_metadata_blocked
found the metadata server blocked there, and it holds no real key: the simulator's Gemini calls go
through gbench's Gemini relay (gemini_relay.GeminiRelay on the bridge gateway; the container gets
its per-run placeholder by name) and the model requests through gbench's endpoint relay
(swebench_common.EndpointRelay, which adds the endpoint's key, base.endpoint_api_key; besides the
/v1 model routes it forwards the server's root /tokenize, TOKENIZE_ROOT_PATHS, which the image's
request policy asks before each model request); an endpoint URL with user info is refused.
The cascade goes in as GBENCH_JUDGE_MODELS (base.gemini_cascade_env). Container output
quoted in an error is redacted.

Harness errors (never the model's outcome): upstream's server answers HTTP 500 when a call it
could not answer raises (the real-API proxy call or the simulator failing) and upstream's client
hands the model that error as the tool's observation; the simulator's "Failed to generate fake
response" (three replies without valid JSON) reaches the model the same way. The image records
every /virtual call (toolbench_server_calls.jsonl: answer type, the Gemini model that simulated
it, a failure and its stage, the query it belongs to) and each query's failure observations over
its whole search tree (tool_observations.json). A query that met a server error, a failed
simulation, a tool-call timeout, a DFSDT step whose model request failed at the endpoint on every
one of upstream's tries (below), or that has no answer is a harness error: it is left out of SoPR
and SoWR, counted (n_harness_error_queries, harness_error_query_kinds), named in
leaderboard_comparable_reason, and sets the status completed_with_errors; a group whose every
query is one publishes no score. The result records the answering Gemini models
(judge_models_used, simulator_models_used, gemini_models_answered), how the cached server answered
(simulator_usage) and one record per test id (sample_traces).

The delegation CLI is version-sensitive; validate it against your checkout (all paths
env-configurable).

Evaluation count: StableToolBench runs every model once and evaluates it three times, reporting
the average (README "Solvable Pass Rate": "we run all models once, evaluate them three times, and
take the average results"; `--evaluate_times 3` in its run_pass_rate.sh / run_preference.sh). gbench
does the same by default (GBENCH_TOOLBENCH_EVAL_TIMES=3): SoPR per group is the mean of the three
evaluation passes (their spread is reported as `sopr_std_by_group`, the "Std" eval_pass_rate.py
prints), and SoWR takes each query's most frequent verdict over the three passes.

Sampling: StableToolBench's DFSDT loop builds the model-under-test's requests as upstream does
(toolbench/inference/LLM/chatgpt_function_model.py chat_completion_request: max_tokens 1024,
frequency_penalty 0, presence_penalty 0, no temperature). Inside the image, gbench's request
settings (docker/toolbench_request_policy.py, hooked into chat_completion_request) change just
these fields:

  * the chat template's thinking switch (`chat_template_kwargs.enable_thinking`) follows
    `--thinking`;
  * the temperature is gbench's rule for every suite (base.resolve_temperature): 1.0 with
    `--thinking`, 0.0 without (upstream sends none; gbench always sends one). `--temperature`, or
    `GBENCH_TOOLBENCH_TEMPERATURE` for this suite alone (it takes precedence), sets another in
    either mode. It is recorded under `official_sampling` and named in
    `leaderboard_comparable_reason`;
  * max_tokens follows the token-limit policy (base.resolve_token_budget): without `--thinking`
    upstream's 1024 is sent exactly on every turn; with `--thinking` that cap, written for
    non-reasoning generation, is not applied: every turn is sent the run's budget
    (`--max-output-tokens`, else the thinking floor), cut to the room the served context window
    leaves after the prompt;
  * with `--thinking` every turn carries a timeout sized to that budget as gbench's own requests
    are (base.request_timeout_s: GBENCH_REQUEST_TIMEOUT_S + budget / GBENCH_MIN_DECODE_TOK_S
    seconds to read the reply, recorded in `model_request_timeout_s`), in place of the openai
    client's default 600 s, which a reply decoding the whole budget can outlast. Without
    `--thinking` the harness client's own timeout applies.

Replies that stop at max_tokens are counted in `model_replies` (with `--thinking` they are named in
`leaderboard_comparable_reason`). A model-under-test request that ends in an error of the served
endpoint or of the connection (for example a timeout, a 5xx, a 401/404/429, a refused connection)
is a harness error, never the model's outcome: upstream's chat_completion_request turns it into an
error reply, and gbench counts it in `model_replies.failed_requests` (`timed_out`,
`failed_by_error` by exception class and HTTP status) and in `harness_errors`, names it in
`leaderboard_comparable_reason`, and reports the status `completed_with_errors`. A request the
endpoint rejected for its content (HTTP 400 or 422, among them a context-window rejection of the
--thinking budget that could not be fitted) is the model's outcome, as gbench's quality runner
(runners/quality.py) and gaia2 count it (base-run suites, bfcl, mcp_bench and the SWE family count
a 400/422 as a harness error; docs/evals/env-knobs.md): the request carried the model's own
conversation, and
upstream passes the error text on as it would for any rejected request; such requests stay in
`model_replies.failed_requests` (the image's record of every error reply the model client got)
and are split out into `model_content_rejections`; they are not harness errors. Upstream's
ChatGPTFunction.parse asks up to 6 times per DFSDT step; the image records, per query, every step
with a request that failed at the endpoint or was rejected for its content
(model_request_failures.jsonl). A step whose last try failed at the endpoint passed the error text
into the search tree as the model's reply, so its query is a harness error (kind
`model_endpoint`), left out of SoPR and SoWR; a step a later try answered left the search
unchanged, so its query is scored, and the step is counted (`model_request_failures`); a step
whose last try was rejected for its content is the model's outcome, its query scored and the step
counted (`steps_rejected`). A step that failed at the endpoint and that the image could not tie to
a query is infra_required. A run in which not one model request got a reply publishes no score
(infra_required), content rejections included: no request then carried text the model had
generated. Requests the served window rejected where upstream's 1024 was sent, or where the
prompt alone fills the window (`rejected_over_context`, left to upstream's own handling), are
counted and named. The judge follows ToolEval's own settings
(below); its temperature is 0.2, not the 0.0 other gbench judges use, because config.yaml fixes
it.

Selection (`--shard I/N`, `--eval-limit`): gbench selects the queries on the host. It reads each
run group's test ids from the image (solvable_queries/test_query_ids/<group>.json, the set
eval_pass_rate.py and eval_preference.py score), orders every (group, query id) pair (by group,
then id), takes the shard first (base.apply_shard_indexed, round-robin over that order), then
draws `--eval-limit` queries from it stratified by group with the seed "toolbench"
(sampling.stratified_sample; never the first N of each group's file). The selection goes to the
container as a file of query ids per group (TB_QUERY_IDS_FILE; the entrypoint runs exactly those
queries), groups with no selected query are not run, and SoPR/SoWR score exactly the selected ids.
The result records `selected_query_ids`, the shard identity (base.shard_identity) and a partial
shard's comparability reason; an empty shard runs nothing and reports `accuracy` null with
`accuracy_withheld_reason`. merge_shards (gbench --merge) recomputes each group's SoPR (per pass,
then averaged over the passes) and SoWR from the union of the shards' per-query records, and the
headline over the groups. StableToolBench runs every model once (and evaluates it three times):
`--attempt-count` above 1 is recorded as not applied (`attempts.not_applied`).

Model id: the id gbench's eval runner passes (the served id it resolved against --models) goes to
the container as TB_MODEL unchanged; the endpoint's model list is not re-read. The endpoint key
(base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY) is added by gbench's endpoint relay.
"""

import asyncio
import contextvars
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from . import base as _base
from . import container_guard, gemini_relay
from .mcp_bench import (check_request_log, every_model_request_failed,
                        image_applies_request_policy, model_request_failures,
                        read_model_request_log, REQUEST_POLICY_LABEL, REQUEST_POLICY_VERSION)
from .gemini_budget import (GEMINI_THINKING_ALLOWANCE, cutoff_reason, gemini_output_limit,
                            output_limit_record)
from .limit_plan import eval_limit_plan
from .swebench_common import EndpointRelay, infra_required
from .base import (MODELS_NOT_SERVED_PREFIX, REASONING_CAP_NOTE, SHARD_REASON, OfficialSampling,
                   add_comparability_reason, apply_shard_indexed, current_shard,
                   gemini_key_live_valid, gbench_cache_dir, get_run_knob, redact_file,
                   redact_text, request_timeout_s, resolve_official_sampling,
                   resolve_token_budget, shard_identity)
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/toolbench.md"

_GROUPS = ["G1_instruction", "G1_category", "G1_tool",
           "G2_category", "G2_instruction", "G3_instruction"]
_METHOD_DEFAULT = "DFS_woFilter_w2"
#: The SoWR reference of the StableToolBench README tables ("Solvable Win Rate: (Reference model:
#: ChatGPT-CoT)", GPT-3.5-Turbo-0613 CoT): `chatgpt_cot` in the published baseline answers.
_REFERENCE_DEFAULT = "chatgpt_cot"
#: The published baseline answers (README: "The inference results (run in Feb 2024) can be found
#: on Huggingface"): raw answer trees per model and group, converted here with upstream's
#: convert_to_answer_format.py as the README's "Evaluation" section converts any model's answers.
BASELINES_REPO = "stabletoolbench/baselines"
BASELINES_REVISION = "ea70398f955e7b58f8807187936818e794ce5140"
BASELINES_FILE = "data_baselines.zip"
BASELINES_SHA256 = "c3c755f68ccd006788d247eb73d644b4c4ea13620ea4702621d921958a80b79c"
#: StableToolBench evaluates every run three times and averages (README; --evaluate_times 3).
_EVAL_TIMES_DEFAULT = 3

# ToolEval, as StableToolBench's pass-rate and preference scripts run it: the prompts are the
# <description> texts of evaluators/tooleval_gpt-3.5-turbo_default/template.txt, verbatim (the
# newline after <description> and before </description> included), sent as the only user message,
# with the function from config.yaml (name, description, parameters) as the one tool and the call
# forced (tooleval.py function_call: tools=[that function], tool_choice=it). gbench's cascade gets
# the same function through Gemini function calling (mode ANY, only that function allowed) and
# reads the call's arguments. With return_reason the parameters also require `reason`.
_CHECK_ANSWER_STATUS = (
    "\nGiving the query and answer, you need give `answer_status` of the answer by following "
    "rules:\n"
    "1. If the answer is a sorry message or not a positive/straight response for the given query, "
    "return \"Unsolved\".\n"
    "2. If the answer is a positive/straight response for the given query, you have to further check.\n"
    "2.1 If the answer is not sufficient to determine whether the solve the query or not, return "
    "\"Unsure\".\n"
    "2.2 If you are confident that the answer is sufficient to determine whether the solve the query "
    "or not, return \"Solved\" or \"Unsolved\".\n\n"
    "Query:\n{query}\nAnswer:\n{answer}\n\n"
    "Now give your reason in \"content\" and `answer_status` of JSON to `check_answer_status`.\n")

_PARSE_ANSWER_STATUS = (
    "\nGiving the query and the correspond execution detail of an answer, you need give "
    "`answer_status` of the answer by following rules:\n"
    "1. If all 'tool' nodes' message indicate that there are errors happened, return \"Unsolved\"\n"
    "2. If you find the information in the \"final_answer\" is not true/valid according to the "
    "messages in 'tool' nodes, return \"Unsolved\"\n"
    "3. If you are unable to verify the authenticity and validity of the information, return "
    "\"Unsure\"\n"
    "4. If there are 'tool' node in the chain contains successful func calling and those calling "
    "indeed solve the query, return \"Solved\"\n\n"
    "Query:\n{query}\nAnswer:\n{answer}\n\n"
    "Now you are requested to give reason in \"content\" and `answer_status` of JSON to "
    "`parse_answer_status`.\n")

_SELECT_BETTER_ANSWER = (
    "\nQuery:\n{query}\n\nAnswer_0:\n{answer_0}\n\nAnswer_1:\n{answer_1}\n\n"
    "Given above query and answers in JSON format, you must follow the rules to select the "
    "relatively better answer and give the index of the answer **(0 for Answer_0, 1 for "
    "Answer_1)**:\n"
    "1. Compare the value of \"final_answer\" in following aspects:\n"
    "- Informative: whether it contains all necessary information to reply to the query.\n"
    "- Factuality: whether it accurately describes what has been done, and what failed in the end.\n"
    "- Reasoning: If answer does not solve the query, whether gives a detailed and accurate reason "
    "for failure.\n"
    "2. If you cannot determine yet, compare the value of \"answer_details\" in following aspects:\n"
    "- Tool calling costs: calculating the percentage of failed and replicated tools calling.\n"
    "- Running costs: calculating the total tokens T used in execution.\n"
    "- Milestone: calculating the milestone(fixed subtasks) reached in execution.\n"
    "- Exploration: whether tries potential useful tools in execution. Just count times of "
    "successful tool calling with different tools/arguments in execution.\n\n"
    "If you have made your decision, calling `select_better_answer`, else if you cannot determine, "
    "select a random answer.\n")

_TEMPLATES = {"check_answer_status": _CHECK_ANSWER_STATUS,
              "parse_answer_status": _PARSE_ANSWER_STATUS,
              "select_better_answer": _SELECT_BETTER_ANSWER}

_STATUS_ENUM = ("Solved", "Unsolved", "Unsure")

#: config.yaml `functions[*].description`, verbatim.
_FUNCTION_DESCRIPTIONS: Dict[str, str] = {
    "check_answer_status": ("Parse the json answer with layerd nodes and return the answer_status "
                            "about the answer"),
    "parse_answer_status": ("Parse the json answer with layerd nodes and return the answer_status "
                            "about the answer"),
    "select_better_answer": ("Select the better answer with a comprehensive investigation on "
                             "given aspects. You should ignore the impact of the order of "
                             "candidate answers."),
}
#: config.yaml `functions[*].parameters`, verbatim.
_FUNCTION_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "check_answer_status": {"type": "object", "properties": {"answer_status": {
        "type": "string", "enum": ["Unsure", "Unsolved", "Solved"]}}, "required": ["answer_status"]},
    "parse_answer_status": {"type": "object", "properties": {"answer_status": {
        "type": "string", "enum": ["Unsure", "Unsolved", "Solved"]}}, "required": ["answer_status"]},
    "select_better_answer": {"type": "object", "properties": {"index": {
        "type": "number", "description": "The `index` value in the selected better answer."}},
        "required": ["index"]},
}
#: config.yaml completions_kwargs `temperature: 0.2`: the grader samples, which is why the scripts
#: evaluate three times.
_JUDGE_TEMPERATURE = 0.2
#: config.yaml completions_kwargs `max_tokens: 1000`, which tooleval.py function_call forwards with
#: every request: the room upstream's gpt-4-turbo (which does not think) has for the reply.
_JUDGE_MAX_TOKENS = 1000
#: What the Gemini judge is sent as max_output_tokens: the 1000 plus the thinking allowance (its
#: thinking counts against max_output_tokens). A reply that still reaches the limit without the
#: forced call is counted (judge_replies_cut_at_max_tokens_by_group) and handled as any reply
#: without the call: the cascade's next model is asked.
_JUDGE_OUTPUT_LIMIT = gemini_output_limit(_JUDGE_MAX_TOKENS)
#: tooleval.py function_call and rtl.py select_better_answer: @retry(stop=stop_after_attempt(3)).
_JUDGE_ATTEMPTS = 3
#: The ToolEval grader of the published StableToolBench leaderboard.
UPSTREAM_JUDGE_MODEL = "gpt-4-turbo"

#: The cached server's cache-miss path at STB_REF (server/main.py, server/config.yml): a call that
#: is not cached goes to the ToolBench real-API proxy with the README's empty toolbench key (which
#: the proxy answers as unauthorized), then to the simulator: CONFIG['model'] (gpt-4-turbo) at
#: temperature 0, max_tokens 1024, response_format json_object, 3 tries for valid JSON; is_save
#: true caches each new response for the rest of the run. gbench's image keeps all of it and puts
#: its Gemini cascade in place of the model (docker/toolbench_patch_simulator.py).
UPSTREAM_SIMULATOR_MODEL = "gpt-4-turbo"
#: server/main.py fake_response_function_chat's max_tokens: the room gpt-4-turbo (which does not
#: think) has for the simulated JSON.
SIMULATOR_MAX_TOKENS = 1024
#: What the image's simulator sends the Gemini models as max_tokens: the 1024 plus the thinking
#: allowance (docker/toolbench_patch_simulator.py carries a copy of the allowance). A reply that
#: still stops there with content is used as it arrived (as upstream uses a cut reply) and
#: counted; one with no content makes the cascade ask its next model, counted too.
SIMULATOR_OUTPUT_LIMIT = gemini_output_limit(SIMULATOR_MAX_TOKENS)
UPSTREAM_REAL_API_PROXY = "http://8.130.32.149:8080/rapidapi"
#: The simulator's API: the Gemini API's OpenAI-compatible endpoint, reached through gbench's
#: Gemini relay (the api_base docker/toolbench_entrypoint.sh writes into server/config.yml is the
#: relay's URL; the relay sends the call to the Gemini API with the key; nothing overrides it).
SIMULATOR_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
#: The read timeout the patched server gives the real-API proxy call (upstream sends none).
REAL_API_TIMEOUT_S = 120
#: The image label of a harness that records every /virtual call, the model requests that failed
#: at the endpoint or were rejected for their content (HTTP 400/422: not endpoint failures), per
#: query, and the answer trees' failure observations, reads the cascade from GBENCH_JUDGE_MODELS
#: and accepts only relay placeholders (docker/toolbench_patch_simulator.py,
#: toolbench_patch_client.py, toolbench_patch_llm.py, toolbench_observations.py,
#: toolbench_entrypoint.sh), and runs the queries gbench selects by id (TB_QUERY_IDS_FILE);
#: gbench refuses an image without it at this version (an older image counts a 400/422 as an
#: endpoint failure and leaves its query out, or ignores the selection and runs every query).
#: Version 5: the cache-miss simulator is sent upstream's max_tokens 1024 plus the Gemini thinking
#: allowance (SIMULATOR_OUTPUT_LIMIT) and each call record counts the simulator replies still cut
#: there (a version-4 image sends the bare 1024, which the Gemini thinking eats into).
HARNESS_LABEL = "org.gbench.toolbench-harness"
HARNESS_VERSION = "5"
#: Where the host writes the selected query ids ({group: [ids]}) in the run's /out directory.
QUERY_IDS_FILE = "selected_query_ids.json"
#: The docker network the harness container runs on (it runs the model's tool calls).
NETWORK = "bridge"
#: Written by the image: one JSON line per /virtual call (docker/toolbench_patch_simulator.py).
SERVER_CALLS = "toolbench_server_calls.jsonl"
#: Written by the image: per query, the failure observations of its whole search tree.
OBSERVATIONS = "tool_observations.json"
#: Written by the image: one JSON line per DFSDT step with a model request that failed at the
#: endpoint or was rejected for its content (group, query id, tries, the counts of each, whether
#: the step ended on an endpoint failure or a rejection; toolbench_patch_llm.py).
MODEL_FAILURES = "model_request_failures.jsonl"
#: HTTP statuses of a model endpoint reply that reject the request's own content: the model's
#: outcome here, never a harness error, as gbench's quality runner and gaia2 count them
#: (runners/quality.py model_status_is_harness_error, gbench/docker/gaia2_run.py
#: endpoint_failure). Not every suite follows this rule: base-run suites, bfcl, mcp_bench and the
#: SWE family count a 400/422 as a harness error (docs/evals/env-knobs.md).
CONTENT_REJECTION_STATUSES = (400, 422)
#: The kinds of harness error a query can have, in order, as they are named in the result.
QUERY_HARNESS_ERROR_LABELS = (
    ("server_error", "a tool call to the cached server failed on the server's side (HTTP 500: "
                     "the real-API proxy call, the simulator cascade or the server raised)"),
    ("simulator_failed", "the simulator gave no valid JSON in three tries (upstream's \"Failed "
                         "to generate fake response\")"),
    ("tool_call_timeout", "a tool call timed out"),
    ("no_answer", "the harness wrote no answer for the query (its DFSDT run raised)"),
    ("model_endpoint", "a DFSDT step's model request whose last try (of upstream's 6) failed at "
                       "the endpoint, so the error text reached the search tree as the model's "
                       "reply"),
)
#: Characters of a judge reason / final answer kept in the per-query records.
REASON_CHARS = 300
ANSWER_CHARS = 500

#: The model-under-test's request at STB_REF (toolbench/inference/LLM/chatgpt_function_model.py
#: chat_completion_request): max_tokens 1024, frequency_penalty 0 and presence_penalty 0 on every
#: DFSDT turn, no temperature.
UPSTREAM_MAX_TOKENS = 1024
TOOLBENCH_SAMPLING = OfficialSampling(
    reference=("THUNLP-MT/StableToolBench@aa4ed9f toolbench/inference/LLM/"
               "chatgpt_function_model.py chat_completion_request: max_tokens 1024, "
               "frequency_penalty 0, presence_penalty 0, no temperature"),
    send_temperature=False,
    params={"frequency_penalty": 0, "presence_penalty": 0},
    max_tokens=UPSTREAM_MAX_TOKENS,
)
#: Where the image appends the model-under-test's reply counts (one line per inference process).
REQUEST_LOG = "model_requests.jsonl"
#: The routes outside /v1 the model endpoint relay forwards (EndpointRelay root_paths): the
#: server's /tokenize, which docker/toolbench_request_policy.py (RequestPolicy.room) asks for each
#: request's prompt count, at `<server root>/tokenize` (vLLM serves it outside /v1).
TOKENIZE_ROOT_PATHS = ("/tokenize",)
#: Credentials of the runner's environment that never reach the container: the Gemini relay's
#: placeholder takes the Gemini keys' names, the endpoint relay adds the endpoint's key.
_CREDENTIAL_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                    gemini_relay.RELAY_URL_ENV, "OPENAI_API_KEY", "GBENCH_MODEL_API_KEY",
                    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


_IMAGE_DEFAULT = "gbench-toolbench:aa4ed9f4-v5"

def _image() -> str:
    return os.environ.get("GBENCH_TOOLBENCH_IMAGE", _IMAGE_DEFAULT)


def _groups() -> List[str]:
    raw = (os.environ.get("GBENCH_TOOLBENCH_GROUPS") or "").replace(",", " ").split()
    return raw or list(_GROUPS)


def _served_model_id(base_url: str, model_name: str) -> str:
    """The model id the container's OpenAI client sends to the endpoint: `model_name`, the served
    id gbench's eval runner resolved and checked against --models, unchanged. The endpoint's model
    list is not re-read (its first entry may be another model)."""
    return model_name


def check_toolbench_prerequisites() -> Tuple[bool, str]:
    """Docker + the locally-built `gbench-toolbench:aa4ed9f4-v5` image (bundles StableToolBench +
    cache + server) + a Gemini API key (GEMINI_API_KEYS, else GEMINI_API_KEY: the gbench Gemini cascade
    judge and the cache-miss simulator)."""
    import shutil
    image = _image()
    build = (f"Build the harness LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f docker/toolbench.Dockerfile docker\n"
             f"It bundles pinned StableToolBench + the 236MB cache + the cached /virtual server. "
             f"See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    if not image_applies_request_policy(image):
        return False, (f"image {image!r} predates the model request settings (the run's thinking "
                       "switch, temperature, --thinking budget and request timeout, and the count "
                       "of failed requests, docker/toolbench_request_policy.py; label "
                       f"{REQUEST_POLICY_LABEL}={REQUEST_POLICY_VERSION}). Rebuild it. " + build)
    if not image_records_harness_errors(image):
        return False, (f"image {image!r} predates the current harness (label {HARNESS_LABEL}="
                       f"{HARNESS_VERSION}): it would not record the tool calls the cached server "
                       "failed or the simulator could not answer, or the queries whose model "
                       "request failed at the endpoint (so they would be scored as the model's "
                       "outcome), it would count a request the endpoint rejected for its content "
                       "(HTTP 400/422, the model's outcome) as an endpoint failure and leave its "
                       "query out, and it may expect a real Gemini key where the container now "
                       "gets only the Gemini relay's placeholder, and its simulator would get "
                       "upstream's bare max_tokens 1024, which the Gemini model's thinking eats "
                       "into, without counting the replies cut there. Rebuild it. " + build)
    keys = _base.gemini_api_keys()
    if not keys:
        return False, ("neither GEMINI_API_KEYS nor GEMINI_API_KEY is set - a Gemini API key powers "
                       "gbench's Gemini cascade judge that scores ToolEval SoPR/SoWR here and the "
                       "cache-miss API simulator.")
    _ok, _why = gemini_key_live_valid(",".join(keys))
    if not _ok:
        if str(_why).startswith(MODELS_NOT_SERVED_PREFIX):
            return False, (f"{redact_text(_why)}; every model of the Gemini cascade must be one "
                           "the API serves for the ToolEval judge and the simulator to run.")
        return False, (f"the Gemini API key(s) (GEMINI_API_KEYS, else GEMINI_API_KEY) were rejected "
                       f"by the Gemini endpoint ({redact_text(_why)}); a valid key is required for "
                       "the Gemini cascade judge and simulator (fail-fast before the container "
                       "run).")
    return real_api_proxy_reachable()


def image_records_harness_errors(image: str) -> bool:
    """True when the image carries HARNESS_LABEL=HARNESS_VERSION."""
    r = subprocess.run(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "' + HARNESS_LABEL + '"}}', image],
                       capture_output=True, text=True)
    return r.returncode == 0 and (r.stdout or "").strip() == HARNESS_VERSION


def real_api_proxy_reachable(timeout_s: float = 20.0) -> Tuple[bool, str]:
    """The ToolBench real-API proxy (server/config.yml toolbench_url) answers. The cached server
    asks it for every call it has not cached, before simulating one; an unreachable proxy makes
    each such call end in a server error instead, so it is a prerequisite. Any HTTP reply counts
    (with the README's empty key it answers that the caller is not authorized)."""
    import urllib.error
    import urllib.request
    body = json.dumps({"category": "", "tool_name": "", "api_name": "", "tool_input": {},
                       "strip": "", "toolbench_key": ""}).encode()
    req = urllib.request.Request(UPSTREAM_REAL_API_PROXY, data=body, method="POST", headers={
        "accept": "application/json", "Content-Type": "application/json", "toolbench_key": ""})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read(4096)
    except urllib.error.HTTPError:
        pass                                  # it answered
    except Exception as e:                    # noqa: BLE001 - DNS / refused / timeout
        return False, (f"the ToolBench real-API proxy {UPSTREAM_REAL_API_PROXY} (StableToolBench "
                       "server/config.yml toolbench_url) is not reachable "
                       f"({type(e).__name__}: {e}). The cached server asks it for every call it "
                       "has not cached before simulating one, so without it those calls end in "
                       "a server error. Provide network egress to it.")
    return True, ""


# --------------------------------------------------------------------------- #
# ToolEval calls through the Gemini cascade
# --------------------------------------------------------------------------- #
def _function_schema(func: str, return_reason: bool) -> Dict[str, Any]:
    """The function's parameters, with `reason` added as tooleval.py does for return_reason."""
    schema = json.loads(json.dumps(_FUNCTION_PARAMETERS[func]))
    if return_reason:
        schema["required"].append("reason")
        schema["properties"]["reason"] = {"type": "string", "description": "explain your answer."}
    return schema


def _function_declaration(func: str, return_reason: bool) -> Dict[str, Any]:
    """The tool tooleval.py sends: config.yaml's function (name, description, parameters)."""
    return {"name": func, "description": _FUNCTION_DESCRIPTIONS[func],
            "parameters": _function_schema(func, return_reason)}


def _judge_generation_config(func: str, return_reason: bool) -> Any:
    """ToolEval's request as a Gemini config: the one function as the only tool, the call forced
    (tool_choice -> function calling mode ANY with only that function allowed), and config.yaml's
    completions_kwargs (temperature 0.2, max_tokens 1000), sent as max_output_tokens
    _JUDGE_OUTPUT_LIMIT (1000 + the Gemini thinking allowance)."""
    try:
        from google.genai import types
    except ImportError:
        return None
    decl = _function_declaration(func, return_reason)
    return types.GenerateContentConfig(
        temperature=_JUDGE_TEMPERATURE,
        max_output_tokens=_JUDGE_OUTPUT_LIMIT,
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
            name=decl["name"], description=decl["description"],
            parameters_json_schema=decl["parameters"])])],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=[func])),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))


def _call_arguments(resp: Any, func: str) -> Optional[str]:
    """The arguments of the reply's call to `func`, as JSON text; None without such a call."""
    for call in getattr(resp, "function_calls", None) or []:
        if getattr(call, "name", None) == func:
            return json.dumps(dict(getattr(call, "args", None) or {}))
    return None


def _cut_at_max_tokens(resp: Any) -> bool:
    """Did the reply stop at the output-token limit (finish reason MAX_TOKENS)?"""
    for cand in getattr(resp, "candidates", None) or []:
        reason = getattr(cand, "finish_reason", None)
        if reason is not None and str(getattr(reason, "name", reason)).endswith("MAX_TOKENS"):
            return True
    return False


def _forced_function(config: Any) -> Optional[str]:
    """The function name a _judge_generation_config config forces."""
    try:
        return config.tool_config.function_calling_config.allowed_function_names[0]
    except (AttributeError, IndexError, TypeError):
        return None


async def _tooleval_cascade(prompt: str, config: Any = None) -> Tuple[Optional[str], str]:
    """One ToolEval request on the gbench judge cascade: base.judge_generate_cascade's models, key
    rotation, rounds and backoff, reading the forced call's arguments instead of the reply text.
    `(arguments JSON, model)`, or `(None, "JUDGE_OUTAGE")` when every model fails every round (a
    reply without the forced call counts as that model failing, as an empty reply does there)."""
    if config is None or not _base.gemini_api_keys():   # google-genai missing, or no key
        return None, "JUDGE_OUTAGE"
    func = _forced_function(config)
    if func is None:
        raise ValueError("toolbench: a ToolEval request needs the config _judge_generation_config "
                         "builds (one forced function)")
    try:
        if not _base._judge_clients():
            return None, "JUDGE_OUTAGE"
    except ImportError:
        return None, "JUDGE_OUTAGE"
    rounds = _base._judge_cascade_rounds()
    verdicts = _VERDICTS.get()
    models = _base.judge_cascade()
    not_served = set()
    cut_seen = False      # a reply of this call stopped at max_output_tokens without the call
    for rnd in range(rounds):
        client = _base._next_judge_client()
        for model in models:
            if model in not_served:
                continue
            try:
                resp = await client.aio.models.generate_content(
                    model=model, contents=prompt, config=config)
                args = _call_arguments(resp, func)
                if args is not None:
                    _base.record_gemini_answer(model)
                    return args, model
                if _cut_at_max_tokens(resp):
                    cut_seen = True
                    if verdicts is not None:
                        verdicts.max_tokens_cut += 1
            except Exception as e:  # noqa: BLE001 - 429/network/timeout: move to next model
                if _not_served(e):
                    not_served.add(model)   # HTTP 404: not asked again in this call
                logger.debug("toolbench judge model %s failed: %s", model, e)
        if len(not_served) >= len(models):
            break
        if rnd < rounds - 1:
            await asyncio.sleep(_base._JUDGE_BACKOFF * (2 ** rnd)
                                + random.uniform(0, _base._JUDGE_BACKOFF))
    if cut_seen and verdicts is not None:
        verdicts.outages_at_cap += 1
    return None, "JUDGE_OUTAGE"


def _not_served(exc: BaseException) -> bool:
    """True for the Gemini API's HTTP 404 (the model is not served for the key)."""
    for attr in ("code", "status_code"):
        if getattr(exc, attr, None) == 404:
            return True
    return False


def _parse_function_args(func: str, text: Optional[str], return_reason: bool
                         ) -> Optional[Dict[str, Any]]:
    """The reply's arguments object when it carries the fields upstream reads, with a valid
    value; else None.

    tooleval.py's own required-key check never runs (it reads `required` with getattr on a dict,
    which is always None), so what a reply needs is what rtl.py reads from it: `answer_status`
    (an AnswerStatus value) for check_answer_status / parse_answer_status, and with return_reason
    also `reason` from the reply whose status check_is_solved returns (not from a
    check_answer_status reply that says Unsure, which is followed by parse_answer_status); `index`
    (int) for select_better_answer, whose `reason` rtl.py never reads."""
    if not text:
        return None
    try:
        ret = json.loads(text)
    except ValueError:
        m = re.search(r"\{[\s\S]*\}", text)
        try:
            ret = json.loads(m.group(0)) if m else None
        except ValueError:
            ret = None
    if not isinstance(ret, dict):
        return None
    for key in _FUNCTION_PARAMETERS[func]["required"]:
        if key not in ret:
            return None
    if (return_reason and func in ("check_answer_status", "parse_answer_status")
            and "reason" not in ret
            and not (func == "check_answer_status" and ret.get("answer_status") == "Unsure")):
        return None
    if "answer_status" in _FUNCTION_PARAMETERS[func]["properties"]:
        if ret.get("answer_status") not in _STATUS_ENUM:
            return None
    elif func == "select_better_answer":
        try:
            ret["index"] = int(ret["index"])
        except (TypeError, ValueError):
            return None
    return ret


class _Verdicts:
    """Counts of ToolEval calls that ended without a usable verdict, and of the Gemini models
    that answered the others."""

    def __init__(self):
        self.outages = 0      # the judge cascade was exhausted
        self.invalid = 0      # _JUDGE_ATTEMPTS replies without a valid arguments object
        self.max_tokens_cut = 0   # replies that stopped at _JUDGE_OUTPUT_LIMIT without the call
        self.outages_at_cap = 0   # judge outages whose cascade saw such a reply
        self.models: Dict[str, int] = {}   # ToolEval replies by the Gemini model that answered


#: The counts of the group being scored (set in _score_all; tasks it starts share the object).
_VERDICTS: "contextvars.ContextVar[Optional[_Verdicts]]" = contextvars.ContextVar(
    "toolbench_verdicts", default=None)
#: The ToolEval calls of the verdict being taken (one query's pass or comparison): _score_all
#: sets a fresh list, every _tooleval call it makes appends its record.
_CALL_LOG: "contextvars.ContextVar[Optional[List[Dict[str, Any]]]]" = contextvars.ContextVar(
    "toolbench_call_log", default=None)


def _log_call(record: Dict[str, Any]) -> None:
    log = _CALL_LOG.get()
    if log is not None:
        log.append(record)


async def _tooleval(func: str, args: Dict[str, str], sem: asyncio.Semaphore, *,
                    return_reason: bool, accept: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """One ToolEval function call: the arguments object, or None (outage / no valid reply).

    Up to _JUDGE_ATTEMPTS replies are asked for, as tooleval.py retries a reply that is not the
    function's arguments; `accept` adds a check of its own (select_better_answer's index)."""
    prompt = _TEMPLATES[func].format(**args)
    config = _judge_generation_config(func, return_reason)
    verdicts = _VERDICTS.get()
    models: List[str] = []
    for attempt in range(1, _JUDGE_ATTEMPTS + 1):
        async with sem:
            text, used = await _tooleval_cascade(prompt, config=config)
        if text is None:
            if verdicts is not None:
                verdicts.outages += 1
            _log_call({"function": func, "outcome": "judge_outage", "attempts": attempt,
                       "models": models})
            return None
        models.append(used)
        if verdicts is not None:
            verdicts.models[used] = verdicts.models.get(used, 0) + 1
        ret = _parse_function_args(func, text, return_reason)
        if ret is not None and (accept is None or accept(ret)):
            rec = {"function": func, "outcome": "verdict", "model": used, "attempts": attempt}
            for key in ("answer_status", "index"):
                if key in ret:
                    rec[key] = ret[key]
            if isinstance(ret.get("reason"), str):
                rec["reason"] = ret["reason"][:REASON_CHARS]
            _log_call(rec)
            return ret
    if verdicts is not None:
        verdicts.invalid += 1
    _log_call({"function": func, "outcome": "invalid_reply", "attempts": _JUDGE_ATTEMPTS,
               "models": models})
    return None


def _final_step_has_finish(answer: Dict[str, Any]) -> bool:
    """Walk the answer_details chain; True iff the last tool node is a `Finish` call (ToolEval:
    the driver requires "'name': 'Finish'" in the final step, else auto-Unsolved)."""
    details = answer.get("answer_details") or []
    node = details[0] if details else None
    last_tool = None
    seen = 0
    while isinstance(node, dict) and seen < 1000:
        seen += 1
        if node.get("role") == "tool":
            last_tool = node.get("message")
        nxt = node.get("next") or []
        node = nxt[0] if nxt else None
    return "'name': 'Finish'" in str(last_tool) if last_tool is not None else False


def _process_answer_for_pref(answer: Dict[str, Any]) -> Dict[str, Any]:
    """base.py process_answer: final_answer[:1000], answer_details[:3000], drop `method`.
    (Upstream mutates the answer in place before any preference call, so the statuses computed
    for a comparison see the processed answer too.)"""
    a = dict(answer)
    a.pop("method", None)
    fa = a.get("final_answer")
    if isinstance(fa, str):
        a["final_answer"] = fa[:1000]
    details = a.get("answer_details")
    if isinstance(details, (list, str)):
        a["answer_details"] = details[:3000]
    return a


async def _check_is_solved(query: str, answer: Dict[str, Any], sem: asyncio.Semaphore, *,
                           return_reason: bool) -> Optional[str]:
    """rtl.py check_is_solved: an empty or give-up final answer is Unsolved; else
    check_answer_status, and parse_answer_status on the whole answer when that says Unsure.
    None when a call has no usable verdict (never scored as Unsolved)."""
    final_answer = str(answer.get("final_answer") or "")
    if final_answer == "" or "give_up_and_restart" in final_answer:
        return "Unsolved"
    ret = await _tooleval("check_answer_status", {"query": query, "answer": final_answer}, sem,
                          return_reason=return_reason)
    if ret is None:
        return None
    status = ret["answer_status"]
    if status == "Unsure":
        ret = await _tooleval("parse_answer_status",
                              {"query": query, "answer": json.dumps(answer)}, sem,
                              return_reason=return_reason)
        if ret is None:
            return None
        status = ret["answer_status"]
    return status


async def _answer_status(query: str, answer: Dict[str, Any], sem: asyncio.Semaphore
                         ) -> Optional[str]:
    """eval_pass_rate.py compute_pass_rate: Unsolved when the final step is not a Finish call,
    else check_is_solved with return_reason=True. None = no usable verdict (excluded)."""
    if not _final_step_has_finish(answer):
        return "Unsolved"
    return await _check_is_solved(query, answer, sem, return_reason=True)


async def _preference_status(query: str, answer: Dict[str, Any], sem: asyncio.Semaphore
                             ) -> Optional[str]:
    """The status eval_preference.py's comparison computes: check_is_solved on the processed
    answer, without the pass-rate script's Finish check."""
    return await _check_is_solved(query, answer, sem, return_reason=False)


_STATUS_SCORE = {"Solved": 1.0, "Unsure": 0.5, "Unsolved": 0.0}


def _identical(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """rtl.py check_identity_answers: same final_answer and same str(answer_details)."""
    return (a.get("final_answer") == b.get("final_answer")
            and str(a.get("answer_details")) == str(b.get("answer_details")))


async def _select_better(query: str, ans0: Dict[str, Any], ans1: Dict[str, Any],
                         sem: asyncio.Semaphore) -> Optional[int]:
    """rtl.py select_better_answer on two (processed) answers: 0 or 1, None without a verdict."""
    ret = await _tooleval("select_better_answer",
                          {"query": query, "answer_0": json.dumps(ans0),
                           "answer_1": json.dumps(ans1)}, sem, return_reason=True,
                          accept=lambda r: r["index"] in (0, 1))
    return None if ret is None else ret["index"]


async def _preference(query: str, cand: Dict[str, Any], ref: Dict[str, Any],
                      sem: asyncio.Semaphore) -> Optional[str]:
    """One eval_preference.py comparison: "cand", "ref", or None without a verdict.

    base.py annotate_preference shuffles the two answers, then rtl.py
    normalized_openai_completions picks: the one Solved answer if exactly one is; else the one
    Unsure answer if exactly one is; else select_better_answer between the two (a random one when
    they are identical). The check_task_solvable call upstream makes before select_better_answer
    is not made: its result is not part of the select_better_answer prompt, so it cannot change
    the choice."""
    pair = [("ref", _process_answer_for_pref(ref)), ("cand", _process_answer_for_pref(cand))]
    random.shuffle(pair)
    statuses = await asyncio.gather(*[_preference_status(query, a, sem) for _, a in pair])
    order = [name for name, _ in pair]

    def _chose(choice: Optional[str], how: str) -> Optional[str]:
        _log_call({"function": "preference", "order": order, "statuses": list(statuses),
                   "choice": choice, "decided_by": how})
        return choice

    if any(st is None for st in statuses):
        return _chose(None, "no status verdict")
    for wanted in ("Solved", "Unsure"):
        hits = [i for i, st in enumerate(statuses) if st == wanted]
        if len(hits) == 1:
            return _chose(pair[hits[0]][0], f"the one {wanted} answer")
        if len(hits) > 1:
            break
    if _identical(pair[0][1], pair[1][1]):
        return _chose(pair[random.randint(0, 1)][0], "identical answers: random")
    idx = await _select_better(query, pair[0][1], pair[1][1], sem)
    return _chose(None if idx is None else pair[idx][0], "select_better_answer")


def _download_baselines() -> str:
    """The pinned data_baselines.zip (sha256-checked); raises infra_required when unavailable."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(BASELINES_REPO, filename=BASELINES_FILE, repo_type="dataset",
                               revision=BASELINES_REVISION)
    except Exception as e:  # noqa: BLE001 - network / hub errors
        raise infra_required(
            "toolbench", f"could not download the SoWR reference answers {BASELINES_REPO}@"
            f"{BASELINES_REVISION[:12]}/{BASELINES_FILE} ({type(e).__name__}: {e}); or set "
            "GBENCH_TOOLBENCH_REFERENCE_DIR to converted reference answers", DOCS_URL) from e
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != BASELINES_SHA256:
        raise infra_required(
            "toolbench", f"{path} has sha256 {digest.hexdigest()}, not the pinned "
            f"{BASELINES_SHA256}; delete it and re-run", DOCS_URL)
    return path


def _tail(text: Optional[str], n: int) -> str:
    """The last `n` characters of a process's output for an error message, with every credential
    value of the environment redacted."""
    return redact_text(text or "")[-n:]


def _missing_ids(answers: Dict[str, Dict[str, Any]], ids: List[str]) -> List[str]:
    """The ids in `ids` that `answers` holds no answer for."""
    return [q for q in ids if q not in answers]


def _baseline_reference_root(reference: str, groups: List[str], image: str) -> str:
    """A directory holding `<reference>/<group>.json` for every group: the published baseline
    answers of `reference`, converted once with the image's convert_to_answer_format.py (as the
    README converts answers before eval_preference.py) and cached under GBENCH_CACHE_DIR.

    A group's file counts (cached or freshly converted) only when it holds an answer for every
    one of the group's test ids (solvable_queries/test_query_ids in the image); a conversion that
    leaves any test id without an answer is infra_required and nothing is cached."""
    import zipfile
    root = os.path.join(gbench_cache_dir(), "toolbench_baselines", BASELINES_REVISION[:12],
                        "converted")
    out_dir = os.path.join(root, reference)
    full_ids = _group_test_ids(image, groups, None)
    todo = [g for g in groups
            if not os.path.isfile(os.path.join(out_dir, f"{g}.json"))
            or not _load_group_file(os.path.join(out_dir, f"{g}.json"))
            or _missing_ids(_load_group_file(os.path.join(out_dir, f"{g}.json")), full_ids[g])]
    if not todo:
        return root
    zip_path = _download_baselines()
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.startswith("data_baselines/")
                 and n.endswith(".json") and n.count("/") == 3]
        models = sorted({n.split("/")[1] for n in names})
        if reference not in models:
            raise infra_required(
                "toolbench", f"GBENCH_TOOLBENCH_REFERENCE={reference!r} is not one of the "
                f"published baselines ({', '.join(models)}); set GBENCH_TOOLBENCH_REFERENCE_DIR "
                "for another reference", DOCS_URL)
        work = tempfile.mkdtemp(prefix="gbench_toolbench_ref_")
        try:
            os.chmod(work, 0o777)   # the container writes the converted files as its own uid
            for g in todo:
                members = [n for n in names if n.split("/")[1] == reference
                           and n.split("/")[2] == g]
                methods = sorted({os.path.basename(n)[:-len(".json")].split("_", 1)[1]
                                  for n in members if "_" in os.path.basename(n)})
                if not members or len(methods) != 1:
                    raise infra_required(
                        "toolbench", f"the published {reference} answers for {g} are missing or "
                        f"mix methods ({methods}) in {BASELINES_FILE}", DOCS_URL)
                src = os.path.join(work, "in", g)
                os.makedirs(src, exist_ok=True)
                for n in members:
                    with open(os.path.join(src, os.path.basename(n)), "wb") as fh:
                        fh.write(zf.read(n))
                cmd = ["docker", "run", "--rm", "--pull", "never", "--network", "none",
                       "-v", f"{work}:/ref:rw",
                       "-w", "/stb/toolbench/tooleval", "--entrypoint", "python", image,
                       "convert_to_answer_format.py", "--answer_dir", f"/ref/in/{g}",
                       "--method", methods[0], "--output", f"/ref/{g}.json"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                converted = os.path.join(work, f"{g}.json")
                if proc.returncode != 0 or not os.path.isfile(converted):
                    raise infra_required(
                        "toolbench", f"converting the published {reference} answers for {g} "
                        f"failed (rc={proc.returncode}): "
                        f"{_tail(proc.stderr or proc.stdout, 400)}", DOCS_URL)
                missing = _missing_ids(_load_group_file(converted), full_ids[g])
                if missing:
                    raise infra_required(
                        "toolbench", f"converting the published {reference} answers for {g} left "
                        f"{len(missing)} of the group's {len(full_ids[g])} test ids without an "
                        f"answer (e.g. {', '.join(missing[:5])}); nothing was cached", DOCS_URL)
                os.makedirs(out_dir, exist_ok=True)
                tmp = os.path.join(out_dir, f".{g}.json.{os.getpid()}")
                shutil.copyfile(converted, tmp)
                os.replace(tmp, os.path.join(out_dir, f"{g}.json"))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return root


def _load_group_file(path: str) -> Dict[str, Dict[str, Any]]:
    """One converted-answers file, {qid: {query, available_tools, answer}}. {} if absent."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(q): v for q, v in data.items() if isinstance(v, dict) and "answer" in v}


#: Reads, in the image, each group's test ids (solvable_queries/test_query_ids/<group>.json, the
#: set eval_pass_rate.py and eval_preference.py take from --test_ids) and the query order of
#: test_instruction/<group>.json (the file the entrypoint runs, or filters by id).
_TEST_IDS_PROBE = (
    "import json, sys\n"
    "root = '/stb/solvable_queries'\n"
    "out = {}\n"
    "for g in sys.argv[1:]:\n"
    "    ids = [str(i) for i in json.load(open(root + '/test_query_ids/' + g + '.json'))]\n"
    "    order = [str(q['query_id']) for q in json.load(open(root + '/test_instruction/' + g "
    "+ '.json'))]\n"
    "    out[g] = {'test_ids': ids, 'instruction_order': order}\n"
    "print('TB_TEST_IDS ' + json.dumps(out))\n"
)


def _group_test_ids(image: str, groups: List[str], per_group_limit: Optional[int]
                    ) -> Dict[str, List[str]]:
    """{group: its test ids}: test_query_ids/<group>.json, among the queries of
    test_instruction/<group>.json (the file the entrypoint runs); eval_preference.py iterates
    exactly these ids. run_toolbench passes `per_group_limit` None and selects a subset itself
    (select_query_ids); a number keeps only the first that many queries of the file."""
    cmd = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint",
           "python", image, "-c",
           _TEST_IDS_PROBE, *groups]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise infra_required("toolbench", "reading the test query ids from the image did not "
                             "finish within 300 s", DOCS_URL) from exc
    line = next((ln for ln in (proc.stdout or "").splitlines() if ln.startswith("TB_TEST_IDS ")),
                None)
    try:
        data = json.loads(line[len("TB_TEST_IDS "):]) if line else None
    except ValueError:
        data = None
    if proc.returncode != 0 or not isinstance(data, dict) or set(data) != set(groups):
        raise infra_required(
            "toolbench", f"could not read solvable_queries/test_query_ids from the image "
            f"(rc={proc.returncode}); rebuild it from docker/toolbench.Dockerfile. tail: "
            f"{_tail(proc.stderr or proc.stdout, 400)}", DOCS_URL)
    out: Dict[str, List[str]] = {}
    for g in groups:
        order = data[g]["instruction_order"]
        ran = set(order if per_group_limit is None else order[:per_group_limit])
        out[g] = [q for q in data[g]["test_ids"] if q in ran]
    return out


def _qid_order(qid: str) -> Tuple[int, Any]:
    """Sort key of a query id: numeric ids by value, others after them by text."""
    return (0, int(qid)) if str(qid).isdigit() else (1, str(qid))


def select_query_ids(test_ids: Dict[str, List[str]], limit: Optional[int]
                     ) -> Tuple[Dict[str, List[str]], Optional[str], int]:
    """({group: the selected test ids, in the order of test_ids[group]}, the shard label or None,
    the number of (group, id) pairs in the full ordered list): every pair ordered by group, then
    id; the shard first (base.apply_shard_indexed); then `limit` pairs drawn stratified by group
    with the seed "toolbench" (sampling.stratified_sample)."""
    pairs = sorted(((g, str(q)) for g, ids in test_ids.items() for q in ids),
                   key=lambda p: (p[0], _qid_order(p[1])))
    selected, label, _positions = apply_shard_indexed(pairs, "toolbench")
    if limit and int(limit) > 0 and len(selected) > int(limit):
        selected = stratified_sample(selected, int(limit), key_fn=lambda p: p[0],
                                     seed="toolbench")
    chosen = set(selected)
    return ({g: [str(q) for q in ids if (g, str(q)) in chosen] for g, ids in test_ids.items()},
            label, len(pairs))


def _reference_gaps(ref_by_group: Dict[str, Dict[str, Dict[str, Any]]],
                    test_ids: Dict[str, List[str]]) -> List[str]:
    """Groups whose reference answers do not cover the test ids the run covers, each with why.
    Empty when SoWR can take its vote on every test id against a reference answer."""
    gaps = []
    for g, ids in test_ids.items():
        answers = ref_by_group.get(g) or {}
        if not answers:
            gaps.append(f"{g} (no {g}.json, or one with no answers)")
            continue
        missing = _missing_ids(answers, ids)
        if missing:
            gaps.append(f"{g} ({len(missing)} of the {len(ids)} test ids the run covers have no "
                        f"reference answer, e.g. {', '.join(missing[:5])})")
    return gaps


def _load_groups(root: str, groups: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{group: {qid: answer}} from `<root>/<group>.json`, the layout upstream's converter writes
    and eval_pass_rate.py reads. Groups are kept apart (no qid merge across groups), as upstream
    scores each --test_set on its own."""
    return {g: _load_group_file(os.path.join(root, f"{g}.json")) for g in groups}


async def _score_all(candidate: Dict[str, Dict[str, Any]],
                     reference: Dict[str, Dict[str, Any]],
                     evaluate_times: int, concurrency: int,
                     sem: Optional[asyncio.Semaphore] = None,
                     test_ids: Optional[List[str]] = None,
                     excluded: Optional[Any] = None) -> Dict[str, Any]:
    """Compute SoPR (candidate) + SoWR (candidate vs reference) for ONE test group.

    `test_ids`: the group's test ids the run covers (_group_test_ids). As in eval_pass_rate.py,
    SoPR scores the candidate's queries among them; as in eval_preference.py, SoWR takes one vote
    per test id. Without it (direct callers) both use the queries the two files hold, which on a
    full run with a complete reference are the same ids. `excluded`: the ids that are harness
    errors (run_toolbench: a tool call the server failed, a failed simulation, no answer); they
    are left out of both scores. The result's `records` hold, per scored id, the verdict of
    every pass and every comparison with the ToolEval calls behind it."""
    sem = sem or asyncio.Semaphore(max(1, concurrency))
    verdicts = _Verdicts()
    _VERDICTS.set(verdicts)
    skip = set(excluded or ())
    if test_ids is not None:
        wanted = set(test_ids)
        candidate = {q: v for q, v in candidate.items() if q in wanted}
    candidate = {q: v for q, v in candidate.items() if q not in skip}
    records: Dict[str, Dict[str, Any]] = {}
    for q, v in candidate.items():
        answer = v.get("answer") or {}
        records[q] = {"query_id": q, "final_answer": str(answer.get("final_answer") or "")[
            :ANSWER_CHARS], "finish_check": _final_step_has_finish(answer),
            "sopr_passes": [], "sowr_rounds": []}

    # --- SoPR (eval_pass_rate.py): per pass, the mean over queries of {1/0.5/0}; the passes are
    # averaged. A query without a usable verdict is left out of that pass.
    async def _sopr_run(npass: int) -> Optional[float]:
        async def _one(q, v):
            log: List[Dict[str, Any]] = []
            _CALL_LOG.set(log)
            st = await _answer_status(v.get("query", ""), v.get("answer") or {}, sem)
            records[q]["sopr_passes"].append({"pass": npass, "status": st, "calls": log})
            return None if st is None else _STATUS_SCORE.get(st, 0.0)
        scores = await asyncio.gather(*[_one(q, v) for q, v in candidate.items()])
        kept = [s for s in scores if s is not None]
        return (sum(kept) / len(kept)) if kept else None

    sopr_runs = [r for r in await asyncio.gather(*[_sopr_run(i) for i in range(evaluate_times)])
                 if r is not None]
    sopr = (sum(sopr_runs) / len(sopr_runs)) if sopr_runs else None
    # eval_pass_rate.py prints np.std over the per-pass solve rates (population std).
    sopr_std = ((sum((r - sopr) ** 2 for r in sopr_runs) / len(sopr_runs)) ** 0.5
                if sopr_runs else None)
    for rec in records.values():
        rec["sopr_passes"].sort(key=lambda p: p["pass"])

    # --- SoWR (eval_preference.py): per pass, one comparison per test id; a query is a win when
    # the candidate got more of the passes' votes than the reference (ties and losses are not). A
    # query only the reference answered is the reference's vote, one only the candidate answered
    # (or neither) is the candidate's, as upstream. A comparison without a verdict (a judge outage
    # or no valid reply) is no vote: upstream's script stops rather than go on without one, so
    # the query's other passes decide, and a query no pass could vote on is left out of the
    # denominator (counted in sowr_queries_without_vote).
    sowr = None
    votes_missing = 0
    without_vote: List[str] = []
    if reference:
        qids = (list(test_ids) if test_ids is not None
                else sorted(set(candidate) | set(reference), key=str))
        qids = [q for q in qids if q not in skip]

        async def _vote(q, nround) -> Optional[str]:
            if q not in reference:
                return "cand"
            if q not in candidate:
                return "ref"
            log: List[Dict[str, Any]] = []
            _CALL_LOG.set(log)
            vote = await _preference(candidate[q].get("query", "")
                                     or reference[q].get("query", ""),
                                     candidate[q].get("answer") or {},
                                     reference[q].get("answer") or {}, sem)
            records[q]["sowr_rounds"].append({"round": nround, "vote": vote, "calls": log})
            return vote

        async def _round(nround) -> Dict[str, Optional[str]]:
            votes = await asyncio.gather(*[_vote(q, nround) for q in qids])
            return dict(zip(qids, votes))

        rounds = await asyncio.gather(*[_round(i) for i in range(evaluate_times)])
        wins = voted = 0
        for q in qids:
            got = [rd.get(q) for rd in rounds]
            votes_missing += sum(1 for v in got if v is None)
            valid = [v for v in got if v is not None]
            if not valid:
                without_vote.append(q)
                continue
            voted += 1
            if valid.count("cand") > valid.count("ref"):
                wins += 1
        sowr = (wins / voted) if voted else None
        for rec in records.values():
            rec["sowr_rounds"].sort(key=lambda r: r["round"])

    return {"sopr": sopr, "sopr_std": sopr_std, "sopr_passes": len(sopr_runs), "sowr": sowr,
            "n_candidate": len(candidate), "n_reference": len(reference),
            "judge_outages": verdicts.outages, "judge_invalid_replies": verdicts.invalid,
            "judge_replies_cut_at_max_tokens": verdicts.max_tokens_cut,
            "judge_outages_after_max_output_tokens": verdicts.outages_at_cap,
            "judge_models_used": dict(sorted(verdicts.models.items())),
            "sowr_votes_missing": votes_missing,
            "sowr_queries_without_vote": without_vote,
            "solved_equivalent": ((sopr or 0.0) * len(candidate) if sopr is not None else None),
            "records": records}


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


async def _score_groups(cand_by_group: Dict[str, Dict[str, Dict[str, Any]]],
                        ref_by_group: Dict[str, Dict[str, Dict[str, Any]]],
                        evaluate_times: int, concurrency: int,
                        test_ids: Optional[Dict[str, List[str]]] = None,
                        excluded: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score every group separately, then average the groups with equal weight.

    StableToolBench reports SoPR/SoWR per test group plus their unweighted "Average"; pooling
    every query instead would weight the groups by size (G3_instruction is the smallest).
    `excluded`: {group: ids that are harness errors}, left out of the group's scores.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    groups = list(cand_by_group)
    per = await asyncio.gather(*[
        _score_all(cand_by_group[g], ref_by_group.get(g) or {}, evaluate_times, concurrency, sem,
                   test_ids=test_ids.get(g) if test_ids is not None else None,
                   excluded=(excluded or {}).get(g))
        for g in groups])
    by_group = dict(zip(groups, per))
    sopr = {g: r["sopr"] for g, r in by_group.items()}
    sowr = {g: r["sowr"] for g, r in by_group.items()}
    return {
        "by_group": by_group,
        # An average over only some groups would be a different (partial) number: None instead.
        "sopr": _mean(list(sopr.values())) if all(v is not None for v in sopr.values()) else None,
        "sowr": _mean(list(sowr.values())) if all(v is not None for v in sowr.values()) else None,
        "n_candidate": sum(r["n_candidate"] for r in per),
    }


# --------------------------------------------------------------------------- #
# The harness container's network, relays, and records
# --------------------------------------------------------------------------- #
def _bridge_gateway(image: str) -> str:
    """The gateway of NETWORK, where gbench's relays listen, after container_guard found the
    cloud metadata server blocked on that network (the container runs the model's tool calls).
    The container reaches the real-API proxy by IP address and the relays by the gateway's
    address, so no host name has to resolve."""
    container_guard.require_metadata_blocked("toolbench", image, NETWORK, DOCS_URL)
    return gemini_relay.network_gateway("toolbench", NETWORK, DOCS_URL)


def _endpoint_key() -> Optional[str]:
    """The model endpoint's key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY, a
    placeholder ignored); gbench's endpoint relay adds it, the container never holds it."""
    return _base.endpoint_api_key()


def read_server_calls(path: str) -> Optional[List[Dict[str, Any]]]:
    """The image's records of its /virtual calls (SERVER_CALLS), or None without the file."""
    if not os.path.isfile(path):
        return None
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _stage(rec: Dict[str, Any]) -> str:
    stage = str(rec.get("stage") or "")
    return stage if stage in ("real_api_proxy", "simulator") else "server"


def simulator_cutoff_reason(usage: Optional[Dict[str, Any]]) -> str:
    """The leaderboard_comparable reason for simulator replies still cut at
    SIMULATOR_OUTPUT_LIMIT (simulator_usage's counts), or "" when there are none. One entry with
    no "; " inside (merge_shards states it again over the union)."""
    u = usage or {}
    cut = int(u.get("simulator_replies_cut_at_max_tokens") or 0)
    empty = int(u.get("simulator_empty_replies_at_max_tokens") or 0)
    outage = int(u.get("simulator_outages_after_max_tokens") or 0)
    if not (cut or empty or outage):
        return ""
    return (f"the cache-miss simulator's Gemini replies stopped at max_tokens "
            f"{SIMULATOR_OUTPUT_LIMIT} (server/main.py's {SIMULATOR_MAX_TOKENS} plus "
            f"{GEMINI_THINKING_ALLOWANCE} for the Gemini model's thinking) in {cut} replies used "
            f"as they arrived and {empty} with no content (the cascade asked its next model), and "
            f"{outage} simulator outage(s) followed such replies")


def judge_cutoff_reason(result: Dict[str, Any]) -> str:
    """The leaderboard_comparable reason for ToolEval judge replies still cut at
    _JUDGE_OUTPUT_LIMIT (the result's per-group counts), or "" when there are none."""
    def total(field: str) -> int:
        return sum(int(v or 0) for v in (result.get(field) or {}).values()
                   if isinstance(v, (int, float)) and not isinstance(v, bool))
    return cutoff_reason("ToolEval judge", total("judge_replies_cut_at_max_tokens_by_group"),
                         total("judge_outages_after_max_output_tokens_by_group"),
                         _JUDGE_MAX_TOKENS)


def summarize_server_calls(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The run's tool-call record: how the cached server answered (`by_type`: upstream's log
    types, `server_error` for a call that raised), the server errors by the stage they failed in,
    failed simulations, the Gemini models that simulated answers, and the same per query
    (`per_query`, keyed by (group, query id))."""
    by_type: Dict[str, int] = {}
    stages: Dict[str, int] = {}
    models: Dict[str, int] = {}
    per: Dict[Tuple[str, str], Dict[str, Any]] = {}
    failed_replies = unattributed = 0
    # simulator replies at SIMULATOR_OUTPUT_LIMIT: with content (used as they arrived), without
    # content (the cascade asked its next model), and calls no model served after such a reply
    cut_at_cap = {"cut": 0, "empty": 0, "outage": 0}
    for rec in records:
        cut_at_cap["cut"] += int(rec.get("simulator_max_tokens_cut") or 0)
        cut_at_cap["empty"] += int(rec.get("simulator_max_tokens_empty") or 0)
        cut_at_cap["outage"] += int(bool(rec.get("simulator_outage_after_max_tokens")))
        kind = str(rec.get("type") or "unknown")
        by_type[kind] = by_type.get(kind, 0) + 1
        error = bool(rec.get("server_error"))
        failed = bool(rec.get("simulator_failed_reply"))
        failed_replies += int(failed)
        if error:
            stages[_stage(rec)] = stages.get(_stage(rec), 0) + 1
        for m in rec.get("simulator_models") or []:
            models[str(m)] = models.get(str(m), 0) + 1
        group, qid = rec.get("group"), rec.get("query_id")
        if not group or qid in (None, ""):
            unattributed += int(error or failed)
            continue
        slot = per.setdefault((str(group), str(qid)), {
            "by_type": {}, "server_errors": 0, "server_error_stages": {},
            "simulator_failed_replies": 0, "simulator_models": {}})
        slot["by_type"][kind] = slot["by_type"].get(kind, 0) + 1
        if error:
            slot["server_errors"] += 1
            slot["server_error_stages"][_stage(rec)] = (
                slot["server_error_stages"].get(_stage(rec), 0) + 1)
        slot["simulator_failed_replies"] += int(failed)
        for m in rec.get("simulator_models") or []:
            slot["simulator_models"][str(m)] = slot["simulator_models"].get(str(m), 0) + 1
    usage = {"tool_calls": len(records),
             "cached": by_type.get("cached_real_response", 0),
             "real_api": by_type.get("real_response", 0),
             "simulated": by_type.get("fake_response", 0),
             "simulator_failed_replies": failed_replies,
             "simulator_replies_cut_at_max_tokens": cut_at_cap["cut"],
             "simulator_empty_replies_at_max_tokens": cut_at_cap["empty"],
             "simulator_outages_after_max_tokens": cut_at_cap["outage"],
             "server_errors": sum(stages.values()),
             "tool_input_parse_errors": by_type.get("tool_input_parse_error", 0)}
    return {"usage": usage, "by_type": dict(sorted(by_type.items())),
            "server_errors_by_stage": dict(sorted(stages.items())),
            "simulator_models_used": dict(sorted(models.items())),
            "unattributed_failures": unattributed, "per_query": per}


def read_observations(path: str) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """{group: {query id: counts}} from the image's OBSERVATIONS file, or None without it."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    groups = data.get("groups") if isinstance(data, dict) else None
    return groups if isinstance(groups, dict) else None


def read_model_failures(path: str) -> Optional[List[Dict[str, Any]]]:
    """The image's records of the DFSDT steps with a model request that failed at the endpoint
    (MODEL_FAILURES), or None without the file (the entrypoint creates it empty)."""
    return read_server_calls(path)


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def summarize_model_failures(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The steps with a model request that failed at the endpoint or was rejected for its content
    (HTTP 400/422): those whose last try failed at the endpoint (`unrecovered`: upstream passed
    the error text on as the model's reply), those whose last try was rejected (`rejected`: the
    model's outcome) and those with an endpoint failure a later try answered (`recovered`), the
    requests of each kind, per query (keyed by (group, query id)), and the ones the image could
    not tie to a query. A step with only rejections that a later try answered is counted in
    `content_rejections` alone."""
    per: Dict[Tuple[str, str], Dict[str, Any]] = {}
    totals = {"steps_unrecovered": 0, "steps_recovered": 0, "steps_rejected": 0,
              "failed_requests": 0, "content_rejections": 0,
              "unattributed_unrecovered": 0, "unattributed_recovered": 0}
    errors: Dict[str, int] = {}
    for rec in records:
        failures = _count(rec.get("endpoint_failures"))
        rejections = _count(rec.get("content_rejections"))
        kind = ("unrecovered" if rec.get("unrecovered") else "rejected" if rec.get("rejected")
                else "recovered" if failures else None)
        totals["failed_requests"] += failures
        totals["content_rejections"] += rejections
        if kind:
            totals[f"steps_{kind}"] += 1
        for e in rec.get("errors") or []:
            errors[str(e)] = errors.get(str(e), 0) + 1
        group, qid = rec.get("group"), rec.get("query_id")
        if not group or qid in (None, ""):
            if kind in ("unrecovered", "recovered"):
                totals[f"unattributed_{kind}"] += 1
            continue
        slot = per.setdefault((str(group), str(qid)), {"unrecovered_steps": 0,
                                                       "recovered_steps": 0, "rejected_steps": 0,
                                                       "content_rejections": 0, "errors": []})
        if kind:
            slot[f"{kind}_steps"] += 1
        slot["content_rejections"] += rejections
        for e in rec.get("errors") or []:
            if str(e) not in slot["errors"]:
                slot["errors"].append(str(e))
    return {**totals, "errors": dict(sorted(errors.items())), "per_query": per}


def split_content_rejections(replies: Optional[Dict[str, Any]]
                             ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """(the request log's counts without the requests the endpoint rejected for their content,
    those requests). The image's request settings count every error the model client received
    other than a context rejection left to upstream as `failed_requests`, keyed in
    `failed_by_error` by exception class and HTTP status ("BadRequestError 400"); the ones with
    an HTTP status in CONTENT_REJECTION_STATUSES are the model's outcome, not harness errors, so
    they are taken out of `failed_requests` and `failed_by_error` and returned as
    {requests, by_error}."""
    none: Dict[str, Any] = {"requests": 0, "by_error": {}}
    if not replies:
        return replies, none
    by_error = dict(replies.get("failed_by_error") or {})
    rejected = {}
    for key, n in by_error.items():
        status = str(key).rsplit(" ", 1)[-1]
        if status.isdigit() and int(status) in CONTENT_REJECTION_STATUSES:
            rejected[key] = _count(n)
    if not rejected:
        return replies, none
    n = sum(rejected.values())
    counts = dict(replies.get("counts") or {})
    counts["failed_requests"] = max(0, _count(counts.get("failed_requests")) - n)
    out = dict(replies)
    out["counts"] = counts
    out["failed_by_error"] = {k: v for k, v in by_error.items() if k not in rejected}
    return out, {"requests": n, "by_error": dict(sorted(rejected.items()))}


def harness_error_queries(test_ids: Dict[str, List[str]],
                          cand_by_group: Dict[str, Dict[str, Dict[str, Any]]],
                          per_query: Dict[Tuple[str, str], Dict[str, Any]],
                          observations: Dict[str, Dict[str, Dict[str, Any]]],
                          model_failures: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
                          ) -> Dict[str, Dict[str, List[str]]]:
    """{group: {query id: [kinds]}} for the test ids the run covers that met a harness error
    (QUERY_HARNESS_ERROR_LABELS): a tool call the cached server failed or the simulator could
    not answer (the server's record of the call, or its observation anywhere in the query's
    search tree), a tool call that timed out, no answer at all, or a DFSDT step whose model
    request failed at the endpoint on its last try (`model_failures`: the per_query of
    summarize_model_failures)."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for g, ids in test_ids.items():
        for q in ids:
            calls = per_query.get((g, str(q))) or {}
            obs = (observations.get(g) or {}).get(str(q)) or {}
            model = (model_failures or {}).get((g, str(q))) or {}
            kinds = []
            if calls.get("server_errors") or obs.get("server_error_observations"):
                kinds.append("server_error")
            if calls.get("simulator_failed_replies") or obs.get("simulator_failed_observations"):
                kinds.append("simulator_failed")
            if obs.get("timeout_observations"):
                kinds.append("tool_call_timeout")
            if str(q) not in (cand_by_group.get(g) or {}):
                kinds.append("no_answer")
            if model.get("unrecovered_steps"):
                kinds.append("model_endpoint")
            if kinds:
                out.setdefault(g, {})[str(q)] = kinds
    return out


def describe_query_harness_errors(kinds: Dict[str, int]) -> str:
    """'<n> <label>, ...' for {kind: queries}."""
    labels = dict(QUERY_HARNESS_ERROR_LABELS)
    return ", ".join(f"{kinds[k]} {labels[k]}" for k, _ in QUERY_HARNESS_ERROR_LABELS
                     if kinds.get(k))


#: Files of the workdir kept with the run's results.
ARTIFACT_FILES = (REQUEST_LOG, SERVER_CALLS, OBSERVATIONS, MODEL_FAILURES)


def keep_artifacts(workdir: str, groups: List[str]) -> Optional[str]:
    """Copy the image's records (ARTIFACT_FILES) and the converted answers (candidate/<group>.json)
    to `<GBENCH_RESULTS_DIR>/toolbench_artifacts/<fresh dir>`, every credential value of the
    environment redacted. None outside a gbench run."""
    run_dir = (os.environ.get("GBENCH_RESULTS_DIR") or "").strip()
    if not run_dir:
        return None
    base_dir = os.path.join(run_dir, "toolbench_artifacts")
    os.makedirs(base_dir, exist_ok=True)
    dest = tempfile.mkdtemp(prefix="run_", dir=base_dir)
    names = list(ARTIFACT_FILES) + [os.path.join("candidate", f"{g}.json") for g in groups]
    for name in names:
        src = os.path.join(workdir, name)
        if not os.path.isfile(src):
            continue
        target = os.path.join(dest, name)
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(src, target)
            redact_file(target)
        except OSError as e:
            logger.warning("toolbench: could not keep %s (%s)", name, e)
    return dest


def remove_workdir(workdir: str) -> List[str]:
    """Remove the run's workdir; the paths that could not be removed (logged)."""
    left: List[str] = []

    def _onerror(_func, path, _exc):
        left.append(path)
    shutil.rmtree(workdir, onerror=_onerror)
    if left:
        logger.warning("toolbench: %d path(s) of the workdir %s could not be removed (e.g. %s)",
                       len(left), workdir, left[0])
    return left


def run_toolbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical ToolBench: StableToolBench DFSDT inference + gbench Gemini-judge SoPR/SoWR."""
    ok, reason = check_toolbench_prerequisites()
    if not ok:
        raise infra_required("toolbench", reason, DOCS_URL)

    # The model-under-test's sampling (see the module note): 1.0 with --thinking, 0.0 without,
    # unless the operator sets another, the thinking switch of the run's mode, and the
    # token-limit policy.
    temperature, temperature_source = resolve_official_sampling(
        "toolbench", TOOLBENCH_SAMPLING, thinking=enable_thinking)
    plan = resolve_token_budget("toolbench", TOOLBENCH_SAMPLING, thinking=enable_thinking,
                                max_output_tokens=kwargs.get("max_output_tokens"))
    budget = plan.max_tokens if plan.policy == "reasoning_run_budget" else None
    # With the --thinking budget, each turn may take as long as gbench's own requests of that
    # budget (base.request_timeout_s); upstream's client keeps openai's 600 s otherwise.
    timeout = request_timeout_s(budget) if budget is not None else None
    expected_settings = dict(enable_thinking=bool(enable_thinking), max_tokens=budget,
                             temperature=temperature, request_timeout_s=timeout)
    limit = kwargs.get("limit")
    limit = int(limit) if limit and int(limit) > 0 else None
    shard = current_shard()

    groups = _groups()
    method = os.environ.get("GBENCH_TOOLBENCH_METHOD", _METHOD_DEFAULT)
    reference = os.environ.get("GBENCH_TOOLBENCH_REFERENCE", _REFERENCE_DEFAULT)
    raw_times = os.environ.get("GBENCH_TOOLBENCH_EVAL_TIMES", str(_EVAL_TIMES_DEFAULT))
    try:
        evaluate_times = int(raw_times)
    except ValueError:
        evaluate_times = 0
    if evaluate_times < 1:
        raise infra_required(
            "toolbench", f"GBENCH_TOOLBENCH_EVAL_TIMES={raw_times!r} is not a positive integer "
            f"(StableToolBench evaluates {_EVAL_TIMES_DEFAULT} times; unset it for that)", DOCS_URL)
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    # The container's OpenAI client sends `model` verbatim to vLLM, which serves the FULL id, not
    # gbench's stripped short name -> resolve the served id (else every call 404s -> rc=1).
    served_model = _served_model_id(endpoint, model_name)
    # The ToolEval grader and the cache-miss simulator are the judge cascade (models, rounds,
    # backoff) on the Gemini API, the one path: no knob names another model, base URL or key.
    judge_models = _base.judge_cascade()
    simulator_models = list(judge_models)
    image = _image()
    parts = urlsplit(endpoint)
    if parts.username or parts.password:
        raise infra_required(
            "toolbench", "the model endpoint URL carries a user name or password. The harness "
            "container runs the model's tool calls, so it never gets a credential: gbench's "
            "endpoint relay reaches the endpoint and adds OPENAI_API_KEY. Serve the model at a URL "
            "without user info and put its key in OPENAI_API_KEY.", DOCS_URL)

    # The queries the run covers (see the module note): each group's test ids from the image,
    # and for --shard / --eval-limit the host-side selection, which the container runs by id.
    all_test_ids = _group_test_ids(image, groups, None)
    selected_ids: Optional[Dict[str, List[str]]] = None
    n_full = sum(len(v) for v in all_test_ids.values())
    if limit or (shard and shard.partial):
        selected_ids, _label, n_full = select_query_ids(all_test_ids, limit)
        if not any(selected_ids.values()):
            reason = (f"empty shard {shard.label if shard else '1/1'}: no items on this node "
                      f"({n_full} in the full set)")
            logger.info("toolbench: %s", reason)
            return {
                "benchmark_type": "eval", "eval_name": "toolbench", "model_name": model_name,
                "thinking": bool(enable_thinking), "status": "success", "status_reason": reason,
                "total_questions": 0, "correct_answers": 0, "accuracy": None,
                "accuracy_withheld_reason": reason, "empty_shard": True,
                "full_set_items": n_full, "sample_traces": [], "selected_query_ids": {},
                **shard_identity(shard), "leaderboard_comparable": False,
                "leaderboard_comparable_reason": reason,
            }
        groups = [g for g in groups if selected_ids.get(g)]
    # One stratified draw over the flattened (group, query) list: --eval-limit N is N queries for
    # the suite, spread over the test groups (never N per group), each answered once.
    per_group = {g: len((selected_ids if selected_ids is not None else all_test_ids).get(g) or [])
                 for g in all_test_ids}
    workload = eval_limit_plan("toolbench", limit, sum(per_group.values()), per_group,
                               setting_noun="group", attempts=1, unit="answers")
    workdir = tempfile.mkdtemp(prefix="gbench_toolbench_")
    os.chmod(workdir, 0o777)   # the container writes /out as root; it hands /out back at exit
    orch_name = "gbench_toolbench_" + os.path.basename(workdir)
    sim_relay = model_relay = None
    keep_dir = None

    def _reap():
        # docker --rm only fires on a clean container EXIT; on a subprocess timeout the client is
        # killed while the container keeps running, so reap the named container in finally + on
        # timeout.
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    try:
        # The container runs the model's tool calls, so it runs on NETWORK only after the metadata
        # server was found blocked there, and holds no real key: the simulator's Gemini calls go
        # to gbench's Gemini relay (a per-run placeholder in their key's place) and the model
        # requests to gbench's endpoint relay (which adds OPENAI_API_KEY), both on the gateway.
        gateway = _bridge_gateway(image)
        sim_relay = gemini_relay.GeminiRelay(
            "toolbench", gateway, docs_url=DOCS_URL, openai_calls=(("POST", "/chat/completions"),),
            native_methods=(), name="toolbench-simulator-relay").start()
        try:
            # root_paths: the image's request policy counts each request's prompt with the
            # server's /tokenize (served at the root, outside /v1) to fit max_tokens to the room
            # the window leaves (2026-10-10 smoke test: refused, 27 of 27, before this).
            model_relay = EndpointRelay(endpoint, gateway, key=_endpoint_key(),
                                        name="toolbench-model-relay",
                                        root_paths=TOKENIZE_ROOT_PATHS)
        except ValueError as e:
            raise infra_required("toolbench", f"the model endpoint is not an http(s) URL with a "
                                 f"host ({e}).", DOCS_URL) from None
        model_relay.TIMEOUT_S = max(EndpointRelay.TIMEOUT_S, int(timeout or 600) + 120)
        try:
            model_relay.start()
        except OSError as e:
            raise infra_required("toolbench", f"gbench's relay to the model endpoint could not "
                                 f"listen on {gateway} ({type(e).__name__}: {e}).",
                                 DOCS_URL) from None
        # Credentials go by name only (`-e NAME`, the value taken from `env`): the only one is the
        # Gemini relay's placeholder. The cascade goes in as GBENCH_JUDGE_MODELS.
        env = {k: v for k, v in os.environ.items() if k not in _CREDENTIAL_VARS}
        env.update(sim_relay.client_env())
        cascade_env = _base.gemini_cascade_env()
        cmd = ["docker", "run", "--rm", "--pull", "never", "--name", orch_name,
               "--network", NETWORK,
               "-v", f"{workdir}:/out:rw",
               "-e", f"TB_MODEL={served_model}", "-e", f"TB_ENDPOINT={model_relay.url}",
               "-e", "GEMINI_API_KEY", "-e", "GEMINI_API_KEYS", "-e", "GEMINI_OPENAI_BASE_URL",
               "-e", gemini_relay.RELAY_URL_ENV,
               "-e", f"GBENCH_JUDGE_MODELS={cascade_env['GBENCH_JUDGE_MODELS']}",
               "-e", f"TB_GROUPS={' '.join(groups)}", "-e", f"TB_METHOD={method}",
               "-e", f"TB_NUM_THREAD={max(1, concurrency)}",
               "-e", f"TB_REAL_API_TIMEOUT_S={REAL_API_TIMEOUT_S}",
               "-e", f"TB_HOST_UID={os.getuid()}", "-e", f"TB_HOST_GID={os.getgid()}",
               # the model request settings the image applies (toolbench_request_policy.py)
               "-e", f"TB_ENABLE_THINKING={'true' if enable_thinking else 'false'}",
               "-e", f"TB_REQUEST_LOG=/out/{REQUEST_LOG}"]
        if budget is not None:
            cmd += ["-e", f"TB_MAX_TOKENS={budget}"]
        if timeout is not None:
            cmd += ["-e", f"TB_REQUEST_TIMEOUT_S={timeout}"]
        cmd += ["-e", f"TB_TEMPERATURE={temperature}"]
        if os.environ.get("GBENCH_TOOLBENCH_MAX_QUERY_COUNT"):
            env["TB_MAX_QUERY_COUNT"] = os.environ["GBENCH_TOOLBENCH_MAX_QUERY_COUNT"]
            cmd += ["-e", "TB_MAX_QUERY_COUNT"]
        # The cache-miss simulator: the Gemini cascade (GBENCH_JUDGE_MODELS above) with the judge's
        # rounds and backoff.
        cmd += ["-e", f"TB_SIMULATOR_ROUNDS={_base._judge_cascade_rounds()}",
                "-e", f"TB_SIMULATOR_BACKOFF={_base._JUDGE_BACKOFF}"]
        # --shard / --eval-limit: the selected query ids per group (select_query_ids), which the
        # entrypoint runs by id (TB_QUERY_IDS_FILE) - NOT the per-task DFS search budget (that is
        # --max_query_count / TB_MAX_QUERY_COUNT, canonical 200).
        if selected_ids is not None:
            with open(os.path.join(workdir, QUERY_IDS_FILE), "w", encoding="utf-8") as fh:
                json.dump({g: selected_ids[g] for g in groups}, fh)
            os.chmod(os.path.join(workdir, QUERY_IDS_FILE), 0o644)
            cmd += ["-e", f"TB_QUERY_IDS_FILE=/out/{QUERY_IDS_FILE}"]
            logger.info("toolbench: %d of %d queries selected (%s) across %d group(s)",
                        sum(len(v) for v in selected_ids.values()), n_full,
                        ", ".join(p for p in ((f"shard {shard.label}" if shard and shard.partial
                                               else ""),
                                              (f"--eval-limit {limit}" if limit else "")) if p),
                        len(groups))
        cmd.append(image)
        # Before the (long) run: the reference converted answers for SoWR, in upstream's
        # model_predictions_converted layout (<root>/<reference model>/<group>.json) - a host dir
        # override, else the published baseline answers of the reference model (ChatGPT-CoT by
        # default), converted and cached - and the test ids the run covers (the selection under
        # --shard / --eval-limit), which SoWR votes on as eval_preference.py does over --test_ids.
        ref_root = (os.environ.get("GBENCH_TOOLBENCH_REFERENCE_DIR")
                    or _baseline_reference_root(reference, groups, image))
        test_ids = ({g: list(selected_ids[g]) for g in groups} if selected_ids is not None
                    else {g: list(all_test_ids.get(g) or []) for g in groups})
        # SoWR is a leaderboard column: every test id the run covers needs a reference answer.
        ref_by_group = _load_groups(os.path.join(ref_root, reference), groups)
        gaps = _reference_gaps(ref_by_group, test_ids)
        if gaps:
            raise infra_required(
                "toolbench", f"the SoWR reference answers ({reference}) under "
                f"{os.path.join(ref_root, reference)} do not cover the run: " + "; ".join(gaps)
                + ". SoWR is a StableToolBench leaderboard column, so the run is not scored "
                "without it: point GBENCH_TOOLBENCH_REFERENCE_DIR at converted answers covering "
                f"every test id (<dir>/{reference}/<group>.json), or unset it to use the "
                "published baselines", DOCS_URL)
        logger.info("toolbench: docker run %s (groups=%s, network=%s)", image, groups, NETWORK)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=16 * 60 * 60,
                                  env=env)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "toolbench", "the toolbench container exceeded its 16h timeout and was killed.",
                DOCS_URL) from exc
        keep_dir = keep_artifacts(workdir, groups)
        if proc.returncode != 0:
            raise infra_required(
                "toolbench",
                f"the toolbench container failed (rc={proc.returncode}). "
                f"tail: {_tail(proc.stderr or proc.stdout, 600)}", DOCS_URL)

        # The entrypoint converts each group to /out/candidate/<group>.json. A group with no
        # parseable answers is a harness failure; scoring the rest would be a partial run.
        # The settings the image applied are checked before any judge call is spent.
        replies = check_request_log("toolbench",
                                    read_model_request_log(os.path.join(workdir, REQUEST_LOG)),
                                    expected_settings, enable_thinking, DOCS_URL)
        harness_replies, rejections = split_content_rejections(replies)
        all_failed = every_model_request_failed(replies)
        if all_failed:
            rejected_text = (f"; {rejections['requests']} of them rejected for their content ("
                             + ", ".join(f"{k} x{v}" for k, v in rejections["by_error"].items())
                             + "), although no request carried text the model had generated"
                             if rejections["requests"] else "")
            raise infra_required(
                "toolbench", f"none of the run's {all_failed} model-under-test request(s) got a "
                "reply (each ended in an error of the served endpoint or of the connection, or "
                f"was rejected{rejected_text}), so every query's outcome is a harness error and "
                "no score is published. Check that the model endpoint is reachable and accepts "
                "the harness's requests.", DOCS_URL)
        server_calls = read_server_calls(os.path.join(workdir, SERVER_CALLS))
        observations = read_observations(os.path.join(workdir, OBSERVATIONS))
        model_failure_records = read_model_failures(os.path.join(workdir, MODEL_FAILURES))
        if server_calls is None or observations is None or model_failure_records is None:
            missing = [n for n, v in ((SERVER_CALLS, server_calls), (OBSERVATIONS, observations),
                                      (MODEL_FAILURES, model_failure_records)) if v is None]
            raise infra_required(
                "toolbench", f"the container wrote no {' / '.join(missing)}, so the tool calls "
                "the cached server failed or the simulator could not answer, or the model "
                "requests that failed at the endpoint, cannot be told apart from the model's "
                "outcome and no score is published. Rebuild the image from "
                "docker/toolbench.Dockerfile.", DOCS_URL)
        model_failures = summarize_model_failures(model_failure_records)
        if model_failures["unattributed_unrecovered"]:
            raise infra_required(
                "toolbench", f"{model_failures['unattributed_unrecovered']} DFSDT step(s) ended on "
                "a model request that failed at the endpoint, and the image could not tie them to "
                "a query, so the queries whose search tree got the error text as the model's "
                "reply cannot be left out of SoPR and SoWR; no score is published rather than one "
                "that scores harness failures as the model's. Check the model endpoint and "
                "rebuild the image from docker/toolbench.Dockerfile.", DOCS_URL)
        cand_by_group = _load_groups(os.path.join(workdir, "candidate"), groups)
        empty = [g for g, answers in cand_by_group.items() if not answers]
        if empty:
            raise infra_required(
                "toolbench", f"the container produced no parseable converted answers for "
                f"group(s) {empty} under /out/candidate (a harness failure, not a 0%)", DOCS_URL)
        calls = summarize_server_calls(server_calls)
        harness_queries = harness_error_queries(test_ids, cand_by_group, calls["per_query"],
                                                observations, model_failures["per_query"])
        excluded = {g: set(qs) for g, qs in harness_queries.items()}
        scores = asyncio.run(_score_groups(cand_by_group, ref_by_group, evaluate_times,
                                           concurrency, test_ids=test_ids, excluded=excluded))
    finally:
        _reap()
        for relay in (sim_relay, model_relay):
            if relay is not None:
                relay.stop()
        remove_workdir(workdir)

    n_harness_queries = sum(len(v) for v in harness_queries.values())
    query_kinds: Dict[str, int] = {}
    for qs in harness_queries.values():
        for kinds in qs.values():
            for k in kinds:
                query_kinds[k] = query_kinds.get(k, 0) + 1
    kinds_text = describe_query_harness_errors(query_kinds)
    # A group without a SoPR or SoWR verdict leaves the headline (the mean over the groups)
    # without a value: the score is withheld (null + the reason, status failed), never
    # published over part of the groups, and the run's diagnostics are kept.
    withheld = withheld_reason(scores, harness_queries, test_ids, kinds_text)

    def _pct(v: Optional[float]) -> Optional[float]:
        return round(v * 100.0, 2) if v is not None else None

    by_group = scores["by_group"]
    outages = sum(int(r.get("judge_outages") or 0) for r in by_group.values())
    invalid = sum(int(r.get("judge_invalid_replies") or 0) for r in by_group.values())
    lost = outages + invalid
    no_vote = {g: list(r.get("sowr_queries_without_vote") or []) for g, r in by_group.items()}
    n_no_vote = sum(len(v) for v in no_vote.values())
    judge_used: Dict[str, int] = {}
    for r in by_group.values():
        for m, n in (r.get("judge_models_used") or {}).items():
            judge_used[m] = judge_used.get(m, 0) + n
    answered = dict(judge_used)
    for m, n in calls["simulator_models_used"].items():
        answered[m] = answered.get(m, 0) + n
    usage = calls["usage"]
    counts = (replies or {}).get("counts") or {}
    official = TOOLBENCH_SAMPLING.record(temperature, temperature_source)
    official.update(plan.record())
    knob = kwargs.get("max_output_tokens")
    if budget is None:
        tokens_note = ((f"--max-output-tokens {knob} not applied: " if knob else "")
                       + f"upstream's max_tokens={UPSTREAM_MAX_TOKENS} is sent exactly on every "
                       "DFSDT turn (chatgpt_function_model.py chat_completion_request)")
    else:
        source = f"--max-output-tokens {knob}" if knob else f"the thinking floor {budget}"
        tokens_note = (f"{REASONING_CAP_NOTE} (upstream sends max_tokens={UPSTREAM_MAX_TOKENS}); "
                       f"every DFSDT turn was sent this run's budget, {source}, cut to the room "
                       "the context window leaves after the prompt, with a request timeout sized "
                       f"to it ({timeout} s to read the reply)")
    sampling_notes = []
    if official.get("temperature_matches_official") is False:
        sampling_notes.append(f"temperature {temperature} set by "
                              f"{temperature_source.split(':', 1)[-1]} (upstream sends none)")
    if replies is None:
        sampling_notes.append("the image wrote no model request log, so replies cut at max_tokens "
                              "are not counted")
    elif budget is not None and counts.get("cut_at_max_tokens"):
        sampling_notes.append(
            f"{counts['cut_at_max_tokens']} model repl(ies) reached this run's max_tokens "
            f"({counts.get('cut_in_reasoning', 0)} inside the reasoning channel with no answer): "
            f"upstream's {UPSTREAM_MAX_TOKENS} was defined for non-reasoning generation and is "
            "not applied with --thinking; raise --max-output-tokens")
    harness_errors, failure_notes = model_request_failures(harness_replies, timeout,
                                                           "StableToolBench")
    sampling_notes += failure_notes
    # Requests the endpoint rejected for their content: the model's outcome (counted, named).
    if rejections["requests"]:
        sampling_notes.append(
            f"{rejections['requests']} model request(s) were rejected by the endpoint for their "
            "content ("
            + ", ".join(f"{k} x{v}" for k, v in rejections["by_error"].items())
            + "): the model's outcome, not harness errors (the request carried the model's own "
            "conversation; upstream passes the error text on)"
            + (f"; {model_failures['steps_rejected']} DFSDT step(s) ended on such a rejection, "
               "their queries scored" if model_failures["steps_rejected"] else ""))
    # Tool calls the cached server failed or the simulator could not answer, and queries with no
    # answer: harness errors, their queries left out of SoPR and SoWR.
    if n_harness_queries:
        harness_errors.append({
            "error": "queries that met a harness error, left out of SoPR and SoWR",
            "queries": n_harness_queries, "by_kind": dict(sorted(query_kinds.items())),
            "by_group": {g: len(v) for g, v in sorted(harness_queries.items())},
            "server_errors_by_stage": calls["server_errors_by_stage"],
            "simulator_failed_replies": usage["simulator_failed_replies"]})
        sampling_notes.append(f"{n_harness_queries} quer(ies) met a harness error and are left "
                              f"out of SoPR and SoWR ({kinds_text})")
    if model_failures["steps_recovered"]:
        sampling_notes.append(
            f"{model_failures['steps_recovered']} DFSDT step(s) got a reply on a later try after a "
            "model request failed at the endpoint (upstream's retries left the search unchanged, "
            "so those queries are scored)")
    if calls["unattributed_failures"]:
        harness_errors.append({"error": "failed tool calls the server could not tie to a query "
                                        "(their queries are found by the answer trees' "
                                        "observations)",
                               "calls": calls["unattributed_failures"]})
        sampling_notes.append(f"{calls['unattributed_failures']} failed tool call(s) carried no "
                              "query id (the answer trees' observations name their queries)")
    if lost:
        harness_errors.append({"error": "ToolEval calls without a usable verdict (a judge outage "
                                        "or no valid reply after 3 tries)",
                               "judge_outages": outages, "judge_invalid_replies": invalid})
    judge_cut = cutoff_reason(
        "ToolEval judge",
        sum(int(r.get("judge_replies_cut_at_max_tokens") or 0) for r in by_group.values()),
        sum(int(r.get("judge_outages_after_max_output_tokens") or 0) for r in by_group.values()),
        _JUDGE_MAX_TOKENS)
    if judge_cut:
        sampling_notes.append(judge_cut)
    sim_cut = simulator_cutoff_reason(usage)
    if sim_cut:
        sampling_notes.append(sim_cut)
    if n_no_vote:
        sampling_notes.append(f"{n_no_vote} quer(ies) had no preference verdict in any pass and "
                              "are left out of the SoWR denominator")
    # The image's per-step records of endpoint failures against the request log's count of them:
    # a failed request the patched client did not record would leave its query scored.
    inconsistent = failure_records_inconsistency(harness_replies, model_failures)
    if inconsistent:
        harness_errors.append({"error": "the request log counts more endpoint failures than the "
                                        "per-step failure records hold (model_failure_records_"
                                        "inconsistent)", **inconsistent})
        sampling_notes.append(
            f"model_failure_records_inconsistent: the request log counts "
            f"{inconsistent['request_log_failed_requests']} failed model request(s) and the "
            f"per-step records {inconsistent['step_records_failed_requests']}, so some queries "
            "whose search tree got an endpoint error may be scored")
    if usage["tool_calls"]:
        sampling_notes.append(
            f"of {usage['tool_calls']} tool call(s), {usage['cached']} were answered from the "
            f"cache, {usage['real_api']} by the real-API proxy and {usage['simulated']} "
            f"({round(100.0 * usage['simulated'] / usage['tool_calls'], 1)}%) by the Gemini "
            "simulator")

    # One record per test id the run covers: its harness errors, its tool calls, and for a
    # scored query every pass's and comparison's verdict with the ToolEval calls behind it.
    sample_traces: List[Dict[str, Any]] = []
    for g in groups:
        records = by_group.get(g, {}).get("records") or {}
        for q in test_ids.get(g) or []:
            q = str(q)
            rec = dict(records.get(q) or {"query_id": q})
            rec["group"] = g
            rec["harness_errors"] = (harness_queries.get(g) or {}).get(q, [])
            call_rec = calls["per_query"].get((g, q)) or {}
            rec["tool_calls"] = call_rec.get("by_type") or {}
            rec["simulator_models"] = call_rec.get("simulator_models") or {}
            rec["server_error_stages"] = call_rec.get("server_error_stages") or {}
            rec["tool_observations"] = (observations.get(g) or {}).get(q) or {}
            rec["model_request_failures"] = model_failures["per_query"].get((g, q)) or {}
            sample_traces.append(rec)

    sopr = _pct(scores["sopr"])
    sowr = _pct(scores.get("sowr"))
    subset = sorted(set(groups)) != sorted(_GROUPS)
    n_covered = sum(len(v) for v in test_ids.values())
    solved_eq = sum(float(r.get("solved_equivalent") or 0.0) for r in by_group.values())
    has_errors = bool(harness_errors)
    if withheld:
        sopr = sowr = None
    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "toolbench",
        "model_name": model_name,
        # Harness errors (a model request the endpoint failed, a tool call the server failed, a
        # failed simulation, a query without an answer, a ToolEval call without a verdict) are
        # never the model's outcome.
        "status": ("failed" if withheld else
                   "completed_with_errors" if has_errors else "success"),
        "accuracy": sopr,                 # headline = unweighted mean of the per-group SoPRs (%)
        "headline_metrics": {"SoPR": sopr, "SoWR": sowr},
        "sopr": sopr,
        "sowr": sowr,
        "sopr_by_group": {g: _pct(r["sopr"]) for g, r in by_group.items()},
        # Spread of the per-pass SoPR within each group (the README's "+-" / eval_pass_rate's Std).
        "sopr_std_by_group": {g: _pct(r.get("sopr_std")) for g, r in by_group.items()},
        "sowr_by_group": {g: _pct(r["sowr"]) for g, r in by_group.items()},
        "queries_by_group": {g: r["n_candidate"] for g, r in by_group.items()},
        # Counts are queries: the test ids the run covers, those scored (no harness error), and
        # the Solved-equivalent over them (Solved 1, Unsure 0.5, averaged over the passes).
        "count_unit": "queries",
        "total_questions": n_covered,
        "effective_n": scores.get("n_candidate"),
        "correct_answers": round(solved_eq, 2),
        "correct_answers_note": ("Solved-equivalent: per group SoPR x scored queries, summed "
                                 "(Solved 1, Unsure 0.5, averaged over the evaluation passes); "
                                 "the headline averages the groups with equal weight"),
        "aggregation": ("per-group SoPR/SoWR, headline = unweighted mean over the evaluated "
                        "groups (the StableToolBench README 'Average' column)"),
        "groups_evaluated": groups,
        "method": method,
        "reference_model": reference,
        "reference_source": (
            f"GBENCH_TOOLBENCH_REFERENCE_DIR ({os.environ['GBENCH_TOOLBENCH_REFERENCE_DIR']})"
            if os.environ.get("GBENCH_TOOLBENCH_REFERENCE_DIR") else
            f"{BASELINES_REPO}@{BASELINES_REVISION} {BASELINES_FILE} ({reference}, converted with "
            "StableToolBench's convert_to_answer_format.py)"),
        "sowr_queries_by_group": {g: len(v) for g, v in test_ids.items()},
        "sowr_votes_missing_by_group": {g: r.get("sowr_votes_missing") for g, r in by_group.items()},
        "sowr_queries_without_vote_by_group": no_vote,
        "judge": "gbench-gemini-cascade",
        "judge_models": judge_models,
        # The Gemini models that actually answered: the ToolEval calls (by group) and the
        # simulator's answers, and both together.
        "judge_models_used": dict(sorted(judge_used.items())),
        "judge_models_used_by_group": {g: r.get("judge_models_used") for g, r in by_group.items()},
        "simulator_models_used": calls["simulator_models_used"],
        "gemini_models_answered": dict(sorted(answered.items())),
        "judge_temperature": _JUDGE_TEMPERATURE,
        "judge_max_output_tokens": _JUDGE_OUTPUT_LIMIT,
        "judge_output_limit": output_limit_record(_JUDGE_MAX_TOKENS,
                                                  "config.yaml completions_kwargs max_tokens",
                                                  UPSTREAM_JUDGE_MODEL),
        # The cached server's cache-miss path: upstream's, with the simulator model replaced.
        "simulator": {
            "models": simulator_models,
            "source": "gbench judge cascade",
            "provider": "Gemini API",
            "base_url": SIMULATOR_BASE,
            "reached_through": "gbench's Gemini relay (gemini_relay.GeminiRelay)",
            "upstream_model": UPSTREAM_SIMULATOR_MODEL,
            "matches_upstream_model": False,
            "request": (f"upstream's: temperature 0, response_format json_object, 3 tries for "
                        f"valid JSON; max_tokens {SIMULATOR_OUTPUT_LIMIT} (upstream's "
                        f"{SIMULATOR_MAX_TOKENS} plus {GEMINI_THINKING_ALLOWANCE} for the Gemini "
                        "model's thinking)"),
            "output_limit": output_limit_record(SIMULATOR_MAX_TOKENS,
                                                "server/main.py fake_response_function_chat "
                                                "max_tokens", UPSTREAM_SIMULATOR_MODEL),
            "real_api_proxy": UPSTREAM_REAL_API_PROXY, "toolbench_key": "", "is_save": True,
            "real_api_timeout_s": REAL_API_TIMEOUT_S,
        },
        # How the cached server answered the run's tool calls (its record of every /virtual
        # call): from the cache, the real-API proxy or the simulator, the failures by stage.
        "simulator_usage": usage,
        "tool_calls_by_type": calls["by_type"],
        "server_errors_by_stage": calls["server_errors_by_stage"],
        # Queries left out of SoPR and SoWR as harness errors, by kind and group, and each one.
        "n_harness_error_queries": n_harness_queries,
        "harness_error_query_kinds": dict(sorted(query_kinds.items())),
        "harness_error_queries_by_group": {g: sorted(v) for g, v in sorted(harness_queries.items())},
        # Every external model of the harness, upstream's and the Gemini one that replaces it.
        "gemini_substitutions": [
            {"role": "ToolEval grader (SoPR/SoWR judge)", "upstream": UPSTREAM_JUDGE_MODEL,
             "gemini_models": judge_models,
             "api": "Gemini API (google-genai function calling, base.judge_generate_cascade's "
                    "models, key rotation, rounds and backoff)"},
            {"role": "cache-miss API simulator", "upstream": UPSTREAM_SIMULATOR_MODEL,
             "gemini_models": simulator_models,
             "api": f"Gemini API OpenAI-compatible endpoint {SIMULATOR_BASE}, through gbench's "
                    "Gemini relay"},
        ],
        "judge_outages_by_group": {g: r.get("judge_outages") for g, r in by_group.items()},
        "judge_invalid_replies_by_group": {g: r.get("judge_invalid_replies")
                                           for g, r in by_group.items()},
        # Judge replies that stopped at _JUDGE_OUTPUT_LIMIT (ToolEval's 1000 plus the thinking
        # allowance) without the forced call; the cascade asked its next model after each. Judge
        # outages whose cascade saw such a reply are counted apart.
        "judge_replies_cut_at_max_tokens_by_group": {
            g: r.get("judge_replies_cut_at_max_tokens") for g, r in by_group.items()},
        "judge_outages_after_max_output_tokens_by_group": {
            g: r.get("judge_outages_after_max_output_tokens") for g, r in by_group.items()},
        "evaluate_times": evaluate_times,
        "sopr_passes_by_group": {g: r.get("sopr_passes") for g, r in by_group.items()},
        "sample_traces": sample_traces,
        "thinking": bool(enable_thinking),
        "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": official,
        "max_output_tokens_note": tokens_note,
        # The image's record of the settings it applied and of the model-under-test's replies:
        # replies, cut_at_max_tokens (finish_reason length), cut_in_reasoning (no answer text, no
        # tool call), fitted_to_context / refitted_after_rejection (--thinking budget cut to the
        # window), failed_requests / timed_out (every error reply the model client received,
        # content rejections included; model_content_rejections splits those out and
        # harness_errors carries the endpoint failures), rejected_over_context (left to
        # upstream's handling), cancelled (abandoned in flight).
        "model_request_settings": (replies or {}).get("settings"),
        "model_replies": dict(counts) if replies is not None else None,
        "model_failed_by_error": (replies or {}).get("failed_by_error"),
        # Of model_replies.failed_requests, the requests the endpoint rejected for their content
        # (HTTP 400/422): the model's outcome, not harness errors ({requests, by_error}).
        "model_content_rejections": rejections,
        "model_max_tokens_sent": (replies or {}).get("max_tokens_sent"),
        "model_request_timeout_s": timeout,
        # The DFSDT steps with a model request that failed at the endpoint or was rejected for its
        # content, per the image's record (model_request_failures.jsonl): ended on an endpoint
        # failure (their queries are harness errors of kind model_endpoint), answered on a later
        # try, or ended on a content rejection (the model's outcome); those queries are scored.
        "model_request_failures": {k: v for k, v in model_failures.items() if k != "per_query"},
        "harness_errors": harness_errors,
        "docker_network": NETWORK,
        "gemini_relay": sim_relay.record() if sim_relay is not None else None,
        "model_endpoint_relay": model_relay.record() if model_relay is not None else None,
        "artifacts_dir": keep_dir,
        "sampling": ("StableToolBench's DFSDT loop builds the model-under-test's requests; gbench's "
                     "in-image request settings set the chat template's thinking switch to the "
                     "run's mode, send temperature 1.0 with --thinking and 0.0 without unless "
                     "the operator sets another, and follow the token-limit policy (upstream's max_tokens 1024 without --thinking, the run's "
                     "budget with it). The Gemini judge cascade samples at ToolEval's temperature "
                     "0.2 with its max_tokens 1000."),
        "metric": ("StableToolBench SoPR (Solvable Pass Rate) headline + SoWR (win rate vs "
                   f"{reference}), each computed per test group over {evaluate_times} evaluation "
                   "pass(es) (SoPR averaged over the passes, SoWR by each query's most frequent "
                   "verdict) and averaged over the groups with equal weight, scored by gbench's "
                   "standard Gemini cascade judge (ToolEval prompts verbatim, the config.yaml "
                   "function forced through function calling, temperature 0.2, max_tokens 1000; "
                   "upstream's ToolEval grader is gpt-4-turbo). Queries that met a harness error "
                   "(a tool call the cached server failed, a failed simulation, a tool-call "
                   "timeout, a DFSDT step whose last model request try failed at the endpoint, "
                   "no answer) are left out of both."),
        # A run here is a gbench-internal number, not a like-for-like StableToolBench leaderboard
        # entry: gbench grades with its standard Gemini cascade (a gbench convention) where the
        # published leaderboard's ToolEval judge is gpt-4-turbo.
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": (
            "graded by gbench's standard Gemini cascade on the Gemini API ("
            + ", ".join(judge_models) + "; a gbench convention); the published "
            "StableToolBench leaderboard's ToolEval judge is gpt-4-turbo, so this is a "
            "gbench-internal number (aggregation matches upstream: per-group SoPR, headline = "
            "their unweighted Average); calls the cache and the real-API proxy do not answer "
            f"are simulated by the gbench Gemini cascade instead of {UPSTREAM_SIMULATOR_MODEL} "
            f"(Gemini API OpenAI-compatible endpoint {SIMULATOR_BASE}: "
            + ", ".join(simulator_models) + ")"
            + "".join(f"; {n}" for n in sampling_notes)
            + (f"; only groups {groups} were run, so the Average is over a subset"
               if subset and not (shard and shard.partial) else "")
            + (f"; {evaluate_times} evaluation pass(es) instead of StableToolBench's "
               f"{_EVAL_TIMES_DEFAULT}" if evaluate_times != _EVAL_TIMES_DEFAULT else "")
            + (f"; --eval-limit {limit}: {n_covered} of the {n_full} queries, drawn stratified "
               "by group with a fixed seed" if limit else "")
            + (f"; {lost} ToolEval call(s) had no usable verdict (judge outage or no valid "
               "reply after 3 tries) and were left out of their pass or comparison"
               if lost else "")),
        # the queries gbench selected for --shard / --eval-limit ({group: ids}; None: every test
        # id of each group ran)
        "selected_query_ids": selected_ids,
        "eval_limit_plan": workload,
        **shard_identity(shard),
        **_attempts_record(kwargs),
    }
    if shard is not None and shard.partial:
        add_comparability_reason(result, SHARD_REASON.format(
            label=shard.label, index=shard.index, count=shard.count))
    if withheld:
        result["accuracy_withheld_reason"] = withheld
        result["status_reason"] = "nothing published: " + withheld
        add_comparability_reason(result, withheld)
    return result


def withheld_reason(scores: Dict[str, Any], harness_queries: Dict[str, Any],
                    test_ids: Dict[str, List[str]], kinds_text: str) -> Optional[str]:
    """Why the headline has no value (None when every group has its SoPR and SoWR): the groups
    without a SoPR verdict, those whose every query is a harness error named apart from those
    whose ToolEval calls gave no usable verdict, and the groups without a SoWR."""
    by_group = scores.get("by_group") or {}
    no_verdict = [g for g, r in by_group.items() if r.get("sopr") is None]
    no_sowr = [g for g, r in by_group.items() if r.get("sowr") is None]
    parts = []
    if no_verdict or (scores.get("sopr") is None and by_group):
        all_harness = [g for g in no_verdict
                       if len(harness_queries.get(g) or {}) >= len(test_ids.get(g) or [])]
        others = [g for g in no_verdict if g not in all_harness]
        text = f"no SoPR verdict for group(s) {no_verdict}"
        details = []
        if all_harness:
            details.append(f"every query of {all_harness} is a harness error ({kinds_text})")
        if others:
            details.append(f"{others} had no usable ToolEval verdict (judge outages or invalid "
                           "replies)")
        parts.append(text + (": " + "; ".join(details) if details else ""))
    if no_sowr or (scores.get("sowr") is None and by_group):
        parts.append(f"no SoWR for group(s) {no_sowr} (no test id had a preference verdict: every "
                     "comparison a judge outage or invalid reply, or every query a harness error)")
    if not parts and not by_group:
        parts.append("no group was scored")
    if not parts:
        return None
    return ("; ".join(parts) + ". The headline averages every group (SoPR and SoWR are "
            "StableToolBench leaderboard columns), so no score is published over part of them; "
            "re-run to replace the harness errors")


def failure_records_inconsistency(harness_replies: Optional[Dict[str, Any]],
                                  model_failures: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """{request_log_failed_requests, step_records_failed_requests} when the request log (content
    rejections left out) counts more endpoint failures than the image's per-step records hold:
    a failed request the patched client did not record leaves its query scored. None when they
    agree (or there is no log)."""
    counts = (harness_replies or {}).get("counts") or {}
    logged = _count(counts.get("failed_requests"))
    recorded = _count(model_failures.get("failed_requests"))
    if logged > recorded:
        return {"request_log_failed_requests": logged, "step_records_failed_requests": recorded}
    return None


def _attempts_record(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """{"attempts": {...not_applied}} when --attempt-count asked for more than 1: StableToolBench
    runs every model once (and evaluates it GBENCH_TOOLBENCH_EVAL_TIMES times)."""
    k = int(kwargs.get("attempt_count") or get_run_knob("attempt_count") or 1)
    if k <= 1:
        return {}
    return {"attempts": {"attempts_requested": k, "attempts_per_sample": 1,
                         "not_applied": ("toolbench: StableToolBench runs every model once and "
                                         "evaluates the answers three times; --attempt-count "
                                         f"{k} is not applied")}}


def _sum_counts(dicts: List[Any]) -> Dict[str, Any]:
    """The key-wise sum of flat count dicts (None entries skipped), in key order."""
    out: Dict[str, Any] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                out.setdefault(k, v)
            else:
                out[k] = out.get(k, 0) + v
    return dict(sorted(out.items()))


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run's results (gbench --merge), recomputed from the union of
    the shards' per-query records (sample_traces, one per group and query id): each group's SoPR as
    eval_pass_rate.py scores it (per evaluation pass the mean of Solved 1 / Unsure 0.5 / Unsolved 0
    over the queries with a verdict, then the mean over the passes) and SoWR as eval_preference.py
    does (a query is a win when the candidate takes more of its votes than the reference; a query
    with no vote is left out), over the queries without a harness error; the headline is the
    unweighted mean of the groups' SoPRs. The Gemini models that answered (judge_models_used,
    simulator_models_used, gemini_models_answered), the harness errors and every count are those
    of all the shards together."""
    from ...merge import MergeError
    union: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        traces = r.get("sample_traces")
        if not isinstance(traces, list):
            raise MergeError("toolbench: a shard carries no per-query records (sample_traces)")
        for t in traces:
            union.setdefault((str(t.get("group")), str(t.get("query_id"))), t)
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for (g, _q), t in sorted(union.items(), key=lambda kv: (kv[0][0], _qid_order(kv[0][1]))):
        by_group.setdefault(g, []).append(t)
    sopr: Dict[str, Optional[float]] = {}
    sowr: Dict[str, Optional[float]] = {}
    scored_n: Dict[str, int] = {}
    for g, recs in by_group.items():
        scored = [t for t in recs if not t.get("harness_errors")]
        scored_n[g] = len(scored)
        passes: Dict[int, List[float]] = {}
        for t in scored:
            for p in t.get("sopr_passes") or []:
                if p.get("status") is not None:
                    passes.setdefault(int(p.get("pass", 0)), []).append(
                        _STATUS_SCORE.get(p["status"], 0.0))
        runs = [sum(v) / len(v) for _k, v in sorted(passes.items()) if v]
        sopr[g] = sum(runs) / len(runs) if runs else None
        wins = voted = 0
        for t in scored:
            votes = [rd.get("vote") for rd in t.get("sowr_rounds") or [] if rd.get("vote")]
            if not votes:
                continue
            voted += 1
            wins += 1 if votes.count("cand") > votes.count("ref") else 0
        sowr[g] = wins / voted if voted else None

    def _pct(v: Optional[float]) -> Optional[float]:
        return round(v * 100.0, 2) if v is not None else None

    merged = dict(results[0])
    merged["sample_traces"] = [t for g in by_group for t in by_group[g]]
    merged["groups_evaluated"] = sorted(by_group)
    merged["sopr_by_group"] = {g: _pct(v) for g, v in sopr.items()}
    merged["sowr_by_group"] = {g: _pct(v) for g, v in sowr.items()}
    merged["queries_by_group"] = scored_n
    merged["total_questions"] = len(union)
    merged["effective_n"] = sum(scored_n.values())
    merged["selected_query_ids"] = {g: [str(t.get("query_id")) for t in recs]
                                    for g, recs in by_group.items()}
    complete = all(v is not None for v in sopr.values()) and sopr
    merged["sopr"] = merged["accuracy"] = (_pct(sum(sopr.values()) / len(sopr)) if complete
                                           else None)
    merged["sowr"] = (_pct(sum(sowr.values()) / len(sowr))
                      if sowr and all(v is not None for v in sowr.values()) else None)
    merged["headline_metrics"] = {"SoPR": merged["sopr"], "SoWR": merged["sowr"]}
    merged["n_harness_error_queries"] = sum(1 for t in union.values() if t.get("harness_errors"))
    # Every count, Gemini model record and harness error over all the shards, not shard 1's.
    for field in ("judge_models_used", "simulator_models_used", "gemini_models_answered",
                  "harness_error_query_kinds", "simulator_usage", "tool_calls_by_type",
                  "server_errors_by_stage", "judge_replies_cut_at_max_tokens_by_group",
                  "judge_outages_after_max_output_tokens_by_group"):
        merged[field] = _sum_counts([r.get(field) for r in results])
    replies = [r.get("model_replies") for r in results]
    merged["model_replies"] = (None if any(x is None for x in replies)
                               else _sum_counts(replies))
    by_group_models: Dict[str, List[Any]] = {}
    for r in results:
        for g, models in (r.get("judge_models_used_by_group") or {}).items():
            by_group_models.setdefault(g, []).append(models)
    merged["judge_models_used_by_group"] = {g: _sum_counts(v)
                                            for g, v in sorted(by_group_models.items())}
    error_queries: Dict[str, set] = {}
    for r in results:
        for g, ids in (r.get("harness_error_queries_by_group") or {}).items():
            error_queries.setdefault(g, set()).update(str(q) for q in ids or [])
    merged["harness_error_queries_by_group"] = {g: sorted(v, key=_qid_order)
                                                for g, v in sorted(error_queries.items())}
    merged["harness_errors"] = [dict(e, shard=r.get("shard")) if isinstance(e, dict) else e
                                for r in results for e in (r.get("harness_errors") or [])]
    for field in ("sopr_std_by_group", "sopr_passes_by_group", "judge_outages_by_group",
                  "judge_invalid_replies_by_group", "sowr_votes_missing_by_group"):
        merged.pop(field, None)
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = (
            f"no SoPR verdict for group(s) {[g for g, v in sopr.items() if v is None]}")
    statuses = {r.get("status") for r in results}
    merged["status"] = "completed_with_errors" if "completed_with_errors" in statuses else "success"
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    # the judge's and the simulator's cut-off reasons are stated once, over the summed counts
    shard_cut_reasons = ({judge_cutoff_reason(r) for r in results}
                         | {simulator_cutoff_reason(r.get("simulator_usage")) for r in results})
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if reason and not reason.startswith(shard_prefix) and not reason.startswith(
                    "empty shard ") and reason not in kept and reason not in shard_cut_reasons:
                kept.append(reason)
    kept.extend(x for x in (judge_cutoff_reason(merged),
                            simulator_cutoff_reason(merged.get("simulator_usage"))) if x)
    merged["leaderboard_comparable"] = False       # the Gemini grader (see run_toolbench)
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    return merged
