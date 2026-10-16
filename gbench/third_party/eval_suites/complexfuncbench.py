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

# Evaluation Suite: complexfuncbench
# Description: ComplexFuncBench (THUDM) - multi-step, multi-constraint tool/function calling

"""gbench native built-in runner for complexfuncbench (Tool Use & Function Calling).

Canonical ComplexFuncBench (THUDM/ComplexFuncBench, arXiv:2501.10132) is NOT an
answer-matching dataset: each of the 1000 samples is a multi-STEP agentic tool-calling
episode (avg ~3, up to ~19 sequential calls, often several PARALLEL calls per step) in
which the model is driven through a conversation while the harness feeds back the
**recorded** observations for the calls it gets right (the "Golden Function Call List
Updating" strategy). Scoring is `ComplexEval`:

  * **Success Rate** - fraction of samples where the model completes the whole golden call
    chain and then stops at the right moment. HEADLINE `accuracy`.
  * **Call Accuracy** - correct calls / total golden calls, pooled.
  * **Completeness** / **Correctness** - a judge scores the final natural-language answer
    (0/1/2) for coverage of the request and factual consistency with the observations.

A predicted call is compared with a golden call as upstream's `utils/compare_method.py`
compare_single_call does (ported verbatim): (1) rule-based exact match, (2) a `value_checker` over
the per-function critical parameters (`utils/exact_match_values.json`, vendored), (3) the
response-based tier, then (4) the LLM equivalence judge. The response-based tier is upstream's
`response_based` with its `utils/rapidapi.py` RapidAPICall('booking-com15') (`utils/tool_info.json`,
vendored): both calls are sent to the live Booking.com API on RapidAPI with RAPID_API_KEY (read
from the environment, never written anywhere), each retried 3 times 2 s apart when it gets no
response (utils.utils.retry(max_attempts=3)), and the pair matches when the two responses are
equal. A call that got no response makes the tier answer "no match" and the judge decides, as
upstream; when the backend could not answer the pair (a call's last reply was an outage,
RESPONSE_TIER_OUTAGE: a network error, HTTP 5xx, 401, 403, 407 or 429, or an HTTP 200 whose body
is not JSON, and neither call got a reply that already makes the tier "no match") and the judge then finds the pair not equivalent,
upstream's verdict is unknown, so the sample is a harness error (`rapidapi_unavailable`), never
the model's failure. The result records every
RapidAPI call (`response_based_tier`, `call_match_tiers`; per sample `response_based_calls`).
Multiple parallel calls in one step are aligned to the golden calls with bge-large-en-v1.5
embeddings + a max-weight assignment (scipy.optimize.linear_sum_assignment, as upstream) before
comparison.

The two GPT-4o judge roles upstream uses run on the Gemini API through gbench's judge cascade
(base.judge_generate_cascade, GEMINI_API_KEYS or GEMINI_API_KEY): the call-equivalence judge
(compare_method.py llm_based, upstream gpt-4o-2024-05-13) and the response completeness /
correctness judge (runner/response_runner.py, upstream gpt-4o-2024-08-06). The upstream prompts
are copied verbatim (prompts/compare.py, prompts/response.py: the system prompts with their few-shot
examples, and the user templates filled as SimpleTemplatePrompt fills them) and sent as upstream's
GPTModel._predict sends them: the system prompt as the system instruction and the filled template as
the user content, at temperature 0.0. Replies are read with upstream's decode_json; the response
judges ask again up to 10 times when the reply has no 0/1/2 score (response_runner.py
retry(max_attempts=10)); the call-equivalence judge is asked once, as upstream's llm_based. Only
the judge model changes. The result records the substitution (`judge_substitution`), the cascade
(`judge_cascade`), the answered judge calls per Gemini model (`judge_models`,
`gemini_models_answered`) and the calls every model failed (`judge_outages`, both roles), and each
sample's record in `sample_traces` keeps every judge call with the model that answered it and its
reply. The
embedding model is reproduced with `transformers.AutoModel` (CLS pooling + L2 normalisation,
bge's FlagModel.encode) so no `FlagEmbedding` dependency is added to the environment; it runs at
the checkpoint's float32 where upstream's FlagModel uses fp16 (recorded: `embedding_dtype`,
`embedding_upstream`).

Harness errors are never the model's outcome. A sample is not scored (left out of the numerator
and the denominator of Success Rate and Call Accuracy, and out of the judge means) when the model
endpoint request failed on all 5 attempts, 20 s apart as upstream's models/qwen.py retries (an HTTP
error, a connection error or timeout, or a malformed reply; upstream's `unknown_error`, which its
evaluation.py writes no result for), when
the Gemini call-equivalence judge gave no verdict (every cascade model failed, or the reply held no
is_equal verdict: upstream's llm_based returns None there and the call then counts as not
equivalent, or raises on a decoded reply without is_equal), when the RapidAPI backend of the
response-based tier could not answer a pair the judge then found not equivalent, or when the
harness raised while driving it. A completeness / correctness judge that
gave no score leaves that score out of its mean (as upstream's -1). Any of them is counted in
`harness_errors`, makes the status `completed_with_errors` and is named in
`leaderboard_comparable_reason`; Success Rate is over the evaluable samples (`n_scored`) on full and
subset runs alike, which on a full run with no harness error is upstream's /1000. A model reply
with neither content nor tool calls is the model's outcome: it ends the sample as a failure
(`empty_model_message`, counted in `n_empty_model_message`), where upstream files it under
`unknown_error` with API failures.

The dataset is pinned: ComplexFuncBench.jsonl of THUDM/ComplexFuncBench at Hub commit
HF_REVISION, checked against HF_SHA256 after download (recorded: `dataset`, `dataset_sha256`).

Because the judges are Gemini, a run here is not a like-for-like leaderboard entry and is reported
with `leaderboard_comparable=False`.

HARD-ERRORS (`infra_required`, never skips, never a fabricated number) if the dataset cannot be
loaded or differs from the pinned sha256, no Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) is set
or the key is rejected, RAPID_API_KEY is unset or empty or the booking-com15 API refuses a live
check with it (an outage reply to each attempt; the response-based tier needs it; it is never
replaced by the judge), or `requests`, `transformers`, `torch` or `scipy` is missing.
See docs/evals/complexfuncbench.md.

Sampling: upstream's model request (models/qwen.py, the vLLM-served path; CFB_SAMPLING):
temperature 0.0 (gbench's rule for every suite, base.resolve_temperature: 0.0 without
`--thinking`, 1.0 with it), tool_choice "auto", and max_tokens 2048 through the token-limit
policy (base.resolve_token_budget): without `--thinking` every turn sends max_tokens 2048 whatever
`--max-output-tokens` says; with `--thinking` the 2048 is not applied (it was written for
non-reasoning generation) and every turn is sent the run's `--max-output-tokens`, else the
thinking floor. The result records `official_sampling`, `max_tokens_sent` and
`max_output_tokens_note`. Override the temperature with `--temperature` or
`GBENCH_COMPLEXFUNCBENCH_TEMPERATURE`. The model driver posts to the gbench endpoint with native
tool-calling (the served model must support tool_calls), and
`chat_template_kwargs.enable_thinking` follows `--thinking`. The Gemini judges run at 0.0, as
upstream's GPTModel.

Request timeout: each model request may take base.request_timeout_s of the turn's max_tokens (the
served context window when none is sent); GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S replaces it. Until
2026-10-11 it was a fixed 600 s, shorter than a --thinking turn of 65536 tokens needs, so such a
turn timed out and was asked again (a hidden cap). The result records `model_client_timeout`
({client, version, timeout_s, retries, source}).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .base import (MODELS_NOT_SERVED_PREFIX, SHARD_REASON, OfficialSampling,
                   add_comparability_reason, apply_shard, current_shard, gemini_api_keys,
                   gemini_key_live_valid,
                   judge_cascade, judge_config, judge_generate_cascade, record_shard, redact_text,
                   resolve_official_sampling, resolve_token_budget)
from .base import (UNCAPPED_TIMEOUT_TOKENS, get_run_knob, judge_client_record,
                   judge_fallback_counts, request_timeout_s)
from .fc_common import eval_limit_plan
from .sampling import stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/complexfuncbench.md"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "_complexfuncbench_data")
_EXACT_MATCH_PATH = os.path.join(_DATA_DIR, "exact_match_values.json")
#: upstream utils/tool_info.json, vendored byte for byte (sha256 TOOL_INFO_SHA256 at
#: UPSTREAM_COMMIT): the RapidAPI host and the URL of every booking-com15 function.
_TOOL_INFO_PATH = os.path.join(_DATA_DIR, "tool_info.json")
TOOL_INFO_SHA256 = "2c106f906171447e1a880771db241a5275c75bb3bb82bccb2f49e6260135bad9"

_HF_REPO = "THUDM/ComplexFuncBench"
_HF_FILE = "ComplexFuncBench.jsonl"
#: The dataset commit gbench reads (the Hub now redirects THUDM/ComplexFuncBench to
#: zai-org/ComplexFuncBench). ComplexFuncBench.jsonl was uploaded once (558831035a, 2025-01-17) and
#: is byte-identical at every later commit, including 484e0c48, the head when the GitHub code
#: revision UPSTREAM_COMMIT (2025-01-20) was made, and this one (2025-01-22, README metadata only).
HF_REVISION = "5dc7739343f8c87aee465931ba66db13a0fb5dd4"
#: sha256 of ComplexFuncBench.jsonl at HF_REVISION (its Git LFS oid; 40,195,573 bytes, 1000 rows),
#: checked after every download: a file that differs is a hard error.
HF_SHA256 = "be1e0f5951b666f6543b946e915a431a39bbda1807481878eb331245276ac088"

#: THUDM/ComplexFuncBench revision the prompts, cascade and judge models were read from.
UPSTREAM_COMMIT = "c37b284e2f2e03ee456115b7c4b7e537f534be37"

#: The per-turn max_tokens every upstream native tool-calling wrapper sends.
UPSTREAM_MAX_TOKENS = 2048
#: The model request of upstream's vLLM-served path (models/qwen.py QwenModel.__call__; the
#: GPT, GLM and Mistral wrappers send the same settings): temperature 0.0, tool_choice "auto",
#: max_tokens 2048. Applied through the token-limit policy (base.resolve_token_budget): a no-think
#: turn sends max_tokens 2048 whatever --max-output-tokens says; a --thinking turn is sent the
#: run's budget instead (the 2048 was written for non-reasoning generation). The temperature is
#: gbench's rule, 0.0 without --thinking and 1.0 with it (base.resolve_temperature), unless the
#: operator sets another.
CFB_SAMPLING = OfficialSampling(
    reference=(f"THUDM/ComplexFuncBench@{UPSTREAM_COMMIT[:12]} models/qwen.py QwenModel.__call__ "
               "(the vLLM-served path; models/gpt.py FunctionCallGPT sends the same): "
               "temperature=0.0, tool_choice='auto', max_tokens=2048"),
    temperature=0.0, max_tokens=UPSTREAM_MAX_TOKENS)

#: The call-comparison tiers gbench runs, in order (utils/compare_method.py compare_single_call).
CALL_MATCH_TIERS = ["rule_based", "value_checker", "response_based", "llm_based"]

#: The RapidAPI tool upstream's CompareFCBase builds (RapidAPICall(tool="booking-com15")), and the
#: variable its utils/rapidapi.py reads the key from.
RAPIDAPI_TOOL = "booking-com15"
RAPID_API_KEY_ENV = "RAPID_API_KEY"
#: utils/rapidapi.py RapidAPICall._call is wrapped in utils.utils.retry(max_attempts=3) (delay
#: 2 s), which calls it again while it returns None (no response).
_RAPIDAPI_ATTEMPTS = 3
_RAPIDAPI_RETRY_DELAY_S = 2.0
#: Seconds one RapidAPI request may take. Upstream's requests.get sets no timeout (it can wait
#: forever); a request past this one is a network error, which upstream's _call turns into None.
_RAPIDAPI_TIMEOUT_S = float(os.environ.get("GBENCH_COMPLEXFUNCBENCH_RAPIDAPI_TIMEOUT_S", "60"))
#: The booking-com15 function the live check before the run calls (utils/tool_info.json lists it).
_RAPIDAPI_CHECK_FUNCTION = "Test_API"
#: The replies of a RapidAPI call that mean the backend could not answer it (an outage of the
#: tool backend, not the model's call): a network error, a timeout, HTTP 5xx, 401 / 403 (the key
#: rejected or not subscribed), 407 (a proxy that wants credentials), 429 (rate limit or quota),
#: or an HTTP 200 whose body is not JSON (booking-com15 always answers JSON: such a body is a
#: proxy's or captive portal's page). A 200 JSON body without `status` stays the backend's answer.
RESPONSE_TIER_OUTAGE = ("network error or timeout", "HTTP 5xx", "HTTP 401", "HTTP 403",
                        "HTTP 407", "HTTP 429", "HTTP 200 unreadable body")
#: The label of a 200 reply whose body could not be decoded as JSON.
UNREADABLE_BODY = "HTTP 200 unreadable body"
_OUTAGE_STATUSES = (401, 403, 407, 429)
#: What gbench records about the response-based tier.
RESPONSE_BASED_TIER = {
    "run": True,
    "upstream": ("utils/compare_method.py response_based: both calls sent to the live "
                 "Booking.com API on RapidAPI (utils/rapidapi.py RapidAPICall('booking-com15'), "
                 "RAPID_API_KEY); equal when the two responses are identical; run after "
                 "value_checker, before llm_based; a call that gets no response, or whose "
                 "predicted call answers an empty object, is no match"),
    "gbench": ("run as upstream, in gbench's process: utils/tool_info.json vendored "
               "(TOOL_INFO_SHA256), RAPID_API_KEY read from the environment and never written "
               "anywhere, requests.get with upstream's headers and parameters (legs as JSON), "
               "each call retried 3 times 2 s apart while it gets no response, a "
               f"{int(_RAPIDAPI_TIMEOUT_S)} s timeout per request (upstream sets none); a pair "
               "the backend could not answer (a call's last reply a network error, HTTP 5xx, "
               "401, 403, 407 or 429, and neither call's reply one that already makes the tier "
               "no match: another HTTP status, or a 200 reply that cannot be read) goes to the "
               "judge as upstream, and is a harness error (rapidapi_unavailable) when the judge "
               "then finds it not equivalent; "
               "upstream's response_based also replaces the step's error list with the string "
               "'API call failed for ...' when the predicted call answers an empty object (a "
               "later error of that step then raises AttributeError); gbench keeps the list and "
               "records the message on the pair (outcome predict_empty_data)"),
}
#: The embedder upstream's utils/compare_method.py loads, recorded next to gbench's
#: (`embedding_backend`, `embedding_dtype`).
EMBEDDING_UPSTREAM = ("FlagEmbedding FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True): fp16 "
                      "weights; gbench runs the checkpoint through transformers AutoModel at its "
                      "float32 dtype (CLS pooling + L2 normalisation, as FlagModel.encode)")
#: The Gemini substitutions for upstream's two GPT-4o judge roles.
JUDGE_SUBSTITUTION = [
    {"role": "call-equivalence judge (utils/compare_method.py llm_based)",
     "upstream": "gpt-4o-2024-05-13 (OpenAI API)",
     "gbench": "Gemini API judge cascade (base.judge_generate_cascade), temperature 0.0"},
    {"role": "response completeness / correctness judge (runner/response_runner.py)",
     "upstream": "gpt-4o-2024-08-06 (OpenAI API)",
     "gbench": "Gemini API judge cascade (base.judge_generate_cascade), temperature 0.0"},
]

#: Location lookups etc. are OPTIONAL golden calls (upstream `free_function_list`): the model
#: may or may not issue them, so they never count for/against it and are auto-consumed.
_FREE_FUNCTION_LIST = [
    "Location_to_Lat_Long", "Search_Hotel_Destination", "Search_Attraction_Location",
    "Search_Car_Location", "Search_Flight_Location", "Taxi_Search_Location",
]

_EMBED_MODEL = os.environ.get("GBENCH_COMPLEXFUNCBENCH_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
_EMBED_DEVICE = os.environ.get("GBENCH_COMPLEXFUNCBENCH_EMBED_DEVICE", "cpu")
_MAX_ROUNDS = int(os.environ.get("GBENCH_COMPLEXFUNCBENCH_MAX_ROUNDS", "40"))
#: The operator's request timeout (seconds) for every model request, replacing the one sized
#: from the turn budget (model_client_timeout); unset by default.
REQUEST_TIMEOUT_ENV = "GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S"

# Upstream normalises Success Rate per domain (150 each; Cross=400; 1000 overall).
_DOMAIN_DENOM = {"Car-Rental": 150, "Hotels": 150, "Attraction": 150, "Flights": 150, "Cross": 400}
_FULL_TOTAL = 1000

_UNEXPECT_CALL_RESP = {
    "api_status": True,
    "content": "There is a problem with your api call, please double-check for possible problems.",
}

#: Attempts per model request before the sample ends in an endpoint failure, and the pause
#: between them: models/qwen.py QwenModel.__call__ is wrapped in retry(max_attempts=5, delay=20),
#: which asks again whenever the request raised.
_REQUEST_ATTEMPTS = 5
_REQUEST_RETRY_DELAY_S = 20.0


#: Seconds each model request may take to open its TCP connection (aiohttp sock_connect).
CONNECT_TIMEOUT_S = 10.0


def model_client_timeout(per_turn_max: Optional[int]) -> Dict[str, Any]:
    """The timeout of each model request (aiohttp ClientTimeout total) and its retries, as the
    result records them (`model_client_timeout`).

    Until 2026-10-11 it was a fixed 600 s, which a --thinking turn sent max_tokens 65536 cannot
    meet (65536 tokens at the ~120 tok/s one request of gemma-4-26B-A4B-it decodes is ~546 s with
    the endpoint otherwise idle, more under load): such a turn timed out, was asked again 4
    times and ended the sample as an endpoint failure, a hidden cap on the turn's length. The
    timeout is now base.request_timeout_s of the turn's max_tokens (2048 without --thinking, the
    run's budget with it), or of the served context window when none is sent;
    GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S replaces it. Upstream's 5 attempts stay. The TCP connect
    keeps a short bound of its own (CONNECT_TIMEOUT_S, aiohttp sock_connect), so an unreachable
    endpoint fails each attempt in seconds, not after the whole request timeout."""
    raw = (os.environ.get(REQUEST_TIMEOUT_ENV) or "").strip()
    if raw:
        timeout, source = int(float(raw)), f"{REQUEST_TIMEOUT_ENV}={raw} (operator)"
    else:
        tokens = int(per_turn_max or get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)
        timeout = request_timeout_s(tokens)
        what = "per-turn max_tokens" if per_turn_max else "served context window"
        source = f"base.request_timeout_s({what}={tokens})"
    try:
        import aiohttp
        version = aiohttp.__version__
    except ImportError:
        version = None
    return {"client": "aiohttp ClientSession (gbench's model driver)", "version": version,
            "timeout_s": timeout, "connect_timeout_s": CONNECT_TIMEOUT_S,
            "retries": _REQUEST_ATTEMPTS - 1,
            "source": source + f"; upstream's retry(max_attempts={_REQUEST_ATTEMPTS}, "
                               f"delay={int(_REQUEST_RETRY_DELAY_S)}) unchanged",
            "replaces": "a fixed 600 s",
            "hidden_cap_removed": not raw}

#: The harness errors a sample can end in. Such a sample is never the model's outcome: it is not
#: scored (left out of every metric, numerator and denominator), it is counted in
#: `harness_errors`, and it makes the run completed_with_errors.
HARNESS_ERROR_KINDS = {
    "endpoint_failure": (f"the model endpoint request failed on all {_REQUEST_ATTEMPTS} attempts "
                         "(an HTTP error, a connection error or timeout, or a malformed reply)"),
    "call_judge_outage": "the Gemini call-equivalence judge failed on every cascade model",
    "call_judge_unparsed_reply": ("the Gemini call-equivalence judge replied without an is_equal "
                                  "verdict"),
    "rapidapi_unavailable": ("the booking-com15 RapidAPI backend of upstream's response_based tier "
                             "could not answer a call pair (a call's last reply, after 3 "
                             "attempts, was a network error, HTTP 5xx, 401, 403, 407 or 429, "
                             "and neither call got another reply) and the Gemini "
                             "call-equivalence judge then found the pair not equivalent, so "
                             "upstream's verdict for the pair is unknown"),
    "exception": "the harness raised while driving the sample",
}
#: The reason a completeness / correctness score carries when the judge gave none.
_RESP_JUDGE_OUTAGE = "judge outage (every cascade model failed)"
_RESP_JUDGE_UNPARSED = "judge reply held no 0/1/2 score"


class _PairUnscored(Exception):
    """A call pair got no verdict gbench can score. A harness error: the sample is not scored.
    `kind` is a HARNESS_ERROR_KINDS key."""

    def __init__(self, kind: str) -> None:
        super().__init__(HARNESS_ERROR_KINDS[kind])
        self.kind = kind


class _CallJudgeNoVerdict(_PairUnscored):
    """The Gemini call-equivalence judge gave no verdict for a call pair (`kind`
    call_judge_outage or call_judge_unparsed_reply)."""


class _ResponseTierUnavailable(_PairUnscored):
    """The RapidAPI backend of the response-based tier could not answer a pair the judge then
    found not equivalent (`kind` rapidapi_unavailable)."""

    def __init__(self) -> None:
        super().__init__("rapidapi_unavailable")


# --------------------------------------------------------------------------- #
# Judge prompts - VERBATIM from upstream prompts/compare.py and prompts/response.py at
# UPSTREAM_COMMIT: the system prompts (their leading '"' character, few-shot examples and output
# notes included) and the user templates, which SimpleTemplatePrompt fills by replacing [args1],
# [args2], ... in order (_fill_template). models/gpt.py GPTModel._predict sends the system prompt
# as the system message and the filled template as the user message at temperature 0.0; gbench
# sends the same two texts as the Gemini system_instruction and the user content
# (_judge_call). tests/test_tool_research_suites_prompts_and_harness_errors.py pins each string to
# upstream's sha256.
# --------------------------------------------------------------------------- #
_COMPARE_SYSTEM = (
    "\"You are an assistant for function call comparison. Your task is to determine whether "
    "two function calls are equivalent based on the conversation history and the function "
    "descriptions, and provide specific reasons.\n"
    "# Instructions:\n"
    "You need to determine whether two function calls are equivalent based on the following "
    "criteria:\n"
    "1. The same parameter can be expressed in different languages. For example: `America`, "
    "`\u7f8e\u56fd` and `\u30a2\u30e1\u30ea\u30ab` are equivalent.\n"
    "2. The same parameter can be expressed in different forms as long as the meaning is the "
    "same. For example, `New York` and `NY` are equivalent, `Shanghai` and `Shanghai City` are "
    "equivalent. `Narita International Airport` and `Tokyo Narita International Airport` are "
    "equivalent. \n"
    "3. A location with or without a country suffix is considered equivalent. For example, Van "
    "Gogh Museum, Amsterdam and Van Gogh Museum are equivalent.\n"
    "4. The order of parameters in a function call can differ, for example: `add(1, 2)` and "
    "`add(2, 1)` are equivalent.\n"
    "5. If a parameter in the function description has a default value, and the current "
    "parameter value is equal to that default value, then the parameter can be omitted in the "
    "function call. For example, if the adults parameter in the Search_Hotels function has a "
    "default value of 1, then the following two function calls are equivalent: "
    "`Search_Hotels(New_York, adults=1)` and `Search_Hotels(New_York)`.\n"
    "\n"
    "# Output:\n"
    "You need to output the result in JSON format, containing the following fields:\n"
    "- **is_equal**: A boolean indicating whether the two function calls are equivalent.\n"
    "- **reason**: Please provide the reason for your judgment.\n"
    "\n"
    "## Example 1\n"
    "output:\n"
    "```JSON\n"
    "{\"is_equal\": true, \"reason\": \"xx and yy are equivalent.\"}\n"
    "```\n"
    "\n"
    "## Example 2\n"
    "output:\n"
    "```JSON\n"
    "{\"is_equal\": false, \"reason\": \"The parameter in call xx is not present in call "
    "yy.\"}\n"
    "```\n"
    "\n"
    "Note: The output must be a valid JSON format that can be properly loaded.\n"
)
_COMPARE_USER_TEMPLATE = (
    "Function list:\n"
    "```JSON\n"
    "[args1]\n"
    "```\n"
    "Conversation history:\n"
    "```JSON\n"
    "[args2]\n"
    "```\n"
    "Function call 1:\n"
    "```JSON\n"
    "[args3]\n"
    "```\n"
    "Function call 2:\n"
    "```JSON\n"
    "[args4]\n"
    "```\n"
    "Please determine whether Function call 1 and Function call 2 are equivalent and provide "
    "your reason.\n"
    "output:\n"
    "\n"
)
_COMPLETE_SYSTEM = (
    "\"You are a helpful response completeness detect assistant. Your task is to evaluate the "
    "response based on whether it fully addresses all aspects of the user's query. \n"
    "# Your Task\n"
    "For each user query and corresponding response, you should determine the completeness of "
    "the response using the following criteria:\n"
    "- If the response covers all requested information and addresses all parts of the user's "
    "query, it should be considered complete and receive a score of 2.\n"
    "- If the response addresses some but not all parts of the user's query, it should be "
    "considered partial and receive a score of 1.\n"
    "- If the response does not address any of the requested information in the user's query, "
    "it should be considered incomplete and receive a score of 0.\n"
    "\n"
    "# Output Format\n"
    "You should output the score for each user query and corresponding response in JSON format "
    "with following keys:\n"
    "- score: the completeness score for the response (0, 1, or 2)\n"
    "- reason: a string describing the reason for the score (e.g. \"all requested information "
    "is addressed\", \"some requested information is not addressed\", \"no requested "
    "information is addressed\")\n"
    "\n"
    "# Example\n"
    "## Example 1\n"
    "input:\n"
    "query: I'm thinking about heading to Paris. Can you tell me which museums are currently "
    "trending?\n"
    "response: The Louvre is the most treding museums in Paris.\n"
    "\n"
    "output:\n"
    "```JSON\n"
    "{\"score\": 2, \"reason\": \"all requested information is addressed\"}\n"
    "```\n"
    "\n"
    "## Example 2\n"
    "input:\n"
    "query: I'm planning to stay in Milan for a week starting on November 2nd, 2024. Help me "
    "find detailed information about the most trending attractions in the center of Milan and "
    "book a hotel nearby this attraction.\n"
    "response: The Milan Cathedral (Duomo di Milano), located in the central square of Milan, "
    "Italy, is the largest Gothic cathedral in the world. It is a Roman Catholic cathedral and "
    "ranks among the top five churches in the world. \n"
    "Construction of the Milan Cathedral began in 1386, the vault was completed in 1500, and "
    "the highest Gothic spire, topped with the gilded statue of the Virgin Mary (La "
    "Madonnina), was built by Giuseppe Perego in 1774. It is a symbol of Milan city. The "
    "entire cathedral was completed in 1965, taking five centuries to finish. Napoleon held "
    "his coronation ceremony at the Milan Cathedral in 1805.\n"
    "\n"
    "output:\n"
    "```JSON\n"
    "{\"score\": 1, \"reason\": \"Only give the detailed information about the most trending "
    "attractions in the center of Milan. Miss the hotel booking part.\"}\n"
    "```\n"
    "\n"
    "## Example 3\n"
    "input:\n"
    "query: My partner and I are gonna fly from Chicago to Tokyo on September 5th, 2024. Then "
    "we'll fly from Tokyo to Shanghai on September 9th, 2024. Finally, we'll go back to "
    "Chicago on September 19th, 2024. Can you give us the info about economy class?\n"
    "response: Sorry, I can't find any info about economy class. \n"
    "\n"
    "output:\n"
    "```JSON\n"
    "{\"score\": 0, \"reason\": \"The response did not give any information about economy "
    "class.\"}\n"
    "```\n"
)
_COMPLETE_USER_TEMPLATE = (
    "input:\n"
    "query: [args1]\n"
    "response: [args2]\n"
    "\n"
    "output:\n"
    "    \n"
)
_CORRECT_SYSTEM = (
    "\"You are a helpful response correctness detect assistant. Your task is to evaluate the "
    "response based on its accuracy in matching the details provided by API response.\n"
    "# Your task\n"
    "Give a dialogue history containing user query, function calls and api responses, you "
    "should determine the correctness of the corresponding respons using the following "
    "criteria:\n"
    "- If the response is consistent with the information provided in the API response, it "
    "should be considered entirely correct and receive a score of 2.\n"
    "- If the response partially matches the information provided in the API response (with "
    "some correct and some incorrect details), it should be considered partially correct and "
    "receive a score of 1.\n"
    "- If the response does not match any of the information provided in the API response, it "
    "should be considered incorrect and receive a score of 0.\n"
    "\n"
    "# Output Format\n"
    "You should output the score for each dialogue history and corresponding response in JSON "
    "format with following keys:\n"
    "- score: the completeness score for the response (0, 1, or 2)\n"
    "- reason: a string describing the reason for the score (e.g. \"all mentioned information "
    "is correct.\", \"Some information is correct, and some are incorrect.\", \"All "
    "information in the response is incorrect.\")\n"
    "\n"
    "## example output 1\n"
    "output:\n"
    "```JSON\n"
    "{\"score\": 2, \"reason\": \"All mentioned information is correct.\"}\n"
    "```\n"
    "\n"
    "## example output 2\n"
    "output:\n"
    "```JSON\n"
    "{\"score\": 1, \"reason\": \"xx is correct, and yy is incorrect.\"}\n"
    "```\n"
    "\n"
    "## example output 3\n"
    "output:\n"
    "```JSON\n"
    "{\"score\": 0, \"reason\": \"All information in the response is incorrect.\"}\n"
    "```\n"
)
_CORRECT_USER_TEMPLATE = (
    "dialogue history: [args1]\n"
    "response: [args2]\n"
    "output:\n"
    "\n"
)
#: sha256 of each prompt string above at UPSTREAM_COMMIT.
PROMPT_SHA256 = {
    "_COMPARE_SYSTEM": "7f7d670cbaeb5ced4c0b025f2c2b28d2136997a50da9a2a2cac8388affc8aeb7",
    "_COMPARE_USER_TEMPLATE": "57bfc74530b2770cae3c4eaaddeacd6c811cd3f18251c3abb220b186a6a5c84e",
    "_COMPLETE_SYSTEM": "cc2c4bf91d0eaed3378d11eb6aa851df9b11528d6a059394610f661129841efe",
    "_COMPLETE_USER_TEMPLATE": "7911d9c582fff5fa8e226d39f29fe32e55a05c8348a749fb7862ee4c605c11f2",
    "_CORRECT_SYSTEM": "0228930f6e953e4aeef89ecd0d59033882217802f0484a55ba675d6400753c7b",
    "_CORRECT_USER_TEMPLATE": "f5e73785a04922c4319c74f85e254619787cf6b1d06ff6185c378f47109c93ba",
}


def _fill_template(template: str, *args: Any) -> str:
    """prompts/prompts.py SimpleTemplatePrompt.__call__: replace [args1], [args2], ... in order
    (an int is turned into its str first)."""
    out = template
    for i, arg in enumerate(args):
        if isinstance(arg, int):
            arg = str(arg)
        out = out.replace(f"[args{i + 1}]", arg)
    return out


# --------------------------------------------------------------------------- #
# Judge reply parsing: upstream utils/utils.py decode_json, and the checks its callers make.
# --------------------------------------------------------------------------- #
def _decode_json(json_str: Optional[str]) -> Any:
    """utils/utils.py decode_json, verbatim: strip the fence characters, drop newlines, turn
    Python booleans into JSON ones and json.loads; None when that fails (or no reply)."""
    if json_str is None:
        return None
    json_str = json_str.strip('```JSON\n').strip('```json\n').strip('\n```')
    json_str = json_str.replace('\n', '').replace('False', 'false').replace('True', 'true')
    try:
        return json.loads(json_str)
    except Exception:
        return None


def _parse_is_equal(text: Optional[str]) -> Optional[bool]:
    """The call-equivalence verdict as utils/compare_method.py llm_based reads it: decode_json,
    then `decode_output['is_equal']`, used for its truth value by compare_single_call. None when
    upstream returns None (no decodable reply) or raises (a decoded reply that is not an object
    with an is_equal field); gbench makes either a harness error, never a verdict."""
    obj = _decode_json(text)
    if not obj or not isinstance(obj, dict) or "is_equal" not in obj:
        return None
    return bool(obj["is_equal"])


def _parse_score(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """runner/response_runner.py completeness_eval / correctness_eval: the decoded reply when it
    is an object whose score is 0, 1 or 2, else None (upstream then asks again)."""
    obj = _decode_json(text)
    if not isinstance(obj, dict) or "score" not in obj:
        return None
    if obj["score"] not in [0, 1, 2]:
        return None
    return obj


#: Characters of a judge reply kept in a per-sample record.
_JUDGE_REPLY_CHARS = 4000


async def _judge_call(system: str, user: str) -> Tuple[Optional[str], str]:
    """One judge request, as GPTModel._predict makes it: `system` as the system message (the
    Gemini system_instruction) and `user` as the user message, at temperature 0.0 and with no
    other generation setting. `(text, model)` from judge_generate_cascade."""
    return await judge_generate_cascade(user, config=judge_config(system_instruction=system))


# --------------------------------------------------------------------------- #
# Embedding model: bge-large-en-v1.5 through transformers AutoModel (CLS pooling + L2 norm,
# faithful to bge's FlagModel.encode, which upstream calls). Cached module-wide.
# --------------------------------------------------------------------------- #
class _Embedder:
    def __init__(self) -> None:
        self._impl = None

    def _lazy(self) -> None:
        if self._impl is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(_EMBED_MODEL)
        mdl = AutoModel.from_pretrained(_EMBED_MODEL).to(_EMBED_DEVICE).eval()
        self._impl = (tok, mdl)

    @property
    def backend(self) -> str:
        return "transformers"

    @property
    def dtype(self) -> Optional[str]:
        """The loaded model's parameter dtype (e.g. "float32"); None while it is not loaded (it
        loads only when a step has several non-exact parallel calls to align)."""
        if self._impl is None:
            return None
        return str(next(self._impl[1].parameters()).dtype).replace("torch.", "")

    def encode(self, texts: List[str]):
        import torch
        self._lazy()
        tok, mdl = self._impl
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(_EMBED_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            emb = out.last_hidden_state[:, 0]  # CLS token, matching bge FlagModel pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()


_EMBEDDER: Optional[_Embedder] = None


def _get_embedder() -> _Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _Embedder()
    return _EMBEDDER


def _assign_max(matrix) -> Tuple[List[int], List[int]]:
    """Max-weight assignment, as upstream: scipy.optimize.linear_sum_assignment(-matrix).

    Returns (row_ind, col_ind) of length min(n_rows, n_cols) maximising the summed weight.
    Assignment order does not affect scoring (each pair is compared independently and keyed by
    its original predicted-call index)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    m = np.asarray(matrix, dtype=float)
    if min(m.shape) == 0:
        return [], []
    rows, cols = linear_sum_assignment(-m)
    return [int(r) for r in rows], [int(c) for c in cols]


# --------------------------------------------------------------------------- #
# RapidAPICall - ported from upstream utils/rapidapi.py (the booking-com15 tool of
# utils/tool_info.json), for the response-based tier.
# --------------------------------------------------------------------------- #
def _load_tool_info() -> Dict[str, Any]:
    """utils/tool_info.json's booking-com15 entry (host, path_params, name_to_url)."""
    with open(_TOOL_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)[RAPIDAPI_TOOL]


def _rapidapi_get(url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: float):
    """utils/rapidapi.py's request: requests.get(url, headers=..., params=...), which honors the
    proxy variables of the environment (HTTPS_PROXY, ALL_PROXY, NO_PROXY) as upstream's does."""
    import requests
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def _reply_is_outage(label: str) -> bool:
    """Whether a RapidAPI call's last reply means the backend could not answer it
    (RESPONSE_TIER_OUTAGE): a network error or timeout (the exception's class), HTTP 5xx, 401,
    403, 407 or 429. Any other HTTP status is the backend's answer to that call."""
    if label == "ok":
        return False
    m = re.match(r"HTTP (\d+)$", label)
    if not m:
        return True
    code = int(m.group(1))
    return code >= 500 or code in _OUTAGE_STATUSES


def _call_state(reply: Dict[str, Any]) -> str:
    """A _RapidAPICall.call reply as the response-based tier reads it: "ok" (a response),
    "outage" (no response, the last reply an outage, _reply_is_outage) or "failed" (no response,
    the backend's own answer: any other HTTP status, or a 200 reply that could not be read)."""
    if reply["response"] is not None:
        return "ok"
    if reply["labels"] and reply["labels"][-1] == UNREADABLE_BODY:
        return "outage"
    if reply["raised"] or not _reply_is_outage(reply["labels"][-1]):
        return "failed"
    return "outage"


class _UnreadableBody(Exception):
    """An HTTP 200 reply whose body is not JSON (upstream's _call raises there too)."""


class _RapidAPICall:
    """utils/rapidapi.py RapidAPICall(tool="booking-com15"): the key header, the URL of each
    function and the request, as upstream sends them. `call` is RapidAPICall._call under
    utils.utils.retry(max_attempts=3): asked again, 2 s apart, while it gets no response; an
    exception after a 200 reply (a body that is not JSON, or has no `status`) is raised at once,
    as upstream's retry does not catch it. The key is never logged or recorded."""

    def __init__(self, tool_info: Dict[str, Any], api_key: str,
                 stats: Optional[Counter] = None) -> None:
        self.name_to_url = tool_info["name_to_url"]
        self.path_params = tool_info["path_params"]
        self._headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": tool_info["host"]}
        #: counts over the run: calls, attempts, and each reply that was not a response
        self.stats = stats if stats is not None else Counter()

    def _call_once(self, func_call: Dict[str, Any]) -> Tuple[Any, str]:
        """One RapidAPICall._call attempt: (response, "ok"), or (None, why: "HTTP <status>" or
        the class of the request's exception). Raises where upstream's _call raises."""
        url = self.name_to_url[func_call["name"]]
        params_copy = copy.deepcopy(func_call["arguments"])
        param_dict = {}
        for path_param in self.path_params:
            if f"{{{path_param}}}" in url and path_param in params_copy:
                param_dict[path_param] = params_copy.pop(path_param)
        url = url.format(**param_dict)
        for k, value in params_copy.items():
            if k == "legs":
                params_copy[k] = json.dumps(value, ensure_ascii=False)
        try:
            response = _rapidapi_get(url, self._headers, params_copy, _RAPIDAPI_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 - upstream's bare except: no response
            return None, type(e).__name__
        if response.status_code == 200:
            try:
                response = response.json()
            except ValueError as e:     # JSONDecodeError: not booking-com15's JSON
                raise _UnreadableBody(str(e)) from e
            if response['status'] == True:  # noqa: E712 - upstream's comparison, verbatim
                if "timestamp" in response:
                    response.pop("timestamp")
                if "data" in response:
                    response = response['data']
            return response, "ok"
        return None, f"HTTP {response.status_code}"

    async def call(self, func_call: Dict[str, Any]) -> Dict[str, Any]:
        """{"response": the response or None, "raised": the reply could not be read (the
        exception's class in the last label), "labels": each attempt's reply}."""
        labels: List[str] = []
        self.stats["calls"] += 1
        out: Dict[str, Any] = {"response": None, "raised": False, "labels": labels}
        for attempt in range(_RAPIDAPI_ATTEMPTS):
            self.stats["attempts"] += 1
            try:
                response, label = await asyncio.to_thread(self._call_once, func_call)
            except _UnreadableBody:
                # not the backend's answer: an outage of the tool backend (RESPONSE_TIER_OUTAGE)
                labels.append(UNREADABLE_BODY)
                out["raised"] = True
                break
            except Exception as e:  # noqa: BLE001 - raised by upstream's _call as well
                labels.append(f"unreadable reply ({type(e).__name__})")
                out["raised"] = True
                break
            labels.append(label)
            if response is not None:
                out["response"] = response
                break
            if attempt < _RAPIDAPI_ATTEMPTS - 1:
                await asyncio.sleep(_RAPIDAPI_RETRY_DELAY_S)
        if out["response"] is None:
            self.stats["calls_without_response"] += 1
            self.stats["no_response:" + labels[-1]] += 1
        return out


def rapidapi_live_check(api_key: str) -> Optional[str]:
    """Before the run: one call of _RAPIDAPI_CHECK_FUNCTION with the key, up to _RAPIDAPI_ATTEMPTS
    attempts 2 s apart. None when booking-com15 answered with anything but an outage
    (_reply_is_outage: a network error, HTTP 5xx, 401, 403, 407 or 429), else why not (the HTTP
    status or the exception's class; the key never appears)."""
    info = _load_tool_info()
    url = info["name_to_url"][_RAPIDAPI_CHECK_FUNCTION]
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": info["host"]}
    why = "no attempt"
    for attempt in range(_RAPIDAPI_ATTEMPTS):
        try:
            status = _rapidapi_get(url, headers, {}, _RAPIDAPI_TIMEOUT_S).status_code
        except Exception as e:  # noqa: BLE001
            why = f"{type(e).__name__} (network error or timeout)"
        else:
            if not _reply_is_outage(f"HTTP {status}"):
                return None
            why = f"HTTP {status}"
        if attempt < _RAPIDAPI_ATTEMPTS - 1:
            time.sleep(_RAPIDAPI_RETRY_DELAY_S)
    return why


# --------------------------------------------------------------------------- #
# CompareFC - ported from upstream utils/compare_method.py. The LLM equivalence judge
# (`llm_based`) is async and uses gbench's Gemini cascade; the response-based tier
# (`response_based`) calls the live booking-com15 API through _RapidAPICall.
# --------------------------------------------------------------------------- #
class _CompareFC:
    def __init__(self, exact_match_dict: Dict[str, List[str]], embedder: _Embedder,
                 judge_usage: Optional[Counter] = None,
                 api_call: Optional[_RapidAPICall] = None) -> None:
        self.exact_match_dict = exact_match_dict
        self.embedding = embedder
        #: upstream's self.api_call (RapidAPICall(tool="booking-com15")); the response-based
        #: tier cannot run without it
        self.api_call = api_call
        self.free_function_list = list(_FREE_FUNCTION_LIST)
        self.free_functions: Dict[str, Dict[str, Any]] = {}
        self.error_message: List[Any] = []
        #: judge calls per Gemini model that answered (JUDGE_OUTAGE: every model failed)
        self.judge_usage = judge_usage if judge_usage is not None else Counter()
        #: why the last llm_based call returned None (a HARNESS_ERROR_KINDS key), else None
        self.judge_failure: Optional[str] = None
        #: every call-equivalence judge call of the sample: the two calls, the Gemini model that
        #: answered (JUDGE_OUTAGE: none did), its reply and the verdict read from it
        self.judge_calls: List[Dict[str, Any]] = []
        #: every response-based comparison of the sample: the two calls, each call's replies and
        #: the outcome (equal, different, no_response, predict_empty_data, unavailable)
        self.response_calls: List[Dict[str, Any]] = []
        #: whether the backend could not answer the last pair the response-based tier compared
        self.response_unavailable = False

    # -- format + free-function bookkeeping --------------------------------- #
    def format_check(self, func_call: Dict[str, Any], functions: List[Dict[str, Any]]):
        name_to_func = {f["name"]: f for f in functions}
        if func_call["name"] not in name_to_func:
            return {"error": f"Function {func_call['name']} is not defined in the function list."}
        used = name_to_func[func_call["name"]]
        required = used["parameters"].get("required", [])
        args = func_call.get("arguments", {})
        if not set(required).issubset(set(args.keys())):
            missing = set(required) - set(args.keys())
            return {"error": f"Function {used['name']} requires parameters {required}, but "
                    f"{list(args.keys())} do not provide {missing}"}
        props = used["parameters"].get("properties", {})
        if not set(args.keys()).issubset(set(props.keys())):
            missing = set(args.keys()) - set(props.keys())
            return {"error": f"Function {used['name']} does not have parameters {missing}"}
        for pname, pval in args.items():
            ptype = props.get(pname, {}).get("type")
            if ptype == "string" and not isinstance(pval, str):
                return {"error": f"Parameter {pname} of function {used['name']} should be a "
                        f"string, but {type(pval)} is provided."}
            if ptype == "number" and not isinstance(pval, (int, float)):
                return {"error": f"Parameter {pname} of function {used['name']} should be a "
                        f"number, but {type(pval)} is provided."}
            if ptype == "boolean" and not isinstance(pval, bool):
                return {"error": f"Parameter {pname} of function {used['name']} should be a "
                        f"boolean, but {type(pval)} is provided."}
            if ptype == "array" and not isinstance(pval, list):
                return {"error": f"Parameter {pname} of function {used['name']} should be an "
                        f"array, but {type(pval)} is provided."}
        return True

    def add_free_function(self, convs: List[Dict[str, Any]]) -> None:
        self.free_functions = {}
        for i, turn in enumerate(convs):
            if "function_call" not in turn:
                continue
            for j, fc in enumerate(turn["function_call"]):
                if fc["name"] in self.free_function_list:
                    key = json.dumps(fc)
                    if key not in self.free_functions:
                        self.free_functions[key] = {"called": False, "obs": convs[i + 1]["content"][j]}

    # -- the 3-tier match cascade: rule_based -> value_checker -> llm_based --- #
    def rule_based(self, predict: Dict[str, Any], golden: Dict[str, Any]) -> bool:
        if predict["name"] != golden["name"]:
            return False
        if sorted(predict.get("arguments", {}).keys()) != sorted(golden.get("arguments", {}).keys()):
            return False
        for k, v in predict.get("arguments", {}).items():
            if k == "categories_filter":
                pred_f = [s.strip() for s in str(v).split(",")]
                gold_f = [s.strip() for s in str(golden["arguments"][k]).split(",")]
                if set(gold_f) == set(pred_f):
                    continue
            if v != golden["arguments"][k]:
                return False
        return True

    def value_checker(self, pred_call: Dict[str, Any], golden_call: Dict[str, Any]):
        if pred_call["name"] != golden_call["name"]:
            return False, {"error_type": "func_error", "content": "Do not call the correct function."}
        # Defensive .get: every dataset function is a booking-com15 tool present in the asset;
        # an unknown name yields no critical params to check (upstream would KeyError).
        param_list = self.exact_match_dict.get(pred_call["name"], [])
        for k, v in golden_call.get("arguments", {}).items():
            if k in param_list:
                if k not in pred_call.get("arguments", {}):
                    return False, {"error_type": "param_missing",
                                   "content": f"Missing parameter {k} in prediction."}
                if k != "categories_filter" and pred_call["arguments"][k] != v:
                    return False, {"error_type": "value_error",
                                   "content": f"Parameter {k} value is not correct in prediction."}
                if k == "categories_filter":
                    gold_f = [s.strip() for s in str(v).split(",")]
                    pred_f = [s.strip() for s in str(pred_call["arguments"][k]).split(",")]
                    if set(gold_f) != set(pred_f):
                        return False, {"error_type": "value_error",
                                       "content": f"Parameter {k} value is not correct in prediction."}
        return True, ""

    async def response_based(self, predict, golden) -> bool:
        """utils/compare_method.py CompareFCBase.response_based: the two calls' live responses
        compared, True when they are equal. As upstream: a predicted call answering an empty
        object is no match, and so is a call that got no response or whose reply could not be
        read; the golden call is sent unless the predicted call's reply raised (upstream's
        `except: return False`). `response_unavailable` is set only when upstream's verdict is
        unknown: a call's last reply was an outage (_call_state "outage": a network error, a
        timeout, HTTP 5xx, 401, 403, 407 or 429) and no call got a reply that already makes the
        tier "no match" (_call_state "failed": any other HTTP status, or a 200 reply that cannot
        be read); compare_single_call then asks the judge, as upstream, and makes a
        not-equivalent verdict a harness error. Every comparison is kept in `response_calls`."""
        if self.api_call is None:
            raise RuntimeError("complexfuncbench: the response-based tier has no RapidAPI client")
        self.response_unavailable = False
        rec: Dict[str, Any] = {"predict": copy.deepcopy(predict), "golden": copy.deepcopy(golden)}
        self.response_calls.append(rec)
        r1 = await self.api_call.call(predict)
        rec["predict_replies"] = list(r1["labels"])
        if not r1["raised"] and r1["response"] == {}:
            # upstream: self.error_message = f"API call failed for {predict}."; return False
            return self._response_outcome(rec, "predict_empty_data",
                                          message=f"API call failed for {predict}.")
        if r1["raised"] and _call_state(r1) == "outage":
            # a 200 body that is not JSON: an outage of the backend, not its answer
            self.response_unavailable = True
            return self._response_outcome(rec, "unavailable")
        if r1["raised"]:
            # upstream's _call raised on the reply (a 200 JSON body without `status`): its except
            # returns False before the golden call is sent
            return self._response_outcome(rec, "predict_unreadable")
        r2 = await self.api_call.call(golden)
        rec["golden_replies"] = list(r2["labels"])
        s1, s2 = _call_state(r1), _call_state(r2)
        if s1 == "failed":
            # the backend's answer to the model's call: no match whatever the golden call got
            return self._response_outcome(rec, "no_response")
        if s2 == "failed":
            # the backend's answer to the golden call: upstream's tier cannot match this pair
            return self._response_outcome(rec, "golden_no_response")
        if "outage" in (s1, s2):
            self.response_unavailable = True
            return self._response_outcome(rec, "unavailable")
        equal = r1["response"] == r2["response"]
        self._response_outcome(rec, "equal" if equal else "different")
        return equal

    def _response_outcome(self, rec: Dict[str, Any], outcome: str,
                          message: Optional[str] = None) -> bool:
        """Record a response-based comparison's outcome (and upstream's message, if any) on the
        pair and in the run's counts; False (the tier found no match)."""
        rec["outcome"] = outcome
        if message is not None:
            rec["message"] = message
        self.api_call.stats[outcome if outcome == "predict_empty_data"
                            else "pairs_" + outcome] += 1
        return False

    async def llm_based(self, functions, history, predict, golden) -> Optional[bool]:
        user = _fill_template(_COMPARE_USER_TEMPLATE,
                              json.dumps(functions, ensure_ascii=False),
                              json.dumps(history, ensure_ascii=False),
                              json.dumps(predict, ensure_ascii=False),
                              json.dumps(golden, ensure_ascii=False))
        text, used = await _judge_call(_COMPARE_SYSTEM, user)
        self.judge_usage[used] += 1
        verdict = _parse_is_equal(text)
        # upstream llm_based makes one request (GPTModel._predict retries only a failed API call,
        # which the cascade does); an undecodable reply is not asked again
        self.judge_calls.append({"predict": predict, "golden": golden, "judge_model": used,
                                 "reply": None if text is None else text[:_JUDGE_REPLY_CHARS],
                                 "is_equal": verdict})
        # None, as upstream's llm_based returns when its judge call yields no verdict; the reason
        # is kept for compare_single_call, which makes it a harness error.
        self.judge_failure = (None if verdict is not None else
                              "call_judge_outage" if used == "JUDGE_OUTAGE" else
                              "call_judge_unparsed_reply")
        return verdict

    # -- multi-call alignment (bge + max-assignment) ----------------------- #
    def mapping_call(self, predict, golden, golden_obs):
        def _sort(call_list):
            for value in call_list:
                value["arguments"] = {k: value["arguments"][k] for k in sorted(value.get("arguments", {}))}
        _sort(predict)
        _sort(golden)

        exact_matches, remaining_predict, remaining_golden = [], [], []
        remaining_predict_index, remaining_golden_index = {}, {}
        matched_indices = set()

        for p_index, p_value in enumerate(predict):
            match_found = False
            for g_index, g_value in enumerate(golden):
                if g_index in matched_indices:
                    continue
                if p_value == g_value:
                    exact_matches.append({"idx": p_index, "pred_call": p_value,
                                          "golden_call": g_value, "golden_obs": golden_obs[g_index]})
                    matched_indices.add(g_index)
                    match_found = True
                    break
                if json.dumps(p_value) in self.free_functions:
                    exact_matches.append({"idx": p_index, "pred_call": p_value,
                                          "golden_call": p_value,
                                          "golden_obs": self.free_functions[json.dumps(p_value)]["obs"]})
                    self.free_functions[json.dumps(p_value)]["called"] = True
                    match_found = True
                    break
            if not match_found:
                remaining_predict.append(p_value)
                remaining_predict_index[len(remaining_predict) - 1] = p_index

        for g_index, g_value in enumerate(golden):
            if g_index not in matched_indices:
                remaining_golden.append(g_value)
                remaining_golden_index[len(remaining_golden) - 1] = g_index

        if not remaining_predict or not remaining_golden:
            return exact_matches

        # 1x1 fast-path: a single remaining pred vs a single remaining golden always aligns to
        # each other (a 1x1 assignment is trivial), so skip the embedder entirely - this is the
        # common single-call-per-step mismatch case and keeps it dependency-free.
        if len(remaining_predict) == 1 and len(remaining_golden) == 1:
            return exact_matches + [{
                "idx": remaining_predict_index[0], "pred_call": remaining_predict[0],
                "golden_call": remaining_golden[0], "golden_obs": golden_obs[remaining_golden_index[0]]}]

        pred_embed = self.embedding.encode([json.dumps(v, ensure_ascii=False) for v in remaining_predict])
        gold_embed = self.embedding.encode([json.dumps(v, ensure_ascii=False) for v in remaining_golden])
        matrix = pred_embed @ gold_embed.T
        row_ind, col_ind = _assign_max(matrix)

        embedding_matches = []
        for i, j in zip(row_ind, col_ind):
            embedding_matches.append({"idx": remaining_predict_index[i],
                                      "pred_call": remaining_predict[i],
                                      "golden_call": remaining_golden[j],
                                      "golden_obs": golden_obs[remaining_golden_index[j]]})
        return exact_matches + embedding_matches

    def remove_called_fc(self, golden, golden_obs):
        pop_index = []
        for single in golden:
            key = json.dumps(single)
            if key in self.free_functions and self.free_functions[key]["called"]:
                pop_index.append(golden.index(single))
        for idx in sorted(pop_index, reverse=True):
            golden.pop(idx)
            golden_obs.pop(idx)
        return golden, golden_obs

    def get_error_message(self, pred_call, golden_call):
        for k, v in golden_call.get("arguments", {}).items():
            if k not in pred_call.get("arguments", {}):
                return {"error_type": "param_missing", "content": f"Missing parameter {k} in prediction."}
            if v != pred_call["arguments"][k]:
                return {"error_type": "value_error", "content": f"Parameter {k} value do not equal to golden."}
        for k in pred_call.get("arguments", {}):
            if k not in golden_call.get("arguments", {}):
                return {"error_type": "param_hallucination", "content": f"Parameter {k} is hallucinated."}
        return None

    async def compare_single_call(self, functions, history, pred_call, golden_call):
        """Upstream's cascade (CALL_MATCH_TIERS): rule_based, value_checker, response_based,
        llm_based.

        A judge that gives no verdict raises _CallJudgeNoVerdict: a harness error, so the sample
        is not scored (upstream counts the call as not equivalent). A pair the response-based
        tier's backend could not answer, which the judge then finds not equivalent, raises
        _ResponseTierUnavailable: upstream's verdict for it is unknown."""
        if self.rule_based(pred_call, golden_call):
            return True, None
        is_valid, err = self.value_checker(pred_call, golden_call)
        if not is_valid:
            return False, err
        if await self.response_based(pred_call, golden_call):
            return True, None
        unavailable = self.response_unavailable
        self.judge_failure = None
        verdict = await self.llm_based(functions, history, pred_call, golden_call)
        if verdict is None:
            raise _CallJudgeNoVerdict(self.judge_failure or "call_judge_unparsed_reply")
        if verdict:
            return True, None
        if unavailable:
            raise _ResponseTierUnavailable()
        return False, None

    async def compare_turn_prediction(self, functions, history, predict, golden, golden_obs):
        self.error_message = []
        golden, golden_obs = self.remove_called_fc(golden, golden_obs)
        if len(golden) == 0:
            # All golden calls this step were already-consumed free functions; nothing to match.
            return self.error_message, {}, [], {}
        match_list = self.mapping_call(predict, golden, golden_obs)
        format_error, success_map, success_matched = {}, {}, []
        for item in match_list:
            msg = self.format_check(item["pred_call"], functions)
            if msg is True:
                is_match, single = await self.compare_single_call(
                    functions, history, item["pred_call"], item["golden_call"])
                if is_match:
                    success_map[item["idx"]] = item["golden_obs"]
                    success_matched.append(item["golden_call"])
                else:
                    self.error_message.append(single or self.get_error_message(
                        item["pred_call"], item["golden_call"]))
            else:
                format_error[item["idx"]] = msg
        return self.error_message, success_map, success_matched, format_error


# --------------------------------------------------------------------------- #
# ModelRunner - ported from upstream runner/base_runner.py + runner/gpt_runner.py.
# Drives the served model through the golden call chain, feeding recorded observations.
# --------------------------------------------------------------------------- #
class _Runner:
    def __init__(self, compare: _CompareFC) -> None:
        self.CompareClass = compare
        self.error_message: List[Any] = []
        self.unexpect_call_resp = _UNEXPECT_CALL_RESP

    # -- golden state machine ---------------------------------------------- #
    def only_free_function(self, temp_fcs) -> bool:
        for call in temp_fcs:
            if call["name"] == "Search_Hotels" and call.get("arguments", {}).get("search_type") == "hotel":
                return True
        return set(fc["name"] for fc in temp_fcs).issubset(set(self.CompareClass.free_function_list))

    def get_success_turn(self, remain_fcs, total_fcs) -> int:
        remain_ids = []
        for idx, fc_list in enumerate(total_fcs):
            for remain_fc in remain_fcs:
                if remain_fc in fc_list:
                    remain_ids.append(idx)
        if not remain_ids:
            return len(total_fcs)
        return max(min(remain_ids), 0)

    def init_golden(self, convs) -> None:
        self.fc_chain, self.obs_chain = [], []
        for turn in convs:
            if "function_call" in turn:
                self.fc_chain.append(turn["function_call"])
            elif turn["role"] == "observation":
                self.obs_chain.append(turn["content"])
        assert len(self.fc_chain) == len(self.obs_chain), "function call / observation length mismatch"
        self.turn_id, self.correct_count = 0, 0
        self.golden_fcs = copy.deepcopy(self.fc_chain[self.turn_id])
        self.golden_obs = copy.deepcopy(self.obs_chain[self.turn_id])
        if self.only_free_function(self.golden_fcs):
            self.update_current_golden()

    def update_current_golden(self) -> None:
        self.turn_id += 1
        if self.turn_id < len(self.fc_chain):
            self.golden_fcs.extend(copy.deepcopy(self.fc_chain[self.turn_id]))
            self.golden_obs.extend(copy.deepcopy(self.obs_chain[self.turn_id]))

    def process_matches(self, success_matched) -> None:
        for matched in success_matched:
            if matched in self.golden_fcs:
                self.golden_obs.pop(self.golden_fcs.index(matched))
                self.golden_fcs.remove(matched)
        if len(success_matched) > 0:
            self.update_current_golden()
        for k, v in self.CompareClass.free_functions.items():
            if v["called"] and json.loads(k) in self.golden_fcs:
                self.golden_obs.pop(self.golden_fcs.index(json.loads(k)))
                self.golden_fcs.remove(json.loads(k))
        if self.only_free_function(self.golden_fcs):
            self.update_current_golden()

    def return_result(self, messages, error_info=None):
        if error_info:
            success_turn = self.get_success_turn(self.golden_fcs, self.fc_chain)
            return messages, error_info, success_turn, self.correct_count
        # Free-function post-process (iterate a copy: upstream mutates during iteration, which
        # can skip elements; the intent is to drop leftover optional calls before the stop check).
        if len(self.golden_fcs) != 0:
            for call in list(self.golden_fcs):
                if call["name"] == "Search_Hotels" and call.get("arguments", {}).get("search_type") == "hotel":
                    if call in self.golden_fcs:
                        self.golden_fcs.remove(call)
                if call["name"] in ["Search_Hotel_Destination", "Search_Attraction_Location",
                                    "Search_Car_Location", "Search_Flight_Location", "Taxi_Search_Location"]:
                    if call in self.golden_fcs:
                        self.golden_fcs.remove(call)
        if self.turn_id < len(self.fc_chain) or len(self.golden_fcs) > 0:
            return self.return_result(messages, {"error_type": "stop_early", "content": "Stop early."})
        if len(self.golden_fcs) == 0:
            return messages, "Success.", len(self.fc_chain), self.correct_count
        raise RuntimeError("Unexpected complexfuncbench return_result state.")

    # -- name sanitisation for OpenAI-style tool schemas ------------------- #
    @staticmethod
    def _sanitize(name: str) -> str:
        return "".join(c if re.match(r"[a-zA-Z0-9_-]", c) else "-" for c in name)[:64]

    def get_standard_functions(self, functions):
        self.name_dict = {api["name"]: self._sanitize(api["name"]) for api in functions}
        tools = [{"type": "function", "function": copy.deepcopy(f)} for f in functions]
        for t in tools:
            t["function"]["name"] = self.name_dict[t["function"]["name"]]
        return tools

    def _decode_tool_call(self, tc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fn = tc.get("function", {})
        sanitized = fn.get("name")
        orig = next((k for k, v in self.name_dict.items() if v == sanitized), None)
        if orig is None:
            return None
        raw = fn.get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return None
        return {"name": orig, "arguments": args if isinstance(args, dict) else {}}

    # -- the driving loop -------------------------------------------------- #
    async def run(self, data, chat):
        """`chat(oai_messages, tools) -> message_dict | None` posts to the gbench endpoint; None
        means the request failed on every attempt (upstream's unknown_error, a harness error)."""
        convs, functions = data["conversations"], data["functions"]
        self.CompareClass.add_free_function(convs)
        tools = self.get_standard_functions(functions)

        query = convs[0]["content"]
        messages = [{"role": "user", "content": query}]          # abstract record (for judges)
        self.messages = messages                                 # kept when the sample ends early
        oai_messages = [{"role": "user", "content": query}]      # what the model actually sees
        self.init_golden(convs)

        rounds = 0
        while True:
            rounds += 1
            if rounds > _MAX_ROUNDS:
                return self.return_result(messages, {"error_type": "max_rounds",
                                                     "content": f"Exceeded {_MAX_ROUNDS} rounds."})
            msg = await chat(oai_messages, tools)
            if msg is None:
                return self.return_result(messages, {"error_type": "unknown_error",
                                                     "content": "llm_response is None"})
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")

            if tool_calls:
                if self.golden_fcs == []:
                    return self.return_result(messages, {"error_type": "func_hallucination",
                                                         "content": "Expected to stop, but model "
                                                         "continued to output function calls."})
                oai_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                function_calls = []
                for tc in tool_calls:
                    fc = self._decode_tool_call(tc)
                    if fc is None:
                        return self.return_result(messages, {"error_type": "decode_error",
                                                             "content": f"{tc.get('function')} is not valid."})
                    function_calls.append(fc)
                messages.append({"role": "assistant", "function_call": function_calls})

                (self.error_message, success_map, success_matched,
                 format_error) = await self.CompareClass.compare_turn_prediction(
                    functions, messages[:-1], copy.deepcopy(function_calls),
                    self.golden_fcs, self.golden_obs)
                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)
                self.correct_count += len(success_map)

                real_time_obs = []
                for t, fc in enumerate(function_calls):
                    if t in success_map:
                        obs = success_map[t]
                    elif t in format_error:
                        obs = format_error[t]
                    else:
                        obs = self.unexpect_call_resp
                    real_time_obs.append(obs)
                    oai_messages.append({"tool_call_id": tool_calls[t].get("id", f"call_{t}"),
                                         "role": "tool", "name": self.name_dict[fc["name"]],
                                         "content": json.dumps(obs, ensure_ascii=False)})
                self.process_matches(success_matched)
                messages.append({"role": "observation", "content": real_time_obs})

            elif content is not None:
                messages.append({"role": "assistant", "content": content})
                return self.return_result(messages, self.error_message)
            else:
                # The model answered with neither content nor tool calls: its own outcome, a
                # failed sample (upstream files this under unknown_error with API failures).
                return self.return_result(messages, {"error_type": "empty_model_message",
                                                     "content": "The model returned neither "
                                                     "content nor tool calls."})


# --------------------------------------------------------------------------- #
# Response completeness / correctness judges (upstream RespEvalRunner -> Gemini).
# --------------------------------------------------------------------------- #
#: runner/response_runner.py: completeness_eval and correctness_eval are wrapped in
#: utils.utils.retry(max_attempts=10) (delay 2 s), which asks again whenever the reply does not
#: decode to an object with a 0/1/2 score.
_RESP_JUDGE_ATTEMPTS = 10
_RESP_JUDGE_RETRY_DELAY_S = 2.0


async def _resp_judge(system: str, user: str, calls: Optional[List[Dict[str, Any]]],
                      role: str, judge_usage: Optional[Counter]) -> Dict[str, Any]:
    """One completeness / correctness verdict, as upstream's retry(max_attempts=10) wrapper gets
    it: up to _RESP_JUDGE_ATTEMPTS requests until a reply decodes to an object with a 0/1/2 score.
    A cascade outage ends it at once (upstream's model call returns None there and its decode_json
    raises). The score and reason are the judge's, as RespEvalRunner.run records them; -1 when it
    gave none (a harness error, left out of the mean)."""
    for attempt in range(_RESP_JUDGE_ATTEMPTS):
        text, used = await _judge_call(system, user)
        if judge_usage is not None:
            judge_usage[used] += 1
        decoded = _parse_score(text)
        if calls is not None:
            calls.append({"role": role, "attempt": attempt + 1, "judge_model": used,
                          "reply": None if text is None else text[:_JUDGE_REPLY_CHARS],
                          "score": None if decoded is None else decoded["score"]})
        if used == "JUDGE_OUTAGE":
            return {"score": -1, "reason": _RESP_JUDGE_OUTAGE}
        if decoded is not None:
            return {"score": decoded["score"], "reason": decoded.get("reason", None)}
        if attempt < _RESP_JUDGE_ATTEMPTS - 1:
            await asyncio.sleep(_RESP_JUDGE_RETRY_DELAY_S)
    return {"score": -1, "reason": _RESP_JUDGE_UNPARSED}


async def _resp_eval(convs, gen_response: str, judge_usage: Optional[Counter] = None,
                     calls: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """runner/response_runner.py RespEvalRunner.run over the Gemini judge cascade. `calls`, when
    given, receives every judge request (role, attempt, answering model, reply, score)."""
    if gen_response == "":
        return {"complete": {"score": -2, "reason": "Do not generate response successfully."},
                "correct": {"score": -2, "reason": "Do not generate response successfully."}}
    complete_user = _fill_template(_COMPLETE_USER_TEMPLATE, convs[0]["content"], gen_response)
    correct_user = _fill_template(_CORRECT_USER_TEMPLATE,
                                  json.dumps(convs[:-1], ensure_ascii=False), gen_response)
    complete = await _resp_judge(_COMPLETE_SYSTEM, complete_user, calls, "completeness",
                                 judge_usage)
    correct = await _resp_judge(_CORRECT_SYSTEM, correct_user, calls, "correctness",
                                judge_usage)
    return {"complete": complete, "correct": correct}


# --------------------------------------------------------------------------- #
# Per-sample evaluation + metric aggregation.
# --------------------------------------------------------------------------- #
async def _evaluate_one(item, chat, exact_match_dict, embedder,
                        judge_usage: Optional[Counter] = None,
                        api_call: Optional[_RapidAPICall] = None) -> Optional[Dict[str, Any]]:
    """One sample's ComplexEval record, or a harness-error record (`harness_error` set to a
    HARNESS_ERROR_KINDS key) for a sample that is not scored. Either carries what upstream's
    evaluation.py process_example writes for a sample (`gen_convs`, the conversation as driven),
    every Gemini judge call made for it (`call_judge_calls`, `resp_judge_calls`) and every
    response-based comparison (`response_based_calls`). `api_call` is the run's _RapidAPICall."""
    compare = _CompareFC(exact_match_dict, embedder, judge_usage, api_call)
    runner = _Runner(compare)

    def _partial(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec["gen_convs"] = copy.deepcopy(getattr(runner, "messages", None) or [])
        rec["call_judge_calls"] = list(compare.judge_calls)
        rec["response_based_calls"] = list(compare.response_calls)
        rec["resp_judge_calls"] = []
        return rec

    try:
        convs, message, turn_id, correct_count = await runner.run(item, chat)
    except _PairUnscored as e:
        logger.warning("complexfuncbench sample %s not scored: %s", item.get("id"), e)
        return _partial({"id": item.get("id"), "request_failed": False, "harness_error": e.kind})
    except Exception as e:
        # Only the exception's class is recorded in the result; the log line is redacted.
        logger.warning("complexfuncbench sample %s not scored: the harness raised %s: %s",
                       item.get("id"), type(e).__name__, redact_text(str(e))[:300])
        return _partial({"id": item.get("id"), "request_failed": False,
                         "harness_error": "exception", "exception": type(e).__name__})

    if isinstance(message, dict) and message.get("error_type") == "unknown_error":
        # chat() returned None: the endpoint request failed on every attempt.
        return _partial({"id": item.get("id"), "request_failed": True,
                         "harness_error": "endpoint_failure", "message": message})

    turn_count = call_count = 0
    for turn in item["conversations"]:
        if turn.get("role") == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])
    real_turn = sum(1 for t in convs if t.get("role") == "assistant" and "function_call" in t)

    resp_eval = None
    resp_calls: List[Dict[str, Any]] = []
    if convs and convs[-1].get("role") == "assistant" and "content" in convs[-1]:
        resp_eval = await _resp_eval(item["conversations"], convs[-1]["content"],
                                     judge_usage, resp_calls)

    return {
        "id": item.get("id"),
        "gen_convs": copy.deepcopy(convs),
        "message": message,
        "count_dict": {"success_turn_num": turn_id, "total_turn_num": turn_count,
                       "correct_call_num": correct_count, "total_call_num": call_count,
                       "real_turn_num": real_turn},
        "resp_eval": resp_eval,
        "request_failed": False,
        "call_judge_calls": list(compare.judge_calls),
        "response_based_calls": list(compare.response_calls),
        "resp_judge_calls": resp_calls,
    }


def _harness_kind(r: Dict[str, Any]) -> Optional[str]:
    """The harness error a per-sample record ended in (a HARNESS_ERROR_KINDS key), else None."""
    kind = r.get("harness_error")
    if kind:
        return kind
    return "endpoint_failure" if r.get("request_failed") else None


def _domain(sample_id: Any) -> str:
    return str(sample_id).rsplit("-", 1)[0]


def _aggregate(results: List[Dict[str, Any]], n_total: int, is_full: bool) -> Dict[str, Any]:
    """ComplexEval (upstream print_results.py basic_metric) over the evaluable samples.

    A sample that ended in a harness error (_harness_kind) is left out of every metric, numerator
    and denominator, on full and subset runs alike, and is counted instead. Success Rate is
    successes / evaluable samples; on a full run with no harness error that is upstream's /1000,
    and each domain's rate is over its evaluable samples (upstream's /150, Cross /400). Call
    Accuracy pools the evaluable samples' calls. Completeness / Correctness average the 0/1/2
    scores (a judge that gave no score, -1, and an empty final answer, -2, are left out, as
    upstream leaves out every score outside 0/1/2). `n_total` is the number of samples the run
    drove; no denominator uses it."""
    results = [r for r in results if r]
    scored = [r for r in results if _harness_kind(r) is None]
    failed = [r for r in results if _harness_kind(r) is not None]
    kinds = Counter(_harness_kind(r) for r in failed)

    success = sum(1 for r in scored if r.get("message") == "Success.")
    corr_calls = sum(r["count_dict"]["correct_call_num"] for r in scored)
    tot_calls = sum(r["count_dict"]["total_call_num"] for r in scored)
    success_rate = round(success / len(scored) * 100.0, 2) if scored else 0.0
    call_accuracy = round(corr_calls / tot_calls * 100.0, 2) if tot_calls else 0.0

    means: Dict[str, Optional[float]] = {}
    unscored = {"completeness": 0, "correctness": 0}
    resp_outages = 0
    for metric, key in (("completeness", "complete"), ("correctness", "correct")):
        vals = []
        for r in scored:
            entry = (r.get("resp_eval") or {}).get(key) or {}
            if entry.get("score") in (0, 1, 2):
                vals.append(entry["score"])
            elif entry.get("score") == -1:
                unscored[metric] += 1
                resp_outages += entry.get("reason") == _RESP_JUDGE_OUTAGE
        means[metric] = round(sum(vals) / len(vals), 4) if vals else None   # raw 0..2 (upstream)

    per_domain = per_domain_n = None
    if is_full:
        succ, n = Counter(), Counter()
        for r in scored:
            n[_domain(r["id"])] += 1
            succ[_domain(r["id"])] += r.get("message") == "Success."
        per_domain = {d: (round(succ[d] / n[d] * 100.0, 2) if n[d] else None)
                      for d in _DOMAIN_DENOM}
        per_domain_n = {d: n[d] for d in _DOMAIN_DENOM}

    return {
        "success_rate": success_rate,
        "call_accuracy": call_accuracy,
        "completeness": means["completeness"],
        "correctness": means["correctness"],
        "n_scored": len(scored),
        "n_request_failed": kinds.get("endpoint_failure", 0),
        "n_success": success,
        "per_domain_success_rate": per_domain,
        "per_domain_evaluated": per_domain_n,
        "n_harness_errors": len(failed),
        "harness_error_samples": dict(sorted(kinds.items())),
        "endpoint_errors": dict(sorted(Counter(
            r.get("endpoint_error") or "unrecorded" for r in failed
            if _harness_kind(r) == "endpoint_failure").items())),
        "exception_types": dict(sorted(Counter(
            r.get("exception") or "unrecorded" for r in failed
            if _harness_kind(r) == "exception").items())),
        "response_judge_unscored": unscored,
        "response_judge_outages": resp_outages,
        "n_empty_model_message": sum(
            1 for r in scored if isinstance(r.get("message"), dict)
            and r["message"].get("error_type") == "empty_model_message"),
    }


# --------------------------------------------------------------------------- #
# Dataset + gate.
# --------------------------------------------------------------------------- #
def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_dataset(limit: Optional[int]) -> List[Dict[str, Any]]:
    """The pinned dataset (HF_REVISION, checked against HF_SHA256): --shard applied to the full
    ordered rows first (base.apply_shard), then stratified (by domain) to `limit` within the
    shard."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE, repo_type="dataset",
                               revision=HF_REVISION)
    except Exception as e:
        raise infra_required(
            "complexfuncbench",
            f"could not fetch {_HF_REPO}/{_HF_FILE} at revision {HF_REVISION} from the HF Hub "
            f"({e}). Needs network/HF access (or an HF cache holding that revision).", DOCS_URL)
    digest = _sha256(path)
    if digest != HF_SHA256:
        raise infra_required(
            "complexfuncbench",
            f"{_HF_FILE} at revision {HF_REVISION} has sha256 {digest}, not the pinned "
            f"{HF_SHA256} ({path}); remove that file from the HF cache and run again.", DOCS_URL)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise infra_required("complexfuncbench", f"{_HF_FILE} loaded but empty.", DOCS_URL)
    rows, _shard_label = apply_shard(rows, "complexfuncbench")
    rows = stratified_sample(rows, limit, lambda r: str((r or {}).get("id", "")).rsplit("-", 1)[0],
                             seed="complexfuncbench")
    return rows


def _load_exact_match_dict() -> Dict[str, List[str]]:
    with open(_EXACT_MATCH_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_complexfuncbench_prerequisites() -> Tuple[bool, str]:
    keys = gemini_api_keys()
    if not keys:
        return False, ("no Gemini key: set GEMINI_API_KEYS or GEMINI_API_KEY (GEMINI_API_KEYS wins "
                       "when both are set). It powers gbench's Gemini judge cascade in the "
                       "call-equivalence and response completeness/correctness roles.")
    _ok, _why = gemini_key_live_valid(",".join(keys))
    if not _ok:
        if str(_why).startswith(MODELS_NOT_SERVED_PREFIX):
            return False, (f"{_why}; every model of the Gemini judge cascade must be one the API "
                           "serves (fail-fast before the full run).")
        return False, (f"the Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) was rejected by the "
                       f"judge endpoint ({_why}); a valid key is required for the Gemini judge "
                       "cascade (fail-fast before the full run).")
    if not (os.environ.get(RAPID_API_KEY_ENV) or "").strip():
        return False, (f"{RAPID_API_KEY_ENV} is not set: upstream's response_based call-match tier "
                       "sends both calls of a pair to the live Booking.com API on RapidAPI "
                       f"({RAPIDAPI_TOOL}, utils/rapidapi.py), with the key of a RapidAPI "
                       f"account subscribed to {RAPIDAPI_TOOL}; it is a tool backend, never "
                       "replaced by the Gemini judge.")
    if not os.path.exists(_EXACT_MATCH_PATH):
        return False, (f"vendored exact_match_values.json missing at {_EXACT_MATCH_PATH}.")
    if not os.path.exists(_TOOL_INFO_PATH) or _sha256(_TOOL_INFO_PATH) != TOOL_INFO_SHA256:
        return False, (f"vendored tool_info.json at {_TOOL_INFO_PATH} is missing or differs from "
                       f"upstream's utils/tool_info.json at {UPSTREAM_COMMIT[:12]} "
                       f"(sha256 {TOOL_INFO_SHA256}).")
    for mod, why in (("requests", "for the response-based tier's RapidAPI calls "
                      "(utils/rapidapi.py requests.get)"),
                     ("huggingface_hub", "to fetch the dataset (pip install huggingface-hub)"),
                     ("transformers", "for the bge-large-en-v1.5 embedder that aligns parallel "
                      "calls (the serving stack's transformers)"),
                     ("torch", "for the bge-large-en-v1.5 embedder (the serving stack's torch)"),
                     ("scipy", "for the max-weight call alignment, "
                      "scipy.optimize.linear_sum_assignment (pip install scipy)")):
        try:
            __import__(mod)
        except Exception:
            return False, f"{mod} is required {why}."
    why_not = rapidapi_live_check(os.environ[RAPID_API_KEY_ENV].strip())
    if why_not:
        return False, (f"the {RAPIDAPI_TOOL} API on RapidAPI refused a live check with "
                       f"{RAPID_API_KEY_ENV} ({why_not} from {_RAPIDAPI_CHECK_FUNCTION} after "
                       f"{_RAPIDAPI_ATTEMPTS} attempts; HTTP 401/403: the key is rejected or its "
                       f"account is not subscribed to {RAPIDAPI_TOOL}; 429: rate limit or quota); "
                       "the response-based tier needs it (fail-fast before the full run).")
    return True, ""


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
#: An HTTP 400 body that says the request did not fit the served context window.
_CONTEXT_WINDOW_RE = re.compile(r"context length|max_model_len", re.IGNORECASE)


async def _post_chat(session, api_url: str, payload: Dict[str, Any],
                     failures: Optional[List[str]] = None,
                     replies: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """POST one model request, up to _REQUEST_ATTEMPTS attempts _REQUEST_RETRY_DELAY_S apart
    (upstream's retry(max_attempts=5, delay=20)). Returns the reply's message, or None when every
    attempt failed (an endpoint failure, a harness error). Each failed attempt appends a label to
    `failures`: "HTTP <status>" ("HTTP 400 (context window)" when the served window rejected the
    request), the exception class of a connection error or timeout, or "malformed reply" for a 200
    without a message. A reply appends its finish_reason, completion tokens and the max_tokens
    the request carried to `replies`."""
    for attempt in range(_REQUEST_ATTEMPTS):
        try:
            async with session.post(api_url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    try:
                        choice = data["choices"][0]
                        msg = choice["message"]
                    except (KeyError, IndexError, TypeError):
                        choice, msg = None, None
                    if isinstance(msg, dict):
                        if replies is not None:
                            usage = data.get("usage") if isinstance(data, dict) else None
                            replies.append({
                                "finish_reason": (choice or {}).get("finish_reason"),
                                "completion_tokens": (usage or {}).get("completion_tokens")
                                if isinstance(usage, dict) else None,
                                "max_tokens_sent": payload.get("max_tokens"),
                                "tool_calls": len(msg.get("tool_calls") or []),
                                "failed_attempts": attempt})
                        return msg
                    label = "malformed reply"
                else:
                    body = await resp.text()
                    label = f"HTTP {resp.status}"
                    if resp.status == 400 and _CONTEXT_WINDOW_RE.search(body or ""):
                        label += " (context window)"
                    logger.warning("complexfuncbench HTTP %s: %s", resp.status,
                                   redact_text((body or "")[:200]))
        except Exception as e:
            label = type(e).__name__
            logger.debug("complexfuncbench request error (attempt %d): %s", attempt + 1,
                         redact_text(str(e)))
        if failures is not None:
            failures.append(label)
        if attempt < _REQUEST_ATTEMPTS - 1:
            await asyncio.sleep(_REQUEST_RETRY_DELAY_S)
    return None


def _sample_trace(idx: int, r: Dict[str, Any], turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The per-sample record of the result: upstream process_example's fields (id, gen_convs,
    message, count_dict, resp_eval), every Gemini judge call with the model that answered it and
    its reply, the model's turns (finish_reason, completion tokens, max_tokens sent) and, for a
    sample that is not scored, its harness error."""
    kind = _harness_kind(r)
    trace: Dict[str, Any] = {
        "sample_idx": idx,
        "id": r.get("id"),
        # the sample's own id, the same in every shard (merge_shards de-duplicates by it)
        "sample_key": f"id:{r.get('id')}",
        "category": _domain(r.get("id")),
        "status": "HARNESS_ERROR" if kind else "OK",
        "harness_error": kind,
        "scoring_excluded": bool(kind),
        "is_correct": (not kind) and r.get("message") == "Success.",
        "message": r.get("message"),
        "count_dict": r.get("count_dict"),
        "resp_eval": r.get("resp_eval"),
        "gen_convs": r.get("gen_convs") or [],
        "call_judge_calls": r.get("call_judge_calls") or [],
        "response_based_calls": r.get("response_based_calls") or [],
        "resp_judge_calls": r.get("resp_judge_calls") or [],
        "turns": turns,
    }
    for key in ("endpoint_error", "exception"):
        if r.get(key):
            trace[key] = r[key]
    return trace


async def _drive_all(model_name, base_url, concurrency, temperature, limit,
                     max_output_tokens=None, enable_thinking=False):
    """Drive every sample. `max_output_tokens` is the per-turn max_tokens each request carries
    (the token-limit policy's value; None sends none)."""
    import aiohttp

    api = base_url.rstrip("/")
    if not api.endswith("/v1"):
        api += "/v1"
    api_url = api + "/chat/completions"

    items = _load_dataset(limit)
    shard = current_shard()
    shard_label = shard.label if shard else None
    if shard_label and not items:
        return {"_empty_shard": shard_label}
    # The planned workload, logged before the first request (each sample is driven once).
    limit_plan = eval_limit_plan("complexfuncbench", limit, len(items), attempts=1,
                                 unit="conversations")
    exact_match_dict = _load_exact_match_dict()
    embedder = _get_embedder()
    judge_usage: Counter = Counter()
    rapidapi_stats: Counter = Counter()
    api_call = _RapidAPICall(_load_tool_info(), (os.environ.get(RAPID_API_KEY_ENV) or "").strip(),
                             rapidapi_stats)

    is_full = ((not limit or limit <= 0 or limit >= _FULL_TOTAL) and len(items) >= _FULL_TOTAL
               and not shard_label)
    n_total = _FULL_TOTAL if is_full else len(items)

    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    client_timeout = model_client_timeout(max_output_tokens)
    timeout = aiohttp.ClientTimeout(total=client_timeout["timeout_s"],
                                    sock_connect=client_timeout["connect_timeout_s"])
    sample_sem = asyncio.Semaphore(max(1, concurrency))
    turns_by_idx: Dict[int, List[Dict[str, Any]]] = {}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def _one(idx, item):
            async with sample_sem:
                failures: List[str] = []
                turns = turns_by_idx.setdefault(idx, [])

                async def chat(oai_messages, tools):
                    payload = {"model": model_name, "messages": oai_messages, "tools": tools,
                               "tool_choice": "auto", "temperature": temperature,
                               **({"max_tokens": int(max_output_tokens)}
                                  if max_output_tokens is not None else {}),
                               "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
                    return await _post_chat(session, api_url, payload, failures, turns)

                r = await _evaluate_one(item, chat, exact_match_dict, embedder, judge_usage,
                                        api_call)
                if r.get("harness_error") == "endpoint_failure":
                    # the last attempt's error of the request that failed
                    r["endpoint_error"] = failures[-1] if failures else "unrecorded"
                return r

        results = await asyncio.gather(*[_one(i, it) for i, it in enumerate(items)])

    agg = _aggregate(results, n_total, is_full)
    agg["_shard"] = shard_label
    agg["_is_full"] = is_full
    agg["_n_items"] = len(items)
    agg["_limit_plan"] = limit_plan
    agg["_client_timeout"] = client_timeout
    agg["_embed_backend"] = embedder.backend
    agg["_embed_dtype"] = embedder.dtype
    agg["_judge_usage"] = judge_usage
    agg["_rapidapi_stats"] = rapidapi_stats
    agg["_sample_traces"] = [_sample_trace(i, r, turns_by_idx.get(i, []))
                             for i, r in enumerate(results) if r]
    return agg


def _counts_text(counts: Dict[str, int]) -> str:
    return ", ".join(f"{k} x{v}" for k, v in counts.items())


def _response_tier_record(stats: Counter) -> Dict[str, Any]:
    """`response_based_tier` of the result: RESPONSE_BASED_TIER plus the run's RapidAPI counts:
    calls (each up to 3 attempts), attempts, calls that got no response by their last reply,
    and the pairs by outcome (equal: matched by this tier; different, no_response,
    predict_unreadable, golden_no_response, predict_empty_data: the judge decided; unavailable:
    the backend could not answer, the judge decided and a not-equivalent verdict is a harness
    error)."""
    rec: Dict[str, Any] = dict(RESPONSE_BASED_TIER)
    rec["tool"] = RAPIDAPI_TOOL
    rec["tool_info_sha256"] = TOOL_INFO_SHA256
    rec["calls"] = int(stats.get("calls", 0))
    rec["attempts"] = int(stats.get("attempts", 0))
    rec["calls_without_response"] = dict(sorted(
        (k.split(":", 1)[1], int(v)) for k, v in stats.items() if k.startswith("no_response:")))
    for key in ("pairs_equal", "pairs_different", "pairs_no_response", "pairs_predict_unreadable",
                "pairs_golden_no_response", "predict_empty_data", "pairs_unavailable"):
        rec[key] = int(stats.get(key, 0))
    return rec


def _harness_error_report(agg: Dict[str, Any], judge_outages: int
                          ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(harness_errors, reason parts) of a run: every harness error, counted and named. None of
    them is the model's outcome, and none is in the metrics (see _aggregate)."""
    kinds = agg.get("harness_error_samples") or {}
    errors: List[Dict[str, Any]] = []
    parts: List[str] = []
    n = int(kinds.get("endpoint_failure", 0))
    if n:
        by_error = dict(agg.get("endpoint_errors") or {})
        errors.append({"error": HARNESS_ERROR_KINDS["endpoint_failure"], "samples_unscored": n,
                       "by_error": by_error})
        parts.append(f"{n} sample(s) not scored because the model endpoint request failed on all "
                     f"{_REQUEST_ATTEMPTS} attempts"
                     + (f" (last error: {_counts_text(by_error)})" if by_error else ""))
    outage = int(kinds.get("call_judge_outage", 0))
    unparsed = int(kinds.get("call_judge_unparsed_reply", 0))
    if outage or unparsed:
        errors.append({"error": "the Gemini call-equivalence judge gave no verdict",
                       "samples_unscored": outage + unparsed, "outages": outage,
                       "unparsed_replies": unparsed})
        parts.append(f"{outage + unparsed} sample(s) not scored because the Gemini "
                     f"call-equivalence judge gave no verdict ({outage} failed on every cascade "
                     f"model, {unparsed} replied without an is_equal verdict)")
    n = int(kinds.get("rapidapi_unavailable", 0))
    if n:
        errors.append({"error": HARNESS_ERROR_KINDS["rapidapi_unavailable"],
                       "samples_unscored": n})
        parts.append(f"{n} sample(s) not scored because the booking-com15 RapidAPI backend of the "
                     "response-based tier could not answer a call pair the Gemini judge then "
                     "found not equivalent")
    n = int(kinds.get("exception", 0))
    if n:
        by_type = dict(agg.get("exception_types") or {})
        errors.append({"error": HARNESS_ERROR_KINDS["exception"], "samples_unscored": n,
                       "by_type": by_type})
        parts.append(f"{n} sample(s) not scored because the harness raised"
                     + (f" ({_counts_text(by_type)})" if by_type else ""))
    unscored = dict(agg.get("response_judge_unscored") or {})
    resp_outages = int(agg.get("response_judge_outages") or 0)
    left_out = sum(unscored.values())
    if left_out:
        errors.append({"error": "the Gemini completeness / correctness judge gave no score",
                       "scores_left_out": unscored, "outages": resp_outages,
                       "unparsed_replies": left_out - resp_outages})
        parts.append(f"the Gemini completeness / correctness judge gave no score for "
                     f"{unscored.get('completeness', 0)} / {unscored.get('correctness', 0)} final "
                     f"answer(s) ({resp_outages} failed on every cascade model, "
                     f"{left_out - resp_outages} replied without a 0/1/2 score), left out of "
                     "those means")
    other = judge_outages - outage - resp_outages
    if other > 0:
        errors.append({"error": "Gemini judge calls that failed on every cascade model",
                       "judge_calls": other})
        parts.append(f"{other} Gemini judge call(s) failed on every cascade model")
    return errors, parts


def run_complexfuncbench(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical ComplexFuncBench: the multi-step golden-driven loop + ComplexEval,
    with gbench's Gemini cascade in the two judge roles."""
    ok, reason = check_complexfuncbench_prerequisites()
    if not ok:
        raise infra_required("complexfuncbench", reason, DOCS_URL)

    # Upstream's model request (CFB_SAMPLING): temperature 0.0 without --thinking, 1.0 with it
    # (gbench's rule) unless the operator sets another, and
    # max_tokens by the token-limit policy (2048 without --thinking).
    temperature, temperature_source = resolve_official_sampling(
        "complexfuncbench", CFB_SAMPLING, thinking=enable_thinking)
    plan = resolve_token_budget("complexfuncbench", CFB_SAMPLING, thinking=enable_thinking,
                                max_output_tokens=kwargs.get("max_output_tokens"))
    per_turn_max = plan.max_tokens

    limit = kwargs.get("limit")
    fallbacks_before = judge_fallback_counts()
    agg = asyncio.run(_drive_all(model_name, base_url, max(1, int(concurrency)),
                                 temperature, limit,
                                 max_output_tokens=per_turn_max,
                                 enable_thinking=enable_thinking))
    if agg.get("_empty_shard"):
        return _empty_shard_result(model_name, enable_thinking, agg["_empty_shard"])

    if agg["n_scored"] == 0:
        kinds = agg.get("harness_error_samples") or {}
        raise infra_required(
            "complexfuncbench",
            "no sample could be scored: every sample ended in a harness error ("
            + (_counts_text(kinds) or "none recorded") + "; see HARNESS_ERROR_KINDS). Check that "
            "the endpoint serves a tool-calling model and that the Gemini judge and the "
            f"{RAPIDAPI_TOOL} RapidAPI backend answer.",
            DOCS_URL)

    shard_label = agg.pop("_shard", None)
    is_full = agg.pop("_is_full")
    n_items = agg.pop("_n_items")
    limit_plan = agg.pop("_limit_plan", None)
    embed_backend = agg.pop("_embed_backend")
    embed_dtype = agg.pop("_embed_dtype", None)
    judge_usage = agg.pop("_judge_usage")
    rapidapi_stats = agg.pop("_rapidapi_stats", None) or Counter()
    sample_traces = agg.pop("_sample_traces", None) or []
    judge_outages = int(judge_usage.pop("JUDGE_OUTAGE", 0))
    harness_errors, harness_parts = _harness_error_report(agg, judge_outages)
    n_empty = int(agg.get("n_empty_model_message") or 0)
    turns = [t for tr in sample_traces for t in (tr.get("turns") or [])]
    at_cap = sum(1 for t in turns if t.get("finish_reason") == "length")
    official = CFB_SAMPLING.record(temperature, temperature_source)
    official.update(plan.record())

    response_tier = _response_tier_record(rapidapi_stats)
    leaderboard_comparable = False
    lc_reason = ("the two judge roles run on the Gemini API (gbench's judge cascade) where "
                 "upstream uses gpt-4o-2024-05-13 (call equivalence) and gpt-4o-2024-08-06 "
                 "(response completeness/correctness)")
    if not is_full and limit:
        lc_reason += (f"; --eval-limit {limit}: {n_items} of ComplexFuncBench's {_FULL_TOTAL} "
                      f"samples (upstream's Success Rate is over all {_FULL_TOTAL})")
    if embed_dtype:
        lc_reason += (f"; the bge-large-en-v1.5 embedder that aligns parallel calls ran at "
                      f"{embed_dtype} where upstream's FlagModel uses fp16")
    if official.get("temperature_matches_official") is False:
        lc_reason += (f"; temperature {temperature} set by {temperature_source} (upstream sends "
                      "0.0)")
    if plan.policy != "official":
        lc_reason += (f"; per-turn max_tokens {per_turn_max} where upstream sends "
                      f"{UPSTREAM_MAX_TOKENS} ({plan.note})")
    if harness_parts:
        lc_reason += ("; harness errors, not the model's outcome and left out of the metrics: "
                      + "; ".join(harness_parts))
    if response_tier["pairs_unavailable"]:
        lc_reason += (f"; the {RAPIDAPI_TOOL} RapidAPI backend could not answer "
                      f"{response_tier['pairs_unavailable']} call pair(s) of the response-based "
                      "tier (" + (_counts_text(response_tier["calls_without_response"])
                                  or "no reply") + "), which went to the Gemini judge")
    if response_tier["predict_empty_data"]:
        lc_reason += (f"; {response_tier['predict_empty_data']} call pair(s) whose predicted call "
                      "answered an empty object in the response-based tier, where upstream "
                      "replaces the step's error list with a string (gbench keeps the list)")
    if n_empty:
        lc_reason += (f"; {n_empty} sample(s) ended in a model reply with neither content nor "
                      "tool calls, scored as the model's failures, where upstream's gpt_runner "
                      "files such a reply under unknown_error, as it does an API failure, and "
                      "writes no result for it")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "complexfuncbench",
        "model_name": model_name,
        "thinking": enable_thinking,
        # Harness errors (endpoint failures, a Gemini judge that gave no verdict or score, a
        # harness exception) are counted, never scored as the model's outcome.
        "status": "completed_with_errors" if harness_errors else "success",
        "accuracy": agg["success_rate"],           # headline = Success Rate (%)
        "success_rate": agg["success_rate"],
        # the headline's numerator and denominator: successful and evaluable samples
        "total_questions": agg["n_scored"],
        "correct_answers": agg["n_success"],
        "effective_n": agg["n_scored"],
        "call_accuracy": agg["call_accuracy"],
        "completeness": agg["completeness"],        # raw mean 0..2 (upstream Complete Score)
        "correctness": agg["correctness"],          # raw mean 0..2 (upstream Correct Score)
        "per_domain_success_rate": agg["per_domain_success_rate"],
        # each domain's evaluable samples on a full run (150 each, Cross 400, with no harness
        # error), the denominators of per_domain_success_rate
        "per_domain_evaluated": agg.get("per_domain_evaluated"),
        "aggregation": ("Success Rate = successful samples / evaluable samples (n_scored: the "
                        "samples that did not end in a harness error), on full and subset runs "
                        "alike; on a full run with no harness error that is upstream's /1000 "
                        "(per domain /150, Cross /400). Call Accuracy pools the evaluable "
                        "samples' calls; Completeness / Correctness average the 0/1/2 scores "
                        "the judge gave."),
        "n_samples": n_items,
        "n_scored": agg["n_scored"],
        "n_success": agg["n_success"],
        "n_request_failed": agg["n_request_failed"],
        "n_harness_errors": int(agg.get("n_harness_errors") or 0),
        "harness_error_samples": dict(agg.get("harness_error_samples") or {}),
        "harness_errors": harness_errors,
        "response_judge_unscored": dict(agg.get("response_judge_unscored")
                                        or {"completeness": 0, "correctness": 0}),
        "n_empty_model_message": n_empty,
        "judge": "gbench-gemini-cascade",
        "judge_substitution": JUDGE_SUBSTITUTION,
        "judge_cascade": judge_cascade(),
        "judge_models": dict(judge_usage),       # answered judge calls per Gemini model
        "gemini_models_answered": dict(sorted(judge_usage.items())),
        "judge_outages": judge_outages,          # judge calls every model failed
        "judge_prompts": ("upstream prompts/compare.py and prompts/response.py verbatim (system "
                          "prompt as the Gemini system_instruction, filled user template as the "
                          f"content, temperature 0.0; PROMPT_SHA256) at {UPSTREAM_COMMIT[:12]}"),
        "call_match_tiers": list(CALL_MATCH_TIERS),
        "response_based_tier": response_tier,
        "upstream_revision": f"THUDM/ComplexFuncBench@{UPSTREAM_COMMIT}",
        "dataset": f"{_HF_REPO}@{HF_REVISION}",
        "dataset_sha256": HF_SHA256,
        "embedding_backend": embed_backend,
        # float32 when a step needed the embedder, None when none did (it was never loaded)
        "embedding_dtype": embed_dtype,
        "embedding_upstream": EMBEDDING_UPSTREAM,
        "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": official,
        "max_tokens_sent": [per_turn_max] if per_turn_max is not None else None,
        # each model request's timeout and retries (sized from the turn budget, not 600 s)
        "model_client_timeout": agg.get("_client_timeout"),
        # the Gemini judge client's timeout, and why its cascade left a model during this run
        "judge_client": judge_client_record(),
        "judge_fallbacks": {k: n - fallbacks_before.get(k, 0)
                            for k, n in judge_fallback_counts().items()
                            if n > fallbacks_before.get(k, 0)},
        # model turns that stopped at max_tokens (at the official 2048 on a no-think run, the
        # protocol's outcome; the run's budget on a --thinking run)
        "turns_stopped_at_max_tokens": at_cap,
        "model_turns": len(turns),
        "metric": ("ComplexEval: Success Rate (headline) + Call Accuracy over the multi-step "
                   "golden-driven tool-calling loop, plus response Completeness/Correctness "
                   "(0-2). Calls compared by rule_based, value_checker, response_based (live "
                   "booking-com15 RapidAPI responses, as upstream), then the Gemini "
                   "call-equivalence judge; call-equivalence + response judges on the Gemini API "
                   "(upstream prompts verbatim; upstream's judges are GPT-4o)."),
        "leaderboard_comparable": leaderboard_comparable,
        "leaderboard_comparable_reason": lc_reason,
        "sample_traces": sample_traces,
    }
    if plan.note:
        result["max_output_tokens_note"] = plan.note
    result["eval_limit_plan"] = limit_plan
    record_shard(result, shard_label)
    attempts = int(kwargs.get("attempt_count") or 1)
    if attempts > 1:
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "complexfuncbench: its harness drives each sample once (upstream's "
                           "gpt_runner loop), so --attempt-count is not applied"}
    if agg["n_request_failed"]:
        result["request_failure_note"] = (
            f"{agg['n_request_failed']} sample(s) not scored because the model endpoint request "
            f"failed on all {_REQUEST_ATTEMPTS} attempts: harness errors, left out of Success "
            "Rate, Call Accuracy and the judge means, which are over the "
            f"{agg['n_scored']} evaluable sample(s).")
    return result


def _empty_shard_result(model_name: str, enable_thinking: bool, shard_label: str
                        ) -> Dict[str, Any]:
    """The result of a shard with no sample: nothing was generated or scored on this node, so
    there is no score (accuracy None), never an error and never 0%."""
    reason = (f"empty shard {shard_label}: no items on this node ({_FULL_TOTAL} in the full "
              "set)")
    result: Dict[str, Any] = {
        "benchmark_type": "eval", "eval_name": "complexfuncbench", "model_name": model_name,
        "thinking": enable_thinking, "status": "success", "status_reason": reason,
        "accuracy": None, "success_rate": None, "accuracy_withheld_reason": reason,
        "total_questions": 0, "correct_answers": 0, "n_scored": 0, "empty_shard": True,
        "full_set_items": _FULL_TOTAL, "sample_traces": [],
        "dataset": f"{_HF_REPO}@{HF_REVISION}"}
    record_shard(result, shard_label)
    return add_comparability_reason(result, reason)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one complexfuncbench run (gbench --merge): ComplexEval
    recomputed (_aggregate) over the union of the shards' per-sample records (their
    sample_traces, de-duplicated by sample id), with the per-domain rates when the union is the
    full set; the shard comparability reasons are dropped."""
    union: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for t in r.get("sample_traces") or []:
            union.setdefault(str(t.get("sample_key") or t.get("id")), t)
    records = [{"id": t.get("id"), "message": t.get("message"),
                "count_dict": t.get("count_dict") or {"correct_call_num": 0,
                                                      "total_call_num": 0},
                "resp_eval": t.get("resp_eval"), "harness_error": t.get("harness_error")}
               for t in union.values()]
    is_full = len(records) >= _FULL_TOTAL
    agg = _aggregate(records, len(records), is_full)
    merged = dict(results[0])
    merged.update({
        "accuracy": agg["success_rate"] if agg["n_scored"] else None,
        "success_rate": agg["success_rate"] if agg["n_scored"] else None,
        "total_questions": agg["n_scored"], "correct_answers": agg["n_success"],
        "effective_n": agg["n_scored"], "call_accuracy": agg["call_accuracy"],
        "completeness": agg["completeness"], "correctness": agg["correctness"],
        "per_domain_success_rate": agg["per_domain_success_rate"],
        "per_domain_evaluated": agg.get("per_domain_evaluated"),
        "n_samples": len(records), "n_scored": agg["n_scored"], "n_success": agg["n_success"],
        "n_request_failed": agg["n_request_failed"],
        "n_harness_errors": int(agg.get("n_harness_errors") or 0),
        "harness_error_samples": dict(agg.get("harness_error_samples") or {}),
        "sample_traces": sorted(union.values(), key=lambda t: str(t.get("id")))})
    judges: Counter = Counter()
    for r in results:
        judges.update(r.get("gemini_models_answered") or {})
    merged["gemini_models_answered"] = dict(sorted(judges.items()))
    merged["judge_models"] = dict(judges)
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if reason and not reason.startswith(shard_prefix) \
                    and not reason.startswith("empty shard ") and reason not in kept:
                kept.append(reason)
    merged["leaderboard_comparable"] = False     # the Gemini judge substitution always applies
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    merged["status"] = ("completed_with_errors"
                        if any(r.get("status") == "completed_with_errors" for r in results)
                        else "success")
    for k in ("accuracy_withheld_reason", "status_reason"):
        merged.pop(k, None)
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = "no shard scored a sample"
    return merged
