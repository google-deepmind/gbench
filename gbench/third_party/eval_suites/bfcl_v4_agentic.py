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

"""BFCL v4 **Agentic** track, driven through Berkeley's own `bfcl-eval` harness.

The v4 agentic track is not an answer-matching dataset: it is a stateful tool-execution
benchmark (agents call memory / web-search tools across turns), so it cannot be scored by
loading rows and comparing strings. gbench therefore runs the canonical harness itself:

* categories (verified from `bfcl_eval`): ``memory_kv``, ``memory_vector``,
  ``memory_rec_sum``, ``web_search_base``, ``web_search_no_snippet``;
* the v4 data ships inside the `bfcl-eval` package (BFCL_v4_memory.json /
  BFCL_v4_web_search.json) - nothing is downloaded from HF;
* generation runs against the gbench endpoint (`REMOTE_OPENAI_BASE_URL` + the harness'
  ``--skip-server-setup``), so no extra GPU server is started;
* the headline is BFCL's own Agentic score (`eval_runner_helper.py`): the unweighted mean of
  the web-search summary (mean of the two web_search categories) and the memory summary
  (mean of the three memory categories). The pooled per-question rate is kept as a
  secondary number, `accuracy_pooled`.

Every run covers all five categories unless the operator narrows them with
``--eval-categories`` (recorded as a subset, never leaderboard-comparable).

Harness image
-------------
`bfcl generate` and `bfcl evaluate` run in the local image built from docker/bfcl.Dockerfile
(`gbench-bfcl:2026.3.23`): the PyPI release bfcl-eval 2026.3.23, byte-identical to gorilla at
fc_common.BFCL_EVAL_COMMIT, installed with its own pins on Python 3.10, plus memory_vector's
encoder (sentence-transformers/all-MiniLM-L6-v2 at a pinned revision). Those pins conflict with
the main gbench environment, which therefore never imports bfcl_eval. Each container gets the
host user's uid/gid, this package directory read-only and the harness project root
($GBENCH_CACHE_DIR/bfcl) at the same path, where results and scores are written and read back. No
Docker, no image, or an image built for another release is a hard error (`infra_required`) with
the build command. Inside the image every bfcl_eval file the track reads or runs must match the
pinned commit (`_AGENTIC_PINNED_BLOBS`), else the suite hard-errors.

Networks: `bfcl generate` (and its resume) talks to the model endpoint and, for the web_search
categories, gbench's Gemini relay, and executes the model's calls with eval() in its own process,
so it runs on docker's bridge network, never the host network: the host's DOCKER-USER rule keeps
the cloud metadata server out of reach of a bridge network only. check_bfcl_prerequisites (before
anything is generated) and _run_cli check that the metadata server is blocked there
(container_guard), that the grounding links' host resolves there for the web_search categories
(container_guard resolve=) and that the bridge gateway, where gbench's relays listen, is
reachable from there: `bfcl generate` always reaches the endpoint through gbench's endpoint relay
(2026-10-13, keyless too), never directly; any failing is a hard error (infra_required) naming
the suite that runs:
bfcl and docs/evals/bfcl.md when the bfcl suite runs this track (suite_context).
`bfcl evaluate` and the other harness jobs (the image probe, the --eval-limit id selection, the
entry counts) need no network and run with `--network none`.

Credentials: no harness container holds a key, a token or a password (every one runs, or may run,
the model's calls). The endpoint key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY;
unset, EMPTY, dummy and base's other placeholders are no key) stays in gbench's process:
`bfcl generate` reaches the endpoint through swebench_common.EndpointRelay on the bridge gateway,
which adds it (and runs on a keyless endpoint too, where it adds no key), and every container
holds the placeholder EMPTY. The Gemini key stays there too: the
web_search `bfcl generate` holds gemini_relay.GeminiRelay's per-run placeholder and address
(GBENCH_GEMINI_RELAY_URL), and the relay adds the key to native generateContent calls of the
cascade's models only. OPENAI_DEFAULT_HEADERS is never passed. A launch whose forwarded variables
would still carry a credential of gbench's environment (a password in a proxy URL, for example)
is a hard error (refuse_forwarded_secrets). The relays' records are in `bfcl_report.relays`.
Upstream's `generate` and `evaluate` load <project root>/.env into their process
(load_dotenv(..., override=True)), where upstream's setup has the keys put, and the project root
is mounted into their containers: a project root that holds a .env is a hard error
(infra_required, dotenv_problem) before any container starts, and the harness child replaces
that load_dotenv with one that loads nothing (_disable_dotenv), so no key, and no
OPENAI_BASE_URL around the endpoint relay, can come from that file.

Token budget: BFCL's harness sends no max_tokens. gbench's endpoint relay fills the run's
--max-output-tokens into each `bfcl generate` request that carries none when the suite that runs
is bfcl_v4_agentic (swe_agentic.AGENT_MAX_TOKENS_SUITES, the user's decision of 2026-10-11 for
agent suites; the relay reads the run's `suite` knob), recorded as
official_sampling.agent_max_tokens_from_run and named in leaderboard_comparable_reason
(`agent_max_tokens_fill`). When the bfcl suite runs this track (knob `bfcl`, not an agent suite:
its other parts send no max_tokens either), nothing is filled, as upstream. 2026-10-13: until
then the endpoint relay started only with a real endpoint key, so on the user's keyless vLLM
smoke run smoke-10-12-235114 tapped 83 bfcl_v4_agentic requests with no max_tokens.

Execution guard: BFCL's agentic loop (BaseHandler.inference_multi_turn_FC ->
multi_turn_utils.execute_multi_turn_func_call) runs each call string with eval(). The harness
child of every `bfcl` command installs the bfcl suite's execution guard (bfcl._ExecutionGuard)
in its place before the CLI dispatches (_install_execution_guard): the environment functions
upstream intends (a public method of the entry's backends with literal arguments) run exactly as
upstream; any other call string gets upstream's own "Function call X is not allowed." Refusals
are counted in `bfcl_report.execution_guard_refusals` and make the result not
leaderboard-comparable.

Search backend
--------------
BFCL's web_search tool (`WebSearchAPI.search_engine_query`) queries DuckDuckGo via SerpAPI
upstream. gbench serves it with Gemini Google Search grounding instead, the Gemini API the judged
suites use: in the harness child of the `bfcl generate` that runs a web_search category, the
method is replaced (a plain function on the class, in memory; no bfcl_eval file changes) by one
that sends the query to the search_tool grounding model cascade (search_tool.search_with_fallback:
the models of base.search_cascade(), which is gbench's one Gemini cascade, base.GEMINI_CASCADE,
unless GBENCH_SEARCH_MODELS / GBENCH_SEARCH_MODEL or GBENCH_JUDGE_MODELS / GBENCH_JUDGE_MODEL
name others; gbench hands the list to the container as GBENCH_SEARCH_MODELS and
GBENCH_JUDGE_MODELS, base.gemini_cascade_env; its global rate limit and its cascade retries) and
returns
BFCL's own result shape: a list of {"title", "href", "body"} ({"title", "href"} in
web_search_no_snippet), cut to `max_results` as upstream. The tool the model sees is BFCL's own
(the function docs in multi_turn_func_doc/web_search.json, the method's arguments, the result
shape); `region` is accepted and not applied (grounding takes none). The result records the
substitution (`bfcl_report.search_backend`, `bfcl_report.search_substitution`), the Gemini models
that answered the searches (`gemini_models_answered`, from the search log, and the cascade asked,
`gemini_cascade`), and a run that scored the Web Search setting is not leaderboard-comparable:
"web search served by Gemini Google Search grounding (BFCL queries DuckDuckGo via SerpAPI)".
fetch_url_content stays the harness' own.

BFCL V4's Agentic track has two settings, each its own leaderboard column: Memory (`memory_*`,
which does not search) and Web Search (the two web_search categories). The Web Search setting runs
whenever a Gemini key is set (GEMINI_API_KEYS, else GEMINI_API_KEY, as base.gemini_api_keys reads
it), the Gemini API does not reject it and it serves every model of the cascade
(base.gemini_key_live_valid, the judged suites' gate; a check that gets no answer does not
block). Without a key, or with a rejected one, it is not
generated and is reported `infra_required` in its own entry (`settings.web_search`, listed in
`settings_infra_required`, naming GEMINI_API_KEY): never scored, never dropped, BFCL's Agentic
score (`overall_accuracy_agentic`) left null, the status `completed_with_errors`. The Memory
setting runs and is scored as usual. A run whose selected categories are all web_search has
nothing else to run and hard-errors (`infra_required`, SearchUnavailable) before generating.

During generation the harness child records every search in a log under the project root (the
backend that served it, the number of results, the error when none did; credentials masked). A
search the whole cascade could not serve gets BFCL's own error text ("Failed to retrieve the
search results from server. Please try again later.") and makes the Web Search setting
`infra_required`: the web_search scores would measure the search service, not the model. The log
is read every 30 s while `bfcl generate` runs, which is stopped as soon as the log shows one; one
recorded in its last seconds is caught after generation. When generate is stopped for the web
search, the memory entries it had not finished are generated by the harness' own resume (a second
`bfcl generate` of the memory categories without `--allow-overwrite`, which keeps the entries
already in the result files, `_llm_response_generation.collect_test_cases`), and only the memory
categories are evaluated. Only that web_search generate uses the Gemini relay; it holds the
relay's placeholder, never the key (see Credentials). Every credential value of a container's
environment is masked in the search results and in every tool result the harness hands the model,
in the search log, and every credential of the run in the harness' result and score files and the
captured output.

Bounds
------
As BFCL's own: each entry by BFCL's 20-step limit per turn, and an entry whose request fails is
recorded as an inference error while the run goes on; `bfcl generate` and `bfcl evaluate` have no
wall clock. Upstream's checker scores such an entry wrong; gbench reads it back from the result
file and, as base does for a request that got no reply, leaves it out of the score as a harness
error (never the model's outcome): out of its category's correct and total (the checker scored
it wrong, so only the total changes), counted under `failed_requests` (`timed_out_requests`,
`over_context_prompts`; per category in `bfcl_report.inference_errors`), in `scoring_excluded`
and `request_harness_errors` (`endpoint_failure`, `endpoint_timeout`, `over_context`), named in
`leaderboard_comparable_reason`, the result completed_with_errors. A memory prerequisite entry
(the `*_prereq` conversations that build a scenario's memory, not scored) whose request failed is
recorded the same way upstream and never flushes its memory snapshot, so the scenario's questions
run against an incomplete memory: counted in `memory_prereq_failed_requests` (per category in
`bfcl_report.prereq_inference_errors`; not in `failed_requests`), and every scored entry of that
scenario in that category is left out of the score (`memory_prereq_failed`: off the category's
total, and off its correct when the score file, which lists every entry the checker scored
wrong, does not list it), the result and the Memory setting completed_with_errors, with a named
reason. A category every entry of which is
left out is unscored (not in `category_accuracy`, listed in `categories_unscored`; its summary
and the Agentic score null, `accuracy` the pooled rate over the entries in the score); when no
entry is left in the score, nothing measured the model and the score is withheld (accuracy null
with `accuracy_withheld_reason`, status failed). Endpoint failures gbench's endpoint relay
counted (swebench_common.relay_result_reasons' `upstream_errors` and `requests_refused`) are
harness errors: listed in `endpoint_failures`, the result completed_with_errors and not
leaderboard-comparable. Each request has the OpenAI client's 2
retries and a timeout sized for the longest
reply the served context window allows (agentic_request_timeout, as the bfcl suite's multi-turn
client; BFCL's own handler for a locally served model sends timeout=72000), not the SDK's 600 s.
GBENCH_BFCL_TIMEOUT_S, when set, bounds each of the two on its own. A generate it stops reads the
search log and names failed searches when they are the cause; otherwise the entries the harness
finished (it writes each as it completes) are scored with `bfcl evaluate --partial-eval`, the
entries it did not finish are harness errors left out of the score (`unfinished`, counted as
timed out), the stop is recorded under `harness_errors`, and the result is completed_with_errors
and not leaderboard-comparable. In an
--eval-limit run the harness rewrites each result file whole after every finished entry
(BaseHandler.write with --run-ids), so a stop during a rewrite can also drop entries it had
finished; they count as unfinished too. An evaluate it stops raises HarnessTimeout, carrying the
record of a generate stopped before it (the bfcl suite then keeps its other parts). A generate
stopped by the wall clock whose log shows a failed search makes the Web Search setting
`infra_required` instead; the memory categories are then resumed as above, the resume bounded on
its own.

Sampling: gbench's rule for every suite (base.resolve_temperature): temperature 1.0 with
`--thinking`, 0.0 without, passed to `bfcl generate --temperature`, which the harness' handler sends
on every request (BFCL's own 0.001, the default of that flag, is recorded under
`official_sampling`: fc_common.BFCL_OFFICIAL_SAMPLING). No max_tokens (the harness' requests carry none). Override the temperature for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BFCL_V4_AGENTIC_TEMPERATURE`, which takes
precedence over both; a value other than BFCL's 0.001 makes the result not
leaderboard-comparable. When the `bfcl` suite runs this track as its agentic part, it hands
over its own temperature, which is used as is. No LLM judge.
"""

from __future__ import annotations

import codecs
import contextlib
import contextvars
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .base import (ENDPOINT_FAILURE_KINDS, MODELS_NOT_SERVED_PREFIX, UNCAPPED_TIMEOUT_TOKENS,
                   add_comparability_reason, current_shard, record_shard,
                   gbench_cache_dir, gemini_api_keys, gemini_cascade_env, gemini_key_live_valid,
                   get_run_knob, record_gemini_answer, redact_secrets, redact_text,
                   request_timeout_s, resolve_official_sampling, resolve_temperature,
                   search_cascade, search_cascade_source, secret_forms, secret_values, suite_env,
                   temperature_env_var)
from .fc_common import BFCL_EVAL_COMMIT, BFCL_OFFICIAL_SAMPLING, eval_limit_plan
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/bfcl_v4_agentic.md"
PILLAR = "Tool Use & Agentic Workflows"

#: Categories that issue live web searches (the rest are memory-only, no search backend).
WEB_SEARCH_CATEGORIES = ("web_search_base", "web_search_no_snippet")
MEMORY_CATEGORIES = ("memory_kv", "memory_vector", "memory_rec_sum")
#: The agentic track, in the harness' order. This is the one category set the suite runs and
#: the CLI validates `--eval-categories` against (evals.eval_category_names imports the two
#: tuples above); check_bfcl_prerequisites refuses a bfcl-eval whose mapping differs.
AGENTIC_CATEGORIES = MEMORY_CATEGORIES + WEB_SEARCH_CATEGORIES

#: The bfcl-eval release gbench runs (bfcl's single-turn checker and multi-turn categories, and
#: this track), in the local harness image: its wheel is byte-identical to bfcl_eval at
#: fc_common.BFCL_EVAL_COMMIT, all 183 files.
BFCL_EVAL_RELEASE = "2026.3.23"

#: Scored questions per web_search category: BFCL_v4_web_search.json at BFCL_EVAL_COMMIT (git blob
#: in _AGENTIC_PINNED_BLOBS) holds 100 entries, and both categories ask all of them.
WEB_SEARCH_QUESTIONS = 100

#: git blob ids at BFCL_EVAL_COMMIT of every bfcl_eval file the agentic track reads or runs (the
#: CLI, the generation driver, the FC handler and its loop, the memory and web-search backends,
#: the agentic checker and aggregation, and the memory / web_search data), relative to the
#: installed package. constants/model_config.py and supported_models.py are left out: they list
#: other models, and gbench registers its own entry at run time.
_AGENTIC_PINNED_BLOBS: Dict[str, str] = {
    "__main__.py": "42f6aaf3fc0844fbc1423849516dfe6e1d16a360",
    "_llm_response_generation.py": "8997958de58c896d6cb3c6e03a513fe6b5094b7d",
    "utils.py": "552a10f36ec10840773f75e50fed01ca19b4bfc6",
    "constants/category_mapping.py": "492142b986438fe8a5a6c38931e567bcd45c0d9b",
    "constants/column_headers.py": "fbd6e992b132108f2529a5daaee7e48be47ebcd4",
    "constants/default_prompts.py": "1f5e9e1e8119b2277bee2198214ee9d6aaf6f6e2",
    "constants/enums.py": "7400315f045419d4a8a28525f3999c1844f5e9af",
    "constants/eval_config.py": "813d6de70333b6137f5fb62e116956b13bb339d3",
    "constants/executable_backend_config.py": "03a8a43fd6b75dd0961504edc6704a728b74db5b",
    "constants/type_mappings.py": "fdbdde48402f8f2812b146f9b7341b6aae9e55b8",
    "model_handler/base_handler.py": "a1025e9cbf5d86bcdc62cd0564006284eaffb00a",
    "model_handler/utils.py": "7abd9a242f27c72399dbdbb0f2749b0464e1a0b8",
    "model_handler/api_inference/openai_completion.py": "357584f8dbfb1161c4f5b0aa4a3b024dbe38120b",
    "eval_checker/eval_runner.py": "2d5c41aac41cd2f1caedf741fb01dfdfd57b1990",
    "eval_checker/eval_runner_helper.py": "9124195a3fe283e17761232c846a4af34d5d9af7",
    "eval_checker/agentic_eval/agentic_checker.py": "ad81695bcdbf5607e7303ad2f2c37337c5614e2e",
    "eval_checker/multi_turn_eval/multi_turn_utils.py": "dfcf6512a2eeb592820f3a0fa0fb564a117facc4",
    "eval_checker/multi_turn_eval/func_source_code/memory_api_metaclass.py":
        "d4436eb564b1545dc327aa53a86a6d4c4015ef2f",
    "eval_checker/multi_turn_eval/func_source_code/memory_kv.py":
        "a2d3fc7ee0649957a8e755c7be7ebc3c62a6af0a",
    "eval_checker/multi_turn_eval/func_source_code/memory_rec_sum.py":
        "8022198997c83d88a7d3b6674c5a30d026726bde",
    "eval_checker/multi_turn_eval/func_source_code/memory_vector.py":
        "5f7c72a798be2daac08141db2df1114d945db54d",
    "eval_checker/multi_turn_eval/func_source_code/web_search.py":
        "496451e3246adc702ed5e2c9e6c16dbb00ccadc2",
    "data/BFCL_v4_memory.json": "e0496f35cc59a35f6932d3f3656ac7cfe3d8d1ae",
    "data/BFCL_v4_web_search.json": "e4fd5c6543a8ba69823b6df9a9a3332daafe3c0a",
    "data/possible_answer/BFCL_v4_memory.json": "7ca66247646cb238a199b06643b81c0f7d80da41",
    "data/possible_answer/BFCL_v4_web_search.json": "3af1a136e2e2d08f1cddeb546ee9e58364ade8dd",
    "data/memory_prereq_conversation/memory_customer.json": "79d23886c15837c797a545f584a9eef0dcf3e475",
    "data/memory_prereq_conversation/memory_finance.json": "20c304099f16293c1be447df73fb73bd3ac9c0e0",
    "data/memory_prereq_conversation/memory_healthcare.json": "0c93f30012f1d8d99074d823f7c90e59c27431fe",
    "data/memory_prereq_conversation/memory_notetaker.json": "346ab7a23a0b35171254d6645c13fb5a58a8c4f5",
    "data/memory_prereq_conversation/memory_student.json": "e987d7ce495bfc4725d4c74afd669ebf0eb1cc26",
    "data/multi_turn_func_doc/memory_kv.json": "236e15117c42c3eb2bce632f91ec13b7c6332594",
    "data/multi_turn_func_doc/memory_rec_sum.json": "dd598a15233a0b4fe59949f061b924acaf59fa6b",
    "data/multi_turn_func_doc/memory_vector.json": "26c550d256755ee748fc603f0c0e62be700546d1",
    "data/multi_turn_func_doc/web_search.json": "ecd4768683d8d6f102cbbadfcb41969b9bd322e3",
}

#: BFCL V4's two Agentic settings, each its own leaderboard column (Web Search, Memory), in the
#: order gbench generates them. Only Web Search searches; memory_* does not.
AGENTIC_SETTINGS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("memory", MEMORY_CATEGORIES), ("web_search", WEB_SEARCH_CATEGORIES))

#: What serves BFCL's web_search tool (WebSearchAPI.search_engine_query) in gbench.
SEARCH_BACKEND = "gemini-google-search-grounding"
#: The Web Search setting's prerequisite, named in its infra_required entry: a Gemini key
#: (GEMINI_API_KEYS, else GEMINI_API_KEY, as base.gemini_api_keys reads it) that the Gemini API
#: does not reject.
SEARCH_KEY_NAME = "GEMINI_API_KEY"
SEARCH_PREREQUISITE = f"Gemini Google Search grounding with {SEARCH_KEY_NAME}"
#: The variables that carry the Gemini key(s). Only the web_search `bfcl generate` receives them.
GEMINI_KEY_NAMES = ("GEMINI_API_KEY", "GEMINI_API_KEYS")
#: search_tool's grounding-cascade knobs, passed to the web_search `bfcl generate` when set.
SEARCH_KNOB_NAMES = ("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", "GBENCH_SEARCH_MAX_RPS",
                     "GBENCH_SEARCH_MAX_BURST", "GBENCH_SEARCH_CASCADE_ROUNDS",
                     "GBENCH_SEARCH_ATTEMPTS", "GBENCH_SEARCH_BACKOFF")
#: The variables that hand gbench's Gemini cascade to the web_search `bfcl generate`
#: (base.gemini_cascade_env: GBENCH_JUDGE_MODELS and GBENCH_SEARCH_MODELS, set by gbench from
#: base.judge_cascade() / base.search_cascade()), so search_tool there asks exactly the models
#: the Gemini relay forwards and the result records.
CASCADE_ENV_NAMES = ("GBENCH_JUDGE_MODELS",)
#: Why a run that scored the Web Search setting is not leaderboard-comparable.
SEARCH_SUBSTITUTION_REASON = ("web search served by Gemini Google Search grounding (BFCL queries "
                              "DuckDuckGo via SerpAPI)")
#: The web search log the harness child writes during generation (under the project root), and
#: the variable that names it in the web_search `bfcl generate` container.
SEARCH_LOG_NAME = "web_search_requests.jsonl"
SEARCH_LOG_ENV = "GBENCH_BFCL_SEARCH_LOG"
#: The error text BFCL's own web_search.py hands the model when its search service's reply holds
#: no result list; a search the grounding cascade could not serve gets the same text.
SEARCH_FAILED_TEXT = "Failed to retrieve the search results from server. Please try again later."
#: What to do when searches failed during a run.
SEARCH_ADVICE = ("Gemini Google Search grounding must answer every search of the run (a Gemini key "
                 "with grounding quota for it; GBENCH_SEARCH_MAX_RPS lowers the request rate); "
                 "re-run once it does.")


class SearchUnavailable(RuntimeError):
    """BFCL's Web Search setting lost its prerequisite, Gemini Google Search grounding with a
    Gemini key the Gemini API accepts: the key is unset or rejected, or a search the grounding
    cascade could not serve reached the model during generation. The setting is reported
    infra_required (web_search_infra_required) while the other settings run; raised as a hard
    error only when no other setting of the run is left. `reason` never holds the key."""

    def __init__(self, reason: str, when: str = "during generation"):
        self.reason, self.when = str(reason), when
        sentence = self.reason.rstrip()
        sentence += "" if sentence.endswith(".") else "."
        ctx = _suite()
        super().__init__(f"[INFRA REQUIRED] {ctx.eval_name}: BFCL's Web Search setting cannot be "
                         f"scored: {sentence} See '{ctx.docs_url}'.")


def web_search_infra_required(categories: Sequence[str], reason: str, when: str,
                              eval_name: Optional[str] = None) -> Dict[str, Any]:
    """The result entry of BFCL's Web Search setting when its prerequisite is missing while the
    other settings run: not run to the end, not scored, the missing prerequisite named. `when` is
    "before generation" or "during generation". The error names `eval_name` (default: the suite
    that runs, suite_context) and its doc."""
    ctx = _suite()
    eval_name = eval_name or ctx.eval_name
    docs_url = ctx.docs_url if eval_name == ctx.eval_name else f"docs/evals/{eval_name}.md"
    sentence = str(reason).rstrip()
    sentence += "" if sentence.endswith(".") else "."
    return {"status": "infra_required", "setting": "web_search",
            "categories": list(categories), "prerequisite": SEARCH_KEY_NAME, "when": when,
            "infra_required": str(reason),
            "error": (f"[INFRA REQUIRED] {eval_name}: BFCL's Web Search setting "
                      f"({', '.join(categories)}) was not scored ({when}): {sentence} "
                      f"See '{docs_url}'."),
            "leaderboard_comparable": False,
            "leaderboard_comparable_reason": (f"BFCL's Web Search setting was not scored (its "
                                              f"prerequisite, {SEARCH_PREREQUISITE}, was not "
                                              f"available {when}; infra_required)"),
            "total_questions": 0, "correct_answers": 0, "accuracy": None}


def default_project_root() -> str:
    """The harness project root when GBENCH_BFCL_PROJECT_ROOT is unset: $GBENCH_CACHE_DIR/bfcl
    (default ~/.cache/gbench/bfcl)."""
    return os.path.join(gbench_cache_dir(), "bfcl")


def harness_project_root() -> Path:
    """The harness project root: GBENCH_BFCL_PROJECT_ROOT, else upstream's own BFCL_PROJECT_ROOT,
    else default_project_root(); mounted into the `bfcl` command containers at the same path."""
    return Path(suite_env("GBENCH_BFCL_PROJECT_ROOT", "BFCL_PROJECT_ROOT")
                or default_project_root())


#: The file upstream's `bfcl generate` and `bfcl evaluate` load into their process
#: (bfcl_eval/__main__.py: load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)), where
#: upstream's setup has the operator put the API keys.
DOTENV_NAME = ".env"


def dotenv_problem(root: Optional[Path] = None) -> Optional[str]:
    """Why the harness cannot run with the project root `root` (default:
    harness_project_root()), or None. The root is mounted into the containers that run the
    model's calls, and upstream's commands load <root>/.env into their process with
    override=True, so a .env there would put whatever keys it holds into such a container and
    could replace the endpoint URL gbench passes (its relay's). gbench passes everything the
    harness needs itself. The file is never read, only named."""
    path = (harness_project_root() if root is None else Path(root)) / DOTENV_NAME
    if not os.path.lexists(path):
        return None
    return (f"{path} exists. The harness project root is mounted into the BFCL harness "
            f"containers, which run the model's calls, and bfcl-eval loads {DOTENV_NAME} from it "
            "into `bfcl generate` and `bfcl evaluate` (load_dotenv with override=True), so the "
            "keys it holds would reach the model's process and its OPENAI_BASE_URL would replace "
            "the endpoint gbench passes. gbench passes everything the harness needs itself: move "
            f"that file out of {path.parent}, or point GBENCH_BFCL_PROJECT_ROOT at a directory "
            "used only by gbench, and re-run.")


def _skip_dotenv(*_args: Any, **_kwargs: Any) -> bool:
    """Stands in for python-dotenv's load_dotenv in the harness child: loads nothing."""
    return False


def _disable_dotenv() -> None:
    """Inside the harness child of a `bfcl` command, after bfcl_eval.__main__ is imported and
    before the CLI dispatches: upstream's `generate` and `evaluate` call load_dotenv(<project
    root>/.env, override=True) first, which here loads nothing (the module's load_dotenv is
    replaced in memory; no bfcl_eval file changes). gbench refuses to start with such a file
    (dotenv_problem); this keeps one that appears during a run out of the process too."""
    main = sys.modules.get("bfcl_eval.__main__")
    if main is not None:
        main.load_dotenv = _skip_dotenv


class _OnePointStatistics:
    """eval_runner_helper's `statistics` in the harness child: the module, except that stdev of
    fewer than two points is NaN (undefined) instead of StatisticsError."""

    def __getattr__(self, name: str) -> Any:
        import statistics
        return getattr(statistics, name)

    @staticmethod
    def stdev(data: Any, xbar: Any = None) -> float:
        import statistics
        data = list(data)
        return float("nan") if len(data) < 2 else statistics.stdev(data, xbar)


def _allow_one_latency() -> None:
    """Inside the harness child, before the CLI dispatches: upstream's evaluate takes
    statistics.stdev of the model's request latencies for its leaderboard CSV, which raises with
    one latency (an --eval-limit run with one entry: 2026-10-10 smoke test, the whole suite
    failed). That standard deviation becomes NaN there; gbench reads no latency column, and no
    score changes (the module reference is replaced in memory; no bfcl_eval file changes)."""
    import bfcl_eval.eval_checker.eval_runner_helper as helper
    helper.statistics = _OnePointStatistics()


# --------------------------------------------------------------------------- #
# the local harness image (docker/bfcl.Dockerfile)
# --------------------------------------------------------------------------- #
#: The image the BFCL suites run bfcl_eval in; GBENCH_BFCL_IMAGE names another tag. Its labels
#: must equal BFCL_IMAGE_LABELS.
BFCL_IMAGE_DEFAULT = f"gbench-bfcl:{BFCL_EVAL_RELEASE}"
BFCL_DOCKERFILE = "gbench/docker/bfcl.Dockerfile"
BFCL_IMAGE_CONTEXT = "gbench/docker"
#: memory_vector's encoder in the image (SentenceTransformer("all-MiniLM-L6-v2") upstream), at the
#: Hub revision that was main when it was pinned.
MEMORY_VECTOR_ENCODER = "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BFCL_IMAGE_LABELS = {"gbench.bfcl.release": BFCL_EVAL_RELEASE,
                     "gbench.bfcl.commit": BFCL_EVAL_COMMIT,
                     "gbench.bfcl.minilm": MEMORY_VECTOR_ENCODER}
#: Where the image sees this package directory (read-only) and the per-run exchange directory.
_IMAGE_SUITES_DIR = "/opt/gbench/eval_suites"
_IMAGE_TP_SUITES_DIR = "/opt/gbench/third_party_eval_suites"
_IMAGE_IO_DIR = "/gbench_io"
#: The harness project root of image jobs that write no results (bfcl_eval creates its result,
#: score and lock directories there when eval_config is imported).
_IMAGE_SCRATCH_ROOT = "/tmp/bfcl_project"
#: Proxy settings a container inherits from the host when they are set.
_PROXY_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


def bfcl_image() -> str:
    """The local image the BFCL suites run the harness in (GBENCH_BFCL_IMAGE overrides it)."""
    return suite_env("GBENCH_BFCL_IMAGE", default=BFCL_IMAGE_DEFAULT) or BFCL_IMAGE_DEFAULT


def image_build_command(image: Optional[str] = None) -> str:
    """The command that builds the harness image (run from the gbench repository root)."""
    return f"docker build -t {image or bfcl_image()} -f {BFCL_DOCKERFILE} {BFCL_IMAGE_CONTEXT}"


def image_problem() -> Optional[str]:
    """None when Docker is reachable and the harness image is built from docker/bfcl.Dockerfile
    at BFCL_EVAL_RELEASE (its labels); else why not, with the build command."""
    image = bfcl_image()
    hint = (f"Build it LOCALLY from the gbench repository root (gbench never pulls it): "
            f"`{image_build_command(image)}`.")
    if not shutil.which("docker"):
        return ("the BFCL harness (bfcl-eval " + BFCL_EVAL_RELEASE + ") runs in a local Docker "
                "image, and the docker CLI is not on PATH. " + hint)
    try:
        info = subprocess.run(["docker", "info"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"the Docker daemon is not reachable ({type(e).__name__}: {e}). {hint}"
    if info.returncode != 0:
        return f"the Docker daemon is not reachable (`docker info` failed). {hint}"
    try:
        insp = subprocess.run(["docker", "image", "inspect", "--format",
                               "{{json .Config.Labels}}", image],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"cannot inspect image {image!r} ({type(e).__name__}: {e}). {hint}"
    if insp.returncode != 0:
        return f"the BFCL harness image {image!r} is not built. {hint}"
    try:
        labels = json.loads(insp.stdout.strip() or "null") or {}
    except ValueError:
        labels = {}
    wrong = [f"{k}={labels.get(k) or 'unset'} (wanted {v})"
             for k, v in BFCL_IMAGE_LABELS.items() if labels.get(k) != v]
    if wrong:
        return (f"image {image!r} was not built from {BFCL_DOCKERFILE} for bfcl-eval "
                f"{BFCL_EVAL_RELEASE}: {'; '.join(wrong)}. Rebuild it: `{image_build_command(image)}`.")
    return None


def _package_preamble(suites_dir: str = _IMAGE_SUITES_DIR,
                      tp_suites_dir: str = _IMAGE_TP_SUITES_DIR) -> str:
    """Python source that makes `gbench.runners.eval_suites` importable from the mounted package
    directory without running any package __init__: the image has bfcl-eval, not gbench's
    dependencies, and the BFCL modules only import the stdlib-only helpers next to them."""
    return ("import sys as _s, types as _t\n"
            "_parent = None\n"
            "for _n, _p in (('gbench', []), ('gbench.runners', []), "
            f"('gbench.runners.eval_suites', [{suites_dir!r}, {tp_suites_dir!r}])):\n"
            "    _m = _t.ModuleType(_n)\n"
            "    _m.__path__ = _p\n"
            "    _s.modules[_n] = _m\n"
            "    if _parent is not None:\n"
            "        setattr(_parent, _n.rsplit('.', 1)[1], _m)\n"
            "    _parent = _m\n")


#: The network of a harness job that needs none: every job but the ones that talk to the model
#: endpoint (bfcl.py's multi-turn part and `bfcl generate`, NETWORK_JOBS).
NO_NETWORK = "none"


def _network_args(network: str) -> List[str]:
    """The `docker run` network arguments of a harness job: `--network none`, or the bridge
    network with this host reachable at host.docker.internal (tau_common.ENDPOINT_NETWORK_ARGS)
    for a job that talks to the model endpoint. The host network is never used: the model's calls
    run in the harness process (BFCL executes them with eval()), and the host's DOCKER-USER rule
    keeps the cloud metadata server out of reach of a bridge network only."""
    from .tau_common import ENDPOINT_NETWORK, ENDPOINT_NETWORK_ARGS
    if network == NO_NETWORK:
        return ["--network", NO_NETWORK]
    if network == ENDPOINT_NETWORK:
        return list(ENDPOINT_NETWORK_ARGS)
    raise ValueError(f"a BFCL harness job runs with network {NO_NETWORK!r} or "
                     f"{ENDPOINT_NETWORK!r}, not {network!r}")


def _docker_run_command(name: str, env_names: Sequence[str],
                        mounts: Sequence[Tuple[str, str, str]], program: str,
                        workdir: Optional[str] = None, network: str = NO_NETWORK) -> List[str]:
    """`docker run` for one harness job. Values of the forwarded variables come from the launching
    process' environment (`-e NAME`), so no credential is on the command line. `network` is
    NO_NETWORK (the default) or, for a job that talks to the model endpoint, the bridge network
    (_network_args)."""
    tp_suites_dir = os.path.dirname(os.path.abspath(__file__))
    suites_dir = os.path.normpath(os.path.join(tp_suites_dir, "..", "..", "runners", "eval_suites"))
    cmd = ["docker", "run", "--rm", "--pull", "never", "--name", name, *_network_args(network),
           "--user", f"{os.getuid()}:{os.getgid()}",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "-e", "HOME=/tmp", "-e", "TQDM_DISABLE", "-e", "TQDM_MININTERVAL"]
    for env_name in sorted(set(env_names)):
        cmd += ["-e", env_name]
    cmd += ["-v", f"{suites_dir}:{_IMAGE_SUITES_DIR}:ro",
            "-v", f"{tp_suites_dir}:{_IMAGE_TP_SUITES_DIR}:ro"]
    for host, inner, mode in mounts:
        cmd += ["-v", f"{host}:{inner}:{mode}"]
    if workdir:
        cmd += ["-w", workdir]
    cmd += [bfcl_image(), "python", "-I", "-B", "-u", "-c", program]
    return cmd


# --------------------------------------------------------------------------- #
# credentials: no key in a container that runs the model's calls
# --------------------------------------------------------------------------- #
#: What a harness container holds as OPENAI_API_KEY / REMOTE_OPENAI_API_KEY when the endpoint
#: key is a real key (endpoint_key): a placeholder; gbench's endpoint relay adds the key.
ENDPOINT_KEY_PLACEHOLDER = "EMPTY"
#: The variables that carry the endpoint key to the harness' OpenAI client.
ENDPOINT_KEY_NAMES = ("OPENAI_API_KEY", "REMOTE_OPENAI_API_KEY")
#: The variable that tells search_tool in the container where the Gemini relay listens
#: (gemini_relay.RELAY_URL_ENV).
GEMINI_RELAY_URL_ENV = "GBENCH_GEMINI_RELAY_URL"
#: The file the harness child of a `bfcl` command records execution-guard refusals in (under the
#: project root), and the variable that names it in the container.
GUARD_LOG_NAME = "execution_guard_refusals.jsonl"
GUARD_LOG_ENV = "GBENCH_BFCL_GUARD_LOG"
#: The host of the links Gemini Google Search grounding returns (Google's grounding redirect
#: URLs), which BFCL's own fetch_url_content follows from the `bfcl generate` container.
GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_PROXY_URL_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_NO_PROXY_NAMES = ("NO_PROXY", "no_proxy")


def endpoint_key(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The model endpoint's key in `environ` (default: this process'), base.endpoint_api_key's
    rule: VLLM_API_KEY, else OPENAI_API_KEY; None when neither holds one (unset or a placeholder:
    EMPTY, dummy, not-needed, ...). It never enters a harness container: gbench's endpoint relay
    adds it (bridge_services)."""
    from .base import endpoint_api_key
    return endpoint_api_key(environ)


class _SuiteContext:
    """The suite a harness check runs for (its name and doc label every infra_required it
    raises) and what that run needs from the bridge network."""

    def __init__(self, eval_name: str, docs_url: str, web_search: bool = False,
                 relays: Optional[List[Dict[str, Any]]] = None):
        self.eval_name, self.docs_url, self.web_search = eval_name, docs_url, web_search
        self.relays = relays


_SUITE: "contextvars.ContextVar[Optional[_SuiteContext]]" = contextvars.ContextVar(
    "bfcl_suite", default=None)


def _suite() -> _SuiteContext:
    return _SUITE.get() or _SuiteContext("bfcl_v4_agentic", DOCS_URL)


@contextlib.contextmanager
def suite_context(eval_name: str, docs_url: str, *, web_search: Optional[bool] = None,
                  relays: Optional[List[Dict[str, Any]]] = None):
    """Inside this block the harness checks and containers of this module run for `eval_name`:
    check_bfcl_prerequisites and `bfcl generate`'s network checks name it and `docs_url` in the
    infra_required they raise (the bfcl suite runs this track as its agentic part). With
    `web_search`, check_bfcl_prerequisites also checks what the web_search `bfcl generate` needs
    from the bridge network; with `relays`, `bfcl generate` appends the records of the relays it
    ran. Unset values are inherited from an enclosing block."""
    outer = _SUITE.get()
    ctx = _SuiteContext(eval_name, docs_url,
                        web_search if web_search is not None else bool(outer and outer.web_search),
                        relays if relays is not None else (outer.relays if outer else None))
    token = _SUITE.set(ctx)
    try:
        yield ctx
    finally:
        _SUITE.reset(token)


def _container_hostnames(web_search: bool, environ: Optional[Dict[str, str]] = None) -> List[str]:
    """The host names a harness container on the bridge network resolves itself that gbench
    checks here: for the web_search categories, the host of the grounding links BFCL's own
    fetch_url_content follows (GROUNDING_REDIRECT_HOST), or the proxy's host when the container
    fetches through a proxy of `environ` (default: this process'; tau_common.container_lookups).
    The Gemini API is reached by gbench's relay on this host, never from the container; the model
    endpoint through the endpoint relay, or at the URL tau_common.require_endpoint_network checks
    (it resolves a named endpoint host itself)."""
    if not web_search:
        return []
    from .tau_common import container_lookups, container_proxy_env
    return container_lookups(None, [GROUNDING_REDIRECT_HOST],
                             container_proxy_env(dict(os.environ if environ is None else environ)))


def _metadata_gate(eval_name: str, docs_url: str, names: Sequence[str]) -> None:
    """container_guard.require_metadata_blocked for the bridge network, with the host names the
    container resolves (infra_required when the metadata server answers there or a name does not
    resolve)."""
    from .container_guard import require_metadata_blocked
    from .tau_common import ENDPOINT_NETWORK
    require_metadata_blocked(eval_name, bfcl_image(), ENDPOINT_NETWORK, docs_url,
                             resolve=list(names))


def _bridge_gateway(eval_name: str, docs_url: str) -> str:
    """The IPv4 gateway of docker's bridge network: where gbench's relays listen, the address
    the harness containers reach this host at."""
    from .gemini_relay import network_gateway
    from .tau_common import ENDPOINT_NETWORK
    return network_gateway(eval_name, ENDPOINT_NETWORK, docs_url)


_RELAY_PROBE_MARK = "GBENCH_RELAY_PROBE="
_RELAY_PROBE_CONNECT_S = 10
_RELAY_PROBE_DOCKER_S = 120
#: The gateways a container of the image on the bridge network was found to reach (a TCP
#: connection to a listener on this host opened); cached for the rest of the process.
_RELAY_HOSTS_OK: Dict[str, bool] = {}
_RELAY_LOCK = threading.Lock()


def _relay_probe_script(host: str, port: int) -> str:
    """python3 source that opens a TCP connection to host:port, closes it and prints the result.
    Nothing is sent on the connection."""
    return ("import socket\n"
            "try:\n"
            f"    socket.create_connection(({host!r}, {int(port)}), "
            f"{_RELAY_PROBE_CONNECT_S}).close()\n"
            "except OSError as e:\n"
            f"    print({_RELAY_PROBE_MARK!r} + 'closed ' + type(e).__name__)\n"
            "else:\n"
            f"    print({_RELAY_PROBE_MARK!r} + 'open')\n")


def _require_relay_reachable(eval_name: str, host: str, port: int, docs_url: str) -> None:
    """infra_required unless a TCP connection to host:port (a listener of this process on the
    bridge gateway) opens from a container of the harness image on the bridge network. Cached
    per host: a host firewall treats the gateway's ports alike."""
    from .tau_common import ENDPOINT_NETWORK
    with _RELAY_LOCK:
        if _RELAY_HOSTS_OK.get(host):
            return
        name = f"gbench-relay-probe-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "--rm", "--pull", "never", "--name", name,
               *_network_args(ENDPOINT_NETWORK), "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--entrypoint", "python3", bfcl_image(),
               "-c", _relay_probe_script(host, port)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=_RELAY_PROBE_DOCKER_S)
        except (OSError, subprocess.SubprocessError) as e:
            if isinstance(e, subprocess.TimeoutExpired):
                try:
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True,
                                   timeout=30)
                except (OSError, subprocess.SubprocessError):
                    pass
            raise infra_required(
                eval_name, f"gbench could not check that its relay on {host} is reachable from "
                f"docker's bridge network ({type(e).__name__}).", docs_url) from None
        verdict = next((line.strip()[len(_RELAY_PROBE_MARK):].split()
                        for line in (proc.stdout or "").splitlines()
                        if line.strip().startswith(_RELAY_PROBE_MARK)), None)
        if verdict and verdict[0] == "open":
            _RELAY_HOSTS_OK[host] = True
            return
    if verdict and verdict[0] == "closed":
        raise infra_required(
            eval_name, f"the harness container on docker's bridge network cannot open a TCP "
            f"connection to {host}:{port} ({' '.join(verdict[1:])}), where gbench's relay "
            "listens: the relay adds the endpoint and Gemini keys outside the container, which "
            "runs the model's calls and so never holds a key. A host firewall must let the "
            f"docker bridge reach this host's address {host}.", docs_url)
    detail = " ".join(((proc.stderr or proc.stdout) or "no output").split())[-300:]
    raise infra_required(
        eval_name, f"gbench could not check that its relay on {host} is reachable from docker's "
        f"bridge network: the probe (python3 in {bfcl_image()}) gave no verdict (docker exit "
        f"{proc.returncode}: {detail}).", docs_url)


def _require_gateway_reachable(eval_name: str, docs_url: str) -> str:
    """The bridge gateway, after checking that a container of the image on the bridge network
    reaches a listener of this process there (a temporary one, closed at once; cached). Raises
    infra_required otherwise."""
    import socket
    gateway = _bridge_gateway(eval_name, docs_url)
    with _RELAY_LOCK:
        if _RELAY_HOSTS_OK.get(gateway):
            return gateway
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind((gateway, 0))
        listener.listen(8)
    except OSError as e:
        raise infra_required(
            eval_name, f"gbench could not listen on the docker bridge gateway {gateway} "
            f"({type(e).__name__}: {e.strerror or e}), where its relays add the endpoint and "
            "Gemini keys outside the harness container.", docs_url) from None
    try:
        _require_relay_reachable(eval_name, gateway, listener.getsockname()[1], docs_url)
    finally:
        listener.close()
    return gateway


def endpoint_gate(eval_name: str, url: str, docs_url: str) -> str:
    """tau_common.require_endpoint_network for a BFCL harness container that reaches the model
    endpoint directly from the bridge network; returns the URL it uses there. The container gets
    this host's proxy variables (container_proxy_env, as _run_cli and run_in_image pass them), so
    when one is set the check is told so (proxied=True, which reads the same variables): a
    request the container sends through the proxy needs only the proxy's host to resolve there,
    as for tau2 / tau3."""
    from .tau_common import container_proxy_env, require_endpoint_network
    if container_proxy_env(dict(os.environ)):
        return require_endpoint_network(eval_name, bfcl_image(), url, docs_url, proxied=True)
    return require_endpoint_network(eval_name, bfcl_image(), url, docs_url)


def bridge_network_check(eval_name: str, base_url: str, docs_url: str, *,
                         web_search: bool = False, key: Any = "",
                         always_relay: bool = False) -> Optional[str]:
    """The checks before a harness container that runs the model's calls starts on docker's
    bridge network (the multi-turn part, `bfcl generate`); raises infra_required when one fails.

    * The cloud metadata server is unreachable there and, for the web_search categories, the
      grounding links' host resolves there (container_guard.require_metadata_blocked with
      _container_hostnames).
    * When a relay is needed (`always_relay`, which `bfcl generate` passes: its endpoint relay
      runs for every run; the endpoint key is a real key; or the web_search categories run), a
      container of the image reaches this host at the bridge gateway, where the relays listen.
    * When the container reaches the endpoint directly (bfcl.py's multi-turn part without a
      real key): tau_common.require_endpoint_network, which for an endpoint on this host also
      checks a TCP connection to host.docker.internal.
    Returns the endpoint URL the container uses when it reaches the endpoint directly, None when
    it goes through the endpoint relay. `key` is the endpoint key (default: endpoint_key())."""
    key = endpoint_key() if key == "" else key
    relayed = bool(key or always_relay)
    names = _container_hostnames(web_search)
    if names or relayed or web_search:
        _metadata_gate(eval_name, docs_url, names)
    if relayed or web_search:
        _require_gateway_reachable(eval_name, docs_url)
    if relayed:
        return None
    return endpoint_gate(eval_name, base_url, docs_url)


class BridgeServices:
    """What a harness container on the bridge network uses in place of credentials
    (bridge_services): the endpoint URL (the endpoint relay's when one runs) and the relays that
    add the keys (and, for an agent suite, the run's --max-output-tokens) outside the container."""

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.gateway: Optional[str] = None
        self.endpoint_relay: Any = None
        self.gemini_relay: Any = None

    def container_env(self, environ: Dict[str, str]) -> Dict[str, str]:
        """The variables to set in the container, given the launching environment: the endpoint
        key's placeholder when the endpoint relay adds the key; the Gemini relay's placeholder
        and address, and gbench's Gemini cascade (base.gemini_cascade_env, the models the relay
        forwards), when it serves the searches; the gateway added to NO_PROXY / no_proxy when a
        proxy is set, so a request to a relay goes to it directly."""
        out: Dict[str, str] = {}
        if self.endpoint_relay is not None:
            out.update({name: ENDPOINT_KEY_PLACEHOLDER for name in ENDPOINT_KEY_NAMES})
        if self.gemini_relay is not None:
            client = self.gemini_relay.client_env()
            out.update({name: client[name] for name in (*GEMINI_KEY_NAMES, GEMINI_RELAY_URL_ENV)})
            out.update(gemini_cascade_env())
        if self.gateway and any((environ.get(n) or "").strip() for n in _PROXY_URL_NAMES):
            for name in _NO_PROXY_NAMES:
                hosts = [h for h in (environ.get(name) or "").split(",") if h.strip()]
                if self.gateway not in (h.strip() for h in hosts):
                    hosts.append(self.gateway)
                out[name] = ",".join(hosts)
        return out

    def record(self) -> Dict[str, Any]:
        """The relays' records for the result (never a key)."""
        return {"endpoint_relay": self.endpoint_relay.record() if self.endpoint_relay else None,
                "gemini_relay": self.gemini_relay.record() if self.gemini_relay else None}


def generate_always_relayed() -> bool:
    """Whether every `bfcl generate` reaches the endpoint through gbench's endpoint relay, with
    or without an endpoint key: True since 2026-10-13 (the module docstring's Token budget; until
    then the relay started only with a real key, so the run's --max-output-tokens, the relay's
    request-shape check and its records never applied on the user's keyless vLLM). The unit tests
    that fake docker keep the keyless shortcut (no relay without a real key) through an autouse
    fixture in tests/conftest.py; the real function stays reachable there as `.original`
    (tests/test_relay_always_tau_bfcl.py)."""
    return True


def _start_endpoint_relay(eval_name: str, docs_url: str, base_url: str, gateway: str,
                          key: Optional[str], request_timeout: Optional[float]) -> Any:
    """swebench_common.EndpointRelay on the bridge gateway for the container's model requests:
    it adds `key` when there is one (None on a keyless endpoint) and, when the suite that runs is
    an agent suite (swe_agentic.AGENT_MAX_TOKENS_SUITES, the run's `suite` knob), the run's
    --max-output-tokens on each completions request that carries none; its per-request limit is
    above the client's own timeout."""
    from .swebench_common import EndpointRelay
    try:
        relay = EndpointRelay(base_url, gateway, key=key or None,
                              name=f"{eval_name}-endpoint-relay")
    except ValueError as e:
        raise infra_required(eval_name, f"the model endpoint URL cannot be relayed ({e}); gbench "
                             "relays the harness' model requests (its relay adds the endpoint "
                             "key, which never enters the harness container, and the run's "
                             "--max-output-tokens for an agent suite).", docs_url) from None
    if request_timeout:
        relay.TIMEOUT_S = max(EndpointRelay.TIMEOUT_S, int(float(request_timeout)) + 60)
    try:
        relay.start()
    except OSError as e:
        raise infra_required(eval_name, f"gbench's endpoint relay could not listen on the docker "
                             f"bridge gateway {gateway} ({type(e).__name__}: {e.strerror or e}).",
                             docs_url) from None
    try:
        _require_relay_reachable(eval_name, gateway, int(relay.url.rsplit(":", 1)[1].split("/")[0]),
                                 docs_url)
    except BaseException:
        relay.stop()
        raise
    return relay


def _start_gemini_relay(eval_name: str, docs_url: str, gateway: str) -> Any:
    """gemini_relay.GeminiRelay on the bridge gateway for the web searches of `bfcl generate`:
    only native generateContent for the grounding models of the cascade; the container holds
    its per-run placeholder."""
    from .gemini_relay import GeminiRelay
    relay = GeminiRelay(eval_name, gateway, docs_url=docs_url, openai_calls=(),
                        native_methods=("generateContent",), native_models=search_models(),
                        name=f"{eval_name}-gemini-relay").start()
    try:
        _require_relay_reachable(eval_name, gateway, int(relay.origin.rsplit(":", 1)[1]),
                                 docs_url)
    except BaseException:
        relay.stop()
        raise
    return relay


@contextlib.contextmanager
def bridge_services(eval_name: str, docs_url: str, base_url: str, *, key: Optional[str] = None,
                    gemini: bool = False, request_timeout: Optional[float] = None,
                    environ: Optional[Dict[str, str]] = None, always_relay: bool = False):
    """The relays a harness container on the bridge network uses instead of keys, for the
    duration of the block (yields a BridgeServices, stopped at the end).

    With a real endpoint key (`key`), or with `always_relay` (which `bfcl generate` passes, keyless
    too, since 2026-10-13), a swebench_common.EndpointRelay on the bridge gateway relays the
    container's model requests (_start_endpoint_relay: it adds the key when there is one): the
    container gets the relay's URL and the placeholder ENDPOINT_KEY_PLACEHOLDER. With
    `gemini`, a gemini_relay.GeminiRelay there serves the grounding searches: the container gets
    its placeholder and address. Before they start, the metadata guard (with the host names the
    container resolves, given the proxy variables of `environ`, the container's launching
    environment) and the gateway's reachability are checked. Without either, nothing starts and
    the container reaches the endpoint directly."""
    names = _container_hostnames(gemini, environ)
    svc = BridgeServices(base_url)
    relayed = bool(key or always_relay)
    if names or relayed or gemini:
        _metadata_gate(eval_name, docs_url, names)
    try:
        if relayed or gemini:
            svc.gateway = _require_gateway_reachable(eval_name, docs_url)
        if relayed:
            svc.endpoint_relay = _start_endpoint_relay(eval_name, docs_url, base_url,
                                                       svc.gateway, key, request_timeout)
            svc.endpoint_url = svc.endpoint_relay.url
        if gemini:
            svc.gemini_relay = _start_gemini_relay(eval_name, docs_url, svc.gateway)
        yield svc
    finally:
        for relay in (svc.gemini_relay, svc.endpoint_relay):
            if relay is not None:
                relay.stop()


def _without_keys(environ: Dict[str, str]) -> Dict[str, str]:
    """`environ` as a harness container's launching environment starts: a real endpoint key
    replaced by ENDPOINT_KEY_PLACEHOLDER (a placeholder stays as it is), and the Gemini key
    variables and GBENCH_GEMINI_RELAY_URL removed (bridge_services sets the Gemini relay's
    placeholder and address where it serves)."""
    from .base import endpoint_api_key
    out = dict(environ)
    for name in ENDPOINT_KEY_NAMES:
        if out.get(name) is not None and endpoint_api_key({"OPENAI_API_KEY": out[name]}):
            out[name] = ENDPOINT_KEY_PLACEHOLDER
    for name in (*GEMINI_KEY_NAMES, GEMINI_RELAY_URL_ENV):
        out.pop(name, None)
    return out


def refuse_forwarded_secrets(eval_name: str, docs_url: str, names: Sequence[str],
                             environ: Dict[str, str], *also: Dict[str, str]) -> None:
    """infra_required when a variable a harness container receives (`names`, values from
    `environ`) carries a credential of this process' environment or of the mappings in `also`
    (a key, a password in a proxy URL): the container runs, or may run, the model's calls."""
    secrets = set(secret_values())
    for extra in also:
        secrets.update(secret_values(extra))
    # _docker_run_command passes these two by name to every container as well
    checked = set(names) | {"TQDM_DISABLE", "TQDM_MININTERVAL"}
    bad = sorted({n for n in checked
                  if any(s in (environ.get(n) or "") for s in secrets)})
    if bad:
        raise infra_required(
            eval_name, f"{', '.join(bad)} would carry a credential into a BFCL harness container "
            "(it runs, or may run, the model's calls); gbench passes no key, token or password "
            "into one (the endpoint and Gemini keys are added by its relays on this host). Remove "
            "the credential from that variable (for a proxy, use one without a password in its "
            "URL) and re-run.", docs_url)


#: A check `_launch` calls every _WATCH_EVERY_S seconds while a container runs (set with
#: `_watching`); when it returns an exception, the container is stopped and that exception raised.
_LAUNCH_WATCH: "contextvars.ContextVar[Optional[Callable[[], Optional[BaseException]]]]" = \
    contextvars.ContextVar("bfcl_launch_watch", default=None)
_WATCH_EVERY_S = 30.0


@contextlib.contextmanager
def _watching(watch: Optional[Callable[[], Optional[BaseException]]]):
    """Containers launched inside this block are watched by `watch` (see _LAUNCH_WATCH)."""
    token = _LAUNCH_WATCH.set(watch)
    try:
        yield
    finally:
        _LAUNCH_WATCH.reset(token)


def _launch(cmd: List[str], env: Dict[str, str], timeout: Optional[float], name: str,
            stream: bool = False) -> Tuple[int, str]:
    """Run a container with `env` as the docker CLI's environment. Returns (exit code, output),
    the output (stdout and stderr, the last 256 KB) with every credential value of `env` masked;
    with `stream`, lines also go to stderr as they come. A run past `timeout` (None: no wall
    clock) stops the container and raises subprocess.TimeoutExpired; a watch installed with
    `_watching` that reports a problem stops it and raises that problem."""
    secrets = secret_values(env)
    # The output is kept as read until it is final, and masked as it is cut. Every form a
    # credential can take in it (as is, JSON-escaped, URL-encoded) that may still be being read
    # can only begin in the last `hold` characters: text from the first place there that starts
    # such a form (a proper prefix of it, a whole shorter credential that begins a longer one
    # included) is kept back unmasked, a cut never falls inside a whole form, and the text before
    # the cut is masked as one piece (redact_text, longest form first), so a credential is
    # masked whole once the rest arrives.
    forms = sorted({form for secret in secrets for form in secret_forms(secret)},
                   key=lambda f: (-len(f), f))
    hold = max((len(form) for form in forms), default=1) - 1

    def whole(text: str, cut: int) -> int:
        """`cut`, moved back to the start of any whole form it would fall inside."""
        moved = True
        while moved:
            moved = False
            for form in forms:
                start = text.find(form, max(0, cut - len(form) + 1))
                if -1 < start < cut < start + len(form):
                    cut, moved = start, True
        return cut

    def safe_cut(text: str) -> int:
        """The largest cut of `text` before which masking is final."""
        cut = len(text)
        for i in range(max(0, len(text) - hold), len(text)):
            tail = text[i:]
            if any(len(form) > len(tail) and form.startswith(tail) for form in forms):
                cut = i
                break
        return whole(text, cut)
    # No form but a multi-line raw value crosses a line break; without one, everything up to the
    # last line break read is final.
    lines_final = not any("\n" in form or "\r" in form for form in forms)
    watch = _LAUNCH_WATCH.get()
    chunks: List[str] = []
    size = [0]

    def emit(text: str) -> None:
        """Keep and stream text that is already masked."""
        chunks.append(text)
        size[0] += len(text)
        while size[0] > 262144 and len(chunks) > 1:
            size[0] -= len(chunks.pop(0))
        if stream:
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except (OSError, ValueError):
                pass

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    def pump() -> None:
        raw = ""                        # read, not yet emitted, not yet masked
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):
            raw += decoder.decode(chunk)
            # Nothing before `final` can be part of a credential still being read.
            final = safe_cut(raw)
            if lines_final:
                cut = min(final, max(raw.rfind("\n"), raw.rfind("\r")) + 1)
            else:
                cut = max(raw.rfind("\n", 0, final), raw.rfind("\r", 0, final)) + 1
            if len(raw) - cut > 65536 + hold:           # a line longer than 64 KB: in pieces
                cut = final
            cut = whole(raw, cut)                       # a line break inside a whole form
            if cut > 0:
                emit(redact_text(raw[:cut], secrets))
                raw = raw[cut:]
        raw += decoder.decode(b"", final=True)
        if raw:
            emit(redact_text(raw, secrets))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    try:
        while True:
            wait = _WATCH_EVERY_S if watch is not None else None
            if deadline is not None:
                left = max(0.0, deadline - time.monotonic())
                wait = left if wait is None else min(wait, left)
            try:
                rc = proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout) from None
                problem = None
                if watch is not None:
                    try:
                        problem = watch()
                    except Exception as e:                  # noqa: BLE001 - never stops a run
                        logger.warning("container watch failed (%s: %s); still waiting",
                                       type(e).__name__, e)
                if problem is not None:
                    raise problem
    except BaseException:
        subprocess.run(["docker", "kill", name], capture_output=True)
        proc.kill()
        proc.wait()
        reader.join(timeout=10)
        raise
    finally:
        if proc.poll() is None:
            subprocess.run(["docker", "kill", name], capture_output=True)
            proc.kill()
    reader.join(timeout=30)
    return rc, "".join(chunks)


def _container_name(kind: str) -> str:
    return f"gbench-bfcl-{kind}-{uuid.uuid4().hex[:10]}"


#: The knob that bounds each harness part on its own (legacy alias BFCL_TIMEOUT_S).
PART_TIMEOUT_KNOB = "GBENCH_BFCL_TIMEOUT_S"


def part_timeout() -> Optional[float]:
    """The wall clock of each harness part, in seconds, or None (the default) for none.

    The parts are the single-turn checker, the multi-turn part, `bfcl generate` and `bfcl
    evaluate`, each timed on its own. BFCL bounds none of them as a whole: each request has its
    client's timeout, each multi-turn entry BFCL's step limit per turn, and an entry whose
    request fails is recorded as an inference error while the run goes on. GBENCH_BFCL_TIMEOUT_S
    (legacy BFCL_TIMEOUT_S) sets one; unset, empty or 0 means none."""
    raw = (suite_env(PART_TIMEOUT_KNOB, "BFCL_TIMEOUT_S") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{PART_TIMEOUT_KNOB}={raw!r} is not a number of seconds") from None
    return value if value > 0 else None


def _seconds(value: Optional[float]) -> str:
    return "unset" if value is None else f"{float(value):g}s"


class HarnessTimeout(RuntimeError):
    """A harness part that ran past GBENCH_BFCL_TIMEOUT_S; its container was stopped.
    `progress` holds the rows the job reported (report_progress) before it was stopped;
    `earlier` the `harness_errors` records of parts of the same run stopped before it (a stopped
    `bfcl evaluate` after a stopped `bfcl generate`)."""

    def __init__(self, eval_name: str, part: str, timeout: Optional[float],
                 progress: Optional[List[Any]] = None,
                 earlier: Optional[List[Dict[str, Any]]] = None):
        self.eval_name, self.part = eval_name, part
        self.timeout = None if timeout is None else float(timeout)
        self.progress = list(progress or [])
        self.earlier = list(earlier or [])
        self.own_error = (f"{eval_name}: {part} did not finish within {PART_TIMEOUT_KNOB}="
                          f"{_seconds(self.timeout)} and was stopped")
        message = self.own_error
        if self.earlier:
            message += "; before it, " + "; ".join(str(r.get("error")) for r in self.earlier)
        super().__init__(message)

    def record(self, **extra: Any) -> Dict[str, Any]:
        """The result's `harness_errors` entry for this part (the earlier parts have their own)."""
        return {"part": self.part, "error": self.own_error, "timeout_s": self.timeout, **extra}


#: Inside an image job: the file finished units of work are appended to (report_progress), and
#: the credential values masked in it. Unset outside a job.
_PROGRESS_PATH: Optional[str] = None
_PROGRESS_SECRETS: List[str] = []
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_NAME = "progress.jsonl"


def report_progress(row: Any) -> None:
    """Inside an image job: record one finished unit of work as a JSON line in the exchange
    directory, so a part stopped by its wall clock keeps what it finished (HarnessTimeout
    .progress). A no-op outside a job."""
    if not _PROGRESS_PATH:
        return
    line = redact_text(json.dumps(row, default=str), _PROGRESS_SECRETS)
    try:
        with _PROGRESS_LOCK, open(_PROGRESS_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    except OSError as e:                  # the job's own result still carries every row
        logger.warning("could not record progress in %s (%s)", _PROGRESS_PATH, e)


def _read_progress(io_dir: str, secrets: List[str]) -> List[Any]:
    """The rows a job reported (a line cut off by the stop is left out), credentials masked."""
    rows: List[Any] = []
    try:
        with open(os.path.join(io_dir, _PROGRESS_NAME), encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return redact_secrets(rows, secrets)


def run_in_image(eval_name: str, module: str, func: str, kwargs: Dict[str, Any], *,
                 timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None,
                 env_names: Sequence[str] = (), project_root: Optional[str] = None,
                 stream: bool = False, part: Optional[str] = None,
                 network: str = NO_NETWORK) -> Any:
    """`gbench.runners.eval_suites.<module>.<func>(**kwargs)` inside the harness image; returns
    its JSON result. kwargs travel in a job file (never on the command line); `env_names` are the
    variables of `env` (default: this process' environment) the container receives. bfcl_eval's
    project root is `project_root`, mounted at the same path, or a scratch directory inside the
    container. A job that raises, or a container that exits without a result, raises RuntimeError
    with the image's (credential-masked) message. A job past `timeout` (None: no wall clock) is
    stopped and raises HarnessTimeout (named `part`) with the rows it reported.

    The job runs with no network unless `network` is the bridge network, for a job that talks to
    the model endpoint at kwargs["base_url"]: then tau_common.require_endpoint_network checks
    first that the cloud metadata server is blocked there and that a loopback endpoint answers
    at host.docker.internal, the base_url the job gets, with the proxy variables of `env_names`
    set for that network. No variable the container receives may carry a credential of this
    process' environment (refuse_forwarded_secrets): a caller whose job needs the endpoint key
    passes a relay's URL and a placeholder (bridge_services), never the key."""
    docs_url = f"docs/evals/{eval_name}.md"
    env = dict(os.environ if env is None else env)
    env["BFCL_PROJECT_ROOT"] = project_root or _IMAGE_SCRATCH_ROOT
    if network != NO_NETWORK:
        from .tau_common import container_proxy_env
        _network_args(network)                  # refuse any network but the bridge first
        kwargs = dict(kwargs, base_url=endpoint_gate(
            eval_name, str(kwargs.get("base_url") or ""), docs_url))
        proxies = container_proxy_env({n: env[n] for n in env_names if n in env})
        env.update(proxies)
        env_names = list(env_names) + [n for n in proxies if n not in env_names]
    refuse_forwarded_secrets(eval_name, docs_url, list(env_names) + ["BFCL_PROJECT_ROOT"], env)
    mounts = [(project_root, project_root, "rw")] if project_root else []
    secrets = secret_values(env)
    io_dir = tempfile.mkdtemp(prefix="gbench_bfcl_io_")
    name = _container_name(func.strip("_").replace("_", "-")[:24])
    try:
        with open(os.path.join(io_dir, "job.json"), "w", encoding="utf-8") as fh:
            json.dump({"module": module, "func": func, "kwargs": kwargs}, fh)
        program = (_package_preamble()
                   + "from gbench.runners.eval_suites.bfcl_v4_agentic import _image_main\n"
                   + f"raise SystemExit(_image_main({_IMAGE_IO_DIR!r}))\n")
        cmd = _docker_run_command(name, list(env_names) + ["BFCL_PROJECT_ROOT"],
                                  [(io_dir, _IMAGE_IO_DIR, "rw"), *mounts], program,
                                  network=network)
        try:
            rc, out = _launch(cmd, env, timeout, name, stream=stream)
        except subprocess.TimeoutExpired:
            raise HarnessTimeout(eval_name, part or f"the harness job {module}.{func}", timeout,
                                 _read_progress(io_dir, secrets)) from None
        result_path = os.path.join(io_dir, "result.json")
        if not os.path.exists(result_path):
            raise RuntimeError(f"{eval_name}: the BFCL harness image ({bfcl_image()}) exited {rc} "
                               f"without a result for {module}.{func}: "
                               f"{' '.join(out.split())[-1500:]}")
        with open(result_path, encoding="utf-8") as fh:
            payload = redact_secrets(json.load(fh), secrets)
        if not payload.get("ok"):
            message = str(payload.get("error") or "")
            if message.startswith("[INFRA REQUIRED]"):
                raise RuntimeError(message)
            raise RuntimeError(f"{eval_name}: {module}.{func} failed in the BFCL harness image "
                               f"({payload.get('error_type')}: {message})")
        return payload.get("value")
    finally:
        shutil.rmtree(io_dir, ignore_errors=True)


def _image_main(io_dir: str) -> int:
    """Entry point inside the image: run the job in `<io_dir>/job.json` and write
    `<io_dir>/result.json` ({"ok": true, "value": ...} or {"ok": false, "error": ...}), every
    credential value of the environment masked."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):      # one line per model request
        logging.getLogger(noisy).setLevel(logging.WARNING)
    global _PROGRESS_PATH, _PROGRESS_SECRETS
    secrets = secret_values(os.environ)
    _PROGRESS_PATH, _PROGRESS_SECRETS = os.path.join(io_dir, _PROGRESS_NAME), secrets
    result_path = os.path.join(io_dir, "result.json")
    try:
        with open(os.path.join(io_dir, "job.json"), encoding="utf-8") as fh:
            job = json.load(fh)
        target = importlib.import_module(f"{__package__}.{job['module']}")
        value = getattr(target, job["func"])(**(job.get("kwargs") or {}))
        payload: Dict[str, Any] = {"ok": True, "value": value}
        code = 0
    except BaseException as e:                                  # noqa: BLE001
        logger.error("BFCL harness job failed: %s", redact_text(f"{type(e).__name__}: {e}", secrets),
                     exc_info=True)
        payload, code = {"ok": False, "error_type": type(e).__name__, "error": str(e)}, 1
    finally:
        _PROGRESS_PATH, _PROGRESS_SECRETS = None, []
    tmp = result_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(redact_secrets(payload, secrets), fh, default=str)
    os.replace(tmp, result_path)
    return code


def image_probe() -> Dict[str, Any]:
    """Inside the image: the installed release, the agentic category mapping and the agentic
    file pin (harness_mismatch)."""
    try:
        version = importlib.metadata.version("bfcl-eval")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"release": version, "agentic_categories": _agentic_categories(),
            "agentic_mismatch": harness_mismatch()}


# --------------------------------------------------------------------------- #
# prerequisites
# --------------------------------------------------------------------------- #
def _agentic_categories() -> List[str]:
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
    return list(TEST_COLLECTION_MAPPING["agentic"])


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def harness_mismatch() -> Optional[str]:
    """None when every file in _AGENTIC_PINNED_BLOBS matches the pinned commit; else why not.
    Runs where bfcl_eval is importable: inside the harness image (image_probe)."""
    import bfcl_eval
    root = Path(bfcl_eval.__file__).resolve().parent
    differ = [rel for rel, blob in _AGENTIC_PINNED_BLOBS.items()
              if not (root / rel).is_file() or _git_blob_sha1((root / rel).read_bytes()) != blob]
    if not differ:
        return None
    try:
        version = importlib.metadata.version("bfcl-eval")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return (f"the bfcl-eval {version} in the harness image ({root}) differs from "
            f"gorilla@{BFCL_EVAL_COMMIT} in {len(differ)} file(s) the agentic track uses "
            f"({', '.join(differ[:4])}{', ...' if len(differ) > 4 else ''}); the image must hold "
            f"bfcl-eval=={BFCL_EVAL_RELEASE}: `{image_build_command()}`")


def search_backend() -> Tuple[Optional[str], str]:
    """(SEARCH_BACKEND, "") when BFCL's web_search tool can be served, else (None, why).

    The gate the judged suites use: a Gemini key is set (base.gemini_api_keys: GEMINI_API_KEYS,
    else GEMINI_API_KEY), the Gemini API does not reject it, and it serves every model of
    gbench's Gemini cascade (base.gemini_key_live_valid, which checks the judge and search
    cascades; a check that gets no answer does not block). The reason never holds the key."""
    keys = gemini_api_keys()
    if not keys:
        return None, ("GEMINI_API_KEY (or GEMINI_API_KEYS) is unset: BFCL's web_search tool is "
                      "served by Gemini Google Search grounding")
    ok, why = gemini_key_live_valid(",".join(keys))
    if not ok and str(why).startswith(MODELS_NOT_SERVED_PREFIX):
        # The key is accepted; a model of gbench's Gemini cascade is not served to it.
        return None, redact_text(f"{why}: BFCL's web_search tool is served by Gemini Google "
                                 f"Search grounding with the models of that cascade", keys)
    if not ok and "GEMINI_OPENAI_BASE_URL" in str(why):
        # The key is not sent to a refused GEMINI_OPENAI_BASE_URL.
        return None, redact_text(f"the Gemini key cannot be used ({why}): BFCL's web_search "
                                 f"tool is served by Gemini Google Search grounding", keys)
    if not ok:
        return None, redact_text(f"the Gemini API rejected the key (GEMINI_API_KEYS / "
                                 f"GEMINI_API_KEY: {why}): BFCL's web_search tool is served by "
                                 f"Gemini Google Search grounding", keys)
    return SEARCH_BACKEND, ""


def search_models() -> List[str]:
    """The grounding models that serve the searches, in cascade order: base.search_cascade()
    (GBENCH_SEARCH_MODELS, else GBENCH_SEARCH_MODEL, else gbench's one Gemini cascade,
    base.judge_cascade(): GBENCH_JUDGE_MODELS, else GBENCH_JUDGE_MODEL, else base.GEMINI_CASCADE).
    The web_search `bfcl generate` receives the same list (BridgeServices.container_env) and the
    Gemini relay forwards only these models."""
    return search_cascade()


def search_substitution() -> Dict[str, Any]:
    """The result's record of what serves BFCL's web_search tool in place of its own search."""
    return {
        "official": ("DuckDuckGo via SerpAPI (bfcl_eval web_search.py "
                     "WebSearchAPI.search_engine_query)"),
        "served_by": ("Gemini Google Search grounding: generateContent with the google_search "
                      "tool, the query as the prompt (search_tool.search_with_fallback)"),
        "models": search_models(),
        # where the list came from: GBENCH_SEARCH_MODELS, GBENCH_SEARCH_MODEL, GBENCH_JUDGE_MODELS,
        # GBENCH_JUDGE_MODEL or "default" (base.GEMINI_CASCADE)
        "models_source": search_cascade_source(),
        "tool_contract": ("BFCL's own: the web_search function docs "
                          "(multi_turn_func_doc/web_search.json), search_engine_query(keywords, "
                          "max_results=10, region='wt-wt') and its result, a list of {title, "
                          "href, body} ({title, href} in web_search_no_snippet) cut to "
                          "max_results; fetch_url_content unchanged"),
        "differences": ("region is not applied (grounding takes none); title is the source's "
                        "title or domain as grounding reports it; href is Google's grounding "
                        "redirect URL to the source, which fetch_url_content follows; body is the "
                        "grounded answer text Gemini attributes to that source, or the start of "
                        "the grounded answer when it attributes none, not a page snippet; a query "
                        "with a grounded answer and no source returns one result titled "
                        "grounded-answer, one with neither an empty list; a search no model of "
                        "the cascade could serve returns BFCL's own error text and makes the Web "
                        "Search setting infra_required"),
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": SEARCH_SUBSTITUTION_REASON,
    }


# --------------------------------------------------------------------------- #
# BFCL's web_search tool served by Gemini grounding (installed inside the harness child)
# --------------------------------------------------------------------------- #
def _bfcl_results(results: List[Dict[str, str]], max_results: Any,
                  show_snippet: bool) -> List[Dict[str, str]]:
    """search_tool's grounded results ([{title, url, snippet}]) in the shape BFCL's
    search_engine_query returns: [{title, href, body}], or [{title, href}] without snippets, cut
    to `max_results` as upstream cuts its results (`[:max_results]`). The "no results" answer is
    the empty list."""
    if len(results) == 1 and results[0] == {"title": "no results", "url": "",
                                            "snippet": "The search returned nothing."}:
        results = []
    out = []
    for row in results[:max_results]:
        item = {"title": row.get("title", ""), "href": row.get("url", "")}
        if show_snippet:
            item["body"] = row.get("snippet", "")
        out.append(item)
    return out


def _serve_web_search_with_gemini(log_path: Optional[str]) -> None:
    """Inside the harness child of the web_search `bfcl generate`: serve BFCL's web_search tool
    with Gemini Google Search grounding and record every search in `log_path` (JSON lines).

    `WebSearchAPI.search_engine_query` is replaced on the class by a plain function (the harness
    maps a backend's methods with inspect.getmembers(instance, inspect.ismethod), which a
    staticmethod would fail) with the same arguments. It sends `keywords` to the search_tool
    grounding cascade (search_with_fallback) and returns BFCL's result shape (_bfcl_results); a
    search no model of the cascade could serve returns BFCL's own error dict (SEARCH_FAILED_TEXT)
    and is logged with its error, which makes the Web Search setting infra_required. Every
    credential value of the environment is masked in the results, in the log and in every tool
    result the harness hands the model (it runs the model's calls with eval() in this process)."""
    if not log_path:
        return
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search as ws
    from bfcl_eval.model_handler import base_handler

    from . import search_tool
    secrets = secret_values(os.environ)
    lock = threading.Lock()

    def record(entry: Dict[str, Any]) -> None:
        line = redact_text(json.dumps(entry), secrets)
        with lock, open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def search_engine_query(self, keywords, max_results=10, region="wt-wt"):
        # A max_results the slice cannot take raises the TypeError upstream's `[:max_results]`
        # raises (the harness hands the model "Error during execution: ..."), before a search is
        # spent on it.
        [][:max_results]
        try:
            results, backend = search_tool.search_with_fallback(str(keywords), max_results=None)
            error = None
            if not str(backend).startswith("gemini:"):
                error = str((results or [{}])[0].get("snippet") or "no results and no error text")
        except Exception as e:                              # noqa: BLE001 - a failed search
            backend, error = "error", f"{type(e).__name__}: {e}"
        if error is not None:
            record({"backend": backend, "results": 0, "error": redact_text(error, secrets)[:300]})
            return {"error": SEARCH_FAILED_TEXT}
        out = redact_secrets(_bfcl_results(results, max_results, bool(self.show_snippet)), secrets)
        record({"backend": backend, "results": len(out), "error": None})
        return out

    search_engine_query.__doc__ = ws.WebSearchAPI.search_engine_query.__doc__
    search_engine_query.gbench_backend = SEARCH_BACKEND
    ws.WebSearchAPI.search_engine_query = search_engine_query

    execute = base_handler.execute_multi_turn_func_call

    def execute_multi_turn_func_call(*args, **kwargs):
        results, instances = execute(*args, **kwargs)
        return [redact_text(r, secrets) for r in results], instances

    base_handler.execute_multi_turn_func_call = execute_multi_turn_func_call


# --------------------------------------------------------------------------- #
# the execution guard (installed inside the harness child of every `bfcl` command)
# --------------------------------------------------------------------------- #
def _install_execution_guard(log_path: Optional[str]) -> None:
    """Inside the harness child of a `bfcl` command, before the CLI dispatches: the bfcl suite's
    execution guard (bfcl._ExecutionGuard, the one its multi-turn part runs under) in place of
    `eval` in bfcl_eval's multi_turn_utils, for the rest of the process, so that `bfcl
    generate`'s agentic loop (BaseHandler.inference_multi_turn_FC ->
    execute_multi_turn_func_call) runs only the environment functions upstream intends.

    Upstream maps every public method of the entry's involved backends (the memory API of a
    memory category, WebSearchAPI for web_search) to `<instance>.<method>`, evaluates the call
    string BFCL's FC decoder renders, `name(k=<repr of a JSON value>)`, and tries to refuse
    anything else with a name blocklist ("Before calling `eval`, we need to make sure that the
    function call is safe"). The guard lets exactly those calls through (an instance of the
    entry's backends, a public method, literal arguments; an unknown bare name raises NameError
    as upstream), with the same result and error text, and answers any other call string (a
    builtin, a module global, code in the name or the arguments) with upstream's own "Function
    call X is not allowed." Each refusal is appended to `log_path` (JSON lines, credentials
    masked) when it is set; the run's result counts and names them."""
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu

    from .bfcl import _ExecutionGuard
    secrets = secret_values(os.environ)
    lock = threading.Lock()

    def refused(source: str) -> None:
        if not log_path:
            return
        line = redact_text(json.dumps({"call": str(source)[:200]}), secrets)
        with lock, open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    mtu.eval = _ExecutionGuard(mtu, on_refused=refused)


def _read_guard_log(path: Optional[Path], secrets: Sequence[str] = ()) -> List[str]:
    """The call strings the execution guard refused during generation (the log
    _install_execution_guard writes), credentials masked."""
    if path is None:
        return []
    return [redact_text(str(row.get("call") or ""), list(secrets))
            for row in _read_search_log(path)]


# --------------------------------------------------------------------------- #
# the web search log (written inside the harness child during generation)
# --------------------------------------------------------------------------- #
def _read_search_log(path: Path) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def search_log_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts over the web search log: the searches, how many each grounding model served
    (`by_backend`), the ones no model served (`failed`, by backend and error in `failures`), and
    the answered searches that had no result (`empty_results`)."""
    by_backend: Dict[str, int] = {}
    failures: Dict[str, int] = {}
    empty = 0
    for row in rows:
        backend = str(row.get("backend") or "unknown")
        by_backend[backend] = by_backend.get(backend, 0) + 1
        error = row.get("error")
        if error or not backend.startswith("gemini:"):
            label = f"{backend}: {error or 'no error text'}"
            failures[label] = failures.get(label, 0) + 1
        elif not row.get("results"):
            empty += 1
    return {"requests": len(rows), "by_backend": dict(sorted(by_backend.items())),
            "failed": sum(failures.values()), "failures": failures, "empty_results": empty}


def _failure_text(summary: Dict[str, Any]) -> str:
    """What went wrong with the searches in `summary`."""
    labels = [f"{n} x {label}" for label, n in sorted(summary["failures"].items(),
                                                      key=lambda kv: -kv[1])]
    return (f"Gemini Google Search grounding served none of the models of its cascade for "
            f"{summary['failed']} of the {summary['requests']} web searches made during "
            f"generation ({'; '.join(labels)}), and the harness answered those searches with its "
            f"error text instead of results")


def check_bfcl_prerequisites(base_url: str) -> Tuple[bool, str, List[str]]:
    """Validate what the track needs before anything is generated: a writable project root that
    holds no .env (dotenv_problem: upstream loads it into the harness process), a
    reachable model endpoint (asked with its key when OPENAI_API_KEY holds one), the harness
    image, and inside it the installed release's agentic category mapping and files
    (image_probe), and last the bridge network `bfcl generate` runs on (bridge_network_check:
    the cloud metadata server blocked there, the host names the container resolves, and the
    bridge gateway where gbench's relays listen: `bfcl generate` always reaches the endpoint
    through its endpoint relay, always_relay), which raises
    infra_required itself, naming the suite that runs (suite_context: bfcl when it runs this
    track as its agentic part, else bfcl_v4_agentic) and its doc. With suite_context's
    `web_search`, the checks the web_search `bfcl generate` needs are included.

    Returns (ok, reason, agentic_categories). The search backend is checked by the caller
    once the categories to run are known: only the web_search categories need it.
    """
    ctx = _suite()
    root = harness_project_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".gbench_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        return False, f"GBENCH_BFCL_PROJECT_ROOT {root} is not writable ({e})", []
    problem = dotenv_problem(root)
    if problem:
        return False, problem, []

    key = endpoint_key()
    try:
        # This process holds the key; the containers never do.
        request: Any = f"{base_url.rstrip('/')}/models"
        if key:
            request = urllib.request.Request(request, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=10) as r:
            if r.status != 200:
                return False, f"model endpoint {base_url} returned HTTP {r.status}", []
    except Exception as e:
        return False, redact_text(f"model endpoint {base_url} is not reachable ({e})",
                                  [key] if key else []), []
    problem = image_problem()
    if problem:
        return False, problem, []
    try:
        found = run_in_image(ctx.eval_name, "bfcl_v4_agentic", "image_probe", {}, timeout=600)
    except Exception as e:
        return False, f"the BFCL harness image could not be probed ({e})", []
    if found.get("release") != BFCL_EVAL_RELEASE:
        return False, (f"the harness image {bfcl_image()} holds bfcl-eval "
                       f"{found.get('release') or 'none'}, not {BFCL_EVAL_RELEASE}: "
                       f"`{image_build_command()}`"), []
    harness_categories = found.get("agentic_categories") or []
    if set(harness_categories) != set(AGENTIC_CATEGORIES):
        return False, (f"the harness image's agentic categories {sorted(harness_categories)} "
                       f"differ from the ones gbench runs and validates --eval-categories "
                       f"against {sorted(AGENTIC_CATEGORIES)}"), []
    if found.get("agentic_mismatch"):
        return False, found["agentic_mismatch"], []
    # `bfcl generate` runs the model's calls on the bridge network: the cloud metadata server must
    # be blocked there, the names it resolves must resolve, and the relays' gateway must be
    # reachable: generate always goes through gbench's endpoint relay, keyless too (2026-10-13;
    # raises infra_required otherwise).
    bridge_network_check(ctx.eval_name, base_url, ctx.docs_url, web_search=ctx.web_search,
                         key=key, always_relay=generate_always_relayed())
    return True, "", list(AGENTIC_CATEGORIES)


def select_categories(available: List[str], requested: Any) -> Tuple[List[str], bool]:
    """(categories to run, is_subset) for the operator's `--eval-categories`, if any.

    An unknown name is an error, not a silent drop: a typo would otherwise shrink the run.
    """
    if not requested:
        return list(available), False
    names = ([c.strip() for c in requested.split(",")] if isinstance(requested, str)
             else [str(c).strip() for c in requested])
    names = [c for c in names if c]
    unknown = [c for c in names if c not in available]
    if unknown:
        raise ValueError(f"{_suite().eval_name}: unknown --eval-categories {unknown}; the agentic "
                         f"categories are {available}")
    chosen = [c for c in available if c in names]
    return chosen, set(chosen) != set(available)


def _unweighted(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def agentic_scores(per_cat: Dict[str, Dict[str, Any]],
                   unscored: Sequence[str] = ()) -> Dict[str, Optional[float]]:
    """BFCL's Agentic aggregation (eval_runner_helper.py): two levels of unweighted means.

    summary_web_search = mean(web_search_base, web_search_no_snippet)
    summary_memory     = mean(memory_kv, memory_vector, memory_rec_sum)
    overall            = mean(summary_web_search, summary_memory)

    BFCL counts a category that did not run as 0 in its overall (and displays N/A;
    bfcl_zero_filled_overall gives that value). Here, as the bfcl suite reports its columns, a
    group is None unless every one of its categories ran and is in the score, and the overall is
    None unless both groups are: a subset or an --eval-limit draw that left a category out has no
    Agentic score (the run's headline is then its pooled accuracy). A category that ran but has
    no entry left in the score (`unscored`: every entry a harness error) makes its group and the
    overall None too, never a mean over the rest. The means are computed from the exact rates and
    rounded once, at the end.
    """
    def acc(c: str) -> float:
        # The exact rate from the counts, as BFCL averages the unrounded accuracies; the
        # 2-decimal `accuracy` is only a display value.
        row = per_cat[c]
        if row.get("total"):
            return 100.0 * row.get("correct", 0) / row["total"]
        return float(row.get("accuracy") or 0.0)

    def group(cats: Tuple[str, ...]) -> Optional[float]:
        if any(c in unscored for c in cats) or not all(c in per_cat for c in cats):
            return None
        return _unweighted([acc(c) for c in cats])

    web, mem = group(WEB_SEARCH_CATEGORIES), group(MEMORY_CATEGORIES)
    overall = _unweighted([web, mem]) if web is not None and mem is not None else None

    def _r(v: Optional[float]) -> Optional[float]:
        return None if v is None else round(v, 2)
    return {"summary_web_search": _r(web), "summary_memory": _r(mem),
            "overall_accuracy_agentic": _r(overall)}


def bfcl_zero_filled_overall(per_cat: Dict[str, Dict[str, Any]]) -> float:
    """BFCL's own Agentic Overall Acc for the categories in `per_cat` (eval_runner_helper.py
    get_category_score counts a category that did not run as 0; both groups are unweighted
    means over their full category lists), from the exact rates, rounded once."""
    def acc(c: str) -> float:
        row = per_cat.get(c)
        if not row:
            return 0.0
        if row.get("total"):
            return 100.0 * row.get("correct", 0) / row["total"]
        return float(row.get("accuracy") or 0.0)
    web = _unweighted([acc(c) for c in WEB_SEARCH_CATEGORIES])
    mem = _unweighted([acc(c) for c in MEMORY_CATEGORIES])
    return round(_unweighted([web, mem]), 2)


# --------------------------------------------------------------------------- #
# model registration
# --------------------------------------------------------------------------- #
#: The variable carrying each request's client timeout (seconds) into the harness container.
REQUEST_TIMEOUT_ENV = "GBENCH_BFCL_REQUEST_TIMEOUT_S"


def agentic_request_timeout() -> int:
    """Seconds one harness request may take: sized, as the bfcl suite's multi-turn client is, for
    the longest reply the served context window allows, since no max_tokens is sent
    (base.request_timeout_s of the window, or of base.UNCAPPED_TIMEOUT_TOKENS when the server
    reports none). BFCL's own handler for a locally served model sends timeout=72000 "to avoid
    timeout errors" (base_oss_handler.py); OpenAICompletionsHandler keeps the OpenAI SDK's 600 s,
    written for OpenAI's hosted API."""
    return request_timeout_s(get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)


def _patched_openai_handler():
    """`OpenAICompletionsHandler` plus gbench's `--thinking` toggle and the request timeout, and
    nothing else.

    Requests, decoding and the multi-turn loop are the harness' own; decode_ast / decode_execute
    are not overridden. A reply without tool calls carries no `tool_calls` field (OpenAI, and
    vLLM since it stopped sending an empty list), so _parse_query_response_FC keeps its text, the
    FC decoder raises on it, and the harness logs "Failed to decode the model response. Proceed
    to next turn." and ends the turn, as for every FC model; that text is what the agentic
    checker grades (the last message that does not decode to a call).
    """
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    class _GbenchOpenAICompletionsHandler(OpenAICompletionsHandler):
        def generate_with_backoff(self, **kwargs):
            # Toggle the served model's reasoning switch to match --thinking. The base handler
            # builds the request itself, so gbench's base.py chat_template_kwargs injection is
            # bypassed here; carry the intent via env (set by run_bfcl_v4_agentic) and add the
            # same extra_body every other suite uses. vLLM reads chat_template_kwargs; the
            # OpenAI client forwards unknown fields only through extra_body.
            if os.environ.get("GBENCH_BFCL_ENABLE_THINKING") == "1":
                eb = dict(kwargs.get("extra_body") or {})
                ctk = dict(eb.get("chat_template_kwargs") or {})
                ctk.setdefault("enable_thinking", True)
                eb["chat_template_kwargs"] = ctk
                kwargs["extra_body"] = eb
            # The per-request timeout (agentic_request_timeout) replaces the client's 600 s;
            # the client's retries are unchanged.
            timeout = os.environ.get(REQUEST_TIMEOUT_ENV)
            if timeout:
                kwargs.setdefault("timeout", float(timeout))
            return super().generate_with_backoff(**kwargs)

    return _GbenchOpenAICompletionsHandler


def _register_model(model_name: str) -> str:
    """Register the served model with BFCL's handler registry (BFCL ships no entry for it).

    Returns the key to pass as ``--model``. The entry uses the harness' OpenAI chat-completions
    handler in FC mode (see below); the served model id is used verbatim so the harness' OpenAI
    client requests the right model. The entry always replaces whatever the registry holds under
    that key: a served id that happens to equal one of BFCL's own entries (a provider's API
    model, a prompt-mode handler) must still be run as the served endpoint, with gbench's
    handler, never through the entry BFCL ships for that name. The metadata names no vendor: the
    served model's organization is the prefix of its id when it has one ("org/name"), else
    unknown, and its license is unknown.
    """
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    # Use the OpenAI *chat completions* handler in function-calling mode, not a local
    # prompt-mode handler, which hand-builds a raw prompt string and parses prompt-style
    # `[func(a=1)]` text back out; against a vLLM-served model with native tool calling that
    # decode fails on every row ("Failed to decode the model response").
    # OpenAICompletionsHandler talks OpenAI tool-calling and honours OPENAI_BASE_URL, which
    # we point at the gbench endpoint.
    org = model_name.split("/", 1)[0] if "/" in model_name else "unknown"
    MODEL_CONFIG_MAPPING[model_name] = ModelConfig(
        model_name=model_name,
        display_name=f"{model_name} (gbench)",
        url="",
        org=org,
        license="unknown",
        model_handler=_patched_openai_handler(),
        input_price=None,
        output_price=None,
        is_fc_model=True,       # native tool calling via /v1/chat/completions
        # As upstream's OpenAICompletionsHandler FC entry (constants/model_config.py): its tools
        # carry '.' as '_' (convert_to_tool), and ast_checker, the only reader, maps names back.
        underscore_to_dot=True,
    )
    logger.info("bfcl_v4_agentic: registered '%s' (OpenAICompletionsHandler, FC mode)", model_name)
    return model_name


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
#: Program of one `bfcl` command, run in the harness image after the package preamble. The
#: model is registered there, before typer dispatches the command (gbench's own process never
#: imports bfcl_eval). The model directories already under the project root's result and score
#: directories are registered too: `bfcl evaluate` builds its leaderboard CSVs over every model
#: directory it finds there and needs each one registered. When the run has web_search
#: categories, GBENCH_BFCL_SEARCH_LOG names the file the child records searches in, and the child
#: serves BFCL's web_search tool with Gemini Google Search grounding
#: (_serve_web_search_with_gemini). Before the CLI dispatches, the child installs the bfcl
#: suite's execution guard in place of `eval` in bfcl_eval's multi_turn_utils
#: (_install_execution_guard), which records each refusal in GBENCH_BFCL_GUARD_LOG, makes
#: upstream's load_dotenv of <project root>/.env load nothing (_disable_dotenv), and lets
#: evaluate summarize one latency (_allow_one_latency).
_CHILD_BOOTSTRAP = """
import sys, os, glob
from gbench.runners.eval_suites.bfcl_v4_agentic import (_disable_dotenv,
                                                        _allow_one_latency,
                                                        _install_execution_guard,
                                                        _register_model,
                                                        _serve_web_search_with_gemini)
_register_model({model!r})
for _p in glob.glob(os.path.join(os.environ["BFCL_PROJECT_ROOT"], "*", "*")):
    if os.path.isdir(_p):
        _name = os.path.basename(_p).replace("_", "/", 1)
        _register_model(_name)
_serve_web_search_with_gemini(os.environ.get("GBENCH_BFCL_SEARCH_LOG"))
_install_execution_guard(os.environ.get("GBENCH_BFCL_GUARD_LOG"))
sys.argv = ["bfcl"] + {args!r}
from bfcl_eval.__main__ import cli
_disable_dotenv()
_allow_one_latency()
cli()
"""


#: The variables of the run environment a `bfcl` command receives in the image, when set. None
#: holds a key: OPENAI_API_KEY / REMOTE_OPENAI_API_KEY hold a placeholder (the endpoint relay adds
#: a real key) and GEMINI_API_KEY(S) the Gemini relay's placeholder, in the web_search
#: `bfcl generate` only. OPENAI_DEFAULT_HEADERS, which upstream's handler would read, is never
#: passed: headers can carry a credential, and gbench's endpoint takes none.
_CHILD_ENV_NAMES = ("BFCL_PROJECT_ROOT", "REMOTE_OPENAI_BASE_URL", "REMOTE_OPENAI_API_KEY",
                    "REMOTE_OPENAI_TOKENIZER_PATH", "OPENAI_BASE_URL", "OPENAI_API_KEY",
                    "GBENCH_BFCL_ENABLE_THINKING", REQUEST_TIMEOUT_ENV, SEARCH_LOG_ENV,
                    GUARD_LOG_ENV, GEMINI_RELAY_URL_ENV) + GEMINI_KEY_NAMES \
    + SEARCH_KNOB_NAMES + CASCADE_ENV_NAMES + _PROXY_NAMES


#: The `bfcl` commands that talk to the model endpoint and run on the bridge network; every
#: other command (`bfcl evaluate`) runs with no network.
NETWORK_JOBS = ("generate",)


def _has_gemini_key(env: Dict[str, str]) -> bool:
    return any((env.get(name) or "").strip() for name in GEMINI_KEY_NAMES)


def _run_cli(args: List[str], env: Dict[str, str], timeout: int,
             model: str = "", backend: Optional[str] = None) -> Tuple[int, str]:
    """Run one `bfcl` command in the harness image (`backend` is informational only): the
    project root is mounted at the same path, and the container gets the _CHILD_ENV_NAMES that
    `env` sets (values through the environment, never the command line). Returns (exit code,
    output) with every credential value of `env` masked; raises subprocess.TimeoutExpired, with
    the container stopped, after `timeout` seconds.

    No key reaches the container (the harness executes the model's calls in it): a real endpoint
    key in `env` (OPENAI_API_KEY) becomes the placeholder ENDPOINT_KEY_PLACEHOLDER, the Gemini key
    variables are removed, and refuse_forwarded_secrets stops a launch that would still pass a
    credential. `bfcl generate` (NETWORK_JOBS) runs on the bridge network inside bridge_services
    (always_relay): gbench's endpoint relay on the bridge gateway relays its model requests, with
    or without an endpoint key (it adds a real key, and for the bfcl_v4_agentic suite the run's
    --max-output-tokens on requests that carry none), and the container gets the relay's URL as
    REMOTE_OPENAI_BASE_URL / OPENAI_BASE_URL. When
    `env` carries a Gemini key (the generate of the web_search categories), gbench's Gemini relay
    serves the searches and the container gets its placeholder and address; the grounding links'
    host must resolve there. The relays' records go to suite_context's `relays` list. Every check
    names the suite that runs (suite_context). `bfcl evaluate` runs with no network.

    Upstream's `generate` and `evaluate` load <project root>/.env into their process: a
    project root that holds one is refused (infra_required, dotenv_problem) before the container
    starts, and the harness child makes that load a no-op (_disable_dotenv)."""
    code = _package_preamble() + _CHILD_BOOTSTRAP.format(model=model, args=args)
    root = env["BFCL_PROJECT_ROOT"]
    name = _container_name(str(args[0]) if args else "cli")
    ctx = _suite()
    problem = dotenv_problem(Path(root))
    if problem:
        raise infra_required(ctx.eval_name, problem, ctx.docs_url)
    original = env
    env = _without_keys(env)
    if not (args and args[0] in NETWORK_JOBS):
        names = [n for n in _CHILD_ENV_NAMES if env.get(n) is not None]
        refuse_forwarded_secrets(ctx.eval_name, ctx.docs_url, names, env, original)
        cmd = _docker_run_command(name, names, [(root, root, "rw")], code, workdir=root,
                                  network=NO_NETWORK)
        return _launch(cmd, env, timeout, name)
    from .tau_common import ENDPOINT_NETWORK, container_proxy_env
    url = original.get("OPENAI_BASE_URL") or original.get("REMOTE_OPENAI_BASE_URL") or ""
    # always_relay: every `bfcl generate` goes through gbench's endpoint relay, keyless too
    # (2026-10-13; the module docstring's Token budget).
    with bridge_services(ctx.eval_name, ctx.docs_url, url, key=endpoint_key(original),
                         gemini=_has_gemini_key(original),
                         request_timeout=original.get(REQUEST_TIMEOUT_ENV),
                         environ=original,
                         always_relay=generate_always_relayed()) as svc:
        try:
            if svc.endpoint_relay is None:
                inner = endpoint_gate(ctx.eval_name, url, ctx.docs_url)
            else:
                inner = svc.endpoint_url
            env = dict(env, **container_proxy_env(env))
            env.update(svc.container_env(env))
            for var in ("OPENAI_BASE_URL", "REMOTE_OPENAI_BASE_URL"):
                if env.get(var) is not None:
                    env[var] = inner
            names = [n for n in _CHILD_ENV_NAMES if env.get(n) is not None]
            refuse_forwarded_secrets(ctx.eval_name, ctx.docs_url, names, env, original)
            cmd = _docker_run_command(name, names, [(root, root, "rw")], code, workdir=root,
                                      network=ENDPOINT_NETWORK)
            return _launch(cmd, env, timeout, name)
        finally:
            if ctx.relays is not None and (svc.endpoint_relay or svc.gemini_relay):
                ctx.relays.append({"command": " ".join(str(a) for a in args[:1]),
                                   **svc.record()})


def _redact_tree(root: Path, secrets: List[str]) -> int:
    """Mask every credential value in the text files under `root` (the harness' result or score
    directory); returns the number of files rewritten. A backstop: credentials are masked where
    they could enter (the search results and tool results in the harness child), so this normally
    finds nothing."""
    changed = 0
    if not secrets or not root.is_dir():
        return 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".csv", ".txt", ".log"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        masked = redact_text(text, secrets)
        if masked != text:
            path.write_text(masked, encoding="utf-8")
            changed += 1
    return changed


def _score_file_name(category: str) -> str:
    """The harness' score file for one category (eval_runner_helper.save_eval_results)."""
    return f"BFCL_v4_{category}_score.json"


def _score_files(score_root: Path, model_name: str,
                 categories: List[str]) -> List[Tuple[str, Path]]:
    """(category, path) of this model's score files for exactly `categories`.

    The harness writes `<score dir>/<model with '/' -> '_'>/<track>/.../BFCL_v4_<cat>_score.json`
    (nested by track, so the search recurses). Only that model directory is searched and only
    the selected categories' files are read: other models' directories and categories this
    run did not select are never mistaken for this run's scores.
    """
    model_dir = score_root / model_name.replace("/", "_")
    wanted = {_score_file_name(c): c for c in categories}
    if not model_dir.is_dir():
        return []
    return [(wanted[p.name], p) for p in sorted(model_dir.rglob("*_score.json"))
            if p.name in wanted]


def _clear_stale_scores(score_root: Path, model_name: str, categories: List[str]) -> int:
    """Delete this model's score files for `categories` before `bfcl evaluate` runs.

    bfcl-eval never deletes a score file; it only overwrites the ones it writes. A failed or
    partial evaluate would otherwise leave an earlier run's files in place, and they would be
    read back as this run's result. Returns the number of files removed.
    """
    removed = 0
    for _cat, path in _score_files(score_root, model_name, categories):
        path.unlink()
        removed += 1
    return removed


def _result_files(result_root: Path, model_name: str, categories: List[str]) -> List[Path]:
    """This model's result files (and memory prerequisite result files) for `categories`, as
    `bfcl generate` names them (BaseHandler.write: <result dir>/<model with '/' -> '_'>/...)."""
    model_dir = result_root / model_name.replace("/", "_")
    wanted = ({f"BFCL_v4_{c}_result.json" for c in categories}
              | {f"BFCL_v4_{c}_prereq_result.json" for c in categories})
    if not model_dir.is_dir():
        return []
    return [p for p in sorted(model_dir.rglob("*_result.json")) if p.name in wanted]


def _clear_results(result_root: Path, model_name: str, categories: List[str]) -> int:
    """Delete this model's result files for `categories` under `result_root`; returns how many."""
    removed = 0
    for path in _result_files(result_root, model_name, categories):
        path.unlink()
        removed += 1
    return removed


def _drop_cut_lines(result_root: Path, model_name: str, categories: List[str]) -> int:
    """Remove the lines of this model's result files that are not whole JSON (a write the stop
    cut off); returns the number of lines removed."""
    dropped = 0
    for path in _result_files(result_root, model_name, categories):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        whole = []
        for line in lines:
            try:
                json.loads(line)
                whole.append(line)
            except ValueError:
                if line.strip():
                    dropped += 1
        if len(whole) != len(lines):
            path.write_text("".join(f"{line}\n" for line in whole), encoding="utf-8")
    return dropped


def _count_unfinished(per_cat: Dict[str, Dict[str, Any]],
                      planned: Dict[str, int]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """(per-category counts over the entries the harness finished, entries not finished), for
    every planned category. A planned entry the stopped `bfcl generate` did not finish was never
    measured: a harness error, left out of its category's correct and total (`unfinished` counts
    them). A category with no finished entry has total 0 (left unscored by _leave_out)."""
    out: Dict[str, Dict[str, Any]] = {}
    missing_total = 0
    for cat, n in planned.items():
        row = per_cat.get(cat) or {"correct": 0, "total": 0}
        scored = min(int(row.get("total") or 0), n)
        correct = min(int(row.get("correct") or 0), scored)
        missing = n - scored
        missing_total += missing
        out[cat] = {"correct": correct, "total": scored,
                    "accuracy": round(100.0 * correct / scored, 2) if scored else None,
                    "unfinished": missing}
    return out, missing_total


def _parse_score_traces(score_root: Path, model_name: str, categories: List[str],
                        result_root: Path) -> List[Dict[str, Any]]:
    """Per-question detail: one trace per scored entry the harness generated (its category's result file), for the
    categories `bfcl evaluate` wrote a score file for: `is_correct` is whether the score file
    (which lists only the entries its checker scored wrong) leaves it out, with the error details
    of the score row for a wrong one, and the model's answer (clipped). So every verdict, correct
    ones included, can be re-derived from the result."""
    wrong: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with_scores = set()
    for cat, path in _score_files(score_root, model_name, categories):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        with_scores.add(cat)
        for line in lines[1:]:                      # line 0 is the category summary
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("id") is not None and not row.get("valid"):
                wrong[(cat, str(row["id"]))] = row
    wanted = {f"BFCL_v4_{c}_result.json": c for c in categories if c in with_scores}
    traces: List[Dict[str, Any]] = []
    for path in _result_files(result_root, model_name, list(wanted.values())):
        cat = wanted.get(path.name)
        if cat is None:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            entry_id = str(row.get("id") or "") if isinstance(row, dict) else ""
            if not entry_id or "_prereq" in entry_id:
                continue
            srow = wrong.get((cat, entry_id))
            err = (srow or {}).get("error") or {}
            det = err.get("details") or {}
            answer = det.get("model_response") if srow else row.get("result")
            traces.append({
                "sample_idx": len(traces),
                "category": cat,
                "question_id": entry_id,
                "is_correct": srow is None,
                "status": "OK",
                "error_type": err.get("error_type"),
                "judge_grade": err.get("error_message"),
                "response_text": str(answer if answer is not None else "")[:4000],
                "gold_answer": str(det.get("possible_answers") or "")[:1000] if srow else None,
            })
    return traces


def _score_error_types(score_root: Path, model_name: str, categories: List[str],
                       exclude: Optional[set] = None) -> Dict[str, int]:
    """How often each BFCL error type occurs across the score files (every failed question),
    leaving out the (category, entry id) pairs in `exclude` (entries left out of the score as
    harness errors)."""
    counts: Dict[str, int] = {}
    exclude = exclude or set()
    for _cat, path in _score_files(score_root, model_name, categories):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and (_cat, str(row.get("id"))) in exclude:
                continue
            err = (row.get("error") or {}).get("error_type") if isinstance(row, dict) else None
            if err:
                counts[err] = counts.get(err, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _parse_scores(score_root: Path, model_name: str, categories: List[str]) -> Dict[str, Any]:
    """Read the harness' per-category score files into gbench's category_accuracy shape."""
    per_cat: Dict[str, Dict[str, Any]] = {}
    # Scores are nested by track, e.g. <model>/agentic/memory/kv/BFCL_v4_memory_kv_score.json;
    # only the selected categories' files are read (see _score_files).
    for cat, path in _score_files(score_root, model_name, categories):
        try:
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        except Exception:
            continue
        acc = first.get("accuracy")
        total = first.get("total_count") or first.get("total") or 0
        correct = first.get("correct_count") or (round(acc * total) if isinstance(acc, (int, float)) else 0)
        if acc is None and total:
            acc = correct / total
        per_cat[cat] = {"correct": int(correct or 0), "total": int(total or 0),
                        "accuracy": round(float(acc or 0.0) * 100.0, 2)}
    return per_cat


def _select_limited_ids(categories: List[str], limit: int,
                        seed: str = "bfcl_v4_agentic",
                        shard: Optional[str] = None) -> Dict[str, List[str]]:
    """Pick `limit` scored questions spread across `categories`, plus their prerequisites.

    `shard` ("I/N", --shard): each category's scored questions, in the harness' dataset order,
    are cut to shard I of N first (round-robin, sampling.shard_select), and `limit` then draws
    within the shard (limit 0: every question of the shard). Prerequisite entries follow the
    questions that depend on them, so two shards may both run one.

    `bfcl generate` has no --limit: it runs the whole category, so `--eval-limit 20` would
    generate all 665 agentic questions. The harness does take an id file (`--run-ids`), so
    gbench builds one.

    Two things the caller should know about the count:

    * Selection is round-robin across categories, and stratified by scenario *within* a
      category, so 20 over 5 categories is 4 each rather than 20 from `memory_kv` - whose
      first 12 ids are all the `customer` scenario out of five.
    * Memory questions `depends_on` `*_prereq` entries that install the memory state. Those
      are pulled in transitively and are NOT scored questions, so the number of generated
      entries exceeds `limit`. Dropping them would score the model on memory it was never
      given.
    """
    from bfcl_eval.utils import load_dataset_entry

    from .sampling import parse_shard, shard_select, stratified_sample

    entries = {c: load_dataset_entry(c) for c in categories}
    by_id = {c: {e["id"]: e for e in rows} for c, rows in entries.items()}
    pools = {c: [e["id"] for e in rows if "_prereq" not in e["id"]] for c, rows in entries.items()}
    if shard:
        pools = {c: shard_select(ids, parse_shard(shard)) for c, ids in pools.items()}
    if not limit or limit <= 0:
        limit = sum(len(ids) for ids in pools.values())

    def _scenario(cat: str, sid: str) -> str:
        """`memory_kv_7-healthcare-3` -> `healthcare`; `web_search_base_7` -> the category."""
        tail = sid[len(cat) + 1:] if sid.startswith(f"{cat}_") else sid
        parts = tail.split("-")
        return parts[1] if len(parts) >= 2 else cat

    # Round-robin the quota across categories, then stratify by scenario inside each.
    quota = {c: 0 for c in categories}
    taken, room = 0, {c: len(pools[c]) for c in categories}
    while taken < limit and any(quota[c] < room[c] for c in categories):
        for c in categories:
            if taken >= limit or quota[c] >= room[c]:
                continue
            quota[c] += 1
            taken += 1

    chosen = {c: stratified_sample(pools[c], quota[c],
                                   key_fn=lambda i, _c=c: _scenario(_c, i),
                                   seed=f"{seed}:{c}") if quota[c] else []
              for c in categories}

    out: Dict[str, List[str]] = {}
    for c, picked in chosen.items():
        if not picked:
            continue
        closure, stack = set(picked), list(picked)
        while stack:
            for dep in (by_id[c].get(stack.pop(), {}).get("depends_on") or []):
                if dep not in closure:
                    closure.add(dep)
                    stack.append(dep)
        out[c] = sorted(closure)
    return out


def scored_entry_counts(categories: List[str]) -> Dict[str, int]:
    """Inside the image: how many scored entries (the `_prereq` entries left out) each category
    has, as `bfcl generate` would run them."""
    from bfcl_eval.utils import load_dataset_entry
    return {c: sum(1 for e in load_dataset_entry(c) if "_prereq" not in e["id"])
            for c in categories}


def _search_watch(log_path: Path) -> Callable[[], Optional[BaseException]]:
    """A watch for the web_search `bfcl generate` (see _watching), reading the web search log.

    A search no grounding model of the cascade could serve has reached the model as BFCL's error
    text, and the web_search scores would measure the search service: generate stops as soon as
    the log shows one, with a SearchUnavailable, which leaves the Web Search setting
    infra_required and the other settings as they are."""
    def watch() -> Optional[BaseException]:
        summary = search_log_summary(_read_search_log(log_path))
        if summary["failed"]:
            return SearchUnavailable(f"{_failure_text(summary)}, so `bfcl generate` was stopped. "
                                     f"{SEARCH_ADVICE}")
        return None
    return watch


def max_tokens_note(fill: Optional[Dict[str, Any]] = None) -> str:
    """The result's `max_output_tokens_note` for the parts that run in the harness. The harness
    sends no max_tokens; with `fill` (agent_max_tokens_fill: gbench's endpoint relay filled the
    run's --max-output-tokens into the requests, bfcl_v4_agentic only) the note says so, else
    each request asked for the room left in the context window, as run_eval_suite writes it for
    a suite that sends none."""
    if fill:
        return (f"max_tokens={fill['value']} (the run's --max-output-tokens) on every `bfcl "
                f"generate` request, filled in by gbench's endpoint relay "
                f"({fill['requests_filled']} request(s)); the official harness sends none")
    knob = get_run_knob("max_output_tokens")
    return ("not sent: the official harness sends no max_tokens, so each request asked for the "
            "room left in the context window (the server's default when the field is absent)"
            + (f"; --max-output-tokens {knob} not applied" if knob else ""))


def agent_max_tokens_fill(relays: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """{value, requests_filled} when gbench's endpoint relay of a `bfcl generate` (the
    `endpoint_relay` records in `relays`, bfcl_report.relays) filled the run's
    --max-output-tokens into at least one request that carried none (swebench_common.EndpointRelay
    `max_tokens_from_run`: on when the suite that runs is in swe_agentic.AGENT_MAX_TOKENS_SUITES,
    i.e. bfcl_v4_agentic, never the bfcl suite; tau_common.sum_relay_fills); None when no relay
    had the fill on or no request went through one (GBENCH_BFCL_REUSE_GENERATIONS=1 with complete
    result files generates nothing). 2026-10-13 review: a relay that filled 0 requests was
    recorded as a cap on every request."""
    from .tau_common import sum_relay_fills
    return sum_relay_fills([((entry or {}).get("endpoint_relay") or {}).get("max_tokens_from_run")
                            for entry in relays or ()])


def _generation_temperature(kwargs: Dict[str, Any], thinking: bool) -> Tuple[float, str]:
    """(temperature, source) for `bfcl generate`.

    - `temperature` with `temperature_source`: already resolved by the caller (the bfcl suite
      hands over its own temperature), used as is;
    - `temperature` alone: an explicit value, below GBENCH_BFCL_V4_AGENTIC_TEMPERATURE and
      --temperature, as for any suite;
    - neither: gbench's rule, 0.0 without --thinking and 1.0 with it, below the same two
      overrides (BFCL's 0.001, BFCL_OFFICIAL_SAMPLING, is only recorded).
    """
    value, source = kwargs.get("temperature"), kwargs.get("temperature_source")
    if value is not None and source:
        own_var = temperature_env_var("bfcl_v4_agentic")
        own = (os.environ.get(own_var) or "").strip()
        if own:
            try:
                differs = abs(float(own) - float(value)) > 1e-9
            except ValueError:
                differs = True
            if differs:
                logger.warning("%s=%s is not used when bfcl runs the agentic track: it generates at "
                               "bfcl's temperature %s (%s); set GBENCH_BFCL_TEMPERATURE or "
                               "--temperature to change it", own_var, own, value, source)
        return float(value), str(source)
    if value is not None:
        return resolve_temperature("bfcl_v4_agentic", value, thinking=thinking)
    temp, src = resolve_official_sampling("bfcl_v4_agentic", BFCL_OFFICIAL_SAMPLING,
                                          thinking=thinking)
    return float(temp), src


def _log_web_infra(entry: Dict[str, Any], eval_name: Optional[str] = None) -> None:
    logger.error("%s: BFCL's Web Search setting is infra_required and not scored (%s): %s; the "
                 "other settings run and are scored", eval_name or _suite().eval_name,
                 entry["when"],
                 str(entry["infra_required"]).rstrip(". "))


def _write_id_file(project_root: Path, ids: Dict[str, List[str]]) -> None:
    """The harness' id file for `bfcl generate --run-ids` (TEST_IDS_TO_GENERATE_PATH): with
    --run-ids the harness takes its categories from this file, not from --test-category."""
    (project_root / "test_case_ids_to_generate.json").write_text(
        json.dumps(ids, indent=2), encoding="utf-8")


def _search_stop_reason(why: str, summary: Dict[str, Any]) -> Optional[str]:
    """Why a `bfcl generate` of the web_search categories that its wall clock stopped cannot be
    scored for the search service's sake (its log shows a failed search), or None when the stop
    was a plain timeout."""
    if not summary["failed"]:
        return None
    return f"{why}: {_failure_text(summary)}. {SEARCH_ADVICE}"


#: How upstream records an entry whose request raised (_llm_response_generation
#: .multi_threaded_inference): its result is this text and the error; its checker scores it wrong.
INFERENCE_ERROR_PREFIX = "Error during inference:"
#: The OpenAI client's text for a request past its timeout (openai.APITimeoutError).
_CLIENT_TIMEOUT_TEXT = "Request timed out"


def _inference_error_entries(result_root: Path, model_name: str, categories: Sequence[str],
                             prereq: bool) -> Dict[str, Dict[str, str]]:
    """{category: {entry id: "failed" | "timed_out" | "over_context"}}: the entries whose request
    raised (upstream's "Error during inference: <error>" result), among the scored entries of
    each category's result file (prereq False) or the memory prerequisite entries, which
    upstream writes to <category>_prereq's own result file (prereq True). `timed_out` is a
    request past the client's timeout, `over_context` a request the server refused for the
    context window (swebench_common.CONTEXT_OVERFLOW_RE), `failed` any other (the endpoint did
    not answer, an HTTP error). Categories without one are left out."""
    from .swebench_common import CONTEXT_OVERFLOW_RE
    wanted = {f"BFCL_v4_{c}_result.json": c for c in categories}
    if prereq:
        wanted.update({f"BFCL_v4_{c}_prereq_result.json": c for c in categories})
    out: Dict[str, Dict[str, str]] = {}
    for path in _result_files(result_root, model_name, list(categories)):
        cat = wanted.get(path.name)
        if cat is None:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            entry_id = str(row.get("id") or "")
            if ("_prereq" in entry_id) != prereq:
                continue
            result = row.get("result")
            if not (isinstance(result, str) and result.startswith(INFERENCE_ERROR_PREFIX)):
                continue
            # (an entry in both files is counted once)
            out.setdefault(cat, {})[entry_id] = (
                "timed_out" if _CLIENT_TIMEOUT_TEXT in result
                else "over_context" if CONTEXT_OVERFLOW_RE.search(result) else "failed")
    return out


def _inference_error_counts(result_root: Path, model_name: str, categories: Sequence[str],
                            prereq: bool) -> Dict[str, Dict[str, int]]:
    """_inference_error_entries per category as {"failed", "timed_out", "over_context"} counts."""
    out: Dict[str, Dict[str, int]] = {}
    for cat, entries in _inference_error_entries(result_root, model_name, categories,
                                                 prereq).items():
        counts = out.setdefault(cat, {"failed": 0, "timed_out": 0, "over_context": 0})
        for kind in entries.values():
            counts[kind] += 1
    return out


def inference_errors(result_root: Path, model_name: str,
                     categories: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """{category: {"failed", "timed_out", "over_context"}} over this model's result files for
    `categories`: the scored entries (the `_prereq` entries left out) whose request raised,
    which upstream records as "Error during inference: <error>" and its checker scores wrong.
    gbench leaves them out of the score as harness errors (harness_error_entries). `timed_out`
    is a request past the client's timeout, `over_context` a request the server refused for the
    context window (swebench_common.CONTEXT_OVERFLOW_RE), `failed` any other (the endpoint did
    not answer, an HTTP error). Categories without one are left out."""
    return _inference_error_counts(result_root, model_name, categories, prereq=False)


def prereq_inference_errors(result_root: Path, model_name: str,
                            categories: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """inference_errors' counts over the memory prerequisite entries of `categories` (the
    `*_prereq` entries, in <category>_prereq's result file as upstream writes them): the
    conversations that build a scenario's memory before its questions run. Upstream records a
    request that raised as "Error during inference: <error>" and goes on; the entry never
    flushes its memory snapshot (BaseHandler.inference_multi_turn_FC / _prompting flush it only
    after the whole conversation), so the scenario's questions run against a memory that lacks
    what it was to write (MemoryAPI._prepare_snapshot loads the last snapshot written, or none).
    These entries are not scored: they are not failed_requests, and each one is a harness
    error of the run (prereq_failure_reason) whose scenario's scored entries are left out of the
    score (harness_error_entries)."""
    return _inference_error_counts(result_root, model_name,
                                   [c for c in categories if c in MEMORY_CATEGORIES],
                                   prereq=True)


def prereq_failure_reason(errors: Dict[str, Dict[str, int]]) -> str:
    """The named reason for `errors` (prereq_inference_errors, non-empty)."""
    n = sum(sum(v.values()) for v in errors.values())
    per_cat = ", ".join(f"{c}: {sum(v.values())}" for c, v in sorted(errors.items()))
    entries = "entry failed its" if n == 1 else "entries failed their"
    return (f"{n} memory prerequisite {entries} request ({per_cat}; "
            "bfcl_report.prereq_inference_errors); the memory entries of those scenarios ran "
            "against an incomplete memory and are left out of the score (harness errors)")


#: The harness-error kind of a scored entry left out of the score, by its inference_errors kind
#: (an entry whose own request raised: base's ENDPOINT_FAILURE_KINDS).
INFERENCE_ERROR_KINDS = {"failed": ENDPOINT_FAILURE_KINDS["FAILED"],
                         "timed_out": ENDPOINT_FAILURE_KINDS["TIMEOUT"],
                         "over_context": ENDPOINT_FAILURE_KINDS["OVER_CONTEXT"]}
#: A scored entry a stopped `bfcl generate` did not finish.
UNFINISHED_KIND = "unfinished"
#: A scored memory entry of a scenario one of whose prerequisite entries' request raised.
PREREQ_FAILED_KIND = "memory_prereq_failed"


def memory_scenario(entry_id: str) -> Optional[str]:
    """The memory scenario of an entry id: `memory_kv_7-healthcare-3` and
    `memory_kv_prereq_10-healthcare-0` -> `healthcare` (upstream's ids,
    utils.process_memory_test_case); None for an id of another shape."""
    parts = str(entry_id).split("-")
    return parts[-2] if len(parts) >= 3 and parts[-2] else None


def _scored_entry_ids(result_root: Path, model_name: str,
                      categories: Sequence[str]) -> Dict[str, List[str]]:
    """{category: the scored entry ids of its result file}: the entries `bfcl evaluate` scores
    (its runner scores every row of the category's result file; the `_prereq` entries are in
    their own file and never scored)."""
    wanted = {f"BFCL_v4_{c}_result.json": c for c in categories}
    out: Dict[str, List[str]] = {}
    for path in _result_files(result_root, model_name, list(categories)):
        cat = wanted.get(path.name)
        if cat is None:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            entry_id = str(row.get("id") or "") if isinstance(row, dict) else ""
            if entry_id and "_prereq" not in entry_id:
                out.setdefault(cat, []).append(entry_id)
    return out


def _score_file_wrong_ids(score_root: Path, model_name: str,
                          categories: Sequence[str]) -> Dict[str, set]:
    """{category: the entry ids its score file lists}: upstream's score file holds a header line
    and one line per entry its checker scored wrong (agentic_runner keeps only the invalid
    entries), so an entry absent from it was scored correct."""
    out: Dict[str, set] = {}
    for cat, path in _score_files(score_root, model_name, list(categories)):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[1:]
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("id") is not None and not row.get("valid"):
                out.setdefault(cat, set()).add(str(row["id"]))
    return out


def harness_error_entries(result_root: Path, score_root: Path, model_name: str,
                          categories: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per category, the scored entries that are harness errors, left out of the score (never
    the model's outcome): {"entries": {entry id: kind}, "correct": how many of them the checker
    scored correct}. A kind is

    * endpoint_failure / endpoint_timeout / over_context (INFERENCE_ERROR_KINDS): the entry's own
      request raised (upstream's "Error during inference", which its checker always scores
      wrong: the result is not a list);
    * memory_prereq_failed (PREREQ_FAILED_KIND): a memory entry of a scenario one of whose
      prerequisite entries' request raised. Upstream gives every scored entry of a scenario
      depends_on = all of that scenario's prerequisites (utils.process_memory_test_case), and a
      failed prerequisite never saves its memory snapshot, so the entry ran against a memory the
      harness never finished building; which of the scenario's questions needed the missing
      conversation cannot be told, so the scenario's entries are all left out.

    `correct` comes from the score file, which lists every entry the checker scored wrong."""
    own = _inference_error_entries(result_root, model_name, categories, prereq=False)
    prereq = _inference_error_entries(result_root, model_name,
                                      [c for c in categories if c in MEMORY_CATEGORIES],
                                      prereq=True)
    broken = {c: {memory_scenario(i) for i in ids} - {None} for c, ids in prereq.items()}
    scored_ids = _scored_entry_ids(result_root, model_name, categories) if broken else {}
    wrong = _score_file_wrong_ids(score_root, model_name, categories) if broken else {}
    out: Dict[str, Dict[str, Any]] = {}
    for cat in categories:
        entries = {i: INFERENCE_ERROR_KINDS[k] for i, k in (own.get(cat) or {}).items()}
        for entry_id in scored_ids.get(cat) or []:
            if entry_id not in entries and memory_scenario(entry_id) in broken.get(cat, set()):
                entries[entry_id] = PREREQ_FAILED_KIND
        if not entries:
            continue
        correct = sum(1 for i, k in entries.items()
                      if k == PREREQ_FAILED_KIND and i not in wrong.get(cat, set()))
        out[cat] = {"entries": entries, "correct": correct}
    return out


def _leave_out(per_cat: Dict[str, Dict[str, Any]], left_out: Dict[str, Dict[str, Any]],
               eval_name: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """(per_cat with each category's harness-error entries (harness_error_entries) taken out of
    its correct and total, the categories nothing is left of). A category with no entry left in
    the score is left unscored: out of the result's category_accuracy, its summary null, never
    0. Counts that cannot hold (more entries left out than the score file scored) mean the
    result and score files disagree: a hard error, never a guessed score."""
    out: Dict[str, Dict[str, Any]] = {}
    unscored: List[str] = []
    for cat, row in per_cat.items():
        gone = left_out.get(cat) or {"entries": {}, "correct": 0}
        total = int(row.get("total") or 0) - len(gone["entries"])
        correct = int(row.get("correct") or 0) - int(gone["correct"])
        if total < 0 or correct < 0 or correct > total:
            raise RuntimeError(
                f"{eval_name}: the result file and the score file of {cat} disagree "
                f"({row.get('correct')}/{row.get('total')} scored, {len(gone['entries'])} "
                f"entries to leave out of it, {gone['correct']} of them correct); the score "
                "cannot be computed")
        if total == 0:
            unscored.append(cat)
            continue
        out[cat] = {**row, "correct": correct, "total": total,
                    "accuracy": round(100.0 * correct / total, 2)}
    return out, unscored


#: The trace status of an entry left out as INFERENCE_ERROR_KINDS kind (base's statuses).
_KIND_STATUS = {v: k for k, v in ENDPOINT_FAILURE_KINDS.items()}


def _mark_left_out(traces: List[Dict[str, Any]],
                   left_out: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The score-file traces with each harness-error entry (harness_error_entries) marked
    `scoring_excluded` with its `harness_error` ({"kind", "detail"}); an entry whose own request
    raised also gets base's status for it (FAILED, TIMEOUT, OVER_CONTEXT)."""
    for t in traces:
        kind = ((left_out.get(str(t.get("category"))) or {}).get("entries") or {}).get(
            str(t.get("question_id")))
        if kind is None:
            continue
        t["scoring_excluded"] = True
        t["harness_error"] = {"kind": kind, "detail": (
            "a prerequisite entry of its memory scenario failed its request, so it ran against "
            "an incomplete memory" if kind == PREREQ_FAILED_KIND
            else "its request failed (upstream's \"Error during inference\")")}
        if kind in _KIND_STATUS:
            t["status"] = _KIND_STATUS[kind]
    return traces


def endpoint_failures(records: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """What swebench_common.EndpointRelay counted as endpoint failures over `records` (its
    record()s; None for a part that reached the endpoint without the relay), in
    swebench_common.relay_result_reasons' shape: {"upstream_errors": {kind: n},
    "requests_refused": n}; None when there were none. A 400 that names the context window is
    not among them (the relay counts it in context_length_rejections, not as an endpoint
    failure; the entry it ends is an over_context inference error, left out of the score)."""
    errors: Dict[str, int] = {}
    refused = 0
    for rec in records:
        if not rec:
            continue
        for kind, n in (rec.get("upstream_errors") or {}).items():
            errors[str(kind)] = errors.get(str(kind), 0) + int(n or 0)
        refused += int(rec.get("requests_refused") or 0)
    errors = {k: v for k, v in sorted(errors.items()) if v}
    if not errors and not refused:
        return None
    return {"upstream_errors": errors, "requests_refused": refused}


def endpoint_failure_reason(failures: Dict[str, Any], where: str) -> str:
    """The named non-comparability reason for `failures` (endpoint_failures) during `where`, in
    swebench_common.relay_result_reasons' words. They are harness errors, never the model's
    outcome; the client retries some of them, and an entry whose request still failed is an
    inference error (failed_requests, left out of the score)."""
    errors = failures.get("upstream_errors") or {}
    failed = sum(int(v) for v in errors.values())
    refused = int(failures.get("requests_refused") or 0)
    parts = []
    if failed:
        parts.append(f"the served endpoint failed {failed} relayed model request(s) ("
                     + ", ".join(f"{k}: {v}" for k, v in sorted(errors.items())) + ")")
    if refused:
        parts.append(f"the relay refused {refused} request(s) to a path it does not forward")
    return f"harness errors during {where}: " + "; ".join(parts)


def _empty_shard_result(model_name: str, enable_thinking: bool, shard: str,
                        categories: List[str]) -> Dict[str, Any]:
    """The result of a shard that holds no entry of any selected category: nothing was
    generated or scored on this node, so there is no score (accuracy None), never an error and
    never 0%."""
    reason = f"empty shard {shard}: no items on this node (categories {', '.join(categories)})"
    result: Dict[str, Any] = {
        "benchmark_type": "eval", "eval_name": _suite().eval_name, "model_name": model_name,
        "mode": "bfcl_eval_harness", "thinking": enable_thinking, "total_questions": 0,
        "correct_answers": 0, "effective_n": 0, "accuracy": None,
        "accuracy_withheld_reason": reason, "status": "success", "status_reason": reason,
        "empty_shard": True, "category_accuracy": {}, "settings": {}, "sample_traces": [],
        "bfcl_report": {"categories_run": [], "shard": shard}}
    record_shard(result, shard)
    return add_comparability_reason(result, reason)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one bfcl_v4_agentic run (gbench --merge): each
    category's correct / total over the entries in the score summed over the shards (the shards
    hold disjoint entries; prerequisite entries are never scored), then BFCL's Agentic
    aggregation (agentic_scores) over those counts, exactly as one run computes it; the other
    counts are summed and the shard comparability reasons dropped."""
    from .base import SHARD_REASON
    per_cat: Dict[str, Dict[str, Any]] = {}
    unscored_any: set = set()
    for r in results:
        unscored_any.update(r.get("categories_unscored") or [])
        for cat, row in (r.get("category_accuracy") or {}).items():
            agg = per_cat.setdefault(cat, {"correct": 0, "total": 0})
            agg["correct"] += int((row or {}).get("correct") or 0)
            agg["total"] += int((row or {}).get("total") or 0)
    for row in per_cat.values():
        row["accuracy"] = round(100.0 * row["correct"] / row["total"], 2) if row["total"] else None
    unscored = sorted(c for c in unscored_any if not (per_cat.get(c) or {}).get("total"))
    per_cat = {c: row for c, row in per_cat.items() if row["total"]}
    scores = agentic_scores(per_cat, unscored=unscored)
    # An empty shard (no entry on its node) sets no field for the union (as
    # merge.generic_merge); 2026-10-13 review: shard 1 empty left the merge without its records.
    live = [r for r in results if not r.get("empty_shard")] or list(results)
    merged = dict(live[0])
    merged.update(scores)
    merged["category_accuracy"] = dict(sorted(per_cat.items()))
    merged["categories_unscored"] = unscored
    total = sum(row["total"] for row in per_cat.values())
    correct = sum(row["correct"] for row in per_cat.values())
    merged["accuracy_pooled"] = round(100.0 * correct / total, 2) if total else None
    merged["overall_accuracy_agentic_bfcl_missing_as_zero"] = (bfcl_zero_filled_overall(per_cat)
                                                               if per_cat else None)
    # BFCL's Agentic score when the union covers every category, else (as one run reports a
    # subset) the pooled accuracy over the entries in the score
    merged["accuracy"] = (scores["overall_accuracy_agentic"]
                          if scores["overall_accuracy_agentic"] is not None
                          else merged["accuracy_pooled"])
    for setting, setting_cats in AGENTIC_SETTINGS:
        entry = (merged.get("settings") or {}).get(setting)
        if isinstance(entry, dict) and "summary" in entry:
            merged.setdefault("settings", {})[setting] = dict(
                entry, summary=scores["summary_memory" if setting == "memory"
                                      else "summary_web_search"])
    merged["effective_n"], merged["correct_answers"] = total, correct
    for field in ("total_questions", "failed_requests", "timed_out_requests",
                  "over_context_prompts", "memory_prereq_failed_requests", "scoring_excluded"):
        merged[field] = sum(int(r.get(field) or 0) for r in results)
    merged["sample_traces"] = [t for r in results for t in (r.get("sample_traces") or [])]
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if reason and not reason.startswith(shard_prefix) \
                    and not reason.startswith("empty shard ") and reason not in kept:
                kept.append(reason)
    # the `bfcl generate` requests gbench's endpoint relay capped, summed over the shards (each
    # shard's reason names its own count)
    from .tau_common import merge_relay_fill_reasons
    relays = [dict(e, shard=r.get("shard")) for r in results
              for e in ((r.get("bfcl_report") or {}).get("relays") or []) if isinstance(e, dict)]
    if relays:      # every shard's relay records (one per `bfcl generate`), not the first's
        merged["bfcl_report"] = dict(merged.get("bfcl_report") or {}, relays=relays)
    fill = agent_max_tokens_fill(relays)
    kept = merge_relay_fill_reasons(merged, kept, fill)
    if fill:
        official = dict(merged.get("official_sampling") or {})
        official["agent_max_tokens_from_run"] = dict(
            official.get("agent_max_tokens_from_run") or {}, max_tokens_sent=fill["value"],
            requests_filled=fill["requests_filled"])
        merged["official_sampling"] = official
        merged["max_output_tokens_note"] = max_tokens_note(fill)
    merged["leaderboard_comparable"] = not kept
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    merged["status"] = ("completed_with_errors"
                        if any(r.get("status") == "completed_with_errors" for r in results)
                        else "success")
    for k in ("accuracy_withheld_reason", "status_reason"):
        merged.pop(k, None)
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = ("no entry of any shard is in the score, so neither "
                                              "BFCL's Agentic score nor the pooled accuracy has a "
                                              "value")
    return merged


def run_bfcl_v4_agentic(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run the BFCL v4 agentic track via the canonical `bfcl-eval` harness.

    One `bfcl generate` and one `bfcl evaluate` over the selected categories. Its two settings,
    Memory and Web Search (AGENTIC_SETTINGS), have their own prerequisites. Web Search is served
    by Gemini Google Search grounding: without a Gemini key, with one the Gemini API rejects
    (before generation), or when a search the grounding cascade could not serve reached the model
    (during generation), that setting is infra_required in its own entry (`settings.web_search`,
    `settings_infra_required`) while Memory runs and is scored. Found before generation, the
    web_search categories are not generated; found during it, generate is stopped and the memory
    entries it had not finished are generated by the harness' own resume (`bfcl generate`
    without --allow-overwrite skips the entries already in the result files). With no memory
    category selected it is a hard error (SearchUnavailable). A run that scored the Web Search
    setting is not leaderboard-comparable (SEARCH_SUBSTITUTION_REASON).

    No harness container holds a key: `bfcl generate` reaches the endpoint through gbench's
    endpoint relay on every run, keyless ones too (generate_always_relayed), and the web searches go through gbench's
    Gemini relay (_run_cli, bridge_services); their records are in `bfcl_report.relays`. The
    harness child runs under the bfcl suite's execution guard (_install_execution_guard); the
    calls it refused are counted in `bfcl_report.execution_guard_refusals` and make the result
    not leaderboard-comparable. Scored entries whose request failed, entries a stopped generate
    did not finish and the memory entries of a scenario whose prerequisite entry's request failed
    (prereq_inference_errors, `memory_prereq_failed_requests`) are harness errors, left out of
    the score (harness_error_entries, `scoring_excluded`, `request_harness_errors`): the result
    and its setting completed_with_errors, not leaderboard-comparable; a category nothing is left
    of is unscored, and a run nothing is left of withholds its score. The prerequisite and
    network checks name the suite that runs (suite_context: bfcl when it runs this track as its
    agentic part)."""
    ctx = _suite()
    ok, reason, categories = check_bfcl_prerequisites(base_url)
    if not ok:
        raise infra_required(ctx.eval_name, reason, ctx.docs_url)
    categories, subset = select_categories(categories, kwargs.get("eval_categories"))
    # --eval-limit and --shard: restrict to an id file rather than running all 665 questions.
    # The draw comes first, so a setting none of whose entries was drawn needs no prerequisite
    # (no Gemini key check, no relay) and is not reported as run.
    limit = kwargs.get("limit")
    shard_spec = current_shard()           # a malformed GBENCH_SHARD raises here
    # This suite's own run takes its shard of every category. bfcl's agentic part (suite_context
    # "bfcl") is run whole in bfcl's shard 1 by bfcl itself, so it is not sharded again here.
    shard_label = (shard_spec.label if shard_spec and shard_spec.partial
                   and ctx.eval_name == "bfcl_v4_agentic" else None)
    limited_ids: Dict[str, List[str]] = {}
    if (limit and int(limit) > 0) or shard_label:
        what = " and ".join(p for p in (f"--eval-limit {limit}" if limit else "",
                                        f"--shard {shard_label}" if shard_label else "") if p)
        try:
            limited_ids = run_in_image(ctx.eval_name, "bfcl_v4_agentic", "_select_limited_ids",
                                       {"categories": list(categories),
                                        "limit": int(limit or 0), "shard": shard_label},
                                       timeout=600)
        except Exception as e:
            # Running the whole track instead would silently ignore the operator's limit/shard.
            raise RuntimeError(f"{ctx.eval_name}: could not build the id file for "
                               f"{what} ({e}) (see {ctx.docs_url})") from e
        if not limited_ids and shard_label:
            # No entry of any category falls in this shard: nothing to generate on this node.
            return _empty_shard_result(model_name, enable_thinking, shard_label, categories)
    # The categories with an entry to run (all of the selected ones, unless an id file limits
    # them); the others were not drawn and are named as such.
    drawn = [c for c in categories if not limited_ids or limited_ids.get(c)]
    not_drawn = [c for c in categories if c not in drawn]
    # The planned workload: --eval-limit N is N scored questions over the categories in total
    # (_select_limited_ids); the memory prerequisite entries they depend on are not scored
    # questions. Logged once, by the suite that runs (bfcl logs its own plan for its parts).
    limited = bool(limit and int(limit) > 0)
    scored_ids = {c: [i for i in limited_ids.get(c, []) if "_prereq" not in i]
                  for c in categories} if limited_ids else {}
    n_prereq = sum(len(ids) for ids in limited_ids.values()) - sum(
        len(ids) for ids in scored_ids.values())
    plan_args = dict(
        per_setting=({c: len(scored_ids[c]) for c in categories} if limited_ids
                     else {c: None for c in categories}),
        attempts=1, setting_word="category",
        note=(f"plus {n_prereq} memory prerequisite entr{'y' if n_prereq == 1 else 'ies'}"
              if n_prereq else ""))
    if ctx.eval_name == "bfcl_v4_agentic":
        limit_plan = eval_limit_plan(ctx.eval_name, limit if limited else None,
                                     sum(len(v) for v in scored_ids.values())
                                     if limited_ids else None, **plan_args)
    else:
        limit_plan = None
    mem_cats = [c for c in drawn if c in MEMORY_CATEGORIES]
    web_cats = [c for c in drawn if c in WEB_SEARCH_CATEGORIES]

    backend: Optional[str] = None
    # The Web Search setting's entry once its prerequisite is found missing (before or during
    # generation); the web_search categories are then not scored.
    web_infra: Optional[Dict[str, Any]] = None
    if web_cats:
        backend, why = search_backend()
        if backend is None:
            if not mem_cats:
                raise SearchUnavailable(f"{why}. The web_search categories cannot run without it.",
                                        "before generation")
            if ctx.eval_name == "bfcl_v4_agentic":
                # This suite's headline is BFCL's Agentic score, which weighs the Web Search
                # setting: a missing prerequisite is a hard error, never a Memory-only number
                # reported as the suite's score. The Memory setting runs alone when selected.
                raise infra_required(
                    ctx.eval_name, f"{why}. BFCL's Web Search setting ({', '.join(web_cats)}) "
                    f"needs it ({SEARCH_PREREQUISITE}), and the suite's score weighs that "
                    "setting; to run the Memory setting alone, select its categories: "
                    f"--eval-categories {','.join(mem_cats)}", ctx.docs_url)
            # bfcl's agentic part (suite_context "bfcl") reports the setting as its own
            # infra_required entry and runs Memory.
            web_infra = web_search_infra_required(web_cats, str(why), "before generation")
            _log_web_infra(web_infra)
    project_root = harness_project_root().resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    # The harness registers the served model under its own id, inside the image (_CHILD_BOOTSTRAP).
    model_key = model_name

    env = dict(os.environ)
    # The endpoint's key (VLLM_API_KEY, else OPENAI_API_KEY), under the names the harness reads;
    # _run_cli replaces it with a placeholder in every container (the endpoint relay adds it).
    key = endpoint_key() or "EMPTY"
    env.update({
        "BFCL_PROJECT_ROOT": str(project_root),
        "REMOTE_OPENAI_BASE_URL": base_url,
        "REMOTE_OPENAI_API_KEY": key,
        # OpenAICompletionsHandler builds its client from these two:
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": key,
    })
    env.setdefault("REMOTE_OPENAI_TOKENIZER_PATH", model_name)
    # Plumb --thinking through the delegated bfcl-eval harness. The harness builds its own
    # requests (OpenAICompletionsHandler._query_FC), so gbench's base.py chat_template_kwargs
    # injection never runs here. Pass the intent via env to the harness container, where the
    # gbench handler subclass injects extra_body={"chat_template_kwargs":{"enable_thinking"}}
    # - the same native-reasoning toggle every other suite uses.
    if enable_thinking:
        env["GBENCH_BFCL_ENABLE_THINKING"] = "1"
    request_timeout = agentic_request_timeout()
    env[REQUEST_TIMEOUT_ENV] = str(request_timeout)
    env.pop(SEARCH_LOG_ENV, None)
    # The execution guard in the harness child records each call it refuses here.
    guard_log = project_root / GUARD_LOG_NAME
    guard_log.write_text("", encoding="utf-8")
    env[GUARD_LOG_ENV] = str(guard_log)
    # The records of gbench's relays (endpoint key, Gemini searches) of each `bfcl generate`.
    relays: List[Dict[str, Any]] = []

    def collecting():
        return suite_context(ctx.eval_name, ctx.docs_url, relays=relays)
    # Every credential of the run is masked in what the harness writes, whichever job saw it.
    secrets = secret_values(env)
    # Only the web_search `bfcl generate` searches: every other job runs without the Gemini key.
    plain_env = {k: v for k, v in env.items() if k not in GEMINI_KEY_NAMES}
    web_env: Optional[Dict[str, str]] = None
    # Web search: the child serves it with Gemini grounding and records every search here (see
    # the module note). It gets the keys as gemini_api_keys reads them (GEMINI_API_KEYS wins):
    # the whole list, and the first key as GEMINI_API_KEY, which search_tool checks.
    search_log: Optional[Path] = None
    if backend:
        keys = gemini_api_keys()
        search_log = project_root / SEARCH_LOG_NAME
        search_log.write_text("", encoding="utf-8")
        web_env = dict(env)
        if keys:
            web_env.update(GEMINI_API_KEYS=",".join(keys), GEMINI_API_KEY=keys[0])
        web_env[SEARCH_LOG_ENV] = str(search_log)

    threads = str(max(1, int(concurrency)))
    timeout = part_timeout()      # None (the default): no wall clock, as upstream

    # `bfcl generate` defaults to --allow-overwrite=False, which does NOT mean "skip the
    # run": it loads whatever is already in <BFCL_PROJECT_ROOT>/result/, subtracts those
    # ids from the work list, and generates nothing when the file is complete
    # (_llm_response_generation.py). The harness then re-scores those stale generations and
    # reports them as this run's result. That cache is BFCL's own, keyed on its project root,
    # so gbench's --skip-existing has no bearing on it.
    #
    # An eval run should measure the model as it is now, so regenerate by default. Set
    # GBENCH_BFCL_REUSE_GENERATIONS=1 (legacy alias BFCL_REUSE_GENERATIONS) to deliberately
    # re-score existing generations instead.
    reuse = suite_env("GBENCH_BFCL_REUSE_GENERATIONS", "BFCL_REUSE_GENERATIONS") == "1"
    score_root = project_root / "score"
    bfcl_temp, bfcl_temp_src = _generation_temperature(kwargs, enable_thinking)
    official = BFCL_OFFICIAL_SAMPLING.record(bfcl_temp, bfcl_temp_src)
    gen_options = ["--num-threads", threads, "--skip-server-setup",
                   "--temperature", str(bfcl_temp)]
    evaluate_options: List[str] = []

    if limited_ids:
        what = " and ".join(p for p in (f"--eval-limit {limit}" if limit else "",
                                        f"--shard {shard_label}" if shard_label else "") if p)
        if limited_ids:
            # Separate result/score dirs: --run-ids updates result files in place, so
            # sharing the full-run directory would leave earlier questions in the file and
            # the harness would score those too. These paths are resolved relative to
            # BFCL_PROJECT_ROOT by the harness CLI.
            tag = "_".join(p for p in (f"limit{int(limit)}" if limit else "",
                                       "shard{}of{}".format(*shard_label.split("/"))
                                       if shard_label else "") if p)
            rdir, sdir = f"result_{tag}", f"score_{tag}"
            gen_options += ["--run-ids", "--result-dir", rdir]
            # --partial-eval: the categories are only partly present by construction, and
            # without it the harness raises on the missing ids.
            evaluate_options += ["--result-dir", rdir, "--score-dir", sdir, "--partial-eval"]
            score_root = project_root / sdir
            n_prereq = sum(1 for ids in limited_ids.values() for i in ids if "_prereq" in i)
            logger.info("%s: %s -> %d scored questions across "
                        "%d categories (+%d prerequisite entries) in %s", ctx.eval_name,
                        what, sum(len(v) for v in limited_ids.values()) - n_prereq,
                        len(limited_ids), n_prereq, rdir)
    if not reuse:
        gen_options.append("--allow-overwrite")
    else:
        logger.warning("%s: GBENCH_BFCL_REUSE_GENERATIONS=1 - re-scoring existing "
                       "generations in %s; the result reflects whenever those were "
                       "produced, not this run.", ctx.eval_name,
                       env.get("BFCL_PROJECT_ROOT", "the project root"))

    result_dir = project_root / (rdir if limited_ids else "result")
    if limited_ids and not reuse:
        # --run-ids updates the limit's result files in place; this model's files for the chosen
        # categories are removed first (as upstream removes them for a full run with
        # --allow-overwrite), so only entries generated by THIS run can be scored.
        _clear_results(result_dir, model_key, categories)
    # The categories this run generates: a small --eval-limit may leave some without any ids,
    # and a Web Search setting whose prerequisite is missing is not generated.
    runnable = [c for c in categories if not (web_infra and c in WEB_SEARCH_CATEGORIES)]
    expected = [c for c in runnable if c in limited_ids] if limited_ids else list(runnable)
    # The scored entries each category is to generate: what a stopped generate counts its
    # unfinished entries against. Only a run with a wall clock can be stopped; a full run's
    # counts are read from the image before generating, so no harness job runs after a stop.
    planned: Optional[Dict[str, int]] = None
    if limited_ids:
        planned = {c: sum(1 for i in limited_ids[c] if "_prereq" not in i) for c in expected}
    elif timeout is not None and expected:
        planned = {c: int(n) for c, n in run_in_image(
            ctx.eval_name, "bfcl_v4_agentic", "scored_entry_counts",
            {"categories": list(expected)}, timeout=600).items()}

    gen_stopped: Optional[HarnessTimeout] = None
    redacted_files = 0
    # The memory entries a stopped generate had not finished, when the Web Search setting's
    # prerequisite was lost during it: generated afterwards by the harness' resume.
    resume_memory = False
    if limited_ids:
        _write_id_file(project_root, {c: limited_ids[c] for c in expected})
    generate_cmd = ["generate", "--model", model_key, "--test-category", ",".join(runnable),
                    *gen_options]
    watch = _search_watch(search_log) if search_log is not None else None
    rc, out = 0, ""
    try:
        with _watching(watch), collecting():
            rc, out = _run_cli(generate_cmd, web_env if backend else plain_env, timeout,
                               model=model_key, backend=backend)
    except SearchUnavailable as e:
        # The watch stopped generate: the Web Search setting is not scored; the memory entries
        # stand and the unfinished ones are generated below.
        web_infra = web_search_infra_required(web_cats, e.reason, "during generation")
        resume_memory = True
        _log_web_infra(web_infra)
    except subprocess.TimeoutExpired as e:
        limit_s = timeout if timeout is not None else e.timeout
        why = f"`bfcl generate` did not finish within {PART_TIMEOUT_KNOB}={_seconds(limit_s)}"
        stop_reason = None
        if search_log is not None:
            stop_reason = _search_stop_reason(why, search_log_summary(_read_search_log(search_log)))
        if stop_reason is not None:
            # Searches failed: the Web Search setting is not scored, and the memory entries are
            # finished by the resume below.
            web_infra = web_search_infra_required(web_cats, stop_reason, "during generation")
            resume_memory = True
            _log_web_infra(web_infra)
        else:
            # The entries the harness finished are in its result files (it writes each one as
            # it completes): they are scored, and the rest are harness errors left out of the
            # score (harness_errors).
            gen_stopped = HarnessTimeout(ctx.eval_name, "`bfcl generate`", limit_s)
            logger.warning("%s; scoring the entries it finished, the rest are left out of the "
                           "score (harness errors)",
                           gen_stopped)
    finally:
        redacted_files += _redact_tree(result_dir, secrets)
    if rc != 0:
        out = redact_text(out, secrets)
        logger.error("bfcl generate failed (rc=%s):\n%s", rc, out[-1500:])
        raise RuntimeError(
            f"{ctx.eval_name}: `bfcl generate` failed (rc={rc}) and must not be skipped: "
            f"{out.strip()[-300:]} (see {ctx.docs_url})")

    mem_expected = [c for c in expected if c in MEMORY_CATEGORIES]
    if resume_memory and mem_expected:
        # upstream's resume: without --allow-overwrite, collect_test_cases loads the entries
        # already in the result files and generates the others (clean_up_memory_prereq_entries
        # drops finished prerequisites from depends_on). A line the stop cut off is removed first.
        _drop_cut_lines(result_dir, model_key, mem_expected)
        if limited_ids:
            _write_id_file(project_root, {c: limited_ids[c] for c in mem_expected})
        resume_cmd = ["generate", "--model", model_key, "--test-category",
                      ",".join(mem_expected), *[o for o in gen_options if o != "--allow-overwrite"]]
        rc, out = 0, ""
        try:
            with collecting():
                rc, out = _run_cli(resume_cmd, plain_env, timeout, model=model_key, backend=None)
        except subprocess.TimeoutExpired as e:
            gen_stopped = HarnessTimeout(
                ctx.eval_name, "`bfcl generate` (the memory categories, resumed)",
                timeout if timeout is not None else e.timeout)
            logger.warning("%s; scoring the entries it finished, the rest are left out of the "
                           "score (harness errors)",
                           gen_stopped)
        finally:
            redacted_files += _redact_tree(result_dir, secrets)
        if rc != 0:
            out = redact_text(out, secrets)
            logger.error("bfcl generate (resume) failed (rc=%s):\n%s", rc, out[-1500:])
            raise RuntimeError(
                f"{ctx.eval_name}: `bfcl generate` of the memory categories (resumed after the "
                f"web search stopped) failed (rc={rc}) and must not be skipped: "
                f"{out.strip()[-300:]} (see {ctx.docs_url})")

    # The calls the execution guard refused during generation (upstream would have run them).
    refused = _read_guard_log(guard_log, secrets)
    if refused:
        logger.warning("%s: %d agentic call(s) were refused by the execution guard, e.g. %r",
                       ctx.eval_name, len(refused), refused[0])

    # A search no grounding model could serve reached the model as the harness' error text: the
    # web_search scores would measure the search service, so that setting is not scored.
    search_summary: Optional[Dict[str, Any]] = None
    # The Gemini models that answered the searches (the log's `gemini:<model>` backends).
    gemini_answered: Dict[str, int] = {}
    if search_log is not None:
        search_summary = search_log_summary(_read_search_log(search_log))
        for backend_name, n in search_summary["by_backend"].items():
            if backend_name.startswith("gemini:"):
                gemini_answered[backend_name[len("gemini:"):]] = int(n)
        for gemini_model, n in gemini_answered.items():
            record_gemini_answer(gemini_model, n)
        if web_infra is None and search_summary["failed"]:
            web_infra = web_search_infra_required(
                web_cats, f"{_failure_text(search_summary)}. {SEARCH_ADVICE}", "during generation")
            _log_web_infra(web_infra)
    # Endpoint failures gbench's endpoint relay counted during generation (harness errors).
    endpoint_fail = endpoint_failures([r.get("endpoint_relay") for r in relays])
    if endpoint_fail:
        logger.error("%s: %s", ctx.eval_name,
                     endpoint_failure_reason(endpoint_fail, "`bfcl generate`"))

    # The categories to score: the Web Search setting is left out once its prerequisite is gone.
    scored = [c for c in expected if not (web_infra and c in WEB_SEARCH_CATEGORIES)]
    if planned is not None:
        planned = {c: n for c, n in planned.items() if c in scored}
    if web_infra is not None and not scored:
        # Nothing else of the run is left to report.
        raise SearchUnavailable(web_infra["infra_required"], web_infra["when"])
    evaluated = [c for c in drawn if not (web_infra and c in WEB_SEARCH_CATEGORIES)]
    evaluate_cmd = ["evaluate", "--model", model_key, "--test-category", ",".join(evaluated),
                    *evaluate_options]

    # Only score files written by THIS evaluate may be read: remove the model's earlier files
    # for the selected categories first, and treat a non-zero exit as a failure even when some
    # files were written.
    stale = _clear_stale_scores(score_root, model_key, categories)
    if stale:
        logger.info("%s: removed %d score file(s) left by an earlier run in %s",
                    ctx.eval_name, stale, score_root)
    run_evaluate = True
    if gen_stopped is not None:
        if planned is None:
            # (Unreachable in a run: only a wall clock stops generate, and then the counts were
            # read before generating.)
            planned = {c: int(n) for c, n in run_in_image(
                ctx.eval_name, "bfcl_v4_agentic", "scored_entry_counts",
                {"categories": list(scored)}, timeout=600).items()}
        # A container stopped mid-write can leave a cut-off last line; the entries are otherwise
        # whole. --partial-eval scores the entries present instead of raising on the others.
        # (In an --eval-limit run the harness rewrites each result file whole after every
        # finished entry, so a stop during a rewrite can also drop entries it had finished;
        # those count as unfinished below.)
        _drop_cut_lines(result_dir, model_key, scored)
        if "--partial-eval" not in evaluate_cmd:
            evaluate_cmd.append("--partial-eval")
        # With no scored entry written at all there is nothing to evaluate (and `bfcl evaluate`
        # would fail on the missing score directory): every planned entry is unfinished.
        run_evaluate = any(not p.name.endswith("_prereq_result.json")
                           for p in _result_files(result_dir, model_key, scored))
    rc, out = 0, ""
    try:
        if run_evaluate:
            rc, out = _run_cli(evaluate_cmd, plain_env, timeout, model=model_key, backend=None)
    except subprocess.TimeoutExpired as e:
        # A stopped generate before it is recorded too: none of its entries is scored.
        earlier = ([gen_stopped.record(entries_planned=sum(planned.values()),
                                       entries_unscored=sum(planned.values()))]
                   if gen_stopped is not None and planned is not None else [])
        raise HarnessTimeout(ctx.eval_name, "`bfcl evaluate`",
                             timeout if timeout is not None else e.timeout,
                             earlier=earlier) from None
    finally:
        redacted_files += _redact_tree(score_root, secrets)
    out = redact_text(out, secrets)
    if redacted_files:
        logger.warning("%s: masked a credential in %d harness file(s) under %s",
                       ctx.eval_name, redacted_files, project_root)
    if rc != 0:
        logger.error("bfcl evaluate failed (rc=%s):\n%s", rc, out[-1500:])
        raise RuntimeError(
            f"{ctx.eval_name}: `bfcl evaluate` failed (rc={rc}) and must not be skipped: "
            f"{out.strip()[-300:]} (see {ctx.docs_url})")
    per_cat = _parse_scores(score_root, model_key, evaluated) if run_evaluate else {}
    harness_errors: List[Dict[str, Any]] = []
    unfinished = 0
    unfinished_by_cat: Dict[str, int] = {}
    if gen_stopped is not None and planned is not None:
        # The entries the harness finished are scored; the ones it did not finish were never
        # measured: harness errors, left out of the score and counted as timed out.
        per_cat, unfinished = _count_unfinished(per_cat, {c: int(planned[c]) for c in scored})
        unfinished_by_cat = {c: int(r["unfinished"]) for c, r in per_cat.items()
                             if r.get("unfinished")}
        harness_errors.append(gen_stopped.record(entries_unfinished=unfinished,
                                                 entries_planned=sum(planned.values())))
    elif not per_cat:
        logger.error("bfcl evaluate wrote no score file (rc=0):\n%s", out[-1500:])
        raise RuntimeError(
            f"{ctx.eval_name}: `bfcl evaluate` exited 0 but wrote no score file for "
            f"{', '.join(evaluated)} under {score_root}: {out.strip()[-300:]} "
            f"(see {ctx.docs_url})")

    # Every category that was generated must have been scored: reporting the rest would be a
    # partial result.
    missing = [c for c in scored if c not in per_cat]
    if missing:
        logger.error("bfcl evaluate left categories unscored:\n%s", out[-1500:])
        raise RuntimeError(
            f"{ctx.eval_name}: `bfcl evaluate` produced no score for {', '.join(missing)} and "
            f"a partial result must not be reported (see {ctx.docs_url})")

    # Scored entries whose request raised (upstream's "Error during inference", which its checker
    # scores wrong): failed requests, as in every other suite, beside the entries a stopped
    # generate did not finish.
    inf_errors = inference_errors(result_dir, model_key, scored) if run_evaluate else {}
    # Memory prerequisite entries whose request raised: not scored entries, so not failed
    # requests, but the memory entries of their scenarios ran against an incomplete memory.
    prereq_errors = prereq_inference_errors(result_dir, model_key, scored)
    prereq_failed = sum(sum(v.values()) for v in prereq_errors.values())
    if prereq_failed:
        logger.error("%s: %s", ctx.eval_name, prereq_failure_reason(prereq_errors))
    # Harness errors are never the model's outcome: the scored entries whose request raised and
    # the memory entries of a scenario whose prerequisite request raised are left out of their
    # category's correct and total (as base leaves out a request that got no reply), like the
    # unfinished entries above; a category nothing is left of is left unscored (null, never 0).
    left_out = (harness_error_entries(result_dir, score_root, model_key, scored)
                if run_evaluate else {})
    per_cat, unscored_cats = _leave_out(per_cat, left_out, ctx.eval_name)
    # {category: {harness-error kind: entries left out of the score}}
    left_out_by_cat: Dict[str, Dict[str, int]] = {}
    for cat, gone in left_out.items():
        for kind in gone["entries"].values():
            kinds = left_out_by_cat.setdefault(cat, {})
            kinds[kind] = kinds.get(kind, 0) + 1
    for cat, n in unfinished_by_cat.items():
        left_out_by_cat.setdefault(cat, {})[UNFINISHED_KIND] = n
    left_out_by_cat = {c: dict(sorted(k.items())) for c, k in sorted(left_out_by_cat.items())}
    request_harness_errors: Dict[str, int] = {}
    for kinds in left_out_by_cat.values():
        for kind, n in kinds.items():
            request_harness_errors[kind] = request_harness_errors.get(kind, 0) + n
    request_harness_errors = dict(sorted(request_harness_errors.items()))
    scoring_excluded = sum(request_harness_errors.values())
    prereq_dependents = request_harness_errors.get(PREREQ_FAILED_KIND, 0)
    if scoring_excluded:
        logger.error("%s: %d scored entries are harness errors, left out of the score (%s)",
                     ctx.eval_name, scoring_excluded,
                     ", ".join(f"{k} {v}" for k, v in request_harness_errors.items()))
    # the entries in the score, and the ones the run was to score
    total = sum(c["total"] for c in per_cat.values())
    correct = sum(c["correct"] for c in per_cat.values())
    entries_run = total + scoring_excluded
    over_context = sum(v["over_context"] for v in inf_errors.values())
    failed_requests = unfinished + sum(sum(v.values()) for v in inf_errors.values())
    timed_out_requests = unfinished + sum(v["timed_out"] for v in inf_errors.values())
    withheld: Optional[str] = None
    all_failed: Optional[str] = None
    if total == 0:
        # Nothing is left in the score: no score, not a 0.
        how = (f"{failed_requests} failed requests: {timed_out_requests} timed out or "
               f"unfinished, {over_context} over the context window, the rest failed")
        if not entries_run:
            all_failed = "no scored entry was generated, so nothing measured the model"
        elif prereq_dependents:
            all_failed = (f"every one of the {entries_run} scored entries failed its request, "
                          f"was not finished or ran against an incomplete memory ({how}; "
                          f"{prereq_dependents} memory entries of a scenario whose prerequisite "
                          "request failed), so nothing measured the model")
        else:
            all_failed = (f"every one of the {entries_run} scored entries failed its request or "
                          f"was not finished ({how}), so nothing measured the model")
        withheld = all_failed + ("; " + endpoint_failure_reason(endpoint_fail, "`bfcl generate`")
                                 if endpoint_fail else "")
        logger.error("%s: score withheld: %s", ctx.eval_name, withheld)
    # categories every entry of which was a harness error (null, never 0), when others are scored
    unscored_note = (f"{', '.join(unscored_cats)} left unscored: every one of "
                     f"{'its' if len(unscored_cats) == 1 else 'their'} entries was a harness "
                     "error left out of the score (bfcl_report.harness_errors_left_out)"
                     if unscored_cats and not withheld else None)
    ran_web = [c for c in evaluated if c in WEB_SEARCH_CATEGORIES]
    scores = agentic_scores(per_cat, unscored=unscored_cats)
    if web_infra is not None:
        # BFCL's Agentic score weighs the Web Search setting, which was not scored: it stays
        # null, as the Web Search column does.
        scores["overall_accuracy_agentic"] = None
    if withheld:
        # Nothing measured the model: no score, not a 0 (the runner reports the suite failed).
        scores = {k: None for k in scores}
    # over the entries in the score: an entry left out as a harness error is not a failed answer
    in_score_exclude = {(cat, str(eid)) for cat, gone in left_out.items()
                        for eid in (gone.get("entries") or {})}
    error_types = _score_error_types(score_root, model_key, evaluated, in_score_exclude)
    # Every deviation from BFCL's own run is listed: the web search served by Gemini grounding, a
    # category subset, an --eval-limit subset, re-scored cached generations, a temperature other
    # than BFCL's, a setting whose prerequisite was missing, entries left out of the score as
    # harness errors, or a server whose replies the harness could not read.
    noncanon = []
    if ran_web:
        # The Web Search setting was scored over Gemini Google Search grounding, not BFCL's
        # DuckDuckGo-via-SerpAPI search.
        noncanon.append(SEARCH_SUBSTITUTION_REASON)
    if web_infra is not None:
        noncanon.append(f"BFCL's Web Search setting ({', '.join(web_cats)}) was not scored: its "
                        f"prerequisite ({SEARCH_PREREQUISITE}) was not available "
                        f"{web_infra['when']} (infra_required: {web_infra['infra_required']})")
    if subset:
        noncanon.append(f"--eval-categories subset ({', '.join(categories)}); BFCL's Agentic "
                        f"score counts the missing categories as 0")
    if limited_ids and limit:
        noncanon.append(f"--eval-limit subset ({int(limit)} ids), not the full category set"
                        + (f"; no entry of {', '.join(not_drawn)} was drawn" if not_drawn else ""))
    if reuse:
        noncanon.append("re-scored cached generations (GBENCH_BFCL_REUSE_GENERATIONS=1), not a fresh run")
    if not official["temperature_matches_official"]:
        noncanon.append(f"temperature {bfcl_temp} set by {bfcl_temp_src} (BFCL generates at 0.001)")
    if harness_errors:
        noncanon.append(f"{gen_stopped}: {unfinished} of the entries it was to generate were not "
                        "finished and are left out of the score (harness_errors)")
    if endpoint_fail:
        noncanon.append(endpoint_failure_reason(endpoint_fail, "`bfcl generate`"))
    if prereq_failed:
        noncanon.append(prereq_failure_reason(prereq_errors))
    own_failures = {k: v for k, v in request_harness_errors.items()
                    if k in INFERENCE_ERROR_KINDS.values()}
    if own_failures:
        n_own = sum(own_failures.values())
        noncanon.append(f"{n_own} scored {'entry' if n_own == 1 else 'entries'} whose request "
                        "failed (" + ", ".join(f"{k}: {v}" for k, v in own_failures.items())
                        + "; bfcl_report.inference_errors) left out of the score as harness "
                        "errors (upstream scores them wrong)")
    if unscored_note:
        noncanon.append(unscored_note)
    if refused:
        noncanon.append(f"{len(refused)} agentic calls refused by the execution guard (upstream "
                        "would have run them)")
    no_last = error_types.get("agentic:no_last_message", 0)
    if total and no_last >= total:
        noncanon.append(f"all {total} scored questions failed with agentic:no_last_message: the "
                        "harness found no final reply to grade in any of them (a server that sends "
                        "an empty tool_calls list with a plain reply causes this; see "
                        f"{DOCS_URL})")
    if all_failed:
        noncanon.append(f"score withheld: {all_failed}")
    # the run's --max-output-tokens on every `bfcl generate` request (bfcl_v4_agentic only; the
    # user's decision of 2026-10-11 for agent suites): a deviation, BFCL sends no max_tokens
    token_fill = agent_max_tokens_fill(relays)
    if token_fill:
        from .swe_agentic import agent_max_tokens_record
        from .tau_common import relay_fill_reason
        fill_rec, _why = agent_max_tokens_record(
            token_fill["value"], "BFCL's harness (OpenAICompletionsHandler)",
            "gbench's endpoint relay (max_tokens_from_run: filled into each `bfcl generate` "
            "request that carried none)")
        fill_rec["requests_filled"] = token_fill["requests_filled"]
        official = dict(official, agent_max_tokens_from_run=fill_rec)
        # in the eval runner's words (evals._stamp_agent_max_tokens), so it is stated once
        noncanon.append(relay_fill_reason(token_fill["value"], token_fill["requests_filled"]))
    leaderboard = not noncanon
    settings: Dict[str, Dict[str, Any]] = {}
    for setting, setting_cats in AGENTIC_SETTINGS:
        selected = [c for c in drawn if c in setting_cats]
        if not selected:
            continue
        if setting == "web_search" and web_infra is not None:
            settings[setting] = web_infra
            continue
        rows = {c: per_cat[c] for c in selected if c in per_cat}
        setting_failed = sum(unfinished_by_cat.get(c, 0) for c in selected) + sum(
            sum(inf_errors[c].values()) for c in selected if c in inf_errors)
        setting_prereq_failed = sum(sum(prereq_errors[c].values())
                                    for c in selected if c in prereq_errors)
        setting_excluded = sum(sum(left_out_by_cat[c].values())
                               for c in selected if c in left_out_by_cat)
        settings[setting] = {
            "status": ("completed_with_errors" if harness_errors or endpoint_fail
                       or setting_failed or setting_prereq_failed or setting_excluded
                       else "success"),
            "failed_requests": setting_failed,
            # its memory prerequisite entries whose request raised (not scored entries)
            **({"prereq_failed_requests": setting_prereq_failed} if setting == "memory" else {}),
            "categories": selected,
            # its categories every entry of which was a harness error (not in the summary)
            "categories_unscored": [c for c in selected if c in unscored_cats],
            "summary": scores["summary_memory" if setting == "memory" else "summary_web_search"],
            # the entries it was to score: in the score, and left out of it as harness errors
            "total_questions": sum(r["total"] for r in rows.values()) + setting_excluded,
            "correct_answers": sum(r["correct"] for r in rows.values()),
            "scoring_excluded": setting_excluded}
    accuracy: Optional[float]
    if web_infra is not None:
        accuracy = scores["summary_memory"]
        metric = ("summary_memory (the Memory setting's mean; BFCL's Agentic score, "
                  "overall_accuracy_agentic, is null because the Web Search setting was not "
                  "scored: settings.web_search is infra_required)")
    else:
        accuracy = scores["overall_accuracy_agentic"]
        metric = ("overall_accuracy_agentic (BFCL: unweighted mean of the web-search and "
                  "memory summaries)")
    pooled_rate = round(correct / total * 100.0, 2) if total else None
    # BFCL's own Agentic Overall Acc over the categories scored (a missing category counted as 0)
    zero_filled = bfcl_zero_filled_overall(per_cat) if per_cat and not withheld else None
    if accuracy is None and not withheld:
        # A category was left unscored, or not run (a category subset or an --eval-limit draw):
        # BFCL's means that weigh it are null, and the pooled rate over the entries in the score
        # is the headline instead, as the bfcl suite reports a subset.
        accuracy = pooled_rate
        why = (unscored_note if unscored_note else
               "the run did not cover every agentic category ("
               + ", ".join(c for c in WEB_SEARCH_CATEGORIES + MEMORY_CATEGORIES
                           if c not in per_cat)
               + " not in the score), and BFCL's Agentic score needs all of them "
               "(overall_accuracy_agentic_bfcl_missing_as_zero is BFCL's own value with the "
               "missing categories counted as 0)")
        metric = ("accuracy_pooled (correct / total over the entries in the score; BFCL's "
                  f"means that weigh a category not in the score are null: {why})")
    web_withheld: Optional[str] = None
    if web_infra is not None and ctx.eval_name == "bfcl_v4_agentic" and not withheld:
        # This suite's run lost the Web Search setting's prerequisite during generation (a search
        # the grounding cascade could not serve reached the model): BFCL's Agentic score cannot
        # be computed, and a Memory-only number is not this suite's score. A harness failure,
        # never the model's outcome: the settings keep what was measured, the score is withheld.
        web_withheld = (f"BFCL's Web Search setting ({', '.join(web_cats)}) was not scored: its "
                        f"prerequisite ({SEARCH_PREREQUISITE}) was lost {web_infra['when']} "
                        f"(infra_required: {web_infra['infra_required']}); the suite's score "
                        "weighs that setting, so it is withheld (settings.memory keeps the "
                        "Memory setting's numbers)")
        logger.error("%s: score withheld: %s", ctx.eval_name, web_withheld)
    if withheld:
        accuracy = None
        status = "failed"
    elif web_withheld:
        accuracy = None
        status = "error"
        metric = ("overall_accuracy_agentic (withheld: the Web Search setting was not scored, see "
                  "accuracy_withheld_reason; settings.memory keeps the Memory setting's numbers)")
    elif (harness_errors or web_infra or failed_requests or endpoint_fail or prereq_failed
          or scoring_excluded):
        status = "completed_with_errors"
    else:
        status = "success"
    result = {
        "benchmark_type": "eval",
        "eval_name": "bfcl_v4_agentic",
        "model_name": model_name,
        "mode": "bfcl_eval_harness",
        "thinking": enable_thinking,
        # the scored entries the run was to score: in the score, and left out of it as harness
        # errors (scoring_excluded)
        "total_questions": entries_run,
        "correct_answers": correct,
        # the entries in the score (the headline's denominator)
        "effective_n": total,
        # scored entries whose request raised (inference errors) plus the entries `bfcl
        # generate` did not finish before GBENCH_BFCL_TIMEOUT_S stopped it: harness errors, left
        # out of the score
        "failed_requests": failed_requests,
        "timed_out_requests": timed_out_requests,
        "over_context_prompts": over_context,
        # memory prerequisite entries whose request raised (not scored entries, so not in
        # failed_requests): harness errors, per category in bfcl_report.prereq_inference_errors;
        # the memory entries of their scenarios are left out of the score
        "memory_prereq_failed_requests": prereq_failed,
        # scored entries left out of the score as harness errors, in all and by kind
        # (endpoint_failure, endpoint_timeout, over_context, unfinished, memory_prereq_failed;
        # per category in bfcl_report.harness_errors_left_out)
        "scoring_excluded": scoring_excluded,
        **({"request_harness_errors": request_harness_errors} if request_harness_errors else {}),
        "harness_errors": harness_errors,
        # endpoint failures gbench's endpoint relay counted (harness errors), when there were any
        **({"endpoint_failures": endpoint_fail} if endpoint_fail else {}),
        # BFCL's Agentic score, not the pooled rate: memory has 465 scored questions to web
        # search's 200, so pooling weighs memory 70% where BFCL weighs the two halves equally.
        "accuracy": accuracy,
        **({"accuracy_withheld_reason": withheld} if withheld else {}),
        "metric": metric,
        **scores,
        "accuracy_pooled": None if withheld else pooled_rate,
        # BFCL's own Agentic Overall Acc (a category that did not run counted as 0)
        "overall_accuracy_agentic_bfcl_missing_as_zero": zero_filled,
        # per category over the entries in the score; a category every entry of which was a
        # harness error is not in it (categories_unscored)
        "category_accuracy": per_cat,
        "categories_unscored": unscored_cats,
        # each Agentic setting the run selected: its categories, summary and counts, or its
        # infra_required entry when its prerequisite was missing
        "settings": settings,
        **({"settings_infra_required": ["web_search"]} if web_infra is not None else {}),
        "temperature": bfcl_temp,
        "temperature_source": bfcl_temp_src,
        "official_sampling": official,
        "max_output_tokens_note": max_tokens_note(token_fill),
        "leaderboard_comparable": leaderboard,
        "leaderboard_comparable_reason": ("; ".join(noncanon) if noncanon else None),
        # one trace per scored entry generated (correct ones included), harness errors marked
        "sample_traces": _mark_left_out(
            _parse_score_traces(score_root, model_key, evaluated, result_dir)
            if run_evaluate else [],
            left_out),
        "bfcl_report": {
            "categories_run": evaluated,
            # selected categories an --eval-limit / --shard draw gave no entry
            "categories_not_drawn": not_drawn,
            "categories_subset": subset,
            "search_backend": (f"{SEARCH_BACKEND} (BFCL's web_search tool served by Gemini "
                               f"Google Search grounding, model cascade "
                               f"{', '.join(search_models())}; Gemini key checked before "
                               f"generation, every search recorded during it)" if backend
                               else f"none: {web_infra['infra_required']}" if web_infra
                               else "not needed (no web_search category selected)"),
            # what served BFCL's web_search tool in place of its own search, and how it differs
            "search_substitution": search_substitution() if backend else None,
            # the searches of the run: per grounding model, failed, answered without results
            "web_search_calls": search_summary,
            # per category, the scored entries whose request raised: failed, timed_out,
            # over_context (upstream's "Error during inference", which upstream scores wrong;
            # here left out of the score)
            "inference_errors": inf_errors,
            # per memory category, the prerequisite entries whose request raised (same split);
            # the scenario questions that depend on them ran against an incomplete memory
            "prereq_inference_errors": prereq_errors,
            # per category, the scored entries left out of the score as harness errors, by kind
            "harness_errors_left_out": left_out_by_cat,
            "categories_unscored": unscored_cats,
            "harness": (f"bfcl-eval {BFCL_EVAL_RELEASE} in the local image {bfcl_image()} "
                        f"({BFCL_DOCKERFILE}; files checked against gorilla@{BFCL_EVAL_COMMIT})"),
            # the memory categories generated again by the harness' resume after the web
            # search stopped during generation
            "memory_resumed": bool(resume_memory and mem_expected),
            # each harness request's client timeout (agentic_request_timeout); retries as the SDK's
            "request_timeout_s": request_timeout,
            # calls the execution guard refused in the harness child (first 20 shown)
            "execution_guard_refusals": len(refused),
            "execution_guard_refused_calls": refused[:20],
            # where the keys stay: no harness container holds one
            "credentials": {
                "endpoint_key": (f"added by gbench's endpoint relay on the docker bridge gateway; "
                                 f"the containers hold {ENDPOINT_KEY_PLACEHOLDER}"
                                 if endpoint_key() else
                                 "none (VLLM_API_KEY / OPENAI_API_KEY unset or a placeholder; "
                                 "the containers hold EMPTY)"),
                "gemini_key": ("added by gbench's Gemini relay on the docker bridge gateway; the "
                               "web_search `bfcl generate` holds a per-run placeholder"
                               if backend else "not used"),
                "dotenv": (f"upstream's <project root>/{DOTENV_NAME} is never loaded: a project "
                           "root that holds one is refused before a container starts, and the "
                           "harness child's load_dotenv loads nothing"),
            },
            # the records of those relays, per `bfcl generate` (never a key)
            "relays": relays,
            "error_types": error_types,
            "leaderboard_comparable": leaderboard,
            "web_search_categories_run": ran_web,
            "project_root": str(project_root),
            # Whether the model was actually queried, or a cached generation re-scored.
            "generations": "reused (GBENCH_BFCL_REUSE_GENERATIONS=1)" if reuse else "regenerated",
            "eval_limit": int(limit) if limited_ids and limit else None,
            "limited_via": "test_case_ids_to_generate.json + --run-ids" if limited_ids else None,
            "score_dir": str(score_root),
        },
        # the Gemini models that answered the web searches ({model: searches}) and the cascade
        # they were asked from (base.search_cascade())
        **({"gemini_models_answered": dict(sorted(gemini_answered.items())),
            "gemini_cascade": search_models()} if search_log is not None else {}),
        "status": status,
    }
    if web_withheld:
        result.update(accuracy_withheld_reason=web_withheld, failure_kind="infra_required",
                      error=web_withheld)
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    if attempts > 1:
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "bfcl_v4_agentic: `bfcl generate` generates each entry once and "
                           "`bfcl evaluate` scores that generation"}
    if limit_plan is not None:
        result["eval_limit_plan"] = limit_plan
    record_shard(result, shard_label)
    if shard_label:
        result["bfcl_report"]["shard"] = shard_label
    return result


