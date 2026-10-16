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

"""The canonical-sync provenance registry is plain, complete text.

Every value in CANONICAL_SYNC is one string literal (implicit concatenation only), ASCII, on one
line when printed, with balanced parentheses and double quotes, so a replacement cut off at an
apostrophe or a stray newline shows up here. The pinned fragments are the replacement texts the
suites' owners requested for their provenance entries: each must appear verbatim, and the cut-off
forms an earlier edit left behind must be gone.

What a result records about its sampling is checked the same way: a TokenBudget record never
says a cap was applied when a different value was sent, and the judge cascade reports the
replies its token limit cut off.
"""

import ast
import json
from pathlib import Path

import pytest

from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import toolbench as TB

SOURCE = (Path(__file__).resolve().parents[1] / "gbench" / "runners" / "eval_suites"
          / "canonical_sync.py")

PINNED = {
    "acebench": [
        "bubblewrap missing or the container not isolated -> infra_required)",
        ("= gbench Gemini judge cascade at APIUSER's own sampling (temperature 0.001, top_p 1, "
         "max_tokens 1000 for the reply, sent as max_output_tokens 1000 + the Gemini thinking "
         "allowance 8192; replies that stopped there and outages after them counted, with text "
         "used as they are, without text passed to the next cascade model, either not "
         "comparable) (recorded; GEMINI_API_KEY else infra_required)"),
        ("sampling model_infer.py LLMInfer (temperature 0.0, top_p 0.9, max_tokens 1024; "
         "OfficialSampling.max_tokens) on every request of the model under test, within LLMInfer's "
         "max_model_len=8192 window (model_infer.py lines 28/36): without --thinking the 1024 is "
         "sent, or the room the window leaves after the prompt as the pinned vllm==0.6.1.post1 "
         "(commit acda0b35d00e) applies it: at most 8193 - P tokens for a prompt of P <= 8192 "
         "tokens (stop_checker.py stops once the length exceeds max_model_len), a reply stopping "
         "there counted apart from truncations, and a prompt of more than 8192 tokens is not sent, "
         "its reply the empty text that vLLM returns for it (scheduler.py FINISHED_IGNORED, finish "
         "reason length; a single-turn row scored on the empty reply, an agent step handing it on "
         "and the conversation going on), counted in upstream_context_window and comparable; "
         "prompts counted as that vLLM counts them (/tokenize of the chat template with "
         "add_special_tokens true, only when bytes + 512 + 1024 exceed 8193), a prompt only "
         "estimated or with a conservative tokenizer count near the window sent the 1024 and not "
         "comparable; with --thinking neither the cap nor the window is applied (run budget, "
         "recorded as official cap not applied to reasoning); replies cut off by any other budget "
         "(an empty reply after reasoning included) counted and not comparable; single-turn rows "
         "scored after generation in worker threads (async_eval_fn, deterministic);"),
    ],
    "agent_dojo": [
        ("no temperature and no max_tokens sent (max_output_tokens_note), as the vllm_parsed "
         "provider"),
    ],
    "api_bank": [
        ("(OfficialSampling send_temperature=False, send_max_tokens=False: "
         "ChatGPTWrapper/GPT4Wrapper post model + messages only, so no temperature and no "
         "max_tokens)"),
    ],
    "arc_agi": [
        ("comparable for full 2-attempt runs in which every attempt got a server reply, no "
         "attempt's last try was cut off at the max_tokens budget (counted by finish_reason length "
         "/ status TRUNCATED, non_convergent relabels included; attempts_cut_off; the same count "
         "drives the >10% status rule), and no operator temperature"),
    ],
    "beam_128k": [
        ("prune_from_tail to the served window leaving upstream's 2,000-token answer room (the "
         "run budget with --thinking), sized for the token-longest question"),
    ],
    "bfcl": [
        ("the agentic handler's per-request timeout sized the same way, the SDK's 2 retries "
         "kept; without a reported window both sized for base.UNCAPPED_TIMEOUT_TOKENS 262144)"),
        ("a stopped evaluate leaves the agentic categories unscored, a generate stopped before "
         "it recorded first)"),
        ("agentic delegated to bfcl_v4_agentic (its web_search tool served by Gemini Google "
         "Search grounding in place of BFCL's DuckDuckGo via SerpAPI search, so a run that "
         "scores it is not comparable: the agentic reason 'web search served by Gemini Google "
         "Search grounding (BFCL queries DuckDuckGo via SerpAPI)' carried into "
         "leaderboard_comparable_reason; only the Web Search setting needs the Gemini key "
         "(GEMINI_API_KEYS, else GEMINI_API_KEY), checked with base.gemini_key_live_valid before "
         "generation (the judged suites' gate): unset or rejected, or a search no grounding "
         "model served during generation (generate stopped at the first one, log read every 30 "
         "s, the unfinished memory entries then generated by the resume built into bfcl "
         "generate, run without --allow-overwrite) -> that setting reported infra_required in "
         "its own entry (web_search_setting, settings_infra_required, prerequisite "
         "GEMINI_API_KEY; never scored, Overall Acc and the Agentic column null, and on a run "
         "of every category the headline withheld (accuracy null, accuracy_withheld_reason; "
         "pooled_accuracy kept), an explicit --eval-categories subset headlined by "
         "pooled_accuracy, completed_with_errors, not comparable) while single-turn, multi-turn "
         "and memory run "
         "and are scored; web_search alone selected -> infra_required; any other prerequisite "
         "lost mid-run -> infra_required with the scored parts logged);"),
    ],
    "bfcl_v4_agentic": [
        ("OpenAICompletionsHandler unmodified except the --thinking chat_template_kwargs toggle "
         "and a per-request timeout sized for the served window (base.request_timeout_s of "
         "max_model_len, else of 262144; the SDK's 2 retries kept; cf. base_oss_handler.py "
         "timeout=72000)"),
        ("Gemini key (GEMINI_API_KEYS, else GEMINI_API_KEY) checked with "
         "base.gemini_key_live_valid before generating, the judged suites' gate (unset or "
         "rejected (HTTP 401/403, a 400 naming the API key) -> the Web Search setting reported "
         "infra_required in its own entry (settings.web_search, settings_infra_required, "
         "prerequisite GEMINI_API_KEY; not generated, never scored, overall_accuracy_agentic "
         "null, accuracy = summary_memory, completed_with_errors, not comparable) while the "
         "Memory setting runs; a check that gets no answer does not block; only web_search "
         "categories selected -> infra_required hard error);"),
        ("a search no grounding model served (BFCL's own error text to the model) -> the Web "
         "Search setting infra_required (search log read every 30 s during generate, stopped at "
         "the first one; after such a stop the memory entries not finished are generated by the "
         "resume built into bfcl generate (the memory categories without --allow-overwrite: "
         "collect_test_cases keeps the finished entries, clean_up_memory_prereq_entries drops "
         "finished prerequisites), without the Gemini relay and bounded on its own, and only the "
         "memory categories are evaluated; only the web_search generate uses the Gemini relay);"),
        ("(cut-off lines dropped; in a limit run a stop during the harness' whole-file rewrite "
         "(update_mode) can also drop finished entries, counted unfinished; with a wall clock a "
         "full run's planned counts are read from the image before generating;"),
        ("a stopped evaluate raises HarnessTimeout (carrying the record of a generate stopped "
         "before it);"),
        ("a stopped generate reads the search log: a failed search makes the Web Search setting "
         "infra_required (memory resumed), otherwise"),
        ("a run that scores it is not comparable ('web search served by Gemini Google Search "
         "grounding (BFCL queries DuckDuckGo via SerpAPI)'; bfcl_report.search_backend and "
         "search_substitution record it);"),
    ],
    "chartqa": [
        ("--thinking the same temperature 0 (the client runs a thinking model with "
         "enable_thinking_kwarg, which keeps the task's generation_kwargs; its o1/o3/o4/gpt-5 "
         "model-name branch is not used) and the 16 not applied (lmms-eval declares no "
         "reasoning-model cap: the run's --max-output-tokens, else the thinking floor, recorded "
         "as not applied to reasoning)"),
        ("every request answered (FAILED / TIMEOUT / OVER_CONTEXT named) and no reply cut off "
         "by the run's token budget (replies_cut_off_by_budget)"),
    ],
    "coco_caption": [
        ("--thinking the same temperature 0 and top_p 1.0 (enable_thinking_kwarg keeps the "
         "task's generation_kwargs; the o1/o3/o4/gpt-5 model-name branch is not used) and the "
         "64 not applied (lmms-eval declares no reasoning-model cap: the run's "
         "--max-output-tokens, else the thinking floor, recorded as not applied to reasoning)"),
        ("every request answered and no reply cut off by the run's token budget "
         "(replies_cut_off_by_budget)"),
    ],
    "copilot_bench_swe": [
        ("agentic default: mini-swe-agent 2.4.2 tool-calling rollout (instance image pulled first "
         "with the host's Docker CLI from the gbench process, swebench_common.host_pull_images, so "
         "the operator's docker login applies and the rollout container gets no Docker client "
         "configuration; a failed pull is an infrastructure failure; turns cut off at the end of "
         "the context window (truncated_turns) are counted and make the run not comparable; turns "
         "with reasoning but no content or action (empty_after_reasoning_turns) are split by their "
         "token ids (swebench_common.classify_empty_reply): the answer inside the thinking channel "
         "is the model's outcome, counted in answer_in_thinking, not a comparability reason; a "
         "native tool call the server's tool-call parser removed (tool_call_dropped_turns) and a "
         "turn without token ids to tell the two apart (unclassified_empty_turns) are not the "
         "model's answer and make the run not comparable; the one-shot "
         "harness pass pulls each row's image the same way; driver follows"),
    ],
    "deepsearch_qa": [
        ("no temperature or max_tokens sent (the notebook's model call sets only the "
         "google_search tool);"),
    ],
    "docvqa": [
        ("--thinking the same temperature 0 (enable_thinking_kwarg keeps the task's "
         "generation_kwargs; the o1/o3/o4/gpt-5 model-name branch is not used) and the 32 not "
         "applied (lmms-eval declares no reasoning-model cap: the run's --max-output-tokens, "
         "else the thinking floor, recorded as not applied to reasoning)"),
        ("every request answered and no reply cut off by the run's token budget "
         "(replies_cut_off_by_budget)"),
    ],
    "gaia": [
        ("web_search tool loop, 20 tool-enabled model turns then forced final; each turn sends "
         "run_gaia.py's max_tokens=4096 without --thinking and the run's budget with it (the "
         "4096 is not applied to reasoning; run_gaia.py's reasoning_effort high and "
         "max_completion_tokens 8192 are keyed on its o1 model_id, not a reasoning-model "
         "setting for a served model; no reasoning_effort is sent in either mode);"),
    ],
    "gdpval": [
        "with no temperature or max_tokens sent (the paper sets neither)",
    ],
    "gorilla_apibench": [
        ("sampling as get_llm_responses.py (n=1, temperature=0, no max_tokens; OfficialSampling "
         "send_max_tokens=False);"),
    ],
    "gpqa": [
        ("both modes = simple_evals.py 'gpt-4o' entry, lines 236-240 (ChatCompletionSampler: "
         "system message OPENAI_SYSTEM_MESSAGE_API 'You are a helpful assistant.', T=0.5)"),
        ("simple_evals.py's o-series entries, lines 145-212 (ResponsesSampler "
         "reasoning_model=True for o3/o4-mini/o1-pro, OChatCompletionSampler for "
         "o1/o1-preview/o1-mini/o3-mini: no system message, temperature or max_tokens), are "
         "keyed on OpenAI model names and are not reasoning settings for a served model, which "
         "takes the chat sampler in both modes"),
        ("any failed/timed-out/empty/over-context request (a request with no reply is a harness "
         "error, left out of the score; an empty reply is scored wrong), a no-think request sent "
         "a max_tokens other than 2048, or a reply cut off before finishing at anything but the no-think 2048 "
         "(a thinking reply at the run's budget or a context-clamped max_tokens, a no-think "
         "reply at a clamped max_tokens)"),
    ],
    "gpqa_diamond": [
        ("any failed/timed-out/empty/over-context request (a request with no reply is a harness "
         "error, left out of the score; an empty reply is scored wrong), a no-think request sent "
         "a max_tokens other than 2048, or a reply cut off before finishing at anything but the no-think 2048 "
         "(a thinking reply at the run's budget or a context-clamped max_tokens, a no-think "
         "reply at a clamped max_tokens)"),
    ],
    "gsm8k": [
        ("temperature 0 + seed 1234 in both modes (OfficialSampling; lm-eval's "
         "openai-chat-completions branch for o1/o3/o4/5 model names, temperature 1 and no stop, "
         "fits OpenAI's hosted API and is not in local-chat-completions, so it is not applied)"),
    ],
    "healthbench": [
        "with gpqa.SIMPLE_EVALS_SAMPLING as in simpleqa and gpqa",
    ],
    "humanitys_last_exam": [
        ("over the Gemini cascade (judge_sampling); a judge reply without a yes/no verdict "
         "(e.g. stopped before the ExtractedAnswer JSON closed, also counted in "
         "judge_replies_incomplete_json) is JUDGE_FAILED: excluded from accuracy and "
         "calibration, counted and named"),
    ],
    "ifeval": [
        ("; leaderboard_comparable=False for --eval-limit, a sampled run, "
         "requests with no reply (harness errors, left out of the headline and ifeval_report) "
         "and empty replies (failed_requests), replies cut off at "
         "the max_tokens budget (truncated_responses), or verdicts that depend on the "
         "whitespace trim"),
    ],
    "infographicvqa": [
        ("--thinking the same temperature 0 (enable_thinking_kwarg keeps the task's "
         "generation_kwargs; the o1/o3/o4/gpt-5 model-name branch is not used) and the 32 not "
         "applied (lmms-eval declares no reasoning-model cap: the run's --max-output-tokens, "
         "else the thinking floor, recorded as not applied to reasoning)"),
        ("every request answered and no reply cut off by the run's token budget "
         "(replies_cut_off_by_budget)"),
    ],
    "loft_scifact": [
        ("leaderboard_comparable also requires every request answered (failed or timed-out "
         "requests named) and no reply cut off at the end of the context window "
         "(replies_cut_off_by_budget)"),
    ],
    "mcp_bench": [
        ("credentials and the endpoint URL passed to docker by name, key values redacted from "
         "server errors before any cut; model-under-test requests as upstream builds them "
         "(llm/provider.py get_completion), with the run's thinking switch "
         "(chat_template_kwargs.enable_thinking), no temperature unless the operator sets one, "
         "upstream's caps exactly without --thinking (planning rounds and the final answer 12000, "
         "summaries 10000, format fixes 8000, compression its target length) and with --thinking "
         "the run's budget fitted to the context window and a request timeout sized to it, "
         "base.request_timeout_s, in place of the openai client's default 600 s read timeout "
         "(gbench/docker/mcp_bench_request_policy.py; image label "
         f"{MB.REQUEST_POLICY_LABEL}={MB.REQUEST_POLICY_VERSION}, else infra_required); the "
         "applied settings checked "
         "and the replies counted (model_replies); a model request that ends in an endpoint or "
         "connection error is a harness error, never the model's outcome "
         "(model_replies.failed_requests, timed_out, model_failed_by_error, harness_errors, status "
         "completed_with_errors), and window rejections left to upstream's handling "
         "(rejected_over_context) and in-flight cancellations such as at task_timeout 5000 s "
         "(cancelled) are counted and named;"),
    ],
    "medxpertqa": [
        ("max_tokens 2048 sent exactly without --thinking; with --thinking the same request with "
         "the run's --max-output-tokens (else the thinking floor) in place of the 2048, which is "
         "recorded as not applied to reasoning;"),
        ("every request of both turns answered and no reply of either turn cut off by the run's "
         "token budget (replies_cut_off_by_budget)"),
    ],
    "mmlu_pro": [
        ("or a TypeError/AttributeError/ImportError/NameError raised by chat.completions.create "
         "itself -> infra_required (the same raised while the stream is read, e.g. a malformed "
         "chunk, fail only that try: asked again after 5 s, as process_one does upstream)"),
        ("32768 or a larger --max-output-tokens with thinking (the larger value is flagged "
         "non-comparable and recorded as official_sampling.max_tokens_applied=false with "
         "max_tokens_sent)"),
    ],
    "mmmu_pro": [
        ("every request answered and no reply cut off by the run's token budget "
         "(replies_cut_off_by_budget)"),
    ],
    "multi_swe_bench": [
        ("the published instance images mswebench/<org>_m_<repo>:pr-N are pulled with the host's "
         "Docker CLI from the gbench process before each language's container "
         "(swebench_common.host_pull_images: the operator's docker login; the MSWE-agent container "
         "gets no Docker client configuration; an image that cannot be pulled is an infrastructure "
         "failure and is not handed to MSWE-agent), then tagged with the name MSWE-agent's "
         "vendored harness gives them (nothing is built; the tags are removed afterwards);"),
        ("sets chat_template_kwargs.enable_thinking, counts endpoint failures and reply outcomes) "
         "and OPENAI_API_KEY=EMPTY; an endpoint failure the relay saw is a harness error "
         "(endpoint_failures; completed_with_errors, not comparable; this also covers a 400 that "
         "MSWE-agent records as exit_cost), and model turns cut off at a limit (truncated_turns) "
         "make the run not comparable, while turns with reasoning but no content "
         "(empty_after_reasoning_turns) are split by their token ids "
         "(swebench_common.classify_empty_reply): the answer inside the thinking channel is the "
         "model's outcome, counted in answer_in_thinking, not a comparability reason; a native "
         "tool call the server's tool-call parser removed (tool_call_dropped_turns) and a turn "
         "without token ids to tell the two apart (unclassified_empty_turns) are not the model's "
         "answer and make the run not comparable;"),
        ("exit_api, an exit_error other than the step limit whose exception text starts with a "
         "model-endpoint failure or the docker daemon's unreachable message "
         "(swebench_common.scaffold_error_harness_reason), or an unfinished run whose setup "
         "failed (setup_error), whose rollout raised before the model's first turn or with a "
         "model-endpoint or docker daemon failure (the driver's rollout_error; an endpoint that "
         "keeps failing: OpenAIModel.query re-raises the openai exception, which "
         "forward_with_error_check does not catch), or that recorded neither, is an "
         "infrastructure failure; a rollout that raised on a model reply with no content "
         "(reasoning only: OpenAIModel.query returns message.content unchecked and the parser's "
         "re.finditer raises TypeError on None; multirun.main logs 'Failed on' and the instance "
         "has no prediction) is the model's outcome, crashed_empty_reply, unresolved with no "
         "patch, counted in empty_reply_crashes, and any other rollout exception after the "
         "model's first turn follows the exit_error rule (crashed); MSWE-agent's OpenAI client "
         "gets a 3660 s timeout (the relay's 3600 s per-turn limit + 60 s) instead of the SDK's "
         "600 s, at which it abandoned and resent a long thinking turn (harness-only, recorded "
         "in agent.openai_client_timeout_s); any other "
         "exit_error, and early_exit (SWEEnv.step: the container stopped answering while it ran "
         "the model's command, which raised, broke the pipe, or timed out and could not be "
         "interrupted), is the model's outcome, unresolved, as a single pass of upstream's run "
         "scores it (upstream's resume with --skip_existing would roll an early_exit instance out "
         "again; gbench rolls each instance out once), with agent_exit_note, counted in "
         "exits_kept_as_model_outcome;"),
    ],
    "multilingual_mmlu": [
        (", a no-think request sent a max_tokens other than 2048, or a reply cut off before "
         "finishing at anything but the no-think 2048 (a thinking reply at the run's budget or a "
         "context-clamped max_tokens, a no-think reply at a clamped max_tokens)"),
    ],
    "nestful": [
        ("sampling as src/eval.py (--temperature default 0.0, --max_tokens default 1000, run.sh "
         "passes neither; OfficialSampling.max_tokens: the 1000 sent exactly without "
         "--thinking, not applied with it (run budget, recorded as official cap not applied to "
         "reasoning); replies cut off by any other budget counted in replies_cut_off, not "
         "comparable);"),
    ],
    "nexusraven_api_eval": [
        ("temperature 0 and no max_tokens for query and repair (build_llm OpenAIChat "
         "gpt_params; LangChain 0.0.294 OpenAIChat sends model + model_kwargs only; "
         "OfficialSampling send_max_tokens=False);"),
    ],
    "omnidocbench": [
        ("leaderboard_comparable also requires every request answered (a page whose request got "
         "no reply from the served endpoint is a harness error: left out of the predictions and "
         "the ground truth the evaluator scores, counted in pages_left_out and "
         "request_harness_errors, status completed_with_errors; when every page is one, the "
         "evaluator does not run, status failed, accuracy null with accuracy_withheld_reason) and "
         "no reply cut off by the run's token budget (replies_cut_off_by_budget)"),
    ],
    "putnam_formal": [
        ("cascade reviews the elaborated answer (bound variables renamed, literals as Lean "
         "writes them and Name literals as the applications they are built from, printed in "
         "full by the checker's print mode in a container of its own with Lean's printing "
         "limits lifted, sent whatever its length; an answer that container does not print "
         "(past its time or 8 GB limit, or a printer error) is not reviewed, not scored and "
         "named in leaderboard_comparable_reason) against the reference answer"),
        ("leaderboard_comparable_reason) against the reference answer; the print mode shows an "
         "answer at its value on one line or not at all: string literals and Bind.bind "
         "continuations wrapped in empty metadata (ProofWidgets HTML and do notation print as "
         "applications), Qq q(...) notation and safe shadowing off, every atom of the delaborated "
         "term a token or keyword, every identifier free of line-breaking characters and escape "
         "brackets, string and character literals re-escaped (U+0080-U+009F, U+2028 and U+2029 "
         "included), else a printer error; stage and judge model recorded per trace); trusted "
         "kernel checker "),
        ("verification harness errors left out of the accuracy and counted as failed requests; "
         "replies cut off by the run's token budget (PutnamBench publishes no generation "
         "harness, so there is no official cap and the house budget applies), empty answers "
         "after reasoning included, counted in replies_cut_off_by_budget and "
         "replies_cut_off_in_reasoning and make the run not leaderboard_comparable; the "
         "image's toolchain, Mathlib and commit "),
    ],
    "ruler": [
        ("the pre-flight uses the budget base sends (the declared cap without --thinking; with "
         "it the run's budget, the caller's max_output_tokens included, larger or smaller than "
         "the cap); leaderboard_comparable also requires every request answered (failed or "
         "timed-out requests named) and no reply cut off by the run's token budget "
         "(replies_cut_off_by_budget)"),
        ("per-task caps as each sample's max_tokens (sent without --thinking; with --thinking "
         "replaced by the run budget under the token-limit policy and recorded in "
         "official_cap_not_applied_to_reasoning; pre-flight sized the same way); "
         "string_match_all"),
    ],
    "screenspot_pointing": [
        ("no temperature and no max_tokens sent (the reference script uses the model's own "
         "GenerationConfig); leaderboard_comparable=False (prompt)"),
    ],
    "seal_tools": [
        ("(model + messages; OfficialSampling send_temperature=False and send_max_tokens=False: "
         "no temperature and no max_tokens, --max-output-tokens not applied; an operator "
         "temperature makes the run non-comparable)"),
    ],
    "simpleqa": [
        ("max_tokens 2048 on every request (gpqa.SIMPLE_EVALS_SAMPLING, the declaration gpqa "
         "passes, applied by the token-limit policy), --max-output-tokens not applied"),
        ("model sampled as simple-evals samples it, with gpqa's shared samplers "
         "(gpqa.SIMPLE_EVALS_SAMPLING, gpqa.simple_evals_messages): no --thinking"),
        ("--thinking = the same chat sampler (system message, temperature 0.5) with the run's "
         "--max-output-tokens instead of the 2048 (official cap not applied to reasoning; "
         "simple-evals' o-series samplers, keyed on OpenAI model names, are not reasoning "
         "settings for a served model)"),
    ],
    "swe_bench_live": [
        ("every selected setting whose prerequisites are present runs, and each selected setting "
         "whose prerequisite is missing is reported in settings.<name> with status infra_required "
         "and the missing prerequisite (never scored; listed in settings_infra_required; suite "
         "status completed_with_errors), in both directions: the Windows tab off a Windows Docker "
         "host, the Python and MultiLang tabs without the Linux harness or SWE-agent-for-eval "
         "image (e.g. on a Windows-containers host), and a setting whose pinned snapshot cannot be "
         "fetched; the suite hard-errors when no selected setting can run "
         "(swe_bench_live.missing_prerequisites) and when the Linux tabs' cloud metadata server "
         "gate fails (swebench_common.require_task_network_blocked on docker's default bridge "
         "and a user-defined bridge network, before the first Linux tab and again before each "
         "SWE-agent batch and each MultiLang harness pass, cached; infra_required for the whole "
         "suite);"),
        ("an instance whose image could not be pulled, no .pred, an exit status naming exit_api, "
         "or an exit_error whose exception text starts with a model-endpoint failure while the "
         "tab's relay saw an endpoint failure is an infrastructure failure (the fork's "
         "LiteLLMModel.query retries with tenacity reraise=True and sends no timeout, so an "
         "endpoint that keeps failing reaches forward_with_handling's catch-all as exit_error "
         "with litellm.<Class>, never RetryError/exit_api, after the relay answered its requests "
         "as failed; the text alone is not enough, since the SWE-ReX runtime that reports "
         "exceptions runs in the model's task container); any other exit_error or "
         "exit_environment_error (e.g. the EOF of a shell the model killed, or a context-window "
         "overflow, as exit_context) is the model's outcome, as upstream counts it, with "
         "agent_exit_note, counted in exits_kept_as_model_outcome; the instance images and "
         "SWE-ReX's builder base python:3.11.9-slim-bookworm are pulled with the host's Docker "
         "CLI from the gbench process before the batch (swebench_common.host_pull_images: the "
         "operator's docker login; the SWE-agent container gets no Docker client configuration), "
         "and so is each submitted instance's image before a Linux harness pass;"),
        ("sets chat_template_kwargs.enable_thinking, and counts endpoint failures and reply "
         "outcomes; the agents get OPENAI_API_KEY=EMPTY; Win-Agent and the harnesses get no "
         "credential variable); an endpoint failure the relay saw is a harness error "
         "(endpoint_failures; tab completed_with_errors, not comparable), and model turns cut off "
         "at a limit (truncated_turns: finish_reason length, the end of the context window since "
         "no max_tokens is sent) make the tab not comparable, while turns with reasoning but no "
         "content or tool call (empty_after_reasoning_turns) are split by their token ids "
         "(swebench_common.classify_empty_reply): the answer inside the thinking channel is the "
         "model's outcome, counted in answer_in_thinking, not a comparability reason; a native "
         "tool call the server's tool-call parser removed (tool_call_dropped_turns) and a turn "
         "without token ids to tell the two apart (unclassified_empty_turns) are not the model's "
         "answer and make the tab not comparable (both summed over the tabs at the top level);"),
        "accuracy = Lite (the leaderboard's default tab; else the first setting that ran);",
    ],
    "swe_bench_multilingual": [
        ("agentic default: mini-swe-agent 2.4.6 tool-calling rollout (instance image pulled first "
         "with the host's Docker CLI from the gbench process, swebench_common.host_pull_images, so "
         "the operator's docker login applies and the rollout container gets no Docker client "
         "configuration; a failed pull is an infrastructure failure; turns cut off at the end of "
         "the context window (truncated_turns) are counted and make the run not comparable; turns "
         "with reasoning but no content or action (empty_after_reasoning_turns) are split by their "
         "token ids (swebench_common.classify_empty_reply): the answer inside the thinking channel "
         "is the model's outcome, counted in answer_in_thinking, not a comparability reason; a "
         "native tool call the server's tool-call parser removed (tool_call_dropped_turns) and a "
         "turn without token ids to tell the two apart (unclassified_empty_turns) are not the "
         "model's answer and make the run not comparable; the one-shot "
         "harness pass pulls each row's image the same way; driver follows"),
    ],
    "swe_bench_pro": [
        ("it forwards only GET/POST requests for /v1/chat/completions, /v1/completions and "
         "/v1/models to the fixed endpoint, refusing any other path"),
    ],
    "swe_lancer": [
        ("the model is reached through gbench's EndpointRelay, which forwards only "
         "/v1/chat/completions, /v1/completions and /v1/models, adds the endpoint key"),
    ],
    "t_eval": [
        ("greedy temperature 0.0 as test.py llm.chat(do_sample=False), max_tokens 512 as "
         "test.py line 115 HFTransformerCasualLM(max_new_tokens=512) for --model_type hf "
         "(OfficialSampling.max_tokens: sent exactly without --thinking, not applied with it; "
         "replies at 512 left out of the truncations in the summed status; replies cut off by "
         "any other budget counted in replies_cut_off, not comparable);"),
    ],
    "textvqa": [
        ("--thinking the same temperature 0 (enable_thinking_kwarg keeps the sampling; the "
         "o1/o3/o4/gpt-5 model-name branch is not used) and the 1024 not applied (lmms-eval "
         "declares no reasoning-model cap: the run's --max-output-tokens, else the thinking "
         "floor, recorded as not applied to reasoning)"),
        ("every request answered and no reply cut off by the run's token budget "
         "(replies_cut_off_by_budget)"),
    ],
    "toolbench": [
        ("grader and simulator on the Gemini API only (the simulator on its OpenAI-compatible "
         "endpoint through gbench's Gemini relay; no knob for another provider, base URL, key or "
         "model; both substitutions recorded in gemini_substitutions; the cascade passed into the "
         "container as GBENCH_JUDGE_MODELS); the container runs the model's tool calls, so it "
         "runs on docker's bridge network after the metadata-server guard and holds no real key: "
         "the Gemini relay's placeholder passed to docker by name and unset in the container once "
         "the server config is written, the model endpoint through gbench's endpoint relay "
         "(the endpoint key, VLLM_API_KEY else OPENAI_API_KEY, added there; an endpoint URL "
         "with user info refused); tool calls the "
         "cached server failed (HTTP 500: the real-API proxy call, the simulator cascade or the "
         "server raised; each /virtual call recorded with its answer type, the Gemini model that "
         "simulated it, the failure stage and the query it belongs to, "
         "toolbench_server_calls.jsonl) or the simulator could not answer (three replies without "
         "valid JSON), found also in each query's whole search tree (tool_observations.json), "
         "tool-call timeouts, queries with a DFSDT step whose last try (of upstream's 6) failed "
         "at the endpoint (the patched model client records each step with a "
         "request that failed at the endpoint, with its group and query id, "
         "model_request_failures.jsonl; a step a later try answered is counted in "
         "model_request_failures and its query scored; a failed step tied to no query is "
         "infra_required) and queries without an answer are harness errors: those queries are "
         "left out of SoPR and SoWR (n_harness_error_queries, harness_error_query_kinds), status "
         "completed_with_errors, and a group whose every query is one publishes no score "
         "(accuracy null with accuracy_withheld_reason, status failed, its records kept); the "
         "real-API proxy call gets a 120 s read timeout (upstream sends "
         "none); ToolEval calls without a verdict are harness errors, a SoWR comparison without "
         "one is no vote, a query no pass could vote on is left out of the SoWR denominator and a "
         "group with no vote publishes no SoWR (infra_required); select_better_answer replies "
         "need only index (rtl.py reads no reason from it); the answering Gemini models "
         "(judge_models_used, simulator_models_used, gemini_models_answered), the cache / "
         "real-API / simulator usage (simulator_usage) and one record per test id (sample_traces) "
         f"recorded; image label {TB.HARNESS_LABEL}={TB.HARNESS_VERSION}, else infra_required;"),
        ("; model-under-test requests as chat_completion_request builds them (max_tokens 1024, "
         "frequency_penalty 0, presence_penalty 0, no temperature), with the run's thinking "
         "switch, no temperature unless the operator sets one, 1024 exactly without --thinking and "
         "with --thinking the run's budget fitted to the context window and a request timeout "
         "sized to it, base.request_timeout_s, in place of the openai client's default 600 s read "
         "timeout (docker/toolbench_request_policy.py; image label "
         f"{MB.REQUEST_POLICY_LABEL}={MB.REQUEST_POLICY_VERSION}, else infra_required; replies "
         "counted in "
         "model_replies; a request that ends in an endpoint or connection error is a harness "
         "error, never the model's outcome: model_replies.failed_requests, timed_out, "
         "model_failed_by_error, harness_errors, status completed_with_errors; window rejections "
         "left to upstream's handling counted as rejected_over_context; a run in which no model "
         "request got a reply publishes no score, infra_required); image packages pinned by "
         "docker/toolbench_constraints.txt"),
    ],
    "wildbench": [
        ("sampling temperature 0, top_p 1.0 and max_tokens 4096 without --thinking "
         "(scripts/_common_vllm.sh; with --thinking the run's --max-output-tokens, the cap "
         "recorded as not applied to reasoning); WB-Elo:"),
    ],
}


WHOLE_METHOD_FRAGMENTS = {
    "aider_polyglot": [
        "enchmark.py --tries 2 in the aider-benchmark:5dc9490b container (native edit format, pass",
        "rcises checkout's HEAD must equal POLYGLOT_BENCHMARK_COMMIT, else infra_required",
    ],
    "aime": [
        "st-chance reprompt; sampling = MathArena's per-model configs/models yaml on both",
        "t out of the score (scoring_excluded; grading errors also under judge_failures)",
    ],
    "bigcodebench": [
        "dy temperature 0.0 with the vllm backend's top_p 1.0 + EOS stops and max_new_tok",
        "accuracy null with accuracy_withheld_reason) when every Instruct request got no reply",
    ],
    "hmmt": [
        "e) finds no answer; sampling = MathArena's per-model configs/models yaml, on fir",
        "t out of the score (scoring_excluded; grading errors also under judge_failures)",
    ],
    "lcb": [
        "enAIChat prompt verbatim from the grader's run_test failure metadata, -4-without",
        "recomputes every LiveCodeBench field from the stored verdicts",
    ],
    "livebench": [
        "gbench-livebench image; the served model's own LiveBench config (livebench/model",
        "nt key reaches the endpoint through gbench's EndpointRelay on the bridge gateway",
    ],
    "mmlu": [
        "mpt verbatim (10 of 14,042 reference prompts show pandas' 'nan' for a 'None' opt",
        "rboard_comparable=True for full 5-shot runs with every question scored",
    ],
    "mmlu_redux": [
        "s=1 in OfficialSampling (recorded); base's token-limit policy: max_tokens=1 exac",
        "nk T=0 runs with no failed requests and at most 10% non-letter replies",
    ],
    "multi_swe_bench": [
        "then tagged with the name MSWE-agent's vendored harness gives them (nothing",
        "shot route hands every instance to run_eval_suite (shard, then limit, same seed)",
    ],
    "multipl_e": [
        "cluded) scores completion \"\" as upstream's do_task records it (the script's afor",
        "utes it over the shards' completions; --shard comes before the --eval-limit draw",
    ],
    "ojbench": [
        "cy: 64k (65536) with thinking, the paper's reasoning-model setting, never below",
        "by the pooled or Python figure; the judge container runs with --network none",
    ],
    "scicode": [
        "sequential per-sub-step gencode protoco",
        "trivial sub-step that only opens test_data.h5 is run that way in the sandbox "
        "(check_step_sandbox), and a failure there is infra_required",
    ],
    "swe_bench_live": [
        "base_commit and image_name only (the row's docker_image, or starryzhang/sweb.eva",
        "the relay corroboration is per tab (one SWE-agent batch), not per instance",
    ],
    "wmdp": [
        "ood MCQA equal to lm-eval v0.4.2 local-completions' _loglikelihood_tokens (log P(",
        "scored; --thinking recorded as not applied (thinking_applied=False)",
    ],
}

#: Text an earlier edit left where a replacement stopped at its first apostrophe.
CUT_OFF_FORMS = {
    "wildbench": "with --thinking the run; WB-Elo",
    "beam_128k": "leaving upstream, sized for",
    "ruler": "per-task caps as each sample;",
    "simpleqa": "with gpqa: no --thinking",
    "screenspot_pointing": "(the reference script uses the model;",
}

#: acebench text replaced by its requested updates.
ACEBENCH_REPLACED = [
    "bubblewrap else infra_required)",
    "judge cascade (recorded; GEMINI_API_KEY",
    "LLMInfer (temperature 0.0, top_p 0.9; OfficialSampling) on every request",
]


def _registry_node() -> ast.Dict:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "CANONICAL_SYNC":
            assert isinstance(node.value, ast.Dict)
            return node.value
    raise AssertionError("CANONICAL_SYNC not found")


def test_every_provenance_value_is_one_plain_string_literal():
    registry = _registry_node()
    suites = [k.value for k in registry.keys]
    assert len(suites) == len(set(suites)), "a suite is listed twice"
    for key, entry in zip(registry.keys, registry.values):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        assert isinstance(entry, ast.Dict), key.value
        fields = [k.value for k in entry.keys]
        assert len(fields) == len(set(fields)), key.value
        for field, value in zip(entry.keys, entry.values):
            assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
                key.value, field.value)


def test_provenance_text_is_ascii_single_line_and_balanced():
    for suite, entry in CS.items():
        for field, text in entry.items():
            where = f"{suite}.{field}"
            assert all(32 <= ord(c) < 127 for c in text), f"{where}: control or non-ASCII"
            depth = 0
            for c in text:
                depth += (c == "(") - (c == ")")
                assert depth >= 0, f"{where}: ')' before its '('"
            assert depth == 0, f"{where}: unclosed '('"
            assert text.count('"') % 2 == 0, f"{where}: unpaired double quote"


@pytest.mark.parametrize("suite", sorted(CUT_OFF_FORMS))
def test_entries_once_cut_at_an_apostrophe_are_whole_again(suite):
    method = CS[suite]["method"]
    assert CUT_OFF_FORMS[suite] not in method
    for text in PINNED[suite]:
        assert text in method, text


def test_acebench_provenance_carries_every_requested_update():
    method = CS["acebench"]["method"]
    for text in ACEBENCH_REPLACED:
        assert text not in method, text
    for text in PINNED["acebench"]:
        assert text in method, text


@pytest.mark.parametrize("suite", sorted(PINNED))
def test_requested_replacement_texts_appear_verbatim(suite):
    method = CS[suite]["method"]
    missing = [text for text in PINNED[suite] if text not in method]
    assert not missing, missing


@pytest.mark.parametrize("suite", sorted(WHOLE_METHOD_FRAGMENTS))
def test_rewritten_methods_are_complete(suite):
    method = CS[suite]["method"]
    head, tail = WHOLE_METHOD_FRAGMENTS[suite]
    assert head in method, head
    assert method.endswith(tail), method[-80:]


def test_prompts_in_provenance_write_newlines_as_backslash_n():
    """docvqa, chartqa, textvqa and infographicvqa print their prompt's newline the same way."""
    for suite in ("docvqa", "chartqa", "textvqa", "infographicvqa"):
        assert "\\nAnswer" in CS[suite]["method"], suite
    assert '".\\nUse the computer tool' in CS["screenspot"]["method"]


def test_provenance_prints_as_json():
    for suite, entry in CS.items():
        assert json.loads(json.dumps(entry)) == entry, suite


# --------------------------------------------------------------------------- #
# what a result records about the token budget and the judge's replies
# --------------------------------------------------------------------------- #
def _explicit_reasoning_cap(cap: int = 8192):
    from gbench.runners.eval_suites import base
    return base.OfficialSampling(reference="ref", temperature=0.0, max_tokens=2048,
                                 reasoning=base.OfficialSampling(reference="ref reasoning",
                                                                 max_tokens=cap))


@pytest.mark.parametrize("knob,applied", [(None, True), (4096, True), (8192, True),
                                          (65536, False)])
def test_a_reasoning_cap_is_recorded_as_applied_only_when_it_is_sent(knob, applied):
    from gbench.runners.eval_suites import base
    plan = base.resolve_token_budget("s", _explicit_reasoning_cap(), thinking=True,
                                     max_output_tokens=knob)
    rec = plan.record()
    assert plan.policy == "official_reasoning"
    assert rec["max_tokens"] == 8192 and rec["max_tokens_sent"] == plan.max_tokens
    assert rec["max_tokens_applied"] is applied
    assert (plan.max_tokens == 8192) is applied
    if applied:
        assert "max_tokens_note" not in rec
    else:
        assert plan.max_tokens == 65536
        assert rec["max_tokens_note"] == base.RAISED_REASONING_CAP_NOTE


def test_no_budget_record_claims_a_cap_that_was_not_sent():
    from gbench.runners.eval_suites import base
    official = _explicit_reasoning_cap()
    for thinking in (False, True):
        for knob in (None, 1024, 2048, 8192, 65536):
            plan = base.resolve_token_budget("s", official, thinking=thinking,
                                             max_output_tokens=knob)
            rec = plan.record()
            if rec.get("max_tokens_applied"):
                assert plan.max_tokens == rec["max_tokens"], (thinking, knob, rec)
    lifted = base.OfficialSampling(reference="ref", max_tokens=100)
    rec = base.resolve_token_budget("s", lifted, thinking=True, max_output_tokens=4096).record()
    assert rec["max_tokens_applied"] is False and rec["max_tokens_sent"] == 4096


class _Resp:
    def __init__(self, text, reason):
        self.text = text
        self.candidates = [type("C", (), {"finish_reason": reason})()]


class _Reason:
    def __init__(self, name):
        self.name = name


def _fake_judge(monkeypatch, replies):
    from gbench.runners.eval_suites import base
    calls = []

    class _Models:
        async def generate_content(self, model, contents, config):
            calls.append(model)
            reply = replies[model]
            if isinstance(reply, Exception):
                raise reply
            return reply

    client = type("Client", (), {"aio": type("Aio", (), {"models": _Models()})()})()
    monkeypatch.setattr(base, "gemini_api_keys", lambda: ["k"])
    monkeypatch.setattr(base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(base, "_next_judge_client", lambda: client)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", ",".join(replies))
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    return calls


def test_the_judge_cascade_reports_replies_its_token_limit_cut(monkeypatch):
    import asyncio
    from gbench.runners.eval_suites import base
    _fake_judge(monkeypatch, {"m1": _Resp(None, _Reason("MAX_TOKENS")),
                              "m2": _Resp('{"a": 1', "MAX_TOKENS")})
    text, model, info = asyncio.run(base.judge_generate_cascade_detailed("q"))
    assert (text, model) == ('{"a": 1', "m2")
    assert info == {"finish_reason": "MAX_TOKENS", "max_tokens_without_text": 1,
                    "refused": False, "block_reasons": []}
    assert asyncio.run(base.judge_generate_cascade("q")) == ('{"a": 1', "m2")


def test_the_judge_cascade_counts_cut_replies_when_it_ends_in_an_outage(monkeypatch):
    import asyncio
    from gbench.runners.eval_suites import base
    _fake_judge(monkeypatch, {"m1": _Resp(None, _Reason("MAX_TOKENS")),
                              "m2": RuntimeError("429"), "m3": _Resp(None, _Reason("SAFETY"))})
    text, model, info = asyncio.run(base.judge_generate_cascade_detailed("q"))
    assert (text, model) == (None, "JUDGE_OUTAGE")
    # m3 blocked the prompt, but m1 and m2 did not: not a refusal of every model
    assert info == {"finish_reason": None, "max_tokens_without_text": 1,
                    "refused": False, "block_reasons": ["finish_reason:SAFETY"]}


# --------------------------------------------------------------------------- #
# image labels named in the provenance carry the version the code requires
# --------------------------------------------------------------------------- #
def _versioned_image_labels():
    """{label: {version}} for every <PREFIX>_LABEL / <PREFIX>_VERSION pair of the eval-suite
    modules whose label is one of gbench's image labels (org.gbench.*); plugin modules excluded."""
    import importlib
    import pkgutil

    import gbench.runners.eval_suites as suites
    plugins = {"swe_bench_pro_agentic", "jetski_agent", "slashbin_agent"}
    labels = {}
    for info in pkgutil.iter_modules(suites.__path__):
        if info.name in plugins:
            continue
        mod = importlib.import_module(f"gbench.runners.eval_suites.{info.name}")
        for attr, label in vars(mod).items():
            version = getattr(mod, attr[:-len("LABEL")] + "VERSION", None)
            if attr.endswith("_LABEL") and isinstance(label, str) \
                    and label.startswith("org.gbench.") and isinstance(version, str):
                labels.setdefault(label, set()).add(version)
    return labels


_IMAGE_LABELS = _versioned_image_labels()


def test_the_image_labels_are_found():
    for label in ("org.gbench.skillsbench-harness", "org.gbench.toolbench-harness",
                  "org.gbench.mcp-bench-harness"):
        assert label in _IMAGE_LABELS, sorted(_IMAGE_LABELS)


@pytest.mark.parametrize("label", sorted(_IMAGE_LABELS))
def test_the_provenance_names_each_image_label_at_the_version_the_code_requires(label):
    """A canonical_sync entry that names an image label (`org.gbench.<name>=<version>`) states
    the version the suite's code requires of the image (its <PREFIX>_VERSION): a stale number
    there once said skillsbench-harness=3 while the code and the image required 4."""
    import re

    from gbench.runners.eval_suites import canonical_sync as cs
    versions = _IMAGE_LABELS[label]
    assert len(versions) == 1, (label, versions)   # modules sharing a label agree on its version
    text = json.dumps(cs.CANONICAL_SYNC) + json.dumps(getattr(cs, "SHARED_SYNC", {}))
    named = re.findall(re.escape(label) + r"=(\w+)", text)
    assert all(v in versions for v in named), (label, named, versions)
