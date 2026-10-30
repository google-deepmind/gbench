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

"""Canonical-sync provenance registry.

One source of truth for when each eval was last reconciled against its upstream benchmark, and
against what exact version. The dispatcher stamps `canonical_sync` onto every eval result, and
`gbench --eval-provenance` prints the table.

Each entry records:
  synced    the reconcile date (YYYY-MM-DD); a second verdict-relevant reconcile on the same
            day appends a serial (2026-09-28.2, then .3), so the anchor still moves
  upstream  the upstream project or package (repo, PyPI name, or "roadmap"/"internal")
  revision  the pinned commit, tag, or version reconciled against (optional)
  dataset   the dataset id and revision the loader reads (optional)
  checksum  a content fingerprint that backs `synced` with a VERIFIABLE hash rather than a
            self-attested date: the sha256 of the pinned dataset snapshot or file, its git
            blob id ("git-blob:<sha1>"), or the upstream tree/commit sha (optional).
            `gbench --eval-provenance` reports an entry as "version-pinned" when it carries
            a `revision` or `checksum`, else "date-only".
  method    how it was reconciled (for example "delegated to pinned harness", "metric reconciled
            via trace-walk", "CLI flags verified against the wheel")

A suite with no entry is reported as status "unverified": it has not yet been formally reconciled,
so `gbench --eval-provenance` doubles as the remaining reconcile worklist. Add entries as suites are
reconciled; do not invent a date or a commit that was not actually checked.

Structured fields. `revision` starts with a pin: a 40- or 64-hex commit or digest, a release
version (1.2.0), or package==version; several pins are separated by ", ". `checksum` starts with
sha256:<64 hex>, git-blob:<40 hex> or git-tree:<40 hex>. Text after the leading pin is a
qualifier. entry_pinned() is the predicate `gbench --eval-provenance` and the tests share.

The code-fingerprint gate. Each suite, and each module several suites share (SHARED_SYNC), has
a recorded code fingerprint (CODE_FP, generated below): the sha256 of its docstring-stripped
source AST, the eval_suites modules only it imports, the vendored data it reads and its Docker
build files. tests/test_canonical_sync.py fails when a fingerprint no longer matches the code, and
names what to do: re-reconcile the change against upstream; when it can change a verdict, bump
the entry's `synced` (and `revision` / `checksum` / `dataset` when the pin moved); then refresh
the fingerprints with

    python gbench/runners/eval_suites/canonical_sync.py --refresh-code-fp

A change that cannot alter any verdict (a refactor) keeps the entry as it is and is declared in
VERDICT_PRESERVING with the new fingerprint and the reason. `--check-bump --base <git ref>` is the
review gate: a fingerprint that moved since <ref> must come with a moved anchor (synced, revision,
checksum, dataset) or that declaration.

The refresh applies the same rule against the last refresh, so it cannot be used to turn the
freshness test green over an undecided change. Next to each fingerprint it records a digest of
the entry's anchors at that moment (FP_ANCHORS, generated below); `--refresh-code-fp` refuses, and
writes nothing, while a recorded fingerprint moved and neither the entry's anchors moved since it
was recorded nor VERDICT_PRESERVING declares exactly the new fingerprint (refresh_violations).

The evals scaffold_id folds in each selected suite's anchors and the anchors of every suite and
shared module it imports (scoring_signal), never the fingerprint itself: the id stays declared
(a human decides a verdict can change and bumps `synced`), and the gate makes sure the decision
is made. Stdlib only: the gate runs this file directly, with nothing installed.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Suites reconciled against a pinned external harness (delegated container / adapter builds).
CANONICAL_SYNC: Dict[str, Dict[str, Any]] = {
    "gaia2": {
        "synced": "2026-10-09.2",
        "upstream": "meta-agents-research-environments (PyPI)",
        "revision": "1.2.0",
        "dataset": ("meta-agents-research-environments/gaia2@78ea3bdbdeec2bdcd6afa5420915d8a22f23e"
                    "d99 and meta-agents-research-environments/gaia2_filesystem@132e26376f5e963bb5"
                    "9f64bcccdd02188cb08dee (local copies in the image, read offline)"),
        "method": ("delegated to are-benchmark (ARE 1.2.0, the current PyPI release, checked "
                   "2026-09-27; its gaia2 scenarios and hybrid scoring: deterministic hard "
                   "validation plus the LLM-judge soft validation, judge prompts upstream "
                   "verbatim); CLI flags verified against the 1.2.0 wheel; judge = gbench "
                   "Gemini cascade (GBENCH_JUDGE_MODELS, passed into the container; the proxy "
                   "does not start without it) through the in-container OpenAI-compatible proxy "
                   "(--judge_provider local --judge_endpoint <proxy> --judge_model "
                   "openai/gbench-cascade) in place of ARE's default Llama-3.3-70B judge, at "
                   "temperature 0 with no max_tokens (the ARE paper's verifier setting; "
                   "gaia2-cli's judge engine sends temperature 0), recorded in "
                   "gemini_substitutions, the answering Gemini model per soft check in "
                   "judge_verdicts and sample_traces, leaderboard_comparable=False; the judge "
                   "path as the current upstream judge runs it (gaia2-cli "
                   "gaia2_cli/judge/engine.py and gaia2_core/judge/checkers.py: the reply "
                   "verbatim, the verdict marker matched case-insensitively), because ARE "
                   "1.2.0's LiteLLMEngine lowercases True/False in every reply so LLMChecker's "
                   "[[True]] never matched (upstream issue #29 and PR #26 open, unfixed on "
                   "main; verified 2026-09-27 in the image: without the hook the signature, "
                   "sanity, cab and tone checkers return None, with it True); the agent "
                   "protocol of the ARE paper (arXiv:2509.17158 section 4.1) applied in the "
                   "hooked engine: temperature 0.5, max_tokens 16000 per turn without "
                   "--thinking (with --thinking the run's budget, the token-limit policy), the "
                   "agent's stop sequences <end_action> and Observation: ending each answer "
                   "(ARE's agent passes them to its engine on every step; the paper's appendix "
                   "B.5 adds them as custom stop sequences for models that continue past a "
                   "single tool call, naming Claude, Kimi and Qwen, and ran GPT-5 without them; "
                   "gbench sends them for every served model), sent as stop as ARE's "
                   "HuggingFaceLLMEngine sends them on a run without thinking (a --thinking run "
                   "sends none, because the served stack matches stop strings inside the "
                   "reasoning too, vLLM v1/engine/detokenizer.py: its agent request is "
                   "streamed, stream_options include_usage and continuous_usage_stats, and "
                   "closed once the answer holds a stop sequence, so the answer, the generation "
                   "on the server and the measured completion_duration end there as a "
                   "server-side stop on the answer would end them; a streamed request still "
                   "running at the request timeout is closed and raises LiteLLM's Timeout the "
                   "first time it runs over (a non-streamed one raises it only after LiteLLM's "
                   "client re-sent it, max_retries=2); a streamed reply's first chunk, content "
                   "'', is not answer text, so a --thinking reply that ended in its reasoning "
                   "gives ARE no reply (LiteLLMEngine's assert, as upstream); ARE's client-side "
                   "split of the answer applies on both runs; stop_sent_to_server, "
                   "stop_applied_by, streamed, stopped_client_side, ran_past_stop and "
                   "stopped_in_reasoning recorded, and answers that still held a stop sequence "
                   "and --thinking replies cut at max_tokens named in "
                   "leaderboard_comparable_reason), the chat template's thinking switch, a "
                   "request timeout sized to the per-turn cap, and 200 steps (ARE's agent "
                   "default is 80); the server-parser policy (base.recover_server_parser_text; "
                   "the launcher holds a copy of its three pure functions, held equal to base's "
                   "by a test): every agent request asks for the generated token ids "
                   "(return_token_ids; a hook on LiteLLM's openai conversion keeps a "
                   "non-streamed reply's token_ids and stop_reason, a streamed reply carries "
                   "them in its chunks); text the server's reasoning parser put in the "
                   "reasoning field on a run without thinking (server_parser_moved_text) and a "
                   "reply the tool-call parser emptied, no content, no reasoning and "
                   "completion_tokens > 0 (server_parser_dropped_text: a native <|tool_call> on "
                   "a request that declared no tools, as ARE declares none), are recovered in "
                   "generation order from the token ids decoded with special tokens kept by "
                   "gbench's DetokenizeRelay (a thread of gbench's process on the bridge "
                   "gateway that only turns token ids into text: the served tokenizer's "
                   "/detokenize, else the run's tokenizer) or, moved text only, from the "
                   "reasoning followed by the content, and handed to ARE as the reply, which "
                   "parses it as any reply (the model's outcome); counted in "
                   "server_parser_events and server_parser_recovered_from and named in "
                   "server_parser_note and leaderboard_comparable_reason, each such call and "
                   "every call without content keeping its rebuilt text (generation_text, 4000 "
                   "characters) with vLLM's stop_reason in gbench_agent_calls.jsonl; the Time "
                   "clock in ARE's measured mode with the client-measured completion_duration "
                   "the hooked engine returns, up to the end of the answer (paper section 4.3); "
                   "the judge endpoint and model are fixed (GBENCH_GAIA2_JUDGE_ENDPOINT / "
                   "GBENCH_GAIA2_JUDGE_MODEL removed); the container runs the agent's tool "
                   "calls, so it runs on docker's bridge network after the metadata-server "
                   "guard (no host name needs to resolve there) and holds no real key: the "
                   "model endpoint through gbench's EndpointRelay, the Gemini API through "
                   "gbench's GeminiRelay (its placeholder by name), no HF token (the datasets "
                   "are read offline), and gbench's DetokenizeRelay on the same gateway, which "
                   "holds no credential; headline = macro_success_rate of ARE's own statistics "
                   "(report_stats.calculate_statistics) recomputed over the scenario runs "
                   "without a harness error (percent; micro when macro is absent; the result's "
                   "macro_/micro_success_rate fields are fractions; ARE's rates as "
                   "benchmark_stats.json writes them kept as are_reported); ARE runs in a child "
                   "process of the launcher with gbench's hooks (gaia2_run.py --are-hooked): "
                   "every LiteLLM call that raises is recorded with its scenario run, and a run "
                   "whose agent's model endpoint call failed on the endpoint's side (any HTTP "
                   "status other than 400 and 422, no connection, timeout, an error inside a "
                   "streamed reply; a 400/422 or context-window rejection stays the model's "
                   "outcome, as runners/quality.py and toolbench count them, where base-run "
                   "suites count every endpoint rejection as a harness error), whose agent "
                   "reply the server's tool-call parser deleted and whose text could not be "
                   "recovered from token ids (server_parser_text_lost), whose judge call or "
                   "validation raised, whose judge reply held no verdict, or whose run raised "
                   "is a harness error left out of every rate whatever ARE validated it as "
                   "(n_harness_error_runs, harness_error_kinds, harness_error_runs; the ones "
                   "ARE validated as successes counted in n_harness_error_success_runs, because "
                   "ARE's greedy event matching and the agent's history make the error's effect "
                   "on the outcome unknowable); no_validation runs are left out by ARE; every "
                   "capability config's outcome is recorded (ARE's cli.py and "
                   "gaia2_submission.py skip a config that raised and report the others with "
                   "exit code 0; gbench_configs.jsonl, failed_configs) and the macro is "
                   "averaged over the capabilities with a validated run (ARE's formula counts "
                   "one without as 0.0; capabilities_without_validated_run); a capability the "
                   "run covers that the macro leaves out (its config raised, ARE did not run "
                   "it, only harness-error runs, none validated) is named in "
                   "missing_capabilities and leaderboard_comparable_reason and sets "
                   "completed_with_errors; these and the proxy's judge outages set status "
                   "completed_with_errors and are named; a failed call tied to no run, no "
                   "validated run left, a run in which not one agent request got a reply (no "
                   "request then carried text the model had generated), or an image without "
                   "label org.gbench.gaia2-harness-errors=7 is infra_required; a scenario "
                   "timeout stays the model's outcome, as ARE counts it (n_timeout_runs); "
                   "per-run records (sample_traces), agent-call and soft-check summaries, the "
                   "judge calls and the orchestrator log kept in the results dir; verified "
                   "2026-09-26 in the rebuilt image with ARE 1.2.0 (a stub model endpoint "
                   "answering 503: 480 failed agent calls tied to their runs, all 6 runs "
                   "excluded, no rate published) and 2026-09-27 (the hooked judge and agent "
                   "engines against stub endpoints: [[True]] verdicts pass, the protocol fields "
                   "sent, the generation time returned; the 1.2.0 wheel's config try/except, "
                   "macro formula and greedy event matching read, and vLLM's "
                   "v1/engine/detokenizer.py checked for stop matching over the reasoning; the "
                   "hooked agent engine against a stub streaming endpoint: a --thinking request "
                   "streamed without stop and closed at <end_action> in the answer while "
                   "Observation: in the reasoning was not matched, the stub saw the client "
                   "disconnect after 7 of 300+ chunks, completion_duration 0.2 s and the usage "
                   "so far returned; a streamed request past a 1 s timeout closed and recorded "
                   "as Timeout 408; the stop sequences of appendix B.5 read in the paper) and "
                   "2026-09-28 (the rebuilt image, no network, against a stub endpoint: a "
                   "no-think reply the tool-call parser emptied and one whose thought the "
                   "reasoning parser moved recovered from their token ids and handed to ARE, a "
                   "streamed --thinking reply whose tool call the server returned recovered "
                   "from the stream's token ids, a streamed reply that ended in its reasoning "
                   "after a first chunk with content '' raised AssertionError as upstream, and "
                   "ARE's loader read a demo scenario from the local copy with "
                   "HF_HUB_OFFLINE=1); gaia2 (revision "
                   "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99) and gaia2_filesystem (revision "
                   "132e26376f5e963bb59f64bcccdd02188cb08dee) baked at build time as local "
                   "copies and read offline (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1: ARE's "
                   "--hf-dataset is the local gaia2 copy, the filesystem app's DEMO_FS_PATH the "
                   "local gaia2_filesystem copy; the build fails unless each of the 7 gaia2 "
                   "configs loads offline with rows), recorded in the result's dataset; the "
                   "launcher decodes every agent reply's token ids and applies "
                   "base.server_parser_residue (copied verbatim) as well as "
                   "recover_server_parser_text: a --thinking reply whose native tool call the "
                   "server's parser deleted after the thought is handed to ARE "
                   "(server_parser_dropped_text), and every reply without content keeps "
                   "generation_text; --shard I/N (GAIA2_SHARD, validated host-side) keeps shard "
                   "I of each config's sorted scenario ids in the stream ARE reads, before its "
                   "--limit, recorded (shard, shard_selection), an image that did not apply it "
                   "is infra_required, and an empty shard gives accuracy None; merge_shards "
                   "recomputes ARE's calculate_statistics over the shards' runs without a "
                   "harness error; num_runs = GBENCH_GAIA2_NUM_RUNS, else max(3, "
                   "--attempt-count) (attempts record); GAIA2_MAX_CONCURRENT = "
                   "GBENCH_GAIA2_MAX_CONCURRENT, else the suite's concurrency (recorded); the "
                   "model key falls back to base.endpoint_api_key; with --eval-limit in run "
                   "mode the launcher's dataset hook keeps a seeded draw of each config's "
                   "(shard's) scenario ids (stratified_indices, seed gaia2), which ARE's "
                   "--limit then runs exactly; summary selection.limit_draw required for a "
                   "limited run"),
    },
    "wildclawbench": {
        "synced": "2026-10-13",
        "upstream": "github.com/internlm/WildClawBench",
        "revision": "316334ccc4a87b9b5635ad73da99b4dfc0b3887e",
        "dataset": "internlm/WildClawBench",
        "method": ("delegated to eval/run_batch.py (OpenClaw 2026.3.11 in "
                   "wildclawbench-ubuntu:v1.3); the external services the harness calls are on "
                   "the Gemini API only: judge = gbench Gemini cascade (the run's list passed "
                   "into the orchestrator as GBENCH_JUDGE_MODELS, base.gemini_cascade_env(); "
                   "every judge request forwarded with the task's own fields but the model, "
                   "temperature included, none when the task sends none, as in "
                   "03_Social_Interaction_task_2 and _3; a judge call that turns thinking off "
                   "(extra_body thinking {type: disabled}: 06_Safety_Alignment tasks 2, 3 and "
                   "6) sent with reasoning_effort low, Gemini thinking level low, the lowest "
                   "level every model of the default cascade accepts, since none can turn "
                   "thinking off (the same request to every model; counted in gemini_usage "
                   "thinking_off, recorded in judge_thinking_off, named in the reason); a model "
                   "the Gemini API answers HTTP 404 for is skipped in that request and never "
                   "makes it a content refusal; the answering model recorded per task, per_task "
                   "gemini_models, and over the run, gemini_models_answered), "
                   "05_Creative_Synthesis_task_11's audio judge "
                   "(openai/gpt-4o-mini-audio-preview) and the OpenRouter helper model the "
                   "prompts of 25 tasks offer the agent = the same cascade, "
                   "04_Search_Retrieval_task_1 web-verification judge = Gemini Google Search "
                   "grounding, the agent's web_search tool = Gemini Google Search grounding "
                   "behind OpenClaw's unchanged brave tool (Brave API origin rewritten to the "
                   "orchestrator proxy, preflight-checked; no Brave key read); the image's "
                   "other search skills (local-web-search and web-search on DuckDuckGo, "
                   "google-search on Google Custom Search) deleted from every task container, "
                   "so web_search is the agent's only search tool; the agent's model calls go "
                   "through the proxy's model relay, which adds the model endpoint's key (task "
                   "containers hold only placeholders; the proxy listens on the task "
                   "containers' address, port 18795 by default); recorded and not "
                   "leaderboard-comparable; re-checked 2026-09-26 against the pinned commit "
                   "(the task ## Env and ## Skills declarations, docker_utils/grading and the "
                   "openclaw runner order, the task_1 grade(), and the OpenClaw 2026.3.11 "
                   "bundle, ports and skills in wildclawbench-ubuntu:v1.3); model-free pipeline "
                   "verified 2026-09-26 with stub model/search/Gemini; task containers on "
                   "docker's default bridge network (the upstream default; "
                   "GBENCH_WILDCLAWBENCH_TASK_NETWORK), checked for the cloud metadata server "
                   "before the run with container_guard.require_metadata_blocked (probe of the "
                   "task image on the task network; reachable or no verdict is infra_required "
                   "naming the host firewall rule); the proxy, on the host network, relays to "
                   "the model URL as given; harness errors (provisioned input missing: "
                   "workspace directory, gt/ directory or a gt file grade() names, checked "
                   "before the run; web-search setup failure; backend failure before the agent "
                   "ran; never run; grading infra failure (a docker cp or loader failure, the "
                   "task container gone, every 120 s grade cap hit of a grade() exec run after "
                   "the leftover processes were stopped, or a cap hit or a Gemini service "
                   "failure of a judge call of a grade() exec whose stop of the leftover "
                   "processes failed for a reason the agent cannot cause: the docker exec timed "
                   "out or could not run, or docker failed while the container and its python3 "
                   "worked) or no numeric score; judge fallback; a grader judge call not served "
                   "that arrived during such a grade() exec; the task network losing DNS or web "
                   "egress (a network monitor container of the task image on the task network, "
                   "gbench's own script, every 15 s; a failed name lookup in the task's logs "
                   "counts only with a monitor round in its window where at least half of the "
                   "names did not resolve; a TCP timeout or one name failing confirms nothing); "
                   "an agent helper-model call failed on the service side (a Gemini refusal of "
                   "the agent's own request is its outcome, no event); an agent web search not "
                   "served; a model-endpoint failure no later relay call of the same task got "
                   "past), attributed by task time window and the requesting container's "
                   "address (grader calls told apart by a per-run grader placeholder key), are "
                   "left out of the headline, counted per kind, status completed_with_errors; "
                   "every process the agent left in a task container stopped before its grading "
                   "starts (the processes that ran before the OpenClaw gateway, e.g. the 03_* "
                   "mock services, kept; grade() unchanged); the metadata guard also resolves "
                   "the tasks' host names (resolve=TASK_HOSTS) and the monitor's first round "
                   "must show DNS and egress; orchestrator label "
                   "org.gbench.wildclawbench-harness-errors=9; the proxy sends each chat "
                   "request's max_tokens / max_completion_tokens, and the grounded path's "
                   "maxOutputTokens, plus the Gemini thinking allowance 8192, returns the "
                   "grounded reply's real finish reason, and counts the replies still cut at "
                   "the raised cap in gemini_usage, named in leaderboard_comparable_reason "
                   "(merge_shards sums gemini_usage over the shards and names the cut replies "
                   "once over the sum); the proxy listens on 127.0.0.1 "
                   "or a task-network gateway only (GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT "
                   "removed); the model relay never follows redirects; GEMINI_OPENAI_BASE_URL "
                   "forwarded only as base.require_gemini_openai_base_url accepts it; OpenClaw "
                   "ports 18789-18793 and 18800-18899 refused for the proxy under the host "
                   "network; model-free pipeline re-verified 2026-09-26 (stub endpoint 503 -> "
                   "harness error); other grader-key events are counted in "
                   "grader_events_not_counted; grade() cap hits are counted in grading_timeouts "
                   "(with_judge_call / without_judge_call / unrecorded / not_isolated_infra), "
                   "and a cap hit or judge failure of an exec whose stop failed for a reason "
                   "the agent can cause (processes left, python3 missing or broken) is scored "
                   "as upstream and counted (not_isolated); a task graded without a verified "
                   "stop names its cause (grading_isolation.stop_failure_causes) and makes the "
                   "status completed_with_errors; log-only name lookups are scored and named "
                   "(name_resolution_errors_unconfirmed); model-free end-to-end run verified "
                   "2026-09-27 (03_Social_Interaction_task_2, stub model and loopback stub "
                   "Gemini: isolated, 2 processes stopped, 2 monitor rounds) and re-run on the "
                   "harness-errors=4 image (isolated, 2 processes stopped, 2 monitor rounds, "
                   "the stub's reply scored 0 by grade() with no judge call, status success); "
                   "the image's proxy checked against a loopback stub Gemini (a model answering "
                   "HTTP 404 skipped, the grader's reply recorded with the model that answered, "
                   "no temperature added to a request that sent none, an agent request every "
                   "served model refused answered HTTP 400 with no harness event); the "
                   "harness-errors=5 image's proxy checked 2026-09-27 against a loopback stub "
                   "Gemini (a grader judge request that turns thinking off sent with "
                   "reasoning_effort low, its temperature 0 and max_tokens 1000 kept, to each "
                   "cascade model tried; a plain one sent without; thinking_off counted by "
                   "kind); host-network mode checks that the OpenClaw ports 18789-18793 and "
                   "18800-18899 are free on 127.0.0.1 (infra_required otherwise); GBENCH_SHARD "
                   "is validated host-side and forwarded normalized, the launcher exits on a "
                   "malformed spec and writes an empty-shard summary for a shard with no "
                   "selected task (accuracy None); merge_shards recomputes global_mean, the "
                   "modality means and weighted_overall from raw_summary.per_task; "
                   "max_output_tokens_note (OpenClaw's own setting); attempts not_applied; "
                   "GBENCH_WILDCLAWBENCH_MODEL_API_KEY falls back to base.endpoint_api_key; "
                   "--eval-limit is a seeded draw balanced across the six categories (the "
                   "launcher's copy of stratified_indices, seed wildclawbench); a limited run "
                   "whose summary lacks selection.limit_draw is infra_required (rebuild)"
                   "; an agent web_search gets a 57 s deadline inside the 60 s web_search timeout "
                   "gbench sets (OpenClaw's default is 30 s; a recorded deviation, 2026-10-13 A/B: "
                   "94.8% answered at 57 s against 65.2% at 27 s); a lookup not answered by then, an answer not delivered in time, and "
                   "a web_search call OpenClaw aborted in the transcript (web_search_aborted) are"
                   " agent_web_search harness errors; a timed-out or failed stop of the leftover "
                   "processes is classified by docker top; the model relay records "
                   "max_tokens_sent, replies_cut_at_max_tokens and endpoint token usage (null, "
                   "never 0, when unreported); per-task artifacts kept under "
                   "wildclawbench_artifacts"),
    },
    "skillsbench": {
        "synced": "2026-10-13",
        "upstream": "github.com/benchflow-ai/skillsbench",
        "revision": "9a1f4dd5f7659f75707435da3ce854b6e48321d1",
        "method": ("delegated to BenchFlow 0.6.3 (deterministic verifier); oracle pipeline "
                   "verified; the orchestrator image must carry label "
                   "org.gbench.skillsbench-harness=7 (its launcher applies GBENCH_SHARD, exits "
                   "on a malformed one, records n_tasks_full and tasks_selected, and runs the "
                   "with-skills condition with --skill-mode with-skill; gbench validates the "
                   "shard and forwards the normalized I/N), else infra_required naming the "
                   "build command; the metadata server gate also checks that the verifiers' and "
                   "agent install's package hosts resolve (swebench_common.TASK_INSTALL_HOSTS). "
                   "A reward-0 task whose verifier could not resolve an install host before its "
                   "tests ran (the rollout's verifier/test-std*.txt), when another such task in "
                   "the run or a fresh probe container of the orchestrator image after the run "
                   "that cannot resolve one of the hosts either corroborates it "
                   "(swebench_common.corroborated_dns_failures), and a task with no reward, is "
                   "a harness error: left out of its condition's mean and n_scored, listed in "
                   "harness_error_tasks, counted in harness_errors, run completed_with_errors. "
                   "A run with nothing left to score is infra_required. An uncorroborated DNS "
                   "failure (listed in dns_failures_kept) or a transcript naming resolver files "
                   "keeps the reward; per (condition, task) sample_traces and merge_shards; "
                   "thinking/temperature/max_output_tokens recorded as owned by the agent and "
                   "not applied; the endpoint key is GBENCH_SKILLSBENCH_API_KEY, else "
                   "VLLM_API_KEY, else OPENAI_API_KEY, added by the relay; draws an "
                   "--eval-limit subset with a seeded draw balanced across the task categories "
                   "(task.md metadata.category; selection.limit_draw)"),
    },
    "mcp_bench": {
        "synced": "2026-10-09.2",
        "upstream": "github.com/Accenture/mcp-bench",
        "revision": "7a8eaeae83a842a2949080acc5473f65e1569daf",
        "method": ("delegated in-container on the README's Python 3.10 (conda create -n "
                   "mcpbench python=3.10; Ubuntu 22.04 python3 is the only interpreter, tzdata "
                   "for a configured timezone) (upstream runner + 28 servers built by every "
                   "step of mcp_servers/install.sh, its one set -e line turned off so the steps "
                   "after metmuseum's failing tsc run, a separate build step failing on "
                   "install.sh drift; MCP client pinned to mcp==1.16.0, the release upstream's "
                   "mcp[cli]>=1.9.0 resolved to on the pinned commit's date 2025-10-07, with "
                   "which utils/collect_mcp_info.py reports 28/28 (unpinned 2.2.0, and the "
                   "newest 1.x with fastmcp 3.4.7, each leave servers without tools); "
                   "commands.json unmodified, every python server on the one interpreter, uv "
                   "run servers on their uv.lock environments with "
                   "UV_PYTHON_PREFERENCE=system); Gemini-cascade judge with upstream prompts "
                   "and 5x stability, each request carrying upstream's {model, messages, "
                   "max_tokens} (no temperature; upstream's max_completion_tokens switch is "
                   "keyed on Azure OpenAI model names and does not apply), one attempt per "
                   "model (OpenAI client max_retries=0) with a model the API answers HTTP 404 "
                   "for skipped for the rest of the call, the run's cascade passed in as "
                   "GBENCH_JUDGE_MODELS (base.gemini_cascade_env()) and the answering Gemini "
                   "model recorded per task (judge_models_per_task, gemini_models_answered), "
                   "sent through gbench's Gemini relay (gemini_relay.GeminiRelay on the "
                   "container network's gateway, forwarding only POST "
                   "/v1beta/openai/chat/completions), which adds and rotates the keys: the "
                   "container, where model-driven servers run model text (sympy parse_expr, the "
                   "OSINT server's commands), holds only the relay's per-run placeholder, never "
                   "a Gemini key (image label org.gbench.mcp-bench-harness=3, else "
                   "infra_required); the five third-party server keys (GOOGLE_MAPS_API_KEY, "
                   "NCI_API_KEY, HF_TOKEN, NPS_API_KEY, NASA_API_KEY) go only to a "
                   "credentialed-servers container of the same image (docker run -d --init "
                   "--entrypoint sleep on a docker network of its own, checked by "
                   "container_guard.require_metadata_blocked with the credentialed servers' API "
                   "hosts, pypi.org and files.pythonhosted.org resolved; no port, volume or "
                   "service, none of the servers that run model text), which runs the five "
                   "servers that need them (Google Maps, BioMCP, Hugging Face, National Parks, "
                   "NASA Data) each session one docker exec from gbench's relay (a thread of "
                   "gbench on the model network's gateway, a per-run token, fixed session "
                   "kinds, gbench/docker/mcp_bench_credentialed.py); the model container's "
                   "commands.json entries for them start gbench's bridge "
                   "(gbench/docker/mcp_bench_bridge.py) and its environment holds no server key "
                   "(its entrypoint refuses one); the servers' tools, requests and replies are "
                   "upstream's, key values masked in replies (checked model-free 2026-09-27: "
                   "28/28 connected with tools, the five bridged catalogs identical to the "
                   "servers started directly); the relay answers a session only once the "
                   "credentialed side says it is up (the stdio server's process started, the "
                   "HTTP server accepts connections, the connection open; ERR otherwise, within "
                   "30 s, 150 s for the HTTP server's start) and frames what comes back, ending "
                   "with a frame that carries docker exec's exit code; a credentialed-server "
                   "session that failed outside the model (not opened, the stdio server exiting "
                   "before it answered, the HTTP server not ready, or lost or ended by a docker "
                   "exec failure while its client still used it) is recorded by the bridge and "
                   "is a harness error of the task (credentialed_relay), and a server exiting "
                   "on its own after it answered is its own outcome, as upstream; the relay's "
                   "failures during the run (kept with their times) and a credentialed-servers "
                   "container not running when gbench stops it are a harness error of the run "
                   "(harness_errors kind credentialed_relay, credentialed_servers_run, status "
                   "completed_with_errors); checked model-free 2026-09-27 on the harness=3 "
                   "image (28/28 connected with tools; a live stdio session to National Parks "
                   "answered initialize, stopping the credentialed container mid-session gave "
                   "exit 137 in the end frame and a relay failure, the next session was refused "
                   "with the docker daemon's reason, and the container was recorded as not "
                   "running); all 104 tasks (upstream's task files as they are): before the run "
                   "the image's launcher must report every server built and the network up "
                   "(every server API host name of SERVER_API_HOSTS resolves in the container "
                   "and a TCP connection to one of them opens; no IP-only test), the 5 "
                   "README-required keys must be set and each accepted by a live request to the "
                   "service its server calls (server_key_problems, in gbench's process before "
                   "any container: NPS /api/v1/parks, NASA /neo/rest/v1/stats, Hugging Face "
                   "/api/whoami-v2, NCI /api/v2/trials, Google Maps once per Maps Platform API "
                   "its server calls; a refusal, or no answer of the service's own after 3 "
                   "attempts, is infra_required; recorded as server_key_check), and the "
                   "connection check must report 28/28 servers connected, each listing tools, "
                   "else infra_required (no bypass); when the run starts the launcher checks "
                   "again and runs no task if a server cannot serve (infra_required); a "
                   "distraction count other than 10 or stability off is named in "
                   "protocol_overrides; credentials and the endpoint URL passed to docker by "
                   "name, key values redacted from server errors before any cut; "
                   "model-under-test requests as upstream builds them (llm/provider.py "
                   "get_completion), with the run's thinking switch "
                   "(chat_template_kwargs.enable_thinking), no temperature unless the operator "
                   "sets one, upstream's caps exactly without --thinking (planning rounds and "
                   "the final answer 12000, summaries 10000, format fixes 8000, compression its "
                   "target length) and with --thinking the run's budget fitted to the context "
                   "window and a request timeout sized to it, base.request_timeout_s, in place "
                   "of the openai client's default 600 s read timeout "
                   "(gbench/docker/mcp_bench_request_policy.py; image label "
                   "org.gbench.model-request-policy=2, else infra_required); the applied "
                   "settings checked and the replies counted (model_replies); a model request "
                   "that ends in an endpoint or connection error is a harness error, never the "
                   "model's outcome (model_replies.failed_requests, timed_out, "
                   "model_failed_by_error, harness_errors, status completed_with_errors), and "
                   "window rejections left to upstream's handling (rejected_over_context) and "
                   "in-flight cancellations such as at task_timeout 5000 s (cancelled) are "
                   "counted and named; images run with docker run --pull never on docker's "
                   "default bridge network (GBENCH_MCP_BENCH_NETWORK; a model endpoint on this "
                   "host reached at the network's gateway), checked for the cloud metadata "
                   "server before the pre-run check with "
                   "container_guard.require_metadata_blocked, whose probe container also "
                   "resolves the servers' API host names and a named model endpoint (the model "
                   "drives tools that run nmap/whois/dig on targets it names and sympy "
                   "parse_expr on expressions it writes; reachable, no verdict or a name that "
                   "does not resolve is infra_required naming the host firewall rule, so a DNS "
                   "outage is found before the run); a run in which no model request got a "
                   "reply publishes no score (infra_required); each task a harness error fell "
                   "in (the last of upstream's 3 get_completion attempts of a call failing on "
                   "the endpoint, attributed per call; no judge evaluation, which upstream's "
                   "aggregator would skip; a resolver failure naming a server API host or the "
                   "container's resolver, or a failed lookup of a server API host, in the "
                   "task's error or a tool's error reply, never in a successful tool result; a "
                   "DNS outage the launcher's 30 s watchdog saw while it ran; a "
                   "credentialed-server session that failed outside the model; a task upstream "
                   "recorded as failed or that raised) is left out of the averages upstream's "
                   "aggregator computes (gbench/docker/mcp_bench_run.py HarnessAccounting, "
                   "recorded in /out/harness_log.json, whose absence is infra_required), "
                   "counted per kind in tasks_unscored and harness_errors, named, status "
                   "completed_with_errors, files weighted by their scored tasks; judge calls of "
                   "scored tasks that failed on every cascade model are a partial judge outage, "
                   "a harness error of the run (harness_errors kind judge_outage_partial with "
                   "the task ids, status completed_with_errors; upstream's average of the "
                   "surviving stability runs kept); a run with no task left to score publishes "
                   "no score (infra_required); Overall = mean of 4 dims, schema understanding = "
                   "mean(valid tool name rate, schema compliance, execution success) per "
                   "arXiv:2508.20453 Table 3 (reproduces 18/20 printed rows); the in-container "
                   "judge sends a relay placeholder only to the relay's URL and, with a real "
                   "key, refuses a GEMINI_OPENAI_BASE_URL that is not the Gemini API over https "
                   "or a loopback address (copy of base.check_gemini_openai_base_url); a model "
                   "endpoint URL with a user name or password is refused (infra_required), "
                   "since the URL would sit where model text runs; GBENCH_SHARD is validated "
                   "host-side and forwarded normalized, and the launcher exits on a malformed "
                   "spec; an empty shard gives accuracy None; execution sequential; attempts "
                   "not_applied; no merge_shards (gbench --merge refuses its shards); "
                   "MCP_TASK_LIMIT is a seeded draw of the shard's task ids balanced across the "
                   "server groups (subset_manifest.limit_draw, selected_task_ids); a limited "
                   "run without limit_draw is infra_required"
                   "; the bridge records no credentialed_relay failure for a connection that "
                   "failed while the credentialed HTTP server was exiting on its own "
                   "(EXIT_SETTLE_S=2 s), so such a task is scored; tasks left out on the bridge "
                   "record alone are counted (unconfirmed_bridge_failures) and named; the image "
                   "build fails naming every commands.json server that is not ready "
                   "(mcp_bench_build_check.py)"),
    },
    "toolbench": {
        "synced": "2026-10-10.3",
        "upstream": "github.com/THUNLP-MT/StableToolBench",
        "revision": "aa4ed9f4737ad98bd706663f01d63623c3427812",
        "dataset": ("StableToolBench solvable_queries (at revision) + "
                    "stabletoolbench/Cache@c6aca0c1cda74d54915eccabb60984dc083fb14c "
                    "server_cache.zip (sha256 "
                    "a3ed72052c2dc961d8e9b71860374d90ded6d475ca6f6449c78fb248cb82a5b2); SoWR "
                    "reference = "
                    "stabletoolbench/baselines@ea70398f955e7b58f8807187936818e794ce5140 "
                    "data_baselines.zip (sha256 "
                    "c3c755f68ccd006788d247eb73d644b4c4ea13620ea4702621d921958a80b79c), "
                    "chatgpt_cot"),
        "method": (("delegated qa_pipeline DFSDT in a local image; cached server with "
                    "upstream's server/config.yml (uncached calls go to the ToolBench real-API "
                    "proxy with the README's empty key, then the simulator; is_save true), the "
                    "simulator's model (gpt-4-turbo) replaced by the Gemini cascade with "
                    "upstream's request (temperature 0, json_object, 3 JSON tries) and "
                    "max_tokens 1024 + the Gemini thinking allowance 8192 (simulator replies "
                    "still cut there, empty ones and outages after them counted in "
                    "simulator_usage and named); ToolEval as the scripts run it: template.txt prompts verbatim, the "
                    "config.yaml function (name, description, parameters; reason where the "
                    "scripts ask) forced through Gemini function calling (mode ANY), "
                    "config.yaml temperature 0.2 and max_tokens 1000 for the reply, sent as "
                    "max_output_tokens 1000 + the Gemini thinking allowance 8192 (replies still "
                    "cut there and outages after them counted and named), 3 tries per call (a call without a valid verdict is left out and "
                    "counted); SoPR per eval_pass_rate.py (Finish check, check_answer_status "
                    "then parse_answer_status on Unsure) over the group's test ids, 3 "
                    "evaluation passes (mean with its std); SoWR per eval_preference.py + "
                    "rtl.py normalized_openai_completions against the published ChatGPT-CoT "
                    "answers converted with convert_to_answer_format.py (a reference answer "
                    "required for every test id the run covers, else infra_required; processed "
                    "answers, the single Solved answer, else the single Unsure answer, else "
                    "select_better_answer; win = more passes than the reference; one vote per "
                    "test id the run covered: every test id, or under --shard / --eval-limit "
                    "the host-side selection (select_query_ids: (group, id) pairs sorted, shard "
                    "first, then a seeded draw stratified by group; the container runs exactly "
                    "those ids, TB_QUERY_IDS_FILE)); headline = unweighted Average over the 6 "
                    "groups; judge = Gemini cascade; grader and simulator on the Gemini API "
                    "only (the simulator on its OpenAI-compatible endpoint through gbench's "
                    "Gemini relay; no knob for another provider, base URL, key or model; both "
                    "substitutions recorded in gemini_substitutions; the cascade passed into "
                    "the container as GBENCH_JUDGE_MODELS); the container runs the model's tool "
                    "calls, so it runs on docker's bridge network after the metadata-server "
                    "guard and holds no real key: the Gemini relay's placeholder passed to "
                    "docker by name and unset in the container once the server config is "
                    "written, the model endpoint through gbench's endpoint relay (the endpoint "
                    "key, VLLM_API_KEY else OPENAI_API_KEY, added there; an endpoint URL with "
                    "user info refused); tool calls the cached server failed (HTTP 500: the "
                    "real-API proxy call, the simulator cascade or the server raised; each "
                    "/virtual call recorded with its answer type, the Gemini model that "
                    "simulated it, the failure stage and the query it belongs to, "
                    "toolbench_server_calls.jsonl) or the simulator could not answer (three "
                    "replies without valid JSON), found also in each query's whole search tree "
                    "(tool_observations.json), tool-call timeouts, queries with a DFSDT step "
                    "whose last try (of upstream's 6) failed at the endpoint "
                    "(the patched model client records each step with a request that "
                    "failed at the endpoint, with its group and query id, "
                    "model_request_failures.jsonl; a step a later try answered is counted in "
                    "model_request_failures and its query scored; a failed step tied to no "
                    "query is infra_required) and queries without an answer are harness errors: "
                    "those queries are left out of SoPR and SoWR (n_harness_error_queries, "
                    "harness_error_query_kinds), status completed_with_errors, and a group "
                    "whose every query is one publishes no score (accuracy null with "
                    "accuracy_withheld_reason, status failed, its records kept); the real-API "
                    "proxy call gets a 120 s read timeout (upstream sends none); ToolEval calls "
                    "without a verdict are harness errors, a SoWR comparison without one is no "
                    "vote, a query no pass could vote on is left out of the SoWR denominator "
                    "and a group with no vote publishes no SoWR (infra_required); "
                    "select_better_answer replies need only index (rtl.py reads no reason from "
                    "it); the answering Gemini models (judge_models_used, "
                    "simulator_models_used, gemini_models_answered), the cache / real-API / "
                    "simulator usage (simulator_usage) and one record per test id "
                    "(sample_traces) recorded; image label org.gbench.toolbench-harness=5, else "
                    "infra_required; images run with docker run --pull never; a model request "
                    "the endpoint rejected for its content (HTTP 400 or 422, a context-window "
                    "rejection of the --thinking budget included) is the model's outcome, as "
                    "runners/quality.py and gaia2 count it (base-run suites count every "
                    "endpoint rejection as a harness error): taken out of "
                    "model_replies.failed_requests and the harness errors, counted in "
                    "model_content_rejections and named, and the patched model client records "
                    "its step (content_rejections; a step that ended on one, steps_rejected, "
                    "leaves its query scored), except in a run in which no model request got a "
                    "reply; model-under-test requests as chat_completion_request builds them "
                    "(max_tokens 1024, frequency_penalty 0, presence_penalty 0, no "
                    "temperature), with the run's thinking switch, no temperature unless the "
                    "operator sets one, 1024 exactly without --thinking and with --thinking the "
                    "run's budget fitted to the context window and a request timeout sized to "
                    "it, base.request_timeout_s, in place of the openai client's default 600 s "
                    "read timeout (docker/toolbench_request_policy.py; image label "
                    "org.gbench.model-request-policy=2, else infra_required; replies counted in "
                    "model_replies; a request that ends in an endpoint or connection error is a "
                    "harness error, never the model's outcome: model_replies.failed_requests, "
                    "timed_out, model_failed_by_error, harness_errors, status "
                    "completed_with_errors; window rejections left to upstream's handling "
                    "counted as rejected_over_context; a run in which no model request got a "
                    "reply publishes no score, infra_required); image packages pinned by "
                    "docker/toolbench_constraints.txt; merge_shards recomputes per-group "
                    "SoPR/SoWR from the per-query records and sums the shards' Gemini model "
                    "records (judge_models_used, simulator_models_used, "
                    "gemini_models_answered), harness errors and counts; --attempt-count "
                    "recorded as not applied")),
    },
    "multipl_e": {
        "synced": "2026-10-09",
        "upstream": "github.com/nuprl/MultiPL-E",
        "revision": "3025a531af7450e7df8b96fe0440e9804480bbad",
        "dataset": ("nuprl/MultiPL-E@28441b6024e71d4a1c1c0f6bf171c935cd5a43f2 (humaneval-<lang>, "
                    "23 languages)"),
        "checksum": ("sha256:604b7afc237ae416e02c97166759b3a56667ad76bd214f6799fcd1d3cfbef29f "
                     "(dataset_builder/chat_completions.py @3025a531, fetched at run time); "
                     "evaluator: evaluation/Dockerfile "
                     "sha256:149513ea171ed19be5ce2e9d0f038f91884f6f996c332afc835518537687ff2d, all"
                     " 72 files of evaluation/ pinned in docker/multipl_e.sha256 and checked at "
                     "image build"),
        "method": ("chat protocol of dataset_builder/chat_completions.py, fetched from the pinned "
                   "commit at run time (sha256-checked, cached; not shipped, BSD-3 with ML "
                   "restriction): Completion signature text, stop_at_stop_token and "
                   "postprocess_completion read from it; DSPy 3.0.0b1 ChatAdapter messages "
                   "verbatim (byte-equal to what the real script sends) + ChatAdapter.parse port "
                   "(DSPy MIT notice carried); a reply parse rejects (whitespace-only included) "
                   "scores completion \"\" as upstream's do_task records it (the script's aforward "
                   "path has no JSONAdapter fallback in DSPy 3.0.0b1; ChatAdapter.acall came in "
                   "dspy 1df67689, after tag c3e79ce8); an empty reply (raw content empty: DSPy "
                   "3.0.0b1 Adapter._call_postprocess tests `if text:`), on which upstream's "
                   "script stops, or an unanswered request makes the run non-comparable (a request"
                   " that got no response from the served endpoint is a harness error: left out of"
                   " accuracy, category_accuracy and attempts, request_harness_errors, status "
                   "completed_with_errors); upstream chat recipe by default: temperature 0.2, no "
                   "top_p; max_tokens by the token-limit policy from OFFICIAL_SAMPLING.max_tokens "
                   "2048 (the script's one mode, --max-tokens default): 2048 without --thinking "
                   "(--max-output-tokens not applied), the run's budget with it (official cap not "
                   "applied to reasoning); a reply that stopped at a limit other than 2048 (with "
                   "--thinking the run's budget, without it a max_tokens the context window "
                   "lowered) is non-comparable (replies_cut_off_by_budget, "
                   "replies_cut_off_in_reasoning), 20 completions/problem (official_sampling "
                   "recorded); evaluator image built from MultiPL-E's own evaluation/ at 3025a531,"
                   " downloaded and sha256-checked at build time (docker/multipl_e.Dockerfile; its"
                   " javatuples download moved to Maven Central, sha256-checked, because "
                   "repo.mavenlibs.com now serves HTML; its FROM pinned by digest and a record "
                   "step appended); pass@1 as pass_k.py computes it (per "
                   "problem c/n over its scored completions, then the mean over problems; each "
                   "language's category_accuracy and the attempts' avg_at_k the same way); a "
                   "completion the evaluator returned no result for while it scored others is a "
                   "harness error (scoring_excluded, harness_error kind no_harness_result, "
                   "multipl_e_unscored): left out of the score, status completed_with_errors; when"
                   " it scored nothing every completion is one (kind no_results), accuracy null "
                   "and status error; any other sampling field (decoding penalties), harness error"
                   " or unscored completions -> non-comparable (infra_required)"
                   "; per-problem pass@1 is keyed by the problem id (problem_id = "
                   "'<lang>/<name>'); merge_shards recomputes it over the shards' completions; "
                   "--shard comes before the --eval-limit draw"),
    },
    "swe_bench_pro": {
        "synced": "2026-10-11.2",
        "upstream": ("github.com/scaleapi/SWE-bench_Pro-os v2/ (642 Harbor tasks, HARD-51 ids, "
                     "locked-protocol tooling) run with Harbor 0.22.0 (PyPI; the v2/GATE.md "
                     "version) in the local image gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt"),
        "revision": "66f92766bba642462d4bbe5479e83f91f9211862",
        "dataset": ("ScaleAI/SWE-bench_Pro@2d52cb3df914a3fcf80c7f66738b3a88ae37fc50 (config "
                    "default = SWE-bench Pro V2, test, 642; config hard = HARD-51; ids equal "
                    "v2/tasks and v2/hard51_ids.txt); v1 (731 tasks, the HF tag v1.0) "
                    "@307bd006a0a3807424d3c5525eaf98ca8e63aed3"),
        "checksum": ("sha256:1067f72c1f43bbc578276888625db3b836cebb535546c9eaf824c5d5dd104a7f "
                     "(data/default/test-00000-of-00001.parquet at the revision; hard: "
                     "sha256:abe9b2280ae470ae5742698979855e40ecc5c83c7aa6c679afb4ee964e951d3c)"),
        "method": ("V2 locked protocol: Harbor agent job locked_mini_swe:LockedMiniSwe "
                   "(mini-swe-agent 2.4.6, v2/tooling/configs/mini_toolcall.yaml sha256 7275afa9, "
                   "agent phase no-network + --allow-agent-host, task.toml 3000 s budgets) then "
                   "the patch_replay:PatchReplayAgent re-grade on pristine images = headline; "
                   "in-sandbox rate and HARD-51 (eval.yaml SWE_Bench_Pro_hard) reported too; "
                   "re-grade verdicts as in Harbor 0.22.0 (a reward is the verdict; a verifier "
                   "that started and raised VerifierTimeoutError/RewardFileNotFoundError/RewardFil"
                   "eEmptyError/VerifierOutputParseError scores 0: the verifier members of "
                   "models/job/config.py RetryConfig.exclude_exceptions, and metrics/base.py "
                   "counts a trial without a reward as 0); one infrastructure retry pass for tasks"
                   " that never reached the model and for re-grades that failed outside the "
                   "verifier (v2/GATE.md re-ran only sandbox errors); a task still without a "
                   "scoreable trial after it is a harness error (agent_infra when its rollout was "
                   "not the model's own outcome, no_regrade_verdict when its re-grade gave no "
                   "verdict; the one-shot mode's unregraded tasks too): left out of the numerators"
                   " and denominators of the headline, the in-sandbox rate (over the tasks whose "
                   "rollout was the model's outcome) and HARD-51, counted in harness_errors, "
                   "completed_with_errors, and the headline withheld (accuracy null, status error)"
                   " when no task was scored; no opt-in variable; tree checked against "
                   "v2/SHA256SUMS at image build; job arguments checked with harbor 0.22.0 "
                   "--print-config; locked config unchanged except "
                   "chat_template_kwargs.enable_thinking; GBENCH_SWEBENCH_PRO_AGENTIC=1 routes to "
                   "the maintained engine (v1 harness, tag v1.0 raw-sample table, labelled a v1 "
                   "number); GBENCH_SWE_BENCH_PRO_MODE=oneshot re-graded by the same patch_replay "
                   "job with the same verdict and retry rule; the agent in each sandbox gets "
                   "OPENAI_API_KEY=EMPTY as a literal harbor argument (no shell expansion); the "
                   "sandboxes reach the endpoint only through gbench's relay, started for every "
                   "run (swebench_common.EndpointRelay listening on "
                   "GBENCH_SWE_BENCH_PRO_TASK_HOST; it forwards only GET/POST requests for "
                   "/v1/chat/completions, /v1/completions and /v1/models to the fixed endpoint, "
                   "refusing any other path, builds each target with urlunsplit from the "
                   "endpoint's scheme, host and port, follows no redirects, and adds "
                   "Authorization: Bearer when the endpoint has a key, recorded under "
                   "harness.key_forwarder, and nothing otherwise, recorded under "
                   "harness.endpoint_relay), so no key enters Harbor, a sandbox or a command line,"
                   " and no request can re-target the relay or read the key back; the endpoint "
                   "failures the relay counts (relay on the result) are harness errors, never the "
                   "model's outcome (endpoint_failures, completed_with_errors, not comparable); a "
                   "rollout the agent budget cut off (mini-swe-agent exit status empty) is its "
                   "patch's verdict, as upstream counts it, unless its agent/mini-swe-agent.txt "
                   "ends, after the last model reply, in tenacity's 'Retrying ... as it raised "
                   "<model-endpoint exception>' record (mini-swe-agent 2.4.6 "
                   "models/utils/retry.py) and the relay counted an endpoint failure between the "
                   "rollout's last trajectory event and the end of its agent phase "
                   "(budget_kill_verdict): that rollout is an agent_infra harness error; every "
                   "cut-off trace records budget_kill; Harbor's own label of the agent trial's "
                   "exception is recorded as agent_harbor_exception and not used (Harbor's "
                   "_classify_exec_error reads it from the agent's output, which carries the "
                   "model's command output); the metadata server gate before any Harbor job "
                   "(require_sandbox_network_blocked) also checks that the hosts every sandbox "
                   "installs the agent from resolve (SANDBOX_INSTALL_HOSTS: deb.debian.org, "
                   "security.debian.org, archive.debian.org, archive.ubuntu.com, astral.sh, "
                   "github.com, pypi.org, files.pythonhosted.org; LockedMiniSwe.install's apt, the"
                   " astral.sh uv installer and uv tool install --python 3.12 "
                   "mini-swe-agent==2.4.6; five teleport verifiers also apt-get install); the v1 "
                   "route's gate checks the metadata server only; budget_kill_verdict needs an "
                   "endpoint-side relay failure (no reply, 5xx, stream error, unparsable 2xx; "
                   "never a 4xx) of a request carrying the rollout's own conversation "
                   "(conversation_fingerprint of the first user message, recorded in "
                   "budget_kill), with no reply to that conversation after it; the retry log "
                   "line alone decides nothing; one-shot applies --shard before its stratified "
                   "limit (presharded); the endpoint key is base.endpoint_api_key; an empty "
                   "shard gives accuracy None"
                   "; a rollout whose in-sandbox verifier failed for an infrastructure reason is "
                   "left out of the auxiliary in_sandbox rate (in_sandbox_verifier_infra); the V2"
                   " run always reaches the endpoint through the relay"),
    },
    "complexfuncbench": {
        "synced": "2026-10-11",
        "upstream": "THUDM/ComplexFuncBench",
        "revision": "c37b284e2f2e03ee456115b7c4b7e537f534be37",
        "dataset": ("THUDM/ComplexFuncBench ComplexFuncBench.jsonl at Hub commit "
                    "5dc7739343f8c87aee465931ba66db13a0fb5dd4 (HF_REVISION; the Hub now redirects "
                    "the repository to zai-org/ComplexFuncBench; the file was uploaded once, at "
                    "558831035a on 2025-01-17, and is identical at every later commit; 1000 rows),"
                    " its sha256 checked at load (a file that differs -> infra_required); recorded"
                    " as dataset and dataset_sha256"),
        "checksum": ("sha256:be1e0f5951b666f6543b946e915a431a39bbda1807481878eb331245276ac088 "
                     "(ComplexFuncBench.jsonl at HF_REVISION, its Git LFS oid)"),
        "method": ("in-process ComplexEval port (golden-driven loop; CompareFC rule_based -> "
                   "value_checker -> response_based -> llm_based, as upstream's "
                   "compare_single_call; response_based sends both calls to the live booking-com15"
                   " API on RapidAPI as upstream's utils/rapidapi.py RapidAPICall does "
                   "(utils/tool_info.json vendored, sha256 "
                   "2c106f906171447e1a880771db241a5275c75bb3bb82bccb2f49e6260135bad9; requests.get"
                   " with upstream's headers and parameters, legs as JSON; each call retried 3 "
                   "times 2 s apart while it gets no response; a 60 s request timeout where "
                   "upstream sets none), with RAPID_API_KEY read from the environment and never "
                   "written (unset, or refused by a live Test_API check -> infra_required; the "
                   "tier is never replaced by the judge); a pair where a call's last reply was an "
                   "outage (network error, timeout, HTTP 5xx/401/403/407/429) and neither call got"
                   " an answer of its own is judged as upstream and, when not equivalent, is a "
                   "harness error rapidapi_unavailable; recorded as call_match_tiers, "
                   "response_based_tier and per-sample response_based_calls); both GPT-4o judge "
                   "roles (gpt-4o-2024-05-13 call equivalence, gpt-4o-2024-08-06 response "
                   "completeness/correctness) on the Gemini API judge cascade with upstream's "
                   "prompts/compare.py and prompts/response.py verbatim (system prompt as the "
                   "Gemini system_instruction, the user template filled as SimpleTemplatePrompt "
                   "fills it, temperature 0.0, as models/gpt.py GPTModel._predict sends them; "
                   "PROMPT_SHA256), replies read with upstream's decode_json, the response judges "
                   "asked up to 10 times for a 0/1/2 score as runner/response_runner.py "
                   "retry(max_attempts=10) does, recorded as judge_substitution/judge_prompts/judg"
                   "e_cascade/judge_models/gemini_models_answered/judge_outages; model requests as"
                   " models/qwen.py (temperature 0.0, tool_choice auto, max_tokens 2048 through "
                   "the token-limit policy: 2048 without --thinking, the run's budget with it; "
                   "official_sampling, max_tokens_sent, max_output_tokens_note), retried 5 times "
                   "20 s apart as its retry(max_attempts=5, delay=20); per-sample records in "
                   "sample_traces (upstream process_example's id, gen_convs, message, count_dict "
                   "and resp_eval, plus every judge call with the Gemini model that answered it "
                   "and its reply, and each model turn's finish_reason, completion tokens and "
                   "max_tokens); total_questions and effective_n are the evaluable samples, "
                   "correct_answers the successes; bge-large-en-v1.5 alignment via transformers "
                   "AutoModel (CLS+L2) at the checkpoint's float32 where upstream's FlagModel uses"
                   " fp16 (use_fp16=True; recorded as embedding_dtype and embedding_upstream) and "
                   "scipy linear_sum_assignment; harness errors are never the model's outcome: a "
                   "sample whose model endpoint request failed on all 5 attempts (upstream's "
                   "unknown_error), whose Gemini call-equivalence judge gave no verdict (an "
                   "outage, or a reply decode_json cannot read as an object with is_equal, where "
                   "upstream's llm_based returns None and the call counts as not equivalent), "
                   "whose response-based pair the booking-com15 backend could not answer and the "
                   "judge then found not equivalent (rapidapi_unavailable) or whose driving raised"
                   " is left out of every metric on full and subset runs (Success Rate over the "
                   "evaluable samples, n_scored, which is upstream's /1000 on a full run without "
                   "harness errors), a completeness/correctness judge that gave no score in its 10"
                   " requests is left out of its mean (upstream's -1), and each is counted in "
                   "harness_errors, makes the status completed_with_errors and is named in "
                   "leaderboard_comparable_reason, which also names an --eval-limit subset, the "
                   "embedder's float32, a temperature other than 0.0, a per-turn budget other than"
                   " 2048, the pairs the RapidAPI backend could not answer and the pairs whose "
                   "predicted call answered an empty object (upstream replaces the step's error "
                   "list with a string there; gbench keeps the list); a model reply with neither "
                   "content nor tool calls is a scored failure (empty_model_message, "
                   "n_empty_model_message; upstream files it under unknown_error); "
                   "leaderboard_comparable=False; --shard is applied to the 1000 rows before the "
                   "stratified --eval-limit; traces carry sample_key and scoring_excluded; "
                   "merge_shards recomputes ComplexEval over the union of the per-sample "
                   "records; an empty shard gives accuracy None; attempts not_applied"),
    },
    "gaia": {
        "synced": "2026-10-08",
        "upstream": ("GAIA paper arXiv:2311.12983 + leaderboard space "
                     "gaia-benchmark/leaderboard@9f133d71362e (question_scorer, submission "
                     "format, per-level task counts); loop limits from huggingface/smolagents "
                     "open_deep_research run_gaia.py@227ef5e"),
        "revision": ("682dd723ee1e1697e00360edccf2366dc8418dd9 (gaia-benchmark/GAIA), "
                     "9f133d71362e77b3539f1514f31b9c101a545fec (gaia-benchmark/leaderboard space)"),
        "dataset": "gaia-benchmark/GAIA (2023_all: validation 165, test 301)",
        "method": ("official system prompt + question with the task file inlined "
                   "(images/audio/video as media parts, .m4a decoded to wav, documents rendered "
                   "to text); web_search tool loop, 20 tool-enabled model turns then forced "
                   "final; each turn sends run_gaia.py's max_tokens=4096 without --thinking and "
                   "the run's budget with it (the 4096 is not applied to reasoning; run_gaia.py's "
                   "reasoning_effort high and max_completion_tokens 8192 are keyed on its o1 "
                   "model_id, not a reasoning-model setting for a served model; no "
                   "reasoning_effort is sent in either mode); no temperature sent (GAIA "
                   "prescribes none; the reference agent sends none); validation (default) scored "
                   "locally with the official question_scorer; GBENCH_GAIA_SPLIT=test runs the "
                   "leaderboard's test split and writes its submission JSONL (task_id, "
                   "model_answer, reasoning_trace), then ends infra_required (only the "
                   "leaderboard upload scores it; tasks that do not fit the served context window "
                   "are listed apart from failed requests); web_search = Gemini Google Search "
                   "grounding (search_tool cascade) in place of the reference agent's search "
                   "tool: each call is one Gemini generateContent request with the google_search "
                   "tool and the model's query as the prompt, and it returns the cited sources' "
                   "titles and URLs with segments of that Gemini answer, no page text, as the "
                   "tool description the model sees (gaia.GROUNDED_SEARCH_DESCRIPTION, recorded "
                   "as search_tool_description), search_backend_kind and "
                   "leaderboard_comparable_reason say; every trace keeps its web_search calls "
                   "(tool_transcript: arguments, answering Gemini backend, the reply the model "
                   "read) in both splits; harness errors (a request with no reply: endpoint "
                   "error, timeout or over-context; a web_search call no grounding model "
                   "answered) are left out of the accuracy and the per-level and per-attachment "
                   "figures, counted in harness_errors, named in leaderboard_comparable_reason, "
                   "status completed_with_errors, score withheld when no sample is left, and the "
                   "test-split submission lists search_backend_failure_tasks; recorded "
                   "(search_backend, search_backend_calls); --shard is applied to the split's "
                   "full ordered rows (base.apply_shard) before the Level-stratified "
                   "--eval-limit draw, the shard's rows passed to run_eval_suite as they are "
                   "(sampling.presharded_for_base); unsharded selection unchanged; a web "
                   "search that raised is recorded (search_failed, per-run transcript store) "
                   "and so makes its task a search_backend_failure harness error; "
                   "leaderboard_comparable=False"),
    },
    # Suites whose metric, prompt and extraction were reconciled against upstream.
    "simpleqa": {
        "synced": "2026-10-10.2",
        "upstream": "openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47",
        "revision": "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94",
        "dataset": ("basicv8vc/SimpleQA (test; byte-identical to simple-evals "
                    "simple_qa_test_set.csv)"),
        "checksum": ("sha256:feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032 (= "
                     "simple-evals simple_qa_test_set.csv)"),
        "method": (("simple-evals GRADER_TEMPLATE verbatim over the Gemini cascade, the grader "
                    "call as grading_sampler: system message \"You are a helpful assistant.\" and "
                    "temperature 0.5 (no response format; its max_tokens 2048 for the reply, sent "
                    "as max_output_tokens 2048 + the Gemini thinking allowance 8192, since the "
                    "Gemini judge's thinking counts against it and gpt-4.1 does not think; "
                    "replies still cut there and outages after it counted and named, summed "
                    "by a sharded run's merge and named once; judge_sampling); gold "
                    "target inserted as the CSV stores it (surrounding whitespace kept, as "
                    "simple-evals passes row[\"answer\"]); first-A/B/C parse (default C); headline "
                    "= is_correct over all graded, f_score + accuracy_given_attempted alongside; "
                    "an empty reply is NOT_ATTEMPTED (the model's answer), a request that got no "
                    "reply from the served endpoint a harness error (not graded, "
                    "scoring_excluded, left out of every figure, counted and named); nothing "
                    "graded -> accuracy and the correct/incorrect/not_attempted split withheld "
                    "(null); leaderboard_comparable=False (judge); model sampled as simple-evals "
                    "samples it, with gpqa's shared samplers (gpqa.SIMPLE_EVALS_SAMPLING, "
                    "gpqa.simple_evals_messages): no --thinking = the gpt-4o chat sampler (system "
                    "message \"You are a helpful assistant.\", temperature 0.5, max_tokens 2048 on "
                    "every request (gpqa.SIMPLE_EVALS_SAMPLING, the declaration gpqa passes, "
                    "applied by the token-limit policy), --max-output-tokens not applied, "
                    "replies_at_sampler_max_tokens counted), --thinking = the same chat sampler "
                    "(system message, temperature 0.5) with the run's --max-output-tokens instead "
                    "of the 2048 (official cap not applied to reasoning; simple-evals' o-series "
                    "samplers, keyed on OpenAI model names, are not reasoning settings for a "
                    "served model)")),
    },
    "mmlu": {
        "synced": "2026-10-09",
        "upstream": "github.com/hendrycks/test evaluate.py @ 4450500f",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "dataset": "cais/mmlu (all; test, dev exemplars)",
        "method": (("Hendrycks 5-shot prompt verbatim (10 of 14,042 reference prompts show "
                    "pandas' 'nan' for a 'None' option; dataset text kept); evaluate.py's "
                    "request per question: text prompt, max_tokens=1, logprobs=100, "
                    "temperature=0, echo left out (top_logprobs[-1] is the same list; on vLLM "
                    "echo+logprobs computes full-vocabulary prompt logprobs at every position); "
                    "letters read from the generated token's top-100 list, -100 for a letter "
                    "not listed, np.argmax (first on a tie); evaluate.py crop rule applied to "
                    "the served window for prompt + 1 generated token (tokens counted with the "
                    "tokenizer's defaults: no BOS for Gemma 4, bos_prepended recorded; "
                    "shots_reduced); pooled accuracy; --shard before --eval-limit (shard "
                    "recorded); a server that rejects logprobs=100 (vLLM needs --max-logprobs "
                    "100) or returns no top list -> infra_required; a list shorter than 100 "
                    "missing a letter -> short_top_logprobs, not comparable; a question with no "
                    "usable reply after the retries is a harness error: left out of accuracy "
                    "and category_accuracy (scoring_excluded, request_harness_errors, "
                    "effective_n), completed_with_errors, named in status_reason, not "
                    "comparable; accuracy null with accuracy_withheld_reason when none is "
                    "scored; --thinking recorded as not applied; each trace's sample_key "
                    "includes the row's position in the pinned test split (row_index), so rows "
                    "with identical content stay distinct when gbench --merge de-duplicates the "
                    "shards' traces; the unsharded selection and scoring are unchanged; "
                    "leaderboard_comparable=True for full 5-shot runs with every question "
                    "scored")),
    },
    "mmlu_pro": {
        "synced": "2026-10-10",
        "upstream": ("github.com/TIGER-AI-Lab/MMLU-Pro evaluate_from_apiX.py @ f418b116 (the "
                     "README's runner for OpenAI-compatible endpoints; CLI defaults --max_tokens "
                     "32768, --retry 3, --retry_wrong 0)"),
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "dataset": "TIGER-Lab/MMLU-Pro (test, 12,032; validation CoT exemplars, 5 per category)",
        "method": (("process_one prompt: header + every validation exemplar of the category "
                    "(format_example verbatim, N/A options dropped) + 'Answer: Let's think step "
                    "by step.', one user message, no system message, same in both modes; "
                    "streaming_request through openai AsyncOpenAI (timeout 3600): temperature "
                    "0.1, top_p 0.95, stream with include_usage, enable_thinking via "
                    "chat_template_kwargs; max_tokens via base's token-limit policy "
                    "(OfficialSampling.max_tokens 32768 plus a reasoning block with the same "
                    "request, since the runner counts a reasoning model's streamed reasoning "
                    "against max_tokens): 32768 exactly without thinking (--max-output-tokens not "
                    "applied), 32768 or a larger --max-output-tokens with thinking (the larger "
                    "value is flagged non-comparable and recorded as "
                    "official_sampling.max_tokens_applied=false with max_tokens_sent), and the "
                    "client count limit equals the max_tokens sent; each try bounded at 3600 s "
                    "with asyncio.wait_for (upstream's asyncio.timeout needs Python 3.11; same "
                    "bound on 3.10+), a TimeoutError asked again after 5 s; client-side tiktoken "
                    "cl100k_base count of prompt + streamed reasoning/content deltas, past "
                    "max_tokens -> MaxTokensExceeded, 3 tries with min(2^n,30) s waits, then "
                    "scored 0; missing usage or errors asked again after 5 s (gbench bound: 10 "
                    "tries, then FAILED: a harness error left out of the score, named in "
                    "status_reason, non-comparable); endpoint HTTP 4xx other than 408/409/429, a "
                    "served window below the longest prompt + max_tokens, or a "
                    "TypeError/AttributeError/ImportError/NameError raised by "
                    "chat.completions.create itself -> infra_required (the same raised while the "
                    "stream is read, e.g. a malformed chunk, fail only that try: asked again "
                    "after 5 s, as process_one does upstream); extract_answer verbatim "
                    "(_PATTERNS, re.findall, last match of the first matching pattern) on the "
                    "content only (reasoning kept in the trace, not scored); no random guess, no "
                    "wrong-answer retries; categories in first-appearance order with 2 s start "
                    "spacing and the 25 s stall check (concurrency = --num_workers), per-category "
                    "and per-failed-try log lines; accuracy = correct/scored questions "
                    "(save_summary; harness errors left out) + per category; --shard before "
                    "--eval-limit; exemplar-load failure hard-errors; leaderboard_comparable=True "
                    "for full 5-shot runs at T=0.1 with max_tokens 32768, no decoding penalties "
                    "and no FAILED question; the openai client holds the placeholder key EMPTY "
                    "(openai 2.54 and later refuse a client without credentials) and its HTTP "
                    "client removes the Authorization header from every request, so still no key "
                    "is sent")),
    },
    "hmmt": {
        "synced": "2026-10-09",
        "upstream": "github.com/eth-sri/matharena",
        "revision": "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b",
        "dataset": "MathArena/hmmt_feb_2026 (train) @ 02fba4f74d8e68e73e66a02d540fd979c05c274c",
        "checksum": ("git-blob:fc7284aa6aa621dc6e6d2b4aaac5ed3e00a062b0 (vendored "
                     "src/matharena/parser.py; every vendored file's blob sha is checked in "
                     "matharena_common.VENDORED_BLOBS)"),
        "method": (("MathArena protocol: configs/competitions/hmmt/hmmt_feb_2026.yaml instruction "
                    "verbatim + \"\\n\\n{problem}\" (runner.py / PureModelSolver.build_query); 4 "
                    "runs/problem always (runner --n default); last-chance reprompt verbatim "
                    "(Runner._build_last_chance_prompt, reasoning dropped) when "
                    "extract_answer(content, True) finds no answer; sampling = MathArena's "
                    "per-model configs/models yaml, on first and last-chance requests with and "
                    "without thinking: for a served model MathArena ships one for "
                    "(hmmt.MATHARENA_MODEL_CONFIGS, the 59 configs with "
                    "other_params.huggingface_id at b89f2f0a, with their git blob shas, matched "
                    "by huggingface_id against the served name and --tokenizer, ties told apart "
                    "by the model field) its sampling fields, reasoning_effort and max_tokens as "
                    "APIClient sends them (null temperature/top_p/top_k dropped, no temperature "
                    "for o1/o3/o4 names, api: vllm limited to the SamplingParams fields); "
                    "max_tokens is the model's own setting, reasoning too, applied by the "
                    "token-limit policy (exactly without thinking, never below "
                    "--max-output-tokens with it; none when the config has none); its reasoning "
                    "switch must match --thinking, OpenRouter provider routing and harness do not "
                    "apply, any other field is non-comparable (matharena_model_config recorded); "
                    "for a model it ships none for (every Gemma) the served model's own "
                    "generation_config.json temperature, top_p and top_k (read as ojbench reads "
                    "it, fields it does not set not sent, unreadable -> infra_required; 1.0 / "
                    "0.95 / 64 for Gemma 4) and --max-output-tokens (official_sampling and "
                    "recommended_sampling recorded); graded by MathArena grader.extract_and_grade "
                    "vendored byte for byte (matharena_src) in the local gbench-matharena:b89f2f0a image "
                    "(antlr4-python3-runtime 4.11), self-checked before generation; comparable at "
                    ">=4 runs, that sampling and no other sampling field, the config's max_tokens "
                    "sent in full, no reply (first or last-chance) stopped at a limit other than "
                    "the config's max_tokens (replies_cut_off_by_budget; for Gemma any reply at "
                    "--max-output-tokens), full set, every response answered and graded "
                    "(infra_required); a first or last-chance request with no reply and a "
                    "response MathArena's grader could not grade are harness errors, left out of "
                    "the score (scoring_excluded; grading errors also under judge_failures)")),
    },
    "aime": {
        "synced": "2026-10-09",
        "upstream": "github.com/eth-sri/matharena",
        "revision": "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b",
        "dataset": "MathArena/aime_2026 (train) @ d2de22f3c656b4f56cf8981212186377d1e23bc3",
        "checksum": ("git-blob:477c34f71173e0df66da83283e3bbe2035936c66 (vendored "
                     "configs/competitions/aime/aime_2026.yaml)"),
        "method": (("MathArena protocol: aime_2026.yaml instruction verbatim + \"\\n\\n{problem}\", "
                    "same prompt with or without thinking; 4 runs/problem always; MathArena "
                    "last-chance reprompt; sampling = MathArena's per-model configs/models yaml "
                    "on both requests, shared with hmmt (hmmt.served_model_sampling): for a "
                    "served model MathArena ships one for (hmmt.MATHARENA_MODEL_CONFIGS, 59 "
                    "open-weight configs at b89f2f0a matched by huggingface_id) its sampling "
                    "fields, reasoning_effort and max_tokens as APIClient sends them, max_tokens "
                    "by the token-limit policy as the model's own setting (exactly without "
                    "thinking, never below --max-output-tokens with it), reasoning switch "
                    "matching --thinking, unmatched fields non-comparable (matharena_model_config "
                    "recorded); for a model it ships none for (every Gemma) the served model's "
                    "own generation_config.json temperature, top_p and top_k (read as ojbench "
                    "reads it, fields it does not set not sent, unreadable -> infra_required; 1.0 "
                    "/ 0.95 / 64 for Gemma 4) and --max-output-tokens (official_sampling and "
                    "recommended_sampling recorded); MathArena grader.extract_and_grade vendored "
                    "byte for byte in the local gbench-matharena:b89f2f0a image; older AIME years are in "
                    "amc_aime; comparable at >=4 runs, that sampling and no other sampling field, "
                    "the config's max_tokens sent in full, no reply (first or last-chance) "
                    "stopped at a limit other than the config's max_tokens "
                    "(replies_cut_off_by_budget; for Gemma any reply at --max-output-tokens), "
                    "full set, every response answered and graded (infra_required); a first or "
                    "last-chance request with no reply and a response MathArena's grader could "
                    "not grade are harness errors, left out of the score (scoring_excluded; "
                    "grading errors also under judge_failures)")),
    },
    "amc_aime": {"synced": "2026-10-09",
                 "revision": "69d78a4a2c840e82d69af6bc742bda09005f6316",
                 "dataset": "AI-MO/aimo-validation-amc @ 69d78a4a2c840e82d69af6bc742bda09005f6316 "
                            "+ AI-MO/aimo-validation-aime @ "
                            "13f9e12f613e720c2a2b2f345dd04b998a29494d",
                 "method": "gbench composite (no upstream protocol): gbench prompt, final_number "
                           "extraction, exact integer match; accuracy_amc/accuracy_aime reported; "
                           "either dataset failing to load raises; pre-cutoff/contaminated, "
                           "leaderboard_comparable=False"},
    "cruxeval": {
        "synced": "2026-10-09",
        "upstream": "github.com/facebookresearch/cruxeval",
        "revision": "190faf16d175b5847b0af05d937872b1fb395942",
        "dataset": ("cruxeval-org/cruxeval@b96af0450242eb4da433032b90998f25588a5d0f (800 "
                    "functions)"),
        "method": (("CRUXEval-I + CRUXEval-O; prompts.py direct (no-think) / CoT (thinking) "
                    "verbatim with chat system prompt; openai_run.py's max_tokens per mode as "
                    "OfficialSampling.max_tokens through base's token-limit policy: a direct "
                    "request carries stop [/ANSWER] and max_tokens=100 exactly "
                    "(--max-output-tokens not applied), a thinking (CoT) request no stop (cut at "
                    "[/ANSWER] afterwards) and the run budget, with openai_run.py's CoT "
                    "max_tokens=1000 recorded as not applied to reasoning; openai_prompt "
                    "extractors; evaluate_score with evaluation/utils_execute.py vendored "
                    "verbatim (unsafe_execute + reliability_guard, 3 s) in bwrap, through lcb's "
                    "grader/child split: the child runs 'assert <stand-in> == <answer>', the "
                    "grader holds the gold output, compares, and writes a nonce-tagged verdict "
                    "(pass = one true comparison + program finished; exit status not used); "
                    "grader self-check before generation; n=10, pass@1 at T=0.2 and pass@5 at "
                    "T=0.8 in both modes (OfficialSampling); per-column leaderboard_comparable "
                    "covers both passes (temperatures, samples, direct requests sent a max_tokens "
                    "other than 100 or direct replies over 100 tokens, failures, thinking, "
                    "limit); --shard before --eval-limit; headline = mean of the two pass@1, "
                    "leaderboard_comparable=False; generations with no reply are left out of "
                    "their problem's samples; a task with no scored generation withholds the "
                    "mean; the pass@5 pass's failures are named in status_reason")),
    },
    "multilingual_mmlu": {
        "synced": "2026-10-09",
        "upstream": "github.com/openai/simple-evals @ 652c89d0",
        "revision": "325a01dc3e173cac1578df94120499aaca2e2504",
        "dataset": "openai/MMMLU (14 language configs, test)",
        "method": (("simple-evals mmlu_eval verbatim: QUERY_TEMPLATE_MULTICHOICE, "
                    "normalize_response, MULTILINGUAL_ANSWER_REGEXES cascade, "
                    "normalize_extracted_answer, exact match; accuracy pooled over languages; "
                    "gpqa.SIMPLE_EVALS_SAMPLING per mode through base's token-limit policy (both "
                    "modes: the gpt-4o entry's ChatCompletionSampler system message 'You are a "
                    "helpful assistant.', T=0.5; no-think: max_tokens=2048 sent exactly, "
                    "--max-output-tokens not applied; thinking: the run's --max-output-tokens, "
                    "the 2048 recorded as not applied to reasoning; simple-evals' o-series "
                    "samplers, keyed on OpenAI model names, are not reasoning settings for a "
                    "served model); --shard before --eval-limit; leaderboard_comparable=True for "
                    "full runs with every request answered at the sampler's temperature, False "
                    "for --eval-limit, an operator temperature, "
                    "failed/timed-out/empty/over-context requests (a request with no reply is a "
                    "harness error, left out of the score; an empty reply is scored wrong; "
                    "simple-evals samplers ask failed and null-content requests again), a "
                    "no-think request sent a max_tokens other than 2048, or a reply cut off "
                    "before finishing at anything but the no-think 2048 (a thinking reply at the "
                    "run's budget or a context-clamped max_tokens, a no-think reply at a clamped "
                    "max_tokens)")),
    },
    "docvqa": {
        "synced": "2026-10-09",
        "upstream": ("github.com/EvolvingLMMs-Lab/lmms-eval tasks/docvqa (docvqa_val) + "
                     "lmms_eval/api/metrics.py anls@42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8"),
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "dataset": ("lmms-lab-encoder/DocVQA@539088ef8a8ada01ac8e2e6d4e372586748a265e (DocVQA, "
                    "validation, 5,349)"),
        "method": ("lmms-eval docvqa_val prompt '{question}\\nAnswer the question using a "
                   "single word or phrase.'; token-limit policy: no-think temperature 0 and "
                   "max_tokens 32 (generation_kwargs, sent by the chat client), --thinking the "
                   "same temperature 0 (enable_thinking_kwarg keeps the task's "
                   "generation_kwargs; the o1/o3/o4/gpt-5 model-name branch is not used) and "
                   "the 32 not applied (lmms-eval declares no reasoning-model cap: the run's "
                   "--max-output-tokens, else the thinking floor, recorded as not applied to "
                   "reasoning); lmms-eval anls (raw-length denominator, 0.5 threshold) on the "
                   "reply as returned (no extraction), mean x100; missing image or dataset "
                   "hard-errors; leaderboard_comparable only full, unsharded, official "
                   "temperature, every request answered and no reply cut off by the run's token "
                   "budget (replies_cut_off_by_budget)"),
    },
    "infographicvqa": {
        "synced": "2026-10-09",
        "upstream": ("github.com/EvolvingLMMs-Lab/lmms-eval tasks/infovqa (infovqa_val) + "
                     "lmms_eval/api/metrics.py anls@42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8"),
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "dataset": ("lmms-lab-encoder/DocVQA@539088ef8a8ada01ac8e2e6d4e372586748a265e "
                    "(InfographicVQA, validation, 2,801)"),
        "method": ("lmms-eval infovqa_val prompt '{question}\\nAnswer the question using a "
                   "single word or phrase.'; token-limit policy: no-think temperature 0 and "
                   "max_tokens 32 (generation_kwargs, sent by the chat client), --thinking the "
                   "same temperature 0 (enable_thinking_kwarg keeps the task's "
                   "generation_kwargs; the o1/o3/o4/gpt-5 model-name branch is not used) and "
                   "the 32 not applied (lmms-eval declares no reasoning-model cap: the run's "
                   "--max-output-tokens, else the thinking floor, recorded as not applied to "
                   "reasoning); lmms-eval anls (raw-length denominator, 0.5 threshold) on the "
                   "reply as returned (no extraction), mean x100; missing image or dataset "
                   "hard-errors; leaderboard_comparable only full, unsharded, official "
                   "temperature, every request answered and no reply cut off by the run's token "
                   "budget (replies_cut_off_by_budget)"),
    },
    "screenspot_pointing": {
        "synced": "2026-10-09.2",
        "upstream": ("ScreenSpot-v2 release (OS-Copilot/ScreenSpot-v2) + "
                     "github.com/njucckevin/SeeClick "
                     "pretrain/screenspot_test.py@0ef37ac4d7aaf37ba7b990e3e3c3ca77e1fb8f93 (box "
                     "conversion, click accuracy, generation config)"),
        "revision": "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c",
        "dataset": ("OS-Copilot/ScreenSpot-v2@5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c "
                    "(screenspot_mobile/desktop/web_v2.json, 501 + 334 + 437 = 1,272 targets; "
                    "screenspotv2_image.zip)"),
        "checksum": ("sha256:fdd1595b64179e31407eed5929fa12e80adc6ba2ee372a3802d36da379ea1825 "
                     "(mobile), 96f494e0b28eead9187c67794b02bde63350b9cb753acc5fc0384fa77c7182a4 "
                     "(desktop), 3e89e245d69f56a8e8f430917dc02f95b6fc36f8c8f2836c2cafe4dad10e823d "
                     "(web), 750555b8b06e3898634c610cd904600feb7ec7ca530e625a00c1416a73d829ee "
                     "(image zip)"),
        "method": ("gbench Point: (x, y) prompt (ScreenSpot defines no model-agnostic prompt; "
                   "the reference script uses SeeClick's own); screenshots sent as data URLs "
                   "with the MIME type of their bytes (109 of the 730 are JPEG, 78 of them "
                   "named .png); box [x,y,w,h]/image size; point-in-box click accuracy; "
                   "headline = accuracy over all targets (OS-Atlas table Avg) + the six "
                   "Mobile/Desktop/Web x Text/Icon cells; no temperature and no max_tokens sent "
                   "(the reference script uses the model's own GenerationConfig); "
                   "leaderboard_comparable=False (prompt)"),
    },
    "charxiv": {
        "synced": "2026-10-09.2",
        "upstream": ("github.com/princeton-nlp/CharXiv (src/constants.py, descriptive_utils.py, "
                     "reasoning_utils.py, generate_lib/gpt.py)"),
        "revision": "7ebe88f78dee387691551f071abcb2b9e1a8025b",
        "dataset": ("princeton-nlp/CharXiv@f441eb632fc62f6f777830a0f47619e6e86459b0 (validation, "
                    "1,000 charts; chart files from images.zip)"),
        "checksum": ("sha256:c4b151cef2aba35523764dd1091d719d64acf81a16904d72f31da2723efbd855 "
                     "(images.zip, verified at load)"),
        "method": ("canonical descriptive + reasoning prompts and rubrics vendored; model request "
                   "as gpt.py builds it (question text, then the chart as a base64 data URL with "
                   "the MIME type of its bytes, image/jpeg for CharXiv's charts; the chart bytes "
                   "are images.zip's <figure_id>.jpg at the dataset revision, the files the "
                   "official harness unzips into images/ and generate.py sends; the HF parquet "
                   "image column holds re-encoded JPEGs and is not used); sampling per mode from "
                   "gpt.py (temperature 0.0, top_p 1.0, seed 42; max_tokens 1000 sent without "
                   "--thinking, --max-output-tokens with it; o1.py is keyed on OpenAI model names "
                   "and not used for a served model); judged by the gbench Gemini cascade (paper: "
                   "gpt-4o-2024-05-13) as evaluate.py grades: descriptive answers grouped by "
                   "template id and graded up to 5 per prompt (build_descriptive_grading_queries "
                   "nq_per_query=5; prompts byte-identical to upstream's builder), reasoning one "
                   "per prompt, a reply failing json.loads / the keys / verify_grading_output "
                   "asked again up to 10 attempts, then JUDGE_FAILED (scored 0, as upstream's -1; "
                   "counted in judge_failures); the upstream judge call's JSON mode, temperature "
                   "0, top_p 1, seed 42 (its max_tokens 256->1024 not sent: written for a "
                   "non-thinking judge; judge_sampling, named in leaderboard_comparable_reason); "
                   "per-trace judge_model / judge_raw / judge_batch, result judge_models and "
                   "judge_cascade; total_questions / correct_answers count the headline "
                   "(reasoning) track; a chart without its image in images.zip hard-errors (a "
                   "failed images.zip download is infra_required); infra_required without "
                   "GEMINI_API_KEY(S) (no relaxed-match fallback); leaderboard_comparable=False "
                   "(judge)"
                   "; JUDGE_FAILED left out of the score, upstream's -1-as-0 reading kept as "
                   "accuracy_upstream_invalid_as_zero; prompts byte-identical to the builder's, "
                   "the reasoning text included; empty descriptive replies graded in the batch"),
    },
    "bundled_detection": {
        "synced": "2026-10-09.2",
        "upstream": "cocodataset/cocoapi pycocotools COCOeval (iouType bbox)",
        "revision": "pycocotools 2.0.11 (as verified)",
        "dataset": ("COCO 2017 val: images.cocodataset.org zips/val2017.zip (5,000 images) + "
                    "annotations/annotations_trainval2017.zip instances_val2017.json (36,781 "
                    "boxes, 446 crowd)"),
        "checksum": ("sha256:4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05 "
                     "(val2017.zip), "
                     "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268 "
                     "(annotations zip), "
                     "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f "
                     "(instances_val2017.json)"),
        "method": ("gbench zero-shot COCO-80 JSON prompt; detections to COCO results (category id "
                   "from label, box in pixels, confidence as score, 1.0 when absent); pycocotools "
                   "COCOeval bbox on every requested image (crowd regions, area ranges, 100 "
                   "detections); headline AP@[.50:.95], all 12 summary stats reported; "
                   "bit-identical to a direct pycocotools run on the official GT at 200 / 500 / "
                   "5,000 images; leaderboard_comparable=False (gbench prompt, self-reported "
                   "confidences)"),
    },
    "omnidocbench": {
        "synced": "2026-10-09",
        "upstream": "github.com/opendatalab/OmniDocBench",
        "revision": "193627ae9e97d89188468ed1ee3b7a856ff76044",
        "dataset": ("opendatalab/OmniDocBench@aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec (v1.6, 1651"
                    " pages)"),
        "method": ("official general-VLM prompt verbatim (Qwen2VL_img2md.py), image+prompt in one "
                   "turn, greedy (temperature 0 as the default, thinking or not); max_tokens 32000"
                   " (max_new_tokens) sent without --thinking, the run's --max-output-tokens with "
                   "it (cap recorded as not applied to reasoning); scored by the pinned evaluator "
                   "in a local image (end2end quick_match composite; image revision verified at "
                   "run time); leaderboard_comparable also requires every request answered (a page"
                   " whose request got no reply from the served endpoint is a harness error: left "
                   "out of the predictions and the ground truth the evaluator scores, counted in "
                   "pages_left_out and request_harness_errors, status completed_with_errors; when "
                   "every page is one, the evaluator does not run, status failed, accuracy null "
                   "with accuracy_withheld_reason) and no reply cut off by the run's token budget "
                   "(replies_cut_off_by_budget); the evaluator container (which compiles the "
                   "model's LaTeX with pdflatex for CDM) and the image-revision check run with "
                   "--network none: the end2end metrics (Edit_dist, CDM, TEDS) load nothing at run"
                   " time (verified 2026-09-27: identical run summary to the bridge network on a "
                   "two-page formula + table sample); an --eval-limit subset is stratified by the "
                   "page's subset (page_attribute.subset: v1.5, equation_hard, layout_hard, "
                   "table_hard), then within each subset by data source, seeded omnidocbench "
                   "(subset_selection), not a contiguous head"),
    },
    "bfcl": {
        "synced": "2026-10-10",
        "upstream": ("gorilla bfcl_eval @6ea5797 = bfcl-eval 2026.3.23 (183/183 files "
                     "byte-identical; the only release accepted), run in the local image "
                     "gbench-bfcl:2026.3.23 (gbench/docker/bfcl.Dockerfile: python:3.10-slim, "
                     "PyPI wheel sha256-checked, the release's own pins + bfcl.constraints.txt, "
                     "soundfile for qwen_agent's import (gorilla issue #1338), "
                     "all-MiniLM-L6-v2@1110a243): single-turn FC request (ported request half: "
                     "language hint + convert_to_tool) scored by "
                     "eval_runner._evaluate_single_ast_entry / _evaluate_single_relevance_entry "
                     "in the image; multi-turn OpenAICompletionsHandler.inference_multi_turn_FC "
                     "+ executable backends + "
                     "_evaluate_single_multi_turn_entry/multi_turn_checker in the image; "
                     "eval_runner_helper aggregation; sampling bfcl_eval/__main__.py generate "
                     "--temperature default 0.001, no max_tokens (_query_FC)"),
        "revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "dataset": ("ShishirPatil/gorilla bfcl_eval/data BFCL_v4, all 22 scoring categories: 13 "
                    "single-turn (3,641 entries, 23 files blob-checked at load), 4 multi-turn "
                    "(800 entries read by bfcl-eval in the image; data, func docs, backends, "
                    "loop, tool conversion and checker blob-checked, 38 files), 5 agentic via "
                    "bfcl_v4_agentic (665; 35 files blob-checked)"),
        "method": (("temperature 0.001 (OfficialSampling) in all three parts, the agentic part "
                    "handed bfcl's resolved value; no max_tokens in any part (OfficialSampling "
                    "send_max_tokens=False; --max-output-tokens not applied, "
                    "max_output_tokens_note); gbench's process never imports bfcl_eval "
                    "(single-turn checker, multi-turn and agentic harness run in the image; no "
                    "Docker, no image, a foreign image or a differing pinned file -> "
                    "infra_required before generation); single-turn: turn-0 messages verbatim + "
                    "convert_to_tool(language-hinted docs), each reply -> the model result "
                    "_parse_query_response_FC records, scored by bfcl_eval's eval_runner (13 "
                    "checker files blob-checked; reply text never a call; 0 validity/error-type "
                    "differences vs the ported checker on 34,104 reply variants over all 3,641 "
                    "entries); multi-turn: bfcl-eval's own loop/backends/checker, request adds "
                    "chat_template_kwargs and penalties only (ground-truth replay 800/800 in the "
                    "image, no max_tokens, temperature 0.001); backend instances per attempt and "
                    "entry, cleanup removes exactly one attempt's entry (12 concurrent attempts: "
                    "none rebuilt mid-entry); execution guard refuses builtin/global/injected "
                    "calls with upstream's error text and bare-name X; no key in the multi-turn "
                    "container (a real OPENAI_API_KEY added by swebench_common.EndpointRelay on "
                    "the bridge gateway, multi_turn_report.endpoint_relay; the container holds "
                    "EMPTY); the relay's upstream_errors/requests_refused (and the agentic "
                    "part's) -> endpoint_failures (multi_turn_report.endpoint_failures, "
                    "agentic_report.endpoint_failures, summed at top level), "
                    "completed_with_errors and a named reason; harness errors never the model's "
                    "outcome, in every part (base's rule): an entry whose request got no reply "
                    "(FAILED/TIMEOUT/OVER_CONTEXT; upstream's inference error, which its checker "
                    "scores wrong) left out of its category's correct and total, counted in "
                    "failed_requests (timed_out_requests / over_context_prompts), "
                    "scoring_excluded and request_harness_errors "
                    "(endpoint_failure/endpoint_timeout/over_context; unfinished and "
                    "memory_prereq_failed from stopped parts and the agentic part), named in "
                    "status_reason and leaderboard_comparable_reason (single-turn by base; "
                    "multi-turn traces marked scoring_excluded with harness_error, "
                    "multi_turn_report.harness_errors_left_out; agentic as bfcl_v4_agentic, "
                    "agentic_report.harness_errors_left_out); a category with no entry left in "
                    "the score -> unscored (categories_unscored, null columns, never 0; "
                    "effective_n = the entries in the score); a multi-turn or agentic part whose "
                    "every entry failed its request -> also a harness_errors entry with "
                    "categories_unscored; nothing scored -> accuracy null + "
                    "accuracy_withheld_reason, status failed; the agentic part's memory "
                    "prerequisite entries whose request failed -> memory_prereq_failed_requests "
                    "(agentic_report.prereq_inference_errors; not in failed_requests; their "
                    "scenario's scored entries left out of the score as memory_prereq_failed), "
                    "completed_with_errors and the agentic part's named reason; the agentic "
                    "part's project root must hold no .env (infra_required before the first "
                    "generation); the Gemini models that answered the web searches in "
                    "gemini_models_answered; the multi-turn endpoint check passes proxied=True "
                    "when a proxy is set; the agentic part's bfcl generate runs under the same "
                    "guard, and its prerequisite and network checks name bfcl and "
                    "docs/evals/bfcl.md; agentic delegated to bfcl_v4_agentic (its web_search "
                    "tool served by Gemini Google Search grounding in place of BFCL's DuckDuckGo "
                    "via SerpAPI search, so a run that scores it is not comparable: the agentic "
                    "reason 'web search served by Gemini Google Search grounding (BFCL queries "
                    "DuckDuckGo via SerpAPI)' carried into leaderboard_comparable_reason; only "
                    "the Web Search setting needs the Gemini key (GEMINI_API_KEYS, else "
                    "GEMINI_API_KEY), checked with base.gemini_key_live_valid before generation "
                    "(the judged suites' gate): unset or rejected, or a search no grounding model "
                    "served during generation (generate stopped at the first one, log read every "
                    "30 s, the unfinished memory entries then generated by the resume built into "
                    "bfcl generate, run without --allow-overwrite) -> that setting reported "
                    "infra_required in its own entry (web_search_setting, "
                    "settings_infra_required, prerequisite GEMINI_API_KEY; never scored, Overall "
                    "Acc and the Agentic column null, and on a run of every category the "
                    "headline withheld (accuracy null, accuracy_withheld_reason; pooled_accuracy "
                    "kept), an explicit --eval-categories subset headlined by pooled_accuracy, "
                    "completed_with_errors, not comparable) "
                    "while single-turn, multi-turn and memory run and are scored; web_search "
                    "alone selected -> infra_required; any other prerequisite lost mid-run -> "
                    "infra_required with the scored parts logged); headline BFCL Overall Acc "
                    "10/10/10/30/40 (every bfcl_summary column equal to "
                    "generate_leaderboard_csv); bounds as upstream "
                    "(_llm_response_generation.multi_threaded_inference per-entry inference "
                    "errors, MAXIMUM_STEP_LIMIT 20 per turn, per-request client timeout: the "
                    "multi-turn client's sized from max_model_len, cf. base_oss_handler.py "
                    "timeout=72000; the agentic handler's per-request timeout sized the same way, "
                    "the SDK's 2 retries kept; without a reported window both sized for "
                    "base.UNCAPPED_TIMEOUT_TOKENS 262144); no whole-part wall clock unless "
                    "GBENCH_BFCL_TIMEOUT_S, which bounds each part on its own (single-turn "
                    "checker, multi-turn part, bfcl generate, bfcl evaluate); a stopped part -> "
                    "harness_errors with the rest kept (finished multi-turn/agentic entries "
                    "scored, unfinished ones TIMEOUT harness errors left out of the score "
                    "(unfinished), unscored single-turn replies HARNESS_ERROR and excluded, a "
                    "stopped evaluate leaves the agentic categories unscored, a generate stopped "
                    "before it recorded first), completed_with_errors, not comparable; the image "
                    "probe and the single-turn checker run with --network none; the multi-turn "
                    "part runs on docker's bridge network (never the host network), after the "
                    "metadata-server guard and a TCP check of a loopback endpoint at "
                    "host.docker.internal, both run before the first generation")),
    },
    "bfcl_live": {
        "synced": "2026-10-09",
        "upstream": ("gorilla bfcl_eval @6ea5797 (= bfcl-eval 2026.3.23; its eval_runner runs in "
                     "the local image gbench-bfcl:2026.3.23)"),
        "revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "dataset": ("ShishirPatil/gorilla bfcl_eval/data BFCL_v4 live_* (6 categories, 2,251 "
                    "entries; files checked against pinned git blob ids at load)"),
        "method": ("BFCL FC mode (bfcl pipeline) at BFCL's temperature 0.001 and no max_tokens "
                   "(OfficialSampling, __main__.py generate default; _query_FC sends none): "
                   "messages verbatim + convert_to_tool; replies scored by bfcl_eval's "
                   "eval_runner (_evaluate_single_ast_entry / _evaluate_single_relevance_entry) "
                   "in the local image gbench-bfcl:2026.3.23 (image and checker files checked "
                   "before generation, else infra_required); reply text never a call (BFCL FC "
                   "rule); headline = BFCL Live Overall Acc (entry-weighted over the 4 live AST "
                   "categories), live irrelevance/relevance alongside; "
                   "leaderboard_comparable=False (one column of BFCL); the checker has no wall "
                   "clock unless GBENCH_BFCL_TIMEOUT_S (a stopped checker leaves the replies "
                   "unscored: HARNESS_ERROR, scoring_excluded, harness_errors)"),
    },
    "seal_tools": {
        "synced": "2026-10-09",
        "upstream": "github.com/fairyshine/Seal-Tools",
        "revision": "ce753ecd60ed08dd376984035de531ab8421f1c6",
        "dataset": ("casey-martin/Seal-Tools test@d0fe2245740d01a22b8fdd22ec1f49e48fcb1fbf (1354 "
                    "rows = dataset_for_finetune test_in_domain + test_out_domain)"),
        "method": ("ToolLearning_eval: row prompt as one user message; no sampling field sent, "
                   "as LLM_Evaluation/src/llm_tools/llms/ChatGPT.py (model + messages; "
                   "OfficialSampling send_temperature=False and send_max_tokens=False: no "
                   "temperature and no max_tokens, --max-output-tokens not applied; an operator "
                   "temperature makes the run non-comparable); "
                   "transform_output_format('ToolLearning') list parse on the "
                   "reasoning-stripped reply; calculate_score_ToolLearning "
                   "str(pred)==str(gold), pooled Format ACC + Tool/Parameter P/R/F1; headline "
                   "Parameter F1; verified vs upstream on 1354x10 synthetic replies"),
    },
    "nestful": {
        "synced": "2026-10-09",
        "upstream": "github.com/IBM/NESTFUL",
        "revision": "fc2c4123e73500a56185a5fb354f05d1c8b4890c",
        "dataset": ("IBM/NESTFUL data_v2/nestful_data.jsonl (1861 rows) from the checkout at the "
                    "pinned revision, the only data source as in upstream run.sh (no Hugging "
                    "Face copy); no checkout -> whole-suite infra_required; executable functions "
                    "missing or at another revision -> the Win Rate alone infra_required "
                    "(win_rate_metric)"),
        "checksum": ("git-blob:ab49f52a8a11289247f7f4c1b5265c01d449c1aa "
                     "(data_v2/nestful_data.jsonl)"),
        "method": (("deepseek_prompt_input byte for byte (tools double JSON-encoded as upstream); "
                    "sampling as src/eval.py (--temperature default 0.0, --max_tokens default "
                    "1000, run.sh passes neither; OfficialSampling.max_tokens: the 1000 sent "
                    "exactly without --thinking, not applied with it (run budget, recorded as "
                    "official cap not applied to reasoning); replies cut off by any other budget "
                    "counted in replies_cut_off, not comparable); parse_deepseek_output (whole "
                    "reply json.loads, reasoning tags stripped) + ground_seq_nested_repsonse, no "
                    "label rewrite; scorer.py calculate_scores (macro F1 Intent/Slot, "
                    "Partial/Full Match) + executable Win Rate run in a worker process inside a "
                    "bubblewrap container of its own whatever GBENCH_SANDBOX says (the reply is "
                    "model-controlled code: 18 offered functions eval/exec/run their arguments; "
                    "no network, environment PATH and HOME=/tmp only, no /proc or /run, only the "
                    "interpreter and the reference functions read-only; the worker reports what "
                    "it sees and is refused unless isolated; sys.argv = upstream scorer.py "
                    "command line; upstream 10 s alarm on its main thread, 30 s kill backstop; a "
                    "killed call makes the run non-comparable; the worker wraps socket name "
                    "lookups, create_connection and connect/connect_ex/sendto/bind, and a call "
                    "that does not win after a failed use of the network is a harness error only "
                    "on a row whose pinned gold needs the network (fa57c28a, the one such row): "
                    "marked nestful_win_blocked_by_sandbox, counted in "
                    "win_rate_network_blocked_calls (ids in win_rate_network_blocked_samples), "
                    "left out of the Win Rate denominator (every answered row blocked -> win_rate "
                    "null + win_rate_withheld_reason), named in status_reason and "
                    "leaderboard_comparable_reason, completed_with_errors, not comparable; on any "
                    "other row it is scored as the reply's loss and counted "
                    "(win_rate_network_blocked_offpath and its _samples); the 10 s alarm "
                    "firing inside a socket call is the call's timeout, not a network use; the "
                    "container has a localhost-only /etc/hosts where bwrap offers --ro-bind-data; "
                    "all 1861 gold sequences: upstream scorer.py with network 1857 wins, "
                    "sandboxed worker 1856 wins + 4 losses + fa57c28a (get_url_content fetches "
                    "example.com) reported as a blocked network use); no working bubblewrap -> "
                    "the Win Rate alone infra_required (win_rate_metric names the sandbox); only "
                    "the Win Rate needs data_v2/executable_functions: without it (missing, "
                    "incomplete or not at the pinned blobs) or without the sandbox the Win Rate "
                    "alone is reported infra_required in its own entry (win_rate_metric, "
                    "metrics_infra_required; win_rate null, never scored, as calculate_scores "
                    "with win_rate_flag=False reports it no) while F1 Intent/Slot and "
                    "Partial/Full Match are computed, completed_with_errors, not comparable; "
                    "headline Full Match; verified vs scorer.py on 1861x10 synthetic replies; "
                    "data files checked against pinned git blob shas at load (a data file that "
                    "differs is a hard error); the checkout's data_v2/nestful_data.jsonl is the "
                    "only data source, as upstream run.sh --dataset uses it (IBM's Hugging Face "
                    "copy is not read): without it the whole suite is infra_required; requests "
                    "with no reply are harness errors left out of every metric and named in "
                    "leaderboard_comparable_reason (not comparable); none answered -> every "
                    "metric null (never 0), status failed")),
    },
    "acebench": {
        "synced": "2026-10-10.2",
        "upstream": "github.com/chenchen0103/ACEBench",
        "revision": "56dd66cf6439b0d9655ee1b353e4cd745c6f664e",
        "dataset": ("ACEBench data_all/data_en + data_all/data_zh, all 17 categories (12 Normal, "
                    "3 Special, 2 Agent): 973 + 967 Normal/Special rows, 50 + 50 agent "
                    "conversations; 68 files blob-checked at load"),
        "method": ("both published settings (English headline, Chinese overall_zh); "
                   "Normal/Special: prompt_en.py / prompt_zh.py verbatim (functions via "
                   "str.format), vendored checker.py/utils.py decode_ast + normal_checker; BinOp "
                   "arguments = upstream's eval(ast.unparse(node)) (literal arithmetic "
                   "in-process; any other BinOp by Python's eval in a bubblewrap container of its "
                   "own (--unshare-all --unshare-user, no capabilities, empty environment, only "
                   "/usr and the interpreter's directories on its root: no /proc, no /run or "
                   "Docker socket, no home or cache; GBENCH_SANDBOX does not apply; checked "
                   "before generation) with that role module's globals, value returned as "
                   "upstream's, a builtin exception as the same exception and any other as a "
                   "same-named Exception subclass; a name only upstream's process has, a "
                   "non-literal value, the evaluator's limits or an oversized literal result "
                   "refused and counted, not comparable; bubblewrap missing or the container not "
                   "isolated -> infra_required) and special_eval substrings; Agent: "
                   "common_inference multi_turn/multi_step inference replayed turn by turn (40 "
                   "steps) with the vendored agent prompts and multi_turn scenario classes "
                   "(_acebench_vendor/, upstream blobs recorded), execute_agent_func_call replica "
                   "with its quirks, calls dispatched without eval (anything else upstream's "
                   "eval() of a call string would run is refused and counted; not comparable), "
                   "BinOp arguments of agent messages evaluated per role module as above, "
                   "agent_multi_turn user simulator (APIModel_user APIUSER, upstream gpt-4o) = "
                   "gbench Gemini judge cascade at APIUSER's own sampling (temperature 0.001, "
                   "top_p 1, max_tokens 1000 for the reply, sent as max_output_tokens 1000 + "
                   "the Gemini thinking allowance 8192; replies that stopped there and outages "
                   "after them counted, with text used as they are, without text passed to the "
                   "next cascade model, either not comparable) (recorded; GEMINI_API_KEY else "
                   "infra_required) "
                   "(without a usable key, unset or rejected by the live check, only "
                   "agent_multi_turn is infra_required, in its own entry: "
                   "agent_multi_turn_category, categories_infra_required; never run or scored, "
                   "not even as 0; agent_summary, the agent and overall columns, both Summaries "
                   "and accuracy null; the other 16 categories run and are scored; "
                   "completed_with_errors, not comparable; a hard error only when "
                   "agent_multi_turn is all the run selected), agent_eval/agent_checker end to "
                   "end + agent_eval_process milestones; sampling model_infer.py LLMInfer "
                   "(temperature 0.0, top_p 0.9, max_tokens 1024; OfficialSampling.max_tokens) on "
                   "every request of the model under test, within LLMInfer's max_model_len=8192 "
                   "window (model_infer.py lines 28/36): without --thinking the 1024 is sent, or "
                   "the room the window leaves after the prompt as the pinned vllm==0.6.1.post1 "
                   "(commit acda0b35d00e) applies it: at most 8193 - P tokens for a prompt of P "
                   "<= 8192 tokens (stop_checker.py stops once the length exceeds max_model_len), "
                   "a reply stopping there counted apart from truncations, and a prompt of more "
                   "than 8192 tokens is not sent, its reply the empty text that vLLM returns for "
                   "it (scheduler.py FINISHED_IGNORED, finish reason length; a single-turn row "
                   "scored on the empty reply, an agent step handing it on and the conversation "
                   "going on), counted in upstream_context_window and comparable; prompts counted "
                   "as that vLLM counts them (/tokenize of the chat template with "
                   "add_special_tokens true, only when bytes + 512 + 1024 exceed 8193), a prompt "
                   "only estimated or with a conservative tokenizer count near the window sent "
                   "the 1024 and not comparable; with --thinking neither the cap nor the window "
                   "is applied (run budget, recorded as official cap not applied to reasoning); "
                   "replies cut off by any other budget (an empty reply after reasoning included) "
                   "counted and not comparable; single-turn rows scored after generation in "
                   "worker threads (async_eval_fn, deterministic); generate_result_csv "
                   "aggregation per language (unweighted normal/special/agent summaries), "
                   "headline Summary = 0.2676 special + 0.578 normal + 0.1545 agent = the "
                   "leaderboard overall, every leaderboard column reported; verified vs "
                   "eval_main.py (Normal/Special) and vs the upstream agent classes in-process on "
                   "all 100 agent conversations x 3 scripted runs (0 mismatches outside guard "
                   "refusals), BinOp arguments checked against upstream "
                   "CommonAgent.decode_function_list / utils.ast_parse (same values and "
                   "exceptions; expressions naming self/value refused); subset runs "
                   "(--eval-limit, a shard) record per language the sub-categories not run "
                   "(subcategories_not_run) and the Summary a model right on every drawn item "
                   "would get (summary_ceiling), and name the undrawn English sub-categories and "
                   "the ceiling in leaderboard_comparable_reason; the user simulator's cascade "
                   "records each answering Gemini model (base.record_gemini_answer) and does not "
                   "ask a model that answered HTTP 404 again within a request"),
    },
    "mrcr": {
        "synced": "2026-10-09",
        "upstream": ("huggingface.co/datasets/openai/mrcr dataset card README 'How to run' "
                     "reference loop"),
        "dataset": "openai/mrcr",
        "revision": "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d",
        "method": ("rows over the served window skipped (as upstream); o200k_base prompt+answer "
                   "bins; autojunk grade; an empty reply counts as 0; a request with no reply from"
                   " the served endpoint is a harness error left out of the mean, mrcr_bins and "
                   "the per-category table; sampling as that reference loop "
                   "(client.chat.completions.create(model=MODEL, messages=messages)): no "
                   "temperature and no max_tokens sent, in both modes (OfficialSampling "
                   "send_temperature=False, send_max_tokens=False; an operator temperature is "
                   "recorded under official_sampling and named); the per-row answer-room fit rule "
                   "still decides which rows are sent; leaderboard_comparable=False (pooled bins; "
                   "the reason also names an --eval-limit subset, an operator temperature and "
                   "requests with no reply); the window is the max_model_len of the /v1/models "
                   "entry that matches the model under test (the first entry's window before, "
                   "another model's on a multi-model endpoint); --shard comes before the "
                   "--eval-limit draw; a partial shard carries the standard shard reason; an "
                   "empty shard gets base's empty row"
                   "; exact_match_rate = verbatim reproductions (the canary gate, then the reply "
                   "equal to the answer with the canary stripped from both; gbench's, not "
                   "upstream's)"),
    },
    "custom_jsonl": {
        "synced": "2026-10-09",
        "method": ("first-present-key gold; keeps falsy golds; contains is a whole-token, "
                   "case-insensitive match; the numeric match reads a number with commas both "
                   "ways, comma-grouped thousands (299,792,458; 1,234.5) and a list (85,90,100), "
                   "passing when either holds the gold; an unknown eval_type or a line with no "
                   "prompt stops the run with an error naming the line"),
    },
    "cyberseceval": {
        "synced": "2026-10-09.2",
        "upstream": ("github.com/meta-llama/PurpleLlama CybersecurityBenchmarks "
                     "(datasets/instruct/instruct-v2.json; requirements.txt semgrep==1.51.0 / "
                     "pyyaml==6.0.1; benchmark/llms/llm_base.py + llms/openai.py sampling; "
                     "benchmark/instruct_or_autocomplete_benchmark.py scoring; "
                     "CodeShield/insecure_code_detector)"),
        "revision": "114d02388213ae791650b10fa61c8785661c2602",
        "dataset": ("PurpleLlama@114d0238 "
                    "CybersecurityBenchmarks/datasets/instruct/instruct-v2.json (1,681 prompts, 8 "
                    "languages; the README's replacement for instruct.json)"),
        "checksum": ("sha256:df19f7c51911d46d0ae5c069b7093728db8e682890a44317a211ac61cb02c181 "
                     "(instruct-v2.json, verified at load)"),
        "method": ("whole-language ICD (insecure_code_detector.analyze, usecase=cyberseceval, "
                   "regex + semgrep) run unmodified in the local image "
                   "gbench-cyberseceval:114d02388213-semgrep-1.51.0 "
                   "(gbench/docker/cyberseceval.Dockerfile: python:3.10-slim by digest; "
                   "PurpleLlama CodeShield fetched at the pinned commit; semgrep 1.51.0 and PyYAML "
                   "6.0.1 as CybersecurityBenchmarks/requirements.txt pins, the rest pinned by "
                   "cyberseceval.constraints.txt; each container --network none, read-only root, "
                   "tmpfs /tmp, --cap-drop ALL, no host variable); the ICD's default "
                   "--project-root / semgrep parameter kept (it scans the target under semgrep "
                   "1.51.0; the earlier INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER=False flip for a "
                   "newer host semgrep removed); image labels, releases, commit and worker sha256 "
                   "checked and a semgrep self-test (Python eval flagged by the semgrep analyzer) "
                   "required before any request, else infra_required; all 8 languages; no injected "
                   "suffix; each test_case_prompt sent verbatim, trailing newline kept "
                   "(query_llm.process_prompt); prompts read from the image's PurpleLlama checkout "
                   "(sha256-checked at build and at load); a file without 1,681 rows over the 8 "
                   "languages refused; sampling as upstream's OpenAI adapter sends it (llm_base.py "
                   "DEFAULT_TEMPERATURE=0.6, DEFAULT_TOP_P=0.9; openai.py OPENAI.query sends no "
                   "max_completion_tokens outside its proprietary reasoning-model list, keyed on "
                   "OpenAI model names and not applied to a served model): temperature 0.6, top_p "
                   "0.9, no max_tokens, in both modes (OfficialSampling send_max_tokens=False); an "
                   "empty reply or a refusal scored secure as upstream's ICD scores the whole "
                   "message; a request with no reply is a harness error left out of the pass rate; "
                   "a program the ICD could not scan (a semgrep run that printed no JSON in 3 "
                   "analyses, a language the ICD lacks, an ICD exception, no result from the "
                   "container) is a harness error left out of the pass rate and counted in "
                   "cyberseceval_report.icd_harness_errors; upstream's per-language total_count / "
                   "vulnerable_suggestion_count / pass_rate in cyberseceval_report.by_language; "
                   "leaderboard_comparable only for a full no-think run at temperature 0.6 with "
                   "one sample per prompt, every request answered and every program scanned, "
                   "otherwise each cause named"),
    },
    "wmdp": {
        "synced": "2026-10-09",
        "upstream": ("lm-evaluation-harness v0.4.2 (4600d6bf) tasks/wmdp + "
                     "lm_eval/models/openai_completions.py local-completions request (per "
                     "centerforaisafety/wmdp README; reference run run_rmu_zephyr.ipynb @c0b6c12b "
                     "uses --model hf)"),
        "revision": "4600d6bf73ba2cf7037ae7feada03315839ef185",
        "dataset": "cais/wmdp@7125571f22f032c56415e7980f48d877dd830ff8 (bio/chem/cyber test, 3,668)",
        "method": (("zero-shot loglikelihood MCQA equal to lm-eval v0.4.2 local-completions' "
                    "_loglikelihood_tokens (log P(continuation | context)); prompts from "
                    "_encode_pair over tokenizer.encode defaults (no BOS for Gemma 4, "
                    "bos_prepended recorded), sent whole like the HF-backend reference run; "
                    "each question is first scored with one next-token request (the context's "
                    "token ids, max_tokens=1, temperature=0.0, seed=1234, logprobs=20, "
                    "logprob_token_ids = the four single-token continuations, "
                    "return_tokens_as_token_ids; the four log-probabilities read by id at the "
                    "position after the context, which equals lm-eval's summed continuation "
                    "log-probability for single-token continuations); a question with a "
                    "multi-token continuation, a reply that does not list all four ids, or a "
                    "server that does not take that request (HTTP 4xx other than 408/429) is "
                    "scored with lm-eval's own request at batch_size=1 (one token-id prompt per "
                    "request, echo=True, max_tokens=0, temperature=0.0, logprobs=10, seed=1234; "
                    "get_result sum of token_logprobs after the context); each trace records "
                    "which request scored it and the result counts "
                    "questions_scored_by_next_token_request / "
                    "questions_scored_by_echo_request; acc pooled over subsets; --shard before "
                    "--eval-limit (shard recorded); a first question that neither request can "
                    "score -> infra_required; a question with no usable reply "
                    "after the retries is a harness error: left out of accuracy and "
                    "category_accuracy (scoring_excluded, request_harness_errors, effective_n), "
                    "completed_with_errors, named in status_reason, not comparable; accuracy null "
                    "with accuracy_withheld_reason when none is scored; --thinking recorded as "
                    "not applied (thinking_applied=False)")),
    },
    "api_bank": {
        "synced": "2026-10-10",
        "upstream": "github.com/AlibabaResearch/DAMO-ConvAI (api-bank)",
        "revision": "f30ccf22b4e2617fab32958d4c03f5c1f2e7dfcf",
        "dataset": ("liminghao1630/API-Bank@12e8158b7628c168f07e8f31fbbe3445e99f44cf (L3) + "
                    "checkout lv1-lv2-samples (L1/L2)"),
        "checksum": ("sha256:66978bd75f70048885822e9c6955b89cc3c837b467010f5ffedc7292a4b265d0 "
                     "(digest of the scored checkout files)"),
        "method": ("L1/L2 messages built as evaluator.py (prompt as system, User/AI turns, earlier"
                   " API results as system messages; nothing merged), the prompts evaluator.py's "
                   "byte for byte, trailing spaces included (PROMPT_SHA256), scored via the "
                   "checkout's Evaluator (execution + check_api_call_correctness); L3 via lv3_apis"
                   " replica; all on the reasoning-stripped reply; no sampling field sent, as "
                   "evaluator.py chatgpt.call(messages) (OfficialSampling send_temperature=False, "
                   "send_max_tokens=False: ChatGPTWrapper/GPT4Wrapper post model + messages only, "
                   "so no temperature and no max_tokens); harness errors: a step whose request "
                   "produced no reply, and a failed call step to one of the APIs whose backend is "
                   "outside the process (Translate via upstream's pinned googletrans==3.1.0a0 with"
                   " httpx 0.13.3, run in the local image gbench-api-bank:googletrans-3.1.0a0 with"
                   " apis/translate.py unchanged and Translator(raise_exception=True), so a Google"
                   " reply other than HTTP 200 raises instead of returning the untranslated text; "
                   "the proxy variables are read when a Translator is built and passed to the "
                   "container, a proxy URL with a password is refused, and the image's releases, "
                   "labels and worker sha256 and the metadata-server block are checked before the "
                   "run (infra_required); recorded as translate_backend; Dictionary via "
                   "api.dictionaryapi.dev, ToolSearcher via a sentence-transformers model) whose "
                   "gold call fails on the backend too, are left out of the level means, counted "
                   "in harness_errors, named in leaderboard_comparable_reason, status "
                   "completed_with_errors, score withheld when no call step is scored; "
                   "ToolSearcher's model loaded when the checkout is checked (infra_required if it"
                   " cannot); ALL_PROXY set to the empty string while scoring when unset "
                   "(apis/translate.py assigns it back unconditionally); ROUGE-L response track; "
                   "L3 sample success only when every L3 call step ran, over the dialogues with a "
                   "known outcome; total_questions and effective_n count the call-track steps; "
                   "headline = mean call Correctness L1-L3, a gbench aggregate of the per-level "
                   "numbers API-Bank reports (leaderboard_comparable=False); checkout verified by "
                   "file digest at load"),
    },
    "humanitys_last_exam": {
        "synced": "2026-10-09",
        "upstream": "centerforaisafety/hle@22ed3074b1e7b134bcbc09028d0ba320839b0655",
        "revision": "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        "dataset": "cais/hle (test)",
        "method": ("official SYSTEM_PROMPT as system turn (text then image); official "
                   "JUDGE_PROMPT verbatim as the only message, structured output (ExtractedAnswer "
                   "schema), max_output_tokens 4096 and no temperature, as run_judge_results.py "
                   "extract_answer sends it (max_completion_tokens=4096, no temperature), over "
                   "the Gemini cascade (judge_sampling); a judge reply without a yes/no verdict "
                   "(e.g. stopped before the ExtractedAnswer JSON closed, also counted in "
                   "judge_replies_incomplete_json) is JUDGE_FAILED: excluded from accuracy and "
                   "calibration, counted and named (upstream counts an unjudged question wrong "
                   "after telling the user to rerun the judge); question and gold answer as the "
                   "dataset stores them; judge_model and judge_reply per trace; a reply that "
                   "reaches the end of the context window (no max_tokens sent) is the protocol's "
                   "outcome (replies_at_context_window), not a budget cut-off; accuracy + Wald "
                   "CI; calib_err(beta=100) verbatim on judge-extracted confidence; sampling "
                   "temperature 0 without --thinking (README; sent to non-reasoning models before "
                   "136e12e), no temperature with --thinking (reasoning-model path; the request "
                   "line is commented out since 136e12e); no max_tokens in either mode (README: "
                   "'Our evaluation uses the maximum number of tokens available'; "
                   "--max_completion_tokens default None; --max-output-tokens not applied); "
                   "leaderboard_comparable=False (judge)"),
    },
    "medxpertqa": {
        "synced": "2026-10-10",
        "upstream": ("github.com/TsinghuaC3I/MedXpertQA eval/@5f1db162 (main.py, "
                     "config/prompt_templates.py, utils.py, model/api_agent.py)"),
        "revision": "7e7c465a68eb2b866926bfa59c8c9d17a8daba65",
        "dataset": "TsinghuaC3I/MedXpertQA (Text 2,450 + MM 2,000, test; images.zip)",
        "method": ("official two-turn zero-shot CoT scored by upstream answer_cleansing; "
                   "overall + Text/MM accuracy; missing images hard-error; sampling as "
                   "APIAgent.get_response's chat branch on both turns, in both modes (its other "
                   "branch and APIAgent.__init__'s 8192 are keyed on model names containing o1, "
                   "o3, qwq, qvq or reasoner, not settings for a served model): system turn, "
                   "temperature 0 (scripts/run.sh), seed 0; max_tokens 2048 sent exactly "
                   "without --thinking; with --thinking the same request with the run's "
                   "--max-output-tokens (else the thinking floor) in place of the 2048, which is "
                   "recorded as not applied to reasoning; "
                   "logprobs=True not requested (confidence only); comparable only full, "
                   "unsharded, official temperature, every request of both turns answered and "
                   "no reply of either turn cut off by the run's token budget "
                   "(replies_cut_off_by_budget)"),
    },
    "deepsearch_qa": {
        "synced": "2026-10-09",
        "upstream": ("google/deepsearchqa + Kaggle starter notebook "
                     "andrewmingwang/deepsearchqa-starter-code v1"),
        "revision": "b2623f8653065c2672de6d941fc5434cd652376c",
        "dataset": "google/deepsearchqa (DSQA-full.csv, eval)",
        "method": ("raw problem prompt; web_search tool loop with no step limit or forced answer; "
                   "no temperature or max_tokens sent (the notebook's model call sets only the "
                   "google_search tool); web_search = Gemini Google Search grounding (search_tool "
                   "cascade) where the notebook's model runs Google Search itself: each call is "
                   "one Gemini generateContent request with the model's query, returning the "
                   "cited sources' titles and URLs with segments of that Gemini answer (no page "
                   "text), as the tool description (gaia.GROUNDED_SEARCH_DESCRIPTION), "
                   "search_backend_kind and leaderboard_comparable_reason say; tool_transcript on "
                   "every trace; notebook rater prompt verbatim via the Gemini cascade, called "
                   "with no generation config as the notebook's rater is, each rated trace "
                   "keeping judge_model, judge_raw and judge_explanation (the notebook's "
                   "ItemRating fields); harness errors (a request with no reply or one failing "
                   "mid tool loop, a web_search call no grounding model answered) are not rated, "
                   "left out of every metric, counted in harness_errors, named in "
                   "leaderboard_comparable_reason, status completed_with_errors, score withheld "
                   "when nothing is rated; an empty reply the server returned stays Fully "
                   "Incorrect; Fully Correct headline with per-item P/R/F1 per aggregate_ratings; "
                   "leaderboard_comparable=False"),
    },
    "frames": {
        "synced": "2026-10-09.2",
        "upstream": "arXiv 2409.12941 v3 (auto_rate_prompt.tex)",
        "revision": "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef",
        "dataset": "google/frames-benchmark (test)",
        "method": ("paper Naive Prompt (closed book, single step); paper auto-rating prompt "
                   "verbatim (U+2019 and U+2013 kept) + system turn over the Gemini cascade; last "
                   "Decision TRUE/FALSE (none -> JUDGE_FAILED: an autorater failure, excluded "
                   "from the accuracy, counted and named); accuracy; judge_model and judge_reply "
                   "per trace; leaderboard_comparable=False (judge); the paper publishes no "
                   "autorater sampling settings, so gbench's deterministic judge temperature 0 "
                   "(judge_sampling)"),
    },
    "wildbench": {
        "synced": "2026-10-10.2",
        "upstream": ("allenai/WildBench@d6b8dcaf377d173d031980f97c16e1a82618c03d (src/eval.py, "
                     "evaluation/eval_template.score.v2.md + eval_template.pairwise.v2.md, "
                     "src/openai_batch_eval/batch_results_format.py, "
                     "leaderboard/data_dir/_create_tables.py + _merge_results.py + all_stat.json, "
                     "leaderboard/wb_elo.py, scripts/_common_vllm.sh) + leaderboard Space "
                     "allenai/WildBench@4b94d1d856ed41580ec150b53c093460a3fb9d0c "
                     "(data_dir/wb_elo_results.LM=-1.json / LM=1500.json config)"),
        "revision": "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec",
        "dataset": ("allenai/WildBench@26c49eb39d7d5ce2099b0bbafed5a88dcce954ec (v2, test, 1,024 "
                    "tasks) + "
                    "allenai/WildBench-V2-Model-Outputs@d6755bc68220df853c0825a733430f73f5af2501 "
                    "(WB-Reward references gpt-4-turbo-2024-04-09, claude-3-haiku-20240307, "
                    "Llama-2-70b-chat-hf) + WB-Elo pool: the 63 configs with upstream score "
                    "results (eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13) at "
                    "d6755bc68220df853c0825a733430f73f5af2501"),
        "checksum": ("sha256:e1fbdc5d46deca34e77f92991c23b0476860e26adf2bf28379a3bb3bf00d3ed6 "
                     "(v2/test-00000-of-00001.parquet); references "
                     "sha256:0e94087c32882ff8687f9a5f0a13a762ba74a77311fac58d07eb750dfe9b44d2 "
                     "(gpt-4-turbo-2024-04-09), "
                     "sha256:6bfda71a23bcb86d76279f6b23040af82c8e5100c3c3e0cf22885f97d40caafa "
                     "(claude-3-haiku-20240307), "
                     "sha256:2234f3794abc4b86fa52e11e62a983b7ffad7459b80185ef1d3f2f5b93c8d133 "
                     "(Llama-2-70b-chat-hf), verified at load; 63 pool parquets sha256 in "
                     "wildbench.WB_ELO_POOL (verified at load); all_stat.json "
                     "sha256:da75f4b2a60d17c2b59f5a9c539bfd87ca1b2369e5da042b3ceaa1ea833c82e9 "
                     "(Arena Elo starting points)"),
        "method": ("official batch pipeline over the Gemini cascade in JSON mode at temperature 0 "
                   "(src/eval.py) with its max_tokens=1024 sent as max_output_tokens (judge "
                   "replies stopped there counted in judge_replies_at_max_output_tokens); "
                   "judge_sampling; WB-Score: eval_template.score.v2.md verbatim, every reply "
                   "judged (an empty one too), batch_results_format parse (json.loads + 'score', "
                   "else JUDGE_FAILED, excluded), a request that got no reply from the served "
                   "endpoint = harness error (not judged in either mode, scoring_excluded, left "
                   "out of WB_score, in no WB-Reward total, no WB-Elo vote); WB_score.task_macro "
                   "(headline; replies empty or ending \"... (truncated)\" skipped in group means "
                   "and Length, as _create_tables) + WB_score = (mean_raw-5)*20 (weights "
                   "0.5/1.25/1/0.75/1.25 over their sum, x10) + WB_score.<group> + Length; "
                   "WB-Reward: eval_template.pairwise.v2.md verbatim vs the three reference "
                   "replies, shorten() to 1,000 words, empty reply shown as '[This model response "
                   "is empty.]', A/B order random.seed(42) one draw per task in v2 order, choice "
                   "read by batch_results_format (else dropped), _create_tables reward "
                   "(+1/+0.5/0/-0.5/-1; truncated or empty comparisons count 0 in the total; "
                   "length margin K) and _merge_results columns for K in "
                   "-1,100,300,500,1000,1500,2000,3000, headline task_macro_reward.K=-1 + "
                   "mixture_of_rewards + RewardScore_Avg; sampling temperature 0, top_p 1.0 and "
                   "max_tokens 4096 without --thinking (scripts/_common_vllm.sh; with --thinking "
                   "the run's --max-output-tokens, the cap recorded as not applied to reasoning); "
                   "WB-Elo: the 63 pool models' published replies re-graded with the same template"
                   " and Gemini cascade (cached per judge cascade and judge config; empty or "
                   "truncated pool replies not sent, since they cannot vote; pool files read one "
                   "model at a time when the gradable replies are listed and again when graded; a "
                   "pool prompt is built only when graded), get_all_votes(margin 3, tie_margin 2),"
                   " compute_single_round K=4 dynamic interval 100, 128 rounds in one process "
                   "seeded with 42 (a gbench choice: upstream's rounds run in forked "
                   "ProcessPoolExecutor workers whose random state is reseeded from OS entropy, so"
                   " they are not reproducible), Arena Elo (hard-en) starting points; WB_Elo "
                   "(length_margin -1) + WB_Elo_LC (1500); on upstream GPT-4o scores all 67 "
                   "published models fall inside the published 95% CI (max |diff| 1.6 / 1.5); "
                   "judge outages and Gemini refusals (JUDGE_REFUSED with block reasons, not "
                   "cached) excluded from every score, counted per mode (WB-Score, WB-Reward, "
                   "WB-Elo pool), named with the refused task ids, status completed_with_errors; "
                   "WB-Reward comparisons and WB-Elo pool verdicts whose judge reply is not "
                   "readable JSON are dropped as batch_results_format drops them and counted as "
                   "judge-side harness errors (wb_reward_judge_failures, "
                   "wb_elo_detail.pool_judge_failures, judge_failures_dropped with how many "
                   "stopped at max_output_tokens), status completed_with_errors, named in "
                   "leaderboard_comparable_reason; per-category rows show the category WB_score "
                   "(pass@7 kept as pass_rate_at_7); leaderboard_comparable=False (judge)"),
    },
    "livebench": {
        "synced": "2026-10-11",
        "upstream": "github.com/LiveBench/LiveBench",
        "revision": "0baa4aa7a78dc0276c0ccf28779dba1113580e18",
        "dataset": ("livebench/{coding,data_analysis,instruction_following,language,math,reasoning} "
                    "(6 core categories, agentic_coding excluded; questions fixed by "
                    "--livebench-release-option, pinned to release 2024-11-25 by default "
                    "(LIVEBENCH_RELEASE: README.md at 0baa4aa7 says not all questions of the "
                    "current release are public on Hugging Face and to pass 2024-11-25, the most "
                    "recent public questions; 1000 active there: coding 128, data_analysis 150, "
                    "instruction_following 200, language 140, math 232, reasoning 150; every "
                    "later release has no public coding question, 682 in all); "
                    "GBENCH_LIVEBENCH_RELEASE overrides it, latest = LiveBench's own default, "
                    "unpinned and non-comparable; HF heads 2025-04-07, e.g. math @ "
                    "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb); the question and ground-truth "
                    "rows are read at run time from each livebench/<category> Hugging Face "
                    "dataset's main branch, not pinned to a revision"),
        "method": ("delegated to LiveBench run_livebench.py + ground-truth scorers in the "
                   "gbench-livebench image; the served model's own LiveBench config "
                   "(livebench/model/model_configs/*.yml, looked up in the image with "
                   "LiveBench's get_model_config under the served id, the --tokenizer id and "
                   "their last path components) is reproduced for the local endpoint: its "
                   "api_kwargs on its leaderboard provider (default updated with the "
                   "default_provider entry, as gen_api_answer.py setup_model merges them), i.e. "
                   "temperature, top_p and other sampling fields, the token cap and "
                   "prompt_prefix/preserve_reasoning, with its reasoning switch mapped to "
                   "chat_template_kwargs.enable_thinking (gemma-4-31b-it at 0baa4aa7: "
                   "temperature 1.0, top_p 0.95, max_tokens 65536, reasoning enabled; the "
                   "request verified against a stub endpoint in an image built from 0baa4aa7, "
                   "2026-09-25); a field with no local counterpart, or a reasoning switch other "
                   "than --thinking, is non-comparable (livebench_model_config); max_tokens by "
                   "the token-limit policy: no-think the official cap exactly (the model "
                   "config's, else gen_api_answer.py --max-tokens default 4096; "
                   "--max-output-tokens not applied), --thinking the model config's cap never "
                   "below --max-output-tokens, else the run's budget (the 4096 default not "
                   "applied to reasoning); the one fixed max_tokens LiveBench sends is lowered "
                   "to fit the served window whenever it would leave the longest prompt less "
                   "than 8192 tokens (or half the window), whatever its source, and recorded "
                   "(max_output_tokens_source ending in fit-to-window, "
                   "official_sampling.max_tokens_sent, and max_output_tokens_note, which then "
                   "states what the token-limit policy gives and the max_tokens each request "
                   "was sent, never naming an unlowered value as sent); an official cap lowered "
                   "that way is non-comparable, naming the --max-model-len it needs; replies "
                   "cut off by a max_tokens other than the official cap are counted from "
                   "LiveBench's answer rows copied out of the container, with and without "
                   "thinking (eval_status token_exhaustion; total_output_tokens at max_tokens; "
                   "above it, completion tokens = (total + the answer's --tokenizer tokens) / 2 "
                   "within 16 tokens of max_tokens, because LiveBench's litellm path adds the "
                   "server's reasoning_tokens to vLLM's completion_tokens, which already "
                   "include them: checked in the 0baa4aa7 image against a stub streaming "
                   "endpoint, completion_tokens 50 + reasoning_tokens 30 recorded as 80, "
                   "2026-09-26) and make the run non-comparable (replies_cut_off_by_budget, "
                   "replies_cut_off_in_reasoning), as do answer rows that cannot be found or "
                   "checked (replies_cut_off_uncounted: no usage, several turns, or no "
                   "--tokenizer); LiveBench's api_error rows other than empty replies (requests "
                   "that failed at the endpoint, which LiveBench scores 0 inside its averages) "
                   "are harness failures: failed_requests and failed_requests_by_category, "
                   "status completed_with_errors, non-comparable, and the headline is withheld "
                   "(accuracy null with accuracy_withheld_reason) together with the score of "
                   "every category that has one (dimension_scores_withheld); requests the "
                   "served vLLM server aborted during the run (server_aborted_requests) cannot "
                   "be placed and withhold the headline and every category score; failed rows "
                   "are not re-asked (LiveBench's --retry-failures would also re-ask the "
                   "model's empty replies); no --force-temperature by default (get_answer uses "
                   "temperature 0 or the question's required_temperature; an operator "
                   "temperature is forwarded as --force-temperature, or written into the config "
                   "when the model config sets a temperature, and makes the run non-comparable; "
                   "official_sampling recorded); enable_thinking via model-config api_kwargs "
                   "chat_template_kwargs (verified on a stub endpoint); the image's LiveBench "
                   "commit is read (livebench_image_commit): any commit other than 0baa4aa7, or "
                   "none readable (no .git in the build context), is non-comparable; default "
                   "image gbench-livebench:0baa4aa7 (LIVEBENCH_IMAGE_DEFAULT: "
                   "docker/livebench.Dockerfile built with a LiveBench checkout at 0baa4aa7 as "
                   "the context, the NLTK and spaCy data its instruction_following grading "
                   "fetches at run time (punkt, punkt_tab, averaged_perceptron_tagger, "
                   "stopwords, en_core_web_sm) baked in, a failed download failing the build, "
                   "never pulled, no fallback to an untagged gbench-livebench; "
                   "GBENCH_LIVEBENCH_IMAGE overrides); the image carries one recorded local "
                   "patch (LIVEBENCH_PATCH, applied by docker/livebench.Dockerfile at build "
                   "time with sha256 checks before (dd6bdce6) and after (19ed4fef), checked in "
                   "the image before each run, recorded on every result as livebench_patch): "
                   "gen_ground_truth_judgment.py line 480 list(set(...)) over old-format "
                   "instruction_following question dicts (TypeError: unhashable type: 'dict', "
                   "so upstream never grades that category; introduced in 366e385, 2025-12-12, "
                   "unchanged at main 85f874fc) de-duplicated by question_id; verified "
                   "2026-09-26 in gbench-livebench:0baa4aa7 rebuilt from a clone at 0baa4aa7 by "
                   "a model-free grading run of stub answers to all 200 instruction_following "
                   "questions of 2024-11-25 (400 judgments; category score equals the mean of "
                   "the four per-task scores); the pooled upstream call labels every "
                   "instruction_following row with one task, so instruction_following tasks "
                   "answered in unequal numbers, or instruction_following scored with no answer "
                   "row copied out to count them, make the run non-comparable; a category "
                   "LiveBench answered but did not score is listed in categories_unscored "
                   "(completed_with_errors, non-comparable); comparable only for a pinned "
                   "release (the default 2024-11-25 or a GBENCH_LIVEBENCH_RELEASE date, not "
                   "latest), all 6 categories scored, full set, LiveBench's (or the model "
                   "config's) own temperature and the policy's budget; the run container "
                   "(inference and grading) runs on docker's bridge network (never the host "
                   "network), after the metadata-server guard and a TCP check of a loopback "
                   "endpoint (--api-base becomes host.docker.internal); the image checks run "
                   "with --network none; the metadata-server probe container also resolves "
                   "huggingface.co (the questions) and the endpoint's host when it is a name "
                   "(infra_required before the run otherwise); a run with no all_groups.csv "
                   "whose output shows a failed host-name lookup, or in which every request "
                   "failed at the endpoint, reports no score (infra_required); the model's code "
                   "in grading keeps general internet access as upstream; question selection "
                   "for --shard, --eval-limit and a multi-word category subset by question id "
                   "(LiveBench's own get_categories_tasks + load_questions in the image, ids "
                   "sorted, shard first, then stratified_sample by category seed 'livebench', "
                   "run_livebench.py --question-id on live_bench); GBENCH_LIVEBENCH_CATEGORIES "
                   "outside the 6 core names refused before docker; per-question judgment "
                   "traces (model_judgment/ground_truth_judgment.jsonl) and merge_shards "
                   "recomputing show_livebench_result's task->category->average aggregation; "
                   "merge_shards leaves an empty shard out of the headline check (a scored "
                   "shard plus an empty one merges to the union's score) and recomputes over "
                   "the union the comparability reasons only a shard's part of the questions "
                   "causes (instruction_following tasks answered in unequal numbers, categories "
                   "no question of a shard fell in); an endpoint key reaches the endpoint "
                   "through gbench's EndpointRelay on the bridge gateway"),
    },
    "beam_128k": {
        "synced": "2026-10-09",
        "upstream": "github.com/mohammadtavakoli78/BEAM",
        "revision": "b2da22eac88bb0874c64665f13457eb99835774a",
        "dataset": ("Mohammadta/BEAM@3205395e897e7318c7b094ef4e6047b9b82dbb03 (100K split = the "
                    "128K tier)"),
        "method": ("long-context path: chat turns + verbatim final turn, and from a "
                   "conversation's third question on the question turns of its questions 2..k-1 "
                   "before its own, as upstream's probing_question_evaluation appends each "
                   "question turn to the saved history list it reuses by reference "
                   "(_question_tail; question_position and carried_question_turns per trace); "
                   "prune_from_tail to the served window leaving upstream's 2,000-token answer "
                   "room (the run budget with --thinking), sized for the token-longest question "
                   "tail (its turn plus the carried question turns); per-rubric-item unified "
                   "judge prompt (Gemini cascade) on the reply with only inline reasoning blocks "
                   "removed (no whitespace stripped), replies parsed as upstream "
                   "(parse_json_response, then json_repair.repair_json from json_repair 0.44.1 as "
                   "upstream pins it, vendored in _json_repair_vendor (PyPI wheel sha256 "
                   "51d82532c3b8263782a301eb7904c75dce5fee8c0d1aba490287fc0ab779ac50, files "
                   "byte-identical, checked before generation)), every judge reply kept with its "
                   "Gemini model (beam_judge_calls); event_ordering tau_norm via LLM alignment of "
                   "the unstripped reply's lines (beam_alignment_calls) + Kendall tau-b; a failed "
                   "request (no reply) is a harness error left out of the ability means, where "
                   "upstream's answer_generation.py records an 'Error in ...' text as the reply "
                   "and judges it, and so is a judge with no usable score in 3 replies (upstream "
                   "raises): counted in harness_errors, named in leaderboard_comparable_reason, "
                   "status completed_with_errors, score withheld when no question is scored; "
                   "headline = mean of the 10 ability means; sampling temperature 0 and no "
                   "max_tokens (answer_generation.py; the reader is built without one); judge as "
                   "src/llm.py BuildLLm (temperature 0, no max_tokens, no response format; "
                   "judge_sampling); leaderboard_comparable=False"),
    },
    "cimemories": {
        "synced": "2026-10-11",
        "upstream": "github.com/facebookresearch/CIMemories",
        "revision": "e39806e92ec66db120c0f3b73c9cb105e7ff5314",
        "dataset": ("facebook/CIMemories@7a2bde79f747d011b23095e6bb035c6f64127d9f "
                    "(cimemories_raw.json = GitHub data/data_openai_gpt-oss-120b.json)"),
        "checksum": "sha256:08101f8bf2ba387fa639a1218cbc8fb8c528742115f9fc5ee4305b2cd094b3cb",
        "method": ("release pipeline: prompt and persona texts read at run time from "
                   "eval/prompts.yaml (sha256 "
                   "92cfdeb0cf42e2c4281b815c050d982785061314996c9e11b56a5dabcf0e2493) and "
                   "data/persona_gold_labels.py (sha256 "
                   "dc4f093fd1822c86e4bae21d24ea53af8165b85f632d9bf61603409a79e99dbd) at the "
                   "revision (CC BY-NC 4.0, not shipped; cached, parsed with yaml.safe_load / "
                   "ast); persona_gold_labels.py labels (3 Westin personas x 10 samples, non-JSON "
                   "replies dropped, weights 0.26/0.64/0.10, unanimous share/private) with the "
                   "Gemini cascade as labeler called with persona_gold_labels.py Model._resolve "
                   "parameters (no temperature; reasoning_effort minimal sent as Gemini "
                   "thinking_level low, the lowest level the cascade head offers; cached by "
                   "labeler cascade + call parameters + prompt sha256, one writer per entry; "
                   "per-context labeler diagnostics, a Table 2 comparison, mixed labeler models "
                   "and zero-match replies named in leaderboard_comparable_reason); eval.py combo "
                   "full, prompts.yaml level 1, 5 generations with no temperature and no "
                   "max_tokens sent (eval.py AsyncModel without --greedy; the server's default, "
                   "the room left in the context window, applies to each generation; "
                   "--max-output-tokens recorded as not applied); generations that reach the end "
                   "of the window before any message are left out and counted, cut messages "
                   "judged and counted; adversary_reveal judge on the Gemini cascade (0.0), "
                   "parse_json semantics; metrics.py Violation@5 + Completeness (a run with no "
                   "scorable trial, e.g. every context discarded by the substituted labeler, is "
                   "failed with its score withheld, never 0; the trials of a discarded context "
                   "are marked excluded_by_protocol, metrics.py's own exclusion, not a harness "
                   "error); 490/490 task, 1,470/1,470 label and the reveal prompts byte-identical "
                   "to the release's; leaderboard_comparable=False (labeler + judge substituted)"
                   "; REVEAL: an unreadable reply is a judge failure (JUDGE_FAILED), left out; "
                   "labeler JSON failures named in the status"),
    },
    "codeforces": {
        "synced": "2026-10-09",
        "revision": "fbe3f6e903ee854eec2e69e9d96d0306cde59baf",
        "dataset": ("open-r1/codeforces @ fbe3f6e903ee854eec2e69e9d96d0306cde59baf "
                    "(verifiable/test + verifiable-prompts + generated_tests mirror)"),
        "method": ("open-r1 card protocol: dataset's own Python prompt, open-r1's extract_code "
                   "verbatim (huggingface/open-r1@1416fa0cf21595d2083b399a2a0bbddd7f6e9563 "
                   "src/open_r1/rewards.py: the last ```python block, closed by the next ``` "
                   "wherever it stands; a reply with none is no submission and fails, counted in "
                   "replies_without_python_block; the reply is never run as code), all official + "
                   "generated tests (mirror must match the declared count) judged as the piston "
                   "codeforces run script the card points to "
                   "(guipenedo/piston@7a38da202481bb7a85647a2c6cdc00990ad051bc "
                   "packages/codeforces/1.0.0/run), ported for python3: a carriage return at each "
                   "input line end stripped, empty output lines dropped, the expected output read "
                   "with $(cat) and written with echo then stripped the same way; no checker: "
                   "normalize_output (whole-word yes/no folded to YES/NO case-insensitively, "
                   "whitespace runs to one space, trimmed) and equality (diff -bq); "
                   "generated_checker: python checker.py input expected output for at most 10 s, "
                   "accepted iff it exits 0 and prints 1 or 100; input_mode file reads input.txt "
                   "and writes output.txt; stack unlimited; program and checker in bwrap; the "
                   "result records extraction and judge; 3x CPython time limit, memory not "
                   "enforced; a generated-test cap is recorded (max_generated_tests) and named in "
                   "the reason; pass rate not Elo, leaderboard_comparable=False"),
    },
    "gsm8k": {
        "synced": "2026-09-25",
        "upstream": ("github.com/EleutherAI/lm-evaluation-harness "
                     "lm_eval/tasks/gsm8k/gsm8k-cot.yaml (gsm8k_cot v3.0)"),
        "revision": "d6de81643928d653435c431bae19945d41d32520",
        "dataset": "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866 (main, test, 1,319)",
        "method": ("gsm8k_cot as local-chat-completions + apply_chat_template "
                   "(fewshot_as_multiturn): 8 yaml exemplars as user 'Q: ..\\nA:' / assistant "
                   "turns; without thinking the request carries until [Q:, </s>, <|im_end|>] as "
                   "stop and max_gen_toks 256 (api_models.py default) as max_tokens, as "
                   "LocalChatCompletion._create_payload sends them; the 256 is "
                   "OfficialSampling.max_tokens, sent exactly by base's token-limit policy "
                   "(--max-output-tokens not applied; replies stopping at 256 counted in "
                   "replies_at_official_max_tokens, not a truncation); with thinking no stop is "
                   "sent, the request carries the run budget (--max-output-tokens, else the "
                   "thinking floor; 'official cap not applied to reasoning' recorded) and the "
                   "reply is cut at until after the reasoning is removed, as lm-eval's HF "
                   "backend applies until; temperature 0 + seed 1234 in both modes "
                   "(OfficialSampling; lm-eval's openai-chat-completions branch for o1/o3/o4/5 "
                   "model names, temperature 1 and no stop, fits OpenAI's hosted API and is not "
                   "in local-chat-completions, so it is not applied); strict-match and "
                   "flexible-extract RegexFilter + take_first, exact_match with ignore_case and "
                   "regexes_to_ignore; headline = strict-match, flexible_extract_accuracy "
                   "alongside; comparable for full 8-shot single-sample T=0 runs; flags n_shot, "
                   "samples, operator temperature, replies over 256 tokens (thinking runs), a "
                   "no-think request sent a max_tokens other than 256, failures, --eval-limit"),
    },

    # Reconciled at the code level (loader dataset/split/config + scorer confirmed
    # against the canonical metric documented in each runner and against docs/RESOURCES.md).
    # Pillar 1: general knowledge and scientific reasoning.
    "gpqa": {
        "synced": "2026-10-10.3",
        "upstream": ("github.com/openai/simple-evals @ 652c89d0 (gpqa_eval.py GPQAEval, "
                     "simple_evals.py n_repeats=10)"),
        "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6",
        "dataset": "Idavidrein/gpqa (gpqa_main, train, 448)",
        "method": (("GPQAEval: examples*10 with one random.Random(0).sample(range(4),4) per copy "
                    "in upstream order, [Correct, Inc1, Inc2, Inc3] permuted; "
                    "QUERY_TEMPLATE_MULTICHOICE user message + first ANSWER_PATTERN_MULTICHOICE "
                    "match; accuracy = mean over all copies (+score:std, per-repeat); "
                    "simple-evals' chat sampler declared once as gpqa.SIMPLE_EVALS_SAMPLING (no "
                    "reasoning block) and applied per mode by base's token-limit policy: both "
                    "modes = simple_evals.py 'gpt-4o' entry, lines 236-240 "
                    "(ChatCompletionSampler: system message OPENAI_SYSTEM_MESSAGE_API 'You are a "
                    "helpful assistant.', T=0.5); no-think sends its max_tokens=2048 as the "
                    "official cap, exactly, --max-output-tokens not applied (replies cut at 2048 "
                    "scored on what arrived, counted in replies_at_sampler_max_tokens and "
                    "replies_at_official_max_tokens, not a truncation); thinking sends the run's "
                    "--max-output-tokens instead (official cap not applied to reasoning: "
                    "max_tokens_applied false, max_tokens_sent); simple_evals.py's o-series "
                    "entries, lines 145-212 (ResponsesSampler reasoning_model=True for "
                    "o3/o4-mini/o1-pro, OChatCompletionSampler for o1/o1-preview/o1-mini/o3-mini: "
                    "no system message, temperature or max_tokens), are keyed on OpenAI model "
                    "names and are not reasoning settings for a served model, which takes the "
                    "chat sampler in both modes; --shard over questions before --eval-limit "
                    "(every copy of a question in one shard); leaderboard_comparable=False for "
                    "--eval-limit, repeats != 10, an operator temperature, any "
                    "failed/timed-out/empty/over-context request (a request with no reply is a "
                    "harness error, left out of the score; an empty reply is scored wrong), a "
                    "no-think request sent a max_tokens other than 2048, or a reply cut off "
                    "before finishing at anything but the no-think 2048 (a thinking reply at the "
                    "run's budget or a context-clamped max_tokens, a no-think reply at a clamped "
                    "max_tokens)"
                    "; a null-content reply is asked again, the same request, as simple-evals' ChatCompletionSampler does (content is None -> 'OpenAI API returned empty response; retrying', no limit), through swebench_common.upstream_reask_relay with reply_content_is_null, up to gpqa.NULL_REPLY_REASKS=5 re-asks; an empty string is a reply; upstream_reask records it; a reply still null after the re-asks is scored as base scores it and makes the run non-comparable"
                    "; only a reply the model ended itself (finish_reason stop) is re-asked: a "
                    "reply cut off at the token budget is the model's own outcome, never re-asked, "
                    "scored as base scores a truncated reply and flagged (cut_off_not_reasked); "
                    "the re-asks stop before the client's per-request timeout "
                    "(EndpointRelay.reask_deadline), and a chain that ends on a reply still "
                    "matching the rule (reasks_exhausted, of them reasks_cut_short) passes that "
                    "reply on to be scored, never a harness error; upstream_reask also records the "
                    "classes of the empty replies seen (answer_in_thinking_seen, "
                    "tool_call_dropped_seen, unclassified_empty_seen, answer_in_thinking_rate_seen)")),
    },
    "gpqa_diamond": {
        "synced": "2026-10-10.3",
        "upstream": ("github.com/openai/simple-evals @ 652c89d0 (gpqa_eval.py GPQAEval default "
                     "variant, n_repeats=10)"),
        "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6",
        "dataset": ("Idavidrein/gpqa (gpqa_diamond, train, 198; rows field-identical and in the "
                    "same order as simple-evals gpqa_diamond.csv)"),
        "method": (("shared with gpqa: 10 repeats with simple-evals' exact permutations, first "
                    "Answer: X match, mean over 1,980 copies; gpqa.SIMPLE_EVALS_SAMPLING per mode "
                    "through base's token-limit policy (both modes: the gpt-4o entry's "
                    "ChatCompletionSampler system message, T=0.5; no-think: max_tokens=2048 sent "
                    "exactly, --max-output-tokens not applied; thinking: the run's "
                    "--max-output-tokens, the 2048 recorded as not applied to reasoning; "
                    "simple-evals' o-series samplers, keyed on OpenAI model names, are not "
                    "reasoning settings for a served model); --shard over questions before "
                    "--eval-limit; leaderboard_comparable=False for --eval-limit, repeats != 10, "
                    "an operator temperature, any failed/timed-out/empty/over-context request (a "
                    "request with no reply is a harness error, left out of the score; an empty "
                    "reply is scored wrong), a no-think request sent a max_tokens other than "
                    "2048, or a reply cut off before finishing at anything but the no-think 2048 "
                    "(a thinking reply at the run's budget or a context-clamped max_tokens, a "
                    "no-think reply at a clamped max_tokens)"
                    "; a null-content reply is asked again, the same request, as simple-evals' ChatCompletionSampler does (content is None -> 'OpenAI API returned empty response; retrying', no limit), through swebench_common.upstream_reask_relay with reply_content_is_null, up to gpqa.NULL_REPLY_REASKS=5 re-asks; an empty string is a reply; upstream_reask records it; a reply still null after the re-asks is scored as base scores it and makes the run non-comparable"
                    "; only a reply the model ended itself (finish_reason stop) is re-asked: a "
                    "reply cut off at the token budget is the model's own outcome, never re-asked, "
                    "scored as base scores a truncated reply and flagged (cut_off_not_reasked); "
                    "the re-asks stop before the client's per-request timeout "
                    "(EndpointRelay.reask_deadline), and a chain that ends on a reply still "
                    "matching the rule (reasks_exhausted, of them reasks_cut_short) passes that "
                    "reply on to be scored, never a harness error; upstream_reask also records the "
                    "classes of the empty replies seen (answer_in_thinking_seen, "
                    "tool_call_dropped_seen, unclassified_empty_seen, answer_in_thinking_rate_seen)")),
    },
    "arc_agi": {
        "synced": "2026-10-10",
        "upstream": ("github.com/arcprize/arc-agi-benchmarking (harness) + "
                     "github.com/fchollet/ARC-AGI (data)"),
        "revision": "28e67d54b05df5be10281892243c509a42a874f1",
        "dataset": ("github.com/fchollet/ARC-AGI@399030444e0ab0cc8b4e199870fb20b863846f34 "
                    "data/evaluation (ARC-AGI-1 evaluation, 400 tasks / 419 pairs; codeload "
                    "archive, cached under $GBENCH_CACHE_DIR/arc_agi)"),
        "checksum": "git-tree:4a19f8327d45dda5e1809ee47be08bf1484b493b",
        "method": (("official data verified against the commit's data/evaluation git tree on "
                    "every load (download or cache; mismatch/unreachable -> infra_required); "
                    "verbatim system_prompt.txt; utils/parsing.py parse_and_validate_json "
                    "verbatim + schemas.Attempt pydantic Union[str, List[List[int]]] (lax "
                    "coercion); ARCTester --retry_attempts 2 (a failed try is asked again once; "
                    "null attempts scored wrong; an attempt whose last try got no reply from the "
                    "endpoint is a harness error, left out of pass@2; effective_n = tasks "
                    "scored); exact grid match; pass@2 per test pair averaged per task "
                    "(score_task / score_submission); --shard over whole tasks before "
                    "--eval-limit; no temperature sent (adapter kwargs = model config fields; "
                    "OfficialSampling send_temperature=False); comparable for full 2-attempt runs "
                    "in which every attempt got a server reply, no attempt's last try was cut off "
                    "at the max_tokens budget (counted by finish_reason length / status "
                    "TRUNCATED, non_convergent relabels included; attempts_cut_off; the same "
                    "count drives the >10% status rule), and no operator temperature")),
    },
    "mmlu_redux": {
        "synced": "2026-10-10",
        "upstream": ("HELM MMLU (stanford-crfm/helm get_mmlu_spec + "
                     "output_format_instructions=mmlu; adapters identical at v0.5.1 ccf1df596b "
                     "and 63754d05), as the MMLU-Redux authors score models (arXiv:2406.04127 "
                     "Sec. 3.2, App. B)"),
        "revision": "372ea425445d51e1ba1188c56e5e893f8138621f",
        "dataset": ("edinburgh-dawg/mmlu-redux-2.0 (57 subjects, test; error_type ok = 5,330 "
                    "rows) + cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe dev (in-context "
                    "examples)"),
        "method": (("multiple_choice_joint 5-shot with the instruction-tuned output-format "
                    "line, sample_examples(seed=0) port, temperature 0 and max_tokens=1 in "
                    "OfficialSampling (recorded); base's token-limit policy: max_tokens=1 "
                    "exactly without thinking (--max-output-tokens not applied; one-token stops "
                    "counted in replies_at_official_max_tokens, the protocol; an empty "
                    "one-token completion is scored 0 as a non-letter reply); with --thinking "
                    "the reasoning channel is on and each request is sent the run budget fitted "
                    "to the window (over-context prompts recorded as OVER_CONTEXT), 'official "
                    "cap not applied to reasoning' recorded, temperature stays 0 (HELM reaches "
                    "a vLLM model through VLLMChatClient, an OpenAIClient whose reasoning-model "
                    "branch, openai_client.py@63754d05 lines 286-314, is taken only for model "
                    "names matching o<digits> or gpt-5), reply = message content with any "
                    "inline thought removed, classified by base.classify_reply: replies cut off "
                    "at the budget scored on what arrived and counted (truncated_responses, "
                    "replies_cut_off_in_reasoning, looping_responses), replies with the answer "
                    "inside the thinking channel scored 0 as the model's outcome "
                    "(answer_in_thinking), replies with nothing counted as failed "
                    "(empty_responses), completed_with_errors when "
                    "more than 10% are cut off or any request failed; a request with no reply "
                    "from the served endpoint (failed, timed out, over the context window) is a "
                    "harness error left out of accuracy, subject_mean_accuracy and "
                    "category_accuracy and named in status_reason, each named in "
                    "leaderboard_comparable_reason; a thinking run is flagged non-comparable; "
                    "exact match after output mapping; verified vs HELM v1.3.0 GPT-4o run: "
                    "14,042/14,042 prompts byte-identical, exact_match agrees 14,042/14,042; "
                    "headline = EM over all ok rows combined (subject mean alongside); --shard "
                    "before --eval-limit (shard recorded); mode-aware WARNING and "
                    "leaderboard_comparable=False when >10% of the replies read are not a "
                    "letter (non_letter_replies: every answered reply without thinking, every "
                    "complete reply with it); each trace's sample_key includes the row's "
                    "position in its subject's pinned test split (row_index), so rows with "
                    "identical content stay distinct when gbench --merge de-duplicates the "
                    "shards' traces; the unsharded selection and scoring are unchanged; "
                    "leaderboard_comparable=True for full no-think T=0 runs with no failed "
                    "requests and at most 10% non-letter replies")),
    },
    "lab_bench": {
        "synced": "2026-10-09",
        "upstream": "github.com/Future-House/LAB-Bench (scripts/score_baseline.py zero-shot CoT)",
        "revision": "998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838",
        "dataset": ("futurehouse/lab-bench@5c77cec648430f30611808808861eb86f81d5eaa (all 8 "
                    "categories, train = 1,967 public; rows equal the repo *-public.jsonl at "
                    "998a8e0a)"),
        "method": (("task.py question builders (ProtocolQA protocol, SuppQA title/DOI, "
                    "FigQA/TableQA images after text), randomize_choices with REFUSE_CHOICE "
                    "(seeded), MCQ_INSTRUCT_TEMPLATE + COT_PROMPT, chembench prepare_mcq_answer + "
                    "run_regex on the raw reply (no gbench extractor); no sampling fields and no "
                    "max_tokens sent (score_baseline.py model_kwargs={'model'}; "
                    "labbench/openai_utils.py create(messages=[msg], **model_kwargs), reused by "
                    "the anyscale and vertex agents; OfficialSampling send_temperature=False and "
                    "send_max_tokens=False: --max-output-tokens not applied, prompts still "
                    "counted for the over-context refusal); accuracy/precision/coverage per "
                    "category and subtask; headline = mean of category accuracies; --shard before "
                    "--eval-limit; image preflight -> infra_required; "
                    "leaderboard_comparable=False (public split only; headline averages "
                    "categories); a request with no reply, including a prompt over the served "
                    "window (which upstream answers as unsure), is a harness error left out of "
                    "every metric")),
    },
    "healthbench": {
        "synced": "2026-10-10.3",
        "upstream": "github.com/openai/simple-evals (healthbench_eval.py)",
        "revision": "652c89d0ca9df547706735883097e9537d40dc47",
        "dataset": ("openai/healthbench@40ee1968852fc57f625934251ac22be47077a8fb "
                    "(2025-05-07-06-14-12_oss_eval.jsonl, = simple-evals INPUT_PATH)"),
        "checksum": "sha256:e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f",
        "method": (("per-rubric-item GRADER_TEMPLATE verbatim with '[points] criterion' "
                    "(Gemini-cascade grader, the call as grading_sampler: system message \"You are "
                    "a helpful assistant.\", temperature 0.5, no response format (the reply read "
                    "by parse_json_to_dict); its max_tokens 2048 for the reply plus the Gemini "
                    "thinking allowance 8192 as max_output_tokens, replies still cut there, on "
                    "every attempt, a retried one included, and outages after it counted and "
                    "named); a request "
                    "that got no reply from the served endpoint is a harness error: not graded "
                    "(not_judged_failed_request), scoring_excluded, left out of the mean, the 0.5 "
                    "pass rate and the per-tag means, counted once in failed_requests and named "
                    "in status_reason and leaderboard_comparable_reason (simple-evals grades a "
                    "request its API rejects as 'No response (bad request).'); unclipped "
                    "per-example score, clipped mean overall and per tag; grader parse failures "
                    "JUDGE_FAILED (scoring_excluded, counted, the replies kept in "
                    "judge_failed_items); every rubric verdict records judge_model and "
                    "grader_attempts; nothing scored -> failed with the score withheld; "
                    "leaderboard_comparable=False; model sampled as simple-evals samples it, with "
                    "gpqa.SIMPLE_EVALS_SAMPLING as in simpleqa and gpqa: chat sampler system "
                    "message \"You are a helpful assistant.\" + temperature 0.5 in both modes (the "
                    "grader sees that system turn, as grade_sample reads "
                    "actual_queried_message_list), max_tokens 2048 on every no-think request "
                    "(--max-output-tokens not applied) and the run's --max-output-tokens with "
                    "--thinking (official cap not applied to reasoning; simple-evals' o-series "
                    "samplers, keyed on OpenAI model names, are not reasoning settings for a "
                    "served model)")),
    },
    "causalbench": {
        "synced": "2026-10-10.3",
        "upstream": ("github.com/causalNLP/corr2cause code/run_model.py (+ "
                     "github.com/zhijing-jin/efficiency nlp.Chatbot @ 69d191bc)"),
        "revision": "21d80350e27a6206f536cf90a09e95d9e856c761",
        "dataset": "causal-nlp/corr2cause@42ba12c769e11ff6427c9f52d7db58e3f9bf3e53 (test, 1,162)",
        "method": ("GPT chat path without CoT: GPT.__init__ system prompt, prompt_tmpl_direct "
                   "(hypothesis rstrip '.'), temperature 0 and GPT.query_unseen_text's "
                   "max_tokens=10 for an 'Answer:' prompt, both in OfficialSampling; base's "
                   "token-limit policy sends 10 exactly without thinking (--max-output-tokens not "
                   "applied; replies at 10 are the protocol's outcome, "
                   "replies_at_official_max_tokens) and the run budget with thinking ('official "
                   "cap not applied to reasoning' recorded; replies over 10 tokens counted; "
                   "flagged); clean_pred_func + rel_normalize verbatim (nltk word_tokenize; "
                   "missing punkt -> infra_required); perf_df2report_dict F1/P/R of the minority "
                   "label (entailment) + Acc, also by num_variables and relation; headline = F1; "
                   "comparable for full no-think single-sample T=0 runs with every request "
                   "answered and every request sent max_tokens=10"
                   "; a null-content reply is asked again, the same request, as efficiency "
                   "Chatbot.ask retries raw_query's AttributeError on None (no limit), through "
                   "swebench_common.upstream_reask_relay with reply_content_is_null, up to "
                   "causalbench.NULL_REPLY_REASKS=5; an empty string is read as contradiction; "
                   "upstream_reask records it; a reply still null after the re-asks is read as "
                   "base reads it and makes the run non-comparable"
                   "; only a reply the model ended itself (finish_reason stop) is re-asked: a "
                   "reply cut off at the token budget is the model's own outcome, never re-asked, "
                   "scored as base scores a truncated reply and flagged (cut_off_not_reasked); the "
                   "re-asks stop before the client's per-request timeout "
                   "(EndpointRelay.reask_deadline), and a chain that ends on a reply still "
                   "matching the rule (reasks_exhausted, of them reasks_cut_short) passes that "
                   "reply on to be scored, never a harness error; upstream_reask also records the "
                   "classes of the empty replies seen (answer_in_thinking_seen, "
                   "tool_call_dropped_seen, unclassified_empty_seen, answer_in_thinking_rate_seen)"),
    },
    "i18n_translate": {
        "synced": "2026-10-09",
        "upstream": "sacrebleu (PyPI) BLEU + CHRF; WMT19 news translation task",
        "revision": "2.6.0",
        "dataset": ("WMT19 news test sets (newstest2019): sacrebleu test set wmt19 = "
                    "data.statmt.org/wmt19/translation-task/test.tgz, 19 directions, 31,387 "
                    "segments"),
        "checksum": ("sha256:5ef66d76e183ed40450086ad6fa7beccda1fbb290c5556b93273d515fd6683fb "
                     "(test.tgz; sacrebleu md5 84de7162d158e28403103b01aeefc39a)"),
        "method": ("gbench translation prompt (WMT defines none); reply scored as returned; per "
                   "direction sacreBLEU corpus BLEU (target default tokenizer: zh / 13a) and "
                   "corpus chrF with signatures; headline = mean BLEU over the 19 directions; "
                   "equal to sacrebleu.corpus_bleu / corpus_chrf; test.tgz fetched by the suite "
                   "(120 s timeout, a .part file renamed after the sha256 check, a cached tarball "
                   "with another sha256 fetched again, download, extraction and text preparation "
                   "under a file lock in the cache dir (portalocker), extraction marked complete "
                   "only when finished (a missing marker or a raw SGML file cut short "
                   "re-extracts), a text file cut short rebuilt): a download failure is "
                   "infra_required for this suite only, a checksum mismatch raises, sacrebleu's "
                   "sys.exit download path is never used; leaderboard_comparable=False (averaged "
                   "headline, gbench prompt)"),
    },
    "ifeval": {
        "synced": "2026-10-09",
        "upstream": ("github.com/google-research/google-research instruction_following_eval "
                     "(vendored in _ifeval_vendor)"),
        "revision": "e6890f85757dd84e27ca6df2dd30651dafad28e0",
        "dataset": ("google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84 + upstream key-2785 "
                    "prompt fix (26d8ccd)"),
        "method": (("upstream checkers + evaluation_lib strict/loose (langdetect required, "
                    "seeded); prompt-level strict headline, instruction-level and loose in "
                    "ifeval_report; --shard before --eval-limit; leaderboard_comparable=False for "
                    "--eval-limit, a sampled run, requests with no reply (harness errors, left "
                    "out of the headline and ifeval_report) and empty replies (failed_requests), "
                    "replies cut off at the max_tokens budget (truncated_responses), or verdicts "
                    "that depend on the whitespace trim")),
    },

    # Pillar 2: mathematics and proofs.
    "imo_answer_bench": {
        "synced": "2026-10-09",
        "upstream": "github.com/google-deepmind/superhuman (imobench) + arXiv 2511.01846 App. A.5",
        "revision": "955558ad67ef24dccd27545c326988bdad1b4471",
        "dataset": ("imobench/answerbench_v2.csv (IMO-AnswerBench v2; algebra-036 closing quote "
                    "repaired)"),
        "checksum": ("sha256:275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9 "
                     "(answerbench_v2.csv)"),
        "method": ("AnswerAutoGrader prompt (App. A.5) on the full response, last "
                   "\\boxed{Correct|Incorrect} decides (Gemini-cascade grader; its last reply "
                   "kept per trace as judge_raw with judge_calls and judge_model); a grader reply "
                   "without a boxed verdict asked again (3 calls in all) then JUDGE_FAILED, a "
                   "grading failure left out of the accuracy (scoring_excluded), counted in "
                   "judge_failures and named in leaderboard_comparable_reason; accuracy averaged "
                   "over >=8 runs; leaderboard_comparable=False; CSV sha256 verified at load; the "
                   "paper publishes no grader sampling settings, so gbench's deterministic judge "
                   "temperature 0 (judge_sampling)"),
    },
    "putnam": {"synced": "2026-10-09.2",
               "upstream": "github.com/trishullab/PutnamBench",
               "revision": "97b5e9458b04e3d1044e8f8afe1724ff133f5a53",
               "dataset": "trishullab/PutnamBench@97b5e945 informal/putnam.json (673 problems, "
                          "345 with a stated answer)",
               "checksum": "git-blob:77675840ebadc2bc9b5a93f390bb0a678f33dbbc "
                           "(informal/putnam.json)",
               "method": ("informal track, a gbench protocol (PutnamBench's leaderboard is formal "
                          "only): the 170 single closed-form golds graded on the answer "
                          "(normalized match with nested braces and a brace-balanced \\boxed; an "
                          "answer that does not match is decided by the IMO-AnswerBench "
                          "AnswerAutoGrader prompt, never by the string comparison alone), the "
                          "rest graded by the Gemini-cascade proof judge with the dataset "
                          "statement and the reference; the proof judge's verdict read from its "
                          "last Grade: CORRECT|INCORRECT line with markdown and LaTeX around it "
                          "removed (a line restating the format, Grade: CORRECT / INCORRECT, is "
                          "not a verdict) or from a reply that is only the verdict word; a judge "
                          "reply without a verdict asked again (3 calls) then JUDGE_FAILED, a "
                          "grading failure left out of the accuracy (scoring_excluded), counted "
                          "in judge_failures and named in leaderboard_comparable_reason "
                          "(completed_with_errors); judge model, reply and calls recorded per "
                          "trace, judge_models / judge_cascade per result; "
                          "leaderboard_comparable=False"
                          "; the judge verdict is read by putnam_formal's grade_line_verdict "
                          "(format echoes refused, HTML tags and LaTeX brace commands stripped); "
                          "replies cut off by the run's budget are counted "
                          "(replies_cut_off_by_budget / replies_cut_off_in_reasoning, "
                          "record_budget_cutoffs) and named")},
    "putnam_formal": {
        "synced": "2026-10-09.2",
        "upstream": "github.com/trishullab/PutnamBench",
        "revision": "97b5e9458b04e3d1044e8f8afe1724ff133f5a53",
        "dataset": ("trishullab/PutnamBench@97b5e945 lean4/src (672 Lean statements, 346 with "
                    "answer placeholders); Lean v4.33.1 + Mathlib v4.33.1 "
                    "(0df444a360eaa60ab8c11dca51a86af692955474) baked into "
                    "docker/putnam_formal.Dockerfile"),
        "checksum": "git-tree:f10ab3f77620e127ca58cae55924f6449da403ce (lean4/src)",
        "method": ("both leaderboard Lean settings: Lean with answer "
                   "(lean4/scripts/rewrite_solutions.py ported verbatim, byte-identical on all "
                   "672) and Lean no answer (answer comment removed; answer faithfulness in two "
                   "stages, as the leaderboard asks each no-answer result to state its review "
                   "procedure (docs/leaderboard.html@97b5e945): a kernel-checked tactic proof of "
                   "equality accepts and a proof of difference rejects (except on the 5 problems "
                   "noted to have several correct answers), otherwise the gbench Gemini judge "
                   "cascade reviews the elaborated answer (bound variables renamed, literals as "
                   "Lean writes them and Name literals as the applications they are built from, "
                   "printed in full by the checker's print mode in a container of its own with "
                   "Lean's printing limits lifted, sent whatever its length; an answer that "
                   "container does not print (past its time or 8 GB limit, or a printer error) is "
                   "not reviewed, not scored and named in leaderboard_comparable_reason) against "
                   "the reference answer; the print mode shows an answer at its value on one line "
                   "or not at all: string literals and Bind.bind continuations wrapped in empty "
                   "metadata (ProofWidgets HTML and do notation print as applications), Qq q(...) "
                   "notation and safe shadowing off, every atom of the delaborated term a token "
                   "or keyword, every identifier free of line-breaking characters and escape "
                   "brackets, string and character literals re-escaped (U+0080-U+009F, U+2028 and "
                   "U+2029 included), else a printer error; stage and judge model recorded per "
                   "trace); trusted kernel checker (Lean.Environment.replay of Lean v4.33.1 on a "
                   "clean Mathlib, theorem type compared with local definitions unfolded, dataset "
                   "helpers identical, standard axioms only; single nonce-tagged verdict line "
                   "with escaped names, ok only with exit code 0); headline = Lean with answer "
                   "(total_questions / correct_answers / effective_n describe it; pooled counts "
                   "in *_all_settings; a setting with no scored attempt has accuracy null), "
                   "solved counts per setting in putnambench_report; the reply's last ```lean "
                   "block found with fences paired line by line (an abandoned unclosed block does "
                   "not swallow the final one); an answer review whose judge gives no Grade: "
                   "verdict line in 3 calls (markdown and LaTeX around the line removed before it "
                   "is read) is JUDGE_FAILED, left out of the accuracy (scoring_excluded) and "
                   "counted in judge_failures; verification harness errors left out of the "
                   "accuracy and counted as failed requests; replies cut off by the run's token "
                   "budget (PutnamBench publishes no generation harness, so there is no official "
                   "cap and the house budget applies), empty answers after reasoning included, "
                   "counted in replies_cut_off_by_budget and replies_cut_off_in_reasoning and "
                   "make the run not leaderboard_comparable; the image's toolchain, Mathlib and "
                   "commit and a live Gemini key are checked at startup"
                   "; lean_blocks closes a block on a glued closing fence and lets a later inline"
                   " block win, so the kernel checks the final proof instead of an abandoned "
                   "lemma; grade_line_verdict refuses format echoes ('CORRECT or INCORRECT', "
                   "'CORRECT | INCORRECT') and strips HTML tags and LaTeX brace commands; "
                   "putnambench_report solved / solved_pct count only scored attempts "
                   "(problems_unscored, named in the reason)"),
    },

    # Pillar 3: coding and algorithmic design.
    "aider_polyglot": {
        "synced": "2026-10-14",
        "upstream": ("github.com/Aider-AI/aider benchmark/benchmark.py + "
                     "github.com/Aider-AI/polyglot-benchmark"),
        "revision": ("5dc9490bb35f9729ef2c95d00a19ccd30c26339c (aider, the image's "
                     "org.gbench.aider.commit label), 7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f "
                     "(polyglot-benchmark, the exercises checkout's HEAD)"),
        "dataset": ("github.com/Aider-AI/polyglot-benchmark@"
                    "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f (225 exercises: cpp 26, go 39, "
                    "java 47, javascript 49, python 34, rust 30; checkout HEAD and unmodified "
                    "tracked files checked, else infra_required)"),
        "method": (("delegated to aider benchmark.py --tries 2 in the aider-benchmark:5dc9490b container "
                    "(native edit format, pass@2 headline over 225 exercises); --thinking sent as "
                    "extra_body.chat_template_kwargs.enable_thinking via an aider/extra_params "
                    "--read-model-settings file, confirmed loaded by an in-container pre-flight "
                    "(aider's own register_models + Model) before benchmark.py (infra_required); "
                    "aider sends no max_tokens in either mode (aider/models.py "
                    "Model.send_completion adds one only from a model's settings, none for an "
                    "openai/<id> model; read from the pre-flight's extra_params), recorded in "
                    "max_output_tokens_note through base.resolve_token_budget; aider's 600 s "
                    "request timeout replaced, through the same extra_params (timeout, with max_retries 0), by "
                    "base.request_timeout_s of the served window (UNCAPPED_TIMEOUT_TOKENS when "
                    "unknown), so a temperature-0 reply longer than 600 s is no longer retried "
                    "for 24 h; a request that times out anyway is an endpoint failure "
                    "(model_endpoint harness error), recorded in model_request_timeout_s; aider's own "
                    "num_exhausted_context_windows (aider/coders/base_coder.py send_message: "
                    "finish_reason length for a model without assistant prefill, as an "
                    "openai/<id> model is, or litellm ContextWindowExceededError; written by "
                    "benchmark.py into each .aider.results.json and summed by summarize_results "
                    "into the exhausted_context_windows column of polyglot_leaderboard.yml) is "
                    "summed into exhausted_context_windows and replies_cut_off_by_budget, and any "
                    "makes the run non-comparable; aider's other leaderboard columns are reported "
                    "(aider_leaderboard_columns, computed as summarize_results computes them over "
                    "the scored exercises); comparable only for the full 225-exercise set, all 6 "
                    "languages, aider's native edit format and no exhausted context window; the "
                    "container runs on docker's bridge network with --add-host "
                    "host.docker.internal:host-gateway, as upstream's benchmark/docker.sh does, "
                    "after the metadata-server guard; aider is always pointed at gbench's relay on "
                    "the bridge gateway (OPENAI_API_BASE becomes the relay URL, since 2026-10-11), "
                    "so a 127.0.0.1-bound endpoint works; the metadata-server probe "
                    "container also resolves the endpoint's host when it is a name and the "
                    "toolchain hosts of the run's languages (Java: services.gradle.org, "
                    "repo.maven.apache.org; Rust: index.crates.io, static.crates.io; upstream's "
                    "image downloads them at test time), infra_required before the run otherwise; "
                    "the container keeps the image's HOME (/root, where rustup installed Rust "
                    "with no RUSTUP_HOME/CARGO_HOME), as upstream's benchmark/docker.sh does; the "
                    "image is upstream's benchmark/Dockerfile plus the setuptools-scm pin and "
                    "jest@29.7.0 (every JavaScript exercise's package.json asks for ^29.7.0; with "
                    "Jest 30 affine-cipher and resistor-color-trio fail whatever the model "
                    "writes), and before benchmark.py the container checks jest 29.x and cargo "
                    "--version (infra_required otherwise); an exercise is a harness error when a "
                    "test run's output (the bare fenced block benchmark.py appends after aider's "
                    "own lines; model replies and instructions are not read) shows a toolchain "
                    "that could not run (toolchain: rustup with no default toolchain) or a failed "
                    "lookup or connection for a toolchain host (toolchain_network), or when aider "
                    "gave up on a request with no reply after its last retry (Retrying in 32.0 "
                    "seconds...) and the logged failure is an endpoint failure (litellm "
                    "APIConnectionError/InternalServerError/BadGatewayError/ServiceUnavailableError/RateLimitError "
                    "or a failed lookup or connection: model_endpoint): left out of every figure, "
                    "counted in harness_errors / harness_error_kinds / harness_error_exercises, "
                    "status completed_with_errors, not comparable; every exercise one: "
                    "infra_required; the full-run check counts them as finished; a request aider "
                    "gave up on for another cause (litellm.Timeout: a non-streamed reply past "
                    "aider's 600 s request timeout) is the model's outcome, scored as upstream "
                    "and counted per request in scored_request_give_ups / "
                    "scored_request_give_up_exercises; GBENCH_AIDER_LANGUAGES is lowercased as "
                    "benchmark.py does; the container has upstream's memory limit "
                    "(benchmark/docker.sh --memory=12g --memory-swap=12g, recorded as "
                    "container_memory_limit with the benchmark.py --threads that share it), so "
                    "model code that allocates without bound is killed inside the container (a "
                    "failed test run, the model's outcome as upstream scores it) instead of "
                    "taking this host's memory; a container that exits 137 (SIGKILL) is named in "
                    "the hard error; after benchmark.py, whatever its exit status (kept as the "
                    "container's), the container prints its cgroup's OOM-kill count (cgroup v2 "
                    "memory.events oom_kill, v1 memory.oom_control) as its last line, recorded as "
                    "container_oom_kills; the limit is shared by the --threads exercises "
                    "benchmark.py runs at once (gbench's eval concurrency; upstream's "
                    "benchmark/README.md runs 10), so a kill cannot be attributed to one "
                    "exercise: any kill, or a count that could not be read (null), makes the run "
                    "not comparable, with the count and thread count in "
                    "leaderboard_comparable_reason, and the 137 / partial-run hard errors name "
                    "the kills; the score is kept as upstream computes it"
                    "; exercise selection on the host for --shard / --eval-limit (sorted "
                    "<lang>/exercises/practice dirs, base.apply_shard_indexed, then "
                    "sampling.stratified_sample by language seed 'aider_polyglot'; a subset "
                    "copied to a read-only tree passed as --exercises-dir; benchmark.py "
                    "--num-tests never passed); per-exercise sample_traces and merge_shards; the "
                    "served id gbench passes is sent as openai/<id> (no /v1/models read); an "
                    "endpoint key reaches the endpoint through gbench's EndpointRelay on the "
                    "bridge gateway; temperature 0.0 no-think / 1.0 thinking (or the operator's) "
                    "sent in the same aider/extra_params (temperature_applied true), --attempt-count recorded as not applied; the image's "
                    "org.gbench.aider.commit label must equal AIDER_COMMIT and the exercises "
                    "checkout's HEAD must equal POLYGLOT_BENCHMARK_COMMIT, else infra_required")),
    },
    "bigcodebench": {
        "synced": "2026-10-09",
        "upstream": "github.com/bigcode-project/bigcodebench",
        "revision": "09dd993f46c3fbf3a799465bb96d524edcb0b199",
        "dataset": ("bigcode/bigcodebench@b74c0d0bf70d2c0bc459be537895cca163007f1a (v0.1.4); hard "
                    "ids: bigcode/bigcodebench-hard@298d2cc7b96612e15e47313c3603ee124cee0c1f"),
        "method": ("both leaderboard splits per task (Instruct headline, Complete) + "
                   "BigCodeBench-Hard figures from the same run (identical rows); upstream "
                   "make_raw_chat_prompt; no-think = default --backend vllm path, response_prefix "
                   "prefill continued via continue_final_message (every Gemma leaderboard row has "
                   "prefill=true; first-char newline checked), tabs -> 4 spaces; thinking = "
                   "--backend openai path (no prefill); recorded as prefill / generation_path; "
                   "no-think: run_codegen greedy temperature 0.0 with the vllm backend's top_p 1.0"
                   " + EOS stops and max_new_tokens 1280 (--max-output-tokens not applied; less "
                   "only where the prompt leaves less room in the served window); thinking: "
                   "gen/util/openai_request.py make_request as it runs for a served model, "
                   "run_codegen's temperature 0.0 with top_p 0.95 (its reasoning-model branch "
                   "keys on the vendor model names o1-, o3-, reasoner and grok-3-mini-beta and is "
                   "not used, so no reasoning_effort is sent); the 1280 max_completion_tokens was "
                   "written for non-reasoning generation and is not applied: each request gets "
                   "the run's --max-output-tokens, else the thinking floor (official_sampling "
                   "recorded; token-limit policy, max_output_tokens_note); bigcodebench.sanitize "
                   "without --calibrate + calibrated unittest execution per split in the LOCAL "
                   "pinned harness image (evaluate container bounded to 64 CPUs, since the harness"
                   " TensorFlow tasks hang when it sees ~128+; per-split work dir removed, HF "
                   "cache kept in the container); calibrated pass@1; comparable only for a full "
                   "run with the path's settings (no-think: greedy, top_p 1.0, 1280; thinking: "
                   "temperature 0.0, top_p 0.95, the run's budget), no other sampling field "
                   "(decoding penalties named; any reasoning_effort makes the run "
                   "non-comparable) and no reply stopped at a limit other than 1280 (with "
                   "thinking, the run's budget; replies_cut_off_by_budget, "
                   "replies_cut_off_in_reasoning), every task scored, "
                   "no harness error (infra_required); sanitize and evaluate containers on "
                   "docker's default bridge (the network upstream's documented docker run gives "
                   "it; both load the dataset from Hugging Face), with the metadata-server guard "
                   "checked before generation; the metadata-server probe also resolves "
                   "huggingface.co before generation (HARNESS_HOSTS); a split whose harness "
                   "produced no results is a harness error: its samples scoring_excluded (status "
                   "HARNESS_ERROR), bigcodebench_harness_errors {split: kind "
                   "dns|hub_unreachable|no_results} (hub_unreachable: datasets' \"Couldn't reach "
                   "'<repo>' on the Hub (...)\" or huggingface_hub LocalEntryNotFoundError, which "
                   "is all the image prints when the Hub cannot be reached, since datasets raises "
                   "it from None), the split left out of splits, run status completed_with_errors "
                   "(error when no split was scored), accuracy null with accuracy_withheld_reason "
                   "when that split is Instruct; a task the harness returned no result for while "
                   "it scored others is a harness error too: left out of accuracy, splits and "
                   "category_accuracy (scoring_excluded, harness_error kind no_harness_result, "
                   "bigcodebench_unscored, status completed_with_errors, non-comparable), and the "
                   "harness's own pass@1 of its split, which leaves that task out, is withheld "
                   "(bigcodebench_pass_at_1_withheld); on the no-think path, text the server's "
                   "reasoning parser moved into the reasoning field is recovered by base's "
                   "server-parser policy (the reply's token ids decoded with special tokens kept, "
                   "else the reasoning field followed by the content) and is what is sanitized and"
                   " scored, as upstream sanitizes the whole generation of its in-process vLLM "
                   "(provider/vllm.py, skip_special_tokens=False); such replies are counted in "
                   "replies_split_by_reasoning_parser, and one rebuilt as reasoning then content "
                   "(the server returned no token ids) makes the run non-comparable; a task whose "
                   "request got no reply from the served endpoint is a harness error: left out of "
                   "accuracy, splits and category_accuracy (request_harness_errors, status "
                   "completed_with_errors, non-comparable); the harness's own pass@1 of its split,"
                   " which counts that task as failed, is withheld (null, "
                   "bigcodebench_pass_at_1_withheld); the headline is withheld (accuracy null with"
                   " accuracy_withheld_reason) when every Instruct request got no reply"),
    },
    "lcb": {
        "synced": "2026-10-09",
        "upstream": "github.com/LiveCodeBench/LiveCodeBench (lcb_runner)",
        "revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
        "dataset": ("livecodebench/code_generation_lite @ 0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
                    " (GBENCH_LCB_VERSION, default release_v6) + livecodebench/execution-v2 @ "
                    "ff6ea0e2a638001006ddcc41259eff23a4283fb2 + livecodebench/test_generation @ "
                    "6f3ac40bbecf81eba15899139d279b077f2816fd (default on)"),
        "method": ("lcb_runner OpenAIChat prompts verbatim, last-block extract_code; grading split"
                   " so the process running model code never holds the expected outputs (child "
                   "runs upstream testing_util / utils_execute helpers, non-dumpable grader "
                   "applies upstream comparisons, verdict equal to run_test on honest programs, "
                   "nonce-tagged); code execution via one grader-answered comparison; 4 scenarios "
                   "by default, reported separately: code generation (headline), code execution, "
                   "test output prediction, self-repair (run_main_repair: rejected programs get "
                   "prompts/self_repair.py OpenAIChat prompt verbatim from the grader's run_test "
                   "failure metadata, -4-without-test-fields/-5 given the messages of the 8eac093f"
                   " grader, one generation each, all programs regraded); >=10 samples/problem "
                   "always (lcb_runner --n); every request, self-repair included, sends what "
                   "runner/oai_runner.py sends with parser.py defaults: temperature 0.2, top_p "
                   "0.95, frequency_penalty 0, presence_penalty 0 (official_sampling recorded); "
                   "max_tokens by the token-limit policy from OFFICIAL_SAMPLING.max_tokens "
                   "(parser.py --max_tokens 2000), self-repair included: 2000 without --thinking "
                   "(--max-output-tokens not applied; less only where the prompt leaves less room "
                   "in the served window), the run's budget with --thinking (official cap not "
                   "applied to reasoning, official_sampling.max_tokens_applied false; lcb_runner "
                   "defines no reasoning-model max_tokens for the OpenAIChat style: OpenAIReason, "
                   "OpenAIReasonPreview, Claude3Thinking and GeminiThinking are vendor-API styles "
                   "with their own prompts, and open reasoning models take --max_tokens on the "
                   "command line); max_output_tokens_note states the mode's choice; comparable "
                   "only at that sampling with no other sampling field (decoding penalties named),"
                   " all four scenarios, a full post-v1 window, every generation and self-repair "
                   "request answered and no reply, generation or self-repair, stopped at a limit "
                   "other than lcb_runner's 2000 (with --thinking the run's budget, without it a "
                   "max_tokens the context window lowered; replies_cut_off_by_budget, and "
                   "replies_cut_off_in_reasoning for those with no answer text); every scenario's "
                   "accuracy (self-repair included), each category_accuracy entry and code "
                   "generation's avg@k are lcb_runner's pass@1 estimate "
                   "(evaluation/pass_k_utils.py compute_metrics_from_results, k=1): the mean over "
                   "problems of c_i/n_i over each problem's scored generations, not the pooled "
                   "correct/total, which differs from it once a harness error leaves problems with"
                   " different numbers of scored generations; generation and self-repair requests "
                   "with no reply are harness errors, left out of their scenario "
                   "(self_repair.harness_errors, effective_n) and named in status_reason"
                   "; the tests are not stored in the traces (a per-problem digest instead); "
                   "merge_shards recomputes every LiveCodeBench field from the stored verdicts"),
    },
    "scicode": {
        "synced": "2026-10-09",
        "upstream": "github.com/scicode-bench/SciCode (protocol ported from e3158ea011d4)",
        "revision": "4510f6a6aa27c43fad7b43da2c59602a86e88480",
        "dataset": ("SciCode1/SciCode@4510f6a6aa27c43fad7b43da2c59602a86e88480 (test) + "
                    "test_data.h5 from "
                    "Srimadh/Scicode-test-data-h5@72c247d3a8410921b2e848e046d71ed63d9a0ddb"),
        "checksum": ("sha256:48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890 "
                     "(test_data.h5)"),
        "method": ("sequential per-sub-step gencode protocol; execution scoring over upstream's "
                   "65-problem/288-step test split; harness errors are left out and named: a "
                   "sub-step whose generation got no reply after the retries, the sub-steps "
                   "after it (not generated) and every sub-step of a crashed problem leave the "
                   "sub-step accuracy, and their problem leaves the headline "
                   "(harness_error_problems, effective_n; accuracy null with "
                   "accuracy_withheld_reason when no problem is scored); the test cases' "
                   "scicode.compare.cmp (imported by 90 test cases over 31 scored sub-steps of "
                   "20 problems) is vendored verbatim from SciCode@e3158ea0 "
                   "(src/scicode/__init__.py, compare/__init__.py and compare/cmp.py; cmp.py "
                   "sha256 8b3e790cc29dc87ef1f0f1b9bfd68c877728c78c40b541a69d10015221e9e235) "
                   "and put on every scored sub-step's PYTHONPATH, its import checked in the "
                   "sandbox before generation (infra_required) and recorded as "
                   "scicode_compare_module; every module the selected problems' "
                   "required_dependencies import (numpy, scipy, sympy, mpl_toolkits.mplot3d, "
                   "...) checked importable in gbench's interpreter, the one the scored code "
                   "runs with, before generation (infra_required naming the package), never a "
                   "failed sub-step; headline = main-problem resolve rate, sub-step accuracy "
                   "secondary; test_data.h5 pinned and sha256-checked (mismatch on the default "
                   "mirror hard-errors, a user file with another digest is non-comparable) "
                   "(infra_required); SciCode gencode.py --temperature default 0 as the suite "
                   "default with and without thinking, and no max_tokens in either mode "
                   "(src/scicode/gen/models.py generate_litellm_response sends none for "
                   "litellm/* models, the vLLM route; recorded through "
                   "base.resolve_token_budget; --max-output-tokens not applied, "
                   "max_output_tokens_note; HTTP timeout sized from the served window, else "
                   "base.UNCAPPED_TIMEOUT_TOKENS) (official_sampling recorded); a sub-step "
                   "reply that reaches the end of the served window (no max_tokens is sent) is "
                   "counted (replies_cut_off_by_budget, replies_cut_off_in_reasoning) and "
                   "non-comparable; problems ordered by problem_id, --shard applied first "
                   "(base.apply_shard_indexed), then the seeded --eval-limit draw; sub-step "
                   "traces carry sample_key and global_idx, merge_shards re-counts the resolve "
                   "rate and sub-step accuracy; an empty shard withholds its score; the "
                   "endpoint key (VLLM_API_KEY, else OPENAI_API_KEY) is sent by gbench's own "
                   "client only"
                   "; each sub-step's test script is built as upstream's test_generated_code.py "
                   "builds it (the assembled code, then from scicode.parse.parse import "
                   "process_hdf5_to_tuple with the loader vendored verbatim as "
                   "_scicode_vendor/scicode/parse/parse.py, the targets, the tests) and run from "
                   "a file, so a program over 128 KiB runs; a script that could not be run at all"
                   " is a harness error (scoring_failed_steps); a context-window rejection is not"
                   " retried: a sub-step prompt that carries the model's own earlier code fails "
                   "that step and the steps after it, the problem staying in the score, as "
                   "gencode.py (no handling) and the fixed denominators make it "
                   "(over_context_steps), and a prompt built from the dataset alone that the "
                   "window cannot hold is a harness error (over_context); replies that reached "
                   "the window repeating themselves are counted (replies_cut_off_looping) and get"
                   " no advice to serve a larger window"
                   "; the step script's directory is bound into the sandbox (the jail empties "
                   "/tmp, where the script is written) and a sh launcher checks that the "
                   "interpreter, the script and test_data.h5 are there inside the sandbox, prints"
                   " a start line, then execs python <file>: a run without that line (also one "
                   "that timed out before it) never reached the step's code and is a harness "
                   "error (scoring_failed_steps), never a failed step; before generation a "
                   "trivial sub-step that only opens test_data.h5 is run that way in the sandbox "
                   "(check_step_sandbox), and a failure there is infra_required"),
    },
    "ojbench": {
        "synced": "2026-10-09",
        "upstream": ("github.com/He-Ren/OJBench (judge built from 5e94480b in "
                     "docker/ojbench.Dockerfile)"),
        "revision": "61cf9986f22c25d08e1657b03742124099c74353",
        "dataset": ("He-Ren/OJBench_testdata@61cf9986f22c25d08e1657b03742124099c74353 "
                    "(prompts/full.jsonl)"),
        "method": ("8 candidates/problem (paper protocol), DMOJ online judge per generation; "
                   "per-language Pass@1/Pass@8 + difficulty pass rates; headline = C++ Pass@1 "
                   "(leaderboard ranking column); max_tokens per the paper by the token-limit "
                   "policy: 64k (65536) with thinking, the paper's reasoning-model setting, never "
                   "below --max-output-tokens (a request clamped below it is non-comparable); "
                   "none without it, the model's default context length (--max-output-tokens not "
                   "applied) (infra_required); sampling = the model's officially recommended "
                   "parameters (paper Section 3.1), read from the served model's "
                   "generation_config.json (--tokenizer or served model name; unreadable -> "
                   "infra_required), 1.0 / 0.95 / 64 for Gemma 4 (official_sampling and "
                   "recommended_sampling recorded); any other sampling field, decoding penalties "
                   "included, is non-comparable; a reply that stopped at another limit (without "
                   "thinking the end of the served window, with it a --max-output-tokens above "
                   "64k or a max_tokens the window lowered) is counted "
                   "(replies_cut_off_by_budget, replies_cut_off_in_reasoning) and non-comparable; "
                   "a generation whose request got no reply from the served endpoint is a harness "
                   "error: left out of every figure (request_harness_errors, named in "
                   "status_reason, status completed_with_errors, non-comparable), named apart "
                   "from a generation the judge returned no verdict for (judge_no_verdict, also "
                   "left out and non-comparable); when C++ generations were run and none was "
                   "scored, the headline is withheld (accuracy null with "
                   "accuracy_withheld_reason), never replaced by the pooled or Python figure; the "
                   "judge container runs with --network none"),
    },
    "copilot_bench_swe": {
        "synced": "2026-10-10.3",
        "upstream": ("SWE-agent/mini-swe-agent 2.4.2 benchmarks/swebench.yaml (sha256 229f178a, "
                     "bundled, unchanged) in the local image gbench-mini-swe-agent-v2:2.4.2-2.4.6 + swebench "
                     "5.0.2 harness in the local image gbench-swebench-v5:5.0.2 (the setup of "
                     "SWE-bench/experiments "
                     "evaluation/verified/20260901_mini-v2.4.2_gemini-3-5-flash)"),
        "revision": "mini-swe-agent 2.4.2, swebench 5.0.2",
        "dataset": ("SWE-bench/SWE-bench_Verified@78f471bf655a3137b2e8a75af1501690ec009ec3 (test, "
                    "500; v5 columns image/eval_script/log_parser/eval_type; one local JSONL "
                    "snapshot feeds the rollout, the one-shot prompts and the harness "
                    "--dataset_name)"),
        "checksum": ("sha256:030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25 "
                     "(data/test-00000-of-00001.parquet at the revision)"),
        "method": (("agentic default: mini-swe-agent 2.4.2 tool-calling rollout (instance image "
                    "pulled first with the host's Docker CLI from the gbench process, "
                    "swebench_common.host_pull_images, so the operator's docker login applies and "
                    "the rollout container gets no Docker client configuration; a failed pull is "
                    "an infrastructure failure; turns cut off at the end of the context window "
                    "(truncated_turns) are counted and make the run not comparable; turns with "
                    "reasoning but no content or action (empty_after_reasoning_turns) are split by "
                    "their token ids (swebench_common.classify_empty_reply): the answer inside the "
                    "thinking channel is the model's outcome, counted in answer_in_thinking, not a "
                    "comparability reason; a native tool call the server's tool-call parser "
                    "removed (tool_call_dropped_turns) and a turn without token ids to tell the "
                    "two apart (unclassified_empty_turns) are not the model's answer and make the "
                    "run not comparable; "
                    "the one-shot harness pass pulls each row's image the same way; driver "
                    "follows run.benchmarks.swebench.process_instance; each rollout container "
                    "runs on docker's default bridge network with OPENAI_API_KEY=EMPTY and "
                    "reaches the served endpoint only through gbench's EndpointRelay on that "
                    "network's gateway (forwards only /v1/chat/completions, /v1/completions and "
                    "/v1/models and adds the endpoint key; checked reachable from that network by "
                    "a probe container of the image before the first rollout, infra_required "
                    "otherwise; an endpoint failure it saw is a harness error: endpoint_failures, "
                    "completed_with_errors, not comparable), and mounts the docker socket because "
                    "upstream's DockerEnvironment starts the instance container with docker run "
                    "and runs each of the model's commands there with docker exec; no temperature "
                    "sent unless set; no per-turn max_tokens, as the bundled config sets none, "
                    "--max-output-tokens recorded as not applied in max_output_tokens_note) in "
                    "swebench/sweb.eval.x86_64 images, submission scored verbatim by swebench "
                    "5.0.2 run_evaluation (TestSpecs of all 500 rows built with 5.0.2; the image "
                    "carries docker-ce-cli 5:29.8.1-1~ubuntu.24.04~noble, which its container "
                    "cleanup uses, and its Ubuntu packages from the archive snapshot "
                    "20260925T120000Z on ubuntu:24.04 by digest, and swebench's dependencies "
                    "pinned by gbench/docker/swebench_v5.constraints.txt); v5 "
                    "infra_failure_ids/ambiguous_failure_ids/failure_reasons recorded as "
                    "upstream's advisory classification (harness/infra_failure.py), no effect on "
                    "score, status or comparability; a crashed final report step is re-run alone "
                    "(swebench's make_run_report on the scoring pass's per-instance report.json "
                    "files, at most 5 times after 0/5/15/30/60 s), so an instance the scoring "
                    "pass left without report.json (e.g. a test timeout) stays in error_ids; an "
                    "instance whose rollout could not run (agent_infra: image pull, container "
                    "start, endpoint), an errored instance whose run_instance.log shows neither a "
                    "patch-apply failure nor a test timeout (harness_no_verdict) and every "
                    "submitted patch of a harness pass with no report (harness_no_report) are "
                    "harness errors: left out of the resolved rate, numerator and denominator "
                    "(scored_instances, effective_n), counted in harness_errors, "
                    "completed_with_errors, and the rate withheld (accuracy null, status error) "
                    "when nothing was scored, while an errored instance whose log (the records "
                    "swebench itself logs) shows the patch did not apply, or the tests over the "
                    "applied patch running past the harness's timeout (test_timed_out, "
                    "test_timeout_instances), stays unresolved, the model's outcome, as upstream "
                    "counts every error_ids instance; the snapshot is checked (500 rows, v5 "
                    "columns, content sha256 "
                    "1939f0b7e4fca575369dad9bca3a843c48d745750ccc866999728c54d3dafd04 per "
                    "swebench_common.snapshot_sha256) before it is written and before every "
                    "reuse; comparable only on config 229f178a, i.e. comparable to the "
                    "SWE-bench/experiments submission 20260901_mini-v2.4.2_gemini-3-5-flash "
                    "(merge 69f10990; the displayed board swe-bench.github.io@193160a4 has no "
                    "2.4.x row); GBENCH_COPILOT_BENCH_SWE_MODE=oneshot opt-in is never comparable")),
    },
    "swe_bench_live": {
        "synced": "2026-10-11",
        "upstream": ("the SWE-bench-Live leaderboard "
                     "SWE-bench-Live/swe-bench-live.github.io@90e8720259d7 (ten tabs, "
                     "leaderboard.js validSets; resolved / report.total) + "
                     "github.com/SWE-bench-Live/SWE-bench-Live fork (Lite, Full, Verified) + "
                     "microsoft/SWE-bench-Live@62d02936a0726fbf4597b05bd07b09fb99eddae6 "
                     "evaluation/evaluation.py with "
                     "microsoft/RepoLaunch@c4b623d930f3728e5338664bb634021b98492cbf (C/C++, C#, "
                     "Java, TS/JS, Go, Rust, Windows) + "
                     "njukenanli/SWE-agent-for-eval@824224225e1e952b48da91f2d1bde38adb2115f2 "
                     "config/default.yaml (the SWE-agent evaluation/README.md names; Python and "
                     "MultiLang tabs) + "
                     "njukenanli/Win-Agent@47d85cffd5342a513fa3fb72ec4b5347ced9bb36 (Windows)"),
        "revision": ("a145aa87c62361532dacaa243398978164b234b7, "
                     "62d02936a0726fbf4597b05bd07b09fb99eddae6, "
                     "c4b623d930f3728e5338664bb634021b98492cbf, "
                     "47d85cffd5342a513fa3fb72ec4b5347ced9bb36, "
                     "824224225e1e952b48da91f2d1bde38adb2115f2 (the SWE-bench-Live fork harness; "
                     "microsoft/SWE-bench-Live main; its RepoLaunch submodule; Win-Agent; "
                     "SWE-agent-for-eval)"),
        "dataset": ("SWE-bench-Live/SWE-bench-Live@b51a86422e10cfd403beb4773e5a2947953e36ec "
                    "(lite, 300, content sha256 "
                    "b9252aeb085eb949f01b8464bee76240d474146c9a4f6aed1bbb693c2b94755e; "
                    "verified, 500 rows / 499 ids, "
                    "d3ae023598f62e0c24a025a5091ec6ef97224b62dd8aeb6224d7b14154e55974) and "
                    "@3da151be953883d3cd4e1ea0dc7a042ff2abf8f4 (full, 1319, "
                    "b231d1932ce614df07b6b4237db2a6f4eb6020213d0d9e919d95a62b976cf95e: the May "
                    "2025 full split every leaderboard Full row is scored on; 12b98891 "
                    "(2025-07-02) grew it to 1,565 and it holds 1,888 rows at b51a8642, which "
                    "no leaderboard row scores); "
                    "SWE-bench-Live/MultiLang@3638632e8153a10ca422c1022bed79023084b5c9 (c 111 "
                    "713f362bef00486539bf457dd08b74713066d764a07fbbea9c12d1936461a0a8, cpp 142 "
                    "6f0c50cca9be037126623f7f8af9d10ffc55f405dc51fdfe14d35b635e57eae5, cs 136 "
                    "4086e1640f6d69ce7a2ef9c7b14d39b510c77cfddc4cae86efef5934177a3236, go 138 "
                    "261851b6adc60e90747bd07e0b48ff5ad6f3ba7a5f8808c4e9b780a9c9fe8528, java 160 "
                    "aa6dce064c88cf51109da87a6e993b8d05b786ef2e1a9da3791e1e8054289480, js 108 "
                    "9cf48e9b78e56670ea210778bad59341fe2fda00e3d2d4e7937f7e35e4c4ef8e, rust 171 "
                    "b60c2965b1def9aa90d1e8b3a22d09d8544336108202a972798e2aa804150917, ts 111 "
                    "a3944c6d25f2d7112910e6d9a76cd5fc7f3dcadb2c5387a33d4db4c20826124f; tabs "
                    "ccpp = c + cpp, csharp = cs, tsjs = js + ts) and "
                    "SWE-bench-Live/Windows@b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9 (test, 66, "
                    "1f2a2e4057b1408b8932b1474cf8c9f0ce94b1862c25c4fdb53525b23fc432c5); the "
                    "board rows of the growing multi-language and Windows tabs keep the totals "
                    "of earlier snapshots (recorded per setting in leaderboard_totals; today "
                    "only Java 160 and Go 138 have rows at the pinned size); one local JSONL "
                    "snapshot per split, checked for row count, the columns its harness and "
                    "rollout read (MultiLang and Windows now include base_commit, which "
                    "SWE-agent-for-eval's loader requires) and content sha256 "
                    "(swebench_common.snapshot_sha256) before it is written and before every "
                    "reuse (the prerequisite check fetches and checks every selected setting "
                    "and compiles every MultiLang log_parser in the image); verified repeats "
                    "conan-io__conan-18153 as published (the harness reads every row, the "
                    "rollout and one-shot prompts run once per instance id, recorded in "
                    "dataset_repeated_instance_ids)"),
        "method": (("every leaderboard tab runs by default, each on its own (--eval-categories "
                    "picks a subset; the selected tabs run at the same time in lanes, the "
                    "Python tabs Lite/Full/Verified one after the other in one lane since their "
                    "instance ids overlap and the fork harness names task containers by "
                    "instance id and one run id, each other tab in a lane of its own, the "
                    "--sandboxes budget dealt over the lanes so the tabs running at once never "
                    "use more, each tab's rollouts, relay accounting, watchdog, harness pass and "
                    "headline exactly as alone, recorded in tab_schedule); every selected setting whose prerequisites are present "
                    "runs, and each selected setting whose prerequisite is missing is reported "
                    "in settings.<name> with status infra_required and the missing prerequisite "
                    "(never scored; listed in settings_infra_required; suite status "
                    "completed_with_errors), in both directions: the Windows tab off a Windows "
                    "Docker host, the Python and MultiLang tabs without the Linux harness or "
                    "SWE-agent-for-eval image (e.g. on a Windows-containers host), and a "
                    "setting whose pinned snapshot cannot be fetched; the suite hard-errors "
                    "when no selected setting can run (swe_bench_live.missing_prerequisites) "
                    "and when the Linux tabs' cloud metadata server gate fails "
                    "(swebench_common.require_task_network_blocked on docker's default bridge "
                    "and a user-defined bridge network, before the first Linux tab and again "
                    "before each SWE-agent batch and each MultiLang harness pass, cached; "
                    "infra_required for the whole suite); the Python and MultiLang tabs run "
                    "SWE-agent-for-eval (the SWE-agent fork microsoft/SWE-bench-Live "
                    "evaluation/README.md names) from the local image "
                    "gbench-sweagent-for-eval:82422422-rex3 "
                    "(ubuntu:24.04 by digest, Ubuntu archive snapshot 20260925T120000Z, Python "
                    "3.12 venv with pip install --editable . and the vendored swerex 1.4.0 "
                    "copied in as its README says, gbench's two patches (the task runtime's "
                    "heredoc fix, sweagent_for_eval_swerex_heredoc.patch, and the tolerant read "
                    "of the model's edited files at submit, sweagent_for_eval_review_decode.patch: "
                    "errors=replace, and a path that is not a regular file, such as a git "
                    "submodule's directory, is not read and its record entry is empty, the "
                    "patch untouched, "
                    "label org.gbench.sweagent_for_eval_agents), dependencies pinned by "
                    "sweagent_for_eval.constraints.txt = the PyPI set of 2026-06-03, then "
                    "gbench's fix of the task containers' runtime applied to that swerex "
                    "(gbench/docker/sweagent_for_eval_swerex_heredoc.patch: SWE-ReX's "
                    "per-instance runtime layer installs swe-rex 1.4.0 and its dependencies at "
                    "pinned versions, swe_bench_live.SWEREX_RUNTIME_PINS, instead of the latest "
                    "swe-rex, with a patched swerex/runtime/local.py, sha256 "
                    "d67d68966754cc73a3c2ae084fdfe90e7fb274e3ae36da6e619e1c4155372ed4: upstream's "
                    "BashSession split every command with bashlex 0.18 and joined the pieces "
                    "with ' ; ', so a heredoc whose operator is not the last word of its line "
                    "(cat <<EOF > file) or that another command follows lost its body or "
                    "terminator and the shell waited until the 90-minute command limit; a "
                    "command holding a heredoc now runs whole, eval of its exact text from a "
                    "file the shell removes before running it, and a piece ending with & is "
                    "no longer followed by ' ; '; every other command is split and joined as "
                    "before), labels org.gbench.sweagent_for_eval, "
                    "org.gbench.sweagent_for_eval_config_sha256 and "
                    "org.gbench.sweagent_for_eval_swerex_runtime checked, an image without the "
                    "last is infra_required with the rebuild command): config/default.yaml "
                    "unmodified (sha256 "
                    "05e707df58e9732ac4053f8bda0c0b175aa10f5c833f5419ba1443a01d9eb981; "
                    "function-calling bash / str_replace_editor / submit, "
                    "per_instance_call_limit 100), its README rollout command sweagent "
                    "run-batch --instances.type swe_bench on a file of instance_id, "
                    "problem_statement, base_commit and image_name only (the row's "
                    "docker_image, or starryzhang/sweb.eval.x86_64.<id> for the Python tabs), "
                    "one batch per tab; the submission is SWE-agent's own (git add -A && git "
                    "diff --cached in /testbed, as the leaderboard's SWE-agent rows submitted); "
                    "no temperature and no max_tokens sent (commented out in the fork's "
                    "LiteLLMModel._single_query; --max-output-tokens recorded as not applied); "
                    "an instance whose image could not be pulled, no .pred, an exit status "
                    "naming exit_api, or an exit_error whose exception text starts with a "
                    "model-endpoint failure while the tab's relay saw an endpoint failure is an "
                    "infrastructure failure (the fork's LiteLLMModel.query retries with "
                    "tenacity reraise=True and sends no timeout, so an endpoint that keeps "
                    "failing reaches forward_with_handling's catch-all as exit_error with "
                    "litellm.<Class>, never RetryError/exit_api, after the relay answered its "
                    "requests as failed; the text alone is not enough, since the SWE-ReX "
                    "runtime that reports exceptions runs in the model's task container); any "
                    "other exit_error or exit_environment_error (e.g. the EOF of a shell the "
                    "model killed, or a context-window overflow, as exit_context) is the "
                    "model's outcome, as upstream counts it, with agent_exit_note, counted in "
                    "exits_kept_as_model_outcome; the instance images and SWE-ReX's builder "
                    "base python:3.11.9-slim-bookworm are pulled with the host's Docker CLI "
                    "from the gbench process before the batch "
                    "(swebench_common.host_pull_images: the operator's docker login; the "
                    "SWE-agent container gets no Docker client configuration), and so is each "
                    "submitted instance's image before a Linux harness pass; such an "
                    "infrastructure failure (agent_infra), an errored instance of the fork's "
                    "report whose run_instance.log shows neither a patch-apply failure nor a "
                    "test timeout (a patch that did not apply and tests over the applied patch "
                    "that ran past the harness's timeout, read from the records swebench itself "
                    "logs, are the model's outcome, unresolved: test_timed_out, "
                    "test_timeout_instances) and a MultiLang or Windows instance in error_ids "
                    "or incomplete_ids (harness_no_verdict), and every submitted patch of a "
                    "harness pass with no report (harness_no_report) are harness errors: left "
                    "out of the tab's resolved rate, numerator and denominator "
                    "(scored_instances and effective_n, rows when the headline is over rows), "
                    "counted in harness_errors, the tab completed_with_errors, and its rate "
                    "withheld (accuracy null, status error) when nothing was scored; the Python "
                    "tabs are scored by the pinned SWE-bench-Live fork (identical to "
                    "microsoft/SWE-bench-Live python-only@ad79b850 in "
                    "run_evaluation/reporting/grading/utils/test_spec) and the MultiLang tabs "
                    "by upstream's evaluation/evaluation.py --platform linux (resolved = its "
                    "success_ids; error_ids are harness errors, failure_ids and empty_patch_ids "
                    "the model's outcome), both in the local image gbench-swe-bench-live:a145aa87 "
                    "(ubuntu:24.04 by digest, apt from the Ubuntu archive snapshot "
                    "20260925T120000Z, docker-ce-cli 5:29.8.0-1~ubuntu.24.04~noble, "
                    "dependencies pinned by swe_bench_live.constraints.txt; labels "
                    "org.gbench.swe_bench_live_fork, org.gbench.swe_bench_live_main and "
                    "org.gbench.repolaunch checked), docker-out-of-docker with "
                    "swe_thread_cap.py mounted; the Windows tab runs Win-Agent's "
                    "config/default.yaml (only dataset, model openai/<served>, base_url and "
                    "workers set) and the same harness with --platform windows on a Windows "
                    "host whose Docker engine runs Windows containers; a Win-Agent error is an "
                    "infrastructure failure before the model's first turn (no Step N... entry "
                    "before the error entry of the instance log) and, after it, only when it "
                    "names a model-endpoint failure while the tab's relay saw an endpoint "
                    "failure (the log also holds the raw output of the model's commands); any "
                    "other is the model's outcome (agent_exit_note, "
                    "exits_kept_as_model_outcome); both agents reach the model through gbench's "
                    "EndpointRelay (forwards only /v1/chat/completions, /v1/completions and "
                    "/v1/models, adds the endpoint key, sets "
                    "chat_template_kwargs.enable_thinking, and counts endpoint failures and "
                    "reply outcomes; the agents get OPENAI_API_KEY=EMPTY; Win-Agent and the "
                    "harnesses get no credential variable); an endpoint failure the relay saw "
                    "is a harness error (endpoint_failures; tab completed_with_errors, not "
                    "comparable), and model turns cut off at a limit (truncated_turns: "
                    "finish_reason length, the end of the context window since no max_tokens is "
                    "sent) make the tab not comparable, while turns with reasoning but no content "
                    "or tool call (empty_after_reasoning_turns) are split by their token ids "
                    "(swebench_common.classify_empty_reply): the answer inside the thinking "
                    "channel is the model's outcome, counted in answer_in_thinking, not a "
                    "comparability reason; a native tool call the server's tool-call parser "
                    "removed (tool_call_dropped_turns) and a turn without token ids to tell the "
                    "two apart (unclassified_empty_turns) are not the model's answer and make the "
                    "tab not comparable (both summed over the tabs at the top level); headline per "
                    "tab = resolved instance ids / rows of the tab in headline_metrics, "
                    "accuracy = Lite (the leaderboard's default tab; else the first setting "
                    "that ran); an infra_required error inside a running setting stops the run, "
                    "other setting failures are recorded per setting (settings_with_errors = "
                    "the settings whose result is status error); comparable for a full clean "
                    "tab with its reference scaffold whose pinned row count equals a board "
                    "total; GBENCH_SWE_BENCH_LIVE_MODE=oneshot opt-in; a crashed fork report "
                    "step is re-run alone (make_run_report on the scoring pass's per-instance "
                    "report.json files, at most 5 times); the Windows tab runs only after "
                    "swebench_common.check_windows_task_network found 169.254.169.254:80 and "
                    "[fd20:ce::254]:80 (the metadata server's IPv6 address) unreachable from "
                    "docker's default network on the Windows-containers engine (docker info "
                    "OSType windows): a PowerShell TcpClient connect to each (an IPv6 TcpClient "
                    "for the IPv6 address), nothing sent, in one of the tab's own instance "
                    "images already on the host (powershell, else pwsh; -EncodedCommand; --pull "
                    "never; nothing pulled). Otherwise, including on a Linux engine or when no "
                    "instance image is present (the reason names the docker pull), the tab is "
                    "infra_required in settings.windows. The agentic Linux tabs' metadata gate "
                    "also checks that SWE-ReX's build hosts resolve (SWEREX_BUILD_HOSTS: "
                    "deb.debian.org, archive.ubuntu.com, www.python.org, pypi.org, "
                    "files.pythonhosted.org); each SWE-agent batch runs under gbench's watchdog "
                    "(swe_bench_live.SweagentWatch): an instance with no file change for 6300 s "
                    "(the fork's 90-minute per-command limit plus 15 minutes, longer than the "
                    "relay's 3600 s per request) is stalled, its task container is removed and "
                    "it is a harness error (agent_stalled), the reason naming the command it was "
                    "running (the batch's per-instance swea-env-<id> trace logger); the "
                    "2026-10-08 rule that ended a dropped-heredoc command after 300 s "
                    "(agent_heredoc_dropped) is retired with the runtime fix; an "
                    "instance SWE-agent already ended is never flagged; a batch with no file "
                    "change for 6900 s is stopped (unfinished instances agent_infra), recorded "
                    "in agent_watchdog; a SIGTERM/SIGHUP to the run stops the batch (also from "
                    "a worker thread) and starts no further tab; the fork's own limits (100 "
                    "calls, 90 minutes per command, "
                    "3 consecutive command timeouts, 5 h of command time) are unchanged and "
                    "recorded in agent; the SWE-agent container runs with --init, every task "
                    "container carries the label org.gbench.sweagent_batch=<batch container> "
                    "(run-batch --instances.deployment.docker_args, the only config "
                    "difference), and every exit path removes both; an empty shard gives "
                    "accuracy None (swe_agentic.mark_empty_shard); the endpoint key is "
                    "base.endpoint_api_key"
                    "; exit_api (a tenacity RetryError, which the fork's model calls never raise)"
                    " is a harness error only when the tab's relay saw an endpoint failure (or no"
                    " relay carried the run), else the model's outcome kept with agent_exit_note;"
                    " the relay corroboration is per tab (one SWE-agent batch), not per instance")),
    },
    "swe_bench_multilingual": {
        "synced": "2026-10-10.3",
        "upstream": ("SWE-agent/mini-swe-agent 2.4.6 benchmarks/swebench.yaml (sha256 9a9c86ac, "
                     "bundled, unchanged) in the local image gbench-mini-swe-agent-v2:2.4.2-2.4.6 + swebench "
                     "5.0.2 harness in the local image gbench-swebench-v5:5.0.2 (the setup of "
                     "SWE-bench/experiments "
                     "evaluation/multilingual/20260902_mini-v2.4.6_gemini-3-5-flash)"),
        "revision": "mini-swe-agent 2.4.6, swebench 5.0.2",
        "dataset": ("SWE-bench/SWE-bench_Multilingual@846e647b9f33c0b51b739d005d13d85493c9af09 "
                    "(test, 300; v5 columns; one local JSONL snapshot feeds the rollout, the "
                    "one-shot prompts and the harness --dataset_name)"),
        "checksum": ("sha256:92abca7cb527b41a9f66d03a26ce441ff7319e3a49f985998fd56be4bb9b08b2 "
                     "(data/test-00000-of-00001.parquet at the revision)"),
        "method": (("agentic default: mini-swe-agent 2.4.6 tool-calling rollout (instance image "
                    "pulled first with the host's Docker CLI from the gbench process, "
                    "swebench_common.host_pull_images, so the operator's docker login applies and "
                    "the rollout container gets no Docker client configuration; a failed pull is "
                    "an infrastructure failure; turns cut off at the end of the context window "
                    "(truncated_turns) are counted and make the run not comparable; turns with "
                    "reasoning but no content or action (empty_after_reasoning_turns) are split by "
                    "their token ids (swebench_common.classify_empty_reply): the answer inside the "
                    "thinking channel is the model's outcome, counted in answer_in_thinking, not a "
                    "comparability reason; a native tool call the server's tool-call parser "
                    "removed (tool_call_dropped_turns) and a turn without token ids to tell the "
                    "two apart (unclassified_empty_turns) are not the model's answer and make the "
                    "run not comparable; "
                    "the one-shot harness pass pulls each row's image the same way; driver "
                    "follows run.benchmarks.swebench.process_instance; each rollout container "
                    "runs on docker's default bridge network with OPENAI_API_KEY=EMPTY and "
                    "reaches the served endpoint only through gbench's EndpointRelay on that "
                    "network's gateway (forwards only /v1/chat/completions, /v1/completions and "
                    "/v1/models and adds the endpoint key; checked reachable from that network by "
                    "a probe container of the image before the first rollout, infra_required "
                    "otherwise; an endpoint failure it saw is a harness error: endpoint_failures, "
                    "completed_with_errors, not comparable), and mounts the docker socket because "
                    "upstream's DockerEnvironment starts the instance container with docker run "
                    "and runs each of the model's commands there with docker exec; no temperature "
                    "sent unless set; no per-turn max_tokens, as the bundled config sets none, "
                    "--max-output-tokens recorded as not applied in max_output_tokens_note) in "
                    "swebench/sweb.eval.x86_64 images, verbatim diff scored by swebench 5.0.2 "
                    "run_evaluation (TestSpecs of all 300 rows built with 5.0.2; the image "
                    "carries docker-ce-cli 5:29.8.1-1~ubuntu.24.04~noble, which its container "
                    "cleanup uses, and its Ubuntu packages from the archive snapshot "
                    "20260925T120000Z on ubuntu:24.04 by digest, and swebench's dependencies "
                    "pinned by gbench/docker/swebench_v5.constraints.txt); v5 "
                    "infra_failure_ids/ambiguous_failure_ids/failure_reasons recorded as advisory "
                    "(harness/infra_failure.py), no effect on score, status or comparability; a "
                    "crashed final report step is re-run alone (swebench's make_run_report on the "
                    "scoring pass's per-instance report.json files, at most 5 times after "
                    "0/5/15/30/60 s), so an instance without report.json stays in error_ids; an "
                    "instance whose rollout could not run (agent_infra: image pull, container "
                    "start, endpoint), an errored instance whose run_instance.log shows neither a "
                    "patch-apply failure nor a test timeout (harness_no_verdict) and every "
                    "submitted patch of a harness pass with no report (harness_no_report) are "
                    "harness errors: left out of the resolved rate, numerator and denominator "
                    "(scored_instances, effective_n), counted in harness_errors, "
                    "completed_with_errors, and the rate withheld (accuracy null, status error) "
                    "when nothing was scored, while an errored instance whose log (the records "
                    "swebench itself logs) shows the patch did not apply, or the tests over the "
                    "applied patch running past the harness's timeout (test_timed_out, "
                    "test_timeout_instances), stays unresolved, the model's outcome, as upstream "
                    "counts every error_ids instance; the snapshot is checked (300 rows, v5 "
                    "columns, content sha256 "
                    "83441ebb91ffa12cb167d02ff816ba4d2d30afa5f036ade5f6ab56bee09e37ce per "
                    "swebench_common.snapshot_sha256) before it is written and before every "
                    "reuse; comparable only on config 9a9c86ac, i.e. comparable to the "
                    "SWE-bench/experiments submission 20260902_mini-v2.4.6_gemini-3-5-flash "
                    "(merge 8fa0575c; the displayed board, swe-bench.github.io@193160a4, still "
                    "shows only 2.0.0a0/2.0.0 rows on the pre-v5 dataset); "
                    "GBENCH_SWE_BENCH_MULTILINGUAL_MODE=oneshot opt-in")),
    },
    "multi_swe_bench": {
        "synced": "2026-10-11",
        "upstream": ("github.com/multi-swe-bench/multi-swe-bench (harness 24f493f8 = PyPI "
                     "multi-swe-bench 1.1.2, pinned in the evals extra) + "
                     "multi-swe-bench/MSWE-agent@88217624b637646b886cb0462995c07559e96f58 (the "
                     "scaffold of the leaderboard's MSWE-agent rows, multi-swe-bench/experiments "
                     "evaluation/<lang>/verified/*_MSWE-agent_*) with the Java repository "
                     "definitions of multi-swe-bench@e0dcb3c4ce0ae27997fb213f2a5294e1eaa487b8"),
        "revision": ("24f493f8a103e72312ded4f6b9c89f081d69cb09, "
                     "88217624b637646b886cb0462995c07559e96f58, "
                     "e0dcb3c4ce0ae27997fb213f2a5294e1eaa487b8 (the harness; MSWE-agent; the "
                     "MSWE-agent Java scaffold ref)"),
        "dataset": ("ByteDance-Seed/Multi-SWE-bench@56ff018c04a38e27ada1e9d0a6d5839a51f88f0d "
                    "(c/cpp/go/java/js/rust/ts = 1,632 instances; kotlin/ and python/ excluded)"),
        "method": ("agentic default: MSWE-agent from the local image gbench-mswe-agent:88217624 "
                   "(python:3.9.21-slim-bookworm by digest, Debian archive snapshot "
                   "20260925T120000Z, requirements.txt + dataclasses-json pinned by "
                   "mswe_agent.constraints.txt = the PyPI set of 2025-04-03; labels "
                   "org.gbench.mswe_agent, org.gbench.mswe_agent_config_sha256, "
                   "org.gbench.mswe_agent_java_repos and org.gbench.mswe_agent_driver=4, the "
                   "version of gbench/docker/mswe_agent_run.py, checked), its multirun.py with the"
                   " README / run.sh arguments (config/default.yaml unmodified sha256 "
                   "913fc5fd89b3bdffe261b9cd2e38b2e45ad9612ea57bf314bcda9f7e314b4537, "
                   "cache_task_images True, pre_build_all_images False, remove_image False, "
                   "skip_existing, per_instance_cost_limit 5.00, max_workers_build_image 16; "
                   "RUNNING_THREADS = --sandboxes), one language per container "
                   "(data/<lang>_verified.jsonl: MSWE-agent reads the language from the file name "
                   "and registers one data file per process); the published instance images "
                   "mswebench/<org>_m_<repo>:pr-N are pulled with the host's Docker CLI from the "
                   "gbench process before each language's container "
                   "(swebench_common.host_pull_images: the operator's docker login; the MSWE-agent"
                   " container gets no Docker client configuration; an image that cannot be pulled"
                   " is an infrastructure failure and is not handed to MSWE-agent), then tagged "
                   "with the name MSWE-agent's vendored harness gives them (nothing is built; the "
                   "tags are removed afterwards); MSWE-agent's own repository reset, apt setup "
                   "(SWEEnv.reset's apt-get update and apt-get install -y jq; when that apt-get "
                   "update fails because a suite is gone from its mirror, as for the 48 cpp "
                   "instances on the harness's gcc:7 base, Debian 10 buster, whose deb.debian.org "
                   "and security.debian.org sources now answer 404, the driver points the "
                   "container's apt sources at archive.debian.org with Acquire::Check-Valid-Until "
                   "false, as SWE-bench_Pro-os@66f92766 v2/tooling/locked_mini_swe.py "
                   "LockedMiniSwe.install does for old Debian images, and MSWE-agent's own apt-get"
                   " update then runs unchanged; nothing changes on an image whose apt-get update "
                   "works; recorded as agent_apt_archive per trace and apt_archive_fallback per "
                   "run; apt still failing stays agent_infra; the harness runs through "
                   "mswe_harness_build.py, upstream's run_evaluation whose image builds get the "
                   "same archive.debian.org repair only when upstream's own Dockerfile fails with "
                   "apt reporting a debian.org suite gone from its mirror (the gcc:7 base images, "
                   "e.g. mswebench/simdjson_m_simdjson:base-cpp-7, exit 100 in the 2026-10-09 "
                   "smoke run): rebuilt from that Dockerfile with the repair step after each FROM "
                   "(label org.gbench.apt-archive-fallback), recorded as "
                   "harness_apt_archive_fallback; the rebuilt base's 420 Debian packages and cmake "
                   "3.14.0 match upstream's published pr-949 image exactly; and the checkperf "
                   "reference that the builds of simdjson-949/954/958 fetch at test time (git "
                   "fetch --depth=1 origin master; today's master no longer configures with the "
                   "gcc:7 images' CMake 3.14) is pinned, by a bare clone of the image's own "
                   "/home/simdjson plus url.insteadOf, to 726c3eb611a1, the master of the clone "
                   "in upstream's published pr-949 image (built 2025-03-18), with which "
                   "upstream's gold patch reproduces the dataset's 46 passing tests exactly; "
                   "recorded as harness_checkperf_reference; a run whose pin could not be set up "
                   "(before the patch is applied) is a harness error, checkperf_reference), "
                   ".gitignore script, rm "
                   "/home/fix.patch, TITLE/CONTENT task text and submit patch (verbatim as "
                   "fix_patch); the served model is registered in OpenAIModel.MODELS and driven by"
                   " OpenAIModel with OPENAI_API_BASE_URL = gbench's EndpointRelay (forwards only "
                   "/v1/chat/completions, /v1/completions and /v1/models, adds the endpoint key, "
                   "sets chat_template_kwargs.enable_thinking, counts endpoint failures and reply "
                   "outcomes) and OPENAI_API_KEY=EMPTY; an endpoint failure the relay saw is a "
                   "harness error (endpoint_failures; completed_with_errors, not comparable; this "
                   "also covers a 400 that MSWE-agent records as exit_cost), and model turns cut "
                   "off at a limit (truncated_turns) make the run not comparable, while turns "
                   "with reasoning but no content (empty_after_reasoning_turns) are split by their "
                   "token ids (swebench_common.classify_empty_reply): the answer inside the "
                   "thinking channel is the model's outcome, counted in answer_in_thinking, not a "
                   "comparability reason; a native tool call the server's tool-call parser removed "
                   "(tool_call_dropped_turns) and a turn without token ids to tell the two apart "
                   "(unclassified_empty_turns) are not the model's answer and make the run not "
                   "comparable; sampling is "
                   "MSWE-agent's get_args defaults, temperature 0.0 and top_p 0.95, no max_tokens "
                   "(--max-output-tokens recorded as not applied); the 50-step limit of the "
                   "leaderboard's MSWE-agent runs (README --per_instance_api_calls_limit 50, which"
                   " the public code does not define; every ByteDance-Seed/Multi-SWE-bench_trajs "
                   "MSWE-agent trajectory stops at 50 model steps with 'Exceed maximum number of "
                   "steps: 50') is raised as that RuntimeError on the 51st model step (exit_error "
                   "with autosubmission, the model's outcome); the definitions of apache/dubbo, "
                   "fasterxml/jackson-core, fasterxml/jackson-databind, "
                   "fasterxml/jackson-dataformat-xml, google/gson and googlecontainertools/jib "
                   "(absent from the vendored harness, covered by the leaderboard's MSWE-agent "
                   "Java rows) are added unchanged from multi-swe-bench@e0dcb3c4 with from "
                   "__future__ import annotations for Python 3.9; exit_api, an exit_error other "
                   "than the step limit whose exception text starts with a model-endpoint failure "
                   "or the docker daemon's unreachable message "
                   "(swebench_common.scaffold_error_harness_reason), or an unfinished run whose "
                   "setup failed (setup_error), whose rollout raised before the model's first "
                   "turn or with a model-endpoint or docker daemon failure (the driver's "
                   "rollout_error; an endpoint that keeps failing: OpenAIModel.query re-raises "
                   "the openai exception, which forward_with_error_check does not catch), or "
                   "that recorded neither, is an infrastructure failure; a rollout that raised on"
                   " a model reply with no content (reasoning only: OpenAIModel.query returns "
                   "message.content unchecked and the parser's re.finditer raises TypeError on "
                   "None; multirun.main logs 'Failed on' and the instance has no prediction) is "
                   "the model's outcome, crashed_empty_reply, unresolved with no patch, counted "
                   "in empty_reply_crashes, and any other rollout exception after the model's "
                   "first turn follows the exit_error rule (crashed); MSWE-agent's OpenAI client "
                   "gets a 3660 s timeout (the relay's 3600 s per-turn limit + 60 s) instead of "
                   "the SDK's 600 s, at which it abandoned and resent a long thinking turn "
                   "(harness-only, recorded in agent.openai_client_timeout_s); "
                   "any other exit_error, and early_exit (SWEEnv.step: the container stopped "
                   "answering while it ran the model's command, which raised, broke the pipe, or "
                   "timed out and could not be interrupted), is the model's outcome, unresolved, "
                   "as a single pass of upstream's run scores it (upstream's resume with "
                   "--skip_existing would roll an early_exit instance out again; gbench rolls each"
                   " instance out once), with agent_exit_note, counted in "
                   "exits_kept_as_model_outcome; verbatim diff scored by "
                   "multi_swe_bench.harness.run_evaluation; a rollout that is an infrastructure "
                   "failure (agent_infra), a submission with a patch and no harness verdict "
                   "(error_ids or absent from final_report; harness_no_verdict) and every "
                   "submitted patch of a harness pass with no final_report (harness_no_report) are"
                   " harness errors: scoring_excluded (out of the numerator and the denominator; "
                   "scored_instances and effective_n are the denominator), counted in "
                   "harness_errors by kind (ids in harness_error_ids), completed_with_errors, and "
                   "error with the score withheld (accuracy null) when none scored (both modes); "
                   "an unresolved instance whose fix-patch run captured no test result (the "
                   "harness's report.json fix_patch_result counts all 0) and whose "
                   "fix-patch-run.log shows a package host failing to resolve (Maven Unknown host,"
                   " java.net.UnknownHostException, Node getaddrinfo EAI_AGAIN or ENOTFOUND, Go "
                   "dial tcp: lookup, curl and Cargo Could not resolve host) is a harness error "
                   "when another such instance in the run or a fresh probe container after the run"
                   " that cannot resolve one of the hosts either corroborates it "
                   "(swebench_common.corroborated_dns_failures): scoring_excluded (out of the "
                   "numerator and the denominator), harness_errors dns, "
                   "multi_swe_bench_report.harness_dns_failures, completed_with_errors, and error "
                   "with the score withheld when every submission is a harness error; "
                   "uncorroborated it stays unresolved (dns_failure_note, dns_failures_kept); both"
                   " metadata server gates (before the MSWE-agent rollouts and before the harness "
                   "pass) also check that the package hosts the instance containers' builds fetch "
                   "from at test time resolve, for the run's languages "
                   "(multi_swe_bench.runtime_hosts: repo.maven.apache.org, oss.sonatype.org, "
                   "plugins.gradle.org, services.gradle.org, registry.npmjs.org, proxy.golang.org,"
                   " index.crates.io, static.crates.io, github.com); comparable for a full clean "
                   "run at MSWE-agent's sampling; GBENCH_MULTI_SWE_BENCH_MODE=oneshot opt-in; the "
                   "metadata server probe before the one-shot harness pass runs in the MSWE-agent "
                   "image, a submitted instance's published image or a local mswebench/* image "
                   "already on the host; nothing is pulled for it (infra_required naming the "
                   "docker pull of the first submitted instance's image and the MSWE-agent build "
                   "otherwise); an empty shard gives accuracy None "
                   "(swe_agentic.mark_empty_shard); the endpoint key is base.endpoint_api_key; "
                   "the one-shot route hands every instance to run_eval_suite (shard, then "
                   "limit, same seed)"),
    },
    "spider2": {
        "synced": "2026-09-27",
        "upstream": "github.com/xlang-ai/Spider2",
        "dataset": ("xlang-ai/Spider2 repository, spider2-lite/ at the revision: "
                    "spider2-lite.jsonl (question and database metadata), evaluation_suite/gold "
                    "(gold SQL, result CSVs and eval config) and resource/documents (external "
                    "knowledge); not the stale HF xlangai/spider2-lite snapshot; the SQLite "
                    "databases (resource/databases/spider2-localdb) are not in the repository: "
                    "the spider2-localdb archive the upstream evaluation_suite README links, "
                    "whose content the run does not check"),
        "revision": "cafb867313aab4e674652054198f383cf4018943",
        "method": ("Spider 2.0-lite execution accuracy vs SQLite with "
                   "evaluation_suite/evaluate.py's local path copied verbatim at the revision: "
                   "extract_sql_query, get_sqlite_result with the result written to CSV in chunks "
                   "of 500 and read back with pd.read_csv (so a number stored as TEXT compares as "
                   "the gold CSV's number), resolve_gold_paths ({id}.csv alone when it exists, "
                   "else the {id}_a, {id}_b variants), compare_pandas_table or "
                   "compare_multi_pandas_table with condition_cols and ignore_order (default "
                   "False) as spider2lite_eval.jsonl gives them; a query that fails, or leaves no "
                   "readable result file (Evaluation Error), scores 0; a missing prerequisite or "
                   "an incomplete checkout (a referenced database, document or gold file absent) "
                   "hard-errors (infra_required); the prompt is gbench's own zero-shot prompt "
                   "(CREATE TABLE DDL from the .sqlite, the external-knowledge document, the "
                   "question, a ```sql block) and the sampling is the run's, since Spider2-lite "
                   "prescribes none (its baselines each build their own; stated in prompt and "
                   "named in leaderboard_comparable_reason); metadata, gold, databases and "
                   "documents come from a local xlang-ai/Spider2 checkout (GBENCH_SPIDER2_REPO_DIR"
                   " / GBENCH_SPIDER2_GOLD_DIR) at the revision (main HEAD of 2026-08-12); the run"
                   " reads each checkout's commit (git rev-parse HEAD, spider2_repo_revisions "
                   "beside spider2_repo_pinned) and names a checkout at another commit, or one "
                   "whose commit cannot be read, in leaderboard_comparable_reason; each query is "
                   "capped at 120 s (gbench's; upstream's local path has no limit), a timeout "
                   "scored 0; a grader crash (SCORING_ERROR: a comparison that raises, upstream's "
                   "Python Script Error, or a missing database or gold file) and a request with no"
                   " reply are harness errors, left out of the score (scoring_excluded) and named "
                   "in status_reason and the comparability reason"),
    },

    # Pillar 4: long context and retrieval.
    "aa_lcr": {
        "synced": "2026-10-09",
        "upstream": ("huggingface.co/datasets/ArtificialAnalysis/AA-LCR (dataset card v1.1) + "
                     "Artificial Analysis Intelligence Benchmarking methodology (General Testing "
                     "Parameters)"),
        "revision": "9a77ef56b717057ade24ceab4d273712a0b4f19e",
        "dataset": "ArtificialAnalysis/AA-LCR (csv + extracted-text zip, v1.1)",
        "method": ("card prompt template + v1.1 equality-checker system/user prompts verbatim, "
                   "sent with nothing the card does not publish (no JSON mode; no temperature, "
                   "the official checker GPT-5.6 Luna (medium) being a reasoning model; "
                   "judge_sampling), Gemini cascade, every judged attempt keeping the checker's "
                   "replies and the Gemini model of each (judge_replies); every listed doc loaded "
                   "(hard error if missing); pass@1 over 3 repeats; General Testing Parameters: "
                   "no-think temperature 0 and max_tokens 16384 (shrunk to the window), "
                   "--thinking temperature the lab's recommendation as AA's methodology gives it "
                   "(the temperature the served model's generation_config.json sets, read from "
                   "--tokenizer else the served model id by "
                   "ojbench.served_model_recommended_sampling, else 0.6; a config that cannot be "
                   "read is infra_required before generation; the source recorded in "
                   "official_sampling) "
                   "and no max_tokens (maximum allowed); harness errors (an attempt whose request "
                   "produced no reply: endpoint error, timeout or over-context; a checker with no "
                   "CORRECT/INCORRECT verdict in 3 replies, JUDGE_FAILED) are left out of the "
                   "accuracy, counted in harness_errors, named in leaderboard_comparable_reason, "
                   "status completed_with_errors, score withheld when no attempt is scored "
                   "(Artificial Analysis does not publish results that persistent API failures "
                   "affected); the checker's reasoning effort (medium) is not mapped to a Gemini "
                   "thinking level; leaderboard_comparable=False"
                   "; a bare verdict is read by the relaxed parse, counted"),
    },
    "longbench_v2_code": {
        "synced": "2026-10-10.3",
        "upstream": "github.com/THUDM/LongBench (LongBench-v2 pred.py, without and with --cot)",
        "revision": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
        "dataset": ("zai-org/LongBench-v2@2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9 (Code "
                    "Repository Understanding, 50 of 503)"),
        "method": ("both official settings run by default on the same questions, each scored and "
                   "reported on its own (headline accuracy = non-CoT; settings.non_cot, "
                   "settings.cot, headline_metrics non_cot_accuracy / cot_accuracy): non-CoT = "
                   "prompts/0shot.txt verbatim, one query at max_new_tokens=128; w/ CoT (pred.py "
                   "--cot) = prompts/0shot_cot.txt verbatim at max_new_tokens=1024, then "
                   "prompts/0shot_cot_ans.txt verbatim with the stripped CoT reply at $COT$ "
                   "(pred.py's .replace chain, $COT$ last) at max_new_tokens=128, scored on that "
                   "answer query; no answer query when the CoT query returns '' (pred.py's if "
                   "output == '': continue): an empty CoT reply is the model's, scored wrong and "
                   "flagged; a CoT or answer query with no reply from the served endpoint is a "
                   "harness error, left out of its setting's score (scoring_excluded, "
                   "request_harness_errors, named per setting in status_reason; accuracy null with"
                   " accuracy_withheld_reason when nothing is scored); each query's cap as "
                   "OfficialSampling.max_tokens through base's token-limit policy (exact without "
                   "thinking, --max-output-tokens not applied; the run budget with thinking, "
                   "'official cap not applied to reasoning' recorded per query); token "
                   "middle-truncation of every query with the served tokenizer to pred.py's "
                   "max_len for the model's context window (README: add the model's context window"
                   " length to config/; config/model2maxlen.json: 120,000 for 128K-context models,"
                   " 200,000 for the 200K claude-3.5-sonnet): 120,000 for a served window up to "
                   "131,072 tokens, else the window - the largest official cap of the two "
                   "settings (1,024, the w/ CoT query's) - 64, one max_len for every query of both "
                   "settings (UPSTREAM_MAX_LEN_RULE), sent as min(that, window"
                   " - the query's budget - 64); the model's own window (config "
                   "max_position_embeddings from the local HF cache) recorded as "
                   "model_context_window; extract_answer verbatim; temperature 0.1 for every query"
                   " in both modes (pred.py query_llm; OfficialSampling); --shard before "
                   "--eval-limit, both settings on the same draw; leaderboard_comparable computed "
                   "per setting and overall (both): True for full all-domain no-think T=0.1 runs "
                   "with every request answered on a window of at least 121,088 tokens (non-CoT "
                   "alone from 120,192); otherwise the reason names the code slice, a thinking "
                   "run, prompts cut shorter than upstream's max_len "
                   "(prompts_cut_shorter_than_upstream), prompts truncated on a served window "
                   "smaller than the model's context window, a no-think request sent a max_tokens "
                   "other than its cap, the temperature, requests with no reply (harness errors, "
                   "left out of the score) and empty replies (scored wrong; pred.py leaves both "
                   "out), --eval-limit; a window with room for the non-CoT query only reports "
                   "settings.cot as infra_required (settings_infra_required, status "
                   "completed_with_errors)"
                   "; UPSTREAM_MAX_LEN_RULE: one max_len per window for both settings"
                   "; a reply whose content is '' or None (pred.py writes no prediction for it "
                   "and a resumed run asks the question again) is asked again, the same request, "
                   "through swebench_common.upstream_reask_relay with "
                   "reply_content_is_null_or_empty, up to longbench_v2_code.NULL_REPLY_REASKS=5, "
                   "for every query of both settings (upstream_reask per setting, per query for "
                   "w/ CoT); only a reply still empty after the re-asks is scored wrong and "
                   "flagged; an answer query asked again with the same CoT (a resumed pred.py "
                   "--cot run asks the CoT query again too) is flagged"
                   "; only a reply the model ended itself (finish_reason stop) is re-asked: a "
                   "reply cut off at the token budget is the model's own outcome, never re-asked, "
                   "scored as base scores a truncated reply and flagged (cut_off_not_reasked); the "
                   "re-asks stop before the client's per-request timeout "
                   "(EndpointRelay.reask_deadline), and a chain that ends on a reply still "
                   "matching the rule (reasks_exhausted, of them reasks_cut_short) passes that "
                   "reply on to be scored, never a harness error; upstream_reask also records the "
                   "classes of the empty replies seen (answer_in_thinking_seen, "
                   "tool_call_dropped_seen, unclassified_empty_seen, answer_in_thinking_rate_seen)"),
    },
    "loft_scifact": {
        "synced": "2026-10-09",
        "upstream": "github.com/google-deepmind/loft",
        "revision": "219f68e470260c22fc53cc8cc2416b78313db774",
        "dataset": "LOFT SciFact 128k bundle (gs://loft-bench/retrieval/scifact.zip, test split)",
        "checksum": ("sha256:f7ef0719c1edb7adc343f3b5411c17868602421db1683fddb5e85aa4dda84a91 "
                     "(scifact.zip)"),
        "method": ("retrieval_scifact_128k_test:few_shot_with_cot prompt as LOFT's registry "
                   "renders it (byte-identical for all 100 queries); sampling temperature 0.0, "
                   "top_p 1.0 and no max_tokens (inference/models.py GenerationConfig sets "
                   "none; --max-output-tokens not applied); extract_prediction + ID->pid "
                   "mapping; recall@1; bundle sha256 verified at download; "
                   "leaderboard_comparable also requires every request answered (failed or "
                   "timed-out requests named) and no reply cut off at the end of the context "
                   "window (replies_cut_off_by_budget)"),
    },
    "ruler": {
        "synced": "2026-10-09.2",
        "upstream": "github.com/NVIDIA/RULER",
        "revision": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
        "method": ("pinned checkout's prepare.py (revision verified at run time) in "
                   "--prepare_for_ns mode with the served chat template, so template + prompt + "
                   "answer_prefix + cap fit the band; input+answer_prefix as one user turn "
                   "(RULER's chat-API client path); sampling temperature 0.0, top_p 1.0, seed 0 "
                   "(config_models.sh + call_api.py, the fields the chat-API client sends); "
                   "per-task caps as each sample's max_tokens (sent without --thinking; with "
                   "--thinking replaced by the run budget under the token-limit policy and "
                   "recorded in official_cap_not_applied_to_reasoning; pre-flight sized the "
                   "same way); string_match_all/part; Avg over 13 tasks x 6 bands; a prompt the "
                   "pre-send check would not send (OVER_CONTEXT) raises infra_required "
                   "(pre-flight on each cell's longest prompt by RULER's token count, and after "
                   "the run), never scored 0; the pre-flight uses the budget base sends (the "
                   "declared cap without --thinking; with it the run's budget, the caller's "
                   "max_output_tokens included, larger or smaller than the cap); "
                   "leaderboard_comparable also requires every request answered (failed or "
                   "timed-out requests named) and no reply cut off by the run's token budget "
                   "(replies_cut_off_by_budget)"),
    },

    # Pillar 5: tool use and agentic workflows.
    "agent_dojo": {
        "synced": "2026-10-11",
        "upstream": "agentdojo (PyPI) / github.com/ethz-spylab/agentdojo",
        "revision": ("0.1.35 (benchmark v1.2.2, attack important_instructions, model label "
                     "vllm_parsed)"),
        "method": ("delegated to the agentdojo package: no-defense pipeline, "
                   "benchmark_suite_without/with_injections; no temperature and no max_tokens "
                   "sent (max_output_tokens_note), as the vllm_parsed provider (OpenAILLM "
                   "default 0.0 goes out as `temperature or NOT_GIVEN`; OfficialSampling "
                   "send_temperature=False, with and without --thinking); headline benign "
                   "utility, ASR separate; comparable only for a full run of the pinned "
                   "package/version/attack/model label whose requests carry no temperature"
                   "; requests carry the served id gbench passes (GBENCH_AGENTDOJO_MODEL "
                   "overrides), recorded as request_model_id, in place of the first id /models "
                   "lists; the endpoint key is base.endpoint_api_key; --shard applies to the "
                   "(suite, user task) list before the global --eval-limit budget, with "
                   "injection pairs following their user task; item_results plus merge_shards; "
                   "an empty shard gives accuracy None; execution sequential; attempts "
                   "not_applied"
                   "; per-episode runners; harness errors are left out instead of stopping the "
                   "run; context-length episodes are scored as AgentDojo scores them, counted"
                   "; replies empty after reasoning that stopped on their own are split by their "
                   "token ids (return_token_ids asked once a reply shows vLLM; "
                   "swebench_common.classify_empty_reply with the served tokenizer's "
                   "<|tool_call> id): answer_in_thinking (the model's outcome), tool_call_dropped "
                   "(a server-parser artifact) and empty_unclassified (no token ids), the last "
                   "two named in leaderboard_comparable_reason"),
    },
    "bfcl_v4_agentic": {
        "synced": "2026-10-13",
        "upstream": ("gorilla bfcl_eval @6ea5797 = bfcl-eval 2026.3.23 (PyPI), in the local "
                     "image gbench-bfcl:2026.3.23 (gbench/docker/bfcl.Dockerfile)"),
        "revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 (bfcl-eval 2026.3.23)",
        "method": (("delegated to bfcl-eval generate/evaluate in the harness image (project root "
                    "mounted at the same path; bfcl generate on docker's bridge network (never "
                    "the host network), after the metadata-server guard and a TCP check of a "
                    "loopback endpoint at host.docker.internal, bfcl evaluate and the other jobs "
                    "with --network none; only the variables the harness reads passed by name, "
                    "none holding a key (a real OPENAI_API_KEY added by "
                    "swebench_common.EndpointRelay on the bridge gateway, the containers holding "
                    "EMPTY; the Gemini key added by gemini_relay.GeminiRelay there, native "
                    "generateContent of the cascade's models only, the web_search generate "
                    "holding its per-run placeholder; OPENAI_DEFAULT_HEADERS never passed; a "
                    "forwarded credential -> infra_required; a harness project root holding a "
                    ".env (upstream's generate/evaluate run load_dotenv(PROJECT_ROOT/.env, "
                    "override=True)) -> infra_required before any container starts, and the "
                    "harness child's load_dotenv replaced by a no-op (_disable_dotenv); relay "
                    "records in bfcl_report.relays); the harness child of every bfcl command runs "
                    "under the bfcl suite's execution guard in place of multi_turn_utils.eval "
                    "(the backend methods upstream maps, literal arguments; anything else "
                    "upstream's \"Function call X is not allowed.\"; refusals in "
                    "bfcl_report.execution_guard_refusals, not comparable); the web_search "
                    "generate's metadata guard also resolves vertexaisearch.cloud.google.com and "
                    "the relays' gateway is checked reachable before generating; gate failures "
                    "name bfcl and docs/evals/bfcl.md when bfcl runs the track; gbench's process "
                    "never imports bfcl_eval, so nothing is written next to a package); the 35 "
                    "bfcl_eval files the track reads or runs and the category mapping are checked "
                    "in the image against 6ea5797 (else infra_required); OpenAICompletionsHandler "
                    "unmodified except the --thinking chat_template_kwargs toggle and a "
                    "per-request timeout sized for the served window (base.request_timeout_s of "
                    "max_model_len, else of 262144; the SDK's 2 retries kept; cf. "
                    "base_oss_handler.py timeout=72000) (no decode override: a reply without tool "
                    "calls ends the turn and its text is what agentic_checker grades; "
                    "underscore_to_dot as upstream's entry for that handler); temperature 0.001 "
                    "and no max_tokens (OfficialSampling, __main__.py generate default; the bfcl "
                    "suite hands over its own temperature); memory_vector encoder "
                    "all-MiniLM-L6-v2@1110a243 in the image, loaded offline on the CPU; "
                    "web_search = BFCL's own tool contract (web_search.json function docs, "
                    "search_engine_query(keywords, max_results=10, region='wt-wt'), a list of "
                    "{title, href, body}, {title, href} without snippets, cut to max_results) "
                    "served by Gemini Google Search grounding in place of the harness' "
                    "DuckDuckGo-via-SerpAPI search: search_engine_query replaced in memory in the "
                    "web_search generate's harness child (a plain function; no bfcl_eval file "
                    "changed) by search_tool.search_with_fallback (generateContent with "
                    "google_search, base.search_cascade() (gbench's one Gemini cascade "
                    "base.GEMINI_CASCADE unless GBENCH_SEARCH_MODELS/GBENCH_SEARCH_MODEL or "
                    "GBENCH_JUDGE_MODELS/GBENCH_JUDGE_MODEL name others; handed to the container "
                    "as GBENCH_SEARCH_MODELS/GBENCH_JUDGE_MODELS via base.gemini_cascade_env; the "
                    "models that answered in gemini_models_answered, gemini_cascade), global rate "
                    "limit, whole-cascade retries; region not applied, href the grounding "
                    "redirect URL, body the grounded text Gemini attributes to the source); a run "
                    "that scores it is not comparable ('web search served by Gemini Google Search "
                    "grounding (BFCL queries DuckDuckGo via SerpAPI)'; bfcl_report.search_backend "
                    "and search_substitution record it); Gemini key (GEMINI_API_KEYS, else "
                    "GEMINI_API_KEY) checked with base.gemini_key_live_valid before generating, "
                    "the judged suites' gate (unset or rejected (HTTP 401/403, a 400 naming the "
                    "API key) -> the Web Search setting reported infra_required in its own entry "
                    "(settings.web_search, settings_infra_required, prerequisite GEMINI_API_KEY; "
                    "not generated, never scored, overall_accuracy_agentic null, accuracy = "
                    "summary_memory, completed_with_errors, not comparable) while the Memory "
                    "setting runs; a check that gets no answer does not block; only web_search "
                    "categories selected -> infra_required hard error); every search recorded in "
                    "the harness child (backend, result count, error; "
                    "bfcl_report.web_search_calls); credentials masked in the search results and "
                    "errors, in every tool result the harness hands the model (it evals the calls "
                    "in its own process), in the search log, the result/score files and the "
                    "container output; a search no grounding model served (BFCL's own error text "
                    "to the model) -> the Web Search setting infra_required (search log read "
                    "every 30 s during generate, stopped at the first one; after such a stop the "
                    "memory entries not finished are generated by the resume built into bfcl "
                    "generate (the memory categories without --allow-overwrite: "
                    "collect_test_cases keeps the finished entries, "
                    "clean_up_memory_prereq_entries drops finished prerequisites), without the "
                    "Gemini relay and bounded on its own, and only the memory categories are "
                    "evaluated; only the web_search generate uses the Gemini relay); "
                    "generate/evaluate have no wall clock unless GBENCH_BFCL_TIMEOUT_S (each on "
                    "its own): a stopped generate reads the search log: a failed search makes the "
                    "Web Search setting infra_required (memory resumed), otherwise the entries "
                    "the harness finished are scored with --partial-eval (cut-off lines dropped; "
                    "in a limit run a stop during the harness' whole-file rewrite (update_mode) "
                    "can also drop finished entries, counted unfinished; with a wall clock a full "
                    "run's planned counts are read from the image before generating; a limit "
                    "run's result files cleared before generating; evaluate skipped when no "
                    "scored entry was written) and the unfinished planned entries are harness "
                    "errors left out of the score (unfinished; counted as timed out; "
                    "harness_errors, completed_with_errors, not comparable); a stopped evaluate "
                    "raises HarnessTimeout (carrying the record of a generate stopped before it); "
                    "category set = MEMORY_CATEGORIES + WEB_SEARCH_CATEGORIES, checked against "
                    "the harness mapping; headline overall_accuracy_agentic = unweighted mean of "
                    "web-search and memory summaries over unrounded category accuracies "
                    "(eval_runner_helper.py); score files cleared before evaluate, non-zero "
                    "evaluate exit and unscored categories hard-error; error_types reported, a "
                    "run where every question is agentic:no_last_message flagged; scored entries "
                    "whose request raised (upstream's \"Error during inference\", which its checker "
                    "scores wrong) -> harness errors left out of the score (base's rule: off the "
                    "category's total; counted in failed_requests / timed_out_requests / "
                    "over_context_prompts, bfcl_report.inference_errors, scoring_excluded and "
                    "request_harness_errors endpoint_failure/endpoint_timeout/over_context, per "
                    "category in bfcl_report.harness_errors_left_out; score-file traces marked "
                    "scoring_excluded with harness_error; named in leaderboard_comparable_reason; "
                    "completed_with_errors); a category with no entry left in the score -> "
                    "unscored (categories_unscored; its group summary and "
                    "overall_accuracy_agentic null, accuracy = accuracy_pooled over the entries "
                    "in the score, the metric says so); result and score files that disagree on "
                    "the counts -> hard error; nothing left in the score -> score withheld "
                    "(accuracy null, accuracy_withheld_reason, status failed); EndpointRelay "
                    "upstream_errors + requests_refused -> endpoint_failures, "
                    "completed_with_errors and a named non-comparability reason "
                    "(relay_result_reasons semantics); memory prerequisite entries (the *_prereq "
                    "conversations, in BFCL_v4_<category>_prereq_result.json) whose request "
                    "raised -> memory_prereq_failed_requests and "
                    "bfcl_report.prereq_inference_errors (failed/timed_out/over_context per "
                    "category; not scored entries, so out of failed_requests; upstream flushes a "
                    "prerequisite's memory snapshot only after its whole conversation, so the "
                    "scenario's questions ran against an incomplete memory), every scored entry "
                    "of that scenario in that category (utils.process_memory_test_case gives each "
                    "depends_on = all of the scenario's prerequisites; scenario read from the "
                    "entry id) left out of the score as memory_prereq_failed (off the total, and "
                    "off the correct count when the score file, which lists every entry the "
                    "checker scored wrong, does not list it), the result and the Memory setting "
                    "(settings.memory.prereq_failed_requests, scoring_excluded, "
                    "categories_unscored) completed_with_errors, and a named non-comparability "
                    "reason; every error of the run names the running suite (bfcl and "
                    "docs/evals/bfcl.md when bfcl runs the track); the endpoint check passes "
                    "proxied=True when a proxy is set; a missing or rejected Gemini key on the "
                    "suite's own run is infra_required before generation, naming "
                    "--eval-categories memory_kv,memory_vector,memory_rec_sum; a mid-run search "
                    "outage withholds the score (status error, failure_kind infra_required); "
                    "bfcl's delegated agentic part is unchanged; the registry entry always uses "
                    "gbench's OpenAICompletionsHandler with neutral metadata (org from an "
                    "'org/name' id prefix, license unknown, no url); --shard (own run) goes "
                    "through the id file, each category's scored ids sharded before the limit; "
                    "merge_shards sums per-category counts and then applies BFCL's aggregation; "
                    "the endpoint key is base.endpoint_api_key; attempts not_applied"
                    "; a group needs every category; a subset headline is pooled; the zero-filled"
                    " value is kept")),
    },
    "browsecomp": {
        "synced": "2026-10-10.2",
        "upstream": "openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47",
        "revision": "f975f60afc3c811de82951165f5a9a921d024e14",
        "dataset": ("smolagents/browse_comp (test; row for row identical to simple-evals "
                    "browse_comp_test_set.csv, 1,266 rows)"),
        "checksum": ("sha256:7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf (= "
                     "simple-evals browse_comp_test_set.csv)"),
        "method": ("per-row XOR-decrypt (canary; simple-evals derive_key/decrypt); QUERY_TEMPLATE "
                   "verbatim as the user turn, closed book, single turn; model sampled as "
                   "simple-evals samples it (gpqa.SIMPLE_EVALS_SAMPLING, "
                   "gpqa.simple_evals_messages): system message \"You are a helpful assistant.\" "
                   "and temperature 0.5 in both modes, max_tokens 2048 on every no-think request "
                   "(--max-output-tokens not applied), the run's --max-output-tokens with "
                   "--thinking (official cap not applied to reasoning); GRADER_TEMPLATE verbatim "
                   "('if there if there', two blank lines, literal 0|\\%|) with the raw decrypted "
                   "problem as [question]; grader call as grading_sampler over the Gemini "
                   "cascade: system message, temperature 0.5, max_output_tokens 2048 plus the "
                   "Gemini thinking allowance 8192, replies still cut there counted and named "
                   "(judge_sampling; summed by a sharded run's merge); verdict = the captured yes/no of correct:\\s*(yes|no) "
                   "(upstream's group(0) == 'yes' comparison never holds; verdicts upstream's "
                   "pattern misses counted and named); a reply without a verdict is JUDGE_FAILED, "
                   "excluded and named (upstream defaults it to 'no'); judge_model and "
                   "judge_reply per trace; accuracy; leaderboard_comparable=False (judge)"),
    },
    "gorilla_apibench": {
        "synced": "2026-10-09",
        "upstream": "github.com/ShishirPatil/gorilla (eval/)",
        "revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "dataset": ("gorilla-llm/APIBench@ac21e1892e634dfa25f8ad75f16cbdbfb0a5736d + "
                    "eval-data/questions/*_{0_shot,bm25,gpt_index,oracle}.jsonl (3 hubs x 4 "
                    "settings, 1775 questions per setting, 7100 total; 12 question files "
                    "blob-checked)"),
        "method": ("all four settings Gorilla ships (0-shot; BM25, GPT-Index and Oracle "
                   "question files as shipped, the retrieved document already in the question: "
                   "the inputs of the repo's own "
                   "eval-data/responses/*_{bm25,gpt_index,oracle}.jsonl; the README's "
                   "live-retrieval path get_llm_responses_retriever.py, with another prompt "
                   "suffix and an OpenAI-embedding index for gpt, is not run); encode_question "
                   "prompt verbatim; sampling as get_llm_responses.py (n=1, temperature=0, no "
                   "max_tokens; OfficialSampling send_max_tokens=False); ast_eval_{th,hf,tf}.py "
                   "ported (tree-sitter sub-tree match vs the hub API database; correct = "
                   "matched entry's domain, else hallucination/error; tree-sitter 0.21.x + "
                   "tree-sitter-python 0.21.0 checked before the run); per-hub and per-setting "
                   "reported on reasoning-stripped text, headline = unweighted 0-shot hub mean "
                   "(leaderboard_comparable=False); verified vs upstream scripts on synthetic "
                   "replies"),
    },
    "mcp_atlas": {
        "synced": "2026-10-12",
        "upstream": "github.com/scaleapi/mcp-atlas (agent environment image 1.2.7)",
        "revision": "f24ba3fb0bfa484c86acb28431fad6d7282455f9",
        "dataset": ("ScaleAI/MCP-Atlas@8c563b55d7c967755f474299848049834d624617 (train, 500 public"
                    " tasks)"),
        "method": ("release agent environment built locally from "
                   "services/agent-environment/Dockerfile (docker/mcp_atlas.Dockerfile: base = "
                   "the 1.2.7 image's uv 0.9.30 bookworm base by digest; unpinned server "
                   "dependencies resolved as of the 1.2.7 build, 2026-07-17T19:29:31Z; "
                   "brave-search, the one search-engine server whose tools are all search "
                   "tools, removed from mcp_server_template.json, label "
                   "gbench.mcp_atlas.search_servers=removed:brave-search; ddg-search "
                   "(duckduckgo-mcp-server==0.5.0), exa (exa-mcp-server@3.2.1) and oxylabs "
                   "(oxylabs-mcp==0.4.1) kept, as the release has them); only the six search "
                   "tools of brave-search, ddg-search, exa and oxylabs (brave_web_search, "
                   "brave_local_search, search, web_search_exa, google_search_scraper, "
                   "amazon_search_scraper; GEMINI_SERVED_TOOLS) = Gemini Google Search "
                   "grounding (search_tool cascade) called from gbench's process (no key in any "
                   "container) behind the same tool names and inputSchemas the pinned servers "
                   "list (sha256-pinned; each name listed once, the running servers' own "
                   "entries giving way to the pinned listing, and a running server whose entry "
                   "differs from it, or that lists a tool it lacks, is infra_required: "
                   "catalog_drift; arguments checked with jsonschema.validate as the "
                   "environment's mcp 1.12.1 server does, 'Input validation error: ...'), "
                   "replies in each server's text layout with the environment's error body, "
                   "cache and 60 s timeout (search tools), recorded in search_substitution "
                   "(BRAVE_API_KEY not read); their page tools run as upstream on the release's "
                   "servers (UPSTREAM_SEARCH_TOOLS): ddg-search_fetch_content in the "
                   "environment container (no key), exa_web_fetch_exa (EXA_API_KEY) and "
                   "oxylabs_universal_scraper / oxylabs_amazon_product_scraper "
                   "(OXYLABS_USERNAME / OXYLABS_PASSWORD) in the credentialed-servers "
                   "container, calls passed on and replies returned unchanged (credential "
                   "values masked), counted in search_substitution.upstream_tool_calls, and a "
                   "page-tool reply saying the server could not reach the Exa or Oxylabs API at "
                   "all (provider_unreachable: oxylabs 'Request error during POST request: "
                   "...'; exa's HTTP 500 error reply whose detail is exactly '<tool> error: "
                   "fetch failed', never page text) or that the provider refused the run's "
                   "account (provider_account_error: exa '<tool> error (401|402|429): ...' and "
                   "the server's stdio rate-limit reply, with Exa's 403 left to the model since "
                   "Exa also uses it for content-safety refusals; oxylabs 'Error: Expecting "
                   "value: line 1 column 1 (char 0)', the reply to the API's HTTP 401 with an "
                   "empty body (a provider-side HTML error page reads the same), 'HTTP "
                   "error during POST request: 401|403|407|429 - ...', and its error reply "
                   "with no job record (\"Error: 'job'\": an account refusal whose JSON "
                   "carries no job; the reply has no status, so a malformed request's HTTP "
                   "400 without a job record reads the same and is excluded too)) is a "
                   "harness error "
                   "(harness_errors.provider_account_error_calls, "
                   "search_substitution.upstream_provider_account_errors); a search call the "
                   "Gemini API fails to serve (every model of the cascade failed, or no answer "
                   "within the 60 s timeout), a Gemini call still running when the task's wall "
                   "clock stops it, or a call that cannot reach the environment is a harness "
                   "error (task HARNESS_ERROR, excluded from every metric, status "
                   "completed_with_errors, harness_errors; a run left with nothing to score has "
                   "accuracy null with accuracy_withheld_reason); mongodb server data = the "
                   "release's data_exports MongoDB dump (sha256 "
                   "83f2919e33d5237f31a5f12362038f4a23e648701d4cfc190f0dd53cfc2f7453) restored "
                   "into a local mongod 8.0.32 (Dockerfile target mongo) on the credentialed "
                   "servers' network unless MONGODB_CONNECTION_STRING is set; run_eval.py "
                   "pre-flight (enabled-servers + all 36 test_servers.py calls: brave-search's "
                   "and ddg-search's through Gemini grounding, exa's and oxylabs' on their own "
                   "servers so their keys are proven, and failed too when the reply says the "
                   "provider is unreachable or refuses the account (a gbench check stricter "
                   "than the release's pass rule), and ddg-search running in the environment "
                   "for its fetch_content) else infra_required; tools = the /list-tools "
                   "catalogs of both containers with the ten pinned search-engine entries, "
                   "filtered to ENABLED_TOOLS via _transformToolCalls; /call-tool replies "
                   "mapped as sandbox-client.ts/agent-eval.ts (a credentialed-servers reply "
                   "with every credential value of gbench's environment masked); tool-call "
                   "arguments that fail JSON parsing are not executed and the parse error is "
                   "fed back; 256 turns/100 calls, no forced answer, last content graded; "
                   "run_eval.py --timeout 1800 per task (scored 0); no temperature and no "
                   "max_tokens sent (litellm-strategy.ts payload {model, messages, tools}; the "
                   "server's default, the room left in the context window, applies to each "
                   "turn; --max-output-tokens recorded as not applied); a lost environment is "
                   "infra_required; score_claims.py extract_claims + verbatim per-claim prompt "
                   "with structured output, empty or ERROR: responses scored 0, the Gemini "
                   "model of each claim's verdict recorded (claim_judge_models); headline "
                   "pass_rate_0.75 = leaderboard 'Pass Rate % (Public 500)'; judge = Gemini "
                   "cascade and search tools = Gemini grounding, leaderboard_comparable=False; "
                   "the environment runs as two containers of the image (docker run --pull "
                   "never), each started with the environment's own ENABLED_SERVERS: the 20 "
                   "servers that need no credential (the model's shell, code and file tools and "
                   "ddg-search among them) with no credential, on a docker network of its own "
                   "and published on 127.0.0.1, and the 15 that need one (exa and oxylabs among "
                   "them) with their variables (by name) on another docker network, with no "
                   "published port and the API on the container's loopback reached through "
                   "docker exec (none of them runs a local shell, code or file tool; e2b-server "
                   "runs code remotely in E2B's sandboxes, which hold no key of the run; exa "
                   "and oxylabs send the model's URLs and queries to the Exa and Oxylabs APIs, "
                   "which fetch the pages from their own infrastructure; github and "
                   "weather-data are fetched unpinned from github.com/geobio when the "
                   "environment starts and were not inspected); calls routed by server after "
                   "main.py's name mapping, the two catalogs merged in template order, the "
                   "pre-flight over both; an unset credential of any server is infra_required "
                   "before anything starts, naming each server with exactly its unset variables "
                   "and the release sample data (data_exports) the data servers need; a "
                   "lara-translate_import_tmx call whose tmx_url its shell command would "
                   "interpret gets gbench's error reply, counted in "
                   "tool_calls.refused_by_gbench; each container only after "
                   "container_guard.require_metadata_blocked found the cloud metadata server "
                   "unreachable from its docker network (reachable or no verdict is "
                   "infra_required naming the host firewall rule) and resolved there the host "
                   "names its servers need (ENVIRONMENT_HOSTS: the npm registry, PyPI, "
                   "github.com and the keyless servers' API hosts; CREDENTIALED_HOSTS: the same "
                   "package hosts and the credentialed servers' API hosts, api.exa.ai and "
                   "realtime.oxylabs.io included; a DNS outage is found before the run, "
                   "recorded in agent_environment.dns_names_checked); "
                   "GBENCH_MCP_ATLAS_SANDBOX_URL is refused when it is configured with a server "
                   "that needs a credential or when a credential variable of the release's "
                   "servers (or GEMINI_API_KEY / GEMINI_API_KEYS) holds a value in its code "
                   "tool's environment or any process environment that tool can read "
                   "(/proc/<pid>/environ, PID 1 included; names only, checked through its own "
                   "code tool before the model starts, no verdict infra_required, recorded in "
                   "agent_environment.external_credential_check), its metadata-server verdict "
                   "comes from container_guard's probe script run through its own code tool "
                   "(mcp-code-executor_execute_code, else mcp-server-code-runner_run-code, "
                   "uncached) before the model starts, with no verdict infra_required, and "
                   "gbench still runs the credentialed servers in its own container; the loader "
                   "applies --shard to the full ordered rows before its seeded --eval-limit "
                   "draw (presharded into run_eval_suite)"
                   "; a provider-side HTTP 5xx of the exa/oxylabs page tools is a harness error "
                   "(provider_unavailable_calls); catalog_drift also names a pinned tool a "
                   "running search server does not list; other credentialed servers' "
                   "account-error replies are counted (tool_calls.account_error_suspects) and "
                   "named, not excluded"),
    },
    "nexusraven_api_eval": {
        "synced": "2026-10-09.2",
        "upstream": "github.com/nexusflowai/NexusRaven (langchain==0.0.294)",
        "revision": "f32882763bb3e220722883821919c14abdd3d156",
        "dataset": ("Nexusflow/NexusRaven_API_evaluation@85c7c326fcfb9c6587daabd9582e7cfaddcc65bd "
                    "(CC-BY-NC-4.0; 5 tasks, 339 queries)"),
        "method": (("evaluator.py SIMPLE agent: LangChain structured-chat prompt as OpenAIChat's "
                    "single user message, stop Observation:, one OutputFixingParser repair; "
                    "temperature 0 and no max_tokens for query and repair (build_llm OpenAIChat "
                    "gpt_params; LangChain 0.0.294 OpenAIChat sends model + model_kwargs only; "
                    "OfficialSampling send_max_tokens=False); build_functions default binding, "
                    "full bound-args dict compare on the reasoning-stripped reply; per-task "
                    "accuracy and the repo's two published zero-shot domain accuracies (README "
                    "docs/Zero-shot_Evaluation.png: Generic = toolalpaca, toolllm; Cybersecurity "
                    "= cve_cpe, emailrep, virustotal; unweighted task means, matching the "
                    "figure's GPT-4/GPT-3.5 bars); headline = unweighted mean of the two domains "
                    "(leaderboard_comparable=False); verified vs LangChain 0.0.294 on 339x12 "
                    "synthetic replies; a query whose request or OutputFixingParser repair call "
                    "got no reply is a harness error left out of every figure "
                    "(output_fixing.harness_errors); a domain that ran with nothing scored "
                    "withholds the headline")),
    },
    "t_eval": {
        "synced": "2026-10-10",
        "upstream": "github.com/open-compass/T-Eval",
        "revision": "58f22406404d7e2a4f36856a19c7f4dc28a0a5f0",
        "dataset": ("lovesnowbest/T-Eval@af355ab2b62cdbae4262ac41c7529ffeae395012 (8 English v2 "
                    "files and the 8 _zh files)"),
        "method": (("both published settings (EN headline, ZH as overall_zh); origin_prompt sent "
                    "one message per turn (function -> user); greedy temperature 0.0 as test.py "
                    "llm.chat(do_sample=False), max_tokens 512 as test.py line 115 "
                    "HFTransformerCasualLM(max_new_tokens=512) for --model_type hf "
                    "(OfficialSampling.max_tokens: sent exactly without --thinking, not applied "
                    "with it; replies at 512 left out of the truncations in the summed status; "
                    "replies cut off by any other budget counted in replies_cut_off, not "
                    "comparable); test.py split_special_tokens applied to every prediction, then "
                    "the vendored upstream evaluators (equal to the pinned commit apart from "
                    "import shims) per sample on the reasoning-stripped reply; Plan/Reason "
                    "embedder per setting as test.py bert_score_model: "
                    "sentence-transformers/all-mpnet-base-v2@e8c3b32e (EN), "
                    "thenlper/gte-large-zh@64c364e5 (ZH); 6 dimensions combined as "
                    "convert_results; request/health counters summed over the 16 files, status "
                    "recomputed, category_accuracy empty (continuous metrics); "
                    "leaderboard_comparable for a full run at 0.0; a request with no reply is a "
                    "harness error: never passed to the evaluators, left out of its file's "
                    "metrics, named per file in status_reason; a file with nothing scored makes "
                    "its setting's Overall null (accuracy_withheld_reason / "
                    "overall_zh_withheld_reason)")),
    },
    "tau2": {
        "synced": "2026-09-29",
        "upstream": "github.com/sierra-research/tau2-bench",
        "revision": ("b7ea9074c1cba482b30687fecdb5c8425fd6f619 (main 2026-09-17 and still main "
                     "on 2026-09-25, version 1.0.1; data/tau2/domains identical to the v1.0.1 "
                     "tag fc0055dc)"),
        "dataset": ("tau2-bench data/tau2/domains at the revision, base split (retail 114, "
                    "airline 50, telecom 114)"),
        "method": ("runs in the local gbench-tau2 image (docker/tau2.Dockerfile: the checkout at "
                   "the revision installed with uv sync --frozen --extra knowledge from its "
                   "uv.lock, litellm 1.81.11, plus the lock's websockets 15.0.1 that import tau2 "
                   "needs; gbench's tau_common mounted read-only; no Docker, no image or an image "
                   "labelled with another revision -> infra_required with the build command); "
                   "inside it tau2 run_domain on the checkout verified at the revision (clean "
                   "src/ and data/, own DATA_DIR) with every base/knowledge dependency at its "
                   "uv.lock version (else infra_required); leaderboard protocol: every base task, "
                   "4 trials, tau2 compute_metrics pass^k; headline = pass^1 averaged over "
                   "retail/airline/telecom with equal weight (the tau2-bench track Overall in "
                   "web/leaderboard Leaderboard.jsx at the revision), pass^1..pass^4 overall and "
                   "per domain; agent sampled per upstream DEFAULT_LLM_ARGS_AGENT (temperature "
                   "0.0 only), sent once, the endpoint key (a placeholder; a real key is added by "
                   "gbench's endpoint relay) added to each agent call and never stored in "
                   "llm_args (tau2 writes llm_args into results.json, runner/helpers.py); user "
                   "simulator and NL-assertion judge = gbench's one Gemini cascade "
                   "(base.judge_cascade(): GBENCH_JUDGE_MODELS, else GBENCH_JUDGE_MODEL, else "
                   "base.GEMINI_CASCADE, handed to the image as GBENCH_JUDGE_MODELS; "
                   "GBENCH_TAU2_USER_LLM / GBENCH_TAU2_EVAL_LLM name one gemini/ model instead, "
                   "no fallback) in place of gpt-5.2 (recommended by "
                   "docs/leaderboard-submission.md and used by Sierra's 2026 standard text "
                   "entries; config default gpt-4.1-2025-04-14) and upstream's gpt-4.1-2025-04-14 "
                   "judge; every call sends upstream's arguments only (temperature 0.0; LiteLLM "
                   "1.81.11's default thinkingLevel for Gemini 3 models removed, the same request "
                   "for every model), each model gets the call's num_retries (3) on an "
                   "openai.APIError with LiteLLM's own retries off (never re-entering the "
                   "cascade), an HTTP 404 skips the model for the run, the answering model is "
                   "recorded (gemini_models_answered, per role and domain, and each NL verdict's "
                   "reward_info.info.nl.judge_model), empty or malformed replies re-requested up "
                   "to 3 attempts; both must be gemini/ models (a user simulator or judge of any "
                   "other provider, or a gemini/ name tau2's LiteLLM routes elsewhere or cannot "
                   "route -> infra_required naming it); no key enters the image: the Gemini gate "
                   "runs on the host (a missing or refused key, or a cascade model the API does "
                   "not serve -> infra_required), gbench's Gemini relay on the docker bridge "
                   "gateway adds the key to the container's generateContent requests (the image "
                   "holds only its per-run placeholder and refuses a real key), and a real "
                   "endpoint key stays with gbench's endpoint relay there, and every credential "
                   "value is redacted from tool results, agent-bound messages, results.json as "
                   "written, the result and the streamed log; one save directory per run "
                   "(eval/model/thinking/config hash/invocation), never resumed; "
                   "leaderboard_comparable_reason names the setup differences gbench detects; a "
                   "simulation that ends in an infrastructure error for any cause (tau2 "
                   "run_with_retry spent its retries) is never scored: left out of pass^k "
                   "(compute_metrics), of tau2_report.mean_reward and of the evaluable counts "
                   "(total_questions, category_accuracy correct/total; "
                   "tau2_report.total_simulations, evaluable_simulations, infra_errors), status "
                   "completed_with_errors, named in leaderboard_comparable_reason; its source is "
                   "read from the traceback tau2 records: from a harness failure (a Gemini user "
                   "simulator or judge outage, an endpoint outage, a tool or grading failure, "
                   "...) it is a harness error, never the agent's outcome (harness_errors: "
                   "infrastructure_error_simulations, sources, error_types, example_error), and "
                   "raised on the agent's own turn (a reply tau2 rejects: no text and no tool "
                   "calls, or tool-call arguments that are not JSON; a request over the "
                   "endpoint's context window) it is not a harness failure and is listed apart "
                   "(agent_turn_errors), excluded from pass^k as upstream does; a failed domain, "
                   "or one whose every simulation ended in an infrastructure error, hard-errors; "
                   "the container runs on docker's bridge network (never the host network) with "
                   "--add-host host.docker.internal:host-gateway, after the metadata-server guard "
                   "and a TCP check of a loopback endpoint (the agent's api_base becomes "
                   "host.docker.internal); the metadata-server probe container also resolves the "
                   "endpoint's host when it is a name (the proxy's host instead when the "
                   "container sends that request through a proxy), so a bridge network without "
                   "DNS for it is infra_required before any task runs (with a real endpoint key "
                   "only the metadata server is checked); a container of the image must open a "
                   "TCP connection to each of gbench's relays on the gateway (else "
                   "infra_required); the container keeps general internet access as upstream's "
                   "harness has, and the banking_knowledge shell commands run in sandbox-runtime "
                   "with upstream's allowedDomains [] (no network); --shard is validated on the "
                   "host and passed in job.json; each domain's base-split task ids (tau2's load "
                   "order) are cut to the shard before the num_tasks head and passed as "
                   "task_ids; a domain with no task in the shard is skipped and named, and an "
                   "all-empty shard gives accuracy None; per-task evaluable/successful trials "
                   "(tau2_report.task_outcomes_by_domain) feed merge_tau_shards, which computes "
                   "pass^k as tau2's compute_metrics does; num_trials = GBENCH_TAU2_NUM_TRIALS, "
                   "else max(4, --attempt-count) (attempts record); max_output_tokens_note (the "
                   "agent sends no max_tokens); the endpoint key is base.endpoint_api_key, "
                   "always relayed, and the container gets EMPTY; the 'domain' kwarg and the "
                   "legacy GitHub task loaders are removed"
                   "; USER_ERROR simulations (the Gemini user simulator broke the message "
                   "protocol, orchestrator.py _check_communication_error) are harness errors "
                   "(user_simulator_protocol), excluded from pass^k and the rewards"),
    },
    "tau3": {
        "synced": "2026-09-29",
        "upstream": "github.com/sierra-research/tau2-bench (banking_knowledge)",
        "revision": ("b7ea9074c1cba482b30687fecdb5c8425fd6f619 (main 2026-09-17 and still main "
                     "on 2026-09-25, version 1.0.1; data identical to v1.0.1 fc0055dc)"),
        "dataset": "tau2-bench data/tau2/domains/banking_knowledge at the revision (97 tasks)",
        "method": ("runs in the local gbench-tau2 image like tau2 (knowledge extra from uv.lock: "
                   "rank-bm25 0.2.2, openai 2.20.0 not used for retrieval; Anthropic "
                   "sandbox-runtime 0.0.23 with ripgrep, bubblewrap and socat for the shell tool; "
                   "the container options bubblewrap needs, for this domain only); tau2 "
                   "run_domain banking_knowledge on the checkout verified at the revision; "
                   "retrieval = one fixed config, upstream's default alltools (retrieval.py "
                   "DEFAULT_RETRIEVAL_VARIANT: BM25 + dense search + shell; simulation.py default "
                   "for banking_knowledge), with one substitution: the dense search embeds with "
                   "gemini-embedding-001 through the Gemini API batchEmbedContents instead of "
                   "OpenAI text-embedding-3-large (a copy of tau2's alltools variant with only "
                   "the dense embedder changed, registered with tau2; tau2's prompt template, "
                   "retrievers, top_k and embeddings cache unpatched; named in "
                   "leaderboard_comparable_reason, so never leaderboard_comparable); no knob "
                   "selects another config (GBENCH_TAU2_RETRIEVAL_CONFIG / TAU2_RETRIEVAL_CONFIG "
                   "removed: a warning when set, never forwarded into the image); GEMINI_API_KEY "
                   "is an external prerequisite that stays on the host (gbench's Gemini relay "
                   "adds it to the container's batchEmbedContents and generateContent requests; "
                   "the image holds only the relay's placeholder), and tau2's registry must hold "
                   "alltools with its dense embedding pipeline (either missing -> infra_required, "
                   "no fallback embedder or config); the documents are embedded once before the "
                   "run (a failure -> infra_required); a Gemini embedding call that fails after "
                   "its retries during a simulation is a harness error, never the agent's "
                   "outcome: raised out of the tool call so the attempt stops instead of the "
                   "agent reading an Error tool result, re-run by tau2's run_with_retry (a "
                   "simulation whose every attempt fails is an infrastructure error, left out of "
                   "pass^k), counted per domain (harness_errors, "
                   "tau2_report.embedding_failures_by_domain), status completed_with_errors, "
                   "named in leaderboard_comparable_reason; a simulation that ends in an "
                   "infrastructure error for any cause is never scored, as in tau2 (left out of "
                   "pass^k, the mean reward and the evaluable counts, status "
                   "completed_with_errors, named in leaderboard_comparable_reason): from a "
                   "harness failure (failed embedding calls, a Gemini user simulator or judge "
                   "outage, an endpoint outage) it is a harness error as in tau2 (a "
                   "harness_errors entry per domain with its sources and error classes), and "
                   "raised on the agent's own turn it is listed apart (agent_turn_errors); the "
                   "shell tool's commands run with every credential removed from their "
                   "environment, and a probe through tau2's SandboxManager (cat + printenv + the "
                   "sandbox PID 1 environment) must succeed and show no credential before the run "
                   "(else infra_required); 4 trials, headline = banking_knowledge pass^1 (the "
                   "tau3-banking leaderboard track), pass^1..pass^4 reported; agent temperature "
                   "0.0 per upstream, sent once, endpoint key added per call; Gemini user "
                   "simulator and judge as in tau2 (gbench's one cascade from "
                   "GBENCH_JUDGE_MODELS, upstream's call arguments only with LiteLLM's default "
                   "thinkingLevel removed, per-model retries without LiteLLM re-entering the "
                   "cascade, the answering model recorded per role, domain and NL verdict), named "
                   "in the reason; credential values redacted from tool results, agent-bound "
                   "messages, results.json as written, the result and the log; one save directory "
                   "per run, never resumed; the container runs on docker's bridge network (never "
                   "the host network) with --add-host host.docker.internal:host-gateway, after "
                   "the metadata-server guard and a TCP check of a loopback endpoint (the agent's "
                   "api_base becomes host.docker.internal); the metadata-server probe container "
                   "also resolves the endpoint's host when it is a name (the proxy's host instead "
                   "when the container sends that request through a proxy), so a bridge network "
                   "without DNS for it is infra_required before any task runs (with a real "
                   "endpoint key only the metadata server is checked); a container of the image "
                   "must open a TCP connection to each of gbench's relays on the gateway (else "
                   "infra_required); the container keeps general internet access as upstream's "
                   "harness has, and the banking_knowledge shell commands run in sandbox-runtime "
                   "with upstream's allowedDomains [] (no network); --shard is validated on the "
                   "host and passed in job.json; each domain's base-split task ids (tau2's load "
                   "order) are cut to the shard before the num_tasks head and passed as "
                   "task_ids; a domain with no task in the shard is skipped and named, and an "
                   "all-empty shard gives accuracy None; per-task evaluable/successful trials "
                   "(tau2_report.task_outcomes_by_domain) feed merge_tau_shards, which computes "
                   "pass^k as tau2's compute_metrics does; num_trials = GBENCH_TAU2_NUM_TRIALS, "
                   "else max(4, --attempt-count) (attempts record); max_output_tokens_note (the "
                   "agent sends no max_tokens); the endpoint key is base.endpoint_api_key, "
                   "always relayed, and the container gets EMPTY; the 'domain' kwarg and the "
                   "legacy GitHub task loaders are removed"
                   "; USER_ERROR simulations (the Gemini user simulator broke the message "
                   "protocol, orchestrator.py _check_communication_error) are harness errors "
                   "(user_simulator_protocol), excluded from pass^k and the rewards"),
    },

    # Pillar 6: multimodal vision and grounding.
    "chartqa": {
        "synced": "2026-10-09",
        "upstream": ("github.com/EvolvingLMMs-Lab/lmms-eval tasks/chartqa (chartqa.yaml + "
                     "utils.py)@42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8 (relaxed_correctness as "
                     "in google-research/pix2struct pix2struct/metrics.py@6fe25c1d)"),
        "revision": "af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
        "dataset": ("ahmed-masry/ChartQA (test, 2,500: 1,250 human + 1,250 augmented; same "
                    "(question, answer) multiset as lmms-lab-encoder/ChartQA@9e63b7df)"),
        "method": ("lmms-eval chartqa prompt '{question}\\nAnswer the question with a single "
                   "word.'; sampling as lmms-eval's chat client "
                   "(models/chat/openai.py@42296f8f) sends it, under the token-limit policy: "
                   "no-think temperature 0 and max_tokens 16 (generation_kwargs), --thinking "
                   "the same temperature 0 (the client runs a thinking model with "
                   "enable_thinking_kwarg, which keeps the task's generation_kwargs; its "
                   "o1/o3/o4/gpt-5 model-name branch is not used) and the 16 not applied "
                   "(lmms-eval declares no reasoning-model cap: the run's --max-output-tokens, "
                   "else the thinking floor, recorded as not applied to reasoning); replies at "
                   "the cap scored as they arrive (replies_at_official_max_tokens); "
                   "chartqa_process_results relaxed_correctness on the reply as returned (no "
                   "extraction); per-half accuracy; leaderboard_comparable only full, "
                   "unsharded, official temperature, every request answered (FAILED / TIMEOUT / "
                   "OVER_CONTEXT named) and no reply cut off by the run's token budget "
                   "(replies_cut_off_by_budget)"),
    },
    "coco_caption": {
        "synced": "2026-10-09",
        "upstream": ("github.com/EvolvingLMMs-Lab/lmms-eval "
                     "tasks/coco_cap/coco_karpathy_test.yaml + "
                     "utils.py@42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8"),
        "revision": "448fdb1bc7b2d09e46881c4541a14d796a3d41e8",
        "dataset": ("yerevann/coco-karpathy@448fdb1bc7b2d09e46881c4541a14d796a3d41e8 (test, 5,000 "
                    "Karpathy test images + Karpathy reference sentences) + "
                    "lmms-lab/COCO-Caption@c3213b54e173ae86517d481de17f31439d6f8c72 (val images "
                    "matched by file name; byte-identical to the images.cocodataset.org URLs "
                    "lmms-eval fetches, 30/30 sampled)"),
        "checksum": ("sha256:ed6fc0915058cad0fb7b4f6d96659263e04de4f64776f2a54e7b7dc4c22ce5db "
                     "(yerevann/coco-karpathy data/test-00000-of-00001.parquet)"),
        "method": ("lmms-eval coco_karpathy_test prompt 'Describe the image briefly.'; no-think "
                   "temperature 0, top_p 1.0, max_tokens 64 (generation_kwargs as the chat "
                   "client sends them; num_beams/do_sample not sent); --thinking the same "
                   "temperature 0 and top_p 1.0 (enable_thinking_kwarg keeps the task's "
                   "generation_kwargs; the o1/o3/o4/gpt-5 model-name branch is not used) and "
                   "the 64 not applied (lmms-eval declares no reasoning-model cap: the run's "
                   "--max-output-tokens, else the thinking floor, recorded as not applied to "
                   "reasoning); corpus BLEU-1..4, METEOR, ROUGE-L, CIDEr via pycocoevalcap PTB "
                   "tokenizer; headline CIDEr x100; every scorer smoke-tested before "
                   "generation; leaderboard_comparable only full, unsharded, official "
                   "temperature, every request answered and no reply cut off by the run's token "
                   "budget (replies_cut_off_by_budget)"),
    },
    "mmmu_pro": {
        "synced": "2026-10-09.2",
        "upstream": "github.com/MMMU-Benchmark/MMMU mmmu-pro/@268471d0",
        "revision": "563f3e84bb3b90893083a1f039cfa13077f2302b",
        "dataset": "MMMU/MMMU_Pro (standard (10 options) + vision, test, 1,730 each)",
        "method": ("official prompts.yaml CoT + Direct, infer_gpt option format and <image "
                   "n>-ordered images, parse_multi_choice_response (seeded random fallback); score"
                   " = mean(standard-10, vision), each the higher of CoT/Direct; infer_gpt.py "
                   "request_with_images: its system message, text then images, no temperature, "
                   "max_tokens 4096 sent without --thinking (with it the run's "
                   "--max-output-tokens, the cap recorded as not applied to reasoning); comparable"
                   " only full, unsharded, both modes, no operator temperature, every request "
                   "answered and no reply cut off by the run's token budget "
                   "(replies_cut_off_by_budget); accuracy withheld (null with "
                   "accuracy_withheld_reason) when a setting has no score: a setting whose "
                   "requests all got no reply names the harness errors, a GBENCH_SHARD shard that "
                   "holds no request of a setting says so (MMMU-Pro needs the merged shards; the "
                   "shard's per-setting scores stay in mmmu_pro_standard_10 / mmmu_pro_vision)"),
    },
    "screenspot": {
        "synced": "2026-10-09",
        "upstream": ("ScreenSpot-v2 release (OS-Copilot/ScreenSpot-v2) + "
                     "github.com/njucckevin/SeeClick "
                     "pretrain/screenspot_test.py@0ef37ac4d7aaf37ba7b990e3e3c3ca77e1fb8f93 (box "
                     "conversion, click accuracy, generation config)"),
        "revision": "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c",
        "dataset": ("OS-Copilot/ScreenSpot-v2@5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c "
                    "(screenspot_mobile/desktop/web_v2.json, 1,272 targets; "
                    "screenspotv2_image.zip), read through "
                    "screenspot_pointing.screenspot_v2_targets"),
        "checksum": ("sha256:fdd1595b64179e31407eed5929fa12e80adc6ba2ee372a3802d36da379ea1825 "
                     "(mobile), 96f494e0b28eead9187c67794b02bde63350b9cb753acc5fc0384fa77c7182a4 "
                     "(desktop), 3e89e245d69f56a8e8f430917dc02f95b6fc36f8c8f2836c2cafe4dad10e823d "
                     "(web), 750555b8b06e3898634c610cd904600feb7ec7ca530e625a00c1416a73d829ee "
                     "(image zip)"),
        "method": ("gbench mouse_click tool prompt ('Click on the UI element: "
                   "\"<instruction>\".\\nUse the computer tool at your disposal.'); screenshots "
                   "sent as their bytes with the MIME type of those bytes; box [x,y,w,h]/image "
                   "size scaled to [0,1000]; point-in-box click accuracy; headline = accuracy "
                   "over all targets + screenspot_v2_cells; no temperature and no max_tokens "
                   "(the reference script uses the model's own GenerationConfig); a local "
                   "ScreenSpot-Pro dir only when GBENCH_SCREENSPOT_DATA_DIR names one (invalid "
                   "dir hard-errors); leaderboard_comparable=False (prompt)"),
    },
    "textvqa": {
        "synced": "2026-10-09",
        "upstream": ("github.com/EvolvingLMMs-Lab/lmms-eval tasks/textvqa + "
                     "models/chat/openai.py@42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8 "
                     "(EvalAIAnswerProcessor as in facebookresearch/mmf "
                     "mmf/utils/m4c_evaluators.py@b0ffc7fb)"),
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "dataset": "lmms-lab/textvqa (validation, 5,000 x 10 answers)",
        "method": ("lmms-eval textvqa prompt question.capitalize() + '\\nAnswer the question "
                   "using a single word or phrase.'; the task sets no generation_kwargs, so the "
                   "chat client's defaults: no-think temperature 0 and max_tokens 1024, "
                   "--thinking the same temperature 0 (enable_thinking_kwarg keeps the "
                   "sampling; the o1/o3/o4/gpt-5 model-name branch is not used) and the 1024 "
                   "not applied (lmms-eval declares no reasoning-model cap: the run's "
                   "--max-output-tokens, else the thinking floor, recorded as not applied to "
                   "reasoning); textvqa_process_results on the reply as returned (EvalAI "
                   "processor, leave-one-out; no extraction), mean x100; leaderboard_comparable "
                   "only full, unsharded, official temperature, every request answered and no "
                   "reply cut off by the run's token budget (replies_cut_off_by_budget)"),
    },

    # Cross-pillar: economic knowledge work + sandboxed terminal.
    "gdpval": {
        "synced": "2026-10-11",
        "revision": "11e7900cdcac61bc4daf59e65feb238acda98fbf",
        "dataset": ("openai/gdpval@11e7900cdcac61bc4daf59e65feb238acda98fbf (train + "
                    "reference_files + deliverable_files)"),
        "method": ("220 gold tasks, 3 generations per task (arXiv:2510.04374 footnote 2) with no "
                   "temperature or max_tokens sent (the paper sets neither), each judged by one "
                   "position-swapped pairwise comparison vs the expert deliverable rendered to "
                   "text (the paper's 3 human graders per sample replaced by the Gemini cascade, "
                   "named in the reason); headline = GDPval win rate (wins + ties over all "
                   "comparisons, Sec. 3.1), wins-only and tie-0.5 mean secondary; a judge reply "
                   "without a valid winner is a judge-outage exclusion, never a tie; a generation "
                   "whose request failed (FAILED/TIMEOUT/OVER_CONTEXT) is a harness error, "
                   "excluded, counted by status and named; an empty deliverable the model returned"
                   " is a loss; both judge verdicts per comparison with the Gemini model and reply"
                   " of each; no scorable comparison -> failed with the score withheld; "
                   "effective_n = distinct tasks scored; gbench pairwise prompt (OpenAI autograder"
                   " not public); .zip files rendered member by member (text/code, documents via "
                   "the same converters, input images and audio as parts; archives without a "
                   "central directory read from local headers); audio inputs sent as input_audio "
                   "parts; a missing converter package or failed download is infra_required; "
                   "NO_REFERENCE for 39 tasks (35 no deliverable, 4 no text rendering); input PDFs"
                   " without a text layer sent as page images; the rendered reference carries no "
                   "label marking it as the expert's (one file: its text alone; several: neutral "
                   "'--- <file name> ---' separators), as the paper's blind comparisons of "
                   "unlabeled deliverables; clips are recorded per trace (inputs_clipped: an input"
                   " file over GBENCH_GDPVAL_MAX_FILE_CHARS 20,000 characters or "
                   "GBENCH_GDPVAL_MAX_PDF_PAGES 10 pages; judge_input_clipped: a rubric over "
                   "GBENCH_GDPVAL_MAX_RUBRIC_CHARS 8,000 cut at a criterion boundary with a "
                   "visible marker, a reference clipped as a file or over "
                   "GBENCH_GDPVAL_MAX_JUDGE_CHARS 24,000, a model deliverable or task prompt over "
                   "24,000, each with a neutral marker), totalled in gdpval_report with "
                   "clip_limits and named in leaderboard_comparable_reason (the paper describes no"
                   " truncation)"),
    },
    "terminal_bench": {
        "synced": "2026-10-11.2",
        "dataset": "terminal-bench/terminal-bench-2-1",
        "upstream": "Harbor framework (Terminal-Bench 2.1)",
        "revision": ("sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a "
                     "(terminal-bench/terminal-bench-2-1 at the TB2.1 leaderboard's "
                     "DATASET_REF, terminal-bench-2-1@7131e4375048a0e408a8fb404b5f499d726b695b "
                     "leaderboard/src/leaderboard/core/hub.py); harness Harbor 0.20.0 in the "
                     "local image gbench-terminal-bench:harbor-0.20.0 (packages at "
                     "terminal-bench-2-1@7131e43 leaderboard/uv.lock)"),
        "method": ("delegated to Harbor 0.20.0 Docker sandboxes (the TB2.1 leaderboard's lock, "
                   "terminal-bench-2-1@7131e43 leaderboard/uv.lock), Harbor's CLI run in the "
                   "local harness image gbench-terminal-bench:harbor-0.20.0 "
                   "(gbench/docker/terminal_bench.Dockerfile: python:3.12-slim by digest, "
                   "'harbor==0.20.0' and every package of its dependency closure at the lock's "
                   "version from gbench/docker/terminal_bench.constraints.txt, the build "
                   "failing on anything else installed, the Docker CLI 29.1.3 with compose "
                   "2.40.3 and buildx 0.37.1 by sha256; never pulled), its labels "
                   "org.gbench.harbor=0.20.0 and "
                   "org.gbench.terminal_bench_lock=7131e4375048a0e408a8fb404b5f499d726b695b "
                   "checked before every run (a missing or other image is infra_required naming "
                   "the build command), docker-out-of-docker (the docker socket, the jobs "
                   "directory at the same path, gbench's Harbor cache $GBENCH_CACHE_DIR/harbor "
                   "as the container's ~/.cache/harbor and the salvage patch read-only are its "
                   "only mounts; the invoking user, host network, every capability dropped, "
                   "no-new-privileges; only OPENAI_API_KEY=EMPTY, OPENAI_BASE_URL, "
                   "HARBOR_TELEMETRY=0 (Harbor's telemetry off, recorded as harness_telemetry) "
                   "and the other set HARBOR_*/proxy variables of the credential-free "
                   "environment reach it), "
                   "recorded as harbor_version and harness_image (tag, image id, Harbor "
                   "release, lock commit); resolved-rate; 5 trials per task (Harbor "
                   "-k/--n-attempts 5, the TB2.1 leaderboard minimum; an explicit "
                   "--attempt-count above 5 raises it, below keeps 5), accuracy = mean over "
                   "scored trials, attempts with avg@k/pass@k/pass^k; a full default run is "
                   "leaderboard_comparable; --shard / --eval-limit select tasks by the pinned "
                   "dataset's sorted names (shard_select, then stratified_sample seed "
                   "'terminal_bench') passed as --include-task-name, never --n-tasks; trial "
                   "records carry task_name, task_id, task_checksum, sample_key, attempt_idx, "
                   "category/difficulty from task.toml; an empty shard withholds its score; the "
                   "harness container gets no credential and an endpoint key goes through "
                   "gbench's EndpointRelay on 127.0.0.1 (host network); Terminus-2 with the "
                   "leaderboard's kwargs (configs/leaderboard.yaml: temperature 1 for every "
                   "terminus-2 agent, sent in both thinking modes; its reasoning_effort values "
                   "are keyed on hosted models and not applied), its default json parser, no "
                   "max_tokens on any turn (Terminus-2 and LiteLLM send none; the token-limit "
                   "policy's not_sent), --agent-timeout-multiplier 1.0 and no thread-cap env in "
                   "the agent's shell or the verifier (leaderboard/SUBMIT.md: "
                   "timeout_multiplier unset or 1.0, no resource overrides); a temperature "
                   "other than 1 (GBENCH_TERMINAL_BENCH_TEMPERATURE, --temperature), "
                   "GBENCH_TB_PARSER, GBENCH_TB_TURN_MAX_TOKENS, GBENCH_TB_TIMEOUT_MULTIPLIER "
                   "and GBENCH_TB_CONTAINER_THREADS are opt-in deviations, each named in "
                   "leaderboard_comparable_reason (official_sampling, agent_parser, "
                   "turn_max_tokens, agent_timeout_multiplier and container_threads on the "
                   "result); before Harbor starts, the metadata server gate "
                   "(swebench_common.require_task_network_blocked on the default bridge and a "
                   "user-defined network) probes with a cached task image already on the host "
                   "(a --shard run's own tasks first; nothing pulled; infra_required naming "
                   "docker pull otherwise) and checks that the verifiers' package hosts resolve "
                   "(swebench_common.TASK_INSTALL_HOSTS). Harness errors are left out of "
                   "total_questions and accuracy (effective_n = total_questions), counted in "
                   "harness_errors (by kind), harness_error_trials and failed_requests, run "
                   "completed_with_errors; a run of only harness errors is infra_required; a "
                   "trial whose result.json cannot be read (parse) and, in a sharded run, a "
                   "selected task Harbor wrote no trial for (missing) are harness errors too, "
                   "never a 0. One kind is a trial without a reward whose Harbor exception is a "
                   "known harness failure, by the phase its result.json records "
                   "(agent_execution.started_at, verifier.started_at) and the exception type "
                   "(terminal_bench.trial_exception_verdict; harness_error exception, with "
                   "harness_error_reason): any exception before the agent started (environment "
                   "or docker start, healthcheck, agent setup), a cancelled trial, a "
                   "model-endpoint failure (ENDPOINT_EXCEPTIONS: the LiteLLM exception classes "
                   "Terminus-2 re-raises after its retries, an installed agent's "
                   "ApiRateLimitError), an exception naming an unreachable docker daemon, and a "
                   "verifier that started and then failed outside the four verifier exceptions "
                   "(AddTestsDirError, DownloadVerifierDirError, ...) unless docker's message "
                   "shows the task container had stopped or its target path was unusable. The "
                   "model's outcome: AgentTimeoutError and a verifier that started and raised "
                   "VerifierTimeoutError, RewardFileNotFoundError, RewardFileEmptyError or "
                   "VerifierOutputParseError (Harbor's default RetryConfig.exclude_exceptions), "
                   "NonZeroAgentExitCodeError, ContextLengthExceededError and "
                   "OutputLengthExceededError (gbench's own attribution: the model's process, "
                   "context or output ran past a limit), and any other exception raised while "
                   "the agent ran, e.g. the RuntimeError tmux send-keys raises after the "
                   "model's own keystrokes ended its tmux session (exception_note on the trial, "
                   "exceptions_kept_as_model_outcome by type), as Harbor's metrics count every "
                   "trial without a reward as 0 (trial_errors and trial_exceptions count every "
                   "exception). The other is a trial that did not pass whose verifier could not "
                   "resolve an install host before its tests started (verifier/test-stdout.txt, "
                   "no pytest session), when another such trial in the run or a fresh probe "
                   "container after the run that cannot resolve one of the hosts either "
                   "corroborates it (swebench_common.corroborated_dns_failures; harness_error "
                   "dns, dns_failure_evidence). An uncorroborated DNS failure, or a transcript "
                   "naming /etc/resolv.conf, /etc/hosts or /etc/nsswitch.conf, keeps the trial "
                   "the model's outcome (dns_failure_note). The dataset is pinned to the TB2.1 "
                   "leaderboard's DATASET_REF (leaderboard/src/leaderboard/core/hub.py): harbor "
                   "run -d "
                   "terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a, "
                   "and a --shard or --eval-limit run enumerates the same reference through "
                   "Harbor's PackageDatasetClient in the harness image, without the docker "
                   "socket (89 tasks on 2026-09-27, and again through the image on 2026-09-28); "
                   "the result records harbor_version, dataset and dataset_ref (the result "
                   "parsing and the salvage patch follow Harbor 0.20.0); a run without "
                   "--shard/--eval-limit expects FULL_TASK_COUNT (89) x k trials and counts the "
                   "ones without a result as harness_errors missing (completed_with_errors, "
                   "partial, not comparable; expected_trials recorded); a non-zero Harbor exit "
                   "(harbor_returncode; Harbor 0.20.0 exits 0 after a completed job) is "
                   "completed_with_errors and not comparable; status_reason names every harness "
                   "error; setting_deviations lists the non-harness reasons; merge_shards "
                   "re-counts the union of the shards' trials"
                   "; harbor run --max-retries 3 (the TB2.1 leaderboard's retry.max_retries, "
                   "configs/leaderboard.yaml at 7131e43; Harbor 0.20.0 RetryConfig default "
                   "exclusions), so a trial that raised a retried exception is rerun and its last"
                   " attempt scored; result trial_max_retries; the harness container mounts the "
                   "daemon socket the docker CLI uses (DOCKER_HOST / docker context; a non-unix "
                   "endpoint is infra_required)"),
    },
    "swe_lancer": {
        "synced": "2026-10-09",
        "upstream": ("github.com/openai/frontier-evals project/swelancer (SWE-Lancer's current "
                     "home; openai/SWELancer-Benchmark is archived and points there)"),
        "revision": "51052cede8cc608f95bb00346635e03759013e5a",
        "dataset": ("project/swelancer/all_swelancer_tasks.csv at the revision (sha256 "
                    "5c3a6d4570b49be0d9fced98f5b32487420b16f25c98d6658830e31fa03f049a; 463 "
                    "Diamond tasks: 198 ic_swe, $189,300; 265 swe_manager, $264,500) and the task "
                    "images SWE-Lancer publishes "
                    "(swelancer/swelancer_x86_<question_id>:releasev1; "
                    "swelancer/swelancer_x86_monolith:releasev1 for swe_manager)"),
        "method": ("registered, not in --evals all (runs when named: the harness edits the Docker "
                   "host's DOCKER-USER iptables rules while it runs and pulls about 1.3 TB of "
                   "task images); both task types run by default, each with upstream README's run "
                   "command (split=diamond, task_type, SimpleAgentSolver with model "
                   "openai/<served model>, AlcatrazComputerRuntime with "
                   "LocalConfig(pull_from_registry=True), docker_image_prefix "
                   "swelancer/swelancer_x86, docker_image_tag releasev1, use_single_image for "
                   "swe_manager, runner concurrency from --sandboxes, "
                   "experimental_use_multiprocessing=False, enable_slackbot=False, "
                   "dummy_recorder, max_retries=2; upstream defaults n_test_runs 3 and "
                   "disable_internet True) through run_swelancer.py's own main in the local image "
                   "gbench-swe-lancer-harness:51052ced (ubuntu:24.04 by digest, apt from the Ubuntu "
                   "archive snapshot 20260925T120000Z, uv 0.12.12 sync --frozen of the project's "
                   "uv.lock, task table sha256 checked, label org.gbench.frontier_evals), "
                   "docker-out-of-docker on the host network with NET_ADMIN; gbench adds only a "
                   "per-task report (a SWELancerEval subclass whose get_full_summary calls "
                   "upstream's, gbench/docker/swe_lancer_eval.py); headlines per task type: "
                   "pass@1 (solved / tasks with a verdict) and dollars earned, plus Diamond "
                   "totals; no temperature and no max_tokens sent (SimpleAgentSolver sends "
                   "neither; an explicit temperature is recorded and not comparable); the model "
                   "is reached through gbench's EndpointRelay, which forwards only "
                   "/v1/chat/completions, /v1/completions and /v1/models, adds the endpoint key "
                   "and sets chat_template_kwargs.enable_thinking (the harness gets "
                   "OPENAI_API_KEY=EMPTY); a task whose rollout ended in a system error "
                   "(RolloutSystemError) got no verdict: a harness error, left out of pass@1 "
                   "(numerator and denominator; scored_tasks and effective_n are the denominator) "
                   "and of available_usd, counted in harness_errors, the task type "
                   "completed_with_errors (partial), and its pass@1 withheld (null, status error) "
                   "when no task of the type got a verdict; comparable only for a clean run of "
                   "both full task types; hard-errors infra_required without Docker, the docker "
                   "socket, the image or iptables access to the host's DOCKER-USER chain"),
    },
    "ui_control_osworld": {
        "synced": "2026-10-09.2",
        "upstream": "github.com/xlang-ai/OSWorld",
        "revision": "fc31a9049664292fcb35d6e501ee1dc839f2cf6d",
        "method": ("registered, not in --evals all (runs when named); delegated to the pinned "
                   "OSWorld run.py in a local orchestrator image "
                   "(gbench/docker/ui_control_osworld.Dockerfile), docker provider with sibling "
                   "QEMU-VM task containers (docker-out-of-docker), mm_agents PromptAgent at "
                   "the harness's own sampling defaults (max_tokens 1500, no reasoning field), "
                   "tasks sequential on one VM; the orchestrator (docker socket and host "
                   "network, as OSWorld's docker provider needs) gets the placeholder "
                   "GBENCH_MODEL_API_KEY=dummy, and a real key (GBENCH_OSWORLD_MODEL_API_KEY) "
                   "stays with gbench's EndpointRelay on 127.0.0.1, which forwards only "
                   "/v1/chat/completions, /v1/completions and /v1/models to the endpoint and "
                   "adds the key (model_endpoint_relay; the endpoint failures it counts make "
                   "the run completed_with_errors); accuracy = mean per-task result.txt over "
                   "the selected tasks that produced one (as OSWorld run.py get_result averages "
                   "the result files present); a task without a result.txt is a harness error, "
                   "left out (n_missing, scoring_excluded, status_reason; "
                   "success_rate_missing_as_zero keeps the raw rate), per-domain breakdown; no "
                   "LLM judge; leaderboard_comparable=False (self-hosted agent, single trial); "
                   "hard-errors infra_required without Docker, the orchestrator image, "
                   "/dev/kvm, the Ubuntu qcow2 or the task VM image at the pinned digest "
                   "happysixd/osworld-docker@sha256:"
                   "0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9 (pulled by "
                   "digest and tagged with the name OSWorld's docker provider starts; recorded "
                   "as task_image); every run writes to a "
                   "fresh GBENCH_OSWORLD_VM_DIR/gbench_runs/<run id> (run_dir), because OSWorld "
                   "run.py get_unfinished/get_result resume from an existing result_dir; the "
                   "orchestrator image must carry label "
                   "org.gbench.ui-control-osworld-launcher=4, else infra_required; --shard is "
                   "validated on the host, forwarded normalized, reported back by the launcher "
                   "and recorded (record_shard, SHARD_REASON); --eval-limit is a seeded draw "
                   "balanced across the domains (stratified_indices, seed ui_control_osworld; "
                   "selection.limit_draw required); an empty shard withholds its score; "
                   "per-task sample_traces; merge_shards re-scores the union"
                   "; a task with a model request that failed on the endpoint's side (no reply, "
                   "or HTTP 5xx/429/401/403/404, recorded per example by the launcher) is a "
                   "harness error left out of the headline and the per-domain rates "
                   "(n_endpoint_failed); the orchestrator always reaches the endpoint through the"
                   " EndpointRelay; GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE other than "
                   "happysixd/osworld-docker and a GBENCH_OSWORLD_LIMIT that is not a positive "
                   "integer are infra_required before any container starts; merge_shards keeps "
                   "each shard's draw as selection_by_shard"),
    },
    "mobilegym": {
        "synced": "2026-10-22",
        "upstream": "github.com/Purewhiter/mobilegym",
        "revision": "5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38",
        "checksum": "sha256:933a8499c71450f135eda70f9107af72b56b42181fc62de76ed1aa4c513a603c",
        "dataset": "mobilegym-data-v0.1.0 (bench_env/splits/test.txt, 256 tasks)",
        "method": ("delegated to Purewhiter/mobilegym@5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38 "
                   "inside local image gbench-mobilegym:5fbd3f34 "
                   "(gbench/docker/mobilegym.Dockerfile, label org.gbench.mobilegym-launcher=2); "
                   "256-task held-out test split (split='test', sample_seed=42) driven by "
                   "canonical GenericAgentV2 (coord_space='norm_0_1000', eval_mode='grounded', "
                   "judge_mode='state', deterministic programmatic state verifiers, zero LLM/VLM "
                   "judge calls); Vite simulator on 127.0.0.1:3000 with frozen simulator clock "
                   "2026-07-12 09:00:00 and matching Python default_tomorrow() (2026-07-13); "
                   "cloud metadata server blocked (container_guard.require_metadata_blocked); "
                   "model calls routed through EndpointRelay on the bridge gateway and token-id "
                   "recovery through DetokenizeRelay (server_parser_policy); modulo sharding "
                   "before seeded difficulty-stratified --eval-limit "
                   "(stratified_indices(key=difficulty, seed=mobilegym)); merge_shards re-scores "
                   "the union of sample_traces"),
    },
    "mobilegym_pp": {
        "synced": "2026-10-22",
        "upstream": "github.com/OpenMobile2/MobileGym-plusplus",
        "revision": "cdde4ad06ed67176b56964dd329b8cb8e0b0001c",
        "checksum": "sha256:933a8499c71450f135eda70f9107af72b56b42181fc62de76ed1aa4c513a603c",
        "dataset": "bench215/official.json (215 tasks) + mobilegym-data-v0.1.0",
        "method": ("delegated to OpenMobile2/MobileGym-plusplus@"
                   "cdde4ad06ed67176b56964dd329b8cb8e0b0001c inside local image "
                   "gbench-mobilegym:5fbd3f34 (gbench/docker/mobilegym.Dockerfile, label "
                   "org.gbench.mobilegym-launcher=2); 215-task official benchmark "
                   "(bench215/official.json, audited instructions applied via task.templates "
                   "preserving FrozenTimeSetupMixin._post_sample) driven by canonical "
                   "Qwen35SessionAgent (interaction_mode='hybrid', coord_space='norm_0_1000', "
                   "deterministic programmatic state verifiers, zero LLM/VLM judge calls); "
                   "Gemma models routed to the XML/user-turn session family so "
                   "tool_chat_template_gemma4.jinja preserves multi-turn observations and "
                   "screenshots; Vite simulator on 127.0.0.1:3000 with frozen simulator clock "
                   "2026-07-12 09:00:00; cloud metadata server blocked; EndpointRelay and "
                   "DetokenizeRelay on the bridge gateway; modulo sharding before seeded "
                   "difficulty-stratified --eval-limit "
                   "(stratified_indices(key=difficulty, seed=mobilegym_pp)); merge_shards "
                   "re-scores the union of sample_traces"),
    },
}


def canonical_sync_for(eval_name: str) -> Dict[str, Any]:
    """Provenance record for a suite, or an honest 'unverified' marker if it has no entry yet.

    A reconciled record carries the suite's recorded code fingerprint as `code_fp` (CODE_FP), so a
    result names the code state its registry entry describes."""
    entry = CANONICAL_SYNC.get(eval_name)
    if entry is None:
        return {"status": "unverified",
                "note": "not yet formally reconciled against upstream (see canonical_sync.py)"}
    out = {"status": "reconciled", **entry}
    if CODE_FP.get(eval_name):
        out["code_fp"] = CODE_FP[eval_name]
    return out


# --------------------------------------------------------------------------- #
# Structured fields
# --------------------------------------------------------------------------- #
#: The fields that anchor an entry to what was reconciled. They are what the evals scaffold_id
#: folds in (scoring_signal) and what the review gate requires to move with a fingerprint; the
#: prose fields (upstream, method, note) are not, so a documentation edit breaks no series.
ANCHOR_FIELDS: Tuple[str, ...] = ("synced", "revision", "checksum", "dataset")

#: `synced`: the reconcile date, with a same-day serial when an entry is re-reconciled on the
#: day of its last bump (a date alone could not move, and neither could the scaffold_id).
SYNCED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:\.[1-9]\d*)?$")

#: One pin: a 40- or 64-hex commit or digest (optionally sha256:), a release version (1.2.0,
#: v2.4.6, 2026.3.23) or package==version.
PIN_TOKEN_RE = re.compile(
    r"^(?:sha256:)?(?:[0-9a-f]{40}|[0-9a-f]{64}|v?\d+(?:\.\d+)+[0-9A-Za-z.+-]*"
    r"|[A-Za-z0-9_.-]+==\d[0-9A-Za-z.+-]*)$")
#: A content checksum: sha256 of the bytes, or the git blob / tree id.
CHECKSUM_RE = re.compile(r"^(?:sha256:[0-9a-f]{64}|git-blob:[0-9a-f]{40}|git-tree:[0-9a-f]{40})$")

#: Entries that cannot carry a pin, and why. Every other entry must be entry_pinned().
DATE_ONLY_EXEMPT: Dict[str, str] = {
    "custom_jsonl": ("the operator's own file: the evals scaffold pins its sha256 "
                     "(dataset_snapshot.custom_jsonl_sha256) instead"),
}


def _leading_token(text: Any) -> str:
    """The first whitespace-separated token of a field, trailing ',' / ';' removed."""
    parts = str(text or "").strip().split()
    return parts[0].rstrip(",;") if parts else ""


def entry_pinned(entry: Dict[str, Any]) -> bool:
    """Whether an entry is version-pinned: its `checksum` starts with a CHECKSUM_RE checksum, or
    its `revision` starts with a PIN_TOKEN_RE pin. Anything else (prose, a date) is date-only."""
    if CHECKSUM_RE.match(_leading_token(entry.get("checksum"))):
        return True
    return bool(PIN_TOKEN_RE.match(_leading_token(entry.get("revision"))))


#: A "<package> <version>" part of a revision (mini-swe-agent 2.4.2).
PACKAGE_VERSION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]* v?\d+(?:\.\d+)+[0-9A-Za-z.+-]*$")


def malformed_anchor_fields(entry: Dict[str, Any]) -> List[str]:
    """The structured fields of an entry that do not lead with their pin: a `revision` whose
    ", "-separated parts (before any parenthesized qualifier) do not each lead with a
    PIN_TOKEN_RE pin or read "<package> <version>", or a `checksum` whose leading token is not a
    CHECKSUM_RE checksum."""
    bad: List[str] = []
    revision = entry.get("revision")
    if revision:
        head = str(revision).split(" (", 1)[0]
        if not all(PIN_TOKEN_RE.match(_leading_token(part)) or PACKAGE_VERSION_RE.match(part.strip())
                   for part in head.split(", ")):
            bad.append("revision")
    checksum = entry.get("checksum")
    if checksum and not CHECKSUM_RE.match(_leading_token(checksum)):
        bad.append("checksum")
    return bad


def anchors(entry: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """The ANCHOR_FIELDS an entry sets, as {field: value}."""
    return {f: str(entry[f]) for f in ANCHOR_FIELDS if entry and entry.get(f)}


# --------------------------------------------------------------------------- #
# Shared modules
# --------------------------------------------------------------------------- #
#: The eval_suites modules several suites import, each with its own entry: an edit to one is one
#: decision here rather than one per importing suite, and every suite that imports it (directly or
#: through another module) folds this entry's anchors into its scaffold signal (scoring_signal).
#: `synced` is the date the module's current verdict-relevant behaviour was declared; bump it when
#: an edit can change a verdict of any suite that imports it.
SHARED_SYNC: Dict[str, Dict[str, str]] = {
    "base": {"synced": "2026-10-22",
             "method": ("the eval engine every native suite runs through (run_eval_suite): "
                        "requests, sampling and token budgets, answer handling, the Gemini judge "
                        "cascade, harness-error accounting, attempts and shards; code "
                        "fingerprint baseline recorded 2026-09-28"
                        "; 2026-09-29: pass^k over the samples with every attempt scored "
                        "(pass_hat_k_samples); vLLM's exact-count and character-count context "
                        "400s retried with a lower max_tokens; a reply at a max_tokens a retry "
                        "lowered is not counted at the official cap; a refused return_token_ids "
                        "field is retried without it; the sample-meta identity keys are not sent "
                        "as request fields (SAMPLE_META_BOOKKEEPING_KEYS)")},
    "limit_plan": {"synced": "2026-10-09.2",
                   "method": ("the --eval-limit plan every suite logs at its start and records as "
                              "eval_limit_plan (samples, per-setting counts, attempts per sample, "
                              "generations); --eval-limit N is N samples per suite, dealt over a "
                              "suite's settings by sampling.allocate_limit")},
    "sampling": {"synced": "2026-10-09.2",
                 "method": ("--eval-limit subsetting and shard selection (allocate_limit deals "
                            "a suite's N samples over its settings, a headline setting first); "
                            "code fingerprint baseline recorded 2026-09-28"
                            "; presharded_for_base passes rows already sharded by a suite's own "
                            "loader to run_eval_suite unchanged")},
    "swebench_common": {"synced": "2026-10-11",
                        "method": ("the SWE-bench execution harness helpers (patch scoring, "
                                   "EndpointRelay, infra_required); code fingerprint baseline "
                                   "recorded 2026-09-28"
                                   "; run_eval_suite applies --shard, then the --eval-limit draw "
                                   "with limit_seed 'swebench_common' (the same instances as "
                                   "before when unsharded); the scorer does not start the "
                                   "harness when no trace names an instance (harness_no_report); "
                                   "EndpointRelay tags endpoint-side failures with the request's "
                                   "conversation fingerprint and records successes; a by-name "
                                   "(unpinned) dataset load warns and adds "
                                   "UNPINNED_DATASET_REASON, and every built-in caller pins a "
                                   "revision; EndpointRelay watches the client connection while "
                                   "it waits for, reads and sends a reply, and when the client "
                                   "leaves (end of file or reset) closes the endpoint's "
                                   "connection of that request, so the endpoint stops generating "
                                   "an abandoned (e.g. SDK-timed-out, resent) non-streamed reply;"
                                   " counted in client_closed_streams"
                                   "; the relay splits replies empty after reasoning that "
                                   "stopped on their own by their token ids "
                                   "(classify_empty_reply; the tool-call token read once from the "
                                   "served /tokenize, Gemma 4 <|tool_call> id 48): "
                                   "replies_answer_in_thinking (the model's outcome), "
                                   "replies_tool_call_dropped (a native tool call the server's "
                                   "parser removed) and replies_empty_unclassified (no token "
                                   "ids); relay_result_reasons makes the last two comparability "
                                   "reasons (tool_call_dropped_turns, unclassified_empty_turns); "
                                   "the relay reads GET /v1/models (owned_by vllm) before the "
                                   "first request so it carries return_token_ids too; "
                                   "upstream_reask_relay re-asks a reply matching a suite's "
                                   "upstream rule (reply_content_is_null, "
                                   "reply_content_is_null_or_empty) up to a limit"
                                   "; 2026-10-10.3: a re-ask rule matches only a reply the model "
                                   "ended itself (finish_reason stop; a reply cut off at the "
                                   "budget is counted in cut_off_not_reasked and never re-asked); "
                                   "EndpointRelay.reask_deadline keeps a re-ask chain inside the "
                                   "client's per-request timeout (base.request_timeout_s of the "
                                   "request's max_tokens, x0.9 less 15 s; each re-ask sent with "
                                   "the time left as its timeout), and a chain that is exhausted, "
                                   "cut short at the deadline or ended by a re-ask the endpoint "
                                   "did not answer passes the last reply on (reasks_exhausted, "
                                   "reasks_cut_short), never a 502; reask_summary records the "
                                   "relay's empty-reply classes and answer_in_thinking_rate_seen; "
                                   "reask_divergence_reasons and merge_reask_summaries are shared")},
    "metrics": {"synced": "2026-09-28",
                "method": ("the trace-walk finalizers for non-binary headline metrics; code "
                           "fingerprint baseline recorded 2026-09-28")},
    "vqa_common": {"synced": "2026-09-28",
                   "method": ("the visual-question-answering metrics (ANLS, relaxed accuracy, VQA "
                              "accuracy); code fingerprint baseline recorded 2026-09-28")},
    "container_guard": {"synced": "2026-09-28",
                        "method": ("the metadata-server guard for containers that run "
                                   "model-controlled code; code fingerprint baseline recorded "
                                   "2026-09-28")},
    "tau_common": {"synced": "2026-10-13",
                   "method": ("the tau2-bench environment harness shared by tau2 and tau3; code "
                              "fingerprint baseline recorded 2026-09-28; --eval-limit per "
                              "domain is a seeded draw (sampling.stratified_sample, seed = "
                              "suite name) passed to tau2 as task_ids (never num_tasks); "
                              "selected_task_ids recorded"
                              "; a USER_ERROR simulation is a harness error "
                              "(user_simulator_protocol) left out of pass^k and the rewards; "
                              "merge_tau_shards unites selected_task_ids and sums "
                              "user_simulator_protocol_errors")},
    "dataset_utils": {"synced": "2026-09-28",
                      "method": ("dataset download and caching helpers; code fingerprint baseline "
                                 "recorded 2026-09-28")},
    "sandbox": {"synced": "2026-09-28",
                "method": ("process isolation for suites that execute model-written code; code "
                           "fingerprint baseline recorded 2026-09-28")},
    "swe_agentic": {"synced": "2026-10-22",
                    "method": ("the agentic SWE rollout protocol shared by the SWE suites; code "
                               "fingerprint baseline recorded 2026-09-28"
                               "; mark_empty_shard gives an empty shard accuracy None"
                               "; the empty turns are classified from the 2.x relay's token ids "
                               "(answer_in_thinking over the relay's completion_replies; "
                               "tool_call_dropped_turns and unclassified_empty_turns, incl. "
                               "trajectory empty turns beyond the relay's, are comparability "
                               "reasons); the 1.x runner has no relay: every empty turn is "
                               "unclassified")},
    "swe_thread_cap": {"synced": "2026-10-10",
                       "method": ("the OpenMP/BLAS thread cap inside SWE execution containers; "
                                  "code fingerprint baseline recorded 2026-09-28; 2026-10-10: "
                                  "SUITE_CONTAINERS (contextvar) shares the box over a whole "
                                  "suite whose parts run at once (swe_bench_live tab lanes), "
                                  "unset everywhere else")},
    "search_tool": {"synced": "2026-10-12",
                    "method": ("the web_search tool backed by Gemini grounding; code fingerprint "
                               "baseline recorded 2026-09-28")},
    "gemini_relay": {"synced": "2026-09-28",
                     "method": ("the host-side Gemini relay for containers that run "
                                "model-controlled code; code fingerprint baseline recorded "
                                "2026-09-28")},
    "fc_common": {"synced": "2026-10-09.2",
                  "method": ("structural parsing and matching for the function-calling suites; "
                             "code fingerprint baseline recorded 2026-09-28")},
    "matharena_common": {"synced": "2026-10-07",
                         "method": ("MathArena's final-answer protocol shared by aime and hmmt; "
                                    "code fingerprint baseline recorded 2026-09-28")},
    "extraction_common": {"synced": "2026-09-28",
                          "method": ("shared answer-extraction helpers; code fingerprint baseline "
                                     "recorded 2026-09-28")},
    "gemini_budget": {"synced": "2026-10-10.2",
                      "method": ("the output limit of a Gemini model in a role written for a "
                                 "non-thinking one (simple-evals' grader, ToolEval's judge, "
                                 "wildbench's judge, ACEBench's user simulator; copies in "
                                 "wildclawbench's proxy and ToolBench's simulator patch): the "
                                 "upstream visible max_tokens plus GEMINI_THINKING_ALLOWANCE "
                                 "8192, replies still cut there counted, summed by a "
                                 "sharded run's merge (merge_with_cutoff_reason) and named "
                                 "once over the union; added 2026-10-10")},
}

#: Modules the fingerprint never reads or follows: maintainer-owned plugin code in this package.
#: Its changes are the maintainer's to version; the suites that call into it are fingerprinted
#: without it.
UNFINGERPRINTED_MODULES = frozenset({"swe_bench_pro_agentic"})

#: Registry infrastructure, not scoring code.
_INFRA_MODULES = frozenset({"__init__", "canonical_sync", "loader"})

#: A declared verdict-preserving change: {name: {"code_fp": <the fingerprint the change produced>,
#: "reason": <why no verdict can change>}}. The review gate accepts a fingerprint that moved without
#: an anchor change only when it is declared here with exactly that fingerprint.
VERDICT_PRESERVING: Dict[str, Dict[str, str]] = {
    'acebench': {"code_fp": '8877be3eaf780809',
        "reason": 'packaging path move only: moved to gbench/third_party/eval_suites with LICENSE; no request or score changes'},
    'beam_128k': {"code_fp": 'd81703aef5ab872a',
        "reason": 'packaging path move only: _json_repair_vendor moved verbatim to gbench/third_party/_json_repair_vendor; no request or score changes'},
    'bfcl_v4_agentic': {"code_fp": '971ad5c13f75e7cc',
        "reason": 'packaging path move only: moved to gbench/third_party/eval_suites with LICENSE; no request or score changes'},
    'causalbench': {"code_fp": '3a04d172e7075640',
        "reason": 'reporting only: headline_metrics and the category rows; no request or score changes'},
    'complexfuncbench': {"code_fp": '535bdbe7036a4b63',
        "reason": 'packaging path move only: moved to gbench/third_party/eval_suites with LICENSE; no request or score changes'},
    'ifeval': {"code_fp": '36baabf70a59be06',
        "reason": 'packaging path move only: moved to gbench/third_party/eval_suites with LICENSE; no request or score changes'},
    'matharena_common': {"code_fp": '7f2cc975019e7777',
        "reason": 'the grading driver removes its temporary directory when it ends, and matharena_src moved verbatim to gbench/third_party/matharena_src; no request, grading or score changes'},
    'metrics': {"code_fp": '4a278892c8f9978f',
        "reason": 'finalize_mean_metric gains the opt-in category_scores (default off) and the replace_category_headline helper; the suites that opt in change their category rows and counts only, and no headline changes'},
    'scicode': {"code_fp": '273f15c37bf7b32d',
        "reason": 'packaging path move only: moved to gbench/third_party/eval_suites with LICENSE; no request or score changes'},
    'spider2': {"code_fp": 'ecb94de7e567ce47',
        "reason": "reporting only: leaderboard_comparable_reason names the --eval-limit subset and replies cut off by the run's token budget (record_budget_cutoffs); no request or score changes"},
    'swe_agentic': {"code_fp": '199c496c59f6ce3b',
        "reason": "run_agentic words a cut-off turn as cut at the run's max_tokens when it sent one; no request or score changes"},
    'swe_bench_live': {"code_fp": 'af477cc386ceecce',
        "reason": 'run_tab_schedule submits the tab lanes inside its stop handling (a signal arriving during submission ends in TabScheduleStopped naming the tabs, not a bare SystemExit); a stopped suite has no score either way, no request, grading or score changes'},
    'swebench_common': {"code_fp": 'c1f99e628860711b',
        "reason": "every started EndpointRelay is registered so the eval runner can attach its record to a suite that raised (relays_started_since), and relay_result_reasons words a cut-off turn as cut at the run's max_tokens when the relay set it; no request or score changes"},
    't_eval': {"code_fp": '1bd22e79189c50ba',
        "reason": 'packaging path move only: moved to gbench/third_party/eval_suites with LICENSE; no request or score changes'},
    'tau2': {"code_fp": '4882d354d4342707',
        "reason": "base image pin only: docker/tau2.Dockerfile's FROM python:3.12-slim is pinned by digest (sha256:78387bc3881b), the base the built gbench-tau2:b7ea9074 image already sits on (its layers start with that base's); nothing else in the build, and no request, harness step or score, changes"},
    'tau3': {"code_fp": '0f5c202d0cecd0f3',
        "reason": "base image pin only: docker/tau2.Dockerfile's FROM python:3.12-slim is pinned by digest (sha256:78387bc3881b), the base the built gbench-tau2:b7ea9074 image already sits on (its layers start with that base's); nothing else in the build, and no request, harness step or score, changes"},
    'tau_common': {"code_fp": '84398b7dfe65ade8',
        "reason": 'docker/tau2_srt/package-lock.json in build context; no request, harness step or score changes'},
    'wildclawbench': {"code_fp": '4d52c606065d1020',
        "reason": 'record workdir JSONL files before flushing final HTTP response bytes in proxy handler, and packaging path move to gbench/third_party/eval_suites with LICENSE; no request or score changes'},
}


def registry_entry(name: str) -> Optional[Dict[str, Any]]:
    """The CANONICAL_SYNC entry of a suite, else the SHARED_SYNC entry of a shared module."""
    return CANONICAL_SYNC.get(name) or SHARED_SYNC.get(name)


def scoring_signal(names: Iterable[str]) -> Dict[str, Any]:
    """What the evals scaffold_id folds in for `names` (the suites a run executes):
    {"suites": {name: anchors}, "depends_on": {module: anchors}}, where depends_on holds every
    suite and shared module those suites import (DEPENDS_ON, transitively) and a name with no
    entry maps to {"status": "unverified"}. Declared values only: the code fingerprint is not an
    input, so the id moves when someone records a verdict-relevant change, and the gate
    (stale_code_fingerprints / bump_violations) makes sure they do."""
    suites: Dict[str, Any] = {}
    deps: Set[str] = set()
    for name in sorted(set(names)):
        entry = CANONICAL_SYNC.get(name)
        suites[name] = anchors(entry) if entry is not None else {"status": "unverified"}
        deps.update(DEPENDS_ON.get(name, ()))
    return {"suites": suites,
            "depends_on": {d: anchors(registry_entry(d)) for d in sorted(deps)}}


# --------------------------------------------------------------------------- #
# Code fingerprints
# --------------------------------------------------------------------------- #
_PKG_DIR = Path(__file__).resolve().parent
#: The Docker build contexts suites build from: gbench/docker and the repository's docker/.
DOCKER_DIRS: Tuple[Path, ...] = (_PKG_DIR.parents[1] / "docker", _PKG_DIR.parents[2] / "docker")
_ROOT = _PKG_DIR.parents[2]


def _skipped(path: Path) -> bool:
    """Backups, caches and hidden files are not code."""
    return (any(part == "__pycache__" or part.startswith(".") for part in path.parts[-3:])
            or ".bak" in path.name or path.suffix in (".pyc", ".pyo", ".orig", ".rej"))


def _norm_ast(node: Any) -> Any:
    """A version-stable, docstring-free form of an AST: node type and non-empty fields, without
    positions, contexts or docstrings, adjacent f-string text parts merged (the Python versions
    split them differently)."""
    if isinstance(node, ast.AST):
        fields = []
        for name, value in ast.iter_fields(node):
            if name in ("ctx", "type_comment", "kind"):
                continue
            if (name == "body" and isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                                       ast.AsyncFunctionDef))
                    and value and isinstance(value[0], ast.Expr)
                    and isinstance(value[0].value, ast.Constant)
                    and isinstance(value[0].value.value, str)):
                value = value[1:]
            if isinstance(node, ast.JoinedStr) and name == "values":
                merged: List[Any] = []
                for part in value:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        if not part.value:
                            continue
                        if merged and isinstance(merged[-1], str):
                            merged[-1] += part.value
                            continue
                        merged.append(part.value)
                    else:
                        merged.append(part)
                value = [m if not isinstance(m, str) else ast.Constant(value=m) for m in merged]
            if value is None or value == []:
                continue
            fields.append([name, _norm_ast(value)])
        return [type(node).__name__, fields]
    if isinstance(node, list):
        return [_norm_ast(v) for v in node]
    return repr(node)


def _parse_source(data: bytes, filename: str) -> Optional[ast.Module]:
    """The AST of Python source, or None when it does not parse (then its bytes are hashed).
    Syntax warnings of vendored code are not the gate's business and are silenced."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(data.decode("utf-8"), filename=filename)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None


def _is_requirement_list(path: Path) -> bool:
    """A pip requirement, constraint or pin list: a .txt named requirements*, *constraints* or
    *.pins, or any .txt in a Docker build context (DOCKER_DIRS)."""
    if path.suffix != ".txt":
        return False
    name = path.name.lower()
    if name.startswith("requirements") or "constraints" in name or name.endswith(".pins.txt"):
        return True
    try:
        resolved = path.resolve()
        return any(resolved.is_relative_to(d.resolve()) for d in DOCKER_DIRS)
    except (OSError, ValueError):
        return False


def _file_digest(path: Path) -> str:
    """sha256 of a file's meaningful content: a Python file's normalized AST (comments,
    formatting and docstrings do not count); a Dockerfile, shell script or requirement /
    constraint / pin list (_is_requirement_list) without its comment and blank lines; anything
    else byte for byte (a data .txt, e.g. prompt content, whose '#' lines are content)."""
    data = path.read_bytes()
    name = path.name
    if path.suffix == ".py":
        tree = _parse_source(data, name)
        payload = (data if tree is None
                   else json.dumps(_norm_ast(tree), separators=(",", ":")).encode("utf-8"))
    elif "Dockerfile" in name or path.suffix == ".sh" or _is_requirement_list(path):
        lines = [ln.rstrip() for ln in data.decode("utf-8", "replace").splitlines()]
        payload = "\n".join(ln for ln in lines
                            if ln.strip() and not ln.lstrip().startswith("#")).encode("utf-8")
    else:
        payload = data
    return hashlib.sha256(payload).hexdigest()


def _tree_files(path: Path) -> List[Path]:
    """A file, or every file under a directory (skipping backups and caches), sorted."""
    if path.is_file():
        return [] if _skipped(path) else [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and not _skipped(p))


class _Package:
    """The eval_suites package as the fingerprint sees it: its modules, its data directories and
    each module's intra-package imports and string constants. `pkg_dir` and `docker_dirs` are
    parameters so a test can point it at a scratch tree."""

    def __init__(self, pkg_dir: Optional[Path] = None,
                 docker_dirs: Optional[Iterable[Path]] = None,
                 root: Optional[Path] = None, suites: Optional[Iterable[str]] = None,
                 shared: Optional[Iterable[str]] = None):
        self.pkg_dir = Path(pkg_dir) if pkg_dir else _PKG_DIR
        self.root = Path(root) if root else (_ROOT if pkg_dir is None else self.pkg_dir.parent)
        self.docker_dirs = [Path(d) for d in (DOCKER_DIRS if docker_dirs is None else docker_dirs)]
        tp_dir = ((self.pkg_dir.parents[1] / "third_party") if pkg_dir is None
                  else (self.root / "gbench" / "third_party"))
        tp_suites_dir = tp_dir / "eval_suites"
        tp_modules = ({p.stem: p for p in tp_suites_dir.glob("*.py") if not _skipped(p)}
                      if tp_suites_dir.is_dir() else {})
        self.modules = {**tp_modules,
                        **{p.stem: p for p in self.pkg_dir.glob("*.py") if not _skipped(p)}}
        tp_dirs = ({d.name: d for d in tp_dir.iterdir()
                    if d.is_dir() and not _skipped(d)
                    and d.name not in ("__pycache__", "eval_suites")}
                   if tp_dir.is_dir() else {})
        tp_suite_dirs = ({d.name: d for d in tp_suites_dir.iterdir()
                          if d.is_dir() and not _skipped(d) and d.name != "__pycache__"}
                         if tp_suites_dir.is_dir() else {})
        self.data_dirs = {**tp_dirs,
                          **tp_suite_dirs,
                          **{d.name: d for d in self.pkg_dir.iterdir()
                             if d.is_dir() and not _skipped(d) and d.name != "__pycache__"}}
        self.suites = set(CANONICAL_SYNC if suites is None else suites) & set(self.modules)
        self.shared = set(SHARED_SYNC if shared is None else shared) & set(self.modules)
        self.boundaries = self.suites | self.shared
        self._parsed: Dict[str, Tuple[Set[str], List[str]]] = {}
        self._texts: Dict[str, str] = {}
        self._digests: Dict[Path, str] = {}
        self._closures: Dict[str, Tuple[List[str], Set[str], Set[str]]] = {}

    def _parse(self, name: str) -> Tuple[Set[str], List[str]]:
        """(the intra-package names a module imports, its string constants)."""
        if name not in self._parsed:
            imports: Set[str] = set()
            strings: List[str] = []
            tree = _parse_source(self.modules[name].read_bytes(), self.modules[name].name)
            if tree is None:
                tree = ast.Module(body=[], type_ignores=[])
            prefixes = ("gbench.runners.eval_suites", "gbench.third_party.eval_suites",
                        "gbench.third_party")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 1:
                        if node.module:
                            imports.add(node.module.split(".")[0])
                        else:
                            imports.update(a.name for a in node.names)
                    elif node.level == 0 and node.module:
                        for prefix in prefixes:
                            if node.module.startswith(prefix):
                                rest = node.module[len(prefix):].lstrip(".")
                                if rest:
                                    imports.add(rest.split(".")[0])
                                else:
                                    imports.update(a.name for a in node.names)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        for prefix in prefixes:
                            if alias.name.startswith(prefix + "."):
                                imports.add(alias.name[len(prefix) + 1:].split(".")[0])
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    strings.append(node.value)
            self._parsed[name] = (imports, strings)
        return self._parsed[name]

    def _closure(self, name: str) -> Tuple[List[str], Set[str], Set[str]]:
        """(the modules only `name` owns: itself and every non-boundary module it reaches,
        the data directories they import or name, the boundary modules they import)."""
        if name in self._closures:
            return self._closures[name]
        own: List[str] = []
        data: Set[str] = set()
        direct: Set[str] = set()
        todo = [name]
        while todo:
            mod = todo.pop()
            if mod in own:
                continue
            own.append(mod)
            imports, strings = self._parse(mod)
            for imp in imports:
                if imp in UNFINGERPRINTED_MODULES or imp in _INFRA_MODULES or imp == name:
                    continue
                if imp in self.modules:
                    if imp in self.boundaries:
                        direct.add(imp)
                    else:
                        todo.append(imp)
                elif imp in self.data_dirs:
                    data.add(imp)
            text = self._text(mod)
            for d in self.data_dirs:
                if len(d) > 4 and d in text:
                    data.add(d)
        self._closures[name] = (sorted(own), data, direct)
        return self._closures[name]

    def _text(self, name: str) -> str:
        """A module's string constants, joined (searched for data directory and file names)."""
        if name not in self._texts:
            self._texts[name] = "\n".join(self._parse(name)[1])
        return self._texts[name]

    def _docker_files(self, name: str, own: List[str]) -> List[Path]:
        """The Docker build files of a module: files in DOCKER_DIRS named after it (<name>.* or
        <name>_*), and those named after a Docker file its own modules mention (a string holding
        mini_swe_agent_v2.Dockerfile brings in mini_swe_agent_v2_run.py too)."""
        entries = [p for d in self.docker_dirs if d.is_dir() for p in sorted(d.iterdir())
                   if not _skipped(p)]
        text = "\n".join(self._text(mod) for mod in own)
        seeds = {name}
        for p in entries:
            if p.is_file() and p.name in text:
                seeds.add(p.name.split(".", 1)[0])
        out: List[Path] = []
        for p in entries:
            if any(p.name == s or p.name.startswith((s + ".", s + "_")) for s in seeds):
                out.extend(_tree_files(p))
        return out

    def files(self, name: str) -> List[Path]:
        """Every file whose content `name`'s fingerprint covers."""
        own, data, _ = self._closure(name)
        out = [self.modules[m] for m in own]
        for d in sorted(data):
            out.extend(_tree_files(self.data_dirs[d]))
        out.extend(self._docker_files(name, own))
        return sorted(set(out))

    def fingerprint(self, name: str) -> str:
        """16 hex characters of the sha256 over (path, content digest) of files(name)."""
        h = hashlib.sha256()
        rows: List[Tuple[str, str]] = []
        for path in self.files(name):
            try:
                rel = path.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                rel = path.name
            if rel.startswith("gbench/third_party/eval_suites/"):
                rel = "gbench/runners/eval_suites/" + rel[len("gbench/third_party/eval_suites/"):]
            if path not in self._digests:
                self._digests[path] = _file_digest(path)
            rows.append((rel, self._digests[path]))
        for rel, digest in sorted(rows):
            h.update(f"{rel}\0{digest}\n".encode("utf-8"))
        return h.hexdigest()[:16]

    def depends_on(self, name: str) -> List[str]:
        """Every boundary module (suite or shared) `name` imports, directly or through others."""
        seen: Set[str] = set()
        todo = sorted(self._closure(name)[2])
        while todo:
            dep = todo.pop()
            if dep in seen or dep == name:
                continue
            seen.add(dep)
            todo.extend(self._closure(dep)[2])
        return sorted(seen)

    def shared_candidates(self) -> Dict[str, List[str]]:
        """Non-boundary modules that two or more boundaries own (each should be in SHARED_SYNC,
        or its fingerprint would count in several entries), with the boundaries that own them."""
        owners: Dict[str, List[str]] = {}
        for b in sorted(self.boundaries):
            for mod in self._closure(b)[0]:
                if mod != b:
                    owners.setdefault(mod, []).append(b)
        return {m: bs for m, bs in owners.items() if len(bs) > 1}


def code_fingerprints(pkg: Optional[_Package] = None) -> Dict[str, str]:
    """The current fingerprint of every suite with an entry and every SHARED_SYNC module."""
    pkg = pkg or _Package()
    return {name: pkg.fingerprint(name) for name in sorted(pkg.boundaries)}


def dependency_table(pkg: Optional[_Package] = None) -> Dict[str, List[str]]:
    """{name: depends_on(name)} for every suite and shared module that imports another one."""
    pkg = pkg or _Package()
    table = {name: pkg.depends_on(name) for name in sorted(pkg.boundaries)}
    return {k: v for k, v in table.items() if v}


def stale_code_fingerprints(pkg: Optional[_Package] = None) -> Dict[str, Tuple[Optional[str], str]]:
    """{name: (recorded, current)} for every fingerprint CODE_FP does not match, including a
    missing record and a record for a name that no longer exists (current '')."""
    current = code_fingerprints(pkg)
    stale = {n: (CODE_FP.get(n), fp) for n, fp in current.items() if CODE_FP.get(n) != fp}
    stale.update({n: (fp, "") for n, fp in CODE_FP.items() if n not in current})
    return stale


def stale_anchor_digests(
        ns: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[Optional[str], str]]:
    """{name: (recorded, current)} for every recorded fingerprint whose FP_ANCHORS digest is
    missing or no longer matches its entry's anchors: an anchor moved after the last refresh.
    `ns` is a namespace of this module (this module when None). A refresh records the new
    digest, so the next code change needs a decision of its own rather than riding on an older
    bump (refresh_violations)."""
    ns = globals() if ns is None else ns
    recorded = ns.get("FP_ANCHORS") or {}
    registry = {**(ns.get("SHARED_SYNC") or {}), **(ns.get("CANONICAL_SYNC") or {})}
    out: Dict[str, Tuple[Optional[str], str]] = {}
    for name in sorted(ns.get("CODE_FP") or {}):
        now = anchor_digest(registry.get(name))
        if recorded.get(name) != now:
            out[name] = (recorded.get(name), now)
    return out


GATE_STEPS = (
    "For each one: re-reconcile the change against its upstream; if it can change a verdict, bump "
    "the entry's synced (a same-day serial such as 2026-09-28.2 when it was already bumped that "
    "day; and revision / checksum / dataset when the pin moved; a shared module's entry is in "
    "SHARED_SYNC); if it cannot, declare it in VERDICT_PRESERVING with the new "
    "fingerprint and the reason. Then run: python gbench/runners/eval_suites/canonical_sync.py "
    "--refresh-code-fp")


def bump_violations(base: Dict[str, Any], head: Dict[str, Any]) -> List[str]:
    """The names whose fingerprint moved from `base` to `head` without a declared decision.

    `base` and `head` are the namespaces of two versions of this module. A name passes when its
    fingerprint did not move, when base recorded none (a new entry, or the gate's adoption), when
    its anchors (ANCHOR_FIELDS of its CANONICAL_SYNC or SHARED_SYNC entry) moved too, or when
    head's VERDICT_PRESERVING declares exactly the new fingerprint with a reason."""
    old_fp = base.get("CODE_FP") or {}
    new_fp = head.get("CODE_FP") or {}
    declared = head.get("VERDICT_PRESERVING") or {}

    def _anchors(ns: Dict[str, Any], name: str) -> Dict[str, str]:
        entry = (ns.get("CANONICAL_SYNC") or {}).get(name) or (ns.get("SHARED_SYNC") or {}).get(name)
        return anchors(entry)

    out: List[str] = []
    for name in sorted(new_fp):
        before, after = old_fp.get(name), new_fp[name]
        if before is None or before == after:
            continue
        if _anchors(base, name) != _anchors(head, name):
            continue
        decl = declared.get(name) or {}
        if decl.get("code_fp") == after and str(decl.get("reason") or "").strip():
            continue
        out.append(name)
    return out


def anchor_digest(entry: Optional[Dict[str, Any]]) -> str:
    """16 hex characters of the sha256 of an entry's anchors (anchors(), canonical JSON): what
    FP_ANCHORS records next to a fingerprint."""
    blob = json.dumps(anchors(entry), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def refresh_violations(ns: Dict[str, Any], fps: Dict[str, str]) -> List[str]:
    """The names a refresh to `fps` would record without a decision, judged against `ns`, the
    namespace of the version of this module being rewritten.

    bump_violations' rule with the last refresh as the base: a name passes when it has no
    recorded fingerprint (a new entry), when its fingerprint did not move, when its entry's
    anchors no longer match the digest recorded with the old fingerprint in FP_ANCHORS (the
    anchor was bumped since), or when VERDICT_PRESERVING declares exactly the new fingerprint
    with a reason. A moved fingerprint whose FP_ANCHORS digest is missing is undecided: every
    recorded fingerprint has one, so a missing digest is never evidence of a bump."""
    old_fp = ns.get("CODE_FP") or {}
    recorded = ns.get("FP_ANCHORS") or {}
    declared = ns.get("VERDICT_PRESERVING") or {}
    out: List[str] = []
    for name in sorted(fps):
        before, after = old_fp.get(name), fps[name]
        if before is None or before == after:
            continue
        entry = ((ns.get("CANONICAL_SYNC") or {}).get(name)
                 or (ns.get("SHARED_SYNC") or {}).get(name))
        if recorded.get(name) is not None and anchor_digest(entry) != recorded[name]:
            continue
        decl = declared.get(name) or {}
        if decl.get("code_fp") == after and str(decl.get("reason") or "").strip():
            continue
        out.append(name)
    return out


class UndecidedCodeChange(ValueError):
    """--refresh-code-fp was asked to record a moved fingerprint that carries no decision."""

    def __init__(self, names: List[str], fps: Dict[str, Tuple[Optional[str], str]]):
        self.names = list(names)
        self.fps = dict(fps)
        super().__init__(
            "Refusing to record fingerprints that moved without a recorded decision: "
            + ", ".join(f"{n} (recorded {self.fps[n][0]}, now {self.fps[n][1]})"
                        for n in self.names)
            + ". Nothing was written. " + GATE_STEPS)


_BEGIN = "# BEGIN GENERATED CODE FINGERPRINTS (canonical_sync.py --refresh-code-fp)"
_END = "# END GENERATED CODE FINGERPRINTS"


def _generated_block(fps: Dict[str, str], deps: Dict[str, List[str]],
                     fp_anchors: Optional[Dict[str, str]] = None) -> str:
    lines = [_BEGIN,
             "#: The recorded code fingerprint of every suite and SHARED_SYNC module.",
             "CODE_FP: Dict[str, str] = {"]
    lines += [f"    {json.dumps(k)}: {json.dumps(v)}," for k, v in sorted(fps.items())]
    lines += ["}",
              "#: anchor_digest() of each entry when its fingerprint was recorded.",
              "FP_ANCHORS: Dict[str, str] = {"]
    lines += [f"    {json.dumps(k)}: {json.dumps(v)},"
              for k, v in sorted((fp_anchors or {}).items())]
    lines += ["}", "#: The suites and shared modules each one imports, directly or through others.",
              "DEPENDS_ON: Dict[str, List[str]] = {"]
    for k, v in sorted(deps.items()):
        head = f"    {json.dumps(k)}: ["
        body = ", ".join(json.dumps(x) for x in v)
        text = head + body + "],"
        if len(text) <= 100:
            lines.append(text)
            continue
        lines.append(head)
        row = "        "
        for x in v:
            item = json.dumps(x) + ", "
            if len(row) + len(item) > 100:
                lines.append(row.rstrip())
                row = "        "
            row += item
        lines.append(row.rstrip())
        lines.append("    ],")
    lines += ["}", _END]
    return "\n".join(lines)


def refresh_code_fingerprints(path: Optional[Path] = None) -> Dict[str, Tuple[Optional[str], str]]:
    """Rewrite the generated block of this file with the current fingerprints, the anchor digest
    of each entry and the dependency table; return what changed ({name: (old, new)}).

    The file's own registry decides: when a recorded fingerprint moved without a decision
    (refresh_violations), UndecidedCodeChange is raised and nothing is written."""
    path = Path(path) if path else Path(__file__).resolve()
    text = path.read_text(encoding="utf-8")
    # The markers as whole lines, so their own definitions above never match.
    start = text.index("\n" + _BEGIN + "\n") + 1
    end = text.index("\n" + _END + "\n", start) + 1 + len(_END)
    ns = _load_version(text)
    recorded: Dict[str, str] = ns.get("CODE_FP") or {}
    pkg = _Package()
    fps = code_fingerprints(pkg)
    changed = {n: (recorded.get(n), fp) for n, fp in fps.items() if recorded.get(n) != fp}
    changed.update({n: (fp, "") for n, fp in recorded.items() if n not in fps})
    undecided = refresh_violations(ns, fps)
    if undecided:
        raise UndecidedCodeChange(undecided, changed)
    registry = {**(ns.get("SHARED_SYNC") or {}), **(ns.get("CANONICAL_SYNC") or {})}
    digests = {n: anchor_digest(registry.get(n)) for n in fps}
    path.write_text(text[:start] + _generated_block(fps, dependency_table(pkg), digests)
                    + text[end:], encoding="utf-8")
    return changed


def _load_version(source: str) -> Dict[str, Any]:
    """The namespace of a version of this module, from its source (run without its __main__)."""
    ns: Dict[str, Any] = {"__name__": "canonical_sync_version", "__file__": str(Path(__file__))}
    exec(compile(source, "canonical_sync_version.py", "exec"), ns)  # noqa: S102 - own history
    return ns


def main(argv: Optional[List[str]] = None) -> int:
    """--check (fingerprints current), --refresh-code-fp (rewrite them, refused while a moved
    fingerprint carries no decision) or --check-bump --base REF (a fingerprint that moved since
    REF comes with a moved anchor or a declaration)."""
    import argparse
    parser = argparse.ArgumentParser(description="canonical_sync code-fingerprint gate")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--refresh-code-fp", action="store_true")
    mode.add_argument("--check-bump", action="store_true")
    parser.add_argument("--base", help="git ref to compare against (--check-bump)")
    args = parser.parse_args(argv)
    if args.refresh_code_fp:
        try:
            changed = refresh_code_fingerprints()
        except UndecidedCodeChange as err:
            print(err)
            return 1
        for name, (old, new) in sorted(changed.items()):
            print(f"{name}: {old} -> {new or '(removed)'}")
        print(f"{len(changed)} fingerprint(s) refreshed.")
        return 0
    stale = stale_code_fingerprints()
    if stale:
        print("Code fingerprints no longer match the code: " + ", ".join(
            f"{n} (recorded {old}, now {new or 'removed'})" for n, (old, new) in sorted(stale.items())))
        print(GATE_STEPS)
        return 1
    moved = stale_anchor_digests()
    if moved:
        print("Anchors changed since the fingerprints were recorded: " + ", ".join(sorted(moved))
              + ". Run: python gbench/runners/eval_suites/canonical_sync.py --refresh-code-fp")
        return 1
    if args.check:
        print("Code fingerprints are current.")
        return 0
    if not args.base:
        parser.error("--check-bump needs --base <git ref>")
    rel = Path(__file__).resolve().relative_to(_ROOT).as_posix()
    shown = subprocess.run(["git", "-C", str(_ROOT), "show", f"{args.base}:{rel}"],
                           capture_output=True, text=True)
    if shown.returncode != 0:
        print(f"Cannot read {rel} at {args.base}: {shown.stderr.strip()}")
        return 2
    head = _load_version(Path(__file__).resolve().read_text(encoding="utf-8"))
    bad = bump_violations(_load_version(shown.stdout), head)
    if bad:
        print("Code changed without a recorded decision since " + args.base + ": " + ", ".join(bad))
        print(GATE_STEPS)
        return 1
    print(f"Every code change since {args.base} carries a recorded decision.")
    return 0


# BEGIN GENERATED CODE FINGERPRINTS (canonical_sync.py --refresh-code-fp)
#: The recorded code fingerprint of every suite and SHARED_SYNC module.
CODE_FP: Dict[str, str] = {
    "aa_lcr": "f72d8ff8795b4974",
    "acebench": "8877be3eaf780809",
    "agent_dojo": "8859bd5500c7e98b",
    "aider_polyglot": "17b051b13e2a7ba6",
    "aime": "833ff412563ea0af",
    "amc_aime": "6763dc4e600cbf8d",
    "api_bank": "dd846408ab4638b5",
    "arc_agi": "fdf47ea3af8a135a",
    "base": "be2696bf62771c98",
    "beam_128k": "d81703aef5ab872a",
    "bfcl": "d72df2e041984d98",
    "bfcl_live": "a386bfd6bb032473",
    "bfcl_v4_agentic": "971ad5c13f75e7cc",
    "bigcodebench": "36ea46a8dc3fa98d",
    "browsecomp": "6670ddb28b657e0a",
    "bundled_detection": "64a51f6cf111ba04",
    "causalbench": "26cc19b6133a2997",
    "chartqa": "6f48631dae9d5a6f",
    "charxiv": "d2522233de0d3fa7",
    "cimemories": "8e24f80b70690ee5",
    "coco_caption": "e7a945e14804fedd",
    "codeforces": "417c7ade6bde46c1",
    "complexfuncbench": "535bdbe7036a4b63",
    "container_guard": "dd2b6aef03c90d81",
    "copilot_bench_swe": "676f44629bc9dcba",
    "cruxeval": "e7dad82e54ad85f9",
    "custom_jsonl": "ed78e392822273b2",
    "cyberseceval": "65a3cc18bf2c7032",
    "dataset_utils": "0fd1990b26fffdab",
    "deepsearch_qa": "f11c53020594df02",
    "docvqa": "82312ee15521a797",
    "extraction_common": "6e2408944e02a857",
    "fc_common": "2e1f1fd3ce64dd3d",
    "frames": "20dcffbd0d6f74ea",
    "gaia": "487f83a834174d38",
    "gaia2": "37072783f181fcc3",
    "gdpval": "f850e4a195db0cdd",
    "gemini_budget": "19b005ca590fa821",
    "gemini_relay": "148f07cfd1e88a8e",
    "gorilla_apibench": "78ef61b770f1404e",
    "gpqa": "7316e28a5b6b946a",
    "gpqa_diamond": "83bab113c3f268bc",
    "gsm8k": "21cee268c20cdf86",
    "healthbench": "2beb47181e7ea943",
    "hmmt": "d7177433acf800e3",
    "humanitys_last_exam": "51fc8202e70e62f2",
    "i18n_translate": "a1f73f36cd860ee0",
    "ifeval": "36baabf70a59be06",
    "imo_answer_bench": "10db7e689a2c0c8a",
    "infographicvqa": "4a56ed8980b264d0",
    "lab_bench": "573a8b61d2fa4b14",
    "lcb": "b7f9a22c6d66d4d1",
    "limit_plan": "a7d1663f62ae3824",
    "livebench": "94201e82a1e6d594",
    "loft_scifact": "d4b94cbc654fe120",
    "longbench_v2_code": "e36a445ffb7f3091",
    "matharena_common": "7f2cc975019e7777",
    "mcp_atlas": "b117d1edc40119c3",
    "mcp_bench": "5be3cf7fa714d93b",
    "medxpertqa": "f68d200324fbb27e",
    "metrics": "4a278892c8f9978f",
    "mmlu": "7715bf80f7058e3f",
    "mmlu_pro": "8a2a41edb38925cc",
    "mmlu_redux": "22640bf43bf32ba4",
    "mmmu_pro": "b25e861a4455f834",
    "mobilegym": "7778a8893a3891f4",
    "mobilegym_pp": "a938d62cbe80c099",
    "mrcr": "8ac5086f100bdc03",
    "multi_swe_bench": "1827f69fb58253de",
    "multilingual_mmlu": "267ff1ed52705f6f",
    "multipl_e": "98a635bbef17f8ee",
    "nestful": "614cfa2b74754fad",
    "nexusraven_api_eval": "69271c5cdef02c4f",
    "ojbench": "e272b1ebc4ff34cd",
    "omnidocbench": "2ed54782544e39a3",
    "putnam": "fda2daa44212e435",
    "putnam_formal": "d3fddf828c75c269",
    "ruler": "41bd62fee5f9fcc4",
    "sampling": "12c2132c1d713f9c",
    "sandbox": "e54065193eaaf6b3",
    "scicode": "273f15c37bf7b32d",
    "screenspot": "8fa119085d98d5bc",
    "screenspot_pointing": "244c6c64749a177f",
    "seal_tools": "63af4bd5e1da8260",
    "search_tool": "3b98cf8635520d6e",
    "simpleqa": "7d614960a82d45b2",
    "skillsbench": "6f778fa8974ab4e2",
    "spider2": "ecb94de7e567ce47",
    "swe_agentic": "6235c964140d9a51",
    "swe_bench_live": "4167471f5ad51c6e",
    "swe_bench_multilingual": "589139e0b8c2e76b",
    "swe_bench_pro": "de4b985e6a61ee61",
    "swe_lancer": "9b4911f5d93de00a",
    "swe_thread_cap": "8b9561e7c7f7c1fb",
    "swebench_common": "c1f99e628860711b",
    "t_eval": "1bd22e79189c50ba",
    "tau2": "4882d354d4342707",
    "tau3": "0f5c202d0cecd0f3",
    "tau_common": "84398b7dfe65ade8",
    "terminal_bench": "485acbf59d7235df",
    "textvqa": "1b137b80e259bba7",
    "toolbench": "19e66b933a5ee91b",
    "ui_control_osworld": "37857029595375d9",
    "vqa_common": "891527af55a8d7d4",
    "wildbench": "5b39dad04dc9e710",
    "wildclawbench": "4d52c606065d1020",
    "wmdp": "064e102e8ed419cb",
}
#: anchor_digest() of each entry when its fingerprint was recorded.
FP_ANCHORS: Dict[str, str] = {
    "aa_lcr": "0eeb84c9c5a19633",
    "acebench": "ec9605611d656172",
    "agent_dojo": "6ea59108bdfc66c9",
    "aider_polyglot": "0eac690986f103ea",
    "aime": "de1425ab70625533",
    "amc_aime": "6c69c599e5ccd2f7",
    "api_bank": "5017e44c4cb0f7bc",
    "arc_agi": "6647fbee1910387f",
    "base": "11b9b68b3b48598e",
    "beam_128k": "edbfdde3fb426c32",
    "bfcl": "7b3aa73d3137646f",
    "bfcl_live": "67934a4134db8770",
    "bfcl_v4_agentic": "8e9bef4ce6815126",
    "bigcodebench": "c7fa10c21a335b05",
    "browsecomp": "f1df87c0320f3580",
    "bundled_detection": "a0c61691d28a99c6",
    "causalbench": "6bdf547466771c74",
    "chartqa": "456b731e6fa12368",
    "charxiv": "02335c09a9a1b0e5",
    "cimemories": "bc5477fc0da1ea2d",
    "coco_caption": "7a1c471f11aae4ac",
    "codeforces": "ab4d5da464d73804",
    "complexfuncbench": "e2aa94486b864b02",
    "container_guard": "8ea8ee6755e2694f",
    "copilot_bench_swe": "1197cf8948aeecdd",
    "cruxeval": "876810005c480d36",
    "custom_jsonl": "a86345574b3618a5",
    "cyberseceval": "31d5517da74b68f7",
    "dataset_utils": "8ea8ee6755e2694f",
    "deepsearch_qa": "ad6b853b6d4cb4a1",
    "docvqa": "802ab231ae3971f2",
    "extraction_common": "8ea8ee6755e2694f",
    "fc_common": "618fb6f9d8e2172d",
    "frames": "8e55a2d508e70957",
    "gaia": "9d50d45cea029a04",
    "gaia2": "368a19c4d7a86102",
    "gdpval": "5feba355102c6721",
    "gemini_budget": "6ba2582d8ad0ba69",
    "gemini_relay": "8ea8ee6755e2694f",
    "gorilla_apibench": "5c37682fdb357908",
    "gpqa": "490e464e87b75700",
    "gpqa_diamond": "c9f73fbc769fdcb1",
    "gsm8k": "b72760ecc3c671dd",
    "healthbench": "bbc01c23e15e3deb",
    "hmmt": "99c696d484a2762b",
    "humanitys_last_exam": "1e7d24cb1cfc6f81",
    "i18n_translate": "c6cd3d78d74c9c0c",
    "ifeval": "5a6407376728900b",
    "imo_answer_bench": "ace829f0f5f989a4",
    "infographicvqa": "35650ab471e2a5e5",
    "lab_bench": "a0fd9156e3930c86",
    "lcb": "208901a7d64d1285",
    "limit_plan": "618fb6f9d8e2172d",
    "livebench": "b2fd1d9c017dd368",
    "loft_scifact": "9a34be18b4813f91",
    "longbench_v2_code": "51b4b98de194a821",
    "matharena_common": "9a3cb0fbc83f5fe2",
    "mcp_atlas": "dd5c4e4508a82de8",
    "mcp_bench": "4ac832afd9c40aa0",
    "medxpertqa": "79264394fed00cec",
    "metrics": "8ea8ee6755e2694f",
    "mmlu": "7bc762bbf3695c54",
    "mmlu_pro": "7b5dc1e34235d5ad",
    "mmlu_redux": "2f4d8ee467c72b67",
    "mmmu_pro": "583262900efc7c10",
    "mobilegym": "96205c8e6054eab0",
    "mobilegym_pp": "c501901f05b754cb",
    "mrcr": "c62dc9289f6b2e1a",
    "multi_swe_bench": "6af51c778ee7d1f6",
    "multilingual_mmlu": "7852f324f3b487f5",
    "multipl_e": "cfa60f72a1a2931f",
    "nestful": "b00c59b703228495",
    "nexusraven_api_eval": "3d1a4341bc7a623c",
    "ojbench": "e35aba08aa10abb0",
    "omnidocbench": "29e690e47345b01e",
    "putnam": "a6e0f63cbce722e4",
    "putnam_formal": "1cce16349eea2c5b",
    "ruler": "0f52b0d8a7202208",
    "sampling": "618fb6f9d8e2172d",
    "sandbox": "8ea8ee6755e2694f",
    "scicode": "39e11d402b483bb4",
    "screenspot": "b9218f4425343f20",
    "screenspot_pointing": "f94997fbaf3f68d9",
    "seal_tools": "b45f416b1aa7bf56",
    "search_tool": "d630982407afdc0f",
    "simpleqa": "5ab27c4c95c24a62",
    "skillsbench": "b7995fbb29b1e5f0",
    "spider2": "dcbb6ecc32f17124",
    "swe_agentic": "11b9b68b3b48598e",
    "swe_bench_live": "87a1cee90ea37850",
    "swe_bench_multilingual": "e374781a3a8cfcc7",
    "swe_bench_pro": "1b5a0cdb8e47aed7",
    "swe_lancer": "9144d83a0751f1bf",
    "swe_thread_cap": "bc6b97c502cfd552",
    "swebench_common": "065d8cc9ad9051eb",
    "t_eval": "b250176d86f308aa",
    "tau2": "0a9a77cd7666ef74",
    "tau3": "a95497cbe0902f9b",
    "tau_common": "6e073e479a631450",
    "terminal_bench": "8451a220761af370",
    "textvqa": "a7311d0cc564cf14",
    "toolbench": "ba29cacb6086ae5b",
    "ui_control_osworld": "8334a8708a3ed1ad",
    "vqa_common": "8ea8ee6755e2694f",
    "wildbench": "b8bfcbeac835e0ae",
    "wildclawbench": "5787a12f5ef01f06",
    "wmdp": "3ecc1d3bc602b9f5",
}
#: The suites and shared modules each one imports, directly or through others.
DEPENDS_ON: Dict[str, List[str]] = {
    "aa_lcr": [
        "base", "container_guard", "limit_plan", "metrics", "ojbench", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "acebench": [
        "base", "container_guard", "fc_common", "gemini_budget", "limit_plan", "metrics",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "agent_dojo": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "aider_polyglot": [
        "base", "container_guard", "gemini_relay", "limit_plan", "livebench", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "tau_common",
    ],
    "aime": [
        "base", "container_guard", "hmmt", "limit_plan", "matharena_common", "metrics", "ojbench",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "amc_aime": [
        "base", "container_guard", "extraction_common", "limit_plan", "metrics", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "api_bank": [
        "aider_polyglot", "base", "container_guard", "fc_common", "gemini_relay", "limit_plan",
        "livebench", "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
        "tau_common",
    ],
    "arc_agi": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "base": [
        "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic", "swe_thread_cap",
        "swebench_common",
    ],
    "beam_128k": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "bfcl": [
        "aider_polyglot", "base", "bfcl_v4_agentic", "container_guard", "fc_common",
        "gemini_relay", "limit_plan", "livebench", "sampling", "sandbox", "search_tool",
        "swe_agentic", "swe_thread_cap", "swebench_common", "tau_common",
    ],
    "bfcl_live": [
        "aider_polyglot", "base", "bfcl", "bfcl_v4_agentic", "container_guard", "fc_common",
        "gemini_relay", "limit_plan", "livebench", "sampling", "sandbox", "search_tool",
        "swe_agentic", "swe_thread_cap", "swebench_common", "tau_common",
    ],
    "bfcl_v4_agentic": [
        "aider_polyglot", "base", "bfcl", "container_guard", "fc_common", "gemini_relay",
        "limit_plan", "livebench", "sampling", "sandbox", "search_tool", "swe_agentic",
        "swe_thread_cap", "swebench_common", "tau_common",
    ],
    "bigcodebench": [
        "aider_polyglot", "base", "container_guard", "gemini_relay", "limit_plan", "livebench",
        "metrics", "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
        "tau_common",
    ],
    "browsecomp": [
        "arc_agi", "base", "container_guard", "fc_common", "gemini_budget", "gpqa", "limit_plan",
        "metrics", "sampling", "sandbox", "simpleqa", "swe_agentic", "swe_thread_cap",
        "swebench_common",
    ],
    "bundled_detection": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "causalbench": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "chartqa": [
        "base", "charxiv", "container_guard", "dataset_utils", "limit_plan", "metrics", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "charxiv": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "cimemories": [
        "base", "container_guard", "fc_common", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "coco_caption": [
        "base", "container_guard", "dataset_utils", "limit_plan", "metrics", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "codeforces": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "complexfuncbench": [
        "base", "container_guard", "fc_common", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "container_guard": [
        "base", "limit_plan", "sampling", "sandbox", "swe_agentic", "swe_thread_cap",
        "swebench_common",
    ],
    "copilot_bench_swe": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "cruxeval": [
        "arc_agi", "base", "container_guard", "lcb", "limit_plan", "metrics", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "custom_jsonl": [
        "base", "container_guard", "extraction_common", "fc_common", "limit_plan", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "cyberseceval": [
        "base", "container_guard", "fc_common", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "deepsearch_qa": [
        "base", "container_guard", "fc_common", "gaia", "limit_plan", "metrics", "sampling",
        "sandbox", "search_tool", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "docvqa": [
        "base", "charxiv", "container_guard", "dataset_utils", "limit_plan", "metrics", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "fc_common": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "frames": [
        "arc_agi", "base", "container_guard", "fc_common", "gemini_budget", "gpqa", "limit_plan",
        "metrics", "sampling", "sandbox", "simpleqa", "swe_agentic", "swe_thread_cap",
        "swebench_common",
    ],
    "gaia": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "search_tool",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "gaia2": [
        "base", "container_guard", "gemini_relay", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "gdpval": [
        "base", "container_guard", "dataset_utils", "fc_common", "limit_plan", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "gemini_relay": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "gorilla_apibench": [
        "base", "container_guard", "fc_common", "limit_plan", "metrics", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "gpqa": [
        "arc_agi", "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "gpqa_diamond": [
        "arc_agi", "base", "container_guard", "gpqa", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "gsm8k": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "healthbench": [
        "arc_agi", "base", "container_guard", "gemini_budget", "gpqa", "limit_plan", "metrics",
        "sampling", "sandbox", "simpleqa", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "hmmt": [
        "base", "container_guard", "limit_plan", "matharena_common", "metrics", "ojbench",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "humanitys_last_exam": [
        "arc_agi", "base", "container_guard", "gemini_budget", "gpqa", "limit_plan", "metrics",
        "sampling", "sandbox", "simpleqa", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "i18n_translate": [
        "base", "container_guard", "fc_common", "limit_plan", "metrics", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "ifeval": [
        "arc_agi", "base", "container_guard", "fc_common", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "imo_answer_bench": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "infographicvqa": [
        "base", "charxiv", "container_guard", "dataset_utils", "limit_plan", "metrics", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "lab_bench": [
        "arc_agi", "base", "container_guard", "fc_common", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "lcb": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "limit_plan": [
        "base", "container_guard", "sampling", "sandbox", "swe_agentic", "swe_thread_cap",
        "swebench_common",
    ],
    "livebench": [
        "aider_polyglot", "base", "container_guard", "gemini_relay", "limit_plan", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "tau_common",
    ],
    "loft_scifact": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "longbench_v2_code": [
        "arc_agi", "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "matharena_common": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "mcp_atlas": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "search_tool",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "mcp_bench": [
        "base", "container_guard", "gemini_relay", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "medxpertqa": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "mmlu": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common", "wmdp",
    ],
    "mmlu_pro": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "mmlu_redux": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "mmmu_pro": [
        "base", "container_guard", "dataset_utils", "limit_plan", "metrics", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "mobilegym": [
        "base", "container_guard", "gaia2", "gemini_relay", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "mobilegym_pp": [
        "base", "container_guard", "gaia2", "gemini_relay", "limit_plan", "mobilegym", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "mrcr": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "multi_swe_bench": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "multilingual_mmlu": [
        "arc_agi", "base", "container_guard", "gpqa", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "multipl_e": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "nestful": [
        "base", "container_guard", "fc_common", "limit_plan", "metrics", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "nexusraven_api_eval": [
        "base", "container_guard", "fc_common", "limit_plan", "metrics", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "ojbench": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "omnidocbench": [
        "base", "charxiv", "container_guard", "dataset_utils", "limit_plan", "metrics", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "putnam": [
        "base", "container_guard", "fc_common", "imo_answer_bench", "limit_plan", "metrics",
        "putnam_formal", "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "putnam_formal": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "ruler": [
        "base", "container_guard", "limit_plan", "metrics", "mrcr", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "sandbox": [
        "base", "container_guard", "limit_plan", "sampling", "swe_agentic", "swe_thread_cap",
        "swebench_common",
    ],
    "scicode": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "screenspot": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox",
        "screenspot_pointing", "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "screenspot_pointing": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "seal_tools": [
        "base", "container_guard", "fc_common", "limit_plan", "metrics", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "simpleqa": [
        "arc_agi", "base", "container_guard", "gemini_budget", "gpqa", "limit_plan", "metrics",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "skillsbench": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "spider2": [
        "base", "container_guard", "limit_plan", "metrics", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "swe_agentic": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_thread_cap",
        "swebench_common",
    ],
    "swe_bench_live": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "swe_bench_multilingual": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "swe_bench_pro": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "swe_lancer": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "swebench_common": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap",
    ],
    "t_eval": [
        "base", "container_guard", "fc_common", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "tau2": [
        "aider_polyglot", "base", "container_guard", "gemini_relay", "limit_plan", "livebench",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "tau_common",
    ],
    "tau3": [
        "aider_polyglot", "base", "container_guard", "gemini_relay", "limit_plan", "livebench",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "tau_common",
    ],
    "tau_common": [
        "aider_polyglot", "base", "container_guard", "gemini_relay", "limit_plan", "livebench",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "terminal_bench": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "textvqa": [
        "base", "charxiv", "container_guard", "dataset_utils", "limit_plan", "metrics", "sampling",
        "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common", "vqa_common",
    ],
    "toolbench": [
        "base", "container_guard", "gemini_budget", "gemini_relay", "limit_plan", "mcp_bench",
        "sampling", "sandbox", "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "ui_control_osworld": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "vqa_common": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common",
    ],
    "wildbench": [
        "arc_agi", "base", "container_guard", "fc_common", "gemini_budget", "gpqa", "limit_plan",
        "metrics", "sampling", "sandbox", "simpleqa", "swe_agentic", "swe_thread_cap",
        "swebench_common",
    ],
    "wildclawbench": [
        "base", "container_guard", "gemini_budget", "limit_plan", "sampling", "sandbox",
        "swe_agentic", "swe_thread_cap", "swebench_common",
    ],
    "wmdp": [
        "base", "container_guard", "limit_plan", "sampling", "sandbox", "swe_agentic",
        "swe_thread_cap", "swebench_common", "vqa_common",
    ],
}
# END GENERATED CODE FINGERPRINTS


if __name__ == "__main__":
    sys.exit(main())
