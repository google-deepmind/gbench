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

# Evaluation Suite: gaia2
# Description: GAIA2 / Meta Agents Research Environments (ARE) - stateful multi-turn agentic benchmark

"""gbench native built-in runner for gaia2 (Tool Use & Agentic Workflows).

Canonical GAIA2 (Meta Agents Research Environments,
github.com/facebookresearch/meta-agents-research-environments; PyPI
`meta-agents-research-environments`; HF dataset `meta-agents-research-environments/gaia2`) is a
STATEFUL, multi-turn, time-sensitive agentic benchmark: the agent acts inside Meta's ARE simulator,
issuing tool calls over many turns against evolving app state (contacts/calendar/email/files with
async events). Scoring is HYBRID - deterministic "hard validation" plus a LOAD-BEARING LLM judge for
soft/semantic checks.

gbench DELEGATES the whole run to the pinned `are-benchmark` CLI (ARE 1.2.0, the current PyPI
release) inside a LOCAL orchestrator image (gbench/docker/gaia2.Dockerfile). Unlike
skillsbench/wildclawbench there is NO docker-out-of-docker and NO sibling task containers - ARE is a
pure in-process Python simulator. What gbench sets (everything else, incl. the judge PROMPTS, is
upstream verbatim):
  * the model-under-test is wired via LiteLLM's `local` provider (`--model openai/<served> --provider
    local --endpoint <gbench's endpoint relay>/v1`);
  * the agent protocol of the ARE paper (arXiv:2509.17158 section 4.1 and appendix B.5):
    at most 16,000 tokens per turn, the agent's stop sequences `<end_action>`
    and `Observation:` ending each answer, and the 200-step limit; ARE 1.2.0's LiteLLMEngine
    sends none of these and its main agent stops at 80 steps. The paper's temperature 0.5 is
    recorded, not sent: the agent samples at gbench's rule for every suite (1.0 with --thinking,
    0.0 without, base.resolve_temperature). Without --thinking the stop
    sequences are sent as `stop`, as ARE's own HuggingFaceLLMEngine sends them. A --thinking run
    sends the chat template's thinking switch, and is never capped below
    --max-output-tokens (the token-limit policy, base.resolve_token_budget); it sends no `stop`,
    because the served stack matches stop strings over the reasoning too (vLLM's detokenizer),
    so a thought containing "Observation:" would end the reply before any answer: its agent
    request is streamed and closed once the answer holds a stop sequence, so the answer ends
    where the paper's stop sequences end it and the server stops generating there. ARE's
    client-side split of the answer at the first stop sequence applies on both runs;
  * the Time capability's clock: ARE's `measured` mode advances the simulated time by the agent
    engine's completion_duration, which LiteLLMEngine never returned; the launcher returns the
    client-measured generation time, as the paper runs it (section 4.3), up to the end of the
    answer (on a --thinking run, up to the stop sequence that closed the streamed reply);
  * the judge path as the current upstream judge runs it (gaia2-cli: the judge reply verbatim, the
    verdict marker matched case-insensitively). ARE 1.2.0's LiteLLMEngine lowercases True/False in
    every reply, so the "[[True]]" verdict of the signature, sanity, cab and tone checkers never
    matched (upstream issue #29, PR #26, unfixed in 1.2.0 and on main);
  * gbench's standard Gemini cascade grades in place of ARE's default Llama-3.3-70B judge
    (meta-llama/Meta-Llama-3.3-70B-Instruct), via an in-container OpenAI-compat proxy
    (`--judge_provider local --judge_endpoint http://127.0.0.1:<port>/v1 --judge_model
    openai/gbench-cascade`) at temperature 0 (the paper's verifier setting). The judge endpoint and
    model are fixed: no knob points the judge anywhere else. The substitution is recorded on the
    result (`gemini_substitutions`), the model that answered every soft check in `judge_verdicts`
    and `sample_traces`, and the substitution in leaderboard_comparable_reason.

Sampling: the agent's requests carry temperature 1.0 with --thinking and 0.0 without (gbench's
rule for every suite, base.resolve_temperature; the ARE paper's 0.5 is recorded under
`official_sampling`); GBENCH_GAIA2_TEMPERATURE, else --temperature, replaces either, and a
value other than 0.5 is named in leaderboard_comparable_reason; max_tokens 16,000 per turn without --thinking (the token-limit
policy: with --thinking the paper's cap, written for non-reasoning generation, is not applied and
every turn is sent the run's budget, --max-output-tokens), the stop sequences (sent as `stop`
without --thinking; with it the streamed request is closed at the first one in the answer), the
chat template's thinking switch of the run, and a request timeout sized to the per-turn budget
(LiteLLM's 600 s for 16,000 tokens, scaled with a larger budget; a streamed request still running
then is closed and recorded as a timeout). The judge samples at temperature 0. Replies whose
answer still held a stop sequence (`agent_calls.ran_past_stop`: generation ran past the action and
the Time clock counted it) and, with --thinking, replies cut at max_tokens are named in
leaderboard_comparable_reason.

Headline `accuracy` = the per-capability equal-weight **macro success rate** from ARE's own
statistics (the GAIA2 Overall; ARE writes it in percent); micro (per-scenario) rate +
per-capability breakdown are also reported. `leaderboard_comparable` is always False: the Gemini
judge, and gbench defaults to the standard phase (`run`, 3 runs per scenario) where the leaderboard
runs the 3-phase `gaia2-run`.

Harness errors (never scored as the model's outcome): ARE turns a judge exception into a failed
validation and a raised agent run into a failure, and its agent loop goes on after a failed model
call, so its own success rates count those as the model's failures. The launcher runs ARE with
gbench's hooks (gbench/docker/gaia2_run.py) and recomputes ARE's statistics over the scenario runs
without a harness error: a run in which a model endpoint call of the agent failed on the endpoint's
side (any HTTP status other than 400 and 422, which reject the request's own content and are the
model's outcome here, as gbench's quality runner and toolbench count them; base-run suites, bfcl,
mcp_bench and the SWE family count a 400/422 as a harness error, see docs/evals/env-knobs.md; no
connection; a timeout; an error inside a streamed reply), the server's tool-call parser deleted an
agent reply beyond recovery (server_parser_text_lost, below), a judge call raised, a judge reply
held no verdict, the validation raised otherwise, or the run raised. Runs ARE could not validate
(`no_validation_runs`) are left out by ARE itself. The excluded runs are counted by kind
(`n_harness_error_runs`, `harness_error_kinds`), named in leaderboard_comparable_reason and set the
status to `completed_with_errors`; ARE's own rates with them are kept as `are_reported`. A run with
a harness error is left out whatever ARE validated it as (a failed judge pairing changes which
agent event ARE's greedy event matching gives each later oracle event, and a failed agent call
puts an error into the agent's history, so its effect on the outcome is unknowable); the ones ARE
validated as successes are counted (`n_harness_error_success_runs`). A failed call that cannot be
tied to a run, no validated run left, or a run in which not one agent request got a reply (no
request then carried text the model had generated, so not even a 400/422 is the model's), is
infra_required (no rate is published). A scenario
timeout is the model's outcome, as ARE counts it (`n_timeout_runs`).

The server-parser policy (base.recover_server_parser_text, applied by the launcher to every agent
reply): every agent request asks for the generated token ids (return_token_ids). On a run without
thinking, text the server's reasoning parser put in the reasoning field is the model's output; in
either mode, a reply with no content, no reasoning and generated tokens lost its text to the
server's tool-call parser (a native tool call on a request that declared no tools: ARE declares
none). Both are recovered in generation order from the token ids, decoded with special tokens kept
by gbench's decoder (DetokenizeRelay on the docker network's gateway: the served tokenizer's
/detokenize, else the run's tokenizer), or (moved text only) from the reasoning followed by the
content, and handed to ARE as the reply, which parses it as it parses any reply: the model's
outcome. Each event is counted and named (`server_parser_events`, `server_parser_note`,
leaderboard_comparable_reason), and each call keeps the recovered text (capped). Deleted text that
cannot be recovered is a harness error (`server_parser_text_lost`): the run is left out of the
rates.

Capabilities: ARE runs each capability config in a try/except, skips one that raised and reports
the others with exit code 0, and its macro formula gives a capability without a validated run a
success rate of 0.0. The launcher records every config's outcome and averages the macro over the
capabilities with a validated run; the runner compares the scored capabilities with the ones the
run covers (the 5 of `run`, the 7 capability groups of `gaia2-run`, or GBENCH_GAIA2_CONFIG), and
each one missing (its config raised, ARE did not run it, every run of it a harness error, or none
validated) is named in leaderboard_comparable_reason and `missing_capabilities` and sets
`completed_with_errors`: the macro is then over fewer capabilities than the GAIA2 Overall.

Model-controlled calls and credentials: the ARE agent's tool calls run in the gbench-gaia2:are-1.2.0
container, so it runs on a docker bridge network, starts only after
container_guard.require_metadata_blocked found the cloud metadata server blocked there, and holds
no real key or token. The
model endpoint is reached through gbench's endpoint relay (swebench_common.EndpointRelay, a thread
of this process on the network's gateway, which adds GBENCH_GAIA2_MODEL_API_KEY), the Gemini API
through gbench's Gemini relay (gemini_relay.GeminiRelay on the same gateway, which adds the
Gemini key): the container gets the relay's per-run placeholder and addresses by name (`docker run
-e NAME`). No HF token goes in: the image holds local copies of gaia2 and gaia2_filesystem at
pinned revisions and runs with the Hub off (HF_HUB_OFFLINE, HF_DATASETS_OFFLINE), so ARE's loader
and filesystem app read them locally and a run makes no Hub request (`dataset` on the result). The
token-id decoder (DetokenizeRelay) listens on the same gateway and only turns token ids into text.
The judge cascade goes in as GBENCH_JUDGE_MODELS (base.gemini_cascade_env).

HARD-ERRORS (infra_required, never skips) if Docker, the orchestrator image (or one that predates
this launcher, label org.gbench.gaia2-harness-errors=7), a Gemini key (GEMINI_API_KEYS, else
GEMINI_API_KEY: the load-bearing judge), a reachable model endpoint or the metadata block on the
docker bridge are missing, or if GEMINI_OPENAI_BASE_URL is neither the Gemini API nor a loopback
address. The public cc-by-4.0 datasets are baked into the image at pinned revisions. See
docs/evals/gaia2.md.

Sharding: `--shard I/N` (validated here, sent to the container as GAIA2_SHARD) runs shard I of N
of each capability config's scenarios: the launcher's hook on the dataset ARE streams keeps only
the shard's scenario ids (sorted, round-robin) and, with --eval-limit, a seeded draw of each
config's share of the run's N scenarios (config_limit_shares: N for the whole run, dealt over the
configs, never N per config; sent as GAIA2_CONFIG_LIMITS, a config with 0 is not run) (the
launcher's copy of sampling.stratified_indices, seed "gaia2"; its summary's
`selection.limit_draw`, which gbench requires for a limited run: an image without it let ARE's
--limit take the first scenarios of each stream), and each kept scenario runs every phase and run
as upstream (gaia2_run.py install_shard_hook). The result records the selection (`selection`). The result records `shard` and the launcher's per-config selection
(`shard_selection`); a shard with no scenario of any configured capability reports no score
(`accuracy` null, `empty_shard`) without starting ARE. `gbench --merge` recomputes ARE's
statistics (report_stats.calculate_statistics: per-capability success rate over the validated
runs, the macro over the capabilities with a validated run, the micro, Pass@k and Pass^k) over the
union of the shards' scenario runs without a harness error (`merge_shards`).

--attempt-count: GAIA2 runs every scenario `num_runs` times (ARE's default 3); an --attempt-count
above 3 raises the runs to that value (GBENCH_GAIA2_NUM_RUNS, when set, decides), a smaller one
never cuts ARE's 3, and a value above 1 is recorded under `attempts`. gaia2-run always runs 3
(ARE's gaia2_submission hard-codes NUM_RUNS for every phase; resolve_num_runs). Concurrency: ARE runs
GBENCH_GAIA2_MAX_CONCURRENT scenarios at once, else the suite's concurrency (--eval-concurrency /
--batch-sizes); ARE's own default (its CPU count) is not used, since the Time capability's clock
advances by the measured generation time, which depends on the load on the endpoint; the value is
recorded (`max_concurrent_scenarios`).
"""

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import urllib.error  # noqa: F401 - the key gate's tests patch urlopen through this module
import urllib.request  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple

from . import container_guard, gemini_relay
from .swebench_common import EndpointRelay, infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/gaia2.md"

_IMAGE_DEFAULT = "gbench-gaia2:are-1.2.0"
_JUDGE_PORT_DEFAULT = 18790
#: ARE 1.2.0's default judge (are/simulation/validation/configs.py DEFAULT_JUDGE_MODEL / _PROVIDER).
ARE_DEFAULT_JUDGE = "meta-llama/Meta-Llama-3.3-70B-Instruct (ARE's default judge, huggingface provider)"
#: Knobs that no longer change anything (setting one logs a warning): the judge is always the
#: gbench Gemini cascade behind the in-container proxy.
_REMOVED_KNOBS = ("GBENCH_GAIA2_JUDGE_ENDPOINT", "GBENCH_GAIA2_JUDGE_MODEL")
#: Credentials of the runner's environment that never reach the container (the Gemini relay's
#: placeholder takes the Gemini keys' names; the endpoint relay adds the model endpoint's key).
_CREDENTIAL_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GBENCH_MODEL_API_KEY", "VLLM_API_KEY",
                    "OPENAI_API_KEY",
                    "GBENCH_GAIA2_MODEL_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                    "GEMINI_OPENAI_BASE_URL", gemini_relay.RELAY_URL_ENV)
# GAIA2's 5 capability configs (CapabilityTag.gaia2_capabilities()); demo/mini are smoke subsets.
_CAPABILITY_CONFIGS = ("adaptability", "ambiguity", "execution", "search", "time")
#: The capability groups gaia2-run adds (gaia2_submission.py: the mini config in the
#: agent2agent and noise phases), keyed as report_stats.calculate_statistics keys them.
_GAIA2_RUN_EXTRA_CAPABILITIES = ("mini_agent2agent", "mini_noise")
#: The image label of an orchestrator whose launcher applies the agent protocol and the upstream
#: judge path, leaves harness-error runs out of the success rates, records every capability
#: config's outcome and averages the macro over the capabilities with a validated run
#: (gaia2_run.py: ARE in a hooked child process, gbench_stats.json), streams a --thinking agent
#: request and closes it at the first stop sequence in the answer, counts every HTTP status but
#: 400/422 as an endpoint failure, applies the server-parser policy (asks for the token ids and
#: recovers text the server's parsers took out of a reply, the text a tool-call parser deleted
#: after a reply's returned reasoning or content included), applies --shard (GAIA2_SHARD) and
#: reads the datasets offline from its local copies; an older image would score judge outages,
#: endpoint failures and unparsed verdicts
#: as the model's failures, cut thinking replies at a stop string inside the reasoning, let a
#: thinking reply run past the action into the Time clock, leave a skipped capability unnamed,
#: drop runs whose reply the server moved into the reasoning field, hand ARE nothing for a reply
#: the tool-call parser deleted (or for a --thinking reply whose answer it deleted after the
#: thought), ignore --shard, or lose capability configs to Hub rate limits.
HARNESS_ERRORS_LABEL = "org.gbench.gaia2-harness-errors"
HARNESS_ERRORS_VERSION = "7"
#: The launcher's harness-error kinds of a scenario run (gaia2_run.HARNESS_ERROR_KINDS), in order,
#: as they are named in leaderboard_comparable_reason.
HARNESS_ERROR_LABELS = (
    ("model_endpoint", "a model endpoint call of the agent failed on the endpoint's side"),
    ("server_parser_text_lost", "the server's tool-call parser deleted an agent reply and its "
                                "text could not be recovered from token ids"),
    ("judge", "a judge call of the validation failed"),
    ("judge_no_verdict", "a judge reply held no verdict"),
    ("validation_exception", "the validation raised"),
    ("run_exception", "the scenario run raised"),
)
#: The docker network the orchestrator runs on (the agent's tool calls run in it).
NETWORK = "bridge"
#: Host names the orchestrator must resolve: none. It reaches gbench's relays by the network's
#: gateway address, and the datasets are read offline from the image's local copies.
DNS_NAMES = ()
#: The agent's stop sequences (ARE base_agent.py step(); the paper's appendix B.5), sent as `stop`
#: without --thinking and applied to the streamed answer with it.
AGENT_STOP_SEQUENCES = ["<end_action>", "Observation:"]
#: The stream options of a streamed (--thinking) agent request (gaia2_run.STREAM_OPTIONS): the
#: usage at the end, and vLLM's usage so far in every chunk.
STREAM_OPTIONS = {"include_usage": True, "continuous_usage_stats": True}
#: How a --thinking run applies them (gaia2_run.client_stop).
STOP_NOTE_THINKING = (
    "not sent to the server on a --thinking run: the served stack matches stop strings over the "
    "whole generated text, the reasoning included (vLLM's detokenizer), so a thought containing "
    "'Observation:' would end the reply before any answer; the agent request is streamed (the "
    "reasoning and the answer arrive in separate fields) and closed once the answer holds a stop "
    "sequence, which ends the generation on the server, so the answer, the measured generation "
    "time that advances the Time capability's clock and the request end where a server-side "
    "stop on the answer would end them; ARE's client-side split of the answer at the first stop "
    "sequence applies as on a run without thinking")
#: The ARE paper's agent protocol (gaia2_run.py applies it in the hooked ARE process).
PAPER_MAX_ITERATIONS = 200
PAPER_MAX_TOKENS = 16000
#: LiteLLM's default request timeout (600 s), for the paper's 16000-token turn.
LITELLM_DEFAULT_TIMEOUT_S = 600
#: ARE's scenario runs per scenario in both modes (scenario_executor.run_dataset num_runs=3).
ARE_DEFAULT_NUM_RUNS = 3


#: What to check for each harness-error kind (the remediation an infra_required message gives).
HARNESS_ERROR_REMEDIATION = {
    "model_endpoint": "check that the model endpoint stays up and reachable from the docker "
                      "bridge network",
    "server_parser_text_lost": "check that the endpoint returns token ids (return_token_ids) and "
                               "serves /detokenize, or pass --tokenizer",
    "judge": "check GEMINI_API_KEY and the Gemini judge cascade (GBENCH_JUDGE_MODELS)",
    "judge_no_verdict": "check the Gemini judge cascade (GBENCH_JUDGE_MODELS): its replies held "
                        "no verdict",
    "validation_exception": "check the orchestrator log (ARE's validation raised)",
    "run_exception": "check the orchestrator log (the scenario run raised)",
}


def harness_remediation(kinds: Dict[str, Any]) -> str:
    """What to check for the harness errors in `kinds` ({kind: runs}), in order; the generic
    endpoint and judge advice when there are none."""
    labels = [k for k, _ in HARNESS_ERROR_LABELS] + sorted(k for k in kinds
                                                          if k not in HARNESS_ERROR_REMEDIATION)
    steps = [HARNESS_ERROR_REMEDIATION.get(k, f"check the orchestrator log ({k})")
             for k in labels if kinds.get(k)]
    if not steps:
        steps = ["check the model endpoint reachability and the Gemini judge cascade / "
                 "GEMINI_API_KEY"]
    return "; ".join(dict.fromkeys(steps))


def explicit_limit_env() -> str:
    """GBENCH_GAIA2_LIMIT, checked: a positive integer, or "" when unset. The launcher reads 0 or
    a negative value as no limit and stops on a non-integer, which would run the full benchmark
    or fail after the container started, so anything else is refused before any container."""
    value = os.environ.get("GBENCH_GAIA2_LIMIT", "").strip()
    if value and not (value.isdigit() and int(value) > 0):
        raise ValueError(f"gaia2: GBENCH_GAIA2_LIMIT={value!r} must be a positive integer "
                         f"(scenarios per capability config); unset it for no limit. "
                         f"See {DOCS_URL}")
    return value


def unscored_capabilities_as_none(per_capability: Any) -> Any:
    """ARE's per_capability statistics with the success_rate of each capability that has no
    validated run set to None (ARE's formula gives it 0.0, kept as success_rate_are_formula), as
    _merge_statistics reports it; anything that is not that dict comes back as it is."""
    if not isinstance(per_capability, dict):
        return per_capability
    out = {}
    for cap, row in per_capability.items():
        if isinstance(row, dict) and not row.get("validated_runs") and "success_rate" in row:
            row = dict(row, success_rate=None, success_rate_are_formula=row.get("success_rate"))
        out[cap] = row
    return out


def run_diagnostics(summary: Dict[str, Any], keep_dir: Optional[str]) -> str:
    """The run's own diagnostics for an infra_required message: the capability configs that
    raised (capability: error) and where the run's artifacts were kept."""
    parts = []
    failed = [f"{c.get('capability')}: {str(c.get('error') or 'raised')[:200]}"
              for c in summary.get("failed_configs") or [] if isinstance(c, dict)]
    if failed:
        parts.append("failed configs: " + "; ".join(failed))
    if keep_dir:
        parts.append(f"artifacts kept in {keep_dir}")
    return (" (" + "; ".join(parts) + ")") if parts else ""


def describe_harness_errors(kinds: Dict[str, Any]) -> str:
    """'<n> <label>, ...' for the summary's harness_error_kinds ({kind: runs})."""
    labels = dict(HARNESS_ERROR_LABELS)
    order = [k for k, _ in HARNESS_ERROR_LABELS] + sorted(k for k in kinds if k not in labels)
    return ", ".join(f"{kinds[k]} {labels.get(k, k)}" for k in order if kinds.get(k))


def _image() -> str:
    return os.environ.get("GBENCH_GAIA2_IMAGE", _IMAGE_DEFAULT)


def _docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def _image_accounts_harness_errors(image: str) -> bool:
    """True when the orchestrator image carries HARNESS_ERRORS_LABEL=HARNESS_ERRORS_VERSION."""
    r = subprocess.run(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "' + HARNESS_ERRORS_LABEL + '"}}', image],
                       capture_output=True, text=True)
    return r.returncode == 0 and (r.stdout or "").strip() == HARNESS_ERRORS_VERSION


# Live Gemini key ping lives in base.py (single source shared by every judge suite); the alias
# keeps the module-global name that this suite's prerequisite check and the tests reference.
from .base import (gemini_key_live_valid as _gemini_key_valid, gemini_openai_base_url,  # noqa: E402
                   require_gemini_openai_base_url, redact_text, redact_file, judge_cascade,
                   SHARD_REASON, add_comparability_reason, current_shard, endpoint_api_key,
                   get_run_knob, record_shard,
                   gemini_api_keys, gemini_cascade_env, MODELS_NOT_SERVED_PREFIX, OfficialSampling,
                   resolve_official_sampling, resolve_token_budget, REASONING_CAP_NOTE,
                   server_parser_note)

#: The ARE paper's agent sampling (arXiv:2509.17158 section 4.1): temperature 0.5 and at most
#: 16,000 tokens per turn; the agent's stop sequences go to the server as `stop`.
GAIA2_SAMPLING = OfficialSampling(
    reference=("ARE paper arXiv:2509.17158 section 4.1: temperature 0.5, generation limit of "
               "16,000 tokens per turn, 200 steps; are/simulation/agents/default_agent/"
               "base_agent.py step: stop = the agent's stop sequences <end_action> and "
               "Observation:, passed to the engine on every step (appendix B.5 adds them as "
               "custom stop sequences for models that tend to continue past a single tool call, "
               "naming Claude, Kimi and Qwen, and ran GPT-5 without them); sent for every served "
               "model as ARE's HuggingFaceLLMEngine sends them; with --thinking applied to the "
               "streamed answer"),
    temperature=0.5, params={"stop": list(AGENT_STOP_SEQUENCES)}, max_tokens=PAPER_MAX_TOKENS)


def agent_timeout_s(max_tokens: Optional[int]) -> int:
    """The agent's request timeout for a per-turn budget (gaia2_run.agent_timeout_s): LiteLLM's
    600 s for the paper's 16000 tokens, scaled with a larger --thinking budget."""
    budget = max(int(max_tokens or PAPER_MAX_TOKENS), PAPER_MAX_TOKENS)
    return int(math.ceil(LITELLM_DEFAULT_TIMEOUT_S * budget / PAPER_MAX_TOKENS))


class DetokenizeRelay:
    """gbench's token-id decoder for the launcher's server-parser policy: an HTTP listener on the
    docker network's gateway that answers POST /detokenize {"tokens": [int, ...]} with
    {"text": <the ids decoded with special tokens kept, or null>}, decoded by `decode`
    (base.decode_generated_token_ids: the served tokenizer's /detokenize, else the run's
    tokenizer). It holds no credential and forwards nothing else; the endpoint's /detokenize is
    reached from this process, as base reaches it."""

    #: The largest request body accepted (a token id list; gaia2_run.DETOKENIZE_MAX_IDS ids).
    MAX_BODY = 4 * 1024 * 1024

    def __init__(self, bind_host: str, decode: Optional[Any] = None,
                 name: str = "gaia2-detokenize"):
        self.bind_host, self.name = bind_host, name
        self._decode = decode
        self._server = None
        self._thread = None
        self.counts = {"requests": 0, "decoded": 0, "undecodable": 0, "refused": 0}

    def _decoder(self):
        if self._decode is not None:
            return self._decode
        from .base import decode_generated_token_ids
        return decode_generated_token_ids

    def start(self) -> "DetokenizeRelay":
        import http.server
        import threading
        relay = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D401 - quiet
                pass

            def _reply(self, code, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802 - http.server API
                relay.counts["requests"] += 1
                try:
                    size = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    size = -1
                if self.path != "/detokenize" or size <= 0 or size > relay.MAX_BODY:
                    relay.counts["refused"] += 1
                    return self._reply(404 if self.path != "/detokenize" else 413,
                                       {"text": None})
                try:
                    data = json.loads(self.rfile.read(size))
                    ids = [int(t) for t in data["tokens"]]
                    if any(isinstance(t, bool) for t in data["tokens"]):
                        raise ValueError("booleans are not token ids")
                except Exception:                                           # noqa: BLE001
                    relay.counts["refused"] += 1
                    return self._reply(400, {"text": None})
                try:
                    text = relay._decoder()(ids)
                except Exception:                                           # noqa: BLE001
                    text = None
                if isinstance(text, str):
                    relay.counts["decoded"] += 1
                else:
                    text = None
                    relay.counts["undecodable"] += 1
                return self._reply(200, {"text": text})

            def do_GET(self):  # noqa: N802
                relay.counts["refused"] += 1
                return self._reply(404, {"text": None})

        self._server = http.server.ThreadingHTTPServer((self.bind_host, 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name=self.name,
                                        daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        port = self._server.server_address[1] if self._server is not None else 0
        return f"http://{self.bind_host}:{port}/detokenize"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def record(self) -> Dict[str, Any]:
        return {"listens": self.url, **self.counts}


def check_gaia2_prerequisites() -> Tuple[bool, str]:
    """Docker + orchestrator image + GEMINI (the load-bearing judge). ARE + the dataset are baked."""
    image = _image()
    build = (f"Build the orchestrator LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/gaia2.Dockerfile gbench/docker\n"
             f"It bundles the pinned ARE simulator + the public GAIA2 dataset. See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if not _docker_image_exists(image):
        return False, f"orchestrator image {image!r} not found. " + build
    if not _image_accounts_harness_errors(image):
        return False, (f"orchestrator image {image!r} predates the current launcher (label "
                       f"{HARNESS_ERRORS_LABEL}={HARNESS_ERRORS_VERSION}): it would run the agent "
                       f"without the paper's protocol (or cut --thinking replies at a stop string "
                       f"inside the reasoning), misread judge verdicts, score harness errors as "
                       f"the model's failures or leave a capability ARE skipped unnamed. Rebuild "
                       f"it. " + build)
    keys = gemini_api_keys()
    if not keys:
        return False, ("GEMINI_API_KEY (or GEMINI_API_KEYS) is unset - GAIA2's soft/semantic "
                       "validation is done by an LLM judge (load-bearing; without it most "
                       "scenarios score no_validation). gbench swaps in its Gemini cascade. See "
                       + DOCS_URL)
    # The Gemini relay sends the key to GEMINI_OPENAI_BASE_URL: only the Gemini API or a loopback
    # address may receive it, whether or not the key is pinged.
    gemini_base, why_base = gemini_openai_base_url()
    if gemini_base is None:
        return False, why_base + ". See " + DOCS_URL
    if not os.environ.get("GBENCH_GAIA2_SKIP_KEY_VALIDATION"):
        ok_g, why_g = _gemini_key_valid(",".join(keys))
        if not ok_g:
            if str(why_g).startswith(MODELS_NOT_SERVED_PREFIX):
                return False, (f"{redact_text(why_g)}; every model of the Gemini cascade must be "
                               f"one the API serves for the judge to run. See " + DOCS_URL)
            return False, (f"GEMINI_API_KEY was rejected by the Gemini API ({redact_text(why_g)}); "
                           f"the judge cascade cannot run. Check the key. See " + DOCS_URL)
    return True, ""


def _task_reachable_endpoint(base_url: str) -> str:
    """The model /v1 endpoint gbench's endpoint relay forwards the agent's requests to (as this
    host reaches it; a /v1 suffix ensured). GBENCH_GAIA2_MODEL_ENDPOINT overrides the whole URL."""
    override = os.environ.get("GBENCH_GAIA2_MODEL_ENDPOINT")
    if override:
        return override
    ep = (base_url or "").rstrip("/")
    return ep if ep.endswith("/v1") else ep + "/v1"


def capability_key(config: str, phase_name: str = "standard") -> str:
    """The key ARE's statistics give a (config, phase) group (gaia2_run.capability_key)."""
    config = str(config)
    return f"{config}_{phase_name or 'standard'}" if config == "mini" else config


def expected_capabilities(mode: str, single_config: str,
                          summary: Optional[Dict[str, Any]] = None) -> List[str]:
    """The capability groups the run covers: gaia2-run's 5 standard configs and its 2 mini
    groups, a single GBENCH_GAIA2_CONFIG, else the 5 configs of `run`; plus any other group the
    launcher recorded ARE running (`configs_attempted`)."""
    if mode == "gaia2-run":
        keys = list(_CAPABILITY_CONFIGS) + list(_GAIA2_RUN_EXTRA_CAPABILITIES)
    elif single_config:
        keys = [capability_key(single_config)]
    else:
        keys = list(_CAPABILITY_CONFIGS)
    for c in (summary or {}).get("configs_attempted") or []:
        key = (c or {}).get("capability") if isinstance(c, dict) else None
        if key and key not in keys:
            keys.append(str(key))
    return keys


def missing_capabilities(summary: Dict[str, Any], expected: List[str]) -> List[Dict[str, str]]:
    """[{capability, why}] for each expected capability without a validated run outside harness
    errors: its config raised in ARE (ARE skipped it and reported the others), ARE never ran it,
    every run of it was a harness error, or none of its runs could be validated."""
    per = summary.get("per_capability") or {}
    failed = {str(c.get("capability")): c for c in summary.get("failed_configs") or []
              if isinstance(c, dict)}
    attempted = {str(c.get("capability")) for c in summary.get("configs_attempted") or []
                 if isinstance(c, dict)}
    out = []
    for key in expected:
        if ((per.get(key) or {}).get("validated_runs") or 0) > 0:
            continue
        if key in failed:
            c = failed[key]
            why = (f"its config raised in ARE ({c.get('error')}: {c.get('message')}), which "
                   "skipped it and reported the other capabilities")
        elif key not in attempted:
            why = "ARE did not run it"
        elif key not in per:
            why = "every scenario run of it hit a harness error"
        else:
            why = "none of its scenario runs could be validated"
        out.append({"capability": key, "why": why})
    return out


def _agent_never_invoked(summary: Dict[str, Any]) -> bool:
    """True iff the run validated scenarios but the container positively reported that the agent was
    never invoked (agent-less `are-benchmark run`: no agent LLM usage in any per-run trace). Older
    summaries omit `agent_invoked` (-> None), so this returns False for them and never false-fires."""
    return bool(summary) and (summary.get("validated_runs") or 0) > 0 \
        and summary.get("agent_invoked") is False


def agent_never_answered(summary: Dict[str, Any]) -> bool:
    """True when the launcher's agent call record (`agent_calls`, the calls that got a reply)
    counts none. A summary without the record says nothing (False)."""
    calls = summary.get("agent_calls")
    return isinstance(calls, dict) and "calls" in calls and not calls.get("calls")


def compute_gaia2_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Headline = macro (per-capability equal-weight) success rate; else micro if macro absent.
    Values are as ARE's benchmark_stats.json writes them: percent (0-100)."""
    if not summary:
        return None
    macro = summary.get("macro_success_rate")
    micro = summary.get("micro_success_rate")
    head = macro if isinstance(macro, (int, float)) else micro
    if not isinstance(head, (int, float)):
        return None
    return {"headline": head, "macro_success_rate": macro, "micro_success_rate": micro}


def gemini_substitutions() -> list:
    """What the Gemini API serves in place of ARE's official external model, as recorded on every
    result."""
    return [{
        "component": "ARE LLM judge (the soft/semantic validation of every scenario)",
        "official": ARE_DEFAULT_JUDGE,
        "served_by": ("gbench Gemini judge cascade through Gemini's OpenAI-compatible chat endpoint, "
                      "behind the in-container proxy (--judge_provider local --judge_endpoint "
                      "http://127.0.0.1:<port>/v1 --judge_model openai/gbench-cascade) and "
                      "gbench's Gemini relay; ARE's judge prompts unchanged, the requested model "
                      "ignored, temperature 0 (the paper's verifier setting)"),
        "models": list(judge_cascade()),
    }]


def gemini_models_answered(usage: Dict[str, Any]) -> Dict[str, int]:
    """{model: judge replies} from the proxy's usage tally (its `failed` count left out)."""
    judge = (usage or {}).get("judge") or {}
    return {str(m): int(n) for m, n in judge.items()
            if m != "failed" and isinstance(n, (int, float)) and not isinstance(n, bool)}


#: Files of the workdir kept with the run's results (all small; ARE's per-scenario traces are not).
ARTIFACT_FILES = ("gaia2_summary.json", "gbench_stats.json", "benchmark_stats.json",
                  "output.jsonl", "gbench_configs.jsonl", "gbench_llm_call_failures.jsonl",
                  "gbench_agent_calls.jsonl",
                  "gbench_judge_verdicts.jsonl", "judge_calls.jsonl", "gemini_usage.json",
                  "harness_events.jsonl", "cascade_judge.log")


def artifacts_dir() -> Optional[str]:
    """`<GBENCH_RESULTS_DIR>/gaia2_artifacts/<fresh dir>`, or None outside a gbench run."""
    run_dir = (os.environ.get("GBENCH_RESULTS_DIR") or "").strip()
    if not run_dir:
        return None
    base_dir = os.path.join(run_dir, "gaia2_artifacts")
    os.makedirs(base_dir, exist_ok=True)
    return tempfile.mkdtemp(prefix="run_", dir=base_dir)


def keep_artifacts(workdir: str, dest: Optional[str], orchestrator_log: str) -> Optional[str]:
    """Copy ARTIFACT_FILES from `workdir` and the orchestrator's output to `dest`, every credential
    value of the environment redacted. Returns `dest` (None when there is none)."""
    if not dest:
        return None
    for name in ARTIFACT_FILES:
        src = os.path.join(workdir, name)
        if os.path.isfile(src):
            try:
                shutil.copyfile(src, os.path.join(dest, name))
                redact_file(os.path.join(dest, name))
            except OSError as e:
                logger.warning("gaia2: could not keep %s (%s)", name, e)
    with open(os.path.join(dest, "gaia2_orchestrator.log"), "w", encoding="utf-8") as f:
        f.write(redact_text(orchestrator_log or ""))
    return dest


def remove_workdir(workdir: str) -> List[str]:
    """Remove the run's workdir; the paths that could not be removed (logged)."""
    left: List[str] = []

    def _onerror(_func, path, _exc):
        left.append(path)
    shutil.rmtree(workdir, onerror=_onerror)
    if left:
        logger.warning("gaia2: %d path(s) of the workdir %s could not be removed (e.g. %s)",
                       len(left), workdir, left[0])
    return left


def run_configs(mode: str, single_config: str) -> List[str]:
    """The scenario configs a run loads (gaia2_run.run_configs): gaia2-run's 5 capability configs
    and the mini config of its Agent2Agent and Noise phases, a single GBENCH_GAIA2_CONFIG, else
    the 5 capability configs of `run`."""
    if mode == "gaia2-run":
        return list(_CAPABILITY_CONFIGS) + ["mini"]
    return [single_config] if single_config else list(_CAPABILITY_CONFIGS)


def config_limit_shares(limit: Optional[int], configs: List[str]) -> Optional[Dict[str, int]]:
    """{config: scenarios it runs} for --eval-limit `limit`: N scenarios for the whole run, dealt
    over the configs by sampling.allocate_limit (seed "gaia2"), so a config gets 0 when N is below
    the number of configs and is then not run. None without a limit (every scenario)."""
    if not limit or int(limit) <= 0:
        return None
    from .sampling import allocate_limit
    return {c: int(n or 0) for c, n in allocate_limit(int(limit), configs, seed="gaia2").items()}


def limit_skipped_capabilities(config_limits: Optional[Dict[str, int]]) -> List[str]:
    """The capability keys of the configs --eval-limit gave no scenario (the mini config's are
    its Agent2Agent and Noise groups): not run, so not expected in the macro."""
    out: List[str] = []
    for config, n in (config_limits or {}).items():
        if n == 0:
            out.extend(_GAIA2_RUN_EXTRA_CAPABILITIES if config == "mini"
                       else [capability_key(config)])
    return out


def limit_plan(limit: Optional[int], config_limits: Optional[Dict[str, int]], mode: str,
               single_config: str, num_runs: int, knob_limit: Optional[int] = None
               ) -> Dict[str, Any]:
    """Log the run's planned workload (one line) and return its `eval_limit_plan`: the scenarios
    per config x ARE's runs per scenario (repeats, not extra samples). In gaia2-run each mini
    scenario runs in both the Agent2Agent and the Noise phase (`phase_runs_per_mini_scenario`),
    which the line counts too. `knob_limit`: GBENCH_GAIA2_LIMIT when it applies (no --eval-limit,
    `run` mode): that many scenarios of each config, at most (a config holding fewer runs all it
    has). A full run's scenario counts are ARE's own (the dataset in the image), not known here."""
    configs = run_configs(mode, single_config)
    if config_limits is None and knob_limit and mode != "gaia2-run":
        per = {c: int(knob_limit) for c in configs}
        n = sum(per.values())
        logger.info("[gaia2] GBENCH_GAIA2_LIMIT %d per config (no --eval-limit): at most %d "
                    "sample(s) over %d config(s) (%s) x %d attempt%s = at most %d scenario_runs",
                    int(knob_limit), n, len(per), ", ".join(f"{c} {v}" for c, v in per.items()),
                    num_runs, "s" if num_runs != 1 else "", n * num_runs)
        return {"limit": None, "limit_source": f"GBENCH_GAIA2_LIMIT={int(knob_limit)} per config",
                "samples": n, "per_setting": per, "attempts_per_sample": num_runs,
                "scenario_runs": n * num_runs}
    if config_limits is None:
        logger.info("[gaia2] full run (no --eval-limit): every scenario of %s x %d run(s)%s",
                    ", ".join(configs), num_runs,
                    " (each mini scenario in both the Agent2Agent and the Noise phase)"
                    if mode == "gaia2-run" else "")
        return {"limit": None, "samples": None, "per_setting": None,
                "attempts_per_sample": num_runs, "scenario_runs": None}
    from .limit_plan import eval_limit_plan
    n = sum(config_limits.values())
    mini = int(config_limits.get("mini") or 0) if mode == "gaia2-run" else 0
    unit = "scenario_runs"
    if mini:
        # the mini scenarios' second phase (Noise after Agent2Agent), in the logged total too
        unit = (f"scenario_runs + {mini * num_runs} (the mini scenario(s)' second phase) = "
                f"{(n + mini) * num_runs} scenario_runs")
    plan = eval_limit_plan("gaia2", limit, n, config_limits,
                           setting_noun="config", attempts=num_runs, unit=unit)
    if mini:
        plan.pop(unit, None)
        plan["phase_runs_per_mini_scenario"] = 2
        plan["scenario_runs"] = (n + mini) * num_runs
    return plan


def resolve_num_runs(mode: str, attempts_requested: int) -> Tuple[int, str]:
    """(ARE's runs per scenario, where the number comes from). In `run`: GBENCH_GAIA2_NUM_RUNS,
    else --attempt-count above ARE's 3, else 3. In gaia2-run always 3: ARE's gaia2_submission
    hard-codes NUM_RUNS = 3 for every phase (its --num_runs never reaches the phases), so another
    number would be recorded but not run; a request for one is ignored with a warning."""
    raw = os.environ.get("GBENCH_GAIA2_NUM_RUNS", "").strip()
    if mode == "gaia2-run":
        if raw or attempts_requested > ARE_DEFAULT_NUM_RUNS:
            logger.warning("gaia2: gaia2-run always runs each scenario %d times (ARE's "
                           "gaia2_submission NUM_RUNS); %s is ignored", ARE_DEFAULT_NUM_RUNS,
                           f"GBENCH_GAIA2_NUM_RUNS={raw}" if raw
                           else f"--attempt-count {attempts_requested}")
        return ARE_DEFAULT_NUM_RUNS, "gaia2-run's NUM_RUNS (ARE's gaia2_submission: 3 per phase)"
    if raw:
        return int(raw), "GBENCH_GAIA2_NUM_RUNS"
    if attempts_requested > ARE_DEFAULT_NUM_RUNS:
        return attempts_requested, "--attempt-count (above ARE's 3 runs)"
    return ARE_DEFAULT_NUM_RUNS, "ARE's default (3 runs per scenario)"


def run_gaia2(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical GAIA2 via the pinned Meta-ARE `are-benchmark` CLI (in-container, no DooD)."""
    ok, reason = check_gaia2_prerequisites()
    if not ok:
        raise infra_required("gaia2", reason, DOCS_URL)

    # --shard I/N: validated here (a malformed GBENCH_SHARD raises before anything starts) and run
    # by the launcher's dataset hook in the container (see the module docstring).
    shard_spec = current_shard()
    shard_label = shard_spec.label if shard_spec and shard_spec.partial else None

    # The judge proxy listens on 127.0.0.1 inside the container (the bridge network gives the
    # container its own network namespace, so no host port is taken): the fixed default unless
    # GBENCH_GAIA2_JUDGE_PORT sets another.
    _env_judge_port = os.environ.get("GBENCH_GAIA2_JUDGE_PORT")
    judge_port = int(_env_judge_port) if _env_judge_port else _JUDGE_PORT_DEFAULT
    model_endpoint = _task_reachable_endpoint(base_url)
    mode = os.environ.get("GBENCH_GAIA2_MODE", "run").strip()          # run | gaia2-run
    single_config = os.environ.get("GBENCH_GAIA2_CONFIG", "").strip()  # narrows `run` (smoke: mini/demo)
    _explicit_limit = explicit_limit_env()
    if limit and _explicit_limit:
        # --eval-limit N is N samples for the whole suite, whatever else is set: the operator's
        # per-config knob would run N per config (or, in gaia2-run, the full benchmark).
        logger.warning("gaia2: GBENCH_GAIA2_LIMIT=%s is ignored because --eval-limit %s is set "
                       "(--eval-limit is the suite's total).", _explicit_limit, limit)
        _explicit_limit = ""
    eff_limit = _explicit_limit
    # --eval-limit N is N scenarios for the whole run, never N per capability config (ARE's
    # --limit applies per config, so passing N on would run 5xN): config_limit_shares deals N over
    # the run's configs and the launcher draws each config's share (a config with 0 is not run).
    # GBENCH_GAIA2_LIMIT, the operator's per-config knob, applies (`run` mode) only without
    # --eval-limit (it was cleared above when --eval-limit is set).
    config_limits = (config_limit_shares(limit, run_configs(mode, single_config))
                     if limit and not _explicit_limit else None)
    if _explicit_limit and mode == "gaia2-run":
        # gaia2_run.py only applies GBENCH_GAIA2_LIMIT in the default `run` mode; the canonical
        # 3-phase x3 `gaia2-run` builds its command without --limit, so the cap would be
        # silently dropped. Warn loudly (do not present a full run as a capped one).
        logger.warning(
            "gaia2: GBENCH_GAIA2_LIMIT (%s) is NOT applied in gaia2-run mode (the canonical "
            "3-phase x3 run takes no per-config cap) - the FULL benchmark will run. Use "
            "--eval-limit (applied in both modes) or the default `run` mode.", eff_limit)
    attempts_requested = max(1, int(kwargs.get("attempt_count")
                                    or get_run_knob("attempt_count", 1) or 1))
    num_runs, num_runs_source = resolve_num_runs(mode, attempts_requested)
    num_runs_raw = (os.environ.get("GBENCH_GAIA2_NUM_RUNS", "").strip()
                    if mode != "gaia2-run" else "")
    # The scenarios ARE runs at once: GBENCH_GAIA2_MAX_CONCURRENT, else the suite's concurrency.
    max_concurrent_raw = os.environ.get("GBENCH_GAIA2_MAX_CONCURRENT", "").strip()
    max_concurrent = int(max_concurrent_raw) if max_concurrent_raw else max(1, int(concurrency))
    max_concurrent_source = ("GBENCH_GAIA2_MAX_CONCURRENT" if max_concurrent_raw
                             else "the suite's concurrency (--eval-concurrency / --batch-sizes)")
    workload = limit_plan(limit, config_limits, mode, single_config, num_runs,
                          knob_limit=int(eff_limit) if eff_limit else None)

    for knob in _REMOVED_KNOBS:
        if os.environ.get(knob) is not None:
            logger.warning("gaia2: %s no longer changes anything: ARE's judge is always the gbench "
                           "Gemini cascade behind the in-container proxy; see %s", knob, DOCS_URL)
    require_gemini_openai_base_url("gaia2", DOCS_URL)

    # The agent protocol: temperature 1.0 with --thinking, 0.0 without (the operator's
    # GBENCH_GAIA2_TEMPERATURE or --temperature wins), the token-limit policy for the per-turn cap, the run's thinking switch,
    # the 200-step limit and a request timeout sized to the cap.
    temperature, temperature_source = resolve_official_sampling(
        "gaia2", GAIA2_SAMPLING, thinking=enable_thinking)
    plan = resolve_token_budget("gaia2", GAIA2_SAMPLING, thinking=enable_thinking,
                                max_output_tokens=kwargs.get("max_output_tokens"))
    turn_tokens = int(plan.max_tokens or PAPER_MAX_TOKENS)
    request_timeout = agent_timeout_s(turn_tokens)

    # The agent's tool calls run in the orchestrator: it starts only on a network where the cloud
    # metadata server is blocked.
    container_guard.require_metadata_blocked("gaia2", _image(), NETWORK, DOCS_URL,
                                             resolve=DNS_NAMES)
    gateway = gemini_relay.network_gateway("gaia2", NETWORK, DOCS_URL)

    workdir = tempfile.mkdtemp(prefix="gbench_gaia2_")
    os.chmod(workdir, 0o777)
    orch_name = "gbench_gaia2_orch_" + os.path.basename(workdir)
    keep_dir = None

    def _reap():
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    judge_relay = model_relay = detok = None
    proc = None
    try:
        judge_relay = gemini_relay.GeminiRelay(
            "gaia2", gateway, docs_url=DOCS_URL, openai_calls=(("POST", "/chat/completions"),),
            native_methods=(), name="gaia2-judge-relay").start()
        # The endpoint's key: GBENCH_GAIA2_MODEL_API_KEY, else the one every pillar sends
        # (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY); a placeholder is no key.
        model_key = (endpoint_api_key({"OPENAI_API_KEY": os.environ.get(
            "GBENCH_GAIA2_MODEL_API_KEY") or ""}) or endpoint_api_key())
        try:
            model_relay = EndpointRelay(model_endpoint, gateway, key=model_key,
                                        name="gaia2-model-relay")
        except ValueError as e:
            raise infra_required("gaia2", f"the model endpoint is not an http(s) URL with a host "
                                 f"({e}); set GBENCH_GAIA2_MODEL_ENDPOINT.", DOCS_URL) from e
        model_relay.TIMEOUT_S = max(EndpointRelay.TIMEOUT_S, request_timeout + 120)
        try:
            model_relay.start()
        except OSError as e:
            raise infra_required("gaia2", f"gbench's relay to the model endpoint could not listen "
                                 f"on {gateway} ({type(e).__name__}: {e}).", DOCS_URL) from e
        # The server-parser policy's decoder (the launcher sends it an agent reply's token ids).
        try:
            detok = DetokenizeRelay(gateway).start()
        except OSError as e:
            raise infra_required("gaia2", f"gbench's token-id decoder could not listen on "
                                 f"{gateway} ({type(e).__name__}: {e}).", DOCS_URL) from e
        # Credentials go in by NAME: Docker copies each value from run_env, so nothing is on a
        # command line; the only credential in the container is the Gemini relay's placeholder.
        run_env = {k: v for k, v in os.environ.items() if k not in _CREDENTIAL_VARS}
        run_env.update(judge_relay.client_env())
        cascade_env = gemini_cascade_env()
        cmd = ["docker", "run", "--rm", "--pull", "never", "--name", orch_name,
               "--network", NETWORK,
               "-v", f"{workdir}:{workdir}:rw",
               "-e", f"GAIA2_WORKDIR={workdir}",
               "-e", f"GBENCH_MODEL_BASE_URL={model_relay.url}",
               "-e", f"GBENCH_DETOKENIZE_URL={detok.url}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               "-e", "GEMINI_API_KEY", "-e", "GEMINI_API_KEYS", "-e", "GEMINI_OPENAI_BASE_URL",
               "-e", gemini_relay.RELAY_URL_ENV,
               "-e", f"GBENCH_JUDGE_MODELS={cascade_env['GBENCH_JUDGE_MODELS']}",
               "-e", f"GAIA2_JUDGE_PORT={judge_port}",
               "-e", f"GAIA2_MODE={mode}",
               "-e", f"GAIA2_AGENT_TEMPERATURE={temperature}",
               "-e", f"GAIA2_AGENT_MAX_TOKENS={turn_tokens}",
               "-e", f"GAIA2_ENABLE_THINKING={'true' if enable_thinking else 'false'}",
               "-e", f"GAIA2_AGENT_TIMEOUT_S={request_timeout}",
               "-e", f"GAIA2_MAX_ITERATIONS={PAPER_MAX_ITERATIONS}",
               "-e", f"GAIA2_HOST_UID={os.getuid()}", "-e", f"GAIA2_HOST_GID={os.getgid()}"]
        if mode != "gaia2-run" and (num_runs != ARE_DEFAULT_NUM_RUNS or num_runs_raw):
            cmd += ["-e", f"GAIA2_NUM_RUNS={num_runs}"]
        cmd += ["-e", f"GAIA2_MAX_CONCURRENT={max_concurrent}"]
        if os.environ.get("GBENCH_GAIA2_SCENARIO_TIMEOUT"):
            cmd += ["-e", f"GAIA2_SCENARIO_TIMEOUT={os.environ['GBENCH_GAIA2_SCENARIO_TIMEOUT']}"]
        if shard_label:
            cmd += ["-e", f"GAIA2_SHARD={shard_label}"]
        for k in ("GBENCH_JUDGE_CASCADE_ROUNDS", "GBENCH_JUDGE_BACKOFF"):
            if os.environ.get(k):
                cmd += ["-e", f"{k}={os.environ[k]}"]
        if single_config:
            cmd += ["-e", f"GAIA2_CONFIG={single_config}"]
        if config_limits is not None:
            cmd += ["-e", f"GAIA2_CONFIG_LIMITS={json.dumps(config_limits, sort_keys=True)}"]
        elif eff_limit:
            cmd += ["-e", f"GAIA2_LIMIT={eff_limit}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_GAIA2_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("gaia2: docker run %s (mode=%s, config=%s)", _image(), mode, single_config or "all-5-capabilities")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                                  env=run_env)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "gaia2", f"the orchestrator exceeded GBENCH_GAIA2_TIMEOUT_S ({timeout_s}s) and was "
                f"killed. Raise the timeout or reduce the scope (GBENCH_GAIA2_CONFIG / GBENCH_GAIA2_LIMIT).",
                DOCS_URL) from exc
        orchestrator_log = ((proc.stdout or "") + ("\n" if proc.stdout else "")
                            + (proc.stderr or ""))
        keep_dir = keep_artifacts(workdir, artifacts_dir(), orchestrator_log)

        summ_path = os.path.join(workdir, "gaia2_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "gaia2", f"the orchestrator produced no summary (rc={proc.returncode}; a harness "
                f"failure, not a 0%). tail: "
                f"{redact_text((proc.stderr or proc.stdout or '')[-800:])}", DOCS_URL)
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
    finally:
        _reap()
        for relay in (judge_relay, model_relay, detok):
            if relay is not None:
                relay.stop()
        # Preserve the workdir (ARE per-scenario logs/traces) for audit when requested. By default
        # it is removed (the entrypoint hands every file to this user first); the small records
        # are kept in the results dir either way (keep_artifacts).
        if os.environ.get("GBENCH_GAIA2_KEEP_WORKDIR"):
            logger.info("[gaia2] GBENCH_GAIA2_KEEP_WORKDIR set: preserving %s for audit "
                        "(ARE per-scenario logs/traces).", workdir)
        else:
            remove_workdir(workdir)

    if summary.get("empty_shard"):
        return dict(_empty_shard_result(model_name, enable_thinking, shard_label or "?", summary),
                    eval_limit_plan=workload)
    if ((config_limits is not None or (eff_limit and mode == "run"))
            and not (summary.get("selection") or {}).get("limit_draw")):
        raise infra_required(
            "gaia2", "the orchestrator does not report how it drew the --eval-limit scenarios (an "
            "image built before the seeded draw, where ARE's --limit took the first scenarios of "
            "each config's stream); rebuild the image from gbench/docker/gaia2.Dockerfile.",
            DOCS_URL)

    # The model must actually have been invoked. `are-benchmark run` without --agent runs
    # agent-less and still returns a real macro=0.0; that is a harness no-op, not a 0% score, and
    # must never be published. The container reports agent_invoked=False only when it positively
    # found no agent LLM usage across the per-run traces (older summaries omit the key -> None ->
    # this guard stays silent, so it never false-fires on a legitimate run).
    if _agent_never_invoked(summary):
        raise infra_required(
            "gaia2", "the agent was never invoked (are-benchmark ran agent-less: no agent LLM usage "
            "in any per-run trace), so every scenario trivially failed. This is a harness bug, not a "
            "0% score. Ensure the run passes --agent (GAIA2_AGENT).", DOCS_URL)

    # Not one agent request got a reply: no request of the run carried text the model had
    # generated, so a rejection of a request's content (HTTP 400/422, otherwise the model's
    # outcome) cannot have come from the model either, and every run's outcome is a harness error.
    if agent_never_answered(summary):
        raise infra_required(
            "gaia2", "not one agent request of the run got a reply from the model endpoint (the "
            "launcher recorded no agent reply), so no request carried text the model had "
            "generated and no run's outcome can be the model's; no success rate is published. "
            "Check the model endpoint and the orchestrator log (gbench_llm_call_failures.jsonl "
            "names the failed calls).", DOCS_URL)

    harness_kinds = dict(summary.get("harness_error_kinds") or {})
    harness_text = describe_harness_errors(harness_kinds)
    if not summary.get("harness_accounting"):
        raise infra_required(
            "gaia2", "the orchestrator wrote no harness-error accounting (gbench_stats.json), so "
            "judge outages and endpoint failures cannot be told apart from the model's failures "
            "and no success rate is published. Rebuild the image from gbench/docker/gaia2.Dockerfile "
            "and check the orchestrator log.", DOCS_URL)
    unattributed = int(summary.get("unattributed_call_failures") or 0)
    if unattributed:
        roles = ", ".join(summary.get("unattributed_call_failure_roles") or []) or "unknown"
        raise infra_required(
            "gaia2", f"{unattributed} failed LLM call(s) ({roles}) could not be tied to a scenario "
            "run, so the runs they failed cannot be left out of the success rates; no success rate "
            "is published rather than one that scores harness failures as the model's"
            f"{run_diagnostics(summary, keep_dir)}. Check the model endpoint and the judge proxy "
            "(GEMINI_API_KEY).", DOCS_URL)
    if not summary.get("validated_runs"):
        raise infra_required(
            "gaia2", "no scenario run produced a validated score outside harness errors ("
            f"{summary.get('n_harness_error_runs') or 0} harness-error run(s)"
            f"{': ' + harness_text if harness_text else ''}; "
            f"{summary.get('no_validation_runs') or 0} run(s) ARE could not validate)"
            f"{run_diagnostics(summary, keep_dir)}. To fix: {harness_remediation(harness_kinds)}.",
            DOCS_URL)
    scored = compute_gaia2_score(summary)
    if scored is None:
        raise infra_required(
            "gaia2", "no scenarios produced a validated score (every run without a validation)"
            f"{run_diagnostics(summary, keep_dir)}. To fix: {harness_remediation(harness_kinds)}.",
            DOCS_URL)

    total_scenarios = summary.get("total_scenarios") or 0
    no_validation = summary.get("no_validation_runs") or 0
    is_full = (not eff_limit or mode == "gaia2-run") and config_limits is None and (
        mode == "gaia2-run" or not single_config)
    # Every capability the run covers must be in the macro. The launcher at this label records
    # each config ARE ran (`configs_attempted`); a capability whose config raised (ARE skipped it
    # and reported the rest with exit code 0), or without a validated run outside harness errors,
    # is left out of the macro and named here.
    expected = [k for k in expected_capabilities(mode, single_config, summary)
                if k not in limit_skipped_capabilities(config_limits)]
    missing = (missing_capabilities(summary, expected) if "configs_attempted" in summary
               else [])
    official = GAIA2_SAMPLING.record(temperature, temperature_source)
    official.update(plan.record())
    official["stop_sent_to_server"] = not enable_thinking
    agent_protocol = {
        "reference": GAIA2_SAMPLING.reference,
        "temperature": temperature, "temperature_source": temperature_source,
        "max_tokens_per_turn": turn_tokens, "stop": list(AGENT_STOP_SEQUENCES),
        "stop_sent_to_server": not enable_thinking,
        "stop_applied_by": "client_stream" if enable_thinking else "server",
        "stop_note": (STOP_NOTE_THINKING if enable_thinking else
                      "sent to the server as `stop` (ARE's HuggingFaceLLMEngine) and applied "
                      "client-side by ARE's split of the answer"),
        "stream_options": dict(STREAM_OPTIONS) if enable_thinking else None,
        "enable_thinking": bool(enable_thinking), "max_iterations": PAPER_MAX_ITERATIONS,
        "request_timeout_s": request_timeout,
        # metadata only: the generated token ids, for the server-parser policy
        "return_token_ids": True,
        "time_capability_clock": ("ARE's measured mode: the simulated clock advances by the "
                                  "client-measured generation time of each agent call, up to "
                                  "the end of its answer"
                                  + (" (the stop sequence that closed the streamed reply)"
                                     if enable_thinking else "")),
        "applied_by_image": summary.get("agent_protocol"),
    }
    if plan.policy == "reasoning_run_budget":
        tokens_note = plan.note
    else:
        knob = kwargs.get("max_output_tokens")
        tokens_note = ((f"--max-output-tokens {knob} not applied: " if knob and
                        int(knob) != PAPER_MAX_TOKENS else "")
                       + f"the ARE paper's limit of {PAPER_MAX_TOKENS} tokens per turn is sent "
                         "with every agent request")

    harness_events = dict(summary.get("harness_events") or {})
    judge_failures = int(harness_events.get("judge") or 0)
    leaderboard = False
    reasons = [f"the judge is the gbench Gemini cascade, not {ARE_DEFAULT_JUDGE}",
               (f"standard phase only ({num_runs} run(s) per scenario); the leaderboard runs the "
                "3-phase gaia2-run (Standard, Agent2Agent-mini, Noise-mini) with Pass@k")
               if mode != "gaia2-run" else "gbench single submission (leaderboard is maintainer-audited)",
               "the judge reply is read verbatim and its verdict matched case-insensitively, as "
               "the current upstream judge (gaia2-cli) does; ARE 1.2.0's engine lowercased "
               "True/False, so its signature, sanity, cab and tone checks could never pass"]
    if not is_full:
        if config_limits is not None:
            reasons.append(f"subset (--eval-limit {limit} scenarios for the run, dealt over the "
                           "configs: " + ", ".join(f"{c} {n}" for c, n in config_limits.items())
                           + f"; {summary.get('n_scenarios_run') or total_scenarios} scenario(s) "
                           "run)")
        else:
            reasons.append(f"subset (config={single_config or 'all'}, limit={eff_limit or 'none'} "
                           f"per config; {summary.get('n_scenarios_run') or total_scenarios} "
                           f"scenario(s) run)")
    if official.get("temperature_matches_official") is False:
        reasons.append(f"agent temperature {temperature} set by "
                       f"{temperature_source.split(':', 1)[-1]} (the ARE paper runs 0.5)")
    if plan.policy == "reasoning_run_budget":
        reasons.append(f"{REASONING_CAP_NOTE}: each agent turn was sent {turn_tokens} tokens "
                       f"(the paper's {PAPER_MAX_TOKENS} per turn were written for non-reasoning "
                       "generation)")
    # Harness errors: counted, named here, left out of the success rates, status
    # completed_with_errors.
    n_harness_runs = int(summary.get("n_harness_error_runs") or 0)
    n_harness_success = int(summary.get("n_harness_error_success_runs") or 0)
    if n_harness_runs:
        reasons.append(f"{n_harness_runs} scenario run(s) hit a harness error and are left out of "
                       f"the success rates ({harness_text})"
                       + (f"; {n_harness_success} of them ARE had validated as successes, left "
                          "out all the same (the error's effect on a run's outcome is unknowable)"
                          if n_harness_success else ""))
    if missing:
        reasons.append(
            f"the macro averages {len(expected) - len(missing)} of the run's {len(expected)} "
            "capabilities, not the GAIA2 Overall: "
            + "; ".join(f"{m['capability']}: {m['why']}" for m in missing))
    if no_validation:
        reasons.append(f"{no_validation} run(s) could not be validated (no_validation), harness "
                       f"errors that ARE leaves out of the success rates")
    if judge_failures:
        reasons.append(f"{judge_failures} judge call(s) were served by no Gemini cascade model "
                       f"(the proxy's harness events; a run whose validation such a call failed "
                       f"is left out of the success rates)")
    # The model's own replies that the Time capability's clock measured past the answer's end.
    agent_calls = dict(summary.get("agent_calls") or {})
    past_stop = int(agent_calls.get("ran_past_stop") or 0)
    if past_stop:
        reasons.append(f"{past_stop} agent repl(ies) still held a stop sequence in the answer: the "
                       "generation ran past the action (ARE's split dropped the rest) and the "
                       "Time capability's clock counted that generation")
    cut = int(agent_calls.get("cut_at_max_tokens") or 0)
    if enable_thinking and cut:
        reasons.append(f"{cut} agent repl(ies) reached this run's max_tokens ({turn_tokens} per "
                       f"turn; {int(agent_calls.get('cut_without_content') or 0)} of them with no "
                       "answer), and the Time capability's clock counted the whole generation; "
                       "raise --max-output-tokens")
    # The server-parser policy's events (the launcher recovered the text the server's parsers
    # took out of an agent reply, or named it lost): counted and named.
    parser_events = {str(k): int(v) for k, v in
                     (agent_calls.get("server_parser_events") or {}).items() if v}
    parser_sources = {str(k): int(v) for k, v in
                      (agent_calls.get("server_parser_recovered_from") or {}).items() if v}
    parser_note = server_parser_note(parser_events, parser_sources) if parser_events else None
    if parser_note:
        reasons.append(parser_note)
    hooks = dict(summary.get("hooks") or {})
    if hooks.get("token_ids_hook") is False:
        reasons.append("the launcher could not keep the token ids of non-streamed agent replies "
                       "(LiteLLM's conversion hook was not installed), so text the server's "
                       "tool-call parser deleted from them could not be recovered")
    has_harness_errors = bool(no_validation or judge_failures or n_harness_runs or missing)

    headline: Dict[str, Any] = {}
    for label, value in (("macro", scored.get("macro_success_rate")),
                         ("micro", scored.get("micro_success_rate")),
                         ("pass@k", summary.get("pass_at_k_percent")),
                         ("pass^k", summary.get("pass_k_percent"))):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            headline[label] = round(float(value), 2)
    runs = list(summary.get("runs") or [])
    gemini_usage = summary.get("gemini_usage") or {}

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "gaia2",
        "model_name": model_name,
        "status": "completed_with_errors" if has_harness_errors else "success",
        # ARE writes its success rates in percent: the headline is the macro rate as written, the
        # *_success_rate fields are the same rates as fractions.
        "accuracy": round(scored["headline"], 2),   # headline = macro success rate (%)
        "headline_metrics": headline,
        "macro_success_rate": (round(scored["macro_success_rate"] / 100.0, 4)
                               if isinstance(scored.get("macro_success_rate"), (int, float)) else None),
        "micro_success_rate": (round(scored["micro_success_rate"] / 100.0, 4)
                               if isinstance(scored.get("micro_success_rate"), (int, float)) else None),
        "pass_at_k_percent": summary.get("pass_at_k_percent"),
        "pass_k_percent": summary.get("pass_k_percent"),
        # The report's counts are scenario runs: every run, the runs scored (validated, no harness
        # error), and the successful ones among them.
        "count_unit": "scenario runs",
        "total_questions": int(summary.get("n_runs") or summary.get("total_runs") or 0),
        "effective_n": int(summary.get("validated_runs") or 0),
        "correct_answers": int(summary.get("success_runs") or 0),
        "total_scenarios": total_scenarios,
        "n_scenarios_run": summary.get("n_scenarios_run"),
        "num_runs_per_scenario": num_runs,
        "no_validation_runs": no_validation,
        "n_timeout_runs": int(summary.get("n_timeout_runs") or 0),
        # ARE's per-capability statistics; a capability without a validated run has no success
        # rate (None, never ARE's 0.0, which is kept as success_rate_are_formula)
        "per_capability": unscored_capabilities_as_none(summary.get("per_capability")),
        "judge_failures": judge_failures,
        "harness_events": harness_events,
        # Scenario runs left out of the success rates as harness errors (by kind, and each run),
        # and ARE's own rates with them counted as failures, for provenance.
        "n_harness_error_runs": n_harness_runs,
        "harness_error_kinds": harness_kinds,
        "harness_error_runs": list(summary.get("harness_error_runs") or []),
        # Of those, the runs ARE had validated as successes (by kind).
        "n_harness_error_success_runs": n_harness_success,
        "harness_error_success_kinds": dict(summary.get("harness_error_success_kinds") or {}),
        "are_reported": summary.get("are_reported") or {},
        # The capabilities the run covers, the ones the macro leaves out (with why), and every
        # capability config ARE ran with its outcome (the ones that raised in failed_configs).
        "expected_capabilities": expected,
        "missing_capabilities": missing,
        "failed_configs": list(summary.get("failed_configs") or []),
        "configs_attempted": summary.get("configs_attempted"),
        # One record per scenario run: ARE's status, exception and rationale, its harness errors,
        # its agent calls and its soft checks with the Gemini models that answered them.
        "sample_traces": runs,
        "agent_protocol": agent_protocol,
        "agent_calls": agent_calls,
        # The server-parser policy (base.recover_server_parser_text, applied by the launcher):
        # events by kind, where the text was recovered from, and the token-id decoder's record.
        "server_parser_events": parser_events,
        "server_parser_recovered_from": parser_sources,
        "server_parser_note": parser_note,
        "token_id_decoder": detok.record() if detok is not None else None,
        "launcher_hooks": hooks,
        # The dataset the image read (the Hub id, the pinned revisions of its local copies,
        # offline).
        "dataset": summary.get("dataset"),
        "judge_verdicts": summary.get("judge_verdicts") or {},
        "gemini_usage": gemini_usage,
        "gemini_models_answered": gemini_models_answered(gemini_usage),
        "gemini_substitutions": gemini_substitutions(),
        "gemini_relay": judge_relay.record() if judge_relay is not None else None,
        "model_endpoint_relay": model_relay.record() if model_relay is not None else None,
        "docker_network": NETWORK,
        "artifacts_dir": keep_dir,
        "mode": mode,
        "thinking": bool(enable_thinking),
        "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": official,
        "max_output_tokens_note": tokens_note,
        "scoring": "ARE hybrid: deterministic hard-validation + LLM judge (gbench Gemini cascade)",
        "sampling": ("the ARE paper's agent protocol, applied in the hooked ARE process: "
                     f"temperature {temperature} ({temperature_source}; the paper's 0.5 is "
                     "recorded, not sent)"
                     + f", max_tokens {turn_tokens} per turn, stop "
                     f"{AGENT_STOP_SEQUENCES} "
                     + ("applied to the streamed answer by the client, which closes the "
                        "request at the first one (none sent to the server on a --thinking run)"
                        if enable_thinking else "sent to the server")
                     + f", {PAPER_MAX_ITERATIONS} steps, the chat template's thinking switch of "
                     "the run; the judge cascade samples at temperature 0"),
        "raw_summary": {k: v for k, v in summary.items() if k != "runs"},
        "metric": ("GAIA2 Overall = per-capability equal-weight macro success rate, ARE's own "
                   "statistics (report_stats.calculate_statistics) over the scenario runs without "
                   "a harness error (deterministic hard-validation + LLM-judge soft-validation; "
                   "percent, runs ARE could not validate left out; runs whose model endpoint "
                   "call, judge call, judge verdict, validation or run failed on the harness side "
                   "left out; the macro averages the capabilities with a validated run, and a "
                   "capability left out is named). Agent runs in Meta's ARE simulator via "
                   "are-benchmark, driven by the "
                   "served model through LiteLLM's OpenAI-compat provider with the ARE paper's "
                   "protocol. This run: mode `" + str(mode) + "` ("
                   + ("the canonical 3 phases" if mode == "gaia2-run"
                      else "the standard phase") + " of "
                   + (f"the {single_config} capability config" if single_config
                      else "all 5 capability configs")
                   + f"), {num_runs} run(s) per scenario ({num_runs_source}); gbench's default "
                   "mode is `run` with ARE's 3 runs per scenario, the canonical 3-phase "
                   "gaia2-run is opt-in (GBENCH_GAIA2_MODE=gaia2-run)."),
        "leaderboard_comparable": leaderboard,
        "leaderboard_comparable_reason": "; ".join(reasons),
        "max_concurrent_scenarios": max_concurrent,
        "max_concurrent_scenarios_source": max_concurrent_source,
        "num_runs_source": num_runs_source,
    }
    if attempts_requested > 1:
        result["attempts"] = {
            "attempts_requested": attempts_requested, "attempts_per_sample": num_runs,
            "source": num_runs_source,
            "note": ("ARE runs every scenario num_runs times (its default 3) and its statistics "
                     "cover them all; --attempt-count maps to num_runs, never below 3")}
    if shard_label:
        applied = summary.get("shard") or {}
        if applied.get("spec") != shard_label:
            raise infra_required(
                "gaia2", f"--shard {shard_label} was requested but the orchestrator did not apply "
                "it (no shard record in its summary); rebuild the image from "
                "gbench/docker/gaia2.Dockerfile.", DOCS_URL)
        record_shard(result, shard_label)
        result["shard_selection"] = applied.get("per_config")
    if summary.get("selection"):
        # the scenarios each config kept (the shard, then the seeded --eval-limit draw)
        result["selection"] = summary["selection"]
    result["eval_limit_plan"] = workload
    if config_limits is not None:
        result["limit_skipped_configs"] = [c for c, n in config_limits.items() if n == 0]
    return result


def _empty_shard_result(model_name: str, enable_thinking: bool, shard_label: str,
                        summary: Dict[str, Any]) -> Dict[str, Any]:
    """The result of a shard with no scenario of any configured capability: ARE was not started,
    so there is no score (accuracy None), never an error and never 0%."""
    reason = f"empty shard {shard_label}: no items on this node"
    result: Dict[str, Any] = {
        "benchmark_type": "eval", "eval_name": "gaia2", "model_name": model_name,
        "thinking": bool(enable_thinking), "status": "success", "status_reason": reason,
        "accuracy": None, "accuracy_withheld_reason": reason, "empty_shard": True,
        "total_questions": 0, "correct_answers": 0, "effective_n": 0, "count_unit": "scenario runs",
        "sample_traces": [], "shard_selection": (summary.get("shard") or {}).get("per_config"),
        "dataset": summary.get("dataset")}
    record_shard(result, shard_label)
    return add_comparability_reason(result, reason)


def _merge_statistics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ARE's report_stats.calculate_statistics over scenario-run records (gaia2_run.py's `runs`:
    config, phase_name, scenario_id, run_number, status), with the macro averaged over the
    capabilities that have a validated run (gaia2_run.macro_over_validated). A run is validated
    unless its status is no_validation; its success_numeric is 1 for success, else 0 (failed,
    exception); success rates are percent. Pass@k / Pass^k: the (scenario, phase) groups among the
    validated runs with any / every run a success, over every (scenario, phase) group."""
    per: Dict[str, Dict[str, int]] = {}
    groups: Dict[Tuple[str, str], List[int]] = {}
    all_groups = set()
    for r in runs:
        cap = capability_key(r.get("config"), r.get("phase_name") or "standard")
        st = per.setdefault(cap, {"total_runs": 0, "validated_runs": 0, "success_runs": 0,
                                  "no_validation_runs": 0})
        st["total_runs"] += 1
        key = (str(r.get("scenario_id")), str(r.get("phase_name") or "standard"))
        all_groups.add(key)
        if r.get("status") == "no_validation":
            st["no_validation_runs"] += 1
            continue
        st["validated_runs"] += 1
        ok = int(r.get("status") == "success")
        st["success_runs"] += ok
        groups.setdefault(key, []).append(ok)
    per_capability = {}
    for cap, st in sorted(per.items()):
        rate = (100.0 * st["success_runs"] / st["validated_runs"]) if st["validated_runs"] else None
        per_capability[cap] = {**st, "success_rate": rate}
    rates = [v["success_rate"] for v in per_capability.values() if v["success_rate"] is not None]
    validated = sum(v["validated_runs"] for v in per_capability.values())
    success = sum(v["success_runs"] for v in per_capability.values())
    n_groups = len(all_groups)
    return {
        "per_capability": per_capability,
        "macro_success_rate": (sum(rates) / len(rates)) if rates else None,
        "micro_success_rate": (100.0 * success / validated) if validated else None,
        "pass_at_k_percent": (100.0 * sum(1 for v in groups.values() if any(v)) / n_groups
                              if n_groups else None),
        "pass_k_percent": (100.0 * sum(1 for v in groups.values() if v and all(v)) / n_groups
                           if n_groups else None),
        "total_runs": sum(v["total_runs"] for v in per_capability.values()),
        "validated_runs": validated, "success_runs": success,
        "no_validation_runs": sum(v["no_validation_runs"] for v in per_capability.values()),
        "total_scenarios": n_groups,
        "capabilities_without_validated_run": sorted(
            k for k, v in per_capability.items() if not v["validated_runs"]),
    }


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one gaia2 run (gbench --merge): ARE's statistics
    recomputed (_merge_statistics) over the union of the shards' scenario runs (sample_traces)
    without a harness error, as each shard's launcher computed its own; the harness-error runs
    are counted; the shard comparability reasons are dropped (every other reason is kept)."""
    union: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for r in results:
        for t in r.get("sample_traces") or []:
            key = (str(t.get("config")), str(t.get("phase_name") or "standard"),
                   str(t.get("scenario_id")), str(t.get("run_number")))
            union.setdefault(key, t)
    runs = [union[k] for k in sorted(union)]
    keep = [t for t in runs if not t.get("harness_errors")]
    stats = _merge_statistics(keep)
    merged = dict(results[0])
    macro, micro = stats["macro_success_rate"], stats["micro_success_rate"]
    merged.update({
        "accuracy": round(macro, 2) if macro is not None else (
            round(micro, 2) if micro is not None else None),
        "macro_success_rate": round(macro / 100.0, 4) if macro is not None else None,
        "micro_success_rate": round(micro / 100.0, 4) if micro is not None else None,
        "pass_at_k_percent": stats["pass_at_k_percent"], "pass_k_percent": stats["pass_k_percent"],
        "headline_metrics": {k: round(float(v), 2) for k, v in (
            ("macro", macro), ("micro", micro), ("pass@k", stats["pass_at_k_percent"]),
            ("pass^k", stats["pass_k_percent"])) if v is not None},
        "per_capability": stats["per_capability"],
        "total_questions": len(runs), "effective_n": stats["validated_runs"],
        "correct_answers": stats["success_runs"], "total_scenarios": stats["total_scenarios"],
        "no_validation_runs": stats["no_validation_runs"],
        "n_harness_error_runs": len(runs) - len(keep),
        "harness_error_runs": [t for t in runs if t.get("harness_errors")],
        "sample_traces": runs, "merged_statistics": stats})
    kinds: Dict[str, int] = {}
    for t in runs:
        for k in t.get("harness_errors") or []:
            kinds[str(k)] = kinds.get(str(k), 0) + 1
    merged["harness_error_kinds"] = dict(sorted(kinds.items()))
    answered: Dict[str, int] = {}
    for r in results:
        for m, n in (r.get("gemini_models_answered") or {}).items():
            answered[m] = answered.get(m, 0) + int(n)
    merged["gemini_models_answered"] = dict(sorted(answered.items()))
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if reason and not reason.startswith(shard_prefix) \
                    and not reason.startswith("empty shard ") and reason not in kept:
                kept.append(reason)
    merged["leaderboard_comparable"] = False      # the Gemini judge substitution always applies
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    merged["status"] = ("completed_with_errors"
                        if any(r.get("status") == "completed_with_errors" for r in results)
                        or len(runs) != len(keep) else "success")
    for k in ("accuracy_withheld_reason", "status_reason", "shard_selection"):
        merged.pop(k, None)
    # each shard drew its own scenarios: one record per shard, never shard 1's for the union
    merged.pop("selection", None)
    selections = {str(r.get("shard") or i + 1): r["selection"] for i, r in enumerate(results)
                  if r.get("selection")}
    if selections:
        merged["selection_by_shard"] = selections
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = ("no validated scenario run outside harness errors "
                                              "in any shard")
    # the shards' summed plan, not the first shard's
    from .limit_plan import merge_plans
    plan = merge_plans(results)
    if plan is not None:
        merged["eval_limit_plan"] = plan
    return merged
