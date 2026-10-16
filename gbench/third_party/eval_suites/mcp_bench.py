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

# Evaluation Suite: mcp_bench
# Description: MCP-Bench (Accenture) - live multi-server Model Context Protocol agent benchmark

"""gbench native built-in runner for mcp_bench (Tool Use & Function Calling).

Canonical MCP-Bench (Accenture/mcp-bench, arXiv:2508.20453): a live agentic benchmark over 28
real MCP servers (250 tools), where the model discovers tools, plans a multi-step trajectory,
invokes tools against the live servers, and grounds a final answer. Scoring = ComplexEval-style:
rule-based tool metrics (schema compliance, valid tool-name rate, execution success) + an LLM
judge scoring six sub-dimensions (task fulfillment, grounding, tool appropriateness, parameter
accuracy, dependency awareness, parallelism/efficiency), each run 5x with randomized order and
averaged (the upstream "judge stability" protocol). The headline is a 0-1 Overall Score = the
mean of four dimensions (schema understanding, task completion, tool usage, planning). As in the
paper's Table 3 (arXiv:2508.20453), schema understanding groups all three rule-based metrics:
valid tool-name rate, schema compliance and execution success.

gbench DELEGATES the whole agent loop to the bundled upstream runner inside a LOCAL container
(gbench/docker/mcp_bench.Dockerfile bakes the runner + all 28 vendored servers, every step of
upstream's install.sh on the README's Python 3.10, and the MCP client pinned to mcp==1.16.0: the
release upstream's declared mcp[cli]>=1.9.0 resolved to on the pinned commit's date, with which
its connection check reports 28/28; see gbench/docker/mcp_bench_constraints.txt). The launcher:

  * points the model-under-test at the gbench /v1 endpoint (openai_compatible provider);
  * injects gbench's ESTABLISHED Gemini cascade as the judge - upstream's exact 6-dimension
    prompts and 5x stability protocol are UNCHANGED; only the model underneath the judge changes
    from o4-mini to the Gemini cascade (base.judge_generate_cascade's model list/rounds/backoff),
    for consistency with every other gbench judged suite. The judge's Gemini key never enters the
    container: model-driven MCP servers run model text there (the Scientific Computing server's
    sympy parse_expr, the OSINT server's commands), so the container gets a per-run placeholder
    of gbench's Gemini relay (gemini_relay.GeminiRelay, a thread of this process listening on the
    container network's gateway) and sends its judge requests to it; the relay adds the real key
    and forwards only POST /v1beta/openai/chat/completions. gbench passes the run's Gemini cascade
    (GBENCH_JUDGE_MODELS, base.gemini_cascade_env()); each judge request carries upstream's model,
    messages and max_tokens and no temperature, the OpenAI client makes one attempt per model
    (max_retries=0) and a model the API answers 404 for is skipped for the rest of the call. The
    result records the models that answered (gemini_models_answered, judge_models_per_task);
  * runs the five servers that need one of the README's keys (CREDENTIALED_SERVERS: Google Maps,
    BioMCP, Hugging Face, National Parks, NASA Data) in a second container of the image, the
    credentialed-servers container (CredentialedServers), the only place the five keys go: on a
    docker network of its own, with no port, volume or service, and none of the servers that run
    model text. The model container holds no key (its entrypoint refuses one); its commands.json
    entries for those five start gbench's bridge (gbench/docker/mcp_bench_bridge.py), which carries
    each session through gbench's relay (CredentialedServersRelay, a thread of this process on the
    network's gateway, `docker exec` into the credentialed container) to the real server
    (gbench/docker/mcp_bench_credentialed.py). The servers, their tools and every request and
    reply are upstream's; a key value in a reply is masked. The residual risk, accepted as
    upstream designed it: the five servers execute the model's tool calls with the real keys in
    their own processes, so a flaw in one of them that let the model's arguments run code there
    would reach those keys; no model-written code runs in that container
    (docs/evals/mcp_bench.md). A session that did not come up (the
    relay answers only after the credentialed side says it is up), a server that exited before
    it answered, and a session lost or ended by a docker exec failure while its client still used
    it, are recorded by the bridge and make the task running then a harness error
    (credentialed_relay); the relay's own failures during the run and a credentialed-servers
    container that is not running at its end are a harness error of the run (status
    completed_with_errors);
  * runs all 104 tasks. Before the run gbench checks that every one of the 28 servers can serve: the
    five API keys the MCP-Bench README lists as required (exactly the keys the `env` lists of the
    pinned mcp_servers/commands.json name) are set and not empty, and each is accepted by the
    service its server calls (server_key_problems: one fixed request with the key, sent the way the
    server sends it; Google Maps once per Maps Platform API its server calls), both checked in
    this process before any container starts (upstream's connection check never uses a key, so a
    refused key would otherwise let its server start and every call of its tools get the refusal,
    scored as the task's tool results), the image's launcher reports every server built and the
    network up (every
    host name of SERVER_API_HOSTS resolves in the container and a connection to one of them opens;
    there is no IP-only test), and MCP-Bench's own connection check (utils/collect_mcp_info.py,
    the README's first step) reports 28/28 servers connected, each listing its tools. Any gap is
    infra_required naming the servers and keys; gbench never runs the tasks of the provisioned
    servers only. The launcher checks the servers again when the run starts and runs no task if
    one cannot serve, which is infra_required too;
  * keeps each task's harness errors out of the score: the launcher (mcp_bench_run.py
    HarnessAccounting) flags a task in which the last attempt of a model-under-test call failed on
    the endpoint's side, the judge made no evaluation, a name lookup failed (a resolver failure in
    the task's error or a tool's error reply naming a server API host or the resolver, or a DNS
    outage its watchdog saw while the task ran), a credentialed-server session failed (see
    above), or that upstream recorded as failed, and upstream's aggregator averages the other tasks only. The
    flagged tasks are counted per kind in `harness_errors`, named in
    `leaderboard_comparable_reason`, and make the status completed_with_errors; so do judge calls
    of scored tasks that failed on every cascade model (a partial judge outage: upstream's
    surviving stability runs are averaged, and the calls are counted and named); a run with no
    task left to score publishes no score (infra_required).

Because the judge is Gemini (the MCP-Bench leaderboard requires o4-mini), `leaderboard_comparable`
is always False.

HARD-ERRORS (infra_required, never skips) if Docker, the locally-built image, GEMINI_API_KEY
(the judge), a required server key or a server is missing, if a service refuses its server key
(or gives no answer of its own to the live check), if the model endpoint URL carries a
user name or password, if a relay cannot listen, if the credentialed-servers container cannot
start, if the container produces no parseable result or no harness record, or if no task is left
to score.
Heavy external provisioning - see docs/evals/mcp_bench.md.

Sampling: the delegated agent loop builds the model-under-test's requests as upstream does
(llm/provider.py LLMProvider.get_completion: the model, the system and user messages and a
max_tokens; no temperature). Inside the image, gbench's request settings
(gbench/docker/mcp_bench_request_policy.py, applied to the model-under-test's provider only)
change just these fields:

  * the chat template's thinking switch (`chat_template_kwargs.enable_thinking`) follows
    `--thinking`;
  * the temperature is gbench's rule for every suite (base.resolve_temperature): 1.0 with
    `--thinking`, 0.0 without (upstream sends none; gbench always sends one). `--temperature`, or
    `GBENCH_MCP_BENCH_TEMPERATURE` for this suite alone (it takes precedence), sets another in
    either mode. It is recorded under `official_sampling` and named in
    `leaderboard_comparable_reason`;
  * max_tokens follows the token-limit policy (base.resolve_token_budget). Without `--thinking`
    upstream's own caps are sent exactly (config/benchmark_config.yaml at the pinned commit:
    llm.planning_tokens 12000 for the planning rounds and the final answer,
    summarization_max_tokens 10000, format_conversion_tokens 8000, and a compression call's target
    length). With `--thinking` those caps, written for non-reasoning generation, are not applied:
    every request is sent the run's budget (`--max-output-tokens`, else the thinking floor), cut to
    the room the served context window leaves after the prompt;
  * with `--thinking` every request carries a timeout sized to that budget as gbench's own
    requests are (base.request_timeout_s: GBENCH_REQUEST_TIMEOUT_S + budget /
    GBENCH_MIN_DECODE_TOK_S seconds to read the reply, recorded in `model_request_timeout_s`), in
    place of the openai client's default 600 s, which a reply decoding the whole budget can
    outlast. Without `--thinking` the harness client's own timeout applies.

Replies that stop at max_tokens are counted in `model_replies` (with `--thinking` they are named
in `leaderboard_comparable_reason`). A model-under-test request that ends in an error of the
served endpoint or of the connection (for example a timeout, a 5xx, a 401/404, a refused
connection, or a context-window rejection of the --thinking budget that could not be fitted) is a
harness error,
never the model's outcome: it is counted in `model_replies.failed_requests` (`timed_out`,
`failed_by_error` by exception class and HTTP status) and in `harness_errors`, named in
`leaderboard_comparable_reason`, and the status is `completed_with_errors`. Requests the served
window rejected where upstream's own max_tokens was sent, or where the prompt alone fills the
window (`rejected_over_context`, left to upstream's own handling), and requests the harness
cancelled in flight (`cancelled`, e.g. at MCP-Bench's task_timeout of 5000 s) are counted and
named. The Gemini cascade judge sends what upstream's judge call sends (the same get_completion):
the model, the system and user messages and the evaluator's max_tokens, and no temperature.

Network: the container (the pre-run check and the run) is on docker's default bridge network, or
the one GBENCH_MCP_BENCH_NETWORK names; a model endpoint on this host (127.0.0.1, localhost) is
reached at that network's gateway (172.17.0.1 on the default bridge), where the served model must
listen, and so is the judge's Gemini relay. The model under test drives tools that run commands
on targets it names (the OSINT server's nmap, whois, dig and host) and parse expressions it
writes (the scientific-computing server's sympy), so before the container starts gbench checks,
with one probe container of the image on that network (container_guard.require_metadata_blocked),
that the cloud metadata server (169.254.169.254) is unreachable and that the servers' API host
names (SERVER_API_HOSTS, and the model endpoint's host when it is a name) resolve; either failing
stops the suite with infra_required naming the host firewall rule (a DNS outage is found here,
before the run, never scored). The credentialed-servers container's own docker network is checked
the same way (its servers' API host names, credentialed_dns_names()) before that container starts.
The relay to it listens on the model container network's gateway, next to the judge's relay.

Values reach the containers by name (`docker run -e NAME`, the value taken from the environment
gbench passes to docker), never on the command line: the five server keys go only to the
credentialed-servers container; the model container gets the endpoint URL (which may carry no
user info), the judge relay's placeholder and addresses (GEMINI_API_KEY, GEMINI_API_KEYS,
GEMINI_OPENAI_BASE_URL, GBENCH_GEMINI_RELAY_URL) and the credentialed relay's address and per-run
token (MCP_CREDENTIALED_RELAY, MCP_CREDENTIALED_TOKEN), and its environment has no server key. The
real Gemini key stays in this process. Any container output quoted in an error has every
credential value redacted.

Sharding: `--shard I/N` IS honoured. Each MCP-Bench task carries a stable, globally-unique
`task_id`, so this runner validates GBENCH_SHARD (a malformed value raises before anything starts),
forwards it normalized into the container, and the launcher selects an interleaved subset -
round-robin over the DETERMINISTIC sorted full task-id list (`ids[I-1::N]`, matching
sampling.shard_select) - BEFORE MCP_TASK_LIMIT, the same compose order as the native path; the
launcher stops on a malformed value itself. `--eval-limit N` (MCP_TASK_LIMIT) is then a seeded
draw of N of the shard's task ids balanced across the server groups (the launcher's copy of
sampling.stratified_indices, seed "mcp_bench"), not the first N in file order; the manifest reports
it (`subset_manifest.limit_draw`, `selected_task_ids`) and gbench refuses an image whose launcher
does not (it took the first N). A sharded run is a subset by construction, so
`tasks_run` / `subset_manifest.tasks_kept` cover only the shard, `is_full` is False, the result
records `shard`, and `leaderboard_comparable` stays False; a shard with no task reports no score
(accuracy None, `empty_shard`). The tasks run one at a time (`execution`), and --attempt-count is
recorded as not applied (`attempts`).
"""

import hmac
import json
import logging
import os
import secrets
import select
import shutil
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from . import container_guard, gemini_relay
from .limit_plan import eval_limit_plan
from .swebench_common import infra_required
from .base import (MODELS_NOT_SERVED_PREFIX, REASONING_CAP_NOTE, OfficialSampling,
                   add_comparability_reason, current_shard, get_run_knob,
                   gemini_api_keys, gemini_cascade_env, gemini_key_live_valid, judge_cascade,
                   record_shard, redact_text, request_timeout_s, resolve_official_sampling,
                   resolve_token_budget, secret_values)

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/mcp_bench.md"

_IMAGE_DEFAULT = "gbench-mcp-bench:7a8eaeae"
#: The server API keys. They go only into the credentialed-servers container (a second container
#: of the image, CredentialedServers), never into the model container, where model text runs.
_SERVER_KEYS = ["GOOGLE_MAPS_API_KEY", "NCI_API_KEY", "HF_TOKEN", "NPS_API_KEY", "NASA_API_KEY"]
#: The servers whose commands.json `env` lists one of _SERVER_KEYS at the pinned commit, with that
#: key and their session kind (commands.json `transport`); they run in the credentialed-servers
#: container (gbench/docker/mcp_bench_credentialed.py) and the model container reaches them through
#: the bridge (gbench/docker/mcp_bench_bridge.py) and gbench's relay (CredentialedServersRelay).
CREDENTIALED_SERVERS = {"Google Maps": ("GOOGLE_MAPS_API_KEY", "http"),
                        "BioMCP": ("NCI_API_KEY", "stdio"),
                        "Hugging Face": ("HF_TOKEN", "stdio"),
                        "National Parks": ("NPS_API_KEY", "stdio"),
                        "NASA Data": ("NASA_API_KEY", "stdio")}
#: The credentialed-servers side of each session, in the image.
CREDENTIALED_SCRIPT = "/app/adapter/mcp_bench_credentialed.py"
#: Accenture/mcp-bench README ("Required API keys include ..."; the loader reads them from
#: mcp_servers/api_key) and the server each one serves.
_REQUIRED_KEYS = {"NPS_API_KEY": "National Parks", "NASA_API_KEY": "NASA Data",
                  "HF_TOKEN": "Hugging Face", "GOOGLE_MAPS_API_KEY": "Google Maps",
                  "NCI_API_KEY": "BioMCP"}
#: The live check of each set server key before the run (server_key_problems): one fixed, read-only
#: request with the key to the service its server calls, sent the way the server sends the key. A
#: key its service refuses still lets the server start and list its tools (upstream's connection
#: check never uses a key), and every call of its tools would then get the service's refusal, which
#: upstream scores as the task's tool results; so a refused key stops the suite with infra_required
#: before any container starts. The replies to a refused key, checked against the live services with
#: a placeholder key: National Parks and NASA HTTP 403 (API_KEY_INVALID), Hugging Face and NCI HTTP
#: 401. {key: (URL, how the key is sent: a header name, "bearer" for an Authorization header, or
#: "query:<parameter>")}; the service must answer HTTP 200.
SERVER_KEY_CHECKS: Dict[str, Tuple[str, str]] = {
    # mcp-server-nationalparks (npsApiClient: header X-Api-Key)
    "NPS_API_KEY": ("https://developer.nps.gov/api/v1/parks?limit=1", "X-Api-Key"),
    # nasa-mcp (server.py: query parameter api_key)
    "NASA_API_KEY": ("https://api.nasa.gov/neo/rest/v1/stats", "query:api_key"),
    # huggingface-mcp-server: upstream's setup requires the token; the pinned server never sends it
    "HF_TOKEN": ("https://huggingface.co/api/whoami-v2", "bearer"),
    # biomcp (NCI Clinical Trials Search API v2: header x-api-key)
    "NCI_API_KEY": ("https://clinicaltrialsapi.cancer.gov/api/v2/trials?size=1", "x-api-key"),
}
#: Google Maps (mcp-google-map, @googlemaps/google-maps-services-js with the key as the `key`
#: parameter): one request per Maps Platform API the server calls (geocode and reverseGeocode,
#: elevation, placesNearby and placeDetails, directions, distancematrix). Each answers HTTP 200 with
#: a `status`: OK or ZERO_RESULTS when the key may use that API, REQUEST_DENIED (with an
#: error_message naming the cause) for a refused key, a project without billing or an API not
#: enabled for the key (checked against the live service with a placeholder key).
GOOGLE_MAPS_KEY_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("Geocoding API", "geocode/json?address=Paris"),
    ("Elevation API", "elevation/json?locations=48.8584,2.2945"),
    ("Places API", "place/nearbysearch/json?location=48.8584,2.2945&radius=100"),
    ("Directions API", "directions/json?origin=Paris&destination=Lyon"),
    ("Distance Matrix API", "distancematrix/json?origins=Paris&destinations=Lyon"),
)
GOOGLE_MAPS_KEY_CHECK_URL = "https://maps.googleapis.com/maps/api/"
#: A check that got no answer of the service's own (a network error or timeout, HTTP 5xx or 429, a
#: Maps OVER_QUERY_LIMIT or UNKNOWN_ERROR) is sent again, up to KEY_CHECK_ATTEMPTS times
#: KEY_CHECK_RETRY_DELAY_S apart; one that still has none stops the suite too (the server could not
#: use the key either).
KEY_CHECK_ATTEMPTS = 3
KEY_CHECK_RETRY_DELAY_S = 2.0
KEY_CHECK_TIMEOUT_S = 30.0
#: The task files the launcher runs by default (mcp_bench_run.py main()).
_TASK_FILES = ("mcpbench_tasks_single_runner_format.json",
               "mcpbench_tasks_multi_2server_runner_format.json",
               "mcpbench_tasks_multi_3server_runner_format.json")
#: Accenture/mcp-bench@7a8eaea README "Basic Usage" step 1: `python ./utils/collect_mcp_info.py`
#: must report every server connected ("28/28 servers connected") and "All successfully
#: connected servers returned tools!".
UPSTREAM_SERVER_COUNT = 28
#: The tasks of the full set (upstream's task files at the pinned commit).
UPSTREAM_TASK_COUNT = 104
#: config/benchmark_config.yaml at the pinned commit: `distraction_servers_default: 10`,
#: `enable_judge_stability: true`, `judge_stability_runs: 5`.
UPSTREAM_DISTRACTION_COUNT = 10
UPSTREAM_JUDGE_STABILITY_RUNS = 5
#: The model-under-test caps at the pinned commit (config/benchmark_config.yaml `llm.*`, read by
#: agent/executor.py): the planning rounds and the final answer, tool-result summaries and JSON
#: format fixes. A compression call sends its target length instead.
UPSTREAM_PLANNING_TOKENS = 12000
UPSTREAM_SUMMARIZATION_TOKENS = 10000
UPSTREAM_FORMAT_CONVERSION_TOKENS = 8000
UPSTREAM_CAPS_TEXT = (f"planning rounds and the final answer {UPSTREAM_PLANNING_TOKENS}, tool-result "
                      f"summaries {UPSTREAM_SUMMARIZATION_TOKENS}, format fixes "
                      f"{UPSTREAM_FORMAT_CONVERSION_TOKENS}, a compression call its target length")
#: Upstream's sampling of the model under test: no temperature, and max_tokens on every request
#: (the largest cap, planning's, is the one recorded; the others are in UPSTREAM_CAPS_TEXT).
MCP_BENCH_SAMPLING = OfficialSampling(
    reference=("Accenture/mcp-bench@7a8eaea llm/provider.py LLMProvider.get_completion "
               "({model, messages, max_tokens}; no temperature); config/benchmark_config.yaml "
               "llm.planning_tokens 12000 (planning rounds and the final answer), "
               "summarization_max_tokens 10000, format_conversion_tokens 8000, and a compression "
               "call's target length"),
    send_temperature=False,
    max_tokens=UPSTREAM_PLANNING_TOKENS,
)
#: The image label that says the image applies the run's model request settings
#: (gbench/docker/mcp_bench_request_policy.py); an image without it is refused. Version 2 adds
#: the request timeout sized to the --thinking budget and the count of failed requests.
REQUEST_POLICY_LABEL = "org.gbench.model-request-policy"
REQUEST_POLICY_VERSION = "2"
#: config/benchmark_config.yaml at the pinned commit: `task_timeout: 5000` (seconds), after which
#: benchmark/runner.py cancels the task's execution (asyncio.wait_for).
UPSTREAM_TASK_TIMEOUT_S = 5000
#: Where the image appends the reply counts of the model under test (the mounted /out).
REQUEST_LOG = "model_requests.json"
#: The image label that says its judge takes gbench's Gemini relay placeholder and relay URL (no
#: real Gemini key enters the container), the credentialed servers run in the credentialed-servers
#: container behind the bridge (no server key enters the model container, whose entrypoint
#: refuses one), its network check resolves the servers' API host names (no IP-only test), and its
#: launcher keeps each task's harness errors out of upstream's averages and writes HARNESS_LOG with
#: the answering Gemini models per task (gbench/docker/mcp_bench_run.py); an image without it is
#: refused. Version 3: the credentialed script says when a session is up before the relay answers,
#: the relay frames what comes back with an end frame, and the bridge records a session that did
#: not come up, a server that exited before it answered and a session lost while in use (an image
#: at version 2 speaks the old session protocol).
HARNESS_LABEL = "org.gbench.mcp-bench-harness"
HARNESS_VERSION = "3"
#: Where the image writes the per-task harness record (the mounted /out).
HARNESS_LOG = "harness_log.json"
#: Its layout version (mcp_bench_run.HARNESS_LOG_VERSION).
HARNESS_LOG_VERSION = 2
#: The API host each online server calls, from its source at the pinned commit. The container
#: must resolve them: container_guard checks them before the container starts, the launcher's
#: network check and its DNS watchdog look them up (MCP_DNS_NAMES). The OSINT Intelligence server
#: works on targets the model names and has none; the six offline servers need no network.
SERVER_API_HOSTS = {
    "BioMCP": "clinicaltrialsapi.cancer.gov",
    "Call for Papers": "www.wikicfp.com",
    "Car Price Evaluator": "parallelum.com.br",
    "Context7": "context7.com",
    "DEX Paprika": "api.dexpaprika.com",
    "FruityVice": "www.fruityvice.com",
    "Game Trends": "store.steampowered.com",
    "Google Maps": "maps.googleapis.com",
    "Huge Icons": "hugeicons.com",
    "Hugging Face": "huggingface.co",
    "Metropolitan Museum": "collectionapi.metmuseum.org",
    "Movie Recommender": "api.themoviedb.org",
    "NASA Data": "api.nasa.gov",
    "National Parks": "developer.nps.gov",
    "NixOS": "search.nixos.org",
    "OKX Exchange": "www.okx.com",
    "OpenAPI Explorer": "oapis.org",
    "Paper Search": "arxiv.org",
    "Reddit": "www.reddit.com",
    "Weather Data": "api.weatherapi.com",
    "Wikipedia": "en.wikipedia.org",
}
#: The judge's only call through the relay: the OpenAI-compatible chat completion.
JUDGE_RELAY_CALLS = frozenset({("POST", "/chat/completions")})
#: What each harness-error kind of a task means (HARNESS_LOG `tasks[].kinds`, set by
#: mcp_bench_run.HarnessAccounting).
HARNESS_ERROR_KINDS = {
    "model_endpoint": ("the last attempt of a model-under-test call failed on the endpoint's side "
                       "or the connection's (upstream's retries did not get past it)"),
    "judge_outage": ("the judge made no evaluation (every judge stability run failed); upstream's "
                     "aggregator would skip the task, so it would silently drop out of the "
                     "averages"),
    "dns_failure": ("a name lookup failed in the task's error or a tool's error reply (a resolver "
                    "that did not answer, naming one of the servers' API host names or the "
                    "container's resolver, or one of the servers' API host names not found)"),
    "dns_outage": ("the container's name resolution was down while the task ran (at least half "
                   "of the servers' API host names did not resolve in a watchdog round)"),
    "credentialed_relay": ("a session to a credentialed server (run in gbench's "
                           "credentialed-servers container) did not come up through gbench's "
                           "relay, its server exited before it answered, or the session was lost "
                           "or ended by a docker exec failure while its client still used it"),
    "task_failed": "upstream recorded the task as failed, and its averages leave it out",
    "task_raised": "the task's execution raised",
}


def server_dns_names() -> List[str]:
    """The servers' API host names (SERVER_API_HOSTS), sorted."""
    return sorted(set(SERVER_API_HOSTS.values()))


#: Hosts the credentialed servers need at start: BioMCP and NASA Data are `uv run` servers, and
#: `uv run` fetches its build requirements from PyPI each time it starts one (as it does upstream).
CREDENTIALED_START_HOSTS = ("pypi.org", "files.pythonhosted.org")


def credentialed_dns_names() -> List[str]:
    """The host names the credentialed-servers container's network must resolve, sorted: the
    credentialed servers' API hosts and CREDENTIALED_START_HOSTS."""
    return sorted({SERVER_API_HOSTS[s] for s in CREDENTIALED_SERVERS if s in SERVER_API_HOSTS}
                  | set(CREDENTIALED_START_HOSTS))


def dns_names(endpoint: str = "") -> List[str]:
    """The host names the container must resolve: server_dns_names(), and the model endpoint's
    host when it is a name (not an IP address or a loopback name)."""
    import ipaddress
    from urllib.parse import urlsplit
    names = server_dns_names()
    try:
        host = (urlsplit(endpoint or "").hostname or "").strip().lower()
    except ValueError:
        host = ""
    if host and host not in _LOOPBACK_HOSTS:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if host not in names:
                names.append(host)
    return names
#: How long the pre-run check may take. The connection check runs upstream's default
#: "individual" mode (one server at a time, up to 5 attempts of 30 s each with backoff, 2 s
#: between servers, 5 s between batches of 5): about 2 minutes when every server connects.
PROBE_TIMEOUT_S = 1800
#: The pre-run check, in the image: the launcher's own provisioning functions in the order
#: main() uses them (the credentialed servers' bridges first, as in the run), then upstream's
#: connection check (utils/collect_mcp_info.py MCPServerInfoCollector, default "individual" mode),
#: which reaches the credentialed servers through the bridge. No server key is in this container
#: (its mcp_servers/api_key is left empty, as the entrypoint leaves it); the credentialed
#: container masks every key value in what its servers send back. A server's error text is cut
#: to 300 characters. Nothing of the benchmark runs.
_PROBE = (
    "import asyncio, json, os, sys\n"
    "sys.path.insert(0, '/app/adapter')\n"
    "import mcp_bench_run as R\n"
    "os.chdir(R.MCPBENCH_DIR)\n"
    "sys.path.insert(0, R.MCPBENCH_DIR)\n"
    "open(os.path.join('mcp_servers', 'api_key'), 'w').close()\n"
    "bridged = R.install_credentialed_bridges()\n"
    "files = [os.path.join(R.MCPBENCH_DIR, 'tasks', f) for f in %r]\n"
    "servers = R._all_task_servers(files)\n"
    "allowed, net = R._available_servers(servers)\n"
    "check = dict(R.LAST_NETWORK_CHECK)\n"
    "built = R._built_servers(R.MCPBENCH_DIR)\n"
    "unbuilt = sorted(servers - built) if built is not None else []\n"
    "if built is not None:\n"
    "    allowed = allowed & built\n"
    "from utils.collect_mcp_info import MCPServerInfoCollector\n"
    "info = asyncio.run(MCPServerInfoCollector('individual').collect_all_info())\n"
    "conn = {n: {'status': s.get('connection_status'), 'tools': len(s.get('tools') or {}), "
    "'error': str(s.get('error') or '')[:300]} for n, s in info['servers'].items()}\n"
    "fails = [{k: f.get(k) for k in ('server', 'op', 'why')} for f in R.read_bridge_failures()]\n"
    "print('MCP_BENCH_PROBE ' + json.dumps({'servers': sorted(servers), "
    "'unavailable': sorted(servers - allowed), 'unbuilt': unbuilt, 'network_up': net, "
    "'network_check': check, 'connections': conn, 'bridged': sorted(bridged), "
    "'bridge_failures': fails[:20]}))\n"
) % (_TASK_FILES,)


def _image() -> str:
    return os.environ.get("GBENCH_MCP_BENCH_IMAGE", _IMAGE_DEFAULT)


#: The docker network of the MCP-Bench container unless GBENCH_MCP_BENCH_NETWORK names another:
#: docker's default bridge, whose traffic the host firewall rule of container_guard covers (the
#: host network is not covered by it).
_NETWORK_DEFAULT = "bridge"
#: The default docker bridge's gateway, where a container on it reaches a model served on this
#: host, when `docker network inspect bridge` does not say otherwise.
_BRIDGE_GATEWAY_DEFAULT = "172.17.0.1"
#: Host names of a model endpoint on this host, which a container on a network other than the
#: host network reaches at its network's gateway instead.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "::1")


def _network() -> str:
    """The docker network the pre-run check and the run use (GBENCH_MCP_BENCH_NETWORK, default
    the docker bridge)."""
    return (os.environ.get("GBENCH_MCP_BENCH_NETWORK") or "").strip() or _NETWORK_DEFAULT


def _network_gateway(network: str) -> Optional[str]:
    """The first gateway address `docker network inspect` reports for `network`, or None."""
    try:
        proc = subprocess.run(["docker", "network", "inspect", network, "--format",
                               "{{json .IPAM.Config}}"], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        config = json.loads((proc.stdout or "").strip() or "null") or []
    except ValueError:
        return None
    for entry in config if isinstance(config, list) else []:
        if isinstance(entry, dict) and entry.get("Gateway"):
            return str(entry["Gateway"])
    return None


def container_endpoint(base_url: str, network: str) -> str:
    """The model endpoint as the MCP-Bench container reaches it, ending in /v1. With the host
    network it is `base_url` itself. On any other network a loopback host (127.0.0.1, localhost,
    0.0.0.0, ::1) is replaced by the network's gateway (`docker network inspect`; 172.17.0.1 for
    docker's default bridge when inspect gives none), where the served model must listen; a
    user-defined network without a gateway is infra_required. Other hosts are kept."""
    from urllib.parse import urlsplit, urlunsplit
    endpoint = (base_url or "").rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    if network == "host":
        return endpoint
    parts = urlsplit(endpoint)
    host = (parts.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        return endpoint
    gateway = _network_gateway(network)
    if gateway is None:
        if network != "bridge":
            raise infra_required(
                "mcp_bench", f"the docker network {network!r} (GBENCH_MCP_BENCH_NETWORK) reports "
                "no gateway address, so the container cannot reach the model endpoint on this "
                f"host ({host}). Use a network with a gateway, or serve the model at an address "
                "the container reaches.", DOCS_URL)
        gateway = _BRIDGE_GATEWAY_DEFAULT
    netloc = f"[{gateway}]" if ":" in gateway else gateway
    if parts.port is not None:
        netloc += f":{parts.port}"
    userinfo = parts.netloc.rpartition("@")[0]
    if userinfo:
        netloc = f"{userinfo}@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _tail(proc: "subprocess.CompletedProcess", n: int) -> str:
    """The last `n` characters of a container's output for an error message, with every
    credential value of the environment redacted."""
    return redact_text(proc.stderr or proc.stdout or "")[-n:]


def _image_labels(image: str) -> Dict[str, Any]:
    """The labels of the local `image` ({} when it has none or cannot be inspected)."""
    proc = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}",
                           image], capture_output=True, text=True)
    if proc.returncode != 0:
        return {}
    try:
        labels = json.loads((proc.stdout or "").strip() or "null") or {}
    except ValueError:
        return {}
    return labels if isinstance(labels, dict) else {}


def image_applies_request_policy(image: str) -> bool:
    """Whether `image` carries REQUEST_POLICY_LABEL at REQUEST_POLICY_VERSION: built from a
    Dockerfile that applies the run's thinking switch, temperature, --thinking budget and request
    timeout to the model under test and counts its failed requests. gbench-toolbench:aa4ed9f4 carries the
    same label."""
    return _image_labels(image).get(REQUEST_POLICY_LABEL) == REQUEST_POLICY_VERSION


def image_carries_harness(image: str) -> bool:
    """Whether `image` carries HARNESS_LABEL at HARNESS_VERSION: its judge takes the relay's
    placeholder, its network check resolves names, and its launcher writes HARNESS_LOG."""
    return _image_labels(image).get(HARNESS_LABEL) == HARNESS_VERSION


def read_model_request_log(path: str) -> Optional[Dict[str, Any]]:
    """The reply counts a delegated image's request settings appended to `path` (one JSON line
    per harness process), summed: {settings, counts, failed_by_error, max_tokens_sent,
    processes}. `settings` is the one dict every process reported, or the list of them when they
    differ. None when the file is absent or holds no record."""
    if not os.path.exists(path):
        return None
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    if not records:
        return None
    counts: Dict[str, int] = {}
    by_error: Dict[str, int] = {}
    sent = set()
    for rec in records:
        for key, value in (rec.get("counts") or {}).items():
            if isinstance(value, int):
                counts[key] = counts.get(key, 0) + value
        for key, value in (rec.get("failed_by_error") or {}).items():
            if isinstance(value, int):
                by_error[str(key)] = by_error.get(str(key), 0) + value
        sent.update(v for v in rec.get("max_tokens_sent") or [] if isinstance(v, int))
    settings = [rec.get("settings") for rec in records]
    return {"settings": settings[0] if all(x == settings[0] for x in settings) else settings,
            "counts": counts, "failed_by_error": dict(sorted(by_error.items())),
            "max_tokens_sent": sorted(sent), "processes": len(records)}


def every_model_request_failed(replies: Optional[Dict[str, Any]]) -> int:
    """The number of failed model-under-test requests when not one request of the run got a reply
    (every task's outcome is then a harness error, so the run publishes no score), else 0."""
    counts = (replies or {}).get("counts") or {}
    failed = int(counts.get("failed_requests") or 0)
    return failed if failed and not int(counts.get("replies") or 0) else 0


def model_request_failures(replies: Optional[Dict[str, Any]], timeout_s: Optional[int],
                           harness: str, cancel_hint: str = ""
                           ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(harness_errors, notes) from the image's request log (read_model_request_log).

    A model-under-test request that ended in an error of the served endpoint or of the
    connection (`failed_requests`, `timed_out` among them, `failed_by_error`) is a harness error,
    never the model's outcome: one `harness_errors` entry carries the counts, and a note names
    them. Requests the served context window rejected and left to the harness's own handling
    (`rejected_over_context`) and requests the harness cancelled in flight (`cancelled`) are
    named in a note each. `timeout_s` is the request timeout the image sent (None: the harness
    client's own)."""
    counts = (replies or {}).get("counts") or {}
    errors: List[Dict[str, Any]] = []
    notes: List[str] = []
    failed = int(counts.get("failed_requests") or 0)
    if failed:
        timed_out = int(counts.get("timed_out") or 0)
        by_error = dict(sorted(((replies or {}).get("failed_by_error") or {}).items()))
        errors.append({"error": "model-under-test requests ended in an error of the served "
                                "endpoint or of the connection",
                       "failed_requests": failed, "timed_out": timed_out, "by_error": by_error,
                       "request_timeout_s": timeout_s})
        parts = []
        if timed_out:
            parts.append(f"{timed_out} timed out after "
                         + (f"the {timeout_s} s request timeout sized to the --thinking budget"
                            if timeout_s else "the harness client's own timeout (the openai "
                            "client's default: 600 s to read the reply)"))
        if by_error:
            parts.append("by error: " + ", ".join(f"{k} x{v}" for k, v in by_error.items()))
        notes.append(f"{failed} model request(s) ended in an error of the served endpoint or of "
                     "the connection" + (f" ({'; '.join(parts)})" if parts else "")
                     + f": harness errors, not the model's outcome ({harness} received an error "
                     "where a reply should have been)")
    over = int(counts.get("rejected_over_context") or 0)
    if over:
        notes.append(f"{over} model request(s) rejected by the served context window (prompt + "
                     f"max_tokens longer than it) were left to {harness}'s own handling, as they "
                     "would be upstream on this window")
    cancelled = int(counts.get("cancelled") or 0)
    if cancelled:
        notes.append(f"{cancelled} model request(s) were cancelled in flight by {harness}"
                     + cancel_hint)
    return errors, notes


def check_request_log(eval_name: str, log: Optional[Dict[str, Any]], expected: Dict[str, Any],
                      thinking: bool, docs_url: str = DOCS_URL) -> Optional[Dict[str, Any]]:
    """`log` (read_model_request_log) once its settings are checked against `expected`, the ones
    the run asked the image to apply. Settings that differ are infra_required: the run would not
    have measured what it records. A missing log is infra_required with --thinking (the switch and
    the budget cannot be confirmed); without it None is returned and the caller names the gap."""
    if log is None:
        if thinking:
            raise infra_required(
                eval_name, "the image wrote no model request log, so the run's thinking switch and "
                "--thinking budget cannot be confirmed to have reached the model under test; "
                "rebuild the image from its Dockerfile and re-run.", docs_url)
        return None
    if log.get("settings") != expected:
        raise infra_required(
            eval_name, f"the image applied the model request settings {log.get('settings')} where "
            f"the run asked for {expected}; rebuild the image from its Dockerfile and re-run.",
            docs_url)
    return log


def _key_check_get(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, str]:
    """GET `url` with `headers`: (HTTP status, the start of the body). A network error or a
    timeout raises."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.status, reply.read(65536).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read(65536).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - the status is what matters
            body = ""
        return e.code, body


def _key_check_once(key: str, value: str) -> Tuple[Optional[bool], str]:
    """One attempt of the live check of `key` (SERVER_KEY_CHECKS, GOOGLE_MAPS_KEY_CHECKS): (True
    when the service accepted it, False when it refused it, None when it gave no answer of its
    own; what it answered, key values masked)."""
    if key == "GOOGLE_MAPS_API_KEY":
        refused, retry = [], []
        for api, query in GOOGLE_MAPS_KEY_CHECKS:
            url = f"{GOOGLE_MAPS_KEY_CHECK_URL}{query}&key={urllib.parse.quote(value, safe='')}"
            try:
                status, body = _key_check_get(url, {}, KEY_CHECK_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001 - no answer: sent again
                retry.append(f"{api}: {type(e).__name__} (network error or timeout)")
                continue
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            state = data.get("status") if isinstance(data, dict) else None
            if status == 200 and state in ("OK", "ZERO_RESULTS"):
                continue
            seen = f"{api}: HTTP {status}" + (f" {state}" if state else "") + (
                f" ({str(data.get('error_message') or '').strip()[:160]})"
                if isinstance(data, dict) and data.get("error_message") else "")
            if status >= 500 or status == 429 or state in ("OVER_QUERY_LIMIT", "UNKNOWN_ERROR"):
                retry.append(seen)
            else:
                refused.append(seen)
        if refused:
            return False, redact_text("; ".join(refused + retry))
        if retry:
            return None, redact_text("; ".join(retry))
        return True, "OK"
    url, how = SERVER_KEY_CHECKS[key]
    headers: Dict[str, str] = {}
    if how.startswith("query:"):
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode({how[6:]: value})
    elif how == "bearer":
        headers["Authorization"] = f"Bearer {value}"
    else:
        headers[how] = value
    try:
        status, body = _key_check_get(url, headers, KEY_CHECK_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 - no answer: sent again
        return None, f"{type(e).__name__} (network error or timeout)"
    if status == 200:
        return True, "HTTP 200"
    seen = redact_text(f"HTTP {status} " + " ".join(body.split())[:200]).strip()
    return (None if status >= 500 or status == 429 else False), seen


def server_key_problems(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The live check of every set server key (SERVER_KEY_CHECKS, GOOGLE_MAPS_KEY_CHECKS) before
    the run: {key: why} for each key its service refused, or that got no answer of the service's
    own after KEY_CHECK_ATTEMPTS attempts; {} when every set key was accepted. An unset key is
    not sent (run_mcp_bench names it). Runs in this process, never in a container; no key value
    is in the answer."""
    env = os.environ if environ is None else environ
    problems: Dict[str, str] = {}
    for key in _REQUIRED_KEYS:
        value = (env.get(key) or "").strip()
        if not value:
            continue
        verdict, seen = None, "no attempt"
        for attempt in range(KEY_CHECK_ATTEMPTS):
            verdict, seen = _key_check_once(key, value)
            if verdict is not None:
                break
            if attempt < KEY_CHECK_ATTEMPTS - 1:
                time.sleep(KEY_CHECK_RETRY_DELAY_S)
        if verdict is True:
            continue
        problems[key] = (f"refused: {seen}" if verdict is False else
                         f"no answer of the service's own after {KEY_CHECK_ATTEMPTS} attempts: "
                         f"{seen}")
    return problems


def check_mcp_bench_prerequisites() -> Tuple[bool, str]:
    """Docker + the locally-built image (with the model request settings) + GEMINI_API_KEY (the
    Gemini cascade judge) + a live check of every set server key (server_key_problems)."""
    image = _image()
    build = (f"Build the harness LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/mcp_bench.Dockerfile gbench/docker\n"
             f"It bundles pinned Accenture/mcp-bench + all 28 vendored MCP servers. See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    if not image_applies_request_policy(image):
        return False, (f"image {image!r} predates the model request settings (the run's thinking "
                       "switch, temperature, --thinking budget and request timeout, and the count "
                       "of failed requests, gbench/docker/mcp_bench_request_policy.py; label "
                       f"{REQUEST_POLICY_LABEL}={REQUEST_POLICY_VERSION}). Rebuild it. " + build)
    if not image_carries_harness(image):
        return False, (f"image {image!r} predates the credentialed-servers container and its "
                       "session protocol, the judge's Gemini relay, the name-resolution check and "
                       "the per-task harness record (gbench/docker/mcp_bench_run.py, "
                       "mcp_bench_bridge.py, mcp_bench_credentialed.py and "
                       f"mcp_bench_cascade_judge.py; label {HARNESS_LABEL}={HARNESS_VERSION}); an "
                       "older image would hold the server keys next to model-driven servers, or "
                       "speak another session protocol to gbench's relay. Rebuild it. " + build)
    keys = gemini_api_keys()
    if not keys:
        return False, ("GEMINI_API_KEY (or GEMINI_API_KEYS) is not set - it powers gbench's Gemini "
                       "cascade judge that scores MCP-Bench's six judge sub-dimensions here.")
    _ok, _why = gemini_key_live_valid(",".join(keys))
    if not _ok and str(_why).startswith(MODELS_NOT_SERVED_PREFIX):
        return False, (f"{redact_text(_why)}; every model of the Gemini cascade must be one the API "
                       "serves for the judge (fail-fast before the container run).")
    if not _ok:
        return False, (f"the Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) was rejected by the judge "
                       f"endpoint ({redact_text(_why)}); a valid key is required for the Gemini "
                       "cascade judge (fail-fast before the container run).")
    problems = server_key_problems()
    if problems:
        unset = [k for k in _REQUIRED_KEYS if not (os.environ.get(k) or "").strip()]
        return False, redact_text(
            "the live check of the server keys failed before the run (one request with each key "
            "to the service its server calls): " + "; ".join(
                f"{k} ({_REQUIRED_KEYS[k]}): {why}" for k, why in problems.items())
            + (". Also unset: " + ", ".join(f"{k} ({_REQUIRED_KEYS[k]})" for k in unset)
               if unset else "")
            + ". A server whose key its service refuses still starts and lists its tools, and "
            "every call of its tools would get the refusal, scored as the task's tool results; "
            f"set working keys (see {DOCS_URL}, The keys) and re-run.")
    return True, ""


def provisioning_probe(image: str, relay_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """{servers, unavailable, unbuilt, network_up, network_check, connections, bridged,
    bridge_failures} from the image: what its launcher can serve, and upstream's connection check
    (utils/collect_mcp_info.py) per server, the credentialed servers reached through the bridge
    and gbench's relay (`relay_env`: CredentialedServersRelay.client_env(), passed by name). No
    server key enters this container. The servers' API host names (MCP_DNS_NAMES) are the ones
    the launcher resolves in its network check; GBENCH_MCP_BENCH_ASSUME_NETWORK reaches the
    launcher as MCP_ASSUME_NETWORK, as in the run. It runs on the run's docker network
    (_network()), so the network is checked where the run has it."""
    cmd = ["docker", "run", "--rm", "--pull", "never", "--network", _network(),
           "-e", "MCP_DNS_NAMES=" + ",".join(server_dns_names())]
    env = {k: v for k, v in os.environ.items() if k not in _SERVER_KEYS}
    for name, value in (relay_env or {}).items():
        env[name] = value
        cmd += ["-e", name]
    assume_network = os.environ.get("GBENCH_MCP_BENCH_ASSUME_NETWORK")
    if assume_network:
        cmd += ["-e", f"MCP_ASSUME_NETWORK={assume_network}"]
    cmd += ["--entrypoint", "python", image, "-c", _PROBE]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
                              env=env)
    except subprocess.TimeoutExpired as exc:
        raise infra_required("mcp_bench", "the pre-run check in the image (the launcher's "
                             "provisioning and upstream's utils/collect_mcp_info.py connection "
                             f"check) did not finish within {PROBE_TIMEOUT_S} s.",
                             DOCS_URL) from exc
    line = next((ln for ln in (proc.stdout or "").splitlines()
                 if ln.startswith("MCP_BENCH_PROBE ")), None)
    try:
        probe = json.loads(line[len("MCP_BENCH_PROBE "):]) if line is not None else None
    except ValueError:
        probe = None
    if proc.returncode != 0 or not isinstance(probe, dict) or not isinstance(
            probe.get("connections"), dict):
        raise infra_required(
            "mcp_bench", f"the image could not report which MCP servers it can serve and connect "
            f"to (rc={proc.returncode}); rebuild the image from "
            f"gbench/docker/mcp_bench.Dockerfile. tail: {_tail(proc, 400)}", DOCS_URL)
    return probe


def provisioning_gaps(probe: Dict[str, Any], env_keys: Optional[Dict[str, str]] = None
                      ) -> List[str]:
    """What keeps the full task set from running: unset required keys, servers the launcher
    cannot serve (with why) and servers that fail upstream's connection check (not connected, or
    connected without listing tools). Empty when every server can serve. A server's error text
    has every credential value redacted before it is cut to 160 characters, so no part of a key
    survives the cut."""
    secrets = secret_values()
    if env_keys is not None:
        secrets = sorted(set(secrets) | set(secret_values(dict(env_keys))), key=len, reverse=True)
    env_keys = os.environ if env_keys is None else env_keys
    gaps = []
    missing = [k for k in _REQUIRED_KEYS if not (env_keys.get(k) or "").strip()]
    if missing:
        gaps.append("required API keys unset: " + ", ".join(
            f"{k} ({_REQUIRED_KEYS[k]})" for k in missing))
    unbuilt = set(probe.get("unbuilt") or [])
    if unbuilt:
        gaps.append("servers that did not build in the image: " + ", ".join(sorted(unbuilt)))
    unresolved = (probe.get("network_check") or {}).get("unresolved") or {}
    if unresolved:
        gaps.append("the container cannot resolve the servers' API host names: " + ", ".join(
            f"{n} ({why})" for n, why in sorted(unresolved.items())))
    rest = [s for s in probe.get("unavailable") or [] if s not in unbuilt]
    if rest:
        why = ("their API host names do not resolve in the container" if unresolved
               else "no network egress from the container" if not probe.get("network_up")
               else "no bridge to gbench's credentialed-servers container")
        gaps.append(f"servers the launcher cannot serve ({why}): " + ", ".join(sorted(rest)))
    failures = probe.get("bridge_failures") or []
    if failures:
        gaps.append("sessions to the credentialed servers that failed through gbench's relay: "
                    + "; ".join(redact_text(f"{f.get('server')} {f.get('op')}: {f.get('why')}",
                                            secrets)[:160] for f in failures[:5]))
    connections = probe.get("connections")
    if isinstance(connections, dict):
        # README: 28/28 connected, and every connected server returned tools.
        bad = []
        for name in sorted(set(connections) | set(probe.get("servers") or [])):
            c = connections.get(name)
            if c is None:
                bad.append(f"{name} (not checked)")
            elif c.get("status") == "success_no_tools" or (c.get("status") == "success"
                                                          and not c.get("tools")):
                bad.append(f"{name} (connected, but listed no tools)")
            elif c.get("status") != "success":
                error = redact_text(str(c.get("error") or ""), secrets)[:160]
                bad.append(f"{name} ({c.get('status') or 'failed'}: {error})")
        if len(connections) != UPSTREAM_SERVER_COUNT:
            bad.append(f"{len(connections)} servers checked, where MCP-Bench has "
                       f"{UPSTREAM_SERVER_COUNT}")
        if bad:
            gaps.append("servers that failed MCP-Bench's own connection check "
                        "(utils/collect_mcp_info.py, which the README expects to report "
                        f"{UPSTREAM_SERVER_COUNT}/{UPSTREAM_SERVER_COUNT} connected, each with "
                        "tools): " + "; ".join(bad))
    return gaps


def _dim(metrics: Dict[str, Any], *keys: str) -> Optional[float]:
    """Mean of the given metric keys, skipping any that are missing/None. None if none present."""
    vals = [metrics[k] for k in keys if isinstance(metrics.get(k), (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def _overall_for_one(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute the four 0-1 dimensions + 0-1 Overall Score from one averaged-metrics dict.

    Judge dimensions are on a 1-10 scale upstream -> divided by 10 to sit on the 0-1 axis with
    the rule-based schema dimension (which is already 0-1). Overall = mean of the four dims.

    Schema understanding is the mean of the three rule-based metrics that MCP-Bench's Table 3
    (arXiv:2508.20453) lists under that heading: Valid Tool Name Rate, Schema Compliance and
    Execution Success (`tool_call_success_rate` in the upstream results_aggregator output).
    Recomputing the paper's 20 leaderboard rows this way reproduces the printed Overall Score
    for 18 rows and is within 0.001 for the other two; leaving Execution Success out matches
    none of them.
    """
    schema = _dim(metrics, "input_schema_compliance", "valid_tool_name_rate",
                  "tool_call_success_rate")  # 0-1
    tc = metrics.get("task_completion_score")
    tu = metrics.get("tool_selection_score")
    pl = metrics.get("planning_effectiveness_and_efficiency_score")
    dims: Dict[str, Optional[float]] = {
        "schema_understanding": schema,
        "task_completion": (tc / 10.0) if isinstance(tc, (int, float)) else None,
        "tool_usage": (tu / 10.0) if isinstance(tu, (int, float)) else None,
        "planning_effectiveness": (pl / 10.0) if isinstance(pl, (int, float)) else None,
    }
    present = [v for v in dims.values() if v is not None]
    if not present:
        return None
    return {"dimensions": dims, "overall": sum(present) / len(present)}


def compute_overall_score(results_json: Any,
                          manifest: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Roll the container's output into the 0-1 Overall Score.

    `results_json` is either a flat averaged-metrics dict (single task file) or a
    {task_file: metrics} map (multi-file). Multi-file dimensions are task-weighted across files
    (weights from the manifest's per-file kept counts; equal weight if unavailable).
    """
    if not isinstance(results_json, dict) or not results_json:
        return None

    # Flat single-file metrics vs {file: metrics}.
    is_multi = all(isinstance(v, dict) for v in results_json.values()) and \
        any(("task_completion_score" in v or "input_schema_compliance" in v)
            for v in results_json.values() if isinstance(v, dict))
    if not is_multi:
        one = _overall_for_one(results_json)
        return {"overall": one["overall"], "dimensions": one["dimensions"],
                "per_file": None} if one else None

    per_file_counts = {}
    if manifest and isinstance(manifest.get("per_file"), dict):
        for fn, c in manifest["per_file"].items():
            per_file_counts[fn] = int(c.get("kept", 0)) if isinstance(c, dict) else 0

    dim_names = ["schema_understanding", "task_completion", "tool_usage", "planning_effectiveness"]
    weighted: Dict[str, List[Tuple[float, float]]] = {d: [] for d in dim_names}
    per_file_out = {}
    for path, metrics in results_json.items():
        if not isinstance(metrics, dict):
            continue
        one = _overall_for_one(metrics)
        if not one:
            continue
        base = os.path.basename(path)
        w = per_file_counts.get(base) or per_file_counts.get(base.replace("filtered_", "")) or 1
        per_file_out[base] = one
        for d in dim_names:
            v = one["dimensions"].get(d)
            if v is not None:
                weighted[d].append((v, float(w)))

    dim_means: Dict[str, Optional[float]] = {}
    for d in dim_names:
        pairs = weighted[d]
        if pairs:
            tot_w = sum(w for _, w in pairs)
            dim_means[d] = sum(v * w for v, w in pairs) / tot_w if tot_w else None
        else:
            dim_means[d] = None
    present = [v for v in dim_means.values() if v is not None]
    if not present:
        return None
    return {"overall": sum(present) / len(present), "dimensions": dim_means, "per_file": per_file_out}


def _relay_bind_host(network: str) -> str:
    """Where a relay of this process listens so that a container on `network` reaches it: the
    network's gateway (`docker network inspect`), or 127.0.0.1 with the host network."""
    return ("127.0.0.1" if network == "host"
            else gemini_relay.network_gateway("mcp_bench", network, DOCS_URL))


class CredentialedServersRelay:
    """gbench's relay between the model container's bridge (gbench/docker/mcp_bench_bridge.py)
    and the credentialed-servers container, a thread of this process.

    It listens where the model container reaches this host (the network's gateway; 127.0.0.1 with
    the host network) on a free port. A session is one TCP connection: a JSON line {"token", "op",
    "server", "port"} from the bridge, answered "OK" or "ERR <reason>", then the stream. The token
    is drawn per run (not a credential of any service: it keeps other containers on the network
    out). `op` is stdio or http-start for a server of CREDENTIALED_SERVERS of that transport, or
    http-connect for a port an open http-start session reported; anything else is refused. Each
    session runs `docker exec -i <credentialed container> python3 CREDENTIALED_SCRIPT <mode>
    <server|port>` (fixed arguments, no shell) and waits for the script's first line, which says
    whether the session is up (the stdio server's process started, the HTTP server accepts
    connections, the connection to it is open): only then is the answer OK (for http-start
    followed by that line, with the server's port); a session that did not come up, or whose
    docker exec ended without the line (the container gone, the exec refused) within
    READY_TIMEOUT_S, is answered "ERR not started: <why>" and recorded as a failure. After OK the
    bridge's bytes go to the session unchanged, and what the session sends back comes in frames:
    "D<hex length>\n" and the bytes, then, when the credentialed side ends, "E{"exit": <docker
    exec's exit code>}\n". The script exits 0 however a session that came up ends, so another exit
    code there is docker exec's own failure; with the bridge still attached it is recorded as a
    failure too, and a stream without the end frame tells the bridge the session was lost. The
    credentialed container masks every key value in what comes back. No key passes through here.
    Each recorded failure keeps its time (`failure_times`; the first ones in detail, `failures`),
    so gbench can tell the run's failures from the pre-run check's."""

    #: The session kinds and the credentialed script's mode for each.
    OPS = {"stdio": "stdio", "http-start": "http", "http-connect": "connect"}
    HEADER_LIMIT = 4096
    HEADER_TIMEOUT_S = 15
    MAX_SESSIONS = 64
    #: How long a session's docker exec may take to end after its input closed.
    EXIT_WAIT_S = 15
    #: How long each session kind may take to come up (the script's first line): an HTTP server
    #: gets the script's own wait for it to accept connections (90 s) and a margin.
    READY_TIMEOUT_S = {"stdio": 30, "http-start": 150, "http-connect": 30}
    #: The most failures kept with their time in the record (every one is counted in `failed`).
    FAILURES_KEPT = 50

    def __init__(self, container: str, bind_host: str,
                 servers: Optional[Dict[str, str]] = None) -> None:
        self.container = container
        self.bind_host = bind_host
        #: {session name (mcp_bench_run.slug of the server): transport}
        self.servers = dict(servers if servers is not None else credentialed_sessions())
        self.token = secrets.token_hex(24)
        self._server: Optional[socketserver.ThreadingTCPServer] = None
        #: The port the relay listened on (kept after stop(), for the record).
        self._port = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.MAX_SESSIONS)
        self._ports: Dict[int, int] = {}
        self._procs: set = set()
        self._stopping = False
        self.sessions: Dict[str, int] = {}
        self.refused: Dict[str, int] = {}
        self.failed: Dict[str, int] = {}
        #: [{t, op, server, why}]: the sessions that failed on the relay's side (not opened, or
        #: ended by a docker exec failure while the bridge was attached), in order (the first
        #: FAILURES_KEPT), and the time of every one.
        self.failures: List[Dict[str, Any]] = []
        self.failure_times: List[float] = []

    # -- lifecycle ---------------------------------------------------------------------------- #
    def start(self) -> "CredentialedServersRelay":
        relay = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self):
                relay._handle(self.request)

        class _Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True
            address_family = socket.AF_INET6 if ":" in self.bind_host else socket.AF_INET

        try:
            self._server = _Server((self.bind_host, 0), _Handler)
        except OSError as exc:
            raise infra_required("mcp_bench", "gbench's relay to the credentialed-servers "
                                 f"container could not listen on {self.bind_host} "
                                 f"({type(exc).__name__}: {exc.strerror or exc}).",
                                 DOCS_URL) from exc
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.2}, daemon=True,
                                        name="mcp-bench-credentialed-relay")
        self._thread.start()
        logger.info("mcp_bench: relay to the credentialed servers listening on %s", self.address)
        return self

    def stop(self) -> None:
        self._stopping = True    # sessions cut from here on are not failures of the run
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    @property
    def address(self) -> str:
        """<host>:<port> the relay listens (or, after stop(), listened) on."""
        port = self._server.server_address[1] if self._server else self._port
        host = f"[{self.bind_host}]" if ":" in self.bind_host else self.bind_host
        return f"{host}:{port}"

    def client_env(self) -> Dict[str, str]:
        """The variables the model container's launcher needs (pass them to docker by name)."""
        return {"MCP_CREDENTIALED_RELAY": self.address, "MCP_CREDENTIALED_TOKEN": self.token}

    def record(self) -> Dict[str, Any]:
        """What a result records about the relay (never the token or a key)."""
        with self._lock:
            return {"listens": self.address, "container": self.container,
                    "servers": dict(sorted(self.servers.items())),
                    "sessions": dict(sorted(self.sessions.items())),
                    "refused": dict(sorted(self.refused.items())),
                    "failed": dict(sorted(self.failed.items())),
                    "failures": [dict(f) for f in self.failures],
                    "failure_times": list(self.failure_times)}

    def _fail(self, op: str, server: str, kind: str, why: str) -> None:
        """Record a session that failed on the relay's side: counted in `failed` by `kind`, its
        time in `failure_times`, and the first FAILURES_KEPT with their details in `failures`."""
        now = round(time.time(), 3)
        with self._lock:
            self.failed[kind] = self.failed.get(kind, 0) + 1
            self.failure_times.append(now)
            if len(self.failures) < self.FAILURES_KEPT:
                self.failures.append({"t": now, "op": op, "server": server,
                                      "why": redact_text(str(why))[:300]})
        logger.warning("mcp_bench: credentialed-server session %s %s failed: %s", op, server,
                       redact_text(str(why))[:300])

    # -- sessions ----------------------------------------------------------------------------- #
    def _bump(self, table: Dict[str, int], key: str) -> None:
        with self._lock:
            table[key] = table.get(key, 0) + 1

    def _refuse(self, sock: socket.socket, why: str) -> None:
        self._bump(self.refused, why)
        try:
            sock.sendall(f"ERR {why}\n".encode("utf-8"))
        except OSError:
            pass

    @staticmethod
    def _read_line(read, limit: int) -> bytes:
        buf = b""
        while not buf.endswith(b"\n") and len(buf) <= limit:
            chunk = read(1)
            if not chunk:
                break
            buf += chunk
        return buf

    def _handle(self, sock: socket.socket) -> None:
        sock.settimeout(self.HEADER_TIMEOUT_S)
        try:
            header = json.loads(self._read_line(sock.recv, self.HEADER_LIMIT).decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            self._refuse(sock, "bad header")
            return
        if not isinstance(header, dict) or not hmac.compare_digest(
                str(header.get("token") or "").encode("utf-8"), self.token.encode("utf-8")):
            self._refuse(sock, "token")
            return
        op, server = header.get("op"), header.get("server")
        if op not in self.OPS or server not in self.servers:
            self._refuse(sock, "op or server")
            return
        if (op == "stdio") != (self.servers[server] == "stdio"):
            self._refuse(sock, "transport")
            return
        arg = str(server)
        if op == "http-connect":
            port = header.get("port")
            with self._lock:
                live = isinstance(port, int) and not isinstance(port, bool) and port in self._ports
            if not live:
                self._refuse(sock, "port")
                return
            arg = str(port)
        if not self._slots.acquire(blocking=False):
            self._refuse(sock, "busy")
            return
        try:
            self._session(sock, op, arg, str(server))
        finally:
            self._slots.release()

    @staticmethod
    def _ready_line(proc: "subprocess.Popen", timeout_s: float, limit: int) -> Optional[bytes]:
        """The session's first line from the credentialed script (b"" when docker exec ended
        without one, None when none came within `timeout_s`). Read a byte at a time, so nothing
        after the line is taken from the stream."""
        fd = proc.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        buf = b""
        while not buf.endswith(b"\n") and len(buf) <= limit:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], min(left, 1.0))
            if not ready:
                continue
            chunk = os.read(fd, 1)
            if not chunk:
                break
            buf += chunk
        return buf

    @staticmethod
    def _stderr_tail(proc: "subprocess.Popen") -> Dict[str, bytes]:
        """Drain docker exec's stderr in a thread (so it never blocks), keeping its last bytes."""
        tail = {"text": b""}

        def drain():
            try:
                while True:
                    chunk = proc.stderr.read(4096)
                    if not chunk:
                        break
                    tail["text"] = (tail["text"] + chunk)[-2000:]
            except (OSError, ValueError):
                pass
        threading.Thread(target=drain, daemon=True).start()
        return tail

    def _wait(self, proc: "subprocess.Popen", timeout_s: float) -> Optional[int]:
        """docker exec's exit code, killing it after `timeout_s`."""
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return proc.returncode

    def _session(self, sock: socket.socket, op: str, arg: str, server: str = "") -> None:
        argv = ["docker", "exec", "-i", self.container, "python3", CREDENTIALED_SCRIPT,
                self.OPS[op], arg]
        server = server or arg
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, bufsize=0)
        except OSError as exc:
            self._fail(op, server, f"docker exec: {type(exc).__name__}",
                       f"docker exec could not run ({type(exc).__name__})")
            self._refuse(sock, "docker exec failed")
            return
        with self._lock:
            self._procs.add(proc)
        errors = self._stderr_tail(proc)
        port = None
        client_gone = threading.Event()
        try:
            # The session is up only when the credentialed script says so (its first line).
            line = self._ready_line(proc, self.READY_TIMEOUT_S[op], self.HEADER_LIMIT)
            try:
                status = json.loads((line or b"").decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                status = {}
            status = status if isinstance(status, dict) else {}
            up = status.get("ready") is True and (
                op != "http-start" or (isinstance(status.get("port"), int)
                                       and not isinstance(status.get("port"), bool)))
            if not up:
                if line is None:
                    why = f"no answer within {self.READY_TIMEOUT_S[op]} s"
                    proc.kill()
                else:
                    why = str(status.get("error") or "docker exec ended without an answer")
                rc = self._wait(proc, self.EXIT_WAIT_S)
                err = " ".join(errors["text"].decode("utf-8", "replace").split())[-200:]
                if not self._stopping:
                    self._fail(op, server, f"{op} not started",
                               f"{why} (exit {rc}{'; ' + err if err else ''})")
                self._refuse(sock, f"not started: {redact_text(why)[:200]}")
                return
            self._bump(self.sessions, op)
            if op == "http-start":
                port = status["port"]
                with self._lock:
                    self._ports[port] = self._ports.get(port, 0) + 1
            sock.settimeout(None)
            sock.sendall(b"OK\n" + (line if op == "http-start" else b""))
            threading.Thread(target=self._feed, args=(sock, proc, client_gone),
                             daemon=True).start()
            while True:
                data = proc.stdout.read(65536)
                if not data:
                    break
                sock.sendall(b"D%x\n" % len(data) + data)
            # The credentialed side ended: say how, so the bridge can tell an end from a loss.
            rc = self._wait(proc, self.EXIT_WAIT_S)
            if rc != 0 and not client_gone.is_set() and not self._stopping:
                err = " ".join(errors["text"].decode("utf-8", "replace").split())[-200:]
                self._fail(op, server, f"{op} ended (exit {rc})",
                           f"the session ended on the credentialed side (docker exec exit {rc}"
                           f"{'; ' + err if err else ''})")
            sock.sendall(b"E" + json.dumps({"exit": rc}).encode("utf-8") + b"\n")
        except OSError:
            pass                    # the bridge went away
        finally:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                proc.stdin.close()
            except OSError:
                pass
            if proc.returncode is None:
                self._wait(proc, self.EXIT_WAIT_S)
            with self._lock:
                self._procs.discard(proc)
                if port is not None:
                    self._ports[port] -= 1
                    if self._ports[port] <= 0:
                        del self._ports[port]

    @staticmethod
    def _feed(sock: socket.socket, proc: "subprocess.Popen",
              client_gone: Optional[threading.Event] = None) -> None:
        """The bridge's bytes to the session; `client_gone` is set when the bridge closed its
        side (its client ended the session) or went away, never when the session's own end
        stops the copy."""
        try:
            while True:
                try:
                    data = sock.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    if client_gone is not None:
                        client_gone.set()
                    break
                try:
                    proc.stdin.write(data)
                    proc.stdin.flush()
                except (OSError, ValueError):
                    break           # the credentialed side ended; the session reports how
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass


def credentialed_sessions() -> Dict[str, str]:
    """{session name: transport} of CREDENTIALED_SERVERS (the session name is the server's name
    lowercased, other characters as one dash, as mcp_bench_run.slug writes it)."""
    import re
    return {re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"): transport
            for name, (_key, transport) in CREDENTIALED_SERVERS.items()}


class CredentialedServers:
    """The credentialed-servers container and gbench's relay to it.

    The container is a second container of the image, started with `docker run -d --init --pull
    never --entrypoint sleep <image> infinity` on a docker network of its own (created per run and
    checked for the cloud metadata server, and for the credentialed servers' API host names, by
    container_guard.require_metadata_blocked before it starts), with the five server keys by name
    and nothing else of the run: no port, no volume, no service. Its only work is the sessions the
    relay opens with `docker exec` (gbench/docker/mcp_bench_credentialed.py), each running one of
    the five credentialed servers, none of which runs model text. The model container never gets
    a key; it reaches these servers through the bridge and the relay (CredentialedServersRelay)."""

    def __init__(self, image: str, base: str, model_network: str) -> None:
        self.image = image
        self.network = f"{base}-cred-net"
        self.container = f"{base}-cred"
        self.model_network = model_network
        self.relay: Optional[CredentialedServersRelay] = None
        self._created: List[str] = []
        self.running_at_stop: Optional[Dict[str, Any]] = None

    def start(self) -> "CredentialedServers":
        try:
            proc = subprocess.run(["docker", "network", "create", self.network],
                                  capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise infra_required("mcp_bench", "`docker network create` for the "
                                     "credentialed-servers container failed: "
                                     f"{redact_text((proc.stderr or proc.stdout).strip())[-300:]}",
                                     DOCS_URL)
            self._created.append("network")
            container_guard.require_metadata_blocked("mcp_bench", self.image, self.network,
                                                     DOCS_URL, resolve=credentialed_dns_names())
            cmd = ["docker", "run", "-d", "--init", "--pull", "never", "--name", self.container,
                   "--network", self.network]
            for key in _SERVER_KEYS:
                cmd += ["-e", key]
            cmd += ["--entrypoint", "sleep", self.image, "infinity"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                  env={**os.environ})
            self._created.append("container")
            if proc.returncode != 0:
                raise infra_required("mcp_bench", "`docker run` of the credentialed-servers "
                                     "container failed: "
                                     f"{redact_text((proc.stderr or proc.stdout).strip())[-300:]}",
                                     DOCS_URL)
            self.relay = CredentialedServersRelay(
                self.container, _relay_bind_host(self.model_network)).start()
        except BaseException:
            self.stop()
            raise
        return self

    def client_env(self) -> Dict[str, str]:
        return self.relay.client_env() if self.relay else {}

    def running(self) -> Tuple[bool, str]:
        """(whether the credentialed-servers container is still running, what docker said)."""
        try:
            proc = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}",
                                   self.container], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"docker inspect failed ({type(exc).__name__})"
        state = (proc.stdout or "").strip()
        if proc.returncode != 0:
            return False, redact_text((proc.stderr or proc.stdout or "").strip())[-200:] or \
                f"docker inspect exit {proc.returncode}"
        return state == "true", f"State.Running={state or '?'}"

    def record(self) -> Dict[str, Any]:
        return {"container": self.container, "network": self.network,
                "servers": sorted(CREDENTIALED_SERVERS),
                "keys": list(_SERVER_KEYS),
                "metadata_guard": (f"unreachable from the docker network {self.network} "
                                   "(container_guard.require_metadata_blocked)"),
                "dns_names_checked": credentialed_dns_names(),
                "relay": self.relay.record() if self.relay else None,
                # whether the container still ran when gbench stopped it (right after the run)
                "running_at_stop": self.running_at_stop}

    def stop(self) -> None:
        if self.relay is not None:
            self.relay.stop()
        if "container" in self._created and self.running_at_stop is None:
            running, state = self.running()
            self.running_at_stop = {"running": running, "state": state}
        if "container" in self._created:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
        if "network" in self._created:
            subprocess.run(["docker", "network", "rm", self.network], capture_output=True)
        self._created = []


def credentialed_run_record(record: Dict[str, Any], run_started: float) -> Dict[str, Any]:
    """What CredentialedServers.record() says about the run that started at `run_started`:
    {sessions_failed (the relay's session failures since then), failures (those kept in detail),
    container_running_at_end (whether the credentialed-servers container still ran when gbench
    stopped it right after the run; None when not recorded), container_state (what docker
    said)}."""
    relay = (record or {}).get("relay") or {}
    failures = [dict(f) for f in relay.get("failures") or []
                if isinstance(f, dict) and float(f.get("t") or 0) >= run_started]
    times = [t for t in relay.get("failure_times") or [] if float(t) >= run_started]
    stopped = (record or {}).get("running_at_stop")
    stopped = stopped if isinstance(stopped, dict) else {}
    running = stopped.get("running")
    return {"sessions_failed": max(len(times), len(failures)), "failures": failures,
            "container_running_at_end": bool(running) if running is not None else None,
            "container_state": str(stopped.get("state") or "not recorded")}


def unconfirmed_bridge_failures(unscored: Dict[str, Any], cred_run: Dict[str, Any]) -> int:
    """The tasks left out as credentialed_relay (the bridge's failure file in the model container)
    while gbench's relay recorded no session failure during the run and the credentialed-servers
    container still ran at its end: nothing on gbench's side corroborates those records. The
    bridge also records sessions that never reached the relay (it did not answer), and a process
    in the model container can write that file, so the two cannot be told apart; the count is
    named, the tasks stay out of the score (the image's aggregation leaves them out)."""
    n = int(((unscored or {}).get("by_kind") or {}).get("credentialed_relay") or 0)
    if not n or cred_run.get("sessions_failed") or cred_run.get("container_running_at_end") is False:
        return 0
    return n


def start_credentialed_servers(image: str, base: str, model_network: str) -> CredentialedServers:
    """CredentialedServers(image, base, model_network), started (infra_required when it cannot)."""
    return CredentialedServers(image, base, model_network).start()


def endpoint_userinfo_refusal(base_url: str) -> Optional[str]:
    """The infra_required reason when the model endpoint URL carries a user name or password
    (None otherwise): the endpoint URL reaches the model container, where model text runs, so no
    credential may ride in it."""
    from urllib.parse import urlsplit
    try:
        netloc = urlsplit((base_url or "").strip()).netloc
    except ValueError:
        return "the model endpoint URL cannot be parsed"
    if "@" in netloc:
        return ("the model endpoint URL carries a user name or password; it would be handed to "
                "the MCP-Bench container, where model-driven servers run model text, so gbench "
                "refuses it. Serve the model at a URL without user info (reachable from the "
                "container's docker network).")
    return None


def _start_judge_relay(network: str) -> "gemini_relay.GeminiRelay":
    """gbench's Gemini relay for the in-container judge, started: it listens where a container on
    `network` reaches this host (the network's gateway, `docker network inspect`; 127.0.0.1 with
    the host network), takes the per-run placeholder the container holds, adds the real key
    (base.gemini_api_keys) and forwards only the judge's call (JUDGE_RELAY_CALLS) to the checked
    GEMINI_OPENAI_BASE_URL. infra_required when no key is set or the address cannot be bound."""
    bind = _relay_bind_host(network)
    return gemini_relay.GeminiRelay("mcp_bench", bind, docs_url=DOCS_URL,
                                    openai_calls=JUDGE_RELAY_CALLS, native_methods=(),
                                    name="mcp-bench-judge-relay").start()


def read_harness_log(path: str) -> Optional[Dict[str, Any]]:
    """The image's per-task harness record (HARNESS_LOG, mcp_bench_run.HarnessAccounting.record),
    or None when it is absent, unreadable, of another layout version or has no task list."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            log = json.load(fh)
    except (OSError, ValueError):
        return None
    if (not isinstance(log, dict) or log.get("version") != HARNESS_LOG_VERSION
            or not isinstance(log.get("tasks"), list)):
        return None
    return log


def harness_accounting(log: Dict[str, Any]) -> Dict[str, Any]:
    """What the harness record says about the run's tasks: {tasks_seen, tasks_scored,
    scored_per_file ({task file basename: tasks scored}), unscored ({count, by_kind, task_ids,
    details}), judge_calls_failed (failed judge calls in the scored tasks: their judge scores
    average fewer stability runs) and judge_calls_failed_tasks (their ids), judge_models (replies
    per Gemini model over the run) and judge_models_per_task ({task id: {model: replies}}), dns
    ({names, rounds, outages, name_failures})}. A task is scored when upstream completed it and
    no harness error fell in it."""
    tasks = [t for t in log.get("tasks") or [] if isinstance(t, dict)]
    flagged = [t for t in tasks if t.get("kinds")]
    scored = [t for t in tasks if not t.get("kinds") and t.get("status") == "completed"]
    by_kind: Dict[str, int] = {}
    for t in flagged:
        for kind in t.get("kinds") or []:
            by_kind[str(kind)] = by_kind.get(str(kind), 0) + 1
    per_file: Dict[str, int] = {}
    for t in scored:
        key = str(t.get("file") or "")
        per_file[key] = per_file.get(key, 0) + 1
    details = []
    for t in flagged:
        for d in t.get("details") or []:
            if len(details) < 20:
                details.append(f"{t.get('task_id')}: {redact_text(str(d))[:200]}")
    return {
        "tasks_seen": len(tasks),
        "tasks_scored": len(scored),
        "scored_per_file": dict(sorted(per_file.items())),
        "unscored": {"count": len(flagged), "by_kind": dict(sorted(by_kind.items())),
                     "task_ids": sorted(str(t.get("task_id")) for t in flagged),
                     "details": details},
        "judge_calls_failed": sum(int(t.get("judge_calls_failed") or 0) for t in scored),
        "judge_calls_failed_tasks": sorted(str(t.get("task_id")) for t in scored
                                           if int(t.get("judge_calls_failed") or 0)),
        "judge_models": _model_counts(t.get("judge_models") for t in tasks),
        "judge_models_per_task": {str(t.get("task_id")): dict(sorted(t["judge_models"].items()))
                                  for t in tasks if isinstance(t.get("judge_models"), dict)
                                  and t["judge_models"]},
        "dns": {"names": len(log.get("dns_names") or []),
                "rounds": int(log.get("dns_rounds") or 0),
                "outages": list(log.get("dns_outages") or []),
                "name_failures": dict(log.get("dns_name_failures") or {})},
    }


def _model_counts(per_task: Any) -> Dict[str, int]:
    """{model: replies} summed over per-task {model: replies} dicts."""
    out: Dict[str, int] = {}
    for models in per_task:
        for model, n in (models or {}).items() if isinstance(models, dict) else ():
            out[str(model)] = out.get(str(model), 0) + int(n or 0)
    return dict(sorted(out.items()))


def _is_multi_file(results_json: Any) -> bool:
    """Whether `results_json` is a {task_file: metrics} map (compute_overall_score's test)."""
    return (isinstance(results_json, dict) and bool(results_json)
            and all(isinstance(v, dict) for v in results_json.values())
            and any(("task_completion_score" in v or "input_schema_compliance" in v)
                    for v in results_json.values() if isinstance(v, dict)))


def scored_results(results_json: Any, scored_per_file: Dict[str, int]) -> Any:
    """`results_json` without the task files in which no task was scored (upstream then keeps
    the metrics of the file before it, or none), for a multi-file result; a single-file result
    as it is."""
    if not _is_multi_file(results_json):
        return results_json
    return {k: v for k, v in results_json.items()
            if scored_per_file.get(os.path.basename(str(k)), 0) > 0}


def mcp_bench_plan(limit: Optional[int]) -> Dict[str, Any]:
    """Log the run's planned workload (one line) and return its `eval_limit_plan`: --eval-limit N
    is N tasks in total (MCP_TASK_LIMIT: the launcher's one seeded draw over the flattened task
    list, balanced across the server groups, never N per group), each run once. A full run runs
    every task the launcher keeps (the manifest's tasks_kept)."""
    if limit and int(limit) > 0:
        n = min(int(limit), UPSTREAM_TASK_COUNT)
        return eval_limit_plan("mcp_bench", int(limit), n, attempts=1, unit="task_runs")
    logger.info("[mcp_bench] full run (no --eval-limit): every task the launcher keeps x 1 "
                "attempt")
    return {"limit": None, "samples": None, "per_setting": None, "attempts_per_sample": 1,
            "task_runs": None}


def plan_from_kept(plan: Dict[str, Any], tasks_kept: Any) -> Dict[str, Any]:
    """The run's `eval_limit_plan` with the tasks the launcher actually kept (its manifest's
    tasks_kept: a shard, a limit above the selectable set, or the servers' availability narrows
    the logged count), one run each; the plan as it is when the manifest has no count."""
    if not isinstance(tasks_kept, int) or isinstance(tasks_kept, bool):
        return plan
    out = dict(plan or {})
    if out.get("samples") is not None and out["samples"] != tasks_kept:
        out["samples_logged"] = out["samples"]
    out["samples"] = out["task_runs"] = int(tasks_kept)
    return out


def run_mcp_bench(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical MCP-Bench: upstream agent loop (28 servers, in-container) + gbench Gemini
    cascade judge, over all 104 tasks."""
    ok, reason = check_mcp_bench_prerequisites()
    if not ok:
        raise infra_required("mcp_bench", reason, DOCS_URL)
    # --shard I/N: validated here (a malformed GBENCH_SHARD raises before anything starts) and
    # forwarded normalized; the launcher selects the shard's task ids before MCP_TASK_LIMIT.
    shard = current_shard()
    shard_label = shard.label if shard and shard.partial else None
    # The endpoint URL goes into the model container: it may carry no credential.
    refusal = endpoint_userinfo_refusal(base_url)
    if refusal:
        raise infra_required("mcp_bench", refusal, DOCS_URL)
    # The five server keys the README requires go to the credentialed-servers container only;
    # every one must be set before anything starts.
    missing_keys = [k for k in _REQUIRED_KEYS if not (os.environ.get(k) or "").strip()]
    if missing_keys:
        raise infra_required(
            "mcp_bench", "required API keys unset: " + ", ".join(
                f"{k} ({_REQUIRED_KEYS[k]})" for k in missing_keys)
            + "; the MCP-Bench README lists all five as required, and gbench runs no task without "
            "every server.", DOCS_URL)

    # The model under test drives tools that run commands and evaluate expressions in this
    # container (for example the OSINT server's nmap/whois/dig on a target it names, and sympy
    # expression parsing in the scientific-computing server), with network access: the container
    # never starts on a docker network where the cloud metadata server answers, nor on one where
    # the servers' API host names (and a named model endpoint) do not resolve.
    network = _network()
    endpoint = container_endpoint(base_url, network)
    container_guard.require_metadata_blocked("mcp_bench", _image(), network, DOCS_URL,
                                             resolve=dns_names(endpoint))

    # The model-under-test's sampling (see the module note): 1.0 with --thinking, 0.0 without,
    # unless the operator sets another, the thinking switch of the run's mode, and the
    # token-limit policy.
    temperature, temperature_source = resolve_official_sampling(
        "mcp_bench", MCP_BENCH_SAMPLING, thinking=enable_thinking)
    plan = resolve_token_budget("mcp_bench", MCP_BENCH_SAMPLING, thinking=enable_thinking,
                                max_output_tokens=kwargs.get("max_output_tokens"))
    budget = plan.max_tokens if plan.policy == "reasoning_run_budget" else None
    # With the --thinking budget, each request may take as long as gbench's own requests of that
    # budget (base.request_timeout_s); upstream's client keeps openai's 600 s otherwise.
    timeout = request_timeout_s(budget) if budget is not None else None
    expected_settings = dict(enable_thinking=bool(enable_thinking), max_tokens=budget,
                             temperature=temperature, request_timeout_s=timeout)

    workdir = tempfile.mkdtemp(prefix="gbench_mcp_bench_")
    os.chmod(workdir, 0o777)  # the container writes /out as its own uid
    orch_name = "gbench_mcp_bench_" + os.path.basename(workdir)

    def _reap():
        # docker --rm only fires on a clean container EXIT; on a subprocess timeout the client is
        # killed while the container keeps running, so reap the named container in finally + on
        # timeout to avoid leaking it across a long or multi-node run.
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    relay = None
    cred = None
    try:
        # The five credentialed servers run in a container of their own, the only one that gets
        # the server keys; the model container reaches them through the bridge and gbench's relay.
        cred = start_credentialed_servers(_image(), orch_name, network)

        # Every task needs its servers: check them all before the (long) run, never shrink the set
        # and never run with a server that cannot serve.
        probe = provisioning_probe(_image(), cred.client_env())
        gaps = provisioning_gaps(probe)
        if gaps:
            raise infra_required(
                "mcp_bench", "not every MCP-Bench server can serve, so the full task set cannot "
                "run (gbench does not run the provisioned servers' tasks alone): "
                + redact_text("; ".join(gaps))
                + ". Set the keys (the MCP-Bench README lists all five as required), provide "
                "network egress, and fix the image for any server that fails the connection "
                "check.", DOCS_URL)

        # The judge's Gemini key stays in this process: the container gets the relay's per-run
        # placeholder and addresses, and the relay adds the key (model-driven servers run model
        # text in the container, where a real key could be read).
        relay = _start_judge_relay(network)
        # The relay placeholder, the relays' addresses and the endpoint URL go by name only (`-e
        # NAME`, the value read from `env`), so none is on the argv, in /proc/<pid>/cmdline, or in
        # the message of a TimeoutExpired, which carries the command. No server key is in `env`.
        env = {k: v for k, v in os.environ.items() if k not in _SERVER_KEYS}
        env.update(relay.client_env())
        env.update(cred.client_env())
        env["MCP_ENDPOINT"] = endpoint
        cmd = ["docker", "run", "--rm", "--pull", "never", "--name", orch_name, "--network", network,
               "-v", f"{workdir}:/out:rw",
               "-e", "MCP_ENDPOINT", "-e", f"MCP_MODEL={model_name}",
               "-e", "GEMINI_API_KEY", "-e", "GEMINI_API_KEYS", "-e", "GEMINI_OPENAI_BASE_URL",
               "-e", gemini_relay.RELAY_URL_ENV,
               "-e", "MCP_CREDENTIALED_RELAY", "-e", "MCP_CREDENTIALED_TOKEN",
               "-e", "MCP_OUTPUT=/out/mcp_bench_results.json",
               "-e", "MCP_MANIFEST=/out/subset_manifest.json",
               "-e", f"MCP_HARNESS_LOG=/out/{HARNESS_LOG}",
               "-e", "MCP_DNS_NAMES=" + ",".join(server_dns_names()),
               # the model request settings the image applies (mcp_bench_request_policy.py)
               "-e", f"MCP_ENABLE_THINKING={'true' if enable_thinking else 'false'}",
               "-e", f"MCP_REQUEST_LOG=/out/{REQUEST_LOG}"]
        if budget is not None:
            cmd += ["-e", f"MCP_MAX_TOKENS={budget}"]
        if timeout is not None:
            cmd += ["-e", f"MCP_REQUEST_TIMEOUT_S={timeout}"]
        cmd += ["-e", f"MCP_TEMPERATURE={temperature}"]
        # The run's judge cascade, always (the judge never falls back to a default of its own),
        # then the cascade's rounds and backoff and the suite's protocol knobs when set.
        cmd += ["-e", f"GBENCH_JUDGE_MODELS={gemini_cascade_env()['GBENCH_JUDGE_MODELS']}"]
        for env_name, mcp_name in (("GBENCH_MCP_BENCH_DISTRACTION", "MCP_DISTRACTION_COUNT"),
                                   ("GBENCH_MCP_BENCH_DISABLE_STABILITY", "MCP_DISABLE_STABILITY"),
                                   ("GBENCH_MCP_BENCH_ASSUME_NETWORK", "MCP_ASSUME_NETWORK"),
                                   ("GBENCH_JUDGE_CASCADE_ROUNDS", "GBENCH_JUDGE_CASCADE_ROUNDS"),
                                   ("GBENCH_JUDGE_BACKOFF", "GBENCH_JUDGE_BACKOFF")):
            v = os.environ.get(env_name)
            if v:
                cmd += ["-e", f"{mcp_name}={v}"]
        lim = kwargs.get("limit")
        if lim and int(lim) > 0:
            cmd += ["-e", f"MCP_TASK_LIMIT={int(lim)}"]
        workload = mcp_bench_plan(lim)
        # Forward the shard ONLY because the launcher consumes it: mcp_bench tasks carry stable,
        # globally-unique task_ids, so the launcher selects an interleaved subset (round-robin over
        # the sorted full task-id list) BEFORE MCP_TASK_LIMIT, matching the native path's compose
        # order. The manifest's tasks_kept then reflects only the shard.
        if shard_label:
            cmd += ["-e", f"GBENCH_SHARD={shard_label}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_MCP_BENCH_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("mcp_bench: docker run %s", _image())
        run_started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "mcp_bench", f"the mcp_bench container exceeded {timeout_s}s and was killed.",
                DOCS_URL) from exc
        if proc.returncode != 0:
            raise infra_required(
                "mcp_bench",
                f"the mcp_bench container failed (rc={proc.returncode}). "
                f"tail: {_tail(proc, 800)}", DOCS_URL)

        results_path = os.path.join(workdir, "mcp_bench_results.json")
        manifest_path = os.path.join(workdir, "subset_manifest.json")
        if not os.path.exists(results_path):
            raise infra_required(
                "mcp_bench", "the container produced no results file (a harness failure, not a 0%). "
                f"tail: {_tail(proc, 500)}", DOCS_URL)
        with open(results_path, encoding="utf-8") as f:
            results_json = json.load(f)
        manifest = None
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        request_log = read_model_request_log(os.path.join(workdir, REQUEST_LOG))
        harness_log = read_harness_log(os.path.join(workdir, HARNESS_LOG))
    finally:
        _reap()
        if relay is not None:
            relay.stop()
        if cred is not None:
            cred.stop()
        shutil.rmtree(workdir, ignore_errors=True)
    relay_record = relay.record()
    cred_record = cred.record()
    # The credentialed side of the run: the sessions the relay could not open or saw fail while a
    # bridge was attached, and whether the credentialed-servers container still ran at its end.
    cred_run = credentialed_run_record(cred_record, run_started)

    if manifest and manifest.get("unavailable_servers"):
        # The pre-run check passed, but when the run started the launcher found servers that
        # could not serve (a build artifact missing, egress gone, a key unset) and ran no task:
        # without them it is not the benchmark's task set.
        unresolved = sorted((manifest.get("network_check") or {}).get("unresolved") or {})
        raise infra_required(
            "mcp_bench", "when the run started the launcher found servers it could not serve ("
            + redact_text(", ".join(str(x) for x in manifest["unavailable_servers"]))
            + f"; network_up={manifest.get('network_up')}"
            + (f"; names that did not resolve: {', '.join(unresolved)}" if unresolved else "")
            + "), so no result is scored: the full task set needs every server. Provide network "
            "egress with name resolution and the server API keys, and fix the image for any "
            "server that did not build.", DOCS_URL)

    lim = kwargs.get("limit")
    if lim and int(lim) > 0 and not (manifest or {}).get("limit_draw"):
        raise infra_required(
            "mcp_bench", "the launcher does not report how it drew the --eval-limit subset (an "
            "image built before the seeded draw, which took the first tasks in file order); "
            "rebuild the image from gbench/docker/mcp_bench.Dockerfile.", DOCS_URL)
    if not results_json and shard_label and manifest and not manifest.get("tasks_kept"):
        # The shard holds no task: nothing ran on this node, so no score (never 0%, never an
        # error).
        reason = f"empty shard {shard_label}: no items on this node"
        empty: Dict[str, Any] = {
            "benchmark_type": "eval", "eval_name": "mcp_bench", "model_name": model_name,
            "status": "success", "status_reason": reason, "accuracy": None,
            "accuracy_withheld_reason": reason, "empty_shard": True, "tasks_run": 0,
            "total_questions": 0, "correct_answers": 0, "subset_manifest": manifest,
            "eval_limit_plan": plan_from_kept(workload, 0)}
        record_shard(empty, shard_label)
        return add_comparability_reason(empty, reason)
    if not results_json:
        raise infra_required(
            "mcp_bench", "no MCP-Bench task was run (the container wrote an empty result; the "
            "shard or --eval-limit may have selected no task).", DOCS_URL)

    replies = check_request_log("mcp_bench", request_log, expected_settings, enable_thinking)
    all_failed = every_model_request_failed(replies)
    if all_failed:
        raise infra_required(
            "mcp_bench", f"none of the run's {all_failed} model-under-test request(s) got a reply "
            "(each ended in an error of the served endpoint or of the connection), so every task's "
            "outcome is a harness error and no score is published. Check that the model endpoint "
            f"is reachable from the container's docker network ({network}; a model on this host "
            "must listen on that network's gateway address).", DOCS_URL)
    if harness_log is None:
        raise infra_required(
            "mcp_bench", f"the image wrote no per-task harness record ({HARNESS_LOG}, layout "
            f"version {HARNESS_LOG_VERSION}), so the tasks a harness error fell in cannot be kept "
            "out of the score; rebuild the image from gbench/docker/mcp_bench.Dockerfile and "
            "re-run.", DOCS_URL)
    acct = harness_accounting(harness_log)
    unscored = acct["unscored"]
    if not acct["tasks_seen"]:
        raise infra_required(
            "mcp_bench", "the container wrote results but its harness record lists no task, so "
            "the tasks a harness error fell in cannot be told apart; rebuild the image from "
            "gbench/docker/mcp_bench.Dockerfile and re-run.", DOCS_URL)
    if not acct["tasks_scored"]:
        raise infra_required(
            "mcp_bench", f"no task could be scored: of the {acct['tasks_seen']} task(s) run, "
            f"{unscored['count']} had a harness error ("
            + (", ".join(f"{k} x{v}" for k, v in unscored["by_kind"].items()) or "none")
            + (f"; {'; '.join(unscored['details'][:5])}" if unscored["details"] else "")
            + "), so every outcome is a harness error and no score is published.", DOCS_URL)
    scored = compute_overall_score(
        scored_results(results_json, acct["scored_per_file"]),
        {"per_file": {f: {"kept": n} for f, n in acct["scored_per_file"].items()}})
    if scored is None or any(v is None for v in scored["dimensions"].values()):
        missing = ([k for k, v in scored["dimensions"].items() if v is None] if scored
                   else ["all"])
        raise infra_required(
            "mcp_bench", "the container output lacks scored dimensions ("
            + ", ".join(missing) + ") although tasks were scored, so no Overall Score is "
            "published.", DOCS_URL)

    tasks_run = manifest.get("tasks_kept") if manifest else None

    # A sharded run covers a round-robin subset of the task set by construction, so it is never the
    # full set; tasks_run already reflects only the shard's kept tasks (denominator honesty).
    shard_spec = shard_label or ""
    sharded = bool(shard_label)
    limited = bool(lim and int(lim) > 0)
    is_full = not (sharded or limited)
    # Operator overrides of upstream's protocol (config/benchmark_config.yaml at the pinned
    # commit: distraction_servers_default 10, enable_judge_stability true, 5 runs), named below.
    overrides = []
    distraction = (os.environ.get("GBENCH_MCP_BENCH_DISTRACTION") or "").strip()
    if distraction and distraction != str(UPSTREAM_DISTRACTION_COUNT):
        overrides.append(f"{distraction} distraction servers per task instead of upstream's "
                         f"{UPSTREAM_DISTRACTION_COUNT}")
    if os.environ.get("GBENCH_MCP_BENCH_DISABLE_STABILITY"):
        overrides.append("judge stability off: one judge pass per task instead of upstream's "
                         f"{UPSTREAM_JUDGE_STABILITY_RUNS} randomized-order passes")
    official = MCP_BENCH_SAMPLING.record(temperature, temperature_source)
    official.update(plan.record())
    official["per_call_max_tokens"] = UPSTREAM_CAPS_TEXT
    if official.get("temperature_matches_official") is False:
        overrides.append(f"temperature {temperature} set by "
                         f"{temperature_source.split(':', 1)[-1]} (upstream sends none)")
    knob = kwargs.get("max_output_tokens")
    counts = (replies or {}).get("counts") or {}
    if budget is None:
        note = ((f"--max-output-tokens {knob} not applied: " if knob else "")
                + f"upstream's own caps are sent exactly ({UPSTREAM_CAPS_TEXT}; "
                "config/benchmark_config.yaml)")
    else:
        source = f"--max-output-tokens {knob}" if knob else f"the thinking floor {budget}"
        note = (f"{REASONING_CAP_NOTE} (upstream sends {UPSTREAM_CAPS_TEXT}); every request of "
                f"the model under test was sent this run's budget, {source}, cut to the room the "
                "context window leaves after the prompt, with a request timeout sized to it "
                f"({timeout} s to read the reply)")
    budget_notes = []
    if replies is None:
        budget_notes.append("the image wrote no model request log, so replies cut at max_tokens "
                            "are not counted")
    elif budget is not None and counts.get("cut_at_max_tokens"):
        budget_notes.append(
            f"{counts['cut_at_max_tokens']} model repl(ies) reached this run's max_tokens "
            f"({counts.get('cut_in_reasoning', 0)} inside the reasoning channel with no answer): "
            "upstream's caps were defined for non-reasoning generation and are not applied with "
            "--thinking; raise --max-output-tokens")
    harness_errors, failure_notes = model_request_failures(
        replies, timeout, "MCP-Bench",
        cancel_hint=(f" (for example at MCP-Bench's task_timeout of {UPSTREAM_TASK_TIMEOUT_S} s, "
                     "config/benchmark_config.yaml, which cancels a task still running)"))
    budget_notes += failure_notes
    if unscored["count"]:
        harness_errors.append({
            "error": "tasks left out of the score: a harness error fell in them",
            "tasks": unscored["count"], "by_kind": unscored["by_kind"],
            "task_ids": unscored["task_ids"], "details": unscored["details"]})
        budget_notes.append(
            f"{unscored['count']} of {acct['tasks_seen']} task(s) left out of the score because "
            "a harness error fell in them (" + ", ".join(
                f"{k} x{v}" for k, v in unscored["by_kind"].items()) + "); the Overall Score "
            f"covers the other {acct['tasks_scored']}")
    if cred_run["sessions_failed"] or cred_run["container_running_at_end"] is False:
        # The credentialed side failed during the run. The bridge records each such session in
        # the model container, so the task running then is already left out of the score
        # (credentialed_relay); this names the run's failures on the relay's side, a safety net
        # for a session the bridge could not record.
        harness_errors.append({
            "error": "credentialed-server sessions failed on gbench's relay, or the "
                     "credentialed-servers container stopped, during the run",
            "kind": "credentialed_relay",
            "sessions_failed": cred_run["sessions_failed"],
            "failures": cred_run["failures"][:10],
            "container_running_at_end": cred_run["container_running_at_end"],
            "container_state": cred_run["container_state"]})
        parts = []
        if cred_run["sessions_failed"]:
            parts.append(f"{cred_run['sessions_failed']} credentialed-server session(s) failed on "
                         "gbench's relay during the run" + (" (" + "; ".join(
                             f"{f.get('server')} {f.get('op')}: {f.get('why')}"
                             for f in cred_run["failures"][:3]) + ")"
                             if cred_run["failures"] else ""))
        if cred_run["container_running_at_end"] is False:
            parts.append("the credentialed-servers container was not running at the end of the "
                         f"run ({cred_run['container_state']})")
        budget_notes.append(redact_text("; ".join(parts)) + " (a harness error of the run; the "
                            "tasks the bridge recorded such a session in are left out of the "
                            "score as credentialed_relay)")
    unconfirmed = unconfirmed_bridge_failures(unscored, cred_run)
    if unconfirmed:
        budget_notes.append(
            f"{unconfirmed} task(s) left out as credentialed_relay on the bridge's record alone: "
            "gbench's relay recorded no session failure during the run (a session that never "
            "reached the relay, or a record written by a process in the model container, which "
            "gbench cannot tell apart)")
    if acct["judge_calls_failed"]:
        # A partial judge outage: some stability runs of these scored tasks have no evaluation.
        # Upstream's evaluator averages the runs that succeeded (kept as the score); the outage is
        # a harness error of the run, counted and named.
        harness_errors.append({
            "error": "judge calls of scored tasks failed on every model of the Gemini cascade "
                     "(upstream's evaluator averages those tasks' remaining stability runs)",
            "kind": "judge_outage_partial",
            "judge_calls_failed": acct["judge_calls_failed"],
            "task_ids": acct["judge_calls_failed_tasks"]})
        budget_notes.append(
            f"{acct['judge_calls_failed']} judge call(s) of {len(acct['judge_calls_failed_tasks'])} "
            "scored task(s) failed on every model of the Gemini cascade (a harness error of the "
            "run); upstream's evaluator averages those tasks' remaining stability runs")

    result: Dict[str, Any] = {
        "eval_limit_plan": plan_from_kept(workload, tasks_run),
        "benchmark_type": "eval",
        "eval_name": "mcp_bench",
        "model_name": model_name,
        # Model-under-test requests that ended in an endpoint or connection error, and tasks a
        # harness error fell in, are harness errors, never the model's outcome.
        "status": "completed_with_errors" if harness_errors else "success",
        "accuracy": round(scored["overall"] * 100.0, 2),   # headline = Overall Score (%)
        "overall_score": round(scored["overall"], 4),        # 0-1
        "dimensions": {k: (round(v, 4) if isinstance(v, (int, float)) else None)
                       for k, v in scored["dimensions"].items()},
        "per_file": scored.get("per_file"),
        "raw_metrics": results_json,
        "tasks_run": tasks_run,
        # Tasks in the Overall Score: upstream completed them and no harness error fell in them.
        "tasks_scored": acct["tasks_scored"],
        "tasks_unscored": unscored,
        "judge_calls_failed": acct["judge_calls_failed"],
        # The launcher's DNS watchdog over the servers' API host names while the tasks ran.
        "dns_watchdog": acct["dns"],
        "is_full": is_full,
        "shard": shard_spec or None,
        "subset_manifest": manifest,
        # The five server keys, each accepted by its service before the run (server_key_problems).
        "server_key_check": {k: ("the Maps Platform APIs its server calls: " + ", ".join(
            api for api, _ in GOOGLE_MAPS_KEY_CHECKS) if k == "GOOGLE_MAPS_API_KEY"
            else SERVER_KEY_CHECKS[k][0].split("?", 1)[0]) + ": accepted"
            for k in _REQUIRED_KEYS},
        # MCP-Bench's own connection check (utils/collect_mcp_info.py), run before the benchmark.
        "server_connection_check": {
            "servers_connected_with_tools": sum(
                1 for c in (probe.get("connections") or {}).values()
                if c.get("status") == "success" and c.get("tools")),
            "servers_checked": len(probe.get("connections") or {}),
            "tools_listed": sum(int(c.get("tools") or 0)
                                for c in (probe.get("connections") or {}).values()),
        },
        "judge": "gbench-gemini-cascade",
        # The Gemini models that answered the judge: {model: replies} over the run and per task,
        # and the cascade the run passed into the container.
        "gemini_models_answered": acct["judge_models"],
        "judge_models_per_task": acct["judge_models_per_task"],
        "gemini_cascade": judge_cascade(),
        "judge_calls_failed_tasks": acct["judge_calls_failed_tasks"],
        # The judge's key stayed in this process: the relay's record (never a key).
        "gemini_relay": relay_record,
        # The credentialed-servers container (the only one with the server keys) and gbench's
        # relay to it (never a key or the relay's token), and what failed there during the run:
        # the relay's session failures and whether the container still ran at its end.
        "credentialed_servers": cred_record,
        "credentialed_servers_run": cred_run,
        # tasks left out as credentialed_relay while gbench's relay saw no failure in the run
        "unconfirmed_bridge_failures": unconfirmed,
        "thinking": bool(enable_thinking),
        "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": official,
        "max_output_tokens_note": note,
        # The image's record of the settings it applied and of the model-under-test's replies:
        # replies, cut_at_max_tokens (finish_reason length), cut_in_reasoning (no answer text),
        # fitted_to_context / refitted_after_rejection (--thinking budget cut to the window),
        # failed_requests / timed_out (endpoint or connection errors: harness errors),
        # rejected_over_context (left to upstream's handling), cancelled (abandoned in flight).
        "model_request_settings": (replies or {}).get("settings"),
        "model_replies": dict(counts) if replies is not None else None,
        "model_failed_by_error": (replies or {}).get("failed_by_error"),
        "model_max_tokens_sent": (replies or {}).get("max_tokens_sent"),
        "model_request_timeout_s": timeout,
        "harness_errors": harness_errors,
        # The container's docker network: the metadata-server guard checked it before the run,
        # and that the host names below resolve on it.
        "docker_network": network,
        "dns_names_checked": dns_names(endpoint),
        "sampling": ("the upstream agent loop builds the model-under-test's requests; gbench's "
                     "in-image request settings set the chat template's thinking switch to the "
                     "run's mode, send temperature 1.0 with --thinking and 0.0 without unless "
                     "the operator sets another, and follow the token-limit policy (upstream's caps without --thinking, the run's budget "
                     "with it)"),
        "metric": ("MCP-Bench Overall Score (0-1) = mean of four dimensions (schema understanding "
                   "[rule-based: valid tool-name rate, schema compliance, execution success], task "
                   "completion, tool usage, planning effectiveness [Gemini-"
                   "judged, 6 sub-dimensions x 5 randomized-order stability runs]). Agent loop + "
                   "28 servers delegated to the upstream runner; judge is gbench's Gemini cascade "
                   "(upstream prompts + stability protocol verbatim; model differs from o4-mini)."),
        # A run here is a gbench-internal number, not a like-for-like MCP-Bench leaderboard entry:
        # gbench grades with its standard Gemini cascade (a gbench convention) where the leaderboard
        # is o4-mini-graded.
        "leaderboard_comparable": False,
        "protocol_overrides": overrides,
        "leaderboard_comparable_reason": (
            "graded by gbench's standard Gemini cascade (a gbench convention); the published "
            "MCP-Bench leaderboard is o4-mini-graded, so this is a gbench-internal number"
            + "".join(f"; {o}" for o in overrides)
            + "".join(f"; {n}" for n in budget_notes)
            + (f"; --eval-limit ran only {int(lim)} tasks (a seeded draw balanced across the "
               "server groups)" if limited else "")
            + (f"; and only shard {shard_spec} (a round-robin subset by construction) was run"
               if sharded else "")),
        # MCP-Bench's runner runs the tasks one at a time; the suite's concurrency is not used.
        "execution": "sequential (MCP-Bench's runner runs one task at a time)",
    }
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    if attempts > 1:
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "mcp_bench: MCP-Bench runs each task once (its judge stability passes "
                           "re-grade that one run)"}
    return result
