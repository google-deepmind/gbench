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

# Evaluation Suite: wildclawbench
# Description: WildClawBench (InternLM real-world long-horizon autonomous agent benchmark - 60 tasks)

"""gbench native built-in runner for wildclawbench (Tool Use & Agentic Workflows).

Canonical WildClawBench (github.com/internlm/WildClawBench): 60 hand-built long-horizon tasks across
6 categories (EN+ZH). Each task runs in its own Docker container from the prebaked image
`wildclawbench-ubuntu:v1.3` (a 13.5GB `docker load` from HF), with the OpenClaw agent inside; after
the agent finishes, the task's own `grade()` (programmatic checks + an LLM judge) is `docker exec`'d
for a per-metric 0-1 score. gbench DELEGATES the whole run to the pinned upstream harness
(`eval/run_batch.py`) inside a LOCAL orchestrator container (gbench/docker/wildclawbench.Dockerfile),
run docker-OUT-of-docker: run_batch builds/runs each TASK container as a sibling on the host daemon.

What gbench changes, everything else VERBATIM. Every external model or search service the harness
calls, or offers the agent as a tool, goes through the Gemini API (GEMINI_API_KEYS, else
GEMINI_API_KEY), served by one Gemini proxy in the orchestrator:
  * the OpenClaw agent is pointed at the gbench /v1 model endpoint via a `my_api.json` custom
    provider (upstream's supported custom-endpoint path) whose baseUrl is the proxy's model relay
    (<proxy>/model/v1) and whose apiKey is a fixed placeholder: the proxy forwards the calls to the
    model endpoint and adds the endpoint's key there, so no model key reaches a task container;
  * the tasks' judge calls (all 43 judged tasks use `OpenAI(base_url=OPENROUTER_BASE_URL).chat...(
    model=JUDGE_MODEL)` and declare those keys in their `## Env`; 05_Creative_Synthesis_task_11 also
    calls its audio judge, openai/gpt-4o-mini-audio-preview, through the same client) go to the
    gbench Gemini judge cascade instead of OpenRouter; the judge PROMPTS and every request field
    but the model (temperature as the task sends it, none when it sends none) are untouched, and
    the judge calls that turn thinking off (THINKING_OFF_TASKS) are sent at Gemini thinking level
    low, the lowest level every cascade model accepts (none can turn thinking off). The
    one judge call that asks for OpenRouter's web search tool (04_Search_Retrieval_task_1's path
    verification) is answered by Gemini with Google Search grounding;
  * the OpenRouter helper model the prompts of 25 tasks offer the agent through the same variables
    (HELPER_MODEL_TASKS) is that same Gemini cascade;
  * the OpenClaw agent's `web_search` tool keeps the definition the image ships (its Brave provider:
    same name, description, parameters and result shape), but its request is sent to the Gemini
    proxy, which answers from Gemini Google Search grounding (search_tool.search_with_fallback). The
    Brave Search API is never contacted and no Brave key exists; no key of any kind is placed in a
    task container;
  * the task image's other search skills are deleted from every task container
    (REMOVED_SEARCH_SKILLS: local-web-search and web-search, both DuckDuckGo, and google-search,
    Google Custom Search), so web_search is the only search tool the agent is offered. Its ordinary
    web access (the shell, web_fetch, the browser skill some tasks install) is unchanged.
Each substitution and the removal are recorded on the result (`gemini_substitutions`,
`gemini_usage`, `removed_search_skills`) and make it not leaderboard-comparable.

Headline `accuracy` = the equal-weight mean of per-task overall_score over the selected tasks
without a harness error (a task without a score among them counts as 0), i.e. run_batch's global_avg
with the harness errors left out. A harness error is a task whose provisioned inputs (workspace
directory, ground-truth files its grade() reads) are missing from the host checkout, whose container
web-search setup failed, whose task container or agent backend failed before the agent ran, that was
never run, whose grading could not run (a docker copy failure, the task container gone, or the 120 s
grade cap hit, with or without a judge call in flight) or returned no numeric score, whose grading
recorded a judge failure, during which the model endpoint (behind the proxy's relay; unless a later
relay call of the same task succeeded) or a Gemini service it depends on failed (a judge call of its
grade(), a helper-model call of its agent failing on the service side, its agent's web_search: no
grounding model answered within the proxy's 57 s deadline inside OpenClaw's web_search timeout
(60 s here, raised from OpenClaw's 30 s default: AGENT_WEB_SEARCH_TIMEOUT_S), the
answer did not reach the agent in time, or OpenClaw aborted the call at that timeout), or
during which the task network lost DNS or web egress (the orchestrator's network monitor; a failed
name lookup in the task's logs counts when the monitor saw the resolver fail, at least half of the
names, in the task's window; a TCP timeout or one name failing does not confirm it). The proxy's
events are attributed to the task whose run window holds their time and whose container made the
request. Before a task's grading starts, the orchestrator stops every process the agent left in its
task container (the processes that ran before the agent, such as the mock services some grade()
functions query, are kept), so nothing the agent started runs while grade() runs; a judge failure
counts only for a request that arrived during a grade() exec run that way. When that stop fails for
a reason the agent cannot cause (the docker exec could not run, or it timed out or docker failed
while the container ran nothing beyond the processes from before the agent and the OpenClaw gateway,
its python3 still working), a Gemini service failure of a judge call during the grade() exec and a
cap hit of it are grading_infra harness errors; when the agent can have caused it (processes left,
python3 missing or broken, more processes running than that, or docker top unable to list them),
they are scored as upstream scores them and counted.
Either way an unverified grading environment makes the status `completed_with_errors`. Such tasks
are counted by kind (`n_harness_errors`, `harness_error_kinds`), named in
leaderboard_comparable_reason and make the status `completed_with_errors`, never scored as the
model's outcome. The agent's own bad requests (the Gemini API refusing a helper-model request for its
content) and a task's own grade() reporting the model's missing or bad output (an "error" next to its
numeric scores) are the model's outcome and stay in the headline. This is NOT the leaderboard's
Overall Score, which follows a weighted multimodal/pure-text breakdown (approximated separately by
`weighted_overall`); the MM/pure-text means, avg time, and avg cost are also reported, with
total_questions / effective_n / correct_answers (the selected tasks, the headline's denominator and
the per-task scores summed over it), the per-turn max_tokens OpenClaw sent (max_tokens_sent) and
the replies cut off at it, the output tokens the model endpoint reported (null, never 0, when it
reported none), and each task's scores, usage and transcript kept in the results
(artifacts_dir).
`leaderboard_comparable` is always False: the Gemini substitutions above, the equal-weight headline,
and a single trial.

HARD-ERRORS (infra_required, never skips) if Docker, the orchestrator image (or one that predates the
harness-error accounting), the provisioned host checkout (with prepared workspace), the loaded task
image, a Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY; also live-validated), or a reachable model
endpoint are missing, if GEMINI_OPENAI_BASE_URL is neither the Gemini API nor a loopback address, if
the cloud metadata server answers from the task network (the model runs shell commands in the task
containers; container_guard.require_metadata_blocked names the host firewall rule), if a container
on the task network cannot resolve the host names the tasks need (TASK_HOSTS, checked by the same
probe), if an explicit proxy port is one of OpenClaw's own ports under the host network, if
GBENCH_WILDCLAWBENCH_TASK_HOST is not a gateway of the task network, if no task produced a score
outside harness errors, if the web-search setup does not apply to the task image, or if the network
monitor's first round finds no DNS or no web egress on the task network (the orchestrator's
preflight). See docs/evals/wildclawbench.md.

Sampling: the model samples at gbench's rule for every suite (base.resolve_temperature):
temperature 1.0 with `--thinking`, 0.0 without; GBENCH_WILDCLAWBENCH_TEMPERATURE, else
--temperature, replaces it in either mode. The orchestrator's model relay forwards the agent's calls to an
EndpointRelay in this process (on 127.0.0.1, which the host-network orchestrator reaches), which
sets `temperature` on every chat-completions request, whatever OpenClaw sent, and adds the model
endpoint's key. The same relay sets the chat template's thinking switch
(`chat_template_kwargs.enable_thinking`: true with --thinking, false without, merged into any
chat_template_kwargs the request carries), which OpenClaw never sends for gbench's provider
(relay_body_fields). The result records `temperature`, `temperature_source`, `temperature_applied`,
`thinking_applied` (true only when every chat-completions request the relay counted carried the
run's switch: `model_temperature_relay.thinking_switch`; else false with
`thinking_not_applied_reason`, also named in leaderboard_comparable_reason) and the relay
(`model_temperature_relay`), and names the temperature in leaderboard_comparable_reason. Before
2026-10-11 the switch was not sent, so a --thinking run reached the model with thinking off (the
2026-10-11 smoke run's relay shape check: enable_thinking absent on 2 of 2 requests); such results
are not comparable. The judge requests keep the temperature each task's grade() sends (none when it sends
none). gbench passes the run's Gemini cascade into the orchestrator (base.gemini_cascade_env()), and
the result records which Gemini models answered (gemini_models_answered, and per task in
raw_summary.per_task[].gemini_models). Network and concurrency: the task containers run on
docker's default bridge network (GBENCH_WILDCLAWBENCH_TASK_NETWORK, the upstream default), each with
its own network namespace, so tasks run in parallel (GBENCH_WILDCLAWBENCH_PARALLEL, default
--concurrency); with the host network the OpenClaw gateway's fixed port forces serial execution
(parallel=1), and the host firewall rule that blocks the metadata server for docker networks does
not cover it.
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from . import container_guard
from .gemini_budget import GEMINI_THINKING_ALLOWANCE
from .limit_plan import eval_limit_plan
from .swebench_common import EndpointRelay, infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/wildclawbench.md"

_IMAGE_DEFAULT = "gbench-wildclawbench:316334cc-v9"
_TASK_IMAGE_DEFAULT = "wildclawbench-ubuntu:v1.3"
_FULL_TASK_COUNT = 60
#: Default port of the orchestrator's Gemini proxy: distinct from gaia2's 18790, and outside the
#: ports the OpenClaw gateway in every host-network task container uses (_OPENCLAW_PORTS). It is
#: in the range 18794-18799 OpenClaw reserves for future services, none of which 2026.3.11 binds.
_JUDGE_PORT_DEFAULT = 18795
#: OpenClaw 2026.3.11's ports in a task container: the gateway (18789, the image's gateway.port),
#: its bridge default (18790), browser control server (gateway + 2 = 18791), control port + 1
#: (18792), canvas host default (18793) and the browser CDP range (18800-18899).
_OPENCLAW_PORTS = frozenset({*range(18789, 18794), *range(18800, 18900)})
#: How the OpenClaw ports are written in messages and docs.
_OPENCLAW_PORTS_TEXT = "18789-18793, 18800-18899"
#: The default docker bridge gateway: the address task containers on a network other than `host`
#: reach the orchestrator's proxy at, unless GBENCH_WILDCLAWBENCH_TASK_HOST names another gateway.
_BRIDGE_GATEWAY_DEFAULT = "172.17.0.1"
#: The image label of an orchestrator whose launcher and proxy report harness errors (task time
#: windows and container addresses, the proxy's harness events with the requesting container, the
#: time the request arrived and whether the Gemini service failed it, the grader's judge calls told
#: apart from the agent's, missing provisioned inputs, the processes the agent left stopped before
#: grading and why a stop failed, every grade() cap hit, the network monitor of the task network
#: with DNS evidence for log-only lookups, harness_errors per task, the answering Gemini model per
#: task) and whose proxy forwards the judge requests' own temperature, sends a request that turns
#: thinking off at THINKING_OFF_LEVEL (counted in gemini_usage `thinking_off`) and tells a model
#: that is not served from a content refusal, and whose launcher stops on a malformed
#: GBENCH_SHARD and reports a shard with none of the selected tasks as an empty shard (nothing
#: run), and whose proxy gives an agent web search a deadline inside OpenClaw's web_search
#: timeout and records an answer that did not reach the agent in time as a harness event, whose
#: launcher counts the web_search calls OpenClaw aborted (web_search_aborted) as harness errors,
#: classifies a timed-out stop of the leftover processes by what still runs, records the
#: max_tokens OpenClaw sent, the replies cut off at it and the token usage the model relay saw,
#: and hands the workdir to the runner's user at exit, and whose proxy adds the Gemini thinking
#: allowance to each chat request's max_tokens, passes the grounded path's real finish reason
#: through and counts the replies still cut at the raised cap (gemini_usage `thinking_allowance`,
#: `max_tokens_cut`, `max_tokens_empty`, `outage_after_max_tokens`), and (version 9, 2026-10-13)
#: whose launcher sets the agent's web_search timeout to AGENT_WEB_SEARCH_TIMEOUT_S (60 s) and whose
#: proxy's search deadline is WEB_SEARCH_DEADLINE_S (57 s); an image without it at this version
#: predates that (an image at version 8 runs the agent's web_search at OpenClaw's 30 s default with
#: a 27 s proxy deadline) and would fold harness failures into the score (a web search OpenClaw aborted
#: among them), let a process the agent left running make its task a harness error, count the
#: agent's own bad requests as harness errors, let the Gemini judge think at its default where the
#: task's judge turns thinking off, run the full set for a malformed shard, or send the tasks'
#: judge caps unchanged so the Gemini judge's thinking eats the verdict's room.
HARNESS_ERRORS_LABEL = "org.gbench.wildclawbench-harness-errors"
HARNESS_ERRORS_VERSION = "9"
#: OpenClaw's web_search timeout in a gbench run, OpenClaw's own default it replaces, and the
#: orchestrator proxy's search deadline: the values the orchestrator image at HARNESS_ERRORS_VERSION
#: applies (wildclawbench_cascade_judge.AGENT_WEB_SEARCH_TIMEOUT_S / UPSTREAM_WEB_SEARCH_TIMEOUT_S /
#: WEB_SEARCH_DEADLINE_S, the one source the launcher and the proxy share; copied here because the
#: runner cannot import the image's files, tests/test_wildclawbench_search_timeout.py keeps them
#: equal). Recorded on every result (gemini_substitutions, leaderboard_comparable_reason).
AGENT_WEB_SEARCH_TIMEOUT_S = 60
UPSTREAM_WEB_SEARCH_TIMEOUT_S = 30
WEB_SEARCH_DEADLINE_S = AGENT_WEB_SEARCH_TIMEOUT_S - 3.0
#: Why the agent's web_search timeout is raised, with the evidence.
WEB_SEARCH_TIMEOUT_REASON = (
    f"OpenClaw's {UPSTREAM_WEB_SEARCH_TIMEOUT_S} s web_search timeout fits Brave, which answers in "
    f"about a second; Gemini Google Search grounding often takes 27-45 s. 2026-10-13 A/B (70 "
    f"recorded agent queries x 3 repeats, search-engine mode): answered in time 65.2% at width 2 "
    f"with a 27 s deadline (61.9% at width 3, 60.5% at width 4), 94.8% (Wilson 95% 90.9-97.1%) at "
    f"width 4 with a 57 s deadline (answered: median 20.2 s, p90 38.4 s); width did not help, the "
    f"deadline did, so the initial width stays 2; width 2 with a 57 s deadline (the shipped "
    f"setting) answered 94.3% (Wilson 95% 90.3-96.7%) in time with no generated text (2026-10-13 "
    f"A/B), and 21 of 21 agent searches in the 2026-10-14 wildclawbench run")
#: Host names the task containers need, resolved from the task network before the run (the
#: metadata guard's probe, `resolve=`) and by the orchestrator's network monitor during it: the
#: package indexes the tasks' warmups install from (pip; npm for agent-browser; apt with the task
#: image's sources: archive.ubuntu.com, security.ubuntu.com, deb.nodesource.com) and sites the tasks
#: name (github.com, arxiv.org, zh.wikipedia.org). Passed to the orchestrator as WILDCLAW_DNS_HOSTS.
TASK_HOSTS = ("pypi.org", "files.pythonhosted.org", "registry.npmjs.org", "archive.ubuntu.com",
              "security.ubuntu.com", "deb.nodesource.com", "github.com", "arxiv.org",
              "zh.wikipedia.org")
#: Hosts whose port 443 the network monitor connects to from the task network (web egress),
#: passed as WILDCLAW_EGRESS_HOSTS.
EGRESS_HOSTS = ("github.com", "pypi.org", "registry.npmjs.org")
#: Knobs that no longer change anything (setting one logs a warning).
_REMOVED_KNOBS = ("GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT",)
#: The proxy path of the model relay (the task containers' my_api.json baseUrl is <origin><path>).
_MODEL_RELAY_PATH = "/model/v1"
#: gbench's Gemini grounding search, mounted read-only into the orchestrator for the Gemini proxy.
_SEARCH_TOOL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runners", "eval_suites",
                 "search_tool.py"))
_SEARCH_TOOL_MOUNT = "/app/gbench/search_tool.py"


def _image() -> str:
    return os.environ.get("GBENCH_WILDCLAWBENCH_IMAGE", _IMAGE_DEFAULT)


def _task_image() -> str:
    return os.environ.get("GBENCH_WILDCLAWBENCH_TASK_IMAGE", _TASK_IMAGE_DEFAULT)


def _harness_dir() -> str:
    return prereqs_path("WildClawBench",
                        os.environ.get("GBENCH_WILDCLAWBENCH_HOST_DIR", "").strip()) or ""


def _docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0


def _image_accounts_harness_errors(image: str) -> bool:
    """True when the orchestrator image carries HARNESS_ERRORS_LABEL=HARNESS_ERRORS_VERSION."""
    r = subprocess.run(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "' + HARNESS_ERRORS_LABEL + '"}}', image],
                       capture_output=True, text=True)
    return r.returncode == 0 and (r.stdout or "").strip() == HARNESS_ERRORS_VERSION


# Live Gemini key ping lives in base.py (single source shared by every judge suite); the alias
# keeps the module-global name that this suite's key gate and the tests reference.
from .base import (gemini_key_live_valid as _gemini_key_valid, gemini_api_keys, redact_text,
                   redact_file, suite_env, free_port, gemini_openai_base_url, require_gemini_openai_base_url,
                   gemini_cascade_env, judge_cascade, search_cascade, MODELS_NOT_SERVED_PREFIX,
                   SHARD_REASON, add_comparability_reason, current_shard, endpoint_api_key,
                   get_run_knob, record_shard, resolve_temperature)


#: The paths on which the temperature relay sets the temperature (EndpointRelay.PATCHED_SUFFIXES):
#: OpenClaw's openai-completions provider calls chat completions.
THINKING_RELAY_PATHS = ("/v1/chat/completions", "/v1/completions")


def thinking_temperature(enable_thinking: bool,
                         kwargs: Optional[Dict[str, Any]] = None) -> Tuple[float, str]:
    """(temperature, source) the relay sets on every model request, in both modes: 1.0 with
    --thinking (source "thinking"), 0.0 without (source "default"), or
    GBENCH_WILDCLAWBENCH_TEMPERATURE, else --temperature (base.resolve_temperature; the runner's
    `temperature` kwarg is the --temperature knob)."""
    value, source = resolve_temperature("wildclawbench", None, thinking=bool(enable_thinking))
    passed = (kwargs or {}).get("temperature")
    if source in ("thinking", "default") and passed is not None and str(passed).strip() != "":
        return float(passed), "cli:--temperature"
    return value, source


def relay_body_fields(enable_thinking: bool, temperature: float) -> Dict[str, Any]:
    """The fields the model relay sets on every chat-completions request of the agent: the run's
    temperature and the chat template's thinking switch (chat_template_kwargs.enable_thinking,
    true with --thinking, false without; merged into any chat_template_kwargs the request
    carries). OpenClaw 2026.3.11 sends no switch for gbench's provider: its models config
    declares no `reasoning`, so OpenClaw reads the model as non-reasoning
    (`configuredModel?.reasoning ?? false`) and its openai-completions client (pi-ai 0.57.1)
    sends neither `enable_thinking` nor `reasoning_effort`. 2026-10-11 smoke run: a --thinking
    run's requests reached the model with no switch (relay shape check: enable_thinking absent
    on 2 of 2 requests), i.e. with thinking off."""
    return {"temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}


class ThinkingSwitchRelay(EndpointRelay):
    """The EndpointRelay the agent's model requests go through, counting the chat-completions
    requests it sent on (after body_fields) and those whose chat_template_kwargs.enable_thinking
    is the run's thinking mode (`thinking_switch` in record()), so thinking_applied is read from
    the requests, not assumed."""

    def __init__(self, *args: Any, enable_thinking: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._thinking = bool(enable_thinking)
        self.chat_requests = 0
        self.chat_requests_with_switch = 0

    def patch_body(self, path: str, body: Optional[bytes], content_type: str) -> Optional[bytes]:
        out = super().patch_body(path, body, content_type)
        if path.partition("?")[0].endswith("/chat/completions"):
            data = self._json_body(out)
            kwargs = data.get("chat_template_kwargs") if data is not None else None
            carried = (isinstance(kwargs, dict) and "enable_thinking" in kwargs
                       and kwargs["enable_thinking"] is self._thinking)
            with self._lock:
                self.chat_requests += 1
                if carried:
                    self.chat_requests_with_switch += 1
        return out

    def record(self) -> Dict[str, Any]:
        rec = super().record()
        with self._lock:
            rec["thinking_switch"] = {
                "field": "chat_template_kwargs.enable_thinking",
                "value": self._thinking,
                "counted_on": ["/v1/chat/completions"],
                "chat_requests": self.chat_requests,
                "chat_requests_with_switch": self.chat_requests_with_switch}
        return rec


def thinking_record(enable_thinking: bool,
                    relay_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """`thinking_applied` from the relay's count (ThinkingSwitchRelay `thinking_switch`): true
    only when the relay saw at least one chat-completions request and every one carried
    chat_template_kwargs.enable_thinking equal to the run's mode (true with --thinking, false
    without); otherwise false with `thinking_not_applied_reason`."""
    switch = (relay_record or {}).get("thinking_switch") or {}
    n = int(switch.get("chat_requests") or 0)
    carried = int(switch.get("chat_requests_with_switch") or 0)
    want = str(bool(enable_thinking)).lower()
    if n <= 0:
        return {"thinking_applied": False,
                "thinking_not_applied_reason": "the relay counted no chat-completions request"}
    if carried != n:
        return {"thinking_applied": False,
                "thinking_not_applied_reason": (
                    f"{n - carried} of {n} chat-completions request(s) went to the model without "
                    f"chat_template_kwargs.enable_thinking={want}")}
    return {"thinking_applied": True}


def busy_openclaw_ports(ports=None, host: str = "127.0.0.1") -> List[int]:
    """The ports of `ports` (default: every OpenClaw port, _OPENCLAW_PORTS) that cannot be bound
    on `host`: a task container on the host network runs the OpenClaw gateway on them, so a
    process (or a concurrent run) already holding one makes the tasks fail mid-run."""
    import socket
    busy = []
    for port in sorted(ports if ports is not None else _OPENCLAW_PORTS):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, int(port)))
            except OSError:
                busy.append(int(port))
    return busy


def check_wildclawbench_prerequisites() -> Tuple[bool, str]:
    """Docker + orchestrator image + provisioned host checkout + loaded task image + Gemini key."""
    image = _image()
    build = (f"Build the orchestrator LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/wildclawbench.Dockerfile gbench/docker\n"
             f"See " + DOCS_URL + " for the full provisioning recipe.")
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if not _docker_image_exists(image):
        return False, f"orchestrator image {image!r} not found. " + build

    harness = _harness_dir()
    if not harness:
        return False, ("GBENCH_WILDCLAWBENCH_HOST_DIR is unset. Provision a WildClawBench checkout "
                       "(clone the harness, `hf download` its workspace, run `bash script/prepare.sh`) "
                       "and point GBENCH_WILDCLAWBENCH_HOST_DIR at it. See " + DOCS_URL)
    if not os.path.isfile(os.path.join(harness, "eval", "run_batch.py")):
        return False, (f"GBENCH_WILDCLAWBENCH_HOST_DIR={harness!r} is not a WildClawBench checkout "
                       f"(eval/run_batch.py missing). See " + DOCS_URL)
    if not os.path.isdir(os.path.join(harness, "workspace")):
        return False, (f"the checkout at {harness!r} has no workspace/ (run the HF workspace "
                       f"download + `bash script/prepare.sh`). See " + DOCS_URL)

    if not _docker_image_exists(_task_image()):
        return False, (f"the task image {_task_image()!r} is not loaded on the host daemon. "
                       f"Download it from HF and `docker load -i wildclawbench-ubuntu_v1.3.tar`. "
                       f"See " + DOCS_URL)
    if not os.path.isfile(_SEARCH_TOOL_PATH):
        return False, (f"gbench's search_tool.py is missing at {_SEARCH_TOOL_PATH} (the Gemini proxy "
                       f"serves the OpenClaw web_search with it). Reinstall gbench. See " + DOCS_URL)
    keys = gemini_api_keys()
    if not keys:
        return False, ("GEMINI_API_KEYS / GEMINI_API_KEY is unset - required for the WildClawBench "
                       "judge cascade (43 of 60 tasks grade with an LLM judge) and for the OpenClaw "
                       "agent's web_search (Gemini Google Search grounding). See " + DOCS_URL)
    # The orchestrator's proxy sends the Gemini keys to GEMINI_OPENAI_BASE_URL (forwarded below):
    # only the Gemini API or a loopback address may receive them, whether or not the key is pinged.
    gemini_base, why_base = gemini_openai_base_url()
    if gemini_base is None:
        return False, why_base + ". See " + DOCS_URL

    # Live-validate the key (a dummy/expired key otherwise only surfaces mid-run as judge/search
    # failures). Only a definitive auth rejection hard-errors (the endpoint was checked above, so a
    # rejection here is the key's); network flakiness is inconclusive and passes. Set
    # GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1 for air-gapped / offline provisioning.
    if not os.environ.get("GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION"):
        ok_g, why_g = _gemini_key_valid(",".join(keys))
        if not ok_g and str(why_g).startswith(MODELS_NOT_SERVED_PREFIX):
            return False, (f"{redact_text(why_g)}; every model of the Gemini cascade must be one "
                           f"the API serves, for the judge cascade and the web search. See "
                           + DOCS_URL)
        if not ok_g:
            return False, (f"the Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) was rejected by the "
                           f"Gemini API ({redact_text(why_g)}); the judge cascade and the web search "
                           f"cannot run. Check the key. See " + DOCS_URL)
    if not _image_accounts_harness_errors(image):
        return False, (f"orchestrator image {image!r} predates the harness-error accounting (label "
                       f"{HARNESS_ERRORS_LABEL}={HARNESS_ERRORS_VERSION}): it would score harness "
                       f"failures as the model's, or the agent's own bad requests as harness "
                       f"errors. Rebuild it. " + build)
    return True, ""


def _model_relay_target(base_url: str) -> str:
    """The model URL the orchestrator's proxy relays the agent's model calls to, ending in /v1:
    GBENCH_WILDCLAWBENCH_TASK_ENDPOINT when set, else the gbench model endpoint as given. The
    proxy runs in the orchestrator on the host network whatever the task network is, so a model
    served on this host at 127.0.0.1 is reached there as it is; nothing is rewritten."""
    url = (os.environ.get("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT") or base_url or "").strip()
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def _network_gateways(network: str) -> List[str]:
    """The gateway addresses of a docker network (`docker network inspect`), [] when unknown."""
    try:
        r = subprocess.run(["docker", "network", "inspect", network, "--format",
                            "{{json .IPAM.Config}}"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []
    try:
        config = json.loads((r.stdout or "").strip() or "null") or []
    except ValueError:
        return []
    return [str(c.get("Gateway")) for c in config if isinstance(c, dict) and c.get("Gateway")]


def _same_address(a: str, b: str) -> bool:
    import ipaddress
    try:
        return ipaddress.ip_address(a) == ipaddress.ip_address(b)
    except ValueError:
        return a == b


def _proxy_host(task_network: str) -> str:
    """The address the orchestrator's Gemini proxy listens on, and the task containers reach it at.

    With the host network: 127.0.0.1. Otherwise GBENCH_WILDCLAWBENCH_TASK_HOST when set, which must
    be a gateway address of the task network (the default bridge when the network is empty), as
    `docker network inspect` reports it; else the default bridge gateway 172.17.0.1. The proxy holds
    the Gemini keys and the model key, so it never listens on another interface of the host, and
    the task containers' judge, web search and model calls never go anywhere but to it."""
    if task_network == "host":
        return "127.0.0.1"
    host = os.environ.get("GBENCH_WILDCLAWBENCH_TASK_HOST", "").strip()
    if not host:
        return _BRIDGE_GATEWAY_DEFAULT
    network = task_network or "bridge"
    gateways = _network_gateways(network)
    if not any(_same_address(host, g) for g in gateways):
        raise infra_required(
            "wildclawbench", f"GBENCH_WILDCLAWBENCH_TASK_HOST={host!r} is not a gateway address of "
            f"the task network {network!r} (docker network inspect reports "
            f"{', '.join(gateways) or 'none'}). The orchestrator's Gemini proxy listens on this "
            f"address and holds the keys, so it can only be the gateway of the task network.",
            DOCS_URL)
    return host


def _search_origin(judge_endpoint: str) -> str:
    """scheme://host:port of the judge endpoint: the Gemini proxy's origin as the task containers
    reach it, where their OpenClaw web_search (the Brave request path at its root) and their model
    calls (_MODEL_RELAY_PATH) are sent."""
    from urllib.parse import urlsplit
    u = urlsplit(judge_endpoint or "")
    return f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else ""


#: Host knobs forwarded into the orchestrator when set: the judge cascade's rounds and backoff and
#: the grounding search's limits. The model lists always go in as base.gemini_cascade_env() gives
#: them (GBENCH_JUDGE_MODELS, GBENCH_SEARCH_MODELS), so the proxy and search_tool call exactly the
#: models the result records. GEMINI_OPENAI_BASE_URL is forwarded apart, by name, as
#: base.require_gemini_openai_base_url returns it.
_FORWARDED_KNOBS = ("GBENCH_JUDGE_CASCADE_ROUNDS", "GBENCH_JUDGE_BACKOFF",
                    "GBENCH_SEARCH_MAX_RPS", "GBENCH_SEARCH_MAX_BURST",
                    "GBENCH_SEARCH_CASCADE_ROUNDS", "GBENCH_SEARCH_ATTEMPTS",
                    "GBENCH_SEARCH_BACKOFF", "GBENCH_SEARCH_HEDGE_DELAY_S",
                    "GBENCH_SEARCH_INITIAL_WIDTH", "GBENCH_SEARCH_REDIRECT_TIMEOUT_S")

#: The tasks whose `## Prompt` offers the agent the OpenRouter helper model (OPENROUTER_BASE_URL /
#: OPENROUTER_API_KEY) at the pinned commit; the Gemini cascade serves it.
HELPER_MODEL_TASKS = (
    "01_Productivity_Flow_task_3_bibtex", "01_Productivity_Flow_task_8_real_image_category",
    "02_Code_Intelligence_task_3_jigsaw_puzzle_zh",
    "02_Code_Intelligence_task_4_jigsaw_puzzle_medium_zh",
    "02_Code_Intelligence_task_5_jigsaw_puzzle_hard_zh",
    "02_Code_Intelligence_task_6_benchmark_vlmeval_ocrbench_zh",
    "02_Code_Intelligence_task_7_connect_the_dots_medium_img_zh",
    "02_Code_Intelligence_task_8_link_a_pix_color_zh",
    "02_Code_Intelligence_task_9_link_a_pix_color_easy_zh",
    "02_Code_Intelligence_task_10_acad_homepage_zh",
    "02_Code_Intelligence_task_11_resume_homepage_zh",
    "02_Code_Intelligence_task_12_connect_the_dots_hard_zh",
    "04_Search_Retrieval_task_7_location_search", "04_Search_Retrieval_task_9_artwork_search",
    "05_Creative_Synthesis_task_1_match_report", "05_Creative_Synthesis_task_2_goal_highlights",
    "05_Creative_Synthesis_task_3_product_poster", "05_Creative_Synthesis_task_4_video_notes",
    "05_Creative_Synthesis_task_5_product_launch_video_to_json",
    "05_Creative_Synthesis_task_6_clothing_outfit_to_model_image",
    "05_Creative_Synthesis_task_7_paper_to_poster",
    "05_Creative_Synthesis_task_8_repo_to_homepage",
    "05_Creative_Synthesis_task_9_repo_to_slides",
    "05_Creative_Synthesis_task_10_social_poster_multi_crop",
    "05_Creative_Synthesis_task_11_video_en_to_zh_dub",
)
#: The judged tasks whose judge call turns thinking off (extra_body={"thinking": {"type":
#: "disabled"}}) at the pinned commit.
THINKING_OFF_TASKS = ("06_Safety_Alignment_task_2_leaked_api",
                      "06_Safety_Alignment_task_3_leaked_api_pswd",
                      "06_Safety_Alignment_task_6_prompt_injection")
#: The Gemini thinking level the proxy sends such a request at
#: (wildclawbench_cascade_judge.THINKING_OFF_LEVEL): every cascade model thinks and none can turn
#: it off; gemini-3.8-flash, gemini-3.7-flash and gemini-3.1-pro-preview offer low / medium /
#: high, gemini-3.6-flash and gemini-3.5-flash also minimal, so "low" is the lowest level every
#: model of the default cascade accepts, and the same request goes to each of them.
THINKING_OFF_LEVEL = "low"
#: The task whose grade() also calls a hardcoded audio model through the OpenRouter client, and
#: that model.
AUDIO_JUDGE_TASK = "05_Creative_Synthesis_task_11_video_en_to_zh_dub"
AUDIO_JUDGE_MODEL = "openai/gpt-4o-mini-audio-preview"


def _judge_models() -> List[str]:
    return list(judge_cascade())


def _search_models() -> List[str]:
    return list(search_cascade())


def thinking_off_mapping(usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """How a judge call that turns thinking off is sent to Gemini, and (with the proxy's `usage`
    tally) how many requests were sent that way, by kind."""
    out: Dict[str, Any] = {
        "tasks": list(THINKING_OFF_TASKS),
        "official": ("extra_body={'thinking': {'type': 'disabled'}} on openai/gpt-5.4 through "
                     "OpenRouter"),
        "sent_as": {"reasoning_effort": THINKING_OFF_LEVEL,
                    "grounded_generation_config": {"thinkingConfig": {
                        "thinkingLevel": THINKING_OFF_LEVEL}}},
        "why": ("the nearest Gemini setting: every model of the cascade thinks and none can turn "
                f"it off; {THINKING_OFF_LEVEL!r} is the lowest thinking level every model of the "
                "default cascade accepts (gemini-3.8-flash, gemini-3.7-flash and "
                "gemini-3.1-pro-preview offer low/medium/high; gemini-3.6-flash and "
                "gemini-3.5-flash also minimal), so the same request goes to each model"),
    }
    if usage is not None:
        counts = (usage or {}).get("thinking_off")
        out["requests"] = dict(counts) if isinstance(counts, dict) else {}
    return out


def agent_web_search_timeout(summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The agent's web_search timeout record: the value gbench sets, OpenClaw's default it
    replaces, the proxy's search deadline, why, and (from the launcher's summary, when given) what
    the run's preflight read back from the task container's OpenClaw config."""
    out: Dict[str, Any] = {
        "agent_web_search_timeout_s": AGENT_WEB_SEARCH_TIMEOUT_S,
        "upstream_timeout_s": UPSTREAM_WEB_SEARCH_TIMEOUT_S,
        "proxy_search_deadline_s": WEB_SEARCH_DEADLINE_S,
        "config_key": "tools.web.search.timeoutSeconds",
        "reason": WEB_SEARCH_TIMEOUT_REASON,
    }
    applied = (summary or {}).get("agent_web_search_timeout")
    if isinstance(applied, dict):
        out["applied_timeout_s"] = applied.get("applied_timeout_s")
        out["image_timeout_s"] = applied.get("image_timeout_s")
    return out


def gemini_substitutions(summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """What the Gemini API serves in place of each official external service, as recorded on
    every result; `summary` (the launcher's) adds the web_search timeout the run applied."""
    judge_models = _judge_models()
    return [
        {
            "component": "LLM judge of the 43 judged tasks (grade() calls the OpenAI SDK at "
                         "OPENROUTER_BASE_URL with model JUDGE_MODEL)",
            "official": "openai/gpt-5.4 through OpenRouter",
            "served_by": ("gbench Gemini judge cascade through Gemini's OpenAI-compatible chat "
                          "endpoint; the requested model is ignored, the prompts and every other "
                          "request field forwarded verbatim, temperature included (none when the "
                          "task sends none); OpenRouter routing fields dropped; a judge call that "
                          "turns thinking off (extra_body thinking {type: disabled}) sent with "
                          f"reasoning_effort {THINKING_OFF_LEVEL!r}, Gemini thinking level "
                          f"{THINKING_OFF_LEVEL}, the lowest level every cascade model accepts "
                          "(none can turn thinking off); any other thinking/reasoning setting "
                          "dropped (the pinned tasks send none); the task's max_tokens sent plus "
                          f"{GEMINI_THINKING_ALLOWANCE} for the Gemini model's thinking"),
            "models": judge_models,
            "thinking_off": thinking_off_mapping(),
            "output_limit": gemini_output_limit(),
        },
        {
            "component": f"audio judge of {AUDIO_JUDGE_TASK} (grade() calls AUDIO_MODEL, "
                         "hardcoded, through the same OpenRouter client, with input_audio parts)",
            "official": f"{AUDIO_JUDGE_MODEL} through OpenRouter",
            "served_by": ("the same gbench Gemini judge cascade (requested model ignored, the "
                          "audio parts and every other field forwarded verbatim)"),
            "models": judge_models,
        },
        {
            "component": "web-search verification judge of 04_Search_Retrieval_task_1 (the judge "
                         "call with tools=[{type: openrouter:web_search}])",
            "official": ("OpenRouter's openrouter:web_search server tool (engine auto, max_results "
                         "5, max_total_results 30) on openai/gpt-5.4"),
            "served_by": ("Gemini generateContent with the google_search tool (Google Search "
                          "grounding) over the judge cascade, with the request's temperature and "
                          "max_tokens; the reply returned as an OpenAI chat completion with the "
                          "grounding sources as url_citation annotations"),
            "models": judge_models,
            "differences": ("the search engine, the queries and the number of results are "
                            "Gemini's; engine/max_results/max_total_results are not applied"),
        },
        {
            "component": ("agent-side helper model offered through OPENROUTER_BASE_URL / "
                          f"OPENROUTER_API_KEY by the prompts of {len(HELPER_MODEL_TASKS)} tasks"),
            "official": "any OpenRouter model the agent requests",
            "served_by": "the same gbench Gemini judge cascade (requested model ignored, every "
                         "other field forwarded verbatim)",
            "tasks": list(HELPER_MODEL_TASKS),
            "models": judge_models,
        },
        {
            "component": "OpenClaw agent web_search tool",
            "official": ("Brave Search API (OpenClaw's brave web-search provider as configured in "
                         "wildclawbench-ubuntu:v1.3, BRAVE_API_KEY)"),
            "served_by": ("Gemini Google Search grounding in search-engine mode (generateContent "
                          "with the google_search tool and the prompt search_tool."
                          "SEARCH_ENGINE_PROMPT_PREFIX + the query, sent to the first two cascade "
                          "models at once; search_tool.search_with_fallback with the proxy's "
                          "deadline) behind the Brave web search request path, answered by the "
                          "orchestrator's Gemini proxy; sources only, no generated text"),
            "models": _search_models(),
            "tool_contract": ("unchanged: the web_search tool OpenClaw 2026.3.11 builds for its brave "
                              "provider (name, description, parameters query/count/country/language/"
                              "freshness/date_after/date_before/search_lang/ui_lang, and the result "
                              "{query, provider, count, tookMs, externalContent, results[{title, url, "
                              "description, published, siteName}]})"),
            "differences": ("country, search_lang/language, ui_lang, freshness and date filters are "
                            "not applied (grounding takes none); title is the source's title or "
                            "domain as grounding reports it; url is the source's real URL (Google's "
                            "grounding redirect resolved, else the redirect URL); description is "
                            "empty (no Gemini-generated text reaches the agent, 2026-10-12); "
                            "published is absent; an answer that cites no source counts as a "
                            "failed lookup, and a query no grounding model could serve with a "
                            "source is an HTTP 502 the agent sees as a failed search; OpenClaw's "
                            f"web_search timeout is {AGENT_WEB_SEARCH_TIMEOUT_S} s, not its "
                            f"{UPSTREAM_WEB_SEARCH_TIMEOUT_S} s default "
                            "(agent_web_search_timeout)"),
            **agent_web_search_timeout(summary),
            "removed_search_skills": [dict(s) for s in REMOVED_SEARCH_SKILLS],
        },
    ]


#: The task image's search skills other than web_search, deleted from every task container by the
#: orchestrator (wildclawbench_run.SEARCH_SKILLS_REMOVED holds the same names), so web_search is the
#: only search tool the agent is offered.
REMOVED_SEARCH_SKILLS = (
    {"skill": "local-web-search", "where": "workspace skill /root/skills/local-web-search-skill",
     "service": "DuckDuckGo HTML search (duckduckgo.com/html)"},
    {"skill": "web-search", "where": "workspace skill /root/skills/web-search",
     "service": "DuckDuckGo search API (the duckduckgo-search package)"},
    {"skill": "google-search", "where": "OpenClaw bundled skill",
     "service": "Google Custom Search JSON API"},
)


_SUBSTITUTION_REASONS = [
    "the judge is the gbench Gemini cascade, not openai/gpt-5.4 through OpenRouter",
    f"{AUDIO_JUDGE_TASK}'s audio judge is the Gemini cascade, not {AUDIO_JUDGE_MODEL}",
    "04_Search_Retrieval_task_1's web-verification judge searches with Gemini Google Search "
    "grounding, not OpenRouter's web_search tool",
    f"the OpenRouter helper model the prompts of {len(HELPER_MODEL_TASKS)} tasks offer the agent "
    "is the Gemini cascade",
    "the OpenClaw agent's web_search is answered by Gemini Google Search grounding returning "
    "sources only, no generated text (each result a cited source's title and real URL with an "
    "empty description), not the Brave Search API",
    "the task image's other search skills (local-web-search and web-search on DuckDuckGo, "
    "google-search on Google Custom Search) are removed, so web_search is the agent's only search "
    "tool",
    f"OpenClaw's web_search timeout is raised from its {UPSTREAM_WEB_SEARCH_TIMEOUT_S} s default to "
    f"{AGENT_WEB_SEARCH_TIMEOUT_S} s (tools.web.search.timeoutSeconds; the Gemini search proxy's "
    f"deadline {WEB_SEARCH_DEADLINE_S:.0f} s): Gemini grounding needs longer than Brave (2026-10-13 "
    f"A/B: 94.8% of searches answered in time at width 4 with a 57 s deadline, 65.2% at width 2 "
    f"with 27 s; width 2 with a 57 s deadline (the shipped setting) answered 94.3% (Wilson 95% "
    f"90.3-96.7%) in time with no generated text (2026-10-13 A/B), and 21 of 21 agent searches in "
    f"the 2026-10-14 wildclawbench run)",
    f"the judge calls of {len(THINKING_OFF_TASKS)} tasks turn thinking off, which no Gemini model "
    f"of the cascade can: they are sent at thinking level {THINKING_OFF_LEVEL}, the lowest one "
    "every model of the default cascade accepts",
]

#: The gemini_usage sections the proxy counts the output limit in (by kind: judge,
#: judge_web_search, agent_helper), and how leaderboard_comparable_reason names them.
_OUTPUT_LIMIT_SECTIONS = (
    ("max_tokens_cut", "repl(ies) returned cut at the raised max_tokens"),
    ("max_tokens_empty", "repl(ies) stopped at the raised max_tokens with no text"),
    ("outage_after_max_tokens", "request(s) no cascade model served after such a reply"),
)


def gemini_output_limit(usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """How the proxy sizes the Gemini replies' output limit, and (with the proxy's `usage`
    tally) the requests whose cap it raised and the replies still cut at it, by kind."""
    out: Dict[str, Any] = {
        "thinking_allowance": GEMINI_THINKING_ALLOWANCE,
        "rule": ("a chat request's max_tokens / max_completion_tokens (and the grounded path's "
                 f"maxOutputTokens) is sent plus {GEMINI_THINKING_ALLOWANCE} for the Gemini "
                 "model's thinking, which counts against max_output_tokens: the tasks' caps "
                 "(128 to 16384 tokens) size the judge's reply (a 128-token cap leaves no room "
                 "for thinking), and every cascade model thinks; a request with no cap is sent "
                 "none"),
    }
    if usage is not None:
        for section in ("thinking_allowance",) + tuple(k for k, _ in _OUTPUT_LIMIT_SECTIONS):
            counts = (usage or {}).get(section)
            key = "requests_raised" if section == "thinking_allowance" else section
            out[key] = dict(counts) if isinstance(counts, dict) else {}
    return out


def output_limit_reason(usage: Optional[Dict[str, Any]]) -> Optional[str]:
    """The leaderboard_comparable_reason entry for Gemini replies still cut at the raised cap
    (gemini_usage's output-limit sections), or None when there are none."""
    parts = []
    for section, text in _OUTPUT_LIMIT_SECTIONS:
        counts = (usage or {}).get(section)
        if not isinstance(counts, dict):
            continue
        by_kind = {k: int(v) for k, v in counts.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v}
        if by_kind:
            parts.append(f"{sum(by_kind.values())} {text} ("
                         + ", ".join(f"{v} {k}" for k, v in sorted(by_kind.items())) + ")")
    if not parts:
        return None
    # parts joined with " and ", never "; ": the reason is one entry of the "; "-joined
    # leaderboard_comparable_reason, kept whole (merge_shards states it again over the union)
    return (f"{_OUTPUT_LIMIT_REASON_PREFIX}(the task's max_tokens plus "
            f"{GEMINI_THINKING_ALLOWANCE} for the Gemini model's thinking): "
            + " and ".join(parts))


#: How output_limit_reason's entry starts (merge_shards drops the shards' entries by it).
_OUTPUT_LIMIT_REASON_PREFIX = "Gemini output limit "


def _is_output_limit_reason(reason: str) -> bool:
    """Whether a leaderboard_comparable_reason entry is (a piece of) output_limit_reason's: its
    whole entry, or a part of an older one that joined its parts with "; "."""
    return reason.startswith(_OUTPUT_LIMIT_REASON_PREFIX) or any(
        re.match(rf"^\d+ {re.escape(text)} \(", reason) for _, text in _OUTPUT_LIMIT_SECTIONS)


def merge_gemini_usage(usages: List[Any]) -> Dict[str, Dict[str, int]]:
    """The shards' proxy tallies (gemini_usage: {section: {kind or model: count}}) summed
    section by section, key by key: each shard's proxy counted only its own tasks' calls."""
    merged: Dict[str, Dict[str, int]] = {}
    for usage in usages:
        for section, counts in (usage or {}).items() if isinstance(usage, dict) else ():
            bucket = merged.setdefault(str(section), {})
            for key, n in (counts or {}).items() if isinstance(counts, dict) else ():
                if isinstance(n, (int, float)) and not isinstance(n, bool):
                    bucket[str(key)] = bucket.get(str(key), 0) + int(n)
    return {k: dict(sorted(v.items())) for k, v in sorted(merged.items())}


#: The harness-error kinds of the launcher's summary (wildclawbench_run.HARNESS_ERROR_KINDS), in
#: order, as they are named in leaderboard_comparable_reason.
HARNESS_ERROR_LABELS = (
    ("missing_input", "provisioned input data (workspace or ground truth) missing"),
    ("search_setup", "web-search setup failed, did not run"),
    ("task_start", "task container or agent backend failed before the agent ran"),
    ("not_run", "never run"),
    ("grading_infra", "grading could not run (a docker copy failure, no transcript loader, the "
                      "task container gone, the 120 s grade cap hit, or a Gemini service failure "
                      "or cap hit after a stop of the agent's processes the agent cannot have "
                      "failed)"),
    ("no_score", "grader returned no numeric score"),
    ("judge_fallback", "grading recorded a judge failure or fell back"),
    ("model_endpoint", "model endpoint failure during the task"),
    ("network", "task network lost DNS or web egress during the task"),
    ("judge", "Gemini judge outage"),
    ("judge_web_search", "Gemini grounded judge outage"),
    ("agent_helper", "Gemini helper-model service failure"),
    ("agent_web_search", "Gemini web search failure or timeout (no grounding model answered in "
                         "time, or OpenClaw aborted the call at its "
                         f"{AGENT_WEB_SEARCH_TIMEOUT_S} s web_search timeout)"),
)


def describe_harness_errors(kinds: Dict[str, Any]) -> str:
    """'<n> <label>, ...' for the summary's harness_error_kinds ({kind: tasks}); a task can have
    several kinds."""
    labels = dict(HARNESS_ERROR_LABELS)
    order = [k for k, _ in HARNESS_ERROR_LABELS] + sorted(k for k in kinds if k not in labels)
    return ", ".join(f"{kinds[k]} {labels.get(k, k)}" for k in order if kinds.get(k))


def _isolation_and_network_reasons(summary: Dict[str, Any]) -> List[str]:
    """The leaderboard_comparable_reason entries for the launcher's grade() cap hits, grading
    isolation, grader-key events it did not count, and the network monitor."""
    out: List[str] = []
    timeouts = summary.get("grading_timeouts") or {}
    counted = sum(int(timeouts.get(k) or 0)
                  for k in ("with_judge_call", "without_judge_call", "unrecorded",
                            "not_isolated_infra"))
    if counted:
        out.append(f"{counted} task(s) whose grade() hit the 120 s cap are harness errors, left "
                   f"out of the headline ({int(timeouts.get('with_judge_call') or 0)} with a "
                   f"judge call in flight, {int(timeouts.get('without_judge_call') or 0)} "
                   f"without one, {int(timeouts.get('unrecorded') or 0)} without an exec record, "
                   f"{int(timeouts.get('not_isolated_infra') or 0)} after a stop of the agent's "
                   f"processes that failed for a reason the agent cannot cause)")
    isolation = summary.get("grading_isolation") or {}
    n_open = int(isolation.get("tasks_not_isolated") or 0)
    if n_open:
        causes = isolation.get("stop_failure_causes") or {}
        shown = ", ".join(f"{n} {c}" for c, n in sorted(causes.items())) or "cause not recorded"
        out.append(f"{n_open} task(s) whose leftover processes could not all be stopped before "
                   f"grading ({shown}): an unverified grading environment; after a stop the agent "
                   f"can have failed, graded as upstream grades them with their judge failures "
                   f"and grade() cap hits ({int(timeouts.get('not_isolated') or 0)}) not counted "
                   f"as harness errors; after one it cannot, a Gemini service failure of a judge "
                   f"call or a cap hit is a grading_infra harness error "
                   f"({int(summary.get('grader_events_counted_infra') or 0)} judge failure "
                   f"event(s))")
    n_unisolated = int(summary.get("grader_events_in_unisolated_grading") or 0)
    if n_unisolated:
        out.append(f"{n_unisolated} judge failure event(s) of the grader arrived during a grade() "
                   f"exec whose leftover processes could not all be stopped")
    n_grader = int(summary.get("grader_events_not_counted") or 0)
    if n_grader:
        out.append(f"{n_grader} judge failure event(s) of a request with the grader key that "
                   f"arrived outside every grade() exec run after the leftover processes were "
                   f"stopped (not counted as harness errors)")
    monitor = summary.get("network_monitor") or {}
    gaps = monitor.get("gaps") or []
    n_unmonitored = int(summary.get("n_tasks_network_unmonitored") or 0)
    if gaps or n_unmonitored:
        out.append(f"the network monitor of the task network has {len(gaps)} gap(s) longer than "
                   f"60 s; {n_unmonitored} task(s) ran partly unmonitored")
    unconfirmed = summary.get("name_resolution_errors_unconfirmed") or {}
    if unconfirmed:
        out.append(f"{len(unconfirmed)} task(s) whose logs show a failed name lookup while the "
                   f"network monitor saw no resolver failure (at least half of the names not "
                   f"resolving) in their run window (scored as run)")
    return out


def _models_answered(by_kind: Dict[str, Any]) -> Dict[str, int]:
    """{model: answered requests} summed over the kinds of the summary's gemini_models_answered."""
    out: Dict[str, int] = {}
    for bucket in by_kind.values():
        for model, n in (bucket or {}).items() if isinstance(bucket, dict) else ():
            out[str(model)] = out.get(str(model), 0) + int(n or 0)
    return dict(sorted(out.items()))


#: Per-task files of the score tree kept with the run's results, and the most bytes kept of each
#: (None: all of it; a larger file keeps its first and last halves around a marker line).
ARTIFACT_TASK_FILES = {"score.json": None, "usage.json": None, "chat.jsonl": 4 * 1024 * 1024,
                       "agent.log": 1024 * 1024, "gateway.log": 256 * 1024}
#: Files of the orchestrator's workdir kept with the run's results.
ARTIFACT_RUN_FILES = ("wildclawbench_summary.json", "harness_events.jsonl", "gemini_answers.jsonl",
                      "gemini_usage.json", "grader_calls.jsonl", "task_windows.jsonl",
                      "model_requests.jsonl", "missing_inputs.json", "search_setup_failures.jsonl",
                      "cascade_judge.log")


def artifacts_dir() -> Optional[str]:
    """`<GBENCH_RESULTS_DIR>/wildclawbench_artifacts/<fresh dir>`, or None outside a gbench run."""
    run_dir = (os.environ.get("GBENCH_RESULTS_DIR") or "").strip()
    if not run_dir:
        return None
    base_dir = os.path.join(run_dir, "wildclawbench_artifacts")
    os.makedirs(base_dir, exist_ok=True)
    return tempfile.mkdtemp(prefix="run_", dir=base_dir)


def _copy_capped(src: str, dst: str, cap: Optional[int]) -> None:
    """Copy `src` to `dst`, keeping at most `cap` bytes (its first and last halves)."""
    size = os.path.getsize(src)
    if cap is None or size <= cap:
        shutil.copyfile(src, dst)
        return
    half = cap // 2
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(fin.read(half))
        fout.write(f"\n[gbench: {size - 2 * half} bytes of {size} left out]\n".encode())
        fin.seek(size - half)
        fout.write(fin.read(half))


def keep_artifacts(workdir: str, dest: Optional[str], orchestrator_log: str = "") -> Optional[str]:
    """Copy the run's records (ARTIFACT_RUN_FILES) and each task's score tree files
    (ARTIFACT_TASK_FILES, under tasks/<category>/<task id>/<run>/) from `workdir` to `dest`, with
    the orchestrator's output, every credential value of the environment redacted. Returns
    `dest` (None when there is none)."""
    if not dest:
        return None
    copies = [(os.path.join(workdir, n), os.path.join(dest, n), None) for n in ARTIFACT_RUN_FILES]
    oc_root = os.path.join(workdir, "output", "openclaw")
    for score in sorted(glob.glob(os.path.join(oc_root, "*", "*", "*", "score.json"))):
        run_dir = os.path.dirname(score)
        rel = os.path.relpath(run_dir, oc_root)
        for name, cap in ARTIFACT_TASK_FILES.items():
            copies.append((os.path.join(run_dir, name), os.path.join(dest, "tasks", rel, name),
                           cap))
    for src, dst, cap in copies:
        if not os.path.isfile(src):
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _copy_capped(src, dst, cap)
            redact_file(dst)
        except OSError as e:
            logger.warning("wildclawbench: could not keep %s (%s)", src, e)
    with open(os.path.join(dest, "wildclawbench_orchestrator.log"), "w", encoding="utf-8") as f:
        f.write(redact_text(orchestrator_log or ""))
    return dest


def remove_workdir(workdir: str, image: Optional[str] = None) -> List[str]:
    """Remove the run's workdir; the paths that could not be removed (logged). The orchestrator
    hands the workdir to this user at exit; what it could not (a killed orchestrator), written as
    root by the task containers, is removed with a throwaway container of the orchestrator image
    (`image`, no network), then the rest again."""
    left: List[str] = []

    def _onerror(_func, path, _exc):
        left.append(path)
    shutil.rmtree(workdir, onerror=_onerror)
    if left and image and os.path.isdir(workdir):
        try:
            subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "find",
                            "-v", f"{workdir}:/w", image, "/w", "-mindepth", "1", "-delete"],
                           capture_output=True, timeout=600)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("wildclawbench: the cleanup container could not run (%s)",
                           type(e).__name__)
        left = []
        shutil.rmtree(workdir, onerror=_onerror)
    if left:
        logger.warning("wildclawbench: %d path(s) of the workdir %s could not be removed (e.g. %s)",
                       len(left), workdir, left[0])
    return left


def compute_wildclawbench_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the headline mean and the modality breakdown out of the launcher summary."""
    if not summary or not isinstance(summary.get("global_mean"), (int, float)):
        return None
    return {
        "global_mean": summary["global_mean"],
        "multimodal_mean": summary.get("multimodal_mean"),
        "pure_text_mean": summary.get("pure_text_mean"),
        "weighted_overall": summary.get("weighted_overall"),
        "avg_time_min": summary.get("avg_time_min"),
        "avg_cost_usd": summary.get("avg_cost_usd"),
    }


def wildclaw_plan(limit: Optional[int], single_task: str) -> Dict[str, Any]:
    """Log the run's planned workload (one line) and return its `eval_limit_plan`: --eval-limit N
    is N tasks in total (WILDCLAW_LIMIT: the launcher's one seeded draw over the selected task ids,
    balanced across the categories, never N per category), each run once; GBENCH_WILDCLAWBENCH_TASK
    runs that one task. A full run runs every selected task (the summary's n_tasks)."""
    lim = int(limit) if limit is not None and int(limit) > 0 else None
    if single_task:
        return eval_limit_plan("wildclawbench", lim, 1, attempts=1, unit="task_runs")
    if lim:
        return eval_limit_plan("wildclawbench", lim, min(lim, _FULL_TASK_COUNT), attempts=1,
                               unit="task_runs")
    logger.info("[wildclawbench] full run (no --eval-limit): every selected task x 1 attempt")
    return {"limit": None, "samples": None, "per_setting": None, "attempts_per_sample": 1,
            "task_runs": None}


def plan_from_summary(plan: Dict[str, Any], n_tasks: Any) -> Dict[str, Any]:
    """The run's `eval_limit_plan` with the tasks the launcher actually ran (the summary's n_tasks:
    a category narrowing, a shard or a limit above the selectable set narrows the logged count),
    one run each; the plan as it is when the summary has no count."""
    if not isinstance(n_tasks, int) or isinstance(n_tasks, bool):
        return plan
    out = dict(plan or {})
    if out.get("samples") is not None and out["samples"] != n_tasks:
        out["samples_logged"] = out["samples"]
    out["samples"] = out["task_runs"] = int(n_tasks)
    return out


def run_wildclawbench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical WildClawBench via the pinned InternLM harness (docker-out-of-docker).

    `limit` (--eval-limit N) caps the run to N of the selected tasks: upstream run_batch has no
    --limit flag, so it is forwarded as WILDCLAW_LIMIT and the launcher draws N ids balanced across
    the categories (a copy of sampling.stratified_indices, seed "wildclawbench"; the summary's
    `selection.limit_draw`, which gbench requires: an image without it took the first N ids, all of
    one category) and drives the harness single-task mode over them (the summary denominator is
    those N tasks, so accuracy stays over exactly them).

    `--shard I/N` (GBENCH_SHARD, set by the gbench CLI) selects shard I of N over the deterministic
    sorted task-id list by round-robin (ids[i-1::n], matching sampling.shard_select), composed BEFORE
    the limit. Upstream run_batch has no --shard flag, so GBENCH_SHARD is forwarded into the container
    and the launcher drives single-task mode over exactly that sharded set, so run-set == score-set:
    the summary's n_tasks/accuracy cover only the shard, is_full is False (n_tasks < 60), and
    leaderboard_comparable stays False (a shard is a subset by construction).
    """
    ok, reason = check_wildclawbench_prerequisites()
    if not ok:
        raise infra_required("wildclawbench", reason, DOCS_URL)

    harness = _harness_dir()
    # Fixed default when free (deterministic single run); else an auto free port so two overlapping
    # wildclawbench (or wildclawbench + gaia2) runs do not collide on the host judge-proxy port. An
    # explicit GBENCH_WILDCLAWBENCH_JUDGE_PORT still forces a specific port.
    _env_judge_port = os.environ.get("GBENCH_WILDCLAWBENCH_JUDGE_PORT")
    judge_port = int(_env_judge_port) if _env_judge_port else free_port(_JUDGE_PORT_DEFAULT)
    # Task-container network. Default "bridge", the upstream default: each task container has its
    # own network namespace and reaches the orchestrator's proxy (judge, web search, model relay) at
    # the bridge gateway, and the host firewall rule that blocks the cloud metadata server for
    # docker networks covers it. "host" shares the host's network stack (the proxy at 127.0.0.1),
    # which that rule does not cover.
    task_network = os.environ.get("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "bridge").strip()
    if task_network == "host" and judge_port in _OPENCLAW_PORTS:
        raise infra_required(
            "wildclawbench", f"GBENCH_WILDCLAWBENCH_JUDGE_PORT={judge_port} is one of the ports "
            f"the OpenClaw gateway in each host-network task container uses "
            f"({_OPENCLAW_PORTS_TEXT}); pick another port.", DOCS_URL)
    if task_network == "host":
        # On the host network every task container's OpenClaw binds these ports on the host
        # itself: one already held (another process, a concurrent run) fails the tasks mid-run.
        busy = busy_openclaw_ports()
        if busy:
            raise infra_required(
                "wildclawbench", f"GBENCH_WILDCLAWBENCH_TASK_NETWORK=host needs the OpenClaw ports "
                f"({_OPENCLAW_PORTS_TEXT}) free on this host, and {len(busy)} of them are in use "
                f"(e.g. {', '.join(str(p) for p in busy[:5])}); free them, or use the default "
                "bridge network (unset GBENCH_WILDCLAWBENCH_TASK_NETWORK), where each task "
                "container has its own ports.", DOCS_URL)
    # --shard I/N: validated here (a malformed GBENCH_SHARD raises before anything starts) and
    # forwarded normalized; the launcher selects shard I of the sorted task ids before the limit.
    shard_spec = current_shard()
    shard_label = shard_spec.label if shard_spec and shard_spec.partial else None
    # The model under test runs shell commands in the task containers (OpenClaw's exec tool), with
    # network access: no task container starts on a network where the cloud metadata server
    # answers. The probe uses the task image on the task network, and also resolves the host names
    # the tasks need there (a host rule that drops all traffic to 169.254.169.254 also drops DNS on
    # GCE, and the tasks would fail on the harness's side).
    container_guard.require_metadata_blocked("wildclawbench", _task_image(), task_network,
                                             DOCS_URL, resolve=TASK_HOSTS)
    # The model URL the proxy relays to always ends in /v1 (the relay maps /model/v1/<path> to
    # <model URL>/<path>), an override included.
    task_endpoint = _model_relay_target(base_url)
    # The run's temperature (1.0 with --thinking, 0.0 without, or the operator's value): the
    # proxy relays to an EndpointRelay here that sets it on every model request (started below).
    applied_temperature = thinking_temperature(bool(enable_thinking), kwargs)
    temperature_relay: Optional[ThinkingSwitchRelay] = None
    # The judge, the agent's web_search (the Brave request path) and its model calls (the model
    # relay, _MODEL_RELAY_PATH) all go to the orchestrator's Gemini proxy, at the one address the
    # task containers reach it at; the proxy listens on that address only (_proxy_host).
    for knob in _REMOVED_KNOBS:
        if os.environ.get(knob) is not None:
            logger.warning("wildclawbench: %s no longer changes anything: the task containers "
                           "always reach the orchestrator's Gemini proxy at the address "
                           "GBENCH_WILDCLAWBENCH_TASK_NETWORK / GBENCH_WILDCLAWBENCH_TASK_HOST "
                           "give; see %s", knob, DOCS_URL)
    bind_host = _proxy_host(task_network)
    url_host = f"[{bind_host}]" if ":" in bind_host else bind_host
    judge_endpoint = f"http://{url_host}:{judge_port}/v1"
    search_origin = _search_origin(judge_endpoint)
    categories = os.environ.get("GBENCH_WILDCLAWBENCH_CATEGORIES", "")
    single_task = os.environ.get("GBENCH_WILDCLAWBENCH_TASK", "")
    parallel = int(os.environ.get("GBENCH_WILDCLAWBENCH_PARALLEL", str(max(1, int(concurrency)))))
    # In host-network mode every sibling task container shares the host net ns, so the OpenClaw
    # gateway's FIXED port (18789) would collide across parallel tasks -> only one binds, the rest
    # fail. Serialize. (bridge mode gives each container its own ns, so parallel is safe there.)
    if task_network == "host" and parallel > 1:
        logger.warning("wildclawbench: task-network=host -> forcing parallel=1 (the OpenClaw gateway "
                       "port is fixed and would collide across host-net siblings). The default "
                       "bridge network runs tasks in parallel.")
        parallel = 1

    # docker-out-of-docker identity mounts (skillsbench pitfall): the harness dir (with workspace/)
    # is bind-mounted into sibling task containers, so it must be at an identity path; the workdir
    # holds output + summary and is TMPDIR.
    workdir = tempfile.mkdtemp(prefix="gbench_wildclawbench_")
    os.chmod(workdir, 0o777)
    orchestrator_log = ""
    keep_dir = None
    orch_name = "gbench_wildclaw_orch_" + os.path.basename(workdir)
    run_label = "gbench_wildclaw_run=" + os.path.basename(workdir)

    def _reap():
        # Reap the orchestrator + any sibling task containers this run labelled (they are NOT --rm
        # and survive an orchestrator kill/timeout - see run_batch's per-task finally not running).
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)
        ps = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={run_label}"],
                            capture_output=True, text=True)
        ids = [x for x in (ps.stdout or "").split() if x]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
        if temperature_relay is not None:
            temperature_relay.stop()

    # The model endpoint's key, for the orchestrator's proxy only (the task containers get a
    # placeholder): GBENCH_WILDCLAWBENCH_MODEL_API_KEY, else the key every pillar sends
    # (VLLM_API_KEY, else OPENAI_API_KEY), else none.
    model_key = (os.environ.get("GBENCH_WILDCLAWBENCH_MODEL_API_KEY") or endpoint_api_key()
                 or "dummy")
    try:
        try:
            # The temperature and the chat template's thinking switch on every chat-completions
            # request (relay_body_fields; OpenClaw sends no switch: 2026-10-11 smoke run).
            temperature_relay = ThinkingSwitchRelay(
                task_endpoint, "127.0.0.1",
                key=None if model_key == "dummy" else model_key,
                body_fields=relay_body_fields(bool(enable_thinking), applied_temperature[0]),
                name="wildclawbench-temperature-relay",
                enable_thinking=bool(enable_thinking)).start()
        except (OSError, ValueError) as e:
            raise infra_required(
                "wildclawbench", "the run samples the model at temperature "
                f"{applied_temperature[0]} with chat_template_kwargs.enable_thinking="
                f"{str(bool(enable_thinking)).lower()}, which gbench sets on the agent's model "
                f"requests through a relay on 127.0.0.1, and it could not start ({type(e).__name__}: "
                f"{redact_text(str(e))}).", DOCS_URL) from e
        task_endpoint = temperature_relay.url
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host",
               "-v", "/var/run/docker.sock:/var/run/docker.sock",
               "-v", f"{harness}:{harness}:rw",             # identity mount (DooD workspace source)
               "-v", f"{workdir}:{workdir}:rw",             # identity mount (output + summary)
               "-v", f"{_SEARCH_TOOL_PATH}:{_SEARCH_TOOL_MOUNT}:ro",   # Gemini grounding search
               "-e", f"WILDCLAW_HARNESS_DIR={harness}",
               "-e", f"WILDCLAW_WORKDIR={workdir}",
               "-e", f"WILDCLAW_TASK_IMAGE={_task_image()}",
               "-e", f"WILDCLAW_TASK_LABEL={run_label}",
               "-e", f"WILDCLAW_JUDGE_PORT={judge_port}",
               "-e", f"WILDCLAW_JUDGE_HOST={bind_host}",
               "-e", f"WILDCLAW_JUDGE_ENDPOINT={judge_endpoint}",
               "-e", f"WILDCLAW_SEARCH_ORIGIN={search_origin}",
               "-e", f"WILDCLAW_MODEL_RELAY_URL={search_origin}{_MODEL_RELAY_PATH}",
               # the model /v1 the proxy relays the agent's calls to
               "-e", f"GBENCH_MODEL_BASE_URL={task_endpoint}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               # Keys go in by NAME: Docker copies each value from run_env, so no key is ever on a
               # command line (argv is visible to every process on the host, and to exception text).
               "-e", "GBENCH_MODEL_API_KEY",
               "-e", "GEMINI_API_KEYS",
               "-e", "GEMINI_API_KEY",
               "-e", f"WILDCLAW_TASK_NETWORK={task_network}",
               # what the orchestrator's network monitor checks on the task network
               "-e", f"WILDCLAW_DNS_HOSTS={','.join(TASK_HOSTS)}",
               "-e", f"WILDCLAW_EGRESS_HOSTS={','.join(EGRESS_HOSTS)}",
               "-e", f"WILDCLAW_PARALLEL={parallel}",
               # the orchestrator hands the workdir (root-owned task output included) to this
               # user at exit, so the per-task artifacts can be kept and the workdir removed
               "-e", f"WILDCLAW_HOST_OWNER={os.getuid()}:{os.getgid()}"]
        # The Gemini keys as gbench reads them (GEMINI_API_KEYS wins; gbench/__init__.py's rule: the
        # list in GEMINI_API_KEYS, its first key in GEMINI_API_KEY). The orchestrator hands them,
        # and the model endpoint's key, to its Gemini proxy only.
        keys = gemini_api_keys()
        run_env = {**os.environ,
                   # the model endpoint's key (model_key), for the orchestrator's proxy only
                   "GBENCH_MODEL_API_KEY": model_key,
                   "GEMINI_API_KEYS": ",".join(keys),
                   "GEMINI_API_KEY": keys[0] if keys else ""}
        # The proxy's Gemini endpoint: only the value base.require_gemini_openai_base_url accepts
        # (the Gemini API over https or a loopback address), by name, whatever
        # GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION says. Unset, the proxy uses the Gemini API.
        run_env.pop("GEMINI_OPENAI_BASE_URL", None)
        if (os.environ.get("GEMINI_OPENAI_BASE_URL") or "").strip():
            run_env["GEMINI_OPENAI_BASE_URL"] = require_gemini_openai_base_url(
                "wildclawbench", DOCS_URL)
            cmd += ["-e", "GEMINI_OPENAI_BASE_URL"]
        # The run's Gemini model lists (GBENCH_JUDGE_MODELS, GBENCH_SEARCH_MODELS), always, so the
        # proxy and search_tool never fall back to a default of their own; then the cascade's and
        # the grounding search's other knobs when set.
        for k, v in gemini_cascade_env().items():
            cmd += ["-e", f"{k}={v}"]
        for k in _FORWARDED_KNOBS:
            if os.environ.get(k):
                cmd += ["-e", f"{k}={os.environ[k]}"]
        # Host knob canonical GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S (legacy alias:
        # WILDCLAW_JUDGE_REQUEST_TIMEOUT_S); forward under the name the in-container judge proxy reads.
        _judge_timeout = suite_env("GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S",
                                   "WILDCLAW_JUDGE_REQUEST_TIMEOUT_S")
        if _judge_timeout:
            cmd += ["-e", f"WILDCLAW_JUDGE_REQUEST_TIMEOUT_S={_judge_timeout}"]
        if categories:
            cmd += ["-e", f"WILDCLAW_CATEGORIES={categories}"]
        if single_task:
            cmd += ["-e", f"WILDCLAW_TASK={single_task}"]
        # --eval-limit N: upstream run_batch has no --limit flag, so the launcher caps by driving the
        # harness single-task mode over the first N selected task ids (denominator capped to match).
        if limit is not None and int(limit) > 0:
            cmd += ["-e", f"WILDCLAW_LIMIT={int(limit)}"]
        workload = wildclaw_plan(limit, single_task)
        # --shard I/N: upstream run_batch has no --shard flag either, so forward GBENCH_SHARD and let
        # the launcher select shard I of N over the deterministic sorted id list (ids[i-1::n], matching
        # sampling.shard_select) BEFORE the limit cap. The launcher drives single-task mode over that
        # sharded set, so run-set == score-set and the summary's n_tasks reflects the shard size.
        if shard_label:
            cmd += ["-e", f"GBENCH_SHARD={shard_label}"]
        if enable_thinking:
            cmd += ["-e", f"WILDCLAW_THINKING={os.environ.get('GBENCH_WILDCLAWBENCH_THINKING', 'high')}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_WILDCLAWBENCH_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("wildclawbench: docker run %s (parallel=%d, network=%s)", _image(), parallel, task_network)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                                  env=run_env)
            orchestrator_log = (proc.stderr or "") + (proc.stdout or "")
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "wildclawbench", f"the orchestrator exceeded GBENCH_WILDCLAWBENCH_TIMEOUT_S "
                f"({timeout_s}s) and was killed; sibling task containers were reaped. Raise the "
                f"timeout or reduce the task set.", DOCS_URL) from exc

        summ_path = os.path.join(workdir, "wildclawbench_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "wildclawbench",
                f"the orchestrator produced no summary (rc={proc.returncode}; a harness failure, not "
                f"a 0%). tail: {redact_text((proc.stderr or proc.stdout or '')[-800:])}", DOCS_URL)
        try:
            with open(summ_path, encoding="utf-8") as f:
                summary = json.load(f)
            if not isinstance(summary, dict):
                raise ValueError(f"a JSON {type(summary).__name__}, not an object")
        except (OSError, ValueError) as e:
            # an orchestrator that died while writing it (2026-10-10: a full disk left it empty)
            raise infra_required(
                "wildclawbench",
                f"the orchestrator's summary is not readable JSON ({e}; rc={proc.returncode}; a "
                f"harness failure, not a 0%). tail: "
                f"{redact_text((proc.stderr or proc.stdout or '')[-800:])}", DOCS_URL) from e
    finally:
        _reap()
        # Per-task evidence (scores, usage, transcripts capped) kept with the results; then the
        # workdir removed, root-owned files the task containers wrote included.
        try:
            keep_dir = keep_artifacts(workdir, artifacts_dir(), orchestrator_log)
        except Exception as e:  # a record of the run, never a reason to lose its result
            logger.warning("wildclawbench: could not keep the run's artifacts (%s)", e)
        remove_workdir(workdir, _image())

    if summary.get("empty_shard"):
        return dict(_empty_shard_result(model_name, enable_thinking, shard_label or "?", summary),
                    eval_limit_plan=plan_from_summary(workload, 0))
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    if limit is not None and int(limit) > 0 and not single_task and not selection.get(
            "limit_draw"):
        raise infra_required(
            "wildclawbench", "the orchestrator does not report how it drew the --eval-limit "
            "subset (an image built before the seeded draw, which took the first ids of the "
            "sorted list, all of one category); rebuild the image from "
            "gbench/docker/wildclawbench.Dockerfile.", DOCS_URL)

    n_tasks = summary.get("n_tasks") or 0
    n_scored = summary.get("n_scored") or 0
    n_headline_tasks = summary.get("n_headline_tasks")
    judge_fallback = summary.get("judge_fallback_tasks") or 0
    n_missing = summary.get("n_missing") or 0
    n_grading_errors = summary.get("n_grading_errors") or 0
    n_search_setup_failures = summary.get("n_search_setup_failures") or 0
    n_harness_errors = summary.get("n_harness_errors") or 0
    harness_kinds = dict(summary.get("harness_error_kinds") or {})
    harness_events = dict(summary.get("harness_events") or {})
    n_unattributed = summary.get("n_unattributed_harness_events") or 0
    harness_text = describe_harness_errors(harness_kinds)
    # The headline covers only tasks without a harness error. With none left there is no score:
    # a hard error, never a fabricated 0.0.
    scored = compute_wildclawbench_score(summary)
    if scored is None or n_scored == 0:
        raise infra_required(
            "wildclawbench",
            f"no task produced a score outside harness errors (n_scored={n_scored} of {n_tasks}"
            f"{'; harness errors: ' + harness_text if harness_text else ''}). Check the task "
            "image, that the model endpoint is reachable from this host (the orchestrator's proxy "
            "relays the agent's model calls), the Gemini key, and DNS and egress on the task "
            "network.", DOCS_URL)
    is_full = (not single_task and not categories and n_tasks >= _FULL_TASK_COUNT)

    # Never leaderboard_comparable: every external model/search call is served by the Gemini API in
    # place of the official service (recorded in gemini_substitutions), the headline is the harness
    # equal-weight mean (the leaderboard Overall uses a weighted MM/pure-text breakdown,
    # approximated separately by weighted_overall), and a run here is a single trial.
    leaderboard = False
    reasons = list(_SUBSTITUTION_REASONS) + [
        "headline is the harness equal-weight global mean; the leaderboard Overall follows a "
        "weighted MM/pure-text breakdown (approximated separately by weighted_overall)",
        "single trial (leaderboard audits multiple)"]
    # Harness errors: counted, named here, left out of the headline, never scored as the model's.
    if n_search_setup_failures:
        reasons.append(f"{n_search_setup_failures} task container(s) could not be pointed at the "
                       f"Gemini web search and did not run (harness errors, left out of the "
                       f"headline)")
    if n_harness_errors:
        reasons.append(f"{n_harness_errors} of {n_tasks} task(s) hit a harness error and are left "
                       f"out of the headline ({harness_text})")
    if n_unattributed:
        reasons.append(f"{n_unattributed} harness event(s) outside every task run ("
                       f"{', '.join(f'{v} {k}' for k, v in sorted(harness_events.items()))})")
    n_recovered = summary.get("recovered_model_endpoint_events") or 0
    if n_recovered:
        reasons.append(f"{n_recovered} model endpoint failure(s) a later relay call of the same "
                       f"task got past (counted, not harness errors)")
    reasons.extend(_isolation_and_network_reasons(summary))
    # A task graded while processes the agent left could not all be stopped, and a grader judge
    # failure during such a grade() exec, leave the grading environment unverified: the run is
    # completed_with_errors even where the task is scored as upstream scores it.
    n_not_isolated = int((summary.get("grading_isolation") or {}).get("tasks_not_isolated") or 0)
    n_unisolated_events = int(summary.get("grader_events_in_unisolated_grading") or 0)
    has_harness_errors = bool(n_harness_errors or n_search_setup_failures or n_unattributed
                              or n_not_isolated or n_unisolated_events)
    if shard_label:
        reasons.append(SHARD_REASON.format(label=shard_label, index=shard_spec.index,
                                           count=shard_spec.count))
    if not is_full:
        reasons.append("not the full 60-task set")
    reasons.append(f"temperature {applied_temperature[0]} set by "
                   f"{applied_temperature[1].split(':', 1)[-1]} on every model request "
                   "(the leaderboard runs OpenClaw's own default)")
    limit_reason = output_limit_reason(summary.get("gemini_usage") or {})
    if limit_reason:
        reasons.append(limit_reason)
    cut = replies_cut(summary)
    if enable_thinking and any(cut.values()):
        reasons.append(f"{sum(cut.values())} model repl(ies) ended at OpenClaw's own per-turn "
                       f"max_tokens ({_caps_text(summary.get('max_tokens_sent'))}) on a "
                       f"--thinking run ({cut['in_reasoning']} while reasoning, "
                       f"{cut['in_content']} in the answer text)")

    relay_record = temperature_relay.record() if temperature_relay is not None else None
    thinking_rec = thinking_record(bool(enable_thinking), relay_record)
    if not thinking_rec["thinking_applied"]:
        reasons.append(f"thinking switch not applied: {thinking_rec['thinking_not_applied_reason']}")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "wildclawbench",
        "model_name": model_name,
        "thinking": enable_thinking,
        # chat_template_kwargs.enable_thinking as the relay counted it on the agent's requests
        # (model_temperature_relay.thinking_switch)
        **thinking_rec,
        "status": "completed_with_errors" if has_harness_errors else "success",
        "accuracy": round(scored["global_mean"] * 100.0, 2),   # headline (%), harness errors out
        "global_mean": round(scored["global_mean"], 4),
        "multimodal_mean": (round(scored["multimodal_mean"], 4)
                            if isinstance(scored.get("multimodal_mean"), (int, float)) else None),
        "pure_text_mean": (round(scored["pure_text_mean"], 4)
                           if isinstance(scored.get("pure_text_mean"), (int, float)) else None),
        "weighted_overall": (round(scored["weighted_overall"], 4)
                             if isinstance(scored.get("weighted_overall"), (int, float)) else None),
        "avg_time_min": (round(scored["avg_time_min"], 2)
                         if isinstance(scored.get("avg_time_min"), (int, float)) else None),
        "avg_cost_usd": (round(scored["avg_cost_usd"], 4)
                         if isinstance(scored.get("avg_cost_usd"), (int, float)) else None),
        "n_tasks": n_tasks,
        "n_scored": n_scored,
        "n_headline_tasks": n_headline_tasks,
        # The summary row: the selected tasks, the headline's denominator (tasks without a
        # harness error) and the per-task rewards summed over it (accuracy = correct/effective_n).
        **headline_counts(summary),
        "n_missing": n_missing,
        "n_grading_errors": n_grading_errors,
        "judge_fallback_tasks": judge_fallback,
        "n_search_setup_failures": n_search_setup_failures,
        # Tasks left out of the headline, by kind (see describe_harness_errors), and the proxy's
        # harness events by kind (model endpoint, Gemini judge / grounded judge / web search).
        "n_harness_errors": n_harness_errors,
        "harness_error_kinds": harness_kinds,
        "harness_events": harness_events,
        "n_unattributed_harness_events": n_unattributed,
        # Model endpoint failures a later relay success of the same task got past (counted, not
        # harness errors), and the selected tasks whose provisioned inputs are missing.
        "recovered_model_endpoint_events": summary.get("recovered_model_endpoint_events") or 0,
        "missing_inputs": dict(summary.get("missing_inputs") or {}),
        # Grading isolation (the processes the agent left stopped before grading), the grade()
        # cap hits by kind, and grader-key events not counted (see _isolation_and_network_reasons).
        "grading_isolation": dict(summary.get("grading_isolation") or {}),
        "grading_timeouts": dict(summary.get("grading_timeouts") or {}),
        "grader_events_not_counted": summary.get("grader_events_not_counted") or 0,
        # The task network's DNS and web egress during the run (the orchestrator's monitor), and
        # the tasks whose logs show a failed name lookup the monitor did not confirm (scored).
        "network_monitor": dict(summary.get("network_monitor") or {}),
        "name_resolution_errors_unconfirmed": dict(
            summary.get("name_resolution_errors_unconfirmed") or {}),
        # The task containers' docker network: the metadata-server guard checked it, and that the
        # host names the tasks need (task_hosts) resolve there.
        "task_network": task_network or "bridge",
        "task_hosts": list(TASK_HOSTS),
        # The search skills the orchestrator's preflight found in the task image and deleted (every
        # task container gets the same setup).
        "removed_search_skills": list(summary.get("removed_search_skills") or []),
        "gemini_substitutions": gemini_substitutions(summary),
        "gemini_usage": summary.get("gemini_usage") or {},
        # The judge calls that turn thinking off (THINKING_OFF_TASKS), how the proxy sent them and
        # how many it sent that way, by kind.
        "judge_thinking_off": thinking_off_mapping(summary.get("gemini_usage") or {}),
        # The output limit the proxy sends the Gemini models (the task's max_tokens plus the
        # thinking allowance), the requests whose cap it raised and the replies still cut at it.
        "gemini_output_limit": gemini_output_limit(summary.get("gemini_usage") or {}),
        # Which Gemini models answered: {model: answered requests} over the judge, the grounded
        # judge, the agent's helper model and its web search (by kind in
        # gemini_models_answered_by_kind, per task in raw_summary.per_task[].gemini_models), and
        # the cascade the run passed into the orchestrator.
        "gemini_models_answered": _models_answered(summary.get("gemini_models_answered") or {}),
        "gemini_models_answered_by_kind": dict(summary.get("gemini_models_answered") or {}),
        "gemini_cascade": _judge_models(),
        "gemini_search_cascade": _search_models(),
        "scoring": "per-task programmatic checks + LLM judge (gbench Gemini cascade); harness global mean",
        "sampling": (f"temperature {applied_temperature[0]} ({applied_temperature[1]}) set by "
                     "gbench's relay on every chat-completions request of the OpenClaw agent "
                     "(gbench's rule for every suite: 1.0 with --thinking, 0.0 without), with "
                     "chat_template_kwargs.enable_thinking="
                     f"{str(bool(enable_thinking)).lower()}"),
        "temperature_applied": True,
        "temperature": applied_temperature[0], "temperature_source": applied_temperature[1],
        **({"model_temperature_relay": {**relay_record,
                                        "sets_temperature_on": list(THINKING_RELAY_PATHS),
                                        # body_fields reach every patched path; only the
                                        # chat requests are counted (thinking_switch.counted_on)
                                        "sets_thinking_switch_on": list(THINKING_RELAY_PATHS)}}
           if relay_record is not None else {}),
        # how the task set was selected (an --eval-limit subset: the launcher's seeded draw)
        "selection": selection or None,
        "eval_limit_plan": plan_from_summary(workload, summary.get("n_tasks")),
        "raw_summary": summary,
        # Per-task scores, usage, transcripts (capped) and the run's records, kept in the results.
        "artifacts_dir": keep_dir,
        # The agent's model requests (the proxy's relay): how many, the per-turn max_tokens
        # OpenClaw sent, the replies that ended at it, and the output tokens (None when the
        # endpoint did not report usage for every task with model traffic: OpenClaw's own tally
        # has nothing to count then, never 0).
        "model_requests": summary.get("model_requests"),
        "replies_cut_at_max_tokens": replies_cut(summary),
        "output_tokens_total": summary.get("output_tokens_total"),
        "tasks_without_token_usage": summary.get("tasks_without_token_usage"),
        "token_usage_note": TOKEN_USAGE_NOTE,
        # web_search calls OpenClaw aborted at its web_search timeout (AGENT_WEB_SEARCH_TIMEOUT_S;
        # their tasks: agent_web_search)
        "web_search_aborted_calls": summary.get("web_search_aborted_calls") or 0,
        "metric": ("WildClawBench headline = the equal-weight mean of per-task overall_score "
                   "(0-1) over the selected tasks without a harness error, a task without a score "
                   "among them counting as 0 (run_batch's global_avg with the harness errors left "
                   "out: a task whose container setup or start failed, that never ran, whose "
                   "grading could not run, hit the 120 s grade cap, returned no numeric score or "
                   "fell back on a judge failure, or during which the model endpoint, a Gemini "
                   "call or the task network's DNS or web egress failed; counted in "
                   "n_harness_errors / harness_error_kinds) - NOT the leaderboard's Overall "
                   "Score, which follows a weighted MM/pure-text breakdown (approximated by "
                   "weighted_overall). "
                   "Agent = OpenClaw in the official wildclawbench-ubuntu:v1.3 image, pointed at the "
                   "gbench model endpoint, its web_search answered by Gemini Google Search "
                   "grounding and the image's other search skills removed; grading is the upstream "
                   "per-task grade() (programmatic + LLM judge, "
                   "the judge served by the gbench Gemini cascade). avg_cost_usd is null unless "
                   "OpenClaw priced the model (a self-hosted model has none)."),
        "leaderboard_comparable": leaderboard,
        "leaderboard_comparable_reason": "; ".join(reasons),
        # OpenClaw's own per-turn setting, not gbench's (the agent's requests are OpenClaw's): the
        # distinct values its requests carried, as the model relay saw them (None: none sent).
        "max_tokens_sent": summary.get("max_tokens_sent"),
        "max_output_tokens_note": _max_tokens_note(kwargs.get("max_output_tokens")),
    }
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    if attempts > 1:
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "wildclawbench: run_batch runs each task once (a single trial)"}
    record_shard(result, shard_label)
    return result


#: What output_tokens_total means (per task: raw_summary.per_task[].output_tokens and
#: token_usage_source).
TOKEN_USAGE_NOTE = ("output tokens as the model endpoint reported them through the proxy's relay "
                    "(or OpenClaw's usage.json when it counted any); null when the endpoint did "
                    "not report usage for every request of a task (OpenClaw's openai-completions "
                    "provider gets none from a streamed reply that carries none), never 0")


def headline_counts(summary: Dict[str, Any]) -> Dict[str, Any]:
    """total_questions (selected tasks), effective_n (tasks without a harness error: the
    headline's denominator) and correct_answers (their overall_score summed, a solved-equivalent
    count), so the summary row reconciles with the headline mean."""
    rows = [r for r in summary.get("per_task") or [] if isinstance(r, dict)]
    clean = [r for r in rows if not r.get("harness_errors")]
    total = summary.get("n_tasks") if summary.get("n_tasks") is not None else len(rows)
    effective = (summary.get("n_headline_tasks") if summary.get("n_headline_tasks") is not None
                 else len(clean))
    return {"total_questions": total, "effective_n": effective,
            "correct_answers": round(sum(float(r["overall_score"]) for r in clean
                                         if isinstance(r.get("overall_score"), (int, float))), 2),
            "correct_answers_note": ("solved-equivalent: per-task overall_score (0-1) summed over "
                                     "the tasks without a harness error")}


def replies_cut(summary: Dict[str, Any]) -> Dict[str, int]:
    """{in_reasoning, in_content}: model replies that ended at OpenClaw's per-turn max_tokens
    (finish_reason length) with no answer text yet, and with some."""
    cut = summary.get("replies_cut_at_max_tokens") or {}
    return {"in_reasoning": int(cut.get("in_reasoning") or 0),
            "in_content": int(cut.get("in_content") or 0)}


def _caps_text(caps: Any) -> str:
    vals = [("none sent" if c is None else str(c)) for c in (caps or [])]
    return "max_tokens " + ", ".join(vals) if vals else "max_tokens not recorded"


def _max_tokens_note(knob: Any = None) -> str:
    """The result's max_output_tokens_note: the agent's requests are OpenClaw's, with OpenClaw's
    own per-turn setting for the model (the harness' models config sets no max_tokens), so
    --max-output-tokens does not reach them; max_tokens_sent records what they carried."""
    return ("not set by gbench: the agent's requests are OpenClaw's, with OpenClaw's own "
            "max_tokens setting for the model (gbench's models config sets none); "
            "max_tokens_sent lists the values the requests carried as the proxy's relay saw them "
            "and replies_cut_at_max_tokens the replies that ended there"
            + (f"; --max-output-tokens {knob} not applied" if knob else ""))


def _empty_shard_result(model_name: str, enable_thinking: bool, shard_label: str,
                        summary: Dict[str, Any]) -> Dict[str, Any]:
    """The result of a shard that holds none of the selected tasks: nothing ran on this node, so
    there is no score (accuracy None), never an error and never 0%."""
    reason = (f"empty shard {shard_label}: no items on this node "
              f"({summary.get('full_set_items')} in the full set)")
    result: Dict[str, Any] = {
        "benchmark_type": "eval", "eval_name": "wildclawbench", "model_name": model_name,
        "thinking": enable_thinking, "status": "success", "status_reason": reason,
        "accuracy": None, "accuracy_withheld_reason": reason, "empty_shard": True,
        "full_set_items": summary.get("full_set_items"), "n_tasks": 0, "n_scored": 0,
        "total_questions": 0, "correct_answers": 0, "raw_summary": summary}
    record_shard(result, shard_label)
    return add_comparability_reason(result, reason)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one wildclawbench run (gbench --merge): the harness
    means recomputed over the union of the shards' per-task rows (raw_summary.per_task), as the
    launcher computes them: the equal-weight mean of overall_score over the tasks without a
    harness error (a task without a score counts 0), the multimodal and pure-text means and
    their equal-weight average (weighted_overall); the proxy's gemini_usage tallies summed over
    the shards, with judge_thinking_off, gemini_output_limit and the output-limit reason (replies
    still cut at the raised cap) recomputed from the sum; the shard comparability reasons are
    dropped."""
    union: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for row in (r.get("raw_summary") or {}).get("per_task") or []:
            union.setdefault(str(row.get("task_id")), row)
    rows = [union[k] for k in sorted(union)]
    clean = [row for row in rows if not row.get("harness_errors")]

    def _overall(subset: List[Dict[str, Any]]) -> Optional[float]:
        if not subset:
            return None
        return sum(float(row["overall_score"]) for row in subset
                   if isinstance(row.get("overall_score"), (int, float))) / len(subset)
    global_mean = _overall(clean)
    mm = _overall([row for row in clean if row.get("modality") == "multimodal"])
    pt = _overall([row for row in clean if row.get("modality") != "multimodal"])
    merged = dict(results[0])
    merged.update({
        "accuracy": round(global_mean * 100.0, 2) if global_mean is not None else None,
        "global_mean": round(global_mean, 4) if global_mean is not None else None,
        "multimodal_mean": round(mm, 4) if mm is not None else None,
        "pure_text_mean": round(pt, 4) if pt is not None else None,
        "weighted_overall": (round(0.5 * mm + 0.5 * pt, 4)
                             if mm is not None and pt is not None else None),
        "n_tasks": len(rows), "n_headline_tasks": len(clean),
        "n_scored": sum(1 for row in clean if isinstance(row.get("overall_score"), (int, float))),
        "n_harness_errors": len(rows) - len(clean),
        "raw_summary": {"per_task": rows, "merged_from_shards": len(results)}})
    kinds: Dict[str, int] = {}
    for row in rows:
        for k in row.get("harness_errors") or []:
            kinds[str(k)] = kinds.get(str(k), 0) + 1
    merged["harness_error_kinds"] = dict(sorted(kinds.items()))
    answered: Dict[str, int] = {}
    for r in results:
        for m, n in (r.get("gemini_models_answered") or {}).items():
            answered[m] = answered.get(m, 0) + int(n)
    merged["gemini_models_answered"] = dict(sorted(answered.items()))
    merged.update(headline_counts({"per_task": rows, "n_tasks": len(rows),
                                   "n_headline_tasks": len(clean)}))
    caps: set = set()
    cut = {"in_reasoning": 0, "in_content": 0}
    for r in results:
        caps.update(c if isinstance(c, int) else None for c in r.get("max_tokens_sent") or [])
        for k, v in replies_cut(r).items():
            cut[k] += v
    merged["max_tokens_sent"] = sorted(caps, key=lambda c: (c is None, c or 0))
    merged["replies_cut_at_max_tokens"] = cut
    merged["model_requests"] = sum(int(r.get("model_requests") or 0) for r in results)
    # thinking_applied over the shards that ran: true only when every one applied it
    ran = [r for r in results if not r.get("empty_shard")]
    merged.pop("thinking_not_applied_reason", None)
    merged["thinking_applied"] = bool(ran) and all(r.get("thinking_applied") is True for r in ran)
    # the relay's thinking_switch counts summed over the shards that ran (never shard 1's alone)
    switches = [(r.get("model_temperature_relay") or {}).get("thinking_switch") for r in ran]
    switches = [sw for sw in switches if isinstance(sw, dict)]
    if isinstance(merged.get("model_temperature_relay"), dict) and switches:
        summed = {**switches[0], "summed_over_shards": len(switches)}
        for key in ("chat_requests", "chat_requests_with_switch"):
            summed[key] = sum(int(sw.get(key) or 0) for sw in switches)
        merged["model_temperature_relay"] = {**merged["model_temperature_relay"],
                                             "thinking_switch": summed}
    if not merged["thinking_applied"]:
        merged["thinking_not_applied_reason"] = "; ".join(
            f"shard {r.get('shard') or i + 1}: "
            f"{r.get('thinking_not_applied_reason') or 'not recorded'}"
            for i, r in enumerate(ran) if r.get("thinking_applied") is not True) or (
            "no shard ran a task")
    tokens = [r.get("output_tokens_total") for r in results]
    merged["output_tokens_total"] = (sum(tokens) if all(isinstance(t, int) for t in tokens)
                                     else None)
    merged["tasks_without_token_usage"] = sum(int(r.get("tasks_without_token_usage") or 0)
                                              for r in results)
    merged["web_search_aborted_calls"] = sum(int(r.get("web_search_aborted_calls") or 0)
                                             for r in results)
    # The proxy's tallies, summed over the shards (each shard's proxy counted its own tasks), and
    # the records built from them, recomputed over the sum: never shard 1's counts alone.
    usage = merge_gemini_usage([r.get("gemini_usage") for r in results])
    merged["gemini_usage"] = usage
    merged["judge_thinking_off"] = thinking_off_mapping(usage)
    merged["gemini_output_limit"] = gemini_output_limit(usage)
    merged["artifacts_dir"] = None
    merged["artifacts_dir_by_shard"] = {str(r.get("shard") or i + 1): r.get("artifacts_dir")
                                        for i, r in enumerate(results)}
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if (reason and not reason.startswith(shard_prefix)
                    and not reason.startswith("empty shard ") and reason not in kept
                    and not _is_output_limit_reason(reason)
                    and not (len(rows) >= _FULL_TASK_COUNT
                             and reason == "not the full 60-task set")):
                kept.append(reason)
    # the replies still cut at the raised output limit, counted once over the union
    limit_reason = output_limit_reason(usage)
    if limit_reason:
        kept.append(limit_reason)
    merged["leaderboard_comparable"] = False     # the Gemini substitutions always apply
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    merged["status"] = ("completed_with_errors"
                        if any(r.get("status") == "completed_with_errors" for r in results)
                        else "success")
    for k in ("accuracy_withheld_reason", "status_reason"):
        merged.pop(k, None)
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = "no task outside harness errors in any shard"
    # the shards' summed plan, not the first shard's
    from .limit_plan import merge_plans
    plan = merge_plans(results)
    if plan is not None:
        merged["eval_limit_plan"] = plan
    return merged
