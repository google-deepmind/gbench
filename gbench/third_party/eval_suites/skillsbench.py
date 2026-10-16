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

# Evaluation Suite: skillsbench
# Description: SkillsBench (benchflow-ai) - deterministic, execution-based agentic skills benchmark

"""gbench native built-in runner for skillsbench (Tool Use & Agentic Workflows).

Canonical SkillsBench (benchflow-ai/skillsbench): 87 tasks, each a self-contained per-task Docker
environment scored by a DETERMINISTIC pytest verifier (reward in [0,1] written to
/logs/verifier/reward.txt) - there is NO LLM judge anywhere. gbench DELEGATES the whole agentic
run to the bundled upstream BenchFlow runner inside a LOCAL orchestrator container
(gbench/docker/skillsbench.Dockerfile), run docker-OUT-of-docker: `bench eval run --backend docker`
builds + runs each TASK container as a sibling on the host daemon. The agent (GBENCH_SKILLSBENCH_AGENT,
default deepagents: the pinned BenchFlow 0.6.3 does not ship the leaderboard's opencode agent) runs
inside each task container, pointed at the gbench /v1 endpoint via an injected openai-compatible
provider; the task's verifier produces the reward. The result's `agent` and metric name the agent
that ran.

Headline `accuracy` = mean reward over the tasks in the primary condition (default without-skills,
the model's raw capability). When both conditions run, the with-skills-vs-without lift (upstream's
signature metric) is also reported.

Because scoring is deterministic (no judge), a FULL run with the leaderboard's opencode agent is in
principle leaderboard-comparable - but gbench runs a single trial (upstream averages 3) and may
cover a subset, and the default agent is deepagents, so `leaderboard_comparable` is set True only
for a full, both-condition run with the opencode agent and otherwise False with the reason. The
summary row reports total_questions (the primary condition's tasks), effective_n (those scored) and
correct_answers (their rewards summed, a solved-equivalent count). Each rollout's result, rewards,
timing, verifier output (test-stdout.txt, test-stderr.txt, ctrf.json) and agent trajectory (capped)
are kept under <results dir>/skillsbench_artifacts/ (`artifacts_dir`), and the work directory,
which the task containers fill with root-owned files, is removed (with a throwaway container of the
orchestrator image for what this user cannot remove).

HARD-ERRORS (infra_required, never skips) if Docker, the locally-built image (or one that
predates the current adapter: label org.gbench.skillsbench-harness=7), or a reachable /v1 endpoint
is missing. There is NO GEMINI dependency here (no judge). Heavy external provisioning
(Docker daemon, network egress for ~28 task builds + the agent install, a served model reachable
from task containers) - see docs/evals/skillsbench.md.

Credentials: the agent hands its environment to every command the model runs in the task
container, so no key enters any container or command line. The orchestrator and the agent get the
placeholder MODEL_KEY_PLACEHOLDER as GBENCH_MODEL_API_KEY (passed to docker by name). When the
served endpoint has a key (GBENCH_SKILLSBENCH_API_KEY, else base.endpoint_api_key: VLLM_API_KEY,
else OPENAI_API_KEY), the task containers reach the model through an EndpointRelay in this process
that adds it, listening at GBENCH_SKILLSBENCH_TASK_HOST (default 172.17.0.1). Before the orchestrator starts, swebench_common.require_task_network_blocked checks,
with a probe container of the orchestrator image, that the cloud metadata server (169.254.169.254)
is unreachable from docker's default bridge and from a user-defined bridge network, where
BenchFlow's task containers run, and that the package hosts the task verifiers and the agent
install fetch from at run time resolve there (swebench_common.TASK_INSTALL_HOSTS; infra_required
otherwise).

Harness errors: a task whose verifier could not resolve those package hosts before its tests ran
(swebench_common.verifier_dns_failure over the rollout's verifier/test-stdout.txt) scored nothing,
when something outside its container corroborates the failure (another such task in the run, or a
fresh probe container after the run that cannot resolve one of the hosts either;
swebench_common.corroborated_dns_failures). Uncorroborated, or when the agent's transcript names a
resolver file, the model may have broken name resolution itself and the reward stands
(`dns_failures_kept`). Such a harness error, and a task that produced no reward, is left out of its
condition's mean, counted in `harness_errors`, and makes the run `completed_with_errors`; a run
with no task left to score hard-errors (infra_required). A task with no reward is never scored 0:
its `harness_error_tasks` entry (kind `no_reward`) carries what its rollout's result.json says
(no_reward_cause): BenchFlow's `benchflow_error_category`, the `stage` that failed
(`environment_build` / `environment_start` for a `docker compose build` / `up` failure,
`sandbox_command` for another failed compose command, `environment_setup`, `agent_setup`,
`verifier`, `agent`, or `unknown`: see no_reward_cause), `agent_ran` (whether the agent made any tool call or used any token) and
an `error_excerpt`. On 2026-10-13 lab-unit-harmonization's environment build failed in the Docker
daemon (containerd could not commit a layer of step 4/7 after its `pip install` had finished)
before the agent started; BenchFlow classified it "other" and did not retry it. The image's
skillsbench_benchflow_build_retry.patch adds that failure to the build failures BenchFlow retries
(twice), and the launcher records each retry (`sandbox_build_retries`, {condition: {task: n}};
None from an image that does not record them).

Sampling: the model samples at gbench's rule for every suite (base.resolve_temperature):
temperature 1.0 with `--thinking`, 0.0 without; GBENCH_SKILLSBENCH_TEMPERATURE, else
--temperature, replaces it in either mode. The task containers reach the model through an
EndpointRelay in this process that sets `temperature` on every chat-completions request (THINKING_RELAY_PATHS), whatever
the agent sent; the result records `temperature`, `temperature_source`, `temperature_applied: true`
and the relay (`model_relay`), and names the temperature in leaderboard_comparable_reason (the
leaderboard runs the agent's own default). The same relay sets the chat template's thinking switch
(`chat_template_kwargs.enable_thinking`: true with --thinking, false without), which deepagents'
ChatOpenAI never sends: `thinking_applied: true` (thinking_record: false, with the reason, for
the oracle and codex agents or when no chat-completions reply was relayed),
`model_relay.sets_body_fields`. Before 2026-10-11
it was not sent, so a --thinking run reached the model with thinking off (`thinking_applied:
false`); such results are not comparable.

Transport timeouts (transport_timeouts): BenchFlow's ACP idle watchdog (600 s by default) killed and
re-ran any trial whose single model turn ran longer than that (2026-10-10 smoke run,
debug-trl-grpo), because the deepagents shim emits no ACP event during a turn. It is a BenchFlow
transport default, not SkillsBench's protocol (the task's [agent].timeout_sec bounds each trial),
so gbench passes `--agent-idle-timeout` T, base.request_timeout_s of the served context window,
and gives the shim's ChatOpenAI the same T as its request timeout (the image's
skillsbench_benchflow_transport.patch, which also pins deepagents and langchain-openai). The result
records `model_client_timeout` (deepagents only) and `agent_idle_timeout` (not for the oracle
agent); results from before 2026-10-11 are not
comparable. The result also has `sampling`, `thinking` (the run's --thinking) and
`max_output_tokens_note` (the agent sets its own
per-request limit; --max-output-tokens is not applied). The launcher runs each task once:
`--attempt-count` above 1 is recorded as not applied (`attempts.not_applied`).

Shard (`--shard I/N`, GBENCH_SHARD): gbench validates it (base.current_shard; a malformed value is
an error) and passes the normalized "I/N" to the launcher, which takes shard I of N of the sorted
task list (or of GBENCH_SKILLSBENCH_TASKS) before the limit and itself stops on a malformed value.
`--eval-limit N` (which wins over GBENCH_SKILLSBENCH_TASKS: the list is then ignored with a
warning, explicit_tasks_env) is a seeded draw of N tasks from the shard,
balanced across the tasks' categories (task.md metadata.category; the launcher's copy of
sampling.stratified_indices, seed "skillsbench"), not the first N of the sorted list; the summary
reports it (`selection.limit_draw`) and gbench refuses an image whose launcher does not (it took
the head). The result records the shard identity (base.shard_identity), the tasks run
(`tasks_selected`), the selection (`selection`) and a partial shard's comparability reason; an empty shard reports `accuracy` null with
`accuracy_withheld_reason`. Each (condition, task) is a trace (`sample_traces`: `condition`,
`task`, `reward`, a harness error with `scoring_excluded`, `sample_key`); merge_shards (gbench
--merge) recomputes each condition's mean reward over the union of the shards' traces.
"""

import copy
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from . import swebench_common
from .base import (SHARD_REASON, UNCAPPED_TIMEOUT_TOKENS, add_comparability_reason,
                   current_shard, get_run_knob, redact_file, request_timeout_s,
                   resolve_temperature, resolve_token_budget, shard_identity)
from .base import endpoint_api_key as _endpoint_api_key
from .limit_plan import eval_limit_plan
from .sampling import allocate_limit
from .sandbox import credential_free_env
from .swebench_common import EndpointRelay, infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/skillsbench.md"

_IMAGE_DEFAULT = "gbench-skillsbench:9a1f4dd5-h7"
_FULL_TASK_COUNT = 87
#: What the orchestrator, and through it the agent in each task container, gets as
#: GBENCH_MODEL_API_KEY: a placeholder, never a key. An endpoint key (GBENCH_SKILLSBENCH_API_KEY)
#: stays in this process: the task containers reach the model through gbench's relay, which adds
#: it (see run_skillsbench).
MODEL_KEY_PLACEHOLDER = "dummy"
#: The endpoint paths the key relay forwards: the relay's own (chat completions, completions,
#: models) plus the Responses API, which BenchFlow's codex agent calls.
RELAY_PATHS = tuple(EndpointRelay.ALLOWED_PATHS) + ("/v1/responses",)
#: The image label of an orchestrator whose adapter (gbench/docker/skillsbench_run.py and
#: skillsbench_entrypoint.sh) is the current one: it applies GBENCH_SHARD, draws an --eval-limit
#: subset with a seeded draw balanced across the task categories (reported in its summary's
#: `selection`) and runs the with-skills arm with `--skill-mode with-skill` (BenchFlow 0.6.3 raises
#: when --skills-dir is set without it). An image without the label, or with another value,
#: predates that adapter: a --shard run would silently run the whole set, a limited run would take
#: the first tasks of the sorted list and a with-skills run (the lift) would fail. Since 7 the
#: image retries the containerd layer-commit failure of an environment build
#: (skillsbench_benchflow_build_retry.patch) and its summary records the retries
#: (`sandbox_build_retries`). Bump it with every change to the adapter files (a test pins their
#: fingerprint to this value).
HARNESS_LABEL = "org.gbench.skillsbench-harness"
HARNESS_VERSION = "7"
#: The paths on which the relay sets the temperature and the chat template's thinking switch
#: (EndpointRelay.PATCHED_SUFFIXES): chat completions and completions, what deepagents (langchain
#: ChatOpenAI) calls. The Responses API (/v1/responses, BenchFlow's codex agent) is relayed but
#: not patched.
THINKING_RELAY_PATHS = ("/v1/chat/completions", "/v1/completions")
#: The agent's pinned packages in the image (skillsbench_benchflow_transport.patch; labels
#: org.gbench.deepagents / org.gbench.langchain_openai), recorded with the request timeout.
DEEPAGENTS_VERSION = "0.7.23"
LANGCHAIN_OPENAI_VERSION = "1.7.0"


#: Agents whose model requests carry no thinking switch the relay sets: the model-free oracle
#: sends none, and BenchFlow's codex agent calls the Responses API (/v1/responses), which the
#: relay forwards unchanged (THINKING_RELAY_PATHS).
UNPATCHED_AGENTS = {
    "oracle": "the oracle agent runs each task's reference solution and sends no model request",
    "codex": ("BenchFlow's codex agent calls the Responses API (/v1/responses), on which the "
              "relay sets neither the temperature nor the thinking switch"),
}


def thinking_record(agent: str, relay_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """`thinking_applied` from what went over the wire: true only when the agent's model requests
    go to a path the relay patches (THINKING_RELAY_PATHS; not UNPATCHED_AGENTS) and the relay
    counted at least one completions reply there; otherwise false with
    `thinking_not_applied_reason`. (2026-10-11 review: a literal true was wrong for
    agent=codex and agent=oracle.)"""
    if agent in UNPATCHED_AGENTS:
        return {"thinking_applied": False, "thinking_not_applied_reason": UNPATCHED_AGENTS[agent]}
    replies = int((relay_record or {}).get("completion_replies") or 0)
    if replies <= 0:
        return {"thinking_applied": False,
                "thinking_not_applied_reason": "the relay counted no chat-completions reply"}
    return {"thinking_applied": True}


def agent_timeouts(agent: str, timeouts: Dict[str, Any]) -> Dict[str, Any]:
    """The transport_timeouts records that apply to `agent`: BenchFlow's idle watchdog for every
    agent but the oracle (which the launcher runs without it), and the request timeout only for
    deepagents, whose shim alone reads GBENCH_MODEL_REQUEST_TIMEOUT_S."""
    out: Dict[str, Any] = {}
    if agent != "oracle":
        out["agent_idle_timeout"] = timeouts["agent_idle_timeout"]
    if agent == "deepagents":
        out["model_client_timeout"] = timeouts["model_client_timeout"]
    return out


def transport_timeouts() -> Dict[str, Any]:
    """The request timeout gbench gives the delegated agent and BenchFlow's idle watchdog, as the
    result records them (`model_client_timeout`, `agent_idle_timeout`): T =
    base.request_timeout_s of the served context window (base.UNCAPPED_TIMEOUT_TOKENS when the
    server reports none), as deepagents sends no max_tokens.

    2026-10-10 smoke run: BenchFlow's ACP idle watchdog (`bench eval run --agent-idle-timeout`,
    default 600 s, benchflow/_utils/config.py DEFAULT_AGENT_IDLE_TIMEOUT_SEC) killed
    debug-trl-grpo with "Agent idle for 600s" and re-ran the trial (691 s lost): the deepagents
    shim streams LangGraph updates per step, so one model turn longer than 600 s sends no ACP
    event (the shim's own comment says a long productive turn "false-fires the idle timeout"). It
    is a BenchFlow transport default, not SkillsBench's protocol: SkillsBench's task-creator notes
    call it "benchflow's idle-600s pitfall" to work around (benchflow issue #211), its leaderboard
    export drops idle-failure trials as failed retries (export_leaderboard_submission.py), and
    the CLI's own help says 0 falls back to the task's wall-clock timeout, the task's
    [agent].timeout_sec, which stays the protocol's bound. gbench raises it to T.
    The shim's ChatOpenAI had no request timeout at all (timeout None: checked in the image with
    the pinned deepagents 0.7.23 / langchain-openai 1.7.0 / openai 3.28.0 against a slow local
    server, 2026-10-11), so it hid no cap, but a request the endpoint never answered waited until
    the task's timeout; it now gets T (GBENCH_MODEL_REQUEST_TIMEOUT_S), a bound no reply within
    the window reaches, and its retries stay the OpenAI SDK's default (2)."""
    tokens = int(get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)
    timeout = request_timeout_s(tokens)
    source = f"base.request_timeout_s(served context window={tokens})"
    return {
        "model_client_timeout": {
            "client": "langchain_openai.ChatOpenAI (deepagents ACP shim, BenchFlow 0.6.3)",
            "version": f"langchain-openai {LANGCHAIN_OPENAI_VERSION}, deepagents "
                       f"{DEEPAGENTS_VERSION}",
            "timeout_s": timeout, "retries": 2,
            "source": f"GBENCH_MODEL_REQUEST_TIMEOUT_S = {source} "
                      "(skillsbench_benchflow_transport.patch); retries: the OpenAI SDK default",
            "replaces": "ChatOpenAI default: no request timeout (timeout None)",
            # there was no cap to remove: T adds a bound (above every reply the window allows)
            "hidden_cap_removed": False},
        "agent_idle_timeout": {
            "watchdog": "BenchFlow ACP idle watchdog (bench eval run --agent-idle-timeout)",
            "timeout_s": timeout, "source": source,
            "replaces": "BenchFlow default 600 s (DEFAULT_AGENT_IDLE_TIMEOUT_SEC)",
            "protocol": ("transport default, not SkillsBench protocol; the task's "
                         "[agent].timeout_sec still bounds each trial"),
            "hidden_cap_removed": True},
    }


def thinking_temperature(enable_thinking: bool,
                         kwargs: Optional[Dict[str, Any]] = None) -> Tuple[float, str]:
    """(temperature, source) the relay sets on every model request, in both modes: 1.0 with
    --thinking (source "thinking"), 0.0 without (source "default"), or
    GBENCH_SKILLSBENCH_TEMPERATURE, else --temperature (base.resolve_temperature; the runner's
    `temperature` kwarg is the --temperature knob)."""
    value, source = resolve_temperature("skillsbench", None, thinking=bool(enable_thinking))
    passed = (kwargs or {}).get("temperature")
    if source in ("thinking", "default") and passed is not None and str(passed).strip() != "":
        return float(passed), "cli:--temperature"
    return value, source


def sampling_note(applied: Tuple[float, str], enable_thinking: Optional[bool] = None) -> str:
    """The result's `sampling`: the temperature and the thinking switch the relay set."""
    value, source = applied
    thinking = ("" if enable_thinking is None else
                "; the relay also sets the chat template's thinking switch "
                f"(chat_template_kwargs.enable_thinking={str(bool(enable_thinking)).lower()}) "
                "on every such request")
    return (f"temperature {value} ({source}) set by gbench's relay on every chat-completions "
            "request of the delegated agent (deepagents) (gbench's rule for every suite: 1.0 "
            "with --thinking, 0.0 without)" + thinking)


def _image() -> str:
    return os.environ.get("GBENCH_SKILLSBENCH_IMAGE", _IMAGE_DEFAULT)


def image_harness_version(image: str) -> Optional[str]:
    """The HARNESS_LABEL value of `image` ("" when the label is absent), or None when docker
    cannot inspect it."""
    return image_label(image, HARNESS_LABEL)


def image_label(image: str, label: str) -> Optional[str]:
    """The `label` value of `image` ("" when the label is absent), or None when docker cannot
    inspect it."""
    r = subprocess.run(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "' + label + '"}}', image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    value = (r.stdout or "").strip()
    return "" if value == "<no value>" else value


def condition_limit_shares(limit: Optional[int], conditions: List[str]
                           ) -> Optional[Dict[str, int]]:
    """{condition: tasks it runs} for --eval-limit `limit` when more than one condition runs:
    N tasks for the whole run, never N per condition (sampling.allocate_limit, seed "skillsbench";
    the launcher draws each share with the same seeded draw, so the conditions run nested task
    sets). None for a run with one condition (the launcher's SKILLSBENCH_TASK_LIMIT is then N
    tasks of it) or without a limit."""
    if not limit or int(limit) <= 0 or len(conditions) < 2:
        return None
    return {c: int(n or 0)
            for c, n in allocate_limit(int(limit), conditions, seed="skillsbench").items()}


def explicit_tasks_env(limit: Optional[int]) -> Optional[str]:
    """GBENCH_SKILLSBENCH_TASKS (the launcher's explicit task list), or None. --eval-limit wins
    over it: with a limit set the knob is ignored with a warning, so the run is the N tasks dealt
    over the conditions, never every listed task in every condition."""
    tasks_env = (os.environ.get("GBENCH_SKILLSBENCH_TASKS") or "").strip() or None
    if tasks_env and limit and int(limit) > 0:
        logger.warning("[skillsbench] --eval-limit %d is set: GBENCH_SKILLSBENCH_TASKS (%s) is "
                       "ignored (--eval-limit wins; unset --eval-limit to run that task list)",
                       int(limit), tasks_env)
        return None
    return tasks_env


def limit_plan(limit: Optional[int], conditions: List[str],
               shares: Optional[Dict[str, int]], explicit_tasks: List[str]) -> Dict[str, Any]:
    """Log the run's planned workload (one line) and return its `eval_limit_plan`. Each (condition,
    task) is one rollout; a run with one condition runs its N tasks, one with both conditions the
    N dealt over them (`shares`). A full run (or a shard) runs the launcher's task list, whose
    count it knows (the full set is _FULL_TASK_COUNT tasks)."""
    lim = int(limit) if limit and int(limit) > 0 else None
    if explicit_tasks:
        per = {c: len(explicit_tasks) for c in conditions}
    elif shares is not None:
        per = dict(shares)
    elif lim:
        per = {conditions[0]: min(lim, _FULL_TASK_COUNT)} if conditions else {}
    else:
        logger.info("[skillsbench] full run (no --eval-limit): every task of the launcher's list "
                    "(%d in the full set) x %d condition(s) (%s) x 1 attempt",
                    _FULL_TASK_COUNT, len(conditions), ", ".join(conditions))
        return {"limit": None, "samples": None, "per_setting": None, "attempts_per_sample": 1,
                "rollouts": None}
    plan = eval_limit_plan("skillsbench", lim, sum(per.values()), per,
                           setting_noun="condition", attempts=1, unit="rollouts")
    # a sample is a (condition, task) rollout; the conditions' draws are nested (or the same
    # explicit list), so the distinct tasks are the largest condition's
    plan["distinct_tasks"] = max(per.values()) if per else 0
    return plan


def check_skillsbench_prerequisites() -> Tuple[bool, str]:
    """Docker (with a reachable daemon) + the locally-built image at the current adapter
    (HARNESS_LABEL=HARNESS_VERSION). No judge/GEMINI needed."""
    image = _image()
    build = (f"Build the harness LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/skillsbench.Dockerfile gbench/docker\n"
             f"It bundles pinned benchflow-ai/skillsbench + the BenchFlow runner. See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    found = image_harness_version(image)
    if found != HARNESS_VERSION:
        return False, (f"image {image!r} predates the current adapter (label {HARNESS_LABEL}="
                       f"{HARNESS_VERSION}; the image has "
                       f"{repr(found) if found else 'none'}): its launcher would ignore a --shard "
                       f"(running the whole task set), take the first tasks of the sorted list "
                       f"for --eval-limit or run the with-skills condition without "
                       f"--skill-mode with-skill. Rebuild it. " + build)
    return True, ""


def _task_reachable_endpoint(base_url: str) -> str:
    """Translate the gbench endpoint into a URL reachable FROM the sibling task containers.

    The orchestrator runs `--network host`, but the TASK containers BenchFlow spawns are on the
    default bridge, so 127.0.0.1 there is the container itself - they reach the host via the docker
    bridge gateway. `GBENCH_SKILLSBENCH_TASK_ENDPOINT` overrides the whole URL; else a
    localhost/127.0.0.1 host is rewritten to `GBENCH_SKILLSBENCH_TASK_HOST` (default 172.17.0.1).
    The run no longer uses it: the task containers always get the relay's address (the relay
    sets the run's temperature on every model request), so GBENCH_SKILLSBENCH_TASK_ENDPOINT has
    no effect (a warning says so).
    """
    override = os.environ.get("GBENCH_SKILLSBENCH_TASK_ENDPOINT")
    if override:
        return override
    ep = base_url.rstrip("/")
    if not ep.endswith("/v1"):
        ep += "/v1"
    host = os.environ.get("GBENCH_SKILLSBENCH_TASK_HOST", "172.17.0.1")
    return re.sub(r"://(127\.0\.0\.1|localhost)(?=[:/])", f"://{host}", ep)


def endpoint_api_key() -> Optional[str]:
    """The served endpoint's key: GBENCH_SKILLSBENCH_API_KEY, else base.endpoint_api_key
    (VLLM_API_KEY, else OPENAI_API_KEY); None when none is set (a placeholder such as "dummy" or
    "EMPTY" counts as unset)."""
    key = (os.environ.get("GBENCH_SKILLSBENCH_API_KEY") or "").strip()
    if key and key not in ("dummy", "EMPTY"):
        return key
    return _endpoint_api_key()


def endpoint_key_source() -> Optional[str]:
    """The variable endpoint_api_key read the key from, or None when there is no key."""
    key = (os.environ.get("GBENCH_SKILLSBENCH_API_KEY") or "").strip()
    if key and key not in ("dummy", "EMPTY"):
        return "GBENCH_SKILLSBENCH_API_KEY"
    from .base import ENDPOINT_KEY_ENVS
    found = _endpoint_api_key()
    if not found:
        return None
    return next((n for n in ENDPOINT_KEY_ENVS if (os.environ.get(n) or "").strip() == found),
                ENDPOINT_KEY_ENVS[0])


def relay_host() -> str:
    """Where the key relay listens: the address the task containers reach this host at,
    GBENCH_SKILLSBENCH_TASK_HOST, default 172.17.0.1 (docker's default bridge gateway)."""
    return (os.environ.get("GBENCH_SKILLSBENCH_TASK_HOST") or "").strip() or "172.17.0.1"


def orchestrator_env() -> Dict[str, str]:
    """The environment of the `docker run` of the orchestrator: this process's without any
    credential variable (sandbox.credential_free_env), with GBENCH_MODEL_API_KEY the placeholder.
    The command line passes GBENCH_MODEL_API_KEY by name only, so docker copies it from here."""
    env = credential_free_env()
    env["GBENCH_MODEL_API_KEY"] = MODEL_KEY_PLACEHOLDER
    return env


def dns_candidates(workdir: str, summary: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """{condition: {task: hosts}} for each task with a reward of 0 whose rollout's verifier
    output (workdir/jobs/<condition>/<task>/**/verifier/test-std*.txt) shows the package hosts it
    could not resolve before its tests ran (swebench_common.verifier_dns_failure), unless the
    agent's transcript there (trajectory/ and agent/) names /etc/resolv.conf, /etc/hosts or
    /etc/nsswitch.conf (the model may have broken name resolution itself)."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for cond, info in ((summary or {}).get("conditions") or {}).items():
        for task, reward in ((info or {}).get("per_task") or {}).items():
            if not isinstance(reward, (int, float)) or reward > 0:
                continue
            jobs = os.path.join(workdir, "jobs", cond, task)
            verifier = sorted(glob.glob(os.path.join(jobs, "**", "verifier", "test-std*.txt"),
                                        recursive=True))
            hosts = swebench_common.verifier_dns_failure(swebench_common.read_texts(verifier))
            if not hosts:
                continue
            agent = sorted(glob.glob(os.path.join(jobs, "**", "trajectory", "*"), recursive=True)
                           + glob.glob(os.path.join(jobs, "**", "agent", "**", "*"),
                                       recursive=True))
            if swebench_common.touched_resolver(swebench_common.read_texts(
                    [p for p in agent if os.path.isfile(p)])):
                continue
            out.setdefault(cond, {})[task] = hosts
    return out


def dns_recheck(hosts: List[str]) -> Optional[List[str]]:
    """The hosts of `hosts` a fresh probe container of the orchestrator image cannot resolve
    after the run, on the task networks; None when it could not tell
    (swebench_common.unresolved_after_run)."""
    return swebench_common.unresolved_after_run("skillsbench", [_image()], hosts,
                                                docs_url=DOCS_URL)


def dns_harness_errors(workdir: str, summary: Dict[str, Any]
                       ) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, Any]]]:
    """(harness errors {condition: {task: hosts}}, kept {condition: {task: {hosts, note}}}): the
    DNS candidates (dns_candidates) whose failure something outside their task container
    corroborates (another candidate in the run, or dns_recheck after the run;
    swebench_common.corroborated_dns_failures) are harness errors; the others keep their reward
    (the model's outcome), with a note."""
    found = dns_candidates(workdir, summary)
    flat = {(cond, task): hosts for cond, tasks in found.items() for task, hosts in tasks.items()}
    errors, kept = swebench_common.corroborated_dns_failures(
        {f"{c}/{t}": h for (c, t), h in flat.items()}, recheck=dns_recheck)
    out: Dict[str, Dict[str, List[str]]] = {}
    notes: Dict[str, Dict[str, Any]] = {}
    for (cond, task), hosts in flat.items():
        key = f"{cond}/{task}"
        if key in errors:
            out.setdefault(cond, {})[task] = hosts
        elif key in kept:
            notes.setdefault(cond, {})[task] = {"hosts": hosts, "note": kept[key]}
    return out, notes


#: A `docker compose` command failing in BenchFlow's docker sandbox
#: (DockerSandbox._run_docker_compose_command's RuntimeError message); group 1 is the subcommand
#: after the last compose file (`build`, `up`, `cp`, `exec`, ...).
_COMPOSE_FAILURE = re.compile(
    r"^Docker compose command failed for environment \S+\. Command: docker compose .*?"
    r"\.ya?ml ([a-z][\w-]*)", re.S)
#: BenchFlow's error categories (benchflow._utils.scoring.classify_error, BenchFlow 0.6.3) of a
#: failure in the agent's phase: its model requests, its ACP transport or its time limits. Such a
#: rollout is in stage `agent` even when the agent produced no token (its first request failed).
AGENT_ERROR_CATEGORIES = frozenset({
    "idle_timeout", "pipe_closed", "acp_error", "api_error", "suspected_api_error",
    "provider_auth", "provider_rate_limit", "timeout"})
#: Characters of a rollout's error kept in a no-reward task's `error_excerpt` (its end, where the
#: failing command's last output is).
ERROR_EXCERPT_CHARS = 600


def no_reward_cause(result: Dict[str, Any]) -> Dict[str, Any]:
    """What a rollout's result.json (BenchFlow RunResult) says about a task that wrote no reward:
    `benchflow_error_category` (BenchFlow's error_category), `stage`, `agent_ran` (any tool call,
    token or trajectory step) and `error_excerpt` (the error's last ERROR_EXCERPT_CHARS
    characters, whitespace collapsed), plus `compose_command` (the subcommand) when a
    `docker compose` command failed. `stage` is, in this order: `environment_build` /
    `environment_start` when the task environment's `docker compose build` / `up` failed;
    `sandbox_command` for any other failed compose command (BenchFlow 0.6.3 checks `cp`, `stop`
    and `down`, used before the agent, before the verifier and at teardown; agent_ran tells
    which side of the agent it was); `environment_setup` for a sandbox startup failure (category
    `sandbox_setup`, `sandbox_startup_info`);
    `agent_setup` when the agent's install failed (`install_failure`); `verifier` when the
    verifier errored; `agent` when the agent ran or the category is one of the agent's phase
    (AGENT_ERROR_CATEGORIES); else `unknown` (the excerpt says what failed). It describes the
    failure; it never gives the task a reward. 2026-10-13 review: every failure other than a
    compose build / up or a verifier error was labelled `agent`, also when the agent never ran."""
    error = str(result.get("error") or "")
    category = result.get("error_category")
    agent = result.get("agent_result") if isinstance(result.get("agent_result"), dict) else {}
    traj = (result.get("trajectory_summary")
            if isinstance(result.get("trajectory_summary"), dict) else {})
    agent_ran = bool(result.get("n_tool_calls") or agent.get("n_tool_calls")
                     or agent.get("total_tokens") or traj.get("steps"))
    compose = _COMPOSE_FAILURE.match(error)
    if compose and compose.group(1) == "build":
        stage = "environment_build"
    elif compose and compose.group(1) == "up":
        stage = "environment_start"
    elif compose:
        stage = "sandbox_command"
    elif category == "sandbox_setup" or result.get("sandbox_startup_info"):
        stage = "environment_setup"
    elif category == "install_failure":
        stage = "agent_setup"
    elif result.get("verifier_error"):
        stage = "verifier"
    elif agent_ran or category in AGENT_ERROR_CATEGORIES:
        stage = "agent"
    else:
        stage = "unknown"
    tail = " ".join((error or str(result.get("verifier_error") or "")).split())
    return {
        "benchflow_error_category": category,
        "stage": stage,
        "agent_ran": agent_ran,
        "error_excerpt": tail[-ERROR_EXCERPT_CHARS:] or None,
        **({"compose_command": compose.group(1)} if compose else {}),
    }


def no_reward_causes(workdir: str, summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{condition: {task: cause}} for each task of the launcher's summary without a reward: its
    newest rollout result.json under workdir/jobs/<condition>/<task>/ (the one the launcher read,
    _reward_from_jobs) read by no_reward_cause, plus `rollout` (that rollout's directory name, as
    kept under artifacts_dir); `{"stage": None, "note": ...}` when there is none or it cannot be
    read."""
    out: Dict[str, Dict[str, Any]] = {}
    for cond, info in ((summary or {}).get("conditions") or {}).items():
        for task, reward in ((info or {}).get("per_task") or {}).items():
            if isinstance(reward, (int, float)):
                continue
            found = sorted(glob.glob(os.path.join(workdir, "jobs", cond, task, "**",
                                                  "result.json"), recursive=True))
            if not found:
                cause: Dict[str, Any] = {"stage": None, "note": "the rollout wrote no result.json"}
            else:
                try:
                    with open(found[-1], encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("not a JSON object")
                    cause = no_reward_cause(data)
                except (OSError, ValueError) as e:
                    cause = {"stage": None,
                             "note": f"the rollout's result.json could not be read ({e})"}
                cause["rollout"] = os.path.basename(os.path.dirname(found[-1]))
            out.setdefault(cond, {})[task] = cause
    return out


def exclude_harness_errors(summary: Dict[str, Any],
                           dns: Dict[str, Dict[str, List[str]]],
                           causes: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A copy of the launcher's summary with each task of `dns` left out of its condition's mean
    (`per_task` None, listed in `harness_errors` with its hosts) and `mean_reward` / `n_scored`
    recomputed over the tasks left; `harness_errors` also lists each task that produced no
    reward (kind `no_reward`, with its cause from `causes`, no_reward_causes, and the environment
    builds BenchFlow re-ran for it, `build_retries`, when the summary records any). A task
    without a reward stays without one."""
    retried = (summary or {}).get("sandbox_build_retries") or {}
    out = copy.deepcopy(summary or {})
    for cond, info in (out.get("conditions") or {}).items():
        per_task = info.get("per_task")
        if not isinstance(per_task, dict):
            continue
        errors: Dict[str, Any] = {}
        for task, reward in list(per_task.items()):
            if task in (dns.get(cond) or {}):
                errors[task] = {"kind": "dns", "hosts": dns[cond][task]}
                per_task[task] = None
            elif not isinstance(reward, (int, float)):
                errors[task] = {"kind": "no_reward", **((causes or {}).get(cond) or {}).get(
                    task, {})}
                if ((retried.get(cond) or {}).get(task)):
                    errors[task]["build_retries"] = retried[cond][task]
        vals = [v for v in per_task.values() if isinstance(v, (int, float))]
        info["mean_reward"] = (sum(vals) / len(vals)) if vals else None
        info["n_scored"] = len(vals)
        info["harness_errors"] = errors
    ws = (out.get("conditions") or {}).get("with-skills", {}).get("mean_reward")
    ns = (out.get("conditions") or {}).get("without-skills", {}).get("mean_reward")
    if isinstance(ws, (int, float)) and isinstance(ns, (int, float)):
        out["with_skills_lift"] = ws - ns
    else:
        out.pop("with_skills_lift", None)
    return out


def compute_skillsbench_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Roll the launcher's summary into a headline mean reward.

    Primary condition = without-skills if present, else the first condition with a mean reward.
    """
    conds = (summary or {}).get("conditions") or {}
    means = {c: v.get("mean_reward") for c, v in conds.items()
             if isinstance(v.get("mean_reward"), (int, float))}
    if not means:
        return None
    primary = "without-skills" if "without-skills" in means else next(iter(means))
    return {
        "primary_condition": primary,
        "mean_reward": means[primary],
        "condition_means": means,
        "with_skills_lift": summary.get("with_skills_lift"),
    }


def run_skillsbench(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical SkillsBench via the upstream BenchFlow runner (docker-out-of-docker) +
    deterministic per-task verifier."""
    shard = current_shard()       # a malformed GBENCH_SHARD is an error before anything runs
    ok, reason = check_skillsbench_prerequisites()
    if not ok:
        raise infra_required("skillsbench", reason, DOCS_URL)
    # The agent runs the model's commands in each task container, with network access: the cloud
    # metadata server must be blocked there before BenchFlow starts the first one.
    swebench_common.require_task_network_blocked(
        "skillsbench", [_image()], docs_url=DOCS_URL,
        hint=f"Build the orchestrator image {_image()} (see {DOCS_URL}).",
        resolve=swebench_common.TASK_INSTALL_HOSTS)

    # The run's temperature (1.0 with --thinking, 0.0 without, or the operator's value): the
    # relay below sets it on every model request of the delegated agent (deepagents).
    applied_temperature = thinking_temperature(bool(enable_thinking), kwargs)
    conditions = os.environ.get("GBENCH_SKILLSBENCH_CONDITIONS", "without-skills")
    # The pinned BenchFlow (0.6.3) does not ship the leaderboard's `opencode` agent; its open-model
    # agents (deepagents/codex/...) read BENCHFLOW_PROVIDER_BASE_URL. Default to deepagents.
    agent = os.environ.get("GBENCH_SKILLSBENCH_AGENT", "deepagents")
    limit = kwargs.get("limit")

    # docker-out-of-docker: BenchFlow bind-mounts its working dirs into sibling task containers, so
    # the workdir must be an IDENTITY mount (same path on host + in the orchestrator) with TMPDIR
    # pointing there - otherwise the sibling sandbox gets empty mounts and no reward is collected.
    workdir = tempfile.mkdtemp(prefix="gbench_skillsbench_")
    os.chmod(workdir, 0o777)  # the container writes as its own uid
    orch_name = "gbench_skillsbench_" + os.path.basename(workdir)
    keep_dir = None

    relay: Optional[EndpointRelay] = None

    def _reap():
        # docker --rm only fires on a clean container EXIT; on a subprocess timeout the client is
        # killed while the orchestrator keeps running, so reap it by name in finally + on timeout.
        # (BenchFlow owns its per-task docker sandbox siblings, which it spawns with their own --rm;
        # gbench has no hook to label them, so their lifecycle stays with BenchFlow.)
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)
        if relay is not None:
            relay.stop()

    try:
        # An endpoint key never leaves this process: the agent in each task container would hand
        # it to every command the model runs there. With a key, the task containers get the
        # address of an EndpointRelay (a thread here, listening where they reach this host) that
        # adds it, and every container gets the placeholder MODEL_KEY_PLACEHOLDER.
        # Every run goes through the same relay, which sets the run's temperature on every
        # chat-completions request (the agent's own client sends its default).
        key = endpoint_api_key()
        why = ("GBENCH_SKILLSBENCH_API_KEY is set" if key
               else "gbench sets the run's temperature")
        if os.environ.get("GBENCH_SKILLSBENCH_TASK_ENDPOINT"):
            logger.warning("skillsbench: GBENCH_SKILLSBENCH_TASK_ENDPOINT is not used while "
                           "%s; the task containers reach the model through gbench's relay "
                           "instead", why)
        # The run's temperature and the chat template's thinking switch (true with --thinking,
        # false without): deepagents' ChatOpenAI sends neither, so before 2026-10-11 a
        # --thinking run reached the model with thinking off (thinking_applied: false).
        body = {"temperature": applied_temperature[0],
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
        timeouts = transport_timeouts()
        request_timeout = timeouts["model_client_timeout"]["timeout_s"]
        try:
            relay = EndpointRelay(base_url, relay_host(), key=key, body_fields=body,
                                  name="skillsbench-key-relay",
                                  allowed_paths=RELAY_PATHS,
                                  # outlasts the agent's own request timeout
                                  timeout_s=request_timeout + 120).start()
        except (OSError, ValueError) as e:
            need = ("the endpoint key (GBENCH_SKILLSBENCH_API_KEY) must not reach the task "
                    "containers, where the model's commands can read the agent's environment; "
                    "gbench relays their model requests through a relay that adds the key"
                    if key else
                    "the run samples the model at temperature "
                    f"{applied_temperature[0]}, which gbench sets on the agent's model "
                    "requests through a relay")
            raise infra_required(
                "skillsbench", f"{need}, and it could not listen on {relay_host()} "
                f"({type(e).__name__}: {e}). Set GBENCH_SKILLSBENCH_TASK_HOST to the address "
                "the task containers reach this host at.", DOCS_URL) from e
        task_endpoint = relay.url
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host",
               "-v", "/var/run/docker.sock:/var/run/docker.sock",
               "-v", f"{workdir}:{workdir}:rw",              # identity mount (DooD)
               "-e", f"SKILLSBENCH_WORKDIR={workdir}",
               "-e", f"GBENCH_MODEL_BASE_URL={task_endpoint}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               # By name: docker copies the placeholder from orchestrator_env(), never a key.
               "-e", "GBENCH_MODEL_API_KEY",
               "-e", f"SKILLSBENCH_CONDITIONS={conditions}",
               "-e", f"SKILLSBENCH_AGENT={agent}",
               "-e", f"SKILLSBENCH_CONCURRENCY={max(1, int(concurrency))}",
               # BenchFlow's idle watchdog and the agent's request timeout (transport_timeouts)
               "-e", "SKILLSBENCH_AGENT_IDLE_TIMEOUT_S="
               f"{timeouts['agent_idle_timeout']['timeout_s']}",
               "-e", f"GBENCH_MODEL_REQUEST_TIMEOUT_S={request_timeout}"]
        tasks_env = explicit_tasks_env(limit)
        cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
        cond_shares = None if tasks_env else condition_limit_shares(limit, cond_list)
        workload = limit_plan(limit, cond_list, cond_shares,
                              [t for t in (tasks_env or "").split(",") if t.strip()])
        if cond_shares is not None:
            # the adapter at HARNESS_VERSION 5 deals them (check_skillsbench_prerequisites)
            cmd += ["-e", "SKILLSBENCH_CONDITION_LIMITS="
                    + json.dumps(cond_shares, sort_keys=True)]
        elif limit and int(limit) > 0:
            cmd += ["-e", f"SKILLSBENCH_TASK_LIMIT={int(limit)}"]
        if tasks_env:
            cmd += ["-e", f"SKILLSBENCH_TASKS={tasks_env}"]
        # --shard I/N: the launcher (_task_list) takes shard I of N of the sorted task list
        # (round-robin, applied before the limit); it gets the validated, normalized "I/N".
        if shard is not None:
            cmd += ["-e", f"GBENCH_SHARD={shard.label}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_SKILLSBENCH_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("skillsbench: docker run %s (agent=%s, conditions=%s)", _image(), agent, conditions)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                                  env=orchestrator_env())
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "skillsbench", f"the skillsbench container exceeded {timeout_s}s and was killed.",
                DOCS_URL) from exc

        summ_path = os.path.join(workdir, "skillsbench_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "skillsbench",
                f"the container produced no summary (rc={proc.returncode}; a harness failure, not a "
                f"0%). tail: {(proc.stderr or proc.stdout or '')[-800:]}", DOCS_URL)
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
        # Read the verifiers' output and the no-reward rollouts' results before the work
        # directory goes.
        dns, dns_kept = dns_harness_errors(workdir, summary)
        causes = no_reward_causes(workdir, summary)
    finally:
        _reap()
        # Each rollout's evidence kept with the results; then the work directory removed, the
        # root-owned files the task containers wrote included.
        try:
            keep_dir = keep_artifacts(workdir, artifacts_dir())
        except Exception as e:  # a record of the run, never a reason to lose its result
            logger.warning("skillsbench: could not keep the rollouts' artifacts (%s)", e)
        remove_workdir(workdir, _image())

    if shard is not None and shard.partial and not summary.get("n_tasks_selected"):
        # An empty shard (more shards than tasks): nothing ran here, nothing is scored.
        reason = (f"empty shard {shard.label}: no items on this node "
                  f"({summary.get('n_tasks_full')} in the full set)")
        logger.info("skillsbench: %s", reason)
        return {
            "benchmark_type": "eval", "eval_name": "skillsbench", "model_name": model_name,
            "thinking": bool(enable_thinking), "status": "success", "status_reason": reason,
            "total_questions": 0, "correct_answers": 0, "accuracy": None,
            "accuracy_withheld_reason": reason, "empty_shard": True,
            "full_set_items": summary.get("n_tasks_full"), "sample_traces": [],
            "tasks_selected": [], **shard_identity(shard), "eval_limit_plan": workload,
            "leaderboard_comparable": False, "leaderboard_comparable_reason": reason,
        }
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    if limit and int(limit) > 0 and not tasks_env and (
            not selection.get("limit_draw")
            or (cond_shares is not None and selection.get("condition_limits") != cond_shares)):
        raise infra_required(
            "skillsbench", "the orchestrator does not report how it drew the --eval-limit subset "
            "(an image built before the seeded draw, which took the first tasks of the sorted "
            "list); rebuild the image from gbench/docker/skillsbench.Dockerfile.", DOCS_URL)
    scoring = exclude_harness_errors(summary, dns, causes)
    errors = {c: (i or {}).get("harness_errors") or {}
              for c, i in (scoring.get("conditions") or {}).items()}
    n_dns = sum(1 for e in errors.values() for x in e.values() if x.get("kind") == "dns")
    n_no_reward = sum(1 for e in errors.values() for x in e.values()
                      if x.get("kind") == "no_reward")
    scored = compute_skillsbench_score(scoring)
    if scored is None:
        hosts = sorted({h for t in dns.values() for hs in t.values() for h in hs})
        raise infra_required(
            "skillsbench", "no task was scored: "
            + (f"{n_dns} task verifier(s) could not resolve {', '.join(hosts)} before their tests "
               "ran (a DNS failure on this host's docker networks: on GCE the VPC DNS resolver is "
               "169.254.169.254, and the metadata server rule must spare port 53), and "
               if n_dns else "")
            + f"{n_no_reward} task(s) produced no reward (harness errors, not the model's "
            "outcome; check Docker egress and that the /v1 endpoint is reachable from task "
            "containers).", DOCS_URL)

    primary = scored["primary_condition"]
    n_tasks = (scoring.get("conditions", {}).get(primary, {}) or {}).get("n_tasks")
    n_scored = (scoring.get("conditions", {}).get(primary, {}) or {}).get("n_scored")
    both_conditions = len(scored["condition_means"]) >= 2
    # A sharded run covers only a subset by construction (n_tasks already reflects the shard size),
    # so it is never "full" regardless of coverage math.
    sharded = shard is not None and shard.partial
    is_full = (not sharded) and (not limit or int(limit) <= 0) and (n_tasks or 0) >= _FULL_TASK_COUNT
    # The run's temperature, set by gbench's relay, is not the agent's default the leaderboard
    # runs, so the result is never leaderboard-comparable (the reasons below name the rest).
    leaderboard = False

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "skillsbench",
        "model_name": model_name,
        # Harness errors (a verifier's DNS failure, a task with no reward) are counted and named
        # below and left out of the means, never scored.
        "status": "completed_with_errors" if (n_dns or n_no_reward) else "success",
        "harness_errors": {k: v for k, v in (("dns", n_dns), ("no_reward", n_no_reward)) if v},
        "harness_error_tasks": {c: e for c, e in errors.items() if e},
        # environment builds BenchFlow re-ran after a transient failure, before the agent started
        # ({condition: {task: n}}; None: the image does not record them)
        "sandbox_build_retries": summary.get("sandbox_build_retries"),
        # Reward-0 tasks whose verifier hit an uncorroborated DNS failure: the model's outcome.
        **({"dns_failures_kept": dns_kept} if dns_kept else {}),
        "accuracy": round(scored["mean_reward"] * 100.0, 2),  # headline = mean reward (%) of primary condition
        "mean_reward": round(scored["mean_reward"], 4),
        "primary_condition": primary,
        "condition_means": {k: round(v, 4) for k, v in scored["condition_means"].items()},
        "with_skills_lift": (round(scored["with_skills_lift"], 4)
                             if isinstance(scored.get("with_skills_lift"), (int, float)) else None),
        "n_tasks": n_tasks,
        "n_scored": n_scored,
        # The summary row: the primary condition's tasks, those scored (the mean's denominator)
        # and their rewards summed (accuracy = correct_answers / effective_n).
        **headline_counts(scoring, primary),
        "agent": agent,
        "image": _image(),
        "image_harness_version": HARNESS_VERSION,
        "scoring": "deterministic per-task verifier (no LLM judge)",
        "sampling": sampling_note(applied_temperature, bool(enable_thinking)),
        # the relay sets the chat template's thinking switch and the temperature on every
        # chat-completions request (model_relay.sets_body_fields)
        "thinking": bool(enable_thinking),
        **thinking_record(agent, relay.record() if relay is not None else None),
        # the agent's request timeout and BenchFlow's idle watchdog (no 600 s defaults), as
        # far as the agent uses them (agent_timeouts), and what the launcher passed (raw_summary)
        **agent_timeouts(agent, timeouts),
        "launcher_timeouts": {"agent_idle_timeout_s": summary.get("agent_idle_timeout_s"),
                              "model_request_timeout_s": summary.get("model_request_timeout_s")},
        "temperature_applied": True, "temperature": applied_temperature[0],
        "temperature_source": applied_temperature[1],
        "max_output_tokens_note": _max_tokens_note(bool(enable_thinking), kwargs),
        **_attempts_record(kwargs),
        **shard_identity(shard),
        # the tasks the launcher ran (its shard of the sorted list, then its seeded draw of
        # --eval-limit tasks balanced across the categories: `selection`)
        "tasks_selected": summary.get("tasks_selected"),
        "selection": selection or None,
        "eval_limit_plan": workload,
        "sample_traces": task_traces(scoring),
        "raw_summary": summary,
        # each rollout's result, rewards, verifier output and trajectory (capped), kept
        "artifacts_dir": keep_dir,
        "metric": ("SkillsBench mean reward over per-task deterministic verifiers (reward in [0,1]); "
                   "headline is the primary condition, with the with-skills-vs-without lift when "
                   f"both conditions run. Agent loop delegated to the upstream BenchFlow runner "
                   f"({agent} agent); no LLM judge."),
        "leaderboard_comparable": leaderboard,
    }
    if relay is not None:
        rec = relay.record()
        if key:
            rec["adds"] = f"Authorization: Bearer <{endpoint_key_source()}>"
        rec["sets_temperature_on"] = list(THINKING_RELAY_PATHS)
        rec["sets_thinking_switch_on"] = list(THINKING_RELAY_PATHS)
        result["model_relay"] = rec
    reasons = []
    if agent != "opencode":
        reasons.append(f"agent '{agent}' differs from the leaderboard's reference agent (opencode); "
                       f"the pinned BenchFlow 0.6.3 default is deepagents")
    if not is_full and not sharded:
        reasons.append("not the full 87-task set")
    if not both_conditions:
        reasons.append("only one condition run (upstream reports with-skills-vs-without)")
    reasons.append(f"temperature {applied_temperature[0]} set by "
                   f"{applied_temperature[1].split(':', 1)[-1]} on every model request "
                   "(the leaderboard runs the agent's own default)")
    if n_dns or n_no_reward:
        reasons.append(f"{n_dns + n_no_reward} task run(s) lost to harness errors ("
                       f"{n_dns} DNS failure(s) in the verifier's package installs, "
                       f"{n_no_reward} with no reward), left out of the means")
    reasons.append("single trial (upstream averages 3)")
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    if sharded:
        add_comparability_reason(result, SHARD_REASON.format(
            label=shard.label, index=shard.index, count=shard.count))
    return result


def headline_counts(scoring: Dict[str, Any], primary: Optional[str]) -> Dict[str, Any]:
    """total_questions (the primary condition's tasks), effective_n (those scored, the mean's
    denominator) and correct_answers (their rewards summed), so the summary row reconciles with
    the headline mean reward."""
    info = ((scoring or {}).get("conditions") or {}).get(primary or "") or {}
    errors = info.get("harness_errors") or {}
    per_task = info.get("per_task") or {}
    rewards = [float(r) for t, r in per_task.items()
               if t not in errors and isinstance(r, (int, float))]
    total = info.get("n_tasks") if info.get("n_tasks") is not None else len(per_task)
    return {"total_questions": total,
            "effective_n": info.get("n_scored") if info.get("n_scored") is not None
            else len(rewards),
            "correct_answers": round(sum(rewards), 2),
            "correct_answers_note": ("solved-equivalent: the primary condition's task rewards "
                                     "(0-1) summed over the scored tasks")}


#: Files of each rollout (jobs/<condition>/<task>/<time>/<task>__<id>/) kept with the run's
#: results, and the most bytes kept of each (None: all of it; a larger file keeps its first and
#: last halves around a marker line). The model-call log (trajectory/llm_trajectory.jsonl, often
#: 100 MB) is not kept; the agent's own trajectory is.
ARTIFACT_ROLLOUT_FILES = {"result.json": None, "rewards.jsonl": None, "timing.json": None,
                          "config.json": None, "verifier/reward.txt": None,
                          "verifier/ctrf.json": 1024 * 1024,
                          "verifier/test-stdout.txt": 1024 * 1024,
                          "verifier/test-stderr.txt": 1024 * 1024,
                          "trajectory/acp_trajectory.jsonl": 4 * 1024 * 1024}


def artifacts_dir() -> Optional[str]:
    """`<GBENCH_RESULTS_DIR>/skillsbench_artifacts/<fresh dir>`, or None outside a gbench run."""
    run_dir = (os.environ.get("GBENCH_RESULTS_DIR") or "").strip()
    if not run_dir:
        return None
    base_dir = os.path.join(run_dir, "skillsbench_artifacts")
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


def keep_artifacts(workdir: str, dest: Optional[str]) -> Optional[str]:
    """Copy each rollout's ARTIFACT_ROLLOUT_FILES and the orchestrator's summary from `workdir`
    to `dest` (rollouts under <condition>/<task>/<rollout>/), every credential value of the
    environment redacted; a file this user cannot read is logged and left out. Returns `dest`
    (None when there is none)."""
    if not dest:
        return None
    copies = [(os.path.join(workdir, "skillsbench_summary.json"),
               os.path.join(dest, "skillsbench_summary.json"), None)]
    jobs = os.path.join(workdir, "jobs")
    for result in sorted(glob.glob(os.path.join(jobs, "*", "*", "*", "*", "result.json"))):
        rollout = os.path.dirname(result)
        cond, task = os.path.relpath(rollout, jobs).split(os.sep)[:2]
        out = os.path.join(dest, cond, task, os.path.basename(rollout))
        for rel, cap in ARTIFACT_ROLLOUT_FILES.items():
            copies.append((os.path.join(rollout, rel), os.path.join(out, rel), cap))
    for src, dst, cap in copies:
        if not os.path.isfile(src):
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _copy_capped(src, dst, cap)
            redact_file(dst)
        except OSError as e:
            logger.warning("skillsbench: could not keep %s (%s)", src, e)
    return dest


def remove_workdir(workdir: str, image: Optional[str] = None) -> List[str]:
    """Remove the run's work directory; the paths that could not be removed (logged). What the
    task containers wrote as root is removed with a throwaway container of the orchestrator image
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
            logger.warning("skillsbench: the cleanup container could not run (%s)",
                           type(e).__name__)
        left = []
        shutil.rmtree(workdir, onerror=_onerror)
    if left:
        logger.warning("skillsbench: %d path(s) of the work directory %s could not be removed "
                       "(e.g. %s)", len(left), workdir, left[0])
    return left


def task_traces(scoring: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One trace per (condition, task) of the scored summary (exclude_harness_errors): its
    `reward` (None for a harness error, listed under `harness_error` with `scoring_excluded`),
    `is_correct` (reward 1.0), `category` (the condition) and `sample_key`."""
    traces: List[Dict[str, Any]] = []
    for cond, info in sorted(((scoring or {}).get("conditions") or {}).items()):
        errors = (info or {}).get("harness_errors") or {}
        for task, reward in sorted(((info or {}).get("per_task") or {}).items()):
            trace: Dict[str, Any] = {
                "condition": cond, "task": task, "category": cond,
                "reward": reward if isinstance(reward, (int, float)) else None,
                "is_correct": isinstance(reward, (int, float)) and reward >= 1.0,
                "sample_key": f"task:{cond}/{task}"}
            if task in errors or not isinstance(reward, (int, float)):
                trace["harness_error"] = errors.get(task) or {"kind": "no_reward"}
                trace["scoring_excluded"] = True
            traces.append(trace)
    return traces


def _max_tokens_note(thinking: bool, kwargs: Dict[str, Any]) -> str:
    """What the agent sends as max_tokens: the delegated agent's own setting (deepagents through
    BenchFlow), which gbench does not change; --max-output-tokens is not applied."""
    knob = kwargs.get("max_output_tokens") or get_run_knob("max_output_tokens")
    return ("the delegated agent (deepagents through BenchFlow) sets its own per-request token "
            "limit, which gbench does not change"
            + (f"; --max-output-tokens {knob} not applied" if knob else ""))


def _attempts_record(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """{"attempts": {...not_applied}} when --attempt-count asked for more than 1: the launcher
    runs each task once per condition."""
    k = int(kwargs.get("attempt_count") or get_run_knob("attempt_count") or 1)
    if k <= 1:
        return {}
    return {"attempts": {"attempts_requested": k, "attempts_per_sample": 1,
                         "not_applied": ("skillsbench: the launcher runs each task once per "
                                         f"condition; --attempt-count {k} is not applied")}}


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run's results (gbench --merge): each condition's mean reward
    recomputed over the union of the shards' (condition, task) traces (harness errors left out),
    the headline the primary condition's (without-skills when it ran), and the lift."""
    from ...merge import MergeError
    union: Dict[str, Dict[str, Any]] = {}
    for r in results:
        traces = r.get("sample_traces")
        if not isinstance(traces, list):
            raise MergeError("skillsbench: a shard carries no task traces (sample_traces)")
        for t in traces:
            union.setdefault(str(t.get("sample_key")), t)
    traces = sorted(union.values(), key=lambda t: (str(t.get("condition")), str(t.get("task"))))
    by_cond: Dict[str, List[float]] = {}
    for t in traces:
        if not t.get("scoring_excluded") and isinstance(t.get("reward"), (int, float)):
            by_cond.setdefault(str(t["condition"]), []).append(float(t["reward"]))
    means = {c: sum(v) / len(v) for c, v in sorted(by_cond.items()) if v}
    merged = dict(results[0])
    merged["sample_traces"] = traces
    merged["condition_means"] = {c: round(v, 4) for c, v in means.items()}
    primary = "without-skills" if "without-skills" in means else next(iter(means), None)
    merged["primary_condition"] = primary
    merged["mean_reward"] = round(means[primary], 4) if primary else None
    merged["accuracy"] = round(means[primary] * 100.0, 2) if primary else None
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = "no task of any shard was scored"
    else:
        merged.pop("accuracy_withheld_reason", None)
    lift = (means["with-skills"] - means["without-skills"]
            if {"with-skills", "without-skills"} <= set(means) else None)
    merged["with_skills_lift"] = round(lift, 4) if lift is not None else None
    tasks = {str(t.get("task")) for t in traces}
    merged["n_tasks"] = len(tasks)
    merged["n_scored"] = len(by_cond.get(primary or "", []))
    primary_traces = [t for t in traces if str(t.get("condition")) == str(primary)]
    merged["total_questions"] = len(primary_traces)
    merged["effective_n"] = merged["n_scored"]
    merged["correct_answers"] = round(sum(by_cond.get(primary or "", [])), 2)
    merged["artifacts_dir"] = None
    merged["artifacts_dir_by_shard"] = {str(r.get("shard") or i + 1): r.get("artifacts_dir")
                                        for i, r in enumerate(results)}
    merged["tasks_selected"] = sorted(tasks)
    merged.pop("raw_summary", None)
    # every shard's harness errors and BenchFlow build retries, not the first shard's (the
    # shards hold disjoint tasks; 2026-10-13 review)
    counts: Dict[str, int] = {}
    error_tasks: Dict[str, Dict[str, Any]] = {}
    retries: Optional[Dict[str, Dict[str, int]]] = None
    for r in results:
        for kind, n in (r.get("harness_errors") or {}).items():
            counts[kind] = counts.get(kind, 0) + int(n or 0)
        for cond, entries in (r.get("harness_error_tasks") or {}).items():
            error_tasks.setdefault(cond, {}).update(entries or {})
        if isinstance(r.get("sandbox_build_retries"), dict):
            retries = retries if retries is not None else {}
            for cond, by_task in r["sandbox_build_retries"].items():
                for task, n in (by_task or {}).items():
                    row = retries.setdefault(cond, {})
                    row[task] = row.get(task, 0) + int(n or 0)
    merged["harness_errors"] = {k: v for k, v in counts.items() if v}
    merged["harness_error_tasks"] = {c: e for c, e in error_tasks.items() if e}
    merged["sandbox_build_retries"] = retries
    statuses = {r.get("status") for r in results}
    merged["status"] = "completed_with_errors" if "completed_with_errors" in statuses else "success"
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if (reason and not reason.startswith(shard_prefix)
                    and not reason.startswith("empty shard ") and reason not in kept
                    and not r.get("leaderboard_comparable", False)):
                kept.append(reason)
    if len(tasks) < _FULL_TASK_COUNT:
        kept.append(f"the merged shards hold {len(tasks)} tasks, not the full "
                    f"{_FULL_TASK_COUNT}")
    merged["leaderboard_comparable"] = not kept
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    # the shards' summed plan, not the first shard's
    from .limit_plan import merge_plans
    plan = merge_plans(results)
    if plan is not None:
        merged["eval_limit_plan"] = plan
    return merged
