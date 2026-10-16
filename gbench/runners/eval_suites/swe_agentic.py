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

"""Agentic SWE-bench rollouts: the default protocol for the SWE family.

SWE-bench-style benchmarks measure whether a model can resolve a real issue while working
inside the repository: read the code, run it, edit files, and submit a diff. gbench runs that
protocol with mini-swe-agent 1.x, the scaffold behind the SWE-bench bash-only leaderboard,
through its public API only:

  * the bundled ``extra/swebench.yaml`` agent config, unmodified (prompt templates, the
    250-step limit, 60 s per command). Images that keep the repository somewhere other than
    ``/testbed`` get that one path replaced by theirs, in the prompt and the environment cwd;
  * ``minisweagent.run.extra.swebench.get_sb_environment`` to start the task's Docker image
    (and its ``run.env_startup_command`` hook where a suite needs one);
  * ``DefaultAgent.run``, whose submission (``git add -A && git diff --cached``) is the patch;
  * ``save_traj`` for the per-instance trajectory file.

`_rollout` follows ``minisweagent.run.extra.swebench.process_instance`` step for step. The
differences are operational, not protocol: the model is pointed at the served OpenAI-compatible
endpoint through litellm with gbench's run knobs (the endpoint's key goes with each request, not
into the config mini-swe-agent writes into trajectory files; credential values found in those
files are replaced as a backstop), each worker gets its own copy of the config
(``get_sb_environment`` writes the image into the shared dict, which races across threads),
the container is cleaned up explicitly, the trajectory files go next to the run's results, and
a rollout that could not run for an infrastructure reason is recorded as such instead of as an
empty patch: a harness error (`agent_infra`), left out of the resolved rate's numerator and
denominator, as is an instance the harness gave no verdict (swebench_common.scored_headline); a
run with nothing left to score withholds its rate (`accuracy` None, status `error`).

Like upstream's bundled configs (1.x ``extra/swebench.yaml`` and 2.x
``benchmarks/swebench.yaml`` set no ``max_tokens``), `run_agentic` sends no per-turn
``max_tokens``: the server grants each turn what the context window has left. The operator's
``--max-output-tokens`` is recorded as not applied (``max_output_tokens_note``).

Suites whose leaderboard runs mini-swe-agent 2.x (tool-calling actions; SWE-bench Verified and
Multilingual) pass ``agent_v2=<release>`` instead. gemma4-env pins 1.x, so the 2.x release runs in
the local image ``gbench-mini-swe-agent-v2:2.4.2-2.4.6`` (gbench/docker/mini_swe_agent_v2.Dockerfile), one
container per instance, driven by gbench/docker/mini_swe_agent_v2_run.py. Each container runs on
docker's default bridge network with the placeholder key ``OPENAI_API_KEY=EMPTY`` and reaches the
served endpoint only through gbench's relay (swebench_common.EndpointRelay on that network's
gateway, which adds the endpoint's key; `require_relay_reachable` checks from that network that
the relay answers before the first rollout); it mounts the docker socket because upstream's
DockerEnvironment starts the instance container with ``docker run`` and runs the model's commands
there with ``docker exec`` (see _v2_container_cmd). That driver follows
2.x's ``run.benchmarks.swebench.process_instance`` with the release's bundled
``benchmarks/swebench.yaml`` and returns the same trace fields as the 1.x path. Replies with
reasoning but no content and no tool call are counted from each reply's raw response in the
instance's trajectory file (`v2_reply_counts`): 2.x files a reply without a tool call as a
format-error message of role "user", so the message role does not identify a model reply.

The model's commands run in each task's container, which has network access, so before the first
rollout `require_rollout_network_blocked` checks that the cloud metadata server is blocked for
it (swebench_common.require_task_network_blocked, in an image already on the host;
infra_required otherwise).

Each suite scores the patches with its existing harness (swebench, or swebench 5.x in its image).
swe_bench_live and multi_swe_bench run their leaderboards' reference scaffolds instead
(SWE-agent-for-eval and MSWE-agent, each in its local image) and use only this module's task,
selection, output-directory and labelling helpers; SWE-bench Pro runs Scale's V2 locked protocol
under Harbor (swe_bench_pro). The one-shot path (the old default: one diff from the issue text,
no repository access) stays available with ``GBENCH_<SUITE>_MODE=oneshot`` and is always
``leaderboard_comparable=False``.
"""

import asyncio
import collections
import concurrent.futures
import copy
import functools
import hashlib
import ipaddress
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import swe_thread_cap
from .base import (
    SecretRedactingFilter, answer_in_thinking_fields, clamp_to_context, decoding_penalties,
    get_run_knob, redact_text, resolve_temperature, secret_values, tally_batch_scores,
)
from .limit_plan import eval_limit_plan
from .sampling import parse_shard, shard_select, stratified_sample
from .sandbox import credential_free_env
from .swebench_common import (
    AGENT_INFRA, EndpointRelay, _reply_outcome, harness_errors_text, infra_required,
    empty_reply_class_reasons, mark_harness_error, record_harness_errors,
    relay_empty_reply_classes, relay_failure_reasons, scored_headline, withhold_on_error,
)

logger = logging.getLogger(__name__)

#: The relay class whose margins size the agents' client timeouts (agent_request_timeouts),
#: bound once: callers and tests that swap this module's EndpointRelay for a stand-in keep the
#: real relay's TOTAL_CAP_MARGIN_S and CLIENT_TIMEOUT_SUSPECT_S.
_RELAY_CLASS = EndpointRelay

AGENTIC = "agentic"
ONESHOT = "oneshot"
MODES = (AGENTIC, ONESHOT)

#: The mini-swe-agent config every suite runs (``minisweagent/config/extra/swebench.yaml``).
AGENT_CONFIG = "extra/swebench.yaml"
#: sha256 of that file in mini-swe-agent 1.14.0 through 1.17.5 (1.13.3 and older ship other
#: configs). A leaderboard row is comparable only when it ran this same file; the SWE-bench
#: Verified board's older mini-SWE-agent 1.x rows (1.0.0 to 1.13.3) ran other configs. A run on
#: any other config is recorded as not leaderboard-comparable.
LEADERBOARD_CONFIG_SHA256 = "a29173c493de31863aa08a41ea851589789ac777a2ed44ce94fd0bcf52db16cb"
#: Oldest mini-swe-agent this runner can drive: `model.cost_tracking` (always passed, see
#: build_agent_config) first exists in 1.15.0; 2.x changed the config layout and action format.
MIN_AGENT_VERSION = (1, 15, 0)
AGENT_REQUIREMENT = "mini-swe-agent>=1.15,<2"
#: Where that config (and the swebench-style images) keep the repository.
BUNDLED_WORKDIR = "/testbed"

#: Keep-alive for the task container. mini-swe-agent's default (2h) is sized for hosted
#: models; a local model spending minutes per turn outlives it, and every command after the
#: container exits fails with "No such container" while the agent carries on.
CONTAINER_TIMEOUT = "7d"
#: Upstream default is 120 s, which a multi-GB instance image does not pull in; its SWE-bench
#: FAQ says to raise it.
PULL_TIMEOUT_S = 3600

#: The one mid-rollout exception that is the model's own outcome: the conversation outgrew the
#: served context window (upstream records it as the exit status and moves on). Anything else
#: raised after the container started (endpoint down, docker exec failing, ...) is
#: infrastructure and is kept out of the model-scoreable set.
_MODEL_ATTRIBUTABLE = ("ContextWindowExceededError",)

# mini-swe-agent 2.x (container runner) ------------------------------------------------------
#: Local image holding one virtualenv per pinned 2.x release (mini_swe_agent_v2.Dockerfile).
AGENT_V2_IMAGE_DEFAULT = "gbench-mini-swe-agent-v2:2.4.2-2.4.6"
AGENT_V2_IMAGE_ENV = "GBENCH_MINI_SWE_AGENT_V2_IMAGE"
#: The config every 2.x SWE-bench submission runs (``mini-extra swebench`` default).
AGENT_V2_CONFIG = "benchmarks/swebench.yaml"
#: sha256 of that file in each supported release. The file changes between 2.x releases (2.0.0
#: still set temperature 0.0; 2.4.x sends none), so a leaderboard row is comparable only when it
#: ran the same release's file.
AGENT_V2_CONFIG_SHA256 = {
    "2.4.2": "229f178a07faa109e3a020c75dfe603b6a24200e1f3aebe257b97098fd3a6fee",
    "2.4.6": "9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e",
}
_AGENT_V2_DRIVER = "/opt/gbench/mini_swe_agent_v2_run.py"
_DOCKER_SOCKET = "/var/run/docker.sock"
#: The docker network of each 2.x rollout container: docker's default bridge, one of the
#: networks the cloud metadata server gate checks before the rollouts (require_rollout_network_
#: blocked), and the one the instance containers the agent starts land on. The container reaches
#: the served endpoint only through gbench's relay, which listens on that network's gateway.
AGENT_V2_NETWORK = "bridge"
#: The key the 2.x rollout container gets as OPENAI_API_KEY: a placeholder. The relay
#: (swebench_common.EndpointRelay, a thread of the gbench process) adds the endpoint's key.
AGENT_V2_KEY_PLACEHOLDER = "EMPTY"
#: The litellm and openai releases the 2.x image's virtualenvs carry (pinned by
#: gbench/docker/mini_swe_agent_v2.constraints.txt), recorded in `model_client_timeout`.
AGENT_V2_LITELLM_VERSION = "1.102.1"


# --------------------------------------------------------------------------------------------
# the agent's model client: timeout, transport retries, optional per-turn cap
# --------------------------------------------------------------------------------------------
#: Seconds an agent's model client waits beyond the relay's safety cap on a model request
#: (EndpointRelay.request_total_s: the relay's limit plus EndpointRelay.TOTAL_CAP_MARGIN_S), so
#: the relay's own answer (its error reply when it gives up on the endpoint) reaches the client
#: first. 2026-10-11 review: a client at the relay's limit plus 60 s gave up 60 s before the
#: relay's cap (limit plus 120 s), and SWE-agent resampled the turn (litellm.Timeout is retried).
AGENT_CLIENT_MARGIN_S = 60
#: Transport-level retries of an agent's model client (the OpenAI SDK's max_retries, which
#: litellm passes on). The SDK default (2) resends a request whose reply did not arrive within
#: the client timeout: a slow, honest reply is then sampled again instead of awaited. With 0 each
#: request is one HTTP attempt; the scaffold's own retry loop (SWE-agent's tenacity Retrying,
#: mini-swe-agent's retry) still runs, on an error the relay reports.
AGENT_CLIENT_RETRIES = 0
#: Run option, on by default (the user's decision of 2026-10-11): every agent request of the
#: suites in AGENT_MAX_TOKENS_SUITES that carries no max_tokens gets the run's
#: --max-output-tokens; GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0 turns it off (upstream: none sent). The
#: upstream scaffolds send none, so a run with it on is recorded as a deviation
#: (official_sampling.agent_max_tokens_from_run, or the result's agent_max_tokens_from_run that
#: evals._stamp_agent_max_tokens adds from the relay records; not comparable).
AGENT_MAX_TOKENS_FROM_RUN_ENV = "GBENCH_AGENT_MAX_TOKENS_FROM_RUN"


@dataclass(frozen=True)
class AgentTimeouts:
    """How long one agent model request may take (agent_request_timeouts)."""
    #: the relay's per-request limit (EndpointRelay timeout_s)
    relay_s: int
    #: the agent's model-client timeout: relay_cap_s + AGENT_CLIENT_MARGIN_S
    client_s: int
    #: the generation length the limit was sized for
    tokens: int
    #: how it was sized, for `model_client_timeout.source`
    source: str
    #: the relay's safety cap on one model request of that size (EndpointRelay.request_total_s:
    #: relay_s + EndpointRelay.TOTAL_CAP_MARGIN_S)
    relay_cap_s: int = 0

    def suspect_s(self) -> Tuple[float, ...]:
        """The client-side timeouts a relay counts a client close near as suspect
        (EndpointRelay client_timeout_suspect_s): its defaults (litellm's and the SDK's 600 s,
        1800 s) and this client's own timeout, so a close at client_s is counted too."""
        return tuple(_RELAY_CLASS.CLIENT_TIMEOUT_SUSPECT_S) + (float(self.client_s),)


def agent_request_timeouts(max_tokens_sent: Optional[int] = None) -> AgentTimeouts:
    """The relay and client timeouts of an agent's model requests, sized the way base sizes
    every request (base.request_timeout_s): for the max_tokens each request is sent, else for
    the served context window (the run knob max_model_len, what the server grants a request
    without max_tokens), else for base.UNCAPPED_TIMEOUT_TOKENS.

    2026-10-10 smoke run: SWE-agent's litellm 1.87 fell back to a 600 s client timeout (its
    request_timeout is the package default 6000, which completion_timeout.py replaces by
    COMPLETION_HTTP_FALLBACK_SECONDS), and the OpenAI SDK made 2 more attempts: a reply needing
    more than 600 s of decode was abandoned and sampled again, 12 times on one trimesh instance.
    Sized this way, only the relay's limit bounds a request."""
    from .base import UNCAPPED_TIMEOUT_TOKENS, request_timeout_s
    if max_tokens_sent:
        tokens = int(max_tokens_sent)
        basis = f"the max_tokens={tokens} each request is sent"
    else:
        window = get_run_knob("max_model_len")
        if window:
            tokens = int(window)
            basis = (f"the served context window ({tokens} tokens), what the server grants a "
                     "request without max_tokens")
        else:
            tokens = UNCAPPED_TIMEOUT_TOKENS
            basis = (f"base.UNCAPPED_TIMEOUT_TOKENS ({tokens}): no max_tokens is sent and the "
                     "served context window is unknown")
    relay_s = int(request_timeout_s(tokens))
    cap_s = relay_s + int(_RELAY_CLASS.TOTAL_CAP_MARGIN_S)
    client_s = client_timeout_for_relay_limit(relay_s)
    return AgentTimeouts(relay_s, client_s, tokens, (
        f"base.request_timeout_s({tokens}) = {relay_s} s (relay limit) for {basis}; the relay's "
        f"safety cap on a model request is {cap_s} s (limit plus "
        f"EndpointRelay.TOTAL_CAP_MARGIN_S={_RELAY_CLASS.TOTAL_CAP_MARGIN_S}); the client waits "
        f"{AGENT_CLIENT_MARGIN_S} s beyond it, so the relay always answers first"), cap_s)


def client_timeout_for_relay_limit(relay_limit_s: float) -> int:
    """The model-client timeout of an agent behind a relay whose per-request limit (timeout_s,
    TIMEOUT_S) is `relay_limit_s`: the relay's safety cap on a model request sized the same way
    (EndpointRelay.request_total_s, the limit plus EndpointRelay.TOTAL_CAP_MARGIN_S) plus
    AGENT_CLIENT_MARGIN_S, so a request the relay still waits for is never given up by the
    client first."""
    return int(relay_limit_s) + int(_RELAY_CLASS.TOTAL_CAP_MARGIN_S) + AGENT_CLIENT_MARGIN_S


def model_client_timeout(client: str, version: Optional[str], timeout_s: Optional[float],
                         retries: Optional[int], source: str,
                         replaces: Optional[str] = None) -> Dict[str, Any]:
    """The `model_client_timeout` record of an agent's model client: which client, its release,
    the timeout and transport retries each request runs under, and where they come from.
    `replaces` names the cap the client had before gbench set them (a hidden client cap that
    resampled long replies), when there was one."""
    rec: Dict[str, Any] = {"client": client, "version": version, "timeout_s": timeout_s,
                           "retries": retries, "source": source}
    if replaces:
        rec["hidden_cap_removed"] = True
        rec["replaces"] = replaces
    return rec


def litellm_client_kwargs(timeouts: AgentTimeouts) -> Dict[str, Any]:
    """The per-request litellm.completion arguments that carry the client timeout: `timeout`
    (litellm's CompletionTimeout.resolve takes it before any global or fallback) and
    `max_retries` AGENT_CLIENT_RETRIES (the OpenAI SDK's retries). Per request, never litellm's
    module globals, so nothing else in the process changes."""
    return {"timeout": timeouts.client_s, "max_retries": AGENT_CLIENT_RETRIES}


def _package_version(name: str) -> Optional[str]:
    """The installed release of `name` in this interpreter, None when it is not installed."""
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


#: The agent suites whose scaffold sends no max_tokens: each agent request gets the run's
#: --max-output-tokens (agent_max_tokens_from_run). The user's decision of 2026-10-11: without a
#: per-reply bound one runaway turn runs to the served window (about 36 min at 120 tok/s, and an
#: instance can have several), which no 4-hour smoke and validation run can absorb. The longest
#: useful agent reply measured on 2026-10-10 was 19,437 tokens; the one reply past 50k filled the
#: window and the instance exited on the context window anyway. mcp_bench and mcp_atlas are not
#: here: their containers reach the endpoint without a gbench relay (mcp_bench's own request
#: policy already sends the run's budget under --thinking).
AGENT_MAX_TOKENS_SUITES = frozenset({
    "swe_bench_live", "swe_bench_pro", "swe_bench_multilingual", "multi_swe_bench",
    "copilot_bench_swe", "terminal_bench", "skillsbench", "wildclawbench", "gaia2", "tau2",
    "tau3", "toolbench", "agent_dojo", "swe_lancer",
    "ui_control_osworld", "bfcl_v4_agentic", "aider_polyglot",
    "mobilegym", "mobilegym_pp"})


def agent_max_tokens_from_run(max_output_tokens: Optional[int] = None) -> Optional[int]:
    """The max_tokens every agent request is sent: the run's --max-output-tokens
    (`max_output_tokens`, else the run knob), never a fixed number. On by default (the user's
    decision of 2026-10-11, AGENT_MAX_TOKENS_SUITES); GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0 turns it
    off, and the scaffolds then send no max_tokens, as upstream. None when off or the run has no
    value."""
    if (os.environ.get(AGENT_MAX_TOKENS_FROM_RUN_ENV) or "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    value = max_output_tokens or get_run_knob("max_output_tokens")
    return int(value) if value else None


def agent_max_tokens_for_suite(suite: Optional[str] = None) -> Optional[int]:
    """agent_max_tokens_from_run for the suite that runs (`suite`, else the `suite` run knob)
    when it is one of AGENT_MAX_TOKENS_SUITES, else None."""
    name = suite or get_run_knob("suite")
    return agent_max_tokens_from_run() if name in AGENT_MAX_TOKENS_SUITES else None


def with_run_max_tokens(body_fields: Dict[str, Any],
                        max_tokens: Optional[int]) -> Dict[str, Any]:
    """A relay's `body_fields` plus max_tokens=`max_tokens` (agent_max_tokens_from_run), which
    the relay sets on every completions request; `body_fields` unchanged when it is None."""
    return dict(body_fields, max_tokens=int(max_tokens)) if max_tokens else dict(body_fields)


def agent_max_tokens_record(max_tokens: int, scaffold: str, via: str) -> Tuple[Dict[str, Any],
                                                                               str]:
    """(official_sampling.agent_max_tokens_from_run record, leaderboard_comparable_reason) of a
    run that sent `max_tokens` with every agent request (agent_max_tokens_from_run). `via` says
    how it was sent (the relay's body fields, the agent's model_kwargs)."""
    rec = {"max_tokens_sent": int(max_tokens), "source": f"{AGENT_MAX_TOKENS_FROM_RUN_ENV}=1 "
           "(the run's --max-output-tokens)", "sent_via": via,
           "upstream": f"{scaffold} sends no max_tokens"}
    return rec, (f"max_tokens={int(max_tokens)} sent with every agent request "
                 f"({AGENT_MAX_TOKENS_FROM_RUN_ENV}=1); {scaffold} sends none")


# --------------------------------------------------------------------------------------------
# mode selection
# --------------------------------------------------------------------------------------------
def _suite_slug(eval_name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(eval_name).upper()).strip("_")


def mode_env_var(eval_name: str) -> str:
    """Per-suite protocol switch, e.g. ``GBENCH_SWE_BENCH_PRO_MODE``."""
    return f"GBENCH_{_suite_slug(eval_name)}_MODE"


def traj_dir_env_var(eval_name: str) -> str:
    """Per-suite trajectory location override, e.g. ``GBENCH_SWE_BENCH_LIVE_TRAJ_DIR``."""
    return f"GBENCH_{_suite_slug(eval_name)}_TRAJ_DIR"


def agent_output_dir(eval_name: str) -> str:
    """A fresh directory for this run's trajectories and the mini-swe-agent log.

    ``GBENCH_<SUITE>_TRAJ_DIR`` if set, else ``<run results dir>/agent_trajectories`` (the
    ``GBENCH_RESULTS_DIR`` a gbench run exports), so the trajectories sit next to the result
    JSON. Outside a gbench run (no results directory) it is a new directory in the system temp
    dir."""
    explicit = (os.environ.get(traj_dir_env_var(eval_name)) or "").strip()
    run_dir = (os.environ.get("GBENCH_RESULTS_DIR") or "").strip()
    base = explicit or (os.path.join(run_dir, "agent_trajectories") if run_dir else "")
    if not base:
        return tempfile.mkdtemp(prefix=f"gbench_{eval_name}_agent_")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{eval_name}_", dir=base)


def resolve_mode(eval_name: str) -> Tuple[str, str]:
    """(mode, where it came from). Agentic unless the suite's MODE knob says ``oneshot``."""
    var = mode_env_var(eval_name)
    raw = (os.environ.get(var) or "").strip().lower()
    if not raw:
        return AGENTIC, "default"
    if raw not in MODES:
        raise ValueError(f"{var}={os.environ.get(var)!r} is not a mode; use 'agentic' "
                         "(the default) or 'oneshot'.")
    return raw, f"env:{var}"


def oneshot_reason(eval_name: str) -> str:
    return (f"one-shot mode ({mode_env_var(eval_name)}=oneshot): one diff generated from the "
            "issue text with no repository access, not the agentic protocol")


def add_noncanon(result: Dict[str, Any], reason: str, first: bool = False) -> None:
    """Record a reason the number is not leaderboard-comparable (idempotent)."""
    prev = result.get("leaderboard_comparable_reason") or ""
    if reason not in prev:
        prev = "; ".join(p for p in ((reason, prev) if first else (prev, reason)) if p)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = prev


def mark_oneshot(result: Dict[str, Any], eval_name: str) -> Dict[str, Any]:
    """Label a one-shot result so it can never be read as the agentic benchmark number."""
    result["mode"] = ONESHOT
    result["mode_source"] = f"env:{mode_env_var(eval_name)}"
    add_noncanon(result, oneshot_reason(eval_name), first=True)
    return result


# --------------------------------------------------------------------------------------------
# prerequisites and config
# --------------------------------------------------------------------------------------------
def _import_minisweagent():
    # Its import prints a banner unless told not to; gbench's log is noisy enough.
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    import minisweagent
    return minisweagent


def _version_tuple(version: Any) -> Optional[Tuple[int, int, int]]:
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def check_agent_prerequisites() -> Tuple[bool, str]:
    """mini-swe-agent >=1.15,<2 + litellm importable, and a reachable Docker daemon."""
    try:
        version = _import_minisweagent().__version__
    except ImportError:
        return False, ("Python package 'mini-swe-agent' (import name minisweagent) is not "
                       f"installed; pip install '{AGENT_REQUIREMENT}' (part of the evals extra: "
                       f"pip install -e \".[evals]\" from the gbench checkout).")
    vt = _version_tuple(version)
    if vt is None or vt < MIN_AGENT_VERSION or vt[0] >= 2:
        return False, (f"mini-swe-agent {version} is installed; the agentic SWE runner needs "
                       f"1.15 or newer within 1.x (pip install '{AGENT_REQUIREMENT}').")
    try:
        import litellm  # noqa: F401
    except ImportError:
        return False, "Python package 'litellm' is not installed (a mini-swe-agent dependency)."
    docker = os.environ.get("MSWEA_DOCKER_EXECUTABLE", "docker")
    if not shutil.which(docker):
        return False, (f"'{docker}' CLI is not found on PATH; the agent works inside each "
                       "task's Docker image.")
    try:
        if subprocess.run([docker, "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to the Docker daemon: {e}"
    return True, ""


def require_agent(eval_name: str, docs_url: str) -> None:
    """Hard-error (infra_required) when the agentic default cannot run."""
    ok, reason = check_agent_prerequisites()
    if not ok:
        raise infra_required(
            eval_name, f"{reason} The agentic protocol (the default) needs it; see the doc for "
            f"the one-shot opt-in {mode_env_var(eval_name)}=oneshot and what it does not measure.",
            docs_url)


def agent_v2_image() -> str:
    return (os.environ.get(AGENT_V2_IMAGE_ENV) or "").strip() or AGENT_V2_IMAGE_DEFAULT


def agent_v2_python(version: str) -> str:
    """Interpreter of the pinned 2.x release inside the image."""
    return f"/opt/mini-swe-agent-{version}/bin/python"


def check_agent_v2_prerequisites(version: str) -> Tuple[bool, str]:
    """Docker CLI + daemon, and the local 2.x image carrying `version`."""
    image = agent_v2_image()
    build = (f"Build the mini-swe-agent 2.x image LOCALLY (gbench never pulls it): "
             f"docker build -t {image} -f gbench/docker/mini_swe_agent_v2.Dockerfile gbench/docker")
    if version not in AGENT_V2_CONFIG_SHA256:
        return False, (f"mini-swe-agent {version} is not a release this runner knows the bundled "
                       f"config of (known: {', '.join(sorted(AGENT_V2_CONFIG_SHA256))}).")
    if not shutil.which("docker"):
        return False, "'docker' CLI is not found on PATH. " + build
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to the Docker daemon: {e}"
    if not os.path.exists(_DOCKER_SOCKET):
        return False, (f"{_DOCKER_SOCKET} does not exist; the 2.x runner mounts it so the agent "
                       "can start each instance image on the host daemon.")
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format",
                            "{{json .Config.Labels}}", image],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, f"Cannot inspect the image {image!r}: {e}"
    if r.returncode != 0:
        return False, f"the image {image!r} is not built. " + build
    try:
        labels = json.loads(r.stdout or "null") or {}
    except ValueError:
        labels = {}
    releases = str(labels.get("org.gbench.mini_swe_agent_versions") or "").split()
    if version not in releases:
        return False, (f"the image {image!r} carries mini-swe-agent {releases or 'no release'}, "
                       f"not {version}; rebuild it. " + build)
    gateway, why = bridge_gateway()
    if not gateway:
        return False, (f"{why}. The 2.x rollout containers run on docker's default bridge "
                       "network and reach the served endpoint through gbench's relay on that "
                       "network's gateway.")
    return True, ""


def bridge_gateway() -> Tuple[Optional[str], str]:
    """(the IPv4 gateway of docker's default bridge network, "") or (None, why): the address a
    2.x rollout container on AGENT_V2_NETWORK reaches this host at, where the relay listens."""
    cmd = ["docker", "network", "inspect", AGENT_V2_NETWORK, "--format",
           "{{range .IPAM.Config}}{{.Gateway}} {{end}}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        return None, f"`{' '.join(cmd[:4])}` failed: {e}"
    if r.returncode != 0:
        return None, (f"`{' '.join(cmd[:4])}` failed: "
                      f"{' '.join((r.stderr or r.stdout or '').split())[:200]}")
    for token in (r.stdout or "").split():
        try:
            if ipaddress.ip_address(token).version == 4:
                return token, ""
        except ValueError:
            continue
    return None, f"docker's {AGENT_V2_NETWORK!r} network has no IPv4 gateway"


#: Seconds the relay probe container may take (container start plus the connection attempt).
_RELAY_PROBE_DOCKER_S = 120
#: Seconds the probe waits for its TCP connection to the relay.
_RELAY_PROBE_CONNECT_S = 10


def require_relay_reachable(eval_name: str, host: str, port: int,
                            docs_url: Optional[str] = None) -> None:
    """infra_required unless a container of the 2.x image on AGENT_V2_NETWORK opens a TCP
    connection to host:port, where the relay listens (nothing is sent on it; the probe is
    container_guard's python3 connection check; --pull never, read-only, no capabilities)."""
    from .container_guard import REACHABLE, _verdict_line, python_probe_script
    docs_url = docs_url or f"docs/evals/{eval_name}.md"
    name = f"gbench-msa2-relay-probe-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "--pull", "never", "--name", name,
           "--network", AGENT_V2_NETWORK, "--read-only", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--entrypoint", "python3", agent_v2_image(),
           "-c", python_probe_script(host, int(port), timeout_s=_RELAY_PROBE_CONNECT_S)]
    where = (f"{host}:{int(port)}, where gbench's relay adds the endpoint's key outside the "
             "rollout containers (they hold only a placeholder key)")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=_RELAY_PROBE_DOCKER_S, env=credential_free_env())
    except (OSError, subprocess.SubprocessError) as e:
        if isinstance(e, subprocess.TimeoutExpired):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        raise infra_required(eval_name, f"gbench could not check that a container on docker's "
                             f"{AGENT_V2_NETWORK!r} network reaches {where} "
                             f"({type(e).__name__}).", docs_url) from None
    found = _verdict_line(proc.stdout or "")
    if found and found[0] == REACHABLE:
        return
    if found:
        raise infra_required(
            eval_name, f"a container of {agent_v2_image()} on docker's {AGENT_V2_NETWORK!r} "
            f"network cannot open a TCP connection to {where}: {' '.join(found)}. A host "
            f"firewall must let that network reach this host's address {host}.", docs_url)
    detail = " ".join(((proc.stderr or proc.stdout) or "no output").split())[-300:]
    raise infra_required(
        eval_name, f"gbench could not check that a container on docker's {AGENT_V2_NETWORK!r} "
        f"network reaches {where}: the probe (python3 in {agent_v2_image()}) gave no verdict "
        f"(docker exit {proc.returncode}: {detail}).", docs_url)


def require_agent_v2(eval_name: str, docs_url: str, version: str) -> None:
    """Hard-error (infra_required) when the 2.x agentic default cannot run."""
    ok, reason = check_agent_v2_prerequisites(version)
    if not ok:
        raise infra_required(
            eval_name, f"{reason} The agentic protocol (the default) runs mini-swe-agent "
            f"{version}; see the doc for the one-shot opt-in {mode_env_var(eval_name)}=oneshot "
            "and what it does not measure.", docs_url)


def describe_agent_v2(version: str) -> Dict[str, Any]:
    """{version, config_name, config_sha256, config} as the release inside the image reports it."""
    cmd = ["docker", "run", "--rm", "--network", "none", agent_v2_image(),
           agent_v2_python(version), _AGENT_V2_DRIVER, "--describe"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"mini-swe-agent {version} in {agent_v2_image()!r} did not describe "
                           f"its config (rc={r.returncode}): {(r.stderr or r.stdout)[-400:]}")
    desc = json.loads(r.stdout.strip().splitlines()[-1])
    if desc.get("version") != version:
        raise RuntimeError(f"{agent_v2_python(version)} runs mini-swe-agent "
                           f"{desc.get('version')}, not {version}")
    return desc


def load_agent_config() -> Tuple[Dict[str, Any], str, str]:
    """(bundled swebench config, its sha256, the mini-swe-agent version)."""
    msa = _import_minisweagent()
    import yaml
    from minisweagent.config import builtin_config_dir
    text = (builtin_config_dir / AGENT_CONFIG).read_text(encoding="utf-8")
    return yaml.safe_load(text), hashlib.sha256(text.encode("utf-8")).hexdigest(), msa.__version__


def agent_temperature(eval_name: str, suite_value: Optional[float], *, thinking: bool,
                      agent_v2: Optional[str]) -> Tuple[float, str]:
    """(temperature, source) of the agent's requests, always sent.

    base.resolve_temperature decides: GBENCH_<SUITE>_TEMPERATURE, then --temperature, else the
    maintainer's rule (1.0 with --thinking, source "thinking"; 0.0 without, source "default").
    The bundled 2.x configs send no temperature; gbench sends one anyway, by that rule."""
    return resolve_temperature(eval_name, suite_value, thinking=thinking)


def _temperature_origin(source: str) -> str:
    """How a sent temperature reads in leaderboard_comparable_reason: the maintainer's rule
    (base.resolve_temperature source "thinking" or "default"), or the operator's setting."""
    if source == "thinking":
        return ("sent by the --thinking rule (every thinking run samples the model under test "
                "at 1.0)")
    if source == "default":
        return ("sent by the no-think rule (every no-think run samples the model under test "
                "at 0.0)")
    return f"set explicitly ({source})"


def build_agent_config(base: Dict[str, Any], *, model_name: str, base_url: str,
                       temperature: float, thinking: bool, max_output_tokens: Optional[int],
                       penalties: Optional[Dict[str, float]] = None,
                       threads: int = 0,
                       client_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The bundled config with the model pointed at the served endpoint and gbench's knobs.

    `client_kwargs` (run_agentic passes litellm_client_kwargs of agent_request_timeouts) go into
    model_kwargs: the client timeout and no transport retries, per request. The endpoint's key
    is not part of it (see endpoint_api_key)."""
    cfg = copy.deepcopy(base)
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/v1"
    model = cfg.setdefault("model", {})
    # Always prefixed: litellm strips one leading provider segment, so a served id that itself
    # starts with "openai/" (openai/gpt-oss-20b) must arrive as "openai/openai/...".
    model["model_name"] = f"openai/{model_name}"
    # A served model has no price; this is mini-swe-agent's documented setting for local models.
    model["cost_tracking"] = "ignore_errors"
    kw = dict(model.get("model_kwargs") or {})
    kw["api_base"] = endpoint
    # No api_key here: mini-swe-agent serializes model_kwargs into every trajectory file
    # (save_traj: dataclasses.asdict(agent.model.config)). EndpointModel adds the key to each
    # request instead (endpoint_api_key), so it never lands in a config or a file.
    kw.pop("api_key", None)
    kw["temperature"] = temperature
    extra_body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    for name, value in (penalties or {}).items():
        # repetition_penalty is not an OpenAI parameter; drop_params would silently remove it.
        if name == "repetition_penalty":
            extra_body[name] = value
        else:
            kw[name] = value
    kw["extra_body"] = extra_body
    if max_output_tokens:
        # A per-turn cap, only when a caller asks for one: run_agentic passes None, as the
        # bundled config sets none. The model class below only clamps it to the window.
        kw["max_tokens"] = int(max_output_tokens)
    kw.update(client_kwargs or {})
    model["model_kwargs"] = kw
    env = cfg.setdefault("environment", {})
    env["environment_class"] = "docker"
    env["container_timeout"] = CONTAINER_TIMEOUT
    env["pull_timeout"] = PULL_TIMEOUT_S
    if threads > 0:
        # Same per-container thread cap the scoring harness runs under (see swe_thread_cap):
        # without it a test the agent runs can stall on OpenMP spin-waits and hit the 60 s
        # command timeout for reasons that have nothing to do with the model.
        env["env"] = {**swe_thread_cap.thread_env(threads), **(env.get("env") or {})}
    return cfg


def build_agent_v2_overrides(base: Dict[str, Any], *, model_name: str, base_url: str,
                             temperature: float, thinking: bool,
                             max_output_tokens: Optional[int],
                             penalties: Optional[Dict[str, float]] = None,
                             threads: int = 0,
                             client_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the 2.x driver merges over the release's bundled config (mini-swe-agent's own
    recursive_merge): the served endpoint and gbench's knobs, nothing else.

    `temperature` is always sent (the bundled 2.4.x config sets none; gbench's rule applies:
    1.0 with --thinking, 0.0 without, unless the operator set one); `max_output_tokens` None (what run_agentic passes) sends no per-turn cap, as the
    bundled config sets none. No API key is written here: the driver's litellm reads
    OPENAI_API_KEY from its environment, where the container has only the placeholder
    AGENT_V2_KEY_PLACEHOLDER (run_agentic points `base_url` at gbench's relay, which adds the
    endpoint's key), so no key lands in the spec or trajectory files. `client_kwargs`
    (run_agentic passes litellm_client_kwargs of agent_request_timeouts) go into model_kwargs,
    which the driver's LitellmModel passes to every litellm.completion call: the client timeout
    and no transport retries (without them litellm 1.102 falls back to a 600 s timeout with 2
    SDK retries)."""
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/v1"
    kw: Dict[str, Any] = {"api_base": endpoint}
    extra_body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    kw["temperature"] = temperature
    for name, value in (penalties or {}).items():
        if name == "repetition_penalty":
            extra_body[name] = value
        else:
            kw[name] = value
    kw["extra_body"] = extra_body
    if max_output_tokens:
        kw["max_tokens"] = int(max_output_tokens)
    kw.update(client_kwargs or {})
    env: Dict[str, Any] = {"container_timeout": CONTAINER_TIMEOUT, "pull_timeout": PULL_TIMEOUT_S}
    if threads > 0:
        base_env = ((base.get("environment") or {}).get("env") or {})
        env["env"] = {**swe_thread_cap.thread_env(threads), **base_env}
    return {
        # Same prefix rule as the 1.x runner: litellm strips one leading provider segment.
        "model": {"model_name": f"openai/{model_name}", "cost_tracking": "ignore_errors",
                  "model_kwargs": kw},
        "environment": env,
    }


def _merged(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """recursive_merge(base, overrides) as mini-swe-agent 2.x computes it (dicts merge, other
    values are replaced), for recording the effective config on the result."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merged(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# --------------------------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------------------------
@dataclass
class AgentTask:
    """One instance for the agent: what it is told, and which container it works in."""
    instance_id: str
    problem_statement: str
    image: str
    category: str
    cwd: str = BUNDLED_WORKDIR
    run_args: Optional[List[str]] = None
    startup_command: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def swebench_image(instance_id: str, namespace: str = "swebench") -> str:
    """Instance image in the swebench naming, as swebench's TestSpec.instance_image_key and
    mini-swe-agent's get_swebench_docker_image_name spell it (x86_64, ``__`` -> ``_1776_``).
    SWE-bench-Live's fork uses the same scheme under namespace ``starryzhang``."""
    iid = instance_id.replace("__", "_1776_")
    return f"docker.io/{namespace}/sweb.eval.x86_64.{iid}:latest".lower()


def swebench_task(row: Dict[str, Any], namespace: str, **meta: Any) -> AgentTask:
    """A SWE-bench-schema row: the agent sees ``problem_statement`` only, repo at /testbed."""
    iid = row["instance_id"]
    return AgentTask(
        instance_id=iid, problem_statement=row["problem_statement"],
        image=row.get("image_name") or swebench_image(iid, namespace),
        category=str(row.get("repo")), meta=dict(meta))


def empty_shard_reason(shard: Tuple[int, int], full_set_items: Optional[int] = None) -> str:
    """The withheld-score reason of a shard with no item: the wording base.run_eval_suite uses
    for an empty shard of a native suite."""
    count = f" ({full_set_items} in the full set)" if full_set_items is not None else ""
    return f"empty shard {shard[0]}/{shard[1]}: no items on this node{count}"


def mark_empty_shard(result: Dict[str, Any], shard: Tuple[int, int],
                     full_set_items: Optional[int] = None) -> Dict[str, Any]:
    """An empty shard's result: status success, no score (accuracy None with the reason), never
    a 0% next to the nodes that measured something."""
    reason = empty_shard_reason(shard, full_set_items)
    result.update(status="success", status_reason=reason, empty_shard=True, accuracy=None,
                  accuracy_withheld_reason=reason)
    if full_set_items is not None:
        result["full_set_items"] = full_set_items
    return result


def select_tasks(tasks: List[AgentTask], limit: Optional[int],
                 eval_name: str) -> Tuple[List[AgentTask], Optional[Tuple[int, int]]]:
    """--shard first (round-robin over the loader's order), then a stratified --eval-limit,
    matching what base.run_eval_suite does for single-turn suites."""
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    if shard:
        tasks = shard_select(tasks, shard)
    if limit and limit > 0:
        tasks = stratified_sample(tasks, limit, key_fn=lambda t: t.category, seed=eval_name)
    return list(tasks), shard


def task_config(config: Dict[str, Any], task: AgentTask) -> Dict[str, Any]:
    """Per-instance copy of the run config (container layout, startup hook)."""
    cfg = copy.deepcopy(config)
    env = cfg.setdefault("environment", {})
    if task.cwd != BUNDLED_WORKDIR:
        env["cwd"] = task.cwd
        agent = cfg.setdefault("agent", {})
        agent["instance_template"] = agent.get("instance_template", "").replace(
            BUNDLED_WORKDIR, task.cwd)
    if task.run_args is not None:
        env["run_args"] = list(task.run_args)
    if task.startup_command:
        cfg.setdefault("run", {})["env_startup_command"] = task.startup_command
    return cfg


# --------------------------------------------------------------------------------------------
# rollout (seams kept module-level so tests can stub the container, model and agent)
# --------------------------------------------------------------------------------------------
_MODEL_CLASS = None


def endpoint_api_key() -> str:
    """The served endpoint's key as each model request sends it: base.endpoint_api_key
    (VLLM_API_KEY, else OPENAI_API_KEY), or the placeholder "EMPTY" for an endpoint without one
    (the OpenAI client needs a value). Read at request time and passed to litellm with that
    request only, from gbench's own process, never stored in the agent config that
    mini-swe-agent writes into trajectory files."""
    from .base import endpoint_api_key as _endpoint_key
    return _endpoint_key() or "EMPTY"


def _endpoint_model_class():
    """LitellmModel that keeps prompt + max_tokens inside the served context window.

    `run_agentic` sends no max_tokens (as upstream), so for the suites this model is a plain
    LitellmModel. The rest applies only to a config built with a per-turn cap
    (`build_agent_config(max_output_tokens=...)`): vLLM rejects a request whose prompt plus
    max_tokens exceeds max_model_len, even when a shorter completion would fit, so the clamp
    (base's, the same one every single-turn suite uses) only lowers the per-turn budget to what
    the window has left.

    When the window is unknown to gbench the clamp cannot run, and the server refuses the
    turn (litellm ContextWindowExceededError). That refusal is caused by gbench's cap, not by
    the conversation, so the turn is resent the way upstream sends it (no max_tokens: the
    server grants what the window has left) and counted in `context_retry_turns`. If the
    prompt alone overflows, the resend fails the same way and ends the rollout as the model's
    outcome, as upstream records it.

    The endpoint's key goes into each request's arguments (endpoint_api_key), not into
    `config.model_kwargs`, which mini-swe-agent's save_traj writes into every trajectory
    file."""
    global _MODEL_CLASS
    if _MODEL_CLASS is None:
        _import_minisweagent()
        from minisweagent.models.litellm_model import LitellmModel

        class EndpointModel(LitellmModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.context_retry_turns = 0

            def query(self, messages, **kwargs):
                kwargs.setdefault("api_key", endpoint_api_key())
                budget = (self.config.model_kwargs or {}).get("max_tokens")
                if budget and "max_tokens" not in kwargs:
                    plain = [{"role": m["role"], "content": m["content"]} for m in messages]
                    clamped, over = clamp_to_context(plain, budget)
                    if over is None and clamped and clamped != budget:
                        kwargs["max_tokens"] = clamped
                try:
                    return super().query(messages, **kwargs)
                except Exception as e:
                    capped = budget and kwargs.get("max_tokens", budget) is not None
                    if not capped or not any(c.__name__ in _MODEL_ATTRIBUTABLE
                                             for c in type(e).__mro__):
                        raise
                # None removes the cap (litellm omits a None max_tokens); it sets no budget.
                reply = super().query(messages, **dict(kwargs, max_tokens=None))
                self.context_retry_turns += 1
                return reply

        _MODEL_CLASS = EndpointModel
    return _MODEL_CLASS


def _start_environment(config: Dict[str, Any], task: AgentTask):
    _import_minisweagent()
    from minisweagent.run.extra.swebench import get_sb_environment
    instance = {"instance_id": task.instance_id, "image_name": task.image,
                "problem_statement": task.problem_statement, **task.meta}
    return get_sb_environment(config, instance)


def _make_model(model_config: Dict[str, Any]):
    return _endpoint_model_class()(**model_config)


def _make_agent(model, env, agent_config: Dict[str, Any]):
    from minisweagent.agents.default import DefaultAgent
    return DefaultAgent(model, env, **agent_config)


def scrub_secrets_in_files(*paths: Optional[str]) -> None:
    """Replace every credential value of the environment (base.secret_values, e.g. the
    OPENAI_API_KEY value) in these files with base.REDACTED. The backstop for the files
    mini-swe-agent writes itself (trajectories, the 2.x runner's result.json), which gbench's
    result and log redaction never sees. A missing or unreadable file is left alone."""
    secrets = secret_values()
    if not secrets:
        return
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            clean = redact_text(text, secrets)
            if clean != text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(clean)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("could not check %s for credentials: %s", path, e)


def _save_traj(agent, path: str, exit_status: Optional[str], patch: str,
               tb: Optional[str], instance_id: str) -> Optional[str]:
    try:
        from pathlib import Path
        from minisweagent.run.utils.save import save_traj
        save_traj(agent, Path(path), exit_status=exit_status, result=patch,
                  extra_info={"traceback": tb} if tb else None, instance_id=instance_id,
                  print_fct=logger.debug)
        scrub_secrets_in_files(path)
        return path
    except Exception as e:     # the trace still carries the messages; the file is a copy
        logger.error("could not write the trajectory file %s: %s", path, e)
        return None


def _trajectory(agent) -> Tuple[List[Dict[str, Any]], int]:
    """(role/content messages with the reasoning channel and finish reason, truncated turns)."""
    out: List[Dict[str, Any]] = []
    truncated = 0
    for m in (getattr(agent, "messages", None) or []):
        entry: Dict[str, Any] = {"role": m.get("role"), "content": m.get("content")}
        choices = (((m.get("extra") or {}).get("response") or {}).get("choices") or [])
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning:
                entry["reasoning"] = reasoning
            finish = choices[0].get("finish_reason")
            if finish:
                entry["finish_reason"] = finish
                truncated += finish == "length"
        out.append(entry)
    return out, truncated


def empty_after_reasoning(messages: Optional[List[Dict[str, Any]]]) -> int:
    """Assistant turns of a 1.x condensed trajectory (_trajectory) that carry reasoning but
    neither content nor an action. 1.x adds every model reply as an assistant message before
    it parses the action, so the role identifies the reply. The 2.x path counts with
    v2_reply_counts instead."""
    return sum(1 for m in messages or []
               if m.get("role") == "assistant" and m.get("reasoning")
               and not str(m.get("content") or "").strip() and not m.get("actions"))


def v2_reply_counts(messages: Any) -> Optional[Dict[str, int]]:
    """Counts over the model replies of a mini-swe-agent 2.x trajectory (the `messages` of the
    `<instance_id>.traj.json` that DefaultAgent.save writes), read from each reply's raw
    response (`extra.response`), whatever role the message carrying it has.

    2.x's LitellmModel.query parses the tool calls before it builds the assistant message. A
    reply without a tool call (or with an unusable one) raises FormatError, and DefaultAgent.run
    then adds only a role "user" format-error message whose `extra.response` is that reply. Every
    reply therefore sits in exactly one message with `extra.response`: an assistant message when
    it had usable tool calls, a user message otherwise. The reply's content, reasoning
    (`reasoning_content` or `reasoning`, as litellm carries vLLM's `reasoning`) and tool calls
    are read from that response, never from the message around it.

    Returns {"model_replies", "truncated_turns" (`finish_reason` "length"),
    "empty_after_reasoning_turns" (stopped on its own with reasoning but neither content nor a
    tool call: the answer inside the thinking channel; a reply cut off at a limit is counted in
    truncated_turns only, the rule swebench_common's relay applies)}, or None when `messages` is not a list or a reply's
    response cannot be read (upstream stores repr(response) when it cannot serialize one)."""
    if not isinstance(messages, list):
        return None
    replies = cut = empty = 0
    for m in messages:
        extra = m.get("extra") if isinstance(m, dict) else None
        if not isinstance(extra, dict) or "response" not in extra:
            continue
        response = extra["response"]
        choices = response.get("choices") if isinstance(response, dict) else None
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)
                and isinstance(choices[0].get("message"), dict)):
            return None
        msg = choices[0]["message"]
        psf = msg.get("provider_specific_fields")
        psf = psf if isinstance(psf, dict) else {}
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning")
                     or psf.get("reasoning_content") or psf.get("reasoning"))
        is_cut, is_empty = _reply_outcome(choices[0].get("finish_reason"), msg.get("content"),
                                          reasoning, msg.get("tool_calls"))
        replies += 1
        cut += int(is_cut)
        empty += int(is_empty and not is_cut)
    return {"model_replies": replies, "truncated_turns": cut,
            "empty_after_reasoning_turns": empty}


#: The observation a mini-swe-agent 2.x turn shows when the environment's BASH_ENV
#: (/root/.bashrc) activates a conda that the official image does not ship: bash prints this for
#: every command, the upstream submit command included, so such a turn reads as a format error.
_BASH_ENV_CONDA_MISSING = re.compile(r"/opt/miniconda3[^\n]*No such file or directory")


def bash_env_conda_turns(messages: Any) -> int:
    """Turns of a mini-swe-agent 2.x trajectory whose observation (a user or tool message after
    a model reply) shows the BASH_ENV .bashrc activating a missing /opt/miniconda3: an upstream
    scaffold interaction (canonical, kept as it is) that can hide the submit command, so the
    zeros it causes look like ordinary format errors. Counted, never a score change."""
    if not isinstance(messages, list):
        return 0
    n = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "tool"):
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        if _BASH_ENV_CONDA_MISSING.search(text or ""):
            n += 1
    return n


def read_v2_trajectory(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """v2_reply_counts of the trajectory file at `path`, plus "api_calls" (the file's
    `info.model_stats.api_calls`, or None); None when there is no such file or it cannot be
    read."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.warning("could not read the trajectory file %s: %s", path, e)
        return None
    counts = v2_reply_counts(data.get("messages") if isinstance(data, dict) else None)
    if counts is None:
        return None
    counts["bash_env_conda_turns"] = bash_env_conda_turns(data.get("messages"))
    info = data.get("info")
    stats = info.get("model_stats") if isinstance(info, dict) else None
    calls = stats.get("api_calls") if isinstance(stats, dict) else None
    counts["api_calls"] = calls if isinstance(calls, int) and not isinstance(calls, bool) \
        else None
    return counts


def _rollout(idx: int, task: AgentTask, config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Run the agent on one instance; never raises (a failure is recorded on the trace)."""
    cfg = task_config(config, task)
    started = time.time()
    env = agent = None
    phase = "environment"
    exit_status: Optional[str] = None
    patch, error, tb = "", None, None
    infra = False
    try:
        env = _start_environment(cfg, task)
        phase = "agent"
        agent = _make_agent(_make_model(cfg["model"]), env, cfg.get("agent", {}))
        # Submitted -> the diff; LimitsExceeded -> "" (upstream records the same).
        exit_status, patch = agent.run(task.problem_statement)
        patch = patch or ""
    except Exception as e:
        exit_status, error, tb = type(e).__name__, str(e), traceback.format_exc()
        infra = phase == "environment" or exit_status not in _MODEL_ATTRIBUTABLE
        logger.error("[agent] %s: %s during %s: %s", task.instance_id, exit_status, phase,
                     error[:300])
    finally:
        traj_file = _save_traj(agent, os.path.join(out_dir, task.instance_id,
                                                   f"{task.instance_id}.traj.json"),
                               exit_status, patch, tb, task.instance_id)
        if env is not None and hasattr(env, "cleanup"):
            try:
                env.cleanup()
            except Exception as e:
                logger.warning("[agent] %s: container cleanup failed: %s", task.instance_id, e)
    messages, truncated = _trajectory(agent)
    model = getattr(agent, "model", None)
    return {
        "sample_idx": idx,
        "source_sample_idx": idx,
        "attempt_idx": 0,
        "category": task.category,
        "messages": [{"role": "user", "content": task.problem_statement}],
        "extra_payload": {"instance_id": task.instance_id, **task.meta},
        "gold_answer": "",
        "response_text": patch,
        # The raw submission. Scorers read this instead of re-extracting a diff from prose.
        "agent_patch": patch,
        "agent_exit_status": exit_status,
        "agent_error": error,
        "agent_failed_phase": phase if error else None,
        "agent_infra_error": infra,
        "agent_image": task.image,
        "agent_steps": int(getattr(model, "n_calls", 0) or 0),
        "trajectory": messages,
        "trajectory_file": traj_file,
        "truncated_turns": truncated,
        "empty_after_reasoning_turns": empty_after_reasoning(messages),
        "context_retry_turns": int(getattr(model, "context_retry_turns", 0) or 0),
        "duration_s": round(time.time() - started, 2),
        "is_correct": False,
        "status": "AGENT_INFRA_ERROR" if infra else "OK",
    }


def _v2_container_cmd(image: str, version: str, out_dir: str, spec_path: str,
                      name: str) -> List[str]:
    """`docker run` of one 2.x rollout: docker's default bridge network (AGENT_V2_NETWORK, checked
    by the cloud metadata server gate before the rollouts; the spec's api_base is gbench's relay
    on that network's gateway), OPENAI_API_KEY the placeholder AGENT_V2_KEY_PLACEHOLDER (no
    credential enters the container), the output directory at the same path, and the caller's
    uid so the files it writes stay the caller's.

    The docker socket is mounted because upstream's design needs a docker daemon: 2.x's
    DockerEnvironment starts the instance image with `docker run` and runs each of the model's
    commands with `docker exec` (an argv list, no shell on this side), so the model's commands
    run in the instance container, which gets no socket. The socket gives the process that turns
    the model's tool calls into those `docker exec` calls root-equivalent access to the host's
    daemon: a flaw in mini-swe-agent, litellm or the driver that let model output run in this
    container would reach it."""
    cmd = ["docker", "run", "--rm", "--name", name, "--network", AGENT_V2_NETWORK,
           "-v", f"{_DOCKER_SOCKET}:{_DOCKER_SOCKET}", "-v", f"{out_dir}:{out_dir}:rw",
           "-e", "HOME=/tmp", "-e", "MSWEA_SILENT_STARTUP=1",
           "-e", f"OPENAI_API_KEY={AGENT_V2_KEY_PLACEHOLDER}"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        try:
            cmd += ["--group-add", str(os.stat(_DOCKER_SOCKET).st_gid)]
        except OSError:
            pass
    return cmd + [image, agent_v2_python(version), _AGENT_V2_DRIVER, "--spec", spec_path]


def _rollout_v2(idx: int, task: AgentTask, overrides: Dict[str, Any], out_dir: str, *,
                image: str, version: str) -> Dict[str, Any]:
    """Run one instance with mini-swe-agent 2.x in its container; never raises.

    The instance image is pulled on the host first (swebench_common.host_pull_images, the
    operator's Docker login): the 2.x container has no Docker client configuration, so the
    `docker run` its DockerEnvironment makes finds the image present. An image that cannot be
    pulled is an infrastructure failure of that instance, and no container starts."""
    from .swebench_common import host_pull_images
    started = time.time()
    inst_dir = os.path.join(out_dir, task.instance_id)
    os.makedirs(inst_dir, exist_ok=True)
    result_path = os.path.join(inst_dir, "result.json")
    spec_path = os.path.join(inst_dir, "spec.json")
    traj_path = os.path.join(inst_dir, f"{task.instance_id}.traj.json")
    # The driver resends a turn (context_retry_turns) only under a per-turn max_tokens.
    capped = bool(((overrides.get("model") or {}).get("model_kwargs") or {}).get("max_tokens"))
    pull = host_pull_images([task.image]).get(task.image) or {}
    if pull.get("state") == "failed":
        # No container started, so no model call was made: every count is a measured zero.
        return _v2_trace(idx, task, started, {
            "exit_status": "ImagePullFailed", "submission": "", "infra_error": True,
            "failed_phase": "environment", "error": f"instance image: {pull.get('error')}",
            "n_calls": 0, "truncated_turns": 0, "context_retry_turns": 0})
    instance = {"instance_id": task.instance_id, "image_name": task.image,
                "problem_statement": task.problem_statement, **task.meta}
    spec = {"instance": instance, "overrides": overrides, "output_dir": out_dir,
            "expected_version": version}
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    name = f"gbench-msa2-{uuid.uuid4().hex[:12]}"
    # The docker CLI gets no credential variable either: the container's key is the placeholder
    # on the command line, and the relay adds the endpoint's key.
    env = credential_free_env()
    proc_tail = ""
    try:
        proc = subprocess.run(_v2_container_cmd(image, version, out_dir, spec_path, name),
                              capture_output=True, text=True, env=env)
        proc_tail = ((proc.stderr or "") + (proc.stdout or ""))[-1500:]
    except Exception as e:
        proc_tail = f"{type(e).__name__}: {e}"
    finally:
        # The driver removes the task container itself; if it died first, remove it by id.
        cid_file = os.path.join(inst_dir, "container_id")
        try:
            with open(cid_file, encoding="utf-8") as f:
                cid = f.read().strip()
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=180)
        except OSError:
            pass
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    scrub_secrets_in_files(result_path, traj_path)
    try:
        with open(result_path, encoding="utf-8") as f:
            res = json.load(f)
    except (OSError, ValueError):
        res = {"exit_status": "RunnerError", "submission": "", "infra_error": True,
               "failed_phase": "environment",
               "error": f"the mini-swe-agent {version} container wrote no result: {proc_tail}"}
        if not capped:
            res["context_retry_turns"] = 0
    return _v2_trace(idx, task, started, res, traj_path)


def _v2_counts(res: Dict[str, Any], traj_path: Optional[str]) -> Dict[str, Optional[int]]:
    """The per-instance counts of a 2.x rollout; None for one that could not be measured.

    The driver's result.json carries `n_calls`, `truncated_turns` and `context_retry_turns`
    (read in the container from the agent and its model). `empty_after_reasoning_turns` comes
    from the replies in the trajectory file (v2_reply_counts). A stand-in for a container that
    wrote no result has none of the driver's counts: the trajectory file supplies the steps
    (its `api_calls`) and the cut-off turns when it can be read, and `context_retry_turns` is
    zero only when no per-turn cap was sent (the caller sets it then)."""
    traj = read_v2_trajectory(traj_path)
    steps = res.get("n_calls")
    truncated = res.get("truncated_turns")
    if traj is not None:
        empty: Optional[int] = traj["empty_after_reasoning_turns"]
        steps = traj["api_calls"] if steps is None else steps
        truncated = traj["truncated_turns"] if truncated is None else truncated
    else:
        # No readable trajectory: nothing to count only when no model call was made.
        empty = 0 if steps == 0 else None
    counts = {"agent_steps": steps, "truncated_turns": truncated,
              "empty_after_reasoning_turns": empty,
              "context_retry_turns": res.get("context_retry_turns")}
    out = {k: None if v is None else int(v) for k, v in counts.items()}
    # informational, not a counter the run needs measured
    out["bash_env_conda_turns"] = int((traj or {}).get("bash_env_conda_turns") or 0)
    return out


def _empty_turn_classes(result: Dict[str, Any], traces: List[Dict[str, Any]],
                        relay_record: Optional[Dict[str, Any]]) -> List[str]:
    """Classify the run's model turns that stopped on their own with reasoning but no content
    and no action, and set `answer_in_thinking` / `answer_in_thinking_of` /
    `answer_in_thinking_rate`, `tool_call_dropped_turns` and `unclassified_empty_turns`; returns
    the non-comparability reasons (swebench_common.empty_reply_class_reasons).

    Telling the answer inside the thinking channel (the model's outcome, base's
    ANSWER_IN_THINKING_POLICY) from a native tool call the server's tool-call parser removed (a
    server-parser artifact) takes the reply's token ids, which only the relay sees (the
    trajectory keeps the parsed reply): with a relay (2.x) the classes and their denominator
    (`completion_replies`, the replies the relay counted them over) are the relay's
    (relay_empty_reply_classes); empty turns in the trajectories beyond the ones the relay
    classified are unclassified. Without one (the in-process 1.x runner) no turn can be told
    apart: every such turn is unclassified, never counted as the model's answer, over the model
    replies of the instances whose count was measured."""
    if relay_record is not None:
        classes = relay_empty_reply_classes(relay_record)
        of = int(relay_record.get("completion_replies") or 0)
        # empty turns the trajectories hold beyond what the relay classified (a reply the relay
        # did not see whole cannot be told apart): unclassified
        seen = sum(classes.values())
        in_trajs = sum(int(t.get("empty_after_reasoning_turns") or 0) for t in traces)
        if in_trajs > seen:
            classes["unclassified"] += in_trajs - seen
    else:
        measured = [t for t in traces if t.get("empty_after_reasoning_turns") is not None]
        classes = {"answer_in_thinking": 0, "tool_call_dropped": 0,
                   "unclassified": sum(int(t["empty_after_reasoning_turns"]) for t in measured)}
        of = sum(int(t.get("model_replies") if t.get("model_replies") is not None
                     else t.get("agent_steps") or 0) for t in measured)
    result.update(answer_in_thinking_fields(classes["answer_in_thinking"], of))
    result["tool_call_dropped_turns"] = classes["tool_call_dropped"]
    result["unclassified_empty_turns"] = classes["unclassified"]
    return empty_reply_class_reasons(classes, "mini-swe-agent")


def _v2_trace(idx: int, task: AgentTask, started: float, res: Dict[str, Any],
              traj_path: Optional[str] = None) -> Dict[str, Any]:
    """One instance's trace from the 2.x driver's result.json (or a stand-in for one) and the
    instance's trajectory file. A count that could not be measured is None on the trace and
    named in `unmeasured_counters`."""
    infra = bool(res.get("infra_error"))
    error = res.get("error")
    patch = res.get("submission") or ""
    counts = _v2_counts(res, traj_path)
    traj_file = res.get("trajectory_file") or (
        traj_path if traj_path and os.path.isfile(traj_path) else None)
    return {
        "sample_idx": idx,
        "source_sample_idx": idx,
        "attempt_idx": 0,
        "category": task.category,
        "messages": [{"role": "user", "content": task.problem_statement}],
        "extra_payload": {"instance_id": task.instance_id, **task.meta},
        "gold_answer": "",
        "response_text": patch,
        "agent_patch": patch,
        "agent_exit_status": res.get("exit_status"),
        "agent_error": error,
        "agent_failed_phase": res.get("failed_phase") if error else None,
        "agent_infra_error": infra,
        "agent_image": task.image,
        "agent_steps": counts["agent_steps"],
        "trajectory": res.get("trajectory") or [],
        "trajectory_file": traj_file,
        "truncated_turns": counts["truncated_turns"],
        "empty_after_reasoning_turns": counts["empty_after_reasoning_turns"],
        "context_retry_turns": counts["context_retry_turns"],
        "bash_env_conda_turns": counts["bash_env_conda_turns"],
        "unmeasured_counters": [k for k, v in counts.items() if v is None],
        "agent_config_sha256": res.get("config_sha256"),
        "duration_s": round(time.time() - started, 2),
        "is_correct": False,
        "status": "AGENT_INFRA_ERROR" if infra else "OK",
    }


def run_args_networks(run_args: Optional[List[str]]) -> List[str]:
    """The docker networks a task's `run_args` name (`--network X`, `--network=X`, `--net X`,
    `--net=X`); none when they name none (the container then lands on docker's default
    bridge)."""
    out: List[str] = []
    args = list(run_args or [])
    for i, arg in enumerate(args):
        for flag in ("--network", "--net"):
            if arg == flag and i + 1 < len(args):
                out.append(args[i + 1])
            elif arg.startswith(flag + "="):
                out.append(arg.split("=", 1)[1])
    return out


def require_rollout_network_blocked(eval_name: str, tasks: List[AgentTask],
                                    agent_v2: Optional[str] = None) -> None:
    """The cloud metadata server gate before the rollouts (swebench_common.
    require_task_network_blocked). mini-swe-agent (1.x in this process, 2.x in its container)
    starts each task's container with `docker run` and no `--network` unless the task's
    `run_args` name one, so on docker's default bridge, and the model's commands run there; each
    2.x rollout container runs on that bridge too (AGENT_V2_NETWORK). The
    probe runs in the 2.x image, or for 1.x in a task image already on the host; nothing is
    pulled for it, and without one the call is infra_required naming `docker pull` of the first
    task's image."""
    from . import swebench_common
    networks = list(swebench_common.TASK_NETWORKS)
    for t in tasks:
        networks.extend(run_args_networks(t.run_args))
    if agent_v2:
        swebench_common.require_task_network_blocked(
            eval_name, [agent_v2_image()], networks=networks,
            hint=f"Build the mini-swe-agent 2.x image {agent_v2_image()} (see the suite's doc).")
        return
    images = [t.image for t in tasks]
    swebench_common.require_task_network_blocked(eval_name, images, networks=networks,
                                                 pull=images[0] if images else None)


def run_rollouts(tasks: List[AgentTask], config: Dict[str, Any], concurrency: int,
                 out_dir: str, eval_name: str = "swe",
                 rollout: Optional[Callable[..., Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Roll out every task on `concurrency` worker threads (upstream's batch model).
    `rollout` defaults to the in-process 1.x `_rollout`."""
    rollout = rollout or _rollout
    traces: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(concurrency or 1))) as ex:
        futures = {ex.submit(rollout, i, t, config, out_dir): i for i, t in enumerate(tasks)}
        for fut in concurrent.futures.as_completed(futures):
            tr = fut.result()
            traces[futures[fut]] = tr
            done += 1
            logger.info("[%s] agent %d/%d: %s -> %s (%s steps)", eval_name, done, len(tasks),
                        tr["extra_payload"]["instance_id"], tr["agent_exit_status"],
                        tr["agent_steps"])
    return [t for t in traces if t is not None]


# --------------------------------------------------------------------------------------------
# the suite-facing entry point
# --------------------------------------------------------------------------------------------
Scorer = Callable[[List[Dict[str, Any]]], Any]
Finalizer = Callable[[Dict[str, Any], Dict[str, Any]], List[str]]


def run_agentic(eval_name: str, model_name: str, base_url: str, concurrency: int,
                tasks: List[AgentTask], scorer: Scorer, metrics: Dict[str, Any], *,
                enable_thinking: bool = False, leaderboard_note: Optional[str] = None,
                finalize: Optional[Finalizer] = None, agent_v2: Optional[str] = None,
                **kwargs: Any) -> Dict[str, Any]:
    """Roll out the agent on `tasks`, score the patches with the suite's harness, and build
    the result.

    `scorer` is the suite's existing async batch scorer; it sees only the traces whose rollout
    actually ran. `finalize(result, metrics)` applies the suite's harness-report checks and
    returns extra non-comparability reasons. `leaderboard_note` states why this scaffold's
    number cannot sit next to the suite's published leaderboard, when that is so.

    `agent_v2` (a release, e.g. "2.4.6") runs that mini-swe-agent 2.x release in the local 2.x
    image with its bundled benchmarks/swebench.yaml instead of the in-process 1.x runner; the
    caller has checked `require_agent_v2`.
    """
    start = time.time()
    limit = kwargs.get("limit")
    full_set_items = len(tasks)
    tasks, shard = select_tasks(list(tasks), limit, eval_name)
    # One instance set, no settings: --eval-limit N rolls out N instances in total.
    plan = eval_limit_plan(eval_name, limit, len(tasks), attempts=1, unit="rollouts")
    temperature, temperature_source = agent_temperature(
        eval_name, kwargs.get("temperature"), thinking=enable_thinking, agent_v2=agent_v2)
    # The bundled configs send no per-turn max_tokens (1.x extra/swebench.yaml sets only
    # temperature and drop_params; 2.x benchmarks/swebench.yaml sets none), so neither does
    # gbench: the operator's --max-output-tokens is recorded as not applied.
    requested_max_tokens = kwargs.get("max_output_tokens") or get_run_knob("max_output_tokens")
    # GBENCH_AGENT_MAX_TOKENS_FROM_RUN (default on; =0 turns it off) sends the run's --max-output-tokens as a
    # per-turn cap, in model_kwargs, where the 2.x driver's EndpointModel (and the 1.x one here)
    # resends a turn the server refused only because of the cap the way upstream sends it.
    run_max_tokens = agent_max_tokens_from_run(kwargs.get("max_output_tokens"))
    # The client timeout and no transport retries (agent_request_timeouts), per request. Sized
    # for the served window even when the option sends a per-turn cap: EndpointModel resends a
    # turn the server refused because of the cap with max_tokens=None (the window), under the
    # same client timeout (2026-10-11 review: sized for a 4096-token cap, the client gave such a
    # resent turn 1772 s while the relay allowed it the window's 34088 s).
    timeouts = agent_request_timeouts(None)
    client_kwargs = litellm_client_kwargs(timeouts)
    penalties = decoding_penalties()
    threads = swe_thread_cap.resolve_threads(concurrency)
    rollout: Optional[Callable[..., Dict[str, Any]]] = None
    if agent_v2:
        desc = describe_agent_v2(agent_v2)
        base_cfg, cfg_sha, version = desc["config"], desc["config_sha256"], desc["version"]
        run_cfg = build_agent_v2_overrides(base_cfg, model_name=model_name, base_url=base_url,
                                           temperature=temperature, thinking=enable_thinking,
                                           max_output_tokens=run_max_tokens,
                                           penalties=penalties, threads=threads,
                                           client_kwargs=client_kwargs)
        config = _merged(base_cfg, run_cfg)
        config_name, want_sha = AGENT_V2_CONFIG, AGENT_V2_CONFIG_SHA256.get(version)
        rollout = functools.partial(_rollout_v2, image=agent_v2_image(), version=version)
    else:
        base_cfg, cfg_sha, version = load_agent_config()
        config = build_agent_config(base_cfg, model_name=model_name, base_url=base_url,
                                    temperature=temperature, thinking=enable_thinking,
                                    max_output_tokens=run_max_tokens, penalties=penalties,
                                    threads=threads, client_kwargs=client_kwargs)
        run_cfg = config
        config_name, want_sha = AGENT_CONFIG, LEADERBOARD_CONFIG_SHA256
    agent_cfg = config.get("agent") or {}
    env_cfg = config.get("environment") or {}
    out_dir = agent_output_dir(eval_name)
    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "thinking": enable_thinking,
        "mode": AGENTIC,
        "mode_source": resolve_mode(eval_name)[1],
        "scoring_mode": "execution",
        "temperature": temperature,
        "temperature_source": temperature_source,
        "max_output_tokens_per_turn": run_max_tokens,
        "max_output_tokens_note": (
            f"--max-output-tokens={run_max_tokens} is sent with every turn "
            f"({AGENT_MAX_TOKENS_FROM_RUN_ENV}=1); the bundled mini-swe-agent {version} config "
            "sends no per-turn max_tokens" if run_max_tokens else
            f"--max-output-tokens={requested_max_tokens} is not applied: the bundled "
            f"mini-swe-agent {version} config sends no per-turn max_tokens (the server grants "
            "each turn what the context window has left)" if requested_max_tokens else
            f"the bundled mini-swe-agent {version} config sends no per-turn max_tokens"),
        "model_client_timeout": model_client_timeout(
            f"litellm (mini-swe-agent {version} LitellmModel, model_kwargs)",
            AGENT_V2_LITELLM_VERSION if agent_v2 else _package_version("litellm"),
            timeouts.client_s, AGENT_CLIENT_RETRIES,
            timeouts.source + ("; the relay's limit is the same value" if agent_v2 else
                               "; no relay (the in-process client calls the endpoint)")
            + ("; sized for the window, not the per-turn cap: a turn the server refuses "
               "because of the cap is resent with no max_tokens" if run_max_tokens else ""),
            replaces=("litellm's fallback client timeout of 600 s "
                      "(COMPLETION_HTTP_FALLBACK_SECONDS) with 2 OpenAI SDK retries, which "
                      "abandoned and resampled any turn longer than 600 s")),
        "decoding_penalties": penalties or None,
        "agent": {
            "scaffold": "mini-swe-agent",
            "version": version,
            "config": config_name,
            "config_sha256": cfg_sha,
            "step_limit": agent_cfg.get("step_limit"),
            "cost_limit": agent_cfg.get("cost_limit"),
            "command_timeout_s": env_cfg.get("timeout"),
            "model": config["model"]["model_name"],
            "api_base": config["model"]["model_kwargs"]["api_base"],
            "container_thread_cap": threads or None,
            "runner": (f"container {agent_v2_image()} ({agent_v2_python(version)})"
                       if agent_v2 else "in-process"),
            "relay_timeout_s": timeouts.relay_s if agent_v2 else None,
        },
        "agent_output_dir": out_dir,
        "eval_limit_plan": plan,
    }
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    if attempts > 1:
        # The harness keys predictions by instance id, so k attempts would collapse into one.
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "one rollout per instance: the harness scores one patch per "
                           "instance id, so an @k would be fabricated"}
    if shard:
        result["shard"] = f"{shard[0]}/{shard[1]}"

    noncanon: List[str] = []
    if leaderboard_note:
        noncanon.append(leaderboard_note)
    if cfg_sha != want_sha:
        if agent_v2:
            noncanon.append(f"mini-swe-agent {version} ships {config_name} with sha256 "
                            f"{cfg_sha[:12]}..., not the {str(want_sha)[:12]}... file of the "
                            f"leaderboard's mini-SWE-agent {version} rows")
        else:
            noncanon.append(f"mini-swe-agent {version} ships {config_name} with sha256 "
                            f"{cfg_sha[:12]}..., not the {want_sha[:12]}... config the "
                            "leaderboard's comparable mini-SWE-agent 1.x rows ran (1.14-1.17.x; "
                            "older 1.x rows used other configs)")
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    if shard:
        noncanon.append(f"shard {shard[0]}/{shard[1]} of the instance set")
    if agent_v2:
        noncanon.append(f"temperature={temperature} {_temperature_origin(temperature_source)}; "
                        f"the bundled mini-swe-agent {version} config sends none")
    elif temperature not in (0.0, 0):
        noncanon.append(f"non-greedy temperature={temperature}"
                        + (" (--thinking samples the model under test at 1.0)"
                           if temperature_source == "thinking" else "")
                        + " (mini-SWE-agent 1.x leaderboard runs use 0.0)")
    if penalties:
        noncanon.append(f"decoding penalties {penalties}")
    if run_max_tokens:
        rec, why = agent_max_tokens_record(run_max_tokens, f"the bundled mini-swe-agent "
                                           f"{version} config", "model_kwargs.max_tokens")
        result["official_sampling"] = {"agent_max_tokens_from_run": rec}
        noncanon.append(why)

    if not tasks:
        result.update(total_questions=0, correct_answers=0, failed_requests=0, accuracy=None,
                      category_accuracy={}, sample_traces=[])
        if shard:
            # An empty shard (N > instance count) is an honest empty subset, not an error, and
            # not a 0% score either: nothing was measured on this node.
            mark_empty_shard(result, shard, full_set_items)
        else:
            result["status"] = "error"
            result["error"] = "no instances were selected, so nothing was evaluated"
            withhold_on_error(result)
        add_noncanon(result, "; ".join(noncanon + ["no instances evaluated"]))
        result["duration_s"] = round(time.time() - start, 2)
        return result

    # The model's commands run in each task's container, with network access: the cloud metadata
    # server must be blocked there before the first one starts (infra_required otherwise).
    require_rollout_network_blocked(eval_name, tasks, agent_v2)

    relay: Optional[EndpointRelay] = None
    if agent_v2:
        # The 2.x containers run on docker's default bridge with the placeholder key: they reach
        # the served endpoint only through this relay, on that network's gateway, which adds the
        # endpoint's key to each request (no credential enters a container).
        docs = f"docs/evals/{eval_name}.md"
        gateway, why = bridge_gateway()
        if not gateway:
            raise infra_required(eval_name, f"{why}; the mini-swe-agent {version} rollout "
                                 "containers reach the served endpoint through gbench's relay "
                                 "on docker's default bridge gateway.", docs)
        from .base import endpoint_api_key as _endpoint_key
        key = _endpoint_key()     # VLLM_API_KEY, else OPENAI_API_KEY; a placeholder is no key
        relay = EndpointRelay(base_url, gateway,
                              key=key if key and key != AGENT_V2_KEY_PLACEHOLDER else None,
                              name="mini-swe-agent-v2-relay", timeout_s=timeouts.relay_s,
                              client_timeout_suspect_s=timeouts.suspect_s())
        try:
            relay.start()
        except OSError as e:
            raise infra_required(eval_name, f"gbench could not listen on docker's default bridge "
                                 f"gateway {gateway} ({type(e).__name__}: {e}), where the relay "
                                 "adds the endpoint's key outside the rollout containers.",
                                 docs) from None
        try:
            require_relay_reachable(eval_name, gateway, int(relay.url.rsplit(":", 1)[1]
                                                           .split("/")[0]), docs)
            run_cfg = build_agent_v2_overrides(base_cfg, model_name=model_name,
                                               base_url=relay.url, temperature=temperature,
                                               thinking=enable_thinking,
                                               max_output_tokens=run_max_tokens,
                                               penalties=penalties, threads=threads,
                                               client_kwargs=client_kwargs)
        except BaseException:
            relay.stop()
            raise
        result["agent"]["api_base"] = run_cfg["model"]["model_kwargs"]["api_base"]

    mlog = logging.getLogger("minisweagent")
    handler = logging.FileHandler(os.path.join(out_dir, "minisweagent.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handler.addFilter(SecretRedactingFilter())      # no credential value reaches this log
    mlog.addHandler(handler)
    try:
        logger.info("[%s] mini-swe-agent %s rollouts: %d instance(s), %d worker(s), "
                    "trajectories in %s", eval_name, version, len(tasks), concurrency, out_dir)
        traces = run_rollouts(tasks, run_cfg, concurrency, out_dir, eval_name, rollout=rollout)
    finally:
        mlog.removeHandler(handler)
        handler.close()
        if relay is not None:
            relay.stop()

    # A rollout that could not run is a harness error: out of the numerator and the denominator.
    for t in traces:
        if t["agent_infra_error"]:
            mark_harness_error(t, AGENT_INFRA, t.get("agent_error") or t.get("agent_exit_status"))
    scoreable = [t for t in traces if not t["agent_infra_error"]]
    if scoreable:
        asyncio.run(scorer(scoreable))
    result.update(metrics)

    total = len(traces)
    infra = total - len(scoreable)
    head = scored_headline(traces)
    result.update({
        "total_questions": total,
        "correct_answers": head["correct_answers"],
        # the resolved rate's denominator: the instances that were scored
        "scored_instances": head["scored_instances"],
        "scoring_excluded": head["scoring_excluded"],
        # every harness error (the agent could not run, the harness gave no verdict or wrote
        # no report), counted and named, never scored
        "failed_requests": head["scoring_excluded"],
        "accuracy": head["accuracy"],
        "category_accuracy": head["category_accuracy"],
        "agent_exit_statuses": dict(collections.Counter(
            str(t["agent_exit_status"]) for t in traces)),
        "agent_steps_total": sum(int(t.get("agent_steps") or 0) for t in traces),
        "sample_traces": traces,
        "status": "success",
    })
    record_harness_errors(result, head)
    # Turns that ended with finish_reason "length" (at the run's --max-output-tokens, sent as
    # max_tokens in the agent's model_kwargs; at the end of the served context window with
    # GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0, as the bundled config sends none): counted, and they make
    # the run not comparable (the token-limit policy: a reply cut off by the budget is never a
    # clean measurement). Turns with reasoning but no content and no action
    # (empty_after_reasoning_turns, counted from the trajectories) are classified below, after
    # the rollouts, from the relay's token ids (_empty_turn_classes).
    result["truncated_turns"] = sum(int(t.get("truncated_turns") or 0) for t in traces)
    result["empty_after_reasoning_turns"] = sum(int(t.get("empty_after_reasoning_turns") or 0)
                                                for t in traces)
    if result["truncated_turns"]:
        noncanon.append(
            f"{result['truncated_turns']} model turn(s) cut off at max_tokens={run_max_tokens} "
            "(finish_reason length: the run's --max-output-tokens, sent in the agent's "
            "model_kwargs, as the bundled config sends none)" if run_max_tokens else
            f"{result['truncated_turns']} model turn(s) cut off at the end of the served context "
            "window (finish_reason length, as the bundled config sends no max_tokens)")
    # Turns resent without a per-turn cap after the server refused prompt + cap (only possible
    # with a capped config; informational).
    result["context_retry_turns"] = sum(int(t.get("context_retry_turns") or 0) for t in traces)
    # Instances with a turn whose observation shows the BASH_ENV .bashrc activating a missing
    # /opt/miniconda3 (an upstream scaffold interaction that can hide the submit command;
    # informational, no score change).
    result["submit_hidden_by_bash_env"] = sum(
        1 for t in traces if int(t.get("bash_env_conda_turns") or 0))
    # A count a rollout could not measure (a 2.x container that wrote no result, or a
    # trajectory file that cannot be read) is None on its trace, never a zero. The totals above
    # cover the other instances, and the run is not comparable: a cut-off or empty turn there
    # would go unseen.
    unmeasured = collections.Counter(c for t in traces for c in t.get("unmeasured_counters") or ())
    if unmeasured:
        result["unmeasured_counters"] = dict(sorted(unmeasured.items()))
        noncanon.append("counts not measured on some instance(s) (runner result or trajectory "
                        "file missing or unreadable): " + ", ".join(
                            f"{c} on {n}" for c, n in sorted(unmeasured.items())))
    if infra == total:
        first = next((t for t in traces if t.get("agent_error")), {})
        result["status"] = "error"
        result["error"] = (
            f"the agent could not run on any of {total} instance(s) (e.g. "
            f"{first.get('extra_payload', {}).get('instance_id')}: "
            f"{first.get('agent_exit_status')}: {str(first.get('agent_error'))[:200]}); this is "
            "an infrastructure failure, not a 0% resolved rate")
        noncanon.append("agent infrastructure failure (no instance could run)")
    elif infra:
        result["status"] = "completed_with_errors"
        result["partial_evaluation"] = {
            "total_instances": total, "scoreable_instances": total - infra,
            "agent_infra_error_instances": infra}
        noncanon.append(f"partial: {infra} of {total} instance(s) could not run the agent "
                        "(infrastructure)")
        logger.warning("[%s] %d of %d instance(s) could not run the agent (container start or "
                       "endpoint failure): harness errors, left out of the score; see "
                       "agent_error on the traces.", eval_name, infra, total)
    if relay is not None:
        # Endpoint failures the relay saw are harness errors (completed_with_errors, not
        # comparable); a rollout one of them ended is already agent_infra (the driver's record).
        result["relay"] = relay.record()
        noncanon.extend(relay_failure_reasons(result, result["relay"]))
    noncanon.extend(_empty_turn_classes(result, traces, result.get("relay")))
    if finalize is not None:
        noncanon.extend(finalize(result, metrics) or [])
    if result.get("status") == "success" and result.get("scoring_excluded"):
        result["status"] = "completed_with_errors"
    if result.get("accuracy") is None and result.get("status") != "error":
        # every instance is a harness error: nothing was scored, the rate is withheld
        result["status"] = "error"
        result["error"] = (f"no instance was scored: all {total} are harness errors "
                           f"({harness_errors_text(result.get('harness_errors') or {})}); this "
                           "is an infrastructure failure, not a 0% resolved rate")
        noncanon.append("harness infrastructure failure (no scoreable instances)")
    withhold_on_error(result)

    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    result["duration_s"] = round(time.time() - start, 2)
    return result
