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

# Evaluation Suite: swe_bench_pro
# Description: SWE-bench Pro V2 (ScaleAI) - re-graded resolved rate under the V2 locked protocol

"""gbench native built-in runner for swe_bench_pro (Coding & Software Engineering).

SWE-bench Pro V2: `ScaleAI/SWE-bench_Pro` config `default` (V2 since 2026-09-22, 642 tasks) and
config `hard` (HARD-51) at a pinned Hub revision, run with Scale's V2 reference setup: the Harbor
task directories and locked-protocol tooling of scaleapi/SWE-bench_Pro-os at a pinned commit
(`v2/`), under Harbor 0.22.0 (the release gate's version) with its local docker backend. Two
Harbor jobs, exactly as `v2/README.md` "Running" documents them:

  1. agent job: `locked_mini_swe:LockedMiniSwe` (mini-swe-agent 2.4.6 installed in each task
     sandbox, `v2/tooling/configs/mini_toolcall.yaml`). The agent phase is offline except the
     model endpoint (`[agent] network_mode = "no-network"` + `--allow-agent-host`), with the
     50-minute budget of each task.toml. The agent's `git diff` is captured as model.patch and
     the task's verifier runs in the same sandbox (the in-sandbox score);
  2. re-grade job: `patch_replay:PatchReplayAgent` applies each model.patch to a pristine image
     and runs the unchanged verifier. This is the authoritative, reported score.

Headline `accuracy` = re-graded resolved rate over the selected V2 tasks that were scored. The
result also carries the in-sandbox rate ("Report the re-graded number; publish both") and both
rates on HARD-51, the second task of the dataset's eval.yaml (SWE_Bench_Pro_hard), computed from
the same trials.
Harbor and the pinned tree run in the LOCAL image
gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt (gbench/docker/swe_bench_pro_v2.Dockerfile),
docker-out-of-docker. The agent in each sandbox gets
OPENAI_API_KEY=EMPTY and reaches the endpoint only through gbench's relay (a KeyForwarder that adds
the endpoint's key when it has one, base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY, so
the key never enters Harbor or a sandbox, where the model's commands could read it). The relay
counts the endpoint's failures: harness errors (relay_failure_reasons: completed_with_errors, not
comparable), and a rollout the budget cut off while the endpoint was failing that rollout's own
requests (endpoint-side failures of requests carrying its conversation, with no reply after them)
is an agent_infra harness error (budget_kill_verdict). SANDBOX_EVAL. A provisioned
machine runs it with no opt-in; a missing prerequisite (Docker, the socket, the image) hard-errors
(infra_required). A real run pulls tens to hundreds of GB of instance images. The sandboxes run
model-controlled code with network access, so before any Harbor job the cloud metadata server gate
(require_sandbox_network_blocked: swebench_common.require_task_network_blocked, probing docker's
default bridge and a user-defined bridge network with the V2 image, and checking that the hosts
every sandbox installs the agent from resolve there, SANDBOX_INSTALL_HOSTS) must pass
(infra_required otherwise); the v1 route checks the metadata server before the engine runs
(require_v1_network_blocked).

Re-grade verdicts follow Harbor 0.22.0 (`regrade_verdict`): a trial whose verifier finished is
scored with its reward; a trial whose verifier started and then timed out or left no usable
reward (VerifierTimeoutError, RewardFileNotFoundError, RewardFileEmptyError,
VerifierOutputParseError: the verifier exceptions Harbor's default RetryConfig does not retry)
is the patch's outcome and scores 0, as Harbor's metrics count a trial without a reward. Any
other failure (sandbox start, image pull, a verifier upload or download error, no trial) is
infrastructure: re-graded once more, then a harness error (`no_regrade_verdict`), as is a task
whose rollout was not the model's own outcome (`agent_infra`, agent_trial_view): left out of
every rate's numerator and denominator, counted in `harness_errors`, the run partial
(completed_with_errors), and its headline withheld (None, status error) when no task was scored.

Mode selection (see `run_swe_bench_pro`): the V2 locked protocol by default; one-shot with
GBENCH_SWE_BENCH_PRO_MODE=oneshot (one diff per task from the task text, re-graded by the same
patch_replay job; never comparable); GBENCH_SWEBENCH_PRO_AGENTIC=1 routes to the separately
maintained agentic engine exactly as before. That engine scores with Scale's v1 harness
(`swe_bench_pro_eval.py`, `run_scripts/`, `jefzda/sweap-images`) against `raw_sample_path()`,
which is still the pinned v1 table (tag v1.0, 731 tasks): the v1 harness can only score v1 rows.
Its result is labelled as a v1-harness number (`ENGINE_V1_NOTE`).

Sampling: gbench's rule for every suite applies: a `--thinking` run is sent temperature 1.0
(temperature_source "thinking"), a no-think run 0.0 (temperature_source "default"), written into
the locked config's `model.model_kwargs`. An explicit `--temperature` or
`GBENCH_SWE_BENCH_PRO_TEMPERATURE` (which takes precedence) beats both. The locked config itself
sets no temperature, so the temperature sent is recorded and makes the run not
leaderboard-comparable. The locked config asks for `reasoning_effort: high`, which reaches a
template like Gemma's only through the server's own mapping; the served model's reasoning switch
is therefore set explicitly from `--thinking` (`chat_template_kwargs.enable_thinking`), and a run
without it is recorded as not comparable. The one-shot mode follows the same rule (1.0 with
--thinking, 0.0 without).

Egress sidecar read timeout: in the agent phase the model requests pass through Harbor's
egress-control sidecar, a gost transparent proxy that waits at most `readTimeout` for the
upstream's response headers, 15 s when unset (go-gost/x v0.10.9). A non-streamed completion sends
its headers only when the reply is complete, so every turn that took longer than 15 s was cut.
2026-10-11 validation run (5 tasks, shared server): 128 of 198 model requests were closed by the
client at <= 15 s (relay client_close_ages max_s 15.0), every agent logged "Server disconnected
without sending a response", 4 of 5 rollouts ended as agent_infra. The image now patches the
sidecar's gost.yaml with readTimeout EGRESS_SIDECAR_READ_TIMEOUT (gost cannot disable the wait:
0 means 15 s), the prerequisite check requires the image label that records it, and the result
records it in `harness.egress_sidecar`. V2 Harbor runs from 2026-09-25 (when the agent-phase
no-network + --allow-agent-host route was added) up to 2026-10-11 ran under the 15 s cut; earlier
runs and the GBENCH_SWEBENCH_PRO_AGENTIC=1 engine route did not send model traffic through it.
"""

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple
from . import swe_thread_cap
from .base import (
    apply_shard, get_run_knob, resolve_temperature, run_eval_suite, strip_thinking_tags,
    suite_env,
)
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample
from . import swe_agentic, swebench_common
from .swebench_common import (
    AGENT_INFRA, EndpointRelay, harness_errors_text, infra_required, mark_harness_error,
    prereqs_path, record_harness_errors, scored_headline, trace_patch, withhold_on_error,
)
#: Harness-error kind of a task whose re-grade gave no verdict on its patch (regrade_verdict).
NO_REGRADE_VERDICT = "no_regrade_verdict"

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_pro.md"
_DATASET = "ScaleAI/SWE-bench_Pro"

# --------------------------------------------------------------------------------------------
# SWE-bench Pro V2 (the default)
# --------------------------------------------------------------------------------------------
#: Hub head since the V2 release (2026-09-22): config `default` = V2 (642), `hard` = HARD-51,
#: `v1` = the original 731 tasks.
_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"
_CONFIG = "default"
_HARD_CONFIG = "hard"
V2_TASK_COUNT = 642
HARD_TASK_COUNT = 51
V2_DATASET_LABEL = (f"{_DATASET}@{_REVISION} (config default = SWE-bench Pro V2, test, "
                    f"{V2_TASK_COUNT}; config hard = HARD-{HARD_TASK_COUNT})")
#: The V2 task directories, verifier and locked-protocol tooling (`v2/`, checked against its
#: SHA256SUMS when the image is built).
HARNESS_REPO = "scaleapi/SWE-bench_Pro-os"
HARNESS_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"
#: Harbor release of the V2 release gate (v2/GATE.md: "Harbor 0.22").
HARBOR_VERSION = "0.22.0"
#: The reference mini-swe-agent entry of the locked protocol (v2/README.md "Running").
AGENT_IMPORT = "locked_mini_swe:LockedMiniSwe"
AGENT_CONFIG = "v2/tooling/configs/mini_toolcall.yaml"
AGENT_CONFIG_SHA256 = "7275afa9920a6fb1b9ec66ab74cee44121db9b0276bb9fa90f5ec0597e523a45"
#: LockedMiniSwe installs `mini-swe-agent{version}` in each sandbox; unpinned it resolves to the
#: newest release, which was 2.4.6 when V2 shipped. gbench pins it (`--ak version=`).
MINI_SWE_AGENT_VERSION = "2.4.6"
REGRADE_IMPORT = "patch_replay:PatchReplayAgent"
V2_IMAGE_DEFAULT = "gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt"
#: readTimeout the image writes into Harbor's egress-control sidecar config (gost.yaml, handler
#: metadata; image label org.gbench.egress_sidecar_read_timeout). gost's own default is 15 s.
EGRESS_SIDECAR_READ_TIMEOUT = "8760h"
EGRESS_SIDECAR_READ_TIMEOUT_S = 8760 * 3600
#: Where the image keeps the pinned SWE-bench_Pro-os tree.
_IMAGE_ROOT = "/opt/SWE-bench_Pro-os"
_DOCKER_SOCKET = "/var/run/docker.sock"
#: Harbor exceptions after which the trial still ran its verifier on the captured patch: the
#: model ran out of its budget (AgentTimeoutError) or the agent process exited non-zero.
_AGENT_OUTCOME_EXCEPTIONS = ("AgentTimeoutError", "NonZeroAgentExitCodeError")
#: mini-swe-agent 2.x exit statuses that are the model's own outcome. An empty status is a
#: rollout the 50-minute budget cut off (the trajectory is saved after every step); it stays the
#: model's outcome unless budget_kill_verdict finds the cut came while an endpoint failure was
#: being retried.
_MODEL_EXIT_STATUSES = ("Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError",
                        "ContextWindowExceededError", "")
#: Extra passes for tasks that never reached the model (agent job) or whose re-grade failed for
#: an infrastructure reason (`regrade_verdict`).
INFRA_RETRIES = 1
#: Harbor exceptions of a verifier that started and ran on the patched tree without producing a
#: usable reward: the verifier budget ran out (a test the patch made hang) or the reward file is
#: missing, empty or unparsable. They are in Harbor 0.22.0's default RetryConfig.exclude_exceptions
#: (models/job/config.py), and its metrics (metrics/base.py) count a trial without a reward as 0,
#: so they are the patch's outcome, not infrastructure.
_VERIFIER_OUTCOME_EXCEPTIONS = ("VerifierTimeoutError", "RewardFileNotFoundError",
                                "RewardFileEmptyError", "VerifierOutputParseError")

# --------------------------------------------------------------------------------------------
# SWE-bench Pro v1: kept for the separately maintained engine (GBENCH_SWEBENCH_PRO_AGENTIC=1)
# and plugin variants, which score with Scale's v1 harness. Do not use for the default path.
# --------------------------------------------------------------------------------------------
#: SWE-bench Pro v1 (731 tasks), the set Scale's v1 harness (swe_bench_pro_eval.py,
#: run_scripts/, jefzda/sweap-images) scores: the HF repo's tag v1.0. V2 rows apply
#: /tests/test_patch.patch, a file the v1 harness never mounts.
_V1_REVISION = "307bd006a0a3807424d3c5525eaf98ca8e63aed3"
_V1_TAG = "v1.0"
#: At that tag the v1 set is the repo's only config, "default" (config "v1" on main is
#: byte-identical, LFS sha256 c8cd7115...).
_V1_CONFIG = "default"
V1_DATASET_LABEL = f"{_DATASET}@{_V1_REVISION} (tag {_V1_TAG}, SWE-bench Pro v1, test)"
#: The v1 table the engine's harness reads (kept under its historical name).
DATASET_LABEL = V1_DATASET_LABEL
V1_PIN_NOTE = (f"scored against SWE-bench Pro v1 ({_DATASET} tag {_V1_TAG}, {_V1_REVISION[:8]}, "
               "731 tasks) with Scale's v1 harness, while upstream's default is SWE-bench Pro V2 "
               "(642 tasks, since 2026-09-22), so this is a v1 number")
#: Appended to the maintained engine's result (GBENCH_SWEBENCH_PRO_AGENTIC=1).
ENGINE_V1_NOTE = ("GBENCH_SWEBENCH_PRO_AGENTIC=1 ran the separately maintained engine, which "
                  "scores with Scale's v1 harness against the pinned v1 raw-sample table, not "
                  "the V2 locked protocol; " + V1_PIN_NOTE)
#: The raw-sample table gbench writes for the v1 harness, named after the pinned revision so a
#: table from another revision (a pre-pin swe_bench_pro_full.csv, or V2 rows) is never reused.
RAW_SAMPLE_NAME = f"swe_bench_pro_full.{_V1_TAG}-{_V1_REVISION[:8]}.csv"
#: Marker of a V2 row that the v1 harness cannot run.
_V2_MARKER = "/tests/test_patch.patch"

#: The output-format instruction the one-shot scorers REQUIRE: they grade the patch pulled from
#: a ```diff block (`extract_patch`); a prompt that does not ask for exactly that yields prose
#: and an empty patch (a structural 0). Exposed as a constant so plugin variants share the SAME
#: instruction. Thinking-mode runs still work: the plan goes in the reasoning channel and
#: `extract_patch` strips it before reading the diff.
PATCH_OUTPUT_INSTRUCTION = (
    "Output ONLY a single unified git diff (`diff --git a/... b/...`) that resolves "
    "the issue, inside a ```diff code block.")


# ============================================================================================
# V2: dataset
# ============================================================================================
def _load_config_rows(config: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        return [dict(r) for r in load_dataset(_DATASET, config, split="test", revision=_REVISION)]
    except Exception as e:
        logger.error(f"Failed to load dataset for swe_bench_pro: {e}")
        raise RuntimeError(f"Could not load dataset for swe_bench_pro ({_DATASET}@{_REVISION}, "
                           f"config {config}): {e}") from e


def load_v2_rows() -> List[Dict[str, Any]]:
    """Every row of the pinned V2 test split. Raises on a load failure, a short split, or a row
    that is not V2 (no ghcr.io/scaleapi/swe-bench_pro-v2 image)."""
    rows = _load_config_rows(_CONFIG)
    if len(rows) != V2_TASK_COUNT:
        raise RuntimeError(f"swe_bench_pro: {V2_DATASET_LABEL} returned {len(rows)} rows, "
                           f"expected {V2_TASK_COUNT}")
    for r in rows:
        iid = r.get("instance_id")
        if not iid or not r.get("problem_statement") or \
                r.get("docker_image") != f"ghcr.io/scaleapi/swe-bench_pro-v2:{iid}":
            raise RuntimeError(f"swe_bench_pro: row {iid!r} of {V2_DATASET_LABEL} is not a V2 "
                               "task (instance_id/problem_statement/docker_image); refusing to "
                               "guess")
    return rows


def load_hard51_ids(v2_ids: Optional[set] = None) -> set:
    """Instance ids of the pinned HARD-51 subset (config `hard`), checked against V2."""
    ids = {r.get("instance_id") for r in _load_config_rows(_HARD_CONFIG)}
    if len(ids) != HARD_TASK_COUNT or (v2_ids is not None and not ids <= v2_ids):
        raise RuntimeError(f"swe_bench_pro: config hard at {_REVISION} is not the "
                           f"{HARD_TASK_COUNT}-task subset of V2 ({len(ids)} ids)")
    return ids


# ============================================================================================
# V2: prerequisites and the Harbor orchestrator container
# ============================================================================================
def v2_image() -> str:
    return (os.environ.get("GBENCH_SWE_BENCH_PRO_IMAGE") or "").strip() or V2_IMAGE_DEFAULT


def check_swe_bench_pro_v2_prerequisites() -> Tuple[bool, str]:
    """Docker + socket + the local V2 image (Harbor HARBOR_VERSION, tree HARNESS_COMMIT)."""
    image = v2_image()
    build = (f"Build the V2 harness image LOCALLY (gbench never pulls it): "
             f"docker build -t {image} -f gbench/docker/swe_bench_pro_v2.Dockerfile gbench/docker")
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH. " + build
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    if not os.path.exists(_DOCKER_SOCKET):
        return False, (f"{_DOCKER_SOCKET} does not exist; Harbor runs in a container and drives "
                       "the task sandboxes through the mounted socket.")
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format",
                            "{{json .Config.Labels}}", image],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, f"Cannot inspect the image {image!r}: {e}"
    if r.returncode != 0:
        return False, f"the SWE-bench Pro V2 harness image {image!r} is not built. " + build
    try:
        labels = json.loads(r.stdout or "null") or {}
    except ValueError:
        labels = {}
    want = {"org.gbench.harbor": HARBOR_VERSION, "org.gbench.swe_bench_pro_os": HARNESS_COMMIT,
            "org.gbench.egress_sidecar_read_timeout": EGRESS_SIDECAR_READ_TIMEOUT}
    wrong = {k: labels.get(k) for k, v in want.items() if labels.get(k) != v}
    if wrong:
        return False, (f"the image {image!r} was built from {wrong}, not {want}; rebuild it. "
                       + build)
    return True, ""


#: The package hosts every V2 sandbox installs the agent from at run time, before the model's
#: first step: LockedMiniSwe.install (v2/tooling/locked_mini_swe.py) installs curl, git, python3
#: and build tools with the image's apt (Debian bullseye/bookworm images: deb.debian.org and
#: security.debian.org; retried against archive.debian.org; Ubuntu images: archive.ubuntu.com),
#: runs astral.sh's uv installer (which downloads uv from GitHub), and `uv tool install --python
#: 3.12 mini-swe-agent==MINI_SWE_AGENT_VERSION` (Python from GitHub, the package from PyPI). Five
#: of the 642 task verifiers (tests/test.sh, gravitational/teleport) also apt-get install. The
#: gate checks that each name resolves on the sandboxes' networks.
SANDBOX_INSTALL_HOSTS = ("deb.debian.org", "security.debian.org", "archive.debian.org",
                         "archive.ubuntu.com", "astral.sh", "github.com", "pypi.org",
                         "files.pythonhosted.org")


def require_sandbox_network_blocked() -> None:
    """The cloud metadata server gate before any Harbor job (swebench_common.
    require_task_network_blocked, probing with the V2 image). Harbor's task sandboxes run the
    in-sandbox agent's commands and the verifier's tests over each patch on the compose network
    it creates per trial (a user-defined bridge network), with egress in the setup and verifier
    phases; the agent install in each sandbox needs SANDBOX_INSTALL_HOSTS to resolve there, which
    the gate checks too."""
    swebench_common.require_task_network_blocked(
        "swe_bench_pro", [v2_image()], docs_url=DOCS_URL,
        hint=f"Build the V2 harness image {v2_image()} (see {DOCS_URL}).",
        resolve=SANDBOX_INSTALL_HOSTS)


def require_v1_network_blocked() -> None:
    """The same gate for the v1 harness route (the maintained engine's mini-swe-agent task
    containers and the v1 harness's containers, on docker's default bridge): the probe runs in the
    V2 image, else in a local v1 instance image (jefzda/sweap-images)."""
    swebench_common.require_task_network_blocked(
        "swe_bench_pro", [v2_image()], docs_url=DOCS_URL, references=["jefzda/sweap-images"],
        hint=f"Build the V2 harness image {v2_image()} (see {DOCS_URL}), or have a "
             "jefzda/sweap-images instance image on the host.")


def task_endpoint(base_url: str) -> str:
    """The served /v1 endpoint as the task sandboxes reach it.

    The sandboxes run on docker bridge networks, where 127.0.0.1 is the sandbox itself; they
    reach the host through the bridge gateway. `GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT` overrides
    the whole URL; otherwise a localhost/127.0.0.1 host becomes `GBENCH_SWE_BENCH_PRO_TASK_HOST`
    (default 172.17.0.1). The server must listen on that address."""
    override = (os.environ.get("GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT") or "").strip()
    ep = override or base_url.rstrip("/")
    if not ep.endswith("/v1"):
        ep = f"{ep}/v1"
    if override:
        return ep
    host = os.environ.get("GBENCH_SWE_BENCH_PRO_TASK_HOST", "172.17.0.1")
    return re.sub(r"://(127\.0\.0\.1|localhost)(?=[:/])", f"://{host}", ep)


def endpoint_host(url: str) -> str:
    """Host part of `url`: the one entry of the agent phase allowlist."""
    m = re.match(r"^[a-z]+://(\[[^\]]+\]|[^/:]+)", url)
    if not m:
        raise ValueError(f"cannot read a host from the endpoint {url!r}")
    return m.group(1).strip("[]")


#: What the task sandboxes get as OPENAI_API_KEY: the placeholder for an endpoint without a key.
#: A real key never enters Harbor or a sandbox (see KeyForwarder).
SANDBOX_API_KEY = "EMPTY"


def endpoint_api_key() -> Optional[str]:
    """The served endpoint's key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY, a
    placeholder such as EMPTY read as no key), or None when it has none."""
    from .base import endpoint_api_key as _endpoint_key
    return _endpoint_key()


class KeyForwarder(EndpointRelay):
    """Relays the task sandboxes' model requests to the served endpoint and adds its key.

    In the locked protocol mini-swe-agent runs inside each task sandbox, and every command the
    model runs there inherits the agent's environment (mini-swe-agent 2.x LocalEnvironment:
    `os.environ | config.env`), so a key handed to the agent is visible to the model; Harbor
    also passes agent variables as `docker compose exec -e NAME=VALUE`, on a command line.
    When the endpoint has a key, the sandboxes get this forwarder's address instead of the
    endpoint and OPENAI_API_KEY=EMPTY; the forwarder, a thread of the gbench process listening
    on the address the sandboxes reach the host at, sends each request on to the endpoint with
    `Authorization: Bearer <key>` and returns the endpoint's reply unchanged (status,
    Content-Type, body, streamed as it arrives).

    It is a swebench_common.EndpointRelay: the target is always the endpoint given here (scheme,
    host and port fixed when the forwarder is made; the checked /v1/... request path is appended
    to its base path with urlunsplit, never by string concatenation), a request whose method is
    not GET/POST or whose path is not one of the relay's allowed endpoint paths (chat
    completions, completions, models) is refused with a 4xx and never forwarded, and redirects
    are not followed, so no request can re-target the forwarder or get the key sent anywhere
    else. Nothing else about the requests changes."""

    def __init__(self, upstream: str, key: str, bind_host: str,
                 body_fields: Optional[Dict[str, Any]] = None,
                 timeout_s: Optional[float] = None,
                 client_timeout_suspect_s: Optional[Sequence[float]] = None):
        super().__init__(upstream, bind_host, key=key, name="swe-pro-key-forwarder",
                         body_fields=body_fields, timeout_s=timeout_s,
                         client_timeout_suspect_s=client_timeout_suspect_s)


def forwarder_host() -> str:
    """The host address the task sandboxes reach the gbench host at (the forwarder listens
    there): GBENCH_SWE_BENCH_PRO_TASK_HOST, default 172.17.0.1 (docker's default bridge)."""
    return (os.environ.get("GBENCH_SWE_BENCH_PRO_TASK_HOST") or "").strip() or "172.17.0.1"


def _orchestrator_cmd(workdir: str, name: str, harbor_args: List[str]) -> List[str]:
    """`docker run` of one Harbor job in the V2 image: host network, the docker socket, the
    work directory at the same path (Harbor bind-mounts trial directories into the sibling
    sandboxes, so host and container paths must agree), the caller's uid.

    The agent gets OPENAI_API_KEY=EMPTY (SANDBOX_API_KEY); an endpoint's real key never enters
    this container (KeyForwarder)."""
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "host",
           "-v", f"{_DOCKER_SOCKET}:{_DOCKER_SOCKET}", "-v", f"{workdir}:{workdir}:rw",
           "-w", _IMAGE_ROOT, "-e", f"PYTHONPATH={_IMAGE_ROOT}/v2/tooling",
           "-e", f"HOME={os.path.join(workdir, 'home')}"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        try:
            cmd += ["--group-add", str(os.stat(_DOCKER_SOCKET).st_gid)]
        except OSError:
            pass
    return cmd + [v2_image(), "harbor", "run", *harbor_args,
                  "--ae", f"OPENAI_API_KEY={SANDBOX_API_KEY}"]


def _run_harbor(workdir: str, harbor_args: List[str], job: str) -> Tuple[int, str]:
    """Run one Harbor job to completion; (return code, output tail). Never raises."""
    os.makedirs(os.path.join(workdir, "home"), exist_ok=True)
    name = f"gbench-swepro-{job}-{uuid.uuid4().hex[:10]}"
    log_path = os.path.join(workdir, f"harbor_{job}.log")
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(_orchestrator_cmd(workdir, name, harbor_args),
                                  stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, text=True)
        rc = proc.returncode
    except Exception as e:
        rc = -1
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\ngbench: could not run the Harbor container: {type(e).__name__}: {e}\n")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-1500:]
    except OSError:
        tail = ""
    return rc, tail


def read_locked_agent_config() -> Tuple[str, str]:
    """(text, sha256) of the locked mini-swe-agent config inside the pinned tree."""
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat",
                        v2_image(), f"{_IMAGE_ROOT}/{AGENT_CONFIG}"],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"cannot read {AGENT_CONFIG} from {v2_image()!r}: {r.stderr[-300:]}")
    return r.stdout, hashlib.sha256(r.stdout.encode("utf-8")).hexdigest()


def build_locked_agent_config(text: str, *, thinking: bool, temperature: float,
                              client_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The locked config plus the served model's reasoning switch and the temperature (always
    sent: 1.0 with --thinking, 0.0 without, unless the operator set one), and `client_kwargs`
    (_run_v2 passes swe_agentic.litellm_client_kwargs: litellm's `timeout` and the OpenAI SDK's
    `max_retries` 0, which mini-swe-agent's LitellmModel passes to every litellm.completion
    call). Nothing else changes: templates, limits and model settings stay as shipped.

    Without them the sandbox's litellm (installed unpinned by LockedMiniSwe; 1.87 through 1.102
    checked) falls back to a 600 s client timeout with 2 SDK retries, so a turn longer than 600 s
    is abandoned and sent again. (2026-10-10 smoke run: the relay saw 186 of 313 replies closed
    by the client; how many were that timeout was not established.)"""
    import yaml
    cfg = copy.deepcopy(yaml.safe_load(text) or {})
    kw = cfg.setdefault("model", {}).setdefault("model_kwargs", {})
    kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    kw["temperature"] = temperature
    kw.update(client_kwargs or {})
    return cfg


def agent_transport(kwargs: Dict[str, Any]) -> Tuple[Optional[int], "swe_agentic.AgentTimeouts"]:
    """(max_tokens the relay sets on every request under GBENCH_AGENT_MAX_TOKENS_FROM_RUN=1, else
    None; the relay and client timeouts sized for it, swe_agentic.agent_request_timeouts)."""
    run_max_tokens = swe_agentic.agent_max_tokens_from_run(kwargs.get("max_output_tokens"))
    return run_max_tokens, swe_agentic.agent_request_timeouts(run_max_tokens)


def _thread_flags(threads: int, flags: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    if threads > 0:
        for var, value in swe_thread_cap.thread_env(threads).items():
            for flag in flags:
                out += [flag, f"{var}={value}"]
    return out


def agent_job_args(*, model_name: str, endpoint: str, config_path: str, jobs_dir: str,
                   concurrency: int, include: Optional[List[str]], threads: int,
                   job: str = "agent") -> List[str]:
    """`harbor run` arguments of the agent job (v2/README.md "Evaluate a model under the locked
    protocol", mini-swe-agent variant)."""
    args = ["-p", f"{_IMAGE_ROOT}/v2/tasks", "-e", "docker", "-n", str(max(1, concurrency)),
            "-a", AGENT_IMPORT, "-m", f"openai/{model_name}",
            "--ak", f"config_file={config_path}", "--ak", f"version={MINI_SWE_AGENT_VERSION}",
            "--ae", f"OPENAI_API_BASE={endpoint}",
            "--allow-agent-host", endpoint_host(endpoint),
            "--jobs-dir", jobs_dir, "--job-name", job, "-y"]
    args += _thread_flags(threads, ("--ae", "--ve"))
    for iid in include or []:
        args += ["-i", iid]
    return args


def regrade_job_args(*, source_job: str, jobs_dir: str, concurrency: int,
                     include: Optional[List[str]], threads: int, job: str = "regrade") -> List[str]:
    """`harbor run` arguments of the fresh-sandbox re-grade (v2/README.md "authoritative
    score")."""
    args = ["-p", f"{_IMAGE_ROOT}/v2/tasks", "-e", "docker", "-n", str(max(1, concurrency)),
            "-a", REGRADE_IMPORT, "-m", "replay", "--ak", f"source_job={source_job}",
            "--jobs-dir", jobs_dir, "--job-name", job, "-y"]
    args += _thread_flags(threads, ("--ve",))
    for iid in include or []:
        args += ["-i", iid]
    return args


# ============================================================================================
# V2: reading Harbor's job directories
# ============================================================================================
def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def job_trials(job_dir: str) -> Dict[str, Dict[str, Any]]:
    """{instance_id: trial} for every trial Harbor wrote under `job_dir`.

    A trial carries its reward (verifier_result.rewards.reward, None when the verifier produced
    none), the exception type Harbor recorded, and its directory."""
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(job_dir):
        return out
    for entry in sorted(os.listdir(job_dir)):
        tdir = os.path.join(job_dir, entry)
        res = _read_json(os.path.join(tdir, "result.json")) if os.path.isdir(tdir) else None
        if not isinstance(res, dict) or not res.get("task_name"):
            continue
        iid = str(res["task_name"]).split("/")[-1]
        rewards = ((res.get("verifier_result") or {}).get("rewards") or {})
        reward = rewards.get("reward")
        exc = res.get("exception_info") or {}
        agent = res.get("agent_execution") or {}
        out[iid] = {"reward": float(reward) if isinstance(reward, (int, float)) else None,
                    "exception": exc.get("exception_type"),
                    "exception_message": str(exc.get("exception_message") or "")[:300],
                    "agent_started": bool(agent.get("started_at")),
                    "agent_started_at": _epoch(agent.get("started_at")),
                    "agent_finished_at": _epoch(agent.get("finished_at")),
                    "verifier_started": bool((res.get("verifier") or {}).get("started_at")),
                    "trial_dir": tdir}
    return out


def _epoch(stamp: Any) -> Optional[float]:
    """Epoch seconds of one of Harbor's ISO 8601 timestamps ("2026-09-27T17:16:43.904227Z");
    None when there is none."""
    import datetime
    if not stamp:
        return None
    try:
        return datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


#: The record tenacity's before_sleep_log writes before each retry of a failed model request in
#: mini-swe-agent 2.x (models/utils/retry.py: up to MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT = 10
#: attempts, wait_exponential(min=4, max=60), reraise=True), which the agent's console log
#: (agent/mini-swe-agent.txt, Harbor's `2>&1 | tee`) holds: "Retrying <fn> in <s> seconds as it
#: raised <Class>: <str(exception)>." (str() of a LiteLLM exception is "litellm.<Class>: ...").
_MODEL_RETRY_RE = re.compile(r"Retrying \S+ in [\d.]+ seconds as it raised (\w+): ")
#: InteractiveAgent.add_messages' header before each model reply it prints.
_REPLY_HEADER_RE = re.compile(r"mini-swe-agent \(step \d+, \$[\d.]+\):")


def pending_endpoint_retry(log_text: str) -> Optional[str]:
    """The model-endpoint exception mini-swe-agent was retrying when its console log ends: the
    exception class of the last retry record after the last model reply's header, when its text
    starts with a model-endpoint exception (swebench_common.scaffold_error_harness_reason);
    None otherwise. The log also carries the model's command output, so a caller only acts on
    it when the host-side relay saw an endpoint failure in the same time window
    (budget_kill_verdict)."""
    text = " ".join((log_text or "").split())
    start = 0
    for m in _REPLY_HEADER_RE.finditer(text):
        start = m.end()
    last = None
    for m in _MODEL_RETRY_RE.finditer(text, start):
        last = m
    if last is None:
        return None
    if swebench_common.scaffold_error_harness_reason(text[last.end():last.end() + 400]) is None:
        return None
    return last.group(1)


def _last_event_at(traj: Dict[str, Any]) -> Optional[float]:
    """Epoch time of the last message of a mini-swe-agent 2.x trajectory (each message's
    extra.timestamp: the model's reply, or the output of the command it ran)."""
    times = []
    for msg in traj.get("messages") or []:
        extra = msg.get("extra") if isinstance(msg, dict) else None
        try:
            times.append(float((extra or {}).get("timestamp")))
        except (TypeError, ValueError):
            continue
    return max(times) if times else None


def budget_kill_verdict(view: Dict[str, Any], relay: Optional[EndpointRelay]) -> Optional[str]:
    """Why a rollout the 50-minute budget cut off (exit status "") is a harness error, or None
    when it stays the model's outcome.

    Upstream counts a cut-off rollout as its patch's verdict, and so does gbench, unless the
    budget ran out while the endpoint was failing THIS rollout's model requests. That takes
    evidence the model cannot produce from its sandbox, where its commands can reach the relay
    and print anything into the agent's log:
      * the relay, on the host side, counted an endpoint-side failure (no reply, a 5xx, a stream
        error or an unparsable 2xx: swebench_common.endpoint_side_failure, never a 4xx a sender
        can provoke) of a request carrying this rollout's own conversation (its trajectory's
        first user message, swebench_common.conversation_fingerprint; a concurrent rollout's
        failure has another fingerprint) between the rollout's last event and the end of its
        agent phase (EndpointRelay.failure_times), and
      * no request of that conversation got a reply after the last such failure
        (EndpointRelay.succeeded_after): the rollout never got going again.
    The agent's log must also end in a retry record naming a model-endpoint exception
    (pending_endpoint_retry); that text alone decides nothing, since the model's command output
    shares the log. Then the model lost the end of its budget to the endpoint (rule: endpoint
    failures are harness errors). A rollout that was waiting on a reply that had not failed (a
    long generation, or an endpoint that stopped answering without an error) cannot be told
    apart from the model's own use of its budget and stays the model's outcome, with the wait
    recorded (budget_kill.last_event_age_s)."""
    kill = view.get("budget_kill") or {}
    if not view.get("model_outcome") or not kill.get("retrying") or relay is None:
        return None
    end = kill.get("ended_at")
    begin = kill.get("last_event_at") or kill.get("agent_started_at")
    if end is None or begin is None:
        return None
    fingerprint = kill.get("conversation_fingerprint")
    if not fingerprint:
        kill["relay_failures_in_window"] = 0
        kill["attribution"] = "no conversation fingerprint in the trajectory"
        return None
    times = relay.failure_times(float(begin) - 1.0, float(end) + 1.0, fingerprint)
    kill["relay_failures_in_window"] = len(times)
    if not times:
        return None
    if relay.succeeded_after(fingerprint, max(times)):
        kill["recovered_after_failure"] = True
        return None
    return (f"the 50-minute budget ran out while mini-swe-agent was retrying a model request "
            f"the endpoint failed ({kill['retrying']}); the relay counted {len(times)} "
            "endpoint-side failure(s) of this rollout's requests between its last event and the "
            "end of its agent phase, and no reply after them")


def agent_trial_view(trial: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What an agent-job trial says about the rollout: the in-sandbox mini-swe-agent exit
    status and step count, the captured patch, the in-sandbox reward, and whether the rollout is
    the model's own outcome (`model_outcome`, so its patch goes to the re-grade) or an
    infrastructure failure.

    Model outcome: the agent phase ran to its end (Harbor then started the verifier, or recorded
    only the model's budget running out / the agent process exiting non-zero), LockedMiniSwe
    captured model.patch (empty when the model changed nothing), and the in-sandbox agent ended
    for a model reason (submitted, limits, format errors, context window, cut off by the
    budget). The in-sandbox verifier's own failure does not disqualify the patch: the re-grade
    is the authoritative verdict."""
    if not trial:
        return {"model_outcome": False, "why": "no trial was written", "patch": "",
                "reward": None}
    tdir = trial["trial_dir"]
    traj_path = os.path.join(tdir, "agent", "mini-swe-agent.trajectory.json")
    traj = _read_json(traj_path) or {}
    info = traj.get("info") or {}
    exit_status = info.get("exit_status")
    n_calls = int(((info.get("model_stats") or {}).get("api_calls")) or 0)
    patch = _read_text(os.path.join(tdir, "agent", "model.patch"))
    view = {"exit_status": exit_status, "n_calls": n_calls, "patch": patch or "",
            "exception": trial.get("exception"), "reward": trial.get("reward"),
            "verifier_started": bool(trial.get("verifier_started")),
            "trajectory_file": traj_path if traj else None}
    exc = trial.get("exception")
    agent_done = trial.get("agent_started") and (
        not exc or exc in _AGENT_OUTCOME_EXCEPTIONS or trial.get("verifier_started"))
    if not agent_done:
        return {**view, "model_outcome": False,
                "why": (f"{exc}: {trial.get('exception_message')}" if exc
                        else "the agent phase did not run")}
    if patch is None:
        return {**view, "model_outcome": False, "why": "model.patch was not captured"}
    if not traj:
        return {**view, "model_outcome": False, "why": "the agent wrote no trajectory"}
    if str(exit_status or "") not in _MODEL_EXIT_STATUSES:
        return {**view, "model_outcome": False,
                "why": f"mini-swe-agent ended with {exit_status} after {n_calls} call(s)"}
    if not exit_status:
        # Cut off by the budget: what the rollout was doing when it ended (budget_kill_verdict).
        last = _last_event_at(traj)
        end = trial.get("agent_finished_at")
        log = _read_text(os.path.join(tdir, "agent", "mini-swe-agent.txt")) or ""
        view["budget_kill"] = {
            "last_event_at": last, "ended_at": end,
            "agent_started_at": trial.get("agent_started_at"),
            "last_event_age_s": round(end - last, 1) if end is not None and last is not None
            else None,
            "retrying": pending_endpoint_retry(log),
            # The rollout's identity in the relay's failure records (budget_kill_verdict).
            "conversation_fingerprint": swebench_common.conversation_fingerprint(
                traj.get("messages"))}
    return {**view, "model_outcome": True, "why": None}


def regrade_verdict(trial: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
    """(reward, None) when a re-grade trial is a verdict on the patch, (None, why) when it is an
    infrastructure failure.

    A reward means the verifier finished, and Harbor's metrics use it whatever else the trial
    recorded. A verifier that started and then raised one of `_VERIFIER_OUTCOME_EXCEPTIONS` ran
    on the patched tree: the patch made its tests hang past the verifier budget, or no usable
    reward came out; Harbor counts that trial as 0 and never retries it, so it scores 0 here too.
    Every other failure (no trial, an exception before the verifier started, a verifier upload or
    download error) happened outside the patch's control."""
    if not trial:
        return None, "no re-grade trial was written"
    reward = trial.get("reward")
    if reward is not None:
        return float(reward), None
    exc = trial.get("exception")
    if exc in _VERIFIER_OUTCOME_EXCEPTIONS and trial.get("verifier_started"):
        return 0.0, None
    if exc:
        return None, f"re-grade {exc}: {trial.get('exception_message')}"
    return None, "the re-grade produced no reward"


def in_sandbox_verdict(view: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """(reward, None) for the agent trial's own verifier result, (None, why) when that verifier
    failed for an infrastructure reason: it started and raised an exception other than
    `_VERIFIER_OUTCOME_EXCEPTIONS` without a reward (regrade_verdict's rule). A trial without a
    reward otherwise scores 0, as Harbor's metrics count it."""
    reward = view.get("reward")
    if reward is not None:
        return float(reward), None
    exc = view.get("exception")
    if view.get("verifier_started") and exc and exc not in _VERIFIER_OUTCOME_EXCEPTIONS:
        return None, f"in-sandbox verifier {exc}"
    return 0.0, None


def _rate(ids: List[str], resolved: set, scored: Optional[set] = None) -> Dict[str, Any]:
    """Resolved over `ids`, or over the ids in `scored` (harness errors left out; the rate is
    None when none was scored)."""
    pool = [i for i in ids if scored is None or i in scored]
    n = len(pool)
    k = sum(1 for i in pool if i in resolved)
    out: Dict[str, Any] = {"total": n, "resolved": k,
                           "accuracy": round(k / n * 100.0, 2) if n else None}
    if scored is not None and n != len(ids):
        out["harness_errors"] = len(ids) - n
    if n == 0 and scored is None:
        out["accuracy"] = 0.0
    return out


# ============================================================================================
# V2: the locked-protocol run
# ============================================================================================
def run_swe_bench_pro_v2(model_name: str, base_url: str, concurrency: int = 4,
                         enable_thinking: bool = False, **kwargs: Any) -> Dict[str, Any]:
    """SWE-bench Pro V2 under the locked protocol: Harbor agent job, then the fresh-sandbox
    re-grade. Assumes `check_swe_bench_pro_v2_prerequisites` passed.

    The sandboxes reach the served endpoint through a swebench_common.EndpointRelay for the
    whole run, listening at forwarder_host(): a KeyForwarder (which adds the key) when the
    endpoint has one (endpoint_api_key), so the key never enters Harbor or a task sandbox, and a
    relay without a key otherwise. Either way the relay counts the endpoint's failures
    (relay_failure_reasons, budget_kill_verdict). The cloud metadata server gate
    (require_sandbox_network_blocked) runs first."""
    require_sandbox_network_blocked()
    key = endpoint_api_key()
    host = forwarder_host()
    # The relay's per-request limit (the agent's client waits that plus a margin) and, under
    # GBENCH_AGENT_MAX_TOKENS_FROM_RUN=1, the max_tokens it sets on every request.
    run_max_tokens, timeouts = agent_transport(kwargs)
    body = swe_agentic.with_run_max_tokens({}, run_max_tokens)
    try:
        relay = (KeyForwarder(base_url, key, host, body_fields=body,
                              timeout_s=timeouts.relay_s,
                              client_timeout_suspect_s=timeouts.suspect_s()) if key else
                 EndpointRelay(base_url, host, name="swe-pro-endpoint-relay", body_fields=body,
                               timeout_s=timeouts.relay_s,
                               client_timeout_suspect_s=timeouts.suspect_s())).start()
    except OSError as e:
        why = ("the served endpoint has a key (VLLM_API_KEY or OPENAI_API_KEY), which must not "
               "reach the task "
               "sandboxes, where the model's commands can read the agent's environment. gbench "
               "relays the sandboxes' requests through a forwarder that adds the key"
               if key else
               "gbench relays the sandboxes' model requests to the served endpoint, so that "
               "the endpoint's failures are counted as harness errors, never as the model's "
               "outcome; the relay")
        raise infra_required(
            "swe_bench_pro", f"{why} and could not listen on {host} ({type(e).__name__}: {e}); "
            "set GBENCH_SWE_BENCH_PRO_TASK_HOST to the address the sandboxes reach this host "
            "at.", DOCS_URL) from e
    try:
        return _run_v2(model_name, base_url, concurrency, enable_thinking, relay, **kwargs)
    finally:
        relay.stop()


def _run_v2(model_name: str, base_url: str, concurrency: int, enable_thinking: bool,
            relay: EndpointRelay, **kwargs: Any) -> Dict[str, Any]:
    """run_swe_bench_pro_v2 once the relay the sandboxes use is running."""
    forwarder = relay if isinstance(relay, KeyForwarder) else None
    start = time.time()
    limit = kwargs.get("limit")
    rows = load_v2_rows()
    hard_ids = load_hard51_ids({r["instance_id"] for r in rows})
    tasks = [swe_agentic.AgentTask(instance_id=r["instance_id"],
                                   problem_statement=r["problem_statement"],
                                   image=r["docker_image"],
                                   category=str(r.get("repo_language") or r.get("repo")))
             for r in rows]
    tasks, shard = swe_agentic.select_tasks(tasks, limit, "swe_bench_pro")
    selected = [t.instance_id for t in tasks]
    # One instance set, no settings: --eval-limit N runs N instances in total.
    plan = eval_limit_plan("swe_bench_pro", limit, len(tasks), attempts=1, unit="rollouts")
    full = not limit and not shard and len(selected) == V2_TASK_COUNT
    temperature, temperature_source = resolve_temperature(
        "swe_bench_pro", kwargs.get("temperature"), thinking=enable_thinking)
    max_tokens = kwargs.get("max_output_tokens") or get_run_knob("max_output_tokens")
    threads = swe_thread_cap.resolve_threads(concurrency)
    workdir = swe_agentic.agent_output_dir("swe_bench_pro")
    jobs_dir = os.path.join(workdir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    endpoint = relay.url
    if (os.environ.get("GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT") or "").strip():
        logger.warning("swe_bench_pro: GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT is not used by the V2 "
                       "run: the sandboxes always reach the endpoint through gbench's relay; "
                       "GBENCH_SWE_BENCH_PRO_TASK_HOST sets the address they reach it at")

    run_max_tokens, timeouts = agent_transport(kwargs)
    if EGRESS_SIDECAR_READ_TIMEOUT_S <= timeouts.client_s:
        raise RuntimeError(f"swe_bench_pro: the egress sidecar's readTimeout "
                           f"({EGRESS_SIDECAR_READ_TIMEOUT}) does not exceed the agent's client "
                           f"timeout ({timeouts.client_s} s); it would cut model replies")
    client_kwargs = swe_agentic.litellm_client_kwargs(timeouts)
    text, cfg_sha = read_locked_agent_config()
    import yaml
    cfg = build_locked_agent_config(text, thinking=enable_thinking, temperature=temperature,
                                    client_kwargs=client_kwargs)
    cfg_path = os.path.join(workdir, "mini_toolcall.gbench.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    result: Dict[str, Any] = {
        "eval_limit_plan": plan,
        "benchmark_type": "eval",
        "eval_name": "swe_bench_pro",
        "model_name": model_name,
        "thinking": enable_thinking,
        "mode": swe_agentic.AGENTIC,
        "mode_source": swe_agentic.resolve_mode("swe_bench_pro")[1],
        "scoring_mode": "execution",
        "temperature": temperature,
        "temperature_source": temperature_source,
        "max_output_tokens_per_turn": run_max_tokens,
        "max_output_tokens_note": (
            f"--max-output-tokens={run_max_tokens} is sent with every request (set by gbench's "
            f"relay, {swe_agentic.AGENT_MAX_TOKENS_FROM_RUN_ENV}=1); the V2 locked config sets "
            "no per-turn cap" if run_max_tokens else
            f"--max-output-tokens={max_tokens} is not applied: the V2 locked config sets no "
            "per-turn cap (the server grants what the context window has left)"
            if max_tokens else "the V2 locked config sets no per-turn cap"),
        "model_client_timeout": swe_agentic.model_client_timeout(
            f"litellm (mini-swe-agent {MINI_SWE_AGENT_VERSION} LitellmModel in each sandbox, "
            "model.model_kwargs of the locked config)",
            "unpinned: the newest litellm when LockedMiniSwe installs mini-swe-agent in the "
            "sandbox (uv tool install ... --with litellm)",
            timeouts.client_s, swe_agentic.AGENT_CLIENT_RETRIES,
            timeouts.source + "; the relay's limit is the same value. Each rollout is also "
            "bounded by its task's 50-minute agent budget (task.toml), unchanged",
            replaces="litellm's fallback client timeout of 600 s "
                     "(COMPLETION_HTTP_FALLBACK_SECONDS) with 2 OpenAI SDK retries"),
        "dataset": V2_DATASET_LABEL,
        "harness": {
            "tree": f"{HARNESS_REPO}@{HARNESS_COMMIT} (v2/)",
            "harbor": HARBOR_VERSION,
            "environment": "docker (Harbor's local backend, docker-out-of-docker)",
            "image": v2_image(),
            "task_endpoint": endpoint,
            "agent_phase_allowlist": [endpoint_host(endpoint)],
            "egress_sidecar": {
                "proxy": f"gost (Harbor {HARBOR_VERSION} harbor-docker-egress-control-sidecar, "
                         "red handler with HTTP sniffing)",
                "read_timeout": EGRESS_SIDECAR_READ_TIMEOUT,
                "sidecar_read_timeout_s": EGRESS_SIDECAR_READ_TIMEOUT_S,
                "hidden_cap_removed": True,
                "replaces": "gost's default readTimeout of 15 s for the upstream's response "
                            "headers (go-gost/x v0.10.9), which cut every non-streamed model "
                            "reply that took longer than 15 s",
            },
            "sandbox_api_key": SANDBOX_API_KEY,
            **({"key_forwarder": {"listens": endpoint, "relays_to": base_url,
                                  "forwards": "GET and POST requests for the /v1/ paths "
                                              f"{', '.join(forwarder.allowed_paths)}, to "
                                              f"{forwarder.upstream} only (no redirects)",
                                  "adds": "Authorization: Bearer <the endpoint key>"}}
               if forwarder else {}),
            **({"endpoint_relay": {"listens": endpoint, "relays_to": base_url,
                                   "forwards": "GET and POST requests for the /v1/ paths "
                                               f"{', '.join(relay.allowed_paths)}, to "
                                               f"{relay.upstream} only (no redirects)",
                                   "adds": "nothing (the endpoint has no key)"}}
               if relay is not None and forwarder is None else {}),
            "regrade": REGRADE_IMPORT,
        },
        "agent": {
            "scaffold": "mini-swe-agent (in-sandbox, Harbor LockedMiniSwe)",
            "import": AGENT_IMPORT,
            "version": MINI_SWE_AGENT_VERSION,
            "config": AGENT_CONFIG,
            "config_sha256": cfg_sha,
            "gbench_config_additions": {"model.model_kwargs.extra_body.chat_template_kwargs."
                                        "enable_thinking": bool(enable_thinking),
                                        "model.model_kwargs.temperature": temperature,
                                        **{f"model.model_kwargs.{k}": v
                                           for k, v in client_kwargs.items()}},
            "model": f"openai/{model_name}",
            "container_thread_cap": threads or None,
        },
        "agent_output_dir": workdir,
    }
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    if attempts > 1:
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "one trial per task: the re-grade scores one patch per task, so an "
                           "@k would be fabricated"}
    if shard:
        result["shard"] = f"{shard[0]}/{shard[1]}"

    noncanon: List[str] = []
    if cfg_sha != AGENT_CONFIG_SHA256:
        noncanon.append(f"{AGENT_CONFIG} in the image has sha256 {cfg_sha[:12]}..., not the "
                        f"pinned {AGENT_CONFIG_SHA256[:12]}...")
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    if shard:
        noncanon.append(f"shard {shard[0]}/{shard[1]} of the task set")
    noncanon.append(f"temperature={temperature} "
                    f"{swe_agentic._temperature_origin(temperature_source)}; the locked "
                    "config sends none")
    if run_max_tokens:
        rec, why = swe_agentic.agent_max_tokens_record(run_max_tokens, "the V2 locked config",
                                                       "the relay's body fields")
        result["official_sampling"] = {"agent_max_tokens_from_run": rec}
        noncanon.append(why)
    if not enable_thinking:
        noncanon.append("the locked config asks for reasoning_effort=high; this run has the "
                        "model's thinking off (run with --thinking)")

    if not selected:
        result.update(total_questions=0, correct_answers=0, failed_requests=0, accuracy=None,
                      category_accuracy={}, sample_traces=[])
        if shard:
            # An empty shard: nothing was measured on this node, so no score (not 0%).
            swe_agentic.mark_empty_shard(result, shard, len(rows))
        else:
            result["status"] = "error"
            result["error"] = "no tasks were selected, so nothing was evaluated"
            withhold_on_error(result)
        swe_agentic.add_noncanon(result, "; ".join(noncanon + ["no tasks evaluated"]))
        result["duration_s"] = round(time.time() - start, 2)
        return result

    include = None if full else selected
    logger.info("[swe_bench_pro] V2 agent job: %d task(s), %d concurrent, endpoint %s, jobs in %s",
                len(selected), concurrency, endpoint, jobs_dir)

    def _agent_job(name: str, ids: Optional[List[str]]) -> Tuple[int, str]:
        return _run_harbor(workdir, agent_job_args(
            model_name=model_name, endpoint=endpoint, config_path=cfg_path, jobs_dir=jobs_dir,
            concurrency=concurrency, include=ids, threads=threads, job=name), name)

    jobs_run: Dict[str, Dict[str, Any]] = {}
    rc, tail_agent = _agent_job("agent", include)
    trials = job_trials(os.path.join(jobs_dir, "agent"))
    jobs_run["agent"] = {"returncode": rc, "trials": len(trials)}
    retried: List[str] = []
    for attempt in range(1, INFRA_RETRIES + 1):
        # A task whose agent phase never started (sandbox, image or agent install failure) never
        # reached the model, so running it again samples nothing twice. The V2 release gate
        # re-ran such trials the same way.
        redo = [i for i in selected if not (trials.get(i) or {}).get("agent_started")]
        if not redo:
            break
        name = f"agent_retry{attempt}"
        rc, tail = _agent_job(name, redo)
        again = job_trials(os.path.join(jobs_dir, name))
        trials.update({i: t for i, t in again.items() if i in redo})
        jobs_run[name] = {"returncode": rc, "trials": len(again)}
        retried += [i for i in redo if i not in retried]
        tail_agent = tail or tail_agent
    views = {iid: agent_trial_view(trials.get(iid)) for iid in selected}
    for iid in selected:
        infra_why = budget_kill_verdict(views[iid], relay)
        if infra_why:
            views[iid].update(model_outcome=False, why=infra_why)
    runnable = [iid for iid in selected if views[iid]["model_outcome"]]

    # patch_replay reads one source job per run, so re-grade per agent job.
    by_source: Dict[str, List[str]] = {}
    for iid in runnable:
        by_source.setdefault(os.path.dirname(trials[iid]["trial_dir"]), []).append(iid)
    regrade_trials: Dict[str, Dict[str, Any]] = {}
    tail_regrade = ""
    for source, ids in sorted(by_source.items()):
        base = "regrade" if os.path.basename(source) == "agent" else \
            f"regrade_{os.path.basename(source)}"
        todo = ids
        for attempt in range(INFRA_RETRIES + 1):
            name = base if attempt == 0 else f"{base}_retry{attempt}"
            logger.info("[swe_bench_pro] V2 re-grade job %s: %d task(s)", name, len(todo))
            rc, tail_regrade = _run_harbor(workdir, regrade_job_args(
                source_job=source, jobs_dir=jobs_dir, concurrency=concurrency,
                include=None if len(todo) == V2_TASK_COUNT else todo, threads=threads,
                job=name), name)
            got = job_trials(os.path.join(jobs_dir, name))
            regrade_trials.update({i: t for i, t in got.items() if i in todo})
            jobs_run[name] = {"returncode": rc, "trials": len(got)}
            # Only infrastructure failures are re-graded again; a verifier that timed out or
            # left no reward on the patched tree is the patch's verdict (regrade_verdict).
            todo = [i for i in todo if regrade_verdict(regrade_trials.get(i))[1] is not None]
            if not todo:
                break
    result["harbor_jobs"] = jobs_run
    if retried:
        result["infra_retried_ids"] = retried
    if relay is not None:
        result["relay"] = relay.record()
    return _assemble_v2(result, tasks, views, regrade_trials, hard_ids, noncanon, full, start,
                        tail_agent, tail_regrade)


def _assemble_v2(result: Dict[str, Any], tasks: List[Any], views: Dict[str, Dict[str, Any]],
                 regrade_trials: Dict[str, Dict[str, Any]], hard_ids: set, noncanon: List[str],
                 full: bool, start: float, tail_agent: str = "",
                 tail_regrade: str = "") -> Dict[str, Any]:
    """Traces, headline, in-sandbox and HARD-51 rates, status and comparability.

    A task with no scoreable trial is a harness error, left out of every rate's numerator and
    denominator: `agent_infra` when its rollout was not the model's own outcome (the sandbox,
    image or agent install failed, the agent phase never ran, no patch or trajectory was
    captured, the in-sandbox agent ended for a non-model reason), `no_regrade_verdict` when its
    re-grade gave no verdict (regrade_verdict). The in-sandbox rate is over the tasks whose
    rollout was the model's outcome. Nothing scored withholds the headline (None, status
    error)."""
    selected = [t.instance_id for t in tasks]
    traces: List[Dict[str, Any]] = []
    resolved, in_sandbox, infra = set(), set(), []
    scored_ids, rollout_ids = set(), set()
    # rollouts whose in-sandbox verifier failed for an infrastructure reason: no in-sandbox
    # verdict, left out of the in-sandbox rate (the re-grade is the headline either way)
    in_sandbox_infra: Dict[str, str] = {}
    for idx, t in enumerate(tasks):
        v = views[t.instance_id]
        rg = regrade_trials.get(t.instance_id)
        reward, regrade_why = regrade_verdict(rg)
        scoreable = bool(v["model_outcome"] and regrade_why is None)
        why = v.get("why")
        if v["model_outcome"] and not scoreable:
            why = regrade_why
        ok = scoreable and reward >= 1.0
        if ok:
            resolved.add(t.instance_id)
        if v["model_outcome"]:
            in_reward, in_why = in_sandbox_verdict(v)
            if in_why is None:
                rollout_ids.add(t.instance_id)
                if in_reward >= 1.0:
                    in_sandbox.add(t.instance_id)
            else:
                in_sandbox_infra[t.instance_id] = in_why
        if scoreable:
            scored_ids.add(t.instance_id)
        else:
            infra.append(t.instance_id)
        trace = {
            "sample_idx": idx, "source_sample_idx": idx, "attempt_idx": 0,
            "category": t.category,
            "messages": [{"role": "user", "content": t.problem_statement}],
            "extra_payload": {"instance_id": t.instance_id, "hard": t.instance_id in hard_ids},
            "gold_answer": "",
            "response_text": v.get("patch") or "",
            "agent_patch": v.get("patch") or "",
            "agent_exit_status": v.get("exit_status"),
            "agent_steps": v.get("n_calls") or 0,
            "agent_error": None if scoreable else why,
            "agent_infra_error": not scoreable,
            "agent_image": t.image,
            "trajectory_file": v.get("trajectory_file"),
            "in_sandbox_reward": v.get("reward"),
            "regrade_reward": rg.get("reward") if rg else None,
            "regrade_exception": rg.get("exception") if rg else None,
            # Harbor's own label of the agent trial's exception (its _classify_exec_error reads
            # it from the agent's output, which carries the model's command output too), recorded
            # as Harbor wrote it; the attribution above does not rely on it.
            "agent_harbor_exception": v.get("exception"),
            **({"budget_kill": v["budget_kill"]} if v.get("budget_kill") else {}),
            "is_correct": ok,
            "status": "OK" if scoreable else "AGENT_INFRA_ERROR",
        }
        if not scoreable:
            mark_harness_error(trace, AGENT_INFRA if not v["model_outcome"]
                               else NO_REGRADE_VERDICT, why)
        traces.append(trace)
    total = len(selected)
    head = scored_headline(traces)
    hard_sel = [i for i in selected if i in hard_ids]
    # Re-grades scored 0 because the verifier ran on the patched tree and timed out or left no
    # reward (regrade_verdict): listed so a reader can tell them from failing tests.
    verifier_zero = {t.instance_id: regrade_trials[t.instance_id].get("exception")
                     for t in tasks if views[t.instance_id]["model_outcome"]
                     and (regrade_trials.get(t.instance_id) or {}).get("reward") is None
                     and regrade_verdict(regrade_trials.get(t.instance_id))[1] is None}
    if verifier_zero:
        result["regrade_verifier_exception_ids"] = verifier_zero
    result.update({
        "total_questions": total,
        "correct_answers": len(resolved),
        # the headline's denominator: the tasks with a re-grade verdict on the model's patch
        "scored_instances": head["scored_instances"],
        "scoring_excluded": head["scoring_excluded"],
        "failed_requests": len(infra),
        "accuracy": head["accuracy"],
        "category_accuracy": head["category_accuracy"],
        "headline": "re-graded resolved rate (patch_replay on a pristine image), V2",
        "in_sandbox": _rate(selected, in_sandbox, rollout_ids),
        # rollouts left out of the in-sandbox rate: their verifier failed for an infrastructure
        # reason (no in-sandbox verdict)
        "in_sandbox_verifier_infra": dict(sorted(in_sandbox_infra.items())),
        "hard51": {"regraded": _rate(hard_sel, resolved, scored_ids),
                   "in_sandbox": _rate(hard_sel, in_sandbox, rollout_ids),
                   "complete": len(hard_sel) == HARD_TASK_COUNT},
        "sample_traces": traces,
        "status": "success",
    })
    record_harness_errors(result, head)
    if infra and len(infra) == total:
        first = next((tr for tr in traces if tr["agent_error"]), {})
        result["status"] = "error"
        result["error"] = (
            f"no task produced a scoreable trial (e.g. {first.get('extra_payload', {}).get('instance_id')}: "
            f"{first.get('agent_error')}); this is an infrastructure failure, not a 0% resolved "
            f"rate. Harbor output tail: {(tail_regrade or tail_agent)[-600:]}")
        noncanon.append("infrastructure failure (no scoreable task)")
    elif infra:
        result["status"] = "completed_with_errors"
        result["partial_evaluation"] = {"total_instances": total,
                                        "scoreable_instances": total - len(infra),
                                        "infra_error_instances": len(infra),
                                        "infra_error_ids": infra}
        noncanon.append(f"partial: {len(infra)} of {total} task(s) produced no scoreable trial "
                        f"(harness errors: {harness_errors_text(head['harness_errors'])}), left "
                        "out of the score")
    withhold_on_error(result)
    if result.get("relay"):
        # Endpoint failures the relay counted: harness errors, never the model's outcome; the
        # run is completed_with_errors and not comparable (a rollout one of them ended is an
        # agent_infra trace: budget_kill_verdict, or an exit status naming the exception).
        noncanon.extend(swebench_common.relay_failure_reasons(result, result["relay"]))
    if not full:
        result["hard51"]["note"] = "subset run: HARD-51 is measured on the selected tasks only"
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    result["duration_s"] = round(time.time() - start, 2)
    return result


# ============================================================================================
# V2: one-shot (opt-in), re-graded by the same patch_replay job
# ============================================================================================
def _extract_patch(text: str) -> str:
    """Delegate to the shared extractor (it strips the thinking channel and a closing fence)."""
    from .swebench_common import extract_patch
    return extract_patch(strip_thinking_tags(text or ""))


def _load_swe_bench_pro_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """V2 tasks as one-shot samples; raises on load/schema failure. --shard is applied to the
    full task list first (base.apply_shard), then the --eval-limit draw (stratified by repo,
    seed "swe_bench_pro") within the shard; the caller hands the result to run_eval_suite with
    swebench_common.presharded, so base keeps exactly these samples."""
    rows, _shard_label = apply_shard(load_v2_rows(), "swe_bench_pro")
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("repo"), seed="swe_bench_pro")
    samples = []
    for item in rows:
        parts = [f"Repository: {item.get('repo')}", "", f"Issue:\n{item['problem_statement']}"]
        if item.get("requirements"):
            parts += ["", f"Requirements:\n{item['requirements']}"]
        if item.get("interface"):
            parts += ["", f"Interface:\n{item['interface']}"]
        parts += ["", PATCH_OUTPUT_INSTRUCTION]
        samples.append(([{"role": "user", "content": "\n".join(parts)}], item.get("patch") or "",
                        {"category": str(item.get("repo_language") or item.get("repo")),
                         "instance_id": item["instance_id"]}))
    logger.info(f"Loaded {len(samples)} swe_bench_pro samples.")
    return samples


def write_replay_source(source_job: str, patches: Dict[str, str]) -> None:
    """A job directory patch_replay reads: <task>__oneshot/result.json + agent/model.patch."""
    for iid, patch in patches.items():
        tdir = os.path.join(source_job, f"{iid}__oneshot")
        os.makedirs(os.path.join(tdir, "agent"), exist_ok=True)
        with open(os.path.join(tdir, "result.json"), "w", encoding="utf-8") as f:
            json.dump({"task_name": iid}, f)
        with open(os.path.join(tdir, "agent", "model.patch"), "w", encoding="utf-8") as f:
            f.write(patch or "")


def _make_v2_oneshot_scorer(concurrency: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = swe_agentic.agent_output_dir("swe_bench_pro")
        jobs_dir = os.path.join(workdir, "jobs")
        patches = {}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if iid:
                patches[iid] = trace_patch(tr, _extract_patch)
        source = os.path.join(jobs_dir, "oneshot")
        write_replay_source(source, patches)
        threads = swe_thread_cap.resolve_threads(concurrency)
        trials: Dict[str, Dict[str, Any]] = {}
        todo, rc, tail, jobs_run = list(patches), 0, "", {}
        for attempt in range(INFRA_RETRIES + 1):
            # Same rule as the agentic re-grade: only infrastructure failures run again.
            name = "regrade" if attempt == 0 else f"regrade_retry{attempt}"
            rc, tail = await asyncio.to_thread(_run_harbor, workdir, regrade_job_args(
                source_job=source, jobs_dir=jobs_dir, concurrency=concurrency,
                include=todo, threads=threads, job=name), name)
            got = job_trials(os.path.join(jobs_dir, name))
            trials.update({i: t for i, t in got.items() if i in todo})
            jobs_run[name] = {"returncode": rc, "trials": len(got)}
            todo = [i for i in todo if regrade_verdict(trials.get(i))[1] is not None]
            if not todo:
                break
        verdicts = {i: regrade_verdict(trials.get(i)) for i in patches}
        missing = [i for i, (_, why) in verdicts.items() if why is not None]
        metrics["swe_bench_pro_report"] = {
            "submitted_instances": len(patches),
            "submitted_with_patch": sum(1 for p in patches.values() if (p or "").strip()),
            "regraded_instances": len(patches) - len(missing),
            "not_regraded_ids": missing, "harbor_returncode": rc, "harbor_jobs": jobs_run,
            "jobs_dir": jobs_dir}
        if missing and len(missing) == len(patches):
            metrics["swe_bench_pro_error"] = (
                f"the V2 re-grade produced no verdict for any of {len(patches)} task(s) "
                f"(e.g. {missing[0]}: {verdicts[missing[0]][1]}); this is a harness failure, not "
                f"a 0% resolved rate. Harbor output tail: {tail[-600:]}")
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            reward, why = verdicts.get(iid, (None, None))
            tr["is_correct"] = bool(reward is not None and reward >= 1.0)
            tr["status"] = "OK"
            if iid in patches and why is not None:
                # No re-grade verdict on the patch (after the retry): a harness error, left out
                # of the score.
                mark_harness_error(tr, NO_REGRADE_VERDICT, why, status="HARNESS_ERROR")
    return _score


def _run_v2_oneshot(model_name: str, base_url: str, concurrency: int, enable_thinking: bool,
                    **kwargs: Any) -> Dict[str, Any]:
    # The re-grade's sandboxes run the tests over each patch with network access: the gate runs
    # before any model call.
    require_sandbox_network_blocked()
    samples = _load_swe_bench_pro_samples(limit=kwargs.get("limit"))
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="swe_bench_pro",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=swebench_common.presharded(samples),   # already this node's shard
        async_eval_fn=_make_v2_oneshot_scorer(concurrency, metrics),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)
    result["dataset"] = V2_DATASET_LABEL
    noncanon = ["single-turn generation, not the V2 locked agentic protocol"]
    rep = metrics.get("swe_bench_pro_report") or {}
    if metrics.get("swe_bench_pro_error"):
        result["status"] = "error"
        result["error"] = metrics["swe_bench_pro_error"]
        noncanon.append("harness error (no valid resolved rate)")
    elif rep.get("not_regraded_ids"):
        result["partial_evaluation"] = {"submitted_instances": rep.get("submitted_instances"),
                                        "not_regraded_instances": len(rep["not_regraded_ids"])}
        if result.get("status") == "success":
            result["status"] = "completed_with_errors"
        noncanon.append(f"partial: {len(rep['not_regraded_ids'])} of "
                        f"{rep.get('submitted_instances')} task(s) got no re-grade verdict "
                        "(harness errors), left out of the score")
    if kwargs.get("limit"):
        noncanon.append(f"subset run (--eval-limit {kwargs.get('limit')})")
    record_harness_errors(result, scored_headline(result.get("sample_traces") or []))
    withhold_on_error(result)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return swe_agentic.mark_oneshot(result, "swe_bench_pro")


# ============================================================================================
# v1 harness interface (maintained engine and plugin variants)
# ============================================================================================
def _harness_dir() -> Optional[str]:
    """The scaleapi/SWE-bench_Pro-os clone the v1 harness runs from."""
    return prereqs_path("SWE-bench_Pro-os",
                        suite_env("GBENCH_SWE_BENCH_PRO_HARNESS_DIR", "SWE_BENCH_PRO_HARNESS_DIR"))


#: Columns `swe_bench_pro_eval.py` reads off the raw-sample frame (lines 96-99, 556-557).
_RAW_SAMPLE_COLUMNS = ("instance_id", "base_commit", "before_repo_set_cmd",
                       "selected_test_files_to_run", "fail_to_pass", "pass_to_pass")


def load_pinned_rows() -> List[Dict[str, Any]]:
    """Every row of the pinned v1 test split (`_V1_REVISION`): the rows behind the v1 harness
    raw-sample table. Raises on a load failure, an empty split, or a V2 row (which the v1
    harness would score with no hidden tests)."""
    try:
        from datasets import load_dataset
        rows = [dict(r) for r in load_dataset(_DATASET, _V1_CONFIG, split="test",
                                              revision=_V1_REVISION)]
    except Exception as e:
        logger.error(f"Failed to load dataset for swe_bench_pro: {e}")
        raise RuntimeError(f"Could not load dataset for swe_bench_pro ({V1_DATASET_LABEL}): "
                           f"{e}") from e
    if not rows:
        raise RuntimeError("Dataset for swe_bench_pro returned empty rows")
    v2 = [r.get("instance_id") for r in rows
          if r.get("docker_image") or _V2_MARKER in str(r.get("before_repo_set_cmd") or "")]
    if v2:
        raise RuntimeError(
            f"swe_bench_pro: {len(v2)} row(s) are SWE-bench Pro V2 (e.g. {v2[0]}: docker_image "
            f"or {_V2_MARKER}), which Scale's v1 harness cannot score; expected the v1 set "
            f"{V1_DATASET_LABEL}")
    return rows


def _read_raw_sample_table(path: str):
    """The table as the v1 harness reads it (swe_bench_pro_eval.py main: `.jsonl` through
    pd.read_json(lines=True), anything else through pd.read_csv), every value as a string."""
    import pandas as pd
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True, dtype=str).fillna("")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _raw_sample_mismatch(path: str, rows: List[Dict[str, Any]]) -> str:
    """Why the table at `path` is not the pinned rows ('' when it is): same instance set, and
    the same value in every column the harness reads."""
    try:
        df = _read_raw_sample_table(path)
    except Exception as e:
        kind = "JSON lines" if path.endswith(".jsonl") else "CSV"
        return f"cannot be read as {kind} ({type(e).__name__}: {e})"
    missing = [c for c in _RAW_SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        return f"lacks the harness columns {missing}"
    table = {r["instance_id"]: r for r in df.to_dict("records")}
    want = {r["instance_id"]: r for r in rows}
    if len(table) != len(df):
        return f"repeats {len(df) - len(table)} instance id(s)"
    if set(table) != set(want):
        return (f"has a different instance set ({len(set(want) - set(table))} pinned instance(s) "
                f"missing, {len(set(table) - set(want))} extra)")
    for iid, row in want.items():
        for col in _RAW_SAMPLE_COLUMNS:
            if str(table[iid][col]) != str(row.get(col) or ""):
                return f"differs from the pinned rows in {col} for {iid}"
    return ""


def _resolve_raw_sample() -> Tuple[Optional[str], str]:
    """(path to the v1 harness' `--raw_sample_path` table, '') or (None, why not).

    `scaleapi/SWE-bench_Pro-os` documents `swe_bench_pro_full.csv` in its README but does not
    ship it. Every column the harness reads is a column of the dataset, already in the
    string-of-python-literal form its `eval()` calls expect, so gbench writes the table from
    the pinned v1 rows (`load_pinned_rows`) as `RAW_SAMPLE_NAME` in the harness directory, or
    the temp dir when the clone is read-only. A table is only ever used after it is checked
    against the pinned rows; one that differs is rewritten. `GBENCH_SWE_BENCH_PRO_RAW_SAMPLE`
    points at the operator's own copy (`.csv`, or `.jsonl` as the harness also reads), which is
    checked the same way and never rewritten.
    """
    explicit = suite_env("GBENCH_SWE_BENCH_PRO_RAW_SAMPLE", "SWE_BENCH_PRO_RAW_SAMPLE")
    hd = _harness_dir()
    if not explicit and not hd:
        return None, "the harness directory is not set"
    try:
        rows = load_pinned_rows()
    except RuntimeError as e:
        return None, (f"the harness raw-sample table cannot be built or checked without the "
                      f"pinned dataset rows (needs the dataset cached or network access): {e}")
    if explicit:
        if not os.path.isfile(explicit):
            return None, f"GBENCH_SWE_BENCH_PRO_RAW_SAMPLE={explicit} does not exist"
        problem = _raw_sample_mismatch(explicit, rows)
        if problem:
            return None, (f"GBENCH_SWE_BENCH_PRO_RAW_SAMPLE={explicit} is not the pinned "
                          f"{V1_DATASET_LABEL} raw-sample table: it {problem}")
        return explicit, ""

    legacy = os.path.join(hd, "swe_bench_pro_full.csv")
    if os.path.isfile(legacy):
        logger.info("swe_bench_pro: ignoring %s (not tied to the pinned revision); the harness "
                    "reads %s", legacy, RAW_SAMPLE_NAME)
    targets = (os.path.join(hd, RAW_SAMPLE_NAME),
               os.path.join(tempfile.gettempdir(), RAW_SAMPLE_NAME))
    for target in targets:
        if os.path.isfile(target) and not _raw_sample_mismatch(target, rows):
            return target, ""

    import pandas as pd
    df = pd.DataFrame(rows)
    missing = [c for c in _RAW_SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("swe_bench_pro: %s is missing the harness columns %s; the raw-sample "
                       "table cannot be generated.", V1_DATASET_LABEL, missing)
        return None, f"{V1_DATASET_LABEL} is missing the harness columns {missing}"
    for target in targets:
        tmp = f"{target}.tmp.{os.getpid()}"
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, target)
            logger.info("swe_bench_pro: wrote the raw-sample table (%d instances, %s) to %s",
                        len(df), V1_DATASET_LABEL, target)
            return target, ""
        except OSError as e:
            logger.info("swe_bench_pro: could not write %s (%s)", target, e)
    return None, (f"the harness raw-sample table could not be written to {targets[0]} or "
                  f"{targets[1]}")


def raw_sample_path() -> Optional[str]:
    """Path to the v1 harness' `--raw_sample_path` table for the pinned v1 set (generated if
    needed), or None when it is unavailable; see `_resolve_raw_sample`. The default V2 path does
    not use it; the separately maintained engine and plugin variants do."""
    return _resolve_raw_sample()[0]


def check_swe_bench_pro_prerequisites() -> Tuple[bool, str]:
    """v1 harness prerequisites (the maintained engine and plugin variants): Docker + docker
    SDK + the Pro harness checkout + the v1 raw-sample table."""
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    try:
        import docker  # noqa: F401
    except ImportError:
        return False, ("Python 'docker' SDK is not installed (pip install -e \".[evals]\" from "
                       "the gbench checkout).")
    hd = _harness_dir()
    if not hd or not os.path.isfile(os.path.join(hd, "swe_bench_pro_eval.py")) \
            or not os.path.isdir(os.path.join(hd, "run_scripts")):
        return False, ("SWE-bench Pro harness not found: set GBENCH_SWE_BENCH_PRO_HARNESS_DIR to a clone of "
                       "scaleapi/SWE-bench_Pro-os (needs swe_bench_pro_eval.py and run_scripts/).")
    path, why = _resolve_raw_sample()
    if path is None:
        return False, (f"the harness raw-sample table is unavailable: {why}. gbench writes "
                       f"{RAW_SAMPLE_NAME} from {V1_DATASET_LABEL}; set "
                       "GBENCH_SWE_BENCH_PRO_RAW_SAMPLE to point at your own copy of that set.")
    return True, ""


def _make_scorer(model_name: str, num_workers: int, metrics: Dict[str, Any]):
    """v1 harness scorer (swe_bench_pro_eval.py against `raw_sample_path()`), kept for plugin
    variants that score single-turn patches on the v1 set. The default V2 path does not use
    it."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        hd = _harness_dir()
        workdir = tempfile.mkdtemp(prefix="gbench_swepro_")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        preds_path = os.path.join(workdir, "preds.json")
        prefix = "gbench__" + re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)[:40]
        # Agentic traces carry the agent's submission; an empty one is the model's recorded
        # outcome (step limit, context window), not an extraction failure.
        agentic = any("agent_patch" in tr for tr in sample_traces)

        preds = []
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if iid:
                preds.append({"instance_id": iid, "patch": trace_patch(tr, _extract_patch),
                              "prefix": prefix})
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)

        raw_sample = raw_sample_path()
        proc = None
        if raw_sample is None:
            logger.error("swe_bench_pro: no raw-sample table for the harness (see "
                         "check_swe_bench_pro_prerequisites for why)")
            metrics["swe_bench_pro_report"] = {"error": "harness raw-sample table unavailable"}
        else:
            cmd = [
                # sys.executable, NOT bare "python": the scorer imports pandas + docker (in the
                # gbench venv). Bare "python" resolves via PATH and, launched from a shell
                # without that venv first, silently ModuleNotFdErrors -> no eval_results.json ->
                # every instance False.
                sys.executable, os.path.join(hd, "swe_bench_pro_eval.py"),
                f"--raw_sample_path={raw_sample}",
                f"--patch_path={preds_path}", f"--output_dir={out_dir}",
                f"--scripts_dir={os.path.join(hd, 'run_scripts')}",
                f"--num_workers={max(1, num_workers)}",
                "--dockerhub_username=jefzda", "--use_local_docker",
            ]
            cmd, threads = swe_thread_cap.apply(cmd, num_workers, "swe_bench_pro")
            metrics["docker_thread_cap"] = threads or None
            # The v1 harness runs the tests over each patch in a container with network access.
            await asyncio.to_thread(require_v1_network_blocked)

            def _run():
                return subprocess.run(cmd, cwd=hd, capture_output=True, text=True)
            proc = await asyncio.to_thread(_run)

        results: Dict[str, bool] = {}
        report_path = os.path.join(out_dir, "eval_results.json")
        if proc is not None and os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                results = {k: bool(v) for k, v in json.load(f).items()}
            # Count patches that were actually submittable: "the model emitted no patch" is not
            # "the patch did not resolve the issue".
            empty = sum(1 for pr in preds if not (pr.get("patch") or "").strip())
            # The harness drops (with a printed warning) any patch whose instance is not in the
            # raw-sample table; such an instance was never evaluated, which is not unresolved.
            not_evaluated = [pr["instance_id"] for pr in preds if pr["instance_id"] not in results]
            metrics["swe_bench_pro_report"] = {
                "total_instances": len(results),
                "resolved_instances": sum(results.values()),
                "empty_patch_instances": empty,
                "submitted_with_patch": len(preds) - empty,
                "submitted_instances": len(preds),
                "not_evaluated_instances": len(not_evaluated),
                "not_evaluated_ids": not_evaluated,
            }
        elif proc is not None:
            logger.error("swe_bench_pro: eval_results.json not found. stderr tail: %s",
                         (proc.stderr or "")[-800:])
            metrics["swe_bench_pro_report"] = {"error": "harness report not produced"}

        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = bool(results.get(iid, False))
            tr["status"] = "OK"

        rep = metrics.get("swe_bench_pro_report") or {}
        if rep.get("error"):
            # No report at all (harness crash) -> promote to a run-level error, never a 0%.
            metrics["swe_bench_pro_error"] = rep["error"]
        elif rep.get("submitted_instances") and \
                rep.get("not_evaluated_instances") == rep.get("submitted_instances"):
            metrics["swe_bench_pro_error"] = (
                "the Pro harness evaluated none of the {} submitted instances (their ids are "
                "not in its raw-sample table); this is a harness failure, not a 0% resolved "
                "rate".format(rep["submitted_instances"]))
        elif not agentic and rep.get("submitted_with_patch") == 0 and rep.get("total_instances"):
            metrics["swe_bench_pro_error"] = (
                "no extractable patch was produced for any of {} instances (all empty); "
                "this measures the generation, not the resolved rate".format(
                    rep["total_instances"]))
        elif rep.get("resolved_instances") == 0 and rep.get("submitted_with_patch"):
            # The Pro harness collapses a failed image pull / docker error into False, so an
            # all-images-failed run looks identical to a genuine 0%. Its stderr markers say so.
            stderr = (proc.stderr or "")
            if ("Failed to pull or find image locally" in stderr
                    or "Error in eval_with_docker" in stderr):
                metrics["swe_bench_pro_error"] = (
                    "the Pro harness could not pull/run instance images (Docker/infra failure) so "
                    "0 of {} submitted patches were evaluated; this is a harness failure, not a 0% "
                    "resolved rate. stderr tail: {}".format(
                        rep.get("submitted_with_patch"), stderr[-400:]))
    return _score


def report_reasons(result: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
    """Apply the v1 scorer's harness verdict to `result` and return the non-comparability
    reasons: a harness error makes the run an error; instances the harness did not evaluate make
    it partial."""
    if metrics.get("swe_bench_pro_error"):
        result["status"] = "error"
        result["error"] = metrics["swe_bench_pro_error"]
        return ["harness error (no valid resolved rate)"]
    rep = metrics.get("swe_bench_pro_report") or {}
    missing = rep.get("not_evaluated_instances") or 0
    if not missing:
        return []
    partial = result.setdefault("partial_evaluation", {})
    partial.update({"harness_submitted_instances": rep.get("submitted_instances"),
                    "harness_not_evaluated_instances": missing})
    if result.get("status") == "success":
        result["status"] = "completed_with_errors"
    return [f"partial: {missing} of {rep.get('submitted_instances')} submitted instance(s) were "
            "not evaluated by the harness"]


# ============================================================================================
# entry point
# ============================================================================================
def run_swe_bench_pro(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench Pro (hard-errors if Docker or the V2 image are absent).

    Modes:
      * default: the V2 locked protocol (Harbor agent job, then the fresh-sandbox re-grade);
      * `GBENCH_SWE_BENCH_PRO_MODE=oneshot`: one diff per task with no repository access,
        re-graded the same way. Structurally near-0 and never leaderboard-comparable;
      * `GBENCH_SWEBENCH_PRO_AGENTIC=1`: the separately maintained engine, routed exactly as
        before (v1 harness prerequisites); its result is labelled with `ENGINE_V1_NOTE`.
    See docs/evals/swe_bench_pro.md.
    """
    mode, _ = swe_agentic.resolve_mode("swe_bench_pro")
    if os.getenv("GBENCH_SWEBENCH_PRO_AGENTIC") == "1":
        if mode == swe_agentic.ONESHOT:
            raise ValueError("GBENCH_SWEBENCH_PRO_AGENTIC=1 and GBENCH_SWE_BENCH_PRO_MODE=oneshot "
                             "ask for different protocols; unset one of them.")
        ok, reason = check_swe_bench_pro_prerequisites()
        if not ok:
            raise infra_required("swe_bench_pro", reason, DOCS_URL)
        # The engine's task containers (the model's commands) and the v1 harness's (the tests
        # over each patch) have network access: the cloud metadata server gate runs first.
        require_v1_network_blocked()
        from .swe_bench_pro_agentic import execute_agentic
        res = execute_agentic(model_name, base_url, concurrency=concurrency,
                              limit=kwargs.get("limit"), only_instance_ids=None,
                              eval_name="swe_bench_pro",
                              temperature=kwargs.get("temperature"),
                              max_output_tokens=kwargs.get("max_output_tokens"))
        if isinstance(res, dict):
            swe_agentic.add_noncanon(res, ENGINE_V1_NOTE)
        return res
    # No-skip: a missing Docker daemon, socket or image HARD-ERRORS rather than emit a skip row;
    # the reason says how to provision it.
    ok, reason = check_swe_bench_pro_v2_prerequisites()
    if not ok:
        raise infra_required("swe_bench_pro", reason, DOCS_URL)
    if mode == swe_agentic.AGENTIC:
        return run_swe_bench_pro_v2(model_name, base_url, concurrency, enable_thinking, **kwargs)
    return _run_v2_oneshot(model_name, base_url, concurrency, enable_thinking, **kwargs)
