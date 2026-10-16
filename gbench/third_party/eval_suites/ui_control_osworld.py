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

# Evaluation Suite: ui_control_osworld
# Description: OSWorld - execution-based desktop computer-use benchmark (369 tasks, deterministic verifier)

"""gbench native built-in runner for ui_control_osworld (Tool Use & Agentic Workflows).

Canonical OSWorld (github.com/xlang-ai/OSWorld): 369 real desktop tasks over an Ubuntu-desktop VM. A
computer-use agent observes screenshots and emits pyautogui actions over many steps; each task is
scored by a DETERMINISTIC per-task evaluator (getters read the final VM state, compare to a
reference) -> a binary/graded result.txt in [0,1]. NO LLM judge (so NO GEMINI dependency).

gbench DELEGATES to the pinned OSWorld harness inside a LOCAL orchestrator image
(gbench/docker/ui_control_osworld.Dockerfile), driven docker-OUT-of-docker: OSWorld's `docker`
provider (DesktopEnv(provider_name="docker")) spawns a SIBLING QEMU-VM container
(happysixd/osworld-docker) per task on the host daemon, bind-mounting the Ubuntu qcow2. The agent
(mm_agents PromptAgent) runs host-side in the orchestrator and calls the served model over /v1
(wired via OPENAI_BASE_URL + a model-name rewrite, since vLLM rejects the routing alias).

Headline `accuracy` = mean per-task success (result.txt) over the selected tasks that produced a
result, as OSWorld's own get_result averages the result.txt files present, less the tasks during
which a model request failed on the endpoint's side. A task with no result.txt is one whose run
raised in OSWorld's loop (the VM, the model endpoint or the evaluator failed); a task during which
a model request got no reply or HTTP 5xx, 429, 401, 403 or 404 (the launcher's per-example record,
`endpoint_failures`: OSWorld's PromptAgent answers such a failure with an empty reply and the task
still writes a result.txt) did not measure the model either. Both are harness errors, never the
model's outcome: left out of the headline and of the per-domain breakdown, counted (`n_missing`,
`n_endpoint_failed`, `scoring_excluded`), named in `status_reason` (status
`completed_with_errors`); with no task scored, `accuracy` is null with
`accuracy_withheld_reason`. The in-container summary's raw rate, which counts a missing task as 0,
is kept as `success_rate_missing_as_zero`. `leaderboard_comparable` is always False (self-hosted
agent, single trial, subset-capable; the OSWorld leaderboard is maintainer-run).

HARD-ERRORS (infra_required, never skips / never a fake number) if Docker, the orchestrator image
(at the current launcher: label org.gbench.ui-control-osworld-launcher=4), `/dev/kvm` (OSWorld's
VM only boots with hardware virtualization), GBENCH_OSWORLD_VM_DIR with the provisioned Ubuntu
qcow2, or the happysixd/osworld-docker task image at the pinned digest is missing (the check names
every one that is), if GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE names another image (OSWorld's docker
provider starts happysixd/osworld-docker by name), if GBENCH_OSWORLD_LIMIT is not a positive
integer, or if the model endpoint is unreachable. See docs/evals/ui_control_osworld.md.

Endpoint key: the orchestrator holds the docker socket and turns the model's replies into the
actions it sends to the task VM, so it gets only the placeholder GBENCH_MODEL_API_KEY=dummy. It
always reaches the endpoint through a swebench_common EndpointRelay on 127.0.0.1, which adds a
real key (GBENCH_OSWORLD_MODEL_API_KEY) when there is one (`model_endpoint_relay`; the endpoint
failures it counts make the run completed_with_errors).

Cloud metadata server: the model's actions run inside the task VM, whose container has network
access, so before the orchestrator starts, swebench_common.require_task_network_blocked checks with
a probe container of the orchestrator image that 169.254.169.254 is unreachable from docker's
default bridge and from a user-defined bridge network (infra_required otherwise).

Sampling: the temperature follows gbench's rule for every suite: 1.0 for a `--thinking` run
(temperature_source "thinking"), 0.0 without it (temperature_source "default"); `--temperature`,
or `GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE` (which takes precedence), overrides it. It reaches the
model twice over: as OSWorld run.py's own `--temperature` (OSWORLD_TEMPERATURE, which the
mm_agents PromptAgent puts in every request payload), and as a field the endpoint relay sets on
every request.
run.py's `--top_p` 0.9 is unchanged. The PromptAgent sends no reasoning/thinking field, so
`enable_thinking` is not applied here; the reasoning mode is the harness's own.

Concurrency: canonical OSWorld run.py runs tasks SEQUENTIALLY on a SINGLE DesktopEnv VM, so a per-task
concurrency/--sandboxes value does not apply (the parallel run_multienv*.py variants are agent-specific
and are not used here).

Sharding and --eval-limit: this suite bypasses base.run_eval_suite, so it validates GBENCH_SHARD
itself (base.current_shard: a malformed value is an error before anything starts) and forwards a
partial shard's normalized "I/N" into the orchestrator container; the launcher
(ui_control_osworld_run.py) selects shard I of N by round-robin over the DETERMINISTIC sorted
(domain, example_id) task list (flat[I-1::N]), then draws the --eval-limit tasks from that shard
with a seeded draw balanced across the domains (a copy of sampling.stratified_indices, seed
"ui_control_osworld"; `selection.limit_draw`), and rebuilds the --test_all_meta_path manifest so
the harness iterates exactly that subset. gbench refuses a GBENCH_OSWORLD_LIMIT that is not a
positive integer before anything starts (infra_required), and the launcher itself stops on a
malformed shard or a limit that is not an integer. The
result records the shard (`shard`, so the central stamp marks it applied), its comparability
reason (base.SHARD_REASON) and one row per selected task (`sample_traces`: `task`, `domain`,
`result`, a task without a result.txt a harness error); an orchestrator that does not report the
shard it applied, or (for a limited run) how it drew the subset, predates this and is refused
(infra_required: rebuild the image). A shard with none of the tasks does not start OSWorld and
reports `accuracy` null with `accuracy_withheld_reason` and `empty_shard`. merge_shards (gbench
--merge) re-scores the union of the shards' task rows.

Run directory: every run writes to a fresh directory, GBENCH_OSWORLD_VM_DIR/gbench_runs/<run id>
(`run_dir`: results/, the subset manifest, the summary), created mode 0700 for the invoking user
(the orchestrator runs as root in its container, which may write there) and handed back to that
user by the launcher at the end (OSWORLD_HOST_OWNER). OSWorld's run.py resumes from the
result.txt files its result_dir already holds (get_unfinished) and the routing alias puts every
model's results in the same place, so a shared result_dir would hand a run another run's results.
"""

import json
import logging
import os
import shutil
import subprocess
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import swebench_common
from .base import SHARD_REASON, current_shard, record_shard, resolve_temperature
from .limit_plan import eval_limit_plan
from .swebench_common import EndpointRelay, infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/ui_control_osworld.md"

#: What the orchestrator gets as GBENCH_MODEL_API_KEY: a placeholder. A real key
#: (GBENCH_OSWORLD_MODEL_API_KEY) stays with gbench's relay (model_endpoint_relay).
MODEL_KEY_PLACEHOLDER = "dummy"

_IMAGE_DEFAULT = "gbench-ui-control-osworld:fc31a904"
#: OSWorld's docker provider (desktop_env/providers/docker/provider.py at the pinned OSWorld
#: commit) starts the task VM container from this image by name, i.e. its `latest` tag, and pulls
#: it when it is missing; gbench never pulls it.
_TASK_IMAGE_DEFAULT = "happysixd/osworld-docker"
#: The manifest digest of that image gbench runs: Docker Hub happysixd/osworld-docker:latest
#: (the repository's only tag, pushed 2024-10-17; linux/amd64, image config
#: sha256:fe8d9a5e5ad6c593d059887ea2c790481b3f32dd42fa961f2441cdbbe2c70cf4). The prerequisite
#: check requires the task image to be this digest, pulled by digest and tagged with the name.
TASK_IMAGE_DIGEST = "sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9"
TASK_IMAGE_PINNED = f"{_TASK_IMAGE_DEFAULT}@{TASK_IMAGE_DIGEST}"
_FULL_TASK_COUNT = 369
#: How the launcher draws an --eval-limit subset (its summary's selection.limit_draw; an image
#: whose launcher does not report it predates the seeded draw and took the head of the list).
LIMIT_DRAW = "stratified_indices(key=domain, seed=ui_control_osworld)"
#: The image label of an orchestrator whose launcher (gbench/docker/ui_control_osworld_run.py and
#: ui_control_osworld_entrypoint.sh) is the current one: it writes into the run's own directory
#: (OSWORLD_RUN_DIR), stops on a malformed shard or limit, reports the shard it applied and how it
#: drew an --eval-limit subset, and does not start OSWorld for an empty shard. An image without the
#: label, or with another value, predates that launcher: it reads and writes a result_dir shared
#: across runs (OSWorld resumes from it) and takes the head of the task list for --eval-limit.
#: Bump it with every change to the launcher files (a test pins their fingerprint to this value).
LAUNCHER_LABEL = "org.gbench.ui-control-osworld-launcher"
#: Version 3 adds the per-example record of the model requests that failed on the endpoint's side
#: (gbench leaves those tasks out of the score) and hands the run directory to the invoking user.
#: Version 4 passes OSWORLD_TEMPERATURE (gbench's resolved temperature) as run.py's --temperature.
LAUNCHER_VERSION = "4"


def _image() -> str:
    return os.environ.get("GBENCH_UI_CONTROL_OSWORLD_IMAGE", _IMAGE_DEFAULT)


def _task_image() -> str:
    """The task VM image OSWorld's docker provider starts: always _TASK_IMAGE_DEFAULT, by name
    (the provider hardcodes it; task_image_override_refusal refuses another value of
    GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE)."""
    return _TASK_IMAGE_DEFAULT


def task_image_override_refusal() -> Optional[str]:
    """The reason to refuse GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE, or None when it is unset or
    names _TASK_IMAGE_DEFAULT: OSWorld's docker provider starts _TASK_IMAGE_DEFAULT by name, so
    another name would pass the digest check while the provider runs what that name points to."""
    raw = (os.environ.get("GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE") or "").strip()
    if not raw or raw == _TASK_IMAGE_DEFAULT:
        return None
    return (f"GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE={raw!r} cannot change the task VM image: "
            f"OSWorld's docker provider starts {_TASK_IMAGE_DEFAULT!r} by name (the pinned "
            f"{TASK_IMAGE_PINNED}, tagged with that name). Unset it.")


def _vm_dir() -> str:
    return os.environ.get("GBENCH_OSWORLD_VM_DIR", "").strip()


def _docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def _image_id(image: str) -> Optional[str]:
    """The image id docker reports for `image`, or None when docker cannot inspect it."""
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip() or None


def image_launcher_version(image: str) -> Optional[str]:
    """The LAUNCHER_LABEL value of `image` ("" when the label is absent), or None when docker
    cannot inspect it."""
    r = subprocess.run(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "' + LAUNCHER_LABEL + '"}}', image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    value = (r.stdout or "").strip()
    return "" if value == "<no value>" else value


def check_ui_control_osworld_prerequisites() -> Tuple[bool, str]:
    """Docker + orchestrator image at the current launcher (LAUNCHER_LABEL=LAUNCHER_VERSION) +
    /dev/kvm + provisioned qcow2 + the task image, at the pinned digest (TASK_IMAGE_PINNED). No
    judge/GEMINI."""
    image = _image()
    build = (f"Build the orchestrator LOCALLY (gbench never pulls it):\n"
             f"  docker build -t {image} -f gbench/docker/ui_control_osworld.Dockerfile gbench/docker\n"
             f"See " + DOCS_URL + " for the full provisioning recipe.")
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    # Everything else is checked in full, so one message names every prerequisite missing.
    missing: List[str] = []
    refusal = task_image_override_refusal()
    if refusal:
        missing.append(refusal)
    if not _docker_image_exists(image):
        missing.append(f"orchestrator image {image!r} not found. " + build)
    else:
        found = image_launcher_version(image)
        if found != LAUNCHER_VERSION:
            missing.append(f"orchestrator image {image!r} predates the current launcher (label "
                           f"{LAUNCHER_LABEL}={LAUNCHER_VERSION}; the image has "
                           f"{repr(found) if found else 'none'}): it would reuse a results "
                           f"directory shared across runs (OSWorld resumes from it), take the "
                           f"head of the task list for --eval-limit, or score a task whose model "
                           f"requests failed on the endpoint's side. Rebuild it. " + build)
    if not os.path.exists("/dev/kvm"):
        missing.append("/dev/kvm is not present - OSWorld's Ubuntu VM only becomes ready within "
                       "its 300s timeout with hardware virtualization (KVM). Run on a host with "
                       "nested virtualization enabled (bare metal or a nested-virt-licensed VM).")
    vm_dir = _vm_dir()
    if not vm_dir:
        missing.append("GBENCH_OSWORLD_VM_DIR is unset. Provision the Ubuntu VM disk once "
                       "(docker_vm_data/Ubuntu.qcow2 from HF xlangai/ubuntu_osworld, ~12GB) and "
                       "point GBENCH_OSWORLD_VM_DIR at the directory containing docker_vm_data/.")
    elif not os.path.isfile(os.path.join(vm_dir, "docker_vm_data", "Ubuntu.qcow2")):
        missing.append(f"{vm_dir!r} has no docker_vm_data/Ubuntu.qcow2 - download + unzip the "
                       f"OSWorld Ubuntu qcow2 there (HF xlangai/ubuntu_osworld).")
    provision = (f"Provision the pinned OSWorld VM runner with `docker pull {TASK_IMAGE_PINNED}` "
                 f"then `docker tag {TASK_IMAGE_PINNED} {_TASK_IMAGE_DEFAULT}` (OSWorld's docker "
                 f"provider starts {_TASK_IMAGE_DEFAULT!r} by name; gbench never pulls it).")
    if not _docker_image_exists(_TASK_IMAGE_DEFAULT):
        missing.append(f"the task VM image {_TASK_IMAGE_DEFAULT!r} is not on the host daemon. "
                       + provision)
    else:
        pinned_id = (_image_id(TASK_IMAGE_PINNED) if _docker_image_exists(TASK_IMAGE_PINNED)
                     else None)
        task_id = _image_id(_TASK_IMAGE_DEFAULT)
        if pinned_id is None or pinned_id != task_id:
            missing.append(f"the task VM image {_TASK_IMAGE_DEFAULT!r} on the host daemon is not "
                           f"the pinned {TASK_IMAGE_PINNED}. " + provision)
        else:
            _VERIFIED_TASK_IMAGE["id"] = task_id
    if missing:
        return False, " ".join(f"({i}) {m}" for i, m in enumerate(missing, 1)) + " See " + DOCS_URL
    return True, ""


#: The id of the task VM image the prerequisite check verified (recorded as task_image.id).
_VERIFIED_TASK_IMAGE: Dict[str, str] = {}


def osworld_limit(raw: Any) -> Optional[int]:
    """GBENCH_OSWORLD_LIMIT (or --eval-limit) as a positive int, None when unset or empty;
    infra_required for anything else (0, a negative number, not an integer), before any container
    starts: the launcher would run the whole task set for 0 or a negative value."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise infra_required("ui_control_osworld", f"GBENCH_OSWORLD_LIMIT / --eval-limit must be "
                             f"a positive integer (the number of tasks to run), got {text!r}; "
                             f"unset it to run every selected task.", DOCS_URL)
    return value


def _model_endpoint(base_url: str) -> str:
    """--network host -> the served model is at 127.0.0.1; just ensure a /v1 suffix."""
    override = os.environ.get("GBENCH_OSWORLD_MODEL_ENDPOINT")
    if override:
        return override
    ep = (base_url or "").rstrip("/")
    return ep if ep.endswith("/v1") else ep + "/v1"


def model_endpoint_key() -> Optional[str]:
    """The served endpoint's key (GBENCH_OSWORLD_MODEL_API_KEY), or None when it has none (unset,
    or the placeholders `dummy` / `EMPTY`)."""
    key = (os.environ.get("GBENCH_OSWORLD_MODEL_API_KEY") or "").strip()
    return key if key and key not in (MODEL_KEY_PLACEHOLDER, "EMPTY") else None


def model_endpoint_relay(model_endpoint: str, key: Optional[str],
                         temperature: Optional[float] = None) -> EndpointRelay:
    """swebench_common.EndpointRelay on 127.0.0.1 (the orchestrator runs on the host network) that
    adds `key` when there is one: the orchestrator, which holds the docker socket and turns the
    model's replies into the actions it sends to the task VM, gets only the relay's address and
    the placeholder, and the relay counts the endpoint's failures. `temperature` (the resolved
    one) is set on every model request."""
    body = {"temperature": float(temperature)} if temperature is not None else None
    try:
        relay = EndpointRelay(model_endpoint, "127.0.0.1", key=key, body_fields=body,
                              name="osworld-model-relay")
    except ValueError as e:
        raise infra_required("ui_control_osworld", f"the model endpoint {model_endpoint!r} is not "
                             f"an http(s) URL with a host ({e}); set GBENCH_OSWORLD_MODEL_ENDPOINT.",
                             DOCS_URL) from e
    try:
        return relay.start()
    except OSError as e:
        raise infra_required("ui_control_osworld", f"gbench's relay to the model endpoint could not "
                             f"listen on 127.0.0.1 ({type(e).__name__}: {e}).", DOCS_URL) from e


def _scored_task(t: Dict[str, Any]) -> bool:
    """A task row of the summary that counts in the score: a numeric result and no model request
    that failed on the endpoint's side."""
    return isinstance(t.get("result"), (int, float)) and not int(t.get("endpoint_failures") or 0)


def compute_ui_control_osworld_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The headline over the tasks that produced a result.txt and had no model request fail on
    the endpoint's side, from the in-container summary.

    With the summary's task rows (`tasks`), the rates are the mean result over those tasks,
    overall and per domain (None where none is scored). A summary without rows (an older
    launcher) gives the rates from its own: its `success_rate` (and each domain's) counts a task
    without a result as 0, over every selected task, so the rate over the scored tasks is rate x
    selected / scored. The summary's own rate is kept as success_rate_missing_as_zero. None when
    the summary has no rate at all."""
    if not summary or not isinstance(summary.get("success_rate"), (int, float)):
        return None
    tasks = [t for t in summary.get("tasks") or [] if isinstance(t, dict)]
    if tasks:
        per_domain_rows: Dict[str, List[Dict[str, Any]]] = {}
        for t in tasks:
            per_domain_rows.setdefault(str(t.get("domain")), []).append(t)
        per_domain = {}
        for domain, rows in sorted(per_domain_rows.items()):
            scored_rows = [float(t["result"]) for t in rows if _scored_task(t)]
            raw = (summary.get("per_domain") or {}).get(domain)
            per_domain[domain] = {
                "success_rate": (sum(scored_rows) / len(scored_rows)) if scored_rows else None,
                "success_rate_missing_as_zero": (raw.get("success_rate")
                                                 if isinstance(raw, dict) else None),
                "n": len(rows), "n_scored": len(scored_rows),
                "n_endpoint_failed": sum(1 for t in rows if int(t.get("endpoint_failures") or 0))}
        values = [float(t["result"]) for t in tasks if _scored_task(t)]
        return {"success_rate": (sum(values) / len(values)) if values else None,
                "success_rate_missing_as_zero": summary["success_rate"],
                "per_domain": per_domain}
    n_selected = int(summary.get("n_selected") or 0)
    n_scored = int(summary.get("n_scored") if summary.get("n_scored") is not None
                   else n_selected)

    def _over_scored(rate: Any, n: int, scored: int) -> Optional[float]:
        if not isinstance(rate, (int, float)):
            return None
        if not n:
            return float(rate)          # no task counts: nothing to correct for
        return float(rate) * n / scored if scored else None

    per_domain = {}
    for domain, d in (summary.get("per_domain") or {}).items():
        if not isinstance(d, dict):
            per_domain[domain] = d          # not the launcher's {success_rate, n, n_scored}
            continue
        d = dict(d)
        n_d = int(d.get("n") or 0)
        scored_d = int(d.get("n_scored") if d.get("n_scored") is not None else n_d)
        d["success_rate_missing_as_zero"] = d.get("success_rate")
        d["success_rate"] = _over_scored(d.get("success_rate"), n_d, scored_d)
        per_domain[domain] = d
    return {"success_rate": _over_scored(summary["success_rate"], n_selected, n_scored),
            "success_rate_missing_as_zero": summary["success_rate"],
            "per_domain": per_domain or summary.get("per_domain")}


def effective_limit(limit: Optional[int]) -> Tuple[Optional[int], bool]:
    """(the task limit the launcher gets, whether it came from GBENCH_OSWORLD_LIMIT). --eval-limit
    wins: with it set GBENCH_OSWORLD_LIMIT is ignored with a warning; without it the knob (a
    positive integer, else infra_required) applies."""
    knob = os.environ.get("GBENCH_OSWORLD_LIMIT", "").strip()
    if limit and int(limit) > 0:
        if knob:
            logger.warning("[ui_control_osworld] --eval-limit %d is set: GBENCH_OSWORLD_LIMIT=%s "
                           "is ignored (--eval-limit wins)", int(limit), knob)
        return osworld_limit(limit), False
    value = osworld_limit(knob or None)
    return value, value is not None


def osworld_plan(limit: Optional[int], domain: str, shard_label: Optional[str], *,
                 knob: bool = False) -> Dict[str, Any]:
    """Log the run's planned workload (one line) and return its `eval_limit_plan`: --eval-limit N
    is N tasks in total, drawn by the launcher across the domains (one flat task list, never N per
    domain), each run once, at most the _FULL_TASK_COUNT tasks of the full set. Without a limit
    the full set is _FULL_TASK_COUNT tasks; a domain or a shard narrows it to a count only the
    launcher's manifest knows (the record is corrected from it after the run: plan_from_summary).
    `knob`: the limit is GBENCH_OSWORLD_LIMIT's (no --eval-limit), which the line names."""
    if knob and limit:
        n = min(int(limit), _FULL_TASK_COUNT)
        logger.info("[ui_control_osworld] GBENCH_OSWORLD_LIMIT %d (no --eval-limit): %d sample(s)"
                    " x 1 attempt = %d episodes", int(limit), n, n)
        return {"limit": None, "limit_source": f"GBENCH_OSWORLD_LIMIT={int(limit)}",
                "samples": n, "per_setting": None,
                "attempts_per_sample": 1, "episodes": n}
    if limit or (not domain and not shard_label):
        return eval_limit_plan("ui_control_osworld", limit,
                               min(int(limit), _FULL_TASK_COUNT) if limit else _FULL_TASK_COUNT,
                               attempts=1, unit="episodes")
    logger.info("[ui_control_osworld] full run (no --eval-limit): every task of %s x 1 attempt",
                ", ".join(x for x in (f"domain {domain}" if domain else "",
                                      f"shard {shard_label}" if shard_label else "") if x))
    return {"limit": None, "samples": None, "per_setting": None, "attempts_per_sample": 1,
            "episodes": None}


def run_ui_control_osworld(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical OSWorld via the pinned harness (docker-out-of-docker QEMU-VM siblings)."""
    # `enable_thinking` and `concurrency` are accepted for a uniform runner signature but are NOT
    # forwarded to the harness (verified against the pinned OSWorld ref):
    #  - reasoning mode: OSWorld's mm_agents PromptAgent sends a plain OpenAI-compat payload (model,
    #    messages, max_tokens, top_p, temperature) with NO reasoning/thinking field, so enable_thinking
    #    has nothing to act on. The agent's reasoning mode IS the canonical protocol; gbench does not
    #    override it (honest-record via the "sampling" field below, not a fabricated "thinking" record).
    #  - concurrency: canonical OSWorld run.py runs tasks strictly SEQUENTIALLY on a SINGLE DesktopEnv
    #    VM (nested for-loop, no multiprocessing; the parallel run_multienv*.py variants are
    #    agent-specific and are not used here), so a per-task concurrency/--sandboxes value does not
    #    apply (documented via the "execution" field below, not silently dropped).
    # --shard I/N: validated here (a malformed GBENCH_SHARD raises ValueError before anything
    # starts) and forwarded normalized for a partial shard only.
    shard_spec = current_shard()
    shard_label = shard_spec.label if shard_spec and shard_spec.partial else None
    # --eval-limit, else GBENCH_OSWORLD_LIMIT: a positive integer, checked before anything starts
    limit_value, knob_limit = effective_limit(limit)
    ok, reason = check_ui_control_osworld_prerequisites()
    if not ok:
        raise infra_required("ui_control_osworld", reason, DOCS_URL)
    workload = osworld_plan(limit_value, os.environ.get("GBENCH_OSWORLD_DOMAIN", "").strip(),
                            shard_label, knob=knob_limit)
    # The model's pyautogui actions run in the task VM, whose container (started by OSWorld's
    # docker provider on docker's default bridge) has network access: the cloud metadata server
    # must be blocked there before the orchestrator starts it.
    swebench_common.require_task_network_blocked(
        "ui_control_osworld", [_image()], docs_url=DOCS_URL,
        hint=f"Build the orchestrator image {_image()} (see {DOCS_URL}).")

    vm_dir = _vm_dir()
    model_endpoint = _model_endpoint(base_url)
    key = model_endpoint_key()
    relay: Optional[EndpointRelay] = None
    domain = os.environ.get("GBENCH_OSWORLD_DOMAIN", "").strip()
    obs_type = os.environ.get("GBENCH_OSWORLD_OBS_TYPE", "screenshot").strip()
    max_steps = os.environ.get("GBENCH_OSWORLD_MAX_STEPS", "15").strip()
    test_meta = os.environ.get("GBENCH_OSWORLD_TEST_META", "test_all.json").strip()
    eff_limit = str(limit_value) if limit_value else ""
    # gbench's rule (1.0 with --thinking, 0.0 without) unless GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE
    # or --temperature sets one; sent as run.py's --temperature and by the relay
    temperature, temperature_source = resolve_temperature("ui_control_osworld",
                                                          thinking=bool(enable_thinking))

    # This run's own directory under the identity-mounted VM dir (results/, the subset manifest,
    # the summary): OSWorld resumes from the result.txt files its result_dir already holds, and
    # every model's results land under the same routing alias, so a directory shared across runs
    # would hand this run another run's results (or its summary).
    run_id = "gbench_osworld_" + uuid.uuid4().hex[:12]
    orch_name = "gbench_osworld_orch_" + run_id
    run_dir = os.path.join(vm_dir, "gbench_runs", run_id)
    try:
        # Only the invoking user's: the orchestrator runs as root in its container (it may write
        # here) and hands the directory back to this user at the end (OSWORLD_HOST_OWNER).
        os.makedirs(run_dir, mode=0o700)
    except OSError as e:
        raise infra_required("ui_control_osworld", f"cannot create this run's directory under "
                             f"GBENCH_OSWORLD_VM_DIR ({run_dir}: {type(e).__name__}: {e}).",
                             DOCS_URL) from e

    def _reap():
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)
        # sibling VM containers are named by the provider; reap by ancestor image as a backstop.
        ps = subprocess.run(["docker", "ps", "-aq", "--filter", f"ancestor={_task_image()}"],
                            capture_output=True, text=True)
        ids = [x for x in (ps.stdout or "").split() if x]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)

    try:
        # The orchestrator always reaches the endpoint through gbench's relay, which counts the
        # endpoint's failures and adds a real key when there is one (stopped in the finally below).
        relay = model_endpoint_relay(model_endpoint, key, temperature)
        agent_endpoint = relay.url
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host",
               "-v", "/var/run/docker.sock:/var/run/docker.sock",
               "-v", f"{vm_dir}:{vm_dir}:rw",                 # identity mount (qcow2 source for siblings)
               "-e", f"OSWORLD_WORKDIR={vm_dir}",             # docker_vm_data/ lives here
               "-e", f"OSWORLD_RUN_DIR={run_dir}",            # this run's results/ + summary
               "-e", f"GBENCH_MODEL_BASE_URL={agent_endpoint}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               # By name: Docker copies the value (the placeholder) from run_env, so nothing is
               # on the command line (argv is visible to every process on the host).
               "-e", "GBENCH_MODEL_API_KEY",
               "-e", f"OSWORLD_OBS_TYPE={obs_type}",
               "-e", f"OSWORLD_MAX_STEPS={max_steps}",
               "-e", f"OSWORLD_TEST_META={test_meta}",
               "-e", f"OSWORLD_TEMPERATURE={temperature}",
               # the run directory goes back to this user at the end
               "-e", f"OSWORLD_HOST_OWNER={os.getuid()}:{os.getgid()}"]
        if domain:
            cmd += ["-e", f"OSWORLD_DOMAIN={domain}"]
        if eff_limit:
            cmd += ["-e", f"OSWORLD_LIMIT={eff_limit}"]
        # A partial shard, validated and normalized above: the launcher selects that interleaved
        # task subset (rebuilds the --test_all_meta_path manifest) BEFORE OSWORLD_LIMIT, matching
        # the native path's compose order; the harness then iterates exactly that subset.
        if shard_label:
            cmd += ["-e", f"GBENCH_SHARD={shard_label}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S", str(48 * 60 * 60)))
        logger.info("ui_control_osworld: docker run %s (obs=%s, domain=%s, limit=%s)",
                    _image(), obs_type, domain or "all", eff_limit or "none")
        run_env = {k: v for k, v in os.environ.items() if k != "GBENCH_OSWORLD_MODEL_API_KEY"}
        run_env["GBENCH_MODEL_API_KEY"] = MODEL_KEY_PLACEHOLDER
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                                  env=run_env)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "ui_control_osworld", f"the orchestrator exceeded "
                f"GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S ({timeout_s}s) and was killed; sibling VM "
                f"containers reaped. Raise the timeout or reduce the scope.", DOCS_URL) from exc

        summ_path = os.path.join(run_dir, "ui_control_osworld_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "ui_control_osworld", f"the orchestrator produced no summary in {run_dir} "
                f"(rc={proc.returncode}; a harness failure, not a 0%; an image built before its "
                f"launcher took OSWORLD_RUN_DIR writes it elsewhere: rebuild it, see {DOCS_URL}). "
                f"tail: {(proc.stderr or proc.stdout or '')[-800:]}", DOCS_URL)
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
    finally:
        _reap()
        if relay is not None:
            relay.stop()

    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    applied = summary.get("shard") if isinstance(summary.get("shard"), dict) else {}
    if shard_label and applied.get("spec") != shard_label:
        raise infra_required(
            "ui_control_osworld", f"--shard {shard_label} was requested but the orchestrator did "
            f"not report applying it (its summary's shard record is {applied or 'missing'}); "
            f"rebuild the image from gbench/docker/ui_control_osworld.Dockerfile.", DOCS_URL)
    if eff_limit and not selection.get("limit_draw"):
        raise infra_required(
            "ui_control_osworld", f"the orchestrator does not report how it drew the --eval-limit "
            f"subset (an image built before the seeded draw, which took the head of the sorted "
            f"task list); rebuild the image from gbench/docker/ui_control_osworld.Dockerfile.",
            DOCS_URL)
    if not summary.get("n_selected"):
        if shard_label:
            return _empty_shard_result(model_name, shard_label, summary, run_dir, workload)
        raise infra_required(
            "ui_control_osworld", f"no task was selected (GBENCH_OSWORLD_DOMAIN="
            f"{domain or 'all'}, GBENCH_OSWORLD_TEST_META={test_meta}); name a domain the manifest "
            "holds.", DOCS_URL)

    scored = compute_ui_control_osworld_score(summary)
    if scored is None:
        raise infra_required(
            "ui_control_osworld", "no tasks produced a result (all VM/infra failures? check /dev/kvm, "
            "the qcow2, the task image, and the model endpoint reachability).", DOCS_URL)

    n_selected = summary.get("n_selected") or 0
    n_missing = summary.get("n_missing") or 0
    task_rows = [t for t in summary.get("tasks") or [] if isinstance(t, dict)]
    # tasks with a result during which a model request failed on the endpoint's side
    n_endpoint_failed = sum(1 for t in task_rows if isinstance(t.get("result"), (int, float))
                            and int(t.get("endpoint_failures") or 0))
    n_scored = (sum(1 for t in task_rows if _scored_task(t)) if task_rows
                else summary.get("n_scored") or 0)
    # A sharded run covers a round-robin subset by construction, so it is never the full set.
    sharded = shard_label is not None
    is_full = (not domain and not eff_limit and not sharded and n_selected >= _FULL_TASK_COUNT)

    reasons = ["self-hosted agent + single trial (the OSWorld leaderboard is maintainer-run)",
               f"observation_type={obs_type}"]
    if not is_full and (domain or eff_limit or not sharded):
        reasons.append(f"subset (domain={domain or 'all'}, limit={eff_limit or 'none'})")
    if sharded:
        reasons.append(SHARD_REASON.format(label=shard_label, index=shard_spec.index,
                                           count=shard_spec.count))
    if n_missing:
        reasons.append(f"{n_missing} task(s) produced no result (harness errors, left out of the "
                       "score)")
    if n_endpoint_failed:
        reasons.append(f"{n_endpoint_failed} task(s) had a model request fail on the endpoint's "
                       "side (harness errors, left out of the score)")
    rate = scored["success_rate"]
    status_reason = harness_error_text(n_missing, n_endpoint_failed, n_selected)
    if rate is None:
        status = "failed"
        status_reason = "nothing was scored: " + (status_reason or "no task produced a result")
    else:
        status = "completed_with_errors" if (n_missing or n_endpoint_failed) else "success"

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "ui_control_osworld",
        "model_name": model_name,
        "status": status,
        "status_reason": status_reason,
        # headline = mean task success (%) over the tasks with a result; None (withheld) if none
        "accuracy": round(rate * 100.0, 2) if rate is not None else None,
        **({"accuracy_withheld_reason": status_reason} if rate is None else {}),
        "success_rate": round(rate, 4) if rate is not None else None,
        "success_rate_missing_as_zero": scored["success_rate_missing_as_zero"],
        "effective_n": n_scored,
        "scoring_excluded": n_missing + n_endpoint_failed,
        "per_domain": scored.get("per_domain"),
        "n_selected": n_selected,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "n_endpoint_failed": n_endpoint_failed,
        "observation_type": obs_type,
        "scoring": "OSWorld deterministic per-task execution evaluators (result.txt in [0,1]); no LLM judge",
        "temperature": temperature,
        "temperature_source": temperature_source,
        "sampling": (f"temperature {temperature} ({temperature_source}; gbench's rule: 1.0 with "
                     "--thinking, 0.0 without), sent as OSWorld run.py's --temperature and set on "
                     "every model request by the endpoint relay; top_p 0.9 (run.py's default). The "
                     "mm_agents PromptAgent sends a plain OpenAI-compat payload with NO "
                     "reasoning/thinking field, so enable_thinking is not applied here"),
        "execution": ("sequential, single-VM: canonical OSWorld run.py drives one DesktopEnv VM through a "
                      "sequential per-task loop (no multiprocessing), so concurrency/--sandboxes does not "
                      "apply"),
        # the selection: the manifest's task count, the limit and how the launcher drew it
        "selection": selection or None,
        "eval_limit_plan": plan_from_summary(workload, n_selected),
        # one row per selected task ("<domain>/<example_id>"): its result.txt value, or a harness
        # error (no result.txt) left out of the score; merge_shards re-scores their union
        "sample_traces": task_traces(summary),
        "run_dir": run_dir,
        # the task VM image OSWorld's docker provider ran (it starts this name): the prerequisite
        # check verified that it is the pinned digest, and its image id
        "task_image": {"image": _task_image(), "pinned": TASK_IMAGE_PINNED,
                       "id": _VERIFIED_TASK_IMAGE.get("id")},
        "raw_summary": summary,
        "metric": ("OSWorld success rate = mean of per-task result.txt (0-1) over the selected "
                   "tasks that produced one (OSWorld get_result; a task without one is a harness "
                   "error, left out and counted in n_missing, and so is a task during which a "
                   "model request failed on the endpoint's side, n_endpoint_failed). Agent = "
                   "mm_agents PromptAgent (pyautogui "
                   "actions, screenshot obs) in a docker-provider Ubuntu VM, driven by the served "
                   "model via OPENAI_BASE_URL. Deterministic execution-based grading; no LLM judge."),
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": "; ".join(reasons),
    }
    if relay is not None:
        # Endpoint failures the relay counted are harness errors of the run (completed_with_
        # errors); the tasks they fell in are the launcher's per-example records above (a task
        # whose run raised wrote no result.txt, one whose request got no reply or a failure status
        # has endpoint_failures), left out of the score.
        result["model_endpoint_relay"] = {
            **relay.record(),
            "adds": ("Authorization: Bearer <GBENCH_OSWORLD_MODEL_API_KEY>" if key
                     else "nothing (the endpoint has no key)")}
        reasons.extend(swebench_common.relay_failure_reasons(result, result["model_endpoint_relay"]))
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    record_shard(result, shard_label)
    return result


def harness_error_text(n_missing: int, n_endpoint_failed: int, n_selected: int
                       ) -> Optional[str]:
    """The status_reason naming the harness errors left out of the score, or None."""
    parts = []
    if n_missing:
        parts.append(f"{n_missing} of {n_selected} task(s) produced no result.txt (the task's run "
                     "raised in OSWorld's loop: VM, model endpoint or evaluator failure)")
    if n_endpoint_failed:
        parts.append(f"{n_endpoint_failed} of {n_selected} task(s) had a model request that got "
                     "no reply or HTTP 5xx, 429, 401, 403 or 404 (OSWorld's agent went on with an "
                     "empty reply)")
    return ("harness errors left out of the score: " + "; ".join(parts)) if parts else None


def task_traces(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The summary's per-task rows as traces: `sample_key` ("task:<domain>/<example_id>"),
    `task`, `domain`, `result` (result.txt in [0,1]), `endpoint_failures` and, for a task without
    a result.txt, `harness_error: "no_result"`, for one during which a model request failed on
    the endpoint's side, `harness_error: "endpoint_failed"`, each with `scoring_excluded` (left
    out of the score)."""
    rows = []
    for t in summary.get("tasks") or []:
        if not isinstance(t, dict) or not t.get("task"):
            continue
        value = t.get("result")
        row = {"sample_key": f"task:{t['task']}", "task": t["task"], "domain": t.get("domain"),
               "result": float(value) if isinstance(value, (int, float)) else None,
               "endpoint_failures": int(t.get("endpoint_failures") or 0)}
        if row["result"] is None:
            row.update({"harness_error": "no_result", "scoring_excluded": True})
        elif row["endpoint_failures"]:
            row.update({"harness_error": "endpoint_failed", "scoring_excluded": True})
        rows.append(row)
    return rows


def plan_from_summary(plan: Dict[str, Any], n_selected: int) -> Dict[str, Any]:
    """The run's `eval_limit_plan` with the samples and episodes the launcher actually selected
    (its manifest: a domain, a shard, or a limit above what the selection holds narrows the
    logged count), one episode each."""
    out = dict(plan or {})
    n = int(n_selected or 0)
    if out.get("samples") is not None and out["samples"] != n:
        out["samples_logged"] = out["samples"]
    out["samples"] = n
    out["episodes"] = n
    return out


def _empty_shard_result(model_name: str, shard_label: str, summary: Dict[str, Any],
                        run_dir: str, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The result of a shard that holds none of the selected tasks: OSWorld was not started, so
    nothing is scored (accuracy null, never 0) and nothing failed."""
    n_full = (summary.get("shard") or {}).get("n_full")
    reason = (f"empty shard {shard_label}: no items on this node"
              + (f" ({n_full} in the full set)" if n_full is not None else ""))
    logger.info("ui_control_osworld: %s", reason)
    result = {
        "benchmark_type": "eval", "eval_name": "ui_control_osworld", "model_name": model_name,
        "status": "success", "status_reason": reason, "total_questions": 0,
        "accuracy": None, "accuracy_withheld_reason": reason, "empty_shard": True,
        "full_set_items": n_full, "n_selected": 0, "n_scored": 0, "n_missing": 0,
        "sample_traces": [], "selection": summary.get("selection"), "run_dir": run_dir,
        "leaderboard_comparable": False, "leaderboard_comparable_reason": reason,
    }
    if plan is not None:
        result["eval_limit_plan"] = plan_from_summary(plan, 0)
    record_shard(result, shard_label)
    return result


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one ui_control_osworld run (gbench --merge): the mean
    result.txt over the union of the shards' tasks that produced one (sample_traces; a task
    without one is a harness error, left out and counted in n_missing), overall and per domain,
    as the run computes it; the shard and empty-shard reasons are dropped."""
    from ...merge import MergeError
    union: Dict[str, Dict[str, Any]] = {}
    for r in results:
        traces = r.get("sample_traces")
        if not isinstance(traces, list):
            raise MergeError("ui_control_osworld: a shard carries no task traces (sample_traces)")
        for t in traces:
            if isinstance(t, dict) and t.get("sample_key"):
                union.setdefault(str(t["sample_key"]), t)
    traces = sorted(union.values(), key=lambda t: str(t.get("task")))

    def _counts(t: Dict[str, Any]) -> bool:
        return (isinstance(t.get("result"), (int, float)) and not t.get("harness_error")
                and not int(t.get("endpoint_failures") or 0))
    scored = [t for t in traces if _counts(t)]
    per_domain: Dict[str, Dict[str, Any]] = {}
    for t in traces:
        d = per_domain.setdefault(str(t.get("domain")), {"n": 0, "n_scored": 0, "_sum": 0.0,
                                                         "_raw": 0.0})
        d["n"] += 1
        if isinstance(t.get("result"), (int, float)):
            d["_raw"] += float(t["result"])
        if _counts(t):
            d["n_scored"] += 1
            d["_sum"] += float(t["result"])
    for d in per_domain.values():
        total, raw = d.pop("_sum"), d.pop("_raw")
        d["success_rate"] = total / d["n_scored"] if d["n_scored"] else None
        d["success_rate_missing_as_zero"] = raw / d["n"] if d["n"] else None
    rate = sum(float(t["result"]) for t in scored) / len(scored) if scored else None
    ran = [r for r in results if not r.get("empty_shard")]
    merged = dict(ran[0] if ran else results[0])
    for key in ("empty_shard", "full_set_items", "accuracy_withheld_reason", "raw_summary",
                "model_endpoint_relay", "run_dir", "selection"):
        merged.pop(key, None)
    n_missing = sum(1 for t in traces if not isinstance(t.get("result"), (int, float)))
    n_endpoint_failed = len(traces) - len(scored) - n_missing
    status_reason = harness_error_text(n_missing, n_endpoint_failed, len(traces))
    merged.update({
        "sample_traces": traces,
        "accuracy": round(rate * 100.0, 2) if rate is not None else None,
        "success_rate": round(rate, 4) if rate is not None else None,
        "success_rate_missing_as_zero": (sum(float(t["result"]) for t in traces
                                             if isinstance(t.get("result"), (int, float)))
                                         / len(traces) if traces else None),
        "per_domain": dict(sorted(per_domain.items())),
        "n_selected": len(traces), "n_scored": len(scored), "n_missing": n_missing,
        "n_endpoint_failed": n_endpoint_failed,
        "effective_n": len(scored), "scoring_excluded": n_missing + n_endpoint_failed,
        "run_dirs": [r.get("run_dir") for r in results if r.get("run_dir")],
        # each shard's own selection (its manifest count, limit and draw)
        "selection_by_shard": {str(r.get("shard") or i + 1): r.get("selection")
                               for i, r in enumerate(results)},
        "status": ("failed" if rate is None else
                   "completed_with_errors" if n_missing or n_endpoint_failed or any(
                       r.get("status") == "completed_with_errors" for r in results)
                   else "success"),
        "status_reason": ("nothing was scored: " + (status_reason or "no task produced a result")
                          if rate is None else status_reason),
    })
    if rate is None:
        merged["accuracy_withheld_reason"] = merged["status_reason"]
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if (reason and not reason.startswith(shard_prefix) and not reason.startswith(
                    "empty shard ") and "task(s) produced no result" not in reason
                    and "on the endpoint's side (harness errors" not in reason
                    and reason not in kept):
                kept.append(reason)
    if n_missing:
        kept.append(f"{n_missing} task(s) produced no result (harness errors, left out of the "
                    "score)")
    if n_endpoint_failed:
        kept.append(f"{n_endpoint_failed} task(s) had a model request fail on the endpoint's "
                    "side (harness errors, left out of the score)")
    if len(traces) < _FULL_TASK_COUNT and not any(k.startswith("subset (") for k in kept):
        kept.append(f"subset (the merged shards hold {len(traces)} of {_FULL_TASK_COUNT} tasks)")
    merged["leaderboard_comparable"] = False       # never comparable (see the module docstring)
    merged["leaderboard_comparable_reason"] = "; ".join(kept)
    # the shards' summed plan, not the first shard's
    from .limit_plan import merge_plans
    plan = merge_plans(results)
    if plan is not None:
        merged["eval_limit_plan"] = plan
    return merged
