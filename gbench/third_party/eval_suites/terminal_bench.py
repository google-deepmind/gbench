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

"""Official Terminal-Bench 2.1 evaluation suite via Harbor framework & Docker sandboxes.

Harness: Harbor 0.20.0 (the TB2.1 leaderboard's lock, terminal-bench-2-1@7131e43
leaderboard/uv.lock), run from the local harness image HARBOR_IMAGE_DEFAULT
(gbench/docker/terminal_bench.Dockerfile: Harbor and every package of its dependency closure at
the lock's versions, gbench/docker/terminal_bench.constraints.txt, plus the Docker CLI with its
compose and buildx plugins), on the dataset pinned to the leaderboard's content hash
(`-d terminal-bench/terminal-bench-2-1@DATASET_REF`, leaderboard/src/leaderboard/core/hub.py).
check_terminal_bench_prerequisites refuses a missing image, or one whose labels name another Harbor
release or lock (infra_required naming the build command); gbench never pulls it. Every harness
container runs with Harbor's telemetry off (HARBOR_TELEMETRY=0, _FORCED_ENV). The result records
`harbor_version`, `harness_image`, `harness_telemetry`, `dataset` and `dataset_ref`.

Harness container (harbor_command): Harbor's local docker backend starts each task's containers on
this host's docker daemon through the mounted docker socket (docker-out-of-docker), and the jobs
directory is mounted at the same path on both sides because Harbor bind-mounts each trial's
directories into its task container. The container runs as the invoking user (with the socket's
group), on the host network, with every capability dropped and no-new-privileges. Its environment
is only the variables gbench names (OPENAI_API_KEY=EMPTY, OPENAI_BASE_URL, and the HARBOR_* and
proxy variables of the credential-free environment that are set), and nothing of this host is
mounted besides the docker socket, the jobs directory, gbench's Harbor cache (harbor_cache_dir,
the container's ~/.cache/harbor) and the salvage patch (read-only). The model's commands never
run there: Terminus-2 types them into tmux in the task containers, which the metadata gate below
covers.

Sampling: the temperature follows gbench's rule for every suite, passed as Terminus-2's
`temperature` agent kwarg: 1.0 for a `--thinking` run (temperature_source "thinking"), 0.0 without
it (temperature_source "default"). The TB2.1 leaderboard's Terminus-2 kwargs
(TERMINAL_BENCH_SAMPLING; configs/leaderboard.yaml: `temperature: 1` for every terminus-2 agent;
its `reasoning_effort` values are keyed on hosted models and do not apply to a served model) are
recorded. Terminus-2 and LiteLLM send no `max_tokens` on any turn (in both thinking modes);
gbench's relay sends the run's `--max-output-tokens` as max_tokens on every turn that carries
none (swe_agentic.AGENT_MAX_TOKENS_SUITES; `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` sends none, as
upstream), recorded as `agent_max_tokens_from_run` (and in `model_endpoint_relay`). Override
the temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_TERMINAL_BENCH_TEMPERATURE`, which takes precedence over both; a value other than 1 is
named in leaderboard_comparable_reason. Agent settings: Terminus-2's default `json` parser, the
agent-timeout multiplier 1.0 (SUBMIT.md: `timeout_multiplier` unset or 1.0) and no environment
overrides in the task container or the verifier (SUBMIT.md: no resource overrides);
GBENCH_TB_PARSER, GBENCH_TB_TURN_MAX_TOKENS, GBENCH_TB_TIMEOUT_MULTIPLIER and
GBENCH_TB_CONTAINER_THREADS are opt-in deviations. Each one set is named in
leaderboard_comparable_reason. There is no LLM judge: each trial is scored by its task's own
verifier tests.

Trials: Harbor runs each task `-k` times (`--n-attempts`), TERMINAL_BENCH_TRIALS (5) by default,
the TB2.1 leaderboard's minimum (leaderboard/SUBMIT.md: at least 5 trials per task). An explicit
`--attempt-count k` above 5 runs k trials; one below it keeps the 5 (a smaller count would measure
less than the protocol does). The headline `accuracy` is the mean over every scored trial (Harbor's
`mean` metric, which is the mean of the per-task means when every task ran k trials); the result
records `attempts` (`attempts_requested`, `attempts_per_sample`, `source`, and avg@k / pass@k /
pass^k over the tasks). A trial that raised an exception Harbor retries is run again up to
TRIAL_MAX_RETRIES (3) times, the leaderboard's `retry: max_retries: 3` (`harbor run
--max-retries 3`; recorded as `trial_max_retries`); its result is the last attempt's.

Before Harbor starts, the run's tasks are downloaded into gbench's Harbor cache when it lacks one
(`harbor dataset download --cache` in the harness image, as `harbor run` does on first use;
`dataset_cache`), and the task images not on the host yet are pulled with the host's docker CLI,
which carries the user's registry login, four at a time (`task_image_prepull`): Harbor pulls in
the harness container anonymously, where the registry's unauthenticated rate limit applies. The
harness container gets the daemon socket the docker CLI uses (DOCKER_HOST, else the current
docker context, else /var/run/docker.sock; a non-unix endpoint is infra_required), mounted at
/var/run/docker.sock.

Task selection (`--shard I/N`, transported as `GBENCH_SHARD`, and `--eval-limit`): this suite
bypasses `base.run_eval_suite`, so it selects the tasks itself, the way the native suites do.
With a shard or a limit set, the runner enumerates the pinned dataset's task names through
Harbor's own package resolver (`PackageDatasetClient.get_dataset_metadata`, the exact names
`--include-task-name` matches at run time), sorts them, takes shard `I` of `N` by round-robin
(`sampling.shard_select`, `sorted[I-1::N]`), then draws `--eval-limit` tasks from that shard with
`sampling.stratified_sample(seed="terminal_bench")` (a seeded draw, not the first N in registry
order), and passes each selected name as `--include-task-name`; `--n-tasks` is never used. So
`--eval-limit 10` and `--shard 1/1 --eval-limit 10` run the same tasks. The result records the
selection (`selected_tasks`) and the shard identity (base.shard_identity); a partial shard is not
leaderboard-comparable (base.SHARD_REASON). A selected task trial Harbor wrote no result for is a
harness error (`harness_errors` `missing`, below), never a 0. An empty shard (more shards than
tasks) runs nothing and reports `accuracy` null with `accuracy_withheld_reason`. If the registry
cannot be reached to enumerate the task list, the run hard-errors rather than silently running the
full set. A malformed GBENCH_SHARD is an error (ValueError), never a full run.

Each trial record carries Harbor's canonical task identity: `task_name`, `task_id` (Harbor's
structured id, as JSON) and `task_checksum`, with `sample_key` (the task) and `attempt_idx` (its
trial number, by trial name), `category` / `difficulty` from the task's `task.toml` `[metadata]`
when Harbor's task cache holds it (filling `category_accuracy`), `is_correct` and, for a harness
error, `scoring_excluded`; traces are sorted by task, then trial.

Model endpoint key: Terminus-2 runs in Harbor's process in the harness container, on the host
network; the task containers, where the model's commands run, get the environment Harbor gives
them. Harbor gets no credential (the named variables above) and `OPENAI_API_KEY=EMPTY`; when the
served endpoint has a key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY), Terminus-2
reaches the endpoint through gbench's EndpointRelay on 127.0.0.1 (reachable on the host network,
not from the task containers' bridge networks), which adds it (`model_endpoint_relay`).

Cloud metadata server: the agent's commands and the verifier's tests run in the task containers,
with network access. Before Harbor starts, `require_task_metadata_blocked` checks that
169.254.169.254 is unreachable from docker's default bridge and from a user-defined bridge network
(the kind Harbor's compose projects create), and that the package hosts each task's verifier
installs its test tools from resolve there (swebench_common.TASK_INSTALL_HOSTS), probing with a
task image of Harbor's local task cache that is already on the host. Nothing is pulled for the
probe: without such an image the check is infra_required, naming the `docker pull` of one of the
run's task images (infra_required too for a reachable metadata server or a name that does not
resolve). Harbor's task cache is gbench's (harbor_cache_dir), the one the harness container uses.

Harness errors: a trial is left out of `total_questions` and `accuracy`, counted in
`harness_errors` (by kind), `harness_error_trials` and `failed_requests`, and makes the run
`completed_with_errors`, when it failed outside the model's control; a run whose every trial is
one hard-errors (infra_required) instead of reporting 0. A trial whose result.json cannot be read
(`parse`) and each expected trial Harbor wrote no result for (`missing`: the selected tasks, or in
a run without a shard or limit the pinned dataset's FULL_TASK_COUNT (89) tasks, x k, less the
trials found; `expected_trials` records the product) are harness errors of the same kind: counted
in `harness_errors` and `failed_requests`, never scored. Harbor 0.20.0's `harbor run` exits 0 once
its job completed, whatever its trials' outcomes, so a non-zero exit (`harbor_returncode`) means
its process failed or was killed: the run is completed_with_errors and not leaderboard-comparable,
whatever it managed to write. `status_reason` names every harness error; `setting_deviations`
lists the settings that differ from the leaderboard's (the harness-error reasons aside).
`merge_shards` (gbench --merge) re-counts the union of the shards' trials the same way (see its
docstring).

* An exception (`harness_error: "exception"`, trial_exception_harness_error): a trial without a
  reward whose Harbor exception happened outside the model's control, read by the phase the
  trial's result.json records and the exception's type. Harness errors: any exception before the
  agent started (environment or docker start, healthcheck, agent setup:
  EnvironmentStartTimeoutError, HealthcheckError, SandboxBuildFailedError, AgentSetupTimeoutError,
  ...), a cancelled trial (CancelledError), a model-endpoint failure (ENDPOINT_EXCEPTIONS:
  LiteLLM's connection, server, timeout, rate-limit and request errors as Terminus-2 re-raises
  them after its retries, an installed agent's ApiRateLimitError), an exception naming an
  unreachable docker daemon, and a verifier that started but could not upload its tests or
  download its output (AddTestsDirError, DownloadVerifierDirError, any other verifier failure
  outside VERIFIER_OUTCOME_EXCEPTIONS) unless docker's own message shows the task container had
  stopped or its target path was unusable (the model's commands can do both). The model's
  outcome: a reward (the verifier finished), AgentTimeoutError and a verifier that started and
  then timed out or left no usable reward (VerifierTimeoutError, RewardFileNotFoundError,
  RewardFileEmptyError, VerifierOutputParseError), which are Harbor's default
  RetryConfig.exclude_exceptions, and NonZeroAgentExitCodeError, ContextLengthExceededError and
  OutputLengthExceededError (gbench's own attribution: the model's process, context or output ran
  past a limit). Any other exception raised while the agent ran (a RuntimeError from tmux
  send-keys after the model's own keystrokes ended its tmux session or container, for example)
  is not a known harness failure, so the trial stays the model's outcome, as Harbor's metrics
  count every trial without a reward as 0, with `exception_note` saying so (counted in
  `exceptions_kept_as_model_outcome`).
* A DNS failure (`harness_error: "dns"`): a trial that did not pass whose verifier could not
  resolve one of those package hosts before its tests started (swebench_common.verifier_dns_failure
  over verifier/test-stdout.txt), when something outside its container corroborates it: another
  such trial in the run, or a fresh probe container after the run that cannot resolve one of the
  hosts either (swebench_common.corroborated_dns_failures, dns_recheck). Uncorroborated, or when
  the agent's transcript names /etc/resolv.conf, /etc/hosts or /etc/nsswitch.conf, the model may
  have broken name resolution itself, so the trial stays the model's outcome (recorded in
  `dns_failure_note`).
"""

import collections
import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import (SHARD_REASON, OfficialSampling, add_comparability_reason, current_shard,
                   UNCAPPED_TIMEOUT_TOKENS, endpoint_api_key, get_run_knob, repetition_onset,
                   repetition_run, request_timeout_s, resolve_official_sampling,
                   resolve_token_budget, shard_identity)
from .limit_plan import eval_limit_plan
from .sampling import shard_select, stratified_sample
from .swe_thread_cap import THREAD_VARS
from . import swebench_common
from .swebench_common import infra_required
from . import terminus_salvage_patch

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=0, desc="", **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def refresh(self):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/terminal_bench.md"
DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"
#: The content hash of the canonical dataset version (terminal-bench-2-1@7131e43
#: leaderboard/src/leaderboard/core/hub.py DATASET_REF, the `ref` of every leaderboard config).
#: Harbor resolves `org/name@sha256:<hash>` to exactly that version, so a registry push cannot
#: change the tasks a run executes.
DATASET_REF = "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
#: What `harbor run -d` and the shard enumeration are given.
DATASET_SPEC = f"{DEFAULT_DATASET}@{DATASET_REF}"
#: The Harbor release of the TB2.1 leaderboard (terminal-bench-2-1@7131e43 leaderboard/uv.lock:
#: harbor 0.20.0; leaderboard/pyproject.toml harbor[modal]>=0.20.0). The result parsing and the
#: salvage patch follow its sources.
HARBOR_VERSION = "0.20.0"
#: The terminal-bench-2-1 commit whose leaderboard/uv.lock the harness image follows (Harbor
#: HARBOR_VERSION and its dependency closure: gbench/docker/terminal_bench.constraints.txt).
TB21_LOCK_COMMIT = "7131e4375048a0e408a8fb404b5f499d726b695b"
#: The local harness image, tagged with its Harbor release (gbench/docker/terminal_bench.Dockerfile,
#: built with harbor_image_build_command).
HARBOR_IMAGE_DEFAULT = f"gbench-terminal-bench:harbor-{HARBOR_VERSION}"
#: The labels the image build sets and the prerequisite check requires.
HARBOR_IMAGE_LABELS = {"org.gbench.harbor": HARBOR_VERSION,
                       "org.gbench.terminal_bench_lock": TB21_LOCK_COMMIT}
#: The docker socket mounted into the harness container (docker-out-of-docker): its path inside
#: the container (where Harbor's docker CLI looks), and the host's default one.
_DOCKER_SOCKET = "/var/run/docker.sock"
#: The docker CLI's default daemon endpoint (no DOCKER_HOST, the default context).
_DEFAULT_DOCKER_HOST = "unix:///var/run/docker.sock"
#: Retries of a trial that raised an exception Harbor retries (RetryConfig's defaults exclude
#: the agent and verifier timeouts, a missing or unreadable reward and a few more): the TB2.1
#: leaderboard's `retry: max_retries: 3` (configs/leaderboard*.yaml at TB21_LOCK_COMMIT), passed
#: as `harbor run --max-retries`. Harbor removes the failed attempt's trial directory and runs the
#: trial again under the same name, so the result is the last attempt's.
TRIAL_MAX_RETRIES = 3
#: How many task images the host pulls at once before Harbor starts (prepull_task_images).
PREPULL_CONCURRENCY = 4
#: HOME in the harness image (world-writable there), and where gbench's Harbor cache is mounted
#: (Harbor's cache is ~/.cache/harbor: harbor.constants.CACHE_DIR in 0.20.0).
_IMAGE_HOME = "/harbor-home"
_IMAGE_HARBOR_CACHE = _IMAGE_HOME + "/.cache/harbor"
#: Where the salvage patch (terminus_salvage_patch.py) is mounted, read-only.
_IMAGE_SALVAGE_PATCH = "/opt/gbench/terminus_salvage_patch.py"
#: Host variables the harness container gets by name, when set in the credential-free
#: environment: Harbor's own settings (HARBOR_REGISTRY_WEBSITE_URL, ...) and the proxy settings,
#: which Harbor saw when it ran on this host. HARBOR_TELEMETRY is not among them: it is always
#: set (_FORCED_ENV).
_PASSED_ENV_PREFIXES = ("HARBOR_",)
_PASSED_ENV_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
                     "http_proxy", "https_proxy", "no_proxy", "all_proxy")
#: Variables every harness container gets with this value, whatever this host's environment has.
#: Harbor 0.20.0 sends usage events to PostHog (harbor/telemetry.py: the command, the agent and
#: model names, the dataset refs, the job's mean reward, token counts and cost) unless
#: HARBOR_TELEMETRY is one of 0/false/no/off/disabled (_telemetry_disabled). gbench turns it off:
#: no event leaves the harness container, and nothing Harbor runs or scores changes.
_FORCED_ENV = {"HARBOR_TELEMETRY": "0"}
#: What the result records as `harness_telemetry`.
HARNESS_TELEMETRY = {"harbor_telemetry": "disabled", "env": "HARBOR_TELEMETRY=0"}
#: The TB2.1 leaderboard's Terminus-2 settings. configs/leaderboard.yaml (and the per-model
#: leaderboard configs) pass `temperature: 1` to every terminus-2 agent; Terminus-2 itself sends
#: no max_tokens (harbor 0.20.0 terminus_2.py and llms/lite_llm.py), so none is sent in either
#: thinking mode.
TERMINAL_BENCH_SAMPLING = OfficialSampling(
    reference=("terminal-bench-2-1@7131e4375048a0e408a8fb404b5f499d726b695b configs/"
               "leaderboard.yaml (terminus-2 kwargs temperature: 1); harbor 0.20.0 "
               "harbor/agents/terminus_2/terminus_2.py (parser_name='json') and llms/lite_llm.py "
               "(no max_tokens); leaderboard/SUBMIT.md (timeout_multiplier unset or 1.0, no "
               "agent or verifier timeout overrides, no resource overrides)"),
    temperature=1.0, send_max_tokens=False)
#: Terminus-2's default action parser (harbor 0.20.0 terminus_2.py:161).
CANONICAL_PARSER = "json"
#: Trials per task: the TB2.1 leaderboard's minimum (leaderboard/SUBMIT.md: at least 5 trials per
#: task), passed to Harbor as `-k`. An explicit --attempt-count above it runs that many; never
#: fewer (trial_count).
TERMINAL_BENCH_TRIALS = 5
#: Tasks in the pinned dataset version (DATASET_REF).
FULL_TASK_COUNT = 89


def trial_count(attempt_count: Optional[int]) -> Tuple[int, str]:
    """(trials per task Harbor runs, where the number came from): TERMINAL_BENCH_TRIALS, or an
    explicit --attempt-count when it is larger. A smaller --attempt-count keeps the protocol's
    count: running fewer trials than the leaderboard would measure less than it does."""
    k = int(attempt_count or 1)
    if k > TERMINAL_BENCH_TRIALS:
        return k, f"--attempt-count {k}"
    return TERMINAL_BENCH_TRIALS, ("TB2.1 leaderboard minimum (leaderboard/SUBMIT.md: at least "
                                   f"{TERMINAL_BENCH_TRIALS} trials per task)"
                                   + (f"; --attempt-count {k} is below it and does not lower it"
                                      if k > 1 else ""))


def select_task_names(full: List[str], shard: Optional[Tuple[int, int]],
                      limit: Optional[int]) -> List[str]:
    """The tasks a run with `shard` (I, N) and `--eval-limit` `limit` names: `full` sorted and
    de-duplicated, shard I of N by round-robin (sampling.shard_select), then a seeded draw of
    `limit` from the shard (sampling.stratified_sample, seed "terminal_bench"); the limit alone
    draws from the whole set the same way, so a 1/1 shard and no shard select the same tasks."""
    names = shard_select(sorted(set(full)), shard)
    if limit and int(limit) > 0:
        names = stratified_sample(names, int(limit), seed="terminal_bench")
    return list(names)


def missing_trials_reason(missing: int, expected_tasks: int, n_trials: int) -> str:
    """The comparability reason for `missing` expected trials without a result."""
    return (f"{missing} trial(s) of the selected tasks without a result ({expected_tasks} "
            f"task(s) x {n_trials} trials; harness errors), left out of the score")


def harbor_exit_reason(returncode: int) -> str:
    """The comparability reason for a Harbor process that exited non-zero. Harbor 0.20.0's
    `harbor run` exits 0 once its job completed, whatever its trials' outcomes, so any other
    status means its process failed or was killed (137: SIGKILL, e.g. the OOM killer)."""
    return (f"Harbor exited with status {returncode} (its process failed or was killed; a "
            "harness error, the run may be incomplete)")


def harness_error_reasons(*, parse: int, missing: int, expected_tasks: int, n_trials: int,
                          returncode: Optional[int], exc: int, dns: int) -> List[str]:
    """The comparability reasons of a run's harness errors, in the order the result lists them:
    unreadable trial results, expected trials without a result, a non-zero Harbor exit, trials
    lost to exceptions outside the model's control, trials lost to DNS failures."""
    reasons: List[str] = []
    if parse:
        reasons.append(f"{parse} trial result(s) could not be read (harness errors), left out "
                       "of the score")
    if missing:
        reasons.append(missing_trials_reason(missing, expected_tasks, n_trials))
    if returncode not in (0, None):
        reasons.append(harbor_exit_reason(returncode))
    if exc:
        reasons.append(f"{exc} trial(s) lost to harness errors (exceptions outside the model's "
                       "control), left out of the score")
    if dns:
        reasons.append(f"{dns} trial(s) lost to harness errors (DNS failures in the verifier's "
                       "package installs), left out of the score")
    return reasons


def harness_status_reason(*, exc: int, dns: int, parse: int, missing: int,
                          expected_trials: int, returncode: Optional[int]) -> Optional[str]:
    """The result's status_reason: the harness errors left out of the score and a non-zero
    Harbor exit, named; None when there are none."""
    parts = [f"{n} {what}" for n, what in (
        (exc, "trial(s) raised an exception outside the model's control"),
        (dns, "trial(s) lost their verifier to a DNS failure"),
        (parse, "trial result(s) could not be read"),
        (missing, f"of the {expected_trials} expected trial(s) have no result")) if n]
    if returncode not in (0, None):
        parts.append(f"Harbor exited with status {returncode}")
    if not parts:
        return None
    return "harness errors, left out of the score: " + "; ".join(parts)


def _task_id_record(task_id: Any) -> Any:
    """Harbor's structured task id (PackageTaskId / GitTaskId / LocalTaskId, a dict in
    result.json) as JSON-safe data, or None."""
    if task_id is None:
        return None
    try:
        return json.loads(json.dumps(task_id, default=str))
    except (TypeError, ValueError):
        return str(task_id)


def _task_name_of(data: Dict[str, Any], trial_name: str) -> str:
    """A trial's task: result.json's `task_name`, else the trial name without Harbor's
    `__<suffix>`."""
    name = data.get("task_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return str(trial_name).rsplit("__", 1)[0]


def _task_metadata(task_id: Any) -> Dict[str, Any]:
    """`category` and `difficulty` from the `[metadata]` of the task's task.toml in Harbor's
    package task cache (harbor_package_cache: `<org>/<name>/<digest>/task.toml`), when it is
    there; {} otherwise."""
    if not isinstance(task_id, dict):
        return {}
    org, name, ref = task_id.get("org"), task_id.get("name"), str(task_id.get("ref") or "")
    if not (org and name and ref.startswith("sha256:")):
        return {}
    path = os.path.join(harbor_package_cache(), str(org), str(name), ref[len("sha256:"):],
                        "task.toml")
    try:
        import tomllib
        with open(path, "rb") as f:
            meta = tomllib.load(f).get("metadata") or {}
    except (OSError, ValueError):
        return {}
    return {k: str(meta[k]) for k in ("category", "difficulty")
            if isinstance(meta, dict) and isinstance(meta.get(k), (str, int)) and str(meta[k])}


def annotate_trials(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The trial records sorted by task, then trial name, each with `attempt_idx` (its position
    among its task's trials), `sample_key` (the task), `is_correct` and, for a harness error,
    `scoring_excluded`."""
    ordered = sorted(trials, key=lambda t: (str(t.get("task_name") or ""),
                                            str(t.get("trial_name") or "")))
    seen: Dict[str, int] = {}
    for t in ordered:
        task = str(t.get("task_name") or t.get("trial_name") or "")
        t["attempt_idx"] = seen.get(task, 0)
        seen[task] = t["attempt_idx"] + 1
        t["sample_key"] = f"task:{task}"
        t["is_correct"] = bool(t.get("passed"))
        if t.get("harness_error"):
            t["scoring_excluded"] = True
        else:
            t.pop("scoring_excluded", None)
    return ordered


def attempts_report(trials: List[Dict[str, Any]], requested: int, k: int,
                    source: str) -> Dict[str, Any]:
    """The `attempts` record (base's shape): the trials asked for and run per task, and avg@k,
    pass@k and pass^k over the tasks with a scored trial."""
    by_task: Dict[str, List[bool]] = {}
    for t in trials:
        if t.get("scoring_excluded"):
            continue
        by_task.setdefault(str(t.get("sample_key")), []).append(bool(t.get("is_correct")))
    scored = [v for vs in by_task.values() for v in vs]
    n = len(by_task)
    return {
        "attempts_requested": max(1, int(requested or 1)),
        "attempts_per_sample": k,
        "source": source,
        "samples": n,
        "generations": len(scored),
        "avg_at_k": round(100.0 * sum(scored) / len(scored), 2) if scored else None,
        "pass_at_k": round(100.0 * sum(1 for v in by_task.values() if any(v)) / n, 2) if n else None,
        "pass_hat_k": (round(100.0 * sum(1 for v in by_task.values() if v and all(v)) / n, 2)
                       if n else None),
        "unstable_samples": sum(1 for v in by_task.values() if any(v) and not all(v)),
    }


def category_accuracy(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Accuracy per task `category` over the scored trials (empty when no trial has one)."""
    stats: Dict[str, Dict[str, int]] = {}
    for t in trials:
        cat = t.get("category")
        if not cat or t.get("scoring_excluded"):
            continue
        st = stats.setdefault(str(cat), {"correct": 0, "total": 0})
        st["total"] += 1
        st["correct"] += 1 if t.get("is_correct") else 0
    return {c: {"correct": v["correct"], "total": v["total"],
                "accuracy": round(100.0 * v["correct"] / v["total"], 2)}
            for c, v in sorted(stats.items()) if v["total"]}



#: Cap on the transcript kept per trial. Enough to see a cycle without carrying megabytes
#: of terminal output into every result file.
_MAX_TRANSCRIPT_CHARS = 40000

#: Per-turn output cap for the agent's LLM calls: none by default. Terminus-2 and LiteLLM both
#: leave `max_tokens` unset (the canonical setting), so vLLM falls back to
#: `max_model_len - prompt` and a single turn can run long on a slow endpoint (~250k tokens, 56
#: minutes at the rate measured on 2026-08-18). `GBENCH_TB_TURN_MAX_TOKENS=<n>` sets an opt-in
#: backstop, named in leaderboard_comparable_reason. With the default `json` parser a turn that
#: hits such a cap cannot be salvaged and is re-asked (on 2026-08-18 an 8192 cap livelocked 2 of
#: 3 trials that way); pair a cap with GBENCH_TB_PARSER=xml, whose parser salvages a truncated
#: turn. Measured per-turn tokens over one healthy trial (22 turns): median 1,665, p90 3,413,
#: max 7,644.
_TURN_MAX_TOKENS_DEFAULT = 0
#: The backstop values the docs suggest for GBENCH_TB_TURN_MAX_TOKENS (never applied unless set):
#: 32768 for a --thinking run is 4.3x the largest turn observed; 8192 without thinking sat 7%
#: above it and livelocked trials with the json parser.
_TURN_MAX_TOKENS_THINKING = 32768
_TURN_MAX_TOKENS_PLAIN = 8192

#: Terminus-2 parses the model's action out of each turn. Its default `json` parser is the
#: canonical one. `GBENCH_TB_PARSER=xml` (terminus_xml_plain_parser, which implements
#: `salvage_truncated_response`) is an opt-in deviation for runs that cap the turn; it changes
#: the action format the model is asked to emit and is named in leaderboard_comparable_reason.
def agent_parser() -> str:
    """Terminus-2's `parser_name` for this run: GBENCH_TB_PARSER, else the canonical 'json'."""
    return (os.environ.get("GBENCH_TB_PARSER") or "").strip() or CANONICAL_PARSER

#: A turn that truncates and is re-asked, repeatedly, with no step recorded in between.
#: This is what a too-small cap looks like from the outside, and it is invisible in the
#: result unless counted: the trial simply reports `AgentTimeoutError`.
_LIVELOCK_TRUNCATIONS = 3

#: Threads to allow inside the task container: no cap by default (0). The TB2.1 leaderboard
#: requires default execution settings with no resource overrides (leaderboard/SUBMIT.md), and
#: the cap reaches both the agent's shell (`--agent-env`) and the verifier's exec
#: (`--verifier-env`), so `GBENCH_TB_CONTAINER_THREADS=<n>` is an opt-in deviation, named in
#: leaderboard_comparable_reason. Why one might set it:
#:
#: Terminal-Bench pins each container's CPU via `task.toml` - measured across the cached
#: dataset on 2026-08-18 (an EARLIER, larger task revision: 167 of 178 tasks declared
#: `cpus = 1`, 6 declared 2, 5 declared 4, none omitted it). The canonical DEFAULT_DATASET is
#: now terminal-bench-2-1 (89 tasks); the exact per-task distribution there has not been
#: re-measured, but `cpus = 1` was the modal AND minimum value, so a thread cap of 1 stays
#: correct-or-conservative regardless of the count. Docker enforces the pin as a cgroup quota
#: (`cpu.max` = "100000 100000"),
#: but nothing rewrites `/proc/cpuinfo`, so `nproc` inside a 1-CPU container still reports
#: the host's 96. Anything that sizes a thread pool off the visible core count therefore
#: spawns 96 threads to share one CPU's worth of quota - the same oversubscription
#: pathology that cost swebench 1800s per instance before `swe_thread_cap` (see the table
#: in that module).
#:
#: 1 is the modal AND minimum declared value, so it is right for 94% of tasks and merely
#: conservative for the rest - single-threaded is slower than a 4-CPU task could manage,
#: but nothing like 24:1 thrashing. A single value is required because Harbor takes agent
#: kwargs once per run, not per task.
#:
#: The CPU LIMIT itself is deliberately untouched: it is part of the benchmark's published
#: environment spec, and raising it would make these numbers incomparable with every other
#: Terminal-Bench result.
_CONTAINER_THREADS_DEFAULT = 0


def container_thread_env() -> Dict[str, str]:
    """Thread-count env for the task container, or {} (the default: none injected).

    Shares `swe_thread_cap.THREAD_VARS` so the two harnesses cap the same runtimes.
    `GBENCH_TB_CONTAINER_THREADS=<n>` injects them (an opt-in deviation); 0 or "off" injects
    nothing.
    """
    raw = (os.environ.get("GBENCH_TB_CONTAINER_THREADS") or "").strip()
    threads = _CONTAINER_THREADS_DEFAULT
    if raw:
        if raw.lower() in ("off", "none", "false"):
            return {}
        try:
            threads = int(raw)
        except ValueError:
            logger.warning(
                "GBENCH_TB_CONTAINER_THREADS=%r is not an integer; using %d.",
                raw, _CONTAINER_THREADS_DEFAULT)
    if threads <= 0:
        return {}
    return {var: str(threads) for var in THREAD_VARS}


def _turn_token_cap(enable_thinking: bool) -> int:
    """Per-turn `max_tokens` for the agent's LLM calls: 0 (none sent, the canonical setting)
    unless GBENCH_TB_TURN_MAX_TOKENS sets an opt-in cap. The same in both thinking modes:
    upstream sends none in either."""
    raw = os.environ.get("GBENCH_TB_TURN_MAX_TOKENS")
    if raw is not None and raw.strip():
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("GBENCH_TB_TURN_MAX_TOKENS=%r is not an integer; ignoring", raw)
    return _TURN_MAX_TOKENS_DEFAULT


#: The LiteLLM version in the harness image (HARBOR_IMAGE_DEFAULT), recorded with the request
#: timeout (`model_client_timeout`).
HARNESS_LITELLM_VERSION = "1.91.0"


def model_client_timeout(turn_cap: int) -> Dict[str, Any]:
    """The request timeout and retries gbench gives Terminus-2's LiteLLM calls
    (`llm_call_kwargs` `timeout` and `max_retries`), as the result records them.

    2026-10-10 investigation: Harbor's LiteLLM wrapper (harbor/llms/lite_llm.py) calls
    litellm.acompletion with no timeout, and LiteLLM 1.91.0 then falls back to 600 s per request
    (litellm_core_utils/completion_timeout.py, COMPLETION_HTTP_FALLBACK_SECONDS) with the OpenAI
    SDK's 2 retries, so a turn that needed more than 600 s of decode was dropped and asked again
    (up to 3 x 600 s per Harbor attempt, and Harbor's own tenacity retry makes 3 attempts): a
    hidden cap on the turn's length. Terminus-2 sends no max_tokens, so the timeout is
    base.request_timeout_s of the served context window (base.UNCAPPED_TIMEOUT_TOKENS when the
    server reports none), or of GBENCH_TB_TURN_MAX_TOKENS when that opt-in cap is set, and the
    SDK retries are 0, so each of Harbor's 3 attempts is one HTTP attempt. The task's own agent
    timeout still bounds the trial."""
    tokens = turn_cap or get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS
    return {"client": "litellm (Harbor LiteLLM wrapper, Terminus-2)",
            "version": HARNESS_LITELLM_VERSION,
            "timeout_s": request_timeout_s(int(tokens)),
            "retries": 0,
            "outer_attempts": 3,
            "source": ("llm_call_kwargs timeout = base.request_timeout_s("
                       + ("GBENCH_TB_TURN_MAX_TOKENS" if turn_cap else "served context window")
                       + f"={int(tokens)}); max_retries 0 (Harbor's tenacity retry, 3 attempts, "
                       "unchanged)"),
            "replaces": "LiteLLM 1.91.0 default 600 s x 3 SDK attempts",
            "hidden_cap_removed": True}


#: The canonical agent-timeout multiplier (leaderboard/SUBMIT.md: unset or 1.0).
CANONICAL_TIMEOUT_MULTIPLIER = "1.0"


def _timeout_multiplier(enable_thinking: bool) -> str:
    """Agent-timeout multiplier: the canonical 1.0 unless GBENCH_TB_TIMEOUT_MULTIPLIER sets an
    opt-in value (named in leaderboard_comparable_reason). Only the agent's clock is scaled;
    the verifier timeout is a correctness bound and never moves."""
    raw = os.environ.get("GBENCH_TB_TIMEOUT_MULTIPLIER")
    if raw:
        try:
            return str(float(raw))
        except ValueError:
            logger.warning("GBENCH_TB_TIMEOUT_MULTIPLIER=%r is not a number; ignoring", raw)
    return CANONICAL_TIMEOUT_MULTIPLIER


#: Shortest agent budget observed before the multiplier: 900s (the large majority of tasks sat
#: at 900s in the 2026-08-18 measurement, taken on an earlier/larger task revision - see the
#: _CONTAINER_THREADS note; the canonical terminal-bench-2-1 set is 89 tasks and its exact
#: distribution has not been re-measured). Used to express the turn cap as "how many turns
#: actually fit", the quantity that decides whether a run was winnable on THIS hardware.
_SHORTEST_TASK_AGENT_TIMEOUT_S = 900


def _turn_budget_report(trials: List[Dict[str, Any]], turn_cap: int,
                        enable_thinking: bool) -> Optional[Dict[str, Any]]:
    """What this endpoint's speed implies for the turn budget, and what it wasted.

    Derived from the trials themselves rather than from a constant, because the constants
    in this file were tuned on one 8xA100 deployment and are meaningless elsewhere.

    Both inputs need correcting for truncation first. Terminus appends to
    `api_request_times_msec` only on the SUCCESS path and Harbor's `n_output_tokens`
    likewise counts only completed turns, so a turn that hits the cap and raises
    `OutputLengthExceededError` is invisible in both. On the 2026-08-18 run that made a
    trial look 22% model-bound when it was ~99%: 4 truncated turns had silently burned
    ~131k tokens and ~1840s of the 2385s agent phase. Uncorrected, this report understates
    model time and OVERSTATES throughput on exactly the trials it exists to diagnose.

    A truncated turn generates the full cap by definition, so it can be priced exactly.
    """
    done_tokens = sum(t.get("output_tokens") or 0 for t in trials)
    done_seconds = sum((t.get("mean_turn_latency_s") or 0) * (t.get("agent_turns") or 0)
                       for t in trials)
    if not done_tokens or done_seconds <= 0:
        return None
    # Rate must come from COMPLETED turns only - that is the pairing we can trust.
    rate = done_tokens / done_seconds
    truncated = sum(t.get("truncated_turns") or 0 for t in trials)
    wasted_tokens = truncated * turn_cap if turn_cap else 0
    wasted_seconds = wasted_tokens / rate if rate else 0
    per_turn = turn_cap / rate if turn_cap and rate else None
    budget = _SHORTEST_TASK_AGENT_TIMEOUT_S * float(_timeout_multiplier(enable_thinking))
    total_tokens = done_tokens + wasted_tokens
    return {
        "observed_tok_s": round(rate, 1),
        "turn_max_tokens": turn_cap or None,
        "seconds_per_turn": round(per_turn, 1) if per_turn else None,
        "shortest_agent_budget_s": int(budget),
        "turns_in_shortest_budget": int(budget / per_turn) if per_turn else None,
        "observed_turns": [t.get("agent_turns") for t in trials],
        # Every token a truncated turn produced was discarded. This is the number that
        # says whether the cap is set anywhere near the right place.
        "truncated_turns": truncated,
        "tokens_useful": done_tokens,
        "tokens_discarded": wasted_tokens,
        "discarded_token_pct": (round(100.0 * wasted_tokens / total_tokens, 1)
                                if total_tokens else 0.0),
        "seconds_lost_to_truncation": int(wasted_seconds),
    }


@contextlib.contextmanager
def _jobs_dir():
    """Harbor's jobs directory, kept when the run produced nothing to show for itself.

    It used to be an unconditional `TemporaryDirectory`, which deleted Harbor's
    `result.json`, `trial.log` and `agent/trajectory.json` on the way out. On 2026-08-18
    that destroyed the only evidence of why a completed 2h40 job scored `total_questions:
    0`, and the same deletion had already made two `AgentTimeoutError` trials
    undiagnosable earlier the same day. Twice is a design fault, not bad luck.

    Kept only on failure so healthy runs still clean up after themselves; point
    `GBENCH_TB_KEEP_JOBS_DIR` at a path (or set it to 1) to always keep.
    """
    keep = (os.environ.get("GBENCH_TB_KEEP_JOBS_DIR") or "").strip()
    base = keep if (keep and keep not in ("1", "true", "yes")) else None
    # absolute: the harness container mounts it at the same path (harbor_command)
    path = os.path.abspath(tempfile.mkdtemp(prefix="gbench_tb_", dir=base))
    failed = False
    try:
        yield path
    except BaseException:
        failed = True
        raise
    finally:
        if keep or failed or not _JOBS_DIR_OK.get(path, False):
            logger.warning(
                "[terminal_bench] keeping Harbor's jobs dir for diagnosis: %s "
                "(result.json / trial.log / agent/trajectory.json live here). Delete it "
                "when done; set GBENCH_TB_KEEP_JOBS_DIR= to disable this.", path)
        else:
            shutil.rmtree(path, ignore_errors=True)
        _JOBS_DIR_OK.pop(path, None)


#: Set true by the parser once a jobs dir has yielded at least one usable trial, so the
#: directory is only preserved when there is something to investigate.
_JOBS_DIR_OK: Dict[str, bool] = {}


def _extract_reward(rdata: Dict[str, Any]) -> Optional[float]:
    """Canonical Harbor per-trial reward, or None when none is present.

    Mirrors Harbor's aggregation (metrics/base.py aggregate_reward_dicts): prefer the 'reward'
    key, else the SOLE value in the rewards dict (a task may name its single reward differently);
    coerce bool -> 1.0/0.0. gbench previously did rewards.get("reward", 0.0), which scored any
    non-'reward'-keyed task as a hard 0 (fail) that Harbor would have counted. Returning None (not
    0.0) keeps a missing reward distinguishable from a real 0.0 so it is never silently a pass/fail.
    """
    vr = rdata.get("verifier_result") or {}
    rewards = vr.get("rewards") if isinstance(vr, dict) else None
    val: Any = None
    if isinstance(rewards, dict) and rewards:
        val = rewards["reward"] if "reward" in rewards else next(iter(rewards.values()), None)
    elif "reward" in rdata:
        val = rdata.get("reward")
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _job_stats_totals(job_stats: Dict[str, Any]) -> Tuple[int, int]:
    """Trial and pass counts from Harbor's JOB-level result.json.

    This is the fallback for when per-trial parsing yields nothing - precisely the case
    that happened on 2026-08-18 - and it was reading three keys Harbor does not emit
    (`n_trials`, `n_passed`, `n_success`) plus a `mean_reward` that lives elsewhere. It
    therefore returned 0 and the run published "nothing was measured" over a completed
    2h40 job. Harbor 0.20.0 puts the totals at `n_completed_trials` and the score under
    `evals[<name>].metrics[0].mean`.
    """
    total = int(job_stats.get("n_completed_trials")
                or job_stats.get("n_trials") or 0)
    evals = job_stats.get("evals")
    means: List[float] = []
    if isinstance(evals, dict):
        for ev in evals.values():
            if not isinstance(ev, dict):
                continue
            if not total:
                total += int(ev.get("n_trials") or 0)
            for m in ev.get("metrics") or []:
                if isinstance(m, dict) and isinstance(m.get("mean"), (int, float)):
                    means.append(float(m["mean"]))
    correct = int(round(sum(means) / len(means) * total)) if (means and total) else 0
    return total, correct


def _truncation_livelock(trial_dir: str) -> Dict[str, Any]:
    """Consecutive truncated turns with no step recorded between them.

    A turn that hits the output cap is re-asked. If it truncates again, and again, the
    agent makes no progress and the trial dies as a bare `AgentTimeoutError` - the result
    looks identical to a hard task. On 2026-08-18 two trials sat at 9 and 7 consecutive
    truncations having recorded 2 and 10 steps, and nothing in the result said so; the
    evidence was only in Harbor's `trial.log`, inside a jobs-dir that is deleted on exit.
    """
    path = os.path.join(trial_dir, "trial.log")
    if not os.path.isfile(path):
        return {}
    run = worst = total = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Output length exceeded" in line:
                    run += 1
                    total += 1
                    worst = max(worst, run)
                elif "Trajectory dumped" in line:
                    run = 0
    except OSError as e:
        logger.debug("Could not read %s: %s", path, e)
        return {}
    if not total:
        return {}
    return {"truncated_turns": total, "longest_truncation_run": worst,
            "unresolved_truncations": run,
            "livelocked": max(worst, run) >= _LIVELOCK_TRUNCATIONS}


def _trajectory_text(trial_dir: str) -> Tuple[str, Optional[int]]:
    """Agent transcript and turn count from Terminus-2's ATIF trajectory files.

    Harbor writes the agent's conversation to `<trial_dir>/agent/trajectory.json` (plus
    `trajectory.cont-N.json` when context summarization splits it), NOT into result.json -
    `AgentContext` carries only token counts, cost and rollout details, which is why
    probing result.json for `messages`/`n_turns` returned nothing on the 2026-08-18 run and
    left two `AgentTimeoutError` trials unexplained. Terminus dumps after every episode and
    again in a `finally`, so the file exists even when the agent times out.
    """
    agent_dir = os.path.join(trial_dir, "agent")
    if not os.path.isdir(agent_dir):
        return "", None
    names = sorted(n for n in os.listdir(agent_dir)
                   if n == "trajectory.json" or n.startswith("trajectory.cont-"))
    parts: List[str] = []
    steps = 0
    for name in names:
        try:
            with open(os.path.join(agent_dir, name), "r", encoding="utf-8") as f:
                traj = json.load(f)
        except Exception as e:
            logger.debug("Could not parse %s: %s", name, e)
            continue
        for step in traj.get("steps") or []:
            if not isinstance(step, dict):
                continue
            steps += 1
            for chunk in (step.get("reasoning_content"), step.get("message")):
                if isinstance(chunk, str) and chunk.strip():
                    parts.append(chunk)
            for call in step.get("tool_calls") or []:
                if isinstance(call, dict):
                    parts.append(json.dumps(call.get("arguments") or call, default=str))
    return "\n".join(parts)[:_MAX_TRANSCRIPT_CHARS], (steps or None)


def _agent_transcript(agent: dict, data: dict) -> str:
    """Best-effort agent transcript from Harbor's result.json.

    Only reachable when the agent was run with `store_all_messages`; the normal path is
    `_trajectory_text`. Kept as a fallback because Harbor's schema varies by agent and
    version. Returns "" when none is present.
    """
    meta = agent.get("metadata") if isinstance(agent.get("metadata"), dict) else {}
    for src in (meta.get("all_messages"), agent.get("messages"), agent.get("transcript"),
                agent.get("history"), data.get("messages"), data.get("transcript")):
        if isinstance(src, list) and src:
            parts = []
            for m in src:
                if isinstance(m, dict):
                    parts.append(str(m.get("content") or m.get("text") or ""))
                else:
                    parts.append(str(m))
            return "\n".join(p for p in parts if p)[:_MAX_TRANSCRIPT_CHARS]
        if isinstance(src, str) and src.strip():
            return src[:_MAX_TRANSCRIPT_CHARS]
    for k in ("agent_output", "output", "stdout", "logs"):
        v = agent.get(k) or data.get(k)
        if isinstance(v, str) and v.strip():
            return v[:_MAX_TRANSCRIPT_CHARS]
    return ""


def harbor_image() -> str:
    """The local harness image gbench runs: HARBOR_IMAGE_DEFAULT, the tag
    harbor_image_build_command produces."""
    return HARBOR_IMAGE_DEFAULT


def harbor_image_build_command(image: Optional[str] = None) -> str:
    """The documented build command of the harness image (run from the repository root)."""
    return (f"docker build -t {image or harbor_image()} "
            f"-f gbench/docker/terminal_bench.Dockerfile gbench/docker")


def harbor_image_record(image: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """`{"image", "id", "labels"}` of the local harness image, or None when it is not on this
    host (or docker cannot tell)."""
    image = image or harbor_image()
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format",
                            "{{json .Id}} {{json .Config.Labels}}", image],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    image_id, _, labels = (r.stdout or "").strip().partition(" ")
    try:
        parsed = json.loads(labels or "null")
        image_id = json.loads(image_id or "null")
    except ValueError:
        parsed, image_id = None, None
    return {"image": image, "id": image_id if isinstance(image_id, str) else None,
            "labels": parsed if isinstance(parsed, dict) else {}}


def docker_socket_path() -> Tuple[Optional[str], str]:
    """(host path of the daemon socket the docker CLI talks to, "") or (None, why not usable).
    DOCKER_HOST when set, else the current docker context's endpoint (`docker context inspect`),
    else the default: a unix:// endpoint's path, which the harness container gets mounted at
    /var/run/docker.sock (so Harbor there drives the daemon gbench checked and probed). Harbor in
    the container can drive no other kind of endpoint (tcp://, ssh://, npipe://). The reason never
    repeats the endpoint, which may carry a user name."""
    host, source = os.environ.get("DOCKER_HOST", "").strip(), "DOCKER_HOST"
    if not host:
        source = "the current docker context"
        try:
            r = subprocess.run(["docker", "context", "inspect", "--format",
                                "{{.Endpoints.docker.Host}}"], capture_output=True, text=True,
                               timeout=20)
            host = (r.stdout or "").strip() if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            host = ""
    if not host or host == _DEFAULT_DOCKER_HOST:
        return _DOCKER_SOCKET, ""
    if host.startswith("unix://") and len(host) > len("unix://"):
        return host[len("unix://"):], ""
    scheme = host.split("://", 1)[0] if "://" in host else "non-unix"
    return None, (f"{source} points the docker CLI at a {scheme}:// daemon endpoint; Harbor runs "
                  "in the harness container and drives the daemon through a unix socket mounted "
                  "into it, so only a unix:// endpoint (a local socket) can be used. Point "
                  "DOCKER_HOST (or the docker context) at the local daemon's socket")


#: The host socket check_terminal_bench_prerequisites last resolved (docker_socket_path), which
#: _container_command mounts.
_VERIFIED_SOCKET: Dict[str, str] = {}


def _socket_source() -> str:
    """The host path of the docker socket the harness container gets (see docker_socket_path)."""
    return _VERIFIED_SOCKET.get("path") or _DOCKER_SOCKET


#: The harness image check_terminal_bench_prerequisites last verified in this process
#: (harbor_image_record), recorded as the result's `harness_image`.
_VERIFIED_IMAGE: Dict[str, Any] = {}


def check_terminal_bench_prerequisites() -> Tuple[bool, str]:
    """Docker (CLI and daemon), the docker socket the harness container mounts, and the local
    harness image with Harbor HARBOR_VERSION built from TB21_LOCK_COMMIT's lock (its labels,
    HARBOR_IMAGE_LABELS)."""
    _VERIFIED_IMAGE.clear()
    image = harbor_image()
    build = (f"Build the harness image LOCALLY (gbench never pulls it), from the repository "
             f"root: {harbor_image_build_command(image)}")
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH. " + build

    try:
        res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=20)
        if res.returncode != 0:
            return False, f"Docker daemon is not running or accessible: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    _VERIFIED_SOCKET.clear()
    socket_path, why = docker_socket_path()
    if socket_path is None:
        return False, why
    if not os.path.exists(socket_path):
        return False, (f"{socket_path} does not exist; Harbor runs in the harness container "
                       "and starts the task containers through that socket, mounted into it.")
    _VERIFIED_SOCKET["path"] = socket_path

    record = harbor_image_record(image)
    if record is None:
        return False, (f"the terminal_bench harness image {image!r} (Harbor {HARBOR_VERSION}, "
                       f"the TB2.1 leaderboard's release) is not on this host. " + build)
    wrong = {k: record["labels"].get(k) for k, v in HARBOR_IMAGE_LABELS.items()
             if record["labels"].get(k) != v}
    if wrong:
        return False, (f"the image {image!r} was built with {wrong}, not {HARBOR_IMAGE_LABELS} "
                       f"(Harbor {HARBOR_VERSION}, terminal-bench-2-1 leaderboard/uv.lock at "
                       f"{TB21_LOCK_COMMIT[:7]}, whose result.json schema gbench's parsing "
                       f"follows); rebuild it. " + build)
    _VERIFIED_IMAGE.update(record)
    return True, ""


def harbor_cache_dir() -> str:
    """gbench's Harbor cache on this host, `$GBENCH_CACHE_DIR/harbor` (default
    ~/.cache/gbench/harbor), mounted as the harness container's ~/.cache/harbor: the task packages
    Harbor downloads (`tasks/packages/<org>/<task>/<content hash>/`) live here."""
    from .base import gbench_cache_dir
    return os.path.join(gbench_cache_dir(), "harbor")


def harbor_package_cache() -> str:
    """Harbor's package task cache on this host (harbor.constants.PACKAGE_CACHE_DIR in the
    harness container): `<org>/<task>/<content hash>/task.toml` (Harbor 0.14 and later) or
    `<org>/<task>/task.toml` (older releases)."""
    return os.path.join(harbor_cache_dir(), "tasks", "packages")


def container_env_names(env: Dict[str, str]) -> List[str]:
    """The variables of `env` (the environment the docker client runs with) the harness container
    gets, by name: OPENAI_API_KEY and OPENAI_BASE_URL when set, then the Harbor and proxy
    settings that are set (_PASSED_ENV_PREFIXES, _PASSED_ENV_NAMES), except the ones gbench always
    sets itself (_FORCED_ENV). Nothing else of this host's environment reaches the container."""
    names = [n for n in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if n in env]
    names += sorted(n for n in env if n not in names and n not in _FORCED_ENV and (
        n in _PASSED_ENV_NAMES or n.startswith(_PASSED_ENV_PREFIXES)))
    return names


def _forced_env_args() -> List[str]:
    """`-e NAME=value` for every variable of _FORCED_ENV."""
    return [a for k, v in sorted(_FORCED_ENV.items()) for a in ("-e", f"{k}={v}")]


def _container_command(argv: Sequence[str], *, jobs_dir: Optional[str] = None,
                       name: Optional[str] = None, env_names: Sequence[str] = (),
                       docker_socket: bool = True, salvage_patch: bool = False) -> List[str]:
    """`docker run` of `argv` in the harness image: removed on exit, tini as PID 1, the host
    network, every capability dropped, no-new-privileges, the invoking user (with the docker
    socket's group), HOME the image's, gbench's Harbor cache at ~/.cache/harbor, _FORCED_ENV
    (Harbor's telemetry off), the variables `env_names` by name (docker copies them from the
    client's environment; a name in _FORCED_ENV is not passed); with `docker_socket` the socket,
    with `jobs_dir` that directory at the same path, with `salvage_patch` the patch file,
    read-only."""
    cmd = ["docker", "run", "--rm", "--init", "--network", "host", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges"]
    if name:
        cmd += ["--name", name]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if docker_socket:
            try:
                cmd += ["--group-add", str(os.stat(_socket_source()).st_gid)]
            except OSError:
                pass
    if docker_socket:
        cmd += ["-v", f"{_socket_source()}:{_DOCKER_SOCKET}"]
    if jobs_dir:
        jobs_dir = os.path.abspath(jobs_dir)
        cmd += ["-v", f"{jobs_dir}:{jobs_dir}:rw"]
    cmd += ["-v", f"{harbor_cache_dir()}:{_IMAGE_HARBOR_CACHE}:rw", "-e", f"HOME={_IMAGE_HOME}"]
    if salvage_patch:
        cmd += ["-v", f"{os.path.abspath(terminus_salvage_patch.__file__)}:"
                      f"{_IMAGE_SALVAGE_PATCH}:ro"]
    cmd += _forced_env_args()
    for env_name in env_names:
        if env_name.split("=", 1)[0] not in _FORCED_ENV:
            cmd += ["-e", env_name]
    return cmd + [harbor_image(), *argv]


def harbor_command(args: List[str], *, jobs_dir: Optional[str] = None,
                   name: Optional[str] = None, env_names: Sequence[str] = ()) -> List[str]:
    """`harbor <args>` in the harness image (_container_command, with the docker socket and
    `jobs_dir` at the same path): Harbor's CLI with the salvage patch installed first
    (terminus_salvage_patch's bootstrap over the mounted patch file) unless
    GBENCH_TB_SALVAGE_PATCH=0, which runs the image's `harbor` unpatched."""
    patched = terminus_salvage_patch.enabled()
    argv = (["python", "-c", terminus_salvage_patch._bootstrap_source(_IMAGE_SALVAGE_PATCH)]
            if patched else ["harbor"])
    return _container_command([*argv, *args], jobs_dir=jobs_dir, name=name, env_names=env_names,
                              salvage_patch=patched)


def harbor_download_command() -> str:
    """The shell command that downloads the pinned dataset (DATASET_SPEC) with the harness image
    into gbench's Harbor cache, as the invoking user (`--cache`: Harbor 0.20.0's content-addressed
    cache layout, the one `harbor run` reads; without it the tasks are exported to the working
    directory), with Harbor's telemetry off (_FORCED_ENV)."""
    cache = harbor_cache_dir()
    return (f"mkdir -p {cache} && docker run --rm --network host --user \"$(id -u):$(id -g)\" "
            f"-v {cache}:{_IMAGE_HARBOR_CACHE} {' '.join(_forced_env_args())} {harbor_image()} "
            f"harbor dataset download {DATASET_SPEC} --cache")


def _prepare_harbor_cache() -> None:
    """Create gbench's Harbor cache, so docker mounts a directory the invoking user owns."""
    os.makedirs(harbor_cache_dir(), exist_ok=True)


def _stop_container(name: str, proc: Any = None) -> None:
    """Stop and remove the harness container `name` after the docker client that ran it was
    interrupted or killed: the client first (it forwards SIGTERM), then `docker stop` (Harbor gets
    SIGTERM and 30 seconds to stop its task containers) and `docker rm -f`. Never raises."""
    try:
        if proc is not None and proc.poll() is None:
            proc.terminate()
    except Exception:                                                   # noqa: BLE001
        pass
    for argv in (["docker", "stop", "--time", "30", name], ["docker", "rm", "-f", name]):
        try:
            subprocess.run(argv, capture_output=True, timeout=120)
        except Exception:                                               # noqa: BLE001
            pass


def _task_image(task_dir: str) -> Optional[str]:
    """`[environment] docker_image` of the task.toml in `task_dir`, or of the newest one in its
    content-hash subdirectories; None when there is none."""
    import tomllib
    candidates = [os.path.join(task_dir, "task.toml")]
    try:
        subdirs = [os.path.join(task_dir, d) for d in os.listdir(task_dir)]
    except OSError:
        return None
    subdirs = [d for d in subdirs if os.path.isfile(os.path.join(d, "task.toml"))]
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(d, "task.toml")), reverse=True)
    candidates += [os.path.join(d, "task.toml") for d in subdirs]
    for path in candidates:
        try:
            with open(path, "rb") as f:
                image = ((tomllib.load(f).get("environment") or {}).get("docker_image") or "")
        except (OSError, ValueError):
            continue
        if isinstance(image, str) and image.strip():
            return image.strip()
    return None


def harbor_task_images(dataset: str = DEFAULT_DATASET,
                       task_names: Optional[List[str]] = None) -> List[str]:
    """The task images (`[environment] docker_image` of each task.toml) of the tasks Harbor's
    local task cache holds under `dataset`'s organisation, each once; with `task_names` (`org/name`
    each, as `--include-task-name` takes them), those tasks only, in that order. The cache is per
    organisation, so without `task_names` a cached task of another dataset of it counts too. []
    when none is cached (Harbor downloads the tasks on the first `harbor run` or `harbor dataset
    download` in the harness container, into gbench's cache: harbor_package_cache)."""
    org = dataset.split("/", 1)[0]
    org_dir = os.path.join(harbor_package_cache(), org)
    if task_names is not None:
        entries = [n.split("/", 1)[1] for n in task_names if n.startswith(org + "/")]
    else:
        try:
            entries = sorted(os.listdir(org_dir))
        except OSError:
            return []
    images: List[str] = []
    for entry in entries:
        image = _task_image(os.path.join(org_dir, entry))
        if image:
            images.append(image)
    return list(dict.fromkeys(images))


def require_task_metadata_blocked(task_names: Optional[List[str]] = None) -> None:
    """The cloud metadata server gate before Harbor starts any task container
    (swebench_common.require_task_network_blocked). The agent's commands and the verifier's tests
    run in each task's container, on the compose network Harbor creates for it (a user-defined
    bridge network) with network access, and each verifier installs its test tools there at run
    time, so the package hosts it needs must resolve (swebench_common.TASK_INSTALL_HOSTS).

    The probe runs in an image already on the host: a task image from the local task cache (the
    images of `task_names`, the run's tasks when the run names them, first, then any other cached
    task's), else the harness image the prerequisite check verified (it has python3 and joins the
    same docker networks; the network verdict does not depend on the image). Nothing is pulled
    for it. Only when none of them is present is the check infra_required, naming what is
    missing."""
    selected = harbor_task_images(DEFAULT_DATASET, task_names) if task_names else []
    images = list(dict.fromkeys([*selected, *harbor_task_images(DEFAULT_DATASET),
                                 harbor_image()]))
    if len(images) > 1:
        hint = (f"Any task image of the dataset serves the probe; Harbor pulls each task's image "
                f"itself when it starts that task, after this check (see {DOCS_URL}).")
    else:
        hint = (f"gbench's Harbor cache ({harbor_package_cache()}) holds no task of the dataset, "
                f"so gbench cannot name a task image, and the harness image {harbor_image()} is "
                f"not present either; build it ({harbor_image_build_command()}) (see "
                f"{DOCS_URL}).")
    swebench_common.require_task_network_blocked(
        "terminal_bench", images, docs_url=DOCS_URL, hint=hint,
        resolve=swebench_common.TASK_INSTALL_HOSTS)


def _cached_task_names(task_names: Optional[List[str]]) -> Tuple[int, int]:
    """(tasks of the run with a task.toml in gbench's Harbor cache, tasks of the run): the named
    tasks, or with `task_names` None the whole pinned dataset (FULL_TASK_COUNT)."""
    org = DEFAULT_DATASET.split("/", 1)[0]
    org_dir = os.path.join(harbor_package_cache(), org)
    if task_names is not None:
        entries = [n.split("/", 1)[1] for n in task_names if n.startswith(org + "/")]
        wanted = len(task_names)
    else:
        try:
            entries = sorted(os.listdir(org_dir))
        except OSError:
            entries = []
        wanted = FULL_TASK_COUNT
    have = sum(1 for e in entries if _task_image(os.path.join(org_dir, e)))
    return min(have, wanted), wanted


def ensure_dataset_cached(task_names: Optional[List[str]] = None,
                          run: Any = None) -> Dict[str, Any]:
    """Download the pinned dataset (DATASET_SPEC) into gbench's Harbor cache with the harness
    image (`harbor dataset download --cache`, as harbor_download_command; no docker socket, as
    the invoking user) when the cache lacks a task of the run, so its task images are known
    before Harbor starts (for the pre-pull and the metadata probe). `harbor run` downloads into
    the same cache on first use, so nothing the run measures changes. A failed download is
    logged and recorded; Harbor then downloads the tasks itself. Returns {cached, of, download:
    None | {returncode, cached_after}}."""
    run = run or subprocess.run
    have, wanted = _cached_task_names(task_names)
    record: Dict[str, Any] = {"cached": have, "of": wanted, "download": None}
    if have >= wanted:
        return record
    _prepare_harbor_cache()
    env = _credential_free_env()
    cmd = _container_command(["harbor", "dataset", "download", DATASET_SPEC, "--cache"],
                             env_names=container_env_names(env), docker_socket=False)
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=1800, env=env,
                   stdin=subprocess.DEVNULL)
        rc = proc.returncode
    except (OSError, subprocess.SubprocessError) as e:
        rc = f"{type(e).__name__}"
    after, _ = _cached_task_names(task_names)
    record["download"] = {"returncode": rc, "cached_after": after}
    if rc != 0:
        logger.warning("[terminal_bench] the dataset download into gbench's Harbor cache ended "
                       "with %s; Harbor downloads the tasks itself when the run starts", rc)
    return record


def prepare_task_images(task_names: Optional[List[str]] = None
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(ensure_dataset_cached's record, prepull_task_images' record) for the run's tasks
    (`task_names`, None for the whole pinned dataset): what runs before Harbor starts."""
    dataset_cache = ensure_dataset_cached(task_names)
    images = (harbor_task_images(DEFAULT_DATASET, task_names) if task_names
              else harbor_task_images(DEFAULT_DATASET))
    return dataset_cache, prepull_task_images(images)


def prepull_task_images(images: Sequence[str],
                        concurrency: int = PREPULL_CONCURRENCY) -> Dict[str, Any]:
    """Make each of the run's task `images` present on this host before Harbor starts
    (swebench_common.host_pull_images: `docker image inspect`, and `docker pull` only for an
    image that is missing, run by this process, so the Docker CLI uses the operator's registry
    login), `concurrency` at a time. Harbor in the harness container has no docker client
    configuration (none is mounted), so its own pulls are anonymous and meet the registry's
    unauthenticated rate limit; it starts each task with `docker compose up` (pull policy:
    missing), so an image present here is not pulled again. A failed pull is recorded (Harbor
    tries again itself; a task whose environment then cannot start is a harness error, never the
    model's). Returns swebench_common.image_pull_summary's record ({images, present, pulled,
    failed: {image: error}})."""
    pulls = swebench_common.host_pull_images(list(images), workers=concurrency)
    summary = swebench_common.image_pull_summary(pulls)
    if summary["failed"]:
        logger.warning("[terminal_bench] %d task image(s) could not be pulled on this host (%s); "
                       "Harbor tries again itself", len(summary["failed"]),
                       ", ".join(sorted(summary["failed"])))
    return summary


def _trial_text(trial_dir: str, *patterns: str) -> str:
    """The text of the files under `trial_dir` matching the glob `patterns`."""
    import glob
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(os.path.join(trial_dir, pattern))))
    return swebench_common.read_texts(list(dict.fromkeys(paths)))


def trial_dns_harness_error(trial_dir: str, passed: bool) -> Dict[str, Any]:
    """The DNS evidence of one trial: {} when there is nothing to record; else
    `dns_failure_hosts` (the package hosts its verifier could not resolve before its tests
    started), with `dns_failure_note` when the agent's transcript names a resolver file (the
    model may have broken name resolution itself; the trial stays the model's outcome). Without
    the note the trial is a DNS candidate: attribute_dns_failures makes it a harness error once
    something outside its container corroborates the failure. A passed trial is always the
    model's outcome."""
    if passed:
        return {}
    hosts = swebench_common.verifier_dns_failure(
        _trial_text(trial_dir, os.path.join("verifier", "test-stdout.txt"),
                    os.path.join("verifier", "test-stderr.txt")))
    if not hosts:
        return {}
    if swebench_common.touched_resolver(_trial_text(trial_dir, os.path.join("agent", "*.json"))):
        return {"dns_failure_hosts": hosts,
                "dns_failure_note": ("the verifier could not resolve its package hosts, and the "
                                     "agent's transcript names /etc/resolv.conf, /etc/hosts or "
                                     "/etc/nsswitch.conf, so the model may have broken name "
                                     "resolution in its container: the trial stays the model's "
                                     "outcome")}
    return {"dns_failure_hosts": hosts}


def dns_recheck(task_names: Optional[List[str]], hosts: List[str]) -> Optional[List[str]]:
    """The hosts of `hosts` a fresh probe container cannot resolve after the run, on the task
    networks, in a cached task image already on the host (the run's own tasks first); None when
    it could not tell (swebench_common.unresolved_after_run)."""
    selected = harbor_task_images(DEFAULT_DATASET, task_names) if task_names else []
    images = list(dict.fromkeys([*selected, *harbor_task_images(DEFAULT_DATASET)]))
    return swebench_common.unresolved_after_run("terminal_bench", images, hosts,
                                                docs_url=DOCS_URL)


def attribute_dns_failures(trials: List[Dict[str, Any]],
                           task_names: Optional[List[str]] = None) -> None:
    """Mark the DNS candidates of `trials` (dns_failure_hosts without a note, not already a
    harness error): `harness_error: "dns"` with `dns_failure_evidence` when
    swebench_common.corroborated_dns_failures finds the failure corroborated (another candidate
    in the run, or dns_recheck after the run), else `dns_failure_note` (the model's outcome)."""
    candidates = {t["trial_name"]: t["dns_failure_hosts"] for t in trials
                  if t.get("dns_failure_hosts") and not t.get("dns_failure_note")
                  and not t.get("harness_error")}
    errors, kept = swebench_common.corroborated_dns_failures(
        candidates, recheck=lambda hosts: dns_recheck(task_names, hosts))
    for t in trials:
        name = t.get("trial_name")
        if name in errors:
            t["harness_error"] = "dns"
            t["dns_failure_evidence"] = errors[name]
        elif name in kept:
            t["dns_failure_note"] = kept[name]


#: Harbor exceptions after which a trial is still the model's outcome. AgentTimeoutError (the
#: agent ran out of its budget; Harbor still runs the verifier) is in Harbor's default
#: RetryConfig.exclude_exceptions. NonZeroAgentExitCodeError (an installed agent's process exited
#: non-zero; Harbor still runs the verifier), ContextLengthExceededError and
#: OutputLengthExceededError (the model's own context or output ran past a limit Terminus-2 could
#: not recover from) are gbench's own attribution: Harbor's default retry policy would retry
#: them, and its metrics count any trial without a reward as 0.
AGENT_OUTCOME_EXCEPTIONS = ("AgentTimeoutError", "NonZeroAgentExitCodeError",
                            "ContextLengthExceededError", "OutputLengthExceededError")
#: Harbor exceptions of a verifier that started and ran the tests on the model's work without a
#: usable reward: its budget ran out, or the reward file is missing, empty or unparsable. They
#: are in Harbor's default RetryConfig.exclude_exceptions, and Harbor's metrics count a trial
#: without a reward as 0 (swe_bench_pro's regrade_verdict follows the same rule).
VERIFIER_OUTCOME_EXCEPTIONS = ("VerifierTimeoutError", "RewardFileNotFoundError",
                               "RewardFileEmptyError", "VerifierOutputParseError")
#: Model-endpoint failures: LiteLLM's exception classes (litellm.exceptions) as Terminus-2
#: re-raises them after its own retries and those of Harbor's LiteLLM wrapper (which turns a
#: context-length refusal into ContextLengthExceededError first), the OpenAI SDK names they
#: subclass, and an installed agent's ApiRateLimitError. The served endpoint is gbench's side of
#: the run, not the model's. One list for the SWE family (swebench_common).
ENDPOINT_EXCEPTIONS = swebench_common.ENDPOINT_EXCEPTIONS
#: A trial Harbor itself stopped.
CANCELLED_EXCEPTIONS = ("CancelledError",)
#: What the docker CLI prints when it cannot reach its daemon. Nothing in a task container can
#: stop the host's docker daemon. One list for the SWE family (swebench_common).
DOCKER_DAEMON_MARKERS = swebench_common.DOCKER_DAEMON_MARKERS
#: docker's words for a task container that had stopped, or for a copy target the container's
#: own files made unusable. The model's commands can do both (stop the container's main process,
#: replace /tests with a file, ...), so a verifier transfer that fails with one of them stays the
#: model's outcome.
CONTAINER_STATE_MARKERS = ("is not running", "is restarting", "no container found",
                           "not a directory", "read-only file system", "cannot overwrite")


def _phase_started(data: Dict[str, Any], phase: str) -> bool:
    """Whether a trial's result.json records `phase` (agent_execution, verifier) as started."""
    timing = data.get(phase)
    return isinstance(timing, dict) and bool(timing.get("started_at"))


def trial_exception_verdict(data: Dict[str, Any], reward: Optional[float]
                            ) -> Tuple[Optional[str], Optional[str]]:
    """(harness error reason, model-outcome note) of a trial's `result.json` (`data`): the reason
    is set when the trial is a harness error, the note when an exception outside the known
    harness failures stays the model's outcome; (None, None) otherwise.

    No exception, or a reward (the verifier finished, and Harbor's metrics use it whatever else
    the trial recorded), is the model's outcome. Otherwise, by the phase the result records
    (Harbor sets `agent_execution.started_at` when the agent starts and `verifier.started_at`
    when the verifier does) and the exception's type:

    * harness errors: a cancelled trial (CANCELLED_EXCEPTIONS); a model-endpoint failure
      (ENDPOINT_EXCEPTIONS); an exception before the agent started (environment or docker start,
      healthcheck, agent setup); an exception naming an unreachable docker daemon
      (DOCKER_DAEMON_MARKERS); a verifier that started and then failed outside
      VERIFIER_OUTCOME_EXCEPTIONS (AddTestsDirError, DownloadVerifierDirError, ...), unless
      docker's message shows the task container had stopped or its target path was unusable
      (CONTAINER_STATE_MARKERS);
    * the model's outcome: AGENT_OUTCOME_EXCEPTIONS; VERIFIER_OUTCOME_EXCEPTIONS of a verifier
      that started; a verifier failure with a CONTAINER_STATE_MARKERS message (with a note); any
      other exception raised while the agent ran (with a note), e.g. the RuntimeError tmux
      send-keys raises after the model's own keystrokes ended its tmux session or container:
      not a known harness failure, and Harbor's metrics count a trial without a reward as 0."""
    info = data.get("exception_info") if isinstance(data.get("exception_info"), dict) else {}
    exc = info.get("exception_type")
    if not exc or reward is not None:
        return None, None
    exc = str(exc)
    message = " ".join(str(info.get("exception_message") or "").split())
    reason = f"{exc}: {message[:200]}" if message else exc
    if exc in CANCELLED_EXCEPTIONS or exc in ENDPOINT_EXCEPTIONS:
        return reason, None
    if exc in AGENT_OUTCOME_EXCEPTIONS:
        return None, None
    text = f"{message} {info.get('exception_traceback') or ''}".lower()
    if _phase_started(data, "verifier"):
        if exc in VERIFIER_OUTCOME_EXCEPTIONS:
            return None, None
        marker = next((m for m in CONTAINER_STATE_MARKERS if m in text), None)
        if marker and not any(m in text for m in DOCKER_DAEMON_MARKERS):
            return None, (f"{exc} after the verifier started, and docker's message shows the "
                          f"task container had stopped or its target path was unusable "
                          f"({marker!r}), which the model's commands can cause: the trial stays "
                          "the model's outcome (Harbor's metrics count a trial without a reward "
                          "as 0)")
        return reason, None
    if not _phase_started(data, "agent_execution"):
        return reason, None           # before the model's first turn
    if any(m in text for m in DOCKER_DAEMON_MARKERS):
        return reason, None
    return None, (f"{exc} raised while the agent ran, which is not a known harness failure (the "
                  "model's own commands can end its tmux session or its container): the trial "
                  "stays the model's outcome (Harbor's metrics count a trial without a reward as "
                  "0)")


def trial_exception_harness_error(data: Dict[str, Any], reward: Optional[float]) -> Optional[str]:
    """None when a trial's `result.json` (`data`) is the model's outcome, else the reason it is
    a harness error (trial_exception_verdict)."""
    return trial_exception_verdict(data, reward)[0]


def _parse_shard() -> Optional[Tuple[int, int]]:
    """`GBENCH_SHARD="I/N"` -> `(I, N)` (1-indexed shard I of N), or None when unset/empty, read
    with base.current_shard: a malformed or out-of-range spec raises ValueError, so a typo
    surfaces instead of silently running the whole set."""
    shard = current_shard()
    return (shard.index, shard.count) if shard else None


def _dataset_task_names(dataset: str) -> List[str]:
    """Full, deterministically SORTED list of task names (`org/name` each) in `dataset`, a
    Harbor package reference (`org/name@ref`; DATASET_SPEC, the pinned content hash).

    Enumerated through Harbor's OWN package-dataset resolver so the strings are byte-identical to
    the ones `harbor run`'s `--include-task-name` filter fnmatches at run time (it compares
    `PackageTaskId.get_name()`, which is `f"{org}/{name}"`). `harbor run -d` is given the same
    reference (cli/jobs.py: `org/name@ref` -> DatasetConfig(name, ref)), so both see the same
    dataset version. A bare `org/name` resolves to `@latest`, as Harbor resolves it.

    Sorted+deduped here because Harbor returns task ids in DB order; `sampling.shard_select` requires
    a stable, reproducible order so the round-robin partition (`sorted[I-1::N]`) is non-overlapping
    and unions back to the full set across machines. The resolver runs in the harness image
    (TASK_NAMES_SCRIPT, _container_command without the docker socket), the Harbor that runs the
    job; a failure raises RuntimeError with Harbor's output.
    """
    ref = dataset if "@" in dataset else f"{dataset}@latest"
    _prepare_harbor_cache()
    env = _credential_free_env()
    cmd = _container_command(["python", "-c", TASK_NAMES_SCRIPT, ref],
                             env_names=container_env_names(env), docker_socket=False)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env,
                          stdin=subprocess.DEVNULL)
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    names: Any = None
    if proc.returncode == 0 and lines:
        try:
            names = json.loads(lines[-1])
        except ValueError:
            names = None
    if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
        tail = " ".join(((proc.stderr or "") + " " + (proc.stdout or "")).split())[-600:]
        raise RuntimeError(f"Harbor's resolver in {harbor_image()} exited {proc.returncode} "
                           f"without a task list for {ref}: {tail}")
    return sorted(set(names))


#: Python the harness image runs to list a dataset's tasks: Harbor's own package resolver
#: (PackageDatasetClient.get_dataset_metadata of `sys.argv[1]`, an `org/name@ref`), each task's
#: `PackageTaskId.get_name()`, as a JSON list on the last line of stdout.
TASK_NAMES_SCRIPT = (
    "import asyncio, json, sys\n"
    "from harbor.registry.client.package import PackageDatasetClient\n"
    "async def _names(ref):\n"
    "    metadata = await PackageDatasetClient().get_dataset_metadata(ref)\n"
    "    return [tid.get_name() for tid in metadata.task_ids]\n"
    "print(json.dumps(asyncio.run(_names(sys.argv[1]))))\n")


def _credential_free_env() -> Dict[str, str]:
    """This process's environment without any credential (sandbox.credential_free_env): what the
    docker client that runs the harness container gets."""
    from .sandbox import credential_free_env
    return credential_free_env(os.environ.copy())


def run_terminal_bench(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run official Terminal-Bench 2.1 evaluation suite via Harbor framework."""
    ok, reason = check_terminal_bench_prerequisites()
    if not ok:
        # No-skip policy: a missing prerequisite must HARD-ERROR, never emit a fabricated 0%/skip
        # row. evals.py isolates the raise into a status:"error" row and continues the sweep.
        raise infra_required("terminal_bench", reason, DOCS_URL)
    # The harness the run executes: the image the check verified (its labels carry
    # HARBOR_VERSION; the check refuses any other release).
    harness_image = {"image": harbor_image(), "id": _VERIFIED_IMAGE.get("id"),
                     "harbor": HARBOR_VERSION, "lock": TB21_LOCK_COMMIT}

    start_time = time.time()
    limit = kwargs.get("limit")
    limit = int(limit) if limit and int(limit) > 0 else None
    total = 0
    correct = 0
    sample_traces: List[Dict[str, Any]] = []
    # Trials per task (Harbor -k): the leaderboard's 5, or a larger explicit --attempt-count.
    attempts_requested = int(kwargs.get("attempt_count") or 1)
    n_trials, trials_source = trial_count(attempts_requested)

    # Task selection (see the module docstring): with a shard or a limit, the pinned dataset's
    # sorted task names, shard I of N first, then a seeded draw of --eval-limit from the shard,
    # passed to Harbor as --include-task-name (never --n-tasks, which takes Harbor's first N in
    # registry order).
    shard_spec = current_shard()
    shard = (shard_spec.index, shard_spec.count) if shard_spec else None
    include_names: Optional[List[str]] = None
    shard_size: Optional[int] = None
    if shard or limit:
        try:
            full = _dataset_task_names(DATASET_SPEC)
        except Exception as e:
            # No fabrication: without the real task list the subset cannot be built, and running
            # the full set (or Harbor's first N) would misreport what was measured.
            wanted = (f"--shard {shard[0]}/{shard[1]}" if shard else "") + (
                f"{' and ' if shard else ''}--eval-limit {limit}" if limit else "")
            raise infra_required(
                "terminal_bench",
                f"{wanted} was requested but Harbor's registry could not be queried to enumerate "
                f"{DATASET_SPEC}'s task list ({type(e).__name__}: {e}); the subset cannot be "
                f"built without it", DOCS_URL) from e
        if not full:
            raise infra_required("terminal_bench", f"Harbor's registry listed no task for "
                                 f"{DATASET_SPEC}; nothing can be selected", DOCS_URL)
        include_names = select_task_names(full, shard, limit)
        shard_size = len(include_names)
        logger.info("[terminal_bench] %s%d of %d tasks%s", (f"shard {shard[0]}/{shard[1]}: "
                                                          if shard else ""),
                    shard_size, len(full),
                    f" (drawn for --eval-limit {limit})" if limit else "")
        if shard_size == 0:
            # An empty shard (more shards than tasks) must NOT invoke Harbor: zero
            # `--include-task-name` filters would run the FULL set. Nothing was measured here, so
            # the score is withheld (never a 0).
            empty = (f"empty shard {shard[0]}/{shard[1]}: no items on this node "
                     f"({len(full)} in the full set)")
            logger.warning("[terminal_bench] %s; nothing to run on this node.", empty)
            temperature, temperature_source = resolve_official_sampling(
                "terminal_bench", TERMINAL_BENCH_SAMPLING, thinking=enable_thinking)
            return {
                "benchmark_type": "eval",
                "eval_name": "terminal_bench",
                "model_name": model_name,
                "thinking": enable_thinking,
                "total_questions": 0,
                "correct_answers": 0,
                "failed_requests": 0,
                "accuracy": None,
                "accuracy_withheld_reason": empty,
                "empty_shard": True,
                "full_set_items": len(full),
                "category_accuracy": {},
                "sample_traces": [],
                "selected_tasks": [],
                "turn_budget": None,
                "temperature": temperature,
                "temperature_source": temperature_source,
                "harbor_version": HARBOR_VERSION,
                "harness_image": harness_image,
                "harness_telemetry": dict(HARNESS_TELEMETRY),
                "dataset": DEFAULT_DATASET,
                "dataset_ref": DATASET_REF,
                **shard_identity(shard_spec),
                "attempts": {"attempts_requested": attempts_requested,
                             "attempts_per_sample": n_trials, "source": trials_source},
                "is_full": False,
                "partial": False,
                "leaderboard_comparable": False,
                "leaderboard_comparable_reason": empty,
                "status": "success",
                "status_reason": empty,
                "duration_s": round(time.time() - start_time, 2),
            }

    # One task set, no settings: --eval-limit N is N tasks in total, each run its trials (Harbor
    # -k: repeats of one task, not extra samples).
    workload = eval_limit_plan(
        "terminal_bench", limit, len(include_names) if include_names is not None
        else FULL_TASK_COUNT, attempts=n_trials, unit="trials")

    # The model's commands run in the task containers, which have network access: the cloud
    # metadata server must be blocked there, and the verifiers' package hosts must resolve,
    # before anything of the run starts (the probe uses an image already on the host).
    require_task_metadata_blocked(include_names)

    # The run's tasks in gbench's Harbor cache (downloaded now when missing, as `harbor run`
    # would on first use), and their images pulled on this host with the user's registry login
    # (Harbor's own pulls in the harness container are anonymous).
    dataset_cache, prepull = prepare_task_images(include_names)

    relay = None
    with _jobs_dir() as tmp_dir, contextlib.ExitStack() as stack:
        # The docker client gets this process's environment without any credential, and the
        # harness container only the variables container_env_names names (Harbor, and through it
        # the task containers' compose environment, sees nothing else); Terminus-2 sends the
        # placeholder key, and an endpoint key is added by gbench's relay on 127.0.0.1 (never
        # handed to Harbor).
        env = _credential_free_env()
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint = f"{endpoint}/v1"
        # Always through gbench's relay, with or without an endpoint key (2026-10-11 smoke run:
        # keyless, the relay never started, so no max_tokens cap from the run's
        # --max-output-tokens, no smoke stop and no request-shape check reached Terminus-2's
        # requests); the relay adds the key only when there is one.
        key = endpoint_api_key()
        relay = swebench_common.EndpointRelay(endpoint, "127.0.0.1", key=key,
                                              name="terminal-bench-model-relay")
        # a Terminus-2 turn sends no max_tokens and may run to the end of the window: the
        # relay outlasts the client's own timeout (model_client_timeout)
        relay.TIMEOUT_S = max(swebench_common.EndpointRelay.TIMEOUT_S, 6 * 60 * 60,
                              model_client_timeout(_turn_token_cap(enable_thinking))
                              ["timeout_s"] + 120)
        try:
            relay.start()
        except OSError as e:
            raise infra_required("terminal_bench", f"gbench's relay to the model endpoint "
                                 f"could not listen on 127.0.0.1 ({type(e).__name__}: {e})",
                                 DOCS_URL) from None
        stack.callback(relay.stop)
        endpoint = relay.url
        env["OPENAI_BASE_URL"] = endpoint
        env["OPENAI_API_KEY"] = "EMPTY"

        # The temperature is gbench's rule (1.0 with --thinking, 0.0 without; the operator's
        # GBENCH_TERMINAL_BENCH_TEMPERATURE or --temperature replaces it); one other than the TB2.1
        # leaderboard's 1 (TERMINAL_BENCH_SAMPLING) is named below. No max_tokens in either thinking mode, as Terminus-2 and LiteLLM send
        # none (the token-limit policy's "not_sent": the server's default, the room left in the
        # context window, so a --thinking turn is never capped below --max-output-tokens).
        temperature, temperature_source = resolve_official_sampling(
            "terminal_bench", TERMINAL_BENCH_SAMPLING, thinking=enable_thinking)
        official = TERMINAL_BENCH_SAMPLING.record(temperature, temperature_source)
        plan = resolve_token_budget("terminal_bench", TERMINAL_BENCH_SAMPLING,
                                    thinking=enable_thinking,
                                    max_output_tokens=kwargs.get("max_output_tokens"))
        official.update(plan.record())
        parser_name = agent_parser()
        agent_name = kwargs.get("agent", "terminus-2")
        harbor_model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
        # GBENCH_TB_TURN_MAX_TOKENS: an opt-in per-turn cap (a backstop for a slow endpoint, where
        # one uncapped turn can run for most of a task's agent budget); none is sent by default.
        turn_cap = _turn_token_cap(enable_thinking)
        # timeout / max_retries reach litellm.acompletion (Harbor's LiteLLM.call merges
        # llm_call_kwargs into its completion kwargs): no hidden 600 s cap (model_client_timeout)
        client_timeout = model_client_timeout(turn_cap)
        llm_call_kwargs: Dict[str, Any] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            "timeout": client_timeout["timeout_s"],
            "max_retries": client_timeout["retries"],
        }
        if turn_cap:
            llm_call_kwargs["max_tokens"] = turn_cap
            tokens_note = (f"GBENCH_TB_TURN_MAX_TOKENS={turn_cap}: every agent turn was sent "
                           f"max_tokens={turn_cap} (an opt-in deviation; Terminus-2 sends none)")
        else:
            tokens_note = plan.note
        # GBENCH_TB_CONTAINER_THREADS (opt-in, none by default) reaches the agent and the verifier
        # through two separate channels, because they run through different code paths in Harbor
        # and neither inherits the other's environment:
        #   agent    -> `--agent-env` -> config.env -> terminus-2's `extra_env` ->
        #               `tmux new-session -e K=V`, so every command the agent types
        #               inherits it (tmux 3.4 in these images; -e needs >= 3.2)
        #   verifier -> `--verifier-env` -> config.verifier.env -> `override_env` on the
        #               verifier's exec
        # See the note at the flags themselves for why the agent side must NOT go through
        # `--agent-kwarg`.
        thread_env = container_thread_env()
        cmd = [
            "harbor",
            "run",
            "-d",
            DATASET_SPEC,
            "--agent",
            agent_name,
            "--model",
            harbor_model,
            "--n-concurrent",
            str(concurrency),
            "--jobs-dir",
            tmp_dir,
            "--agent-kwarg",
            f"api_base={endpoint}",
            "--agent-kwarg",
            f"temperature={temperature}",
            "--agent-kwarg",
            f"parser_name={parser_name}",
            "--agent-kwarg",
            f"llm_call_kwargs={json.dumps(llm_call_kwargs)}",
        ]
        for key, value in thread_env.items():
            # `--agent-env`, NOT `--agent-kwarg extra_env={...}`. Terminus-2 does take an
            # `extra_env` constructor kwarg, but AgentFactory already fills it from
            # `config.env` and then splats the agent kwargs on top:
            #     create_agent_from_name(..., extra_env=extra_env, **agent_kwargs)
            # so supplying it through kwargs is a guaranteed
            # `TypeError: got multiple values for keyword argument 'extra_env'`, which
            # killed the whole job 8 seconds in on 2026-08-19. `--agent-env` populates
            # `config.env`, which is the same destination by the supported route.
            cmd.extend(["--agent-env", f"{key}={value}"])
            # The verifier is a separate exec with its own env; without this the SCORING
            # run still thrashes and a timeout there fails a correct solution.
            cmd.extend(["--verifier-env", f"{key}={value}"])

        # The agent-timeout multiplier: the canonical 1.0 unless GBENCH_TB_TIMEOUT_MULTIPLIER
        # sets one (an opt-in deviation for a slow endpoint; on 2026-08-18 every task of a
        # 3-task --thinking run timed out after 10-25 minutes). Only the AGENT's clock is scaled;
        # the verifier timeout is a correctness bound and never moves.
        tmult_arg = _timeout_multiplier(enable_thinking)
        cmd.extend(["--agent-timeout-multiplier", tmult_arg])

        # Trials per task (Harbor -k / --n-attempts): the leaderboard's count (trial_count), and
        # the leaderboard's retries of a trial that raised (TRIAL_MAX_RETRIES).
        cmd.extend(["--n-attempts", str(n_trials)])
        cmd.extend(["--max-retries", str(TRIAL_MAX_RETRIES)])
        if include_names is not None:
            # Name the selected tasks exactly; `--include-task-name` fnmatches each `org/name`.
            # `--n-tasks` is never passed: it would take Harbor's first N in registry order.
            for task_name in include_names:
                cmd.extend(["--include-task-name", task_name])

        import threading

        full_output_lines = []
        total_expected = (shard_size if include_names is not None else FULL_TASK_COUNT) * n_trials
        seen_trials = set()
        correct_trials = 0

        # Harbor's CLI runs in the harness image the prerequisite check verified
        # (harbor_command), with the salvage coercion installed inside the harbor process:
        # Harbor's XML-salvage path returns a bare str where the caller expects an LLMResponse,
        # killing the trial with AttributeError (reachable only with GBENCH_TB_PARSER=xml and a
        # turn cap; with the canonical json parser the branch never runs).
        container = f"gbench-terminal-bench-{uuid.uuid4().hex[:12]}"
        _prepare_harbor_cache()
        cmd = harbor_command(cmd[1:], jobs_dir=tmp_dir, name=container,
                             env_names=container_env_names(env))
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Consume any direct stdout from Harbor asynchronously
            def _read_stdout():
                if proc.stdout:
                    line_iter = iter(proc.stdout.readline, "") if hasattr(proc.stdout, "readline") else iter(proc.stdout or [])
                    for line in line_iter:
                        full_output_lines.append(line)

            stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
            stdout_thread.start()

            with tqdm(total=total_expected, desc="Eval [TERMINAL_BENCH]") as pbar:
                # Monitor trial progress via result.json files on disk
                while proc.poll() is None:
                    time.sleep(2)
                    for root, _, files in os.walk(tmp_dir):
                        if "result.json" in files:
                            rf_path = os.path.join(root, "result.json")
                            try:
                                with open(rf_path, "r", encoding="utf-8") as rf:
                                    rdata = json.load(rf)
                                if isinstance(rdata, dict):
                                    if "n_total_trials" in rdata and rdata["n_total_trials"] and pbar.total != rdata["n_total_trials"]:
                                        pbar.total = rdata["n_total_trials"]
                                        pbar.refresh()
                                    if "verifier_result" in rdata or "trial_name" in rdata:
                                        trial_id = rdata.get("trial_name", os.path.basename(root))
                                        if trial_id not in seen_trials:
                                            seen_trials.add(trial_id)
                                            r_val = _extract_reward(rdata)
                                            if r_val is not None and r_val >= 1.0:
                                                correct_trials += 1
                                            pbar.update(1)
                                            acc = (correct_trials / len(seen_trials) * 100.0) if seen_trials else 0.0
                                            pbar.set_postfix(correct=f"{correct_trials}/{len(seen_trials)} ({acc:.1f}%)")
                            except Exception:
                                pass

                proc.wait()
                stdout_thread.join(timeout=1.0)

                # Final sweep to catch any remaining results on completion
                for root, _, files in os.walk(tmp_dir):
                    if "result.json" in files:
                        rf_path = os.path.join(root, "result.json")
                        try:
                            with open(rf_path, "r", encoding="utf-8") as rf:
                                rdata = json.load(rf)
                            if isinstance(rdata, dict) and ("verifier_result" in rdata or "trial_name" in rdata):
                                trial_id = rdata.get("trial_name", os.path.basename(root))
                                if trial_id not in seen_trials:
                                    seen_trials.add(trial_id)
                                    r_val = _extract_reward(rdata)
                                    if r_val is not None and r_val >= 1.0:
                                        correct_trials += 1
                                    pbar.update(1)
                                    acc = (correct_trials / len(seen_trials) * 100.0) if seen_trials else 0.0
                                    pbar.set_postfix(correct=f"{correct_trials}/{len(seen_trials)} ({acc:.1f}%)")
                        except Exception:
                            pass

            full_output = "".join(full_output_lines)
            if proc.returncode != 0:
                logger.error(f"Harbor CLI exited with code {proc.returncode}")
        except Exception as e:
            # A crashed external harness produced no trustworthy number: hard-error rather than
            # record a fabricated 0.0 row (no-skip/no-partial policy). The _jobs_dir context keeps
            # Harbor's logs for diagnosis (failed=True) on the way out.
            logger.error(f"Failed to execute Harbor CLI: {e}")
            raise infra_required(
                "terminal_bench",
                f"the Harbor harness failed to execute ({type(e).__name__}: {e}) - a harness/infra "
                "failure, not a model result",
                DOCS_URL) from e
        finally:
            # The container outlives a docker client that was interrupted or killed (still
            # running, or ended by a signal): stop it so Harbor can stop its task containers.
            rc = getattr(proc, "returncode", None) if proc is not None else None
            if proc is not None and (proc.poll() is None or (isinstance(rc, int) and rc < 0)):
                _stop_container(container, proc)

        # Specifically scan for Harbor result.json files (ignoring config.json, lock.json).
        # The sibling `agent/trajectory.json` IS read - see `_trajectory_text`.
        trial_results = []
        parse_failures: List[str] = []
        job_stats = {}

        for root, _, files in os.walk(tmp_dir):
            for file in files:
                if file == "result.json":
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath, "r", encoding="utf-8") as rf:
                            data = json.load(rf)
                        if isinstance(data, dict):
                            if "stats" in data and isinstance(data["stats"], dict):
                                job_stats = data["stats"]
                            elif "verifier_result" in data or "trial_name" in data:
                                reward_val = _extract_reward(data)
                                is_passed = reward_val is not None and reward_val >= 1.0
                                trial_name = data.get("trial_name", os.path.basename(root))
                                exc = data.get("exception_info", {}).get("exception_type") if data.get("exception_info") else None
                                # `AgentTimeoutError` on its own does not say WHY the agent
                                # ran out of time - a hard task and a model looping on the
                                # same command look identical in the result. Harbor's
                                # jobs-dir is a TemporaryDirectory that is deleted on exit,
                                # so unless the evidence is lifted out here it is gone:
                                # after the 2026-08-18 run there was no way to tell whether
                                # the two timed-out trials were stuck in a loop.
                                agent = data.get("agent_result") if isinstance(
                                    data.get("agent_result"), dict) else {}
                                transcript, traj_steps = _trajectory_text(root)
                                if not transcript:
                                    transcript = _agent_transcript(agent, data)
                                meta = agent.get("metadata") if isinstance(
                                    agent.get("metadata"), dict) else {}
                                # `n_episodes` and `api_request_times_msec` separate the two
                                # explanations a timeout cannot: few slow turns (the model
                                # is generating too much per turn) vs many fast ones (the
                                # agent is thrashing). Terminus records both on the context
                                # in a `finally`, so they survive the timeout.
                                times = meta.get("api_request_times_msec") or []
                                times = [t for t in times if isinstance(t, (int, float))]
                                # An exception outside the model's control (environment,
                                # docker, agent setup, the endpoint, ...) is a harness error;
                                # otherwise a verifier that could not resolve its package hosts
                                # before its tests ran is a DNS candidate (attribute_dns_failures).
                                exc_why, exc_note = trial_exception_verdict(data, reward_val)
                                task_id = _task_id_record(data.get("task_id"))
                                trial_results.append({
                                    "trial_name": trial_name,
                                    # Harbor's canonical task identity (TrialResult)
                                    "task_name": _task_name_of(data, trial_name),
                                    "task_id": task_id,
                                    "task_checksum": data.get("task_checksum"),
                                    **_task_metadata(task_id),
                                    "reward": reward_val,  # None when the trial recorded no reward
                                    "passed": is_passed,
                                    "exception": exc,
                                    **({"harness_error": "exception",
                                        "harness_error_reason": exc_why} if exc_why
                                       else trial_dns_harness_error(root, is_passed)),
                                    **({"exception_note": exc_note} if exc_note else {}),
                                    **_truncation_livelock(root),
                                    "agent_turns": meta.get("n_episodes") or traj_steps,
                                    "summarization_count": meta.get("summarization_count"),
                                    "output_tokens": agent.get("n_output_tokens"),
                                    "mean_turn_latency_s": (
                                        round(sum(times) / len(times) / 1000.0, 1)
                                        if times else None),
                                    "max_turn_latency_s": (
                                        round(max(times) / 1000.0, 1) if times else None),
                                    "response_text": transcript,
                                    "repetition_run": repetition_run(transcript),
                                    "repetition_onset": repetition_onset(transcript),
                                })
                    except Exception as e:
                        # WARNING, not debug: on 2026-08-18 Harbor completed 3/3 trials in
                        # 2h40 and gbench published `total_questions: 0` because every
                        # per-trial parse failed silently here. A swallowed exception on
                        # the only path that turns work into a score is not a debug detail.
                        parse_failures.append(f"{fpath}: {type(e).__name__}: {e}")
                        logger.warning("Could not parse %s: %s: %s",
                                       fpath, type(e).__name__, e)

        # Keep the jobs dir when any TRIAL errored too, not just when parsing did. On
        # 2026-08-19 two trials died with a bare `AttributeError` after 3 turns; parsing
        # succeeded, so the dir was cleaned up and the traceback went with it. A trial
        # that crashed is exactly when the logs are wanted.
        trial_errors = [t for t in trial_results if t.get("exception")]
        # DNS candidates become harness errors only when corroborated (another candidate, or a
        # fresh probe after the run); the rest stay the model's outcome with a note.
        attribute_dns_failures(trial_results, include_names)
        # Harness errors (the module docstring): an exception outside the model's control, or a
        # corroborated DNS failure in the verifier's package installs. Never the model's
        # outcome: left out of the score, counted and named.
        harness_trials = sorted((t for t in trial_results if t.get("harness_error")),
                                key=lambda t: str(t.get("trial_name")))
        exc_trials = [t for t in harness_trials if t["harness_error"] == "exception"]
        dns_trials = [t for t in harness_trials if t["harness_error"] == "dns"]
        dns_hosts = sorted({h for t in dns_trials for h in t.get("dns_failure_hosts") or ()})
        if trial_errors:
            logger.warning(
                "[terminal_bench] %d/%d trials raised: %s. %d of them failed outside the model's "
                "control (harness errors, left out of the score); the others are the model's "
                "outcome (its budget or context ran out, or its verifier ran without a reward).",
                len(trial_errors), len(trial_results),
                ", ".join(f"{t['trial_name']}={t['exception']}" for t in trial_errors),
                len(exc_trials))
        if dns_trials:
            logger.warning(
                "[terminal_bench] %d/%d trials lost to DNS failures in the verifier's package "
                "installs (%s): harness errors, left out of the score.", len(dns_trials),
                len(trial_results), ", ".join(dns_hosts))
        _JOBS_DIR_OK[tmp_dir] = (bool(trial_results) and not parse_failures
                                 and not trial_errors and not harness_trials)
        if trial_results:
            # by task, then trial: attempt_idx, sample_key, is_correct, scoring_excluded
            sample_traces = annotate_trials(trial_results)
            total = len(trial_results) - len(harness_trials)
            correct = sum(1 for t in trial_results if t.get("passed"))
        elif job_stats:
            total, correct = _job_stats_totals(job_stats)

        # `measured_total` is what Harbor actually produced; it drives the "nothing was measured"
        # hard-error below. A trial whose result.json could not be read (parse_failures) and each
        # expected trial Harbor wrote no result for (the selected tasks, or in a full run the
        # pinned dataset's FULL_TASK_COUNT tasks, x k, less the trials found) are harness errors
        # too: counted and named, never scored as the model's 0 (so neither is in `total`).
        measured_total = len(trial_results) if trial_results else total
        expected_tasks = shard_size if include_names is not None else FULL_TASK_COUNT
        missing_tasks = 0
        if trial_results:
            missing_tasks = max(0, expected_tasks * n_trials - len(trial_results)
                                - len(parse_failures))
        harbor_rc = proc.returncode
        noted = [t for t in trial_results if t.get("exception_note")]
        if noted:
            logger.warning(
                "[terminal_bench] %d trial(s) raised an exception while the agent ran that is not "
                "a known harness failure (%s): they stay the model's outcome (no reward, 0).",
                len(noted), ", ".join(f"{t['trial_name']}={t['exception']}" for t in noted))
        if parse_failures or missing_tasks:
            logger.warning(
                "[terminal_bench] %d trial result(s) could not be read and %d of the %d "
                "expected trial(s) have no result: harness errors, left out of the score (see "
                "%s).", len(parse_failures), missing_tasks, expected_tasks * n_trials, DOCS_URL)
        accuracy = (correct / total * 100.0) if total > 0 else 0.0
        duration = time.time() - start_time
        # A timed-out trial says nothing about WHY on its own, and the turn cap below was
        # sized on one particular deployment. Derive what this endpoint actually did -
        # tokens the agent generated over the time it spent waiting on the model - and turn
        # it into the number that decides whether the run was ever winnable: how many turns
        # fit in the shortest agent budget. On slower hardware that number collapses, and a
        # collapsed turn budget is the explanation a bare `AgentTimeoutError` withholds.
        stuck = [t for t in trial_results if t.get("livelocked")]
        if stuck:
            logger.warning(
                "[terminal_bench] %d/%d trials LIVELOCKED on the output cap: a turn hit "
                "max_tokens=%s, could not be salvaged, and was re-asked (worst run: %d "
                "consecutive truncations with no step recorded). These trials measure the "
                "opt-in cap, not the model. Raise or unset GBENCH_TB_TURN_MAX_TOKENS, or set "
                "GBENCH_TB_PARSER=xml - the canonical 'json' parser cannot salvage a truncated "
                "turn. Affected: %s",
                len(stuck), len(trial_results), turn_cap,
                max(t.get("longest_truncation_run") or 0 for t in stuck),
                ", ".join(t["trial_name"] for t in stuck))
        budget_report = _turn_budget_report(trial_results, turn_cap, enable_thinking)
        if budget_report and (budget_report.get("turns_in_shortest_budget") or 99) < 10:
            logger.warning(
                "[terminal_bench] this endpoint decodes at ~%.0f tok/s, so a %d-token turn "
                "costs ~%.0fs and only ~%d turns fit the shortest task's agent budget "
                "(%ds). Terminus needs more than that, so timeouts here are a SIZING "
                "result, not a model result (GBENCH_TB_TURN_MAX_TOKENS and "
                "GBENCH_TB_TIMEOUT_MULTIPLIER are opt-in deviations, named in "
                "leaderboard_comparable_reason).",
                budget_report["observed_tok_s"], turn_cap,
                budget_report["seconds_per_turn"],
                budget_report["turns_in_shortest_budget"],
                budget_report["shortest_agent_budget_s"])

        # Zero trials means nothing was measured -> hard-error (no-skip/no-partial), never a
        # fabricated 0.0 row. Previously this returned status:"failed" accuracy:0.0, which cli.py
        # still renders as a measured number. Keyed on `measured_total` (what Harbor produced), not
        # the shard-adjusted denominator, so a sharded run whose tasks all vanished still hard-errors.
        if measured_total == 0:
            tail = (full_output.strip() or
                    ("Harbor CLI produced no trials" if proc.returncode == 0
                     else f"Harbor CLI exited {proc.returncode}"))
            raise infra_required(
                "terminal_bench",
                f"the Harbor harness produced zero parseable trials so nothing was measured "
                f"(returncode={proc.returncode}). Tail: {tail[-800:]}",
                DOCS_URL)
        unread = (f"; {len(parse_failures)} more trial result(s) could not be read"
                  if parse_failures else "")
        if harness_trials and len(harness_trials) == measured_total:
            # Every trial is a harness error: the score is withheld, never reported as 0.
            if not exc_trials:
                raise infra_required(
                    "terminal_bench",
                    f"every one of the {measured_total} trial(s) lost its verifier to a DNS "
                    f"failure: the task containers could not resolve {', '.join(dns_hosts)} "
                    "while the verifier installed its test tools, so no trial was scored (a "
                    "harness error, not the model's outcome). Check name resolution on this "
                    "host's docker networks: on GCE the VPC DNS resolver is 169.254.169.254, and "
                    "the metadata server rule must spare port 53 (see "
                    "docs/evals/toolchains.md#cloud-vms-the-metadata-server)" + unread + ".",
                    DOCS_URL)
            kinds = sorted({t["exception"] for t in exc_trials if t.get("exception")})
            first = exc_trials[0].get("harness_error_reason") or ""
            raise infra_required(
                "terminal_bench",
                f"every one of the {measured_total} trial(s) failed outside the model's control, "
                f"so no trial was scored (harness errors, not the model's outcome): "
                f"{len(exc_trials)} raised {', '.join(kinds)} (e.g. {first[:300]})"
                + (f", and {len(dns_trials)} lost their verifier to a DNS failure "
                   f"({', '.join(dns_hosts)})" if dns_trials else "")
                + unread + ". Check Docker, the task images and that the model endpoint answers; "
                "Harbor's logs are kept in the jobs dir.", DOCS_URL)

        # leaderboard_comparable honesty. The defaults are the TB2.1 leaderboard's settings (its
        # trials per task included); a run is comparable ONLY when every opt-in knob is at its
        # canonical value AND the full task set ran AND no trials were lost to parse/harness
        # errors. A partial shard's reason is base.SHARD_REASON (added below).
        tmult = float(tmult_arg)
        # The settings that differ from the leaderboard's (the same on every shard of a run) ...
        settings: List[str] = []
        if limit:
            settings.append(f"subset run (--eval-limit {limit})")
        if official.get("temperature_matches_official") is False:
            origin = ("the no-think rule" if temperature_source == "default"
                      else temperature_source.split(':', 1)[-1])
            settings.append(f"agent temperature {temperature} set by {origin} (the TB2.1 "
                            "leaderboard runs terminus-2 at temperature 1)")
        if tmult != 1.0:
            settings.append(f"agent-timeout-multiplier={tmult} (GBENCH_TB_TIMEOUT_MULTIPLIER; the "
                            "leaderboard requires 1.0)")
        if turn_cap:
            settings.append(f"per-turn max_tokens cap={turn_cap} (GBENCH_TB_TURN_MAX_TOKENS; "
                            "Terminus-2 sends none)")
        if parser_name != CANONICAL_PARSER:
            settings.append(f"parser={parser_name!r} (GBENCH_TB_PARSER; the canonical terminus-2 "
                            "default is 'json')")
        if thread_env:
            settings.append(f"thread cap {thread_env.get('OMP_NUM_THREADS')} "
                            f"({', '.join(sorted(thread_env))}) injected into the agent's shell and "
                            "the verifier's environment (GBENCH_TB_CONTAINER_THREADS; the "
                            "leaderboard requires no resource overrides)")
        if n_trials < TERMINAL_BENCH_TRIALS:
            settings.append(f"{n_trials} trial(s) per task (the TB2.1 leaderboard requires at "
                            f"least {TERMINAL_BENCH_TRIALS})")
        # ... and the harness errors (merge_shards recomputes these over the shards' union).
        noncanon = settings + harness_error_reasons(
            parse=len(parse_failures), missing=missing_tasks, expected_tasks=expected_tasks,
            n_trials=n_trials, returncode=harbor_rc, exc=len(exc_trials), dns=len(dns_trials))

        result = {
            "benchmark_type": "eval",
            "eval_name": "terminal_bench",
            "model_name": model_name,
            "eval_limit_plan": workload,
            "thinking": enable_thinking,
            "total_questions": total,
            # the scored trials (harness errors are not in total_questions): Effective N
            "effective_n": total,
            "correct_answers": correct,
            "failed_requests": len(harness_trials) + len(parse_failures) + missing_tasks,
            # trials per task (Harbor -k) and avg@k / pass@k / pass^k over the tasks
            "attempts": attempts_report(sample_traces, attempts_requested, n_trials,
                                        trials_source),
            "accuracy": round(accuracy, 2),
            # per task category (task.toml [metadata] from Harbor's task cache), scored trials
            "category_accuracy": category_accuracy(sample_traces),
            "sample_traces": sample_traces,
            "turn_budget": budget_report,
            "temperature": temperature,
            "temperature_source": temperature_source,
            # Selection transparency: the shard identity (base.shard_identity, when a shard was
            # set), the tasks named to Harbor (None: the whole dataset), and `is_full`, True only
            # when the entire canonical task set was requested (no shard, no limit).
            # `total_questions` above counts the scored trials.
            **shard_identity(shard_spec),
            "selected_tasks": include_names,
            "is_full": not shard and not limit,
            "agent_parser": parser_name,
            "container_threads": int(thread_env.get("OMP_NUM_THREADS", 0)) or None,
            # The harness and dataset version the run executed (Harbor HARBOR_VERSION in the
            # harness image the prerequisite check verified; the dataset pinned to the
            # leaderboard's content hash).
            "harbor_version": HARBOR_VERSION,
            "harness_image": harness_image,
            # Harbor's PostHog usage events, off in every harness container (_FORCED_ENV).
            "harness_telemetry": dict(HARNESS_TELEMETRY),
            "dataset": DEFAULT_DATASET,
            "dataset_ref": DATASET_REF,
            "agent": agent_name,
            "agent_timeout_multiplier": tmult,
            "turn_max_tokens": turn_cap or None,
            # the request timeout and retries of the agent's LiteLLM calls (no 600 s default)
            "model_client_timeout": client_timeout,
            "official_sampling": official,
            "max_output_tokens_note": tokens_note,
            "salvage_patch": terminus_salvage_patch.enabled(),
            # A run that lost trials to parse/harness errors is a partial measurement, not a clean
            # score - surface it so it is never read as complete.
            "parse_failures": len(parse_failures),
            # Every trial that recorded a Harbor exception, the model's outcome ones included.
            "trial_errors": len(trial_errors),
            "trial_exceptions": dict(sorted(collections.Counter(
                str(t["exception"]) for t in trial_errors).items())),
            # Exceptions raised while the agent ran that are not a known harness failure: the
            # model's outcome (exception_note on the trial), counted by type.
            "exceptions_kept_as_model_outcome": dict(sorted(collections.Counter(
                str(t["exception"]) for t in noted).items())),
            # Harness errors (an exception outside the model's control, a corroborated DNS
            # failure in the verifier's package installs, an unreadable trial result, a selected
            # task without a trial): counted and named, never scored.
            "harness_errors": {k: v for k, v in (("exception", len(exc_trials)),
                                                 ("dns", len(dns_trials)),
                                                 ("parse", len(parse_failures)),
                                                 ("missing", missing_tasks)) if v},
            "harness_error_trials": sorted(t["trial_name"] for t in harness_trials),
            # the trials the run was to produce: the selected tasks (the pinned dataset's
            # FULL_TASK_COUNT in a full run) x k; `missing` counts those without a result
            "expected_trials": expected_tasks * n_trials,
            # Harbor's exit status: anything but 0 means its process failed (Harbor exits 0 after
            # a completed job whatever its trials' outcomes), named in the reasons below
            "harbor_returncode": harbor_rc,
            "partial": bool(parse_failures or harness_trials or missing_tasks),
            # the settings that differ from the leaderboard's, without the harness-error reasons
            "setting_deviations": settings,
            "leaderboard_comparable": not noncanon,
            "leaderboard_comparable_reason": "; ".join(noncanon) or "full canonical run",
            "status": ("completed_with_errors" if (harness_trials or parse_failures
                                                   or missing_tasks
                                                   or harbor_rc not in (0, None))
                       else "success"),
            # the harness errors, then (informational, status unchanged) the exceptions kept as
            # the model's outcome
            "status_reason": _with_note(harness_status_reason(
                exc=len(exc_trials), dns=len(dns_trials), parse=len(parse_failures),
                missing=missing_tasks, expected_trials=expected_tasks * n_trials,
                returncode=harbor_rc), model_outcome_exceptions_note(dict(collections.Counter(
                    str(t["exception"]) for t in noted)))),
            # Harbor's retries of a trial that raised (the leaderboard's retry.max_retries)
            "trial_max_retries": TRIAL_MAX_RETRIES,
            # the run's tasks in gbench's Harbor cache before the run, and the task images pulled
            # on this host with the user's registry login before Harbor started
            "dataset_cache": dataset_cache,
            "task_image_prepull": prepull,
            "duration_s": round(duration, 2),
            # the relay that added the endpoint key (None: the endpoint has no key)
            "model_endpoint_relay": relay.record() if relay is not None else None,
        }
        if shard_spec is not None and shard_spec.partial:
            if not noncanon:
                result["leaderboard_comparable_reason"] = ""
            add_comparability_reason(result, SHARD_REASON.format(
                label=shard_spec.label, index=shard_spec.index, count=shard_spec.count))
        return result


def model_outcome_exceptions_note(kept: Dict[str, int]) -> str:
    """The status_reason clause for trials that raised an exception while the agent ran that is
    not a known harness failure, scored as the model's outcome ("" when there is none)."""
    n = sum(int(v or 0) for v in (kept or {}).values())
    if not n:
        return ""
    return (f"{n} trial(s) raised an exception not known as a harness failure while the agent "
            f"ran and are scored as the model's outcome (exceptions_kept_as_model_outcome: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(kept.items())) + ")")


def _with_note(reason: Optional[str], note: str) -> Optional[str]:
    """`reason` with `note` appended ("; "), `note` alone when there is no reason."""
    if not note:
        return reason
    return f"{reason}; {note}" if reason else note


def _counter_sum(dicts: List[Any]) -> Dict[str, int]:
    """The per-key sum of the {name: count} dicts in `dicts` (non-dicts ignored), sorted."""
    total: collections.Counter = collections.Counter()
    for d in dicts:
        if isinstance(d, dict):
            total.update({str(k): int(v or 0) for k, v in d.items()})
    return dict(sorted(total.items()))


def _setting_reasons(result: Dict[str, Any]) -> List[str]:
    """A shard's setting deviations: `setting_deviations`, else (a result written before it was
    recorded) its comparability reasons without the shard, empty-shard and harness-error ones."""
    if isinstance(result.get("setting_deviations"), list):
        return [str(s) for s in result["setting_deviations"]]
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    out = []
    for reason in str(result.get("leaderboard_comparable_reason") or "").split("; "):
        if (not reason or reason == "full canonical run" or reason.startswith(shard_prefix)
                or reason.startswith("empty shard ") or reason.startswith("Harbor exited ")
                or reason.split(" ", 1)[0].isdigit()):
            continue
        out.append(reason)
    return out


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one terminal_bench run (gbench --merge).

    The trials of every shard (sample_traces, one per trial name) are re-counted as the run
    counts them: `accuracy` = passed / scored trials (harness errors left out), `attempts`
    (avg@k / pass@k / pass^k over the tasks) and `category_accuracy` over the union; the harness
    errors are summed per kind (`harness_errors`: exception and dns from the traces, parse and
    missing from each shard's counts; `harness_error_trials`, `failed_requests`, `trial_errors`,
    `trial_exceptions`, `exceptions_kept_as_model_outcome`, `expected_trials`); `selected_tasks`
    is the union of the shards' selections and `is_full` says whether it is the whole pinned
    dataset. The comparability reasons are the shards' setting deviations plus the union's
    harness errors (each shard's non-zero Harbor exit named with its shard); the shard and
    empty-shard reasons are dropped. Node-specific records are kept per shard
    (`duration_s_by_shard`, `turn_budget_by_shard`, `model_endpoint_relay_by_shard`)."""
    from ...merge import MergeError
    union: Dict[str, Dict[str, Any]] = {}
    for r in results:
        traces = r.get("sample_traces")
        if not isinstance(traces, list):
            raise MergeError("terminal_bench: a shard carries no trial traces (sample_traces)")
        for t in traces:
            if not isinstance(t, dict):
                continue
            key = str(t.get("trial_name") or f"{t.get('sample_key')}#{t.get('attempt_idx', 0)}")
            union.setdefault(key, dict(t))
    traces = annotate_trials(list(union.values()))
    ran = [r for r in results if not r.get("empty_shard")]
    first = ran[0] if ran else results[0]
    harness = [t for t in traces if t.get("harness_error")]
    exc = sum(1 for t in harness if t.get("harness_error") == "exception")
    dns = sum(1 for t in harness if t.get("harness_error") == "dns")
    parse = sum(int(r.get("parse_failures") or 0) for r in results)
    missing = sum(int((r.get("harness_errors") or {}).get("missing") or 0) for r in results)
    total = len(traces) - len(harness)
    correct = sum(1 for t in traces if t.get("passed") and not t.get("harness_error"))
    attempts = first.get("attempts") or {}
    k = int(attempts.get("attempts_per_sample") or TERMINAL_BENCH_TRIALS)
    selected = sorted({str(n) for r in results for n in (r.get("selected_tasks") or [])})
    expected = sum(int(r.get("expected_trials") or 0) for r in results) or len(selected) * k
    failed_exits = [(str(r.get("shard") or "?"), r.get("harbor_returncode")) for r in results
                    if r.get("harbor_returncode") not in (0, None)]

    settings: List[str] = []
    for r in ran:
        for s in _setting_reasons(r):
            if s not in settings:
                settings.append(s)
    limited = any(s.startswith("subset run") for s in settings)
    is_full = not limited and len(selected) >= FULL_TASK_COUNT
    kept = list(settings)
    if not is_full and not limited:
        kept.append(f"the merged shards hold {len(selected)} tasks, not the full "
                    f"{FULL_TASK_COUNT}")
    kept += harness_error_reasons(parse=parse, missing=missing, expected_tasks=len(selected),
                                  n_trials=k, returncode=None, exc=exc, dns=dns)
    kept += [f"shard {label}: {harbor_exit_reason(rc)}" for label, rc in failed_exits]

    merged = dict(first)
    for key in ("empty_shard", "full_set_items", "accuracy_withheld_reason"):
        merged.pop(key, None)
    status_reason = harness_status_reason(exc=exc, dns=dns, parse=parse, missing=missing,
                                          expected_trials=expected, returncode=None)
    if failed_exits:
        exits = "; ".join(f"Harbor exited with status {rc} on shard {label}"
                          for label, rc in failed_exits)
        status_reason = (status_reason + "; " + exits) if status_reason else (
            "harness errors, left out of the score: " + exits)
    kept_as_model = _counter_sum([r.get("exceptions_kept_as_model_outcome") for r in results])
    status_reason = _with_note(status_reason, model_outcome_exceptions_note(kept_as_model))
    # Each node builds its own harness image: one record per shard, and the common one only when
    # every shard's image carries the same Harbor release and lock commit.
    images = {str(r.get("shard") or "?"): r.get("harness_image") for r in ran}
    labels = {json.dumps({k: (img or {}).get(k) for k in ("harbor", "lock")}, sort_keys=True)
              for img in images.values()}
    common_image = (dict(first.get("harness_image") or {}, id=None)
                    if len(labels) == 1 and first.get("harness_image") else None)
    merged.update({
        "sample_traces": traces,
        "total_questions": total,
        "effective_n": total,
        "correct_answers": correct,
        "accuracy": round(100.0 * correct / total, 2) if total else None,
        "failed_requests": len(harness) + parse + missing,
        "attempts": attempts_report(traces, int(attempts.get("attempts_requested") or 1), k,
                                    str(attempts.get("source") or "")),
        "category_accuracy": category_accuracy(traces),
        "selected_tasks": selected,
        "is_full": is_full,
        "expected_trials": expected,
        "harbor_returncode": failed_exits[0][1] if failed_exits else 0,
        "parse_failures": parse,
        "trial_errors": sum(int(r.get("trial_errors") or 0) for r in results),
        "trial_exceptions": _counter_sum([r.get("trial_exceptions") for r in results]),
        "exceptions_kept_as_model_outcome": kept_as_model,
        "harness_image": common_image,
        "harness_image_by_shard": images,
        "harness_image_note": (None if common_image is not None else
                               "the shards' harness images carry different Harbor releases or "
                               "lock commits (harness_image_by_shard)"),
        "task_image_prepull_by_shard": {str(r.get("shard") or "?"): r.get("task_image_prepull")
                                        for r in ran},
        "dataset_cache_by_shard": {str(r.get("shard") or "?"): r.get("dataset_cache")
                                   for r in ran},
        "task_image_prepull": None,
        "dataset_cache": None,
        "harness_errors": {kind: n for kind, n in (("exception", exc), ("dns", dns),
                                                    ("parse", parse), ("missing", missing)) if n},
        "harness_error_trials": sorted(str(t.get("trial_name")) for t in harness),
        "partial": bool(harness or parse or missing),
        "setting_deviations": settings,
        "leaderboard_comparable": not kept,
        "leaderboard_comparable_reason": "; ".join(kept) or "full canonical run",
        "status": ("completed_with_errors" if (harness or parse or missing or failed_exits
                                               or any(r.get("status") == "completed_with_errors"
                                                      for r in results))
                   else "success"),
        "status_reason": status_reason,
        "duration_s": round(sum(float(r.get("duration_s") or 0.0) for r in results), 2),
        "duration_s_by_shard": {str(r.get("shard") or "?"): r.get("duration_s") for r in results},
        "turn_budget": None,
        "turn_budget_by_shard": {str(r.get("shard") or "?"): r.get("turn_budget") for r in ran},
        "model_endpoint_relay": None,
        "model_endpoint_relay_by_shard": {str(r.get("shard") or "?"): r.get("model_endpoint_relay")
                                          for r in ran},
    })
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = "no trial of any shard was scored"
    return merged
