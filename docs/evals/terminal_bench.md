# TerminalBench 2.1 (TB2) Evaluation Setup Guide

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.


## Overview
`terminal_bench` evaluates autonomous agent execution across interactive Linux terminal troubleshooting tasks, Bash scripts, and CLI workflows.

## Prerequisites
Terminal-Bench executes candidate commands inside isolated Docker sandbox containers managed by the **Harbor** agent evaluation framework.

---

### 1. System Requirements (Docker Engine)
`pip install docker` only installs the Python SDK. The actual **Docker Engine daemon** must be installed on your Linux host:

```bash
# 1. Check if Docker is already installed
docker --version && docker info

# 2. If Docker is NOT installed, install Docker Engine:
# Option A: Standard Ubuntu/Debian package (zero config)
sudo apt-get update && sudo apt-get install -y docker.io
# Option B: Upstream Docker CE (if preferred)
# Follow https://docs.docker.com/engine/install/

# 3. Ensure Docker service is running
sudo systemctl enable --now docker

# 4. Grant non-root user permissions to access /var/run/docker.sock
sudo groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker
```

---

### 2. The harness image: Harbor 0.20.0, built locally
gbench runs Harbor from a local image, `gbench-terminal-bench:harbor-0.20.0`
([`gbench/docker/terminal_bench.Dockerfile`](../../gbench/docker/terminal_bench.Dockerfile)),
never from its own Python environment and never from a `harbor` executable on `PATH`. The image
holds Harbor at the TB2.1 leaderboard's release (terminal-bench-2-1 `leaderboard/uv.lock` at
commit `7131e43`: harbor 0.20.0) with every package of Harbor's dependency closure at the version
that lock resolves
([`terminal_bench.constraints.txt`](../../gbench/docker/terminal_bench.constraints.txt); the build
fails if anything else ends up installed), plus the Docker CLI 29.1.3 with its compose 2.40.3 and
buildx 0.37.1 plugins (each checked against its sha256), on `python:3.12-slim` at a fixed digest
(Harbor 0.20.0 needs Python 3.12 or later). Build it once, from the repository root (gbench never
pulls it):

```bash
docker build -t gbench-terminal-bench:harbor-0.20.0 -f gbench/docker/terminal_bench.Dockerfile gbench/docker
docker run --rm gbench-terminal-bench:harbor-0.20.0 harbor --version   # 0.20.0
```

Before a run, gbench checks that the image is on the host and that its labels name Harbor 0.20.0
and that lock (`org.gbench.harbor`, `org.gbench.terminal_bench_lock`); a missing image, or one built
from anything else, **hard-errors** (`infra_required`) with the build command. The result records
`harbor_version`, `harness_image` (the tag, the image id, the Harbor release and the lock
commit) and `harness_telemetry` (Harbor's telemetry off, step 3).

How the harness container runs (`terminal_bench.harbor_command`): Harbor's local docker backend
starts each task's containers on this host's docker daemon through the mounted docker socket
(docker-out-of-docker). The socket is the one the docker CLI talks to, the daemon gbench checks
and probes: `DOCKER_HOST` when set, else the current docker context's endpoint (`docker context
inspect`), else `/var/run/docker.sock`; it must exist, and it is mounted at `/var/run/docker.sock`
in the container (with its group). A `tcp://`, `ssh://` or other non-unix endpoint is
`infra_required`: Harbor in the container can only drive a mounted unix socket. gbench mounts Harbor's jobs
directory at the same path inside and outside the container, because Harbor bind-mounts each
trial's directories into its task container. The container runs as your user (with the socket's
group), on the host network, with every capability dropped and `no-new-privileges`. It gets no
credential: its environment is only `OPENAI_API_KEY=EMPTY`, `OPENAI_BASE_URL`, `HARBOR_TELEMETRY=0`
(step 3) and the other `HARBOR_*` and proxy variables of gbench's credential-free environment that
are set, and nothing of this host is mounted besides the docker socket, the jobs directory, gbench's Harbor cache and the salvage
patch (read-only). The model's commands never run in it: Terminus-2 types them into tmux in the
task containers, which the metadata check in step 4 covers.

---

### 3. The Terminal-Bench 2.1 (TB2.1) dataset
gbench runs the dataset pinned to the TB2.1 leaderboard's content hash (`DATASET_REF` in
terminal-bench-2-1 `leaderboard/src/leaderboard/core/hub.py`, the `ref` of every leaderboard
config), so a registry push cannot change the tasks a run executes: `harbor run -d
terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`
(89 tasks). The result records `dataset` and `dataset_ref`. Harbor keeps the tasks it downloads in
gbench's Harbor cache, `$GBENCH_CACHE_DIR/harbor` (default `~/.cache/gbench/harbor`), which gbench
mounts as the harness container's `~/.cache/harbor` (a `~/.cache/harbor` of a Harbor installed on
the host is not used). Download the pinned version there with the image, from the
[Harbor Registry](https://hub.harborframework.com/datasets):

```bash
CACHE="${GBENCH_CACHE_DIR:-$HOME/.cache/gbench}/harbor"
mkdir -p "$CACHE"
docker run --rm --network host --user "$(id -u):$(id -g)" \
    -v "$CACHE:/harbor-home/.cache/harbor" -e HARBOR_TELEMETRY=0 gbench-terminal-bench:harbor-0.20.0 \
    harbor dataset download terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a --cache
```

(`--cache` writes Harbor's content-addressed cache layout, the one `harbor run` reads; without it
Harbor 0.20.0 exports the tasks into the working directory.) gbench also reads the task images from
this cache (`$CACHE/tasks/packages/terminal-bench/<task>/<content hash>/task.toml`,
`[environment] docker_image`) for the check in step 4 and for the pre-pull. gbench runs that
download itself before a run whose tasks the cache lacks (`dataset_cache` in the result: the tasks
cached before, the download's exit and the tasks cached after; a failed download is logged, and
Harbor downloads the tasks itself). It then pulls the run's task images that are not on the host
yet with the host's docker CLI, four at a time (`task_image_prepull`: `present`, `pulled`,
`failed`): your `docker login` applies there, while Harbor pulls in the harness container, which
has no docker client configuration, anonymously and meets Docker Hub's unauthenticated rate
limit. Harbor starts each task with `docker compose up` (pull policy `missing`), so a pre-pulled
image is not pulled again; an image that could not be pulled is left to Harbor, and a task whose
environment then cannot start is a harness error (below). The check in step 4 runs before both
and probes with a task image already on the host, else with the harness image (the network
verdict does not depend on the image); nothing is pulled for it.

With `--shard` or `--eval-limit`, gbench lists the pinned dataset's tasks with Harbor's own
resolver in the same image (`terminal_bench.TASK_NAMES_SCRIPT`, without the docker socket).

Harbor 0.20.0 sends usage telemetry to PostHog by default (its `harbor/telemetry.py`: the command,
the agent and model names, the dataset refs, the job's mean reward, token counts and cost). gbench
turns it off: every harness container it starts gets `HARBOR_TELEMETRY=0` (Harbor's own opt-out,
`_telemetry_disabled`), whatever your shell sets `HARBOR_TELEMETRY` to, and the image sets the same
value, so a `docker run` of the image by hand sends nothing either (the download command above
passes it too). It changes nothing Harbor runs or scores. The result records it as `harness_telemetry`
(`{"harbor_telemetry": "disabled", "env": "HARBOR_TELEMETRY=0"}`).

---

### 4. On a cloud VM: block the metadata server for docker containers
The agent's commands and the verifier's tests run in each task's container, on the compose network
Harbor creates for it (a user-defined bridge network), with network access, and a docker container
reaches the VM's metadata server at `169.254.169.254`, which hands out the VM's service-account
token. Before Harbor starts, gbench checks that it is unreachable from docker's default bridge and
from a user-defined bridge network (gbench creates a temporary one, `gbench-metadata-probe-<id>`,
and removes it right after): a short-lived probe container (`--pull never`) opens and closes a TCP
connection to `169.254.169.254:80` and sends nothing. The probe runs in a task image from the local
task cache that is already on the host (the images of a `--shard` run's own tasks first); nothing
is pulled for it. When none is present it **hard-errors** (`infra_required`) naming the `docker
pull` of a task image (step 3); with no cached task at all it asks for `harbor dataset download`
first. Each task's verifier installs its test tools in the task container at run time (`apt-get`
from the Ubuntu or Debian archive, uv from `astral.sh`, `uvx` fetching Python from GitHub and the
test packages from PyPI), so the same probe container also resolves `archive.ubuntu.com`,
`deb.debian.org`, `astral.sh`, `github.com`, `pypi.org` and `files.pythonhosted.org`
(`swebench_common.TASK_INSTALL_HOSTS`); a name that does not resolve **hard-errors** too, naming
the DNS-preserving rule below. A connection that opens, or a probe without a clear verdict,
**hard-errors** (`infra_required`) naming the fix. Block it once on the host; gbench never runs
sudo or changes the host's firewall itself:

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server).

---

## Running the Evaluation in gbench

### Full Benchmark Run (All Tasks)
```bash
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --max-output-tokens 65536 \
       --sandboxes 32
```

Harbor runs each task **5 times** (`-k 5`, the TB2.1 leaderboard's minimum: leaderboard/SUBMIT.md
asks for at least 5 trials per task), so a full run is 89 x 5 = 445 trials. `--attempt-count k`
above 5 runs k trials per task; a smaller value keeps the 5. The headline `accuracy` is the mean
over every scored trial (Harbor's `mean` metric); `attempts` records `attempts_requested`,
`attempts_per_sample`, its `source`, and avg@k / pass@k / pass^k over the tasks.

### Fast Smoke Test (`--eval-limit`)
Run a quick single-task or small-sample smoke test to verify setup. `--eval-limit N` is N tasks in
total, each run its trials (Harbor `-k`, 5 by default: repeats of one task, not extra samples);
the run logs its planned workload on one line at its start (e.g. `[terminal_bench] --eval-limit 5:
5 sample(s) x 5 attempts = 25 trials`) and records it as `eval_limit_plan`.

```bash
# Smoke test on 1 task
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --max-output-tokens 65536 \
       --eval-limit 1

# Quick test on 5 tasks with 16 parallel sandbox containers
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --max-output-tokens 65536 \
       --eval-limit 5 \
       --sandboxes 16
```

`--max-output-tokens` is sent on every agent turn. Terminus-2 sends no `max_tokens`, so gbench's
relay puts the run's `--max-output-tokens` on each request that carries none
(`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns this off, and then none is sent, as upstream). The
result records `agent_max_tokens_from_run`, `max_tokens_sent`, `max_output_tokens_note` and
`model_endpoint_relay.max_tokens_from_run`, and names the cap in `leaderboard_comparable_reason`.

### Task selection (`--eval-limit`, `--shard`)
With `--eval-limit` or `--shard I/N`, gbench enumerates the pinned dataset's task names through
Harbor's own resolver, sorts them, takes the shard first (task I, I+N, ... of the sorted list),
then draws `--eval-limit` tasks from it with a fixed seed (`sampling.stratified_sample`, seed
`terminal_bench`), and passes each as `--include-task-name`; `--n-tasks` (Harbor's first N in
registry order) is never used, so `--eval-limit 10` and `--shard 1/1 --eval-limit 10` run the same
tasks. The result records `selected_tasks` and the shard identity (`shard`, `shard_index`,
`shard_count`, `shard_applied`); a partial shard is not comparable (`sharded run (--shard I/N) ...`).
An empty shard runs nothing and reports `accuracy: null` with `accuracy_withheld_reason`. A malformed
`GBENCH_SHARD` is an error. Each trial record carries Harbor's task identity (`task_name`,
`task_id` as JSON, `task_checksum`), `sample_key` (the task), `attempt_idx` (its trial number),
`is_correct`, and `category` / `difficulty` from the task's `task.toml` `[metadata]` when Harbor's
task cache holds it (filling `category_accuracy`); traces are sorted by task, then trial.

`gbench --merge` recombines the shards through `terminal_bench.merge_shards`: the union of the
shards' trials re-counted as a run counts them (`accuracy` = passed / scored trials, `attempts`
and `category_accuracy` over the union), the harness errors summed per kind (`harness_errors`,
`harness_error_trials`, `failed_requests`, `expected_trials`, `trial_errors`, `trial_exceptions`),
`selected_tasks` the union of the shards' selections (`is_full` when it is all 89 tasks), and the
comparability reasons the shards' setting deviations (`setting_deviations`) plus the union's harness
errors (a shard whose Harbor exited non-zero is named). Node-specific records stay per shard
(`duration_s_by_shard`, `turn_budget_by_shard`, `model_endpoint_relay_by_shard`,
`harness_image_by_shard`, `task_image_prepull_by_shard`, `dataset_cache_by_shard`): each node
builds its own harness image, so `harness_image` is the common record (without an image id) only
when every shard's image carries the same Harbor release and lock commit, else null with
`harness_image_note`.

### Model endpoint key
Terminus-2 runs in Harbor's process in the harness container (step 2), on the host network; the
model's commands run in the task containers, whose compose environment Harbor builds from its own.
Harbor gets no credential (only the variables named in step 2) and `OPENAI_API_KEY=EMPTY`.
Terminus-2 always reaches the endpoint through gbench's relay on `127.0.0.1`
(`swebench_common.EndpointRelay`; reachable on the host network, not from the task containers'
bridge networks), with or without a key (since 2026-10-11). The relay adds the key only when there
is one (`VLLM_API_KEY`, else `OPENAI_API_KEY`). It also puts the run's `--max-output-tokens` on
requests that carry no `max_tokens`, applies the smoke stop and checks request shape. The result
records it under `model_endpoint_relay` (never the key). An endpoint on this host is reached at the
address you give (`127.0.0.1` works: the harness container shares the host network). The relay
waits for one request as long as the agent's own request timeout plus 120 s (below), and at least
6 hours.

### Request timeout of the agent's model calls
Harbor's LiteLLM wrapper calls `litellm.acompletion` without a timeout, and LiteLLM 1.91.0 (the
harness image's) then gives each request 600 s (`COMPLETION_HTTP_FALLBACK_SECONDS`), with the OpenAI
SDK's 2 retries; Harbor retries a failed call 3 times itself. Terminus-2 sends no `max_tokens`, so
a turn that needed more than 600 s of decode was dropped and asked again: a hidden cap on the turn's
length (found by the 2026-10-10 investigation). gbench now passes `timeout` and `max_retries` in
Terminus-2's `llm_call_kwargs`, which Harbor's `LiteLLM.call` merges into its completion kwargs:
`timeout` is `base.request_timeout_s` of the served context window (33,968 s for a 262,144-token
window), or of `GBENCH_TB_TURN_MAX_TOKENS` when that opt-in cap is set, and `max_retries` is 0, so
each of Harbor's 3 attempts is one HTTP attempt. A check in the harness image against a slow local
server (2026-10-11) showed a 2 s timeout ending each attempt at 2 s with 3 requests in all (9 with
the SDK's default retries), a 9,392 s timeout getting the reply, and LiteLLM's default resolving to
600 s. The task's own agent timeout still bounds each trial. The result records
`model_client_timeout` (`{client, version, timeout_s, retries, outer_attempts, source, replaces,
hidden_cap_removed}`); results from before 2026-10-11 are not comparable on long turns.

### Concurrency Separation (`--sandboxes` vs `--batch-sizes`)
When running mixed evaluation campaigns (e.g. `--evals all` or combining academic QA evals with agent benchmarks):
- `--batch-sizes 128` controls concurrent HTTP API requests for high-throughput academic benchmarks (`mmlu_pro`, `aime`, `gsm8k`).
- `--sandboxes 32` or `--sandboxes 64` explicitly caps the number of simultaneous Docker containers spawned on the host system to prevent container daemon exhaustion while maintaining maximum throughput.

## Behavior when prerequisites are missing

This suite **hard-errors** (never skips, never reports a fabricated 0%): if Docker or its socket is
unavailable, if the harness image with Harbor 0.20.0 is not built (step 2), if the metadata server
check (step 4) fails, if the task list of a `--shard` or `--eval-limit` run cannot be enumerated, or
if the Harbor harness crashes or completes with **zero parseable trials**,
the run raises `infra_required` and is recorded as a `status:"error"` row (the sweep continues).
Provision the prerequisites and re-run. A run that lost some trials to harness errors (below) is
marked `partial: true` (with `parse_failures` and `harness_errors` counts) and is not comparable.

**Harness errors are left out of the score.** A trial that failed outside the model's control is
recorded with `harness_error` (`"exception"` or `"dns"`, below), left out of `total_questions` and
`accuracy`, counted in `harness_errors` (`{"exception": N, "dns": M}`, the trials in
`harness_error_trials`, also `failed_requests`), and the run is `completed_with_errors`,
`partial: true` and not comparable. A trial whose `result.json` cannot be read (`"parse": N`) and
each expected trial Harbor wrote no result for (`"missing": N`: the selected tasks, or in a run
without a shard or `--eval-limit` all 89 tasks of the pinned dataset, x 5, less the trials found;
`expected_trials` records the product) are harness errors too: counted in `harness_errors` and
`failed_requests`, never scored as a 0. So a full run that Harbor ended early (killed, say, after
15 of 445 trials) is `completed_with_errors`, `partial: true` and not comparable, never a clean
score over the trials it did write. Harbor 0.20.0's `harbor run` exits 0 once its job completed,
whatever its trials' outcomes, so a non-zero exit (`harbor_returncode`) means its process failed or
was killed: the run is `completed_with_errors` and not comparable (`Harbor exited with status N`)
even when every trial has a result. `status_reason` names every harness error. A run whose every
trial is a harness error hard-errors (`infra_required`) instead of reporting 0. `trial_errors`
counts every trial that recorded a Harbor exception and `trial_exceptions` counts them by type,
the model's outcome ones included.

- **Exceptions outside the model's control** (`harness_error: "exception"`, with
  `harness_error_reason`): a trial without a reward whose Harbor exception is a known harness
  failure, read from the phase its `result.json` records (Harbor sets `agent_execution.started_at`
  when the agent starts and `verifier.started_at` when the verifier does) and the exception's
  type:
  - any exception before the agent started: an environment or docker start failure
    (`EnvironmentStartTimeoutError`, `HealthcheckError`, `SandboxBuildFailedError`, a compose
    error), an agent setup failure (`AgentSetupTimeoutError`);
  - a trial Harbor cancelled (`CancelledError`);
  - a model endpoint failure: LiteLLM's `APIConnectionError`, `InternalServerError`,
    `ServiceUnavailableError`, `BadGatewayError`, `Timeout`, `RateLimitError`, `BadRequestError`,
    `APIError`, ... as Terminus-2 re-raises them after its retries (a context-length refusal
    becomes `ContextLengthExceededError` first), and an installed agent's `ApiRateLimitError`;
  - an exception whose message names an unreachable docker daemon (`Cannot connect to the Docker
    daemon`, `error during connect`);
  - a verifier that started and then failed outside the four verifier exceptions below, such as
    a tests upload or output download error (`AddTestsDirError`, `DownloadVerifierDirError`),
    unless docker's own message shows the task container had stopped or its target path was
    unusable (`is not running`, `no container found`, `not a directory`, `read-only file
    system`, ...), which the model's commands can cause: then the trial stays the model's outcome,
    with `exception_note`.

  The model's outcome: a trial with a reward (the verifier finished); `AgentTimeoutError` (Harbor
  still runs the verifier) and a verifier that started and then timed out or left no usable reward
  (`VerifierTimeoutError`, `RewardFileNotFoundError`, `RewardFileEmptyError`,
  `VerifierOutputParseError`), the exceptions in Harbor's default `RetryConfig.exclude_exceptions`;
  and `NonZeroAgentExitCodeError`, `ContextLengthExceededError` and `OutputLengthExceededError`,
  gbench's own attribution (the model's process, context or output ran past a limit; Harbor's
  default retry policy would retry them). Any other exception raised while the agent ran is not a
  known harness failure, so the trial stays the model's outcome, as Harbor's metrics count every
  trial without a reward as 0: for example the `RuntimeError` tmux send-keys raises (`failed to
  send ... keys ... no server running`) after the model's own keystrokes (`exit`, `tmux
  kill-server`, `kill -9 -1`) ended its tmux session or container. Such a trial carries
  `exception_note` saying so, `exceptions_kept_as_model_outcome` counts them by type, and
  `status_reason` names them (informational: the status stays as the harness errors make it; a
  merged run names the union's).
- **DNS failures during the run.** A trial that did not pass, whose verifier output
  (`verifier/test-stdout.txt` in Harbor's trial directory) shows a package host under
  `ubuntu.com`, `debian.org`, `astral.sh`, `github.com`, `githubusercontent.com`, `pypi.org`,
  `pythonhosted.org`, `python.org`, `pytorch.org`, `nodejs.org`, `npmjs.org` or `npmjs.com`
  failing to resolve before its tests started (apt's `Temporary failure resolving`, curl's `Could
  not resolve host`, uv's `dns error`, urllib3's `Failed to resolve`, ... and no pytest `test
  session starts` header), scored nothing. It is a harness error (`harness_error: "dns"`, with
  `dns_failure_hosts` and `dns_failure_evidence`) when something outside its task container
  corroborates the failure: another such trial in the run, or a fresh probe container after the
  run (container_guard's probe, not its cache, in a cached task image, on docker's default bridge
  and a new user-defined network) that cannot resolve one of the hosts either. Otherwise the model
  may have broken name resolution in its own container, so the trial stays the model's outcome,
  with `dns_failure_note` saying why; the same when the agent's transcript (`agent/*.json`) names
  `/etc/resolv.conf`, `/etc/hosts` or `/etc/nsswitch.conf`. Once the tests have started, their
  verdict stands.

## Harbor version

The TB2.1 leaderboard runs Harbor **0.20.0** (`leaderboard/uv.lock`), and the result parsing here
follows its `result.json` schema (`n_completed_trials`, `evals[<name>].metrics[].mean`,
`verifier_result.rewards`) and its Terminus-2 source (the salvage patch). The prerequisite check
requires the harness image built with exactly that release and the lock's package versions (step 2)
and the result records `harbor_version` and `harness_image`. If gbench is interrupted, it stops the
harness container (`docker stop`, which gives Harbor 30 seconds to stop its task containers) and
removes it.

## Configuration knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `GBENCH_TERMINAL_BENCH_TEMPERATURE` | *(unset)* | replaces the agent's temperature (default: gbench's rule for every suite, 1.0 with `--thinking`, `temperature_source` `"thinking"`, and 0.0 without, `temperature_source` `"default"`; sent as Terminus-2's `temperature` agent kwarg). Precedes `--temperature`. A temperature other than the TB2.1 leaderboard's 1 (so every no-think run) is named in `leaderboard_comparable_reason`. |
| `GBENCH_TB_PARSER` | `json` | Terminus-2 action parser. `json` is Terminus-2's default (canonical). `xml` (opt-in, non-canonical) can salvage a turn that hit a `GBENCH_TB_TURN_MAX_TOKENS` cap, which `json` cannot (a truncated turn is re-asked, and can livelock). |
| `GBENCH_TB_TURN_MAX_TOKENS` | `0` (Terminus-2 sends none) | a per-turn `max_tokens` that Terminus-2 itself sends on its LLM calls. At `0` Terminus-2 sends no `max_tokens`, and gbench's relay sends the run's `--max-output-tokens` (none with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`). It replaces the run's `--max-output-tokens`, because the relay fills only requests that carry none. A value (e.g. `32768` with `--thinking`, paired with `GBENCH_TB_PARSER=xml`) is an opt-in backstop for a slow endpoint, where one uncapped turn can use most of a task's agent budget; non-canonical. |
| `GBENCH_TB_TIMEOUT_MULTIPLIER` | `1.0` | multiplies Harbor's per-task agent timeout (`--agent-timeout-multiplier`). The leaderboard requires `1.0`; another value (for a slow local endpoint) is non-canonical. The verifier timeout is never scaled. |
| `GBENCH_TB_CONTAINER_THREADS` | `0` (none) | opt-in thread cap (`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/RAYON_NUM_THREADS`) injected into the agent's shell (`--agent-env`) and the verifier's environment (`--verifier-env`). Tasks pin `cpus=1`, but `nproc` still reports the host's cores, so a runtime can oversubscribe; the leaderboard requires no resource overrides, so any value is non-canonical. |
| `GBENCH_TB_SALVAGE_PATCH` | *(on)* | the coercion of Harbor's XML-salvage return value (reachable only with `GBENCH_TB_PARSER=xml` and a turn cap), mounted read-only into the harness container and installed before Harbor's CLI starts. `0` runs the image's `harbor` unpatched. |
| `GBENCH_TB_KEEP_JOBS_DIR` | *(unset)* | keep Harbor's jobs dir (`result.json` / `trial.log` / `agent/trajectory.json`) for diagnosis. A path keeps there; `1` keeps in a temp dir; unset keeps only on failure. |

## Leaderboard comparability

The defaults are the TB2.1 leaderboard's settings, except the temperature (gbench's rule above):
Harbor 0.20.0 (in the harness image, with the
lock's package versions), the dataset at its pinned
content hash, Terminus-2 at `temperature: 1` (configs/leaderboard.yaml; its `reasoning_effort`
values are keyed on hosted models and do not apply to a served model) with its default `json`
parser (Terminus-2 itself sends no `max_tokens`; by default gbench's relay sends the run's
`--max-output-tokens`, a recorded deviation that makes the run not comparable;
`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` restores upstream's none), `timeout_multiplier` 1.0 and no environment overrides in the task
container or the verifier (leaderboard/SUBMIT.md). The result records them (`harbor_version`,
`harness_image`, `harness_telemetry`, `dataset`, `dataset_ref`, `agent_parser`, `agent_timeout_multiplier`, `turn_max_tokens`,
`container_threads`, `official_sampling`, `max_output_tokens_note`).

It also runs the leaderboard's **5 trials per task** (`-k 5`) with its **retries**: a trial that
raised an exception Harbor retries (Harbor 0.20.0's `RetryConfig` defaults: every exception but
the agent and verifier timeouts, a missing, empty or unreadable reward, a usage limit, a safety
refusal, an authentication error and an unknown model) is run again up to 3 times
(`retry: max_retries: 3` in configs/leaderboard.yaml; `harbor run --max-retries 3`,
`trial_max_retries` in the result). Harbor removes the failed attempt's directory and reruns the
trial under the same name, so its result, and the exception gbench classifies, are the last
attempt's. The suite itself marks a full `--thinking` run with every default
`leaderboard_comparable: true` (`leaderboard_comparable_reason: "full canonical run"`), and that is
the result's verdict only when gbench's relay filled no `max_tokens` (no `--max-output-tokens`, or
`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`). With the default fill the eval runner
(`evals._stamp_agent_max_tokens`) sets `leaderboard_comparable: false` and appends the cap
(`max_tokens=<N> (the run's --max-output-tokens) sent with <n> agent request(s) whose upstream
harness sends none`) to that reason. Otherwise
`leaderboard_comparable_reason` names every deviation of the run: a `--shard` or `--eval-limit`
subset, a temperature other than 1 (a no-think run's 0.0, `GBENCH_TERMINAL_BENCH_TEMPERATURE` or
`--temperature`), each
opt-in knob set to a non-canonical value (`GBENCH_TB_TIMEOUT_MULTIPLIER`,
`GBENCH_TB_TURN_MAX_TOKENS`, `GBENCH_TB_PARSER`, `GBENCH_TB_CONTAINER_THREADS`; listed without the
harness errors in `setting_deviations`), trials lost to harness errors (missing trials of the 89 x
5 included) and a non-zero Harbor exit. The canonical `terminal-bench-2-1` set is **89 tasks**.
