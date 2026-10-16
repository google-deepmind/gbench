# ui_control_osworld setup

> **Status: registered, not in `--evals all`.** `--evals all` leaves `ui_control_osworld` out
> (`NAMED_ONLY_SUITES` in `gbench/runners/evals.py`) because a graded run needs `/dev/kvm`, which
> accelerator serving hosts lack. `--evals ui_control_osworld` runs it, and it hard-errors
> (`infra_required`) naming whatever is missing: Docker, the orchestrator image, `/dev/kvm`, the
> Ubuntu qcow2 or the task VM image (once Docker answers, every missing one in the same message).
> See docs/evals_roadmap.md entry #3.

Canonical **OSWorld** ([github.com/xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)), an
**execution-based** desktop computer-use benchmark: **369 real tasks** over an Ubuntu-desktop VM. A
computer-use agent observes screenshots and emits `pyautogui` mouse/keyboard actions over many
steps; each task is scored by a **deterministic per-task evaluator** (getters read the final VM
state and compare to a reference), writing a `result.txt` value in [0,1]. There is no LLM judge, so no Gemini key is needed.

Pinned harness: commit `fc31a9049664292fcb35d6e501ee1dc839f2cf6d`.

## What gbench changes (everything else is upstream verbatim)
- Delegates to the pinned OSWorld harness inside a LOCAL orchestrator image
  (`gbench/docker/ui_control_osworld.Dockerfile`), driven **docker-out-of-docker**: OSWorld's
  `docker` provider (`DesktopEnv(provider_name="docker")`) spawns a **sibling QEMU-VM container**
  (`happysixd/osworld-docker`, which gbench requires to be the pinned digest
  `sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9`) per task on the host daemon,
  bind-mounting the Ubuntu qcow2.
- **Model wiring**: the agent (`mm_agents.PromptAgent`) runs host-side in the orchestrator and calls
  the served model over `/v1`. Its `call_llm` routes by model-name prefix; the `gpt` branch honours
  `OPENAI_BASE_URL`. Since vLLM rejects an unknown model id, gbench passes `--model gpt-4o` (a routing
  alias for the OpenAI-compatible path), sets `OPENAI_BASE_URL` to the served endpoint, and monkeypatches
  the outgoing payload `model` field to the real served name.
- **The docker socket.** The orchestrator container mounts `/var/run/docker.sock` and runs on the
  host network, as OSWorld's `docker` provider needs: it starts each task's QEMU-VM container on the
  host daemon itself and reaches the VM's control server and the served model from there. The
  model's `pyautogui` actions run inside the VM, not in the orchestrator. The residual risk: the
  socket gives the orchestrator process, which turns the model's replies into those actions,
  root-equivalent access to the host's docker daemon (and so to whatever the host can reach), and
  the host network lets it reach every service listening on the host; a flaw in the harness or
  the launcher that let model output run inside the orchestrator would reach both. gbench keeps
  upstream's design (the provider cannot run without a daemon it can reach) and keeps credentials
  out of that container: it gets `GBENCH_MODEL_API_KEY=dummy`, and the orchestrator's
  `OPENAI_BASE_URL` is always gbench's relay (`swebench_common.EndpointRelay`, a thread of the
  gbench process on `127.0.0.1`), which forwards only `/v1/chat/completions`, `/v1/completions`
  and `/v1/models` to the endpoint and adds the key when there is one
  (`GBENCH_OSWORLD_MODEL_API_KEY`); the result records it as `model_endpoint_relay`, and the
  endpoint failures it counts make the run `completed_with_errors` (`endpoint_failures`).
- **DooD identity-mount** (the skillsbench/wildclaw pitfall): the provider bind-mounts the qcow2 via
  `os.path.abspath(path_to_vm)`, and `VMS_DIR` is `./docker_vm_data` relative to CWD, so the
  orchestrator chdir's to the identity-mounted `GBENCH_OSWORLD_VM_DIR` (which holds
  `docker_vm_data/Ubuntu.qcow2`), and that path is identical on the host daemon.
- **A fresh directory per run.** OSWorld's `run.py` resumes from the `result.txt` files its
  `result_dir` already holds (`get_unfinished` skips those tasks and `get_result` reads them), and
  the routing alias puts every model's results under the same `gpt-4o` path, so a `result_dir`
  shared across runs would hand one run another run's (another model's) results. Every run
  therefore writes to its own directory, `$GBENCH_OSWORLD_VM_DIR/gbench_runs/<run id>/`
  (`results/`, the subset manifest and the summary; the result records it as `run_dir`), created
  mode `0700` for your user (the orchestrator runs as root in its container, which may write there;
  a Docker daemon with user-namespace remapping cannot, and the run then produces no summary). The
  launcher hands the directory back to your user when it ends (`OSWORLD_HOST_OWNER`). gbench
  keeps it after the run (the per-step screenshots and trajectories are there); delete old ones
  when you no longer need them. An orchestrator image without the current launcher label (below)
  is refused before the run starts.

## Metric
Headline **`accuracy`** = mean per-task success (`result.txt`) over the selected tasks that
produced a `result.txt`, as OSWorld's own `get_result` averages the result files present, less the
tasks during which a model request failed on the endpoint's side. A task without a `result.txt`
is a task whose run raised in OSWorld's loop (the VM, the model endpoint or the evaluator failed).
A model request that got no reply, or HTTP 5xx, 429, 401, 403 or 404, does not stop the task:
OSWorld's `PromptAgent` goes on with an empty reply and the task still writes a `result.txt`, so
the launcher records every model request that raised or got a reply other than HTTP 200 in that
example's result directory (`gbench_endpoint_failures.jsonl`; a request rejection, HTTP 400, 413
or 422 as for a context overflow, is recorded but stays the model's outcome, as upstream's own
handling decides), and the summary counts the endpoint's per task (`endpoint_failures`). Both are
harness errors, never the model's outcome: left out of the headline and of each domain's
`success_rate`, counted (`n_missing`, `n_endpoint_failed`, `scoring_excluded`; `effective_n` is
`n_scored`) and named in `status_reason` (status `completed_with_errors`); with no task scored,
`accuracy` is null with `accuracy_withheld_reason` and `status` is `failed`. The in-container
summary's raw rate, which counts a missing task as 0, is kept as `success_rate_missing_as_zero`
(overall and per domain). The per-domain breakdown and `n_selected`/`n_scored`/`n_missing` are
also reported. gbench never fabricates a scalar; a run that produces no summary hard-errors.

## Prerequisites (the suite HARD-ERRORS via `infra_required`, never skips / never a fake number)

### 1. A host with `/dev/kvm` (nested virtualization), the binding blocker
OSWorld's Ubuntu VM only becomes ready within its 300 s `/screenshot` timeout with **hardware
virtualization**. Without `/dev/kvm` (e.g. a GCP VM without nested-virt licensing) QEMU falls back to
TCG software emulation and the desktop will not boot in time, so the suite hard-errors. Run on bare
metal or a nested-virt-enabled VM.

### 2. Build the LOCAL orchestrator image (gbench never pulls it)
```bash
docker build -t gbench-ui-control-osworld:fc31a904 -f gbench/docker/ui_control_osworld.Dockerfile gbench/docker
```
Bakes the pinned OSWorld harness + its deps (torch/transformers/opencv/pyautogui) + the docker SDK,
installed against `gbench/docker/ui_control_osworld.constraints.txt` (the resolved set of the image
whose model wiring was checked; the build fails on any other set, and the Python and Debian sets are
recorded in `/opt/gbench`), +
the gbench adapter. The image carries the label `org.gbench.ui-control-osworld-launcher=4` (the
launcher that writes into the run's own directory, applies and reports `--shard`, draws
`--eval-limit` across the domains, skips an empty shard, records the model requests that failed on
the endpoint's side per example, hands the run directory back to your user and passes gbench's
temperature as run.py's `--temperature`); gbench refuses an
image without it,
or with another value (`infra_required` with the build command). Rebuild after pulling gbench
changes to `gbench/docker/ui_control_osworld_*`. (These deps hard-conflict with the vLLM stack, hence a container, not the gbench serving environment.)

### 3. Provision the Ubuntu VM disk (`GBENCH_OSWORLD_VM_DIR`)
Download + unzip the OSWorld Ubuntu qcow2 (~12 GB) into `docker_vm_data/`, and point
`GBENCH_OSWORLD_VM_DIR` at the parent (it is identity-mounted; each run writes to
`gbench_runs/<run id>/` under it):
```bash
mkdir -p /srv/osworld/docker_vm_data && cd /srv/osworld/docker_vm_data
hf download xlangai/ubuntu_osworld Ubuntu.qcow2.zip --repo-type dataset --local-dir .
unzip Ubuntu.qcow2.zip          # -> Ubuntu.qcow2
export GBENCH_OSWORLD_VM_DIR=/srv/osworld
```

### 4. The task VM image + the served model
```bash
# the canonical OSWorld VM runner, pinned by digest, under the name OSWorld's docker provider starts
docker pull happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9
docker tag happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9 happysixd/osworld-docker
```
OSWorld's docker provider starts the VM container from `happysixd/osworld-docker` by name (its
`latest` tag) and would pull that tag itself if it were missing; gbench never pulls it. The digest
is Docker Hub's `happysixd/osworld-docker:latest` (the repository's only tag, pushed 2024-10-17;
linux/amd64), and the tag makes the name the provider uses point at it. Before every run gbench
checks that `happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9` is on the
host daemon and that `happysixd/osworld-docker`, the name the provider starts, is that same image;
a missing or different image **hard-errors** (`infra_required`) with the two commands above. The
provider's name cannot be changed, so `GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE` set to any other
image is refused (`infra_required`). The result records `task_image` (the name, the pinned
reference and the verified image id).
- A **served model at `/v1`** reachable from the orchestrator (with `--network host`, `127.0.0.1`).
  For screenshot tasks the model must be multimodal.

### 5. On a cloud VM: block the metadata server for docker containers
The model's pyautogui actions run inside the task VM, whose container (started by OSWorld's docker
provider on docker's default bridge) has network access, and a docker container reaches the VM's
metadata server at `169.254.169.254`, which hands out the VM's service-account token. Before the
orchestrator starts, gbench checks that it is unreachable from docker's default bridge and from a
user-defined bridge network (gbench creates a temporary one, `gbench-metadata-probe-<id>`, and
removes it right after): a short-lived probe container of `gbench-ui-control-osworld:fc31a904`
(`--pull never`) opens and closes a TCP connection to `169.254.169.254:80` and sends nothing. A
connection that opens, or a probe without a clear verdict, hard-errors (`infra_required`) naming
the fix. Block it once on the host; gbench never runs sudo or changes the host's firewall itself:
```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```
See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server).

## Run
```bash
export GBENCH_OSWORLD_VM_DIR=/srv/osworld
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ui_control_osworld --max-output-tokens 65536
```
(`--max-output-tokens` is required for every eval run; the OSWorld agent sends its own per-step
`max_tokens`, so the value does not reach this suite's requests.)

## Sharding and `--eval-limit`
`--shard I/N` (validated by gbench: a malformed value is an error before anything starts) is
forwarded normalized into the orchestrator, whose launcher takes shard I of N of the sorted
`(domain, example_id)` task list by round-robin (task I, I+N, ...) and rebuilds the manifest the
harness iterates. `--eval-limit N` (or, only when `--eval-limit` is not set,
`GBENCH_OSWORLD_LIMIT`, which must be a positive integer: anything else, 0 and negative numbers
included, is `infra_required` before any container starts; with `--eval-limit` set the knob is
ignored with a warning) is then a seeded draw of N tasks
from the shard, balanced across the domains (the launcher's copy of `sampling.stratified_indices`,
seed `ui_control_osworld`: the same tasks on every machine), not the head of the sorted list, which
is one domain. The result records `shard` (so the run is marked `shard_applied`), the partial
shard's comparability reason, `selection` (the manifest's task count, the limit and
`limit_draw`) and one row per selected task in `sample_traces` (`task` = `<domain>/<example_id>`,
`domain`, `result`, `endpoint_failures`; a task without a `result.txt`, or with a model request
that failed on the endpoint's side, is a harness error with `scoring_excluded`).
The run logs its planned workload on one line at its start and records it as `eval_limit_plan`;
after the run the record's `samples` and `episodes` are the tasks the launcher actually selected
(a domain, a shard or a limit above the selection's size narrows them; the logged count is kept
as `samples_logged` when it differs), and `gbench --merge` sums the shards' plans.
An orchestrator that does not report the shard it applied, or for a limited run how it drew the
subset, predates this launcher: the run hard-errors naming the rebuild. A shard with none of the
tasks does not start OSWorld and reports `accuracy: null` with `accuracy_withheld_reason`
(`empty_shard`). `gbench --merge` re-scores the union of the shards' task rows
(`ui_control_osworld.merge_shards`: the mean `result.txt` over the tasks that produced one,
overall and per domain; a task without one, or with an endpoint failure, stays a harness error),
and keeps each shard's `selection` in `selection_by_shard`.

## Split topology: nested-virt CPU VM + a remote GPU-served model

**Use this when your GPU host cannot provide `/dev/kvm`.** GCP **accelerator machine families
(A2/A3/G2) do not support nested virtualization** (nor does E2), so an H100 box such as
`a3-highgpu-8g` can never expose `/dev/kvm`, and no container can conjure it (a container shares the
host kernel). OSWorld needs *no GPU* of its own - it is VM/CPU work that only needs a reachable model
endpoint - so run the orchestrator on a small **nested-virt-enabled CPU VM** and point it at the model
still served on the GPU box. Nothing in the runner changes; only the endpoint URL and the host differ.

**1. Serve the model on the GPU host so the CPU VM can reach it.** vLLM binds `127.0.0.1` by default;
start it with `--host 0.0.0.0` and open the port inside your VPC:
```bash
# on the GPU host (e.g. the a3-highgpu-8g box)
vllm serve google/gemma-4-26B-A4B-it --host 0.0.0.0 --port 8000   # ...your usual serving flags
# allow the CPU VM's subnet to reach it (adjust to your VPC/tags)
gcloud compute firewall-rules create allow-osworld-to-vllm \
  --network=<your-vpc> --direction=INGRESS --action=ALLOW \
  --rules=tcp:8000 --source-ranges=<cpu-vm-subnet-cidr>
```

**2. Create a nested-virt CPU VM** (Haswell+; N2 / N2D / C3 are eligible - NOT E2 / A-series). Put it
in the **same zone + VPC** as the GPU host so it reaches the model over the internal network:
```bash
gcloud compute instances create osworld-runner \
  --zone=<same-zone-as-gpu-host> \
  --machine-type=n2-standard-8 \
  --enable-nested-virtualization \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB --boot-disk-type=pd-ssd \
  --network=<your-vpc> --subnet=<same-subnet>
```

**3. Confirm `/dev/kvm` exists on the new VM:**
```bash
sudo apt-get update && sudo apt-get install -y cpu-checker && sudo kvm-ok   # -> "KVM acceleration can be used"
ls -l /dev/kvm                                                              # -> present
```

**4. Provision the OSWorld prerequisites on that VM** exactly as in the sections above, on the CPU VM:
   install Docker (add your user to the `docker` group), install gbench from its checkout (`pip install -e .`;
   a `--remote-endpoint` run needs neither vLLM nor a GPU, see the deps policy), then do prereq steps
   **2-4**: build the orchestrator image, download+unzip the Ubuntu qcow2 into
   `$GBENCH_OSWORLD_VM_DIR/docker_vm_data/`, and pull and tag the pinned task image
   (`docker pull happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9` then
   `docker tag happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9 happysixd/osworld-docker`).

**5. Run, pointing `--remote-endpoint` at the GPU host's internal IP** (the orchestrator runs with
`--network host`, so it reaches that IP directly):
```bash
export GBENCH_OSWORLD_VM_DIR=/srv/osworld
GPU_HOST_IP=$(gcloud compute instances describe <gpu-host-name> --zone=<zone> \
  --format='get(networkInterfaces[0].networkIP)')
gbench --evals-only \
       --remote-endpoint "http://${GPU_HOST_IP}:8000/v1" \
       --tokenizer google/gemma-4-26B-A4B-it \
       --evals ui_control_osworld --max-output-tokens 65536
# smoke first: GBENCH_OSWORLD_LIMIT=3 (or --eval-limit 3)
```

**6. Tear down** the CPU VM when done (it bills while it exists):
`gcloud compute instances delete osworld-runner --zone=<zone>`.

> Notes. The model must be **multimodal** for the default `screenshot` observation type. If the GPU
> host is unreachable by IP, verify the firewall rule and that vLLM was started with `--host 0.0.0.0`
> (`curl http://${GPU_HOST_IP}:8000/v1/models` from the CPU VM should list the model). Everything else
> - reaping, the identity-mount, scoring - is identical to the single-host path.

## Environment knobs
| Env var | Default | Meaning |
|---|---|---|
| `GBENCH_UI_CONTROL_OSWORLD_IMAGE` | `gbench-ui-control-osworld:fc31a904` | orchestrator image tag |
| `GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE` | `happysixd/osworld-docker` | must be unset or `happysixd/osworld-docker`: OSWorld's docker provider starts that name (the pinned digest, step 4); another value is `infra_required` |
| `GBENCH_OSWORLD_VM_DIR` | (required) | identity-mounted dir holding `docker_vm_data/Ubuntu.qcow2`; each run writes to its own `gbench_runs/<run id>/` under it |
| `GBENCH_OSWORLD_MODEL_ENDPOINT` | (derived) | override the model `/v1` URL |
| `GBENCH_OSWORLD_MODEL_API_KEY` | (none) | the endpoint's key: added by gbench's relay, never passed to the orchestrator (which gets the placeholder `dummy`) |
| `GBENCH_OSWORLD_DOMAIN` | (all) | single OSWorld domain, or `all` |
| `GBENCH_OSWORLD_LIMIT` | (none) | total tasks (smoke), a positive integer: a seeded draw balanced across the domains, within the shard (applies only without `--eval-limit`: with `--eval-limit` set it is ignored with a warning) |
| `GBENCH_OSWORLD_OBS_TYPE` | `screenshot` | `screenshot` / `a11y_tree` / `screenshot_a11y_tree` / `som` |
| `GBENCH_OSWORLD_MAX_STEPS` | 15 | per-task step budget |
| `GBENCH_OSWORLD_TEST_META` | `test_all.json` | manifest under `evaluation_examples/` |
| `GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S` | 172800 | orchestrator wall-clock cap (a timeout is infra_required and the sibling VMs are reaped) |
| `GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE` / `--temperature` | gbench's rule: 1.0 with `--thinking`, 0.0 without | the model's temperature, replacing the rule (`GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE` wins over `--temperature`). Sent as OSWorld run.py's `--temperature` (the launcher's `OSWORLD_TEMPERATURE`; the mm_agents PromptAgent puts it in every request) and set on every request by gbench's endpoint relay. Recorded as `temperature` / `temperature_source`. `top_p` stays run.py's 0.9; `enable_thinking` is not applied (the PromptAgent sends no reasoning field) |

## What hard-errors (`infra_required`, never a skip / never a fake number)
- Docker CLI/daemon not reachable, or the orchestrator image not built (or built before the
  current launcher: a label `org.gbench.ui-control-osworld-launcher` other than `4`).
- `/dev/kvm` absent (no hardware virtualization).
- `GBENCH_OSWORLD_VM_DIR` unset, or missing `docker_vm_data/Ubuntu.qcow2`.
- The `happysixd/osworld-docker` task image not on the host daemon, or not the pinned digest
  `sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9` (step 4).
- The cloud metadata server reachable from a docker container, or the check without a clear
  verdict (step 5).
- The orchestrator produced no summary in this run's directory (a harness failure,
  `status:error`, not 0%), or it does not report the shard it applied or (for a limited run) how it
  drew the subset: an image that predates the launcher; rebuild it.
- No task selected (a `GBENCH_OSWORLD_DOMAIN` the manifest does not hold); a shard with no task is
  not an error (above).

## Leaderboard comparability
Always **`leaderboard_comparable=False`**: a self-hosted agent + single trial, subset-capable, and
the official OSWorld leaderboard is maintainer-run. Scoring itself is the canonical deterministic
execution evaluator (no judge), so the per-task numbers are faithful; missing tasks are harness
errors, left out of the headline and named in `status_reason` and in the reason.
