<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Distributed evaluation: many GPUs through GKE, or `--shard` across nodes

## 1. Problem

Running the academic suites of `gbench` with reasoning (`--thinking`) generates millions of output
tokens. On a single server the sweep is bound by queue depth, for example:
- **`mmlu_pro`** (12,032 questions): at least 6.7 hours (TIGER-Lab evaluate_from_apiX.py starts
  requests at least 2 s apart; `--shard` splits it)
- **`lcb`** (454 questions): about 2.3 hours
- **`codeforces`** (422 questions): about 1.8 hours

Two ways to spread the work are described here:
- **One client, many servers** (Steps 1-5): serve the model on several GKE pods and put a local
  round-robin proxy in front of them, so one `gbench` run drives them all.
  `scripts/eval_cluster.sh` automates this.
- **Many clients, one shard each** (Step 4b): each node runs `--shard I/N` of every suite against
  its own endpoint, and `gbench --merge` combines the results.

Node pools, machine types and GPU quotas are your project's; nothing here depends on a particular
pool name, zone or quota.

---

## 2. Step-by-step

### Step 1: GPU node pools that scale from 0

Create one or more GKE node pools for the GPU type you will use, at 0 nodes so they cost nothing
until a run scales them, for example:

```bash
gcloud container node-pools create POOL \
    --project=PROJECT --cluster=CLUSTER --region=REGION --node-locations=ZONE \
    --machine-type=MACHINE_TYPE \
    --accelerator=type=ACCELERATOR,count=GPUS_PER_NODE,gpu-driver-version=latest \
    --num-nodes=0 --disk-size=500GB --disk-type=DISK_TYPE
```

Use the disk type the machine type supports (for example `hyperdisk-balanced` on machine types
that do not take `pd-ssd`).

---

### Step 2: The Kubernetes manifest (`vllm-gke.yaml`)

[`vllm-gke.yaml`](../vllm-gke.yaml) defines an internal LoadBalancer Service and a Deployment
that starts at **0 replicas**. It names no model and no model recipe: its launcher always runs
`vllm serve MODEL --tensor-parallel-size TP --data-parallel-size DP --gpu-memory-utilization 0.95
--enable-chunked-prefill --disable-custom-all-reduce --port 8000` and adds the rest from
environment variables:

| Variable | Meaning |
| --- | --- |
| `MODEL_NAME` | the model to serve (required; the container exits when it is empty) |
| `TENSOR_PARALLEL_SIZE`, `DATA_PARALLEL_SIZE` | the parallel layout of each pod: the manifest sets 1 and 4, and requests 4 GPUs per pod (`TENSOR_PARALLEL_SIZE` x `DATA_PARALLEL_SIZE` must equal the pod's GPU request); the launcher uses 1 for a variable set empty |
| `MAX_MODEL_LEN` | the served window; empty serves the model's native context |
| `TOOL_CALL_PARSER` | adds `--enable-auto-tool-choice --tool-call-parser NAME` |
| `REASONING_PARSER` | adds `--reasoning-parser NAME`; empty or `none` adds nothing (`none` states that the model has no reasoning parser). A --thinking run needs the model's parser: without it vLLM returns the reasoning inside the answer text, which the suites would score as the answer |
| `VLLM_SERVE_ARGS` | any other `vllm serve` arguments, shell-quoted words (no default); it may not set `--model`, `--port`, `--host`, the parallel sizes (`--tensor-parallel-size` / `-tp`, `--data-parallel-size` / `-dp`), `--max-model-len`, `--enable-auto-tool-choice`, the parsers, `--served-model-name`, `--api-key`, `--hf-token` or `--config`, or an abbreviation of one (vLLM's parser accepts `--max-model` for `--max-model-len`), or the container exits with an error. It may repeat one of the fixed flags above (for example `--gpu-memory-utilization 0.90`): the last value wins |

The image is pinned by tag and digest (`vllm/vllm-openai:v0.30.0@sha256:8a69ffad...`, see the
manifest). `scripts/eval_cluster.sh` sets every variable, the pod's GPU request (TP x DP), the
node pool and the image itself. By hand (GPUS = TP x DP; the node-pool pin is needed only when
the cluster has several GPU pools):

```bash
kubectl apply -f vllm-gke.yaml
kubectl patch deployment vllm-deployment --type='json' -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/nvidia.com~1gpu", "value": "GPUS"},
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/nvidia.com~1gpu", "value": "GPUS"}]'
kubectl patch deployment vllm-deployment \
    -p '{"spec":{"template":{"spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":"POOL"}}}}}'
kubectl set env deployment/vllm-deployment MODEL_NAME=MODEL TENSOR_PARALLEL_SIZE=TP \
    DATA_PARALLEL_SIZE=DP TOOL_CALL_PARSER=NAME REASONING_PARSER=NAME \
    VLLM_SERVE_ARGS='--trust-remote-code'      # an example value: take the model's own recipe
kubectl set image deployment/vllm-deployment vllm=IMAGE   # only to change the pinned image
kubectl scale deployment/vllm-deployment --replicas=PODS
```

Without the GPU patch a pod keeps the manifest's 4 GPUs: a TP x DP above 4 then fails at vLLM
startup, and one below 4 leaves GPUs idle.

Model recipe flags (image limits, processor settings, remote code) belong in `VLLM_SERVE_ARGS`.

---

### Step 3: Tunnel every pod and balance the tunnels

`kubectl port-forward svc/...` sends everything to a single pod. To spread the load over every
pod, tunnel each pod to its own local port and put the proxy in front of them.

#### A. Tunnel each pod to a local port (`8001`, `8002`, ...)
```bash
PODS=($(kubectl get pods -l app=vllm -o jsonpath='{.items[*].metadata.name}'))
PORT=8001
for pod in "${PODS[@]}"; do
  kubectl port-forward --address 127.0.0.1 pod/$pod $PORT:8000 > /dev/null 2>&1 &
  ((PORT++))
done
```

#### B. Start the round-robin proxy on port 8000
```bash
python3 scripts/async_proxy.py --port 8000 --backends 8001,8002,8003,8004 --expect-model MODEL
```

[`scripts/async_proxy.py`](../scripts/async_proxy.py) binds 127.0.0.1, refuses a busy port (never
another one), streams responses as they arrive (server-sent events keep their chunking), skips a
backend that refuses the connection (a dropped tunnel) and never re-sends a request a backend has
accepted, and before it listens checks that every backend's `/v1/models` lists `MODEL` (a
backend serving another model stops it). It does not cap the requests in flight to the backends
(`--max-connections 0`, the default; aiohttp's own default would queue every request past its
100th), so the client's concurrency reaches the pods; `--max-connections N` sets a cap, and a
request past it waits for a free connection. Every request holds two sockets, so the proxy raises
its open-file soft limit to the hard limit at start. It needs aiohttp, a gbench dependency.

---

### Step 4: Run `gbench`

```bash
nohup gbench --evals-only \
  --evals gpqa_diamond mmlu_pro aime \
  --models MODEL --tokenizer MODEL \
  --batch-sizes 512 --sandboxes 96 \
  --thinking \
  --max-output-tokens 65536 \
  --remote-endpoint http://127.0.0.1:8000/v1 \
  --results-dir ./results/evalcloud-think \
  > /tmp/evalcloud-think.log 2>&1 &
```

`--models` names the id the endpoint serves; gbench checks it against `/v1/models` before any
suite runs.

---

### Step 4b: Split the suites across nodes with `--shard` and merge the results

Instead of one client driving every pod through a proxy, each node can run its own shard of every
suite against its own endpoint, and the results are merged afterwards:

```bash
# on node I of N (or export GBENCH_SHARD=I/N instead of the flag)
gbench --evals-only --evals gpqa_diamond mmlu_pro aime --shard I/N \
  --models MODEL --tokenizer MODEL --max-output-tokens 65536 \
  --remote-endpoint http://127.0.0.1:8000/v1 --results-dir ./results/node-I

# on any machine, once every node is done
gbench --merge ./results/node-1 ./results/node-2 ... ./results/node-N --results-dir ./results
```

Give every node its own `--results-dir` (as above). A node pointed at a run directory another
run is still writing writes its own `<timestamp>` directory inside it instead
(`results_dir_not_shared` in its metadata.json), so two nodes running at the same time never
overwrite each other's results; a node started after another finished in that directory
continues it in place only when that directory's metadata.json records the same host, shard and
gbench commit; otherwise it writes its own `<timestamp>` directory too (`results_dir_not_shared`).

- **The split.** `--shard I/N` takes shard I of the suite's full, deterministically ordered item
  list, round-robin, so the N shards are disjoint and union to the full set. `--eval-limit` is
  drawn AFTER the split, from the shard (N shards run up to N x `--eval-limit` items). A
  malformed `--shard` or `GBENCH_SHARD` is an error before anything runs. `--shard` applies to
  the eval suites; the Golden Set, quality and performance pillars run in full on every node.
- **Shard identity on every result.** `shard` ("I/N"), `shard_index`, `shard_count` and
  `shard_applied: true`; each trace carries `global_idx` (its position in the full list) and
  `sample_key` (the item's own id, else a hash of it), the same on every node. A partial shard is
  not leaderboard-comparable (`sharded run (--shard I/N) ...` in
  `leaderboard_comparable_reason`). An empty shard (more shards than items) is an honest empty
  row: status `success`, accuracy `null` with the reason (`empty_shard`), nothing generated.
  metadata.json records the run's `shard` and `eval_limit`.
- **Which suites shard.** The native suites that run through `base.run_eval_suite`, and these
  suites with their own loop or harness, each applying the shard before `--eval-limit` and
  recording it: `aider_polyglot`, `livebench`, `scicode`, `toolbench`, `terminal_bench` (5 trials
  per task, selected by task name), `skillsbench`, `tau2`, `tau3`, `bfcl_v4_agentic` (its own
  run; inside `bfcl` the agentic part runs whole on shard 1), `agent_dojo`, `complexfuncbench`,
  `gaia2`, `wildclawbench`, `mcp_bench`, `ui_control_osworld` and the SWE family (one-shot and
  agentic).
- **Suites that cannot shard** run their full set on every node and record
  `shard_applied: false` with `shard_requested`; `--merge` refuses them (their full-set result
  from any one node is the result). `--shard` splits the eval suites only: the other pillars a
  node runs (performance, quality, Golden Set) run in full, with a warning, and their records
  say so the same way (metadata.json lists them under `shard_not_applied`). Run those pillars
  on one node and the shard nodes with `--evals-only`.
- **The merge.** `--merge` checks the directories hold shards 1..N of one run (the same N,
  `--eval-limit`, gbench commit and evals scaffold, each shard once, each `success` or
  `completed_with_errors`) and writes one result per suite under
  `--results-dir/merged-<timestamp>/`: the suite's own `merge_shards` when its headline is a
  corpus or macro metric, or its fields need recomputing (`agent_dojo`, `aider_polyglot`,
  `amc_aime`, `bfcl_v4_agentic`, `coco_caption`, `codeforces`, `complexfuncbench`, `gaia2`,
  `gpqa`, `gpqa_diamond`, `lcb`, `livebench`, `mmlu`, `mmlu_redux`, `multipl_e`, `omnidocbench`,
  `scicode`, `skillsbench`, `tau2`, `tau3`, `terminal_bench`, `toolbench`, `ui_control_osworld`,
  `wildclawbench`: `lcb` recomputes every LiveCodeBench field from the stored verdicts,
  `codeforces` its coverage and `amc_aime` its components; `livebench` refuses a question found
  in two shards, as the generic merge does; `gaia2` and `ui_control_osworld` keep each shard's
  draw as `selection_by_shard`),
  else the traces re-counted (every shard's traces kept, an item being its `sample_key` at its
  `global_idx`: rows with identical content stay apart, and an item found in two shards is
  refused, since the shards then overlap; accuracy, categories and @k recomputed). A suite without `merge_shards` whose traces cannot be re-counted into its
  headline (no traces, no `sample_key`, or a headline that is not correct over scored, as for
  `mcp_bench`) is refused, and so is a suite whose own merge fails. The merged result drops the
  shard reason when the union is the full set and keeps every other comparability reason. The
  merged status is judged over the union: each shard's causes are named with their shard, and the
  over-10% budget-cut rule and the server-parser counts are recounted from the union's traces;
  each node's server record is kept as `server_by_shard`. Missing,
  repeated or failed shards exit 2 and name them. No field of a merged result is one shard's
  value standing for the union: counts are summed, the Gemini models every shard used are
  combined (which models answered, per role or group, with their call counts; after a suite's
  own `merge_shards` too), `duration_s` is the sum (`duration_s_by_shard` keeps each), and a
  field the shards disagree on that no rule combines moves to `merge_unreconciled_fields` with
  each shard's value. Every suite is checked and merged before anything is written: a refusal
  writes nothing, `--merge ... --dry-run` checks and merges and writes nothing, and each merge
  writes a new `merged-<timestamp>` directory of its own.

---

### Step 5: One command (`eval_cluster.sh`)

[`scripts/eval_cluster.sh`](../scripts/eval_cluster.sh) runs Steps 2-4 and releases the GPUs on
exit (Ctrl+C safe):

```bash
scripts/eval_cluster.sh MODEL "EVALS" --gpu TYPE --num-gpus N --tp N --pool POOL[,POOL...] \
    --project P --cluster C --region R [--dp N] [--nodes N] [--max-model-len N] \
    [--tool-call-parser X] [--reasoning-parser X] [--vllm-serve-args '...'] [--image IMAGE] \
    [--results-dir DIR] [--setup-only]
```

For example:

```bash
scripts/eval_cluster.sh org/model "gpqa_diamond mmlu_pro aime" --gpu h100 --num-gpus 8 --tp 1 \
    --reasoning-parser NAME --pool POOL --project P --cluster C --region REGION
```

`--pool`, `--gpu`, `--num-gpus`, `--tp`, `--project`, `--cluster` and `--region` are required;
`--reasoning-parser` is required for a thinking run (the default): pass the model's parser, `none`
if it has none, or use `--no-thinking`; the script stops before touching the cluster otherwise;
`MODEL` and `"EVALS"` are the only positional arguments (an extra one is an error, so there is
no positional results directory), and there is no default model recipe (`--tp` depends on the
model's size; the parsers and serve arguments come from the model's own recipe). `--vllm-serve-args`
may not set an argument the script sets, or an abbreviation of one (`--max-model`), and `--image`
must be pinned by a release version tag (e.g. `v0.30.0`) or a digest (`image@sha256:...`); a
moving tag such as `latest` is refused. The script's
header lists every option (`--nodes` defaults by `--gpu`, `--batch-size` 2048, `--sandboxes` 96,
`--max-output-tokens` 65536, `--no-thinking`, `--proxy-port`, ...).

**How it works**:
1. Uses the first `--pool` that exists: configures its autoscaling for the requested nodes and
   pins the deployment to it.
2. Sets the manifest's variables (`kubectl set env`) and the image, scales the deployment and
   waits for every pod to be ready.
3. Tunnels every pod to a free local port, each by its own `kubectl` process (a bind failure
   stops the script), and checks that each pod's `/v1/models` lists `MODEL`. A tunnel that drops
   later is restarted on the same port (logged under `RESULTS_DIR/cluster/`).
4. Starts the proxy on a free port (or `--proxy-port`; a busy one is an error) and checks it lists
   `MODEL` too.
5. Records the serve settings in `RESULTS_DIR/cluster/serving.json` and runs `gbench --evals-only
   --models MODEL ... --batch-sizes 2048` (the `--batch-size` default) against the proxy.
6. On exit stops only the processes it started, scales the deployment and the node pool to 0, and
   exits with gbench's exit code.

With `--setup-only` it leaves the tunnels and the proxy running and prints their PIDs, how to use
them, and an `auto_cleanup_after_gbench.sh --pid <pid> --kill-pids ...` line that tears them down
and releases the pool once your gbench run ends.

`scripts/deploy_a3_vms.sh` is the VM alternative (one vLLM container per 8-GPU VM, always started
with `--gpu-memory-utilization 0.95 --enable-chunked-prefill --port 8000` besides the parallel
sizes): `--tp` is required, the model flags (`--max-model-len`, the parsers, `--vllm-serve-args`) have no defaults
(it warns when no `--reasoning-parser` is given, since a gbench --thinking run against these
servers needs it; `none` passes nothing),
and vLLM listens on port 8000 of every interface of the VM without authentication, so admit that
port only from the machine that runs gbench with a VPC firewall rule (or use an SSH tunnel).
