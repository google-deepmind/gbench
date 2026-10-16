#!/usr/bin/env bash
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

# eval_cluster.sh: serve a model on a GKE GPU node pool (vllm-gke.yaml), tunnel every pod to
# this machine, balance the tunnels behind scripts/async_proxy.py, run gbench against the proxy,
# and release the GPUs on exit (Ctrl+C safe). See docs/distributed-gbench.md.
#
# Usage:
#   scripts/eval_cluster.sh MODEL "EVALS" --gpu TYPE --num-gpus N --tp N --pool POOL[,POOL...] \
#       --project P --cluster C --region R [options]
#
# Examples:
#   scripts/eval_cluster.sh org/model "gpqa_diamond mmlu_pro aime" --gpu h100 --num-gpus 8 \
#       --tp 1 --reasoning-parser NAME --pool h100-pool --project P --cluster C --region us-central1
#   scripts/eval_cluster.sh org/model "bfcl" --gpu gb200 --num-gpus 4 --tp 2 --pool a4x-pool \
#       --tool-call-parser NAME --no-thinking --vllm-serve-args '--trust-remote-code' ...
#
# Nothing here is specific to a model. The model's serving recipe comes from the flags:
#   --tp N                tensor parallel size (required: it depends on the model's size)
#   --dp N                data parallel size (default: --num-gpus / --tp)
#   --max-model-len N     cap the served window (default: the model's native context)
#   --tool-call-parser X  the model's vLLM tool-call parser (tool-calling evals need it)
#   --reasoning-parser X  the model's vLLM reasoning parser; required for a thinking run (the
#                         default): without it vLLM returns the reasoning inside the answer text,
#                         which the suites would score as the answer. `none` states that the model
#                         has none (nothing is passed to vLLM); or use --no-thinking.
#   --vllm-serve-args '...'  any other `vllm serve` arguments (shell-quoted words); an argument
#                         this script sets (or an abbreviation of one, e.g. --max-model) is refused
#   --image IMAGE         the vLLM image, pinned by a release version tag (e.g. v0.30.0) or a
#                         digest (image@sha256:...) (default: the manifest's pin)
#
# MODEL and "EVALS" are the only positional arguments (an extra one is an error). Parser names are
# read without surrounding whitespace.
#
# Other options: --nodes N (default by --gpu: 4, l4 8), --batch-size N (2048), --sandboxes N (96),
# --max-output-tokens N (65536), --no-thinking, --results-dir DIR, --timeout SECONDS (rollout),
# --tunnels-per-pod N (default --dp), --proxy-port N (default: a free port; a busy one is an
# error), --proxy-script PATH, --manifest-path PATH, --setup-only (leave the tunnels and proxy
# running and print how to use and tear them down). PYTHON selects the interpreter that runs the
# proxy and the checks (default: python3; it needs aiohttp, a gbench dependency).
#
# What it guarantees before gbench runs: every tunnel is bound to a free local port by its own
# kubectl process (a bind failure stops the script), every pod's /v1/models lists MODEL, the
# proxy is alive on its port and lists MODEL too. A tunnel that drops later is restarted on the
# same port (logged in RESULTS_DIR/cluster/); the proxy sends new requests to the other pods
# meanwhile. Only the processes this script started are stopped on exit.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/async_proxy.py"
MANIFEST_PATH="$REPO_ROOT/vllm-gke.yaml"
PYTHON="${PYTHON:-python3}"

die() { echo "ERROR: $*" >&2; exit 1; }

# The value without its leading and trailing whitespace (a parser name of only spaces is empty).
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Empty when IMAGE is pinned: name@sha256:<64 hex digits>, or a tag that reads as a release
# version (v0.30.0, 0.30.0, v0.10.1.1, v0.6.3.post1, v0.30.0-cu129). The tag is what follows ':'
# in the last path component, so a registry port (registry:5000/vllm) is not a tag. Otherwise
# prints the problem.
image_pin_problem() {
    local image="$1" last tag lower part parts
    local digest_re='@sha256:[0-9a-f]{64}$'
    local version_re='^v?[0-9]+(\.[0-9]+){1,3}((rc|post)[0-9]+)?([.-][A-Za-z0-9]+)*$'
    if [[ "$image" == *@* ]]; then
        [[ "$image" =~ $digest_re ]] && return 0
        echo "--image $image has a malformed digest; pin image@sha256:<64 hex digits> or a released version tag."
        return 0
    fi
    last="${image##*/}"
    if [[ "$last" != *:* || "$last" == *: ]]; then
        echo "--image $image has no tag or digest; pin a released version tag or a digest (image@sha256:...)."
        return 0
    fi
    tag="${last#*:}"
    lower="$(printf '%s' "$tag" | tr '[:upper:]' '[:lower:]')"
    read -ra parts <<< "${lower//[._-]/ }"
    for part in "${parts[@]}"; do
        case "$part" in
            latest|nightly|main|master|edge|stable)
                echo "--image $image is a moving tag; pin a released version tag or a digest (image@sha256:...)."
                return 0;;
        esac
    done
    if ! [[ "$tag" =~ $version_re ]]; then
        echo "--image $image: tag '$tag' is not a release version (e.g. v0.30.0); pin a released version tag or a digest (image@sha256:...)."
    fi
    return 0
}

# --- Defaults ---
MODEL=""
EVAL_SUITE=""
GPU_TYPE=""
TP=""
DP=""
BATCH_SIZE=2048
THINKING=true
MAX_OUTPUT_TOKENS=65536
RESULTS_DIR=""
SANDBOXES=96
USER_NODES=""
USER_REGION=""
USER_TIMEOUT=""
USER_GPUS=""
SETUP_ONLY=false
CLI_POOLS=()
IMAGE=""
MAX_MODEL_LEN=""
TOOL_CALL_PARSER=""
REASONING_PARSER=""
VLLM_SERVE_ARGS=""
PROXY_PORT=""
TUNNELS_PER_POD=""
PROJECT_ID="${PROJECT_ID:-}"
CLUSTER_NAME="${CLUSTER_NAME:-}"

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --evals) EVAL_SUITE="$2"; shift 2;;
        --gpu) GPU_TYPE="$2"; shift 2;;
        --num-gpus|--gpus) USER_GPUS="$2"; shift 2;;
        --pool|--pools|--candidate-pools)
            IFS=',' read -ra POOL_ARR <<< "$2"
            for p in "${POOL_ARR[@]}"; do
                CLI_POOLS+=("$p")
            done
            shift 2;;
        --tp) TP="$2"; shift 2;;
        --dp) DP="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --sandboxes) SANDBOXES="$2"; shift 2;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2;;
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        --nodes) USER_NODES="$2"; shift 2;;
        --region) USER_REGION="$2"; shift 2;;
        --project) PROJECT_ID="$2"; shift 2;;
        --cluster) CLUSTER_NAME="$2"; shift 2;;
        --image) IMAGE="$2"; shift 2;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --tool-call-parser) TOOL_CALL_PARSER="$(trim "$2")"; shift 2;;
        --reasoning-parser) REASONING_PARSER="$(trim "$2")"; shift 2;;
        --vllm-serve-args) VLLM_SERVE_ARGS="$2"; shift 2;;
        --proxy-port) PROXY_PORT="$2"; shift 2;;
        --tunnels-per-pod) TUNNELS_PER_POD="$2"; shift 2;;
        --proxy-script) PROXY_SCRIPT="$2"; shift 2;;
        --manifest-path) MANIFEST_PATH="$2"; shift 2;;
        --timeout) USER_TIMEOUT="$2"; shift 2;;
        --no-thinking) THINKING=false; shift;;
        --setup-only) SETUP_ONLY=true; shift;;
        -*) die "Unknown option: $1";;
        *) POSITIONAL_ARGS+=("$1"); shift;;
    esac
done

[ ${#POSITIONAL_ARGS[@]} -le 2 ] || die "unexpected argument(s): ${POSITIONAL_ARGS[*]:2} (the positional arguments are MODEL and \"EVALS\"; everything else is an option, e.g. --results-dir DIR)."
if [ ${#POSITIONAL_ARGS[@]} -ge 1 ]; then
    MODEL="${POSITIONAL_ARGS[0]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -ge 2 ]; then
    EVAL_SUITE="${POSITIONAL_ARGS[1]}"
fi

is_uint() { [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]; }

[ -n "$MODEL" ] || die "the model is required (first argument or --model): the id vllm serves, e.g. org/model."
[ ${#CLI_POOLS[@]} -gt 0 ] || die "--pool is required: the GKE node pool(s) to use."
CANDIDATE_POOLS=("${CLI_POOLS[@]}")
is_uint "${USER_GPUS:-}" || die "--num-gpus is required: the GPUs of one pod (e.g. 4 for gb200, 8 for h100)."
GPUS_PER_POD="$USER_GPUS"
is_uint "${TP:-}" || die "--tp is required: the tensor parallel size the model needs (it depends on the model's size and the GPU memory)."
if [ -z "$DP" ]; then
    [ $((GPUS_PER_POD % TP)) -eq 0 ] || die "--num-gpus ($GPUS_PER_POD) is not a multiple of --tp ($TP)."
    DP=$((GPUS_PER_POD / TP))
fi
is_uint "$DP" || die "--dp must be a positive integer."
[ $((TP * DP)) -eq "$GPUS_PER_POD" ] || die "--tp ($TP) x --dp ($DP) = $((TP * DP)), must equal --num-gpus ($GPUS_PER_POD)."
if [ -n "$MAX_MODEL_LEN" ]; then is_uint "$MAX_MODEL_LEN" || die "--max-model-len must be a positive integer."; fi
TUNNELS_PER_POD="${TUNNELS_PER_POD:-$DP}"
is_uint "$TUNNELS_PER_POD" || die "--tunnels-per-pod must be a positive integer."
# A thinking run needs the model's reasoning parser on the server this script starts: without one
# vLLM returns the reasoning inside the answer text, and every suite would score it as the answer.
if [ "$THINKING" = true ] && [ -z "$REASONING_PARSER" ]; then
    die "a thinking run (the default) needs --reasoning-parser NAME: without the model's reasoning parser vLLM returns the reasoning inside the answer text, which would be scored as the answer. Pass the parser the model's serving recipe names, --reasoning-parser none if the model has none, or --no-thinking."
fi

# --- Node count per GPU type (the cluster shape, not the model) ---
case "$GPU_TYPE" in
    gb200|b200|h100|a100) DEFAULT_NODES=4;;
    l4) DEFAULT_NODES=8;;
    "") die "--gpu is required (gb200, b200, h100, a100 or l4)";;
    *) die "Unknown GPU type '$GPU_TYPE'. Choose: gb200, b200, h100, a100, l4";;
esac
TARGET_NODES="${USER_NODES:-$DEFAULT_NODES}"
is_uint "$TARGET_NODES" || die "--nodes must be a positive integer."

[ -n "$PROJECT_ID" ] || die "--project is required."
[ -n "$CLUSTER_NAME" ] || die "--cluster is required."
[ -n "$USER_REGION" ] || die "--region is required."
REGION="$USER_REGION"

# --- Files and tools ---
[ -f "$PROXY_SCRIPT" ] || die "the proxy script $PROXY_SCRIPT does not exist (--proxy-script)."
[ -f "$MANIFEST_PATH" ] || die "the manifest $MANIFEST_PATH does not exist (--manifest-path)."
for tool in kubectl gcloud curl gbench; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' is not on PATH."
done
"$PYTHON" -c 'import aiohttp' 2>/dev/null || die "$PYTHON cannot import aiohttp (the proxy needs it; set PYTHON to the interpreter gbench is installed in)."

# --- The image: the manifest's pin unless --image names another pinned one ---
if [ -n "$IMAGE" ]; then
    why="$(image_pin_problem "$IMAGE")"
    [ -z "$why" ] || die "$why"
fi

# --- The serve settings: shell-quoted words that must not repeat what this script sets ---
SERVE_CHECK="$("$PYTHON" - "$VLLM_SERVE_ARGS" <<'PY'
import shlex, sys
owned = {
    "--port": "the port is fixed at 8000 in the pod",
    "--host": "the pod binds vLLM's default",
    "--tensor-parallel-size": "--tp", "-tp": "--tp",
    "--data-parallel-size": "--dp", "-dp": "--dp",
    "--max-model-len": "--max-model-len",
    "--enable-auto-tool-choice": "--tool-call-parser", "--tool-call-parser": "--tool-call-parser",
    "--reasoning-parser": "--reasoning-parser",
    "--served-model-name": "the model argument (the served id is checked against it)",
    "--api-key": "VLLM_API_KEY in the pod environment (and in this shell)",
    "--hf-token": "the hf-secret",
    "--model": "the model argument", "--config": "the flags of this script",
}
try:
    tokens = shlex.split(sys.argv[1])
except ValueError as e:
    print(f"--vllm-serve-args could not be parsed ({e})"); sys.exit(1)
if tokens and not tokens[0].startswith("-"):
    print(f"--vllm-serve-args must start with an option, got {tokens[0]!r}"); sys.exit(1)
for tok in tokens:
    if not tok.startswith("-") or tok in ("-", "--"):
        continue
    name = tok.split("=", 1)[0]
    if name.startswith("--"):
        name = "--" + name[2:].replace("_", "-")
    owner = owned.get(name)
    if owner is None and name.startswith("--") and len(name) > 3:
        # vLLM's parser accepts an unambiguous prefix of a long option (--max-model 4096)
        prefixed = [k for k in owned if k.startswith(name)]
        if prefixed:
            owner = owned[prefixed[0]]
            name = f"{name} (an abbreviation of {prefixed[0]})"
    if owner is not None:
        print(f"--vllm-serve-args may not set {name}: use {owner}"); sys.exit(1)
PY
)" || die "$SERVE_CHECK"

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="results/eval_${GPU_TYPE}_$(date +%m%d_%H%M%S)"
fi
LOG_DIR="$RESULTS_DIR/cluster"
mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"

echo "=========================================================="
echo "Distributed GPU Evaluation Sweep"
echo "=========================================================="
echo "  Cluster:     $CLUSTER_NAME ($REGION)"
echo "  Model:       $MODEL"
echo "  Evals:       ${EVAL_SUITE:-all}"
echo "  GPU type:    $GPU_TYPE ($GPUS_PER_POD GPUs/pod, $TARGET_NODES pods)"
echo "  TP=$TP  DP=$DP"
echo "  Image:       ${IMAGE:-the manifest's pinned image}"
echo "  Max len:     ${MAX_MODEL_LEN:-the model's native context}"
echo "  Parsers:     tool=${TOOL_CALL_PARSER:-none} reasoning=${REASONING_PARSER:-none}"
echo "  Serve args:  ${VLLM_SERVE_ARGS:-none}"
echo "  Batch size:  $BATCH_SIZE"
echo "  Sandboxes:   $SANDBOXES"
echo "  Thinking:    $THINKING"
echo "  Max tokens:  $MAX_OUTPUT_TOKENS"
echo "  Results:     $RESULTS_DIR"
echo "  Setup only:  $SETUP_ONLY"
echo "=========================================================="

# --- Cleanup state: only what this script started ---
ACTIVE_POOL=""
TUNNEL_PIDS=()
PROXY_PID=""

cleanup() {
    echo ""
    echo "Cleaning up the resources this run started..."
    if [ -n "$PROXY_PID" ]; then
        kill "$PROXY_PID" 2>/dev/null || true
        echo "  Proxy (PID $PROXY_PID) stopped"
    fi
    for pid in "${TUNNEL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [ ${#TUNNEL_PIDS[@]} -gt 0 ]; then
        echo "  ${#TUNNEL_PIDS[@]} port-forward tunnels stopped"
    fi
    kubectl scale deployment/vllm-deployment --replicas=0 >/dev/null 2>&1 || true
    echo "  Deployment scaled to 0 replicas"
    if [ -n "$ACTIVE_POOL" ]; then
        gcloud container clusters resize "$CLUSTER_NAME" \
            --node-pool="$ACTIVE_POOL" --num-nodes=0 --region="$REGION" \
            --project="$PROJECT_ID" --async --quiet >/dev/null 2>&1 || true
        echo "  Node pool '$ACTIVE_POOL' scaling to 0 (async)"
    fi
    echo "Cleanup complete."
}

on_signal() { exit 130; }

# Setup-only leaves everything running on success; on a failure it still tears down.
if [ "$SETUP_ONLY" = false ]; then
    trap cleanup EXIT
else
    trap 'rc=$?; [ "$rc" -ne 0 ] && cleanup' EXIT
fi
trap on_signal INT TERM

free_port() {
    "$PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

port_is_free() {
    "$PYTHON" - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
try:
    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
        sys.exit(1)
except OSError:
    pass
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

# Exit 0 when http://127.0.0.1:PORT/v1/models lists MODEL; prints what it serves otherwise.
serves_model() {
    local port="$1"
    local body
    local auth=()
    [ -n "${VLLM_API_KEY:-}" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")
    body="$(curl -sf --max-time 10 "${auth[@]}" "http://127.0.0.1:$port/v1/models" 2>/dev/null)" || { echo "no answer"; return 1; }
    "$PYTHON" - "$MODEL" "$body" <<'PY'
import json, sys
try:
    ids = [str(m.get("id")) for m in json.loads(sys.argv[2]).get("data") or [] if isinstance(m, dict)]
except Exception as e:
    print(f"unreadable /v1/models ({e})"); sys.exit(1)
if sys.argv[1] not in ids:
    print(f"serves {ids or 'no model'}"); sys.exit(1)
PY
}

# ============================================================
# 1. Apply the manifest (0 replicas until configured)
# ============================================================
if ! kubectl get secret hf-secret >/dev/null 2>&1; then
    HF_SECRET_VALUE="${HF_TOKEN:-}"
    if [ -z "$HF_SECRET_VALUE" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
        HF_SECRET_VALUE="$(cat "$HOME/.cache/huggingface/token")"
    fi
    if [ -n "$HF_SECRET_VALUE" ]; then
        kubectl create secret generic hf-secret --from-literal=token="$HF_SECRET_VALUE" >/dev/null 2>&1 \
            || echo "WARNING: could not create the hf-secret; gated models will not download." >&2
    fi
    unset HF_SECRET_VALUE
fi

echo "Applying manifest: $MANIFEST_PATH"
kubectl apply -f "$MANIFEST_PATH" || die "kubectl apply -f $MANIFEST_PATH failed."

# ============================================================
# 2. GPU pool autoscaling
# ============================================================
echo ""
echo "Configuring GPU pool autoscaling for $GPU_TYPE..."
for POOL in "${CANDIDATE_POOLS[@]}"; do
    if ! gcloud container node-pools describe "$POOL" --cluster="$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        echo ">> Pool '$POOL' does not exist, skipping."
        continue
    fi
    echo ">> Configuring autoscaling on '$POOL' (total-max-nodes: $TARGET_NODES, location-policy: ANY)..."
    gcloud container node-pools update "$POOL" \
        --cluster="$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID" \
        --enable-autoscaling --total-min-nodes=0 --total-max-nodes="$TARGET_NODES" \
        --location-policy=ANY --quiet 2>&1 || true
    ACTIVE_POOL="$POOL"
    break
done
[ -n "$ACTIVE_POOL" ] || die "no valid GPU pool among: ${CANDIDATE_POOLS[*]}."

# ============================================================
# 3. Configure the model, parallelism and GPU count, then scale
# ============================================================
echo ""
echo "Configuring: $MODEL (TP=$TP, DP=$DP, $GPUS_PER_POD GPUs/pod) on $TARGET_NODES nodes"
kubectl patch deployment vllm-deployment --type='json' -p="[
  {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/limits/nvidia.com~1gpu\", \"value\": \"$GPUS_PER_POD\"},
  {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/requests/nvidia.com~1gpu\", \"value\": \"$GPUS_PER_POD\"}
]" || die "could not set the GPU count on the deployment."
kubectl patch deployment vllm-deployment -p "{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"cloud.google.com/gke-nodepool\":\"$ACTIVE_POOL\"}}}}}" \
    || die "could not pin the deployment to node pool $ACTIVE_POOL."
if [ -n "$IMAGE" ]; then
    kubectl set image deployment/vllm-deployment vllm="$IMAGE" || die "could not set the image $IMAGE."
fi
kubectl set env deployment/vllm-deployment \
    MODEL_NAME="$MODEL" \
    TENSOR_PARALLEL_SIZE="$TP" \
    DATA_PARALLEL_SIZE="$DP" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    TOOL_CALL_PARSER="$TOOL_CALL_PARSER" \
    REASONING_PARSER="$REASONING_PARSER" \
    VLLM_SERVE_ARGS="$VLLM_SERVE_ARGS" || die "could not set the serve settings on the deployment."
SERVED_IMAGE="$(kubectl get deployment vllm-deployment -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)"

kubectl scale deployment/vllm-deployment --replicas="$TARGET_NODES" || die "could not scale the deployment."
ROLLOUT_TIMEOUT="${USER_TIMEOUT:-$(( 600 + TARGET_NODES * 45 ))}"
kubectl patch deployment vllm-deployment -p "{\"spec\":{\"progressDeadlineSeconds\":$ROLLOUT_TIMEOUT}}" >/dev/null 2>&1 || true
echo "Waiting up to ${ROLLOUT_TIMEOUT}s for all $TARGET_NODES pods to load the model..."
kubectl rollout status deployment/vllm-deployment --timeout="${ROLLOUT_TIMEOUT}s" || die "the deployment rollout failed (kubectl logs -l app=vllm)."

# ============================================================
# 4. Tunnel each pod to free local ports, then start the proxy
# ============================================================
echo ""
echo "Setting up the tunnels..."
read -ra PODS <<< "$(kubectl get pods -l app=vllm --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}')"
[ ${#PODS[@]} -gt 0 ] || die "no running vllm pod found."
BACKEND_PORTS=()

# One supervised `kubectl port-forward` on 127.0.0.1:PORT. A forward that fails before it is
# ready ends the supervisor (the caller then stops the script); one that drops later is
# restarted on the same port, logged.
start_tunnel() {
    local pod="$1" port="$2" log="$3"
    (
        child=""
        trap '[ -n "$child" ] && kill "$child" 2>/dev/null; exit 0' TERM INT
        while true; do
            kubectl port-forward --address 127.0.0.1 "pod/$pod" "$port:8000" >>"$log" 2>&1 &
            child=$!
            wait "$child"
            rc=$?
            if ! grep -q "Forwarding from 127.0.0.1:$port" "$log"; then
                echo "tunnel: port-forward of $pod to 127.0.0.1:$port failed before it was ready (exit $rc)" >>"$log"
                exit 1
            fi
            echo "tunnel: port-forward of $pod on 127.0.0.1:$port exited (exit $rc) at $(date); restarting in 3s" >>"$log"
            sleep 3
        done
    ) </dev/null >>"$log" 2>&1 &
    TUNNEL_PIDS+=($!)
}

for pod in "${PODS[@]}"; do
    for t in $(seq 1 "$TUNNELS_PER_POD"); do
        port="$(free_port)" || die "could not find a free local port."
        log="$LOG_DIR/tunnel-$pod-$t.log"
        : > "$log"
        start_tunnel "$pod" "$port" "$log"
        pid="${TUNNEL_PIDS[-1]}"
        ready=false
        for _ in $(seq 1 60); do
            if grep -q "Forwarding from 127.0.0.1:$port" "$log"; then ready=true; break; fi
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        if [ "$ready" != true ]; then
            echo "--- $log ---" >&2; tail -5 "$log" >&2
            die "the tunnel to $pod on 127.0.0.1:$port did not come up."
        fi
        why="$(serves_model "$port")" || die "pod $pod (127.0.0.1:$port) does not serve $MODEL: $why."
        BACKEND_PORTS+=("$port")
        echo "  $pod -> 127.0.0.1:$port (tunnel $t/$TUNNELS_PER_POD)"
    done
done

if [ -n "$PROXY_PORT" ]; then
    is_uint "$PROXY_PORT" || die "--proxy-port must be a port number."
    port_is_free "$PROXY_PORT" || die "--proxy-port $PROXY_PORT is in use by another process; pick a free port (or leave it unset for a free one)."
else
    PROXY_PORT="$(free_port)" || die "could not find a free local port for the proxy."
fi
BACKENDS_CSV=$(IFS=,; echo "${BACKEND_PORTS[*]}")
"$PYTHON" "$PROXY_SCRIPT" --host 127.0.0.1 --port "$PROXY_PORT" --backends "$BACKENDS_CSV" \
    --expect-model "$MODEL" </dev/null >>"$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
proxy_ready=false
for _ in $(seq 1 120); do
    kill -0 "$PROXY_PID" 2>/dev/null || break
    if grep -q "async_proxy listening on http://127.0.0.1:$PROXY_PORT" "$LOG_DIR/proxy.log"; then
        proxy_ready=true; break
    fi
    sleep 0.5
done
if [ "$proxy_ready" != true ]; then
    echo "--- $LOG_DIR/proxy.log ---" >&2; tail -5 "$LOG_DIR/proxy.log" >&2
    die "the proxy did not start on 127.0.0.1:$PROXY_PORT."
fi
why="$(serves_model "$PROXY_PORT")" || die "the proxy on 127.0.0.1:$PROXY_PORT does not serve $MODEL: $why."
ENDPOINT="http://127.0.0.1:$PROXY_PORT/v1"
echo "Proxy live: $ENDPOINT -> [$BACKENDS_CSV] (every backend lists $MODEL)"

# What served the run (gbench cannot see a remote server's settings): recorded next to the results.
"$PYTHON" - "$LOG_DIR/serving.json" "$MODEL" "${SERVED_IMAGE:-$IMAGE}" "$TP" "$DP" "$GPUS_PER_POD" \
    "$GPU_TYPE" "$MAX_MODEL_LEN" "$TOOL_CALL_PARSER" "$REASONING_PARSER" "$VLLM_SERVE_ARGS" \
    "$ENDPOINT" "${PODS[@]}" <<'PY' || die "could not write $LOG_DIR/serving.json."
import json, shlex, sys
(out, model, image, tp, dp, gpus, gpu_type, max_len, tool, reasoning, serve_args,
 endpoint, *pods) = sys.argv[1:]
with open(out, "w") as f:
    json.dump({"model": model, "image": image or None, "tensor_parallel_size": int(tp),
               "data_parallel_size": int(dp), "gpus_per_pod": int(gpus), "gpu_type": gpu_type,
               "pods": pods, "max_model_len": int(max_len) if max_len else None,
               "tool_call_parser": tool or None, "reasoning_parser": reasoning or None,
               "vllm_serve_args": shlex.split(serve_args), "endpoint": endpoint}, f, indent=2)
PY

# ============================================================
# 5. Setup-only mode: print how to use and tear down, and exit
# ============================================================
if [ "$SETUP_ONLY" = true ]; then
    echo ""
    echo "=========================================================="
    echo "SETUP COMPLETE - proxy running on $ENDPOINT"
    echo "=========================================================="
    kubectl get pods -l app=vllm -o wide
    echo ""
    echo "Run gbench against it:"
    echo "  gbench --evals-only --evals ... --models $MODEL --tokenizer $MODEL --remote-endpoint $ENDPOINT ..."
    echo ""
    echo "Tear down when done:"
    echo "  kill $PROXY_PID ${TUNNEL_PIDS[*]}"
    echo "  kubectl scale deployment/vllm-deployment --replicas=0"
    echo "  gcloud container clusters resize $CLUSTER_NAME --node-pool=$ACTIVE_POOL --num-nodes=0 --region=$REGION --project=$PROJECT_ID --quiet"
    echo "or, once gbench (PID <pid>) finishes:"
    echo "  scripts/auto_cleanup_after_gbench.sh --pid <pid> --kill-pids \"$PROXY_PID ${TUNNEL_PIDS[*]}\" --cluster $CLUSTER_NAME --node-pool $ACTIVE_POOL --region $REGION --project $PROJECT_ID"
    exit 0
fi

# ============================================================
# 6. Run gbench
# ============================================================
EVAL_SUITE="${EVAL_SUITE:-all}"
read -ra EVAL_WORDS <<< "$EVAL_SUITE"
GBENCH_ARGS=(
    --evals-only
    --evals "${EVAL_WORDS[@]}"
    --models "$MODEL"
    --tokenizer "$MODEL"
    --batch-sizes "$BATCH_SIZE"
    --sandboxes "$SANDBOXES"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --remote-endpoint "$ENDPOINT"
    --results-dir "$RESULTS_DIR"
)
if [ "$THINKING" = true ]; then
    GBENCH_ARGS+=(--thinking)
fi

echo ""
echo "Command: gbench ${GBENCH_ARGS[*]}"
echo ""
gbench "${GBENCH_ARGS[@]}"
GBENCH_RC=$?

echo ""
echo "=========================================================="
echo "gbench exited with $GBENCH_RC. Results: $RESULTS_DIR"
echo "=========================================================="
exit "$GBENCH_RC"
