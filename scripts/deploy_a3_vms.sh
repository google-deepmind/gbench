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

# deploy_a3_vms.sh: probe zones for 4x a3-highgpu-8g (8 H100 each) and start one vLLM container
# per VM serving MODEL.
#
# Usage:
#   scripts/deploy_a3_vms.sh --project P --model MODEL --vm-prefix PREFIX --tp N [options]
#
# Nothing here is specific to a model. The model's serving recipe comes from the flags:
#   --tp N                   tensor parallel size (required; --tp x --dp = 8 GPUs per VM)
#   --dp N                   data parallel size (default: 8 / --tp)
#   --max-model-len N        cap the served window (default: the model's native context)
#   --tool-call-parser NAME  the model's vLLM tool-call parser (tool-calling evals need it)
#   --reasoning-parser NAME  the model's vLLM reasoning parser (`none`: it has none). gbench
#                            --thinking runs against these servers need it: without it vLLM
#                            returns the reasoning inside the answer text.
#   --vllm-serve-args '...'  any other `vllm serve` arguments (shell-quoted words); an argument
#                            this script sets (or an abbreviation of one) is refused
#   --image IMAGE            the vLLM image, pinned by a release version tag (e.g. v0.30.0) or a
#                            digest (image@sha256:...) (default below)
#
# Parser names are read without surrounding whitespace.
#
# Network exposure: each container uses --network=host and vLLM listens on port 8000 of every
# interface of the VM, without authentication. Restrict it with a VPC firewall rule that admits
# port 8000 only from the machine that runs gbench (or reach it through an SSH tunnel).

set -euo pipefail

DEFAULT_IMAGE="vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
GPUS_PER_VM=8
PYTHON="${PYTHON:-python3}"

usage() {
    echo "Usage: $0 --project PROJECT --model MODEL --vm-prefix VM_PREFIX --tp N [--dp N] [--max-model-len N] [--tool-call-parser NAME] [--reasoning-parser NAME] [--vllm-serve-args '...'] [--image IMAGE]" >&2
    exit 1
}
die() { echo "Error: $*" >&2; exit 1; }

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

PROJECT_ID=""
MODEL=""
VM_PREFIX=""
TP=""
DP=""
MAX_MODEL_LEN=""
TOOL_CALL_PARSER=""
REASONING_PARSER=""
VLLM_SERVE_ARGS=""
IMAGE="$DEFAULT_IMAGE"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT_ID="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --vm-prefix) VM_PREFIX="$2"; shift 2;;
        --tp) TP="$2"; shift 2;;
        --dp) DP="$2"; shift 2;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --tool-call-parser) TOOL_CALL_PARSER="$(trim "$2")"; shift 2;;
        --reasoning-parser) REASONING_PARSER="$(trim "$2")"; shift 2;;
        --vllm-serve-args) VLLM_SERVE_ARGS="$2"; shift 2;;
        --image) IMAGE="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown option: $1" >&2; usage;;
    esac
done

if [[ -z "$PROJECT_ID" || -z "$MODEL" || -z "$VM_PREFIX" ]]; then
    echo "Error: Missing required arguments." >&2
    usage
fi
[[ "$TP" =~ ^[1-9][0-9]*$ ]] || die "--tp is required: the tensor parallel size the model needs."
if [ -z "$DP" ]; then
    [ $((GPUS_PER_VM % TP)) -eq 0 ] || die "--tp ($TP) does not divide the $GPUS_PER_VM GPUs of a VM."
    DP=$((GPUS_PER_VM / TP))
fi
[[ "$DP" =~ ^[1-9][0-9]*$ ]] || die "--dp must be a positive integer."
[ $((TP * DP)) -eq "$GPUS_PER_VM" ] || die "--tp ($TP) x --dp ($DP) must equal the $GPUS_PER_VM GPUs of a VM."
if [ -n "$MAX_MODEL_LEN" ]; then
    [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || die "--max-model-len must be a positive integer."
fi
IMAGE_PROBLEM="$(image_pin_problem "$IMAGE")"
[ -z "$IMAGE_PROBLEM" ] || die "$IMAGE_PROBLEM"

# The container's arguments, each word quoted for the VM's shell. --vllm-serve-args is split
# like a shell command line and may not repeat an argument set here.
QUOTED_EXTRA="$("$PYTHON" - "$VLLM_SERVE_ARGS" <<'PY'
import shlex, sys
owned = {"--port", "--host", "--tensor-parallel-size", "-tp", "--data-parallel-size", "-dp",
         "--max-model-len", "--enable-auto-tool-choice", "--tool-call-parser",
         "--reasoning-parser", "--served-model-name", "--api-key", "--hf-token", "--model",
         "--config"}
try:
    tokens = shlex.split(sys.argv[1])
except ValueError as e:
    sys.exit(f"--vllm-serve-args could not be parsed ({e})")
if tokens and not tokens[0].startswith("-"):
    sys.exit(f"--vllm-serve-args must start with an option, got {tokens[0]!r}")
for tok in tokens:
    if not tok.startswith("-") or tok in ("-", "--"):
        continue
    name = tok.split("=", 1)[0]
    if name.startswith("--"):
        name = "--" + name[2:].replace("_", "-")
    hit = name if name in owned else None
    if hit is None and name.startswith("--") and len(name) > 3:
        # vLLM's parser accepts an unambiguous prefix of a long option (--max-model 4096)
        hit = next((k for k in sorted(owned) if k.startswith(name)), None)
        if hit:
            name = f"{name} (an abbreviation of {hit})"
    if hit:
        sys.exit(f"--vllm-serve-args may not set {name}: this script sets it (see its flags)")
print(" ".join(shlex.quote(t) for t in tokens))
PY
)" || die "invalid --vllm-serve-args (see above)"
SERVE_ARGS=("$MODEL" --tensor-parallel-size "$TP" --data-parallel-size "$DP"
            --gpu-memory-utilization 0.95 --enable-chunked-prefill --port 8000)
if [ -n "$MAX_MODEL_LEN" ]; then SERVE_ARGS+=(--max-model-len "$MAX_MODEL_LEN"); fi
if [ -n "$TOOL_CALL_PARSER" ]; then
    SERVE_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [ -n "$REASONING_PARSER" ] && [ "$(printf '%s' "$REASONING_PARSER" | tr '[:upper:]' '[:lower:]')" != none ]; then
    SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
QUOTED_SERVE_ARGS="$(printf '%q ' "${SERVE_ARGS[@]}")$QUOTED_EXTRA"
QUOTED_IMAGE="$(printf '%q' "$IMAGE")"

# Candidate zones (adjust to your project's regions / GPU quota)
CANDIDATE_ZONES=(
    "us-east4-b"
    "us-east4-a"
    "us-east4-c"
    "europe-west4-b"
    "europe-west4-c"
    "us-east7-b"
    "us-central1-b"
    "us-central1-a"
)

echo "=========================================================="
echo "Fast Multi-Zone A3 (32x H100) Cluster Provisioning"
echo "Project:   $PROJECT_ID"
echo "Model:     $MODEL"
echo "Image:     $IMAGE"
echo "Serve:     vllm serve $QUOTED_SERVE_ARGS"
if [ -z "$REASONING_PARSER" ]; then
    echo "WARNING: no --reasoning-parser: a gbench --thinking run against these servers would score the reasoning text as the answer (pass the model's parser, or --reasoning-parser none if it has none)." >&2
fi
echo "VM Prefix: $VM_PREFIX"
echo "Probing zones with available quota..."
echo "=========================================================="

ACTIVE_ZONE=""

for ZONE in "${CANDIDATE_ZONES[@]}"; do
    echo -n "Probing zone '$ZONE' for 4x a3-highgpu-8g machines... "

    FAILED=0
    for i in {1..4}; do
        gcloud compute instances create "${VM_PREFIX}-$i" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --machine-type=a3-highgpu-8g \
            --maintenance-policy=TERMINATE \
            --image-family=common-cu124-ubuntu-2204 \
            --image-project=deeplearning-platform-release \
            --boot-disk-size=500GB \
            --boot-disk-type=pd-ssd \
            --metadata=install-nvidia-driver=True \
            --quiet >/dev/null 2>&1 || { FAILED=1; break; }
    done

    if [ "$FAILED" -eq 0 ]; then
        echo "SUCCESS! 4x A3 instances created in $ZONE."
        ACTIVE_ZONE="$ZONE"
        break
    else
        echo "Out of stock / exhausted in $ZONE. Cleaning up and trying next zone..."
        for i in {1..4}; do
            gcloud compute instances delete "${VM_PREFIX}-$i" --zone="$ZONE" --project="$PROJECT_ID" --quiet >/dev/null 2>&1 || true
        done
    fi
done

if [ -z "$ACTIVE_ZONE" ]; then
    echo "Error: Could not allocate 4x A3 H100 machines across any candidate zones."
    exit 1
fi

echo ""
echo "=========================================================="
echo "Cluster allocated in: $ACTIVE_ZONE"
echo "Starting vLLM Docker containers across all 32 GPUs..."
echo "=========================================================="

NODE_IPS=($(gcloud compute instances list --project="$PROJECT_ID" --filter="name ~ ^${VM_PREFIX}-[1-4]$" --format="value(networkInterfaces[0].networkIP)"))
echo "Node IPs: ${NODE_IPS[*]}"

for i in {1..4}; do
    VM_NAME="${VM_PREFIX}-$i"
    echo "Launching vLLM on $VM_NAME..."
    gcloud compute ssh "$VM_NAME" --zone="$ACTIVE_ZONE" --project="$PROJECT_ID" --command="
        docker rm -f vllm 2>/dev/null || true
        docker run -d --name vllm --rm \
          --gpus all \
          --ipc=host \
          --network=host \
          -v /root/.cache/huggingface:/root/.cache/huggingface \
          -e HF_TOKEN=\"\${HF_TOKEN:-}\" \
          $QUOTED_IMAGE \
          $QUOTED_SERVE_ARGS
    " &
done
wait

echo "=========================================================="
echo "All 32x H100 GPUs are provisioning and serving $MODEL"
echo "Primary Node IP: ${NODE_IPS[0]:-unknown}"
echo "Port 8000 of every VM is open to whatever can reach it: restrict it with a VPC firewall rule."
echo "=========================================================="
