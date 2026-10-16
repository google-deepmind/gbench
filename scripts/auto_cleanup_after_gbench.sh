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

# auto_cleanup_after_gbench.sh: wait for one gbench process to finish, then stop the proxy and
# tunnels a `scripts/eval_cluster.sh --setup-only` run started (by PID, as it printed them) and
# release the cluster's GPUs. Nothing else on the host is touched: no process is matched by name.
#
# Usage:
#   scripts/auto_cleanup_after_gbench.sh --pid GBENCH_PID --kill-pids "PROXY_PID TUNNEL_PID ..." \
#       --cluster CLUSTER --node-pool POOL --region REGION --project PROJECT [--log FILE]

set -euo pipefail

usage() {
    echo "Usage: $0 --pid GBENCH_PID --kill-pids \"PID ...\" --cluster CLUSTER --node-pool POOL --region REGION --project PROJECT [--log FILE]" >&2
    exit 1
}

GBENCH_PID=""
KILL_PIDS=""
CLUSTER_NAME=""
NODE_POOL=""
REGION=""
PROJECT_ID=""
LOG_FILE="${TMPDIR:-/tmp}/gbench_auto_cleanup.log"
POLL_S="${AUTO_CLEANUP_POLL_S:-15}"     # how often the gbench process is checked
FLUSH_S="${AUTO_CLEANUP_FLUSH_S:-15}"   # the wait for file writes after it exits

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) GBENCH_PID="$2"; shift 2;;
        --kill-pids) KILL_PIDS="$2"; shift 2;;
        --cluster) CLUSTER_NAME="$2"; shift 2;;
        --node-pool) NODE_POOL="$2"; shift 2;;
        --region) REGION="$2"; shift 2;;
        --project) PROJECT_ID="$2"; shift 2;;
        --log) LOG_FILE="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown argument: $1" >&2; usage;;
    esac
done

if [[ -z "$GBENCH_PID" || -z "$CLUSTER_NAME" || -z "$NODE_POOL" || -z "$REGION" || -z "$PROJECT_ID" ]]; then
    echo "Error: missing required arguments." >&2
    usage
fi
for pid in $GBENCH_PID $KILL_PIDS; do
    [[ "$pid" =~ ^[0-9]+$ ]] || { echo "Error: '$pid' is not a process id." >&2; exit 1; }
done
kill -0 "$GBENCH_PID" 2>/dev/null || { echo "Error: no process $GBENCH_PID to wait for." >&2; exit 1; }

exec >> "$LOG_FILE" 2>&1

echo "=========================================================="
echo "Starting gbench monitor at $(date)"
echo "Cluster:   $CLUSTER_NAME ($REGION) [Project: $PROJECT_ID]"
echo "Node Pool: $NODE_POOL"
echo "PID:       $GBENCH_PID"
echo "Stops:     ${KILL_PIDS:-nothing}"
echo "=========================================================="

# Only the given gbench process is waited for.
while kill -0 "$GBENCH_PID" 2>/dev/null; do
    sleep "$POLL_S"
done

echo ""
echo "gbench (PID $GBENCH_PID) finished at $(date); waiting ${FLUSH_S}s for file writes to flush..."
sleep "$FLUSH_S"

echo "Executing cluster teardown..."
for pid in $KILL_PIDS; do
    if kill "$pid" 2>/dev/null; then
        echo ">> Stopped PID $pid"
    else
        echo ">> PID $pid had already exited"
    fi
done

echo ">> Scaling deployment/vllm-deployment to 0..."
kubectl scale deployment/vllm-deployment --replicas=0 2>/dev/null || true

echo ">> Resizing GKE node pool $NODE_POOL to 0..."
gcloud container clusters resize "$CLUSTER_NAME" \
  --node-pool="$NODE_POOL" \
  --num-nodes=0 \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --quiet

echo ""
echo "=========================================================="
echo "All resources released at $(date)."
echo "=========================================================="
