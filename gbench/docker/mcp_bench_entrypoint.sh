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
# Entrypoint for the gbench-mcp-bench image (the model container). Wires the model-under-test at
# the gbench /v1 endpoint (openai_compatible provider), leaves the server key file the MCP-Bench
# loader reads empty, and runs the gbench launcher (which points the five credentialed servers at
# the bridge to gbench's credentialed-servers container, injects the Gemini cascade judge, checks
# that every server of the task set can serve, runs no task if one cannot, and otherwise delegates
# to upstream main()). The container runs model text (the Scientific Computing server's sympy
# parse_expr, the OSINT server's commands), so it refuses to start when one of the five server
# keys is set: they belong only in the credentialed-servers container.
#
# Required env (docker run -e): MCP_ENDPOINT (gbench /v1), MCP_MODEL (served model name),
#                               GEMINI_API_KEY (the cascade judge's key: gbench passes its Gemini
#                               relay's per-run placeholder, never a real key, together with
#                               GEMINI_OPENAI_BASE_URL and GBENCH_GEMINI_RELAY_URL, the relay's
#                               addresses; the relay, in the gbench process, adds the real key).
# Optional: GEMINI_API_KEYS (the same placeholder; a real key list is rotated through),
#           MCP_CREDENTIALED_RELAY / MCP_CREDENTIALED_TOKEN (gbench's relay to its
#           credentialed-servers container, where the five server keys are; without them the
#           five credentialed servers cannot serve and no task runs),
#           MCP_DISTRACTION_COUNT, MCP_DISABLE_STABILITY, MCP_TASK_LIMIT, GBENCH_SHARD,
#           MCP_ENABLE_THINKING / MCP_MAX_TOKENS / MCP_REQUEST_TIMEOUT_S / MCP_TEMPERATURE /
#           MCP_REQUEST_LOG (the model request settings the launcher applies; see
#           mcp_bench_run.py).
#           MCP_DNS_NAMES / MCP_HARNESS_LOG (the network check and the per-task harness record;
#           see mcp_bench_run.py).
# gbench passes the relay placeholder and the endpoint URL by name (docker run -e NAME), so no
# such value is on its command line.
set -uo pipefail

MCPBENCH_DIR="${MCPBENCH_DIR:-/app/mcp-bench}"
ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
cd "$MCPBENCH_DIR"

# --- model-under-test: hijack the llama-3-1-8b openai_compatible slot -> gbench /v1 ---
: "${MCP_ENDPOINT:?MCP_ENDPOINT (gbench /v1 base url) is required}"
: "${MCP_MODEL:?MCP_MODEL (served model name) is required}"
export LLAMA_3_1_8B_API_KEY="${MCP_ENDPOINT_API_KEY:-dummy}"
export LLAMA_3_1_8B_BASE_URL="$MCP_ENDPOINT"
export LLAMA_3_1_8B_MODEL="$MCP_MODEL"
export MCP_MODEL_CONFIG_NAME="llama-3-1-8b"

# --- server API keys: never in this container ---
for k in GOOGLE_MAPS_API_KEY NCI_API_KEY HF_TOKEN NPS_API_KEY NASA_API_KEY; do
  if [ -n "${!k:-}" ]; then
    echo "mcp_bench: $k is set in the model container, where model text runs; the server keys" \
         "belong only in gbench's credentialed-servers container. Not starting." >&2
    exit 2
  fi
done
# the loader reads mcp_servers/api_key (KEY = VALUE lines): empty here
KEYFILE="$MCPBENCH_DIR/mcp_servers/api_key"
: > "$KEYFILE"

# --- judge (cascade) needs GEMINI_API_KEY; fail fast if absent ---
: "${GEMINI_API_KEY:?GEMINI_API_KEY is required (powers the gbench Gemini cascade judge)}"
# a relay placeholder is only usable with the relay's address
case "$GEMINI_API_KEY" in
  gbench-gemini-relay-*)
    : "${GBENCH_GEMINI_RELAY_URL:?GBENCH_GEMINI_RELAY_URL is required with a relay placeholder key}"
    ;;
esac

exec python "$ADAPTER_DIR/mcp_bench_run.py"
