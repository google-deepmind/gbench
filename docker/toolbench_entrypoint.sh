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
# gbench toolbench container entrypoint: start the cached /virtual server, run StableToolBench's
# DFSDT inference against the model-under-test's served /v1 endpoint, convert the answer trees, and
# write them to /out. The SoPR/SoWR judge runs in gbench (Gemini cascade) after this exits.
#
# The container runs the model's tool calls (the DFSDT loop dispatches them to the /virtual
# server in this container), so it holds no real key: the gbench runner starts it on a docker
# bridge network whose metadata server it found blocked, the model endpoint is gbench's endpoint
# relay (which adds the endpoint's key) and the simulator's Gemini API is gbench's Gemini relay
# (which adds the Gemini key), both on the network's gateway.
#
# Required env: TB_MODEL (served model id), TB_ENDPOINT (the endpoint relay's /v1 URL),
#               GEMINI_API_KEYS / GEMINI_API_KEY (the Gemini relay's per-run placeholder; a real
#               key is refused), GBENCH_GEMINI_RELAY_URL (the relay's origin; the simulator's
#               base URL is it + /v1beta/openai/, and GEMINI_OPENAI_BASE_URL, when set, must be
#               that URL), GBENCH_JUDGE_MODELS (the Gemini cascade gbench passes,
#               base.judge_cascade(); the simulator's models, tried in order).
# Optional:     TB_GROUPS (default all 6), TB_METHOD (DFS_woFilter_w2), TB_NUM_THREAD (4),
#               TB_QUERY_IDS_FILE (a JSON file {group: [query ids]}: run exactly those queries of
#               each group, in that order; gbench's selection for --shard / --eval-limit; unset
#               runs every query of each group),
#               TB_SIMULATOR_ROUNDS / TB_SIMULATOR_BACKOFF (the cascade's rounds and backoff),
#               TB_REAL_API_TIMEOUT_S (read timeout of the real-API proxy call, default 120),
#               TB_ENABLE_THINKING / TB_MAX_TOKENS / TB_REQUEST_TIMEOUT_S / TB_TEMPERATURE /
#               TB_REQUEST_LOG (the model request settings /stb/toolbench_request_policy.py
#               applies to every DFSDT request; reply and failed-request counts appended to
#               TB_REQUEST_LOG, default /out/model_requests.jsonl), TB_HOST_UID / TB_HOST_GID
#               (everything written to /out is handed to them at exit).
# Written to /out: candidate/<group>.json (the converted answers), model_requests.jsonl,
#               toolbench_server_calls.jsonl (one record per /virtual call, the patched server),
#               model_request_failures.jsonl (one record per DFSDT step with a model request that
#               failed at the endpoint: its group, query id and whether upstream's tries recovered
#               it; the patched model client), tool_observations.json (per query, the
#               harness-failure observations of its whole search tree,
#               /stb/toolbench_observations.py).
set -uo pipefail

finish() {
  if [ -n "${TB_HOST_UID:-}" ] && [ -n "${TB_HOST_GID:-}" ]; then
    chown -R "${TB_HOST_UID}:${TB_HOST_GID}" /out 2>/dev/null || true
  fi
}
trap finish EXIT

: "${TB_MODEL:?TB_MODEL (served model id) is required}"
: "${TB_ENDPOINT:?TB_ENDPOINT (the endpoint relay /v1 URL) is required}"
if [ -z "${GBENCH_JUDGE_MODELS:-}" ]; then
  echo "[toolbench] GBENCH_JUDGE_MODELS is not set (gbench passes its Gemini cascade, the simulator's models)" >&2
  exit 2
fi
SIMULATOR_KEYS="${GEMINI_API_KEYS:-${GEMINI_API_KEY:-}}"
if [ -z "$SIMULATOR_KEYS" ]; then
  echo "[toolbench] GEMINI_API_KEYS or GEMINI_API_KEY (the Gemini relay's placeholder) is required" >&2
  exit 2
fi
# Only the relay's placeholder is accepted (the model's tool calls run in this container).
if ! python3 - "$SIMULATOR_KEYS" <<'PYKEY'
import re, sys
keys = [k.strip() for k in re.split(r"[;,]", sys.argv[1]) if k.strip()]
sys.exit(0 if keys and all(re.fullmatch(r"gbench-gemini-relay-[0-9a-f]{48}", k) for k in keys) else 1)
PYKEY
then
  echo "[toolbench] the Gemini key is not a gbench Gemini relay placeholder; a real key is refused" >&2
  exit 7
fi
RELAY="${GBENCH_GEMINI_RELAY_URL:-}"
RELAY="${RELAY%/}"
if ! printf '%s' "$RELAY" | grep -Eq '^https?://[A-Za-z0-9.-]+:[0-9]+$'; then
  echo "[toolbench] GBENCH_GEMINI_RELAY_URL is not an http(s) origin (scheme://host:port)" >&2
  exit 7
fi
SIMULATOR_BASE="${RELAY}/v1beta/openai/"
if [ -n "${GEMINI_OPENAI_BASE_URL:-}" ] && [ "${GEMINI_OPENAI_BASE_URL%/}/" != "$SIMULATOR_BASE" ]; then
  echo "[toolbench] GEMINI_OPENAI_BASE_URL is not the Gemini relay's URL" >&2
  exit 7
fi

TB_GROUPS="${TB_GROUPS:-G1_instruction G1_category G1_tool G2_category G2_instruction G3_instruction}"
TB_METHOD="${TB_METHOD:-DFS_woFilter_w2}"
TB_NUM_THREAD="${TB_NUM_THREAD:-4}"
# Per-task DFS SEARCH budget (canonical StableToolBench default 200). This is NOT a task count -
# TB_QUERY_IDS_FILE (below) selects the test instances.
TB_MAX_QUERY_COUNT="${TB_MAX_QUERY_COUNT:-200}"
export TB_SIMULATOR_ROUNDS="${TB_SIMULATOR_ROUNDS:-3}"
export TB_SIMULATOR_BACKOFF="${TB_SIMULATOR_BACKOFF:-1.0}"
export TB_SERVER_LOG="/out/toolbench_server_calls.jsonl"
# The cached /virtual server listens inside the container (its own network namespace).
TB_SERVER_PORT="${TB_SERVER_PORT:-8080}"

# Cached server config: upstream's server/config.yml at the pinned commit, with the simulator on
# the Gemini API's OpenAI-compatible endpoint through gbench's Gemini relay (the gbench Gemini
# cascade instead of gpt-4-turbo; config.yml's `model` is only the cascade's first model, which the
# patched server does not read). As upstream: a call that is not cached goes to the ToolBench
# real-API proxy first (toolbench_url) with the README's empty toolbench key, which the proxy
# answers as unauthorized, so the simulator answers it; is_save true caches the new response for
# the rest of the run.
cat > /stb/server/config.yml <<EOF
api_key: "${SIMULATOR_KEYS}"
api_base: ${SIMULATOR_BASE}
model: ${GBENCH_JUDGE_MODELS%%,*}
temperature: 0
toolbench_url: http://8.130.32.149:8080/rapidapi
tools_folder: "./tools"
cache_folder: "./tool_response_cache"
is_save: true
port: ${TB_SERVER_PORT}
log_file: "./server.log"
EOF
# The server reads the placeholder from config.yml; nothing started from here on (the server, the
# DFSDT loop that talks to the model under test, the converter) holds it in its environment.
unset GEMINI_API_KEY GEMINI_API_KEYS SIMULATOR_KEYS GEMINI_OPENAI_BASE_URL GBENCH_GEMINI_RELAY_URL

mkdir -p /out
: > "$TB_SERVER_LOG"
echo "[toolbench] starting cached /virtual server on port ${TB_SERVER_PORT}..."
( cd /stb/server && python main.py >/stb/server/server.stdout 2>&1 & )
up=0
for i in $(seq 1 90); do
  # any HTTP reply (even 404) means it is up; curl prints 000 while nothing listens
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${TB_SERVER_PORT}/" 2>/dev/null)
  if [ -n "$code" ] && [ "$code" != "000" ]; then up=1; break; fi
  sleep 2
done
if [ "$up" != 1 ]; then
  echo "[toolbench] the cached /virtual server did not answer on port ${TB_SERVER_PORT} within 180 s:" >&2
  # (main.py prints its config at start-up, the placeholder included: that line is left out)
  grep -v "api_key" /stb/server/server.stdout | tail -20 >&2
  exit 3
fi
echo "[toolbench] server probe done (http=${code})"

export TB_REQUEST_LOG="${TB_REQUEST_LOG:-/out/model_requests.jsonl}"
# One record per DFSDT step with a model request that failed at the endpoint (created empty, so
# its presence shows the patched model client ran; docker/toolbench_patch_llm.py, patch 4).
export TB_MODEL_FAILURE_LOG="/out/model_request_failures.jsonl"
: > "$TB_MODEL_FAILURE_LOG"
export OPENAI_API_BASE="$TB_ENDPOINT"
export OPENAI_BASE_URL="$TB_ENDPOINT"
export OPENAI_API_KEY="EMPTY"
export SERVICE_URL="http://localhost:${TB_SERVER_PORT}/virtual"
# /stb for `toolbench.*`; the inference dir for its repo-local `Tree` package (upstream relies on
# Python auto-adding the script's dir - make it explicit so imports resolve either way).
export PYTHONPATH=/stb:/stb/toolbench/inference

mkdir -p /work/answers /work/converted
rc=0
for g in $TB_GROUPS; do
  qfile="/stb/solvable_queries/test_instruction/${g}.json"
  if [ ! -f "$qfile" ]; then echo "[toolbench] MISSING query file: $qfile" >&2; rc=1; continue; fi
  # TB_QUERY_IDS_FILE selects the test instances of each group by query id (gbench computes the
  # selection on the host: the shard, then a seeded draw for --eval-limit). --max_query_count is
  # the per-task DFS SEARCH budget, NOT a task count, so it cannot select instances; filtering the
  # JSON list is the only way. Unset -> run the full canonical group.
  runfile="$qfile"; nsel="full group"
  if [ -n "${TB_QUERY_IDS_FILE:-}" ]; then
    runfile="/work/queries_${g}.json"
    if ! nsel=$(python - "$qfile" "$runfile" "$TB_QUERY_IDS_FILE" "$g" <<'PYSELECT'
import json, sys
src, dst, sel, group = sys.argv[1:5]
want = [str(q) for q in (json.load(open(sel)).get(group) or [])]
queries = {str(q["query_id"]): q for q in json.load(open(src))}
missing = [q for q in want if q not in queries]
if missing:
    sys.stderr.write(f"[toolbench] {group}: selected query ids not in {src}: {missing[:5]}\n")
    sys.exit(4)
json.dump([queries[q] for q in want], open(dst, "w"))
print(f"{len(want)} selected queries")
PYSELECT
    ); then rc=4; continue; fi
  fi
  echo "[toolbench] inference group ${g} (${nsel})..."
  # The pinned StableToolBench qa_pipeline has NO --test_ids arg (passing it is an argparse-fatal
  # "unrecognized arguments" -> rc=2). It also reads the served endpoint from --base_url (rapidapi*.py
  # has its OPENAI_API_BASE env read commented out and uses self.args.base_url, default
  # api.openai.com) -- so the endpoint MUST be passed as a flag, else every model call silently goes
  # to OpenAI instead of the gbench-served /v1. TB_GROUP tags the group's /virtual calls for the
  # server's record (docker/toolbench_patch_client.py).
  TB_GROUP="$g" python /stb/toolbench/inference/qa_pipeline_multithread.py \
      --backbone_model chatgpt_function --chatgpt_model "$TB_MODEL" --openai_key EMPTY \
      --base_url "$TB_ENDPOINT" \
      --max_observation_length 1024 --method "$TB_METHOD" \
      --input_query_file "$runfile" --output_answer_file "/work/answers/${g}" \
      --tool_root_dir /stb/server/tools \
      --single_chain_max_step 50 --max_query_count "$TB_MAX_QUERY_COUNT" --num_thread "$TB_NUM_THREAD" \
      || rc=$?
done

echo "[toolbench] counting harness-failure observations in the answer trees..."
# shellcheck disable=SC2086
python /stb/toolbench_observations.py /work/answers "$TB_METHOD" /out/tool_observations.json $TB_GROUPS \
  || rc=$?

echo "[toolbench] converting answer trees..."
# The pinned convert_to_answer_format.py takes FLAGS (--answer_dir/--method/--output), converts ONE
# answer dir at a time, and only reads files whose name contains the method. Inference wrote per-group
# dirs of {qid}_{method}.json, so convert per group into /work/converted/<group>.json. (Passing only
# --method, as before, is argparse-fatal: --answer_dir is required -> rc=2.)
for g in $TB_GROUPS; do
  [ -d "/work/answers/${g}" ] || continue
  python /stb/toolbench/tooleval/convert_to_answer_format.py \
      --answer_dir "/work/answers/${g}" --method "$TB_METHOD" \
      --output "/work/converted/${g}.json" || rc=$?
done

mkdir -p /out/candidate
cp -r /work/converted/. /out/candidate/ 2>/dev/null || true
echo "[toolbench] done (rc=$rc)"
exit "$rc"
