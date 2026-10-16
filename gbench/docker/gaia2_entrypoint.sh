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
# Entrypoint for the gbench-gaia2 image. Starts the Gemini judge proxy (gaia2_cascade_judge.py:
# ARE's judge calls are answered by the gbench Gemini cascade instead of ARE's default
# Llama-3.3-70B), waits until it answers, then runs the launcher, which drives `are-benchmark`. No
# docker-out-of-docker here (ARE is a pure in-process simulator).
#
# The judge is always this proxy: without a Gemini key, or when the proxy does not start, there is
# no run (the launcher never starts, so no summary is written and the gbench runner hard-errors).
# The container holds no real key: GEMINI_API_KEY / GEMINI_API_KEYS carry a placeholder of gbench's
# Gemini relay, which the proxy presents to the relay (GBENCH_GEMINI_RELAY_URL); the launcher, and
# with it ARE and the agent, runs without the placeholder and the relay's address.
#
# When the run ends, everything written to $GAIA2_WORKDIR is handed to GAIA2_HOST_UID:GAIA2_HOST_GID
# (the gbench runner's user), so the runner can keep and remove the workdir.
set -uo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export GAIA2_WORKDIR="${GAIA2_WORKDIR:-/out}"
mkdir -p "$GAIA2_WORKDIR"

JUDGE_PID=""
finish() {
  if [ -n "$JUDGE_PID" ]; then kill "$JUDGE_PID" 2>/dev/null || true; fi
  if [ -n "${GAIA2_HOST_UID:-}" ] && [ -n "${GAIA2_HOST_GID:-}" ]; then
    chown -R "${GAIA2_HOST_UID}:${GAIA2_HOST_GID}" "$GAIA2_WORKDIR" 2>/dev/null || true
  fi
}
trap finish EXIT

if [ -z "${GEMINI_API_KEYS:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "gaia2: GEMINI_API_KEY is unset - ARE's load-bearing LLM judge runs on the Gemini cascade." >&2
  exit 5
fi

JUDGE_PORT="${GAIA2_JUDGE_PORT:-18790}"
GAIA2_JUDGE_PORT="$JUDGE_PORT" WILDCLAW_WORKDIR="$GAIA2_WORKDIR" \
    nohup python3 "$ADAPTER_DIR/gaia2_cascade_judge.py" \
    > "$GAIA2_WORKDIR/cascade_judge.log" 2>&1 &
JUDGE_PID=$!

# Wait (up to 30 s) until the proxy answers its health endpoint.
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
    break
  fi
  if python3 -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:${JUDGE_PORT}/health', timeout=2); sys.exit(0)" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if [ "$ready" != 1 ]; then
  echo "gaia2: the Gemini judge proxy did not start; its log follows." >&2
  tail -n 20 "$GAIA2_WORKDIR/cascade_judge.log" >&2
  exit 6
fi

# Not exec: the EXIT trap must stop the proxy and hand the workdir over when the launcher returns.
env -u GEMINI_API_KEYS -u GEMINI_API_KEY -u GBENCH_GEMINI_RELAY_URL -u GEMINI_OPENAI_BASE_URL \
    python3 "$ADAPTER_DIR/gaia2_run.py"
exit $?
