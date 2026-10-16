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
# Entrypoint for the gbench-wildclawbench orchestrator image. Starts the Gemini proxy
# (wildclawbench_cascade_judge.py: the tasks' verbatim OpenAI-SDK judge calls, and the OpenClaw
# agent's web_search, both answered through the Gemini API, plus the relay of the agent's model
# calls), waits until it answers, then runs the launcher, which drives the pinned InternLM harness
# docker-out-of-docker.
#
# The Gemini keys (GEMINI_API_KEYS / GEMINI_API_KEY) and the model endpoint's key
# (GBENCH_MODEL_API_KEY) are handed to the proxy only: the launcher, and with it run_batch and every
# `docker run` / `docker exec` it issues for a task container, runs without them.
#
# The proxy listens on WILDCLAW_JUDGE_HOST (default 127.0.0.1): the runner passes the address the
# task containers reach it at (127.0.0.1 with the host network, the docker bridge gateway
# otherwise), so it is never open on every interface.
#
# DooD identity-mount contract (skillsbench pitfall): run_batch bind-mounts each task's
# <workspace>/exec into the sibling task container, and `-v <path>` from inside a container resolves
# on the HOST. So WILDCLAW_HARNESS_DIR (with its workspace/ data) must be an IDENTITY mount (same
# path on host + orchestrator); the gbench runner arranges that. TMPDIR is pointed at the workdir.
set -uo pipefail

HARNESS_DIR="${WILDCLAW_HARNESS_DIR:-/app/WildClawBench}"
ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export WILDCLAW_WORKDIR="${WILDCLAW_WORKDIR:-/out}"
export TMPDIR="${WILDCLAW_WORKDIR}"
mkdir -p "$WILDCLAW_WORKDIR"

# The Docker CLI must reach the mounted host daemon (docker-out-of-docker).
if ! docker version >/dev/null 2>&1; then
  echo "wildclawbench: cannot reach the Docker daemon - mount /var/run/docker.sock into the container." >&2
  exit 3
fi

# The harness root must be present (identity-mounted checkout with its workspace/ data).
if [ ! -f "$HARNESS_DIR/eval/run_batch.py" ]; then
  echo "wildclawbench: harness not found at $HARNESS_DIR (identity-mount the provisioned WildClawBench checkout)." >&2
  exit 4
fi

# Every judge call and every agent web search goes through the Gemini API; without a key there is
# no run.
if [ -z "${GEMINI_API_KEYS:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "wildclawbench: GEMINI_API_KEYS / GEMINI_API_KEY is unset - the judge and the OpenClaw web_search both need it." >&2
  exit 5
fi

# The placeholder key each task's grade() sends to the proxy (the launcher sets it as
# OPENROUTER_API_KEY in the grading exec): the proxy tells the grader's judge calls from the agent's
# by it. A new random value for every run, given to the proxy and the launcher; not a credential.
WILDCLAW_GRADER_KEY="gbench-grader-$(python3 -c 'import secrets; print(secrets.token_hex(12))')"
export WILDCLAW_GRADER_KEY

# Start the Gemini proxy (OpenAI-compatible judge + Brave-shaped web search + model relay) in the
# background.
JUDGE_PORT="${WILDCLAW_JUDGE_PORT:-18795}"
JUDGE_HOST="${WILDCLAW_JUDGE_HOST:-127.0.0.1}"
WILDCLAW_JUDGE_PORT="$JUDGE_PORT" WILDCLAW_JUDGE_HOST="$JUDGE_HOST" \
    nohup python3 "$ADAPTER_DIR/wildclawbench_cascade_judge.py" \
    > "$WILDCLAW_WORKDIR/cascade_judge.log" 2>&1 &
JUDGE_PID=$!
# On exit: stop the proxy, then hand the workdir (the task containers write their output there as
# root) to the user the gbench runner runs as (WILDCLAW_HOST_OWNER, uid:gid), so the runner can
# keep the per-task artifacts and remove the workdir.
cleanup() {
  kill "$JUDGE_PID" 2>/dev/null || true
  if [ -n "${WILDCLAW_HOST_OWNER:-}" ]; then
    chown -R "$WILDCLAW_HOST_OWNER" "$WILDCLAW_WORKDIR" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait (up to 30 s) until the proxy answers its health endpoint on its listen address.
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
    break
  fi
  if python3 -c "import sys, urllib.request; urllib.request.urlopen('http://${JUDGE_HOST}:${JUDGE_PORT}/health', timeout=2); sys.exit(0)" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if [ "$ready" != 1 ]; then
  echo "wildclawbench: the Gemini proxy did not start; its log follows." >&2
  tail -n 20 "$WILDCLAW_WORKDIR/cascade_judge.log" >&2
  exit 6
fi

# Not exec: the EXIT trap must stop the proxy when the launcher returns.
env -u GEMINI_API_KEYS -u GEMINI_API_KEY -u GBENCH_MODEL_API_KEY \
    python3 "$ADAPTER_DIR/wildclawbench_run.py"
exit $?
