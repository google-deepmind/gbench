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
# Entrypoint for the gbench-skillsbench image. Runs the launcher, which drives BenchFlow 0.6.3
# docker-out-of-docker. CRITICAL: BenchFlow bind-mounts its working dirs into sibling task
# containers, so SKILLSBENCH_WORKDIR must be an IDENTITY mount (same path on host + orchestrator)
# and TMPDIR must point there; the harness sets this up. The model-under-test is the gbench /v1
# endpoint (must be reachable FROM the sibling task containers); the deterministic verifier scores.
set -uo pipefail

SKILLSBENCH_DIR="${SKILLSBENCH_DIR:-/app/skillsbench}"
ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export SKILLSBENCH_WORKDIR="${SKILLSBENCH_WORKDIR:-/out}"
export TMPDIR="${SKILLSBENCH_WORKDIR}"
mkdir -p "$SKILLSBENCH_WORKDIR"
cd "$SKILLSBENCH_DIR"

AGENT="${SKILLSBENCH_AGENT:-deepagents}"
if [ "$AGENT" != "oracle" ]; then
  : "${GBENCH_MODEL_BASE_URL:?GBENCH_MODEL_BASE_URL (a /v1 reachable from task containers) is required}"
  : "${GBENCH_MODEL_NAME:?GBENCH_MODEL_NAME (served model name) is required}"
fi
export GBENCH_MODEL_API_KEY="${GBENCH_MODEL_API_KEY:-dummy}"

# The Docker CLI must reach the mounted host daemon (docker-out-of-docker).
if ! docker version >/dev/null 2>&1; then
  echo "skillsbench: cannot reach the Docker daemon - mount /var/run/docker.sock into the container." >&2
  exit 3
fi

exec python3 "$ADAPTER_DIR/skillsbench_run.py"
