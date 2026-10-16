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
# Entrypoint for the gbench-ui-control-osworld image. Starts a virtual display (OSWorld's agent deps
# pyautogui/pynput import an X display at module load), then runs the launcher, which drives the
# OSWorld harness docker-out-of-docker (it spawns sibling QEMU-VM containers on the host daemon).
set -uo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export OSWORLD_DIR="${OSWORLD_DIR:-/app/OSWorld}"
export OSWORLD_WORKDIR="${OSWORLD_WORKDIR:-/out}"
mkdir -p "$OSWORLD_WORKDIR"

# The docker SDK must reach the mounted host daemon (docker-out-of-docker).
if ! python3 -c "import docker; docker.from_env().ping()" >/dev/null 2>&1; then
  echo "ui_control_osworld: cannot reach the Docker daemon via the SDK - mount /var/run/docker.sock." >&2
  exit 3
fi

# Virtual display for pyautogui/pynput imports.
Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 &
XVFB_PID=$!
export DISPLAY=:99
sleep 1
trap 'kill "$XVFB_PID" 2>/dev/null || true' EXIT

if [ ! -e /dev/kvm ]; then
  echo "ui_control_osworld: WARNING /dev/kvm not present in this container - the sibling QEMU VM will" >&2
  echo "  fall back to TCG software emulation and almost certainly miss the 300s VM-ready timeout." >&2
fi

exec python3 "$ADAPTER_DIR/ui_control_osworld_run.py"
