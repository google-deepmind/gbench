#!/bin/sh
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
#
# Entrypoint of gbench's quality container (docker/quality.Dockerfile). gbench starts it as root
# with only CAP_SETUID and CAP_SETGID (docker run --cap-drop ALL --cap-add SETUID --cap-add SETGID
# --security-opt no-new-privileges), which it needs to start its two processes as their users:
#
#   1. It reads the per-run relay token from the first line of standard input (never from its
#      arguments, its environment or a file), then points standard input at /dev/null.
#   2. It starts the relay forwarder (quality_relay_forwarder.py) as user gbrelay, handing it the
#      token on a pipe, and waits until the forwarder listens.
#   3. It runs its arguments (the QA suite command) in /opt/gemmaclaw as user qa. A process whose
#      user changes from root to another drops every capability, and no-new-privileges keeps it
#      from gaining any back, so the suite and every tool the model under test runs hold none.
#
# gbench reads STATUS_DIR afterwards (entrypoint.json, forwarder-errors.jsonl). It is owned by
# root and group gbrelay, mode 0770: user qa can neither read nor write it.
set -u
STATUS_DIR=/gbench-status

fail() {
    printf '{"error": "%s"}\n' "$1" > "$STATUS_DIR/entrypoint.json"
    echo "gbench quality container: $1" >&2
    exit "$2"
}

forwarder_alive() {
    [ -r "/proc/$fwd/status" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$fwd/status"
}

[ "$(id -u)" = 0 ] || fail "the entrypoint must start as root; it starts the suite as user qa" 96
[ "$#" -gt 0 ] || fail "no command to run" 95

token=""
IFS= read -r token || true
exec 0</dev/null
[ -n "$token" ] || fail "no relay token on standard input" 97

printf '%s\n' "$token" | setpriv --reuid=gbrelay --regid=gbrelay --clear-groups --inh-caps=-all \
    python3 /opt/gbench/quality_relay_forwarder.py &
fwd=$!
token=""
unset token

n=0
while [ ! -e "$STATUS_DIR/forwarder-ready" ]; do
    n=$((n + 1))
    if [ "$n" -gt 300 ] || ! forwarder_alive; then
        fail "the relay forwarder did not start" 98
    fi
    sleep 0.1
done

cd /opt/gemmaclaw || fail "no /opt/gemmaclaw" 94
setpriv --reuid=qa --regid=qa --clear-groups --inh-caps=-all -- "$@"
rc=$?
alive=false
if forwarder_alive; then
    alive=true
fi
printf '{"command_exit": %d, "forwarder_alive_at_end": %s}\n' "$rc" "$alive" \
    > "$STATUS_DIR/entrypoint.json"
exit "$rc"
