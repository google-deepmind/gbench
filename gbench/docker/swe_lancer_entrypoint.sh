#!/bin/bash
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
# Entrypoint of the gbench-swe-lancer-harness image (gbench/docker/swe_lancer_harness.Dockerfile).
#
# SWE-Lancer's harness (alcatraz, disable_internet=True) cuts each task container off the internet
# with iptables rules in the Docker host's DOCKER-USER chain. The container runs on the host
# network with NET_ADMIN, so the iptables here edit the host's rules; this picks the iptables
# backend (nft or legacy) whose tables hold Docker's DOCKER-USER chain.
#
#   swe_lancer_entrypoint.sh --probe       print the backend that sees DOCKER-USER (exit 1: none)
#   swe_lancer_entrypoint.sh CMD [ARG...]  run CMD with that backend selected, then give the files
#                                          under $GBENCH_WORKDIR back to $GBENCH_UID:$GBENCH_GID
set -u

select_backend() {
    for backend in iptables-nft iptables-legacy; do
        if "/usr/sbin/${backend}" -S DOCKER-USER >/dev/null 2>&1; then
            update-alternatives --set iptables "/usr/sbin/${backend}" >/dev/null 2>&1 || return 1
            echo "${backend}"
            return 0
        fi
    done
    return 1
}

if [ "${1:-}" = "--probe" ]; then
    select_backend || { echo "no iptables backend sees the DOCKER-USER chain" >&2; exit 1; }
    exit 0
fi

select_backend >/dev/null || echo "gbench: no iptables backend sees the DOCKER-USER chain" >&2
"$@"
rc=$?
if [ -n "${GBENCH_WORKDIR:-}" ] && [ -n "${GBENCH_UID:-}" ]; then
    chown -R "${GBENCH_UID}:${GBENCH_GID:-${GBENCH_UID}}" "${GBENCH_WORKDIR}" 2>/dev/null || true
fi
exit "${rc}"
