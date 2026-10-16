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
# Container entrypoint for MobileGym and MobileGym++ evaluations.
#
# Drops outbound traffic to cloud instance-metadata endpoints before starting
# the simulator and Python harness (see gbench/runners/eval_suites/container_guard.py).
set -eu

for cidr in 169.254.169.254/32 169.254.170.2/32 100.100.100.200/32 192.0.0.192/32; do
    iptables -I OUTPUT -d "$cidr" -j REJECT 2>/dev/null || true
done
ip6tables -I OUTPUT -d fd00:ec2::254/128 -j REJECT 2>/dev/null || true

exec python /opt/gbench/mobilegym_run.py "$@"

