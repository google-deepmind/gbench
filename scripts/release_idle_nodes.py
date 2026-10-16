#!/usr/bin/env python3
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

import argparse
import json
import subprocess
import sys

parser = argparse.ArgumentParser(description="Release idle GKE nodes from MIGs.")
parser.add_argument("--project", required=True, help="GCP project ID")
parser.add_argument("--node-pool-label", required=True, help="GKE nodepool label value (e.g. a2-spot-pool-multi)")
parser.add_argument("--mig-filter", required=True, help="Substring or regex filter for managed instance groups")
args = parser.parse_args()

project_id = args.project
node_pool_label = args.node_pool_label
mig_filter = args.mig_filter

print("Finding active vLLM pods and nodes...")
pods_raw = subprocess.check_output(["kubectl", "get", "pods", "-l", "app=vllm", "-o", "json"])
pods = json.loads(pods_raw)["items"]
busy_nodes = set(p["spec"].get("nodeName") for p in pods if p["spec"].get("nodeName"))

nodes_raw = subprocess.check_output([
    "kubectl", "get", "nodes",
    "-l", f"cloud.google.com/gke-nodepool={node_pool_label}",
    "-o", "json"
])
all_nodes = json.loads(nodes_raw)["items"]

idle_by_zone = {}
for n in all_nodes:
    name = n["metadata"]["name"]
    zone = n["metadata"]["labels"].get("topology.kubernetes.io/zone", "")
    if name not in busy_nodes:
        idle_by_zone.setdefault(zone, []).append(name)

total_idle = sum(len(v) for v in idle_by_zone.values())
print(f"Total nodes: {len(all_nodes)}")
print(f"Busy nodes with vLLM: {len(busy_nodes)}")
print(f"Idle nodes to delete: {total_idle}")

if total_idle == 0:
    print("No idle nodes found. Done.")
    sys.exit(0)

migs_raw = subprocess.check_output([
    "gcloud", "compute", "instance-groups", "managed", "list",
    f"--filter=name ~ {mig_filter}",
    f"--project={project_id}",
    "--format=json"
])
migs = json.loads(migs_raw)
mig_by_zone = {}
for m in migs:
    z = m["zone"].split("/")[-1]
    mig_by_zone[z] = m["name"]

for zone, instances in idle_by_zone.items():
    if not instances:
        continue
    mig_name = mig_by_zone.get(zone)
    if not mig_name:
        print(f"Warning: No MIG found for zone {zone}")
        continue
    print(f"Releasing {len(instances)} idle instances in {zone} from MIG {mig_name}...")
    cmd = [
        "gcloud", "compute", "instance-groups", "managed", "delete-instances",
        mig_name,
        "--instances=" + ",".join(instances),
        f"--zone={zone}",
        f"--project={project_id}",
        "--quiet"
    ]
    subprocess.run(cmd, check=True)
    print(f"  Released {len(instances)} instances in {zone}.")

print("All idle nodes deleted successfully!")
