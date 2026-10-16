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
"""Count, per query, the tool observations of StableToolBench's raw answer trees that are
failures of the harness rather than answers of a tool.

Usage (the gbench toolbench entrypoint, after the DFSDT inference):
    python toolbench_observations.py <answers root> <method> <output json> <group> [<group> ...]

<answers root>/<group>/<query_id>_<method>.json are the files qa_pipeline_multithread.py writes;
every node of the whole search tree (`tree`), not only the chain the converter keeps, is read, so
a failure the model met on a branch it later left is counted too. The observations counted are
the ones upstream's client (rapidapi_multithread.py `_step`) writes for a /virtual call that did
not get a tool answer:
  * `server_error_observations`: '{"error": "request invalid, data error. status_code=<code>"...'
    (the server answered HTTP 500 or another non-200) or '...data error"' (a body that is not
    JSON): the server failed, for example the real-API proxy or the simulator cascade raised;
  * `simulator_failed_observations`: upstream's '"Failed to generate fake response"' reply (the
    simulator gave no valid JSON in three tries), also when the server replays it from its cache;
  * `timeout_observations`: '{"error": "Timeout error..."' (a client timeout; the /virtual call
    itself has none, upstream's other service URLs do).
The output is {"groups": {group: {query_id: {"tool_calls", "server_error_observations",
"simulator_failed_observations", "timeout_observations"}}}}.
"""

import json
import os
import sys

SERVER_ERROR_MARKER = "request invalid, data error"
SIMULATOR_FAILED_MARKER = "Failed to generate fake response"
TIMEOUT_MARKER = '"error": "Timeout error...'
KINDS = ("server_error_observations", "simulator_failed_observations", "timeout_observations")


def _observations(node, out):
    """Every `observation` string of the tree under `node` (dicts and lists, iteratively deep)."""
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            obs = cur.get("observation")
            if isinstance(obs, str) and obs:
                out.append(obs)
            for key, value in cur.items():
                if key != "observation" and isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(cur, list):
            stack.extend(v for v in cur if isinstance(v, (dict, list)))
    return out


def count_answer(data):
    """The counts of one raw answer file (its parsed JSON)."""
    root = data.get("tree") if isinstance(data, dict) else None
    observations = _observations(root if root is not None else data, [])
    counts = {"tool_calls": len(observations)}
    counts["server_error_observations"] = sum(SERVER_ERROR_MARKER in o for o in observations)
    counts["simulator_failed_observations"] = sum(SIMULATOR_FAILED_MARKER in o
                                                  for o in observations)
    counts["timeout_observations"] = sum(TIMEOUT_MARKER in o for o in observations)
    return counts


def scan(answers_root, method, groups):
    """{group: {query_id: counts}} over <answers_root>/<group>/<query_id>_<method>.json."""
    suffix = f"_{method}.json"
    out = {}
    for group in groups:
        per = out.setdefault(group, {})
        folder = os.path.join(answers_root, group)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(suffix):
                continue
            qid = name[:-len(suffix)]
            try:
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                per[qid] = {"tool_calls": 0, "unreadable": True,
                            **{k: 0 for k in KINDS}}
                continue
            per[qid] = count_answer(data)
    return out


def main(argv):
    if len(argv) < 5:
        sys.stderr.write(__doc__)
        return 2
    answers_root, method, output, groups = argv[1], argv[2], argv[3], argv[4:]
    result = {"groups": scan(answers_root, method, groups)}
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(tmp, output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
