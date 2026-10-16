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
"""gbench build-time patch of StableToolBench's DFSDT client (rapidapi_multithread.py).

Every tool call the DFSDT loop sends to the cached /virtual server carries two more request
headers: X-Gbench-Group (the test group being run, TB_GROUP, which the entrypoint sets per
group) and X-Gbench-Query-Id (the query's query_id). The patched server
(docker/toolbench_patch_simulator.py) only writes them into its record of the call, so gbench
can tell which query met a server error, a failed simulation, a cached or a simulated answer.
The server never forwards them (it builds its own request to the real-API proxy), and nothing
else of the call changes: the payload, the toolbench_key header, the timeout and the handling of
the reply are upstream's. The query's ChatGPTFunction (method_converter) gets the same query id,
which the patched model client (docker/toolbench_patch_llm.py, patch 4) writes into its record
of a model request that failed at the endpoint.

Each replacement is asserted to match exactly, so the build FAILS if upstream drifts.
"""

import re

PATH = "/stb/toolbench/inference/Downstream_tasks/rapidapi_multithread.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

INIT_OLD = '        self.input_description = query_json["query"]\n'
INIT_NEW = ('        self.input_description = query_json["query"]\n'
            '        # gbench: the query id the /virtual calls of this query are tagged with\n'
            '        self._gbench_query_id = query_json.get("query_id")\n')
assert src.count(INIT_OLD) == 1, "toolbench client patch: __init__ anchor not found"
src = src.replace(INIT_OLD, INIT_NEW, 1)

CALL_OLD = '                        headers = {"toolbench_key": self.toolbench_key}\n'
CALL_NEW = ('                        headers = {"toolbench_key": self.toolbench_key}\n'
            '                        # gbench: tag the call for the server\'s record (never forwarded)\n'
            '                        headers["X-Gbench-Group"] = os.environ.get("TB_GROUP", "")\n'
            '                        headers["X-Gbench-Query-Id"] = str(\n'
            '                            getattr(self, "_gbench_query_id", "") or "")\n')
assert src.count(CALL_OLD) == 1, "toolbench client patch: /virtual headers anchor not found"
src = src.replace(CALL_OLD, CALL_NEW, 1)
assert re.search(r"^import os$", src, re.M), \
    "toolbench client patch: the module no longer imports os"

# The query's model: its ChatGPTFunction carries the query id too, so the model requests that
# failed at the endpoint are recorded with the query they belong to
# (docker/toolbench_patch_llm.py, patch 4). Nothing of the request changes.
LLM_OLD = ("            llm_forward = ChatGPTFunction(model=self.args.chatgpt_model, "
           "openai_key=openai_key, base_url=self.args.base_url)\n")
LLM_NEW = LLM_OLD + ("            # gbench: the query id this query's model requests are recorded with\n"
                     "            llm_forward._gbench_query_id = getattr(env, \"_gbench_query_id\", "
                     "None)\n")
assert src.count(LLM_OLD) == 1, "toolbench client patch: the ChatGPTFunction anchor not found"
src = src.replace(LLM_OLD, LLM_NEW, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("toolbench_patch_client: patched", PATH)
