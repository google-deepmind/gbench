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
"""gbench build-time patch for StableToolBench's convert_to_answer_format.py (ToolEval).

`process_valid_data` accesses `message['content']` (and tool/function response `content`) with a bare
subscript. An assistant turn is valid in the OpenAI schema with `content: null` / no content key
(e.g. `tool_calls` present, or `tool_calls: null` with an empty completion). When the served model
emits such a turn, convert dies with `KeyError: 'content'` and the whole judge run fails rc=1 - and
it is MODEL-OUTPUT dependent, so it strikes intermittently (one run converts fine, the next crashes).
Degrade every unguarded content read to `''` (an empty node), which is the correct meaning of an
empty turn and never changes a well-formed conversation.

Each replacement is asserted to match EXACTLY so the build FAILS on upstream drift rather than
silently shipping an unpatched image. Mirrors docker/toolbench_patch_llm.py.
"""

PATH = "/stb/toolbench/tooleval/convert_to_answer_format.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

REPLACEMENTS = [
    # plain assistant turn with no 'content' key (tool_calls:null + empty completion) -> KeyError.
    (
        "                                        message=message['content'])",
        "                                        message=(message.get('content') or ''))",
    ),
    # a malformed tool turn may lack 'content'.
    (
        "                                response = message2['content']",
        "                                response = (message2.get('content') or '')",
    ),
    # legacy function_call path: the following turn may lack 'content'.
    (
        "                    'response':conversation[index+1]['content'] if message['function_call']['name']!='Finish' else ''",
        "                    'response':(conversation[index+1].get('content') or '') if message['function_call']['name']!='Finish' else ''",
    ),
]

for old, new in REPLACEMENTS:
    assert old in src, f"toolbench convert patch: line not found (upstream drifted?): {old.strip()[:70]!r}"
    src = src.replace(old, new, 1)

# Guard: no bare message['content'] / message2['content'] subscript survives.
assert "message['content']" not in src and "message2['content']" not in src, \
    "toolbench convert patch: a bare ['content'] subscript survived"

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("toolbench_patch_convert: patched", PATH)
