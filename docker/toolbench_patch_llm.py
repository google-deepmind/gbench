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
"""gbench build-time patch for StableToolBench's chat_completion_request (chatgpt_function backbone).

Two upstream defects make the served-endpoint (non-GPT) path unusable, so the DFSDT loop never
produces an answer tree and the container dies rc=1 (patches 1 and 2); patch 3 applies the run's
model request settings, and patch 4 records the model requests that failed at the endpoint per
query:

  1. `if model.startswith("gpt"): ... else: raise NotImplementedError("Model not supported")` rejects
     any model whose id does not start with "gpt" (e.g. `google/gemma-4-26B-A4B-it`). gbench always
     targets an OpenAI-COMPATIBLE served /v1 endpoint (base_url is set), so route ANY model through
     the OpenAI client.
  2. `import pdb;  pdb.set_trace()` in the except handler drops a NON-INTERACTIVE container into pdb,
     which reads EOF -> BdbQuit -> the retry wrapper reports RetryError and the whole run dies.
     Remove it (keep the error-dict return).
  3. The request goes through gbench's request settings (/stb/toolbench_request_policy.py, a copy
     of docker/toolbench_request_policy.py): the chat template's thinking switch follows the
     run's mode (TB_ENABLE_THINKING), the operator's temperature is sent when set
     (TB_TEMPERATURE), and with --thinking the run's budget (TB_MAX_TOKENS) replaces the 1024
     max_tokens, cut to the room the context window leaves after the prompt, and the request
     carries a timeout sized to it (TB_REQUEST_TIMEOUT_S); without --thinking the request is
     upstream's. The counts of the replies and of the failed requests are appended to
     TB_REQUEST_LOG at exit.
  4. Each DFSDT step (ChatGPTFunction.parse, which asks up to 6 times) with a request that failed
     at the endpoint, or that the endpoint rejected for its content (HTTP 400/422), is appended to
     TB_MODEL_FAILURE_LOG with its test group, query id and how it ended, so gbench can leave out
     the queries whose search tree got an endpoint failure's error text as the model's reply. A
     400/422 is the model's outcome here, as gbench's quality runner and gaia2 count it (its
     request carried the model's own conversation; other gbench suites count a 400/422 as a
     harness error, docs/evals/env-knobs.md); such steps are recorded so they stay visible, and
     their queries are scored. The context-window rejections among them that the request
     settings left to upstream (their rejected_over_context) are also counted apart.

Each replacement is asserted to match EXACTLY, so the build FAILS if upstream drifts rather than
silently shipping an unpatched image. Mirrors docker/ojbench_patch_judger.py.
"""

PATH = "/stb/toolbench/inference/LLM/chatgpt_function_model.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()

# --- Patch 1: the model gate -> always use the OpenAI-compatible client. ---
GATE_OLD = (
    '        if model.startswith("gpt"):\n'
    "            client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)\n"
    "        else:\n"
    '            raise NotImplementedError("Model not supported")\n'
)
GATE_NEW = (
    "        # gbench: route ANY model through the OpenAI-compatible client (we always target a\n"
    '        # served /v1 endpoint via base_url); upstream hard-gated on model.startswith("gpt").\n'
    "        client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)\n"
)
assert GATE_OLD in src, "toolbench patch: model-gate block not found (upstream drifted?)"
src = src.replace(GATE_OLD, GATE_NEW, 1)

# --- Patch 2: remove the interactive pdb drop in the except handler. ---
PDB_OLD = '        import pdb;  pdb.set_trace()\n        return {"error": str(e), "total_tokens": 0}\n'
PDB_NEW = '        return {"error": str(e), "total_tokens": 0}\n'
assert PDB_OLD in src, "toolbench patch: pdb.set_trace block not found (upstream drifted?)"
src = src.replace(PDB_OLD, PDB_NEW, 1)

# Guard: no interactive pdb drop survives anywhere in the file.
assert "pdb.set_trace" not in src, "toolbench patch: a pdb.set_trace survived"

# --- Patch 3: the run's model request settings. ---
IMPORT_OLD = "import json\nimport traceback\n"
IMPORT_NEW = (
    "import json\nimport traceback\n\n"
    "# gbench: the run's model request settings (/stb/toolbench_request_policy.py): the thinking\n"
    "# switch, the operator's temperature and the --thinking budget; reply counts to TB_REQUEST_LOG.\n"
    "from toolbench_request_policy import RequestPolicy as _GbenchRequestPolicy\n"
    "_GBENCH_POLICY = _GbenchRequestPolicy(\n"
    "    \"TB_\", base_url=os.environ.get(\"TB_ENDPOINT\")).install_dump_at_exit()\n"
)
assert src.count(IMPORT_OLD) == 1, "toolbench patch: import block not found (upstream drifted?)"
src = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
CREATE_OLD = "        openai_response = client.chat.completions.create(**json_data)\n"
CREATE_NEW = ("        openai_response = _GBENCH_POLICY.call(client.chat.completions.create, "
              "json_data, api_key=key)\n")
assert src.count(CREATE_OLD) == 1, "toolbench patch: the create call not found (upstream drifted?)"
src = src.replace(CREATE_OLD, CREATE_NEW, 1)
REQUEST_OLD = '        "max_tokens": 1024,\n'
assert src.count(REQUEST_OLD) == 1, "toolbench patch: upstream's max_tokens 1024 not found"

# --- Patch 4: per query, the model requests that failed at the endpoint. ---
# Upstream's chat_completion_request turns a failed request into an error reply, and
# ChatGPTFunction.parse asks again up to TRY_TIME (6) times; when every try failed, the error text
# goes into the search tree as the model's reply. Each try's outcome is noted for the step being
# asked (a thread-local list: every query runs its DFSDT search in one thread). A try failed at
# the endpoint when RequestPolicy marked its error with ENDPOINT_FAILURE_ATTR and its HTTP status
# is not 400 or 422; a try with HTTP status 400 or 422 was rejected for its content (the model's
# outcome: the request carried the model's own conversation). A step with either is appended to
# TB_MODEL_FAILURE_LOG with the test group (TB_GROUP), the query id
# (docker/toolbench_patch_client.py sets it on the query's ChatGPTFunction), the counts, and how
# it ended: `unrecovered` (its last try failed at the endpoint), `rejected` (its last try was
# rejected), or neither (a later try got a reply). `content_rejections` counts every 400/422 try;
# of those, the context-window rejections RequestPolicy left to upstream's handling (its
# rejected_over_context: a 400 it did not mark with ENDPOINT_FAILURE_ATTR) are also counted
# apart, as `over_context_rejections`, present only when there were any. Nothing of the request,
# the reply or upstream's retries changes.
HELPERS_OLD = ("_GBENCH_POLICY = _GbenchRequestPolicy(\n"
               "    \"TB_\", base_url=os.environ.get(\"TB_ENDPOINT\")).install_dump_at_exit()\n")
HELPERS_NEW = HELPERS_OLD + (
    "\n"
    "# gbench: per DFSDT step (ChatGPTFunction.parse), the outcome of each request of the model\n"
    "# under test; a step with a request that failed at the endpoint, or that the endpoint\n"
    "# rejected for its content (HTTP 400/422: the model's outcome), goes to TB_MODEL_FAILURE_LOG\n"
    "# with its test group and query id (docker/toolbench_patch_llm.py, patch 4).\n"
    "import threading as _gbench_threading\n"
    "from toolbench_request_policy import ENDPOINT_FAILURE_ATTR as _GBENCH_ENDPOINT_FAILURE\n"
    "_GBENCH_TLS = _gbench_threading.local()\n"
    "_GBENCH_LOG_LOCK = _gbench_threading.Lock()\n"
    "_GBENCH_CONTENT_REJECTION_STATUSES = (400, 422)\n"
    "\n"
    "\n"
    "def _gbench_note_try(error):\n"
    "    \"\"\"Note one try of the step being asked: a reply (error None) or the request's error.\"\"\"\n"
    "    tries = getattr(_GBENCH_TLS, \"tries\", None)\n"
    "    if tries is None:\n"
    "        return\n"
    "    if error is None:\n"
    "        tries.append({\"ok\": True})\n"
    "        return\n"
    "    status = getattr(error, \"status_code\", None)\n"
    "    status = status if isinstance(status, int) and not isinstance(status, bool) else None\n"
    "    rejected = status in _GBENCH_CONTENT_REJECTION_STATUSES\n"
    "    marked = bool(getattr(error, _GBENCH_ENDPOINT_FAILURE, False))\n"
    "    tries.append({\"ok\": False, \"error\": type(error).__name__, \"status\": status,\n"
    "                  \"rejected\": rejected,\n"
    "                  \"over_context\": rejected and not marked,\n"
    "                  \"endpoint_failure\": not rejected and marked})\n"
    "\n"
    "\n"
    "def _gbench_record_step(llm, error_code, tries):\n"
    "    \"\"\"Append the step to TB_MODEL_FAILURE_LOG when a request of it failed at the endpoint\n"
    "    or was rejected for its content (HTTP 400/422): unrecovered when the step ended on an\n"
    "    endpoint failure (upstream passed the error text on as the model's reply), rejected when\n"
    "    it ended on a content rejection (the model's outcome), neither when a later try got a\n"
    "    reply.\"\"\"\n"
    "    failed = [t for t in tries if not t.get(\"ok\") and t.get(\"endpoint_failure\")]\n"
    "    rejected = [t for t in tries if not t.get(\"ok\") and t.get(\"rejected\")]\n"
    "    path = os.environ.get(\"TB_MODEL_FAILURE_LOG\")\n"
    "    if not (failed or rejected) or not path:\n"
    "        return\n"
    "    last = tries[-1]\n"
    "    ended_on_error = bool(error_code != 0 and not last.get(\"ok\"))\n"
    "    rec = {\"group\": os.environ.get(\"TB_GROUP\", \"\"),\n"
    "           \"query_id\": str(getattr(llm, \"_gbench_query_id\", \"\") or \"\"),\n"
    "           \"tries\": len(tries), \"endpoint_failures\": len(failed),\n"
    "           \"content_rejections\": len(rejected),\n"
    "           \"unrecovered\": bool(ended_on_error and last.get(\"endpoint_failure\")),\n"
    "           \"rejected\": bool(ended_on_error and last.get(\"rejected\")),\n"
    "           \"errors\": sorted({t[\"error\"] + (f\" {t['status']}\" if t.get(\"status\") else \"\")\n"
    "                              for t in failed + rejected})}\n"
    "    over_context = [t for t in rejected if t.get(\"over_context\")]\n"
    "    if over_context:\n"
    "        rec[\"over_context_rejections\"] = len(over_context)\n"
    "    line = (json.dumps(rec) + \"\\n\").encode(\"utf-8\")\n"
    "    with _GBENCH_LOG_LOCK:\n"
    "        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)\n"
    "        try:\n"
    "            os.write(fd, line)\n"
    "        finally:\n"
    "            os.close(fd)\n"
)
assert src.count(HELPERS_OLD) == 1, "toolbench patch: the settings block of patch 3 not found"
src = src.replace(HELPERS_OLD, HELPERS_NEW, 1)
REPLY_OLD = "        json_data = openai_response.dict()\n        return json_data\n"
REPLY_NEW = ("        json_data = openai_response.dict()\n"
             "        _gbench_note_try(None)\n"
             "        return json_data\n")
assert src.count(REPLY_OLD) == 1, "toolbench patch: the reply return not found (upstream drifted?)"
src = src.replace(REPLY_OLD, REPLY_NEW, 1)
ERROR_OLD = '        return {"error": str(e), "total_tokens": 0}\n'
ERROR_NEW = '        _gbench_note_try(e)\n        return {"error": str(e), "total_tokens": 0}\n'
assert src.count(ERROR_OLD) == 1, "toolbench patch: the error return not found (upstream drifted?)"
src = src.replace(ERROR_OLD, ERROR_NEW, 1)
# The step wrapper goes at the end of the module, around upstream's ChatGPTFunction.parse (the
# image's build asserts it is installed: docker/toolbench.Dockerfile).
WRAPPER = (
    "\n\n"
    "# gbench: every step of the DFSDT search (ChatGPTFunction.parse, upstream's) notes its\n"
    "# requests' outcomes and records one that failed at the endpoint or was rejected for its\n"
    "# content (patch 4 above).\n"
    "if \"ChatGPTFunction\" in globals():\n"
    "    _gbench_upstream_parse = ChatGPTFunction.parse\n"
    "\n"
    "    def _gbench_parse(self, *args, **kwargs):\n"
    "        _GBENCH_TLS.tries = []\n"
    "        try:\n"
    "            out = _gbench_upstream_parse(self, *args, **kwargs)\n"
    "        finally:\n"
    "            tries, _GBENCH_TLS.tries = _GBENCH_TLS.tries, None\n"
    "        _gbench_record_step(self, out[1], tries)\n"
    "        return out\n"
    "\n"
    "    ChatGPTFunction.parse = _gbench_parse\n"
)
assert "_gbench_parse" not in src, "toolbench patch: patch 4 applied twice"
MAIN_OLD = '\nif __name__ == "__main__":\n'
if src.count(MAIN_OLD) == 1:
    src = src.replace(MAIN_OLD, WRAPPER + MAIN_OLD, 1)
else:
    src = src.rstrip("\n") + "\n" + WRAPPER

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("toolbench_patch_llm: patched", PATH)
