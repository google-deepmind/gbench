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
"""gbench build-time patch of StableToolBench's cached server (server/main.py).

1. The cache-miss simulator's model. Upstream's server (`fake_response_function_chat`) simulates
   the response to an API call that is neither in the cache nor answered by the real-API proxy
   with ONE model, CONFIG['model'] (gpt-4-turbo in the committed server/config.yml). gbench
   replaces that model, and only that, with its Gemini cascade on the Gemini API's
   OpenAI-compatible endpoint, reached through gbench's Gemini relay (CONFIG['api_base'] and
   CONFIG['api_key'], which docker/toolbench_entrypoint.sh sets to the relay's URL and to the
   relay's per-run placeholder), in the shape of base.judge_generate_cascade:
     * the models in GBENCH_JUDGE_MODELS (comma list; gbench passes base.judge_cascade()), tried
       in order; a model the API answers HTTP 404 for is not asked again in that call;
     * TB_SIMULATOR_ROUNDS rounds, with TB_SIMULATOR_BACKOFF * 2**round (+ jitter) seconds
       between rounds;
     * one client per placeholder in CONFIG['api_key'], rotated per round.
   Every request keeps upstream's fields: the system and user messages, CONFIG['temperature'] (0)
   and response_format json_object; upstream's three tries for a reply that is not valid JSON
   stay as they are. Its max_tokens is upstream's 1024 for the reply plus
   GEMINI_THINKING_ALLOWANCE (8192, gbench's gemini_budget value; a test checks the copy): the
   Gemini models think and their thinking counts against max_tokens on the OpenAI-compatible
   endpoint, where gpt-4-turbo does not think, so the JSON keeps upstream's room. A reply that
   still stops there (finish_reason "length") with content is used as it arrived, as upstream
   uses a reply cut at its max_tokens, and counted (`simulator_max_tokens_cut`); one with no
   content counts as that model failing (`simulator_max_tokens_empty`), and a call no model then
   served is marked `simulator_outage_after_max_tokens`. A reply without content counts as that
   model failing; when every model fails every round the last error is raised, as upstream's
   single call raises (the server then answers HTTP 500, as upstream).

2. The real-API proxy call (CONFIG['toolbench_url']) is upstream's request, with a read timeout
   (TB_REAL_API_TIMEOUT_S, default 120 s): upstream sends it with none, so a proxy that never
   answers would hang the query (and the run) until the container's own time limit. A proxy that
   answers, whatever it answers, goes on as upstream.

3. The record of every /virtual call. The route runs upstream's get_virtual_response unchanged
   and, after it, appends one JSON line to TB_SERVER_LOG (default /out/toolbench_server_calls.jsonl):
   the test group and query id the client tagged the call with (X-Gbench-Group /
   X-Gbench-Query-Id, docker/toolbench_patch_client.py), the tool and API, upstream's log type of
   the answer (cached_real_response, real_response, fake_response, tool_input_parse_error), the
   real-API proxy's HTTP status, the Gemini model that simulated the answer and the cascade
   attempts, the simulator replies cut at the output limit (above), `simulator_failed_reply`
   when the answer is upstream's "Failed to generate fake response" (three replies without
   valid JSON, also when replayed from the cache), and for a call that raised (HTTP 500)
   `server_error` with the stage it failed in (real_api_proxy, simulator or server) and the
   exception class. No key and no tool input is written. gbench
   counts these calls (toolbench.py): a server error and a failed simulation are harness errors,
   and the queries that met one are left out of the scores.

Each replacement is asserted to match exactly, so the build FAILS if upstream drifts rather than
silently shipping an unpatched image. Mirrors docker/toolbench_patch_llm.py.
"""

PATH = "/stb/server/main.py"
#: Room for the Gemini model's thinking on top of upstream's max_tokens 1024 (a copy of
#: gbench/runners/eval_suites/gemini_budget.py GEMINI_THINKING_ALLOWANCE; this script runs in the
#: image build and cannot import gbench).
GEMINI_THINKING_ALLOWANCE = 8192

with open(PATH, encoding="utf-8") as f:
    src = f.read()

# --- 1. The simulator call -> the Gemini cascade. ---
CALL_OLD = (
    "    client = OpenAI(\n"
    "        api_key = OPENAI_API_KEY,\n"
    "        base_url = OPENAI_API_BASE,\n"
    "    )\n"
    "    max_retries = 3 \n"
    "    flag = False\n"
    "    for attempt in range(max_retries):\n"
    "        response = client.chat.completions.create(\n"
    "            model = CONFIG['model'],\n"
    "            messages=[system_prompt, user_prompt],\n"
    "            max_tokens = 1024,\n"
    "            temperature=CONFIG['temperature'],\n"
    "            response_format={\"type\": \"json_object\"},\n"
    "        )\n"
    "        result = response.choices[0].message.content"
)
CALL_NEW = (
    "    # gbench: the simulator model is the gbench Gemini cascade (_gbench_simulate); the\n"
    "    # request fields and the three tries for a reply that is not valid JSON are upstream's;\n"
    "    # max_tokens is upstream's 1024 for the reply plus room for the Gemini model's thinking.\n"
    "    max_retries = 3 \n"
    "    flag = False\n"
    "    for attempt in range(max_retries):\n"
    "        response = _gbench_simulate(\n"
    "            messages=[system_prompt, user_prompt],\n"
    "            max_tokens = 1024 + _GBENCH_THINKING_ALLOWANCE,\n"
    "            temperature=CONFIG['temperature'],\n"
    "            response_format={\"type\": \"json_object\"},\n"
    "        )\n"
    "        result = response.choices[0].message.content"
)
assert src.count(CALL_OLD) == 1, \
    "toolbench patch: simulator call block not found (upstream drifted?)"
src = src.replace(CALL_OLD, CALL_NEW, 1)

# --- 2. The real-API proxy call -> the same request with a read timeout, recorded. ---
PROXY_OLD = ("    real_response = requests.post(CONFIG['toolbench_url'], headers=headers, "
             "data=json.dumps(data))\n")
PROXY_NEW = ("    real_response = _gbench_real_api_post(CONFIG['toolbench_url'], headers=headers, "
             "data=json.dumps(data))\n")
assert src.count(PROXY_OLD) == 1, \
    "toolbench patch: real-API proxy call not found (upstream drifted?)"
src = src.replace(PROXY_OLD, PROXY_NEW, 1)

# --- 3. The /virtual route -> upstream's function, then the record of the call. ---
ROUTE_OLD = ("@app.post('/virtual')\n"
             "# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))\n"
             "def get_virtual_response(request: Request, info: Info):\n")
ROUTE_NEW = ("# gbench: registered on /virtual by _gbench_virtual (below), which records the call.\n"
             "# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))\n"
             "def get_virtual_response(request: Request, info: Info):\n")
assert src.count(ROUTE_OLD) == 1, "toolbench patch: /virtual route not found (upstream drifted?)"
src = src.replace(ROUTE_OLD, ROUTE_NEW, 1)

ANCHOR = "def fake_response_function_chat(api_example, tool_input, api_doc):\n"
HELPER = '''# gbench: the cache-miss simulator on the gbench Gemini cascade, the real-API proxy call with
# a read timeout, and the record of every /virtual call (see docker/toolbench_patch_simulator.py
# in gbench).
import itertools as _gbench_itertools
import random as _gbench_random
import re as _gbench_re
import threading as _gbench_threading

_GBENCH_CLIENTS = {"cycle": None}
_GBENCH_CALL = _gbench_threading.local()
_GBENCH_LOG_LOCK = _gbench_threading.Lock()
#: upstream's reply after three simulator replies without valid JSON (fake_response_function_chat)
_GBENCH_FAILED_REPLY = "Failed to generate fake response"
#: room for the Gemini model's thinking on top of upstream's max_tokens 1024
_GBENCH_THINKING_ALLOWANCE = __GBENCH_THINKING_ALLOWANCE__


def _gbench_models():
    raw = os.environ.get("GBENCH_JUDGE_MODELS") or ""
    return [m.strip() for m in raw.split(",") if m.strip()]


def _gbench_next_client():
    if _GBENCH_CLIENTS["cycle"] is None:
        raw = str(OPENAI_API_KEY or "")
        keys = list(dict.fromkeys(k.strip() for k in _gbench_re.split(r"[;,]", raw) if k.strip()))
        clients = [OpenAI(api_key=k, base_url=OPENAI_API_BASE, max_retries=0) for k in keys]
        _GBENCH_CLIENTS["cycle"] = _gbench_itertools.cycle(
            clients or [OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, max_retries=0)])
    return next(_GBENCH_CLIENTS["cycle"])


def _gbench_note(**fields):
    call = getattr(_GBENCH_CALL, "record", None)
    if call is not None:
        call.update(fields)


def _gbench_simulate(**request):
    _gbench_note(stage="simulator")
    models = _gbench_models()
    if not models:
        raise RuntimeError("GBENCH_JUDGE_MODELS is not set: the simulator has no Gemini model")
    rounds = max(1, int(os.environ.get("TB_SIMULATOR_ROUNDS") or 3))
    backoff = float(os.environ.get("TB_SIMULATOR_BACKOFF") or 1.0)
    not_served = set()
    last = None
    attempts = 0
    empty_at_cap = 0
    for rnd in range(rounds):
        client = _gbench_next_client()
        for model in models:
            if model in not_served:
                continue
            attempts += 1
            try:
                response = client.chat.completions.create(model=model, **request)
                content = response.choices[0].message.content
                at_cap = getattr(response.choices[0], "finish_reason", None) == "length"
                if content is not None and content.strip() != "":
                    call = getattr(_GBENCH_CALL, "record", None)
                    if call is not None:
                        call["simulator_attempts"] = call.get("simulator_attempts", 0) + attempts
                        call.setdefault("simulator_models", []).append(model)
                        if at_cap:  # used as it arrived, as upstream uses a cut reply; counted
                            call["simulator_max_tokens_cut"] = (
                                call.get("simulator_max_tokens_cut", 0) + 1)
                    return response
                if at_cap:  # the thinking took the whole limit: the next model is asked
                    empty_at_cap += 1
                    call = getattr(_GBENCH_CALL, "record", None)
                    if call is not None:
                        call["simulator_max_tokens_empty"] = (
                            call.get("simulator_max_tokens_empty", 0) + 1)
                last = ValueError("empty simulator reply from " + model)
            except Exception as e:  # 429 / network / a model not served: the next model
                if getattr(e, "status_code", None) == 404:
                    not_served.add(model)
                last = e
        if len(not_served) >= len(models):
            break
        if rnd < rounds - 1:
            time.sleep(backoff * (2 ** rnd) + _gbench_random.uniform(0, backoff))
    _gbench_note(simulator_attempts=attempts)
    if empty_at_cap:
        _gbench_note(simulator_outage_after_max_tokens=True)
    raise last if last is not None else RuntimeError("no simulator model answered")


def _gbench_real_api_post(url, **kwargs):
    _gbench_note(stage="real_api_proxy")
    timeout = float(os.environ.get("TB_REAL_API_TIMEOUT_S") or 120)
    response = requests.post(url, timeout=timeout, **kwargs)
    _gbench_note(stage="server", real_api_status=response.status_code)
    return response


def _gbench_write_record(record):
    path = os.environ.get("TB_SERVER_LOG") or "/out/toolbench_server_calls.jsonl"
    line = json.dumps(record, default=str)
    with _GBENCH_LOG_LOCK:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\\n")
        except OSError as e:  # a record only; the call goes on as upstream
            print(f"gbench: could not write {path}: {e}")


_gbench_upstream_write_log = write_log


def write_log(request, response, type):
    _gbench_note(type=type)
    return _gbench_upstream_write_log(request, response, type)


def _gbench_failed_reply(result):
    if isinstance(result, dict):
        return result.get("error") == _GBENCH_FAILED_REPLY
    if isinstance(result, str):
        return _GBENCH_FAILED_REPLY in result
    return False


'''
assert src.count(ANCHOR) == 1, \
    "toolbench patch: fake_response_function_chat not found (upstream drifted?)"
HELPER = HELPER.replace("__GBENCH_THINKING_ALLOWANCE__", str(int(GEMINI_THINKING_ALLOWANCE)))
src = src.replace(ANCHOR, HELPER + ANCHOR, 1)

MAIN = 'if __name__ == "__main__":\n'
ROUTE = '''# gbench: the /virtual route, upstream's get_virtual_response unchanged, then one record line.
@app.post('/virtual')
def _gbench_virtual(request: Request, info: Info):
    record = {"time": time.time(), "group": request.headers.get("x-gbench-group") or None,
              "query_id": request.headers.get("x-gbench-query-id") or None,
              "category": info.category, "tool": info.tool_name, "api": info.api_name,
              "stage": "cache", "type": None}
    _GBENCH_CALL.record = record
    try:
        result = get_virtual_response(request, info)
    except Exception as e:
        record["server_error"] = True
        record["error"] = type(e).__name__
        record["type"] = "server_error"
        _gbench_write_record(record)
        raise
    finally:
        _GBENCH_CALL.record = None
    record.pop("stage", None)
    record["simulator_failed_reply"] = _gbench_failed_reply(result)
    _gbench_write_record(record)
    return result


'''
assert src.count(MAIN) == 1, "toolbench patch: __main__ block not found (upstream drifted?)"
src = src.replace(MAIN, ROUTE + MAIN, 1)

# Guards: the simulator no longer calls CONFIG['model'] directly, the helpers and the recorded
# route are in place, and /virtual is registered once.
assert "model = CONFIG['model']" not in src, "toolbench patch: a direct simulator call survived"
assert src.count("_gbench_simulate(") == 2
assert src.count("@app.post('/virtual')") == 1
assert "requests.post(CONFIG['toolbench_url']" not in src
assert "max_tokens = 1024 + _GBENCH_THINKING_ALLOWANCE," in src
assert f"_GBENCH_THINKING_ALLOWANCE = {int(GEMINI_THINKING_ALLOWANCE)}\n" in src

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("toolbench_patch_simulator: patched", PATH)
