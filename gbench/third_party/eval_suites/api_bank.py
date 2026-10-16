# -*- coding: utf-8 -*-
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

# Evaluation Suite: api_bank
# Description: API-Bank (Li et al., EMNLP 2023 - runnable tool-use benchmark)

"""gbench native built-in runner for api_bank (Tool Use & Function Calling).

Canonical API-Bank (Li et al., EMNLP 2023; arXiv:2304.08244; code
AlibabaResearch/DAMO-ConvAI/api-bank; dataset liminghao1630/API-Bank) is a RUNNABLE
benchmark: a simulated pool of ~73 Python APIs backed by a local fake DB. It measures three
progressively harder abilities, each a headline column, and TWO tracks per level:

  * Level-1 Call            (all API descriptions given -> emit the right call)
  * Level-2 Retrieve+Call   (must first ToolSearcher-retrieve the API, then call it)
  * Level-3 Plan+Retrieve+Call
  Correctness (Accuracy) = EXECUTION-BASED: execute the predicted call against the simulated
  backend and compare via each API's `check_api_call_correctness` against the gold result.
  Response quality = ROUGE-L F over the AI-turn generations. Level-3 also reports a dialogue
  sample success rate = (50 - #dialogues-with-any-errored-step)/50. No LLM judge anywhere.

Sourcing: L1/L2 drive from the checkout's raw conversations (lv1-lv2-samples/*) via the
canonical `Sample.from_chat_history` + `Evaluator.evaluate` (reused verbatim from the
checkout for fidelity: fresh ToolManager per call, execution + `check_api_call_correctness`).
L3 drives from the HF dataset (test-data/level-3-batch-inf*.json + level-3.json) against the
checkout's lv3_apis backend. The execution backend is a user-provided CHECKOUT pointed at by
`GBENCH_APIBANK_DIR` (this suite hard-errors with clone instructions if it is absent, and if its
scoring files differ from the pinned api-bank tree, APIBANK_COMMIT). The L3 HF files are read
at _HF_REVISION. The prompts are evaluator.py's, byte for byte (trailing spaces included;
PROMPT_SHA256).

Translate backend: upstream's apis/translate.py calls the synchronous googletrans API of the release
its requirements.txt pins, googletrans==3.1.0a0 (httpx 0.13.3), which cannot be installed next to
gbench. gbench runs translate.py unchanged in its process and hands each googletrans call to that
release in the local image TRANSLATE_IMAGE (gbench/docker/api_bank.Dockerfile; one container per
call, docker's bridge network, read-only, unprivileged, no key or token): `googletrans` is, for the
checkout's modules, a stand-in module whose Translator runs googletrans 3.1.0a0's
Translator(raise_exception=True).translate in that image and whose LANGUAGES are 3.1.0a0's. The
proxy variables (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, NO_PROXY, upper and lower case) are read when
a Translator is built, as httpx 0.13.3 reads them in Translator(), and passed to the container (a
loopback proxy as host.docker.internal). raise_exception=True makes a Google reply other than HTTP
200 an exception ("Unexpected status code ..."), where googletrans' default returns the
untranslated text as the translation, which would score an outage as the model's wrong call.
Before any request is sent, check_backend_prerequisites checks the image (googletrans 3.1.0a0,
httpx 0.13.3, the worker file this checkout ships), that no proxy variable carries a credential,
and that the cloud metadata server is blocked on the bridge network (container_guard); any of
them failing is infra_required. The result records `translate_backend`.

Harness errors are never the model's outcome. A step whose request produced no reply (an endpoint
error, a timeout, a prompt longer than the served window) is not scored. Three simulated APIs
reach outside the process (BACKEND_APIS: Translate through googletrans, Dictionary through
api.dictionaryapi.dev, ToolSearcher through a sentence-transformers model loaded on every call):
a failed step that called the gold one of them, whose execution came back with an exception, and
whose gold call fails on the backend too, is a harness error (the backend could not answer any
call), not the model's error. Such steps are left out of the level means, counted in
`harness_errors` and named in `leaderboard_comparable_reason`; the status is then
completed_with_errors, and a run with no scored call step withholds its score (accuracy null,
status failed). The Level-3 sample success rate, lv3_evaluator's (50 - errored dialogues) / 50,
is computed only when every Level-3 call step was run, over the dialogues whose outcome is known.

Sampling: as evaluator.py, which calls the model with `chatgpt.call(messages)` (utils.py
ChatGPTWrapper sends only `model` and `messages`): no sampling field other than the temperature is
sent. The temperature is gbench's rule for every suite (base.resolve_temperature): 1.0 with
`--thinking`, 0.0 without (evaluator.py sends none). `--temperature`, or
`GBENCH_API_BANK_TEMPERATURE` for this suite (it takes precedence), sets another; the result
records upstream's setting under `official_sampling`. No
max_tokens is sent either, in either mode (the server grants the room left in the context window;
--max-output-tokens is not applied and `max_output_tokens_note` says so). No LLM judge, so no
GEMINI_API_KEY.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import types
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .base import (OfficialSampling, redact_text, run_eval_suite, secret_values,
                   strip_thinking_tags)
from .sampling import stratified_sample
from .fc_common import base_suite_plan, run_attempts
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/api_bank.md"

_ENV = "GBENCH_APIBANK_DIR"
_HF_REPO = "liminghao1630/API-Bank"
#: HF revision of the L3 test data (2023-11-17, the upload the api-bank README points to).
_HF_REVISION = "12e8158b7628c168f07e8f31fbbe3445e99f44cf"
#: AlibabaResearch/DAMO-ConvAI commit of the api-bank/ tree the checkout must match (the last
#: commit touching api-bank/, 2023-11-17; api-bank/ is unchanged at HEAD).
APIBANK_COMMIT = "f30ccf22b4e2617fab32958d4c03f5c1f2e7dfcf"
#: sha256 over sorted "<path> <git blob sha1>" lines of the files scoring reads: evaluator.py,
#: tool_manager.py, api_call_extraction.py and everything under apis/, lv3_apis/ and
#: lv1-lv2-samples/ (344 files at APIBANK_COMMIT). init_database/ is excluded because the
#: simulated APIs write to it.
_CHECKOUT_DIGEST = "66978bd75f70048885822e9c6955b89cc3c837b467010f5ffedc7292a4b265d0"
_DIGEST_FILES = ("evaluator.py", "tool_manager.py", "api_call_extraction.py")
_DIGEST_DIRS = ("apis", "lv3_apis", "lv1-lv2-samples")

#: evaluator.py: `response = chatgpt.call(messages)`; utils.py ChatGPTWrapper.call (and
#: GPT4Wrapper.call, which evaluator.py builds) posts {"model", "messages"} and nothing else, so
#: the API's default sampling applies and no max_tokens is sent.
API_BANK_SAMPLING = OfficialSampling(
    reference=(f"AlibabaResearch/DAMO-ConvAI@{APIBANK_COMMIT[:12]} api-bank/evaluator.py "
               "(chatgpt.call(messages); utils.py ChatGPTWrapper sends model and messages only)"),
    send_temperature=False,
    send_max_tokens=False,
)

# Prompt templates, VERBATIM from api-bank/evaluator.py (api_call_prompt and response_prompt at
# APIBANK_COMMIT), the canonical single-turn inputs. Written line by line so the trailing spaces
# upstream's lines carry ("...actual parameters. ", "...in your output. ", "Input: ") survive any
# editor; PROMPT_SHA256 pins both strings to upstream's.
API_CALL_PROMPT = (
    "\n"
    "Based on the given API description and the existing conversation history 1..t, please "
    "generate the API request that the AI should call in step t+1 and output it in the format of "
    "[ApiName(key1='value1', key2='value2', ...)], replace the ApiName with the actual API name, "
    "and replace the key and value with the actual parameters. \n"
    "Your output should start with a square bracket \"[\" and end with a square bracket \"]\". "
    "Do not output any other explanation or prompt or the result of the API call in your "
    "output. \n"
    "This year is 2023.\n"
    "Input: \n"
    "User: [User's utterence]\n"
    "AI: [AI's utterence]\n"
    "\n"
    "Expected output:\n"
    "[ApiName(key1='value1', key2='value2', ...)]\n"
    "\n"
    "API descriptions:\n"
)

RESPONSE_PROMPT = (
    "\n"
    "Based on the given API description and the existing conversation history 1..t, please "
    "generate the next dialog that the AI should response after the API call t.\n"
    "This year is 2023.\n"
    "Input: \n"
    "User: [User's utterence]\n"
    "AI: [AI's utterence]\n"
    "[ApiName(key1='value1', key2='value2', \u2026)]\n"
    "\n"
    "Expected output:\n"
    "AI: [AI's utterence]\n"
    "\n"
    "API descriptions:\n"
)

#: sha256 of api_call_prompt / response_prompt in api-bank/evaluator.py at APIBANK_COMMIT.
PROMPT_SHA256 = {
    "API_CALL_PROMPT": "bf428a1d00aa56b8520e62b45fc4c1a65512c0fdda8d9993c8e6e07d527ce892",
    "RESPONSE_PROMPT": "363519dc74125e1f38721bf6b9fe1a9cf8c7bf570a3a5dda0c08646fad5e2383",
}

#: The simulated APIs whose backend reaches outside gbench's process, so a failure of it is a
#: harness error, not the model's call (see _backend_harness_error). No other API of apis/ or
#: lv3_apis/ does (lv3_apis/get_weather.py imports requests but answers from its own table).
BACKEND_APIS = {
    "Translate": ("apis/translate.py: googletrans, the Google Translate web service "
                  "(translate.googleapis.com); upstream's requirements.txt pin "
                  "googletrans==3.1.0a0 runs in the local image TRANSLATE_IMAGE"),
    "Dictionary": "apis/dictionary.py: requests.get to https://api.dictionaryapi.dev",
    "ToolSearcher": ("apis/tool_search.py and lv3_apis/tool_search.py: "
                     "SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2'), loaded "
                     "on every call"),
}
#: The embedding model ToolSearcher loads on every call.
TOOL_SEARCHER_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
#: lv3_evaluator.py: Sample Accuracy = (50 - dialogues with an errored call step) / 50, over the
#: level-3-batch-inf.json steps (all of them predicted).
L3_DIALOGUES = 50

# --------------------------------------------------------------------------- #
# The Translate API's googletrans, in its own image
# --------------------------------------------------------------------------- #
#: The googletrans release api-bank/requirements.txt pins, and the httpx release it pins.
GOOGLETRANS_RELEASE = "3.1.0a0"
HTTPX_RELEASE = "0.13.3"
#: The local image googletrans runs in, and how it is built (from the gbench repository root).
TRANSLATE_IMAGE = "gbench-api-bank:googletrans-3.1.0a0"
TRANSLATE_DOCKERFILE = "gbench/docker/api_bank.Dockerfile"
TRANSLATE_IMAGE_CONTEXT = "gbench/docker"
TRANSLATE_IMAGE_LABELS = {"org.gbench.suite": "api_bank",
                          "org.gbench.api-bank.googletrans": GOOGLETRANS_RELEASE,
                          "org.gbench.api-bank.httpx": HTTPX_RELEASE}
#: The worker the image runs (a copy is baked into the image; its sha256 must match this file).
_TRANSLATE_WORKER = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "docker",
    "api_bank_translate_worker.py"))
_WORKER_REPLY_MARK = "GBENCH_TRANSLATE_WORKER="
#: The host googletrans 3.1.0a0 sends its requests to (DEFAULT_CLIENT_SERVICE_URLS).
TRANSLATE_HOST = "translate.googleapis.com"
#: Seconds one worker container may take (googletrans' httpx client times out each request after
#: 5 s; the rest is the container's start).
_TRANSLATE_WORKER_TIMEOUT_S = 120
#: The proxy variables a Translator reads when it is built (httpx 0.13.3 honors http, https and
#: all), and passes on to the worker container.
_PROXY_URL_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                   "all_proxy")
_NO_PROXY_VARS = ("NO_PROXY", "no_proxy")
#: The run's googletrans calls: calls, and the failures by kind.
_TRANSLATE_STATS: Counter = Counter()
#: The exception text of each Translate call that failed on the harness's or the backend's side
#: (gbench's worker container, a transport error, or Google answering 429, 5xx, 403 or 407), by
#: kind: the step whose model call failed with it is a harness error without a gold re-run.
_BACKEND_FAILURES: Dict[str, str] = {}
#: httpx 0.13.3 transport exceptions (the worker reports the type): the request never got an
#: answer from Google.
_TRANSPORT_ERRORS = frozenset({"ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
                               "PoolTimeout", "TimeoutException", "NetworkError", "ProxyError",
                               "RemoteProtocolError", "ReadError", "WriteError", "CloseError"})
#: googletrans 3.1.0a0 raises this for a non-200 answer; these statuses are Google's side (rate
#: limit, server error) or the network's (a proxy refusing), not the text's.
_BACKEND_STATUS = re.compile(r'Unexpected status code "(429|5\d\d|403|407)"')
_PROBE: Dict[str, Any] = {}
_PROBE_LOCK = threading.Lock()


def lookup_proxy_env(proxy_env: Dict[str, str]) -> Dict[str, str]:
    """`proxy_env` as the name-lookup check reads it: httpx 0.13.3 (the worker) falls back to
    ALL_PROXY / all_proxy for a scheme without its own proxy, and the check reads only
    <scheme>_proxy, so the all-scheme proxy is copied into HTTPS_PROXY / HTTP_PROXY where
    neither spelling is set."""
    out = dict(proxy_env)
    all_proxy = out.get("ALL_PROXY") or out.get("all_proxy")
    if all_proxy:
        for scheme in ("HTTPS", "HTTP"):
            if not (out.get(f"{scheme}_PROXY") or out.get(f"{scheme.lower()}_proxy")):
                out[f"{scheme}_PROXY"] = all_proxy
    return out


def translate_image_build_command() -> str:
    """The command that builds TRANSLATE_IMAGE (run from the gbench repository root)."""
    return f"docker build -t {TRANSLATE_IMAGE} -f {TRANSLATE_DOCKERFILE} {TRANSLATE_IMAGE_CONTEXT}"


def translate_proxy_env(environ: Dict[str, str]) -> Dict[str, str]:
    """The proxy variables of `environ` as the worker container gets them: the set ones, a
    loopback proxy host as host.docker.internal (tau_common.container_url)."""
    from .tau_common import container_url
    out = {n: container_url(environ[n]) for n in _PROXY_URL_VARS if (environ.get(n) or "").strip()}
    for n in _NO_PROXY_VARS:
        if (environ.get(n) or "").strip():
            out[n] = environ[n]
    return out


def _url_has_password(value: str) -> bool:
    """Whether a proxy URL carries a password (`scheme://user:password@host`), of any length."""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(value if "://" in value else f"http://{value}")
        return bool(parts.password)
    except ValueError:
        return "@" in value


def _proxy_credentials(proxy_env: Dict[str, str]) -> List[str]:
    """The proxy variables whose value holds a credential (a password in the proxy URL, or a
    credential value of this process' environment, a key): none may enter a container that runs
    a model's call."""
    secrets = secret_values()
    return sorted(n for n, v in proxy_env.items()
                  if _url_has_password(v or "") or any(s in (v or "") for s in secrets))


class _TranslateWorkerError(Exception):
    """A failure of a googletrans call in the worker (its message is str(exception), as
    translate.py records it), or of the worker container itself."""


def _run_translate_worker(request: Dict[str, Any], proxy_env: Dict[str, str]) -> Dict[str, Any]:
    """One request to api_bank_translate_worker.py in a fresh TRANSLATE_IMAGE container on the
    bridge network; its reply ({"result": ...} or {"error": {"type", "message"}}). Raises
    _TranslateWorkerError when the container gives no reply."""
    from .tau_common import ENDPOINT_NETWORK_ARGS
    bad = _proxy_credentials(proxy_env)
    if bad:
        raise _TranslateWorkerError(
            f"gbench's googletrans worker was not started: {', '.join(bad)} would carry a "
            "credential into its container")
    name = f"gbench-api-bank-translate-{uuid.uuid4().hex[:12]}"
    argv = ["docker", "run", "--rm", "-i", "--pull", "never", "--name", name,
            "--label", "org.gbench.suite=api_bank", *ENDPOINT_NETWORK_ARGS,
            "--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--memory", "512m", "--pids-limit", "128"]
    for n in sorted(proxy_env):
        argv += ["-e", n]
    argv.append(TRANSLATE_IMAGE)
    child_env = {k: v for k, v in os.environ.items()
                 if k not in _PROXY_URL_VARS and k not in _NO_PROXY_VARS}
    child_env.update(proxy_env)
    try:
        proc = subprocess.run(argv, input=json.dumps(request), capture_output=True, text=True,
                              timeout=_TRANSLATE_WORKER_TIMEOUT_S, env=child_env)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        raise _TranslateWorkerError(
            f"gbench's googletrans worker gave no reply within {_TRANSLATE_WORKER_TIMEOUT_S} s")
    except OSError as e:
        raise _TranslateWorkerError(
            f"gbench's googletrans worker could not start ({type(e).__name__}: {e})")
    for line in (proc.stdout or "").splitlines():
        if line.startswith(_WORKER_REPLY_MARK):
            try:
                reply = json.loads(line[len(_WORKER_REPLY_MARK):])
            except ValueError:
                break
            if isinstance(reply, dict):
                return reply
            break
    shown = redact_text(" ".join((proc.stderr or proc.stdout or "no output").split()))[:300]
    raise _TranslateWorkerError(
        f"gbench's googletrans worker gave no reply (docker exit {proc.returncode}: {shown})")


def _translate_probe() -> Dict[str, Any]:
    """The image's probe reply (googletrans and httpx releases, the worker's sha256,
    googletrans.LANGUAGES / LANGCODES), run once per process."""
    with _PROBE_LOCK:
        if not _PROBE:
            reply = _run_translate_worker({"op": "probe"}, {})
            if "error" in reply:
                raise _TranslateWorkerError(
                    "the googletrans image's probe failed: "
                    f"{(reply.get('error') or {}).get('message')}")
            _PROBE.update(reply)
        return dict(_PROBE)


class _Translated:
    """The fields of googletrans' Translated the worker returns (translate.py reads `.text`)."""

    def __init__(self, fields: Dict[str, Any]) -> None:
        self.text = fields.get("text")
        self.src = fields.get("src")
        self.dest = fields.get("dest")
        self.origin = fields.get("origin")
        self.pronunciation = fields.get("pronunciation")

    def __repr__(self) -> str:
        return (f"Translated(src={self.src}, dest={self.dest}, text={self.text}, "
                f"pronunciation={self.pronunciation})")


class _WorkerTranslator:
    """googletrans.Translator as apis/translate.py uses it: built with no argument, then
    `.translate(text, dest=..., src=...)`, run by googletrans 3.1.0a0 in TRANSLATE_IMAGE. The
    proxy variables are read here, when it is built, as httpx 0.13.3's client reads them in
    googletrans' Translator()."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            raise TypeError("gbench's googletrans stand-in takes no argument (apis/translate.py "
                            "builds Translator() with none)")
        self._proxy_env = translate_proxy_env(dict(os.environ))

    def translate(self, text: Any, dest: str = "en", src: str = "auto", **kwargs: Any):
        if kwargs:
            raise TypeError(f"unexpected googletrans arguments {sorted(kwargs)}")
        _TRANSLATE_STATS["calls"] += 1
        try:
            reply = _run_translate_worker({"op": "translate", "text": text, "dest": dest,
                                           "src": src}, self._proxy_env)
        except _TranslateWorkerError as e:
            _TRANSLATE_STATS["worker_failed"] += 1
            _BACKEND_FAILURES[str(e)] = "translate_worker"
            raise
        if "error" in reply:
            err = reply.get("error") or {}
            message = str(err.get("message") or "")
            _TRANSLATE_STATS["error:" + str(err.get("type") or "unknown")] += 1
            if str(err.get("type") or "") in _TRANSPORT_ERRORS:
                _BACKEND_FAILURES[message] = "translate_transport"
            elif _BACKEND_STATUS.search(message):
                _BACKEND_FAILURES[message] = "translate_backend_status"
            raise _TranslateWorkerError(message)
        return _Translated(reply.get("result") or {})


def _googletrans_getattr(name: str) -> Any:
    if name in ("LANGUAGES", "LANGCODES"):
        return dict(_translate_probe()[name.lower()])
    raise AttributeError(f"module 'googletrans' has no attribute {name!r}")


_SHIM = types.ModuleType("googletrans", "googletrans 3.1.0a0, run in gbench's api_bank image")
_SHIM.Translator = _WorkerTranslator
_SHIM.__version__ = "3.1.0-alpha"
_SHIM.__getattr__ = _googletrans_getattr
_MISSING = object()


def _bind_googletrans() -> Any:
    """Make `googletrans` the stand-in for the checkout's modules: in sys.modules while the
    checkout is imported, and in apis/translate.py's globals if that module was imported before.
    Returns what sys.modules held, for _in_checkout to restore."""
    prev = sys.modules.get("googletrans", _MISSING)
    sys.modules["googletrans"] = _SHIM
    mod = sys.modules.get("apis.translate")
    if mod is not None and getattr(mod, "googletrans", None) is not _SHIM:
        mod.googletrans = _SHIM
        mod.Translator = _WorkerTranslator
    return prev


#: The device ToolSearcher's encoder runs on. Upstream's SentenceTransformer(...) takes cuda:0 when
#: a GPU is visible, where the model server under test holds the memory (2026-10-10 smoke test:
#: CUDA out of memory, the suite failed before its first request).
TOOL_SEARCHER_DEVICE = "cpu"
_CPU_ENCODERS: Dict[Any, Any] = {}


def _cpu_encoder_class(real: Any) -> Any:
    """`real` (sentence_transformers.SentenceTransformer) with TOOL_SEARCHER_DEVICE as its
    default device; a device the caller passes is kept."""
    if real not in _CPU_ENCODERS:
        class _CpuSentenceTransformer(real):
            def __init__(self, *args: Any, device: Any = None, **kwargs: Any) -> None:
                super().__init__(*args, device=device or TOOL_SEARCHER_DEVICE, **kwargs)
        _CPU_ENCODERS[real] = _CpuSentenceTransformer
    return _CPU_ENCODERS[real]


def _bind_tool_searcher_device() -> Any:
    """Make the checkout's SentenceTransformer load on TOOL_SEARCHER_DEVICE: in the
    sentence_transformers package while the checkout is imported, and in the globals of
    apis/tool_search.py and lv3_apis/tool_search.py if they were imported before. Returns the
    package's class, for _in_checkout to restore (None without sentence-transformers)."""
    try:
        import sentence_transformers as st
    except ImportError:
        return None
    real = st.SentenceTransformer
    if real in _CPU_ENCODERS.values():
        real = real.__mro__[1]
    cpu = _cpu_encoder_class(real)
    st.SentenceTransformer = cpu
    for name in ("apis.tool_search", "lv3_apis.tool_search"):
        mod = sys.modules.get(name)
        if mod is not None:
            mod.SentenceTransformer = cpu
    return real


def check_translate_backend() -> Dict[str, Any]:
    """Before any request is sent: the Translate API's googletrans can run as upstream pins it.
    infra_required when Docker or TRANSLATE_IMAGE is missing, the image holds another googletrans
    or httpx release or another worker than this checkout's, a proxy variable carries a
    credential, or the cloud metadata server is reachable on the bridge network the worker runs
    on. Returns the image's probe."""
    hint = (f"Build it LOCALLY from the gbench repository root (gbench never pulls it): "
            f"`{translate_image_build_command()}`.")
    why = ("the Translate API runs googletrans " + GOOGLETRANS_RELEASE + " (api-bank's "
           "requirements.txt pin) in the local image " + TRANSLATE_IMAGE)
    if not shutil.which("docker"):
        raise infra_required("api_bank", f"{why}, and the docker CLI is not on PATH. {hint}",
                             DOCS_URL)
    try:
        insp = subprocess.run(["docker", "image", "inspect", "--format",
                               "{{json .Config.Labels}}", TRANSLATE_IMAGE],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise infra_required("api_bank", f"{why}; the Docker daemon is not reachable "
                             f"({type(e).__name__}). {hint}", DOCS_URL) from e
    if insp.returncode != 0:
        raise infra_required("api_bank", f"{why}, which is not built (or the Docker daemon is not "
                             f"reachable). {hint}", DOCS_URL)
    try:
        labels = json.loads(insp.stdout.strip() or "null") or {}
    except ValueError:
        labels = {}
    wrong = [f"{k}={labels.get(k) or 'unset'}" for k, v in TRANSLATE_IMAGE_LABELS.items()
             if labels.get(k) != v]
    if wrong:
        raise infra_required("api_bank", f"{why}; the image was not built from "
                             f"{TRANSLATE_DOCKERFILE} ({', '.join(wrong)}). {hint}", DOCS_URL)
    proxy_env = translate_proxy_env(dict(os.environ))
    bad = _proxy_credentials(proxy_env)
    if bad:
        raise infra_required(
            "api_bank", f"{', '.join(bad)} would carry a credential into the googletrans worker's "
            "container (it runs the model's Translate calls); gbench passes no key, token or "
            "password into one. Use a proxy without a password in its URL and re-run.", DOCS_URL)
    try:
        probe = _translate_probe()
    except _TranslateWorkerError as e:
        raise infra_required("api_bank", f"{why}, which did not answer its probe ({e}). {hint}",
                             DOCS_URL) from e
    if not os.path.isfile(_TRANSLATE_WORKER):
        raise infra_required("api_bank", f"{why}, whose worker {_TRANSLATE_WORKER} is not in this "
                             "gbench installation; run gbench from its repository.", DOCS_URL)
    with open(_TRANSLATE_WORKER, "rb") as f:
        worker_sha = hashlib.sha256(f.read()).hexdigest()
    if (probe.get("googletrans") != GOOGLETRANS_RELEASE or probe.get("httpx") != HTTPX_RELEASE
            or probe.get("worker_sha256") != worker_sha):
        with _PROBE_LOCK:
            _PROBE.clear()
        raise infra_required(
            "api_bank", f"{why}; the image holds googletrans {probe.get('googletrans')}, httpx "
            f"{probe.get('httpx')} and a worker of sha256 {str(probe.get('worker_sha256'))[:12]} "
            f"(wanted {GOOGLETRANS_RELEASE}, {HTTPX_RELEASE}, {worker_sha[:12]}): rebuild it. "
            f"{hint}", DOCS_URL)
    from .container_guard import require_metadata_blocked
    from .tau_common import ENDPOINT_NETWORK, container_lookups
    require_metadata_blocked("api_bank", TRANSLATE_IMAGE, ENDPOINT_NETWORK, DOCS_URL,
                             resolve=container_lookups(None, [TRANSLATE_HOST],
                                                       lookup_proxy_env(proxy_env)))
    return probe


def _translate_record(probe: Dict[str, Any]) -> Dict[str, Any]:
    """`translate_backend` of the result: where googletrans ran and the run's calls."""
    errors = {k.split(":", 1)[1]: int(v) for k, v in sorted(_TRANSLATE_STATS.items())
              if k.startswith("error:")}
    return {
        "googletrans": probe.get("googletrans"), "httpx": probe.get("httpx"),
        "image": TRANSLATE_IMAGE, "dockerfile": TRANSLATE_DOCKERFILE,
        "host": TRANSLATE_HOST, "raise_exception": True,
        "proxy_variables": sorted(translate_proxy_env(dict(os.environ))),
        "calls": int(_TRANSLATE_STATS.get("calls", 0)),
        "calls_raised": errors,
        "worker_failed": int(_TRANSLATE_STATS.get("worker_failed", 0)),
        "note": ("upstream's googletrans==3.1.0a0 in a container of its own, "
                 "Translator(raise_exception=True): a Google reply other than HTTP 200 raises "
                 "(googletrans' default returns the untranslated text as the translation)"),
    }


# --------------------------------------------------------------------------- #
# Checkout access
# --------------------------------------------------------------------------- #
def _checkout_dir() -> str:
    d = prereqs_path("apibank/api-bank", (os.environ.get(_ENV) or "").strip()) or ""
    hint = (f"Set {_ENV} to the api-bank/ dir of a DAMO-ConvAI checkout at {APIBANK_COMMIT}, e.g.:\n"
            "  git clone --depth 1 --filter=blob:none --sparse "
            "https://github.com/AlibabaResearch/DAMO-ConvAI <dir>\n"
            f"  (cd <dir> && git sparse-checkout set api-bank && git fetch --depth 1 origin "
            f"{APIBANK_COMMIT} && git checkout FETCH_HEAD)\n"
            f"  export {_ENV}=<dir>/api-bank")
    if not d or not os.path.isdir(d):
        raise infra_required("api_bank", f"API-Bank checkout not found ({_ENV}={d!r}). {hint}", DOCS_URL)
    if not os.path.isfile(os.path.join(d, "tool_manager.py")) or not os.path.isdir(os.path.join(d, "apis")):
        raise infra_required("api_bank", f"{_ENV}={d!r} is not the api-bank/ dir (no tool_manager.py/apis/). {hint}", DOCS_URL)
    if checkout_digest(d) != _CHECKOUT_DIGEST:
        raise infra_required(
            "api_bank",
            f"{_ENV}={d!r} does not match the pinned api-bank tree (DAMO-ConvAI {APIBANK_COMMIT}): "
            "evaluator.py / tool_manager.py / api_call_extraction.py / apis / lv3_apis / "
            f"lv1-lv2-samples differ or carry extra files. Use a clean checkout of that commit. {hint}",
            DOCS_URL)
    check_backend_prerequisites(d)
    return d


def checkout_digest(d: str) -> str:
    """The _CHECKOUT_DIGEST of a local api-bank/ dir (bytecode caches ignored)."""
    import hashlib
    paths = [p for p in _DIGEST_FILES if os.path.isfile(os.path.join(d, p))]
    for sub in _DIGEST_DIRS:
        for root, dirs, files in os.walk(os.path.join(d, sub)):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for fn in files:
                if not fn.endswith(".pyc"):
                    paths.append(os.path.relpath(os.path.join(root, fn), d).replace(os.sep, "/"))
    lines = []
    for rel in sorted(paths):
        with open(os.path.join(d, rel), "rb") as f:
            data = f.read()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        lines.append(f"{rel} {blob}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


@contextlib.contextmanager
def _in_checkout(d: str):
    """ToolManager hardcodes ./apis and ./init_database, so run backend work with CWD=checkout
    and the checkout on sys.path. Post-generation scoring is serial, so the global chdir is safe.
    While inside, `googletrans` is the stand-in that runs upstream's pinned release in
    TRANSLATE_IMAGE (_bind_googletrans); sys.modules gets its previous entry back afterwards.
    ToolSearcher's encoder loads on TOOL_SEARCHER_DEVICE (_bind_tool_searcher_device)."""
    if d not in sys.path:
        sys.path.insert(0, d)
    old = os.getcwd()
    os.chdir(d)
    prev = _bind_googletrans()
    encoder = _bind_tool_searcher_device()
    try:
        yield
    finally:
        os.chdir(old)
        if encoder is not None:
            import sentence_transformers
            sentence_transformers.SentenceTransformer = encoder
        if prev is _MISSING:
            sys.modules.pop("googletrans", None)
        else:
            sys.modules["googletrans"] = prev


def _split_by_uppercase(s: str) -> str:
    return "".join(" " + c if c.isupper() else c for c in s).strip()


def _build_messages(system_prompt: str, chat_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """evaluator.py's message construction, verbatim: the prompt as a system message, then one
    message per history item (User -> user, AI -> assistant, API -> a `system` message holding
    `[Api(k='v', ...)] Response: <output>`). Nothing is merged or re-roled."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in chat_history:
        role = item.get("role")
        if role == "User":
            msgs.append({"role": "user", "content": item["text"]})
        elif role == "AI":
            msgs.append({"role": "assistant", "content": item["text"]})
        elif role == "API":
            params = ", ".join("{}='{}'".format(k, v) for k, v in item["param_dict"].items())
            msgs.append({"role": "system", "content": "[{}({})] Response: {}".format(
                item["api_name"], params, str(item["result"]["output"]))})
        else:
            raise ValueError("Invalid chat role: {}".format(role))
    return msgs


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_L12 = [(1, "level-1-given-desc"), (2, "level-2-toolsearcher")]


def _load_l12_samples(d: str) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Build L1/L2 samples from the checkout raw conversations (reusing evaluator.Sample)."""
    samples = []
    with _in_checkout(d):
        import evaluator as ev  # noqa: E402  (checkout module)
        from tool_manager import ToolManager  # noqa: E402
        shared_tm = ToolManager()  # read-only description lookups (state not used)
        desc_cache: Dict[str, str] = {}

        def desc(name: str) -> str:
            if name not in desc_cache:
                desc_cache[name] = shared_tm.get_api_description(name)
            return desc_cache[name]

        for level, subdir in _L12:
            ddir = os.path.join("lv1-lv2-samples", subdir)
            for fname in sorted(f for f in os.listdir(ddir) if f.endswith(".jsonl")):
                with open(os.path.join(ddir, fname), encoding="utf-8") as f:
                    history = [json.loads(line) for line in f]
                for sid, s in enumerate(ev.Sample.from_chat_history(history)):
                    role = s.ground_truth.get("role")
                    if role == "API":
                        api_desc = desc("ToolSearcher") if level == 2 else "\n".join(desc(a) for a in s.apis)
                        prompt, track = API_CALL_PROMPT + api_desc, "api"
                    elif role == "AI":
                        api_desc = "\n".join(desc(a) for a in s.apis)
                        prompt, track = RESPONSE_PROMPT + api_desc, "response"
                    else:
                        continue
                    messages = _build_messages(prompt, s.chat_history)
                    gold = json.dumps({"level": level, "track": track, "file": fname, "sample_id": sid})
                    samples.append((messages, gold, {"category": f"api_bank::L{level}_{track}"}))
    return samples


def _hf(fn: str):
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(_HF_REPO, filename=fn, repo_type="dataset", revision=_HF_REVISION)
    except Exception as e:
        raise infra_required("api_bank", f"{_HF_REPO}@{_HF_REVISION} {fn} could not be "
                             f"downloaded ({e}).", DOCS_URL) from e
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_l3_samples() -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """L3 batch-inference steps from HF (call steps + response steps)."""
    samples = []
    for fn, track in [("test-data/level-3-batch-inf.json", "api"),
                      ("test-data/level-3-batch-inf-response.json", "response")]:
        for r in _hf(fn):
            content = (str(r.get("instruction", "")) + "\n" + str(r.get("input", ""))).strip()
            gold = {"level": 3, "track": track, "sample_id": r.get("sample_id")}
            if track == "api":
                gold["api_id"] = r.get("api_id")
            samples.append(([{"role": "user", "content": content}], json.dumps(gold),
                            {"category": f"api_bank::L3_{track}"}))
    return samples


# --------------------------------------------------------------------------- #
# Scoring - reuse the checkout's Evaluator for L1/L2; replicate lv3_evaluator for L3.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _proxy_env():
    """ALL_PROXY for the scoring phase. apis/translate.py saves os.environ['ALL_PROXY'], sets it to
    '' and assigns the saved value back, which raises TypeError when the variable is unset (None
    cannot be an environment value), so a Translate call could never succeed. An unset ALL_PROXY
    is set to '' (no proxy) while scoring and unset again afterwards; a set one is restored."""
    had = "ALL_PROXY" in os.environ
    old = os.environ.get("ALL_PROXY")
    if not had:
        os.environ["ALL_PROXY"] = ""
    try:
        yield
    finally:
        if had:
            os.environ["ALL_PROXY"] = old
        else:
            os.environ.pop("ALL_PROXY", None)


def _exec_exception(res: Any) -> Optional[str]:
    """The exception of an executed call: the result's `exception`, or the text Evaluator.evaluate
    returned when the call raised (never its "API Name Mismatch" verdict). None when it ran."""
    if isinstance(res, dict):
        exc = res.get("exception")
        return str(exc) if exc else None
    if isinstance(res, str) and res and not res.startswith("API Name Mismatch"):
        return res
    return None


def _gold_call_failure(tool_manager: Any, api_name: str, params: Dict[str, Any]) -> Optional[str]:
    """Why the step's GOLD call fails on the backend now (its exception), or None when it runs."""
    try:
        out = tool_manager.api_call(api_name, **dict(params or {}))
    except Exception as e:  # noqa: BLE001 - any failure of the gold call is the backend's
        return f"{type(e).__name__}: {e}"
    return _exec_exception(out)


def _backend_harness_error(pred_api: Optional[str], gold_api: Optional[str], res: Any,
                           rerun_gold) -> Optional[Dict[str, str]]:
    """A harness error for a failed call step whose API reaches outside the process
    (BACKEND_APIS), or None when the failure is the model's.

    It is the backend's when the model called the gold API and the execution came back with an
    exception that is plainly the harness's or the backend's (_BACKEND_FAILURES: gbench's worker
    container failed or timed out, a transport error, or Google answered 429, 5xx, 403 or 407),
    whatever a gold re-run would give; or with another exception that the gold call itself
    shares now (`rerun_gold()`, its exception): then no call could have passed the step. A call
    to another API, a failure without an exception (a different output) and a failure the gold
    call does not share (for example "Language not supported.") stay the model's outcome."""
    if not gold_api or gold_api not in BACKEND_APIS or pred_api != gold_api:
        return None
    exc = _exec_exception(res)
    if not exc:
        return None
    kind = _BACKEND_FAILURES.get(str(exc))
    if kind is not None:
        first = (str(exc).strip().splitlines() or [""])[0]
        return {"kind": "backend_unavailable", "api": gold_api,
                "detail": redact_text(f"{kind}: {first}")[:300]}
    why = rerun_gold()
    if not why:
        return None
    first = (str(why).strip().splitlines() or [""])[0]
    return {"kind": "backend_unavailable", "api": gold_api,
            "detail": redact_text(first)[:300]}


#: base's statuses of a request that produced no reply, and the harness error each is.
_ENDPOINT_ERRORS = {"FAILED": "endpoint_failure", "TIMEOUT": "endpoint_timeout",
                    "OVER_CONTEXT": "over_context"}


def _endpoint_harness_error(t: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """A harness error for a step whose model request produced no reply (base's FAILED, TIMEOUT
    or OVER_CONTEXT: an endpoint error, a timeout, or a prompt longer than the served context
    window), else None. An empty reply the server did return is the model's outcome."""
    kind = _ENDPOINT_ERRORS.get(str(t.get("status") or ""))
    if kind is None:
        return None
    return {"kind": kind, "api": None,
            "detail": redact_text(str(t.get("error") or "no reply"))[:300]}


def _mark_harness_error(t: Dict[str, Any], err: Dict[str, str]) -> None:
    t["is_correct"] = False
    t["api_bank_score"] = None
    t["harness_error"] = err
    t["scoring_excluded"] = True
    t["status"] = "HARNESS_ERROR"


def _score_traces(d: str, traces: List[Dict[str, Any]]) -> set:
    """Score in place; return the set of L3 dialogue sample_ids that had an errored call step
    (a harness error is not one; _l3_dialogues reads those from the traces)."""
    l3_gt = None  # lazily fetched from HF only if there are L3 traces
    l3_errored: set = set()
    with _in_checkout(d), _proxy_env():
        import evaluator as ev  # noqa: E402
        from tool_manager import ToolManager  # noqa: E402
        from api_call_extraction import get_api_call, parse_api_call  # noqa: E402

        l12_cache: Dict[Tuple[int, str], Any] = {}
        l3_tm = ToolManager("./lv3_apis")  # shared across L3 samples (matches lv3_evaluator)

        for t in traces:
            try:
                blob = json.loads(t.get("gold_answer") or "{}")
            except Exception:
                blob = {}
            level, track = blob.get("level"), blob.get("track")
            t["api_bank_level"], t["api_bank_track"] = level, track
            t["api_bank_sample_id"] = blob.get("sample_id")
            endpoint = _endpoint_harness_error(t)
            if endpoint:
                # the request produced no reply: not the model's step
                _mark_harness_error(t, endpoint)
                continue
            # inline reasoning (<thought>...</thought>) is not part of the reply, as in base
            pred = strip_thinking_tags(t.get("response_text") or "")

            if level in (1, 2):
                key = (level, blob.get("file"))
                if key not in l12_cache:
                    subdir = "level-1-given-desc" if level == 1 else "level-2-toolsearcher"
                    with open(os.path.join("lv1-lv2-samples", subdir, blob["file"]), encoding="utf-8") as f:
                        hist = [json.loads(line) for line in f]
                    l12_cache[key] = ev.Evaluator(ev.Sample.from_chat_history(hist))
                evaluator, sid = l12_cache[key], blob.get("sample_id")
                if track == "api":
                    correct, res, pname = False, None, None
                    call = get_api_call(pred)
                    if call:
                        try:
                            pname = parse_api_call(call)[0]
                        except Exception:
                            pname = None
                        try:
                            correct, res = evaluator.evaluate(sid, call)
                        except Exception as e:  # noqa: BLE001 - a malformed call is the model's
                            correct, res = False, f"{type(e).__name__}: {e}"
                    t["is_correct"] = bool(correct)
                    t["api_bank_score"] = 1.0 if correct else 0.0
                    if not correct:
                        t["api_bank_detail"] = redact_text(str(res))[:500] if res else (
                            "no [ApiName(...)] call in the reply")
                        gt = evaluator.dataset[sid].ground_truth
                        err = _backend_harness_error(
                            pname, gt.get("api_name"), res,
                            lambda: _gold_call_failure(ToolManager(), gt.get("api_name"),
                                                       gt.get("param_dict")))
                        if err:
                            _mark_harness_error(t, err)
                else:
                    try:
                        score = float(evaluator.evaluate(sid, pred)) if pred else 0.0
                    except Exception:
                        score = 0.0
                    t["api_bank_score"] = score
                    t["is_correct"] = score >= 0.5

            elif level == 3:
                if l3_gt is None:
                    l3_gt = _hf("test-data/level-3.json")
                sid = blob.get("sample_id")
                if track == "api":
                    aid = blob.get("api_id")
                    try:
                        gt = l3_gt[sid]["apis"][aid]
                        gt_name, gt_out = gt["api_name"], gt["output"]
                    except Exception:
                        gt, gt_name, gt_out = {}, None, None
                    correct, pname, pres = False, None, None
                    call = get_api_call(pred)
                    if not call or gt_name is None:
                        pres = "no [ApiName(...)] call in the reply" if not call else None
                    else:
                        try:
                            pname, pparams = parse_api_call(call)
                            if pname == "ToolSearcher" and "keywords" in pparams:
                                pparams["keywords"] = _split_by_uppercase(pparams["keywords"])
                            pres = l3_tm.api_call(pname, **pparams)
                            gt_api = l3_tm.init_tool(gt_name)
                            correct = bool(gt_api.check_api_call_correctness(pres, gt_out))
                        except Exception as e:  # noqa: BLE001
                            pres = f"{type(e).__name__}: {e}"
                            correct = False
                    t["is_correct"] = correct
                    t["api_bank_score"] = 1.0 if correct else 0.0
                    if not correct:
                        t["api_bank_detail"] = redact_text(str(pres))[:500] if pres else None

                        def _rerun_l3(gt=gt, gt_name=gt_name):
                            params = dict(gt.get("input") or {})
                            if gt_name == "ToolSearcher" and "keywords" in params:
                                params["keywords"] = _split_by_uppercase(str(params["keywords"]))
                            return _gold_call_failure(l3_tm, gt_name, params)
                        err = _backend_harness_error(pname, gt_name, pres, _rerun_l3)
                        if err:
                            _mark_harness_error(t, err)
                        else:
                            l3_errored.add(sid)
                else:
                    try:
                        gt_resp = l3_gt[sid].get("response", "")
                    except Exception:
                        gt_resp = ""
                    clean = pred.replace("User:", "").replace("AI:", "").strip()
                    try:
                        score = float(ev.calculate_rouge_l_score(gt_resp, clean)) if clean else 0.0
                    except Exception:
                        score = 0.0
                    t["api_bank_score"] = score
                    t["is_correct"] = score >= 0.5
    return l3_errored


def _l3_dialogues(traces: List[Dict[str, Any]], errored: set) -> Tuple[set, set]:
    """(the L3 dialogues with an errored call step, those whose only failed steps are harness
    errors: their outcome is unknown)."""
    harness = {t.get("api_bank_sample_id") for t in traces
               if t.get("api_bank_level") == 3 and t.get("api_bank_track") == "api"
               and t.get("harness_error")}
    return set(errored), harness - set(errored)


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(100.0 * sum(vals) / len(vals), 2) if vals else None


def check_backend_prerequisites(d: str) -> None:
    """ToolSearcher, the gold first step of every Level-2 dialogue and of many Level-3 steps,
    loads TOOL_SEARCHER_MODEL with sentence-transformers on every call: infra_required when it
    cannot be loaded on TOOL_SEARCHER_DEVICE (no sentence-transformers, or the model is neither
    cached nor downloadable).
    The Translate API's googletrans must run as upstream pins it (check_translate_backend).
    _checkout_dir runs both, before any request is sent."""
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(TOOL_SEARCHER_MODEL, device=TOOL_SEARCHER_DEVICE)
    except Exception as e:  # noqa: BLE001
        raise infra_required(
            "api_bank", f"the ToolSearcher backend cannot load {TOOL_SEARCHER_MODEL} "
            f"({type(e).__name__}: {redact_text(str(e))[:200]}); it needs sentence-transformers "
            "and the model (downloaded from the Hugging Face Hub on first use, or in the HF "
            "cache).", DOCS_URL) from e
    check_translate_backend()


def run_api_bank(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run API-Bank: 3 levels x (execution Correctness + response ROUGE-L) + L3 success rate."""
    _TRANSLATE_STATS.clear()
    d = _checkout_dir()

    samples = _load_l12_samples(d) + _load_l3_samples()
    l3_call_steps = sum(1 for s in samples if (s[2] or {}).get("category") == "api_bank::L3_api")
    limit = kwargs.get("limit")
    if limit:
        samples = stratified_sample(samples, limit,
                                    key_fn=lambda s: (s[2] or {}).get("category"), seed="api_bank")
    if not samples:
        raise RuntimeError("api_bank produced no samples")

    limit_plan = base_suite_plan("api_bank", samples, limit, run_attempts(kwargs))
    res = run_eval_suite(
        eval_name="api_bank",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=lambda pred, gold: False,  # placeholder; real scoring is execution-based below
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,  # already limited above
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=API_BANK_SAMPLING,
    )

    traces = res.get("sample_traces", []) or []
    l3_errored, l3_unknown = _l3_dialogues(traces, _score_traces(d, traces) or set())

    def pick(level, track):
        return [t.get("api_bank_score") for t in traces
                if t.get("api_bank_level") == level and t.get("api_bank_track") == track
                and not t.get("scoring_excluded")]

    dims = {
        "L1_call": _mean(pick(1, "api")), "L1_response": _mean(pick(1, "response")),
        "L2_call": _mean(pick(2, "api")), "L2_response": _mean(pick(2, "response")),
        "L3_call": _mean(pick(3, "api")), "L3_response": _mean(pick(3, "response")),
    }
    # lv3_evaluator's Sample Accuracy, (50 - errored dialogues) / 50, assumes every L3 call step
    # was predicted. A dialogue whose only failures are harness errors has no known outcome and
    # leaves the denominator; a run that did not score every L3 call step gets none.
    l3_run = [t for t in traces if t.get("api_bank_level") == 3 and t.get("api_bank_track") == "api"]
    known = L3_DIALOGUES - len(l3_unknown)
    l3_note = None
    if l3_run and len(l3_run) == l3_call_steps and known > 0:
        dims["L3_sample_success"] = round(100.0 * (known - len(l3_errored)) / known, 2)
        if l3_unknown:
            l3_note = (f"over {known} of the {L3_DIALOGUES} dialogues: {len(l3_unknown)} "
                       "had only harness errors among their failed steps")
    else:
        dims["L3_sample_success"] = None
        l3_note = (f"not computed: lv3_evaluator's (50 - errored dialogues) / 50 needs every one "
                   f"of the {l3_call_steps} Level-3 call steps, and this run scored "
                   f"{len(l3_run)}")

    call_traces = [t for t in traces if t.get("api_bank_track") == "api"]
    harness = [t for t in call_traces if t.get("harness_error")]
    scored_calls = [t for t in call_traces if not t.get("scoring_excluded")]
    call_levels = {f"L{lv}_call": dims[f"L{lv}_call"] for lv in (1, 2, 3)}
    headline = _mean_pct([v for v in call_levels.values() if v is not None])

    res["dimension_scores"] = dims
    if l3_note:
        res["L3_sample_success_note"] = l3_note
    # run_eval_suite tallied the placeholder verdicts (all wrong); count the scored call-track
    # verdicts instead, per level (the L{1,2,3}_call numbers). Response ROUGE-L is not a
    # correct/total metric and stays in dimension_scores.
    cats: Dict[str, Dict[str, int]] = {}
    for t in scored_calls:
        if t.get("category"):
            st = cats.setdefault(t["category"], {"correct": 0, "total": 0})
            st["total"] += 1
            st["correct"] += int(bool(t.get("is_correct")))
    res["category_accuracy"] = {c: {**st, "accuracy": round(100.0 * st["correct"] / st["total"], 2)}
                                for c, st in sorted(cats.items())}
    # The headline is over the call track: its steps are the questions shown next to it; the
    # response track (ROUGE-L) is reported apart.
    res["correct_answers"] = sum(1 for t in scored_calls if t.get("is_correct"))
    res["total_questions"] = len(call_traces)
    res["effective_n"] = len(scored_calls)
    res["response_track_samples"] = sum(1 for t in traces if t.get("api_bank_track") == "response")
    all_harness = [t for t in traces if t.get("harness_error")]
    kinds: Dict[str, Any] = {}
    for t in all_harness:
        err = t["harness_error"]
        if err["kind"] == "backend_unavailable":
            by_api = kinds.setdefault("backend_unavailable", {})
            by_api[err["api"]] = by_api.get(err["api"], 0) + 1
        else:
            kinds[err["kind"]] = kinds.get(err["kind"], 0) + 1
    res["harness_errors"] = kinds
    res["n_harness_errors"] = len(all_harness)
    res["backend_apis"] = dict(BACKEND_APIS)
    res["translate_backend"] = _translate_record(dict(_PROBE))
    res["metric"] = ("API-Bank headline = mean execution Correctness over L1/L2/L3 (call track); "
                     "response ROUGE-L + L3 sample-success in dimension_scores. No LLM judge.")
    res["dataset_revision"] = (f"AlibabaResearch/DAMO-ConvAI@{APIBANK_COMMIT} api-bank (L1/L2 + "
                               f"backend); {_HF_REPO}@{_HF_REVISION} (L3)")
    reasons = ["API-Bank reports each level separately; the headline is gbench's unweighted mean "
               "of the three call-correctness levels (dimension_scores holds the per-level "
               "numbers)"]
    if limit:
        reasons.append(f"--eval-limit {limit} (a subset of the samples)")
    if (res.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {res.get('temperature')} set by {res.get('temperature_source')} "
                       "(evaluator.py sends none)")
    if all_harness:
        parts = []
        backend = kinds.get("backend_unavailable") or {}
        if backend:
            parts.append(f"{sum(backend.values())} call step(s) whose API backend (outside "
                         "gbench's process) failed on the gold call too ("
                         + ", ".join(f"{api} x{n}" for api, n in sorted(backend.items())) + ")")
        for kind in ("endpoint_failure", "endpoint_timeout", "over_context"):
            if kinds.get(kind):
                parts.append(f"{kinds[kind]} step(s) whose model request produced no reply "
                             f"({kind})")
        reasons.append("harness errors, not the model's outcome and left out of the level "
                       "means: " + "; ".join(parts))
        levels_lost = [lv for lv, v in call_levels.items() if v is None
                       and any(t.get("api_bank_level") == int(lv[1]) for t in call_traces)]
        if levels_lost:
            reasons.append(f"{', '.join(levels_lost)} has no scored step (every one was a harness "
                           "error), so the headline averages the other levels")
    res["leaderboard_comparable"] = False
    res["leaderboard_comparable_reason"] = "; ".join(reasons)
    if all_harness:
        why = f"{len(all_harness)} step(s) with a harness error ({', '.join(sorted(kinds))})"
        if res.get("status") == "success":
            res["status"] = "completed_with_errors"
        res["status_reason"] = "; ".join(x for x in (res.get("status_reason"), why) if x)
    if headline is None:
        # nothing on the call track was scored: no number to report
        res["accuracy"] = None
        res["status"] = "failed"
        res["accuracy_withheld_reason"] = (
            "no call-track step was scored" + (f": every one of the {len(call_traces)} was a "
                                               "harness error" if call_traces else ""))
    else:
        res["accuracy"] = headline
    res["eval_limit_plan"] = limit_plan
    return res


def _mean_pct(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 2) if vals else None
