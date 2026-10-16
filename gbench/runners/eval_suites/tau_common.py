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

"""Shared full-environment harness for tau2 / tau3 (Sierra tau2-bench).

The canonical tau-bench is a multi-turn *environment* benchmark: an LLM user and the agent converse
over several turns, tools mutate a domain database, and each task is scored by the simulator's
oracle (final DB-state check) plus an nl_assertion LLM judge, yielding a per-task reward. It cannot
be measured from a single response, so gbench runs the REAL simulator.

Where it runs. tau2-bench pins its own dependencies (uv.lock; e.g. litellm 1.81.11, while the gbench
environment carries a newer litellm) and its banking_knowledge shell tool needs Anthropic's
sandbox-runtime with bubblewrap, so the simulator runs in a LOCAL image built from
docker/tau2.Dockerfile: a tau2-bench checkout at TAU2_BENCH_REVISION installed from its own uv.lock.
`run_tau_suite` (host side, called by tau2.py / tau3.py) checks that the image exists, mounts this
package directory read-only into it and runs `run_tau_env` there; the result and the per-task
traces come back through bind mounts. A missing Docker or image hard-errors (infra_required) with
the build command; there is no in-process fallback. The container runs on docker's default bridge
network, never the host network, and only after container_guard.require_metadata_blocked has
found the cloud metadata server blocked there (the host's DOCKER-USER rule). The container holds
no key: its Gemini calls go to gbench's Gemini relay and its agent calls to gbench's endpoint
relay, both threads of the gbench process on the bridge gateway (`_bridge_services`, which checks
that a container reaches them). The endpoint relay runs for every run, keyless too (2026-10-13):
it adds a real endpoint key when there is one and the run's --max-output-tokens on each agent
request that carries none (swe_agentic.AGENT_MAX_TOKENS_SUITES). The container keeps general
internet access, as upstream's harness has; the agent's shell commands (banking_knowledge) run in sandbox-runtime with upstream's
settings, which allow no network.
Inside the image, `run_tau_env` refuses to run unless the imported tau2 is the pinned, unmodified
checkout and the installed packages match its uv.lock (`check_tau_env_prerequisites`).

Design (the upstream tau2-bench `run_domain` setup, with the substitutions listed below):
  * Three distinct LLM roles:
      - agent  = the model under test, routed to the gbench endpoint via LiteLLM's
                 `openai/<model>` with `api_base` in its llm_args and the endpoint key added to
                 each call (`_agent_guard`), sampled by gbench's rule: temperature 0.0
                 without --thinking (as upstream's DEFAULT_LLM_ARGS_AGENT) and 1.0 with it,
                 unless --temperature / GBENCH_<SUITE>_TEMPERATURE set one;
      - user   = user simulator: gbench's Gemini cascade (base.judge_cascade(): the models of
                 GBENCH_JUDGE_MODELS, else GBENCH_JUDGE_MODEL, else base.GEMINI_CASCADE, in
                 order), or the one gemini/<model> GBENCH_TAU2_USER_LLM names;
                 src/tau2/config.py defaults to gpt-4.1-2025-04-14, and gpt-5.2 is the model
                 docs/leaderboard-submission.md recommends and Sierra's 2026 standard text
                 entries use;
      - judge  = nl_assertion evaluator: the same cascade, or GBENCH_TAU2_EVAL_LLM's one
                 gemini/<model>; upstream's DEFAULT_LLM_NL_ASSERTIONS is gpt-4.1-2025-04-14.
    Every external model a tau run calls goes through the Gemini API: a user simulator or judge of
    any other provider hard-errors (`check_role_providers`, `check_role_credentials`).
    Neither is the model under test: the agent is the local endpoint. A gemini/ call goes
    through its role's models in order (`_build_gemini_cascade`): each model gets the call's own
    retry budget (tau2's num_retries, as upstream), LiteLLM's own retries are off so they never
    re-enter the cascade, and the next model is asked only when one keeps failing. Every call
    sends upstream's arguments only (a temperature; LiteLLM's default thinking level for Gemini 3
    models is removed, `_install_gemini_param_guard`), the same for every model, and the result
    records which model answered, per role and domain, and per NL-assertion verdict. The agent
    (openai/<local>) is never cascaded.
  * We deliberately DO NOT set OPENAI_API_BASE globally: tau2's evaluator/user also
    issue LLM calls, and redirecting all OpenAI traffic to the endpoint would corrupt
    grading. Only the agent is routed, per-call.
  * Retries on an empty or malformed reply, and markdown-fence stripping, apply ONLY to the
    gemini/ user simulator and judge calls (part of running the Gemini cascade in place of
    upstream's models). The agent's requests are sent once and its replies are used as they
    are, exactly as upstream tau2 does (its litellm `num_retries` covers transport errors only).
  * banking_knowledge (tau3) retrieval has one fixed configuration: upstream's default,
    `alltools` (BM25 + dense search + shell), with one substitution: the dense search embeds with
    gemini-embedding-001 through the Gemini API (`GeminiEmbedder`, batchEmbedContents) instead of
    OpenAI text-embedding-3-large. gbench registers it with tau2 as `alltools-gemini`, a copy of
    upstream's alltools variant whose dense pipeline names the Gemini embedder, and nothing else
    differs (`_register_gemini_retrieval`). No knob selects another configuration. GEMINI_API_KEY
    is its prerequisite; without it the suite hard-errors (`check_retrieval_credentials`). A
    Gemini embedding call that still fails after its retries during a simulation is a harness
    error, never the agent's outcome: the simulation attempt stops instead of showing the agent a
    tool error, tau2 re-runs it as it re-runs any failed attempt, and the failure is counted per
    domain, makes the status completed_with_errors and is named in
    leaderboard_comparable_reason (`_embedding_failure_guard`).
  * A simulation that ends in an infrastructure error (tau2's run_with_retry records one when
    every attempt of a simulation raised) is never scored: tau2's compute_metrics leaves it out
    of pass^k and gbench leaves it out of the mean reward and of the evaluable counts
    (`total_questions`, `category_accuracy`). gbench reads where the error came from in the
    traceback tau2 records (`_infra_error_source`). One from a harness failure (an outage of the
    Gemini user simulator or judge, of the model endpoint, failed Gemini embedding calls, ...) is
    a harness error, never the agent's outcome, counted per domain in `harness_errors`. One
    raised on the agent's own turn (an agent reply tau2 rejects, such as a message with neither
    text nor tool calls, or an agent request over the endpoint's context window) is not a harness
    failure: tau2 leaves it out as upstream does, and gbench counts it per domain in
    `agent_turn_errors`. Either kind, with its sources and error classes, makes the status
    completed_with_errors and is named in leaderboard_comparable_reason.
  * Credentials never reach the model under test or the saved files. The container runs the
    agent's tool calls (and tau3's shell commands, as the same user), so it holds no key: the
    Gemini gate runs on the host, gbench's Gemini relay adds the Gemini key on the host and the
    container gets only its per-run placeholder and address (`_bridge_services`,
    `check_gemini_route`), and a real endpoint key stays with gbench's endpoint relay, which
    adds it (the agent always reaches the endpoint through that relay). An endpoint key that is
    no real key (EMPTY, dummy, ...) travels under a private name
    that the driver removes from the environment before tau2 starts, and it is added to each
    agent call rather than stored in llm_args (tau2 records llm_args in every results.json). The
    shell tool's commands run with every credential removed from their environment, tool results
    and agent-bound messages are redacted, and results.json, the result and the streamed log are
    written with every credential value replaced by `[REDACTED]` (`_install_credential_guards`,
    `_redact_tree`).

The version is pinned: the imported tau2 package must come from a sierra-research/tau2-bench git
checkout at TAU2_BENCH_REVISION with unmodified src/ and data/ (`check_tau2_revision`), with the
packages of its uv.lock installed (`check_tau2_dependencies`); anything else hard-errors
(infra_required). The image satisfies both by construction.

The run protocol is the leaderboard's: every task of each domain's `base` split, 4 trials per task,
tau2's own pass^k (compute_metrics) per domain. The headline `accuracy` is pass^1 of the
leaderboard track: for tau2 the unweighted mean of retail, airline and telecom (the tau2-bench
track's Overall at the pinned revision), for tau3 banking_knowledge (the tau3-banking track).
The other knobs keep upstream's src/tau2/config.py defaults (agent and user temperature 0.0, 200
max steps, 10 max errors, seed 300; a --thinking run samples the agent at 1.0 instead) and are overridable via env (GBENCH_TAU2_TEMPERATURE /
GBENCH_TAU2_TOP_K / GBENCH_TAU2_TOP_P / GBENCH_TAU2_NUM_TRIALS / GBENCH_TAU2_MAX_STEPS /
GBENCH_TAU2_MAX_ERRORS / GBENCH_TAU2_SEED / GBENCH_TAU2_USER_LLM / GBENCH_TAU2_EVAL_LLM /
GBENCH_TAU2_USER_TEMPERATURE); any override that differs from a leaderboard submission is named
in `leaderboard_comparable_reason`. Every knob here uses the canonical `GBENCH_TAU2_<KNOB>` name;
the bare `TAU2_<KNOB>` names still work as deprecated aliases (canonical wins; a one-time
deprecation warning is logged when a legacy name is used). `run_tau_suite` forwards these knobs
into the image.

Every run writes its simulations to a directory of its own, named after the eval, the model, the
thinking mode, a hash of the run configuration and a per-invocation id, and tau2 is never allowed
to resume from an existing file (auto_resume off, and its console prompt answered "n"). By
default that directory lives under `<run results dir>/tau_traces/` and is kept as the run's
per-task traces (messages + reward breakdown); `GBENCH_TAU2_SAVE_TRACES=<path>` moves it and
`GBENCH_TAU2_SAVE_TRACES=` (empty) keeps it in a temporary directory that is removed at the end.
See `_resolve_tau_trace_dir`, `run_tau_suite` and `run_tau_env`.
"""

import contextlib
import hashlib
import importlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import suite_env
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

#: The tau2-bench revision gbench runs: sierra-research/tau2-bench main as of 2026-09-17 (package
#: version 1.0.1). Its data/tau2/domains tree is byte-identical to the v1.0.1 release tag
#: (TAU2_BENCH_V101_COMMIT); the commits in between change the voice stack, the batch runner
#: (rebuilt around a work queue), the submission docs and the leaderboard website, which at this
#: revision splits the text leaderboard into a tau2-bench track (retail, airline, telecom) and a
#: tau3-banking track (banking_knowledge).
TAU2_BENCH_REVISION = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"
TAU2_BENCH_VERSION = "1.0.1"
TAU2_BENCH_V101_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU2_BENCH_REPO = "https://github.com/sierra-research/tau2-bench"

#: The local image the suites run in (docker/tau2.Dockerfile, tagged with the revision's prefix);
#: GBENCH_TAU2_IMAGE names another tag. The Dockerfile's `gbench.tau2.revision` label must equal
#: TAU2_BENCH_REVISION.
TAU2_IMAGE_DEFAULT = f"gbench-tau2:{TAU2_BENCH_REVISION[:8]}"
TAU2_DOCKERFILE = "docker/tau2.Dockerfile"
TAU2_IMAGE_CONTEXT = "docker/tau2_srt"
TAU2_IMAGE_LABEL = "gbench.tau2.revision"

#: Leaderboard tracks at TAU2_BENCH_REVISION (web/leaderboard/src/components/Leaderboard.jsx):
#: the tau2-bench track's Overall is the unweighted mean of these three domains' pass^k, and
#: banking_knowledge is ranked on its own in the tau3-banking track.
CORE_DOMAINS = ("retail", "airline", "telecom")
KNOWLEDGE_DOMAINS = ("banking_knowledge",)

_PATCHED = False   # robustness patches are process-global; apply once
#: Which of those patches are active in this process: `completion` re-sends a litellm request
#: whose reply is empty or carries empty/unparseable tool arguments (up to 3 attempts in all);
#: `generate` strips markdown fences from generated text (and records the role of each call).
#: Both act ONLY on gemini/ calls (the Gemini user simulator and judge), never on the agent under
#: test.
_RETRY_PATCHES: Dict[str, bool] = {"completion": False, "generate": False}
_QUIETED = False   # cosmetic-log suppression; apply once
_TAU_PROGRESS_DESC = "tau"   # tqdm bar label; run_tau_env sets it per domain


# --------------------------------------------------------------------------------------------- #
# Credentials: forwarded only where used, never visible to the model under test, never saved
# --------------------------------------------------------------------------------------------- #
#: The environment variable that carries the model endpoint's key (the host's OPENAI_API_KEY, the
#: key gbench's other suites send to the endpoint; EMPTY when unset) into the image. `run_tau_env`
#: removes it from the environment before tau2 starts and adds it to each agent call
#: (`_agent_guard`), so it is in no tool's environment, no llm_args and no trace. The image has no
#: OPENAI_API_KEY: every external model a run calls (user simulator, judge, tau3's retrieval
#: embeddings) goes through the Gemini API with GEMINI_API_KEY.
_AGENT_KEY_ENV = "GBENCH_TAU_AGENT_KEY"

#: The one provider whose credentials enter the image, and the only provider of the user simulator
#: and the judge.
_GEMINI_PROVIDER = "gemini"

#: Variable names treated as credentials: removed from the shell tool's environment, and their
#: values (split on commas / semicolons / whitespace) redacted from traces, results and logs.
_SECRET_NAME_RE = re.compile(
    r"(^|_)(API_?KEYS?|KEYS?|TOKENS?|SECRETS?|PASSWORDS?|PASSWD|CREDENTIALS?|CONNECTION_STRING)$",
    re.IGNORECASE)
_URL_USERINFO_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://([^/\s@]+)@")
#: Values shorter than this are not treated as secrets (placeholders such as EMPTY or dummy);
#: redacting them would corrupt ordinary text.
_MIN_SECRET_LEN = 8
REDACTED = "[REDACTED]"
#: The credential values of the run in progress (set by `run_tau_env`), longest first.
_SECRETS: List[str] = []


def _is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME_RE.search(name or ""))


def _secret_values(environ: Optional[Dict[str, str]] = None, extra: Tuple[str, ...] = ()) -> List[str]:
    """Every credential value in `environ` (credential-named variables, and the user:password part
    of any URL), plus `extra`, longest first, so a key is replaced before any shorter overlap."""
    env = os.environ if environ is None else environ
    found = set()

    def add(value: Optional[str]) -> None:
        value = (value or "").strip()
        if len(value) >= _MIN_SECRET_LEN:
            found.add(value)

    for name, value in env.items():
        if not value:
            continue
        if _is_secret_name(name):
            add(value)
            for part in re.split(r"[,;\s]+", value):
                add(part)
        for match in _URL_USERINFO_RE.finditer(value):
            add(match.group(1))
            add(match.group(1).split(":", 1)[-1])
    for value in extra:
        add(value)
    return sorted(found, key=lambda v: (-len(v), v))


def _redact_text(text: Optional[str], secrets: List[str]) -> Optional[str]:
    """`text` with every secret (and its JSON-escaped form) replaced by [REDACTED]."""
    if not text or not secrets:
        return text
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
        escaped = json.dumps(secret)[1:-1]
        if escaped != secret and escaped in text:
            text = text.replace(escaped, REDACTED)
    return text


def _redact(obj: Any, secrets: List[str]) -> Any:
    """A copy of a JSON-like value with every secret replaced in its strings."""
    if not secrets:
        return obj
    if isinstance(obj, str):
        return _redact_text(obj, secrets)
    if isinstance(obj, dict):
        return {k: _redact(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_redact(v, secrets) for v in obj)
    return obj


def _redact_tree(root: Optional[str], secrets: List[str]) -> List[str]:
    """Rewrite every file under `root` that contains a secret with it replaced; returns the files
    rewritten. The last guard on what stays in the trace directory."""
    if not root or not secrets or not os.path.isdir(root):
        return []
    needles = []
    for secret in secrets:
        needles.append((secret.encode("utf-8"), REDACTED.encode("utf-8")))
        escaped = json.dumps(secret)[1:-1]
        if escaped != secret:
            needles.append((escaped.encode("utf-8"), REDACTED.encode("utf-8")))
    rewritten = []
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            new = data
            for needle, repl in needles:
                if needle in new:
                    new = new.replace(needle, repl)
            if new != data:
                tmp = path + ".redact.tmp"
                with open(tmp, "wb") as fh:
                    fh.write(new)
                os.replace(tmp, path)
                rewritten.append(path)
    return rewritten


def _tool_environment(environ: Optional[Dict[str, str]] = None,
                      secrets: Optional[List[str]] = None) -> Dict[str, str]:
    """The environment a tool command runs with: the process environment minus every credential
    (credential-named variables, and any variable whose value holds a credential value)."""
    env = os.environ if environ is None else environ
    secrets = _SECRETS if secrets is None else secrets
    return {k: v for k, v in env.items()
            if not _is_secret_name(k) and not any(s in v for s in secrets)}


class _ScrubbedSubprocess:
    """Stands in for the `subprocess` module inside tau2's sandbox_manager: the shell tool's
    `srt --settings <file> <command>` runs with `_tool_environment()` instead of inheriting the
    driver's environment (upstream passes that whole environment, provider keys included, into
    the sandbox, where `printenv` would show them to the agent). Everything else is the real
    module."""

    def __init__(self, real: Any):
        self._real = real

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("env") is None:
            kwargs["env"] = _tool_environment()
        return self._real.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _install_tool_env_scrub() -> None:
    """Make tau2's sandbox_manager run shell commands without credentials (idempotent)."""
    import tau2.knowledge.sandbox_manager as _sm
    if not isinstance(getattr(_sm, "subprocess", None), _ScrubbedSubprocess):
        _sm.subprocess = _ScrubbedSubprocess(subprocess)


#: The agent's route for the run in progress: {"model", "api_base", "api_key"} (`run_tau_env`).
_AGENT_ROUTE: Dict[str, str] = {}

#: What the agent's requests carried, counted at tau2's completion chokepoint (`_agent_guard`)
#: for the run in progress: {"requests": n, "enable_thinking": {"True"|"False"|"None": n},
#: "timeout_s": {"<value>": n}}. litellm 1.81.11 (the image's) sends extra_body's fields in the
#: request's JSON body and uses `timeout` as the HTTP client's timeout (in-image mock check,
#: 2026-10-11), so this is the request shape on the wire.
_AGENT_WIRE: Dict[str, Any] = {}
_AGENT_WIRE_LOCK = threading.Lock()

#: The litellm version in the tau2 image (gbench-tau2), recorded with the agent's timeout.
TAU2_IMAGE_LITELLM_VERSION = "1.81.11"
#: tau2's generate() sends num_retries=3 (src/tau2/config.py DEFAULT_MAX_RETRIES), which LiteLLM
#: 1.81 passes to the OpenAI client as max_retries; gbench leaves it as upstream sets it.
_TAU2_AGENT_RETRIES = 3


def agent_client_timeout() -> Dict[str, Any]:
    """The request timeout gbench gives the agent's (the model under test's) LiteLLM calls, as the
    result records it (`model_client_timeout`). Only the agent: the Gemini user simulator and
    judge keep their own settings.

    2026-10-10 investigation: tau2's generate() calls litellm.completion with no timeout, and
    LiteLLM 1.81.11 then uses 600 s per request (main.py completion: `timeout or
    kwargs.get("request_timeout", 600)`; the REQUEST_TIMEOUT variable is not read there), with
    tau2's num_retries=3 as the OpenAI client's retries, so an agent turn that needed more than
    600 s of decode was dropped and asked again: a hidden cap on the turn's length. tau2's agent
    sends no max_tokens, so the timeout is base.request_timeout_s of the served context window
    (base.UNCAPPED_TIMEOUT_TOKENS when the server reports none); the retries stay upstream's.
    Since 2026-10-13 gbench's endpoint relay fills the run's --max-output-tokens into those
    requests (`_bridge_services`); the window-sized timeout is kept: it is at least as long as
    one for the filled value, so it never cuts a reply the relay lets through."""
    from .base import UNCAPPED_TIMEOUT_TOKENS, get_run_knob, request_timeout_s
    tokens = int(get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)
    return {"client": "litellm (tau2 generate(), the agent's calls only)",
            "version": TAU2_IMAGE_LITELLM_VERSION,
            "timeout_s": request_timeout_s(tokens),
            "retries": _TAU2_AGENT_RETRIES,
            # in-image mock check 2026-10-11: with num_retries=3, LiteLLM 1.81.11 makes 7 HTTP
            # attempts of one generate() call whose every attempt times out
            "http_attempts_if_every_attempt_times_out": 7,
            "source": (f"llm_args_agent timeout = base.request_timeout_s(served context window="
                       f"{tokens}); retries: tau2's num_retries=3, unchanged"),
            "replaces": "LiteLLM 1.81.11 default 600 s per request",
            "hidden_cap_removed": True}


def _note_agent_request(kwargs: Dict[str, Any]) -> None:
    """Count one agent request's shape (`_AGENT_WIRE`)."""
    extra = kwargs.get("extra_body") if isinstance(kwargs.get("extra_body"), dict) else {}
    ctk = extra.get("chat_template_kwargs") if isinstance(extra.get("chat_template_kwargs"),
                                                          dict) else {}
    thinking = str(ctk.get("enable_thinking")) if "enable_thinking" in ctk else "None"
    timeout = str(kwargs.get("timeout"))
    with _AGENT_WIRE_LOCK:
        _AGENT_WIRE["requests"] = int(_AGENT_WIRE.get("requests", 0)) + 1
        by = _AGENT_WIRE.setdefault("enable_thinking", {})
        by[thinking] = by.get(thinking, 0) + 1
        tos = _AGENT_WIRE.setdefault("timeout_s", {})
        tos[timeout] = tos.get(timeout, 0) + 1


def agent_wire_record(enable_thinking: bool) -> Dict[str, Any]:
    """The run's agent request shape (`_AGENT_WIRE`) and `thinking_applied`: with --thinking,
    every agent request carried enable_thinking true; without, none carried true (a no-think
    request sends no switch, as upstream: the served model's default, off for Gemma 4)."""
    with _AGENT_WIRE_LOCK:
        wire = json.loads(json.dumps(_AGENT_WIRE))
    n = int(wire.get("requests", 0))
    by = wire.get("enable_thinking") or {}
    if enable_thinking:
        applied = bool(n) and int(by.get("True", 0)) == n
    else:
        applied = bool(n) and not by.get("True")
    return {"agent_request_shape": {**wire, "requests": n,
                                    "field": "extra_body.chat_template_kwargs.enable_thinking",
                                    "no_think_sends": "no switch (upstream's request; the "
                                                      "served model's default)"},
            "thinking_applied": applied}


def reset_agent_wire() -> None:
    with _AGENT_WIRE_LOCK:
        _AGENT_WIRE.clear()


def _agent_guard(inner):
    """Wrap a litellm `completion`: a call to the agent's model at the agent's api_base gets the
    endpoint key (kept out of llm_args, which tau2 writes into every results.json) and has every
    credential value redacted from the messages it sends. Every other call passes unchanged."""

    def completion(*args: Any, **kwargs: Any) -> Any:
        route = dict(_AGENT_ROUTE)
        model = kwargs.get("model", args[0] if args else None)
        if route and model == route.get("model") and kwargs.get("api_base") == route.get("api_base"):
            kwargs["api_key"] = route.get("api_key") or "EMPTY"
            _note_agent_request(kwargs)
            if _SECRETS and "messages" in kwargs:
                kwargs["messages"] = _redact(kwargs["messages"], list(_SECRETS))
        return inner(*args, **kwargs)

    completion._gbench_agent_guard = True
    return completion


class _RedactingJson:
    """Stands in for the `json` module inside tau2's runner.checkpoint, which appends every
    finished simulation to results.json with json.dump: the text is written with every credential
    value of the run replaced."""

    def __init__(self, real: Any):
        self._real = real

    def dump(self, obj: Any, fp: Any, **kwargs: Any) -> None:
        fp.write(self.dumps(obj, **kwargs))

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        return _redact_text(self._real.dumps(obj, **kwargs), list(_SECRETS))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _redacting_save(orig):
    """Wrap tau2's Results.save (the first write of results.json) to write the JSON form with
    every credential value replaced; the directory form (voice runs only) is redacted after
    writing."""

    def save(self, path, format="json"):
        secrets = list(_SECRETS)
        if format == "json" and secrets:
            from pathlib import Path
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                fh.write(_redact_text(self.model_dump_json(indent=2), secrets))
            return None
        out = orig(self, path, format=format)
        _redact_tree(os.path.dirname(os.path.abspath(str(path))), secrets)
        return out

    save._gbench_redacting = True
    return save


def _redacting_get_response(orig):
    """Wrap tau2's Environment.get_response so a tool result never carries a credential value to
    the agent (or into the trajectory)."""

    def get_response(self, message):
        resp = orig(self, message)
        content = getattr(resp, "content", None)
        if isinstance(content, str) and _SECRETS:
            redacted = _redact_text(content, list(_SECRETS))
            if redacted != content:
                resp.content = redacted
        return resp

    get_response._gbench_redacting = True
    return get_response


def _install_credential_guards(domain: str) -> None:
    """Install, once per process, the patches that keep credentials away from the model under test
    and out of the saved files: the per-call endpoint key and message redaction on the agent's
    litellm calls, redaction of tool results and of results.json writes, and (banking_knowledge)
    the credential-free shell environment and the guard that turns a failed Gemini embedding call
    into a harness error (`_embedding_failure_guard`). Raises if any cannot be installed: without
    them the agent's calls would carry no endpoint key, credentials could reach the traces, and a
    Gemini outage would reach the agent as its own tool error."""
    import litellm
    import tau2.runner.checkpoint as _ckpt
    import tau2.utils.llm_utils as _llm
    from tau2.data_model.simulation import Results
    from tau2.environment.environment import Environment

    if not getattr(_llm.completion, "_gbench_agent_guard", False):
        _llm.completion = _agent_guard(_llm.completion)
    for holder in (litellm, getattr(litellm, "main", None)):
        if holder is not None and hasattr(holder, "completion") \
                and not getattr(holder.completion, "_gbench_agent_guard", False):
            holder.completion = _agent_guard(holder.completion)
    if not getattr(Results.save, "_gbench_redacting", False):
        Results.save = _redacting_save(Results.save)
    if not isinstance(getattr(_ckpt, "json", None), _RedactingJson):
        _ckpt.json = _RedactingJson(_ckpt.json)
    if not getattr(Environment.get_response, "_gbench_redacting", False):
        Environment.get_response = _redacting_get_response(Environment.get_response)
    if domain == "banking_knowledge":
        _install_tool_env_scrub()
        if not getattr(Environment.get_response, "_gbench_embedding_guard", False):
            Environment.get_response = _embedding_failure_guard(Environment.get_response)


def _role_provider(model: Optional[str]) -> str:
    """The provider LiteLLM routes a role's model to, as named in errors: the `<provider>/`
    prefix, else anthropic for a bare claude-* name, vertex_ai for a bare gemini-* name (LiteLLM
    1.81.11 routes those to Vertex AI, not to the Gemini API) and openai for any other bare name
    (gpt-4.1-2025-04-14, gpt-5.2, ...). Only gemini/ is accepted. Inside the image
    `check_role_credentials` also asks LiteLLM itself."""
    m = (model or "").strip().lower()
    if "/" in m:
        return m.split("/", 1)[0]
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "vertex_ai"
    return "openai"


def _unsupported_role(role: str, model: str, provider: str) -> str:
    return (f"the {role} {model} is served by the LiteLLM provider '{provider}', but gbench calls "
            "the user simulator and the judge only through the Gemini API; set "
            "GBENCH_TAU2_USER_LLM / GBENCH_TAU2_EVAL_LLM to a gemini/<model>, or leave them unset "
            "for gbench's Gemini cascade (GBENCH_JUDGE_MODELS). See docs/evals/tau2.md.")


#: The two Gemini roles of a tau run: {role key: (name in messages, canonical knob, legacy
#: knob)}. Each role's models are its knob's one model when that is set (no fallback), else
#: gbench's Gemini cascade (base.judge_cascade(): GBENCH_JUDGE_MODELS, else GBENCH_JUDGE_MODEL,
#: else base.GEMINI_CASCADE), tried in that order (`_build_gemini_cascade`).
_USER_ROLE = "user_simulator"
_JUDGE_ROLE = "nl_assertion_judge"
_ROLE_KNOBS = {_USER_ROLE: ("user simulator", "GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM"),
               _JUDGE_ROLE: ("NL-assertion judge", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM")}


def _bare_model(name: Optional[str]) -> str:
    """A Gemini model name without LiteLLM's `gemini/` prefix or the API's `models/` prefix."""
    name = (name or "").strip()
    for prefix in ("gemini/", "models/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


#: Set by the host in the tau2 container next to GBENCH_JUDGE_MODELS (`_bridge_services`): where
#: the host's cascade came from (base.judge_cascade_source()). The container always receives the
#: list in GBENCH_JUDGE_MODELS, so its own reading would always say "GBENCH_JUDGE_MODELS".
CASCADE_SOURCE_ENV = "TAU_JUDGE_CASCADE_SOURCE"
_CASCADE_SOURCES = ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "default")


def _cascade_from(env: Dict[str, str]) -> Tuple[List[str], str]:
    """(models, source) of gbench's Gemini cascade as `env` sets it, by base.judge_cascade()'s
    rule (GBENCH_JUDGE_MODELS, else GBENCH_JUDGE_MODEL, else base.GEMINI_CASCADE) applied to the
    mapping; source is "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL" or "default". In the tau2
    image the process environment's source is the host's (CASCADE_SOURCE_ENV)."""
    from .base import (GEMINI_CASCADE, JUDGE_MODEL_ENV, JUDGE_MODELS_ENV, judge_cascade,
                       judge_cascade_source)
    if env is os.environ:
        source = judge_cascade_source()
        handed = (os.environ.get(CASCADE_SOURCE_ENV) or "").strip()
        if _in_image() and handed in _CASCADE_SOURCES:
            source = handed
        return [_bare_model(m) for m in judge_cascade()], source
    models = [_bare_model(m) for m in (env.get(JUDGE_MODELS_ENV) or "").split(",")
              if _bare_model(m)]
    if models:
        return models, JUDGE_MODELS_ENV
    single = _bare_model(env.get(JUDGE_MODEL_ENV))
    if single:
        return [single], JUDGE_MODEL_ENV
    return list(GEMINI_CASCADE), "default"


def _role_override(role: str, environ: Optional[Dict[str, str]] = None
                   ) -> Tuple[Optional[str], Optional[str]]:
    """(model, knob name) when the role's knob (GBENCH_TAU2_USER_LLM / GBENCH_TAU2_EVAL_LLM, or
    its legacy alias) is set, else (None, None)."""
    env = os.environ if environ is None else environ
    _label, canonical, legacy = _ROLE_KNOBS[role]
    for name in (canonical, legacy):
        value = (env.get(name) or "").strip()
        if value:
            if name == legacy and env is os.environ:
                suite_env(canonical, legacy)          # logs the one-time deprecation warning
            return value, name
    return None, None


def _role_cascade(role: str, environ: Optional[Dict[str, str]] = None) -> List[str]:
    """The `gemini/<model>` names the role tries, in order: its knob's model alone when set, else
    every model of gbench's Gemini cascade. A knob naming another provider is returned as it is
    (check_role_providers refuses it)."""
    env = os.environ if environ is None else environ
    override, _knob = _role_override(role, env)
    if override:
        return [override]
    return [f"gemini/{m}" for m in _cascade_from(env)[0]]


def _role_source(role: str, environ: Optional[Dict[str, str]] = None) -> str:
    """Where the role's models come from: its knob's name, or the cascade's source
    ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL" or "default" for base.GEMINI_CASCADE)."""
    env = os.environ if environ is None else environ
    override, knob = _role_override(role, env)
    return knob if override else _cascade_from(env)[1]


def _role_models(environ: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """(role, model) of the user simulator and the NL-assertion judge, read from `environ`: the
    first model each tries (`_role_cascade`)."""
    return [(_ROLE_KNOBS[role][0], _role_cascade(role, environ)[0])
            for role in (_USER_ROLE, _JUDGE_ROLE)]


def check_role_providers(environ: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """The user simulator and the judge must be gemini/ models (the Gemini API is the only
    provider a run calls); checked on the host before the container starts."""
    for role, model in _role_models(environ):
        provider = _role_provider(model)
        if provider != _GEMINI_PROVIDER:
            return False, _unsupported_role(role, model, provider)
    return True, ""


def _in_image() -> bool:
    """True when this module runs inside the gbench-tau2 image (imported under _IMAGE_PACKAGE)."""
    return __name__.split(".", 1)[0] == _IMAGE_PACKAGE


def _litellm_provider(model: str) -> Optional[str]:
    """Inside the image: the provider tau2's own LiteLLM (the uv.lock version, loaded by tau2 by
    the time the prerequisites run) routes `model` to, or "" when it cannot route the name.
    None elsewhere, where another LiteLLM version may be loaded."""
    litellm = sys.modules.get("litellm") if _in_image() else None
    if litellm is None or not hasattr(litellm, "get_llm_provider"):
        return None
    try:
        return str(litellm.get_llm_provider(model)[1] or "")
    except Exception:                                            # noqa: BLE001
        return ""


def _quiet_tau2_noise() -> None:
    """Quiet tau2's redundant output so gbench logs stay readable. Two parts:

    1. Silence two cosmetic ERROR logs *at the source* (a loguru handler-filter does not
       survive: tau2's batch runner calls `logger.remove(); logger.add(...)` when it runs,
       wiping any filter). We no-op the emitting functions - neither value affects scoring:
         - get_response_cost: litellm can't price a custom model name -> 0.0;
         - get_commit_hash: `git rev-parse HEAD` fails off a repo -> "unknown".
    2. Mute tau2's rich console (`ConsoleDisplay.console`) - the per-task "Simulation
       Overview" panels and the live "Status: X/N complete" progress reprints, which are
       redundant because gbench builds its own summary from results.simulations, and which
       spam the log when stdout is redirected to a file (rich can't update in place). Set
       GBENCH_TAU2_VERBOSE=1 (bare TAU2_VERBOSE still works) to keep the full tau2 panels.
       Real errors still surface via loguru.

    Best-effort across tau2 versions; patches the by-value import in runner.helpers too.
    """
    global _QUIETED
    if _QUIETED:
        return
    _QUIETED = True
    try:
        import tau2.utils.llm_utils as _llm            # called module-locally (llm_utils:420)
        _llm.get_response_cost = lambda *a, **k: 0.0
    except Exception as e:
        logger.debug("tau2: could not patch get_response_cost: %s", e)
    try:
        import tau2.utils.utils as _u
        _u.get_commit_hash = lambda *a, **k: "unknown"
        import tau2.runner.helpers as _h               # `from ...utils import get_commit_hash`
        if hasattr(_h, "get_commit_hash"):
            _h.get_commit_hash = lambda *a, **k: "unknown"
    except Exception as e:
        logger.debug("tau2: could not patch get_commit_hash: %s", e)
    if suite_env("GBENCH_TAU2_VERBOSE", "TAU2_VERBOSE") != "1":
        try:
            from rich.console import Console
            import tau2.utils.display as _disp
            _disp.ConsoleDisplay.console = Console(quiet=True)   # all panels route through this
        except Exception as e:
            logger.debug("tau2: could not mute rich console: %s", e)
        # litellm logs a Gemini-3+ sampling DeprecationWarning on every user-sim/judge call
        # (twice per call, via two loggers) - raise its loggers to ERROR so they don't flood.
        try:
            os.environ.setdefault("LITELLM_LOG", "ERROR")
            for _n in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
                logging.getLogger(_n).setLevel(logging.ERROR)
            import litellm as _ll
            _ll.suppress_debug_info = True
        except Exception as e:
            logger.debug("tau2: could not quiet litellm logging: %s", e)
        # The openai/httpx/httpcore HTTP clients log routine "Retrying request .../
        # HTTP Request: POST ..." at INFO for every agent call - pure noise here (gbench
        # builds its own summary). Raise them to WARNING; genuine failures still surface.
        try:
            for _n in ("openai", "openai._base_client", "httpx", "httpcore"):
                logging.getLogger(_n).setLevel(logging.WARNING)
        except Exception as e:
            logger.debug("tau2: could not quiet http client logging: %s", e)
        # Muting the console also killed tau2's useful 30s "Status: X/N complete" heartbeat
        # (StatusMonitor._monitor prints it through the same console). Redirect that heartbeat
        # to gbench's logger so the run shows live per-task progress like every other eval,
        # WITHOUT the per-task panels. Reuses tau2's own counters/reward math.
        _redirect_tau2_progress()


def _tau_progress_postfix(self, _time) -> Dict[str, Any]:
    """Build the tqdm postfix (avg reward, in-flight count, oldest elapsed) from a monitor."""
    post: Dict[str, Any] = {}
    try:
        results = self._simulation_results
        if results is not None:
            rewards = [s.reward_info.reward for s in results.simulations
                       if getattr(s, "reward_info", None) is not None]
            if rewards:
                post["reward"] = f"{sum(rewards) / len(rewards):.3f}"
        with self._lock:
            running = list(self.running_tasks.values())
        post["running"] = len(running)
        if running:
            now = _time.time()
            post["oldest"] = f"{max(now - i['start_time'] for i in running):.0f}s"
    except Exception:
        pass
    return post


def _redirect_tau2_progress() -> None:
    """Drive a tqdm progress bar from tau2's StatusMonitor - like every other gbench eval.

    tau2's native progress ("Status: X/N complete ...") prints through the console we mute.
    Instead we attach a tqdm bar to StatusMonitor (the same `tqdm(total=...)` mechanism
    `run_eval_suite` uses for every other suite): it advances per completed task and its
    postfix (avg reward, in-flight count, oldest elapsed) refreshes on a fixed cadence
    (min(GBENCH_TAU2_PROGRESS_SECS, 5)s) so it stays live between completions. On a TTY / `tail -f`
    this renders as one updating line, exactly like the other evals' `Eval [X]` bars.

    Falls back to periodic gbench-logger lines if tqdm is unavailable. Never fatal.
    """
    try:
        import time as _time
        from tau2.runner.progress import StatusMonitor
    except Exception as e:
        logger.debug("tau2: could not import StatusMonitor for progress: %s", e)
        return

    interval = max(1.0, float(suite_env("GBENCH_TAU2_PROGRESS_SECS", "TAU2_PROGRESS_SECS", default="30")))

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if tqdm is None:
        # --- fallback: one gbench-logger line per interval (no tqdm available) ---
        def _logger_monitor(self) -> None:
            while not self._stop_event.wait(timeout=interval):
                try:
                    with self._lock:
                        completed, total = self.completed_count, self.total_count
                    post = _tau_progress_postfix(self, _time)
                    logger.info("%s progress: %d/%d done | reward %s | %s running (oldest %s)",
                                _TAU_PROGRESS_DESC, completed, total,
                                post.get("reward", "n/a"), post.get("running", 0),
                                post.get("oldest", "-"))
                except Exception:
                    pass
        StatusMonitor._monitor = _logger_monitor
        return

    # --- tqdm bar, updated per-task; postfix refreshed by the monitor thread ---
    bar_interval = min(interval, 5.0)
    _orig_start = StatusMonitor.start
    _orig_finished = StatusMonitor.task_finished
    _orig_stop = StatusMonitor.stop

    def start(self) -> None:
        try:
            self._pbar = tqdm(total=self.total_count, initial=self.completed_count,
                              desc=f"Eval [{_TAU_PROGRESS_DESC}]", unit="task")
        except Exception:
            self._pbar = None
        _orig_start(self)                       # starts the (patched) _monitor thread

    def task_finished(self, task_key) -> None:
        _orig_finished(self, task_key)          # increments completed_count, pops running
        pb = getattr(self, "_pbar", None)
        if pb is not None:
            try:
                pb.update(1)                    # tqdm.update is thread-safe across workers
            except Exception:
                pass

    def stop(self) -> None:
        _orig_stop(self)
        pb = getattr(self, "_pbar", None)
        if pb is not None:
            try:
                pb.set_postfix(_tau_progress_postfix(self, _time), refresh=True)
                pb.close()
            except Exception:
                pass
            self._pbar = None

    def _monitor(self) -> None:                 # keep postfix + elapsed live between completions
        while not self._stop_event.wait(timeout=bar_interval):
            pb = getattr(self, "_pbar", None)
            if pb is None:
                continue
            try:
                pb.set_postfix(_tau_progress_postfix(self, _time), refresh=True)
            except Exception:
                pass

    StatusMonitor.start = start
    StatusMonitor.task_finished = task_finished
    StatusMonitor.stop = stop
    StatusMonitor._monitor = _monitor


def _user_llm() -> str:
    """The first model the user simulator tries (the model tau2's TextRunConfig names)."""
    return _role_cascade(_USER_ROLE)[0]


def _eval_llm() -> str:
    """The first model the NL-assertion judge tries (tau2's DEFAULT_LLM_NL_ASSERTIONS)."""
    return _role_cascade(_JUDGE_ROLE)[0]


def _import_tau2() -> Tuple[bool, str]:
    """Attempt a real `import tau2`.

    We import (not just find_spec) because tau2 can be *findable* yet not *importable* -
    e.g. at the pinned revision `import tau2` needs `websockets` from the voice extra, and on
    Python 3.13 its voice module imports the removed stdlib `audioop`. Callers need the real
    failure reason, so we return (ok, error_message).
    """
    # tau2 emits import-time loguru noise before it ever does useful work: a DEBUG
    # "Registry info: {...}" block (domains/agents/users/task_sets) and a WARNING
    # "No .env file found" (it looks for an optional .env to load keys; we pass keys via
    # the environment, so a missing .env is irrelevant). Configure loguru *before*
    # importing: raise the threshold to WARNING (kills the DEBUG dump) and add a filter
    # that drops the benign .env line while keeping real warnings (e.g. task retries).
    # GBENCH_TAU2_VERBOSE=1 keeps everything.
    if suite_env("GBENCH_TAU2_VERBOSE", "TAU2_VERBOSE") != "1":
        try:
            from loguru import logger as _loguru

            def _drop_import_noise(record):
                return "No .env file found" not in record["message"]

            _loguru.remove()
            _loguru.add(sys.stderr, level="WARNING", filter=_drop_import_noise)
        except Exception:
            pass
    try:
        import tau2  # noqa: F401
        return True, ""
    except Exception as e:  # ImportError + anything tau2.__init__ raises
        return False, f"{type(e).__name__}: {e}"


def _git(root: str, *args: str) -> Optional[str]:
    """Output of a read-only git command in `root`, or None when git fails."""
    try:
        proc = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                              timeout=30)
    except Exception:                                            # noqa: BLE001
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def tau2_provenance() -> Dict[str, Any]:
    """Where the imported tau2 package comes from: its checkout, commit, local edits and data dir.

    tau2-bench is not on PyPI and every release keeps its task data in the checkout, so the
    revision that runs is the git commit of the checkout holding the imported `src/tau2`
    (the gbench-tau2 image's checkout, or an editable install). The package version is read from that
    checkout's pyproject.toml, since dist metadata can belong to a different install.
    """
    info: Dict[str, Any] = {"package_dir": None, "checkout": None, "commit": None,
                            "modified": None, "data_dir": None, "version": None}
    try:
        import tau2
        pkg_dir = os.path.dirname(os.path.realpath(tau2.__file__))
    except Exception as e:                                       # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    info["package_dir"] = pkg_dir
    root = _git(pkg_dir, "rev-parse", "--show-toplevel")
    if root and os.path.realpath(os.path.join(root, "src", "tau2")) == pkg_dir:
        info["checkout"] = root
        info["commit"] = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain", "--untracked-files=no", "--", "src", "data")
        info["modified"] = [ln.strip().split(None, 1)[-1]
                            for ln in (status or "").splitlines() if ln.strip()]
        try:
            with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
                m = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.MULTILINE)
            info["version"] = m.group(1) if m else None
        except OSError:
            pass
    try:
        from tau2.utils.utils import DATA_DIR
        info["data_dir"] = os.path.realpath(str(DATA_DIR))
    except Exception:                                            # noqa: BLE001
        pass
    return info


def image_build_command(image: Optional[str] = None) -> str:
    """The command that builds the local tau2 image (run from the gbench repository root)."""
    return (f"docker build -t {image or tau2_image()} -f {TAU2_DOCKERFILE} {TAU2_IMAGE_CONTEXT}")


def check_tau2_revision() -> Tuple[bool, str]:
    """The imported tau2 must be a clean checkout at TAU2_BENCH_REVISION reading its own data."""
    info = tau2_provenance()
    install = (f"run the suite through the local image, which holds tau2-bench at the pinned "
               f"revision {TAU2_BENCH_REVISION} (`{image_build_command()}`); see "
               "docs/evals/tau2.md")
    if not info.get("checkout"):
        return False, (f"cannot verify the tau2-bench revision: the imported tau2 package "
                       f"({info.get('package_dir') or info.get('error')}) is not the src/tau2 of "
                       f"a git checkout; {install}")
    if info.get("commit") != TAU2_BENCH_REVISION:
        return False, (f"tau2-bench checkout {info['checkout']} is at commit "
                       f"{info.get('commit')} (version {info.get('version')}), not the pinned "
                       f"{TAU2_BENCH_REVISION} (version {TAU2_BENCH_VERSION}); {install}")
    if info.get("modified"):
        shown = ", ".join(info["modified"][:5]) + (" ..." if len(info["modified"]) > 5 else "")
        return False, (f"tau2-bench checkout {info['checkout']} has local changes under src/ or "
                       f"data/ ({shown}); restore them (`git -C {info['checkout']} checkout -- src "
                       "data`) so the pinned harness and task data run unmodified")
    expected_data = os.path.realpath(os.path.join(info["checkout"], "data"))
    if info.get("data_dir") != expected_data:
        return False, (f"tau2 reads its task data from {info.get('data_dir')} instead of the pinned "
                       f"checkout's {expected_data}; unset TAU2_DATA_DIR (or point it there)")
    return True, ""


def _normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_tau2_dependencies(checkout: Optional[str] = None) -> Tuple[bool, str]:
    """The packages tau2-bench depends on (its base dependencies plus the `knowledge` extra) must be
    installed at the versions its own uv.lock pins at the checked-out revision.

    tau2-bench declares ranges in pyproject.toml (e.g. `litellm>=1.80.15,<1.82.7`) and pins exact
    versions in uv.lock; its install instructions sync from that lock. The gbench image installs
    exactly those versions, so this holds there by construction; any other interpreter (for
    example one with a newer litellm) hard-errors instead of running a different LLM client stack.
    """
    import tomllib
    from importlib.metadata import PackageNotFoundError, version as _dist_version
    root = checkout or tau2_provenance().get("checkout")
    if not root:
        return False, "cannot check tau2-bench's dependency pins: no tau2-bench checkout found"
    try:
        with open(os.path.join(root, "pyproject.toml"), "rb") as fh:
            project = tomllib.load(fh)["project"]
        with open(os.path.join(root, "uv.lock"), "rb") as fh:
            packages = tomllib.load(fh).get("package", [])
    except (OSError, KeyError, ValueError) as e:
        return False, (f"cannot read tau2-bench's pyproject.toml / uv.lock in {root} "
                       f"({type(e).__name__}: {e})")
    locked: Dict[str, set] = {}
    for pkg in packages:
        if "name" in pkg and "version" in pkg:
            locked.setdefault(_normalize_dist(pkg["name"]), set()).add(str(pkg["version"]))
    requirements = list(project.get("dependencies") or []) + list(
        (project.get("optional-dependencies") or {}).get("knowledge") or [])
    mismatches = []
    for req in requirements:
        name = _normalize_dist(re.split(r"[\s<>=!~;\[(]", req.strip(), maxsplit=1)[0])
        want = locked.get(name)
        if not want:
            mismatches.append(f"{name} (not in uv.lock)")
            continue
        try:
            have = _dist_version(name)
        except PackageNotFoundError:
            have = None
        if have not in want:
            mismatches.append(f"{name} {have or 'not installed'} "
                              f"(uv.lock: {', '.join(sorted(want))})")
    if mismatches:
        return False, ("the installed packages do not match tau2-bench's uv.lock at the pinned "
                       f"revision: {'; '.join(mismatches)}. Run the suite through the local image "
                       f"(`{image_build_command()}`); see docs/evals/tau2.md")
    return True, ""


def check_tau_env_prerequisites() -> Tuple[bool, str]:
    """tau2-bench actually importable + the Gemini key for the user simulator and judge."""
    ok, err = _import_tau2()
    if not ok:
        if "No module named 'tau2'" in err:
            return False, ("tau2-bench is not importable here. gbench runs it in the local image "
                           f"built from {TAU2_DOCKERFILE} (`{image_build_command()}`); see "
                           "docs/evals/tau2.md.")
        hint = ""
        if "audioop" in err:
            hint = (" tau2's voice module imports the stdlib `audioop`, removed in Python 3.13 - "
                    "install the backport: `pip install audioop-lts`.")
        return False, f"tau2-bench is installed but failed to import ({err}).{hint}"
    raw_key = os.getenv("GEMINI_API_KEY", "")
    if raw_key and re.search(r"[;,]", raw_key):
        os.environ["GEMINI_API_KEY"] = _gemini_key({"GEMINI_API_KEY": raw_key})

    if (_user_llm().startswith("gemini/") or _eval_llm().startswith("gemini/")) \
            and not os.getenv("GEMINI_API_KEY"):
        return False, ("tau2 environment needs GEMINI_API_KEY (or GEMINI_API_KEYS): every external "
                       "model a tau run calls (the user simulator, the nl-assertion judge and "
                       "tau3's retrieval embeddings) goes through the Gemini API. See "
                       "docs/evals/tau2.md.")
    ok_rev, why_rev = check_tau2_revision()
    if not ok_rev:
        return False, why_rev
    ok_dep, why_dep = check_tau2_dependencies()
    if not ok_dep:
        return False, why_dep
    ok_roles, why_roles = check_role_credentials()
    if not ok_roles:
        return False, why_roles
    ok_route, why_route = check_gemini_route()
    if not ok_route:
        return False, why_route
    key = os.getenv("GEMINI_API_KEY") or ""
    if (_user_llm().startswith("gemini/") or _eval_llm().startswith("gemini/")) and key \
            and not _relay_placeholder(key):
        # A real key here means the driver runs outside the image (the image only ever holds the
        # relay's placeholder, whose real key run_tau_suite checked on the host).
        from .base import MODELS_NOT_SERVED_PREFIX, gemini_key_live_valid
        _ok, _why = gemini_key_live_valid(key)
        if not _ok:
            if str(_why).startswith(MODELS_NOT_SERVED_PREFIX):
                return False, (f"{_why}; every model the tau2 user simulator and NL-assertion "
                               "judge may call must be one the Gemini API serves.")
            return False, (f"the Gemini API refused GEMINI_API_KEY ({_why}); a valid key is "
                           "required for the tau2 user simulator / nl-assertion judge "
                           "(fail-fast).")
    return True, ""


def _relay_placeholder(key: Optional[str]) -> bool:
    """True when `key` is gbench's Gemini relay placeholder (gemini_relay.is_placeholder)."""
    from .gemini_relay import is_placeholder
    return is_placeholder(key)


def _relay_origin() -> Optional[str]:
    """The Gemini relay's origin (http://<address>:<port>) from GBENCH_GEMINI_RELAY_URL, or None
    when it is unset or not a plain http origin."""
    from urllib.parse import urlsplit
    from .gemini_relay import RELAY_URL_ENV
    raw = (os.environ.get(RELAY_URL_ENV) or "").strip().rstrip("/")
    if not raw or any(c.isspace() for c in raw):
        return None
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "http" or not parts.hostname or not port or "@" in parts.netloc \
            or parts.path or parts.query or parts.fragment:
        return None
    return raw


def check_gemini_route() -> Tuple[bool, str]:
    """Where this process sends its Gemini calls. The image gets only the placeholder of gbench's
    Gemini relay (a thread of the gbench process on the docker bridge gateway, which adds the
    real key; run_tau_suite) and the relay's address: the placeholder needs that address, and a
    real Gemini key inside the image is refused (the process shares its container and user with
    the agent's tool calls). Outside the image a real key goes to the Gemini API directly."""
    from .gemini_relay import RELAY_URL_ENV
    key = _gemini_key()
    if _relay_placeholder(key):
        if _relay_origin() is None:
            return False, ("the tau2 driver holds gbench's Gemini relay placeholder but no valid "
                           f"relay address ({RELAY_URL_ENV}); run the suite through gbench, which "
                           "starts the relay. See docs/evals/tau2.md.")
        return True, ""
    if key and _in_image():
        return False, ("a real Gemini key reached the tau2 container, which runs the agent's tool "
                       "calls; gbench passes it only its Gemini relay's placeholder. Run the suite "
                       "through gbench. See docs/evals/tau2.md.")
    return True, ""


def _gemini_native_base() -> str:
    """The base URL of this process' native Gemini calls (LiteLLM's gemini/ user simulator and
    judge, the tau3 embedder): <relay origin>/v1beta when the key is the relay's placeholder (the
    relay forwards /v1beta/models/<model>:<method> to the Gemini API with the real key), else the
    Gemini API itself. A placeholder without a relay address raises (check_gemini_route)."""
    if _relay_placeholder(_gemini_key()):
        origin = _relay_origin()
        if origin is None:
            raise GeminiRouteError(check_gemini_route()[1])
        return origin + "/v1beta"
    return _GEMINI_API_BASE


class GeminiRouteError(RuntimeError):
    """The process holds the relay placeholder but has no relay address."""


def check_role_credentials() -> Tuple[bool, str]:
    """Inside the image: the user simulator and the judge are gemini/ models that tau2's LiteLLM
    routes to the Gemini API, and GEMINI_API_KEY is set.

    A model of any other provider (openai/, anthropic/, openrouter/, vertex_ai/, a bare gemini-*
    name, ...) hard-errors naming the provider, as does a gemini/ name that LiteLLM routes
    elsewhere or cannot route at all.
    """
    for role, model in (("user simulator", _user_llm()), ("NL-assertion judge", _eval_llm())):
        provider = _role_provider(model)
        if provider != _GEMINI_PROVIDER:
            return False, _unsupported_role(role, model, provider)
        routed = _litellm_provider(model)
        if routed == "":
            return False, (f"LiteLLM cannot route the {role} {model} to a provider; write it as "
                           "gemini/<model> in GBENCH_TAU2_USER_LLM / GBENCH_TAU2_EVAL_LLM. "
                           "See docs/evals/tau2.md.")
        if routed and routed != provider:
            return False, (f"LiteLLM routes the {role} {model} to the provider '{routed}', not to "
                           "the Gemini API; write it as gemini/<model>. See docs/evals/tau2.md.")
        if not os.getenv("GEMINI_API_KEY"):
            return False, (f"the {role} {model} needs GEMINI_API_KEY (or GEMINI_API_KEYS), the "
                           "Gemini API key. See docs/evals/tau2.md.")
    return True, ""


def check_banking_prerequisites() -> Tuple[bool, str]:
    """Extra prerequisites for the banking_knowledge (tau3) RAG domain.

    banking_knowledge is a knowledge-retrieval domain with one retrieval configuration: upstream's
    alltools, whose dense search gbench runs with gemini-embedding-001. It needs tau2-bench's
    `knowledge` extra (rank-bm25), upstream's alltools in tau2's registry with the Gemini key
    (`check_retrieval_credentials`), and, for alltools' shell tool, a working sandbox whose
    environment holds no credential (`check_shell_sandbox`).
    """
    try:
        import rank_bm25  # noqa: F401
    except Exception:
        return False, ("banking_knowledge (tau3) needs the BM25 retrieval backend (rank-bm25, "
                       "tau2-bench's `knowledge` extra, installed in the gbench-tau2 image). See "
                       "'docs/evals/tau3.md' for setup.")
    ok, why = check_retrieval_credentials()
    if not ok:
        return False, why
    return check_shell_sandbox()


#: tau2-bench's default banking_knowledge retrieval config at TAU2_BENCH_REVISION
#: (retrieval.py DEFAULT_RETRIEVAL_VARIANT; simulation.py sets it when none is given): BM25 +
#: dense search with OpenAI text-embedding-3-large + the shell tool. 21 of the leaderboard's
#: standard banking_knowledge entries at the revision use it. It is the only configuration gbench
#: runs.
_UPSTREAM_RETRIEVAL_CONFIG = "alltools"
_UPSTREAM_DENSE_EMBED_MODEL = "text-embedding-3-large"
_UPSTREAM_DENSE_EMBEDDER = f"openai:{_UPSTREAM_DENSE_EMBED_MODEL}"

#: gbench's substitution for alltools' dense embedder: gemini-embedding-001 through the Gemini
#: API's batchEmbedContents, registered with tau2 as embedder_type "gemini". The variant tau2
#: runs is `alltools-gemini`: upstream's alltools variant with only the dense pipeline's
#: embedder_type / embedder_model changed (`_register_gemini_retrieval`).
GEMINI_EMBED_MODEL = "gemini-embedding-001"
_GEMINI_EMBEDDER_TYPE = "gemini"
_GEMINI_DENSE_EMBEDDER = f"{_GEMINI_EMBEDDER_TYPE}:{GEMINI_EMBED_MODEL}"
_GEMINI_RETRIEVAL_VARIANT = "alltools-gemini"
#: The dense embedders a banking_knowledge run uses (recorded on the result).
_DENSE_EMBEDDERS = [_GEMINI_DENSE_EMBEDDER]
#: The substitution as named in leaderboard_comparable_reason.
GEMINI_EMBED_REASON = (f"retrieval embeddings by {GEMINI_EMBED_MODEL} "
                       f"(upstream: {_UPSTREAM_DENSE_EMBED_MODEL})")
#: The Gemini API's host: every external model call of a tau run ends there (the user simulator
#: and the judge through LiteLLM's gemini/ provider, tau3's retrieval embeddings through
#: _GEMINI_API_BASE). The container reaches it through gbench's Gemini relay on the bridge
#: gateway (`_gemini_native_base`), which resolves it on this host.
GEMINI_API_HOST = "generativelanguage.googleapis.com"
_GEMINI_API_BASE = f"https://{GEMINI_API_HOST}/v1beta"
#: Texts per batchEmbedContents request (the Gemini API rejects larger batches); tau2's own
#: embedders send every text in one request, so a longer list is split into consecutive batches
#: and the vectors are concatenated in order.
_GEMINI_EMBED_BATCH = 100
#: Attempts per request on a 429, a 5xx or a transport error (exponential backoff with jitter);
#: any other HTTP error fails at once.
_GEMINI_EMBED_ATTEMPTS = 4
_GEMINI_EMBED_BACKOFF_S = 2.0
_GEMINI_EMBED_TIMEOUT_S = 120


def _upstream_alltools_problem(registry: Dict[str, Any]) -> str:
    """Why upstream's alltools in tau2's retrieval `registry` cannot be run with the Gemini
    embedder ("" when it can): it must be registered and have a dense embedding pipeline, the one
    pipeline gbench substitutes."""
    upstream = registry.get(_UPSTREAM_RETRIEVAL_CONFIG)
    if upstream is None:
        return (f"tau2's banking_knowledge retrieval registry has no {_UPSTREAM_RETRIEVAL_CONFIG} "
                "config at the pinned revision")
    dense = getattr(upstream, "kb_search_dense", None)
    if dense is None or getattr(dense, "type", None) != "embedding":
        return (f"tau2's {_UPSTREAM_RETRIEVAL_CONFIG} retrieval config has no dense embedding "
                "pipeline at the pinned revision, so its embedder cannot be substituted")
    return ""


def check_retrieval_credentials() -> Tuple[bool, str]:
    """banking_knowledge runs one retrieval configuration: upstream's alltools, with its dense
    search on gemini-embedding-001 through the Gemini API. tau2's registry must hold alltools with
    its dense embedding pipeline, and the Gemini key must be set (else infra_required: nothing
    falls back to another embedder or configuration)."""
    try:
        from tau2.domains.banking_knowledge import retrieval as _r
        registry = dict(getattr(_r, "RETRIEVAL_VARIANTS", {}))
    except Exception as e:                                       # noqa: BLE001
        return False, (f"cannot load tau2's banking_knowledge retrieval registry "
                       f"({type(e).__name__}: {e}); see 'docs/evals/tau3.md'.")
    problem = _upstream_alltools_problem(registry)
    if problem:
        return False, f"{problem}; see 'docs/evals/tau3.md'."
    if not _gemini_key():
        return False, (f"banking_knowledge retrieval (upstream's alltools) runs its dense "
                       f"search with {GEMINI_EMBED_MODEL} through the Gemini API, gbench's "
                       f"stand-in for upstream's OpenAI {_UPSTREAM_DENSE_EMBED_MODEL}, and "
                       "needs GEMINI_API_KEY (or GEMINI_API_KEYS). See 'docs/evals/tau3.md'.")
    return True, ""


#: The probe command: `cat` must print the exported file, and neither the command's environment
#: (`printenv`) nor that of the sandbox's PID 1 (bubblewrap, in its own PID namespace; were the
#: namespace shared, PID 1 would be the driver, which holds the keys) may show a credential. The
#: path is relative after `cd /` because tau2 refuses commands with absolute paths.
_SHELL_PROBE_COMMAND = ("cat gbench_probe.txt && printenv && cd / && "
                        "tr '\\0' '\\n' < proc/1/environ")


def check_shell_sandbox() -> Tuple[bool, str]:
    """Run one command through tau2's own SandboxManager and check that it really executed and
    that its environment holds no credential.

    The shell tool runs every agent command as `srt --settings <file> <command>` (Anthropic's
    sandbox-runtime over bubblewrap). tau2 checks only that the binaries are on PATH and raises
    only when srt reports "sandbox dependencies are not available"; any other failure (a
    different program named `srt`, or bubblewrap unable to create its namespaces) comes back to
    the agent as an ordinary tool result on every call. tau2 also hands srt the whole process
    environment, provider keys included; gbench runs the tool's commands with every credential
    removed (`_install_tool_env_scrub`). This probe exports one file into a fresh sandbox,
    requires `cat` to return its content and the command's and the sandbox PID 1's environments
    to show none of the run's credential values, so either failure hard-errors before the run.
    """
    try:
        from tau2.knowledge.sandbox_manager import SandboxManager
        _install_tool_env_scrub()
        with SandboxManager(allow_writes=False) as sandbox:
            sandbox.export_documents([{"id": "gbench_probe", "title": "gbench probe",
                                       "content": "gbench sandbox probe"}], file_format="txt")
            code, out, err = sandbox.run_command(_SHELL_PROBE_COMMAND)
    except Exception as e:                                       # noqa: BLE001
        return False, (f"banking_knowledge's shell tool cannot start its sandbox "
                       f"({type(e).__name__}: {_redact_text(str(e).strip(), _SECRETS)[:400]}); "
                       "see 'docs/evals/tau3.md'.")
    seen = (out or "") + (err or "")
    leaked = sorted(name for name, value in os.environ.items()
                    if _is_secret_name(name) and value and len(value) >= _MIN_SECRET_LEN
                    and value in seen)
    if any(s in seen for s in _SECRETS) or leaked:
        return False, ("banking_knowledge's shell tool can read credentials in its environment "
                       f"({', '.join(leaked) or 'a credential value'}), which the model under "
                       "test must never see; see 'docs/evals/tau3.md'.")
    if code != 0 or "gbench sandbox probe" not in (out or ""):
        detail = " ".join(_redact_text((err or "") + " " + (out or ""), _SECRETS).split())[:400]
        return False, (f"banking_knowledge's shell tool does not work here: a probe command in "
                       f"tau2's sandbox exited {code} ({detail or 'no output'}), so every shell "
                       "call would return that error to the agent. Run the suite through the "
                       "gbench-tau2 image, which provides sandbox-runtime 0.0.23 and the kernel "
                       "features bubblewrap needs; see 'docs/evals/tau3.md'.")
    return True, ""


#: What `_setup_banking_retrieval` last selected, so the result can record it.
_LAST_RETRIEVAL: Dict[str, Any] = {}


def _gemini_key(environ: Optional[Dict[str, str]] = None) -> str:
    """The Gemini API key: the first key of GEMINI_API_KEYS when that list is set (it wins over
    GEMINI_API_KEY throughout gbench), else of GEMINI_API_KEY; "" when neither is set."""
    env = os.environ if environ is None else environ
    for name in ("GEMINI_API_KEYS", "GEMINI_API_KEY"):
        keys = [k.strip() for k in re.split(r"[;,]", env.get(name) or "") if k.strip()]
        if keys:
            return keys[0]
    return ""


class GeminiEmbeddingError(RuntimeError):
    """A Gemini embedding call that failed after its retries (raised by GeminiEmbedder)."""


class TauHarnessError(RuntimeError):
    """A harness failure during a simulation (a Gemini embedding call that failed after its
    retries), raised out of the tool call so that tau2 stops the simulation attempt instead of
    showing the agent a tool error. tau2's run_with_retry re-runs the attempt like any failed
    one, and records an infrastructure error once its retries are spent."""


#: The domain whose simulations are running (`_run_one_domain`); harness errors are counted under
#: it. Domains run one after another in a process.
_CURRENT_DOMAIN: Dict[str, Optional[str]] = {"domain": None}
#: Gemini embedding calls that failed after their retries, by domain.
_EMBEDDING_FAILURES: Dict[str, int] = {}
#: Simulation attempts the embedding-failure guard stopped, by domain.
_EMBEDDING_ABORTS: Dict[str, int] = {}
_HARNESS_LOCK = threading.Lock()
#: Per thread: the last Gemini embedding failure since the current tool call started. A tool call
#: runs synchronously on the simulation's thread (tau2's Environment.get_response ->
#: make_tool_call -> KB_search_dense -> the query encoder -> GeminiEmbedder), so the guard around
#: get_response sees exactly the failures of that call.
_EMBED_FAILURE_SEEN = threading.local()


def _note_embedding_failure(message: str = "") -> None:
    """Count a Gemini embedding call that failed after its retries: in
    `llm_failures()["embedding_error"]`, under the running domain (when a domain is running), and
    for this thread's current tool call (`_embedding_failure_guard`)."""
    domain = _CURRENT_DOMAIN.get("domain")
    with _HARNESS_LOCK:
        _LLM_FAILURES["embedding_error"] = _LLM_FAILURES.get("embedding_error", 0) + 1
        if domain:
            _EMBEDDING_FAILURES[domain] = _EMBEDDING_FAILURES.get(domain, 0) + 1
    _EMBED_FAILURE_SEEN.message = message or "Gemini embeddings failed"


def _embedding_failure_guard(orig):
    """Wrap tau2's Environment.get_response: a tool call during which a Gemini embedding call
    failed after its retries raises TauHarnessError instead of returning the tool message.

    Upstream's get_response turns every exception of a tool into an `Error: ...` tool result that
    the agent reads (and that counts toward max_errors), so an outage of gbench's Gemini embedder
    would be scored as the agent's outcome. Raised out of get_response, the error stops the
    simulation attempt; tau2's run_with_retry re-runs it as it re-runs any failed attempt and, once
    its retries are spent, records the simulation as an infrastructure error, which compute_metrics
    leaves out. The failures are counted per domain (`_note_embedding_failure`,
    `_EMBEDDING_ABORTS`)."""
    import functools

    @functools.wraps(orig)
    def get_response(self, message):
        _EMBED_FAILURE_SEEN.message = None
        resp = orig(self, message)
        failure = getattr(_EMBED_FAILURE_SEEN, "message", None)
        _EMBED_FAILURE_SEEN.message = None
        if failure is not None:
            domain = _CURRENT_DOMAIN.get("domain")
            if domain:
                with _HARNESS_LOCK:
                    _EMBEDDING_ABORTS[domain] = _EMBEDDING_ABORTS.get(domain, 0) + 1
            tool = getattr(message, "name", None) or "a tool"
            raise TauHarnessError(f"harness error, not the agent's outcome: {failure} during "
                                  f"{tool}; the simulation attempt is stopped")
        return resp

    get_response._gbench_embedding_guard = True
    return get_response


def embedding_failures() -> Dict[str, int]:
    """Gemini embedding calls that failed after their retries, by domain."""
    with _HARNESS_LOCK:
        return dict(_EMBEDDING_FAILURES)


def reset_harness_errors() -> None:
    with _HARNESS_LOCK:
        _EMBEDDING_FAILURES.clear()
        _EMBEDDING_ABORTS.clear()
    _EMBED_FAILURE_SEEN.message = None


class GeminiEmbedder:
    """A tau2 embedder (the BaseEmbedder interface: `embed`, `get_name`) over the Gemini API.

    `embed(texts)` sends the texts to models/<model>:batchEmbedContents, at most
    _GEMINI_EMBED_BATCH per request, each as one content part with no task type and the model's
    default output dimensionality, which is how tau2's OpenAIEmbedder sends texts to OpenAI
    (input and model only); the vectors come back in input order as a (len(texts), dim) array.
    The key (`_gemini_key`, read at call time) travels in the x-goog-api-key header, never in the
    URL, and no error message carries it. In the image that key is the placeholder of gbench's
    Gemini relay and the requests go to the relay (`_gemini_native_base`), which adds the real
    key on the host and forwards them to the Gemini API. A 429, a 5xx or a transport error is
    retried up to _GEMINI_EMBED_ATTEMPTS times with backoff; the final failure raises
    GeminiEmbeddingError and is counted (`_note_embedding_failure`): in
    `llm_failures()["embedding_error"]`, per domain, and for the tool call in progress, which
    `_embedding_failure_guard` then turns into a harness error. There is no other embedder to
    fall back to.
    """

    def __init__(self, model: str = GEMINI_EMBED_MODEL):
        self.model = model

    def get_name(self) -> str:
        return f"{_GEMINI_EMBEDDER_TYPE}_{self.model}"

    def embed(self, texts: List[str]) -> Any:
        import numpy as np
        if not texts:
            raise ValueError("No text to embed.")
        vectors: List[List[float]] = []
        for start in range(0, len(texts), _GEMINI_EMBED_BATCH):
            vectors.extend(self._embed_batch(list(texts[start:start + _GEMINI_EMBED_BATCH])))
        if len({len(v) for v in vectors}) != 1:
            text = f"Gemini embeddings ({self.model}) returned vectors of different lengths"
            _note_embedding_failure(text)
            raise GeminiEmbeddingError(text)
        return np.array(vectors)

    def _fail(self, message: str, key: str) -> GeminiEmbeddingError:
        text = _redact_text(message, _secret_values({}, extra=tuple(_SECRETS) + (key,)))
        text = f"Gemini embeddings ({self.model}) failed: {text}"
        _note_embedding_failure(text)
        return GeminiEmbeddingError(text)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        import http.client
        import urllib.error
        import urllib.request
        key = _gemini_key()
        if not key:
            raise self._fail("GEMINI_API_KEY (or GEMINI_API_KEYS) is not set", key)
        try:
            url = f"{_gemini_native_base()}/models/{self.model}:batchEmbedContents"
        except GeminiRouteError as e:
            raise self._fail(str(e), key) from None
        body = json.dumps({"requests": [
            {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}}
            for t in texts]}).encode("utf-8")
        data: Any = None
        for attempt in range(_GEMINI_EMBED_ATTEMPTS):
            request = urllib.request.Request(url, data=body, method="POST", headers={
                "Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(request, timeout=_GEMINI_EMBED_TIMEOUT_S) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                try:
                    detail = " ".join(e.read().decode("utf-8", "replace").split())[:300]
                except Exception:                                # noqa: BLE001
                    detail = ""
                error, retry = f"HTTP {e.code} {detail}".strip(), e.code == 429 or e.code >= 500
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
                error, retry = f"{type(e).__name__}: {e}", True
            if not retry or attempt == _GEMINI_EMBED_ATTEMPTS - 1:
                raise self._fail(f"{error} (attempt {attempt + 1} of {_GEMINI_EMBED_ATTEMPTS})",
                                 key)
            time.sleep(_GEMINI_EMBED_BACKOFF_S * (2 ** attempt)
                       + random.uniform(0, _GEMINI_EMBED_BACKOFF_S))
        embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise self._fail(f"expected {len(texts)} embeddings, got "
                             f"{len(embeddings) if isinstance(embeddings, list) else 'none'}", key)
        out = []
        for item in embeddings:
            values = item.get("values") if isinstance(item, dict) else None
            if not isinstance(values, list) or not values:
                raise self._fail("an embedding without values", key)
            out.append([float(v) for v in values])
        return out


def _install_gemini_embedder() -> None:
    """Make tau2's embedding pipelines build GeminiEmbedder for embedder_type "gemini": the
    EMBEDDER_REGISTRY of its document indexer and of its query encoder (idempotent). Raises when
    either registry is missing; there is no other embedder to use."""
    from tau2.knowledge.document_preprocessors import embedding_indexer as _indexer
    from tau2.knowledge.input_preprocessors import embedding_encoder as _encoder
    for module in (_indexer, _encoder):
        registry = getattr(module, "EMBEDDER_REGISTRY", None)
        if not isinstance(registry, dict):
            raise RuntimeError(f"{module.__name__} has no EMBEDDER_REGISTRY at the pinned tau2 "
                               "revision; the Gemini embedder cannot be installed")
        registry[_GEMINI_EMBEDDER_TYPE] = GeminiEmbedder


def _register_gemini_retrieval() -> Any:
    """Register `alltools-gemini` with tau2 and return it: a deep copy of upstream's alltools
    variant (same prompt template, BM25 pipeline, shell, top_k and every other field) whose name
    is alltools-gemini and whose dense pipeline has embedder_type "gemini" and embedder_model
    gemini-embedding-001. Raises when upstream's alltools is missing or has no dense embedding
    pipeline."""
    import copy
    from tau2.domains.banking_knowledge import retrieval as _r
    problem = _upstream_alltools_problem(_r.RETRIEVAL_VARIANTS)
    if problem:
        raise RuntimeError(problem)
    upstream = _r.RETRIEVAL_VARIANTS[_UPSTREAM_RETRIEVAL_CONFIG]
    _install_gemini_embedder()
    variant = copy.deepcopy(upstream)
    variant.name = _GEMINI_RETRIEVAL_VARIANT
    variant.kb_search_dense.embedder_type = _GEMINI_EMBEDDER_TYPE
    variant.kb_search_dense.embedder_model = GEMINI_EMBED_MODEL
    _r.RETRIEVAL_VARIANTS[_GEMINI_RETRIEVAL_VARIANT] = variant
    return variant


def _dense_tool_description(variant: Any) -> Optional[str]:
    """The KB_search_dense line of the agent's policy, as tau2 renders it for `variant`
    (retrieval.format_all_tools_dense_instructions fills all_tools.md's placeholder from the
    variant's embedder)."""
    from tau2.domains.banking_knowledge import retrieval as _r
    render = getattr(_r, "format_all_tools_dense_instructions", None)
    return render(variant) if render else None


def _warm_gemini_document_embeddings() -> int:
    """Embed the knowledge base's documents once, before any simulation starts, through tau2's own
    EmbeddingIndexer and embeddings cache, exactly as the dense pipeline of every simulation
    indexes them (tau2's warm_kb_cache knows only upstream's config names, so it does not warm
    alltools-gemini). Returns the number of documents."""
    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    from tau2.domains.banking_knowledge.retrieval import get_or_create_docs
    from tau2.knowledge.document_preprocessors.embedding_indexer import EmbeddingIndexer
    docs = get_or_create_docs(get_knowledge_base())
    EmbeddingIndexer(embedder_type=_GEMINI_EMBEDDER_TYPE,
                     embedder_params={"model": GEMINI_EMBED_MODEL}).process(docs, {})
    return len(docs)


def _setup_banking_retrieval() -> Tuple[str, None]:
    """The banking_knowledge retrieval config for TextRunConfig: (name, kwargs).

    The one configuration gbench runs: upstream's default alltools, as `alltools-gemini`
    (`_register_gemini_retrieval`). The Gemini embedder is installed, the variant registered and
    the documents embedded before the run. Any failure raises (the domain then hard-errors);
    nothing falls back to upstream's OpenAI embedder or to another config.
    """
    variant = _register_gemini_retrieval()
    description = _dense_tool_description(variant)
    try:
        documents = _warm_gemini_document_embeddings()
    except Exception as e:
        raise RuntimeError(f"embedding the banking_knowledge documents with "
                           f"{GEMINI_EMBED_MODEL} through the Gemini API failed "
                           f"({type(e).__name__}: {_redact_text(str(e), _SECRETS)[:400]})"
                           ) from e
    _LAST_RETRIEVAL.update({
        "retrieval_config": _UPSTREAM_RETRIEVAL_CONFIG,
        "retrieval_variant": _GEMINI_RETRIEVAL_VARIANT,
        "embedders": list(_DENSE_EMBEDDERS), "embed_model": _GEMINI_DENSE_EMBEDDER,
        "embedder": {"embedder": _GEMINI_DENSE_EMBEDDER,
                     "api": f"Gemini API {_GEMINI_API_BASE}/models/{GEMINI_EMBED_MODEL}"
                            ":batchEmbedContents",
                     "through": ("gbench's Gemini relay on the docker bridge gateway, which "
                                 "adds the key" if _relay_placeholder(_gemini_key()) else
                                 "the Gemini API directly"),
                     "upstream_embedder": _UPSTREAM_DENSE_EMBEDDER,
                     "batch_size": _GEMINI_EMBED_BATCH,
                     "documents": documents,
                     "dense_tool_description": description}})
    logger.info("tau3: banking_knowledge retrieval config %s run as %s (dense embedder: %s)",
                _UPSTREAM_RETRIEVAL_CONFIG, _GEMINI_RETRIEVAL_VARIANT, _GEMINI_DENSE_EMBEDDER)
    return _GEMINI_RETRIEVAL_VARIANT, None


def _strip_md_fences(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    c = text.strip()
    if c.startswith("```"):
        lines = c.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


#: Transport-level failures seen during a tau run, by class, counted at every failed attempt
#: (including the ones a retry or the next cascade model recovered from). At TAU2_BENCH_REVISION
#: an exception from the agent, the user simulator or the judge ends the simulation attempt, and
#: tau2's run_with_retry records a simulation whose every attempt raised as an infrastructure
#: error (`infra_errors`, `harness_errors`); these counters show the failures behind those and
#: the recovered ones. They were added after a tau3 run on 2026-08-20, with an earlier tau2,
#: reported `infra_errors: 0` and an accuracy of 9.28% while hitting 4
#: ContextWindowExceededError (prompt reached 262,145 tokens against the endpoint's 262,144
#: limit) plus 1 timeout.
_LLM_FAILURES: Dict[str, int] = {}


def _note_llm_failure(exc: BaseException) -> None:
    name = type(exc).__name__
    text = f"{name}: {exc}"
    if "ContextWindowExceeded" in text:
        key = "context_window_exceeded"
    elif "Timeout" in text or "timed out" in text.lower():
        key = "timeout"
    elif "RateLimit" in text or "429" in text:
        key = "rate_limited"
    else:
        key = "other_llm_error"
    _LLM_FAILURES[key] = _LLM_FAILURES.get(key, 0) + 1


def llm_failures() -> Dict[str, int]:
    return dict(_LLM_FAILURES)


def reset_llm_failures() -> None:
    _LLM_FAILURES.clear()


def _resolve_tau_trace_dir() -> Optional[str]:
    """Where to keep tau2's per-task simulation traces (an absolute path). ON BY DEFAULT.

    Precedence:
      * `GBENCH_TAU2_SAVE_TRACES` set to a path  -> use it (explicit override);
      * `GBENCH_TAU2_SAVE_TRACES` set but empty  -> DISABLE (opt-out);
      * `GBENCH_TAU2_SAVE_TRACES` unset          -> default to `<run results dir>/tau_traces`
        (`GBENCH_RESULTS_DIR`, exported by LogManager), so tau matches every other eval and
        its traces live beside the eval JSONs. If no run dir is known (e.g. a bare unit
        test), returns None rather than dumping traces somewhere surprising.

    tau2 always writes its results somewhere (run_domain joins `save_to` onto its own
    data/simulations directory, so a relative path would land inside the tau2 checkout), so
    None does not stop the writing: `run_tau_env` then uses a temporary directory it removes
    at the end.
    """
    explicit = suite_env("GBENCH_TAU2_SAVE_TRACES", "TAU2_SAVE_TRACES")
    if explicit is not None:
        return os.path.abspath(explicit.strip()) if explicit.strip() else None
    run_dir = os.getenv("GBENCH_RESULTS_DIR")
    return os.path.abspath(os.path.join(run_dir, "tau_traces")) if run_dir else None


def _safe_name(text: Any, limit: int = 80) -> str:
    """A filesystem-safe path component."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._-")
    return (s or "x")[:limit]


def _run_tag(eval_name: str, model_name: str, enable_thinking: bool,
             run_config: Dict[str, Any]) -> Tuple[str, str]:
    """(directory name, config hash) unique to this eval, model, thinking mode, configuration
    and invocation. The invocation id (UTC time + random suffix) makes every call distinct even
    when everything else matches, so no run can ever pick up another run's simulations."""
    cfg_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    invocation = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    tag = "_".join((_safe_name(eval_name), _safe_name(model_name),
                    "think" if enable_thinking else "nothink", cfg_hash, invocation))
    return tag, cfg_hash


def _decline_prompt(*_a, **_k) -> str:
    return "n"


def _refuse_resume_prompts() -> None:
    """Answer "n" to tau2's resume prompt.

    tau2's try_resume (runner/checkpoint.py) asks `ConsoleDisplay.console.input(...)` whether to
    resume when the save file already exists, or when the run config changed. gbench runs are
    non-interactive, and a "y" would make tau2 skip every (trial, task, seed) already in the file
    and report those simulations as this run's. With auto_resume off and this answer, an
    existing file makes tau2 raise instead (which fails the domain). Re-applied on every run,
    since `_quiet_tau2_noise` may swap the console object.
    """
    try:
        import tau2.utils.display as _disp
        if hasattr(_disp, "ConsoleDisplay") and hasattr(_disp.ConsoleDisplay, "console"):
            _disp.ConsoleDisplay.console.input = _decline_prompt
    except Exception as e:                                       # noqa: BLE001
        logger.debug("tau2: could not patch the console prompt: %s", e)


#: Per thread: the Gemini role of the tau2 call in progress (set by the generate() wrappers that
#: `_install_robustness_patches` puts into the user simulator's and the judge's modules), and,
#: while `_install_judge_model_record`'s wrapper runs, the models that answered.
_ROLE_STATE = threading.local()
#: The role each tau2 module's generate() calls are made for.
_MODULE_ROLES = {"tau2.user.user_simulator": _USER_ROLE,
                 "tau2.user.user_simulator_streaming": _USER_ROLE,
                 "tau2.evaluator.evaluator_nl_assertions": _JUDGE_ROLE,
                 "tau2.environment.utils.interface_agent": "env_interface",
                 "tau2.evaluator.hallucination_reviewer": "hallucination_reviewer",
                 "tau2.agent.llm_agent": "agent",
                 "tau2.utils.llm_utils": None}
#: Replies each Gemini model gave in this process, by (domain, role, model); the result records
#: them (`gemini_models_answered`, `tau2_report.gemini_models_answered_by_role` / `_by_domain`).
_GEMINI_ANSWERS: Dict[Tuple[Optional[str], str, str], int] = {}
_GEMINI_LOCK = threading.Lock()
#: Models that answered HTTP 404 (not served for the key): skipped for the rest of the process.
_NOT_SERVED: Dict[str, str] = {}
#: Attempts per model on a transient error when a call names no num_retries (tau2's generate()
#: always passes its DEFAULT_MAX_RETRIES, 3).
_DEFAULT_NUM_RETRIES = 3


def _current_role() -> Optional[str]:
    return getattr(_ROLE_STATE, "role", None)


def _role_chain() -> List[str]:
    """The bare model names the Gemini call in progress may use: the user simulator's models for
    its calls, the NL-assertion judge's for every other gemini/ call (the judge, and tau2's
    environment interface, which _configure_evaluator_llm points at the judge's model)."""
    role = _USER_ROLE if _current_role() == _USER_ROLE else _JUDGE_ROLE
    return [_bare_model(m) for m in _role_cascade(role)]


def _is_not_served(exc: BaseException) -> bool:
    """An HTTP 404 (litellm's NotFoundError): the model is not served for this key."""
    return getattr(exc, "status_code", None) == 404 or type(exc).__name__ == "NotFoundError"


def _is_transient(exc: BaseException) -> bool:
    """LiteLLM's own rule for retrying a completion(num_retries=n) call: an openai.APIError (every
    API status, timeout and connection error LiteLLM raises is one)."""
    try:
        import openai
    except Exception:                                            # noqa: BLE001
        return False
    return isinstance(exc, openai.APIError)


def _gemini_route() -> Dict[str, str]:
    """The LiteLLM arguments of a gemini/ call: the relay's /v1beta base and the placeholder when
    this process holds the relay's placeholder (LiteLLM then sends <base>/models/<model>:
    generateContent with the key in x-goog-api-key), nothing otherwise."""
    key = _gemini_key()
    if _relay_placeholder(key):
        return {"api_base": _gemini_native_base(), "api_key": key}
    return {}


def _note_gemini_answer(model: str) -> None:
    """Count a reply `model` gave: per domain and role, for the verdict in progress
    (`_install_judge_model_record`), and in base.record_gemini_answer."""
    role = _current_role() or "other"
    domain = _CURRENT_DOMAIN.get("domain")
    with _GEMINI_LOCK:
        key = (domain, role, model)
        _GEMINI_ANSWERS[key] = _GEMINI_ANSWERS.get(key, 0) + 1
    seen = getattr(_ROLE_STATE, "answers", None)
    if seen is not None:
        seen.append(model)
    try:
        from .base import record_gemini_answer
        record_gemini_answer(model)
    except Exception:                                            # noqa: BLE001
        pass


def gemini_answers() -> Dict[str, Any]:
    """The replies of this process' run: {"total": {model: n}, "by_role": {role: {model: n}},
    "by_domain": {domain: {role: {model: n}}}}."""
    with _GEMINI_LOCK:
        items = list(_GEMINI_ANSWERS.items())
    total: Dict[str, int] = {}
    by_role: Dict[str, Dict[str, int]] = {}
    by_domain: Dict[str, Dict[str, Dict[str, int]]] = {}
    for (domain, role, model), n in sorted(items, key=lambda kv: (str(kv[0][0]), kv[0][1],
                                                                  kv[0][2])):
        total[model] = total.get(model, 0) + n
        by_role.setdefault(role, {})[model] = by_role.get(role, {}).get(model, 0) + n
        dom = by_domain.setdefault(domain or "setup", {}).setdefault(role, {})
        dom[model] = dom.get(model, 0) + n
    return {"total": total, "by_role": by_role, "by_domain": by_domain}


def reset_gemini_answers() -> None:
    with _GEMINI_LOCK:
        _GEMINI_ANSWERS.clear()
        _NOT_SERVED.clear()


def _build_gemini_cascade(inner_completion, backoff, cascade_fn=None):
    """Wrap a litellm `completion` so a gemini/<model> call goes through its role's models in
    order: the user simulator's or the NL-assertion judge's (`_role_cascade`: the role's knob's one
    model, else gbench's Gemini cascade, the same order as every other Gemini text role of gbench).
    `cascade_fn`, when given, returns the bare model names instead (tests).

    * The requested model is tried first (it is the head of its role's models), then the rest.
    * Each model gets the call's own retry budget, `num_retries` (tau2's generate() sends 3, as
      upstream): on a transient error (an openai.APIError, LiteLLM's rule) the same model is asked
      again, up to 1 + num_retries times, and only then does the next model get the call.
      LiteLLM itself is called with num_retries=0, so its internal retry (which calls
      litellm.completion again) never re-enters this cascade and never multiplies the attempts.
    * A model that answers HTTP 404 is not served for the key: it is not retried and is skipped
      for the rest of the process. The next model starts after `backoff * 2**i` s plus jitter.
    * In the image every call goes to gbench's Gemini relay (`_gemini_route`) with the relay's
      placeholder as its key.
    * The reply's model is counted (`_note_gemini_answer`), so the result names the models that
      answered. When every model fails, the last error is raised (tau2 then ends the simulation
      attempt; one whose every attempt failed is a harness error).
    Non-gemini models - the agent under test, routed as openai/<local> - pass straight through,
    once, uncascaded."""

    def _chain() -> List[str]:
        models = cascade_fn() if cascade_fn is not None else _role_chain()
        return ["gemini/" + _bare_model(m) for m in models if _bare_model(m)]

    def _completion_cascade(*args, **kwargs):
        model = _call_model(args, kwargs)
        if not _is_substitute_model(model):
            return inner_completion(*args, **kwargs)
        if args:                              # LiteLLM's positional model argument
            args = args[1:]
        kwargs.pop("model", None)
        retries = kwargs.pop("num_retries", None)
        retries = _DEFAULT_NUM_RETRIES if retries is None else max(0, int(retries))
        chain = _chain()
        ordered = [model] + [m for m in chain if m != model]
        route = _gemini_route()
        last: Optional[BaseException] = None
        asked = 0
        for m in ordered:
            bare = _bare_model(m)
            if bare in _NOT_SERVED:
                continue
            if asked:
                time.sleep(backoff * (2 ** (asked - 1)) + random.uniform(0, backoff))
            asked += 1
            for attempt in range(1 + retries):
                try:
                    resp = inner_completion(*args, **{**kwargs, **route, "model": m,
                                                      "num_retries": 0})
                except Exception as e:                           # noqa: BLE001
                    last = e
                    if _is_not_served(e):
                        _NOT_SERVED[bare] = f"{type(e).__name__}"
                        logger.warning("tau2: Gemini model %s answered HTTP 404 (the API does not "
                                       "serve it for this key); the cascade goes on without it "
                                       "for the rest of the run, see docs/evals/tau2.md", bare)
                        break
                    if attempt == 0 and not _is_transient(e):
                        break
                    if attempt < retries:
                        logger.debug("tau2: %s failed (%s); retry %d/%d on the same model", m,
                                     type(e).__name__, attempt + 1, retries)
                        continue
                    break
                else:
                    _note_gemini_answer(bare)
                    return resp
            logger.debug("tau2: judge/user model %s failed (%s)", m, type(last).__name__)
        if last is None:
            raise RuntimeError(f"none of the Gemini models {', '.join(ordered)} is served for this "
                               "key (each answered HTTP 404 earlier in the run)")
        raise last

    return _completion_cascade


def _is_substitute_model(model: Any) -> bool:
    """True for a gemini/ model, i.e. the Gemini cascade standing in for upstream's user
    simulator or judge. The agent under test is routed as openai/<model> and never matches."""
    return isinstance(model, str) and model.startswith("gemini/")


def _call_model(args: tuple, kwargs: Dict[str, Any]) -> Any:
    """The `model` argument of a litellm.completion / tau2 generate call."""
    return kwargs.get("model", args[0] if args else None)


def _install_gemini_param_guard() -> None:
    """Keep LiteLLM from adding a thinking level of its own to the gemini/ requests.

    For every Gemini 3 model LiteLLM 1.81.11 (GoogleAIStudioGeminiConfig.map_openai_params, from
    VertexGeminiConfig) sets generationConfig.thinkingConfig.thinkingLevel when the caller asked
    for none: "minimal" for a name containing gemini-3-flash, "low" for any other, so the models
    of the cascade would not all get the same parameters and none would get upstream's. Upstream
    tau2 sends the user simulator and the judge only a temperature (and the user's tools), so
    with this guard a request that names no reasoning_effort / thinking / thinkingConfig carries
    no thinkingConfig at all, for every model alike: each answers at the Gemini API's own
    default. Raises when the pinned LiteLLM has no such method (the run then hard-errors)."""
    import litellm
    config = getattr(litellm, "GoogleAIStudioGeminiConfig", None)
    orig = getattr(config, "map_openai_params", None) if config is not None else None
    if orig is None:
        raise RuntimeError("LiteLLM has no GoogleAIStudioGeminiConfig.map_openai_params, so gbench "
                           "cannot keep it from adding a thinking level to the Gemini user "
                           "simulator's and judge's requests")
    if getattr(orig, "_gbench_no_default_thinking", False):
        return

    def map_openai_params(self, *args, **kwargs):
        bound = dict(zip(("non_default_params", "optional_params", "model", "drop_params"), args))
        bound.update(kwargs)
        asked = bool(set(bound.get("non_default_params") or {})
                     & {"reasoning_effort", "thinking"})
        preset = "thinkingConfig" in (bound.get("optional_params") or {})
        out = orig(self, *args, **kwargs)
        if not asked and not preset and isinstance(out, dict):
            out.pop("thinkingConfig", None)
        return out

    map_openai_params._gbench_no_default_thinking = True
    config.map_openai_params = map_openai_params


def _install_judge_model_record() -> bool:
    """Record in each simulation which Gemini model gave its NL-assertion verdict: tau2's
    NLAssertionsEvaluator.calculate_reward is wrapped to add `judge_model` (and `judge_models`
    when replies came from more than one) to the RewardInfo it returns, which tau2 keeps as
    reward_info.info["nl"] in results.json. Returns whether it is installed."""
    try:
        from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
    except Exception as e:                                       # noqa: BLE001
        logger.warning("tau2: could not record the NL-assertion judge's model per verdict: %s", e)
        return False
    func = getattr(NLAssertionsEvaluator.__dict__.get("calculate_reward"), "__func__", None)
    if func is None:
        logger.warning("tau2: NLAssertionsEvaluator.calculate_reward is not a classmethod here; "
                       "the judge's model is not recorded per verdict")
        return False
    if getattr(func, "_gbench_judge_model", False):
        return True

    def calculate_reward(cls, *args, **kwargs):
        previous = getattr(_ROLE_STATE, "answers", None)
        _ROLE_STATE.answers = []
        try:
            reward_info = func(cls, *args, **kwargs)
            models = list(_ROLE_STATE.answers)
        finally:
            _ROLE_STATE.answers = previous
        if models and reward_info is not None:
            extra: Dict[str, Any] = {"judge_model": models[-1]}
            if len(set(models)) > 1:
                extra["judge_models"] = models
            try:
                reward_info.info = dict(getattr(reward_info, "info", None) or {}, **extra)
            except Exception:                                    # noqa: BLE001
                pass
        return reward_info

    calculate_reward._gbench_judge_model = True
    NLAssertionsEvaluator.calculate_reward = classmethod(calculate_reward)
    return True


def _verdict_judge_model(sim: Any) -> Optional[str]:
    """The model of a simulation's NL-assertion verdict (`_install_judge_model_record`), if any."""
    info = getattr(getattr(sim, "reward_info", None), "info", None)
    if not isinstance(info, dict):
        return None
    nl = info.get("nl") if isinstance(info.get("nl"), dict) else info
    model = nl.get("judge_model") if isinstance(nl, dict) else None
    return model if isinstance(model, str) and model else None


def _install_robustness_patches(retries: int = 3) -> None:
    """The patches that run the Gemini substitution for upstream's user simulator and judge.

    * LiteLLM's thinking-level default for Gemini 3 models is removed (`_install_gemini_param_
      guard`, required: raises when it cannot be installed).
    * litellm.completion (and litellm.main.completion, which LiteLLM's own retries call) counts
      every failed attempt (`llm_failures`) and re-requests a gemini/ reply that is empty
      (no text but whitespace and no tool call) or carries empty / unparseable tool arguments, up
      to `retries` attempts in all.
    * tau2's chokepoint, tau2.utils.llm_utils.completion (the only completion call tau2's
      generate() makes), runs gemini/ calls through the role's models (`_build_gemini_cascade`).
      Only there: LiteLLM's internal retries call litellm.completion, which has no cascade.
    * The generate() of each tau2 module that calls it records the role of its calls
      (`_MODULE_ROLES`) and, for gemini/ calls, strips stray markdown fences from the text.
    * Each simulation's NL-assertion verdict records the model that gave it
      (`_install_judge_model_record`).
    The agent (the model under test) goes through unchanged, sent once, as in upstream tau2."""
    global _PATCHED
    if _PATCHED:
        return
    _install_gemini_param_guard()
    _PATCHED = True

    # --- litellm.completion: count failures; re-request an empty/malformed gemini/ reply ---
    _completion_retry = None
    try:
        import json as _json
        import litellm
        _orig_completion = litellm.completion

        def _completion_retry(*args, **kwargs):
            try:
                resp = _orig_completion(*args, **kwargs)
            except BaseException as e:      # noqa: BLE001 - record, then re-raise unchanged
                _note_llm_failure(e)
                raise
            if not _is_substitute_model(_call_model(args, kwargs)):
                return resp                 # the agent: sent once, as upstream
            for attempt in range(retries - 1):
                try:
                    msg = resp.choices[0].message
                    tool_calls = getattr(msg, "tool_calls", None)
                    if not (msg.content and str(msg.content).strip()) and not tool_calls:
                        raise ValueError("empty response")
                    if tool_calls:
                        for tc in tool_calls:
                            a = tc.function.arguments
                            if not a:
                                raise ValueError("empty tool arguments")
                            _json.loads(a)  # unparseable -> retry
                    return resp
                except (IndexError, AttributeError):
                    return resp  # unexpected shape; hand back untouched
                except (ValueError, TypeError):
                    logger.debug("tau2: empty/malformed response, retry %d/%d", attempt + 2, retries)
                    try:
                        resp = _orig_completion(*args, **kwargs)
                    except BaseException as e:  # noqa: BLE001 - record, then re-raise
                        _note_llm_failure(e)
                        raise
            return resp

        litellm.completion = _completion_retry
        _RETRY_PATCHES["completion"] = True
        try:
            import litellm.main
            litellm.main.completion = _completion_retry
        except Exception:
            pass
    except Exception as e:
        logger.warning("tau2: could not install litellm retry patch: %s", e)

    # --- the role cascade, at tau2's chokepoint only (never on litellm.completion) ---
    try:
        import tau2.utils.llm_utils as _llm
        _cascade_backoff = float(suite_env("GBENCH_TAU2_LLM_BACKOFF", "TAU2_LLM_BACKOFF",
                                           default="1.0"))
        _llm.completion = _build_gemini_cascade(_completion_retry or _llm.completion,
                                                _cascade_backoff)
    except Exception as e:
        logger.warning("tau2: could not install judge/user cascade: %s", e)

    # --- tau2 generate(): the role of the call; for gemini/ roles, strip markdown fences ---
    try:
        importlib.import_module("tau2.utils.llm_utils")
        for modname, role in _MODULE_ROLES.items():
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            if not hasattr(mod, "generate"):
                continue
            mod.generate = _generate_wrapper(mod.generate, role)
            _RETRY_PATCHES["generate"] = True
    except Exception as e:
        logger.warning("tau2: could not install generate() patch: %s", e)

    _install_judge_model_record()


def _generate_wrapper(orig_generate, role: Optional[str]):
    """tau2's generate() for one module: sets the call's role (`_ROLE_STATE`) and strips stray
    markdown fences from a gemini/ reply's text. Failures are counted by the completion patch."""

    def _generate_retry(*args, **kwargs):
        previous = _current_role()
        if role is not None:
            _ROLE_STATE.role = role
        try:
            res = orig_generate(*args, **kwargs)
        finally:
            _ROLE_STATE.role = previous
        if not _is_substitute_model(_call_model(args, kwargs)):
            return res                      # the agent: unchanged, as upstream
        content = getattr(res, "content", None)
        if content is not None:
            stripped = _strip_md_fences(content)
            if stripped != content:
                try:
                    res.content = stripped
                except Exception:
                    pass
        return res

    _generate_retry._gbench_role = role
    return _generate_retry


#: upstream's DEFAULT_LLM_NL_ASSERTIONS_ARGS (src/tau2/config.py): the judge's call arguments.
_NL_ASSERTIONS_ARGS = {"temperature": 0.0}


def _configure_evaluator_llm(eval_model: str) -> None:
    """Point tau2's nl-assertion evaluator + env-interface LLMs at `eval_model`.

    `from X import Y` copies values at import time, so we patch both the config module
    and the already-imported evaluator module. Best-effort across versions.
    """
    eval_args = dict(_NL_ASSERTIONS_ARGS)
    try:
        import tau2.config as cfg
        for attr, val in (("DEFAULT_LLM_NL_ASSERTIONS", eval_model),
                          ("DEFAULT_LLM_NL_ASSERTIONS_ARGS", eval_args),
                          ("DEFAULT_LLM_ENV_INTERFACE", eval_model),
                          ("DEFAULT_LLM_ENV_INTERFACE_ARGS", eval_args)):
            if hasattr(cfg, attr):
                setattr(cfg, attr, val)
    except Exception as e:
        logger.warning("tau2: could not set evaluator config: %s", e)
    try:
        import tau2.evaluator.evaluator_nl_assertions as nl
        if hasattr(nl, "DEFAULT_LLM_NL_ASSERTIONS"):
            nl.DEFAULT_LLM_NL_ASSERTIONS = eval_model
            nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = eval_args
    except Exception:
        pass


#: What `_build_agent_args` last resolved, so the result can record it (13).
_LAST_AGENT_TEMPERATURE: Dict[str, Any] = {}

#: tau2-bench's agent sampling default: src/tau2/config.py DEFAULT_LLM_TEMPERATURE_AGENT = 0.0,
#: DEFAULT_LLM_ARGS_AGENT = {"temperature": 0.0} (unchanged through v1.0.1). Only temperature
#: is set upstream; top_p / top_k are left to the serving stack.
_UPSTREAM_AGENT_TEMPERATURE = 0.0


def _apply_legacy_temperature_alias(eval_name: str) -> None:
    """Copy the deprecated bare TAU2_TEMPERATURE onto the canonical per-suite name when only the
    legacy one is set, so the standard resolver owns the precedence."""
    from .base import temperature_env_var
    canon = temperature_env_var(eval_name or "tau2")   # GBENCH_TAU2_TEMPERATURE / GBENCH_TAU3_TEMPERATURE
    legacy = os.getenv("TAU2_TEMPERATURE")
    if legacy and not os.getenv(canon):
        os.environ[canon] = legacy
        logger.warning("env var TAU2_TEMPERATURE is a deprecated alias; use %s instead.", canon)


def _build_agent_args(base_url: str, enable_thinking: bool,
                      eval_name: str = "tau2") -> Dict[str, Any]:
    """LiteLLM args routing ONLY the agent to the gbench endpoint, per-call.

    The endpoint key is not among them: tau2 records llm_args in every results.json, so the key
    is added to each agent call by `_agent_guard` instead."""
    # The agent is the model under test, so it must honour the same precedence as every
    # other suite, owned by the standard resolver: GBENCH_<EVAL>_TEMPERATURE (per-suite env
    # override) > --temperature > the maintainer's rule: 1.0 for a --thinking run, 0.0 for a
    # no-think run (which is also upstream's agent default). The bare TAU2_TEMPERATURE is now a
    # deprecated ALIAS of the canonical GBENCH_TAU2_TEMPERATURE (it no longer outranks the
    # canonical name or the CLI): if it is set and the canonical is not, we copy it onto the
    # canonical name and let the resolver decide. The evaluator/judge LLM stays at 0.0 and is
    # deliberately NOT routed through this - grading should not drift with the run's sampling
    # knob. The suite value passed is upstream's agent default (temperature 0.0, nothing else),
    # recorded only; the rule's no-think 0.0 samples the agent the way `tau2 run` does. Resolving under `eval_name`
    # keeps GBENCH_TAU3_TEMPERATURE working when tau3 delegates here.
    from .base import resolve_temperature
    _apply_legacy_temperature_alias(eval_name)
    agent_temp, temp_source = resolve_temperature(eval_name or "tau2", _UPSTREAM_AGENT_TEMPERATURE,
                                                  thinking=bool(enable_thinking))
    _LAST_AGENT_TEMPERATURE["value"] = agent_temp
    _LAST_AGENT_TEMPERATURE["source"] = temp_source
    args: Dict[str, Any] = {
        "temperature": agent_temp,
        "api_base": base_url,
    }
    # top_p / top_k are opt-in: upstream sets neither, so by default the server's own defaults
    # (the model's generation_config) apply, exactly as for a `tau2 run` agent.
    top_p = suite_env("GBENCH_TAU2_TOP_P", "TAU2_TOP_P")
    if top_p:
        args["top_p"] = float(top_p)
    extra_body: Dict[str, Any] = {}
    top_k = suite_env("GBENCH_TAU2_TOP_K", "TAU2_TOP_K")
    if top_k:
        extra_body["top_k"] = int(top_k)
    # The chat template's thinking switch with --thinking; a no-think run sends none, as
    # upstream (the served model's default, off for Gemma 4). agent_wire_record records it.
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        args["extra_body"] = extra_body
    # The agent's request timeout (agent_client_timeout): no hidden 600 s LiteLLM default.
    args["timeout"] = agent_client_timeout()["timeout_s"]
    _LAST_AGENT_TEMPERATURE["top_p"] = args.get("top_p")
    _LAST_AGENT_TEMPERATURE["top_k"] = extra_body.get("top_k")
    return args


#: Per-domain pass^k (fractions, unrounded) from tau2's own metrics, filled by `_run_one_domain`.
_DOMAIN_PASS_HAT_K: Dict[str, Dict[str, float]] = {}
#: Per-domain results.json each run wrote, filled by `_run_one_domain`.
_DOMAIN_TRACES: Dict[str, str] = {}
#: Per domain: {task id: [evaluable trials, successful trials]}, filled by `_run_one_domain`.
_DOMAIN_TASK_OUTCOMES: Dict[str, Dict[str, List[int]]] = {}
#: Per domain: {model: simulations whose NL-assertion verdict it gave}, read from each
#: simulation's reward info (`_install_judge_model_record`), filled by `_run_one_domain`.
_DOMAIN_JUDGE_MODELS: Dict[str, Dict[str, int]] = {}
#: Per-domain simulations tau2 recorded as infrastructure errors, filled by `_run_one_domain`:
#: `_infra_error_summary` of all of them, plus "harness" and "agent_turn", the same summary of
#: the ones from a harness failure and of the ones raised on the agent's own turn
#: (`_AGENT_TURN_SOURCES`). tau2's run_with_retry records one when every attempt of a simulation
#: raised (an outage of the Gemini user simulator or judge, of the model endpoint, a failed Gemini
#: embedding call, an agent reply tau2 rejects, ...), with the last error's class, message and
#: traceback in `info`.
_DOMAIN_INFRA: Dict[str, Dict[str, Any]] = {}
#: Per domain: simulations tau2 ended with USER_ERROR (orchestrator.py
#: _check_communication_error: the user simulator sent a message with both text and tool calls,
#: or an empty one), which its evaluator scores 0. The user simulator is gbench's Gemini
#: substitute, so such a simulation is a harness error (kind user_simulator_protocol): left out
#: of pass^k and the rewards, counted and named.
_DOMAIN_USER_ERRORS: Dict[str, int] = {}
#: How much of one infrastructure error's message the result keeps.
_INFRA_ERROR_TEXT_CHARS = 300
#: Where an infrastructure error came from (`_infra_error_source`, read from the traceback tau2's
#: run_with_retry keeps for the simulation's last attempt). These two are the agent's own turn,
#: not harness failures: "agent_reply" (tau2 rejected the agent's reply: a message with neither
#: text nor tool calls fails AssistantMessage.validate(), tool-call arguments that are not JSON
#: fail json.loads in tau2's generate) and "agent_context_overflow" (the agent's request was over
#: the model endpoint's context window). Every other source is a harness failure:
#: "model_endpoint" (the request to the model endpoint failed otherwise: connection, timeout, HTTP
#: error), "user_simulator" (the Gemini user simulator's call failed, or it replied with neither
#: text nor tool calls), "grading" (tau2's grading after the conversation: the Gemini
#: NL-assertion judge, or replaying the conversation's tool calls), "tool" (a tool call raised out
#: of the environment), "gemini_embeddings" (a Gemini embedding call failed after its retries:
#: TauHarnessError, GeminiEmbeddingError), "other" (any other exception, none of it on the
#: agent's turn, such as building the environment) and "unrecorded" (no traceback recorded).
_AGENT_TURN_SOURCES = ("agent_reply", "agent_context_overflow")

#: tau2-bench's upstream defaults (src/tau2/config.py at TAU2_BENCH_REVISION) that gbench
#: substitutes, used to name each substitution in `leaderboard_comparable_reason`.
_UPSTREAM_USER_LLM = "gpt-4.1-2025-04-14"
_UPSTREAM_NL_ASSERTIONS_LLM = "gpt-4.1-2025-04-14"
#: The user simulator recommended by docs/leaderboard-submission.md at TAU2_BENCH_REVISION and used
#: by Sierra's 2026 standard text entries (web/leaderboard/public/submissions; one other 2026
#: standard entry used the config default above).
_LEADERBOARD_USER_LLM = "gpt-5.2"
#: docs/leaderboard-submission.md: "4+ trials" per domain, pass_1..pass_4 reported per domain.
_LEADERBOARD_MIN_TRIALS = 4
_LEADERBOARD_TRIALS = 4


def _resolve_trials() -> Tuple[int, Optional[Dict[str, Any]]]:
    """(trials per task, the result's `attempts` record or None).

    GBENCH_TAU2_NUM_TRIALS sets the trials when it is set. Otherwise the leaderboard's 4, raised
    to --attempt-count when that asks for more: tau2's trials are its attempts (pass^k over k
    trials), and a smaller --attempt-count never cuts the protocol's 4. With --attempt-count
    above 1 the record says how many trials ran and why."""
    from .base import get_run_knob
    try:
        requested = max(1, int(get_run_knob("attempt_count", 1) or 1))
    except (TypeError, ValueError):
        requested = 1
    explicit = suite_env("GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS")
    if explicit:
        trials, source = int(explicit), "GBENCH_TAU2_NUM_TRIALS"
    elif requested > _LEADERBOARD_TRIALS:
        trials, source = requested, "--attempt-count (above the leaderboard's 4 trials)"
    else:
        trials, source = _LEADERBOARD_TRIALS, "the leaderboard's 4 trials per task"
    record = None
    if requested > 1:
        record = {"attempts_requested": requested, "attempts_per_sample": trials,
                  "source": source,
                  "note": ("tau2 runs every task `num_trials` times and reports pass^1..pass^k "
                           "over them; --attempt-count maps to num_trials, never below the "
                           f"leaderboard's {_LEADERBOARD_TRIALS}")}
    return trials, record


def _num_trials() -> int:
    """Trials per task (_resolve_trials)."""
    return _resolve_trials()[0]


def _shard_task_ids(domain: str, shard: Optional[str], limit: Optional[int],
                    seed: str = "tau2") -> Tuple[List[str], int]:
    """(the task ids of `domain` this run selects, the domain's task count): every task of the
    split tau2 runs (TextRunConfig's task_split_name default, "base") in tau2's own load order,
    round-robin shard I of N when `shard` is set (sampling.shard_select), then a seeded draw of
    `limit` of them (sampling.stratified_sample, seed `seed`: the suite name), not tau2's
    --num-tasks head of the file order. The run passes the ids to tau2 as task_ids."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.helpers import get_tasks
    from .sampling import parse_shard, shard_select, stratified_sample
    split = TextRunConfig.model_fields["task_split_name"].default
    ids = [str(t.id) for t in get_tasks(task_set_name=domain, task_split_name=split)]
    selected = shard_select(ids, parse_shard(shard)) if shard else list(ids)
    if limit and limit > 0:
        selected = stratified_sample(selected, int(limit), seed=seed)
    return selected, len(ids)


def domain_limit_shares(limit: Optional[int], domains: List[str], available: Dict[str, int],
                        seed: str) -> Dict[str, Optional[int]]:
    """{domain: tasks it runs}: --eval-limit N is N tasks for the whole run, never N per domain,
    dealt over the domains by sampling.allocate_limit (seed: the suite name; each domain capped at
    the tasks it lists), so a domain gets 0 when N is below the number of domains. None for every
    domain without a limit."""
    from .sampling import allocate_limit
    return allocate_limit(limit, list(domains), available=available or None, seed=seed)


def _plan_counts(domains: List[str], listed: Dict[str, List[str]],
                 shares: Dict[str, Optional[int]], selecting: Any) -> Optional[Dict[str, int]]:
    """{domain: tasks this run holds} for the plan: the listed tasks capped by the domain's
    share; None for a full run, whose task lists are tau2's own (not listed here)."""
    if not selecting:
        return None
    return {d: (len(listed.get(d) or []) if shares.get(d) is None
                else min(int(shares[d]), len(listed.get(d) or []))) for d in domains}


def _limit_plan(eval_name: str, limit: Optional[int], planned: Optional[Dict[str, int]],
                trials: int, domains: Sequence[str] = ()) -> Dict[str, Any]:
    """Log the run's planned workload (one line) and return its `eval_limit_plan`: the tasks per
    domain x the trials per task (tau2's num_trials: repeats of each task, not extra samples).
    A full run lets tau2 run every task of each domain, so its counts are tau2's own."""
    if planned is not None:
        from .limit_plan import eval_limit_plan
        return eval_limit_plan(eval_name, limit, sum(planned.values()), planned,
                               setting_noun="domain", attempts=trials, unit="conversations")
    logger.info("[%s] full run (no --eval-limit): every task of %d domain(s) (%s) x %d trial(s)",
                eval_name, len(domains), ", ".join(domains), trials)
    return {"limit": None, "samples": None, "per_setting": None,
            "attempts_per_sample": trials, "conversations": None}


def _max_steps() -> int:
    return int(suite_env("GBENCH_TAU2_MAX_STEPS", "TAU2_MAX_STEPS", default="200"))


def _max_errors() -> int:
    return int(suite_env("GBENCH_TAU2_MAX_ERRORS", "TAU2_MAX_ERRORS", default="10"))


def _seed() -> int:
    return int(suite_env("GBENCH_TAU2_SEED", "TAU2_SEED", default="300"))


def _user_temperature() -> float:
    return float(suite_env("GBENCH_TAU2_USER_TEMPERATURE", "TAU2_USER_TEMPERATURE", default="0.0"))


def _tau2_version() -> Optional[str]:
    """The tau2-bench version that runs: the checkout's pyproject.toml, else dist metadata."""
    try:
        version = tau2_provenance().get("version")
        if version:
            return version
    except Exception:                                            # noqa: BLE001
        pass
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version("tau2")
    except Exception:                                            # noqa: BLE001
        return None


def _leaderboard_track(eval_name: str, domains: List[str]) -> str:
    """The leaderboard track the headline belongs to (tracks as of TAU2_BENCH_REVISION)."""
    if list(domains) == list(KNOWLEDGE_DOMAINS):
        return "tau3-banking (banking_knowledge pass^k)"
    if sorted(domains) == sorted(CORE_DOMAINS):
        return "tau2-bench (Overall = unweighted mean of retail, airline, telecom pass^k)"
    return (f"tau2-bench per-domain column(s) {', '.join(domains)} (the track Overall needs "
            "retail, airline and telecom)")


def _gemini_call_parameters() -> Dict[str, Any]:
    """What the gemini/ user-simulator and judge calls send besides the conversation (and the
    user's tools), recorded on the result. Upstream's tau2 sends its user simulator and judge a
    temperature and nothing else; so does gbench, for every model of the cascade alike."""
    relayed = _relay_placeholder(_gemini_key())
    return {
        "user_simulator": {"temperature": _user_temperature()},
        "nl_assertion_judge": dict(_NL_ASSERTIONS_ARGS),
        "thinking": ("none sent: no reasoning_effort or thinkingConfig, and LiteLLM's default "
                     "thinkingLevel for Gemini 3 models is removed "
                     "(_install_gemini_param_guard), so every model answers at the Gemini "
                     "API's own default"),
        "attempts_per_model": ("1 + the call's num_retries (tau2's generate() sends 3) on an "
                               "openai.APIError, LiteLLM's retry rule; an HTTP 404 skips the "
                               "model; LiteLLM's own retries are off (num_retries=0)"),
        "api": ("POST /v1beta/models/<model>:generateContent through gbench's Gemini relay on "
                "the docker bridge gateway, which adds the key" if relayed else
                "LiteLLM's gemini/ provider, straight to the Gemini API"),
    }


def _role_text(role: str) -> str:
    """How a role's models read in leaderboard_comparable_reason: "gbench's Gemini cascade a > b >
    c", or "<model> alone (GBENCH_TAU2_USER_LLM)" when its knob names one."""
    override, knob = _role_override(role)
    if override:
        return f"{_bare_model(override)} alone ({knob})"
    return "gbench's Gemini cascade " + " > ".join(_bare_model(m) for m in _role_cascade(role))


def _comparability_gaps(domains: List[str], trials: int, limit: Optional[int],
                        infra: int,
                        infra_by_domain: Optional[Dict[str, Dict[str, Any]]] = None
                        ) -> List[str]:
    """The ways this run's setup differs from a tau2-bench leaderboard submission, as far as
    gbench can detect them. `infra` is the number of simulations that ended in an infrastructure
    error, and `infra_by_domain` their per-domain detail (`_DOMAIN_INFRA`'s shape)."""
    gaps: List[str] = []
    if trials < _LEADERBOARD_MIN_TRIALS:
        gaps.append(f"{trials} trial(s) per task, while the tau2-bench leaderboard ranks pass^k "
                    f"over {_LEADERBOARD_MIN_TRIALS}+ trials (GBENCH_TAU2_NUM_TRIALS is set below "
                    f"the default {_LEADERBOARD_TRIALS}, so pass^k stops at pass^{trials})")
    if limit and limit > 0:
        gaps.append("an --eval-limit task subset (the leaderboard requires every task of a domain)")
    user = _user_llm()
    if user.split("/", 1)[-1] not in (_UPSTREAM_USER_LLM, _LEADERBOARD_USER_LLM):
        cascade = (f" ({_role_text(_USER_ROLE)}; an empty or malformed reply is requested "
                   "again, up to 3 attempts)" if _is_substitute_model(user) else "")
        gaps.append(f"user simulator {user}{cascade} instead of {_LEADERBOARD_USER_LLM} "
                    "(recommended by docs/leaderboard-submission.md and used by Sierra's 2026 "
                    f"standard text entries) or src/tau2/config.py's default {_UPSTREAM_USER_LLM}; "
                    "the submission guide counts a non-default user simulator as a custom "
                    "submission and the leaderboard reports each entry's user simulator, so "
                    "compare only with entries that used the same one")
    user_temp = _user_temperature()
    if user_temp != 0.0:
        gaps.append(f"user simulator at temperature {user_temp} (upstream's "
                    "DEFAULT_LLM_TEMPERATURE_USER is 0.0)")
    evaluator = _eval_llm()
    if evaluator.split("/", 1)[-1] != _UPSTREAM_NL_ASSERTIONS_LLM:
        which = f" ({_role_text(_JUDGE_ROLE)})" if _is_substitute_model(evaluator) else ""
        gaps.append(f"NL-assertion judge {evaluator}{which} instead of upstream's "
                    f"{_UPSTREAM_NL_ASSERTIONS_LLM}; it grades every task whose reward basis "
                    "includes NL_ASSERTION")
    temp = _LAST_AGENT_TEMPERATURE.get("value")
    if (temp is not None and temp != _UPSTREAM_AGENT_TEMPERATURE) \
            or _LAST_AGENT_TEMPERATURE.get("top_p") is not None \
            or _LAST_AGENT_TEMPERATURE.get("top_k") is not None:
        why = (" (--thinking samples the model under test at 1.0)"
               if _LAST_AGENT_TEMPERATURE.get("source") == "thinking" else "")
        gaps.append(f"agent sampled at temperature {temp}{why}"
                    f" top_p {_LAST_AGENT_TEMPERATURE.get('top_p')}"
                    f" top_k {_LAST_AGENT_TEMPERATURE.get('top_k')} "
                    f"(upstream sends only temperature {_UPSTREAM_AGENT_TEMPERATURE})")
    if "banking_knowledge" in domains:
        gaps.append(GEMINI_EMBED_REASON)
    gaps.extend(_embedding_failure_gaps(domains))
    if infra:
        by_domain = infra_by_domain or {}
        n_agent = sum(_infra_parts(by_domain.get(d) or {})["agent_turn"]["count"]
                      for d in domains)
        n_harness = infra - n_agent
        if n_harness > 0:
            detail = _infra_detail(domains, infra_by_domain, "harness")
            gaps.append(f"{n_harness} simulation(s) ended in infrastructure errors from harness "
                        f"failures, not the agent's outcome{f' ({detail})' if detail else ''}: "
                        "tau2's metrics exclude them, so pass^k covers fewer than every task x "
                        "trial")
        if n_agent:
            detail = _infra_detail(domains, infra_by_domain, "agent_turn")
            gaps.append(f"{n_agent} simulation(s) ended in infrastructure errors raised on the "
                        f"agent's own turn ({detail}), not harness failures: tau2's metrics "
                        "exclude them as upstream does, so pass^k covers fewer than every task x "
                        "trial and does not count these simulations against the model")
    return gaps


def _error_types_text(error_types: Optional[Dict[str, int]]) -> str:
    """`{"APIConnectionError": 2}` -> "APIConnectionError x2" (most frequent first)."""
    items = sorted((error_types or {}).items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name} x{count}" for name, count in items)


def _infra_parts(info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{"harness": ..., "agent_turn": ...}: the two parts of a `_DOMAIN_INFRA` entry, each with
    at least a count. An entry without them (its count arrived without per-simulation detail) is
    all harness: nothing in it shows an error raised on the agent's turn."""
    agent = info.get("agent_turn") or {"count": 0}
    harness = info.get("harness") or {
        "count": max(0, int(info.get("count") or 0) - int(agent.get("count") or 0)),
        "error_types": info.get("error_types") or {}, "sources": info.get("sources") or {},
        "causes": info.get("causes") or {}, "example_error": info.get("example_error")}
    return {"harness": harness, "agent_turn": agent}


def _infra_detail(domains: List[str],
                  infra_by_domain: Optional[Dict[str, Dict[str, Any]]],
                  part: Optional[str] = None) -> str:
    """Per domain of `domains`: its infrastructure-error simulations (only the "harness" or the
    "agent_turn" part when `part` names one) with the source and class of their errors, e.g.
    "retail 2 [model_endpoint/APIConnectionError x2], airline 1 [grading/Timeout x1]" (the
    classes alone when the sources are not known)."""
    parts = []
    for domain in domains:
        info = (infra_by_domain or {}).get(domain) or {}
        if part:
            info = _infra_parts(info)[part]
        if not info.get("count"):
            continue
        types = _error_types_text(info.get("causes") or info.get("error_types"))
        parts.append(f"{domain} {info['count']}{f' [{types}]' if types else ''}")
    return ", ".join(parts)


def _embedding_failure_gaps(domains: List[str]) -> List[str]:
    """One entry per domain of `domains` whose simulations hit a Gemini embedding call that
    failed after its retries (a harness error)."""
    with _HARNESS_LOCK:
        failures = {d: n for d, n in _EMBEDDING_FAILURES.items() if d in domains}
        aborts = dict(_EMBEDDING_ABORTS)
    return [f"{count} Gemini embedding call(s) ({GEMINI_EMBED_MODEL}) failed in {domain} after "
            f"their retries: harness errors, not the agent's outcome "
            f"({aborts.get(domain, 0)} simulation attempt(s) stopped there and re-run by tau2, "
            "which counts a simulation whose every attempt failed as an infrastructure error)"
            for domain, count in sorted(failures.items()) if count]


#: The `error` text of the per-domain infrastructure-error records.
_HARNESS_INFRA_RECORD = ("simulations ended in an infrastructure error from a harness failure "
                         "(tau2 records one when every attempt of a simulation failed; "
                         "compute_metrics leaves it out of pass^k)")
_AGENT_TURN_INFRA_RECORD = ("simulations ended in an infrastructure error raised on the agent's "
                            "own turn, not a harness failure (tau2 records one when every attempt "
                            "of a simulation failed; compute_metrics leaves it out of pass^k, as "
                            "upstream does)")


def _infra_records(domains: List[str],
                   infra_by_domain: Optional[Dict[str, Dict[str, Any]]], part: str,
                   error: str) -> List[Dict[str, Any]]:
    """One record per domain of `domains` whose `part` ("harness" or "agent_turn") of the
    infrastructure-error simulations is not empty."""
    records: List[Dict[str, Any]] = []
    for domain in domains:
        info = _infra_parts((infra_by_domain or {}).get(domain) or {})[part]
        if not info.get("count"):
            continue
        records.append({
            "error": error, "domain": domain, "infrastructure_error_simulations": info["count"],
            "sources": dict(info.get("sources") or {}),
            "error_types": dict(info.get("error_types") or {}),
            "example_error": info.get("example_error")})
    return records


def _harness_error_records(domains: List[str],
                           infra_by_domain: Optional[Dict[str, Dict[str, Any]]] = None
                           ) -> List[Dict[str, Any]]:
    """The result's `harness_errors` entries: one per domain of `domains` with failed Gemini
    embedding calls, then one per domain of `domains` with simulations that ended in an
    infrastructure error from a harness failure (`infra_by_domain`, `_DOMAIN_INFRA`'s shape: an
    outage of the Gemini user simulator or judge, of the model endpoint, failed embedding
    calls, ...; every source but `_AGENT_TURN_SOURCES`)."""
    with _HARNESS_LOCK:
        failures = {d: n for d, n in _EMBEDDING_FAILURES.items() if d in domains}
        aborts = dict(_EMBEDDING_ABORTS)
    records: List[Dict[str, Any]] = [
        {"error": f"Gemini embedding calls ({GEMINI_EMBED_MODEL}) failed after their retries",
         "domain": domain, "failed_embedding_calls": count,
         "stopped_simulation_attempts": aborts.get(domain, 0)}
        for domain, count in sorted(failures.items()) if count]
    records.extend(_infra_records(domains, infra_by_domain, "harness", _HARNESS_INFRA_RECORD))
    return records


def _agent_turn_error_records(domains: List[str],
                              infra_by_domain: Optional[Dict[str, Dict[str, Any]]] = None
                              ) -> List[Dict[str, Any]]:
    """The result's `agent_turn_errors` entries: one per domain of `domains` with simulations
    that ended in an infrastructure error raised on the agent's own turn (`_AGENT_TURN_SOURCES`:
    an agent reply tau2 rejects, an agent request over the endpoint's context window). They are
    not harness failures; tau2 leaves them out of pass^k as it does every infrastructure error."""
    return _infra_records(domains, infra_by_domain, "agent_turn", _AGENT_TURN_INFRA_RECORD)


def _is_infra_error(sim: Any) -> bool:
    """tau2 records a simulation whose retries were exhausted as INFRASTRUCTURE_ERROR, with no
    reward_info; compute_metrics leaves such simulations out."""
    reason = getattr(sim, "termination_reason", None)
    reason = getattr(reason, "value", reason)
    return reason == "infrastructure_error" or getattr(sim, "reward_info", None) is None


def _is_user_error(sim: Any) -> bool:
    """tau2 ended the simulation with USER_ERROR: the user simulator broke the protocol
    (orchestrator.py _check_communication_error), a failure of gbench's substituted Gemini user
    simulator, never the agent's outcome."""
    reason = getattr(sim, "termination_reason", None)
    return getattr(reason, "value", reason) == "user_error"


def _with_simulations(results: Any, sims: List[Any]) -> Any:
    """`results` (tau2's Results) with only `sims` as its simulations."""
    copy_with = getattr(results, "model_copy", None)
    if callable(copy_with):
        return copy_with(update={"simulations": list(sims)})
    import copy as _copy
    out = _copy.copy(results)
    out.simulations = list(sims)
    return out


_TRACEBACK_HEADER = "Traceback (most recent call last):"
_TRACEBACK_FRAME = re.compile(r'^\s*File "([^"]+)", line \d+, in (.+)$')


def _traceback_frames(text: str) -> List[Tuple[str, str, str]]:
    """(file, function, source line) of each frame of the traceback of the exception
    run_with_retry caught, outermost first. traceback.format_exc() prints chained causes first
    and that exception's traceback last; it is the last one whose frames include tau2's
    runner/progress.py (run_with_retry calls every attempt from there)."""
    blocks = []
    for block in (text or "").split(_TRACEBACK_HEADER)[1:]:
        frames: List[List[str]] = []
        for line in block.splitlines():
            match = _TRACEBACK_FRAME.match(line)
            if match:
                frames.append([match.group(1).replace("\\", "/"), match.group(2).strip(), ""])
            elif frames and line and not line.startswith(" "):
                break                               # the exception line ends the frames
            elif frames and not frames[-1][2] and line.strip() \
                    and not set(line.strip()) <= set("^~"):
                frames[-1][2] = line.strip()
        if frames:
            blocks.append([tuple(f) for f in frames])
    for frames in reversed(blocks):
        if any(path.endswith("/tau2/runner/progress.py") for path, _f, _s in frames):
            return frames
    return blocks[-1] if blocks else []


def _infra_error_source(sim: Any) -> str:
    """Where a simulation's infrastructure error came from (see `_AGENT_TURN_SOURCES` for the
    names): read from the class and the traceback of the last attempt's error, which tau2's
    run_with_retry keeps in `info`, at the innermost frame that shows whose turn it was."""
    info = getattr(sim, "info", None)
    info = info if isinstance(info, dict) else {}
    error_type = str(info.get("error_type") or "")
    if error_type in (TauHarnessError.__name__, GeminiEmbeddingError.__name__):
        return "gemini_embeddings"
    frames = _traceback_frames(str(info.get("error_traceback") or ""))
    if not frames:
        return "unrecorded"
    overflow = "ContextWindowExceeded" in f"{error_type} {info.get('error') or ''}"
    for i in range(len(frames) - 1, -1, -1):
        path, func, source = frames[i]
        if "/tau2/evaluator/" in path:
            return "grading"
        if "/tau2/user/" in path or source.startswith("user_msg.validate("):
            return "user_simulator"
        if source.startswith("agent_msg.validate("):
            return "agent_reply"                    # a message with neither text nor tool calls
        if "/tau2/agent/" in path:
            # The agent's call: the request itself failed (tau2's generate was inside its
            # completion call), or tau2 could not read the reply (e.g. tool-call arguments that
            # are not JSON).
            requested = any(p.endswith("/tau2/utils/llm_utils.py") and "completion(" in s
                            for p, _f, s in frames[i + 1:])
            if not requested:
                return "agent_reply"
            return "agent_context_overflow" if overflow else "model_endpoint"
        if func == "_execute_tool_calls" or "/tau2/environment/" in path \
                or "/tau2/domains/" in path:
            return "tool"
    return "other"


def _infra_error_summary(sims: List[Any]) -> Dict[str, Any]:
    """The summary of infrastructure-error simulations `_DOMAIN_INFRA` keeps: their number, the
    class of the last error of each (tau2's run_with_retry keeps it in info["error_type"]; a
    record without one counts as "unknown"), where each came from (`_infra_error_source`), the
    two together ("<source>/<class>"), and the first error message, redacted and cut to
    _INFRA_ERROR_TEXT_CHARS."""
    types: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    causes: Dict[str, int] = {}
    example = None
    for sim in sims:
        info = getattr(sim, "info", None)
        info = info if isinstance(info, dict) else {}
        name = str(info.get("error_type") or "unknown")
        source = _infra_error_source(sim)
        types[name] = types.get(name, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        causes[f"{source}/{name}"] = causes.get(f"{source}/{name}", 0) + 1
        if example is None and info.get("error"):
            text = " ".join(str(info["error"]).split())
            example = _redact_text(text, list(_SECRETS))[:_INFRA_ERROR_TEXT_CHARS]
    return {"count": len(sims), "error_types": types, "sources": sources, "causes": causes,
            "example_error": example}


def _domain_infra_entry(sims: List[Any]) -> Dict[str, Any]:
    """`_DOMAIN_INFRA`'s entry for a domain's infrastructure-error simulations: the summary of
    all of them, and of the harness part and the agent-turn part apart."""
    agent = [s for s in sims if _infra_error_source(s) in _AGENT_TURN_SOURCES]
    harness = [s for s in sims if _infra_error_source(s) not in _AGENT_TURN_SOURCES]
    return dict(_infra_error_summary(sims), harness=_infra_error_summary(harness),
                agent_turn=_infra_error_summary(agent))


def _run_one_domain(domain, model_name, base_url, concurrency, limit, enable_thinking,
                    eval_name="tau2", save_dir=None, task_ids=None):
    """Run a single tau2 domain via the Python API; return (total, reward_sum, perfect, infra).

    tau2 writes its SimulationResults (per-task messages + reward breakdown) to
    `<save_dir>/<domain>/results.json` (run_domain joins `save_to` onto its data/simulations
    directory; an absolute path replaces that directory). `save_dir` must be unique to this
    run (`run_tau_env` builds it); when it is not given, a fresh temporary directory is used.
    The file must not exist yet: tau2 is never allowed to resume.

    `task_ids` (a sharded or limited run: _shard_task_ids) runs exactly those tasks; otherwise
    every task of the split. `limit` is not passed as tau2's --num-tasks (the head of the file
    order): the selection is task_ids.
    """
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    if not save_dir:
        save_dir = tempfile.mkdtemp(prefix=f"gbench_{_safe_name(eval_name)}_{_safe_name(domain)}_")
    save_to = os.path.join(os.path.abspath(save_dir), domain)
    if os.path.exists(save_to):
        raise RuntimeError(f"{save_to} already exists; gbench never resumes a tau2 run, so it "
                           "refuses to write into another run's simulations")

    # Harness errors (failed Gemini embedding calls) are counted under this domain.
    _CURRENT_DOMAIN["domain"] = domain

    # Credentials stay out of the agent's view and out of the files tau2 writes, and (for
    # banking_knowledge) a failed Gemini embedding call is a harness error rather than a tool
    # error the agent sees (hard error if the guards cannot be installed).
    _install_credential_guards(domain)

    # banking_knowledge (tau3) is a RAG domain with one retrieval config: upstream's alltools,
    # run as alltools-gemini. Other domains (airline/retail/telecom) ignore retrieval_config.
    retrieval_config = None
    retrieval_config_kwargs = None
    if domain == "banking_knowledge":
        retrieval_config, retrieval_config_kwargs = _setup_banking_retrieval()

    config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=f"openai/{model_name}",
        llm_args_agent=_build_agent_args(base_url, enable_thinking, eval_name),
        llm_user=_user_llm(),
        llm_args_user={"temperature": _user_temperature()},
        num_trials=_num_trials(),
        max_steps=_max_steps(),
        max_errors=_max_errors(),
        max_concurrency=max(1, concurrency),
        seed=_seed(),
        log_level="ERROR",
        task_ids=(list(task_ids) if task_ids is not None else None),
        num_tasks=None,
        retrieval_config=retrieval_config,
        retrieval_config_kwargs=retrieval_config_kwargs,
        save_to=save_to,
        verbose_logs=False,
        auto_resume=False,
    )
    try:
        results = run_domain(config)
    finally:
        rewritten = _redact_tree(save_to, list(_SECRETS))
        if rewritten:
            logger.warning("%s: redacted credential values from %d trace file(s) of domain %s",
                           eval_name, len(rewritten), domain)
    _DOMAIN_TRACES[domain] = os.path.join(save_to, "results.json")
    sims = list(results.simulations)
    # A simulation the user simulator ended with a protocol error is a harness error: left out
    # of pass^k and the rewards like an infrastructure error (counted in _DOMAIN_USER_ERRORS).
    user_errors = {id(s) for s in sims if _is_user_error(s) and not _is_infra_error(s)}
    _DOMAIN_USER_ERRORS[domain] = len(user_errors)
    scored_results = (_with_simulations(results, [s for s in sims if id(s) not in user_errors])
                      if user_errors else results)
    # pass^k exactly as tau2 computes it for the leaderboard (compute_metrics: simulations with
    # an infrastructure error excluded, k from 1 up to the trial count).
    from tau2.metrics.agent_metrics import compute_metrics
    if sims:
        _DOMAIN_PASS_HAT_K[domain] = {
            f"pass^{int(k)}": float(v)
            for k, v in sorted(compute_metrics(scored_results).pass_hat_ks.items())}
    reward_sum = 0.0
    perfect = 0
    infra_sims = []
    judge_models: Dict[str, int] = {}
    # Per task: [evaluable trials, successful trials] (tau2's is_successful), what pass^k is
    # computed from; merged across shards by merge_tau_shards.
    outcomes: Dict[str, List[int]] = {}
    for s in sims:
        if _is_infra_error(s):
            infra_sims.append(s)
            continue
        if id(s) in user_errors:
            continue
        r = float(s.reward_info.reward or 0.0)
        reward_sum += r
        if r >= 1.0:
            perfect += 1
        task_id = getattr(s, "task_id", None)
        if task_id is not None:
            row = outcomes.setdefault(str(task_id), [0, 0])
            row[0] += 1
            row[1] += int(abs(r - 1.0) <= 1e-6)
        judge = _verdict_judge_model(s)
        if judge:
            judge_models[judge] = judge_models.get(judge, 0) + 1
    if judge_models:
        _DOMAIN_JUDGE_MODELS[domain] = dict(sorted(judge_models.items()))
    _DOMAIN_TASK_OUTCOMES[domain] = dict(sorted(outcomes.items()))
    if infra_sims:
        # Never scored: counted per domain with their sources and error classes, the harness
        # failures apart from the errors raised on the agent's own turn (`_run_domains` puts them
        # in harness_errors / agent_turn_errors and in the comparability reason).
        _DOMAIN_INFRA[domain] = _domain_infra_entry(infra_sims)
    return len(sims), reward_sum, perfect, len(infra_sims)


#: Age (hours) above which a leftover tau retrieval sandbox is pruned at the start of a tau
#: run. tau2's SandboxManager stages a ~700-file knowledge base into
#: /tmp/agentic_search_<id>/ per simulation and cleans up only on graceful __exit__; a sim
#: killed or timed out (e.g. a context-window overflow) leaks its dir. On a RAM-backed tmpfs
#: these accumulate across runs until the INODE table fills and unrelated evals crash with a
#: misleading "No space left on device" (measured 2026-08-20: 1,141 leaked dirs, ~800K
#: inodes, killed aider_polyglot). gbench runs tau suites sequentially, so at the start of a
#: tau run nothing tau owns any such dir; pruning those older than this threshold is safe and
#: bounds accumulation to a single run. 0 disables. See docs/evals/large-sweep-setup.md.
_TAU_SANDBOX_MAX_AGE_H = float(os.environ.get("GBENCH_TAU_SANDBOX_MAX_AGE_H", "2"))


def _prune_stale_search_sandboxes() -> None:
    """Remove leftover /tmp/agentic_search_* dirs older than the age threshold. Never raises."""
    if _TAU_SANDBOX_MAX_AGE_H <= 0:
        return
    import glob
    import shutil
    import time
    try:
        cutoff = time.time() - _TAU_SANDBOX_MAX_AGE_H * 3600.0
    except Exception:            # time.time patched out in some sandboxes
        return
    base = os.path.join(tempfile.gettempdir(), "agentic_search_*")
    removed = 0
    for path in glob.glob(base):
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("tau: pruned %d stale retrieval sandbox(es) from %s (older than %.1fh); "
                    "see docs/evals/large-sweep-setup.md", removed,
                    tempfile.gettempdir(), _TAU_SANDBOX_MAX_AGE_H)


def _run_config(eval_name: str, domains: List[str], model_name: str, enable_thinking: bool,
                limit: Optional[int], trials: int, shard: Optional[str] = None
                ) -> Dict[str, Any]:
    """Everything that shapes a run's simulations, recorded on the result and hashed into the
    run's directory name."""
    from .base import resolve_temperature
    _apply_legacy_temperature_alias(eval_name)
    temperature, temperature_source = resolve_temperature(eval_name or "tau2",
                                                          _UPSTREAM_AGENT_TEMPERATURE,
                                                          thinking=bool(enable_thinking))
    return {
        "eval": eval_name, "domains": list(domains), "agent_llm": f"openai/{model_name}",
        "thinking": bool(enable_thinking), "num_trials": trials, "eval_limit": limit,
        "shard": shard,
        "max_steps": _max_steps(), "max_errors": _max_errors(), "seed": _seed(),
        "agent_temperature": temperature, "agent_temperature_source": temperature_source,
        "agent_top_p": suite_env("GBENCH_TAU2_TOP_P", "TAU2_TOP_P"),
        "agent_top_k": suite_env("GBENCH_TAU2_TOP_K", "TAU2_TOP_K"),
        "user_llm": _user_llm(), "user_temperature": _user_temperature(), "eval_llm": _eval_llm(),
        "user_llm_models": _role_cascade(_USER_ROLE), "eval_llm_models": _role_cascade(_JUDGE_ROLE),
        "retrieval_config": (_UPSTREAM_RETRIEVAL_CONFIG if "banking_knowledge" in domains
                             else None),
        "retrieval_embedders": (list(_DENSE_EMBEDDERS) if "banking_knowledge" in domains
                                else None),
        "tau2_revision": TAU2_BENCH_REVISION,
    }


def run_tau_env(
    eval_name: str,
    domains: List[str],
    model_name: str,
    base_url: str,
    concurrency: int,
    limit: Optional[int],
    docs_url: str,
    enable_thinking: bool = False,
    shard: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the real tau2 simulator per domain in THIS process and aggregate into a standard result.

    `shard` ("I/N", validated on the host) runs shard I of N of each domain's tasks
    (_shard_task_ids), before the --eval-limit cap.

    This is the half that runs inside the gbench-tau2 image (`run_tau_suite` starts it there);
    its prerequisite check refuses any other interpreter (wrong tau2 revision, modified checkout,
    or packages that differ from tau2-bench's uv.lock).

    `accuracy` is the leaderboard headline: pass^1 (tau2 compute_metrics, infrastructure errors
    excluded) averaged over the domains with equal weight, which for tau2's retail/airline/
    telecom is the tau2-bench track's Overall and for tau3 is banking_knowledge's pass^1.
    `pass_k` carries pass^1..pass^k (k = trials) at the same level, `pass_k_by_domain` per
    domain; `tau2_report.mean_reward` keeps the mean reward pooled over every simulation that
    did not end in an infrastructure error, and `total_questions` / `category_accuracy` count
    those evaluable simulations. Simulations that did end in one are never scored: the ones from
    a harness failure are harness errors, listed per domain in `harness_errors`, and the ones
    raised on the agent's own turn (`_AGENT_TURN_SOURCES`) are listed per domain in
    `agent_turn_errors`; both are named in `leaderboard_comparable_reason` and make the status
    completed_with_errors. Every run writes to a directory of its own and never resumes.
    Hard-errors (infra_required) if prerequisites are missing, or any domain fails or has no
    evaluable simulation.

    The endpoint key is taken from GBENCH_TAU_AGENT_KEY (set by `run_tau_suite`; EMPTY when
    absent), which is removed from the environment here, before anything else runs. The
    returned result has every credential value of the run redacted.
    """
    agent_key = os.environ.pop(_AGENT_KEY_ENV, None) or "EMPTY"
    secrets = _secret_values(os.environ, extra=(agent_key,))
    _SECRETS[:] = secrets
    _AGENT_ROUTE.clear()
    _AGENT_ROUTE.update({"model": f"openai/{model_name}", "api_base": base_url,
                         "api_key": agent_key})
    try:
        return _redact(_run_tau_env(eval_name, domains, model_name, base_url, concurrency, limit,
                                    docs_url, enable_thinking, shard), secrets)
    finally:
        _AGENT_ROUTE.clear()
        _SECRETS.clear()


def _run_tau_env(eval_name, domains, model_name, base_url, concurrency, limit, docs_url,
                 enable_thinking, shard=None) -> Dict[str, Any]:
    _prune_stale_search_sandboxes()
    ok, reason = check_tau_env_prerequisites()
    if not ok:
        raise infra_required(eval_name, reason, docs_url)
    if "banking_knowledge" in domains:
        ok_b, reason_b = check_banking_prerequisites()
        if not ok_b:
            raise infra_required(eval_name, reason_b, docs_url)

    _quiet_tau2_noise()
    _refuse_resume_prompts()
    reset_llm_failures()
    reset_agent_wire()
    reset_harness_errors()
    reset_gemini_answers()
    _DOMAIN_PASS_HAT_K.clear()
    _DOMAIN_TRACES.clear()
    _DOMAIN_TASK_OUTCOMES.clear()
    _DOMAIN_JUDGE_MODELS.clear()
    _DOMAIN_INFRA.clear()
    _DOMAIN_USER_ERRORS.clear()
    _LAST_RETRIEVAL.clear()
    _install_robustness_patches()
    _configure_evaluator_llm(_eval_llm())

    # --eval-limit is a TOTAL budget across the domains being run, never per domain: _run_domains
    # deals it over the domains (sampling.allocate_limit) once it has listed their tasks. With
    # --shard the budget applies within the shard.
    limit = int(limit) if limit and int(limit) > 0 else None
    trials, attempts = _resolve_trials()

    # One directory per run: eval, model, thinking mode, config hash and invocation id. It is
    # created here (never reused), so tau2 cannot find another run's results.json to resume.
    run_config = _run_config(eval_name, domains, model_name, enable_thinking, limit,
                             trials, shard)
    tag, config_hash = _run_tag(eval_name, model_name, enable_thinking, run_config)
    trace_root = _resolve_tau_trace_dir()
    scratch = None if trace_root else tempfile.mkdtemp(prefix="gbench_tau_")
    run_dir = os.path.join(trace_root or scratch, tag)
    try:
        os.makedirs(run_dir, exist_ok=False)
    except FileExistsError as e:
        raise infra_required(eval_name, f"the run directory {run_dir} already exists; gbench "
                             "never resumes a tau2 run", docs_url) from e
    except OSError as e:
        raise infra_required(eval_name, f"cannot create the tau2 run directory {run_dir} "
                             f"({type(e).__name__}: {e}); set GBENCH_TAU2_SAVE_TRACES to a "
                             "writable directory", docs_url) from e
    try:
        result = _run_domains(eval_name, domains, model_name, base_url, concurrency,
                              limit, docs_url, enable_thinking, trials, run_dir,
                              run_config, config_hash, keep_traces=trace_root is not None,
                              shard=shard)
        if attempts:
            result["attempts"] = attempts
        return result
    finally:
        _CURRENT_DOMAIN["domain"] = None
        _redact_tree(run_dir, list(_SECRETS))
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _run_domains(eval_name, domains, model_name, base_url, concurrency, limit,
                 docs_url, enable_thinking, trials, run_dir, run_config, config_hash,
                 keep_traces, shard=None) -> Dict[str, Any]:
    """Run `domains` and build the result. `limit` (--eval-limit) is the number of tasks for the
    whole run, dealt over the domains (domain_limit_shares); a domain whose share is 0 is not
    run."""
    category_accuracy: Dict[str, Any] = {}
    mean_reward_by_domain: Dict[str, float] = {}
    total = 0
    reward_sum_all = 0.0
    perfect_all = 0
    infra_all = 0
    # Simulations the user simulator ended with a protocol error (USER_ERROR), per domain:
    # harness errors, left out of pass^k and the rewards.
    user_all = 0
    user_by_domain: Dict[str, int] = {}
    # Per domain: the simulations that ended in an infrastructure error (`_DOMAIN_INFRA`'s shape).
    infra_by_domain: Dict[str, Dict[str, Any]] = {}

    # tau2's own per-task console is muted by default (it spams a file-redirected log), so
    # progress is shown via a tqdm bar driven by StatusMonitor - one per domain, exactly like
    # every other eval's `Eval [X]` bar. GBENCH_TAU2_VERBOSE=1 restores tau2's live per-task panels.
    global _TAU_PROGRESS_DESC
    limited = bool(limit and limit > 0)
    # --shard and --eval-limit: each domain's tasks are listed in tau2's order (shard I of N
    # first: _shard_task_ids), --eval-limit's tasks are dealt over the domains
    # (domain_limit_shares) and each domain's share is a seeded draw from its list, passed to
    # tau2 as task_ids; a domain with no task in this shard, or a share of 0, is skipped and listed
    # (empty_shard_domains / limit_skipped_domains).
    domain_task_counts: Dict[str, int] = {}
    selected_task_ids: Dict[str, List[str]] = {}
    listed: Dict[str, List[str]] = {}
    if shard or limited:
        for domain in domains:
            try:
                listed[domain], domain_task_counts[domain] = _shard_task_ids(
                    domain, shard, None, seed=eval_name)
            except Exception as e:
                wanted = " and ".join(x for x in (f"--shard {shard}" if shard else "",
                                                  "--eval-limit" if limited else "") if x)
                raise infra_required(
                    eval_name, f"could not list the tasks of domain '{domain}' for {wanted} "
                    f"({type(e).__name__}: {e})", docs_url) from e
            if not listed[domain] and not shard:
                raise infra_required(eval_name, f"tau2 lists no task for domain '{domain}'",
                                     docs_url)
    shares = domain_limit_shares(limit, domains, {d: len(v) for d, v in listed.items()},
                                 eval_name)
    limit_skipped = [d for d in domains if limited and shares.get(d) == 0 and listed.get(d)]
    planned = _plan_counts(domains, listed, shares, shard or limited)
    plan = _limit_plan(eval_name, limit, planned, trials, domains)
    scope = (f"{limit} task(s) over the domains" if limited else "all tasks")
    logger.info("%s: running %d domain(s) %s via tau2 simulator (%s, concurrency=%d, trials=%d, "
                "run dir %s); each task is a multi-turn agent<->user-simulator conversation "
                "scored by an LLM judge, so a domain runs for several minutes - a tqdm progress "
                "bar follows (set GBENCH_TAU2_VERBOSE=1 for per-task detail)",
                eval_name, len(domains), domains, scope, max(1, concurrency), trials, run_dir)

    empty_domains: List[str] = []
    for i, domain in enumerate(domains, 1):
        _TAU_PROGRESS_DESC = f"{eval_name} {domain}" if len(domains) > 1 else eval_name
        task_ids = None
        if shard or limited:
            if domain in limit_skipped:
                logger.info("%s: --eval-limit %s gives domain '%s' no task; not run",
                            eval_name, limit, domain)
                continue
            task_ids = listed[domain]
            if limited:
                from .sampling import stratified_sample
                task_ids = stratified_sample(task_ids, int(shares[domain]), seed=eval_name)
            selected_task_ids[domain] = list(task_ids)
            if not task_ids:
                logger.info("%s: shard %s has no task of domain '%s' (%d tasks); skipped",
                            eval_name, shard, domain, domain_task_counts[domain])
                empty_domains.append(domain)
                continue
        logger.info("%s: [%d/%d] starting domain '%s' ...", eval_name, i, len(domains), domain)
        # A failed or empty domain stops the run: averaging over the domains that did finish
        # would report a partial number as the suite's score.
        try:
            d_total, d_reward, d_perfect, d_infra = _run_one_domain(
                domain, model_name, base_url, concurrency, shares.get(domain), enable_thinking,
                eval_name, save_dir=run_dir, task_ids=task_ids)
        except Exception as e:
            logger.error("%s env: domain %s failed: %s", eval_name, domain, e, exc_info=True)
            raise infra_required(
                eval_name, f"tau2 simulator failed on domain '{domain}' ({type(e).__name__}: {e}); "
                "check the tau2 install, GEMINI_API_KEY, and the model endpoint", docs_url) from e
        if d_total == 0:
            raise infra_required(
                eval_name, f"tau2 simulator produced no simulations for domain '{domain}'",
                docs_url)
        if "pass^1" not in _DOMAIN_PASS_HAT_K.get(domain, {}):
            d_embed = embedding_failures().get(domain, 0)
            embed_note = (f"; {d_embed} Gemini embedding call(s) failed after their retries"
                          if d_embed else "")
            detail = _DOMAIN_INFRA.get(domain) or {}
            error_types = detail.get("error_types") or {}
            types_note = (f": {_error_types_text(error_types)}"
                          if set(error_types) - {"unknown"} else "")
            agent_part = _infra_parts(detail)["agent_turn"]
            agent_note = (f"; {agent_part['count']} of them raised on the agent's own turn "
                          f"({_error_types_text(agent_part.get('causes'))}), not harness "
                          "failures" if agent_part.get("count") else "")
            d_user = _DOMAIN_USER_ERRORS.get(domain, 0)
            ended = ("an infrastructure error or a user-simulator protocol error (USER_ERROR, "
                     f"{d_user})" if d_user else "an infrastructure error")
            raise infra_required(
                eval_name, f"tau2 simulator produced no evaluable simulation for domain "
                f"'{domain}' (all {d_total} ended in {ended}{types_note}{agent_note}"
                f"{embed_note})", docs_url)
        d_user = _DOMAIN_USER_ERRORS.get(domain, 0)
        if d_user:
            user_by_domain[domain] = d_user
            user_all += d_user
            logger.warning("%s: domain '%s' had %d simulation(s) the Gemini user simulator ended "
                           "with a protocol error (tau2 USER_ERROR): harness errors, left out of "
                           "pass^k", eval_name, domain, d_user)
        total += d_total
        reward_sum_all += d_reward
        perfect_all += d_perfect
        infra_all += d_infra
        if d_infra:
            detail = _DOMAIN_INFRA.get(domain) or {}
            infra_by_domain[domain] = dict(detail, count=d_infra) if detail else {
                "count": d_infra, "error_types": {}, "example_error": None}
            parts = _infra_parts(infra_by_domain[domain])
            logger.warning("%s: domain '%s' had %d simulation(s) that ended in an infrastructure "
                           "error, which tau2's metrics leave out: %d from harness failures, "
                           "not the agent's outcome, and %d raised on the agent's own turn (%s)",
                           eval_name, domain, d_infra, parts["harness"]["count"],
                           parts["agent_turn"]["count"],
                           _error_types_text(infra_by_domain[domain].get("causes")
                                             or infra_by_domain[domain].get("error_types"))
                           or "error class not recorded")
        d_pass1 = 100.0 * _DOMAIN_PASS_HAT_K[domain]["pass^1"]
        # Mean reward over the simulations that did not end in an infrastructure error, as
        # tau2's own compute_metrics avg_reward (an infrastructure error is never scored), nor in
        # a user-simulator protocol error (a harness error).
        d_evaluable = d_total - d_infra - d_user
        d_mean = d_reward / d_evaluable if d_evaluable > 0 else 0.0
        mean_reward_by_domain[domain] = round(d_mean, 4)
        # Counts over the evaluable simulations, the ones pass^1 covers; an infrastructure error
        # is in `simulations` and `infra_errors` only, so it never reads as a failed simulation.
        category_accuracy[domain] = {
            "correct": d_perfect,
            "total": d_evaluable,
            "accuracy": round(d_pass1, 2),       # the domain's pass^1 (%)
            "simulations": d_total,
            "infra_errors": d_infra,
            **({"user_simulator_protocol_errors": d_user} if d_user else {}),
        }
        logger.info("%s: [%d/%d] domain '%s' done - pass^1 %.2f%%, mean reward %.3f "
                    "(%d/%d evaluable simulations perfect, %d infrastructure errors)", eval_name,
                    i, len(domains), domain, d_pass1, d_mean, d_perfect, d_evaluable, d_infra)
        d_embed = embedding_failures().get(domain, 0)
        if d_embed:
            logger.warning("%s: domain '%s' had %d Gemini embedding call(s) that failed after "
                           "their retries; each is a harness error (the simulation attempt was "
                           "stopped and re-run), not the agent's outcome", eval_name, domain,
                           d_embed)

    if shard and len(empty_domains) + len(limit_skipped) == len(domains):
        out = _empty_shard_result(eval_name, model_name, domains, shard, domain_task_counts,
                                  trials, run_config, config_hash)
        out["eval_limit_plan"] = plan
        return out
    # The domains that ran (a shard may hold no task of a domain; --eval-limit may give a domain
    # no task).
    all_domains = list(domains)
    domains = [d for d in domains if d not in empty_domains and d not in limit_skipped]
    gaps = _comparability_gaps(domains, trials, limit, infra_all, infra_by_domain)
    if limit_skipped:
        gaps.append(f"--eval-limit {limit} for the run gives {', '.join(limit_skipped)} no task: "
                    f"the headline averages {', '.join(domains)} only")
    if user_all:
        gaps.append(f"{user_all} simulation(s) the Gemini user simulator ended with a protocol "
                    f"error (tau2 USER_ERROR: a message with both text and tool calls, or an "
                    f"empty one; " + ", ".join(f"{d} x{n}" for d, n in sorted(user_by_domain.items()))
                    + "): harness errors, not the agent's outcome, left out of pass^k")
    if empty_domains:
        gaps.append(f"shard {shard} holds no task of {', '.join(empty_domains)}: the headline "
                    f"averages {', '.join(domains)} only (every shard merged restores the "
                    "track's domains)")
    harness_errors = _harness_error_records(domains, infra_by_domain)
    harness_errors.extend(
        {"error": ("simulations the Gemini user simulator ended with a protocol error (tau2 "
                   "USER_ERROR, orchestrator.py _check_communication_error), which tau2's "
                   "evaluator scores 0; left out of pass^k"),
         "kind": "user_simulator_protocol", "domain": d, "simulations": n}
        for d, n in sorted(user_by_domain.items()))
    agent_turn_errors = _agent_turn_error_records(domains, infra_by_domain)
    embed_by_domain = embedding_failures()
    # The leaderboard's numbers: tau2's pass^k per domain in % (infrastructure errors excluded,
    # as compute_metrics does), and their unweighted mean over the domains for every k that all
    # domains reached, which is the tau2-bench track's Overall for retail/airline/telecom.
    pass_k_by_domain = {d: {k: round(100.0 * v, 2) for k, v in _DOMAIN_PASS_HAT_K[d].items()}
                        for d in domains}
    ks = set.intersection(*(set(_DOMAIN_PASS_HAT_K[d]) for d in domains))
    pass_k = {k: round(sum(100.0 * _DOMAIN_PASS_HAT_K[d][k] for d in domains) / len(domains), 2)
              for k in sorted(ks, key=lambda s: int(s.split("^", 1)[1]))}
    pass1_by_domain = {d: pass_k_by_domain[d]["pass^1"] for d in domains}
    answers = gemini_answers()
    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "mode": "full_environment",
        # The evaluable simulations (the ones pass^1 covers) and the perfect ones among them;
        # tau2_report.total_simulations and infra_errors count the rest.
        "total_questions": total - infra_all - user_all,
        "correct_answers": perfect_all,
        # Headline: pass^1 of the leaderboard track (equal-weight mean over the domains).
        "accuracy": pass_k["pass^1"],
        "metric": ("pass^1 (tau2 compute_metrics, infrastructure errors excluded) averaged over "
                   "the domains with equal weight; " + _leaderboard_track(eval_name, domains)),
        "leaderboard_track": _leaderboard_track(eval_name, domains),
        "pass_k": pass_k,
        "pass_k_by_domain": pass_k_by_domain,
        "pass1_by_domain": pass1_by_domain,
        "pass1_domain_mean": pass_k["pass^1"],
        "category_accuracy": category_accuracy,
        # tau2 grades with its own harness (DB-state check + nl-assertion judge). The ways the
        # setup differs from a leaderboard submission that gbench detects (trials, task subset,
        # user simulator, NL-assertion judge, agent sampling, tau3 retrieval embeddings, failed
        # Gemini embedding calls, infra errors) are named here.
        "leaderboard_comparable": not gaps,
        "leaderboard_comparable_reason": "; ".join(gaps) if gaps else None,
        **({"shard": shard, "empty_shard_domains": empty_domains} if shard else {}),
        **({"limit_skipped_domains": limit_skipped} if limit_skipped else {}),
        "eval_limit_plan": plan,
        # a sharded or limited run: each domain's task count and the task ids it ran
        **({"domain_task_counts": domain_task_counts, "selected_task_ids": selected_task_ids}
           if selected_task_ids else {}),
        "domains_requested": all_domains,
        "tau2_report": {
            # Mean reward pooled over every simulation that did not end in an infrastructure
            # error (tau2's avg_reward leaves them out too), so each domain weighs by its task
            # count; not a leaderboard number.
            "mean_reward": (round(reward_sum_all / (total - infra_all - user_all), 4)
                            if total - infra_all - user_all > 0 else 0.0),
            "mean_reward_by_domain": mean_reward_by_domain,
            "pass_hat_k": {d: {k: round(v, 4) for k, v in _DOMAIN_PASS_HAT_K[d].items()}
                           for d in domains},
            "perfect_tasks": perfect_all,
            "total_simulations": total,
            "evaluable_simulations": total - infra_all - user_all,
            "infra_errors": infra_all,
            # user-simulator protocol errors (USER_ERROR), per domain: harness errors
            "user_simulator_protocol_errors": user_all,
            "user_simulator_protocol_errors_by_domain": dict(sorted(user_by_domain.items())),
            "num_trials": trials,
            # Per domain and task: [evaluable trials, successful trials] (tau2's is_successful),
            # which pass^k is computed from (merge_tau_shards recomputes it over every shard).
            "task_outcomes_by_domain": {d: dict(_DOMAIN_TASK_OUTCOMES.get(d) or {})
                                        for d in domains},
            "agent_llm": f"openai/{model_name}",
            "agent_sampling": {"temperature": _LAST_AGENT_TEMPERATURE.get("value"),
                               "top_p": _LAST_AGENT_TEMPERATURE.get("top_p"),
                               "top_k": _LAST_AGENT_TEMPERATURE.get("top_k")},
            "user_llm": _user_llm(),
            "eval_llm": _eval_llm(),
            # The models each role tries, in order, and where they come from (the role's knob,
            # or gbench's Gemini cascade: GBENCH_JUDGE_MODELS / GBENCH_JUDGE_MODEL / default).
            "user_llm_models": _role_cascade(_USER_ROLE),
            "eval_llm_models": _role_cascade(_JUDGE_ROLE),
            "user_llm_source": _role_source(_USER_ROLE),
            "eval_llm_source": _role_source(_JUDGE_ROLE),
            # Which Gemini models answered: replies per role, per domain and role, and the
            # model of each simulation's NL-assertion verdict per domain (each simulation's
            # reward_info.info.nl.judge_model in results.json).
            "gemini_models_answered_by_role": answers["by_role"],
            "gemini_models_answered_by_domain": answers["by_domain"],
            "nl_judge_models_by_domain": {d: dict(_DOMAIN_JUDGE_MODELS.get(d) or {})
                                          for d in domains},
            "gemini_models_not_served": dict(_NOT_SERVED),
            # What every gemini/ call sends besides the conversation: upstream's sampling
            # arguments and nothing else, the same for every model of the cascade.
            "gemini_call_parameters": _gemini_call_parameters(),
            # Retries of empty/malformed replies and fence stripping, applied only to gemini/
            # user-simulator and judge calls; the agent's requests go out once, as upstream.
            "substitute_retries": {k: v for k, v in _RETRY_PATCHES.items()},
            # banking_knowledge: the stock config the run follows (alltools, the only one), the
            # variant tau2 ran (the Gemini-embedding alltools-gemini) and its dense embedder.
            "retrieval_config": _LAST_RETRIEVAL.get("retrieval_config"),
            "retrieval_variant": _LAST_RETRIEVAL.get("retrieval_variant"),
            "embed_model": _LAST_RETRIEVAL.get("embed_model"),
            "embedders": _LAST_RETRIEVAL.get("embedders"),
            "retrieval_embedder": _LAST_RETRIEVAL.get("embedder"),
            # Gemini embedding calls that failed after their retries, by domain (every domain
            # that embeds is listed, with 0 when none failed); harness errors, see
            # `harness_errors`.
            "embedding_failures_by_domain": (
                {d: embed_by_domain.get(d, 0) for d in domains if d in KNOWLEDGE_DOMAINS}
                or None),
            # Credential values are replaced by [REDACTED] in results.json, this result and the
            # streamed log; the shell tool runs without any credential in its environment.
            "credentials_redacted": True,
            "tau2_version": _tau2_version(),
            "tau2_revision": TAU2_BENCH_REVISION,
            "run_config": run_config,
            "config_hash": config_hash,
            "run_dir": run_dir if keep_traces else None,
            "traces": dict(_DOMAIN_TRACES) if keep_traces else None,
        },
        # Transport failures seen during the run, by class: every failed LLM call attempt
        # (agent, user simulator, judge), including the ones a retry or the next cascade model
        # recovered from, so this is not a count of lost simulations. A simulation whose every
        # attempt failed is an infrastructure error (`infra_errors`, `harness_errors`,
        # `agent_turn_errors`).
        "llm_failures": llm_failures(),
        "llm_failures_total": sum(llm_failures().values()),
        # Replies per Gemini model over every role of the run (the user simulator, the
        # NL-assertion judge), and the cascade they were asked from (base.judge_cascade(), as
        # the host handed it over in GBENCH_JUDGE_MODELS); per role in tau2_report.
        "gemini_models_answered": answers["total"],
        "gemini_cascade": _cascade_from(os.environ)[0],
        # Harness errors, never the agent's outcome, per domain: Gemini embedding calls that
        # failed after their retries (the simulation attempt each one hit was stopped and re-run
        # by tau2), and simulations that ended in an infrastructure error from a harness failure
        # (an outage of the Gemini user simulator or judge, of the model endpoint, failed
        # embedding calls, ...), with their sources and error classes. Any of them makes the
        # status completed_with_errors, and each is named in leaderboard_comparable_reason.
        "harness_errors": harness_errors,
        # Simulations that ended in an infrastructure error raised on the agent's own turn (an
        # agent reply tau2 rejects, an agent request over the endpoint's context window), per
        # domain: not harness failures, left out of pass^k by tau2 as upstream does. They too
        # make the status completed_with_errors (pass^k covers fewer than every task x trial)
        # and are named in leaderboard_comparable_reason.
        "agent_turn_errors": agent_turn_errors,
        # The model under test's sampling knob, so a run can be audited from the artifact
        # like every other suite. The user simulator and judge stay at 0.0 and are separate.
        # Default 0.0 = upstream's DEFAULT_LLM_TEMPERATURE_AGENT; 1.0 with --thinking
        # (temperature_source "thinking").
        "temperature": _LAST_AGENT_TEMPERATURE.get("value"),
        "temperature_source": _LAST_AGENT_TEMPERATURE.get("source"),
        # The run's thinking switch and whether every agent request carried it
        # (agent_request_shape: the agent's requests counted at tau2's completion chokepoint).
        "thinking": bool(enable_thinking),
        **agent_wire_record(bool(enable_thinking)),
        # the agent's request timeout and retries (no 600 s LiteLLM default)
        "model_client_timeout": agent_client_timeout(),
        # (a user-simulator protocol error has a harness_errors record)
        "status": "completed_with_errors" if (harness_errors or infra_all) else "success",
    }


def _empty_shard_result(eval_name: str, model_name: str, domains: List[str], shard: str,
                        domain_task_counts: Dict[str, int], trials: int,
                        run_config: Dict[str, Any], config_hash: str) -> Dict[str, Any]:
    """The result of a shard that holds no task of any domain: nothing ran on this node, so
    there is no score (accuracy None with the reason), never an error and never 0%."""
    reason = (f"empty shard {shard}: no items on this node "
              f"({sum(domain_task_counts.values())} tasks in the full set)")
    return {
        "benchmark_type": "eval", "eval_name": eval_name, "model_name": model_name,
        "mode": "full_environment", "total_questions": 0, "correct_answers": 0,
        "accuracy": None, "accuracy_withheld_reason": reason, "status": "success",
        "status_reason": reason, "empty_shard": True, "shard": shard,
        "empty_shard_domains": list(domains), "domain_task_counts": dict(domain_task_counts),
        "domains_requested": list(domains),
        "full_set_items": sum(domain_task_counts.values()),
        "leaderboard_track": _leaderboard_track(eval_name, domains),
        "leaderboard_comparable": False, "leaderboard_comparable_reason": reason,
        "category_accuracy": {},
        "tau2_report": {"num_trials": trials, "run_config": run_config,
                        "config_hash": config_hash, "task_outcomes_by_domain": {},
                        "tau2_revision": TAU2_BENCH_REVISION},
    }


def _pass_hat_k(outcomes: Dict[str, List[int]]) -> Dict[str, float]:
    """tau2's pass^k of one domain (metrics/agent_metrics.py compute_metrics) from its per-task
    [evaluable trials, successes]: for k = 1 up to the fewest evaluable trials of any task, the
    mean over the tasks of comb(successes, k) / comb(trials, k)."""
    import math
    rows = [(int(n), int(c)) for n, c in outcomes.values() if int(n) > 0]
    if not rows:
        return {}
    max_k = min(n for n, _c in rows)
    return {f"pass^{k}": sum(math.comb(c, k) / math.comb(n, k) for n, c in rows) / len(rows)
            for k in range(1, max_k + 1)}


def merge_tau_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one tau2 / tau3 run (gbench --merge): each domain's
    per-task outcomes (tau2_report.task_outcomes_by_domain) are united over the shards and its
    pass^k recomputed as tau2 computes it (_pass_hat_k); the headline is pass^1 averaged over the
    domains with equal weight, as run_tau_env reports it; counts and harness errors are summed;
    the shard comparability reasons are dropped (every other reason is kept)."""
    from .base import SHARD_REASON
    by_domain: Dict[str, Dict[str, List[int]]] = {}
    requested: List[str] = []
    for r in results:
        for d in r.get("domains_requested") or []:
            if d not in requested:
                requested.append(d)
        for d, rows in ((r.get("tau2_report") or {}).get("task_outcomes_by_domain")
                        or {}).items():
            by_domain.setdefault(d, {}).update(rows or {})
    domains = [d for d in (requested or sorted(by_domain)) if by_domain.get(d)]
    # An empty shard (no task on its node) sets no field for the union (as merge.generic_merge):
    # 2026-10-13 review, a merge whose shard 1 was empty said no agent request was capped.
    live = [r for r in results if not r.get("empty_shard")] or list(results)
    merged = dict(live[0])
    passes = {d: _pass_hat_k(by_domain[d]) for d in domains}
    ks = set.intersection(*(set(passes[d]) for d in domains)) if domains else set()
    pass_k = {k: round(sum(100.0 * passes[d][k] for d in domains) / len(domains), 2)
              for k in sorted(ks, key=lambda x: int(x.split("^", 1)[1]))}
    merged["pass_k"] = pass_k
    merged["pass_k_by_domain"] = {d: {k: round(100.0 * v, 2) for k, v in passes[d].items()}
                                  for d in domains}
    merged["pass1_by_domain"] = {d: merged["pass_k_by_domain"][d].get("pass^1") for d in domains}
    merged["accuracy"] = pass_k.get("pass^1")
    merged["pass1_domain_mean"] = merged["accuracy"]
    merged["leaderboard_track"] = _leaderboard_track(str(merged.get("eval_name") or "tau2"),
                                                     domains)
    merged["total_questions"] = sum(sum(int(n) for n, _c in by_domain[d].values())
                                    for d in domains)
    merged["correct_answers"] = sum(sum(int(c) for _n, c in by_domain[d].values())
                                    for d in domains)
    merged["category_accuracy"] = {
        d: {"correct": sum(int(c) for _n, c in by_domain[d].values()),
            "total": sum(int(n) for n, _c in by_domain[d].values()),
            "accuracy": merged["pass_k_by_domain"][d].get("pass^1")} for d in domains}
    report = dict(merged.get("tau2_report") or {})
    report["task_outcomes_by_domain"] = by_domain
    report["merged_pass_hat_k"] = {d: {k: round(v, 4) for k, v in passes[d].items()}
                                   for d in domains}
    merged["tau2_report"] = report
    for field in ("harness_errors", "agent_turn_errors"):
        rows = [dict(rec, shard=r.get("shard")) for r in results
                for rec in (r.get(field) or []) if isinstance(rec, dict)]
        merged[field] = rows
    answered: Dict[str, int] = {}
    for r in results:
        for m, n in (r.get("gemini_models_answered") or {}).items():
            answered[m] = answered.get(m, 0) + int(n)
    merged["gemini_models_answered"] = dict(sorted(answered.items()))
    # every shard's selected task ids, per domain (a limited or sharded run records them)
    selected: Dict[str, set] = {}
    for r in results:
        for d, ids in (r.get("selected_task_ids") or {}).items():
            selected.setdefault(str(d), set()).update(str(i) for i in ids or [])
    if selected:
        merged["selected_task_ids"] = {d: sorted(ids) for d, ids in sorted(selected.items())}
    else:
        merged.pop("selected_task_ids", None)
    users: Dict[str, int] = {}
    for r in results:
        for d, n in ((r.get("tau2_report") or {}).get("user_simulator_protocol_errors_by_domain")
                     or {}).items():
            users[str(d)] = users.get(str(d), 0) + int(n or 0)
    report["user_simulator_protocol_errors_by_domain"] = dict(sorted(users.items()))
    report["user_simulator_protocol_errors"] = sum(users.values())
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if (reason and not reason.startswith(shard_prefix) and not reason.startswith("shard ")
                    and not reason.startswith("empty shard ") and reason not in kept):
                kept.append(reason)
    missing = [d for d in requested if d not in domains]
    if missing:
        kept.append(f"no shard ran a task of {', '.join(missing)}")
    # the agent requests gbench's endpoint relay capped, summed over the shards (each shard's
    # reason names its own count)
    fill = sum_relay_fills([((r.get("tau2_report") or {}).get("endpoint_relay") or {})
                            .get("max_tokens_from_run") for r in results])
    kept = merge_relay_fill_reasons(merged, kept, fill)
    if fill:
        if isinstance(merged.get("official_sampling"), dict):   # not the shard's own dict
            merged["official_sampling"] = dict(merged["official_sampling"])
        record_agent_max_tokens(merged, {"max_tokens_from_run": fill}, None)
        # the relay record's fill is the shards' sum too (its other counters stay one shard's)
        if isinstance(report.get("endpoint_relay"), dict):
            report["endpoint_relay"] = dict(report["endpoint_relay"], max_tokens_from_run=dict(
                fill, summed_over_shards=len(results)))
    merged["leaderboard_comparable"] = not kept
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    merged["status"] = ("completed_with_errors"
                        if any(r.get("status") == "completed_with_errors" for r in results)
                        else "success")
    for k in ("accuracy_withheld_reason", "status_reason", "empty_shard_domains",
              "domain_task_counts"):
        merged.pop(k, None)
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = "no shard produced an evaluable simulation"
    return merged


# --------------------------------------------------------------------------------------------- #
# Host side: run the suite in the local gbench-tau2 image
# --------------------------------------------------------------------------------------------- #
#: Where the image sees this package directory (read-only), the job/result exchange directory, the
#: trace root and the persistent cache (tau2 keeps banking_knowledge document embeddings under
#: `data/.embeddings_cache` relative to its working directory, as upstream does in its checkout).
_IMAGE_SUITES_DIR = "/opt/gbench/eval_suites"
_IMAGE_IO_DIR = "/gbench_io"
_IMAGE_TRACE_DIR = "/gbench_traces"
_IMAGE_CACHE_DIR = "/gbench_cache"
#: The package name this directory is imported under in the image. It is created empty, so the
#: suites' registry (eval_suites/__init__.py) and the rest of gbench are never imported there:
#: only tau_common and the stdlib-only helpers it imports (base, swebench_common, gemini_relay,
#: sampling, swe_thread_cap).
_IMAGE_PACKAGE = "gbench_tau_suites"

#: Environment forwarded into the image: every tau knob and the run-shaping variables below.
#: No credential enters it: the launcher adds gbench's Gemini relay placeholder and address and
#: the run's Gemini cascade (GBENCH_JUDGE_MODELS), `_bridge_services`. Host paths are not
#: forwarded (the image has its own checkout and data), and the trace location is set by the
#: launcher.
_FORWARD_PREFIXES = ("GBENCH_TAU2_", "GBENCH_TAU3_", "TAU2_")
_FORWARD_NAMES = ("GBENCH_TAU_SANDBOX_MAX_AGE_H", "LITELLM_LOG",
                  "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
#: Knobs that no longer change anything (setting one logs a warning, `run_tau_suite`): the former
#: embedder knobs (the embedder is gemini-embedding-001, fixed), the former OpenAI key and
#: gateway (no OpenAI service is called) and the former retrieval-config knob (banking_knowledge
#: always runs upstream's alltools with the Gemini embedder).
_REMOVED_EMBED_KNOBS = ("GBENCH_TAU2_EMBED_API_KEY", "TAU2_EMBED_API_KEY",
                        "GBENCH_TAU2_EMBED_MODEL", "TAU2_EMBED_MODEL",
                        "GBENCH_TAU2_EMBED_BASE_URL", "TAU2_EMBED_BASE_URL",
                        "GBENCH_TAU2_EMBED_BATCH", "TAU2_EMBED_BATCH")
_REMOVED_OPENAI_KNOBS = ("GBENCH_TAU2_OPENAI_API_KEY", "GBENCH_TAU2_OPENAI_BASE_URL")
_REMOVED_RETRIEVAL_KNOBS = ("GBENCH_TAU2_RETRIEVAL_CONFIG", "TAU2_RETRIEVAL_CONFIG")
_REMOVED_KNOBS = _REMOVED_EMBED_KNOBS + _REMOVED_OPENAI_KNOBS + _REMOVED_RETRIEVAL_KNOBS
#: Knobs never passed by name: host paths, launcher settings and the removed knobs.
_NOT_FORWARDED = ("GBENCH_TAU2_SAVE_TRACES", "TAU2_SAVE_TRACES", "GBENCH_TAU2_BENCH_SRC",
                  "TAU2_BENCH_SRC", "TAU2_DATA_DIR", "GBENCH_TAU2_IMAGE", "GBENCH_TAU2_ENV_RUN",
                  "TAU2_ENV_RUN") + _REMOVED_KNOBS

#: Host variables that never enter the image under their own name: the endpoint's key and URL
#: (the key travels as GBENCH_TAU_AGENT_KEY when it is no real key, else it stays with gbench's
#: endpoint relay), the Gemini credentials and their endpoint (gbench's Gemini relay adds the
#: key), and the variables the launcher sets itself (`_bridge_services`).
_HOST_ONLY = ("VLLM_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
              "GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
              "GBENCH_GEMINI_RELAY_URL", "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL",
              "TAU_JUDGE_CASCADE_SOURCE")


def _real_endpoint_key(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The model endpoint's key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY, the
    key every pillar sends to the endpoint), or None when it has none (unset or a placeholder
    such as EMPTY). A key never enters the image: gbench's endpoint relay adds it
    (`_bridge_services`)."""
    from .base import endpoint_api_key
    return endpoint_api_key(os.environ if environ is None else environ)


#: bubblewrap, which sandbox-runtime uses for banking_knowledge's shell tool, creates user, mount,
#: network and PID namespaces and mounts its own /proc. Docker's default seccomp and AppArmor
#: profiles and its masked /proc paths prevent that, so a banking_knowledge run lifts those three
#: (the container still drops every capability and sets no-new-privileges). Retail, airline and
#: telecom have no shell tool and run with Docker's defaults.
_SHELL_SANDBOX_DOCKER_OPTS = ("--security-opt", "seccomp=unconfined",
                              "--security-opt", "apparmor=unconfined",
                              "--security-opt", "systempaths=unconfined")


# --------------------------------------------------------------------------------------------- #
# Harness containers that reach the model endpoint: the docker bridge network
# --------------------------------------------------------------------------------------------- #
#: The docker network of every harness container that reaches the model endpoint and runs
#: model-controlled code or tool calls: tau2 / tau3 (this module), the BFCL multi-turn part and
#: `bfcl generate` (bfcl.py, bfcl_v4_agentic.py), livebench and aider_polyglot. The host's
#: DOCKER-USER rule keeps the cloud metadata server out of reach of a bridge network, never of
#: the host network (container_guard), so none of them uses the host network. The helpers below
#: are shared by those suites.
ENDPOINT_NETWORK = "bridge"
#: The name those containers reach this host by: Docker's `host-gateway`, the address of this
#: host on the default bridge (upstream aider's benchmark/docker.sh uses the same mapping).
HOST_ALIAS = "host.docker.internal"
#: The `docker run` arguments of those containers.
ENDPOINT_NETWORK_ARGS = ("--network", ENDPOINT_NETWORK, "--add-host", f"{HOST_ALIAS}:host-gateway")
_LOOPBACK_URL = re.compile(r"^([a-z][a-z0-9+.-]*://(?:[^/?#@]*@)?)"
                           r"(localhost|127(?:\.\d{1,3}){3}|0\.0\.0\.0|\[::1\])(?=[:/?#]|$)", re.I)
_PROXY_URL_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_NO_PROXY_VARS = ("NO_PROXY", "no_proxy")
#: Seconds the endpoint probe waits for its TCP connection, and for `docker run` as a whole.
_ENDPOINT_PROBE_CONNECT_S = 10
_ENDPOINT_PROBE_DOCKER_S = 120
_ENDPOINT_PROBE_MARK = "GBENCH_ENDPOINT_PROBE="
_ENDPOINT_REACHABLE: Dict[Tuple[str, int], bool] = {}
_ENDPOINT_LOCK = threading.Lock()


def container_url(url: Optional[str]) -> str:
    """`url` as a container on ENDPOINT_NETWORK reaches it: a loopback host (localhost,
    127.0.0.0/8, 0.0.0.0, [::1]) becomes HOST_ALIAS, the address of this host there; any other URL
    is returned unchanged."""
    return _LOOPBACK_URL.sub(lambda m: m.group(1) + HOST_ALIAS, url or "", count=1)


def _without_userinfo(url: str) -> str:
    """`url` without a user name or password in its authority (the rest unchanged)."""
    from urllib.parse import urlsplit, urlunsplit
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def _url_authority(url: str) -> str:
    """The authority (netloc) of `url`, "" when it cannot be parsed."""
    from urllib.parse import urlsplit
    try:
        return urlsplit(url.strip()).netloc
    except ValueError:
        return ""


def proxy_credentials(environ: Dict[str, str]) -> List[str]:
    """The proxy variables of `environ` whose URL carries a user name or password (their names,
    never the values): credentials that stay on the host."""
    return sorted(name for name in _PROXY_URL_VARS
                  if "@" in _url_authority(environ.get(name) or ""))


def container_proxy_env(environ: Dict[str, str]) -> Dict[str, str]:
    """The proxy variables of `environ` as a container on ENDPOINT_NETWORK needs them: a proxy on
    a loopback address becomes HOST_ALIAS, a user name or password in a proxy URL is left out (a
    credential of the host never enters the container; its requests go to gbench's relays on
    the gateway, which bypass the proxy; tau2's agent requests always go through gbench's
    endpoint relay, `_bridge_services`), and when a proxy is set HOST_ALIAS is added to
    NO_PROXY and no_proxy, so a request to an endpoint on this host goes to it directly. Returns
    only the variables to set in the container (empty when no proxy is set)."""
    out = {name: _without_userinfo(container_url(environ[name])) for name in _PROXY_URL_VARS
           if (environ.get(name) or "").strip()}
    if out:
        for name in _NO_PROXY_VARS:
            hosts = [h for h in (environ.get(name) or "").split(",") if h.strip()]
            if HOST_ALIAS not in (h.strip() for h in hosts):
                hosts.append(HOST_ALIAS)
            out[name] = ",".join(hosts)
    return out


def _endpoint_probe_script(host: str, port: int) -> str:
    """python3 source that opens a TCP connection to host:port, closes it and prints the result.
    Nothing is sent on the connection."""
    return ("import socket\n"
            "try:\n"
            f"    socket.create_connection(({host!r}, {int(port)}), "
            f"{_ENDPOINT_PROBE_CONNECT_S}).close()\n"
            "except OSError as e:\n"
            f"    print({_ENDPOINT_PROBE_MARK!r} + 'closed ' + type(e).__name__ + ' '"
            " + str(getattr(e, 'errno', '') or ''))\n"
            "else:\n"
            f"    print({_ENDPOINT_PROBE_MARK!r} + 'open')\n")


def _force_remove(name: str) -> None:
    """`docker rm -f <name>`, bounded by container_guard.PROBE_RM_TIMEOUT_S; a wedged daemon or a
    missing docker CLI only logs, so the error being raised (or the result) is kept."""
    from .container_guard import PROBE_RM_TIMEOUT_S
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True,
                       timeout=PROBE_RM_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not remove the container %s (%s)", name, type(exc).__name__)


def _bridge_tcp_probe(eval_name: str, image: str, host: str, port: int, docs_url: str,
                      what: str) -> Tuple[Optional[List[str]], Any]:
    """(verdict words, the docker process) of a TCP connection to host:port opened from a
    container of `image` on ENDPOINT_NETWORK (nothing is sent on it): ["open"] (cached per host
    and port for the rest of the process), ["closed", <error>, ...] or None (no verdict). A docker
    call that fails or times out raises infra_required saying gbench could not check `what`."""
    with _ENDPOINT_LOCK:
        if _ENDPOINT_REACHABLE.get((host, port)):
            return ["open"], None
        name = f"gbench-endpoint-probe-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "--rm", "--pull", "never", "--name", name,
               *ENDPOINT_NETWORK_ARGS, "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--entrypoint", "python3", image,
               "-c", _endpoint_probe_script(host, port)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=_ENDPOINT_PROBE_DOCKER_S)
        except (OSError, subprocess.SubprocessError) as e:
            if isinstance(e, subprocess.TimeoutExpired):
                _force_remove(name)
            raise infra_required(
                eval_name, f"gbench could not check that {what} on the docker bridge network "
                f"({type(e).__name__}).", docs_url) from None
        verdict = next((line.strip()[len(_ENDPOINT_PROBE_MARK):].split()
                        for line in (proc.stdout or "").splitlines()
                        if line.strip().startswith(_ENDPOINT_PROBE_MARK)), None)
        if verdict and verdict[0] == "open":
            _ENDPOINT_REACHABLE[(host, port)] = True
        return verdict, proc


def _require_relay_reachable(eval_name: str, image: str, host: str, port: int, docs_url: str,
                             relay: str) -> None:
    """infra_required unless a container of `image` on ENDPOINT_NETWORK opens a TCP connection to
    host:port, where gbench's `relay` (a thread of this process on the bridge gateway) listens."""
    verdict, proc = _bridge_tcp_probe(eval_name, image, host, port, docs_url,
                                      f"gbench's {relay} at {host}:{port} is reachable")
    if verdict and verdict[0] == "open":
        return
    if verdict and verdict[0] == "closed":
        raise infra_required(
            eval_name, f"{eval_name}'s harness container on docker's bridge network cannot open a "
            f"TCP connection to {host}:{port} ({' '.join(verdict[1:])}), where gbench's {relay} "
            "listens: the relay adds the key outside the container, which runs the agent's tool "
            "calls and so never holds one. A host firewall must let the docker bridge reach this "
            f"host's address {host}.", docs_url)
    detail = " ".join(((proc.stderr or proc.stdout) or "no output").split())[-300:]
    raise infra_required(
        eval_name, f"gbench could not check that its {relay} at {host}:{port} is reachable from "
        f"docker's bridge network: the probe (python3 in {image}) gave no verdict (docker exit "
        f"{proc.returncode}: {detail}).", docs_url)


def _require_endpoint_reachable(eval_name: str, image: str, url: str, inner: str,
                                docs_url: str) -> None:
    """infra_required unless a TCP connection to `inner` (the loopback endpoint `url` as the
    container reaches it) opens from a container of `image` on ENDPOINT_NETWORK. The answer is
    cached per host and port for the rest of the process."""
    from urllib.parse import urlsplit
    parts = urlsplit(inner)
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise infra_required(eval_name, "the model endpoint URL has an invalid port.",
                             docs_url) from None
    host = parts.hostname or HOST_ALIAS
    shown = f"{urlsplit(url).scheme}://{urlsplit(url).hostname}:{port}"
    verdict, proc = _bridge_tcp_probe(eval_name, image, host, port, docs_url,
                                      f"the model endpoint {shown} answers")
    if verdict and verdict[0] == "open":
        return
    if verdict and verdict[0] == "closed":
        raise infra_required(
            eval_name, f"the model endpoint {shown} is on this host, and {eval_name}'s harness "
            f"container reaches it at {HOST_ALIAS}:{port} from the docker bridge network (the "
            "host network would put the cloud metadata server within reach of the model's code), "
            f"but no TCP connection opened there ({' '.join(verdict[1:])}). Serve the model on "
            "an address the docker bridge reaches: vLLM as gbench starts it listens on every "
            "interface; a server bound to 127.0.0.1 only does not, and a host firewall must let "
            "the docker bridge reach the port.", docs_url)
    detail = " ".join(((proc.stderr or proc.stdout) or "no output").split())[-300:]
    raise infra_required(
        eval_name, f"gbench could not check that the model endpoint {shown} answers on the docker "
        f"bridge network: the probe (python3 in {image}) gave no verdict (docker exit "
        f"{proc.returncode}: {detail}).", docs_url)


#: How the harness toolchains report a host name that did not resolve: glibc's getaddrinfo
#: messages (Python's socket.gaierror, curl, Gradle and the JVM, cargo), Java's
#: UnknownHostException, urllib3's NameResolutionError ("Failed to resolve '<host>'"), Node's
#: getaddrinfo codes, Go's resolver and Rust's.
DNS_FAILURE = re.compile(
    r"Temporary failure in name resolution|Name or service not known|"
    r"No address associated with hostname|Could not resolve host|Couldn't resolve host|"
    r"UnknownHostException|NameResolutionError|Failed to resolve '|"
    r"getaddrinfo (?:ENOTFOUND|EAI_AGAIN)|nodename nor servname provided|"
    r"failed to lookup address information|"
    r"lookup \S+ on \S+: (?:no such host|server misbehaving|i/o timeout)", re.I)
#: How they report a connection that could not be made.
CONNECT_FAILURE = re.compile(
    r"Connection timed out|connect timed out|Network is unreachable|No route to host|"
    r"Connection refused|Connection reset", re.I)
_HOST_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.-_")


def _names_host(line: str, hosts: Iterable[str]) -> bool:
    """Whether `line` names one of `hosts` or a subdomain of one (whole names only)."""
    return any(re.search(r"(?<![a-z0-9_.-])(?:[a-z0-9_-]+\.)*" + re.escape(h.lower())
                         + r"(?![a-z0-9_-]|\.[a-z0-9_-])", line, re.I) for h in hosts)


def network_failures(text: str, hosts: Optional[Iterable[str]] = None, *,
                     dns_only: bool = False, window: int = 3) -> List[str]:
    """The lines of `text` that show a failed host-name lookup (DNS_FAILURE) or, unless
    `dns_only`, a connection that could not be made (CONNECT_FAILURE). With `hosts`, only
    failures for one of those hosts (or a subdomain) count: the line, or one of the `window` lines
    before it, must name it (toolchains print the failing URL on the line before the error).
    Each line is returned stripped and cut to 300 characters, in order, without repeats."""
    hosts = list(hosts) if hosts is not None else None
    lines = (text or "").splitlines()
    found: List[str] = []
    for i, line in enumerate(lines):
        if not (DNS_FAILURE.search(line) or (not dns_only and CONNECT_FAILURE.search(line))):
            continue
        if hosts is not None and not any(_names_host(prev, hosts)
                                         for prev in lines[max(0, i - window):i + 1]):
            continue
        shown = " ".join(line.split())[:300]
        if shown not in found:
            found.append(shown)
    return found


def _dns_name(host: Optional[str]) -> Optional[str]:
    """`host` as a name a container looks up in DNS, or None: an IP address, HOST_ALIAS (from
    --add-host), `localhost` and a name the probe cannot carry (not ASCII after IDNA) are not."""
    import ipaddress
    host = (host or "").strip().rstrip(".").lower()
    if not host or host in (HOST_ALIAS, "localhost") or host.endswith(".localhost"):
        return None
    try:
        ipaddress.ip_address(host.strip("[]"))
        return None
    except ValueError:
        pass
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    return host if set(host) <= _HOST_CHARS and host[0] not in "-." else None


def _proxy_for(scheme: str, host: str, proxy_env: Dict[str, str]) -> Optional[str]:
    """The proxy URL a client in the container sends a request for `scheme`://`host` through, by
    the proxy variables it gets (`proxy_env`; lowercase first, as curl and httpx read them), or
    None when it goes direct (no proxy for that scheme, or NO_PROXY / no_proxy covers the host)."""
    from urllib.request import proxy_bypass_environment
    scheme = (scheme or "http").lower()
    proxy = (proxy_env.get(f"{scheme}_proxy") or proxy_env.get(f"{scheme.upper()}_PROXY")
             or "").strip()
    if not proxy:
        return None
    bypass = (proxy_env.get("no_proxy") or proxy_env.get("NO_PROXY") or "").strip()
    if bypass and proxy_bypass_environment(host, {"no": bypass}):
        return None
    return proxy


def endpoint_container_hosts(eval_name: str) -> Tuple[List[str], bool]:
    """(the host names `eval_name`'s harness container on ENDPOINT_NETWORK looks up itself at run
    time besides the model endpoint's, whether that container gets this host's proxy variables):

    * tau2 / tau3: none (their user simulator, judge and tau3's embeddings reach the Gemini API
      through gbench's Gemini relay at the bridge gateway's address, `_bridge_services`); the
      container gets the proxy variables (container_proxy_env);
    * livebench: Hugging Face (livebench.CONTAINER_HOSTS: its questions);
    * aider_polyglot: the hosts the exercises' tests fetch from, per language of the run
      (aider_polyglot.container_hosts);
    * any other suite: none (it can pass its own to require_endpoint_network)."""
    if eval_name in ("tau2", "tau3"):
        return [], True
    if eval_name == "livebench":
        from .livebench import CONTAINER_HOSTS
        return list(CONTAINER_HOSTS), False
    if eval_name == "aider_polyglot":
        from .aider_polyglot import container_hosts
        return list(container_hosts()), False
    return [], False


def container_lookups(url: Optional[str], hosts: Iterable[str],
                      proxy_env: Optional[Dict[str, str]] = None) -> List[str]:
    """The host names a container on ENDPOINT_NETWORK resolves itself: the model endpoint's host
    (`url` as the container reaches it) when it is a DNS name, then each of `hosts` (reached over
    https). A request that goes through a proxy of `proxy_env` (the proxy variables the container
    gets) is resolved by the proxy, so the proxy's host is looked up instead."""
    from urllib.parse import urlsplit
    env = proxy_env or {}
    names: List[str] = []

    def add(host: Optional[str]) -> None:
        name = _dns_name(host)
        if name and name not in names:
            names.append(name)

    try:
        endpoint = urlsplit(container_url(url))
        targets = [((endpoint.scheme or "http"), endpoint.hostname)] if endpoint.hostname else []
    except ValueError:
        targets = []
    targets += [("https", h) for h in hosts]
    for scheme, host in targets:
        name = _dns_name(host)
        if not name:
            continue
        proxy = _proxy_for(scheme, name, env)
        if proxy:
            try:
                add(urlsplit(proxy if "://" in proxy else "http://" + proxy).hostname)
            except ValueError:
                pass
        else:
            add(name)
    return names


def require_endpoint_network(eval_name: str, image: str, url: str, docs_url: str, *,
                             resolve: Optional[Iterable[str]] = None,
                             proxied: Optional[bool] = None) -> str:
    """The checks before a harness container that reaches the model endpoint and runs
    model-controlled code or tool calls starts on ENDPOINT_NETWORK; returns `url` as that
    container reaches it (container_url).

    1. The cloud metadata server is unreachable from that network, and the host names the
       container looks up itself resolve there (container_guard.require_metadata_blocked with
       `resolve`, one probe container; infra_required naming the host's DOCKER-USER rule
       otherwise). The names are container_lookups of the endpoint URL and of `resolve`, which
       defaults to the suite's own (endpoint_container_hosts); `proxied` (default: the suite's)
       says whether the container gets this host's proxy variables, in which case a request
       they send through a proxy needs only the proxy's host.
    2. An endpoint on this host (a loopback URL, reached there at HOST_ALIAS) accepts a TCP
       connection from a container of `image` on that network, so a server bound to 127.0.0.1
       only is a hard error before the run, not a run of failed requests. Any other URL is used as
       given.
    `image` is the suite's own local image (nothing is pulled). Both answers are cached for the
    rest of the process."""
    from .container_guard import require_metadata_blocked
    hosts, suite_proxied = endpoint_container_hosts(eval_name)
    if resolve is not None:
        hosts = list(resolve)
    proxy_env = container_proxy_env(dict(os.environ)) if (
        suite_proxied if proxied is None else proxied) else {}
    names = container_lookups(url, hosts, proxy_env)
    require_metadata_blocked(eval_name, image, ENDPOINT_NETWORK, docs_url, resolve=names)
    inner = container_url(url)
    if inner != (url or ""):
        _require_endpoint_reachable(eval_name, image, url, inner, docs_url)
    return inner


def tau2_image() -> str:
    """The local image tau2 / tau3 run in (GBENCH_TAU2_IMAGE overrides the default tag)."""
    return suite_env("GBENCH_TAU2_IMAGE", default=TAU2_IMAGE_DEFAULT) or TAU2_IMAGE_DEFAULT


def _build_hint() -> str:
    return (f"Build it LOCALLY from the gbench repository root (gbench never pulls it): "
            f"`{image_build_command()}`.")


def check_tau_image() -> Tuple[bool, str]:
    """Docker reachable and the local tau2 image present, built at TAU2_BENCH_REVISION."""
    image = tau2_image()
    if not shutil.which("docker"):
        return False, ("tau2 and tau3 run tau2-bench in a local Docker image, but the docker CLI "
                       "is not on PATH. " + _build_hint())
    try:
        info = subprocess.run(["docker", "info"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return False, (f"the Docker daemon is not reachable ({type(e).__name__}: {e}). "
                       + _build_hint())
    if info.returncode != 0:
        return False, "the Docker daemon is not reachable (`docker info` failed). " + _build_hint()
    try:
        insp = subprocess.run(["docker", "image", "inspect", "--format",
                               "{{index .Config.Labels \"" + TAU2_IMAGE_LABEL + "\"}}", image],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"cannot inspect image {image!r} ({type(e).__name__}: {e}). " + _build_hint()
    if insp.returncode != 0:
        return False, f"the tau2-bench image {image!r} is not built. " + _build_hint()
    revision = (insp.stdout or "").strip()
    if revision != TAU2_BENCH_REVISION:
        return False, (f"image {image!r} was built for tau2-bench revision "
                       f"{revision or 'unknown'}, not the pinned {TAU2_BENCH_REVISION}; rebuild "
                       "it. " + _build_hint())
    return True, ""


def _image_id(image: str) -> Optional[str]:
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def _forwarded_env(environ: Optional[Dict[str, str]] = None,
                   domains: Optional[List[str]] = None,
                   bridge_env: Optional[Dict[str, str]] = None) -> List[str]:
    """Names of the variables passed into the image (values stay out of argv): the tau knobs and
    run-shaping variables the host has set, GBENCH_TAU_AGENT_KEY, the variables of
    `bridge_env` (`_bridge_services`: the Gemini relay's placeholder and address, the run's
    Gemini cascade) and NO_PROXY / no_proxy when a proxy is set (container_proxy_env). No
    credential of the host enters the image, whatever the domains (the `domains` argument does
    not change the names). `_launch_env` supplies their values."""
    env = os.environ if environ is None else environ
    names = {_AGENT_KEY_ENV}
    for name in env:
        if name in _NOT_FORWARDED or name in _HOST_ONLY or _is_secret_name(name):
            continue
        if name in _FORWARD_NAMES or name.startswith(_FORWARD_PREFIXES):
            names.add(name)
    names.update(bridge_env or {})
    names.update(container_proxy_env(env))
    return sorted(names)


def _launch_env(environ: Optional[Dict[str, str]] = None,
                bridge_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment of the `docker run` process, which Docker copies into the image for each
    `-e NAME` on the command line (`_forwarded_env` decides those names): the host environment
    without its credentials and endpoints (`_HOST_ONLY`), GBENCH_TAU_AGENT_KEY=EMPTY (a real
    endpoint key is added outside the container by gbench's endpoint relay; a placeholder is no
    key), the proxy variables as the bridge network needs them (container_proxy_env), and
    `bridge_env`."""
    src = dict(os.environ if environ is None else environ)
    env = {k: v for k, v in src.items() if k not in _HOST_ONLY}
    env[_AGENT_KEY_ENV] = "EMPTY"     # a real key stays with gbench's endpoint relay
    env.update(container_proxy_env(src))
    env.update(bridge_env or {})
    return env


def _image_bootstrap(suites_dir: str = _IMAGE_SUITES_DIR, io_dir: str = _IMAGE_IO_DIR) -> str:
    """The `python -c` program the image runs: import this module from the mounted package
    directory under an empty package (so nothing else of gbench is imported) and run the job."""
    return ("import importlib, sys, types\n"
            f"pkg = types.ModuleType({_IMAGE_PACKAGE!r})\n"
            f"pkg.__path__ = [{suites_dir!r}]\n"
            f"sys.modules[{_IMAGE_PACKAGE!r}] = pkg\n"
            f"tc = importlib.import_module({_IMAGE_PACKAGE + '.tau_common'!r})\n"
            f"sys.exit(tc._image_main({io_dir!r}))\n")


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)
    os.replace(tmp, path)


def _image_main(io_dir: str) -> int:
    """Entry point inside the image: run `run_tau_env` for `<io_dir>/job.json` and write
    `<io_dir>/result.json` ({"ok": true, "result": ...} or {"ok": false, "error": ...}), with every
    credential value of the environment (the endpoint key included) redacted."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    result_path = os.path.join(io_dir, "result.json")
    secrets = _secret_values(os.environ)       # before run_tau_env removes GBENCH_TAU_AGENT_KEY
    try:
        with open(os.path.join(io_dir, "job.json"), encoding="utf-8") as fh:
            job = json.load(fh)
        from .base import set_run_knobs
        set_run_knobs(**(job.get("run_knobs") or {}))
        result = run_tau_env(job["eval_name"], list(job["domains"]), job["model_name"],
                             job["base_url"], int(job["concurrency"]), job.get("limit"),
                             job["docs_url"], enable_thinking=bool(job.get("enable_thinking")),
                             shard=job.get("shard"))
        _write_json_atomic(result_path, _redact({"ok": True, "result": result}, secrets))
        return 0
    except BaseException as e:                                   # noqa: BLE001
        # The host redacts the streamed output, traceback included.
        logger.error("tau2 run in the image failed: %s", _redact_text(str(e), secrets),
                     exc_info=True)
        _write_json_atomic(result_path, _redact({"ok": False, "error_type": type(e).__name__,
                                                 "error": str(e)}, secrets))
        return 1


def _docker_run_command(name: str, image: str, domains: List[str], io_dir: str,
                        cache_dir: str, trace_root: Optional[str], env_names: List[str],
                        suites_dir: Optional[str] = None) -> List[str]:
    """The `docker run` command for one suite run (no secret values on the command line)."""
    suites_dir = suites_dir or os.path.dirname(os.path.abspath(__file__))
    cmd = ["docker", "run", "--rm", "--name", name, *ENDPOINT_NETWORK_ARGS,
           "--user", f"{os.getuid()}:{os.getgid()}",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
    if "banking_knowledge" in domains:
        cmd += list(_SHELL_SANDBOX_DOCKER_OPTS)
    # LiteLLM would otherwise fetch its model cost map from raw.githubusercontent.com at import;
    # the bundled copy it ships is used instead (no network call, no effect on any request).
    cmd += ["-e", "HOME=/tmp", "-e", "USER=gbench", "-e", "LOGNAME=gbench",
            "-e", "LITELLM_LOCAL_MODEL_COST_MAP=True",
            "-e", f"GBENCH_TAU2_SAVE_TRACES={_IMAGE_TRACE_DIR if trace_root else ''}"]
    for env_name in env_names:
        cmd += ["-e", env_name]
    cmd += ["-v", f"{suites_dir}:{_IMAGE_SUITES_DIR}:ro",
            "-v", f"{io_dir}:{_IMAGE_IO_DIR}:rw",
            "-v", f"{cache_dir}:{_IMAGE_CACHE_DIR}:rw", "-w", _IMAGE_CACHE_DIR]
    if trace_root:
        cmd += ["-v", f"{trace_root}:{_IMAGE_TRACE_DIR}:rw"]
    cmd += [image, "python", "-I", "-B", "-u", "-c", _image_bootstrap()]
    return cmd


def _launch_image(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run the container with `env` (default `_launch_env()`) as the docker CLI's environment,
    streaming its output (progress bar, per-domain lines) to stderr line by line with every
    credential value of the host and of `env` redacted. Returns (exit code, the last part of the
    redacted output)."""
    env = _launch_env() if env is None else env
    secrets = _secret_values(os.environ, extra=tuple(_secret_values(env)))
    tail: List[str] = []
    pending = ""

    def emit(text: str) -> None:
        text = _redact_text(text, secrets)
        try:
            sys.stderr.write(text)
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        tail.append(text)
        if len(tail) > 64:
            del tail[:-64]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):
            pending += chunk.decode("utf-8", "replace")
            cut = max(pending.rfind("\n"), pending.rfind("\r"))
            if cut >= 0:
                emit(pending[:cut + 1])
                pending = pending[cut + 1:]
            elif len(pending) > 65536:
                emit(pending)
                pending = ""
        if pending:
            emit(pending)
        return proc.wait(), "".join(tail)[-4000:]
    finally:
        if proc.poll() is None:
            proc.kill()


def _tau_cache_dir() -> str:
    """The host directory the container works in: `<gbench cache>/tau2`, where tau2 keeps
    banking_knowledge's document embeddings under data/.embeddings_cache. tau2 keys that cache on
    the documents and the embedder's type and model (knowledge/embeddings_cache.py), so the
    Gemini vectors (embedder_type "gemini", model gemini-embedding-001) never mix with vectors of
    another embedder."""
    from .base import gbench_cache_dir
    return os.path.join(gbench_cache_dir(), "tau2")


def _host_path(path: Optional[str], trace_root: Optional[str]) -> Optional[str]:
    """Map a path under the image's trace mount back to the host."""
    if not path or not trace_root:
        return path
    if path == _IMAGE_TRACE_DIR or path.startswith(_IMAGE_TRACE_DIR + "/"):
        return os.path.join(trace_root, os.path.relpath(path, _IMAGE_TRACE_DIR))
    return path


class _TauBridge:
    """What the tau2 container uses in place of credentials (`_bridge_services`): gbench's Gemini
    relay (a thread of this process on the docker bridge gateway, which adds the Gemini key) and
    gbench's endpoint relay there, for every run (it adds a real endpoint key when there is one
    and the run's --max-output-tokens on agent requests that carry none). `env` is
    what the container gets instead: the relay's placeholder and address, the run's Gemini
    cascade and, with a proxy, NO_PROXY / no_proxy naming the gateway."""

    def __init__(self, agent_base_url: str):
        self.agent_base_url = agent_base_url
        self.gateway: Optional[str] = None
        self.gemini_relay: Any = None
        self.endpoint_relay: Any = None
        self.env: Dict[str, str] = {}
        #: The host's proxy variables whose credentials stayed out of the container.
        self.proxy_credentials_withheld: List[str] = []

    def record(self) -> Dict[str, Any]:
        """The relays' records for the result (never a key)."""
        return {"gateway": self.gateway,
                "proxy_credentials_withheld": list(self.proxy_credentials_withheld),
                "gemini_relay": self.gemini_relay.record() if self.gemini_relay else None,
                "endpoint_relay": (self.endpoint_relay.record()
                                   if self.endpoint_relay is not None
                                   and hasattr(self.endpoint_relay, "record") else None)}


def _require_gemini_access(eval_name: str, docs_url: str) -> None:
    """The Gemini gate, on the host (the image never holds the key): a key must be configured
    and not refused, and every model the user simulator and the judge may call must be served
    (base.gemini_key_live_valid, then the role knobs' models with base.gemini_models_available).
    A network error is inconclusive and never blocks. infra_required otherwise."""
    from .base import (MODELS_NOT_SERVED_PREFIX, gemini_api_keys, gemini_key_live_valid,
                       gemini_models_available, judge_cascade)
    keys = gemini_api_keys()
    if not keys:
        raise infra_required(
            eval_name, "tau2 and tau3 need GEMINI_API_KEY (or GEMINI_API_KEYS): every external "
            "model a tau run calls (the user simulator, the nl-assertion judge and tau3's "
            "retrieval embeddings) goes through the Gemini API, by way of gbench's Gemini relay "
            "on this host. See docs/evals/tau2.md.", docs_url)
    ok, why = gemini_key_live_valid(",".join(keys))
    if ok:
        cascade = {_bare_model(m) for m in judge_cascade()}
        extra = []
        for role in (_USER_ROLE, _JUDGE_ROLE):
            override, knob = _role_override(role)
            if override and _bare_model(override) not in cascade \
                    and (_bare_model(override), knob) not in extra:
                extra.append((_bare_model(override), knob))
        if extra:
            ok, why = gemini_models_available(models=extra)
    if ok:
        return
    if str(why).startswith(MODELS_NOT_SERVED_PREFIX):
        raise infra_required(
            eval_name, f"{why}; every model the tau2 user simulator and NL-assertion judge may "
            "call must be one the Gemini API serves.", docs_url)
    raise infra_required(
        eval_name, f"the Gemini API refused the key (GEMINI_API_KEYS / GEMINI_API_KEY): {why}; a "
        "valid key is required for the tau2 user simulator, the NL-assertion judge and tau3's "
        "embeddings.", docs_url)


def _relay_models(domains: List[str]) -> List[str]:
    """The models the Gemini relay forwards for the container: both roles' models and, for
    banking_knowledge, the retrieval embedder's."""
    models: List[str] = []
    for role in (_USER_ROLE, _JUDGE_ROLE):
        for m in _role_cascade(role):
            if _bare_model(m) not in models:
                models.append(_bare_model(m))
    if "banking_knowledge" in domains:
        models.append(GEMINI_EMBED_MODEL)
    return models


def _port_of(url: str) -> int:
    from urllib.parse import urlsplit
    return int(urlsplit(url).port or 0)


def agent_always_relayed() -> bool:
    """Whether the agent's (the model under test's) requests always go through gbench's endpoint
    relay, with or without an endpoint key: True since 2026-10-13 (`_bridge_services`; until then
    the relay started only with a real key or a proxy with credentials, so the run's
    --max-output-tokens, the relay's request-shape check and its records never applied on the
    user's keyless vLLM). The unit tests that fake docker keep the keyless shortcut (the agent
    reaches the endpoint directly without a real key) through an autouse fixture in
    tests/conftest.py; the real function stays reachable there as `.original`
    (tests/test_relay_always_tau_bfcl.py)."""
    return True


@contextlib.contextmanager
def _bridge_services(eval_name: str, image: str, base_url: str, agent_base_url: Optional[str],
                     domains: List[str], docs_url: str):
    """The relays the tau2 container uses instead of keys, for the duration of the block (yields
    a _TauBridge; the relays stop at the end).

    The container runs the agent's tool calls (tau3's shell commands too, as the same user), so
    it never holds a key. Before anything starts, the Gemini gate runs on the host
    (`_require_gemini_access`). Then gbench's Gemini relay (gemini_relay.GeminiRelay) starts on
    the docker bridge gateway, forwarding only native generateContent (and, for
    banking_knowledge, batchEmbedContents) for the run's models (`_relay_models`); the container
    gets its placeholder as GEMINI_API_KEY and its address as GBENCH_GEMINI_RELAY_URL. gbench's
    endpoint relay (swebench_common.EndpointRelay) starts there too, for every run: it adds the
    endpoint key when there is a real one (`_real_endpoint_key`; key=None on a keyless endpoint)
    and puts the run's --max-output-tokens on each agent request that carries none
    (swe_agentic.AGENT_MAX_TOKENS_SUITES has tau2 and tau3; the relay reads the run's `suite`
    knob). The agent's base URL is the relay's and GBENCH_TAU_AGENT_KEY is EMPTY. Only the agent
    (the model under test) goes through it: the user simulator, the NL-assertion judge and tau3's
    embedder call Gemini through the Gemini relay, unchanged and uncapped. The endpoint relay is
    no smoke relay (smoke_from_env=False): a tau2 domain whose every simulation ends at a relay's
    GBENCH_SMOKE_STOP has no evaluable simulation and hard-errors, so a `gbench --smoke` tau
    episode runs to its own end, as before. A container of the image must reach each relay
    (`_require_relay_reachable`). `agent_base_url` is the URL the agent uses when no endpoint
    relay runs, which happens only under the unit tests' keyless shortcut (agent_always_relayed
    patched in tests/conftest.py); a run always gives the agent the endpoint relay's URL.

    2026-10-13: until then the endpoint relay started only with a real endpoint key or a proxy
    with credentials, so on the user's keyless vLLM the agent's requests went straight to the
    server: smoke run smoke-10-12-235114 tapped 86 tau2 and 133 tau3 agent requests, none with
    max_tokens (the run's --max-output-tokens 65536 never applied), and no relay request-shape
    check ran."""
    from .base import JUDGE_MODELS_ENV, gemini_cascade_env
    from .gemini_relay import RELAY_URL_ENV, GeminiRelay, network_gateway
    _require_gemini_access(eval_name, docs_url)
    bridge = _TauBridge(agent_base_url or base_url)
    gateway = network_gateway(eval_name, ENDPOINT_NETWORK, docs_url)
    bridge.gateway = gateway
    methods = ["generateContent"] + (["batchEmbedContents"]
                                     if "banking_knowledge" in domains else [])
    try:
        bridge.gemini_relay = GeminiRelay(
            eval_name, gateway, docs_url=docs_url, openai_calls=(), native_methods=methods,
            native_models=_relay_models(domains), name=f"{eval_name}-gemini-relay").start()
        _require_relay_reachable(eval_name, image, gateway, _port_of(bridge.gemini_relay.origin),
                                 docs_url, "Gemini relay")
        key = _real_endpoint_key()
        # A proxy with credentials stays on the host (container_proxy_env leaves them out), so
        # the agent's requests then leave from this process, through gbench's endpoint relay,
        # which uses the host's proxy settings.
        bridge.proxy_credentials_withheld = proxy_credentials(dict(os.environ))
        # Every run, keyless too (agent_always_relayed, 2026-10-13; see the docstring): the relay
        # adds a real key when there is one and the run's --max-output-tokens to agent requests
        # that carry none.
        if agent_always_relayed() or key or bridge.proxy_credentials_withheld:
            from .swebench_common import EndpointRelay
            try:
                relay = EndpointRelay(base_url, gateway, key=key or None,
                                      name=f"{eval_name}-endpoint-relay",
                                      smoke_from_env=False)
                # outlasts the agent's own request timeout (agent_client_timeout)
                relay.TIMEOUT_S = max(int(getattr(relay, "TIMEOUT_S", 0) or 0),
                                      agent_client_timeout()["timeout_s"] + 120)
            except ValueError as e:
                raise infra_required(
                    eval_name, f"the model endpoint URL cannot be relayed ({e}); gbench relays "
                    "the agent's requests (its relay adds the endpoint key, which never enters "
                    "the tau2 container, and the run's --max-output-tokens).", docs_url) from None
            try:
                relay.start()
            except OSError as e:
                raise infra_required(
                    eval_name, f"gbench's endpoint relay could not listen on the docker bridge "
                    f"gateway {gateway} ({type(e).__name__}: {e.strerror or e}).",
                    docs_url) from None
            bridge.endpoint_relay = relay
            _require_relay_reachable(eval_name, image, gateway, _port_of(relay.url), docs_url,
                                     "endpoint relay")
            bridge.agent_base_url = relay.url
        from .base import judge_cascade_source
        env = {"GEMINI_API_KEY": bridge.gemini_relay.placeholder,
               RELAY_URL_ENV: bridge.gemini_relay.origin,
               JUDGE_MODELS_ENV: gemini_cascade_env()[JUDGE_MODELS_ENV],
               # where that list came from on the host (the container's own reading would
               # always say GBENCH_JUDGE_MODELS)
               CASCADE_SOURCE_ENV: judge_cascade_source()}
        if bridge.endpoint_relay is not None:
            env[_AGENT_KEY_ENV] = "EMPTY"
        proxies = container_proxy_env(dict(os.environ))
        if proxies:
            for name in _NO_PROXY_VARS:
                hosts = [h for h in (proxies.get(name) or "").split(",") if h.strip()]
                if gateway not in (h.strip() for h in hosts):
                    hosts.append(gateway)
                env[name] = ",".join(hosts)
        bridge.env = env
        yield bridge
    finally:
        for relay in (bridge.gemini_relay, bridge.endpoint_relay):
            if relay is not None:
                try:
                    relay.stop()
                except Exception as e:                           # noqa: BLE001
                    logger.warning("%s: could not stop a relay (%s)", eval_name,
                                   type(e).__name__)


def run_tau_suite(
    eval_name: str,
    domains: List[str],
    model_name: str,
    base_url: str,
    concurrency: int,
    limit: Optional[int],
    docs_url: str,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Run tau2 / tau3 in the local gbench-tau2 image and return `run_tau_env`'s result.

    The container runs on docker's default bridge network (ENDPOINT_NETWORK), never the host
    network: the agent's tool calls and banking_knowledge's shell tool run in it, and the host's
    DOCKER-USER rule keeps the cloud metadata server out of reach of a bridge network only.
    Before it starts, container_guard.require_metadata_blocked checks that the metadata server is
    blocked there. The agent always reaches the endpoint through gbench's endpoint relay on the
    bridge gateway (`_bridge_services`, every run since 2026-10-13, keyless too: it adds a real
    endpoint key and the run's --max-output-tokens on agent requests that carry none), so the
    endpoint's own reachability from the container is not checked; the relay's is. The
    container holds no key: the Gemini gate runs here, and gbench's Gemini relay and endpoint
    relay add the keys on this host. The
    container also gets the host user's uid/gid (files it writes are the operator's), this
    package directory read-only, a per-run exchange directory for job.json / result.json, the
    trace root (when traces are kept) and `_tau_cache_dir` as its working directory. No Docker,
    no image, an image built at another revision, no Gemini key or a refused one, or a user
    simulator / judge that is not a gemini/ model hard-errors.
    """
    from .base import current_shard, get_run_knob
    # --shard (GBENCH_SHARD) is validated here, on the host: a malformed value raises before
    # anything starts; the container gets the normalized "I/N" in job.json.
    shard_spec = current_shard()
    shard = shard_spec.label if shard_spec else None
    removed = [n for n in _REMOVED_KNOBS if os.environ.get(n) is not None]
    if removed:
        logger.warning("%s: %s no longer change anything: every external model a tau run calls "
                       "goes through the Gemini API with GEMINI_API_KEY, and banking_knowledge "
                       "always runs upstream's alltools retrieval, whose dense search embeds with "
                       "%s; see docs/evals/tau3.md", eval_name, ", ".join(removed),
                       GEMINI_EMBED_MODEL)
    ok, why = check_role_providers()
    if not ok:
        raise infra_required(eval_name, why, docs_url)
    # gbench's Gemini relay sends the key to GEMINI_OPENAI_BASE_URL's origin, accepted only as
    # the Gemini API over https or a loopback address; any other value is a hard error here.
    from .base import require_gemini_openai_base_url
    require_gemini_openai_base_url(eval_name, docs_url)
    ok, why = check_tau_image()
    if not ok:
        what = ("tau3 runs tau2-bench's banking_knowledge domain" if eval_name == "tau3"
                else f"{eval_name} runs the tau2-bench simulator")
        raise infra_required(eval_name, f"{what} in the local gbench-tau2 image: {why}", docs_url)
    image = tau2_image()
    # The container runs on the bridge network: the metadata server must be blocked there. The
    # agent reaches gbench's endpoint relay on the bridge gateway (every run since 2026-10-13,
    # keyless too; `_bridge_services` checks that a container reaches it), never the endpoint
    # itself, so the endpoint's own reachability from the container is not checked.
    if agent_always_relayed() or _real_endpoint_key():
        from .container_guard import require_metadata_blocked
        require_metadata_blocked(eval_name, image, ENDPOINT_NETWORK, docs_url, resolve=[])
        agent_base_url = None
    else:
        # the pre-2026-10-13 keyless path, kept for the unit tests' keyless shortcut
        # (tests/conftest.py): the agent reaches a loopback endpoint at HOST_ALIAS
        agent_base_url = require_endpoint_network(eval_name, image, base_url, docs_url)
    trace_root = _resolve_tau_trace_dir()
    cache_dir = _tau_cache_dir()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        if trace_root:
            os.makedirs(trace_root, exist_ok=True)
    except OSError as e:
        raise infra_required(eval_name, f"cannot create the tau2 cache or trace directory "
                             f"({type(e).__name__}: {e}); set GBENCH_TAU2_SAVE_TRACES to a "
                             "writable directory", docs_url) from e
    with _bridge_services(eval_name, image, base_url, agent_base_url, list(domains),
                          docs_url) as bridge:
        result = _run_in_image(eval_name, image, domains, model_name, bridge, concurrency, limit,
                               docs_url, enable_thinking, trace_root, cache_dir, get_run_knob,
                               shard=shard)
        result.setdefault("tau2_report", {}).update(bridge.record())
    record_agent_max_tokens(result, bridge.record().get("endpoint_relay"),
                            get_run_knob("max_output_tokens"))
    return result


#: The comparability reason evals._stamp_agent_max_tokens gives a result whose relays filled the
#: run's --max-output-tokens into agent requests, word for word (relay_fill_reason). A suite
#: result that states it is not given it twice: the stamp adds its reason only when the text is
#: absent. 2026-10-13 review: record_agent_max_tokens and bfcl_v4_agentic stated it in other words,
#: so a result carried the same deviation twice.
RELAY_FILL_REASON = re.compile(r"max_tokens=\d+ \(the run's --max-output-tokens\) sent with \d+ "
                               r"agent request\(s\) whose upstream harness sends none")


def relay_fill_reason(value: int, filled: int) -> str:
    """evals._stamp_agent_max_tokens's reason for `filled` agent requests sent with max_tokens
    `value` (RELAY_FILL_REASON)."""
    return (f"max_tokens={int(value)} (the run's --max-output-tokens) sent with {int(filled)} "
            "agent request(s) whose upstream harness sends none")


def sum_relay_fills(fills: List[Any]) -> Optional[Dict[str, int]]:
    """{value, requests_filled} over relay `max_tokens_from_run` records (EndpointRelay.record):
    the largest value and the requests filled summed; None when no request was filled (no
    record, the fill off, or a relay no agent request went through)."""
    value = filled = 0
    for fill in fills:
        if isinstance(fill, dict) and fill.get("value"):
            value = max(value, int(fill["value"]))
            filled += int(fill.get("requests_filled") or 0)
    return {"value": value, "requests_filled": filled} if value and filled else None


def merge_relay_fill_reasons(merged: Dict[str, Any], kept: List[str],
                             fill: Optional[Dict[str, int]]) -> List[str]:
    """`kept` (the merged comparability reasons) without the shards' relay-fill reasons
    (RELAY_FILL_REASON, each with its own shard's count) and with one over the shards' summed
    `fill`; the eval runner's `agent_max_tokens_from_run` of `merged` (one shard's) is set to the
    sum too. For merge_tau_shards and bfcl_v4_agentic.merge_shards."""
    out = [r for r in kept if not RELAY_FILL_REASON.fullmatch(r)]
    if fill:
        out.append(relay_fill_reason(fill["value"], fill["requests_filled"]))
        if isinstance(merged.get("agent_max_tokens_from_run"), dict):
            merged["agent_max_tokens_from_run"] = dict(
                merged["agent_max_tokens_from_run"], max_tokens=fill["value"],
                requests_filled=fill["requests_filled"])
    return out


def record_agent_max_tokens(result: Dict[str, Any], relay_record: Optional[Dict[str, Any]],
                            knob: Any) -> Dict[str, Any]:
    """Record in `result` the max_tokens the agent's requests went out with. tau2's agent sends
    none (its llm_args carry the temperature, the request timeout, the chat template's thinking
    switch with --thinking, and top_p / top_k when set), so gbench's endpoint relay fills the
    run's --max-output-tokens into each of them (swe_agentic.AGENT_MAX_TOKENS_SUITES; the user's
    decision of 2026-10-11), as its record's `max_tokens_from_run` {value, requests_filled}
    shows. When it filled at least one request, `max_tokens_sent` is that value,
    official_sampling.agent_max_tokens_from_run holds swe_agentic.agent_max_tokens_record and
    leaderboard_comparable_reason names the deviation (upstream sends none) in the eval runner's
    words (relay_fill_reason), so evals._stamp_agent_max_tokens does not add it again. Without
    a fill (no --max-output-tokens, GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0, or no agent request went
    through the relay) nothing is recorded as sent. 2026-10-13: before the relay started on
    keyless endpoints the result said "not sent" while the run had --max-output-tokens 65536;
    the same day's review found a relay that filled 0 requests recorded as a cap on every request,
    and the reason stated twice."""
    fill = (relay_record or {}).get("max_tokens_from_run") or {}
    value = fill.get("value")
    filled = int(fill.get("requests_filled") or 0)
    if result.get("empty_shard"):
        result["max_tokens_sent"] = None
        result["max_output_tokens_note"] = "no agent request: the shard holds no task"
        return result
    if not value or not filled:
        result["max_tokens_sent"] = None
        if value:
            result["max_output_tokens_note"] = (
                f"not sent: gbench's endpoint relay was set to fill max_tokens={int(value)} (the "
                "run's --max-output-tokens), but no agent request went through it")
            return result
        result["max_output_tokens_note"] = (
            "not sent: tau2-bench's agent llm_args carry no max_tokens (as upstream), so the "
            "server's default applies (the room left in the context window)"
            + (f"; --max-output-tokens {knob} not applied (the relay's fill was off: "
               "GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0)" if knob else ""))
        return result
    from .swe_agentic import agent_max_tokens_record
    rec, _why = agent_max_tokens_record(int(value), "tau2-bench's agent",
                                        "gbench's endpoint relay (max_tokens_from_run: filled "
                                        "into each agent request that carried none)")
    rec["requests_filled"] = filled
    result["max_tokens_sent"] = int(value)
    result["max_output_tokens_note"] = (
        f"max_tokens={int(value)} (the run's --max-output-tokens) on every agent request, filled "
        f"in by gbench's endpoint relay ({filled} request(s)); tau2-bench's agent "
        "sends none. The Gemini user simulator and judge are not capped")
    result.setdefault("official_sampling", {})["agent_max_tokens_from_run"] = rec
    reasons = [r for r in str(result.get("leaderboard_comparable_reason") or "").split("; ")
               if r and not RELAY_FILL_REASON.fullmatch(r)]
    reasons.append(relay_fill_reason(int(value), filled))
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result


def _image_run_knobs(get_run_knob) -> Dict[str, Any]:
    """The run knobs the image's run_tau_env reads: the temperature, the attempt count and, when
    the server reported one, its context window (it sizes the agent's request timeout,
    agent_client_timeout)."""
    knobs = {"temperature": get_run_knob("temperature"),
             "attempt_count": get_run_knob("attempt_count")}
    if get_run_knob("max_model_len"):
        knobs["max_model_len"] = get_run_knob("max_model_len")
    return knobs


def _run_in_image(eval_name, image, domains, model_name, bridge, concurrency, limit, docs_url,
                  enable_thinking, trace_root, cache_dir, get_run_knob,
                  shard: Optional[str] = None) -> Dict[str, Any]:
    """Start the container for one run (the relays of `bridge` are up) and read its result.
    `shard` is the host-validated "I/N" of --shard (None: every task)."""
    io_dir = tempfile.mkdtemp(prefix=f"gbench_{_safe_name(eval_name)}_io_")
    name = f"gbench-{_safe_name(eval_name)}-{uuid.uuid4().hex[:10]}"
    launch_env = _launch_env(bridge_env=bridge.env)
    host_secrets = _secret_values(os.environ, extra=tuple(_secret_values(launch_env)))
    try:
        # job.json is readable inside the container (the shell tool's sandbox sees its files), so
        # it holds no credential: those travel as environment variables named on the command line.
        _write_json_atomic(os.path.join(io_dir, "job.json"), {
            "eval_name": eval_name, "domains": list(domains), "model_name": model_name,
            "base_url": bridge.agent_base_url, "concurrency": int(concurrency), "limit": limit,
            "docs_url": docs_url, "enable_thinking": bool(enable_thinking), "shard": shard,
            "run_knobs": _image_run_knobs(get_run_knob)})
        cmd = _docker_run_command(name, image, list(domains), io_dir, cache_dir, trace_root,
                                  _forwarded_env(domains=list(domains), bridge_env=bridge.env))
        logger.info("%s: running tau2-bench %s in image %s (container %s)", eval_name,
                    TAU2_BENCH_REVISION[:12], image, name)
        try:
            code, tail = _launch_image(cmd, launch_env)
        finally:
            _force_remove(name)
        result_path = os.path.join(io_dir, "result.json")
        if not os.path.exists(result_path):
            raise infra_required(eval_name, f"the tau2 container {name} exited {code} without a "
                                 f"result; last output: "
                                 f"{_redact_text(' '.join(tail.split()), host_secrets)[-1500:]}",
                                 docs_url)
        with open(result_path, encoding="utf-8") as fh:
            payload = _redact(json.load(fh), host_secrets)
        if not payload.get("ok"):
            message = str(payload.get("error") or "")
            if message.startswith("[INFRA REQUIRED]"):
                logger.error(message)
                raise RuntimeError(message)
            raise infra_required(eval_name, f"the tau2 run in image {image} failed "
                                 f"({payload.get('error_type')}: {message})", docs_url)
        result = payload["result"]
        report = result.setdefault("tau2_report", {})
        report["run_dir"] = _host_path(report.get("run_dir"), trace_root)
        if isinstance(report.get("traces"), dict):
            report["traces"] = {d: _host_path(p, trace_root) for d, p in report["traces"].items()}
        report["image"] = {"name": image, "id": _image_id(image),
                           "dockerfile": TAU2_DOCKERFILE,
                           "shell_sandbox_docker_opts": ("banking_knowledge" in domains)}
        # The image already redacts its files; this pass also covers any host-only credential.
        if report.get("run_dir") and os.path.isdir(report["run_dir"]):
            _redact_tree(report["run_dir"], host_secrets)
        return result
    finally:
        shutil.rmtree(io_dir, ignore_errors=True)
