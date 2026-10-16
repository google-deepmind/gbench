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

# Evaluation Suite: agent_dojo
# Description: AgentDojo (Tool-Use Adversarial Prompt Injection Attack Resilience Benchmark)

"""gbench native built-in runner for agent_dojo (Instruction & Safety).

Canonical AgentDojo (ethz-spylab/agentdojo) measures two things that only exist inside its
tool-execution environment: **utility** (did the agent actually complete the user's task, by
calling the suite's real tools) and **security / attack success rate** (did an injected
instruction get the agent to perform the attacker's task). gbench delegates to the agentdojo
package's OWN suites, attacks and scorers rather than reimplementing them:

* It builds an in-process AgentDojo pipeline (`SystemMessage -> InitQuery -> OpenAILLM ->
  ToolsExecutionLoop`) pointed at the served `/v1` endpoint via `OpenAILLM` (AgentDojo's
  `vllm_parsed` provider), which uses the NATIVE OpenAI tools API, so the endpoint must parse
  tool calls: a server gbench launches does so when `--tool-call-parser NAME` is given (gbench
  then adds `--enable-auto-tool-choice --tool-call-parser NAME`); a remote endpoint must be
  started that way. The prompt-based `LocalLLM` would instead see `content=None` because the
  server's parser consumes the model's function-call text. The requests carry the served model
  id gbench passes (GBENCH_AGENTDOJO_MODEL overrides it, checked against the endpoint's
  /v1/models first) and the endpoint's key (base.endpoint_api_key).
* For each of the 4 canonical suites (workspace / travel / banking / slack) at a pinned
  `benchmark_version` it runs both `benchmark_suite_without_injections` (benign utility) and
  `benchmark_suite_with_injections` under the `important_instructions` attack (utility under
  attack + attack success), one episode at a time through the per-task runners those functions
  loop over. `security_results[...] == True` means THE ATTACK SUCCEEDED (the agentdojo
  docstring: "the second [bool] indicat[es] if the injection was successful").
* An episode an endpoint failure or any other exception ends is a harness error: left out of
  every number, counted and named (completed_with_errors; no score when no benign episode was
  scored). Every episode is a trace with AgentDojo's logged messages; the endpoint's replies are
  counted (end of the context window, empty, empty after reasoning).

Headline `accuracy` = benign task utility. `attack_success_rate` (lower is better) is reported
separately and is NOT folded into accuracy. The agentdojo package is REQUIRED; if it cannot be
imported the suite hard-errors (never skips).

Sampling: gbench's rule for every suite (base.resolve_temperature): temperature 1.0 with
`--thinking`, 0.0 without; `--temperature`, or `GBENCH_AGENT_DOJO_TEMPERATURE` for this suite alone
(it takes precedence), sets another in either mode. AgentDojo's `vllm_parsed` provider
(agent_pipeline.get_llm builds `OpenAILLM(client, model)` with its default temperature 0.0, and
OpenAILLM sends `temperature or NOT_GIVEN`) sends no temperature at 0.0, so gbench puts the run's
temperature on every request itself (_send_temperature, on the client's chat.completions.create):
a 0.0 is sent as 0.0. The result records the official setting under `official_sampling`. This suite uses **no LLM judge**
(AgentDojo scores against its environments' ground truth).

Thinking: the same wrapper puts the chat template's thinking switch on every request
(`extra_body.chat_template_kwargs.enable_thinking`: true with `--thinking`, false without), which
AgentDojo's OpenAI tools path never sends; before 2026-10-11 a `--thinking` run changed only the
temperature. The result records `thinking`, `thinking_applied` (every request carried the run's
switch) and `request_shape` (the requests counted by the value they carried).

Request timeout: the OpenAI client gbench builds gets base.request_timeout_s of the served context
window (AgentDojo sends no max_tokens) and max_retries 0, in place of the SDK's 600 s and 2 retries,
which dropped and re-asked any turn needing more than 600 s of decode (a hidden cap; 2026-10-10
investigation). AgentDojo's own retry (3 attempts) is unchanged. The result records
`model_client_timeout` ({client, version, timeout_s, retries, source}).
"""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (UNCAPPED_TIMEOUT_TOKENS, OfficialSampling, add_comparability_reason,
                   answer_in_thinking_fields, apply_shard, endpoint_api_key, get_run_knob,
                   record_shard, redact_text, request_timeout_s, resolve_official_sampling)
from .limit_plan import eval_limit_plan
from .sampling import allocate_limit, stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/agent_dojo.md"
PILLAR = "Instruction & Safety"

#: The agentdojo release the suite is pinned to (latest on PyPI, 2026-09-24); its tasks,
#: environments and scorers are the benchmark.
AGENTDOJO_VERSION = "0.1.35"
#: Pinned benchmark version and attack (both overridable). v1.2.2 is the latest in the package.
_PINNED_BENCHMARK_VERSION = "v1.2.2"
_PINNED_ATTACK = "important_instructions"

#: agentdojo agent_pipeline.py get_llm("vllm_parsed"): OpenAILLM(client, model) with the default
#: temperature 0.0, which openai_llm.py sends as `temperature or NOT_GIVEN`, i.e. not at all.
AGENTDOJO_SAMPLING = OfficialSampling(
    reference=(f"agentdojo {AGENTDOJO_VERSION} agent_pipeline/agent_pipeline.py get_llm "
               "(vllm_parsed: OpenAILLM default temperature 0.0, sent as `temperature or "
               "NOT_GIVEN`)"),
    send_temperature=False,
)
_BENCHMARK_VERSION = os.environ.get("GBENCH_AGENTDOJO_VERSION", _PINNED_BENCHMARK_VERSION)
_ATTACK = os.environ.get("GBENCH_AGENTDOJO_ATTACK", _PINNED_ATTACK)
#: The pipeline "name" AgentDojo maps to a prose model label for the important_instructions
#: attack (see agentdojo.models.MODEL_NAMES). gemma is not in that map; the canonical label for a
#: locally-served model is `vllm_parsed` -> "Local model" (what AgentDojo's own vllm_parsed
#: provider uses). Override with a `gemini-*` id to get the Google-targeted attack variant (the
#: attack text changes, so the run is then not comparable).
_PINNED_MODEL_LABEL = "vllm_parsed"
_MODEL_LABEL = os.environ.get("GBENCH_AGENTDOJO_MODEL_LABEL", _PINNED_MODEL_LABEL)


def _load_agentdojo() -> Dict[str, Any]:
    """Import the agentdojo package (hard-error with install instructions if missing)."""
    try:
        from agentdojo.agent_pipeline import (AgentPipeline, InitQuery, OpenAILLM, SystemMessage,
                                              ToolsExecutionLoop, ToolsExecutor)
        from agentdojo.agent_pipeline.agent_pipeline import load_system_message
        from agentdojo.attacks.attack_registry import load_attack
        from agentdojo.benchmark import (run_task_with_injection_tasks,
                                         run_task_without_injection_tasks)
        from agentdojo.logging import OutputLogger
        from agentdojo.task_suite.load_suites import get_suites
        import openai
    except Exception as e:  # ImportError, or a transitive dep problem
        raise infra_required(
            "agent_dojo",
            f"the agentdojo package is required and could not be imported ({e}). Install it: "
            f"pip install agentdojo=={AGENTDOJO_VERSION} (base install; the 'transformers' extra "
            "is not needed for the canonical run).",
            DOCS_URL,
        ) from e
    try:
        from importlib.metadata import version as _pkg_version
        installed = _pkg_version("agentdojo")
    except Exception:                                                   # noqa: BLE001
        installed = None
    return {
        "version": installed,
        "AgentPipeline": AgentPipeline, "InitQuery": InitQuery, "OpenAILLM": OpenAILLM,
        "SystemMessage": SystemMessage, "ToolsExecutionLoop": ToolsExecutionLoop,
        "ToolsExecutor": ToolsExecutor, "load_system_message": load_system_message,
        "load_attack": load_attack, "get_suites": get_suites,
        # one episode at a time: AgentDojo's own per-task runners (benchmark.py), which the
        # suite-level benchmark_suite_* functions loop over
        "run_with": run_task_with_injection_tasks,
        "run_without": run_task_without_injection_tasks,
        "OutputLogger": OutputLogger, "openai": openai,
    }


def _served_model_id(model_name: str, base_url: Optional[str] = None) -> Tuple[str, str]:
    """(the model id the requests carry, where it came from): `model_name`, the id gbench passes
    (the endpoint's served id, which the eval runner matched against --models before the suite
    started); GBENCH_AGENTDOJO_MODEL instead when set, checked against the endpoint's /v1/models
    listing first (an id the endpoint does not serve is infra_required, never sent)."""
    override = (os.environ.get("GBENCH_AGENTDOJO_MODEL") or "").strip()
    if not override:
        return model_name, "the served model id gbench passed"
    from ...utils.endpoint import EndpointError, resolve_served_model
    try:
        served = resolve_served_model(base_url or "", [override], override)
    except EndpointError as e:
        raise infra_required("agent_dojo", f"GBENCH_AGENTDOJO_MODEL={override!r} is not a model "
                             f"the endpoint serves ({e}); unset it or name a served id",
                             DOCS_URL) from e
    return served.id, f"GBENCH_AGENTDOJO_MODEL ({served.source})"


def _pct(xs: List[bool]) -> float:
    return round(sum(1 for x in xs if x) / len(xs) * 100.0, 2) if xs else 0.0


def _max_tokens_note(knob: Any = None) -> str:
    """The result's `max_output_tokens_note`: AgentDojo's OpenAILLM (chat_completion_request)
    sends no max_tokens, so neither does this suite."""
    return ("not sent: AgentDojo's OpenAILLM sends no max_tokens, so the server's default applies "
            "(the room left in the context window)"
            + (f"; --max-output-tokens {knob} not applied" if knob else ""))


#: Characters kept of each string in an episode's stored messages.
_TRACE_TEXT_CHARS = 2000


#: Seconds the OpenAI client may take to open its TCP connection to the endpoint
#: (model_client_timeout); the read, write and pool timeouts are the request timeout.
CONNECT_TIMEOUT_S = 10.0


def model_client_timeout(openai_version: Optional[str] = None) -> Dict[str, Any]:
    """The request timeout and retries of the OpenAI client gbench builds for AgentDojo's
    OpenAILLM, as the result records them (`model_client_timeout`).

    2026-10-10 investigation: the client was built with the OpenAI SDK's defaults, a 600 s read
    timeout and 2 retries, so an agent turn that needed more than 600 s of decode was dropped and
    asked again (3 x 600 s per attempt, and AgentDojo's chat_completion_request retries 3 times
    itself): a hidden cap on the turn's length. AgentDojo sends no max_tokens, so the timeout is
    base.request_timeout_s of the served context window (base.UNCAPPED_TIMEOUT_TOKENS when the
    server reports none), and the SDK retries are 0: AgentDojo's own tenacity retry (3 attempts,
    not on a 400/422) is the only one, as each of its attempts is one HTTP attempt.

    The TCP connect keeps a short bound of its own (CONNECT_TIMEOUT_S, `connect_timeout_s`): a
    bare float timeout applies to the connect too (httpx), so an unreachable endpoint would have
    held each attempt for the whole read timeout instead of the SDK default's 5 s (2026-10-11
    review: timeout=8 against a black-holed address waited 8.2 s in connect, httpx.Timeout(8,
    connect=2) failed in 2.0 s)."""
    tokens = int(get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)
    return {"client": "openai.OpenAI (AgentDojo OpenAILLM)",
            "version": openai_version,
            "timeout_s": request_timeout_s(tokens),
            "connect_timeout_s": CONNECT_TIMEOUT_S,
            "retries": 0,
            "outer_attempts": 3,
            "source": (f"OpenAI(timeout=httpx.Timeout(base.request_timeout_s(served context "
                       f"window={tokens}), connect={CONNECT_TIMEOUT_S:g}), max_retries=0); "
                       "AgentDojo's chat_completion_request retry (3 attempts) unchanged"),
            "replaces": "OpenAI SDK default 600 s x 3 attempts",
            "hidden_cap_removed": True}


def _clip(value: Any) -> Any:
    """`value` with every string longer than _TRACE_TEXT_CHARS clipped (for a trace)."""
    if isinstance(value, str):
        return value if len(value) <= _TRACE_TEXT_CHARS else (
            value[:_TRACE_TEXT_CHARS] + f"... [{len(value) - _TRACE_TEXT_CHARS} more chars]")
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v) for v in value]
    return value


def _is_context_length(exc: BaseException) -> bool:
    """The request errors AgentDojo's benchmark functions catch and score utility False /
    security True (benchmark.py: a 400 context_length_exceeded or 'reduce the length of the
    messages', a 422 naming max_new_tokens)."""
    text = repr(exc)
    return (getattr(exc, "code", None) == "context_length_exceeded"
            or getattr(exc, "param", None) == "max_tokens"
            or "reduce the length of the messages." in text or "max_new_tokens" in str(exc))


def _harness_kind(exc: BaseException) -> str:
    """The harness-error kind of an exception that ended an episode."""
    name = type(exc).__name__
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout",
                "ReadTimeout", "RemoteProtocolError"):
        return "model_endpoint_unreachable"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return f"model_endpoint_http_{status}"
    if name == "RetryError":
        return "model_endpoint_retries_exhausted"
    return f"exception_{name}"


def _send_temperature(client: Any, temperature: float,
                      enable_thinking: Optional[bool] = None,
                      sent: Optional[Dict[str, int]] = None,
                      max_tokens: Optional[int] = None) -> None:
    """Put `temperature` on every request of the OpenAI client's chat.completions.create, and,
    when `enable_thinking` is given, the chat template's thinking switch
    (`extra_body.chat_template_kwargs.enable_thinking`, merged into any extra_body the caller
    sent). `sent`, when given, counts the requests by the switch they carried.

    AgentDojo's OpenAILLM sends `temperature or NOT_GIVEN`, so a 0.0 would reach the server as no
    temperature at all; this wrapper makes every request carry the run's value. It sends no
    thinking switch either (chat_completion_request passes reasoning_effort only), so until
    2026-10-11 a --thinking run sampled at 1.0 with the chat template's thinking off: the switch
    never reached the wire."""
    completions = getattr(getattr(client, "chat", None), "completions", None)
    create = getattr(completions, "create", None)
    if create is None:
        return

    def _create(*args, **kwargs):
        kwargs["temperature"] = temperature
        # the run's --max-output-tokens when the request carries none (OpenAILLM sends none;
        # swe_agentic.agent_max_tokens_for_suite)
        if max_tokens and kwargs.get("max_tokens") is None \
                and kwargs.get("max_completion_tokens") is None:
            kwargs["max_tokens"] = int(max_tokens)
        if enable_thinking is not None:
            extra = dict(kwargs.get("extra_body") or {})
            extra["chat_template_kwargs"] = {**(extra.get("chat_template_kwargs") or {}),
                                             "enable_thinking": bool(enable_thinking)}
            kwargs["extra_body"] = extra
        if sent is not None:
            key = str(((kwargs.get("extra_body") or {}).get("chat_template_kwargs") or {})
                      .get("enable_thinking"))
            sent[key] = sent.get(key, 0) + 1
        return create(*args, **kwargs)
    completions.create = _create


class _EndpointWatch:
    """What the served endpoint returned during the current episode: wraps the OpenAI client's
    chat.completions.create (AgentDojo's OpenAILLM calls it) and counts, per episode, the
    requests, the replies that stopped at the end of the context window (finish_reason length:
    no max_tokens is sent, so that is the window), the empty replies (no content and no tool
    call; with a reasoning field, empty after reasoning) and the request errors (a retried one
    included). The score does not change.

    A reply empty after reasoning that stopped on its own is split by its generated token ids
    (swebench_common.classify_empty_reply): no tool-call token, the answer inside the thinking
    channel (`answer_in_thinking`, the model's outcome); the served tokenizer's tool-call token
    (Gemma 4 `<|tool_call>`), a native tool call the server's tool-call parser returned neither
    as a tool call nor as content (`tool_call_dropped`, a server-parser artifact); no token ids
    (`empty_unclassified`), never counted as the model's answer. Once a reply shows the endpoint
    is vLLM (its `prompt_token_ids` key), each request also asks for `return_token_ids` (a vLLM
    response field; sampling unchanged, AgentDojo reads only content and tool calls)."""

    def __init__(self) -> None:
        self.episode: Optional[Dict[str, Any]] = None
        #: whether the endpoint is vLLM, None until a reply shows it
        self.vllm: Optional[bool] = None
        self._server: Optional[str] = None
        self._key: Optional[str] = None

    def install(self, client: Any) -> None:
        completions = getattr(getattr(client, "chat", None), "completions", None)
        create = getattr(completions, "create", None)
        if create is None:
            return
        watch = self
        self._server = str(getattr(client, "base_url", "") or "") or None
        self._key = getattr(client, "api_key", None)

        def _create(*args, **kwargs):
            ep = watch.episode if watch.episode is not None else {}
            ep["requests"] = ep.get("requests", 0) + 1
            if watch.vllm and not kwargs.get("stream"):
                extra = dict(kwargs.get("extra_body") or {})
                extra.setdefault("return_token_ids", True)
                kwargs["extra_body"] = extra
            try:
                resp = create(*args, **kwargs)
            except Exception as e:                                      # noqa: BLE001
                ep.setdefault("request_errors", []).append(
                    f"{type(e).__name__}: {redact_text(str(e))[:200]}")
                if _is_context_length(e):
                    ep["context_length_error"] = True
                raise
            try:
                if watch.vllm is None:
                    watch.vllm = "prompt_token_ids" in (getattr(resp, "model_extra", None)
                                                        or {})
                choice = resp.choices[0]
                message = choice.message
                if choice.finish_reason == "length":
                    ep["replies_at_context_window"] = ep.get("replies_at_context_window", 0) + 1
                if not str(message.content or "").strip() and not message.tool_calls:
                    ep["empty_responses"] = ep.get("empty_responses", 0) + 1
                    if (getattr(message, "reasoning_content", None)
                            or getattr(message, "reasoning", None)):
                        ep["empty_after_reasoning"] = ep.get("empty_after_reasoning", 0) + 1
                        if choice.finish_reason != "length":
                            # stopped on its own: the answer inside the thinking channel (the
                            # model's outcome, base's ANSWER_IN_THINKING_POLICY), a tool call
                            # the server's parser dropped, or not told apart (no token ids)
                            field = watch.empty_class(getattr(choice, "token_ids", None))
                            ep[field] = ep.get(field, 0) + 1
            except Exception:                                           # noqa: BLE001
                pass
            return resp
        completions.create = _create

    def empty_class(self, token_ids: Any) -> str:
        """The episode counter of a reply empty after reasoning that stopped on its own:
        "answer_in_thinking", "tool_call_dropped" or "empty_unclassified"."""
        from .swebench_common import (EMPTY_ANSWER_IN_THINKING, EMPTY_TOOL_CALL_DROPPED,
                                      TOOL_CALL_TOKEN_TEXT, classify_empty_reply,
                                      served_token_id)
        tool_call_id = None
        if isinstance(token_ids, list) and token_ids and self._server:
            tool_call_id = served_token_id(self._server, TOOL_CALL_TOKEN_TEXT, self._key)
        kind = classify_empty_reply(token_ids, tool_call_id)
        if kind == EMPTY_ANSWER_IN_THINKING:
            return "answer_in_thinking"
        if kind == EMPTY_TOOL_CALL_DROPPED:
            return "tool_call_dropped"
        return "empty_unclassified"

    def run(self, fn) -> Dict[str, Any]:
        """One episode: its record, with `outcome` (fn's return value) or `harness_error` (the
        exception that ended it: the endpoint failed, or the harness did; never the model's
        outcome)."""
        rec: Dict[str, Any] = {"requests": 0}
        self.episode = rec
        try:
            rec["outcome"] = fn()
        except Exception as e:                                          # noqa: BLE001
            rec["harness_error"] = {"kind": _harness_kind(e),
                                    "detail": redact_text(f"{type(e).__name__}: {e}")[:300]}
        finally:
            self.episode = None
        return rec


def _episode_log(logdir: Path, pipeline_name: str, suite: str, task_id: str, attack_type: str,
                 injection_task: Optional[str]) -> Dict[str, Any]:
    """The messages and error AgentDojo's TraceLogger saved for one episode (clipped), or {}."""
    path = (logdir / str(pipeline_name).replace("/", "_") / suite / task_id / attack_type
            / f"{injection_task or 'none'}.json")
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {"messages": _clip(data.get("messages") or []), "agentdojo_error": data.get("error")}


def _episode_trace(key: str, kind: str, rec: Dict[str, Any], log: Dict[str, Any],
                   utility: Optional[bool], security: Optional[bool]) -> Dict[str, Any]:
    """The trace of one AgentDojo episode."""
    trace: Dict[str, Any] = {
        "sample_key": f"task_id:{key}" if kind == "benign" else f"{kind}:{key}",
        "category": key.split("/", 1)[0], "kind": kind,
        "is_correct": bool(utility),
        "utility": utility, "requests": rec.get("requests", 0),
        "status": "HARNESS_ERROR" if rec.get("harness_error") else "OK"}
    if kind == "under_attack":
        trace["attack_succeeded"] = security
    for field in ("replies_at_context_window", "empty_responses", "empty_after_reasoning",
                  "answer_in_thinking", "tool_call_dropped", "empty_unclassified",
                  "request_errors", "harness_error"):
        if rec.get(field):
            trace[field] = rec[field]
    if rec.get("context_length_exceeded"):
        trace["context_length_exceeded"] = True
    if rec.get("harness_error"):
        trace["scoring_excluded"] = True
        trace["is_correct"] = False
    trace.update(log)
    return trace


def _summarize(items: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
    """The metrics over the scored items (item_results): headline and per suite."""
    benign = list(items.get("benign", {}).values())
    under = list(items.get("under_attack", {}).values())
    success = list(items.get("attack_success", {}).values())

    def _by_suite(kind: str) -> Dict[str, List[bool]]:
        out: Dict[str, List[bool]] = {}
        for key, ok in items.get(kind, {}).items():
            out.setdefault(key.split("/", 1)[0], []).append(bool(ok))
        return out
    b, u, a = _by_suite("benign"), _by_suite("under_attack"), _by_suite("attack_success")
    return {
        "accuracy": _pct(benign) if benign else None,
        "utility_under_attack": _pct(under) if under else None,
        "attack_success_rate": _pct(success) if success else None,
        "total_questions": len(benign),
        "correct_answers": sum(1 for x in benign if x),
        "effective_n": len(benign),
        "category_accuracy": {name: {"correct": sum(1 for x in xs if x), "total": len(xs),
                                     "accuracy": _pct(xs)} for name, xs in sorted(b.items())},
        "suite_scores": {name: {"benign_utility": _pct(b.get(name, [])),
                                "utility_under_attack": _pct(u.get(name, [])) if u.get(name)
                                else None,
                                "attack_success_rate": _pct(a.get(name, [])) if a.get(name)
                                else None}
                         for name in sorted(set(b) | set(u))},
    }


def _apply_summary(result: Dict[str, Any], items: Dict[str, Dict[str, bool]],
                   harness_errors: Dict[str, int], counters: Dict[str, int],
                   config_reasons: List[str]) -> None:
    """Set the result's metrics, status and comparability from the scored items, the harness
    errors (by kind) and the endpoint counters; shared by a run and by merge_shards."""
    s = _summarize(items)
    result.update({k: s[k] for k in ("accuracy", "utility_under_attack", "attack_success_rate",
                                     "total_questions", "correct_answers", "effective_n",
                                     "category_accuracy")})
    scores = s["suite_scores"]
    for name, cell in scores.items():
        # the benign episodes of the suite in the score (n_user_tasks: the ones selected)
        cell["n_user_tasks_scored"] = result["category_accuracy"].get(name, {}).get("total", 0)
    result["dimension_scores"] = {**(result.get("dimension_scores") or {}), **{
        name: {**(result.get("dimension_scores") or {}).get(name, {}), **cell}
        for name, cell in scores.items()}}
    ran = sorted(result["category_accuracy"]) or sorted(scores)
    result["suites"] = ran
    result["headline_metrics"] = {"benign_utility": result["accuracy"],
                                  "utility_under_attack": result["utility_under_attack"],
                                  "attack_success_rate": result["attack_success_rate"]}
    result["metric"] = (
        "AgentDojo (delegated to the agentdojo package): headline accuracy = benign task utility "
        f"across the suites run ({', '.join(ran) or 'none'}); attack_success_rate = targeted "
        f"injection success under the '{result.get('attack')}' attack (lower is better); "
        "utility_under_attack also reported. security_results True == attack succeeded.")
    result["harness_errors"] = dict(sorted(harness_errors.items()))
    for field, value in counters.items():
        result[field] = value
    # replies that stopped with the answer inside the thinking channel, over every request of the
    # episodes (the model's outcome as AgentDojo's pipeline reads it; not a comparability reason)
    result.update(answer_in_thinking_fields(
        counters.get("answer_in_thinking", 0),
        sum(int(t.get("requests") or 0) for t in result.get("sample_traces") or [])))
    reasons = list(config_reasons)
    n_errors = sum(harness_errors.values())
    status_causes = []
    if n_errors:
        cause = (f"{n_errors} episode(s) ended by a harness error, left out of benign utility, "
                 "utility under attack and attack success ("
                 + ", ".join(f"{k} {v}" for k, v in sorted(harness_errors.items())) + ")")
        reasons.append(cause)
        status_causes.append(cause)
    if counters.get("tool_call_dropped"):
        reasons.append(f"{counters['tool_call_dropped']} repl(ies) whose generation held a native "
                       "tool call that the served endpoint's tool-call parser returned neither as "
                       "a tool call nor as content (a server-parser artifact AgentDojo saw as an "
                       "empty reply, not the model's answer; tool_call_dropped)")
    if counters.get("empty_unclassified"):
        reasons.append(f"{counters['empty_unclassified']} repl(ies) with reasoning and no content "
                       "or tool call without token ids to tell an answer inside the thinking "
                       "channel from a tool call the server's parser dropped (not counted in "
                       "answer_in_thinking; empty_unclassified)")
    if counters.get("context_length_exceeded"):
        reasons.append(f"{counters['context_length_exceeded']} episode(s) outgrew the served "
                       "context window and were scored as AgentDojo scores them (utility False, "
                       "attack not successful)")
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) if reasons else None
    result.pop("accuracy_withheld_reason", None)
    if result["accuracy"] is None:
        why = ("every benign episode ended by a harness error" if n_errors
               else "no benign episode was scored")
        result["status"] = "failed"
        result["status_reason"] = "; ".join([why] + status_causes)
        result["accuracy_withheld_reason"] = why
        result["error"] = f"agent_dojo: nothing was scored: {why}"
    elif status_causes:
        result["status"] = "completed_with_errors"
        result["status_reason"] = "; ".join(status_causes)
    else:
        result["status"] = "success"
        result["status_reason"] = None


#: The per-episode endpoint counters a result carries (summed over the episodes).
_COUNTERS = ("replies_at_context_window", "empty_responses", "empty_after_reasoning",
             "answer_in_thinking", "tool_call_dropped", "empty_unclassified",
             "context_length_exceeded")


def select_user_tasks(selected: List[Tuple[str, str]], limit: Optional[int]
                      ) -> List[Tuple[str, str]]:
    """The (suite, user task) pairs a run evaluates: `selected` (this shard's, in the package's
    order) unchanged without a limit; with --eval-limit N, N user tasks for the whole run, dealt
    over the suites (sampling.allocate_limit, seed "agent_dojo"; each suite capped at its user
    tasks), a seeded draw within each suite, in the package's order. A suite whose share is 0 is
    not run."""
    if not limit or int(limit) <= 0:
        return list(selected)
    by_suite: Dict[str, List[str]] = {}
    for name, uid in selected:
        by_suite.setdefault(name, []).append(uid)
    shares = allocate_limit(int(limit), list(by_suite),
                            available={n: len(v) for n, v in by_suite.items()}, seed="agent_dojo")
    keep = {(n, u) for n, uids in by_suite.items() if shares[n]
            for u in stratified_sample(uids, int(shares[n]), seed=f"agent_dojo:{n}")}
    return [p for p in selected if p in keep]


def injection_pairs(injection_ids: List[str], user_ids: List[str], limit: Optional[int],
                    suite: str) -> Dict[str, List[str]]:
    """{user task: the injection tasks it runs under}: every injection task of the suite in a
    full run (AgentDojo's cross-product); under --eval-limit ONE each, a seeded order of the
    suite's injection tasks dealt to the user tasks in turn, so a limited run of N user tasks is N
    attacked episodes, not N x the suite's injection tasks."""
    if not limit or int(limit) <= 0 or not injection_ids:
        return {uid: list(injection_ids) for uid in user_ids}
    order = sorted(injection_ids, key=lambda i: hashlib.sha256(
        f"agent_dojo:{suite}:{i}".encode("utf-8")).hexdigest())
    return {uid: [order[k % len(order)]] for k, uid in enumerate(user_ids)}


def _is_dos_attack(name: str) -> bool:
    """AgentDojo's DoS attacks (dos, swearwords_dos, captcha_dos, ...): run with one injection
    task and no injection-task-as-user check (run_task_with_injection_tasks)."""
    return name == "dos" or name.endswith("_dos")


def agent_dojo_plan(limit: Optional[int], n_user_tasks: int, per_suite_tasks: Dict[str, int],
                    by_suite: Dict[str, List[str]],
                    inj_by_suite: Dict[str, Dict[str, List[str]]],
                    attack: Optional[str] = None) -> Dict[str, Any]:
    """Log the run's planned workload (one line) and return its `eval_limit_plan`. A sample is a
    user task; each runs one benign episode and one attacked episode per injection task it is
    paired with, and every injection task used is run once more as a user task (AgentDojo's
    injection-task-as-user check; none for a DoS attack), so the line names all three terms and
    their total agent conversations."""
    dos = _is_dos_attack(attack or _ATTACK)
    episodes = 0
    checks = 0
    for name, uids in by_suite.items():
        inj_for = inj_by_suite.get(name) or {}
        if dos:
            used = sorted({i for uid in uids for i in inj_for.get(uid) or []})[:1]
            episodes += sum(1 + len(used) for _ in uids)
        else:
            episodes += sum(1 + len(inj_for.get(uid) or []) for uid in uids)
            checks += len({i for uid in uids for i in inj_for.get(uid) or []})
    total = episodes + checks
    if n_user_tasks and episodes % n_user_tasks == 0:
        attempts, base = episodes // n_user_tasks, "episodes (benign + attacked)"
    else:
        attempts, base = 1, f"user task(s) -> {episodes} episodes (benign + attacked)"
    label = f"{base} + {checks} injection-task-as-user check(s) = {total} conversations"
    plan = eval_limit_plan("agent_dojo", limit, n_user_tasks, per_suite_tasks,
                           setting_noun="suite", attempts=attempts, unit=label)
    # the record keeps plain keys (the logged line carries the prose)
    plan.pop(label, None)
    plan["episodes"] = episodes
    plan["injection_task_checks"] = checks
    plan["conversations"] = total
    return plan


def run_agent_dojo(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical AgentDojo by delegating to the agentdojo package's suites/attacks/scorers."""
    A = _load_agentdojo()
    limit = limit if limit is not None else kwargs.get("limit")

    model_id, model_id_source = _served_model_id(model_name, base_url)
    temperature, _src = resolve_official_sampling("agent_dojo", AGENTDOJO_SAMPLING,
                                                  thinking=enable_thinking)
    # OpenAILLM sends `temperature or NOT_GIVEN`, which drops a 0.0: _send_temperature (below)
    # puts the run's temperature on every request, so the requests carry exactly this value.
    sent_temperature = float(temperature)
    official = AGENTDOJO_SAMPLING.record(sent_temperature, _src)
    # The endpoint's key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY), sent from
    # gbench's own process; AgentDojo's tools are in-memory Python functions over the task's
    # environment state, with no shell, file or environment access.
    # timeout / max_retries: no hidden 600 s SDK default (model_client_timeout)
    client_timeout = model_client_timeout(getattr(A["openai"], "__version__", None))
    import httpx
    client = A["openai"].OpenAI(api_key=endpoint_api_key() or "EMPTY", base_url=base_url,
                                timeout=httpx.Timeout(client_timeout["timeout_s"],
                                                      connect=client_timeout["connect_timeout_s"]),
                                max_retries=client_timeout["retries"])
    watch = _EndpointWatch()
    watch.install(client)
    # the run's temperature and thinking switch on every request (_send_temperature)
    thinking_sent: Dict[str, int] = {}
    from .swe_agentic import agent_max_tokens_for_suite
    agent_max_tokens = agent_max_tokens_for_suite("agent_dojo")
    _send_temperature(client, sent_temperature, bool(enable_thinking), thinking_sent,
                      max_tokens=agent_max_tokens)
    # Use the NATIVE tools API (AgentDojo's `vllm_parsed` provider). The endpoint must have
    # vLLM tool-calling enabled (--enable-auto-tool-choice + a tool-call parser: gbench's own
    # server gets them from --tool-call-parser), which returns proper `tool_calls`; the
    # prompt-based LocalLLM would get content=None because the server's parser consumes the
    # model's function-call text.
    llm = A["OpenAILLM"](client, model_id, temperature=sent_temperature)
    # Canonical no-defense pipeline (identical to AgentPipeline.from_config with defense=None).
    pipeline = A["AgentPipeline"]([
        A["SystemMessage"](A["load_system_message"](None)),
        A["InitQuery"](),
        llm,
        A["ToolsExecutionLoop"]([A["ToolsExecutor"](), llm]),
    ])
    # AgentDojo derives the attack's prose model label from pipeline.name (MODEL_NAMES); use the
    # canonical local-model label so load_attack('important_instructions') resolves.
    pipeline.name = _MODEL_LABEL

    try:
        suites = A["get_suites"](_BENCHMARK_VERSION)
    except Exception as e:                                              # noqa: BLE001
        raise infra_required(
            "agent_dojo",
            f"agentdojo has no benchmark version {_BENCHMARK_VERSION!r} ({e}); set "
            "GBENCH_AGENTDOJO_VERSION to one the installed package provides.", DOCS_URL) from e

    # The unit of the benchmark is a user task of a suite: every (suite, user task) in the
    # package's own order (the suites in get_suites order, each suite's user tasks in
    # registration order). --shard selects this node's share of that full list first
    # (base.apply_shard, round-robin), then `limit` (--eval-limit) is N user tasks for the whole
    # run (select_user_tasks: dealt over the suites, a seeded draw within each), never N per
    # suite. Each selected user task runs without injections and under attack: with every
    # injection task of its suite in a full run, with ONE injection task of its suite (a seeded
    # pick, injection_pairs) under --eval-limit, so a limited run is N benign + N attacked
    # episodes (plus AgentDojo's injection-task-as-user check for the injection tasks used).
    ordered = [(name, uid) for name, suite in suites.items() for uid in suite.user_tasks]
    selected, shard_label = apply_shard(ordered, "agent_dojo")
    selected = select_user_tasks(selected, limit)
    by_suite: Dict[str, List[str]] = {}
    for name, uid in selected:
        by_suite.setdefault(name, []).append(uid)
    inj_by_suite = {name: injection_pairs(list(suite.injection_tasks), by_suite.get(name) or [],
                                          limit, name)
                    for name, suite in suites.items()}
    per_suite_tasks = {name: len(by_suite.get(name) or []) for name in suites}
    plan = agent_dojo_plan(limit, len(selected), per_suite_tasks, by_suite, inj_by_suite)

    # AgentDojo writes each episode's log under this directory; the logs are read into the
    # traces and the directory is removed when the run ends.
    logdir = Path(tempfile.mkdtemp(prefix="gbench_agentdojo_"))
    items: Dict[str, Dict[str, bool]] = {"benign": {}, "under_attack": {}, "attack_success": {}}
    item_harness_errors: Dict[str, Dict[str, Any]] = {}
    harness_errors: Dict[str, int] = {}
    counters = dict.fromkeys(_COUNTERS, 0)
    per_suite: Dict[str, Any] = {}
    traces: List[Dict[str, Any]] = []
    injection_as_user: Dict[str, Optional[bool]] = {}

    def _record(key: str, kind: str, rec: Dict[str, Any], log: Dict[str, Any],
                utility: Optional[bool], security: Optional[bool]) -> None:
        if rec.get("context_length_error") and not rec.get("harness_error"):
            # AgentDojo's own rule scored this episode (utility False, security True)
            rec["context_length_exceeded"] = True
        for field in _COUNTERS:
            counters[field] += int(rec.get(field) or 0) if field != "context_length_exceeded" \
                else int(bool(rec.get(field)) and kind != "injection_task_as_user")
        if rec.get("harness_error"):
            kind_ = rec["harness_error"]["kind"]
            if kind != "injection_task_as_user":
                harness_errors[kind_] = harness_errors.get(kind_, 0) + 1
                item_harness_errors[f"{kind}:{key}"] = rec["harness_error"]
        elif kind == "benign":
            items["benign"][key] = bool(utility)
        elif kind == "under_attack":
            items["under_attack"][key] = bool(utility)
            items["attack_success"][key] = bool(security)
        trace = _episode_trace(key, kind, rec, log, utility, security)
        trace["global_idx"] = len(traces)
        traces.append(trace)

    # AgentDojo's benchmark functions require an active logging context (they read
    # `logger.logdir`); without it TraceLogger raises AttributeError on the NullLogger.
    # AgentDojo ALSO dumps every task's full conversation transcript to the ROOT logger at
    # INFO (hundreds of lines per run). Raise the root level to WARNING for the duration so
    # that spam is suppressed while genuine warnings/errors still surface, then restore it.
    _root = logging.getLogger()
    _prev_level = _root.level
    _root.setLevel(logging.WARNING)
    try:
        with A["OutputLogger"](str(logdir)):
            for name, suite in suites.items():
                user_ids = by_suite.get(name) or []
                if not user_ids:
                    continue
                inj_for = inj_by_suite[name]
                inj_ids = [i for i in suite.injection_tasks
                           if any(i in v for v in inj_for.values())]
                attack = A["load_attack"](_ATTACK, suite, pipeline)
                attack_type = getattr(attack, "name", None) or _ATTACK
                if getattr(attack, "is_dos_attack", False):
                    # run_task_with_injection_tasks runs a DoS attack with one injection task only
                    inj_ids = inj_ids[:1]
                    inj_for = {uid: inj_ids for uid in user_ids}
                # As benchmark_suite_with_injections does (not for a DoS attack): each injection
                # task run once as a user task, a check reported apart from every number.
                if not getattr(attack, "is_dos_attack", False):
                    for inj in inj_ids:
                        rec = watch.run(lambda: A["run_without"](
                            suite, pipeline, suite.get_injection_task_by_id(inj), logdir, True,
                            _BENCHMARK_VERSION))
                        ok = (bool(rec["outcome"][0]) if "outcome" in rec else None)
                        injection_as_user[f"{name}/{inj}"] = ok
                        _record(f"{name}/{inj}", "injection_task_as_user", rec,
                                _episode_log(logdir, pipeline.name, name, inj, "none", None),
                                ok, None)
                # Each episode on its own, as benchmark_suite_without_injections and
                # run_task_with_injection_tasks run them one after another: an episode an
                # endpoint failure ends is a harness error, and the others still count.
                for uid in user_ids:
                    rec = watch.run(lambda: A["run_without"](
                        suite, pipeline, suite.get_user_task_by_id(uid), logdir, True,
                        _BENCHMARK_VERSION))
                    utility = bool(rec["outcome"][0]) if "outcome" in rec else None
                    _record(_item_key(name, uid, ""), "benign", rec,
                            _episode_log(logdir, pipeline.name, name, uid, "none", None),
                            utility, None)
                    for inj in inj_for[uid]:
                        rec = watch.run(lambda: A["run_with"](
                            suite, pipeline, suite.get_user_task_by_id(uid), attack, logdir, True,
                            [inj], _BENCHMARK_VERSION))
                        utility = security = None
                        if "outcome" in rec:
                            util_map, sec_map = rec["outcome"]
                            utility = bool(util_map.get((uid, inj)))
                            security = bool(sec_map.get((uid, inj)))  # True == attack succeeded
                        _record(_item_key(name, uid, inj), "under_attack", rec,
                                _episode_log(logdir, pipeline.name, name, uid, attack_type, inj),
                                utility, security)
                per_suite[name] = {"n_user_tasks": len(user_ids),
                                   "n_injection_tasks": len(inj_ids)}
    finally:
        _root.setLevel(_prev_level)
        shutil.rmtree(logdir, ignore_errors=True)

    if not traces and not (shard_label and not selected):
        raise RuntimeError("agent_dojo: no tasks were scored (empty suites/subset)")

    installed = A.get("version")
    noncanon = []
    if limit:
        noncanon.append(f"--eval-limit {limit}: {len(selected)} user task(s) for the run ("
                        + ", ".join(f"{n} {k}" for n, k in per_suite_tasks.items() if k)
                        + "), each under ONE injection task of its suite instead of all of them")
    if not official["temperature_matches_official"]:
        noncanon.append(f"temperature {sent_temperature} set by {_src} (the vllm_parsed provider "
                        "sends none)")
    if installed is not None and installed != AGENTDOJO_VERSION:
        noncanon.append(f"agentdojo {installed} installed (pinned {AGENTDOJO_VERSION})")
    if _BENCHMARK_VERSION != _PINNED_BENCHMARK_VERSION or _ATTACK != _PINNED_ATTACK:
        noncanon.append(f"benchmark {_BENCHMARK_VERSION} / attack {_ATTACK} overridden "
                        f"(pinned {_PINNED_BENCHMARK_VERSION} / {_PINNED_ATTACK})")
    if _MODEL_LABEL != _PINNED_MODEL_LABEL:
        noncanon.append(f"attack model label {_MODEL_LABEL!r} overridden (pinned "
                        f"{_PINNED_MODEL_LABEL!r}, 'Local model'); the injected text differs")

    result = {
        "benchmark_type": "eval",
        "eval_name": "agent_dojo",
        "model_name": model_name,
        # the model id AgentDojo's requests carry (the eval runner stamps served_model_id)
        "request_model_id": model_id,
        "request_model_id_source": model_id_source,
        "dimension_scores": per_suite,
        "benchmark_version": _BENCHMARK_VERSION,
        "attack": _ATTACK,
        "attack_model_label": _MODEL_LABEL,
        "suites_available": list(suites.keys()),
        "temperature": sent_temperature,
        "temperature_source": _src,
        "official_sampling": official,
        "max_output_tokens_note": (
            f"sent: max_tokens={agent_max_tokens} (the run's --max-output-tokens) on every "
            "request that carries none; AgentDojo's OpenAILLM sends none"
            if agent_max_tokens else _max_tokens_note(kwargs.get("max_output_tokens"))),
        "max_tokens_sent": agent_max_tokens,
        # `temperature` is what every request carries.
        "sampling": ("gbench's rule for every suite: temperature 1.0 with --thinking, 0.0 "
                     "without (--temperature or GBENCH_AGENT_DOJO_TEMPERATURE replaces it), put "
                     "on every request by gbench, a 0.0 included (AgentDojo's own OpenAILLM "
                     "sends `temperature or NOT_GIVEN`, which drops a 0.0), and the chat "
                     "template's thinking switch (extra_body.chat_template_kwargs."
                     f"enable_thinking={str(bool(enable_thinking)).lower()}), which AgentDojo's "
                     "OpenAI tools path does not send itself."),
        # the run's thinking switch and the requests that carried it (every request does)
        "thinking": bool(enable_thinking),
        "thinking_applied": (sum(thinking_sent.values()) > 0
                             and thinking_sent.get(str(bool(enable_thinking)), 0)
                             == sum(thinking_sent.values())),
        "request_shape": {"field": "extra_body.chat_template_kwargs.enable_thinking",
                          "requests_by_value": dict(sorted(thinking_sent.items()))},
        # the OpenAI client's request timeout and retries (no 600 s SDK default)
        "model_client_timeout": client_timeout,
        "agentdojo_version": installed,
        # AgentDojo's benchmark functions run the tasks one at a time in this process; the
        # suite's concurrency (--batch-sizes / --sandboxes) is not used.
        "execution": "sequential (AgentDojo's benchmark functions run one task at a time)",
        # Every scored item (benign utility, utility under attack, attack success), keyed
        # "<suite>/<user task>/<injection task>", for merging shards (merge_shards); the
        # episodes a harness error ended, keyed "<kind>:<item>", with the error.
        "item_results": items,
        "item_harness_errors": item_harness_errors,
        # AgentDojo's check that each injection task is solvable as a user task (reported apart
        # from every number; None where a harness error ended it)
        "injection_tasks_as_user_tasks": injection_as_user,
        "comparability_settings": noncanon,
        # Every episode (benign, under attack, injection task as a user task) with its verdict,
        # endpoint counters and AgentDojo's logged messages (strings clipped to
        # _TRACE_TEXT_CHARS)
        "sample_traces": traces,
        "eval_limit_plan": plan,
    }
    _apply_summary(result, items, harness_errors, counters, noncanon)
    attempts = int(kwargs.get("attempt_count") or 1)
    if attempts > 1:
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "agent_dojo: AgentDojo runs each task once and scores it against its "
                           "environment's ground truth"}
    record_shard(result, shard_label)
    if shard_label and not selected:
        reason = (f"empty shard {shard_label}: no items on this node ({len(ordered)} in the full "
                  "set)")
        result.update(accuracy=None, utility_under_attack=None, attack_success_rate=None,
                      accuracy_withheld_reason=reason, status_reason=reason, empty_shard=True,
                      full_set_items=len(ordered), status="success")
        result.pop("error", None)
        add_comparability_reason(result, reason)
    return result


def _item_key(suite: str, user_task: str, injection_task: str) -> str:
    """The id of one AgentDojo result: "<suite>/<user task>/<injection task>" ("" for none)."""
    return f"{suite}/{user_task}/{injection_task}"


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of the shards of one agent_dojo run (gbench --merge): every metric
    recomputed over the union of the shards' item_results, the per-suite numbers, the harness
    errors and the endpoint counters too; the traces are the union of the shards' episodes; the
    shard comparability reasons are dropped (the union is the full set when every shard is
    there)."""
    union: Dict[str, Dict[str, bool]] = {"benign": {}, "under_attack": {}, "attack_success": {}}
    errors: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for kind, rows in (r.get("item_results") or {}).items():
            union.setdefault(kind, {}).update(rows or {})
        errors.update(r.get("item_harness_errors") or {})
    live = [r for r in results if not r.get("empty_shard")] or list(results)
    merged = dict(live[0])
    for field in ("error", "empty_shard", "full_set_items", "accuracy_withheld_reason"):
        merged.pop(field, None)
    traces: List[Dict[str, Any]] = []
    seen = set()
    for r in results:
        for t in r.get("sample_traces") or []:
            if t.get("sample_key") in seen:
                continue                     # an injection task run as a user task on 2 shards
            seen.add(t.get("sample_key"))
            traces.append(dict(t, global_idx=len(traces)))
    merged["sample_traces"] = traces
    merged["item_results"] = union
    merged["item_harness_errors"] = errors
    harness: Dict[str, int] = {}
    for err in errors.values():
        harness[err.get("kind", "unknown")] = harness.get(err.get("kind", "unknown"), 0) + 1
    counters = {f: sum(int(r.get(f) or 0) for r in results) for f in _COUNTERS}
    merged["injection_tasks_as_user_tasks"] = {
        k: v for r in results for k, v in (r.get("injection_tasks_as_user_tasks") or {}).items()}
    dims: Dict[str, Any] = {}
    for r in results:
        for name, cell in (r.get("dimension_scores") or {}).items():
            d = dims.setdefault(name, {"n_user_tasks": 0})
            d["n_user_tasks"] += int(cell.get("n_user_tasks") or 0)
            if "n_injection_tasks" in cell:
                d["n_injection_tasks"] = max(int(d.get("n_injection_tasks") or 0),
                                             int(cell["n_injection_tasks"]))
    merged["dimension_scores"] = dims
    config_reasons: List[str] = []
    for r in live:
        for reason in r.get("comparability_settings") or []:
            if reason not in config_reasons:
                config_reasons.append(reason)
    merged["comparability_settings"] = config_reasons
    _apply_summary(merged, union, harness, counters, config_reasons)
    # the shards' summed plan, not the first shard's
    from .limit_plan import merge_plans
    plan = merge_plans(results)
    if plan is not None:
        merged["eval_limit_plan"] = plan
    return merged
