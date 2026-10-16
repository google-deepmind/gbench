#!/usr/bin/env python3
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

# In-container launcher for MobileGym (`mobilegym`, 256-task test split) and MobileGym++
# (`mobilegym_pp`, 215-task bench215/official.json benchmark), run INSIDE the
# `gbench-mobilegym:5fbd3f34` image.

from __future__ import annotations

import argparse
import asyncio
import collections
import contextvars
import dataclasses
import datetime
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

SIMULATED_TIME = "2026-07-12 09:00:00"
SIMULATED_DATE = datetime.date(2026, 7, 12)
SIMULATOR_PORT = int(os.environ.get("GBENCH_MOBILEGYM_SIMULATOR_PORT", "3000"))
SIMULATOR_URL = (
    f"http://127.0.0.1:{SIMULATOR_PORT}/"
    "?mobilegym_simulated_time=2026-07-12+09%3A00%3A00&mobilegym_time_flowing=false"
)

_CURRENT_TASK_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_CURRENT_TASK_ID", default=None
)
_LOCK = threading.Lock()
_TASK_CALL_STATS: Dict[str, Dict[str, Any]] = collections.defaultdict(
    lambda: {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "endpoint_failures": [],
        "content_rejections": [],
        "server_parser_moved_text": 0,
        "server_parser_dropped_text": 0,
        "server_parser_text_lost": 0,
        "stopped_in_reasoning": 0,
    }
)


def _stable_key(key):
    if key is None:
        return "<none>"
    try:
        hash(key)
    except TypeError:
        return repr(key)
    return key


def stratified_indices(keys, limit, seed=None):
    """gbench's sampling.stratified_indices, copied (this launcher runs in its image without
    gbench; a test holds the copy equal): the ascending indices of `limit` items, round-robin
    across the categories `keys` gives (seeded order of categories, seeded order within each), so
    a limited run samples every category before any twice; every index when `limit` is None,
    not positive or at least len(keys)."""
    total = len(keys)
    if limit is None or limit <= 0 or limit >= total:
        return list(range(total))
    digest = hashlib.sha256((seed or "gbench").encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(_stable_key(key), []).append(index)
    for indices in groups.values():
        rng.shuffle(indices)
    order = list(groups)
    rng.shuffle(order)
    chosen = []
    while len(chosen) < limit:
        progressed = False
        for key in order:
            bucket = groups[key]
            if not bucket:
                continue
            chosen.append(bucket.pop())
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return sorted(chosen)


# ---------------------------------------------------------------------------
# The server-parser policy: gbench's base.recover_server_parser_text (the four pure functions are
# copied from gbench/runners/eval_suites/base.py; test_mobilegym.py holds them equal).
# ---------------------------------------------------------------------------
SERVER_PARSER_MOVED_TEXT = "server_parser_moved_text"
SERVER_PARSER_DROPPED_TEXT = "server_parser_dropped_text"
SERVER_PARSER_TEXT_LOST = "server_parser_text_lost"
SERVER_PARSER_EVENTS = (
    SERVER_PARSER_MOVED_TEXT,
    SERVER_PARSER_DROPPED_TEXT,
    SERVER_PARSER_TEXT_LOST,
)
DETOKENIZE_URL = os.environ.get("GBENCH_DETOKENIZE_URL", "").strip()
RECOVERED_TEXT_CAP = 4000


class ParserRecovery(NamedTuple):
    event: Optional[str]
    text: Optional[str]
    recovered_from: Optional[str]


def server_parser_signal(
    *,
    content: Optional[str],
    reasoning: Optional[str],
    tool_calls: Any,
    thinking: bool,
    completion_tokens: Optional[int],
) -> Optional[str]:
    if tool_calls:
        return None
    if not thinking and str(reasoning or "").strip():
        return SERVER_PARSER_MOVED_TEXT
    if (
        not str(content or "").strip()
        and not str(reasoning or "").strip()
        and int(completion_tokens or 0) > 0
    ):
        return SERVER_PARSER_DROPPED_TEXT
    return None


def generation_text_from_token_ids(
    token_ids: Any,
    decode: Optional[Callable[[List[int]], Any]],
    finish_reason: Optional[str] = None,
    stop_reason: Any = None,
) -> Optional[str]:
    if decode is None or not isinstance(token_ids, (list, tuple)) or not token_ids:
        return None
    try:
        ids = [int(t) for t in token_ids]
    except (TypeError, ValueError):
        return None
    if finish_reason == "stop" and not isinstance(stop_reason, str):
        ids = ids[:-1]
    if not ids:
        return ""
    try:
        text = decode(ids)
    except Exception:
        return None
    if not isinstance(text, str):
        return None
    if isinstance(stop_reason, str) and stop_reason:
        cut = text.find(stop_reason)
        if cut >= 0:
            text = text[:cut]
    return text


def recover_server_parser_text(
    *,
    content: Optional[str],
    reasoning: Optional[str],
    tool_calls: Any,
    thinking: bool,
    completion_tokens: Optional[int],
    generation_text: Optional[str] = None,
) -> ParserRecovery:
    signal = server_parser_signal(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        thinking=thinking,
        completion_tokens=completion_tokens,
    )
    if signal is None:
        return ParserRecovery(None, None, None)
    have_generation = isinstance(generation_text, str) and bool(generation_text.strip())
    if signal == SERVER_PARSER_MOVED_TEXT:
        if have_generation:
            return ParserRecovery(SERVER_PARSER_MOVED_TEXT, generation_text, "token_ids")
        return ParserRecovery(
            SERVER_PARSER_MOVED_TEXT,
            str(reasoning or "") + str(content or ""),
            "reasoning_then_content",
        )
    if have_generation:
        return ParserRecovery(SERVER_PARSER_DROPPED_TEXT, generation_text, "token_ids")
    if isinstance(generation_text, str):
        return ParserRecovery(None, None, None)
    return ParserRecovery(SERVER_PARSER_TEXT_LOST, None, None)


_PARSER_MARKUP_RE = re.compile(r"<[^<>\s]{1,64}>")


def server_parser_residue(
    *,
    content: Optional[str],
    reasoning: Optional[str],
    tool_calls: Any,
    thinking: bool,
    generation_text: Optional[str],
) -> ParserRecovery:
    if tool_calls or not isinstance(generation_text, str) or not generation_text.strip():
        return ParserRecovery(None, None, None)
    gen = generation_text
    r = str(reasoning or "").strip()
    c = str(content or "").strip()
    if not r and not c:
        return ParserRecovery(None, None, None)
    pos = 0
    reasoning_end = 0
    if r:
        at = gen.find(r)
        if at < 0:
            return ParserRecovery(None, None, None)
        pos = reasoning_end = at + len(r)
    if c:
        at = gen.find(c, pos)
        if at < 0:
            return ParserRecovery(None, None, None)
        pos = at + len(c)
    tail = gen[pos:]
    visible = _PARSER_MARKUP_RE.sub("", tail)
    if not any(ch.isalnum() for ch in visible):
        return ParserRecovery(None, None, None)
    if thinking and r:
        return ParserRecovery(SERVER_PARSER_DROPPED_TEXT, gen[reasoning_end:], "token_ids")
    return ParserRecovery(SERVER_PARSER_DROPPED_TEXT, gen, "token_ids")


def _detokenize_via_relay(token_ids: List[int]) -> Optional[str]:
    if not DETOKENIZE_URL or not token_ids:
        return None
    body = json.dumps({"tokens": list(token_ids)}).encode("utf-8")
    req = urllib.request.Request(
        DETOKENIZE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = data.get("text") if "text" in data else data.get("prompt")
    return out if isinstance(out, str) else None


def apply_server_parser_policy(
    raw: Any,
    *,
    thinking: bool,
    text: str = "",
    reasoning: str = "",
    tool_calls: Any = None,
    expected_tool_calls: bool = False,
    detokenize=_detokenize_via_relay,
) -> Dict[str, Any]:
    choice = ((raw.get("choices") or [None])[0] if isinstance(raw, dict) else None) or {}
    msg = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = raw.get("usage") if isinstance(raw, dict) and isinstance(raw.get("usage"), dict) else {}
    body = str(text if text else (msg.get("content") or ""))
    thought = str(
        reasoning if reasoning else (msg.get("reasoning") or msg.get("reasoning_content") or "")
    )
    tcs = tool_calls if tool_calls is not None else msg.get("tool_calls")
    comp = int(usage.get("completion_tokens") or 0)
    raw_ids = choice.get("token_ids") if isinstance(choice, dict) else None
    finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    stop_reason = choice.get("stop_reason") if isinstance(choice, dict) else None

    decoded = generation_text_from_token_ids(
        raw_ids, detokenize, finish_reason=finish, stop_reason=stop_reason
    )
    rec = recover_server_parser_text(
        content=body,
        reasoning=thought,
        tool_calls=tcs if expected_tool_calls else None,
        thinking=thinking,
        completion_tokens=comp,
        generation_text=decoded,
    )
    if rec.event is None and decoded and not expected_tool_calls:
        res = server_parser_residue(
            content=body,
            reasoning=thought,
            tool_calls=tcs if expected_tool_calls else None,
            thinking=thinking,
            generation_text=decoded,
        )
        if res.event is not None:
            rec = res
    return {
        "text": rec.text if rec.text is not None else body,
        "reasoning": "" if rec.event == SERVER_PARSER_MOVED_TEXT else thought,
        "event": rec.event,
        "recovered_from": rec.recovered_from,
        "generation_text": decoded,
        "finish_reason": finish,
    }


def _classify_llm_exception(exc: BaseException) -> Tuple[bool, Optional[int], str]:
    """Return `(is_harness_error, status_code, message)` for an LLM exception.

    HTTP 400 and 422 from the model endpoint are content/context rejections (scored as 0.0
    model outcomes); HTTP 401/403/404/408/429/5xx, connection errors, and timeouts are
    endpoint/infrastructure failures (`is_harness_error=True`, excluded from denominator).
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    msg = f"{type(exc).__name__}: {exc}"
    if isinstance(status, int):
        if status in (400, 422):
            return False, status, msg
        return True, status, msg
    low = msg.lower()
    if "400" in low or "422" in low or "maximum context length" in low or "context_length_exceeded" in low:
        return False, 400, msg
    return True, None, msg


def _install_common_patches(
    *,
    suite: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> None:
    """Install transport, parser, clock, and Chromium stability patches in-process."""
    # 1. Freeze Python default_tomorrow() and calendar today() to match SIMULATED_DATE (2026-07-12 -> 2026-07-13).
    import bench_env.task.utils as task_utils

    def _frozen_tomorrow() -> str:
        return (SIMULATED_DATE + datetime.timedelta(days=1)).isoformat()

    task_utils.default_tomorrow = _frozen_tomorrow
    try:
        import bench_env.task.calendar.tasks as cal_tasks

        cal_tasks.default_tomorrow = _frozen_tomorrow

        class _FrozenDateProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(datetime.date, name)

            @staticmethod
            def today() -> datetime.date:
                return SIMULATED_DATE

        class _FrozenDatetimeModuleProxy:
            date = _FrozenDateProxy()

            def __getattr__(self, name: str) -> Any:
                return getattr(datetime, name)

        cal_tasks.datetime = _FrozenDatetimeModuleProxy()  # type: ignore[assignment]
    except Exception:
        pass

    # 2. Harden MobileGymEnv Chromium flags, dialog auto-accept, and deterministic step sleep.
    from bench_env.env.mobile_gym import MobileGymEnv

    orig_get_launch_args = MobileGymEnv.get_launch_args

    @staticmethod
    def _patched_get_launch_args(headless: bool = True, *args: Any, **kwargs: Any) -> Any:
        res = orig_get_launch_args(headless, *args, **kwargs)
        extra_flags = (
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1, EXCLUDE localhost",
        )
        if isinstance(res, dict):
            flag_list = list(res.get("args") or [])
            for flag in extra_flags:
                if flag not in flag_list:
                    flag_list.append(flag)
            res["args"] = flag_list
            return res
        flag_list = list(res)
        for flag in extra_flags:
            if flag not in flag_list:
                flag_list.append(flag)
        return flag_list

    MobileGymEnv.get_launch_args = _patched_get_launch_args

    orig_attach = MobileGymEnv._attach_page_listeners

    def _patched_attach(self: Any) -> None:
        orig_attach(self)
        if getattr(self, "_page", None) is not None:
            def _accept_dialog(dialog: Any) -> None:
                try:
                    asyncio.ensure_future(dialog.accept())
                except Exception:
                    pass
            self._page.on("dialog", _accept_dialog)

    MobileGymEnv._attach_page_listeners = _patched_attach

    # Force deterministic post-action delay (0.8s instead of delay * random.uniform(0.5, 1.125)).
    import bench_env.env.mobile_gym as mg_mod
    orig_reset = MobileGymEnv.reset

    async def _patched_reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.delay_after_action = 0.8
        return await orig_reset(self, *args, **kwargs)

    MobileGymEnv.reset = _patched_reset

    class _DeterministicRandomProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(random, name)

        @staticmethod
        def uniform(a: float, b: float) -> float:
            if (a, b) == (0.5, 1.125):
                return 1.0
            return 0.8

    mg_mod.random = _DeterministicRandomProxy()  # type: ignore[assignment]

    # 3. Patch LLMClient.chat to enforce non-streaming token_ids capture, server-parser recovery,
    #    no <think> history bloat, and per-task endpoint error attribution.
    from bench_env.llm.openai_chat import ChatResult, LLMClient

    def _patched_chat(
        self: Any,
        messages: Optional[list[dict[str, Any]]] = None,
        *,
        args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if messages is None:
            messages = kwargs.pop("messages", [])
        task_id = _CURRENT_TASK_ID.get() or "_unattributed"
        base_defaults = getattr(self, "default_args", None)
        if base_defaults is None:
            base_defaults = getattr(self, "default_kwargs", None)
        call_kwargs: dict[str, Any] = dict(base_defaults or {})
        if args:
            call_kwargs.update(args)
        call_kwargs.update(kwargs)
        call_kwargs.pop("stream", None)
        call_kwargs.pop("stream_print", None)
        call_kwargs["temperature"] = temperature
        call_kwargs["max_tokens"] = max_tokens

        # Strip Google AI Studio extra_body={"extra_body": {"google": ...}} when talking to vLLM/OpenAI.
        eb = dict(call_kwargs.get("extra_body") or {})
        eb.pop("extra_body", None)
        eb.pop("top_k", None)
        ct_kwargs = dict(eb.get("chat_template_kwargs") or {})
        ct_kwargs["enable_thinking"] = bool(enable_thinking)
        eb["chat_template_kwargs"] = ct_kwargs
        eb["return_token_ids"] = True
        call_kwargs["extra_body"] = eb

        if "tools" not in call_kwargs and call_kwargs.get("tool_choice") == "none":
            call_kwargs.pop("tool_choice", None)

        expected_tools = bool(call_kwargs.get("tools")) and call_kwargs.get("tool_choice") != "none"
        max_retries = int(getattr(self, "max_retries", 2))
        timeout_s = float(getattr(self, "total_timeout_s", 300.0) or 300.0)

        last_err: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                time.sleep(min(2.0 ** attempt, 8.0))
            try:
                req_start = time.time()
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=timeout_s,
                    **call_kwargs,
                )
                latency = time.time() - req_start
                if hasattr(resp, "model_dump"):
                    raw = resp.model_dump()
                elif isinstance(resp, dict):
                    raw = dict(resp)
                else:
                    raw = {"choices": [{"message": {"content": str(resp)}}]}

                choice0 = (getattr(resp, "choices", None) or [None])[0]
                msg = getattr(choice0, "message", None)
                if msg is not None:
                    content = str(getattr(msg, "content", None) or "")
                    reasoning = (
                        getattr(msg, "reasoning_content", None)
                        or getattr(msg, "reasoning", None)
                    )
                    if reasoning is None and hasattr(self, "_extract_reasoning_from_message"):
                        reasoning = self._extract_reasoning_from_message(msg)
                    raw_tcs = getattr(msg, "tool_calls", None)
                else:
                    msg_dict = (raw.get("choices") or [{}])[0].get("message") or {}
                    content = str(msg_dict.get("content") or "")
                    reasoning = msg_dict.get("reasoning_content") or msg_dict.get("reasoning")
                    raw_tcs = msg_dict.get("tool_calls")

                if reasoning is not None:
                    reasoning = str(reasoning)

                usage = getattr(resp, "usage", None)
                if usage is not None:
                    prompt_toks = int(getattr(usage, "prompt_tokens", 0) or 0)
                    comp_toks = int(getattr(usage, "completion_tokens", 0) or 0)
                else:
                    u_dict = raw.get("usage") or {}
                    prompt_toks = int(u_dict.get("prompt_tokens", 0) or 0)
                    comp_toks = int(u_dict.get("completion_tokens", 0) or 0)

                policy = apply_server_parser_policy(
                    raw,
                    thinking=bool(enable_thinking),
                    text=content,
                    reasoning=reasoning or "",
                    tool_calls=raw_tcs,
                    expected_tool_calls=expected_tools,
                )
                ev = policy.get("event")
                with _LOCK:
                    st = _TASK_CALL_STATS[task_id]
                    st["calls"] += 1
                    st["prompt_tokens"] += prompt_toks
                    st["completion_tokens"] += comp_toks
                    if ev == SERVER_PARSER_MOVED_TEXT:
                        st["server_parser_moved_text"] += 1
                    elif ev == SERVER_PARSER_DROPPED_TEXT:
                        st["server_parser_dropped_text"] += 1
                    elif ev == SERVER_PARSER_TEXT_LOST:
                        st["server_parser_text_lost"] += 1

                content = policy.get("text") or ""
                reasoning = policy.get("reasoning") or reasoning

                # If content is empty on a thinking run and reasoning holds the final tag, rescue it
                # without polluting multi-turn history with raw <think> blocks.
                if not content.strip() and reasoning:
                    if ("<answer>" in reasoning.lower() or "<tool_call>" in reasoning.lower()
                            or "mobile_use" in reasoning):
                        content = reasoning
                    else:
                        with _LOCK:
                            _TASK_CALL_STATS[task_id]["stopped_in_reasoning"] += 1

                # Strip reasoning_content/reasoning from raw message dict so Qwen35SessionAgent
                # does not copy raw reasoning into multi-turn assistant history.
                if isinstance(raw, dict):
                    try:
                        raw_msg = (raw.get("choices") or [{}])[0].get("message")
                        if isinstance(raw_msg, dict):
                            raw_msg.pop("reasoning_content", None)
                            raw_msg.pop("reasoning", None)
                    except Exception:
                        pass

                return ChatResult(
                    content=content,
                    reasoning=reasoning,
                    latency_s=latency,
                    raw=raw,
                )
            except Exception as exc:
                last_err = exc
                is_harness, status, msg_str = _classify_llm_exception(exc)
                if not is_harness:
                    with _LOCK:
                        _TASK_CALL_STATS[task_id]["content_rejections"].append(
                            {"status": status, "error": msg_str}
                        )
                    raise
                if attempt >= max_retries:
                    with _LOCK:
                        _TASK_CALL_STATS[task_id]["endpoint_failures"].append(
                            {"status": status, "error": msg_str}
                        )
                    raise
        assert last_err is not None
        raise last_err

    LLMClient.chat = _patched_chat

    # 4. Suite-specific agent protocol adjustments.
    if suite == "mobilegym":
        from bench_env.agent.generic_v2 import GenericAgentV2

        orig_parse = GenericAgentV2._parse_llm_output

        def _patched_parse(self: Any, raw_text: str) -> tuple[str, dict[str, Any]]:
            cleaned = re.sub(
                r"<(think|thought)>.*?</\1>\s*",
                "",
                raw_text or "",
                flags=re.DOTALL | re.IGNORECASE,
            )
            cleaned = re.sub(
                r"<\|channel\|>thought\s*.*?<\|channel\|>",
                "",
                cleaned,
                flags=re.DOTALL,
            )
            return orig_parse(self, cleaned if cleaned.strip() else raw_text)

        GenericAgentV2._parse_llm_output = _patched_parse
    elif suite == "mobilegym_pp":
        import bench_env.hybrid_agent.qwen35_session as q35

        orig_session_family = q35.session_family

        def _patched_session_family(model_name: str) -> str:
            folded = (model_name or "").casefold()
            if "gemma" in folded:
                # Route Gemma to the XML/user-turn session family so tool_chat_template_gemma4.jinja
                # never drops Turn 2+ observations or emits unclosed <|tool_response> tokens.
                return "gemini"
            return orig_session_family(model_name)

        q35.session_family = _patched_session_family

        orig_parse_tool_call = q35.parse_session_tool_call
        orig_parse_model_output = q35._parse_model_output
        orig_transform_mobile_use = q35.transform_mobile_use
        _gui_actions = {
            "click",
            "long_press",
            "swipe",
            "type",
            "answer",
            "system_button",
            "wait",
            "terminate",
            "open_app",
            "double_tap",
            "left_click",
            "drag",
        }
        _sys_buttons = {"home": "Home", "back": "Back", "menu": "Menu", "enter": "Enter"}
        _meta_keys = {
            "name",
            "tool",
            "function",
            "arguments",
            "parameters",
            "parameter",
            "args",
            "action_input",
        }
        _gemma4_call_re = re.compile(
            r"<\|tool_call\|?>\s*call:([A-Za-z0-9_.:-]+)\s*(\{[\s\S]*\})\s*(?:<\|?tool_call\|?>|$)"
        )

        def _parse_json_or_gemma4_dict(raw_payload: str) -> Optional[dict[str, Any]]:
            payload = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                (raw_payload or "").strip(),
                flags=re.IGNORECASE,
            ).strip()
            if not payload.startswith("{"):
                return None
            if '<|"|>' in payload:
                payload = re.sub(
                    r'<\|"\|>([\s\S]*?)<\|"\|>',
                    lambda m: json.dumps(m.group(1), ensure_ascii=False),
                    payload,
                )
            for candidate in (
                payload,
                re.sub(r",\s*([}\]])", r"\1", payload),
                re.sub(
                    r",\s*([}\]])",
                    r"\1",
                    re.sub(
                        r'("(?:\\.|[^"\\])*")|([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:',
                        lambda m: m.group(1)
                        if m.group(1)
                        else f'{m.group(2)}"{m.group(3)}":',
                        payload,
                    ),
                ),
            ):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
            return None

        def _normalize_session_call(name: Any, args: Any) -> dict[str, Any]:
            name_str = str(name or "").strip()
            if ":" in name_str:
                name_str = name_str.split(":")[-1].strip()
            call_args = dict(args) if isinstance(args, dict) else {}
            for nk in ("parameters", "parameter", "arguments", "args", "action_input"):
                nested = call_args.get(nk)
                if isinstance(nested, dict):
                    call_args.pop(nk, None)
                    for k, v in nested.items():
                        if k == "action" and call_args.get("action") == "mobile_use":
                            call_args["action"] = v
                        else:
                            call_args.setdefault(k, v)
            if call_args.get("action") == "mobile_use" and call_args.get("action_type"):
                call_args["action"] = str(call_args.pop("action_type")).strip()
            act = str(call_args.get("action") or "").strip()
            if (
                name_str == "mobile_use"
                and act
                and (
                    "__" in act
                    or (
                        act not in _gui_actions
                        and act.lower() not in _sys_buttons
                        and act != "mobile_use"
                    )
                )
            ):
                call_args.pop("action", None)
                return {"name": act, "arguments": call_args}
            if name_str in _gui_actions:
                call_args.setdefault("action", name_str)
                name_str = "mobile_use"
                act = str(call_args.get("action") or "").strip()
            if name_str == "mobile_use":
                if act.lower() in _sys_buttons:
                    call_args["action"] = "system_button"
                    call_args.setdefault("button", _sys_buttons[act.lower()])
                    act = "system_button"
                if act.lower() == "terminate":
                    st = str(call_args.get("status") or "").strip().lower()
                    if st not in {"success", "failure"}:
                        call_args["status"] = (
                            "failure"
                            if ("fail" in st or "infeas" in st or "abort" in st)
                            else "success"
                        )
            return {"name": name_str, "arguments": call_args}

        def _patched_parse_session_tool_call(text: str) -> Optional[dict[str, Any]]:
            res = orig_parse_tool_call(text)
            if isinstance(res, dict) and res.get("name"):
                norm = _normalize_session_call(res["name"], res.get("arguments"))
                if norm.get("arguments") or norm.get("name") != "mobile_use":
                    return norm
            cleaned = re.sub(
                r"<(think|thought)>.*?</\1>\s*",
                "",
                text or "",
                flags=re.DOTALL | re.IGNORECASE,
            )
            cleaned = re.sub(
                r"<\|?channel\|?>thought\s*.*?<\|?channel\|?>",
                "",
                cleaned,
                flags=re.DOTALL,
            )
            for m_gemma in reversed(list(_gemma4_call_re.finditer(cleaned))):
                g_name, g_payload = m_gemma.group(1), m_gemma.group(2)
                g_data = _parse_json_or_gemma4_dict(g_payload)
                if isinstance(g_data, dict):
                    return _normalize_session_call(g_name, g_data)
            candidates = [m.group(1).strip() for m in q35._TOOL_CALL_RE.finditer(cleaned)]
            if not candidates:
                m_json = re.search(r"\{[\s\S]*\}", cleaned)
                if m_json:
                    candidates = [m_json.group(0).strip()]
            for raw_payload in reversed(candidates):
                data = _parse_json_or_gemma4_dict(raw_payload)
                if not isinstance(data, dict):
                    continue
                name = data.get("name") or data.get("tool") or data.get("function")
                args = (
                    data.get("arguments")
                    or data.get("parameters")
                    or data.get("parameter")
                    or data.get("args")
                    or data.get("action_input")
                )
                if isinstance(args, str):
                    args = _parse_json_or_gemma4_dict(args)
                if name:
                    extra_args = {k: v for k, v in data.items() if k not in _meta_keys}
                    call_args = dict(args) if isinstance(args, dict) else {}
                    for k, v in extra_args.items():
                        call_args.setdefault(k, v)
                    return _normalize_session_call(name, call_args)
                action = data.get("action") or data.get("action_type")
                if action:
                    call_args = dict(data)
                    call_args["action"] = str(action).strip()
                    return _normalize_session_call("mobile_use", call_args)
                if "coordinate" in data and len(data) <= 2:
                    call_args = dict(data)
                    call_args.setdefault("action", "click")
                    return _normalize_session_call("mobile_use", call_args)
                if len(data) == 1:
                    k, v = next(iter(data.items()))
                    if isinstance(v, dict):
                        if k == "mobile_use" or "action" in v:
                            return _normalize_session_call("mobile_use", v)
                        return _normalize_session_call(str(k), v)
            return res

        def _patched_parse_model_output(
            raw: Any, content: str, *, prefer_xml: bool = False
        ) -> tuple[Optional[dict[str, Any]], Optional[str], dict[str, Any]]:
            tool_call, tool_call_id, assistant = orig_parse_model_output(
                raw, content, prefer_xml=prefer_xml
            )
            if isinstance(tool_call, dict) and tool_call.get("name") and not assistant.get("tool_calls"):
                assistant["content"] = (
                    f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"
                )
            return tool_call, tool_call_id, assistant

        def _patched_transform_mobile_use(
            action: str, arguments: dict[str, Any], width: int, height: int
        ) -> dict[str, Any]:
            if action == "terminate":
                st = str(arguments.get("status") or "").strip().lower()
                if st not in {"success", "failure"}:
                    arguments = dict(arguments)
                    arguments["status"] = (
                        "failure"
                        if ("fail" in st or "infeas" in st or "abort" in st)
                        else "success"
                    )
            return orig_transform_mobile_use(action, arguments, width, height)

        q35.parse_session_tool_call = _patched_parse_session_tool_call
        q35._parse_model_output = _patched_parse_model_output
        q35.transform_mobile_use = _patched_transform_mobile_use

        if os.environ.get("GBENCH_MOBILEGYM_SAVE_SCREENSHOTS") != "1":
            from bench_env.hybrid_agent.artifacts import HybridArtifactRecorder

            def _noop_write_screenshot(self: Any, stem: str, observation: Any) -> str:
                return f"{stem}.jpg"

            HybridArtifactRecorder._write_screenshot = _noop_write_screenshot


def _start_simulator(repo_root: Path, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["VITE_CDN_BASE"] = "/cdn"
    env["VITE_GW_ALLOW_HOSTS"] = "none"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "npx",
            "vite",
            "--host",
            "127.0.0.1",
            "--port",
            str(SIMULATOR_PORT),
            "--strictPort",
        ],
        cwd=str(repo_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 60.0
    url = f"http://127.0.0.1:{SIMULATOR_PORT}/"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Vite simulator exited early with code {proc.returncode}; see {log_path}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    raise RuntimeError(f"Timed out waiting 60s for Vite simulator at {url}; see {log_path}")


def _stop_simulator(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _select_tasks_mobilegym(
    *,
    limit: Optional[int],
    shard_index: int,
    num_shards: int,
    explicit_task_ids: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]]]:
    import bench_env.splits as splits_mod
    from bench_env.task import TaskRegistry

    registry = TaskRegistry()
    raw_split = (
        splits_mod.resolve_split("test")
        if hasattr(splits_mod, "resolve_split")
        else splits_mod.get_split("test")
    )
    split_ids = sorted(raw_split)
    flat: List[Dict[str, Any]] = []
    for tid in split_ids:
        if hasattr(registry, "get_by_id"):
            cls = registry.get_by_id(tid)
            diff = str(getattr(cls, "difficulty", None) or "L2")
            apps = list(getattr(cls, "apps", None) or [])
            suite_name = str(
                getattr(cls, "_suite", None)
                or (tid.split(".", 1)[0] if "." in tid else "")
            )
        else:
            meta = registry.get_metadata(tid) or {}
            diff = str(meta.get("difficulty") or "L2")
            apps = list(meta.get("apps") or [])
            suite_name = str(meta.get("suite") or "")
        flat.append(
            {
                "task_id": tid,
                "difficulty": diff,
                "apps": apps,
                "suite": suite_name,
            }
        )
    n_full = len(flat)
    if explicit_task_ids:
        wanted = set(explicit_task_ids)
        selected = [item for item in flat if item["task_id"] in wanted]
        selection = {
            "n_full": n_full,
            "limit": limit,
            "n_selected": len(selected),
            "limit_draw": None,
        }
        return selected, selection, None

    shard_rec: Optional[Dict[str, Any]] = None
    if num_shards > 1:
        flat = flat[shard_index - 1 :: num_shards]
    if limit is not None and limit > 0:
        idx = stratified_indices([x["difficulty"] for x in flat], limit, "mobilegym")
        flat = [flat[i] for i in idx]
    if num_shards > 1:
        shard_rec = {
            "spec": f"{shard_index}/{num_shards}",
            "n_full": n_full,
            "n_selected": len(flat),
        }
    selection = {
        "n_full": n_full,
        "limit": limit,
        "n_selected": len(flat),
        "limit_draw": (
            "stratified_indices(key=difficulty, seed=mobilegym)"
            if (limit is not None and limit > 0)
            else None
        ),
    }
    return flat, selection, shard_rec


def _select_tasks_mobilegym_pp(
    repo_root: Path,
    *,
    limit: Optional[int],
    shard_index: int,
    num_shards: int,
    explicit_task_ids: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]]]:
    official_path = repo_root / "bench215" / "official.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    raw_tasks = sorted(official.get("tasks") or [], key=lambda r: str(r["task_id"]))
    reg: Any = None
    try:
        from bench_env.task import TaskRegistry

        reg = TaskRegistry()
    except Exception:
        reg = None

    flat: List[Dict[str, Any]] = []
    for item in raw_tasks:
        tid = str(item["task_id"])
        cls: Any = None
        if reg is not None and hasattr(reg, "get_by_id"):
            try:
                cls = reg.get_by_id(tid)
            except Exception:
                cls = None
        diff = str(item.get("difficulty") or getattr(cls, "difficulty", None) or "L2")
        raw_apps = item.get("apps") if item.get("apps") is not None else getattr(cls, "apps", None)
        if isinstance(raw_apps, str):
            apps = [a.strip() for a in re.split(r"[,、]", raw_apps) if a.strip()]
        else:
            apps = list(raw_apps or [])
        max_steps = int(item.get("max_steps") or getattr(cls, "max_steps", None) or 45)
        norm = dict(item)
        norm["task_id"] = tid
        norm["difficulty"] = diff
        norm["apps"] = apps
        norm["app_count"] = int(item.get("app_count") or len(apps) or 1)
        norm["max_steps"] = max_steps
        flat.append(norm)

    n_full = len(flat)
    if explicit_task_ids:
        wanted = set(explicit_task_ids)
        selected = [item for item in flat if item["task_id"] in wanted]
        selection = {
            "n_full": n_full,
            "limit": limit,
            "n_selected": len(selected),
            "limit_draw": None,
        }
        return selected, selection, None

    shard_rec: Optional[Dict[str, Any]] = None
    if num_shards > 1:
        flat = flat[shard_index - 1 :: num_shards]
    if limit is not None and limit > 0:
        idx = stratified_indices([x.get("difficulty") or "L2" for x in flat], limit, "mobilegym_pp")
        flat = [flat[i] for i in idx]
    if num_shards > 1:
        shard_rec = {
            "spec": f"{shard_index}/{num_shards}",
            "n_full": n_full,
            "n_selected": len(flat),
        }
    selection = {
        "n_full": n_full,
        "limit": limit,
        "n_selected": len(flat),
        "limit_draw": (
            "stratified_indices(key=difficulty, seed=mobilegym_pp)"
            if (limit is not None and limit > 0)
            else None
        ),
    }
    return flat, selection, shard_rec


def _is_harness_task_error(task_id: str, error_str: Optional[str]) -> Tuple[bool, Optional[str]]:
    stats = _TASK_CALL_STATS.get(task_id) or {}
    if stats.get("endpoint_failures"):
        first = stats["endpoint_failures"][0]
        return True, f"endpoint_failure: {first.get('error')}"
    if stats.get("server_parser_text_lost", 0) > 0:
        return True, "server_parser_text_lost"
    if not error_str:
        return False, None
    if stats.get("content_rejections"):
        return False, None
    low = error_str.lower()
    if any(
        tok in low
        for tok in (
            "400",
            "422",
            "maximum context length",
            "context_length_exceeded",
            "format_error",
            "invalid_tool_call",
        )
    ):
        return False, None
    return True, error_str


def _run_mobilegym_suite(
    args: argparse.Namespace,
    selected: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    from bench_env.config import RunnerConfig
    from bench_env.runner import ParallelRunner
    from bench_env.runner.base import BaseRunner

    orig_run_episode = BaseRunner.run_episode

    @staticmethod
    async def _tracked_run_episode(env: Any, agent: Any, task: Any, *a: Any, **kw: Any) -> Any:
        _CURRENT_TASK_ID.set(getattr(task, "id", None))
        return await orig_run_episode(env, agent, task, *a, **kw)

    BaseRunner.run_episode = _tracked_run_episode

    save_screens = os.environ.get("GBENCH_MOBILEGYM_SAVE_SCREENSHOTS") == "1"
    cfg = RunnerConfig(
        agent="generic_v2",
        model_name=args.model,
        model_base_url=args.api_base,
        model_api_key=args.api_key or "EMPTY",
        env_url=SIMULATOR_URL,
        headless=True,
        coord_space="norm_0_1000",
        task_ids=[item["task_id"] for item in selected],
        parallel=max(1, int(args.workers)),
        isolation="browsers",
        sample_seed=int(args.sample_seed),
        eval_mode="grounded",
        judge_mode="state",
        no_save_trajectory=not save_screens,
        runs_dir=output_dir / "episodes",
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        quiet=True,
    )

    async def _exec() -> Any:
        runner = await ParallelRunner.from_config(cfg)
        return await runner.run()

    episode_results = asyncio.run(_exec())
    by_id = {item["task_id"]: item for item in selected}
    rows: List[Dict[str, Any]] = []
    for r in episode_results:
        tid = str(r.task_id)
        meta = by_id.get(tid, {})
        err_str = r.error
        is_harness, harness_reason = _is_harness_task_error(tid, err_str)
        stats = _TASK_CALL_STATS.get(tid, {})
        rows.append(
            {
                "task_id": tid,
                "difficulty": str(getattr(r, "difficulty", "") or meta.get("difficulty") or "L2"),
                "apps": list(getattr(r, "apps", None) or meta.get("apps") or []),
                "app_count": len(getattr(r, "apps", None) or meta.get("apps") or []),
                "suite": str(getattr(r, "suite", "") or meta.get("suite") or ""),
                "success": bool(r.success) and not is_harness,
                "goal_success": bool(r.goal_success) and not is_harness,
                "clean": bool(r.no_unexpected_changes),
                "progress": 0.0 if is_harness else float(r.progress or 0.0),
                "false_complete": False if is_harness else bool(r.false_complete),
                "overdue_termination": False if is_harness else bool(r.overdue_termination),
                "steps": int(r.steps or 0),
                "max_steps": int(getattr(r, "max_steps", 0) or 0),
                "stop_reason": str(r.execution.stop_reason or ""),
                "runtime_s": float(r.execution.runtime_s or 0.0),
                "error": err_str,
                "harness_error": is_harness,
                "harness_error_reason": harness_reason,
                "llm_calls": int(stats.get("calls", 0)),
                "prompt_tokens": int(stats.get("prompt_tokens", 0)),
                "completion_tokens": int(stats.get("completion_tokens", 0)),
                "server_parser_moved_text": int(stats.get("server_parser_moved_text", 0)),
                "server_parser_dropped_text": int(stats.get("server_parser_dropped_text", 0)),
                "server_parser_text_lost": int(stats.get("server_parser_text_lost", 0)),
                "stopped_in_reasoning": int(stats.get("stopped_in_reasoning", 0)),
            }
        )
    return rows


def _run_mobilegym_pp_suite(
    args: argparse.Namespace,
    selected: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    import bench_env.hybrid_benchmark.runner as hrunner
    from bench_env.hybrid_benchmark.runner import HybridBenchmarkConfig, run_benchmark

    by_id = {str(item["task_id"]): item for item in selected}
    orig_run_one = hrunner._run_one

    async def _patched_run_one(
        env: Any, agent: Any, task: Any, config: Any, worker_id: int
    ) -> dict[str, Any]:
        tid = str(getattr(task, "id", ""))
        _CURRENT_TASK_ID.set(tid)
        meta = by_id.get(tid) or {}
        inst = meta.get("instruction")
        if inst:
            task.templates = [str(inst)]
            task.task_name = None
            if hasattr(task, "spec") and hasattr(task.spec, "instruction"):
                try:
                    task.spec = dataclasses.replace(task.spec, instruction=str(inst))
                except Exception:
                    pass
            task._instruction_override = None
        if meta.get("max_steps"):
            task.max_steps = int(meta["max_steps"])
        return await orig_run_one(env, agent, task, config, worker_id)

    hrunner._run_one = _patched_run_one

    suites_csv = (
        ",".join(
            sorted(
                {
                    str(item["task_id"]).split(".", 1)[0]
                    for item in selected
                    if "." in str(item["task_id"])
                }
            )
        )
        or "core"
    )
    cfg = HybridBenchmarkConfig(
        env_url=SIMULATOR_URL,
        model_base_url=args.api_base,
        model_api_key=args.api_key or "EMPTY",
        model_name=args.model,
        output_dir=output_dir / "hybrid_run",
        suite=suites_csv,
        task_ids=tuple(str(item["task_id"]) for item in selected),
        parallel=max(1, int(args.workers)),
        max_steps=None,
        headless=True,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        runtime="qwen35_session",
        enable_thinking=bool(args.enable_thinking),
        require_think_tags=False,
        interaction_mode="hybrid",
        quiet=True,
    )

    asyncio.run(run_benchmark(cfg))
    results_jsonl = cfg.output_dir / "results.jsonl"
    raw_rows: List[Dict[str, Any]] = []
    if results_jsonl.is_file():
        for line in results_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_rows.append(json.loads(line))

    rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        tid = str(r.get("task_id") or "")
        meta = by_id.get(tid) or {}
        err_str = r.get("error") or r.get("judge_error")
        is_harness, harness_reason = _is_harness_task_error(tid, err_str)
        stats = _TASK_CALL_STATS.get(tid, {})
        goal_ok = bool(r.get("success")) and not is_harness
        clean_ok = bool(r.get("clean", True))
        stop_reason = str(r.get("stop_reason") or "")
        strict_ok = goal_ok and clean_ok and stop_reason == "COMPLETE"
        apps = list(meta.get("apps") or r.get("apps") or [])
        rows.append(
            {
                "task_id": tid,
                "difficulty": str(meta.get("difficulty") or r.get("difficulty") or "L2"),
                "apps": apps,
                "app_count": int(meta.get("app_count") or len(apps) or 1),
                "tier": str(meta.get("tier") or ""),
                "optimal_steps": int(meta.get("optimal_steps") or 0),
                "optimal_tool_calls": int(meta.get("optimal_tool_calls") or 0),
                "success": goal_ok,
                "strict_success": strict_ok,
                "goal_success": goal_ok,
                "clean": clean_ok,
                "progress": 0.0 if is_harness else float(r.get("progress") or 0.0),
                "false_complete": (
                    False if is_harness else bool(stop_reason == "COMPLETE" and not strict_ok)
                ),
                "overdue_termination": (
                    False if is_harness else bool(stop_reason == "MAX_STEPS" and goal_ok)
                ),
                "steps": int(r.get("steps") or 0),
                "max_steps": int(meta.get("max_steps") or r.get("max_steps") or 45),
                "stop_reason": stop_reason,
                "tool_calls": list(r.get("tool_calls") or []),
                "app_tool_calls": list(r.get("app_tool_calls") or []),
                "runtime_s": float(r.get("runtime_s") or 0.0),
                "error": err_str,
                "harness_error": is_harness,
                "harness_error_reason": harness_reason,
                "llm_calls": int(stats.get("calls", 0)),
                "prompt_tokens": int(stats.get("prompt_tokens", 0)),
                "completion_tokens": int(stats.get("completion_tokens", 0)),
                "server_parser_moved_text": int(stats.get("server_parser_moved_text", 0)),
                "server_parser_dropped_text": int(stats.get("server_parser_dropped_text", 0)),
                "server_parser_text_lost": int(stats.get("server_parser_text_lost", 0)),
                "stopped_in_reasoning": int(stats.get("stopped_in_reasoning", 0)),
            }
        )
    return rows


def aggregate_rows(
    suite: str,
    rows: List[Dict[str, Any]],
    *,
    selection: Dict[str, Any],
    shard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scored = [r for r in rows if not r.get("harness_error")]
    harness_failed = [r for r in rows if r.get("harness_error")]
    n_scored = len(scored)
    n_success = sum(1 for r in scored if r.get("success"))
    sr = (n_success / n_scored) if n_scored > 0 else None
    pr = (sum(float(r.get("progress") or 0.0) for r in scored) / n_scored) if n_scored > 0 else 0.0
    fc = (sum(1 for r in scored if r.get("false_complete")) / n_scored) if n_scored > 0 else 0.0
    ot = (sum(1 for r in scored if r.get("overdue_termination")) / n_scored) if n_scored > 0 else 0.0
    use_rate = (sum(1 for r in scored if not r.get("clean", True)) / n_scored) if n_scored > 0 else 0.0

    by_diff: Dict[str, List[float]] = collections.defaultdict(list)
    by_app_count: Dict[str, List[float]] = collections.defaultdict(list)
    for r in scored:
        diff = str(r.get("difficulty") or "L2")
        by_diff[diff].append(1.0 if r.get("success") else 0.0)
        ac = min(4, max(1, int(r.get("app_count") or len(r.get("apps") or []) or 1)))
        by_app_count[f"{ac}_app"].append(1.0 if r.get("success") else 0.0)

    difficulty_breakdown = {
        k: {"accuracy": sum(v) / len(v), "n": len(v), "passed": int(sum(v))}
        for k, v in sorted(by_diff.items())
    }
    app_count_breakdown = {
        k: {"accuracy": sum(v) / len(v), "n": len(v), "passed": int(sum(v))}
        for k, v in sorted(by_app_count.items())
    }

    return {
        "suite": suite,
        "simulated_time": SIMULATED_TIME,
        "n_full": selection.get("n_full", len(rows)),
        "n_selected": selection.get("n_selected", len(rows)),
        "empty_selection": len(rows) == 0 and selection.get("n_selected", 0) == 0,
        "selection": selection,
        "shard": shard,
        "total": len(rows),
        "n_scored": n_scored,
        "n_passed": n_success,
        "n_harness_errors": len(harness_failed),
        "harness_error_tasks": [
            {"task_id": r["task_id"], "reason": r.get("harness_error_reason")}
            for r in harness_failed
        ],
        "accuracy": sr,
        "progress_rate": pr,
        "false_complete_rate": fc,
        "overdue_termination_rate": ot,
        "unexpected_side_effect_rate": use_rate,
        "difficulty_breakdown": difficulty_breakdown,
        "app_count_breakdown": app_count_breakdown,
        "server_parser": {
            "moved_text": sum(int(r.get("server_parser_moved_text", 0)) for r in rows),
            "dropped_text": sum(int(r.get("server_parser_dropped_text", 0)) for r in rows),
            "text_lost": sum(int(r.get("server_parser_text_lost", 0)) for r in rows),
            "stopped_in_reasoning": sum(int(r.get("stopped_in_reasoning", 0)) for r in rows),
        },
        "token_usage": {
            "llm_calls": sum(int(r.get("llm_calls", 0)) for r in rows),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in rows),
            "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in rows),
        },
        "per_task": rows,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gbench in-container MobileGym / MobileGym++ runner")
    p.add_argument("--suite", choices=["mobilegym", "mobilegym_pp"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-base", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--task-ids", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_shards < 1 or args.shard_index < 1 or args.shard_index > args.num_shards:
        sys.stderr.write(f"Invalid shard specification: {args.shard_index}/{args.num_shards}\n")
        return 2

    repo_root = Path(
        os.environ.get(
            "GBENCH_MOBILEGYM_PP_ROOT" if args.suite == "mobilegym_pp" else "GBENCH_MOBILEGYM_ROOT",
            "/opt/mobilegym-pp" if args.suite == "mobilegym_pp" else "/opt/mobilegym",
        )
    )
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.suite}_summary.json"

    _install_common_patches(
        suite=args.suite,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )

    explicit_ids = (
        [t.strip() for t in args.task_ids.split(",") if t.strip()]
        if args.task_ids
        else None
    )
    if args.suite == "mobilegym":
        selected, selection, shard_rec = _select_tasks_mobilegym(
            limit=args.limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            explicit_task_ids=explicit_ids,
        )
    else:
        selected, selection, shard_rec = _select_tasks_mobilegym_pp(
            repo_root,
            limit=args.limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            explicit_task_ids=explicit_ids,
        )

    if not selected:
        summary = aggregate_rows(args.suite, [], selection=selection, shard=shard_rec)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0

    sim_proc: Optional[subprocess.Popen] = None
    try:
        sim_proc = _start_simulator(repo_root, output_dir / "vite_simulator.log")
        if args.suite == "mobilegym":
            rows = _run_mobilegym_suite(args, selected, output_dir)
        else:
            rows = _run_mobilegym_pp_suite(args, selected, output_dir)
    finally:
        _stop_simulator(sim_proc)

    summary = aggregate_rows(args.suite, rows, selection=selection, shard=shard_rec)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
