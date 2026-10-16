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

"""tau2 / tau3: where an infrastructure error came from, and the evaluable counts.

tau2's run_with_retry records a simulation whose every attempt raised as an infrastructure error,
and compute_metrics leaves it out of pass^k, whatever raised. gbench reads the source from the
traceback tau2 records (`_infra_error_source`):

  * a harness failure (the model endpoint, the Gemini user simulator or judge, a tool, the Gemini
    embeddings, anything else off the agent's turn) is a harness error: a `harness_errors` entry,
    named in leaderboard_comparable_reason as a harness failure, never the agent's outcome;
  * an error raised on the agent's own turn (a reply tau2 rejects: no text and no tool calls, or
    tool-call arguments that are not JSON; a request over the endpoint's context window) is not a
    harness failure: an `agent_turn_errors` entry, named in the reason as such.

Both leave the status completed_with_errors. `category_accuracy` and `total_questions` count the
evaluable simulations (the ones pass^1 covers), so an infrastructure error never reads as a
failed simulation.

No network, no Docker, no model server, no Gemini call: tau2 is stubbed; the tracebacks below are
real records of the pinned tau2.
"""

import inspect
import json
import os
import sys
import threading
import types
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import tau_common as TC

ROOT = Path(__file__).resolve().parent.parent
GEMINI_KEY = "gemini-mediums-2m-key-0123456789"
ENDPOINT_KEY = "endpoint-mediums-2m-key-9876543210"


#: Records tau2 wrote at TAU2_BENCH_REVISION, captured in the gbench-tau2 image (docker run
#: --network none) with gbench's real patches installed and litellm replaced by a stand-in that
#: fails the way each scenario says: error_type, error and error_traceback as run_with_retry
#: keeps them in the simulation's info. The stand-in's frame (/probe/probe.py) sits where
#: litellm's own frames sit in a live run.
REAL = {
    'agent_empty': {
        "error_type": 'ValueError',
        "error": (
            'AssistantMessage must have either content or tool_calls. Got AssistantMessage\n'
            'is_final_chunk: True'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 861, in ste'
            'p\n'
            '    agent_msg.validate()\n'
            '  File "/opt/tau2-bench/src/tau2/data_model/message.py", line 287, in validate\n'
            '    raise ValueError(\n'
            'ValueError: AssistantMessage must have either content or tool_calls. Got Assista'
            'ntMessage\n'
            'is_final_chunk: True\n'),
    },
    'agent_badjson': {
        "error_type": 'JSONDecodeError',
        "error": (
            'Expecting property name enclosed in double quotes: line 1 column 2 (char 1)'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 858, in ste'
            'p\n'
            '    agent_msg, self.agent_state = self.agent.generate_next_message(\n'
            '                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 111, in generate_next'
            '_message\n'
            '    assistant_message = self._generate_next_message(message, state)\n'
            '                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 128, in _generate_nex'
            't_message\n'
            '    assistant_message = generate(\n'
            '                        ^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1587, in _generate_retry\n'
            '    return _orig_generate(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 440, in generate\n'
            '    arguments=json.loads(tool_call.function.arguments),\n'
            '              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/usr/local/lib/python3.12/json/__init__.py", line 346, in loads\n'
            '    return _default_decoder.decode(s)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/usr/local/lib/python3.12/json/decoder.py", line 338, in decode\n'
            '    obj, end = self.raw_decode(s, idx=_w(s, 0).end())\n'
            '               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/usr/local/lib/python3.12/json/decoder.py", line 354, in raw_decode\n'
            '    obj, end = self.scan_once(s, idx)\n'
            '               ^^^^^^^^^^^^^^^^^^^^^^\n'
            'json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes:'
            ' line 1 column 2 (char 1)\n'),
    },
    'agent_ctx': {
        "error_type": 'ContextWindowExceededError',
        "error": (
            'litellm.ContextWindowExceededError: litellm.BadRequestError: prompt too long'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 858, in ste'
            'p\n'
            '    agent_msg, self.agent_state = self.agent.generate_next_message(\n'
            '                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 111, in generate_next'
            '_message\n'
            '    assistant_message = self._generate_next_message(message, state)\n'
            '                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 128, in _generate_nex'
            't_message\n'
            '    assistant_message = generate(\n'
            '                        ^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1587, in _generate_retry\n'
            '    return _orig_generate(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 418, in generate\n'
            '    raise e\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 409, in generate\n'
            '    response = completion(\n'
            '               ^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 330, in completion\n'
            '    return inner(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1459, in _completion_cascad'
            'e\n'
            '    return inner_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1510, in _completion_retry\n'
            '    resp = _orig_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/probe/probe.py", line 47, in fake\n'
            '    raise litellm.ContextWindowExceededError(message="prompt too long", model=mo'
            'del,\n'
            'litellm.exceptions.ContextWindowExceededError: litellm.ContextWindowExceededErro'
            'r: litellm.BadRequestError: prompt too long\n'),
    },
    'agent_down': {
        "error_type": 'APIConnectionError',
        "error": (
            'litellm.APIConnectionError: Connection refused'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 858, in ste'
            'p\n'
            '    agent_msg, self.agent_state = self.agent.generate_next_message(\n'
            '                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 111, in generate_next'
            '_message\n'
            '    assistant_message = self._generate_next_message(message, state)\n'
            '                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 128, in _generate_nex'
            't_message\n'
            '    assistant_message = generate(\n'
            '                        ^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1587, in _generate_retry\n'
            '    return _orig_generate(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 418, in generate\n'
            '    raise e\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 409, in generate\n'
            '    response = completion(\n'
            '               ^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 330, in completion\n'
            '    return inner(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1459, in _completion_cascad'
            'e\n'
            '    return inner_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1510, in _completion_retry\n'
            '    resp = _orig_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/probe/probe.py", line 50, in fake\n'
            '    raise litellm.APIConnectionError(message="Connection refused", llm_provider='
            '"openai",\n'
            'litellm.exceptions.APIConnectionError: litellm.APIConnectionError: Connection re'
            'fused\n'),
    },
    'agent_500': {
        "error_type": 'InternalServerError',
        "error": (
            'litellm.InternalServerError: boom'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 858, in ste'
            'p\n'
            '    agent_msg, self.agent_state = self.agent.generate_next_message(\n'
            '                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 111, in generate_next'
            '_message\n'
            '    assistant_message = self._generate_next_message(message, state)\n'
            '                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/agent/llm_agent.py", line 128, in _generate_nex'
            't_message\n'
            '    assistant_message = generate(\n'
            '                        ^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1587, in _generate_retry\n'
            '    return _orig_generate(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 418, in generate\n'
            '    raise e\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 409, in generate\n'
            '    response = completion(\n'
            '               ^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 330, in completion\n'
            '    return inner(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1459, in _completion_cascad'
            'e\n'
            '    return inner_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1510, in _completion_retry\n'
            '    resp = _orig_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/probe/probe.py", line 53, in fake\n'
            '    raise litellm.InternalServerError(message="boom", llm_provider="openai", mod'
            'el=model)\n'
            'litellm.exceptions.InternalServerError: litellm.InternalServerError: boom\n'),
    },
    'user_outage': {
        "error_type": 'APIConnectionError',
        "error": (
            'litellm.APIConnectionError: user sim down'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 837, in ste'
            'p\n'
            '    user_msg, self.user_state = self.user.generate_next_message(\n'
            '                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/user/user_simulator.py", line 201, in generate_'
            'next_message\n'
            '    user_message = self._generate_next_message(message, state)\n'
            '                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/user/user_simulator.py", line 235, in _generate'
            '_next_message\n'
            '    assistant_message = generate(\n'
            '                        ^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1594, in _generate_retry\n'
            '    res = _orig_generate(*args, **kwargs)\n'
            '          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 418, in generate\n'
            '    raise e\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 409, in generate\n'
            '    response = completion(\n'
            '               ^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 330, in completion\n'
            '    return inner(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1472, in _completion_cascad'
            'e\n'
            '    raise last\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1465, in _completion_cascad'
            'e\n'
            '    return inner_completion(*args, **{**kwargs, "model": m})\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1510, in _completion_retry\n'
            '    resp = _orig_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/probe/probe.py", line 34, in fake\n'
            '    raise litellm.APIConnectionError(message="user sim down", llm_provider="gemi'
            'ni",\n'
            'litellm.exceptions.APIConnectionError: litellm.APIConnectionError: user sim down\n'),
    },
    'user_empty': {
        "error_type": 'ValueError',
        "error": (
            'UserMessage must have either content or tool_calls. Got UserMessage\n'
            'is_final_chunk: True'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 840, in ste'
            'p\n'
            '    user_msg.validate()\n'
            '  File "/opt/tau2-bench/src/tau2/data_model/message.py", line 287, in validate\n'
            '    raise ValueError(\n'
            'ValueError: UserMessage must have either content or tool_calls. Got UserMessage\n'
            'is_final_chunk: True\n'),
    },
    'judge_outage': {
        "error_type": 'APIConnectionError',
        "error": (
            'litellm.APIConnectionError: judge down'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 76, in run_simulati'
            'on\n'
            '    reward_info = evaluate_simulation(\n'
            '                  ^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/evaluator/evaluator.py", line 217, in evaluate_'
            'simulation\n'
            '    nl_reward_info = NLEvaluator.calculate_reward(\n'
            '                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/evaluator/evaluator_nl_assertions.py", line 46,'
            ' in calculate_reward\n'
            '    nl_assertions_checks = cls.evaluate_nl_assertions(\n'
            '                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/evaluator/evaluator_nl_assertions.py", line 121'
            ', in evaluate_nl_assertions\n'
            '    assistant_message = generate(\n'
            '                        ^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1594, in _generate_retry\n'
            '    res = _orig_generate(*args, **kwargs)\n'
            '          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 418, in generate\n'
            '    raise e\n'
            '  File "/opt/tau2-bench/src/tau2/utils/llm_utils.py", line 409, in generate\n'
            '    response = completion(\n'
            '               ^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 330, in completion\n'
            '    return inner(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1472, in _completion_cascad'
            'e\n'
            '    raise last\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1465, in _completion_cascad'
            'e\n'
            '    return inner_completion(*args, **{**kwargs, "model": m})\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1510, in _completion_retry\n'
            '    resp = _orig_completion(*args, **kwargs)\n'
            '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/probe/probe.py", line 30, in fake\n'
            '    raise litellm.APIConnectionError(message="judge down", llm_provider="gemini"'
            ',\n'
            'litellm.exceptions.APIConnectionError: litellm.APIConnectionError: judge down\n'),
    },
    'tool_harness': {
        "error_type": 'TauHarnessError',
        "error": (
            "harness error, not the agent's outcome: Gemini embeddings failed: 503 during fin"
            'd_user_id_by_zip; the simulation attempt is stopped'),
        "error_traceback": (
            'Traceback (most recent call last):\n'
            '  File "/opt/tau2-bench/src/tau2/runner/progress.py", line 75, in run_with_retry\n'
            '    simulation = run_fn()\n'
            '                 ^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 555, in _execute\n'
            '    return run_single_task(\n'
            '           ^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/batch.py", line 422, in run_single_task\n'
            '    simulation = run_simulation(\n'
            '                 ^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/runner/simulation.py", line 59, in run_simulati'
            'on\n'
            '    simulation = orchestrator.run()\n'
            '                 ^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 280, in run\n'
            '    self.step()\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 881, in ste'
            'p\n'
            '    tool_results = self._execute_tool_calls(self.message.tool_calls)\n'
            '                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/tau2-bench/src/tau2/orchestrator/orchestrator.py", line 325, in _ex'
            'ecute_tool_calls\n'
            '    tool_result = self.environment.get_response(tool_call)\n'
            '                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
            '  File "/opt/gbench/eval_suites/tau_common.py", line 1131, in get_response\n'
            '    raise TauHarnessError(f"harness error, not the agent\'s outcome: {failure} du'
            'ring "\n'
            "gbench_tau_suites.tau_common.TauHarnessError: harness error, not the agent's out"
            'come: Gemini embeddings failed: 503 during find_user_id_by_zip; the simulation a'
            'ttempt is stopped\n'),
    },
}


def _record(name, **over):
    """A simulation as tau2's run_with_retry records the REAL scenario `name`."""
    info = dict(REAL[name], failed_after_attempts=4)
    info.update(over)
    return types.SimpleNamespace(task_id=name, termination_reason="infrastructure_error",
                                 reward_info=None, info=info)


def _ok(task, reward):
    return types.SimpleNamespace(task_id=task, termination_reason="agent_stop",
                                 reward_info=types.SimpleNamespace(reward=reward), info=None)


# =========================================================================== #
# the source of each real record
# =========================================================================== #
@pytest.mark.parametrize("name, source", [
    ("agent_empty", "agent_reply"),             # AssistantMessage.validate(): no text, no tools
    ("agent_badjson", "agent_reply"),           # json.loads of the agent's tool arguments
    ("agent_ctx", "agent_context_overflow"),    # the agent's request over the context window
    ("agent_down", "model_endpoint"),           # connection refused
    ("agent_500", "model_endpoint"),            # HTTP 500 from the endpoint
    ("user_outage", "user_simulator"),          # the Gemini user simulator's call failed
    ("user_empty", "user_simulator"),           # the Gemini user simulator's empty reply
    ("judge_outage", "grading"),                # the Gemini NL-assertion judge's call failed
    ("tool_harness", "gemini_embeddings"),      # TauHarnessError out of the tool call
])
def test_each_real_record_is_attributed(name, source):
    assert TC._infra_error_source(_record(name)) == source
    agent_side = source in TC._AGENT_TURN_SOURCES
    assert agent_side is name.startswith(("agent_empty", "agent_badjson", "agent_ctx"))


def test_the_agent_turn_sources_are_exactly_the_agents_own():
    assert TC._AGENT_TURN_SOURCES == ("agent_reply", "agent_context_overflow")


def test_a_tool_exception_that_is_not_the_embedding_guard_is_a_tool_failure():
    assert TC._infra_error_source(_record("tool_harness", error_type="RuntimeError")) == "tool"


def test_a_context_overflow_is_read_from_the_class_or_the_message():
    by_text = _record("agent_ctx", error_type="BadRequestError",
                      error="litellm.BadRequestError: ContextWindowExceededError: too long")
    assert TC._infra_error_source(by_text) == "agent_context_overflow"
    # the same frames with an endpoint error that is not an overflow
    down = _record("agent_ctx", error_type="APIConnectionError", error="refused")
    assert TC._infra_error_source(down) == "model_endpoint"


def test_the_traceback_of_the_caught_exception_is_the_one_read():
    """format_exc() prints chained causes first; a cause's frames (here a user-simulator frame)
    and a traceback quoted inside the error message must not decide the source."""
    cause = ("Traceback (most recent call last):\n"
             '  File "/opt/tau2-bench/src/tau2/user/user_simulator.py", line 1, in x\n'
             "    y()\n"
             "httpx.ConnectError: refused\n\n"
             "The above exception was the direct cause of the following exception:\n\n")
    quoted = ("\nRemote says: Traceback (most recent call last):\n"
              '  File "/srv/app.py", line 3, in handler\n'
              "    raise Boom()\n")
    rec = _record("agent_down",
                  error_traceback=cause + REAL["agent_down"]["error_traceback"] + quoted)
    assert TC._infra_error_source(rec) == "model_endpoint"
    frames = TC._traceback_frames(rec.info["error_traceback"])
    assert frames[0][0].endswith("/tau2/runner/progress.py")
    assert frames[0][1:] == ("run_with_retry", "simulation = run_fn()")


def test_records_without_a_usable_traceback():
    assert TC._infra_error_source(_record("agent_down", error_traceback="")) == "unrecorded"
    assert TC._infra_error_source(types.SimpleNamespace(info=None)) == "unrecorded"
    only_runner = "\n".join(REAL["agent_down"]["error_traceback"].splitlines()[:3]) + \
        "\nRuntimeError: x\n"
    assert TC._infra_error_source(_record("agent_down", error_traceback=only_runner)) == "other"


def test_the_summary_carries_sources_and_causes():
    sims = [_record("agent_empty"), _record("agent_down"), _record("agent_down")]
    entry = TC._domain_infra_entry(sims)
    assert entry["count"] == 3
    assert entry["sources"] == {"agent_reply": 1, "model_endpoint": 2}
    assert entry["causes"] == {"agent_reply/ValueError": 1,
                               "model_endpoint/APIConnectionError": 2}
    assert entry["harness"]["count"] == 2 and entry["agent_turn"]["count"] == 1
    assert entry["agent_turn"]["example_error"].startswith("AssistantMessage must have either")
    assert entry["harness"]["example_error"] == "litellm.APIConnectionError: Connection refused"


# =========================================================================== #
# the run result
# =========================================================================== #
_TAU_ENV = ("GBENCH_TAU2_TEMPERATURE", "TAU2_TEMPERATURE", "GBENCH_TAU3_TEMPERATURE",
            "GBENCH_TAU2_TOP_P", "TAU2_TOP_P", "GBENCH_TAU2_TOP_K", "TAU2_TOP_K",
            "GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
            "GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", "GBENCH_TAU2_USER_TEMPERATURE",
            "TAU2_USER_TEMPERATURE", "GBENCH_TAU2_SAVE_TRACES", "TAU2_SAVE_TRACES",
            "GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENAI_API_KEY", TC._AGENT_KEY_ENV)


def _tau2_modules(sims_by_domain):
    """TextRunConfig, run_domain (writes results.json, returns the canned simulations) and
    compute_metrics (pass^1 over the simulations that did not end in an infrastructure error)."""
    names = ["tau2", "tau2.data_model", "tau2.data_model.simulation", "tau2.run",
             "tau2.metrics", "tau2.metrics.agent_metrics"]
    m = {n: types.ModuleType(n) for n in names}

    class TextRunConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def run_domain(config):
        os.makedirs(config.save_to, exist_ok=True)
        with open(os.path.join(config.save_to, "results.json"), "w") as fh:
            fh.write("{}")
        return types.SimpleNamespace(simulations=list(sims_by_domain[config.domain]))

    def compute_metrics(results):
        scored = [s.reward_info.reward for s in results.simulations if s.reward_info is not None]
        return types.SimpleNamespace(pass_hat_ks={1: sum(scored) / len(scored)} if scored else {})

    m["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
    m["tau2.run"].run_domain = run_domain
    m["tau2.metrics.agent_metrics"].compute_metrics = compute_metrics
    return m


@pytest.fixture
def tau(monkeypatch, tmp_path):
    for name in _TAU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv(TC._AGENT_KEY_ENV, ENDPOINT_KEY)
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", str(tmp_path / "traces"))
    for name, value in (("_SECRETS", []), ("_AGENT_ROUTE", {}), ("_LAST_RETRIEVAL", {}),
                        ("_LLM_FAILURES", {}), ("_EMBEDDING_FAILURES", {}),
                        ("_EMBEDDING_ABORTS", {}), ("_DOMAIN_INFRA", {}),
                        ("_CURRENT_DOMAIN", {"domain": None}),
                        ("_EMBED_FAILURE_SEEN", threading.local())):
        monkeypatch.setattr(TC, name, value)
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("check_banking_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_install_credential_guards", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1")):
        monkeypatch.setattr(TC, name, value)

    def run(sims_by_domain, eval_name="tau2"):
        with mock.patch.dict(sys.modules, _tau2_modules(sims_by_domain)):
            return TC.run_tau_env(eval_name, list(sims_by_domain), "m", "http://x/v1", 1, None,
                                  f"docs/evals/{eval_name}.md")
    return run


def _gaps(r):
    return (r["leaderboard_comparable_reason"] or "").split("; ")


def test_an_empty_agent_reply_is_not_labelled_a_harness_error(tau):
    """The review's case: retail with 3 scored simulations (2 perfect) and 1 that tau2 recorded as
    an infrastructure error because the agent replied with neither text nor tool calls."""
    r = tau({"retail": [_ok(0, 1.0), _ok(1, 0.0), _ok(2, 1.0), _record("agent_empty")]})
    # canonical: tau2 leaves it out of pass^k; the run still says so
    assert r["accuracy"] == round(100.0 * 2 / 3, 2)
    assert r["status"] == "completed_with_errors"
    assert r["harness_errors"] == []
    assert r["agent_turn_errors"] == [{
        "error": ("simulations ended in an infrastructure error raised on the agent's own turn, "
                  "not a harness failure (tau2 records one when every attempt of a simulation "
                  "failed; compute_metrics leaves it out of pass^k, as upstream does)"),
        "domain": "retail", "infrastructure_error_simulations": 1,
        "sources": {"agent_reply": 1}, "error_types": {"ValueError": 1},
        "example_error": ("AssistantMessage must have either content or tool_calls. Got "
                          "AssistantMessage is_final_chunk: True")}]
    assert ("1 simulation(s) ended in infrastructure errors raised on the agent's own turn "
            "(retail 1 [agent_reply/ValueError x1]), not harness failures: tau2's metrics "
            "exclude them as upstream does, so pass^k covers fewer than every task x trial and "
            "does not count these simulations against the model") in _gaps(r)
    reason = r["leaderboard_comparable_reason"]
    assert "not the agent's outcome" not in reason and "from harness failures" not in reason
    assert r["leaderboard_comparable"] is False


def test_the_counts_are_over_the_evaluable_simulations(tau):
    """The review's second case: the same retail run printed '2/4 (66.67%)'."""
    r = tau({"retail": [_ok(0, 1.0), _ok(1, 0.0), _ok(2, 1.0), _record("agent_down")]})
    assert r["category_accuracy"] == {"retail": {"correct": 2, "total": 3, "accuracy": 66.67,
                                                 "simulations": 4, "infra_errors": 1}}
    assert (r["correct_answers"], r["total_questions"]) == (2, 3)
    rep = r["tau2_report"]
    assert (rep["total_simulations"], rep["evaluable_simulations"],
            rep["infra_errors"]) == (4, 3, 1)
    assert rep["perfect_tasks"] == 2 and rep["mean_reward"] == round(2 / 3, 4)
    # the per-category line evals.py prints
    s = r["category_accuracy"]["retail"]
    assert f"{s.get('correct', 0)}/{s.get('total', 0)}" == "2/3"


def test_a_clean_run_keeps_its_counts(tau):
    r = tau({"airline": [_ok(0, 1.0), _ok(1, 0.0)], "retail": [_ok(0, 1.0), _ok(1, 1.0)]})
    assert r["status"] == "success"
    assert r["harness_errors"] == [] and r["agent_turn_errors"] == []
    assert (r["correct_answers"], r["total_questions"]) == (3, 4)
    assert r["category_accuracy"]["airline"] == {"correct": 1, "total": 2, "accuracy": 50.0,
                                                 "simulations": 2, "infra_errors": 0}
    assert not any("infrastructure errors" in g for g in _gaps(r))


def test_harness_and_agent_turn_errors_are_reported_apart(tau):
    outage = dict(REAL["user_outage"])
    outage["error"] = f"{outage['error']} (key={GEMINI_KEY})"
    r = tau({"airline": [_ok(0, 1.0), _record("agent_ctx"), _record("judge_outage")],
             "retail": [_ok(0, 1.0), _record("user_outage", error=outage["error"]),
                        _record("agent_badjson"), _record("agent_500")],
             "telecom": [_ok(0, 0.0), _ok(1, 1.0)]})
    assert r["status"] == "completed_with_errors"
    assert r["tau2_report"]["infra_errors"] == 5
    harness = {h["domain"]: h for h in r["harness_errors"]}
    agent = {h["domain"]: h for h in r["agent_turn_errors"]}
    assert set(harness) == {"airline", "retail"} and set(agent) == {"airline", "retail"}
    assert harness["airline"]["sources"] == {"grading": 1}
    assert harness["retail"]["sources"] == {"user_simulator": 1, "model_endpoint": 1}
    assert harness["retail"]["infrastructure_error_simulations"] == 2
    assert agent["airline"]["sources"] == {"agent_context_overflow": 1}
    assert agent["retail"]["sources"] == {"agent_reply": 1}
    assert agent["retail"]["error_types"] == {"JSONDecodeError": 1}
    gaps = _gaps(r)
    assert ("3 simulation(s) ended in infrastructure errors from harness failures, not the "
            "agent's outcome (airline 1 [grading/APIConnectionError x1], retail 2 "
            "[model_endpoint/InternalServerError x1, user_simulator/APIConnectionError x1]): "
            "tau2's metrics exclude them, so pass^k covers fewer than every task x trial") in gaps
    assert any(g.startswith("2 simulation(s) ended in infrastructure errors raised on the agent's "
                            "own turn (airline 1 [agent_context_overflow/ContextWindowExceededError"
                            " x1], retail 1 [agent_reply/JSONDecodeError x1]), not harness")
               for g in gaps)
    # evaluable counts per domain
    assert r["category_accuracy"]["retail"] == {"correct": 1, "total": 1, "accuracy": 100.0,
                                                "simulations": 4, "infra_errors": 3}
    assert (r["correct_answers"], r["total_questions"]) == (3, 4)
    dumped = json.dumps(r)
    assert GEMINI_KEY not in dumped and ENDPOINT_KEY not in dumped


def test_an_agent_turn_error_alone_still_makes_the_run_completed_with_errors(tau):
    r = tau({"retail": [_ok(0, 1.0), _record("agent_ctx")]})
    assert r["status"] == "completed_with_errors"
    assert r["harness_errors"] == [] and len(r["agent_turn_errors"]) == 1


def test_a_domain_lost_on_the_agents_turn_hard_errors_saying_so(tau):
    with pytest.raises(RuntimeError) as ei:
        tau({"retail": [_record("agent_empty"), _record("agent_empty"), _record("agent_down")]})
    msg = str(ei.value)
    assert msg.startswith("[INFRA REQUIRED]")
    assert ("(all 3 ended in an infrastructure error: ValueError x2, APIConnectionError x1; 2 of "
            "them raised on the agent's own turn (agent_reply/ValueError x2), not harness "
            "failures)") in msg


def test_the_status_rule_and_the_new_fields_are_in_the_code():
    src = inspect.getsource(TC._run_domains)
    assert '"completed_with_errors" if (harness_errors or infra_all) else "success"' in src
    assert '"agent_turn_errors": agent_turn_errors' in src
    assert '"total_questions": total - infra_all' in src


# =========================================================================== #
# docs
# =========================================================================== #
def _read(*parts):
    return " ".join((ROOT.joinpath(*parts)).read_text(encoding="utf-8").split())


def test_the_docs_describe_the_sources_and_the_counts():
    tau2, tau3 = _read("docs", "evals", "tau2.md"), _read("docs", "evals", "tau3.md")
    for text in ("**Infrastructure errors are never scored.**", "`agent_turn_errors`",
                 "`agent_reply`", "`agent_context_overflow`", "`model_endpoint`",
                 "`user_simulator`", "`grading`", "`tool`", "`gemini_embeddings`", "`other`",
                 "`unrecorded`", "`sources`", "`tau2_report.evaluable_simulations`",
                 "`tau2_report.total_simulations` counts every simulation",
                 "a message with neither text nor tool calls, or tool-call arguments that are "
                 "not valid JSON",
                 "infrastructure errors raised on the agent's own turn (<domain> <n> "
                 "[<source>/<error class> x<n>], ...), not harness failures: tau2's metrics "
                 "exclude them as upstream does",
                 "so an infrastructure error never reads as a failed simulation"):
        assert text in tau2, text
    assert "harness errors that are not the agent's outcome (<domain>" not in tau2
    assert "perfect / total simulation counts" not in tau2
    for text in ("`agent_turn_errors`", "`evaluable_simulations`",
                 "are the perfect (reward >= 1) and evaluable simulations",
                 "is listed apart as not a harness failure"):
        assert text in tau3, text
    assert "are the perfect (reward >= 1) and total simulations" not in tau3
    for doc in ("tau2.md", "tau3.md"):
        text = (ROOT / "docs" / "evals" / doc).read_text(encoding="utf-8")
        new = [l for l in text.splitlines() if "agent_turn" in l or "evaluable_sim" in l]
        assert new and all(l.isascii() and len(l) <= 100 for l in new), doc


def test_the_suite_docstrings_follow_the_code():
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    for mod in (T2, T3, TC):
        doc = " ".join((mod.__doc__ or "").split())
        assert "agent_turn_errors" in doc, mod.__name__
        assert "raised on the agent's own turn" in doc, mod.__name__
