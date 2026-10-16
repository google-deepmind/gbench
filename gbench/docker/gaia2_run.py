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
#
# gbench launcher for GAIA2 / Meta-ARE, run INSIDE the gbench-gaia2 image. It drives the pinned
# `are-benchmark` CLI (verified against the 1.2.0 wheel: cli.py + cli/shared_params.py) and parses
# the emitted benchmark_stats.json - it NEVER fabricates a score.
#
# Modes (GAIA2_MODE):
#   run       (default) - `are-benchmark run` with NO --hf-config runs ALL 5 GAIA2 capability
#                         configs (adaptability/ambiguity/execution/search/time via
#                         CapabilityTag.gaia2_capabilities()), the standard phase only, --num_runs
#                         runs per scenario (ARE's default 3), and writes ONE aggregated
#                         benchmark_stats.json. A single --hf-config (GAIA2_CONFIG) narrows it.
#   gaia2-run           - the full canonical leaderboard submission: 3 phases (Standard / A2A-mini /
#                         Noise-mini) x --num_runs (default 3), Pass@k/Pass^k. Very heavy; opt-in.
#
# Sharding (--shard I/N, GAIA2_SHARD="I/N" from the gbench runner, which validated it): ARE's CLI
# has no shard flag, and its HF loader (huggingface_loader._create_huggingface_scenario_iterator)
# streams load_dataset(<dataset>, name=<config>, streaming=True)[split] and stops at --limit
# (first-N per config). In the hooked ARE process, install_shard_hook wraps datasets.load_dataset
# for the scenario dataset: each loaded config's split keeps only shard I of N of its scenario ids
# (the ids sorted, round-robin: shard_scenario_ids), as an IterableDataset filter in the stream's
# own order, so ARE's --limit then takes the first N of the shard and every phase and run of a
# kept scenario runs as upstream. Each selection is recorded (gbench_shard.jsonl, the summary's
# `shard`). When no configured capability has a scenario in the shard, ARE is not started and the
# summary says so (`empty_shard`). See gaia2.py + docs/evals/gaia2.md.
# --eval-limit (GAIA2_CONFIG_LIMITS, both modes): the gbench runner deals --eval-limit N over the
# run's configs (N scenarios for the whole run, never N per config) and passes {config: share}; the
# same hook keeps a seeded draw of each config's share of its (shard's) sorted scenario ids
# (select_scenario_ids, LIMIT_DRAW), and a config whose share is 0 is not run: the run_dataset hook
# raises LimitSkipped before ARE loads it (ARE logs it as a failed config and goes on; the
# launcher records it under `limit_skipped`, never as a config that raised). GAIA2_LIMIT (the
# operator's GBENCH_GAIA2_LIMIT, `run` mode) is the older knob: that many scenarios of EACH config,
# a seeded draw passed on as ARE's --limit. The summary's `selection` records either.
#
# Model-under-test wiring (litellm_engine.py: provider "local" -> custom_llm_provider=None +
# api_base=endpoint): pass --model openai/<served> so LiteLLM routes to the OpenAI-compat /v1. The
# endpoint is gbench's relay on the docker network's gateway (it adds the endpoint's key; the
# container holds none).
# Judge: always --judge_provider local --judge_endpoint http://127.0.0.1:$GAIA2_JUDGE_PORT/v1
# --judge_model openai/gbench-cascade, the in-container Gemini judge proxy (gaia2_cascade_judge.py);
# ARE's judge PROMPTS are untouched, only the model call is swapped. Nothing moves the judge
# elsewhere.
#
# Env contract (set by the entrypoint / docker run -e):
#   GAIA2_WORKDIR         output dir (mounted; holds benchmark_stats.json + our summary). default /out
#   GBENCH_MODEL_NAME     served model name (sent as openai/<name>)
#   GBENCH_MODEL_BASE_URL model /v1 endpoint (gbench's relay on the docker network's gateway)
#   GAIA2_JUDGE_PORT      Gemini judge proxy port (default 18790)
#   GAIA2_MODE            run | gaia2-run    (default run)
#   GAIA2_CONFIG          single capability config (narrows `run`; e.g. mini/demo for smoke)
#   GAIA2_LIMIT           scenarios per config (GBENCH_GAIA2_LIMIT; `run` mode): a seeded draw of
#                         the config's (shard's) scenario ids (LIMIT_DRAW), not ARE's first N
#   GAIA2_CONFIG_LIMITS   --eval-limit dealt over the configs, JSON {config: scenarios} (both modes;
#                         a config with 0 is not run); takes the place of GAIA2_LIMIT
#   GAIA2_NUM_RUNS        runs per scenario, both modes (ARE's default 3)
#   GAIA2_HF_DATASET      dataset passed to --hf-dataset (default GAIA2_LOCAL_DATASET, the image's
#                         local copy of meta-agents-research-environments/gaia2; see "Dataset")
#   GAIA2_SPLIT           split (default validation)
#   GBENCH_DETOKENIZE_URL gbench's token-id decoder on the docker network's gateway (the server-parser
#                         policy below; unset: token ids are not decoded)
#   GAIA2_SCENARIO_TIMEOUT / GAIA2_MAX_CONCURRENT  passthrough tuning
#   GAIA2_SHARD           "I/N": run shard I of N of each config's scenarios (see Sharding)
#   The agent protocol (the gbench runner passes the values it resolved):
#   GAIA2_AGENT_TEMPERATURE  the agent's temperature ("" sends none; default 0.5)
#   GAIA2_AGENT_MAX_TOKENS   the agent's max_tokens per turn (default 16000)
#   GAIA2_ENABLE_THINKING    true | false: the chat template's thinking switch (unset: not sent)
#   GAIA2_AGENT_TIMEOUT_S    seconds one agent request may take (default LiteLLM's 600)
#   GAIA2_MAX_ITERATIONS     the main agent's step limit (default 200)
#
# Dataset: the image holds local copies of meta-agents-research-environments/gaia2 and
# gaia2_filesystem at pinned revisions (gaia2.Dockerfile) and runs with HF_HUB_OFFLINE=1 and
# HF_DATASETS_OFFLINE=1: ARE's loader (huggingface_loader.py, load_dataset(..., streaming=True))
# reads the local gaia2 copy (--hf-dataset GAIA2_LOCAL_DATASET), and the filesystem app's fallback
# (config.DEMO_FS_PATH, set by the image) reads the local gaia2_filesystem copy, so a run makes no
# Hugging Face Hub request (anonymous Hub requests were rate-limited with 429s that dropped whole
# capability configs). The summary records the dataset, its revision and the local path.
#
# The agent protocol (ARE paper arXiv:2509.17158, section 4.1: "temperature set to 0.5, and
# generation limits of 16,000 tokens per turn", at most 200 steps). ARE 1.2.0's LiteLLMEngine sends
# none of it (its completion() call carries only model, messages, api_base and api_key; the
# ModelConfig defaults max_tokens=16384 / temperature=0.5 are never forwarded), and its main agent
# stops at 80 iterations (ARESimulationAgent max_iterations=80). In the hooked ARE process the
# agent's requests carry the temperature, the per-turn max_tokens, the chat template's thinking
# switch and the agent's stop sequences (['<end_action>', 'Observation:'], which ARE's agent passes
# to its engine on every step; the paper, appendix B.5, adds them as custom stop sequences "for
# models that tend to continue past a single tool call (Claude, Kimi, Qwen)" and ran GPT-5 without
# them; gbench sends them for every served model, as ARE's HuggingFaceLLMEngine does), and the
# main agent gets the 200-step limit. The stop sequences end the answer, not the reasoning:
#   * on a run without thinking they are sent as `stop`, as ARE's own HuggingFaceLLMEngine sends
#     them, and the server ends the reply at the first one;
#   * on a --thinking run the served stack (vLLM's detokenizer) would match a `stop` over the
#     whole generated text, the reasoning included, so a thought containing 'Observation:' would
#     end the reply before any answer. So no `stop` is sent; the request is streamed (vLLM sends
#     the reasoning and the answer in separate fields) and the client closes it once the answer
#     holds a stop sequence, which ends the generation on the server (vLLM aborts a request whose
#     client went away). The reply then ends where a server-side stop on the answer would end it,
#     and the measured generation time (below) stops there too.
# ARE's own client-side split of the answer at the first stop sequence applies on both runs.
# Thinking-run replies that stopped with no answer and a non-empty reasoning field are counted
# (stopped_in_reasoning), and so are replies whose answer still held a stop sequence
# (ran_past_stop: the server did not end the reply at it) and replies closed at one
# (stopped_client_side). A streamed request that runs longer than the request timeout
# (GAIA2_AGENT_TIMEOUT_S) is closed and raises LiteLLM's Timeout (an endpoint failure) the first
# time it runs over; a non-streamed request that runs that long raises the same Timeout, but only
# after LiteLLM's client has re-sent it (max_retries=2).
#
# The server-parser policy (gbench's base.recover_server_parser_text; the three pure functions are
# copied below and a test holds them equal to base's): every agent request asks the served
# endpoint for the generated token ids (return_token_ids; LiteLLM drops a choice's token_ids and
# stop_reason when it converts the reply, so a hook keeps them). On a run without thinking, text
# the server's reasoning parser put in the reasoning field (a thought channel the model opened on
# its own, or the parser's first-feed rule) is the model's output; in either mode, a reply with no
# content, no reasoning and completion_tokens > 0 lost its text to the server's tool-call parser (a
# native tool call on a request that declared no tools: ARE declares none, and reads only the
# content, so a tool call the server returned is equally unseen). Both are recovered in generation
# order, the token ids decoded with special tokens kept (gbench's decoder, GBENCH_DETOKENIZE_URL:
# the served tokenizer's /detokenize, else the run's tokenizer), else (moved text only) the
# reasoning followed by the content, and handed to ARE as the reply, which parses it as it parses
# any reply (the model's outcome: a native tool call gets ARE's format-error feedback). A reply that
# kept its content or reasoning is checked too (base.server_parser_residue, copied below): the
# returned reasoning and content are located in the generation, and generated text after them
# outside tag markup is text the tool-call parser deleted (server_parser_dropped_text): on a
# --thinking reply whose answer was a native tool call, the text after the thought is handed to
# ARE. Each event is counted (server_parser_moved_text, server_parser_dropped_text) and recorded
# with the call, the recovered text capped; the token ids of every reply are decoded, and every
# reply without content, and every recovered one, keeps the text it was rebuilt from
# (generation_text; null only when the reply carried no token ids or they could not be decoded).
# Deleted text that cannot be recovered (no token ids, or undecodable) is a harness error
# (server_parser_text_lost), and ARE gets no reply (LiteLLMEngine's assert, as upstream). A
# --thinking reply that ended inside the reasoning channel with nothing generated after it stays
# the model's outcome (stopped_in_reasoning).
#
# The Time capability clock (paper section 4.3: the environment's time is paused while the model
# generates and then "incremented by the generation duration measured on the client side"): ARE's
# `measured` mode advances the clock by the engine's `completion_duration`, which LiteLLMEngine
# never returns (it returns no metadata, so the clock did not move). The hooked agent engine
# returns the client-measured wall time of the request that answered, up to the end of the reply
# (for a streamed reply, up to the chunk that completed the stop sequence), with the reply's token
# counts (a streamed request asks vLLM for the usage so far in every chunk:
# stream_options.continuous_usage_stats).
#
# The judge path (upstream issue facebookresearch/meta-agents-research-environments#29, PR #26):
# ARE 1.2.0's LiteLLMEngine rewrites every reply with `.replace("False", "false").replace("True",
# "true")` (for the agent's JSON actions), and create_judge_engine always builds that engine, so a
# judge verdict "[[True]]" arrives as "[[true]]" and LLMChecker's case-sensitive
# `success_str in response` never matches for the signature, sanity, cab and tone checkers. The
# current upstream judge (gaia2-cli: gaia2_cli/judge/engine.py create_litellm_engine returns the
# reply content verbatim, at temperature 0; gaia2_core/judge/checkers.py LLMChecker matches
# success_str / failure_str case-insensitively) has neither defect. The hooked judge engine returns
# the reply verbatim and LLMChecker matches as gaia2_core does; ARE's prompts and every other check
# are unchanged. The agent keeps ARE's rewrite (its JSON action parser needs it).
#
# Harness errors (rule: a failure of the harness or of a service is never scored as the model's
# outcome). ARE turns a judge exception into a failed validation (validation/base.py
# validate_current_turn: success=False, rationale "Exception") and an agent run that raised into a
# failure (scenario_runner.py), and its agent loop logs a failed model call and goes on, so its
# success rates count all of them as the model's failures. So ARE runs in a child process started
# from this file (`gaia2_run.py --are-hooked <are-benchmark arguments>`) with gbench's hooks:
#   * every LLM call ARE makes through LiteLLM (litellm_engine.completion: the agent's calls to the
#     model endpoint and the judge's calls to the proxy) that raises is recorded with its scenario
#     run (ARE's per-thread scenario id and run number, and the config and phase being run) in
#     gbench_llm_call_failures.jsonl; so is a soft check whose judge reply held no verdict, and an
#     agent reply whose text the server's tool-call parser deleted beyond recovery;
#   * every agent call is recorded (finish reason, vLLM's stop_reason, token counts, duration,
#     whether the content was empty, the server-parser event and the recovered text, capped) in
#     gbench_agent_calls.jsonl, and every soft check (checker, verdict, the Gemini model that
#     answered, the judge's reply) in gbench_judge_verdicts.jsonl;
#   * when ARE writes its report (benchmark_stats.json, unchanged), gbench also classifies every
#     scenario run (ARE's own per-run rows, report_stats.combine_results_to_dataframe) and writes
#     gbench_stats.json: ARE's statistics (report_stats.calculate_statistics) recomputed over the
#     runs without a harness error, the runs by kind, and the failed calls no run could be tied to.
# A run is a harness error (HARNESS_ERROR_KINDS) when a model endpoint call of its agent failed on
# the endpoint's side (model_endpoint), the server's tool-call parser deleted an agent reply beyond
# recovery (server_parser_text_lost), a judge call of its validation raised (judge), a judge
# reply held no verdict (judge_no_verdict), its validation raised otherwise
# (validation_exception), or its run raised (run_exception); runs ARE could not validate
# (no_validation) are left out by ARE itself. A scenario timeout is the model's outcome, as ARE and
# the leaderboard count it. A run with a harness error is left out whatever ARE validated it as: a
# failed pairing changes which agent event ARE's greedy event matching (validation/judge.py
# _match_agent_oracle_event) gives each later oracle event, and a failed agent call puts an error
# into the agent's history, so the error's effect on the outcome cannot be told; the runs ARE
# validated as successes among them are counted (n_harness_error_success_runs).
#
# Capability configs (ARE's cli.py and gaia2_submission.py run each config in a try/except, log a
# config that raised and report the others with exit code 0): every run_dataset call is recorded
# with its config, phase and outcome in gbench_configs.jsonl, and gbench_stats.json carries the
# configs that raised (failed_configs). ARE's macro rate gives a capability without a validated
# run a success rate of 0.0 (report_stats._calculate_success_rate_stats); gbench's recomputed macro
# averages the capabilities with a validated run (the run-level std/sem ARE computes already do)
# and names the others (capabilities_without_validated_run), so the gbench runner can name every
# capability the macro leaves out.

import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time

WORKDIR = os.environ.get("GAIA2_WORKDIR", "/out")
#: The GAIA2 dataset on the Hugging Face Hub, and the image's local copy of it (gaia2.Dockerfile).
HUB_DATASET = "meta-agents-research-environments/gaia2"
LOCAL_DATASET = os.environ.get("GAIA2_LOCAL_DATASET", "")
DATASET = os.environ.get("GAIA2_HF_DATASET") or LOCAL_DATASET or HUB_DATASET
SPLIT = os.environ.get("GAIA2_SPLIT", "validation")

#: The ARE paper's agent protocol (arXiv:2509.17158 section 4.1).
PAPER_AGENT_TEMPERATURE = 0.5
PAPER_AGENT_MAX_TOKENS = 16000
PAPER_MAX_ITERATIONS = 200
#: LiteLLM's default request timeout (litellm 1.71.1 main.py: 600 s), which ARE's engine keeps.
LITELLM_DEFAULT_TIMEOUT_S = 600
#: The agent's stop sequences (are/simulation/agents/default_agent/base_agent.py step()).
AGENT_STOP_SEQUENCES = ("<end_action>", "Observation:")


def _model_arg():
    name = os.environ.get("GBENCH_MODEL_NAME", "served")
    return name if name.startswith("openai/") else f"openai/{name}"


#: The judge model tag ARE sends to the proxy (the proxy runs the Gemini cascade and ignores it).
JUDGE_MODEL = "openai/gbench-cascade"


def _judge_args():
    jport = os.environ.get("GAIA2_JUDGE_PORT", "18790")
    return ["--judge_provider", "local", "--judge_endpoint", f"http://127.0.0.1:{jport}/v1",
            "--judge_model", JUDGE_MODEL]


def _base_cmd(command):
    cmd = ["are-benchmark", command,
           "--model", _model_arg(),
           "--provider", "local",
           "--endpoint", os.environ.get("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
           "--hf-dataset", DATASET,
           "--hf-split", SPLIT,
           "--output_dir", WORKDIR]
    cmd += _judge_args()
    if os.environ.get("GAIA2_SCENARIO_TIMEOUT"):
        cmd += ["--scenario_timeout", os.environ["GAIA2_SCENARIO_TIMEOUT"]]
    if os.environ.get("GAIA2_MAX_CONCURRENT"):
        cmd += ["--max_concurrent_scenarios", os.environ["GAIA2_MAX_CONCURRENT"]]
    return cmd


def agent_settings(env=None):
    """The agent protocol this run applies, from the GAIA2_AGENT_* variables (see the file
    header): {temperature (None: none sent), max_tokens, enable_thinking (None: not sent),
    timeout_s, max_iterations}."""
    env = os.environ if env is None else env
    raw_t = env.get("GAIA2_AGENT_TEMPERATURE")
    temperature = PAPER_AGENT_TEMPERATURE if raw_t is None else (
        float(raw_t) if raw_t.strip() else None)
    raw = (env.get("GAIA2_AGENT_MAX_TOKENS") or "").strip()
    max_tokens = int(raw) if raw else PAPER_AGENT_MAX_TOKENS
    switch = (env.get("GAIA2_ENABLE_THINKING") or "").strip().lower()
    enable_thinking = {"true": True, "false": False}.get(switch)
    raw = (env.get("GAIA2_AGENT_TIMEOUT_S") or "").strip()
    timeout_s = int(raw) if raw else LITELLM_DEFAULT_TIMEOUT_S
    raw = (env.get("GAIA2_MAX_ITERATIONS") or "").strip()
    max_iterations = int(raw) if raw else PAPER_MAX_ITERATIONS
    return {"temperature": temperature, "max_tokens": max_tokens,
            "enable_thinking": enable_thinking, "timeout_s": timeout_s,
            "max_iterations": max_iterations}


#: The stream options of a streamed agent request: the usage at the end, and vLLM's usage so far
#: in every chunk, so a reply closed at a stop sequence still has its token counts.
STREAM_OPTIONS = {"include_usage": True, "continuous_usage_stats": True}


def server_stop(settings):
    """True when the agent's stop sequences are sent to the server as `stop`: on a run without
    thinking. The server matches stop strings over the reasoning too, so a --thinking run sends
    none and ends the answer at them on the client (client_stop; see the file header)."""
    return settings.get("enable_thinking") is not True


def client_stop(settings, stop_sequences):
    """True when the agent request is streamed and closed at the first stop sequence in the
    answer: a --thinking run (server_stop False) whose request carries stop sequences."""
    return bool([s for s in (stop_sequences or []) if s]) and not server_stop(settings)


def agent_completion_kwargs(settings, stop_sequences):
    """The fields an agent request carries on top of ARE's own (model, messages, api_base,
    api_key): the temperature, max_tokens, the stop sequences as `stop` (server_stop) or the
    streamed request that the client closes at them (client_stop: stream, stream_options), the
    request timeout and the chat template's thinking switch."""
    kw = {"max_tokens": settings["max_tokens"], "timeout": settings["timeout_s"]}
    if settings.get("temperature") is not None:
        kw["temperature"] = settings["temperature"]
    stops = [s for s in (stop_sequences or []) if s]
    if stops and server_stop(settings):
        kw["stop"] = stops
    elif client_stop(settings, stops):
        kw["stream"] = True
        kw["stream_options"] = dict(STREAM_OPTIONS)
    if settings.get("enable_thinking") is not None:
        kw["extra_body"] = {"chat_template_kwargs": {
            "enable_thinking": bool(settings["enable_thinking"])}}
    return kw


class StreamDeadline(Exception):
    """A streamed agent reply was still running when the request timeout ran out."""


def _field(obj, name):
    """`obj.name` for an object or `obj[name]` for a dict; None when absent."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def read_stream(chunks, stop_sequences, deadline=None, clock=None):
    """Read a streamed agent reply (the served endpoint's chat.completion.chunk objects, or dicts
    of the same shape) until its answer holds one of `stop_sequences` or the stream ends. The
    reasoning channel (`reasoning`, or the older `reasoning_content`) is collected, never matched.
    Raises StreamDeadline when a chunk arrives after `deadline` (a `clock()` value; the clock is
    time.monotonic unless one is given). Returns
    {content (None when no chunk carried answer text: vLLM's first chunk always carries
    content '', which is not answer text), reasoning (the reasoning text, or None),
    reasoning_chars, finish_reason (the server's; "stop" when the reply was closed at a stop
    sequence), stop_sequence (the one that closed it, or None), stop_reason (vLLM's, from the
    last chunk that carried one), token_ids (the generated token ids the chunks carried with
    return_token_ids, or None), tool_call_deltas (chunks whose delta carried tool calls), usage
    (the last usage the stream reported, or None)}."""
    clock = clock or time.monotonic
    stops = [s for s in (stop_sequences or []) if s]
    longest = max((len(s) for s in stops), default=0)
    content, seen, reasoning_parts = "", False, []
    finish, usage, hit, stop_reason = None, None, None, None
    token_ids, tool_deltas = None, 0
    for chunk in chunks:
        u = _field(chunk, "usage")
        if u is not None:
            usage = u
        for choice in _field(chunk, "choices") or []:
            if (_field(choice, "index") or 0) != 0:
                continue
            ids = _field(choice, "token_ids")
            if isinstance(ids, (list, tuple)):
                token_ids = (token_ids or []) + list(ids)
            delta = _field(choice, "delta")
            reasoning = _field(delta, "reasoning") or _field(delta, "reasoning_content")
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
            if _field(delta, "tool_calls"):
                tool_deltas += 1
            piece = _field(delta, "content")
            if isinstance(piece, str) and piece:
                seen = True
                if stops and hit is None:
                    start = max(0, len(content) - longest + 1)
                    content += piece
                    window = content[start:]
                    found = [(window.find(s), s) for s in stops if s in window]
                    if found:
                        hit = min(found)[1]
                else:
                    content += piece
            if _field(choice, "finish_reason"):
                finish = _field(choice, "finish_reason")
            if _field(choice, "stop_reason") is not None:
                stop_reason = _field(choice, "stop_reason")
        if hit is not None:
            finish = "stop"
            break
        if deadline is not None and clock() > deadline:
            raise StreamDeadline("the streamed agent reply was still running when the request "
                                 "timeout ran out")
    reasoning_text = "".join(reasoning_parts)
    return {"content": content if seen else None,
            "reasoning": reasoning_text if reasoning_parts else None,
            "reasoning_chars": len(reasoning_text), "finish_reason": finish,
            "stop_sequence": hit, "stop_reason": stop_reason, "token_ids": token_ids,
            "tool_call_deltas": tool_deltas, "usage": usage}


def close_stream(stream):
    """Close a streamed reply's HTTP response (the served endpoint then stops generating)."""
    for obj in (stream, getattr(stream, "response", None)):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - the reply is already read
                pass
            return


def stream_reply_info(read):
    """reply_info's fields for a streamed reply (read_stream's result)."""
    usage = read.get("usage")
    details = _field(usage, "completion_tokens_details")

    def _n(obj, name):
        v = _field(obj, name) if obj is not None else None
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
    content = read.get("content")
    ids = read.get("token_ids")
    return {"finish_reason": read.get("finish_reason"),
            "stop_reason": read.get("stop_reason"),
            "token_ids_returned": len(ids) if isinstance(ids, list) else None,
            "tool_call_deltas": int(read.get("tool_call_deltas") or 0),
            "content_null": content is None,
            "content_chars": len(content) if isinstance(content, str) else 0,
            "reasoning_chars": int(read.get("reasoning_chars") or 0),
            "prompt_tokens": _n(usage, "prompt_tokens"),
            "completion_tokens": _n(usage, "completion_tokens"),
            "total_tokens": _n(usage, "total_tokens"),
            "reasoning_tokens": _n(details, "reasoning_tokens")}


def agent_timeout_s(max_tokens):
    """The agent's request timeout for a per-turn budget: LiteLLM's 600 s for the paper's 16000
    tokens, scaled with a larger (--thinking) budget, so a timeout always means the endpoint
    decoded fewer than 16000 / 600 (about 27) tokens per second for the request."""
    budget = max(int(max_tokens or PAPER_AGENT_MAX_TOKENS), PAPER_AGENT_MAX_TOKENS)
    return int(math.ceil(LITELLM_DEFAULT_TIMEOUT_S * budget / PAPER_AGENT_MAX_TOKENS))


#: Written by the hooked ARE process: one JSON line per LLM call that raised (and per soft check
#: without a verdict, per agent reply the server's tool-call parser deleted beyond recovery).
CALL_FAILURES = "gbench_llm_call_failures.jsonl"
#: Written by the hooked ARE process: one JSON line per agent call.
AGENT_CALLS = "gbench_agent_calls.jsonl"
#: Written by the hooked ARE process: one JSON line per soft check (LLMChecker call).
JUDGE_VERDICTS = "gbench_judge_verdicts.jsonl"
#: Written by the hooked ARE process with ARE's report: per-run classification and ARE's
#: statistics recomputed without the harness-error runs.
GBENCH_STATS = "gbench_stats.json"
#: Written by the hooked ARE process: one JSON line per run_dataset call (config, phase, the
#: capability key ARE's statistics use, whether it returned or raised).
CONFIGS = "gbench_configs.jsonl"
#: Written by the hooked ARE process when it installs gbench's hooks: which ones it installed.
HOOKS = "gbench_hooks.json"
#: Written by the hooked ARE process: one JSON line per (config, split) the --shard hook filtered.
SHARD_RECORD = "gbench_shard.jsonl"
#: The capability configs of the default `run` mode (CapabilityTag.gaia2_capabilities()), and the
#: extra config gaia2-run loads for its Agent2Agent and Noise phases.
CAPABILITY_CONFIGS = ("adaptability", "ambiguity", "execution", "search", "time")
GAIA2_RUN_EXTRA_CONFIGS = ("mini",)
#: Characters of a config's exception message kept in CONFIGS.
MESSAGE_CHARS = 300
#: The kinds of harness error a scenario run can have, in the order they are reported.
HARNESS_ERROR_KINDS = ("model_endpoint", "server_parser_text_lost", "judge", "judge_no_verdict",
                       "validation_exception", "run_exception")
#: HTTP statuses of a model endpoint reply that reject the request's own content (the model's
#: outcome here): 400 (vLLM's answer to an invalid or over-long request) and 422 (a validation
#: error on servers that answer it with 422). Every other status (5xx; 401/403/404: the endpoint's
#: key, access or served-model name; 408, 429; any other 4xx) is a failure of the endpoint. This is
#: the rule of gbench's quality runner (runners/quality.py model_status_is_harness_error) and
#: toolbench; gbench does not apply one rule in every suite yet: base-run suites, bfcl and the SWE
#: family count a 400/422 as a harness error (docs/evals/env-knobs.md).
REQUEST_REJECTION_STATUSES = frozenset({400, 422})
#: LiteLLM exception classes that reject the request's own content (the model's outcome).
REQUEST_REJECTIONS = frozenset({"BadRequestError", "ContextWindowExceededError",
                                "UnprocessableEntityError", "ContentPolicyViolationError",
                                "UnsupportedParamsError", "JSONSchemaValidationError"})
#: ARE's exception for a scenario that ran past --scenario_timeout: the model's outcome.
SCENARIO_TIMEOUT = "ScenarioTimeoutError"
#: ARE's validation rationale when the judge raised (validation/base.py validate_current_turn).
VALIDATION_EXCEPTION = "Exception"
#: Characters of a judge reply kept in gbench_judge_verdicts.jsonl.
REPLY_CHARS = 600
#: Characters of a recovered agent reply (the server-parser policy) kept in gbench_agent_calls.jsonl.
RECOVERED_CHARS = 4000
#: The request field that asks vLLM for the generated token ids (metadata only: it changes nothing
#: in the generation). Merged into every agent request's extra_body.
TOKEN_IDS_FIELD = "return_token_ids"


# --- The server-parser policy: copied from gbench's base.py (tests hold them equal) -------------
SERVER_PARSER_MOVED_TEXT = "server_parser_moved_text"
SERVER_PARSER_DROPPED_TEXT = "server_parser_dropped_text"
SERVER_PARSER_TEXT_LOST = "server_parser_text_lost"
SERVER_PARSER_EVENTS = (SERVER_PARSER_MOVED_TEXT, SERVER_PARSER_DROPPED_TEXT,
                        SERVER_PARSER_TEXT_LOST)


class ParserRecovery(collections.namedtuple("ParserRecovery",
                                            ("event", "text", "recovered_from"))):
    """What the server-parser policy made of one reply (recover_server_parser_text)."""


def server_parser_signal(*, content, reasoning, tool_calls, thinking, completion_tokens):
    """Which server-parser path a reply shows, from the reply alone (no decoding).

    SERVER_PARSER_MOVED_TEXT: the request was sent with thinking off, the reply has a non-empty
    reasoning field and no bound tool call. SERVER_PARSER_DROPPED_TEXT: the reply has no content,
    no reasoning and no tool call, and the server generated tokens (completion_tokens > 0), in
    either thinking mode. None otherwise."""
    if tool_calls:
        return None
    if not thinking and str(reasoning or "").strip():
        return SERVER_PARSER_MOVED_TEXT
    if (not str(content or "").strip() and not str(reasoning or "").strip()
            and int(completion_tokens or 0) > 0):
        return SERVER_PARSER_DROPPED_TEXT
    return None


def generation_text_from_token_ids(token_ids, decode, finish_reason=None, stop_reason=None):
    """The server's output text rebuilt from a reply's generated token ids, as vLLM builds it
    before any parser runs.

    `decode(ids)` must keep special tokens (tokenizer.decode(ids, skip_special_tokens=False), what
    vLLM's /detokenize returns) and return None when it cannot decode. A reply that finished on
    a stop token (finish_reason "stop" with no stop string: the end-of-sequence token or a stop
    token id, which vLLM appends to token_ids but not to the text) has that last id left out; a
    matched stop string (`stop_reason` a str) cuts the text where it starts, as the server's text
    is cut. None when there are no ids or they cannot be decoded."""
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
    except Exception:                                                       # noqa: BLE001
        return None
    if not isinstance(text, str):
        return None
    if isinstance(stop_reason, str) and stop_reason:
        cut = text.find(stop_reason)
        if cut >= 0:
            text = text[:cut]
    return text


def recover_server_parser_text(*, content, reasoning, tool_calls, thinking, completion_tokens,
                               generation_text=None):
    """The server-parser policy for one reply (see SERVER_PARSER_MOVED_TEXT above).

    `thinking` is the request's effective enable_thinking; `generation_text` is the reply's
    generation rebuilt from its token ids (generation_text_from_token_ids), None when the server
    returned no token ids or they could not be decoded.

    * moved text (server_parser_signal): the model's output is `generation_text` when it holds
      text, else the reasoning field followed by the content;
    * dropped text: the model's output is `generation_text` when it holds text; a generation that
      decodes to nothing but whitespace is an empty reply (the model's, no event); with no
      `generation_text` the reply is SERVER_PARSER_TEXT_LOST, a harness error;
    * anything else: no event, the reply is left as it is.
    Pure (no I/O): suites with their own model client apply the same function."""
    signal = server_parser_signal(content=content, reasoning=reasoning, tool_calls=tool_calls,
                                  thinking=thinking, completion_tokens=completion_tokens)
    if signal is None:
        return ParserRecovery(None, None, None)
    have_generation = isinstance(generation_text, str) and bool(generation_text.strip())
    if signal == SERVER_PARSER_MOVED_TEXT:
        if have_generation:
            return ParserRecovery(SERVER_PARSER_MOVED_TEXT, generation_text, "token_ids")
        return ParserRecovery(SERVER_PARSER_MOVED_TEXT, str(reasoning or "") + str(content or ""),
                              "reasoning_then_content")
    if have_generation:
        return ParserRecovery(SERVER_PARSER_DROPPED_TEXT, generation_text, "token_ids")
    if isinstance(generation_text, str):
        return ParserRecovery(None, None, None)
    return ParserRecovery(SERVER_PARSER_TEXT_LOST, None, None)


#: Markup a parser removes around the segments it returns: a tag with no whitespace inside
#: (`<|...|>`, `<...>`), the form special and added tokens of chat and reasoning templates take.
_PARSER_MARKUP_RE = re.compile(r"<[^<>\s]{1,64}>")


def server_parser_residue(*, content, reasoning, tool_calls, thinking, generation_text):
    """The server-parser policy's check for text a tool-call parser deleted from a reply that
    still carries content or reasoning (server_parser_signal sees neither case).

    Two shapes, both from a request that declared no tools: a reply sent with thinking off whose
    content is the text BEFORE a native tool call the parser lifted out and dropped (content
    'Let me look that up.', the call gone), and a --thinking reply whose reasoning holds the
    thought and whose answer was a native tool call the parser dropped (content empty). Rule:
    locate the returned reasoning, then the returned content, verbatim in `generation_text` (the
    reply's generation rebuilt from its token ids, special tokens kept); the text after the last
    located segment is the tail. When the tail holds a letter or digit outside tag markup
    (_PARSER_MARKUP_RE), the parser removed generated text: SERVER_PARSER_DROPPED_TEXT, and the
    model's output is the generation after the reasoning on a --thinking reply (the reasoning
    stays the reasoning), else the whole generation. A segment that cannot be located, a tail of
    markup and whitespace only, a reply with bound tool calls, or no generation text: no event.
    Pure (no I/O)."""
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
# --- end of the copy -----------------------------------------------------------------------------


#: The largest token-id list sent to gbench's decoder in one request.
DETOKENIZE_MAX_IDS = 262144


def decode_token_ids(ids, url=None, timeout_s=60):
    """`ids` decoded with special tokens kept by gbench's decoder on the docker network's gateway
    (GBENCH_DETOKENIZE_URL: the served tokenizer's /detokenize, else the run's tokenizer), or None
    when there is no decoder, it cannot decode them, or it cannot be reached."""
    import urllib.request
    url = url if url is not None else os.environ.get("GBENCH_DETOKENIZE_URL", "")
    if not url or not isinstance(ids, (list, tuple)) or len(ids) > DETOKENIZE_MAX_IDS:
        return None
    try:
        body = json.dumps({"tokens": [int(t) for t in ids]}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read())
    except Exception:  # noqa: BLE001 - no decoder: the reply's text is not recovered from ids
        return None
    text = payload.get("text") if isinstance(payload, dict) else None
    return text if isinstance(text, str) else None


def choice_extras(response_object):
    """[{token_ids, stop_reason}] per choice of a raw chat.completion reply (a dict): the fields
    vLLM adds to each choice that LiteLLM's conversion drops."""
    choices = response_object.get("choices") if isinstance(response_object, dict) else None
    out = []
    for choice in choices or []:
        choice = choice if isinstance(choice, dict) else {}
        ids = choice.get("token_ids")
        out.append({"token_ids": list(ids) if isinstance(ids, (list, tuple)) else None,
                    "stop_reason": choice.get("stop_reason")})
    return out


def with_token_ids(kwargs):
    """An agent request's completion kwargs with TOKEN_IDS_FIELD added to its extra_body."""
    kw = dict(kwargs)
    kw["extra_body"] = {**(kw.get("extra_body") or {}), TOKEN_IDS_FIELD: True}
    return kw


def apply_server_parser_policy(content, reasoning, info, thinking, token_ids, decode):
    """(the reply ARE gets, event, recovered_from, generation text) for one agent reply: the
    server-parser policy with the generation rebuilt from its token ids. ARE declares no tools and
    reads only the content, so a tool call the server returned (tool_call_deltas of a streamed
    reply) is not seen by the harness: the reply counts as one without a tool call.

    The token ids, when the reply carries them, are always decoded (the generation text is
    returned so every reply without content, and every recovered one, can be read). A reply with
    no content and no reasoning, or one whose reasoning the server moved out of a no-think reply,
    goes through recover_server_parser_text; any other reply through server_parser_residue, which
    finds generated text the tool-call parser deleted after the returned content or reasoning
    (e.g. a --thinking reply whose answer was a native tool call: the text after the thought is
    what ARE gets, the model's output). Both copies of base's functions, model-agnostic."""
    completion_tokens = int(info.get("completion_tokens") or 0)
    generation = None
    if isinstance(token_ids, (list, tuple)) and token_ids:
        generation = generation_text_from_token_ids(token_ids, decode, info.get("finish_reason"),
                                                    info.get("stop_reason"))
    signal = server_parser_signal(content=content, reasoning=reasoning, tool_calls=None,
                                  thinking=thinking, completion_tokens=completion_tokens)
    if signal is None:
        rec = server_parser_residue(content=content, reasoning=reasoning, tool_calls=None,
                                    thinking=thinking, generation_text=generation)
        if rec.event is None:
            return content, None, None, generation
        return rec.text, rec.event, rec.recovered_from, generation
    rec = recover_server_parser_text(content=content, reasoning=reasoning, tool_calls=None,
                                     thinking=thinking, completion_tokens=completion_tokens,
                                     generation_text=generation)
    text = rec.text if rec.text is not None else (None if rec.event else content)
    return text, rec.event, rec.recovered_from, generation


def judge_endpoint():
    return f"http://127.0.0.1:{os.environ.get('GAIA2_JUDGE_PORT', '18790')}/v1"


def endpoint_failure(exc):
    """True when a model endpoint call's exception is a failure of the endpoint (any HTTP status
    other than 400 and 422: 5xx, 401, 403, 404, 408, 429 and the other 4xx; or no HTTP status at
    all: a refused connection, a timeout, a client failure, an error event inside a streamed
    reply), False when the endpoint refused the request's own content (HTTP 400/422, a context
    window rejection). Every agent request carries the per-turn cap and a timeout sized to it
    (agent_timeout_s), so a timeout means the endpoint decoded the capped reply too slowly: a
    failure of the endpoint, not a long generation of the model."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status not in REQUEST_REJECTION_STATUSES
    return type(exc).__name__ not in REQUEST_REJECTIONS


def classify_run(row, failures):
    """The harness-error kinds of one scenario run: `row` is ARE's per-run row (status,
    has_exception, exception_type, rationale), `failures` the recorded failed calls of that run."""
    kinds = set()
    if any(f.get("role") == "agent" and f.get("endpoint_failure") for f in failures):
        kinds.add("model_endpoint")
    if any(f.get("role") == "agent" and f.get("server_parser_text_lost") for f in failures):
        kinds.add("server_parser_text_lost")
    if any(f.get("role") == "judge" and not f.get("no_verdict") for f in failures):
        kinds.add("judge")
    if any(f.get("role") == "judge" and f.get("no_verdict") for f in failures):
        kinds.add("judge_no_verdict")
    if row.get("has_exception") and row.get("exception_type") != SCENARIO_TIMEOUT \
            and row.get("status") != "no_validation":
        if row.get("rationale") == VALIDATION_EXCEPTION:
            if "judge" not in kinds:
                kinds.add("validation_exception")
        else:
            kinds.add("run_exception")
    return [k for k in HARNESS_ERROR_KINDS if k in kinds]


def _run_key(config, phase_name, scenario_id, run_number):
    rn = None if run_number is None else str(run_number)
    return (str(config), str(phase_name or "standard"), str(scenario_id), rn)


def capability_key(config, phase_name):
    """The key ARE's report_stats.calculate_statistics gives a (config, phase) group in
    per_capability: '<config>_<phase>' for the mini config, else the config."""
    config = str(config)
    return f"{config}_{phase_name or 'standard'}" if config == "mini" else config


def _read_jsonl(name, workdir=None):
    """The JSON objects of one of the launcher's .jsonl files (unreadable lines skipped)."""
    try:
        with open(os.path.join(workdir or WORKDIR, name), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def read_call_failures(workdir=None):
    """({run key: [failures]}, [failures tied to no run]) from CALL_FAILURES."""
    by_run, loose = {}, []
    for rec in _read_jsonl(CALL_FAILURES, workdir):
        if not rec.get("scenario_id"):
            loose.append(rec)
            continue
        by_run.setdefault(_run_key(rec.get("config"), rec.get("phase_name"), rec["scenario_id"],
                                   rec.get("run_number")), []).append(rec)
    return by_run, loose


def macro_over_validated(stats):
    """ARE's statistics with the macro success rate averaged over the capabilities that have a
    validated run (ARE's own formula gives a capability without one a success rate of 0.0);
    returns the capability keys left out. The rate ARE's formula gives is kept as
    macro_success_rate_are_formula. Leaves anything that is not ARE's statistics dict alone."""
    per = stats.get("per_capability") if isinstance(stats, dict) else None
    if not isinstance(per, dict):
        return []
    unscored = sorted(k for k, v in per.items() if not (v or {}).get("validated_runs"))
    if not unscored:
        return []
    rates = [float((v or {}).get("success_rate") or 0.0) for k, v in per.items()
             if k not in unscored]
    g = stats.setdefault("global", {})
    g["macro_success_rate_are_formula"] = g.get("macro_success_rate")
    g["macro_success_rate"] = (sum(rates) / len(rates)) if rates else None
    return unscored


def gbench_statistics(rows, by_run, loose, calculate, configs=None):
    """The gbench_stats.json content: `rows` are ARE's per-run rows (dicts), `calculate` ARE's
    report_stats.calculate_statistics applied to the rows it is given (a list of row dicts),
    `configs` the run_dataset calls recorded in CONFIGS."""
    runs, keep, kinds_count = [], [], {}
    success_kinds, n_success = {}, 0
    matched = set()
    for row in rows:
        key = _run_key(row.get("config"), row.get("phase_name"), row.get("base_scenario_id"),
                       row.get("run_number"))
        failures = by_run.get(key, [])
        if failures:
            matched.add(key)
        kinds = classify_run(row, failures)
        for k in kinds:
            kinds_count[k] = kinds_count.get(k, 0) + 1
        if kinds and row.get("status") == "success":
            n_success += 1
            for k in kinds:
                success_kinds[k] = success_kinds.get(k, 0) + 1
        if not kinds:
            keep.append(row)
        runs.append({"scenario_id": row.get("base_scenario_id"), "run_number": row.get("run_number"),
                     "config": row.get("config"), "phase_name": row.get("phase_name"),
                     "status": row.get("status"), "exception_type": row.get("exception_type"),
                     "rationale": (str(row.get("rationale"))[:300]
                                   if row.get("rationale") is not None else None),
                     "harness_errors": kinds,
                     "failed_calls": [{"role": f.get("role"), "error": f.get("error"),
                                       "status": f.get("status")} for f in failures][:10]})
    unmatched = loose + [f for key, fs in by_run.items() if key not in matched for f in fs]
    stats = calculate(keep)
    unscored = macro_over_validated(stats)
    configs = [dict(c) for c in (configs or []) if isinstance(c, dict)]
    return {"statistics": stats,
            "n_runs": len(rows),
            "n_harness_error_runs": len(rows) - len(keep),
            "harness_error_kinds": kinds_count,
            # Runs left out as harness errors that ARE had validated as successes (by kind).
            "n_harness_error_success_runs": n_success,
            "harness_error_success_kinds": success_kinds,
            "capabilities_without_validated_run": unscored,
            "configs": configs,
            "failed_configs": [c for c in configs if not c.get("ok")],
            "unattributed_call_failures": len(unmatched),
            "unattributed_call_failure_roles": sorted({str(f.get("role")) for f in unmatched}),
            "runs": runs}


_CONTEXT = {"config": None, "phase_name": "standard"}
_FAIL_LOCK = threading.Lock()
_TLS = threading.local()


def _current_run():
    """(scenario id, run number) of the calling thread's scenario run, or (None, None)."""
    try:
        from are.simulation import logging_config
        return logging_config.get_logger_scenario_id(), logging_config.get_logger_run_number()
    except Exception:  # noqa: BLE001 - a record with no run is reported as unattributed
        return None, None


def _run_fields():
    scenario_id, run_number = _current_run()
    return {"scenario_id": scenario_id, "run_number": run_number,
            "config": _CONTEXT["config"], "phase_name": _CONTEXT["phase_name"]}


def _append(name, rec):
    with _FAIL_LOCK:
        with open(os.path.join(WORKDIR, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _record_call_failure(api_base, exc):
    """Append one failed LLM call of the ARE process to CALL_FAILURES."""
    role = "judge" if str(api_base or "").rstrip("/").startswith(judge_endpoint()) else "agent"
    status = getattr(exc, "status_code", None)
    rec = {"role": role, **_run_fields(), "error": type(exc).__name__,
           "status": status if isinstance(status, int) and not isinstance(status, bool) else None,
           "endpoint_failure": endpoint_failure(exc) if role == "agent" else True}
    _append(CALL_FAILURES, rec)


def reply_reasoning(response):
    """The reasoning text of a non-streamed agent reply: the served reasoning field is `reasoning`
    (LiteLLM keeps it in provider_specific_fields) or the older `reasoning_content`; None when
    there is none."""
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return None
    reasoning = getattr(message, "reasoning_content", None)
    psf = getattr(message, "provider_specific_fields", None) or {}
    if not reasoning and isinstance(psf, dict):
        reasoning = psf.get("reasoning") or psf.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else None


def response_extras(response):
    """{token_ids, stop_reason} of a non-streamed agent reply's first choice, as the hook on
    LiteLLM's conversion kept them (choice_extras; `_hidden_params["gbench_choices"]`)."""
    hidden = getattr(response, "_hidden_params", None)
    choices = hidden.get("gbench_choices") if isinstance(hidden, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else {}
    return {"token_ids": first.get("token_ids"), "stop_reason": first.get("stop_reason")}


def reply_info(response):
    """What an agent reply shows: finish reason, vLLM's stop_reason, content and reasoning
    sizes, token counts, how many token ids came back."""
    try:
        choice = response.choices[0]
        message = choice.message
    except (AttributeError, IndexError, TypeError):
        return {}
    content = getattr(message, "content", None)
    reasoning = reply_reasoning(response)
    extras = response_extras(response)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)

    def _n(obj, name):
        v = getattr(obj, name, None) if obj is not None else None
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
    ids = extras.get("token_ids")
    return {"finish_reason": getattr(choice, "finish_reason", None),
            "stop_reason": extras.get("stop_reason"),
            "token_ids_returned": len(ids) if isinstance(ids, list) else None,
            "content_null": content is None,
            "content_chars": len(content) if isinstance(content, str) else 0,
            "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
            "prompt_tokens": _n(usage, "prompt_tokens"),
            "completion_tokens": _n(usage, "completion_tokens"),
            "total_tokens": _n(usage, "total_tokens"),
            "reasoning_tokens": _n(details, "reasoning_tokens")}


def stopped_in_reasoning(info, enable_thinking):
    """True when a reply on a --thinking run stopped (finish reason 'stop') with no content and a
    non-empty reasoning field: generation ended inside the reasoning channel. Counted so the
    effect is visible; with thinking on it is the model's outcome."""
    return (enable_thinking is True and info.get("finish_reason") == "stop"
            and not info.get("content_chars") and bool(info.get("reasoning_chars")))


def ran_past_stop(content, stop_sequences):
    """True when the reply's answer holds one of the stop sequences although the reply was not
    ended at it (a streamed reply closed at a stop sequence is not passed here): the model
    generated past the action, and ARE's client-side split drops the rest. Neither protocol
    leaves one (the server's `stop` without --thinking, the client's close with it), so a count
    means the server did not honor the stop."""
    return isinstance(content, str) and any(s and s in content for s in (stop_sequences or []))


def _checker_names():
    """{id(prompt templates): checker name} for ARE's soft checkers (the llm_utils constants
    build_llm_checkers uses by default)."""
    from are.simulation.validation.utils import llm_utils
    names = {}
    for attr in dir(llm_utils):
        if attr.endswith("_TEMPLATES") and "CHECKER" in attr:
            names[id(getattr(llm_utils, attr))] = (
                attr.replace("_PROMPT_TEMPLATES", "").replace("_TEMPLATES", "").lower())
    return names


def verdict_of(response, success_str, failure_str):
    """gaia2_core LLMChecker's vote on one judge reply: True / False, or None when it holds
    neither marker (matched case-insensitively)."""
    if response is None:
        return None
    low = response.lower()
    if success_str.lower() in low:
        return True
    if failure_str.lower() in low:
        return False
    return None


#: Whether install_token_ids_hook found LiteLLM's openai conversion to wrap (recorded in the
#: summary: without it a non-streamed reply's token ids are not kept).
_TOKEN_IDS_HOOK = {"installed": False}


def install_token_ids_hook():
    """Keep the fields vLLM adds to each choice of a chat.completion reply (token_ids with
    return_token_ids, and stop_reason) that LiteLLM's conversion to a ModelResponse drops: the
    OpenAI-compatible provider (litellm.llms.openai.openai) converts every reply with its module
    global convert_to_model_response_object, which is wrapped to store choice_extras of the raw
    reply in the ModelResponse's _hidden_params["gbench_choices"]. Returns True when installed.
    The image build fails when the pinned LiteLLM no longer has that global."""
    try:
        from litellm.llms.openai import openai as litellm_openai
    except Exception:  # noqa: BLE001 - recorded: token ids of non-streamed replies not kept
        return False
    orig = getattr(litellm_openai, "convert_to_model_response_object", None)
    if orig is None:
        return False
    if getattr(orig, "_gbench_token_ids", False):
        _TOKEN_IDS_HOOK["installed"] = True
        return True

    def convert(*args, **kwargs):
        out = orig(*args, **kwargs)
        try:
            raw = kwargs["response_object"] if "response_object" in kwargs else (
                args[0] if args else None)
            extras = choice_extras(raw)
            if extras and out is not None:
                hidden = getattr(out, "_hidden_params", None)
                out._hidden_params = {**(hidden if isinstance(hidden, dict) else {}),
                                      "gbench_choices": extras}
        except Exception:  # noqa: BLE001 - the reply itself is unchanged
            pass
        return out
    convert._gbench_token_ids = True
    litellm_openai.convert_to_model_response_object = convert
    _TOKEN_IDS_HOOK["installed"] = True
    return True


def install_are_hooks():
    """Record every failed LiteLLM call with its scenario run, apply the agent protocol and the
    upstream judge path (see the file header), track the config and phase being run, and write
    GBENCH_STATS next to ARE's own report. Runs inside the ARE process."""
    from are.simulation.agents.default_agent import are_simulation_main
    from are.simulation.agents.llm.litellm import litellm_engine
    from are.simulation.agents.llm.llm_engine import LLMEngineException
    from are.simulation.benchmark import cli, gaia2_submission
    from are.simulation.validation.utils import llm_utils
    from litellm.exceptions import APIError, AuthenticationError
    from litellm.types.utils import Choices, ModelResponse

    settings = agent_settings()
    token_ids_hook = install_token_ids_hook()
    with open(os.path.join(WORKDIR, HOOKS), "w", encoding="utf-8") as f:
        json.dump({"token_ids_hook": token_ids_hook,
                   "detokenizer": bool(os.environ.get("GBENCH_DETOKENIZE_URL"))}, f)

    orig_completion = litellm_engine.completion

    def completion(*args, **kwargs):
        try:
            return orig_completion(*args, **kwargs)
        except Exception as exc:
            _record_call_failure(kwargs.get("api_base"), exc)
            raise
    litellm_engine.completion = completion

    def streamed_reply(engine, request, stop_sequences, started):
        """(read_stream's result, reply_info fields, duration) of a streamed agent request
        (client_stop): read until the answer holds a stop sequence or the stream ends, then
        closed. A failure while reading is recorded like a failed call; a reply still running
        after the request timeout is closed and raises LiteLLM's Timeout."""
        endpoint = engine.model_config.endpoint
        wrapper = litellm_engine.completion(**request)
        stream = getattr(wrapper, "completion_stream", None)
        try:
            if stream is None:
                raise RuntimeError("LiteLLM returned no stream for a streamed agent request")
            try:
                read = read_stream(stream, stop_sequences,
                                   deadline=started + settings["timeout_s"])
            except StreamDeadline as exc:
                from litellm.exceptions import Timeout
                raise Timeout(message=f"{exc} ({settings['timeout_s']} s)",
                              model=engine.model_config.model_name,
                              llm_provider="openai") from exc
            duration = time.monotonic() - started
        except Exception as exc:
            _record_call_failure(endpoint, exc)
            raise
        finally:
            close_stream(stream)
        return read, stream_reply_info(read), duration

    def chat_completion(self, messages, stop_sequences=[], **kwargs):  # noqa: B006 - ARE's API
        """LiteLLMEngine.chat_completion (ARE 1.2.0) with the agent protocol, the returned
        metadata and the verbatim judge reply; everything else as upstream."""
        judge = str(self.model_config.endpoint or "").rstrip("/").startswith(judge_endpoint())
        try:
            converted = [self._convert_message_to_litellm_format(m) for m in messages]
            provider = (self.model_config.provider
                        if self.model_config.provider != "local" else None)
            extra = ({} if judge
                     else with_token_ids(agent_completion_kwargs(settings, stop_sequences)))
            request = dict(model=self.model_config.model_name, custom_llm_provider=provider,
                           messages=converted, api_base=self.model_config.endpoint,
                           api_key=self.model_config.api_key, mock_response=self.mock_response,
                           **extra)
            started = time.monotonic()
            if extra.get("stream"):
                read, info, duration = streamed_reply(self, request, stop_sequences, started)
                res, closed_at = read["content"], read["stop_sequence"]
                reasoning, token_ids = read.get("reasoning"), read.get("token_ids")
            else:
                response = litellm_engine.completion(**request)
                duration = time.monotonic() - started
                assert type(response) is ModelResponse
                assert len(response.choices) >= 1
                assert type(response.choices[0]) is Choices
                res = response.choices[0].message.content
                if judge:
                    _TLS.judge_model = getattr(response, "model", None)
                    assert res is not None
                    for stop_token in stop_sequences:
                        res = res.split(stop_token)[0]
                    return res, None
                info, closed_at = reply_info(response), None
                reasoning = reply_reasoning(response)
                token_ids = response_extras(response).get("token_ids")
            # The server-parser policy: text the server's parsers took out of the reply is the
            # model's output, recovered from its token ids (see the file header).
            res, event, recovered_from, generation = apply_server_parser_policy(
                res, reasoning, info, settings.get("enable_thinking") is True, token_ids,
                decode_token_ids)
            call = {**_run_fields(), "duration_s": round(duration, 3),
                    "max_tokens": settings["max_tokens"],
                    "stop_sent": bool(extra.get("stop")),
                    "streamed": bool(extra.get("stream")),
                    "stopped_client_side": closed_at is not None,
                    "server_parser": event, "recovered_from": recovered_from,
                    "stopped_in_reasoning": (event is None and stopped_in_reasoning(
                        info, settings.get("enable_thinking"))),
                    "ran_past_stop": (closed_at is None
                                      and ran_past_stop(res, stop_sequences)),
                    **info}
            if event is not None or not info.get("content_chars"):
                # the text the reply was rebuilt from, so every empty or recovered reply can be
                # read (capped)
                call["generation_text"] = (generation[:RECOVERED_CHARS]
                                           if isinstance(generation, str) else None)
            _append(AGENT_CALLS, call)
            if event == SERVER_PARSER_TEXT_LOST:
                _append(CALL_FAILURES, {"role": "agent", **_run_fields(),
                                        "error": "ServerParserTextLost", "status": None,
                                        "endpoint_failure": False,
                                        "server_parser_text_lost": True,
                                        "completion_tokens": info.get("completion_tokens"),
                                        "token_ids_returned": info.get("token_ids_returned")})
            assert res is not None
            res = res.replace("False", "false").replace("True", "true")
            for stop_token in stop_sequences:
                res = res.split(stop_token)[0]
            metadata = {"completion_duration": duration,
                        "prompt_tokens": info.get("prompt_tokens", 0),
                        "completion_tokens": info.get("completion_tokens", 0),
                        "total_tokens": info.get("total_tokens", 0),
                        "reasoning_tokens": info.get("reasoning_tokens", 0)}
            return res, metadata
        except (AuthenticationError, APIError) as e:
            raise LLMEngineException("Auth error in litellm.") from e
    litellm_engine.LiteLLMEngine.chat_completion = chat_completion

    names = _checker_names()

    def checker_call(self, user_prompt_args):
        """LLMChecker.__call__ with gaia2_core's case-insensitive vote, each check recorded."""
        votes, replies = [], []
        for _ in range(self.num_votes):
            _TLS.judge_model = None
            response = self.judge(user_prompt_args)
            vote = verdict_of(response, self.success_str, self.failure_str)
            replies.append({"served_model": getattr(_TLS, "judge_model", None), "vote": vote,
                            "reply": (response or "")[:REPLY_CHARS]})
            if vote is not None:
                votes.append(vote)
        verdict = (sum(votes) >= len(votes) / 2) if votes else None
        checker = names.get(id(self.prompt_templates), self.success_str)
        _append(JUDGE_VERDICTS, {**_run_fields(), "checker": checker, "verdict": verdict,
                                 "replies": replies})
        if verdict is None:
            _append(CALL_FAILURES, {"role": "judge", **_run_fields(), "error": "NoVerdict",
                                    "status": None, "endpoint_failure": True, "no_verdict": True,
                                    "checker": checker})
        return verdict
    llm_utils.LLMChecker.__call__ = checker_call

    orig_agent_init = are_simulation_main.ARESimulationAgent.__init__

    def agent_init(self, *args, **kwargs):
        kwargs.setdefault("max_iterations", settings["max_iterations"])
        return orig_agent_init(self, *args, **kwargs)
    are_simulation_main.ARESimulationAgent.__init__ = agent_init

    for module in (cli, gaia2_submission):
        orig_run_dataset = module.run_dataset

        def run_dataset(*args, _orig=orig_run_dataset, **kwargs):
            config = kwargs.get("config")
            phase = kwargs.get("phase_name") or "standard"
            _CONTEXT["config"] = config
            _CONTEXT["phase_name"] = phase
            rec = {"config": config, "phase_name": phase,
                   "capability": capability_key(config, phase)}
            try:
                result = _orig(*args, **kwargs)
            except Exception as exc:
                # ARE logs the config, skips it and reports the others: recorded here so the
                # gbench runner can name it.
                _append(CONFIGS, {**rec, "ok": False, "error": type(exc).__name__,
                                  "message": str(exc)[:MESSAGE_CHARS]})
                raise
            _append(CONFIGS, {**rec, "ok": True, "error": None, "message": None})
            return result
        module.run_dataset = run_dataset

    orig_reports = cli.generate_and_save_reports

    def generate_and_save_reports(results, *args, **kwargs):
        try:
            return orig_reports(results, *args, **kwargs)
        finally:
            _write_gbench_stats(results)
    cli.generate_and_save_reports = generate_and_save_reports


def _write_gbench_stats(results):
    """Classify ARE's scenario runs and write GBENCH_STATS (runs inside the hooked ARE process)."""
    import polars as pl
    from are.simulation.benchmark.report_stats import (calculate_statistics,
                                                        combine_results_to_dataframe)
    df = combine_results_to_dataframe(results)
    rows = df.to_dicts()
    by_run, loose = read_call_failures()

    def calculate(keep):
        kept = pl.DataFrame(keep, schema=df.schema) if keep else df.clear()
        return calculate_statistics(kept)

    stats = gbench_statistics(rows, by_run, loose, calculate, configs=_read_jsonl(CONFIGS))
    with open(os.path.join(WORKDIR, GBENCH_STATS), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)


def parse_shard(spec):
    """(I, N) of a GAIA2_SHARD "I/N" spec (1 <= I <= N), None when it is unset or empty; a
    malformed spec raises ValueError (the gbench runner validated it already)."""
    text = str(spec or "").strip()
    if not text:
        return None
    try:
        i, n = (int(x) for x in text.split("/", 1))
    except ValueError:
        raise ValueError(f"GAIA2_SHARD must be I/N, got {spec!r}") from None
    if n < 1 or not 1 <= i <= n:
        raise ValueError(f"GAIA2_SHARD I/N needs 1 <= I <= N, got {spec!r}")
    return i, n


def shard_scenario_ids(ids, shard):
    """Shard I of N of a config's scenario ids: the distinct ids sorted, every N-th starting at
    the I-th (round-robin, as gbench's sampling.shard_select), so the shards of one config never
    overlap and together hold every id."""
    i, n = shard
    ordered = sorted({str(x) for x in ids if x})
    return ordered[i - 1::n]


def _is_scenario_dataset(path):
    return str(path) == str(DATASET)


#: The seed of the --eval-limit draw (the suite name, as gbench's sampling.stratified_sample uses).
LIMIT_SEED = "gaia2"
#: How a limited run selects each config's scenarios; the summary's selection.limit_draw (gbench
#: checks it). A config is one capability, so the draw has one category: a seeded subset.
LIMIT_DRAW = f"stratified_indices(key=none, seed={LIMIT_SEED}) per config"


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


def parse_limit(value):
    """GAIA2_LIMIT as a positive int, None when it is unset, empty or not positive; a value that
    is not an integer raises ValueError."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        n = int(text)
    except ValueError:
        raise ValueError(f"GAIA2_LIMIT must be an integer, got {value!r}") from None
    return n if n > 0 else None


def parse_config_limits(value):
    """GAIA2_CONFIG_LIMITS ({config: scenarios}, JSON) as a dict of non-negative ints, None when it
    is unset or empty; anything else raises ValueError (never read as "no limit", a full run)."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        raise ValueError(f"GAIA2_CONFIG_LIMITS must be a JSON object, got {value!r}") from None
    if not isinstance(data, dict) or not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in data.values()):
        raise ValueError(f"GAIA2_CONFIG_LIMITS must map configs to non-negative integers, "
                         f"got {value!r}")
    return {str(k): int(v) for k, v in data.items()}


class LimitSkipped(RuntimeError):
    """Raised by the run_dataset hook for a config whose --eval-limit share is 0 (ARE catches it,
    logs the config as failed and runs the others)."""


def select_scenario_ids(ids, shard=None, limit=None):
    """The scenario ids a config keeps: the distinct ids sorted, shard I of N (shard_scenario_ids)
    when `shard` is set, then a seeded draw of `limit` of them (stratified_indices, LIMIT_SEED)
    when `limit` is set; sorted order kept."""
    ordered = shard_scenario_ids(ids, shard) if shard else sorted({str(x) for x in ids if x})
    if limit is not None and limit == 0:
        return []
    if limit:
        ordered = [ordered[k] for k in stratified_indices([None] * len(ordered), limit,
                                                          LIMIT_SEED)]
    return ordered


def install_shard_hook(shard, limit=None, config_limits=None):
    """Wrap datasets.load_dataset so that each split of the scenario dataset ARE loads keeps only
    this run's scenarios (select_scenario_ids over the split's scenario ids: the shard, then the
    --eval-limit draw: the config's share in `config_limits`, else `limit`), as an IterableDataset
    filter in the stream's own order; every selection goes to SHARD_RECORD."""
    import datasets
    orig = datasets.load_dataset

    def load_dataset(path, *args, **kwargs):
        ds = orig(path, *args, **kwargs)
        if not _is_scenario_dataset(path) or not isinstance(ds, datasets.IterableDatasetDict):
            return ds
        config = kwargs.get("name") if kwargs.get("name") is not None else (
            args[0] if args else None)
        lim = (config_limits.get(str(config), 0) if config_limits is not None else limit)
        for split in list(ds.keys()):
            ids = [row.get("scenario_id") for row in ds[split]]
            keep = frozenset(select_scenario_ids(ids, shard, lim))
            _append(SHARD_RECORD, {"config": config, "split": split,
                                   "shard": f"{shard[0]}/{shard[1]}" if shard else None,
                                   "limit": lim,
                                   "config_limits": config_limits,
                                   "limit_draw": (LIMIT_DRAW if lim or config_limits is not None
                                                  else None),
                                   "scenarios": len({str(x) for x in ids if x}),
                                   "selected": len(keep), "scenario_ids": sorted(keep)})
            ds[split] = ds[split].filter(lambda row, _keep=keep: row.get("scenario_id") in _keep)
        return ds
    datasets.load_dataset = load_dataset


def run_configs(env=None):
    """The configs the run loads: GAIA2_CONFIG, else the 5 capability configs (and, for
    gaia2-run, the mini config of its Agent2Agent and Noise phases)."""
    env = os.environ if env is None else env
    if (env.get("GAIA2_MODE") or "run").strip() == "gaia2-run":
        return list(CAPABILITY_CONFIGS) + list(GAIA2_RUN_EXTRA_CONFIGS)
    single = (env.get("GAIA2_CONFIG") or "").strip()
    return [single] if single else list(CAPABILITY_CONFIGS)


def shard_selection(shard, configs, load=None):
    """{config: [this shard's scenario ids]} for the run's configs, read from the scenario
    dataset (the image's local copy) before ARE starts."""
    if load is None:
        from datasets import load_dataset as load
    out = {}
    for config in configs:
        ds = load(DATASET, name=config, streaming=True)
        out[config] = shard_scenario_ids([row.get("scenario_id") for row in ds[SPLIT]], shard)
    return out


def shard_summary(workdir=None):
    """The summary's `shard` record: the spec and each (config, split) selection SHARD_RECORD
    holds (the ids left out, the counts kept); None for a run without a shard."""
    rows = _read_jsonl(SHARD_RECORD, workdir)
    if not rows or not rows[0].get("shard"):
        return None
    return {"spec": rows[0].get("shard"),
            "per_config": [{k: r.get(k) for k in ("config", "split", "scenarios", "selected")}
                           for r in rows]}


def selection_summary(workdir=None):
    """The summary's `selection` record: the --eval-limit and how it was drawn (LIMIT_DRAW), and
    the scenario ids each (config, split) kept; None when the hook selected nothing (no shard,
    no limit)."""
    rows = _read_jsonl(SHARD_RECORD, workdir)
    if not rows:
        return None
    limits = parse_config_limits(os.environ.get("GAIA2_CONFIG_LIMITS"))
    return {"limit": rows[0].get("limit"), "limit_draw": rows[0].get("limit_draw"),
            "config_limits": limits,
            "limit_skipped": [c for c, n in (limits or {}).items() if n == 0],
            "per_config": [{k: r.get(k) for k in ("config", "split", "scenarios", "selected",
                                                  "scenario_ids")} for r in rows]}


def install_limit_skip(config_limits):
    """Wrap ARE's run_dataset (both entry points, outside the CONFIGS hook) so that a config whose
    --eval-limit share is 0 raises LimitSkipped before ARE loads it: ARE skips the config and runs
    the others, and CONFIGS never records it (it was not attempted)."""
    from are.simulation.benchmark import cli, gaia2_submission
    for module in (cli, gaia2_submission):
        inner = module.run_dataset

        def run_dataset(*args, _inner=inner, **kwargs):
            config = kwargs.get("config")
            if config_limits.get(str(config), 0) == 0:
                raise LimitSkipped(f"--eval-limit gives config {config!r} no scenario")
            return _inner(*args, **kwargs)
        module.run_dataset = run_dataset


def _run_hooked_are(args):
    """`gaia2_run.py --are-hooked <are-benchmark arguments>`: ARE's CLI with gbench's hooks."""
    install_are_hooks()
    shard = parse_shard(os.environ.get("GAIA2_SHARD"))
    # --eval-limit dealt over the configs (both modes), else GBENCH_GAIA2_LIMIT in `run` mode
    # (gaia2-run passes no --limit): a seeded draw per config
    config_limits = parse_config_limits(os.environ.get("GAIA2_CONFIG_LIMITS"))
    limit = (parse_limit(os.environ.get("GAIA2_LIMIT"))
             if (os.environ.get("GAIA2_MODE") or "run").strip() != "gaia2-run" else None)
    if config_limits is not None:
        limit = None
        install_limit_skip(config_limits)
    if shard or limit or config_limits is not None:
        install_shard_hook(shard, limit, config_limits)
    from are.simulation.benchmark import main as are_main
    sys.argv = ["are-benchmark", *args]
    return are_main()


def _run_are():
    mode = os.environ.get("GAIA2_MODE", "run").strip()
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "dummy"  # litellm requires a key; gbench's relay adds the endpoint's
    if mode == "gaia2-run":
        cmd = _base_cmd("gaia2-run") + ["--num_runs", os.environ.get("GAIA2_NUM_RUNS", "3")]
    else:
        cmd = _base_cmd("run")
        # CRITICAL: `are-benchmark run` defaults --agent to None, and ScenarioRunner._run() then takes
        # the `_run_without_agent` path (env.join() + validate ONLY) - the model-under-test is NEVER
        # invoked, so every scenario trivially fails with "Agent count 0" and the suite reports a
        # fabricated 0.0%. The agentic model run requires an explicit agent; "default" is ARE's only
        # agent (AgentBuilder().list_agents() == ['default']) and is what `gaia2-run` uses internally.
        cmd += ["--agent", os.environ.get("GAIA2_AGENT", "default")]
        if os.environ.get("GAIA2_CONFIG"):
            cmd += ["--hf-config", os.environ["GAIA2_CONFIG"]]   # else ALL 5 capability configs
        if os.environ.get("GAIA2_LIMIT"):
            cmd += ["--limit", os.environ["GAIA2_LIMIT"]]
        if os.environ.get("GAIA2_NUM_RUNS"):
            cmd += ["--num_runs", os.environ["GAIA2_NUM_RUNS"]]   # else ARE's default 3
    sys.stderr.write("[gaia2] " + " ".join(cmd) + "\n")
    sys.stderr.write("[gaia2] agent protocol: " + json.dumps(agent_settings()) + "\n")
    # ARE's CLI (cmd[0] is are-benchmark) in a child process that installs gbench's hooks first.
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--are-hooked", *cmd[1:]],
                          env=env)
    return proc.returncode


def _gemini_usage():
    """The judge proxy's Gemini call tally ($GAIA2_WORKDIR/gemini_usage.json), or {}."""
    try:
        with open(os.path.join(WORKDIR, "gemini_usage.json"), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _harness_events():
    """{kind: count} of the judge proxy's harness events ($GAIA2_WORKDIR/harness_events.jsonl:
    judge calls no Gemini cascade model served)."""
    counts = {}
    for rec in _read_jsonl("harness_events.jsonl"):
        kind = rec.get("kind")
        if kind is None:
            continue
        counts[str(kind)] = counts.get(str(kind), 0) + 1
    return counts


def agent_call_summary(workdir=None):
    """Counts over gbench_agent_calls.jsonl: calls, finish reasons, replies with no content
    (null or empty, as the server sent them), the server-parser events (text the server's
    parsers took out of a reply, recovered or lost) and where the text was recovered from,
    replies that came with token ids, thinking-run replies that stopped inside the reasoning
    channel, replies whose answer still held a stop sequence, calls sent `stop`, streamed calls
    and those closed at a stop sequence in the answer, replies cut at max_tokens (and those with
    no content), and token and time totals."""
    calls = _read_jsonl(AGENT_CALLS, workdir)
    finish = {}
    events, sources = {}, {}
    out = {"calls": len(calls), "finish_reasons": finish, "null_content": 0, "empty_content": 0,
           "server_parser_events": events, "server_parser_recovered_from": sources,
           "token_ids_returned": 0,
           "stopped_in_reasoning": 0, "ran_past_stop": 0,
           "stop_sent": 0, "streamed": 0, "stopped_client_side": 0,
           "cut_at_max_tokens": 0, "cut_without_content": 0,
           "completion_tokens": 0, "reasoning_tokens": 0, "max_completion_tokens": 0,
           "generation_seconds": 0.0}
    for c in calls:
        reason = str(c.get("finish_reason"))
        finish[reason] = finish.get(reason, 0) + 1
        no_content = not c.get("content_chars")
        out["null_content"] += int(bool(c.get("content_null")))
        out["empty_content"] += int(no_content and not c.get("content_null"))
        if c.get("server_parser"):
            events[str(c["server_parser"])] = events.get(str(c["server_parser"]), 0) + 1
        if c.get("recovered_from"):
            sources[str(c["recovered_from"])] = sources.get(str(c["recovered_from"]), 0) + 1
        out["token_ids_returned"] += int(c.get("token_ids_returned") is not None)
        out["stopped_in_reasoning"] += int(bool(c.get("stopped_in_reasoning")))
        out["ran_past_stop"] += int(bool(c.get("ran_past_stop")))
        out["stop_sent"] += int(bool(c.get("stop_sent")))
        out["streamed"] += int(bool(c.get("streamed")))
        out["stopped_client_side"] += int(bool(c.get("stopped_client_side")))
        if reason == "length":
            out["cut_at_max_tokens"] += 1
            out["cut_without_content"] += int(no_content)
        out["completion_tokens"] += int(c.get("completion_tokens") or 0)
        out["reasoning_tokens"] += int(c.get("reasoning_tokens") or 0)
        out["max_completion_tokens"] = max(out["max_completion_tokens"],
                                           int(c.get("completion_tokens") or 0))
        out["generation_seconds"] += float(c.get("duration_s") or 0.0)
    out["generation_seconds"] = round(out["generation_seconds"], 1)
    return out


def judge_verdict_summary(workdir=None):
    """Counts over gbench_judge_verdicts.jsonl: soft checks by checker and verdict, and the
    Gemini models that answered them."""
    by_checker, models = {}, {}
    records = _read_jsonl(JUDGE_VERDICTS, workdir)
    for r in records:
        name = str(r.get("checker"))
        verdict = {True: "true", False: "false"}.get(r.get("verdict"), "no_verdict")
        slot = by_checker.setdefault(name, {"true": 0, "false": 0, "no_verdict": 0})
        slot[verdict] += 1
        for rep in r.get("replies") or []:
            model = rep.get("served_model") or "unknown"
            models[model] = models.get(model, 0) + 1
    return {"checks": len(records), "by_checker": by_checker, "answered_by": models}


def per_run_calls(workdir=None):
    """{run key: {agent_calls, no_content, cut_at_max_tokens, judge_checks, judge_models}}."""
    per = {}

    def slot(rec):
        key = _run_key(rec.get("config"), rec.get("phase_name"), rec.get("scenario_id"),
                       rec.get("run_number"))
        return per.setdefault(key, {"agent_calls": 0, "no_content": 0, "cut_at_max_tokens": 0,
                                    "judge_checks": 0, "judge_models": {}})
    for c in _read_jsonl(AGENT_CALLS, workdir):
        s = slot(c)
        s["agent_calls"] += 1
        s["no_content"] += int(not c.get("content_chars"))
        s["cut_at_max_tokens"] += int(c.get("finish_reason") == "length")
    for r in _read_jsonl(JUDGE_VERDICTS, workdir):
        s = slot(r)
        s["judge_checks"] += 1
        for rep in r.get("replies") or []:
            m = rep.get("served_model") or "unknown"
            s["judge_models"][m] = s["judge_models"].get(m, 0) + 1
    return per


def dataset_record(env=None):
    """The dataset the run read: the Hub id, the pinned revision of the image's local copy, the
    local copy, what --hf-dataset was given, and whether the Hub was off (HF_HUB_OFFLINE)."""
    env = os.environ if env is None else env
    return {"dataset": HUB_DATASET, "revision": env.get("GAIA2_DATASET_REVISION") or None,
            "filesystem_dataset": "meta-agents-research-environments/gaia2_filesystem",
            "filesystem_revision": env.get("GAIA2_FS_REVISION") or None,
            "local_copy": env.get("GAIA2_LOCAL_DATASET") or None,
            "filesystem_local_copy": env.get("DEMO_FS_PATH") or None,
            "hf_dataset_arg": env.get("GAIA2_HF_DATASET") or env.get("GAIA2_LOCAL_DATASET")
            or HUB_DATASET, "split": env.get("GAIA2_SPLIT", "validation"),
            "hub_offline": (env.get("HF_HUB_OFFLINE") or "").strip() in ("1", "true", "True")}


def hooks_record(workdir=None):
    """The hooked ARE process's HOOKS record ({token_ids_hook, detokenizer}), or {}."""
    try:
        with open(os.path.join(workdir or WORKDIR, HOOKS), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _summarize():
    """Parse ARE's benchmark_stats.json -> gbench summary. Never invents a scalar."""
    stats_path = os.path.join(WORKDIR, "benchmark_stats.json")
    if not os.path.exists(stats_path):
        return None
    try:
        data = json.load(open(stats_path, encoding="utf-8"))
    except Exception:
        return None
    stats = data.get("statistics") or {}
    g = stats.get("global") or {}
    per_cap = stats.get("per_capability") or {}
    stats_are_global = dict(g)
    # Guard the agent-less no-op: `are-benchmark run` without --agent runs scenarios agent-less,
    # never invokes the model, and STILL reports a real macro=0.0 that would be published as a fake
    # score. A genuine agentic run leaves at least one per-run lite trace carrying agent LLM usage /
    # interaction history; scan for that so gbench (gaia2.py) can hard-error if the model was never
    # exercised (measured 2026-09-11). Only report True when we positively see agent activity.
    agent_invoked = False
    # `run` writes WORKDIR/lite/, gaia2-run one lite/ per phase and config directory.
    for p in glob.glob(os.path.join(WORKDIR, "**", "lite", "*.json"), recursive=True):
        try:
            t = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if t.get("per_agent_llm_usage_stats") or t.get("per_agent_interaction_histories"):
            agent_invoked = True
            break
    # Headline = per-capability equal-weight macro success rate (the GAIA2 Overall). micro = per-
    # scenario. ARE writes every success rate in percent (0-100; report_stats.py multiplies the
    # mean of success_numeric by 100), and leaves runs it could not validate (no_validation) out
    # of them. The rates come from gbench_stats.json: ARE's statistics recomputed over the runs
    # without a harness error. ARE's own global block, harness errors included, is kept as
    # are_reported for provenance.
    try:
        gb = json.load(open(os.path.join(WORKDIR, GBENCH_STATS), encoding="utf-8"))
    except Exception:
        gb = None
    accounted = isinstance(gb, dict) and isinstance(gb.get("statistics"), dict)
    if accounted:
        stats = gb["statistics"]
        g = stats.get("global") or {}
        per_cap = stats.get("per_capability") or {}
    runs = [dict(r) for r in (gb or {}).get("runs") or [] if isinstance(r, dict)]
    per_run = per_run_calls()
    for r in runs:
        r.update(per_run.get(_run_key(r.get("config"), r.get("phase_name"), r.get("scenario_id"),
                                      r.get("run_number")), {}))
    configs = (gb or {}).get("configs")
    if not isinstance(configs, list):
        configs = _read_jsonl(CONFIGS)
    settings = agent_settings()
    return {
        "harness_accounting": accounted,
        "macro_success_rate": g.get("macro_success_rate"),
        # ARE's macro formula (a capability without a validated run counted as 0.0), when the
        # recomputed macro left such a capability out (macro_over_validated).
        "macro_success_rate_are_formula": g.get("macro_success_rate_are_formula"),
        "micro_success_rate": g.get("micro_success_rate"),
        "pass_at_k_percent": g.get("pass_at_k_percent"),
        "pass_k_percent": g.get("pass_k_percent"),
        "total_scenarios": g.get("total_scenarios"),
        "total_runs": g.get("total_runs"),
        "validated_runs": g.get("validated_runs"),
        "success_runs": g.get("success_runs"),
        "no_validation_runs": g.get("no_validation_runs"),
        "exception_runs": g.get("exception_runs"),
        "agent_invoked": agent_invoked,
        # ARE's per-capability block holds one success_rate (percent) and the run counts.
        "per_capability": {
            c: {k: (v or {}).get(k) for k in ("success_rate", "total_scenarios", "total_runs",
                                               "validated_runs", "no_validation_runs",
                                               "exception_runs", "pass_at_k_percent",
                                               "pass_k_percent")}
            for c, v in per_cap.items()
        },
        # Every scenario run (ARE's per-run row, its harness errors, its agent and judge calls);
        # runs left out of the rates as harness errors, by kind; failed calls no run was tied to.
        "n_runs": (gb or {}).get("n_runs"),
        "n_scenarios_run": len({(r.get("config"), r.get("scenario_id")) for r in runs}) or None,
        "n_timeout_runs": sum(1 for r in runs if r.get("exception_type") == SCENARIO_TIMEOUT),
        "n_harness_error_runs": (gb or {}).get("n_harness_error_runs"),
        "harness_error_kinds": (gb or {}).get("harness_error_kinds") or {},
        "n_harness_error_success_runs": (gb or {}).get("n_harness_error_success_runs") or 0,
        "harness_error_success_kinds": (gb or {}).get("harness_error_success_kinds") or {},
        # The capability configs ARE ran (each run_dataset call), those that raised, and the
        # capabilities the recomputed macro leaves out for want of a validated run.
        "configs_attempted": configs,
        "failed_configs": [c for c in configs if isinstance(c, dict) and not c.get("ok")],
        "capabilities_without_validated_run":
            (gb or {}).get("capabilities_without_validated_run") or [],
        "unattributed_call_failures": (gb or {}).get("unattributed_call_failures") or 0,
        "unattributed_call_failure_roles": (gb or {}).get("unattributed_call_failure_roles") or [],
        "harness_error_runs": [r for r in runs if r.get("harness_errors")],
        "runs": runs,
        "are_reported": {k: (stats_are_global or {}).get(k) for k in (
            "macro_success_rate", "micro_success_rate", "total_runs", "validated_runs",
            "no_validation_runs", "exception_runs")},
        "metadata": data.get("metadata"),
        "gemini_usage": _gemini_usage(),
        "harness_events": _harness_events(),
        "agent_protocol": {**settings, "stop_sent_to_server": server_stop(settings),
                           "stop_applied_by": ("server" if server_stop(settings)
                                               else "client_stream"),
                           "stream_options": (None if server_stop(settings)
                                              else dict(STREAM_OPTIONS))},
        "agent_calls": agent_call_summary(),
        "judge_verdicts": judge_verdict_summary(),
        "dataset": dataset_record(),
        "hooks": hooks_record(),
        "shard": shard_summary(),
        "selection": selection_summary(),
    }


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    shard = parse_shard(os.environ.get("GAIA2_SHARD"))
    if shard:
        selection = shard_selection(shard, run_configs())
        if not any(selection.values()):
            # No configured capability has a scenario in this shard: nothing runs on this node.
            empty = {"empty_shard": True, "shard": {"spec": f"{shard[0]}/{shard[1]}",
                                                    "per_config": [
                                                        {"config": c, "split": SPLIT,
                                                         "selected": 0}
                                                        for c in selection]},
                     "harness_accounting": True, "dataset": dataset_record()}
            with open(os.path.join(WORKDIR, "gaia2_summary.json"), "w", encoding="utf-8") as f:
                json.dump(empty, f, indent=2)
            sys.stderr.write(f"[gaia2] shard {shard[0]}/{shard[1]} holds no scenario of "
                             f"{', '.join(selection)}; ARE not started\n")
            return 0
    rc = _run_are()
    summary = _summarize()
    if summary is None:
        failed = [c for c in _read_jsonl(CONFIGS) if not c.get("ok")]
        if failed:
            sys.stderr.write("[gaia2] capability config(s) that raised in ARE: " + "; ".join(
                f"{c.get('capability')} ({c.get('error')}: {c.get('message')})"
                for c in failed) + "\n")
        sys.stderr.write(f"[gaia2] no benchmark_stats.json produced (are-benchmark rc={rc}); "
                         f"a harness failure, not a 0%.\n")
        # leave no summary file -> the gbench runner raises infra_required
        return rc or 1
    summary["are_returncode"] = rc
    with open(os.path.join(WORKDIR, "gaia2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    sys.stderr.write(f"[gaia2] summary: macro={summary.get('macro_success_rate')} "
                     f"micro={summary.get('micro_success_rate')} "
                     f"scenarios={summary.get('total_scenarios')} "
                     f"no_validation={summary.get('no_validation_runs')}\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--are-hooked":
        sys.exit(_run_hooked_are(sys.argv[2:]))
    sys.exit(main())
