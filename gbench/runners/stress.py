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

"""Stress test runner - capacity under a latency SLO.

SOTA serving-capacity methodology: drive the endpoint OPEN-LOOP at a target
arrival rate (QPS) with Poisson arrivals and no concurrency cap, sweep the
rate to find the maximum sustainable rate ("goodput") that keeps BOTH P99
TTFT and P99 TPOT within their SLOs. This is what academic/industry serving
benchmarks (MLPerf server scenario, vLLM/Orca/Sarathi/DistServe) report -
not a closed-loop concurrency ceiling.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import re
import math
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import BenchmarkConfig, get_max_model_len
from ..core.models import ModelConfig, ModelFormat
from ..utils.log_manager import write_redacted_json
from ..analysis.perf_load import (
    INDETERMINATE, NOT_DETECTED, QG_ALPHA, QG_BATCHES, QG_MATERIAL_SLOPE, SATURATED, assign_workers,
    cohort, completion_bound_s,
    completion_fraction, merged_schedule, offered_rate_ci, percentile_inf, queue_growth,
    request_count,
)
from ..analysis.perf_sweep import (
    contaminated, drain_gap_s, drained, engine_above_client, foreign_requests, reseed_start,
    summarize_brackets,
)
from ..analysis.perf_method import (
    bucket_index, counter_delta, hist_delta, hist_itl_gate, hist_quantile_bucket,
    methodology_record, parse_metrics, per_token_gaps, per_token_tpot_ms,
    smallest_finite_edge_ms,
)
from .perf_payload import PayloadFactory, chat_messages, special_token_count, stable_u32
from .serving import ServingBenchmarkRunner

# vLLM benchmark imports will be loaded dynamically in active methods

logger = logging.getLogger(__name__)


def _tokenizer_trust(config) -> bool:
    """trust_remote_code for gbench's own tokenizer loads (core.config.tokenizer_trust_remote_code)."""
    from ..core.config import tokenizer_trust_remote_code
    return tokenizer_trust_remote_code(config)[0]


def _delta_piece(choice: dict, content_field: str) -> str:
    """Extract the streamed text from one SSE `choices[i]` object.

    Counts content OR reasoning/thinking tokens: a reasoning model (e.g. gemma via
    Ollama) streams its reply as `reasoning`/`reasoning_content` deltas, which is real
    decode - not an empty completion. Mirrors the serving pillar's parser; ignoring
    them made every stress request look like a 0-token reply -> 'no usable stream'."""
    if content_field == "delta":
        d = choice.get("delta") or {}
        return d.get("content") or d.get("reasoning") or d.get("reasoning_content") or ""
    return choice.get("text") or ""


def _stress_client_worker(task: dict) -> dict:
    """Run ONE client process's share of an open-loop rate point.

    Module-level (picklable for the 'spawn' start method) so K copies run in
    separate OS processes - this is what removes the single-asyncio-client
    CPU-contention confound at the knee: SSE parsing for many concurrent streams
    no longer starves one event loop and inflates client-side TTFT/TPOT. With
    task["t0"] and task["offsets"] it sends request k at the absolute epoch
    t0 + offsets[k] (the parent's merged schedule), recording planned time and lag;
    without them (warmup, preflight) it sends back to back. Uses
    epoch ``time.time()`` (NOT perf_counter, whose origin is per-process) so the
    parent can merge records across workers on a common clock. Records per request:
    planned/send/lag, ttft, tpot, itls (+ the text of each chunk), done, ok,
    usage tokens and finish_reason.

    Args:
        task: dict with payloads, idx, t0 + offsets (scheduled points), api_url,
            content_field, request_timeout, max_point_s, itl_cap, headers.

    Returns:
        {"records": [...], "ready_at": epoch when the worker could start sending}.
    """
    import aiohttp  # local import: worker-only dependency

    payloads = task["payloads"]
    qps = float(task.get("qps") or 0.0)
    seed = int(task.get("seed") or 0)
    api_url = task["api_url"]
    content_field = task["content_field"]
    headers = task.get("headers") or None       # the endpoint key, from gbench's own process
    request_timeout = float(task["request_timeout"])
    max_point_s = float(task["max_point_s"])
    itl_cap = int(task["itl_cap"])

    rng = random.Random(seed)
    records: list = []
    state = {"sched_lag_s": 0.0}

    async def _run():
        connector = aiohttp.TCPConnector(limit=0)  # open-loop: no client cap
        timeout = aiohttp.ClientTimeout(total=request_timeout)

        async def _send(session, payload, idx, planned=None):
            now = time.time()
            rec = {"send": now, "ttft": None, "tpot": None,
                   "itls": None, "done": None, "ok": False, "idx": idx,
                   "planned": planned, "lag": (now - planned) if planned is not None else None,
                   "prompt_tokens": None, "completion_tokens": None, "finish_reason": None}
            records.append(rec)
            ct = []      # arrival epoch of each streamed content chunk
            pieces = []  # the text of each chunk (tokens per chunk are counted later,
                         # in the parent, never on the send path)
            try:
                async with session.post(api_url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        rec["done"] = time.time()
                        return
                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:") or "[DONE]" in line:
                            continue
                        try:
                            obj = json.loads(line[5:].strip())
                            usage = obj.get("usage") or {}
                            if usage.get("prompt_tokens") is not None:
                                rec["prompt_tokens"] = int(usage["prompt_tokens"])
                            if usage.get("completion_tokens") is not None:
                                rec["completion_tokens"] = int(usage["completion_tokens"])
                            chs = obj.get("choices") or []
                            ch = chs[0] if chs else {}
                            if ch.get("finish_reason"):
                                rec["finish_reason"] = ch["finish_reason"]
                            piece = _delta_piece(ch, content_field)
                        except Exception:
                            piece = ""
                        if piece:
                            ct.append(time.time())
                            pieces.append(piece)
                rec["done"] = time.time()
                if ct:
                    rec["ttft"] = (ct[0] - rec["send"]) * 1000.0
                    if len(ct) >= 2:
                        gaps = [(ct[j] - ct[j - 1]) * 1000.0
                                for j in range(1, len(ct))]
                        rec["tpot"] = sum(gaps) / len(gaps)
                        # Subsample gaps (uniform, preserves the distribution) so
                        # the parent's pooled P99-ITL tail gate stays cheap to
                        # pickle back on long-decode (8192 gaps/req).
                        if len(gaps) > itl_cap:
                            step = len(gaps) / itl_cap
                            keep = [int(k * step) for k in range(itl_cap)]
                        else:
                            keep = range(len(gaps))
                        rec["itls"] = [gaps[j] for j in keep]
                        # gap j ends when chunk j+1 arrives: its tokens are what the gap
                        # delivered (the parent turns this into per-token gaps).
                        rec["itl_pieces"] = [pieces[j + 1] for j in keep]
                    # A stream the server ended with abort/error is a failure even if
                    # it carried tokens.
                    rec["ok"] = rec["finish_reason"] not in ("abort", "error")
                elif (rec["finish_reason"] in ("stop", "length")
                      and (rec["completion_tokens"] or 0) > 0):
                    # A reply the server finished whose tokens carried no visible text
                    # (special tokens a non-vLLM server strips, partial UTF-8 bytes): it
                    # was served, so it is measured, with its end-to-end time as an upper
                    # bound for TTFT (flagged; no gaps).
                    rec["ttft"] = (rec["done"] - rec["send"]) * 1000.0
                    rec["ttft_unseen"] = True
                    rec["ok"] = True
            except Exception:
                rec["done"] = time.time()

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            idxs = task.get("idx") or list(range(len(payloads)))
            state["ready_at"] = time.time()
            if task.get("t0") is not None:
                # The point's merged schedule: send request k at the absolute epoch
                # t0 + offsets[k]. A send already past due goes out at once and its lag
                # is recorded (the client-bound checks read it).
                t0, offsets = float(task["t0"]), task["offsets"]
                for k, payload in enumerate(payloads):
                    planned = t0 + float(offsets[k])
                    delay = planned - time.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    tasks.append(asyncio.create_task(_send(session, payload, idxs[k], planned)))
            else:
                # Single-stream probes (warmup, preflight): back to back.
                for k, payload in enumerate(payloads):
                    tasks.append(asyncio.create_task(_send(session, payload, idxs[k])))
            pending = set(tasks)
            deadline = time.time() + max_point_s
            while pending:
                rem = deadline - time.time()
                if rem <= 0:
                    for t in pending:
                        t.cancel()
                    break
                _d, pending = await asyncio.wait(pending, timeout=min(30.0, rem))

    asyncio.run(_run())
    return {"records": records, "sched_lag_s": state["sched_lag_s"],
            "ready_at": state.get("ready_at")}


class StressContaminated(RuntimeError):
    """Other traffic kept the server busy (a point stayed contaminated after a re-measure,
    or the server never drained): the pillar ends with no headline."""


def certify_settings(config) -> tuple:
    """(mode, budget_min) of the certification step: flag, then env, then default (an
    explicit 0 budget is kept: it means "do not certify" in auto mode)."""
    mode = getattr(config, "stress_certify", None)
    if mode is None:
        mode = os.environ.get("GBENCH_STRESS_CERTIFY") or "auto"
    budget = getattr(config, "stress_certify_budget", None)
    if budget is None:
        env = os.environ.get("GBENCH_STRESS_CERTIFY_BUDGET_MIN")
        budget = env if env not in (None, "") else CERTIFY_BUDGET_MIN_DEFAULT
    budget = float(budget)
    if budget < 0:
        raise ValueError(f"certification budget must be >= 0 minutes (got {budget:g})")
    return str(mode).lower(), budget


CERTIFY_BUDGET_MIN_DEFAULT = 120.0


def _sig(x: float, digits: int = 4) -> float:
    """A rate rounded to significant digits (fixed decimals collapse a slow device's
    rates to 0 or stop halving)."""
    return float(f"{float(x):.{digits}g}")


def _noop(_x=None):
    """Picklable no-op: starts a client process (pool warm-up)."""
    return None


def _knee_stats(knees: list) -> dict:
    """Per-repeat bracket passes: count, median, min and max. No mean/CV/CI: with a few
    repeats on a x2 grid those are false precision (the brackets and the certification
    carry the uncertainty)."""
    vals = np.asarray([k for k in knees if k is not None and np.isfinite(k)], dtype=float)
    out = {"reps": int(vals.size)}
    if vals.size:
        out.update(median=float(np.median(vals)), min=float(np.min(vals)),
                   max=float(np.max(vals)))
    return out


class StressTestRunner:
    """Runner for stress testing to find max sustainable throughput under SLO.

    Sweeps the OPEN-LOOP arrival rate (QPS, Poisson) to find the maximum
    sustainable rate that keeps P99 TTFT and P99 TPOT within their SLOs.
    Supports both text-only and multimodal stress tests.
    """

    # ── Open-loop request-rate (QPS) sweep parameters ─────────
    # Stress finds the max sustainable arrival rate (goodput) that keeps both
    # latency SLOs - the SOTA capacity-under-SLO measurement, not a concurrency
    # ceiling. Load is Poisson (exponential inter-arrival), concurrency uncapped.
    INITIAL_QPS = 4.0            # starting arrival rate (req/s)
    MIN_QPS = 0.5               # lower bound for a downward search
    MIN_DOWN_QPS = 0.05         # floor for the ascending sweep's rare downward fallback
    MAX_QPS = 8192.0           # safety cap: high enough that a fast multi-GPU server finds its
                               # real SLO-break knee out of the box (512 censored fast servers -
                               # they passed the SLO at every point up to the cap). Override with
                               # --stress-max-qps for an even more powerful host.

    # ── One rate point ────────────────────────────────────────────────────────
    # A point sends max(MIN_PROMPTS, rate x MEASURE_ARRIVAL_S, rate x (3 x latency + 10 s))
    # requests on one merged Poisson schedule (perf_load.request_count; never fewer) and is
    # judged on its cohort: the requests PLANNED after its ramp-up (median latency). See
    # _test_rate for the gates.
    MEASURE_ARRIVAL_S = 60      # seconds of arrivals at least
    MIN_PROMPTS = 40            # request floor per point
    MAX_PROMPTS = 8000          # used only as an explicit --stress-max-prompts cap
    MIN_STEADY = 15             # min cohort requests to judge a point

    # ── Multi-process load generator ──────────────────────────────────────────
    # SSE parsing is spread across up to NUM_CLIENT_PROCS OS processes so a single
    # Python event loop's CPU cost never becomes the knee (the classic
    # single-asyncio-client confound). The parent draws ONE merged Poisson schedule
    # (perf_load.merged_schedule) and gives arrival i to worker i mod k; every worker
    # sends at absolute epoch times, so the offered process is the same for any k.
    NUM_CLIENT_PROCS = 32       # max client processes (capped to cpu_count-1); small hosts
                               # auto-cap to cpu_count-1, and low rates use 1 (below).
    ITL_SAMPLE_CAP = 512        # max inter-token gaps kept per request (P99 pool)
    # Minimum requests per worker: low-rate points use one worker (a worker per handful
    # of requests only adds process overhead); multi-process starts where the client
    # CPU cost is the real risk.
    MIN_PROMPTS_PER_WORKER = 30

    # ── Repeats ────────────────────────────────────────────────────────────────
    # The sweep runs STRESS_REPS times with varied seeds; each gives a bracket
    # [pass, fail], the median repeat's limit is certified (_certify_knee), and the
    # result says whether the brackets agree.
    STRESS_REPS = 3
    # Share of a point that is not in its cohort (the first L50 of sends), used only to
    # size the prompt floor when --stress-min-samples lowers the sample floor.
    FILL_FRAC = 0.30
    DRAIN_FRAC = 0.15
    # Completion-fraction gate: at least this share of the cohort must finish within the
    # end-to-end time the SLOs imply (TTFT SLO + (max_tokens - 1) x ITL SLO). Calibrated
    # on simulated servers (docs/perf_evidence/step5/completion_calib_*.txt): 0% false
    # fails under capacity for 1, 4 and 256 slots; 96% detection of 1.25x overload at
    # 256 slots. The queue-growth test (perf_load) is the primary saturation check.
    KEEP_UP_FRAC = 0.90
    MAX_FAILURE_FRAC = 0.02     # a point is INVALID if > this fraction of OFFERED
                                # requests error/drop (non-200, timeout, truncated).
                                # Failures leave achieved & slo_attainment (both
                                # successful-only denominators) untouched, so an
                                # ungated point with drops silently inflates the knee.
    REQUEST_TIMEOUT_S = 900     # per-request ceiling floor; raised to 1.25 x the SLO-implied
                                # e2e for long outputs (_request_timeout)
    MAX_POINT_S = 1200          # wait after the LAST send for stragglers; raised the same
                                # way (_max_point_s), so a request the SLOs allow is never cut
    WARMUP_PROMPTS = 20         # (legacy) count; the gentle warmup below uses fewer, sequential
    WARMUP_PROMPTS_GENTLE = 3   # sequential (concurrency-1) warmup requests - never floods the queue
    PREFLIGHT_PROBES = 3        # sequential single-stream requests to establish the latency floor

    # Multimodal stress request shape. MM input is images + padded text sized to
    # the campaign's input length; output = campaign output (ignore_eos). Needs
    # the server started with --limit-mm-per-prompt image=IMAGES_PER_REQUEST.
    # The per-image token count is the one the MM preflight measures on the server
    # (ServingBenchmarkRunner._assert_mm_images_processed), else the model's declared count.
    IMAGES_PER_REQUEST = 4
    MIN_MM_TEXT_TOKENS = 16

    # ── Dual latency SLO (P99) ────────────────────────────────
    # TTFT = prefill responsiveness; TPOT = decode smoothness. A rate is
    # "sustainable" only if BOTH percentiles hold.
    DEFAULT_TTFT_THRESHOLD_MS = 5000   # laptop-friendly default (CLI overridable)
    DEFAULT_TPOT_THRESHOLD_MS = 200    # ~5 tok/s per stream floor (CLI overridable)

    def __init__(
        self,
        config: BenchmarkConfig,
        ttft_threshold_ms: int = None,
        tpot_threshold_ms: int = None,
    ):
        """Initialize stress test runner.

        Args:
            config: Benchmark configuration
            ttft_threshold_ms: P99 TTFT SLO in ms (overrides default)
            tpot_threshold_ms: P99 TPOT SLO in ms (overrides default)
        """
        self.config = config
        self.ttft_threshold_ms = ttft_threshold_ms or self.DEFAULT_TTFT_THRESHOLD_MS
        self.tpot_threshold_ms = tpot_threshold_ms or self.DEFAULT_TPOT_THRESHOLD_MS
        self.initial_qps = 1.0 if config.remote_endpoint else self.INITIAL_QPS
        self._serving_runner: Optional[ServingBenchmarkRunner] = None
        self._tested_rates = {}  # Cache tested QPS points (current rep)
        self._tokenizer = None  # Cached tokenizer
        self._api_model_id = None  # Cached served model id
        self._vision_tokens_per_image: Optional[int] = None
        self._vision_tokens_source: Optional[str] = None
        self._multimodal = False  # Whether current run is multimodal
        self._mm_image_dir = None  # Temp dir for generated images
        self._mm_image_paths = []  # Generated image file paths
        # Client processes: cap to cpu_count-1 (leave one core for the parent +
        # OS), overridable via config.stress_client_procs. 'spawn' context avoids
        # inheriting any parent state into the workers.
        # Flag (config) > GBENCH_STRESS_* env > class default, uniformly for every
        # stress knob (client procs / reps / max qps / max prompts).
        want = (getattr(config, "stress_client_procs", None)
                or (os.environ.get("GBENCH_STRESS_CLIENT_PROCS") or None)
                or self.NUM_CLIENT_PROCS)
        self._num_client_procs = max(1, min(int(want), max(1, (os.cpu_count() or 2) - 1)))
        self._reps = int(getattr(config, "stress_reps", None)
                         or (os.environ.get("GBENCH_STRESS_REPS") or None)
                         or self.STRESS_REPS)
        # MAX_QPS safety cap + MAX_PROMPTS steady-window size are overridable so a
        # fast model on strong hardware is not censored at the 512 default. Precedence:
        # --stress-max-qps / --stress-max-prompts (config) > GBENCH_STRESS_* env > default.
        _mq = getattr(config, "stress_max_qps", None)
        if _mq is None:
            _mq = os.environ.get("GBENCH_STRESS_MAX_QPS") or None
        self.max_qps = float(_mq) if _mq is not None else self.MAX_QPS
        _mp = getattr(config, "stress_max_prompts", None)
        if _mp is None:
            _mp = os.environ.get("GBENCH_STRESS_MAX_PROMPTS") or None
        self.max_prompts = int(_mp) if _mp is not None else self.MAX_PROMPTS
        # No cap by default (points get the requests the method needs; that only ever
        # adds requests); an explicit value is honoured as a recorded cap.
        self._max_prompts_explicit = _mp is not None
        # Min cohort requests to judge a point (a lower value only makes the bracket sweep
        # cheaper; certification probes are sized for the percentile). Precedence:
        # --stress-min-samples (config) > GBENCH_STRESS_MIN_SAMPLES env > class default.
        _msamp = getattr(config, "stress_min_samples", None)
        if _msamp is None:
            _msamp = os.environ.get("GBENCH_STRESS_MIN_SAMPLES") or None
        self._min_steady = max(1, int(_msamp)) if _msamp is not None else self.MIN_STEADY
        # Prompts-per-point floor scales with the sample floor: enough to yield
        # ~_min_steady completions after dropping FILL/DRAIN, capped at the class
        # default (so the DEFAULT min-samples keeps the original 40-prompt floor and
        # byte-for-byte behavior; a lower floor makes each point smaller and faster).
        _steady_frac = max(0.1, 1.0 - self.FILL_FRAC - self.DRAIN_FRAC)
        self._min_prompts = (self.MIN_PROMPTS if self._min_steady >= self.MIN_STEADY
                             else min(self.MIN_PROMPTS,
                                      max(self._min_steady + 3,
                                          int(self._min_steady / _steady_frac) + 3)))
        self._rep_seed = 83     # base seed for the current rep (set per rep)
        self._rep_idx = 0       # current rep (payload nonce)
        self._point_seq = 0     # measured points so far in this rep (payload nonce)
        # Request content: the seed reproduces lengths/base tokens/base images, a fresh
        # salt per invocation makes every byte new (perf_payload). GBENCH_PERF_SEED
        # overrides the seed.
        self._payloads = PayloadFactory(seed=int(os.environ.get("GBENCH_PERF_SEED") or 83))
        self._payload_meta: list = []
        self._point_log: list = []      # every measurement of the current run
        self._cert_points: list = []
        self._prompt_overhead = None   # server prompt tokens beyond the planned text
        self._media_dirs: list = []
        self._real_pool: list = []
        self._real_seen: set = set()
        self._real_cursor = 0
        self._prompt_reuse = 0
        # Single-stream latency measured by the preflight, used to seed the ascending
        # sweep's start rate near half the implied capacity (set in _preflight_floor_check).
        self._probe_ttft_ms = None
        self._probe_e2e_ms = None
        self._client_pool = None  # reusable multi-process client pool (per sweep)
        # Multiprocessing context for the client pool. 'spawn' avoids inheriting any
        # parent state (asyncio loop, sockets) into the workers. This used to sit after a
        # `return` in _images_per_request() -> unreachable, so `_mp_ctx` was never set and
        # every _get_client_pool() raised AttributeError, silently forcing the
        # single-process fallback while the artifact still claimed multi-process. Init here.
        try:
            self._mp_ctx = mp.get_context("spawn")
        except ValueError:
            self._mp_ctx = mp.get_context()

    def _images_per_request(self) -> int:
        """Images per MM request: config override (--personal sets 1) else default 4."""
        return int(getattr(self.config, "mm_images_per_request", None) or self.IMAGES_PER_REQUEST)

    def _request_headers(self) -> dict:
        """The endpoint key header for the load generator's requests (VLLM_API_KEY, else
        OPENAI_API_KEY; base.endpoint_auth_headers), {} when none is set. The client workers
        are gbench's own host processes."""
        from .eval_suites.base import endpoint_auth_headers
        return endpoint_auth_headers()

    def _generate_requests(self, num_prompts: int, model=None, format=None) -> list:
        """Generate sample requests for the stress sweep.

        Neutral default is a fixed 128-in / 128-out random shape - the laptop
        smoke-test workload. When the user pins a geometry (a --campaign or
        explicit --input/--output-lengths, i.e. config.workload_shape_explicit),
        stress the model at THAT shape and dataset instead, reusing the serving
        runner's dataset generation (random / sharegpt / custom / hf).
        """
        if (getattr(self.config, "workload_shape_explicit", False)
                and self._serving_runner is not None and model is not None):
            in_len = self.config.input_lengths[0] if self.config.input_lengths else 128
            out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
            requests, tok = self._serving_runner._generate_sample_requests(
                model=model,
                format=format,
                num_prompts=num_prompts,
                input_len=in_len,
                output_len=out_len,
            )
            if tok is not None:
                self._tokenizer = tok
            return requests

        # Synthetic prompts are built per point in _build_payloads; this is only reached
        # for a real dataset with a pinned geometry (above).
        raise RuntimeError("stress real-dataset prompts need a pinned workload (--campaign or "
                           "explicit --input-lengths/--output-lengths) and a loaded model")

    def _workload_shape(self) -> tuple:
        """(dataset, input_len, output_len) the stress sweep will use."""
        if getattr(self.config, "workload_shape_explicit", False):
            in_len = self.config.input_lengths[0] if self.config.input_lengths else 128
            out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
            return (self.config.dataset, in_len, out_len)
        return ("random", 128, 128)

    def _per_image(self) -> int:
        """The per-image token count the MM workload is sized with (measured by the preflight,
        else the model's declared count). Raises when neither is known."""
        if not self._vision_tokens_per_image:
            raise RuntimeError("the per-image token count is unknown: the model declares none "
                               "and the multimodal preflight did not measure one")
        return int(self._vision_tokens_per_image)

    def _mm_text_tokens(self, in_len: int, out_len: int) -> int:
        """Text-token budget for a MM request: the FULL campaign input text, with
        the images added ON TOP (not replacing text), so MM is a strict superset
        of the text workload (same text + 4 images) and is directly comparable to
        it. Only clamped down if text + images + output would overflow the context.
        Shared by the request builder AND the reported workload length so the two
        never drift."""
        img_tokens = self._images_per_request() * self._per_image()
        max_tokens = max(int(out_len), 16)
        max_len = self._context_window()
        return max(self.MIN_MM_TEXT_TOKENS,
                   min(int(in_len),
                       max_len - img_tokens - max_tokens - 128))

    def _mm_effective_input_len(self, in_len: int, out_len: int) -> int:
        """Actual MM prefill length = clamped text tokens + image placeholder
        tokens. This is what the server prefills (and what drives TTFT), NOT the
        campaign's nominal text input - for small-input campaigns the images
        dominate, so labeling the row with the nominal input understates it ~9x."""
        return int(self._mm_text_tokens(in_len, out_len)
                   + self._images_per_request() * self._per_image())

    def _workload_desc(self) -> str:
        """Human-readable workload description for the config banner."""
        _ds, in_len, out_len = self._workload_shape()
        if getattr(self, "_multimodal", False):
            return (f"multimodal ({self._images_per_request()} unique 256x256 images + text, "
                    f"{max(out_len, 16)} out)")
        if _ds == "sharegpt":
            return "sharegpt (dynamic real prompt lengths)"
        return f"{_ds} in={in_len} out={out_len}"

    def _setup_mm_images(self, num_images: int = 50) -> str:
        """Generate synthetic images for multimodal stress test.

        Creates a temp directory with random JPEG images.

        Args:
            num_images: Number of images to generate

        Returns:
            Path to the temp directory containing images
        """
        import numpy as np
        from PIL import Image

        img_dir = tempfile.mkdtemp(prefix="stress_mm_images_")
        self._mm_image_paths = []

        for i in range(num_images):
            # 256x256 random images - small enough for fast I/O,
            # large enough to exercise the vision encoder
            img = Image.fromarray(self._payloads.base_image(i))
            img_path = os.path.join(img_dir, f"stress_img_{i:03d}.jpg")
            img.save(img_path, format="JPEG", quality=85)
            self._mm_image_paths.append(img_path)

        logger.info(f"Generated {num_images} synthetic images in {img_dir}")
        self._mm_image_dir = img_dir
        return img_dir

    def _cleanup_mm_images(self):
        """Clean up temp image directory."""
        if self._mm_image_dir and os.path.exists(self._mm_image_dir):
            import shutil
            shutil.rmtree(self._mm_image_dir, ignore_errors=True)
            logger.info(f"Cleaned up MM image directory: {self._mm_image_dir}")
            self._mm_image_dir = None
            self._mm_image_paths = []

    MM_FILLER = "Analyze the attached images and the following context in detail. "

    def _build_text(self, n_tokens: int, phase: str = "sweep", point: int = 0,
                    i: int = 0) -> tuple:
        """MM user text of n_tokens tokens, nonce first: (text, mismatch); mismatch is
        None without a tokenizer (length approximate)."""
        return self._payloads.filler_prompt(self._tokenizer, int(n_tokens), self.MM_FILLER,
                                            phase, self._rep_idx, point, i)

    # ── Request content (perf_payload): unique by construction ────────────────
    def _chat_messages(self, content) -> list:
        """The shared perf message layout: the no-think system prompt only in no-think
        mode (never under --thinking), then the user turn. Same for text and MM."""
        sr = self._serving_runner
        sys_prompt = (None if getattr(self.config, "thinking", False)
                      else getattr(sr, "NOTHINK_SYSTEM", None))
        return chat_messages(content, sys_prompt)

    def _request_extras(self, phase: str, point: int, real_dataset: bool) -> dict:
        """Fields every payload carries besides content: token-accurate usage in the
        final chunk, and on a vLLM server a cache_salt (per run for synthetic content,
        per point for real-dataset prompts that may repeat) and skip_special_tokens False
        (a reply of special tokens would otherwise stream as empty chunks: no first token
        is seen and the request reads as never finished)."""
        extra = {"stream_options": {"include_usage": True}}
        sr = self._serving_runner
        is_vllm = getattr(sr, "is_vllm_server", None)
        if callable(is_vllm) and is_vllm():
            salt = self._payloads.salt
            extra["cache_salt"] = (f"{salt}.{phase}.{self._rep_idx}.{point}" if real_dataset
                                   else salt)
            extra["skip_special_tokens"] = False
        return extra

    def _context_window(self) -> int:
        """The served context length: what the remote endpoint reports, else the window
        gbench serves at (never the 4096 default for a remote that reports more)."""
        sr = self._serving_runner
        fn = getattr(sr, "served_max_model_len", None)
        if callable(fn):
            v = fn()
            if v:
                return int(v)
        return get_max_model_len(getattr(self.config, "max_model_len", None))

    def _point_media_dir(self, phase: str, point: int) -> Optional[str]:
        """A fresh directory under the allowed local media path for one point's images
        (local server only; a remote endpoint gets data URLs)."""
        if self.config.remote_endpoint or not self._mm_image_dir:
            return None
        d = os.path.join(self._mm_image_dir, f"{phase}_{self._rep_idx:02d}_{point:03d}")
        os.makedirs(d, exist_ok=True)
        self._media_dirs.append(d)
        return d

    def _release_media(self) -> None:
        """Delete the per-point image files once their requests are done."""
        import shutil
        for d in self._media_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._media_dirs = []

    def _real_dataset(self) -> bool:
        ds, _i, _o = self._workload_shape()
        return ds not in ("random",) and getattr(self.config, "workload_shape_explicit", False)

    def _real_prompts(self, n: int, model, format) -> list:
        """n real-dataset prompts, disjoint from every prompt this runner already sent
        (never edited). When the dataset runs out, prompts repeat: counted in
        self._prompt_reuse and isolated per point by cache_salt on vLLM."""
        out = []
        tries = 0
        while len(out) < n and tries < 4:
            want = max(n, 2 * len(self._real_pool) or n)
            if len(self._real_pool) - self._real_cursor < n - len(out):
                fresh = self._generate_requests(len(self._real_seen) + want, model, format)
                for r in fresh:
                    pr = getattr(r, "prompt", "")
                    if pr not in self._real_seen:
                        self._real_seen.add(pr)
                        self._real_pool.append(r)
            take = self._real_pool[self._real_cursor:self._real_cursor + (n - len(out))]
            self._real_cursor += len(take)
            out.extend(take)
            tries += 1
        if len(out) < n and self._real_pool:
            short = n - len(out)
            self._prompt_reuse += short
            out.extend(self._real_pool[k % len(self._real_pool)] for k in range(short))
        return out

    def _build_payloads(self, model, format, num_prompts: int, phase: str = "sweep",
                        point: int = 0) -> tuple:
        """Build (api_url, content_field, [payload, ...]) for one batch of requests.

        Built once in the PARENT before the point's clock starts and split across the
        client workers. Every request is new to the server's caches (perf_payload): a
        nonce from the run salt and (phase, rep, point, index) leads the user text, and
        multimodal images carry a per-request code. Text uses /v1/completions on a local
        vLLM and a coherent chat prompt on a remote endpoint (which may ignore
        ignore_eos); multimodal uses /v1/chat/completions with N images + campaign-sized
        text. self._payload_meta[i] holds payload i's planned prompt tokens (None when
        not planned) and its length tolerance.
        """
        model_path = self._api_model_id or self._serving_runner._resolve_model_id(model, format)
        # Normalize like serving.py: a remote endpoint already ending in /v1 must
        # not get a second /v1 appended (that 404s every request). Works for the
        # local path too (base_url has no /v1 -> the /v1 branch is taken).
        base_url = (self.config.remote_endpoint or f"http://127.0.0.1:{self._serving_runner.server_port}").rstrip("/")
        _ds, in_len, out_len = self._workload_shape()
        max_tokens = max(int(out_len), 16)
        # Portable no-think for the perf workload (mirrors serving); {} under --thinking.
        _snt = getattr(self._serving_runner, "no_think_fields", None)
        nt = _snt() if callable(_snt) else {}
        f = self._payloads
        rep = self._rep_idx
        real = self._real_dataset()
        extra = self._request_extras(phase, point, real)
        meta = []

        if self._multimodal:
            api_url = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
            n_img = self._images_per_request()
            text_tokens = self._mm_text_tokens(in_len, out_len) if num_prompts > 0 else 0
            media = self._point_media_dir(phase, point) if num_prompts > 0 else None
            payloads = []
            for i in range(num_prompts):
                text, mism = self._build_text(text_tokens, phase, point, i)
                urls = []
                for k in range(n_img):
                    data = f.unique_image_jpeg(phase, rep, point, i, k)
                    if media is None:
                        import base64
                        urls.append("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
                    else:
                        path = os.path.join(media, f"r{i:06d}_{k}.jpg")
                        with open(path, "wb") as fh:
                            fh.write(data)
                        urls.append(f"file://{path}")
                content = [{"type": "text", "text": text}]
                content += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
                payloads.append({
                    "model": model_path, "messages": self._chat_messages(content),
                    "max_tokens": max_tokens, "ignore_eos": True, "stream": True, **nt, **extra,
                })
                meta.append({"planned": (text_tokens + n_img * self._per_image()
                                         if mism is not None else None),
                             "tol": abs(mism or 0)})
            self._payload_meta = meta
            return api_url, "delta", payloads

        if real:
            # Real dataset: prompts are never edited; disjoint per point (see _real_prompts).
            reqs = self._real_prompts(num_prompts, model, format)
            prompts = [getattr(r, "prompt", "") for r in reqs]
            if self.config.remote_endpoint:
                chat_url = (f"{base_url}/chat/completions" if base_url.endswith("/v1")
                            else f"{base_url}/v1/chat/completions")
                payloads = [{
                    "model": model_path, "messages": self._chat_messages(pr),
                    "max_tokens": max_tokens, "ignore_eos": True, "stream": True, **nt, **extra,
                } for pr in prompts]
                self._payload_meta = [{"planned": None, "tol": 0}] * len(payloads)
                return chat_url, "delta", payloads
            api_url = f"{base_url}/completions" if base_url.endswith("/v1") else f"{base_url}/v1/completions"
            payloads = [{"model": model_path, "prompt": pr, "max_tokens": max_tokens,
                         "ignore_eos": True, "stream": True, **nt, **extra} for pr in prompts]
            self._payload_meta = [{"planned": None, "tol": 0}] * len(payloads)
            return api_url, "text", payloads

        # REMOTE endpoints (e.g. Ollama) commonly IGNORE ignore_eos, so a random-token
        # /v1/completions prompt draws an immediate-EOS (empty) reply - every rate point
        # then records 0 completions and an infinite TTFT. Mirror the serving pillar's
        # proven remote path: a COHERENT chat prompt of in_len tokens + a long-output
        # directive over /v1/chat/completions, which reliably sustains output. LOCAL vLLM
        # keeps the controlled-shape random-token prompts + /v1/completions + ignore_eos.
        if (self.config.remote_endpoint and self._serving_runner is not None
                and self._tokenizer is not None):
            chat_url = (f"{base_url}/chat/completions" if base_url.endswith("/v1")
                        else f"{base_url}/v1/chat/completions")
            passage, directive = self._serving_runner._text_prompt_parts(out_len)
            payloads = []
            for i in range(num_prompts):
                text, mism = f.filler_prompt(self._tokenizer, int(in_len), passage, phase, rep,
                                             point, i, suffix=directive)
                payloads.append({
                    "model": model_path, "messages": self._chat_messages(text),
                    "max_tokens": max_tokens, "ignore_eos": True, "stream": True, **nt, **extra,
                })
                meta.append({"planned": int(in_len) if mism is not None else None,
                             "tol": abs(mism or 0)})
            self._payload_meta = meta
            return chat_url, "delta", payloads

        api_url = f"{base_url}/completions" if base_url.endswith("/v1") else f"{base_url}/v1/completions"
        explicit = getattr(self.config, "workload_shape_explicit", False)
        lens = f.planned_lengths(num_prompts, int(in_len), 0.0 if explicit else 0.5,
                                 phase, rep, point)
        payloads = []
        for i in range(num_prompts):
            if self._tokenizer is None:
                raise RuntimeError("the stress test sizes its prompts with the model's tokenizer: "
                                   "pass --tokenizer <hf-id-or-path> and install transformers")
            text, mism = f.random_token_prompt(
                self._tokenizer, max(1, lens[i] - special_token_count(self._tokenizer)),
                phase, rep, point, i)
            payloads.append({"model": model_path, "prompt": text, "max_tokens": max_tokens,
                             "ignore_eos": True, "stream": True, **nt, **extra})
            meta.append({"planned": (max(1, lens[i] - special_token_count(self._tokenizer))
                                     if mism is not None else None),
                         "tol": abs(mism or 0)})
        self._payload_meta = meta
        return api_url, "text", payloads

    # Server-side ITL SLO: seconds to wait for the /metrics scrape. The knee gates
    # on the SERVER's own inter_token_latency_seconds (true decode smoothness) when
    # the endpoint exposes /metrics, because the client-measured P99 ITL is
    # confounded by the multiproc-async client's SSE read scheduling under bursty
    # load (verified: at 3 req/s MM the client measured P99 ITL 195-1193ms while the
    # server's own P99 ITL was ~25ms). Falls back to client ITL when /metrics is
    # absent (non-vLLM endpoints).
    METRICS_TIMEOUT_S = 5

    def _metrics_url(self) -> Optional[str]:
        """The served endpoint's Prometheus /metrics URL (server root + /metrics),
        or None if we don't have an endpoint. /metrics lives at the SERVER ROOT,
        not under /v1, so strip a trailing /v1."""
        if getattr(self.config, "remote_endpoint", None):
            base = self.config.remote_endpoint
        elif self._serving_runner is not None and getattr(
                self._serving_runner, "server_port", None):
            base = f"http://127.0.0.1:{self._serving_runner.server_port}"
        else:
            return None
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        return base + "/metrics"

    def _metrics_snapshot(self) -> Optional[dict]:
        """One parsed /metrics scrape (perf_method.parse_metrics), or None."""
        url = self._metrics_url()
        if not url:
            return None
        try:
            raw = urllib.request.urlopen(url, timeout=self.METRICS_TIMEOUT_S).read().decode()
        except Exception:
            return None
        return parse_metrics(raw)

    def _payload_checks(self, records: list, meta: list, before: Optional[dict],
                        after: Optional[dict]) -> dict:
        """Did the server see the prompts we planned, and did any cache serve them?

        prompt_len_mismatch: requests whose usage.prompt_tokens differs from planned +
        the calibrated overhead by more than their own re-tokenization tolerance.
        Cache: summed over the point, prefix-cache hits stay within n_sent x overhead
        when content is unique (each request shares only the template plus a few
        leading nonce-code tokens, and hits come in whole blocks; see the deviation
        note in the perf plan), so more hits mean content was reused. It is a backstop:
        reuse smaller than the slack (overhead minus the cached template blocks per
        request) is not visible. Any multimodal-cache hit means an image was reused.
        The encoder cache has no counter, so its reuse can't be observed."""
        out: dict = {"cache_telemetry_note": "encoder-cache reuse is not observable"}
        ov = self._prompt_overhead
        checked = bad = 0
        for r in records:
            i, pt = r.get("idx"), r.get("prompt_tokens")
            if i is None or pt is None or ov is None or i >= len(meta):
                continue
            m = meta[i]
            if m.get("planned") is None:
                continue
            checked += 1
            if abs(pt - (m["planned"] + ov)) > m.get("tol", 0):
                bad += 1
        out["prompt_len_checked"] = checked
        out["prompt_len_mismatch"] = bad if checked else None     # None = not checked
        hits = counter_delta(before, after, "prefix_cache_hits")
        queries = counter_delta(before, after, "prefix_cache_queries")
        mm_hits = counter_delta(before, after, "mm_cache_hits")
        out["prefix_cache_hits"] = hits
        out["prefix_cache_queries"] = queries
        out["mm_cache_hits"] = mm_hits
        n_sent = len([r for r in records if r.get("send") is not None])
        prefix_checkable = hits is not None and ov is not None and not self._real_dataset()
        mm_checkable = self._multimodal and mm_hits is not None
        suspect = []
        if prefix_checkable and hits > n_sent * max(0, ov):
            suspect.append("prefix")
        if mm_checkable and mm_hits > 0:
            suspect.append("multimodal")
        # None = not observable (no /metrics, no calibrated overhead, or a real dataset
        # whose prompts may legitimately share prefixes): never reported as clean.
        out["cache_reuse_suspect"] = suspect if (prefix_checkable or mm_checkable) else None
        if not (prefix_checkable or mm_checkable):
            out["cache_telemetry_note"] += "; cache reuse not observable for this point"
        elif not prefix_checkable:
            out["cache_telemetry_note"] += "; prefix reuse not checked (multimodal cache only)"
        if suspect or (checked and bad):
            logger.warning(f"Payload check: {bad}/{checked} prompt-length mismatches; cache "
                           f"reuse suspected in {suspect or 'none'} (prefix hits {hits}, "
                           f"mm hits {mm_hits}).")
        return out

    def _payload_record(self) -> dict:
        """What the result records about its request content: the seed and salt, where
        the prompts came from, the calibrated overhead, and any prompt reuse."""
        ds, _i, _o = self._workload_shape()
        if self._multimodal:
            source = "synthetic_multimodal"
        elif self._real_dataset():
            source = f"dataset:{ds}"
        elif self.config.remote_endpoint and self._tokenizer is not None:
            source = "synthetic_chat"
        else:
            source = "random_tokens"
        return {**self._payloads.record(), "workload_prompt_source": source,
                "prompt_overhead_tokens": self._prompt_overhead,
                "prompts_reused": self._prompt_reuse,
                "cache_salt_sent": bool(self._request_extras("sweep", 0, False).get("cache_salt"))}

    @staticmethod
    def _payload_rollup(rep_results: list) -> Optional[dict]:
        """Run-level summary of the per-point payload checks; None when no point could be
        checked (no /metrics or no calibrated overhead)."""
        pts = [pt for rr in rep_results for pt in rr.get("tested_rates", [])]
        cache_checked = [pt for pt in pts if pt.get("cache_reuse_suspect") is not None]
        len_checked = [pt for pt in pts if pt.get("prompt_len_mismatch") is not None]
        if not cache_checked and not len_checked:
            return None
        return {
            # Points whose cache reuse could be judged (None-suspect points are "not
            # observable", never counted as clean).
            "points_cache_checked": len(cache_checked),
            "points_suspect": sorted({pt["request_rate_qps"] for pt in cache_checked
                                      if pt.get("cache_reuse_suspect")}),
            "points_len_checked": len(len_checked),
            "len_mismatch_total": sum(int(pt.get("prompt_len_mismatch") or 0)
                                      for pt in len_checked),
        }

    def _calibrate_prompt_overhead(self, model, format) -> None:
        """Send one calibration request (max_tokens 1, not streamed) and record how
        many prompt tokens the server adds beyond the planned text (BOS, chat template,
        system prompt). None when the server reports no usage."""
        try:
            api_url, _cf, payloads = self._build_payloads(model, format, 1, phase="calibrate")
            meta = self._payload_meta[0] if self._payload_meta else {}
            if not payloads or meta.get("planned") is None:
                return
            p = dict(payloads[0])
            p.update({"stream": False, "max_tokens": 1})
            p.pop("stream_options", None)
            req = urllib.request.Request(
                api_url, data=json.dumps(p).encode(),
                headers={"Content-Type": "application/json", **self._request_headers()})
            with urllib.request.urlopen(req, timeout=self._request_timeout()) as resp:
                pt = (json.loads(resp.read()).get("usage") or {}).get("prompt_tokens")
            if pt is not None:
                self._prompt_overhead = int(pt) - int(meta["planned"])
                logger.info(f"Prompt overhead calibrated: {self._prompt_overhead} tokens "
                            f"beyond the planned text ({pt} prompt tokens).")
        except Exception as e:
            logger.warning(f"Prompt-overhead calibration failed ({e}); prompt lengths "
                           f"and prefix-cache reuse will not be checked.")
        finally:
            self._release_media()

    def _server_itl_snapshot(self) -> Optional[tuple]:
        """Cumulative vllm:inter_token_latency_seconds histogram, summed across DP
        engines: ({le: count}, total_count). None if /metrics is unreachable or the
        metric is absent (so the caller falls back to client-measured ITL). This
        build names the per-token metric inter_token_latency_seconds (NOT
        time_per_output_token_seconds)."""
        url = self._metrics_url()
        if not url:
            return None
        try:
            raw = urllib.request.urlopen(url, timeout=self.METRICS_TIMEOUT_S).read().decode()
        except Exception:
            return None
        buckets: dict = {}
        count = 0.0
        for ln in raw.splitlines():
            if ln.startswith("vllm:inter_token_latency_seconds_bucket"):
                m = re.search(r'le="([^"]+)".*?\}\s+([\d.eE+]+)', ln)
                if m:
                    le = float("inf") if m.group(1) == "+Inf" else float(m.group(1))
                    buckets[le] = buckets.get(le, 0.0) + float(m.group(2))
            elif ln.startswith("vllm:inter_token_latency_seconds_count"):
                count += float(ln.split()[-1])
        if not buckets or count <= 0:
            return None
        return buckets, count

    @staticmethod
    def _hist_delta_p99_ms(before: Optional[tuple], after: Optional[tuple],
                           pct: float = 99.0) -> Optional[float]:
        """Upper edge (ms) of the histogram bucket holding the P99 inter-token latency
        of the tokens emitted BETWEEN two cumulative snapshots (perf_method). None if
        either snapshot is missing or no tokens were emitted in the window."""
        d = hist_delta(before, after)
        if d is None:
            return None
        _lo, hi = hist_quantile_bucket(d[0], d[1], pct)
        return float("inf") if hi == float("inf") else hi * 1000.0

    def _warn_itl_slo_vs_edges(self, snap: Optional[tuple]) -> None:
        """Warn once per run when the ITL SLO can't be decided from the server's
        histogram: below its smallest finite edge (the client value then decides every
        point), or between two edges (points whose P99 bucket straddles it fall back)."""
        if not snap or getattr(self, "_itl_edge_warned", False):
            return
        self._itl_edge_warned = True
        buckets = snap[0]
        low = smallest_finite_edge_ms(buckets)
        slo = float(self.tpot_threshold_ms)
        if low is not None and slo < low:
            logger.warning(f"ITL SLO {slo:g} ms is below the server's smallest ITL bucket edge "
                           f"({low:g} ms): the client-measured P99 decides every point.")
        elif not any(abs(le * 1000.0 - slo) < 1e-6 for le in buckets):
            logger.warning(f"ITL SLO {slo:g} ms is not a server ITL bucket edge: a point whose "
                           f"P99 bucket straddles it is decided by the client-measured P99.")

    def _measure_rate_point(
        self,
        model: ModelConfig,
        format: ModelFormat,
        qps: float,
        num_prompts: int,
        min_n: Optional[int] = None,
    ) -> dict:
        """Measure one open-loop arrival rate: one merged Poisson schedule sent by up to K
        client processes at absolute times, analysed on its planned-send cohort
        (_measure_rate_point_inner, _cohort_metrics). The point's images are removed
        afterwards whatever happens."""
        try:
            return self._measure_rate_point_inner(model, format, qps, num_prompts, min_n)
        finally:
            self._release_media()            # never leave a point's images on disk

    SAMPLER_PERIOD_S = 1.0          # /metrics sampler cadence during a point
    MIN_WINDOW_S = 10.0             # shortest measurement interval worth judging
    START_MARGIN_S = 2.0            # minimum lead from scheduling to T0

    def _run_attempt(self, model, format, qps: float, n: int, point: int) -> dict:
        """One attempt at a point: build n unique payloads, draw the merged Poisson
        schedule, start every worker against the same absolute T0, sample /metrics at
        1 Hz while it runs, and return everything the analysis needs."""
        api_url, content_field, payloads = self._build_payloads(
            model, format, n, phase="sweep", point=point)
        meta = list(self._payload_meta)
        offsets = merged_schedule(qps, len(payloads), stable_u32(
            self._payloads.seed, self._payloads.call, self._rep_idx, point))
        # Workers scale with request count (>= MIN_PROMPTS_PER_WORKER each); the
        # schedule is the same for any k (arrival i goes to worker i mod k).
        k = max(1, min(self._num_client_procs,
                       len(payloads) // self.MIN_PROMPTS_PER_WORKER or 1))
        parts = assign_workers(len(payloads), k)
        self._warm_pool(k)
        margin = max(self.START_MARGIN_S, 1.5 * getattr(self, "_delivery_s", 0.0))
        t0 = time.time() + margin
        tasks = [{
            "payloads": [payloads[i] for i in part], "idx": part,
            "offsets": [offsets[i] for i in part], "t0": t0,
            "api_url": api_url, "content_field": content_field,
            "request_timeout": self._request_timeout(), "max_point_s": self._max_point_s(),
            "itl_cap": self.ITL_SAMPLE_CAP, "headers": self._request_headers(),
        } for part in parts if part]
        snaps, stop = [], threading.Event()

        def _sample():
            while not stop.is_set():
                t_req = time.time()
                snap = self._metrics_snapshot()
                snaps.append((t_req, time.time(), snap))
                stop.wait(self.SAMPLER_PERIOD_S)

        sampler = threading.Thread(target=_sample, daemon=True)
        sampler.start()
        submitted = time.time()
        records, ready, n_procs = [], [], len(tasks)
        try:
            try:
                for out in self._get_client_pool().map(_stress_client_worker, tasks):
                    records.extend(out.get("records", []))
                    if out.get("ready_at"):
                        ready.append(out["ready_at"])
            except Exception as e:
                # Pool unusable (spawn blocked, a worker died). Payloads may already
                # have reached the server, so the caller re-runs the point once under a
                # new payload key; otherwise the point has no data (inconclusive).
                logger.warning(f"Client process pool failed ({e}); re-running the point once "
                               f"under a new payload key.")
                self._shutdown_client_pool()
                return {"retry": "client pool failed", "n_procs": 1}
        finally:
            stop.set()
            sampler.join(5)
            t_req = time.time()
            snaps.append((t_req, time.time(), self._metrics_snapshot()))   # after the drain
            self._release_media()
        if ready:
            self._delivery_s = max(ready) - submitted
        return {"records": records, "meta": meta, "offsets": offsets, "t0": t0,
                "snaps": snaps, "n_procs": n_procs,
                "late_start": bool(ready) and max(ready) > t0, "planned_n": len(payloads)}

    @staticmethod
    def _nearest_snapshot(snaps: list, t: float, with_times: bool = False):
        """The sampled /metrics scrape closest to instant t (its midpoint), or None; with
        ``with_times`` the (t_request, t_response, snapshot) triple."""
        best = None
        for t_req, t_resp, snap in snaps:
            if snap is None:
                continue
            d = abs((t_req + t_resp) / 2.0 - t)
            if best is None or d < best[0]:
                best = (d, (t_req, t_resp, snap))
        if not best:
            return None
        return best[1] if with_times else best[1][2]

    def _window_short(self, att: dict) -> Optional[float]:
        """The point's median e2e if its schedule was too short for it (window - L50 <
        MIN_WINDOW_S, or 3 x L50 > window), else None."""
        ok = [r for r in att["records"] if r.get("ok") and r.get("done") is not None]
        l50 = float(np.median([r["done"] - r["send"] for r in ok])) if ok else 0.0
        window = att["offsets"][-1]
        short = (window - l50 < max(self.MIN_WINDOW_S, 2 * self.SAMPLER_PERIOD_S)
                 or 3.0 * l50 > window)
        return l50 if short else None

    def _overloaded(self, metrics: dict, l50: Optional[float]) -> bool:
        """Decisively overloaded: a growing queue, too few requests finished within the
        SLO-implied time, or a median latency beyond that time."""
        cf = metrics.get("completion_fraction")
        return bool(metrics.get("saturation") == SATURATED
                    or (cf is not None and cf < self.KEEP_UP_FRAC)
                    or (l50 or 0.0) > (metrics.get("completion_bound_s") or float("inf")))

    def _measure_rate_point_inner(self, model, format, qps: float, num_prompts: int,
                                  min_n: Optional[int] = None) -> dict:
        """Run a point and analyse it on its planned-send cohort. Re-runs (each under a
        new payload key, never with fewer requests): after a late worker start or a
        client-pool failure; once when its schedule was too short for the latency it had
        and it was not already decisively overloaded; once when other traffic
        contaminated it (a second contaminated observation ends the pillar)."""
        n = num_prompts
        reruns: list = []
        metrics = used = None
        contaminated_seen = 0
        window_rerun_done = False
        attempts = 0
        while attempts < 4:
            attempts += 1
            att = self._run_attempt(model, format, qps, n, self._point_seq)
            self._point_seq += 1                     # never resend a payload
            if att.get("retry") or att.get("late_start"):
                reason = att.get("retry") or "late start"
                if reason == "late start":
                    self._delivery_s = 2.0 * max(getattr(self, "_delivery_s", 0.0),
                                                 self.START_MARGIN_S)
                if reruns.count(reason) >= 1:      # one retry per cause
                    break
                reruns.append(reason)
                continue
            m = self._cohort_metrics(att["records"], qps, att["t0"], att["offsets"],
                                     att["snaps"], att["meta"], flags=[],
                                     planned_n=att["planned_n"])
            self._log_point(qps, m, attempt=attempts)
            if m.get("contaminated"):
                contaminated_seen += 1
                if contaminated_seen >= 2:
                    raise StressContaminated(
                        f"other traffic on the server during the point at {qps:g} req/s on "
                        f"two measurements (foreign requests: {m.get('foreign_requests')})")
                if attempts >= 4:
                    reruns.append("attempt budget exhausted")
                    break
                reruns.append("contaminated")
                logger.warning(f"Point at {qps:g} req/s contaminated by other traffic "
                               f"({m.get('foreign_requests')}); draining and re-measuring once.")
                self._ensure_drained()
                continue
            metrics, used = m, att
            l50 = self._window_short(att)
            if l50 is None or self._overloaded(m, l50) or window_rerun_done:
                break
            if attempts >= 4:
                break
            window_rerun_done = True
            reruns.append("window too short")
            # Re-size from the latency the point had, never fewer requests than before.
            n = min(2 * n, max(n, int(math.ceil(qps * (3.0 * l50 + 10.0))),
                               self._probe_prompts(qps, min_n, l50)))
        if metrics is None:
            return self._cohort_metrics([], qps, None, None, [], None,
                                        flags=reruns + ["no data"])
        flags = [f for f in (metrics.get("flags") or []) if f != "window too short"]
        l50 = self._window_short(used)
        if l50 is not None and not self._overloaded(metrics, l50):
            flags.append("window too short")      # still too short: inconclusive
        metrics["flags"] = flags
        metrics["reruns"] = reruns
        metrics["num_client_procs"] = used["n_procs"]
        return metrics

    def _log_point(self, qps: float, m: dict, attempt: int) -> None:
        """Run-level log of every measurement (all repeats, certification probes,
        contaminated attempts), kept in every result, contaminated or not."""
        keep = ("saturation", "completion_fraction", "p99_ttft_ms", "p99_tpot_ms",
                "cohort_n", "contaminated", "foreign_requests", "engine_above_client",
                "achieved_qps", "flags", "planned_n", "sent_n")
        # point_seq (with the payload call and rep) is the merged_schedule seed input.
        self._point_log.append({"qps": qps, "rep": self._rep_idx, "attempt": attempt,
                                "point_seq": self._point_seq - 1,
                                "payload_call": self._payloads.call,
                                **{k: m.get(k) for k in keep}})

    def _prompt_cap(self) -> Optional[int]:
        """An explicit --stress-max-prompts / GBENCH_STRESS_MAX_PROMPTS (a recorded cap);
        None by default (no cap: points get the requests the method needs)."""
        return self.max_prompts if getattr(self, "_max_prompts_explicit", False) else None

    def _cohort_metrics(self, records: list, qps: float, t0: Optional[float],
                        offsets: Optional[list], snaps: list, meta: Optional[list],
                        flags: Optional[list] = None, planned_n: Optional[int] = None) -> dict:
        """Everything _test_rate gates on, computed on the point's cohort: the requests
        whose PLANNED send falls in [T0 + L50, T_last_arrival] (L50 = the point's median
        e2e). Saturation is the queue-growth test on cohort TTFT; the completion fraction
        uses the end-to-end time the SLOs imply; server-side values use the /metrics
        scrapes nearest the interval's ends."""
        flags = list(flags or [])
        out = {"arrival_qps": qps, "flags": flags, "planned_n": planned_n,
               "sent_n": sum(1 for r in records if r.get("send") is not None)}
        if not records or t0 is None:
            out.update({"achieved_qps": 0.0, "slo_attainment": 0.0, "p99_ttft_ms": float("inf"),
                        "p99_tpot_ms": float("inf"), "p99_itl_ms": float("inf"),
                        "p99_itl_per_token_ms": None, "completed": 0,
                        "failed": len(records), "steady_n": 0, "saturation": INDETERMINATE,
                        "completion_fraction": None, "client_bound": False,
                        "p99_itl_server_ms": None, "itl_gate_outcome": None})
            return out
        ok = [r for r in records if r.get("ok") and r.get("done") is not None]
        l50 = float(np.median([r["done"] - r["send"] for r in ok])) if ok else 0.0
        t_last = t0 + float(offsets[-1])
        t_lo = t0 + l50
        coh = cohort(records, t_lo, t_last)
        if not coh:
            # The median latency outlasted the whole schedule (a deep backlog): judge
            # every request, recorded as such, rather than nothing.
            flags.append("cohort = all requests (median latency > schedule)")
            t_lo = t0
            coh = cohort(records, t0, t_last)
        span = max(1e-6, t_last - t_lo)
        _ds, _in, out_len = self._workload_shape()
        max_tokens = max(int(out_len), 16)
        bound = completion_bound_s(self.ttft_threshold_ms, self.tpot_threshold_ms, max_tokens)
        t_end = max([r["done"] for r in ok] + [t_last])

        # SLO statistics on the cohort (a request that never finished counts as +inf).
        coh_ok = [r for r in coh if r.get("ok") and r.get("done") is not None]
        stats = self._steady_state_metrics(coh_ok, qps, preselected=True)
        ttfts = [r["ttft"] if (r.get("ok") and r.get("ttft") is not None) else float("inf")
                 for r in coh]
        stats["p99_ttft_ms"] = percentile_inf(ttfts, 99)
        stats["slo_attainment"] = (stats.get("_slo_ok_n", 0) / len(coh)) if coh else 0.0
        stats.pop("_slo_ok_n", None)
        out.update(stats)

        # Saturation: queue growth of TTFT over the cohort's requests that got a first
        # token. Requests that never did are counted by the completion fraction and the
        # +inf P99; giving them their age at the end of the point instead would trend
        # DOWN with send time (later sends are younger) and mask a growing queue.
        with_ttft = [r for r in coh if r.get("send") is not None and r.get("ttft") is not None]
        qg = queue_growth([r["planned"] for r in with_ttft], [r["ttft"] for r in with_ttft])
        cf = completion_fraction(coh, t_last, bound)
        n_done_in = sum(1 for r in records if r.get("ok") and r.get("done") is not None
                        and t_lo <= r["done"] <= t_last)
        out.update({
            "saturation": qg["outcome"], "queue_growth": qg,
            "completion_fraction": cf, "completion_bound_s": bound,
            "cohort_n": len(coh), "steady_n": len(coh), "cohort_window_s": span,
            "cohort_offered_qps": len(coh) / span,
            # completions inside the measurement window per second (any request's)
            "achieved_qps": n_done_in / span,
            "offered": offered_rate_ci(len(offsets), float(offsets[-1])),
            "completed": len(ok), "failed": len(records) - len(ok),
            "short_replies": sum(1 for r in ok if r.get("finish_reason") == "stop"
                                 and (r.get("completion_tokens") or max_tokens) < max_tokens),
            # served replies whose tokens carried no visible text (TTFT = e2e, an upper bound)
            "no_text_replies": sum(1 for r in ok if r.get("ttft_unseen")),
        })
        out.update(self._client_bound_checks(records, planned_n or len(records)))

        # Server side: payload/cache checks over the whole point; the ITL gate, spec
        # decode and tokens per step over the measurement interval.
        whole_b = self._nearest_snapshot(snaps, t0)
        whole_a = self._nearest_snapshot(snaps, t_end + 10 * self.SAMPLER_PERIOD_S)
        out.update(self._payload_checks(records, meta or [], whole_b, whole_a))
        snap_lo = self._nearest_snapshot(snaps, t_lo)
        snap_hi = self._nearest_snapshot(snaps, t_last)
        itl_lo, itl_hi = (snap_lo or {}).get("itl"), (snap_hi or {}).get("itl")
        self._warn_itl_slo_vs_edges(itl_hi)
        d = hist_delta(itl_lo, itl_hi)
        _sd = counter_delta(snap_lo, snap_hi, "spec_draft_tokens")
        out["spec_decode_active"] = (_sd > 0) if _sd is not None else None
        gen = counter_delta(snap_lo, snap_hi, "generation_tokens")
        n_req = counter_delta(snap_lo, snap_hi, "request_success")
        dec = (gen - n_req) if (gen is not None and n_req is not None) else gen
        tps = (dec / d[1]) if (dec and d is not None and d[1]) else None
        out["tokens_per_engine_step"] = tps
        if out["spec_decode_active"] and d is not None and tps and tps > 1.0:
            limit = self.tpot_threshold_ms * tps
            out["itl_gate_outcome_per_step"] = hist_itl_gate(d[0], d[1], limit)["outcome"]
            out["itl_gate_limit_per_step_ms"] = limit
        if d is not None:
            out.update(hist_itl_gate(d[0], d[1], self.tpot_threshold_ms))
            out["itl_gate_outcome"] = out.pop("outcome")
        else:
            out["p99_itl_server_ms"] = None
            out["itl_gate_outcome"] = None
        out["itl_interval_s"] = [t_lo - t0, t_last - t0]
        # Foreign traffic over the interval (another client on the same server): finished
        # requests the server counted that we did not send, and engine work above our
        # own in-flight count.
        lo_t = self._nearest_snapshot(snaps, t_lo, with_times=True)
        hi_t = self._nearest_snapshot(snaps, t_last, with_times=True)
        fr = foreign_requests(lo_t, hi_t, records) if (lo_t and hi_t) else None
        excess = engine_above_client(snaps, records, t_lo, t_last)
        out["foreign_requests"] = fr
        out["engine_above_client"] = excess
        out["contaminated"] = contaminated(fr, excess)
        return out

    # Client-bound limits (starting values, pinned in the stress method version): at
    # least 99% of planned sends issued, send lag not trending up, p99 lag within
    # max(150 ms, 3% of the TTFT SLO).
    CLIENT_SENT_FRAC = 0.99
    CLIENT_LAG_SLOPE_MS_PER_S = 1.0
    CLIENT_LAG_P99_MS = 150.0
    CLIENT_LAG_P99_SLO_FRAC = 0.03

    def _client_bound_checks(self, records: list, planned_n: int) -> dict:
        """Did the CLIENT keep to the schedule? A late send makes the offered load lower
        than labelled, so a client-bound point is inconclusive, never a pass."""
        sent = [r for r in records if r.get("send") is not None and r.get("lag") is not None]
        frac = len(sent) / planned_n if planned_n else 0.0
        lags = np.array([max(0.0, r["lag"]) * 1000.0 for r in sent]) if sent else np.array([])
        p99 = float(np.percentile(lags, 99)) if len(lags) else 0.0
        slope = 0.0
        if len(sent) >= 20:
            x = np.array([r["planned"] for r in sent])
            order = np.argsort(x)
            xb = np.array([b.mean() for b in np.array_split(x[order], 10)])
            yb = np.array([b.mean() for b in np.array_split(lags[order], 10)])
            xc = xb - xb.mean()
            slope = float((xc * (yb - yb.mean())).sum() / max(1e-12, (xc ** 2).sum()))
        p99_limit = max(self.CLIENT_LAG_P99_MS, self.CLIENT_LAG_P99_SLO_FRAC * self.ttft_threshold_ms)
        reasons = []
        if frac < self.CLIENT_SENT_FRAC:
            reasons.append(f"only {frac:.1%} of planned sends issued")
        if slope > self.CLIENT_LAG_SLOPE_MS_PER_S:
            reasons.append(f"send lag growing {slope:.1f} ms/s")
        if p99 > p99_limit:
            reasons.append(f"p99 send lag {p99:.0f} ms > {p99_limit:.0f} ms")
        return {"client_bound": bool(reasons), "client_bound_reasons": reasons,
                "client_sent_frac": frac, "client_lag_p99_ms": p99,
                "client_lag_slope_ms_per_s": slope}

    def _slo_bound_s(self) -> float:
        _ds, _in, out_len = self._workload_shape()
        return completion_bound_s(self.ttft_threshold_ms, self.tpot_threshold_ms,
                                  max(int(out_len), 16))

    def _request_timeout(self) -> float:
        """Per-request client timeout: never shorter than the e2e time the SLOs allow."""
        return max(float(self.REQUEST_TIMEOUT_S), 1.25 * self._slo_bound_s())

    def _max_point_s(self) -> float:
        return max(float(self.MAX_POINT_S), 1.25 * self._slo_bound_s() + 60.0)

    def _warm_pool(self, k: int) -> None:
        """Start k client processes before T0 (the pool is lazy: a cold spawn + import
        takes about as long as the start margin, which would make the first point of a
        run start late)."""
        if getattr(self, "_warm_k", 0) >= k:
            return
        try:
            list(self._get_client_pool().map(_noop, range(k)))
            self._warm_k = k
        except Exception:                                                 # noqa: BLE001
            pass

    def _get_client_pool(self) -> ProcessPoolExecutor:
        """Lazily create ONE reusable client process pool for the whole sweep.

        Reused across every rate point and rep so the K workers stay warm - no
        per-point re-spawn cost and, more importantly, no repeated worker
        start-up stagger eating into the ~MEASURE_ARRIVAL_S window (which would
        depress actual_send_qps and spuriously trip client_bound).
        """
        if self._client_pool is None:
            self._client_pool = ProcessPoolExecutor(
                max_workers=self._num_client_procs, mp_context=self._mp_ctx)
        return self._client_pool

    def _shutdown_client_pool(self):
        """Tear down the reusable client pool (end of sweep, or after a fault)."""
        self._warm_k = 0
        if self._client_pool is not None:
            try:
                self._client_pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._client_pool = None

    def _steady_state_metrics(self, records: list, qps: float, preselected: bool = False) -> dict:
        """SLO statistics of a set of finished requests (the point's cohort): P99 TTFT,
        per-token TPOT, inter-chunk and per-token ITL, mean e2e and how many met both
        SLOs. The set is chosen by the caller (_cohort_metrics: by planned send time);
        nothing here trims by completion order."""
        recs = [r for r in records if r.get("send") is not None]
        steady = [r for r in recs if r.get("ok") and r.get("done") is not None]
        if not steady:
            return {"achieved_qps": 0.0, "arrival_qps": qps, "slo_attainment": 0.0,
                    "p99_ttft_ms": float("inf"), "p99_tpot_ms": float("inf"),
                    "p99_itl_ms": float("inf"), "p99_itl_per_token_ms": None,
                    "completed": 0, "failed": len(recs), "steady_n": 0, "_slo_ok_n": 0}
        achieved = 0.0
        # Per-token TPOT from the server's token count (falls back to the mean chunk
        # gap when the server sends no usage).
        for r in steady:
            r["tpot_tok"] = per_token_tpot_ms((r["done"] - r["send"]) * 1000.0, r.get("ttft"),
                                              r.get("completion_tokens"))
        n_fb = sum(1 for r in steady if r["tpot_tok"] is None)
        tpot_source = ("per_token_usage" if n_fb == 0 else
                       "inter_chunk_mean" if n_fb == len(steady) else
                       f"mixed ({n_fb} of {len(steady)} requests chunk mean)")

        def _tpot(r):
            return r["tpot_tok"] if r["tpot_tok"] is not None else r.get("tpot")
        ok_slo = [
            r for r in steady
            if r["ttft"] is not None and r["ttft"] <= self.ttft_threshold_ms
            and (_tpot(r) is None or _tpot(r) <= self.tpot_threshold_ms)
        ]
        slo_attain = len(ok_slo) / len(steady)
        slo_ok_n = len(ok_slo)

        def _p99(vals):
            # Proper (interpolated) P99 via numpy. The old index hack
            # vals[min(int(n*0.99), n-1)] returns the MAX for every n<=100, so at
            # low QPS (steady_n ~15-55) the "P99 TTFT/ITL" gate degenerated to a
            # MAX gate - one slow prefill failed the whole rate point and the knee
            # flipped run-to-run.
            vals = [v for v in vals if v is not None]
            return float(np.percentile(vals, 99)) if vals else 0.0

        # P99 of the gaps between streamed chunks across the steady set (workers
        # subsampled each request to ITL_SAMPLE_CAP): the stutter a user sees. The ITL
        # SLO gates on the server histogram first, then on the per-token value below.
        steady_itls = []
        for r in steady:
            gaps = r.get("itls")
            if gaps:
                steady_itls.extend(gaps)
        p99_itl = _p99(steady_itls) if steady_itls else 0.0
        p99_itl_tok = self._p99_per_token_itl(steady, _p99)
        # Mean end-to-end latency over the steady set (done-send), for Little's-Law
        # N_users = arrival_rate * mean_e2e at the knee (concurrent-users capacity).
        mean_e2e_ms = float(np.mean([(r["done"] - r["send"]) * 1000.0
                                     for r in steady]))

        return {
            "achieved_qps": achieved,
            "arrival_qps": qps,
            "slo_attainment": slo_attain,
            "_slo_ok_n": slo_ok_n,
            "p99_ttft_ms": _p99([r["ttft"] for r in steady]),
            "p99_tpot_ms": _p99([_tpot(r) for r in steady]),
            "tpot_source": tpot_source,
            "p99_itl_ms": p99_itl,                    # inter-chunk (what a user sees)
            "p99_itl_per_token_ms": p99_itl_tok,      # None without a tokenizer
            "mean_e2e_ms": mean_e2e_ms,
            "p90_e2e_s": float(np.percentile([r["done"] - r["send"] for r in steady], 90)),
            "completed": len(steady),
            "failed": len(recs) - len(steady),
            "steady_n": len(steady),
        }

    PER_TOKEN_PIECE_CAP = 200_000   # chunk texts tokenized per point (uniform subsample)

    def _per_token_recount_ok(self) -> bool:
        """One-time self-check that tokenizing a chunk's text alone counts its tokens: a
        real passage decoded token by token must re-tokenize to the same count (±1%).
        SentencePiece
        tokenizers that add a dummy prefix (e.g. Llama 2) count many chunks as two,
        which would understate per-token gaps; then the per-token path is off and the
        reason recorded (the inter-chunk gap is used, labelled as such)."""
        if getattr(self, "_recount_for", None) != id(self._tokenizer):
            self._recount_ok = None               # a new tokenizer (next model): re-check
            self._recount_for = id(self._tokenizer)
        if getattr(self, "_recount_ok", None) is None:
            tok = self._tokenizer
            ok, why = True, None
            try:
                # Realistic text, decoded token by token as a stream would deliver it.
                text = (self._serving_runner.PASSAGE if self._serving_runner is not None
                        and getattr(self._serving_runner, "PASSAGE", None) else
                        "The quick brown fox jumps over the lazy dog. ") * 4
                ids = tok(text, add_special_tokens=False).input_ids
                pieces = [tok.decode([int(i)]) for i in ids]
                recount = sum(len(x) for x in tok(pieces, add_special_tokens=False).input_ids)
                if abs(recount - len(ids)) > max(1, 0.01 * len(ids)):
                    ok, why = False, (f"re-tokenizing streamed tokens one by one counts {recount} "
                                      f"tokens for {len(ids)} with this tokenizer")
            except Exception as e:                                       # noqa: BLE001
                ok, why = False, f"self-check failed: {e}"
            self._recount_ok, self._recount_why = ok, why
            if not ok:
                logger.warning(f"Per-token client ITL disabled ({why}); the client fallback "
                               f"is the inter-chunk gap.")
        return self._recount_ok

    def _p99_per_token_itl(self, steady: list, p99) -> Optional[float]:
        """P99 of per-token gaps: each inter-chunk gap divided by the tokens its chunk
        carried, counted by tokenizing the chunk texts here in the parent after the
        point (never in the send-path workers). None without a tokenizer."""
        if self._tokenizer is None or not self._per_token_recount_ok():
            return None
        gaps, pieces = [], []
        for r in steady:
            g, pc = r.get("itls") or [], r.get("itl_pieces") or []
            if len(g) == len(pc):
                gaps.extend(g)
                pieces.extend(pc)
        if not gaps:
            return None
        if len(gaps) > self.PER_TOKEN_PIECE_CAP:
            step = len(gaps) / self.PER_TOKEN_PIECE_CAP
            keep = [int(k * step) for k in range(self.PER_TOKEN_PIECE_CAP)]
            gaps, pieces = [gaps[k] for k in keep], [pieces[k] for k in keep]
        try:
            counts = [len(x) for x in self._tokenizer(pieces, add_special_tokens=False).input_ids]
        except Exception:
            return None
        return p99(per_token_gaps(gaps, counts))

    def _warmup(self, model, format) -> None:
        """A few SEQUENTIAL (concurrency-1) discarded requests to warm the engine.

        Warms text + MM CUDA graphs before the sweep. Sent strictly one-at-a-time,
        NOT the old open-loop 20-reqs-at-1-req/s pass: on a slow single-stream server
        that over-capacity warmup floods the backend's request queue, and the backlog
        then starves the preflight probe and the first sweep points (spurious
        no-stream / TOO-FEW). Sequential warming builds no queue on any hardware.
        """
        try:
            api_url, content_field, payloads = self._build_payloads(
                model, format, self.WARMUP_PROMPTS_GENTLE, phase="warmup")
        except Exception as e:
            logger.warning(f"Warmup skipped (payload build failed: {e}).")
            return
        if not payloads:
            return
        logger.info(f"Warmup ({len(payloads)} sequential reqs, discarded)...")
        try:
            self._warmup_send(payloads, api_url, content_field)
        finally:
            self._release_media()

    def _warmup_send(self, payloads, api_url, content_field) -> None:
        for p in payloads:
            try:
                _stress_client_worker({
                    "payloads": [p], "qps": 0.0, "seed": 83,
                    "api_url": api_url, "content_field": content_field,
                    "request_timeout": self._request_timeout(),
                    "max_point_s": self._max_point_s(), "itl_cap": self.ITL_SAMPLE_CAP,
                    "headers": self._request_headers(),
                })
            except Exception as e:
                logger.warning(f"Warmup request failed (continuing): {e}")
                return

    def _test_rate(self, qps: float, model, format, min_n: Optional[int] = None,
                   cache: Optional[dict] = None) -> tuple:
        """Measure one open-loop arrival rate at steady state.

        Returns (passed, p99_ttft_ms, p99_tpot_ms, achieved_qps). A rate is
        SUSTAINABLE iff, on the point's cohort (planned sends in [T0 + L50, T_last]):
        the queue is not growing (perf_load.queue_growth is not SATURATED), at least
        KEEP_UP_FRAC of the cohort finished within the end-to-end time the SLOs imply,
        P99 TTFT and P99 ITL meet their SLOs, the cohort holds >= MIN_STEADY requests,
        failures stay under MAX_FAILURE_FRAC, and the client kept to its schedule.
        """
        if not qps or qps <= 0:
            raise ValueError(f"stress point rate must be > 0 (got {qps!r})")
        store = self._tested_rates if cache is None else cache
        if qps in store:
            c = store[qps]
            return c["passed"], c["p99_ttft_ms"], c["p99_tpot_ms"], c["achieved_qps"]

        # Never fewer requests than the old floor max(min_prompts, qps x 60 s); more when
        # the loaded latency needs a longer window (perf_load.request_count). The
        # latency estimate is the previous passing point's p90 e2e, else the preflight's.
        l_est = getattr(self, "_last_pass_p90_s", None) or (
            (self._probe_e2e_ms or 0.0) / 1000.0 or None)
        num_prompts = self._probe_prompts(qps, min_n, l_est)

        self._point_seq += 1
        r = self._measure_rate_point(model, format, qps, num_prompts, min_n=min_n)

        achieved = float(r.get("achieved_qps", 0.0) or 0.0)
        attain = float(r.get("slo_attainment", 0.0) or 0.0)
        p99_ttft = float(r.get("p99_ttft_ms", 0.0) or 0.0)
        p99_tpot = float(r.get("p99_tpot_ms", 0.0) or 0.0)
        p99_itl = float(r.get("p99_itl_ms", 0.0) or 0.0)
        _tok = r.get("p99_itl_per_token_ms")
        p99_itl_tok = float(_tok) if _tok is not None else None
        _srv = r.get("p99_itl_server_ms")   # None when /metrics is absent
        p99_itl_server = float(_srv) if _srv is not None else None
        mean_e2e = float(r.get("mean_e2e_ms", 0.0) or 0.0)
        steady_n = int(r.get("steady_n", 0) or 0)
        completed = int(r.get("completed", 0) or 0)
        failed = int(r.get("failed", 0) or 0)
        client_bound = bool(r.get("client_bound", False))
        saturation = r.get("saturation")
        cf = r.get("completion_fraction")
        flags = list(r.get("flags") or [])
        # Saturation: the queue-growth test (SATURATED fails the point) and the share of
        # the cohort finished within the SLO-implied e2e time. NOT_DETECTED is not proof
        # of keeping up: the SLO gates below then decide.
        keep_up = saturation != SATURATED and (cf is None or cf >= self.KEEP_UP_FRAC)
        inconclusive = "window too short" in flags or "no data" in flags
        # Decode-cadence SLO on the P99 ITL tail. The server's own histogram
        # (/metrics) decides when it can: PASS/FAIL are provable from the bucket
        # counts (perf_method.hist_itl_gate). INDETERMINATE (the limit sits inside a
        # bucket straddling P99) or no /metrics falls back to the client value, then
        # to the per-request TPOT; the source that decided is recorded.
        itl_outcome = r.get("itl_gate_outcome")
        spec = bool(r.get("spec_decode_active"))
        if spec and p99_itl_tok and p99_itl_tok > 0:
            # Speculative/MTP decoding: the server records one gap per engine step (several
            # tokens), not per token, so the per-token SLO is judged on the client's
            # per-token gap at every point of the run.
            itl_metric, itl_source = p99_itl_tok, "client_per_token (spec decode)"
            itl_ok = itl_metric <= self.tpot_threshold_ms
        elif spec and r.get("itl_gate_outcome_per_step") in ("PASS", "FAIL"):
            # No tokenizer: the server gate with the SLO converted to a per-step limit
            # (SLO x tokens per engine step over the point).
            itl_metric, itl_source = p99_itl_server, "server (per-step limit, spec decode)"
            itl_ok = r["itl_gate_outcome_per_step"] == "PASS"
        elif itl_outcome in ("PASS", "FAIL") and not spec:
            itl_metric = p99_itl_server
            itl_source = "server"
            itl_ok = itl_outcome == "PASS"
        else:
            # Client fallback: the per-token gap (chunk gaps divided by the tokens each
            # chunk carried) is comparable with a per-token SLO; the raw inter-chunk gap
            # overstates it when chunks carry several tokens, so it only decides when no
            # tokenizer was available.
            if p99_itl_tok and p99_itl_tok > 0:
                itl_metric, itl_source = p99_itl_tok, "client_per_token"
            elif p99_itl > 0:
                itl_metric, itl_source = p99_itl, "client_inter_chunk"
            else:
                itl_metric, itl_source = p99_tpot, "tpot"
            if spec:
                _ps = r.get("itl_gate_outcome_per_step")
                itl_source += (" (spec decode: per-step gate indeterminate)" if _ps == "INDETERMINATE"
                               else " (spec decode: per-step gate unavailable)")
            elif itl_outcome == "INDETERMINATE":
                itl_source += " (server indeterminate)"
            itl_ok = itl_metric <= self.tpot_threshold_ms
        itl_limit = (r.get("itl_gate_limit_per_step_ms") if itl_source.startswith("server (per-step")
                     else self.tpot_threshold_ms)
        slo_ok = (p99_ttft <= self.ttft_threshold_ms) and itl_ok
        # Cross-check: the client's per-token P99 should fall within one histogram bucket
        # of the server's P99 bucket. Skipped under speculative decoding, where the server
        # records one gap per engine step (several tokens), not per token.
        itl_disagree = None
        edges = sorted(float(k) if k != "inf" else float("inf") for k in
                       ((r.get("itl_hist_delta_ms") or {}).keys())) if r.get("itl_hist_delta_ms") else []
        if (itl_outcome in ("PASS", "FAIL") and p99_itl_tok and edges
                and not r.get("spec_decode_active")):
            edges_s = [e / 1000.0 for e in edges]
            srv_k = bucket_index(min(p99_itl_server, edges[-1]) if p99_itl_server else 0.0, edges_s)
            itl_disagree = abs(bucket_index(p99_itl_tok, edges_s) - srv_k) > 1
        enough = steady_n >= self._min_steady
        # Failure gate: a point where a non-trivial fraction of OFFERED load
        # errored/dropped is NOT sustainable. Failures never lower achieved or
        # slo_attainment (both computed over successful requests only), so without
        # this gate a partially-failing point passes and inflates the knee.
        offered = completed + failed
        low_failure = (failed / offered) <= self.MAX_FAILURE_FRAC if offered > 0 else True
        # A CLIENT-BOUND point is inconclusive - the load generator, not the server,
        # was the bottleneck - so it must NOT count as a sustainable server-capacity
        # result, or the reported knee reflects the client's max send rate rather
        # than server capacity. Excluding it makes the search move DOWN (conservative:
        # under-, never over-reports), and it is surfaced prominently in the log.
        # Short replies are failures only when the server is known to honour ignore_eos
        # (preflight); otherwise they are recorded and the metrics use realized lengths.
        if getattr(self, "_ignore_eos_honored", None) is True:
            failed += int(r.get("short_replies") or 0)
            offered = completed + failed
            low_failure = (failed / offered) <= self.MAX_FAILURE_FRAC if offered > 0 else True
        # A certification probe passes only on a full cohort (a P99 needs its sample).
        cohort_short = bool(min_n) and int(r.get("cohort_n") or 0) < int(min_n)
        if cohort_short:
            flags.append(f"cohort {int(r.get('cohort_n') or 0)} < {int(min_n)} required")
            inconclusive = True
        passed = (keep_up and slo_ok and enough and low_failure and (not client_bound)
                  and not inconclusive and not r.get("contaminated"))
        # A decisive overload (queue growth, backlog, P99 TTFT over the SLO; unfinished
        # requests count as +inf) is a failure even when the certification cohort came up
        # short: overload at a lower realized load implies overload at the labelled rate.
        # Not when the client fell behind (a busy client inflates its own TTFT and
        # completion times, so the "overload" may be the load generator's) or there is
        # no data: those stay inconclusive.
        decisive_fail = (not client_bound and "no data" not in flags
                         and ((not keep_up) or p99_ttft > self.ttft_threshold_ms))
        verdict = ("pass" if passed else "fail" if decisive_fail
                   else "inconclusive" if (inconclusive or client_bound) else "fail")
        if passed:
            # Latency estimate for the next point's request count.
            self._last_pass_p90_s = r.get("p90_e2e_s") or getattr(self, "_last_pass_p90_s", None)

        store[qps] = {
            "request_rate_qps": qps,
            "certification_probe": bool(min_n),
            "arrival_qps": qps,
            "num_prompts": num_prompts,
            "num_client_procs": int(r.get("num_client_procs", 1)),  # ACTUAL procs used
            "achieved_qps": achieved,                 # cohort finished within the bound / window
            "planned_n": r.get("planned_n"),
            "sent_n": r.get("sent_n"),
            "offered": r.get("offered"),              # realized offered rate + Poisson CI
            "cohort_n": r.get("cohort_n"),
            "cohort_offered_qps": r.get("cohort_offered_qps"),
            "saturation": saturation,                 # SATURATED / NOT_DETECTED / INDETERMINATE
            "queue_growth": r.get("queue_growth"),
            "completion_fraction": cf,
            "completion_bound_s": r.get("completion_bound_s"),
            "short_replies": r.get("short_replies"),
            "no_text_replies": r.get("no_text_replies"),
            "flags": flags,
            "contaminated": r.get("contaminated"),
            "foreign_requests": r.get("foreign_requests"),
            "engine_above_client": r.get("engine_above_client"),
            "client_bound": client_bound,
            "client_bound_reasons": r.get("client_bound_reasons"),
            "client_lag_p99_ms": r.get("client_lag_p99_ms"),
            "client_sent_frac": r.get("client_sent_frac"),
            "request_throughput": achieved,   # back-compat alias for run()/reports
            "slo_attainment": attain,
            "completed": completed,
            "failed": failed,
            "p99_ttft_ms": p99_ttft,
            "p99_tpot_ms": p99_tpot,
            "p99_itl_ms": p99_itl,                    # client inter-chunk gap
            "p99_itl_per_token_ms": p99_itl_tok,      # client per-token gap
            "tpot_source": r.get("tpot_source"),
            "server_itl_kind": "per_engine_step_gap",
            "itl_gated_ms": itl_metric,               # the value that decided the ITL SLO
            "itl_gated_limit_ms": itl_limit,          # ...against this limit
            "tokens_per_engine_step": r.get("tokens_per_engine_step"),
            "spec_decode_active": r.get("spec_decode_active"),
            "itl_client_server_disagree": itl_disagree,
            "p99_itl_server_ms": p99_itl_server,      # upper edge of the P99 bucket (None if absent)
            "p99_itl_server_bucket": r.get("p99_itl_server_bucket"),
            "itl_gate_outcome": itl_outcome,          # PASS / FAIL / INDETERMINATE / None
            "itl_gate_edge_ms": r.get("itl_gate_edge_ms"),
            "itl_frac_over_limit": r.get("itl_frac_over_limit"),
            "itl_hist_delta_ms": r.get("itl_hist_delta_ms"),
            # Did the server see exactly the planned prompts, uncached? (perf_payload)
            "prompt_len_checked": r.get("prompt_len_checked"),
            "prompt_len_mismatch": r.get("prompt_len_mismatch"),
            "prefix_cache_hits": r.get("prefix_cache_hits"),
            "prefix_cache_queries": r.get("prefix_cache_queries"),
            "mm_cache_hits": r.get("mm_cache_hits"),
            "cache_reuse_suspect": r.get("cache_reuse_suspect"),
            "itl_slo_source": itl_source,             # which ITL gated this point
            "mean_e2e_ms": mean_e2e,
            "steady_n": steady_n,
            "passed": passed,
            # pass / fail / inconclusive (the point could not be judged: too short a window,
            # no data, a short certification cohort, or a client that fell behind)
            "verdict": verdict,
        }

        if verdict == "inconclusive":
            status = ("⚑ CLIENT-BOUND" if client_bound    # load-gen bottleneck, NOT a pass
                      else "⚑ INCONCLUSIVE")              # short window/cohort or no data
        elif saturation == SATURATED:
            status = "⚠ SATURATED"     # the queue grew over the cohort
        elif not keep_up:
            status = "⚠ BACKLOG"       # too few requests finished within the SLO-implied time
        elif inconclusive and not decisive_fail:
            status = "⚑ INCONCLUSIVE"  # window too short / no data even after a re-run
        elif not enough:
            status = "⚠ TOO-FEW"       # not enough cohort requests to trust
        elif not low_failure:
            status = "⚠ FAILURES"      # too many offered requests errored/dropped
        elif client_bound:
            status = "⚑ CLIENT-BOUND"  # load-gen bottleneck: inconclusive, NOT a pass
        elif not slo_ok:
            status = "⚠ SLO-MISS"      # keeps up but latency SLO not met
        else:
            status = "✓ PASSED"
        if itl_disagree:
            status += "  [client/server ITL disagree >1 bucket]"
        if spec and not getattr(self, "_spec_noted", False):
            self._spec_noted = True
            logger.info("Speculative decoding active: the server ITL counts one gap per engine "
                        "step, so the per-token SLO is judged on the client per-token gap, or "
                        "(no tokenizer) on the server histogram against SLO x tokens per step.")
        logger.info(
            f"{qps:>7.1f} req/s   ach {achieved:>6.1f}   TTFT {p99_ttft:>7.0f}ms   "
            f"ITL {(r.get('p99_itl_server_bucket') if itl_source == 'server' else f'{itl_metric:.0f}ms')} ({itl_source})   SLO {attain*100:>4.0f}%   {status}"
        )
        return passed, p99_ttft, p99_tpot, achieved

    def _sweep_start_qps(self) -> float:
        """Starting arrival rate for the ascending sweep.

        Seed it near HALF the single-stream capacity implied by the preflight's e2e
        latency (capacity ~= 1000/e2e_ms), so it starts comfortably INSIDE the
        sustainable zone on a slow server (few requests -> the start almost always
        passes, avoiding the downward fallback) while still being high enough to reach
        a fast server's knee in a few doublings. Falls back to initial_qps when no
        probe latency is available."""
        e2e_ms = getattr(self, "_probe_e2e_ms", None)
        if e2e_ms and e2e_ms > 0:
            start = 0.5 * (1000.0 / e2e_ms)   # half the single-stream capacity
            return float(min(self.max_qps, max(self._min_down_qps(), start)))
        return float(min(self.max_qps, self.initial_qps))

    def _min_down_qps(self) -> float:
        """Floor of the downward search, scaled to the device: min(0.05, 0.05 / single-
        stream e2e). A fixed 0.05 req/s floor sits above a slow device's limit."""
        e2e = (getattr(self, "_probe_e2e_ms", None) or 0.0) / 1000.0
        return min(self.MIN_DOWN_QPS, self.MIN_DOWN_QPS / e2e) if e2e > 1.0 else self.MIN_DOWN_QPS

    def _point(self, qps: float, model, format, min_n: Optional[int] = None,
               cache: Optional[dict] = None) -> bool:
        """Drain, then measure one rate (True only on a pass). The verdict, including
        "inconclusive", is in the point's record (_point_verdict)."""
        return self._point_verdict(qps, model, format, min_n, cache) == "pass"

    def _point_verdict(self, qps: float, model, format, min_n: Optional[int] = None,
                       cache: Optional[dict] = None) -> str:
        """Drain (never measure on a server that did not drain: wait again; a second
        consecutive timeout ends the pillar as contaminated), measure, and return
        "pass" / "fail" / "inconclusive"."""
        self._ensure_drained()
        self._test_rate(qps, model, format, min_n=min_n, cache=cache)
        store = self._tested_rates if cache is None else cache
        return (store.get(qps) or {}).get("verdict", "fail")

    def _ensure_drained(self) -> None:
        if not self._drain_or_raise():
            self._drain_or_raise()

    def _sweep_once(self, model, format) -> dict:
        """One ASCENDING open-loop sweep: the bracket [last pass, first failure].

        Low to high, x2 per step, stopping at the first failure, so the server is pushed
        over capacity once per repeat. If the seeded start already fails, rates halve;
        after a second overloaded probe the start is re-seeded from a fresh single-stream
        probe (half its capacity) instead of halving further. Every point starts on a
        drained server. The bracket is refined and certified afterwards
        (_certify_knee), once, on the median repeat."""
        self._tested_rates = {}
        self._point_seq = 0
        start = _sig(self._sweep_start_qps())
        last_pass, first_fail, overloads = 0.0, None, 0
        floor = self._min_down_qps()

        if self._point(start, model, format):
            last_pass = start
            nxt = _sig(last_pass * 2)
            while nxt <= self.max_qps:
                if not self._point(nxt, model, format):
                    first_fail, overloads = nxt, overloads + 1
                    break
                last_pass = nxt
                nxt = _sig(last_pass * 2)
            else:
                logger.info(f"Reached the safety cap {self.max_qps} req/s.")
        else:
            first_fail, overloads = start, 1
            probe = _sig(start / 2)
            reseeded = False
            while probe >= floor and probe not in self._tested_rates:
                if self._point(probe, model, format):
                    last_pass = probe
                    break
                first_fail, overloads = probe, overloads + 1
                if overloads >= 2 and not reseeded:
                    reseeded = True
                    e2e = self._fresh_single_stream_e2e(model, format)
                    seed = reseed_start(e2e, floor=floor)
                    probe = _sig(min(probe / 2, seed)) if seed else _sig(probe / 2)
                else:
                    probe = _sig(probe / 2)
        best = self._tested_rates.get(last_pass, {})
        return {
            "knee_qps": last_pass,
            "bracket": {"pass": last_pass, "fail": first_fail},
            "overloaded_probes": overloads,
            "below_floor": last_pass == 0.0,
            "floor_qps": floor,
            "achieved_at_knee": best.get("achieved_qps", 0.0),
            "best": best,
            "tested_rates": sorted(self._tested_rates.values(), key=lambda r: r["request_rate_qps"]),
        }

    # ── Certification of the reported limit ──────────────────────────────────────
    # The bracket's pass is decided on ~40-hundreds of requests, where a P99 is noise:
    # the sweep finds the region, certification decides the number. Each certification
    # probe uses enough requests for the gated percentile (statistics.tail_validity: a
    # cohort of 1099 for a P99, about 1400 sent), the same rule on every device; geometric bisection stops at a 1.19
    # high/low ratio (+-9%).
    CERTIFY_STOP_RATIO = 1.19
    CERTIFY_BUDGET_MIN = CERTIFY_BUDGET_MIN_DEFAULT
    CERTIFY_PERCENTILE = 99.0

    def _certify_n(self) -> int:
        from ..analysis.statistics import tail_validity
        n = 10
        while not tail_validity(n, self.CERTIFY_PERCENTILE)[0]:
            n = int(n * 1.25) + 1
        return n

    def _probe_prompts(self, qps: float, min_n: Optional[int] = None,
                       l_est: Optional[float] = None) -> int:
        """Requests a point sends (the one formula _test_rate and the certification cost
        both use): never below the floor; a certification probe adds the share of the
        schedule its cohort window excludes so the cohort itself holds min_n."""
        n = request_count(qps, self._min_prompts, self._min_steady, l_est,
                          window_s=self.MEASURE_ARRIVAL_S, cap=self._prompt_cap())
        if min_n:
            n = max(n, int(min_n * 1.25) + int(qps * (l_est or 0.0)))
        return n

    def _probe_cost_min(self, qps: float, min_n: int, l_est: float) -> float:
        """Wall time of one certification probe: its schedule, the tail of its slowest
        requests (3 x latency) and the drain before it."""
        n = self._probe_prompts(qps, min_n, l_est)
        return (n / qps + 3.0 * l_est + drain_gap_s(getattr(self, "_cold_preprocess_s", None))
                ) / 60.0

    def _certify_knee(self, model, format, bracket: dict) -> dict:
        """Certify a limit inside [pass, fail] with percentile-sized probes, within the
        time budget.

        Geometric bisection between the bracket's pass and fail (or the rate cap); a
        probe passes only if its cohort holds n_cert requests. Outcomes:
          certified      value = the highest certified rate; resolution_pct is +-9% only
                         when the interval reached the 1.19 ratio (else wider, said so);
          refuted        probes failed at and below the bracket's pass down to the
                         floor (refuted_down_to): no limit is reported;
          unverified     the budget ran out first: the bracket's pass stands, labelled
                         UNCERTIFIED, with needed_min (the full remaining time)."""
        self._cert_points = []
        mode, budget_min = certify_settings(self.config)
        bracket_pass, bracket_fail = float(bracket.get("pass") or 0.0), bracket.get("fail")
        out = {"mode": mode, "budget_min": budget_min, "certified": False, "value": None,
               "outcome": "not run", "probes": [], "n_per_probe": self._certify_n(),
               "stop_ratio": self.CERTIFY_STOP_RATIO}
        if bracket_fail is None and bracket_pass >= float(self.max_qps):
            out.update(certified=False, outcome="censored",
                       reason=f"passed at the --stress-max-qps cap ({self.max_qps:g}); raise it "
                              f"to measure the limit")
            return out
        if mode == "off" or bracket_pass <= 0 or (mode == "auto" and budget_min <= 0):
            out["reason"] = ("certification off" if mode == "off"
                             else "no passing rate to certify" if bracket_pass <= 0
                             else "certification budget 0 (off)")
            return out
        n_cert = out["n_per_probe"]
        lat = getattr(self, "_last_pass_p90_s", None) or (self._probe_e2e_ms or 1000.0) / 1000.0
        spent = 0.0
        cache: dict = {}
        floor = self._min_down_qps()

        lo = bracket_pass
        if bracket_fail:
            hi = float(bracket_fail)
        else:                                     # never failed: bounded by the rate cap
            hi = min(2.0 * lo, float(self.max_qps))
            out["censored_at_cap"] = hi >= float(self.max_qps)

        def remaining_min(lo_v, hi_v):
            steps = (math.ceil(math.log2(max(1.0, math.log(hi_v / lo_v)
                                            / math.log(self.CERTIFY_STOP_RATIO)))) + 1
                     if hi_v / lo_v > self.CERTIFY_STOP_RATIO else 0)
            mid = math.sqrt(lo_v * hi_v)
            return steps * self._probe_cost_min(mid, n_cert, lat) + \
                self._probe_cost_min(lo_v, n_cert, lat)      # + one step-down reserve

        def probe(rate):
            """True / False / None (budget) / "inconclusive" (judged twice, never)."""
            nonlocal spent
            for attempt in range(2):
                cost = self._probe_cost_min(rate, n_cert, lat)
                if mode != "always" and spent + cost > budget_min:
                    return None
                t = time.time()
                cache.pop(rate, None)               # a retest measures afresh
                v = self._point_verdict(rate, model, format, min_n=n_cert, cache=cache)
                spent += (time.time() - t) / 60.0
                out["probes"].append({"rate": rate, "verdict": v})
                if v != "inconclusive":
                    return v == "pass"
            return "inconclusive"

        best, out_of_budget, undecided = None, False, None
        while hi / lo > self.CERTIFY_STOP_RATIO:
            mid = _sig(math.sqrt(lo * hi))
            if not lo < mid < hi:                   # no rate left between them at 4 digits
                break
            res = probe(mid)
            if res is None:
                out_of_budget = True
                out["needed_min"] = round(spent + remaining_min(lo, hi), 1)
                break
            if res == "inconclusive":
                undecided = mid                     # never a failure, never the new "hi"
                break
            if res:
                lo, best = mid, mid
            else:
                hi = mid
        refuted_to = None
        if best is None and not out_of_budget and undecided is None:
            # Nothing above the bracket's pass certified: certify the pass itself, then
            # step down by sqrt(2) (never report 0; below the floor is said so).
            rate = bracket_pass
            while rate >= floor and rate > 0:
                rate = _sig(rate)
                res = probe(rate)
                if res is None:
                    out_of_budget = True
                    out["needed_min"] = round(spent + remaining_min(max(floor, rate / 2), rate), 1)
                    break
                if res == "inconclusive":
                    undecided = rate
                    break
                if res:
                    best = rate
                    hi = min(hi, _sig(rate * math.sqrt(2.0))) if rate < bracket_pass else hi
                    break
                refuted_to = rate
                nxt = _sig(rate / math.sqrt(2.0))
                if not nxt < rate:
                    break
                rate = nxt
            if best is None and not out_of_budget and undecided is None:
                out["below_floor"] = floor
        out["spent_min"] = round(spent, 1)
        self._cert_points = sorted(cache.values(), key=lambda r: r["request_rate_qps"])
        if best is not None:
            censored = bool(out.get("censored_at_cap")) and not any(
                not p.get("verdict") == "pass" for p in out["probes"] if p["rate"] > best)
            if censored:
                out.update(certified=True, value=best, outcome="certified",
                           interval=[best, None], resolution_pct=None,
                           note=f"lower bound, censored at --stress-max-qps {self.max_qps:g}; "
                                f"raise it to measure")
            else:
                ratio = hi / best
                out.update(certified=True, value=best, outcome="certified", interval=[best, hi],
                           resolution_pct=round((math.sqrt(ratio) - 1) * 100, 1))
                if ratio > self.CERTIFY_STOP_RATIO + 1e-9:
                    out["note"] = (f"certified lower bound (resolution +-{out['resolution_pct']}%)"
                                   + (f"; +-9% needs ~{out['needed_min']:.0f} min "
                                      f"(--stress-certify-budget {int(out['needed_min']) + 1})"
                                      if out.get("needed_min") else ""))
        elif undecided is not None:
            out.update(outcome="unverified",
                       reason=f"the certification probe at {undecided:g} req/s stayed "
                              f"inconclusive after a retest (see its flags)")
        elif refuted_to is not None:
            out.update(outcome="refuted", refuted_down_to=refuted_to,
                       reason=f"certification probes failed at and below the sweep's pass, "
                              f"down to {refuted_to:g} req/s")
        else:
            out.update(outcome="unverified",
                       reason="the certification budget ran out before any probe passed")
        return out

    def _fresh_single_stream_e2e(self, model, format) -> Optional[float]:
        """Median e2e (s) of a few sequential requests on a drained server."""
        try:
            self._ensure_drained()
            api_url, cf, payloads = self._build_payloads(model, format, self.PREFLIGHT_PROBES,
                                                         phase="probe", point=self._point_seq)
            e2es = []
            for p in payloads:
                out = _stress_client_worker({"payloads": [p], "api_url": api_url,
                                             "content_field": cf,
                                             "request_timeout": self._request_timeout(),
                                             "max_point_s": self._max_point_s(),
                                             "itl_cap": self.ITL_SAMPLE_CAP,
                                             "headers": self._request_headers()})
                e2es += [r["done"] - r["send"] for r in out.get("records", [])
                         if r.get("ok") and r.get("done")]
            self._release_media()
            return float(np.median(e2es)) if e2es else None
        except StressContaminated:
            raise
        except Exception:                                                 # noqa: BLE001
            return None

    # ── Drain (perf_sweep.drained) ───────────────────────────────────────────────
    def _drain_or_raise(self) -> bool:
        """Wait until the server is idle (no running/waiting request, no request
        finishing or being admitted between two scrapes). Without /metrics, wait a fixed
        cool-down. A drain that times out twice in a row ends the pillar as
        contaminated (something else keeps the server busy)."""
        if not self._metrics_url():
            return True                           # no server known (nothing to drain)
        gap = drain_gap_s(getattr(self, "_cold_preprocess_s", None))
        lat = getattr(self, "_last_pass_p90_s", None) or (self._probe_e2e_ms or 1000.0) / 1000.0
        timeout = max(60.0, 3.0 * lat)
        t_end = time.time() + timeout
        while True:
            before = self._metrics_snapshot()
            time.sleep(gap)
            after = self._metrics_snapshot()
            state = drained(before, after)
            if state is not None:
                self._metrics_seen = True
            elif not getattr(self, "_metrics_seen", False):   # no /metrics: fixed cool-down
                time.sleep(max(5.0, lat))
                self._drain_timeouts = 0
                return True
            # (a failed scrape on a server whose /metrics parsed before is "not drained")
            if state:
                self._drain_timeouts = 0
                return True
            if time.time() >= t_end:
                self._drain_timeouts = getattr(self, "_drain_timeouts", 0) + 1
                if self._drain_timeouts >= 2:
                    raise StressContaminated(
                        "the server never drained between points (twice): other traffic is "
                        "keeping it busy")
                logger.warning("Server not drained after %.0f s; waiting once more before "
                               "measuring.", timeout)
                return False

    def _save_stress_result(self, final_result: dict, model, format, multimodal: bool) -> None:
        """Persist a stress result to results/performance/stress_<model>_<fmt>[_<campaign>]_<mode>.json
        (one file per campaign; the record carries the campaign too)."""
        lm = self.config.log_manager
        results_dir = lm.results_dir / "performance"
        results_dir.mkdir(parents=True, exist_ok=True)
        mode_suffix = "mm" if multimodal else "text"
        camp = getattr(self.config, "campaign", None)
        camp = camp if isinstance(camp, str) and camp else None
        if camp:
            final_result.setdefault("campaign", camp)
        output_file = results_dir / (f"stress_{model.short_name}_{format.value}"
                                     f"{'_' + camp if camp else ''}_{mode_suffix}.json")
        write_redacted_json(output_file, final_result, default=str)
        logger.info(f"Results saved to: {output_file}")

    def _preflight_floor_check(self, model, format, multimodal: bool) -> Optional[dict]:
        """Fast-fail probe run BEFORE the open-loop sweep on slow hardware.

        The sweep is an OPEN-LOOP arrival test: arrivals fire on a schedule
        regardless of whether the server keeps up, so queueing only ADDS to the
        single-stream latency. If even ONE request at a time already exceeds the
        P99 TTFT SLO, no arrival rate can be sustainable - the whole descending
        ramp would just report TOO-FEW for tens of minutes. So send a few STRICTLY
        SEQUENTIAL requests, take the best-case (min) single-stream TTFT, and if it
        already blows the SLO, return an early-exit result (sustainable QPS = 0).
        Return None to proceed with the normal sweep (the common case; on capable
        hardware the probe is a handful of quick requests).
        """
        try:
            api_url, content_field, payloads = self._build_payloads(
                model, format, self.PREFLIGHT_PROBES, phase="preflight")
        except Exception as e:
            logger.warning(f"Stress preflight skipped (payload build failed: {e}); running sweep.")
            return None
        if not payloads:
            return None
        ttfts, e2es, probe_outs = [], [], []
        for p in payloads[:self.PREFLIGHT_PROBES]:
            # One payload => the worker sends exactly one request and awaits it:
            # a true single-stream (concurrency-1) latency sample. qps is irrelevant
            # for a single request (no inter-arrival wait is taken).
            try:
                out = _stress_client_worker({
                    "payloads": [p], "qps": 0.0, "seed": 83,
                    "api_url": api_url, "content_field": content_field,
                    "request_timeout": self._request_timeout(),
                    "max_point_s": self._max_point_s(), "itl_cap": self.ITL_SAMPLE_CAP,
                    "headers": self._request_headers(),
                })
            except Exception as e:
                logger.warning(f"Stress preflight probe errored ({e}); running sweep.")
                return None
            probe_outs.append(out)
            for rec in out.get("records", []):
                if rec.get("ok") and rec.get("ttft") is not None:
                    ttfts.append(float(rec["ttft"]))
                    if rec.get("done") and rec.get("send"):
                        e2es.append((rec["done"] - rec["send"]) * 1000.0)
        self._release_media()
        if not ttfts:
            # Not one clean single-stream request produced output, even after warmup.
            # The sweep would then report 0 completions / infinite TTFT for every rate
            # (a doomed ~20-min grind), so bail now with an honest diagnostic instead.
            logger.warning(
                "\n" + "=" * 78 + "\n"
                "  STRESS TEST SKIPPED - endpoint returned no usable stream\n"
                + "=" * 78 + "\n"
                "  No single-stream probe produced any output tokens (after warmup), so the\n"
                "  open-loop sweep cannot measure capacity. Likely the endpoint does not\n"
                "  stream chat/completions as expected, or the model emits an immediate EOS\n"
                "  for this workload. Check the endpoint, or use --no-stress-test.\n"
                + "=" * 78)
            _ds, _in, _out = self._workload_shape()
            return {
                "stress_test": True,
                "below_stress_floor": True,
                "preflight_no_stream": True,
                "slo_met": False,
                "max_sustainable_qps": 0.0,
                "max_sustainable_throughput": 0.0,
                "preflight_probes": 0,
                "ttft_threshold_ms": self.ttft_threshold_ms,
                "itl_threshold_ms": self.tpot_threshold_ms,
                "tpot_threshold_ms": self.tpot_threshold_ms,
                "reps": 0,
                "workload_dataset": "multimodal" if multimodal else _ds,
                "workload_input_length": _in,
                "workload_output_length": _out,
                "multimodal": multimodal,
                "model": model.short_name,
                "format": format.value,
                "note": ("stress requests produced no output tokens against this endpoint "
                         "(0 completions); the sweep cannot measure capacity. Verify the "
                         "endpoint streams chat/completions, or use --no-stress-test."),
            }
        # Does the server honour ignore_eos? (every probe reached max_tokens). Only then
        # are short replies failures during the sweep; otherwise they are recorded.
        _ds, _in, _out = self._workload_shape()
        _mt = max(int(_out), 16)
        _ct = [rec.get("completion_tokens") for o in probe_outs for rec in o.get("records", [])
               if rec.get("ok")]
        self._ignore_eos_honored = (None if not _ct or any(c is None for c in _ct)
                                    else all(c >= _mt for c in _ct))
        best_ttft = min(ttfts)
        # Stash the single-stream latency so the ascending sweep can seed its start
        # rate near half the implied capacity (1000/e2e), on both slow and fast servers.
        self._probe_ttft_ms = best_ttft
        if e2es:
            self._probe_e2e_ms = min(e2es)
        logger.info(
            f"Stress preflight: best single-stream TTFT {best_ttft:.0f}ms over "
            f"{len(ttfts)} probe(s) vs P99 TTFT SLO {self.ttft_threshold_ms}ms.")
        # Best single-stream per-token decode time (server token count): load only
        # slows decode, so if one request at a time already decodes slower than the ITL
        # SLO, no rate can pass the ITL gate.
        tok_ms = [((rec["done"] - rec["send"]) * 1000.0 - float(rec["ttft"]))
                  / (int(rec["completion_tokens"]) - 1)
                  for o in probe_outs for rec in o.get("records", [])
                  if rec.get("ok") and rec.get("ttft") is not None
                  and rec.get("done") is not None and rec.get("send") is not None
                  and (rec.get("completion_tokens") or 0) > 1]
        best_tok = min(tok_ms) if tok_ms else None
        if best_tok is not None:
            logger.info(f"Stress preflight: best single-stream per-token time {best_tok:.0f}ms "
                        f"vs P99 ITL SLO {self.tpot_threshold_ms}ms.")
        if best_ttft <= self.ttft_threshold_ms and (best_tok is None
                                                    or best_tok <= self.tpot_threshold_ms):
            return None   # single stream can meet the SLOs -> the sweep is meaningful
        # Below the floor: no open-loop rate can pass. Bail with an honest result.
        if best_ttft > self.ttft_threshold_ms:
            why, fix = (f"A single request already takes {best_ttft:.0f}ms to first token, above "
                        f"the P99 TTFT SLO of {self.ttft_threshold_ms}ms.", "--stress-threshold <ms>")
        else:
            why, fix = (f"A single request already decodes at {best_tok:.0f}ms per token, above "
                        f"the P99 ITL SLO of {self.tpot_threshold_ms}ms.", "--itl-slo-ms <ms>")
        logger.warning(
            "\n" + "=" * 78 + "\n"
            f"  STRESS TEST SKIPPED - hardware below the stress floor\n"
            + "=" * 78 + "\n"
            f"  {why}\n"
            f"  Open-loop arrivals only add queueing, so no arrival rate can meet the SLOs.\n"
            f"  -> Use --no-stress-test on this machine (the serving latency sweep already\n"
            f"     captures its single-stream TTFT/TPOT), or raise the SLO with\n"
            f"     {fix} if you want a capacity number at this latency.\n"
            + "=" * 78)
        _ds, _in, _out = self._workload_shape()
        return {
            "stress_test": True,
            "below_stress_floor": True,
            "slo_met": False,
            "max_sustainable_qps": 0.0,
            "max_sustainable_throughput": 0.0,
            "preflight_single_stream_ttft_ms": best_ttft,
            "preflight_single_stream_e2e_ms": (min(e2es) if e2es else None),
            "preflight_single_stream_per_token_ms": best_tok,
            "below_stress_floor_reason": "TTFT" if best_ttft > self.ttft_threshold_ms else "ITL",
            "preflight_probes": len(ttfts),
            "ttft_threshold_ms": self.ttft_threshold_ms,
            "itl_threshold_ms": self.tpot_threshold_ms,
            "tpot_threshold_ms": self.tpot_threshold_ms,
            "reps": 0,
            "workload_dataset": "multimodal" if multimodal else _ds,
            "workload_input_length": _in,
            "workload_output_length": _out,
            "multimodal": multimodal,
            "model": model.short_name,
            "format": format.value,
            "note": (f"{why} No open-loop arrival rate can meet the SLOs on this hardware. "
                     f"Use --no-stress-test, or raise {fix.split()[0]} to get a capacity number."),
        }

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> dict:
        """Sweep the open-loop arrival rate to find max sustainable QPS under SLO.

        Args:
            model: Model configuration
            format: Model format (hf, or remote-endpoint)
            multimodal: Whether to test multimodal (sends real images)

        Returns:
            Dictionary with stress test results
        """
        mode_str = "Multimodal" if multimodal else "Text"
        self._tested_rates = {}  # Reset cache
        # A new payload call per run (model/format/mode): no request of this run repeats
        # one of an earlier run on this runner, and per-run counters start at zero.
        self._payloads.new_call()
        self._last_pass_p90_s = None      # this run's latency, not the previous run's
        self._point_log = []              # every measurement of this run
        self._cert_points = []
        self._drain_timeouts = 0
        self._cold_preprocess_s = None
        self._real_pool, self._real_seen, self._real_cursor = [], set(), 0
        self._prompt_reuse = 0
        self._prompt_overhead = None
        self._multimodal = multimodal
        # Per-image token count for this model: its declared count until the MM preflight
        # measures the server's (None when it declares none).
        _declared = getattr(model, "vision_tokens_per_image", None)
        self._vision_tokens_per_image = _declared if isinstance(_declared, int) and _declared > 0 \
            else None
        self._vision_tokens_source = ((getattr(model, "vision_tokens_source", None) or "config")
                                      if self._vision_tokens_per_image else None)

        if self.config.dry_run:
            logger.info("[DRY RUN] Would run open-loop QPS stress sweep")
            return {
                "dry_run": True,
                "stress_test": True,
                "model": model.short_name,
                "format": format.value,
                "multimodal": multimodal,
                "ttft_threshold_ms": self.ttft_threshold_ms,
                "tpot_threshold_ms": self.tpot_threshold_ms,
            }

        try:
            # Set up multimodal images if needed
            mm_media_path = None
            if multimodal:
                mm_media_path = self._setup_mm_images(num_images=50)

            self._serving_runner = ServingBenchmarkRunner(self.config)
            # The tokenizer first: without it no prompt can be sized, so fail before
            # starting a server.
            tokenizer_path = self._serving_runner._tokenizer_path(model, format)
            # It is needed for text request generation AND for sizing the
            # MM padded-text prompt, so load it in both modes.
            from gbench.utils import safe_get_tokenizer
            self._tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer,
                trust_remote_code=_tokenizer_trust(self.config))
            if self._tokenizer is None:
                # Prompts are sized with the tokenizer; without one the workload would be
                # unsized text, so the pillar does not run.
                raise RuntimeError(
                    f"Could not load a tokenizer for '{self.config.tokenizer or tokenizer_path}' - "
                    f"the stress test sizes its prompts with it. Pass --tokenizer <hf-id-or-path> "
                    f"and install transformers (pip install transformers).")
            logger.info(f"Tokenizer loaded for: {self.config.tokenizer or tokenizer_path}")

            # Start vLLM server once (reused across all rate points)
            if not self.config.remote_endpoint:
                if not self._serving_runner._start_server(
                    model, format, allowed_local_media_path=mm_media_path,
                    mm_limit_images=self._images_per_request() if multimodal else 1,
                ):
                    # A server that never comes ready must fail the campaign, not
                    # silently sweep a dead endpoint and record a phantom 0-QPS
                    # "measurement". The raise is caught below -> failed dict.
                    raise RuntimeError("Failed to start vLLM server for stress test")

            self._api_model_id = self._serving_runner._resolve_model_id(model, format)

            # Preflight: prove the server ingests images before sweeping, so an MM
            # stress knee can never be measured on silently-text-only requests.
            if multimodal:
                self._serving_runner._mm_image_paths = self._mm_image_paths
                base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self._serving_runner.server_port}"
                _t = time.time()
                measured_per_img = self._serving_runner._assert_mm_images_processed(
                    self._api_model_id, base_url)
                # The preflight sends two cold requests (text only, then with images):
                # their mean time bounds the server's cold multimodal preprocessing, which
                # sets the spacing of the drain check (requests still in preprocessing are
                # invisible to the engine gauges).
                self._cold_preprocess_s = (time.time() - _t) / 2.0
                # Size + report the MM workload against the SERVER-MEASURED per-image
                # token count, not a declared or default one: processor settings
                # (e.g. --mm-processor-kwargs) change what the server charges per image,
                # and a wrong count mis-sizes the padded text budget and the reported
                # prefill.
                if measured_per_img and measured_per_img >= self.MIN_MM_TEXT_TOKENS:
                    if self._vision_tokens_per_image and \
                            int(measured_per_img) != int(self._vision_tokens_per_image):
                        logger.info(
                            f"MM stress: using server-measured "
                            f"{int(measured_per_img)} vision tokens/image "
                            f"(declared {int(self._vision_tokens_per_image)}).")
                    self._vision_tokens_per_image = int(measured_per_img)
                    self._vision_tokens_source = "measured"

            logger.info(
                f"\n{'='*66}\n"
                f"  STRESS TEST - open-loop QPS sweep (capacity under SLO)\n"
                f"{'='*66}\n"
                f"  Mode:      {mode_str}\n"
                f"  Model:     {model.short_name} ({format.value})\n"
                f"  Workload:  {self._workload_desc()}\n"
                f"  Sustainable: no growing queue, >= {self.KEEP_UP_FRAC:.0%} of requests done within the\n"
                f"               SLO-implied time, P99 TTFT <= {self.ttft_threshold_ms}ms AND P99 ITL <= {self.tpot_threshold_ms}ms,\n"
                f"               few failures, client on schedule (on each point's planned-send cohort).\n"
                f"  Sweep:     {self._reps} repeats of an ascending x2 ramp (stop at first miss, cap {self.max_qps} req/s);\n"
                f"             the median repeat's limit is then certified (probes of >= {self._certify_n()} cohort requests)\n"
                f"  Client:    up to {self._num_client_procs} processes sending one merged Poisson schedule\n"
                f"{'='*66}"
            )
            # Warmup once (discarded) so the first measured point isn't cold; the
            # server is reused across every rep, so one warmup suffices.
            self._warmup(model, format)
            # How many prompt tokens the server adds to our planned text (template,
            # BOS, system prompt): the per-request length and cache checks need it.
            self._calibrate_prompt_overhead(model, format)

            # Fast-fail floor check, AFTER warmup so it probes a warm (not cold-loading)
            # model: if even a single stream can't meet the TTFT SLO, no open-loop
            # arrival rate can (arrivals only add queueing), so bail with a clear result
            # instead of grinding every rep's descending ramp into TOO-FEW.
            pf = self._preflight_floor_check(model, format, multimodal)
            if pf is not None:
                self._serving_runner._stamp_server(pf, model, format)
                pf.update(methodology_record("stress"))
                pf.update(self._payload_record())
                self._save_stress_result(pf, model, format, multimodal)
                return pf

            # The sweep runs STRESS_REPS times (varied, deterministic seeds); each gives a
            # bracket [pass, fail]. The headline is the median repeat's pass, certified
            # once with percentile-sized probes (_certify_knee).
            reps = max(1, self._reps)
            rep_results, knees = [], []
            for rep in range(reps):
                self._rep_seed = 83 + rep * 101   # varied, deterministic per rep
                self._rep_idx = rep
                logger.info("=" * 78)
                logger.info(f"Sweep rep {rep + 1}/{reps} ({self._num_client_procs} client procs)")
                logger.info(
                    f"{'Rate':>9}   {'Achieved':>9}   {'P99 TTFT':>9}   "
                    f"{'P99 ITL':>9}   {'%SLO':>5}   Status"
                )
                logger.info("-" * 78)
                sw = self._sweep_once(model, format)
                rep_results.append(sw)
                knees.append(sw["knee_qps"])
                logger.info(f"  rep {rep + 1}: bracket [{sw['bracket']['pass']}, "
                            f"{sw['bracket']['fail']}] req/s, {sw['overloaded_probes']} "
                            f"overloaded probe(s)")

            summary = summarize_brackets([rr["bracket"] for rr in rep_results], step=2.0)
            kstats = _knee_stats(knees)
            med_idx = summary.get("median_rep", 0)
            med = rep_results[med_idx] if rep_results else {}
            logger.info("-" * 78)
            logger.info(f"Certifying the median repeat's bracket {med.get('bracket')} "
                        f"(probes of >= {self._certify_n()} cohort requests)...")
            self._rep_idx = reps            # certification probes: their own payload keys
            cert = self._certify_knee(model, format, med.get("bracket") or {})
            if summary.get("headline", 0.0) == 0.0 and not cert["certified"]:
                max_qps = None                     # nothing passed down to the floor
            elif cert.get("outcome") == "refuted":
                max_qps = None                     # certification failed below the pass
            elif cert["certified"]:
                max_qps = cert["value"]
                cert_pts = [pt for pt in (self._cert_points or [])]
                best = next((pt for pt in reversed(cert_pts)
                             if pt.get("request_rate_qps") == max_qps), None) or med.get("best") or {}
            else:
                max_qps = summary.get("headline", 0.0)
                best = med.get("best") or {}
            if max_qps is None:
                best = {}
            max_tp = best.get("request_throughput", 0.0)
            label = "CERTIFIED" if cert["certified"] else "UNCERTIFIED"
            logger.info(
                f"✅ Max sustainable arrival rate: "
                f"{(f'{max_qps:.3g} req/s ' + label) if max_qps is not None else 'none'} "
                f"(brackets {[ (b['pass'], b['fail']) for b in summary.get('brackets', [])]}, "
                f"consistent={summary.get('consistent')})")
            if cert.get("note"):
                logger.info(f"   {cert['note']}")
            elif cert.get("needed_min"):
                logger.info(f"   Certifying it needs ~{cert['needed_min']:.0f} min; re-run with "
                            f"--stress-certify-budget {int(cert['needed_min']) + 1} to certify.")
            elif cert.get("reason"):
                logger.info(f"   Not certified: {cert['reason']}")

            _ds, _in, _out = self._workload_shape()
            # For MM, the reported input length is the ACTUAL prefill (clamped text
            # + image placeholder tokens), not the campaign's nominal text input -
            # otherwise small-input campaigns (chat-like/decode-heavy in=128) are
            # labeled 128 while the server actually prefills ~1136 tokens/req.
            _in_reported = self._mm_effective_input_len(_in, _out) if multimodal else _in
            # Report the ACTUAL max client workers observed across the sweep, not the
            # configured cap: low-QPS points intentionally use 1 worker; a pool failure
            # re-runs the point (there is no in-process fallback). engaged=False means
            # multi-proc never ran.
            _all_points = [pt for rr in rep_results for pt in rr.get("tested_rates", [])]
            actual_max_procs = max((int(pt.get("num_client_procs", 1)) for pt in _all_points),
                                   default=1)
            multiproc_engaged = actual_max_procs > 1
            # No rate passed: below the floor only when the median repeat's points failed
            # on the server (SLO, saturation, backlog); else name what blocked them.
            _no_rate_reason = None
            _med_pts = sorted(med.get("tested_rates", []) if rep_results else [],
                              key=lambda p: p["request_rate_qps"])
            if max_qps is None and _med_pts:
                # The lowest rate tried decides: a server failure there is "below the
                # floor"; otherwise name what kept it from being judged.
                pt = _med_pts[0]
                _off = (pt.get("completed") or 0) + (pt.get("failed") or 0)
                if pt.get("verdict") == "inconclusive":
                    _no_rate_reason = ("client-bound (the load generator fell behind)"
                                       if pt.get("client_bound") else
                                       "inconclusive (" + "; ".join(pt.get("flags") or ["see flags"]) + ")")
                elif _off and (pt.get("failed") or 0) / _off > self.MAX_FAILURE_FRAC:
                    _no_rate_reason = "request failures (errors or timeouts from the server)"
            final_result = {
                "stress_test": True,
                "method": ("open_loop_steady_state_goodput_multiproc" if multiproc_engaged
                           else "open_loop_steady_state_goodput_singleproc"),
                "reps": reps,
                "num_client_procs": actual_max_procs,            # ACTUAL max workers used
                "num_client_procs_configured": self._num_client_procs,
                "client_multiproc_engaged": multiproc_engaged,
                "keep_up_frac": self.KEEP_UP_FRAC,
                "saturation_test": {"batches": QG_BATCHES, "alpha": QG_ALPHA,
                                    "material_slope_s_per_s": QG_MATERIAL_SLOPE,
                                    "statistic": "batch-median OLS slope of cohort TTFT"},
                "client_bound_limits": {"sent_frac": self.CLIENT_SENT_FRAC,
                                        "lag_slope_ms_per_s": self.CLIENT_LAG_SLOPE_MS_PER_S,
                                        "lag_p99_ms": self.CLIENT_LAG_P99_MS,
                                        "lag_p99_slo_frac": self.CLIENT_LAG_P99_SLO_FRAC},
                "ttft_threshold_ms": self.ttft_threshold_ms,
                "itl_threshold_ms": self.tpot_threshold_ms,
                "tpot_threshold_ms": self.tpot_threshold_ms,  # back-compat alias
                "workload_dataset": "multimodal" if multimodal else _ds,
                "workload_input_length": _in_reported,
                "workload_input_length_nominal": _in,
                "workload_mm_images": self._images_per_request() if multimodal else 0,
                "per_image_tokens": self._vision_tokens_per_image if multimodal else None,
                "per_image_tokens_source": self._vision_tokens_source if multimodal else None,
                "workload_output_length": _out,
                "max_sustainable_qps": max_qps,
                "max_sustainable_qps_certified": cert["certified"],
                "certification": {k: v for k, v in cert.items()},
                "certification_points": getattr(self, "_cert_points", []),
                # Bracket summary: per-repeat [pass, fail], whether the repeats agree
                # (highest pass < lowest fail), and whether their spread is below the
                # x2 grid (then the spread is "not assessed": it would be false precision).
                "brackets": summary,
                "max_sustainable_qps_stats": (kstats if not summary.get("spread_below_step")
                                              else {"reps": kstats.get("reps"),
                                                    "note": "spread below one grid step: not assessed"}),
                "max_sustainable_throughput": max_tp,
                "per_rep_knees": knees,
                "overloaded_probes_per_rep": [rr.get("overloaded_probes") for rr in rep_results],
                "sweep_points": med.get("tested_rates", []) if rep_results else [],
                "sweep_points_rep": med_idx,
                "sweep_points_all_reps": [rr["tested_rates"] for rr in rep_results],
                "multimodal": multimodal,
                "model": model.short_name,
                "format": format.value,
                "slo_met": bool(max_qps),
                "below_measurable_floor": (
                    (cert.get("below_floor") or med.get("floor_qps"))
                    if max_qps is None and cert.get("outcome") != "refuted"
                    and not _no_rate_reason else None),
                "no_rate_reason": _no_rate_reason if max_qps is None else None,
                "point_log": list(self._point_log),
            }
            self._serving_runner._stamp_server(final_result, model, format)
            final_result.update(methodology_record("stress"))
            final_result.update(self._payload_record())
            # The certification probes decide the headline: include them in every rollup.
            _cert_rr = [{"tested_rates": list(getattr(self, "_cert_points", []) or [])}]
            final_result["payload_check"] = self._payload_rollup(rep_results + _cert_rr)
            final_result["ignore_eos_honored"] = getattr(self, "_ignore_eos_honored", None)
            _pts = [pt for rr in rep_results + _cert_rr for pt in rr.get("tested_rates", [])]
            if self._tokenizer is None:
                _why = "no tokenizer"
            elif (getattr(self, "_recount_for", None) == id(self._tokenizer)
                  and getattr(self, "_recount_ok", None) is False):
                _why = getattr(self, "_recount_why", None)
            else:
                _why = None
            final_result["per_token_itl_disabled_reason"] = _why
            final_result["itl_disagree_points"] = sum(
                1 for pt in _pts if pt.get("itl_client_server_disagree"))
            final_result["spec_decode_points"] = sum(
                1 for pt in _pts if pt.get("spec_decode_active"))
            if best:
                # How the knee itself was judged: a NOT_DETECTED saturation verdict is not
                # proof the server kept up (detection is weak at low concurrency).
                for k in ("saturation", "completion_fraction", "flags", "cohort_n",
                          "cache_reuse_suspect", "prompt_len_mismatch"):
                    final_result[f"knee_{k}"] = best.get(k)
                if best.get("saturation") == NOT_DETECTED:
                    final_result["knee_saturation_note"] = (
                        "queue growth not detected at the knee; the TTFT/ITL SLO gates and "
                        "the completion fraction decided it (the growth test reads TTFT, so "
                        "overload that shows only as KV-cache preemption - longer end-to-end "
                        "times, not later first tokens - is left to those gates)")
                final_result["p99_ttft_ms"] = best.get("p99_ttft_ms", 0)
                final_result["p99_tpot_ms"] = best.get("p99_tpot_ms", 0)
                final_result["p99_itl_ms"] = best.get("p99_itl_ms", 0)   # client inter-chunk
                for k in ("p99_itl_per_token_ms", "tpot_source", "server_itl_kind",
                          "spec_decode_active", "itl_client_server_disagree", "itl_gated_ms",
                          "itl_gated_limit_ms", "tokens_per_engine_step"):
                    final_result[k] = best.get(k)
                # Server-side P99 ITL (from /metrics) and which source gated the knee's
                # ITL SLO (itl_gated_ms / itl_gated_limit_ms hold the deciding value).
                final_result["p99_itl_server_ms"] = best.get("p99_itl_server_ms")
                final_result["p99_itl_server_bucket"] = best.get("p99_itl_server_bucket")
                final_result["itl_slo_source"] = best.get("itl_slo_source")
                final_result["mean_e2e_ms"] = best.get("mean_e2e_ms", 0)
                final_result["slo_attainment"] = best.get("slo_attainment", 0)
                final_result["request_throughput"] = best.get("request_throughput", 0)
                # Little's Law: concurrent-users capacity ≈ arrival_rate * mean_e2e
                # at the knee (the closed-loop equivalent of the open-loop knee).
                mean_e2e_s = float(best.get("mean_e2e_ms", 0.0)) / 1000.0
                final_result["little_law_n_users"] = (
                    max_qps * mean_e2e_s if (mean_e2e_s > 0 and max_qps) else 0.0)

            # Save result
            self._save_stress_result(final_result, model, format, multimodal)

            return final_result

        except StressContaminated as e:
            # Other traffic: no headline, never a partial result; the measured points
            # stay in the JSON for inspection.
            logger.error(f"Stress test ended: {e}. No limit is reported; re-run with the "
                         f"server to yourself.")
            res = {"stress_test": True, "failed": True, "failure_kind": "contaminated",
                   "status": "contaminated", "error": str(e), "multimodal": multimodal,
                   "model": model.short_name, "format": format.value,
                   "max_sustainable_qps": None,
                   "point_log": list(self._point_log),
                   "points_measured": sorted(self._tested_rates.values(),
                                             key=lambda r: r["request_rate_qps"])}
            try:
                self._save_stress_result(res, model, format, multimodal)
            except Exception:                                             # noqa: BLE001
                pass
            return res
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stress_test": True,
                "failed": True,
                "failure_kind": "harness",
                "error": str(e),
                # Carry the mode so cli.py stamps benchmark_type correctly
                # (stress_test_multimodal vs stress_test). Without it a FAILED
                # MM stress is mis-stamped as a text failure, which the
                # exit-code guard would treat as fatal.
                "multimodal": multimodal,
                "model": model.short_name,
                "format": format.value,
            }

        finally:
            # Tear down the reusable client pool before the server so no worker
            # processes linger between campaigns.
            self._shutdown_client_pool()
            if self._serving_runner and not self.config.remote_endpoint:
                self._serving_runner._cleanup_server()
            # Clean up MM images
            if multimodal:
                self._cleanup_mm_images()

    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
        include_text: bool = True,
        include_multimodal: bool = True,
    ) -> list[dict]:
        """Run stress tests - text and/or multimodal per the caller's gates.

        Both use the open-loop QPS sweep; text drives vLLM's benchmark engine,
        multimodal sends image+text chat requests with Poisson arrivals.

        Args:
            include_text: run the text stress pass (gated off by --multimodal-only).
            include_multimodal: run the multimodal pass - but only for MM-capable
                models (gated off by --text-only).
        """
        results = []

        if include_text:
            logger.info("Running text stress test...")
            text_result = self.run(model, format, multimodal=False)
            results.append(text_result)

        # Image-based MM stress requires VISION -> gate on supports_multimodal
        # (has_vision), not category==MULTIMODAL (which is vision OR audio).
        if include_multimodal and model.supports_multimodal:
            logger.info("Running multimodal stress test...")
            mm_result = self.run(model, format, multimodal=True)
            results.append(mm_result)

        return results
