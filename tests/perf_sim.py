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

"""Deterministic server simulators for the stress pillar's statistics (test helper).

Each simulator turns absolute send times into request records shaped like the stress
workers' ({send, done, ttft, ok, tpot, itls, completion_tokens, finish_reason}), so the
schedule, the cohort selection and the saturation gate can be tested over many seeds in
milliseconds, with no server and no clock.

Servers:
  UnlimitedServer   every request starts on arrival; decode time ~ lognormal (+ heavy
                    tail). It keeps up at any rate, so the gate must PASS it.
  CapacityServer    C parallel slots, FIFO queue, decode time ~ lognormal: capacity is
                    C / mean(service). Latency is flat below capacity.
  BatchingServer    continuous batching like vLLM: one engine step at a time; a step
                    takes a + b x (decoding requests) + c x (prefill tokens this step),
                    at most M running requests, a chunked-prefill token budget per step.
                    Latency rises smoothly with load and settles slowly near capacity.

All servers share the request shape (prompt and output tokens, prefill rate) and the
failure modes of a real run: a point cutoff (requests still running at max_point_s never
finish: ok False, done None), random failures, and servers that ignore ignore_eos
(short replies with finish_reason "stop").
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class Latency:
    """Lognormal duration (seconds) with median ``median_s`` and log-sd ``sigma``; with
    probability ``tail_p`` a request is ``tail_mult`` times slower (heavy tail)."""
    median_s: float
    sigma: float = 0.3
    tail_p: float = 0.0
    tail_mult: float = 10.0

    def mean_s(self) -> float:
        base = self.median_s * math.exp(self.sigma ** 2 / 2.0)
        return base * (1.0 - self.tail_p + self.tail_p * self.tail_mult)

    def p90_s(self) -> float:
        return self.median_s * math.exp(1.2816 * self.sigma)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        x = self.median_s * np.exp(self.sigma * rng.standard_normal(n))
        if self.tail_p > 0:
            x = np.where(rng.random(n) < self.tail_p, x * self.tail_mult, x)
        return x


@dataclass
class Shape:
    """Request shape and run-level failure modes shared by every simulator."""
    prompt_tokens: int = 128
    out_tokens: int = 128
    prefill_tps: float = 8000.0           # unloaded prefill rate (tokens/s)
    max_point_s: Optional[float] = None   # requests not done by send0 + this never finish
    fail_p: float = 0.0                   # independent request failures
    honor_ignore_eos: bool = True         # False: replies stop early (finish 'stop')
    itl_samples: int = 8                  # inter-token gaps kept per record


def _finish(recs: List[dict], shape: Shape, rng: np.random.Generator, t_start: float) -> List[dict]:
    for r in recs:
        if shape.fail_p and rng.random() < shape.fail_p:
            r.update(ok=False, done=r["done"], ttft=None)
        if shape.max_point_s is not None and r["done"] is not None \
                and r["done"] > t_start + shape.max_point_s:
            r.update(ok=False, done=None)
    return recs


def _record(send: float, ttft_s: float, done: float, gaps_s, shape: Shape,
            n_out: int) -> dict:
    gaps_ms = [g * 1000.0 for g in gaps_s]
    tpot = ((done - send - ttft_s) / max(1, n_out - 1)) * 1000.0
    keep = gaps_ms if len(gaps_ms) <= shape.itl_samples else \
        [gaps_ms[int(k * len(gaps_ms) / shape.itl_samples)] for k in range(shape.itl_samples)]
    return {"send": send, "done": done, "ttft": ttft_s * 1000.0, "ok": True, "tpot": tpot,
            "itls": keep, "completion_tokens": n_out, "prompt_tokens": shape.prompt_tokens,
            "finish_reason": "length" if n_out == shape.out_tokens else "stop"}


def _n_out(shape: Shape, rng: np.random.Generator) -> int:
    if shape.honor_ignore_eos:
        return shape.out_tokens
    return int(max(1, rng.integers(1, shape.out_tokens + 1)))


class UnlimitedServer:
    """Starts every request on arrival; decode time ~ ``decode`` (seconds)."""

    def __init__(self, decode: Latency, shape: Optional[Shape] = None):
        self.decode, self.shape = decode, shape or Shape()

    def p90_latency_s(self) -> float:
        return self.shape.prompt_tokens / self.shape.prefill_tps + self.decode.p90_s()

    def serve(self, sends: List[float], rng: np.random.Generator) -> List[dict]:
        sh = self.shape
        dec = self.decode.sample(rng, len(sends))
        out = []
        for s, d in zip(sends, dec):
            n = _n_out(sh, rng)
            d = float(d) * n / sh.out_tokens
            ttft = sh.prompt_tokens / sh.prefill_tps
            gap = d / max(1, n - 1)
            out.append(_record(s, ttft, s + ttft + d, [gap] * max(0, n - 1), sh, n))
        return _finish(out, sh, rng, sends[0] if sends else 0.0)


class CapacityServer:
    """C parallel slots, FIFO; each request holds a slot for prefill + decode."""

    def __init__(self, slots: int, decode: Latency, shape: Optional[Shape] = None):
        self.slots, self.decode, self.shape = int(slots), decode, shape or Shape()

    def service_mean_s(self) -> float:
        return self.shape.prompt_tokens / self.shape.prefill_tps + self.decode.mean_s()

    def capacity_qps(self) -> float:
        return self.slots / self.service_mean_s()

    def p90_latency_s(self) -> float:
        return self.shape.prompt_tokens / self.shape.prefill_tps + self.decode.p90_s()

    def serve(self, sends: List[float], rng: np.random.Generator) -> List[dict]:
        sh = self.shape
        dec = self.decode.sample(rng, len(sends))
        free = [0.0] * self.slots
        heapq.heapify(free)
        out = []
        for s, d in sorted(zip(sends, dec)):
            n = _n_out(sh, rng)
            d = float(d) * n / sh.out_tokens
            start = max(s, heapq.heappop(free))
            pre = sh.prompt_tokens / sh.prefill_tps
            heapq.heappush(free, start + pre + d)
            gap = d / max(1, n - 1)
            out.append(_record(s, (start - s) + pre, start + pre + d, [gap] * max(0, n - 1), sh, n))
        return _finish(out, sh, rng, sends[0] if sends else 0.0)


@dataclass
class BatchingServer:
    """Continuous batching: one engine step at a time. A step takes
    a + b * (decoding requests) + c * (prefill tokens processed in it); every decoding
    request emits one token per step; at most ``max_running`` requests are admitted;
    prefill is chunked with ``prefill_budget`` tokens per step, oldest first."""
    a: float = 0.010                       # fixed step cost (s)
    b: float = 0.0002                      # per decoding request (s)
    c: float = 0.00002                     # per prefill token (s)
    max_running: int = 256
    prefill_budget: int = 2048
    shape: Shape = field(default_factory=Shape)
    # KV-cache budget in tokens (None = unlimited). When the running requests' KV
    # (prompt + generated tokens) exceeds it, the newest running request is preempted
    # by recompute (vLLM's default): it goes back to the front of the queue and must
    # re-prefill prompt + generated tokens before decoding again. Overload then shows as
    # growing end-to-end latency and in-flight count, not as TTFT growth.
    kv_budget: Optional[int] = None

    def step_time(self, n_decode: int, prefill_tokens: int) -> float:
        return self.a + self.b * n_decode + self.c * prefill_tokens

    def capacity_qps(self) -> float:
        """Saturated request throughput: a full batch of max_running requests, each
        needing out_tokens decode steps plus its share of prefill."""
        m, sh = self.max_running, self.shape
        per_step = self.step_time(m, 0)
        prefill_s = self.c * sh.prompt_tokens        # prefill work per request
        return m / (sh.out_tokens * per_step + m * prefill_s)

    def p90_latency_s(self) -> float:
        sh = self.shape
        return sh.out_tokens * self.step_time(1, 0) + self.c * sh.prompt_tokens + self.a

    def serve(self, sends: List[float], rng: np.random.Generator) -> List[dict]:
        sh = self.shape
        sends = sorted(sends)
        n = len(sends)
        n_out = [_n_out(sh, rng) for _ in range(n)]
        prefill_left = [sh.prompt_tokens] * n
        emitted = [0] * n
        first = [None] * n
        last_tok = [None] * n
        gaps: List[List[float]] = [[] for _ in range(n)]
        done = [None] * n
        waiting: List[int] = []            # arrived, not admitted (FIFO)
        prefilling: List[int] = []         # admitted, prefill not finished (oldest first)
        decoding: List[int] = []
        nxt = 0
        t = sends[0] if n else 0.0
        cutoff = (sends[0] + sh.max_point_s) if (n and sh.max_point_s) else math.inf
        while nxt < n or waiting or prefilling or decoding:
            while nxt < n and sends[nxt] <= t:
                waiting.append(nxt)
                nxt += 1
            while waiting and len(prefilling) + len(decoding) < self.max_running:
                prefilling.append(waiting.pop(0))
            if not prefilling and not decoding:
                t = sends[nxt]             # idle until the next arrival
                continue
            if self.kv_budget is not None:
                def _kv(i):
                    return sh.prompt_tokens - prefill_left[i] + max(0, emitted[i] - 1)
                while decoding and sum(_kv(i) for i in decoding + prefilling) > self.kv_budget:
                    v = decoding.pop()                       # newest decoding request
                    prefill_left[v] = sh.prompt_tokens + emitted[v]   # recompute
                    waiting.insert(0, v)
                    self.preemptions = getattr(self, "preemptions", 0) + 1
            budget, used, finished_prefill = self.prefill_budget, 0, []
            for i in prefilling:
                if budget <= 0:
                    break
                take = min(budget, prefill_left[i])
                prefill_left[i] -= take
                budget -= take
                used += take
                if prefill_left[i] == 0:
                    finished_prefill.append(i)
            dt = self.step_time(len(decoding), used)
            t += dt
            if t > cutoff:
                break
            for i in decoding:
                emitted[i] += 1
                gaps[i].append(t - last_tok[i])
                last_tok[i] = t
            still = []
            for i in decoding:
                if emitted[i] >= n_out[i]:
                    done[i] = t
                else:
                    still.append(i)
            decoding = still
            for i in finished_prefill:     # first token at the end of the prefill step
                prefilling.remove(i)
                if first[i] is None:
                    first[i] = last_tok[i] = t
                    emitted[i] = 1
                if emitted[i] >= n_out[i]:
                    done[i] = t
                else:
                    decoding.append(i)        # (a resumed request keeps its tokens)
        out = []
        for i, s in enumerate(sends):
            if done[i] is None:
                out.append({"send": s, "done": None, "ttft": ((first[i] - s) * 1000.0
                            if first[i] is not None else None), "ok": False, "tpot": None,
                            "itls": None, "completion_tokens": None,
                            "prompt_tokens": sh.prompt_tokens, "finish_reason": None})
            else:
                out.append(_record(s, first[i] - s, done[i], gaps[i], sh, n_out[i]))
        return _finish(out, sh, rng, sends[0] if n else 0.0)


def poisson_sends(rate: float, n: int, rng: np.random.Generator, t0: float = 0.0) -> List[float]:
    """n Poisson arrival times at ``rate`` req/s after t0 (no send AT t0)."""
    return list(t0 + np.cumsum(rng.exponential(1.0 / rate, n)))


def run(server, rate: float, n: int, seed: int,
        schedule: Optional[Callable] = None) -> List[dict]:
    """Records of one simulated point: n arrivals at ``rate`` (Poisson unless a schedule
    function (rate, n, rng) -> sends is given), served by ``server``."""
    rng = np.random.default_rng(seed)
    sends = (schedule or poisson_sends)(rate, n, rng)
    return server.serve(sends, rng)
