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

"""The perf test stubs themselves behave as specified (so later gate tests can trust
them): simulators keep up / saturate where theory says, the HTTP stub streams the
chunks, usage and /metrics it promises."""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_http_stub import StubServer  # noqa: E402
from perf_sim import (  # noqa: E402
    BatchingServer, CapacityServer, Latency, Shape, UnlimitedServer, poisson_sends, run,
)

from gbench.analysis.perf_method import hist_delta, parse_metrics  # noqa: E402

SEEDS = range(200)


def test_poisson_sends_have_the_target_rate_and_no_send_at_t0():
    rates = []
    for seed in SEEDS:
        s = poisson_sends(10.0, 600, np.random.default_rng(seed))
        assert s[0] > 0.0 and all(b > a for a, b in zip(s, s[1:]))
        rates.append(len(s) / s[-1])
    # mean realized rate within 1% of target over 200 seeds (each ~ +-4%)
    assert abs(np.mean(rates) - 10.0) < 0.1


def test_unlimited_server_latency_does_not_grow_with_send_time():
    srv = UnlimitedServer(Latency(median_s=3.0, sigma=0.6, tail_p=0.02))
    slopes = []
    for seed in SEEDS:
        recs = run(srv, 50.0, 3000, seed)
        x = np.array([r["send"] for r in recs])
        y = np.array([r["done"] - r["send"] for r in recs])
        slopes.append(np.polyfit(x, y, 1)[0])
    assert abs(np.mean(slopes)) < 2e-3                       # no trend on average


def test_capacity_server_converges_to_slots_over_service_time():
    srv = CapacityServer(slots=4, decode=Latency(median_s=1.0, sigma=0.4))
    cap = srv.capacity_qps()
    for seed in range(20):
        recs = run(srv, 3.0 * cap, 4000, seed)                # heavily overloaded
        done = sorted(r["done"] for r in recs)
        mid = done[len(done) // 4: 3 * len(done) // 4]
        achieved = (len(mid) - 1) / (mid[-1] - mid[0])
        assert abs(achieved - cap) / cap < 0.06               # C / mean(S), not C


def test_capacity_server_queue_grows_above_capacity():
    srv = CapacityServer(slots=1, decode=Latency(median_s=0.5, sigma=0.3))
    recs = run(srv, 1.25 * srv.capacity_qps(), 600, 1)
    early = np.mean([r["ttft"] for r in recs[:100]])
    late = np.mean([r["ttft"] for r in recs[-100:]])
    assert late > 5 * early


def _achieved(recs):
    done = sorted(r["done"] for r in recs if r["ok"])
    mid = done[len(done) // 4: 3 * len(done) // 4]
    return (len(mid) - 1) / (mid[-1] - mid[0])


def test_batching_server_latency_rises_with_load_below_capacity():
    """Continuous batching: more running requests make every step slower, so e2e grows
    smoothly with load well below capacity (unlike a fixed-slot server)."""
    srv = BatchingServer(max_running=64)
    cap = srv.capacity_qps()
    e2e = []
    for rho in (0.2, 0.5, 0.8):
        recs = run(srv, rho * cap, 600, 3)
        e2e.append(np.median([r["done"] - r["send"] for r in recs if r["ok"]]))
    assert e2e[0] < e2e[1] < e2e[2]
    assert e2e[2] > 1.5 * e2e[0]


@pytest.mark.parametrize("m", [1, 4, 64])
def test_batching_server_saturates_at_its_capacity(m):
    srv = BatchingServer(max_running=m, shape=Shape(out_tokens=32))
    cap = srv.capacity_qps()
    recs = run(srv, 3.0 * cap, 1500, 5)
    assert abs(_achieved(recs) - cap) / cap < 0.1


def test_ttft_is_prefill_plus_wait_not_a_share_of_decode():
    """SIM-2: a 90 s decode has a sub-second TTFT when nothing queues."""
    srv = UnlimitedServer(Latency(90.0, 0.3), Shape(prompt_tokens=128, out_tokens=2048))
    recs = run(srv, 0.2, 40, 1)
    assert max(r["ttft"] for r in recs) < 100.0              # 128 / 8000 tok/s = 16 ms


def test_failure_modes():
    sh = Shape(max_point_s=30.0, fail_p=0.1, honor_ignore_eos=False)
    srv = CapacityServer(1, Latency(1.0, 0.3), sh)
    recs = run(srv, 2.0 * srv.capacity_qps(), 200, 2)         # overloaded: queue -> cutoff
    assert any(r["done"] is None and not r["ok"] for r in recs)          # cut off
    assert any(not r["ok"] and r["done"] is not None for r in recs)      # failed
    oks = [r for r in recs if r["ok"]]
    assert any(r["finish_reason"] == "stop" and r["completion_tokens"] < 128 for r in oks)


# ── HTTP stub (real sockets; small, fast) ─────────────────────────────────────

def _post_stream(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks, usage = [], None
    with urllib.request.urlopen(req, timeout=10) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or "[DONE]" in line:
                continue
            obj = json.loads(line[5:])
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                chunks.append(ch.get("text") or (ch.get("delta") or {}).get("content") or "")
    return chunks, usage


def test_http_stub_streams_multi_token_chunks_with_usage_and_metrics():
    stub = StubServer(prefill_ms=2, per_token_ms=1, chunk_tokens=4)
    base = stub.start()
    try:
        before = parse_metrics(urllib.request.urlopen(base[:-3] + "/metrics").read().decode())
        chunks, usage = _post_stream(base + "/completions", {
            "model": "stub", "prompt": "x", "max_tokens": 16, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}})
        after = parse_metrics(urllib.request.urlopen(base[:-3] + "/metrics").read().decode())
    finally:
        stub.stop()
    # first token alone, then 4-token chunks: 1 + 4 + 4 + 4 + 3
    assert [c.count("tok") for c in chunks] == [1, 4, 4, 4, 3]
    assert usage["completion_tokens"] == 16
    assert after["request_success"] - before["request_success"] == 1
    assert after["generation_tokens"] - before["generation_tokens"] == 16
    d = hist_delta(before["itl"], after["itl"])
    assert d is not None and d[1] == 15                      # one gap per 1-token step


def test_http_stub_non_vllm_mode_has_no_metrics_or_usage():
    stub = StubServer(vllm=False)
    base = stub.start()
    try:
        chunks, usage = _post_stream(base + "/chat/completions", {
            "model": "stub", "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 4, "stream": True, "stream_options": {"include_usage": True}})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base[:-3] + "/metrics")
    finally:
        stub.stop()
    assert chunks and usage is None


def test_http_stub_foreign_requests_show_up_in_the_counter():
    stub = StubServer()
    base = stub.start()
    try:
        m0 = parse_metrics(urllib.request.urlopen(base[:-3] + "/metrics").read().decode())
        urllib.request.urlopen(urllib.request.Request(base[:-3] + "/admin/foreign?n=20", data=b""))
        m1 = parse_metrics(urllib.request.urlopen(base[:-3] + "/metrics").read().decode())
    finally:
        stub.stop()
    assert m1["request_success"] - m0["request_success"] == 20


class _CharTok:
    """4 characters per token ('tok ' is one token): exact counts for the stub's text."""
    vocab_size = 256
    all_special_ids = [0]

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return type("E", (), {"input_ids": [[0] * (len(t) // 4) for t in text]})()
        return type("E", (), {"input_ids": [ord(c) % 256 for c in text]})()

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.mark.slow
@pytest.mark.parametrize("spec", [False, True])
def test_stress_point_end_to_end_against_the_stub(monkeypatch, spec):
    """Real client worker processes, real sockets: one stress point against a stub that
    streams 4-token chunks. Per-token client ITL must be ~1/4 of the inter-chunk gap, the
    server histogram must gate the ITL, and every request must be unique and uncached."""
    from gbench.core.config import BenchmarkConfig
    from gbench.runners.serving import ServingBenchmarkRunner
    from gbench.runners.stress import StressTestRunner

    stub = StubServer(prefill_ms=5, per_token_ms=2, chunk_tokens=4, spec_decode=spec)
    base = stub.start()
    r = None
    try:
        cfg = BenchmarkConfig()
        cfg.remote_endpoint = base
        r = StressTestRunner(cfg, tpot_threshold_ms=200)
        r._serving_runner = ServingBenchmarkRunner(cfg)
        r._api_model_id = "stub"
        r._tokenizer = _CharTok()
        r._recount_ok, r._recount_for = True, id(r._tokenizer)
        # 33 = 1 + 8 x 4 tokens: every step after the first carries exactly 4 under spec
        monkeypatch.setattr(r, "_workload_shape", lambda: ("random", 64, 33))
        # 300 requests at 20 req/s: a 15 s schedule, above the 10 s minimum window
        m = r._measure_rate_point(None, None, qps=20.0, num_prompts=300)
    finally:
        if r is not None:
            r._shutdown_client_pool()
        stub.stop()
    assert m["completed"] == 300 and m["failed"] == 0
    assert m["flags"] == [] and m["planned_n"] == 300 and m["sent_n"] == 300
    assert m["client_bound"] is False, m.get("client_bound_reasons")
    assert m["saturation"] in ("NOT_DETECTED",)               # unlimited stub keeps up
    assert m["completion_fraction"] == 1.0
    assert m["num_client_procs"] >= 2                         # real worker processes
    assert stub.requests_seen == 300
    assert m["p99_itl_per_token_ms"] < m["p99_itl_ms"] / 2       # 4 tokens per chunk
    assert m["itl_gate_outcome"] == "PASS"
    assert m["tpot_source"] == "per_token_usage"
    assert m["prefix_cache_hits"] == 0
    assert m["spec_decode_active"] is (True if spec else None)   # no counter = not observable
    if spec:
        # 4-token engine steps: the per-step limit is the per-token SLO x 4
        assert m["tokens_per_engine_step"] == pytest.approx(4.0, rel=0.02)
        assert m["itl_gate_limit_per_step_ms"] == pytest.approx(800.0, rel=0.02)
    else:
        assert m["tokens_per_engine_step"] == pytest.approx(1.0, rel=0.02)
        # one token per step: the client per-token P99 lies within the server's P99 bucket
        lo = float(m["p99_itl_server_bucket"].strip("(]").split(",")[0])
        assert lo * 0.8 <= m["p99_itl_per_token_ms"] <= m["p99_itl_server_ms"] * 1.25


@pytest.mark.slow
def test_remote_text_serving_reaches_batch_64(monkeypatch):
    """Issue 9: remote text serving is not capped at the 32-thread default pool."""
    import asyncio

    from gbench.core.config import BenchmarkConfig
    from gbench.runners.serving import ServingBenchmarkRunner
    stub = StubServer(prefill_ms=200, per_token_ms=5, vllm=False)
    base = stub.start()
    try:
        cfg = BenchmarkConfig()
        cfg.remote_endpoint = base
        r = ServingBenchmarkRunner(cfg)
        reqs = [type("Q", (), {"prompt": f"p{i}", "expected_output_len": 8})() for i in range(64)]
        m = asyncio.run(r._run_lightweight_remote_benchmark(base, "stub", reqs, batch_size=64))
    finally:
        stub.stop()
    assert m["completed"] == 64
    assert m["peak_in_flight"] == 64
    assert m["client_loop_lag_max_ms"] < 1000
