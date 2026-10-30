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

"""Stress runner regression tests."""

import json
import re
import urllib.request

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.runners.stress import StressTestRunner, _delta_piece
from gbench.runners.serving import ServingBenchmarkRunner


def test_delta_piece_counts_reasoning_tokens():
    """A reasoning model (e.g. gemma via Ollama) streams its reply as
    reasoning/reasoning_content deltas. The stress SSE parser must count those as
    real decode - counting only `content` made every request look empty and the
    whole sweep skip with 'no usable stream'."""
    # chat deltas
    assert _delta_piece({"delta": {"content": "hi "}}, "delta") == "hi "
    assert _delta_piece({"delta": {"reasoning": "think "}}, "delta") == "think "
    assert _delta_piece({"delta": {"reasoning_content": "rc "}}, "delta") == "rc "
    assert _delta_piece({"delta": {}}, "delta") == ""            # empty -> not counted
    assert _delta_piece({"delta": {"role": "assistant"}}, "delta") == ""
    # legacy /v1/completions text field
    assert _delta_piece({"text": "tok "}, "text") == "tok "
    assert _delta_piece({}, "text") == ""


def test_client_pool_multiproc_is_not_dead_code():
    """`self._mp_ctx` used to be initialized AFTER a `return` in _images_per_request()
    (unreachable), so `_get_client_pool()` raised AttributeError and every rate point
    silently fell back to a single in-process client while the artifact still reported
    `num_client_procs=8` / method=...multiproc / "contention-free". This asserts the
    context is set at construction and the pool spawns real worker processes."""
    r = StressTestRunner(BenchmarkConfig())
    assert r._mp_ctx is not None                       # was never set (the bug)
    assert r._mp_ctx.get_start_method() == "spawn"
    r._num_client_procs = 2                             # keep the spawn test light
    pool = r._get_client_pool()
    try:
        # Workers actually spawn and execute (would raise on the dead-code bug).
        assert list(pool.map(abs, [-5, -6])) == [5, 6]
    finally:
        r._shutdown_client_pool()
    assert r._client_pool is None


def _text_url(remote_endpoint, local_port=8000, monkeypatch=None):
    """Build the stress TEXT api_url for a given endpoint via the real code path."""
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = remote_endpoint
    r = StressTestRunner(cfg)
    r._api_model_id = "m"
    r._multimodal = False
    if remote_endpoint is None:
        r._serving_runner = type("S", (), {"server_port": local_port})()
    # Stub the heavy request generation; we only care about the URL.
    r._generate_requests = lambda n, model=None, format=None: [type("R", (), {"prompt": "hi"})()]
    r._tokenizer = _CharTok()          # prompts are sized with a tokenizer (required)
    api_url, _field, _payloads = r._build_payloads(model=None, format=None, num_prompts=1)
    return api_url


def _mm_url(remote_endpoint):
    """Build the stress MULTIMODAL api_url via the real code path (num_prompts=0 skips image IO)."""
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = remote_endpoint
    r = StressTestRunner(cfg)
    r._api_model_id = "m"
    r._multimodal = True
    api_url, _field, _payloads = r._build_payloads(model=None, format=None, num_prompts=0)
    return api_url


def test_remote_endpoint_with_v1_is_not_doubled():
    """--remote-endpoint <URL>/v1 (the documented convention) must not become /v1/v1/...

    Regression: the stress runner blindly appended `/v1/completions` to a base_url that
    already ended in `/v1`, so every request 404'd (completed=0/failed=N at every rate,
    goodput 0.00 req/s) while serving - which normalizes the same endpoint - worked fine.
    """
    assert _text_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/completions"
    assert _mm_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/chat/completions"
    # Trailing slash on the /v1 base must also normalize.
    assert _text_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1/completions"


def test_remote_endpoint_without_v1_gets_v1_appended():
    assert _text_url("http://host:9000") == "http://host:9000/v1/completions"
    assert _mm_url("http://host:9000") == "http://host:9000/v1/chat/completions"


def test_local_serving_url_gets_v1_appended():
    assert _text_url(None, local_port=8123) == "http://127.0.0.1:8123/v1/completions"


def test_stress_max_qps_and_prompts_default_env_flag_precedence(monkeypatch):
    """The default safety cap (8192 QPS) + prompt cap (8000) are high enough that a fast
    multi-GPU server finds its real knee out of the box, and stay overridable. Precedence:
    config flag > GBENCH_STRESS_* env > class default."""
    monkeypatch.delenv("GBENCH_STRESS_MAX_QPS", raising=False)
    monkeypatch.delenv("GBENCH_STRESS_MAX_PROMPTS", raising=False)
    r = StressTestRunner(BenchmarkConfig())
    assert r.max_qps == StressTestRunner.MAX_QPS == 8192.0
    assert r.max_prompts == StressTestRunner.MAX_PROMPTS == 8000
    # Client-proc ceiling raised in lockstep so the load generator can actually OFFER 8192 QPS on a
    # big host (the runtime value still auto-caps to cpu_count-1). Assert the class constant, which
    # is machine-independent (the computed _num_client_procs depends on the CI host's core count).
    assert StressTestRunner.NUM_CLIENT_PROCS == 32

    monkeypatch.setenv("GBENCH_STRESS_MAX_QPS", "4096")
    monkeypatch.setenv("GBENCH_STRESS_MAX_PROMPTS", "8000")
    r_env = StressTestRunner(BenchmarkConfig())
    assert r_env.max_qps == 4096.0
    assert r_env.max_prompts == 8000

    cfg = BenchmarkConfig()
    cfg.stress_max_qps = 2048
    cfg.stress_max_prompts = 5000
    r_flag = StressTestRunner(cfg)          # flag set -> wins over the env above
    assert r_flag.max_qps == 2048.0
    assert r_flag.max_prompts == 5000


def test_stress_client_procs_and_reps_flag_env_precedence(monkeypatch):
    """--stress-client-procs / --stress-reps are uniform with the new knobs:
    flag > GBENCH_STRESS_* env > default (client procs stay auto-capped to cpu_count-1)."""
    monkeypatch.setattr("gbench.runners.stress.os.cpu_count", lambda: 8)
    monkeypatch.delenv("GBENCH_STRESS_CLIENT_PROCS", raising=False)
    monkeypatch.delenv("GBENCH_STRESS_REPS", raising=False)
    r = StressTestRunner(BenchmarkConfig())
    assert r._reps == StressTestRunner.STRESS_REPS == 3

    # small values so the cpu_count-1 cap never masks the assertion
    monkeypatch.setenv("GBENCH_STRESS_CLIENT_PROCS", "2")
    monkeypatch.setenv("GBENCH_STRESS_REPS", "5")
    r_env = StressTestRunner(BenchmarkConfig())
    assert r_env._num_client_procs == 2
    assert r_env._reps == 5

    cfg = BenchmarkConfig()
    cfg.stress_client_procs = 3
    cfg.stress_reps = 7
    r_flag = StressTestRunner(cfg)          # flag wins over the env above
    assert r_flag._num_client_procs == 3
    assert r_flag._reps == 7


def test_mm_effective_input_len_tracks_probed_vision_tokens():
    """The reported MM effective input length must track the per-image token count
    actually charged by the server, not a model family's constant: a processor setting
    (e.g. max_soft_tokens=1120) makes the server charge ~4x a 280 default, which the old
    constant under-reported ~3.7x (128+4*280=1248 vs 128+4*1120=4608) and mis-sized the
    padded-text budget. With no count known at all, sizing refuses instead of guessing."""
    cfg = BenchmarkConfig()
    cfg.max_model_len = 262144   # large context so the text budget is not clamped
    r = StressTestRunner(cfg)
    assert r._images_per_request() == 4
    assert not hasattr(r, "PER_IMAGE_SOFT_TOKENS")
    with pytest.raises(RuntimeError, match="per-image token count is unknown"):
        r._mm_effective_input_len(128, 128)
    # A declared count of 280: 128 text + 4*280 images.
    r._vision_tokens_per_image = 280
    assert r._mm_effective_input_len(128, 128) == 128 + 4 * 280 == 1248
    # Server-measured (gemma-4 deployed, 1120): 128 text + 4*1120 images.
    r._vision_tokens_per_image = 1120
    assert r._mm_effective_input_len(128, 128) == 128 + 4 * 1120 == 4608


def test_mm_text_budget_clamps_when_images_fill_context():
    """At the uniform 4096 context, 4 images at 1120 tok/img (4480) already exceed
    the window, so the padded-text budget must clamp to the MIN floor rather than go
    negative. This only surfaces once the true (probed) per-image count is used; with
    the old 280 the images fit and the clamp never engaged (masking the overflow)."""
    cfg = BenchmarkConfig()          # default max_model_len -> uniform 4096
    r = StressTestRunner(cfg)
    r._vision_tokens_per_image = 1120
    # 4*1120=4480 > 4096 -> text clamps to MIN_MM_TEXT_TOKENS, not a negative value.
    assert r._mm_text_tokens(128, 128) == r.MIN_MM_TEXT_TOKENS
    assert r._mm_effective_input_len(128, 128) == r.MIN_MM_TEXT_TOKENS + 4 * 1120
    # With the old 280 default the images fit, so text stays at the full 128.
    r._vision_tokens_per_image = 280
    assert r._mm_text_tokens(128, 128) == 128


def test_mm_preflight_returns_server_measured_per_image(monkeypatch):
    """_assert_mm_images_processed returns the SERVER-MEASURED per-image soft-token
    gain = (img prompt_tokens - text prompt_tokens) / n_img, so the stress runner can
    size + report the MM workload against what the server actually charges (gemma-4:
    1120 via --hf-overrides) instead of the 280 default. Robust to any backend."""
    cfg = BenchmarkConfig()          # remote_endpoint None -> file:// image URLs, no file IO
    r = ServingBenchmarkRunner(cfg)
    r._mm_image_paths = ["/tmp/probe_never_opened.jpg"]   # non-empty so the probe runs
    r._images_per_request = lambda: 4

    class _Resp:
        def __init__(self, ptok):
            self._b = json.dumps({"usage": {"prompt_tokens": ptok}}).encode("utf-8")
        def read(self):
            return self._b
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    # text-only = 10 tok; +4 images = 10 + 4*1120 -> per_img = 1120
    vals = iter([10, 10 + 4 * 1120])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(next(vals)))
    got = r._assert_mm_images_processed("m", "http://127.0.0.1:8000", per_img_tokens=280)
    assert got == 1120

    # Skipped-probe path (no images) returns None, so callers keep the default.
    r._mm_image_paths = []
    assert r._assert_mm_images_processed("m", "http://127.0.0.1:8000") is None


def test_hist_delta_p99_ms_from_histogram_delta():
    """P99 of the inter-token-latency of requests that ran between two cumulative
    histogram snapshots; None when a snapshot is missing or the window is empty."""
    before = ({0.01: 0.0, 0.02: 0.0, float("inf"): 0.0}, 0.0)
    after = ({0.01: 90.0, 0.02: 99.0, float("inf"): 100.0}, 100.0)
    assert StressTestRunner._hist_delta_p99_ms(before, after) == 20.0   # le=0.02 -> 20ms
    assert StressTestRunner._hist_delta_p99_ms(None, after) is None
    assert StressTestRunner._hist_delta_p99_ms(before, before) is None  # 0 tokens in window


def test_metrics_url_strips_v1():
    """/metrics lives at the server ROOT; a /v1 endpoint must not become /v1/metrics."""
    cfg = BenchmarkConfig(); cfg.remote_endpoint = "http://127.0.0.1:8000/v1"
    assert StressTestRunner(cfg)._metrics_url() == "http://127.0.0.1:8000/metrics"
    cfg2 = BenchmarkConfig(); cfg2.remote_endpoint = "http://host:9000"
    assert StressTestRunner(cfg2)._metrics_url() == "http://host:9000/metrics"
    r = StressTestRunner(BenchmarkConfig())              # local serve, no endpoint
    r._serving_runner = type("S", (), {"server_port": 8123})()
    assert r._metrics_url() == "http://127.0.0.1:8123/metrics"


def test_server_itl_snapshot_parses_inter_token_latency(monkeypatch):
    """Parses vllm:inter_token_latency_seconds buckets/count summed across DP engines."""
    metrics = "\n".join([
        'vllm:inter_token_latency_seconds_bucket{engine="0",le="0.01"} 40.0',
        'vllm:inter_token_latency_seconds_bucket{engine="1",le="0.01"} 50.0',
        'vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf"} 45.0',
        'vllm:inter_token_latency_seconds_bucket{engine="1",le="+Inf"} 55.0',
        'vllm:inter_token_latency_seconds_count{engine="0"} 45.0',
        'vllm:inter_token_latency_seconds_count{engine="1"} 55.0',
    ])
    class _R:
        def read(self): return metrics.encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    cfg = BenchmarkConfig(); cfg.remote_endpoint = "http://127.0.0.1:8000/v1"
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: _R())
    snap = StressTestRunner(cfg)._server_itl_snapshot()
    assert snap is not None
    buckets, count = snap
    assert count == 100.0
    assert buckets[0.01] == 90.0 and buckets[float("inf")] == 100.0


def _rate_metrics(**over):
    base = dict(achieved_qps=3.0, arrival_qps=3.0, slo_attainment=1.0,
                p99_ttft_ms=60.0, p99_tpot_ms=8.0, p99_itl_ms=800.0,
                p99_itl_server_ms=25.0, mean_e2e_ms=1000.0, steady_n=500,
                completed=200, failed=0, client_bound=False,
                actual_send_qps=3.0, client_sched_lag_s=0.0, num_client_procs=8)
    base.update(over); return base


def test_knee_gates_on_server_itl_not_client_confound(monkeypatch):
    """The MM confound: client P99 ITL 800ms (would trip the 200ms SLO) but the
    server's own P99 ITL is 25ms -> the rate is SUSTAINABLE. Gating on server ITL
    fixes the ~7x MM knee under-report; without /metrics it correctly falls back to
    the (failing) client ITL; a genuinely high server ITL still fails."""
    r = StressTestRunner(BenchmarkConfig())
    # (a) server ITL healthy -> PASS despite confounded client ITL
    monkeypatch.setattr(r, "_measure_rate_point",
                        lambda *a, **k: _rate_metrics(itl_gate_outcome="PASS"))
    passed, *_ = r._test_rate(3.0, None, None)
    assert passed is True
    assert r._tested_rates[3.0]["itl_slo_source"] == "server"
    # (b) no server metrics -> fall back to client ITL (800ms) -> FAIL
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point",
                        lambda *a, **k: _rate_metrics(p99_itl_server_ms=None))
    passed_b, *_ = r._test_rate(3.0, None, None)
    assert passed_b is False
    assert r._tested_rates[3.0]["itl_slo_source"] == "client_inter_chunk"
    # (c) server ITL genuinely high -> real preemption -> FAIL
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point",
                        lambda *a, **k: _rate_metrics(p99_itl_server_ms=350.0,
                                                      itl_gate_outcome="FAIL"))
    passed_c, *_ = r._test_rate(3.0, None, None)
    assert passed_c is False
    assert r._tested_rates[3.0]["itl_slo_source"] == "server"


class _StubModel:
    short_name = "stub"
    name = "stub-model"


class _StubFormat:
    value = "hf"


def _preflight_runner(monkeypatch, ttft_ms):
    """A StressTestRunner wired so _preflight_floor_check needs no server/tokenizer:
    _build_payloads returns dummy payloads, and every single-stream probe returns a
    record with the given TTFT."""
    r = StressTestRunner(BenchmarkConfig(), ttft_threshold_ms=5000, tpot_threshold_ms=200)
    monkeypatch.setattr(r, "_build_payloads",
                        lambda *a, **k: ("http://x/v1/completions", "text",
                                         [{"i": 1}, {"i": 2}, {"i": 3}]))
    monkeypatch.setattr(r, "_workload_shape", lambda: ("random", 128, 128))
    monkeypatch.setattr(
        "gbench.runners.stress._stress_client_worker",
        lambda task: {"records": [{"ok": True, "ttft": ttft_ms, "send": 0.0,
                                   "done": ttft_ms / 1000.0 + 1.0}], "sched_lag_s": 0.0})
    return r


def test_preflight_bails_below_floor(monkeypatch):
    """Single-stream TTFT above the SLO -> no open-loop rate can pass -> early-exit
    result (below_stress_floor), NOT a harness failure."""
    r = _preflight_runner(monkeypatch, ttft_ms=20000.0)   # 20s >> 5000ms SLO
    out = r._preflight_floor_check(_StubModel(), _StubFormat(), multimodal=False)
    assert out is not None
    assert out["below_stress_floor"] is True
    assert out["max_sustainable_qps"] == 0.0
    assert out["slo_met"] is False
    assert out["preflight_single_stream_ttft_ms"] == 20000.0
    assert not out.get("failed")            # NOT routed to the failure bucket / exit code
    assert out["multimodal"] is False


def test_preflight_proceeds_when_single_stream_meets_slo(monkeypatch):
    """Single-stream TTFT under the SLO -> return None so the normal sweep runs."""
    r = _preflight_runner(monkeypatch, ttft_ms=120.0)     # 120ms << 5000ms SLO
    out = r._preflight_floor_check(_StubModel(), _StubFormat(), multimodal=False)
    assert out is None


def test_preflight_bails_when_endpoint_returns_no_stream(monkeypatch):
    """If not one single-stream request produces output tokens, the request path is
    broken for this endpoint (e.g. 0 completions) -> bail fast with a diagnostic
    instead of falling through to a doomed sweep."""
    r = StressTestRunner(BenchmarkConfig(), ttft_threshold_ms=5000, tpot_threshold_ms=200)
    monkeypatch.setattr(r, "_build_payloads",
                        lambda *a, **k: ("http://x/v1/completions", "text", [{"i": 1}]))
    monkeypatch.setattr(r, "_workload_shape", lambda: ("random", 128, 128))
    monkeypatch.setattr("gbench.runners.stress._stress_client_worker",
                        lambda task: {"records": [{"ok": False, "ttft": None}], "sched_lag_s": 0.0})
    out = r._preflight_floor_check(_StubModel(), _StubFormat(), multimodal=False)
    assert out is not None
    assert out["preflight_no_stream"] is True
    assert out["max_sustainable_qps"] == 0.0
    assert not out.get("failed")


class _CharTok:
    """One token per character (vocab 0..255): exact, deterministic lengths."""
    vocab_size = 256
    all_special_ids = [0]

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [ord(c) % 256 for c in text]})()

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _SR:
    NOTHINK_SYSTEM = "Answer directly."

    def _text_prompt_parts(self, out_len):
        return "a coherent passage about water systems", " Write at length."

    def _resolve_model_id(self, *a, **k):
        return "gemma4-qat:4b"

    def is_vllm_server(self):
        return False


def _remote_runner(monkeypatch, **cfg_over):
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = "http://localhost:11434/v1"
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    r = StressTestRunner(cfg)
    r._tokenizer = _CharTok()
    r._api_model_id = "gemma4-qat:4b"
    r._serving_runner = _SR()
    monkeypatch.setattr(r, "_workload_shape", lambda: ("random", 128, 128))
    return r


def test_build_payloads_remote_uses_chat_with_coherent_prompt(monkeypatch):
    """Remote endpoints must use /v1/chat/completions + a coherent prompt (mirroring
    the serving pillar) so a backend that ignores ignore_eos (e.g. Ollama) still emits
    tokens - not the random /v1/completions prompt that draws an immediate EOS."""
    r = _remote_runner(monkeypatch)
    url, field, payloads = r._build_payloads(model=None, format=None, num_prompts=2)
    assert url == "http://localhost:11434/v1/chat/completions"   # chat, not completions; /v1 not doubled
    assert field == "delta"
    assert len(payloads) == 2
    assert payloads[0]["messages"][0]["role"] == "system"
    user = payloads[0]["messages"][1]["content"]
    assert re.match(r"[0-9a-f]{8} ", user) and "coherent passage" in user
    assert user.endswith("Write at length.")
    assert len(user) == 128                                       # exactly the planned length
    assert "prompt" not in payloads[0]                            # not the legacy completions shape
    assert "cache_salt" not in payloads[0]                        # not a vLLM server
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert r._payload_meta[0] == {"planned": 128, "tol": 0}


def test_thinking_drops_the_system_prompt(monkeypatch):
    r = _remote_runner(monkeypatch, thinking=True)
    _u, _f, payloads = r._build_payloads(model=None, format=None, num_prompts=1)
    assert [m["role"] for m in payloads[0]["messages"]] == ["user"]


def test_every_phase_point_and_rep_sends_new_bytes(monkeypatch):
    """The regression gate for issue 1: no payload repeats across warmup, preflight,
    calibration, every sweep point and every rep, and two runs with the same seed differ."""
    from gbench.runners.perf_payload import assert_unique
    seen = []
    for run in range(2):
        r = _remote_runner(monkeypatch)
        for phase in ("warmup", "preflight", "calibrate"):
            seen += r._build_payloads(None, None, 3, phase=phase)[2]
        for rep in range(3):
            r._rep_idx = rep
            for point in range(1, 5):
                seen += r._build_payloads(None, None, 10, phase="sweep", point=point)[2]
    assert_unique(seen)


def test_mm_payloads_have_unique_images_and_nonce_text(monkeypatch):
    from gbench.runners.perf_payload import assert_unique
    r = _remote_runner(monkeypatch)
    r._multimodal = True
    r._vision_tokens_per_image = 100
    _u, _f, payloads = r._build_payloads(None, None, 6, phase="sweep", point=1)
    assert_unique(payloads)
    imgs = [c["image_url"]["url"] for p in payloads for c in p["messages"][-1]["content"]
            if c["type"] == "image_url"]
    assert len(imgs) == 6 * 4 and len(set(imgs)) == len(imgs)
    assert all(u.startswith("data:image/jpeg;base64,") for u in imgs)
    text = payloads[0]["messages"][-1]["content"][0]["text"]
    assert re.match(r"[0-9a-f]{8} ", text)
    assert payloads[0]["messages"][0]["role"] == "system"         # same layout as text


def test_vllm_server_gets_one_cache_salt_per_run(monkeypatch):
    r = _remote_runner(monkeypatch)
    r._serving_runner.is_vllm_server = lambda: True
    p1 = r._build_payloads(None, None, 2, phase="sweep", point=1)[2]
    p2 = r._build_payloads(None, None, 2, phase="sweep", point=2)[2]
    assert p1[0]["cache_salt"] == p2[1]["cache_salt"] == r._payloads.salt


def test_mm_text_uses_the_served_context_window(monkeypatch):
    """Issue 7: a remote endpoint reporting 262144 must not clamp MM text to 16 tokens."""
    r = _remote_runner(monkeypatch)
    r._vision_tokens_per_image = 1091
    r._serving_runner.served_max_model_len = lambda: 262144
    assert r._mm_text_tokens(128, 128) == 128
    assert r._mm_text_tokens(4096, 128) == 4096


def test_payload_checks_flag_length_mismatch_and_cache_reuse(monkeypatch):
    r = _remote_runner(monkeypatch)
    r._prompt_overhead = 20
    meta = [{"planned": 128, "tol": 0}, {"planned": 128, "tol": 1}]
    recs = [{"idx": 0, "send": 1.0, "prompt_tokens": 148},     # exact
            {"idx": 1, "send": 1.0, "prompt_tokens": 151}]     # off by 3 > tol 1
    before = {"prefix_cache_hits": 0.0, "mm_cache_hits": 0.0}
    out = r._payload_checks(recs, meta, before, {"prefix_cache_hits": 40.0, "mm_cache_hits": 0.0})
    assert out["prompt_len_checked"] == 2 and out["prompt_len_mismatch"] == 1
    assert out["cache_reuse_suspect"] == []                      # 40 <= 2 requests x 20
    out = r._payload_checks(recs, meta, before, {"prefix_cache_hits": 300.0, "mm_cache_hits": 0.0})
    assert out["cache_reuse_suspect"] == ["prefix"]
    r._multimodal = True
    out = r._payload_checks(recs, meta, before, {"prefix_cache_hits": 0.0, "mm_cache_hits": 1.0})
    assert "multimodal" in out["cache_reuse_suspect"]


def test_stress_min_samples_flag_env_precedence(monkeypatch):
    """--stress-min-samples (config) > GBENCH_STRESS_MIN_SAMPLES env > class default 15,
    and the prompts-per-point floor scales down with a smaller sample floor (so points
    are faster) while the DEFAULT keeps the original 40-prompt floor unchanged."""
    monkeypatch.delenv("GBENCH_STRESS_MIN_SAMPLES", raising=False)
    r = StressTestRunner(BenchmarkConfig())
    assert r._min_steady == r.MIN_STEADY            # default 15
    assert r._min_prompts == r.MIN_PROMPTS          # default 40, unchanged

    monkeypatch.setenv("GBENCH_STRESS_MIN_SAMPLES", "5")
    r_env = StressTestRunner(BenchmarkConfig())
    assert r_env._min_steady == 5
    assert r_env._min_prompts < r_env.MIN_PROMPTS   # scaled down for a smaller floor
    assert r_env._min_prompts >= 5                  # but still enough to yield the samples

    cfg = BenchmarkConfig()
    cfg.stress_min_samples = 8                       # flag/config wins over the env above
    r_flag = StressTestRunner(cfg)
    assert r_flag._min_steady == 8


def test_lowered_min_samples_makes_a_too_few_point_pass(monkeypatch):
    """A point that meets the SLO and keeps up but has only 8 steady completions is
    TOO-FEW at the default floor (15) yet PASSES once the floor is lowered to 5."""
    def _point():
        return dict(achieved_qps=1.0, slo_attainment=1.0, p99_ttft_ms=100.0,
                    p99_tpot_ms=20.0, p99_itl_ms=20.0, p99_itl_server_ms=None,
                    mean_e2e_ms=100.0, steady_n=8, completed=8, failed=0,
                    client_bound=False, actual_send_qps=1.0, client_sched_lag_s=0.0,
                    num_client_procs=1)
    r_default = StressTestRunner(BenchmarkConfig())            # floor 15
    monkeypatch.setattr(r_default, "_measure_rate_point", lambda *a, **k: _point())
    passed_default, *_ = r_default._test_rate(1.0, None, None)
    assert passed_default is False
    assert r_default._tested_rates[1.0]["passed"] is False     # TOO-FEW (8 < 15)

    cfg = BenchmarkConfig(); cfg.stress_min_samples = 5
    r_low = StressTestRunner(cfg)                              # floor 5
    monkeypatch.setattr(r_low, "_measure_rate_point", lambda *a, **k: _point())
    passed_low, *_ = r_low._test_rate(1.0, None, None)
    assert passed_low is True                                  # 8 >= 5 -> counts


def _fake_capacity_sweep(r, monkeypatch, cap_qps):
    """Wire _test_rate so any rate <= cap_qps passes and anything above fails,
    recording the probe order. Simulates a server with a hard capacity ceiling."""
    probed = []
    def fake_test_rate(qps, model, format, min_n=None, cache=None):
        probed.append(qps)
        passed = qps <= cap_qps
        (r._tested_rates if cache is None else cache)[qps] = {
            "request_rate_qps": qps, "achieved_qps": (qps if passed else 0.0),
            "passed": passed, "verdict": "pass" if passed else "fail",
            "steady_n": (20 if passed else 0),
            "p99_ttft_ms": (100.0 if passed else 9e9),
        }
        return passed, (100.0 if passed else 9e9), 0.0, (qps if passed else 0.0)
    monkeypatch.setattr(r, "_test_rate", fake_test_rate)
    return probed


def test_sweep_is_ascending_and_stops_at_first_fail(monkeypatch):
    """The sweep must probe LOW -> HIGH and stop at the first SLO miss, so a slow
    server's sustainable rates are measured before any over-capacity probe (which
    would flood its queue and spuriously fail every later lower rate)."""
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 2500.0                       # -> cap est 0.4 -> start ~0.2
    probed = _fake_capacity_sweep(r, monkeypatch, cap_qps=0.35)
    out = r._sweep_once(None, None)

    assert probed == sorted(probed)                # strictly ascending, no downward probes
    assert 0 < out["knee_qps"] <= 0.35             # knee is a sustainable rate
    assert sum(1 for q in probed if q > 0.35) <= 1 # at most ONE over-capacity probe (the boundary)
    assert max(probed) == probed[-1]               # the last probe is the highest (stopped there)


def test_sweep_ramps_down_when_start_seeded_too_high(monkeypatch):
    """If the seeded start already misses the SLO (server slower than its single-stream
    estimate), the sweep halves downward to find the small sustainable rate."""
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 100.0                        # -> cap est 10 -> start ~5 (too high)
    probed = _fake_capacity_sweep(r, monkeypatch, cap_qps=0.6)
    out = r._sweep_once(None, None)
    assert 0 < out["knee_qps"] <= 0.6
    assert probed[0] > 0.6                          # first probe was the (failing) high seed
    assert probed[-1] <= 0.6                        # ended on a passing lower rate


def _table(points, capsys):
    from gbench.cli import print_results_summary
    res = {"benchmark_type": "stress_test", "model": "m", "format": "hf", "multimodal": False,
           "ttft_threshold_ms": 5000, "itl_threshold_ms": 200, "reps": 1,
           "sweep_points": points, "max_sustainable_qps": 0}
    try:
        print_results_summary([res], BenchmarkConfig())
    except Exception:
        pass                                      # only the stress table matters here
    return capsys.readouterr().out


def test_stress_table_labels_each_itl_source(capsys):
    base = dict(arrival_qps=1.0, achieved_qps=1.0, p99_ttft_ms=50.0, slo_attainment=1.0,
                passed=True)
    out = _table([
        {**base, "itl_slo_source": "server", "p99_itl_server_bucket": "(50,75]"},
        {**base, "itl_slo_source": "client_per_token", "p99_itl_per_token_ms": 38.0,
         "p99_itl_ms": 90.0},
        {**base, "itl_slo_source": "client_inter_chunk", "p99_itl_ms": 41.0},
        {**base, "itl_slo_source": "tpot", "p99_itl_ms": 0, "p99_tpot_ms": 12.0},
    ], capsys)
    assert "(50,75]" in out and "38ms c" in out and "41ms ch" in out and "12ms t" in out
    assert "server histogram bucket" in out and "per-token gap" in out and "between streamed chunks" in out and "TPOT" in out


def test_stress_table_server_only_prints_no_fallback_footnote(capsys):
    out = _table([dict(arrival_qps=1.0, achieved_qps=1.0, p99_ttft_ms=50.0, slo_attainment=1.0,
                       passed=True, itl_slo_source="server", p99_itl_server_bucket="(10,25]")],
                 capsys)
    assert "per-token gap" not in out and "ms t'" not in out


@pytest.mark.parametrize("slo,expect", [(5, "below the server's smallest"),
                                        (250, "not a server ITL bucket edge"), (200, None)])
def test_itl_slo_edge_warning_once(caplog, slo, expect):
    r = StressTestRunner(BenchmarkConfig(), tpot_threshold_ms=slo)
    snap = ({0.01: 1.0, 0.025: 1.0, 0.2: 1.0, float("inf"): 1.0}, 1.0)
    with caplog.at_level("WARNING"):
        r._warn_itl_slo_vs_edges(snap)
        r._warn_itl_slo_vs_edges(snap)
    msgs = [m for m in caplog.messages if "ITL SLO" in m]
    if expect is None:
        assert msgs == []
    else:
        assert len(msgs) == 1 and expect in msgs[0]



def test_two_stress_runs_on_one_runner_never_share_payloads(monkeypatch):
    """F3: a second run() (another model/format/mode) bumps the payload call, so its
    multimodal images don't trip the repeat guard and its text never repeats run 1."""
    from gbench.runners.perf_payload import assert_unique
    r = _remote_runner(monkeypatch)
    r._vision_tokens_per_image = 100
    seen = []
    for run in range(2):
        r._payloads.new_call()                    # what run() does first
        for mm in (False, True):
            r._multimodal = mm
            seen += r._build_payloads(None, None, 4, phase="sweep", point=1)[2]
    assert_unique(seen)


def test_payload_checks_report_not_observable_as_none(monkeypatch):
    """F8: no /metrics and no calibrated overhead must read as 'not checked', never clean."""
    r = _remote_runner(monkeypatch)
    out = r._payload_checks([{"idx": 0, "send": 1.0, "prompt_tokens": 5}],
                            [{"planned": 5, "tol": 0}], None, None)
    assert out["cache_reuse_suspect"] is None
    assert out["prompt_len_mismatch"] is None
    assert "not observable" in out["cache_telemetry_note"]


def test_payload_rollup():
    reps = [{"tested_rates": [
        {"request_rate_qps": 1.0, "cache_reuse_suspect": [], "prompt_len_mismatch": 0},
        {"request_rate_qps": 2.0, "cache_reuse_suspect": ["prefix"], "prompt_len_mismatch": 3},
        {"request_rate_qps": 4.0, "cache_reuse_suspect": None, "prompt_len_mismatch": None}]}]
    roll = StressTestRunner._payload_rollup(reps)
    assert roll == {"points_cache_checked": 2, "points_suspect": [2.0],
                    "points_len_checked": 2, "len_mismatch_total": 3}
    # Usage but no /metrics (e.g. Ollama): lengths checked, cache NOT observable.
    ollama = StressTestRunner._payload_rollup([{"tested_rates": [
        {"request_rate_qps": 1.0, "cache_reuse_suspect": None, "prompt_len_mismatch": 0}]}])
    assert ollama["points_cache_checked"] == 0 and ollama["points_len_checked"] == 1
    assert StressTestRunner._payload_rollup([{"tested_rates": [
        {"request_rate_qps": 1.0, "cache_reuse_suspect": None, "prompt_len_mismatch": None}]}]) is None


def _steady_recs(n=40, chunk_tokens=4, gap_ms=40.0):
    """Records of requests streaming chunks of `chunk_tokens` tokens every gap_ms."""
    recs = []
    for i in range(n):
        send = 100.0 + i * 0.1
        recs.append({"send": send, "done": send + 2.0, "ok": True, "ttft": 50.0,
                     "tpot": gap_ms, "itls": [gap_ms] * 20,
                     "itl_pieces": ["abcd" * chunk_tokens] * 20,
                     "completion_tokens": 20 * chunk_tokens + 1, "prompt_tokens": 10})
    return recs


class _FourCharTok:
    """One token per 4 characters."""
    def __call__(self, texts, add_special_tokens=False):
        return type("E", (), {"input_ids": [[0] * (len(t) // 4) for t in texts]})()


def test_stress_per_token_itl_divides_chunk_gaps():
    r = StressTestRunner(BenchmarkConfig())
    r._tokenizer = _FourCharTok()
    r._recount_ok, r._recount_for = True, id(r._tokenizer)   # self-check tested separately
    m = r._steady_state_metrics(_steady_recs(chunk_tokens=4, gap_ms=40.0), qps=1.0)
    assert m["p99_itl_ms"] == pytest.approx(40.0)              # inter-chunk
    assert m["p99_itl_per_token_ms"] == pytest.approx(10.0)    # 4 tokens per chunk
    assert m["tpot_source"] == "per_token_usage"
    # (2000 ms - 50 ms) / (81 - 1) tokens
    assert m["p99_tpot_ms"] == pytest.approx(1950.0 / 80.0)


def test_stress_without_tokenizer_has_no_per_token_itl():
    r = StressTestRunner(BenchmarkConfig())
    m = r._steady_state_metrics(_steady_recs(), qps=1.0)
    assert m["p99_itl_per_token_ms"] is None


def test_fallback_gate_prefers_per_token_and_crosscheck_skips_spec_decode(monkeypatch):
    r = StressTestRunner(BenchmarkConfig(), tpot_threshold_ms=50)
    # No server histogram: per-token 12 ms decides (inter-chunk 90 ms would fail).
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        p99_itl_server_ms=None, p99_itl_ms=90.0, p99_itl_per_token_ms=12.0))
    passed, *_ = r._test_rate(3.0, None, None)
    assert passed and r._tested_rates[3.0]["itl_slo_source"] == "client_per_token"
    # Decisive server gate in (10,25], client per-token 200 ms: disagreement > 1 bucket...
    hist = {"10": 0.0, "25": 990.0, "50": 1000.0, "inf": 1000.0}
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        p99_itl_server_ms=25.0, itl_gate_outcome="PASS", p99_itl_per_token_ms=200.0,
        itl_hist_delta_ms=hist, spec_decode_active=False))
    r._test_rate(3.0, None, None)
    assert r._tested_rates[3.0]["itl_client_server_disagree"] is True
    # ...but not judged under speculative decoding (one server gap per engine step).
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        p99_itl_server_ms=25.0, itl_gate_outcome="PASS", p99_itl_per_token_ms=200.0,
        itl_hist_delta_ms=hist, spec_decode_active=True))
    r._test_rate(3.0, None, None)
    assert r._tested_rates[3.0]["itl_client_server_disagree"] is None


def test_spec_decode_gates_on_client_per_token_not_server_step(monkeypatch):
    """Item 1: under speculative decoding a decisive server PASS (per-step gaps) must not
    decide a per-token SLO; the client per-token gap does, at every point."""
    r = StressTestRunner(BenchmarkConfig(), tpot_threshold_ms=50)
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        p99_itl_server_ms=25.0, itl_gate_outcome="PASS", p99_itl_per_token_ms=80.0,
        spec_decode_active=True))
    passed, *_ = r._test_rate(3.0, None, None)
    pt = r._tested_rates[3.0]
    assert passed is False
    assert pt["itl_slo_source"] == "client_per_token (spec decode)"
    assert pt["itl_gated_ms"] == 80.0 and pt["itl_gated_limit_ms"] == 50


def test_spec_decode_without_tokenizer_uses_per_step_limit(monkeypatch):
    r = StressTestRunner(BenchmarkConfig(), tpot_threshold_ms=50)
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        p99_itl_server_ms=100.0, itl_gate_outcome="FAIL", p99_itl_per_token_ms=None,
        spec_decode_active=True, itl_gate_outcome_per_step="PASS",
        itl_gate_limit_per_step_ms=150.0, tokens_per_engine_step=3.0))
    passed, *_ = r._test_rate(3.0, None, None)
    pt = r._tested_rates[3.0]
    assert passed is True and pt["itl_slo_source"].startswith("server (per-step limit")
    assert pt["itl_gated_limit_ms"] == 150.0


def test_recount_self_check_rejects_dummy_prefix_tokenizers():
    class DummyPrefix:
        def __call__(self, t, add_special_tokens=False):
            if isinstance(t, list):
                return type("E", (), {"input_ids": [[1] + [0] * len(x.split()) for x in t]})()
            return type("E", (), {"input_ids": [0] * len(t.split())})()

        def decode(self, ids):
            return "w"
    r = StressTestRunner(BenchmarkConfig())
    r._tokenizer = DummyPrefix()
    assert r._per_token_recount_ok() is False
    assert r._p99_per_token_itl(_steady_recs(), lambda v: max(v)) is None


def test_tpot_source_reports_mixed():
    r = StressTestRunner(BenchmarkConfig())
    recs = _steady_recs()
    for rec in recs[::2]:
        rec["completion_tokens"] = None
    m = r._steady_state_metrics(recs, qps=1.0)
    assert m["tpot_source"].startswith("mixed (")


def test_spec_decode_indeterminate_per_step_never_uses_the_per_token_server_gate(monkeypatch):
    """Skeptic defect B: spec decode, no tokenizer, per-step gate INDETERMINATE -> the
    client fallback decides (labelled), never per-step gaps against the per-token SLO."""
    r = StressTestRunner(BenchmarkConfig(), tpot_threshold_ms=100)
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        p99_itl_server_ms=300.0, itl_gate_outcome="FAIL", p99_itl_per_token_ms=None,
        p99_itl_ms=60.0, spec_decode_active=True, itl_gate_outcome_per_step="INDETERMINATE",
        itl_gate_limit_per_step_ms=230.0))
    passed, *_ = r._test_rate(3.0, None, None)
    pt = r._tested_rates[3.0]
    assert pt["itl_slo_source"].startswith("client_inter_chunk") and passed is True


def test_recount_check_reruns_for_a_new_tokenizer():
    """Skeptic defect D: one runner, two models: the self-check follows the tokenizer."""
    r = StressTestRunner(BenchmarkConfig())
    r._tokenizer = object()                    # breaks the check -> disabled
    assert r._per_token_recount_ok() is False

    class Good:
        def __call__(self, t, add_special_tokens=False):
            if isinstance(t, list):
                return type("E", (), {"input_ids": [[0] for _ in t]})()
            return type("E", (), {"input_ids": [0] * len(t.split())})()

        def decode(self, ids):
            return "w"
    r._tokenizer = Good()
    assert r._per_token_recount_ok() is True


# ── step 5: cohort analysis on simulated servers ──────────────────────────────

def _sim_point(server, qps, n, seed=1):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    from gbench.analysis.perf_load import merged_schedule
    offsets = merged_schedule(qps, n, seed)
    t0 = 1000.0
    recs = server.serve([t0 + o for o in offsets], np.random.default_rng(seed))
    for r in recs:
        r["planned"], r["lag"] = r["send"], 0.0
    return recs, t0, offsets


def _cohort_runner():
    r = StressTestRunner(BenchmarkConfig(), ttft_threshold_ms=5000, tpot_threshold_ms=200)
    r._workload_shape = lambda: ("random", 128, 128)
    return r


def test_cohort_metrics_unlimited_server_passes():
    from perf_sim import Latency, UnlimitedServer
    recs, t0, off = _sim_point(UnlimitedServer(Latency(1.0, 0.3)), 5.0, 300)
    m = _cohort_runner()._cohort_metrics(recs, 5.0, t0, off, [], [], planned_n=300)
    assert m["saturation"] == "NOT_DETECTED"
    assert m["completion_fraction"] == 1.0
    assert m["client_bound"] is False
    assert m["cohort_n"] >= 250 and m["steady_n"] == m["cohort_n"]
    assert 4.0 < m["cohort_offered_qps"] < 6.5
    assert m["offered"]["low"] <= 5.0 <= m["offered"]["high"]


def test_cohort_metrics_detect_heavy_overload():
    from perf_sim import BatchingServer
    srv = BatchingServer(max_running=256, b=0.00005)
    q = 1.25 * srv.capacity_qps()
    recs, t0, off = _sim_point(srv, q, int(q * 120))
    m = _cohort_runner()._cohort_metrics(recs, q, t0, off, [], [], planned_n=len(off))
    assert m["saturation"] == "SATURATED"


def test_unfinished_requests_count_against_the_cohort():
    from perf_sim import CapacityServer, Latency, Shape
    srv = CapacityServer(1, Latency(1.0, 0.3), Shape(max_point_s=40.0))
    q = 2.0 * srv.capacity_qps()
    recs, t0, off = _sim_point(srv, q, 120)
    m = _cohort_runner()._cohort_metrics(recs, q, t0, off, [], [], planned_n=120)
    assert m["p99_ttft_ms"] == float("inf")                  # never-finished = +inf
    assert m["completion_fraction"] < 0.9


def test_client_bound_checks():
    r = _cohort_runner()
    on_time = [{"send": t, "planned": t, "lag": 0.001} for t in range(100)]
    assert r._client_bound_checks(on_time, 100)["client_bound"] is False
    few = r._client_bound_checks(on_time[:90], 100)
    assert few["client_bound"] and "planned sends" in few["client_bound_reasons"][0]
    late = [{"send": t + 0.01 * t, "planned": t, "lag": 0.01 * t} for t in range(100)]
    lb = r._client_bound_checks(late, 100)
    assert lb["client_bound"] and any("growing" in x for x in lb["client_bound_reasons"])


def _fake_attempts(r, lat_by_attempt, monkeypatch):
    seen = []

    def fake_attempt(model, fmt, qps, n, point):
        lat = lat_by_attempt[min(len(seen), len(lat_by_attempt) - 1)]
        seen.append((n, point))
        offs = [i / qps for i in range(1, n + 1)]
        recs = [{"send": 1000 + o, "planned": 1000 + o, "done": 1000 + o + lat, "ttft": 50.0,
                 "ok": True, "lag": 0.0, "completion_tokens": 128, "itls": [10.0],
                 "finish_reason": "length"} for o in offs]
        return {"records": recs, "meta": [], "offsets": offs, "t0": 1000.0, "snaps": [],
                "n_procs": 1, "late_start": False, "planned_n": n}
    monkeypatch.setattr(r, "_run_attempt", fake_attempt)
    return seen


def test_window_too_short_reruns_once_and_a_good_rerun_is_not_inconclusive(monkeypatch):
    """A schedule shorter than 3 x the measured median latency is re-run once (new point
    key, at most twice the requests); a re-run that fixes it leaves no inconclusive flag."""
    r = _cohort_runner()
    seen = _fake_attempts(r, [4.0, 4.0], monkeypatch)
    r._point_seq = 7
    m = r._measure_rate_point_inner(None, None, qps=2.0, num_prompts=20)   # 10 s < 3 x 4 s
    assert m["reruns"] == ["window too short"] and m["flags"] == []
    assert len(seen) == 2 and seen[1][1] == 8 and 20 < seen[1][0] <= 40


def test_window_still_too_short_after_rerun_is_inconclusive(monkeypatch):
    r = _cohort_runner()
    _fake_attempts(r, [4.0, 30.0], monkeypatch)                    # latency grows on re-run
    m = r._measure_rate_point_inner(None, None, qps=2.0, num_prompts=20)
    assert "window too short" in m["flags"]


def test_an_overloaded_point_is_not_rerun(monkeypatch):
    """A point that already failed completion (backlog) is judged as it is, never
    relabelled INCONCLUSIVE by a re-run."""
    r = _cohort_runner()
    seen = _fake_attempts(r, [200.0], monkeypatch)                 # e2e far beyond the bound
    m = r._measure_rate_point_inner(None, None, qps=2.0, num_prompts=20)
    assert len(seen) == 1 and "window too short" not in m["flags"]
    assert m["completion_fraction"] is not None and m["completion_fraction"] < 0.9



def test_certification_bisects_inside_the_bracket(monkeypatch):
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 1000.0
    probed = _fake_capacity_sweep(r, monkeypatch, cap_qps=0.7)
    out = r._certify_knee(None, None, {"pass": 0.5, "fail": 1.0})
    assert out["certified"] and 0.5 <= out["value"] <= 0.7
    hi = min([p["rate"] for p in out["probes"] if p["verdict"] == "fail"] + [1.0])
    assert hi / out["value"] <= r.CERTIFY_STOP_RATIO + 1e-6       # +-9% resolution
    assert out["n_per_probe"] >= 900                                # P99 needs ~1000


def test_certification_steps_down_instead_of_reporting_zero(monkeypatch):
    cfg = BenchmarkConfig()
    cfg.stress_certify = "always"
    r = StressTestRunner(cfg)
    r._probe_e2e_ms = 1000.0
    _fake_capacity_sweep(r, monkeypatch, cap_qps=0.2)             # the bracket's pass was noise
    out = r._certify_knee(None, None, {"pass": 0.5, "fail": 1.0})
    assert out["certified"] and out["value"] <= 0.2


def test_certification_budget_leaves_it_uncertified_with_the_time_needed(monkeypatch):
    cfg = BenchmarkConfig()
    cfg.stress_certify_budget = 1.0                               # 1 minute
    r = StressTestRunner(cfg)
    r._probe_e2e_ms = 4000.0
    _fake_capacity_sweep(r, monkeypatch, cap_qps=0.05)
    out = r._certify_knee(None, None, {"pass": 0.03, "fail": 0.06})
    assert out["certified"] is False and out["needed_min"] > 1.0 and out["probes"] == []


def test_certification_off():
    cfg = BenchmarkConfig()
    cfg.stress_certify = "off"
    out = StressTestRunner(cfg)._certify_knee(None, None, {"pass": 1.0, "fail": 2.0})
    assert out["certified"] is False and out["reason"] == "certification off"


def test_halving_reseeds_after_a_second_overload(monkeypatch):
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 100.0                                       # start 5 req/s, far too high
    probed = _fake_capacity_sweep(r, monkeypatch, cap_qps=0.3)
    monkeypatch.setattr(r, "_fresh_single_stream_e2e", lambda m, f: 1.0)   # reseed 0.5
    out = r._sweep_once(None, None)
    assert probed[:3] == [5.0, 2.5, 0.5]                         # 5 fail, 2.5 fail, reseed 0.5
    assert out["overloaded_probes"] >= 2 and 0 < out["knee_qps"] <= 0.3


def test_unfinished_requests_do_not_mask_a_growing_queue():
    """Live E2B finding: at 3x capacity most cohort requests never finish. Their age at
    the end of the point shrinks with send time, so using it as TTFT masked the growth;
    the queue-growth test must read the requests that did get a first token."""
    from perf_sim import CapacityServer, Latency, Shape
    srv = CapacityServer(4, Latency(2.0, 0.3), Shape(max_point_s=120.0))
    q = 3.0 * srv.capacity_qps()
    recs, t0, off = _sim_point(srv, q, int(q * 90))
    assert sum(1 for r in recs if not r["ok"]) > len(recs) / 3        # many never finished
    m = _cohort_runner()._cohort_metrics(recs, q, t0, off, [], [], planned_n=len(off))
    assert m["saturation"] == "SATURATED"


def test_stress_headline_says_certified_or_uncertified(capsys):
    from gbench.cli import print_results_summary
    base = {"benchmark_type": "stress_test", "model": "m", "format": "hf", "multimodal": False,
            "ttft_threshold_ms": 5000, "itl_threshold_ms": 200, "reps": 3,
            "sweep_points": [], "max_sustainable_qps": 0.81,
            "brackets": {"brackets": [{"pass": 0.8, "fail": 1.6}] * 3, "consistent": True}}
    for res, want in (({**base, "max_sustainable_qps_certified": True,
                        "certification": {"n_per_probe": 1000, "stop_ratio": 1.19}}, "CERTIFIED"),
                      ({**base, "max_sustainable_qps_certified": False,
                        "certification": {"needed_min": 540.0}}, "certifying needs ~540 min")):
        try:
            print_results_summary([res], BenchmarkConfig())
        except Exception:
            pass
        assert want in capsys.readouterr().out



def test_certification_refuted_under_budget_reports_no_headline_reason(monkeypatch):
    """Probes failed at and below the sweep's pass: refuted (no limit), even when the
    budget then stops the step-down; the time to go lower is reported."""
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 1000.0
    _fake_capacity_sweep(r, monkeypatch, cap_qps=0.2)
    out = r._certify_knee(None, None, {"pass": 0.5, "fail": 1.0})
    assert out["outcome"] == "refuted" and not out["certified"]
    assert out["refuted_down_to"] <= 0.5 and "needed_min" in out


def test_certification_never_probes_above_the_cap(monkeypatch):
    cfg = BenchmarkConfig()
    cfg.stress_max_qps = 100.0
    r = StressTestRunner(cfg)
    r._probe_e2e_ms = 10.0
    probed = _fake_capacity_sweep(r, monkeypatch, cap_qps=1e9)
    out = r._certify_knee(None, None, {"pass": 90.0, "fail": None})
    assert max(probed) <= 100.0 and out.get("censored_at_cap") is True


def test_certification_claims_resolution_only_when_reached(monkeypatch):
    cfg = BenchmarkConfig()
    cfg.stress_certify = "always"
    r = StressTestRunner(cfg)
    r._probe_e2e_ms = 100.0
    _fake_capacity_sweep(r, monkeypatch, cap_qps=0.7)
    out = r._certify_knee(None, None, {"pass": 0.5, "fail": 1.0})
    assert out["certified"] and out["resolution_pct"] <= 9.1     # sqrt(1.19) - 1


def _contam_runner(monkeypatch, plan):
    """plan: per attempt, 'late' | 'clean' | 'dirty'."""
    r = _cohort_runner()
    r._point_log = []
    calls = []

    def fake_attempt(model, fmt, qps, n, point):
        kind = plan[len(calls)]
        calls.append((n, point))
        if kind == "late":
            return {"late_start": True}
        offs = [i / qps for i in range(1, n + 1)]
        recs = [{"send": 1000 + o, "planned": 1000 + o, "done": 1000 + o + 0.5, "ttft": 50.0,
                 "ok": True, "lag": 0.0, "completion_tokens": 128, "itls": [10.0],
                 "finish_reason": "length"} for o in offs]
        return {"records": recs, "meta": [], "offsets": offs, "t0": 1000.0, "snaps": [],
                "n_procs": 1, "late_start": False, "planned_n": n, "dirty": kind == "dirty"}
    orig = r._cohort_metrics

    def fake_cohort(records, qps, t0, offs, snaps, meta, flags=None, planned_n=None):
        m = orig(records, qps, t0, offs, snaps, meta, flags=flags, planned_n=planned_n)
        m["contaminated"] = plan[len(calls) - 1] == "dirty"
        return m
    monkeypatch.setattr(r, "_run_attempt", fake_attempt)
    monkeypatch.setattr(r, "_cohort_metrics", fake_cohort)
    monkeypatch.setattr(r, "_drain_or_raise", lambda: True)
    return r, calls


def test_one_contaminated_observation_is_remeasured_not_fatal(monkeypatch):
    r, calls = _contam_runner(monkeypatch, ["clean"][:0] + ["late", "dirty", "clean"])
    m = r._measure_rate_point_inner(None, None, qps=10.0, num_prompts=200)
    assert m["contaminated"] is False and "contaminated" in m["reruns"]
    assert len({p for _, p in calls}) == len(calls)            # new payload key each time


def test_contaminated_then_late_start_never_passes_on_dirty_data(monkeypatch):
    r, _ = _contam_runner(monkeypatch, ["dirty", "late", "clean"])
    m = r._measure_rate_point_inner(None, None, qps=10.0, num_prompts=200)
    assert m["contaminated"] is False                           # judged on the clean one


def test_two_contaminated_observations_end_the_pillar(monkeypatch):
    from gbench.runners.stress import StressContaminated
    r, _ = _contam_runner(monkeypatch, ["dirty", "dirty"])
    with pytest.raises(StressContaminated):
        r._measure_rate_point_inner(None, None, qps=10.0, num_prompts=200)
    assert len(r._point_log) == 2                               # both kept for the JSON


def test_contaminated_point_never_passes_the_gate(monkeypatch):
    r = StressTestRunner(BenchmarkConfig())
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        itl_gate_outcome="PASS", contaminated=True))
    passed, *_ = r._test_rate(3.0, None, None)
    assert passed is False



def test_inconclusive_certification_probe_is_never_a_failure(monkeypatch):
    """A probe whose cohort came up short (or that stayed otherwise undecidable) is
    retested once; still inconclusive -> 'unverified', never 'refuted' or a new 'hi'."""
    cfg = BenchmarkConfig()
    cfg.stress_certify = "always"
    r = StressTestRunner(cfg)
    r._probe_e2e_ms = 100.0
    calls = []

    def ftr(qps, model, format, min_n=None, cache=None):
        calls.append(qps)
        (r._tested_rates if cache is None else cache)[qps] = {
            "request_rate_qps": qps, "passed": False, "verdict": "inconclusive",
            "p99_ttft_ms": 1.0, "p99_tpot_ms": 1.0, "achieved_qps": qps,
            "flags": ["cohort 999 < 1000 required"]}
        return False, 1.0, 1.0, qps
    monkeypatch.setattr(r, "_test_rate", ftr)
    out = r._certify_knee(None, None, {"pass": 0.5, "fail": 1.0})
    assert out["outcome"] == "unverified" and "refuted_down_to" not in out
    assert len(calls) == 2 and calls[0] == calls[1]                  # one retest


def test_certification_censored_at_the_cap_is_a_lower_bound(monkeypatch):
    cfg = BenchmarkConfig()
    cfg.stress_max_qps = 100.0
    cfg.stress_certify = "always"
    r = StressTestRunner(cfg)
    r._probe_e2e_ms = 10.0
    _fake_capacity_sweep(r, monkeypatch, cap_qps=1e9)
    out = r._certify_knee(None, None, {"pass": 60.0, "fail": None})
    assert out["certified"] and out["interval"][1] is None and out["resolution_pct"] is None
    assert "censored" in out["note"]
    out2 = r._certify_knee(None, None, {"pass": 100.0, "fail": None})
    assert out2["outcome"] == "censored" and not out2["certified"]


def test_rerun_never_shrinks_requests(monkeypatch):
    """A window-too-short re-run sized from a SMALLER latency estimate keeps n."""
    r = _cohort_runner()
    seen = _fake_attempts(r, [4.0, 0.1], monkeypatch)
    r._probe_e2e_ms = 50.0                                     # tiny queue-free latency
    r._measure_rate_point_inner(None, None, qps=2.0, num_prompts=20)
    assert seen[1][0] >= seen[0][0]


# ── Final-review regressions (2026-10-15) ─────────────────────────────────────

def _overloaded(**over):
    return _rate_metrics(saturation="SATURATED", completion_fraction=0.4,
                         p99_ttft_ms=float("inf"), itl_gate_outcome="PASS", **over)


def test_decisive_overload_fails_even_with_a_short_cohort(monkeypatch):
    """A saturated point is a failure even when its cohort came up short (overload at a
    lower realized load implies overload at the label); a late client or no data keeps
    it inconclusive (a busy client inflates its own TTFT)."""
    r = StressTestRunner(BenchmarkConfig())
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _overloaded(cohort_n=900))
    r._test_rate(3.0, None, None, min_n=1100)
    assert r._tested_rates[3.0]["verdict"] == "fail"
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point",
                        lambda *a, **k: _overloaded(cohort_n=1200, client_bound=True))
    r._test_rate(3.0, None, None, min_n=1100)
    assert r._tested_rates[3.0]["verdict"] == "inconclusive"
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point",
                        lambda *a, **k: _overloaded(cohort_n=0, flags=["no data"]))
    r._test_rate(3.0, None, None, min_n=1100)
    assert r._tested_rates[3.0]["verdict"] == "inconclusive"
    # a healthy point with a short cohort only cannot pass: inconclusive
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics(
        itl_gate_outcome="PASS", cohort_n=900))
    r._test_rate(3.0, None, None, min_n=1100)
    assert r._tested_rates[3.0]["verdict"] == "inconclusive"


def test_rate_must_be_positive():
    with pytest.raises(ValueError):
        StressTestRunner(BenchmarkConfig())._test_rate(0.0, None, None)


def _run_runner(monkeypatch, measure, e2e_ms=1000.0, **cfgkw):
    """A runner whose run() needs no server: the measurement is ``measure``."""
    import types
    from gbench.runners import stress as S

    class _SR:
        server_port = 1
        payloads = types.SimpleNamespace(record=lambda: {})

        def __init__(self, cfg):
            pass

        def _resolve_model_id(self, m, f):
            return "m"

        def _tokenizer_path(self, m, f):
            return "t"

        def _stamp_server(self, res, m, f):
            pass
    monkeypatch.setattr(S, "ServingBenchmarkRunner", _SR)
    import gbench.utils as U
    monkeypatch.setattr(U, "safe_get_tokenizer", lambda *a, **k: _CharTok())
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = "http://x:1/v1"
    for k, v in cfgkw.items():
        setattr(cfg, k, v)
    r = S.StressTestRunner(cfg)
    r._warmup = lambda m, f: None
    r._calibrate_prompt_overhead = lambda m, f: None

    def pf(m, f, mm):
        r._probe_e2e_ms, r._ignore_eos_honored = e2e_ms, True
    r._preflight_floor_check = pf
    r._drain_or_raise = lambda: True
    r._save_stress_result = lambda *a: None
    r._fresh_single_stream_e2e = lambda m, f: e2e_ms / 1000.0
    r._payload_record = lambda: {}
    r._measure_rate_point = measure
    model = types.SimpleNamespace(short_name="m", vision_tokens_per_image=None,
                                  supports_multimodal=False)
    return r, model, types.SimpleNamespace(value="remote")


def _cap_measure(cap, short_when_over=False, calls=None, over=None):
    from gbench.analysis.perf_load import NOT_DETECTED, SATURATED

    def m(model, fmt, qps, n, min_n=None):
        if calls is not None:
            calls.append(qps)
        ok = qps <= cap
        coh = int(n * (0.9 if ok or not short_when_over else 0.6))
        d = dict(achieved_qps=qps, slo_attainment=1.0 if ok else 0.3,
                 p99_ttft_ms=100.0 if ok else float("inf"), p99_tpot_ms=20.0, p99_itl_ms=20.0,
                 steady_n=coh, completed=n, failed=0, cohort_n=coh,
                 saturation=NOT_DETECTED if ok else SATURATED,
                 completion_fraction=1.0 if ok else 0.5, flags=[], client_bound=False,
                 mean_e2e_ms=1000.0, p90_e2e_s=1.2, num_client_procs=1)
        d.update((over or {}).get("all", {}))
        return d
    return m


def test_certification_with_short_cohort_overloaded_probes_still_certifies(monkeypatch):
    """Overloaded certification probes whose cohort is short lower the interval's top;
    they never leave the run 'unverified' on the sweep's noisy pass."""
    r, model, fmt = _run_runner(monkeypatch, _cap_measure(2.5, short_when_over=True),
                                stress_certify="always")
    res = r.run(model, fmt)
    c = res["certification"]
    assert c["outcome"] == "certified" and 2.0 <= res["max_sustainable_qps"] <= 2.5


def test_slow_device_descent_ends_without_hang_or_zero_rate(monkeypatch):
    """1100 s per request: rates are rounded to significant digits (never 0) and the
    downward search ends at the floor."""
    calls = []
    r, model, fmt = _run_runner(monkeypatch, _cap_measure(0.0, calls=calls),
                                e2e_ms=1_100_000.0, stress_reps=1)
    res = r.run(model, fmt)
    assert not res.get("failed") and res["max_sustainable_qps"] is None
    assert calls and min(calls) > 0 and len(calls) < 40
    assert res["below_measurable_floor"]


def test_no_rate_names_the_blocker_not_the_slo(monkeypatch):
    """When every point was client-bound (not a server failure) the result names that,
    instead of claiming the device is below the floor."""
    from gbench.analysis.perf_load import NOT_DETECTED

    def m(model, fmt, qps, n, min_n=None):
        return dict(achieved_qps=qps, slo_attainment=1.0, p99_ttft_ms=100.0, p99_tpot_ms=20.0,
                    p99_itl_ms=20.0, steady_n=n, completed=n, failed=0, cohort_n=n,
                    saturation=NOT_DETECTED, completion_fraction=1.0, flags=[],
                    client_bound=True, mean_e2e_ms=1000.0, p90_e2e_s=1.2, num_client_procs=1)
    r, model, fmt = _run_runner(monkeypatch, m, stress_reps=1, stress_certify="off")
    res = r.run(model, fmt)
    assert res["max_sustainable_qps"] is None and res["below_measurable_floor"] is None
    assert "client-bound" in res["no_rate_reason"]


def test_payload_check_includes_the_certification_probes(monkeypatch):
    m = _cap_measure(2.5, over={"all": {"cache_reuse_suspect": [], "prompt_len_mismatch": 0}})
    r, model, fmt = _run_runner(monkeypatch, m, stress_certify="always", stress_reps=1)
    res = r.run(model, fmt)
    n_sweep = len(res["sweep_points_all_reps"][0])
    assert res["payload_check"]["points_cache_checked"] == n_sweep + len(res["certification_points"])
    assert res["knee_cache_reuse_suspect"] == [] and res["knee_prompt_len_mismatch"] == 0


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _drain_runner(monkeypatch, snaps):
    from gbench.runners import stress as S
    clock = _Clock()
    monkeypatch.setattr(S, "time", clock)
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 1000.0
    monkeypatch.setattr(r, "_metrics_url", lambda: "http://x/metrics")
    it = iter(snaps)
    r.scrapes = 0

    def snap():
        r.scrapes += 1
        return next(it)
    monkeypatch.setattr(r, "_metrics_snapshot", snap)
    return r


_IDLE = {"running": 0.0, "waiting": 0.0, "request_success": 5.0, "prompt_tokens": 9.0}
_BUSY = {"running": 50.0, "waiting": 300.0, "request_success": 5.0, "prompt_tokens": 9.0}


def test_a_failed_scrape_is_not_a_drained_server(monkeypatch):
    import itertools
    # /metrics worked (busy), then one scrape failed: keep waiting until really idle
    r = _drain_runner(monkeypatch, [_BUSY, _BUSY, None, _BUSY, _IDLE, _IDLE])
    assert r._drain_or_raise() is True and r.scrapes == 6
    # a failed scrape in a later drain, after /metrics was seen, is not "drained" either
    r2 = _drain_runner(monkeypatch, [_IDLE, _IDLE, None, _IDLE, _IDLE, _IDLE])
    assert r2._drain_or_raise() is True and r2.scrapes == 2
    assert r2._drain_or_raise() is True and r2.scrapes == 6
    r = _drain_runner(monkeypatch, itertools.repeat(_BUSY))
    from gbench.runners.stress import StressContaminated
    with pytest.raises(StressContaminated):
        r._ensure_drained()                     # busy throughout: second timeout ends it


def test_no_metrics_at_all_waits_a_fixed_cooldown(monkeypatch):
    import itertools
    from gbench.runners import stress as S
    r = _drain_runner(monkeypatch, itertools.repeat(None))
    t0 = S.time.time()
    assert r._drain_or_raise() is True
    assert S.time.time() - t0 >= 5.0


def test_certification_budget_zero_means_off(monkeypatch):
    from gbench.runners.stress import certify_settings
    cfg = BenchmarkConfig()
    cfg.stress_certify_budget = 0.0
    out = StressTestRunner(cfg)._certify_knee(None, None, {"pass": 1.0, "fail": 2.0})
    assert out["certified"] is False and out["probes"] == [] and "budget 0" in out["reason"]
    monkeypatch.setenv("GBENCH_STRESS_CERTIFY_BUDGET_MIN", "0")
    assert certify_settings(BenchmarkConfig())[1] == 0.0
    monkeypatch.setenv("GBENCH_STRESS_CERTIFY", "OFF")
    assert certify_settings(BenchmarkConfig())[0] == "off"
    cfg.stress_certify_budget = -1.0
    with pytest.raises(ValueError):
        certify_settings(cfg)


def test_preflight_bails_when_single_stream_decode_misses_the_itl_slo(monkeypatch):
    r = _preflight_runner(monkeypatch, ttft_ms=120.0)
    # 128 tokens in 63.62 s after the first: 500 ms per token > 200 ms ITL SLO
    monkeypatch.setattr(
        "gbench.runners.stress._stress_client_worker",
        lambda task: {"records": [{"ok": True, "ttft": 120.0, "send": 0.0,
                                   "done": 0.12 + 0.5 * 127, "completion_tokens": 128}],
                      "sched_lag_s": 0.0})
    out = r._preflight_floor_check(_StubModel(), _StubFormat(), multimodal=False)
    assert out["below_stress_floor"] and out["below_stress_floor_reason"] == "ITL"
    assert out["preflight_single_stream_per_token_ms"] == pytest.approx(500.0)
    assert "--itl-slo-ms" in out["note"]


def test_stress_without_a_tokenizer_fails_instead_of_sending_unsized_text(monkeypatch):
    import gbench.utils as U
    calls = []
    r, model, fmt = _run_runner(monkeypatch, _cap_measure(2.5, calls=calls), stress_reps=1)
    monkeypatch.setattr(U, "safe_get_tokenizer", lambda *a, **k: None)
    res = r.run(model, fmt)
    assert res.get("failed") and res.get("failure_kind") == "harness"
    assert "--tokenizer" in str(res.get("error")) and calls == []      # nothing was measured


def test_build_payloads_without_a_tokenizer_raises():
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = "http://127.0.0.1:8000/v1"
    r = StressTestRunner(cfg)
    r._api_model_id = "m"
    r._multimodal = False
    with pytest.raises(RuntimeError, match="tokenizer"):
        r._build_payloads(model=None, format=None, num_prompts=1)


def test_vllm_perf_requests_keep_special_tokens_visible():
    """Live finding (2026-10-15): ~15% of random-token prompts made gemma-4 reply with
    special tokens only (<turn|>, <eos>), which vLLM strips from the streamed text, so no
    first token was seen and a healthy 1 req/s point read as BACKLOG. vLLM requests now
    send skip_special_tokens False (serving and stress)."""
    from gbench.runners.serving import ServingBenchmarkRunner
    r = StressTestRunner(BenchmarkConfig())
    r._serving_runner = type("S", (), {"is_vllm_server": lambda self: True})()
    r._rep_idx = 0
    assert r._request_extras("sweep", 1, False)["skip_special_tokens"] is False
    r._serving_runner = type("S", (), {"is_vllm_server": lambda self: False})()
    assert "skip_special_tokens" not in r._request_extras("sweep", 1, False)
    s = ServingBenchmarkRunner(BenchmarkConfig())
    s.is_vllm_server = lambda: True
    assert s._extra_body(False, "serving", 0)["skip_special_tokens"] is False
    s.is_vllm_server = lambda: False
    assert s._extra_body(False, "serving", 0) == {}


def test_a_served_reply_without_visible_text_is_measured_not_lost(monkeypatch):
    """A reply the server finished (usage > 0, finish_reason length) whose chunks carried
    no visible text is a measured request, TTFT = its e2e (upper bound), flagged."""
    import asyncio, json as _json
    from gbench.runners import stress as S

    class _Resp:
        status = 200
        def __init__(self):
            async def gen():
                for obj in ({"choices": [{"text": ""}]}, {"choices": [{"text": "", "finish_reason": "length"}]},
                            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 8}}):
                    yield ("data: " + _json.dumps(obj) + "\n").encode()
                yield b"data: [DONE]\n"
            self.content = gen()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _Session:
        def __init__(self, *a, **k): pass
        def post(self, *a, **k): return _Resp()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    import aiohttp
    monkeypatch.setattr(aiohttp, "ClientSession", _Session)
    out = S._stress_client_worker({"payloads": [{"prompt": "x"}], "api_url": "http://x/v1/completions",
                                   "content_field": "text", "request_timeout": 5, "max_point_s": 5,
                                   "itl_cap": 8, "headers": {}})
    rec = out["records"][0]
    assert rec["ok"] and rec["ttft_unseen"] and rec["ttft"] is not None and rec["itls"] is None


def test_campaign_lengths_never_leak_into_the_next_campaign():
    """Live-review finding (2026-10-15): chat-like sets no lengths, so in a multi-campaign
    run it inherited the previous campaign's (long-decode: 8192 output tokens into its
    4096 window). Every campaign starts from the run's own lengths."""
    import types
    from gbench.cli import apply_campaign_to_config
    cfg = BenchmarkConfig()
    cfg.input_lengths, cfg.output_lengths = [128], [512]
    args = types.SimpleNamespace(input_lengths=None, output_lengths=None, dataset=None,
                                 dataset_path=None, max_model_len=None)
    apply_campaign_to_config(cfg, "long-decode", args)
    assert cfg.output_lengths == [8192]
    apply_campaign_to_config(cfg, "chat-like", args)
    assert (cfg.input_lengths, cfg.output_lengths, cfg.dataset) == ([128], [512], "sharegpt")


def test_each_campaign_gets_its_own_result_file(tmp_path):
    from gbench.utils.log_manager import LogManager
    lm = LogManager.__new__(LogManager)
    lm.results_dir = tmp_path
    a = lm.get_serving_result_path("m", "hf", 1, campaign="agentic")
    b = lm.get_serving_result_path("m", "hf", 1, campaign="mixed")
    assert a != b and "agentic" in a.name and lm.get_serving_result_path("m", "hf", 1).name == "serve_m_hf_b1.json"
    cfg = BenchmarkConfig()
    cfg.campaign = "agentic"
    cfg.log_manager = type("LM", (), {"results_dir": tmp_path})()
    r = StressTestRunner(cfg)
    res = {"x": 1}
    r._save_stress_result(res, _StubModel(), _StubFormat(), multimodal=False)
    assert (tmp_path / "performance" / "stress_stub_hf_agentic_text.json").exists()
    assert res["campaign"] == "agentic"
