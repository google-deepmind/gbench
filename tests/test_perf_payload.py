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


"""Unique-by-construction perf payloads (gbench.runners.perf_payload)."""

import io
import re
import subprocess
import sys

import numpy as np
import pytest

from gbench.runners.perf_payload import (
    PayloadFactory, assert_unique, chat_messages, decode_to_target_len, payload_fingerprint,
    stable_u32,
)


class _CharTok:
    """Deterministic toy tokenizer: one token per character (vocab = 0..255)."""
    vocab_size = 256
    all_special_ids = [0]

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [ord(c) % 256 for c in text]})()

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


TOK = _CharTok()


def test_stable_seed_is_the_same_in_another_interpreter():
    here = stable_u32(83, "tok", "sweep", 0, 1, 2)
    code = ("from gbench.runners.perf_payload import stable_u32;"
            "print(stable_u32(83, 'tok', 'sweep', 0, 1, 2))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == here


def test_same_seed_new_salt_same_lengths_different_bytes():
    a, b = PayloadFactory(seed=7), PayloadFactory(seed=7)
    assert a.salt != b.salt
    assert a.planned_lengths(50, 128, 0.5, "sweep", 0, 3) == b.planned_lengths(50, 128, 0.5, "sweep", 0, 3)
    ta, _ = a.random_token_prompt(TOK, 128, "sweep", 0, 3, 0)
    tb, _ = b.random_token_prompt(TOK, 128, "sweep", 0, 3, 0)
    assert ta != tb and len(ta) == len(tb) == 128


def test_nonce_is_first_and_lengths_exact():
    f = PayloadFactory(seed=1, salt="abcd1234")
    text, mismatch = f.random_token_prompt(TOK, 100, "warmup", 0, 0, 5)
    assert text.startswith(f.nonce("warmup", 0, 0, 5)) and " w0.00.000.000005. " in text
    assert mismatch == 0 and len(text) == 100
    t2, m2 = f.filler_prompt(TOK, 300, "lorem ipsum dolor sit amet", "sweep", 1, 2, 3, suffix=" END")
    assert t2.startswith(f.nonce("sweep", 1, 2, 3)) and t2.endswith(" END")
    assert m2 == 0 and len(t2) == 300


def test_nonce_leads_with_a_per_request_code():
    """Two nonces must differ in their very first characters: a shared readable head
    would be a prefix every request shares, which the prefix cache then serves."""
    f = PayloadFactory(seed=1)
    a, b = f.nonce("sweep", 0, 1, 0), f.nonce("sweep", 0, 1, 1)
    c = f.nonce("sweep", 0, 2, 0)
    for x, y in ((a, b), (a, c)):
        common = len(__import__("os").path.commonprefix([x, y]))
        assert common <= 2, (x, y)


def test_new_call_changes_every_key():
    f = PayloadFactory(seed=1, salt="s")
    first = (f.nonce("serving", 0, 0, 0), f.random_token_prompt(TOK, 64, "serving", 0, 0, 0)[0],
             f.unique_image_jpeg("probe", 0, 0, 0, 0))
    f.new_call()
    second = (f.nonce("serving", 0, 0, 0), f.random_token_prompt(TOK, 64, "serving", 0, 0, 0)[0],
              f.unique_image_jpeg("probe", 0, 0, 0, 0))     # no "repeated" error
    assert all(x != y for x, y in zip(first, second))


def test_short_input_warns(caplog):
    f = PayloadFactory(seed=1)
    with caplog.at_level("WARNING"):
        f.random_token_prompt(TOK, 8, "sweep", 0, 0, 0)
    assert any("shorter than the per-request nonce" in m for m in caplog.messages)


def test_real_tokenizer_cache_bound_holds():
    """The invariant the prefix-reuse flag relies on, checked on real Gemma nonces: with
    each request's shared prefix = template (S tokens) + the longest leading-token match
    with any earlier nonce of its point, the prefix-cache hits (whole 16-token blocks)
    summed over the point never exceed n x overhead, where overhead = S + the template
    tokens after the user text (3 for Gemma chat, 0 for raw completions)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it", local_files_only=True)
    except Exception:
        pytest.skip("Gemma tokenizer not in the local HF cache")
    f = PayloadFactory(seed=83)
    n = 2000                     # one run: several points and reps, one cache_salt
    keys = [(rep, point, i) for rep in range(2) for point in range(1, 5) for i in range(250)]
    ids = [tok(f.nonce("sweep", *k), add_special_tokens=False).input_ids for k in keys]
    shared = [0] * n
    by_first: dict = {}
    for i, x in enumerate(ids):
        best = 0
        for j in by_first.get(x[0], []):
            y, k = ids[j], 0
            while k < min(len(x), len(y)) and x[k] == y[k]:
                k += 1
            best = max(best, k)
        by_first.setdefault(x[0], []).append(i)
        shared[i] = best
    block = 16
    # Raw /v1/completions: no template before the text except an optional BOS, nothing
    # after it. Chat templates: any leading length, and >= 3 tokens after the user turn
    # (Gemma 3, Llama 3 and Qwen 5). A layout with a long leading template and NOTHING
    # after it is not covered by the bound (none of these servers produce one).
    layouts = [(S, 0) for S in (0, 1)] + [(S, 3) for S in range(0, 64)]
    for S, after in layouts:
        hits = sum(((S + sh) // block) * block for sh in shared)
        assert hits <= n * (S + after), (S, after, hits)


def test_allowed_vocab_is_cached():
    f = PayloadFactory(seed=1)
    f.random_token_prompt(TOK, 64, "sweep", 0, 0, 0)
    arr = f._allowed[id(TOK)]
    f.random_token_prompt(TOK, 64, "sweep", 0, 0, 1)
    assert f._allowed[id(TOK)] is arr


def test_every_phase_rep_point_index_is_distinct():
    f = PayloadFactory(seed=1)
    seen = set()
    for phase in ("warmup", "preflight", "sweep", "refine", "drain", "serving"):
        for rep in range(3):
            for point in range(4):
                for i in range(20):
                    seen.add(f.random_token_prompt(TOK, 64, phase, rep, point, i)[0])
    assert len(seen) == 6 * 3 * 4 * 20


def test_image_code_survives_jpeg_and_bytes_are_unique():
    from PIL import Image
    f = PayloadFactory(seed=3)
    imgs = [f.unique_image_jpeg("sweep", 0, 0, i, k) for i in range(60) for k in range(4)]
    assert len({bytes(x) for x in imgs}) == len(imgs)
    # decoded pixels differ too (the code is in full-amplitude 8x8 blocks)
    pix = {np.asarray(Image.open(io.BytesIO(x))).tobytes() for x in imgs}
    assert len(pix) == len(imgs)
    # same resolution as the base image (per-image token count unchanged)
    assert Image.open(io.BytesIO(imgs[0])).size == (256, 256)


def test_image_repeat_is_refused():
    f = PayloadFactory(seed=3, salt="s")
    f.unique_image_jpeg("sweep", 0, 0, 0, 0)
    g = PayloadFactory(seed=3, salt="s")
    g._image_hashes = f._image_hashes
    with pytest.raises(RuntimeError):
        g.unique_image_jpeg("sweep", 0, 0, 0, 0)


def test_chat_messages_one_layout():
    assert chat_messages("hi", None) == [{"role": "user", "content": "hi"}]
    assert chat_messages("hi", "sys")[0] == {"role": "system", "content": "sys"}


def test_assert_unique_ignores_transport_fields():
    p1 = {"model": "m", "prompt": "a", "max_tokens": 1, "stream": True}
    p2 = {"model": "m", "prompt": "a", "max_tokens": 2, "stream": False}
    assert payload_fingerprint(p1) == payload_fingerprint(p2)
    with pytest.raises(RuntimeError):
        assert_unique([p1, p2])
    assert_unique([p1, {**p1, "prompt": "b"}])


def test_decode_to_target_len_extends_and_truncates():
    rng = np.random.default_rng(0)
    t, m = decode_to_target_len(TOK, [65] * 10, 20, rng)
    assert m == 0 and len(t) == 20 and t.startswith("A" * 10)
    t, m = decode_to_target_len(TOK, [65] * 30, 20, rng)
    assert m == 0 and t == "A" * 20


# ── serving: every iteration sends new requests ───────────────────────────────

def _serving(remote=True, dataset="random"):
    from gbench.core.config import BenchmarkConfig
    from gbench.runners.serving import ServingBenchmarkRunner
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = "http://h:1/v1" if remote else None
    cfg.dataset = dataset
    cfg.input_lengths, cfg.output_lengths = [512], [16]
    return ServingBenchmarkRunner(cfg)


def test_serving_synthetic_iterations_never_repeat(monkeypatch):
    r = _serving()
    monkeypatch.setattr("gbench.utils.safe_get_tokenizer", lambda *a, **k: TOK)
    monkeypatch.setattr("gbench.utils.check_vllm_available", lambda: True)
    monkeypatch.setattr(r, "_tokenizer_path", lambda m, f: "tok")
    prompts = []
    for phase, reps in (("serving_warmup", 2), ("serving", 3)):
        for rep in range(reps):
            reqs, _t = r._requests_for(None, None, 5, phase, rep, 0)
            prompts += [q.prompt for q in reqs]
    assert len(set(prompts)) == len(prompts) == 25
    assert all(re.match(r"[0-9a-f]{8} ", p) and len(p) == 512 for p in prompts)


def test_serving_real_dataset_slices_are_disjoint_then_counted(monkeypatch):
    r = _serving(dataset="sharegpt")
    pool = [type("Q", (), {"prompt": f"real {i}"})() for i in range(12)]
    monkeypatch.setattr(r, "_generate_sample_requests", lambda *a, **k: (pool[:a[2]], None))
    got = []
    for rep in range(3):
        reqs, _ = r._requests_for(None, None, 4, "serving", rep, 12)
        got += [q.prompt for q in reqs]
    assert len(set(got)) == 12 and r._prompt_reuse == 0
    r._requests_for(None, None, 4, "serving", 3, 12)               # dataset exhausted
    assert r._prompt_reuse == 4


def test_serving_cache_salt_only_for_vllm():
    r = _serving()
    r._is_vllm = False
    assert r._extra_body(False, "serving", 0) == {}
    r._is_vllm = True
    c = r.payloads.call
    assert r._extra_body(False, "serving", 0) == {"cache_salt": f"{r.payloads.salt}.{c}",
                                                  "skip_special_tokens": False}
    assert r._extra_body(True, "serving", 2) == {"cache_salt": f"{r.payloads.salt}.{c}.serving.2",
                                                 "skip_special_tokens": False}


def test_serving_context_window_is_the_served_one():
    r = _serving()
    r._served[("m", "remote")] = {"served_model_id": "m", "served_max_model_len": 262144}
    assert r.served_max_model_len() == 262144
    assert _serving(remote=False).served_max_model_len() == 4096   # gbench's own default


def test_serving_two_configs_never_share_requests(monkeypatch):
    """F1/F2: one runner, two batch sizes (two run_with_iterations calls) never send the
    same prompt or image, and the synthetic cache_salt differs per call."""
    r = _serving()
    r._is_vllm = True
    monkeypatch.setattr("gbench.utils.safe_get_tokenizer", lambda *a, **k: TOK)
    monkeypatch.setattr("gbench.utils.check_vllm_available", lambda: True)
    monkeypatch.setattr(r, "_tokenizer_path", lambda m, f: "tok")
    prompts, salts, imgs = [], [], []
    for _batch in (1, 4):
        r._new_payload_call()
        for rep in range(2):
            reqs, _t = r._requests_for(None, None, 5, "serving", rep, 0)
            prompts += [q.prompt for q in reqs]
            salts.append(r._extra_body(False, "serving", rep)["cache_salt"])
        imgs += [r.payloads.unique_image_jpeg("probe", 0, 0, 0, k) for k in range(4)]
    assert len(set(prompts)) == len(prompts)
    assert len(set(salts)) == 2 and len(set(imgs)) == len(imgs)


def test_serving_real_pool_resets_per_call(monkeypatch):
    """F5: a second model/config gets its own dataset pool and its own reuse count."""
    r = _serving(dataset="sharegpt")
    calls = []

    def gen(*a, **k):
        calls.append(a[2])
        return ([type("Q", (), {"prompt": f"p{len(calls)}-{i}"})() for i in range(4)], None)
    monkeypatch.setattr(r, "_generate_sample_requests", gen)
    r._new_payload_call()
    r._requests_for(None, None, 8, "serving", 0, 8)
    assert r._prompt_reuse == 4
    r._new_payload_call()
    assert r._prompt_reuse == 0 and r._real_pool == []
    r._requests_for(None, None, 2, "serving", 0, 2)
    assert len(calls) == 2


def test_serving_dedupes_sampler_repeats(monkeypatch):
    """F6: repeats returned by the sampler are counted as reuse, not sent as new."""
    r = _serving(dataset="sharegpt")
    same = type("Q", (), {"prompt": "dup"})()
    monkeypatch.setattr(r, "_generate_sample_requests", lambda *a, **k: ([same] * 6, None))
    r._new_payload_call()
    r._requests_for(None, None, 6, "serving", 0, 6)
    assert r._prompt_reuse == 5


def test_serving_tpot_is_per_token_from_output_lens():
    r = _serving()
    # MM/remote shape: one request, 4-token chunks every 40 ms, 81 tokens in 2000 ms.
    res = {"raw_ttfts_ms": [50.0], "raw_itl_blocks_ms": [[40.0] * 20],
           "raw_e2els_ms": [2000.0], "output_lens": [81]}
    out = r._extract_raw_samples(res)
    assert out["tpot_ms"] == [pytest.approx(1950.0 / 80.0)]
    assert out["itl_blocks_ms"] == [[40.0] * 20]          # ITL stays inter-chunk
    # vLLM benchmark() shape (seconds), failed request skipped by index.
    res2 = {"ttfts": [0.05, 0.0], "itls": [[0.04] * 20, []], "output_lens": [81, 0]}
    out2 = r._extract_raw_samples(res2)
    assert out2["tpot_ms"] == [pytest.approx((50.0 + 800.0 - 50.0) / 80.0)]


def test_serving_tpot_falls_back_and_says_so():
    """Item 2: no server usage -> chunk mean, counted as a fallback (never labelled per token)."""
    r = _serving()
    res = {"raw_ttfts_ms": [50.0, 50.0], "raw_itl_blocks_ms": [[30.0] * 3, [40.0] * 20],
           "raw_e2els_ms": [140.0, 2000.0], "output_lens": [4, 81],
           "output_lens_usage": [None, 81]}
    out = r._extract_raw_samples(res)
    assert out["tpot_ms"][0] == pytest.approx(30.0) and out["tpot_fallbacks"] == 1
    assert out["tpot_ms"][1] == pytest.approx(1950.0 / 80.0)


def _remote_runner_no_vllm(monkeypatch, dataset):
    from gbench.core.config import BenchmarkConfig
    from gbench.runners.serving import ServingBenchmarkRunner
    import gbench.utils as U
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = "http://127.0.0.1:11434/v1"
    cfg.dataset = dataset
    cfg.input_lengths, cfg.output_lengths = [300], [64]
    r = ServingBenchmarkRunner(cfg)
    monkeypatch.setattr(U, "check_vllm_available", lambda: False)
    monkeypatch.setattr(U, "safe_get_tokenizer", lambda *a, **k: TOK)
    monkeypatch.setattr(r, "_tokenizer_path", lambda m, f: "tok")
    return r


def test_remote_text_without_vllm_builds_sized_coherent_prompts(monkeypatch):
    """A remote run without vLLM still gets the tokenizer-sized coherent prompts (never a
    repeated filler sentence of the wrong length)."""
    r = _remote_runner_no_vllm(monkeypatch, "random")
    reqs, tok = r._generate_sample_requests(None, None, 4)
    assert tok is TOK and len(reqs) == 4
    assert all(q.prompt_len == 300 and len(q.prompt) == 300 for q in reqs)
    assert "quick brown fox" not in reqs[0].prompt
    assert len({q.prompt for q in reqs}) == 4


@pytest.mark.parametrize("dataset", ["custom", "hf", "sharegpt"])
def test_remote_real_dataset_without_vllm_is_an_error_not_a_substitute(monkeypatch, dataset):
    r = _remote_runner_no_vllm(monkeypatch, dataset)
    with pytest.raises(RuntimeError, match="vLLM's dataset loaders"):
        r._generate_sample_requests(None, None, 4)


@pytest.mark.parametrize("dataset", ["random", "custom"])
def test_missing_tokenizer_is_a_config_error_that_reaches_the_user(monkeypatch, dataset):
    """No tokenizer: one clear error, raised through the per-iteration handlers (not
    swallowed into "All iterations failed")."""
    import gbench.utils as U
    from gbench.runners.serving import ServingConfigError
    r = _remote_runner_no_vllm(monkeypatch, dataset)
    monkeypatch.setattr(U, "check_vllm_available", lambda: True)
    monkeypatch.setattr(U, "safe_get_tokenizer", lambda *a, **k: None)
    r.config.dataset_path = "x.jsonl"
    with pytest.raises(ServingConfigError, match="tokenizer"):
        r._generate_sample_requests(None, None, 4)
    monkeypatch.setattr(r, "_metrics_snapshot", lambda: None)
    with pytest.raises(ServingConfigError, match="tokenizer"):
        r._run_text_with_iterations(None, None, batch_size=1, num_prompts=4)
