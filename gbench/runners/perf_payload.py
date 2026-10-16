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

"""Unique-by-construction request content for the serving and stress pillars.

A performance number is only trustworthy if every request is new to the server's
caches (prefix/KV cache, multimodal processor and encoder caches). Each request
therefore carries:

* a nonce at the very start of its user text (before any image and any shared
  filler), built from a per-invocation salt plus (phase, rep, point, index), so two
  runs never share a prefix beyond the system prompt and chat template;
* for multimodal, images whose encoded bytes are unique (a request code written into
  whole 8x8 blocks at full amplitude, which survives JPEG encoding);
* seeded content and lengths: the reproducibility seed fixes the planned lengths,
  base images and base tokens, and the salt only changes bytes.

Derived seeds use blake2b (stable across interpreter processes), never hash().
"""

import hashlib
import io
import secrets
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# One-letter phase codes keep the nonce short and its shape fixed.
PHASES: Dict[str, str] = {
    "warmup": "w", "preflight": "p", "probe": "b", "sweep": "s", "refine": "r",
    "drain": "d", "calibrate": "c", "serving": "v", "serving_warmup": "u", "peak": "k",
}

DEFAULT_SEED = 83


def stable_u32(*key) -> int:
    """A 32-bit seed derived from ``key``; identical in every process and run."""
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=4).digest(), "big")


def stable_u64(*key) -> int:
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "big")


def new_salt() -> str:
    """A fresh cache-busting salt (16 hex chars), drawn once per invocation."""
    return secrets.token_hex(8)


def special_token_count(tokenizer) -> int:
    """Tokens the tokenizer adds around a prompt (e.g. BOS). vLLM's RandomDataset counts
    them inside --input-len, so the random-token text is that much shorter."""
    fn = getattr(tokenizer, "num_special_tokens_to_add", None)
    try:
        return int(fn()) if callable(fn) else 0
    except Exception:                                                     # noqa: BLE001
        return 0


def _ids(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def decode_to_target_len(tokenizer, ids: List[int], target: int, rng: np.random.Generator,
                         max_retry: int = 10) -> Tuple[str, int]:
    """Decode ``ids`` and re-encode until the text is exactly ``target`` tokens.

    Same procedure as vLLM's gen_prompt_decode_to_target_len (some token sequences
    re-tokenize to a different count): truncate or extend at the END, so the nonce at
    the front survives. Returns (text, mismatch); mismatch = final - target when the
    retry budget runs out, else 0.
    """
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    seq = list(ids)
    for _ in range(max_retry + 1):
        text = tokenizer.decode(seq)
        seq = _ids(tokenizer, text)
        if len(seq) == target:
            return text, 0
        if len(seq) > target:
            seq = seq[:target]
        elif vocab > 0:
            extra = [int(t) for t in rng.integers(0, vocab, size=target - len(seq))
                     if int(t) not in special]
            seq = seq + extra
        else:
            break
    text = tokenizer.decode(seq)
    return text, len(_ids(tokenizer, text)) - target


class PayloadFactory:
    """Builds the request content for one gbench invocation.

    ``seed`` reproduces lengths, base tokens and base images. ``salt`` (fresh per
    invocation unless given) makes the bytes new. Both are recorded in the result.
    """

    def __init__(self, seed: Optional[int] = None, salt: Optional[str] = None):
        self.seed = DEFAULT_SEED if seed is None else int(seed)
        self.salt = salt or new_salt()
        self._image_hashes: set = set()
        self._base_images: Dict[int, np.ndarray] = {}
        self._allowed: Dict[int, np.ndarray] = {}
        self._short_warned = False
        # Which pillar call (one serving batch size / campaign / model, one stress run)
        # the content belongs to. Runners bump it with new_call() so two calls of one
        # invocation never share a request.
        self.call = 0

    def new_call(self) -> int:
        self.call += 1
        return self.call

    def record(self) -> dict:
        return {"payload_seed": self.seed, "payload_salt": self.salt}

    # ── nonce ────────────────────────────────────────────────────────────────
    def nonce(self, phase: str, rep: int, point: int, i: int) -> str:
        """Nonce for one request; goes at the very start of its user text.

        It LEADS with an 8-hex code hashed from (salt, call, phase, rep, point, i), so
        two requests diverge within their first token or two after the system prompt
        and chat template; the readable fields follow for debugging. (A readable head
        first, e.g. "Run <salt> req", would be a 20-30 token prefix shared by every
        request of the run, which the prefix cache then serves.)"""
        code = hashlib.blake2b(repr((self.salt, self.call, phase, rep, point, i)).encode(),
                               digest_size=4).hexdigest()
        return (f"{code} {PHASES[phase]}{self.call}.{rep % 100:02d}.{point % 1000:03d}."
                f"{i % 1000000:06d}. ")

    def rng(self, *key) -> np.random.Generator:
        """A generator seeded from the reproducibility seed (and the call), never the salt."""
        return np.random.default_rng(stable_u32(self.seed, self.call, *key))

    def _allowed_tokens(self, tokenizer) -> np.ndarray:
        key = id(tokenizer)
        if key not in self._allowed:
            special = set(getattr(tokenizer, "all_special_ids", []) or [])
            vocab = int(tokenizer.vocab_size)
            self._allowed[key] = np.array([t for t in range(vocab) if t not in special])
        return self._allowed[key]

    def _check_room(self, planned_len: int, nonce_len: int) -> None:
        if planned_len < nonce_len and not self._short_warned:
            self._short_warned = True
            import logging
            logging.getLogger(__name__).warning(
                f"input length {planned_len} tokens is shorter than the per-request nonce "
                f"({nonce_len} tokens): prompts are cut inside the nonce and may repeat.")

    # ── lengths ──────────────────────────────────────────────────────────────
    def planned_lengths(self, n: int, input_len: int, range_ratio: float,
                        phase: str, rep: int, point: int) -> List[int]:
        """Per-request input lengths, uniform in [in·(1-r), in·(1+r)]; seed-only."""
        if range_ratio <= 0:
            return [int(input_len)] * n
        lo = max(1, int(np.floor(input_len * (1.0 - range_ratio))))
        hi = max(lo, int(np.ceil(input_len * (1.0 + range_ratio))))
        return [int(x) for x in self.rng("len", phase, rep, point).integers(lo, hi + 1, size=n)]

    # ── text ─────────────────────────────────────────────────────────────────
    def random_token_prompt(self, tokenizer, planned_len: int, phase: str, rep: int,
                            point: int, i: int) -> Tuple[str, int]:
        """vLLM-RandomDataset-style prompt (a run of consecutive allowed token ids from
        a seeded offset) of exactly ``planned_len`` tokens, nonce first.
        Returns (text, mismatch)."""
        rng = self.rng("tok", planned_len, phase, rep, point, i)
        nonce_ids = _ids(tokenizer, self.nonce(phase, rep, point, i))
        self._check_room(planned_len, len(nonce_ids))
        allowed = self._allowed_tokens(tokenizer)
        n_body = max(0, planned_len - len(nonce_ids))
        off = int(rng.integers(0, len(allowed)))
        body = allowed[(off + np.arange(n_body)) % len(allowed)].tolist()
        return decode_to_target_len(tokenizer, nonce_ids + body, planned_len, rng)

    def filler_prompt(self, tokenizer, planned_len: int, filler: str, phase: str, rep: int,
                      point: int, i: int, suffix: str = "") -> Tuple[str, int]:
        """Coherent text prompt of ``planned_len`` tokens: nonce + filler (rotated by
        the seed) + ``suffix`` (e.g. a long-output directive). Returns (text, mismatch).
        Without a tokenizer the length is approximate (mismatch None)."""
        words = filler.split()
        rot = int(self.rng("rot", phase, rep, point, i).integers(0, max(1, len(words))))
        body = " ".join(words[rot:] + words[:rot])
        nonce = self.nonce(phase, rep, point, i)
        if tokenizer is None:
            reps = max(1, planned_len // max(1, len(words)) + 1)
            return nonce + (body + " ") * reps + suffix, None
        n_ids = _ids(tokenizer, nonce)
        self._check_room(planned_len, len(n_ids))
        s_ids = _ids(tokenizer, suffix) if suffix else []
        tile = _ids(tokenizer, body + " ")
        budget = max(0, planned_len - len(n_ids) - len(s_ids))
        text, mismatch = "", 0
        for _ in range(8):
            # Tile boundaries can merge on re-tokenization; correct the budget by the
            # observed mismatch until the whole prompt is exactly planned_len.
            b_ids = (tile * (budget // max(1, len(tile)) + 1))[:max(0, budget)]
            text = tokenizer.decode(n_ids + b_ids) + suffix
            mismatch = len(_ids(tokenizer, text)) - planned_len
            if mismatch == 0:
                break
            budget -= mismatch
        return text, mismatch

    # ── images ───────────────────────────────────────────────────────────────
    def base_image(self, j: int, size: int = 256) -> np.ndarray:
        if j not in self._base_images:
            self._base_images[j] = self.rng("img", j).integers(
                0, 256, size=(size, size, 3), dtype=np.uint8)
        return self._base_images[j]

    def unique_image_jpeg(self, phase: str, rep: int, point: int, i: int, k: int,
                          n_base: int = 50, quality: int = 85) -> bytes:
        """JPEG bytes of base image (i*imgs+k) % n_base with a 64-bit request code
        written into whole 8x8 blocks (black/white, two rows of 32 blocks), so the
        code survives JPEG quantization and the encoded bytes are unique. Raises if
        the bytes repeat an image already built by this factory."""
        from PIL import Image

        arr = self.base_image((i * 7 + k) % n_base).copy()
        code = stable_u64(self.salt, self.call, phase, rep, point, i, k)
        h, w = arr.shape[:2]
        per_row = w // 8
        for bit in range(64):
            r, c = divmod(bit, per_row)
            val = 255 if (code >> bit) & 1 else 0
            arr[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8, :] = val
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        digest = hashlib.blake2b(data, digest_size=16).digest()
        if digest in self._image_hashes:
            raise RuntimeError(f"payload image repeated for (call {self.call}, {phase}, rep {rep}, "
                               f"point {point}, request {i}, image {k}): a caller reused a "
                               f"request key (missing new_call()?)")
        self._image_hashes.add(digest)
        return data


def chat_messages(user_content, system_prompt: Optional[str]) -> list:
    """The one message layout every perf request uses: the no-think system prompt
    (only when given, i.e. no-think mode), then the user turn."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def payload_fingerprint(payload: dict) -> str:
    """Hash of everything that reaches the model (text + image bytes), for the
    uniqueness test."""
    h = hashlib.blake2b(digest_size=16)

    def _walk(x):
        if isinstance(x, dict):
            for k in sorted(x):
                if k in ("stream", "stream_options", "max_tokens", "ignore_eos"):
                    continue
                h.update(k.encode())
                _walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                _walk(v)
        else:
            h.update(repr(x).encode())

    _walk(payload)
    return h.hexdigest()


def assert_unique(payloads: Sequence[dict]) -> None:
    """Raise if any two payloads would reach the model with identical content."""
    seen = {}
    for idx, p in enumerate(payloads):
        fp = payload_fingerprint(p)
        if fp in seen:
            raise RuntimeError(f"duplicate perf payloads at {seen[fp]} and {idx}")
        seen[fp] = idx
