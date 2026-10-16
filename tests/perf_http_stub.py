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

"""A small OpenAI-compatible streaming server for the perf pillars' real-process tests
(test helper; runs in a background thread, so client worker processes reach it over
TCP like a real endpoint).

Per request it waits ``prefill_ms`` (scaled clock: milliseconds, so tests stay fast),
emits the first token, then decodes at ``per_token_ms`` per token. Engine steps carry 1
token, or ``chunk_tokens`` under ``spec_decode``; the SSE stream coalesces
``chunk_tokens`` tokens per chunk (like vLLM's stream interval). ``slots`` caps
concurrent decoding (FIFO), which gives the server a finite capacity. In ``vllm`` mode it
also serves a vLLM-like /metrics (request/generation/prompt counters updated as the work
happens, running/waiting gauges, an inter-token latency histogram with one observation
per engine step) and the final include_usage chunk; with ``vllm=False`` it has neither,
like many non-vLLM servers.
POST /admin/foreign?n=K adds K finished requests nobody here sent ("foreign" traffic).
"""

import asyncio
import json
import socket
import threading
import time
from typing import Optional

ITL_EDGES = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class StubServer:
    def __init__(self, prefill_ms: float = 5.0, per_token_ms: float = 2.0,
                 chunk_tokens: int = 1, slots: Optional[int] = None, vllm: bool = True,
                 honor_ignore_eos: bool = True, spec_decode: bool = False):
        self.prefill_ms, self.per_token_ms = prefill_ms, per_token_ms
        self.chunk_tokens, self.slots, self.vllm = max(1, chunk_tokens), slots, vllm
        self.honor_ignore_eos, self.spec_decode = honor_ignore_eos, spec_decode
        self.tokens_per_step = self.chunk_tokens if spec_decode else 1
        self.c = {"generation_tokens": 0.0, "prompt_tokens": 0.0, "spec_draft_tokens": 0.0}
        self.success = {"length": 0.0, "stop": 0.0}
        self.running = 0
        self.waiting = 0
        self.itl_buckets = [0.0] * (len(ITL_EDGES) + 1)
        self.itl_count = 0.0
        self.requests_seen = 0
        self._loop = None
        self._thread = None
        self.port = None

    # ── handlers ──────────────────────────────────────────────────────────────
    def _observe_itl(self, gap_s: float) -> None:
        self.itl_count += 1
        for k, le in enumerate(ITL_EDGES):
            if gap_s <= le:
                self.itl_buckets[k] += 1
        self.itl_buckets[-1] += 1

    def _metrics_text(self) -> str:
        lab = 'engine="0",model_name="stub"'
        lines = [f'vllm:request_success_total{{{lab},finished_reason="{k}"}} {v}'
                 for k, v in self.success.items()]
        lines += [f'vllm:generation_tokens_total{{{lab}}} {self.c["generation_tokens"]}',
                 f'vllm:prompt_tokens_total{{{lab}}} {self.c["prompt_tokens"]}',
                 f'vllm:num_requests_running{{{lab}}} {float(self.running)}',
                 f'vllm:num_requests_waiting{{{lab}}} {float(self.waiting)}',
                 f'vllm:prefix_cache_hits_total{{{lab}}} 0.0',
                 f'vllm:prefix_cache_queries_total{{{lab}}} {self.c["prompt_tokens"]}']
        if self.spec_decode:
            lines.append(f'vllm:spec_decode_num_draft_tokens_total{{{lab}}} {self.c["spec_draft_tokens"]}')
        for le, n in zip(ITL_EDGES, self.itl_buckets):
            lines.append(f'vllm:inter_token_latency_seconds_bucket{{{lab},le="{le}"}} {n}')
        lines.append(f'vllm:inter_token_latency_seconds_bucket{{{lab},le="+Inf"}} {self.itl_buckets[-1]}')
        lines.append(f'vllm:inter_token_latency_seconds_count{{{lab}}} {self.itl_count}')
        return "\n".join(lines) + "\n"

    async def _completion(self, request):
        from aiohttp import web
        body = await request.json()
        chat = "messages" in body
        n_out = int(body.get("max_tokens") or 16)
        if not (self.honor_ignore_eos and body.get("ignore_eos")):
            n_out = max(1, n_out // 2)                     # stops early, like a real EOS
        prompt = json.dumps(body.get("messages") or body.get("prompt") or "")
        n_in = max(1, len(prompt) // 4)
        self.requests_seen += 1
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        self.waiting += 1
        try:
            if self._sem is not None:
                await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        try:
            await asyncio.sleep(self.prefill_ms / 1000.0)
            self.c["prompt_tokens"] += n_in
            sent = 0
            k = self.chunk_tokens
            while sent < n_out:
                # First token right after prefill; then chunks of k tokens, each k decode
                # tokens long (k engine steps, or one k-token step under spec decode).
                take = 1 if sent == 0 else min(k, n_out - sent)
                if sent:
                    gap_s = take * self.per_token_ms / 1000.0
                    await asyncio.sleep(gap_s)
                    if self.vllm:
                        steps = max(1, take // self.tokens_per_step)
                        for _ in range(steps):
                            self._observe_itl(gap_s / steps)
                sent += take
                self.c["generation_tokens"] += take
                if self.spec_decode and sent > 1:
                    self.c["spec_draft_tokens"] += take
                text = "tok " * take
                choice = ({"index": 0, "delta": {"content": text}} if chat
                          else {"index": 0, "text": text})
                finished = sent >= n_out
                if finished:
                    choice["finish_reason"] = "length" if n_out == body.get("max_tokens") else "stop"
                await resp.write(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())
            if self.vllm and (body.get("stream_options") or {}).get("include_usage"):
                usage = {"prompt_tokens": n_in, "completion_tokens": n_out,
                         "total_tokens": n_in + n_out}
                await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            self.success["length" if n_out == body.get("max_tokens") else "stop"] += 1
        finally:
            self.running -= 1
            if self._sem is not None:
                self._sem.release()
        return resp

    async def _metrics(self, request):
        from aiohttp import web
        if not self.vllm:
            return web.Response(status=404)
        return web.Response(text=self._metrics_text())

    async def _models(self, request):
        from aiohttp import web
        return web.json_response({"data": [{"id": "stub", "owned_by": "vllm" if self.vllm else "other",
                                            "max_model_len": 8192}]})

    async def _foreign(self, request):
        from aiohttp import web
        n = int(request.query.get("n", "1"))
        self.success["length"] += n
        return web.json_response({"added": n})

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> str:
        from aiohttp import web
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))               # bound before the server uses it: no race
        self.port = sock.getsockname()[1]
        ready = threading.Event()
        self._error = None

        def _run():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._sem = asyncio.Semaphore(self.slots) if self.slots else None
                app = web.Application()
                app.router.add_post("/v1/completions", self._completion)
                app.router.add_post("/v1/chat/completions", self._completion)
                app.router.add_get("/metrics", self._metrics)
                app.router.add_get("/v1/models", self._models)
                app.router.add_post("/admin/foreign", self._foreign)
                self._runner = web.AppRunner(app, shutdown_timeout=1.0)
                self._loop.run_until_complete(self._runner.setup())
                self._loop.run_until_complete(web.SockSite(self._runner, sock).start())
            except BaseException as e:            # noqa: BLE001 - surfaced by start()
                self._error = e
                ready.set()
                return
            ready.set()
            self._loop.run_forever()
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not ready.wait(10):
            raise RuntimeError("stub server did not start within 10 s")
        if self._error is not None:
            raise RuntimeError(f"stub server failed to start: {self._error}")
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        if self._loop is None or self._thread is None:
            return
        loop = self._loop

        def _halt():
            for task in asyncio.all_tasks(loop):
                task.cancel()                     # in-flight streams end now
            loop.stop()
        loop.call_soon_threadsafe(_halt)
        self._thread.join(10)
        if self._thread.is_alive():
            raise RuntimeError("stub server thread did not stop")
