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

"""Serving benchmark runner using vLLM Python API directly."""

import asyncio
import json
import logging
import os
import requests
import shutil
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from ..core.config import (
    BenchmarkConfig,
    cuda_visible_devices_env,
    get_max_model_len,
    get_server_timeout,
    resolve_server_port,
    serve_args_for,
    serve_settings_record,
)
from ..core.models import ModelConfig, ModelFormat
from ..utils import process_groups
from ..utils.dependency_checks import VLLM_BIN_ENV, vllm_cli, vllm_engine_record
from ..utils.log_manager import write_redacted_json
from ..analysis.perf_method import (
    counter_delta, methodology_record, parse_metrics, per_token_tpot_ms,
)
from .perf_payload import PayloadFactory, chat_messages, special_token_count
from ..analysis.statistics import (
    aggregate_benchmark_results,
    compute_statistics,
    format_statistics_summary,
    summarize_latency_metric,
    validate_repeatability,
)

# vLLM benchmark imports will be loaded dynamically in active methods

logger = logging.getLogger(__name__)


def _tokenizer_trust(config) -> bool:
    """trust_remote_code for gbench's own tokenizer loads (core.config.tokenizer_trust_remote_code)."""
    from ..core.config import tokenizer_trust_remote_code
    return tokenizer_trust_remote_code(config)[0]

#: --max-logprobs for the eval and quality servers gbench starts: the top-logprobs count
#: of mmlu's canonical request (hendrycks/test evaluate.py, logprobs=100; mmlu.TOP_LOGPROBS).
EVAL_SERVER_MAX_LOGPROBS = 100

#: The address a server gbench launches binds, when set (e.g. 127.0.0.1 to keep the eval server
#: off the network, when no selected suite reaches it from a docker bridge). Unset: performance
#: servers bind PERF_SERVER_HOST; the quality / Golden Set server binds LOOPBACK_HOST when the run
#: selects no eval suite (quality and the Golden Set reach it from gbench's own process); with eval
#: suites selected it binds all interfaces (the vllm serve default), because suites whose task
#: containers run on docker's default bridge reach it at the bridge gateway (e.g. 172.17.0.1).
SERVER_HOST_ENV = "GBENCH_SERVER_HOST"
#: The loopback address: reached only from this host.
LOOPBACK_HOST = "127.0.0.1"
#: Performance servers are reached only by gbench on 127.0.0.1: binding there keeps outside
#: requests out of the latency and throughput measurements.
PERF_SERVER_HOST = LOOPBACK_HOST
#: How many ports an automatically chosen server port is retried with when another process takes
#: the port between gbench's check and the server's bind.
PORT_RACE_ATTEMPTS = 3


def server_host_env_error(value: Optional[str] = None) -> Optional[str]:
    """Why GBENCH_SERVER_HOST (or `value`) cannot be the bind address of a server gbench
    launches, or None when it can (unset, a loopback address: 127.0.0.0/8, ::1, localhost, or a
    wildcard: 0.0.0.0, ::). Every gbench client of such a server targets 127.0.0.1, so a server
    bound only to another address would be unreachable to every pillar."""
    import ipaddress
    raw = (os.environ.get(SERVER_HOST_ENV) if value is None else value) or ""
    host = raw.strip()
    if not host or host.lower() == "localhost":
        return None
    try:
        addr = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        addr = None
    if addr is not None and (addr.is_loopback or addr.is_unspecified):
        return None
    return (f"{SERVER_HOST_ENV}={host!r}: gbench reaches the server it launches at 127.0.0.1, "
            "so it must bind a loopback address (127.0.0.1, ::1, localhost) or all interfaces "
            "(0.0.0.0, ::); unset it for gbench's own choice")


def _configured_port() -> Optional[int]:
    """GBENCH_SERVER_PORT as an int when it is one, else None (the port is chosen at launch)."""
    raw = (os.environ.get("GBENCH_SERVER_PORT") or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _listening_inodes(port: int) -> Optional[Set[str]]:
    """The socket inodes listening on TCP ``port`` in this network namespace (/proc/net/tcp and
    tcp6, state 0A), or None when /proc cannot be read (not Linux)."""
    found: Set[str] = set()
    read_any = False
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="ascii") as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        read_any = True
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":
                continue
            try:
                if int(parts[1].rsplit(":", 1)[1], 16) == int(port):
                    found.add(parts[9])
            except (ValueError, IndexError):
                continue
    return found if read_any else None


def _session_socket_inodes(session_id: int) -> Optional[Set[str]]:
    """The socket inodes held by the processes of session ``session_id`` (a server gbench started
    with start_new_session=True leads its own session, so its API server processes, workers and
    EngineCore are all in it), or None when /proc cannot be read."""
    inodes: Set[str] = set()
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return None
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
                stat = f.read()
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[3]) != int(session_id):        # field 6 (session) after pid/comm
                continue
            fd_dir = f"/proc/{pid}/fd"
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if target.startswith("socket:["):
                    inodes.add(target[8:-1])
        except (OSError, ValueError, IndexError):
            continue
    return inodes


def listener_owned_by_session(port: int, session_id: int) -> Optional[bool]:
    """Whether every listener on TCP ``port`` belongs to session ``session_id``: True (only the
    server gbench started listens there), False (some other process listens on the port too:
    a foreign server, or a second SO_REUSEPORT listener splitting the traffic), None (nothing
    listens yet, or /proc cannot tell)."""
    listening = _listening_inodes(port)
    if not listening:
        return None
    ours = _session_socket_inodes(session_id)
    if ours is None:
        return None
    return listening <= ours


class ServingConfigError(RuntimeError):
    """A setup problem that fails every iteration the same way (no tokenizer, a dataset
    that needs vLLM): raised through the per-iteration handlers so the user sees it."""


def _require_tokenizer(tokenizer, name: str, what: str):
    if tokenizer is None:
        raise ServingConfigError(
            f"Could not load a tokenizer for '{name}' - {what} needs one. Pass --tokenizer "
            f"<hf-id-or-path> and install transformers (pip install transformers); a gated "
            f"model also needs Hugging Face auth (HF_TOKEN).")
    return tokenizer


class ServingBenchmarkRunner:
    """Runner for vLLM serving benchmarks using Python API."""

    # Multimodal request shape - kept in lockstep with StressTestRunner so all
    # three MM pillars measure the same campaign-sized workload: N images + text
    # padded to the campaign input length, output = campaign output length with
    # ignore_eos. Needs the server started with --limit-mm-per-prompt image=N.
    # The per-image token count is the server's own (measured by the MM preflight,
    # _assert_mm_images_processed), never a model family's constant.
    IMAGES_PER_REQUEST = 4
    MIN_MM_TEXT_TOKENS = 16
    # No-think system prompt (default). A reasoning model (e.g. gemma-4 on Ollama)
    # otherwise streams its chain-of-thought into the `reasoning` field and delays/
    # replaces `content` - inflating TTFT and, when the whole budget goes to
    # reasoning, producing 0-content "empty" replies. gbench sends no system
    # message, so the backend injects the model's default ("...with reasoning...")
    # system; sending OUR own system prompt is ONE no-think lever, but it is NOT
    # sufficient on every backend (Ollama's gemma reasons regardless over /v1), so the
    # perf path ALSO sends the portable switches below. --thinking skips both.
    NOTHINK_SYSTEM = ("Answer directly with your final response only. Do not show "
                      "any reasoning, analysis, or thinking. Follow the user's "
                      "length and formatting instructions.")

    def no_think_fields(self) -> dict:
        """Portable no-think request fields for the PERF path. The serving/stress
        workload is a controlled, fixed-length generation, so a reasoning prefix is
        uncontrolled noise that must not enter the measurement. NOTHINK_SYSTEM alone
        does not stop some backends, so also send the switches vLLM/SGLang honor;
        backends that don't support them ignore them (HTTP 200, no error). Returns {}
        under --thinking (then the model reasons on purpose)."""
        if getattr(self.config, "thinking", False):
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}

    def __init__(self, config: BenchmarkConfig):
        """Initialize the serving benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.server_process: Optional[subprocess.Popen] = None
        self._server_log_file = None  # File handle for server logs
        self._server_lock = None      # the endpoint lock held while our server runs
        # The port of the server this runner launched: chosen in _start_server
        # (resolve_server_port: GBENCH_SERVER_PORT when set and free, else the first free port
        # from 8000). Every consumer (the perf and eval URLs, the CLI's local_endpoint) reads it
        # after the start.
        self.server_port: Optional[int] = _configured_port()
        self.server_port_source: Optional[str] = None
        # The --host of the server this runner launched (None = all interfaces) and why
        # (_server_host_choice); whether that server enforces an endpoint key (_server_env).
        self.server_host: Optional[str] = None
        self.server_host_source: Optional[str] = None
        self.server_key_enforced: Optional[bool] = None
        # The served id and how it was resolved, per (model, format) (_resolve_model_id).
        self._served: Dict[tuple, Dict[str, Any]] = {}
        # The per-image token count the MM workload is sized and labelled with, and its source
        # ("measured" by the MM preflight, else the model's declared "config"/"processor" value).
        self._vision_tokens_per_image: Optional[int] = None
        self._vision_tokens_source: Optional[str] = None
        self._mm_image_dir = None  # Temp dir for generated images
        self._mm_image_paths = []  # Generated image file paths
        self._mm_tokenizer = None  # Lazily loaded; sizes MM padded-text prompt
        # Request content (perf_payload): the seed reproduces lengths/base tokens/base
        # images, a fresh salt per invocation makes every byte new. GBENCH_PERF_SEED
        # overrides the seed.
        self.payloads = PayloadFactory(seed=int(os.environ.get("GBENCH_PERF_SEED") or 83))
        self._is_vllm: Optional[bool] = None
        self._real_pool: list = []
        self._prompt_reuse = 0

    def _images_per_request(self) -> int:
        """Images per MM request: config override (--personal sets 1) else default 4."""
        return int(getattr(self.config, "mm_images_per_request", None) or self.IMAGES_PER_REQUEST)

    def _per_image_tokens(self, model: ModelConfig) -> tuple:
        """(tokens per image, source) the MM workload is sized and labelled with: the count the
        MM preflight measured on this server ("measured"), else the model's declared fixed count
        (its config or processor config). Raises when neither is known: sizing the text budget
        with a guessed count could overflow the context or mislabel the input length."""
        if self._vision_tokens_per_image:
            return int(self._vision_tokens_per_image), self._vision_tokens_source or "measured"
        declared = getattr(model, "vision_tokens_per_image", None)
        if isinstance(declared, int) and declared > 0:
            return declared, getattr(model, "vision_tokens_source", None) or "config"
        raise RuntimeError(
            f"the per-image token count of {model.name} is unknown: it declares no fixed count "
            "and the multimodal preflight did not measure one")

    def _build_mm_text(self, n_tokens: int, tokenizer) -> str:
        """Text prompt of ~n_tokens tokens used to pad a multimodal request."""
        base = "Analyze the attached images and the following context in detail. "
        if not tokenizer or n_tokens <= 0:
            return base * max(1, n_tokens // 12)  # rough char-based fallback
        filler = base * (max(1, n_tokens // 8) + 8)
        ids = tokenizer(filler).input_ids[:n_tokens]
        return tokenizer.decode(ids)

    PASSAGE = (
        "Consider how a large city keeps itself supplied with fresh water. "
        "Reservoirs collect rainfall across a wide watershed, aqueducts carry "
        "it toward the population, and treatment plants remove sediment and "
        "pathogens before distribution. Engineers balance demand against "
        "capacity, plan for droughts, and maintain aging pipes that leak more "
        "with every passing year. The same logic of collection, transport, "
        "treatment, and delivery reappears in power grids, food systems, and "
        "communication networks, each a quiet feat of coordination that most "
        "residents never pause to notice. "
    )

    def _text_prompt_parts(self, out_len: int) -> tuple:
        """(passage, directive) of the coherent chat prompt used against REMOTE endpoints,
        which may ignore ignore_eos (e.g. Ollama): a real passage padded to the input
        length plus an explicit long-output directive keeps the reply sustained while the
        input length stays controlled. The nonce leads it (perf_payload.filler_prompt)."""
        target_words = max(150, int(out_len * 0.5))
        directive = (
            f"\n\nUsing the passage above as context, write a detailed, "
            f"well-structured explanation of at least {target_words} words about "
            f"how complex systems stay reliable. Develop several distinct points."
        )
        return self.PASSAGE, directive

    def served_max_model_len(self) -> int:
        """The context window requests are sized against: what a remote endpoint reports
        in /v1/models for the resolved model, else the window gbench serves at."""
        if self.config.remote_endpoint:
            for rec in self._served.values():
                if rec.get("served_max_model_len"):
                    return int(rec["served_max_model_len"])
        return get_max_model_len(getattr(self.config, "max_model_len", None))

    def is_vllm_server(self) -> bool:
        """True for a server gbench launched, or a remote endpoint whose /v1/models says
        owned_by=vllm (so vLLM-only request fields like cache_salt are safe to send)."""
        if self._is_vllm is None:
            if not self.config.remote_endpoint:
                self._is_vllm = True
            else:
                try:
                    from ..utils.endpoint import list_served_models
                    models = list_served_models(self.config.remote_endpoint)
                    self._is_vllm = bool(models) and all(
                        str(m.get("owned_by", "")).lower() == "vllm" for m in models)
                except Exception:                                         # noqa: BLE001
                    self._is_vllm = False
        return self._is_vllm

    def _extra_body(self, real_dataset: bool, phase: str, rep: int) -> Dict[str, Any]:
        """vLLM-only request fields ({} on other servers): cache_salt, per run for
        synthetic content (already unique), per (phase, rep) for real-dataset prompts that
        may repeat; and skip_special_tokens False, so every generated token reaches the
        client as text (a reply of special tokens, e.g. <turn|> or <eos> under ignore_eos,
        would otherwise stream as empty chunks: no first token, no gaps)."""
        if not self.is_vllm_server():
            return {}
        salt = self.payloads.salt
        call = self.payloads.call
        return {"cache_salt": f"{salt}.{call}.{phase}.{rep}" if real_dataset
                else f"{salt}.{call}", "skip_special_tokens": False}

    def _metrics_snapshot(self) -> Optional[dict]:
        """One parsed /metrics scrape of the measured server, or None."""
        base = (self.config.remote_endpoint
                or (f"http://127.0.0.1:{self.server_port}" if self.server_port else None))
        if not base:
            return None
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        try:
            import urllib.request
            raw = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode()
        except Exception:                                                 # noqa: BLE001
            return None
        return parse_metrics(raw)

    @staticmethod
    def _cache_delta(before: Optional[dict], after: Optional[dict]) -> Dict[str, Any]:
        """Cache counters over one iteration (None when /metrics doesn't expose them)."""
        return {k: counter_delta(before, after, k)
                for k in ("prefix_cache_hits", "prefix_cache_queries",
                          "mm_cache_hits", "mm_cache_queries")}

    def _resolve_model_id(self, model: ModelConfig, format: ModelFormat) -> str:
        """The id every request of this runner sends as "model".

        A server gbench launched serves ``model.get_model_path(format)``: vllm serve registers
        its model argument as the id (--served-model-name is refused in --vllm-serve-args), and
        the readiness check verified the listing. A remote endpoint: resolve_served_model over
        its /v1/models listing (--served-model-id first, then the model's HF id, path, name and
        short name). An endpoint that lists none of them raises EndpointError, a harness error
        naming --served-model-id (never another model's id); one whose listing cannot be read
        gets the configured id, and the result records that."""
        key = (model.name, getattr(format, "value", str(format)))
        cached = self._served.get(key)
        if cached:
            return cached["served_model_id"]
        if not self.config.remote_endpoint:
            rec = {"served_model_id": model.get_model_path(format),
                   "served_model_id_source": "the model argument of the server gbench launched"}
        else:
            from ..utils.endpoint import resolve_served_model
            served = resolve_served_model(
                self.config.remote_endpoint,
                [model.hf_model_id, model.get_model_path(format), model.name, model.short_name],
                getattr(self.config, "served_model_id", None), local=False)
            rec = {"served_model_id": served.id, "served_model_id_source": served.source,
                   "served_max_model_len": served.max_model_len}
            logger.info(f"Remote endpoint serves {model.name} as '{served.id}' ({served.source})")
        self._served[key] = rec
        return rec["served_model_id"]

    def _tokenizer_path(self, model: ModelConfig, format: ModelFormat) -> str:
        """The tokenizer the prompts are built with: the model's HF id against a remote endpoint
        (whose served id may be a tag no tokenizer loads from; --tokenizer overrides both), else
        the path the server serves."""
        return model.hf_model_id if self.config.remote_endpoint else model.get_model_path(format)

    def _stamp_server(self, result: Dict[str, Any], model: ModelConfig,
                      format: ModelFormat) -> None:
        """Record on a result the served id (and how it was resolved) and the server it measured
        (server_record)."""
        rec = self.server_record(model, format)
        for key in ("served_model_id", "served_model_id_source"):
            if key in rec:
                result[key] = rec[key]
        result["server"] = rec
        # Request-content provenance (perf_payload).
        result.update(self.payloads.record())
        result["prompts_reused"] = self._prompt_reuse
        # The measurement-method stamp (perf_method).
        result.update(methodology_record("serving"))

    def _request_headers(self) -> Dict[str, str]:
        """The endpoint key header (VLLM_API_KEY, else OPENAI_API_KEY) for requests to the served
        model; {} when none is set. Sent from gbench's own process only."""
        from .eval_suites.base import endpoint_auth_headers
        return endpoint_auth_headers()

    def server_record(self, model: Optional[ModelConfig] = None,
                      format: Optional[ModelFormat] = None) -> Dict[str, Any]:
        """What a performance result records about the server it measured: for a server gbench
        launched, its port, its bind address and why (host_source), whether it enforces an
        endpoint key (endpoint_key_enforced; never the key), the GPUs and parallel layout, the
        operator's serve settings and the vLLM that ran; for a remote endpoint, the endpoint. Both
        record the served model id and how it was resolved."""
        rec: Dict[str, Any] = {}
        if self.config.remote_endpoint:
            rec["endpoint"] = self.config.remote_endpoint
        else:
            rec.update({
                "endpoint": (f"http://127.0.0.1:{self.server_port}/v1"
                             if self.server_port else None),
                "port": self.server_port,
                "port_source": self.server_port_source,
                "host": self.server_host,
                "host_source": self.server_host_source,
                "endpoint_key_enforced": self.server_key_enforced,
                "gpus": list(getattr(self.config, "selected_gpus", None) or []) or None,
                "tensor_parallel_size": getattr(self.config, "tensor_parallel_size", None),
                "data_parallel_size": getattr(self.config, "data_parallel_size", 1) or 1,
                "serve_settings": serve_settings_record(self.config),
                "vllm": vllm_engine_record(),
            })
        if model is not None and format is not None:
            served = self._served.get((model.name, getattr(format, "value", str(format))))
            if served:
                rec.update(served)
        return rec


    def _wait_for_gpu_memory(self, min_free_gb: Optional[float] = None, max_wait_seconds: int = 120):
        """Wait for GPU memory to be reclaimed by CUDA driver.

        Uses nvidia-smi to actively poll until sufficient memory is free.
        Respects CUDA_VISIBLE_DEVICES to only check the assigned GPUs.

        Args:
            min_free_gb: Minimum free memory required in GB. None => 80% of the
                GPU's total VRAM (scales to any card; the old fixed 70GB made this
                wait the full timeout on every <70GB GPU).
            max_wait_seconds: Maximum time to wait before giving up
        """
        if self.config.remote_endpoint:
            return

        # The GPUs this run's servers use: the ones select_gpus chose (by UUID), else the
        # environment's CUDA_VISIBLE_DEVICES, else all.
        selected = list(getattr(self.config, "selected_gpus", None) or [])
        gpu_ids = ",".join(selected) if selected else os.environ.get("CUDA_VISIBLE_DEVICES", "")

        if min_free_gb is None:
            _total = self._gpu_total_gb(gpu_ids)
            min_free_gb = 0.8 * _total if _total > 0 else 8.0

        logger.info(f"Waiting for GPU memory to be reclaimed (need {min_free_gb:.0f}GB free)...")

        # Build nvidia-smi command targeting only our assigned GPUs
        smi_cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        if gpu_ids:
            smi_cmd.extend(["--id=" + gpu_ids])

        for elapsed in range(0, max_wait_seconds, 5):
            try:
                result = subprocess.run(
                    smi_cmd,
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Check ALL assigned GPUs have enough free memory
                    free_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                    if free_values:
                        min_free_mb = min(free_values)
                        free_gb = min_free_mb / 1024
                        if free_gb >= min_free_gb:
                            logger.info(f"GPU memory ready: {free_gb:.1f}GB free (min across {len(free_values)} GPU(s))")
                            return
                        logger.info(f"  GPU memory: {free_gb:.1f}GB free (min), waiting... ({elapsed}s)")
            except Exception:
                pass  # nvidia-smi failed, fall back to fixed wait
            time.sleep(5)
        
        logger.warning(f"GPU memory wait timed out after {max_wait_seconds}s, proceeding anyway")

    @staticmethod
    def _gpu_total_gb(gpu_ids: str) -> float:
        """Total memory of the smallest of the given GPUs (nvidia-smi --id: indices or UUIDs,
        comma-separated; all GPUs when empty), in GB; 0.0 when nvidia-smi cannot be read."""
        cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        if gpu_ids:
            cmd.append("--id=" + gpu_ids)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                totals = [float(x) / 1024 for x in result.stdout.split() if x.strip()]
                if totals:
                    return min(totals)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return 0.0

    def _setup_mm_images(self, num_images: int = 50) -> str:
        """Generate synthetic images for multimodal serving benchmark.

        Creates a temp directory with random JPEG images.

        Args:
            num_images: Number of images to generate

        Returns:
            Path to the temp directory containing images
        """
        import numpy as np
        from PIL import Image

        img_dir = tempfile.mkdtemp(prefix="serving_mm_images_")
        self._mm_image_paths = []

        for i in range(num_images):
            # 256x256 random images - small enough for fast I/O,
            # large enough to exercise the vision encoder
            img = Image.fromarray(self.payloads.base_image(i))
            img_path = os.path.join(img_dir, f"serve_img_{i:03d}.jpg")
            img.save(img_path, format="JPEG", quality=85)
            self._mm_image_paths.append(img_path)

        logger.info(f"Generated {num_images} synthetic images in {img_dir}")
        self._mm_image_dir = img_dir
        return img_dir

    def _cleanup_mm_images(self):
        """Clean up temp image directory."""
        if self._mm_image_dir and os.path.exists(self._mm_image_dir):
            shutil.rmtree(self._mm_image_dir, ignore_errors=True)
            logger.info(f"Cleaned up MM image directory: {self._mm_image_dir}")
            self._mm_image_dir = None
            self._mm_image_paths = []

    def _generate_sample_requests(
        self,
        model: ModelConfig,
        format: ModelFormat,
        num_prompts: int,
        input_len: int = 512,
        output_len: int = 128,
        phase: str = "serving",
        rep: int = 0,
    ) -> tuple[list[Any], Any]:
        """Generate sample requests for benchmarking.

        Synthetic prompts are unique by construction (perf_payload): a nonce from the run
        salt and (phase, rep, index) leads each prompt, so no request repeats another of
        this run or any earlier run. Real datasets are never edited; each (phase, rep)
        gets its own disjoint slice where the dataset is large enough.

        Args:
            model: Model configuration
            format: Model format
            num_prompts: Number of prompts to generate
            input_len: Input token length
            output_len: Output token length

        Returns:
            Tuple of (list of SampleRequest, tokenizer)
        """
        from gbench.utils import check_vllm_available, require_vllm_engine

        if not check_vllm_available():
            if not self.config.remote_endpoint:
                require_vllm_engine("Serving benchmark datasets")
            # A remote run needs vLLM only to load a real dataset (its dataset loaders);
            # the synthetic workload below is built from the tokenizer alone. A real
            # dataset is never replaced by other text.
            if self.config.dataset in ("sharegpt", "custom", "hf"):
                what = ("--campaign chat-like (ShareGPT)" if self.config.dataset == "sharegpt"
                        else f"--dataset {self.config.dataset}")
                raise ServingConfigError(
                    f"{what} needs vLLM's dataset loaders: install vllm (pip install vllm), "
                    f"or use a synthetic workload (another campaign, or --dataset random).")

        if self.config.dataset == "sharegpt":
            from vllm.benchmarks.datasets import ShareGPTDataset
            from gbench.utils import safe_get_tokenizer
            tokenizer_path = self._tokenizer_path(model, format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer,
                trust_remote_code=_tokenizer_trust(self.config))
            _require_tokenizer(tokenizer, self.config.tokenizer or tokenizer_path,
                               "the serving workload")
            
            # Use ShareGPT default path convention or download
            sharegpt_path = getattr(self.config, "dataset_path", None) or os.environ.get("SHAREGPT_PATH", os.path.join(os.environ.get("GBENCH_CACHE_DIR") or os.path.expanduser("~/.cache/gbench"), "ShareGPT_V3_unfiltered_cleaned_split.json"))
            if not os.path.exists(sharegpt_path):
                # Download just-in-time if missing
                os.makedirs(os.path.dirname(sharegpt_path), exist_ok=True)
                import urllib.request
                logger.info(f"Downloading ShareGPT dataset to {sharegpt_path}...")
                urllib.request.urlretrieve("https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json", sharegpt_path)

            dataset = ShareGPTDataset(dataset_path=sharegpt_path)
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                no_oversample=True,      # never pad with silent repeats
            )
            return requests, tokenizer
        elif self.config.dataset == "custom":
            from vllm.benchmarks.datasets import CustomDataset
            from gbench.utils import safe_get_tokenizer
            tokenizer_path = self._tokenizer_path(model, format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer,
                trust_remote_code=_tokenizer_trust(self.config))
            _require_tokenizer(tokenizer, self.config.tokenizer or tokenizer_path,
                               "the serving workload")
            
            dataset_path = getattr(self.config, "dataset_path", None)
            if not dataset_path:
                raise ValueError("--dataset-path is required when using custom dataset")
                
            dataset = CustomDataset(dataset_path=dataset_path)
            out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                output_len=out_len,
                no_oversample=True,      # never pad with silent repeats
            )
            return requests, tokenizer
        elif self.config.dataset == "hf":
            from vllm.benchmarks.datasets import get_samples
            from gbench.utils import safe_get_tokenizer
            import argparse
            tokenizer_path = self._tokenizer_path(model, format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer,
                trust_remote_code=_tokenizer_trust(self.config))
            _require_tokenizer(tokenizer, self.config.tokenizer or tokenizer_path,
                               "the serving workload")
            
            dataset_path = getattr(self.config, "dataset_path", None)
            if not dataset_path:
                raise ValueError("--dataset-path is required when using hf dataset")
                
            args = argparse.Namespace(
                dataset_name="hf",
                dataset_path=dataset_path,
                hf_name=dataset_path,
                hf_split="train",
                hf_subset=None,
                disable_shuffle=False,
                seed=83,
                num_prompts=num_prompts,
                custom_output_len=self.config.output_lengths[0] if self.config.output_lengths else 128,
                skip_chat_template=False,
                request_id_prefix="",
                no_oversample=True,
            )
            requests = get_samples(args, tokenizer)
            return requests, tokenizer
        else:
            from gbench.utils import safe_get_tokenizer

            tokenizer_path = self._tokenizer_path(model, format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer,
                trust_remote_code=_tokenizer_trust(self.config))
            _require_tokenizer(tokenizer, self.config.tokenizer or tokenizer_path,
                               "the serving workload")

            # Use config's input/output lengths
            in_len = self.config.input_lengths[0] if self.config.input_lengths else input_len
            out_len = self.config.output_lengths[0] if self.config.output_lengths else output_len

            # REMOTE endpoints may ignore ignore_eos (e.g. Ollama), so gibberish
            # RandomDataset prompts draw empty / immediate-EOS replies. Use a
            # COHERENT prompt (real passage padded to in_len + an explicit long-
            # output directive), varied per request, so replies are sustained.
            # LOCAL vLLM honors ignore_eos, so it KEEPS the validated RandomDataset
            # controlled-shape workload below - the local/large-machine path is
            # untouched.
            if self.config.remote_endpoint:
                from dataclasses import dataclass

                @dataclass
                class _TextRequest:
                    prompt: str
                    prompt_len: int
                    expected_output_len: int
                    multi_modal_data: Optional[dict] = None

                passage, directive = self._text_prompt_parts(out_len)
                reqs = []
                for i in range(num_prompts):
                    text, mism = self.payloads.filler_prompt(
                        tokenizer, int(in_len), passage, phase, rep, 0, i, suffix=directive)
                    if i == 0 and mism:
                        logger.warning(
                            f"--input-lengths {in_len} is below the coherent-prompt floor "
                            f"(nonce + long-output directive); the actual input is "
                            f"{int(in_len) + mism} tokens.")
                    reqs.append(_TextRequest(prompt=text, prompt_len=int(in_len) + (mism or 0),
                                             expected_output_len=out_len))
                return reqs, tokenizer

            # range_ratio=0.0 => every request is EXACTLY (in_len, out_len).
            # For a latency measurement this is the right design: it isolates the
            # shape so the reported percentiles reflect system/queueing variance,
            # not input-length variance (which would smear the tails and make two
            # models incomparable). It also means all requests share one output
            # length, so pooled ITL/TPOT percentiles are clean. Deterministic
            # shape also removes the 1.5x tail, so max_model_len only needs to
            # clear (in_len+out_len) - campaign_ctx() sizes it with slack.
            # Same token construction as vLLM's RandomDataset (a run of consecutive
            # allowed token ids from a seeded offset, decoded to exactly in_len tokens),
            # with the nonce first so no two prompts share a prefix.
            from vllm.benchmarks.datasets import SampleRequest
            requests = []
            for i in range(num_prompts):
                text, _mism = self.payloads.random_token_prompt(
                    tokenizer, max(1, int(in_len) - special_token_count(tokenizer)),
                    phase, rep, 0, i)
                requests.append(SampleRequest(prompt=text, prompt_len=int(in_len),
                                              expected_output_len=int(out_len)))
            return requests, tokenizer

    async def _run_mm_serving_benchmark(
        self,
        model: ModelConfig,
        format: ModelFormat,
        max_concurrency: int,
        num_prompts: int,
        phase: str = "serving",
        rep: int = 0,
    ) -> dict:
        """Run multimodal serving benchmark via aiohttp + SSE.

        Every request is built before the clock starts and is new to the server's
        caches: a nonce leads its text and each image carries a per-request code
        (perf_payload), so no image is ever served from the processor/encoder cache.

        Sends chat completion requests with images, measures TTFT, TPOT, ITL
        from SSE streaming response tokens.

        Args:
            model: Model configuration
            format: Model format
            max_concurrency: Maximum concurrent requests (from batch_size)
            num_prompts: Number of requests to send

        Returns:
            Dictionary with serving metrics matching text serving output format
        """
        import aiohttp

        model_path = self._resolve_model_id(model, format)
        base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"
        url = base_url.rstrip("/")
        api_url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"

        # Honor the campaign shape (mirror the MM stress path): N images + text
        # padded to the campaign input length, output = campaign output length
        # with ignore_eos. Without this every campaign sent the same
        # 1-image/128-token stub, so all MM serving rows measured one workload
        # mislabeled per campaign.
        in_len = self.config.input_lengths[0] if self.config.input_lengths else 1024
        out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
        max_tokens = max(int(out_len), 16)
        n_img = self._images_per_request()
        per_img, per_img_source = self._per_image_tokens(model)
        img_tokens = n_img * per_img
        max_len = self.served_max_model_len()
        # Full campaign text with images ON TOP (not replacing text): MM = same
        # text as the text row + 4 images, so it is a strict superset and directly
        # comparable. Clamp only if it would overflow the context window.
        target_text = min(int(in_len), max_len - img_tokens - max_tokens - 128)
        target_text = max(self.MIN_MM_TEXT_TOKENS, target_text)
        if self._mm_tokenizer is None:
            from gbench.utils import safe_get_tokenizer
            tok_path = self._tokenizer_path(model, format)
            self._mm_tokenizer = safe_get_tokenizer(tok_path, custom_tokenizer=self.config.tokenizer,
                trust_remote_code=_tokenizer_trust(self.config))
            _require_tokenizer(self._mm_tokenizer, self.config.tokenizer or tok_path,
                               "multimodal serving (it sizes the request text)")
        media = None
        if not self.config.remote_endpoint and self._mm_image_dir:
            media = os.path.join(self._mm_image_dir, f"{phase}_{rep:02d}")
            os.makedirs(media, exist_ok=True)
        extra = self._extra_body(False, phase, rep)
        sys_prompt = None if getattr(self.config, "thinking", False) else self.NOTHINK_SYSTEM
        payloads = []
        for request_id in range(num_prompts):
            text, _m = self.payloads.filler_prompt(
                self._mm_tokenizer, int(target_text),
                "Analyze the attached images and the following context in detail. ",
                phase, rep, 0, request_id)
            image_urls = []
            for k in range(n_img):
                data = self.payloads.unique_image_jpeg(phase, rep, 0, request_id, k)
                if media is None:
                    import base64
                    image_urls.append("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
                else:
                    path = os.path.join(media, f"r{request_id:06d}_{k}.jpg")
                    with open(path, "wb") as fh:
                        fh.write(data)
                    image_urls.append(f"file://{path}")
            content = [{"type": "text", "text": text}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
            payload = {
                "model": model_path,
                "messages": chat_messages(content, sys_prompt),
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stream": True,
                # Ask the server for token-accurate usage in the final SSE chunk so
                # output length is real tokens (matching the text path's tokenizer
                # accounting), not a count of non-empty content deltas.
                "stream_options": {"include_usage": True},
            }
            payload.update(self.no_think_fields())   # portable no-think for the perf workload
            payload.update(extra)
            payloads.append(payload)

        semaphore = asyncio.Semaphore(max_concurrency)
        all_ttfts = []       # Time to first token (ms), one per request
        all_tpots = []       # Mean inter-token latency (ms), one per request
        all_itls = []        # Inter-token latencies (ms), flattened (back-compat)
        all_itl_blocks = []  # Inter-token latencies (ms) grouped per request
        all_e2els = []       # End-to-end latencies (ms), one per request
        all_out_lens = []    # Output tokens, one per request
        all_usage_lens = []  # Server-reported output tokens (None when absent)
        all_prompt_lens = [] # Server-measured input (prompt) tokens, one per request
        completed = 0
        empty = 0            # HTTP 200 but the model produced 0 content tokens
        failed = 0           # transport / HTTP errors
        error_reasons = []   # reason string per failed request
        total_output_tokens = 0

        async def send_mm_request(session, request_id, pbar=None):
            nonlocal completed, empty, failed, total_output_tokens
            payload = payloads[request_id]
            async with semaphore:
                start = time.perf_counter()
                content_times = []  # arrival time of each REAL content token
                try:
                    async with session.post(api_url, json=payload) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.debug(
                                f"MM request {request_id} HTTP {resp.status}: "
                                f"{error_text[:200]}"
                            )
                            failed += 1
                            error_reasons.append(f"http_{resp.status}")
                            return
                        # Count only chunks carrying real generated text, not the
                        # role-delta / finish control chunks (mirrors the text
                        # path at _run_lightweight_remote_benchmark), so TPOT and
                        # output throughput aren't biased by 1-2 empty SSE lines.
                        usage_tokens = None
                        usage_prompt = None
                        n_data_chunks = 0
                        finish_reason = None
                        choice_keys = set()
                        delta_keys = set()
                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line.startswith("data:") or "[DONE]" in line:
                                continue
                            n_data_chunks += 1
                            try:
                                data = json.loads(line[5:].strip())
                                choices = data.get("choices") or []
                                if choices:
                                    choice_keys.update(choices[0].keys())
                                    _d = choices[0].get("delta") or {}
                                    delta_keys.update(_d.keys())
                                    if choices[0].get("finish_reason"):
                                        finish_reason = choices[0].get("finish_reason")
                                    # Count content OR reasoning/thinking tokens
                                    # (see text path): an all-reasoning reply is
                                    # real decode, not an empty completion.
                                    delta = (_d.get("content") or _d.get("reasoning")
                                             or _d.get("reasoning_content") or "")
                                else:
                                    delta = ""
                                # Final chunk (include_usage) carries token-accurate
                                # usage with an empty choices list.
                                usage = data.get("usage")
                                if usage and usage.get("completion_tokens") is not None:
                                    usage_tokens = int(usage["completion_tokens"])
                                if usage and usage.get("prompt_tokens") is not None:
                                    usage_prompt = int(usage["prompt_tokens"])
                            except Exception:
                                delta = ""
                            if delta:
                                content_times.append(time.perf_counter())

                    end = time.perf_counter()

                    if not content_times:
                        # HTTP 200 with 0 content chunks: an EMPTY completion.
                        # Diagnose WHY: completion_tokens>0 => tokens generated but
                        # missed (parse bug; keys show where); ==0 => genuinely empty.
                        logger.warning(
                            "EMPTY MM reply diag (req %s): usage.completion_tokens=%s "
                            "finish_reason=%s data_chunks=%s choice_keys=%s delta_keys=%s",
                            request_id, usage_tokens, finish_reason, n_data_chunks,
                            sorted(choice_keys), sorted(delta_keys))
                        empty += 1
                        return

                    ttft_ms = (content_times[0] - start) * 1000
                    e2el_ms = (end - start) * 1000
                    # Output length = server-reported tokens when available (accurate),
                    # else the count of non-empty content chunks (a chunk can carry several
                    # tokens, so that is a lower bound). TTFT/ITL timing uses content_times.
                    num_tokens = usage_tokens if usage_tokens is not None else len(content_times)
                    total_output_tokens += num_tokens

                    all_ttfts.append(ttft_ms)
                    all_e2els.append(e2el_ms)
                    all_out_lens.append(num_tokens)
                    all_usage_lens.append(usage_tokens)      # None: server sent no usage
                    if usage_prompt is not None:
                        all_prompt_lens.append(usage_prompt)

                    # Per-request gaps between streamed chunks (ms), kept grouped for
                    # block-bootstrap CIs (ITL is autocorrelated within a request) and
                    # flattened for the pooled percentiles. A chunk can carry several
                    # tokens, so TPOT is per token from the server's count instead.
                    req_itls_ms = [
                        (content_times[j] - content_times[j - 1]) * 1000
                        for j in range(1, len(content_times))
                    ]
                    all_itl_blocks.append(req_itls_ms)
                    if req_itls_ms:
                        all_itls.extend(req_itls_ms)
                    _tp = per_token_tpot_ms(e2el_ms, ttft_ms, usage_tokens)
                    if _tp is None and req_itls_ms:
                        _tp = sum(req_itls_ms) / len(req_itls_ms)   # no usage: chunk mean
                    if _tp is not None:
                        all_tpots.append(_tp)

                    completed += 1
                except Exception as e:
                    logger.debug(f"MM request {request_id} error: {e}")
                    failed += 1
                    error_reasons.append(
                        "timeout" if isinstance(e, asyncio.TimeoutError)
                        else type(e).__name__)
                finally:
                    if pbar:
                        pbar.update(1)

        connector = aiohttp.TCPConnector(limit=max_concurrency + 20)
        # aiohttp total = a TOTAL wall-clock cap on the whole request. Tunable via
        # --timeout (config.timeout); see config for the text-vs-MM semantics.
        timeout = aiohttp.ClientTimeout(total=getattr(self.config, "timeout", 600) or 600)
        from tqdm import tqdm
        try:
            with tqdm(total=num_prompts, desc="MM Serving") as pbar:
                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout, headers=self._request_headers()
                ) as session:
                    overall_start = time.perf_counter()
                    tasks = [
                        send_mm_request(session, i, pbar)
                        for i in range(num_prompts)
                    ]
                    await asyncio.gather(*tasks)
                    overall_elapsed = time.perf_counter() - overall_start
        finally:
            if media:
                shutil.rmtree(media, ignore_errors=True)   # never leave images on disk

        if not all_ttfts:
            raise RuntimeError(
                f"All {num_prompts} MM serving requests produced no tokens "
                f"(empty={empty}, error={failed})"
            )

        # Compute percentile metrics matching text serving output format
        import numpy as np
        ttft_arr = np.array(all_ttfts)
        e2el_arr = np.array(all_e2els)
        request_throughput = completed / overall_elapsed
        output_throughput = total_output_tokens / overall_elapsed

        reason_counts = {}
        for _rsn in error_reasons:
            reason_counts[_rsn] = reason_counts.get(_rsn, 0) + 1

        result = {
            "request_throughput": request_throughput,
            "output_throughput": output_throughput,
            "total_output_tokens": int(total_output_tokens),
            "completed": completed,
            "empty": empty,
            "failed": failed,
            "error_reasons": reason_counts,
            "total": num_prompts,
            "duration": overall_elapsed,
            # Actual MM prefill = clamped text + image placeholder tokens (what
            # drives TTFT), plus the nominal campaign input for reference. Small
            # campaigns (in=128) collapse text to the floor, so the real prefill is
            # dominated by the 4 images - report both so the row isn't mislabeled.
            "input_length": int(target_text + img_tokens),
            "input_length_nominal": int(in_len),
            # The per-image token count the workload was sized and labelled with, and where it
            # came from ("measured" by the preflight on this server, else the model's declared
            # value).
            "per_image_tokens": int(per_img),
            "per_image_tokens_source": per_img_source,
            # Server-measured ACTUAL prefill (usage.prompt_tokens) - the real
            # image cost as the server counts it. None when the server omits usage.
            "input_length_actual": (int(np.median(all_prompt_lens))
                                    if all_prompt_lens else None),
            "mm_images": int(n_img),
            "output_length": int(out_len),
            # TTFT metrics
            "mean_ttft_ms": float(np.mean(ttft_arr)),
            "median_ttft_ms": float(np.median(ttft_arr)),
            "p99_ttft_ms": float(np.percentile(ttft_arr, 99)),
            # E2EL metrics
            "mean_e2el_ms": float(np.mean(e2el_arr)),
            "median_e2el_ms": float(np.median(e2el_arr)),
            "p99_e2el_ms": float(np.percentile(e2el_arr, 99)),
        }

        # TPOT metrics (need >1 token per request)
        if all_tpots:
            tpot_arr = np.array(all_tpots)
            result["mean_tpot_ms"] = float(np.mean(tpot_arr))
            result["median_tpot_ms"] = float(np.median(tpot_arr))
            result["p99_tpot_ms"] = float(np.percentile(tpot_arr, 99))

        # ITL metrics
        if all_itls:
            itl_arr = np.array(all_itls)
            result["mean_itl_ms"] = float(np.mean(itl_arr))
            result["median_itl_ms"] = float(np.median(itl_arr))
            result["p99_itl_ms"] = float(np.percentile(itl_arr, 99))

        # Raw per-request samples (ms) so the pooled aggregator can recompute
        # percentiles + bootstrap CIs across iterations exactly as for the text
        # path (never averaging per-iteration percentiles).
        result["raw_ttfts_ms"] = all_ttfts
        result["raw_itl_blocks_ms"] = all_itl_blocks
        result["raw_e2els_ms"] = all_e2els
        result["output_lens"] = all_out_lens
        result["output_lens_usage"] = all_usage_lens

        return result

    async def _run_benchmark_async(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
        input_requests: list[Any],
        tokenizer,
        request_rate: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Run serving benchmark using vLLM's benchmark() API directly.

        Args:
            model: Model configuration
            format: Model format
            batch_size: Max concurrency (closed-loop). Ignored when
                ``request_rate`` is given.
            num_prompts: Number of prompts
            input_requests: Pre-generated sample requests
            tokenizer: Tokenizer
            request_rate: If set, drive the benchmark OPEN-LOOP at this
                arrival rate (QPS) with Poisson arrivals and NO concurrency
                cap (max_concurrency=None) - the SOTA capacity-under-SLO
                setup. If None, keep the closed-loop behaviour: request rate
                from config, max_concurrency=batch_size.

        Returns:
            Dictionary with benchmark results
        """
        from gbench.utils import check_vllm_available, require_vllm_engine

        model_path = self._resolve_model_id(model, format)
        base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"

        if self.config.remote_endpoint or not check_vllm_available():
            if not self.config.remote_endpoint:
                require_vllm_engine("Serving benchmark execution")
            return await self._run_lightweight_remote_benchmark(
                base_url=base_url,
                model_id=model_path,
                input_requests=input_requests,
                batch_size=batch_size,
                extra_body=extra_body,
            )

        from vllm.benchmarks.serve import benchmark, TaskType
        api_url = f"{base_url}/v1/completions"

        # Open-loop (rate-driven, no concurrency cap) vs closed-loop (fixed
        # concurrency at the config/infinite rate).
        if request_rate is not None:
            rate = float(request_rate)
            max_conc = None
        else:
            rate = float("inf") if self.config.request_rate == "inf" else float(self.config.request_rate)
            max_conc = batch_size

        result = await benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="openai",  # Maps to async_request_openai_completions
            api_url=api_url,
            base_url=base_url,
            model_id=model_path,
            model_name=model_path,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=None,
            request_rate=rate,
            burstiness=1.0,
            max_concurrency=max_conc,
            disable_tqdm=False,
            num_warmups=0,  # We handle warmups ourselves
            profile=False,
            selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
            # Per-iteration record only; the authoritative percentiles are
            # recomputed from POOLED raw samples in _aggregate_serving_results
            # (averaging per-iteration percentiles is statistically invalid).
            selected_percentiles=[50.0, 95.0, 99.0, 99.9],
            # RandomDataset prompts are gibberish token IDs; with EOS honored an
            # instruct model stops after a handful of tokens, so decode-heavy
            # (out=2048) / long-decode (out=8192) would measure TPOT over almost
            # no decode and output_throughput would collapse. vLLM's own random
            # benchmark CLI forces ignore_eos for exactly this reason; match it so
            # every request emits its requested output length and the decode /
            # goodput numbers are real and comparable across the 4 models.
            ignore_eos=True,
            goodput_config_dict={},
            lora_modules=None,
            # vLLM's request functions send only OPENAI_API_KEY; extra_headers replace it with
            # the endpoint key gbench resolves (VLLM_API_KEY, the key the server enforces, else
            # OPENAI_API_KEY), the header every other request of the run carries.
            extra_headers=self._request_headers() or None,
            extra_body=extra_body or None,
        )

        return result

    async def _run_lightweight_remote_benchmark(
        self,
        base_url: str,
        model_id: str,
        input_requests: list,
        batch_size: int,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Pure-Python async streaming benchmark runner for remote HTTP endpoints."""
        import time
        import urllib.request
        import json
        import asyncio
        from dataclasses import dataclass

        @dataclass
        class MetricResult:
            metrics: dict

        url = base_url.rstrip("/")
        api_url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"
        if not api_url.startswith("http://") and not api_url.startswith("https://"):
            api_url = f"http://{api_url}"
        req_headers = self._request_headers()

        semaphore = asyncio.Semaphore(batch_size)
        in_flight = peak_in_flight = 0     # evidence that batch_size was really reached
        max_loop_lag = 0.0                 # client event-loop lag (s)

        class _HTTPStatus(Exception):
            def __init__(self, status):
                super().__init__(f"HTTP {status}")
                self.status = status

        all_ttfts = []       # ms, one per COMPLETED request
        all_itl_blocks = []  # ms, per-request inter-token latency lists
        all_e2els = []       # ms, one per COMPLETED request
        all_out_lens = []    # output tokens, one per COMPLETED request
        all_usage_lens = []  # server-reported output tokens (None when absent)
        all_prompt_lens = [] # server-measured input (prompt) tokens, one per COMPLETED request
        completed = 0
        empty = 0            # HTTP 200 but the model produced 0 content tokens
        failed = 0           # transport / HTTP errors (exceptions / non-200)
        error_reasons = []   # reason string per failed request
        total_output_tokens = 0
        start_time = time.perf_counter()

        async def _send_single_request(session, req):
            """Stream one request; return per-request samples or None on empty.

            Mirrors the MM path (_run_mm_serving_benchmark): TPOT/ITL come from
            inter-arrival gaps between REAL content chunks (tail-EXCLUDED - the
            numerator never includes the post-last-token usage chunk / [DONE] /
            connection close), and a request that produced <=1 content chunk
            yields NO TPOT/ITL sample (it has no inter-token interval) instead of
            a full-e2e-latency value masquerading as a per-token time. Output
            length is the server's token-accurate usage when present, else the
            content-chunk count.
            """
            nonlocal in_flight, peak_in_flight
            # No-think by default: our system prompt overrides the backend's
            # default (reasoning-enabling) one. --thinking skips it (model reasons).
            _pl = {
                "model": model_id,
                "messages": chat_messages(req.prompt, None if getattr(
                    self.config, "thinking", False) else self.NOTHINK_SYSTEM),
                "max_tokens": getattr(req, "expected_output_len", 128),
                "stream": True,
                # Force full-length decode where honored (a local vLLM engine);
                # a remote OpenAI-compat endpoint may drop it (Ollama does).
                "ignore_eos": True,
                "stream_options": {"include_usage": True},
            }
            _pl.update(self.no_think_fields())       # portable no-think for the perf workload
            _pl.update(extra_body or {})

            req_start = time.perf_counter()
            content_times = []   # arrival time of each REAL content token
            usage_tokens = None
            usage_prompt = None
            # Empty-reply diagnostics: why does a 200 yield 0 content chunks?
            n_data_chunks = 0
            finish_reason = None
            choice_keys = set()
            delta_keys = set()

            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            try:
                async with session.post(api_url, json=_pl) as resp:
                    if resp.status != 200:
                        await resp.text()
                        raise _HTTPStatus(resp.status)
                    async for raw in resp.content:
                        line_str = raw.decode("utf-8", errors="ignore").strip()
                        if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                            continue
                        n_data_chunks += 1
                        try:
                            data = json.loads(line_str[6:])
                            choices = data.get("choices") or []
                            if choices:
                                choice_keys.update(choices[0].keys())
                                _d = choices[0].get("delta") or {}
                                delta_keys.update(_d.keys())
                                if choices[0].get("finish_reason"):
                                    finish_reason = choices[0].get("finish_reason")
                                # Count a generated token from the content OR the
                                # reasoning/thinking channel (a reasoning model streams
                                # real decode tokens there).
                                delta = (_d.get("content") or _d.get("reasoning")
                                         or _d.get("reasoning_content") or "")
                            else:
                                delta = ""
                            # Final chunk (include_usage) has empty choices + usage.
                            usage = data.get("usage")
                            if usage and usage.get("completion_tokens") is not None:
                                usage_tokens = int(usage["completion_tokens"])
                            if usage and usage.get("prompt_tokens") is not None:
                                usage_prompt = int(usage["prompt_tokens"])
                        except Exception:
                            delta = ""
                        if delta:
                            content_times.append(time.perf_counter())
            finally:
                in_flight -= 1

            req_end = time.perf_counter()
            if not content_times:
                # HTTP 200 but 0 content chunks: an EMPTY completion. Diagnose WHY:
                # usage.completion_tokens>0 => server DID generate tokens we missed
                # (parse bug; delta_keys/choice_keys show where the text went);
                # ==0/None with finish_reason=stop => genuinely empty from the model.
                logger.warning(
                    "EMPTY text reply diag: usage.completion_tokens=%s "
                    "finish_reason=%s data_chunks=%s choice_keys=%s delta_keys=%s",
                    usage_tokens, finish_reason, n_data_chunks,
                    sorted(choice_keys), sorted(delta_keys))
                return None

            ttft_ms = (content_times[0] - req_start) * 1000
            e2el_ms = (req_end - req_start) * 1000
            num_tokens = usage_tokens if usage_tokens is not None else len(content_times)
            # Per-request inter-token latencies (ms); empty for a 1-token reply.
            req_itls_ms = [
                (content_times[j] - content_times[j - 1]) * 1000
                for j in range(1, len(content_times))
            ]
            return {
                "ttft_ms": ttft_ms,
                "itls_ms": req_itls_ms,
                "e2el_ms": e2el_ms,
                "num_tokens": num_tokens,
                "usage_tokens": usage_tokens,
                "prompt_tokens": usage_prompt,
            }

        def _classify_error(e):
            import socket
            if isinstance(e, _HTTPStatus):
                return f"http_{e.status}"
            reason = getattr(e, "reason", None)
            if (isinstance(e, (socket.timeout, TimeoutError, asyncio.TimeoutError))
                    or isinstance(reason, socket.timeout)
                    or "timed out" in str(e).lower()):
                return "timeout"
            return type(e).__name__

        async def _worker(session, req):
            nonlocal completed, empty, failed, total_output_tokens
            async with semaphore:
                try:
                    r = await _send_single_request(session, req)
                except Exception as e:
                    # Transport / HTTP error -> a real failure, with a reason.
                    failed += 1
                    error_reasons.append(_classify_error(e))
                    logger.warning(f"Remote benchmark request error: {e}")
                    return
                if r is None:
                    # HTTP 200 with 0 content tokens: an EMPTY completion, tracked
                    # separately from failures (it is not an error).
                    empty += 1
                    return
                all_ttfts.append(r["ttft_ms"])
                all_itl_blocks.append(r["itls_ms"])
                all_e2els.append(r["e2el_ms"])
                all_out_lens.append(r["num_tokens"])
                all_usage_lens.append(r.get("usage_tokens"))
                if r.get("prompt_tokens") is not None:
                    all_prompt_lens.append(r["prompt_tokens"])
                completed += 1
                total_output_tokens += r["num_tokens"]

        import aiohttp

        async def _loop_lag():
            nonlocal max_loop_lag
            while True:
                t = time.perf_counter()
                await asyncio.sleep(0.1)
                max_loop_lag = max(max_loop_lag, time.perf_counter() - t - 0.1)

        # The read timeout is a STALL cap (no bytes for this long, e.g. a hung prefill),
        # not a cap on total decode, as urllib's per-read timeout was. Tunable via
        # --timeout (config.timeout).
        req_timeout = getattr(self.config, "timeout", 600) or 600
        connector = aiohttp.TCPConnector(limit=batch_size + 20)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=req_timeout,
                                        sock_connect=req_timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout,
                                         headers=req_headers) as session:
            lag_task = asyncio.create_task(_loop_lag())
            try:
                await asyncio.gather(*[_worker(session, req) for req in input_requests])
            finally:
                lag_task.cancel()

        dur = time.perf_counter() - start_time

        def p50(lst):
            return float(np.percentile(lst, 50)) if lst else 0.0
        def p99(lst):
            return float(np.percentile(lst, 99)) if lst else 0.0

        # Per-request TPOT per token from the server's count; the chunk mean only when
        # the server sent no usage.
        tpots = []
        for t, e, b, u in zip(all_ttfts, all_e2els, all_itl_blocks, all_usage_lens):
            v = per_token_tpot_ms(e, t, u)
            if v is None and b:
                v = sum(b) / len(b)
            if v is not None:
                tpots.append(v)
        itls_flat = [v for b in all_itl_blocks for v in b]

        reason_counts = {}
        for _rsn in error_reasons:
            reason_counts[_rsn] = reason_counts.get(_rsn, 0) + 1

        metrics = {
            "request_throughput": completed / dur if dur > 0 else 0.0,
            "output_throughput": total_output_tokens / dur if dur > 0 else 0.0,
            "total_output_tokens": int(total_output_tokens),
            "completed": completed,
            "empty": empty,
            "failed": failed,
            "error_reasons": reason_counts,
            "total": len(input_requests),
            "duration": dur,
            # Raw per-request samples (ms) so the POOLED aggregator recomputes
            # percentiles + bootstrap CIs exactly as for the MM/local paths, and
            # so the text remote path shares ONE estimator + ONE aggregation with
            # them (no legacy scalar fallback -> no mean-as-p50 divergence, no
            # ITL=TPOT copies, and failed requests are counted, not silently lost).
            "raw_ttfts_ms": all_ttfts,
            "raw_itl_blocks_ms": all_itl_blocks,
            "raw_e2els_ms": all_e2els,
            "output_lens": all_out_lens,
            "output_lens_usage": all_usage_lens,
            "peak_in_flight": peak_in_flight,
            "client_loop_lag_max_ms": max_loop_lag * 1000.0,
            # Server-measured ACTUAL input tokens (usage.prompt_tokens), median
            # over completed requests; None when the server omits usage. Mirrors
            # the MM path so both rows can report actual (not just nominal) input.
            "input_length_actual": (int(np.median(all_prompt_lens))
                                    if all_prompt_lens else None),
        }
        if all_ttfts:
            metrics["mean_ttft_ms"] = float(np.mean(all_ttfts))
            metrics["median_ttft_ms"] = p50(all_ttfts)
            metrics["p99_ttft_ms"] = p99(all_ttfts)
            metrics["mean_e2el_ms"] = float(np.mean(all_e2els))
            metrics["median_e2el_ms"] = p50(all_e2els)
            metrics["p99_e2el_ms"] = p99(all_e2els)
        if tpots:
            metrics["mean_tpot_ms"] = float(np.mean(tpots))
            metrics["median_tpot_ms"] = p50(tpots)
            metrics["p99_tpot_ms"] = p99(tpots)
        if itls_flat:
            metrics["mean_itl_ms"] = float(np.mean(itls_flat))
            metrics["median_itl_ms"] = p50(itls_flat)
            metrics["p99_itl_ms"] = p99(itls_flat)

        return metrics

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: Optional[str] = None,
        multimodal: bool = False,
        manage_server: bool = True,
    ) -> dict:
        """Run serving benchmark for a specific configuration (text only).

        For multimodal, use run_with_iterations() which routes to
        _run_mm_with_iterations() for real image requests.

        Args:
            model: Model configuration
            format: Model format (HF, or REMOTE for an endpoint)
            batch_size: Number of concurrent requests
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset to use (optional, ignored - we use RandomDataset)
            multimodal: If True, delegates to _run_mm_with_iterations()

        Returns:
            Dictionary with benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        num_prompts = num_prompts or self.config.num_prompts

        # Route multimodal to the proper MM implementation
        if multimodal:
            return self._run_mm_with_iterations(
                model, format, batch_size, num_prompts, manage_server=manage_server
            )

        lm = self.config.log_manager

        # Generate output filename
        output_file = lm.get_serving_result_path(
            model.short_name, format.value, batch_size, multimodal,
            campaign=getattr(self.config, "campaign", None),
        )

        # Start vLLM server (unless the caller owns its lifecycle, e.g. run_all
        # reusing one server across batch sizes)
        if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format):
                raise RuntimeError("Failed to start vLLM server")

        try:
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would run benchmark for {model.short_name}")
                return {"dry_run": True}

            # Log clear benchmark configuration banner
            logger.info(
                f"\n{'='*60}\n"
                f"  BENCHMARK CONFIGURATION\n"
                f"{'='*60}\n"
                f"  Type:           Serving\n"
                f"  Mode:           Text\n"
                f"  Model:          {model.short_name} ({format.value})\n"
                f"  Batch size:     {batch_size}\n"
                f"  Num prompts:    {num_prompts}\n"
                f"{'='*60}"
            )

            # Generate sample requests
            logger.info(f"Generating {num_prompts} sample requests...")
            self._new_payload_call()
            input_requests, tokenizer = self._generate_sample_requests(
                model, format, num_prompts
            )

            # Run benchmark
            result = asyncio.run(self._run_benchmark_async(
                model, format, batch_size, num_prompts, input_requests, tokenizer
            ))

            # Save result to file
            result["model"] = model.short_name
            result["model_short"] = model.short_name
            result["model_name"] = model.name
            result["format"] = format.value
            result["batch_size"] = batch_size
            result["multimodal"] = multimodal
            result["output_token_throughput"] = result.get("output_throughput", result.get("output_token_throughput", 0.0))
            self._stamp_server(result, model, format)
            self._stamp_campaign(result)
            write_redacted_json(output_file, result)

            # Log key metrics
            req_tput = result.get("request_throughput", "N/A")
            ttft = result.get("mean_ttft_ms", "N/A")
            tpot = result.get("mean_tpot_ms", "N/A")
            logger.info(
                f"Results: request_throughput={req_tput:.2f} req/s, "
                f"TTFT={ttft:.2f}ms, TPOT={tpot:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise RuntimeError(f"Serving benchmark failed: {e}")
        
        finally:
            # Cleanup server unless the caller owns its lifecycle
            if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
                self._cleanup_server()

    def _get_output_path(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool = False,
    ) -> Path:
        """Generate output path for benchmark results.
        
        Args:
            model: Model configuration
            format: Model format
            batch_size: Batch size used
            multimodal: Whether this is a multimodal benchmark
            
        Returns:
            Path to the output JSON file
        """
        lm = self.config.log_manager
        return lm.get_serving_result_path(
            model.short_name, format.value, batch_size, multimodal,
            campaign=getattr(self.config, "campaign", None),
        )

    def run_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: Optional[str] = None,
        multimodal: bool = False,
        manage_server: bool = True,
    ) -> dict:
        """Run benchmark with multiple iterations and statistical analysis.

        For text: uses vLLM's benchmark() API with RandomDataset.
        For multimodal: uses aiohttp + SSE with real images via /v1/chat/completions.

        Args:
            model: Model configuration
            format: Model format (HF, or REMOTE for an endpoint)
            batch_size: Number of concurrent requests
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset to use (optional)
            multimodal: If True, send real image requests

        Returns:
            Dictionary with aggregated results and statistics
        """
        # No single-iteration shortcut: even num_iterations==1 flows through the
        # pooled aggregator so every text-serving result carries pooled
        # percentiles + bootstrap CIs + tail_validity (uniform, self-describing
        # output). The pooled path handles one rep fine (no run-to-run CV).

        # Log clear benchmark configuration banner
        mode_str = "Multimodal" if multimodal else "Text"
        logger.info(
            f"\n{'='*60}\n"
            f"  BENCHMARK CONFIGURATION\n"
            f"{'='*60}\n"
            f"  Type:           Serving\n"
            f"  Mode:           {mode_str}\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Batch size:     {batch_size}\n"
            f"  Num prompts:    {num_prompts or self.config.num_prompts}\n"
            f"{'='*60}"
        )

        logger.info(
            f"Running {self.config.num_iterations} iterations "
            f"(+{self.config.warmup_iterations} warmup)"
        )

        num_prompts = num_prompts or self.config.num_prompts
        self._new_payload_call()

        if multimodal:
            return self._run_mm_with_iterations(
                model, format, batch_size, num_prompts, manage_server=manage_server
            )
        else:
            return self._run_text_with_iterations(
                model, format, batch_size, num_prompts, manage_server=manage_server
            )

    def _new_payload_call(self) -> None:
        """Start a new payload call (one batch size / campaign / model): no request of
        it repeats one of an earlier call, and its counters start at zero."""
        self.payloads.new_call()
        self._real_pool, self._real_cursor, self._prompt_reuse = [], 0, 0

    def _real_dataset(self) -> bool:
        return self.config.dataset in ("sharegpt", "custom", "hf")

    def _requests_for(self, model, format, num_prompts: int, phase: str, rep: int,
                      total_hint: int) -> tuple:
        """(requests, tokenizer) for one warmup or measured iteration, never repeating a
        request of an earlier one: synthetic prompts carry a per-(phase, rep) nonce; real
        datasets give each call the next disjoint slice of one pool sampled up front
        (total_hint requests), and count any shortfall in self._prompt_reuse."""
        if not self._real_dataset():
            return self._generate_sample_requests(model, format, num_prompts,
                                                  phase=phase, rep=rep)
        if not self._real_pool:
            pool, tok = self._generate_sample_requests(model, format, max(num_prompts, total_hint))
            # The sampler may return repeats (a dataset smaller than asked for): keep one
            # of each, so every shortfall is counted as reuse below.
            seen, uniq = set(), []
            for q in pool:
                key = repr(getattr(q, "prompt", q))
                if key not in seen:
                    seen.add(key)
                    uniq.append(q)
            self._real_pool, self._real_tok, self._real_cursor = uniq, tok, 0
        pool = self._real_pool
        take = pool[self._real_cursor:self._real_cursor + num_prompts]
        self._real_cursor += len(take)
        if len(take) < num_prompts:
            short = num_prompts - len(take)
            self._prompt_reuse += short
            take = take + [pool[k % len(pool)] for k in range(short)]
        return take, self._real_tok

    def _run_text_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
        manage_server: bool = True,
    ) -> dict:
        """Run text serving benchmark with multiple iterations.

        Every warmup and measured iteration sends requests none of the others sent
        (_requests_for), so iteration 2 is never served from iteration 1's prefix cache.
        """
        # Start server (unless the caller reuses one across batch sizes)
        if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format):
                raise RuntimeError("Failed to start vLLM server")

        try:
            warmup_total = self.config.warmup_iterations
            iter_total = self.config.num_iterations
            n_warm = min(2, num_prompts) if self.config.remote_endpoint else num_prompts
            total_hint = n_warm * warmup_total + num_prompts * iter_total
            real = self._real_dataset()
            logger.info(f"Generating sample requests ({num_prompts} per iteration)...")

            # Warmup iterations
            for i in range(warmup_total):
                progress = "█" * (i + 1) + "░" * (warmup_total - i - 1)
                logger.info(f"Warmup [{progress}] {i+1}/{warmup_total} starting...")
                iter_start = time.time()
                try:
                    warmup_reqs, tokenizer = self._requests_for(
                        model, format, n_warm, "serving_warmup", i, total_hint)
                    asyncio.run(self._run_benchmark_async(
                        model, format, min(batch_size, len(warmup_reqs)), len(warmup_reqs),
                        warmup_reqs, tokenizer,
                        extra_body=self._extra_body(real, "serving_warmup", i),
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  Warmup {i+1} completed in {elapsed:.1f}s")
                except ServingConfigError:
                    raise
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.warning(f"Warmup iteration {i+1} failed after {elapsed:.1f}s: {e}")

            # Actual benchmark iterations
            results = []
            for i in range(iter_total):
                progress = "█" * (i + 1) + "░" * (iter_total - i - 1)
                logger.info(f"Benchmark [{progress}] {i+1}/{iter_total} starting...")
                iter_start = time.time()
                try:
                    input_requests, tokenizer = self._requests_for(
                        model, format, num_prompts, "serving", i, total_hint)
                    before = self._metrics_snapshot()
                    result = asyncio.run(self._run_benchmark_async(
                        model, format, batch_size, num_prompts, input_requests, tokenizer,
                        extra_body=self._extra_body(real, "serving", i),
                    ))
                    if isinstance(result, dict):
                        result["cache"] = self._cache_delta(before, self._metrics_snapshot())
                    elapsed = time.time() - iter_start
                    logger.info(f"  Benchmark {i+1} completed in {elapsed:.1f}s")
                    results.append(result)
                except ServingConfigError:
                    raise
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.error(f"Iteration {i+1} failed after {elapsed:.1f}s: {e}")
                    continue

            return self._aggregate_serving_results(
                results, model, format, batch_size, multimodal=False
            )

        finally:
            if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
                self._cleanup_server()

    def _assert_mm_images_processed(self, model_path: str, base_url: str,
                                    per_img_tokens: Optional[int] = None) -> Optional[int]:
        """Preflight guard: prove the server actually INGESTS images before we
        trust any multimodal number. Sends one chat request with
        IMAGES_PER_REQUEST images + trivial text and checks that the server's
        reported ``usage.prompt_tokens`` grows by a meaningful number of tokens per
        image versus the same text alone (backend-agnostic: no model's per-image
        count is assumed; ``per_img_tokens`` is accepted for callers and unused).
        Raises if not, so a mis-configured server (no --limit-mm-per-prompt /
        images silently dropped) can NEVER produce text-only results mislabeled
        as multimodal.

        Returns the SERVER-MEASURED per-image token count (int) = the prompt-token
        gain per image the server actually charges for these images, whatever its
        processor settings (e.g. --mm-processor-kwargs). Returns None when the probe
        is skipped (dry_run / no images). Callers size and report the MM workload
        with it.
        """
        if self.config.dry_run or not self._mm_image_paths:
            return None
        import urllib.request
        n_img = self._images_per_request()

        # base_url may already include /v1 (remote endpoints like Ollama's
        # http://host:11434/v1); only append /v1 when it isn't there, else the
        # URL becomes /v1/v1/... -> 404.
        _b = base_url.rstrip("/")
        chat_url = f"{_b}/chat/completions" if _b.endswith("/v1") else f"{_b}/v1/chat/completions"
        _timeout = int(getattr(self.config, "mm_probe_timeout", None) or 60)

        def _prompt_tokens(content):
            payload = {"model": model_path, "max_tokens": 1, "stream": False,
                       "messages": [{"role": "user", "content": content}]}
            req = urllib.request.Request(
                chat_url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **self._request_headers()})
            with urllib.request.urlopen(req, timeout=_timeout) as resp:
                d = json.loads(resp.read())
            return int((d.get("usage") or {}).get("prompt_tokens", 0))

        TEXT = "Describe."
        # Probe images are new to the server (perf_payload) and never sent again, so
        # the probe neither hits nor warms the processor cache for the measurement.
        urls = []
        for k in range(n_img):
            data = self.payloads.unique_image_jpeg("probe", 0, 0, 0, k)
            if self.config.remote_endpoint:
                import base64
                urls.append("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
            else:
                probe_dir = os.path.dirname(self._mm_image_paths[0])
                path = os.path.join(probe_dir, f"probe_{self.payloads.salt}_{k}.jpg")
                with open(path, "wb") as fh:
                    fh.write(data)
                urls.append(f"file://{path}")
        try:
            text_ptok = _prompt_tokens([{"type": "text", "text": TEXT}])
            img_ptok = _prompt_tokens(
                [{"type": "text", "text": TEXT}]
                + [{"type": "image_url", "image_url": {"url": u}} for u in urls])
        except Exception as e:
            raise RuntimeError(f"MM preflight request failed: {e}") from e

        # Self-calibrating: the image(s) must add a meaningful number of prompt
        # tokens vs the SAME text alone. Backend-AGNOSTIC - no per-image count is
        # assumed (backends and processor settings differ widely), so an absolute
        # floor would false-flag a working endpoint. A dropped image adds ~0. NOTE: with n_img>1
        # this checks the AVERAGE per-image gain, so it won't catch a partial
        # ingestion of a subset - but --personal uses n_img=1 (all-or-nothing) and
        # the local vLLM path processes all images.
        MIN_PER_IMG = 16
        per_img = (img_ptok - text_ptok) / max(1, n_img)
        if per_img < MIN_PER_IMG:
            raise RuntimeError(
                f"MM PREFLIGHT FAILED: images added only ~{per_img:.0f} tok/img "
                f"(text-only={text_ptok}, +{n_img} image(s)={img_ptok}). Images are "
                f"NOT being processed end-to-end - refusing to report text-only "
                f"numbers as multimodal. Check the endpoint accepts images.")
        logger.info(f"MM preflight OK: images add ~{per_img:.0f} tok/img "
                    f"(text {text_ptok} -> {img_ptok} with {n_img} image(s)); "
                    f"vision path confirmed live.")
        return int(round(per_img))

    def _run_mm_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
        manage_server: bool = True,
    ) -> dict:
        """Run multimodal serving benchmark with multiple iterations.

        Sets up images, starts server with --allowed-local-media-path,
        runs warmup + benchmark iterations using aiohttp + SSE,
        then aggregates and cleans up. When ``manage_server`` is False the
        caller (run_all) has already set up the images and started one shared
        server, so this method only runs the iterations.
        """
        # Set up images + start the server only when we own the lifecycle.
        if manage_server:
            mm_media_path = self._setup_mm_images(num_images=50)
            if not self.config.dry_run and not self.config.remote_endpoint:
                if not self._start_server(
                    model, format, allowed_local_media_path=mm_media_path,
                    mm_limit_images=self._images_per_request(),
                ):
                    self._cleanup_mm_images()
                    raise RuntimeError("Failed to start vLLM server for MM serving")

        try:
            # Preflight: prove images are actually ingested before any measurement, and size
            # and label the workload with the per-image count this server measured.
            base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"
            self._vision_tokens_per_image = self._vision_tokens_source = None  # this model's own
            measured = self._assert_mm_images_processed(
                self._resolve_model_id(model, format), base_url)
            if measured:
                self._vision_tokens_per_image, self._vision_tokens_source = int(measured), "measured"

            # Warmup iterations
            warmup_total = self.config.warmup_iterations
            for i in range(warmup_total):
                progress = "█" * (i + 1) + "░" * (warmup_total - i - 1)
                logger.info(f"MM Warmup [{progress}] {i+1}/{warmup_total} starting...")
                iter_start = time.time()
                try:
                    # A remote endpoint has no local server to warm - a couple of
                    # requests suffice to prime connections/caches. Mirror the text
                    # path (which slices to 2) so remote MM warmup doesn't run the
                    # full campaign serially (~20 min under --personal on a laptop).
                    warmup_prompts = (min(2, num_prompts)
                                      if self.config.remote_endpoint else num_prompts)
                    asyncio.run(self._run_mm_serving_benchmark(
                        model, format,
                        max_concurrency=min(batch_size, warmup_prompts),
                        num_prompts=warmup_prompts, phase="serving_warmup", rep=i,
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  MM Warmup {i+1} completed in {elapsed:.1f}s")
                except ServingConfigError:
                    raise
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.warning(f"MM Warmup {i+1} failed after {elapsed:.1f}s: {e}")

            # Actual benchmark iterations
            results = []
            iter_total = self.config.num_iterations
            for i in range(iter_total):
                progress = "█" * (i + 1) + "░" * (iter_total - i - 1)
                logger.info(f"MM Benchmark [{progress}] {i+1}/{iter_total} starting...")
                iter_start = time.time()
                try:
                    before = self._metrics_snapshot()
                    result = asyncio.run(self._run_mm_serving_benchmark(
                        model, format, max_concurrency=batch_size, num_prompts=num_prompts,
                        phase="serving", rep=i,
                    ))
                    result["cache"] = self._cache_delta(before, self._metrics_snapshot())
                    elapsed = time.time() - iter_start
                    logger.info(f"  MM Benchmark {i+1} completed in {elapsed:.1f}s")
                    results.append(result)
                except ServingConfigError:
                    raise
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.error(f"MM Iteration {i+1} failed after {elapsed:.1f}s: {e}")
                    continue

            return self._aggregate_serving_results(
                results, model, format, batch_size, multimodal=True
            )

        finally:
            if manage_server:
                if not self.config.dry_run and not self.config.remote_endpoint:
                    self._cleanup_server()
                self._cleanup_mm_images()

    def _extract_raw_samples(self, result: dict) -> dict:
        """Normalize one iteration's result to raw per-request samples (ms).

        Handles the text path (vLLM benchmark() result: ``ttfts`` in seconds,
        ``itls`` as per-request lists in seconds) and the MM path (``raw_*``
        arrays already in ms). Per-request TPOT is per token, from the server's token
        count; ITL stays the gap between streamed chunks (what a user sees); e2el is
        the measured value (MM) or reconstructed as ttft + sum(itls) (text), which is
        vLLM's own e2el definition.

        Returns:
            {ttft_ms, itl_blocks_ms, e2el_ms, tpot_ms} - ttft/e2el/tpot are one
            value per request; itl_blocks_ms is one list per request.
        """
        out_lens: list = []
        if "raw_ttfts_ms" in result:  # MM path (already ms, only successes recorded)
            ttft_ms = [float(t) for t in result.get("raw_ttfts_ms", [])]
            itl_blocks = [[float(v) for v in b]
                          for b in result.get("raw_itl_blocks_ms", [])]
            e2el_ms = [float(v) for v in result.get("raw_e2els_ms", [])]
            # Only the server's own counts make TPOT per token (output_lens falls back
            # to the chunk count when the server sent no usage).
            out_lens = list(result.get("output_lens_usage")
                            if "output_lens_usage" in result else result.get("output_lens") or [])
        else:  # text path (vLLM benchmark()), seconds -> ms
            # This vLLM's benchmark() records ttfts/itls/e2els for SUCCESSFUL
            # requests ONLY (serve.py `if outputs[i].success:`), so failed/aborted
            # requests are already absent here - they never reach the pooled
            # percentiles and do not inflate n. The ttft>0 guard is kept as a
            # defensive backstop (a 0-ttft success from some endpoint). Failure
            # COUNTS are surfaced in _aggregate_serving_results via the result's
            # `completed`/`failed` fields, not inferred from ttft.
            raw_ttfts = result.get("ttfts", [])
            raw_itls = result.get("itls", [])
            raw_lens = result.get("output_lens", []) or []
            ttft_ms, itl_blocks = [], []
            for i, t in enumerate(raw_ttfts):
                if t is None or float(t) <= 0.0:
                    continue
                ttft_ms.append(float(t) * 1000.0)
                gaps = raw_itls[i] if i < len(raw_itls) else []
                itl_blocks.append([float(v) * 1000.0 for v in (gaps or [])])
                out_lens.append(raw_lens[i] if i < len(raw_lens) else None)
            e2el_ms = []
        recon_e2e = []
        for i, ttft in enumerate(ttft_ms):
            gaps = itl_blocks[i] if i < len(itl_blocks) else []
            recon_e2e.append(ttft + sum(gaps))
        if not e2el_ms:
            e2el_ms = recon_e2e
        # Per-request TPOT per TOKEN: (e2e - ttft) / (tokens - 1) with the server's token
        # count; the mean gap between streamed chunks overstates it when a chunk carries
        # several tokens, so it is only the fallback when no count exists.
        tpot_ms, fallbacks = [], 0
        for i, ttft in enumerate(ttft_ms):
            n = out_lens[i] if i < len(out_lens) else None
            v = per_token_tpot_ms(e2el_ms[i] if i < len(e2el_ms) else None, ttft, n)
            if v is None:
                gaps = itl_blocks[i] if i < len(itl_blocks) else []
                v = (sum(gaps) / len(gaps)) if gaps else None
                fallbacks += v is not None
            if v is not None:
                tpot_ms.append(v)
        return {"ttft_ms": ttft_ms, "itl_blocks_ms": itl_blocks,
                "e2el_ms": e2el_ms, "tpot_ms": tpot_ms, "tpot_fallbacks": fallbacks}

    def _aggregate_serving_results(
        self,
        results: list[dict],
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool,
    ) -> dict:
        """Aggregate multi-iteration serving results the statistically-correct way.

        POOLS the raw per-request samples across all iterations and recomputes
        each percentile ONCE from the pooled distribution (averaging
        per-iteration percentiles is invalid - you cannot average a P99). Reports
        mean + P50/P95/P99/P99.9 with a bootstrap 95% CI on P99 (block bootstrap
        for autocorrelated ITL) and a tail_validity flag per percentile, so an
        under-sampled tail (inevitable at batch_size=1 / long outputs) is
        surfaced with its uncertainty, never passed off as a hard number.
        Throughput rates are averaged across reps (valid - independent rate
        estimates) with CV. Falls back to legacy scalar aggregation for the
        remote lightweight path, which carries no raw samples.
        """
        if not results:
            raise RuntimeError("All iterations failed")

        # Key PRESENCE, not truthiness: a raw-capable result whose sample arrays
        # are EMPTY (e.g. every request returned an empty completion -> raw_ttfts_ms
        # == []) must still enter the pooled path so the all-empty guard below flags
        # it as failed. Testing truthiness dropped such a result to the legacy scalar
        # path, which never sets failed=True, so an all-empty run read as a success.
        valid = [
            r for r in results
            if isinstance(r, dict) and ("ttfts" in r or "raw_ttfts_ms" in r)
        ]
        if not valid:
            return self._aggregate_serving_results_legacy(
                results, model, format, batch_size, multimodal
            )

        # Pool raw per-request samples across all iterations.
        pooled_ttft, pooled_e2e, pooled_tpot, itl_blocks = [], [], [], []
        pooled_out_lens = []
        per_iter_p99_ttft = []
        tot_completed = tot_failed = tot_empty = 0
        merged_reasons = {}
        for r in valid:
            raw = self._extract_raw_samples(r)
            pooled_ttft.extend(raw["ttft_ms"])
            pooled_e2e.extend(raw["e2el_ms"])
            pooled_tpot.extend(raw["tpot_ms"])
            itl_blocks.extend(raw["itl_blocks_ms"])
            pooled_out_lens.extend(r.get("output_lens") or [])
            if raw["ttft_ms"]:
                per_iter_p99_ttft.append(float(np.percentile(raw["ttft_ms"], 99)))
            tot_completed += int(r.get("completed", len(raw["ttft_ms"])) or 0)
            tot_failed += int(r.get("failed", 0) or 0)
            tot_empty += int(r.get("empty", 0) or 0)
            for _rsn, _cnt in (r.get("error_reasons") or {}).items():
                merged_reasons[_rsn] = merged_reasons.get(_rsn, 0) + int(_cnt)

        # Surface outcomes. Percentiles above are success-only (clean). EMPTY
        # completions (HTTP 200, 0 tokens - e.g. a backend that ignores ignore_eos
        # replying to a nonsense prompt) are NOT errors: they are counted and
        # reported separately and EXCLUDED from the failure rate. Only real errors
        # (exceptions / non-200) count as failures.
        offered = tot_completed + tot_empty + tot_failed
        fail_rate = (tot_failed / offered) if offered > 0 else 0.0
        empty_rate = (tot_empty / offered) if offered > 0 else 0.0
        if tot_failed or tot_empty:
            _bits = []
            if tot_failed:
                _bits.append(f"{tot_failed} error"
                             + (f" {merged_reasons}" if merged_reasons else ""))
            if tot_empty:
                _bits.append(f"{tot_empty} empty (0-token)")
            logger.warning(
                f"Serving {model.short_name} batch={batch_size} mm={multimodal}: "
                f"{tot_completed}/{offered} ok; " + ", ".join(_bits)
                + "; latency percentiles reflect successful requests only."
            )

        # Every request failed (no first token) across all iterations: there are
        # no latency samples, so this config genuinely failed. Surface it as a
        # failure (which the exit-code guard + reporting handle) rather than
        # writing a result with throughput but no latency keys.
        if not pooled_ttft:
            _why = ("all requests returned EMPTY (0-token) completions"
                    if tot_empty and not tot_failed
                    else "no successful requests")
            logger.error(
                f"Serving config produced no usable latency samples for "
                f"{model.short_name} batch={batch_size} mm={multimodal}: {_why} "
                f"(ok={tot_completed}, empty={tot_empty}, error={tot_failed})."
            )
            return {
                "failed": True,
                "error": f"no usable serving samples: {_why} "
                         f"(ok={tot_completed}, empty={tot_empty}, error={tot_failed})",
                "completed_requests": tot_completed,
                "empty_requests": tot_empty,
                "failed_requests": tot_failed,
                "error_reasons": merged_reasons,
                "model": model.short_name, "model_short": model.short_name,
                "model_name": model.name, "format": format.value,
                "batch_size": batch_size, "multimodal": multimodal,
                "failure_kind": "harness",
            }

        agg: dict = {}
        # Authoritative pooled percentiles + bootstrap CIs + tail validity.
        agg.update(summarize_latency_metric("ttft", pooled_ttft, ci_percentile=99.0))
        agg.update(summarize_latency_metric("tpot", pooled_tpot, ci_percentile=99.0))
        agg.update(summarize_latency_metric("e2el", pooled_e2e, ci_percentile=99.0))
        agg.update(summarize_latency_metric("itl", None, blocks=itl_blocks,
                                            ci_percentile=99.0))

        # Back-compat scalar keys (downstream reporting reads these), sourced from
        # the POOLED distribution - NOT averaged across iterations.
        for m in ("ttft", "tpot", "itl", "e2el"):
            if f"{m}_p50_ms" in agg:
                agg[f"mean_{m}_ms"] = agg.get(f"{m}_mean_ms")
                agg[f"median_{m}_ms"] = agg.get(f"{m}_p50_ms")
                agg[f"p99_{m}_ms"] = agg.get(f"{m}_p99_ms")

        # Per-request DISPERSION (CV over the pooled samples), so the report's
        # high-variance guard has a real signal to fire on. Previously no tpot CV
        # key existed on any path, so the "indicative only ... TPOT CV" caption
        # was dead code; this is the genuine per-request spread the caption means.
        if pooled_tpot:
            agg["tpot_cv_percent"] = compute_statistics(pooled_tpot)["cv_percent"]
        if pooled_ttft:
            agg["ttft_cv_percent"] = compute_statistics(pooled_ttft)["cv_percent"]

        # A real dataset replays different prompts in each iteration (never repeats),
        # so requests/s moves with their reply lengths; tokens/s is what repeats.
        real = (not multimodal) and getattr(self.config, "dataset", None) in ("sharegpt", "custom", "hf")
        if real:
            agg["workload_dataset"] = self.config.dataset
        # Throughput: averaging RATES across independent reps IS valid; keep CV.
        for m in ("request_throughput", "output_throughput"):
            vals = [r.get(m) for r in valid if r.get(m) is not None]
            if vals:
                st = compute_statistics(vals)
                agg[m] = st["mean"]
                agg[f"{m}_mean"] = st["mean"]
                agg[f"{m}_std"] = st["std"]
                agg[f"{m}_cv_percent"] = st["cv_percent"]
                is_valid, msg = validate_repeatability(
                    {f"{m}_cv_percent": st["cv_percent"]}, m,
                    self.config.min_acceptable_cv_percent,
                )
                if real and m == "request_throughput" and is_valid is not None:
                    is_valid = None
                    msg = (f"not judged: CV% = {st['cv_percent']:.2f}% across iterations that "
                           f"replay different {self.config.dataset} prompts (tokens/s is judged)")
                agg[f"{m}_repeatability_valid"] = is_valid
                logger.info(f"{m}: mean={st['mean']:.2f} {msg}")
                if is_valid is None:
                    agg[f"{m}_repeatability"] = ("not assessed (1 iteration)" if len(vals) < 2
                                                 else "not judged (different prompts per iteration)")
        agg["repeatability_metric"] = "output_throughput" if real else "request_throughput"
        agg["repeatability_valid"] = agg.get(f"{agg['repeatability_metric']}_repeatability_valid")
        agg["output_token_throughput"] = agg.get("output_throughput", 0.0)

        # Run-to-run stability: CV of each iteration's own P99 TTFT (a summary
        # statistic across independent reps - averaging its CV IS valid).
        if len(per_iter_p99_ttft) >= 2:
            agg["p99_ttft_run_to_run_cv_percent"] = \
                compute_statistics(per_iter_p99_ttft)["cv_percent"]

        # Provenance / metadata.
        agg["num_iterations"] = len(valid)
        agg["pooled_sample_count"] = len(pooled_ttft)
        agg["completed_requests"] = tot_completed
        agg["empty_requests"] = tot_empty
        agg["failed_requests"] = tot_failed
        agg["offered_requests"] = offered
        agg["failure_rate"] = fail_rate
        agg["empty_rate"] = empty_rate
        agg["error_reasons"] = merged_reasons
        # Report the input shape so rows aren't unlabeled. For MM this is the ACTUAL
        # prefill (clamped text + 4 image placeholder tokens) carried on the raw
        # result; for text it's the campaign's nominal input length.
        if multimodal:
            _first = valid[0]
            agg["input_length"] = _first.get("input_length")
            agg["input_length_nominal"] = _first.get("input_length_nominal")
            agg["mm_images"] = _first.get("mm_images", self._images_per_request())
            agg["per_image_tokens"] = _first.get("per_image_tokens")
            agg["per_image_tokens_source"] = _first.get("per_image_tokens_source")
        else:
            agg["input_length"] = (self.config.input_lengths[0]
                                   if self.config.input_lengths else None)
        # ACTUAL server-measured prefill (usage.prompt_tokens): the median over every
        # measured iteration (each already the median of its requests), never the first
        # iteration alone. The difference to the nominal length is the chat template /
        # system prompt the server adds.
        _acts = [r.get("input_length_actual") for r in valid
                 if isinstance(r.get("input_length_actual"), (int, float))
                 and r.get("input_length_actual") > 0]
        if _acts:
            agg["input_length_actual"] = int(np.median(_acts))
            if isinstance(agg.get("input_length"), (int, float)):
                agg["input_length_template_tokens"] = (agg["input_length_actual"]
                                                       - int(agg["input_length"]))
        agg["output_length"] = (self.config.output_lengths[0]
                                if self.config.output_lengths else None)
        # Realized decode length: a remote backend that ignores ignore_eos (e.g.
        # Ollama) stops at natural EOS, so the nominal output_length is NOT what
        # was generated - record the measured median so the JSON is honest.
        # Exclude non-positive lengths (the LOCAL vLLM benchmark() path records 0
        # for failed/aborted requests; remote/MM already append success-only), so a
        # high-failure local run doesn't drag the realized-length median toward 0.
        _real_lens = [x for x in pooled_out_lens
                      if isinstance(x, (int, float)) and x > 0]
        if _real_lens:
            agg["output_length_realized"] = int(np.median(_real_lens))
        # ignore_eos is only guaranteed honored by a LOCAL vLLM engine; a remote
        # OpenAI-compat endpoint may silently drop it (Ollama does), so the decode
        # shape is deterministic ONLY for local runs. Record requested-vs-honored
        # honestly instead of asserting the backend complied.
        # Observed, not assumed: honoured only when EVERY request returned the requested
        # number of tokens.
        # Per request, against ITS OWN max_tokens, from the server's usage counts only
        # (never chunk counts; failed requests excluded). Not assessed for real datasets
        # (their per-request targets vary) or when the server sends no usage.
        _want = agg.get("output_length")
        _lens = [x for x in pooled_out_lens if isinstance(x, (int, float)) and x > 0]
        _usage = [u for r in valid for u in (r.get("output_lens_usage") or [])
                  if isinstance(u, (int, float)) and u > 0]
        _empty = sum(int(r.get("empty") or 0) for r in valid)     # 200 with no tokens
        if self._real_dataset() or not _usage or not _want:
            _honored, _obs = None, None
        else:
            _n_full = sum(1 for x in _usage if x >= _want)
            _honored = _n_full == len(_usage) and _empty == 0
            _obs = {"n": _n_full, "of": len(_usage) + _empty}
        agg["serving_shape_deterministic"] = _honored
        agg["ignore_eos"] = True                  # requested (back-compat key)
        agg["ignore_eos_honored"] = _honored
        agg["ignore_eos_observed"] = _obs
        # Per-request samples (ms), so the numbers can be re-derived from the JSON.
        agg["samples"] = {"ttft_ms": pooled_ttft, "e2el_ms": pooled_e2e,
                          "tpot_ms": pooled_tpot, "output_tokens": _lens}
        agg["model"] = model.short_name
        agg["model_short"] = model.short_name
        agg["model_name"] = model.name
        agg["format"] = format.value
        agg["batch_size"] = batch_size
        agg["multimodal"] = multimodal
        self._stamp_server(agg, model, format)
        cache = {}
        for r in results:
            for k, v in ((r or {}).get("cache") or {}).items():
                if v is not None:
                    cache[k] = cache.get(k, 0.0) + float(v)
        # Client evidence that the requested concurrency was really reached.
        _peaks = [r.get("peak_in_flight") for r in valid if r.get("peak_in_flight") is not None]
        if _peaks:
            agg["peak_in_flight"] = max(_peaks)
            agg["peak_in_flight_per_iteration"] = _peaks
            agg["batch_size_reached"] = max(_peaks) >= batch_size
            if not agg["batch_size_reached"]:
                logger.warning(f"Serving batch {batch_size}: the client never had more than "
                               f"{max(_peaks)} requests in flight (fewer requests than the "
                               f"batch size, or the endpoint slowed the client).")
            agg["client_loop_lag_max_ms"] = max(r.get("client_loop_lag_max_ms") or 0.0
                                                for r in valid)
        # Iterations that failed are not in the pooled numbers: say how many ran.
        agg["iterations_requested"] = int(getattr(self.config, "num_iterations", len(results)))
        agg["iterations_ok"] = len(valid)     # the iterations the pooled numbers use
        if agg["iterations_ok"] < agg["iterations_requested"]:
            logger.warning(f"Serving: only {agg['iterations_ok']}/{agg['iterations_requested']} "
                           f"iterations succeeded; the numbers pool those only.")
        agg["itl_kind"] = "inter_chunk"        # gaps between streamed chunks
        # per_token: (e2e - ttft) / (tokens - 1) from the server's counts; chunk mean
        # for requests whose server sent no usage.
        n_tp = n_fb = 0
        for r in valid:                        # the iterations the pooled numbers use
            try:
                x = self._extract_raw_samples(r)
            except Exception:                                             # noqa: BLE001
                continue
            n_tp += len(x["tpot_ms"])
            n_fb += x["tpot_fallbacks"]
        agg["tpot_kind"] = ("per_token" if n_fb == 0 else
                            "inter_chunk_mean" if n_fb >= n_tp else
                            f"mixed ({n_fb} of {n_tp} requests chunk mean)")

        # Raw cache counters summed over the measured iterations (None without /metrics).
        # Verdict: any multimodal-cache hit means an image was reused (every image is
        # unique by construction). No prefix verdict here: serving does not calibrate the
        # per-request template overhead that bound needs (stress does).
        agg["cache"] = cache or None
        agg["cache_reuse_suspect"] = (
            (["multimodal"] if cache.get("mm_cache_hits") else [])
            if multimodal and "mm_cache_hits" in cache else None)
        agg["cache_telemetry_note"] = ("raw counters; prefix-cache verdict not computed for "
                                       "serving; encoder-cache reuse is not observable")
        if agg["cache_reuse_suspect"]:
            logger.warning(f"MM serving: {cache['mm_cache_hits']:.0f} multimodal-cache hits during "
                           f"the measured iterations (expected 0): something reused an image.")

        # Honest headline log: median (solid) + P99 with CI + validity.
        tv = agg.get("ttft_tail_validity", {}).get("p99", {})
        logger.info(
            f"Pooled {len(pooled_ttft)} req over {len(valid)} iter | "
            f"TTFT P50={agg.get('ttft_p50_ms', float('nan')):.1f}ms "
            f"P99={agg.get('ttft_p99_ms', float('nan')):.1f}ms "
            f"[{agg.get('ttft_p99_ci_low_ms', float('nan')):.1f},"
            f"{agg.get('ttft_p99_ci_high_ms', float('nan')):.1f}] "
            f"(P99 valid={tv.get('valid')}, need n>={tv.get('required_n')}) | "
            f"ITL P50={agg.get('itl_p50_ms', float('nan')):.1f}ms "
            f"P99={agg.get('itl_p99_ms', float('nan')):.1f}ms"
        )

        output_file = self._get_output_path(model, format, batch_size, multimodal)
        self._stamp_campaign(agg)
        write_redacted_json(output_file, agg)
        logger.info(f"Results saved to: {output_file}")
        return agg

    def _aggregate_serving_results_legacy(
        self,
        results: list[dict],
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool,
    ) -> dict:
        """Scalar aggregation for results without raw samples (remote lightweight
        path). Averages per-iteration scalar metrics; kept only for that path."""
        if not results:
            raise RuntimeError("All iterations failed")

        serving_metrics = [
            "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
            "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
            "request_throughput", "output_throughput",
        ]
        aggregated = aggregate_benchmark_results(results, serving_metrics)

        for metric in ["request_throughput", "mean_ttft_ms"]:
            is_valid, msg = validate_repeatability(
                aggregated, metric, self.config.min_acceptable_cv_percent
            )
            logger.info(f"{metric}: {msg}")
            aggregated[f"{metric}_repeatability_valid"] = is_valid

        for metric in ["request_throughput", "mean_ttft_ms", "mean_tpot_ms"]:
            summary = format_statistics_summary(aggregated, metric)
            logger.info(summary)

        aggregated["model"] = model.short_name
        aggregated["model_short"] = model.short_name
        aggregated["model_name"] = model.name
        aggregated["format"] = format.value
        aggregated["batch_size"] = batch_size
        aggregated["multimodal"] = multimodal
        self._stamp_server(aggregated, model, format)
        aggregated["output_token_throughput"] = aggregated.get(
            "output_throughput_mean", aggregated.get("output_throughput", 0.0))
        output_file = self._get_output_path(model, format, batch_size, multimodal)
        self._stamp_campaign(aggregated)
        write_redacted_json(output_file, aggregated)
        logger.info(f"Results saved to: {output_file}")
        return aggregated


    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> list[dict]:
        """Run all serving benchmark configurations for a model.

        Args:
            model: Model configuration
            format: Model format
            multimodal: If True, use random-mm dataset for image+text

        Returns:
            List of aggregated result dictionaries (includes failed runs with failed=True)
        """
        results = []
        configs = self.config.get_serving_configs()

        # Start ONE server and reuse it across all batch sizes. Batch size only
        # changes client concurrency, not the server config, so reloading the
        # model per batch size was pure waste (~3.5min each for a 26B). Safe now
        # that prefix caching is disabled - no cache leaks between configs. Only
        # when we actually own a local server; remote/dry-run keep the inner
        # methods managing (which no-op for those paths).
        manage_here = not self.config.dry_run and not self.config.remote_endpoint
        mm_media_path = None
        server_up = False
        try:
            if manage_here:
                if multimodal:
                    mm_media_path = self._setup_mm_images(num_images=50)
                if not self._start_server(
                    model, format, allowed_local_media_path=mm_media_path,
                    mm_limit_images=self._images_per_request() if multimodal else 1,
                ):
                    if multimodal:
                        self._cleanup_mm_images()
                    raise RuntimeError("Failed to start vLLM server")
                server_up = True

            for cfg in configs:
                try:
                    # Use iteration-aware runner; when we hold the shared server
                    # tell it not to start/stop its own.
                    result = self.run_with_iterations(
                        model,
                        format,
                        batch_size=cfg["batch_size"],
                        num_prompts=cfg["num_prompts"],
                        dataset=cfg.get("dataset"),
                        multimodal=multimodal,
                        manage_server=not manage_here,
                    )
                    # Add config metadata to result
                    result["batch_size"] = cfg["batch_size"]
                    result["multimodal"] = multimodal
                    # P99 reliability: True when this config was sampled below the
                    # stable-P99 threshold (long-output campaigns) - see
                    # BenchmarkConfig._serving_num_prompts.
                    result["low_confidence"] = cfg.get("low_confidence", False)
                    result["num_prompts"] = cfg.get("num_prompts")
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Failed benchmark for {model.short_name}: {e}"
                    )
                    # Track failure in results
                    results.append({
                        "failed": True,
                        "failure_kind": "harness",
                        "error": str(e),
                        "model": model.short_name,
                        "format": format.value,
                        "batch_size": cfg["batch_size"],
                        "multimodal": multimodal,
                    })
        finally:
            if server_up:
                self._cleanup_server()
            if manage_here and multimodal:
                self._cleanup_mm_images()

        return results

    def _server_host_choice(self, context: str) -> Tuple[Optional[str], str]:
        """(--host, why) of a server gbench launches; a host of None is vllm serve's default, all
        interfaces. GBENCH_SERVER_HOST when set; else PERF_SERVER_HOST (127.0.0.1) for a
        performance server; for the quality / Golden Set / eval server, LOOPBACK_HOST when the run
        selects no eval suite (config.evals empty: quality and the Golden Set reach the server
        from gbench's own process), else all interfaces, because task containers of eval suites on
        docker's default bridge reach it at the bridge gateway."""
        override = (os.environ.get(SERVER_HOST_ENV) or "").strip()
        if override:
            bad = server_host_env_error(override)
            if bad:
                raise ValueError(bad)
            return override, SERVER_HOST_ENV
        if context != "native":
            return PERF_SERVER_HOST, "performance server: reached only by gbench"
        if not getattr(self.config, "evals", None):
            return LOOPBACK_HOST, ("no eval suite selected: quality and the Golden Set reach the "
                                   "server from gbench's own process")
        return None, ("eval suites selected: task containers on docker's default bridge reach "
                      "the server at the bridge gateway, so it binds all interfaces (vllm serve's "
                      "default)")

    def _server_host(self, context: str) -> Optional[str]:
        """The --host of a server gbench launches (_server_host_choice); None = all interfaces."""
        return self._server_host_choice(context)[0]

    def _server_env(self) -> Dict[str, str]:
        """The environment of a server gbench launches: gbench's own, with CUDA_VISIBLE_DEVICES
        set to the GPUs the run selected, and VLLM_API_KEY (the only key source of the server:
        --api-key is refused in --vllm-serve-args, config.GBENCH_OWNED_SERVE_ARGS) set to exactly
        the key gbench sends it. vllm serve enforces any non-empty VLLM_API_KEY verbatim, while
        gbench's requests send base.endpoint_api_key's value (surrounding whitespace stripped,
        placeholders such as EMPTY or dummy treated as no key): so a placeholder value is removed
        (the server enforces no key, as gbench sends none) and a real one is passed stripped.
        Records self.server_key_enforced."""
        from .eval_suites.base import endpoint_api_key
        env = {**os.environ,
               **cuda_visible_devices_env(getattr(self.config, "selected_gpus", None))}
        raw = os.environ.get("VLLM_API_KEY")
        key = endpoint_api_key({"VLLM_API_KEY": raw or ""})
        if key:
            env["VLLM_API_KEY"] = key
        else:
            env.pop("VLLM_API_KEY", None)
            if (raw or "").strip():
                logger.info("VLLM_API_KEY holds a placeholder value, which gbench treats as no "
                            "key and never sends: the server gbench launches is started without "
                            "an endpoint key.")
        self.server_key_enforced = bool(key)
        return env

    def build_server_cmd(
        self,
        model: ModelConfig,
        format: ModelFormat,
        context: str = "perf",
        port: Optional[int] = None,
        allowed_local_media_path: Optional[str] = None,
        mm_limit_images: int = 1,
    ) -> list:
        """The `vllm serve` argv of a server gbench launches, without launching anything (the
        dry run prints it; _start_server runs it).

        ``context`` "perf" (serving / stress; the uniform comparison context, prefix caching
        off, bound to 127.0.0.1) or "native" (the eval / quality / Golden Set server: the
        model's own context window unless --max-model-len is given, vLLM's default prefix
        caching, --max-logprobs for mmlu, the operator's --tool-call-parser /
        --reasoning-parser; bound per _server_host_choice). ``port`` defaults to this runner's
        port. The operator's serve settings (config.serve_args_for) come last. Raises ValueError
        for settings that contradict each other."""
        port = port if port is not None else self.server_port
        cmd = [*vllm_cli(), "serve", model.get_model_path(format),
               "--port", str(port) if port is not None else "<port>"]
        host = self._server_host(context)
        if host:
            cmd += ["--host", host]

        gpu_mem = self.config.gpu_memory_utilization
        if gpu_mem:
            cmd += ["--gpu-memory-utilization", str(gpu_mem)]
        if self.config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")

        # Disable automatic prefix caching for PERF benchmarking, as defence in depth:
        # every perf request is already unique by construction (perf_payload: salt +
        # nonce + coded images), and the results record the cache counters they saw.
        # Off = no request can ever be served from a cached prefix, and the server is
        # safe to reuse across configs.
        # Eval/quality servers keep vLLM's default (on): scores do not depend on
        # it, and long-context and multi-turn suites re-send a growing shared
        # prefix every turn, which would otherwise be re-prefilled each time.
        if context != "native":
            cmd.append("--no-enable-prefix-caching")

        if self.config.max_num_batched_tokens:
            cmd += ["--max-num-batched-tokens", str(self.config.max_num_batched_tokens)]
        # Limit max concurrent sequences for uniform memory usage
        if self.config.max_num_seqs:
            cmd += ["--max-num-seqs", str(self.config.max_num_seqs)]

        # Multi-GPU: --num-gpus = tensor-parallel x data-parallel replicas (parallel_layout).
        if self.config.tensor_parallel_size and self.config.tensor_parallel_size > 1:
            cmd += ["--tensor-parallel-size", str(self.config.tensor_parallel_size)]
        dp = int(getattr(self.config, "data_parallel_size", 1) or 1)
        if dp > 1:
            cmd += ["--data-parallel-size", str(dp)]

        # Perf pillars: uniform context length for fair comparison (overridable via
        # --max-model-len for long-context campaigns). Evals/quality: the model's native
        # window. Serving an eval at 4096 silently clamped every answer budget to what was
        # left of 4096 tokens (a 65536-token thinking budget became <4k) and broke the
        # long-context suites.
        # Native servers look only at the USER's --max-model-len: a --campaign also writes
        # config.max_model_len (its perf window), which must not cap an eval server.
        explicit = getattr(self.config, "max_model_len_explicit", None)
        if not (context == "native" and not explicit):
            max_model_len = (explicit if context == "native"
                             else get_max_model_len(self.config.max_model_len))
            cmd += ["--max-model-len", str(max_model_len)]

        # Eval and quality servers: mmlu sends hendrycks/test evaluate.py's logprobs=100,
        # which vLLM rejects above its default --max-logprobs 20. The cap only validates
        # requests; it changes no suite's request or score.
        if context == "native":
            cmd += ["--max-logprobs", str(EVAL_SERVER_MAX_LOGPROBS)]

        # Allow loading local media files (needed for multimodal benchmarks)
        if allowed_local_media_path:
            cmd += ["--allowed-local-media-path", allowed_local_media_path]

        # Allow multiple images per request (default vLLM cap is 1). Needed for
        # multimodal stress, which sends IMAGES_PER_REQUEST images per prompt.
        # vLLM expects a JSON dict value, e.g. {"image": 4} - NOT image=4.
        if mm_limit_images and mm_limit_images > 1:
            cmd += ["--limit-mm-per-prompt", json.dumps({"image": mm_limit_images})]
            # Disable the multimodal preprocessor cache so repeated images are
            # re-preprocessed every request - consistent with APC being off, so
            # host-side image preprocessing is actually measured rather than
            # served from a 4GB cache after the first hit.
            cmd += ["--mm-processor-cache-gb", "0"]

        extra = serve_args_for(self.config, context)
        # --eval-max-soft-tokens: the model processor's max_soft_tokens for vision evals, on the
        # eval server only. It is a processor setting, so it goes in --mm-processor-kwargs (the
        # kwargs vLLM hands the HF processor); --hf-overrides changes the model config, where
        # the processor never reads it.
        soft = getattr(self.config, "eval_max_soft_tokens", None)
        if soft is not None and context == "native":
            if any(t.split("=", 1)[0] in ("--mm-processor-kwargs", "--mm_processor_kwargs")
                   for t in extra):
                raise ValueError("--eval-max-soft-tokens and a --mm-processor-kwargs in "
                                 "--vllm-serve-args both set the processor's kwargs: put "
                                 "max_soft_tokens inside --mm-processor-kwargs instead")
            cmd += ["--mm-processor-kwargs", json.dumps({"max_soft_tokens": int(soft)})]

        # The operator's model-specific settings, last (config.serve_args_for).
        cmd += extra
        return cmd

    def _start_server(
        self,
        model: ModelConfig,
        format: ModelFormat,
        allowed_local_media_path: Optional[str] = None,
        mm_limit_images: int = 1,
        context: str = "perf",
    ) -> bool:
        """Start a vLLM server for this model and wait until it is ready and verified as ours.

        The port: resolve_server_port() (GBENCH_SERVER_PORT when set and free, else an error
        naming it; unset, the first free port from 8000), stored in self.server_port for every
        consumer. The command: build_server_cmd(). The environment: _server_env()
        (CUDA_VISIBLE_DEVICES set to the GPUs the run selected, config.selected_gpus, and
        VLLM_API_KEY set to exactly the key gbench sends). Ready means: the child is alive, the
        port's listener belongs to the child's process group (not a leftover or foreign server
        that answers /health on the same port), and /v1/models lists the model it was started
        with. An automatically chosen port that another process takes before the server binds
        it is retried with the next free port (PORT_RACE_ATTEMPTS).

        Args:
            model: Model configuration
            format: Model format
            allowed_local_media_path: If set, allow loading local media from this path
            context: "perf" serves the uniform 4096-token context the perf pillars compare
                at; "native" (evals, quality, Golden Set) serves the model's own context window.
                An explicit --max-model-len wins in both cases.

        Returns:
            True if server started successfully, False otherwise
        """
        if self.server_process is not None:
            logger.warning("Server already running, cleaning up first")
            self._cleanup_server()

        logger.info(f"Starting vLLM server for {model.name} ({format.value})")
        if context == "native" and not getattr(self.config, "max_model_len_explicit", None):
            logger.info("Serving the model's native context window (pass --max-model-len "
                        "to cap it, or -1 to fit it to GPU memory, if its KV cache does "
                        "not fit)")

        for attempt in range(1, PORT_RACE_ATTEMPTS + 1):
            port, source = resolve_server_port()
            if port is None:
                logger.error(f"Cannot start the vLLM server: {source}")
                return False
            self.server_port, self.server_port_source = port, source
            self.server_host, self.server_host_source = self._server_host_choice(context)
            # The endpoint lock first: a port another gbench run holds fails before the
            # model is loaded.
            if not self._hold_server_lock(context):
                return False
            outcome = self._launch_and_wait(model, format, context, allowed_local_media_path,
                                            mm_limit_images)
            if outcome == "ready":
                return True
            self._release_server_lock()
            if outcome == "port_taken" and source == "auto" and attempt < PORT_RACE_ATTEMPTS:
                logger.warning(f"Port {port} was taken by another process before the server "
                               "bound it; retrying on the next free port")
                continue
            if outcome == "port_taken":
                logger.error(f"Port {port} is in use by another process (it answered on the port "
                             "gbench chose before the server bound it)"
                             + ("; set GBENCH_SERVER_PORT to a free port"
                                if source != "auto" else ""))
            return False
        return False

    def _launch_and_wait(self, model: ModelConfig, format: ModelFormat, context: str,
                         allowed_local_media_path: Optional[str],
                         mm_limit_images: int) -> str:
        """One launch attempt on self.server_port: "ready", "port_taken" (another process
        answers on the port) or "failed". Cleans up after itself unless ready."""
        try:
            cmd = self.build_server_cmd(model, format, context=context, port=self.server_port,
                                        allowed_local_media_path=allowed_local_media_path,
                                        mm_limit_images=mm_limit_images)
        except ValueError as e:
            logger.error(f"Cannot start the vLLM server: {e}")
            return "failed"
        server_log_path = None
        try:
            # Redirect server output to log file to avoid pipe buffer deadlock.
            # When using PIPE, the OS buffer (~64KB) can fill up during model
            # loading and CUDA graph capture, causing the server to block.
            # Writing to a file avoids this issue while preserving logs.
            lm = self.config.log_manager
            server_log_path = lm.get_server_log_path(model.short_name, format.value)
            logger.info(f"Server logs will be written to: {server_log_path}")
            logger.info("Server command: " + " ".join(str(c) for c in cmd))

            env = self._server_env()
            self._server_log_file = open(server_log_path, "a")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=self._server_log_file,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                start_new_session=True,  # New process group for clean killpg
                env=env,
            )
            process_groups.track(self.server_process)

            # Load time scales with the model size (torch.compile + CUDA graphs on first run).
            timeout = get_server_timeout(model.total_params_b, model.is_moe)
            status = self._wait_for_server_ready(
                timeout, expected_model_id=model.get_model_path(format))
            if status is True:
                logger.info(f"vLLM server started successfully on port {self.server_port}")
                from gbench.utils import verify_endpoint_functional
                is_up, err_msg, max_model_len = verify_endpoint_functional(
                    f"http://127.0.0.1:{self.server_port}", model_id=model.get_model_path(format))
                if not is_up:
                    logger.error(f"Local server on port {self.server_port} is not functional: "
                                 f"{err_msg}")
                    self._cleanup_server()
                    return "failed"
                logger.info(f"Local server on port {self.server_port} is functional and "
                            "answering requests.")
                if max_model_len:
                    logger.info(f"   Server reported max_model_len: {max_model_len} tokens")
                self._served[(model.name, format.value)] = {
                    "served_model_id": model.get_model_path(format),
                    "served_model_id_source": "the model argument of the server gbench launched"}
                return "ready"
            if status == "port_taken":
                self._cleanup_server()
                return "port_taken"
            logger.error(f"Server failed to become ready. Check logs at: {server_log_path}")
            if context == "native" and not getattr(self.config, "max_model_len_explicit", None):
                logger.error("It was serving the model's native context window. If the log "
                             "shows a KV-cache or max_model_len error, cap the window with "
                             "--max-model-len <tokens>, or pass -1 to fit it to GPU memory.")
            self._cleanup_server()
            return "failed"

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self._cleanup_server()
            return "failed"

    def _wait_for_server_ready(self, timeout: int = 120,
                               expected_model_id: Optional[str] = None):
        """Wait until the server this runner started is ready and is the one answering.

        Each poll first checks that the child is alive (a dead child is never "ready", whatever
        answers on the port). A /health 200 counts only when the port's listeners all belong to
        the child's process group (listener_owned_by_session; a leftover or foreign server on the
        port, or a second SO_REUSEPORT listener, returns "port_taken"), and, when
        ``expected_model_id`` is given, /v1/models must list it.

        Returns:
            True when ready; "port_taken" when another process answers on the port; False when
            the child died, the served model is not the expected one, or the timeout passed.
        """
        url = f"http://127.0.0.1:{self.server_port}/health"
        start_time = time.time()

        logger.info("Waiting for server to be ready...")

        while time.time() - start_time < timeout:
            proc = self.server_process
            if proc is not None and proc.poll() is not None:
                logger.error("Server process died during startup (see its log)")
                return False
            try:
                response = requests.get(url, timeout=1)
                healthy = response.status_code == 200
            except requests.exceptions.RequestException:
                healthy = False
            if healthy:
                # A GBENCH_VLLM_BIN command may run the server outside gbench's process tree
                # (a container, a wrapper that detaches), so ownership is not checked for it;
                # the model listing still is.
                owned = (listener_owned_by_session(self.server_port, proc.pid)
                         if proc is not None and not os.environ.get(VLLM_BIN_ENV, "").strip()
                         else None)
                if owned is False:
                    logger.error(f"Another process answers on port {self.server_port} (the "
                                 "listener is not the server gbench started)")
                    return "port_taken"
                if expected_model_id and not self._lists_model(expected_model_id):
                    logger.error(f"The server on port {self.server_port} does not serve "
                                 f"'{expected_model_id}': a different server answered")
                    return False
                if owned is None:
                    logger.info("Could not confirm from /proc that the port's listener is the "
                                "server gbench started; the model listing matched.")
                return True
            time.sleep(2)

        return False

    def _lists_model(self, model_id: str) -> bool:
        """Whether this runner's server lists ``model_id`` at /v1/models."""
        from ..utils.endpoint import EndpointError, list_served_models
        try:
            entries = list_served_models(f"http://127.0.0.1:{self.server_port}/v1")
        except EndpointError as e:
            logger.error(str(e))
            return False
        return model_id in [str(m.get("id")) for m in entries]

    def _hold_server_lock(self, context: str) -> bool:
        """Hold the endpoint lock (utils.endpoint_lock) on the port this runner is about to
        serve on, until _cleanup_server: exclusive for a performance server, shared for the
        native server of evals, quality and the Golden Set. Another gbench run pointed at
        this port then fails fast instead of sending traffic into a measurement. False when
        another gbench run already holds it."""
        import contextlib
        from ..utils import endpoint_lock
        self._release_server_lock()
        key = f"local:{self.server_port}"      # == endpoint_key() for any local address
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(endpoint_lock.hold(key, exclusive=(context == "perf"),
                                                   log=logger.warning))
        except endpoint_lock.EndpointBusy as e:
            logger.error(f"{e}")
            return False
        self._server_lock = stack
        return True

    def _stamp_campaign(self, record: Dict[str, Any]) -> None:
        """Record the campaign in a result file (its name carries it too)."""
        camp = getattr(self.config, "campaign", None)
        if isinstance(camp, str) and camp and isinstance(record, dict):
            record.setdefault("campaign", camp)

    def _release_server_lock(self) -> None:
        if self._server_lock is not None:
            try:
                self._server_lock.close()
            finally:
                self._server_lock = None

    def _cleanup_server(self):
        """Clean up the vLLM server process group this runner started (idempotent).

        Note: After killing the process, we wait for GPU memory to be
        reclaimed by the CUDA driver. This typically takes 5-15 seconds.
        """
        if self.server_process is None:
            self._release_server_lock()
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
                self._server_log_file = None
            return

        logger.info("Cleaning up vLLM server...")

        try:
            # Kill the entire process group (API server + EngineCore + workers)
            # vLLM spawns EngineCore as a grandchild process that terminate()
            # cannot reach, leaving zombie GPU-holding processes.
            process_groups.stop_group(self.server_process, grace_s=10.0)
            logger.info("vLLM server stopped")

            # Wait for GPU memory to be reclaimed by CUDA driver
            # Use active polling instead of fixed wait for reliability
            self._wait_for_gpu_memory()

        except Exception as e:
            logger.error(f"Error cleaning up server: {e}")
        finally:
            self.server_process = None
            self._release_server_lock()
            # Close the log file handle
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
                self._server_log_file = None
