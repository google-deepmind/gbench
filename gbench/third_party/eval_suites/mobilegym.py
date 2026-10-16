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

# Evaluation Suites: mobilegym, mobilegym_pp
# Description: MobileGym (256-task test split, pure-GUI mobile benchmark) and MobileGym++
#              (215-task official benchmark, hybrid app-tools + GUI mobile benchmark)

"""gbench native built-in runners for `mobilegym` and `mobilegym_pp` (Tool Use & Agentic Workflows).

* `mobilegym` runs the canonical 256-task held-out test split (`bench_env/splits/test.txt`)
  from `Purewhiter/mobilegym@5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38` with the canonical
  `GenericAgentV2` pure-GUI agent (`coord_space="norm_0_1000"`, `eval_mode="grounded"`,
  `judge_mode="state"`, `sample_seed=42`).
* `mobilegym_pp` runs the canonical 215-task official benchmark (`bench215/official.json`)
  from `OpenMobile2/MobileGym-plusplus@cdde4ad06ed67176b56964dd329b8cb8e0b0001c` with the
  canonical `Qwen35SessionAgent` hybrid session runtime (`interaction_mode="hybrid"`,
  `coord_space="norm_0_1000"`).

Both suites run completely offline inside the local `gbench-mobilegym:5fbd3f34` image
(`gbench/docker/mobilegym.Dockerfile`) with:
* deterministic per-task programmatic state verifiers (zero LLM/VLM judge calls, no `GEMINI_API_KEY`);
* frozen simulator clock (`2026-07-12 09:00:00`) and matching Python `default_tomorrow()` (`2026-07-13`);
* cloud metadata server guard (`container_guard.require_metadata_blocked`), `EndpointRelay` for model
  calls, and `DetokenizeRelay` for `gbench`'s server-parser recovery policy;
* deterministic modulo sharding (`flat[I-1::N]`) followed by seeded difficulty-stratified `--eval-limit`
  (`sampling.stratified_indices(key=difficulty, seed=<suite>)`).

Sampling: default temperature 0.0 (1.0 with --thinking); override with --temperature or
GBENCH_MOBILEGYM_TEMPERATURE.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from . import container_guard, gemini_relay, swe_agentic
from .base import (
    SHARD_REASON,
    current_shard,
    endpoint_api_key,
    record_shard,
    resolve_temperature,
)
from .gaia2 import DetokenizeRelay
from .limit_plan import eval_limit_plan, merge_plans
from .swebench_common import EndpointRelay, infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/mobilegym.md"

_IMAGE_DEFAULT = "gbench-mobilegym:5fbd3f34"
BUILD_HINT = (
    "docker build -t gbench-mobilegym:5fbd3f34 -f gbench/docker/mobilegym.Dockerfile gbench/docker"
)
UPSTREAM_REPO = "https://github.com/Purewhiter/mobilegym"
UPSTREAM_COMMIT = "5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38"
UPSTREAM_DATA_SHA256 = "933a8499c71450f135eda70f9107af72b56b42181fc62de76ed1aa4c513a603c"
SIMULATED_TIME = "2026-07-12 09:00:00"

LAUNCHER_LABEL = "org.gbench.mobilegym-launcher"
LAUNCHER_VERSION = "2"
NETWORK = "bridge"

MOBILEGYM_FULL_COUNT = 256
MOBILEGYM_PP_FULL_COUNT = 215
MIN_INOTIFY_INSTANCES = 1024

MODEL_KEY_PLACEHOLDER = "dummy"
_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "VLLM_API_KEY",
    "GEMINI_API_KEY",
    "GEMINI_API_KEYS",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def _image() -> str:
    return (os.environ.get("GBENCH_MOBILEGYM_IMAGE") or "").strip() or _IMAGE_DEFAULT


def _docker_image_exists(image: str) -> bool:
    return (
        subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    )


def image_launcher_version(image: str) -> Optional[str]:
    r = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            '{{index .Config.Labels "' + LAUNCHER_LABEL + '"}}',
            image,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    val = (r.stdout or "").strip()
    return "" if val == "<no value>" else val


def _check_inotify_ceiling() -> Optional[str]:
    path = "/proc/sys/fs/inotify/max_user_instances"
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            val = int(f.read().strip())
    except Exception:
        return None
    if val < MIN_INOTIFY_INSTANCES:
        return (
            f"host fs.inotify.max_user_instances={val} is below {MIN_INOTIFY_INSTANCES} "
            "(Playwright Chromium + Vite file watchers exhaust low inotify limits and cause "
            "60s _wait_ready timeouts); run `sudo sysctl -w fs.inotify.max_user_instances=8192`."
        )
    return None


def _check_common_prerequisites() -> Tuple[bool, str]:
    image = _image()
    build = (
        f"Build the local MobileGym image (gbench never pulls it):\n"
        f"  {BUILD_HINT}\n"
        f"See {DOCS_URL} for provisioning details."
    )
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build

    missing: List[str] = []
    if not _docker_image_exists(image):
        missing.append(f"local image {image!r} not found. " + build)
    else:
        found = image_launcher_version(image)
        if found != LAUNCHER_VERSION:
            missing.append(
                f"local image {image!r} predates the current launcher "
                f"(label {LAUNCHER_LABEL}={LAUNCHER_VERSION}, found {found or 'none'}); "
                f"rebuild it. {build}"
            )
    inotify_err = _check_inotify_ceiling()
    if inotify_err:
        missing.append(inotify_err)
    if missing:
        return False, " ".join(f"({i}) {m}" for i, m in enumerate(missing, 1))
    return True, ""


def check_mobilegym_prerequisites() -> Tuple[bool, str]:
    """Docker + local `gbench-mobilegym:5fbd3f34` image at current launcher + host inotify ceiling."""
    return _check_common_prerequisites()


def check_mobilegym_pp_prerequisites() -> Tuple[bool, str]:
    """Docker + local `gbench-mobilegym:5fbd3f34` image at current launcher + host inotify ceiling."""
    return _check_common_prerequisites()


def _model_endpoint(base_url: str) -> str:
    ep = (base_url or "").rstrip("/")
    return ep if ep.endswith("/v1") else ep + "/v1"


def _resolve_workers(concurrency: int, sandboxes: Optional[int] = None) -> Tuple[int, str]:
    raw = (os.environ.get("GBENCH_MOBILEGYM_WORKERS") or "").strip()
    if raw:
        try:
            val = int(raw)
        except ValueError as exc:
            raise infra_required(
                "mobilegym",
                f"GBENCH_MOBILEGYM_WORKERS must be a positive integer, got {raw!r}.",
                DOCS_URL,
            ) from exc
        if val <= 0:
            raise infra_required(
                "mobilegym",
                f"GBENCH_MOBILEGYM_WORKERS must be a positive integer, got {raw!r}.",
                DOCS_URL,
            )
        return val, "GBENCH_MOBILEGYM_WORKERS"
    if sandboxes is not None:
        return max(1, min(int(sandboxes), 32)), "--sandboxes (capped at 32 browser workers)"
    return max(1, min(int(concurrency or 2), 16)), "concurrency (capped at 16 browser workers)"


def _limit_draw(suite_name: str) -> str:
    return f"stratified_indices(key=difficulty, seed={suite_name})"


def _full_count(suite_name: str) -> int:
    return MOBILEGYM_PP_FULL_COUNT if suite_name == "mobilegym_pp" else MOBILEGYM_FULL_COUNT


def _suite_plan(
    suite_name: str,
    limit: Optional[int],
    shard_label: Optional[str],
) -> Dict[str, Any]:
    full_n = _full_count(suite_name)
    if limit or not shard_label:
        return eval_limit_plan(
            suite_name,
            limit,
            min(int(limit), full_n) if limit else full_n,
            attempts=1,
            unit="episodes",
        )
    return {
        "limit": None,
        "samples": None,
        "per_setting": None,
        "attempts_per_sample": 1,
        "episodes": None,
    }


def _plan_from_summary(plan: Dict[str, Any], n_selected: int) -> Dict[str, Any]:
    out = dict(plan or {})
    n = int(n_selected or 0)
    if out.get("samples") is not None and out["samples"] != n:
        out["samples_logged"] = out["samples"]
    out["samples"] = n
    out["episodes"] = n
    return out


def _category_accuracy(scored: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_cat: Dict[str, List[float]] = collections.defaultdict(list)
    for t in scored:
        tid = str(t.get("task_id") or "")
        cat = str(t.get("category") or (tid.split(".", 1)[0] if "." in tid else tid) or "unknown")
        by_cat[cat].append(float(t.get("score") or 0.0))
    return {
        cat: {
            "total": len(vals),
            "correct": int(round(sum(vals))),
            "accuracy": round(100.0 * sum(vals) / len(vals), 2),
        }
        for cat, vals in sorted(by_cat.items())
        if vals
    }


def _task_traces(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in summary.get("per_task") or []:
        if not isinstance(item, dict) or not item.get("task_id"):
            continue
        tid = str(item["task_id"])
        cat = tid.split(".", 1)[0] if "." in tid else tid
        is_harness = bool(item.get("harness_error"))
        score_val = None if is_harness else (1.0 if item.get("success") else 0.0)
        row: Dict[str, Any] = {
            "sample_key": f"task:{tid}",
            "task_id": tid,
            "category": cat,
            "difficulty": str(item.get("difficulty") or "L2"),
            "apps": list(item.get("apps") or []),
            "app_count": int(item.get("app_count") or len(item.get("apps") or []) or 1),
            "score": score_val,
            "success": bool(item.get("success")) and not is_harness,
            "goal_success": bool(item.get("goal_success")) and not is_harness,
            "clean": bool(item.get("clean", True)),
            "progress": 0.0 if is_harness else float(item.get("progress") or 0.0),
            "false_complete": False if is_harness else bool(item.get("false_complete")),
            "overdue_termination": False if is_harness else bool(item.get("overdue_termination")),
            "steps": int(item.get("steps") or 0),
            "max_steps": int(item.get("max_steps") or 0),
            "stop_reason": str(item.get("stop_reason") or ""),
            "runtime_s": float(item.get("runtime_s") or 0.0),
            "llm_calls": int(item.get("llm_calls") or 0),
            "prompt_tokens": int(item.get("prompt_tokens") or 0),
            "completion_tokens": int(item.get("completion_tokens") or 0),
            "server_parser_moved_text": int(item.get("server_parser_moved_text") or 0),
            "server_parser_dropped_text": int(item.get("server_parser_dropped_text") or 0),
            "server_parser_text_lost": int(item.get("server_parser_text_lost") or 0),
            "stopped_in_reasoning": int(item.get("stopped_in_reasoning") or 0),
        }
        if "strict_success" in item:
            row["strict_success"] = bool(item.get("strict_success")) and not is_harness
        if "tool_calls" in item:
            row["tool_calls"] = list(item.get("tool_calls") or [])
        if "app_tool_calls" in item:
            row["app_tool_calls"] = list(item.get("app_tool_calls") or [])
        if item.get("error"):
            row["error"] = str(item["error"])
        if is_harness:
            row["harness_error"] = str(item.get("harness_error_reason") or "harness_error")
            row["scoring_excluded"] = True
        rows.append(row)
    return rows


def _empty_shard_result(
    suite_name: str,
    model_name: str,
    shard_label: str,
    summary: Dict[str, Any],
    workload: Dict[str, Any],
) -> Dict[str, Any]:
    n_full = (summary.get("shard") or {}).get("n_full") or _full_count(suite_name)
    reason = f"empty shard {shard_label}: no items on this node ({n_full} in the full set)"
    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": suite_name,
        "model_name": model_name,
        "status": "success",
        "status_reason": reason,
        "total_questions": 0,
        "accuracy": None,
        "accuracy_withheld_reason": reason,
        "empty_shard": True,
        "full_set_items": n_full,
        "n_selected": 0,
        "n_scored": 0,
        "n_endpoint_failed": 0,
        "sample_traces": [],
        "selection": summary.get("selection"),
        "eval_limit_plan": _plan_from_summary(workload, 0),
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": reason,
    }
    record_shard(result, shard_label)
    return result


def _run_suite_common(
    suite_name: str,
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 2,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    shard_spec = current_shard()
    shard_label = shard_spec.label if shard_spec and shard_spec.partial else None
    if limit is not None and int(limit) <= 0:
        raise infra_required(
            suite_name,
            f"--eval-limit must be a positive integer, got {limit!r}.",
            DOCS_URL,
        )

    ok, reason = _check_common_prerequisites()
    if not ok:
        raise infra_required(suite_name, reason, DOCS_URL)

    image = _image()
    container_guard.require_metadata_blocked(suite_name, image, NETWORK, DOCS_URL)
    gateway = gemini_relay.network_gateway(suite_name, NETWORK, DOCS_URL)

    temperature, temperature_source = resolve_temperature(
        suite_name,
        thinking=bool(enable_thinking),
    )
    run_max_tokens = swe_agentic.agent_max_tokens_from_run(kwargs.get("max_output_tokens"))
    turn_tokens = int(run_max_tokens or 4096)
    max_tokens_source = (
        "--max-output-tokens" if run_max_tokens else "default (4096)"
    )
    workers, workers_source = _resolve_workers(concurrency, kwargs.get("sandboxes"))
    workload = _suite_plan(suite_name, limit, shard_label)

    model_endpoint = _model_endpoint(base_url)
    model_key = endpoint_api_key()
    workdir = tempfile.mkdtemp(prefix=f"gbench_{suite_name}_")
    os.chmod(workdir, 0o777)
    orch_name = f"gbench_{suite_name}_{os.path.basename(workdir)}"

    def _reap() -> None:
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    model_relay: Optional[EndpointRelay] = None
    detok: Optional[DetokenizeRelay] = None
    try:
        try:
            model_relay = EndpointRelay(
                model_endpoint,
                gateway,
                key=model_key,
                body_fields=swe_agentic.with_run_max_tokens(
                    {"temperature": float(temperature)},
                    turn_tokens,
                ),
                name=f"{suite_name}-model-relay",
            ).start()
        except Exception as exc:
            raise infra_required(
                suite_name,
                f"gbench's model endpoint relay could not start on {gateway} ({type(exc).__name__}: {exc}).",
                DOCS_URL,
            ) from exc

        try:
            detok = DetokenizeRelay(gateway, name=f"{suite_name}-detokenize").start()
        except OSError as exc:
            raise infra_required(
                suite_name,
                f"gbench's token-id decoder could not listen on {gateway} ({type(exc).__name__}: {exc}).",
                DOCS_URL,
            ) from exc

        cmd: List[str] = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--name",
            orch_name,
            "--network",
            NETWORK,
            "--shm-size",
            "16g",
            "-v",
            f"{workdir}:/out:rw",
            "-e",
            "OPENAI_API_KEY",
            "-e",
            f"GBENCH_DETOKENIZE_URL={detok.url}",
            "-e",
            f"NO_PROXY=127.0.0.1,localhost,{gateway}",
            "-e",
            f"no_proxy=127.0.0.1,localhost,{gateway}",
            image,
            "--suite",
            suite_name,
            "--model",
            model_name,
            "--api-base",
            model_relay.url,
            "--api-key",
            MODEL_KEY_PLACEHOLDER,
            "--output-dir",
            "/out",
            "--workers",
            str(workers),
            "--temperature",
            str(temperature),
            "--max-tokens",
            str(turn_tokens),
            "--sample-seed",
            "42",
        ]
        if enable_thinking:
            cmd.append("--enable-thinking")
        if limit is not None:
            cmd += ["--limit", str(int(limit))]
        if shard_spec and shard_spec.partial:
            cmd += [
                "--shard-index",
                str(shard_spec.index),
                "--num-shards",
                str(shard_spec.count),
            ]

        timeout_s = int(os.environ.get("GBENCH_MOBILEGYM_TIMEOUT", str(24 * 60 * 60)))
        run_env = {k: v for k, v in os.environ.items() if k not in _CREDENTIAL_VARS}
        run_env["OPENAI_API_KEY"] = MODEL_KEY_PLACEHOLDER
        stop_progress = threading.Event()
        expected_total = int(workload.get("samples") or _full_count(suite_name))

        def _watch_progress() -> None:
            try:
                from tqdm import tqdm
            except Exception:
                return
            seen_ids: set[str] = set()
            passed_cnt = 0
            with tqdm(
                total=expected_total,
                desc=f"Eval [{suite_name.upper()}]",
                unit="ep",
            ) as pbar:
                while not stop_progress.wait(timeout=1.5):
                    for root, _, files in os.walk(workdir):
                        if "results.jsonl" not in files:
                            continue
                        rpath = os.path.join(root, "results.jsonl")
                        try:
                            with open(rpath, "r", encoding="utf-8") as rf:
                                lines = rf.read().splitlines()
                        except Exception:
                            continue
                        for line in lines:
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            tid = str(row.get("id") or row.get("task_id") or "")
                            if not tid or tid in seen_ids:
                                continue
                            seen_ids.add(tid)
                            ok = bool(
                                row.get("is_success")
                                if suite_name == "mobilegym"
                                else row.get("success")
                            )
                            if ok:
                                passed_cnt += 1
                            pbar.update(1)
                            acc = (passed_cnt / len(seen_ids) * 100.0) if seen_ids else 0.0
                            pbar.set_postfix(
                                passed=f"{passed_cnt}/{len(seen_ids)} ({acc:.1f}%)"
                            )

        progress_thread = threading.Thread(target=_watch_progress, daemon=True)
        progress_thread.start()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=run_env,
            )
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                suite_name,
                f"the container exceeded GBENCH_MOBILEGYM_TIMEOUT ({timeout_s}s) and was killed.",
                DOCS_URL,
            ) from exc
        finally:
            stop_progress.set()
            progress_thread.join(timeout=2.0)

        summ_path = os.path.join(workdir, f"{suite_name}_summary.json")
        if not os.path.isfile(summ_path):
            raise infra_required(
                suite_name,
                f"the container produced no summary at {summ_path} (rc={proc.returncode}). "
                f"tail: {(proc.stderr or proc.stdout or '')[-800:]}",
                DOCS_URL,
            )
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
    finally:
        _reap()
        if model_relay is not None:
            model_relay.stop()
        if detok is not None:
            detok.stop()
        shutil.rmtree(workdir, ignore_errors=True)

    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    applied = summary.get("shard") if isinstance(summary.get("shard"), dict) else {}
    if shard_label and applied.get("spec") != shard_label:
        raise infra_required(
            suite_name,
            f"--shard {shard_label} was requested but the container summary reported "
            f"{applied or 'missing'}; rebuild {image} ({BUILD_HINT}).",
            DOCS_URL,
        )
    if limit is not None and selection.get("limit_draw") != _limit_draw(suite_name):
        raise infra_required(
            suite_name,
            f"the container summary did not report {_limit_draw(suite_name)!r} for --eval-limit; "
            f"rebuild {image} ({BUILD_HINT}).",
            DOCS_URL,
        )
    if summary.get("empty_selection") or not summary.get("n_selected"):
        if shard_label:
            return _empty_shard_result(suite_name, model_name, shard_label, summary, workload)
        raise infra_required(suite_name, "no tasks were selected to run.", DOCS_URL)

    traces = _task_traces(summary)
    scored = [t for t in traces if not t.get("scoring_excluded")]
    n_selected = len(traces)
    n_scored = len(scored)
    n_harness_errors = n_selected - n_scored
    rate = (sum(float(t["score"]) for t in scored) / n_scored) if n_scored > 0 else None

    full_n = _full_count(suite_name)
    sharded = shard_label is not None
    is_full = not limit and not sharded and n_selected >= full_n

    reasons: List[str] = []
    if not is_full and (limit or not sharded):
        reasons.append(f"subset (limit={limit or 'none'}, selected={n_selected}/{full_n})")
    if sharded and shard_spec is not None:
        reasons.append(
            SHARD_REASON.format(
                label=shard_label,
                index=shard_spec.index,
                count=shard_spec.count,
            )
        )
    if n_harness_errors > 0:
        reasons.append(
            f"{n_harness_errors} task(s) hit a harness/endpoint error and were excluded from the score"
        )

    status_reason = (
        f"harness errors left out of the score: {n_harness_errors} of {n_selected} task(s)"
        if n_harness_errors > 0
        else None
    )
    if rate is None:
        status = "failed"
        status_reason = "nothing was scored: " + (status_reason or "all tasks hit harness errors")
    else:
        status = "completed_with_errors" if n_harness_errors > 0 else "success"

    pr = sum(float(t["progress"]) for t in scored) / n_scored if n_scored > 0 else 0.0
    fc = sum(1 for t in scored if t.get("false_complete")) / n_scored if n_scored > 0 else 0.0
    ot = sum(1 for t in scored if t.get("overdue_termination")) / n_scored if n_scored > 0 else 0.0
    use_rate = sum(1 for t in scored if not t.get("clean", True)) / n_scored if n_scored > 0 else 0.0
    n_correct = int(round(sum(float(t["score"]) for t in scored))) if n_scored > 0 else 0

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": suite_name,
        "model_name": model_name,
        "status": status,
        "status_reason": status_reason,
        "total_questions": n_selected,
        "correct_answers": n_correct,
        "accuracy": round(rate * 100.0, 2) if rate is not None else None,
        **({"accuracy_withheld_reason": status_reason} if rate is None else {}),
        "category_accuracy": _category_accuracy(scored),
        "success_rate": round(rate, 4) if rate is not None else None,
        "progress_rate": round(pr * 100.0, 2),
        "false_complete_rate": round(fc * 100.0, 2),
        "overdue_termination_rate": round(ot * 100.0, 2),
        "unexpected_side_effect_rate": round(use_rate * 100.0, 2),
        "effective_n": n_scored,
        "scoring_excluded": n_harness_errors,
        "n_selected": n_selected,
        "n_scored": n_scored,
        "n_endpoint_failed": n_harness_errors,
        "difficulty_breakdown": summary.get("difficulty_breakdown") or {},
        "app_count_breakdown": summary.get("app_count_breakdown") or {},
        "simulated_time": SIMULATED_TIME,
        "temperature": temperature,
        "temperature_source": temperature_source,
        "max_tokens_sent": turn_tokens,
        "max_output_tokens": turn_tokens,
        "max_output_tokens_source": max_tokens_source,
        "workers": workers,
        "workers_source": workers_source,
        "thinking": bool(enable_thinking),
        "enable_thinking": bool(enable_thinking),
        "selection": selection or None,
        "eval_limit_plan": _plan_from_summary(workload, n_selected),
        "sample_traces": traces,
        "server_parser": summary.get("server_parser") or {},
        "token_usage": summary.get("token_usage") or {},
        "leaderboard_comparable": bool(is_full and n_harness_errors == 0 and status == "success"),
        "leaderboard_comparable_reason": "; ".join(reasons) if reasons else None,
    }
    if model_relay is not None:
        result["model_endpoint_relay"] = model_relay.record()
    if detok is not None:
        result["detokenize_relay"] = detok.record()
    record_shard(result, shard_label)
    return result


def run_mobilegym(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 2,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical MobileGym (256-task test split, pure-GUI GenericAgentV2)."""
    return _run_suite_common(
        "mobilegym",
        model_name=model_name,
        base_url=base_url,
        limit=limit,
        concurrency=concurrency,
        enable_thinking=enable_thinking,
        **kwargs,
    )


def run_mobilegym_pp(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 2,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical MobileGym++ (215-task official.json benchmark, hybrid Qwen35SessionAgent)."""
    return _run_suite_common(
        "mobilegym_pp",
        model_name=model_name,
        base_url=base_url,
        limit=limit,
        concurrency=concurrency,
        enable_thinking=enable_thinking,
        **kwargs,
    )


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-score the union of shards' `sample_traces` for `mobilegym` or `mobilegym_pp`."""
    from ...merge import MergeError

    if not results:
        raise MergeError("mobilegym: cannot merge an empty shard list")
    suite_name = str(results[0].get("eval_name") or "mobilegym")
    full_n = _full_count(suite_name)

    union: Dict[str, Dict[str, Any]] = {}
    for r in results:
        traces = r.get("sample_traces")
        if not isinstance(traces, list):
            raise MergeError(f"{suite_name}: a shard carries no sample_traces")
        for t in traces:
            if isinstance(t, dict) and t.get("sample_key"):
                union.setdefault(str(t["sample_key"]), t)

    traces = sorted(union.values(), key=lambda t: str(t.get("task_id") or t.get("sample_key")))
    scored = [t for t in traces if not t.get("scoring_excluded") and isinstance(t.get("score"), (int, float))]
    n_selected = len(traces)
    n_scored = len(scored)
    n_harness_errors = n_selected - n_scored
    rate = (sum(float(t["score"]) for t in scored) / n_scored) if n_scored > 0 else None

    pr = sum(float(t.get("progress") or 0.0) for t in scored) / n_scored if n_scored > 0 else 0.0
    fc = sum(1 for t in scored if t.get("false_complete")) / n_scored if n_scored > 0 else 0.0
    ot = sum(1 for t in scored if t.get("overdue_termination")) / n_scored if n_scored > 0 else 0.0
    use_rate = sum(1 for t in scored if not t.get("clean", True)) / n_scored if n_scored > 0 else 0.0
    n_correct = int(round(sum(float(t["score"]) for t in scored))) if n_scored > 0 else 0

    by_diff: Dict[str, List[float]] = collections.defaultdict(list)
    by_app_count: Dict[str, List[float]] = collections.defaultdict(list)
    for t in scored:
        diff = str(t.get("difficulty") or "L2")
        by_diff[diff].append(float(t["score"]))
        ac = min(4, max(1, int(t.get("app_count") or len(t.get("apps") or []) or 1)))
        by_app_count[f"{ac}_app"].append(float(t["score"]))

    difficulty_breakdown = {
        k: {"accuracy": sum(v) / len(v), "n": len(v), "passed": int(sum(v))}
        for k, v in sorted(by_diff.items())
    }
    app_count_breakdown = {
        k: {"accuracy": sum(v) / len(v), "n": len(v), "passed": int(sum(v))}
        for k, v in sorted(by_app_count.items())
    }

    ran = [r for r in results if not r.get("empty_shard")]
    merged = dict(ran[0] if ran else results[0])
    for key in (
        "empty_shard",
        "full_set_items",
        "accuracy_withheld_reason",
        "model_endpoint_relay",
        "detokenize_relay",
        "selection",
        "shard",
    ):
        merged.pop(key, None)

    status_reason = (
        f"harness errors left out of the score: {n_harness_errors} of {n_selected} task(s)"
        if n_harness_errors > 0
        else None
    )
    if rate is None:
        status = "failed"
        status_reason = "nothing was scored: " + (status_reason or "no task produced a result")
    else:
        status = (
            "completed_with_errors"
            if (n_harness_errors > 0 or any(r.get("status") == "completed_with_errors" for r in results))
            else "success"
        )

    reasons: List[str] = []
    if n_selected < full_n:
        reasons.append(f"subset (the merged shards hold {n_selected} of {full_n} tasks)")
    if n_harness_errors > 0:
        reasons.append(
            f"{n_harness_errors} task(s) hit a harness/endpoint error and were excluded from the score"
        )

    merged.update(
        {
            "sample_traces": traces,
            "total_questions": n_selected,
            "correct_answers": n_correct,
            "accuracy": round(rate * 100.0, 2) if rate is not None else None,
            "category_accuracy": _category_accuracy(scored),
            "success_rate": round(rate, 4) if rate is not None else None,
            "progress_rate": round(pr * 100.0, 2),
            "false_complete_rate": round(fc * 100.0, 2),
            "overdue_termination_rate": round(ot * 100.0, 2),
            "unexpected_side_effect_rate": round(use_rate * 100.0, 2),
            "difficulty_breakdown": difficulty_breakdown,
            "app_count_breakdown": app_count_breakdown,
            "n_selected": n_selected,
            "n_scored": n_scored,
            "n_endpoint_failed": n_harness_errors,
            "effective_n": n_scored,
            "scoring_excluded": n_harness_errors,
            "selection_by_shard": {
                str(r.get("shard") or i + 1): r.get("selection")
                for i, r in enumerate(results)
            },
            "status": status,
            "status_reason": status_reason,
            "leaderboard_comparable": bool(
                n_selected >= full_n and n_harness_errors == 0 and status == "success"
            ),
            "leaderboard_comparable_reason": "; ".join(reasons) if reasons else None,
        }
    )
    if rate is None:
        merged["accuracy_withheld_reason"] = status_reason
    plan = merge_plans(results)
    if plan is not None:
        merged["eval_limit_plan"] = plan
    return merged
