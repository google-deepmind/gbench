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

# Evaluation Suite: swe_lancer
# Description: SWE-Lancer (OpenAI) - real-world freelance SWE tasks, run with SWE-Lancer's own agent and harness

"""gbench native built-in runner for swe_lancer (Coding & Software Engineering).

SWE-Lancer as its current upstream runs it: openai/frontier-evals `project/swelancer` at
HARNESS_COMMIT (the old openai/SWELancer-Benchmark repository is archived and points there),
with its own solver, SimpleAgentSolver (up to 40 turns; each turn the model replies with one
Python block that runs in the task container, or `<user-tool>`, or DONE), its own task table
(all_swelancer_tasks.csv, TASKS_SHA256: 463 Diamond tasks, 198 IC SWE and 265 SWE Manager), its
own grading (IC SWE: the task's hidden end-to-end tests, up to n_test_runs=3 runs; SWE Manager:
the selected proposal against the correct one) and the task images SWE-Lancer publishes
(swelancer/swelancer_x86_<task>:releasev1 per IC SWE task; swelancer/swelancer_x86_monolith:
releasev1 for SWE Manager, which upstream supports only with that image), with the internet
disabled for the task containers (disable_internet=True, the only setting upstream considers
valid). Both variants run by default, each as upstream's run command does it
(`swelancer.task_type=ic_swe` / `swe_manager`), each with its own headline: pass@1 (tasks solved
/ tasks with a verdict; a rollout that ended in a system error is a harness error, left out of
pass@1 and counted in `harness_errors`) and dollars earned.

The harness runs in the LOCAL image gbench-swe-lancer-harness:51052ced
(gbench/docker/swe_lancer_harness.Dockerfile), docker-out-of-docker, on the host network with
NET_ADMIN: alcatraz adds iptables rules for each task container to the host's DOCKER-USER chain
and removes them afterwards. gbench/docker/swe_lancer_eval.py starts it through
run_swelancer.py's own main and adds only a per-task report file.

The model: SimpleAgentSolver builds its OpenAI client from OPENAI_API_KEY and, for an
`openai/<model>` name, the SDK's OPENAI_BASE_URL. gbench points it at an EndpointRelay (a thread
of the gbench process) that forwards to the served endpoint, adds the endpoint's key (the harness
gets the placeholder EMPTY, so no key is in its environment or the run logs it writes) and sets
the served model's reasoning switch (`chat_template_kwargs.enable_thinking` from --thinking),
which SimpleAgentSolver's request cannot carry.

Sampling: SimpleAgentSolver sends no temperature and no max_tokens (OFFICIAL_SAMPLING); gbench's
relay sends the run's `--max-output-tokens` as max_tokens on every model request, with or without
`--thinking` (swe_agentic.AGENT_MAX_TOKENS_SUITES; `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` sends none,
as upstream), recorded as `agent_max_tokens_from_run` (and in `relay.max_tokens_from_run`). The
temperature follows gbench's rule for every suite: 1.0 for a `--thinking` run (temperature_source
"thinking"), 0.0 without it (temperature_source "default"). Set a temperature for a whole run
with `--temperature`, or for this suite alone with `GBENCH_SWE_LANCER_TEMPERATURE`, which takes
precedence over both. The relay
sets the temperature on every model request and the result records it; since SimpleAgentSolver
sends none, the run is not leaderboard-comparable.

Registered but left out of `--evals all` (evals.NAMED_ONLY_SUITES): `--evals swe_lancer` runs
it. It hard-errors (infra_required) naming whatever is missing: Docker, the docker socket, the
local harness image, or iptables access to the host's DOCKER-USER chain from that image.

Cloud metadata server: before the first task container starts,
swebench_common.require_task_network_blocked checks, with a probe container of the harness image,
that 169.254.169.254 is unreachable from docker's default bridge and from a user-defined bridge
network (the kind alcatraz creates for the task containers, whose internet it disables only once
each has started); infra_required otherwise.
"""

import json
import logging
import os
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import (
    OfficialSampling, gbench_cache_dir, get_run_knob, resolve_official_sampling,
    resolve_token_budget,
)
from . import swebench_common
from .limit_plan import eval_limit_plan
from .sampling import allocate_limit, parse_shard, shard_select, stratified_sample
from .swebench_common import EndpointRelay, infra_required

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_lancer.md"
HARNESS_REPO = "openai/frontier-evals"
#: openai/frontier-evals main (2026-04-21); project/swelancer last changed in 5305e936
#: (2025-12-02).
HARNESS_COMMIT = "51052cede8cc608f95bb00346635e03759013e5a"
#: sha256 of project/swelancer/all_swelancer_tasks.csv at HARNESS_COMMIT.
TASKS_SHA256 = "5c3a6d4570b49be0d9fced98f5b32487420b16f25c98d6658830e31fa03f049a"
_IMAGE_DEFAULT = "gbench-swe-lancer-harness:51052ced"
_DOCKER_SOCKET = "/var/run/docker.sock"
_SWELANCER_DIR = "/opt/frontier-evals/project/swelancer"
_PYTHON = f"{_SWELANCER_DIR}/.venv/bin/python"
_ENTRY = "import swe_lancer_eval; swe_lancer_eval.entry()"

#: The two task types upstream's run command takes (swelancer.task_type), in the paper's order.
VARIANTS = ("ic_swe", "swe_manager")
VARIANT_LABELS = {"ic_swe": "IC SWE", "swe_manager": "SWE Manager"}
#: Tasks and dollars of each variant in the table at HARNESS_COMMIT (all of them in the Diamond
#: set; the release notes: 198 IC SWE tasks of the paper's 237 were kept, verified offline).
VARIANT_TASKS = {"ic_swe": 198, "swe_manager": 265}
VARIANT_AVAILABLE_USD = {"ic_swe": 189300.0, "swe_manager": 264500.0}
SPLIT = "diamond"
#: The images upstream's README runs with (docker_image_prefix / docker_image_tag).
TASK_IMAGE_PREFIX = "swelancer/swelancer_x86"
TASK_IMAGE_TAG = "releasev1"
SOLVER = "swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver"
#: SimpleAgentSolver.run: `max_turns = 40`.
SOLVER_MAX_TURNS = 40
#: runner.* as upstream's README run commands set them (concurrency comes from --sandboxes).
RUNNER_ARGS = {"experimental_use_multiprocessing": "False", "enable_slackbot": "False",
               "recorder": "nanoeval.recorder:dummy_recorder", "max_retries": "2"}
#: project/swelancer/sample.env, which README step "Environment variables" copies to .env (the
#: API keys it lists stay empty: the relay adds the endpoint's key).
SAMPLE_ENV = {"USE_WEB_PROXY": "false", "EXPENSIFY_URL": "https://www.expensify.com/",
              "NEW_EXPENSIFY_URL": "https://new.expensify.com/"}
#: What SimpleAgentSolver sends: client.chat.completions.create(messages, model,
#: reasoning_effort=NotGiven): no temperature, no max_tokens.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"{HARNESS_REPO}@{HARNESS_COMMIT} project/swelancer/swelancer/solvers/"
               "swelancer_agent/solver.py get_model_response"),
    send_temperature=False, send_max_tokens=False)
#: The paper's IC SWE Diamond numbers were measured on 237 tasks; the current release keeps 198.
REFERENCE_NOTE = ("the SWE-Lancer paper's IC SWE Diamond numbers were measured on 237 tasks; the "
                  "current release (2025-07-17) keeps 198 of them, so an IC SWE number compares "
                  "with runs on this release; SWE Manager Diamond is the paper's 265 tasks")


def image() -> str:
    return (os.environ.get("GBENCH_SWE_LANCER_IMAGE") or "").strip() or _IMAGE_DEFAULT


def _build_hint() -> str:
    return (f"Build it LOCALLY (gbench never pulls it):\n  docker build -t {image()} -f "
            "gbench/docker/swe_lancer_harness.Dockerfile gbench/docker\nSee " + DOCS_URL)


def _labels(img: str) -> Tuple[bool, Dict[str, Any]]:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}",
                        img], capture_output=True, text=True)
    if r.returncode != 0:
        return False, {}
    try:
        return True, json.loads(r.stdout or "null") or {}
    except ValueError:
        return True, {}


def check_swe_lancer_prerequisites() -> Tuple[bool, str]:
    """Docker + its socket, the local harness image at HARNESS_COMMIT, and iptables access to
    the host's DOCKER-USER chain from that image (the harness's disable_internet)."""
    import shutil
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    if not os.path.exists(_DOCKER_SOCKET):
        return False, (f"{_DOCKER_SOCKET} does not exist; the harness starts each task container "
                       "on the host daemon through it.")
    found, labels = _labels(image())
    if not found:
        return False, f"the SWE-Lancer harness image {image()!r} is not built. " + _build_hint()
    if labels.get("org.gbench.frontier_evals") != HARNESS_COMMIT:
        return False, (f"the image {image()!r} carries {HARNESS_REPO} "
                       f"{labels.get('org.gbench.frontier_evals')!r}, not {HARNESS_COMMIT}; "
                       "rebuild it. " + _build_hint())
    if labels.get("org.gbench.swelancer_tasks_sha256") != TASKS_SHA256:
        return False, (f"the image {image()!r} carries a task table with sha256 "
                       f"{labels.get('org.gbench.swelancer_tasks_sha256')!r}, not {TASKS_SHA256}; "
                       "rebuild it. " + _build_hint())
    try:
        r = subprocess.run(["docker", "run", "--rm", "--network", "host", "--cap-add",
                            "NET_ADMIN", "--cap-add", "NET_RAW", image(), "--probe"],
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, f"the iptables probe could not run in {image()!r}: {e}"
    if r.returncode != 0:
        return False, (
            "SWE-Lancer disables each task container's internet (disable_internet=True, the "
            "only setting upstream considers valid) with iptables rules in the Docker host's "
            "DOCKER-USER chain, and the harness container (host network, NET_ADMIN) cannot see "
            f"that chain: {(r.stderr or r.stdout).strip()[-300:]}. It needs a Linux Docker host "
            "whose daemon manages iptables (the default) and permission to start a container "
            "with --network host --cap-add NET_ADMIN.")
    return True, ""


def _manifest_cache() -> str:
    return os.path.join(gbench_cache_dir(), "swe_lancer", "issues_manifest.json")


_TASKS_SCRIPT = (
    "import csv, hashlib, json, sys\n"
    f"p = '{_SWELANCER_DIR}/all_swelancer_tasks.csv'\n"
    "sha = hashlib.sha256(open(p, 'rb').read()).hexdigest()\n"
    "csv.field_size_limit(sys.maxsize)\n"
    "rows = [{'question_id': r['question_id'], 'variant': r['variant'], 'set': r['set'],\n"
    "         'price': float(r['price'])} for r in csv.DictReader(open(p, encoding='utf-8'))]\n"
    "print(json.dumps({'sha256': sha, 'tasks': rows}))\n"
)


def _task_table(data: Any) -> Optional[List[Dict[str, Any]]]:
    """The rows of a task table record ({"sha256": TASKS_SHA256, "tasks": [row, ...]}, each
    row a dict), else None: a record of another shape (the old SWELancer-Benchmark runner kept
    a list of issues in this same cache file) or of another table."""
    if not isinstance(data, dict) or data.get("sha256") != TASKS_SHA256:
        return None
    rows = data.get("tasks")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return None
    return rows


def load_tasks() -> List[Dict[str, Any]]:
    """The task table at HARNESS_COMMIT (question_id, variant, set, price), read from the
    harness image once and cached; its sha256 must be TASKS_SHA256. A cache file that does not
    hold that table (another shape, another sha256) is read again from the image and
    replaced."""
    cache = _manifest_cache()
    try:
        with open(cache, encoding="utf-8") as f:
            rows = _task_table(json.load(f))
        if rows is not None:
            return rows
    except (OSError, ValueError):
        pass
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", _PYTHON,
                        image(), "-c", _TASKS_SCRIPT], capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"swe_lancer: could not read the task table from {image()!r}: "
                           f"{(r.stderr or r.stdout)[-300:]}")
    lines = (r.stdout or "").strip().splitlines()
    try:
        data = json.loads(lines[-1]) if lines else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise RuntimeError(f"swe_lancer: the task table script in {image()!r} printed no "
                           f"table: {(r.stdout or r.stderr or '')[-300:]!r}")
    if data.get("sha256") != TASKS_SHA256:
        raise RuntimeError(f"swe_lancer: the task table in {image()!r} has sha256 "
                           f"{data.get('sha256')}, not {TASKS_SHA256}")
    if _task_table(data) is None:
        raise RuntimeError(f"swe_lancer: the task table in {image()!r} has no task rows")
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass
    return data["tasks"]


def resolve_variants(variant: Optional[str] = None,
                     eval_categories: Optional[str] = None) -> List[str]:
    """The variants a run evaluates, in VARIANTS order: `variant` alone when given, else the
    names among VARIANTS in `eval_categories`, else both."""
    if variant:
        names = [variant]
    elif eval_categories:
        names = [c.strip() for c in str(eval_categories).split(",") if c.strip()]
        names = [n for n in names if n in VARIANTS] or list(VARIANTS)
    else:
        return list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"swe_lancer: {unknown} are not SWE-Lancer task types; use "
                         f"{', '.join(VARIANTS)}")
    return [v for v in VARIANTS if v in names]


def select_tasks(tasks: List[Dict[str, Any]], variant: str, limit: Optional[int]
                 ) -> Tuple[List[str], Optional[Tuple[int, int]]]:
    """The question ids of `variant` a run evaluates: --shard (round-robin over the table's
    order), then a sample of `limit` of them (this task type's share of --eval-limit:
    variant_shares), as the other agentic suites select. `limit` 0 selects none."""
    ids = [t["question_id"] for t in tasks if t["variant"] == variant and t["set"] == SPLIT]
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    if shard:
        ids = shard_select(ids, shard)
    if limit is not None and limit == 0:
        return [], shard
    if limit and limit > 0:
        ids = stratified_sample(ids, limit, None, seed=f"swe_lancer:{variant}")
    return list(ids), shard


def variant_shares(tasks: List[Dict[str, Any]], variants: Sequence[str], limit: Optional[int]
                   ) -> Dict[str, Optional[int]]:
    """{task type: tasks it runs}: `--eval-limit N` is N tasks for the whole suite, dealt over
    the task types run (sampling.allocate_limit, seed "swe_lancer", each capped at the tasks of
    its type in this shard), never N per task type; a type whose share is 0 is not run. None for
    every type without a limit (all of its tasks)."""
    available = {v: len(select_tasks(tasks, v, None)[0]) for v in variants}
    return allocate_limit(limit, list(variants), available=available, seed="swe_lancer")


def harness_command(name: str, workdir: str, variant: str, taskset: Optional[Sequence[str]],
                    model_name: str, relay_url: str, concurrency: int,
                    report_path: str) -> List[str]:
    """`docker run` of one upstream run (one task type) in the harness image."""
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "host",
           "--cap-add", "NET_ADMIN", "--cap-add", "NET_RAW",
           "-v", f"{_DOCKER_SOCKET}:{_DOCKER_SOCKET}", "-v", f"{workdir}:{workdir}:rw",
           "-w", _SWELANCER_DIR,
           "-e", f"OPENAI_BASE_URL={relay_url}", "-e", "OPENAI_API_KEY=EMPTY",
           "-e", f"GBENCH_WORKDIR={workdir}"]
    for key, value in SAMPLE_ENV.items():
        cmd += ["-e", f"{key}={value}"]
    if hasattr(os, "getuid"):
        cmd += ["-e", f"GBENCH_UID={os.getuid()}", "-e", f"GBENCH_GID={os.getgid()}"]
    args = [f"swelancer.split={SPLIT}", f"swelancer.task_type={variant}",
            f"swelancer.solver={SOLVER}", f"swelancer.solver.model=openai/{model_name}",
            "swelancer.solver.computer_runtime="
            "nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime",
            "swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig",
            "swelancer.solver.computer_runtime.env.pull_from_registry=True",
            f"swelancer.docker_image_prefix={TASK_IMAGE_PREFIX}",
            f"swelancer.docker_image_tag={TASK_IMAGE_TAG}",
            f"swelancer.runs_dir={os.path.join(workdir, 'runs')}",
            f"swelancer.report_path={report_path}"]
    if variant == "swe_manager":
        args.append("swelancer.use_single_image=True")
    if taskset is not None:
        args.append(f"swelancer.taskset={json.dumps(list(taskset))}")
    args.append(f"runner.concurrency={max(1, int(concurrency or 1))}")
    args += [f"runner.{k}={v}" for k, v in RUNNER_ARGS.items()]
    return cmd + [image(), _PYTHON, "-c", _ENTRY, *args]


def _run_variant(variant: str, ids: List[str], full: bool, model_name: str, relay_url: str,
                 concurrency: int, workdir: str) -> Dict[str, Any]:
    """One upstream run of one task type, and its result."""
    start = time.time()
    report_path = os.path.join(workdir, f"{variant}_report.json")
    name = f"gbench-swelancer-{variant}-{uuid.uuid4().hex[:10]}"
    cmd = harness_command(name, workdir, variant, None if full else ids, model_name, relay_url,
                          concurrency, report_path)
    log_path = os.path.join(workdir, f"harness_{variant}.log")
    rc = None
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        report = None
    res: Dict[str, Any] = {"variant": variant, "harness_log": log_path, "harness_rc": rc,
                           "duration_s": round(time.time() - start, 2)}
    if not report:
        res.update(status="error", total_questions=len(ids), correct_answers=0, accuracy=None,
                   error=f"the SWE-Lancer harness wrote no report (exit {rc}); see {log_path}",
                   sample_traces=[])
        return res
    rows = report.get("tasks") or []
    traces = []
    for i, row in enumerate(rows):
        err = row.get("system_error")
        traces.append({
            "sample_idx": i, "source_sample_idx": i, "attempt_idx": row.get("attempt_id", 0),
            "category": variant,
            "extra_payload": {"question_id": row.get("question_id"), "variant": variant,
                              "price": row.get("price")},
            "is_correct": bool(row.get("correct")), "earned_usd": row.get("earned") or 0.0,
            "available_usd": row.get("available") or row.get("price") or 0.0,
            "system_error": err, "run_dir": row.get("run_dir"),
            "input_tokens": row.get("input_tokens", 0),
            "output_tokens": row.get("output_tokens", 0),
            "reasoning_tokens": row.get("reasoning_tokens", 0),
            "status": "SYSTEM_ERROR" if err else "OK"})
    for t in traces:
        if t["system_error"]:
            # A rollout that ended in a system error got no verdict: a harness error, left out
            # of pass@1's numerator and denominator and of the dollars available.
            swebench_common.mark_harness_error(t, "system_error", t["system_error"])
    scored = [t for t in traces if not t.get("scoring_excluded")]
    total = len(traces)
    correct = sum(1 for t in scored if t["is_correct"])
    errors = total - len(scored)
    earned = round(sum(float(t["earned_usd"] or 0.0) for t in scored), 2)
    available = round(sum(float(t["available_usd"] or 0.0) for t in scored), 2)
    missing = sorted(set(ids) - {t["extra_payload"]["question_id"] for t in traces})
    res.update({
        "total_questions": total, "scored_tasks": len(scored), "effective_n": len(scored),
        "correct_answers": correct,
        "failed_requests": errors + len(missing),
        "accuracy": round(correct / len(scored) * 100.0, 2) if scored else None,
        "earned_usd": earned, "available_usd": available,
        "system_errors": errors, "summary": report.get("summary"),
        "sample_traces": traces, "status": "success",
        "runs_dir": os.path.join(workdir, "runs"),
    })
    if errors or missing:
        res["harness_errors"] = {k: v for k, v in (("system_error", errors),
                                                   ("missing", len(missing))) if v}
    if missing:
        res["missing_tasks"] = missing
    if not scored:
        res["status"] = "error"
        res["error"] = (f"no {VARIANT_LABELS[variant]} task produced a verdict ({errors} system "
                        f"error(s) of {total}"
                        + (f", {len(missing)} task(s) without a result" if missing else "")
                        + "); this is an infrastructure failure, not a 0% pass rate")
        res["accuracy_withheld_reason"] = res["error"]
    elif errors or missing:
        res["status"] = "completed_with_errors"
        res["partial_evaluation"] = {"total_tasks": total + len(missing),
                                     "system_error_tasks": errors,
                                     "tasks_without_result": len(missing)}
    return res


def _combine(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ran = list(results)
    ok = [v for v in ran if results[v].get("status") != "error"]
    total = sum(results[v].get("total_questions") or 0 for v in ok)
    # pass@1's denominator: the tasks with a verdict (system errors left out)
    scored = sum(results[v].get("scored_tasks", results[v].get("total_questions")) or 0
                 for v in ok)
    correct = sum(results[v].get("correct_answers") or 0 for v in ok)
    earned = round(sum(results[v].get("earned_usd") or 0.0 for v in ok), 2)
    available = round(sum(results[v].get("available_usd") or 0.0 for v in ok), 2)
    headline: Dict[str, Any] = {}
    for v in ran:
        good = results[v].get("status") != "error"
        headline[f"{v}_pass_at_1"] = results[v].get("accuracy") if good else None
        headline[f"{v}_earned_usd"] = results[v].get("earned_usd") if good else None
    headline["diamond_pass_at_1"] = round(correct / scored * 100.0, 2) if scored else None
    headline["diamond_earned_usd"] = earned if ok else None
    traces: List[Dict[str, Any]] = []
    for v in ran:
        offset = len(traces)
        for tr in results[v].get("sample_traces") or []:
            tr = dict(tr)
            tr["sample_idx"] += offset
            tr["source_sample_idx"] += offset
            traces.append(tr)
    errored = [v for v in ran if results[v].get("status") == "error"]
    if len(errored) == len(ran):
        status = "error"
    elif errored or any(results[v].get("status") == "completed_with_errors" for v in ran):
        status = "completed_with_errors"
    else:
        status = "success"
    out: Dict[str, Any] = {
        "variants": ran, "total_questions": total, "scored_tasks": scored, "effective_n": scored,
        "correct_answers": correct,
        "accuracy": headline["diamond_pass_at_1"] if ok else None,
        "earned_usd": earned, "available_usd": available,
        "headline_metrics": headline,
        "headline_metrics_note": (
            "pass@1 (%) = tasks solved / tasks and dollars earned, per task type and over the "
            "Diamond tasks run (SWE-Lancer's eval: score 1 = solved; earned = price x score)"),
        "category_accuracy": {v: {"correct": results[v].get("correct_answers", 0),
                                  "total": results[v].get("scored_tasks",
                                                          results[v].get("total_questions", 0)),
                                  "accuracy": results[v].get("accuracy")}
                              for v in ok},
        "settings": {v: {k: val for k, val in results[v].items() if k != "sample_traces"}
                     for v in ran},
        "sample_traces": traces, "status": status,
    }
    if errored:
        out["settings_with_errors"] = errored
        if status == "error":
            out["error"] = "; ".join(f"{v}: {results[v].get('error')}" for v in ran)
    kinds: Dict[str, int] = {}
    for v in ran:
        for k, n in (results[v].get("harness_errors") or {}).items():
            kinds[k] = kinds.get(k, 0) + n
    if kinds:
        out["harness_errors"] = dict(sorted(kinds.items()))
    if out["accuracy"] is None:
        out["accuracy_withheld_reason"] = out.get("error") or "no task produced a verdict"
    return out


def run_swe_lancer(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """SWE-Lancer with SimpleAgentSolver on the official task set, both task types by default."""
    ok, reason = check_swe_lancer_prerequisites()
    if not ok:
        raise infra_required("swe_lancer", reason, DOCS_URL)
    # The model's Python blocks and the tests run in the task containers, on the network
    # alcatraz creates (a user-defined bridge network) before it disables their internet: the
    # cloud metadata server must be blocked there before the first one starts.
    swebench_common.require_task_network_blocked(
        "swe_lancer", [image()], docs_url=DOCS_URL, hint=_build_hint())
    start = time.time()
    variants = resolve_variants(kwargs.get("variant"), kwargs.get("eval_categories"))
    limit = kwargs.get("limit")
    tasks = load_tasks()
    temperature, temperature_source = resolve_official_sampling(
        "swe_lancer", OFFICIAL_SAMPLING, thinking=enable_thinking)
    budget = resolve_token_budget("swe_lancer", OFFICIAL_SAMPLING, thinking=enable_thinking,
                                  max_output_tokens=kwargs.get("max_output_tokens"))
    body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    body["temperature"] = temperature
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    from . import swe_agentic
    workdir = swe_agentic.agent_output_dir("swe_lancer")
    try:
        relay = EndpointRelay(base_url, "127.0.0.1", key=key if key and key != "EMPTY" else None,
                              body_fields=body, name="swe-lancer-relay").start()
    except OSError as e:
        raise infra_required("swe_lancer", f"gbench's relay to the served endpoint could not "
                             f"listen on 127.0.0.1 ({type(e).__name__}: {e}).", DOCS_URL) from e
    results: Dict[str, Dict[str, Any]] = {}
    shard = None
    shares = variant_shares(tasks, variants, limit)
    selection = {v: select_tasks(tasks, v, shares[v]) for v in variants}
    plan = eval_limit_plan("swe_lancer", limit,
                           sum(len(ids) for ids, _ in selection.values()),
                           {v: len(ids) for v, (ids, _) in selection.items()},
                           setting_noun="task type", attempts=1, unit="rollouts")
    try:
        for variant in variants:
            ids, shard = selection[variant]
            if limit and not ids:
                continue                      # its share of --eval-limit is 0: not run
            full = not limit and not shard and len(ids) == VARIANT_TASKS[variant]
            results[variant] = _run_variant(variant, ids, full, model_name, relay.url,
                                            concurrency, workdir)
    finally:
        relay.stop()
    result: Dict[str, Any] = {
        "benchmark_type": "eval", "eval_name": "swe_lancer", "model_name": model_name,
        "thinking": enable_thinking, "mode": "agentic", "scoring_mode": "execution",
        "temperature": temperature, "temperature_source": temperature_source,
        "official_sampling": {**OFFICIAL_SAMPLING.record(temperature, temperature_source),
                              **budget.record()},
        "max_output_tokens_per_turn": None, "max_output_tokens_note": budget.note,
        "dataset": (f"{HARNESS_REPO}@{HARNESS_COMMIT} project/swelancer/all_swelancer_tasks.csv "
                    f"(sha256 {TASKS_SHA256}; split {SPLIT})"),
        "agent": {"solver": SOLVER, "max_turns": SOLVER_MAX_TURNS,
                  "model": f"openai/{model_name}", "reasoning_effort": None},
        "harness": {"tree": f"{HARNESS_REPO}@{HARNESS_COMMIT} project/swelancer",
                    "image": image(), "entry": "swelancer/run_swelancer.py main "
                                               "(gbench/docker/swe_lancer_eval.py)",
                    "task_images": {"ic_swe": f"{TASK_IMAGE_PREFIX}_<question_id>:"
                                              f"{TASK_IMAGE_TAG}",
                                    "swe_manager": f"{TASK_IMAGE_PREFIX}_monolith:"
                                                   f"{TASK_IMAGE_TAG}"},
                    "n_test_runs": 3, "disable_internet": True, "runner": dict(RUNNER_ARGS),
                    "environment": dict(SAMPLE_ENV)},
        "relay": relay.record(), "agent_output_dir": workdir,
        "reference_note": REFERENCE_NOTE,
    }
    result.update(_combine(results))
    result["eval_limit_plan"] = plan
    if shard:
        result["shard"] = f"{shard[0]}/{shard[1]}"
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    if attempts > 1:
        result["attempts"] = {"attempts_requested": attempts, "attempts_per_sample": 1,
                              "not_applied": "SWE-Lancer's run command runs each task once "
                                             "(n_tries=1); pass@1 is its metric"}
    noncanon: List[str] = []
    missing_variants = [v for v in VARIANTS if v not in variants]
    if missing_variants:
        noncanon.append(f"task types not run: {', '.join(missing_variants)}")
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit} for the suite: "
                        + ", ".join(f"{len(selection[v][0])} {VARIANT_LABELS[v]}"
                                    for v in variants) + " task(s))")
    if shard:
        noncanon.append(f"shard {shard[0]}/{shard[1]} of the task set")
    noncanon.append(f"temperature={temperature} "
                    f"{swe_agentic._temperature_origin(temperature_source)}; "
                    "SimpleAgentSolver sends none")
    for v in results:                 # the task types that ran (a share of 0 runs none)
        res = results[v]
        if res.get("status") == "error":
            noncanon.append(f"{VARIANT_LABELS[v]}: {res.get('error')}")
        elif res.get("status") == "completed_with_errors":
            noncanon.append(f"partial: {res.get('system_errors', 0)} {VARIANT_LABELS[v]} "
                            "task(s) ended in a system error (infrastructure), left out of the "
                            "score")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    result["duration_s"] = round(time.time() - start, 2)
    logger.info("swe_lancer: %s", {v: (results[v].get("accuracy"), results[v].get("earned_usd"))
                                   for v in results})
    return result
