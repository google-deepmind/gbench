#!/usr/bin/env python3
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
#
# gbench launcher for SkillsBench, run INSIDE the gbench-skillsbench image. It drives the BenchFlow
# 0.6.3 CLI directly (the repo's own open-model runner script targets an OLDER benchflow: it uses
# `--backend`/opencode, but the locked 0.6.3 CLI uses `--sandbox` and ships deepagents/codex/etc.,
# configuring the model via BENCHFLOW_PROVIDER_BASE_URL/API_KEY). For each condition x task it runs
#   uv run bench eval run --tasks-dir <task> --agent <agent> --sandbox docker --jobs-dir <jobs>
#       [--model <model> --agent-env BENCHFLOW_PROVIDER_BASE_URL=.. --agent-env BENCHFLOW_PROVIDER_API_KEY=..]
#       [--skills-dir <task>/environment/skills]      # with-skills condition
# and parses each rollout's result.json `rewards.reward` (the DETERMINISTIC per-task verifier; no
# LLM judge). Scoring is upstream's own reward - gbench adds nothing.
#
# docker-out-of-docker: BenchFlow bind-mounts its working dirs into sibling task containers, so the
# workdir MUST be an identity mount (same path on host + orchestrator) and TMPDIR must point there,
# else the sibling sandbox gets empty mounts and no reward is collected (the entrypoint/harness set
# this up). Env contract (set by the entrypoint / docker run -e):
#   SKILLSBENCH_DIR         repo root (default /app/skillsbench)
#   SKILLSBENCH_WORKDIR     identity-mounted host path for jobs + summary + TMPDIR (default /out)
#   SKILLSBENCH_AGENT       agent (default deepagents; 'oracle' for a model-free pipeline smoke)
#   GBENCH_MODEL_BASE_URL   /v1 reachable FROM task containers (model path only)
#   GBENCH_MODEL_NAME       served model name (model path only)
#   GBENCH_MODEL_API_KEY    the placeholder key gbench passes (never a real key: an endpoint key
#                           stays in the gbench process, whose relay adds it)
#   SKILLSBENCH_CONDITIONS  comma list from {with-skills,without-skills} (default without-skills)
#   SKILLSBENCH_TASKS       comma list to limit tasks (default all-minus-excludes)
#   SKILLSBENCH_TASK_LIMIT  draw N tasks when no explicit subset: a seeded draw balanced across the
#                           tasks' categories (task.md metadata.category; LIMIT_DRAW), not the
#                           first N of the sorted list; a value that is not an integer stops the
#                           launcher
#   SKILLSBENCH_CONDITION_LIMITS  --eval-limit dealt over the conditions, JSON {condition: tasks}
#                           (gbench sets it when more than one condition runs): each condition's
#                           share is the same seeded draw (nested task sets), a share of 0 runs none
#   SKILLSBENCH_CONCURRENCY per-task bench concurrency (default 1)
#   GBENCH_SHARD            "I/N": run only shard I of N of the sorted task list (round-robin
#                           full_sorted[I-1::N]); applied BEFORE SKILLSBENCH_TASK_LIMIT, and within
#                           the SKILLSBENCH_TASKS subset when one is given. Score set == run set.
#                           A malformed or out-of-range value stops the launcher (exit 1): it is
#                           never read as "no shard" (a full run).
#   SKILLSBENCH_AGENT_IDLE_TIMEOUT_S  passed as `bench eval run --agent-idle-timeout` (BenchFlow's
#                           ACP idle watchdog, 600 s by default: the deepagents shim emits no ACP
#                           event during one model turn, so a turn longer than that was killed and
#                           the trial re-run, 2026-10-10 debug-trl-grpo); gbench sets it to its
#                           request timeout. Unset: BenchFlow's default. The task's own agent
#                           timeout still bounds the trial.
#   GBENCH_MODEL_REQUEST_TIMEOUT_S    passed to the agent with --agent-env: the deepagents shim's
#                           ChatOpenAI request timeout (skillsbench_benchflow_transport.patch)
# The summary records n_tasks_full (the list the shard was taken from), n_tasks_selected and
# selection (the limit and LIMIT_DRAW for a limited run; gbench checks it), and
# sandbox_build_retries ({condition: {task: n}}): the times BenchFlow re-ran a task environment's
# `docker compose build` after a transient failure (its "Retrying Docker build for" warning; the
# image's skillsbench_benchflow_build_retry.patch adds the containerd layer-commit failure of
# 2026-10-13 to the failures it retries). A retried build happens before the agent starts.

import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys

SKILLSBENCH_DIR = os.environ.get("SKILLSBENCH_DIR", "/app/skillsbench")
WORKDIR = os.environ.get("SKILLSBENCH_WORKDIR", "/out")
_EXCLUDED = {"mhc-layer-impl"}


def _shard():
    """Parse GBENCH_SHARD='I/N' -> (i, n), or None when it is unset or empty (matches gbench
    sampling.parse_shard). A malformed or out-of-range value exits the launcher: running the
    whole task set under a shard the operator asked for would misreport what was measured."""
    spec = os.environ.get("GBENCH_SHARD", "").strip()
    if not spec:
        return None
    try:
        i, n = (int(x) for x in spec.split("/", 1))
    except ValueError:
        i = n = 0
    if not (n >= 1 and 1 <= i <= n):
        sys.exit(f"GBENCH_SHARD must be I/N with 1 <= I <= N (e.g. 1/8), got {spec!r}")
    return (i, n)


def _apply_shard(ids):
    """Round-robin shard I of N over the DETERMINISTIC sorted id list: full_sorted[i-1::n].

    No-op when no shard is set, so non-sharded runs are byte-identical to before.
    """
    sh = _shard()
    if not sh:
        return ids
    i, n = sh
    return sorted(ids)[i - 1::n]


def _full_task_list():
    """The list a shard is taken from: SKILLSBENCH_TASKS when given, else every task (sorted)."""
    explicit = [t.strip() for t in os.environ.get("SKILLSBENCH_TASKS", "").split(",") if t.strip()]
    if explicit:
        return explicit
    tasks_dir = os.path.join(SKILLSBENCH_DIR, "tasks")
    return sorted(d for d in os.listdir(tasks_dir)
                  if d not in _EXCLUDED and os.path.isfile(os.path.join(tasks_dir, d, "task.md")))


#: The seed of the --eval-limit draw (the suite name, as gbench's sampling.stratified_sample uses).
LIMIT_SEED = "skillsbench"
#: How a limited run selects its tasks; the summary's selection.limit_draw (gbench checks it).
LIMIT_DRAW = f"stratified_indices(key=category, seed={LIMIT_SEED})"


def _stable_key(key):
    if key is None:
        return "<none>"
    try:
        hash(key)
    except TypeError:
        return repr(key)
    return key


def stratified_indices(keys, limit, seed=None):
    """gbench's sampling.stratified_indices, copied (this launcher runs in its image without
    gbench; a test holds the copy equal): the ascending indices of `limit` items, round-robin
    across the categories `keys` gives (seeded order of categories, seeded order within each), so
    a limited run samples every category before any twice; every index when `limit` is None,
    not positive or at least len(keys)."""
    total = len(keys)
    if limit is None or limit <= 0 or limit >= total:
        return list(range(total))
    digest = hashlib.sha256((seed or "gbench").encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(_stable_key(key), []).append(index)
    for indices in groups.values():
        rng.shuffle(indices)
    order = list(groups)
    rng.shuffle(order)
    chosen = []
    while len(chosen) < limit:
        progressed = False
        for key in order:
            bucket = groups[key]
            if not bucket:
                continue
            chosen.append(bucket.pop())
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return sorted(chosen)


def _task_category(task):
    """The task's `category` from its task.md front matter (metadata.category), or None."""
    try:
        with open(os.path.join(SKILLSBENCH_DIR, "tasks", task, "task.md"), encoding="utf-8") as f:
            head = f.read(4000)
    except OSError:
        return None
    front = head.split("---", 2)[1] if head.startswith("---") and head.count("---") >= 2 else head
    m = re.search(r"^\s*category:\s*(\S.*?)\s*$", front, re.M)
    return m.group(1).strip("'\"") if m else None


def _limit():
    """SKILLSBENCH_TASK_LIMIT as a positive int, or None when it is unset, empty or not positive;
    a value that is not an integer stops the launcher (never read as "no limit", a full run)."""
    raw = (os.environ.get("SKILLSBENCH_TASK_LIMIT") or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        sys.exit(f"SKILLSBENCH_TASK_LIMIT must be an integer, got {raw!r}")
    return n if n > 0 else None


def _condition_limits():
    """SKILLSBENCH_CONDITION_LIMITS ({condition: tasks}, JSON; the gbench runner's deal of
    --eval-limit over the conditions) as a dict of non-negative ints, or None when it is unset or
    empty; anything else stops the launcher (never read as "no limit", a full run)."""
    raw = (os.environ.get("SKILLSBENCH_CONDITION_LIMITS") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        sys.exit(f"SKILLSBENCH_CONDITION_LIMITS must be a JSON object, got {raw!r}")
    if not isinstance(data, dict) or not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in data.values()):
        sys.exit(f"SKILLSBENCH_CONDITION_LIMITS must map conditions to non-negative integers, "
                 f"got {raw!r}")
    return {str(k): int(v) for k, v in data.items()}


def condition_tasks(allt, limits):
    """{condition: its tasks} for SKILLSBENCH_CONDITION_LIMITS `limits`: each condition's share
    drawn from the shard's tasks `allt` with the same seeded draw (stratified_indices over the
    categories, LIMIT_SEED), so a smaller share is a subset of a larger one (the conditions run
    nested task sets and the lift compares them on the tasks both ran); a share of 0 runs none."""
    cats = [_task_category(t) for t in allt]
    return {c: ([allt[k] for k in stratified_indices(cats, n, LIMIT_SEED)] if n else [])
            for c, n in limits.items()}


def _task_list(full=None):
    explicit = [t.strip() for t in os.environ.get("SKILLSBENCH_TASKS", "").split(",") if t.strip()]
    full = _full_task_list() if full is None else full
    if explicit:
        # Shard WITHIN the explicit subset (the "full task list" is that subset here).
        return _apply_shard(full)
    # Shard FIRST (over the full sorted set), then the limit draws within the shard, balanced
    # across the tasks' categories (sorted order kept), not the head of the sorted list.
    allt = _apply_shard(full)
    lim = _limit()
    if lim:
        allt = [allt[k] for k in stratified_indices([_task_category(t) for t in allt], lim,
                                                    LIMIT_SEED)]
    return allt


def _reward_from_jobs(jobs_dir):
    """Return the reward float from the newest result.json under jobs_dir (or None)."""
    results = sorted(glob.glob(os.path.join(jobs_dir, "**", "result.json"), recursive=True))
    if not results:
        return None
    try:
        with open(results[-1], encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    rewards = d.get("rewards")
    if isinstance(rewards, dict) and isinstance(rewards.get("reward"), (int, float)):
        return float(rewards["reward"])
    if isinstance(rewards, (int, float)):
        return float(rewards)
    if isinstance(d.get("reward"), (int, float)):
        return float(d["reward"])
    return None


#: BenchFlow's warning before it re-runs a task environment's build (sandbox/docker.py,
#: DockerSandbox._run_docker_compose_build), logged to the `bench` process's stderr.
_BUILD_RETRY = re.compile(
    r"Retrying Docker build for \S+ after transient failure \(attempt \d+/\d+")


def build_retries(text):
    """How many times the `bench` output `text` says BenchFlow re-ran an environment build."""
    return len(_BUILD_RETRY.findall(text or ""))


def _run_one(task, condition, agent, jobs_dir, retries=None):
    """The task's reward (None when the rollout wrote none); with a dict `retries`, the number of
    environment builds BenchFlow re-ran is stored there under the task when it is not 0."""
    task_dir = os.path.join(SKILLSBENCH_DIR, "tasks", task)
    cmd = ["uv", "run", "bench", "eval", "run",
           "--tasks-dir", task_dir, "--agent", agent, "--sandbox", "docker",
           "--jobs-dir", jobs_dir,
           "--concurrency", os.environ.get("SKILLSBENCH_CONCURRENCY", "1")]
    if agent != "oracle":
        model = os.environ.get("GBENCH_MODEL_NAME", "")
        if model:
            cmd += ["--model", model]
        base_url = os.environ.get("GBENCH_MODEL_BASE_URL", "")
        api_key = os.environ.get("GBENCH_MODEL_API_KEY", "dummy")
        if base_url:
            cmd += ["--agent-env", f"BENCHFLOW_PROVIDER_BASE_URL={base_url}",
                    "--agent-env", f"BENCHFLOW_PROVIDER_API_KEY={api_key}",
                    "--agent-env", f"OPENAI_BASE_URL={base_url}",
                    "--agent-env", f"OPENAI_API_KEY={api_key}"]
        request_timeout = os.environ.get("GBENCH_MODEL_REQUEST_TIMEOUT_S", "").strip()
        if request_timeout:
            cmd += ["--agent-env", f"GBENCH_MODEL_REQUEST_TIMEOUT_S={request_timeout}"]
        idle = os.environ.get("SKILLSBENCH_AGENT_IDLE_TIMEOUT_S", "").strip()
        if idle:
            cmd += ["--agent-idle-timeout", idle]
    if condition == "with-skills":
        skills_dir = os.path.join(task_dir, "environment", "skills")
        if os.path.isdir(skills_dir):
            # BenchFlow 0.6.3 defaults --skill-mode to no-skill and RAISES when --skills-dir is
            # set without with-skill, so the with-skills arm (the lift metric) needs both flags.
            cmd += ["--skills-dir", skills_dir, "--skill-mode", "with-skill"]
    env = dict(os.environ)
    env["TMPDIR"] = WORKDIR
    proc = subprocess.run(cmd, cwd=SKILLSBENCH_DIR, env=env, capture_output=True, text=True)
    n_retries = build_retries((proc.stderr or "") + "\n" + (proc.stdout or ""))
    if n_retries and retries is not None:
        retries[task] = n_retries
    reward = _reward_from_jobs(jobs_dir)
    if reward is None:
        sys.stderr.write(f"[skillsbench] {condition}/{task}: no reward "
                         f"(rc={proc.returncode}) tail: {(proc.stderr or proc.stdout or '')[-300:]}\n")
    return reward


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    agent = os.environ.get("SKILLSBENCH_AGENT", "deepagents")
    conditions = [c.strip() for c in os.environ.get("SKILLSBENCH_CONDITIONS", "without-skills").split(",") if c.strip()]
    full = _full_task_list()
    tasks = _task_list(full)

    explicit = [t.strip() for t in os.environ.get("SKILLSBENCH_TASKS", "").split(",") if t.strip()]
    limit = None if explicit else _limit()
    cond_limits = None if explicit else _condition_limits()
    per_condition = None
    if cond_limits is not None:
        # --eval-limit dealt over the conditions: each runs its share (nested draws)
        per_condition = condition_tasks(_apply_shard(full), cond_limits)
        tasks = sorted({t for ts in per_condition.values() for t in ts})
    out = {"conditions": {}, "agent": agent, "n_tasks_selected": len(tasks),
           # what the launcher passed to `bench eval run` (None: BenchFlow's default)
           "agent_idle_timeout_s": (os.environ.get("SKILLSBENCH_AGENT_IDLE_TIMEOUT_S", "").strip()
                                    or None) if agent != "oracle" else None,
           "model_request_timeout_s": (os.environ.get("GBENCH_MODEL_REQUEST_TIMEOUT_S",
                                                      "").strip() or None)
           if agent != "oracle" else None,
           "n_tasks_full": len(full), "tasks_selected": list(tasks),
           "selection": {"limit": limit, "limit_draw": (LIMIT_DRAW if limit or cond_limits
                                                         is not None else None),
                         "condition_limits": cond_limits},
           # {condition: {task: n}}: environment builds BenchFlow re-ran (none: {})
           "sandbox_build_retries": {}}
    for cond in conditions:
        per_task = {}
        retried = {}
        for task in (per_condition.get(cond, []) if per_condition is not None else tasks):
            jobs_dir = os.path.join(WORKDIR, "jobs", cond, task)
            os.makedirs(jobs_dir, exist_ok=True)
            per_task[task] = _run_one(task, cond, agent, jobs_dir, retried)
        if retried:
            out["sandbox_build_retries"][cond] = retried
        vals = [v for v in per_task.values() if isinstance(v, (int, float))]
        out["conditions"][cond] = {
            "mean_reward": (sum(vals) / len(vals)) if vals else None,
            "n_scored": len(vals),
            "n_tasks": len(per_task),
            "per_task": per_task,
        }
    ws = out["conditions"].get("with-skills", {}).get("mean_reward")
    ns = out["conditions"].get("without-skills", {}).get("mean_reward")
    if per_condition is not None:
        # nested task sets: the lift over the tasks both conditions scored
        w = out["conditions"].get("with-skills", {}).get("per_task") or {}
        n = out["conditions"].get("without-skills", {}).get("per_task") or {}
        common = [t for t in w if isinstance(w[t], (int, float))
                  and isinstance(n.get(t), (int, float))]
        ws = sum(w[t] for t in common) / len(common) if common else None
        ns = sum(n[t] for t in common) / len(common) if common else None
        out["with_skills_lift_tasks"] = common
    if isinstance(ws, (int, float)) and isinstance(ns, (int, float)):
        out["with_skills_lift"] = ws - ns

    with open(os.path.join(WORKDIR, "skillsbench_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
