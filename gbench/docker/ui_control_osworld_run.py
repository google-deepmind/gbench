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
# gbench launcher for OSWorld, run INSIDE the gbench-ui-control-osworld image. It drives the pinned
# OSWorld harness (run.py's test loop) IN-PROCESS so the model-wiring monkeypatch takes effect, then
# parses the per-example result.txt tree -> a gbench summary. It NEVER fabricates a score.
#
# Model wiring: OSWorld's mm_agents.PromptAgent.call_llm routes by model-name PREFIX; the "gpt"
# branch honours OPENAI_BASE_URL. But vLLM strictly rejects an unknown model name, so we (1) pass
# --model gpt-4o (a routing alias -> the OpenAI-compat path), (2) set OPENAI_BASE_URL to the served
# /v1 endpoint, and (3) monkeypatch mm_agents.agent.requests.post to rewrite the outgoing payload
# `model` field to the real served name. Scoring is upstream's DETERMINISTIC per-task evaluators
# (result.txt in [0,1]) - no judge.
#
# docker-out-of-docker IDENTITY-MOUNT contract (the skillsbench/wildclaw pitfall): OSWorld's docker
# provider bind-mounts the Ubuntu qcow2 into the sibling VM container via os.path.abspath(path_to_vm),
# and VMS_DIR is "./docker_vm_data" relative to CWD - so the launcher chdir's to the identity-mounted
# WORKDIR (which holds docker_vm_data/Ubuntu.qcow2; this run's results/ go to its OSWORLD_RUN_DIR
# under it), and that path must be identical on the host daemon (the gbench runner arranges the
# identity mount).
#
# Env contract (set by the entrypoint / docker run -e):
#   OSWORLD_DIR            harness root (default /app/OSWorld)
#   OSWORLD_WORKDIR        identity-mounted workdir holding docker_vm_data/ (qcow2) (default /out)
#   OSWORLD_RUN_DIR        this run's fresh directory (results/, test_subset.json, the summary);
#                          default OSWORLD_WORKDIR. Never shared across runs: OSWorld resumes from
#                          the result.txt files its result_dir holds
#   GBENCH_MODEL_NAME      served model name (rewritten into the outgoing payload)
#   GBENCH_MODEL_BASE_URL  model /v1 endpoint (OPENAI_BASE_URL; --network host -> 127.0.0.1)
#   GBENCH_MODEL_API_KEY   api key (dummy for vLLM)
#   OSWORLD_DOMAIN         single domain, or "all" (default all)
#   GBENCH_SHARD           "I/N" round-robin shard over the DETERMINISTIC sorted (domain, example_id)
#                          full task list (applied BEFORE OSWORLD_LIMIT); unset/empty => no shard;
#                          a malformed or out-of-range value stops the launcher (exit status naming
#                          it), never a full run
#   OSWORLD_LIMIT          cap total examples (smoke; applied WITHIN the shard): a seeded draw
#                          balanced across the domains (stratified_indices, seed
#                          "ui_control_osworld"), not the head of the sorted list
#   OSWORLD_MAX_STEPS      per-task step budget (default 15)
#   OSWORLD_TEMPERATURE    run.py's --temperature, the model's sampling temperature (gbench sets
#                          it: 1.0 with --thinking, 0.0 without, unless the operator set one);
#                          unset/empty => run.py's default
#   OSWORLD_OBS_TYPE       screenshot | a11y_tree | screenshot_a11y_tree | som (default screenshot)
#   OSWORLD_TEST_META      manifest filename under evaluation_examples/ (default test_all.json)
#   OSWORLD_HOST_OWNER     uid:gid the run's directory is handed to at the end (the gbench runner's
#                          user), so the results this root process wrote belong to that user
#
# Endpoint failures: OSWorld's PromptAgent answers a failed model request with an empty reply and
# the task goes on to be evaluated (it writes result.txt), so a failure of the model endpoint would
# score as the model's 0. The request wrapper records every model request that raised or got a
# reply other than HTTP 200 in the running example's result directory (ENDPOINT_FAILURES, one JSON
# line each); one the model's request cannot cause (no reply, HTTP 5xx or 429, a refused key or
# unknown route: 401, 403, 404) is `harness: true`, and the summary counts those per task
# (`endpoint_failures`); gbench leaves such a task out of the score. A request rejection (HTTP
# 400, 413, 422, as for a context overflow) is recorded with `harness: false`: upstream's own
# handling decides, and it stays the model's outcome.

import hashlib
import json
import os
import random
import sys

OSWORLD_DIR = os.environ.get("OSWORLD_DIR", "/app/OSWorld")
WORKDIR = os.environ.get("OSWORLD_WORKDIR", "/out")


def _run_dir():
    """This run's own directory (results/, the subset manifest, the summary): OSWORLD_RUN_DIR,
    which the gbench runner sets to a fresh directory under the identity-mounted WORKDIR; WORKDIR
    itself when unset. OSWorld's run.py resumes from whatever result.txt files its result_dir
    already holds (get_unfinished), and the routing alias makes every model's results land in the
    same place, so a result_dir shared across runs would hand one run another run's results."""
    return os.environ.get("OSWORLD_RUN_DIR", "").strip() or WORKDIR
_ROUTING_ALIAS = "gpt-4o"   # takes PromptAgent.call_llm's OPENAI_BASE_URL path
#: Written in each example's result directory: one JSON line per model request that raised or got
#: a reply other than HTTP 200 ({t, status, error, harness}).
ENDPOINT_FAILURES = "gbench_endpoint_failures.jsonl"
#: Reply statuses that are a failure of the endpoint, never something the model's request causes.
ENDPOINT_FAILURE_STATUSES = frozenset({401, 403, 404, 429})
#: The result directory of the example the harness is running (set around run_single_example).
_CURRENT = {"dir": None}


def endpoint_failure(status=None, exc=None):
    """True when a model request that raised (`exc`) or got HTTP `status` failed on the
    endpoint's side (no reply, 5xx, 429, 401, 403, 404); False for a request rejection (400, 413,
    422: the request's own content, e.g. a context overflow)."""
    if exc is not None:
        return True
    return status in ENDPOINT_FAILURE_STATUSES or 500 <= int(status) <= 599


def record_endpoint_failure(status=None, exc=None, now=None):
    """Append one failed model request to the running example's ENDPOINT_FAILURES (nothing when no
    example runs)."""
    import time
    directory = _CURRENT["dir"]
    if not directory:
        return
    line = {"t": time.time() if now is None else now, "status": status,
            "error": type(exc).__name__ if exc is not None else None,
            "harness": endpoint_failure(status, exc)}
    try:
        with open(os.path.join(directory, ENDPOINT_FAILURES), "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass


def _install_model_rewrite(served: str):
    """Rewrite the outgoing OpenAI-compat payload `model` (routing alias) -> the real served name,
    since vLLM rejects an unknown model id, and record every chat/completions request that raised
    or got a reply other than HTTP 200 (record_endpoint_failure). Scoped to chat/completions POSTs
    only; the reply goes back to the agent unchanged."""
    import mm_agents.agent as A
    orig_post = A.requests.post

    def _post(url, *args, **kwargs):
        j = kwargs.get("json")
        chat = "chat/completions" in str(url)
        if isinstance(j, dict) and j.get("model") == _ROUTING_ALIAS and chat:
            j = dict(j)
            j["model"] = served
            kwargs["json"] = j
        try:
            response = orig_post(url, *args, **kwargs)
        except Exception as exc:
            if chat:
                record_endpoint_failure(exc=exc)
            raise
        if chat and getattr(response, "status_code", 200) != 200:
            record_endpoint_failure(status=int(response.status_code))
        return response

    A.requests.post = _post


def _install_example_tracking():
    """Wrap lib_run_single.run_single_example (run.py calls it through the module) so the request
    wrapper knows the result directory of the example running."""
    import lib_run_single as L
    orig = L.run_single_example

    def run_single_example(agent, env, example, max_steps, instruction, args,
                           example_result_dir, scores, *rest, **kw):
        _CURRENT["dir"] = example_result_dir
        try:
            return orig(agent, env, example, max_steps, instruction, args, example_result_dir,
                        scores, *rest, **kw)
        finally:
            _CURRENT["dir"] = None

    L.run_single_example = run_single_example


def endpoint_failures(example_dir):
    """The number of failed model requests of one example that are the endpoint's (`harness`)."""
    n = 0
    try:
        with open(os.path.join(example_dir, ENDPOINT_FAILURES), encoding="utf-8") as f:
            for line in f:
                try:
                    n += 1 if json.loads(line).get("harness") else 0
                except ValueError:
                    continue
    except OSError:
        return 0
    return n


def _shard():
    """Parse GBENCH_SHARD="I/N" (1-indexed shard I of N). Returns (i, n), or None when it is unset
    or empty. Launchers run inside the container and cannot import gbench, so this mirrors
    sampling.parse_shard inline: a malformed or out-of-range spec stops the launcher (exit status
    naming it), never a silent full run."""
    spec = os.environ.get("GBENCH_SHARD", "").strip()
    if not spec:
        return None
    try:
        i, n = (int(x) for x in spec.split("/", 1))
    except ValueError:
        sys.exit(f"GBENCH_SHARD must be I/N with integers (1 <= I <= N), got {spec!r}")
    if not (n >= 1 and 1 <= i <= n):
        sys.exit(f"GBENCH_SHARD must be I/N with 1 <= I <= N, got {spec!r}")
    return i, n


def _limit():
    """OSWORLD_LIMIT as a positive int, or None when it is unset, empty or not positive; a value
    that is not an integer stops the launcher."""
    raw = os.environ.get("OSWORLD_LIMIT", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        sys.exit(f"OSWORLD_LIMIT must be an integer, got {raw!r}")
    return n if n > 0 else None


#: The seed of the --eval-limit draw (the suite name, as gbench's sampling.stratified_sample uses).
LIMIT_SEED = "ui_control_osworld"
#: How a limited run selects its tasks; the summary's selection.limit_draw (gbench checks it).
LIMIT_DRAW = f"stratified_indices(key=domain, seed={LIMIT_SEED})"


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


def _build_subset_meta():
    """Load evaluation_examples/<manifest> (domain -> [example_ids]); optionally filter by domain;
    apply the shard (round-robin over the DETERMINISTIC sorted (domain, example_id) full task list)
    FIRST, then the limit draws WITHIN the shard (a seeded draw balanced across the domains:
    stratified_indices, LIMIT_DRAW); rebuild + write the per-domain subset used as
    --test_all_meta_path. Compose order matches the native path (base.run_eval_suite: shard then
    limit). Returns (path, selected {domain: [ids]}, selection record)."""
    manifest = os.environ.get("OSWORLD_TEST_META", "test_all.json")
    src = os.path.join(OSWORLD_DIR, "evaluation_examples", manifest)
    with open(src, encoding="utf-8") as f:
        meta = json.load(f)
    domain = os.environ.get("OSWORLD_DOMAIN", "all").strip()
    if domain and domain != "all":
        meta = {domain: meta.get(domain, [])}

    # Flatten to a DETERMINISTIC, SORTED full task list so the shard partition is reproducible across
    # machines and matches sampling.shard_select's "already-ordered sequence" contract.
    flat = sorted((d, ex) for d, ids in meta.items() for ex in ids)
    n_full = len(flat)

    # Shard FIRST: shard I of N = flat[I-1::N] (non-overlapping; the N shards union back to the full set).
    shard = _shard()
    if shard:
        i, n = shard
        flat = flat[i - 1::n]

    # Then the limit draws WITHIN the shard: a seeded draw balanced across the domains (sorted
    # order kept), not the head of the sorted list (which is one domain).
    limit = _limit()
    if limit:
        flat = [flat[k] for k in stratified_indices([d for d, _ex in flat], limit, LIMIT_SEED)]

    # Rebuild the per-domain manifest from the selected set, preserving deterministic order. Empty
    # domains fall out naturally (only selected pairs are added).
    selected: dict = {}
    for d, ex in flat:
        selected.setdefault(d, []).append(ex)

    out = os.path.join(_run_dir(), "test_subset.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(selected, f)
    selection = {"n_full": n_full, "limit": limit, "n_selected": len(flat),
                 "limit_draw": LIMIT_DRAW if limit else None}
    shard_record = ({"spec": f"{shard[0]}/{shard[1]}", "n_full": n_full, "n_selected": len(flat)}
                    if shard else None)
    return out, selected, dict(selection, shard=shard_record)


def _run_osworld(subset_path):
    import run as OS  # OSWorld run.py (safe to import: no __main__ side effects)
    _install_example_tracking()
    obs = os.environ.get("OSWORLD_OBS_TYPE", "screenshot")
    results_dir = os.path.join(_run_dir(), "results")
    sys.argv = ["run.py",
                "--provider_name", "docker",
                "--headless",
                "--action_space", "pyautogui",
                "--observation_type", obs,
                "--model", _ROUTING_ALIAS,
                "--result_dir", results_dir,
                "--test_all_meta_path", subset_path,
                "--max_steps", os.environ.get("OSWORLD_MAX_STEPS", "15"),
                "--domain", "all"]   # subset already narrowed by _build_subset_meta
    temperature = os.environ.get("OSWORLD_TEMPERATURE", "").strip()
    if temperature:
        sys.argv += ["--temperature", str(float(temperature))]
    args = OS.config()
    with open(subset_path, encoding="utf-8") as f:
        test_all_meta = json.load(f)
    test_file_list = OS.get_unfinished(args.action_space, args.model, args.observation_type,
                                       args.result_dir, test_all_meta)
    OS.test(args, test_file_list)
    return args.action_space, args.model, obs, results_dir


def _summarize(action_space, use_model, obs, results_dir, selected):
    """Parse result_dir/<action_space>/<obs>/<use_model>/<domain>/<example>/result.txt (float in
    [0,1]). The rates here count a missing result.txt as 0 over the SELECTED examples; the gbench
    runner recomputes them over the tasks with a result (a missing one is a harness error) and
    keeps these as success_rate_missing_as_zero. n_scored / n_missing / tasks say which is which."""
    base = os.path.join(results_dir, action_space, obs, use_model)

    def _read(domain, ex):
        p = os.path.join(base, domain, ex, "result.txt")
        if not os.path.exists(p):
            return None
        try:
            return float(open(p, encoding="utf-8").read().strip())
        except Exception:
            try:
                return float(eval(open(p, encoding="utf-8").read().strip()))  # some tasks write exprs
            except Exception:
                return None

    per_domain, all_scores, n_selected, n_scored = {}, [], 0, 0
    for domain, ids in selected.items():
        dvals = []
        for ex in ids:
            n_selected += 1
            v = _read(domain, ex)
            if isinstance(v, (int, float)):
                n_scored += 1
                dvals.append(v)
                all_scores.append(v)
            else:
                dvals.append(0.0)          # missing/errored task counts as 0
                all_scores.append(0.0)
        per_domain[domain] = {"success_rate": (sum(dvals) / len(dvals)) if dvals else None,
                              "n": len(ids), "n_scored": sum(1 for e in ids if _read(domain, e) is not None)}
    overall = (sum(all_scores) / len(all_scores)) if all_scores else None
    tasks = [{"task": f"{domain}/{ex}", "domain": domain, "example_id": ex,
              "result": _read(domain, ex),
              "endpoint_failures": endpoint_failures(os.path.join(base, domain, ex))}
             for domain, ids in selected.items() for ex in ids]
    return {
        "success_rate": overall,
        "n_selected": n_selected,
        "n_scored": n_scored,
        "n_missing": n_selected - n_scored,
        # tasks during which a model request failed on the endpoint's side (ENDPOINT_FAILURES)
        "n_endpoint_failed": sum(1 for t in tasks if t["endpoint_failures"]),
        "per_domain": per_domain,
        "action_space": action_space,
        "observation_type": obs,
        "model_label": use_model,
        # the selected tasks ("<domain>/<example_id>"), each one's result (None: no result.txt)
        # and its model requests that failed on the endpoint's side
        "tasks": tasks,
    }


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(_run_dir(), exist_ok=True)
    os.chdir(WORKDIR)                       # docker_vm_data/ under the identity mount
    os.environ["OPENAI_BASE_URL"] = os.environ.get("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    os.environ["OPENAI_API_KEY"] = os.environ.get("GBENCH_MODEL_API_KEY", "dummy")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, OSWORLD_DIR)
    _install_model_rewrite(os.environ.get("GBENCH_MODEL_NAME", "served"))

    subset_path, selected, selection = _build_subset_meta()
    shard_record = selection.pop("shard")
    if not selected:
        # Nothing selected (a shard with none of the tasks): OSWorld is not started.
        summary = {"success_rate": None, "n_selected": 0, "n_scored": 0, "n_missing": 0,
                   "per_domain": {}, "tasks": [], "empty_selection": True}
    else:
        action_space, use_model, obs, results_dir = _run_osworld(subset_path)
        summary = _summarize(action_space, use_model, obs, results_dir, selected)
    summary["selection"] = selection
    summary["shard"] = shard_record
    summary_path = os.path.join(_run_dir(), "ui_control_osworld_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    hand_over(_run_dir())
    sys.stderr.write(f"[ui_control_osworld] summary: success_rate={summary['success_rate']} "
                     f"n_selected={summary['n_selected']} n_scored={summary['n_scored']}\n")
    return 0


def hand_over(path):
    """Give `path` and everything under it to OSWORLD_HOST_OWNER (uid:gid), the gbench runner's
    user; nothing when it is unset or malformed. Never raises."""
    raw = os.environ.get("OSWORLD_HOST_OWNER", "").strip()
    try:
        uid, gid = (int(x) for x in raw.split(":", 1))
    except ValueError:
        return
    for root, dirs, files in os.walk(path):
        for name in [*dirs, *files]:
            try:
                os.lchown(os.path.join(root, name), uid, gid)
            except OSError:
                pass
    try:
        os.lchown(path, uid, gid)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
