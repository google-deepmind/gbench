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
# gbench launcher for WildClawBench, run INSIDE the gbench-wildclawbench orchestrator image. It
# drives the pinned InternLM harness (`eval/run_batch.py`) docker-out-of-docker: run_batch spawns
# one sibling task container per task (from `wildclawbench-ubuntu:v1.3`, mounted host docker.sock),
# runs the OpenClaw agent inside pointed at the gbench /v1 model endpoint (via an injected
# `my_api.json` custom provider), then `docker exec`s the task's own `grade()` for the score.
#
# What gbench adds (everything else is upstream, untouched):
#   * writes my_api.json -> OpenClaw custom provider pointing at the model relay of the
#     orchestrator's proxy (WILDCLAW_MODEL_RELAY_URL), with the fixed non-secret apiKey
#     "gbench-model-relay": the proxy forwards the agent's model calls to the gbench model endpoint
#     and adds the endpoint's key there, so no model key is written into a task container;
#   * sets OPENROUTER_BASE_URL / OPENROUTER_API_KEY / JUDGE_MODEL so the tasks' verbatim
#     OpenAI-SDK judge calls hit the gbench Gemini proxy (wildclawbench_cascade_judge.py) instead of
#     OpenRouter/gpt-5.4 (all 43 judged tasks declare these three keys in their `## Env`, so
#     run_grading injects them; OPENROUTER_API_KEY is the fixed placeholder "gbench-cascade");
#   * serves the OpenClaw agent's web_search from Gemini Google Search grounding: right after
#     run_batch starts each task container, the launcher rewrites the Brave API origin
#     (https://api.search.brave.com) in that container's OpenClaw bundle to the orchestrator's Gemini
#     proxy (WILDCLAW_SEARCH_ORIGIN), sets tools.web.search.apiKey to a fixed non-secret
#     placeholder and tools.web.search.timeoutSeconds to AGENT_WEB_SEARCH_TIMEOUT_S (60 s in place
#     of OpenClaw's 30 s default, a recorded deviation: Gemini grounding needs longer than Brave),
#     and drops the `-e BRAVE_API_KEY=` run_batch passes. The tool keeps the name,
#     description, parameters and result shape the image ships; no key of any kind is placed in the
#     task container. The same step deletes the image's other search skills (SEARCH_SKILLS_REMOVED:
#     local-web-search and web-search on DuckDuckGo, google-search on Google Custom Search), so
#     web_search is the only search tool the agent is offered. A throwaway container checks the
#     setup applies to the task image before the run starts (preflight);
#   * starts each task container on WILDCLAW_TASK_NETWORK (default the docker bridge) and records
#     its addresses there; gives each task's grade() a per-run grader placeholder key
#     (WILDCLAW_GRADER_KEY) so the proxy tells the grader's judge calls from the agent's; checks,
#     before the run, that each selected task's provisioned inputs (workspace, ground truth) are
#     in the host checkout;
#   * stops, when a task's grading starts (its first `docker cp` into the task container, before
#     the ground truth is copied in), every process left in the task container except PID 1 and
#     the processes that ran right before the OpenClaw gateway started (the image's and the task
#     warmup's, such as the mock services the 03_* tasks' grade() queries; listed then with
#     their start times, before the agent ran anything). So nothing the agent started is running
#     while grade() runs or can read the grader key from its environment. A stop that fails is
#     told apart by its cause: one the agent cannot cause (the docker exec timed out or could not
#     run, or docker failed while the container still ran with its python3) or one it can
#     (processes left, python3 missing or broken). Upstream only stops the gateway after grading;
#     grade() itself is unchanged;
#   * runs a network monitor container (the task image, on the task network, gbench's own code)
#     for the whole run: every NETWORK_MONITOR_INTERVAL_S it resolves the host names the tasks
#     need (WILDCLAW_DNS_HOSTS) and opens a TCP connection to port 443 of WILDCLAW_EGRESS_HOSTS.
#     Its first round must show DNS and web egress before any task starts;
#   * points OUTPUT_SUBDIR at the identity-mounted workdir so score.json/usage.json land on the
#     host-visible mount, then walks that tree -> a single wildclawbench_summary.json with the
#     headline (the equal-weight mean of per-task overall_score over the selected tasks without a
#     harness error), the modality (multimodal/pure-text) breakdown, avg time/cost, the harness
#     errors per task and by kind (HARNESS_ERROR_KINDS: a task with a provisioned input missing,
#     whose container setup failed, that produced no score, whose grading failed, hit the 120 s
#     cap or fell back on a judge failure, during which the model endpoint or a Gemini call
#     failed, as the proxy's harness events, attributed by time window and container address,
#     show, or during which the task network lost DNS or web egress, as the network monitor
#     shows), the Gemini call
#     tally and the count of task containers whose web-search setup failed, so the runner can
#     report harness errors apart from the model's score.
#
# DooD identity-mount contract (the skillsbench pitfall): run_batch bind-mounts each task's
# `<workspace>/exec` dir into the sibling container, and `-v <path>` from inside a container is a
# HOST path. So the harness ROOT (with its `workspace/` data) MUST be at an identity path (same on
# host + orchestrator). The gbench runner arranges that by identity-mounting the provisioned host
# checkout; this launcher just runs run_batch from there.
#
# Env contract (set by the entrypoint / docker run -e):
#   WILDCLAW_HARNESS_DIR   harness root, identity-mounted (default /app/WildClawBench)
#   WILDCLAW_WORKDIR       identity-mounted host path for output + summary (default /out)
#   GBENCH_MODEL_NAME      served model name (sent as the OpenAI `model`)
#   WILDCLAW_MODEL_RELAY_URL the proxy's model relay as the task containers reach it
#                            (<proxy origin>/model/v1): the my_api.json baseUrl (required)
#   WILDCLAW_JUDGE_ENDPOINT  OPENROUTER_BASE_URL the tasks' judge client uses (the Gemini proxy)
#   WILDCLAW_SEARCH_ORIGIN   origin of the Gemini proxy as the task containers reach it; replaces
#                            the Brave API origin in their OpenClaw bundle (required)
#   WILDCLAW_TASK_IMAGE    the task image (default wildclawbench-ubuntu:v1.3)
#   WILDCLAW_TASK_NETWORK  task-container docker network (default bridge)
#   WILDCLAW_DNS_HOSTS     comma list of host names the network monitor resolves (the runner
#                          passes wildclawbench.TASK_HOSTS; default DNS_HOSTS_DEFAULT)
#   WILDCLAW_EGRESS_HOSTS  comma list of hosts the network monitor connects to on port 443 (the
#                          runner passes wildclawbench.EGRESS_HOSTS; default EGRESS_HOSTS_DEFAULT)
#   WILDCLAW_GRADER_KEY    the grader placeholder key (the entrypoint draws one per run)
#   WILDCLAW_TASK_LABEL    run-scoped label put on every task container (for reaping)
#   WILDCLAW_CATEGORIES    comma list of categories (default: all six)
#   WILDCLAW_TASK          path to a single task .md (smoke; overrides categories)
#   WILDCLAW_LIMIT         draw N tasks of the selected set, balanced across the categories
#                          (--eval-limit; LIMIT_DRAW, seeded; default none)
#   WILDCLAW_PARALLEL      per-run parallelism (default 1)
#   WILDCLAW_THINKING      OpenClaw thinking level (default unset)
# The Gemini keys and the model endpoint's key (GBENCH_MODEL_API_KEY), and the model endpoint URL
# the relay forwards to (GBENCH_MODEL_BASE_URL), are for the proxy: the entrypoint starts this
# launcher without the keys.

import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid

# The agent's web_search timeout the launcher writes to every task container's OpenClaw config,
# and OpenClaw's own default it replaces: one source of truth with the proxy's search deadline
# (wildclawbench_cascade_judge.WEB_SEARCH_DEADLINE_S = AGENT_WEB_SEARCH_TIMEOUT_S - 3), which sits
# next to this file in /app/adapter (stdlib only, nothing runs at import).
from wildclawbench_cascade_judge import (AGENT_WEB_SEARCH_TIMEOUT_S,  # noqa: E402
                                         UPSTREAM_WEB_SEARCH_TIMEOUT_S)

HARNESS_DIR = os.environ.get("WILDCLAW_HARNESS_DIR", "/app/WildClawBench")
WORKDIR = os.environ.get("WILDCLAW_WORKDIR", "/out")
ALL_CATEGORIES = [
    "01_Productivity_Flow", "02_Code_Intelligence", "03_Social_Interaction",
    "04_Search_Retrieval", "05_Creative_Synthesis", "06_Safety_Alignment",
]
_PROVIDER = "gbench-model"


def _task_index():
    """frontmatter `id` -> {category, modality, path} for every task .md in the harness."""
    out = {}
    for tf in glob.glob(os.path.join(HARNESS_DIR, "tasks", "*", "*task_*.md")):
        try:
            head = open(tf, encoding="utf-8").read(4000)
        except OSError:
            continue
        mid = re.search(r"^id:\s*(.+)$", head, re.M)
        if not mid:
            continue
        mmod = re.search(r"^modality:\s*(.+)$", head, re.M)
        out[mid.group(1).strip()] = {
            "category": os.path.basename(os.path.dirname(tf)),
            "modality": (mmod.group(1).strip() if mmod else "pure-text"),
            "path": tf,
        }
    return out


def _limit():
    """--eval-limit N forwarded as WILDCLAW_LIMIT -> the number of tasks drawn from the selected
    set (LIMIT_DRAW). Unset / empty / non-positive means no cap (None); a value that is not an
    integer stops the launcher (never read as "no cap", a full run)."""
    raw = os.environ.get("WILDCLAW_LIMIT", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        sys.exit(f"WILDCLAW_LIMIT must be an integer, got {raw!r}")
    return n if n > 0 else None


#: The seed of the --eval-limit draw (the suite name, as gbench's sampling.stratified_sample uses).
LIMIT_SEED = "wildclawbench"
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


def _shard():
    """--shard I/N forwarded as GBENCH_SHARD -> (i, n), or None when it is unset or empty. Parsed
    inline because launcher scripts run INSIDE the container and cannot import gbench
    (sampling.parse_shard). The gbench runner validated the spec before forwarding it; a malformed
    or out-of-range value here still stops the launcher (exit status naming it), never a silent
    full run."""
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


def _selected_task_ids(index):
    """The task ids this run SELECTED - the true denominator (a task that never produced a
    score.json still counts as 0, matching upstream print_global_summary and no-skip honesty).
    GBENCH_SHARD selects shard I of N over the deterministic sorted id list (round-robin
    ids[i-1::n], matching sampling.shard_select) FIRST; WILDCLAW_LIMIT then draws N of that
    shard's ids, balanced across the categories (stratified_indices, LIMIT_DRAW; sorted order
    kept), not the head of the sorted list, which is one category (same compose order as the
    native path)."""
    single = os.environ.get("WILDCLAW_TASK", "").strip()
    if single:
        try:
            head = open(single, encoding="utf-8").read(4000)
        except OSError:
            return []
        mid = re.search(r"^id:\s*(.+)$", head, re.M)
        return [mid.group(1).strip()] if mid else []
    cats = [c.strip() for c in os.environ.get("WILDCLAW_CATEGORIES", "").split(",") if c.strip()]
    cats = set(cats or ALL_CATEGORIES)
    ids = sorted(tid for tid, meta in index.items() if meta["category"] in cats)
    shard = _shard()
    if shard is not None:
        i, n = shard
        ids = ids[i - 1::n]           # shard FIRST (over the full sorted set)
    limit = _limit()
    if limit is None:
        return ids
    # then draw within the shard, balanced across the categories
    keys = [index[tid]["category"] for tid in ids]
    return [ids[k] for k in stratified_indices(keys, limit, LIMIT_SEED)]


#: The apiKey the task containers' OpenClaw sends to the model relay. Not a credential: the proxy
#: replaces it with the model endpoint's own key (GBENCH_MODEL_API_KEY, which only the proxy holds).
MODEL_KEY_PLACEHOLDER = "gbench-model-relay"


def _model_relay_url():
    """The proxy's model relay as the task containers reach it (WILDCLAW_MODEL_RELAY_URL)."""
    return os.environ.get("WILDCLAW_MODEL_RELAY_URL", "").strip().rstrip("/")


def _write_models_config():
    """Write my_api.json: an OpenClaw custom provider aimed at the proxy's model relay, with the
    placeholder apiKey (the proxy adds the model endpoint's key)."""
    model = os.environ.get("GBENCH_MODEL_NAME", "served")
    cfg = {"providers": {_PROVIDER: {
        "baseUrl": _model_relay_url(),
        "apiKey": MODEL_KEY_PLACEHOLDER,
        "api": "openai-completions",
        "models": [{"id": model, "name": model}],
    }}}
    path = os.path.join(WORKDIR, "my_api.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path, f"{_PROVIDER}/{model}"


#: The Brave Search API origin compiled into the image's OpenClaw bundle (BRAVE_SEARCH_ENDPOINT /
#: BRAVE_LLM_CONTEXT_ENDPOINT in dist/*.js of OpenClaw 2026.3.11). Rewritten per task container to
#: the orchestrator's Gemini proxy, which answers the web search from Gemini Google Search grounding.
BRAVE_API_ORIGIN = "https://api.search.brave.com"
#: The value tools.web.search.apiKey is set to in each task container. Not a credential: the Gemini
#: proxy ignores it. It replaces the image's "${BRAVE_API_KEY}" reference, without which the OpenClaw
#: gateway refuses to start (WEB_SEARCH_KEY_UNRESOLVED_NO_FALLBACK).
SEARCH_KEY_PLACEHOLDER = "gbench-gemini-grounding"
#: The OpenClaw config of the task image (the one run_batch's docker_utils edits too).
OPENCLAW_CONFIG = "/root/.openclaw/openclaw.json"
#: Written by the launcher, one JSON line per task container whose web-search setup failed.
SEARCH_SETUP_FAILURES = "search_setup_failures.jsonl"
#: The task image's search skills other than the web_search tool (their SKILL.md `name`), deleted
#: from every task container: local-web-search (DuckDuckGo HTML scraping, workspace skill
#: /root/skills/local-web-search-skill), web-search (the DuckDuckGo search API through
#: duckduckgo-search, workspace skill /root/skills/web-search) and google-search (the Google Custom
#: Search API, an OpenClaw bundled skill). With them gone, the Gemini-served web_search is the only
#: search tool the agent is offered.
SEARCH_SKILLS_REMOVED = ("local-web-search", "web-search", "google-search")

# Runs with the task image's python3 (3.10) inside a task container. Rewrites every OpenClaw bundle
# file that holds the Brave API origin, checks none is left, then points tools.web.search.apiKey at
# the placeholder, sets tools.web.search.timeoutSeconds to TIMEOUT (AGENT_WEB_SEARCH_TIMEOUT_S; the
# image sets none, so OpenClaw would use its 30 s default; what the image had is printed as
# image_timeout_seconds), and deletes every skill named in SKILLS from OpenClaw's skill directories (the
# workspace's skills/, ~/.openclaw/skills and the bundled skills/), checking none is left. Exits
# non-zero (message on stderr) when the image is not the one this was built against (no bundle file
# holds the origin, the provider is not brave, or brave.mode is set) or a search skill cannot be
# deleted.
_SEARCH_SETUP_SCRIPT = r"""
import json, os, pathlib, shutil, sys
ORIGIN, BRAVE, PLACEHOLDER, CONFIG, TIMEOUT, SKILLS = sys.argv[1:7]
SKILLS = [s for s in SKILLS.split(",") if s]
root = pathlib.Path(os.path.realpath(shutil.which("openclaw") or "/nonexistent")).parent
dist = root / "dist"
if not dist.is_dir():
    sys.exit("wildclawbench web-search setup: OpenClaw bundle not found (%s)" % dist)
needle, repl = BRAVE.encode(), ORIGIN.encode()
patched = []
for p in sorted(dist.rglob("*.js")):
    data = p.read_bytes()
    if needle in data:
        p.write_bytes(data.replace(needle, repl))
        patched.append(str(p.relative_to(root)))
left = [str(p) for p in dist.rglob("*.js") if needle in p.read_bytes()]
if not patched or left:
    sys.exit("wildclawbench web-search setup: expected the Brave API origin in the OpenClaw bundle "
             "(patched=%d, left=%d)" % (len(patched), len(left)))
cfg_path = pathlib.Path(CONFIG)
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
search = cfg.get("tools", {}).get("web", {}).get("search")
if not isinstance(search, dict) or search.get("provider") != "brave":
    sys.exit("wildclawbench web-search setup: the image's tools.web.search provider is not brave")
if (search.get("brave") or {}).get("mode") not in (None, "web"):
    sys.exit("wildclawbench web-search setup: brave.mode %r is not served" % search["brave"]["mode"])
search["apiKey"] = PLACEHOLDER
image_timeout = search.get("timeoutSeconds")
search["timeoutSeconds"] = int(TIMEOUT)
cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def skill_name(skill_md):
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    for line in text[3:end if end > 0 else len(text)].splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "name":
            return value.strip().strip("\"'")
    return None

defaults = (cfg.get("agents") or {}).get("defaults") or {}
workspace = pathlib.Path(os.path.expanduser(defaults.get("workspace")
                                            or str(cfg_path.parent / "workspace")))
roots = [workspace / "skills", cfg_path.parent / "skills", root / "skills"]

def search_skills():
    found = []
    for skill_root in roots:
        if skill_root.is_dir():
            for md in sorted(skill_root.glob("*/SKILL.md")):
                name = skill_name(md)
                if name in SKILLS:
                    found.append((name, md.parent))
    return found

removed = []
for name, skill_dir in search_skills():
    try:
        shutil.rmtree(skill_dir)
    except OSError as e:
        sys.exit("wildclawbench web-search setup: cannot delete the %s skill at %s (%s)"
                 % (name, skill_dir, e))
    removed.append({"name": name, "dir": str(skill_dir)})
left = search_skills()
if left:
    sys.exit("wildclawbench web-search setup: search skills left: %s"
             % ", ".join("%s (%s)" % (n, d) for n, d in left))
written = json.loads(cfg_path.read_text(encoding="utf-8"))["tools"]["web"]["search"]
if written.get("timeoutSeconds") != int(TIMEOUT):
    sys.exit("wildclawbench web-search setup: tools.web.search.timeoutSeconds reads %r, not %s"
             % (written.get("timeoutSeconds"), TIMEOUT))
print(json.dumps({"patched_files": len(patched), "provider": search["provider"],
                  "enabled": search.get("enabled", True), "removed_search_skills": removed,
                  "timeout_seconds": written["timeoutSeconds"],
                  "image_timeout_seconds": image_timeout}))
"""


def _search_origin():
    """The Gemini proxy origin the task containers reach (WILDCLAW_SEARCH_ORIGIN, no trailing /)."""
    return os.environ.get("WILDCLAW_SEARCH_ORIGIN", "").strip().rstrip("/")


def _setup_args(origin):
    """The setup script's arguments (after `-c <script>`)."""
    return [origin, BRAVE_API_ORIGIN, SEARCH_KEY_PLACEHOLDER, OPENCLAW_CONFIG,
            str(int(AGENT_WEB_SEARCH_TIMEOUT_S)), ",".join(SEARCH_SKILLS_REMOVED)]


def search_setup_command(container):
    """docker exec argv that points `container`'s OpenClaw web_search at the Gemini proxy, sets its
    web_search timeout to AGENT_WEB_SEARCH_TIMEOUT_S and deletes the image's other search
    skills."""
    return ["docker", "exec", container, "python3", "-c", _SEARCH_SETUP_SCRIPT,
            *_setup_args(_search_origin())]


def preflight_search_setup(run=subprocess.run):
    """Apply the web-search setup to a throwaway, network-less container of the task image, so an
    image it does not fit fails the run before any task starts. Returns (ok, message)."""
    origin = _search_origin()
    if not re.match(r"^https?://[^/\s]+$", origin):
        return False, f"WILDCLAW_SEARCH_ORIGIN must be an http(s) origin, got {origin!r}"
    image = os.environ.get("WILDCLAW_TASK_IMAGE", "wildclawbench-ubuntu:v1.3")
    cmd = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3", image, "-c",
           _SEARCH_SETUP_SCRIPT, *_setup_args(origin)]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, (f"the web-search setup does not apply to task image {image!r}: "
                       f"{(r.stderr or r.stdout or '').strip()[-600:]}")
    return True, (r.stdout or "").strip()


#: The task (its frontmatter id) the current thread is running, set by _track_task_windows: run_batch
#: runs each task, with its container start, in one thread.
_CURRENT_TASK = threading.local()
#: Written by the launcher, one JSON line per task run: {task_id, start, end} (time.time()), whether
#: the agent backend failed before the agent ran (backend_error: run_single_task's result "error",
#: set only when the container start or setup raised), whether run_single_task itself raised, the
#: grading error message run_single_task returned, if any, the task container's addresses on its
#: docker network (ips; none with the host network), the processes that ran right before the
#: OpenClaw gateway started (process_baseline: [[pid, start time], ...]), the stop of the
#: processes left when its grading started (isolation: {ok, stopped, left, error, cause, start,
#: end}), its grade() execs (grading: [{start, end, timed_out, isolated, stop_cause,
#: container_down, name_resolution_errors}]) and the lines of its logs that show a failed name
#: lookup (name_resolution_errors: {log file: lines}).
TASK_WINDOWS = "task_windows.jsonl"
#: Written by the Gemini proxy (wildclawbench_cascade_judge.HARNESS_EVENTS): one JSON line per
#: failure of a service a task depends on, {time, kind, detail, client, start, service_failure}
#: (start: when the failed chat request arrived; service_failure: the Gemini service failed it,
#: as opposed to the request being refused), plus a model_endpoint_recovered line for the first
#: model relay success after a failure (per client).
HARNESS_EVENTS = "harness_events.jsonl"
#: The proxy's event kinds of a request that carried the grader key (a task's grade()).
GRADER_EVENT_KINDS = ("judge", "judge_web_search")
#: Written by the Gemini proxy (wildclawbench_cascade_judge.GRADER_CALLS): one JSON line per judge
#: request of a task's grade(), {start, end, client, ok}.
GRADER_CALLS = "grader_calls.jsonl"
#: Written by the Gemini proxy (wildclawbench_cascade_judge.GEMINI_ANSWERS): one JSON line per
#: answered request, {time, start, kind, model, client, grader}; the summary records the answering
#: Gemini models per task.
GEMINI_ANSWERS = "gemini_answers.jsonl"
#: Written by the Gemini proxy (wildclawbench_cascade_judge.MODEL_REQUESTS): one JSON line per chat
#: request its model relay passed on, {time, start, client, status, stream, max_tokens,
#: finish_reason, content, reasoning, usage}; the summary records the max_tokens OpenClaw sent,
#: the replies cut off at it and the token usage the endpoint reported, per task.
MODEL_REQUESTS = "model_requests.jsonl"
#: The tool result text of a web_search call OpenClaw aborted at its web_search timeout
#: (AGENT_WEB_SEARCH_TIMEOUT_S, imported above from the proxy; the Node.js AbortError message).
WEB_SEARCH_ABORTED = "This operation was aborted"
#: Written by the launcher before the run: {task_id: [what is missing]} for the selected tasks whose
#: provisioned inputs (workspace directory, ground-truth files) are not in the host checkout.
MISSING_INPUTS = "missing_inputs.json"
#: The environment variable that holds the placeholder key the launcher gives each task's grade()
#: (OPENROUTER_API_KEY in its `docker exec`): the proxy tells the grader's judge requests apart from
#: the agent's by it. The entrypoint draws a new random value for every run; it is not a credential.
GRADER_KEY_ENV = "WILDCLAW_GRADER_KEY"


# --------------------------------------------------------------------------- #
# Grading isolation: nothing the agent started runs while grade() runs.
# --------------------------------------------------------------------------- #
# Both scripts run with the task image's python3 (`docker exec <container> python3 -c`). scan()
# lists the live processes of the container's PID namespace as {pid: start time} (/proc/<pid>/stat
# field 22, in clock ticks since boot, so a reused pid is a different process), zombies left out.
_PROC_SCAN = r"""
# gbench-wildclaw-helper
import json, os, signal, sys, time
def scan(skip):
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in skip:
            continue
        try:
            with open("/proc/%d/stat" % pid, "rb") as f:
                data = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        fields = data[data.rfind(")") + 2:].split()
        if len(fields) < 20 or fields[0] in ("Z", "X"):
            continue
        out[pid] = int(fields[19])
    return out
"""
#: Prints [[pid, start time], ...] of the container's processes, its own left out.
_PROCESS_SNAPSHOT_SCRIPT = _PROC_SCAN + r"""
print(json.dumps(sorted([pid, start] for pid, start in scan({os.getpid()}).items())))
"""
#: Stops every process of the container except PID 1, itself and the [pid, start time] pairs in
#: argv[1] (JSON): SIGSTOP to all of them, so none can fork, then SIGKILL, again until none is
#: found (at most 50 rounds), then waits up to 10 s for them to exit. Prints {"stopped": n,
#: "left": [pid, ...]}; exits 3 when a process is left, else 0.
_STOP_LEFTOVER_SCRIPT = _PROC_SCAN + r"""
keep = {(int(p), int(s)) for p, s in json.loads(sys.argv[1])}
skip = {1, os.getpid()}
def targets():
    return {pid: start for pid, start in scan(skip).items() if (pid, start) not in keep}
stopped = set()
for _ in range(50):
    found = targets()
    if not found:
        break
    for sig in (signal.SIGSTOP, signal.SIGKILL):
        for pid in found:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
    stopped.update(found.items())
    time.sleep(0.1)
left = targets()
deadline = time.time() + 10
while left and time.time() < deadline:
    time.sleep(0.2)
    left = targets()
print(json.dumps({"stopped": len(stopped), "left": sorted(left)}))
sys.exit(3 if left else 0)
"""
#: The limit for one `docker exec` of the process snapshot or of the stop, in seconds.
PROCESS_EXEC_TIMEOUT_S = 120
#: The transcript loader upstream's grading.py copies into the task container before grade().
TRANSCRIPT_LOADER = "/tmp/_transcript_loader.py"

#: container -> the processes that ran right before its OpenClaw gateway started.
_BASELINES = {}
#: container -> the result of the stop when its grading started.
_ISOLATION = {}
_STATE_LOCK = threading.Lock()


def _tail(text, limit=300):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else "..." + text[-limit:]


def snapshot_processes(container, run=subprocess.run):
    """[[pid, start time], ...] of the processes running in `container`. Raises RuntimeError when
    docker cannot list them (the task then fails before the agent runs: a harness error)."""
    try:
        r = run(["docker", "exec", container, "python3", "-c", _PROCESS_SNAPSHOT_SCRIPT],
                capture_output=True, text=True, timeout=PROCESS_EXEC_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(f"wildclawbench: could not list the processes of {container} before "
                           f"the agent started ({type(e).__name__})") from e
    if getattr(r, "returncode", 1) != 0:
        raise RuntimeError(f"wildclawbench: could not list the processes of {container} before "
                           f"the agent started (exit {r.returncode}: {_tail(r.stderr)})")
    try:
        procs = json.loads((r.stdout or "").strip().splitlines()[-1])
        return sorted([int(p), int(s)] for p, s in procs)
    except (ValueError, IndexError, TypeError) as e:
        raise RuntimeError(f"wildclawbench: the process list of {container} is unreadable "
                           f"({_tail(r.stdout)})") from e


def container_down(container, run=subprocess.run):
    """True when docker reports `container` as not running or gone; False when it runs or docker
    cannot say."""
    try:
        r = run(["docker", "inspect", "-f", "{{.State.Running}}", container],
                capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return False
    if getattr(r, "returncode", 1) != 0:
        return "no such" in str(r.stderr or "").lower()
    return str(r.stdout or "").strip() == "false"


#: Why the stop of a task's leftover processes failed (stop_leftover_processes' `cause`): "infra"
#: when the agent cannot have caused it (no process list was taken before the agent ran, the
#: docker exec could not run, or the exec timed out or docker failed while the container still
#: ran with no process beyond the ones from before the agent and the OpenClaw gateway, and for a
#: failed exec its python3 still worked), "model" when it can (processes left after the stop,
#: python3 missing, broken or printing something else, or a timed-out or failed exec while more
#: processes than that ran, such as a fork storm, or while docker top could not list them:
#: `stop_note` says which), "container_down" when docker no longer runs the container (its grade()
#: exec then fails too, a grading_infra harness error).
STOP_FAILURE_CAUSES = ("infra", "model", "container_down")
#: Marks gbench's own processes in a task container (the snapshot and the stop scripts' first
#: line), so a count of the container's processes leaves them out.
HELPER_MARK = "gbench-wildclaw-helper"
#: Processes beyond the pre-agent snapshot a container may run with nothing the agent started:
#: the OpenClaw gateway the harness starts after the snapshot.
HARNESS_PROCESSES_AFTER_SNAPSHOT = 1


def processes_beyond_snapshot(container, keep, run=subprocess.run):
    """How many processes `container` runs beyond the pre-agent snapshot `keep` (gbench's own
    helpers left out), as `docker top` lists them from the daemon's side (no exec into the
    container, so it works while the container is overloaded); None when docker cannot list them."""
    try:
        r = run(["docker", "top", container, "-o", "pid,args"], capture_output=True, text=True,
                timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if getattr(r, "returncode", 1) != 0:
        return None
    lines = [ln for ln in str(r.stdout or "").splitlines()[1:] if ln.strip()]
    live = [ln for ln in lines if HELPER_MARK not in ln]
    return max(0, len(live) - len(keep or []))


def _cause_from_processes(container, keep, run, why):
    """(cause, stop_note) for a stop exec that timed out or failed while the container runs: the
    agent can have caused it when more processes run than before the agent plus the OpenClaw
    gateway, or when docker top cannot list them (the ambiguous case, left as the model's)."""
    extra = processes_beyond_snapshot(container, keep, run)
    if extra is None:
        return "model", (f"{why}, and docker top could not list the container's processes: left "
                         f"as the model's outcome")
    if extra > HARNESS_PROCESSES_AFTER_SNAPSHOT:
        return "model", (f"{why} while {extra} process(es) beyond the ones from before the agent "
                         f"ran (the agent's)")
    return "infra", f"{why} with no process of the agent's left"


def _python3_works(container, run=subprocess.run):
    """True when `python3 -c pass` runs in `container` (docker exec exit 0)."""
    try:
        r = run(["docker", "exec", container, "python3", "-c", "pass"], capture_output=True,
                text=True, timeout=PROCESS_EXEC_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return getattr(r, "returncode", 1) == 0


def stop_leftover_processes(container, keep, run=subprocess.run):
    """Stop every process in `container` except PID 1 and the [pid, start time] pairs in `keep`
    (the processes that ran before the agent). Returns {ok, stopped, left, error, cause, start,
    end} (plus container_down when docker could not run the stop); ok only when the stop ran and
    no process is left; cause (STOP_FAILURE_CAUSES) says, for a failed stop, whether the agent can
    have caused it. `keep` None (no snapshot was taken) stops nothing: ok False, cause infra."""
    result = {"ok": False, "stopped": 0, "left": [], "error": None, "cause": None,
              "start": time.time()}
    if keep is None:
        result["error"] = "no process list of the task container from before the agent ran"
        result["cause"] = "infra"
        result["end"] = time.time()
        return result
    try:
        r = run(["docker", "exec", container, "python3", "-c", _STOP_LEFTOVER_SCRIPT,
                 json.dumps(keep)], capture_output=True, text=True,
                timeout=PROCESS_EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        result["error"] = f"the stop did not finish within {PROCESS_EXEC_TIMEOUT_S} s"
        if container_down(container, run):
            result["container_down"] = True
            result["cause"] = "container_down"
        else:
            result["cause"], result["stop_note"] = _cause_from_processes(
                container, keep, run, "the stop timed out")
    except OSError as e:
        result["error"] = f"docker exec could not run ({type(e).__name__})"
        result["cause"] = "infra"
    else:
        parsed = None
        for line in reversed(str(r.stdout or "").strip().splitlines()):
            try:
                parsed = json.loads(line)
                break
            except ValueError:
                continue
        good = (isinstance(parsed, dict) and isinstance(parsed.get("stopped"), int)
                and isinstance(parsed.get("left"), list))
        if good:
            result["stopped"], result["left"] = parsed["stopped"], parsed["left"]
        rc = getattr(r, "returncode", 1)
        if rc == 0 and good and not result["left"]:
            result["ok"] = True
        else:
            result["error"] = (f"exit {rc}"
                               + (f", {len(result['left'])} process(es) left" if result["left"]
                                  else "")
                               + (f": {_tail(r.stderr)}" if r.stderr else "")
                               + ("" if good else f" (output: {_tail(r.stdout, 120)})"))
            if good or rc in (126, 127):
                result["cause"] = "model"         # processes left, or python3 missing/unusable
            elif rc != 0 and container_down(container, run):
                result["container_down"] = True
                result["cause"] = "container_down"
            elif rc != 0 and _python3_works(container, run):
                # docker failed while the container and python3 work: the agent's doing only
                # when its processes still run
                result["cause"], result["stop_note"] = _cause_from_processes(
                    container, keep, run, f"the stop exec failed (exit {rc})")
            else:
                result["cause"] = "model"
    result["end"] = time.time()
    return result


# --------------------------------------------------------------------------- #
# Name-lookup failures in a task's logs.
# --------------------------------------------------------------------------- #
#: Text of a failed name lookup as glibc, Python, curl, Node.js, urllib3 and Rust tools print it.
NAME_RESOLUTION_ERRORS = ("Temporary failure in name resolution", "EAI_AGAIN",
                          "Could not resolve host", "Name or service not known",
                          "getaddrinfo ENOTFOUND", "No address associated with hostname",
                          "NameResolutionError", "failed to lookup address information")
#: The most bytes of one log file the scan reads.
MAX_LOG_SCAN_BYTES = 256 * 1024 * 1024


def count_name_resolution_errors(text):
    """The number of lines of `text` that show a failed name lookup."""
    return sum(1 for line in str(text or "").splitlines()
               if any(p in line for p in NAME_RESOLUTION_ERRORS))


def scan_name_resolution_errors(paths):
    """{file name: lines that show a failed name lookup} for the files in `paths` (missing files
    and files without such a line are left out)."""
    out = {}
    for path in paths:
        n, read = 0, 0
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    read += len(line)
                    if any(p in line for p in NAME_RESOLUTION_ERRORS):
                        n += 1
                    if read > MAX_LOG_SCAN_BYTES:
                        break
        except OSError:
            continue
        if n:
            out[os.path.basename(path)] = out.get(os.path.basename(path), 0) + n
    return out


def _iso_time(value):
    """Seconds since the epoch of an ISO 8601 timestamp ("2026-09-28T22:15:24.072Z"), or None."""
    import datetime
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def count_web_search_aborts(path):
    """The agent's web_search calls OpenClaw aborted at its web_search timeout, from a task's
    transcript (chat.jsonl): a web_search tool result that is an error reading WEB_SEARCH_ABORTED
    and was written at least AGENT_WEB_SEARCH_TIMEOUT_S - 1 s after the assistant message that
    made the call (a call aborted sooner, such as by the end of the task's time, is not counted;
    one whose call time is unknown is). The proxy's reply never reached the agent: a harness
    error of the search service (agent_web_search), whatever the proxy recorded."""
    calls, n, read = {}, 0, 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                read += len(line)
                if read > MAX_LOG_SCAN_BYTES:
                    break
                if "toolCall" not in line and "toolResult" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                msg = rec.get("message") if isinstance(rec, dict) else None
                if not isinstance(msg, dict):
                    continue
                when = _iso_time(rec.get("timestamp"))
                if msg.get("role") == "assistant":
                    for part in msg.get("content") or []:
                        if isinstance(part, dict) and part.get("type") == "toolCall":
                            calls[part.get("id")] = when
                    continue
                if msg.get("role") != "toolResult" or msg.get("toolName") != "web_search" or \
                        not msg.get("isError"):
                    continue
                text = " ".join(str(p.get("text") or "") for p in msg.get("content") or []
                                if isinstance(p, dict)).strip()
                if text != WEB_SEARCH_ABORTED:
                    continue
                called = calls.get(msg.get("toolCallId"))
                if called is None or when is None or \
                        when - called >= AGENT_WEB_SEARCH_TIMEOUT_S - 1.0:
                    n += 1
    except OSError:
        return 0
    return n


# --------------------------------------------------------------------------- #
# Network monitor: DNS and web egress of the task network for the whole run.
# --------------------------------------------------------------------------- #
#: Host names the task containers need: the package indexes the tasks' warmups install from
#: (pip, npm, apt with the image's sources) and sites the tasks name. The runner passes
#: wildclawbench.TASK_HOSTS (the same names) as WILDCLAW_DNS_HOSTS.
DNS_HOSTS_DEFAULT = ("pypi.org", "files.pythonhosted.org", "registry.npmjs.org",
                     "archive.ubuntu.com", "security.ubuntu.com", "deb.nodesource.com",
                     "github.com", "arxiv.org", "zh.wikipedia.org")
#: Hosts whose port 443 the monitor connects to (WILDCLAW_EGRESS_HOSTS from the runner's
#: wildclawbench.EGRESS_HOSTS).
EGRESS_HOSTS_DEFAULT = ("github.com", "pypi.org", "registry.npmjs.org")
#: Seconds from the start of one monitor round to the next.
NETWORK_MONITOR_INTERVAL_S = 15
#: A stretch of a run longer than this without a monitor round is a gap in the monitoring.
NETWORK_MONITOR_GAP_S = 60
#: How long the launcher waits for the monitor's first round before the run starts, in seconds.
NETWORK_MONITOR_FIRST_ROUND_S = 120
#: The workdir subdirectory mounted into the monitor, and the file it writes there: one JSON line
#: per round, {t, dns: {name: "ok" | "fail <why>"}, tcp: {host: "ok" | "fail <why>"}}.
NETWORK_MONITOR_DIR = "netmon"
NETWORK_MONITOR = "network_monitor.jsonl"
#: The host firewall rule that blocks the metadata server and keeps DNS (container_guard's
#: FIREWALL_COMMAND; this image cannot import gbench).
FIREWALL_COMMAND = "sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP"

# Runs with the task image's python3 in the monitor container: each round resolves every name
# (all at once, 20 s budget), then connects to port 443 of each egress host that resolved (5 s),
# and appends the round to argv[1]; a new round starts every argv[2] seconds.
_NETWORK_MONITOR_SCRIPT = r"""
import json, socket, sys, threading, time
out_path, interval = sys.argv[1], float(sys.argv[2])
names = [n for n in sys.argv[3].split(",") if n]
egress = [n for n in sys.argv[4].split(",") if n]
DNS_BUDGET, TCP_TIMEOUT = 20.0, 5.0

def in_threads(fn, items, budget):
    res = {}
    workers = []
    for item in items:
        w = threading.Thread(target=fn, args=(item, res))
        w.daemon = True
        w.start()
        workers.append(w)
    end = time.time() + budget
    for w in workers:
        w.join(max(0.0, end - time.time()))
    return dict(res)

def resolve(name, res):
    try:
        infos = socket.getaddrinfo(name, 443, 0, socket.SOCK_STREAM)
        v4 = [i for i in infos if i[0] == socket.AF_INET]
        res[name] = ("ok", (v4 or infos)[0])
    except socket.gaierror as e:
        res[name] = ("fail", "gaierror:%s" % e.errno)
    except Exception as e:
        res[name] = ("fail", type(e).__name__)

def connect(item, res):
    host, info = item
    s = None
    try:
        s = socket.socket(info[0], socket.SOCK_STREAM)
        s.settimeout(TCP_TIMEOUT)
        s.connect(info[4])
        res[host] = ("ok", "")
    except socket.timeout:
        res[host] = ("fail", "timeout")
    except OSError as e:
        res[host] = ("fail", "errno:%s" % e.errno)
    finally:
        if s is not None:
            s.close()

every = list(dict.fromkeys(names + egress))
while True:
    t = time.time()
    found = in_threads(resolve, every, DNS_BUDGET)
    dns = {}
    for n in names:
        word, detail = found.get(n, ("fail", "timeout"))
        dns[n] = "ok" if word == "ok" else "fail " + detail
    ready = [(h, found[h][1]) for h in egress if found.get(h, ("fail",))[0] == "ok"]
    opened = in_threads(connect, ready, TCP_TIMEOUT + 2)
    tcp = {}
    for h in egress:
        if found.get(h, ("fail",))[0] != "ok":
            tcp[h] = "fail no-address"
        else:
            word, detail = opened.get(h, ("fail", "timeout"))
            tcp[h] = "ok" if word == "ok" else "fail " + detail
    with open(out_path, "a") as f:
        f.write(json.dumps({"t": t, "dns": dns, "tcp": tcp}) + "\n")
    time.sleep(max(1.0, interval - (time.time() - t)))
"""


def _host_list(env, default):
    raw = os.environ.get(env, "").strip()
    return [h.strip() for h in raw.split(",") if h.strip()] if raw else list(default)


def dns_hosts():
    """The host names the network monitor resolves (WILDCLAW_DNS_HOSTS, else the default)."""
    return _host_list("WILDCLAW_DNS_HOSTS", DNS_HOSTS_DEFAULT)


def egress_hosts():
    """The hosts the network monitor connects to on port 443 (WILDCLAW_EGRESS_HOSTS, else the
    default)."""
    return _host_list("WILDCLAW_EGRESS_HOSTS", EGRESS_HOSTS_DEFAULT)


def network_monitor_command(name, out_dir):
    """docker run argv of the network monitor container `name`: the task image on the task
    network, gbench's script, read-only, no capabilities, `out_dir` its only mount."""
    net = os.environ.get("WILDCLAW_TASK_NETWORK", "bridge").strip()
    label = os.environ.get("WILDCLAW_TASK_LABEL", "").strip()
    image = os.environ.get("WILDCLAW_TASK_IMAGE", "wildclawbench-ubuntu:v1.3")
    cmd = ["docker", "run", "-d", "--pull", "never", "--name", name]
    if label:
        cmd += ["--label", label]
    if net:
        cmd += ["--network", net]
    cmd += ["--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{out_dir}:{out_dir}:rw", "--entrypoint", "python3", image,
            "-c", _NETWORK_MONITOR_SCRIPT, os.path.join(out_dir, NETWORK_MONITOR),
            str(NETWORK_MONITOR_INTERVAL_S), ",".join(dns_hosts()), ",".join(egress_hosts())]
    return cmd


def network_rounds():
    """The monitor's rounds, oldest first: [{t, dns, tcp}]."""
    rounds = []
    for r in _read_jsonl(os.path.join(NETWORK_MONITOR_DIR, NETWORK_MONITOR)):
        try:
            t = float(r["t"])
        except (KeyError, TypeError, ValueError):
            continue
        rounds.append({"t": t, "dns": r.get("dns") if isinstance(r.get("dns"), dict) else {},
                       "tcp": r.get("tcp") if isinstance(r.get("tcp"), dict) else {}})
    return sorted(rounds, key=lambda x: x["t"])


def classify_round(rd):
    """{down, failed, dns_degraded, dns_failed, tcp_failed} of one monitor round: down when no
    name resolved or no connection opened (the task network has no DNS, or no web egress), failed
    when anything failed, dns_degraded when at least half of the names did not resolve (the
    resolver itself failing, not one site's name or a TCP timeout)."""
    dns_failed = sorted(n for n, v in rd["dns"].items() if str(v) != "ok")
    tcp_failed = sorted(h for h, v in rd["tcp"].items() if str(v) != "ok")
    down = bool((rd["dns"] and len(dns_failed) == len(rd["dns"]))
                or (rd["tcp"] and len(tcp_failed) == len(rd["tcp"])))
    degraded = bool(rd["dns"] and 2 * len(dns_failed) >= len(rd["dns"]))
    return {"down": down, "failed": bool(dns_failed or tcp_failed), "dns_degraded": degraded,
            "dns_failed": dns_failed, "tcp_failed": tcp_failed}


def monitor_gaps(rounds, span):
    """[[from, to], ...]: the stretches of `span` (start, end) longer than NETWORK_MONITOR_GAP_S
    without a monitor round (all of it when there is no round)."""
    start, end = span
    ts = sorted(rd["t"] for rd in rounds)
    before = [t for t in ts if t < start]
    after = [t for t in ts if t > end]
    edges = ([before[-1]] if before else [start]) + [t for t in ts if start <= t <= end] + \
        ([after[0]] if after else [end])
    return [[a, b] for a, b in zip(edges, edges[1:]) if b - a > NETWORK_MONITOR_GAP_S]


def start_network_monitor(run=subprocess.run):
    """Start the network monitor container. Returns (name, None), or (None, why)."""
    out_dir = os.path.join(WORKDIR, NETWORK_MONITOR_DIR)
    try:
        os.makedirs(out_dir, exist_ok=True)
        os.chmod(out_dir, 0o777)
    except OSError as e:
        return None, f"cannot create {out_dir} ({e})"
    name = "gbench-wildclaw-netmon-" + uuid.uuid4().hex[:12]
    try:
        r = run(network_monitor_command(name, out_dir), capture_output=True, text=True,
                timeout=300)
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, f"docker run did not start it ({type(e).__name__})"
    if getattr(r, "returncode", 1) != 0:
        return None, f"docker run failed (exit {r.returncode}: {_tail(r.stderr)})"
    return name, None


def wait_first_round(name, run=subprocess.run, timeout_s=NETWORK_MONITOR_FIRST_ROUND_S,
                     sleep=time.sleep, clock=time.time):
    """(the monitor's first round, None), or (None, why) when it gives none in `timeout_s` or
    its container stops."""
    deadline = clock() + timeout_s
    while True:
        rounds = network_rounds()
        if rounds:
            return rounds[0], None
        if container_down(name, run):
            logs = ""
            try:
                logs = run(["docker", "logs", "--tail", "20", name], capture_output=True,
                           text=True, timeout=60).stderr
            except (subprocess.TimeoutExpired, OSError):
                pass
            return None, f"the monitor container stopped ({_tail(logs)})"
        if clock() >= deadline:
            return None, f"the monitor gave no result within {timeout_s} s"
        sleep(2)


def network_down_message(first):
    """The preflight failure for a first monitor round that is down."""
    c = classify_round(first)
    net = os.environ.get("WILDCLAW_TASK_NETWORK", "bridge").strip() or "bridge"
    if first["dns"] and len(c["dns_failed"]) == len(first["dns"]):
        shown = ", ".join(f"{n}: {first['dns'][n]}" for n in c["dns_failed"])
        return (f"containers on the task network {net!r} cannot resolve host names ({shown}), "
                "which the tasks need (package installs, the sites they name). On GCE the VPC "
                "DNS resolver is 169.254.169.254, the metadata server's address, so a host "
                "firewall rule that drops all traffic to it also drops DNS: block the metadata "
                f"server with `{FIREWALL_COMMAND}` instead. See docs/evals/wildclawbench.md.")
    shown = ", ".join(f"{h}: {first['tcp'][h]}" for h in c["tcp_failed"])
    return (f"containers on the task network {net!r} cannot open a connection to the web "
            f"({shown} on port 443), which the tasks need. Check the host's egress. See "
            "docs/evals/wildclawbench.md.")


def stop_network_monitor(name, run=subprocess.run):
    try:
        run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as e:
        sys.stderr.write(f"[wildclawbench] could not remove the network monitor {name} "
                         f"({type(e).__name__}); the runner removes it by label\n")


def _append_jsonl(name, record):
    try:
        with open(os.path.join(WORKDIR, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        sys.stderr.write(f"[wildclawbench] could not write {name}: {e}\n")


def _read_jsonl(name):
    """The JSON object lines of $WILDCLAW_WORKDIR/<name> ([] when it does not exist)."""
    try:
        with open(os.path.join(WORKDIR, name), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _record_search_setup_failure(container, detail):
    _append_jsonl(SEARCH_SETUP_FAILURES, {
        "container": container, "task_id": getattr(_CURRENT_TASK, "task_id", None),
        "detail": detail[-600:]})


def _count_search_setup_failures():
    try:
        with open(os.path.join(WORKDIR, SEARCH_SETUP_FAILURES), encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _track_task_windows(run_batch_module):
    """Wrap run_batch.run_single_task (looked up by name when run_batch's main() calls it) so each
    task run records its task id for its thread (_CURRENT_TASK, read when its container's
    web-search setup fails) and its time window in TASK_WINDOWS (to attribute the proxy's harness
    events to the tasks running at their time)."""
    orig = run_batch_module.run_single_task

    def run_single_task(task, *args, **kwargs):
        task_id = (task or {}).get("task_id") if isinstance(task, dict) else None
        _CURRENT_TASK.task_id = task_id
        start = time.time()
        record = {"task_id": task_id, "start": start, "raised": True, "ips": [], "grading": []}
        _CURRENT_TASK.record = record
        try:
            result = orig(task, *args, **kwargs)
            scores = result.get("scores") if isinstance(result, dict) else None
            record.update(
                raised=False,
                backend_error=bool(isinstance(result, dict) and result.get("error")),
                grading_error=(str(scores["error"])[:300]
                               if isinstance(scores, dict) and scores.get("error") else None))
            return result
        finally:
            _CURRENT_TASK.task_id = None
            _CURRENT_TASK.record = None
            # The task's gateway and agent logs, and the transcript copied next to them.
            logs = record.pop("_logs", [])
            if logs:
                logs.append(os.path.join(os.path.dirname(logs[0]), "chat.jsonl"))
            lookups = scan_name_resolution_errors(logs)
            graded = sum(int(g.get("name_resolution_errors") or 0)
                         for g in record.get("grading") or [])
            if graded:
                lookups["grade"] = graded
            if lookups:
                record["name_resolution_errors"] = lookups
            aborted = sum(count_web_search_aborts(p) for p in logs
                          if os.path.basename(p) == "chat.jsonl")
            if aborted:
                record["web_search_aborted"] = aborted
            record["end"] = time.time()
            _append_jsonl(TASK_WINDOWS, record)

    run_batch_module.run_single_task = run_single_task


def _strip_brave_env(cmd):
    """`cmd` without the `-e BRAVE_API_KEY=...` pair docker_utils.start_container always passes."""
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] == "-e" and i + 1 < len(cmd) and str(cmd[i + 1]).startswith("BRAVE_API_KEY="):
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out


def _patch_task_containers(docker_utils):
    """Adapt the container start in docker_utils.start_container (`docker run -d ...`); docker
    exec/cp/rm are untouched.

    * Network: WILDCLAW_TASK_NETWORK (default `bridge`, the upstream default) is injected as
      `--network`; the task container reaches the orchestrator's proxy at that network's gateway
      (`host`: at 127.0.0.1). An empty value leaves docker's default. WILDCLAW_TASK_LABEL labels the
      container for reaping on the host. Once it is up, its addresses on its network are recorded
      for its task (the proxy's events carry the client address, so an event is attributed to the
      task whose container made the request).
    * Grading: the task's `docker exec ... /tmp/_grade_runner.py` gets OPENROUTER_API_KEY set to the
      run's grader placeholder (WILDCLAW_GRADER_KEY), so the proxy can tell the grader's judge calls
      from the agent's; its start, end, whether it hit the 120 s cap, whether the processes left
      in the container had all been stopped (isolated), whether docker still ran the container
      after a failed exec, and the lines of its output that show a failed name lookup are
      recorded for the task.
    * Grading isolation: docker_utils.run_background (the OpenClaw gateway, then the agent) first
      lists the container's processes once (snapshot_processes), before the gateway starts; the
      first `docker cp` of the grading into the container (the ground truth, the transcript
      loader or the grade runner), or else the grade() exec, first stops every process started
      since, except PID 1 (stop_leftover_processes). A snapshot that cannot be taken fails the
      task's start (a harness error). The log files run_background writes are noted for the
      name-lookup scan.
    * Web search: `-e BRAVE_API_KEY=...` is dropped and, once the container is up, its OpenClaw
      web_search is pointed at the Gemini proxy and the image's other search skills are deleted
      (search_setup_command). When that setup fails, the start is reported as failed
      (start_container raises and run_batch records the task as errored) and the failure is
      written to SEARCH_SETUP_FAILURES with the task's id: the task is a harness error, left out
      of the headline, and never runs against a web search that is not the Gemini one.
    """
    net = os.environ.get("WILDCLAW_TASK_NETWORK", "bridge").strip()
    label = os.environ.get("WILDCLAW_TASK_LABEL", "").strip()
    grader_key = os.environ.get(GRADER_KEY_ENV, "").strip()
    orig_run = docker_utils.subprocess.run
    tmp_workspace = str(getattr(docker_utils, "TMP_WORKSPACE", "")
                        or os.environ.get("TMP_WORKSPACE", "/tmp_workspace")).rstrip("/")
    #: In-container destinations of the grading's `docker cp`s (grade_the_task, run_grading).
    grading_paths = {f"{tmp_workspace}/gt", TRANSCRIPT_LOADER, GRADE_RUNNER}

    def _before_grading(container):
        """Stop the processes left in `container`, once, when its grading starts."""
        with _STATE_LOCK:
            done = _ISOLATION.get(container)
            keep = _BASELINES.get(container)
        if done is None:
            done = stop_leftover_processes(container, keep, orig_run)
            with _STATE_LOCK:
                _ISOLATION[container] = done
            record = getattr(_CURRENT_TASK, "record", None)
            if record is not None:
                record["isolation"] = done
            if not done.get("ok"):
                sys.stderr.write(f"[wildclawbench] {container}: the processes left in the task "
                                 f"container could not all be stopped before grading "
                                 f"({done.get('error')})\n")
        return done

    def _run(cmd, *a, **k):
        if isinstance(cmd, list) and cmd[:2] == ["docker", "exec"] and GRADE_RUNNER in cmd:
            return _grade_exec(cmd, *a, **k)
        if isinstance(cmd, list) and cmd[:2] == ["docker", "cp"] and len(cmd) >= 4:
            container, sep, path = str(cmd[-1]).partition(":")
            if sep and container and path.rstrip("/") in grading_paths:
                _before_grading(container)
            return orig_run(cmd, *a, **k)
        if not (isinstance(cmd, list) and cmd[:3] == ["docker", "run", "-d"]):
            return orig_run(cmd, *a, **k)
        cmd = _strip_brave_env(cmd)
        inject = []
        if net and "--network" not in cmd:
            inject += ["--network", net]
        if label and "--label" not in cmd:
            inject += ["--label", label]
        if inject:
            cmd = cmd[:3] + inject + cmd[3:]
        started = orig_run(cmd, *a, **k)
        if getattr(started, "returncode", 1) != 0:
            return started
        container = cmd[cmd.index("--name") + 1] if "--name" in cmd else (started.stdout or "").strip()
        record = getattr(_CURRENT_TASK, "record", None)
        if record is not None:
            record["ips"] = container_ips(container, orig_run)
        setup = orig_run(search_setup_command(container), capture_output=True, text=True)
        if setup.returncode == 0:
            return started
        detail = (setup.stderr or setup.stdout or "").strip()
        _record_search_setup_failure(container, detail)
        return subprocess.CompletedProcess(
            cmd, 1, stdout=started.stdout,
            stderr=f"wildclawbench: pointing OpenClaw web_search at the Gemini proxy failed: {detail}")

    def _grade_exec(cmd, *a, **k):
        at = cmd.index(GRADE_RUNNER)
        container = str(cmd[at - 2]) if at >= 4 else ""
        isolation = _before_grading(container)
        if grader_key:
            cmd = [f"OPENROUTER_API_KEY={grader_key}" if str(x).startswith("OPENROUTER_API_KEY=")
                   else x for x in cmd]
        entry = {"start": time.time(), "timed_out": False, "isolated": bool(isolation.get("ok"))}
        if not entry["isolated"]:
            entry["stop_cause"] = isolation.get("cause")
        record = getattr(_CURRENT_TASK, "record", None)
        try:
            result = orig_run(cmd, *a, **k)
            if getattr(result, "returncode", 0) != 0 and container_down(container, orig_run):
                entry["container_down"] = True
            lookups = count_name_resolution_errors(
                f"{getattr(result, 'stdout', '') or ''}\n{getattr(result, 'stderr', '') or ''}")
            if lookups:
                entry["name_resolution_errors"] = lookups
            return result
        except subprocess.TimeoutExpired:
            entry["timed_out"] = True
            raise
        finally:
            entry["end"] = time.time()
            if record is not None:
                record.setdefault("grading", []).append(entry)

    docker_utils.subprocess.run = _run

    orig_background = getattr(docker_utils, "run_background", None)
    if callable(orig_background):
        def run_background(task_id, bash_cmd, log_path, *a, **k):
            record = getattr(_CURRENT_TASK, "record", None)
            with _STATE_LOCK:
                listed = task_id in _BASELINES
            if not listed:
                procs = snapshot_processes(task_id, orig_run)
                with _STATE_LOCK:
                    _BASELINES[task_id] = procs
                if record is not None:
                    record["process_baseline"] = procs
            if record is not None:
                record.setdefault("_logs", []).append(str(log_path))
            return orig_background(task_id, bash_cmd, log_path, *a, **k)

        # run_batch imports the agent runner, which binds run_background by name, after this.
        docker_utils.run_background = run_background


#: The grading runner upstream's grading.py copies into the task container and `docker exec`s.
GRADE_RUNNER = "/tmp/_grade_runner.py"


def container_ips(container, run=subprocess.run):
    """The addresses of `container` on its docker networks (`docker inspect`), [] with the host
    network or when docker cannot say."""
    try:
        r = run(["docker", "inspect", "-f",
                 "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{.GlobalIPv6Address}} {{end}}",
                 container], capture_output=True, text=True)
    except Exception:  # an address is a record; a failure leaves the task on time windows only
        return []
    if getattr(r, "returncode", 1) != 0:
        return []
    import ipaddress
    out = set()
    for x in str(r.stdout or "").split():
        try:
            out.add(str(ipaddress.ip_address(x)))
        except ValueError:      # "<no value>", "invalid IP" and other non-address words
            continue
    return sorted(out)


def _run_batch(models_config_path, model_arg):
    output_root = os.path.join(WORKDIR, "output")
    # run_batch + docker_utils read these at IMPORT time, so set them before importing.
    os.environ["OUTPUT_SUBDIR"] = output_root       # absolute -> lands on the identity mount
    os.environ["OPENROUTER_BASE_URL"] = os.environ.get(
        "WILDCLAW_JUDGE_ENDPOINT", os.environ.get("OPENROUTER_BASE_URL", ""))
    # Fixed placeholders, not credentials: the Gemini proxy ignores both (it serves the judge with the
    # Gemini cascade). run_batch passes them into the task containers of the tasks that declare them.
    os.environ["OPENROUTER_API_KEY"] = "gbench-cascade"
    os.environ["JUDGE_MODEL"] = "gbench-cascade"
    # run_batch substitutes MY_PROXY_API_KEY for a "${MY_PROXY_API_KEY}" in my_api.json; ours holds
    # the relay placeholder already, so this is the same placeholder, never the endpoint's key.
    os.environ["MY_PROXY_API_KEY"] = MODEL_KEY_PLACEHOLDER
    os.environ["DOCKER_IMAGE"] = os.environ.get("WILDCLAW_TASK_IMAGE", "wildclawbench-ubuntu:v1.3")
    # No Brave key exists in a gbench run (docker_utils reads BRAVE_API_KEY at import; the task
    # containers' web search is the Gemini proxy, see _patch_task_containers).
    os.environ.pop("BRAVE_API_KEY", None)

    # Drive the harness IN-PROCESS so the task-container monkeypatch takes effect (a subprocess would
    # not see it). Run from the identity-mounted harness root so `from src...` + workspace/ resolve.
    sys.path.insert(0, HARNESS_DIR)
    os.chdir(HARNESS_DIR)
    from src.utils import docker_utils           # noqa: E402
    _patch_task_containers(docker_utils)
    from eval import run_batch as RB             # noqa: E402
    _track_task_windows(RB)

    parallel = os.environ.get("WILDCLAW_PARALLEL", "1")
    common = ["--agent-backend", "openclaw", "--model", model_arg,
              "--models-config", models_config_path, "--parallel", str(parallel)]
    if os.environ.get("WILDCLAW_THINKING"):
        common += ["--thinking", os.environ["WILDCLAW_THINKING"]]

    single = os.environ.get("WILDCLAW_TASK", "").strip()
    invocations = []
    if single:
        invocations.append(["--task", single])
    elif _limit() is not None or _shard() is not None:
        # --eval-limit N and/or --shard I/N. Upstream run_batch has no --limit or --shard flag (only
        # --task / --category), so drive its single-task mode over the SELECTED task ids.
        # _selected_task_ids applies the SAME shard-then-cap, so the tasks we run == the tasks
        # _summarize uses as the denominator (run-set == score-set; n_tasks reflects the shard).
        index = _task_index()
        for tid in _selected_task_ids(index):
            path = index.get(tid, {}).get("path")
            if path:
                invocations.append(["--task", path])
    else:
        cats = [c.strip() for c in os.environ.get("WILDCLAW_CATEGORIES", "").split(",") if c.strip()]
        cats = cats or ALL_CATEGORIES
        for c in cats:
            invocations.append(["--category", c])

    for inv in invocations:
        argv = ["run_batch.py"] + inv + common
        sys.stderr.write(f"[wildclawbench] run_batch {' '.join(inv + common)}\n")
        sys.argv = argv
        # run_batch.main() sys.exit(1)s on a single-task failure; a category run never aborts. Don't
        # let that abort the launcher - the score tree is the source of truth.
        try:
            RB.main()
        except SystemExit:
            pass
        except Exception as exc:  # one category failing must not sink the rest
            sys.stderr.write(f"[wildclawbench] run_batch {inv} raised: {exc}\n")
    return output_root


def _score_of(score_json):
    """Per-task headline = overall_score, else mean of numeric metrics (harness semantics)."""
    numeric = {k: v for k, v in score_json.items() if isinstance(v, (int, float))}
    if not numeric:
        return None
    if "overall_score" in numeric:
        return float(numeric["overall_score"])
    return sum(numeric.values()) / len(numeric)


# The upstream tasks record a judge outage in several shapes: a `*_judge_method` OR bare
# `judge_method` set to a deterministic fallback ({regex,rule,keyword,heuristic}_fallback / "failed"),
# or a `*_judge_error` / bare `judge_error` key. Match all of them so judge_fallback_tasks is honest.
_FALLBACK_VALUES = {"regex_fallback", "rule_fallback", "keyword_fallback",
                    "heuristic_fallback", "failed"}


def _judge_fell_back(score_json):
    """True if any judge metric silently used a deterministic fallback (a judge outage)."""
    for k, v in score_json.items():
        kl = k.lower()
        if kl == "judge_error" or kl.endswith("judge_error"):
            return True
        if kl == "judge_method" or kl.endswith("judge_method"):
            if isinstance(v, str):
                vl = v.strip().lower()
                if vl in _FALLBACK_VALUES or "fallback" in vl or vl == "failed":
                    return True
    return False


def _collect(output_root, index):
    """Walk output/openclaw/<category>/<task_id>/<suffix>/{score.json,usage.json} -> rows by task_id.

    One row per task_id (the frontmatter id; newest suffix run wins), joined with modality.
    """
    oc_root = os.path.join(output_root, "openclaw")
    rows = {}
    for score_path in glob.glob(os.path.join(oc_root, "*", "*", "*", "score.json")):
        run_dir = os.path.dirname(score_path)
        task_id = os.path.basename(os.path.dirname(run_dir))
        category = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
        try:
            score_json = json.load(open(score_path, encoding="utf-8"))
        except Exception:
            continue
        usage = {}
        up = os.path.join(run_dir, "usage.json")
        if os.path.exists(up):
            try:
                usage = json.load(open(up, encoding="utf-8"))
            except Exception:
                usage = {}
        mtime = os.path.getmtime(score_path)
        prev = rows.get(task_id)
        if prev and prev["_mtime"] >= mtime:
            continue
        rows[task_id] = {
            "_mtime": mtime,
            "category": index.get(task_id, {}).get("category", category),
            "task_id": task_id,
            "modality": index.get(task_id, {}).get("modality", "pure-text"),
            "overall_score": _score_of(score_json),
            "grading_error": score_json.get("error"),
            "judge_fallback": _judge_fell_back(score_json),
            "missing": False,
            "elapsed_time_s": usage.get("elapsed_time"),
            # OpenClaw's own tally: its openai-completions provider gets no token usage from a
            # streamed reply the endpoint sends without it, and prices a custom model at 0, so a
            # 0 here is "not reported", not "none" (usage_fields decides what is recorded)
            "cost_usd": usage.get("cost_usd"),
            "output_tokens": usage.get("output_tokens"),
            "agent_request_count": usage.get("request_count"),
        }
    return rows


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def model_request_fields(requests):
    """What the relayed chat requests of one task (MODEL_REQUESTS records) say: the number of
    requests, the distinct per-turn max_tokens OpenClaw sent (None for a request that sent none),
    the replies that ended at that cap (finish_reason "length"), split into those with no answer
    text (cut while reasoning) and those with some, and the token usage when the endpoint reported
    it for every request (else None)."""
    caps = {r.get("max_tokens") if isinstance(r.get("max_tokens"), int) else None
            for r in requests}
    cut = {"in_reasoning": 0, "in_content": 0}
    for r in requests:
        if r.get("finish_reason") == "length":
            cut["in_content" if r.get("content") else "in_reasoning"] += 1
    usages = [r.get("usage") for r in requests]
    complete = bool(requests) and all(
        isinstance(u, dict) and isinstance(u.get("completion_tokens"), int) for u in usages)
    return {
        "model_requests": len(requests),
        "max_tokens_sent": sorted(caps, key=lambda c: (c is None, c or 0)),
        "replies_cut_at_max_tokens": cut,
        "relay_output_tokens": (sum(u["completion_tokens"] for u in usages) if complete
                                else None),
        "relay_input_tokens": (sum(int(u.get("prompt_tokens") or 0) for u in usages)
                               if complete else None),
    }


def usage_fields(row, relayed):
    """(output_tokens, cost_usd, token_usage_source) recorded for a task: the endpoint's own
    usage as the model relay saw it when it reported it for every request; else OpenClaw's tally
    (usage.json) when it counted tokens; else None, never 0, when the task made model requests
    (OpenClaw's tally is 0 then because nothing reported the usage). A cost is recorded only when
    OpenClaw priced something (a self-hosted model has no price)."""
    cost = row.get("cost_usd") if _positive(row.get("cost_usd")) else None
    if isinstance(relayed.get("relay_output_tokens"), int):
        return relayed["relay_output_tokens"], cost, "model relay (endpoint usage)"
    if _positive(row.get("output_tokens")):
        return row["output_tokens"], cost, "agent (usage.json)"
    if relayed.get("model_requests") or _positive(row.get("agent_request_count")):
        return None, cost, None
    return row.get("output_tokens"), cost, None


#: The kinds of harness error a task can have, in the order they are reported. A task with any of
#: them did not run, or was not graded as the model's own outcome, so it is left out of the headline:
#:   missing_input    a provisioned input its grading needs is not in the host checkout: its
#:                    workspace directory, the gt/ directory its grade() reads, or a ground-truth
#:                    file grade() names (missing_inputs(), checked before the run)
#:   search_setup     its container's web-search setup failed (the task did not run)
#:   task_start       the agent backend failed before the agent ran (run_single_task's result
#:                    "error": the task container's start or setup raised), or run_single_task
#:                    itself raised
#:   not_run          run_batch never ran it (no run record and no score)
#:   grading_infra    its grading could not run: upstream's grading.py failed to copy into the
#:                    container ("docker cp ..."), found no transcript loader, docker no longer
#:                    ran the container after a failed grade() exec, or the grade() exec hit its
#:                    120 s cap ("... timed out after ..."), with or without a judge call of that
#:                    grade() in flight, after the processes left in the container had been
#:                    stopped; or the stop of those processes failed for a reason the agent
#:                    cannot cause (STOP_FAILURE_CAUSES "infra") and then the grade() exec hit
#:                    its cap or a judge call of it failed on the Gemini service's side. A cap
#:                    hit or a judge failure of a grade() exec whose stop failed for a reason the
#:                    agent can cause is scored as upstream scores it, and counted
#:   no_score         its score.json holds no numeric metric (the grader produced no score, e.g.
#:                    a missing ground-truth directory)
#:   judge_fallback   its grade() recorded a judge failure or fell back to a deterministic check
#:   model_endpoint   the model endpoint failed during it (unreachable, HTTP 5xx or 429, or a
#:                    reply cut off; the proxy's model relay saw it) and no later relay call of
#:                    the same task succeeded (a failure the agent's retry got past is counted in
#:                    the summary's recovered_model_endpoint_events, not a harness error)
#:   network          the task network lost DNS or web egress while it ran: a network monitor
#:                    round in its run window where no name resolved or no connection opened, or
#:                    a round where at least half of the names did not resolve (the resolver
#:                    failing) while its logs (gateway, agent, transcript, grade() output) show a
#:                    failed name lookup. A failed lookup in its logs without such a round in its
#:                    window (a TCP timeout or one name failing says nothing about the resolver
#:                    the task used) is the model's outcome, counted per task
#:                    (name_resolution_errors_unconfirmed)
#:   judge, judge_web_search
#:                    a judge call of its grade() (the judge, the grounded judge) was served by
#:                    no cascade model, or the proxy could not serve the request, and the request
#:                    arrived while a grade() exec of that task ran after the processes left in
#:                    its container had been stopped (any other request with the grader key is
#:                    counted in grader_events_not_counted, not a harness error)
#:   agent_helper     a call of its agent to the OpenRouter helper model (the Gemini cascade)
#:                    failed on every attempt for any reason other than the Gemini API refusing
#:                    the request's own content (HTTP 400, 413 or 422 on every attempt), which is
#:                    the agent's outcome and never an event
#:   agent_web_search a web_search of its agent was served by no grounding model within the
#:                    proxy's deadline, its answer could not be written or came after OpenClaw's
#:                    web_search timeout (AGENT_WEB_SEARCH_TIMEOUT_S; the proxy's event), or its transcript shows a
#:                    web_search call OpenClaw aborted at that timeout (web_search_aborted)
#: An event is attributed to the task whose run window holds its time and, when the task
#: container's address is known (not with the host network), whose container made the request.
#: Everything else is the model's outcome, scored as run_batch scores it: a score.json whose
#: "error" sits next to numeric metrics (the task's grade() reporting the model's missing or bad
#: output, e.g. "results.md not found"), or a grade() that raised on its own (no score, 0).
HARNESS_ERROR_KINDS = ("missing_input", "search_setup", "task_start", "not_run", "grading_infra",
                       "no_score", "judge_fallback", "model_endpoint", "network", "judge",
                       "judge_web_search", "agent_helper", "agent_web_search")
#: The proxy's event for the first model relay success after a failure, per client.
MODEL_ENDPOINT_RECOVERED = "model_endpoint_recovered"


def grading_infra_failure(message):
    """True for a grading error message upstream's src/utils/grading.py (at the pinned commit)
    writes when the grading itself could not run: a docker cp failure, the missing transcript
    loader, or the grade() exec timing out (subprocess.TimeoutExpired: "... timed out after N
    seconds")."""
    m = str(message or "")
    return (m.startswith("docker cp") or m.startswith("transcript loader module not found")
            or grading_timeout(m))


def grading_timeout(message):
    """True for the grading error of a grade() exec that hit upstream's 120 s cap."""
    return "timed out after" in str(message or "")


# --------------------------------------------------------------------------- #
# Provisioned inputs a task's grading needs (checked before the run).
# --------------------------------------------------------------------------- #
#: A grade() source that reads the task's ground truth ("gt" as a path component, or gt_dir).
_GT_USE = re.compile(r"""["']gt["']|/gt["'/]|\bgt_dir\b|\bGT_DIR\b""")
#: Ground-truth files a grade() names (a name with an extension): "gt" / "<file>",
#: gt_dir / "<file>", GT_DIR / "<file>" and /tmp_workspace/gt/<file>.
_GT_FILES = (
    re.compile(r"""["']gt["']\s*/\s*["']([^"'/\s]+\.[A-Za-z0-9]+)["']"""),
    re.compile(r"""\b(?:gt_dir|GT_DIR)\s*/\s*["']([^"'/\s]+\.[A-Za-z0-9]+)["']"""),
    re.compile(r"""/gt/([A-Za-z0-9_.\-]+\.[A-Za-z0-9]+)(?=["'\s)])"""),
)


def _task_sections(task_md):
    """{section title: text} of a task .md as upstream's task_parser splits it (`## <title>`
    headers after the frontmatter), each text with its code fence stripped."""
    try:
        content = open(task_md, encoding="utf-8").read()
    except OSError:
        return {}
    m = re.match(r"^---\s*\n.*?\n---\s*\n(.*)", content, re.DOTALL)
    body = m.group(1) if m else content
    sections, current, lines = {}, None, []
    for line in body.split("\n"):
        header = re.match(r"^##\s+(.+)$", line)
        if header:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current, lines = header.group(1).strip(), []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    out = {}
    for title, text in sections.items():
        text = re.sub(r"^```[^\n]*\n?", "", text.strip())
        out[title] = re.sub(r"\n?```$", "", text).strip()
    return out


def missing_inputs(task_md):
    """What the task's grading needs from the host checkout that is not there: its workspace
    directory (## Workspace Path, relative to the harness root), the gt/ directory its grade()
    reads (## Automated Checks), and each ground-truth file grade() names. [] when everything is
    there, or when the task declares no workspace path."""
    sections = _task_sections(task_md)
    workspace = sections.get("Workspace Path", "").strip()
    if not workspace:
        return []
    if not os.path.isabs(workspace):
        workspace = os.path.normpath(os.path.join(HARNESS_DIR, workspace))
    if not os.path.isdir(workspace):
        return [f"workspace directory {workspace}"]
    checks = sections.get("Automated Checks", "")
    if not _GT_USE.search(checks):
        return []
    gt_dir = os.path.join(workspace, "gt")
    if not os.path.isdir(gt_dir) or not os.listdir(gt_dir):
        return [f"ground-truth directory {gt_dir} (missing or empty)"]
    names = sorted({n for rx in _GT_FILES for n in rx.findall(checks)})
    return [f"ground-truth file {os.path.join(gt_dir, n)}" for n in names
            if not os.path.exists(os.path.join(gt_dir, n))]


def find_missing_inputs(index, task_ids):
    """{task id: [what is missing]} for the tasks in `task_ids` (see missing_inputs)."""
    out = {}
    for tid in task_ids:
        path = (index.get(tid) or {}).get("path")
        gaps = missing_inputs(path) if path else []
        if gaps:
            out[tid] = gaps
    return out


def _read_json(name):
    try:
        with open(os.path.join(WORKDIR, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _harness_error_inputs():
    """(task ids whose container setup failed, {task id: [run records]}, [proxy events],
    [grader calls], {task id: [missing inputs]}) from the files the launcher and the proxy write
    in WORKDIR. An event is {time, kind, client, start, service_failure} (start: when its request
    arrived, else None; service_failure: the Gemini service failed it, None when not recorded); a
    grader call {start, end, client}."""
    setup = {r.get("task_id") for r in _read_jsonl(SEARCH_SETUP_FAILURES) if r.get("task_id")}
    runs = {}
    for r in _read_jsonl(TASK_WINDOWS):
        try:
            r["start"], r["end"] = float(r["start"]), float(r["end"])
        except (KeyError, TypeError, ValueError):
            continue
        runs.setdefault(r.get("task_id"), []).append(r)
    events = []
    for r in _read_jsonl(HARNESS_EVENTS):
        try:
            ev = {"time": float(r["time"]), "kind": str(r["kind"]),
                  "client": r.get("client") or None, "start": None,
                  "service_failure": (bool(r["service_failure"])
                                      if r.get("service_failure") is not None else None)}
        except (KeyError, TypeError, ValueError):
            continue
        try:
            ev["start"] = float(r["start"]) if r.get("start") is not None else None
        except (TypeError, ValueError):
            pass
        events.append(ev)
    calls = []
    for r in _read_jsonl(GRADER_CALLS):
        try:
            calls.append({"start": float(r["start"]), "end": float(r["end"]),
                          "client": r.get("client") or None})
        except (KeyError, TypeError, ValueError):
            continue
    missing = _read_json(MISSING_INPUTS)
    return setup, runs, events, calls, (missing if isinstance(missing, dict) else {})


def _from_task(rec, client):
    """Whether a request from `client` may come from the task run `rec`: always when either
    address is unknown (the host network), else when it is one of the task container's."""
    ips = rec.get("ips") or []
    return not client or not ips or client in ips


#: How a grade() cap hit is recorded (the summary's grading_timeouts): with_judge_call and
#: without_judge_call (a judge call of that grade() overlapped the timed-out exec, or none did),
#: unrecorded (no timed-out exec record) and not_isolated_infra (every timed-out exec ran after a
#: stop of the leftover processes that failed for a reason the agent cannot cause, and one of them
#: did) are harness errors; not_isolated (the stop failed for a reason the agent can cause) is
#: scored as upstream scores it.
GRADING_TIMEOUT_KINDS = ("with_judge_call", "without_judge_call", "unrecorded",
                         "not_isolated_infra", "not_isolated")


def _grading_windows(recs, isolated_only=True):
    """[(start, end, run record, exec entry)] of the grade() execs of these run records (with
    isolated_only, only those that ran after the processes left in the container had been
    stopped; a record without the flag, from an older launcher, counts as isolated)."""
    out = []
    for rec in recs:
        for g in rec.get("grading") or []:
            if isolated_only and not g.get("isolated", True):
                continue
            try:
                out.append((float(g["start"]), float(g["end"]), rec, g))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _grading_timeout_kind(recs, calls):
    """The GRADING_TIMEOUT_KINDS entry for a task whose grading hit the 120 s cap."""
    timed = [(rec, g) for rec in recs for g in (rec.get("grading") or []) if g.get("timed_out")]
    if not timed:
        return "unrecorded"
    isolated = [(rec, g) for rec, g in timed if g.get("isolated", True)]
    if not isolated:
        if any(g.get("stop_cause") == "infra" for _rec, g in timed):
            return "not_isolated_infra"
        return "not_isolated"
    for rec, g in isolated:
        try:
            g_start, g_end = float(g["start"]), float(g["end"])
        except (KeyError, TypeError, ValueError):
            continue
        for call in calls:
            if (call["start"] <= g_end and call["end"] >= g_start
                    and _from_task(rec, call["client"])):
                return "with_judge_call"
    return "without_judge_call"


def _summarize(output_root, index):
    """Roll the score tree into a summary. The denominator is the SELECTED task set. A task with a
    harness error (HARNESS_ERROR_KINDS: a provisioned input missing, it did not run, its container
    or grading could not run, its grade() hit the 120 s cap, it got no score, its grading fell
    back on a judge failure, the model endpoint or a Gemini call failed while it ran, or the task
    network lost DNS or web egress) is listed with its kinds and left out of the headline and the
    modality means. The headline is the equal-weight mean of overall_score over the other
    selected tasks, a task without a score among them counting as 0 (run_batch's global_avg).
    Each task also lists the Gemini models that answered its requests (gemini_models)."""
    found = _collect(output_root, index)
    selected = _selected_task_ids(index)
    rows = []
    for tid in selected:
        if tid in found:
            r = found[tid]
        else:
            r = {"category": index.get(tid, {}).get("category", "?"), "task_id": tid,
                 "modality": index.get(tid, {}).get("modality", "pure-text"),
                 "overall_score": None, "grading_error": None, "judge_fallback": False,
                 "missing": True, "elapsed_time_s": None, "cost_usd": None, "output_tokens": None,
                 "agent_request_count": None}
        rows.append({k: v for k, v in r.items() if k != "_mtime"})
    # If the selection could not be resolved (no index / unknown task), fall back to whatever scored
    # so a smoke run still reports - but this should not happen for a normal categories/all run.
    if not rows:
        rows = [{k: v for k, v in r.items() if k != "_mtime"} for r in found.values()]

    setup_failed, runs, events, grader_calls, missing = _harness_error_inputs()
    answers = _gemini_answers()
    relayed_requests = _model_requests()
    requests_attributed = set()
    answers_attributed = set()
    attributed, recovered_used, grader_not_counted = set(), set(), set()
    grader_unisolated, grader_counted_infra = set(), set()
    recoveries = [e for e in events if e["kind"] == MODEL_ENDPOINT_RECOVERED]
    rounds = network_rounds()
    checked = [(rd, classify_round(rd)) for rd in rounds]
    all_recs = [x for recs in runs.values() for x in recs]
    gaps = (monitor_gaps(rounds, (min(x["start"] for x in all_recs),
                                  max(x["end"] for x in all_recs))) if all_recs else [])
    timeouts = {k: 0 for k in GRADING_TIMEOUT_KINDS}
    unconfirmed = {}
    for r in rows:
        kinds = set()
        recs = runs.get(r["task_id"], [])
        if missing.get(r["task_id"]):
            kinds.add("missing_input")
            r["missing_inputs"] = list(missing[r["task_id"]])
        if r["task_id"] in setup_failed:
            kinds.add("search_setup")
        if any(x.get("backend_error") or x.get("raised") for x in recs):
            kinds.add("task_start")
        if r.get("missing") and not recs:
            kinds.add("not_run")
        messages = [r.get("grading_error")] + [x.get("grading_error") for x in recs]
        if any(grading_infra_failure(m) and not grading_timeout(m) for m in messages):
            kinds.add("grading_infra")
        if any(grading_timeout(m) for m in messages):
            # Every cap hit of a grade() exec that ran with nothing the agent started left in its
            # container is a harness error, with or without a judge call in flight.
            r["grading_timeout"] = _grading_timeout_kind(recs, grader_calls)
            timeouts[r["grading_timeout"]] += 1
            if r["grading_timeout"] != "not_isolated":
                kinds.add("grading_infra")      # not_isolated_infra included
        if any(g.get("container_down") for x in recs for g in x.get("grading") or []):
            kinds.add("grading_infra")
        isolations = [x["isolation"] for x in recs if isinstance(x.get("isolation"), dict)]
        if isolations:
            r["grading_isolated"] = all(i.get("ok") for i in isolations)
            r["processes_stopped"] = sum(int(i.get("stopped") or 0) for i in isolations)
            if not r["grading_isolated"]:
                failed_stop = next(i for i in isolations if not i.get("ok"))
                r["grading_isolation_error"] = str(failed_stop.get("error"))
                r["grading_isolation_cause"] = str(failed_stop.get("cause") or "unknown")
        if not r.get("missing") and not isinstance(r.get("overall_score"), (int, float)):
            kinds.add("no_score")
        if r.get("judge_fallback"):
            kinds.add("judge_fallback")
        all_grading = _grading_windows(recs, isolated_only=False)
        for i, ev in enumerate(events):
            if ev["kind"] == MODEL_ENDPOINT_RECOVERED:
                continue
            windows = [x for x in recs if x["start"] <= ev["time"] <= x["end"]
                       and _from_task(x, ev["client"])]
            if not windows:
                continue
            attributed.add(i)
            if ev["kind"] in GRADER_EVENT_KINDS:
                # A request with the grader key counts only when it arrived while a grade() exec
                # of this task ran with nothing the agent started left in its container; in an
                # exec whose stop failed for a reason the agent cannot cause, a failure of the
                # Gemini service's (not a refused request) counts as grading_infra.
                arrived = ev["start"] if ev["start"] is not None else ev["time"]
                during = [g for a, b, x, g in all_grading
                          if a <= arrived <= b and _from_task(x, ev["client"])]
                if not any(g.get("isolated", True) for g in during):
                    if during:
                        grader_unisolated.add(i)
                        if ev["service_failure"] and any(g.get("stop_cause") == "infra"
                                                         for g in during):
                            grader_counted_infra.add(i)
                            kinds.add("grading_infra")
                            continue
                    grader_not_counted.add(i)
                    continue
            if ev["kind"] == "model_endpoint" and any(
                    ev["time"] < rc["time"] <= x["end"] and _from_task(x, rc["client"])
                    and (not ev["client"] or not rc["client"] or rc["client"] == ev["client"])
                    for rc in recoveries for x in windows):
                recovered_used.add(i)
                continue
            kinds.add(ev["kind"])
        # The task network: the monitor's rounds in this task's run windows, confirmed failed
        # name lookups in its logs.
        in_window = [c for rd, c in checked if any(x["start"] <= rd["t"] <= x["end"] for x in recs)]
        lookups = {}
        for x in recs:
            for src, n in (x.get("name_resolution_errors") or {}).items():
                lookups[src] = lookups.get(src, 0) + int(n or 0)
        lookups = {k: v for k, v in lookups.items() if v}
        if any(c["down"] for c in in_window) or (lookups and
                                                 any(c["dns_degraded"] for c in in_window)):
            kinds.add("network")
            r["network_rounds_failed"] = sum(1 for c in in_window if c["failed"])
        elif lookups:
            r["name_resolution_errors_unconfirmed"] = lookups
            unconfirmed[r["task_id"]] = lookups
        if any(x["start"] < b and x["end"] > a for x in recs for a, b in gaps):
            r["network_unmonitored"] = True
        # web_search calls OpenClaw aborted at its timeout: the search service's reply never
        # reached the agent (the proxy's own record of that is an agent_web_search event)
        aborted = sum(int(x.get("web_search_aborted") or 0) for x in recs)
        if aborted:
            r["web_search_aborted"] = aborted
            kinds.add("agent_web_search")
        # The agent's model requests the relay passed on: the max_tokens OpenClaw sent, replies
        # cut off at it, and the token usage when the endpoint reported it.
        mine = []
        for j, q in enumerate(relayed_requests):
            if any(x["start"] <= q["time"] <= x["end"] and _from_task(x, q["client"])
                   for x in recs):
                requests_attributed.add(j)
                mine.append(q)
        relayed = model_request_fields(mine)
        r["model_requests"] = relayed["model_requests"]
        r["max_tokens_sent"] = relayed["max_tokens_sent"]
        if any(relayed["replies_cut_at_max_tokens"].values()):
            r["replies_cut_at_max_tokens"] = relayed["replies_cut_at_max_tokens"]
        r["output_tokens"], r["cost_usd"], r["token_usage_source"] = usage_fields(r, relayed)
        if relayed["relay_input_tokens"] is not None:
            r["input_tokens"] = relayed["relay_input_tokens"]
        # The Gemini models that answered this task's requests, by kind.
        models = {}
        for j, ans in enumerate(answers):
            if any(x["start"] <= ans["time"] <= x["end"] and _from_task(x, ans["client"])
                   for x in recs):
                answers_attributed.add(j)
                bucket = models.setdefault(ans["kind"], {})
                bucket[ans["model"]] = bucket.get(ans["model"], 0) + 1
        if models:
            r["gemini_models"] = {k: dict(sorted(v.items())) for k, v in sorted(models.items())}
        r["harness_errors"] = [k for k in HARNESS_ERROR_KINDS if k in kinds] + \
            sorted(k for k in kinds if k not in HARNESS_ERROR_KINDS)
    clean = [r for r in rows if not r["harness_errors"]]
    host_failures = {"dns": {}, "tcp": {}}
    for _rd, c in checked:
        for part in ("dns", "tcp"):
            for h in c[f"{part}_failed"]:
                host_failures[part][h] = host_failures[part].get(h, 0) + 1

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    def _overall(subset):
        n = len(subset)
        s = sum(r["overall_score"] for r in subset if isinstance(r["overall_score"], (int, float)))
        return (s / n) if n else None      # a task without a score contributes 0, and 1 to n

    n_tasks = len(rows)
    mm_rows = [r for r in rows if r["modality"] == "multimodal"]
    pt_rows = [r for r in rows if r["modality"] != "multimodal"]
    mm_overall = _overall([r for r in clean if r["modality"] == "multimodal"])
    pt_overall = _overall([r for r in clean if r["modality"] != "multimodal"])
    weighted = None
    if mm_overall is not None and pt_overall is not None:
        weighted = 0.5 * mm_overall + 0.5 * pt_overall
    kind_counts = {}
    for r in rows:
        for k in r["harness_errors"]:
            kind_counts[k] = kind_counts.get(k, 0) + 1
    event_counts = {}
    for ev in events:
        if ev["kind"] != MODEL_ENDPOINT_RECOVERED:
            event_counts[ev["kind"]] = event_counts.get(ev["kind"], 0) + 1
    n_events = sum(event_counts.values())
    cut_total = {"in_reasoning": 0, "in_content": 0}
    for r in rows:
        for k, v in (r.get("replies_cut_at_max_tokens") or {}).items():
            cut_total[k] = cut_total.get(k, 0) + int(v or 0)
    caps_sent = {c for q in relayed_requests
                 for c in [q["max_tokens"] if isinstance(q.get("max_tokens"), int) else None]}
    with_traffic = [r for r in rows if r.get("model_requests") or
                    _positive(r.get("agent_request_count"))]
    return {
        "n_tasks": n_tasks,
        "n_headline_tasks": len(clean),                  # the headline's denominator
        "n_scored": sum(1 for r in clean if isinstance(r["overall_score"], (int, float))),
        "n_missing": sum(1 for r in rows if r.get("missing")),
        "n_grading_errors": sum(1 for r in rows if r.get("grading_error")),
        "n_harness_errors": n_tasks - len(clean),        # tasks left out of the headline
        "harness_error_kinds": kind_counts,              # {kind: tasks}
        "harness_events": event_counts,                  # {kind: proxy events}
        # model endpoint failures a later relay success of the same task got past (not errors)
        "recovered_model_endpoint_events": len(recovered_used),
        "n_unattributed_harness_events": n_events - len(attributed),
        # proxy events of requests with the grader key that arrived outside every grade() exec
        # run with nothing the agent started left in its container (not harness errors)
        "grader_events_not_counted": len(grader_not_counted),
        # of the grader-key events, those that arrived during a grade() exec whose leftover
        # processes could not all be stopped (the run's status names them), and those of them
        # counted as grading_infra (a service failure after a stop the agent cannot have failed)
        "grader_events_in_unisolated_grading": len(grader_unisolated),
        "grader_events_counted_infra": len(grader_counted_infra),
        # the Gemini models that answered the run's requests, by kind, and the answers outside
        # every task run
        "gemini_models_answered": _answer_counts(answers),
        "gemini_answers_unattributed": len(answers) - len(answers_attributed),
        "missing_inputs": {k: list(v) for k, v in sorted(missing.items())},
        # tasks whose grade() hit the 120 s cap, by GRADING_TIMEOUT_KINDS
        "grading_timeouts": timeouts,
        # the stop of the processes left in the task containers when their grading started
        "grading_isolation": {
            "tasks_isolated": sum(1 for r in rows if r.get("grading_isolated") is True),
            "tasks_not_isolated": sum(1 for r in rows if r.get("grading_isolated") is False),
            "stop_failure_causes": _cause_counts(rows),
            "processes_stopped": sum(int(r.get("processes_stopped") or 0) for r in rows),
        },
        # the task network's DNS and web egress during the run
        "network_monitor": {
            "interval_s": NETWORK_MONITOR_INTERVAL_S,
            "rounds": len(checked),
            "failed_rounds": sum(1 for _rd, c in checked if c["failed"]),
            "down_rounds": sum(1 for _rd, c in checked if c["down"]),
            "host_failures": host_failures,
            "gaps": gaps,
        },
        "n_tasks_network_unmonitored": sum(1 for r in rows if r.get("network_unmonitored")),
        # failed name lookups in a task's logs with no monitor round in its window where the
        # resolver failed (at least half of the names; a TCP timeout or one name confirms
        # nothing): scored as run
        "name_resolution_errors_unconfirmed": unconfirmed,
        "global_mean": _overall(clean),                  # headline, harness errors left out
        "multimodal_mean": mm_overall,
        "pure_text_mean": pt_overall,
        "weighted_overall": weighted,                    # 0.5*MM + 0.5*pure-text (approx of leaderboard)
        "n_multimodal": len(mm_rows),
        "n_pure_text": len(pt_rows),
        "avg_time_min": (_mean([r["elapsed_time_s"] for r in rows]) / 60.0)
                        if _mean([r["elapsed_time_s"] for r in rows]) is not None else None,
        "avg_cost_usd": _mean([r["cost_usd"] for r in rows]),
        # the agent's model requests the proxy relayed: the per-turn max_tokens OpenClaw sent
        # (None: a request that sent none), the replies that ended at it (cut while reasoning:
        # no answer text yet), the output tokens when the endpoint reported them for every task
        # with model traffic (else None: OpenClaw's own tally has no usage to count)
        "model_requests": len(relayed_requests),
        "model_requests_unattributed": len(relayed_requests) - len(requests_attributed),
        "max_tokens_sent": sorted(caps_sent, key=lambda c: (c is None, c or 0)),
        "replies_cut_at_max_tokens": cut_total,
        "output_tokens_total": (sum(int(r["output_tokens"]) for r in with_traffic)
                                if with_traffic and all(isinstance(r.get("output_tokens"), int)
                                                        for r in with_traffic) else None),
        "tasks_without_token_usage": sum(1 for r in with_traffic
                                         if not isinstance(r.get("output_tokens"), int)),
        # web_search calls OpenClaw aborted at its web_search timeout (AGENT_WEB_SEARCH_TIMEOUT_S;
        # the tasks: agent_web_search)
        "web_search_aborted_calls": sum(int(r.get("web_search_aborted") or 0) for r in rows),
        "judge_fallback_tasks": sum(1 for r in rows if r["judge_fallback"]),
        "per_task": sorted(rows, key=lambda r: (r["category"], r["task_id"])),
    }


def _gemini_answers():
    """The proxy's answered requests (GEMINI_ANSWERS): [{time, kind, model, client}]."""
    out = []
    for r in _read_jsonl(GEMINI_ANSWERS):
        try:
            out.append({"time": float(r["time"]), "kind": str(r["kind"]),
                        "model": str(r.get("model") or "unknown"),
                        "client": r.get("client") or None})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _model_requests():
    """The chat requests the proxy's model relay passed on (MODEL_REQUESTS): [{time, client,
    max_tokens, finish_reason, content, reasoning, usage}]."""
    out = []
    for r in _read_jsonl(MODEL_REQUESTS):
        try:
            rec = dict(r)
            rec["time"] = float(r["time"])
            rec["client"] = r.get("client") or None
        except (KeyError, TypeError, ValueError):
            continue
        out.append(rec)
    return out


def _answer_counts(answers):
    """{kind: {model: answers}} over the answered requests."""
    out = {}
    for ans in answers:
        bucket = out.setdefault(ans["kind"], {})
        bucket[ans["model"]] = bucket.get(ans["model"], 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _cause_counts(rows):
    """{cause: tasks} over the tasks whose leftover processes could not all be stopped."""
    out = {}
    for r in rows:
        if r.get("grading_isolated") is False:
            cause = r.get("grading_isolation_cause") or "unknown"
            out[cause] = out.get(cause, 0) + 1
    return dict(sorted(out.items()))


def _gemini_usage():
    """The Gemini proxy's call tally ($WILDCLAW_WORKDIR/gemini_usage.json), or {} if it made none."""
    try:
        with open(os.path.join(WORKDIR, "gemini_usage.json"), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _empty_shard_summary():
    """The summary of a shard that holds none of the selected tasks (None when the run is not
    sharded or the shard holds some): nothing runs on this node."""
    shard = _shard()
    if shard is None or os.environ.get("WILDCLAW_TASK", "").strip():
        return None
    index = _task_index()
    if _selected_task_ids(index):
        return None
    cats = [c.strip() for c in os.environ.get("WILDCLAW_CATEGORIES", "").split(",") if c.strip()]
    cats = set(cats or ALL_CATEGORIES)
    return {"empty_shard": True, "shard": f"{shard[0]}/{shard[1]}", "n_tasks": 0, "n_scored": 0,
            "full_set_items": sum(1 for meta in index.values() if meta["category"] in cats),
            "per_task": []}


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    empty = _empty_shard_summary()
    if empty is not None:
        with open(os.path.join(WORKDIR, "wildclawbench_summary.json"), "w", encoding="utf-8") as f:
            json.dump(empty, f, indent=2)
        sys.stderr.write(f"[wildclawbench] shard {empty['shard']} holds none of the selected "
                         f"tasks; nothing runs on this node\n")
        return 0
    if not re.match(r"^https?://[^/\s]+/\S*$", _model_relay_url()):
        sys.stderr.write(f"[wildclawbench] PREFLIGHT FAILED: WILDCLAW_MODEL_RELAY_URL must be the "
                         f"proxy's http(s) model relay URL, got {_model_relay_url()!r}\n")
        return 5
    ok, detail = preflight_search_setup()
    if not ok:
        sys.stderr.write(f"[wildclawbench] PREFLIGHT FAILED: {detail}\n")
        return 5
    sys.stderr.write(f"[wildclawbench] web-search setup preflight: {detail}\n")
    try:
        setup = json.loads(detail)
    except ValueError:
        setup = {}
    setup = setup if isinstance(setup, dict) else {}
    # The task network's DNS and web egress, watched for the whole run; the first round must show
    # both before any task starts.
    monitor, why = start_network_monitor()
    if monitor is None:
        sys.stderr.write(f"[wildclawbench] PREFLIGHT FAILED: the network monitor of the task "
                         f"network could not start: {why}\n")
        return 5
    try:
        first, why = wait_first_round(monitor)
        if first is None:
            sys.stderr.write(f"[wildclawbench] PREFLIGHT FAILED: the network monitor of the task "
                             f"network gave no verdict: {why}\n")
            return 5
        if classify_round(first)["down"]:
            sys.stderr.write(f"[wildclawbench] PREFLIGHT FAILED: {network_down_message(first)}\n")
            return 5
        sys.stderr.write(f"[wildclawbench] network monitor first round: {json.dumps(first)}\n")
        models_config_path, model_arg = _write_models_config()
        # Provisioned inputs the selected tasks' grading needs, checked before the run: a task
        # whose input is missing still runs, and is a harness error (missing_input), never 0.
        index = _task_index()
        missing = find_missing_inputs(index, _selected_task_ids(index))
        with open(os.path.join(WORKDIR, MISSING_INPUTS), "w", encoding="utf-8") as f:
            json.dump(missing, f, indent=2)
        if missing:
            sys.stderr.write(f"[wildclawbench] provisioned inputs missing for {len(missing)} "
                             f"task(s): {', '.join(sorted(missing))}\n")
        output_root = _run_batch(models_config_path, model_arg)
    finally:
        stop_network_monitor(monitor)
    index = _task_index()
    summary = _summarize(output_root, index)
    summary["web_search_origin"] = _search_origin()
    # What the preflight found and deleted in the task image (every task container gets the same).
    summary["web_search_setup"] = setup
    # The agent's web_search timeout this launcher set (tools.web.search.timeoutSeconds), OpenClaw's
    # own default it replaced, and what the preflight read back from the task image's config.
    summary["agent_web_search_timeout"] = {
        "timeout_s": AGENT_WEB_SEARCH_TIMEOUT_S, "upstream_timeout_s": UPSTREAM_WEB_SEARCH_TIMEOUT_S,
        "applied_timeout_s": setup.get("timeout_seconds"),
        "image_timeout_s": setup.get("image_timeout_seconds")}
    summary["removed_search_skills"] = [
        s.get("name") for s in setup.get("removed_search_skills") or [] if isinstance(s, dict)]
    summary["n_search_setup_failures"] = _count_search_setup_failures()
    summary["gemini_usage"] = _gemini_usage()
    # how the task set was selected (an --eval-limit subset is LIMIT_DRAW; gbench checks it)
    limit = None if os.environ.get("WILDCLAW_TASK", "").strip() else _limit()
    summary["selection"] = {"limit": limit, "limit_draw": LIMIT_DRAW if limit else None}
    with open(os.path.join(WORKDIR, "wildclawbench_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    sys.stderr.write(f"[wildclawbench] summary: n_tasks={summary['n_tasks']} "
                     f"scored={summary['n_scored']} global_mean={summary['global_mean']} "
                     f"harness_errors={summary.get('n_harness_errors')} "
                     f"{summary.get('harness_error_kinds')} "
                     f"search_setup_failures={summary['n_search_setup_failures']} "
                     f"grading_timeouts={summary.get('grading_timeouts')} "
                     f"grading_isolation={summary.get('grading_isolation')} "
                     f"network_monitor_rounds="
                     f"{(summary.get('network_monitor') or {}).get('rounds')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
