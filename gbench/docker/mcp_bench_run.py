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
# gbench launcher for MCP-Bench, run INSIDE the gbench-mcp-bench image. It:
#   1. injects the gbench Gemini cascade judge into BenchmarkRunner (monkeypatch on __init__,
#      so it fires on the CLI path too and the hard-coded Azure o4-mini judge never runs);
#   2. checks that every server the task files need can serve here (built in the image; a
#      key-gated server's key set; network for the online servers: every host name in
#      MCP_DNS_NAMES resolves and a TCP connection to one of them opens). If any cannot, it
#      runs NO task: it writes the manifest naming those servers and an empty result, which
#      gbench turns into infra_required. It never runs the tasks of the servers that can serve
#      alone. gbench's pre-run check has already required all 28 servers, so this is a safety
#      net for a server lost between that check and the run;
#   3. points the model-under-test at the gbench /v1 endpoint via the openai_compatible provider,
#      and applies the run's model request settings to that provider's requests only
#      (mcp_bench_request_policy.py): the thinking switch, the operator's temperature when set,
#      and with --thinking the run's budget in place of upstream's max_tokens and a request
#      timeout sized to it; the counts of its replies and of its failed requests go to
#      MCP_REQUEST_LOG;
#   4. delegates to upstream benchmark.runner.main() (reusing all its glue) and writes the
#      averaged-metrics JSON to /out. The gbench harness computes the 0-1 Overall Score from it;
#   5. keeps each task's harness errors out of those averages (HarnessAccounting): a task in
#      which the last attempt of a model-under-test get_completion call ended in an endpoint or
#      connection error, a judge evaluation that could not be made, a failed name lookup (a
#      resolver failure naming one of MCP_DNS_NAMES or the container's resolver, or a lookup of
#      one of MCP_DNS_NAMES that failed, in a tool's error reply; or a DNS outage seen by the
#      watchdog while the task ran), a session to a credentialed server that failed outside the
#      model (not opened, its server gone before it answered, or lost while in use; the bridge's
#      record), or a task upstream recorded as failed, is left out of the metrics upstream's
#      aggregator computes, and the per-task record, with the Gemini models that answered its
#      judge calls, goes to MCP_HARNESS_LOG.
#
# The judge's key is gbench's per-run Gemini relay placeholder (GEMINI_API_KEY, GEMINI_API_KEYS),
# and its requests go to the relay at GEMINI_OPENAI_BASE_URL / GBENCH_GEMINI_RELAY_URL, a thread
# of the gbench process that adds the real key: no real Gemini key is in this container, where
# model-driven MCP servers run model text (see mcp_bench_cascade_judge.py). No server key is in it
# either: the five servers that need one (CREDENTIALED_SERVERS: commands.json entries whose `env`
# lists one of CREDENTIAL_KEYS) run in gbench's credentialed-servers container, and
# install_credentialed_bridges points their commands.json entries at mcp_bench_bridge.py, which
# carries each session there through gbench's relay (MCP_CREDENTIALED_RELAY,
# MCP_CREDENTIALED_TOKEN). The entrypoint refuses to start when one of the keys is set here.
#
# Env contract (set by the entrypoint / docker run -e):
#   MCPBENCH_DIR            repo root (default /app/mcp-bench)
#   MCP_MODEL_CONFIG_NAME   the --models key hijacked for the endpoint (default llama-3-1-8b)
#   MCP_TASK_FILES          comma-sep source task files (default the 3 canonical files)
#   MCP_OUTPUT              output JSON path (default /out/mcp_bench_results.json)
#   MCP_MANIFEST            run manifest path (default /out/subset_manifest.json)
#   MCP_HARNESS_LOG         the per-task harness record (default /out/harness_log.json)
#   MCP_DNS_NAMES           comma-separated host names the servers' APIs are reached at
#                           (gbench sets them; default api.github.com)
#   MCP_DISTRACTION_COUNT   pass-through to --distraction-count (optional)
#   MCP_DISABLE_STABILITY   if set, adds --disable-judge-stability
#   MCP_TASK_LIMIT          the number of tasks run (applied AFTER the shard): a seeded draw of the
#                           shard's task ids balanced across the server groups (LIMIT_DRAW), not
#                           the first tasks in file order
#   GBENCH_SHARD            "I/N" round-robin shard over the DETERMINISTIC sorted full task-id list
#                           (ids[I-1::N], 1-indexed); applied BEFORE MCP_TASK_LIMIT, matching the
#                           native path (base.run_eval_suite: shard then limit)
#   MCP_ENABLE_THINKING     true|false: the chat template's thinking switch for the model under
#                           test; when set, the request settings below are applied and counted
#   MCP_MAX_TOKENS          --thinking only: the run's budget, sent in place of upstream's caps
#   MCP_REQUEST_TIMEOUT_S   --thinking only: each request's timeout (seconds to read the reply),
#                           sized by gbench to the budget
#   MCP_TEMPERATURE         the operator's temperature (upstream sends none)
#   MCP_REQUEST_LOG         where the reply counts are appended (default /out/model_requests.json)
#   MCP_ASSUME_NETWORK      if set, the network check of step 2 is skipped (gbench's own name
#                           check before the container starts still runs)
#   MCP_CREDENTIALED_RELAY  <host>:<port> of gbench's relay to the credentialed-servers container
#   MCP_CREDENTIALED_TOKEN  the relay's per-run token (not a credential of any service)

import asyncio
import hashlib
import json
import os
import random
import re
import socket
import sys
import threading
import time

MCPBENCH_DIR = os.environ.get("MCPBENCH_DIR", "/app/mcp-bench")
ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MCPBENCH_DIR)
sys.path.insert(0, ADAPTER_DIR)

# --- server provisioning model (names match mcp_servers/commands.json + task group `servers`) ---
_OFFLINE = {"Bibliomantic", "Math MCP", "Medical Calculator", "Scientific Computing",
            "Time MCP", "Unit Converter"}
#: The keys MCP-Bench's README requires (mcp_bench_credentialed.CREDENTIAL_KEYS, gbench's
#: mcp_bench._SERVER_KEYS). A server whose commands.json `env` lists one runs in gbench's
#: credentialed-servers container, reached through the bridge; none of them is ever set here.
CREDENTIAL_KEYS = ("GOOGLE_MAPS_API_KEY", "NCI_API_KEY", "HF_TOKEN", "NPS_API_KEY", "NASA_API_KEY")
#: The servers that need one of CREDENTIAL_KEYS (commands.json at the pinned commit): available
#: only through the bridge, with network egress.
CREDENTIALED_SERVERS = {"Google Maps": "GOOGLE_MAPS_API_KEY", "BioMCP": "NCI_API_KEY",
                        "Hugging Face": "HF_TOKEN", "National Parks": "NPS_API_KEY",
                        "NASA Data": "NASA_API_KEY"}
#: Where the launcher writes the relay's address and token for the bridge, and where the bridge
#: records the sessions that failed outside the model (mcp_bench_bridge.BRIDGE_CONFIG /
#: BRIDGE_FAILURES).
BRIDGE_CONFIG = "/tmp/gbench-mcp-bench-credentialed.json"
BRIDGE_FAILURES = "/tmp/gbench-mcp-bench-bridge-failures.jsonl"
BRIDGE_SCRIPT = os.path.join(ADAPTER_DIR, "mcp_bench_bridge.py")
#: The bridges installed by install_credentialed_bridges ({server name: slug}), empty until then.
BRIDGED = {}


def slug(name):
    """mcp_bench_credentialed.slug: "Google Maps" -> "google-maps"."""
    out, dash = [], False
    for c in str(name).lower():
        if c.isalnum():
            out.append(c)
            dash = False
        elif not dash and out:
            out.append("-")
            dash = True
    return "".join(out).strip("-")


def bridged_command(name, spec):
    """The commands.json entry that starts the bridge to credentialed server `name` in place of
    the server: same cwd and transport (an http server keeps its port and endpoint), no `env`."""
    entry = {k: v for k, v in spec.items() if k not in ("cmd", "env")}
    if spec.get("transport") == "http":
        port = spec.get("port") or 3001
        entry["cmd"] = f"python {BRIDGE_SCRIPT} http {slug(name)} --port {port}"
    else:
        entry["cmd"] = f"python {BRIDGE_SCRIPT} stdio {slug(name)}"
    entry["env"] = []
    return entry


def install_credentialed_bridges(mcpbench_dir=None, environ=None):
    """Point the credentialed servers' commands.json entries at the bridge and write the bridge's
    configuration, when gbench passed its relay (MCP_CREDENTIALED_RELAY, MCP_CREDENTIALED_TOKEN).
    The upstream table is kept as commands.upstream.json (written once). Returns {name: slug} of
    the bridged servers ({} without a relay: those servers then cannot serve here)."""
    environ = os.environ if environ is None else environ
    relay = (environ.get("MCP_CREDENTIALED_RELAY") or "").strip()
    token = (environ.get("MCP_CREDENTIALED_TOKEN") or "").strip()
    BRIDGED.clear()
    if not relay or not token:
        return {}
    base = os.path.join(mcpbench_dir or MCPBENCH_DIR, "mcp_servers")
    live, pristine = os.path.join(base, "commands.json"), os.path.join(base, "commands.upstream.json")
    if not os.path.isfile(pristine):
        with open(live, encoding="utf-8") as fh:
            upstream = fh.read()
        with open(pristine, "w", encoding="utf-8") as fh:
            fh.write(upstream)
    with open(pristine, encoding="utf-8") as fh:
        table = json.load(fh)
    for name, spec in table.items():
        if isinstance(spec, dict) and any(k in CREDENTIAL_KEYS for k in spec.get("env") or []):
            table[name] = bridged_command(name, spec)
            BRIDGED[name] = slug(name)
    with open(live, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)
    fd = os.open(BRIDGE_CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"relay": relay, "token": token}, fh)
    return dict(BRIDGED)


def read_bridge_failures(path=BRIDGE_FAILURES):
    """The bridge's failed sessions: [{t, server, op, why}] ([] when there is none)."""
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    rec["t"] = float(rec["t"])
                except (ValueError, KeyError, TypeError):
                    continue
                out.append(rec)
    except OSError:
        pass
    return out


#: The host name checked when MCP_DNS_NAMES is unset (gbench always sets it).
DEFAULT_DNS_NAMES = ("api.github.com",)
#: How long the name lookups of one check may take together, in seconds.
DNS_LOOKUP_TIMEOUT_S = 15
#: The seconds to open the egress test connection.
EGRESS_CONNECT_TIMEOUT_S = 5
#: The last network_check() result (the pre-run check reports it).
LAST_NETWORK_CHECK = {}


def dns_names():
    """The host names the servers' APIs are reached at (MCP_DNS_NAMES, comma-separated)."""
    names = [n.strip() for n in os.environ.get("MCP_DNS_NAMES", "").split(",") if n.strip()]
    return names or list(DEFAULT_DNS_NAMES)


def resolve_all(names, timeout_s=DNS_LOOKUP_TIMEOUT_S):
    """{name: None when it resolved, else why not} for every name, looked up at the same time,
    all within `timeout_s` (a lookup still running then is "no answer")."""
    found = {}

    def look(name):
        try:
            socket.getaddrinfo(name, 443, 0, socket.SOCK_STREAM)
            found[name] = None
        except socket.gaierror as e:
            found[name] = "gaierror:" + str(e.errno)
        except Exception as e:  # noqa: BLE001 - reported as the lookup's failure
            found[name] = type(e).__name__
    workers = [threading.Thread(target=look, args=(n,), daemon=True) for n in names]
    for w in workers:
        w.start()
    end = time.monotonic() + timeout_s
    for w in workers:
        w.join(max(0.0, end - time.monotonic()))
    return {n: (found[n] if n in found else "no answer within %d s" % timeout_s) for n in names}


def network_check():
    """{up, assumed, names, unresolved, egress}: every name of dns_names() resolves and a TCP
    connection to one of them (port 443) opens. An IP-only test would pass on a network whose
    name resolution is down, where the servers' API calls fail, so there is none. With
    MCP_ASSUME_NETWORK set nothing is checked here (gbench's name check before the container
    starts still runs)."""
    names = dns_names()
    if os.environ.get("MCP_ASSUME_NETWORK"):
        result = {"up": True, "assumed": True, "names": len(names), "unresolved": {},
                  "egress": None}
    else:
        unresolved = {n: why for n, why in resolve_all(names).items() if why}
        egress = False
        if not unresolved:
            for name in names:
                try:
                    socket.create_connection((name, 443), timeout=EGRESS_CONNECT_TIMEOUT_S).close()
                    egress = True
                    break
                except Exception:  # noqa: BLE001 - the next name is tried
                    continue
        result = {"up": not unresolved and egress, "assumed": False, "names": len(names),
                  "unresolved": dict(sorted(unresolved.items())), "egress": egress}
    LAST_NETWORK_CHECK.clear()
    LAST_NETWORK_CHECK.update(result)
    return result


def _network_up() -> bool:
    return network_check()["up"]


def _available_servers(all_servers):
    """(servers that can serve here, network up): the offline servers always; with the network
    up, every online server, a credentialed one only when its bridge is installed
    (install_credentialed_bridges: gbench passed its relay to the credentialed-servers
    container)."""
    net = _network_up()
    avail = set()
    for s in all_servers:
        if s in _OFFLINE:
            avail.add(s)
        elif not net:
            continue
        elif s in CREDENTIALED_SERVERS:
            if s in BRIDGED:
                avail.add(s)
        else:  # network server
            avail.add(s)
    return avail, net


def _built_servers(mcpbench_dir):
    """Servers whose entrypoint artifact (and, for `uv run` servers, .venv) actually exists in the
    image = the ones that BUILT and can be spawned. Returns None if commands.json is unreadable
    (then no build-based check is applied). A server that did not build is reported and no task
    runs, so its tasks are never scored as model failures. The upstream table
    (commands.upstream.json, kept by install_credentialed_bridges) is read when it exists: a
    credentialed server runs from the same image in gbench's credentialed-servers container."""
    cj_path = os.path.join(mcpbench_dir, "mcp_servers", "commands.upstream.json")
    if not os.path.isfile(cj_path):
        cj_path = os.path.join(mcpbench_dir, "mcp_servers", "commands.json")
    try:
        with open(cj_path, encoding="utf-8") as f:
            cj = json.load(f)
    except Exception:
        return None
    base = os.path.join(mcpbench_dir, "mcp_servers")
    built = set()
    for name, spec in cj.items():
        cwd = (spec.get("cwd") or "").replace("../", "")
        d = os.path.join(base, cwd)
        if not os.path.isdir(d):
            continue
        cmd = spec.get("cmd", "")
        art = next((p for p in cmd.split() if p.endswith(".js") or p.endswith(".py")), None)
        # explicit-file entrypoints must exist; `python -m`/`uv run <pkg>` invocations have no
        # file to check (their deps live in system python / the uv venv) so are assumed present.
        art_ok = os.path.isfile(os.path.join(d, art)) if art else True
        venv_ok = os.path.isdir(os.path.join(d, ".venv")) if cmd.startswith("uv run") else True
        if art_ok and venv_ok:
            built.add(name)
    return built


def _all_task_servers(task_files):
    servers = set()
    for path in task_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for g in data.get("server_tasks", []):
            servers.update(g.get("servers") or [])
    return servers


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


def _sharded_task_ids(task_files, shard):
    """The set of task_ids selected for `shard` = (i, n), via round-robin over the DETERMINISTIC
    SORTED full task-id list across ALL task files: ids[i-1::n] (1-indexed, non-overlapping; the N
    shards union back to the full set, matching sampling.shard_select). Returns None when `shard`
    is None (keep every task). Each mcp_bench task carries a stable, globally-unique task_id, so an
    interleaved subset is exactly expressible."""
    if shard is None:
        return None
    ids = []
    for path in task_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for g in data.get("server_tasks", []):
            for t in g.get("tasks", []):
                tid = t.get("task_id")
                if tid is not None:
                    ids.append(tid)
    ids = sorted(set(ids))
    i, n = shard
    return set(ids[i - 1::n])


#: The seed of the --eval-limit draw (the suite name, as gbench's sampling.stratified_sample uses).
LIMIT_SEED = "mcp_bench"
#: How a limited run selects its tasks; the manifest's limit_draw (gbench checks it).
LIMIT_DRAW = f"stratified_indices(key=server group, seed={LIMIT_SEED})"


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


def _limited_task_ids(task_files, keep_ids, budget):
    """The task_ids a run with MCP_TASK_LIMIT `budget` keeps: of the tasks in scope (`keep_ids`,
    the shard; every task when None), sorted by task_id, a seeded draw of `budget` balanced across
    the server groups (the group's servers, sorted and joined by "+"): LIMIT_DRAW."""
    seen = {}
    for path in task_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for g in data.get("server_tasks", []):
            key = "+".join(sorted(g.get("servers") or []))
            for t in g.get("tasks", []):
                tid = t.get("task_id")
                if tid is not None and (keep_ids is None or tid in keep_ids):
                    seen.setdefault(tid, key)
    ids = sorted(seen)
    return {ids[k] for k in stratified_indices([seen[t] for t in ids], budget, LIMIT_SEED)}


def _filter_file(path, allowed, out_dir, budget=None, keep_ids=None):
    """Rewrite `path` down to the tasks that will actually run; return
    (out_path, kept, dropped, remaining_budget).

    Composition matches the native path (base.run_eval_suite: shard FIRST, then the limit):
      * SHARD: if `keep_ids` is not None, only tasks whose `task_id` is in it are in scope for this
        shard. Tasks outside the shard belong to another shard and are neither kept nor counted as
        dropped (so the denominator reflects only the shard).
      * PROVISIONING: a group is runnable only if every server it needs is in `allowed`; an
        in-shard task of any other group is dropped. main() passes every server of the task set
        here, because it runs no task at all when one of them cannot serve.
      * LIMIT: main() applies MCP_TASK_LIMIT before this, as the task ids `keep_ids` holds (a
        seeded draw within the shard, _limited_task_ids); a `budget` given here still caps at
        `budget` tasks in file order and counts the in-shard, provisioned tasks past it as dropped.
    A dropped task counts against the denominator (reported in the manifest) - never a silent skip."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    kept_groups, kept, dropped = [], 0, 0
    for g in data.get("server_tasks", []):
        need = set(g.get("servers") or [])
        # SHARD first: restrict to this shard's tasks (all tasks when not sharding).
        in_shard = [t for t in g.get("tasks", [])
                    if keep_ids is None or t.get("task_id") in keep_ids]
        if not in_shard:
            continue
        if not need.issubset(allowed):
            dropped += len(in_shard)          # in-shard but unprovisioned -> dropped (counts as 0)
            continue
        sel = []
        for t in in_shard:
            if budget is None or budget > 0:
                sel.append(t)
                if budget is not None:
                    budget -= 1
            else:
                dropped += 1                  # in-shard, provisioned, past the limit
        if sel:
            g = dict(g)                       # copy before slicing so the loaded data is untouched
            g["tasks"] = sel
            kept_groups.append(g)
            kept += len(sel)
    data["server_tasks"] = kept_groups
    data["total_tasks"] = kept
    out_path = os.path.join(out_dir, "filtered_" + os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return out_path, kept, dropped, budget


# --- per-task harness errors ------------------------------------------------------------------
#: The version of MCP_HARNESS_LOG's layout (gbench refuses another). Version 2 adds each task's
#: judge_models and the credentialed_relay kind.
HARNESS_LOG_VERSION = 2
#: How often the DNS watchdog looks up MCP_DNS_NAMES while the tasks run, in seconds.
DNS_WATCH_INTERVAL_S = 30
#: A resolver that did not answer (glibc's EAI_AGAIN, node's EAI_AGAIN, dig/host's "no servers
#: could be reached"): a harness error only when the text names one of MCP_DNS_NAMES or the
#: container's resolver (a name the model chose can fail on its own, e.g. SERVFAIL maps to
#: EAI_AGAIN, and `dig @<server the model chose>` can find no server).
RESOLVER_DOWN = re.compile(r"EAI_AGAIN|Temporary failure in name resolution|"
                           r"no servers could be reached|\[Errno -3\]", re.I)
#: A name the resolver reported as not found: a harness error only when the text names one of
#: MCP_DNS_NAMES (the servers' API hosts exist; a name the model chose may not).
NAME_NOT_FOUND = re.compile(r"ENOTFOUND|Name or service not known|nodename nor servname|"
                            r"No address associated with hostname|Could not resolve host|"
                            r"\[Errno -2\]|\[Errno -5\]|NameResolutionError|Failed to resolve",
                            re.I)
#: The most of one text scanned for a failed name lookup.
_SCAN_CHARS = 20000
#: The harness-error kinds of a task (MCP_HARNESS_LOG `tasks[].kinds`).
HARNESS_KINDS = ("model_endpoint", "judge_outage", "dns_failure", "dns_outage",
                 "credentialed_relay", "task_failed", "task_raised")


def resolver_addresses(path="/etc/resolv.conf"):
    """The nameserver addresses of the container's resolver configuration ([] when unreadable)."""
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "nameserver":
                    out.append(parts[1])
    except OSError:
        pass
    return out


class HarnessAccounting:
    """Which tasks a harness error fell in, so that upstream's averages leave them out.

    Upstream runs the tasks one after another (BenchmarkRunner._run_single_file_benchmark_core),
    so a task's window (begin() to end()) is the time its own agent loop and judge ran. A task
    is flagged with:
      * model_endpoint: the last attempt of a get_completion call of the model under test ended
        in an error mcp_bench_request_policy counted as a failed request (an endpoint or
        connection error upstream's retries did not get past); a context-window rejection left to
        upstream's handling, or a call whose last attempt got an empty reply, is the model's
        outcome and flags nothing, whatever an earlier attempt or a concurrent call did;
      * judge_outage: the task completed but upstream's evaluator returned no evaluation (every
        judge stability run failed); upstream's aggregator skips such a result, so the task
        would silently drop out of the averages: it is counted and named as a harness error;
      * dns_failure: an error of the task (the task's error, or a tool call's error reply: not a
        successful tool result, whose text may quote anything) shows a resolver that did not
        answer naming one of MCP_DNS_NAMES or the container's resolver (RESOLVER_DOWN), or a
        failed lookup of one of MCP_DNS_NAMES (NAME_NOT_FOUND);
      * dns_outage: the watchdog found at least half of MCP_DNS_NAMES unresolvable while the task
        ran;
      * credentialed_relay: a session to a credentialed server failed outside the model while
        the task ran (the bridge's BRIDGE_FAILURES: not opened, its server exited before it
        answered, or the session was lost or ended by a docker exec failure while in use);
      * task_failed: upstream recorded the task as failed (its own averages leave it out);
      * task_raised: the task's execution raised.
    scored() is what upstream's aggregator gets: the results of the tasks with no flag. Each
    task also records the Gemini models that answered its judge calls (judge_models)."""

    def __init__(self, names=(), watch_interval_s=DNS_WATCH_INTERVAL_S, resolvers=None,
                 bridge_failures=BRIDGE_FAILURES):
        self.names = list(names)
        self.resolvers = list(resolver_addresses() if resolvers is None else resolvers)
        self.bridge_failures = bridge_failures
        self.watch_interval_s = watch_interval_s
        self._lock = threading.Lock()
        self._round = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.tasks = []
        self.current = None
        self.dns_rounds = 0
        self.dns_outages = []
        self.dns_name_failures = {}

    # -- tasks --------------------------------------------------------------------------------- #
    def begin(self, task_id, task_file=None):
        rec = {"task_id": task_id, "file": os.path.basename(task_file) if task_file else None,
               "start": time.time(), "end": None, "status": None, "kinds": [], "details": [],
               "judge_calls_failed": 0, "judge_models": {}}
        with self._lock:
            self.tasks.append(rec)
            self.current = rec
        return rec

    def flag(self, kind, detail="", rec=None):
        """Flag `rec` (the running task by default) with `kind`; False when no task runs."""
        with self._lock:
            rec = rec if rec is not None else self.current
            if rec is None:
                return False
            if kind not in rec["kinds"]:
                rec["kinds"].append(kind)
            if detail and len(rec["details"]) < 8:
                rec["details"].append((kind + ": " + str(detail))[:300])
            return True

    def judge_call_failed(self):
        """A judge call of the running task failed on every model of the cascade (the stability
        run it belonged to is skipped by upstream's evaluator)."""
        with self._lock:
            if self.current is not None:
                self.current["judge_calls_failed"] += 1

    def judge_answered(self, model):
        """A judge call of the running task got a reply from `model`."""
        with self._lock:
            if self.current is not None:
                models = self.current["judge_models"]
                models[str(model)] = models.get(str(model), 0) + 1

    def dns_kind(self, text):
        """True when `text` shows a failed name lookup that is a harness error: a resolver that did
        not answer, naming one of the servers' API host names or the container's resolver, or a
        server API host name not found."""
        text = str(text or "")[:_SCAN_CHARS]
        named = any(n in text for n in self.names)
        if RESOLVER_DOWN.search(text) and (named or any(r in text for r in self.resolvers)):
            return True
        return bool(NAME_NOT_FOUND.search(text) and named)

    @staticmethod
    def _texts(result):
        """The error texts of upstream's result for a task: its error and each tool call's error
        reply (never a successful tool result)."""
        yield result.get("error")
        for step in result.get("execution_results") or []:
            if isinstance(step, dict):
                yield step.get("error")

    def end(self, rec, result):
        """Close `rec` with upstream's result for it (None when the execution raised)."""
        self.settle()
        with self._lock:
            rec["end"] = time.time()
            if self.current is rec:
                self.current = None
        if not isinstance(result, dict):
            rec["status"] = "raised"
            self.flag("task_raised", "the task's execution raised", rec)
            return
        rec["status"] = result.get("status")
        if result.get("task_id") is not None:
            rec["task_id"] = result["task_id"]
        if rec["status"] == "completed" and result.get("evaluation") is None:
            self.flag("judge_outage", "upstream's evaluator returned no evaluation", rec)
        elif rec["status"] != "completed":
            self.flag("task_failed", result.get("error") or rec["status"], rec)
        for text in self._texts(result):
            if text and self.dns_kind(text):
                self.flag("dns_failure", " ".join(str(text).split())[:200], rec)
                break
        self._bridge_failures_in(rec)

    def _bridge_failures_in(self, rec):
        """Flag `rec` credentialed_relay when a bridge session failed while it ran."""
        start, end = rec["start"], rec["end"] or time.time()
        for f in read_bridge_failures(self.bridge_failures):
            if start <= f["t"] <= end:
                self.flag("credentialed_relay", f"{f.get('server')} {f.get('op')}: "
                                                f"{f.get('why')}", rec)

    def flagged_ids(self):
        with self._lock:
            return {t["task_id"] for t in self.tasks if t["kinds"]}

    def scored(self, results):
        """`results` without the tasks a harness error fell in."""
        flagged = self.flagged_ids()
        return [r for r in results if not (isinstance(r, dict) and r.get("task_id") in flagged)]

    # -- the DNS watchdog ---------------------------------------------------------------------- #
    def watch_once(self):
        """One watchdog round: look up every name; a round in which at least half of them fail
        is a DNS outage, flagged on the task running then."""
        with self._round:
            failed = sorted(n for n, why in resolve_all(self.names).items() if why)
            with self._lock:
                self.dns_rounds += 1
                for n in failed:
                    self.dns_name_failures[n] = self.dns_name_failures.get(n, 0) + 1
            if failed and 2 * len(failed) >= len(self.names):
                with self._lock:
                    self.dns_outages.append({"t": round(time.time(), 1),
                                             "failed": len(failed), "of": len(self.names),
                                             "names": failed[:10]})
                self.flag("dns_outage", "%d of %d names did not resolve" % (len(failed),
                                                                           len(self.names)))

    def settle(self):
        """Wait for a watchdog round in flight, so a task's flags are final when it ends."""
        with self._round:
            pass

    def _watch(self):
        while not self._stop.wait(self.watch_interval_s):
            try:
                self.watch_once()
            except Exception as e:  # noqa: BLE001 - a round that could not run is recorded
                with self._lock:
                    self.dns_outages.append({"t": round(time.time(), 1),
                                             "error": type(e).__name__})
                self.flag("dns_outage", "the watchdog round failed (%s)" % type(e).__name__)

    def start(self):
        if self.names and self._thread is None:
            self._thread = threading.Thread(target=self._watch, name="mcp-bench-dns-watch",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=DNS_LOOKUP_TIMEOUT_S + 5)

    # -- record -------------------------------------------------------------------------------- #
    def record(self):
        with self._lock:
            models = {}
            for t in self.tasks:
                for m, n in t["judge_models"].items():
                    models[m] = models.get(m, 0) + n
            return {"version": HARNESS_LOG_VERSION, "dns_names": list(self.names),
                    "dns_rounds": self.dns_rounds, "dns_outages": list(self.dns_outages),
                    "dns_name_failures": dict(sorted(self.dns_name_failures.items())),
                    "bridged_servers": dict(BRIDGED),
                    "judge_models": dict(sorted(models.items())),
                    "tasks": [{k: t[k] for k in ("task_id", "file", "status", "kinds", "details",
                                                 "judge_calls_failed", "judge_models")}
                              for t in self.tasks]}

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.record(), f, indent=2)


def install_harness_accounting(runner_cls, aggregator_cls, judge_cls, harness):
    """Hook `harness` into upstream: each BenchmarkRunner.execute_single_task_with_model call is
    a task window, ResultsAggregator.aggregate_current_metrics (the metrics upstream writes)
    gets harness.scored(results), and the judge reports its failed calls."""
    original_execute = runner_cls.execute_single_task_with_model

    async def execute_single_task_with_model(self, task_info, *args, **kwargs):
        task = (task_info or {}).get("task") if isinstance(task_info, dict) else None
        rec = harness.begin((task or {}).get("task_id", "unknown"),
                            getattr(self, "tasks_file", None))
        result = None
        try:
            result = await original_execute(self, task_info, *args, **kwargs)
            return result
        finally:
            harness.end(rec, result)

    original_aggregate = aggregator_cls.aggregate_current_metrics

    def aggregate_current_metrics(self, results):
        return original_aggregate(self, harness.scored(results))

    runner_cls.execute_single_task_with_model = execute_single_task_with_model
    aggregator_cls.aggregate_current_metrics = aggregate_current_metrics
    judge_cls.outage_hook = harness.judge_call_failed
    judge_cls.answer_hook = harness.judge_answered
    return harness


def install_request_policy(harness=None):
    """Apply the run's model request settings (mcp_bench_request_policy.RequestPolicy) to the
    requests of the model under test: the provider upstream's LLMFactory creates for the
    MCP_MODEL_CONFIG_NAME slot. The judge (CascadeGeminiJudge) and any other provider are left
    alone. With `harness` (HarnessAccounting), a get_completion call of that provider whose last
    attempt ended in an error the policy counted as a failed request (an endpoint or connection
    error) flags the running task (model_endpoint); the attribution is per call (a context
    variable), so a concurrent call or an earlier attempt of the same call decides nothing.
    Returns the policy, or None when MCP_ENABLE_THINKING is unset (nothing is applied)."""
    if os.environ.get("MCP_ENABLE_THINKING") is None:
        return None
    import contextvars
    from llm.factory import LLMFactory
    from mcp_bench_request_policy import ENDPOINT_FAILURE_ATTR, RequestPolicy
    last_attempt = contextvars.ContextVar("mcp_bench_last_attempt", default=None)

    os.environ.setdefault("MCP_REQUEST_LOG", "/out/model_requests.json")
    policy = RequestPolicy("MCP_", base_url=os.environ.get("LLAMA_3_1_8B_BASE_URL")
                           or os.environ.get("MCP_ENDPOINT"))
    slot = os.environ.get("MCP_MODEL_CONFIG_NAME", "llama-3-1-8b")
    original = LLMFactory.create_llm_provider

    async def create_llm_provider(model_config):
        provider = await original(model_config)
        if getattr(model_config, "name", None) == slot:
            completions = provider.client.chat.completions
            create = completions.create
            api_key = getattr(provider.client, "api_key", None)

            async def create_with_policy(**kwargs):
                holder = last_attempt.get()
                try:
                    response = await policy.acall(create, kwargs, api_key=api_key)
                except BaseException as exc:
                    if holder is not None:
                        holder["endpoint_error"] = bool(getattr(exc, ENDPOINT_FAILURE_ATTR,
                                                                False))
                    raise
                if holder is not None:
                    holder["endpoint_error"] = False
                return response
            completions.create = create_with_policy
            get_completion = getattr(provider, "get_completion", None)
            if harness is not None and get_completion is not None:
                async def get_completion_counted(*args, **kwargs):
                    holder = {"endpoint_error": False}
                    token = last_attempt.set(holder)
                    try:
                        return await get_completion(*args, **kwargs)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if holder["endpoint_error"]:
                            harness.flag("model_endpoint", type(exc).__name__)
                        raise
                    finally:
                        last_attempt.reset(token)
                provider.get_completion = get_completion_counted
        return provider

    LLMFactory.create_llm_provider = staticmethod(create_llm_provider)
    return policy.install_dump_at_exit()


def main():
    import benchmark.runner as R
    from mcp_bench_cascade_judge import CascadeGeminiJudge

    # 0) the credentialed servers run in gbench's credentialed-servers container: point their
    #    commands.json entries at the bridge (before upstream reads the table)
    install_credentialed_bridges()

    # 1) inject the cascade judge (fires even on the CLI path; neutralizes the Azure default)
    _orig_init = R.BenchmarkRunner.__init__

    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        if getattr(self, "_judge_provider", None) is None:
            self._judge_provider = CascadeGeminiJudge()
    R.BenchmarkRunner.__init__ = _patched_init

    out = os.environ.get("MCP_OUTPUT", "/out/mcp_bench_results.json")
    manifest_path = os.environ.get("MCP_MANIFEST", "/out/subset_manifest.json")
    harness_path = os.environ.get("MCP_HARNESS_LOG",
                                  os.path.join(os.path.dirname(out) or "/out", "harness_log.json"))
    out_dir = os.path.dirname(out) or "/out"
    harness = HarnessAccounting(dns_names())

    default_tasks = ",".join(os.path.join(MCPBENCH_DIR, "tasks", f) for f in (
        "mcpbench_tasks_single_runner_format.json",
        "mcpbench_tasks_multi_2server_runner_format.json",
        "mcpbench_tasks_multi_3server_runner_format.json"))
    task_files = [t.strip() for t in os.environ.get("MCP_TASK_FILES", default_tasks).split(",") if t.strip()]

    # 2) Every server of the task set must be able to serve: built in the image, its key set when
    #    it needs one, network for the online servers (every MCP_DNS_NAMES name resolves and a
    #    connection opens). If any cannot, no task runs (the manifest names them and gbench reports
    #    infra_required); the tasks of the servers that can serve are never run alone.
    all_servers = _all_task_servers(task_files)
    allowed, net = _available_servers(all_servers)
    built = _built_servers(MCPBENCH_DIR)
    unbuilt = sorted(all_servers - built) if built is not None else []
    if built is not None:
        allowed = allowed & built
    unavailable = sorted(all_servers - allowed)
    lim = os.environ.get("MCP_TASK_LIMIT")
    budget = int(lim) if (lim and int(lim) > 0) else None

    # 3) SHARD (round-robin over the DETERMINISTIC sorted full task-id list) FIRST, then the
    #    MCP_TASK_LIMIT budget - matching the native path's compose order. keep_ids is None
    #    (keep every task) unless a shard is active.
    shard = _shard()
    shard_manifest = {"i": shard[0], "n": shard[1]} if shard else None
    manifest = {
        "network_up": net,
        "network_check": dict(LAST_NETWORK_CHECK),
        "available_servers": sorted(allowed),
        "unavailable_servers": unavailable,
        "unbuilt_servers": unbuilt,
        "task_limit": budget,
        "limit_draw": LIMIT_DRAW if budget is not None else None,
        "shard": shard_manifest,
    }
    run_files, per_file = [], {}
    total_kept = total_dropped = 0
    if unavailable:
        unresolved = sorted(LAST_NETWORK_CHECK.get("unresolved") or {})
        print("mcp_bench: servers that cannot serve here: " + ", ".join(unavailable)
              + " (network_up=" + str(net)
              + ("; names that did not resolve: " + ", ".join(unresolved) if unresolved else "")
              + "); no task is run.", file=sys.stderr)
    elif shard is None and budget is None:
        # The whole task set: upstream's task files as they are.
        run_files = list(task_files)
        for path in task_files:
            with open(path, encoding="utf-8") as f:
                n = sum(len(g.get("tasks") or []) for g in json.load(f).get("server_tasks", []))
            per_file[os.path.basename(path)] = {"kept": n, "dropped": 0}
            total_kept += n
    else:
        if shard:
            print(f"mcp_bench: shard {shard[0]}/{shard[1]} over the sorted full task-id list",
                  file=sys.stderr)
        keep_ids = _sharded_task_ids(task_files, shard)
        if budget is not None:
            # the limit within the shard: a seeded draw balanced across the server groups, not
            # the first tasks in file order
            keep_ids = _limited_task_ids(task_files, keep_ids, budget)
            manifest["selected_task_ids"] = sorted(keep_ids)
            budget = None
        for path in task_files:
            fp, kept, dropped, budget = _filter_file(path, all_servers, out_dir, budget, keep_ids)
            per_file[os.path.basename(path)] = {"kept": kept, "dropped": dropped}
            total_kept += kept
            total_dropped += dropped
            if kept:
                run_files.append(fp)
    manifest.update({"per_file": per_file, "tasks_kept": total_kept,
                     "tasks_dropped": total_dropped})
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not run_files:
        if not unavailable:
            print("mcp_bench: the shard / task limit selected no task.", file=sys.stderr)
        # an empty output, so the harness reports the manifest's reason, never a score
        with open(out, "w", encoding="utf-8") as f:
            json.dump({}, f)
        harness.dump(harness_path)
        return

    # 4) build argv (model-under-test env is set by the entrypoint) + delegate to upstream main()
    argv = ["run_benchmark.py",
            "--models", os.environ.get("MCP_MODEL_CONFIG_NAME", "llama-3-1-8b"),
            "--tasks-file", ",".join(run_files),
            "--output", out]
    dc = os.environ.get("MCP_DISTRACTION_COUNT")
    if dc is not None and dc != "":
        argv += ["--distraction-count", str(dc)]
    if os.environ.get("MCP_DISABLE_STABILITY"):
        argv += ["--disable-judge-stability"]
    sys.argv = argv

    import benchmark.results_aggregator as RA
    install_harness_accounting(R.BenchmarkRunner, RA.ResultsAggregator, CascadeGeminiJudge,
                               harness)
    policy = install_request_policy(harness)
    harness.start()
    try:
        asyncio.run(R.main())
    finally:
        harness.stop()
        harness.dump(harness_path)
        if policy is not None:
            policy.dump()


if __name__ == "__main__":
    main()
