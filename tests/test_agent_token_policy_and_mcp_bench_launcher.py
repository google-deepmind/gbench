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

"""mcp_atlas, cimemories, mcp_bench and toolbench: the token-limit policy, the mcp_bench image
and launcher, credential cuts, and the toolbench grader and simulator records.

  * mcp_atlas / cimemories - their harnesses send no max_tokens, so no request carries one, in
    either mode and whether or not the context window is known; --max-output-tokens is recorded
    as not applied.
  * mcp_bench probe/gaps   - a key is redacted before a server's error text is cut, so no part
    of it survives the cut.
  * mcp_bench launcher     - it runs no task when a server of the task set cannot serve, and runs
    upstream's task files as they are for the whole task set.
  * mcp_bench image        - Python 3.10 (the README's install env) is the only interpreter, with
    a configured timezone; the MCP client pin is the release upstream's range resolved to on the
    pinned commit's date; the install.sh drift guard fails the build.
  * toolbench              - ToolEval's grader config is the official one; the simulator record
    names its API base and does not call upstream's own model a substitution.
"""

import asyncio
import json
import os
import subprocess
import sys
import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import cimemories as CIM
from gbench.runners.eval_suites import mcp_atlas as ATLAS
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
if _DOCKER_DIR not in sys.path:
    sys.path.insert(0, _DOCKER_DIR)
import mcp_bench_run as RUN  # noqa: E402

NASA = "nasa+secret/KEY-0123456789abcdef"
GEMINI = "AIzaSyTESTgeminiKEY0000001111"


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


def _doc(name):
    return " ".join(_read("docs", "evals", name).split())


# --------------------------------------------------------------------------- #
# mcp_atlas / cimemories: no max_tokens, as their harnesses send none
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sampling", [ATLAS.MCP_ATLAS_SAMPLING, CIM.CIMEMORIES_SAMPLING])
@pytest.mark.parametrize("thinking", [False, True])
def test_token_limit_policy_sends_no_max_tokens_for_these_harnesses(sampling, thinking):
    plan = base.resolve_token_budget("x", sampling, thinking=thinking, max_output_tokens=4096)
    assert plan.policy == "not_sent" and plan.max_tokens is None and plan.uncapped
    assert plan.note.startswith("not sent: the official harness sends no max_tokens")
    assert plan.note.endswith("--max-output-tokens 4096 not applied")


def _run_suite(eval_name, sampling, *, thinking, tool_loop=None, attempts=1, tool_turns=0):
    """run_eval_suite with the suite's own settings and a fake transport; a known context window
    and --max-output-tokens 4096. Returns (requests seen, result)."""
    seen = []
    replies = {"n": 0}

    async def send(**k):
        async with k["semaphore"]:
            pass
        seen.append(k)
        replies["n"] += 1
        if replies["n"] <= tool_turns:
            call = {"id": f"c{replies['n']}", "type": "function",
                    "function": {"name": "calculator_add", "arguments": "{}"}}
            return base.Reply("", [call], "tool_calls", completion_tokens=3)
        return base.Reply("answer " * 5, None, "stop", completion_tokens=5)

    async def tool(name, args):
        return "3"

    base._RUN_KNOBS["max_output_tokens"] = 4096
    base._RUN_KNOBS["max_model_len"] = 131072
    samples = [([{"role": "user", "content": "q"}], "a", {"category": "c"})]
    with mock.patch.object(base, "_send_single_request", send):
        result = base.run_eval_suite(
            eval_name=eval_name, model_name="m", base_url="http://x", concurrency=1,
            samples=samples, eval_fn=lambda r, g: True, thinking=thinking,
            official_sampling=sampling, tool_loop=tool_loop,
            tool_executor=tool if tool_loop is not None else None,
            attempt_count=attempts, supports_attempts=attempts > 1)
    return seen, result


@pytest.mark.parametrize("thinking", [False, True])
def test_mcp_atlas_sends_no_max_tokens_on_any_turn(thinking):
    seen, res = _run_suite("mcp_atlas", ATLAS.MCP_ATLAS_SAMPLING, thinking=thinking,
                           tool_loop=ATLAS.MCP_ATLAS_TOOL_LOOP, tool_turns=2)
    assert len(seen) == 3                                   # two tool turns, then the answer
    assert [k["max_output_tokens"] for k in seen] == [None, None, None]
    assert all("max_tokens" not in (k["extra_payload"] or {}) for k in seen)
    # gbench's rule on every turn: 0.0 without --thinking, 1.0 with it (the harness sends none)
    assert all(k["temperature"] == (1.0 if thinking else 0.0) for k in seen)
    rec = res["official_sampling"]
    assert rec["temperature_matches_official"] is False
    assert rec["send_max_tokens"] is False and rec["max_tokens"].startswith("not sent")
    assert "--max-output-tokens 4096 not applied" in res["max_output_tokens_note"]


@pytest.mark.parametrize("thinking", [False, True])
def test_cimemories_sends_no_max_tokens_on_any_generation(thinking):
    seen, res = _run_suite("cimemories", CIM.CIMEMORIES_SAMPLING, thinking=thinking,
                           attempts=CIM._NUM_TRIALS)
    assert len(seen) == CIM._NUM_TRIALS
    assert all(k["max_output_tokens"] is None for k in seen)
    assert all("max_tokens" not in (k["extra_payload"] or {}) for k in seen)
    assert res["official_sampling"]["send_max_tokens"] is False
    assert "--max-output-tokens 4096 not applied" in res["max_output_tokens_note"]


def test_mcp_atlas_run_passes_its_harness_sampling(monkeypatch):
    got = {}
    monkeypatch.setattr(ATLAS, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(ATLAS, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(ATLAS, "_open_sandbox", lambda: ATLAS._Sandbox("http://sandbox"))
    monkeypatch.setattr(ATLAS, "_wait_ready", lambda sandbox, timeout_s: None)
    monkeypatch.setattr(ATLAS, "_preflight", lambda sandbox: {})
    monkeypatch.setattr(ATLAS, "_list_tools", lambda sandbox: [])
    monkeypatch.setattr(ATLAS, "_load_mcp_atlas_samples", lambda limit=None, catalog=None: [])
    monkeypatch.setattr(ATLAS, "_stop_container", lambda sandbox: None)

    def fake_suite(**kw):
        got.update(kw)
        return {"sample_traces": []}

    monkeypatch.setattr(ATLAS, "run_eval_suite", fake_suite)
    ATLAS.run_mcp_atlas("m", "http://x", max_output_tokens=4096)
    assert got["official_sampling"] is ATLAS.MCP_ATLAS_SAMPLING
    assert got["tool_loop"] is ATLAS.MCP_ATLAS_TOOL_LOOP


def test_no_stale_budget_comment_and_docs_say_the_field_is_never_sent():
    for mod in ("cimemories.py", "mcp_atlas.py"):
        src = _read("gbench", "runners", "eval_suites", mod)
        assert "base.py defaults when unset" not in src, mod
        assert "base.resolve_token_budget" in src, mod
    atlas = _doc("mcp_atlas.md")
    assert "the field is never sent, whether or not the window is known" in atlas
    assert "left out when the window is unknown" not in atlas
    cim = _doc("cimemories.md")
    assert "`timeout: 300` and `num_retries: 5`" in cim
    assert "never judges a failed request as a message the model sent" in cim


# --------------------------------------------------------------------------- #
# mcp_bench: no part of a key survives a cut
# --------------------------------------------------------------------------- #
def _no_key_part(text, key, n=6):
    return not any(key[i:i + n] in text for i in range(len(key) - n + 1))


def test_provisioning_gaps_redact_before_the_cut(monkeypatch):
    from urllib.parse import quote
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "x" * 8 + k.lower())
    monkeypatch.setenv("NASA_API_KEY", NASA)
    raw_key_error = "e" * 150 + NASA                        # the key crosses the 160 cut
    encoded_error = "e" * 150 + quote(NASA + "/", safe="")
    probe = {"servers": ["A", "B"], "unavailable": [], "unbuilt": [], "network_up": True,
             "connections": {"A": {"status": "failed", "error": raw_key_error},
                             "B": {"status": "failed", "error": encoded_error}}}
    gaps = " ".join(MB.provisioning_gaps(probe))
    assert "[REDACTED]" in gaps and _no_key_part(gaps, NASA)
    # env_keys (the caller's own key values) are redacted too
    monkeypatch.delenv("NASA_API_KEY")
    keys = {k: "x" * 8 + k.lower() for k in MB._REQUIRED_KEYS}
    keys["NASA_API_KEY"] = NASA
    gaps = " ".join(MB.provisioning_gaps(probe, env_keys=keys))
    assert "[REDACTED]" in gaps and _no_key_part(gaps, NASA)


def test_in_image_probe_redacts_every_forwarded_key_before_the_cut(tmp_path):
    """Run the probe script itself against a stand-in launcher and connection check. No server
    key is forwarded to the probe any more: the credentialed servers run in gbench's
    credentialed-servers container, which masks every key value in what they send back
    (tests/test_mcp_wildclawbench_credentials_and_cascade.py), and the probe reaches them
    through the bridge. So the probe writes an empty key file, installs the bridges first, cuts each server error at 300
    characters and reports the bridged servers and the sessions the relay did not open."""
    root = tmp_path / "mcpbench"
    (root / "mcp_servers").mkdir(parents=True)
    (root / "mcp_servers" / "api_key").write_text("STALE = value\n")
    (root / "utils").mkdir()
    (root / "utils" / "__init__.py").write_text("")
    (root / "utils" / "collect_mcp_info.py").write_text(
        "import json, os\n"
        "class MCPServerInfoCollector:\n"
        "    def __init__(self, mode):\n"
        "        self.mode = mode\n"
        "    async def collect_all_info(self):\n"
        "        assert 'NASA_API_KEY' not in os.environ\n"
        "        errs = {'raw': 'e' * 290 + '*' * 40, 'url': 'e' * 400, 'json': 'short'}\n"
        "        return {'servers': {n: {'connection_status': 'failed', 'tools': {}, 'error': e}\n"
        "                            for n, e in errs.items()}}\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "mcp_bench_run.py").write_text(
        f"MCPBENCH_DIR = {str(root)!r}\n"
        "LAST_NETWORK_CHECK = {'up': True}\n"
        "def install_credentialed_bridges():\n    return {'NASA Data': 'nasa-data'}\n"
        "def read_bridge_failures():\n"
        "    return [{'t': 1.0, 'server': 'nasa-data', 'op': 'stdio', 'why': 'refused'}]\n"
        "def _all_task_servers(files):\n    return {'raw', 'url', 'json'}\n"
        "def _available_servers(servers):\n    return set(servers), True\n"
        "def _built_servers(d):\n    return None\n")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(adapter),
           "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-c", MB._PROBE], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=60)
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("MCP_BENCH_PROBE "))
    report = json.loads(line[len("MCP_BENCH_PROBE "):])
    conn = report["connections"]
    assert set(conn) == {"raw", "url", "json"}
    for name, c in conn.items():
        assert len(c["error"]) <= 300, name
    assert report["bridged"] == ["NASA Data"]
    assert report["bridge_failures"] == [{"server": "nasa-data", "op": "stdio", "why": "refused"}]
    # the key file upstream's loader reads is left empty in the model container
    assert (root / "mcp_servers" / "api_key").read_text() == ""
    for key in MB._SERVER_KEYS:
        assert key not in MB._PROBE


# --------------------------------------------------------------------------- #
# mcp_bench launcher: no task runs without every server
# --------------------------------------------------------------------------- #
@pytest.fixture
def launcher(monkeypatch, tmp_path):
    """mcp_bench_run.main() against two task files, a stand-in upstream runner and judge."""
    tasks = []
    for name, groups in (("single.json", [(["Math MCP"], ["a", "b"]),
                                          (["National Parks"], ["c"])]),
                         ("multi.json", [(["Math MCP", "Time MCP"], ["d"])])):
        path = tmp_path / name
        path.write_text(json.dumps({"server_tasks": [
            {"servers": s, "tasks": [{"task_id": t} for t in ids]} for s, ids in groups]}))
        tasks.append(str(path))
    out = tmp_path / "out"
    out.mkdir()
    ran = []

    class _Runner:
        def __init__(self, *a, **k):
            self._judge_provider = None

        async def execute_single_task_with_model(self, task_info, *a, **k):
            return {}

    class _Aggregator:
        def aggregate_current_metrics(self, results):
            return {}

    async def _main():
        ran.append(list(sys.argv))

    runner_mod = types.ModuleType("benchmark.runner")
    runner_mod.BenchmarkRunner, runner_mod.main = _Runner, _main
    aggregator_mod = types.ModuleType("benchmark.results_aggregator")
    aggregator_mod.ResultsAggregator = _Aggregator
    bench_mod = types.ModuleType("benchmark")
    bench_mod.runner = runner_mod
    bench_mod.results_aggregator = aggregator_mod
    judge_mod = types.ModuleType("mcp_bench_cascade_judge")
    judge_mod.CascadeGeminiJudge = type("CascadeGeminiJudge", (), {})
    monkeypatch.setitem(sys.modules, "benchmark", bench_mod)
    monkeypatch.setitem(sys.modules, "benchmark.runner", runner_mod)
    monkeypatch.setitem(sys.modules, "benchmark.results_aggregator", aggregator_mod)
    monkeypatch.setitem(sys.modules, "mcp_bench_cascade_judge", judge_mod)
    monkeypatch.setattr(RUN, "MCPBENCH_DIR", str(tmp_path / "no_repo"))   # no commands.json
    monkeypatch.setattr(sys, "argv", ["x"])
    monkeypatch.setenv("MCP_TASK_FILES", ",".join(tasks))
    monkeypatch.setenv("MCP_OUTPUT", str(out / "res.json"))
    monkeypatch.setenv("MCP_MANIFEST", str(out / "manifest.json"))
    monkeypatch.setenv("MCP_ASSUME_NETWORK", "1")
    for k in ("MCP_TASK_LIMIT", "GBENCH_SHARD", "NPS_API_KEY", "MCP_NO_SUBSET_FILTER"):
        monkeypatch.delenv(k, raising=False)

    def go():
        RUN.main()
        return (ran, json.loads((out / "manifest.json").read_text()),
                json.loads((out / "res.json").read_text()) if (out / "res.json").exists()
                else None)
    return tasks, go


def test_launcher_runs_no_task_when_a_server_cannot_serve(launcher, monkeypatch):
    tasks, go = launcher
    monkeypatch.setenv("MCP_NO_SUBSET_FILTER", "1")          # no longer an escape hatch
    ran, manifest, res = go()
    assert ran == [] and res == {}
    assert manifest["unavailable_servers"] == ["National Parks"]
    assert manifest["tasks_kept"] == 0


def _bridged(monkeypatch, servers=("National Parks",)):
    """The launcher with gbench's relay to the credentialed-servers container: those servers'
    bridges are installed (a credentialed server serves only through its bridge)."""
    bridged = {s: RUN.slug(s) for s in servers}
    monkeypatch.setattr(RUN, "BRIDGED", dict(bridged))
    monkeypatch.setattr(RUN, "install_credentialed_bridges", lambda *a, **k: dict(bridged))


def test_launcher_runs_upstreams_task_files_as_they_are(launcher, monkeypatch):
    tasks, go = launcher
    _bridged(monkeypatch)
    ran, manifest, _ = go()
    [argv] = ran
    assert argv[argv.index("--tasks-file") + 1] == ",".join(tasks)
    assert manifest["unavailable_servers"] == [] and manifest["tasks_kept"] == 4
    assert manifest["per_file"] == {"single.json": {"kept": 3, "dropped": 0},
                                    "multi.json": {"kept": 1, "dropped": 0}}


def test_launcher_limit_selects_within_the_full_server_set(launcher, monkeypatch):
    tasks, go = launcher
    _bridged(monkeypatch)
    monkeypatch.setenv("MCP_TASK_LIMIT", "1")
    ran, manifest, _ = go()
    [argv] = ran
    files = argv[argv.index("--tasks-file") + 1].split(",")
    assert [os.path.basename(f) for f in files] == ["filtered_single.json"]
    assert manifest["tasks_kept"] == 1 and manifest["task_limit"] == 1


def test_run_reports_the_servers_the_launcher_could_not_serve(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "x" * 8 + k.lower())
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    connected = {f"S{i}": {"status": "success", "tools": 3} for i in range(28)}
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: {
        "servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
        "connections": connected})
    # the credentialed-servers container (tests/test_mcp_wildclawbench_credentials_and_cascade.py
    # covers it): a stand-in
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: types.SimpleNamespace(
                            client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                                                "MCP_CREDENTIALED_TOKEN": "t0k"},
                            stop=lambda: None, record=lambda: {}))

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                json.dump({}, fh)                            # the launcher ran no task
            with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                json.dump({"unavailable_servers": ["Reddit"], "network_up": True,
                           "tasks_kept": 0}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    # the judge's Gemini relay (tests/test_mcp_credential_isolation_and_dns.py covers it): a
    # stand-in, docker is stubbed
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48},
        stop=lambda: None, record=lambda: {}))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*could not serve.*Reddit"):
        MB.run_mcp_bench("m", "http://x/v1")


def test_launcher_and_entrypoint_headers_match_the_behaviour():
    entry = _read("gbench", "docker", "mcp_bench_entrypoint.sh")
    launch = _read("gbench", "docker", "mcp_bench_run.py")
    for text in (entry, launch):
        assert "MCP_NO_SUBSET_FILTER" not in text
        assert "canonical-when-provisioned" not in text
        assert "filters tasks to the provisioned servers" not in text
    assert "runs no task if one cannot" in entry
    assert "runs NO task" in launch
    src = _read("gbench", "runners", "eval_suites", "mcp_bench.py")
    assert "BEFORE the provisioning filter" not in src
    assert "checked again after the run" not in _doc("mcp_bench.md")


# --------------------------------------------------------------------------- #
# mcp_bench image: Python 3.10 and a drift guard that fails the build
# --------------------------------------------------------------------------- #
def test_mcp_bench_image_runs_the_readme_python():
    df = _read("gbench", "docker", "mcp_bench.Dockerfile")
    assert "python3.11" not in df and "deadsnakes" not in df
    assert "python3 python3-venv python3-dev python-is-python3" in df
    assert "assert sys.version_info[:2] == (3, 10)" in df
    # a configured local timezone, which Time MCP's server needs to start
    assert "gnupg sudo tzdata" in df and "test -e /etc/localtime" in df
    comments = " ".join(ln.lstrip("# ") for ln in df.splitlines() if ln.startswith("#"))
    assert "`conda create -n mcpbench python=3.10`" in comments
    assert "python3 -m pip install --no-cache-dir \"mcp[cli]==1.16.0\"" in df
    # the pin: the release mcp[cli]>=1.9.0 resolved to on the pinned commit's date (2025-10-07)
    pin = [ln for ln in _read("gbench", "docker", "mcp_bench_constraints.txt").splitlines()
           if ln.strip() and not ln.startswith("#")]
    assert [ln for ln in pin if ln.lower().startswith("mcp==")] == ["mcp==1.16.0"]
    assert "7a8eaea, 2025-10-07" in _read("gbench", "docker", "mcp_bench_constraints.txt")
    doc = _doc("mcp_bench.md")
    assert "`conda create -n mcpbench python=3.10`" in doc and "Python 3.11" not in doc
    assert "`tzdata`" in doc and "`mcp==1.16.0`" in doc and "BrokenResourceError" not in doc


def test_mcp_bench_install_sh_drift_guard_is_its_own_failing_step(tmp_path):
    df = _read("gbench", "docker", "mcp_bench.Dockerfile")
    start = df.index("RUN [ \"$(grep -c '^set -e' /app/mcp-bench/mcp_servers/install.sh)\" = 1 ]")
    guard = df[start:df.index("\nRUN ", start + 1)]
    assert "exit 1" in guard and "; true" not in guard and "install.sh; \\" not in guard
    # the guard's shell logic: a drifted script (no `set -e`, or two) fails the step
    body = guard[len("RUN "):].replace("\\\n", " ")
    for script, rc in (("#!/bin/bash\nset -e\n", 0), ("#!/bin/bash\n", 1),
                       ("set -e\nset -e\n", 1)):
        d = tmp_path / f"case{rc}{len(script)}" / "app" / "mcp-bench" / "mcp_servers"
        d.mkdir(parents=True)
        (d / "install.sh").write_text(script)
        root = str(d.parents[2])
        cmd = body.replace("/app/mcp-bench", root + "/app/mcp-bench")
        got = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        assert got.returncode == rc, (script, got.stderr)


# --------------------------------------------------------------------------- #
# toolbench: the grader config and the simulator record
# --------------------------------------------------------------------------- #
def test_toolbench_tooleval_config_is_the_official_one():
    """config.yaml completions_kwargs (temperature 0.2, max_tokens 1000) with the function call
    forced; the judge model is the only substitution."""
    assert T._JUDGE_TEMPERATURE == 0.2 and T._JUDGE_MAX_TOKENS == 1000
    pytest.importorskip("google.genai")     # declared (evals extra): CI's skip gate fails a skip
    for func in T._TEMPLATES:
        cfg = T._judge_generation_config(func, True)
        assert cfg is not None, "google-genai imports but no generation config was built"
        # ToolEval's 1000 for the reply plus the Gemini thinking allowance
        assert cfg.temperature == 0.2 and cfg.max_output_tokens == 1000 + 8192
        fcc = cfg.tool_config.function_calling_config
        assert str(getattr(fcc.mode, "value", fcc.mode)) == "ANY"
        assert fcc.allowed_function_names == [func]
    src = _read("gbench", "runners", "eval_suites", "toolbench.py")
    assert src.count("generate_content(") == 1
    assert "model=model, contents=prompt, config=config)" in src


def _tb_answer(solved):
    return {"query": "q", "answer": {"final_answer": "S" if solved else "N",
                                     "answer_details": []}}


@pytest.fixture
def tb_run(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR",
              "GBENCH_TOOLBENCH_REFERENCE", "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_SHARD",
              "GBENCH_TOOLBENCH_SIMULATOR_MODEL", "GBENCH_TOOLBENCH_SIMULATOR_BASE",
              "GBENCH_TOOLBENCH_SIMULATOR_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_served_model_id", lambda endpoint, fallback: fallback)
    # the bridge network's gateway (after the metadata guard) and the image's records of its
    # tool calls: stubbed, no docker (the relays listen on loopback)
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: "127.0.0.1")
    monkeypatch.setattr(T, "read_server_calls", lambda path: [])
    monkeypatch.setattr(T, "read_observations", lambda path: {})

    async def status(query, answer, sem):
        return "Solved"

    async def pref(query, cand, ref, sem):
        return "cand"

    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    ref_root = tmp_path / "ref"
    (ref_root / T._REFERENCE_DEFAULT).mkdir(parents=True)
    for g in T._GROUPS:
        (ref_root / T._REFERENCE_DEFAULT / f"{g}.json").write_text(
            json.dumps({"1": _tb_answer(False), "2": _tb_answer(False)}))
    monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: str(ref_root))
    monkeypatch.setattr(T, "_group_test_ids",
                        lambda image, groups, limit: {g: ["1", "2"] for g in groups})

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["docker", "run"]:
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
            for g in T._GROUPS:
                with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                    json.dump({"1": _tb_answer(True), "2": _tb_answer(True)}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    return lambda: T.run_toolbench("m", "http://x:8000/v1", 2)


def test_toolbench_simulator_records_the_gemini_api_base(tb_run):
    res = tb_run()
    sim = res["simulator"]
    assert sim["base_url"] == T.SIMULATOR_BASE
    assert sim["provider"] == "Gemini API" and "base_url_source" not in sim
    assert sim["matches_upstream_model"] is False
    assert f"instead of {T.UPSTREAM_SIMULATOR_MODEL}" in res["leaderboard_comparable_reason"]
    entry = _read("docker", "toolbench_entrypoint.sh")
    # the Gemini API is reached through gbench's Gemini relay (its URL; the key stays with it)
    assert 'SIMULATOR_BASE="${RELAY}/v1beta/openai/"' in entry
    assert res["simulator"]["reached_through"].startswith("gbench's Gemini relay")


def test_toolbench_simulator_ignores_an_operator_api_base_and_model(tb_run, monkeypatch):
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_BASE",
                       "https://user:hunter2pass@llm.example.com:8443/v1/?key=querysecret99")
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_MODEL", "some-model")
    res = tb_run()
    sim = res["simulator"]
    assert sim["base_url"] == T.SIMULATOR_BASE and sim["source"] == "gbench judge cascade"
    text = json.dumps(res)
    assert "hunter2pass" not in text and "querysecret99" not in text
    assert "llm.example.com" not in text and "some-model" not in text
    assert f"instead of {T.UPSTREAM_SIMULATOR_MODEL}" in res["leaderboard_comparable_reason"]


def test_toolbench_simulator_cannot_be_pointed_at_openai(tb_run, monkeypatch):
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_MODEL", T.UPSTREAM_SIMULATOR_MODEL)
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_BASE", "https://api.openai.com/v1")
    res = tb_run()
    reason = res["leaderboard_comparable_reason"]
    assert "api.openai.com" not in json.dumps(res)
    assert f"instead of {T.UPSTREAM_SIMULATOR_MODEL}" in reason
    assert res["simulator"]["matches_upstream_model"] is False
    assert res["simulator"]["models"] != [T.UPSTREAM_SIMULATOR_MODEL]
    assert res["leaderboard_comparable"] is False
    doc = _doc("toolbench.md")
    assert "`simulator.base_url`" in doc and "`matches_upstream_model`" in doc


# --------------------------------------------------------------------------- #
# the local images are never pulled
# --------------------------------------------------------------------------- #
def test_every_docker_run_of_these_suites_refuses_to_pull():
    import re
    for mod in ("mcp_bench.py", "toolbench.py", "mcp_atlas.py"):
        src = _read("gbench", "runners", "eval_suites", mod)
        runs = re.findall(r'\["docker", "run",[^\]]*', src)
        assert runs, mod
        for run in runs:
            assert '"--pull", "never"' in run, (mod, run[:80])


def test_mcp_bench_probe_and_run_commands_carry_pull_never(monkeypatch):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, 'MCP_BENCH_PROBE {"connections": {}}', "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    MB.provisioning_probe("img")
    assert seen[0][:5] == ["docker", "run", "--rm", "--pull", "never"]


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
