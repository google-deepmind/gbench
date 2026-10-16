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

"""Deterministic unit tests for the mcp_bench canonical harness.

Covers the pieces that don't need Docker, a live model, or the Gemini judge: the 0-1 Overall
Score aggregation, the cascading-judge config + backoff schedule (with a mocked client), and the
launcher's server-availability + task-filtering logic."""

import asyncio
import itertools
import json
import os
import sys

import pytest

from gbench.runners.eval_suites import mcp_bench as M

# The container adapter (judge + launcher) lives under gbench/docker; add it to the path.
_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import mcp_bench_cascade_judge as CJ  # noqa: E402
import mcp_bench_run as RUN  # noqa: E402


# --------------------------------------------------------------------------- #
# Overall Score aggregation.
# --------------------------------------------------------------------------- #
def _metrics(schema=0.8, name=0.9, tc=8.0, tu=6.0, pl=7.0):
    return {"input_schema_compliance": schema, "valid_tool_name_rate": name,
            "task_completion_score": tc, "tool_selection_score": tu,
            "planning_effectiveness_and_efficiency_score": pl, "tool_call_success_rate": 0.5}


def test_overall_single_file():
    r = M.compute_overall_score(_metrics())
    d = r["dimensions"]
    # Table 3 of arXiv:2508.20453: schema understanding = mean(valid tool name, schema
    # compliance, execution success).
    assert d["schema_understanding"] == pytest.approx((0.8 + 0.9 + 0.5) / 3)
    assert d["task_completion"] == pytest.approx(0.8)   # 8.0/10
    assert d["tool_usage"] == pytest.approx(0.6)
    assert d["planning_effectiveness"] == pytest.approx(0.7)
    assert r["overall"] == pytest.approx(((0.8 + 0.9 + 0.5) / 3 + 0.8 + 0.6 + 0.7) / 4)


def test_overall_multi_file_task_weighted():
    results = {
        "filtered_single.json": _metrics(schema=1.0, name=1.0, tc=10.0, tu=10.0, pl=10.0),  # perfect
        "filtered_multi.json": _metrics(schema=0.0, name=0.0, tc=0.0, tu=0.0, pl=0.0),       # zero
    }
    manifest = {"per_file": {"single.json": {"kept": 90}, "multi.json": {"kept": 10}}}
    r = M.compute_overall_score(results, manifest)
    # schema dim per file = mean(name, schema, execution success=0.5): 2.5/3 and 0.5/3, then
    # task-weighted: (2.5/3*90 + 0.5/3*10)/100 ; task_completion: (1.0*90+0*10)/100 = 0.9 ; etc.
    schema = (2.5 / 3 * 90 + 0.5 / 3 * 10) / 100
    assert r["dimensions"]["schema_understanding"] == pytest.approx(schema)
    assert r["dimensions"]["task_completion"] == pytest.approx(0.9)
    assert r["overall"] == pytest.approx((schema + 0.9 * 3) / 4)


def test_overall_missing_keys_are_skipped():
    r = M.compute_overall_score({"input_schema_compliance": 0.5, "valid_tool_name_rate": 0.5})
    # only the schema dimension is present -> overall == that dimension
    assert r["dimensions"]["task_completion"] is None
    assert r["overall"] == pytest.approx(0.5)


def test_overall_empty_returns_none():
    assert M.compute_overall_score({}) is None
    assert M.compute_overall_score({"unrelated": 1}) is None


# --------------------------------------------------------------------------- #
# Cascade judge: config resolution + backoff schedule + short-circuit.
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, content):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = None


class _FakeCompletions:
    def __init__(self, script):
        self._script = script
        self.models = []

    async def create(self, **kw):
        self.models.append(kw["model"])
        out = self._script(kw["model"])
        if isinstance(out, Exception):
            raise out
        return _FakeResp(out)


class _FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(script)})()


def _make_judge(monkeypatch, script, models=None, rounds=3, backoff=1.0):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    if models is None:
        monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
        monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    else:
        monkeypatch.setenv("GBENCH_JUDGE_MODELS", ",".join(models))
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", str(rounds))
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", str(backoff))
    j = CJ.CascadeGeminiJudge()
    j._async_client = _FakeClient(script)
    # one client per key, rotated per round (as base._next_judge_client): here the one fake
    j._async_clients = [j._async_client]
    j._client_cycle = itertools.cycle(j._async_clients)
    return j


def test_cascade_default_models_and_knobs(monkeypatch):
    j = _make_judge(monkeypatch, lambda m: "ok")
    assert j._cascade == CJ._DEFAULT_JUDGE_CASCADE
    assert j._rounds == 3 and j._backoff == 1.0
    assert j.provider_type == "openai_compatible"


def test_cascade_first_model_success_short_circuits(monkeypatch):
    j = _make_judge(monkeypatch, lambda m: '{"score": 1}')
    out = asyncio.run(j.get_completion("sys", "user", 100))
    assert out == '{"score": 1}'
    assert j._async_client.chat.completions.models == [CJ._DEFAULT_JUDGE_CASCADE[0]]  # only one call


def test_cascade_falls_through_then_outage_with_backoff_schedule(monkeypatch):
    sleeps = []

    async def _fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr(CJ.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(CJ.random, "uniform", lambda a, b: 0.0)

    j = _make_judge(monkeypatch, lambda m: RuntimeError("429"), rounds=3, backoff=1.0)
    with pytest.raises(RuntimeError, match="JUDGE_OUTAGE"):
        asyncio.run(j.get_completion("sys", "user", 100))
    # every model tried every round
    assert len(j._async_client.chat.completions.models) == 3 * len(CJ._DEFAULT_JUDGE_CASCADE)
    # backoff only BETWEEN rounds: rounds-1 sleeps, durations backoff*2**rnd (+uniform=0)
    assert sleeps == [1.0, 2.0]


def test_cascade_second_model_recovers(monkeypatch):
    good = CJ._DEFAULT_JUDGE_CASCADE[1]

    def script(m):
        return "recovered" if m == good else RuntimeError("down")
    j = _make_judge(monkeypatch, script)
    out = asyncio.run(j.get_completion("s", "u", 50))
    assert out == "recovered"
    assert j._async_client.chat.completions.models == CJ._DEFAULT_JUDGE_CASCADE[:2]


def test_cascade_clean_and_parse_json():
    j = CJ.CascadeGeminiJudge()
    assert j.clean_and_parse_json('```json\n{"score": 2}\n```') == {"score": 2}
    assert j.clean_and_parse_json('{"a": 1}') == {"a": 1}
    assert j.clean_and_parse_json('prefix {"b": 3} suffix')["b"] == 3


# --------------------------------------------------------------------------- #
# Launcher: server availability + task-group filtering.
# --------------------------------------------------------------------------- #
def test_availability_network_up_keys_gate(monkeypatch):
    monkeypatch.setattr(RUN, "_network_up", lambda: True)
    # a key in the model container makes no server available: the credentialed servers serve only
    # through the bridge to gbench's credentialed-servers container
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.setattr(RUN, "BRIDGED", {"Hugging Face": "hugging-face", "BioMCP": "biomcp"})
    servers = {"Math MCP", "Wikipedia", "Google Maps", "Hugging Face", "BioMCP"}
    avail, net = RUN._available_servers(servers)
    assert net is True
    assert "Math MCP" in avail          # offline
    assert "Wikipedia" in avail         # network-only
    assert "BioMCP" in avail            # bridged
    assert "Hugging Face" in avail      # bridged
    assert "Google Maps" not in avail   # no bridge


def test_availability_network_down_offline_only(monkeypatch):
    monkeypatch.setattr(RUN, "_network_up", lambda: False)
    servers = {"Math MCP", "Time MCP", "Wikipedia", "Google Maps"}
    avail, net = RUN._available_servers(servers)
    assert net is False
    assert avail == {"Math MCP", "Time MCP"}   # only the offline set


def test_built_servers_checks_artifacts_and_venvs(tmp_path):
    srv = tmp_path / "mcp_servers"
    srv.mkdir()
    commands = {
        "Node OK": {"cmd": "node build/index.js", "cwd": "../node-ok"},
        "Node Broken": {"cmd": "node dist/cli.js", "cwd": "../node-broken"},
        "Py File": {"cmd": "python server.py", "cwd": "../py-file"},
        "Module": {"cmd": "python -m foo.bar", "cwd": "../module-only"},   # no file to check -> built
        "Uv Venv": {"cmd": "uv run biomcp run", "cwd": "../uv-venv"},
        "Uv NoVenv": {"cmd": "uv run python -m x", "cwd": "../uv-novenv"},
    }
    (srv / "commands.json").write_text(json.dumps(commands))
    # `../<dir>` cwd maps to mcp_servers/<dir> (the upstream loader convention).
    (srv / "node-ok" / "build").mkdir(parents=True); (srv / "node-ok" / "build" / "index.js").write_text("//")
    (srv / "node-broken").mkdir()                     # dist/cli.js missing
    (srv / "py-file").mkdir(); (srv / "py-file" / "server.py").write_text("#")
    (srv / "module-only").mkdir()
    (srv / "uv-venv" / ".venv").mkdir(parents=True)
    (srv / "uv-novenv").mkdir()                       # no .venv
    built = RUN._built_servers(str(tmp_path))
    assert built == {"Node OK", "Py File", "Module", "Uv Venv"}
    assert "Node Broken" not in built and "Uv NoVenv" not in built


def test_launcher_leaves_upstream_server_commands_as_they_are():
    """Every `python ...` server runs on the image's one interpreter, as upstream's single
    environment and commands.json have it; nothing repoints a server at a venv of its own."""
    assert not hasattr(RUN, "_patch_venv_commands") and not hasattr(RUN, "_VENV_SERVERS")
    with open(RUN.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert ".venv/bin/python" not in src and 'open(cj_path, "w"' not in src
    assert "_patch_venv_commands" not in M._PROBE


def test_filter_file_group_subset_and_budget(tmp_path):
    data = {"server_tasks": [
        {"servers": ["Math MCP"], "tasks": [{"task_id": "a"}, {"task_id": "b"}]},
        {"servers": ["Wikipedia"], "tasks": [{"task_id": "c"}]},              # dropped (not allowed)
        {"servers": ["Math MCP", "Time MCP"], "tasks": [{"task_id": "d"}]},
    ]}
    src = tmp_path / "tasks.json"
    src.write_text(json.dumps(data))
    allowed = {"Math MCP", "Time MCP"}
    out, kept, dropped, budget = RUN._filter_file(str(src), allowed, str(tmp_path))
    assert kept == 3 and dropped == 1
    kept_groups = json.loads(open(out).read())["server_tasks"]
    assert len(kept_groups) == 2

    # budget caps kept tasks: budget=2 keeps only the first group (2 tasks), then stops
    out2, kept2, dropped2, budget2 = RUN._filter_file(str(src), allowed, str(tmp_path), budget=2)
    assert kept2 == 2 and budget2 == 0
