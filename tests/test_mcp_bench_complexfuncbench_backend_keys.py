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

"""The canonical tool backends' keys of mcp_bench and complexfuncbench.

* mcp_bench: the keys gbench requires are exactly the ones MCP-Bench requires at the pinned
  commit (README "Required API keys include ...", mcp_servers/api_key, and the `env` lists of
  mcp_servers/commands.json), each named with its server; a key that is unset or empty stops the
  suite with infra_required before any container, probe, relay or metadata check starts, and the
  message names exactly the missing keys, never a value.
* complexfuncbench: RAPID_API_KEY unset or empty is infra_required before the run (no live
  call, no model request, no judge call), and the response-based tier never falls back to the
  Gemini judge.
* Both docs say what each key is for, where to get it and what it costs, and their run commands
  use flags the CLI has.

Docker, Gemini, RapidAPI and the network are stubbed; nothing is written outside tmp.
"""

import asyncio
import os
import re
import shlex
import subprocess
import sys
from collections import Counter

import pytest

from gbench.runners.eval_suites import complexfuncbench as C
from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import mcp_bench as MB

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import mcp_bench_credentialed as CRED  # noqa: E402
import mcp_bench_run as MRUN  # noqa: E402

#: Accenture/mcp-bench@7a8eaeae83a842a2949080acc5473f65e1569daf: the README's "Required API keys
#: include" list (NPS_API_KEY, NASA_API_KEY, HF_TOKEN, GOOGLE_MAPS_API_KEY, NCI_API_KEY; the same
#: five lines are mcp_servers/api_key) and the commands.json server whose `env` lists each one
#: (no other commands.json entry lists a key).
UPSTREAM_REQUIRED_KEYS = {"NPS_API_KEY": "National Parks", "NASA_API_KEY": "NASA Data",
                          "HF_TOKEN": "Hugging Face", "GOOGLE_MAPS_API_KEY": "Google Maps",
                          "NCI_API_KEY": "BioMCP"}
#: Variables some vendored servers read that upstream neither lists nor passes.
NOT_UPSTREAM_KEYS = ("SEMANTIC_SCHOLAR_API_KEY", "OPENFDA_API_KEY", "ALPHAGENOME_API_KEY",
                     "CBIO_TOKEN", "WEATHER_API_KEY")


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


# =========================================================================== #
# mcp_bench
# =========================================================================== #
def test_the_required_keys_are_exactly_upstreams_set():
    assert MB._REQUIRED_KEYS == UPSTREAM_REQUIRED_KEYS
    assert set(MB._SERVER_KEYS) == set(UPSTREAM_REQUIRED_KEYS)
    assert {server: key for server, (key, _t) in MB.CREDENTIALED_SERVERS.items()} == {
        server: key for key, server in UPSTREAM_REQUIRED_KEYS.items()}
    assert set(CRED.CREDENTIAL_KEYS) == set(MRUN.CREDENTIAL_KEYS) == set(UPSTREAM_REQUIRED_KEYS)
    # the model container's entrypoint refuses exactly these keys
    loop = re.search(r"^for k in ([A-Z_ ]+); do$", _read("gbench/docker/mcp_bench_entrypoint.sh"),
                     re.M)
    assert loop and set(loop.group(1).split()) == set(UPSTREAM_REQUIRED_KEYS)
    assert not set(NOT_UPSTREAM_KEYS) & (set(MB._REQUIRED_KEYS) | set(MB._SERVER_KEYS))


class _Started(Exception):
    pass


@pytest.fixture
def mb_stubbed(monkeypatch):
    """run_mcp_bench with the image/Docker/Gemini prerequisites passing and everything that would
    start something (docker, the metadata guard, the credentialed container, the probe, the
    judge relay) recording the attempt and failing."""
    seen = {"prereq": 0, "started": []}

    def prereq():
        seen["prereq"] += 1
        return True, ""
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", prereq)

    def started(what):
        def _f(*a, **k):
            seen["started"].append(what)
            argv = a[0] if a and isinstance(a[0], list) else None
            raise _Started(" ".join(argv[:3]) if argv else what)
        return _f
    monkeypatch.setattr(MB.subprocess, "run", started("subprocess.run"))
    monkeypatch.setattr(MB.subprocess, "Popen", started("subprocess.Popen"))
    monkeypatch.setattr(CG, "require_metadata_blocked", started("metadata guard"))
    monkeypatch.setattr(MB, "start_credentialed_servers", started("credentialed container"))
    monkeypatch.setattr(MB, "provisioning_probe", started("probe"))
    monkeypatch.setattr(MB, "_start_judge_relay", started("judge relay"))
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    for k in NOT_UPSTREAM_KEYS:
        monkeypatch.setenv(k, "not-an-upstream-key-value")
    return seen


@pytest.mark.parametrize("missing", sorted(UPSTREAM_REQUIRED_KEYS))
@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_or_empty_key_stops_the_run_before_anything_starts(mb_stubbed, monkeypatch,
                                                                     missing, value):
    for k in UPSTREAM_REQUIRED_KEYS:
        monkeypatch.setenv(k, f"value-of-{k.lower()}-0123456789")
    if value is None:
        monkeypatch.delenv(missing)
    else:
        monkeypatch.setenv(missing, value)
    with pytest.raises(RuntimeError) as info:
        MB.run_mcp_bench("m", "http://127.0.0.1:8000/v1")
    msg = str(info.value)
    assert msg.startswith("[INFRA REQUIRED] mcp_bench cannot run and must not be skipped: ")
    assert f"required API keys unset: {missing} ({UPSTREAM_REQUIRED_KEYS[missing]});" in msg
    assert "the MCP-Bench README lists all five as required" in msg
    assert MB.DOCS_URL in msg
    for k in UPSTREAM_REQUIRED_KEYS:
        assert f"value-of-{k.lower()}" not in msg
        if k != missing:
            assert k not in msg.split("; the MCP-Bench README")[0]
    # checked before generation: nothing was started, only the local prerequisites ran
    assert mb_stubbed["started"] == [] and mb_stubbed["prereq"] == 1


def test_every_missing_key_is_named_in_one_message(mb_stubbed, monkeypatch):
    for k in UPSTREAM_REQUIRED_KEYS:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError) as info:
        MB.run_mcp_bench("m", "http://127.0.0.1:8000/v1")
    listed = str(info.value).split("required API keys unset: ", 1)[1].split(";", 1)[0]
    assert listed == ", ".join(f"{k} ({s})" for k, s in UPSTREAM_REQUIRED_KEYS.items())
    assert mb_stubbed["started"] == []


def test_the_five_keys_set_let_the_run_past_the_key_check(mb_stubbed, monkeypatch):
    for k in UPSTREAM_REQUIRED_KEYS:
        monkeypatch.setenv(k, "set")
    with pytest.raises(_Started) as info:
        MB.run_mcp_bench("m", "http://127.0.0.1:8000/v1")
    # the next step reads the run's docker network (its gateway, for the model endpoint on this
    # host); no other key is needed
    assert str(info.value) == "docker network inspect"


# =========================================================================== #
# complexfuncbench
# =========================================================================== #
@pytest.mark.parametrize("value", [None, "", "  \t "])
def test_an_unset_or_empty_rapid_api_key_is_infra_required_before_the_run(monkeypatch, value):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-0123456789")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(C, "gemini_key_live_valid", lambda k: (True, ""))
    if value is None:
        monkeypatch.delenv(C.RAPID_API_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(C.RAPID_API_KEY_ENV, value)
    calls = []
    monkeypatch.setattr(C, "rapidapi_live_check", lambda k: calls.append("live check"))
    monkeypatch.setattr(C, "_rapidapi_get", lambda *a, **k: calls.append("rapidapi"))

    async def no_judge(*a, **k):
        calls.append("judge")
    monkeypatch.setattr(C, "judge_generate_cascade", no_judge)

    async def no_drive(*a, **k):
        calls.append("drive")
    monkeypatch.setattr(C, "_drive_all", no_drive)
    ok, why = C.check_complexfuncbench_prerequisites()
    assert not ok and why.startswith("RAPID_API_KEY is not set: ")
    assert "never replaced by the Gemini judge" in why and "booking-com15" in why
    with pytest.raises(RuntimeError) as info:
        C.run_complexfuncbench("m", "http://x/v1", 1)
    assert str(info.value).startswith("[INFRA REQUIRED] complexfuncbench cannot run")
    assert "RAPID_API_KEY is not set" in str(info.value) and C.DOCS_URL in str(info.value)
    assert calls == []


def test_the_response_based_tier_never_falls_back_to_the_judge():
    cmp = C._CompareFC(C._load_exact_match_dict(), C._Embedder(), Counter(), None)
    judged = []

    async def judge(*a, **k):
        judged.append(a)
        return True
    cmp.llm_based = judge
    pred = {"name": "Search_Foo", "arguments": {"q": "New York"}}
    gold = {"name": "Search_Foo", "arguments": {"q": "NY"}}
    with pytest.raises(RuntimeError, match="has no RapidAPI client"):
        asyncio.run(cmp.compare_single_call([], [], pred, gold))
    assert judged == []


# =========================================================================== #
# The docs: every key, what it is for, where to get it, what it costs, the run command
# =========================================================================== #
def _keys_table(doc):
    section = doc.split("### The keys", 1)[1].split("\n### ", 1)[0]
    return {m.group(1): m.group(0) for m in re.finditer(r"^\| `([A-Z_]+)`[^\n]*$", section, re.M)}


def test_the_mcp_bench_doc_has_a_row_for_every_key():
    doc = _read("docs/evals/mcp_bench.md")
    assert doc.isascii()
    rows = _keys_table(doc)
    assert set(rows) == set(UPSTREAM_REQUIRED_KEYS) | {"GEMINI_API_KEY"}
    for key, row in rows.items():
        cells = [c.strip() for c in row.strip("|").split(" | ")]
        assert len(cells) == 5, (key, cells)
        assert "https://" in cells[3], key                     # where to get it
        assert re.search(r"free|billing", cells[4]), key      # what it costs
        if key in UPSTREAM_REQUIRED_KEYS:
            assert UPSTREAM_REQUIRED_KEYS[key] in cells[1], key   # the server it serves
    maps = rows["GOOGLE_MAPS_API_KEY"]
    for api in ("Geocoding API", "Elevation API", "Places API", "Directions API",
                "Distance Matrix API"):
        assert api in maps, api
    assert "billing" in maps and "DEMO_KEY" in rows["NASA_API_KEY"]
    flat = _flat(doc)
    assert "### Checking the keys" in doc
    for key in UPSTREAM_REQUIRED_KEYS:
        assert f'k["{key}"]' in doc, key                        # the check reads each key
    for name in NOT_UPSTREAM_KEYS[:4]:
        assert f"`{name}`" in flat, name
    assert "each key that is set is checked with a live request to the service its server " \
           "calls" in flat


def test_the_complexfuncbench_doc_says_where_to_get_the_keys():
    flat = _flat(_read("docs/evals/complexfuncbench.md"))
    assert "https://rapidapi.com/DataCrawler/api/booking-com15" in flat
    assert "https://aistudio.google.com/apikey" in flat
    assert "BASIC, free, 50 requests a month" in flat
    assert "unset or empty" in flat and "never replaced by the Gemini judge" in flat
    assert "no `HF_TOKEN` is needed" in flat


@pytest.mark.parametrize("doc, suite", [("docs/evals/mcp_bench.md", "mcp_bench"),
                                        ("docs/evals/complexfuncbench.md", "complexfuncbench")])
def test_the_run_commands_use_flags_the_cli_has(doc, suite):
    from gbench import cli
    parser = cli.create_parser()
    text = _read(doc)
    run = text.split("## Run", 1)[1]
    blocks = re.findall(r"```bash\n(.*?)```", run, re.S)
    commands = [" ".join(line.rstrip("\\").strip() for line in b.splitlines()
                         if not line.startswith("export "))
                for b in blocks]
    commands = [c for c in commands if c.strip().startswith("gbench ")]
    assert commands, doc
    for command in commands:
        args = parser.parse_args(shlex.split(command)[1:])
        assert args.evals == [suite] and args.max_output_tokens
        assert args.remote_endpoint or args.models
    # every env var the run block exports is a key the doc describes
    exported = set(re.findall(r"\b([A-Z][A-Z0-9_]+)=\.\.\.", run.split("```bash", 1)[1]))
    assert exported and all(f"`{name}`" in text for name in exported)
