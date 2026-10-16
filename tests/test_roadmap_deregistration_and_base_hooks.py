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

"""Deregistration, renames, cache isolation, the base-runner hooks and the shared docs.

  * cybergym is deregistered like lmarena_web_agent: not registered, and `--evals <name>` for
    any ROADMAP_SUITES entry exits with an error naming docs/evals_roadmap.md.
  * culer -> longbench_v2_code and nexus_function_calling -> nexusraven_api_eval: old names
    resolve with a warning, old temperature and knob variables are deprecated aliases, and
    --eval-limit draws the same rows.
  * every gbench cache root follows GBENCH_CACHE_DIR, which tests/conftest.py points at a
    temporary directory for each test.
  * base.py hooks: official sampling (OfficialSampling), tool-loop argument errors and a
    per-sample wall clock (ToolLoopProtocol), prefill fields forwarded to the chat request and
    /tokenize, and a public shard helper.
  * provenance, NOTICE, docs and uv.lock stay in step with the code.

No model server, judge API or dataset hub is contacted.
"""

import asyncio
import json
import os
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gbench.runners import evals as E
from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO / "gbench" / "runners" / "eval_suites"
DOCS = REPO / "docs" / "evals"

RENAMES = {"culer": "longbench_v2_code", "nexus_function_calling": "nexusraven_api_eval"}
PRE_RENAME_SEEDS = {"longbench_v2_code": "culer", "nexusraven_api_eval": "nexus_function_calling"}


# The README was split: these docs hold sections moved verbatim out of the README, so a
# negative check that used to scan the whole README scans README.md plus all of them.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)
_README_AND_SPLIT_DOCS = ("README.md", *_README_SPLIT_DOCS)


class _Stop(Exception):
    pass


def _cli(monkeypatch, *argv):
    """Run gbench.cli.main through validation only; returns the parsed args."""
    from gbench import cli
    captured = {}
    real = cli.create_parser

    def _parser():
        p = real()
        orig = p.parse_args
        p.parse_args = lambda a=None: captured.setdefault("args", orig(a))
        return p

    def _stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(cli, "create_parser", _parser)
    monkeypatch.setattr(cli, "setup_logging", _stop)
    with pytest.raises(_Stop):
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "1024", *argv])
    return captured["args"]


def _runner_cfg(tmp_path, evals):
    return SimpleNamespace(evals=evals, eval_custom_jsonl=None, eval_plugins_dir=None,
                           batch_sizes=[1], remote_endpoint="http://127.0.0.1:1/v1", tokenizer="t",
                           eval_max_output_tokens=1024, skip_existing=False,
                           log_manager=SimpleNamespace(results_dir=tmp_path, base_dir=tmp_path,
                                                       save_result=lambda p, r: None))


# --------------------------------------------------------------------------- #
# deregistration: cybergym joins the roadmap suites
# --------------------------------------------------------------------------- #
def test_cybergym_is_not_registered_anywhere():
    from gbench.runners.eval_suites import SUITES, RENAMED_SUITES
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    import gbench.runners.eval_suites as pkg
    builtin = {s for _, su in E.BUILTIN_PILLARS for s in su}
    assert "cybergym" not in SUITES and "cybergym" not in builtin
    assert "cybergym" not in CANONICAL_SYNC and "cybergym" not in RENAMED_SUITES
    assert "run_cybergym" not in pkg.__all__ and not hasattr(pkg, "run_cybergym")
    assert not (SUITE_DIR / "cybergym.py").exists()
    assert not (DOCS / "cybergym.md").exists()


def test_roadmap_suites_are_the_deregistered_roadmap_entries():
    from gbench.runners.eval_suites import SUITES, ROADMAP_SUITES
    roadmap = (REPO / "docs" / "evals_roadmap.md").read_text(encoding="utf-8")
    assert {"cybergym", "lmarena_web_agent"} <= set(ROADMAP_SUITES)
    for name, entry in ROADMAP_SUITES.items():
        assert name not in SUITES, name
        assert f"### {entry}: `{name}`" in roadmap, name


@pytest.mark.parametrize("name", ["cybergym", "lmarena_web_agent"])
def test_resolving_a_roadmap_suite_names_the_roadmap(name):
    from gbench.runners.eval_suites import resolve_suite_name
    with pytest.raises(ValueError, match=r"docs/evals_roadmap\.md") as err:
        resolve_suite_name(name)
    assert name in str(err.value)


@pytest.mark.parametrize("name", ["cybergym", "lmarena_web_agent"])
def test_cli_exits_with_the_roadmap_error_not_an_unknown_suite(name, capsys):
    from gbench import cli
    with pytest.raises(SystemExit) as ex:
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "1024", "--evals", f"mmlu,{name}"])
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "docs/evals_roadmap.md" in err and name in err
    assert "Unknown eval suite" not in err


def test_an_unknown_name_is_still_an_unknown_suite(capsys):
    from gbench import cli
    with pytest.raises(SystemExit):
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "1024", "--evals", "no_such_suite"])
    assert "Unknown eval suite 'no_such_suite'" in capsys.readouterr().err


def test_catalogs_carry_no_runnable_cybergym_row():
    # README.md and the docs split out of it: never a suite-table row or a suite-doc link
    # (links in docs/ are relative, so "evals/cybergym.md" also catches "docs/evals/cybergym.md").
    all_mentions = []
    for rel in _README_AND_SPLIT_DOCS:
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "| `cybergym`" not in text and "evals/cybergym.md" not in text, rel
        all_mentions += [(rel, l) for l in text.splitlines() if "`cybergym`" in l]
    # The suite catalog moved verbatim to docs/eval-suites.md (links rewritten relative to docs/).
    catalog = (REPO / "docs" / "eval-suites.md").read_text(encoding="utf-8")
    # Named only in the list of deregistered roadmap suites, never as a suite-table row.
    mentions = [l for l in catalog.splitlines() if "`cybergym`" in l]
    assert len(all_mentions) == 1 and all_mentions[0][0] == "docs/eval-suites.md", all_mentions
    assert len(mentions) == 1 and "deregistered" in catalog[catalog.index(mentions[0]) - 400:
                                                             catalog.index(mentions[0])]
    for rel in ("docs/THIRD_PARTY_DATASETS.md", "docs/evals/env-knobs.md"):
        assert "cybergym" not in (REPO / rel).read_text(encoding="utf-8"), rel
    resources = (REPO / "docs" / "RESOURCES.md").read_text(encoding="utf-8")
    line = next(l for l in resources.splitlines() if "CyberGym" in l)
    assert "deregistered (roadmap-only, not runnable)" in line and "evals/cybergym.md" not in line


# --------------------------------------------------------------------------- #
# renames
# --------------------------------------------------------------------------- #
def test_rename_map_and_registry():
    from gbench.runners.eval_suites import SUITES, RENAMED_SUITES
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    builtin = {s for _, su in E.BUILTIN_PILLARS for s in su}
    for old, new in RENAMES.items():
        assert RENAMED_SUITES[old] == new
        assert new in SUITES and new in builtin and new in CANONICAL_SYNC, new
        assert old not in SUITES and old not in builtin and old not in CANONICAL_SYNC, old
        assert (DOCS / f"{new}.md").exists() and (SUITE_DIR / f"{new}.py").exists(), new
        assert not (DOCS / f"{old}.md").exists() and not (SUITE_DIR / f"{old}.py").exists(), old
    assert len(builtin) == 88


def test_renamed_modules_carry_their_registry_names():
    from gbench.runners.eval_suites import longbench_v2_code, nexusraven_api_eval
    for mod, name in ((longbench_v2_code, "longbench_v2_code"),
                      (nexusraven_api_eval, "nexusraven_api_eval")):
        assert mod.EVAL_NAME == name and mod.DOCS_URL == f"docs/evals/{name}.md"
        assert callable(getattr(mod, f"run_{name}"))


def test_old_names_resolve_in_the_cli_and_the_runner(monkeypatch, tmp_path):
    args = _cli(monkeypatch, "--evals", "culer,nexus_function_calling")
    assert args.evals == ["longbench_v2_code", "nexusraven_api_eval"]
    ran = []
    monkeypatch.setattr(E, "_resolve_remote_model_id", lambda url, fallback: "m")
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(E.EvalsBenchmarkRunner, "_run_single_eval",
                        lambda self, eval_name, **k: ran.append(eval_name) or {"accuracy": 0.0})
    model = SimpleNamespace(name="m", short_name="m", hf_model_id="org/m")
    results = E.EvalsBenchmarkRunner(_runner_cfg(tmp_path, list(RENAMES))).run(
        model, SimpleNamespace(value="remote-endpoint"))
    assert ran == list(RENAMES.values())
    assert all(r["canonical_sync"]["status"] == "reconciled" for r in results)


@pytest.mark.parametrize("old,new", sorted(RENAMES.items()))
def test_old_temperature_variables_are_deprecated_aliases(monkeypatch, old, new):
    old_var, new_var = base.temperature_env_var(old), base.temperature_env_var(new)
    monkeypatch.delenv(new_var, raising=False)
    monkeypatch.setenv(old_var, "0.3")
    assert base.resolve_temperature(new) == (0.3, f"env:{old_var}")
    monkeypatch.setenv(new_var, "0.7")
    assert base.resolve_temperature(new) == (0.7, f"env:{new_var}")


def test_longbench_knobs_read_the_culer_names_as_aliases(monkeypatch):
    from gbench.runners.eval_suites import longbench_v2_code as L
    for var in ("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "GBENCH_CULER_ALL_DOMAINS",
                "CULER_ALL_DOMAINS", "GBENCH_LONGBENCH_V2_CODE_WINDOW", "GBENCH_CULER_WINDOW"):
        monkeypatch.delenv(var, raising=False)
    assert L._all_domains() is False and L._window_override() is None
    monkeypatch.setenv("GBENCH_CULER_ALL_DOMAINS", "1")
    monkeypatch.setenv("GBENCH_CULER_WINDOW", "131072")
    assert L._all_domains() is True and L._window_override() == 131072
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_WINDOW", "65536")
    assert L._window_override() == 65536
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_WINDOW", "lots")
    with pytest.raises(RuntimeError, match="GBENCH_LONGBENCH_V2_CODE_WINDOW"):
        L._window_override()


def test_limit_seeds_keep_the_pre_rename_strings():
    lb = (SUITE_DIR / "longbench_v2_code.py").read_text(encoding="utf-8")
    assert 'seed="culer"' in lb and 'limit_seed="culer"' in lb
    nx = (SUITE_DIR / "nexusraven_api_eval.py").read_text(encoding="utf-8")
    assert 'seed="nexus_function_calling"' in nx and 'limit_seed="nexus_function_calling"' in nx


def test_longbench_passes_the_old_seed_to_run_eval_suite(monkeypatch):
    from gbench.runners.eval_suites import longbench_v2_code as L
    seen = {}
    base.set_run_knobs(max_model_len=262144)
    monkeypatch.setattr(L, "_load_tokenizer", lambda name: object())
    monkeypatch.setattr(L, "_load_longbench_v2_code_samples", lambda **kw: [])
    monkeypatch.setattr(L, "run_eval_suite",
                        lambda **kw: seen.update(kw) or {"sample_traces": [], "temperature": 0.1})
    res = L.run_longbench_v2_code("m", "http://x/v1", limit=5)
    assert seen["eval_name"] == "longbench_v2_code" and seen["limit_seed"] == "culer"
    assert "longbench_v2_code_report" in res and "culer_report" not in res


@pytest.mark.parametrize("name", sorted(RENAMES.values()))
def test_renamed_docs_run_the_new_name_with_the_mandatory_budget(name):
    doc = (DOCS / f"{name}.md").read_text(encoding="utf-8")
    assert doc.splitlines()[0] == f"# {name}"
    run = doc.split("## Run", 1)[1]
    assert f"--evals {name}" in run and "--max-output-tokens" in run
    assert doc.isascii()


def test_catalogs_use_the_new_ids():
    for rel in (*_README_AND_SPLIT_DOCS, "docs/RESOURCES.md",
                "docs/THIRD_PARTY_DATASETS.md", "docs/evals/env-knobs.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        for old in RENAMES:
            rest = text.replace(f"renamed from `{old}`", "")
            assert f"`{old}`" not in rest, (rel, old)
    # The suite catalog moved verbatim to docs/eval-suites.md; its links are relative to docs/.
    catalog = (REPO / "docs" / "eval-suites.md").read_text(encoding="utf-8")
    for name in RENAMES.values():
        assert f"`{name}`" in catalog and f"](evals/{name}.md)" in catalog, name


# --------------------------------------------------------------------------- #
# cache isolation
# --------------------------------------------------------------------------- #
def test_the_fixture_points_the_cache_root_at_a_temporary_directory(tmp_path_factory):
    root = Path(os.environ["GBENCH_CACHE_DIR"])
    assert root.is_dir()
    assert str(root).startswith(str(tmp_path_factory.getbasetemp()))
    assert root != Path.home() / ".cache" / "gbench"
    assert base.gbench_cache_dir() == str(root)


def test_every_cache_root_follows_gbench_cache_dir(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import bfcl, bfcl_v4_agentic, loft_scifact, swe_lancer
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    assert loft_scifact._cache_dir() == str(tmp_path / "loft")
    assert bfcl._default_project_root() == str(tmp_path / "bfcl")
    assert bfcl_v4_agentic.default_project_root() == str(tmp_path / "bfcl")
    assert swe_lancer._manifest_cache() == str(tmp_path / "swe_lancer" / "issues_manifest.json")
    ruler_src = (SUITE_DIR / "ruler.py").read_text(encoding="utf-8")
    assert 'Path(gbench_cache_dir()) / "ruler"' in ruler_src
    assert 'os.getenv("GBENCH_CACHE_DIR")' in (SUITE_DIR / "multi_swe_bench.py").read_text(
        encoding="utf-8")


def test_quality_checkout_default_follows_gbench_cache_dir(monkeypatch, tmp_path):
    from gbench.core.config import BenchmarkConfig
    from gbench.runners.quality import QualityBenchmarkRunner
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "results"
    config.results_dir.mkdir()
    config.initialize()
    runner = QualityBenchmarkRunner(config)
    assert runner.gemmaclaw_repo_path == tmp_path / "gemmaclaw"


def test_a_dataset_snapshot_written_by_a_test_lands_in_the_temporary_cache(monkeypatch):
    """The failure this fixture exists for: a fake SWE snapshot written into the user cache."""
    from gbench.runners.eval_suites import swebench_common
    rows = [{"instance_id": "a__b-1", "problem_statement": "p"}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    path = swebench_common.pinned_swe_snapshot("org/ds", "test", "0" * 40, "copilot_bench_swe")
    assert path.startswith(os.environ["GBENCH_CACHE_DIR"])
    assert not path.startswith(str(Path.home() / ".cache" / "gbench"))


# --------------------------------------------------------------------------- #
# base hooks: fakes
# --------------------------------------------------------------------------- #
def _samples(n=1, prefix="q"):
    return [([{"role": "user", "content": f"{prefix}{i}"}], "x", {"category": f"c{i % 2}"})
            for i in range(n)]


def _run(send, samples=None, **kw):
    with mock.patch.object(base, "_send_single_request", send):
        return base.run_eval_suite(eval_name="aime", model_name="m", base_url="http://x",
                                   concurrency=kw.pop("concurrency", 1),
                                   samples=samples or _samples(), **kw)


def _recorder():
    sent = []

    async def _send(**k):
        sent.append(k)
        return base.Reply("x", None, "stop", completion_tokens=1)
    return sent, _send


class _Resp:
    status = 200

    async def json(self):
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self):
        self.sent = []

    def post(self, url, json=None):
        self.sent.append(json)
        return _Resp()


def _send_real(extra_payload, temperature):
    session = _Session()

    async def go():
        return await base._send_single_request(
            session=session, api_url="http://x/v1/chat/completions", model_name="m",
            messages=[{"role": "user", "content": "q"}], extra_payload=extra_payload,
            semaphore=asyncio.Semaphore(1), pbar=None, max_output_tokens=64,
            temperature=temperature)
    reply = asyncio.run(go())
    return reply, session.sent[0]


# --------------------------------------------------------------------------- #
# E1: official sampling
# --------------------------------------------------------------------------- #
def test_official_temperature_is_the_default_and_is_recorded():
    sent, send = _recorder()
    official = base.OfficialSampling(reference="upstream run.py@abc", temperature=0.6)
    # no-think: 0.0 (the maintainer's rule), the official value recorded
    res = _run(send, eval_fn=lambda r, g: True, thinking=False, official_sampling=official)
    assert sent[0]["temperature"] == 0.0
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    rec = res["official_sampling"]
    assert rec["reference"] == "upstream run.py@abc" and rec["temperature"] == 0.6
    assert rec["temperature_matches_official"] is False and rec["temperature_source"] == "default"
    # --thinking: 1.0 (the maintainer's rule), recorded as not the official value
    sent.clear()
    res = _run(send, eval_fn=lambda r, g: True, thinking=True, official_sampling=official)
    assert sent[0]["temperature"] == 1.0
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    rec = res["official_sampling"]
    assert rec["temperature"] == 0.6 and rec["temperature_matches_official"] is False


def test_the_operator_still_wins_over_the_official_value(monkeypatch):
    sent, send = _recorder()
    official = base.OfficialSampling(reference="r", temperature=0.6)
    monkeypatch.setenv("GBENCH_AIME_TEMPERATURE", "0.2")
    res = _run(send, eval_fn=lambda r, g: True, official_sampling=official)
    assert sent[0]["temperature"] == 0.2 and res["temperature_source"] == "env:GBENCH_AIME_TEMPERATURE"
    assert res["official_sampling"]["temperature_matches_official"] is False
    monkeypatch.delenv("GBENCH_AIME_TEMPERATURE")
    base.set_run_knobs(temperature=1.0)
    res = _run(send, eval_fn=lambda r, g: True, official_sampling=official)
    assert res["temperature"] == 1.0 and res["temperature_source"] == "cli:--temperature"
    assert res["official_sampling"]["temperature_matches_official"] is False


def test_an_official_harness_that_sends_no_temperature_still_gets_the_rules_temperature():
    sent, send = _recorder()
    official = base.OfficialSampling(reference="r", send_temperature=False)
    res = _run(send, eval_fn=lambda r, g: True, official_sampling=official)
    assert sent[0]["temperature"] == 0.0
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    assert res["official_sampling"]["temperature_matches_official"] is False
    base.set_run_knobs(temperature=0.3)
    res = _run(send, eval_fn=lambda r, g: True, official_sampling=official)
    assert sent[-1]["temperature"] == 0.3 and res["temperature_source"] == "cli:--temperature"


def test_the_request_omits_temperature_when_none_is_resolved():
    _, payload = _send_real({}, None)
    assert "temperature" not in payload
    _, payload = _send_real({}, 0.0)
    assert payload["temperature"] == 0.0


def test_official_params_are_sent_unless_the_suite_sets_them():
    sent, send = _recorder()
    official = base.OfficialSampling(reference="r", temperature=0.2,
                                     params={"top_p": 0.95, "top_k": 20})
    res = _run(send, eval_fn=lambda r, g: True, official_sampling=official)
    assert sent[0]["extra_payload"]["top_p"] == 0.95 and sent[0]["extra_payload"]["top_k"] == 20
    assert res["official_sampling"]["params"] == {"top_p": 0.95, "top_k": 20}
    sent.clear()
    _run(send, eval_fn=lambda r, g: True, official_sampling=official, extra_payload={"top_p": 0.5})
    assert sent[0]["extra_payload"]["top_p"] == 0.5 and sent[0]["extra_payload"]["top_k"] == 20


@pytest.mark.parametrize("params", [{"temperature": 0.3}, {"best_of": 2}])
def test_official_params_that_cannot_be_sent_are_refused(params):
    sent, send = _recorder()
    with pytest.raises(ValueError, match="official sampling params"):
        _run(send, eval_fn=lambda r, g: True,
             official_sampling=base.OfficialSampling(reference="r", params=params))
    assert not sent


def test_a_suite_without_official_sampling_is_unchanged():
    sent, send = _recorder()
    res = _run(send, eval_fn=lambda r, g: True)
    assert "official_sampling" not in res
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    # a suite's own temperature does not apply (the maintainer's rule)
    res = _run(send, eval_fn=lambda r, g: True, temperature=0.4)
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    assert "top_p" not in sent[-1]["extra_payload"]


def test_resolve_official_sampling_for_suites_outside_run_eval_suite():
    o = base.OfficialSampling(reference="r", temperature=0.7)
    assert base.resolve_official_sampling("aime", o) == (0.0, "default")
    unset = base.OfficialSampling(reference="r")
    assert base.resolve_official_sampling("aime", unset, thinking=True) == (1.0, "thinking")
    none = base.OfficialSampling(reference="r", send_temperature=False)
    assert base.resolve_official_sampling("aime", none) == (0.0, "default")
    assert o.record(0.7, "official")["temperature_matches_official"] is True
    assert unset.record(1.0, "default")["temperature_matches_official"] is None


# --------------------------------------------------------------------------- #
# E2: tool-loop argument errors and the per-sample wall clock
# --------------------------------------------------------------------------- #
def _tool_loop_send(requests):
    async def _send(**k):
        requests.append([dict(m) for m in k["messages"]])
        if len(requests) == 1:
            return base.Reply("", [{"id": "c1", "type": "function",
                                    "function": {"name": "lookup", "arguments": "{bad json"}}],
                              "tool_calls", completion_tokens=3)
        return base.Reply("final answer", None, "stop", completion_tokens=2)
    return _send


def _tool_run(protocol):
    requests, calls = [], []

    async def executor(name, args):
        calls.append((name, args))
        return "result"
    res = _run(_tool_loop_send(requests), eval_fn=lambda r, g: True, tool_executor=executor,
               tool_loop=protocol)
    return res, requests, calls


def test_invalid_tool_arguments_are_fed_back_as_an_error_when_the_protocol_says_so():
    proto = base.ToolLoopProtocol(reference="t", max_rounds=4, forced_final=False,
                                  invalid_arguments_as_error=True)
    res, requests, calls = _tool_run(proto)
    assert calls == []                                    # the tool is not executed
    tool_msg = requests[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"].startswith("Error: ") and len(tool_msg["content"].splitlines()) == 1
    trace = res["sample_traces"][0]
    assert trace["tool_argument_errors"] == 1 and trace["response_text"] == "final answer"
    assert res["tool_loop_protocol"]["invalid_arguments_as_error"] is True


def test_by_default_invalid_tool_arguments_still_parse_to_an_empty_dict():
    res, requests, calls = _tool_run(base.ToolLoopProtocol(reference="t", forced_final=False))
    assert calls == [("lookup", {})]
    assert requests[1][-1]["content"] == "result"
    assert "tool_argument_errors" not in res["sample_traces"][0]
    assert res["tool_loop_protocol"]["max_sample_wall_s"] is None
    assert "wall_clock_limited" not in res


def _timed_send(durations):
    async def _send(**k):
        async with k["semaphore"]:
            await asyncio.sleep(durations[k["messages"][-1]["content"]])
        return base.Reply("x", None, "stop", completion_tokens=1)
    return _send


def test_a_sample_past_its_wall_clock_scores_zero_and_is_counted():
    proto = base.ToolLoopProtocol(reference="run_eval.py --timeout", max_sample_wall_s=0.3)
    samples = [([{"role": "user", "content": "slow"}], "x", {}),
               ([{"role": "user", "content": "fast"}], "x", {})]
    res = _run(_timed_send({"slow": 3.0, "fast": 0.0}), samples=samples, concurrency=2,
               eval_fn=lambda r, g: True, tool_loop=proto)
    by_text = {t["messages"][0]["content"]: t for t in res["sample_traces"]}
    slow = by_text["slow"]
    assert slow["status"] == "WALL_CLOCK_LIMIT" and slow["is_correct"] is False
    assert slow["health"] == "wall_clock_limit" and "0.3 s" in slow["error"]
    assert by_text["fast"]["status"] == "OK" and by_text["fast"]["is_correct"] is True
    assert res["wall_clock_limited"] == 1 and res["failed_requests"] == 0
    assert res["correct_answers"] == 1 and res["total_questions"] == 2


def test_the_wall_clock_starts_when_the_sample_gets_a_request_slot():
    """Three 0.3 s samples through one slot: the last waits 0.6 s in the queue, which must not
    count against its 0.5 s limit."""
    proto = base.ToolLoopProtocol(reference="t", max_sample_wall_s=0.5)
    samples = [([{"role": "user", "content": f"s{i}"}], "x", {}) for i in range(3)]
    res = _run(_timed_send({f"s{i}": 0.3 for i in range(3)}), samples=samples, concurrency=1,
               eval_fn=lambda r, g: True, tool_loop=proto)
    assert res["wall_clock_limited"] == 0
    assert all(t["status"] == "OK" for t in res["sample_traces"])


def test_waiting_for_a_slot_on_a_later_turn_does_not_count_either():
    """One slot: sample A's second turn waits 0.45 s while sample B holds the slot. A's own time
    is 0.2 s, well inside its 0.6 s limit, although 0.65 s pass between its first request and
    its answer."""
    proto = base.ToolLoopProtocol(reference="t", max_sample_wall_s=0.6, forced_final=False)
    turns = {}

    async def send(**k):
        who = k["messages"][0]["content"]
        turns[who] = turns.get(who, 0) + 1
        async with k["semaphore"]:
            await asyncio.sleep(0.45 if who == "B" else 0.1)
        if who == "A" and turns[who] == 1:
            return base.Reply("", [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}],
                              "tool_calls", completion_tokens=1)
        return base.Reply("done", None, "stop", completion_tokens=1)

    async def executor(name, args):
        return "r"
    samples = [([{"role": "user", "content": "A"}], "x", {}),
               ([{"role": "user", "content": "B"}], "x", {})]
    res = _run(send, samples=samples, concurrency=1, eval_fn=lambda r, g: True,
               tool_executor=executor, tool_loop=proto)
    assert res["wall_clock_limited"] == 0 and turns == {"A": 2, "B": 1}
    assert all(t["status"] == "OK" for t in res["sample_traces"])


def test_a_batch_scorer_cannot_score_a_wall_clock_stopped_sample():
    proto = base.ToolLoopProtocol(reference="t", max_sample_wall_s=0.2)

    async def lenient(traces):
        for t in traces:
            t["is_correct"] = True
    samples = [([{"role": "user", "content": "slow"}], "x", {})]
    res = _run(_timed_send({"slow": 3.0}), samples=samples, async_eval_fn=lenient, tool_loop=proto)
    assert res["sample_traces"][0]["is_correct"] is False and res["correct_answers"] == 0


# --------------------------------------------------------------------------- #
# E3: prefilled assistant turns
# --------------------------------------------------------------------------- #
def test_prefill_fields_reach_the_chat_request():
    _, payload = _send_real({"continue_final_message": True, "add_generation_prompt": False}, 0.0)
    assert payload["continue_final_message"] is True and payload["add_generation_prompt"] is False
    assert {"continue_final_message", "add_generation_prompt"} <= base.ALLOWED_API_KEYS


def test_bigcodebench_prefill_probe_passes_against_the_real_request_builder():
    from gbench.runners.eval_suites import bigcodebench
    assert bigcodebench._prefill_supported() is True


def test_prefill_fields_reach_tokenize(monkeypatch):
    bodies = []

    class _Tok:
        def read(self):
            return json.dumps({"count": 42}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _Tok()
    monkeypatch.setattr(base.urllib.request, "urlopen", fake_urlopen)
    base.set_run_knobs(tokenize_url="http://127.0.0.1:1", served_model="m")
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "```python\n"}]
    got = base._prompt_token_bounds(msgs, None, {"enable_thinking": False},
                                    base.prefill_fields({"continue_final_message": True,
                                                         "add_generation_prompt": False}))
    assert got == (42, 42, "server")
    assert bodies[-1]["continue_final_message"] is True
    assert bodies[-1]["add_generation_prompt"] is False
    base._prompt_token_bounds([{"role": "user", "content": "other"}], None, None)
    assert bodies[-1]["add_generation_prompt"] is True and "continue_final_message" not in bodies[-1]


def test_run_eval_suite_counts_and_sends_the_prefill_on_the_first_turn_only(monkeypatch):
    counted, requests = [], []

    async def fake_clamp(messages, max_tokens, tools=None, chat_template_kwargs=None,
                         prefill=None):
        counted.append(prefill)
        return max_tokens, None
    monkeypatch.setattr(base, "clamp_to_context_async", fake_clamp)
    sample = [([{"role": "user", "content": "q"}, {"role": "assistant", "content": "pre"}], "x",
               {"continue_final_message": True, "add_generation_prompt": False})]
    send = _tool_loop_send(requests)
    seen = []

    async def spy(**k):
        seen.append(dict(k["extra_payload"]))
        return await send(**k)

    async def executor(name, args):
        return "r"
    _run(spy, samples=sample, eval_fn=lambda r, g: True, tool_executor=executor,
         tool_loop=base.ToolLoopProtocol(reference="t", forced_final=False))
    assert counted[0] == {"continue_final_message": True, "add_generation_prompt": False}
    assert seen[0]["continue_final_message"] is True
    assert "continue_final_message" not in seen[1] and "add_generation_prompt" not in seen[1]


# --------------------------------------------------------------------------- #
# E4: the public shard helper
# --------------------------------------------------------------------------- #
def test_apply_shard_selects_round_robin_and_labels_the_result(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    assert base.apply_shard(list(range(10)), "x") == (list(range(10)), None)
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    assert base.apply_shard(list(range(10)), "x") == ([1, 4, 7], "2/3")
    assert base.record_shard({}, "2/3") == {"shard": "2/3"}
    assert base.record_shard({}, None) == {}
    monkeypatch.setenv("GBENCH_SHARD", "4/3")
    with pytest.raises(ValueError):
        base.apply_shard([1, 2, 3])


def test_run_eval_suite_records_the_shard_and_limits_within_it(monkeypatch):
    sent, send = _recorder()
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    res = _run(send, samples=_samples(8), eval_fn=lambda r, g: True)
    assert res["shard"] == "1/2" and res["total_questions"] == 4
    assert sorted(k["messages"][0]["content"] for k in sent) == ["q0", "q2", "q4", "q6"]
    sent.clear()
    res = _run(send, samples=_samples(8), eval_fn=lambda r, g: True, limit=2)
    assert {k["messages"][0]["content"] for k in sent} <= {"q0", "q2", "q4", "q6"}
    assert res["total_questions"] == 2
    monkeypatch.delenv("GBENCH_SHARD")
    assert "shard" not in _run(send, samples=_samples(2), eval_fn=lambda r, g: True)


# --------------------------------------------------------------------------- #
# provenance, notices, docs, lock
# --------------------------------------------------------------------------- #
def test_canonical_sync_carries_the_updated_pins_and_no_internal_labels():
    from gbench.runners.eval_suites import swe_bench_pro, wildbench
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as C
    assert swe_bench_pro._REVISION in C["swe_bench_pro"]["dataset"]
    assert "SWE-bench/SWE-bench_Verified@78f471bf" in C["copilot_bench_swe"]["dataset"]
    assert wildbench._REVISION in C["wildbench"]["dataset"] and "1,024" in C["wildbench"]["dataset"]
    assert C["tau2"]["revision"].startswith("b7ea9074c1cba482b30687fecdb5c8425fd6f619")
    assert "MathArena/hmmt_feb_2026" in C["hmmt"]["dataset"]
    assert "Violation@5" in C["cimemories"]["method"]
    assert "no relaxed-match fallback" in C["charxiv"]["method"]
    # Work-tracking labels must not be stamped into results.
    tracking = re.compile(r"\b(?:Wave \d|Group [A-D]:|WS\d)")
    for name, entry in C.items():
        for value in entry.values():
            assert not tracking.search(value), (name, value)
    src = (SUITE_DIR / "canonical_sync.py").read_text(encoding="utf-8")
    assert max(len(l) for l in src.splitlines()[:30]) <= 100


def test_notice_inventories_the_vendored_code_with_licenses():
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert notice.isascii()
    for s in ("_ifeval_vendor/", "_teval_vendor/", "charxiv_constants.py", "matharena_src/",
              "simpleqa.py", "healthbench.py", "humanitys_last_exam.py", "wildbench.py",
              "bfcl.py", "nestful.py", "acebench.py", "api_bank.py", "seal_tools.py",
              "toolbench.py", "spider2.py", "t_eval.py", "lab_bench.py", "vqa_common.py",
              "complexfuncbench.py", "mcp_atlas.py", "docker/multipl_e.Dockerfile"):
        assert s in notice, s
    assert notice.count("Permission is hereby granted") == 1        # the one MIT text
    assert "Copyright (c) 2024 OpenAI" in notice and "Copyright (c) 2024 SRI Lab" in notice
    assert "Redistributions of source code must retain" in notice     # BSD texts carried
    fetched = notice.split("Fetched at run time, not shipped", 1)[1].split(
        "Upstream license not confirmed", 1)[0]
    assert "cimemories" in fetched and "CC BY-NC 4.0" in fetched
    assert "multipl_e" in fetched and "Machine Learning Restriction" in fetched
    assert "(1) and (2)" not in notice


def _norm_requirements(reqs):
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    out = set()
    for r in reqs:
        if isinstance(r, str):
            req = Requirement(r)
            name, marker, spec = req.name, str(req.marker or ""), str(req.specifier)
        else:
            name, spec = r["name"], str(SpecifierSet(r.get("specifier", "")))
            marker = str(Marker(r["marker"])) if r.get("marker") else ""
        out.add((canonicalize_name(name), marker, spec))
    return out


def test_uv_lock_matches_pyproject():
    """uv.lock records the project's requirements; a stale lock no longer matches pyproject."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    root = next(p for p in lock["package"] if p["name"] == "gbench")
    declared = list(project["dependencies"])
    for extra, reqs in project["optional-dependencies"].items():
        # a requirement with its own marker (harbor's python_full_version) gets the extra
        # added to that marker, as uv.lock writes it
        declared += [(f"{r.split(';', 1)[0].strip()}; {r.split(';', 1)[1].strip()} and "
                      f"extra == '{extra}'") if ";" in r else f"{r}; extra == '{extra}'"
                     for r in reqs]
    locked = root["metadata"]["requires-dist"]
    assert _norm_requirements(declared) == _norm_requirements(locked)
    assert set(root["metadata"]["provides-extras"]) == set(project["optional-dependencies"])


def test_shared_docs_follow_the_code():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "86 canonical Python suites" in readme
    # README.md and every doc split out of it (the catalog count lives in docs/eval-suites.md)
    for rel in _README_AND_SPLIT_DOCS:
        assert "85 canonical" not in (REPO / rel).read_text(encoding="utf-8"), rel
    # The installation notes moved verbatim to docs/installation.md.
    install = (REPO / "docs" / "installation.md").read_text(encoding="utf-8")
    assert "leaves those\n>   2 tests erroring" in install
    knobs = (DOCS / "env-knobs.md").read_text(encoding="utf-8")
    for s in ("GBENCH_MATHARENA_IMAGE", "GBENCH_MCP_ATLAS_IMAGE", "GBENCH_SWEBENCH_V5_IMAGE",
              "GBENCH_LONGBENCH_V2_CODE_WINDOW", "cimemories_labels/", "OfficialSampling",
              "GBENCH_PUTNAM_FORMAL_CHECK_TIMEOUT_S"):
        assert s in knobs, s
    for gone in ("GBENCH_AIME_INCLUDE_PRECUTOFF", "GBENCH_BFCL_SEARCH_BACKEND"):
        assert gone not in knobs, gone
    tools = (DOCS / "toolchains.md").read_text(encoding="utf-8")
    assert "multi-swe-bench==1.1.2" in tools and "separate venv" not in tools
    assert "lmarena_web_agent" not in tools
    roadmap = (REPO / "docs" / "evals_roadmap.md").read_text(encoding="utf-8")
    assert "EVALS_ALL_EXCLUDE" not in roadmap and "`lcb` self-repair scenario" not in roadmap
    assert "`multipl_e` DSPy JSONAdapter retry" not in roadmap
    assert "`cimemories` Violation-at-n" not in roadmap


def test_cli_help_and_key_handling_follow_ups():
    from gbench import cli
    action = next(a for a in cli.create_parser()._actions
                  if "--eval-categories" in a.option_strings)
    assert "multi_turn" in action.help and "all_scoring" in action.help
    src = (REPO / "gbench" / "cli.py").read_text(encoding="utf-8")
    assert "raw_gemini_key" not in src


def test_multi_swe_bench_refuses_any_other_harness_release(monkeypatch):
    from gbench.runners.eval_suites import multi_swe_bench as M
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(M.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(M, "_installed_harness_version", lambda: "1.2.0")
    ok, reason = M.check_multi_swe_bench_prerequisites()
    assert ok is False and "1.2.0" in reason and "multi-swe-bench==1.1.2" in reason
    monkeypatch.setattr(M, "_installed_harness_version", lambda: M.HARNESS_RELEASE)
    assert M.check_multi_swe_bench_prerequisites() == (True, "")


def test_beam_records_the_json_repair_release():
    from gbench.runners.eval_suites import beam_128k as B
    assert B.UPSTREAM_JSON_REPAIR == "0.44.1"
    assert B._json_repair_version() == "0.44.1"          # the vendored release, not the installed one
    src = (SUITE_DIR / "beam_128k.py").read_text(encoding="utf-8")
    assert 'result["json_repair_version"] = _json_repair_version()' in src
