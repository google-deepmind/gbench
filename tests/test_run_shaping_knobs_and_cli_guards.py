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

"""Cross-cutting interfaces: the run-shaping knob registry, the declared dependencies, the CLI
reference in the README and the CLI's local-run guards."""

import argparse
import ast
import re
import sys
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
SUITES_DIR = REPO / "gbench" / "runners" / "eval_suites"

#: Maintainer-owned plugin modules: never read.
_PLUGIN_MODULES = {"swe_bench_pro_agentic.py", "jetski_agent.py", "slashbin_agent.py"}


def _suite_module_env_names():
    """{GBENCH_* name: [module, ...]} for every name an eval_suites module (plugins and the
    provenance registry's prose excepted) spells out."""
    found = {}
    for path in sorted(SUITES_DIR.glob("*.py")):
        if path.name in _PLUGIN_MODULES or path.name == "canonical_sync.py":
            continue
        for name in re.findall(r"\bGBENCH_[A-Z0-9_]*[A-Z0-9]\b", path.read_text(encoding="utf-8")):
            found.setdefault(name, []).append(path.name)
    return found


def _suite_temperature_names():
    from gbench.runners import eval_suites as E
    names = set(E.SUITES) | set(base.RENAMED_SUITES)
    return {base.temperature_env_var(n) for n in names}


def _run_shaping_names():
    names = set(base.RUN_SHAPING_GLOBAL_ENV)
    for knobs in base.RUN_SHAPING_SUITE_ENV.values():
        names |= set(knobs)
    # a deprecated alias is snapshotted under its canonical knob
    for aliases in base.RUN_SHAPING_ALIASES.values():
        names |= set(aliases)
    return names


def test_every_suite_knob_is_classified_run_shaping_or_not():
    run_shaping = _run_shaping_names()
    temps = _suite_temperature_names()
    unclassified = sorted(
        f"{name} ({', '.join(sorted(set(mods)))})"
        for name, mods in _suite_module_env_names().items()
        if name not in run_shaping and name not in temps and name not in base.NON_RUN_SHAPING_ENV)
    assert not unclassified, unclassified


def test_no_knob_is_both_run_shaping_and_not():
    assert not (_run_shaping_names() & set(base.NON_RUN_SHAPING_ENV))
    assert set(base.NON_RUN_SHAPING_ENV.values()) <= {"infra", "credential", "diagnostic",
                                                       "internal", "removed"}
    classes = set(base.RUN_SHAPING_GLOBAL_ENV.values())
    for knobs in base.RUN_SHAPING_SUITE_ENV.values():
        classes |= set(knobs.values())
    assert classes <= {"dataset", "prompt", "decode", "scoring"}


def test_the_named_run_shaping_knobs_move_the_snapshot(monkeypatch):
    named = {
        "terminal_bench": ["GBENCH_TB_PARSER", "GBENCH_TB_TIMEOUT_MULTIPLIER",
                           "GBENCH_TB_TURN_MAX_TOKENS", "GBENCH_TB_CONTAINER_THREADS"],
        "tau2": ["GBENCH_TAU2_TOP_P", "GBENCH_TAU2_TOP_K", "GBENCH_TAU2_SEED",
                 "GBENCH_TAU2_MAX_ERRORS", "GBENCH_TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM"],
        "toolbench": ["GBENCH_TOOLBENCH_METHOD", "GBENCH_TOOLBENCH_REFERENCE",
                      "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_TOOLBENCH_GROUPS",
                      "GBENCH_TOOLBENCH_MAX_QUERY_COUNT"],
        "wildclawbench": ["GBENCH_WILDCLAWBENCH_THINKING", "GBENCH_WILDCLAWBENCH_TASK"],
        "skillsbench": ["GBENCH_SKILLSBENCH_AGENT", "GBENCH_SKILLSBENCH_CONDITIONS"],
        "scicode": ["GBENCH_SCICODE_WITH_BACKGROUND"],
        "ui_control_osworld": ["GBENCH_OSWORLD_OBS_TYPE", "GBENCH_OSWORLD_LIMIT"],
        "lcb": ["GBENCH_LCB_MIN_DATE", "GBENCH_LCB_INCLUDE_TEST_GEN",
                "GBENCH_LCB_INCLUDE_SELF_REPAIR"],
        "ruler": ["GBENCH_RULER_LENGTHS"],
        "mcp_bench": ["GBENCH_MCP_BENCH_DISTRACTION"],
        "gaia2": ["GBENCH_GAIA2_NUM_RUNS", "GBENCH_GAIA2_MAX_CONCURRENT"],
        "agent_dojo": ["GBENCH_AGENTDOJO_ATTACK"],
        "gaia": ["GBENCH_SEARCH_MAX_RESULTS"],
        "bfcl_v4_agentic": ["GBENCH_BFCL_ENABLE_THINKING"],
    }
    for suite, knobs in named.items():
        for knob in knobs:
            monkeypatch.delenv(knob, raising=False)
            before = base.run_shaping_env([suite], environ={})
            after = base.run_shaping_env([suite], environ={knob: "x"})
            assert before != after, (suite, knob)


def test_a_legacy_alias_moves_the_snapshot_under_its_canonical_name():
    snap = base.run_shaping_env(["tau2"], environ={"TAU2_NUM_TRIALS": "1"})
    assert snap == {"dataset": {"GBENCH_TAU2_NUM_TRIALS": "1"}}
    both = base.run_shaping_env(["tau2"], environ={"TAU2_NUM_TRIALS": "1",
                                                   "GBENCH_TAU2_NUM_TRIALS": "2"})
    assert both == {"dataset": {"GBENCH_TAU2_NUM_TRIALS": "2"}}


def test_every_legacy_alias_of_a_run_shaping_knob_is_declared():
    """suite_env(canonical, *legacy) for a run-shaping canonical: each legacy name is in
    RUN_SHAPING_ALIASES (or is itself a declared knob)."""
    run_shaping = _run_shaping_names()
    missing = []
    for path in sorted(SUITES_DIR.glob("*.py")):
        if path.name in _PLUGIN_MODULES or path.name == "canonical_sync.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "suite_env"):
                continue
            args = [a.value for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if not args or args[0] not in run_shaping:
                continue
            for legacy in args[1:]:
                if legacy not in base.RUN_SHAPING_ALIASES.get(args[0], ()) \
                        and legacy not in run_shaping:
                    missing.append(f"{path.name}: {args[0]} <- {legacy}")
    assert not missing, missing


# --------------------------------------------------------------------------- #
# README CLI reference
# --------------------------------------------------------------------------- #
def test_every_cli_option_has_a_row_in_the_readme_cli_table():
    from gbench import cli
    readme = (REPO / "docs" / "cli-reference.md").read_text(encoding="utf-8")
    start = readme.index("## Exhaustive CLI Argument Reference")
    end = readme.find("\n## ", start + 5)
    end = len(readme) if end < 0 else end
    documented = set()
    for line in readme[start:end].splitlines():
        if line.startswith("| `"):
            documented |= set(re.findall(r"`(--[A-Za-z0-9][A-Za-z0-9-]*)", line.split("|")[1]))
    missing = []
    for action in cli.create_parser()._actions:
        if action.help == argparse.SUPPRESS:
            continue
        longs = [o for o in action.option_strings if o.startswith("--")]
        if not longs or set(longs) & {"--help", "--version"}:
            continue
        if not set(longs) & documented:
            missing.append("/".join(longs))
    assert not missing, f"README CLI table has no row for: {missing}"


def test_the_stress_max_qps_help_names_no_stale_cap():
    from gbench import cli
    action = next(a for a in cli.create_parser()._actions
                  if "--stress-max-qps" in a.option_strings)
    assert "512" not in action.help and "censored at the cap" in action.help


# --------------------------------------------------------------------------- #
# declared dependencies
# --------------------------------------------------------------------------- #
def _project():
    import tomllib
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_harbor_and_loguru_are_declared_where_the_code_needs_them():
    """Harbor runs in the local gbench-terminal-bench image, so no extra installs it on the host
    (installing it next to gbench would move packages other suites import); loguru, networkx
    and jinja2 are declared where the code that imports them needs them."""
    from packaging.requirements import Requirement
    extras = _project()["project"]["optional-dependencies"]
    assert "terminal-bench" not in extras
    assert not any(Requirement(r).name == "harbor" for reqs in extras.values() for r in reqs)
    assert any(Requirement(r).name == "loguru" for r in extras["evals"])
    assert any(Requirement(r).name == "networkx" for r in extras["evals"])
    assert any(Requirement(r).name == "jinja2" for r in extras["dev"])


def test_no_uv_conflict_splits_the_extras():
    project = _project()
    assert "conflicts" not in project.get("tool", {}).get("uv", {})
    assert project["project"]["requires-python"] == ">=3.11"


def test_the_unit_test_workflow_installs_the_extras_in_one_step():
    import yaml
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text())
    runs = [s.get("run", "") for s in wf["jobs"]["pytest"]["steps"]]
    install = next(r for r in runs if '-e ".[dev,evals,omnidocbench]"' in r)
    assert "terminal-bench" not in install
    gate = next(r for r in runs if "-p ci_skip_gate" in r)
    assert "--skip-gate-extras dev,evals,omnidocbench " in gate


# --------------------------------------------------------------------------- #
# CLI: local --thinking needs the reasoning parser; `none` is recorded, never passed
# --------------------------------------------------------------------------- #
def _main(monkeypatch, argv):
    from gbench import cli
    monkeypatch.setattr(sys, "argv", ["gbench", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code


def test_a_local_thinking_eval_run_without_a_reasoning_parser_is_refused(monkeypatch, capsys):
    code = _main(monkeypatch, ["--models", "org/m", "--num-gpus", "1", "--evals", "gsm8k",
                               "--thinking", "--max-output-tokens", "1024", "--dry-run"])
    assert code == 2
    err = capsys.readouterr().err
    assert "--reasoning-parser" in err and "--reasoning-parser none" in err


def test_serve_args_never_pass_the_none_parser_to_vllm():
    from gbench.core import config as C
    cfg = C.BenchmarkConfig()
    cfg.tool_call_parser, cfg.reasoning_parser, cfg.vllm_serve_args = None, "none", []
    assert "--reasoning-parser" not in C.serve_args_for(cfg, "native")
    assert C.serve_settings_record(cfg)["reasoning_parser"] == "none"
    cfg.reasoning_parser = "some_parser"
    assert C.serve_args_for(cfg, "native") == ["--reasoning-parser", "some_parser"]
    assert C.serve_args_for(cfg, "perf") == []


def test_the_host_owner_text_names_both_bind_addresses():
    from gbench.core import config as C
    text = C.GBENCH_OWNED_SERVE_ARGS["--host"]
    assert "GBENCH_SERVER_HOST" in text and "127.0.0.1" in text


def test_provenance_pinned_is_the_registrys_predicate():
    from gbench import cli
    from gbench.runners.eval_suites import canonical_sync as cs
    assert not hasattr(cli, "_PINNED_REVISION_RE")
    for entry in cs.CANONICAL_SYNC.values():
        assert cli.provenance_pinned(entry) == cs.entry_pinned(entry)


def test_list_models_needs_no_removed_registry(capsys):
    import types
    from gbench.cli import _handle_list
    assert _handle_list(types.SimpleNamespace(list_what="models", eval_plugins_dir=None)) == 0
    out = capsys.readouterr().out
    assert "no built-in model list" in out and "unavailable" not in out


# --------------------------------------------------------------------------- #
# routed suite fixes
# --------------------------------------------------------------------------- #
def test_gaia_shards_the_full_set_before_its_limit_draw(monkeypatch, tmp_path):
    import pandas as pd
    import huggingface_hub
    from gbench.runners.eval_suites import gaia
    from gbench.runners.eval_suites.sampling import presharded_for_base, shard_select
    rows = [{"Question": f"q{i}", "Final answer": str(i), "Level": str(1 + i % 3),
             "task_id": f"t{i}", "file_name": ""} for i in range(12)]
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(tmp_path))
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.DataFrame(rows))
    monkeypatch.setattr(gaia, "_hf_token", lambda: None)
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    samples = gaia._load_gaia_samples(limit=2)
    shard_ids = {r["task_id"] for r in shard_select(rows, (2, 3))}
    assert len(samples) == 2 and {m["task_id"] for _, _, m in samples} <= shard_ids
    laid_out = presharded_for_base(samples)
    assert [s for s in shard_select(laid_out, (2, 3))] == samples


def test_mmlu_pro_builds_its_client_with_a_placeholder_key_it_never_sends():
    import asyncio
    import httpx
    from gbench.runners.eval_suites import mmlu_pro
    request = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions",
                            headers={"Authorization": f"Bearer {mmlu_pro.CLIENT_KEY_PLACEHOLDER}"})
    asyncio.run(mmlu_pro._drop_authorization(request))
    assert "authorization" not in request.headers


def test_aa_lcr_thinking_temperature_is_the_served_models_recommendation(monkeypatch):
    from gbench.runners.eval_suites import aa_lcr, ojbench
    monkeypatch.setattr(ojbench, "served_model_recommended_sampling",
                        lambda name=None: ({"temperature": 0.8, "top_p": 0.9}, "org/m/gc.json"))
    sampling = aa_lcr.reasoning_sampling("org/m")
    assert sampling.temperature == 0.8 and "org/m/gc.json" in sampling.reference
    monkeypatch.setattr(ojbench, "served_model_recommended_sampling",
                        lambda name=None: ({}, "org/m/gc.json"))
    assert aa_lcr.reasoning_sampling("org/m").temperature == 0.6

    def unreadable(name=None):
        raise RuntimeError("could not read generation_config.json of 'org/m'")
    monkeypatch.setattr(ojbench, "served_model_recommended_sampling", unreadable)
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*generation_config"):
        aa_lcr.reasoning_sampling("org/m")


def test_quality_refuses_a_placeholder_context_window(tmp_path, monkeypatch):
    """No window listed, no --max-model-len and no config.json read: the pillar stops rather than
    declare the registry's placeholder window to GemmaClaw."""
    import test_evals_endpoint_routing_and_reuse as R
    from gbench.core.models import ModelConfig
    unread = ModelConfig(short_name="m", hf_model_id="org/m")
    assert unread.metadata_source == "defaults" and not unread.total_params_b
    with pytest.raises(RuntimeError, match="served context window is unknown"):
        R._quality_run(tmp_path, monkeypatch, [{"id": "org/m"}], model=unread,
                       remote_endpoint="http://127.0.0.1:1/v1")
    res, _seen, vllm, _m, _r = R._quality_run(
        tmp_path, monkeypatch, [{"id": "org/m", "max_model_len": 16384}], model=unread,
        remote_endpoint="http://127.0.0.1:1/v1")
    assert vllm["contextWindow"] == 16384 and "max_model_len" in res["context_window_source"]
