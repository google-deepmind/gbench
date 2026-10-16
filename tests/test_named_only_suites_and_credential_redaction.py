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

"""Shared-file contracts: registry shape, credential redaction, official max_tokens absence,
the summary's headline column, dependency pins, and the shared docs in step with the code.

  * ROADMAP_SUITES holds only suites with no implementation; ui_control_osworld and swe_lancer
    are registered, left out of `--evals all`, run when named, and hard-error on a missing
    prerequisite;
  * no credential value of the environment reaches a result file, a summary, a CSV or a log
    line, whatever a suite, a reply or an exception message carried;
  * OfficialSampling(send_max_tokens=False) sends no max_tokens of its own;
  * the eval summary carries a suite's headline_metrics;
  * sacrebleu / pycocotools are declared and locked, and the docs match the code.

No model server, judge API or dataset hub is contacted.
"""

import asyncio
import csv
import json
import logging
import os
import re
import tomllib
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO / "gbench" / "runners" / "eval_suites"
DOCS = REPO / "docs"


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# The README was split: these docs hold sections moved verbatim out of the README.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)


def _readme_with(*docs: str) -> str:
    """README.md followed by the moved-out doc(s) that now hold the asserted text."""
    return "\n".join(_read(rel) for rel in ("README.md", *docs))


# --------------------------------------------------------------------------- #
# registry: roadmap-only vs named-only suites
# --------------------------------------------------------------------------- #
def test_roadmap_suites_hold_only_suites_with_no_implementation():
    from gbench.runners.eval_suites import SUITES, ROADMAP_SUITES
    assert set(ROADMAP_SUITES) == {"cybergym", "lmarena_web_agent"}
    for name in ROADMAP_SUITES:
        assert name not in SUITES
        assert not (SUITE_DIR / f"{name}.py").exists(), name


def test_named_only_suites_are_registered_but_not_in_evals_all():
    from gbench.runners import evals as E
    from gbench.runners.eval_suites import SUITES, resolve_suite_name
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    assert set(E.NAMED_ONLY_SUITES) == {"swe_lancer", "ui_control_osworld"}
    in_pillars = {s for _, su in E.BUILTIN_PILLARS for s in su}
    for name in E.NAMED_ONLY_SUITES:
        assert name in SUITES and name in in_pillars and name in CANONICAL_SYNC, name
        assert name not in E.builtin_all_suites(), name
        assert resolve_suite_name(name) == name
    pillar_of = {s: p for p, su in E.BUILTIN_PILLARS for s in su}
    assert pillar_of["ui_control_osworld"].startswith("5.")
    assert pillar_of["swe_lancer"].startswith("3.")
    assert pillar_of["acebench"].startswith("5.")
    assert len(E.builtin_all_suites()) == len(in_pillars) - 2


def _run_evals(tmp_path, monkeypatch, evals, fake_single=None):
    from gbench.core.config import BenchmarkConfig
    from gbench.core.models import ModelFormat
    from gbench.runners.evals import EvalsBenchmarkRunner
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    config = BenchmarkConfig()
    config.evals = evals
    config.remote_endpoint = "http://127.0.0.1:1/v1"
    config.batch_sizes = [2]
    config.skip_existing = False
    config.results_dir = tmp_path / "results"
    config.initialize()
    model = mock.MagicMock()
    model.name = model.short_name = model.hf_model_id = "m"
    called = []

    def default_single(eval_name, model_name, base_url, num_threads):
        called.append(eval_name)
        return {"status": "success", "total_questions": 1, "correct_answers": 1,
                "accuracy": 100.0, "category_accuracy": {}, "thinking": False}

    with mock.patch.object(EvalsBenchmarkRunner, "_run_single_eval",
                           side_effect=fake_single or default_single), \
         mock.patch("gbench.runners.eval_suites.discover_and_register_plugins", return_value={}), \
         mock.patch("gbench.runners.evals._resolve_remote_model_id", return_value="m"), \
         mock.patch("gbench.runners.evals._server_max_model_len", return_value=None):
        results = EvalsBenchmarkRunner(config).run(model, ModelFormat.HF)
    return called, results, config


def test_evals_all_leaves_the_named_only_suites_out(tmp_path, monkeypatch):
    called, _results, _config = _run_evals(tmp_path, monkeypatch, ["all"])
    assert "mmlu" in called and "acebench" in called
    assert "swe_lancer" not in called and "ui_control_osworld" not in called


@pytest.mark.parametrize("name", ["ui_control_osworld", "swe_lancer"])
def test_a_named_only_suite_runs_when_named(name, tmp_path, monkeypatch):
    called, _results, _config = _run_evals(tmp_path, monkeypatch, [name])
    assert called == [name]


@pytest.mark.parametrize("name", ["ui_control_osworld", "swe_lancer"])
def test_the_cli_accepts_a_named_only_suite(name):
    from gbench import cli

    class Passed(Exception):
        pass
    with mock.patch.object(cli, "setup_logging", side_effect=Passed):
        with pytest.raises(Passed):
            cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                      "--max-output-tokens", "1024", "--evals", name])


def test_ui_control_osworld_hard_errors_naming_dev_kvm(monkeypatch):
    from gbench.runners.eval_suites import ui_control_osworld as U
    monkeypatch.setattr(U.shutil, "which", lambda n: "/usr/bin/docker")
    monkeypatch.setattr(U.subprocess, "run", lambda *a, **k: mock.MagicMock(returncode=0))
    monkeypatch.setattr(U.os.path, "exists",
                        lambda p: False if p == "/dev/kvm" else os.path.lexists(p))
    with pytest.raises(RuntimeError, match=r"/dev/kvm") as err:
        U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:1/v1")
    assert str(err.value).startswith("[INFRA REQUIRED]")


def test_swe_lancer_hard_errors_naming_the_task_image(monkeypatch):
    from gbench.runners.eval_suites import swe_lancer as SL
    monkeypatch.setattr(SL.os.path, "exists", lambda p: True)
    monkeypatch.setattr(__import__("shutil"), "which", lambda n: "/usr/bin/docker")

    def fake_run(cmd, *a, **k):
        # docker info succeeds; the harness image is not built
        return mock.MagicMock(returncode=1 if cmd[:3] == ["docker", "image", "inspect"] else 0)
    monkeypatch.setattr(SL.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as err:
        SL.run_swe_lancer("m", "http://127.0.0.1:1/v1", 1)
    text = str(err.value)
    assert text.startswith("[INFRA REQUIRED]") and "gbench-swe-lancer-harness:51052ced" in text
    assert "swe_lancer_harness.Dockerfile" in text
    assert "SWELANCER_RUN" not in text


def test_swe_lancer_subset_run_is_not_leaderboard_comparable(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import swe_lancer as SL
    monkeypatch.setenv("GBENCH_SWE_LANCER_TRAJ_DIR", str(tmp_path))
    monkeypatch.setattr(SL, "check_swe_lancer_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SL, "load_tasks", lambda: [
        {"question_id": "1", "variant": "ic_swe", "set": "diamond", "price": 100.0},
        {"question_id": "2", "variant": "ic_swe", "set": "diamond", "price": 100.0}])
    monkeypatch.setattr(SL, "_run_variant", lambda v, ids, full, *a: {
        "status": "success", "total_questions": len(ids), "correct_answers": 0,
        "accuracy": 0.0, "earned_usd": 0.0, "available_usd": 100.0, "sample_traces": []})
    result = SL.run_swe_lancer("m", "http://127.0.0.1:1/v1", 1, limit=1, variant="ic_swe")
    assert result["leaderboard_comparable"] is False
    assert ("subset run (--eval-limit 1 for the suite: 1 IC SWE task(s))"
            in result["leaderboard_comparable_reason"])
    assert "task types not run: swe_manager" in result["leaderboard_comparable_reason"]


def test_ui_control_osworld_passes_the_endpoint_key_by_name(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import ui_control_osworld as U
    key = "osworldKEY-FAKE-0000000001"
    monkeypatch.setenv("GBENCH_OSWORLD_MODEL_API_KEY", key)
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites", lambda: (True, ""))
    seen = {}

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            seen["cmd"], seen["env"] = list(cmd), dict(k.get("env") or {})
            # the orchestrator writes its summary into this run's own directory
            run_dir = next(c.split("=", 1)[1] for c in cmd
                           if c.startswith("OSWORLD_RUN_DIR="))
            (Path(run_dir) / "ui_control_osworld_summary.json").write_text(json.dumps(
                {"success_rate": 0.5, "per_domain": {}, "n_selected": 2, "n_scored": 2,
                 "n_missing": 0}), encoding="utf-8")
        return mock.MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(U.subprocess, "run", fake_run)
    result = U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:1/v1")
    assert result["accuracy"] == 50.0
    assert not any(key in part for part in seen["cmd"])
    assert "GBENCH_MODEL_API_KEY" in seen["cmd"]
    # The orchestrator holds the docker socket, so it gets the placeholder by name; the real key
    # stays with gbench's relay, which it reaches instead of the endpoint.
    assert seen["env"]["GBENCH_MODEL_API_KEY"] == U.MODEL_KEY_PLACEHOLDER
    assert key not in json.dumps(seen["env"])
    assert f"GBENCH_MODEL_BASE_URL={result['model_endpoint_relay']['listens']}" in seen["cmd"]


def test_docs_describe_the_named_only_suites():
    readme = _readme_with("docs/eval-suites.md", "docs/cli-reference.md")
    assert "| | `swe_lancer` |" in readme and "| | `ui_control_osworld` |" in readme
    assert "`all` runs every built-in suite except `swe_lancer` and `ui_control_osworld`" in readme
    para = readme[readme.index("Two suites are **deregistered"):][:1200]
    assert "`lmarena_web_agent`" in para and "`cybergym`" in para
    assert "`ui_control_osworld`" in para and "`swe_lancer`" in para
    for rel in ("docs/evals/swe_lancer.md", "docs/evals/ui_control_osworld.md"):
        text = _read(rel)
        assert "Status: registered, not in `--evals all`" in text, rel
        assert "deregistered" not in text and "SWELANCER_RUN" not in text, rel
        assert "--max-output-tokens" in text.split("## Run", 1)[-1], rel
    roadmap = _read("docs/evals_roadmap.md")
    for name in ("swe_lancer", "ui_control_osworld"):
        entry = roadmap[roadmap.index(f": `{name}`"):][:4000]
        assert "NAMED_ONLY_SUITES" in entry and "ROADMAP_SUITES" not in entry, name
    resources = _read("docs/RESOURCES.md")
    for line in resources.splitlines():
        if "SWE-Lancer" in line or "OSWorld" in line:
            assert "deregistered" not in line and "not in `--evals all`" in line, line
    assert "SWE_LANCER_RUN" not in _read("docs/evals/env-knobs.md")
    assert "is deregistered" not in _read("docs/evals/toolchains.md")


# --------------------------------------------------------------------------- #
# credential redaction
# --------------------------------------------------------------------------- #
FAKE_SECRETS = {
    "GEMINI_API_KEY": "AIzaFAKEgemini0000000001",
    "GEMINI_API_KEYS": "AIzaFAKElistA00000000001,AIzaFAKElistB00000000002",
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    "OPENAI_API_KEY": "sk-" + "FAKEopenai000000000001",
    "SERPAPI_API_KEY": "serpFAKE0000000000000001",
    "HF_TOKEN": "hf_FAKEhuggingface000001",
    "ANTHROPIC_API_KEY": "sk-ant-FAKE000000000001",
    "GBENCH_TOOLBENCH_SIMULATOR_KEY": "simFAKE00000000000000001",
    "SOME_VENDOR_API_KEY": "vendorFAKE00000000000001",
    "SOME_SERVICE_TOKEN": "tokFAKE00000000000000001",
}
ALL_FAKE_VALUES = ([v for v in FAKE_SECRETS.values()]
                   + FAKE_SECRETS["GEMINI_API_KEYS"].split(","))


@pytest.fixture
def fake_secrets(monkeypatch):
    for name, value in FAKE_SECRETS.items():
        monkeypatch.setenv(name, value)
    # knobs that look credential-named but hold short or numeric values stay untouched
    monkeypatch.setenv("GBENCH_VIDEO_SOFT_TOKENS", "24960000")
    monkeypatch.setenv("PLACEHOLDER_API_KEY", "EMPTY")
    return FAKE_SECRETS


def test_secret_values_cover_the_known_names_and_the_suffix_rule(fake_secrets):
    values = base.secret_values()
    for value in ALL_FAKE_VALUES:
        assert value in values, value
    assert "24960000" not in values and "EMPTY" not in values
    assert values == sorted(values, key=lambda v: (-len(v), v))
    assert base.is_secret_env_name("GEMINI_API_KEYS") and base.is_secret_env_name("X_TOKEN")
    assert not base.is_secret_env_name("GBENCH_EVAL_LIMIT")


def test_redaction_is_copy_on_write(fake_secrets):
    clean = {"a": ["x", {"b": 1}], "c": ("y",)}
    assert base.redact_secrets(clean) is clean
    dirty = {"a": ["x", {"b": FAKE_SECRETS["HF_TOKEN"]}], "keep": clean}
    out = base.redact_secrets(dirty)
    assert out["a"][1]["b"] == base.REDACTED and out["keep"] is clean
    assert dirty["a"][1]["b"] == FAKE_SECRETS["HF_TOKEN"]          # the input is not mutated
    escaped = json.dumps({"k": 'say "' + FAKE_SECRETS["OPENAI_API_KEY"] + '"'})
    assert FAKE_SECRETS["OPENAI_API_KEY"] not in base.redact_text(escaped)


def test_url_userinfo_is_redacted(monkeypatch):
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://user:passFAKE123456@db:27017/x")
    out = base.redact_text("connecting to mongodb://user:passFAKE123456@db:27017/x")
    assert "passFAKE123456" not in out


def _scan(root: Path):
    hits = []
    for path in root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            for value in ALL_FAKE_VALUES:
                if value.encode() in data:
                    hits.append((str(path.relative_to(root)), value))
    return hits


def test_no_credential_reaches_any_file_a_run_writes(fake_secrets, tmp_path, monkeypatch):
    """A native suite whose prompt and reply carry keys, a suite with its own result dict and
    trace file, a suite that raises with a key in the message, the summaries, the CSV and the
    run log: every file under the results directory is scanned for every fake key value."""
    from gbench import cli
    from gbench.runners.eval_suites import SUITES
    secret_blob = " ".join(ALL_FAKE_VALUES)

    async def echo_send(**k):
        async with k["semaphore"]:
            pass
        text = k["messages"][-1]["content"]
        return base.Reply(f"echo {text} {FAKE_SECRETS['GEMINI_API_KEY']}", None, "stop",
                          completion_tokens=3)

    def native_suite(**kw):
        samples = [([{"role": "user", "content": f"q{i} key={secret_blob}"}], "A",
                    {"category": "c", "note": FAKE_SECRETS["SERPAPI_API_KEY"]}) for i in range(3)]
        with mock.patch.object(base, "_send_single_request", echo_send):
            return base.run_eval_suite(eval_name="mmlu", model_name=kw["model_name"],
                                       base_url=kw["base_url"], concurrency=2, samples=samples,
                                       eval_fn=lambda resp, gold: False)

    def own_result_suite(**kw):
        run_dir = Path(os.environ["GBENCH_RESULTS_DIR"])
        # a suite-written trace file goes through LogManager-style JSON as well
        base_result = {"status": "success", "total_questions": 1, "correct_answers": 0,
                       "accuracy": 0.0, "category_accuracy": {},
                       "command": ["docker", "run", "-e",
                                   f"GEMINI_API_KEY={FAKE_SECRETS['GEMINI_API_KEY']}"],
                       "harness_log": f"Authorization: Bearer {FAKE_SECRETS['OPENAI_API_KEY']}",
                       "headline_metrics": {"violation_at_5": 1.0,
                                            "note": FAKE_SECRETS["SOME_SERVICE_TOKEN"]}}
        assert run_dir.is_dir()
        return base_result

    def raising_suite(**kw):
        raise RuntimeError("[INFRA REQUIRED] Command ['docker', 'run', '-e', "
                           f"'NASA_API_KEY={FAKE_SECRETS['SOME_VENDOR_API_KEY']}'] timed out; "
                           f"key list {FAKE_SECRETS['GEMINI_API_KEYS']}")

    # the real ui_control_osworld runner, its orchestrator summary carrying a key the harness logged
    from gbench.runners.eval_suites import ui_control_osworld as U
    vm_dir = tmp_path / "vm"
    vm_dir.mkdir()
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(vm_dir))
    monkeypatch.setenv("GBENCH_OSWORLD_MODEL_API_KEY", FAKE_SECRETS["SOME_VENDOR_API_KEY"])
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites", lambda: (True, ""))

    def osworld_docker(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            run_dir = next(c.split("=", 1)[1] for c in cmd
                           if c.startswith("OSWORLD_RUN_DIR="))
            (Path(run_dir) / "ui_control_osworld_summary.json").write_text(json.dumps(
                {"success_rate": 1.0, "per_domain": {"os": 1.0}, "n_selected": 1, "n_scored": 1,
                 "n_missing": 0, "agent_log": f"POST with key {FAKE_SECRETS['OPENAI_API_KEY']}"}),
                encoding="utf-8")
        return mock.MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(U.subprocess, "run", osworld_docker)

    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    log_path = tmp_path / "results" / "run.log"
    log_path.parent.mkdir(parents=True)
    handler = logging.FileHandler(log_path)
    root = logging.getLogger()
    root.addHandler(handler)
    old_factory = logging.getLogRecordFactory()
    # a suite's own logger with its own file handler (an agent log): redacted at record creation
    agent_log = tmp_path / "results" / "agent.log"
    agent_logger = logging.getLogger("gbench.test.agent")
    agent_handler = logging.FileHandler(agent_log)
    old_level = root.level
    root.setLevel(logging.INFO)
    try:
        base.install_log_redaction()
        agent_logger.addHandler(agent_handler)
        agent_logger.propagate = False
        agent_logger.warning("model call with Authorization: Bearer %s",
                             FAKE_SECRETS["OPENAI_API_KEY"])
        with mock.patch.dict(SUITES, {"mmlu": native_suite, "swe_lancer": own_result_suite,
                                      "mcp_bench": raising_suite}):
            from gbench.core.config import BenchmarkConfig
            from gbench.core.models import ModelFormat
            from gbench.runners.evals import EvalsBenchmarkRunner
            config = BenchmarkConfig()
            config.evals = ["mmlu", "swe_lancer", "ui_control_osworld", "mcp_bench"]
            config.remote_endpoint = "http://127.0.0.1:1/v1"
            config.batch_sizes = [2]
            config.skip_existing = False
            config.eval_max_output_tokens = 256
            config.results_dir = tmp_path / "results"
            config.initialize()
            model = mock.MagicMock()
            model.name = model.short_name = model.hf_model_id = "m"
            with mock.patch("gbench.runners.eval_suites.discover_and_register_plugins",
                            return_value={}), \
                 mock.patch("gbench.runners.evals._resolve_remote_model_id", return_value="m"), \
                 mock.patch("gbench.runners.evals._server_max_model_len", return_value=None):
                results = EvalsBenchmarkRunner(config).run(model, ModelFormat.HF)
            lm = config.log_manager
            lm.save_summary({"results": results, "argv_note": FAKE_SECRETS["HF_TOKEN"]})
            lm.save_metadata({"endpoint": "http://u:" + FAKE_SECRETS["ANTHROPIC_API_KEY"] + "@h"})
            lm.save_csv("extra.csv", [{"a": FAKE_SECRETS["GBENCH_TOOLBENCH_SIMULATOR_KEY"]}], ["a"])
            lm.save_log(lm.results_dir / "logs" / "harness.log",
                        FAKE_SECRETS["OPENAI_API_KEY"], FAKE_SECRETS["SERPAPI_API_KEY"])
            ok = [r for r in results if r.get("status") != "error"]
            bad = [r for r in results if r.get("status") == "error"]
            cli._save_eval_summary_csv(lm.results_dir, ok, bad, [], {})
            logging.getLogger("gbench.test").error(
                "harness said %s", FAKE_SECRETS["SOME_VENDOR_API_KEY"])
            try:
                raise ValueError(f"boom {FAKE_SECRETS['HF_TOKEN']}")
            except ValueError:
                logging.getLogger("gbench.test").exception("failed")
    finally:
        handler.flush()
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()
        agent_logger.removeHandler(agent_handler)
        agent_logger.propagate = True
        agent_handler.close()
        logging.setLogRecordFactory(old_factory)
    assert {r["eval_name"] for r in results} == {"mmlu", "swe_lancer", "ui_control_osworld",
                                                  "mcp_bench"}
    osworld = next(r for r in results if r["eval_name"] == "ui_control_osworld")
    assert osworld["status"] == "success" and base.REDACTED in osworld["raw_summary"]["agent_log"]
    written = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "results").rglob("*")
                     if p.is_file())
    assert any(w.endswith("eval_summary.csv") for w in written), written
    assert any("eval_mcp_bench_" in w for w in written), written
    assert base.REDACTED in log_path.read_text(encoding="utf-8")
    assert base.REDACTED in agent_log.read_text(encoding="utf-8")
    # (vm/ holds the stand-in harness's own summary file, outside what gbench writes)
    assert _scan(tmp_path / "results") == []


# --------------------------------------------------------------------------- #
# official absence of max_tokens
# --------------------------------------------------------------------------- #
class _FakeResponse:
    status = 200

    async def json(self):
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self):
        self.payloads = []

    def post(self, url, json=None):
        self.payloads.append(json)
        return _FakeResponse()


def test_a_request_without_a_budget_sends_no_max_tokens():
    session = _FakeSession()
    reply = asyncio.run(base._send_single_request(
        session=session, api_url="http://x/v1/chat/completions", model_name="m",
        messages=[{"role": "user", "content": "hi"}], extra_payload={},
        semaphore=asyncio.Semaphore(1), pbar=None, max_output_tokens=None, temperature=None))
    assert reply.text == "ok"
    assert "max_tokens" not in session.payloads[0] and "temperature" not in session.payloads[0]


def _capture_budget_run(official, monkeypatch, window=None, knob=512):
    sent = []

    async def send(**k):
        async with k["semaphore"]:
            pass
        sent.append(k["max_output_tokens"])
        return base.Reply("x", None, "stop", completion_tokens=1)
    base._RUN_KNOBS["max_output_tokens"] = knob
    base._RUN_KNOBS["max_model_len"] = window
    if window:
        monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (100, 100, "server"))
    with mock.patch.object(base, "_send_single_request", send):
        result = base.run_eval_suite(eval_name="aime", model_name="m", base_url="http://x",
                                     concurrency=1,
                                     samples=[([{"role": "user", "content": "q"}], "1", {})],
                                     eval_fn=lambda r, g: True, official_sampling=official)
    return sent, result


def test_send_max_tokens_false_sends_the_room_left_and_ignores_the_run_budget(monkeypatch):
    official = base.OfficialSampling(reference="upstream agent (no max_tokens)",
                                     send_temperature=False, send_max_tokens=False)
    sent, result = _capture_budget_run(official, monkeypatch, window=None)
    assert sent == [None]
    assert "--max-output-tokens 512 not applied" in result["max_output_tokens_note"]
    assert result["official_sampling"]["send_max_tokens"] is False
    assert "not sent" in result["official_sampling"]["max_tokens"]
    # A known window changes nothing on the wire: upstream sends no max_tokens, so neither does
    # gbench (the prompt count only refuses a prompt that cannot fit).
    sent, result = _capture_budget_run(official, monkeypatch, window=4096)
    assert sent == [None] and result["sample_traces"][0]["max_tokens_sent"] is None


def test_the_default_still_sends_the_run_budget(monkeypatch):
    official = base.OfficialSampling(reference="upstream", temperature=0.0)
    sent, result = _capture_budget_run(official, monkeypatch, window=None)
    assert sent == [512]
    assert "max_output_tokens_note" not in result
    assert "send_max_tokens" not in result["official_sampling"]


def test_resolve_official_sampling_refuses_non_sampling_params():
    bad = base.OfficialSampling(reference="x", temperature=0.0, params={"max_tokens": 5})
    with pytest.raises(ValueError, match="max_tokens"):
        base.resolve_official_sampling("scicode", bad)
    ok = base.OfficialSampling(reference="x", temperature=0.0, params={"top_p": 1.0})
    assert base.resolve_official_sampling("scicode", ok) == (0.0, "default")


# --------------------------------------------------------------------------- #
# the summary shows every headline number
# --------------------------------------------------------------------------- #
def test_the_eval_summary_csv_carries_headline_metrics(tmp_path):
    from gbench import cli
    row = {"eval_name": "cimemories", "model_short": "m", "format": "hf", "status": "success",
           "total_questions": 10, "correct_answers": 7, "accuracy": 70.0,
           "category_accuracy": {"a": {"total": 5, "correct": 3, "accuracy": 60.0},
                                 "b": {"total": 5, "correct": 4, "accuracy": 80.0}},
           "headline_metrics": {"violation_at_5": 12.5, "completeness": 70.0}}
    path = cli._save_eval_summary_csv(tmp_path, [row], [], [("1. X", ["cimemories"])], {})
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    overall = next(r for r in rows if r["Subcategory"] == "OVERALL")
    assert overall["Headline Metrics"] == "violation_at_5=12.5; completeness=70.0"
    assert all(r["Headline Metrics"] == "" for r in rows if r["Subcategory"] in ("a", "b"))


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
def test_sacrebleu_and_pycocotools_are_declared_and_locked():
    project = tomllib.loads(_read("pyproject.toml"))
    evals = project["project"]["optional-dependencies"]["evals"]
    assert "sacrebleu==2.6.0" in evals and "pycocotools==2.0.11" in evals
    assert "json_repair" in evals
    data = project["tool"]["setuptools"]["package-data"]["gbench"]
    assert "third_party/_json_repair_vendor/LICENSE" in data
    assert (REPO / "gbench" / "third_party" / "_json_repair_vendor" / "LICENSE").is_file()
    lock = _read("uv.lock")
    for name, version in (("sacrebleu", "2.6.0"), ("pycocotools", "2.0.11"),
                          ("portalocker", "4.4.0")):
        block = lock[lock.index(f'name = "{name}"\nversion'):][:120]
        assert f'version = "{version}"' in block, name
    for spec in ('{ name = "sacrebleu", marker = "extra == \'evals\'", specifier = "==2.6.0" }',
                 '{ name = "pycocotools", marker = "extra == \'evals\'", specifier = "==2.0.11" }'):
        assert spec in lock, spec


# --------------------------------------------------------------------------- #
# shared docs and provenance in step with the code
# --------------------------------------------------------------------------- #
def test_shared_docs_state_the_temperature_rule():
    """README, the --temperature help, plugins.md and scaffold-versioning.md state the rule:
    1.0 with --thinking, 0.0 without, for every suite; --temperature and the per-suite env
    variable override it; no official no-think temperature is applied."""
    readme = _readme_with(*_README_SPLIT_DOCS)  # the --temperature row: docs/cli-reference.md
    row = next(l for l in readme.splitlines() if l.startswith("| `--temperature` |"))
    assert "| `1.0` with `--eval-thinking`, else `0.0`, for every suite |" in row
    assert "official or upstream temperature is never applied" in row
    assert "GBENCH_<EVAL>_TEMPERATURE" in row
    for gone in ("the suite's official temperature", "send no temperature at all",
                 "canonical-comparable", "temperature 0.1"):
        assert gone not in readme, gone
    src = _read("gbench/cli.py")
    help_text = src[src.index('"--temperature",'):src.index('"--attempt-count",')]
    flat = " ".join(help_text.replace('"', " ").split())
    assert "EVERY suite: 1.0 with --thinking, 0.0 without it" in flat
    assert "upstream temperature is never applied" in flat
    assert "declares its upstream harness's" not in flat
    plugins = " ".join(_read("docs/evals/plugins.md").split())
    assert "the same for every suite: 1.0 with `--eval-thinking`, else 0.0" in plugins
    assert "own no-think default" not in plugins
    scaffold = _read("docs/scaffold-versioning.md")
    assert "1.0 with `--thinking`, else 0.0" in scaffold
    assert "think-aware default" not in scaffold


def test_env_knobs_follow_the_code():
    knobs = _read("docs/evals/env-knobs.md")
    section = knobs[knobs.index("- Suites that declare official sampling"):
                    knobs.index("- Renamed suites still read")]
    # The maintainer's rule (base.resolve_temperature): 1.0 with --thinking, 0.0 without, for
    # every suite; official temperatures are never applied, so none is listed per suite.
    policy = knobs[knobs.index("- Temperature (the maintainer's rule"):
                   knobs.index("- Suites that declare official")]
    flat = " ".join(policy.split())
    assert "a `--thinking` run samples the model under test at temperature 1.0" in flat
    assert "a run without `--thinking` at 0.0" in flat
    assert "official or upstream temperature is never applied" in flat
    assert "`GBENCH_<SUITE>_TEMPERATURE`, then `--temperature`" in flat
    official = section[:section.index("- No max_tokens sent")]
    for gone in ("No temperature sent", "without `--thinking`; 1.0 with it", "  - 0:",
                 "BFCL's 0.001", "0.5 without", "the no-think temperature"):
        assert gone not in official, gone
    for gone in ("upstream's 0.0", "ARE paper's agent temperature", "BFCL's 0.001",
                 "ACEBench's 0.0", "official temperature)"):
        assert gone not in knobs, gone
    from gbench.runners.eval_suites import tau_common
    tau = knobs[knobs.index("- tau2 and tau3 (shared"):knobs.index("- wildclawbench:")]
    for name in tau_common._FORWARD_NAMES:
        assert f"`{name}`" in tau or name.lower() == name, name
    for name in ("GBENCH_TAU2_OPENAI_API_KEY", "GBENCH_TAU2_OPENAI_BASE_URL", "LITELLM_LOG",
                 "`HTTP_PROXY`", "GEMINI_OPENAI_BASE_URL"):
        assert name in tau, name
    knob_list = tau[:tau.index("The\n  assistant temperature")]
    assert "GBENCH_TAU2_EMBED_MODEL" not in knob_list
    assert ("except credential-named ones, the removed\n  knobs below, `GBENCH_TAU2_IMAGE` and "
            "`GBENCH_TAU2_SAVE_TRACES`") in tau
    assert "A user simulator or judge that is not a `gemini/` model" in tau
    assert "gemini-embedding-001" in tau and "OPENROUTER_API_KEY" not in tau
    assert "`GBENCH_TAU2_RETRIEVAL_CONFIG`" not in knob_list
    for knob in ("GBENCH_LCB_INCLUDE_SELF_REPAIR", "GBENCH_BIGCODEBENCH_CPUS",
                 "chat_completions-3025a531af74.py", "GBENCH_ACEBENCH_TEMPERATURE",
                 "GBENCH_BFCL_SEARCH_LOG", "GBENCH_T_EVAL_DEVICE", "toolbench_baselines/",
                 "GBENCH_TOOLBENCH_REFERENCE` (SoWR reference", "GBENCH_SWE_LANCER_IMAGE"):
        assert knob in knobs, knob
    osworld = knobs.index("- ui_control_osworld (not in `--evals all`")
    assert knobs.index("### Pillar 5") < osworld < knobs.index("### Pillar 6")


def test_readme_rows_follow_the_code():
    readme = _readme_with("docs/cli-reference.md", "docs/eval-suites.md",
                          "docs/environment-variables.md")
    row = next(l for l in readme.splitlines() if l.startswith("| `--eval-categories` |"))
    assert row.endswith("Only with `--evals bfcl`, `bfcl_v4_agentic`, `swe_bench_live` or `all`. |")
    row = next(l for l in readme.splitlines() if l.startswith("| `--max-output-tokens` |"))
    assert "max_output_tokens_note" in row and "Their one-shot modes apply it." in row
    row = next(l for l in readme.splitlines() if "| `acebench` |" in l)
    assert row.startswith("| **5. Tool Use & Agentic Workflows** | `acebench` |")
    assert "GEMINI_API_KEY" in row
    for needle in ("WB-Elo (Raw and LC)", "newstest2019", "pycocotools COCOeval",
                   "official OS-Copilot release", "+ self-repair, reported separately",
                   "`tau2` / `tau3` (user simulator and judge; tau3's retrieval embeddings with "
                   "`gemini-embedding-001`)",
                   "SoWR against the published ChatGPT-CoT answers",
                   "passing MCP-Bench's own connection check"):
        assert needle in readme, needle
    sandboxes = next(l for l in readme.splitlines() if l.startswith("| `--sandboxes` |"))
    assert "`swe_lancer`" in sandboxes


def test_notice_third_party_and_resources_follow_the_code():
    notice = _read("NOTICE")
    for needle in ("gbench/third_party/_json_repair_vendor/",
                   "51d82532c3b8263782a301eb7904c75dce5fee8c0d1aba490287fc0ab779ac50",
                   "Stefano Baccianella", "compute_elo_based_on_votes", "wb_elo_results.LM=-1.json",
                   "request_with_images", "njucckevin/SeeClick", "stabletoolbench/baselines",
                   "gbench/third_party/_acebench_vendor/", "execute_agent_func_call",
                   "config.yaml function descriptions"):
        assert needle in notice, needle
    assert "(deregistered suite)" not in notice
    assert not [c for c in notice if ord(c) > 127]
    tpd = _read("docs/THIRD_PARTY_DATASETS.md")
    for needle in ("`OS-Copilot/ScreenSpot-v2`", "`stabletoolbench/baselines`", "all_stat.json",
                   "images.cocodataset.org", "data.statmt.org/wmt19", "`data_all/data_zh`",
                   "*_{0_shot,bm25,gpt_index,oracle}.jsonl", "WB-Elo pool replies of 63 models"):
        assert needle in tpd, needle
    assert "detection-datasets/coco" not in tpd and "`wmt/wmt19`" not in tpd
    assert "`screenspot_pointing`), `allenai/WildBench`" not in tpd
    resources = _read("docs/RESOURCES.md")
    for needle in ("OS-Copilot/ScreenSpot-v2", "data.statmt.org/wmt19", "val2017.zip",
                   "spaces/allenai/WildBench", "stabletoolbench/baselines", "thenlper/gte-large-zh",
                   "text-embedding-3-large", "Oracle retriever settings"):
        assert needle in resources, needle
    assert "detection-datasets/coco" not in resources and "wmt/wmt19" not in resources
    tools = _read("docs/evals/toolchains.md")
    for needle in ("pip install sacrebleu==2.6.0", "pip install pycocotools==2.0.11",
                   "docker-ce-cli 5:29.8.1-1~ubuntu.24.04~noble", "swebench_v5.constraints.txt",
                   "tau3's retrieval embeddings with gemini-embedding-001",
                   "Network access to huggingface.co",
                   "acebench's user simulator"):
        assert needle in tools, needle


def test_roadmap_follows_the_code():
    roadmap = _read("docs/evals_roadmap.md")
    assert "**`wildbench` WB-Elo.**" not in roadmap        # implemented in wildbench.py
    assert "**`lcb` self-repair scenario.**" not in roadmap
    assert "`multipl_e` DSPy JSONAdapter retry" not in roadmap
    from gbench.runners.eval_suites import wildbench
    assert wildbench.WB_ELO_LENGTH_MARGINS == {"WB_Elo": -1, "WB_Elo_LC": 1500}


def test_canonical_sync_follows_the_updates():
    from gbench.runners.evals import BUILTIN_PILLARS
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    # (SUITES itself may hold plugins other tests registered in this session)
    assert set(CS) == {s for _, su in BUILTIN_PILLARS for s in su} | {"custom_jsonl"}
    text = json.dumps(CS)
    assert "--rewrite_reports" not in text and "alltools-gemini" not in text
    assert "WB-Elo: the 63 pool models" in CS["wildbench"]["method"]
    assert CS["wildbench"]["method"].index("WB_score.task_macro (headline") >= 0
    assert "newstest2019" in CS["i18n_translate"]["dataset"]
    assert "OS-Copilot/ScreenSpot-v2" in CS["screenspot_pointing"]["dataset"]
    assert "pycocotools" in CS["bundled_detection"]["upstream"]
    assert "0.44.1" in CS["beam_128k"]["method"]
    assert "self-repair" in CS["lcb"]["method"] and "self-repair not run" not in CS["lcb"]["method"]
    assert "text-embedding-3-large" in CS["tau3"]["method"]
    assert "gemini-embedding-001" in CS["tau3"]["method"]
    assert "GBENCH_TAU2_OPENAI" not in text
    assert "sanitize without --calibrate" in CS["bigcodebench"]["method"]
    assert "no --force-temperature by default" in CS["livebench"]["method"]
    assert "printed in full" in CS["putnam_formal"]["method"]
    assert "infra_required" in CS["bfcl"]["method"].split("agentic delegated to bfcl_v4_agentic")[1]
    assert CS["ui_control_osworld"]["revision"] == "fc31a9049664292fcb35d6e501ee1dc839f2cf6d"
    assert "SimpleAgentSolver" in CS["swe_lancer"]["method"]
    for name, entry in CS.items():
        assert not [c for c in json.dumps(entry) if ord(c) > 127], name
        assert not re.search(r"\b[Ww]ave ?\d|\bW2[a-f]?\b", json.dumps(entry)), name


def test_the_putnam_test_files_are_named_after_the_suite():
    # docs/evals/putnam_formal.md names the suite's tests by the glob tests/test_*putnam*.py, so a
    # file that glob collects holds putnam tests only and is named after the suite.
    assert (REPO / "tests" / "test_putnam_formal_payload.py").exists()
    matched = sorted(p.name for p in (REPO / "tests").glob("test_*putnam*.py"))
    assert matched and all(n.startswith("test_putnam") for n in matched), matched


def test_env_knobs_names_every_redacted_credential_variable():
    knobs = _read("docs/evals/env-knobs.md")
    bullet = knobs[knobs.index("- Credential values never reach"):]
    bullet = bullet[:bullet.index("gbench/runners/eval_suites/base.py`)")]
    assert [n for n in base.SECRET_ENV_NAMES if f"`{n}`" not in bullet] == []
    assert base.REDACTED in bullet


def test_new_text_in_the_shared_docs_is_ascii():
    for rel in ("docs/evals/env-knobs.md", "docs/evals/toolchains.md", "NOTICE",
                "docs/THIRD_PARTY_DATASETS.md", "docs/evals/swe_lancer.md"):
        text = _read(rel)
        assert not [c for c in text if ord(c) > 127], rel
