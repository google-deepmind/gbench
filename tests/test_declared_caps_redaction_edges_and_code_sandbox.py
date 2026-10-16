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

"""Shared-file contracts: one token-limit rule for every declared cap, the credential backstop's
edge cases (JSON integrity, files still being written, the CLI finally), the code sandbox, and
the shared docs and registries in step with the code.

No model server, judge API or dataset hub is contacted; every file is written under tmp_path.
"""

import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
KEY = "sk-" + "FAKEtest00000000key0001"


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _run(official=None, *, thinking=False, knob=4096, meta=None, finish="stop", n=1,
         health_text=None):
    """run_eval_suite with a fake transport; returns (requests seen, result)."""
    seen = []
    text = health_text or ("x " * 10)

    async def send(**k):
        async with k["semaphore"]:
            pass
        seen.append(k)
        return base.Reply(text, None, finish, completion_tokens=5)
    base._RUN_KNOBS["max_output_tokens"] = knob
    base._RUN_KNOBS["max_model_len"] = None
    samples = [([{"role": "user", "content": f"q{i}"}], "1", dict(meta or {})) for i in range(n)]
    with mock.patch.object(base, "_send_single_request", send):
        result = base.run_eval_suite(eval_name="aime", model_name="m", base_url="http://x",
                                     concurrency=1, samples=samples, eval_fn=lambda r, g: True,
                                     thinking=thinking, official_sampling=official)
    return seen, result


CAP = base.OfficialSampling(reference="upstream sampler (max_tokens=100)", temperature=0.0,
                            max_tokens=100)
NONE_SENT = base.OfficialSampling(reference="agent (no max_tokens)", send_max_tokens=False)


# --------------------------------------------------------------------------- #
# token-limit policy: one rule for declared caps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("declared", [256, 32768])
def test_thinking_replaces_a_declared_cap_like_an_official_sampling_cap(declared):
    seen, res = _run(None, thinking=True, knob=4096, meta={"max_tokens": declared})
    assert [k["max_output_tokens"] for k in seen] == [4096]
    assert res["official_cap_not_applied_to_reasoning"] == {
        "declared_max_tokens": [declared], "samples": 1, "sent": 4096}
    # the same rule as an OfficialSampling.max_tokens of that size
    seen2, _ = _run(CAP._replace(max_tokens=declared), thinking=True, knob=4096)
    assert [k["max_output_tokens"] for k in seen2] == [4096]


def test_a_suite_that_sends_no_max_tokens_cannot_declare_one_per_request():
    with pytest.raises(ValueError, match="send_max_tokens=False"):
        _run(NONE_SENT, meta={"max_tokens": 2048})
    # with --thinking the declared cap is replaced by the run's budget, which is none
    seen, res = _run(NONE_SENT, thinking=True, meta={"max_tokens": 2048})
    assert [k["max_output_tokens"] for k in seen] == [None]
    assert res["official_cap_not_applied_to_reasoning"]["sent"] is None


def test_replies_at_an_official_cap_never_count_as_truncations():
    loop = "the same words again and again " * 400
    seen, res = _run(CAP, knob=65536, finish="length", n=4, health_text=loop)
    assert res["looping_responses"] == 4
    assert res["replies_at_official_max_tokens"] == 4
    assert res["genuinely_truncated"] == 0
    assert res["status"] == "success"
    seen, res = _run(CAP, knob=65536, finish="length", n=4)
    assert res["replies_at_official_max_tokens"] == 4 and res["genuinely_truncated"] == 0
    # the same replies at a budget that is not an official cap are genuine truncations
    seen, res = _run(None, knob=65536, finish="length", n=4)
    assert res["genuinely_truncated"] == 4 and res["status"] == "completed_with_errors"


def test_reasoning_effort_reaches_the_request():
    assert "reasoning_effort" in base.ALLOWED_API_KEYS


# --------------------------------------------------------------------------- #
# credential redaction: JSON integrity and files still being written
# --------------------------------------------------------------------------- #
def test_a_key_ending_in_a_backslash_leaves_valid_json(tmp_path, monkeypatch):
    from gbench.utils.log_manager import write_redacted_json
    odd = "Zk9pQ2x5bWFAKE\\"
    monkeypatch.setenv("ODD_API_KEY", odd)
    path = write_redacted_json(tmp_path / "r.json", {"error": f"auth failed for {odd}"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"error": f"auth failed for {base.REDACTED}"}
    # redact_tree on JSON another process wrote: same result
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"e": odd}), encoding="utf-8")
    base.redact_tree(tmp_path)
    assert json.loads(other.read_text(encoding="utf-8")) == {"e": base.REDACTED}


def test_a_live_file_written_without_append_gets_no_nul_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    log = tmp_path / "harness.log"
    with open(log, "w") as writer:                       # O_TRUNC, no O_APPEND
        writer.write(f"args: api_key={KEY}\n")
        writer.flush()
        assert base.redact_file(str(log)) is True
        writer.write("next line\n")
        writer.flush()
    data = log.read_bytes()
    assert b"\x00" not in data and KEY.encode() not in data
    assert data.startswith(f"args: api_key={base.REDACTED}\n".encode())
    assert data.endswith(b"next line\n")


@pytest.mark.parametrize("live", [True, False])
def test_bytes_appended_during_a_rewrite_are_kept(tmp_path, monkeypatch, live):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    log = tmp_path / "server.log"
    log.write_text(f"start {KEY}\n", encoding="utf-8")
    if not live:
        old = time.time() - 10 * base.LIVE_FILE_WINDOW_S
        os.utime(log, (old, old))
    real = base.redact_bytes

    def appending(data, secrets=None):
        # another process appends between redact_file's read and its rewrite
        with open(log, "ab") as other:
            other.write(b"appended meanwhile\n")
        return real(data, secrets)
    monkeypatch.setattr(base, "redact_bytes", appending)
    assert base.redact_file(str(log)) is True
    data = log.read_bytes()
    assert KEY.encode() not in data and b"\x00" not in data
    assert data.startswith(f"start {base.REDACTED}\n".encode())
    assert data.rstrip(b"\n").endswith(b"appended meanwhile")
    assert (len(data) == len(f"start {KEY}\n") + len("appended meanwhile\n")) is live


def test_the_cli_sweep_runs_however_the_run_ends():
    tree = ast.parse(_read("gbench/cli.py"))
    # main() validates the arguments and hands the run to _run(), which holds the sweep
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
    tries = [n for n in ast.walk(main) if isinstance(n, ast.Try) and n.finalbody
             and "_redact_run_results(config)" in ast.unparse(n.finalbody)]
    assert len(tries) == 1
    body = ast.unparse(tries[0].body)
    assert "for model in models" in body and "print_results_summary(all_results, config)" in body


# --------------------------------------------------------------------------- #
# code sandbox
# --------------------------------------------------------------------------- #
def test_the_code_sandbox_isolates_processes_and_hides_the_docker_socket():
    from gbench.runners.eval_suites import sandbox
    argv = sandbox.wrap_argv(["python3", "-c", "pass"], force=True)
    assert argv[:6] == ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/run"]
    for flag in ("--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net"):
        assert flag in argv, flag
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv.index("--unshare-pid") < argv.index("--proc")
    assert argv[-3:] == ["python3", "-c", "pass"]


def test_model_code_gets_no_credentials(monkeypatch):
    from gbench.runners.eval_suites import sandbox
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("GBENCH_TEST_ENDPOINT_URL", f"https://user:{KEY}@host/v1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    env = sandbox.credential_free_env()
    assert "OPENAI_API_KEY" not in env and "GBENCH_TEST_ENDPOINT_URL" not in env
    assert env["OMP_NUM_THREADS"] == "1"
    # also in the explicit unsandboxed mode, and for an env a caller builds itself
    monkeypatch.setenv("GBENCH_SANDBOX", "none")
    sandbox.sandbox_available.cache_clear()
    try:
        r = sandbox.run_sandboxed(
            [sys.executable, "-c", "import json, os; print(json.dumps([sorted(k for k in os.environ "
                                   "if 'KEY' in k or 'ENDPOINT_URL' in k), "
                                   "os.environ.get('OMP_NUM_THREADS')]))"],
            capture_output=True, text=True, timeout=60, env={**os.environ, "OMP_NUM_THREADS": "2"})
    finally:
        sandbox.sandbox_available.cache_clear()
    assert r.returncode == 0, r.stderr
    names, threads = json.loads(r.stdout)
    assert threads == "2"
    assert "OPENAI_API_KEY" not in names and "GBENCH_TEST_ENDPOINT_URL" not in names, names
    # What is left is the environment's harmless variables that merely contain KEY (GPG_KEY in
    # the official python images, for one): no credential name and no credential value.
    from gbench.runners.eval_suites import base
    secrets = base.secret_values()
    for name in names:
        assert not base.is_secret_env_name(name), name
        assert not any(s in os.environ.get(name, "") for s in secrets), name


# --------------------------------------------------------------------------- #
# shared docs and registries in step with the code
# --------------------------------------------------------------------------- #
def test_readme_token_row_lists_every_suite_with_an_official_cap():
    readme = _read("docs/cli-reference.md")
    row = next(l for l in readme.splitlines() if l.startswith("| `--max-output-tokens` |"))
    for needle in ("acebench 1024", "chartqa 16", "docvqa and infographicvqa 32", "textvqa 1024",
                   "coco_caption 64", "charxiv 1000", "mmmu_pro and wildbench 4096",
                   "omnidocbench 32000", "medxpertqa 2048", "aa_lcr 16384", "mmlu_redux 1",
                   "livebench 4096", "ojbench sends none", "lab_bench", "loft_scifact",
                   "beam_128k", "screenspot and screenspot_pointing", "aider_polyglot",
                   "swe_lancer", "lcb 2000 (self-repair included)",
                   "replies_at_official_max_tokens", "whether it is larger or smaller"):
        assert needle in row, needle
    assert "its own request path sends it" not in row
    assert "self-repair requests send it" not in row


def test_env_knobs_follow_the_code():
    knobs = _read("docs/evals/env-knobs.md")
    section = knobs[knobs.index("- Suites that declare official sampling"):
                    knobs.index("- Renamed suites still read")]
    assert "is sent the run's budget" in section and "larger or smaller" in section
    assert "cannot declare one" in section and "genuinely_truncated" in section
    for knob in ("GBENCH_SWE_LANCER_IMAGE", "GBENCH_SWE_LANCER_TRAJ_DIR",
                 "GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", "GBENCH_SWE_BENCH_LIVE_HARNESS_DIR",
                 "APIUSER's 0.001"):
        assert knob in knobs, knob
    for gone in ("GBENCH_SWE_LANCER_HARNESS_DIR", "GBENCH_SWE_LANCER_EVAL_CMD",
                 "`SWELancer-Benchmark` (swe_lancer)", "(`GBENCH_SANDBOX` applies",
                 "`/stream-logs`", "web service"):
        assert gone not in knobs, gone
    assert not [c for c in knobs if ord(c) > 127]


def test_canonical_sync_follows_the_updates():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    from gbench.runners.eval_suites import swe_bench_live, swe_lancer
    assert CS["swe_lancer"]["revision"] == swe_lancer.HARNESS_COMMIT
    assert swe_lancer.TASKS_SHA256 in CS["swe_lancer"]["dataset"]
    assert "SimpleAgentSolver" in CS["swe_lancer"]["method"]
    assert swe_bench_live.WINDOWS_REVISION in CS["swe_bench_live"]["dataset"]
    assert "every leaderboard tab" in CS["swe_bench_live"]["method"]
    assert CS["screenspot"]["checksum"] == CS["screenspot_pointing"]["checksum"]
    assert "OS-Copilot/ScreenSpot-v2" in CS["screenspot"]["dataset"]
    assert "string literals labelled" not in CS["putnam_formal"]["method"]
    assert "one containing a string literal" not in CS["putnam_formal"]["method"]
    assert "Name literals as the applications they are built from" in CS["putnam_formal"]["method"]
    assert "EndpointRelay" in CS["swe_bench_pro"]["method"]
    assert CS["aider_polyglot"]["synced"] >= "2026-09-26"
    assert "same as docvqa" not in CS["infographicvqa"]["method"]
    for name, entry in CS.items():
        text = json.dumps(entry)
        assert all(ord(c) < 128 for c in text), name
        assert not re.search(r"\b[Ww]ave ?\d|\bW2[a-h]?\b", text), name


def test_swe_docs_and_registry_follow_the_suites():
    from gbench import cli
    from gbench.runners import evals as E
    from gbench.runners.eval_suites import swe_bench_live
    help_text = re.sub(r"\s+", " ", cli.create_parser().format_help())
    assert ("settings (lite, full, verified, ccpp, csharp, java, tsjs, go, rust, windows; "
            "default all ten)") in help_text
    assert set(swe_bench_live.SETTINGS) == {"lite", "full", "verified", "ccpp", "csharp",
                                            "java", "tsjs", "go", "rust", "windows"}
    assert "swelancer` task image" not in _read("gbench/runners/evals.py")
    assert "swe_lancer" in E.NAMED_ONLY_SUITES
    readme = _read("README.md") + "\n" + _read("docs/cli-reference.md")
    assert "`ccpp`, `csharp`, `java`, `tsjs`, `go`, `rust`, `windows`; default all ten" in readme
    # the negative checks covered the whole pre-split README: keep them over every moved doc
    moved = readme + "\n" + "\n".join(_read(f"docs/{n}.md") for n in (
        "installation", "environment-variables", "performance", "eval-suites", "golden-set",
        "remote-endpoints", "results", "deployment"))
    assert "Expensify/App" not in moved and "GBENCH_SWE_LANCER_HARNESS_DIR" not in moved
    tools = _read("docs/evals/toolchains.md")
    assert "gbench-swe-lancer-harness:51052ced" in tools and "GBENCH_SWE_LANCER_HARNESS_DIR" not in tools
    assert "mcp==1.16.0" in tools and "mcp==1.10.1" not in tools
    resources = _read("docs/RESOURCES.md")
    assert "OS-Copilot/ScreenSpot-v2" in resources and "HongxinLi" not in resources
    assert "HongxinLi" not in _read("docs/THIRD_PARTY_DATASETS.md")
    roadmap = _read("docs/evals_roadmap.md")
    entry = roadmap[roadmap.index(": `swe_lancer`"):][:3000]
    assert "NAMED_ONLY_SUITES" in entry and "1.3 TB" in entry and "Expensify/App at HEAD, which no\nlonger builds" in entry


def test_the_host_needs_no_uv():
    import tomllib
    project = tomllib.loads(_read("pyproject.toml"))
    evals_extra = project["project"]["optional-dependencies"]["evals"]
    assert not [d for d in evals_extra if re.match(r"uv\b", d)]
    assert '\nname = "uv"\n' not in _read("uv.lock")
