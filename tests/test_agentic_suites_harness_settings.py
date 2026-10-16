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

"""The agentic suites' canonical settings and harness accounting.

terminal_bench
  * Harbor is checked, and run, in the local harness image, at the TB2.1 leaderboard's release
    (0.20.0, the image's labels); gbench's interpreter and a `harbor` on PATH are never used;
  * the defaults are the leaderboard's (temperature 1, json parser, no max_tokens, timeout
    multiplier 1.0, no thread-cap env), the dataset is pinned to its content hash, and every
    opt-in deviation is named;
skillsbench
  * an image whose adapter predates the current one is refused (label gate);
gaia2
  * the launcher's server-parser policy is base's (the copied functions stay equal), recovers the
    text the server's parsers took from a reply out of its token ids, and names what it cannot;
  * a streamed reply whose first chunk carries content '' and nothing else is a reply without
    content (vLLM's stream shape);
  * the datasets are read offline from the image; the 400/422 rule is not claimed to be gbench's
    everywhere;
mcp_atlas
  * ddg-search's fetch_content runs as upstream in the environment container, never on Gemini.

Docker, Gemini, Harbor's registry and the model endpoint are stubbed throughout; the only sockets
are loopback ones this file opens.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import terminal_bench as TB

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gbench" / "docker"))
sys.path.insert(0, str(REPO / "tests"))
import gaia2_run as GRUN  # noqa: E402
from test_gaia2_toolbench_judge_protocol_and_credentials import _engine, _gaia2_run, _stub_are  # noqa: E402


def _read(*rel):
    return (REPO.joinpath(*rel)).read_text(encoding="utf-8")


def _flat(text):
    return " ".join(text.split())


# =========================================================================== #
# terminal_bench: the Harbor that runs is the Harbor that is checked
# =========================================================================== #
def _docker_ok(monkeypatch):
    monkeypatch.setattr(TB.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker"
                        else None)            # no `harbor` on PATH at all
    monkeypatch.setattr(TB.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="", stderr=""))


def test_harbor_is_checked_in_the_harness_image_at_the_leaderboards_release(monkeypatch,
                                                                             tmp_path):
    _docker_ok(monkeypatch)
    sock = tmp_path / "docker.sock"
    sock.write_text("")
    monkeypatch.setattr(TB, "_DOCKER_SOCKET", str(sock))
    build = ("docker build -t gbench-terminal-bench:harbor-0.20.0 "
             "-f gbench/docker/terminal_bench.Dockerfile gbench/docker")
    assert TB.harbor_image_build_command() == build
    monkeypatch.setattr(TB, "harbor_image_record", lambda image=None: None)
    ok, why = TB.check_terminal_bench_prerequisites()
    assert not ok and "is not on this host" in why and build in why
    assert "pip install" not in why and "uv tool install" not in why
    for labels in ({"org.gbench.harbor": "0.14.0",
                    "org.gbench.terminal_bench_lock": TB.TB21_LOCK_COMMIT},
                   {"org.gbench.harbor": "0.20.0", "org.gbench.terminal_bench_lock": "0" * 40},
                   {}):
        monkeypatch.setattr(TB, "harbor_image_record", lambda image=None, _l=labels: {
            "image": TB.HARBOR_IMAGE_DEFAULT, "id": "sha256:x", "labels": _l})
        ok, why = TB.check_terminal_bench_prerequisites()
        assert not ok and "rebuild it" in why and build in why and not TB._VERIFIED_IMAGE
    monkeypatch.setattr(TB, "harbor_image_record", lambda image=None: {
        "image": TB.HARBOR_IMAGE_DEFAULT, "id": "sha256:abc", "labels": dict(TB.HARBOR_IMAGE_LABELS)})
    assert TB.check_terminal_bench_prerequisites() == (True, "")   # PATH has no harbor
    assert TB._VERIFIED_IMAGE["id"] == "sha256:abc"
    assert TB.HARBOR_VERSION == "0.20.0"
    assert TB.HARBOR_IMAGE_LABELS == {"org.gbench.harbor": "0.20.0",
                                      "org.gbench.terminal_bench_lock": TB.TB21_LOCK_COMMIT}
    monkeypatch.setattr(TB, "_DOCKER_SOCKET", str(tmp_path / "absent.sock"))
    ok, why = TB.check_terminal_bench_prerequisites()
    assert not ok and "absent.sock does not exist" in why


def test_harbor_runs_in_the_harness_image_with_or_without_the_salvage_patch(monkeypatch):
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv("GBENCH_TB_SALVAGE_PATCH", raising=False)
        else:
            monkeypatch.setenv("GBENCH_TB_SALVAGE_PATCH", value)
        cmd = TB.harbor_command(["run", "-d", TB.DATASET_SPEC], jobs_dir="/tmp/j", name="c")
        assert cmd[:2] == ["docker", "run"] and sys.executable not in cmd
        at = cmd.index(TB.HARBOR_IMAGE_DEFAULT)
        mounts = [cmd[i + 1] for i, a in enumerate(cmd[:at]) if a == "-v"]
        patch = [m for m in mounts if m.endswith(":/opt/gbench/terminus_salvage_patch.py:ro")]
        if value is None:
            assert cmd[at + 1:at + 3] == ["python", "-c"]
            assert "from harbor.cli.main import app" in cmd[at + 3]
            assert "/opt/gbench/terminus_salvage_patch.py" in cmd[at + 3]
            assert cmd[at + 4:] == ["run", "-d", TB.DATASET_SPEC] and len(patch) == 1
        else:
            assert cmd[at + 1:] == ["harbor", "run", "-d", TB.DATASET_SPEC] and not patch


def _argv(monkeypatch, thinking, env=None):
    for k in ("GBENCH_TB_TURN_MAX_TOKENS", "GBENCH_TB_TIMEOUT_MULTIPLIER", "GBENCH_TB_PARSER",
              "GBENCH_TB_CONTAINER_THREADS", "GBENCH_TERMINAL_BENCH_TEMPERATURE", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    seen = {}
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))

    def popen(cmd, *a, **k):
        seen["cmd"] = cmd
        raise RuntimeError("argv captured")
    monkeypatch.setattr(TB.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError):
        TB.run_terminal_bench("m", "http://x:8000/v1", 2, enable_thinking=thinking)
    return seen["cmd"]


def _kwargs(cmd):
    return dict(cmd[i + 1].split("=", 1) for i, a in enumerate(cmd) if a == "--agent-kwarg")


@pytest.mark.parametrize("thinking", [False, True])
def test_the_default_run_is_the_tb21_leaderboards_settings(monkeypatch, thinking):
    B.set_run_knobs()
    B._RUN_KNOBS.pop("temperature", None)
    cmd = _argv(monkeypatch, thinking)
    assert cmd[:2] == ["docker", "run"] and TB.HARBOR_IMAGE_DEFAULT in cmd
    assert cmd[cmd.index("-d") + 1] == TB.DATASET_SPEC
    assert TB.DATASET_SPEC == ("terminal-bench/terminal-bench-2-1@sha256:"
                               "7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a")
    kw = _kwargs(cmd)
    # gbench's rule: 1.0 with --thinking, 0.0 without
    assert kw["temperature"] == ("1.0" if thinking else "0.0") and kw["parser_name"] == "json"
    llm = json.loads(kw["llm_call_kwargs"])
    assert "max_tokens" not in llm                         # Terminus-2 sends none
    assert llm["extra_body"] == {"chat_template_kwargs": {"enable_thinking": thinking}}
    assert cmd[cmd.index("--agent-timeout-multiplier") + 1] == "1.0"
    assert "--agent-env" not in cmd and "--verifier-env" not in cmd


def test_opt_in_knobs_still_reach_harbor(monkeypatch):
    B._RUN_KNOBS.pop("temperature", None)
    cmd = _argv(monkeypatch, True, env={
        "GBENCH_TB_TURN_MAX_TOKENS": "32768", "GBENCH_TB_PARSER": "xml",
        "GBENCH_TB_TIMEOUT_MULTIPLIER": "2", "GBENCH_TB_CONTAINER_THREADS": "1",
        "GBENCH_TERMINAL_BENCH_TEMPERATURE": "0.6"})
    kw = _kwargs(cmd)
    assert kw["temperature"] == "0.6" and kw["parser_name"] == "xml"
    assert json.loads(kw["llm_call_kwargs"])["max_tokens"] == 32768
    assert cmd[cmd.index("--agent-timeout-multiplier") + 1] == "2.0"
    assert "OMP_NUM_THREADS=1" in cmd and cmd.count("--verifier-env") == len(TB.THREAD_VARS)


def _trial(root, name, reward):
    trial = os.path.join(root, "job", name)
    os.makedirs(os.path.join(trial, "agent"))
    with open(os.path.join(trial, "agent", "trajectory.json"), "w") as f:
        json.dump({"steps": [{"message": "ls"}]}, f)
    with open(os.path.join(trial, "result.json"), "w") as f:
        json.dump({"trial_name": name, "verifier_result": {"rewards": {"reward": reward}},
                   "agent_result": {"n_output_tokens": 4000,
                                    "metadata": {"n_episodes": 10,
                                                 "api_request_times_msec": [2000.0] * 10}}}, f)


def _run(monkeypatch, env=None, thinking=True):
    for k in ("GBENCH_TB_TURN_MAX_TOKENS", "GBENCH_TB_TIMEOUT_MULTIPLIER", "GBENCH_TB_PARSER",
              "GBENCH_TB_CONTAINER_THREADS", "GBENCH_TERMINAL_BENCH_TEMPERATURE", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    B._RUN_KNOBS.pop("temperature", None)
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))

    class _Proc:
        returncode = 0
        stdout = iter(())

        def poll(self):
            return 0

        def wait(self):
            return 0

    def popen(cmd, **kw):
        # a complete run of a two-task dataset: the leaderboard's 5 trials of each task
        jobs = cmd[cmd.index("--jobs-dir") + 1]
        for k in range(TB.TERMINAL_BENCH_TRIALS):
            _trial(jobs, f"t-pass__{k}", 1.0)
            _trial(jobs, f"t-fail__{k}", 0.0)
        return _Proc()
    monkeypatch.setattr(TB.subprocess, "Popen", popen)
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", 2)
    return TB.run_terminal_bench("m", "http://h:1/v1", 1, enable_thinking=thinking)


def test_a_default_run_records_the_harness_and_names_only_the_trial_count(monkeypatch):
    res = _run(monkeypatch)
    assert res["harbor_version"] == "0.20.0"
    assert res["dataset"] == TB.DEFAULT_DATASET and res["dataset_ref"] == TB.DATASET_REF
    assert res["agent_parser"] == "json" and res["agent_timeout_multiplier"] == 1.0
    assert res["turn_max_tokens"] is None and res["container_threads"] is None
    assert res["temperature"] == 1.0 and res["temperature_source"] == "thinking"
    off = res["official_sampling"]
    assert off["temperature_matches_official"] is True and off["send_max_tokens"] is False
    assert "not sent" in res["max_output_tokens_note"]
    # the leaderboard's trials per task are run by default (Harbor -k 5): nothing to name
    assert res["attempts"]["attempts_per_sample"] == TB.TERMINAL_BENCH_TRIALS == 5
    assert res["leaderboard_comparable"] is True
    assert res["leaderboard_comparable_reason"] == "full canonical run"
    # no cap: the budget report has no per-turn figure, and the run does not trip over it
    assert res["turn_budget"]["turns_in_shortest_budget"] is None
    assert res["accuracy"] == 50.0


def test_a_no_think_run_sends_zero_and_names_it(monkeypatch):
    res = _run(monkeypatch, thinking=False)
    assert res["temperature"] == 0.0 and res["temperature_source"] == "default"
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert ("agent temperature 0.0 set by the no-think rule"
            in res["leaderboard_comparable_reason"])


def test_every_opt_in_deviation_is_named(monkeypatch):
    res = _run(monkeypatch, env={
        "GBENCH_TB_TURN_MAX_TOKENS": "32768", "GBENCH_TB_PARSER": "xml",
        "GBENCH_TB_TIMEOUT_MULTIPLIER": "2", "GBENCH_TB_CONTAINER_THREADS": "1",
        "GBENCH_TERMINAL_BENCH_TEMPERATURE": "0.6"})
    reason = res["leaderboard_comparable_reason"]
    for needle in ("agent temperature 0.6", "agent-timeout-multiplier=2.0",
                   "per-turn max_tokens cap=32768", "parser='xml'",
                   "thread cap 1", "the leaderboard requires no resource overrides"):
        assert needle in reason, needle
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "GBENCH_TB_TURN_MAX_TOKENS=32768" in res["max_output_tokens_note"]
    assert res["container_threads"] == 1 and res["turn_max_tokens"] == 32768


def test_the_shard_enumeration_asks_for_the_pinned_dataset_version(monkeypatch, capsys):
    asked, runs = [], []

    def run(cmd, **kw):
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='noise\n["terminal-bench/b", '
                                           '"terminal-bench/a", "terminal-bench/a"]\n', stderr="")
    monkeypatch.setattr(TB.subprocess, "run", run)
    assert TB._dataset_task_names(TB.DATASET_SPEC) == ["terminal-bench/a", "terminal-bench/b"]
    assert TB._dataset_task_names("org/name") == ["terminal-bench/a", "terminal-bench/b"]
    for cmd, ref in zip(runs, (TB.DATASET_SPEC, "org/name@latest")):
        # Harbor's own resolver, in the harness image, without the docker socket
        at = cmd.index(TB.HARBOR_IMAGE_DEFAULT)
        assert cmd[:2] == ["docker", "run"] and cmd[at + 1:] == [
            "python", "-c", TB.TASK_NAMES_SCRIPT, ref]
        assert not any("docker.sock" in a for a in cmd)

    # the script itself: PackageDatasetClient.get_dataset_metadata of the reference it is given
    class Client:
        async def get_dataset_metadata(self, name):
            asked.append(name)
            tid = types.SimpleNamespace(get_name=lambda: "terminal-bench/a")
            return types.SimpleNamespace(task_ids=[tid])

    pkg = types.ModuleType("harbor.registry.client.package")
    pkg.PackageDatasetClient = Client
    for name in ("harbor", "harbor.registry", "harbor.registry.client"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "harbor.registry.client.package", pkg)
    monkeypatch.setattr(sys, "argv", ["-c", TB.DATASET_SPEC])
    exec(compile(TB.TASK_NAMES_SCRIPT, "<script>", "exec"), {})
    assert asked == [TB.DATASET_SPEC]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == ["terminal-bench/a"]

    # no task list: an error naming Harbor's output (the caller turns it into infra_required)
    monkeypatch.setattr(TB.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="registry unreachable"))
    with pytest.raises(RuntimeError, match="registry unreachable"):
        TB._dataset_task_names(TB.DATASET_SPEC)


def test_the_terminal_bench_guide_follows_the_code():
    doc = _flat(_read("docs", "evals", "terminal_bench.md"))
    for needle in (TB.harbor_image_build_command(), TB.DATASET_REF, "`harbor_version`",
                   "`harness_image`",
                   "`dataset_ref`", "| `GBENCH_TB_PARSER` | `json` |",
                   "| `GBENCH_TB_TURN_MAX_TOKENS` | `0` (Terminus-2 sends none) |",
                   "| `GBENCH_TB_TIMEOUT_MULTIPLIER` | `1.0` |",
                   "| `GBENCH_TB_CONTAINER_THREADS` | `0` (none) |",
                   "`temperature: 1`", "**5 trials per task**", "`-k 5`",
                   '`leaderboard_comparable_reason: "full canonical run"`'):
        assert needle in doc, needle
    assert "install 'harbor" not in doc and "uv tool install" not in doc
    assert "pip install docker harbor" not in doc and "deliberately non-canonical" not in doc


# =========================================================================== #
# skillsbench: a stale image is refused
# =========================================================================== #
def _skills_docker(monkeypatch, label):
    monkeypatch.setattr(SB.shutil, "which", lambda name: "/usr/bin/docker")

    def run(cmd, *a, **k):
        if "--format" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=label + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(SB.subprocess, "run", run)


def test_skillsbench_refuses_an_image_without_the_current_adapter_label(monkeypatch):
    for stale in ("", "<no value>", "1"):
        _skills_docker(monkeypatch, stale)
        ok, why = SB.check_skillsbench_prerequisites()
        assert not ok and "predates the current adapter" in why
        assert f"{SB.HARNESS_LABEL}={SB.HARNESS_VERSION}" in why and "docker build" in why
    _skills_docker(monkeypatch, SB.HARNESS_VERSION)
    assert SB.check_skillsbench_prerequisites() == (True, "")


#: The adapter files' fingerprint at each HARNESS_VERSION. A change to either file must come
#: with a new version (and a rebuilt image): update both here.
_SKILLSBENCH_ADAPTER_SHA256 = {
    "2": "d6bb5019a9d9436fca0e3d41a6713fe976f2b5525d05e26dfe3633864786ef98",
    "3": "57ef3322b52c6aed418f5ae93b55c62e6acb72742393a134c3d02c4512b9754b",
    "4": "093eedfd100877d8200f04d8a5e98c6b2d3f3e4a821e7eb653c102668034ed2f",
    "5": "8e5177e93519b75d80dde728512202815b12e24ed654a5cc2e5649a8580f11fc",
    "6": "bc9d12b49a39ac7f54e4c5a49d5198044eabf475041962cce96ea35e3a08ff25",
    "7": "df297c5a5a3d01ebf82c926a1474d0021a4ef1038535170a746d502031f243e3",
}


def test_skillsbench_label_follows_the_adapter():
    df = _read("gbench", "docker", "skillsbench.Dockerfile")
    assert f'LABEL {SB.HARNESS_LABEL}="{SB.HARNESS_VERSION}"' in df
    digest = hashlib.sha256()
    for name in ("skillsbench_run.py", "skillsbench_entrypoint.sh"):
        digest.update((REPO / "gbench" / "docker" / name).read_bytes())
    assert _SKILLSBENCH_ADAPTER_SHA256.get(SB.HARNESS_VERSION) == digest.hexdigest(), (
        "the skillsbench adapter changed: bump skillsbench.HARNESS_VERSION and the Dockerfile "
        "label, rebuild the image, and record the new fingerprint here")
    doc = _flat(_read("docs", "evals", "skillsbench.md"))
    assert f"`{SB.HARNESS_LABEL}={SB.HARNESS_VERSION}`" in doc


# =========================================================================== #
# gaia2: the server-parser policy in the launcher
# =========================================================================== #
_CASES = [
    dict(content=None, reasoning="thought", tool_calls=None, thinking=False, completion_tokens=9),
    dict(content="x", reasoning="thought", tool_calls=None, thinking=False, completion_tokens=9),
    dict(content=None, reasoning="thought", tool_calls=None, thinking=True, completion_tokens=9),
    dict(content=None, reasoning=None, tool_calls=None, thinking=False, completion_tokens=15),
    dict(content="", reasoning="", tool_calls=None, thinking=True, completion_tokens=15),
    dict(content=None, reasoning=None, tool_calls=None, thinking=False, completion_tokens=0),
    dict(content=None, reasoning="r", tool_calls=[{"id": "1"}], thinking=False,
         completion_tokens=9),
    dict(content="answer", reasoning=None, tool_calls=None, thinking=False, completion_tokens=9),
]


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("generation", [None, "", "  ", "<|tool_call>call:f{}<tool_call|>"])
def test_the_launchers_copy_of_the_policy_equals_base(case, generation):
    assert GRUN.server_parser_signal(**case) == B.server_parser_signal(**case)
    got = GRUN.recover_server_parser_text(**case, generation_text=generation)
    want = B.recover_server_parser_text(**case, generation_text=generation)
    assert tuple(got) == tuple(want)
    assert GRUN.SERVER_PARSER_EVENTS == B.SERVER_PARSER_EVENTS


@pytest.mark.parametrize("ids,finish,stop", [
    ([1, 2, 3], "stop", None), ([1, 2, 3], "stop", "Obs"), ([1, 2, 3], "length", None),
    ([], "stop", None), (["x"], "stop", None), ([7], "stop", None)])
def test_the_launchers_token_id_rebuild_equals_base(ids, finish, stop):
    def decode(xs):
        return "".join(f"<{x}>" for x in xs) + "tail Obs more"
    assert GRUN.generation_text_from_token_ids(ids, decode, finish, stop) == \
        B.generation_text_from_token_ids(ids, decode, finish, stop)
    assert GRUN.generation_text_from_token_ids(ids, None, finish, stop) is None


def _vllm_chunk(finish=None, token_ids=None, stop_reason=None, **delta):
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    if token_ids is not None:
        choice["token_ids"] = token_ids
    if stop_reason is not None:
        choice["stop_reason"] = stop_reason
    return {"choices": [choice],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}


def test_a_stream_whose_first_chunk_is_an_empty_role_chunk_has_no_content():
    out = GRUN.read_stream(iter([
        _vllm_chunk(role="assistant", content=""),            # vLLM's first chunk, always
        _vllm_chunk(reasoning="plan ", token_ids=[5, 6]),
        _vllm_chunk(reasoning="more", token_ids=[7]),
        _vllm_chunk(finish="stop", token_ids=[1], stop_reason=None)]),
        ["<end_action>"])
    assert out["content"] is None and out["reasoning"] == "plan more"
    assert out["token_ids"] == [5, 6, 7, 1] and out["finish_reason"] == "stop"
    out = GRUN.read_stream(iter([
        _vllm_chunk(role="assistant", content=""),
        _vllm_chunk(tool_calls=[{"index": 0}], token_ids=[9, 9]),
        _vllm_chunk(finish="stop", token_ids=[1], stop_reason=106)]), ["<end_action>"])
    assert out["content"] is None and out["tool_call_deltas"] == 1
    assert out["stop_reason"] == 106 and out["token_ids"] == [9, 9, 1]


@pytest.fixture
def thinking_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18790")
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "true")
    monkeypatch.setenv("GAIA2_AGENT_MAX_TOKENS", "65536")
    monkeypatch.delenv("GBENCH_DETOKENIZE_URL", raising=False)
    chunks = {"list": []}

    def responder(kw, ModelResponse):
        assert kw.get("stream") is True and kw["extra_body"]["return_token_ids"] is True
        return types.SimpleNamespace(completion_stream=iter(chunks["list"]))

    are = _stub_are(monkeypatch, responder)
    GRUN.install_are_hooks()
    return types.SimpleNamespace(are=are, chunks=chunks, workdir=tmp_path)


def test_a_thinking_reply_that_ended_in_its_reasoning_raises_as_upstream(thinking_engine):
    # vLLM's stream shape: the role chunk with content '' first, then reasoning only
    thinking_engine.chunks["list"] = [_vllm_chunk(role="assistant", content=""),
                                      _vllm_chunk(reasoning="I will call the tool"),
                                      _vllm_chunk(finish="stop")]
    agent = _engine(thinking_engine.are, "http://172.17.0.1:5678/v1")
    with pytest.raises(AssertionError):
        agent([{"role": "user", "content": "q"}], stop_sequences=["<end_action>"])
    summary = GRUN.agent_call_summary(str(thinking_engine.workdir))
    assert summary["null_content"] == 1 and summary["empty_content"] == 0
    assert summary["stopped_in_reasoning"] == 1 and summary["server_parser_events"] == {}


def test_a_thinking_reply_the_tool_parser_took_is_recovered_from_its_stream_ids(
        thinking_engine, monkeypatch):
    monkeypatch.setattr(GRUN, "decode_token_ids",
                        lambda ids: "Thought: t\nAction:\n{\"x\": True}" if ids == [9, 9] else None)
    thinking_engine.chunks["list"] = [_vllm_chunk(role="assistant", content=""),
                                      _vllm_chunk(tool_calls=[{"index": 0}], token_ids=[9, 9]),
                                      _vllm_chunk(finish="stop", token_ids=[1])]
    agent = _engine(thinking_engine.are, "http://172.17.0.1:5678/v1")
    res, _meta = agent([{"role": "user", "content": "q"}], stop_sequences=["<end_action>"])
    assert res == 'Thought: t\nAction:\n{"x": true}'      # ARE's own rewrite still applies
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(thinking_engine.workdir))
    assert call["server_parser"] == GRUN.SERVER_PARSER_DROPPED_TEXT
    assert call["recovered_from"] == "token_ids" and call["tool_call_deltas"] == 1
    assert call["token_ids_returned"] == 3


@pytest.fixture
def plain_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18790")
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "false")
    for k in ("GAIA2_AGENT_MAX_TOKENS", "GAIA2_AGENT_TEMPERATURE", "GBENCH_DETOKENIZE_URL"):
        monkeypatch.delenv(k, raising=False)
    reply = {"content": None, "reasoning": None, "ids": None, "stop_reason": None,
             "finish": "stop"}

    def responder(kw, ModelResponse):
        assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False},
                                    "return_token_ids": True}
        r = ModelResponse(reply["content"], "served", reasoning=reply["reasoning"],
                          finish_reason=reply["finish"])
        # what install_token_ids_hook keeps from the raw reply
        r._hidden_params = {"gbench_choices": [{"token_ids": reply["ids"],
                                                "stop_reason": reply["stop_reason"]}]}
        return r

    are = _stub_are(monkeypatch, responder)
    GRUN.install_are_hooks()
    return types.SimpleNamespace(are=are, reply=reply, workdir=tmp_path)


class _Detok:
    """gaia2.DetokenizeRelay on loopback with a stand-in decoder."""

    def __init__(self, table):
        self.relay = G.DetokenizeRelay("127.0.0.1", decode=lambda ids: table.get(tuple(ids)))

    def __enter__(self):
        return self.relay.start()

    def __exit__(self, *a):
        self.relay.stop()


def test_a_deleted_reply_is_recovered_through_gbenchs_decoder(plain_engine, monkeypatch):
    plain_engine.reply.update(ids=[31, 32, 1])
    with _Detok({(31, 32): "Thought: ok\nAction:\n{}<end_action>Observation: x"}) as relay:
        monkeypatch.setenv("GBENCH_DETOKENIZE_URL", relay.url)
        agent = _engine(plain_engine.are, "http://172.17.0.1:5678/v1")
        res, _ = agent([{"role": "user", "content": "q"}],
                       stop_sequences=["<end_action>", "Observation:"])
        record = relay.record()
    assert res == "Thought: ok\nAction:\n{}"                  # ARE's split still applies
    assert record["decoded"] == 1 and record["requests"] == 1
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(plain_engine.workdir))
    assert call["server_parser"] == GRUN.SERVER_PARSER_DROPPED_TEXT and call["content_null"]
    assert call["generation_text"].startswith("Thought: ok")
    assert GRUN._read_jsonl(GRUN.CALL_FAILURES, str(plain_engine.workdir)) == []


def test_a_deleted_reply_without_token_ids_is_a_harness_error(plain_engine):
    plain_engine.reply.update(ids=None)
    agent = _engine(plain_engine.are, "http://172.17.0.1:5678/v1")
    with pytest.raises(AssertionError):                       # ARE gets no reply, as upstream
        agent([{"role": "user", "content": "q"}], stop_sequences=["<end_action>"])
    [fail] = GRUN._read_jsonl(GRUN.CALL_FAILURES, str(plain_engine.workdir))
    assert fail["server_parser_text_lost"] and fail["error"] == "ServerParserTextLost"
    assert fail["endpoint_failure"] is False
    by_run, loose = GRUN.read_call_failures(str(plain_engine.workdir))
    assert not loose and GRUN.classify_run({"status": "success"}, next(iter(by_run.values()))) \
        == ["server_parser_text_lost"]
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(plain_engine.workdir))
    assert call["server_parser"] == GRUN.SERVER_PARSER_TEXT_LOST
    assert call["generation_text"] is None


def test_moved_text_is_the_whole_generation_from_the_token_ids(plain_engine, monkeypatch):
    # a spontaneous thought plus the action: the generation, in order, from the token ids; the
    # stop string the server matched cuts it as the server cut the text
    plain_engine.reply.update(content="Action:\n{}", reasoning="thinking",
                              ids=[4, 5, 6], stop_reason="<end_action>")
    table = {(4, 5, 6): "<|channel>thought\nthinking<channel|>Action:\n{}<end_action>tail"}
    with _Detok(table) as relay:
        monkeypatch.setenv("GBENCH_DETOKENIZE_URL", relay.url)
        agent = _engine(plain_engine.are, "http://172.17.0.1:5678/v1")
        res, _ = agent([{"role": "user", "content": "q"}], stop_sequences=["<end_action>"])
    assert res == "<|channel>thought\nthinking<channel|>Action:\n{}"
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(plain_engine.workdir))
    assert call["server_parser"] == GRUN.SERVER_PARSER_MOVED_TEXT
    assert call["recovered_from"] == "token_ids" and call["stop_reason"] == "<end_action>"


def test_the_token_id_hook_keeps_what_litellms_conversion_drops(monkeypatch):
    openai_mod = types.ModuleType("litellm.llms.openai.openai")

    def convert(response_object=None, model_response_object=None, **kw):
        return types.SimpleNamespace(_hidden_params=None, raw=response_object)
    openai_mod.convert_to_model_response_object = convert
    for name in ("litellm", "litellm.llms", "litellm.llms.openai"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["litellm.llms.openai"].openai = openai_mod
    monkeypatch.setitem(sys.modules, "litellm.llms.openai.openai", openai_mod)
    assert GRUN.install_token_ids_hook() is True
    assert GRUN.install_token_ids_hook() is True                   # idempotent
    wrapped = openai_mod.convert_to_model_response_object
    out = wrapped(response_object={"choices": [{"message": {}, "token_ids": [1, 2],
                                                "stop_reason": "Obs"}]})
    assert out._hidden_params == {"gbench_choices": [{"token_ids": [1, 2],
                                                      "stop_reason": "Obs"}]}
    assert GRUN.response_extras(out) == {"token_ids": [1, 2], "stop_reason": "Obs"}


def test_the_decoder_relay_only_turns_token_ids_into_text():
    import urllib.error
    import urllib.request
    with _Detok({(1, 2): "ab"}) as relay:
        assert GRUN.decode_token_ids([1, 2], url=relay.url) == "ab"
        assert GRUN.decode_token_ids([3], url=relay.url) is None      # undecodable -> null
        base_url = relay.url.rsplit("/", 1)[0]
        for path, body in (("/detokenize", b'{"tokens": [true]}'), ("/detokenize", b"nope"),
                           ("/other", b'{"tokens": [1]}')):
            req = urllib.request.Request(base_url + path, data=body,
                                         headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(req, timeout=5)
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base_url + "/detokenize", timeout=5)      # GET
        rec = relay.record()
    assert rec["decoded"] == 1 and rec["undecodable"] == 1 and rec["refused"] == 4
    assert GRUN.decode_token_ids([1], url="") is None


def test_gaia2_passes_the_decoder_and_names_the_parser_events(monkeypatch, tmp_path):
    res, [(cmd, _)], guard = _gaia2_run(monkeypatch, tmp_path, summary={
        "agent_calls": {"calls": 9, "server_parser_events": {
            B.SERVER_PARSER_MOVED_TEXT: 2, B.SERVER_PARSER_DROPPED_TEXT: 1},
            "server_parser_recovered_from": {"token_ids": 3}},
        "hooks": {"token_ids_hook": False, "detokenizer": True},
        "dataset": GRUN.dataset_record({"GAIA2_DATASET_REVISION": "rev", "HF_HUB_OFFLINE": "1",
                                        "GAIA2_LOCAL_DATASET": "/opt/gaia2/gaia2"})})
    assert any(a.startswith("GBENCH_DETOKENIZE_URL=http://172.17.0.1:") for a in cmd)
    [(args, kwargs)] = guard
    assert kwargs["resolve"] == () and G.DNS_NAMES == ()
    assert res["server_parser_events"] == {B.SERVER_PARSER_MOVED_TEXT: 2,
                                           B.SERVER_PARSER_DROPPED_TEXT: 1}
    assert res["server_parser_note"] == B.server_parser_note(
        {B.SERVER_PARSER_MOVED_TEXT: 2, B.SERVER_PARSER_DROPPED_TEXT: 1}, {"token_ids": 3})
    reason = res["leaderboard_comparable_reason"]
    assert res["server_parser_note"] in reason
    assert "could not keep the token ids of non-streamed agent replies" in reason
    assert res["dataset"]["hub_offline"] is True and res["dataset"]["revision"] == "rev"
    assert res["agent_protocol"]["return_token_ids"] is True


def test_the_image_reads_the_datasets_offline_at_pinned_revisions():
    df = _read("gbench", "docker", "gaia2.Dockerfile")
    for needle in ("HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1",
                   "DEMO_FS_PATH=/opt/gaia2/gaia2_filesystem/demo_filesystem",
                   "GAIA2_LOCAL_DATASET=/opt/gaia2/gaia2",
                   "ARG GAIA2_REVISION=78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99",
                   "ARG GAIA2_FS_REVISION=132e26376f5e963bb59f64bcccdd02188cb08dee",
                   "assert all(counts.values())",
                   "'convert_to_model_response_object(' in inspect.getsource",
                   f'LABEL {G.HARNESS_ERRORS_LABEL}="{G.HARNESS_ERRORS_VERSION}"'):
        assert needle in df, needle
    assert "do NOT force HF_HUB_OFFLINE" not in df
    rec = GRUN.dataset_record({"GAIA2_LOCAL_DATASET": "/opt/gaia2/gaia2", "HF_HUB_OFFLINE": "1",
                               "DEMO_FS_PATH": "/opt/x"})
    assert rec["hf_dataset_arg"] == "/opt/gaia2/gaia2" and rec["dataset"] == GRUN.HUB_DATASET
    assert rec["filesystem_local_copy"] == "/opt/x" and rec["hub_offline"] is True


def test_nothing_claims_gbench_counts_400_422_the_same_everywhere():
    for rel in (("gbench", "runners", "eval_suites", "gaia2.py"),
                ("gbench", "docker", "gaia2_run.py"), ("gbench", "docker", "gaia2.Dockerfile"),
                ("docs", "evals", "gaia2.md"),
                ("gbench", "runners", "eval_suites", "toolbench.py"),
                ("docker", "toolbench_patch_llm.py"), ("docs", "evals", "toolbench.md")):
        text = _flat(_read(*rel).replace("#", " "))
        assert "everywhere" not in text.replace("gbench does not apply one rule everywhere",
                                                ""), rel
    doc = _flat(_read("docs", "evals", "gaia2.md"))
    assert "gbench does not apply one rule in every suite yet" in doc
    tb_doc = _flat(_read("docs", "evals", "toolbench.md"))
    assert "gbench does not apply one rule in every suite yet" in tb_doc
    # the suites that count a 400/422 as a harness error are named where the split is described
    for rel in (("docs", "evals", "gaia2.md"), ("docs", "evals", "toolbench.md")):
        assert "base-run suites, bfcl, mcp_bench and the SWE family" in _flat(_read(*rel)), rel


def test_mcp_bench_still_counts_every_endpoint_error_as_a_harness_error():
    """The 400/422 split is recorded, not unified: mcp_bench keeps its rule (every model request
    that ended in an endpoint or connection error is a harness error)."""
    from gbench.runners.eval_suites import mcp_bench
    errors, notes = mcp_bench.model_request_failures(
        {"counts": {"failed_requests": 2, "timed_out": 0},
         "failed_by_error": {"BadRequestError 400": 2}}, None, "MCP-Bench")
    assert errors and errors[0]["failed_requests"] == 2
    assert any("harness errors" in n for n in notes)


def test_the_gaia2_guide_names_the_new_accounting():
    doc = _flat(_read("docs", "evals", "gaia2.md"))
    for needle in ("`server_parser_text_lost`", "`server_parser_moved_text`",
                   "`server_parser_dropped_text`", "`return_token_ids`", "`DetokenizeRelay`",
                   "`HF_HUB_OFFLINE=1`", "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99",
                   "(Claude, Kimi, Qwen)", "`max_retries=2`",
                   f"`{G.HARNESS_ERRORS_LABEL}={G.HARNESS_ERRORS_VERSION}`"):
        assert needle in doc, needle
    assert "reply_in_reasoning" not in doc and "as a non-streamed one does" not in doc


# =========================================================================== #
# mcp_atlas: ddg-search's fetch_content runs as upstream
# =========================================================================== #
def _env_server(servers):
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            listed = [[s, "OK"] for s in servers]
            return self._send({"servers": listed, "total": len(listed), "online": len(listed),
                               "offline": 0})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))
                              or b"{}")
            calls.append(body)
            return self._send([{"type": "text", "text": "Example Domain ... [Content info]"}])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", calls


def test_ddg_fetch_content_goes_to_the_environment_not_to_gemini(monkeypatch):
    # (the URL context lookup is gone: every page tool runs on the release's server)
    assert not hasattr(M, "url_context_page")
    monkeypatch.setattr(M, "_ground", lambda *a, **k: pytest.fail("Gemini grounded"))
    srv, url, calls = _env_server(M.KEYLESS_SERVERS)
    sb = M._Sandbox(url)
    token = M._ACTIVE_SANDBOX.set(sb)
    try:
        out = asyncio.run(M.execute_mcp_tool("ddg-search_fetch_content",
                                             {"url": "https://example.com/"}, 0))
    finally:
        M._ACTIVE_SANDBOX.reset(token)
        srv.shutdown()
        srv.server_close()
    assert out.startswith("Example Domain")
    assert calls == [{"tool_name": "ddg-search_fetch_content",
                      "tool_args": {"url": "https://example.com/"}}]
    assert sb.search_calls == 0 and sb.gemini_outages == 0


def test_the_preflight_needs_ddg_search_running_in_the_environment(monkeypatch):
    srv, url, _ = _env_server([s for s in M.ENVIRONMENT_SERVERS if s != "ddg-search"])
    try:
        with pytest.raises(RuntimeError) as err:
            M._preflight(M._Sandbox(url))
    finally:
        srv.shutdown()
        srv.server_close()
    msg = str(err.value)
    assert "does not run ddg-search" in msg and "ddg-search_fetch_content" in msg


def test_the_catalog_lists_each_ddg_tool_once():
    env = [{"name": "context7_resolve-library-id"},
           {"name": "ddg-search_search", "description": "env copy"},
           {"name": "ddg-search_fetch_content", "description": "env copy"},
           {"name": "fetch_fetch"}]
    names = [t["name"] for t in M.merged_catalog(env)]
    assert names.count("ddg-search_search") == names.count("ddg-search_fetch_content") == 1
    ddg = next(t for t in M.merged_catalog(env) if t["name"] == "ddg-search_fetch_content")
    assert ddg["description"] != "env copy"      # the pinned server listing (sha256-pinned)


def test_the_image_keeps_ddg_search():
    df = _read("docker", "mcp_atlas.Dockerfile")
    # only brave-search leaves the template: ddg-search, exa and oxylabs run for their page tools
    assert "ARG REMOVED_SEARCH_SERVERS=brave-search\n" in df
    assert M.IMAGE_SEARCH_LABEL_VALUE == "removed:brave-search"
    assert "ddg-search" in M.KEYLESS_SERVERS and "ddg-search" not in M.CREDENTIALED_SERVERS
    doc = _flat(_read("docs", "evals", "mcp_atlas.md"))
    assert "run as upstream" in doc and "(open question)" not in doc
