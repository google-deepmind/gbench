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

"""toolbench and the quality runner call no model or service provider other than the Gemini API.

toolbench: the ToolEval grader and the cache-miss API simulator run on the Gemini cascade only; no
variable points the simulator at another base URL, key or model, and the container, which runs the
model's tool calls, gets only gbench's Gemini relay placeholder by name (the relay holds the keys,
GEMINI_API_KEYS wins). quality: the GemmaClaw QA suite's alternate model and
image-generation model are Gemini models (never openai/gpt-5.4 or openai/gpt-image-1), and no
GemmaClaw process gets a key of another provider. Everything runs on stubs: no docker, no network,
no Gemini call; files are written under tmp only.
"""

import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import quality as Q
from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import toolbench as T

ROOT = Path(__file__).resolve().parents[1]

GEM_A = "AIzaSyTESTgemA0000000000000001"
GEM_B = "AIzaSyTESTgemB0000000000000002"
GEM_C = "AIzaSyTESTgemC0000000000000003"
# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
OPENAI = "sk-" + "TESTopenai000000000000000001"
ANTHROPIC = "sk-ant-TESTanthropic00000000001"
OPENROUTER = "sk-or-TESTopenrouter0000000001"
SIM_KEY = "simulator-TESTsecret-00000001"

#: Provider variables a host may carry; none may reach a GemmaClaw process or the toolbench image.
FOREIGN_ENV = {
    "OPENAI_API_KEY": OPENAI, "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "ANTHROPIC_API_KEY": ANTHROPIC, "ANTHROPIC_OAUTH_TOKEN": "tok-TESTant-0001",
    "OPENROUTER_API_KEY": OPENROUTER, "OPENCLAW_LIVE_OPENAI_KEY": "sk-TESTlive-0001",
    "OPENCLAW_LIVE_ANTHROPIC_KEY": "sk-TESTlive-0002", "GOOGLE_API_KEY": "AIzaTESTgoogle-0001",
    "AWS_SECRET_ACCESS_KEY": "aws-TESTsecret-0001", "BRAVE_API_KEY": "brave-TEST-0001",
    "MISTRAL_API_KEY": "mistral-TEST-0001", "OPENAI_API_KEY_2": "sk-TESTnumbered-0001",
    "XAI_API_KEYS": "xai-TEST-0001", "HF_TOKEN": "hf_TEST0001", "GITHUB_TOKEN": "ghp_TEST0001",
    "CLAUDE_CODE_OAUTH_TOKEN": "tok-TESTclaude-0001",
}


def _read(*parts):
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# toolbench
# --------------------------------------------------------------------------- #
_GONE = ("GBENCH_TOOLBENCH_SIMULATOR_BASE", "GBENCH_TOOLBENCH_SIMULATOR_KEY",
         "GBENCH_TOOLBENCH_SIMULATOR_MODEL", "TB_SIMULATOR_KEY", "TB_SIMULATOR_BASE")


def test_toolbench_has_no_knob_for_another_simulator_provider():
    src = _read("gbench", "runners", "eval_suites", "toolbench.py")
    entry = _read("docker", "toolbench_entrypoint.sh")
    doc = _read("docs", "evals", "toolbench.md")
    for name in _GONE:
        assert name not in src, name
        assert name not in entry, name
        assert name not in doc, name
    assert T.SIMULATOR_BASE == "https://generativelanguage.googleapis.com/v1beta/openai/"
    # the simulator's base is gbench's Gemini relay (which sends the call to T.SIMULATOR_BASE)
    assert 'SIMULATOR_BASE="${RELAY}/v1beta/openai/"' in entry
    config_block = entry[entry.index("cat > /stb/server/config.yml"):entry.index("\nEOF\n")]
    assert "openai" not in config_block.lower()


def _tb_answer(solved):
    return {"query": "q", "answer": {"final_answer": "S" if solved else "N",
                                     "answer_details": []}}


@pytest.fixture
def tb(monkeypatch, tmp_path):
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR",
              "GBENCH_TOOLBENCH_REFERENCE", "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_SHARD",
              "GBENCH_TOOLBENCH_MAX_QUERY_COUNT", "GBENCH_TOOLBENCH_TEMPERATURE",
              "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    monkeypatch.setenv("GEMINI_API_KEYS", f"{GEM_B},{GEM_C}")
    for k, v in FOREIGN_ENV.items():
        monkeypatch.setenv(k, v)
    # the removed knobs, set to another provider: they must change nothing
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_KEY", SIM_KEY)
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_MODEL", "gpt-4-turbo")
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
    runs = []

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["docker", "run"]:
            runs.append((cmd, kw))
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
            for g in T._GROUPS:
                with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                    json.dump({"1": _tb_answer(True), "2": _tb_answer(True)}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    return runs


def _env_names(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]


def test_toolbench_container_gets_only_the_gemini_relay_placeholder_by_name(tb):
    res = T.run_toolbench("m", "http://x:8000/v1", 2)
    [(cmd, kw)] = tb
    names = _env_names(cmd)
    # by name only; the container gets the relay's placeholder, never a Gemini key (the relay
    # holds GEMINI_API_KEYS, which wins over GEMINI_API_KEY, as the judge reads them)
    assert "GEMINI_API_KEY" in names and "GEMINI_API_KEYS" in names
    assert kw["env"]["GEMINI_API_KEYS"].startswith("gbench-gemini-relay-")
    assert kw["env"]["GEMINI_API_KEY"] == kw["env"]["GEMINI_API_KEYS"]
    argv = " ".join(cmd)
    for secret in (GEM_A, GEM_B, GEM_C, SIM_KEY, *FOREIGN_ENV.values()):
        assert secret not in argv
    for secret in (GEM_A, GEM_B, GEM_C):     # the docker client's own environment neither
        assert secret not in json.dumps(kw["env"])
    # nothing of another provider is handed to the container
    passed = {n.split("=", 1)[0] for n in names}
    assert not passed & set(FOREIGN_ENV) and not passed & set(_GONE)
    assert f"GBENCH_JUDGE_MODELS={','.join(B.judge_cascade())}" in names
    text = json.dumps(res)
    for secret in (GEM_A, GEM_B, GEM_C, SIM_KEY, OPENAI, ANTHROPIC, OPENROUTER):
        assert secret not in text
    assert "api.openai.com" not in text


def test_toolbench_result_records_both_gemini_substitutions(tb):
    res = T.run_toolbench("m", "http://x:8000/v1", 2)
    cascade = B.judge_cascade()
    subs = {s["role"]: s for s in res["gemini_substitutions"]}
    assert set(subs) == {"ToolEval grader (SoPR/SoWR judge)", "cache-miss API simulator"}
    for s in subs.values():
        assert s["upstream"] == "gpt-4-turbo" and s["gemini_models"] == cascade
        assert s["api"].startswith("Gemini API")
    assert T.SIMULATOR_BASE in subs["cache-miss API simulator"]["api"]
    assert "gbench's Gemini relay" in subs["cache-miss API simulator"]["api"]
    sim = res["simulator"]
    assert sim["models"] == cascade and sim["provider"] == "Gemini API"
    assert sim["base_url"] == T.SIMULATOR_BASE and sim["matches_upstream_model"] is False
    assert res["judge_models"] == cascade
    assert res["leaderboard_comparable"] is False
    reason = res["leaderboard_comparable_reason"]
    assert ("graded by gbench's standard Gemini cascade on the Gemini API ("
            + ", ".join(cascade)) in reason
    assert "instead of gpt-4-turbo" in reason
    assert f"Gemini API OpenAI-compatible endpoint {T.SIMULATOR_BASE}" in reason


def _prereq_stubs(monkeypatch, seen):
    monkeypatch.setattr(T.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(T.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(T, "image_applies_request_policy", lambda image: True)
    monkeypatch.setattr(T, "image_records_harness_errors", lambda image: True)
    monkeypatch.setattr(T, "real_api_proxy_reachable", lambda: (True, ""))

    def live(key):
        seen.append(key)
        return True, ""

    monkeypatch.setattr(T, "gemini_key_live_valid", live)


def test_toolbench_prerequisites_take_gemini_api_keys_alone(monkeypatch):
    seen = []
    _prereq_stubs(monkeypatch, seen)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", f"{GEM_B};{GEM_C}")
    assert T.check_toolbench_prerequisites() == (True, "")
    assert seen == [f"{GEM_B},{GEM_C}"]


def test_toolbench_prerequisites_name_both_key_variables_when_none_is_set(monkeypatch):
    seen = []
    _prereq_stubs(monkeypatch, seen)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI)     # another provider's key is no substitute
    ok, why = T.check_toolbench_prerequisites()
    assert ok is False and "GEMINI_API_KEYS" in why and "GEMINI_API_KEY" in why
    assert seen == [] and OPENAI not in why


def _entrypoint_head(tmp_path):
    """The entrypoint up to (and with) the key unset, writing config.yml under tmp_path and
    dumping the environment the processes it starts next would inherit."""
    entry = _read("docker", "toolbench_entrypoint.sh")
    marker = ("unset GEMINI_API_KEY GEMINI_API_KEYS SIMULATOR_KEYS GEMINI_OPENAI_BASE_URL "
              "GBENCH_GEMINI_RELAY_URL\n")
    assert entry.count(marker) == 1
    head = entry[:entry.index(marker) + len(marker)]
    head = head.replace("/stb/server/config.yml", str(tmp_path / "config.yml"))
    script = tmp_path / "head.sh"
    script.write_text(head + f'env > "{tmp_path / "env.txt"}"\n')
    return script


PH_A = "gbench-gemini-relay-" + "a1" * 24
PH_B = "gbench-gemini-relay-" + "b2" * 24
RELAY = "http://172.17.0.1:4321"


def _run_head(tmp_path, extra):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "TB_MODEL": "m",
           "TB_ENDPOINT": "http://172.17.0.1:5678/v1", "GBENCH_JUDGE_MODELS": "gemini-x,gemini-y",
           "GBENCH_GEMINI_RELAY_URL": RELAY, **extra}
    return subprocess.run(["bash", str(_entrypoint_head(tmp_path))], env=env,
                          capture_output=True, text=True, timeout=60)


def test_toolbench_entrypoint_writes_a_gemini_relay_only_simulator_config(tmp_path):
    proc = _run_head(tmp_path, {"GEMINI_API_KEYS": f"{PH_A},{PH_B}", "GEMINI_API_KEY": PH_A,
                                "TB_SIMULATOR_BASE": "https://api.openai.com/v1",
                                "TB_SIMULATOR_KEY": OPENAI, "OPENAI_API_KEY": OPENAI})
    assert proc.returncode == 0, proc.stderr
    cfg = (tmp_path / "config.yml").read_text()
    assert f'api_key: "{PH_A},{PH_B}"' in cfg
    assert f"api_base: {RELAY}/v1beta/openai/" in cfg and "model: gemini-x" in cfg
    assert "openai.com" not in cfg and OPENAI not in cfg
    after = (tmp_path / "env.txt").read_text()
    for secret in (PH_A, PH_B, RELAY):
        assert secret not in after          # nothing started next holds the placeholder


def test_toolbench_entrypoint_refuses_a_real_gemini_key_and_needs_a_placeholder(tmp_path):
    proc = _run_head(tmp_path, {"GEMINI_API_KEY": PH_A})
    assert proc.returncode == 0, proc.stderr
    assert f'api_key: "{PH_A}"' in (tmp_path / "config.yml").read_text()
    real = tmp_path / "real"
    real.mkdir()
    proc = _run_head(real, {"GEMINI_API_KEY": GEM_A})
    assert proc.returncode == 7 and "a real key is refused" in proc.stderr
    assert not (real / "config.yml").exists() and GEM_A not in proc.stderr
    none = tmp_path / "none"
    none.mkdir()
    proc = _run_head(none, {"OPENAI_API_KEY": OPENAI})
    assert proc.returncode == 2 and "GEMINI_API_KEYS or GEMINI_API_KEY" in proc.stderr
    assert not (none / "config.yml").exists()


def test_toolbench_doc_knobs_table_names_only_gemini_keys():
    doc = _read("docs", "evals", "toolbench.md")
    assert "| `GEMINI_API_KEYS` / `GEMINI_API_KEY` | *(one required)* |" in doc
    assert "`gemini_substitutions`" in doc and "`judge_models`" in doc
    assert "no variable points the simulator or the\ngrader at another provider" in doc


# --------------------------------------------------------------------------- #
# quality (GemmaClaw QA suite)
# --------------------------------------------------------------------------- #
def test_quality_gemini_models_are_gemmaclaws_google_defaults_not_openai():
    assert Q.GEMINI_ALT_MODEL == "google/gemini-3.1-pro-preview"
    assert Q.GEMINI_IMAGE_GENERATION_MODEL == "google/gemini-3.1-flash-image-preview"
    assert Q.UPSTREAM_ALT_MODEL == "openai/gpt-5.4"
    assert Q.UPSTREAM_IMAGE_GENERATION_MODEL == "openai/gpt-image-1"
    for m in (Q.GEMINI_ALT_MODEL, Q.GEMINI_IMAGE_GENERATION_MODEL):
        assert m.startswith("google/gemini-")


def test_quality_provider_free_env_drops_every_foreign_key_and_keeps_the_rest():
    base = {**FOREIGN_ENV, "GEMINI_API_KEY": GEM_A, "GEMINI_API_KEYS": f"{GEM_B};{GEM_C}",
            "VLLM_API_KEY": "vllm-key", "PATH": "/usr/bin", "HOME": "/h",
            "OPENCLAW_CONFIG_PATH": "/x.json", "GBENCH_CACHE_DIR": "/c"}
    env, removed = Q.provider_free_env(base)
    assert removed == sorted(FOREIGN_ENV)
    assert not set(env) & set(FOREIGN_ENV)
    assert "GEMINI_API_KEY" not in env and "GEMINI_API_KEYS" not in env
    assert env["VLLM_API_KEY"] == "vllm-key" and env["PATH"] == "/usr/bin"
    assert env["OPENCLAW_CONFIG_PATH"] == "/x.json" and env["GBENCH_CACHE_DIR"] == "/c"
    suite, removed2 = Q.sandbox_suite_env(base, home="/r/home", npm_cache="/c/npm",
                                          config_path="/r/openclaw.json",
                                          relay_base_url="http://127.0.0.1:9/v1beta",
                                          model_api_key="gbench-relay-test")
    assert removed2 == removed
    assert "GEMINI_API_KEY" not in suite and "GEMINI_API_KEYS" not in suite
    for secret in (GEM_A, GEM_B, GEM_C, "vllm-key"):
        assert all(secret not in v for v in suite.values())
    assert suite["VLLM_API_KEY"] == "gbench-relay-test"
    assert Q._gemini_keys(base) == [GEM_B, GEM_C]                   # GEMINI_API_KEYS wins
    assert Q._gemini_keys({"GEMINI_API_KEY": f"{GEM_A},{GEM_C}"}) == [GEM_A, GEM_C]


@pytest.fixture
def qcfg(tmp_path):
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "results"
    config.results_dir.mkdir()
    config.gemmaclaw_commit = "0123456789abcdef0123456789abcdef01234567"
    config.gemmaclaw_path = str(tmp_path / "gemmaclaw")
    (tmp_path / "gemmaclaw").mkdir()
    config.remote_endpoint = "http://gbench-test.invalid/v1"
    config.initialize()
    return config


@pytest.fixture
def qmodel():
    return ModelConfig(short_name="gemma-4-E2B-it", hf_model_id="google/gemma-4-E2B-it",
                       total_params_b=5.2, is_moe=False)


def test_quality_suite_runs_gemini_alt_model_in_a_gemini_only_env(qcfg, qmodel, monkeypatch):
    for k, v in FOREIGN_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    monkeypatch.setenv("GEMINI_API_KEYS", f"{GEM_B},{GEM_C}")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-TEST-key")
    runner = Q.QualityBenchmarkRunner(qcfg)
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._cleanup_server = mock.MagicMock()
    runner._remove_container = mock.MagicMock()
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="c0ffee")
    runner._image = "gbench-quality:test"
    patch_record = {"file": Q.GEMINI_IMAGE_PATCH_FILE, "change": "c", "state": "applied",
                    "sha256": "0" * 64}
    # the image's edit records, and the suite's outputs read back from its container
    runner._image_metadata = lambda image, dest: {"patches": [patch_record], "commit": "c0ffee"}
    runner._collect_container_outputs = lambda name: {
        "summary": {"counts": {"total": 4, "passed": 3, "failed": 1, "blocked": 0}},
        "scenario_summaries": {}, "entrypoint": {"command_exit": 0,
                                                 "forwarder_alive_at_end": True},
        "forwarder_errors": []}
    calls = []

    def fake_popen(argv, stdin=None, stdout=None, stderr=None, env=None, **kw):
        container_env = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv)
                             if a == "--env")
        calls.append((list(argv), container_env, dict(env or {})))
        proc = mock.MagicMock()
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        proc.returncode = 0
        return proc

    monkeypatch.setattr(Q.subprocess, "Popen", fake_popen)
    with mock.patch("gbench.runners.evals._server_max_model_len", return_value=32768):
        res = runner.run(qmodel, ModelFormat.HF)
    [(cmd, env, docker_env)] = calls
    assert cmd[cmd.index("--alt-model") + 1] == Q.GEMINI_ALT_MODEL
    assert cmd[cmd.index("--model") + 1].startswith("vllm/")
    assert "openai/gpt-5.4" not in cmd and "openai/gpt-image-1" not in cmd
    assert not set(env) & set(FOREIGN_ENV)
    # no Gemini key and no endpoint key reach the suite: they stay with gbench's key relay
    assert "GEMINI_API_KEYS" not in env and "GEMINI_API_KEY" not in env
    for secret in (GEM_A, GEM_B, GEM_C, "vllm-TEST-key", *FOREIGN_ENV.values()):
        assert all(secret not in v for v in env.values())
        assert secret not in json.dumps(cmd) and secret not in json.dumps(docker_env)
    assert env["VLLM_API_KEY"] == Q.KEY_PLACEHOLDER
    assert cmd[cmd.index("--network") + 1] == "none"          # inside gbench's container
    assert env["OPENCLAW_ENABLE_PRIVATE_QA_CLI"] == "1"
    assert res["pass_rate"] == 75.0
    assert res["alternate_model"] == Q.GEMINI_ALT_MODEL
    assert res["image_generation_model"] == Q.GEMINI_IMAGE_GENERATION_MODEL
    subs = {s["role"]: s for s in res["gemini_substitutions"]}
    assert subs["alternate model"]["upstream"] == "openai/gpt-5.4"
    assert subs["alternate model"]["gemini_model"] == Q.GEMINI_ALT_MODEL
    assert subs["image-generation model"]["upstream"] == "openai/gpt-image-1"
    assert subs["image-generation model"]["gemini_model"] == Q.GEMINI_IMAGE_GENERATION_MODEL
    assert all(s["api"].startswith("Gemini API") for s in subs.values())
    assert res["gemmaclaw_patches"] == [patch_record]
    removed = res["non_gemini_env_removed"]
    assert removed == sorted(removed) and set(FOREIGN_ENV) <= set(removed)
    assert all(Q._is_provider_env(n) for n in removed)      # names of provider variables only
    assert "GEMINI_API_KEY" not in removed and "VLLM_API_KEY" not in removed
    assert res["leaderboard_comparable"] is False
    for m in (Q.GEMINI_ALT_MODEL, Q.GEMINI_IMAGE_GENERATION_MODEL, "openai/gpt-5.4",
              "openai/gpt-image-1"):
        assert m in res["leaderboard_comparable_reason"]
    saved = Path(qcfg.log_manager.get_quality_result_path(
        qmodel.short_name, ModelFormat.HF.value)).read_text()
    for secret in (GEM_A, GEM_B, GEM_C, *FOREIGN_ENV.values()):
        assert secret not in saved and secret not in json.dumps(res)


def test_quality_setup_subprocesses_carry_no_provider_key(qcfg, monkeypatch):
    for k, v in FOREIGN_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GEMINI_API_KEY", GEM_A)
    env = Q.QualityBenchmarkRunner(qcfg)._git_rewrite_safe_env()
    assert not set(env) & set(FOREIGN_ENV)
    assert "GEMINI_API_KEY" not in env and "GEMINI_API_KEYS" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def _upstream_index(repo: Path) -> Path:
    path = repo / Q.GEMINI_IMAGE_PATCH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('export const liveFrontierProviderDefinition = {\n'
                    '  mode: "live-frontier",\n'
                    '  defaultModel: (options) => options?.preferredLiveModel ?? '
                    '"openai/gpt-5.4",\n'
                    + Q._IMAGE_PATCH_OLD
                    + "  usesFastModeByDefault: isQaFastModeModelRef,\n};\n", encoding="utf-8")
    return path


def test_quality_image_patch_applies_once_reverts_and_refuses_drift(tmp_path):
    path = _upstream_index(tmp_path)
    original = path.read_text()
    rec = Q.apply_gemini_image_patch(tmp_path)
    patched = path.read_text()
    assert rec["state"] == "applied" and rec["file"] == Q.GEMINI_IMAGE_PATCH_FILE
    assert '"google/gemini-3.1-flash-image-preview"' in patched
    assert '["google"]' in patched and "openai/gpt-image-1" not in patched
    import hashlib
    assert rec["sha256"] == hashlib.sha256(patched.encode()).hexdigest()
    assert Q.apply_gemini_image_patch(tmp_path)["state"] == "already applied"
    assert path.read_text() == patched
    Q.revert_gemini_image_patch(tmp_path)
    assert path.read_text() == original
    path.write_text(original.replace('"openai/gpt-image-1"', '"openai/gpt-image-2"'))
    with pytest.raises(RuntimeError, match="infra_required"):
        Q.apply_gemini_image_patch(tmp_path)
    with pytest.raises(RuntimeError, match="infra_required"):
        Q.apply_gemini_image_patch(tmp_path / "missing")


def _upstream_api(repo: Path) -> Path:
    path = repo / Q.GEMINI_RELAY_PATCH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(Q._RELAY_PATCH_OLD_1 + "  let url: URL;\n  return normalized;\n}\n\n"
                    "export function f(params) {\n  return g({\n" + Q._RELAY_PATCH_OLD_2
                    + "  });\n}\n", encoding="utf-8")
    return path


def test_quality_setup_patches_the_image_build_context_and_never_the_checkout(qcfg,
                                                                            monkeypatch):
    repo = Path(qcfg.gemmaclaw_path)
    path = _upstream_index(repo)
    api = _upstream_api(repo)

    def git(*args):
        return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                              cwd=str(repo), check=True, capture_output=True,
                              text=True).stdout.strip()
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "upstream")
    sha = git("rev-parse", "HEAD")
    qcfg.gemmaclaw_commit = sha
    before = path.read_text() + api.read_text()
    for k, v in FOREIGN_ENV.items():
        monkeypatch.setenv(k, v)
    real_run = Q.subprocess.run
    seen = []

    def fake_run(cmd, *a, env=None, **kw):
        seen.append((list(cmd), dict(env or {})))
        if cmd[0] != "docker":
            return real_run(cmd, *a, env=env, **kw)            # git runs for real
        if cmd[1:3] == ["image", "inspect"]:
            built = any(c[:2] == ["docker", "build"] for c, _e in seen)
            return subprocess.CompletedProcess(cmd, 0 if built else 1,
                                               sha + "\n" if built else "", "")
        ctx = Path(cmd[-1])
        seen.append((["context"], {"text": (ctx / "src" / Q.GEMINI_IMAGE_PATCH_FILE).read_text()
                                   + (ctx / "src" / Q.GEMINI_RELAY_PATCH_FILE).read_text()}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    runner = Q.QualityBenchmarkRunner(qcfg)
    assert runner._setup_gemmaclaw_repo() == sha
    [build_text] = [e["text"] for c, e in seen if c == ["context"]]
    assert Q._IMAGE_PATCH_NEW in build_text              # patched in the image's build context
    assert Q._RELAY_PATCH_NEW_1 in build_text and Q._RELAY_PATCH_NEW_2 in build_text
    assert path.read_text() + api.read_text() == before  # the checkout is only read
    git_cmds = [c for c, _e in seen if c[0] == "git"]
    assert git_cmds and not any("checkout" in c for c in git_cmds)
    for c, env in seen:
        if c[0] in ("git", "docker"):
            assert not set(env) & set(FOREIGN_ENV)
    assert runner._image == Q.quality_image_tag(sha)


def _preflight_stubs(monkeypatch):
    monkeypatch.setattr(Q.shutil, "which", lambda x: "/usr/bin/" + x)

    def fake_run(cmd, **kw):
        out = {"node": "v22.19.0\n", "npx": "10.0.0\n", "git": "git version 2.43.0\n"}
        return subprocess.CompletedProcess(cmd, 0, out.get(os.path.basename(cmd[0]), ""), "")

    monkeypatch.setattr(Q.subprocess, "run", fake_run)


def test_quality_preflight_requires_a_gemini_key_and_names_it(qcfg, qmodel, monkeypatch):
    _preflight_stubs(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI)          # no substitute for a Gemini key
    runner = Q.QualityBenchmarkRunner(qcfg)
    assert runner._preflight_check() is False
    assert "GEMINI_API_KEYS" in runner._preflight_failure
    assert "GEMINI_API_KEY" in runner._preflight_failure
    with pytest.raises(RuntimeError, match="infra_required: neither GEMINI_API_KEYS nor "
                                           "GEMINI_API_KEY is set"):
        runner.run(qmodel, ModelFormat.HF)
    monkeypatch.setenv("GEMINI_API_KEYS", GEM_B)
    assert runner._preflight_check() is True and runner._preflight_failure is None


def test_quality_doc_matches_the_runner():
    doc = _read("docs", "quality.md")
    for needle in (Q.GEMINI_ALT_MODEL, Q.GEMINI_IMAGE_GENERATION_MODEL, Q.UPSTREAM_ALT_MODEL,
                   Q.UPSTREAM_IMAGE_GENERATION_MODEL, Q.GEMINI_IMAGE_PATCH_FILE,
                   "`gemini_substitutions`", "`gemmaclaw_patches`", "`non_gemini_env_removed`",
                   "`leaderboard_comparable: false`", "--alt-model"):
        assert needle in doc, needle
    for line in Q._IMAGE_PATCH_OLD.splitlines() + Q._IMAGE_PATCH_NEW.splitlines():
        assert line in doc, line
    for prefix in Q._PROVIDER_ENV_PREFIXES:
        assert f"`{prefix}`" in doc, prefix
    for suffix in Q._PROVIDER_ENV_SUFFIXES:
        assert f"`{suffix}`" in doc, suffix
    for name in Q._PROVIDER_ENV_NAMES | Q._KEPT_ENV:
        assert f"`{name}`" in doc, name
