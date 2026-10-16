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

"""Canonical-fidelity fixes for tau2 / tau3, gdpval, toolbench and mcp_bench.

  * tau2/tau3 - every run writes to its own directory (eval, model, thinking mode, config hash,
                invocation id) and never resumes: auto_resume is off and tau2's resume prompt is
                answered "n". The tau2-bench revision is pinned and checked. Leaderboard
                protocol by default: 4 trials, pass^k per domain, headline = pass^1 of the
                leaderboard track (tau2: mean of retail/airline/telecom; tau3: banking). The
                Gemini retry patches never touch the agent.
  * gdpval    - a judge reply without a valid winner is an outage, never a tie; .zip files are
                opened and their members rendered (text, documents, images for inputs), also
                when the archive is truncated; 3 generations per task (paper footnote 2).
  * toolbench - StableToolBench's 3 evaluation passes by default, with the per-group spread.
  * docs      - pins, protocols and the upstream per-turn token caps are stated.

No network, no Docker, no model server, no judge API: everything external is stubbed.
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import zipfile
from unittest import mock

import pytest

from gbench.runners.eval_suites import gdpval as G
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(__file__)


def _doc(name):
    with open(os.path.join(_HERE, "..", "docs", "evals", name), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# tau2 / tau3: a fake tau2 package that behaves like upstream's run_domain
# --------------------------------------------------------------------------- #
_TAU_ENV = ("GBENCH_TAU2_TEMPERATURE", "TAU2_TEMPERATURE", "GBENCH_TAU3_TEMPERATURE",
            "GBENCH_TAU2_TOP_P", "TAU2_TOP_P", "GBENCH_TAU2_TOP_K", "TAU2_TOP_K",
            "GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
            "GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", "GBENCH_TAU2_USER_TEMPERATURE",
            "TAU2_USER_TEMPERATURE", "GBENCH_TAU2_SAVE_TRACES", "TAU2_SAVE_TRACES",
            "GBENCH_RESULTS_DIR")


class _FakeTau2:
    """run_domain as upstream does it: results go to `<save_to>/results.json` (an absolute
    save_to replaces tau2's data/simulations dir); an existing file is resumed when auto_resume
    is set (its simulations are returned as the run's) and refused otherwise. Two tasks per
    domain; a model whose name ends in "-good" solves every trial, any other model none."""

    def __init__(self):
        self.configs = []

    def modules(self):
        fake = self

        class TextRunConfig:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        def run_domain(config):
            fake.configs.append(config)
            path = os.path.join(config.save_to, "results.json")
            if os.path.exists(path):
                if not config.auto_resume:
                    raise FileExistsError(f"File {path} already exists.")
                with open(path) as fh:
                    rewards = json.load(fh)
            else:
                good = config.llm_agent.endswith("-good")
                rewards = [1.0 if good else 0.0] * (2 * config.num_trials)
                os.makedirs(config.save_to, exist_ok=True)
                with open(path, "w") as fh:
                    json.dump(rewards, fh)
            sims = [types.SimpleNamespace(reward_info=types.SimpleNamespace(reward=r),
                                          termination_reason="agent_stop") for r in rewards]
            return types.SimpleNamespace(simulations=sims, num_trials=config.num_trials)

        def compute_metrics(results):
            rate = sum(s.reward_info.reward for s in results.simulations) / len(results.simulations)
            return types.SimpleNamespace(
                pass_hat_ks={k: rate for k in range(1, results.num_trials + 1)})

        mods = {n: types.ModuleType(n) for n in (
            "tau2", "tau2.data_model", "tau2.data_model.simulation", "tau2.run", "tau2.metrics",
            "tau2.metrics.agent_metrics")}
        mods["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
        mods["tau2.run"].run_domain = run_domain
        mods["tau2.metrics.agent_metrics"].compute_metrics = compute_metrics
        return mods


@pytest.fixture
def fake_tau(monkeypatch, tmp_path):
    for k in _TAU_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("check_banking_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_install_credential_guards", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1")):
        monkeypatch.setattr(TC, name, value)

    def banking():
        TC._LAST_RETRIEVAL.update({"retrieval_config": "alltools",
                                   "retrieval_variant": "alltools-gemini",
                                   "embedders": ["gemini:gemini-embedding-001"],
                                   "embed_model": "gemini:gemini-embedding-001"})
        return "alltools-gemini", None

    monkeypatch.setattr(TC, "_setup_banking_retrieval", banking)
    fake = _FakeTau2()
    with mock.patch.dict(sys.modules, fake.modules()):
        yield fake


def _run(model, domains=("airline",), eval_name="tau2", thinking=False, limit=None):
    return TC.run_tau_env(eval_name, list(domains), model, "http://x/v1", 2, limit,
                          f"docs/evals/{eval_name}.md", enable_thinking=thinking)


def test_two_models_in_one_results_dir_never_share_simulations(fake_tau, tmp_path):
    good = _run("org/model-A-good")
    bad = _run("org/model-B")
    assert good["accuracy"] == 100.0
    assert bad["accuracy"] == 0.0, "model B must not be scored with model A's simulations"
    paths = [c.save_to for c in fake_tau.configs]
    assert len(set(paths)) == 2 and all(os.path.isabs(p) for p in paths)
    assert all(c.auto_resume is False for c in fake_tau.configs)
    root = os.path.join(str(tmp_path), "results", "tau_traces")
    names = sorted(os.listdir(root))
    assert names[0].startswith("tau2_org_model-A-good_nothink_")
    assert names[1].startswith("tau2_org_model-B_nothink_")
    assert good["tau2_report"]["traces"]["airline"] == os.path.join(paths[0], "results.json")


def test_thinking_toggle_and_rerun_get_their_own_directories(fake_tau):
    _run("m-good")
    _run("m-good")
    _run("m-good", thinking=True)
    paths = [c.save_to for c in fake_tau.configs]
    assert len(set(paths)) == 3, "every invocation writes to a fresh directory"
    assert "_think_" in paths[2] and "_nothink_" in paths[0]


def test_config_changes_change_the_config_hash(fake_tau, monkeypatch):
    a = _run("m-good")["tau2_report"]
    monkeypatch.setenv("GBENCH_TAU2_SEED", "301")
    b = _run("m-good")["tau2_report"]
    assert a["config_hash"] != b["config_hash"]
    assert a["run_config"]["seed"] == 300 and b["run_config"]["seed"] == 301
    assert a["run_config"]["tau2_revision"] == TC.TAU2_BENCH_REVISION


def test_an_existing_save_path_is_refused_not_resumed(fake_tau, tmp_path):
    (tmp_path / "run" / "airline").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="never resumes"):
        TC._run_one_domain("airline", "m", "http://x/v1", 2, None, False, "tau2",
                           save_dir=str(tmp_path / "run"))
    assert fake_tau.configs == [], "tau2 must not even be called"


def test_run_one_domain_without_a_save_dir_uses_a_fresh_directory(fake_tau):
    TC._run_one_domain("airline", "m", "http://x/v1", 2, None, False, "tau2")
    TC._run_one_domain("airline", "m", "http://x/v1", 2, None, False, "tau2")
    a, b = (c.save_to for c in fake_tau.configs)
    assert a != b and os.path.isabs(a)


def test_disabled_traces_still_write_to_a_unique_temp_dir_that_is_removed(fake_tau, monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", "")
    r = _run("m-good")
    save_to = fake_tau.configs[0].save_to
    assert r["tau2_report"]["run_dir"] is None and r["tau2_report"]["traces"] is None
    assert not os.path.exists(save_to), "the temporary run directory is removed"


def test_relative_trace_dir_is_made_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", "traces")
    assert TC._resolve_tau_trace_dir() == str(tmp_path / "traces")


def test_resume_prompt_is_answered_no(monkeypatch):
    console = types.SimpleNamespace(input=lambda *a, **k: "y")
    display = types.ModuleType("tau2.utils.display")
    display.ConsoleDisplay = type("ConsoleDisplay", (), {"console": console})
    with mock.patch.dict(sys.modules, {"tau2": types.ModuleType("tau2"),
                                       "tau2.utils": types.ModuleType("tau2.utils"),
                                       "tau2.utils.display": display}):
        TC._refuse_resume_prompts()
    assert console.input("resume the run? (y/n)") == "n"
    import inspect
    assert '"y"' not in inspect.getsource(TC._install_robustness_patches)


def test_default_protocol_is_four_trials_and_upstream_defaults(fake_tau):
    r = _run("m-good")
    cfg = fake_tau.configs[0]
    assert (cfg.num_trials, cfg.max_steps, cfg.max_errors, cfg.seed) == (4, 200, 10, 300)
    assert r["tau2_report"]["num_trials"] == 4
    assert set(r["pass_k"]) == {"pass^1", "pass^2", "pass^3", "pass^4"}
    assert not hasattr(cfg, "hallucination_retries"), "upstream's default applies"


def test_fewer_trials_are_a_named_gap(fake_tau, monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "1")
    r = _run("m-good")
    assert "1 trial(s)" in r["leaderboard_comparable_reason"]
    assert set(r["pass_k"]) == {"pass^1"}


def test_tau2_headline_is_the_track_overall(fake_tau, monkeypatch):
    rates = {"retail": 0.7, "airline": 0.4, "telecom": 0.6}

    def one(domain, *a, **k):
        TC._build_agent_args("http://x/v1", False, "tau2")
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": rates[domain], "pass^2": rates[domain] / 2}
        return 8, 8 * rates[domain], int(8 * rates[domain]), 0

    monkeypatch.setattr(TC, "_run_one_domain", one)
    r = _run("m", domains=TC.CORE_DOMAINS)
    assert r["accuracy"] == pytest.approx(56.67) and r["pass1_domain_mean"] == r["accuracy"]
    assert r["pass_k"] == {"pass^1": pytest.approx(56.67), "pass^2": pytest.approx(28.33)}
    assert r["leaderboard_track"].startswith("tau2-bench (Overall")
    assert r["category_accuracy"]["airline"]["accuracy"] == 40.0


def test_tau3_headline_is_banking_pass1(fake_tau, monkeypatch):
    from gbench.runners.eval_suites import tau3 as T3
    monkeypatch.setattr(T3, "run_tau_suite", TC.run_tau_env)   # what the image runs
    r = T3.run_tau3("m-good", "http://x/v1", 2)
    assert r["accuracy"] == 100.0 and r["leaderboard_track"].startswith("tau3-banking")
    assert fake_tau.configs[0].domain == "banking_knowledge"
    assert fake_tau.configs[0].retrieval_config == "alltools-gemini"
    assert "retrieval config" not in r["leaderboard_comparable_reason"]
    assert TC.GEMINI_EMBED_REASON in r["leaderboard_comparable_reason"]


def test_single_domain_is_a_per_domain_column_not_the_overall(fake_tau):
    r = _run("m-good", domains=("retail",))
    assert "per-domain" in r["leaderboard_track"]


def test_domain_with_only_infra_errors_is_a_hard_error(fake_tau, monkeypatch):
    def one(domain, *a, **k):
        return 4, 0.0, 0, 4          # every simulation an infrastructure error, no pass^k

    monkeypatch.setattr(TC, "_run_one_domain", one)
    with pytest.raises(RuntimeError, match="no evaluable simulation"):
        _run("m")


def test_default_gemini_user_and_judge_are_named(fake_tau):
    reason = _run("m-good")["leaderboard_comparable_reason"]
    assert "user simulator gemini/" in reason and "Gemini cascade" in reason
    assert "NL-assertion judge gemini/" in reason
    assert "re-send" not in reason and "for the agent" not in reason


def test_is_infra_error_reads_the_termination_reason():
    sim = types.SimpleNamespace(termination_reason=types.SimpleNamespace(value="infrastructure_error"),
                                reward_info=types.SimpleNamespace(reward=0.0))
    assert TC._is_infra_error(sim)
    assert TC._is_infra_error(types.SimpleNamespace(termination_reason="agent_stop",
                                                    reward_info=None))
    assert not TC._is_infra_error(types.SimpleNamespace(
        termination_reason="agent_stop", reward_info=types.SimpleNamespace(reward=1.0)))


# --------------------------------------------------------------------------- #
# tau2 revision pin
# --------------------------------------------------------------------------- #
def _prov(**kw):
    base = {"package_dir": "/c/src/tau2", "checkout": "/c", "commit": TC.TAU2_BENCH_REVISION,
            "modified": [], "data_dir": os.path.realpath("/c/data"), "version": "1.0.1"}
    base.update(kw)
    return base


@pytest.mark.parametrize("prov, fragment", [
    (_prov(), None),
    (_prov(commit="5ebebbe8"), "not the pinned"),
    (_prov(checkout=None), "not the src/tau2 of a git checkout"),
    (_prov(modified=["src/tau2/config.py"]), "local changes"),
    (_prov(data_dir="/elsewhere/data"), "TAU2_DATA_DIR"),
])
def test_revision_check(monkeypatch, prov, fragment):
    monkeypatch.setattr(TC, "tau2_provenance", lambda: prov)
    ok, why = TC.check_tau2_revision()
    if fragment is None:
        assert ok and why == ""
    else:
        assert not ok and fragment in why
        if fragment in ("not the pinned", "not the src/tau2 of a git checkout"):
            assert TC.TAU2_BENCH_REVISION in why and "docker build" in why


def test_prerequisites_include_the_revision_check(monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "openai/gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "openai/gpt-4.1-2025-04-14")
    monkeypatch.setattr(TC, "_import_tau2", lambda: (True, ""))
    monkeypatch.setattr(TC, "check_tau2_revision", lambda: (False, "wrong revision"))
    assert TC.check_tau_env_prerequisites() == (False, "wrong revision")


def test_legacy_diagnostic_urls_use_the_pinned_revision():
    # The single-turn diagnostics no longer fetch tasks from GitHub: the only task data a tau
    # run reads is the pinned checkout's, in the image.
    import inspect
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    assert not hasattr(T2, "SIERRA_TAU2_DOMAINS") and not hasattr(T3, "SIERRA_TAU3_TELECOM_URL")
    for mod in (T2, T3):
        assert "raw.githubusercontent.com" not in inspect.getsource(mod)


# --------------------------------------------------------------------------- #
# tau2: the Gemini retry patches never touch the agent
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=content, tool_calls=None))]


def test_retry_and_fence_patches_apply_only_to_gemini_calls(monkeypatch):
    calls = []

    def completion(*a, **k):
        calls.append(k["model"])
        return _Resp("" if len(calls) < 3 else "ok")

    gen_calls = []

    def generate(model, messages, **k):
        gen_calls.append(model)
        return types.SimpleNamespace(content="```\nhello\n```", tool_calls=None)

    litellm = types.ModuleType("litellm")
    litellm.completion = completion
    llm_utils = types.ModuleType("tau2.utils.llm_utils")
    llm_utils.generate = generate
    llm_utils.completion = completion
    mods = {"litellm": litellm, "tau2": types.ModuleType("tau2"),
            "tau2.utils": types.ModuleType("tau2.utils"), "tau2.utils.llm_utils": llm_utils}
    monkeypatch.setattr(TC, "_PATCHED", False)
    monkeypatch.setattr(TC, "_RETRY_PATCHES", {"completion": False, "generate": False})
    monkeypatch.setattr(TC, "_build_gemini_cascade", lambda inner, backoff: inner)
    # LiteLLM's thinking-level guard and the per-verdict judge record need the real LiteLLM /
    # tau2 classes (tests/test_tau_gemini_cascade_and_relays.py covers both)
    monkeypatch.setattr(TC, "_install_gemini_param_guard", lambda: None)
    monkeypatch.setattr(TC, "_install_judge_model_record", lambda: False)
    with mock.patch.dict(sys.modules, mods):
        TC._install_robustness_patches()
        agent = litellm.completion(model="openai/m", messages=[])
        assert calls == ["openai/m"] and agent.choices[0].message.content == "", \
            "the agent's empty reply is used as it is, sent once"
        calls.clear()
        user = litellm.completion(model="gemini/x", messages=[])
        assert calls == ["gemini/x"] * 3 and user.choices[0].message.content == "ok"
        assert llm_utils.generate("openai/m", []).content == "```\nhello\n```"
        assert llm_utils.generate("gemini/x", []).content == "hello"
    assert gen_calls == ["openai/m", "gemini/x"]


# --------------------------------------------------------------------------- #
# gdpval: judge verdicts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply, winner", [
    ('{"winner": "A", "reason": "x"}', "a"),
    ('verdict: {"winner": "tie"}', "tie"),
    ('{"winner": " b "}', "b"),
    ('{"reason": "the reference is clearly better"}', None),
    ('{"winner": null}', None),
    ('{"winner": "Deliverable B"}', None),
    ('{"winner": "reference"}', None),
    ('["A"]', None),
    ("A is better", None),
    ("", None),
])
def test_parse_winner_accepts_only_a_b_or_tie(reply, winner):
    assert G._parse_winner(reply) == winner


def _gd_trace(task_id, text="model answer"):
    return {"extra_payload": {"task_id": task_id},
            "gold_answer": json.dumps([{"score": 1, "criterion": "c"}]),
            "response_text": text, "messages": [{"role": "user", "content": task_id}]}


def _gd_score(traces, reply):
    refs = {t["extra_payload"]["task_id"]: {"text": "expert", "note": None, "n_files": 1}
            for t in traces}
    metrics = {}

    async def fake(prompt):
        return reply, "gemini-x"

    with mock.patch.object(G, "judge_generate_cascade", fake):
        asyncio.run(G._make_scorer("j", 2, metrics, refs)(traces))
    return metrics["gdpval_report"]


@pytest.mark.parametrize("reply", ['{"reason": "the reference is clearly better"}',
                                   '{"winner": "Deliverable B"}', '{"winner": null}'])
def test_reply_without_a_valid_winner_is_an_outage_never_a_tie(reply):
    traces = [_gd_trace("t1")]
    rep = _gd_score(traces, reply)
    assert rep["ties"] == 0 and rep["win_rate"] is None
    assert rep["tasks_judge_outage"] == 1 and rep["comparisons_judge_outage"] == 1
    assert rep["judge_invalid_replies"] == 2
    assert traces[0]["judge_grade"] == "JUDGE_OUTAGE" and not traces[0].get("is_correct")


def test_scorer_counts_tasks_and_comparisons_separately():
    traces = [_gd_trace(t) for t in ("a", "a", "a", "b", "b", "b")]
    rep = _gd_score(traces, '{"winner": "tie"}')
    assert rep["comparisons_in_win_rate"] == 6 and rep["tasks_in_win_rate"] == 2
    assert rep["ties"] == 6 and rep["win_rate"] == 1.0


def test_gdpval_generates_the_papers_three_samples_per_task(monkeypatch):
    seen = {}

    def fake_suite(**kw):
        seen.update(kw)
        return {"total_questions": 0, "temperature": 0.0}

    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(G, "_load_gdpval_samples", lambda limit=None: ([], {}))
    monkeypatch.setattr(G, "run_eval_suite", fake_suite)
    r = G.run_gdpval("m", "http://x", 1)
    assert seen["attempt_count"] == 3 and seen["supports_attempts"] is True
    assert r["samples_per_task"] == 3
    assert "greedy (temperature 0.0)" in r["leaderboard_comparable_reason"]
    G.run_gdpval("m", "http://x", 1, attempt_count=5)
    assert seen["attempt_count"] == 5
    G.run_gdpval("m", "http://x", 1, attempt_count=1)
    assert seen["attempt_count"] == 3


# --------------------------------------------------------------------------- #
# gdpval: .zip files
# --------------------------------------------------------------------------- #
_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00"
        b"\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
        b"\x00\x00\x00IEND\xaeB`\x82")


def _zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return str(path)


def test_zip_deliverable_renders_its_text_members(tmp_path):
    local = _zip(tmp_path / "app.zip", {
        "app/src/Status.tsx": "export const Status = () => <div role='status'/>;\n",
        "app/main.tf": 'resource "aws_lambda_function" "f" {}\n',
        "app/Dockerfile": "FROM node:20\n",
        "app/package-lock.json": '{"lockfileVersion": 3}',
        "app/node_modules/react/index.js": "module.exports = REACT_SOURCE;",
        "__MACOSX/app/._main.tf": "\x00\x05\x16\x07",
        "app/.DS_Store": "\x00\x00",
        "app/logo.png": _PNG,
        "app/part.step": "ISO-10303-21;",
    })
    text, note = G.render_local_file(local, "app.zip")
    assert note is None
    assert "export const Status" in text and "aws_lambda_function" in text and "FROM node:20" in text
    assert "REACT_SOURCE" not in text and "lockfileVersion" not in text and "ISO-10303" not in text
    assert "### app/src/Status.tsx" in text
    assert "4 OS/VCS/dependency/lockfile file(s) not rendered" in text
    assert "1 .step" in text and "1 image(s)" in text
    assert "archive contents" not in text, "no bare file listing"


def test_zip_with_only_audio_has_no_text_reference(tmp_path):
    local = _zip(tmp_path / "stems.zip", {"STEMS/BASS.wav": b"RIFF....WAVE",
                                         "__MACOSX/STEMS/._BASS.wav": b"\x00"})
    with mock.patch.object(G, "_fetch", return_value=local):
        text, note = G.render_reference(["deliverable_files/x/stems.zip"])
    assert text is None and "no text member" in note and ".wav" in note
    traces = [_gd_trace("t")]
    metrics = {}
    refs = {"t": {"text": text, "note": note, "n_files": 1}}
    asyncio.run(G._make_scorer("j", 1, metrics, refs)(traces))
    assert metrics["gdpval_report"]["tasks_no_reference_by_reason"]["no_text_rendering"] == 1


class _Unseekable(io.RawIOBase):
    """A write-only stream without seek, so zipfile writes data descriptors (flag bit 3)."""

    def __init__(self):
        self.buf = bytearray()

    def writable(self):
        return True

    def write(self, b):
        self.buf += bytes(b)
        return len(b)


def test_truncated_zip_is_read_from_its_local_headers(tmp_path):
    out = _Unseekable()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("proj/README.md", "# Private mixer\n" * 20)
        z.writestr("proj/contracts/Mixer.sol", "contract Mixer { }\n" * 20)
        z.writestr("proj/big.js", base64.b64encode(os.urandom(30000)).decode("ascii"))
    data = bytes(out.buf)
    cut = data.index(b"proj/big.js") + 200        # break inside the last member, no end record
    p = tmp_path / "cut.zip"
    p.write_bytes(data[:cut])
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(str(p))
    text, note = G.render_local_file(str(p), "cut.zip")
    assert note is None and "# Private mixer" in text and "contract Mixer" in text
    assert "### proj/big.js" not in text, "the cut-off member is not rendered"
    assert "no readable central directory" in text and "up to the first damaged one" in text


def test_zip_input_photographs_reach_the_model_as_images(tmp_path):
    local = _zip(tmp_path / "Photographs.zip", {"Photographs/a.png": _PNG,
                                                "Photographs/b.jpg": b"\xff\xd8\xff\xe0jpeg",
                                                "Photographs/notes.txt": "taken July 2"})
    with mock.patch.object(G, "_fetch", return_value=local):
        parts, rendered, unavailable = G.attachment_parts(["reference_files/x/Photographs.zip"])
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert (rendered, unavailable) == (1, 0)
    assert urls[0].startswith("data:image/png;base64,")
    assert urls[1].startswith("data:image/jpeg;base64,")
    assert "taken July 2" in parts[0]["text"]


def test_zip_input_with_only_video_is_a_note(tmp_path):
    local = _zip(tmp_path / "reel.zip", {"reel/a.mp4": b"\x00\x00\x00\x18ftypmp42",
                                         "reel/b.mov": b"\x00\x00\x00\x14ftypqt"})
    with mock.patch.object(G, "_fetch", return_value=local):
        parts, rendered, unavailable = G.attachment_parts(["reference_files/x/reel.zip"])
    assert (rendered, unavailable) == (0, 1) and "nothing this suite can render" in parts[0]["text"]


def test_audio_inputs_are_sent_as_input_audio_parts(tmp_path):
    wav = tmp_path / "take.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    reel = _zip(tmp_path / "reel.zip", {"reel/a.mp4": b"\x00\x00\x00\x18ftypmp42",
                                        "reel/boom.mp3": b"ID3\x03"})
    paths = {"reference_files/x/take.wav": str(wav), "reference_files/x/reel.zip": reel}
    with mock.patch.object(G, "_fetch", side_effect=lambda rel: paths[rel]):
        parts, rendered, unavailable = G.attachment_parts(list(paths))
    audio = [p["input_audio"] for p in parts if p["type"] == "input_audio"]
    assert (rendered, unavailable) == (2, 0)
    assert [a["format"] for a in audio] == ["wav", "mp3"]
    assert audio[0]["data"] == base64.b64encode(wav.read_bytes()).decode("ascii")
    assert any("1 .mp4" in p.get("text", "") for p in parts)


def test_zip_member_document_needs_its_converter(tmp_path, monkeypatch):
    monkeypatch.setitem(G._CONVERTER_PACKAGES, "pptx", ("gbench_no_such_module", "python-pptx"))
    local = _zip(tmp_path / "deck.zip", {"deck/Slides.pptx": b"PK"})
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*python-pptx"):
        G.render_local_file(local, "deck.zip")


def test_zip_member_notebook_is_rendered_with_the_notebook_converter(tmp_path):
    nb = json.dumps({"cells": [{"cell_type": "code", "source": "print(42)", "outputs": []}]})
    local = _zip(tmp_path / "nb.zip", {"work/analysis.ipynb": nb})
    text, note = G.render_local_file(local, "nb.zip")
    assert note is None and "print(42)" in text and "### work/analysis.ipynb" in text


# --------------------------------------------------------------------------- #
# toolbench: StableToolBench's 3 evaluation passes
# --------------------------------------------------------------------------- #
def _tb(solved):
    return {"query": "q", "answer": {"final_answer": "S" if solved else "N", "answer_details": []}}


def test_sopr_is_the_mean_over_the_passes_with_its_spread(monkeypatch):
    seq = iter(["Solved", "Unsolved", "Solved", "Unsure", "Solved", "Solved"])

    async def status(query, answer, sem):
        return next(seq)

    monkeypatch.setattr(T, "_answer_status", status)
    # 2 queries x 3 passes: per-pass solve rates 0.5, 0.75, 1.0 (gather keeps pass order)
    r = asyncio.run(T._score_all({"1": _tb(True), "2": _tb(True)}, {}, 3, 1))
    assert r["sopr"] == pytest.approx(0.75) and r["sopr_passes"] == 3
    assert r["sopr_std"] == pytest.approx((((0.25 ** 2) * 2) / 3) ** 0.5)


def _run_toolbench(monkeypatch, eval_times=None):
    async def status(query, answer, sem):
        return "Solved"

    async def pref(query, cand, ref, sem):
        return "cand"

    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_served_model_id", lambda endpoint, fallback: fallback)
    # the bridge network's gateway (after the metadata guard) and the image's records of its
    # tool calls: stubbed, no docker (the relays listen on loopback)
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: "127.0.0.1")
    monkeypatch.setattr(T, "read_server_calls", lambda path: [])
    monkeypatch.setattr(T, "read_observations", lambda path: {})
    # gbench's Gemini relay (started by the run) needs a key; it never reaches the container
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTfixtureKEY000000000000000000000")
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR",
              "GBENCH_TOOLBENCH_EVAL_TIMES"):
        monkeypatch.delenv(k, raising=False)
    if eval_times is not None:
        monkeypatch.setenv("GBENCH_TOOLBENCH_EVAL_TIMES", eval_times)
    # The reference answers (one per test id, as SoWR needs) and the image's test ids, stubbed.
    ref_dir = tempfile.TemporaryDirectory(prefix="gbench_test_tbref_")
    os.makedirs(os.path.join(ref_dir.name, T._REFERENCE_DEFAULT))
    for g in T._GROUPS:
        with open(os.path.join(ref_dir.name, T._REFERENCE_DEFAULT, f"{g}.json"), "w") as fh:
            json.dump({"1": _tb(False)}, fh)
    monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: ref_dir.name)
    monkeypatch.setattr(T, "_group_test_ids", lambda image, groups, limit: {g: ["1"] for g in groups})

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
            for g in T._GROUPS:
                with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                    json.dump({"1": _tb(True)}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    with ref_dir:
        return T.run_toolbench("m", "http://x:8000/v1", 2)


def test_toolbench_evaluates_three_times_by_default(monkeypatch):
    r = _run_toolbench(monkeypatch)
    assert r["evaluate_times"] == 3
    assert set(r["sopr_passes_by_group"].values()) == {3}
    assert set(r["sopr_std_by_group"].values()) == {0.0}
    assert "evaluation pass" not in r["leaderboard_comparable_reason"]


def test_toolbench_other_evaluation_counts_are_named(monkeypatch):
    r = _run_toolbench(monkeypatch, eval_times="1")
    assert "1 evaluation pass(es) instead of StableToolBench's 3" in r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("value", ["0", "-2", "three"])
def test_toolbench_invalid_evaluation_count_is_a_hard_error(monkeypatch, value):
    with pytest.raises(RuntimeError, match="GBENCH_TOOLBENCH_EVAL_TIMES"):
        _run_toolbench(monkeypatch, eval_times=value)


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_tau_docs_state_the_pin_protocol_and_run_directories():
    tau2, tau3 = _doc("tau2.md"), _doc("tau3.md")
    assert TC.TAU2_BENCH_REVISION in tau2 and TC.TAU2_BENCH_REVISION in tau3
    flat = " ".join(tau2.split())
    assert "4 trials per task" in flat and "tau2-bench leaderboard track's Overall" in flat
    assert "never allowed to resume" in tau2 and "tau3-banking" in tau3
    assert "1, upstream's default" not in tau2
    assert "--max-output-tokens" in tau2 and "--max-output-tokens" in tau3


def test_gdpval_doc_counts_and_protocol():
    doc = " ".join(_doc("gdpval.md").split())
    assert "39 of the 220 tasks" in doc and "The other 181 enter the rate" in doc
    assert "gbench generates **once**" not in doc and "**3 times**" in doc
    assert "never as a tie" in doc and "--max-output-tokens" in doc


def test_delegated_harness_docs_name_upstream_token_caps():
    assert "max_tokens=1024" in _doc("toolbench.md")
    assert "planning_tokens: 12000" in _doc("mcp_bench.md")
    tb = _doc("toolbench.md")
    assert "| `GBENCH_TOOLBENCH_EVAL_TIMES` | `3` |" in tb
    assert "repeats add nothing" not in tb
