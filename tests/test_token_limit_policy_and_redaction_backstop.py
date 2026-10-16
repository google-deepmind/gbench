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

"""Shared-file contracts: the token-limit policy, the credential backstop after serialization,
and the shared docs and registries in step with the code.

  * token-limit policy (base.resolve_token_budget): a no-think request sends the official cap
    exactly (or none when the official sampler sends none); a --thinking request is never
    capped below --max-output-tokens, an official cap is recorded as not applied to reasoning,
    and explicit reasoning-model settings are used when declared; suites that declare no
    official cap are unchanged;
  * a declared per-request max_tokens is sent as the clamped value, and a reply that stops at an
    official cap is the protocol's outcome, not a truncation;
  * every writer under the results directory redacts the final text it writes, and files that
    suites, harnesses or subprocesses write there themselves are swept;
  * the redaction set: URL passwords but not bare user names, no placeholders, registered and
    huggingface_hub-file tokens, and a log call that never raises under concurrent environment
    writes.

No model server, judge API or dataset hub is contacted; every file is written under tmp_path.
"""

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# token-limit policy
# --------------------------------------------------------------------------- #
def _run(official=None, *, thinking=False, knob=4096, meta=None, window=None, finish="stop",
         tool_executor=None, eval_name="aime", n=1, extra_payload=None):
    """run_eval_suite with a fake transport; returns (requests seen, result)."""
    seen = []

    async def send(**k):
        async with k["semaphore"]:
            pass
        seen.append(k)
        return base.Reply("x " * 10, None, finish, completion_tokens=5)
    base._RUN_KNOBS["max_output_tokens"] = knob
    base._RUN_KNOBS["max_model_len"] = window
    samples = [([{"role": "user", "content": f"q{i}"}], "1", dict(meta or {})) for i in range(n)]
    with mock.patch.object(base, "_send_single_request", send):
        result = base.run_eval_suite(eval_name=eval_name, model_name="m", base_url="http://x",
                                     concurrency=1, samples=samples, eval_fn=lambda r, g: True,
                                     thinking=thinking, official_sampling=official,
                                     tool_executor=tool_executor, extra_payload=extra_payload)
    return seen, result


CAP = base.OfficialSampling(reference="upstream sampler (max_tokens=100)", temperature=0.0,
                            max_tokens=100)


def test_no_think_sends_the_official_cap_exactly():
    seen, res = _run(CAP, knob=65536)
    assert [k["max_output_tokens"] for k in seen] == [100]
    assert res["max_tokens_sent"] == [100]
    assert res["max_output_tokens_note"].startswith("--max-output-tokens 65536 not applied")
    assert "max_tokens=100" in res["max_output_tokens_note"]
    rec = res["official_sampling"]
    assert rec["max_tokens"] == 100 and rec["max_tokens_applied"] is True
    # no floor either: a cap below the 16,384 house floor is sent as is
    seen, _ = _run(CAP, knob=None)
    assert [k["max_output_tokens"] for k in seen] == [100]


def test_no_think_official_cap_is_not_cut_by_the_tool_loop_ceiling():
    big = base.OfficialSampling(reference="agent loop (max_tokens=20000)", max_tokens=20000)

    async def no_tools(name, args):
        return "{}"
    seen, res = _run(big, knob=65536, tool_executor=no_tools)
    assert [k["max_output_tokens"] for k in seen] == [20000]
    assert res["tool_loop_protocol"]["per_turn_max_tokens"] == 20000


def test_thinking_does_not_apply_a_cap_written_for_non_reasoning_generation():
    seen, res = _run(CAP, thinking=True, knob=65536)
    assert [k["max_output_tokens"] for k in seen] == [65536]
    assert base.REASONING_CAP_NOTE in res["max_output_tokens_note"]
    assert "--max-output-tokens 65536" in res["max_output_tokens_note"]
    rec = res["official_sampling"]
    assert rec["max_tokens"] == 100 and rec["max_tokens_applied"] is False
    assert rec["max_tokens_note"] == base.REASONING_CAP_NOTE and rec["max_tokens_sent"] == 65536
    # without --max-output-tokens the thinking floor is the budget
    seen, res = _run(CAP, thinking=True, knob=None)
    assert [k["max_output_tokens"] for k in seen] == [base.THINKING_MIN_OUTPUT_TOKENS]
    assert "thinking floor" in res["max_output_tokens_note"]


def test_thinking_uses_explicit_reasoning_settings():
    none_sent = base.OfficialSampling(
        reference="chat sampler (T=0.5, max_tokens=2048)", temperature=0.5, max_tokens=2048,
        reasoning=base.OfficialSampling(reference="o-series sampler", send_temperature=False,
                                        send_max_tokens=False))
    seen, res = _run(none_sent, thinking=True, knob=16384)
    assert [k["max_output_tokens"] for k in seen] == [None]
    # the reasoning settings' max_tokens apply; the temperature is the thinking rule's 1.0
    assert [k["temperature"] for k in seen] == [1.0]
    rec = res["official_sampling"]
    assert rec["reasoning_settings"] is True and rec["reference"] == "o-series sampler"
    assert rec["send_max_tokens"] is False
    assert res["max_output_tokens_note"].startswith(
        "not sent: the official reasoning-model sampler sends no max_tokens")
    # the chat settings still govern the no-think run
    seen, res = _run(none_sent, thinking=False, knob=16384)
    # (the temperature is the no-think rule's 0.0, not the chat sampler's 0.5)
    assert [(k["max_output_tokens"], k["temperature"]) for k in seen] == [(2048, 0.0)]
    assert "reasoning_settings" not in res["official_sampling"]


def test_an_explicit_reasoning_cap_is_never_below_max_output_tokens():
    reasoning_cap = base.OfficialSampling(
        reference="chat", temperature=0.2, max_tokens=2000,
        reasoning=base.OfficialSampling(reference="reasoning (max_completion_tokens=25000)",
                                        send_temperature=False, max_tokens=25000))
    seen, res = _run(reasoning_cap, thinking=True, knob=65536)
    assert [k["max_output_tokens"] for k in seen] == [65536]
    assert "never capped below --max-output-tokens" in res["max_output_tokens_note"]
    assert res["official_sampling"]["max_tokens_sent"] == 65536
    seen, res = _run(reasoning_cap, thinking=True, knob=16384)
    assert [k["max_output_tokens"] for k in seen] == [25000]
    assert res["official_sampling"]["max_tokens_applied"] is True


@pytest.mark.parametrize("thinking", [False, True])
def test_suites_without_an_official_cap_are_unchanged(thinking):
    plain = base.OfficialSampling(reference="upstream", temperature=0.0)
    for official in (plain, None):
        seen, res = _run(official, thinking=thinking, knob=8192)
        assert [k["max_output_tokens"] for k in seen] == [8192]
        assert "max_output_tokens_note" not in res
        if official is not None:
            assert "max_tokens" not in res["official_sampling"]
        seen, _ = _run(official, thinking=thinking, knob=None)
        floor = base.THINKING_MIN_OUTPUT_TOKENS if thinking else base.DEFAULT_MIN_OUTPUT_TOKENS
        assert [k["max_output_tokens"] for k in seen] == [floor]


def test_resolve_token_budget_serves_suites_outside_run_eval_suite():
    plan = base.resolve_token_budget("acebench", CAP, thinking=False, max_output_tokens=4096)
    assert (plan.policy, plan.max_tokens) == ("official", 100)
    plan = base.resolve_token_budget("acebench", CAP, thinking=True, max_output_tokens=4096)
    assert (plan.policy, plan.max_tokens) == ("reasoning_run_budget", 4096)
    assert plan.record()["max_tokens_applied"] is False
    none = base.OfficialSampling(reference="agent", send_max_tokens=False)
    plan = base.resolve_token_budget("mcp_atlas", none, thinking=True, max_output_tokens=4096)
    assert plan.uncapped and plan.max_tokens is None
    assert base.resolve_token_budget("x", None, thinking=True).policy == "house"


def test_resolve_official_sampling_uses_the_reasoning_settings_when_thinking():
    both = base.OfficialSampling(reference="chat", temperature=0.5,
                                 reasoning=base.OfficialSampling(reference="r",
                                                                 send_temperature=False))
    assert base.resolve_official_sampling("gpqa", both) == (0.0, "default")
    # thinking: 1.0 whatever the reasoning settings say (the maintainer's rule)
    assert base.resolve_official_sampling("gpqa", both, thinking=True) == (1.0, "thinking")


@pytest.mark.parametrize("bad", [
    dict(max_tokens=0), dict(max_tokens="100"), dict(max_tokens=True),
    dict(max_tokens=100, send_max_tokens=False), dict(params={"max_tokens": 5})])
def test_official_caps_are_validated(bad):
    with pytest.raises(ValueError):
        base.check_official_sampling_params("x", base.OfficialSampling(reference="r", **bad))
    reasoning = base.OfficialSampling(reference="c", reasoning=base.OfficialSampling(
        reference="r", **bad))
    with pytest.raises(ValueError):
        base.check_official_sampling_params("x", reasoning)


# --------------------------------------------------------------------------- #
# per-request declared caps
# --------------------------------------------------------------------------- #
def test_a_declared_cap_is_sent_without_thinking_and_lifted_with_it():
    seen, res = _run(None, knob=65536, meta={"max_tokens": 256}, n=2)
    assert [k["max_output_tokens"] for k in seen] == [256, 256]
    assert all("max_tokens" not in k["extra_payload"] for k in seen)
    assert "official_cap_not_applied_to_reasoning" not in res
    seen, res = _run(None, thinking=True, knob=65536, meta={"max_tokens": 256}, n=2)
    assert [k["max_output_tokens"] for k in seen] == [65536, 65536]
    lifted = res["official_cap_not_applied_to_reasoning"]
    assert lifted == {"declared_max_tokens": [256], "samples": 2, "sent": 65536}
    assert base.REASONING_CAP_NOTE in res["max_output_tokens_note"]
    assert res["sample_traces"][0]["official_cap_not_applied"] == 256
    # a declared cap above the run's budget is replaced too, as an OfficialSampling.max_tokens is
    seen, res = _run(None, thinking=True, knob=4096, meta={"max_tokens": 32768})
    assert [k["max_output_tokens"] for k in seen] == [4096]
    assert res["official_cap_not_applied_to_reasoning"] == {
        "declared_max_tokens": [32768], "samples": 1, "sent": 4096}
    assert "replies_at_official_max_tokens" not in res


def test_the_clamped_value_of_a_declared_cap_is_what_the_wire_carries(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (1500, 1500, "server"))
    wire = []

    class _Resp:
        status = 200

        async def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1}}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        def post(self, url, json=None):
            wire.append(json)
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False
    base._RUN_KNOBS["max_output_tokens"] = 4096
    base._RUN_KNOBS["max_model_len"] = 2100
    with mock.patch.object(base.aiohttp, "ClientSession", _Session):
        res = base.run_eval_suite(eval_name="gsm8k", model_name="m", base_url="http://x",
                                  concurrency=1,
                                  samples=[([{"role": "user", "content": "q"}], "1",
                                            {"max_tokens": 2000})],
                                  eval_fn=lambda r, g: True)
    assert [p["max_tokens"] for p in wire] == [res["sample_traces"][0]["max_tokens_sent"]]
    assert wire[0]["max_tokens"] < 2000


def test_replies_at_an_official_cap_are_the_protocols_outcome(caplog):
    with caplog.at_level("WARNING"):
        seen, res = _run(CAP, knob=65536, finish="length", n=4)
    assert res["replies_at_official_max_tokens"] == 4 and res["truncated_responses"] == 4
    assert res["status"] == "success"
    assert not [r for r in caplog.records if "Raise --max-output-tokens" in r.getMessage()]
    # the same replies at a budget that is not an official cap are still a measurement problem
    seen, res = _run(None, knob=65536, finish="length", n=4)
    assert res["status"] == "completed_with_errors"
    assert "replies_at_official_max_tokens" not in res


def test_uncapped_truncation_names_the_context_window(caplog):
    none = base.OfficialSampling(reference="agent", send_max_tokens=False)
    with caplog.at_level("WARNING"):
        seen, res = _run(none, knob=65536, finish="length")
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("reached the end of the context window (no max_tokens is sent" in m for m in msgs)
    assert not any("Raise --max-output-tokens" in m or "max_tokens=None" in m for m in msgs)
    # the request timeout is sized for an uncapped reply when the window is unknown
    assert res["request_timeout_s"] == base.request_timeout_s(base.UNCAPPED_TIMEOUT_TOKENS)
    seen, res = _run(none, knob=65536, window=131072)
    assert [k["max_output_tokens"] for k in seen] == [None]
    assert res["request_timeout_s"] == base.request_timeout_s(131072)


# --------------------------------------------------------------------------- #
# credential redaction: the set of values
# --------------------------------------------------------------------------- #
# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
KEY = "sk-" + "FAKEtest00000000key0001"


def test_url_userinfo_contributes_only_a_password():
    env = {"DOCKER_HOST": "ssh://postgres@buildhost",
           "DB_URL": "postgresql://reader:pw-FAKE-123456@db/x"}
    values = base.secret_values(env)
    assert "postgres" not in values and "reader" not in values
    assert "pw-FAKE-123456" in values


def test_placeholders_are_not_secrets():
    env = {"OPENAI_API_KEY": "placeholder", "VLLM_API_KEY": "not-needed",
           "X_API_KEY": "sk-no-key-required", "Y_TOKEN": "anything", "Z_API_KEY": "EMPTY"}
    assert base.secret_values(env) == []


def test_registered_and_token_file_credentials_are_redacted(tmp_path, monkeypatch):
    token_file = tmp_path / "hf_token"
    token_file.write_text("hf_FAKEfileToken000000001\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_file))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    base.register_secret("gcp-identity-FAKE-000000001")
    base.register_secret("EMPTY")                      # a placeholder is never registered
    values = base.secret_values()
    assert "hf_FAKEfileToken000000001" in values and "gcp-identity-FAKE-000000001" in values
    assert "EMPTY" not in values
    text = base.redact_text("a hf_FAKEfileToken000000001 b gcp-identity-FAKE-000000001")
    assert text == f"a {base.REDACTED} b {base.REDACTED}"


def test_url_encoded_forms_and_colliding_keys(monkeypatch):
    monkeypatch.setenv("ODD_API_KEY", "abc/def+ghi=jkl")
    for form in ("abc%2Fdef%2Bghi%3Djkl", "abc/def%2Bghi%3Djkl"):
        assert "abc" not in base.redact_text(f"?key={form}&x=1")
    monkeypatch.setenv("A_TOKEN", "tokA-FAKE-00000001")
    monkeypatch.setenv("B_TOKEN", "tokB-FAKE-00000002")
    out = base.redact_secrets({"tokA-FAKE-00000001": 1, "tokB-FAKE-00000002": 2})
    assert sorted(out.values()) == [1, 2] and len(out) == 2
    assert all(k.startswith(base.REDACTED) for k in out)


def test_a_log_call_never_raises_while_other_threads_change_the_environment():
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    log = logging.getLogger("gbench.test.env_race")
    handler = _Keep()
    log.addHandler(handler)
    log.propagate = False
    log.setLevel(logging.INFO)
    old = logging.getLogRecordFactory()
    stop = threading.Event()
    errors = []

    def churn():
        i = 0
        while not stop.is_set():
            os.environ[f"GBENCH_TEST_ENV_RACE_{i % 50}"] = "v" * 12
            os.environ.pop(f"GBENCH_TEST_ENV_RACE_{(i + 25) % 50}", None)
            i += 1
    t = threading.Thread(target=churn, daemon=True)
    try:
        base.install_log_redaction(log)
        t.start()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                log.info("hello %s", 1)
            except Exception as e:                                         # noqa: BLE001
                errors.append(e)
    finally:
        stop.set()
        t.join()
        for i in range(50):
            os.environ.pop(f"GBENCH_TEST_ENV_RACE_{i}", None)
        logging.setLogRecordFactory(old)
        log.removeHandler(handler)
        log.propagate = True
        log.setLevel(logging.NOTSET)
    assert not errors and records


# --------------------------------------------------------------------------- #
# credential redaction: every writer under the results directory
# --------------------------------------------------------------------------- #
class _Carrier:
    def __str__(self):
        return f"object holding {KEY}"


def _leaks(root: Path, needles):
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            hits += [(str(path), n) for n in needles if n.encode() in data]
    return hits


def _needles():
    from urllib.parse import quote
    return [KEY, quote(KEY, safe="")]


def test_log_manager_redacts_after_serialization(tmp_path, monkeypatch):
    from gbench.utils.log_manager import LogManager
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    lm = LogManager(tmp_path / "results")
    lm.save_result(lm.results_dir / "r.json", {
        "set": {f"k={KEY}"}, "error": RuntimeError(KEY), "path": Path(f"/tmp/{KEY}"),
        "object": _Carrier(), "bytes": KEY.encode(), "tuple": (KEY,)})
    lm.save_metadata({"cmd": ["x", _Carrier()]})
    lm.save_summary({"exc": ValueError(f"url ?key={KEY}")})
    lm.save_csv("rows.csv", [{"a": _Carrier(), "b": Path(KEY)}], ["a", "b"])
    lm.save_log(lm.results_dir / "logs" / "x.log", f"out {KEY}".encode(), RuntimeError(KEY))
    assert _leaks(tmp_path, _needles()) == []
    data = json.loads((lm.results_dir / "r.json").read_text(encoding="utf-8"))
    assert base.REDACTED in data["error"] and base.REDACTED in data["object"]


def test_eval_summary_csv_redacts_after_serialization(tmp_path, monkeypatch):
    from gbench import cli
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    row = {"eval_name": "mmlu", "model_short": _Carrier(), "format": "hf", "status": "success",
           "total_questions": 1, "correct_answers": 1, "accuracy": 100.0,
           "category_accuracy": {}, "headline_metrics": {"note": Path(KEY)}}
    path = cli._save_eval_summary_csv(tmp_path, [row], [], [], {})
    assert path is not None and _leaks(tmp_path, _needles()) == []


def test_serving_stress_and_quality_results_use_the_redacting_writer(tmp_path, monkeypatch):
    import ast
    for rel in ("gbench/runners/serving.py", "gbench/runners/stress.py",
                "gbench/runners/quality.py"):
        tree = ast.parse(_read(rel))
        dumps = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "dump"
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "json"]
        # quality.py still writes two files outside the results directory: the edit records
        # in the image's build context and the run's openclaw.json, both in temporary directories
        allowed = 2 if rel.endswith("quality.py") else 0
        assert len(dumps) == allowed, rel
        assert "write_redacted_json(" in _read(rel), rel
    from gbench.runners.stress import StressTestRunner
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    runner = object.__new__(StressTestRunner)
    runner.config = mock.MagicMock()
    runner.config.log_manager.results_dir = tmp_path
    model = mock.MagicMock()
    model.short_name = "m"
    fmt = mock.MagicMock()
    fmt.value = "hf"
    runner._save_stress_result({"error": RuntimeError(KEY), "levels": [{"x": _Carrier()}]},
                               model, fmt, False)
    assert (tmp_path / "performance" / "stress_m_hf_text.json").exists()
    assert _leaks(tmp_path, _needles()) == []


def test_redact_tree_rewrites_what_others_wrote_and_skips_links(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    root = tmp_path / "results"
    (root / "agent_trajectories" / "s").mkdir(parents=True)
    traj = root / "agent_trajectories" / "s" / "t.traj.json"
    traj.write_text(json.dumps({"model_kwargs": {"api_key": KEY}}), encoding="utf-8")
    blob = root / "sub.bin"
    blob.write_bytes(b"\x00\x01" + KEY.encode() + b"\xff")
    outside = tmp_path / "outside.txt"
    outside.write_text(KEY, encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    os.mkfifo(root / "pipe")
    report = base.redact_tree(root)
    assert sorted(report["redacted"]) == sorted([str(traj), str(blob)])
    assert KEY not in traj.read_text(encoding="utf-8") and KEY.encode() not in blob.read_bytes()
    assert outside.read_text(encoding="utf-8") == KEY       # a symlink is not followed
    assert base.redact_tree(root) == {"redacted": [], "unwritable": []}
    locked = root / "locked.log"
    locked.write_text(f"key {KEY}", encoding="utf-8")
    locked.chmod(0o444)
    (root).chmod(0o555)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running as a user that can write a read-only file")
        report = base.redact_tree(root)
    finally:
        root.chmod(0o755)
        locked.chmod(0o644)
    assert report["unwritable"] == [str(locked)]


def test_a_run_sweeps_files_suites_write_themselves(tmp_path, monkeypatch):
    """A stand-in suite writes a trajectory and a submission of its own under
    GBENCH_RESULTS_DIR (the way minisweagent, gaia and tau do): after the run no file under the
    results directory holds the key, in any form."""
    from gbench.core.config import BenchmarkConfig
    from gbench.core.models import ModelFormat
    from gbench.runners.eval_suites import SUITES
    from gbench.runners.evals import EvalsBenchmarkRunner
    from urllib.parse import quote
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)

    def writing_suite(**kw):
        run_dir = Path(os.environ["GBENCH_RESULTS_DIR"])
        traj = run_dir / "agent_trajectories" / "mmlu_x" / "task.traj.json"
        traj.parent.mkdir(parents=True)
        traj.write_text(json.dumps({"model_kwargs": {"api_key": KEY}}), encoding="utf-8")
        sub = run_dir / "gaia_submission" / "answers.jsonl"
        sub.parent.mkdir(parents=True)
        sub.write_text(f'{{"url": "http://h/?key={quote(KEY, safe="")}"}}\n', encoding="utf-8")
        return {"status": "success", "total_questions": 1, "correct_answers": 1,
                "accuracy": 100.0, "category_accuracy": {}, "error_obj": RuntimeError(KEY)}

    def raising_suite(**kw):
        (Path(os.environ["GBENCH_RESULTS_DIR"]) / "harness.log").write_text(
            f"Authorization: Bearer {KEY}", encoding="utf-8")
        raise RuntimeError("harness failed")

    config = BenchmarkConfig()
    config.evals = ["mmlu", "gsm8k"]
    config.remote_endpoint = "http://127.0.0.1:1/v1"
    config.batch_sizes = [1]
    config.skip_existing = False
    config.eval_max_output_tokens = 256
    config.results_dir = tmp_path / "results"
    config.initialize()
    model = mock.MagicMock()
    model.name = model.short_name = model.hf_model_id = "m"
    with mock.patch.dict(SUITES, {"mmlu": writing_suite, "gsm8k": raising_suite}), \
         mock.patch("gbench.runners.eval_suites.discover_and_register_plugins", return_value={}), \
         mock.patch("gbench.runners.evals._resolve_remote_model_id", return_value="m"), \
         mock.patch("gbench.runners.evals._server_max_model_len", return_value=None):
        results = EvalsBenchmarkRunner(config).run(model, ModelFormat.HF)
    assert {r["eval_name"]: r["status"] for r in results} == {"mmlu": "success",
                                                               "gsm8k": "error"}
    assert _leaks(tmp_path / "results", _needles()) == []
    traj = next((tmp_path / "results").rglob("task.traj.json"))
    assert base.REDACTED in traj.read_text(encoding="utf-8")


def test_the_cli_sweeps_its_results_directory(tmp_path, monkeypatch):
    from gbench import cli
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "server_m_hf.log").write_text(f"api_key={KEY}", encoding="utf-8")
    config = mock.MagicMock()
    config.log_manager.results_dir = tmp_path
    cli._redact_run_results(config)
    assert _leaks(tmp_path, _needles()) == []
    src = _read("gbench/cli.py")
    main_src = src[src.index("def main("):src.index("def _redact_run_results(")]
    assert main_src.index("print_results_summary(all_results, config)") < \
        main_src.index("_redact_run_results(config)")


def test_gcp_identity_tokens_are_registered():
    for rel in ("gbench/runners/quality.py", "gbench/runners/golden.py"):
        assert "register_secret(" in _read(rel), rel


def test_a_declared_cap_records_the_unapplied_flag():
    seen, res = _run(None, knob=65536, meta={"max_tokens": 256})
    assert res["official_max_tokens_sent"] == [256]
    assert res["max_output_tokens_note"] == (
        "--max-output-tokens 65536 not applied to the requests that carry the suite's official "
        "max_tokens (256)")


# --------------------------------------------------------------------------- #
# shared files in step with the code
# --------------------------------------------------------------------------- #
def test_swe_bench_live_takes_its_leaderboard_settings_as_categories():
    from gbench import cli
    from gbench.runners import evals as E
    from gbench.runners.eval_suites import swe_bench_live
    assert E.eval_category_names()["swe_bench_live"] == set(swe_bench_live.SETTINGS)
    assert E.route_eval_categories("verified,simple_python", "swe_bench_live") == "verified"
    assert E.route_eval_categories("verified", "bfcl") is None

    class Passed(Exception):
        pass
    with mock.patch.object(cli, "setup_logging", side_effect=Passed):
        with pytest.raises(Passed):
            cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                      "--max-output-tokens", "1024", "--evals", "swe_bench_live",
                      "--eval-categories", "verified"])
    with pytest.raises(SystemExit):
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "1024", "--evals", "mmlu",
                  "--eval-categories", "verified"])
    parser_help = cli.create_parser().format_help()
    assert "swe_bench_live takes its SWE-bench-Live leaderboard" in re.sub(r"\s+", " ",
                                                                          parser_help)


def test_list_pillars_tags_the_named_only_suites(capsys):
    from gbench import cli
    args = mock.MagicMock()
    args.list_what = "pillars"
    assert cli._handle_list(args) == 0
    out = capsys.readouterr().out
    assert "swe_lancer*" in out and "ui_control_osworld*" in out
    assert "* runs only when named; not in --evals all" in out
    assert "in --evals all)" in out


def test_matharena_helpers_take_the_served_models_recommended_sampling():
    from gbench.runners.eval_suites import matharena_common as MA
    qwen = base.OfficialSampling(reference="qwen3.5_27b.yaml", temperature=1.0,
                                 params={"top_p": 0.95, "top_k": 20})
    assert MA.request_payload(None, qwen) == {"top_p": 0.95, "top_k": 20}
    assert MA.request_payload(None) == dict(MA.SAMPLING.params)
    result = {"temperature": 1.0, "attempts": {"attempts_per_sample": 4}}
    assert MA.comparability_reasons(result, {}, None, payload={"top_p": 0.95, "top_k": 20},
                                    recommended=qwen) == []
    reasons = MA.comparability_reasons(result, {}, None, payload={"top_p": 0.95, "top_k": 64},
                                       recommended=qwen)
    assert reasons == ["sampling top_k=64; MathArena uses the provider's recommended top_k=20"]
    comp = MA.Competition(key="hmmt/x", name="X", dataset="d", revision="r")
    with mock.patch.object(MA, "image", return_value="gbench-matharena:b89f2f0a"):
        assert MA.matharena_record(comp, {}, qwen)["sampling"] == "qwen3.5_27b.yaml"
        assert MA.matharena_record(comp, {})["sampling"] == MA.SAMPLING.reference


def test_canonical_sync_follows_the_token_limit_policy():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    assert CS["swe_lancer"]["synced"] >= "2026-09-25"     # reconciled on or after that date
    assert "(Lite, Full, Verified)" in CS["swe_bench_live"]["upstream"]
    assert "@3da151be953883d3cd4e1ea0dc7a042ff2abf8f4" in CS["swe_bench_live"]["dataset"]
    for name in ("lcb", "multipl_e"):
        assert "official cap not applied to reasoning" in CS[name]["method"], name
    assert "reasoning-model branch" in CS["bigcodebench"]["method"]
    assert "evaluate_from_apiX.py" in CS["mmlu_pro"]["upstream"]
    assert "gbench-bfcl:2026.3.23" in CS["bfcl"]["upstream"]
    assert "JSONAdapter retry not replicated" not in CS["multipl_e"]["method"]
    assert "printed in full" in CS["putnam_formal"]["method"]
    for name, entry in CS.items():
        for value in entry.values():
            assert all(ord(c) < 128 for c in str(value)), name


def test_shared_docs_follow_the_token_limit_policy():
    # the asserted README rows moved to the CLI reference, suite catalog and env-var docs
    readme = "\n".join(_read(rel) for rel in (
        "README.md", "docs/cli-reference.md", "docs/eval-suites.md",
        "docs/environment-variables.md"))
    row = next(l for l in readme.splitlines() if l.startswith("| `--max-output-tokens` |"))
    assert "`base.resolve_token_budget`" in row and "not applied to reasoning" in row
    assert "Their one-shot modes apply it." in row and "max_output_tokens_note" in row
    lancer = next(l for l in readme.splitlines() if l.startswith("| | `swe_lancer` |"))
    assert "SimpleAgentSolver" in lancer and "`DOCKER-USER`" in lancer
    knobs = _read("docs/evals/env-knobs.md")
    policy = knobs[knobs.index("- Token-limit policy"):knobs.index("- Renamed suites still read")]
    for needle in ("official cap not applied to reasoning", "`OfficialSampling.reasoning`",
                   "replies_at_official_max_tokens", "official_cap_not_applied_to_reasoning"):
        assert needle in policy, needle
    bullet = knobs[knobs.index("- Credential values never reach"):]
    bullet = bullet[:bullet.index("## Cross cutting knobs")]
    for needle in ("a user name without a", "PLACEHOLDER_SECRET_VALUES", "URL-encoded",
                   "`redact_tree`", "`HF_TOKEN_PATH`", "`GBENCH_GCP_AUTH`"):
        assert needle in bullet, needle
    assert "GBENCH_MCP_BENCH_NO_SUBSET" not in knobs
    tools = _read("docs/evals/toolchains.md")
    assert "pip install mcp jsonschema" not in tools and "gbench-bfcl:2026.3.23" in tools
    assert "pip install bfcl-eval" not in tools
    roadmap = _read("docs/evals_roadmap.md")
    assert "JSONAdapter" not in roadmap
    notice = _read("NOTICE")
    assert "BSD 3-Clause License with Machine Learning Restriction (MultiPL-E;" not in notice
    fetched = notice[notice.index("Fetched at run time, not shipped"):]
    assert "docker/multipl_e.sha256" in fetched and "Wellesley College" in fetched
    assert "docker/toolbench_patch_simulator.py" in notice
    assert "evaluate_from_apiX.py" in notice
    for rel in ("docs/evals/env-knobs.md", "docs/evals/toolchains.md", "NOTICE",
                "docs/THIRD_PARTY_DATASETS.md"):
        assert not [c for c in _read(rel) if ord(c) > 127], rel
    changed_lines = [l for l in readme.splitlines()
                     if l.startswith(("| `--max-output-tokens` |", "| `--eval-categories` |",
                                      "| | `swe_bench_live` |", "| | `swe_lancer` |",
                                      "| | `bfcl", "| `GBENCH_SANDBOX` |"))]
    changed_lines += [l for l in _read("docs/RESOURCES.md").splitlines()
                      if "SWE-bench Live" in l or "BFCL" in l or "MCP-Bench" in l
                      or "ToolBench" in l]
    assert len(changed_lines) >= 10
    assert not [l for l in changed_lines if any(ord(c) > 127 for c in l)]


def test_dependencies_follow_the_token_limit_policy():
    import tomllib
    project = tomllib.loads(_read("pyproject.toml"))
    extras = project["project"]["optional-dependencies"]
    assert "bfcl-eval==2026.3.23" in extras["dev"]
    assert not any(d.startswith("bfcl-eval") for d in extras["evals"])
    assert "openai>=2.31" in extras["evals"]
    lock = _read("uv.lock")
    assert '{ name = "bfcl-eval", marker = "extra == \'dev\'", specifier = "==2026.3.23" }' in lock
    assert '{ name = "openai", marker = "extra == \'evals\'", specifier = ">=2.31" }' in lock


def test_a_reply_at_a_budget_larger_than_the_official_cap_is_still_a_truncation():
    reasoning_cap = base.OfficialSampling(
        reference="chat", max_tokens=2000,
        reasoning=base.OfficialSampling(reference="reasoning", max_tokens=25000))
    seen, res = _run(reasoning_cap, thinking=True, knob=65536, finish="length", n=4)
    assert [k["max_output_tokens"] for k in seen] == [65536] * 4
    assert res["status"] == "completed_with_errors"
    assert "replies_at_official_max_tokens" not in res
    seen, res = _run(reasoning_cap, thinking=True, knob=16384, finish="length", n=4)
    assert res["replies_at_official_max_tokens"] == 4 and res["status"] == "success"


def test_a_swept_file_keeps_its_inode_for_a_process_still_appending(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    log = tmp_path / "server.log"
    with open(log, "a") as writer:
        writer.write(f"args: api_key={KEY}\n")
        writer.flush()
        inode = log.stat().st_ino
        assert base.redact_file(str(log)) is True
        writer.write("later line\n")
        writer.flush()
    assert log.stat().st_ino == inode
    text = log.read_text(encoding="utf-8")
    assert KEY not in text and "later line" in text and base.REDACTED in text
