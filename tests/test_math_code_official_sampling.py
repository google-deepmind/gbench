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

"""Official sampling for the maths and code suites, LiveCodeBench's self-repair scenario,
MultiPL-E's run-time fetch of its chat script, and BigCodeBench's backend fields, sanitize step and
harness CPU bound.

Offline unless stated: datasets and the model endpoint are stubbed. The LCB graders run for real
(unsandboxed under conftest's GBENCH_SANDBOX=none), and MathArena's grader runs in this
interpreter (`run_driver(where="host")`) for answers that do not need sympy's LaTeX parser. One
MultiPL-E test downloads the real upstream script and runs only when GitHub is reachable.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import types

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bigcodebench as B
from gbench.runners.eval_suites import lcb as L
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import matharena_common as MA
from gbench.runners.eval_suites import multipl_e as M
from gbench.runners.eval_suites import ojbench as OJ
from gbench.runners.eval_suites import scicode as SC
from gbench.runners.eval_suites import tau_common as TC_GUARD


# --------------------------------------------------------------------------- #
# MathArena (hmmt / aime): Gemma 4's recommended sampling on every request     #
# --------------------------------------------------------------------------- #
def _run_hmmt_offline(monkeypatch, replies, last_chance_text, thinking):
    from gbench.runners.eval_suites import hmmt as H
    real = MA.run_driver
    monkeypatch.setattr(MA, "run_driver",
                        lambda mode, items, config, **kw: real(mode, items, config, where="host"))
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    from gbench.runners.eval_suites import ojbench as OJ
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",   # a Gemma 4 model is served
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(MA, "self_check", lambda config: None)
    rows = [{"problem_idx": 1, "problem": "one", "answer": "5", "problem_type": ["Algebra"]},
            {"problem_idx": 2, "problem": "two", "answer": "7", "problem_type": ["Geometry"]}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    sent = []

    async def fake_send(**k):
        sent.append(k)
        msgs = k["messages"]
        if msgs[-1]["content"] == MA.LAST_CHANCE_PROMPT:
            return base.Reply(text=last_chance_text, tool_calls=None, finish_reason="stop",
                              completion_tokens=2)
        problem = msgs[-1]["content"].rsplit("\n\n", 1)[1]
        return base.Reply(text=replies[problem], tool_calls=None, finish_reason="stop",
                          completion_tokens=3)

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    res = H.run_hmmt("m", "http://x/v1", 2, enable_thinking=thinking, max_output_tokens=4096)
    return res, sent


@pytest.mark.parametrize("thinking", [False, True])
def test_matharena_sends_the_recommended_sampling_on_first_and_last_chance_requests(
        monkeypatch, thinking):
    monkeypatch.delenv("GBENCH_HMMT_TEMPERATURE", raising=False)
    res, sent = _run_hmmt_offline(monkeypatch, {"one": "\\boxed{5}", "two": "seven"},
                                  "\\boxed{7}", thinking)
    lasts = [k for k in sent if k["messages"][-1]["content"] == MA.LAST_CHANCE_PROMPT]
    firsts = [k for k in sent if k not in lasts]
    assert len(firsts) == 8 and len(lasts) == 4
    # gbench's temperature (1.0 with thinking, 0.0 without); top_p and top_k are Gemma 4's
    want = 1.0 if thinking else 0.0
    for k in firsts + lasts:
        assert k["temperature"] == want
        assert k["extra_payload"]["top_p"] == 0.95 and k["extra_payload"]["top_k"] == 64
    rec = res["official_sampling"]
    assert rec["temperature"] == 1.0 and rec["params"] == {"top_p": 0.95, "top_k": 64}
    # Gemma 4's recommended 1.0 is also every thinking run's (source "thinking")
    assert rec["temperature_matches_official"] is thinking
    assert rec["temperature_source"] == ("thinking" if thinking else "default")
    assert res["temperature"] == want and res["leaderboard_comparable"] is thinking
    assert "generation_config.json" in res["matharena"]["sampling"]


def test_matharena_operator_temperature_reaches_both_requests_and_is_named(monkeypatch):
    monkeypatch.setenv("GBENCH_HMMT_TEMPERATURE", "0.0")
    res, sent = _run_hmmt_offline(monkeypatch, {"one": "\\boxed{5}", "two": "seven"},
                                  "\\boxed{7}", True)
    assert {k["temperature"] for k in sent} == {0.0}
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert "temperature=0.0" in res["leaderboard_comparable_reason"]


def test_matharena_names_a_changed_sampling_parameter():
    full = {"temperature": 1.0, "attempts": {"attempts_per_sample": 4}}
    assert MA.comparability_reasons(full, {}, None) == []
    changed = MA.comparability_reasons(full, {}, None,
                                       payload=MA.request_payload({"top_p": 0.5}))
    assert len(changed) == 1 and "top_p=0.5" in changed[0] and "top_p=0.95" in changed[0]
    pen = MA.comparability_reasons({**full, "decoding_penalties": {"repetition_penalty": 1.1}},
                                   {}, None)
    assert pen and "decoding penalties" in pen[0]


# --------------------------------------------------------------------------- #
# OJBench, SciCode, LiveBench, MultiPL-E, BigCodeBench: official sampling      #
# --------------------------------------------------------------------------- #
def _oj_trace(lang, ok):
    return {"extra_payload": {"row": {"id": 1, "language": lang, "difficulty": "easy"}},
            "response_text": "code", "is_correct": ok}


@pytest.mark.parametrize("temp,comparable", [(1.0, True), (0.7, False), (0.0, False)])
def test_ojbench_samples_with_the_models_recommended_parameters(monkeypatch, temp, comparable):
    monkeypatch.delenv("GBENCH_OJBENCH_SAMPLES", raising=False)
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",   # a Gemma 4 model is served
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "_load_ojbench_samples", lambda limit=None: [])
    seen = {}

    def fake_suite(**k):
        seen.update(k)
        traces = [_oj_trace(lang, a < 4) for lang in ("cpp", "python") for a in range(8)]
        return {"accuracy": 50.0, "temperature": temp, "status": "success",
                "scoring_excluded": 0, "max_tokens_sent": [65536], "sample_traces": traces}

    monkeypatch.setattr(OJ, "run_eval_suite", fake_suite)
    r = OJ.run_ojbench("m", "http://h/v1", 1, enable_thinking=True)
    official = seen["official_sampling"]
    assert official.temperature == 1.0 and official.params == {"top_p": 0.95, "top_k": 64}
    assert "temperature" not in seen          # resolved from the official setting, not the CLI
    assert r["leaderboard_comparable"] is comparable
    if not comparable:
        assert f"temperature={temp}" in r["leaderboard_comparable_reason"]


def test_scicode_official_temperature_is_zero_without_thinking_and_one_with_it(monkeypatch):
    monkeypatch.delenv("GBENCH_SCICODE_TEMPERATURE", raising=False)
    assert base.resolve_official_sampling("scicode", SC.OFFICIAL_SAMPLING,
                                          thinking=False) == (0.0, "default")
    # every thinking run samples at 1.0 (base.resolve_temperature), SciCode's 0 notwithstanding
    t, src = base.resolve_official_sampling("scicode", SC.OFFICIAL_SAMPLING, thinking=True)
    assert (t, src) == (1.0, "thinking")
    assert SC.OFFICIAL_SAMPLING.record(t, src)["temperature_matches_official"] is False
    monkeypatch.setenv("GBENCH_SCICODE_TEMPERATURE", "0.6")
    t, src = base.resolve_official_sampling("scicode", SC.OFFICIAL_SAMPLING, thinking=True)
    assert (t, src) == (0.6, "env:GBENCH_SCICODE_TEMPERATURE")
    rec = SC.OFFICIAL_SAMPLING.record(t, src)
    assert rec["temperature_matches_official"] is False and "gencode.py" in rec["reference"]


def _livebench_run(monkeypatch, tmp_path, temperature_env=None):
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "2025-04-02")
    monkeypatch.delenv("GBENCH_LIVEBENCH_CATEGORIES", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_EXTRA_ARGS", raising=False)
    if temperature_env is None:
        monkeypatch.delenv("GBENCH_LIVEBENCH_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("GBENCH_LIVEBENCH_TEMPERATURE", temperature_env)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None,
                                              "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    seen = {}

    def fake_run(cmd, **k):
        seen["inner"] = cmd[-1]
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\nm,50.0," + ",".join("50.0" for _ in cols)
            + "\n", encoding="utf-8")
        # instruction_following is scored: its answer rows are copied out too
        if_dir = (tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "instruction_following"
                  / "paraphrase" / "model_answer")
        if_dir.mkdir(parents=True, exist_ok=True)
        (if_dir / "m.jsonl").write_text(json.dumps(
            {"question_id": "i1", "choices": [{"index": 0, "turns": ["ok"]}],
             "total_output_tokens": 10}) + "\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return LB.run_livebench("m", "http://h/v1", 4, max_output_tokens=4096), seen["inner"]


def test_livebench_no_think_sends_zero_in_the_config_by_default(monkeypatch, tmp_path):
    r, inner = _livebench_run(monkeypatch, tmp_path)
    assert "--force-temperature" not in inner
    cfg = json.load(open(tmp_path / LB._MODEL_CONFIG_FILE))
    assert cfg["api_kwargs"]["default"]["temperature"] == 0.0
    assert r["temperature"] == 0.0 and r["temperature_source"] == "default"
    assert "required_temperature" in r["official_sampling"]["reference"]
    # get_answer's own temperature is 0, so the run stays comparable
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


def test_livebench_writes_an_operator_temperature_into_the_config_and_flags_it(monkeypatch,
                                                                              tmp_path):
    r, inner = _livebench_run(monkeypatch, tmp_path, temperature_env="0.7")
    assert "--force-temperature" not in inner
    cfg = json.load(open(tmp_path / LB._MODEL_CONFIG_FILE))
    assert cfg["api_kwargs"]["default"]["temperature"] == 0.7
    assert r["temperature"] == 0.7 and r["leaderboard_comparable"] is False
    assert ("temperature=0.7 (env:GBENCH_LIVEBENCH_TEMPERATURE; sent as the config's "
            "temperature)") in r["leaderboard_comparable_reason"]
    assert LB._build_command("i", "/o", "m", "d", "http://h/v1", 1, None, "", "")[-1].count(
        "--force-temperature") == 0


def test_multipl_e_official_sampling_is_the_chat_recipe(monkeypatch):
    monkeypatch.delenv("GBENCH_MULTIPL_E_SAMPLES", raising=False)
    monkeypatch.delenv("GBENCH_MULTIPL_E_LANGS", raising=False)
    monkeypatch.setattr(M, "check_multipl_e_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_load_multipl_e_samples", lambda **kw: [])
    seen = {}
    for temp, comparable in ((0.2, True), (0.0, False)):
        monkeypatch.setattr(M, "run_eval_suite", lambda **k: seen.update(k) or {
            "accuracy": 10.0, "temperature": temp, "status": "success", "sample_traces": [],
            "max_tokens_sent": [M.UPSTREAM_MAX_TOKENS]})
        r = M.run_multipl_e("m", "http://h/v1", 1)
        assert seen["official_sampling"].temperature == 0.2
        assert not seen["official_sampling"].params and "temperature" not in seen
        assert r["leaderboard_comparable"] is comparable
        assert r["chat_script"]["sha256"] == M.CHAT_SCRIPT_SHA256


def test_bigcodebench_each_path_sends_its_backend_fields(monkeypatch):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])
    seen = {}
    monkeypatch.setattr(B, "run_eval_suite", lambda **k: seen.update(k) or {
        "accuracy": 0.0, "temperature": 0.0, "status": "success", "sample_traces": []})
    B.run_bigcodebench("m", "http://h/v1", 1)                        # no-think: vllm path
    vllm = seen["official_sampling"]
    assert vllm.temperature == 0.0 and vllm.params["top_p"] == 1.0
    # upstream's 6 stop strings; the server accepts 4, and all 6 are applied to the reply
    assert B.EOS == ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__",
                     "\ndef main(", "\nprint("]
    assert vllm.params["stop"] == B.EOS[:4]
    B.run_bigcodebench("m", "http://h/v1", 1, enable_thinking=True)  # openai path
    assert seen["official_sampling"].params == {"top_p": 0.95}
    assert seen["official_sampling"].temperature == 0.0
    r = B.run_bigcodebench("m", "http://h/v1", 1, extra_payload={"top_p": 0.9})
    assert r["leaderboard_comparable"] is False and "top_p=0.9" in r["leaderboard_comparable_reason"]


def test_bigcodebench_prefill_request_carries_the_prefill_and_backend_fields():
    """The no-think request as base builds it: prefill fields (forwarded by base) plus the vllm
    backend's top_p and stop strings."""
    sent = {}

    class _Resp:
        status = 200

        async def json(self):
            return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                    "usage": {}}

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, url, json=None, **kw):
            sent.update(json or {})
            return _Resp()

    msgs = [{"role": "user", "content": "u"}, B.prefill_message()]
    payload = {**B.SAMPLING[True].params, **B._PREFILL_REQUEST}
    asyncio.run(base._send_single_request(
        session=_Session(), api_url="http://probe/chat/completions", model_name="m",
        messages=msgs, extra_payload=payload, semaphore=asyncio.Semaphore(1), pbar=None,
        max_output_tokens=16, temperature=0.0))
    assert sent["continue_final_message"] is True and sent["add_generation_prompt"] is False
    assert sent["stop"] == B.SERVER_STOP and sent["top_p"] == 1.0 and sent["temperature"] == 0.0
    assert base.prefill_fields(sent) == B._PREFILL_REQUEST
    assert B._prefill_supported() is True


def test_bigcodebench_sanitizes_like_generate_py_and_replaces_tabs_on_the_prefill_path(
        monkeypatch):
    calls, written = [], {}

    def fake_run(cmd, **k):
        calls.append(cmd)
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        if "bigcodebench.sanitize" in cmd:
            with open(os.path.join(mount, "samples.jsonl"), encoding="utf-8") as f:
                written.update({json.loads(l)["task_id"]: json.loads(l)["solution"] for l in f})
        else:
            json.dump({"eval": {t: [{"status": "pass", "details": {}}] for t in written}},
                      open(os.path.join(mount, "samples_eval_results.json"), "w"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    traces = [
        {"extra_payload": {"task_id": "BigCodeBench/0", "split": "instruct", **B._PREFILL_REQUEST},
         "response_text": "\ndef f():\n\treturn 1\n```", "category": "instruct"},
        {"extra_payload": {"task_id": "BigCodeBench/1", "split": "instruct"},
         "response_text": "```python\ndef g():\n\treturn 2\n```", "category": "instruct"},
    ]
    asyncio.run(B._make_scorer("full", 2, {}, "img")(traces))
    sanitize = next(c for c in calls if "bigcodebench.sanitize" in c)
    evaluate = next(c for c in calls if "bigcodebench.sanitize" not in c)
    assert "--calibrate" not in sanitize
    assert evaluate[evaluate.index("--samples") + 1] == "/app/samples-sanitized.jsonl"
    assert "--calibrated" not in evaluate                  # evaluate's default: calibrated=True
    assert written["BigCodeBench/0"] == "\ndef f():\n    return 1\n```"      # vllm path
    assert written["BigCodeBench/1"] == "```python\ndef g():\n\treturn 2\n```"  # openai path
    assert all(t["is_correct"] for t in traces)


def _evaluate_command(monkeypatch, parallel):
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        if "bigcodebench.sanitize" not in cmd:
            json.dump({"eval": {"BigCodeBench/0": [{"status": "pass", "details": {}}]}},
                      open(os.path.join(mount, "samples_eval_results.json"), "w"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    metrics = {}
    traces = [{"extra_payload": {"task_id": "BigCodeBench/0", "split": "instruct"},
               "response_text": "```python\ndef f():\n    return 1\n```", "category": "instruct"}]
    asyncio.run(B._make_scorer("full", parallel, metrics, "img")(traces))
    return next(c for c in calls if "bigcodebench.sanitize" not in c), metrics


def test_bigcodebench_harness_sees_a_bounded_cpu_set(monkeypatch):
    """On a many-CPU host the evaluate container gets the first MAX_HARNESS_CPUS CPUs and no
    more test workers than that."""
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_CPUS", raising=False)
    monkeypatch.setattr(B.os, "sched_getaffinity", lambda pid: set(range(208)), raising=False)
    cmd, metrics = _evaluate_command(monkeypatch, parallel=200)
    cpus = cmd[cmd.index("--cpuset-cpus") + 1].split(",")
    assert cpus == [str(i) for i in range(B.MAX_HARNESS_CPUS)] and B.MAX_HARNESS_CPUS == 64
    assert cmd.index("--cpuset-cpus") < cmd.index("img")          # a docker option
    assert cmd[cmd.index("--parallel") + 1] == "64"
    assert metrics["harness_cpus"] == 64 and metrics["harness_parallel"] == 64
    cmd, _ = _evaluate_command(monkeypatch, parallel=8)
    assert cmd[cmd.index("--parallel") + 1] == "8"
    # a small host keeps all of its CPUs
    monkeypatch.setattr(B.os, "sched_getaffinity", lambda pid: {0, 2, 5}, raising=False)
    cmd, metrics = _evaluate_command(monkeypatch, parallel=8)
    assert cmd[cmd.index("--cpuset-cpus") + 1] == "0,2,5" and metrics["harness_parallel"] == 3
    # the knob
    monkeypatch.setattr(B.os, "sched_getaffinity", lambda pid: set(range(208)), raising=False)
    monkeypatch.setenv("GBENCH_BIGCODEBENCH_CPUS", "16")
    assert B.harness_cpus() == list(range(16))
    monkeypatch.setenv("GBENCH_BIGCODEBENCH_CPUS", "zero")
    with pytest.raises(ValueError, match="GBENCH_BIGCODEBENCH_CPUS"):
        B.harness_cpus()


# --------------------------------------------------------------------------- #
# MultiPL-E: chat_completions.py fetched at the pinned commit                   #
# --------------------------------------------------------------------------- #
_FIXTURE = b'''"""A stand-in with the shape of chat_completions.py."""
from tqdm.auto import tqdm


class Completion(dspy.Signature):
    """
    Line one of the stand-in instructions.
    Line two.

    Last line.
    """

    lang: str = dspy.InputField(
        description="Language description"
    )
    prefix: str = dspy.InputField(description="Prefix description")
    completion: str = dspy.OutputField(
        description="Completion description"
    )


def stop_at_stop_token(decoded_string, stop_tokens):
    return decoded_string.split(stop_tokens[0])[0] if stop_tokens else decoded_string


def postprocess_completion(
    lang: str,
    problem_filename: Path,
    stop_tokens: List[str],
    prompt: str,
    completion: str,
) -> str:
    tqdm.write(f"note for {problem_filename}")
    return "PP[" + lang + "|" + prompt + "|" + stop_at_stop_token(completion, stop_tokens) + "]"
'''


def _use_fixture(monkeypatch, data=_FIXTURE):
    monkeypatch.setattr(M, "_UPSTREAM", None)
    monkeypatch.setattr(M, "CHAT_SCRIPT_SHA256", hashlib.sha256(data).hexdigest())
    fetched = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return data

    def fake_urlopen(url, timeout=None):
        fetched.append(url)
        return _Resp()

    monkeypatch.setattr(M.urllib.request, "urlopen", fake_urlopen)
    return fetched


def test_multipl_e_fetches_verifies_and_caches_the_upstream_script(monkeypatch):
    fetched = _use_fixture(monkeypatch)
    msgs = M.chat_messages("jl", "function f()\n")
    assert fetched == [M.CHAT_SCRIPT_URL] and "3025a531af74" in M.CHAT_SCRIPT_URL
    cache = M.chat_script_cache_path()
    assert cache.read_bytes() == _FIXTURE
    assert str(cache).startswith(os.environ["GBENCH_CACHE_DIR"])      # conftest's tmp cache
    system = msgs[0]["content"]
    assert "1. `lang` (str): Language description\n2. `prefix` (str): Prefix description\n" in system
    assert "2. `completion` (str): Completion description\n" in system
    assert system.endswith("your objective is: \n        Line one of the stand-in instructions."
                           "\n        Line two.\n        \n        Last line.")
    # the fetched functions are what runs
    assert M.postprocess_completion(["\nX"], "p", "body\nX tail", lang="jl") == "PP[jl|p|body]"
    assert M.stop_at_stop_token("a;b", [";"]) == "a"
    # a second process reads the cache and does not download again
    monkeypatch.setattr(M, "_UPSTREAM", None)
    monkeypatch.setattr(M.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("the cached copy must be used"))
    assert M.chat_messages("jl", "function f()\n") == msgs


def test_multipl_e_refuses_a_script_with_another_checksum(monkeypatch):
    _use_fixture(monkeypatch)
    monkeypatch.setattr(M, "CHAT_SCRIPT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="expected 0000"):
        M.upstream()
    assert not M.chat_script_cache_path().exists()


def test_multipl_e_hard_errors_when_the_script_cannot_be_fetched(monkeypatch):
    monkeypatch.setattr(M, "_UPSTREAM", None)

    def offline(url, timeout=None):
        raise OSError("network is unreachable")

    monkeypatch.setattr(M.urllib.request, "urlopen", offline)
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(M.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0))
    ok, why = M.check_multipl_e_prerequisites()
    assert ok is False and "could not download" in why and M.CHAT_SCRIPT_SHA256 in why
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        M.run_multipl_e("m", "http://h/v1", 1)


def test_multipl_e_module_ships_no_upstream_text():
    src = open(M.__file__, encoding="utf-8").read()
    for text in ("I will give you the prefix", "The prefix of the program to complete",
                 "The programming language of the program", "Produces the prefix of",
                 "Assume that this is a complete Markdown block",
                 "may not be used as training data"):
        assert text not in src, text


def _github_reachable() -> bool:
    import socket
    try:
        socket.create_connection(("raw.githubusercontent.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _github_reachable(), reason="raw.githubusercontent.com is not reachable")
def test_multipl_e_real_upstream_script_renders_the_same_messages(monkeypatch):
    """Downloads the pinned chat_completions.py. The rendered system message must match the known
    sha256 of DSPy's ChatAdapter system message for that signature."""
    monkeypatch.setattr(M, "_UPSTREAM", None)
    msgs = M.chat_messages("jl", "function add(x::Int64, y::Int64)::Int64\n")
    assert hashlib.sha256(msgs[0]["content"].encode()).hexdigest() == (
        "27bac113ee2007fc77cd09311415d4ccf83b23dc8c6ac10753997be1773dfb7c")
    assert M.postprocess_completion(["\nend"], "function f(x)\n",
                                    "```julia\nfunction f(x)\n  x\nend\nextra\n```") == \
        "function f(x)\n  x"


# --------------------------------------------------------------------------- #
# LiveCodeBench: official sampling and the self-repair scenario                 #
# --------------------------------------------------------------------------- #
_SR_SYSTEM = (
    "You are a helpful programming assistant and an expert Python programmer. You are helping a "
    "user write a program to solve a problem. The user has written some code, but it has some "
    "errors and is not passing the tests. You will help the user by first giving a concise (at "
    "most 2-3 sentences) textual explanation of what is wrong with the code. After you have "
    "pointed out what is wrong with the code, you will then generate a fixed version of the "
    "program. You must put the entired fixed program within code delimiters only for once.")


def test_lcb_self_repair_prompt_is_lcb_runners_openaichat_prompt():
    meta = json.dumps({"output": "6\n", "inputs": "2 3", "expected": "5", "error_code": -2,
                       "error_message": "Wrong answer"})
    msgs = L._self_repair_prompt("Add two numbers.", "print(6)", False, meta)
    assert msgs[0] == {"role": "system", "content": _SR_SYSTEM}
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == (
        "### Question:\nAdd two numbers.\n\n"
        "### Answer:\n```python\nprint(6)\n```\n\n"
        "The above code is incorrect and got a wrong answer.\nInput: 2 3\n"
        "Generated Output: 6\n\nExpected: 5\n"
        "### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
        "(do not directly test on the sample inputs). Enclose your code within delimiters as "
        "follows.\n```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n\n\n"
        "First reason about the code providing a textual explanation of what is wrong with the "
        "code and then generate a fixed of the program enclosed code delimiters.")
    assert L._self_repair_prompt("q", "c", True, meta) == ""       # accepted: kept as is


def test_lcb_self_repair_metadata_shapes():
    wa = {"output": "1", "inputs": "x", "expected": "2", "error_code": -2}
    assert L._self_repair_metadata(wa) == (json.dumps(wa), "wrong_answer")
    setup = {"error_code": -4, "error_message": "Error during testing: invalid syntax"}
    mj, kind = L._self_repair_metadata(setup)
    assert kind == "compilation_error"
    assert L._get_check_prompt("q", False, mj) == (
        "The above code is incorrect and got the following compilation error.\n"
        "Error during testing: invalid syntax")
    for none in (None, {}, {"error_code": -5, "error_message": "TestRunnerError"}):
        mj, kind = L._self_repair_metadata(none)
        assert kind == "no_failure_message" and L._get_check_prompt("q", False, mj) == ""
    rt = {"error": "E()", "error_code": -4, "error_message": "Runtime Error", "inputs": "i",
          "expected": "o"}
    assert L._self_repair_metadata(rt)[1] == "runtime_error"


_REF_META = r'''
import builtins, json, sys
import numpy, pandas
p = json.loads(sys.stdin.read())
ns = {"__builtins__": builtins.__dict__, "__name__": "lcb_testing_util"}
exec(compile(p["source"], "lcb_testing_util.py", "exec"), ns)
res, meta = ns["run_test"]({"input_output": p["io"]}, test=p["code"], debug=False, timeout=6)
print("REF", json.dumps(meta, default=str))
'''


@pytest.mark.parametrize("code,tests,fn", [
    ("print(6)", [("2 3", "5")], None),
    ("raise ValueError('boom')", [("2 3", "5")], None),
    ("def f(:", [("2 3", "5")], None),
    ("print('x' * 1000)", [("1", "1")], None),
    ("x = input()\nprint(x)\nprint(x)", [("a", "a"), ("b", "b")], None),
    ("class Solution:\n    def add(self, a, b):\n        return a - b", [("2\n3", "5")], "add"),
    ("class Solution:\n    def sub(self, a, b):\n        return a - b", [("2\n3", "5")], "add"),
    ("class Solution:\n    def add(self, a, b):\n        return [][1]", [("2\n3", "5")], "add"),
    ("class Solution:\n    def add(self, a, b):\n        return (a, b)", [("2\n3", "5")], "add"),
])
def test_lcb_failure_metadata_is_run_tests(code, tests, fn):
    """The grader reports the metadata upstream's run_test (vendored verbatim) returns."""
    io = json.dumps({"inputs": [t[0] for t in tests], "outputs": [t[1] for t in tests],
                     "fn_name": fn})
    r = subprocess.run([sys.executable, "-c", _REF_META], capture_output=True, text=True,
                       timeout=120, input=json.dumps({"source": L._LCB_TESTING_UTIL, "io": io,
                                                      "code": code}))
    ref = json.loads(next(l[4:] for l in r.stdout.splitlines()[::-1] if l.startswith("REF ")))
    gold = {"task": "generation", "fn_name": fn,
            "tests": [{"input": i, "output": o} for i, o in tests]}
    verdict, meta = L._judge_lcb_sample("```python\n" + code + "\n```", gold)
    assert verdict is False and meta == ref


def test_lcb_self_check_requires_the_failure_metadata(monkeypatch):
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    monkeypatch.setattr(L, "_judge_lcb_sample", lambda resp, gold: (False, None))
    reason = L._grader_self_check()
    assert reason and "failure metadata" in reason
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", "0")
    assert L._grader_self_check() is None


_STDIN_Q = "Read two integers and print their sum."
_CALL_Q = "Return a + b."
_GOOD_STDIN = "```python\nprint(sum(map(int, input().split())))\n```"
_GOOD_CALL = "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```"


def _lcb_offline(monkeypatch, repair_reply):
    """run_lcb on two problems: the stdin one is always answered wrongly first, the call-based
    one correctly. `repair_reply(messages)` answers the self-repair prompts."""
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    monkeypatch.delenv("GBENCH_LCB_TEMPERATURE", raising=False)
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    samples = [
        ([{"role": "system", "content": L._SYSTEM_MESSAGE_GENERIC},
          {"role": "user", "content": L._generation_prompt(_STDIN_Q, "")}],
         {"task": "generation", "fn_name": None, "question_content": _STDIN_Q,
          "tests": [{"input": "2 3", "output": "5"}, {"input": "4 4", "output": "8"}]},
         {"category": "code_gen_easy", "contest_date": "2025-02-01"}),
        ([{"role": "system", "content": L._SYSTEM_MESSAGE_GENERIC},
          {"role": "user", "content": L._generation_prompt(_CALL_Q, "class Solution: ...")}],
         {"task": "generation", "fn_name": "add", "question_content": _CALL_Q,
          "tests": [{"input": "2\n3", "output": "5"}]},
         {"category": "code_gen_medium", "contest_date": "2025-02-01"}),
    ]
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: samples)
    sent = []

    async def fake_send(**k):
        sent.append(k)
        msgs = k["messages"]
        if msgs[0]["content"] == L._SR_SYSTEM_MESSAGE_GENERIC:
            return base.Reply(repair_reply(msgs), None, "stop", completion_tokens=5)
        text = _GOOD_CALL if _CALL_Q in msgs[1]["content"] else "```python\nprint(6)\n```"
        return base.Reply(text, None, "stop", completion_tokens=5)

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    res = L.run_lcb("m", "http://x/v1", 4, max_output_tokens=2048)
    return res, sent


def test_lcb_self_repair_repairs_rejected_programs_and_regrades_all(monkeypatch):
    res, sent = _lcb_offline(monkeypatch, lambda msgs: "The sum was wrong.\n" + _GOOD_STDIN)
    repairs = [k for k in sent if k["messages"][0]["content"] == L._SR_SYSTEM_MESSAGE_GENERIC]
    firsts = [k for k in sent if k not in repairs]
    assert len(firsts) == 20                           # 2 problems x lcb_runner's n=10
    assert len(repairs) == 10                          # only the 10 rejected stdin programs
    for k in firsts + repairs:     # lcb_runner's top_p, gbench's no-think 0.0, on every request
        assert k["temperature"] == 0.0 and k["extra_payload"]["top_p"] == 0.95
    user = repairs[0]["messages"][1]["content"]
    assert user.startswith("### Question:\n" + _STDIN_Q + "\n\n### Answer:\n```python\nprint(6)\n```")
    assert ("The above code is incorrect and got a wrong answer.\nInput: 2 3\n"
            "Generated Output: 6\n\nExpected: 5") in user
    assert res["accuracy"] == 50.0                     # code generation: the call-based half
    sr = res["scenarios"]["self_repair"]
    assert sr == {"correct": 20, "total": 20, "accuracy": 100.0}
    rep = res["self_repair"]
    assert rep["repair_requests"] == 10 and rep["kept_accepted_programs"] == 10
    assert rep["failure_messages"] == {"wrong_answer": 10} and rep["failed_requests"] == 0
    assert rep["at_k"]["samples"] == 2 and rep["at_k"]["pass_at_k"] == 100.0
    assert res["lcb_scenarios"][-1] == "self_repair"
    assert res["official_sampling"]["temperature_matches_official"] is False
    stdin_tr = next(t for t in res["sample_traces"] if t["category"] == "code_gen_easy")
    assert stdin_tr["lcb_grading"]["error_code"] == -2
    assert stdin_tr["lcb_self_repair"]["is_correct"] is True
    assert stdin_tr["lcb_self_repair"]["kept"] is False
    call_tr = next(t for t in res["sample_traces"] if t["category"] == "code_gen_medium")
    assert call_tr["lcb_self_repair"] == {"kept": True, "is_correct": True}
    # the only reasons: gbench's temperature, and this two-problem fixture has no code
    # execution or test output sample
    assert res["leaderboard_comparable_reason"] == (
        "temperature=0.0 (default); lcb_runner samples at 0.2; "
        "scenario(s) not run: code_execution, test_output_prediction (no sample loaded) "
        "(LiveCodeBench publishes all four)")


def test_lcb_self_repair_failed_requests_are_failed_programs_and_named(monkeypatch):
    res, _ = _lcb_offline(monkeypatch, lambda msgs: None)
    assert res["scenarios"]["self_repair"]["correct"] == 10       # the kept call-based half
    assert res["self_repair"]["failed_requests"] == 10
    assert res["leaderboard_comparable"] is False
    # a repair request with no reply is a harness error, left out of the self-repair pass@1
    assert "10 self-repair request(s) got no reply from the served endpoint" in \
        res["leaderboard_comparable_reason"]
    assert res["self_repair"]["harness_errors"] == 10
    assert res["scenarios"]["self_repair"]["total"] == 10


def test_lcb_generation_requests_without_a_response_are_named(monkeypatch):
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: [])
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", "0")
    monkeypatch.delenv("GBENCH_LCB_TEMPERATURE", raising=False)
    traces = [{"sample_idx": i, "source_sample_idx": 0, "category": "code_execution",
               "is_correct": i == 0, "response_text": None if i == 1 else "x"}
              for i in range(10)]
    monkeypatch.setattr(L, "run_eval_suite", lambda **k: {
        "accuracy": 10.0, "temperature": 0.2, "sample_traces": traces,
        "max_tokens_sent": [L._LCB_MAX_TOKENS], "attempts": {"attempts_per_sample": 10}})
    res = L.run_lcb("m", "http://x/v1", 1)
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "scenario(s) not run: code_generation, test_output_prediction (no sample loaded); "
        "self_repair (left out with GBENCH_LCB_INCLUDE_SELF_REPAIR=0) (LiveCodeBench publishes "
        "all four); 1 generation request(s) got no reply from the served endpoint (harness errors, left out "
        "of the score; lcb_runner retries a failed request up to 10 times)")
    traces[1]["response_text"] = ""                   # an empty reply is the model's, not lost
    assert "got no response" not in L.run_lcb("m", "http://x/v1", 1)["leaderboard_comparable_reason"]


def test_lcb_self_repair_can_be_left_out(monkeypatch):
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: [])
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", "0")
    monkeypatch.setattr(L, "_run_self_repair", lambda *a, **k: pytest.fail("not requested"))
    trace = {"sample_idx": 0, "category": "code_gen_easy", "is_correct": False}
    seen = {}
    monkeypatch.setattr(L, "run_eval_suite", lambda **k: seen.update(k) or {
        "accuracy": 0.0, "temperature": 0.2, "sample_traces": [trace]})
    res = L.run_lcb("m", "http://x/v1", 1)
    assert "self_repair" not in res["lcb_scenarios"] and "self_repair" not in res["scenarios"]
    assert seen["official_sampling"] is L.OFFICIAL_SAMPLING
    assert seen["extra_payload"]["top_p"] == 0.95
