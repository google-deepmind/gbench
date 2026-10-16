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

"""Official sampling fields for the maths and code suites, beyond the temperature.

* Decoding penalties (GBENCH_*_PENALTY), or any other sampling field upstream does not send, make
  lcb, multipl_e, bigcodebench and ojbench not comparable; lcb sends lcb_runner's zero penalties.
* The official max_tokens is what every no-think request carries: lcb 2000 (self-repair
  included), multipl_e 2048, bigcodebench 1280; scicode sends none, as SciCode does.
* MultiPL-E scores a reply ChatAdapter.parse rejects as upstream does (completion ""), because
  DSPy 3.0.0b1's async path has no JSONAdapter fallback; an empty reply stops upstream's script,
  so it makes the run not comparable.
* The MultiPL-E evaluator is built from MultiPL-E's own evaluation/ directory, downloaded and
  checksum-checked at build time; gbench ships no copy of the recipe.
* ojbench samples with the served model's own recommended settings; hmmt and aime name a served
  model whose recommended settings are not Gemma 4's.
* bigcodebench removes its per-split work directory.

Offline: the endpoint, datasets and Docker are stubbed. The LCB graders run for real (unsandboxed
under conftest's GBENCH_SANDBOX=none). One test inspects the locally built MultiPL-E source image
and runs only when that image exists.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bigcodebench as B
from gbench.runners.eval_suites import lcb as L
from gbench.runners.eval_suites import matharena_common as MA
from gbench.runners.eval_suites import multipl_e as M
from gbench.runners.eval_suites import ojbench as OJ
from gbench.runners.eval_suites import scicode as SC

REPO = Path(__file__).resolve().parents[1]
PENALTY = {"repetition_penalty": 1.3}


def _gemma_served(monkeypatch):
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))


# --------------------------------------------------------------------------- #
# Decoding penalties and other sampling fields break comparability             #
# --------------------------------------------------------------------------- #
def _lcb_run(monkeypatch, suite_result, **kwargs):
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    monkeypatch.delenv("GBENCH_LCB_TEMPERATURE", raising=False)
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: [])
    monkeypatch.setattr(L, "_run_self_repair", lambda traces, **kw: {
        "correct": 1, "total": 1, "accuracy": 100.0, "failed_requests": 0})
    trace = {"sample_idx": 0, "source_sample_idx": 0, "category": "code_gen_easy",
             "is_correct": True, "response_text": "x"}
    traces = [dict(trace, sample_idx=i) for i in range(10)]
    # every scenario the run loads ran
    traces.append(dict(trace, sample_idx=10, source_sample_idx=1, category="code_execution"))
    if L._include_test_gen():
        traces.append(dict(trace, sample_idx=11, source_sample_idx=2, category="test_gen_easy"))
    monkeypatch.setattr(L, "run_eval_suite", lambda **k: {
        "accuracy": 100.0, "temperature": 0.2, "sample_traces": traces,
        "max_tokens_sent": [L._LCB_MAX_TOKENS], "attempts": {"attempts_per_sample": 10},
        **suite_result})
    return L.run_lcb("m", "http://x/v1", 1, **kwargs)


def test_lcb_decoding_penalty_is_not_comparable(monkeypatch):
    assert _lcb_run(monkeypatch, {})["leaderboard_comparable"] is True
    r = _lcb_run(monkeypatch, {"decoding_penalties": PENALTY})
    assert r["leaderboard_comparable"] is False
    assert "repetition_penalty=1.3 (lcb_runner: 'not sent')" in r["leaderboard_comparable_reason"]
    # lcb_runner itself sends frequency and presence penalties of 0
    r = _lcb_run(monkeypatch, {"decoding_penalties": {"frequency_penalty": 0.0,
                                                      "presence_penalty": 0.0}})
    assert r["leaderboard_comparable"] is True
    r = _lcb_run(monkeypatch, {"decoding_penalties": {"presence_penalty": 0.5}})
    assert "presence_penalty=0.5 (lcb_runner: 0)" in r["leaderboard_comparable_reason"]
    r = _lcb_run(monkeypatch, {}, extra_payload={"seed": 7})
    assert "seed=7" in r["leaderboard_comparable_reason"]


def test_lcb_official_sampling_is_what_the_runner_sends():
    assert L.OFFICIAL_SAMPLING.temperature == 0.2
    assert L.OFFICIAL_SAMPLING.params == {"top_p": 0.95, "frequency_penalty": 0,
                                          "presence_penalty": 0}
    assert L._LCB_MAX_TOKENS == 2000


def test_lcb_names_a_scenario_that_was_left_out_and_a_short_budget(monkeypatch):
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_TEST_GEN", "0")
    r = _lcb_run(monkeypatch, {"max_tokens_sent": [1500, 2000]})
    reason = r["leaderboard_comparable_reason"]
    assert "scenario(s) not run: test_output_prediction (left out with " in reason
    assert "max_tokens=[1500, 2000]; lcb_runner sends 2000" in reason


def _mpe_run(monkeypatch, suite_result, metrics_update=None, **kwargs):
    monkeypatch.delenv("GBENCH_MULTIPL_E_SAMPLES", raising=False)
    monkeypatch.delenv("GBENCH_MULTIPL_E_LANGS", raising=False)
    monkeypatch.setattr(M, "check_multipl_e_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_load_multipl_e_samples", lambda **k: [])

    def fake_make_scorer(image, workers, metrics):
        metrics.update(metrics_update or {})
        return None

    monkeypatch.setattr(M, "_make_scorer", fake_make_scorer)
    monkeypatch.setattr(M, "run_eval_suite", lambda **k: {
        "accuracy": 50.0, "temperature": 0.2, "status": "success",
        "max_tokens_sent": [M.UPSTREAM_MAX_TOKENS], **suite_result})
    return M.run_multipl_e("m", "http://h/v1", 1, **kwargs)


def test_multipl_e_decoding_penalty_is_not_comparable(monkeypatch):
    assert _mpe_run(monkeypatch, {})["leaderboard_comparable"] is True
    r = _mpe_run(monkeypatch, {"decoding_penalties": PENALTY})
    assert r["leaderboard_comparable"] is False
    assert "sampling repetition_penalty=1.3 (the upstream chat script sends no such field)" \
        in r["leaderboard_comparable_reason"]
    r = _mpe_run(monkeypatch, {}, extra_payload={"top_p": 0.95})
    assert "top_p=0.95" in r["leaderboard_comparable_reason"]


def _bcb_run(monkeypatch, suite_result, thinking=False, **kwargs):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])
    monkeypatch.setattr(B, "_make_scorer", lambda subset, parallel, metrics, image: None)
    # Both paths are greedy. A thinking run sends make_request's request for a served model with
    # the run's budget (the thinking floor here) in place of the 1280, as the token-limit policy
    # says; without --thinking every request carries the 1280.
    mode = ({"temperature": 0.0, "max_tokens_sent": [base.THINKING_MIN_OUTPUT_TOKENS]}
            if thinking else {"temperature": 0.0, "max_tokens_sent": [B.UPSTREAM_MAX_NEW_TOKENS]})
    monkeypatch.setattr(B, "run_eval_suite", lambda **k: {
        "accuracy": 50.0, "status": "success", **mode, **suite_result})
    return B.run_bigcodebench("m", "http://h/v1", 1, enable_thinking=thinking, **kwargs)


def test_bigcodebench_decoding_penalty_is_not_comparable(monkeypatch):
    for thinking in (False, True):
        assert _bcb_run(monkeypatch, {}, thinking)["leaderboard_comparable"] is True
        r = _bcb_run(monkeypatch, {"decoding_penalties": PENALTY}, thinking)
        assert r["leaderboard_comparable"] is False
        assert "repetition_penalty=1.3" in r["leaderboard_comparable_reason"]
        assert "(repetition_penalty='not sent')" in r["leaderboard_comparable_reason"]
    r = _bcb_run(monkeypatch, {"max_tokens_sent": [4096]})
    assert "max_tokens=[4096] (run_codegen's max_new_tokens is 1280" \
        in r["leaderboard_comparable_reason"]


def _oj_run(monkeypatch, suite_result, recommended=None, **kwargs):
    monkeypatch.delenv("GBENCH_OJBENCH_SAMPLES", raising=False)
    monkeypatch.delenv("GBENCH_OJBENCH_TEMPERATURE", raising=False)
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "_load_ojbench_samples", lambda limit=None: [])
    rec = dict(OJ.GEMMA4_RECOMMENDED if recommended is None else recommended)
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                        lambda model_name=None: (dict(rec), "org/model/generation_config.json"))
    seen = {}

    def fake_suite(**k):
        seen.update(k)
        traces = []
        for lang in ("cpp", "python"):
            for a in range(8):
                traces.append({"extra_payload": {"row": {"id": 1, "language": lang,
                                                         "difficulty": "easy"}},
                               "response_text": "code", "is_correct": a < 4})
        return {"accuracy": 50.0, "temperature": rec.get("temperature"), "status": "success",
                "scoring_excluded": 0, "max_tokens_sent": [65536], "sample_traces": traces,
                **suite_result}

    monkeypatch.setattr(OJ, "run_eval_suite", fake_suite)
    return OJ.run_ojbench("m", "http://h/v1", 1, enable_thinking=True, **kwargs), seen


def test_ojbench_decoding_penalty_is_not_comparable(monkeypatch):
    r, _ = _oj_run(monkeypatch, {})
    assert r["leaderboard_comparable"] is True
    r, _ = _oj_run(monkeypatch, {"decoding_penalties": PENALTY})
    assert r["leaderboard_comparable"] is False
    assert "sampling repetition_penalty=1.3 (the model's recommended repetition_penalty=not set)" \
        in r["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# ojbench: the served model's own recommended settings                        #
# --------------------------------------------------------------------------- #
def test_ojbench_reads_the_recommended_settings_from_a_local_generation_config(monkeypatch,
                                                                              tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_length": 9}))
    monkeypatch.setattr(OJ, "get_run_knob", lambda name, default=None:
                        str(tmp_path) if name == "tokenizer" else default)
    rec, source = OJ.served_model_recommended_sampling("ignored")
    assert rec == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert source == f"{tmp_path}/generation_config.json"
    official = OJ.official_sampling(rec, source)
    assert official.temperature == 0.6 and official.params == {"top_p": 0.95, "top_k": 20}
    assert source in official.reference


def test_ojbench_unreadable_generation_config_is_a_named_error(monkeypatch, tmp_path):
    monkeypatch.setattr(OJ, "get_run_knob", lambda name, default=None:
                        str(tmp_path) if name == "tokenizer" else default)
    with pytest.raises(RuntimeError, match="generation_config.json"):
        OJ.served_model_recommended_sampling("m")
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "run_eval_suite", lambda **k: pytest.fail("must not generate"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*recommended temperature"):
        OJ.run_ojbench("m", "http://h/v1", 1)


def test_ojbench_samples_with_another_models_recommended_settings(monkeypatch):
    qwen_like = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    r, seen = _oj_run(monkeypatch, {}, recommended=qwen_like)
    official = seen["official_sampling"]
    assert official.temperature == 0.7 and official.params == {"top_p": 0.8, "top_k": 20}
    assert r["recommended_sampling"] == {"source": "org/model/generation_config.json", **qwen_like}
    assert r["leaderboard_comparable"] is True
    # a model that recommends no temperature gets none sent
    none_set = OJ.official_sampling({"top_p": 0.9}, "x")
    assert none_set.send_temperature is False and none_set.params == {"top_p": 0.9}


def test_matharena_names_a_served_model_that_is_not_gemma(monkeypatch):
    # The mismatch check this test covered is gone: hmmt and aime now apply the served model's own
    # settings, so the record names the model and the fields actually sent.
    from gbench.runners.eval_suites import hmmt as H
    _gemma_served(monkeypatch)
    gemma = H.served_model_sampling("hmmt", "m", "d", False)
    assert gemma.record == {"source": "stub", **OJ.GEMMA4_RECOMMENDED}
    assert gemma.model_config is None
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                        lambda model_name=None: ({"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                                                 "q/generation_config.json"))
    other = H.served_model_sampling("hmmt", "m", "d", False)
    assert other.sampling.params == {"top_p": 0.95, "top_k": 20}
    assert "q/generation_config.json" in other.sampling.reference

    def unreadable(model_name=None):
        raise RuntimeError("could not read generation_config.json of 'm'")

    monkeypatch.setattr(OJ, "served_model_recommended_sampling", unreadable)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        H.served_model_sampling("hmmt", "m", "d", False)


def test_hmmt_and_aime_send_the_served_models_recommended_fields(monkeypatch):
    # hmmt and aime now send the served model's own recommended settings (MathArena's rule), so a
    # model that recommends no top_k gets none sent and the run stays comparable.
    from gbench.runners.eval_suites import aime as A
    from gbench.runners.eval_suites import hmmt as H
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MA, "self_check", lambda config: None)
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                        lambda model_name=None: ({"temperature": 1.0, "top_p": 0.95},
                                                 "x/generation_config.json"))
    result = {"accuracy": 50.0, "temperature": 1.0, "attempts": {"attempts_per_sample": 4},
              "sample_traces": []}
    for mod, loader in ((H, "_load_hmmt_samples"), (A, "_load_aime_samples")):
        seen = {}
        monkeypatch.setattr(mod, loader, lambda *a, **k: [])
        monkeypatch.setattr(mod, "run_eval_suite", lambda **kw: seen.update(kw) or dict(result))
        run = H.run_hmmt if mod is H else A.run_aime
        r = run("m", "http://x/v1", 1)
        assert seen["official_sampling"].params == {"top_p": 0.95}
        assert "x/generation_config.json" in seen["official_sampling"].reference
        assert r["recommended_sampling"] == {"source": "x/generation_config.json",
                                             "temperature": 1.0, "top_p": 0.95}
        assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


# --------------------------------------------------------------------------- #
# The official max_tokens on every request                                    #
# --------------------------------------------------------------------------- #
_STDIN_Q = "Read two integers and print their sum."
_GOOD_STDIN = "```python\nprint(sum(map(int, input().split())))\n```"


def test_lcb_every_request_carries_lcb_runners_sampling_and_budget(monkeypatch):
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    monkeypatch.delenv("GBENCH_LCB_TEMPERATURE", raising=False)
    monkeypatch.setenv("GBENCH_REPETITION_PENALTY", "1.2")
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    samples = [([{"role": "system", "content": L._SYSTEM_MESSAGE_GENERIC},
                 {"role": "user", "content": L._generation_prompt(_STDIN_Q, "")}],
                {"task": "generation", "fn_name": None, "question_content": _STDIN_Q,
                 "tests": [{"input": "2 3", "output": "5"}]},
                {"category": "code_gen_easy", "contest_date": "2025-02-01"})]
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: samples)
    sent = []

    async def fake_send(**k):
        sent.append(k)
        if k["messages"][0]["content"] == L._SR_SYSTEM_MESSAGE_GENERIC:
            return base.Reply(_GOOD_STDIN, None, "stop", completion_tokens=5)
        return base.Reply("```python\nprint(6)\n```", None, "stop", completion_tokens=5)

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=16384)     # the run's --max-output-tokens
    res = L.run_lcb("m", "http://x/v1", 2, max_output_tokens=16384)
    repairs = [k for k in sent if k["messages"][0]["content"] == L._SR_SYSTEM_MESSAGE_GENERIC]
    assert len(sent) == 20 and len(repairs) == 10
    for k in sent:
        assert k["max_output_tokens"] == 2000
        payload = k["extra_payload"]
        assert payload["top_p"] == 0.95 and payload["frequency_penalty"] == 0
        assert payload["presence_penalty"] == 0 and payload["repetition_penalty"] == 1.2
    assert res["max_tokens_sent"] == [2000] and res["self_repair"]["max_output_tokens"] == 2000
    assert "--max-output-tokens 16384 not applied" in res["max_output_tokens_note"]
    assert res["leaderboard_comparable"] is False
    assert "repetition_penalty=1.2 (lcb_runner: 'not sent')" in res["leaderboard_comparable_reason"]


def test_multipl_e_and_bigcodebench_samples_carry_the_upstream_budget(monkeypatch):
    # The upstream cap is the suite's OfficialSampling.max_tokens, resolved per mode by base's
    # token-limit policy (the stub-endpoint token-limit tests check the wire); samples declare
    # none.
    import datasets
    monkeypatch.setattr(M, "chat_messages", lambda lang, prefix: [{"role": "user",
                                                                   "content": prefix}])
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [
        {"name": "HumanEval_0", "prompt": "p", "tests": "t", "stop_tokens": []}])
    mpe = M._load_multipl_e_samples(langs=["jl"])
    assert "max_tokens" not in mpe[0][2]
    assert M.OFFICIAL_SAMPLING.max_tokens == M.UPSTREAM_MAX_TOKENS == 2048

    def fake_bcb_load(name, split=None, revision=None):
        return [{"task_id": "BigCodeBench/0", "instruct_prompt": "Do X.",
                 "complete_prompt": "def task_func():\n", "entry_point": "task_func"}]

    monkeypatch.setattr(datasets, "load_dataset", fake_bcb_load)
    bcb = B._with_prefill(B._load_bigcodebench_samples(subset="hard"))
    assert all("max_tokens" not in m for _, _, m in bcb)
    assert B.SAMPLING[True].max_tokens == B.SAMPLING[False].max_tokens == 1280
    assert all(m["continue_final_message"] is True for _, _, m in bcb)


def test_scicode_sends_no_max_tokens():
    assert SC.OFFICIAL_SAMPLING.send_max_tokens is False
    assert SC.OFFICIAL_SAMPLING.record(0.0, "official")["send_max_tokens"] is False
    calls = []

    class _Completions:
        def create(self, **kw):
            calls.append(kw)
            msg = types.SimpleNamespace(content="```python\nx = 1\n```", reasoning_content=None)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                usage=types.SimpleNamespace(completion_tokens=3))

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    out = SC._complete(client, "m", "prompt", 0.0, None, False)
    assert out["text"].startswith("```python") and "max_tokens" not in calls[0]
    SC._complete(client, "m", "prompt", 0.0, 64, False)
    assert calls[1]["max_tokens"] == 64


# --------------------------------------------------------------------------- #
# MultiPL-E: scored as DSPy 3.0.0b1's async path scores it                     #
# --------------------------------------------------------------------------- #
_FIXTURE = b'''"""A stand-in with the shape of chat_completions.py."""
from tqdm.auto import tqdm


class Completion(dspy.Signature):
    """
    Stand-in instructions.
    """

    lang: str = dspy.InputField(description="Language")
    prefix: str = dspy.InputField(description="Prefix")
    completion: str = dspy.OutputField(description="Completion")


def stop_at_stop_token(decoded_string, stop_tokens):
    return decoded_string


def postprocess_completion(lang, problem_filename, stop_tokens, prompt, completion):
    if completion.startswith("```"):
        completion = "\\n".join(completion.split("\\n")[1:-1])
    return completion
'''


def _use_fixture(monkeypatch):
    monkeypatch.setattr(M, "_UPSTREAM", None)
    monkeypatch.setattr(M, "CHAT_SCRIPT_SHA256", hashlib.sha256(_FIXTURE).hexdigest())

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return _FIXTURE

    monkeypatch.setattr(M.urllib.request, "urlopen", lambda url, timeout=None: _Resp())


def test_multipl_e_scores_parse_failures_like_upstream_and_empty_replies_apart(monkeypatch):
    _use_fixture(monkeypatch)
    written = []

    def fake_run(cmd, **k):
        out = next(a.split(":", 1)[0] for a in cmd if a.endswith(":/out:rw"))
        for name in sorted(os.listdir(out)):
            if name.endswith(".json") and not name.endswith(".results.json"):
                data = json.load(open(os.path.join(out, name)))
                written.append(data)
                ok = data["completions"] == ["x = 1"]
                json.dump({"name": data["name"], "results": [
                    {"status": "OK" if ok else "Exception", "exit_code": 0 if ok else 1}]},
                    open(os.path.join(out, name.replace(".json", ".results.json")), "w"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    gold = json.dumps({"name": "HumanEval_0", "language": "py", "prompt": "",
                       "tests": "", "stop_tokens": []})
    good = "[[ ## reasoning ## ]]\nr\n\n[[ ## completion ## ]]\n```\nx = 1\n```\n\n[[ ## completed ## ]]"
    traces = [{"gold_answer": gold, "response_text": good},
              {"gold_answer": gold, "response_text": "x = 1"},        # no ChatAdapter fields
              {"gold_answer": gold, "response_text": ""},             # empty reply
              {"gold_answer": gold, "response_text": None}]           # no response
    metrics = {}
    asyncio.run(M._make_scorer("img", 1, metrics)(traces))
    assert [w["completions"] for w in written] == [["x = 1"], [""], [""], [""]]
    assert [t["is_correct"] for t in traces] == [True, False, False, False]
    assert metrics["chat_adapter_parse_failures"] == 1
    assert metrics["chat_adapter_empty_responses"] == 1
    assert metrics["multipl_e_no_response"] == 1
    assert 'completion "" as upstream records it' in traces[1]["scoring_note"]
    r = _mpe_run(monkeypatch, {}, {"chat_adapter_parse_failures": 5})
    assert r["leaderboard_comparable"] is True
    r = _mpe_run(monkeypatch, {}, {"chat_adapter_empty_responses": 2, "multipl_e_no_response": 1})
    reason = r["leaderboard_comparable_reason"]
    assert "2 empty response(s) scored as empty programs; upstream's script stops" in reason
    assert "1 request(s) got no response" in reason
    assert "JSONAdapter" not in reason


def test_multipl_e_describes_no_jsonadapter_retry():
    src = (REPO / "gbench" / "runners" / "eval_suites" / "multipl_e.py").read_text()
    doc = (REPO / "docs" / "evals" / "multipl_e.md").read_text()
    for text in (src, doc):
        assert "1df67689" in text and "c3e79ce8" in text
        assert "not replicate" not in text and "retries them once" not in text


# --------------------------------------------------------------------------- #
# MultiPL-E evaluator image: upstream's evaluation/ at build time             #
# --------------------------------------------------------------------------- #
def _dockerfile():
    return (REPO / "docker" / "multipl_e.Dockerfile").read_text()


def test_multipl_e_dockerfile_fetches_and_checks_upstream_instead_of_carrying_it():
    df = _dockerfile()
    assert df.isascii()
    for upstream_step in ("dmd-compiler", "julialang-s3", "mono-devel", "swift-5.7-RELEASE",
                          "linux-install.sh", "download.dartlang.org", "COPY src /code"):
        assert upstream_step not in df, upstream_step
    assert "ref=3025a531af7450e7df8b96fe0440e9804480bbad" in df
    assert "codeload.github.com/nuprl/MultiPL-E/tar.gz/${ref}" in df
    assert "sha256sum --strict -c /multipl_e.sha256" in df and "diff /tmp/want /tmp/have" in df
    assert "COPY multipl_e.sha256 /multipl_e.sha256" in df
    assert "repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar" in df
    assert "2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a" in df
    assert '"3b6b9332dd449316e0e94da6642b9d83f811333c15c328e06687183262c1be2d  Dockerfile"' in df
    assert 'ENTRYPOINT ["tar", "-c", "."]' in df


def test_multipl_e_manifest_pins_every_file_of_upstreams_evaluation_directory():
    lines = (REPO / "docker" / "multipl_e.sha256").read_text().splitlines()
    entries = dict(reversed(line.split("  ", 1)) for line in lines)
    assert len(entries) == len(lines) == 72
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in entries.values())
    # evaluation/Dockerfile at 3025a531 (git blob 0808bfb0271f3270c9c665804d1188b0f6079867)
    assert entries["Dockerfile"] == \
        "149513ea171ed19be5ce2e9d0f038f91884f6f996c332afc835518537687ff2d"
    assert "src/main.py" in entries and "src/containerized_eval.py" in entries
    assert not any(p.startswith("/") or ".." in p for p in entries)


def test_multipl_e_prerequisite_names_the_two_step_build(monkeypatch):
    monkeypatch.setattr(M.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(M.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=1 if cmd[:3] == ["docker", "image", "inspect"] else 0))
    ok, why = M.check_multipl_e_prerequisites()
    assert not ok
    assert "docker build -t gbench-multipl-e-src:3025a531 -f docker/multipl_e.Dockerfile docker" in why
    assert "docker run --rm gbench-multipl-e-src:3025a531 | docker build -t gbench-multipl-e:3025a531 -" in why


def _have_source_image():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", "gbench-multipl-e-src:3025a531"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _have_source_image(), reason="gbench-multipl-e-src:3025a531 is not built here")
def test_multipl_e_source_image_prints_the_checked_context():
    tar = subprocess.run(["docker", "run", "--rm", "gbench-multipl-e-src:3025a531"], capture_output=True,
                         timeout=300, check=True).stdout
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        names = {m.name[2:] if m.name.startswith("./") else m.name
                 for m in tf.getmembers() if m.isfile()}
        dockerfile = tf.extractfile("./Dockerfile").read()
        main = tf.extractfile("./src/main.py").read()
    manifest = dict(reversed(l.split("  ", 1))
                    for l in (REPO / "docker" / "multipl_e.sha256").read_text().splitlines())
    assert names == set(manifest)
    assert hashlib.sha256(main).hexdigest() == manifest["src/main.py"]
    assert hashlib.sha256(dockerfile).hexdigest() == \
        "3b6b9332dd449316e0e94da6642b9d83f811333c15c328e06687183262c1be2d"


# --------------------------------------------------------------------------- #
# bigcodebench: the per-split work directory is removed                       #
# --------------------------------------------------------------------------- #
def test_bigcodebench_removes_its_work_directory_and_keeps_the_hf_cache_out_of_it(monkeypatch):
    made, calls = [], []
    real = B.tempfile.mkdtemp

    def mkdtemp(*a, **k):
        made.append(real(*a, **k))
        return made[-1]

    def fake_run(cmd, **k):
        calls.append(cmd)
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        if "bigcodebench.sanitize" not in cmd:
            json.dump({"eval": {"BigCodeBench/0": [{"status": "pass", "details": {}}]}},
                      open(os.path.join(mount, "samples_eval_results.json"), "w"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(B.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(B.subprocess, "run", fake_run)
    traces = [{"extra_payload": {"task_id": "BigCodeBench/0", "split": s}, "response_text": "x"}
              for s in ("instruct", "complete")]
    asyncio.run(B._make_scorer("full", 1, {}, "img")(traces))
    assert len(made) == 2 and not any(os.path.exists(d) for d in made)
    assert all(t["is_correct"] for t in traces)
    for cmd in calls:
        assert "HF_HOME=/tmp/.hf" in cmd and not any(a.startswith("HF_HOME=/app") for a in cmd)


# --------------------------------------------------------------------------- #
# docs                                                                         #
# --------------------------------------------------------------------------- #
def test_lcb_doc_cites_tests_by_function_name():
    doc = (REPO / "docs" / "evals" / "lcb.md").read_text()
    assert not re.search(r"test_w\d|test_ws\d", doc)
    cited = set(re.findall(r"`(test_lcb_\w+)`", doc))
    assert len(cited) == 6
    defined = set()
    for path in (REPO / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_lcb_\w+)\(", path.read_text(), re.M))
    assert cited <= defined, cited - defined
