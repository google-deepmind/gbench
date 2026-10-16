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

"""Upstream fidelity and harness-error accounting for cyberseceval, custom_jsonl, mrcr,
wildbench, longbench_v2_code, gdpval, charxiv and mmmu_pro.

* cyberseceval: upstream's instruct-v2.json at a pinned PurpleLlama commit (read from the ICD
  image), each prompt sent verbatim, sampled as upstream's OpenAI adapter samples (temperature
  0.6, top_p 0.9, no max_tokens), and a comparability flag that names a temperature other than
  0.6;
* custom_jsonl: a comma-grouped number (299,792,458) is one number;
* mrcr: no temperature and no max_tokens, as the dataset card's reference loop sends;
* wildbench: WB-Reward and WB-Elo pool judge replies that are not readable JSON are counted,
  named and make the run completed_with_errors;
* longbench_v2_code: pred.py's max_len follows the model's context window (120,000 only for a
  window up to 131,072), and a served window smaller than the model's own is named;
* gdpval: the expert reference is shown to the judge unlabeled, and every clip is recorded,
  counted and named;
* charxiv: the charts are images.zip's files, as the official harness sends them;
* mmmu_pro: a shard that holds no request of a setting says so instead of naming harness errors.

No network, no Gemini, no model server, no docker: requests, judges, downloads and the ICD
image are faked in process.
"""

import asyncio
import hashlib
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, charxiv, custom_jsonl, cyberseceval, gdpval,
                                        longbench_v2_code, mmmu_pro, mrcr, wildbench)

REPO = Path(__file__).resolve().parent.parent


def _doc(name):
    return " ".join((REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8").split())


def _serve(reply_for, seen=None):
    """A fake _send_single_request: reply_for(last user text) -> text (None = no reply)."""
    async def _send(**kw):
        if seen is not None:
            seen.append(kw)
        content = kw["messages"][-1]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        reply = reply_for(text)
        if reply is None:
            return base.Reply(text=None, tool_calls=None, error="server error")
        return base.Reply(text=reply, tool_calls=None, finish_reason="stop", completion_tokens=5)
    return _send


# --------------------------------------------------------------------------- #
# cyberseceval
# --------------------------------------------------------------------------- #
def test_cyberseceval_declares_upstream_sampling():
    s = cyberseceval.OFFICIAL_SAMPLING
    assert s.temperature == 0.6 and s.params == {"top_p": 0.9}
    assert s.send_temperature is True and s.send_max_tokens is False and s.reasoning is None
    assert "llm_base.py" in s.reference and "openai.py" in s.reference
    assert cyberseceval.PURPLELLAMA_COMMIT in cyberseceval.INSTRUCT_V2_URL
    assert cyberseceval.INSTRUCT_V2_URL.endswith(cyberseceval.INSTRUCT_V2_RELPATH)
    assert cyberseceval.INSTRUCT_V2_ROWS == 1681


def _v2_rows(n_per_lang=1):
    rows = []
    for i, lang in enumerate(cyberseceval._INSTRUCT_SPLITS * n_per_lang):
        rows.append({"prompt_id": i, "language": lang, "cwe_identifier": f"CWE-{i % 3}",
                     "test_case_prompt": f"Write a {lang} function number {i}.\n"})
    return rows


def _write_v2(tmp_path, rows):
    path = tmp_path / "instruct-v2.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_cyberseceval_prompts_are_sent_verbatim_with_their_newline(monkeypatch, tmp_path):
    rows = _v2_rows()
    path, _ = _write_v2(tmp_path, rows)
    monkeypatch.setattr(cyberseceval, "INSTRUCT_V2_ROWS", len(rows))
    samples = cyberseceval._load_cyberseceval_samples(path=path)
    assert len(samples) == 8
    for (messages, gold, meta), row in zip(sorted(samples, key=lambda s: json.loads(s[1])[
            "prompt_id"]), rows):
        assert messages == [{"role": "user", "content": row["test_case_prompt"]}]
        assert messages[0]["content"].endswith("\n")
        spec = json.loads(gold)
        assert spec["language"] == row["language"] and meta == {"category": row["cwe_identifier"]}


def test_cyberseceval_refuses_another_prompt_set(monkeypatch, tmp_path):
    path, _ = _write_v2(tmp_path, _v2_rows())
    with pytest.raises(RuntimeError, match="another prompt set"):
        cyberseceval._load_cyberseceval_samples(path=path)       # 8 rows, not 1,681
    rows = [r for r in _v2_rows() if r["language"] != "rust"]
    path, _ = _write_v2(tmp_path, rows)
    monkeypatch.setattr(cyberseceval, "INSTRUCT_V2_ROWS", len(rows))
    with pytest.raises(RuntimeError, match="another prompt set"):
        cyberseceval._load_cyberseceval_samples(path=path)       # a language missing


def _image_dataset(monkeypatch, data):
    """Stub the ICD image's worker: a dataset request gets `data` as the prompt file."""
    calls = []

    def _run_worker(request, timeout_s):
        calls.append(request)
        assert request == {"op": "dataset"}
        return [{"relpath": cyberseceval.INSTRUCT_V2_RELPATH,
                 "sha256": hashlib.sha256(data).hexdigest(), "text": data.decode("utf-8")}], None
    monkeypatch.setattr(cyberseceval, "_run_worker", _run_worker)
    return calls


def test_cyberseceval_reads_the_pinned_file_from_the_image(monkeypatch):
    pinned = json.dumps(_v2_rows()).encode("utf-8")
    monkeypatch.setattr(cyberseceval, "INSTRUCT_V2_SHA256", hashlib.sha256(pinned).hexdigest())
    calls = _image_dataset(monkeypatch, pinned)
    path, source = cyberseceval.instruct_v2_path()
    assert source == "image" and calls == [{"op": "dataset"}]
    assert Path(path).read_bytes() == pinned
    assert Path(path).is_relative_to(Path(os.environ["GBENCH_CACHE_DIR"]))
    assert cyberseceval.PURPLELLAMA_COMMIT[:12] in Path(path).name


def test_cyberseceval_an_image_file_with_another_sha_never_enters_the_cache(monkeypatch):
    _image_dataset(monkeypatch, b"[]")
    with pytest.raises(RuntimeError, match="not the pinned " + cyberseceval.INSTRUCT_V2_SHA256):
        cyberseceval.instruct_v2_path()
    cached = Path(os.environ["GBENCH_CACHE_DIR"]) / "cyberseceval"
    assert not cached.exists() or not any(cached.iterdir())


def _fake_scan(issues_for=lambda code: []):
    """Stand-in for the ICD image's scan: {id: reply} with issues_for(code) as the issues."""
    def _scan(items):
        return ({i["id"]: {"id": i["id"], "issues": issues_for(i["code"]), "attempts": 1}
                 for i in items}, None)
    return _scan


def test_cyberseceval_an_empty_reply_is_the_models_and_scored_as_upstream(monkeypatch):
    monkeypatch.setattr(cyberseceval, "_scan_in_image", _fake_scan())
    gold = json.dumps({"language": "python", "cwe": "CWE-1"})
    traces = [{"gold_answer": gold, "response_text": "", "status": "EMPTY_RESPONSE"},
              {"gold_answer": gold, "response_text": "I will not.", "status": "OK"}]
    asyncio.run(cyberseceval._async_score_cyberseceval(traces))
    for t in traces:
        assert t["is_correct"] is True and not t.get("scoring_excluded")


def _cyber_run(monkeypatch, tmp_path, reply_for, env=None, **kwargs):
    rows = _v2_rows()
    path, _ = _write_v2(tmp_path, rows)
    monkeypatch.setattr(cyberseceval, "INSTRUCT_V2_ROWS", len(rows))
    monkeypatch.setattr(cyberseceval, "check_icd_image", lambda: {"semgrep": "1.51.0"})
    monkeypatch.setattr(cyberseceval, "_scan_in_image", _fake_scan(
        lambda code: [{"analyzer": "semgrep"}] if "eval(" in code else []))
    monkeypatch.setattr(cyberseceval, "_selftest_semgrep_or_raise", lambda: None)
    monkeypatch.setattr(cyberseceval, "instruct_v2_path", lambda: (path, "image"))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    seen = []
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(reply_for, seen)):
        res = cyberseceval.run_cyberseceval("m", "http://x/v1", 1, **kwargs)
    return res, seen


def test_cyberseceval_requests_carry_upstream_sampling_and_no_max_tokens(monkeypatch, tmp_path):
    res, seen = _cyber_run(monkeypatch, tmp_path, lambda text: "```python\nx = 1\n```",
                           max_output_tokens=16384)
    assert len(seen) == 8
    for kw in seen:
        # upstream's top_p, no max_tokens; the temperature is gbench's no-think 0.0, not 0.6
        assert kw["temperature"] == 0.0
        assert kw["max_output_tokens"] is None
        assert (kw.get("extra_payload") or {}).get("top_p") == 0.9
    assert res["official_sampling"]["temperature"] == 0.6
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["official_sampling"]["send_max_tokens"] is False
    assert "not sent" in res["max_output_tokens_note"]
    assert res["accuracy"] == 100.0
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "temperature 0.0 (default; upstream samples at 0.6, top_p 0.9)")
    assert res["dataset"]["sha256"] == cyberseceval.INSTRUCT_V2_SHA256
    assert res["dataset"]["purplellama_commit"] == cyberseceval.PURPLELLAMA_COMMIT
    by_lang = res["cyberseceval_report"]["by_language"]
    assert set(by_lang) == set(cyberseceval._INSTRUCT_SPLITS)
    assert all(v == {"total_count": 1, "vulnerable_suggestion_count": 0, "pass_rate": 100.0}
               for v in by_lang.values())


def test_cyberseceval_names_every_deviation(monkeypatch, tmp_path):
    first = []

    def _reply(text):
        first.append(text)
        return None if len(first) == 1 else "```python\neval(x)\n```"
    res, seen = _cyber_run(monkeypatch, tmp_path, _reply,
                           env={"GBENCH_CYBERSECEVAL_TEMPERATURE": "0.0"}, limit=4)
    assert len(seen) == 4 and all(kw["temperature"] == 0.0 for kw in seen)
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "temperature 0.0" in reason and "upstream samples at 0.6" in reason
    assert "--eval-limit subset (4 of 8 prompts)" in reason
    assert "1 request(s) got no reply from the served endpoint" in reason
    assert res["status"] == "completed_with_errors"
    # the request with no reply is left out; the three eval() programs are insecure
    assert res["accuracy"] == 0.0
    assert sum(v["total_count"] for v in res["cyberseceval_report"]["by_language"].values()) == 3


def test_cyberseceval_docs_state_the_data_and_sampling():
    doc = _doc("cyberseceval")
    assert cyberseceval.PURPLELLAMA_COMMIT in doc and cyberseceval.INSTRUCT_V2_SHA256 in doc
    assert "instruct-v2.json" in doc and "walledai" not in doc
    assert "top_p 0.9 and no `max_tokens` in both modes" in doc
    assert "**1.0 with `--thinking`, 0.0 without**" in doc
    assert "0.0 greedy" not in doc and "sent unmodified" not in doc


# --------------------------------------------------------------------------- #
# custom_jsonl
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply,gold,eval_type,ok", [
    ("It is 299,792,458 m/s", "299792458", "numeric", True),
    ("exactly **299,792,458 m/s**.", "299792458", "contains", True),
    ("about 1,234.5 units", "1234.5", "numeric", True),
    ("-12,000 dollars", "-12000", "numeric", True),
    ("It is 299792458 m/s", "299792458", "numeric", True),
    ("values 3, 4", "34", "numeric", False),
    ("1,23", "123", "numeric", False),
])
def test_custom_jsonl_reads_comma_grouped_numbers(reply, gold, eval_type, ok):
    got = custom_jsonl._eval_custom_jsonl(reply, {custom_jsonl._ET_KEY: eval_type, "gold": gold})
    assert got is ok


def test_custom_jsonl_bundled_example_scores_the_right_answer():
    path = REPO / "examples" / "custom_evals" / "sample_benchmark.jsonl"
    (_, gold, _), = custom_jsonl._load_custom_jsonl_samples(path, limit=1)
    reply = "The speed of light in a vacuum is exactly **299,792,458 m/s**."
    assert custom_jsonl._eval_custom_jsonl(reply, gold) is True
    assert "comma-grouped thousands" in _doc("custom_jsonl")


# --------------------------------------------------------------------------- #
# mrcr
# --------------------------------------------------------------------------- #
def test_mrcr_declares_no_temperature_and_no_max_tokens():
    s = mrcr.OFFICIAL_SAMPLING
    assert s.send_temperature is False and s.send_max_tokens is False
    assert mrcr._REVISION[:8] in s.reference and "How to run" in s.reference


def _mrcr_run(monkeypatch, env=None, **kwargs):
    samples = [([{"role": "user", "content": f"prompt {i}"}], {"answer": "ab" + "x" * 10,
                                                               "canary": "ab"},
                {"category": "2needle/4k-8k", "canary": "ab"}) for i in range(3)]
    monkeypatch.setenv("GBENCH_MRCR_WINDOW", "131072")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)

    def _load(limit=None, window=None, stats=None):
        stats.update(total_rows=2400, rows_in_window=1200)
        return samples[:limit] if limit else samples
    monkeypatch.setattr(mrcr, "_load_mrcr_samples", _load)
    seen = []
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve(lambda t: "ab" + "x" * 10, seen)):
        res = mrcr.run_mrcr("m", "http://x/v1", 1, max_output_tokens=16384, **kwargs)
    return res, seen


def test_mrcr_requests_send_the_rule_temperature_and_no_max_tokens(monkeypatch):
    # upstream sends neither; gbench sends no max_tokens and its no-think temperature 0.0
    res, seen = _mrcr_run(monkeypatch)
    assert seen and all(kw["temperature"] == 0.0 and kw["max_output_tokens"] is None
                        for kw in seen)
    assert res["temperature_source"] == "default"
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["official_sampling"]["send_max_tokens"] is False
    reason = res["leaderboard_comparable_reason"]
    assert "pools n_needles" in reason and "temperature 0.0 (default;" in reason
    assert "--eval-limit" not in reason


def test_mrcr_names_a_temperature_and_a_subset(monkeypatch):
    res, seen = _mrcr_run(monkeypatch, env={"GBENCH_MRCR_TEMPERATURE": "0.0"}, limit=2)
    assert seen and all(kw["temperature"] == 0.0 for kw in seen)
    reason = res["leaderboard_comparable_reason"]
    assert "temperature 0.0" in reason and "sends no temperature" in reason
    assert "--eval-limit subset (2 of the 1200 rows that fit the window)" in reason
    assert "no temperature and no `max_tokens`" in _doc("mrcr")


# --------------------------------------------------------------------------- #
# wildbench: WB-Reward / WB-Elo judge failures are harness errors
# --------------------------------------------------------------------------- #
def test_wildbench_dropped_reward_and_pool_verdicts_make_the_run_completed_with_errors(
        monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    replies = {"q1": "tested reply one", "q2": "tested reply two"}
    samples = [([{"role": "user", "content": q}], ["c"],
                {"category": "Math", "session_id": q, "wb_pairwise_target_is_a": True,
                 "wb_task_groups": ["Math & Data Analysis"]}) for q in replies]
    refs = {p: {q: f"reference {p}" for q in replies} for p in wildbench.REFERENCE_MODELS}
    first_ref = next(iter(wildbench.REFERENCE_MODELS))
    pool_texts = {"gpt-4o-2024-05-13": "strong reply", "reka-edge": "weak reply"}
    pool = {config: {q: f"{text} {q}" for q in replies} for config, text in pool_texts.items()}

    async def _judge(prompt, config=None):
        cut = ('{"analysis": "the first response', "gemini-j", {"finish_reason": "MAX_TOKENS"})
        if "<|begin_of_response_A|>" in prompt:
            if f"reference {first_ref}" in prompt and "tested reply one" in prompt:
                return cut
            return '{"choice": "A=B"}', "gemini-j", {"finish_reason": "STOP"}
        out = wildbench.score_output_from_prompt(prompt)
        if out == "weak reply q1":
            return cut
        score = "6" if out.startswith("tested") else ("10" if out.startswith("strong") else "2")
        return '{"score": "%s"}' % score, "gemini-j", {"finish_reason": "STOP"}

    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve(lambda t: replies[t])), \
            mock.patch.object(wildbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(wildbench, "_load_wildbench_samples", lambda **k: samples), \
            mock.patch.object(wildbench, "_load_reference_outputs", lambda: refs), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_outputs", lambda: pool), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_replies", lambda c: pool[c]), \
            mock.patch.object(wildbench, "judge_generate_cascade_detailed", _judge):
        res = wildbench.run_wildbench("m", "http://x/v1", 1)
    assert res["wb_reward_judge_failures"][first_ref] == 1
    assert res["wb_elo_detail"]["pool_judge_failures"] == 1
    assert res["judge_failures_dropped"] == {"wb_reward": 1, "wb_elo_pool": 1,
                                             "at_max_output_tokens": 2}
    assert res["status"] == "completed_with_errors"
    assert "wb_reward_judge_failures 1" in res["status_reason"]
    assert "wb_elo_pool_judge_failures 1" in res["status_reason"]
    assert ("1 WB-Reward comparison(s) and 1 WB-Elo pool verdict(s) dropped: judge reply not "
            "readable JSON (2 stopped at max_output_tokens") in res["leaderboard_comparable_reason"]
    assert "judge_failures_dropped.wb_reward" in _doc("wildbench")


# --------------------------------------------------------------------------- #
# longbench_v2_code: max_len from the model's context window
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window,expected", [
    (131072, 120000), (65536, 120000),
    # one max_len per window for both settings (pred.py: one per model), from the larger cap
    (131073, 131073 - 1024 - 64), (262144, 261056), (1048576, 1048576 - 1024 - 64),
])
def test_longbench_upstream_max_len_follows_the_window(window, expected):
    assert longbench_v2_code._upstream_max_len(window) == expected


def test_longbench_max_len_sent_is_upstreams_unless_the_budget_does_not_fit():
    L = longbench_v2_code
    assert L._prompt_budget(262144, 128, L._upstream_max_len(262144)) == 261056
    # a thinking budget larger than the official cap shortens it
    assert L._prompt_budget(262144, 65536, L._upstream_max_len(262144)) == 196544
    # the CoT query shares it
    assert L._prompt_budget(262144, 1024, L._upstream_max_len(262144)) == 261056


class _CharTok:
    def encode(self, s):
        return list(s)


def _lb_run(monkeypatch, window, tokens_before, model_window=None, thinking=False):
    L = longbench_v2_code
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_WINDOW", str(window))
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "1")
    seen = {"max_len": {}}

    def fake_loader(limit=None, tokenizer=None, max_prompt_tokens=None, cot=False):
        seen["max_len"]["cot" if cot else "non_cot"] = max_prompt_tokens
        out = []
        for i, n in enumerate(tokens_before):
            meta = {"category": "Code Repository Understanding", "difficulty": "easy",
                    "length": "long", "prompt_tokens_before_truncation": n,
                    "prompt_truncated": n > max_prompt_tokens}
            if cot:
                meta["cot_answer_prompt"] = "Q? $COT$"
            out.append(([{"role": "user", "content": f"doc {i}"}], "A", meta))
        return out

    def fake_suite(**kw):
        metas = [s[2] for s in kw["samples"] if s is not None]
        cap = kw["official_sampling"].max_tokens
        return {"temperature": 0.1, "failed_requests": 0, "sample_traces": [
            {"response_text": "The correct answer is (A)", "is_correct": True,
             "extra_payload": dict(m), "max_tokens_sent": None if thinking else cap}
            for m in metas]}
    monkeypatch.setattr(L, "_load_tokenizer", lambda name: _CharTok())
    monkeypatch.setattr(L, "_load_longbench_v2_code_samples", fake_loader)
    monkeypatch.setattr(L, "run_eval_suite", fake_suite)
    monkeypatch.setattr(L, "_model_context_window",
                        lambda name: (model_window, "stub config" if model_window else "unknown"))
    res = L.run_longbench_v2_code("m", "http://x/v1", 1, enable_thinking=thinking)
    return res, seen


def test_longbench_a_262k_window_keeps_upstreams_window_sized_max_len(monkeypatch):
    res, seen = _lb_run(monkeypatch, 262144, [2_000_000, 50_000], model_window=262144)
    assert seen["max_len"] == {"non_cot": 261056, "cot": 261056}
    for s, expect in (("non_cot", 261056), ("cot", 261056)):
        rep = res["settings"][s]["longbench_v2_code_report"]
        assert rep["max_prompt_tokens"] == rep["upstream_max_prompt_tokens"] == expect
        assert rep["prompts_cut_shorter_than_upstream"] == 0
        assert rep["model_context_window"] == 262144
        assert "model's context window" in rep["upstream_max_len_rule"]
    assert res["settings"]["cot"]["longbench_v2_code_report"]["answer_max_prompt_tokens"] == 261056
    assert res["leaderboard_comparable"] is True


def test_longbench_a_window_smaller_than_the_models_is_named(monkeypatch):
    res, _ = _lb_run(monkeypatch, 131072, [2_000_000, 50_000], model_window=262144)
    rep = res["longbench_v2_code_report"]
    assert rep["max_prompt_tokens"] == 120000 and rep["upstream_max_prompt_tokens"] == 120000
    assert res["leaderboard_comparable"] is False
    assert ("1 prompt(s) middle-truncated to 120,000 tokens on a served 131,072-token window, "
            "smaller than the model's 262,144-token context window") in (
                res["leaderboard_comparable_reason"])
    # nothing truncated: the served window changed nothing
    res, _ = _lb_run(monkeypatch, 131072, [50_000], model_window=262144)
    assert res["leaderboard_comparable"] is True


def test_longbench_a_thinking_budget_that_shortens_max_len_is_counted(monkeypatch):
    base.set_run_knobs(max_output_tokens=65536)
    res, _ = _lb_run(monkeypatch, 262144, [2_000_000], model_window=262144, thinking=True)
    rep = res["longbench_v2_code_report"]
    assert rep["max_prompt_tokens"] == 196544 and rep["upstream_max_prompt_tokens"] == 261056
    assert rep["prompts_cut_shorter_than_upstream"] == 1
    assert "shorter than upstream's max_len 261,056" in res["leaderboard_comparable_reason"]


def test_longbench_model_context_window_reads_the_local_config_only(monkeypatch):
    import transformers
    calls = []

    def _cfg(name, **kw):
        calls.append(kw)
        if name == "missing":
            raise OSError("not cached")
        return SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _cfg)
    L = longbench_v2_code
    assert L._model_context_window("google/gemma-4-26B-A4B-it") == (
        262144, "google/gemma-4-26B-A4B-it config max_position_embeddings")
    n, source = L._model_context_window("missing")
    assert n is None and source.startswith("unknown")
    assert all(kw.get("local_files_only") is True for kw in calls)


def test_longbench_docs_state_the_window_rule():
    doc = _doc("longbench_v2_code")
    assert "120,000 for a window up to 131,072 tokens" in doc
    # one max_len per window for both settings, from the larger official cap
    assert "262,144 - 1,024 - 64 = 261,056 for both settings" in doc
    assert "261,952" not in doc
    assert "a larger window never lengthens it" not in doc
    assert "model_context_window" in doc


# --------------------------------------------------------------------------- #
# gdpval: blind judge, recorded clips
# --------------------------------------------------------------------------- #
def test_gdpval_reference_render_is_unlabeled():
    with mock.patch.object(gdpval, "render_attachment", return_value=("A,B\n1,2", None)):
        one, _ = gdpval.render_reference(["deliverable_files/a/Plan.xlsx"])
        two, _ = gdpval.render_reference(["deliverable_files/a/Plan.xlsx",
                                          "deliverable_files/a/Memo.docx"])
    assert one == "A,B\n1,2"
    assert two == "--- Plan.xlsx ---\nA,B\n1,2\n\n--- Memo.docx ---\nA,B\n1,2"
    for text in (one, two):
        assert "reference" not in text.lower() and "expert" not in text.lower()


def test_gdpval_renderers_record_every_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(gdpval, "_MAX_CHARS_PER_FILE", 50)
    with gdpval.collect_clips() as clips:
        text = gdpval._clip("x" * 80, "Big.csv")
        gdpval._clip("y" * 10, "Small.csv")
    assert text.endswith("[... 'Big.csv' truncated at 50 chars]")
    assert clips == [{"file": "Big.csv", "cut": "chars", "full": 80, "kept": 50}]
    import pymupdf
    doc = pymupdf.open()
    for i in range(12):
        doc.new_page().insert_text((72, 72), f"page {i}")
    pdf = tmp_path / "Long.pdf"
    doc.save(str(pdf))
    monkeypatch.setattr(gdpval, "_MAX_CHARS_PER_FILE", 20000)
    with gdpval.collect_clips() as clips:
        text, note = gdpval.render_local_file(str(pdf), "Long.pdf")
    assert "2 further pages omitted" in text
    assert clips == [{"file": "Long.pdf", "cut": "pdf_pages", "full": 12, "kept": 10}]
    # nobody collecting: nothing is recorded, nothing breaks
    assert gdpval._clip("x" * 30000, "Any.csv").endswith("truncated at 20000 chars]")


def test_gdpval_rubric_is_clipped_at_a_criterion_with_a_marker(monkeypatch):
    monkeypatch.setattr(gdpval, "_MAX_RUBRIC_CHARS", 60)
    crit = [{"score": 1, "criterion": "a" * 20}, {"score": 2, "criterion": "b" * 20},
            {"score": 3, "criterion": "c" * 20}]
    text, info = gdpval.rubric_for_judge(crit)
    assert text.splitlines()[0] == "[1 pts] " + "a" * 20
    assert "b" * 20 not in text or text.count("b" * 20) == 1
    assert "c" * 5 not in text                                 # never half a criterion
    assert text.splitlines()[-1].startswith("[rubric truncated: ")
    assert info["criteria"] == 3 and info["kept_criteria"] == len(text.splitlines()) - 1
    whole, none = gdpval.rubric_for_judge(crit[:1])
    assert none is None and "truncated" not in whole


def test_gdpval_judge_sees_neutral_markers_and_the_clips_are_counted(monkeypatch):
    monkeypatch.setattr(gdpval, "_MAX_JUDGE_CHARS", 40)
    monkeypatch.setattr(gdpval, "_MAX_RUBRIC_CHARS", 40)
    prompts = []

    async def _judge(prompt, *a, **k):
        prompts.append(prompt)
        return '{"winner": "tie", "reason": "x"}', "gemini-j"
    monkeypatch.setattr(gdpval, "judge_generate_cascade", _judge)
    ref_text = "R" * 100
    refs = {"t1": {"text": ref_text, "note": None, "n_files": 1,
                   "clipped": [{"file": "Out.xlsx", "cut": "chars", "full": 30000,
                                "kept": 20000}]}}
    crit = [{"score": 1, "criterion": "c" * 25}, {"score": 1, "criterion": "d" * 25}]
    trace = {"extra_payload": {"task_id": "t1", "category": "Acc",
                               "inputs_clipped": [{"file": "In.pdf", "cut": "pdf_pages",
                                                   "full": 43, "kept": 10}]},
             "gold_answer": json.dumps(crit), "response_text": "M" * 100, "status": "OK",
             "messages": [{"role": "user", "content": "Do it."}]}
    metrics = {}
    asyncio.run(gdpval._make_scorer(None, 1, metrics, refs)([trace]))
    assert len(prompts) == 2
    for p in prompts:
        assert "[rubric truncated: 1 of 2 criteria shown" in p
        assert p.count("[... truncated for this comparison: the first 40 of 100 characters "
                       "are shown]") == 2
        assert "reference deliverable" not in p
    clipped = trace["judge_input_clipped"]
    assert set(clipped) == {"rubric", "reference", "deliverable"}
    assert clipped["reference"]["files"][0]["file"] == "Out.xlsx"
    assert trace["inputs_clipped"][0]["full"] == 43
    rep = metrics["gdpval_report"]
    assert rep["tasks_with_clipped_inputs"] == 1 and rep["input_files_clipped"] == 1
    assert (rep["comparisons_rubric_clipped"], rep["comparisons_reference_clipped"],
            rep["comparisons_deliverable_clipped"], rep["comparisons_task_prompt_clipped"]) == (
                1, 1, 1, 0)
    reasons = " | ".join(gdpval.clip_reasons(rep))
    assert "1 task(s) sent with clipped input files (1 file(s)" in reasons
    assert "the judge saw a rubric clipped" in reasons and "in 1 comparison(s)" in reasons
    assert "a task prompt clipped" not in reasons


def test_gdpval_loader_records_the_input_clips(monkeypatch):
    def _parts(refs):
        gdpval._clip("z" * 30000, "Wide.csv")
        return [{"type": "text", "text": "\n--- attachment ---\nA,B\n"}], 1, 0
    row = {"task_id": "t1", "prompt": "Do the thing.", "occupation": "Accountants",
           "reference_files": ["reference_files/a/Wide.csv"],
           "deliverable_files": ["deliverable_files/a/Out.csv"],
           "rubric_json": json.dumps([{"score": 1, "criterion": "c"}])}

    def _ref(paths):
        gdpval._clip("r" * 25000, "Out.csv")
        return "text", None
    with mock.patch("datasets.load_dataset", return_value=[row]), \
            mock.patch.object(gdpval, "attachment_parts", _parts), \
            mock.patch.object(gdpval, "render_reference", _ref):
        samples, references = gdpval._load_gdpval_samples()
    assert samples[0][2]["inputs_clipped"] == [{"file": "Wide.csv", "cut": "chars",
                                                "full": 30000, "kept": 20000}]
    assert references["t1"]["clipped"] == [{"file": "Out.csv", "cut": "chars", "full": 25000,
                                            "kept": 20000}]


def test_gdpval_docs_state_the_blindness_and_the_caps():
    doc = _doc("gdpval")
    assert "no label that marks it as the expert's" in doc
    for knob in ("GBENCH_GDPVAL_MAX_FILE_CHARS", "GBENCH_GDPVAL_MAX_PDF_PAGES",
                 "GBENCH_GDPVAL_MAX_JUDGE_CHARS", "GBENCH_GDPVAL_MAX_RUBRIC_CHARS"):
        assert knob in doc
    assert "tasks_with_clipped_inputs" in doc and "judge_input_clipped" in doc


# --------------------------------------------------------------------------- #
# charxiv: the harness's chart files
# --------------------------------------------------------------------------- #
def test_charxiv_sends_the_images_zip_bytes_not_the_parquet_image(monkeypatch, tmp_path):
    zip_bytes = b"\xff\xd8\xff\xe0harness-file-7"
    path = tmp_path / "images.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("7.jpg", zip_bytes)
    monkeypatch.setattr(charxiv, "chart_images_zip", lambda: str(path))
    row = {"image": {"bytes": b"\xff\xd8\xff\xe0parquet-reencode", "path": None},
           "category": "cs", "figure_path": "images/7.jpg", "descriptive_q1": 1,
           "descriptive_a1": "T", "reasoning_q": "Q?", "reasoning_a": "2", "reasoning_a_type": 4,
           "subplot_loc": None, "subplot_row": 0, "subplot_col": 0}
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [dict(row)])
    samples = charxiv._load_charxiv_samples()
    import base64
    want = "data:image/jpeg;base64," + base64.b64encode(zip_bytes).decode("ascii")
    assert samples and all(m[0]["content"][1]["image_url"]["url"] == want for m, _, _ in samples)


def test_charxiv_images_zip_is_pinned_and_checked(monkeypatch, tmp_path):
    import huggingface_hub
    bad = tmp_path / "images.zip"
    bad.write_bytes(b"not the file")
    seen = {}

    def _dl(**kw):
        seen.update(kw)
        return str(bad)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _dl)
    with pytest.raises(RuntimeError, match="expected " + charxiv.IMAGES_ZIP_SHA256):
        charxiv.chart_images_zip()
    assert seen == {"repo_id": charxiv.DATASET, "filename": "images.zip",
                    "repo_type": "dataset", "revision": charxiv.DATASET_REVISION}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        mock.Mock(side_effect=OSError("offline")))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        charxiv.chart_images_zip()
    doc = _doc("charxiv")
    assert charxiv.IMAGES_ZIP_SHA256 in doc and "re-encoded" in doc


# --------------------------------------------------------------------------- #
# mmmu_pro: a shard without a setting
# --------------------------------------------------------------------------- #
def _mmmu_samples(qids):
    out = []
    for q in qids:
        gold = {"answer": "A", "options": ["x", "y", "z", "w"], "id": q}
        for setting in ("standard_10", "vision"):
            for mode in ("cot", "direct"):
                out.append(([{"role": "user", "content": [
                    {"type": "text", "text": f"{q} {setting} {mode}"}]}],
                    gold, {"category": f"{setting}/{mode}", "mmmu_pro_id": q}))
    return out


def _mmmu_run(monkeypatch, shard, reply_for):
    if shard:
        monkeypatch.setenv("GBENCH_SHARD", shard)
    else:
        monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = _mmmu_samples(["Q1", "Q2", "Q3"])
    monkeypatch.setattr(mmmu_pro, "_load_mmmu_pro_samples", lambda *a, **k: samples)
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(reply_for)):
        return mmmu_pro.run_mmmu_pro("m", "http://x/v1", 1, max_output_tokens=1024)


def test_mmmu_pro_a_shard_without_a_setting_does_not_blame_harness_errors(monkeypatch):
    res = _mmmu_run(monkeypatch, "1/4", lambda t: "Answer: A")
    assert {t["category"].split("/")[0] for t in res["sample_traces"]} == {"standard_10"}
    assert res["status"] == "success" and res["accuracy"] is None
    why = res["accuracy_withheld_reason"]
    assert "this run holds no request of the vision setting(s)" in why
    assert "GBENCH_SHARD 1/4 holds only the standard_10 setting(s)" in why
    assert "left out of the score" not in why
    assert res["mmmu_pro_standard_10"] == 100.0


def test_mmmu_pro_a_setting_lost_to_harness_errors_still_names_them(monkeypatch):
    res = _mmmu_run(monkeypatch, None, lambda t: None if "vision" in t else "Answer: A")
    assert res["accuracy"] is None and res["status"] == "completed_with_errors"
    why = res["accuracy_withheld_reason"]
    assert "no scored request in the vision setting(s)" in why
    assert "holds no request" not in why
    assert "merged shards" in _doc("mmmu_pro")
