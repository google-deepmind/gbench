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

"""simple-evals request settings for simpleqa and healthbench, the WMT19 download path of
i18n_translate, WildBench's WB-Elo pool (lazy prompts, cache key, seed wording) and the
ScreenSpot-v2 screenshot MIME types.

No model server, judge or network is contacted: requests, judges and downloads are replaced.
"""
import asyncio
import hashlib
import io
import json
import os
import tarfile
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, gpqa, healthbench, i18n_translate,
                                        screenspot_pointing, simpleqa, wildbench)

def _patch_detailed(module, judge):
    """Patch a suite's judge_generate_cascade_detailed with a (text, model) judge stub; the
    detailed call returns (text, model, info) with nothing cut off at the grader's cap."""
    async def detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {"finish_reason": None, "max_tokens_without_text": 0}
    return mock.patch.object(module, "judge_generate_cascade_detailed", detailed)


def _simpleqa_detailed(judge):
    """A (text, model) judge stub as simpleqa's detailed cascade call (text, model, info)."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed


def _as_detailed(judge):
    """A (text, model) judge stub as the cascade's detailed call, (text, model, info), which
    wildbench uses to tell a refusal from an outage."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs" / "evals"


def _doc(name):
    return (DOCS / f"{name}.md").read_text(encoding="utf-8")


def _serve(replies, seen, finish_reason="stop"):
    """A fake _send_single_request answering by the last user text."""
    async def _send(**kw):
        seen.append(kw)
        content = kw["messages"][-1]["content"]
        return base.Reply(text=replies[content], tool_calls=None, finish_reason=finish_reason,
                          completion_tokens=5)
    return _send


# --------------------------------------------------------------------------- #
# simpleqa / healthbench: what simple-evals' samplers send, per mode
# --------------------------------------------------------------------------- #
def _simpleqa(thinking, seen, judge_configs, finish_reason="stop"):
    samples = [([{"role": "user", "content": "Who?"}], "Ada", {"category": "x"})]

    async def _judge(prompt, config=None):
        judge_configs.append(config)
        return "A", "gemini-x"

    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve({"Who?": "Ada"}, seen, finish_reason)), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples", lambda limit=None: samples), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed",
                              _simpleqa_detailed(_judge)):
        return simpleqa.run_simpleqa("m", "http://x/v1", 1, enable_thinking=thinking,
                                     max_output_tokens=16384)


def _healthbench(thinking, seen, judge_configs):
    rubric = json.dumps([{"criterion": "Says hi", "points": 5, "tags": []}])
    samples = [([{"role": "user", "content": "hello"}], rubric,
                {"category": "theme:x", "example_tags": ["theme:x"]})]

    async def _judge(prompt, config=None):
        judge_configs.append(config)
        return '{"explanation": "ok", "criteria_met": true}', "gemini-x"

    with mock.patch.object(base, "_send_single_request", side_effect=_serve({"hello": "hi"}, seen)), \
            mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(healthbench, "_load_healthbench_samples",
                              lambda limit=None: samples), \
            _patch_detailed(healthbench, _judge):
        return healthbench.run_healthbench("m", "http://x/v1", 1, enable_thinking=thinking,
                                           max_output_tokens=16384)


@pytest.mark.parametrize("run", [_simpleqa, _healthbench], ids=["simpleqa", "healthbench"])
def test_no_think_run_sends_the_chat_sampler_request(run):
    base.set_run_knobs(max_output_tokens=16384)
    seen, configs = [], []
    res = run(False, seen, configs)
    req = seen[0]
    assert req["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert req["temperature"] == 0.0        # gbench's no-think rule, not the sampler's 0.5
    # the request's own max_tokens: 2048, not --max-output-tokens (16384) or a floor
    assert req["max_output_tokens"] == 2048 and "max_tokens" not in req["extra_payload"]
    assert res["max_tokens_sent"] == [2048]
    assert res["sampler_max_tokens"] == 2048 and res["replies_at_sampler_max_tokens"] == 0
    # base's official-cap note: the 2048 is the chat sampler's cap in gpqa.SIMPLE_EVALS_SAMPLING
    official = res["official_sampling"]
    assert res["max_output_tokens_note"] == (
        "--max-output-tokens 16384 not applied: the official harness sends max_tokens=2048 with "
        f"every request ({official['reference']})")
    assert res["official_max_tokens_sent"] == [2048]
    assert official["temperature"] == 0.5 and official["temperature_matches_official"] is False
    assert "max_tokens=2048" in official["reference"] and "send_max_tokens" not in official
    assert (official["max_tokens"], official["max_tokens_applied"]) == (2048, True)


@pytest.mark.parametrize("run", [_simpleqa, _healthbench], ids=["simpleqa", "healthbench"])
def test_thinking_run_sends_the_chat_sampler_request(run):
    # A served model takes simple-evals' chat sampler with --thinking too (its o-series
    # samplers are keyed on OpenAI model names); neither the 2048 nor the 0.5 is applied to
    # reasoning: the request is sampled at 1.0 and the result says it does not match.
    base.set_run_knobs(max_output_tokens=16384)
    seen, configs = [], []
    res = run(True, seen, configs)
    req = seen[0]
    assert req["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert req["temperature"] == 1.0 and res["temperature_source"] == "thinking"
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert ("temperature 1.0 (simple-evals' chat sampler samples at 0.5)"
            in res["leaderboard_comparable_reason"])
    # the run's budget, not the 2048, and nothing in the payload
    assert req["max_output_tokens"] == 16384 and "max_tokens" not in (req["extra_payload"] or {})
    assert res["sampler_max_tokens"] is None and res["replies_at_sampler_max_tokens"] == 0
    official = res["official_sampling"]
    assert "send_max_tokens" not in official and official["send_temperature"] is True
    assert (official["max_tokens"], official["max_tokens_applied"],
            official["max_tokens_sent"]) == (2048, False, 16384)
    assert res["max_output_tokens_note"].startswith("official cap not applied to reasoning")


def test_a_reply_cut_at_2048_is_counted_and_graded():
    seen, configs = [], []
    res = _simpleqa(False, seen, configs, finish_reason="length")
    assert res["replies_at_sampler_max_tokens"] == 1
    assert res["accuracy"] == 100.0                  # graded on what arrived, as simple-evals


@pytest.mark.parametrize("run", [_simpleqa, _healthbench], ids=["simpleqa", "healthbench"])
def test_the_grader_call_follows_simple_evals_grading_sampler(run):
    seen, configs = [], []
    res = run(False, seen, configs)
    cfg = configs[0]
    assert cfg.temperature == 0.5
    assert cfg.system_instruction == "You are a helpful assistant."
    # grading_sampler's max_tokens=2048 for the reply plus the Gemini thinking allowance
    assert cfg.max_output_tokens == 2048 + 8192
    assert res["judge_sampling"]["temperature"] == 0.5
    assert res["judge_sampling"]["max_output_tokens"] == 2048 + 8192
    assert "max_tokens=2048" in res["judge_sampling"]["max_output_tokens_note"]
    assert "gpt-4.1-2025-04-14" in res["judge_sampling"]["reference"]


def test_healthbench_grader_sends_no_response_format():
    # simple-evals' grading_sampler sends no response_format, so neither grader asks for JSON mode
    cfg = simpleqa.grader_config()
    assert cfg.response_mime_type is None and cfg.temperature == 0.5
    assert base.judge_config().temperature == 0.0      # the house default is untouched
    src = (REPO / "gbench/runners/eval_suites/healthbench.py").read_text(encoding="utf-8")
    assert "grader_config(response_mime_type" not in src


def test_simple_evals_samples_carry_the_sampler_messages_and_no_cap():
    # the 2048 is declared once, in gpqa.SIMPLE_EVALS_SAMPLING, not in each sample
    samples = [([{"role": "user", "content": "Who?"}], "Ada", {"category": "x"})]
    (msgs, _g, meta), = simpleqa.simple_evals_samples(samples, False)
    assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert meta == {"category": "x"}
    # the same messages with --thinking: the chat sampler is a served model's path
    (msgs2, _g, meta), = simpleqa.simple_evals_samples(samples, True)
    assert msgs2 == msgs and meta == {"category": "x"}


def test_a_request_that_got_less_than_2048_is_named():
    result = {"sample_traces": [{"max_tokens_sent": 2048}, {"max_tokens_sent": 900}]}
    reasons = simpleqa.record_sampler_max_tokens(result, thinking=False)
    assert reasons == ["1 request(s) got less than simple-evals' max_tokens=2048 because the "
                       "prompt left too little of the context window"]
    assert simpleqa.record_sampler_max_tokens(dict(result), thinking=True) == []


def test_simple_evals_helpers_are_shared_with_gpqa():
    assert healthbench.simple_evals_messages is gpqa.simple_evals_messages
    assert simpleqa.simple_evals_messages is gpqa.simple_evals_messages
    assert simpleqa.GRADER_SYSTEM_MESSAGE == gpqa.SIMPLE_EVALS_SYSTEM_MESSAGE
    src = (REPO / "gbench/runners/eval_suites/simpleqa.py").read_text(encoding="utf-8")
    assert "CHAT_SYSTEM_MESSAGE =" not in src and "def simple_evals_messages" not in src


def test_simple_evals_docs_state_max_tokens_and_the_grader_settings():
    for name in ("simpleqa", "healthbench"):
        doc = _doc(name)
        assert doc.isascii()
        assert "**`max_tokens` 2048**" in doc and "`replies_at_sampler_max_tokens`" in doc
        assert "`sampler_max_tokens`" in doc and "`max_output_tokens_note`" in doc
        assert "temperature **0.5**" in doc and "`judge_sampling`" in doc
        assert "pinned at temperature 0" not in doc and "temperature 0.0" not in doc
    assert "is sent as every request's\n`max_tokens`" not in _doc("simpleqa")


# --------------------------------------------------------------------------- #
# i18n_translate: the WMT19 download never ends the process
# --------------------------------------------------------------------------- #
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_wmt19_tarball(tmp_path, per_direction=2):
    """A test.tgz laid out like the release (sgm/...), `per_direction` segments per direction."""
    from sacrebleu.dataset import DATASETS
    root = tmp_path / "tgz_src"
    for direction in i18n_translate.DIRECTIONS:
        src_rel, ref_rel = DATASETS["wmt19"].langpairs[direction]
        for rel, kind in ((src_rel, "src"), (ref_rel, "ref")):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            segs = "".join(f'<seg id="{i + 1}">{kind} {direction} {i}</seg>\n'
                           for i in range(per_direction))
            path.write_text(f'<refset>\n<doc sysid="ref" docid="d" genre="news" '
                            f'origlang="en">\n{segs}</doc>\n</refset>\n', encoding="utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(root / "sgm", arcname="sgm")
    return buf.getvalue()


@pytest.fixture
def fake_wmt19(tmp_path, monkeypatch):
    data = _fake_wmt19_tarball(tmp_path)
    monkeypatch.setattr(i18n_translate, "TEST_SET_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(i18n_translate, "DIRECTIONS", {d: 2 for d in i18n_translate.DIRECTIONS})
    raw = Path(os.environ["GBENCH_CACHE_DIR"]) / "sacrebleu" / "wmt19" / "raw"
    return data, raw


def test_a_failed_download_is_infra_required_not_a_process_exit(monkeypatch):
    def _offline(*a, **k):
        raise urllib.error.URLError("network unreachable")
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen", _offline)
    with pytest.raises(RuntimeError, match=r"^\[INFRA REQUIRED\] i18n_translate") as err:
        i18n_translate.run_i18n_translate("m", "http://x/v1", 1)
    assert "data.statmt.org/wmt19/translation-task/test.tgz" in str(err.value)


def test_a_download_with_another_checksum_raises_and_is_not_cached(monkeypatch):
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(b"not the wmt19 tarball"))
    with pytest.raises(RuntimeError, match="expected 5ef66d76") as err:
        i18n_translate._wmt19_dataset()
    assert "INFRA REQUIRED" not in str(err.value)
    raw = Path(os.environ["GBENCH_CACHE_DIR"]) / "sacrebleu" / "wmt19" / "raw"
    assert not (raw / "wmt19.test.tgz").exists() and (raw / "wmt19.test.tgz.part").exists()


def test_the_download_uses_a_timeout(monkeypatch, fake_wmt19):
    data, raw = fake_wmt19
    seen = {}

    def _urlopen(url, timeout=None):
        seen["url"], seen["timeout"] = url, timeout
        return _Resp(data)
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen", _urlopen)
    samples = i18n_translate._load_i18n_translate_samples()
    assert seen == {"url": "https://data.statmt.org/wmt19/translation-task/test.tgz",
                    "timeout": i18n_translate.DOWNLOAD_TIMEOUT_S}
    assert len(samples) == 38 and (raw / "wmt19.test.tgz").read_bytes() == data
    assert not (raw / "wmt19.test.tgz.part").exists()


def test_a_tarball_that_was_never_extracted_is_extracted(monkeypatch, fake_wmt19):
    data, raw = fake_wmt19
    raw.mkdir(parents=True)
    (raw / "wmt19.test.tgz").write_bytes(data)

    def _no_network(*a, **k):
        raise AssertionError("no download expected")
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen", _no_network)
    samples = i18n_translate._load_i18n_translate_samples()
    assert len(samples) == 38
    # sacrebleu yields reference lines as read (BLEU rstrips them, chrF ignores whitespace)
    assert samples[0][1].rstrip("\n") == "ref cs-de 0"
    assert samples[0][2] == {"category": "cs-de", "segment": 0}


def test_a_cached_tarball_with_another_checksum_is_fetched_again(monkeypatch, fake_wmt19):
    data, raw = fake_wmt19
    raw.mkdir(parents=True)
    (raw / "wmt19.test.tgz").write_bytes(data[:100])        # cut short
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen", lambda *a, **k: _Resp(data))
    assert len(i18n_translate._load_i18n_translate_samples()) == 38
    assert (raw / "wmt19.test.tgz").read_bytes() == data


def test_a_text_file_cut_short_is_rebuilt(monkeypatch, fake_wmt19):
    data, raw = fake_wmt19
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen", lambda *a, **k: _Resp(data))
    first = i18n_translate._load_i18n_translate_samples()
    src = raw.parent / "wmt19.de-en.src"
    src.write_text("only one line\n", encoding="utf-8")
    assert i18n_translate._load_i18n_translate_samples() == first


def test_a_sacrebleu_exit_becomes_an_exception(monkeypatch, fake_wmt19):
    data, raw = fake_wmt19
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen", lambda *a, **k: _Resp(data))
    dataset = i18n_translate._wmt19_dataset()

    def _exit(*a, **k):
        raise SystemExit(1)
    monkeypatch.setattr(dataset, "source", _exit)
    with pytest.raises(RuntimeError, match="sacrebleu stopped while preparing WMT19 cs-de"):
        i18n_translate._direction_text(dataset, "cs-de", 2)


def test_i18n_scoring_equals_sacrebleu_without_a_skip():
    import sacrebleu                                   # declared in the evals extra
    refs = ["The cat sat on the mat.", "\"Yes,\" he said."]
    hyps = ["The cat sat on a mat.", None]
    traces = [{"category": "de-en", "extra_payload": {"category": "de-en", "segment": i},
               "gold_answer": r, "response_text": h} for i, (r, h) in enumerate(zip(refs, hyps))]
    out = i18n_translate.wmt_scores(traces)["de-en"]
    h = [(x or "").strip() for x in hyps]
    assert out["bleu"] == round(sacrebleu.corpus_bleu(h, [refs]).score, 2)
    assert out["chrf"] == round(sacrebleu.corpus_chrf(h, [refs]).score, 2)
    assert sacrebleu.__version__ == i18n_translate.SACREBLEU_RELEASE


def test_i18n_doc_describes_the_download_path():
    doc = _doc("i18n_translate")
    assert doc.isascii()
    for needle in ("`infra_required`) for this suite\n  only", "wmt19.test.tgz.part",
                   "never asked to download", "extracted again", "120 s read timeout",
                   "sacrebleu==2.6.0` (declared in the `evals` extra"):
        assert needle in doc, needle


# --------------------------------------------------------------------------- #
# WildBench: headline, both WB-Elo columns, lazy pool prompts, cache key, seed wording
# --------------------------------------------------------------------------- #
_POOL = {"reka-edge": "weak reply", "gpt-4o-2024-05-13": "strong reply"}


def _wb_run(monkeypatch, loads):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    replies = {"q1": "tested reply one", "q2": "tested reply two"}
    samples = [([{"role": "user", "content": q}], ["c"],
                {"category": "Math", "session_id": q, "wb_pairwise_target_is_a": True,
                 "wb_task_groups": ["Math & Data Analysis"]}) for q in replies]
    refs = {p: {q: "reference" for q in replies} for p in wildbench.REFERENCE_MODELS}
    pool = {config: {q: f"{text} {q}" for q in replies} for config, text in _POOL.items()}

    def _replies(config):
        loads.append(config)
        return pool[config]

    async def _judge(prompt, config=None):
        if "<|begin_of_response_A|>" in prompt:
            return '{"choice": "A=B"}', "gemini-p"
        out = wildbench.score_output_from_prompt(prompt)
        score = "6" if out.startswith("tested") else "10" if out.startswith("strong") else "2"
        return '{"score": "%s"}' % score, "gemini-s"

    seen = []
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies, seen)), \
            mock.patch.object(wildbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(wildbench, "_load_wildbench_samples", lambda **k: samples), \
            mock.patch.object(wildbench, "_load_reference_outputs", lambda: refs), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_outputs", lambda: pool), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_replies", _replies), \
            mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        return wildbench.run_wildbench("m", "http://x/v1", 1)


def test_headline_is_task_macro_and_both_wb_elo_columns_are_computed(monkeypatch):
    loads = []
    res = _wb_run(monkeypatch, loads)
    assert res["accuracy"] == res["wb_score_task_macro"] is not None
    assert isinstance(res["wb_elo"], float) and isinstance(res["wb_elo_lc"], float)
    cols = res["wb_elo_detail"]["columns"]
    assert cols["WB_Elo"]["length_margin"] == -1 and cols["WB_Elo_LC"]["length_margin"] == 1500
    assert set(res["wb_elo_tables"]) == {"WB_Elo", "WB_Elo_LC"}
    # the pool files are read again for grading, once per model, in model order
    assert loads == sorted(_POOL)
    # a second run takes every pool verdict from the cache and reads no pool file again
    loads2 = []
    res2 = _wb_run(monkeypatch, loads2)
    assert loads2 == [] and res2["wb_elo_detail"]["pool_verdicts_from_cache"] == 4
    assert res2["wb_elo"] == res["wb_elo"] and res2["wb_elo_lc"] == res["wb_elo_lc"]


def test_pool_items_hold_no_prompt_and_no_reply():
    samples = [([{"role": "user", "content": "q"}], ["c"], {"session_id": "s1"})]
    items = wildbench.wb_elo_pool_items(samples, {"a": {"s1": "  a fine reply  "}})
    assert items == [{"config": "a", "session_id": "s1", "output_chars": len("a fine reply"),
                      "prompt_sha256": hashlib.sha256(wildbench.build_wb_judge_prompt(
                          samples[0][0], ["c"], "  a fine reply  ").encode("utf-8")).hexdigest()}]


def _pool_items(n_sessions, configs):
    samples = [([{"role": "user", "content": f"q{i}"}], ["c"], {"session_id": f"s{i}"})
               for i in range(n_sessions)]
    pool = {c: {f"s{i}": f"{c} reply {i}" for i in range(n_sessions)} for c in configs}
    tasks = {m["session_id"]: (msgs, ck) for msgs, ck, m in samples}
    return wildbench.wb_elo_pool_items(samples, pool), tasks, pool


def test_pool_grading_reads_each_model_once_in_model_order():
    items, tasks, pool = _pool_items(5, ["c3", "c1", "c2"])      # items interleave the models
    loads, prompts = [], []

    def _replies(config):
        loads.append(config)
        return pool[config]

    async def _judge(prompt, config=None):
        prompts.append(wildbench.score_output_from_prompt(prompt))
        return '{"score": "5"}', "gemini-s"

    with mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        out = asyncio.run(wildbench.grade_wb_elo_pool(items, tasks, _replies,
                                                      judge_concurrency=2))
    assert loads == ["c1", "c2", "c3"]
    assert [p.split()[0] for p in prompts] == ["c1"] * 5 + ["c2"] * 5 + ["c3"] * 5
    assert sorted(out["grades"]) == ["c1", "c2", "c3"] and out["cached"] == 0


@pytest.mark.parametrize("concurrency", [1, 4])
def test_pool_grading_holds_at_most_two_models_replies(concurrency):
    import gc
    import weakref
    items, tasks, pool = _pool_items(6, ["a", "b", "c", "d"])
    refs = {}
    alive_at_load = []

    class _Replies(dict):
        pass

    def _replies(config):
        gc.collect()
        alive_at_load.append(sum(1 for r in refs.values() if r() is not None))
        holder = _Replies(pool[config])
        refs[config] = weakref.ref(holder)
        return holder

    async def _judge(prompt, config=None):
        await asyncio.sleep(0)
        return '{"score": "5"}', "gemini-s"

    with mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        asyncio.run(wildbench.grade_wb_elo_pool(items, tasks, _replies,
                                                judge_concurrency=concurrency))
    gc.collect()
    assert len(alive_at_load) == 4 and max(alive_at_load) <= 1
    assert all(r() is None for r in refs.values())               # nothing kept after grading


def test_a_prompt_that_changed_since_loading_raises():
    items, tasks, pool = _pool_items(1, ["a"])

    async def _judge(prompt, config=None):
        return '{"score": "5"}', "gemini-s"

    with mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        with pytest.raises(RuntimeError, match="changed between loading and grading"):
            asyncio.run(wildbench.grade_wb_elo_pool(
                items, tasks, lambda c: {"s0": "another reply"}, judge_concurrency=1))


def test_vote_lengths_give_the_same_votes_as_the_reply_text():
    scores = {"s1": {"a": 9, "b": 4, "c": 8, "d": 2}}
    outputs = {"s1": {"a": "x" * 10, "b": "yy", "c": "cut... (truncated)", "d": ""}}
    lengths = {"s1": {m: wildbench.vote_length(o) for m, o in outputs["s1"].items()}}
    assert lengths == {"s1": {"a": 10, "b": 2, "c": None, "d": None}}
    assert wildbench.wb_elo_votes(scores, outputs, 3, 2) == \
        wildbench.wb_elo_votes_from_lengths(scores, lengths, 3, 2)
    assert [(v["model_1"], v["model_2"], v["winner"]) for v in
            wildbench.wb_elo_votes(scores, outputs, 3, 2)] == [("a", "b", "a")]


def test_pool_cache_key_follows_the_judge_config(monkeypatch):
    first = wildbench.wb_elo_cache_dir()
    monkeypatch.setattr(wildbench, "_pool_judge_config",
                        lambda: base.judge_config(response_mime_type="text/plain"))
    assert wildbench.wb_elo_cache_dir() != first
    src = (REPO / "gbench/runners/eval_suites/wildbench.py").read_text(encoding="utf-8")
    assert "json_mode temperature=0" not in src


def test_wb_elo_seed_is_described_as_a_gbench_choice():
    src = (REPO / "gbench/runners/eval_suites/wildbench.py").read_text(encoding="utf-8")
    doc = _doc("wildbench")
    for text in (src, doc):
        assert "num_processes=1 path" not in text and "upstream's\n  `num_processes=1` path" \
            not in text
        assert "OS entropy" in text
    assert "a gbench choice that makes a run deterministic" in doc
    # src/eval.py's judge max_tokens is sent as upstream sends it
    assert "**`max_output_tokens` 9216** is sent" in doc
    assert "about 564 million characters" in doc and doc.isascii()
    assert wildbench.WB_ELO_CONFIG["seed"] == 42


# --------------------------------------------------------------------------- #
# screenspot_pointing: each screenshot with the MIME type of its bytes
# --------------------------------------------------------------------------- #
def _image(fmt):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), (255, 0, 0)).save(buf, fmt)
    return buf.getvalue()


def test_screenshots_are_sent_with_the_mime_type_of_their_bytes(tmp_path, monkeypatch):
    files = {"a.png": _image("JPEG"), "b.png": _image("PNG"), "c.jpg": _image("JPEG")}
    zpath = tmp_path / "screenspotv2_image.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, data in files.items():
            zf.writestr(f"screenspotv2_image/{name}", data)
    paths = {"screenspotv2_image.zip": str(zpath)}
    for (platform, (filename, _sha)), name in zip(screenspot_pointing.SPLITS.items(), files):
        p = tmp_path / filename
        p.write_text(json.dumps([{"img_filename": name, "bbox": [0, 0, 20, 10],
                                  "instruction": f"open {name}", "data_type": "text"}]))
        paths[filename] = str(p)
    monkeypatch.setattr(screenspot_pointing, "_fetch", lambda filename, sha256: paths[filename])
    monkeypatch.setattr(screenspot_pointing, "SPLIT_SIZES", {"mobile": 1, "desktop": 1, "web": 1})
    samples = screenspot_pointing._load_screenspot_pointing_samples()
    urls = {s[2]["img_filename"]: s[0][0]["content"][0]["image_url"]["url"] for s in samples}
    assert urls["a.png"].startswith("data:image/jpeg;base64,")
    assert urls["b.png"].startswith("data:image/png;base64,")
    assert urls["c.jpg"].startswith("data:image/jpeg;base64,")
    import base64
    assert base64.b64decode(urls["a.png"].split(",", 1)[1]) == files["a.png"]
    assert samples[0][1]["bbox"] == [0.0, 0.0, 0.5, 0.5]


def test_screenspot_doc_counts_and_mime():
    doc = " ".join(_doc("screenspot_pointing").split())
    assert "3 of its 1,272 desktop targets" not in doc and "3 of its 334 desktop targets" in doc
    assert "the PNG bytes of the release" not in doc and "image/jpeg" in doc
    src = (REPO / "gbench/runners/eval_suites/screenspot_pointing.py").read_text(encoding="utf-8")
    assert "data:image/png;base64," not in src


# --------------------------------------------------------------------------- #
# humanitys_last_exam: no max_tokens of its own, in either mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thinking", [False, True])
def test_hle_sends_no_max_tokens_of_its_own(thinking):
    from gbench.runners.eval_suites import humanitys_last_exam as hle
    base.set_run_knobs(max_output_tokens=4096)
    samples = [([{"role": "system", "content": hle.SYSTEM_PROMPT},
                 {"role": "user", "content": "Q?"}], "42", {"category": "Math"})]
    seen = []

    async def _judge(contents, config=None):
        return json.dumps({"extracted_final_answer": "42", "reasoning": "r", "correct": "yes",
                           "confidence": 90}), "gemini-x"

    with mock.patch.object(base, "_send_single_request", side_effect=_serve({"Q?": "Answer: 42"}, seen)), \
            mock.patch.object(hle, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(hle, "_load_humanitys_last_exam_samples", lambda limit=None: samples), \
            mock.patch.object(hle, "judge_generate_cascade", _judge):
        res = hle.run_humanitys_last_exam("m", "http://x/v1", 1, enable_thinking=thinking)
    assert seen[0]["max_output_tokens"] is None
    assert "max_tokens" not in (seen[0]["extra_payload"] or {})
    assert res["official_sampling"]["send_max_tokens"] is False
    assert "maximum number of tokens" in res["official_sampling"]["reference"]
    assert "--max-output-tokens 4096 not applied" in res["max_output_tokens_note"]
    # HLE sends reasoning models no temperature; a --thinking run samples at 1.0 regardless
    assert seen[0]["temperature"] == (1.0 if thinking else 0.0)


def test_hle_doc_states_no_max_tokens():
    doc = " ".join(_doc("humanitys_last_exam").split())
    assert "Our evaluation uses the maximum number of tokens available" in doc
    assert "is sent as every request's `max_tokens`" not in doc
    assert "`send_max_tokens: false`" in doc
