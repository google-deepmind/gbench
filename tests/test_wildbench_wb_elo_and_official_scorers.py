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

"""Judged, vision and long-context suites: WildBench WB-Elo and headline, simple-evals and HLE
sampling, BEAM's pinned json_repair, MMMU-Pro's request, and requests without a reply in the
suites that can report leaderboard_comparable=True.

No model server or judge is contacted: requests, judges and downloads are replaced.
"""
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, beam_128k, chartqa, docvqa, healthbench,
                                        humanitys_last_exam, infographicvqa, loft_scifact,
                                        mmmu_pro, simpleqa, textvqa, wildbench)
from gbench.runners.eval_suites.vqa_common import lost_requests, lost_requests_reason

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
SUITES = REPO / "gbench" / "runners" / "eval_suites"


def _doc(name):
    return (DOCS / f"{name}.md").read_text(encoding="utf-8")


def _serve(replies, seen=None, completion_tokens=5):
    """A fake _send_single_request answering by the last user text (None = failed request)."""
    async def _send(**kw):
        if seen is not None:
            seen.append(kw)
        content = kw["messages"][-1]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        reply = replies[text]
        if reply is None:
            return base.Reply(text=None, tool_calls=None, error="server error")
        return base.Reply(text=reply, tool_calls=None, finish_reason="stop",
                          completion_tokens=completion_tokens)
    return _send


# --------------------------------------------------------------------------- #
# Reference: allenai/WildBench leaderboard/wb_elo.py @ d6b8dcaf377d (Apache-2.0), the two
# functions WB-Elo is built from, copied verbatim apart from the module globals they read
# (passed in here) and the dead use_regressed_as_init branch.
# --------------------------------------------------------------------------- #
def _upstream_get_all_votes(all_scores_by_id_model, all_outputs_by_id_model, margin=2,
                            tie_margin=1):
    votes = []
    covered_pairs_ids = set()
    for session_id, scores_by_model in all_scores_by_id_model.items():
        for model_1, score_1 in scores_by_model.items():
            for model_2, score_2 in scores_by_model.items():
                if model_1 == model_2:
                    continue
                vote_item = {}
                vote_item["session_id"] = session_id
                vote_item["model_1"] = model_1
                vote_item["model_2"] = model_2
                vote_item["score_1"] = score_1
                vote_item["score_2"] = score_2
                model_1_output = all_outputs_by_id_model[session_id][model_1]
                model_2_output = all_outputs_by_id_model[session_id][model_2]
                vote_item["length_1"] = len(model_1_output)
                vote_item["length_2"] = len(model_2_output)
                if len(model_1_output) == 0 or len(model_2_output) == 0:
                    continue
                if model_1_output.endswith("... (truncated)") or model_2_output.endswith("... (truncated)"):
                    continue
                if score_1 > score_2 and score_1 - score_2 >= margin:
                    vote_item["winner"] = model_1
                elif score_2 > score_1 and score_2 - score_1 >= margin:
                    vote_item["winner"] = model_2
                else:
                    if abs(score_1 - score_2) <= tie_margin:
                        vote_item["winner"] = "tie"
                    else:
                        continue
                model_str = "_".join(sorted([model_1, model_2]))
                uniq_vote_id = f"{session_id}_{model_str}"
                if uniq_vote_id in covered_pairs_ids:
                    continue
                votes.append(vote_item)
                covered_pairs_ids.add(uniq_vote_id)
    return votes


def _upstream_compute_single_round(votes, K, init_elos, dynamic, interval=10, length_margin=-1):
    elo = init_elos.copy() if init_elos is not None else {}
    sample_votes = random.sample(votes, len(votes))
    for vote in sample_votes:
        if vote["model_1"] not in elo:
            elo[vote["model_1"]] = 1000
        if vote["model_2"] not in elo:
            elo[vote["model_2"]] = 1000
    vote_update_cnt = defaultdict(int)
    for vote in sample_votes:
        model_1 = vote["model_1"]
        model_2 = vote["model_2"]
        if model_1 in init_elos and model_2 in init_elos:
            continue
        elo_1 = elo[model_1]
        elo_2 = elo[model_2]
        expected_1 = 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))
        expected_2 = 1 / (1 + 10 ** ((elo_1 - elo_2) / 400))
        if length_margin > 0:
            abs_len_diff = abs(vote["length_1"] - vote["length_2"])
            if abs_len_diff > length_margin:
                vote["winner"] = "tie"
        if vote["winner"] == model_1:
            score_1 = 1
            score_2 = 0
        elif vote["winner"] == model_2:
            score_1 = 0
            score_2 = 1
        else:
            score_1 = 0.5
            score_2 = 0.5
        if model_1 not in init_elos:
            elo[model_1] += K * (score_1 - expected_1)
        else:
            if dynamic:
                elo[model_1] += K * (score_1 - expected_1)
                if interval > 0 and vote_update_cnt[model_1] % interval == 0:
                    elo[model_1] = (elo[model_1] + init_elos[model_1]) / 2
        if model_2 not in init_elos:
            elo[model_2] += K * (score_2 - expected_2)
        else:
            if dynamic:
                elo[model_2] += K * (score_2 - expected_2)
                if interval > 0 and vote_update_cnt[model_2] % interval == 0:
                    elo[model_2] = (elo[model_2] + init_elos[model_2]) / 2
        vote_update_cnt[model_1] += 1
        vote_update_cnt[model_2] += 1
    return elo


def _random_pool(n_sessions=40, n_models=7, seed=3):
    rng = random.Random(seed)
    names = [f"m{i}" for i in range(n_models)]
    models = []
    for name in names:
        items = []
        for s in range(n_sessions):
            if rng.random() < 0.1:
                continue                                    # a dropped judge reply
            out = rng.choice(["", "x" * rng.randint(1, 3000), "y" * 10 + "... (truncated)"])
            if rng.random() < 0.8 and out == "":
                out = "z" * rng.randint(1, 3000)
            items.append((f"s{s}", str(rng.randint(1, 10)), out))
        models.append((name, items))
    return names, models


# --------------------------------------------------------------------------- #
# WB-Elo: votes and ratings reproduce wb_elo.py
# --------------------------------------------------------------------------- #
def test_wb_elo_votes_equal_upstream_get_all_votes():
    _names, models = _random_pool()
    scores, outputs = wildbench.wb_elo_scores(models)
    for margin, tie in ((3, 2), (2, 1), (2, 2)):
        ours = wildbench.wb_elo_votes(scores, outputs, margin, tie)
        theirs = _upstream_get_all_votes(scores, outputs, margin, tie)
        assert ours == theirs
        assert ours, "the random pool must produce votes"


def test_wb_elo_scores_int_the_score_and_strip_the_reply():
    scores, outputs = wildbench.wb_elo_scores([("a", [("s1", "7", "  hi  ")]),
                                               ("b", [("s1", 8.0, "x"), ("s2", "7.5", "y")])])
    assert scores == {"s1": {"a": 7, "b": 8}, "s2": {"b": 7}}
    assert outputs["s1"]["a"] == "hi"


@pytest.mark.parametrize("length_margins", [[-1], [1500], [-1, 1500], [1000, -1]])
def test_wb_elo_ratings_equal_upstream_rounds_bitwise(length_margins):
    names, models = _random_pool()
    scores, outputs = wildbench.wb_elo_scores(models)
    votes = wildbench.wb_elo_votes(scores, outputs, 3, 2)
    init = {"m0": 1200.0, "m1": 1100.0, "m2": 1150.0, "unused": 1000.0}
    rounds = 6
    ours = wildbench.wb_elo_ratings(votes, init, K=4, num_rounds=rounds, dynamic=True,
                                    interval=5, length_margins=length_margins, seed=42)
    for lm in length_margins:
        random.seed(42)                                     # compute_wb_elo, one per run
        runs = [_upstream_compute_single_round([dict(v) for v in votes], 4, init, True, 5, lm)
                for _ in range(rounds)]
        for model, stat in ours[lm].items():
            values = [r[model] for r in runs]
            assert stat["avg"] == sum(values) / len(values), (lm, model)
            assert stat["init_elo"] == init.get(model, "-")
        assert set(ours[lm]) == set(runs[0])
        avgs = [s["avg"] for s in ours[lm].values()]
        assert avgs == sorted(avgs, reverse=True)
    assert ours[length_margins[0]]["unused"]["avg"] == 1000.0     # no vote: keeps its Arena Elo


def test_wb_elo_tables_are_the_published_pool_and_settings():
    assert len(wildbench.WB_ELO_POOL) == 63
    assert len(wildbench.WB_ELO_INIT) == 43
    names = {wildbench.wb_elo_name(c) for c in wildbench.WB_ELO_POOL}
    assert len(names & set(wildbench.WB_ELO_INIT)) == 39
    assert set(wildbench.WB_ELO_INIT) - names == {"Llama-2-13b-chat-hf", "Mistral-7B-Instruct-v0.1",
                                                  "Yi-34B-Chat", "zephyr-7b-beta"}
    assert wildbench.wb_elo_name("dbrx-instruct@together") == "dbrx-instruct"
    assert wildbench.WB_ELO_INIT["gpt-4o-2024-05-13"] == 1280.0
    assert wildbench.WB_ELO_CONFIG == {"K": 4, "num_rounds": 128, "margin": 3, "tie_margin": 2,
                                       "dynamic": True, "interval": 100, "seed": 42}
    assert wildbench.WB_ELO_LENGTH_MARGINS == {"WB_Elo": -1, "WB_Elo_LC": 1500}
    assert wildbench.WB_ELO_TARGET not in names and wildbench.WB_ELO_TARGET not in wildbench.WB_ELO_INIT
    # the three WB-Reward references are pool models with the same pinned file
    for config, sha in wildbench.REFERENCE_MODELS.values():
        assert wildbench.WB_ELO_POOL[config] == sha
    assert all(len(sha) == 64 for sha in wildbench.WB_ELO_POOL.values())


def test_pool_items_skip_replies_that_cannot_vote():
    samples = [([{"role": "user", "content": "q"}], ["c"], {"session_id": "s1"})]
    pool = {"a": {"s1": "fine"}, "b": {"s1": "   "}, "c": {"s1": "cut... (truncated)"}}
    items = wildbench.wb_elo_pool_items(samples, pool)
    assert [(i["config"], i["output_chars"]) for i in items] == [("a", len("fine"))]
    prompt = wildbench.build_wb_judge_prompt(samples[0][0], ["c"], "fine")
    assert items[0]["prompt_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    with pytest.raises(RuntimeError, match="no reply"):
        wildbench.wb_elo_pool_items(samples, {"a": {}})


def test_pool_cache_key_follows_the_judge_cascade(monkeypatch):
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-a,gemini-b")
    first = wildbench.wb_elo_cache_dir()
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-c")
    second = wildbench.wb_elo_cache_dir()
    assert first != second
    assert first.startswith(os.path.join(os.environ["GBENCH_CACHE_DIR"], "wildbench",
                                         "wb_elo_pool"))


def test_missing_pool_file_is_infra_required(monkeypatch):
    import huggingface_hub

    def _fail(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        next(wildbench._load_wb_elo_pool_outputs())      # the models are read one at a time


# --------------------------------------------------------------------------- #
# WildBench end to end: WB-Elo columns, pool cache, headline
# --------------------------------------------------------------------------- #
_POOL = {"gpt-4o-2024-05-13": "strong reply", "reka-edge": "weak reply"}


def _wb_run(monkeypatch, pool_verdict, replies=None, calls=None, shard=None):
    if shard:
        monkeypatch.setenv("GBENCH_SHARD", shard)
    else:
        monkeypatch.delenv("GBENCH_SHARD", raising=False)
    replies = replies or {"q1": "tested reply one", "q2": "tested reply two"}
    samples = [([{"role": "user", "content": q}], ["c"],
                {"category": "Math", "session_id": q, "wb_pairwise_target_is_a": True,
                 "wb_task_groups": ["Math & Data Analysis"]}) for q in replies]
    refs = {p: {q: "reference" for q in replies} for p in wildbench.REFERENCE_MODELS}
    pool = {config: {q: f"{text} {q}" for q in replies} for config, text in _POOL.items()}
    calls = calls if calls is not None else []

    async def _judge(prompt, config=None):
        if "<|begin_of_response_A|>" in prompt:
            return '{"choice": "A=B"}', "gemini-p"
        out = wildbench.score_output_from_prompt(prompt)
        calls.append(out)
        if out.startswith("tested"):
            return '{"score": "6"}', "gemini-s"
        return pool_verdict(out)

    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies)), \
            mock.patch.object(wildbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(wildbench, "_load_wildbench_samples", lambda **k: samples), \
            mock.patch.object(wildbench, "_load_reference_outputs", lambda: refs), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_outputs", lambda: pool), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_replies", lambda config: pool[config]), \
            mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        res = wildbench.run_wildbench("m", "http://x/v1", 1)
    return res, calls


def _pool_scores(out):
    return ('{"score": "%s"}' % ("10" if out.startswith("strong") else "2"), "gemini-s")


def test_run_reports_wb_elo_and_uses_task_macro_as_headline(monkeypatch):
    res, calls = _wb_run(monkeypatch, _pool_scores)
    assert res["accuracy"] == res["wb_score_task_macro"] == pytest.approx(10 / 4.75 * 2 * 1, abs=1e-3)
    assert res["wb_score"] == pytest.approx((6 - 5) * 20)
    # the tested model (6) loses to gpt-4o (10) by 4 and beats reka-edge (2) by 4 on both tasks
    detail = res["wb_elo_detail"]
    assert detail["target_votes"] == 4 and detail["votes"] == 6
    assert detail["pool_replies_graded"] == 4 and detail["pool_verdicts_from_cache"] == 0
    assert detail["config"] == wildbench.WB_ELO_CONFIG
    assert detail["pool_judge_models"] == {"gemini-s": 4}
    assert res["wb_elo"] == detail["columns"]["WB_Elo"]["avg"]
    assert res["wb_elo_lc"] == detail["columns"]["WB_Elo_LC"]["avg"]
    # gpt-4o has an Arena Elo (1280); the tested model and reka-edge start from 1000
    table = res["wb_elo_tables"]["WB_Elo"]
    assert table["gpt-4o-2024-05-13"] > res["wb_elo"] > table["reka-edge"]
    assert set(table) >= set(wildbench.WB_ELO_INIT) | {wildbench.WB_ELO_TARGET, "reka-edge"}
    assert "WB-Elo re-grades the 63 pool models" in res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    # the pool verdicts are cached under $GBENCH_CACHE_DIR; a second run asks only for the
    # tested model's replies
    cached = Path(detail["pool_cache_dir"])
    assert cached.is_relative_to(Path(os.environ["GBENCH_CACHE_DIR"]))
    assert sorted(p.name for p in cached.iterdir()) == ["gpt-4o-2024-05-13.jsonl", "reka-edge.jsonl"]
    res2, calls2 = _wb_run(monkeypatch, _pool_scores)
    assert all(c.startswith("tested") for c in calls2) and len(calls2) == 2
    assert res2["wb_elo_detail"]["pool_verdicts_from_cache"] == 4
    assert res2["wb_elo"] == res["wb_elo"] and res2["wb_elo_lc"] == res["wb_elo_lc"]


def test_pool_is_graded_on_the_shard_the_run_covers(monkeypatch):
    res, calls = _wb_run(monkeypatch, _pool_scores, shard="1/2")
    assert res["wb_elo_detail"]["pool_replies_graded"] == 2          # 2 pool models x 1 task
    assert len(res["sample_traces"]) == 1 and res["wb_elo_detail"]["target_votes"] == 2
    assert "GBENCH_SHARD 1/2 shard (the WB-Elo pool is graded on the same tasks)" in \
        res["leaderboard_comparable_reason"]


def test_pool_outages_are_not_cached_and_are_reported(monkeypatch):
    res, _ = _wb_run(monkeypatch, lambda out: (None, "JUDGE_OUTAGE"))
    detail = res["wb_elo_detail"]
    assert detail["pool_judge_outages"] == 4 and detail["target_votes"] == 0
    assert res["wb_elo"] is None and res["wb_elo_lc"] is None
    assert res["status"] == "completed_with_errors"
    assert "4 WB-Elo pool repl(ies) not graded (judge outage)" in res["leaderboard_comparable_reason"]
    assert not any(Path(detail["pool_cache_dir"]).glob("*.jsonl")) or all(
        p.read_text() == "" for p in Path(detail["pool_cache_dir"]).glob("*.jsonl"))


def test_dropped_pool_verdicts_cast_no_vote(monkeypatch):
    res, _ = _wb_run(monkeypatch, lambda out: ("not json", "gemini-s"))
    detail = res["wb_elo_detail"]
    assert detail["pool_judge_failures"] == 4 and detail["target_votes"] == 0
    assert res["wb_elo"] is None


# --------------------------------------------------------------------------- #
# WB-Score: replies ending in "... (truncated)" count in WB_score only
# --------------------------------------------------------------------------- #
def test_task_macro_and_length_skip_truncated_replies_like_create_tables():
    def _t(raw, text, groups):
        return {"wb_score_raw": raw, "response_text": text,
                "extra_payload": {"wb_task_groups": groups}}
    traces = [_t(9, "good", ["Math & Data Analysis"]),
              _t(1, "long reply... (truncated)", ["Math & Data Analysis"]),
              _t(3, "", ["Creative Tasks"])]
    macro, groups = wildbench.wb_task_macro(traces)
    assert groups == {"Math & Data Analysis": 80.0}
    assert macro == pytest.approx(80.0 / 4.75)
    assert wildbench.wb_length(traces) == 4.0
    # a judged trace records whether its score-prompt reply is truncated
    assert wildbench._output_truncated({"wb_output_truncated": True, "response_text": "x"})


# --------------------------------------------------------------------------- #
# simpleqa / healthbench: simple-evals' chat sampler (system message + 0.5) in both modes
# --------------------------------------------------------------------------- #
def _simpleqa_run(thinking, seen, graded):
    samples = [([{"role": "user", "content": "Who?"}], "Ada", {"category": "x"})]

    async def _judge(prompt, config=None):
        graded.append(prompt)
        return "A", "gemini-x"

    with mock.patch.object(base, "_send_single_request", side_effect=_serve({"Who?": "Ada"}, seen)), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples", lambda limit=None: samples), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed",
                              _simpleqa_detailed(_judge)):
        return simpleqa.run_simpleqa("m", "http://x/v1", 1, enable_thinking=thinking)


def test_simpleqa_samples_as_simple_evals_chat_and_reasoning_rows():
    seen, graded = [], []
    res = _simpleqa_run(False, seen, graded)
    assert seen[0]["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert seen[0]["temperature"] == 0.0                # gbench's no-think rule, not the 0.5
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "Question: Who?\n" in graded[0]                      # the user turn, not the system one
    assert res["accuracy"] == 100.0
    # A served model takes the chat sampler with --thinking too (simple-evals' o-series
    # samplers are keyed on OpenAI model names), sampled at 1.0, which the result names.
    seen, graded = [], []
    res = _simpleqa_run(True, seen, graded)
    assert [m["role"] for m in seen[0]["messages"]] == ["system", "user"]
    assert seen[0]["temperature"] == 1.0
    assert res["official_sampling"]["send_temperature"] is True
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "Question: Who?\n" in graded[0]
    assert ("temperature 1.0 (simple-evals' chat sampler samples at 0.5)"
            in res["leaderboard_comparable_reason"])


def test_simpleqa_names_an_operator_temperature(monkeypatch):
    monkeypatch.setenv("GBENCH_SIMPLEQA_TEMPERATURE", "0.0")
    res = _simpleqa_run(False, [], [])
    assert "temperature 0.0 (simple-evals' chat sampler samples at 0.5)" in \
        res["leaderboard_comparable_reason"]


def test_healthbench_grader_sees_the_sampler_system_turn():
    rubric = json.dumps([{"criterion": "Says hi", "points": 5, "tags": []}])
    samples = [([{"role": "user", "content": "hello"}], rubric,
                {"category": "theme:x", "example_tags": ["theme:x"]})]
    # The chat sampler in both modes: the system turn is sent, and graded, with --thinking too.
    for thinking, system in ((False, True), (True, True)):
        seen, graded = [], []

        async def _judge(prompt, config=None):
            graded.append(prompt)
            return '{"explanation": "ok", "criteria_met": true}', "gemini-x"

        with mock.patch.object(base, "_send_single_request",
                               side_effect=_serve({"hello": "hi"}, seen)), \
                mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
                mock.patch.object(healthbench, "_load_healthbench_samples",
                                  lambda limit=None: samples), \
                _patch_detailed(healthbench, _judge):
            res = healthbench.run_healthbench("m", "http://x/v1", 1, enable_thinking=thinking)
        assert (seen[0]["messages"][0]["role"] == "system") is system
        assert ("system: You are a helpful assistant.\n\nuser: hello" in graded[0]) is system
        assert seen[0]["temperature"] == (1.0 if thinking else 0.0)
        assert res["accuracy"] == 100.0


def test_simple_evals_docs_state_the_sampler():
    for name in ("simpleqa", "healthbench"):
        doc = _doc(name)
        assert doc.isascii()
        assert "You are a helpful assistant." in doc and "0.5" in doc
        assert "`official_sampling`" in doc
        assert "(0.0 greedy, 1.0 with `--eval-thinking`)" not in doc


# --------------------------------------------------------------------------- #
# HLE: 0 without thinking, no temperature with it (officially; gbench sends 1.0 / 0.0)
# --------------------------------------------------------------------------- #
def test_hle_official_sampling_per_mode():
    chat, reasoning = (humanitys_last_exam.official_sampling(False),
                       humanitys_last_exam.official_sampling(True))
    assert chat.temperature == 0.0 and chat.send_temperature is True
    assert reasoning.send_temperature is False
    assert "136e12e" in reasoning.reference and "22ed3074b1e7" in chat.reference
    src = (SUITES / "humanitys_last_exam.py").read_text(encoding="utf-8")
    assert "official_sampling=official_sampling(enable_thinking)" in src
    doc = _doc("humanitys_last_exam")
    assert "no temperature" in doc and "136e12e" in doc
    assert "**temperature 1.0 with `--eval-thinking` and 0.0 without**" in doc


# --------------------------------------------------------------------------- #
# beam_128k: json_repair 0.44.1, vendored
# --------------------------------------------------------------------------- #
def test_vendored_json_repair_is_the_pinned_wheel_byte_for_byte():
    vendor = SUITES.parents[1] / "third_party" / "_json_repair_vendor"
    for name, sha in beam_128k.JSON_REPAIR_VENDOR_SHA256.items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == sha, name
    assert sorted(p.name for p in vendor.iterdir() if p.suffix == ".py") == sorted(
        beam_128k.JSON_REPAIR_VENDOR_SHA256)
    lic = (vendor / "LICENSE").read_text(encoding="utf-8")
    assert lic.startswith("MIT License") and "Stefano Baccianella" in lic
    assert hashlib.sha256((vendor / "LICENSE").read_bytes()).hexdigest() == \
        "c2b8d0a3c32136b34289c5ed31edcc1e64be7f1f00990935b9ef004308a2155f"


def test_beam_repairs_with_the_vendored_release(monkeypatch):
    monkeypatch.setitem(sys.modules, "json_repair", None)       # the installed one is not used
    beam_128k._require_json_repair()
    assert beam_128k.parse_judge_reply('{"score": 1.0, "reason": "ok"')["score"] == 1.0
    # 0.44.1 repairs text without JSON to '""' (later releases return ''), which loads as ""
    assert beam_128k.parse_judge_reply("no json here") == ""
    assert beam_128k._json_repair_version() == "0.44.1"
    src = (SUITES / "beam_128k.py").read_text(encoding="utf-8")
    assert "from json_repair import" not in src and "import json_repair" not in src
    doc = _doc("beam_128k")
    assert "_json_repair_vendor" in doc and "51d82532" in doc
    assert "gbench uses the installed release" not in doc


# --------------------------------------------------------------------------- #
# requests without a reply make a run not comparable
# --------------------------------------------------------------------------- #
def test_lost_requests_reason_counts_each_status():
    traces = [{"status": "FAILED"}, {"status": "TIMEOUT"}, {"status": "OVER_CONTEXT"},
              {"status": "OK"}, {"status": "EMPTY_RESPONSE"}, {"status": "FAILED"}]
    assert lost_requests(traces) == 4
    assert lost_requests_reason(traces) == ("4 request(s) got no reply (2 failed, 1 timed out, "
                                            "1 over the context window) (harness errors, left "
                                            "out of the score)")
    assert lost_requests_reason([{"status": "OK"}, {"status": "EMPTY_RESPONSE"}]) is None


@pytest.mark.parametrize("module,run,loader,gold", [
    (chartqa, "run_chartqa", "_load_chartqa_samples", ["14"]),
    (textvqa, "run_textvqa", "_load_textvqa_samples", ["stop"] * 10),
    (docvqa, "run_docvqa", "_load_docvqa_samples", ["no."]),
    (infographicvqa, "run_infographicvqa", "_load_infographicvqa_samples", ["12"]),
])
def test_vqa_runs_with_a_lost_request_are_not_comparable(module, run, loader, gold, monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = [([{"role": "user", "content": [{"type": "text", "text": q}]}], gold,
                {"category": "x"}) for q in ("q1", "q2")]
    replies = {"q1": gold[0], "q2": None}

    async def _send(**kw):
        text = kw["messages"][-1]["content"][0]["text"]
        if replies[text] is None:
            return base.Reply(text=None, tool_calls=None, error="server error")
        return base.Reply(text=replies[text], tool_calls=None, finish_reason="stop",
                          completion_tokens=2)

    with mock.patch.object(base, "_send_single_request", side_effect=_send), \
            mock.patch.object(module, loader, lambda **k: samples):
        res = getattr(module, run)("m", "http://x/v1", 1)
    assert res["leaderboard_comparable"] is False
    assert "1 request(s) got no reply (1 failed)" in res["leaderboard_comparable_reason"]
    replies["q2"] = gold[0]
    with mock.patch.object(base, "_send_single_request", side_effect=_send), \
            mock.patch.object(module, loader, lambda **k: samples):
        res = getattr(module, run)("m", "http://x/v1", 1)
    assert res["leaderboard_comparable"] is True


@pytest.mark.parametrize("name", ["coco_caption", "loft_scifact", "ruler", "omnidocbench",
                                  "medxpertqa", "mmmu_pro", "chartqa", "textvqa", "docvqa",
                                  "infographicvqa"])
def test_every_conditionally_comparable_suite_checks_lost_requests(name):
    src = (SUITES / f"{name}.py").read_text(encoding="utf-8")
    if name == "omnidocbench":
        # A page with no reply is left out of the evaluator's input (it would score it as an
        # empty transcription) and named by the suite itself
        # (tests/test_code_exec_endpoint_failures_and_aider_memory.py).
        assert 'trace.get("harness_error")' in src and "got no reply from the served" in src
    else:
        assert "lost_requests_reason(" in src, name
    doc = _doc(name)
    assert doc.isascii(), name
    assert "no reply" in doc or "got a reply" in doc, name


def test_loft_run_with_a_lost_request_is_not_comparable(monkeypatch):
    samples = [([{"role": "user", "content": "q1"}], ["d1"], {"category": "x"}),
               ([{"role": "user", "content": "q2"}], ["d2"], {"category": "x"})]
    replies = {"q1": "Final Answer: ['d1']", "q2": None}
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies)), \
            mock.patch.object(loft_scifact, "_load_loft_scifact_samples", lambda **k: samples):
        res = loft_scifact.run_loft_scifact("m", "http://x/v1", 1)
    assert res["leaderboard_comparable"] is False
    assert "got no reply" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# mmmu_pro: infer_gpt.py's system turn, no temperature, 4,096-token cap counted
# --------------------------------------------------------------------------- #
def test_mmmu_pro_request_follows_infer_gpt(monkeypatch):
    assert mmmu_pro.SYSTEM_MESSAGE == ("You are ChatGPT, a large language model trained by OpenAI, "
                                       "based on the GPT-4 architecture.")
    assert mmmu_pro.OFFICIAL_SAMPLING.send_temperature is False
    assert "268471d0" in mmmu_pro.OFFICIAL_SAMPLING.reference
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        traces = [{"category": meta["category"], "is_correct": True,
                   "completion_tokens": 5000 if i == 0 else 10,
                   "status": "FAILED" if i == 1 else "OK"}
                  for i, (_m, _g, meta) in enumerate(kw["samples"])]
        return {"sample_traces": traces, "accuracy": 100.0, "temperature": 0.3,
                "official_sampling": {"temperature_matches_official": False}}

    samples = [([{"role": "system", "content": mmmu_pro.SYSTEM_MESSAGE},
                 {"role": "user", "content": [{"type": "text", "text": "t"}]}],
                {"answer": "A", "options": ["a", "b"], "id": "i"},
                {"category": cat}) for cat in ("standard_10/cot", "standard_10/direct",
                                               "vision/cot", "vision/direct")]
    monkeypatch.setattr(mmmu_pro, "run_eval_suite", fake_run)
    monkeypatch.setattr(mmmu_pro, "_load_mmmu_pro_samples", lambda *a, **k: samples)
    res = mmmu_pro.run_mmmu_pro("m", "http://x/v1", 1)
    assert seen["official_sampling"] is mmmu_pro.OFFICIAL_SAMPLING and "temperature" not in seen
    # infer_gpt.py's max_tokens=4096 is the official cap, sent by base's token-limit policy
    assert mmmu_pro.OFFICIAL_SAMPLING.max_tokens == 4096
    assert "replies_over_upstream_cap" not in res
    reason = res["leaderboard_comparable_reason"]
    assert "temperature 0.3 (infer_gpt.py sends none)" in reason
    assert "4096-token cap" not in reason
    assert "1 request(s) got no reply (1 failed)" in reason
    doc = _doc("mmmu_pro")
    assert mmmu_pro.SYSTEM_MESSAGE.replace(" GPT-4 architecture.", "") .split(",")[0] in doc
    assert "`official_sampling`" in doc and "gbench's default" not in doc


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_wildbench_doc_describes_wb_elo_and_the_headline():
    doc = _doc("wildbench")
    assert doc.isascii()
    for s in ("`wb_elo`", "`wb_elo_lc`", "`wb_elo_detail`", "`wb_elo_tables`", "4b94d1d8",
              "margin=3, tie_margin=2", "interval=100", "128 rounds", "1,500 characters",
              "`gbench:model-under-test`", "$GBENCH_CACHE_DIR/wildbench/wb_elo_pool",
              "`accuracy` (headline) = `wb_score_task_macro`", "status: EMPTY_RESPONSE",
              "64,512", "sorts its main table", "--max-output-tokens"):
        assert s in doc, s
    assert "Not computed: WB-Elo" not in doc
    assert "Judged traces end with `status: OK`" not in doc
    src = (SUITES / "wildbench.py").read_text(encoding="utf-8")
    assert "WB-Elo, the leaderboard's third column, is not computed" not in src


def test_cache_docs_name_gbench_cache_dir():
    assert "$GBENCH_CACHE_DIR/ruler/" in _doc("ruler")
    assert "$GBENCH_CACHE_DIR/loft/" in _doc("loft_scifact")


def test_chartqa_doc_names_the_lmms_eval_dataset_copy():
    doc = _doc("chartqa")
    assert "lmms-lab-encoder/ChartQA" in doc and "pixel-identical" in doc


# --------------------------------------------------------------------------- #
# screenspot_pointing: the official ScreenSpot-v2 release, cells, sampling
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import bundled_detection, i18n_translate, screenspot_pointing  # noqa: E402


def _fake_screenspot(tmp_path, monkeypatch, rows_by_platform, images):
    import zipfile
    files = {}
    for platform, (name, _sha) in screenspot_pointing.SPLITS.items():
        files[name] = tmp_path / name
        files[name].write_text(json.dumps(rows_by_platform.get(platform, [])))
    files[screenspot_pointing.IMAGE_ZIP[0]] = tmp_path / "img.zip"
    with zipfile.ZipFile(files[screenspot_pointing.IMAGE_ZIP[0]], "w") as zf:
        for name, data in images.items():
            zf.writestr(f"screenspotv2_image/{name}", data)
    monkeypatch.setattr(screenspot_pointing, "_fetch", lambda name, sha: str(files[name]))
    monkeypatch.setattr(screenspot_pointing, "SPLIT_SIZES",
                        {p: len(rows_by_platform.get(p, [])) for p in screenspot_pointing.SPLITS})


def _png(width, height):
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_screenspot_pointing_pins_the_official_release():
    assert screenspot_pointing.DATASET == "OS-Copilot/ScreenSpot-v2"
    assert screenspot_pointing.DATASET_REVISION == "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c"
    assert sum(screenspot_pointing.SPLIT_SIZES.values()) == screenspot_pointing.N_TARGETS == 1272
    assert screenspot_pointing.IMAGE_ZIP[1].startswith("750555b8")
    assert screenspot_pointing.CELLS == ("mobile/text", "mobile/icon", "desktop/text",
                                         "desktop/icon", "web/text", "web/icon")
    assert screenspot_pointing.normalized_box([10, 20, 30, 40], 100, 200) == [0.1, 0.1, 0.4, 0.3]
    assert screenspot_pointing.OFFICIAL_SAMPLING.send_temperature is False


def test_screenspot_pointing_loads_the_release_files(tmp_path, monkeypatch):
    png = _png(200, 100)
    rows = {"mobile": [{"img_filename": "m.png", "bbox": [20, 10, 40, 20], "instruction": "tap",
                        "data_type": "icon", "data_source": "ios"}],
            "web": [{"img_filename": "w.png", "bbox": [0, 0, 200, 100], "instruction": "open",
                     "data_type": "text", "data_source": "shop"}]}
    _fake_screenspot(tmp_path, monkeypatch, rows, {"m.png": png, "w.png": png})
    samples = screenspot_pointing._load_screenspot_pointing_samples()
    assert [m["category"] for _, _, m in samples] == ["mobile/icon", "web/text"]
    msgs, gold, meta = samples[0]
    assert gold["bbox"] == [0.1, 0.1, 0.3, 0.3] and gold["bbox_normalized"] is True
    assert msgs[0]["content"][0]["image_url"]["url"].endswith(
        __import__("base64").b64encode(png).decode())
    assert msgs[0]["content"][1]["text"].startswith("Target Element: tap\n")
    # the centre of the box is inside it; a point outside is not; an edge past 1.0 still scales
    assert screenspot_pointing._eval_screenspot_pointing("Point: (200, 200)", gold) is True
    assert screenspot_pointing._eval_screenspot_pointing("Point: (500, 500)", gold) is False
    edge = {"bbox": [0.5, 0.5, 1.0004, 0.9], "bbox_normalized": True}
    assert screenspot_pointing._eval_screenspot_pointing("Point: (800, 700)", edge) is True
    _fake_screenspot(tmp_path, monkeypatch, rows, {"m.png": png})
    with pytest.raises(RuntimeError, match="has no image 'w.png'"):
        screenspot_pointing._load_screenspot_pointing_samples()


def test_screenspot_pointing_run_reports_cells_and_sends_the_rule_temperature(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = [([{"role": "user", "content": [{"type": "text", "text": q}]}],
                {"bbox": [0.0, 0.0, 0.5, 0.5], "bbox_normalized": True}, {"category": cell})
               for q, cell in (("a", "mobile/text"), ("b", "web/icon"))]
    replies = {"a": "Point: (100, 100)", "b": None}
    sent = []

    async def _send(**kw):
        sent.append(kw)
        text = kw["messages"][-1]["content"][0]["text"]
        if replies[text] is None:
            return base.Reply(text=None, tool_calls=None, error="server error")
        return base.Reply(text=replies[text], tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", side_effect=_send), \
            mock.patch.object(screenspot_pointing, "_load_screenspot_pointing_samples",
                              lambda **k: samples):
        res = screenspot_pointing.run_screenspot_pointing("m", "http://x/v1", 1)
    assert all(k["temperature"] == 0.0 for k in sent)     # gbench's no-think rule
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["screenspot_v2_cells"]["mobile/text"] == 100.0
    # its only request got no reply: a harness error, so the cell has no scored target
    assert res["screenspot_v2_cells"]["web/icon"] is None
    assert res["screenspot_v2_cells"]["desktop/text"] is None
    reason = res["leaderboard_comparable_reason"]
    assert "gbench pointing prompt" in reason and "1 request(s) got no reply" in reason
    assert res["leaderboard_comparable"] is False
    doc = _doc("screenspot_pointing")
    for s in ("OS-Copilot/ScreenSpot-v2", "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c",
              "`screenspot_v2_cells`", "`official_sampling`", "HongxinLi/ScreenSpot_v2"):
        assert s in doc, s
    assert doc.isascii()


# --------------------------------------------------------------------------- #
# bundled_detection: official COCO files, pycocotools COCOeval
# --------------------------------------------------------------------------- #
def test_bundled_detection_pins_the_official_coco_files():
    assert bundled_detection.IMAGES_ZIP[0] == "http://images.cocodataset.org/zips/val2017.zip"
    assert bundled_detection.ANNOTATIONS_ZIP[0].endswith("annotations_trainval2017.zip")
    assert bundled_detection.INSTANCES_MEMBER[0] == "annotations/instances_val2017.json"
    assert (bundled_detection.N_IMAGES, bundled_detection.N_ANNOTATIONS) == (5000, 36781)
    assert len(bundled_detection.CATEGORY_ID) == 80 and bundled_detection.CATEGORY_ID["toothbrush"] == 90
    assert bundled_detection.COCO_STATS[0] == "AP" and len(bundled_detection.COCO_STATS) == 12


def _det_trace(image_id, objects, dets, width=640, height=480):
    objs = []
    for i, (label, x, y, w, h, crowd) in enumerate(objects):
        objs.append({"id": i, "category_id": bundled_detection.CATEGORY_ID[label], "label": label,
                     "bbox": [x, y, w, h], "area": w * h, "iscrowd": crowd,
                     "box_2d": [y / height * 1000, x / width * 1000, (y + h) / height * 1000,
                                (x + w) / width * 1000]})
    return {"gold_answer": {"image_id": image_id, "width": width, "height": height,
                            "objects": objs}, "response_text": json.dumps(dets)}


def test_bundled_detection_scores_with_cocoeval():
    pytest.importorskip("pycocotools")
    # one exact cat, one missed dog, a crowd region that must not count as a miss, and a
    # detection with an unknown label and one with a degenerate box
    exact = [0.0, 0.0, 500.0, 500.0]                      # ymin, xmin, ymax, xmax of the cat
    t1 = _det_trace(7, [("cat", 0, 0, 320, 240, 0), ("person", 400, 300, 100, 100, 1)],
                    [{"label": "cat", "box_2d": exact, "confidence": 0.9},
                     {"label": "unicorn", "box_2d": exact},
                     {"label": "dog", "box_2d": [10, 10, 10, 50]}])
    t2 = _det_trace(3, [("dog", 0, 0, 100, 100, 0)], [])
    res = bundled_detection._compute_map([t1, t2])
    assert res["unknown_labels"] == 1 and res["invalid_boxes"] == 1 and res["detections"] == 1
    # cat AP = 1; dog AP = 0 (no detection); person has only a crowd region -> no AP
    assert res["map"] == pytest.approx(0.5)
    assert res["stats"]["AP50"] == pytest.approx(0.5)
    assert bundled_detection._eval_bundled_detection(t1["response_text"], t1["gold_answer"]) is True


def test_bundled_detection_matches_pycocotools_on_the_same_ground_truth():
    pytest.importorskip("pycocotools")
    import contextlib
    import io as _io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    rng = random.Random(5)
    traces = []
    for image_id in (42, 7, 19, 3):
        objects, dets = [], []
        for _ in range(rng.randint(1, 6)):
            label = rng.choice(["cat", "dog", "car", "person"])
            x, y = rng.uniform(0, 400), rng.uniform(0, 300)
            w, h = rng.uniform(5, 200), rng.uniform(5, 150)
            objects.append((label, x, y, w, h, int(rng.random() < 0.1)))
            if rng.random() < 0.8:
                j = [v + rng.uniform(-20, 20) for v in (y / 480 * 1000, x / 640 * 1000,
                                                        (y + h) / 480 * 1000, (x + w) / 640 * 1000)]
                dets.append({"label": label if rng.random() > 0.2 else "car", "box_2d": j,
                             "confidence": round(rng.random(), 1)})
        traces.append(_det_trace(image_id, objects, dets))
    ours = bundled_detection._compute_map(traces)
    gt_dataset = {"images": [], "annotations": [], "categories": [
        {"id": i, "name": n} for n, i in bundled_detection.CATEGORY_ID.items()]}
    results = []
    for t in traces:
        g = t["gold_answer"]
        gt_dataset["images"].append({"id": g["image_id"], "width": 640, "height": 480})
        for o in g["objects"]:
            gt_dataset["annotations"].append({"id": len(gt_dataset["annotations"]) + 1,
                                              "image_id": g["image_id"], **{k: o[k] for k in (
                                                  "category_id", "bbox", "area", "iscrowd")}})
        for d in json.loads(t["response_text"]):
            ymin, xmin, ymax, xmax = d["box_2d"]
            w, h = (xmax - xmin) / 1000 * 640, (ymax - ymin) / 1000 * 480
            if w > 0 and h > 0:
                results.append({"image_id": g["image_id"],
                                "category_id": bundled_detection.CATEGORY_ID[d["label"]],
                                "bbox": [xmin / 1000 * 640, ymin / 1000 * 480, w, h],
                                "score": d["confidence"]})
    with contextlib.redirect_stdout(_io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dataset
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(results), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    for ours_v, theirs in zip(ours["stats"].values(), ev.stats):
        assert (ours_v is None and theirs < 0) or ours_v == float(theirs)


def test_bundled_detection_run_reports_coco_stats(monkeypatch):
    pytest.importorskip("pycocotools")
    trace = _det_trace(1, [("cat", 0, 0, 320, 240, 0)],
                       [{"label": "cat", "box_2d": [0, 0, 500, 500], "confidence": 1.0}])
    monkeypatch.setattr(bundled_detection, "_load_bundled_detection_samples", lambda limit=None: [])
    monkeypatch.setattr(bundled_detection, "run_eval_suite",
                        lambda **kw: {"accuracy": 100.0, "sample_traces": [trace]})
    res = bundled_detection.run_bundled_detection("m", "http://x/v1", 1)
    assert res["accuracy"] == res["map_50_95"] == res["coco_stats"]["AP"] == 100.0
    assert set(res["coco_stats"]) == set(bundled_detection.COCO_STATS)
    assert "gbench's prompt" in res["leaderboard_comparable_reason"]
    assert "on 1 of the 5,000 val2017 images" in res["leaderboard_comparable_reason"]
    doc = _doc("bundled_detection")
    for s in ("COCOeval", "instances_val2017.json", "`coco_stats`", "4f7e2ccb", "113a836d",
              "$GBENCH_CACHE_DIR/coco/", "5,000"):
        assert s in doc, s
    assert "detection-datasets/coco" not in doc and doc.isascii()


def test_bundled_detection_requires_pycocotools(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycocotools", None)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", None)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        bundled_detection._require_pycocotools()


# --------------------------------------------------------------------------- #
# i18n_translate: WMT19 newstest2019, sacreBLEU
# --------------------------------------------------------------------------- #
def test_i18n_translate_is_wmt19_with_every_direction():
    assert len(i18n_translate.DIRECTIONS) == 19 and i18n_translate.N_SEGMENTS == 31387
    assert sum(i18n_translate.DIRECTIONS.values()) == 31387
    assert i18n_translate.TEST_SET == "wmt19" and i18n_translate.SACREBLEU_RELEASE == "2.6.0"
    assert set(code for d in i18n_translate.DIRECTIONS for code in d.split("-")) == set(
        i18n_translate.LANGUAGE_NAMES)


def test_i18n_translate_scores_each_direction_with_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")
    refs = {"de-en": ["The cat sat on the mat.", "\"Yes,\" he said."],
            "en-zh": ["\u6211\u7231\u5317\u4eac\u5929\u5b89\u95e8\u3002", "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002"]}
    hyps = {"de-en": ["The cat sat on a mat.", "\"Yes,\" he said."],
            "en-zh": ["\u6211\u7231\u5317\u4eac\u3002", None]}
    traces = [{"category": d, "extra_payload": {"category": d, "segment": i},
               "gold_answer": r, "response_text": hyps[d][i]}
              for d in refs for i, r in enumerate(refs[d])][::-1]      # any order
    out = i18n_translate.wmt_scores(traces)
    for d in refs:
        h = [(x or "").strip() for x in hyps[d]]
        bleu = sacrebleu.corpus_bleu(h, [refs[d]], tokenize="zh" if d == "en-zh" else "13a")
        assert out[d]["bleu"] == round(bleu.score, 2)
        assert out[d]["chrf"] == round(sacrebleu.corpus_chrf(h, [refs[d]]).score, 2)
    assert "tok:zh" in out["en-zh"]["bleu_signature"] and "tok:13a" in out["de-en"]["bleu_signature"]
    # quotes are part of the system output (the reference is quoted)
    assert i18n_translate._hypothesis({"response_text": '  "Yes," he said. '}) == '"Yes," he said.'


def test_i18n_translate_requires_sacrebleu(monkeypatch):
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        i18n_translate._require_sacrebleu()
