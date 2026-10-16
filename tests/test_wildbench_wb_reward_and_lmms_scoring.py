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

"""Judged, vision and long-context suites: WildBench WB-Reward, official sampling defaults,
and lmms-eval scoring of the raw reply for chartqa / textvqa / docvqa / infographicvqa.

No model server or judge is contacted: requests, judges and downloads are replaced.
"""
import asyncio
import hashlib
import random
import re
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (aa_lcr, base, beam_128k, chartqa, charxiv,
                                        coco_caption, docvqa, humanitys_last_exam,
                                        infographicvqa, loft_scifact, medxpertqa,
                                        omnidocbench, ruler, textvqa, wildbench)
from gbench.runners.eval_suites.vqa_common import (anls_score, lmms_anls, lmms_chartqa_relaxed,
                                                   lmms_eval_sampling, lmms_textvqa_accuracy)


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


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc(name):
    return (DOCS / f"{name}.md").read_text(encoding="utf-8")


def _serve(replies, seen=None, completion_tokens=5):
    """A fake _send_single_request answering by the last user text (None = failed request)."""
    async def _send(**kw):
        if seen is not None:
            seen.append(kw)
        content = kw["messages"][-1]["content"]
        text = content if isinstance(content, str) else content[-1]["text"]
        reply = replies[text]
        if reply is None:
            return base.Reply(text=None, tool_calls=None, error="server error")
        return base.Reply(text=reply, tool_calls=None, finish_reason="stop",
                          completion_tokens=completion_tokens)
    return _send


# --------------------------------------------------------------------------- #
# WildBench: pairwise prompt, A/B order, parsing
# --------------------------------------------------------------------------- #
def test_pairwise_and_score_templates_are_verbatim():
    # sha256 of allenai/WildBench evaluation/eval_template.{pairwise,score}.v2.md @ d6b8dcaf377d
    assert _sha(wildbench.PAIRWISE_TEMPLATE) == \
        "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"
    assert _sha(wildbench.EVAL_TEMPLATE) == \
        "c5984ae77b009f8dad0e8e37093348345f4e8d3099ce812bd7799ef7b73f005f"


def test_pairwise_order_is_seed_42_one_draw_per_task_in_split_order():
    ids = [f"s{i}" for i in range(50)]
    rng = random.Random(42)
    assert wildbench.pairwise_assignment(ids) == {sid: rng.random() < 0.5 for sid in ids}
    # draws 0.639, 0.025, 0.275, 0.223, 0.737: the tested model is A for the 2nd-4th tasks
    first = wildbench.pairwise_assignment(ids[:5])
    assert [first[s] for s in ids[:5]] == [False, True, True, True, False]


def test_limited_loader_keeps_the_full_split_order(monkeypatch):
    import sys
    import types
    rows = [{"session_id": f"s{i}", "conversation_input": [{"role": "user", "content": f"q{i}"}],
             "checklist": ["c"], "primary_tag": ["Math", "Editing"][i % 2], "secondary_tags": []}
            for i in range(8)]
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *a, **k: rows))
    monkeypatch.setattr(wildbench, "N_TASKS", 8)
    full = wildbench.pairwise_assignment([r["session_id"] for r in rows])
    samples = wildbench._load_wildbench_samples(limit=3)
    assert len(samples) == 3
    for _, _, meta in samples:
        assert meta["wb_pairwise_target_is_a"] == full[meta["session_id"]]


def test_shorten_is_upstreams():
    assert wildbench.shorten("a b c d", 2) == "a b... (truncated)"
    assert wildbench.shorten("a b c d", -1) == "a b c d"
    assert wildbench.shorten("a b", 2) == "a b"


def test_pairwise_prompt_cuts_to_1000_words_and_marks_an_empty_reply():
    msgs = [{"role": "user", "content": "w " * 1200}, {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now answer"}]
    long_reply = " ".join(f"t{i}" for i in range(1500))
    p = wildbench.build_wb_pairwise_prompt(msgs, ["Is it right?"], long_reply, "")
    a, b = wildbench.pairwise_outputs_from_prompt(p)
    assert a.endswith("... (truncated)") and len(a[:-len("... (truncated)")].split(" ")) == 1000
    assert b == wildbench.EMPTY_RESPONSE_PLACEHOLDER
    assert "<|begin_of_history|>\n\nUSER: " in p and "... (truncated)\n\n<|end_of_history|>" in p
    assert "<|begin_of_query|>\n\nnow answer\n\n<|end_of_query|>" in p
    assert "- Is it right?\n" in p and "{$" not in p
    outcome = wildbench.pairwise_outcome("A+", True, a, b)
    assert outcome["target_truncated"] and outcome["reference_empty"]
    assert outcome["winner"] == "target" and outcome["extent"] == 1


@pytest.mark.parametrize("reply,choice", [
    ('{"analysis of A": "x", "choice": "A++"}', "A++"),
    ('{"choice": "B+"}', "B+"),
    ('{"choice": "A=B"}', "A=B"),
    ('```json\n{"choice": "A+"}\n```', None),       # batch_results_format: json.loads or drop
    ('{"choice": "A+++"}', None),
    ('{"choice": "[A++ or A+]"}', None),
    ('["choice"]', None),
    ('{"reason": "no choice"}', None),
    ("", None),
])
def test_pairwise_choice_is_read_as_the_batch_pipeline_reads_it(reply, choice):
    assert wildbench.parse_wb_choice(reply) == choice


@pytest.mark.parametrize("reply,score", [
    ('{"strengths": "s", "weaknesses": "w", "score": "7"}', 7.0),
    ('{"score": 9}', 9.0),
    ('```\n{"score": "6"}\n```', None),
    ('{"score": "[1~10]"}', None),
    ('"score": "8"', None),
])
def test_score_is_read_as_the_batch_pipeline_reads_it(reply, score):
    assert wildbench.parse_wb_score(reply) == score


# --------------------------------------------------------------------------- #
# WildBench: reward aggregation (_create_tables.py / _merge_results.py)
# --------------------------------------------------------------------------- #
def _outcome(choice, target_is_a=True, t="x" * 10, r="y" * 10, groups=("Creative Tasks",)):
    return {**wildbench.pairwise_outcome(choice, target_is_a, t, r), "groups": list(groups)}


def test_reward_table_follows_create_tables():
    outs = [_outcome("A++"),                                   # win_much
            _outcome("A+", t="x" * 50, r="y" * 10),            # win, or tie for K < 40
            _outcome("A=B"),                                   # tie
            _outcome("B+", target_is_a=True),                  # lose
            _outcome("A++", target_is_a=False),                # lose_much (reference is A)
            _outcome("A++", t="z... (truncated)"),             # truncated: not counted
            _outcome("B++", r=wildbench.EMPTY_RESPONSE_PLACEHOLDER)]   # empty: not counted
    row = wildbench.wb_reward_table(outs, -1)
    assert (row["win_much"], row["win"], row["tie"], row["lose"], row["lose_much"]) == (1, 1, 1, 1, 1)
    assert row["total"] == 7                                  # skipped rows stay in the total
    assert row["reward"] == pytest.approx((1 + 0.5 - 0.5 - 1) / 7)
    assert row["task_categorized_rewards"] == {"Creative Tasks": pytest.approx(0.0)}
    assert row["task_macro_reward"] == pytest.approx(0.0)
    assert row["avg_len"] == pytest.approx((10 + 50 + 10 + 10 + 10) / 5)
    penalised = wildbench.wb_reward_table(outs, 30)          # 50 > 10 + 30: the slight win ties
    assert (penalised["win"], penalised["tie"]) == (0, 2)
    assert wildbench.wb_reward_table(outs, 40)["win"] == 1   # 50 > 10 + 40 is false


def test_task_macro_reward_divides_by_all_five_weights():
    outs = [_outcome("A++", groups=["Coding & Debugging"]), _outcome("B++", groups=["Creative Tasks"])]
    row = wildbench.wb_reward_table(outs, -1)
    assert row["task_macro_reward"] == pytest.approx((1 * 1.25 + (-1) * 0.5) / 4.75)


def test_reward_columns_use_upstream_names_and_mixtures():
    tables = {"gpt4t": wildbench.wb_reward_table([_outcome("B++")], -1),
              "haiku": wildbench.wb_reward_table([_outcome("A=B")], -1),
              "llama": wildbench.wb_reward_table([_outcome("A++")], -1)}
    cols = wildbench.wb_reward_columns(tables, -1, 20.0)
    assert cols["gpt4t_reward.K=-1"] == -100.0 and cols["llama_reward.K=-1"] == 100.0
    assert cols["mixture_of_rewards.K=-1"] == 0.0
    assert cols["task_macro_reward.K=-1"] == pytest.approx((-0.5 + 0 + 0.5) / 4.75 * 100 / 3)
    assert cols["mixture_of_rewards.Creative Tasks.K=-1"] == 0.0
    assert cols["haiku_reward.Coding & Debugging.K=-1"] is None      # no such task here
    assert cols["RewardScore_Avg"] == pytest.approx((20.0 + cols["task_macro_reward.K=-1"]) / 2)
    missing = wildbench.wb_reward_columns({"gpt4t": tables["gpt4t"]}, -1, 20.0)
    assert missing["mixture_of_rewards.K=-1"] is None and missing["RewardScore_Avg"] is None


# --------------------------------------------------------------------------- #
# WildBench: score mode judges empty replies; references are pinned and verified
# --------------------------------------------------------------------------- #
def test_score_mode_judges_an_empty_reply_but_not_a_failed_request():
    prompts = []

    async def _judge(prompt, config=None):
        prompts.append(prompt)
        return '{"strengths": "N/A", "weaknesses": "empty", "score": "2"}', "gemini-x"

    empty = {"messages": [{"role": "user", "content": "q"}], "gold_answer": ["c"],
             "response_text": "", "status": "PENDING_JUDGE"}
    failed = {"messages": [{"role": "user", "content": "q"}], "gold_answer": ["c"],
              "response_text": None, "status": "FAILED"}
    with mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        asyncio.run(wildbench._async_judge_wildbench([empty, failed]))
    assert len(prompts) == 1 and "<|begin_of_response|>\n\n\n\n<|end_of_response|>" in prompts[0]
    assert empty["wb_score_raw"] == 2.0 and empty["wb_output_chars"] == 0
    assert "wb_score_raw" not in failed and failed["judge_grade"] == "not_judged_failed_request"


def test_reference_loader_checks_the_pinned_files(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "ref.parquet"
    pq.write_table(pa.table({"session_id": ["a", "b"], "output": [["ra"], ["rb"]],
                             "generator": ["m", "m"]}), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    import huggingface_hub
    seen = []

    def _download(repo, filename, repo_type=None, revision=None):
        seen.append((repo, filename, repo_type, revision))
        return str(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr(wildbench, "N_TASKS", 2)
    monkeypatch.setattr(wildbench, "REFERENCE_MODELS", {"gpt4t": ("cfg", digest)})
    assert wildbench._load_reference_outputs() == {"gpt4t": {"a": "ra", "b": "rb"}}
    assert seen == [("allenai/WildBench-V2-Model-Outputs", "cfg/train-00000-of-00001.parquet",
                     "dataset", "d6755bc68220df853c0825a733430f73f5af2501")]
    monkeypatch.setattr(wildbench, "REFERENCE_MODELS", {"gpt4t": ("cfg", "0" * 64)})
    with pytest.raises(RuntimeError, match="sha256"):
        wildbench._load_reference_outputs()

    def _offline(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _offline)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        wildbench._load_reference_outputs()


def test_the_three_references_are_the_leaderboard_ones():
    assert {p: cfg for p, (cfg, _) in wildbench.REFERENCE_MODELS.items()} == {
        "gpt4t": "gpt-4-turbo-2024-04-09", "haiku": "claude-3-haiku-20240307",
        "llama": "Llama-2-70b-chat-hf"}
    assert wildbench.LENGTH_MARGINS == (-1, 100, 300, 500, 1000, 1500, 2000, 3000)
    assert wildbench.DEFAULT_LENGTH_MARGIN == -1 and wildbench.PAIRWISE_MAX_WORDS == 1000


# --------------------------------------------------------------------------- #
# WildBench: end to end
# --------------------------------------------------------------------------- #
def _wb_run(monkeypatch, replies, pairwise, thinking=False):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = [([{"role": "user", "content": q}], ["c"],
                {"category": "Math", "session_id": q, "wb_pairwise_target_is_a": q == "q1",
                 "wb_task_groups": ["Math & Data Analysis"]}) for q in replies]
    refs = {p: {q: f"reference {p}" for q in replies} for p in wildbench.REFERENCE_MODELS}
    sent, judged = [], []

    async def _judge(prompt, config=None):
        judged.append(prompt)
        if "<|begin_of_response_A|>" in prompt:
            return pairwise(prompt)
        return '{"score": "8"}', "gemini-s"

    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies, sent)), \
            mock.patch.object(wildbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(wildbench, "_load_wildbench_samples", lambda **k: samples), \
            mock.patch.object(wildbench, "_load_reference_outputs", lambda: refs), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_outputs", lambda: {}), \
            mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        res = wildbench.run_wildbench("m", "http://x/v1", 1, enable_thinking=thinking)
    return res, sent, judged


def test_run_reports_wb_reward_next_to_wb_score(monkeypatch):
    def _pairwise(prompt):
        a, b = wildbench.pairwise_outputs_from_prompt(prompt)
        # the tested reply wins much when it is shown, the empty one is a tie
        if "good answer" in (a, b):
            return ('{"choice": "%s"}' % ("A++" if a == "good answer" else "B++"), "gemini-p")
        return '{"choice": "A=B"}', "gemini-p"

    res, sent, judged = _wb_run(monkeypatch, {"q1": "good answer", "q2": None}, _pairwise)
    assert res["accuracy"] == res["wb_score_task_macro"]
    # q2 got no reply from the endpoint: a harness error, not compared and in no total
    assert res["wb_reward_by_reference"] == {"gpt4t": 100.0, "haiku": 100.0, "llama": 100.0}
    assert res["wb_reward_mix"] == 100.0
    # q1 wins much in Math & Data Analysis, the one comparison made
    assert res["wb_reward_task_macro"] == pytest.approx(100 * 1 / 4.75, abs=1e-3)
    assert res["wb_reward_length_margin"] == -1
    assert set(res["wb_reward"]) == {str(k) for k in wildbench.LENGTH_MARGINS}
    ref = res["wb_reward"]["-1"]["references"]["gpt4t"]
    assert (ref["win_much"], ref["total"]) == (1, 1)
    assert res["wb_reward_score_avg"] == pytest.approx(
        (res["wb_score_task_macro"] + res["wb_reward_task_macro"]) / 2, abs=1e-3)
    assert sum(1 for p in judged if "<|begin_of_response_A|>" in p) == 3
    empty_prompts = [p for p in judged if wildbench.EMPTY_RESPONSE_PLACEHOLDER in p]
    assert len(empty_prompts) == 0                            # the failed request is not judged
    q2 = res["sample_traces"][1]
    assert q2["scoring_excluded"] is True and q2["harness_error"]
    assert {r["judge_grade"] for r in q2["wb_pairwise"].values()} == {"not_judged_failed_request"}
    assert res["wb_reward_judge_models"] == {"gemini-p": 3}
    assert res["judge_models"] == {"gemini-s": 1}             # score mode: the answered task
    assert "gpt-4-turbo-2024-04-09" in res["leaderboard_comparable_reason"]
    # WildBench's sampling is the no-think default
    assert all(k["temperature"] == 0.0 and k["extra_payload"]["top_p"] == 1.0 for k in sent)
    assert res["official_sampling"]["temperature_matches_official"] is True
    rec = res["sample_traces"][0]["wb_pairwise"]["gpt4t"]
    assert rec["reference_model"] == "gpt-4-turbo-2024-04-09" and rec["target_side"] in "AB"


def test_pairwise_outages_and_parse_failures_are_counted(monkeypatch):
    calls = {"n": 0}

    def _pairwise(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, "JUDGE_OUTAGE"
        return "not json", "gemini-p"

    res, _, _ = _wb_run(monkeypatch, {"q1": "a"}, _pairwise, thinking=True)
    assert sum(res["wb_reward_judge_outages"].values()) == 1
    assert sum(res["wb_reward_judge_failures"].values()) == 2
    assert res["wb_reward_mix"] is None                       # no reference has a reply read
    assert res["status"] != "success"
    assert "lost to a judge outage" in res["leaderboard_comparable_reason"]
    # a --thinking run samples at 1.0, not WildBench's 0, and says so
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "temperature 1.0 (WildBench decodes at 0)" in res["leaderboard_comparable_reason"]


def test_wildbench_sends_its_cap_without_thinking(monkeypatch):
    base._RUN_KNOBS["max_output_tokens"] = 16384
    res, sent, _ = _wb_run(monkeypatch, {"q1": "a"}, lambda p: ('{"choice": "A=B"}', "g"))
    assert [k["max_output_tokens"] for k in sent] == [4096]
    assert res["official_sampling"]["max_tokens"] == 4096
    assert res["official_sampling"]["max_tokens_applied"] is True
    assert "replies_over_upstream_cap" not in res
    assert wildbench.UPSTREAM_MAX_TOKENS == 4096


def test_wildbench_doc_describes_wb_reward():
    doc = _doc("wildbench")
    assert doc.isascii()
    for s in ("WB-Reward", "`wb_reward_task_macro`", "`wb_reward_mix`", "`RewardScore_Avg`",
              "d6755bc68220df853c0825a733430f73f5af2501", "eval_template.pairwise.v2.md",
              "random.seed(42)", "K = -1", "`official_sampling`", "WB-Elo",
              "`replies_at_official_max_tokens`", "--max-output-tokens"):
        assert s in doc, s
    assert "An empty reply scores 1 without a judge call" not in doc


# --------------------------------------------------------------------------- #
# official sampling defaults
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module,temperature,params", [
    (wildbench, 0.0, {"top_p": 1.0}),
    (beam_128k, 0.0, None),
    (coco_caption, 0.0, {"top_p": 1.0}),
    (chartqa, 0.0, None),
    (textvqa, 0.0, None),
    (docvqa, 0.0, None),
    (infographicvqa, 0.0, None),
    (charxiv, 0.0, {"top_p": 1.0, "seed": 42}),
    (medxpertqa, 0.0, {"seed": 0}),
    (ruler, 0.0, {"top_p": 1.0, "seed": 0}),
    (loft_scifact, 0.0, {"top_p": 1.0}),
    (omnidocbench, 0.0, None),
])
def test_official_sampling_is_declared_and_passed(module, temperature, params):
    official = module.OFFICIAL_SAMPLING
    assert isinstance(official, base.OfficialSampling)
    assert official.temperature == temperature and official.send_temperature is True
    assert (official.params or None) == params
    assert all(k in base.ALLOWED_API_KEYS for k in (official.params or {}))
    assert official.reference.isascii() and "@" in official.reference
    src = (SUITES / f"{module.__name__.rsplit('.', 1)[-1]}.py").read_text(encoding="utf-8")
    assert "official_sampling=OFFICIAL_SAMPLING" in src
    # the suite no longer passes the CLI value as its own temperature
    assert 'temperature=kwargs.get("temperature")' not in src


def test_aa_lcr_follows_artificial_analysis_per_mode():
    assert aa_lcr.official_sampling(False).temperature == 0.0
    # reasoning models: the lab's generation_config.json temperature, else AA's 0.6 (never a
    # temperature assumed for every model)
    assert aa_lcr.official_sampling(True).temperature == aa_lcr.AA_REASONING_TEMPERATURE == 0.6
    assert aa_lcr.official_sampling(True, {"temperature": 0.7}, "x").temperature == 0.7
    src = (SUITES / "aa_lcr.py").read_text(encoding="utf-8")
    assert "reasoning_sampling(model_name) if enable_thinking" in src


def test_hle_sends_temperature_zero_without_thinking_and_one_with_it(monkeypatch):
    samples = [([{"role": "user", "content": "q1"}], "g", {"category": "Math"})]

    async def _judge(prompt, config=None):
        return ('{"extracted_final_answer": "g", "reasoning": "r", "correct": "yes", '
                '"confidence": 90, "strict": true}'), "gemini-x"

    # HLE sends reasoning models no temperature; a --thinking run samples at 1.0 regardless and
    # the result names the mismatch.
    for thinking, temperature in ((False, 0.0), (True, 1.0)):
        sent = []
        with mock.patch.object(base, "_send_single_request",
                               side_effect=_serve({"q1": "Answer: g"}, sent)), \
                mock.patch.object(humanitys_last_exam, "gemini_required_skip", lambda *a: None), \
                mock.patch.object(humanitys_last_exam, "_load_humanitys_last_exam_samples",
                                  lambda limit=None: samples), \
                mock.patch.object(humanitys_last_exam, "judge_generate_cascade", _judge):
            res = humanitys_last_exam.run_humanitys_last_exam("m", "http://x/v1", 1,
                                                              enable_thinking=thinking)
        assert sent[0]["temperature"] == temperature
        assert res["official_sampling"]["temperature_matches_official"] is (not thinking)
        assert (("temperature 1.0 (HLE sends reasoning models no temperature)"
                 in res["leaderboard_comparable_reason"]) is thinking)
        assert ("temperature 0" not in res["leaderboard_comparable_reason"])
    monkeypatch.setenv("GBENCH_HUMANITYS_LAST_EXAM_TEMPERATURE", "0.7")
    with mock.patch.object(base, "_send_single_request", side_effect=_serve({"q1": "x"})), \
            mock.patch.object(humanitys_last_exam, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(humanitys_last_exam, "_load_humanitys_last_exam_samples",
                              lambda limit=None: samples), \
            mock.patch.object(humanitys_last_exam, "judge_generate_cascade", _judge):
        res = humanitys_last_exam.run_humanitys_last_exam("m", "http://x/v1", 1)
    assert "temperature 0.7 (HLE evaluates at 0)" in res["leaderboard_comparable_reason"]


@pytest.mark.parametrize("name", ["wildbench", "humanitys_last_exam", "aa_lcr", "beam_128k",
                                  "coco_caption", "chartqa", "textvqa", "docvqa",
                                  "infographicvqa", "charxiv", "medxpertqa", "ruler",
                                  "loft_scifact", "omnidocbench"])
def test_docs_state_the_official_sampling(name):
    doc = _doc(name)
    assert doc.isascii(), name
    assert "`official_sampling`" in doc, name
    assert "run default (0.0 greedy / 1.0 with `--thinking`)" not in doc, name
    runs = [l for l in doc.splitlines() if l.startswith("gbench --evals-only")]
    assert runs, name


# --------------------------------------------------------------------------- #
# chartqa / textvqa / docvqa / infographicvqa: lmms-eval scoring of the raw reply
# --------------------------------------------------------------------------- #
def test_lmms_anls_divides_by_the_stored_gold_length():
    # edit distance on "pari" vs "paris" (1), length = max(len("  Paris"), len("Pari")) = 7
    assert lmms_anls("Pari", ["  Paris"]) == pytest.approx(1 - 1 / 7)
    assert anls_score("Pari", ["  Paris"]) == pytest.approx(1 - 1 / 5)    # helper differs
    # the reply is stripped first (lmms-eval's task layer), so its padding does not count
    assert lmms_anls("Pariss  \n", ["Paris"]) == pytest.approx(1 - 1 / 6)
    assert lmms_anls("  PARIS ", ["paris"]) == 1.0
    assert lmms_anls("London", ["Paris"]) == 0
    assert lmms_anls(None, ["Paris"]) == 0


def test_lmms_textvqa_is_leave_one_out_over_every_answer():
    answers = ["stop"] * 3 + [""] * 7
    # each of the 3 "stop" annotators sees 2 others (2/3), each empty one sees 3 (1): 0.9
    assert lmms_textvqa_accuracy("Stop.", answers) == pytest.approx((3 * 2 / 3 + 7 * 1) / 10)
    assert lmms_textvqa_accuracy("**stop**", ["stop"] * 10) == 0.0     # no extraction
    assert lmms_textvqa_accuracy("go", ["stop"] * 10) == 0.0


def test_lmms_chartqa_only_strips_surrounding_whitespace():
    assert lmms_chartqa_relaxed(" 14 ", "14") is True
    assert lmms_chartqa_relaxed("Yes\n", "Yes") is True          # api/task.py strips results
    assert lmms_chartqa_relaxed("Yes.", "Yes") is False
    assert lmms_chartqa_relaxed("yes", ["Yes"]) is True
    assert lmms_chartqa_relaxed("Answer: 14", "14") is False
    assert lmms_chartqa_relaxed("**14**", "14") is False
    assert lmms_chartqa_relaxed(None, "14") is False


def test_lmms_eval_sampling_follows_the_chat_client_per_mode():
    official = lmms_eval_sampling("lmms-eval tasks/x.yaml @42296f8f", 16)
    assert (official.temperature, official.max_tokens, official.send_temperature) == (0.0, 16, True)
    # the enable_thinking path keeps the task's sampling: no reasoning-model settings
    assert official.reasoning is None and official.for_mode(True) is official


@pytest.mark.parametrize("module,run,loader,gold,cap", [
    (chartqa, "run_chartqa", "_load_chartqa_samples", "14", 16),
    (textvqa, "run_textvqa", "_load_textvqa_samples", ["14"] * 10, 1024),
    (docvqa, "run_docvqa", "_load_docvqa_samples", ["14"], 32),
    (infographicvqa, "run_infographicvqa", "_load_infographicvqa_samples", ["14"], 32),
])
def test_vqa_suites_send_the_task_cap_per_the_token_limit_policy(module, run, loader, gold, cap,
                                                                monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = 65536
    samples = [([{"role": "user", "content": [{"type": "text", "text": "q1"}]}], gold,
                {"category": "c"})]

    def _at_cap(sent):
        async def _send(**kw):
            sent.append(kw)
            return base.Reply(text="14", tool_calls=None, finish_reason="length",
                              completion_tokens=kw["max_output_tokens"])
        return _send

    # no-think: the task's cap is sent; a reply stopping there is lmms-eval's own outcome
    sent = []
    with mock.patch.object(base, "_send_single_request", side_effect=_at_cap(sent)), \
            mock.patch.object(module, loader, lambda **k: samples):
        res = getattr(module, run)("m", "http://x/v1", 1)
    assert [(k["max_output_tokens"], k["temperature"]) for k in sent] == [(cap, 0.0)]
    assert res["accuracy"] == 100.0 and res["leaderboard_comparable"] is True
    assert res["replies_at_official_max_tokens"] == 1 and "replies_over_upstream_cap" not in res
    assert res["max_output_tokens_note"].startswith("--max-output-tokens 65536 not applied")
    # --thinking: sampled at 1.0, not the task's 0; the cap is not applied to reasoning (the
    # run's budget)
    sent = []
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve({"q1": "14"}, sent)), \
            mock.patch.object(module, loader, lambda **k: samples):
        res = getattr(module, run)("m", "http://x/v1", 1, enable_thinking=True)
    assert [(k["max_output_tokens"], k["temperature"]) for k in sent] == [(65536, 1.0)]
    assert "reasoning_settings" not in res["official_sampling"]
    assert res["official_sampling"]["max_tokens_applied"] is False
    assert res["official_sampling"]["max_tokens_sent"] == 65536
    assert res["official_sampling"]["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith(
        "temperature 1.0 (lmms-eval decodes greedily)")
    monkeypatch.setenv(base.temperature_env_var(module.__name__.rsplit(".", 1)[-1]), "0.5")
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve({"q1": "14"}, sent)), \
            mock.patch.object(module, loader, lambda **k: samples):
        res = getattr(module, run)("m", "http://x/v1", 1)
    assert res["leaderboard_comparable"] is False
    assert "temperature 0.5 (lmms-eval decodes greedily)" in res["leaderboard_comparable_reason"]


def test_vqa_suites_no_longer_extract_before_scoring():
    for name in ("chartqa", "textvqa", "docvqa", "infographicvqa"):
        src = (SUITES / f"{name}.py").read_text(encoding="utf-8")
        assert "extract_short_answer" not in src, name
        assert re.search(r"lmms_(anls|textvqa_accuracy|chartqa_relaxed)", src), name
        doc = _doc(name)
        assert "no answer extraction" in doc and "`leaderboard_comparable` is always" not in doc
